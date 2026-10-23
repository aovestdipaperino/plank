// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Flint behind [`NativeEngine`], over `crates/flint-engine`.
//!
//! The chat format is the GGUF's own (`flint_engine::template`): no turn
//! closes, so a section's tokens never depend on what follows. The model
//! writes `<think>`/`</think>` and its tool calls as single control tokens
//! whose spellings are the tags, so the stream needs no translation; plank's
//! system text carries the tools block between the `FLINT_TOOLS_*` fence
//! lines, which this family turns into the template's tools turn. Whatever
//! follows the fence joins the prose in the first system turn, so the tools
//! turn comes last, as the trained template renders it.
//!
//! [`NativeEngine`]: crate::nativeengine::NativeEngine

use std::path::Path;
use std::sync::Arc;

use flint_engine::config::FlintConfig;
use flint_engine::file::GgufFile;
use flint_engine::gguf::Value;
use flint_engine::model::Model;
use flint_engine::template::{self, Kind, Piece};
use flint_engine::tokenizer::FlintTokenizer;
use gemma_engine::Device;

use crate::engine::{EngineError, Utf8Stream};
use crate::sysprompt::{FLINT_TOOLS_CLOSE, FLINT_TOOLS_OPEN};

/// The largest context Flint opens by default, before RAM sizing: its
/// attention memory grows with `t × keys`, so the file's 262,144 is only
/// reached through an explicit `--ctx`.
pub const FLINT_DEFAULT_CTX_CAP: usize = 65_536;

/// Bytes one prefill chunk's f32 attention scores may take.
const SCORE_BUDGET: usize = 2 << 30;

/// Flint's part of [`crate::nativeengine::NativeEngine`]: its tokenizer,
/// its template and the GGUF's `top_k`.
#[derive(Debug)]
pub struct FlintFamily {
    model: Arc<Model>,
    tok: FlintTokenizer,
    top_k: Option<usize>,
    heads: usize,
}

/// Flint behind [`crate::engine::Engine`]: one model, one live session, and
/// the token transcript that describes what the session's KV was built from.
pub type FlintEngine = crate::nativeengine::NativeEngine<FlintFamily>;

fn flint_error(e: flint_engine::Error) -> EngineError {
    EngineError::new(e.0)
}

/// Tokens per prefill forward with `keys` positions already in the cache.
///
/// A chunk of `t` tokens attends over `keys + t` positions, so its f32
/// score copy is `heads × t × (keys + t) × 4` bytes. The chunk is the
/// largest `t` keeping that at or below 2 GiB, the positive root of
/// `t² + keys·t − 2 GiB / (4·heads)`, clamped to `32..=512`.
pub(crate) fn chunk_for(heads: usize, keys: usize) -> usize {
    let budget = (SCORE_BUDGET / (heads.max(1) * 4)) as u128;
    let k = keys as u128;
    // t = (√(k² + 4·budget) − k) / 2, rounded down; then stepped down while
    // the product still overshoots (the root of a rounded square can).
    let mut t = ((k * k + 4 * budget).isqrt() - k) / 2;
    while t > 0 && t * (k + t) > budget {
        t -= 1;
    }
    usize::try_from(t).unwrap_or(usize::MAX).clamp(32, 512)
}

/// Whether a `file_len`-byte GGUF is larger than the Metal buffer `limit`
/// (`None`: no Metal device, so no limit). The no-copy buffer covers the
/// mapping rounded up to whole `page`s, so that rounded length is compared.
fn exceeds_gpu_buffer(file_len: u64, page: u64, limit: Option<u64>) -> bool {
    let bound = file_len.div_ceil(page.max(1)).saturating_mul(page.max(1));
    limit.is_some_and(|l| bound > l)
}

/// The VM page size, the granularity a mapping is bound to Metal at.
fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(n).ok().filter(|&n| n > 0).unwrap_or(16_384)
}

/// Gigabytes (2^30 bytes) with one decimal, for the refusal message.
#[allow(clippy::cast_precision_loss)]
fn gb(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / f64::from(1u32 << 30))
}

/// The tools fence in a system text: the prose before
/// [`FLINT_TOOLS_OPEN`], the non-empty lines up to the first
/// [`FLINT_TOOLS_CLOSE`] after it, and everything after that close. Both
/// cuts take the *first* marker, because the text after the close (MCP
/// server instructions, a user prompt) is not plank's and may spell either.
/// `None` without a whole fence.
fn split_fence(text: &str) -> Option<(&str, Vec<String>, &str)> {
    let open = text.find(FLINT_TOOLS_OPEN)?;
    let body = open + FLINT_TOOLS_OPEN.len();
    let close = body + text[body..].find(FLINT_TOOLS_CLOSE)?;
    let lines = text[body..close]
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    Some((
        &text[..open],
        lines,
        &text[close + FLINT_TOOLS_CLOSE.len()..],
    ))
}

/// One section as template pieces.
///
/// A system text holding the tools fence becomes one system turn of the
/// prose followed by the text after the fence, then the template's tools
/// turn over the fenced lines (each line plain). That is the trained
/// layout: the template writes every system message first and the tools
/// turn right before the first other message. The prose is trusted up to
/// `trusted_len` (an offset into the whole text, which the prose starts);
/// the text after the fence (MCP server instructions, the user's system
/// text) is never trusted, so trust stays one contiguous prefix: the
/// prose, then the rest escaped as plain, then the tools pieces. Anything
/// else is the template's own section; only a system section has a
/// trusted prefix.
fn pieces(kind: Kind, text: &str, trusted_len: usize) -> Vec<Piece> {
    if kind != Kind::System {
        return template::render(kind, text, 0);
    }
    let Some((prose, lines, rest)) = split_fence(text) else {
        return template::render(kind, text, trusted_len);
    };
    let mut out = template::render(
        Kind::System,
        &format!("{prose}{rest}"),
        trusted_len.min(prose.len()),
    );
    out.extend(template::render_tools(&lines));
    out
}

/// `general.name` when it names a Flint model, else `"Flint"`: the
/// tool-syntax lookup keys on that prefix.
fn name_of(h: &flint_engine::gguf::Header) -> String {
    match h.get("general.name") {
        Some(Value::Str(s)) if s.starts_with("Flint") => s.clone(),
        _ => "Flint".to_owned(),
    }
}

impl FlintFamily {
    /// Keeps the `k` largest logits (and any tied with the k-th), setting
    /// the rest to `-inf`. Greedy picks the maximum anyway, so it is left
    /// alone, as is a `k` of 0 or one covering every logit.
    pub(crate) fn filter_logits_k(logits: &mut [f32], k: usize, greedy: bool) {
        if greedy || k == 0 || k >= logits.len() {
            return;
        }
        let mut sorted = logits.to_vec();
        let (_, kth, _) = sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
        let threshold = *kth;
        for l in logits.iter_mut() {
            if l.total_cmp(&threshold).is_lt() {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

impl crate::nativeengine::NativeFamily for FlintFamily {
    type Model = Model;
    type Translator = ();
    const LABEL: &'static str = "flint";

    /// Refuses a file larger than the Metal device's `maxBufferLength`
    /// before mapping it: such a file cannot be one no-copy buffer, and the
    /// copy fallback would double the resident weights.
    fn open(path: &Path, device: &Device, cap: usize) -> Result<(Arc<Model>, Self), EngineError> {
        if device.is_metal() {
            let len = std::fs::metadata(path)
                .map_err(|e| EngineError::new(format!("{}: {e}", path.display())))?
                .len();
            let limit = flint_engine::max_buffer_len_of(device);
            if exceeds_gpu_buffer(len, page_size(), limit) {
                return Err(EngineError::new(format!(
                    "Flint's {} GB file is larger than this Mac's GPU buffer limit ({} GB)",
                    gb(len),
                    gb(limit.unwrap_or(0))
                )));
            }
        }
        let model = Model::open_with_ctx(path, device, cap).map_err(flint_error)?;
        let tok = FlintTokenizer::from_header(model.header()).map_err(flint_error)?;
        let config = model.config();
        let top_k = config.sampling.top_k.and_then(|k| usize::try_from(k).ok());
        let heads = config.heads;
        Ok((
            model.clone(),
            Self {
                model,
                tok,
                top_k,
                heads,
            },
        ))
    }

    fn probe(path: &Path) -> Result<(usize, usize), EngineError> {
        let file = GgufFile::open(path).map_err(flint_error)?;
        let c = FlintConfig::from_header(&file.header).map_err(flint_error)?;
        Ok((c.context_length, c.kv_bytes_per_token()))
    }

    fn default_ctx_cap() -> Option<usize> {
        Some(FLINT_DEFAULT_CTX_CAP)
    }

    fn kv_bytes_per_token(model: &Model) -> usize {
        model.kv_bytes_per_token()
    }

    fn model_name(&self) -> String {
        name_of(self.model.header())
    }

    /// None: the GGUF's `add_bos_token` is false.
    fn bos(&self) -> Option<u32> {
        None
    }

    /// Flint's sections depend on neither the previous kind nor the
    /// reasoning level.
    fn render(
        &self,
        kind: Kind,
        text: &str,
        _prev: Option<Kind>,
        _think: bool,
        trusted_len: usize,
    ) -> Vec<u32> {
        self.tok.encode_pieces(&pieces(kind, text, trusted_len))
    }

    fn generation_prefix(&self, _last: Option<Kind>, _think: bool) -> Vec<u32> {
        self.tok.encode_pieces(&template::generation_prefix())
    }

    /// The GGUF's `eog_token_ids`: `<|end_of_text|>`, `<|user|>`,
    /// `<|observation|>` and `<|developer|>`.
    fn is_stop(&self, id: u32) -> bool {
        self.tok.is_eog(id)
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tok.encode_plain(text).len()
    }

    /// A control id streams as its spelling (`<think>`, `<tool_call>`, ...),
    /// ending any multi-byte run first; any other id goes through the UTF-8
    /// stream.
    fn translate(&self, _tr: &mut (), id: u32, utf8: &mut Utf8Stream) -> String {
        if self.tok.is_control(id) {
            let mut out = utf8.flush();
            out.push_str(&String::from_utf8_lossy(&self.tok.piece_bytes(id)));
            return out;
        }
        utf8.push(self.tok.piece_bytes(id))
    }

    fn filter_logits(&self, logits: &mut [f32], greedy: bool) {
        if let Some(k) = self.top_k.filter(|&k| k > 0) {
            Self::filter_logits_k(logits, k, greedy);
        }
    }

    fn prefill_chunk(&self, keys: usize) -> usize {
        chunk_for(self.heads, keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineEvent, GenerationOptions, Prompt, ThinkMode};
    use crate::nativeengine::{NativeFamily, to_u32};
    use crate::sysprompt::{FLINT_TOOLS_CLOSE, FLINT_TOOLS_OPEN};
    use flint_engine::config::{FlintConfig, Sampling};
    use flint_engine::gguf::Value;
    use flint_engine::template::{self, Kind, Piece};
    use flint_engine::testgguf::{FLINT_MARKERS, tiny_flint_builder, tiny_vocab};
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The id [`tiny_vocab`] gives a marker: `256 + its index`.
    fn marker(spelling: &str) -> u32 {
        let i = FLINT_MARKERS
            .iter()
            .position(|(s, _)| *s == spelling)
            .unwrap();
        256 + u32::try_from(i).unwrap()
    }

    /// `flint-engine`'s test config with a 256-token window.
    fn config() -> FlintConfig {
        FlintConfig {
            layers: 4,
            hidden: 128,
            ffn: 128,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            rope_base: 10000.0,
            rms_eps: 1e-6,
            context_length: 256,
            sliding_window: 4,
            swa: vec![false, true, true, false],
            rope: vec![false, true, true, false],
            dense_layers: 1,
            experts: 8,
            experts_used: 2,
            expert_ffn: 64,
            shared_ffn: 64,
            weights_norm: true,
            weights_scale: 1.0,
            sampling: Sampling {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
            },
        }
    }

    /// The seed whose greedy reply to [`TURN`] is a few tokens ending on an
    /// EOG id (found by search; the weights are seeded, so it is fixed).
    ///
    /// Seed 54 replies with three tokens, then `<|end_of_text|>`. Every
    /// greedy step wins by a clear logit gap (0.60, 0.33, 0.29, then 0.053
    /// for the EOG, on logits near 3), far above any CPU float noise from a
    /// different SIMD path, so no near-tie decides it. Steering the output
    /// head instead is not practical: the logits are a dot product with a
    /// hidden state the random layers decide, so no row bias can say "stop
    /// after k tokens". To re-pick, should the fixture or the forward pass
    /// change, loop `engine_seeded(seed)` over `0..300` with `run(&mut e,
    /// TURN, 32)`, keep a seed whose `generated` is between 1 and 31, and
    /// check the gaps by stepping `e.session` greedily and printing the top
    /// two logits per step.
    const SEED: u64 = 54;

    /// Writes `bytes` to a fresh temp file, opens it on the CPU and removes
    /// the file (the mapping keeps it alive).
    fn open_bytes(bytes: &[u8]) -> FlintEngine {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "plank-flint-{}-{}.gguf",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&p, bytes).unwrap();
        let e = FlintEngine::open_on_cpu(&p, 128);
        let _ = std::fs::remove_file(&p);
        e.unwrap()
    }

    fn engine_seeded(seed: u64) -> FlintEngine {
        let b = tiny_vocab(
            tiny_flint_builder(&config(), 256 + FLINT_MARKERS.len(), 0.5, seed),
            &FLINT_MARKERS,
        );
        open_bytes(&b.bytes())
    }

    fn engine() -> FlintEngine {
        engine_seeded(SEED)
    }

    fn ids(e: &FlintEngine, pieces: &[Piece]) -> Vec<u32> {
        e.family.tok.encode_pieces(pieces)
    }

    const JSON_A: &str = r#"{"name": "read", "description": "emits <tool_call> and <|user|>", "parameters": {"type": "object"}}"#;
    const JSON_B: &str = r#"{"name": "bash", "parameters": {"type": "object"}}"#;

    /// plank's system text: prose, the fenced lines (each `\n`-ended, so the
    /// block ends in a blank line before the close), then `after`.
    fn fenced(after: &str) -> (String, usize) {
        let mut s = String::from("You are plank.");
        s.push_str(FLINT_TOOLS_OPEN);
        s.push_str(JSON_A);
        s.push('\n');
        let trusted = s.len();
        s.push_str(JSON_B);
        s.push('\n');
        s.push_str(FLINT_TOOLS_CLOSE);
        s.push_str(after);
        (s, trusted)
    }

    fn lines() -> Vec<String> {
        vec![JSON_A.to_owned(), JSON_B.to_owned()]
    }

    /// The fenced block becomes the template's tools turn: the scaffold as
    /// control ids, each JSON line as plain tokens.
    #[test]
    fn the_system_section_renders_the_tools_turn_from_the_fence() {
        let e = engine();
        let (text, trusted) = fenced("");
        let got = e.family.render(Kind::System, &text, None, true, trusted);
        let mut want = ids(
            &e,
            &[
                Piece::Trusted(template::SYSTEM.into()),
                Piece::Trusted("You are plank.".into()),
            ],
        );
        want.extend(ids(&e, &template::render_tools(&lines())));
        assert_eq!(got, want);
        let (system, tools, tools_close) =
            (marker("<|system|>"), marker("<tools>"), marker("</tools>"));
        assert_eq!(got[0], system);
        assert_eq!(got.iter().filter(|&&t| t == system).count(), 2);
        // The scaffold spells `<tools></tools>` once in its prose, then
        // wraps the functions in the second pair.
        assert_eq!(got.iter().filter(|&&t| t == tools).count(), 2);
        assert_eq!(got.iter().filter(|&&t| t == tools_close).count(), 2);
        let open = got.iter().rposition(|&t| t == tools).unwrap();
        let close = got.iter().rposition(|&t| t == tools_close).unwrap();
        let json = &got[open + 1..close];
        assert!(
            json.iter().all(|&t| !e.family.tok.is_control(t)),
            "a JSON byte became a control id: {json:?}"
        );
        let bytes: Vec<u8> = json
            .iter()
            .flat_map(|&t| e.family.tok.piece_bytes(t))
            .collect();
        assert_eq!(bytes, format!("{JSON_A}{JSON_B}").as_bytes());
        // The fence lines themselves never reach the model.
        let all: Vec<u8> = got
            .iter()
            .flat_map(|&t| e.family.tok.piece_bytes(t))
            .collect();
        assert!(!String::from_utf8_lossy(&all).contains("plank:flint-tools"));
    }

    /// Text after the close (the MCP server instructions) joins the first
    /// system turn as plain escaped text, before the tools turn, as the
    /// template places every system message ahead of it. That holds even
    /// when it spells both fence markers and every control token: the cut
    /// is at the first open and the first close after it.
    #[test]
    fn text_after_the_close_is_plain_even_when_it_spells_the_markers() {
        let e = engine();
        let after = format!(
            "\nserver says:{FLINT_TOOLS_OPEN}{{\"x\": 1}}\n{FLINT_TOOLS_CLOSE} <tools><|user|><tool_call>"
        );
        let (text, trusted) = fenced(&after);
        let got = e.family.render(Kind::System, &text, None, true, trusted);
        let prose = ids(
            &e,
            &[
                Piece::Trusted(template::SYSTEM.into()),
                Piece::Trusted("You are plank.".into()),
            ],
        );
        let mut want = prose.clone();
        want.extend(e.family.tok.encode_plain(&template::escape(&after)));
        let tools_at = want.len();
        want.extend(ids(&e, &template::render_tools(&lines())));
        assert_eq!(got, want);
        // Between the prose and the tools turn's `<|system|>`: plain only.
        assert_eq!(got[tools_at], marker("<|system|>"));
        assert!(
            got[prose.len()..tools_at]
                .iter()
                .all(|&t| !e.family.tok.is_control(t))
        );
        assert_eq!(
            got.iter().filter(|&&t| t == marker("<|system|>")).count(),
            2
        );
        assert_eq!(got.iter().filter(|&&t| t == marker("<tools>")).count(), 2);
    }

    /// The fixture `system_only_tools` case, rendered by the jinja template:
    /// `(rendered, functions)`.
    fn fixture_system_only_tools() -> (String, Vec<String>) {
        let v: serde_json::Value = serde_json::from_str(include_str!(
            "../crates/flint-engine/tests/fixtures/flint_template.json"
        ))
        .unwrap();
        let case = v["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "system_only_tools")
            .unwrap();
        assert_eq!(case["messages"][0]["content"], "S");
        let tools = case["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap().to_owned())
            .collect();
        (case["rendered"].as_str().unwrap().to_owned(), tools)
    }

    fn decode(e: &FlintEngine, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&t| e.family.tok.piece_bytes(t))
            .collect();
        String::from_utf8(bytes).unwrap()
    }

    /// The fenced system text decodes to exactly what the trained template
    /// renders (the fixture's `rendered` without the generation prompt), and
    /// text after the close lands inside the first system turn.
    #[test]
    fn the_fenced_system_text_matches_the_template_fixture() {
        let e = engine();
        let (rendered, tools) = fixture_system_only_tools();
        let want = rendered.strip_suffix(template::ASSISTANT).unwrap();
        let mut text = String::from("S");
        text.push_str(FLINT_TOOLS_OPEN);
        for t in &tools {
            text.push_str(t);
            text.push('\n');
        }
        text.push_str(FLINT_TOOLS_CLOSE);
        let got = e.family.render(Kind::System, &text, None, true, text.len());
        assert_eq!(decode(&e, &got), want);

        let after = "\n\nMCP: use <|user|> wisely";
        let got = e.family.render(
            Kind::System,
            &format!("{text}{after}"),
            None,
            true,
            text.len(),
        );
        // jinja: the system message's escaped content, then the tools turn.
        let head = "<|system|>S";
        let jinja = format!(
            "{head}\n\nMCP: use &lt;|user|&gt; wisely{}",
            &want[head.len()..]
        );
        assert_eq!(decode(&e, &got), jinja);
    }

    /// Without the fence (`--tools off`, or an open with no close) the
    /// system text renders as the template's system section.
    #[test]
    fn a_system_text_without_a_whole_fence_renders_as_the_template_does() {
        let e = engine();
        for text in [
            "plain <tools> system".to_owned(),
            format!("prose{FLINT_TOOLS_OPEN}{JSON_A}\n"),
        ] {
            let got = e.family.render(Kind::System, &text, None, false, 5);
            assert_eq!(got, ids(&e, &template::render(Kind::System, &text, 5)));
        }
        assert_eq!(
            e.family.render(Kind::User, "hi <|user|>", None, false, 99),
            ids(&e, &template::render(Kind::User, "hi <|user|>", 0))
        );
    }

    /// The prose before the fence is trusted only up to `trusted_len`.
    #[test]
    fn the_prose_is_trusted_only_up_to_the_trusted_length() {
        let e = engine();
        let (text, _) = fenced("");
        let got = e.family.render(Kind::System, &text, None, false, 3);
        let mut want = ids(&e, &template::render(Kind::System, "You are plank.", 3));
        want.extend(ids(&e, &template::render_tools(&lines())));
        assert_eq!(got, want);
    }

    fn opts(n: i32) -> GenerationOptions {
        GenerationOptions {
            n_predict: n,
            ctx_size: 128,
            temperature: 0.0,
            think_mode: ThinkMode::Off,
            ..GenerationOptions::default()
        }
    }

    const TURN: &str = "[system]\nsys\n[user]\nhello\n";

    /// Generates greedily; returns the streamed text and the token count.
    fn run(e: &mut FlintEngine, flat: &str, n: i32) -> (String, i32) {
        let mut text = String::new();
        let stats = e
            .generate(
                Prompt::Flat(flat),
                &opts(n),
                &|| false,
                &|| true,
                &mut |ev| {
                    if let EngineEvent::Text(t) = ev {
                        text.push_str(&t);
                    }
                },
            )
            .unwrap();
        (text, stats.generated)
    }

    /// Greedy generation on the tiny model ends on an EOG id before the
    /// budget, records the reply as an assistant span (the EOG id neither
    /// recorded nor evaluated), and the next turn reuses every token.
    #[test]
    fn a_generation_stops_at_an_eog_id_and_records_the_reply() {
        let mut e = engine();
        let (reply, generated) = run(&mut e, TURN, 32);
        assert!(
            generated > 0 && generated < 32,
            "generated {generated}: {reply:?}"
        );
        let session = e.session.tokens().to_vec();
        assert_eq!(to_u32(e.transcript.tokens()), session);
        // The EOG id that ended the reply was neither recorded nor evaluated.
        assert!(!e.family.tok.is_eog(*session.last().unwrap()));
        let last = e.transcript.spans().last().unwrap();
        assert_eq!(last.role, crate::ds4tokens::SpanRole::Assistant);
        assert_eq!(last.text, reply);
        assert_eq!(e.kinds.last(), Some(&Kind::Assistant));
        // The reply opened with the generation prefix.
        let prefix_at = session.len() - usize::try_from(generated).unwrap() - 1;
        assert_eq!(session[prefix_at], marker("<|assistant|>"));

        let t2 = format!("{TURN}[assistant]\n{reply}\n[user]\nmore\n");
        let probe = e.kv_reuse_probe(&t2, ThinkMode::Off).unwrap();
        assert_eq!(usize::try_from(probe.common).unwrap(), session.len());
        assert_eq!(probe.live, probe.common);
    }

    #[test]
    fn generation_starts_at_the_assistant_marker() {
        let e = engine();
        assert_eq!(
            e.family.generation_prefix(Some(Kind::User), true),
            vec![marker("<|assistant|>")]
        );
        assert!(e.family.is_stop(marker("<|user|>")));
        assert!(e.family.is_stop(marker("<|end_of_text|>")));
        assert!(!e.family.is_stop(marker("<|assistant|>")));
        assert_eq!(e.family.bos(), None);
    }

    /// Control ids stream as their spelling (`<think>` and the call tags
    /// need no translation), and a multi-byte character split over byte
    /// tokens streams whole.
    #[test]
    fn control_ids_stream_as_their_spelling() {
        let e = engine();
        let mut tr = ();
        let mut u = crate::engine::Utf8Stream::default();
        let mut out = String::new();
        let mut toks = vec![marker("<think>")];
        toks.extend(e.family.tok.encode_plain("é"));
        toks.push(marker("</think>"));
        toks.push(marker("<tool_call>"));
        for t in toks {
            out.push_str(&e.family.translate(&mut tr, t, &mut u));
        }
        out.push_str(&u.flush());
        assert_eq!(out, "<think>é</think><tool_call>");
    }

    #[test]
    fn top_k_keeps_only_the_k_largest_logits_when_sampling() {
        let e = engine();
        let f = &e.family;
        let mut l = vec![0.0f32, 5.0, 1.0, 4.0, 3.0];
        FlintFamily::filter_logits_k(&mut l, 2, false);
        assert!(l[1].is_finite() && l[3].is_finite());
        assert!(
            l[0] == f32::NEG_INFINITY && l[2] == f32::NEG_INFINITY && l[4] == f32::NEG_INFINITY
        );
        let mut g = vec![0.0f32, 5.0, 1.0];
        FlintFamily::filter_logits_k(&mut g, 1, true);
        assert_eq!(g, vec![0.0, 5.0, 1.0], "greedy is untouched");
        let mut ties = vec![2.0f32, 3.0, 2.0, 1.0];
        FlintFamily::filter_logits_k(&mut ties, 2, false);
        assert_eq!(ties, vec![2.0, 3.0, 2.0, f32::NEG_INFINITY], "ties kept");
        let mut short = vec![1.0f32, 2.0];
        FlintFamily::filter_logits_k(&mut short, 20, false);
        assert_eq!(short, vec![1.0, 2.0]);
        // The GGUF's top_k (20) is what `filter_logits` applies.
        let mut wide: Vec<f32> = (0..30u8).map(f32::from).collect();
        f.filter_logits(&mut wide, false);
        assert_eq!(wide.iter().filter(|x| x.is_finite()).count(), 20);
    }

    /// One f32 score copy (`heads × t × (keys + t) × 4` bytes) stays at or
    /// below 2 GiB, between 32 and 512 tokens.
    #[test]
    fn the_prefill_chunk_shrinks_with_depth() {
        assert_eq!(chunk_for(32, 1_000), 512);
        assert!(chunk_for(32, 65_536) <= 256);
        assert_eq!(chunk_for(32, 65_536), 255);
        assert_eq!(chunk_for(32, 10_000_000), 32);
        for keys in [0, 1_000, 30_000, 65_536, 200_000] {
            let t = chunk_for(32, keys);
            if t > 32 {
                assert!(32 * t * (keys + t) * 4 <= 2 << 30, "keys {keys}: {t}");
            }
        }
    }

    #[test]
    fn the_default_context_is_capped() {
        assert_eq!(
            crate::nativeengine::default_ctx(
                512 << 30,
                262_144,
                200_704,
                Some(FLINT_DEFAULT_CTX_CAP)
            ),
            65_536
        );
        assert_eq!(FlintFamily::default_ctx_cap(), Some(65_536));
    }

    #[test]
    fn a_file_larger_than_the_gpu_buffer_is_refused() {
        const PAGE: u64 = 16_384;
        assert!(exceeds_gpu_buffer(22_863_084_160, PAGE, Some(20 << 30)));
        assert!(!exceeds_gpu_buffer(
            22_863_084_160,
            PAGE,
            Some(86_586_540_032)
        ));
        assert!(!exceeds_gpu_buffer(22_863_084_160, PAGE, None));
        // The mapping is bound rounded up to a whole page, as `no_copy`
        // binds it: a 100-byte file needs a 16 KiB buffer.
        assert!(!exceeds_gpu_buffer(100, PAGE, Some(PAGE)));
        assert!(exceeds_gpu_buffer(100, PAGE, Some(PAGE - 1)));
        assert!(exceeds_gpu_buffer(PAGE + 1, PAGE, Some(2 * PAGE - 1)));
        assert!(!exceeds_gpu_buffer(PAGE + 1, PAGE, Some(2 * PAGE)));
    }

    #[test]
    fn the_name_is_the_files_when_it_says_flint() {
        assert_eq!(engine().model_name(), "Flint");
        for (name, want) in [("Flint 2 Mini", "Flint 2 Mini"), ("Mystery", "Flint")] {
            let b = tiny_vocab(
                tiny_flint_builder(&config(), 256 + FLINT_MARKERS.len(), 0.5, SEED),
                &FLINT_MARKERS,
            )
            .kv("general.name", &Value::Str(name.into()));
            assert_eq!(open_bytes(&b.bytes()).model_name(), want);
        }
    }

    /// The header alone gives the window and the KV size.
    #[test]
    fn probe_reads_the_header() {
        let p = std::env::temp_dir().join(format!("plank-flint-probe-{}.gguf", std::process::id()));
        let b = tiny_vocab(
            tiny_flint_builder(&config(), 256 + FLINT_MARKERS.len(), 0.5, SEED),
            &FLINT_MARKERS,
        );
        std::fs::write(&p, b.bytes()).unwrap();
        let got = FlintFamily::probe(Path::new(&p));
        let _ = std::fs::remove_file(&p);
        assert_eq!(got.unwrap(), (256, config().kv_bytes_per_token()));
        assert_eq!(engine().ctx_size(), 128);
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Gemma 4 behind [`Engine`], over the native-Rust `gemma-engine` crate.
//!
//! The transcript discipline is the ds4 engine's: the rendered transcript is
//! split into sections, the leading sections that match recorded spans keep
//! their tokens verbatim, and only the rest is rendered (with Gemma's chat
//! template) and tokenized. Unlike the ds4 KV, Gemma's is prefix-truncatable,
//! so a prompt that diverges anywhere keeps the KV up to that token and
//! prefills only the remainder.
//!
//! Gemma writes its reasoning in a `<|channel>thought\n…<channel|>` block,
//! which [`ThinkTranslator`] streams as `<think>…</think>` so the renderer
//! sees what it sees from every other local model. Past turns' thoughts stay
//! in history: recorded tokens are reused verbatim.

use std::path::Path;
use std::sync::Arc;

use gemma_engine::Device;
use gemma_engine::model::Model;
use gemma_engine::sample::Sampler;
use gemma_engine::session::Session;
use gemma_engine::template::{self, Kind};
use gemma_engine::tokenizer::GemmaTokenizer;

use crate::ds4tokens::{SectionKey, SpanRole, TokenTranscript, parse_sections};
use crate::engine::{
    Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, KvReuse, PrefillProgress,
    Prompt, ThinkMode, Utf8Stream,
};
use crate::kvcache::KVCache;

/// Tokens a prefill evaluates between progress reports (the session's own
/// chunk size, so each report is one forward).
const PREFILL_CHUNK: usize = 512;

/// The context used when the caller asks for none (`ctx <= 0`).
const DEFAULT_CTX: usize = 32_768;

/// Bytes of a thought block's channel name held back before giving up on
/// finding its `\n`. The model writes `thought\n`; anything longer is not a
/// channel name and is streamed rather than lost.
const CHANNEL_NAME_MAX: usize = 64;

/// The engine: one model, one live session, and the token transcript that
/// describes what the session's KV was built from.
#[derive(Debug)]
pub struct GemmaEngine {
    model: Arc<Model>,
    session: Session,
    transcript: TokenTranscript,
    /// The template kind of each span, parallel to `transcript.spans()`.
    kinds: Vec<Kind>,
    think: ThinkMode,
    trusted_len: usize,
    /// `"metal"` or `"cpu"`: where the model runs.
    device_name: &'static str,
    /// Set by `warm_reset` until the walk's first append or sync: the window
    /// in which `kvtier` restores a tier checkpoint. A restore there keeps the
    /// warm buffer, because the walk goes on to append every tier again —
    /// restored ones included — and taking the checkpoint's own transcript
    /// would hold those tiers twice.
    warm_pending: bool,
}

/// Ids the decode loop and the stream translator look up once.
#[derive(Debug, Clone, Copy)]
struct Controls {
    turn_close: Option<u32>,
    resp_open: Option<u32>,
    channel_open: Option<u32>,
    channel_close: Option<u32>,
}

impl Controls {
    fn of(tok: &GemmaTokenizer) -> Self {
        Self {
            turn_close: tok.control_id(template::TURN_CLOSE),
            resp_open: tok.control_id(template::RESP_OPEN),
            channel_open: tok.control_id(template::CHANNEL_OPEN),
            channel_close: tok.control_id(template::CHANNEL_CLOSE),
        }
    }
}

/// Whether sampling `id` ends the reply. A stop token is neither recorded nor
/// evaluated: the next section's rendering supplies it (`<turn|>` opens a
/// user section, `<|tool_response>` opens a tool result).
fn is_stop(tok: &GemmaTokenizer, c: Controls, id: u32) -> bool {
    id == tok.eos() || Some(id) == c.turn_close || Some(id) == c.resp_open
}

/// Turns Gemma's thought channel into `<think>` tags while streaming.
///
/// `<|channel>` becomes `<think>` and the channel name after it (`thought\n`)
/// is swallowed; `<channel|>` becomes `</think>`. Any other control id is
/// emitted as its spelling so the tool-call parser sees `<|tool_call>`,
/// `<|"|>` and `<tool_call|>`; plain ids go through the UTF-8 stream.
#[derive(Debug, Default)]
pub struct ThinkTranslator {
    /// Inside a channel name: text is held until its `\n`.
    naming: bool,
    held: String,
    /// Looked up on the first token.
    controls: Option<Controls>,
}

impl ThinkTranslator {
    /// The text `id` contributes to the stream.
    pub fn push(&mut self, tok: &GemmaTokenizer, id: u32, utf8: &mut Utf8Stream) -> String {
        let c = *self.controls.get_or_insert_with(|| Controls::of(tok));
        if tok.is_control(id) {
            // A control id ends any multi-byte run and any channel name.
            let mut out = utf8.flush();
            if self.naming {
                out.insert_str(0, &std::mem::take(&mut self.held));
                self.naming = false;
            }
            if Some(id) == c.channel_open {
                self.naming = true;
                out.push_str("<think>");
            } else if Some(id) == c.channel_close {
                out.push_str("</think>");
            } else {
                out.push_str(&String::from_utf8_lossy(&tok.piece_bytes(id)));
            }
            return out;
        }
        let text = utf8.push(tok.piece_bytes(id));
        if !self.naming {
            return text;
        }
        self.held.push_str(&text);
        if let Some(nl) = self.held.find('\n') {
            self.naming = false;
            let rest = self.held[nl + 1..].to_owned();
            self.held.clear();
            return rest;
        }
        if self.held.len() > CHANNEL_NAME_MAX {
            self.naming = false;
            return std::mem::take(&mut self.held);
        }
        String::new()
    }
}

/// The span role a template kind is recorded under.
fn role_of(kind: Kind) -> SpanRole {
    match kind {
        Kind::System => SpanRole::System,
        Kind::User | Kind::ToolResult => SpanRole::User,
        Kind::Assistant => SpanRole::Assistant,
    }
}

/// The section tag `template::classify` reads for a span role.
fn tag_of(role: SpanRole) -> &'static str {
    match role {
        SpanRole::System => "system",
        SpanRole::User => "user",
        SpanRole::Assistant => "assistant",
    }
}

fn to_i32(ids: &[u32]) -> Vec<i32> {
    ids.iter()
        .map(|&t| i32::try_from(t).unwrap_or(i32::MAX))
        .collect()
}

fn to_u32(ids: &[i32]) -> Vec<u32> {
    ids.iter().map(|&t| u32::try_from(t).unwrap_or(0)).collect()
}

fn count(n: usize) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn engine_error(e: gemma_engine::Error) -> EngineError {
    EngineError::new(e.0)
}

/// How the template renders: what [`reconcile_into`] needs besides the
/// transcript it edits.
#[derive(Debug, Clone, Copy)]
struct Render<'a> {
    tok: &'a GemmaTokenizer,
    think: bool,
    trusted_len: usize,
}

impl<'a> Render<'a> {
    fn new(tok: &'a GemmaTokenizer, think: ThinkMode, trusted_len: usize) -> Self {
        Self {
            tok,
            think: !matches!(think, ThinkMode::Off),
            trusted_len,
        }
    }

    /// Renders one section after `kinds`' last span and appends it.
    fn push(
        &self,
        transcript: &mut TokenTranscript,
        kinds: &mut Vec<Kind>,
        role: &str,
        text: &str,
    ) {
        let prev = kinds.last().copied();
        let kind = template::classify(role, text, prev);
        let pieces = template::render(kind, text, prev, self.think, self.trusted_len);
        let mut ids = Vec::new();
        if transcript.spans().is_empty()
            && let Some(bos) = self.tok.bos()
        {
            ids.push(bos);
        }
        ids.extend(self.tok.encode_pieces(&pieces));
        transcript.push_span(role_of(kind), 0, text.to_owned(), &to_i32(&ids));
        kinds.push(kind);
    }

    /// Brings `transcript` in line with the rendered transcript `flat`: the
    /// leading spans whose (role, text) match keep their tokens, the rest are
    /// rendered from text.
    fn reconcile(&self, transcript: &mut TokenTranscript, kinds: &mut Vec<Kind>, flat: &str) {
        let sections = parse_sections(flat);
        let keys: Vec<SectionKey> = sections
            .iter()
            .filter_map(|(role, text)| {
                SpanRole::from_tag(role).map(|role| SectionKey {
                    role,
                    text: text.clone(),
                })
            })
            .collect();
        let keep = transcript.common_prefix(&keys);
        crate::engine::kv_debug(|| {
            format!(
                "gemma reconcile: {} spans held, {} sections in, kept {keep}",
                transcript.spans().len(),
                keys.len()
            )
        });
        transcript.truncate_spans(keep);
        kinds.truncate(keep);
        for (role, text) in sections.iter().skip(keep) {
            self.push(transcript, kinds, role, text);
        }
    }

    /// The tokens a generation after `kinds`' last span starts with.
    fn generation_prefix(&self, kinds: &[Kind]) -> Vec<u32> {
        self.tok
            .encode_pieces(&template::generation_prefix(kinds.last().copied()))
    }
}

/// Bytes of RAM on this machine, or `usize::MAX` when it cannot be read (so
/// the KV check never refuses on a failed probe).
#[cfg(target_os = "macos")]
fn physical_memory() -> usize {
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is NUL-terminated, `bytes` and `len` are valid out
    // pointers sized for `hw.memsize`'s u64, and no new value is set.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut bytes).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        usize::try_from(bytes).unwrap_or(usize::MAX)
    } else {
        usize::MAX
    }
}

/// Bytes of RAM on this machine, or `usize::MAX` when it cannot be read.
#[cfg(not(target_os = "macos"))]
fn physical_memory() -> usize {
    // SAFETY: sysconf only reads system configuration.
    let (pages, size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    match (usize::try_from(pages), usize::try_from(size)) {
        (Ok(p), Ok(s)) => p.saturating_mul(s),
        _ => usize::MAX,
    }
}

/// Metal when this machine has it, else the CPU.
fn pick_device() -> (Device, &'static str) {
    #[cfg(target_os = "macos")]
    if let Ok(d) = Device::new_metal(0) {
        return (d, "metal");
    }
    (Device::Cpu, "cpu")
}

impl GemmaEngine {
    /// Opens the Gemma 4 GGUF at `path` on Metal (the CPU when there is no
    /// Metal device) with a `ctx`-token context, clamped to the model's; a
    /// `ctx` of zero or less means `min(32768, context_length)`.
    ///
    /// # Errors
    /// When the file does not load as a Gemma 4 model, or when the KV for
    /// `ctx` tokens would take more than half the machine's memory.
    pub fn open(path: &Path, ctx: i32) -> Result<Self, EngineError> {
        let (device, name) = pick_device();
        Self::open_on(path, ctx, &device, name)
    }

    /// [`GemmaEngine::open`] forced onto the CPU, for tests.
    ///
    /// # Errors
    /// As [`GemmaEngine::open`].
    #[cfg(test)]
    pub fn open_on_cpu(path: &Path, ctx: i32) -> Result<Self, EngineError> {
        Self::open_on(path, ctx, &Device::Cpu, "cpu")
    }

    fn open_on(
        path: &Path,
        ctx: i32,
        device: &Device,
        device_name: &'static str,
    ) -> Result<Self, EngineError> {
        let cap = usize::try_from(ctx)
            .ok()
            .filter(|&c| c > 0)
            .unwrap_or(DEFAULT_CTX);
        // Rotary tables are built for the capped context only, which also
        // makes `context_length()` the clamped context.
        let model = Model::open_with_ctx(path, device, cap).map_err(engine_error)?;
        let ctx = model.context_length();
        let need = model.kv_bytes_per_token().saturating_mul(ctx);
        if need > physical_memory() / 2 {
            return Err(EngineError::new(format!(
                "a {ctx}-token context needs {} GB of KV; lower it with --ctx",
                need.div_ceil(1 << 30)
            )));
        }
        let session = Session::new(model.clone(), ctx);
        Ok(Self {
            model,
            session,
            transcript: TokenTranscript::new(),
            kinds: Vec::new(),
            think: ThinkMode::default(),
            trusted_len: 0,
            device_name,
            warm_pending: false,
        })
    }

    /// Where the model runs: `"metal"` or `"cpu"`.
    #[must_use]
    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    fn render(&self) -> Render<'_> {
        Render::new(&self.model.tokenizer, self.think, self.trusted_len)
    }

    /// Reconciles the token transcript with the rendered transcript `flat`.
    fn reconcile(&mut self, flat: &str) {
        Render::new(&self.model.tokenizer, self.think, self.trusted_len).reconcile(
            &mut self.transcript,
            &mut self.kinds,
            flat,
        );
    }

    /// Drops every recorded token, for a change that alters the system turn.
    fn invalidate(&mut self) {
        self.session.truncate(0);
        self.transcript = TokenTranscript::new();
        self.kinds.clear();
        self.warm_pending = false;
    }

    /// Prefills `toks` after the session's first `base` tokens (the session
    /// already holds exactly those), reporting progress per chunk. Returns the
    /// last logits, or `None` when interrupted.
    fn prefill(
        &mut self,
        base: usize,
        toks: &[u32],
        interrupt: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let total_abs = base + toks.len();
        on_event(EngineEvent::Prefill(PrefillProgress::primed(
            count(base),
            count(total_abs),
        )));
        let start = std::time::Instant::now();
        let mut total = count(total_abs);
        let mut last = None;
        for chunk in toks.chunks(PREFILL_CHUNK) {
            match self
                .session
                .prefill(chunk, interrupt)
                .map_err(engine_error)?
            {
                Some(l) => last = Some(l),
                None => return Ok(None),
            }
            on_event(EngineEvent::Prefill(PrefillProgress::from_absolute(
                count(base),
                count(self.session.tokens().len()),
                &mut total,
                start.elapsed().as_secs_f64(),
            )));
        }
        Ok(last)
    }

    /// Fails before touching the session when `need` tokens cannot fit.
    fn check_fits(&self, need: usize) -> Result<(), EngineError> {
        if need > self.session.ctx() {
            return Err(EngineError::new(format!(
                "context full: {need} tokens > {}",
                self.session.ctx()
            )));
        }
        Ok(())
    }

    /// Records a sampled reply: the generation prefix it followed and its ids,
    /// as a new assistant span — or as more of the last one when that is
    /// already an assistant reply (a resumed `/btw` pass).
    fn record_reply(&mut self, text: String, prefix: &[u32], reply: &[u32]) {
        if reply.is_empty() {
            return;
        }
        let mut ids = to_i32(prefix);
        ids.extend(to_i32(reply));
        if self.transcript.last_is_assistant() {
            self.transcript.extend_last_span(&text, &ids);
        } else {
            self.transcript
                .push_span(SpanRole::Assistant, 0, text, &ids);
            self.kinds.push(Kind::Assistant);
        }
    }
}

impl Engine for GemmaEngine {
    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        self.warm_pending = false;
        self.reconcile(prompt.flat());
        let prefix = self.render().generation_prefix(&self.kinds);
        let mut full = to_u32(self.transcript.tokens());
        full.extend_from_slice(&prefix);
        self.check_fits(full.len())?;

        // The last prompt token is always evaluated: its logits pick the
        // first reply token.
        let mut common = common_prefix(self.session.tokens(), &full);
        if common == full.len() {
            common = common.saturating_sub(1);
        }
        let live = self.session.tokens().len();
        // Below a restored snapshot's sliding window the session empties.
        let common = self.session.truncate(common);
        crate::engine::kv_debug(|| {
            format!(
                "gemma generate: prompt={} live={live} reused={common}",
                full.len()
            )
        });
        let Some(mut logits) = self.prefill(common, &full[common..], interrupt, on_event)? else {
            return Ok(GenerationStats {
                interrupted: true,
                ctx_used: count(self.session.tokens().len()),
                ..GenerationStats::default()
            });
        };

        let model = self.model.clone();
        let tok = &model.tokenizer;
        let controls = Controls::of(tok);
        let mut sampler = Sampler::new(if opts.seed != 0 {
            opts.seed
        } else {
            0x2545_f491_4f6c_dd1d
        });
        let mut translator = ThinkTranslator::default();
        let mut utf8 = Utf8Stream::default();
        let mut reply: Vec<u32> = Vec::new();
        let mut reply_text = String::new();
        let mut interrupted = false;
        let mut generated: i32 = 0;
        let start = std::time::Instant::now();
        let mut steady_mark: Option<(std::time::Instant, i32)> = None;
        while (opts.n_predict < 0 || generated < opts.n_predict)
            && self.session.tokens().len() < self.session.ctx()
        {
            if steady_mark.is_none()
                && start.elapsed().as_secs_f64() >= crate::engine::STEADY_WARMUP_SECS
            {
                steady_mark = Some((std::time::Instant::now(), generated));
            }
            if interrupt() {
                interrupted = true;
                break;
            }
            let t = sampler.sample(&logits, opts.temperature, opts.top_p, opts.min_p, greedy());
            if is_stop(tok, controls, t) {
                break;
            }
            let text = translator.push(tok, t, &mut utf8);
            if !text.is_empty() {
                reply_text.push_str(&text);
                on_event(EngineEvent::Text(text));
            }
            reply.push(t);
            generated += 1;
            logits = self.session.step(t).map_err(engine_error)?;
        }
        let tail = utf8.flush();
        if !tail.is_empty() {
            reply_text.push_str(&tail);
            on_event(EngineEvent::Text(tail));
        }
        self.record_reply(reply_text, &prefix, &reply);

        let secs = start.elapsed().as_secs_f64();
        Ok(GenerationStats {
            generated,
            tps: if secs > 0.0 {
                f64::from(generated) / secs
            } else {
                0.0
            },
            steady_tps: crate::engine::rate_since(steady_mark, generated),
            ctx_used: count(self.session.tokens().len()),
            interrupted,
            usage: None,
            spec: crate::engine::SpecStats::default(),
        })
    }

    fn emits_think_tags(&self) -> bool {
        true
    }

    /// The candle KV truncates to the common prefix in `generate`, so fork
    /// snapshots and ladder rungs would only serialise the whole f32 cache
    /// (about 114 KB per token on E4B) for nothing.
    fn kv_truncates_exactly(&self) -> bool {
        true
    }

    /// Truncates the session to its common prefix with `transcript`, the
    /// truncation the next `generate` would make, so a snapshot taken before
    /// that turn does not record a sidechain's tail. A restored session keeps
    /// only `window + SNAPSHOT_SLACK` positions per sliding layer, so a tail
    /// longer than the slack would otherwise turn the resumed turn into a
    /// rebuild from zero. Left alone during a warm walk, which owns the
    /// transcript until `warm_sync`.
    fn sync_to_prefix(&mut self, transcript: &str) {
        if self.warm_pending {
            return;
        }
        self.reconcile(transcript);
        let want = to_u32(self.transcript.tokens());
        let common = common_prefix(self.session.tokens(), &want);
        let kept = self.session.truncate(common);
        crate::engine::kv_debug(|| {
            format!(
                "gemma sync_to_prefix: transcript={} common={common} kept={kept}",
                want.len()
            )
        });
    }

    /// Reports the reusable prefix as the live end: `live == common`.
    ///
    /// Gemma's KV truncates exactly, so a prompt that diverges behind the live
    /// end keeps every token up to the divergence and prefills only the rest.
    /// That is never the rebuild-from-zero shape `KvReuse::rebuilds_from_zero`
    /// describes (the ds4 sync's), which the agent answers by restoring a
    /// ladder rung or fork snapshot and by skipping prompt suggestions.
    /// Reporting the real session length would buy both for nothing.
    ///
    /// The exception is a session restored from a snapshot, whose sliding
    /// layers hold only their window plus `SNAPSHOT_SLACK`: diverging so far
    /// behind its end that [`Session::reusable`] is 0 empties it, so the probe
    /// reports the real length, the rebuild shape. So does a prompt sharing no
    /// token at all, which keeps nothing on any session.
    fn kv_reuse_probe(&mut self, transcript: &str, _think: ThinkMode) -> Option<KvReuse> {
        let render = self.render();
        let mut tokens = self.transcript.clone();
        let mut kinds = self.kinds.clone();
        render.reconcile(&mut tokens, &mut kinds, transcript);
        let mut full = to_u32(tokens.tokens());
        full.extend(render.generation_prefix(&kinds));
        let common = common_prefix(self.session.tokens(), &full);
        let live = if common > 0 && self.session.reusable(common) == common {
            common
        } else {
            self.session.tokens().len()
        };
        Some(KvReuse {
            live: count(live),
            common: count(common),
        })
    }

    fn set_think_mode(&mut self, mode: ThinkMode) {
        if mode != self.think {
            self.think = mode;
            self.invalidate();
        }
    }

    fn set_trusted_system_prefix(&mut self, len: usize) {
        if len != self.trusted_len {
            self.trusted_len = len;
            self.invalidate();
        }
    }

    fn count_tokens(&self, text: &str) -> i32 {
        count(self.model.tokenizer.encode_plain(text).len())
    }

    fn get_kv(&mut self) -> Option<KVCache> {
        Some(KVCache::new(
            self.session.snapshot().ok()?,
            self.transcript.clone(),
        ))
    }

    /// Restores the session and, outside a warm walk, the transcript the
    /// checkpoint was captured with. Inside one (right after `warm_reset`) the
    /// warm buffer stays: the walk re-appends every tier, and `warm_sync`
    /// matches it against the restored session's own tokens.
    fn set_kv(&mut self, cache: &KVCache) -> Result<(), EngineError> {
        self.session.restore(cache.kv()).map_err(engine_error)?;
        if self.warm_pending {
            return Ok(());
        }
        self.transcript = cache.transcript().clone();
        self.kinds.clear();
        for span in self.transcript.spans() {
            let kind =
                template::classify(tag_of(span.role), &span.text, self.kinds.last().copied());
            self.kinds.push(kind);
        }
        Ok(())
    }

    fn can_release_gpu(&self) -> bool {
        true
    }

    /// Places the system turn in the transcript. Nothing is prefilled until
    /// [`Engine::warm_sync`], so a checkpoint restored in between is not paid
    /// for twice.
    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.transcript = TokenTranscript::new();
        self.kinds.clear();
        self.reconcile(&format!("[system]\n{system}\n"));
        self.warm_pending = true;
        Ok(())
    }

    fn warm_append(&mut self, text: Option<&str>) -> Result<(), EngineError> {
        self.warm_pending = false;
        if let Some(text) = text {
            // Trimmed as `parse_sections` trims, so the span matches the
            // section the next turn's transcript carries.
            Render::new(&self.model.tokenizer, self.think, self.trusted_len).push(
                &mut self.transcript,
                &mut self.kinds,
                "user",
                text.trim_end(),
            );
        }
        Ok(())
    }

    fn warm_sync(&mut self, on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        self.warm_pending = false;
        let want = to_u32(self.transcript.tokens());
        self.check_fits(want.len())?;
        let common = common_prefix(self.session.tokens(), &want);
        let common = self.session.truncate(common);
        if common == want.len() {
            return Ok(false);
        }
        self.prefill(common, &want[common..], &|| false, on_event)?;
        Ok(true)
    }

    fn ctx_size(&self) -> i32 {
        count(self.session.ctx())
    }

    fn model_name(&self) -> String {
        self.model.name.clone()
    }

    fn is_local(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineEvent, GenerationOptions, Prompt, ThinkMode};
    use crate::kvcache::KVCache;

    fn engine() -> GemmaEngine {
        let p = std::env::temp_dir().join(format!(
            "plank-gemma-{}-{:?}.gguf",
            std::process::id(),
            std::thread::current().id()
        ));
        gemma_engine::testgguf::write_tiny(&p, &gemma_engine::testgguf::TinyConfig::default())
            .unwrap();
        GemmaEngine::open_on_cpu(&p, 256).unwrap()
    }

    fn opts() -> GenerationOptions {
        GenerationOptions {
            n_predict: 4,
            ctx_size: 256,
            temperature: 0.0,
            think_mode: ThinkMode::Off,
            ..GenerationOptions::default()
        }
    }

    fn run(e: &mut GemmaEngine, flat: &str) -> String {
        let mut text = String::new();
        e.generate(
            Prompt::Flat(flat),
            &opts(),
            &|| false,
            &|| true,
            &mut |ev| {
                if let EngineEvent::Text(t) = ev {
                    text.push_str(&t);
                }
            },
        )
        .unwrap();
        text
    }

    #[test]
    fn the_second_turn_prefills_only_its_suffix() {
        let mut e = engine();
        let t1 = "[system]\nsys\n[user]\nhello\n";
        let reply = run(&mut e, t1);
        let live = e.session.tokens().len();
        let t2 = format!("{t1}[assistant]\n{reply}\n[user]\nmore\n");
        let probe = e.kv_reuse_probe(&t2, ThinkMode::Off).unwrap();
        assert_eq!(
            usize::try_from(probe.common).unwrap(),
            live,
            "the whole first turn must be reused"
        );
    }

    /// The recorded reply (generation prefix plus sampled ids, stop token
    /// excluded) is exactly what the session evaluated.
    #[test]
    fn the_transcript_describes_the_live_kv_after_a_turn() {
        let mut e = engine();
        run(&mut e, "[system]\nsys\n[user]\nhello\n");
        assert_eq!(to_u32(e.transcript.tokens()), e.session.tokens());
        assert_eq!(e.kinds.len(), e.transcript.spans().len());
    }

    /// `warm_reset` only places tokens, so a checkpoint restored after it is
    /// not paid for twice; `warm_sync` prefills once and then has nothing to do.
    #[test]
    fn the_warm_walk_prefills_only_at_sync() {
        let mut e = engine();
        e.warm_reset("sys").unwrap();
        e.warm_append(Some("project context\n")).unwrap();
        assert!(e.session.tokens().is_empty());
        assert!(e.warm_sync(&mut |_| {}).unwrap());
        assert_eq!(to_u32(e.transcript.tokens()), e.session.tokens());
        assert!(!e.warm_sync(&mut |_| {}).unwrap());
        let live = e.session.tokens().len();
        let probe = e
            .kv_reuse_probe(
                "[system]\nsys\n[user]\nproject context\n[user]\nhi\n",
                ThinkMode::Off,
            )
            .unwrap();
        assert_eq!(usize::try_from(probe.common).unwrap(), live);
    }

    /// `kvtier::warm` with a two-tier hit: reset, restore the checkpoint taken
    /// after `[sys, stable]`, then append every tier as the walk does. Each
    /// tier is held once and only the volatile tier is prefilled.
    #[test]
    fn a_tier_restore_holds_each_tier_once() {
        let mut first = engine();
        first.warm_reset("sys").unwrap();
        first.warm_append(Some("stable")).unwrap();
        first.warm_sync(&mut |_| {}).unwrap();
        let checkpoint = first.get_kv().unwrap();
        assert_eq!(checkpoint.transcript().spans().len(), 2);

        let mut e = engine();
        e.warm_reset("sys").unwrap();
        e.set_kv(&checkpoint).unwrap();
        let restored = e.session.tokens().len();
        e.warm_append(None).unwrap();
        e.warm_append(Some("stable")).unwrap();
        e.warm_append(Some("volatile")).unwrap();
        let texts: Vec<&str> = e
            .transcript
            .spans()
            .iter()
            .map(|s| s.text.as_str())
            .collect();
        assert_eq!(texts, ["sys", "stable", "volatile"]);
        assert_eq!(e.kinds.len(), 3);
        assert!(e.warm_sync(&mut |_| {}).unwrap());
        let volatile = e.transcript.spans()[2].ntokens;
        assert_eq!(e.session.tokens().len(), restored + volatile);
        assert_eq!(to_u32(e.transcript.tokens()), e.session.tokens());
    }

    /// Outside a warm walk a restore brings back the checkpoint's transcript,
    /// so a resumed session's spans match its KV.
    #[test]
    fn a_session_restore_takes_the_checkpoint_transcript() {
        let mut a = engine();
        run(&mut a, "[system]\nsys\n[user]\nhello\n");
        let snap = a.get_kv().unwrap();
        let mut b = engine();
        b.set_kv(&snap).unwrap();
        assert_eq!(b.transcript, a.transcript);
        assert_eq!(b.kinds, a.kinds);
    }

    /// A prompt diverging behind the live end reports `live == common`: the
    /// KV truncates there, so nothing rebuilds from zero.
    #[test]
    fn a_divergence_behind_the_live_end_is_not_a_rebuild() {
        let mut e = engine();
        run(&mut e, "[system]\nsys\n[user]\nhello\n");
        let live = e.session.tokens().len();
        let probe = e
            .kv_reuse_probe("[system]\nsys\n[user]\nsomething else\n", ThinkMode::Off)
            .unwrap();
        assert_eq!(probe.live, probe.common);
        assert!(usize::try_from(probe.common).unwrap() < live);
        assert!(probe.common > 0 && !probe.rebuilds_from_zero());
    }

    /// A restored snapshot keeps only the sliding window (plus
    /// `SNAPSHOT_SLACK`) of each sliding layer, so a prompt diverging well
    /// behind its end cannot truncate
    /// there: the probe says so (`live` stays the session length, so the
    /// agent sees a rebuild), and the turn rebuilds from scratch to the same
    /// result as a fresh engine.
    #[test]
    fn a_divergence_below_a_restored_window_rebuilds_truthfully() {
        let mut a = engine();
        // Longer than the window plus the snapshot slack, so the restore is
        // trimmed and the divergence lands below its floor.
        let long = "a long enough prompt ".repeat(6);
        run(
            &mut a,
            &format!("[system]\nsys\n[user]\nhello there, {long}\n"),
        );
        let snap = a.get_kv().unwrap();
        let mut b = engine();
        b.set_kv(&snap).unwrap();
        let live = b.session.tokens().len();
        let other = "[system]\nsys\n[user]\nsomething else\n";
        let probe = b.kv_reuse_probe(other, ThinkMode::Off).unwrap();
        assert_eq!(usize::try_from(probe.live).unwrap(), live);
        assert!(probe.common > 0 && probe.rebuilds_from_zero());
        let got = run(&mut b, other);
        assert_eq!(to_u32(b.transcript.tokens()), b.session.tokens());
        assert_eq!(got, run(&mut engine(), other));
    }

    /// Runs one turn and returns its reply with the number of prompt tokens
    /// it prefilled (the first progress event's `total`: the cached prefix
    /// excluded).
    fn run_prefilled(e: &mut GemmaEngine, flat: &str) -> (String, usize) {
        let mut text = String::new();
        let mut prefilled = None;
        e.generate(
            Prompt::Flat(flat),
            &opts(),
            &|| false,
            &|| true,
            &mut |ev| match ev {
                EngineEvent::Text(t) => text.push_str(&t),
                EngineEvent::Prefill(p) if prefilled.is_none() => {
                    prefilled = Some(usize::try_from(p.total).unwrap());
                }
                _ => {}
            },
        )
        .unwrap();
        (text, prefilled.expect("a prefill event"))
    }

    /// Tokens `generate` evaluates for `flat`: the rendered transcript plus
    /// the generation prefix.
    fn prompt_len(flat: &str) -> usize {
        let mut e = engine();
        e.reconcile(flat);
        e.transcript.tokens().len() + e.render().generation_prefix(&e.kinds).len()
    }

    /// A transcript whose user turn is `chars` bytes, for prompts past the
    /// tiny model's window plus the snapshot slack.
    fn long_turn(chars: usize) -> String {
        let body: String = "the quick brown fox jumps over the lazy dog "
            .chars()
            .cycle()
            .take(chars)
            .collect();
        format!("[system]\nsys\n[user]\n{}\n", body.trim_end())
    }

    /// The `/resume` path: a snapshot taken at the end of a turn, restored
    /// on a fresh engine (trimmed, since the turn is longer than the window
    /// plus the slack), prefills only the next user turn.
    #[test]
    fn a_restored_turn_end_prefills_only_the_next_turn() {
        let mut a = engine();
        let t1 = long_turn(100);
        let reply = run(&mut a, &t1);
        let snap = a.get_kv().unwrap();
        let mut b = engine();
        b.set_kv(&snap).unwrap();
        let live = b.session.tokens().len();
        let t2 = format!("{t1}[assistant]\n{reply}\n[user]\nmore\n");
        let probe = b.kv_reuse_probe(&t2, ThinkMode::Off).unwrap();
        assert!(!probe.rebuilds_from_zero(), "{probe:?}");
        let (_, prefilled) = run_prefilled(&mut b, &t2);
        assert_eq!(prefilled, prompt_len(&t2) - live);
    }

    /// A sidechain (a suggestion, a memory pass) runs on the live session and
    /// leaves its tail there. Ending it through `sync_to_prefix`, as the agent
    /// does, drops that tail, so a snapshot taken afterwards ends at the
    /// parent's prefix and the next real turn after a restore prefills only
    /// its own suffix instead of rebuilding from zero.
    #[test]
    fn a_snapshot_after_a_synced_sidechain_resumes_without_a_rebuild() {
        let mut a = engine();
        let t1 = long_turn(30);
        let reply = run(&mut a, &t1);
        let parent = format!("{t1}[assistant]\n{reply}\n");
        let parent_len = a.session.tokens().len();
        // The sidechain: the parent plus a task well past the window + slack.
        let task = "x".repeat(100);
        run(&mut a, &format!("{parent}[user]\n{task}\n"));
        assert!(a.session.tokens().len() > parent_len + 100);
        a.sync_to_prefix(&parent);
        assert_eq!(a.session.tokens().len(), parent_len);
        let snap = a.get_kv().unwrap();
        let mut b = engine();
        b.set_kv(&snap).unwrap();
        let t2 = format!("{parent}[user]\nmore\n");
        let probe = b.kv_reuse_probe(&t2, ThinkMode::Off).unwrap();
        assert!(!probe.rebuilds_from_zero(), "{probe:?}");
        let (got, prefilled) = run_prefilled(&mut b, &t2);
        assert_eq!(prefilled, prompt_len(&t2) - parent_len);
        // The same turn on the engine that took the snapshot, never restored.
        assert_eq!(got, run(&mut a, &t2));
    }

    /// The parent's own reply is kept: syncing to the transcript the session
    /// already matches drops nothing.
    #[test]
    fn syncing_to_the_live_transcript_keeps_every_token() {
        let mut e = engine();
        let t1 = "[system]\nsys\n[user]\nhello\n";
        let reply = run(&mut e, t1);
        let live = e.session.tokens().len();
        e.sync_to_prefix(&format!("{t1}[assistant]\n{reply}\n"));
        assert_eq!(e.session.tokens().len(), live);
    }

    /// The system turn carries `<|think|>` only while thinking is on, so a
    /// change of level drops every recorded token.
    #[test]
    fn a_think_level_change_drops_the_transcript() {
        let mut e = engine();
        run(&mut e, "[system]\nsys\n[user]\nhello\n");
        e.set_think_mode(ThinkMode::Medium);
        e.set_think_mode(ThinkMode::Off);
        assert!(e.transcript.is_empty() && e.session.tokens().is_empty());
        let mut on = engine();
        on.set_think_mode(ThinkMode::Max);
        on.reconcile("[system]\nsys\n");
        let mut off = engine();
        off.set_think_mode(ThinkMode::Off);
        off.reconcile("[system]\nsys\n");
        assert_ne!(on.transcript.tokens(), off.transcript.tokens());
    }

    #[test]
    fn rerender_from_text_matches_recorded_tokens() {
        let mut a = engine();
        let flat = "[system]\nsys\n[user]\nhi\n[assistant]\nok\n[user]\n<tool_result>Tool result 1 (read):\nx\n</tool_result>\n";
        a.reconcile(flat);
        let mut b = engine();
        b.reconcile(flat);
        assert_eq!(a.transcript.tokens(), b.transcript.tokens());
        let snap = a.get_kv();
        // A fresh engine restoring the blob agrees with its own re-render.
        let mut c = engine();
        c.set_kv(&KVCache::new(
            snap.unwrap().kv().to_vec(),
            crate::ds4tokens::TokenTranscript::new(),
        ))
        .unwrap();
        c.reconcile(flat);
        assert_eq!(c.transcript.tokens(), a.transcript.tokens());
    }

    #[test]
    fn byte_fallback_pieces_stream_as_whole_chars() {
        let e = engine();
        let ids = e.model.tokenizer.encode_plain("✓");
        let mut u = crate::engine::Utf8Stream::default();
        let mut out = String::new();
        for id in ids {
            out.push_str(&u.push(e.model.tokenizer.piece_bytes(id)));
        }
        out.push_str(&u.flush());
        assert_eq!(out, "✓");
    }

    #[test]
    fn prompt_beyond_context_is_an_error() {
        let mut e = engine();
        let long = format!("[system]\n{}\n", "word ".repeat(400));
        let err = e
            .generate(
                Prompt::Flat(&long),
                &opts(),
                &|| false,
                &|| true,
                &mut |_| {},
            )
            .unwrap_err();
        assert!(err.to_string().contains("context full"), "{err}");
    }

    #[test]
    fn a_refused_snapshot_leaves_the_engine_untouched() {
        let mut e = engine();
        run(&mut e, "[system]\nsys\n[user]\nhello\n");
        let before = e.session.tokens().to_vec();
        assert!(
            e.set_kv(&KVCache::new(
                vec![0; 16],
                crate::ds4tokens::TokenTranscript::new()
            ))
            .is_err()
        );
        assert_eq!(e.session.tokens(), before);
    }

    #[test]
    fn channel_tokens_become_think_tags() {
        let e = engine();
        let mut tr = ThinkTranslator::default();
        let tok = &e.model.tokenizer;
        let mut out = String::new();
        let mut u = crate::engine::Utf8Stream::default();
        for id in tok.encode_trusted("<|channel>thought\nplan<channel|>Answer") {
            out.push_str(&tr.push(tok, id, &mut u));
        }
        out.push_str(&u.flush());
        assert_eq!(out, "<think>plan</think>Answer");
    }

    fn stream(
        e: &mut GemmaEngine,
        flat: &str,
        o: &GenerationOptions,
    ) -> (String, GenerationStats, i32) {
        let mut text = String::new();
        let mut prefilled = 0;
        let stats = e
            .generate(
                Prompt::Flat(flat),
                o,
                &|| false,
                &|| false,
                &mut |ev| match ev {
                    EngineEvent::Text(t) => text.push_str(&t),
                    EngineEvent::Prefill(p) => prefilled = prefilled.max(p.total),
                    _ => {}
                },
            )
            .unwrap();
        (text, stats, prefilled)
    }

    /// Opt-in check on the real model: text streams, the reply ends on the
    /// model's own `<turn|>`, and the next turn reuses the whole first one.
    /// `PLANK_GEMMA_GGUF=~/.plank/models/gemma-4-E4B-it-Q4_K_M.gguf cargo test
    /// --release --lib real_model_smoke -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real model: set PLANK_GEMMA_GGUF"]
    fn real_model_smoke() {
        let Ok(path) = std::env::var("PLANK_GEMMA_GGUF") else {
            return;
        };
        let t = std::time::Instant::now();
        let mut e = GemmaEngine::open(Path::new(&path), 4096).unwrap();
        eprintln!(
            "open: {:.2?} on {} ({}, ctx {})",
            t.elapsed(),
            e.device_name(),
            e.model_name(),
            e.ctx_size()
        );
        e.set_think_mode(ThinkMode::Off);
        let o = GenerationOptions {
            n_predict: 64,
            ctx_size: 4096,
            temperature: 0.0,
            think_mode: ThinkMode::Off,
            ..GenerationOptions::default()
        };
        let t1 = "[system]\nYou are terse.\n[user]\nSay hello.\n";
        let t = std::time::Instant::now();
        let (text, stats, prefilled) = stream(&mut e, t1, &o);
        eprintln!(
            "turn 1: {text:?} — prefilled {prefilled}, generated {} at {:.1} tok/s, {:.2?} total",
            stats.generated,
            stats.tps,
            t.elapsed()
        );
        assert!(!text.is_empty(), "nothing streamed");
        assert!(
            stats.generated < o.n_predict,
            "the reply did not stop on its own"
        );
        assert_eq!(to_u32(e.transcript.tokens()), e.session.tokens());

        let t2 = format!("{t1}[assistant]\n{text}\n[user]\nNow say goodbye.\n");
        let probe = e.kv_reuse_probe(&t2, ThinkMode::Off).unwrap();
        eprintln!("probe: live {} common {}", probe.live, probe.common);
        assert_eq!(probe.common, probe.live, "turn 1 must be reused whole");
        let t = std::time::Instant::now();
        let (text2, stats2, prefilled2) = stream(&mut e, &t2, &o);
        eprintln!(
            "turn 2: {text2:?} — prefilled {prefilled2}, generated {} at {:.1} tok/s, {:.2?} total",
            stats2.generated,
            stats2.tps,
            t.elapsed()
        );

        // Thinking on: the thought channel must stream as <think> tags.
        let mut e = GemmaEngine::open(Path::new(&path), 4096).unwrap();
        e.set_think_mode(ThinkMode::Medium);
        let o = GenerationOptions {
            n_predict: 512,
            think_mode: ThinkMode::Medium,
            ..o
        };
        let t = std::time::Instant::now();
        let (text3, stats3, _) = stream(
            &mut e,
            "[system]\nYou are terse.\n[user]\nA bat and a ball cost 1.10 together; the bat costs 1.00 more than the ball. What does the ball cost?\n",
            &o,
        );
        eprintln!(
            "think on: {text3:?} — generated {} at {:.1} tok/s, {:.2?} total",
            stats3.generated,
            stats3.tps,
            t.elapsed()
        );
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Gemma 4 behind [`Engine`], over the native-Rust `gemma-engine` crate.
//!
//! The transcript discipline, the KV and the warm walk are
//! [`NativeEngine`]'s; this module supplies Gemma's part of it as
//! [`GemmaFamily`]: its tokenizer, its chat template and its stop ids.
//!
//! Gemma writes its reasoning in a `<|channel>thought\n…<channel|>` block,
//! which [`ThinkTranslator`] streams as `<think>…</think>` so the renderer
//! sees what it sees from every other local model. Past turns' thoughts stay
//! in history: recorded tokens are reused verbatim.
//!
//! [`Engine`]: crate::engine::Engine
//! [`NativeEngine`]: crate::nativeengine::NativeEngine

use std::path::Path;
use std::sync::Arc;

use gemma_engine::Device;
use gemma_engine::model::Model;
use gemma_engine::template::{self, Kind};
use gemma_engine::tokenizer::GemmaTokenizer;

use crate::engine::{EngineError, Utf8Stream};
use crate::nativeengine::engine_error;

/// Bytes of a thought block's channel name held back before giving up on
/// finding its `\n`. The model writes `thought\n`; anything longer is not a
/// channel name and is streamed rather than lost.
const CHANNEL_NAME_MAX: usize = 64;

/// Gemma 4's part of [`NativeEngine`]: its tokenizer, template and thought
/// channel.
///
/// [`NativeEngine`]: crate::nativeengine::NativeEngine
#[derive(Debug)]
pub struct GemmaFamily {
    model: Arc<Model>,
    controls: Controls,
}

/// Gemma 4 behind [`crate::engine::Engine`]: one model, one live session, and
/// the token transcript that describes what the session's KV was built from.
pub type GemmaEngine = crate::nativeengine::NativeEngine<GemmaFamily>;

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

impl crate::nativeengine::NativeFamily for GemmaFamily {
    type Model = Model;
    type Translator = ThinkTranslator;
    const LABEL: &'static str = "gemma";
    const SUPPORTS_CUDA: bool = true;

    fn open(path: &Path, device: &Device, cap: usize) -> Result<(Arc<Model>, Self), EngineError> {
        let model = Model::open_with_ctx(path, device, cap).map_err(engine_error)?;
        let controls = Controls::of(&model.tokenizer);
        Ok((model.clone(), Self { model, controls }))
    }

    fn probe(path: &Path) -> Result<(usize, usize), EngineError> {
        let l = gemma_engine::model::probe(path).map_err(engine_error)?;
        Ok((l.context_length, l.kv_bytes_per_token))
    }

    fn kv_bytes_per_token(model: &Model) -> usize {
        model.kv_bytes_per_token()
    }

    fn model_name(&self) -> String {
        self.model.name.clone()
    }

    fn bos(&self) -> Option<u32> {
        self.model.tokenizer.bos()
    }

    fn render(
        &self,
        kind: Kind,
        text: &str,
        prev: Option<Kind>,
        think: bool,
        trusted_len: usize,
    ) -> Vec<u32> {
        self.model
            .tokenizer
            .encode_pieces(&template::render(kind, text, prev, think, trusted_len))
    }

    fn generation_prefix(&self, last: Option<Kind>, _think: bool) -> Vec<u32> {
        self.model
            .tokenizer
            .encode_pieces(&template::generation_prefix(last))
    }

    /// A stop token is neither recorded nor evaluated: the next section's
    /// rendering supplies it (`<turn|>` opens a user section,
    /// `<|tool_response>` opens a tool result).
    fn is_stop(&self, id: u32) -> bool {
        let tok = &self.model.tokenizer;
        id == tok.eos()
            || Some(id) == self.controls.turn_close
            || Some(id) == self.controls.resp_open
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.model.tokenizer.encode_plain(text).len()
    }

    fn translate(&self, tr: &mut ThinkTranslator, id: u32, utf8: &mut Utf8Stream) -> String {
        tr.push(&self.model.tokenizer, id, utf8)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        Engine, EngineEvent, GenerationOptions, GenerationStats, Prompt, ThinkMode,
    };
    use crate::nativeengine::to_u32;

    /// The shared [`crate::nativeengine::default_ctx`] as Gemma calls it:
    /// the file's limits.
    fn default_ctx(ram: usize, limits: gemma_engine::model::Limits) -> usize {
        crate::nativeengine::default_ctx(ram, limits.context_length, limits.kv_bytes_per_token)
    }

    const GIB: usize = 1 << 30;
    const E4B: gemma_engine::model::Limits = gemma_engine::model::Limits {
        context_length: 131_072,
        kv_bytes_per_token: 114_688,
    };
    const B12: gemma_engine::model::Limits = gemma_engine::model::Limits {
        context_length: 262_144,
        kv_bytes_per_token: 688_128,
    };

    /// The default grows with the machine: a third of RAM for the KV, in
    /// whole 1024-token steps, capped at what the model was trained for.
    #[test]
    fn the_default_context_is_a_third_of_ram_of_kv() {
        assert_eq!(
            default_ctx(128 * GIB, E4B),
            131_072,
            "E4B gets its full window"
        );
        assert_eq!(default_ctx(128 * GIB, B12), 66_560);
        assert_eq!(default_ctx(16 * GIB, B12), 8_192, "the floor");
    }

    #[test]
    fn unknown_ram_keeps_the_old_default() {
        assert_eq!(default_ctx(0, B12), 32_768);
        assert_eq!(default_ctx(usize::MAX, B12), 32_768);
    }

    #[test]
    fn a_short_model_is_never_opened_past_its_own_window() {
        let tiny = gemma_engine::model::Limits {
            context_length: 4_096,
            kv_bytes_per_token: 1,
        };
        assert_eq!(default_ctx(128 * GIB, tiny), 4_096);
        assert_eq!(default_ctx(0, tiny), 4_096);
    }
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
        let mut e = GemmaEngine::open(Path::new(&path), 4096, None).unwrap();
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
        let mut e = GemmaEngine::open(Path::new(&path), 4096, None).unwrap();
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

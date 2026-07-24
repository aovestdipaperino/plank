// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Inference engine abstraction.
//!
//! The C agent calls directly into the ds4 engine. Plank keeps that surface
//! behind a narrow trait so the UX layer works against any backend; a stub
//! echo engine makes the agent runnable end-to-end without a model.

use std::fmt::Debug;

/// Reasoning mode requested for a generation, mirroring `ds4_think_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkMode {
    /// Model decides whether to emit a thinking block.
    #[default]
    Auto,
    /// Force a thinking block.
    On,
    /// Suppress thinking.
    Off,
}

/// Sampling and length options for one generation pass.
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    /// Maximum tokens to generate; negative means unlimited.
    pub n_predict: i32,
    /// Context window size in tokens.
    pub ctx_size: i32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Minimum-probability sampling threshold.
    pub min_p: f32,
    /// RNG seed.
    pub seed: u64,
    /// Reasoning mode.
    pub think_mode: ThinkMode,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            n_predict: -1,
            ctx_size: 0,
            temperature: 0.6,
            top_p: 0.95,
            min_p: 0.0,
            seed: 0,
            think_mode: ThinkMode::Auto,
        }
    }
}

/// Progress reported by the engine while prefilling a prompt.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillProgress {
    /// Tokens prefilled so far.
    pub done: i32,
    /// Total tokens to prefill.
    pub total: i32,
    /// Prefill throughput in tokens per second.
    pub tps: f64,
}

/// Role of a structured chat message handed to a provider engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    /// System / developer instructions.
    System,
    /// A human turn.
    User,
    /// A model turn.
    Assistant,
    /// A tool observation fed back to the model.
    Tool,
}

/// A tool call reconstructed from an assistant turn, carrying the synthetic
/// provider-native id that pairs it to its later tool-result message (§4.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallRef {
    /// Provider tool-call id (`tool_call_id` for `OpenAI`, `tool_use.id` for
    /// Anthropic). Threaded through so multi-turn tool conversations are
    /// well-formed per each API's schema.
    pub id: String,
    /// Tool name as chosen by the model.
    pub name: String,
    /// Arguments as a compact JSON **object** string (never a bare scalar).
    pub arguments: String,
}

/// One structured message for a provider engine (§4.4).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Speaker role.
    pub role: ChatRole,
    /// Message text.
    pub content: String,
    /// For [`ChatRole::Tool`] messages: the provider tool-call id being
    /// answered, when one is available.
    pub tool_call_id: Option<String>,
    /// For [`ChatRole::Assistant`] messages: the tool calls this turn issued,
    /// each with the id its matching tool-result message echoes. Empty for
    /// turns that made no tool call.
    pub tool_calls: Vec<ToolCallRef>,
}

impl ChatMessage {
    /// Convenience constructor with no tool-call id and no tool calls.
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }
}

/// A machine-readable tool definition for a provider engine (§4.3/§4.4).
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Tool name (matches plank's dispatch table).
    pub name: String,
    /// Human-readable description sent to the provider.
    pub description: String,
    /// JSON Schema (an object schema) for the tool parameters.
    pub parameters: serde_json::Value,
}

/// Structured turn input for provider engines that set
/// [`Engine::wants_structured`]. Borrows the caller's owned buffers.
#[derive(Debug, Clone, Copy)]
pub struct StructuredTurn<'a> {
    /// The provider system prompt (never the DS4 byte-parity prompt, §4.4).
    pub system: &'a str,
    /// Conversation messages in order.
    pub messages: &'a [ChatMessage],
    /// Tool registry offered to the provider.
    pub tools: &'a [ToolSpec],
    /// The flat rendered transcript, as a fallback for engines that ignore
    /// structure (keeps [`Prompt::flat`] total).
    pub rendered: &'a str,
}

/// Engine input, widened for provider backends (design §4.4).
///
/// Local engines ([`EchoEngine`], the ds4 engine, the remote ds4 client) only
/// ever read [`Prompt::Flat`] — the exact `render_transcript` bytes, preserving
/// byte parity. Provider engines read [`Prompt::Structured`].
#[derive(Debug, Clone, Copy)]
pub enum Prompt<'a> {
    /// The flattened transcript text, as historically passed to `generate`.
    Flat(&'a str),
    /// Structured messages + tool registry for a provider backend.
    Structured(&'a StructuredTurn<'a>),
}

impl<'a> Prompt<'a> {
    /// The flat transcript bytes, regardless of variant. For a structured turn
    /// this is the pre-rendered fallback string.
    #[must_use]
    pub fn flat(&self) -> &'a str {
        match self {
            Prompt::Flat(s) => s,
            Prompt::Structured(t) => t.rendered,
        }
    }
}

/// Events streamed by [`Engine::generate`].
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Prefill progress update.
    Prefill(PrefillProgress),
    /// A piece of generated text (may split UTF-8 across pieces).
    Text(String),
    /// A human-facing note the front-end should surface alongside progress
    /// (e.g. why the system-prompt cache is being rebuilt). May be multi-line.
    Notice(String),
}

/// Per-pass token accounting reported by an online provider. Local engines do
/// not populate this (there is no billed usage to report); providers fill it
/// from the API's `usage` block so the agent can tally `/usage` across a session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Prompt tokens billed this pass (for Anthropic, the *uncached* remainder;
    /// the cache figures below are reported separately).
    pub input_tokens: i32,
    /// Completion tokens generated this pass.
    pub output_tokens: i32,
    /// Prompt tokens served from the provider's cache this pass (0 when the
    /// provider does not report caching, e.g. OpenAI-compatible gateways).
    pub cache_read_tokens: i32,
    /// Prompt tokens written to the provider's cache this pass (0 when none).
    pub cache_write_tokens: i32,
}

impl TokenUsage {
    /// Accumulates another pass's usage into this running total.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(other.cache_read_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(other.cache_write_tokens);
    }
}

/// Outcome of a generation pass.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Number of tokens generated.
    pub generated: i32,
    /// Generation throughput in tokens per second.
    pub tps: f64,
    /// Context tokens in use after the pass.
    pub ctx_used: i32,
    /// True when generation stopped because of an interrupt.
    pub interrupted: bool,
    /// Billed token usage for this pass, when the engine is an online provider.
    pub usage: Option<TokenUsage>,
}

/// Engine error with a human-readable message.
#[derive(Debug)]
pub struct EngineError {
    message: String,
    /// True when the backend does not implement the requested operation, so
    /// the caller can fall back rather than treat it as a hard failure.
    unsupported: bool,
}

impl EngineError {
    /// Creates an error from any message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            unsupported: false,
        }
    }

    /// Marks an operation the engine does not implement (e.g. an engine
    /// without [`Engine::generate_aside`]); callers fall back instead of
    /// surfacing it as a failure.
    #[must_use]
    pub fn unsupported() -> Self {
        Self {
            message: "operation not supported by this engine".to_string(),
            unsupported: true,
        }
    }

    /// Whether this error signals an unimplemented operation.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        self.unsupported
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for EngineError {}

/// Narrow inference surface the agent runs against.
///
/// The transcript is plain text with the chat template already applied by the
/// caller; the engine streams events and returns final stats. `interrupt`
/// is polled between tokens so Ctrl-C can stop a generation promptly.
pub trait Engine: Debug + Send {
    /// Runs one generation pass over `transcript`, streaming events.
    ///
    /// `greedy` is polled before each token sample; while it returns true the
    /// engine samples argmax (temperature 0) regardless of `opts`, mirroring
    /// the C's `worker_sample_with_mode`. The caller derives it from the
    /// streaming parser state so tool-call stanzas are sampled deterministically.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend fails.
    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError>;

    /// Whether this engine wants a [`Prompt::Structured`] input (a provider
    /// backend) rather than the flat rendered transcript. Local engines return
    /// `false`, so the agent keeps passing `Prompt::Flat` and byte parity holds.
    fn wants_structured(&self) -> bool {
        false
    }

    /// Answers a one-shot, tool-free prompt without disturbing the live
    /// generation state, then restores it exactly. Returns the aside's stats;
    /// its text is streamed via `on_event` as [`EngineEvent::Text`].
    ///
    /// Intended for a mid-generation `/btw` aside: the engine snapshots the
    /// frozen main-task KV, answers `prompt` destructively on the same
    /// session (greedy off, tool-call stanzas ignored by the caller), then
    /// restores the snapshot so the main task resumes with zero re-prefill.
    /// Restore is unconditional — an interrupted or failed aside still leaves
    /// the main session valid.
    ///
    /// # Errors
    /// The default implementation returns [`EngineError::unsupported`] so
    /// [`EchoEngine`] and remote engines need no change; callers detect it and
    /// fall back to the boundary-scheduled queue. Real engines return
    /// [`EngineError`] on a backend failure.
    fn generate_aside(
        &mut self,
        _prompt: &str,
        _opts: &GenerationOptions,
        _interrupt: &dyn Fn() -> bool,
        _on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        Err(EngineError::unsupported())
    }

    /// Whether [`generate_aside`](Self::generate_aside) is really implemented
    /// (vs. the default `unsupported` stub). The worker checks this before a
    /// mid-generation `/btw` suspend so it can fall back to the boundary queue
    /// synchronously, without a throwaway aside call. Default `false`.
    fn supports_aside(&self) -> bool {
        false
    }

    /// Approximate token count of `text` for context accounting.
    fn count_tokens(&self, text: &str) -> i32 {
        // ~4 bytes per token is the usual rough estimate.
        i32::try_from(text.len() / 4).unwrap_or(i32::MAX)
    }

    /// Warms the KV cache with the system prompt before the first turn.
    ///
    /// Restores a disk checkpoint at `checkpoint` when its stored fingerprint
    /// still matches this model and system prompt; otherwise prefills the
    /// system prompt (streaming progress via `on_event`) and saves a fresh
    /// checkpoint. Returns `true` when a prefill happened (cache miss).
    ///
    /// The default implementation is a no-op returning `false`.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend fails to prefill.
    fn warm_system_prompt(
        &mut self,
        _system: &str,
        _checkpoint: Option<&std::path::Path>,
        _on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Captures the live session KV as serialized bytes for a checkpoint.
    ///
    /// Returns `None` when the engine has no snapshot support (the stub echo
    /// engine) or has no live session yet; callers then fall back to a
    /// transcript-only checkpoint that re-prefills on rollback.
    fn snapshot_kv(&mut self) -> Option<Vec<u8>> {
        None
    }

    /// Restores session KV previously captured by [`Engine::snapshot_kv`],
    /// so the next turn resumes with (near-)zero re-prefill.
    ///
    /// The default implementation reports lack of support; the echo engine and
    /// any transcript-only rollback rely on it returning an error rather than
    /// pretending to restore.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the engine cannot restore KV state.
    fn restore_kv(&mut self, _bytes: &[u8]) -> Result<(), EngineError> {
        Err(EngineError::new("engine does not support KV snapshots"))
    }

    /// Context window size in tokens.
    fn ctx_size(&self) -> i32;

    /// Human-readable model name for status displays; empty when unknown.
    fn model_name(&self) -> String {
        String::new()
    }
}

/// Incremental UTF-8 decoder for byte-level token streams.
///
/// Byte-level BPE tokenizers split multi-byte characters (emoji, CJK) across
/// tokens; decoding each token independently mangles them into replacement
/// characters. [`push`](Self::push) emits only the complete prefix and carries
/// an unfinished trailing sequence (at most 3 bytes) into the next call;
/// [`flush`](Self::flush) drains whatever remains — lossily — at end of stream.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    carry: Vec<u8>,
}

impl Utf8Stream {
    /// Appends `bytes` and returns the decoded complete prefix.
    ///
    /// Genuinely invalid sequences decode to U+FFFD; only a *possibly
    /// incomplete* trailing sequence is withheld for the next call.
    pub fn push(&mut self, bytes: impl AsRef<[u8]>) -> String {
        self.carry.extend_from_slice(bytes.as_ref());
        let keep = Self::incomplete_tail_len(&self.carry);
        let split = self.carry.len() - keep;
        let out = String::from_utf8_lossy(&self.carry[..split]).into_owned();
        self.carry.drain(..split);
        out
    }

    /// Decodes any carried bytes lossily and resets the stream.
    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.carry).into_owned();
        self.carry.clear();
        out
    }

    /// Length of a trailing byte run that could still become a valid UTF-8
    /// sequence once more bytes arrive; 0 when the tail is complete or
    /// already irrecoverably invalid.
    fn incomplete_tail_len(bytes: &[u8]) -> usize {
        // A lead byte sits at most 3 bytes from the end of an incomplete
        // sequence (a 4-byte sequence missing its last byte).
        let scan = bytes.len().min(3);
        for dist in 1..=scan {
            let b = bytes[bytes.len() - dist];
            if b & 0xC0 == 0x80 {
                continue; // continuation byte — keep looking for the lead
            }
            let expected = match b {
                0xC0..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF7 => 4,
                _ => 1, // ASCII or invalid lead: nothing to wait for
            };
            return if expected > dist { dist } else { 0 };
        }
        0
    }
}

/// Stub engine that echoes a canned reply; keeps the agent runnable without a model.
#[derive(Debug, Default)]
pub struct EchoEngine {
    ctx_size: i32,
}

impl EchoEngine {
    /// Creates an echo engine with the given context size.
    #[must_use]
    pub fn new(ctx_size: i32) -> Self {
        Self { ctx_size }
    }
}

impl Engine for EchoEngine {
    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        _opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        _greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let transcript = prompt.flat();
        // Simulate a short prefill so the live progress bar is exercised even
        // without a real model.
        let total = self.count_tokens(transcript).max(1);
        for step in 1..=8 {
            if interrupt() {
                return Ok(GenerationStats {
                    interrupted: true,
                    ..GenerationStats::default()
                });
            }
            on_event(EngineEvent::Prefill(PrefillProgress {
                done: total * step / 8,
                total,
                tps: 0.0,
            }));
        }
        // The 🦀 straddles the 8-byte chunk boundary below, keeping the
        // stub honest about split multi-byte characters.
        let reply = format!(
            "(echo engine 🦀) no model loaded; transcript is {} bytes\n",
            transcript.len()
        );
        // Chunk at byte boundaries like a byte-level tokenizer would, carrying
        // split multi-byte characters across chunks via `Utf8Stream`.
        let mut utf8 = Utf8Stream::default();
        for piece in reply.as_bytes().chunks(8) {
            if interrupt() {
                return Ok(GenerationStats {
                    interrupted: true,
                    ..GenerationStats::default()
                });
            }
            let text = utf8.push(piece);
            if !text.is_empty() {
                on_event(EngineEvent::Text(text));
            }
        }
        let tail = utf8.flush();
        if !tail.is_empty() {
            on_event(EngineEvent::Text(tail));
        }
        Ok(GenerationStats {
            generated: self.count_tokens(&reply),
            tps: 0.0,
            ctx_used: self.count_tokens(transcript),
            interrupted: false,
            usage: None,
        })
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }
}

#[cfg(test)]
mod tests {
    use super::{EchoEngine, Engine, EngineError, EngineEvent, GenerationOptions, Utf8Stream};

    // Feeds a 🦀 (4 UTF-8 bytes) split the way a byte-level tokenizer emits
    // it: each fragment alone is invalid UTF-8 and must be carried, not
    // lossy-decoded into replacement chars (the "???" bug).
    #[test]
    fn utf8_stream_reassembles_split_emoji() {
        let crab = "🦀".as_bytes(); // F0 9F A6 80
        for split in 1..crab.len() {
            let mut s = Utf8Stream::default();
            let first = s.push(&crab[..split]);
            let second = s.push(&crab[split..]);
            assert_eq!(format!("{first}{second}"), "🦀", "split at {split}");
            assert_eq!(s.flush(), "");
        }
    }

    #[test]
    fn utf8_stream_passes_ascii_through() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(b"hello "), "hello ");
        assert_eq!(s.push("🦀!".as_bytes()), "🦀!");
        assert_eq!(s.flush(), "");
    }

    // Genuinely invalid bytes must not stall the stream waiting for a
    // continuation that never comes.
    #[test]
    fn utf8_stream_lossy_on_invalid_bytes() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push([0x80, 0x80]), "\u{FFFD}\u{FFFD}");
        // A truncated sequence still pending at end of stream flushes lossily.
        assert_eq!(s.push([0xF0, 0x9F]), "");
        assert_eq!(s.flush(), "\u{FFFD}");
    }

    // The echo stub chunks its reply at 8-byte boundaries; emoji spanning a
    // boundary must survive intact in the streamed events.
    #[test]
    fn echo_engine_streams_emoji_intact() {
        let mut engine = EchoEngine::new(4096);
        let mut streamed = String::new();
        engine
            .generate(
                super::Prompt::Flat("[user]\nhi\n"),
                &GenerationOptions::default(),
                &|| false,
                &|| false,
                &mut |e| {
                    if let EngineEvent::Text(t) = e {
                        streamed.push_str(&t);
                    }
                },
            )
            .expect("echo generate");
        assert!(streamed.contains('🦀'), "emoji mangled: {streamed:?}");
        assert!(!streamed.contains('\u{FFFD}'), "lossy bytes: {streamed:?}");
    }

    #[test]
    fn unsupported_error_flag() {
        assert!(EngineError::unsupported().is_unsupported());
        assert!(!EngineError::new("boom").is_unsupported());
    }

    // An engine without a real `generate_aside` (EchoEngine, remote engines)
    // returns `unsupported`, which the worker uses to fall back to the
    // boundary-scheduled queue rather than treating it as a failure.
    #[test]
    fn aside_unsupported_falls_back() {
        let mut engine = EchoEngine::new(4096);
        let transcript = "[user]\nmain task\n".to_string();
        let mut events = Vec::new();
        let err = engine
            .generate_aside(
                "[user]\nbtw question\n",
                &GenerationOptions::default(),
                &|| false,
                &mut |e| events.push(e),
            )
            .expect_err("EchoEngine has no aside support");
        assert!(
            err.is_unsupported(),
            "must signal a fallback, not a failure"
        );
        assert!(events.is_empty(), "the default impl streams nothing");
        // The caller's transcript is untouched — the aside never ran.
        assert_eq!(transcript, "[user]\nmain task\n");
    }
}

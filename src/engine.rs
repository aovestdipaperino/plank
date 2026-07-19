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

/// Events streamed by [`Engine::generate`].
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// Prefill progress update.
    Prefill(PrefillProgress),
    /// A piece of generated text (may split UTF-8 across pieces).
    Text(String),
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
}

/// Engine error with a human-readable message.
#[derive(Debug)]
pub struct EngineError {
    message: String,
}

impl EngineError {
    /// Creates an error from any message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
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
        transcript: &str,
        opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError>;

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

    /// Saves the live session KV as a fingerprinted payload file at `path`.
    ///
    /// The payload is a rebuildable cache of the current KV state; the
    /// caller computes `fingerprint` from the model, system prompt, and
    /// rendered transcript (see `session::payload_fingerprint`) so a later
    /// load can detect staleness. Returns `true` when a payload was written.
    ///
    /// The default implementation is a no-op returning `false` (backends
    /// without KV state, like the echo stub, have nothing to persist).
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend fails to snapshot.
    fn save_session_payload(
        &mut self,
        _path: &std::path::Path,
        _fingerprint: &str,
    ) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Restores the live session KV from a payload written by
    /// [`Engine::save_session_payload`].
    ///
    /// Loads only when the file exists and its stored fingerprint matches
    /// `fingerprint` exactly; anything else returns `false` and the caller
    /// falls back to re-prefilling the transcript — a stale payload is
    /// rebuilt, never trusted. Returns `true` when the KV was restored.
    ///
    /// The default implementation is a no-op returning `false`.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend fails while restoring.
    fn load_session_payload(
        &mut self,
        _path: &std::path::Path,
        _fingerprint: &str,
    ) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Context window size in tokens.
    fn ctx_size(&self) -> i32;

    /// Human-readable model name for status displays; empty when unknown.
    fn model_name(&self) -> String {
        String::new()
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
        transcript: &str,
        _opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        _greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
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
        let reply = format!(
            "(echo engine) no model loaded; transcript is {} bytes\n",
            transcript.len()
        );
        for piece in reply.as_bytes().chunks(8) {
            if interrupt() {
                return Ok(GenerationStats {
                    interrupted: true,
                    ..GenerationStats::default()
                });
            }
            on_event(EngineEvent::Text(
                String::from_utf8_lossy(piece).into_owned(),
            ));
        }
        Ok(GenerationStats {
            generated: self.count_tokens(&reply),
            tps: 0.0,
            ctx_used: self.count_tokens(transcript),
            interrupted: false,
        })
    }

    fn ctx_size(&self) -> i32 {
        self.ctx_size
    }
}

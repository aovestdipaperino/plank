// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Inference engine abstraction.
//!
//! The C agent calls directly into the ds4 engine. Plank keeps that surface
//! behind a narrow trait so the UX layer works against any backend; a stub
//! echo engine makes the agent runnable end-to-end without a model.

use std::fmt::Debug;

/// Reasoning level requested for a generation, mirroring `ds4_think_mode`.
///
/// The engine has exactly three states — the C's `DS4_THINK_NONE` / `_HIGH` /
/// `_MAX` — and plank long exposed only the first two (an `Auto` and an `On`
/// that both mapped to `HIGH`). Today's levels are richer than the engine's
/// because the top and bottom of the range are *prompt* levels, not engine
/// levels: `Max` and `Low` are both `HIGH` at the FFI boundary, distinguished
/// by an effort preamble prepended ahead of the system prompt (see
/// [`effort_prefix`]).
///
/// `Low` is a plank extension with no C counterpart — [`THINK_LOW_PREFIX`] is
/// text plank invented, unlike [`THINK_MAX_PREFIX`], which `tests/c_parity.rs`
/// holds byte-equal to the C. Treat its effect as unverified: the model has no
/// trained response to it.
///
/// [`effort_prefix`]: ThinkMode::effort_prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkMode {
    /// Suppress thinking: the assistant prefix opens with `</think>`.
    Off,
    /// Ordinary thinking plus the brief-reasoning preamble in
    /// [`THINK_LOW_PREFIX`]. A plank extension, not a C level.
    Low,
    /// Ordinary thinking (the C's `DS4_THINK_HIGH`). The default.
    #[default]
    Medium,
    /// Ordinary thinking plus the reasoning-effort preamble, prepended ahead of
    /// the system prompt. Needs a context of at least [`THINK_MAX_MIN_CONTEXT`].
    Max,
}

impl ThinkMode {
    /// Every level, in increasing order of effort. Lets callers and tests
    /// enumerate the levels without restating them (and so without silently
    /// missing one that is added later).
    pub const ALL: [Self; 4] = [Self::Off, Self::Low, Self::Medium, Self::Max];

    /// The level's name, matching the C's `ds4_think_mode_name` for the two
    /// shared levels and naming `Medium` as the user types it to [`parse`].
    ///
    /// [`parse`]: ThinkMode::parse
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::Max => "max",
        }
    }

    /// The level's name abbreviated to a fixed three columns: `off`, `low`,
    /// `med`, `max`.
    ///
    /// For the status footer, where every level must occupy the same width — a
    /// segment that grows and shrinks as the level changes shifts everything to
    /// its right. Prose contexts (`/think`, `/repro`) use [`name`] instead, and
    /// the KV fingerprint keys on [`name`] so this stays a display concern.
    ///
    /// [`name`]: ThinkMode::name
    #[must_use]
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "med",
            Self::Max => "max",
        }
    }

    /// Parses a level typed by the user, case-insensitively. `high` and `none`
    /// are accepted as the C's names for `Medium` and `Off`, and `med` because
    /// that is the spelling the status footer shows.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "no" => Some(Self::Off),
            "low" | "brief" => Some(Self::Low),
            "medium" | "med" | "high" | "on" => Some(Self::Medium),
            "max" | "maximum" => Some(Self::Max),
            _ => None,
        }
    }

    /// Whether the assistant prefix opens a thinking block at this level
    /// (the C's `ds4_think_mode_enabled`).
    #[must_use]
    pub fn thinks(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// The effort preamble this level prepends ahead of the system prompt, or
    /// `None` when it prepends nothing.
    ///
    /// This is the single answer to "does the prompt *prefix* depend on the
    /// level", which is what decides whether a level change invalidates the
    /// token transcript and KV. `Off` and `Medium` differ only in the per-turn
    /// assistant prefix, which is re-derived every turn and never cached, so
    /// both return `None` and moving between them is free.
    #[must_use]
    pub fn effort_prefix(self) -> Option<&'static str> {
        match self {
            Self::Off | Self::Medium => None,
            Self::Low => Some(THINK_LOW_PREFIX),
            Self::Max => Some(THINK_MAX_PREFIX),
        }
    }
}

/// The reasoning-effort preamble `Max` prepends ahead of the system prompt,
/// byte-for-byte the C's `DS4_REASONING_EFFORT_MAX_PREFIX` (`refs/ds4/ds4.c`).
/// Checked against the C source by `tests/c_parity.rs`.
pub const THINK_MAX_PREFIX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to \
     resolve the root cause, rigorously stress-testing your logic against all potential paths, \
     edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, \
     considered alternative, and rejected hypothesis to ensure absolutely no assumption is left \
     unchecked.\n\n",
);

/// The brief-reasoning preamble [`ThinkMode::Low`] prepends ahead of the system
/// prompt, in the same position and by the same mechanism as
/// [`THINK_MAX_PREFIX`].
///
/// **Unlike every other model-facing string in plank, this one has no C
/// counterpart** — it is not in `refs/ds4`, so `tests/c_parity.rs` cannot check
/// it and the model was not trained on it. It is a prompt-level experiment: the
/// engine has no reasoning dial below `DS4_THINK_HIGH`, so asking for brevity in
/// text is the only lever available. Expect the model to sometimes ignore it.
/// If it proves ineffective, delete the level rather than escalating the wording.
pub const THINK_LOW_PREFIX: &str = concat!(
    "Reasoning Effort: Low. Think only as much as this task actually requires.\n",
    "Keep your deliberation short and direct: establish what is being asked, settle on an \
     approach, and act. Do not restate the problem back to yourself, enumerate alternatives you \
     have already rejected, or re-verify steps that are not in doubt.\n",
    "If a problem turns out to be genuinely hard, think for as long as it needs — this is a floor \
     on brevity, not a ceiling on care.\n\n",
);

/// Smallest context [`ThinkMode::Max`] is allowed at, the C's
/// `DS4_THINK_MAX_MIN_CONTEXT`. The model guidance recommends think-max only with at
/// least a 384K-token window; below it the preamble asks for a reasoning budget
/// the context is not meant to hold, so `/think max` is refused rather than
/// silently downgraded (the C downgrades instead — see
/// `ds4_think_mode_for_context`).
pub const THINK_MAX_MIN_CONTEXT: i32 = 393_216;

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
    /// Recover from a tool call the model starts inside an unclosed `<think>`
    /// by force-feeding `</think>` and letting it continue (the C server's
    /// `chat_think_tool_recovery`).
    ///
    /// Off by default because it only makes sense where an in-think stanza is
    /// otherwise wasted: the caller enables it when in-think tool calls are
    /// prohibited, and leaves it off when they are dispatched as-is.
    pub think_tool_recovery: bool,
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
            think_mode: ThinkMode::Medium,
            think_tool_recovery: false,
        }
    }
}

/// Progress reported by the engine while prefilling a prompt.
///
/// Both counts are relative to the cached prefix: they describe *this pass's*
/// work, not the absolute position in the prompt. A turn that reuses 8000
/// cached tokens and prefills 200 new ones reports `total == 200`, so the bar
/// spans `[0, 200]` and matches the throughput figure beside it.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillProgress {
    /// Tokens prefilled so far in this pass (cached prefix excluded).
    pub done: i32,
    /// Total tokens this pass must prefill (cached prefix excluded).
    pub total: i32,
    /// Prefill throughput in tokens per second.
    pub tps: f64,
}

impl PrefillProgress {
    /// Build progress from an *absolute* engine position.
    ///
    /// The ds4 backend reports `cur` as the absolute position within the
    /// prompt — the cached prefix (`base`) is already included (see
    /// `ds4_cli.c:251`, which subtracts it). So `base` is a floor for the bar
    /// and the subtrahend for per-prefill throughput, never an offset to add.
    /// Both reported counts are then rebased to `base`, so the bar spans only
    /// the tokens this pass actually evaluates.
    ///
    /// `total` is taken by mutable reference and stays *absolute* because the
    /// backend can genuinely re-evaluate a few tokens the common-prefix probe
    /// counted as cached; on overshoot the estimated total grows with ~5%
    /// headroom so the bar keeps advancing instead of parking at 100%.
    /// Reaching `total` exactly is a completed prefill, not an overshoot.
    pub fn from_absolute(base: i32, cur: i32, total: &mut i32, elapsed_secs: f64) -> Self {
        let floor = base.max(0).min((*total).max(0));
        let abs_done = cur.max(floor);
        if abs_done > *total {
            *total = abs_done + ((*total) / 20).max(1);
        }
        // Only the tokens actually evaluated in this pass count toward tok/s —
        // and toward the bar.
        let done = abs_done - floor;
        let tps = if elapsed_secs > 0.0 {
            f64::from(done) / elapsed_secs
        } else {
            0.0
        };
        Self {
            done,
            total: *total - floor,
            tps,
        }
    }

    /// The priming event emitted before a turn's sync, reporting how much of
    /// the prompt the live KV already holds.
    ///
    /// `cached` and `total` are absolute prompt counts; the event reports the
    /// remainder, so a partially cached prompt starts at `done == 0` out of
    /// the tokens still to prefill — the bar must not read 100% before any
    /// work has happened. A *fully* cached prompt is the opposite case: there
    /// is no work to do, so it reports an empty (already complete) range.
    /// Clamping that one short instead would freeze the bar at 99.99% for the
    /// whole time-to-first-token with no further event ever arriving to
    /// correct it, which reads as a hung prefill (#64 follow-up).
    #[must_use]
    pub fn primed(cached: i32, total: i32) -> Self {
        let total = total.max(0);
        let remaining = (total - cached.clamp(0, total)).max(0);
        Self {
            done: 0,
            total: remaining,
            tps: 0.0,
        }
    }

    /// Whether this progress reports a finished prefill — i.e. the engine is
    /// now sampling, not prefilling. Drives the status line so a fully cached
    /// turn does not claim to be "prefilling" while it waits for the first
    /// token.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.done >= self.total
    }
}

/// How many prompt tokens the engine will *actually* reuse, given the matching
/// prefix length `common` and the live checkpoint length `pos`.
///
/// `ds4_session_common_prefix` answers "how many leading tokens match?", which
/// is **not** the same as "how many will be kept". `ds4_session_sync` reuses the
/// live KV only when the prompt *extends* the live end
/// (`prompt_len >= pos && starts_with(prompt, checkpoint)`); anything else takes
/// the reset branch and re-prefills from zero. The C is explicit about why:
/// "Extending exactly at the live end is safe; rewriting behind it is not an
/// in-place operation" — the backend still holds raw SWA rows, compressed KV
/// rows, indexer rows, and compressor frontiers for the old suffix, and a token
/// count cannot roll those back.
///
/// The case that matters is a prompt that is a strict *prefix* of the live
/// checkpoint, which `/new` and `/clear` produce: a fresh session's rendered
/// transcript is a prefix of the one it replaced. There `common` equals the full
/// prompt length while the engine silently rebuilds everything, so reporting
/// `common` primes the progress bar as complete and the ensuing multi-thousand
/// token prefill runs with no feedback at all — a hung-looking stall.
///
/// `common == pos` is the whole test: it implies both that every live token
/// matched (so `starts_with` holds) and that `prompt_len >= pos`.
#[must_use]
pub fn reusable_prefix(pos: i32, common: i32) -> i32 {
    if pos > 0 && common == pos { pos } else { 0 }
}

/// Which of two interleaved streams an event belongs to, so a front-end can
/// route the main task and a concurrent aside to different places.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsideStream {
    /// The main task's continuation.
    Main,
    /// The `/btw` aside running beside it on a fork.
    Aside,
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

    /// Sets the reasoning level for every prompt built from now on.
    ///
    /// [`ThinkMode::Max`] prepends [`THINK_MAX_PREFIX`] ahead of the system
    /// prompt, so changing the level changes the token prefix and every cached
    /// KV prefix below it. An engine that caches tokens must drop them here —
    /// the C does the same via `ds4_session_invalidate` when `/think max`
    /// toggles the prefix in and out of the transcript.
    ///
    /// The level still travels per-generation in
    /// [`GenerationOptions::think_mode`], which drives the assistant prefix;
    /// this call is only about the prefix that sits *above* the transcript.
    /// Engines that build no prompt of their own ignore it.
    fn set_think_mode(&mut self, _mode: ThinkMode) {}

    /// Byte length of the leading span of the system prompt that is trusted
    /// control text (`sysprompt::SplitSystemPrompt::trusted_len`).
    ///
    /// A local engine tokenizes that span as *rendered chat*, so the literal
    /// `｜DSML｜` in the prompt's examples becomes the model's dedicated DSML
    /// vocabulary token instead of a spelled-out BPE sequence — what the model
    /// was trained to read, and to emit. The remainder (MCP schemas, `-sys`
    /// text) stays plain content so it cannot forge control tokens.
    ///
    /// Set once before the first prompt is built, and again whenever the system
    /// prompt is rebuilt. Engines that do not tokenize locally ignore it.
    fn set_trusted_system_prefix(&mut self, _len: usize) {}

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

    /// Answers a one-shot, tool-free prompt on a *forked* session, leaving this
    /// one completely untouched (`docs/SESSION-CLONE-DESIGN.md` §6.1).
    ///
    /// This is the non-destructive tier above [`generate_aside`]. Where that
    /// one answers on the live session and relies on an unconditional restore
    /// to undo the damage, this forks the session first, so the main task's KV,
    /// cursor and transcript are never written to at all — an interrupted or
    /// failed aside cannot corrupt them, because it never touched them.
    ///
    /// The cost is a second full KV for the aside's lifetime. Callers gate it
    /// accordingly; the three tiers (fork, destructive-with-restore, boundary
    /// queue) each degrade cleanly into the next.
    ///
    /// # Errors
    /// The default implementation returns [`EngineError::unsupported`] so
    /// [`EchoEngine`] and remote engines fall through to [`generate_aside`].
    /// Real engines return [`EngineError`] on a backend failure, including a
    /// refusal to allocate the fork.
    fn generate_aside_forked(
        &mut self,
        _prompt: &str,
        _opts: &GenerationOptions,
        _interrupt: &dyn Fn() -> bool,
        _on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        Err(EngineError::unsupported())
    }

    /// Whether [`generate_aside_forked`](Self::generate_aside_forked) is really
    /// implemented. Checked before an aside so the caller can pick the tier
    /// without a throwaway call. Default `false`.
    fn supports_forked_aside(&self) -> bool {
        false
    }

    /// Continues the main task **and** answers an aside at the same time,
    /// interleaving both at token granularity on this thread
    /// (`docs/SESSION-CLONE-DESIGN.md` §6.2).
    ///
    /// The main generation runs on the live session; the aside runs on a fork,
    /// so neither can disturb the other. Events are tagged
    /// [`AsideStream`](crate::engine::AsideStream) so a caller can route the
    /// aside to a side panel while the main task keeps flowing to the main log.
    /// Returns `(main, aside)` stats.
    ///
    /// One Metal queue means this is time-slicing: the main task does not
    /// finish sooner than it would have. What changes is that it does not
    /// *stop* — the alternative is freezing it for the whole aside.
    ///
    /// # Errors
    /// The default implementation returns [`EngineError::unsupported`] so
    /// callers fall back to the freeze/answer/resume path. Real engines return
    /// [`EngineError`] on a backend failure, including a refused fork.
    /// `on_stream_end` fires as soon as one of the two finishes, rather than
    /// when the call returns. An aside is usually done in a slice or two while
    /// the main task still has hundreds of tokens to run; without this its
    /// answer would sit unterminated on screen for the rest of the turn.
    fn generate_multiplexed(
        &mut self,
        _main_prompt: &str,
        _aside_prompt: &str,
        _opts: &GenerationOptions,
        _interrupt: &dyn Fn() -> bool,
        _on_event: &mut dyn FnMut(AsideStream, EngineEvent),
        _on_stream_end: &mut dyn FnMut(AsideStream),
    ) -> Result<(GenerationStats, GenerationStats), EngineError> {
        Err(EngineError::unsupported())
    }

    /// Whether [`generate_multiplexed`](Self::generate_multiplexed) is really
    /// implemented. Default `false`.
    fn supports_multiplexing(&self) -> bool {
        false
    }

    /// Approximate token count of `text` for context accounting.
    fn count_tokens(&self, text: &str) -> i32 {
        // ~4 bytes per token is the usual rough estimate.
        i32::try_from(text.len() / 4).unwrap_or(i32::MAX)
    }

    /// Captures the live session KV, or `None` when the engine has no snapshot
    /// support (the stub echo engine, remote engines) or no live session yet.
    ///
    /// The returned cache carries the token transcript the KV was captured
    /// with, so a later [`set_kv`](Self::set_kv) resumes with the exact token
    /// buffer rather than rebuilding from text and re-prefilling.
    fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
        None
    }

    /// Restores session KV previously captured by [`get_kv`](Self::get_kv).
    ///
    /// # Errors
    /// Returns [`EngineError`] when the engine cannot restore KV state. The
    /// default reports lack of support rather than pretending to restore.
    fn set_kv(&mut self, _cache: &crate::kvcache::KVCache) -> Result<(), EngineError> {
        Err(EngineError::new("engine does not support KV snapshots"))
    }

    /// Begins a warm walk: resets the cumulative warm token buffer to the
    /// system prompt's tokens. No prefill happens yet.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend cannot open a session.
    fn warm_reset(&mut self, _system: &str) -> Result<(), EngineError> {
        Ok(())
    }

    /// Appends `text` to the cumulative warm token buffer as one user message
    /// (nothing when `None`, for the system tier, whose tokens [`warm_reset`]
    /// already placed). Does **not** prefill.
    ///
    /// This must be called for *every* tier of the walk, including tiers whose
    /// KV was already restored from disk and therefore need no sync: the buffer
    /// has to describe the whole restored prefix, or the next sync would hand
    /// the backend a token buffer missing the intermediate tiers and discard
    /// the restored KV.
    ///
    /// Each tier is its own message, so a tier boundary is a clean
    /// chat-template message boundary and a snapshot taken there is a genuinely
    /// reproducible token prefix — never a mid-message split whose tokenization
    /// could shift under BPE merges. No trailing assistant prefix is appended,
    /// so the cached prefix ends exactly at the last tier and the first turn's
    /// common-prefix accounting is unperturbed (#63).
    ///
    /// [`warm_reset`]: Engine::warm_reset
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend cannot tokenize the text.
    fn warm_append(&mut self, _text: Option<&str>) -> Result<(), EngineError> {
        Ok(())
    }

    /// Prefills the session up to the cumulative warm buffer's end. Returns
    /// `true` when a prefill actually ran.
    ///
    /// # Errors
    /// Returns [`EngineError`] when the backend fails to prefill.
    fn warm_sync(&mut self, _on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Context window size in tokens.
    fn ctx_size(&self) -> i32;

    /// Human-readable model name for status displays; empty when unknown.
    fn model_name(&self) -> String {
        String::new()
    }

    /// How many sub-agent sidechains may generate against this engine at once.
    ///
    /// The default of 1 is the honest answer for every KV-backed engine: one
    /// live session means two concurrent sidechains would interleave and
    /// corrupt the shared prefix. A stateless engine — HTTP to a provider — has
    /// no such constraint and overrides this.
    ///
    /// The value is a *capability*, not a budget: the user's
    /// `agents.maxParallel` is minimised against it, so a local engine forces
    /// serial dispatch no matter what the setting says.
    ///
    /// Routing local sidechains through [`crate::host::EngineHost`] sessions
    /// (which already do admission and round-robin scheduling) would let a
    /// KV-backed engine report more than 1; that is the seam for parallel local
    /// sub-agents, which remains future work.
    fn max_parallel(&self) -> usize {
        1
    }

    /// Whether this engine runs the model on this machine's own weights.
    ///
    /// Drives the status bar's blinking brain, so the operator can see *which*
    /// engine is working — the point being a `provider: local` sub-agent under a
    /// remote main agent, where nothing else on screen distinguishes the two.
    /// False by default: a provider or a remote host is somebody else's GPU.
    fn is_local(&self) -> bool {
        false
    }
}

/// Text force-fed to close an unclosed `<think>` before the model retries a
/// tool call. Matches the C server's injection byte for byte.
pub const THINK_RECOVERY_TEXT: &str = "</think>\n\n";

/// Decides when a generation loop should force-close an unclosed `<think>`
/// because the model started a tool call inside it.
///
/// Port of the policy half of `chat_think_tool_recovery`. Waiting for a
/// `</think>` that never comes stalls the turn: the stanza is never scanned as
/// executable and gets dropped at parse time. Rather than rewrite sampled
/// context, recover *forward* — feed `</think>` plus a blank line and let the
/// model continue. Measured on the real model, that position predicts a fresh
/// stanza opening so strongly that the call restarts cleanly on the executable
/// side of the close. Re-emitting the stanza opening as well is
/// counterproductive: with the dangling opening right before the close and a
/// forced copy right after it, the model reads the call as already made and
/// ends the turn. The dangling opening stays harmlessly inside reasoning.
///
/// The caller owns the actual injection (tokenizing and evaluating
/// [`THINK_RECOVERY_TEXT`], and checking it has the budget to do so); this type
/// only tracks think state and answers [`should_recover`](Self::should_recover).
#[derive(Debug)]
pub struct ThinkToolRecovery {
    inside: bool,
    /// Offset into the accumulated text where the next scan starts. Held back
    /// by [`crate::dsml::TOOL_START_SCAN_HOLD`] so an opening split across
    /// future tokens is still seen from its first byte.
    scan_from: usize,
}

impl ThinkToolRecovery {
    /// Starts a tracker. `inside` is whether the prompt's assistant prefix
    /// already opened a thinking block.
    #[must_use]
    pub fn new(inside: bool) -> Self {
        Self {
            inside,
            scan_from: 0,
        }
    }

    /// True when the model is currently inside an unclosed `<think>`.
    #[must_use]
    pub fn inside_think(&self) -> bool {
        self.inside
    }

    /// Re-examines the accumulated reply and reports whether to inject now.
    ///
    /// Call after every decoded token with the whole reply so far; detection
    /// works on accumulated text, so the marker's tokenization does not matter.
    /// A lone `<` or a partial marker leaves decoding untouched.
    pub fn should_recover(&mut self, text: &str) -> bool {
        // `</think>` anywhere in the reply ends the block, including one the
        // model closed on its own after we decided to wait.
        if self.inside && text.contains("</think>") {
            self.inside = false;
        } else if !self.inside && text.contains("<think>") {
            // A block reopened after a close (or opened mid-reply under
            // ThinkMode::Auto) counts again.
            self.inside = !text
                .rsplit("<think>")
                .next()
                .unwrap_or("")
                .contains("</think>");
        }
        if !self.inside {
            return false;
        }
        let scan_from = self.scan_from.min(text.len());
        // Never split a UTF-8 character: the markers are multi-byte.
        let scan_from = (0..=scan_from)
            .rev()
            .find(|i| text.is_char_boundary(*i))
            .unwrap_or(0);
        if crate::dsml::find_tool_start(&text[scan_from..]).is_none() {
            self.scan_from = text.len().saturating_sub(crate::dsml::TOOL_START_SCAN_HOLD);
            return false;
        }
        true
    }

    /// Records a completed injection: thinking is closed and the scan resumes
    /// past the marker that triggered it. `text_len` is the length of the
    /// accumulated reply *including* the appended [`THINK_RECOVERY_TEXT`].
    pub fn injected(&mut self, text_len: usize) {
        self.scan_from = text_len;
        self.inside = false;
    }

    /// Records that the caller could not recover here (no context or token
    /// budget). The stream is left as generated for the parse-time fallback;
    /// skipping past the marker stops the scan retrying it every token.
    pub fn skipped(&mut self, text_len: usize) {
        self.scan_from = text_len;
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
    use super::{
        EchoEngine, Engine, EngineError, EngineEvent, GenerationOptions, PrefillProgress,
        THINK_LOW_PREFIX, THINK_MAX_PREFIX, ThinkMode, ThinkToolRecovery, Utf8Stream,
        reusable_prefix,
    };

    // A KV-backed engine holds one live session, so concurrent sidechains on it
    // would interleave and corrupt the shared prefix. 1 is the honest default.
    #[test]
    fn echo_engine_is_serial_by_default() {
        assert_eq!(EchoEngine::new(4096).max_parallel(), 1);
    }

    // The default is ordinary thinking, the level plank has always run at.
    #[test]
    fn think_mode_defaults_to_medium() {
        assert_eq!(ThinkMode::default(), ThinkMode::Medium);
        assert_eq!(GenerationOptions::default().think_mode, ThinkMode::Medium);
    }

    // The footer's segment must not change width with the level.
    #[test]
    fn think_mode_short_names_are_a_fixed_width() {
        let names: Vec<&str> = ThinkMode::ALL.iter().map(|l| l.short_name()).collect();
        assert!(
            names.iter().all(|n| n.chars().count() == 3),
            "{names:?} must all be three columns wide"
        );
        // And each still parses back, since it is what the user sees and copies.
        for level in ThinkMode::ALL {
            assert_eq!(ThinkMode::parse(level.short_name()), Some(level));
        }
    }

    #[test]
    fn think_mode_round_trips_through_its_name() {
        for level in ThinkMode::ALL {
            assert_eq!(ThinkMode::parse(level.name()), Some(level), "{level:?}");
        }
    }

    // The C's own names for the two levels it shares with us, plus the casing
    // and spacing a user actually types.
    #[test]
    fn think_mode_parses_the_c_names_and_sloppy_input() {
        assert_eq!(ThinkMode::parse("none"), Some(ThinkMode::Off));
        assert_eq!(ThinkMode::parse("high"), Some(ThinkMode::Medium));
        assert_eq!(ThinkMode::parse("  MAX "), Some(ThinkMode::Max));
        assert_eq!(ThinkMode::parse("Medium"), Some(ThinkMode::Medium));
        assert_eq!(ThinkMode::parse("more"), None);
        assert_eq!(ThinkMode::parse(""), None);
    }

    // Only `off` suppresses the thinking block; `max` and `low` think like
    // `medium` and differ only by the effort preamble on top.
    #[test]
    fn only_off_suppresses_thinking() {
        assert!(!ThinkMode::Off.thinks());
        assert!(ThinkMode::Low.thinks());
        assert!(ThinkMode::Medium.thinks());
        assert!(ThinkMode::Max.thinks());
    }

    // `low` is spelled the way the user types it, and does not collide with the
    // C's `high`/`none` aliases.
    #[test]
    fn think_mode_parses_low() {
        assert_eq!(ThinkMode::parse("low"), Some(ThinkMode::Low));
        assert_eq!(ThinkMode::parse(" BRIEF "), Some(ThinkMode::Low));
        assert_eq!(ThinkMode::Low.name(), "low");
        assert_eq!(ThinkMode::Low.short_name(), "low");
    }

    // The prefix is what decides KV invalidation, so exactly the two preamble
    // levels must report one, and the two preambles must differ.
    #[test]
    fn only_low_and_max_carry_an_effort_prefix() {
        assert_eq!(ThinkMode::Off.effort_prefix(), None);
        assert_eq!(ThinkMode::Medium.effort_prefix(), None);
        assert_eq!(ThinkMode::Low.effort_prefix(), Some(THINK_LOW_PREFIX));
        assert_eq!(ThinkMode::Max.effort_prefix(), Some(THINK_MAX_PREFIX));
        assert_ne!(THINK_LOW_PREFIX, THINK_MAX_PREFIX);
    }

    // Off↔Medium is free (no prefix moves); every other pair costs a re-prefill.
    // This is the property `set_think_mode` and `/think` both key on.
    #[test]
    fn effort_prefix_identifies_the_free_level_changes() {
        let changed = |a: ThinkMode, b: ThinkMode| a.effort_prefix() != b.effort_prefix();
        assert!(!changed(ThinkMode::Off, ThinkMode::Medium));
        assert!(changed(ThinkMode::Medium, ThinkMode::Low));
        assert!(changed(ThinkMode::Low, ThinkMode::Max));
        assert!(changed(ThinkMode::Off, ThinkMode::Low));
    }

    // The preamble must be plain text with no role wrapper or DSML control
    // string: it is tokenized as rendered chat and sits ahead of the system
    // message, so a stray marker would land outside any role.
    #[test]
    fn think_low_prefix_is_bare_prose() {
        assert!(!THINK_LOW_PREFIX.contains('｜'), "no DSML control markers");
        assert!(!THINK_LOW_PREFIX.contains("<think>"));
        assert!(
            THINK_LOW_PREFIX.ends_with("\n\n"),
            "must separate itself from the system prompt that follows"
        );
        // No indentation leaked in from the source literal's continuations.
        assert!(
            !THINK_LOW_PREFIX.lines().any(|l| l.starts_with(' ')),
            "leading whitespace in model-facing text: {THINK_LOW_PREFIX:?}"
        );
    }

    // The trigger: a complete stanza opening while thinking is still open.
    #[test]
    fn think_recovery_fires_on_a_stanza_opened_inside_think() {
        let mut r = ThinkToolRecovery::new(true);
        assert!(!r.should_recover("let me check the file"));
        assert!(r.should_recover(
            "let me check the file<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">"
        ));
    }

    // A partial marker must not force a close: the C explicitly keeps decoding
    // until the opening is complete.
    #[test]
    fn think_recovery_ignores_partial_markers() {
        let mut r = ThinkToolRecovery::new(true);
        assert!(!r.should_recover("thinking <"));
        assert!(!r.should_recover("thinking <｜DSML｜tool_call"));
        assert!(!r.should_recover("thinking <｜DSML｜invoke name=\"read\">"));
    }

    // Detection is on accumulated text, so an opening split across tokens is
    // still seen — the scan window is held back past the longest opening.
    #[test]
    fn think_recovery_sees_an_opening_split_across_tokens() {
        let mut r = ThinkToolRecovery::new(true);
        let mut text = String::new();
        let full = format!("{}<｜DSML｜tool_calls>", "reasoning ".repeat(40));
        let mut fired = false;
        for c in full.chars() {
            text.push(c);
            if r.should_recover(&text) {
                fired = true;
                break;
            }
        }
        assert!(fired, "the opening must be detected across many pushes");
        assert_eq!(
            text, full,
            "it must fire exactly when the opening completes"
        );
    }

    // Outside thinking there is nothing to recover: a normal stanza is already
    // on the executable side.
    #[test]
    fn think_recovery_is_inert_outside_think() {
        let mut r = ThinkToolRecovery::new(false);
        assert!(!r.should_recover("<｜DSML｜tool_calls>"));

        let mut closed = ThinkToolRecovery::new(true);
        assert!(!closed.should_recover("done</think>"));
        assert!(!closed.inside_think());
        assert!(!closed.should_recover("done</think><｜DSML｜tool_calls>"));
    }

    // After an injection thinking is closed, so a stanza that continues past
    // the forced close does not trigger a second one.
    #[test]
    fn think_recovery_does_not_re_fire_after_injecting() {
        let mut r = ThinkToolRecovery::new(true);
        let mut text = "hmm<｜DSML｜tool_calls>".to_owned();
        assert!(r.should_recover(&text));
        text.push_str(super::THINK_RECOVERY_TEXT);
        r.injected(text.len());
        assert!(!r.inside_think());
        text.push_str("<｜DSML｜tool_calls><｜DSML｜invoke name=\"read\">");
        assert!(!r.should_recover(&text));
    }

    // No budget to recover: the marker is skipped rather than retried on every
    // subsequent token, and thinking stays open for the parse-time fallback.
    #[test]
    fn think_recovery_skip_does_not_retry_the_same_marker() {
        let mut r = ThinkToolRecovery::new(true);
        let text = "hmm<｜DSML｜tool_calls>".to_owned();
        assert!(r.should_recover(&text));
        r.skipped(text.len());
        assert!(r.inside_think(), "thinking is still open");
        assert!(!r.should_recover(&text), "the same marker must not re-fire");
    }

    // The markers are multi-byte, so the held-back scan window must never be
    // cut mid-character.
    #[test]
    fn think_recovery_scan_window_respects_char_boundaries() {
        let mut r = ThinkToolRecovery::new(true);
        let mut text = String::new();
        for _ in 0..200 {
            text.push('｜');
            assert!(!r.should_recover(&text));
        }
    }

    // Cold prefill: nothing cached, so the absolute position is the progress
    // and throughput counts every token.
    #[test]
    fn prefill_progress_cold_tracks_absolute_position() {
        let mut total = 100;
        for cur in 0..=99 {
            let p = PrefillProgress::from_absolute(0, cur, &mut total, 1.0);
            assert_eq!(p.done, cur);
            assert_eq!(p.total, 100);
            assert!((p.tps - f64::from(cur)).abs() < 1e-9);
        }
    }

    // Regression for #74: with a warm KV the reported position already
    // includes the cached prefix, so it must not be added again — `done` stays
    // inside [base, total] and tok/s reflects only this pass.
    #[test]
    fn prefill_progress_warm_does_not_double_count_base() {
        let base = 8000;
        let mut total = 8200;
        for cur in base..=8200 {
            let p = PrefillProgress::from_absolute(base, cur, &mut total, 2.0);
            assert!(p.done >= 0, "done {} below zero", p.done);
            assert!(p.done <= 200, "done {} overshoots total", p.done);
            assert_eq!(p.total, 200, "the bar spans only the new tokens");
            let expected = f64::from(cur - base) / 2.0;
            assert!((p.tps - expected).abs() < 1e-9);
        }
    }

    // A priming callback can report a position below the cached base; the bar
    // must clamp to the floor rather than go backwards or negative.
    #[test]
    fn prefill_progress_clamps_below_base() {
        let mut total = 500;
        let p = PrefillProgress::from_absolute(300, 120, &mut total, 1.0);
        assert_eq!(p.done, 0);
        assert!((p.tps - 0.0).abs() < 1e-9);
        assert_eq!(p.total, 200);
    }

    // Genuine overshoot (the backend re-evaluates tokens the common-prefix
    // probe counted as cached) still grows the estimated total with headroom.
    #[test]
    fn prefill_progress_grows_total_on_overshoot() {
        let mut total = 100;
        let p = PrefillProgress::from_absolute(0, 100, &mut total, 1.0);
        assert_eq!(p.done, 100);
        assert_eq!(p.total, 100, "reaching total exactly is completion");
        let p = PrefillProgress::from_absolute(0, 101, &mut total, 1.0);
        assert_eq!(p.done, 101);
        assert_eq!(p.total, 106);
        assert_eq!(total, 106);
    }

    /// Regression (#64 follow-up): with the tier cache working, a turn whose
    /// prompt is entirely in KV is the *normal* case, not a rare one. The
    /// priming event is then the only prefill event of the turn — nothing
    /// follows it, because there is nothing to prefill. Reporting it one token
    /// short leaves the status bar parked at 99.99% for the whole
    /// time-to-first-token, which is indistinguishable from a hang.
    // A prompt that is a strict PREFIX of the live checkpoint — what `/new` and
    // `/clear` produce — reuses NOTHING: `ds4_session_sync` cannot rewrite
    // behind the live end, so it resets and re-prefills. Reporting the matching
    // prefix here is what made a ~20s rebuild look like "100% reused" with a
    // progress bar primed as complete.
    #[test]
    fn a_prompt_shorter_than_the_live_kv_reuses_nothing() {
        // /new: 2509-token fresh prompt, 2760 tokens live, all 2509 match.
        assert_eq!(
            reusable_prefix(2760, 2509),
            0,
            "a strict prefix of the live KV is rebuilt from zero, not reused"
        );
        // The bar must therefore NOT prime as complete for that turn.
        assert!(!PrefillProgress::primed(reusable_prefix(2760, 2509), 2509).is_complete());
    }

    #[test]
    fn a_prompt_extending_the_live_kv_reuses_the_whole_checkpoint() {
        // The normal turn: live KV is 2810, prompt is 2818, all 2810 match.
        assert_eq!(reusable_prefix(2810, 2810), 2810);
        // A divergence behind the live end also rebuilds: 2653 of 2760 matched.
        assert_eq!(
            reusable_prefix(2760, 2653),
            0,
            "diverging before the live end forces a full rebuild"
        );
        // No live checkpoint at all.
        assert_eq!(reusable_prefix(0, 0), 0);
    }

    #[test]
    fn a_fully_cached_prompt_primes_as_complete() {
        let p = PrefillProgress::primed(13121, 13121);
        assert_eq!(p.done, 0, "nothing to prefill: report it finished");
        assert_eq!(p.total, 0);
        assert!(p.is_complete());

        // Over-cached (the KV holds more than this prompt) is still complete.
        assert!(PrefillProgress::primed(20000, 13121).is_complete());
    }

    #[test]
    fn a_partially_cached_prompt_primes_short_of_complete() {
        let p = PrefillProgress::primed(13110, 13121);
        assert_eq!(p.done, 0);
        assert_eq!(p.total, 11, "only the uncached remainder is on the bar");
        assert!(
            !p.is_complete(),
            "real work remains; the bar must not read 100%"
        );

        // A cold prompt primes at zero and is only complete if there is
        // genuinely nothing to do.
        let cold = PrefillProgress::primed(0, 13121);
        assert_eq!(cold.done, 0);
        assert_eq!(cold.total, 13121);
        assert!(!cold.is_complete());
        assert!(PrefillProgress::primed(0, 0).is_complete());
    }

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

    // The tier-warm capability is opt-in: engines without a KV to prefill
    // (EchoEngine, remote/provider) must inherit the no-op default so both
    // front-end warm paths degrade cleanly (issues #63, #64).
    #[test]
    fn echo_engine_reports_no_kv_support_and_warms_as_a_no_op() {
        let mut e = EchoEngine::new(4096);
        assert!(e.get_kv().is_none(), "the stub engine has no KV to capture");
        assert!(e.set_kv(&crate::kvcache::KVCache::default()).is_err());
        // Warming must still run end-to-end against the stub: no prefill, no error.
        e.warm_reset("SYSTEM").unwrap();
        e.warm_append(Some("tier text")).unwrap();
        assert!(!e.warm_sync(&mut |_| {}).unwrap());
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

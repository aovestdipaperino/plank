// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Persistent user preferences (`settings.json`).
//!
//! Holds the settings that are *stable preferences* rather than per-run
//! choices: engine defaults, UI tuning, safety defaults, and the MCP handshake
//! timeout. Operational flags (`--prompt`, `--ui console`, `--ui-remote`,
//! `--trace`, `--chdir`, `--seed`, and the serve/control options) describe one
//! invocation and deliberately have no settings key.
//!
//! Files are read from `~/.plank/settings.json` then `<cwd>/.plank/settings.json`,
//! the later file winning key by key. The full precedence chain is:
//!
//! ```text
//! built-in defaults < ~/.plank/settings.json < ./.plank/settings.json < env < CLI flags
//! ```
//!
//! A missing file, unreadable file, malformed JSON, or a value of the wrong
//! type all fall back to the default for that key: a broken settings file
//! degrades plank's preferences, never its ability to start.
//!
//! Secrets are excluded by design. `./.plank/settings.json` lives inside the
//! working tree and is easy to commit by accident, so the provider API key
//! stays on the environment and the command line.
//!
//! ```json
//! {
//!   "engine": { "model": "~/models/ds4.gguf", "threads": 8, "backend": "metal",
//!               "power": 80, "ctx": 262144, "thinkingToolCalls": false },
//!   "ui":     { "respectGitignore": true, "popupRows": 15, "indexRefreshSecs": 5,
//!               "historySize": 512, "showToolCalls": false, "showToolResults": false,
//!               "showThinking": true, "crtOff": true, "easterEggs": true,
//!               "screensaver": "1m", "screensaverFace": "matrix" },
//!   "safety": { "sandbox": true, "btwSuspend": false },
//!   "mcp":    { "timeoutSecs": 30 },
//!   "ask":    { "maxOptions": 7 },
//!   "agents": { "autoRoute": true, "maxParallel": 4 },
//!   "git":    { "signCommits": true },
//!   "context": { "microcompact": true, "shortReminder": true },
//!   "memory": { "autoExtract": true, "extractEveryNTurns": 1 }
//! }
//! ```
//!
//! ## Live vs. restart-bound
//!
//! Most fields here are read fresh at the point of use (`crate::settings::active()`
//! has no caching layer), so a `/config` save that goes through [`reinstall`]
//! takes effect on the very next read — no restart. But a setting captured once
//! into some other long-lived value at startup stays stale until something
//! re-pushes it; `install`/`reinstall` are the chosen choke point for that (see
//! `ui.reducedMotion` → `crate::anim`, `ui.notifications` → `crate::notify`), and
//! `crate::complete`/`crate::editor::History::live` show the alternative of simply
//! reading `active()` again at the moment it matters (`ui.respectGitignore`,
//! `ui.historySize`) instead of threading a push channel through.
//!
//! A few fields are restart-bound **by design** and are not worth chasing:
//!
//! - `engine.*` (`model`, `backend`, `threads`, `ctx`, `power`) — the `Engine`
//!   is constructed once at startup from these; swapping it live would mean
//!   tearing down and rebuilding the whole inference stack mid-session.
//! - `safety.sandbox`, `safety.btwSuspend` — copied into `AgentConfig` once at
//!   startup.
//! - `tools.recall`, `tools.fanout`, `tools.runCode`, `tools.bashNotify`,
//!   `tools.remember`, `git.signCommits` — these
//!   feed the system prompt text, which is built once per session and then
//!   KV-cached (see `docs/KV-CACHE.md`); applying a change live would silently
//!   invalidate a cache the model's prefill is relying on to be exactly what it
//!   was before.
//!
//! Do not try to make the settings above live — a restart (or, in the KV case,
//! a fresh session) is the honest answer for them.
//!
//! `ui.historySize` sits in between: `History::live()` re-reads it on every
//! trim rather than capturing it, so growing the cap live works and shrinking
//! it evicts down to the new size on the next entry added — but it does not
//! retroactively resize a history that already holds more than the new cap
//! until that next add happens.

use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use crate::tools::mcp::{Json, json_escape, json_parse, json_write};

/// Popup rows offered by `@` completion when unset.
pub const DEFAULT_POPUP_ROWS: usize = 15;
/// Seconds a built file index is trusted before a refresh is allowed.
pub const DEFAULT_INDEX_REFRESH_SECS: u64 = 5;
/// Prompt history entries retained when unset.
pub const DEFAULT_HISTORY_SIZE: usize = 512;
/// Seconds an MCP server has to answer a request when unset.
pub const DEFAULT_MCP_TIMEOUT_SECS: u64 = 30;
/// Most options the `ask` tool accepts when unset.
pub const DEFAULT_ASK_MAX_OPTIONS: usize = 7;
/// Fewest options the `ask` tool ever accepts; a choice needs two arms. Not
/// configurable — a one-option "choice" is a degenerate question.
pub const ASK_MIN_OPTIONS: usize = 2;

/// Engine defaults: the same knobs as `-m`, `-t`, `--backend`, `--power`, `-c`.
///
/// `model` replaces what used to be a hardcoded convention — plank falls back
/// to `~/.plank/ds4flash.gguf` only when neither this key nor `-m` is given.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineSettings {
    /// Model file to load; overridden by `-m`/`--model`.
    pub model: Option<PathBuf>,
    /// Worker thread count; overridden by `-t`/`--threads`.
    pub threads: Option<i32>,
    /// Backend name (`metal`, `cuda`, `cpu`); overridden by `--backend`.
    pub backend: Option<String>,
    /// GPU power cap percent; overridden by `--power`.
    pub power: Option<i32>,
    /// Context window in tokens; overridden by `-c`/`--ctx`.
    pub ctx: Option<i32>,
    /// Whether DSML tool calls the model emits inside `<think></think>` are
    /// dispatched. Default false.
    ///
    /// `false` is strict `refs/ds4` parity: an in-think stanza is discarded
    /// with a `[tool call ignored: ...]` notice, and the tools prompt keeps the
    /// line forbidding such calls. The C agent only ever behaved this way, so
    /// it stays the default; turn it on (`/config engine.thinkingToolCalls
    /// true`) to let the model act from inside its reasoning.
    pub thinking_tool_calls: bool,
}

/// UI behaviour that used to be magic numbers in the source.
// The display toggles (showToolCalls/showToolResults/showThinking) are
// genuinely independent on/off knobs, not a state machine to model as an enum.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiSettings {
    /// Whether `@` completion honours `.gitignore` for untracked files.
    pub respect_gitignore: bool,
    /// Rows the `@` completion popup offers at most.
    pub popup_rows: usize,
    /// Seconds before the file index may be rebuilt.
    pub index_refresh_secs: u64,
    /// Prompt history entries retained.
    pub history_size: usize,
    /// Show the model's tool-call banners (`🛠️ …`). Off by default so the UI
    /// stays uncluttered; the DSML is always parsed regardless.
    pub show_tool_calls: bool,
    /// Echo tool result text (observations) into the scrollback. Off by
    /// default; the model always receives the results either way.
    pub show_tool_results: bool,
    /// Render the model's thinking text (dimmed) in the scrollback. Off by
    /// default; when off, and plank was started with `--debug` (or `/debug
    /// on`), the raw model stream (thinking, answer, tool-call markup) is
    /// instead mirrored to a `tdk` listening on port 7878, if
    /// one is up (see `debugmirror`), so the thinking is not simply lost. When
    /// on, plank never connects to the console at all.
    pub show_thinking: bool,
    /// When native macOS desktop notifications fire at turn lifecycle points
    /// (turn complete/interrupted past the threshold, and awaiting input):
    /// `always`, `unfocused` (only while the terminal window is not focused),
    /// or `never`. `true`/`false` are accepted as legacy spellings of
    /// always/never. Default `always`.
    pub notifications: crate::notify::NotifyMode,
    /// Minimum turn duration, in seconds, before a completed turn notifies.
    /// Awaiting-input notifications ignore this. Default 10.
    pub notify_after_secs: u64,
    /// Play the CRT power-off animation of the final frame on clean TUI
    /// exit. On by default; see issue #54.
    pub crt_off: bool,
    /// Collapse every TUI animation (throbber, shimmer, pulse, flash,
    /// stall-fade) to a static fallback. Off by default; see issue #61.
    pub reduced_motion: bool,
    /// How long the TUI must sit idle before the starfield screensaver comes
    /// up. Default one minute; `never` switches it off. Unlike the arcade
    /// games this is not an easter egg, so `ui.easterEggs` does not gate it.
    pub screensaver: crate::arcade::ScreensaverDelay,
    /// Which ambient screen the screensaver puts up: the matrix rain, the
    /// starfield, the minions, or a fresh draw each time.
    pub screensaver_face: crate::arcade::ScreensaverFace,
    /// A plugin face pinned by name, as `<plugin>:<face>`.
    ///
    /// Kept beside the built-in enum rather than folded into it. The enum is
    /// `Copy` and exhaustively matched in half a dozen places; widening it to
    /// carry a `String` would touch all of them to express something only the
    /// screensaver opener cares about. `None` means the built-in field decides,
    /// which is every session with no screensaver plugin installed.
    pub screensaver_face_plugin: Option<String>,
    /// Whether the arcade easter eggs (`/pelota`, …) exist. On by
    /// default. Turned off they are not merely hidden — they stop being known
    /// commands, so the line goes to the model like any other unrecognized
    /// slash command, which is what a deployment that wants no games at all
    /// needs. See [`crate::arcade`].
    pub easter_eggs: bool,
    /// Use the built-in Ctrl-G editor (`crate::miniedit`) rather than shelling
    /// out to `$EDITOR`. On by default. Ignored in builds without the
    /// `builtin_editor` feature, which always use `$EDITOR`.
    pub builtin_editor: bool,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            popup_rows: DEFAULT_POPUP_ROWS,
            index_refresh_secs: DEFAULT_INDEX_REFRESH_SECS,
            history_size: DEFAULT_HISTORY_SIZE,
            show_tool_calls: false,
            show_tool_results: false,
            show_thinking: false,
            notifications: crate::notify::NotifyMode::Always,
            notify_after_secs: 10,
            crt_off: true,
            reduced_motion: false,
            screensaver: crate::arcade::ScreensaverDelay::default(),
            screensaver_face: crate::arcade::ScreensaverFace::default(),
            screensaver_face_plugin: None,
            easter_eggs: true,
            builtin_editor: true,
        }
    }
}

/// Persisted defaults for the two-sided safety flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SafetySettings {
    /// Default for the bash write sandbox (on where sandbox-exec exists);
    /// overridden by `--sandbox`/`--no-sandbox`.
    pub sandbox: Option<bool>,
    /// Default for `/btw` mid-generation suspend; overridden by
    /// `--btw-suspend`/`--disable-btw-suspend`.
    pub btw_suspend: Option<bool>,
}

/// MCP client tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpSettings {
    /// Seconds an MCP server has to answer before it is considered dead.
    pub timeout_secs: u64,
}

impl Default for McpSettings {
    fn default() -> Self {
        Self {
            timeout_secs: DEFAULT_MCP_TIMEOUT_SECS,
        }
    }
}

/// `ask` tool tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskSettings {
    /// Most options the `ask` tool accepts (the minimum is fixed at 2).
    pub max_options: usize,
}

impl Default for AskSettings {
    fn default() -> Self {
        Self {
            max_options: DEFAULT_ASK_MAX_OPTIONS,
        }
    }
}

/// Startup update-available detection (issue #56).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateSettings {
    /// Whether to check the GitHub Releases API at startup for a newer plank
    /// release. Best-effort, rate-limited to once/day, silent on failure; set
    /// to `false` to disable the check (and its network request) entirely.
    pub check: bool,
}

impl Default for UpdateSettings {
    fn default() -> Self {
        Self { check: true }
    }
}

/// Bounds on the `agent` tool: whether the model may route to configured
/// definitions on its own initiative, and how wide a remote fan-out may get.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSettings {
    /// Whether configured definitions are offered to the model at all. With
    /// this off the `agent` tool still works, but only as a general-purpose
    /// sub-agent — the roster is withheld. `/subagent <name>` is unaffected;
    /// this governs model initiative only.
    pub auto_route: bool,
    /// Ceiling on concurrent sub-agents in one tool-call block. Only reachable
    /// for provider-backed definitions; a KV-backed engine reports
    /// `max_parallel() == 1` and forces serial regardless of this value.
    pub max_parallel: usize,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            auto_route: true,
            max_parallel: 4,
        }
    }
}

/// Git conventions the model is told to follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitSettings {
    /// Whether the system prompt asks the model to sign the commits it makes
    /// with plank's attribution trailer. Set to `false` to leave commit
    /// messages entirely to the model and the repository's own conventions.
    pub sign_commits: bool,
}

impl Default for GitSettings {
    fn default() -> Self {
        Self { sign_commits: true }
    }
}

/// Tool-dispatch tuning: loop guards and call deadlines.
// Flat on/off feature flags; the length is not complexity.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolsSettings {
    /// Whether the repeat-tool advisory is on. Default on; a deliberate
    /// deviation from the C reference, documented in
    /// `docs/SYSTEM-PROMPT-OVERRIDES.md`.
    pub repeat_advisory: bool,
    /// Whether the loop guards run at all. Default on.
    ///
    /// One switch over all three: the repetition guard (byte window and think
    /// budget), the tool-call loop guard (identical-call cycles and the
    /// no-progress budget), and the repeat advisory that rides on it. Turning
    /// it off lets a generation cycle forever, which is occasionally what you
    /// want while diagnosing the cycle itself.
    ///
    /// Read through [`crate::guard::guards_enabled`] at every check rather than
    /// captured at turn start, so `/loopguard` takes effect on the generation
    /// already running.
    pub loop_guards: bool,
    /// Dispatch-level wall-clock deadline in seconds for a single tool call.
    /// `0` (the default) is off — parity is untouched until a user opts in.
    /// Bash keeps its own model-supplied timeout; this is the outer bound.
    pub call_timeout_sec: u64,
    /// A tool result larger than this many bytes is spilled to
    /// `~/.plank/spill/<session-id>/` and replaced inline by a bounded preview
    /// plus a locator. Defaults high enough that ordinary sessions never spill.
    pub spill_max_bytes: usize,
    /// How many bytes of a spilled result stay inline as the preview.
    pub spill_preview_bytes: usize,
    /// Whether the `recall` tool is offered to the model (M8). Default off:
    /// the C agent has no such tool, so advertising it changes the system
    /// prompt and churns the `fp1` fingerprint — a deliberate, versioned
    /// deviation, documented in `docs/SYSTEM-PROMPT-OVERRIDES.md`.
    pub recall: bool,
    /// Whether the `fanout` tool is offered to the model (M9). Default off:
    /// a new model-facing tool that runs independent subtasks and joins their
    /// reports deterministically. On the `ds4_engine` path subtasks are
    /// interleaved on one Metal queue, not parallel — the description promises
    /// a deterministic join, not speed.
    pub fanout: bool,
    /// Whether the `run_code` tool is offered to the model (M10). Default off:
    /// a new model-facing tool that runs a small script of named operations
    /// (read/glob/edit/bash) through the existing tool dispatch path, so the
    /// consent and sandbox checks apply. Advertising it changes the system
    /// prompt and churns the `fp1` fingerprint — a deliberate, versioned
    /// deviation, documented in `docs/SYSTEM-PROMPT-OVERRIDES.md`.
    pub run_code: bool,
    /// Whether a bash job that finishes after the model stopped watching it
    /// wakes the model with a notification (`docs/BACKGROUND-TASKS.md`).
    /// Default off while the feature is in beta: on, it also appends one
    /// sentence to the shell rules, which churns the `fp1` fingerprint.
    pub bash_notify: bool,
    /// Whether the no-progress budget ends a turn: a generation that emits
    /// `NO_PROGRESS_BYTE_BUDGET` bytes without changing a file is stopped and the user told why. Default off — the byte budget is a
    /// heuristic, and a long read-only investigation is a legitimate turn, so
    /// this rung is opt-in even when the other loop guards are on.
    pub no_progress_guard: bool,
    /// Whether the `remember` tool is offered to the model. Advertising it
    /// changes the system prompt and churns the `fp1` fingerprint once, the
    /// same price `recall` already paid. Its writes land on disk and take
    /// effect at the next session start, so no Tier 2 checkpoint is
    /// invalidated mid-session.
    pub remember: bool,
}

impl Default for ToolsSettings {
    fn default() -> Self {
        Self {
            repeat_advisory: true,
            loop_guards: true,
            call_timeout_sec: 0,
            spill_max_bytes: 1_048_576,
            spill_preview_bytes: 4096,
            recall: true,
            fanout: true,
            run_code: true,
            bash_notify: false,
            no_progress_guard: false,
            remember: true,
        }
    }
}

/// Ceiling on `agents.maxParallel`; a higher configured value clamps to this.
pub const AGENT_MAX_PARALLEL: usize = 16;

/// The whole of `settings.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Settings {
    /// Engine defaults.
    pub engine: EngineSettings,
    /// UI behaviour.
    pub ui: UiSettings,
    /// Safety defaults.
    pub safety: SafetySettings,
    /// MCP client tuning.
    pub mcp: McpSettings,
    /// `ask` tool tuning.
    pub ask: AskSettings,
    /// Startup update-available detection.
    pub update: UpdateSettings,
    /// `agent` tool bounds: model routing and fan-out width.
    pub agents: AgentSettings,
    /// Git-worktree isolation tuning.
    pub worktree: WorktreeSettings,
    /// KV-cache retention.
    pub kvcache: KvCacheSettings,
    /// Transcript-shrinking behaviour short of full compaction.
    pub context: ContextSettings,
    /// Git conventions the model is told to follow.
    pub git: GitSettings,
    /// Tool-dispatch tuning: loop guards and call deadlines.
    pub tools: ToolsSettings,
    /// How persistent memory maintains itself.
    pub memory: MemorySettings,
    /// Predictive next-prompt suggestions.
    pub suggestions: SuggestionSettings,
    /// Values set for plugin-declared `config` options, keyed
    /// `<component-id>.<option>`.
    ///
    /// A flat map rather than a nested block because the keys are component
    /// ids the user's plugins happen to have, not a schema plank knows at
    /// compile time. Values are strings for the same reason: a component
    /// declares the type, and `ConfigOption::accepts` validates against that
    /// declaration rather than against anything settings.rs believes.
    pub plugin_config: std::collections::BTreeMap<String, String>,
    /// Provenance of each effective settings key (`section.key`), keyed by the
    /// addressing `/config` and `configform::FIELDS` use. Populated during
    /// overlay; CLI overrides are recorded separately on `AgentConfig`.
    pub provenance: std::collections::BTreeMap<String, crate::provenance::Provenance>,
}

/// `worktree` block: how [`crate::worktree`] builds a new working copy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorktreeSettings {
    /// Cone-mode sparse-checkout paths. Empty (the default) checks out
    /// everything; setting it keeps a worktree of a large repo small.
    pub sparse_paths: Vec<String>,
    /// Directories symlinked from the main checkout rather than duplicated,
    /// e.g. `target` or `node_modules`.
    pub symlink_directories: Vec<String>,
    /// Give each sub-agent its own throwaway worktree, so parallel agents
    /// cannot overwrite each other's edits. Off by default: it costs a checkout
    /// per agent, and the agent's work then has to be merged back.
    pub isolate_agents: bool,
}

/// `context` block: transcript-shrinking behaviour short of full compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextSettings {
    /// Whether micro-compaction (clearing old tool-result bodies in place)
    /// runs at all. On by default. Because it rewrites transcript text in
    /// place, it invalidates the engine's KV cache from that point on, which
    /// a snapshot ladder mitigates but cannot eliminate; turning this off
    /// stops every in-place rewrite, at the cost of relying on full
    /// summary-based compaction alone to reclaim context under pressure.
    pub microcompact: bool,
    /// Whether the pressure-based system-prompt reminder (re-injected once
    /// 50K tokens have passed since the prompt was last seen) is the short
    /// form: the tool-call syntax reminder plus the roster of tool names,
    /// instead of the C's full tools prompt. On by default: the short form is
    /// a few hundred tokens against several thousand, and every one of them
    /// is prefilled once and then occupies context for the rest of the
    /// session. Turning it off restores the C reference behaviour.
    pub short_reminder: bool,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            microcompact: true,
            short_reminder: true,
        }
    }
}

/// `memory` block: how persistent memory maintains itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySettings {
    /// Whether the extraction sidechain runs at all. **On by default** since
    /// 5.1.7, and worth knowing the price of: the pass is not a background job
    /// in any useful sense — it runs
    /// synchronously on the turn thread after the model's final answer, and
    /// each run costs a whole-session KV snapshot, a prefill of the excerpt,
    /// a generation and a KV restore, none of which shows in the turn stats.
    /// On a local Metal model that is seconds of visible stall per answer,
    /// so it is opt-in. Off leaves the `remember` tool and `/remember`
    /// working — only the passive pass stops, and rendering falls back to
    /// plain budgeted display with counters that nothing ever bumps.
    pub auto_extract: bool,
    /// Run the pass every N eligible turns. An eligible turn is one that
    /// ended with no tool calls and in which the model did not itself call
    /// `remember`. `1` means every eligible turn.
    pub extract_every_n_turns: u32,
    /// A turn shorter than this does not trigger the extraction pass. The
    /// pass costs a KV snapshot, a prefill, a generation and a restore, and
    /// a four-second exchange is rarely worth that. `0` disables the floor.
    ///
    /// A turn under the floor *defers* its span rather than discarding it:
    /// `ExtractState::should_run` returns without advancing
    /// `processed_depth`, so the next turn that clears the floor reads the
    /// short turns too. Nothing said to the model is ever lost to this gate.
    pub min_turn_seconds: u32,
    /// Ask the model, before enqueuing an extraction pass, whether the span
    /// contains anything worth remembering (`Engine::decide`). Off by
    /// default: the threshold below has not been calibrated against real
    /// extraction outcomes, and a gate that wrongly says no loses a memory
    /// silently. Off leaves the pass behaving exactly as it did before the
    /// gate existed, and an engine without `supports_decide` ignores this.
    pub gate: bool,
    /// Probability of "yes", as a percent, at or above which the gate lets
    /// the pass run. An abstention — the model did not commit to a letter —
    /// always runs the pass: an unsure gate must not suppress a memory.
    pub gate_percent: u32,
    /// Messages a gate-rejected span may accumulate before an extraction pass
    /// runs anyway. `0` (the default) means a rejected span is finished
    /// immediately and never reconsidered. See `ExtractState::held_span_cap`.
    pub held_span_cap: u32,
    /// Percentage points added to `gate_percent` for each model family, to
    /// cancel that family's prior toward the letter "yes" is shown under.
    /// Measured offline by `/memory calibrate`; all zero until then.
    pub gate_bias: GateBias,
    /// Per-type character budgets for the rendered memory section.
    pub budgets: crate::memory::Budgets,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            auto_extract: true,
            extract_every_n_turns: 1,
            min_turn_seconds: 30,
            gate: false,
            gate_percent: 60,
            held_span_cap: 0,
            gate_bias: GateBias::default(),
            budgets: crate::memory::Budgets::default(),
        }
    }
}

impl MemorySettings {
    /// The gate threshold, as a percent, for a model of `family`:
    /// `gate_percent` moved by that family's calibrated letter bias and kept
    /// within 0..=100.
    #[must_use]
    pub fn gate_threshold_percent(&self, family: crate::gguf::ModelFamily) -> u32 {
        let t = i64::from(self.gate_percent) + i64::from(self.gate_bias.for_family(family));
        // Clamped to 0..=100 on the line above, so the conversion cannot fail.
        u32::try_from(t.clamp(0, 100)).unwrap_or(0)
    }
}

/// Per-family letter-bias correction for the memory gate, in percentage points
/// (`memory.gateBias`).
///
/// The gate asks its yes/no question with yes as A, always, and reads P(yes).
/// A family with a prior toward A as such inflates that number, and the
/// correction is the mean of that inflation measured by asking the same spans
/// with the letters swapped (`/memory calibrate`, `decide::bias_report`).
/// Measured per family because the prior belongs to the weights: a value
/// calibrated on one family says nothing about another. Positive raises the
/// bar. Bounded to ±[`GateBias::MAX_ABS`], far past any measured value, so a
/// typo cannot switch the gate fully on or off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateBias {
    pub ds4: i32,
    pub ds41: i32,
    pub qwen: i32,
}

impl GateBias {
    /// Largest correction accepted from a settings file, either direction.
    pub const MAX_ABS: i32 = 50;

    /// The settings key a family's correction is stored under.
    #[must_use]
    pub fn key(family: crate::gguf::ModelFamily) -> &'static str {
        match family {
            crate::gguf::ModelFamily::Ds4 => "ds4",
            crate::gguf::ModelFamily::Ds41 => "ds41",
            crate::gguf::ModelFamily::Qwen => "qwen",
        }
    }

    /// The correction for `family`.
    #[must_use]
    pub fn for_family(&self, family: crate::gguf::ModelFamily) -> i32 {
        match family {
            crate::gguf::ModelFamily::Ds4 => self.ds4,
            crate::gguf::ModelFamily::Ds41 => self.ds41,
            crate::gguf::ModelFamily::Qwen => self.qwen,
        }
    }
}

/// Predictive next-prompt suggestions in the TUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuggestionSettings {
    /// Generate a suggestion after each turn and show it as ghost text.
    ///
    /// On by default. The cost argument rests on the cold-KV skip: the
    /// expensive case — a session whose KV would rebuild from zero — is
    /// exactly the case the generation declines.
    pub enabled: bool,
    /// Token budget for the suggestion generation. A suggestion is one short
    /// line; anything past this is a model that misunderstood the
    /// instruction, and the sanitizer would reject it anyway.
    pub max_tokens: u32,
    /// How long a queued memory job may wait behind suggestions before it
    /// takes the idle slot back. `0` means suggestions never get priority.
    pub memory_starvation_seconds: u32,
}

impl Default for SuggestionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tokens: 160,
            memory_starvation_seconds: 300,
        }
    }
}

/// `kvcache` block: retention for persisted KV blobs.
///
/// Ages are in days and measured from a blob's `last_used`. A pinned blob and a
/// blob with a surviving child both outlive their TTL — see [`crate::kvgc`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheSettings {
    /// Days a session KV payload survives after its last use.
    pub ttl_session_days: u64,
    /// Days a system-prompt or project-stable checkpoint survives after its
    /// last use.
    pub ttl_tier_days: u64,
    /// Hard size ceiling in bytes, enforced after the TTL sweep: survivors are
    /// evicted least-recently-used first until the total fits. `0` disables the
    /// ceiling — it means unbounded, never "evict everything". Pinned, active
    /// and parent-of-a-survivor nodes are spared even when that leaves the
    /// total over budget.
    pub max_bytes: u64,
}

impl Default for KvCacheSettings {
    fn default() -> Self {
        Self {
            ttl_session_days: 14,
            ttl_tier_days: 30,
            max_bytes: 21_474_836_480,
        }
    }
}

/// Reads a positive integer member, ignoring absent, non-numeric, and
/// out-of-range values so one bad key cannot discard the rest of the file.
fn num<T: TryFrom<i64>>(obj: Option<&Json>, key: &str) -> Option<T> {
    let Some(Json::Num(n)) = obj?.get(key) else {
        return None;
    };
    // `as` on a non-finite or huge f64 saturates rather than failing, so the
    // range has to be checked before the cast.
    if !n.is_finite() || *n < MIN_SAFE || *n > MAX_SAFE {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)] // range-checked directly above
    T::try_from(*n as i64).ok()
}

/// Bounds outside which an `f64 -> i64` cast is not exact.
const MAX_SAFE: f64 = 9_007_199_254_740_992.0;
const MIN_SAFE: f64 = -MAX_SAFE;

fn boolean(obj: Option<&Json>, key: &str) -> Option<bool> {
    match obj?.get(key) {
        Some(Json::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// Reads an array-of-strings member, dropping non-string and empty entries so
/// one malformed element cannot discard the whole list.
fn strings(obj: Option<&Json>, key: &str) -> Option<Vec<String>> {
    let Some(Json::Arr(items)) = obj?.get(key) else {
        return None;
    };
    Some(
        items
            .iter()
            .filter_map(|v| match v {
                Json::Str(s) if !s.is_empty() => Some(s.clone()),
                _ => None,
            })
            .collect(),
    )
}

fn string(obj: Option<&Json>, key: &str) -> Option<String> {
    match obj?.get(key) {
        Some(Json::Str(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

impl Settings {
    /// Parses one `settings.json`, overlaying `self` key by key.
    ///
    /// Unknown keys are ignored, so a newer plank's file stays loadable by an
    /// older one. Provenance is recorded as [`Origin::UserSettings`] — the
    /// test-only convenience; the layered loaders use [`overlay_from`](Self::overlay_from).
    #[cfg(test)]
    fn overlay(&mut self, text: &str) {
        self.overlay_from(text, &crate::provenance::Origin::UserSettings);
    }

    /// [`overlay`](Self::overlay) with the provenance origin the file came from,
    /// so `/config --resolved` can report which layer won each key.
    // A mechanical key-by-key overlay; the length is not complexity.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn overlay_from(&mut self, text: &str, origin: &crate::provenance::Origin) {
        let Some(mut root) = json_parse(text) else {
            return;
        };
        // A plugin layer sits below the user, but "below" is not enough for a
        // key the user never set: the plugin's value would win by default.
        // Which model runs, which backend, what a worktree symlinks and which
        // tools exist are never a plugin's to supply, so those sections are
        // dropped before the overlay looks at them. `crate::plugins` warns
        // about each dropped key from the same list.
        if matches!(origin, crate::provenance::Origin::Plugin(_))
            && let Json::Obj(members) = &mut root
        {
            members.retain(|(k, _)| !crate::plugins::PLUGIN_REFUSED_SECTIONS.contains(&k.as_str()));
        }
        let engine = root.get("engine");
        if let Some(v) = string(engine, "model") {
            self.engine.model = Some(expand_tilde(&v));
            self.note("engine.model", origin);
        }
        if let Some(v) = num(engine, "threads") {
            self.engine.threads = Some(v);
            self.note("engine.threads", origin);
        }
        if let Some(v) = string(engine, "backend") {
            self.engine.backend = Some(v);
            self.note("engine.backend", origin);
        }
        if let Some(v) = num(engine, "power") {
            self.engine.power = Some(v);
            self.note("engine.power", origin);
        }
        if let Some(v) = num(engine, "ctx") {
            self.engine.ctx = Some(v);
            self.note("engine.ctx", origin);
        }
        if let Some(v) = boolean(engine, "thinkingToolCalls") {
            self.engine.thinking_tool_calls = v;
            self.note("engine.thinkingToolCalls", origin);
        }

        let ui = root.get("ui");
        if let Some(v) = boolean(ui, "respectGitignore") {
            self.ui.respect_gitignore = v;
            self.note("ui.respectGitignore", origin);
        }
        // A zero-row popup or zero-entry history would silently disable the
        // feature rather than tune it; treat those as unset.
        if let Some(v) = num::<usize>(ui, "popupRows").filter(|v| *v > 0) {
            self.ui.popup_rows = v;
            self.note("ui.popupRows", origin);
        }
        if let Some(v) = num(ui, "indexRefreshSecs") {
            self.ui.index_refresh_secs = v;
            self.note("ui.indexRefreshSecs", origin);
        }
        if let Some(v) = num::<usize>(ui, "historySize").filter(|v| *v > 0) {
            self.ui.history_size = v;
            self.note("ui.historySize", origin);
        }
        if let Some(v) = boolean(ui, "showToolCalls") {
            self.ui.show_tool_calls = v;
            self.note("ui.showToolCalls", origin);
        }
        if let Some(v) = boolean(ui, "showToolResults") {
            self.ui.show_tool_results = v;
            self.note("ui.showToolResults", origin);
        }
        if let Some(v) = boolean(ui, "showThinking") {
            self.ui.show_thinking = v;
            self.note("ui.showThinking", origin);
        }
        if let Some(v) = string(ui, "screensaver")
            && let Some(d) = crate::arcade::ScreensaverDelay::parse(&v)
        {
            self.ui.screensaver = d;
            self.note("ui.screensaver", origin);
        }
        // A face is either one plank ships or one a plugin contributes. The
        // built-in spellings win, so a plugin cannot capture the word "matrix"
        // by naming a face that; anything containing `:` is a plugin address
        // and is kept verbatim for the opener to resolve, because the plugin
        // set is not known at settings-parse time.
        if let Some(v) = string(ui, "screensaverFace") {
            if let Some(f) = crate::arcade::ScreensaverFace::parse(&v) {
                self.ui.screensaver_face = f;
                self.ui.screensaver_face_plugin = None;
            } else if v.contains(':') {
                self.ui.screensaver_face_plugin = Some(v);
            }
            self.note("ui.screensaverFace", origin);
        }
        // `notifications` accepts a mode string (always/unfocused/never) or
        // the legacy booleans (true=always, false=never).
        if let Some(v) = boolean(ui, "notifications") {
            self.ui.notifications = if v {
                crate::notify::NotifyMode::Always
            } else {
                crate::notify::NotifyMode::Never
            };
            self.note("ui.notifications", origin);
        } else if let Some(v) =
            string(ui, "notifications").and_then(|s| crate::notify::NotifyMode::parse(&s))
        {
            self.ui.notifications = v;
            self.note("ui.notifications", origin);
        }
        if let Some(v) = num(ui, "notifyAfterSecs") {
            self.ui.notify_after_secs = v;
            self.note("ui.notifyAfterSecs", origin);
        }
        if let Some(v) = boolean(ui, "crtOff") {
            self.ui.crt_off = v;
            self.note("ui.crtOff", origin);
        }
        if let Some(v) = boolean(ui, "reducedMotion") {
            self.ui.reduced_motion = v;
            self.note("ui.reducedMotion", origin);
        }
        if let Some(v) = boolean(ui, "easterEggs") {
            self.ui.easter_eggs = v;
            self.note("ui.easterEggs", origin);
        }
        if let Some(v) = boolean(ui, "builtinEditor") {
            self.ui.builtin_editor = v;
            self.note("ui.builtinEditor", origin);
        }

        let safety = root.get("safety");
        if let Some(v) = boolean(safety, "sandbox") {
            self.safety.sandbox = Some(v);
            self.note("safety.sandbox", origin);
        }
        if let Some(v) = boolean(safety, "btwSuspend") {
            self.safety.btw_suspend = Some(v);
            self.note("safety.btwSuspend", origin);
        }

        if let Some(v) = num::<u64>(root.get("mcp"), "timeoutSecs").filter(|v| *v > 0) {
            self.mcp.timeout_secs = v;
            self.note("mcp.timeoutSecs", origin);
        }

        // A max below the fixed minimum would make every `ask` call impossible;
        // clamp it up rather than silently breaking the tool.
        if let Some(v) = num::<usize>(root.get("ask"), "maxOptions") {
            self.ask.max_options = v.max(ASK_MIN_OPTIONS);
            self.note("ask.maxOptions", origin);
        }

        if let Some(v) = boolean(root.get("update"), "check") {
            self.update.check = v;
            self.note("update.check", origin);
        }

        let tools = root.get("tools");
        if let Some(v) = boolean(tools, "repeatAdvisory") {
            self.tools.repeat_advisory = v;
            self.note("tools.repeatAdvisory", origin);
        }
        if let Some(v) = boolean(tools, "loopGuards") {
            self.tools.loop_guards = v;
            self.note("tools.loopGuards", origin);
        }
        if let Some(v) = boolean(tools, "noProgressGuard") {
            self.tools.no_progress_guard = v;
            self.note("tools.noProgressGuard", origin);
        }
        if let Some(v) = num::<u64>(tools, "callTimeoutSec") {
            self.tools.call_timeout_sec = v;
            self.note("tools.callTimeoutSec", origin);
        }
        if let Some(v) = num::<usize>(tools, "spillMaxBytes") {
            self.tools.spill_max_bytes = v;
            self.note("tools.spillMaxBytes", origin);
        }
        if let Some(v) = num::<usize>(tools, "spillPreviewBytes") {
            self.tools.spill_preview_bytes = v;
            self.note("tools.spillPreviewBytes", origin);
        }
        if let Some(v) = boolean(tools, "recall") {
            self.tools.recall = v;
            self.note("tools.recall", origin);
        }
        if let Some(v) = boolean(tools, "fanout") {
            self.tools.fanout = v;
            self.note("tools.fanout", origin);
        }
        if let Some(v) = boolean(tools, "runCode") {
            self.tools.run_code = v;
            self.note("tools.runCode", origin);
        }
        if let Some(v) = boolean(tools, "bashNotify") {
            self.tools.bash_notify = v;
            self.note("tools.bashNotify", origin);
        }
        if let Some(v) = boolean(tools, "remember") {
            self.tools.remember = v;
            self.note("tools.remember", origin);
        }

        self.overlay_agents_and_worktree(&root, origin);
    }

    /// Records that `origin` set the settings key `key` (`section.key`).
    fn note(&mut self, key: &str, origin: &crate::provenance::Origin) {
        self.provenance
            .entry(key.to_string())
            .or_insert_with(|| crate::provenance::Provenance::new(origin.clone()))
            .note(origin.clone());
    }

    /// The `suggestions` section of [`overlay_agents_and_worktree`], split out
    /// to keep each function under the length lint.
    fn overlay_suggestions(&mut self, root: &Json, origin: &crate::provenance::Origin) {
        if let Some(v) = boolean(root.get("suggestions"), "enabled") {
            self.suggestions.enabled = v;
            self.note("suggestions.enabled", origin);
        }
        if let Some(v) = num::<u32>(root.get("suggestions"), "maxTokens") {
            self.suggestions.max_tokens = v.max(1);
            self.note("suggestions.maxTokens", origin);
        }
        if let Some(v) = num::<u32>(root.get("suggestions"), "memoryStarvationSeconds") {
            self.suggestions.memory_starvation_seconds = v;
            self.note("suggestions.memoryStarvationSeconds", origin);
        }
    }

    /// The `agents` and `worktree` half of [`overlay`](Self::overlay), split out
    /// only to keep each function under the length lint.
    /// `memory.gateBias`: per-family, signed, clamped to ±`GateBias::MAX_ABS`.
    fn overlay_gate_bias(&mut self, root: &Json, origin: &crate::provenance::Origin) {
        if let Some(b) = root.get("memory").and_then(|m| m.get("gateBias")) {
            let set = |key: &str, field: &mut i32| {
                if let Some(v) = num::<i32>(Some(b), key) {
                    *field = v.clamp(-GateBias::MAX_ABS, GateBias::MAX_ABS);
                }
            };
            set("ds4", &mut self.memory.gate_bias.ds4);
            set("ds41", &mut self.memory.gate_bias.ds41);
            set("qwen", &mut self.memory.gate_bias.qwen);
            self.note("memory.gateBias", origin);
        }
    }

    fn overlay_agents_and_worktree(&mut self, root: &Json, origin: &crate::provenance::Origin) {
        let agents = root.get("agents");
        if let Some(v) = boolean(agents, "autoRoute") {
            self.agents.auto_route = v;
            self.note("agents.autoRoute", origin);
        }
        if let Some(v) = num::<usize>(agents, "maxParallel") {
            self.agents.max_parallel = v.clamp(1, AGENT_MAX_PARALLEL);
            self.note("agents.maxParallel", origin);
        }

        let worktree = root.get("worktree");
        if let Some(v) = strings(worktree, "sparsePaths") {
            self.worktree.sparse_paths = v;
            self.note("worktree.sparsePaths", origin);
        }
        if let Some(v) = strings(worktree, "symlinkDirectories") {
            self.worktree.symlink_directories = v;
            self.note("worktree.symlinkDirectories", origin);
        }
        if let Some(v) = boolean(worktree, "isolateAgents") {
            self.worktree.isolate_agents = v;
            self.note("worktree.isolateAgents", origin);
        }

        let kvcache = root.get("kvcache");
        if let Some(v) = num::<u64>(kvcache, "ttlSessionDays") {
            self.kvcache.ttl_session_days = v;
            self.note("kvcache.ttlSessionDays", origin);
        }
        if let Some(v) = num::<u64>(kvcache, "ttlTierDays") {
            self.kvcache.ttl_tier_days = v;
            self.note("kvcache.ttlTierDays", origin);
        }
        if let Some(v) = num::<u64>(kvcache, "maxBytes") {
            self.kvcache.max_bytes = v;
            self.note("kvcache.maxBytes", origin);
        }

        if let Some(v) = boolean(root.get("context"), "microcompact") {
            self.context.microcompact = v;
            self.note("context.microcompact", origin);
        }
        if let Some(v) = boolean(root.get("context"), "shortReminder") {
            self.context.short_reminder = v;
            self.note("context.shortReminder", origin);
        }

        if let Some(v) = boolean(root.get("memory"), "autoExtract") {
            self.memory.auto_extract = v;
            self.note("memory.autoExtract", origin);
        }
        if let Some(v) = num::<u32>(root.get("memory"), "extractEveryNTurns") {
            self.memory.extract_every_n_turns = v.max(1);
            self.note("memory.extractEveryNTurns", origin);
        }
        if let Some(v) = num::<u32>(root.get("memory"), "minTurnSeconds") {
            self.memory.min_turn_seconds = v;
            self.note("memory.minTurnSeconds", origin);
        }
        if let Some(v) = boolean(root.get("memory"), "gate") {
            self.memory.gate = v;
            self.note("memory.gate", origin);
        }
        if let Some(v) = num::<u32>(root.get("memory"), "gatePercent") {
            self.memory.gate_percent = v.min(100);
            self.note("memory.gatePercent", origin);
        }
        if let Some(v) = num::<u32>(root.get("memory"), "heldSpanCap") {
            self.memory.held_span_cap = v;
            self.note("memory.heldSpanCap", origin);
        }
        self.overlay_gate_bias(root, origin);
        if let Some(b) = root.get("memory").and_then(|m| m.get("budgets")) {
            let set = |key: &str, field: &mut usize| {
                if let Some(v) = num::<usize>(Some(b), key) {
                    *field = v;
                }
            };
            set("user", &mut self.memory.budgets.user);
            set("feedback", &mut self.memory.budgets.feedback);
            set("project", &mut self.memory.budgets.project);
            set("reference", &mut self.memory.budgets.reference);
            self.note("memory.budgets", origin);
        }

        self.overlay_suggestions(root, origin);

        if let Some(v) = boolean(root.get("git"), "signCommits") {
            self.git.sign_commits = v;
            self.note("git.signCommits", origin);
        }

        // Merged key by key rather than replaced wholesale: a project file that
        // sets one plugin option must not silently drop the user's settings for
        // every other one, which is what assigning the map would do.
        if let Some(Json::Obj(entries)) = root.get("pluginConfig") {
            for (key, value) in entries {
                if let Json::Str(v) = value {
                    self.plugin_config.insert(key.clone(), v.clone());
                    self.note(&format!("pluginConfig.{key}"), origin);
                }
            }
        }
    }

    /// Overlays `low` files then `high` files, in that order, on the
    /// built-in defaults. Exists so plugin settings can sit strictly below
    /// the user and project files.
    #[must_use]
    pub fn load_from_paths(low: &[PathBuf], high: &[PathBuf]) -> Self {
        let mut s = Self::default();
        // Provenance per file: `low` is the plugin layer, `high` is the user
        // file then the project file (see `paths_in`). Overlay runs low-to-high,
        // so a later layer demotes the earlier one to shadowed.
        let low_origins = low
            .iter()
            .map(|p| (p, crate::provenance::Origin::Plugin(String::new())));
        let high_origins = high.iter().enumerate().map(|(i, p)| {
            let origin = if i == 0 {
                crate::provenance::Origin::UserSettings
            } else {
                crate::provenance::Origin::ProjectSettings
            };
            (p, origin)
        });
        for (p, origin) in low_origins.chain(high_origins) {
            if let Ok(text) = std::fs::read_to_string(p) {
                s.overlay_from(&text, &origin);
            }
        }
        s
    }

    /// [`load`](Self::load) with plugin-contributed settings applied first, so
    /// `defaults < plugins < ~/.plank < ./.plank`.
    #[must_use]
    pub fn load_with_plugins(plugin_paths: &[PathBuf]) -> Self {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let cwd = std::env::current_dir().unwrap_or_default();
        Self::load_with_plugins_in(home.as_deref(), &cwd, plugin_paths)
    }

    /// Hermetic seam for [`load_with_plugins`](Self::load_with_plugins): takes
    /// `home`/`cwd` explicitly instead of reading the environment, so tests can
    /// exercise the real precedence composition without touching `HOME`.
    #[must_use]
    pub fn load_with_plugins_in(home: Option<&Path>, cwd: &Path, plugin_paths: &[PathBuf]) -> Self {
        Self::load_from_paths(plugin_paths, &Self::paths_in(home, cwd))
    }

    /// Loads `~/.plank/settings.json` then `<cwd>/.plank/settings.json`.
    #[must_use]
    pub fn load() -> Self {
        Self::load_from_paths(&[], &Self::paths())
    }

    /// The files [`load`](Self::load) consults, in increasing precedence.
    #[must_use]
    pub fn paths() -> Vec<PathBuf> {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        match std::env::current_dir() {
            Ok(cwd) => Self::paths_in(home.as_deref(), &cwd),
            Err(_) => home
                .map(|home| crate::home::plank_home_in(home).join("settings.json"))
                .into_iter()
                .collect(),
        }
    }

    /// Hermetic seam for [`paths`](Self::paths): takes `home`/`cwd` explicitly
    /// instead of reading the environment.
    #[must_use]
    pub fn paths_in(home: Option<&Path>, cwd: &Path) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Some(home) = home {
            paths.push(crate::home::plank_home_in(home).join("settings.json"));
        }
        paths.push(cwd.join(".plank").join("settings.json"));
        paths
    }

    /// The settings files that actually exist, for the startup note.
    #[must_use]
    pub fn existing_paths() -> Vec<PathBuf> {
        Self::paths().into_iter().filter(|p| p.is_file()).collect()
    }
}

/// One line naming every setting that is actually in effect, or `None` when
/// the files changed nothing.
///
/// A settings file can move you off Metal onto the CPU or shrink the context,
/// and both are invisible once the UI is up — you just notice plank is slow.
/// This makes the cause self-diagnosing.
///
/// `cfg` is consulted so a setting a CLI flag overrode is *not* reported: the
/// note lists what is in force, never what a file merely asked for.
// A flat, one-line-per-setting audit of what is in force; long by nature, but
// each arm is a trivial compare-and-push, so the length is not complexity.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn startup_note(s: &Settings, cfg: &crate::config::AgentConfig) -> Option<String> {
    let d = Settings::default();
    let mut parts: Vec<String> = Vec::new();

    // Engine and safety keys: reported only when the parsed config still
    // carries the file's value, i.e. no flag overrode it.
    if let Some(m) = &s.engine.model
        && cfg.model_path.as_ref() == Some(m)
    {
        parts.push(format!("model={}", m.display()));
    }
    if let Some(t) = s.engine.threads
        && cfg.n_threads == t
    {
        parts.push(format!("threads={t}"));
    }
    if let Some(b) = s.engine.backend.as_deref()
        && cfg.backend == crate::config::parse_backend(b)
    {
        parts.push(format!("backend={b}"));
    }
    if let Some(p) = s.engine.power
        && cfg.power_percent == p
    {
        parts.push(format!("power={p}"));
    }
    if let Some(c) = s.engine.ctx
        && cfg.generation.ctx_size == c
    {
        parts.push(format!("ctx={c}"));
    }
    if let Some(v) = s.safety.sandbox
        && cfg.sandbox_override == Some(v)
    {
        parts.push(format!("sandbox={v}"));
    }
    if let Some(v) = s.safety.btw_suspend
        && cfg.btw.suspend == v
    {
        parts.push(format!("btwSuspend={v}"));
    }

    // Read straight from settings at stream setup, so like the UI keys below
    // any non-default value is in force.
    if s.engine.thinking_tool_calls != d.engine.thinking_tool_calls {
        parts.push(format!(
            "thinkingToolCalls={}",
            s.engine.thinking_tool_calls
        ));
    }

    // UI and MCP keys have no flag, so any non-default value is in force.
    if s.ui.respect_gitignore != d.ui.respect_gitignore {
        parts.push(format!("respectGitignore={}", s.ui.respect_gitignore));
    }
    if s.ui.popup_rows != d.ui.popup_rows {
        parts.push(format!("popupRows={}", s.ui.popup_rows));
    }
    if s.ui.index_refresh_secs != d.ui.index_refresh_secs {
        parts.push(format!("indexRefreshSecs={}", s.ui.index_refresh_secs));
    }
    if s.ui.history_size != d.ui.history_size {
        parts.push(format!("historySize={}", s.ui.history_size));
    }
    if s.ui.show_tool_calls != d.ui.show_tool_calls {
        parts.push(format!("showToolCalls={}", s.ui.show_tool_calls));
    }
    if s.ui.show_tool_results != d.ui.show_tool_results {
        parts.push(format!("showToolResults={}", s.ui.show_tool_results));
    }
    if s.ui.show_thinking != d.ui.show_thinking {
        parts.push(format!("showThinking={}", s.ui.show_thinking));
    }
    if s.ui.notifications != d.ui.notifications {
        parts.push(format!("notifications={}", s.ui.notifications.as_str()));
    }
    if s.ui.notify_after_secs != d.ui.notify_after_secs {
        parts.push(format!("notifyAfterSecs={}", s.ui.notify_after_secs));
    }
    if s.ui.crt_off != d.ui.crt_off {
        parts.push(format!("crtOff={}", s.ui.crt_off));
    }
    if s.ui.reduced_motion != d.ui.reduced_motion {
        parts.push(format!("reducedMotion={}", s.ui.reduced_motion));
    }
    if s.ui.screensaver_face != d.ui.screensaver_face {
        parts.push(format!(
            "screensaverFace={}",
            s.ui.screensaver_face.as_str()
        ));
    }
    if s.ui.screensaver != d.ui.screensaver {
        parts.push(format!("screensaver={}", s.ui.screensaver.as_str()));
    }
    if s.ui.easter_eggs != d.ui.easter_eggs {
        parts.push(format!("easterEggs={}", s.ui.easter_eggs));
    }
    if s.ui.builtin_editor != d.ui.builtin_editor {
        parts.push(format!("builtinEditor={}", s.ui.builtin_editor));
    }
    if s.mcp.timeout_secs != d.mcp.timeout_secs {
        parts.push(format!("timeoutSecs={}", s.mcp.timeout_secs));
    }
    if s.ask.max_options != d.ask.max_options {
        parts.push(format!("maxOptions={}", s.ask.max_options));
    }
    if s.update.check != d.update.check {
        parts.push(format!("update.check={}", s.update.check));
    }
    if s.agents.auto_route != d.agents.auto_route {
        parts.push(format!("agents.autoRoute={}", s.agents.auto_route));
    }
    if s.agents.max_parallel != d.agents.max_parallel {
        parts.push(format!("agents.maxParallel={}", s.agents.max_parallel));
    }

    if parts.is_empty() {
        return None;
    }
    let from = Settings::existing_paths()
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let from = if from.is_empty() {
        "settings".to_string()
    } else {
        from
    };
    Some(format!(
        "plank: settings in effect ({from}): {}",
        parts.join(", ")
    ))
}

/// Expands a leading `~/` against `$HOME`, leaving other paths untouched.
fn expand_tilde(s: &str) -> PathBuf {
    match (s.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(s),
    }
}

/// The project-scoped settings file, `<cwd>/.plank/settings.json` — the
/// highest-precedence file and where `/config` writes.
#[must_use]
pub fn project_path() -> Option<PathBuf> {
    std::env::current_dir()
        .ok()
        .map(|cwd| cwd.join(".plank").join("settings.json"))
}

fn upsert(obj: &mut Vec<(String, Json)>, key: &str, val: Json) {
    if let Some(slot) = obj.iter_mut().find(|(k, _)| k == key) {
        slot.1 = val;
    } else {
        obj.push((key.to_string(), val));
    }
}

/// Upserts an optional value, or removes the key when `val` is `None` so that
/// an unset optional is reflected as absence (its built-in default on reload).
fn upsert_opt(obj: &mut Vec<(String, Json)>, key: &str, val: Option<Json>) {
    match val {
        Some(v) => upsert(obj, key, v),
        None => obj.retain(|(k, _)| k != key),
    }
}

/// Returns the named section object, creating it if absent or non-object.
fn section<'a>(root: &'a mut Vec<(String, Json)>, name: &str) -> &'a mut Vec<(String, Json)> {
    let idx = if let Some(i) = root.iter().position(|(k, _)| k == name) {
        if !matches!(root[i].1, Json::Obj(_)) {
            root[i].1 = Json::Obj(Vec::new());
        }
        i
    } else {
        root.push((name.to_string(), Json::Obj(Vec::new())));
        root.len() - 1
    };
    match &mut root[idx].1 {
        Json::Obj(o) => o,
        _ => unreachable!("just ensured an object"),
    }
}

#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
fn inum(v: i32) -> Json {
    Json::Num(v as f64)
}

#[allow(clippy::cast_precision_loss)]
fn unum(v: u64) -> Json {
    Json::Num(v as f64)
}

/// Pretty-prints a JSON value with two-space indentation (objects only get
/// multi-line treatment; scalars and arrays stay compact via [`json_write`]).
fn write_pretty(out: &mut String, v: &Json, indent: usize) {
    match v {
        Json::Obj(members) if !members.is_empty() => {
            out.push_str("{\n");
            for (i, (k, val)) in members.iter().enumerate() {
                for _ in 0..=indent {
                    out.push_str("  ");
                }
                json_escape(out, k);
                out.push_str(": ");
                write_pretty(out, val, indent + 1);
                if i + 1 < members.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            for _ in 0..indent {
                out.push_str("  ");
            }
            out.push('}');
        }
        other => json_write(out, other),
    }
}

impl Settings {
    /// Serializes these settings to `path`, preserving any unknown keys already
    /// present (so a newer plank's file survives an older binary's write).
    ///
    /// # Errors
    /// Returns `Err` if the parent directory cannot be created or the write fails.
    #[allow(clippy::too_many_lines)]
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        let mut root: Vec<(String, Json)> = match std::fs::read_to_string(path) {
            Ok(t) => match json_parse(&t) {
                Some(Json::Obj(o)) => o,
                _ => Vec::new(),
            },
            Err(_) => Vec::new(),
        };

        {
            let e = section(&mut root, "engine");
            upsert_opt(
                e,
                "model",
                self.engine
                    .model
                    .as_ref()
                    .map(|p| Json::Str(p.display().to_string())),
            );
            upsert_opt(e, "threads", self.engine.threads.map(inum));
            upsert_opt(e, "backend", self.engine.backend.clone().map(Json::Str));
            upsert_opt(e, "power", self.engine.power.map(inum));
            upsert_opt(e, "ctx", self.engine.ctx.map(inum));
            upsert(
                e,
                "thinkingToolCalls",
                Json::Bool(self.engine.thinking_tool_calls),
            );
        }
        {
            let u = section(&mut root, "ui");
            upsert(u, "respectGitignore", Json::Bool(self.ui.respect_gitignore));
            upsert(u, "popupRows", unum(self.ui.popup_rows as u64));
            upsert(u, "indexRefreshSecs", unum(self.ui.index_refresh_secs));
            upsert(u, "historySize", unum(self.ui.history_size as u64));
            upsert(u, "showToolCalls", Json::Bool(self.ui.show_tool_calls));
            upsert(u, "showToolResults", Json::Bool(self.ui.show_tool_results));
            upsert(u, "showThinking", Json::Bool(self.ui.show_thinking));
            upsert(
                u,
                "notifications",
                Json::Str(self.ui.notifications.as_str().to_string()),
            );
            upsert(u, "notifyAfterSecs", unum(self.ui.notify_after_secs));
            upsert(u, "crtOff", Json::Bool(self.ui.crt_off));
            upsert(
                u,
                "screensaver",
                Json::Str(self.ui.screensaver.as_str().to_string()),
            );
            upsert(
                u,
                "screensaverFace",
                Json::Str(self.ui.screensaver_face.as_str().to_string()),
            );
            upsert(u, "easterEggs", Json::Bool(self.ui.easter_eggs));
            upsert(u, "builtinEditor", Json::Bool(self.ui.builtin_editor));
        }
        {
            let s = section(&mut root, "safety");
            upsert_opt(s, "sandbox", self.safety.sandbox.map(Json::Bool));
            upsert_opt(s, "btwSuspend", self.safety.btw_suspend.map(Json::Bool));
        }
        upsert(
            section(&mut root, "mcp"),
            "timeoutSecs",
            unum(self.mcp.timeout_secs),
        );
        upsert(
            section(&mut root, "ask"),
            "maxOptions",
            unum(self.ask.max_options as u64),
        );
        upsert(
            section(&mut root, "update"),
            "check",
            Json::Bool(self.update.check),
        );
        {
            let a = section(&mut root, "agents");
            upsert(a, "autoRoute", Json::Bool(self.agents.auto_route));
            upsert(a, "maxParallel", unum(self.agents.max_parallel as u64));
        }
        upsert(
            section(&mut root, "git"),
            "signCommits",
            Json::Bool(self.git.sign_commits),
        );
        upsert(
            section(&mut root, "context"),
            "microcompact",
            Json::Bool(self.context.microcompact),
        );
        upsert(
            section(&mut root, "context"),
            "shortReminder",
            Json::Bool(self.context.short_reminder),
        );
        {
            let m = section(&mut root, "memory");
            upsert(m, "autoExtract", Json::Bool(self.memory.auto_extract));
            upsert(
                m,
                "extractEveryNTurns",
                unum(u64::from(self.memory.extract_every_n_turns)),
            );
            upsert(
                m,
                "minTurnSeconds",
                unum(u64::from(self.memory.min_turn_seconds)),
            );
            upsert(m, "gate", Json::Bool(self.memory.gate));
            upsert(m, "gatePercent", unum(u64::from(self.memory.gate_percent)));
            upsert(m, "heldSpanCap", unum(u64::from(self.memory.held_span_cap)));
            // Absent until calibrated, so an uncalibrated settings file does
            // not grow three zeros nobody chose.
            let b = self.memory.gate_bias;
            upsert_opt(
                m,
                "gateBias",
                (b != GateBias::default()).then(|| {
                    Json::Obj(vec![
                        ("ds4".to_string(), inum(b.ds4)),
                        ("ds41".to_string(), inum(b.ds41)),
                        ("qwen".to_string(), inum(b.qwen)),
                    ])
                }),
            );
        }
        {
            let s = section(&mut root, "suggestions");
            upsert(s, "enabled", Json::Bool(self.suggestions.enabled));
            upsert(s, "maxTokens", unum(u64::from(self.suggestions.max_tokens)));
            upsert(
                s,
                "memoryStarvationSeconds",
                unum(u64::from(self.suggestions.memory_starvation_seconds)),
            );
        }
        {
            let t = section(&mut root, "tools");
            upsert(t, "repeatAdvisory", Json::Bool(self.tools.repeat_advisory));
            upsert(t, "loopGuards", Json::Bool(self.tools.loop_guards));
            upsert(
                t,
                "noProgressGuard",
                Json::Bool(self.tools.no_progress_guard),
            );
            upsert(t, "callTimeoutSec", unum(self.tools.call_timeout_sec));
            upsert(t, "spillMaxBytes", unum(self.tools.spill_max_bytes as u64));
            upsert(
                t,
                "spillPreviewBytes",
                unum(self.tools.spill_preview_bytes as u64),
            );
            upsert(t, "recall", Json::Bool(self.tools.recall));
            upsert(t, "fanout", Json::Bool(self.tools.fanout));
            upsert(t, "runCode", Json::Bool(self.tools.run_code));
            upsert(t, "bashNotify", Json::Bool(self.tools.bash_notify));
            upsert(t, "remember", Json::Bool(self.tools.remember));
        }

        let mut out = String::new();
        write_pretty(&mut out, &Json::Obj(root), 0);
        out.push('\n');
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, out).map_err(|e| e.to_string())
    }
}

/// Process-wide settings. A swappable `&'static` (the payload is `Box::leak`ed)
/// so [`reinstall`] can update it live from `/config` without changing the
/// zero-cost `&'static` contract that [`active`]'s many call sites rely on. The
/// per-swap leak is bounded — swaps happen only on explicit user action.
static ACTIVE: RwLock<Option<&'static Settings>> = RwLock::new(None);

/// Installs the process-wide settings. Later calls are ignored.
///
/// Call once from `main` before the UI starts. Code that reads settings via
/// [`active`] sees built-in defaults until this runs, which is what tests and
/// library consumers get.
pub fn install(settings: Settings) {
    // Recover rather than panic on a poisoned lock: settings are advisory and a
    // stale guard is harmless here.
    crate::anim::set_reduced_motion(settings.ui.reduced_motion);
    // Seeds the notification mode the same way `reducedMotion` seeds `anim`:
    // this is the one choke point both front-ends now rely on instead of each
    // calling `notify::set_mode` once at startup themselves.
    crate::notify::set_mode(settings.ui.notifications);
    let mut slot = ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if slot.is_none() {
        *slot = Some(Box::leak(Box::new(settings)));
    }
    drop(slot);
    // showThinking may already be off in the loaded config at startup (it is
    // the new default), so the debug-console mirror must be reconciled here
    // too, not only on a later `/config` change.
    crate::debugmirror::reconcile();
}

/// Replaces the process-wide settings (used by `/config` after a save), so the
/// current session picks up the change on its next [`active`] read.
pub fn reinstall(settings: Settings) {
    crate::anim::set_reduced_motion(settings.ui.reduced_motion);
    // `ui.notifications` was previously seeded once at startup and never
    // re-pushed, so changing it in `/config` had no effect until restart.
    crate::notify::set_mode(settings.ui.notifications);
    *ACTIVE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::leak(Box::new(settings)));
    // The common choke point for both `/config showThinking <bool>` and the
    // interactive config form save: reconciling here, rather than at each
    // call site, is what makes the toggle take effect immediately instead of
    // waiting for the next turn to notice.
    crate::debugmirror::reconcile();
}

/// Session-only override of `ui.showThinking`, set by the footer's brain
/// click. `None` means "no override": the persisted `ui.show_thinking` stands.
///
/// Deliberately *not* a field on [`Settings`]: a value living there would be
/// written out by [`Settings::save_to`] the next time anything saved the
/// settings (a later `/config <any key>`, or a config-form save), which is
/// exactly the accidental persistence the override exists to prevent. This
/// layer is never serialised and never reaches disk.
#[cfg(not(test))]
static SHOW_THINKING_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(SHOW_THINKING_NONE);

// Scoped to the calling thread in tests for the same reason as
// `TEST_OVERRIDE` below: libtest runs tests concurrently in one process, and a
// process-wide override would leak between them.
#[cfg(test)]
thread_local! {
    static SHOW_THINKING_OVERRIDE: std::cell::Cell<i8> =
        const { std::cell::Cell::new(SHOW_THINKING_NONE) };
}

/// The `SHOW_THINKING_OVERRIDE` encoding of `None`; `0`/`1` are `false`/`true`.
const SHOW_THINKING_NONE: i8 = -1;

/// The live session override of `ui.showThinking`, or `None` when none is set.
#[must_use]
pub fn show_thinking_override() -> Option<bool> {
    #[cfg(test)]
    let raw = SHOW_THINKING_OVERRIDE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let raw = SHOW_THINKING_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    (raw != SHOW_THINKING_NONE).then_some(raw != 0)
}

/// Session-only override of `tools.loopGuards`, set by `/loopguard`
/// (`/lg`). `None` means "no override": the persisted `tools.loop_guards`
/// stands.
///
/// Deliberately *not* a field on [`Settings`], for the same reason as
/// [`SHOW_THINKING_OVERRIDE`]: a value living there would be written out by
/// [`Settings::save_to`] the next time anything saved the settings (a later
/// `/config <any key>`), which is exactly the accidental persistence this
/// layer exists to prevent. Never serialised, never reaches disk, and — per
/// the loop guard's own doc comment — outlives `/clear`/`/new`, since it is
/// process state rather than session state: it defaults on as a protection,
/// and turning it off is a diagnostic act whose whole point is to survive the
/// investigation that prompted it.
#[cfg(not(test))]
static LOOP_GUARDS_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(LOOP_GUARDS_NONE);

// Scoped to the calling thread in tests for the same reason as
// `SHOW_THINKING_OVERRIDE` above: libtest runs tests concurrently in one
// process, and a process-wide override would leak between them.
#[cfg(test)]
thread_local! {
    static LOOP_GUARDS_OVERRIDE: std::cell::Cell<i8> =
        const { std::cell::Cell::new(LOOP_GUARDS_NONE) };
}

/// The `LOOP_GUARDS_OVERRIDE` encoding of `None`; `0`/`1` are `false`/`true`.
const LOOP_GUARDS_NONE: i8 = -1;

/// The live session override of `tools.loopGuards`, or `None` when none is
/// set.
#[must_use]
pub fn loop_guards_override() -> Option<bool> {
    #[cfg(test)]
    let raw = LOOP_GUARDS_OVERRIDE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let raw = LOOP_GUARDS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    (raw != LOOP_GUARDS_NONE).then_some(raw != 0)
}

/// Sets (or with `None` clears) the session override.
///
/// Unlike `set_show_thinking_override`, there is no side effect to replicate
/// here: [`reinstall`]'s four effects (`anim::set_reduced_motion`,
/// `notify::set_mode`, swapping the `ACTIVE` slot, `debugmirror::reconcile`)
/// each depend on a field other than `tools.loop_guards`, and the guards
/// themselves are read fresh from [`crate::guard::guards_enabled`] at every
/// check rather than cached anywhere that would need reconciling.
pub fn set_loop_guards_override(value: Option<bool>) {
    let raw = value.map_or(LOOP_GUARDS_NONE, i8::from);
    #[cfg(test)]
    LOOP_GUARDS_OVERRIDE.with(|c| c.set(raw));
    #[cfg(not(test))]
    LOOP_GUARDS_OVERRIDE.store(raw, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a scoped suspension of the loop guards is in force; see
/// [`suspend_loop_guards`].
#[cfg(not(test))]
static LOOP_GUARDS_SUSPENDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// Thread-scoped in tests for the same reason as `LOOP_GUARDS_OVERRIDE`.
#[cfg(test)]
thread_local! {
    static LOOP_GUARDS_SUSPENDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[must_use]
fn loop_guards_suspended() -> bool {
    #[cfg(test)]
    {
        LOOP_GUARDS_SUSPENDED.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        LOOP_GUARDS_SUSPENDED.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn set_loop_guards_suspended(value: bool) {
    #[cfg(test)]
    LOOP_GUARDS_SUSPENDED.with(|c| c.set(value));
    #[cfg(not(test))]
    LOOP_GUARDS_SUSPENDED.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Silences the loop guards until the returned value is dropped.
///
/// For a turn plank drives from a canned multi-phase prompt rather than from
/// something the user typed — `/init` is the one today. Those phases repeat
/// tool calls *by design*: the survey re-reads the tree the interview asked
/// about, and the guards read that as the loop they exist to stop, so a turn
/// the user asked for is refused mid-phase for doing exactly what its prompt
/// told it to.
///
/// A scope rather than a flag, because the restore has to survive the early
/// return on a failed turn: leaving the guards down after `/init` errored
/// would disarm the protection for the rest of the session with nothing on
/// screen saying so. Re-entrant: the previous value is restored, not `false`.
///
/// This is deliberately *below* the `/loopguard` session override in
/// [`loop_guards_effective`]. A user who typed `/loopguard on` said something
/// about this session that a canned prompt does not get to overrule, and one
/// who typed `/loopguard off` is already where the suspension would put them.
#[must_use = "the guards are re-armed when this value is dropped"]
pub fn suspend_loop_guards() -> LoopGuardsSuspended {
    let previous = loop_guards_suspended();
    set_loop_guards_suspended(true);
    LoopGuardsSuspended { previous }
}

/// The scope handle returned by [`suspend_loop_guards`]; re-arms the guards on
/// drop.
#[derive(Debug)]
pub struct LoopGuardsSuspended {
    previous: bool,
}

impl Drop for LoopGuardsSuspended {
    fn drop(&mut self) {
        set_loop_guards_suspended(self.previous);
    }
}

/// Whether the loop guards are armed *right now*: the `/loopguard` session
/// override if one is set, else off inside a [`suspend_loop_guards`] scope,
/// else the persisted `tools.loopGuards`.
///
/// [`crate::guard::guards_enabled`] is the sole reader of this; every guard
/// check goes through it rather than reading `tools.loop_guards` directly.
#[must_use]
pub fn loop_guards_effective() -> bool {
    if let Some(explicit) = loop_guards_override() {
        return explicit;
    }
    if loop_guards_suspended() {
        return false;
    }
    active().tools.loop_guards
}

/// Sets (or with `None` clears) the session override, then reconciles the
/// debug-console mirror.
///
/// The reconcile is the one side effect of [`reinstall`] that depends on
/// `showThinking`: the mirror is connected only while thinking is *hidden*, so
/// an override that changed the effective value without reconciling would
/// leave a console attached (or detached) against what the user just asked
/// for. `reinstall`'s other effects — `anim::set_reduced_motion`,
/// `notify::set_mode`, and swapping the `ACTIVE` slot — read fields this layer
/// cannot touch, so they are deliberately not replicated.
pub fn set_show_thinking_override(value: Option<bool>) {
    let raw = value.map_or(SHOW_THINKING_NONE, i8::from);
    #[cfg(test)]
    SHOW_THINKING_OVERRIDE.with(|c| c.set(raw));
    #[cfg(not(test))]
    SHOW_THINKING_OVERRIDE.store(raw, std::sync::atomic::Ordering::Relaxed);
    crate::debugmirror::reconcile();
}

/// Whether thinking is displayed *right now*: the session override if one is
/// set, else the persisted `ui.showThinking`.
///
/// Every display decision reads this. The persisted field is read directly
/// only where the *configured* value is the subject: saving settings,
/// rendering the config form, and the session file's render record.
#[must_use]
pub fn show_thinking_effective() -> bool {
    show_thinking_override().unwrap_or_else(|| active().ui.show_thinking)
}

// Test-only settings override, scoped to the calling thread. The libtest
// harness runs each test on its own thread, so this lets one test exercise a
// non-default setting without disturbing the process-wide slot that tests
// running concurrently read.
#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<&'static Settings>> =
        const { std::cell::Cell::new(None) };
}

/// Makes [`active`] return `settings` for the current thread only.
///
/// The payload is leaked to keep [`active`]'s `&'static` contract; that is
/// bounded and harmless in a test process.
#[cfg(test)]
pub fn install_for_test(settings: Settings) {
    TEST_OVERRIDE.with(|c| c.set(Some(Box::leak(Box::new(settings)))));
}

/// The process-wide settings, or the built-in defaults before [`install`].
#[must_use]
pub fn active() -> &'static Settings {
    static FALLBACK: OnceLock<Settings> = OnceLock::new();
    #[cfg(test)]
    if let Some(s) = TEST_OVERRIDE.with(std::cell::Cell::get) {
        return s;
    }
    // References are `Copy`, so the `&'static` escapes the read guard cleanly.
    if let Some(s) = *ACTIVE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        s
    } else {
        FALLBACK.get_or_init(Settings::default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_json(text: &str) -> Settings {
        let mut s = Settings::default();
        s.overlay(text);
        s
    }

    #[test]
    fn defaults_match_the_previously_hardcoded_constants() {
        let s = Settings::default();
        assert!(s.ui.respect_gitignore);
        assert_eq!(s.ui.popup_rows, 15);
        assert_eq!(s.ui.index_refresh_secs, 5);
        assert_eq!(s.ui.history_size, 512);
        assert_eq!(s.mcp.timeout_secs, 30);
        assert_eq!(s.engine.model, None);
        assert_eq!(s.safety.sandbox, None);
        assert!(!s.memory.gate);
        assert_eq!(s.memory.gate_percent, 60);
        assert_eq!(s.memory.held_span_cap, 0);
        assert!(s.suggestions.enabled);
        assert_eq!(s.suggestions.max_tokens, 160);
        assert_eq!(s.suggestions.memory_starvation_seconds, 300);
    }

    #[test]
    fn the_memory_gate_defaults_to_off_and_holds_nothing() {
        let s = Settings::default();
        assert!(!s.memory.gate, "the gate ships off");
        assert_eq!(s.memory.gate_percent, 60);
        assert_eq!(s.memory.held_span_cap, 0, "no holding by default");
    }

    #[test]
    fn the_memory_gate_keys_are_read_from_json() {
        let s =
            from_json(r#"{ "memory": { "gate": true, "gatePercent": 80, "heldSpanCap": 12 } }"#);
        assert!(s.memory.gate);
        assert_eq!(s.memory.gate_percent, 80);
        assert_eq!(s.memory.held_span_cap, 12);
    }

    #[test]
    fn an_out_of_range_gate_percent_is_clamped_rather_than_rejected() {
        let s = from_json(r#"{ "memory": { "gatePercent": 400 } }"#);
        assert_eq!(s.memory.gate_percent, 100);
    }

    #[test]
    fn the_kvcache_budget_defaults_to_twenty_gigabytes() {
        // A real ceiling, not a warning: the TTL sweep alone puts no upper
        // bound on the cache.
        assert_eq!(Settings::default().kvcache.max_bytes, 21_474_836_480);
    }

    #[test]
    fn kvcache_block_overlays_and_defaults() {
        let mut s = Settings::default();
        assert_eq!(s.kvcache.ttl_session_days, 14);
        assert_eq!(s.kvcache.ttl_tier_days, 30);
        assert_eq!(s.kvcache.max_bytes, 21_474_836_480, "20 GB by default");

        s.overlay(r#"{"kvcache":{"ttlSessionDays":7,"ttlTierDays":60,"maxBytes":21474836480}}"#);
        assert_eq!(s.kvcache.ttl_session_days, 7);
        assert_eq!(s.kvcache.ttl_tier_days, 60);
        assert_eq!(s.kvcache.max_bytes, 21_474_836_480);
    }

    #[test]
    fn a_bad_kvcache_value_does_not_discard_its_siblings() {
        let mut s = Settings::default();
        s.overlay(r#"{"kvcache":{"ttlSessionDays":"soon","ttlTierDays":60}}"#);
        assert_eq!(
            s.kvcache.ttl_session_days, 14,
            "bad value keeps the default"
        );
        assert_eq!(s.kvcache.ttl_tier_days, 60, "sibling still applies");
    }

    #[test]
    fn reads_every_group() {
        let s = from_json(
            r#"{"engine":{"threads":8,"backend":"cpu","power":80,"ctx":262144},
                "ui":{"respectGitignore":false,"popupRows":25,
                      "indexRefreshSecs":30,"historySize":4096},
                "safety":{"sandbox":true,"btwSuspend":false},
                "mcp":{"timeoutSecs":90},
                "ask":{"maxOptions":10}}"#,
        );
        assert_eq!(s.ask.max_options, 10);
        assert_eq!(s.engine.threads, Some(8));
        assert_eq!(s.engine.backend.as_deref(), Some("cpu"));
        assert_eq!(s.engine.power, Some(80));
        assert_eq!(s.engine.ctx, Some(262_144));
        assert!(!s.ui.respect_gitignore);
        assert_eq!(s.ui.popup_rows, 25);
        assert_eq!(s.ui.index_refresh_secs, 30);
        assert_eq!(s.ui.history_size, 4096);
        assert_eq!(s.safety.sandbox, Some(true));
        assert_eq!(s.safety.btw_suspend, Some(false));
        assert_eq!(s.mcp.timeout_secs, 90);
    }

    #[test]
    fn tool_display_is_off_by_default_and_opt_in() {
        let d = Settings::default();
        assert!(!d.ui.show_tool_calls, "tool calls hidden by default");
        assert!(!d.ui.show_tool_results, "tool results hidden by default");
        let s = from_json(r#"{"ui":{"showToolCalls":true,"showToolResults":true}}"#);
        assert!(s.ui.show_tool_calls);
        assert!(s.ui.show_tool_results);
        // Surfaced in the startup note only when turned on.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("showToolCalls=true"), "{note}");
        assert!(note.contains("showToolResults=true"), "{note}");
        assert_eq!(note_for(&Settings::default(), &[]), None);
    }

    #[test]
    fn a_legacy_tools_key_is_ignored_without_erroring() {
        // `tools.task` / `tools.agent` / `tools.planMode` were removed when the
        // three tools became unconditional. An old settings file that still
        // carries them must load like any other file with unknown keys.
        let s = from_json(
            r#"{"tools":{"task":true,"agent":true,"planMode":true},"ask":{"maxOptions":5}}"#,
        );
        assert_eq!(s.ask.max_options, 5);
        let note = note_for(&s, &[]).expect("a note");
        assert!(
            !note.contains("tools."),
            "legacy key leaked into the note: {note}"
        );
    }

    #[test]
    fn the_screensaver_face_defaults_to_matrix_and_accepts_its_spellings() {
        use crate::arcade::ScreensaverFace;

        // The rain is what an untouched install shows.
        assert_eq!(
            Settings::default().ui.screensaver_face,
            ScreensaverFace::Matrix
        );

        for (text, want) in [
            ("matrix", ScreensaverFace::Matrix),
            ("rain", ScreensaverFace::Matrix),
            ("random", ScreensaverFace::Random),
            ("either", ScreensaverFace::Random),
            ("  MATRIX  ", ScreensaverFace::Matrix),
        ] {
            let mut s = Settings::default();
            s.overlay(&format!("{{\"ui\":{{\"screensaverFace\":\"{text}\"}}}}"));
            assert_eq!(s.ui.screensaver_face, want, "parsing {text:?}");
            assert_eq!(s.ui.screensaver_face_plugin, None, "parsing {text:?}");
        }

        // A plugin face is kept verbatim for the opener to resolve: the
        // installed components are not known at settings-parse time.
        let mut s = Settings::default();
        s.overlay("{\"ui\":{\"screensaverFace\":\"screensavers:starfield\"}}");
        assert_eq!(
            s.ui.screensaver_face_plugin.as_deref(),
            Some("screensavers:starfield")
        );
        // And selecting a built-in again clears it.
        s.overlay("{\"ui\":{\"screensaverFace\":\"matrix\"}}");
        assert_eq!(s.ui.screensaver_face_plugin, None);

        // An unusable value leaves the default in place rather than breaking
        // the rest of the file, like every other key here.
        let mut s = Settings::default();
        s.overlay(r#"{"ui":{"screensaverFace":"lava lamp","crtOff":false}}"#);
        assert_eq!(s.ui.screensaver_face, ScreensaverFace::Matrix);
        assert!(!s.ui.crt_off, "one bad key must not discard the others");

        // The delay and the face are independent: setting one leaves the other.
        let mut s = Settings::default();
        s.overlay(r#"{"ui":{"screensaverFace":"starfield"}}"#);
        assert_eq!(s.ui.screensaver, crate::arcade::ScreensaverDelay::M1);
    }

    #[test]
    fn screensaver_defaults_to_one_minute_and_accepts_the_four_choices() {
        use crate::arcade::ScreensaverDelay;
        use std::time::Duration;

        let s = Settings::default();
        assert_eq!(s.ui.screensaver, ScreensaverDelay::M1);
        assert_eq!(s.ui.screensaver.duration(), Some(Duration::from_mins(1)));

        for (text, want) in [
            ("1m", ScreensaverDelay::M1),
            ("2m", ScreensaverDelay::M2),
            ("5m", ScreensaverDelay::M5),
            ("never", ScreensaverDelay::Never),
        ] {
            let mut s = Settings::default();
            s.overlay(&format!(r#"{{"ui":{{"screensaver":"{text}"}}}}"#));
            assert_eq!(s.ui.screensaver, want, "parsing {text}");
        }

        // Never means never: no idle stretch ever elapses.
        assert_eq!(ScreensaverDelay::Never.duration(), None);

        // An unknown value leaves the default rather than switching it off.
        let mut s = Settings::default();
        s.overlay(r#"{"ui":{"screensaver":"7m"}}"#);
        assert_eq!(s.ui.screensaver, ScreensaverDelay::M1);
    }

    #[test]
    fn show_thinking_defaults_off_and_can_be_turned_on() {
        assert!(
            !Settings::default().ui.show_thinking,
            "thinking hidden (mirrored to the debug console) by default"
        );
        let s = from_json(r#"{"ui":{"showThinking":true}}"#);
        assert!(s.ui.show_thinking);
        // Only the non-default (on) value is surfaced in the startup note.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("showThinking=true"), "{note}");
    }

    #[test]
    fn ask_max_options_defaults_to_seven_and_clamps_up_to_the_minimum() {
        assert_eq!(Settings::default().ask.max_options, 7);
        // A max below the fixed minimum of 2 would make every ask impossible;
        // it clamps up rather than breaking the tool.
        assert_eq!(from_json(r#"{"ask":{"maxOptions":1}}"#).ask.max_options, 2);
        assert_eq!(from_json(r#"{"ask":{"maxOptions":0}}"#).ask.max_options, 2);
        assert_eq!(
            from_json(r#"{"ask":{"maxOptions":12}}"#).ask.max_options,
            12
        );
    }

    #[test]
    fn a_later_file_overlays_only_the_keys_it_sets() {
        let mut s = from_json(r#"{"ui":{"popupRows":25,"historySize":4096}}"#);
        s.overlay(r#"{"ui":{"popupRows":5}}"#);
        assert_eq!(s.ui.popup_rows, 5, "later file wins");
        assert_eq!(s.ui.history_size, 4096, "untouched key survives");
    }

    #[test]
    fn malformed_json_leaves_the_defaults_intact() {
        // A broken settings file must not stop plank from starting.
        for bad in ["", "{", "not json at all", "[]", "null"] {
            assert_eq!(from_json(bad), Settings::default(), "input {bad:?}");
        }
    }

    #[test]
    fn a_wrongly_typed_value_falls_back_to_its_default() {
        let s = from_json(r#"{"ui":{"popupRows":"lots","respectGitignore":"yes"},"mcp":{}}"#);
        assert_eq!(s.ui.popup_rows, 15);
        assert!(s.ui.respect_gitignore);
        assert_eq!(s.mcp.timeout_secs, 30);
    }

    #[test]
    fn zero_and_negative_sizes_are_rejected_rather_than_disabling_the_feature() {
        let s = from_json(r#"{"ui":{"popupRows":0,"historySize":-3},"mcp":{"timeoutSecs":0}}"#);
        assert_eq!(s.ui.popup_rows, 15);
        assert_eq!(s.ui.history_size, 512);
        assert_eq!(s.mcp.timeout_secs, 30);
    }

    #[test]
    fn unknown_keys_are_ignored() {
        let s = from_json(r#"{"ui":{"popupRows":7,"futureKey":1},"newGroup":{"x":2}}"#);
        assert_eq!(s.ui.popup_rows, 7);
    }

    #[test]
    fn a_non_finite_number_does_not_saturate_into_a_value() {
        // `as i64` on a huge f64 saturates rather than failing, so the guard
        // has to reject it before the cast.
        let s = from_json(r#"{"ui":{"popupRows":1e309}}"#);
        assert_eq!(s.ui.popup_rows, 15);
    }

    fn note_for(s: &Settings, args: &[&str]) -> Option<String> {
        let flags: Vec<String> = args.iter().map(ToString::to_string).collect();
        let cfg = crate::config::parse_options_with(s, &flags).unwrap();
        startup_note(s, &cfg)
    }

    #[test]
    fn no_note_when_settings_change_nothing() {
        assert_eq!(note_for(&Settings::default(), &[]), None);
    }

    #[test]
    fn the_note_names_the_slow_settings() {
        // The exact situation that made plank mysteriously slow: a settings
        // file quietly moved it off Metal onto the CPU.
        let s = from_json(r#"{"engine":{"backend":"cpu","threads":3,"ctx":65536}}"#);
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("backend=cpu"), "{note}");
        assert!(note.contains("threads=3"), "{note}");
        assert!(note.contains("ctx=65536"), "{note}");
    }

    #[test]
    fn a_setting_a_flag_overrode_is_not_reported() {
        // The note must describe what is in force, never what a file asked
        // for: reporting `backend=cpu` while running on Metal would send
        // someone chasing the wrong cause.
        let s = from_json(r#"{"engine":{"backend":"cpu","threads":3}}"#);
        let note = note_for(&s, &["--metal"]).expect("threads still applies");
        assert!(!note.contains("backend"), "{note}");
        assert!(note.contains("threads=3"), "{note}");
        assert_eq!(note_for(&s, &["--metal", "-t", "16"]), None);
    }

    #[test]
    fn ui_and_mcp_keys_are_reported_since_no_flag_can_override_them() {
        let s = from_json(r#"{"ui":{"popupRows":4,"historySize":7},"mcp":{"timeoutSecs":45}}"#);
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("popupRows=4"), "{note}");
        assert!(note.contains("historySize=7"), "{note}");
        assert!(note.contains("timeoutSecs=45"), "{note}");
    }

    /// A key that parses but never serialises is lost the next time anything
    /// saves the settings, silently and permanently. Parsing tests cannot
    /// catch that, so the three gate keys get a real write-then-read-back.
    #[test]
    fn the_memory_gate_keys_survive_a_save_and_reload() {
        let dir = std::env::temp_dir().join(format!("plank-cfg-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let _ = std::fs::remove_file(&path);

        let mut s = Settings::default();
        s.memory.gate = true;
        s.memory.gate_percent = 85;
        s.memory.held_span_cap = 7;
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let back = from_json(&text);
        assert!(back.memory.gate, "gate lost on save:\n{text}");
        assert_eq!(back.memory.gate_percent, 85, "gatePercent lost:\n{text}");
        assert_eq!(back.memory.held_span_cap, 7, "heldSpanCap lost:\n{text}");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn suggestions_are_on_by_default() {
        let s = Settings::default();
        assert!(
            s.suggestions.enabled,
            "on by default, with the cold-KV skip carrying the cost"
        );
        assert_eq!(s.suggestions.max_tokens, 160);
        assert_eq!(s.suggestions.memory_starvation_seconds, 300);
    }

    #[test]
    fn the_suggestion_keys_are_read_from_json() {
        let s = from_json(
            r#"{ "suggestions": { "enabled": false, "maxTokens": 24, "memoryStarvationSeconds": 60 } }"#,
        );
        assert!(!s.suggestions.enabled);
        assert_eq!(s.suggestions.max_tokens, 24);
        assert_eq!(s.suggestions.memory_starvation_seconds, 60);
    }

    /// A key that parses but never serialises is lost on the next save, silently.
    #[test]
    fn the_suggestion_keys_survive_a_save_and_reload() {
        let dir = std::env::temp_dir().join(format!("plank-cfg-sugg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let _ = std::fs::remove_file(&path);

        let mut s = Settings::default();
        s.suggestions.enabled = false;
        s.suggestions.max_tokens = 17;
        s.suggestions.memory_starvation_seconds = 42;
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let back = from_json(&text);
        assert!(!back.suggestions.enabled, "enabled lost on save:\n{text}");
        assert_eq!(back.suggestions.max_tokens, 17, "maxTokens lost:\n{text}");
        assert_eq!(
            back.suggestions.memory_starvation_seconds, 42,
            "starvation lost:\n{text}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn save_to_round_trips_and_preserves_unknown_keys() {
        let dir = std::env::temp_dir().join(format!("plank-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        // Seed a file with a key this binary does not know about.
        std::fs::write(&path, "{\"future\":{\"nope\":1},\"ui\":{\"popupRows\":3}}").unwrap();

        let mut s = Settings::default();
        s.ui.show_thinking = false;
        s.ui.popup_rows = 9;
        s.mcp.timeout_secs = 45;
        s.engine.ctx = Some(8192);
        s.engine.backend = None; // unset -> absent
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("\"future\""),
            "unknown section preserved:\n{text}"
        );
        assert!(!text.contains("backend"), "unset optional omitted");

        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert!(!reloaded.ui.show_thinking);
        assert_eq!(reloaded.ui.popup_rows, 9);
        assert_eq!(reloaded.mcp.timeout_secs, 45);
        assert_eq!(reloaded.engine.ctx, Some(8192));
        assert_eq!(reloaded.engine.backend, None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_to_persists_every_tools_field() {
        // Regression test for the bug where `save_to` wrote only
        // `tools.loopGuards` (and, before that commit, nothing at all under
        // `tools`) while `/config tools.<key> ...` claimed success. Every
        // field of `ToolsSettings` must round-trip through disk.
        let dir = std::env::temp_dir().join(format!("plank-cfg-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        // Flip every field away from its default.
        s.tools.repeat_advisory = !s.tools.repeat_advisory;
        s.tools.loop_guards = !s.tools.loop_guards;
        s.tools.call_timeout_sec = 42;
        s.tools.spill_max_bytes = 777;
        s.tools.spill_preview_bytes = 123;
        s.tools.recall = !s.tools.recall;
        s.tools.fanout = !s.tools.fanout;
        s.tools.run_code = !s.tools.run_code;
        s.tools.bash_notify = !s.tools.bash_notify;
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);

        assert_eq!(reloaded.tools.repeat_advisory, s.tools.repeat_advisory);
        assert_eq!(reloaded.tools.loop_guards, s.tools.loop_guards);
        assert_eq!(reloaded.tools.call_timeout_sec, s.tools.call_timeout_sec);
        assert_eq!(reloaded.tools.spill_max_bytes, s.tools.spill_max_bytes);
        assert_eq!(
            reloaded.tools.spill_preview_bytes,
            s.tools.spill_preview_bytes
        );
        assert_eq!(reloaded.tools.recall, s.tools.recall);
        assert_eq!(reloaded.tools.fanout, s.tools.fanout);
        assert_eq!(reloaded.tools.run_code, s.tools.run_code);
        assert_eq!(reloaded.tools.bash_notify, s.tools.bash_notify);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_command_persists_tools_fields_to_disk() {
        // Drives the same path `/config tools.<key> <value>` uses
        // (`configform::set_from_path`) rather than calling `save_to`
        // directly, so a regression in the command's own persistence step
        // (not just in `save_to`) is caught too.
        use crate::configform::set_from_path;

        let dir = std::env::temp_dir().join(format!("plank-cfg-cmd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        set_from_path(&mut s, "tools.fanout", "false").unwrap();
        set_from_path(&mut s, "tools.recall", "false").unwrap();
        set_from_path(&mut s, "tools.repeatAdvisory", "false").unwrap();
        set_from_path(&mut s, "tools.callTimeoutSec", "30").unwrap();
        set_from_path(&mut s, "tools.spillMaxBytes", "2048").unwrap();
        set_from_path(&mut s, "tools.spillPreviewBytes", "256").unwrap();
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);

        assert!(!reloaded.tools.fanout);
        assert!(!reloaded.tools.recall);
        assert!(!reloaded.tools.repeat_advisory);
        assert_eq!(reloaded.tools.call_timeout_sec, 30);
        assert_eq!(reloaded.tools.spill_max_bytes, 2048);
        assert_eq!(reloaded.tools.spill_preview_bytes, 256);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn install_and_reinstall_push_notification_mode() {
        // Regression test: `ui.notifications` used to be seeded once at
        // startup by two call sites in ui.rs and never re-pushed, so a
        // `/config` change had no effect until restart. `install`/`reinstall`
        // are now the one choke point, mirroring `reducedMotion`.
        use crate::notify::{self, NotifyMode};

        let mut s = Settings::default();
        s.ui.notifications = NotifyMode::Never;
        install(s);
        assert_eq!(notify::mode(), NotifyMode::Never, "install seeds the mode");

        let mut s = Settings::default();
        s.ui.notifications = NotifyMode::Unfocused;
        reinstall(s);
        assert_eq!(
            notify::mode(),
            NotifyMode::Unfocused,
            "reinstall re-seeds the mode live"
        );

        // `notify::MODE` is a process-wide global; leave it at the default so
        // other tests in this process are not affected by test order.
        notify::set_mode(NotifyMode::Always);
    }

    #[test]
    fn active_is_the_defaults_until_installed() {
        // Tests never call `install`, so every consumer sees the defaults.
        assert_eq!(active().ui.popup_rows, 15);
    }

    #[test]
    fn notification_defaults_and_overlay() {
        use crate::notify::NotifyMode;
        let s = Settings::default();
        assert_eq!(s.ui.notifications, NotifyMode::Always);
        assert_eq!(s.ui.notify_after_secs, 10);

        // Legacy boolean spelling still parses.
        let mut s = Settings::default();
        s.overlay(r#"{ "ui": { "notifications": false, "notifyAfterSecs": 30 } }"#);
        assert_eq!(s.ui.notifications, NotifyMode::Never);
        assert_eq!(s.ui.notify_after_secs, 30);

        let mut s = Settings::default();
        s.overlay(r#"{ "ui": { "notifications": "unfocused" } }"#);
        assert_eq!(s.ui.notifications, NotifyMode::Unfocused);
        // A bad mode string is ignored, keeping the default.
        s.overlay(r#"{ "ui": { "notifications": "sometimes" } }"#);
        assert_eq!(s.ui.notifications, NotifyMode::Unfocused);

        // Bad value ignored, default retained.
        let mut s = Settings::default();
        s.overlay(r#"{ "ui": { "notifyAfterSecs": "nope" } }"#);
        assert_eq!(s.ui.notify_after_secs, 10);
    }

    #[test]
    fn crt_off_defaults_on_and_can_be_turned_off() {
        assert!(Settings::default().ui.crt_off, "default is on");

        let s = from_json(r#"{"ui":{"crtOff":false}}"#);
        assert!(!s.ui.crt_off);

        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("crtOff=false"), "{note}");
    }

    #[test]
    fn easter_eggs_default_on_and_can_be_turned_off() {
        assert!(Settings::default().ui.easter_eggs, "default is on");

        let s = from_json(r#"{"ui":{"easterEggs":false}}"#);
        assert!(!s.ui.easter_eggs);

        // Only the non-default (off) value surfaces in the startup note, so a
        // settings file that quietly removes the games cannot hide.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("easterEggs=false"), "{note}");
        assert!(
            note_for(&Settings::default(), &[]).is_none(),
            "the default surfaced a note"
        );

        // A non-boolean is ignored rather than read as off.
        let mut bad = Settings::default();
        bad.overlay(r#"{"ui":{"easterEggs":"nope"}}"#);
        assert!(bad.ui.easter_eggs, "a bad value disabled the arcade");
    }

    #[test]
    fn builtin_editor_defaults_on_and_overlays_off() {
        let mut s = Settings::default();
        assert!(
            s.ui.builtin_editor,
            "Ctrl-G uses the built-in editor by default"
        );
        s.overlay(r#"{"ui":{"builtinEditor":false}}"#);
        assert!(!s.ui.builtin_editor);
    }

    #[test]
    fn update_check_defaults_on_and_can_be_disabled() {
        assert!(Settings::default().update.check, "default is on");
        let s = from_json(r#"{"update":{"check":false}}"#);
        assert!(!s.update.check);
        // Only the non-default (off) value surfaces in the startup note.
        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("update.check=false"), "{note}");
        // Round-trips through save.
        let dir = std::env::temp_dir().join(format!("plank-update-chk-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        s.save_to(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&std::fs::read_to_string(&path).unwrap());
        assert!(!reloaded.update.check);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn agent_settings_default_and_round_trip() {
        let d = Settings::default();
        assert!(d.agents.auto_route, "model routing on by default");
        assert_eq!(d.agents.max_parallel, 4);

        let s = from_json(r#"{"agents":{"autoRoute":false,"maxParallel":8}}"#);
        assert!(!s.agents.auto_route);
        assert_eq!(s.agents.max_parallel, 8);

        let note = note_for(&s, &[]).expect("a note");
        assert!(note.contains("agents.autoRoute=false"), "{note}");
        assert!(note.contains("agents.maxParallel=8"), "{note}");

        let dir = std::env::temp_dir().join(format!("plank-agent-set-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        // An unrelated top-level key must survive the write.
        std::fs::write(&path, r#"{"unknownTop":{"keep":1}}"#).unwrap();
        s.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("unknownTop"),
            "unknown keys preserved: {text}"
        );
        let back = from_json(&text);
        assert!(!back.agents.auto_route);
        assert_eq!(back.agents.max_parallel, 8);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn max_parallel_clamps_out_of_range_values() {
        assert_eq!(
            from_json(r#"{"agents":{"maxParallel":0}}"#)
                .agents
                .max_parallel,
            1,
            "zero would disable the feature silently; clamp up"
        );
        assert_eq!(
            from_json(r#"{"agents":{"maxParallel":999}}"#)
                .agents
                .max_parallel,
            AGENT_MAX_PARALLEL
        );
        // A non-numeric value leaves the default standing rather than poisoning
        // it — same policy as every other numeric key.
        assert_eq!(
            from_json(r#"{"agents":{"maxParallel":"lots"}}"#)
                .agents
                .max_parallel,
            4
        );
    }

    #[test]
    fn crt_off_round_trips_through_save() {
        let dir = std::env::temp_dir().join(format!("plank-crt-off-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        s.ui.crt_off = false;
        s.save_to(&path).unwrap();

        let mut reloaded = Settings::default();
        let text = std::fs::read_to_string(&path).unwrap();
        reloaded.overlay(&text);
        assert!(!reloaded.ui.crt_off);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn thinking_tool_calls_defaults_off_and_overlays_on() {
        let mut s = Settings::default();
        assert!(
            !s.engine.thinking_tool_calls,
            "tool calls inside <think> are ignored by default (C parity)"
        );
        s.overlay(r#"{"engine":{"thinkingToolCalls":true}}"#);
        assert!(s.engine.thinking_tool_calls);
        // A non-boolean value is ignored rather than flipping the default.
        let mut s2 = Settings::default();
        s2.overlay(r#"{"engine":{"thinkingToolCalls":"nope"}}"#);
        assert!(!s2.engine.thinking_tool_calls);
    }

    #[test]
    fn memory_settings_default_pass_on_tools_on_and_round_trip() {
        use crate::configform::set_from_path;

        // The passive pass is on by default since 5.1.7, as are the
        // `remember`/`forget` tools.
        let s = Settings::default();
        assert!(
            s.memory.auto_extract,
            "the extraction pass must default to on"
        );
        assert_eq!(s.memory.extract_every_n_turns, 1);
        assert!(s.tools.remember);
        assert_eq!(s.memory.budgets, crate::memory::Budgets::default());

        // Every value set here is the non-default one, so a stale default
        // cannot make the assertions vacuously true.
        let mut s = Settings::default();
        set_from_path(&mut s, "memory.autoExtract", "false").unwrap();
        set_from_path(&mut s, "memory.extractEveryNTurns", "4").unwrap();
        set_from_path(&mut s, "memory.minTurnSeconds", "45").unwrap();
        set_from_path(&mut s, "tools.remember", "false").unwrap();
        assert!(!s.memory.auto_extract);
        assert_eq!(s.memory.extract_every_n_turns, 4);
        assert_eq!(s.memory.min_turn_seconds, 45);
        assert!(!s.tools.remember);
    }

    #[test]
    fn memory_settings_overlay_from_json() {
        let mut s = Settings::default();
        s.overlay(
            r#"{"memory":{"autoExtract":false,"extractEveryNTurns":7,"budgets":{"user":10,"feedback":20,"project":30,"reference":40}},"tools":{"remember":false}}"#,
        );
        assert!(
            !s.memory.auto_extract,
            "overlay must flip the on default off"
        );
        assert_eq!(s.memory.extract_every_n_turns, 7);
        assert_eq!(s.memory.budgets.user, 10);
        assert_eq!(s.memory.budgets.feedback, 20);
        assert_eq!(s.memory.budgets.project, 30);
        assert_eq!(s.memory.budgets.reference, 40);
        assert!(!s.tools.remember);
    }

    #[test]
    fn memory_extract_every_n_turns_overlay_clamps_zero_to_one() {
        // The default is already 1, so overlaying anything other than 0
        // would leave this vacuously true. Use a non-1 baseline, overlay 0,
        // and confirm the clamp — not the default — produced the 1.
        let mut s = Settings::default();
        s.memory.extract_every_n_turns = 9;
        s.overlay(r#"{"memory":{"extractEveryNTurns":0}}"#);
        assert_eq!(
            s.memory.extract_every_n_turns, 1,
            "0 must clamp to 1, not pass through"
        );
    }

    #[test]
    fn memory_min_turn_seconds_defaults_to_thirty_seconds() {
        let s = Settings::default();
        assert_eq!(
            s.memory.min_turn_seconds, 30,
            "a turn shorter than thirty seconds must not trigger the pass by default"
        );
    }

    #[test]
    fn memory_min_turn_seconds_overlay_keeps_zero() {
        // 0 means "no floor" and is a real value here, unlike
        // extractEveryNTurns where 0 clamps to 1. Start from a non-zero,
        // non-default baseline so neither the default nor a clamp can make
        // this vacuously true.
        let mut s = Settings::default();
        s.memory.min_turn_seconds = 45;
        s.overlay(r#"{"memory":{"minTurnSeconds":0}}"#);
        assert_eq!(
            s.memory.min_turn_seconds, 0,
            "0 must pass through as 'no floor', not clamp"
        );

        let mut s = Settings::default();
        s.overlay(r#"{"memory":{"minTurnSeconds":300}}"#);
        assert_eq!(s.memory.min_turn_seconds, 300);
    }

    #[test]
    fn memory_gate_bias_defaults_to_zero_and_leaves_the_threshold_alone() {
        let s = Settings::default();
        assert_eq!(s.memory.gate_bias, GateBias::default());
        for f in [
            crate::gguf::ModelFamily::Ds4,
            crate::gguf::ModelFamily::Ds41,
            crate::gguf::ModelFamily::Qwen,
        ] {
            assert_eq!(s.memory.gate_threshold_percent(f), 60);
        }
    }

    #[test]
    fn memory_gate_bias_overlay_is_per_family_signed_and_clamped() {
        let mut s = Settings::default();
        s.overlay(r#"{"memory":{"gateBias":{"ds4":7,"qwen":-4,"ds41":500}}}"#);
        assert_eq!(s.memory.gate_bias.ds4, 7);
        assert_eq!(s.memory.gate_bias.qwen, -4);
        assert_eq!(s.memory.gate_bias.ds41, GateBias::MAX_ABS);
        assert_eq!(
            s.memory
                .gate_threshold_percent(crate::gguf::ModelFamily::Ds4),
            67
        );
        assert_eq!(
            s.memory
                .gate_threshold_percent(crate::gguf::ModelFamily::Qwen),
            56
        );
        // 60 + 50 clamps to 100, not 110.
        assert_eq!(
            s.memory
                .gate_threshold_percent(crate::gguf::ModelFamily::Ds41),
            100
        );
    }

    #[test]
    fn memory_gate_bias_round_trips_through_save_to_and_is_absent_when_zero() {
        let dir = std::env::temp_dir().join(format!(
            "plank-gate-bias-cfg-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let s = Settings::default();
        s.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("gateBias"), "{text}");

        let mut s = Settings::default();
        s.memory.gate_bias.ds4 = 7;
        s.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert_eq!(reloaded.memory.gate_bias.ds4, 7);
        assert_eq!(reloaded.memory.gate_bias.qwen, 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn memory_min_turn_seconds_round_trips_through_save_to() {
        let dir = std::env::temp_dir().join(format!(
            "plank-memory-floor-cfg-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        s.memory.min_turn_seconds = 90; // the non-default value
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert_eq!(reloaded.memory.min_turn_seconds, 90);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn memory_settings_round_trip_through_save_to() {
        let dir = std::env::temp_dir().join(format!("plank-memory-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        s.memory.auto_extract = false; // the non-default value
        s.memory.extract_every_n_turns = 6;
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert!(!reloaded.memory.auto_extract);
        assert_eq!(reloaded.memory.extract_every_n_turns, 6);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn microcompact_defaults_true_and_overlays_false() {
        let s = Settings::default();
        assert!(s.context.microcompact, "on by default");
        let mut s2 = Settings::default();
        s2.overlay(r#"{"context":{"microcompact":false}}"#);
        assert!(!s2.context.microcompact);
        // A non-boolean value is ignored rather than flipping the default.
        let mut s3 = Settings::default();
        s3.overlay(r#"{"context":{"microcompact":"nope"}}"#);
        assert!(s3.context.microcompact);
    }

    #[test]
    fn short_reminder_defaults_true_overlays_false_and_round_trips() {
        let s = Settings::default();
        assert!(s.context.short_reminder, "on by default");
        let mut s2 = Settings::default();
        s2.overlay(r#"{"context":{"shortReminder":false}}"#);
        assert!(!s2.context.short_reminder);
        // A non-boolean value is ignored rather than flipping the default.
        let mut s3 = Settings::default();
        s3.overlay(r#"{"context":{"shortReminder":"nope"}}"#);
        assert!(s3.context.short_reminder);

        let dir = std::env::temp_dir().join(format!("plank-short-reminder-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        let mut s = Settings::default();
        s.context.short_reminder = false;
        s.save_to(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"shortReminder\": false"), "{text}");
        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert!(!reloaded.context.short_reminder);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn microcompact_round_trips_through_save_to() {
        let dir = std::env::temp_dir().join(format!("plank-microcompact-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");

        let mut s = Settings::default();
        s.context.microcompact = false;
        s.save_to(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"microcompact\": false"), "{text}");

        let mut reloaded = Settings::default();
        reloaded.overlay(&text);
        assert!(!reloaded.context.microcompact);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_from_paths_lets_high_win_over_low() {
        // This covers the primitive only: later slice wins. It does not by
        // itself prove real-world plugin/user precedence — that guarantee
        // lives in `load_with_plugins_in_never_lets_a_plugin_beat_the_user`.
        let dir = std::env::temp_dir().join(format!("plank-settings-prim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let low = dir.join("low.json");
        std::fs::write(&low, r#"{"kvcache":{"maxBytes":111}}"#).expect("write");

        let s = Settings::load_from_paths(std::slice::from_ref(&low), &[]);
        assert_eq!(s.kvcache.max_bytes, 111);

        let high = dir.join("high.json");
        std::fs::write(&high, r#"{"kvcache":{"maxBytes":222}}"#).expect("write");
        let s = Settings::load_from_paths(&[low], &[high]);
        assert_eq!(s.kvcache.max_bytes, 222);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tools_bash_notify_parses_and_defaults_off() {
        assert!(!Settings::default().tools.bash_notify);
        let dir = std::env::temp_dir().join(format!("plank-settings-bn-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("settings.json");
        std::fs::write(&path, r#"{"tools":{"bashNotify":true}}"#).expect("write");
        // The user layer: a plugin layer may not set `tools.*` at all.
        let s = Settings::load_from_paths(&[], std::slice::from_ref(&path));
        assert!(s.tools.bash_notify);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_plugin_layer_cannot_set_engine_worktree_tools_or_plugin_config() {
        let plugin_text = r#"{"engine":{"model":"/evil.gguf","backend":"cpu","threads":3},
            "worktree":{"symlinkDirectories":["node_modules"]},
            "tools":{"runCode":true,"repeatAdvisory":false},
            "pluginConfig":{"k":"v"},
            "ui":{"popupRows":9},"kvcache":{"maxBytes":111}}"#;
        let base = Settings::default();
        let mut s = Settings::default();
        s.overlay_from(
            plugin_text,
            &crate::provenance::Origin::Plugin("demo".into()),
        );
        assert_eq!(s.engine, base.engine, "engine.* from a plugin is refused");
        assert_eq!(
            s.worktree, base.worktree,
            "worktree.* from a plugin is refused"
        );
        assert_eq!(s.tools, base.tools, "tools.* from a plugin is refused");
        assert!(
            s.plugin_config.is_empty(),
            "pluginConfig from a plugin is refused"
        );
        assert_eq!(s.ui.popup_rows, 9, "cosmetic keys still apply");
        assert_eq!(s.kvcache.max_bytes, 111);
        assert!(
            !s.provenance.contains_key("engine.model"),
            "a refused key leaves no provenance: {:?}",
            s.provenance.keys().collect::<Vec<_>>()
        );

        // The same text from a user file is applied in full: the refusal is
        // about the layer, not the keys.
        let mut u = Settings::default();
        u.overlay_from(plugin_text, &crate::provenance::Origin::UserSettings);
        assert_eq!(u.engine.threads, Some(3));
        assert_eq!(u.plugin_config.get("k").map(String::as_str), Some("v"));
    }

    #[test]
    fn a_plugin_settings_file_does_not_change_the_effective_model() {
        let dir =
            std::env::temp_dir().join(format!("plank-settings-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("home");
        let cwd = dir.join("cwd");
        std::fs::create_dir_all(home.join(".plank")).expect("mkdir home");
        std::fs::create_dir_all(&cwd).expect("mkdir cwd");
        let plugin = dir.join("plugin-settings.json");
        std::fs::write(&plugin, r#"{"engine":{"model":"/evil.gguf"}}"#).expect("write plugin");

        let s = Settings::load_with_plugins_in(Some(&home), &cwd, &[plugin]);
        assert_eq!(
            s.engine.model,
            Settings::default().engine.model,
            "a plugin must not pick the model"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_with_plugins_in_never_lets_a_plugin_beat_the_user() {
        // Exercises the composed loader `load_with_plugins_in`, not the raw
        // `load_from_paths` primitive. A real user settings file lives under
        // `home/.plank/settings.json`; a plugin settings file sets the same
        // key to a different value. If the two arguments to `load_from_paths`
        // inside `load_with_plugins_in` were ever swapped (i.e. it called
        // `load_from_paths(&Self::paths_in(...), plugin_paths)` instead of
        // `load_from_paths(plugin_paths, &Self::paths_in(...))`), the plugin
        // path would land in the `high` slot and this assertion would flip to
        // seeing the plugin's value (111) instead of the user's (222) — so
        // this test fails under that swap.
        let dir = std::env::temp_dir().join(format!("plank-settings-comp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("home");
        let cwd = dir.join("cwd");
        std::fs::create_dir_all(home.join(".plank")).expect("mkdir home");
        std::fs::create_dir_all(&cwd).expect("mkdir cwd");

        let user = home.join(".plank").join("settings.json");
        std::fs::write(&user, r#"{"kvcache":{"maxBytes":222}}"#).expect("write user");

        let plugin = dir.join("plugin-settings.json");
        std::fs::write(&plugin, r#"{"kvcache":{"maxBytes":111}}"#).expect("write plugin");

        let s = Settings::load_with_plugins_in(Some(&home), &cwd, &[plugin]);
        assert_eq!(
            s.kvcache.max_bytes, 222,
            "the user's ~/.plank/settings.json must win over a plugin's setting"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_with_plugins_in_lets_the_later_plugin_win() {
        // "plugin settings, in plugin load order, later winning" — two
        // plugin-contributed files disagreeing on the same key, no user file
        // in play at all.
        let dir = std::env::temp_dir().join(format!("plank-settings-multi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("home");
        let cwd = dir.join("cwd");
        std::fs::create_dir_all(&home).expect("mkdir home");
        std::fs::create_dir_all(&cwd).expect("mkdir cwd");

        let plugin_a = dir.join("plugin-a-settings.json");
        std::fs::write(&plugin_a, r#"{"kvcache":{"maxBytes":111}}"#).expect("write a");
        let plugin_b = dir.join("plugin-b-settings.json");
        std::fs::write(&plugin_b, r#"{"kvcache":{"maxBytes":333}}"#).expect("write b");

        let s = Settings::load_with_plugins_in(Some(&home), &cwd, &[plugin_a, plugin_b]);
        assert_eq!(
            s.kvcache.max_bytes, 333,
            "the later plugin in load order must win over an earlier one"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provenance_records_plugin_below_user_below_project() {
        // The plugin-below-user rule from CLAUDE.md must be visible in the
        // provenance, not just in the docs: a plugin setting loses to the user
        // file, which loses to the project file, and each loser is shadowed.
        let dir = std::env::temp_dir().join(format!("plank-provenance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("home");
        let cwd = dir.join("cwd");
        std::fs::create_dir_all(home.join(".plank")).expect("mkdir home");
        std::fs::create_dir_all(cwd.join(".plank")).expect("mkdir cwd");

        let plugin = dir.join("plugin-settings.json");
        std::fs::write(&plugin, r#"{"kvcache":{"maxBytes":111}}"#).expect("write plugin");
        let user = home.join(".plank").join("settings.json");
        std::fs::write(&user, r#"{"kvcache":{"maxBytes":222}}"#).expect("write user");
        let project = cwd.join(".plank").join("settings.json");
        std::fs::write(&project, r#"{"kvcache":{"maxBytes":333}}"#).expect("write project");

        let s = Settings::load_with_plugins_in(Some(&home), &cwd, &[plugin]);
        assert_eq!(s.kvcache.max_bytes, 333);
        let p = s
            .provenance
            .get("kvcache.maxBytes")
            .expect("provenance recorded");
        assert_eq!(p.origin, crate::provenance::Origin::ProjectSettings);
        assert_eq!(
            p.shadowed,
            vec![
                crate::provenance::Origin::Plugin(String::new()),
                crate::provenance::Origin::UserSettings,
            ],
            "plugin then user, in increasing precedence, both shadowed by project"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provenance_records_a_key_only_when_it_takes_effect() {
        // A malformed value is ignored by overlay, so it must not be recorded
        // as provenance — the resolved dump lists effective keys only.
        let mut s = Settings::default();
        s.overlay(r#"{"kvcache":{"maxBytes":"soon"}}"#);
        assert!(!s.provenance.contains_key("kvcache.maxBytes"));
        assert_eq!(s.kvcache.max_bytes, 21_474_836_480);
    }
}

#[cfg(test)]
mod loop_guard_suspension_tests {
    use super::*;

    /// The suspension layers *under* the `/loopguard` override and *over* the
    /// persisted setting, and — the part that matters when a turn fails — it
    /// lifts on drop rather than on a matching call someone has to remember.
    #[test]
    fn suspension_is_scoped_and_loses_to_an_explicit_override() {
        set_loop_guards_override(None);
        assert!(loop_guards_effective(), "default is armed");

        {
            let _scope = suspend_loop_guards();
            assert!(!loop_guards_effective(), "suspended inside the scope");

            // A user who said something explicit outranks a canned prompt, in
            // both directions.
            set_loop_guards_override(Some(true));
            assert!(loop_guards_effective(), "/loopguard on wins over a scope");
            set_loop_guards_override(Some(false));
            assert!(!loop_guards_effective());
            set_loop_guards_override(None);

            // Re-entrant: an inner scope restores the outer one, not "armed".
            {
                let _inner = suspend_loop_guards();
                assert!(!loop_guards_effective());
            }
            assert!(!loop_guards_effective(), "inner drop kept the outer scope");
        }
        assert!(loop_guards_effective(), "re-armed on drop");
    }

    /// Dropping the scope on the error path is the whole reason it is a scope:
    /// a turn that fails must not leave the protection off for the session.
    #[test]
    fn suspension_lifts_even_when_the_scope_body_fails() {
        fn turn_that_fails() -> Result<(), &'static str> {
            let _scope = suspend_loop_guards();
            assert!(!loop_guards_effective());
            Err("the turn failed")?;
            unreachable!("the early return above is the point of the test")
        }

        set_loop_guards_override(None);
        assert!(turn_that_fails().is_err());
        assert!(loop_guards_effective(), "re-armed despite the early return");
    }
}

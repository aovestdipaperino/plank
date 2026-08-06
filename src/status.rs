// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal prompt, status footer, and progress rendering.
//!
//! Port of the "Terminal Prompt, Status Footer" section of `ds4_agent.c`:
//! the compact one-line footer, the prefill progress bar with an embedded
//! t/s readout, and the user prompt echo styling.

/// ANSI style opening for the status footer row.
pub const STATUS_STYLE_START: &str = "\x1b[48;5;238;38;5;252m";
/// ANSI style reset.
pub const STATUS_STYLE_END: &str = "\x1b[0m";
/// 256-color index of the theme color (military green) used for accents
/// such as the filled portion of the progress bar.
pub const THEME_COLOR: u8 = 106;

/// Shades swept across the spinner verb, ordered outermost column first so the
/// last entry lands on the center of the highlight.
///
/// These are lightness steps of the theme's own hue rather than a flat white
/// flash: pure white against the military green reads as a blown-out glitch and
/// draws the eye harder than the text it decorates, while a graded ramp reads as
/// light moving across the word. The window is
/// `2 * (SHIMMER_RAMP.len() - 1) + 1` columns wide, so adding a shade widens and
/// softens the sweep in one edit.
///
/// 143 `#afaf5f` → 149 `#afd75f` → 192 `#d7ff87`, all in the same
/// olive/green family as [`THEME_COLOR`] (106 `#87af5f`).
pub const SHIMMER_RAMP: [u8; 3] = [143, 149, 192];

/// Milliseconds per shimmer step (one display column of travel).
pub const SHIMMER_STEP_MS: u64 = 200;
/// ANSI style for the filled portion of the progress bar (theme color).
pub const STATUS_BAR_FILL: &str = "\x1b[48;5;238;38;5;106;1m";
/// ANSI style for the queued-prompt preview rows.
pub const QUEUE_STYLE: &str = "\x1b[38;5;87;1m";

/// Powerline branch glyph (U+E0A0), shown before the git branch name. Requires
/// a Powerline-patched or Nerd Font to render.
pub const POWERLINE_BRANCH: char = '\u{e0a0}';

/// Leading glyph of the **think segment** (`🧠 medium`), the reasoning level
/// shown just before the ctx gauge.
///
/// Also the anchor the TUI splits the footer on: it sits between the dir prefix
/// and the body, so [`crate::tui`] peels it as its own span rather than letting
/// `push_dir_prefix` mistake it for part of the branch name.
pub const THINK_MARK: &str = "🧠";

const PROGRESS_BAR_WIDTH: usize = 32;

/// Worker lifecycle state mirrored from `agent_worker_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerState {
    /// Waiting for user input.
    #[default]
    Idle,
    /// Prefilling prompt tokens.
    Prefill,
    /// Sampling assistant tokens.
    Generating,
    /// Summarizing the transcript to reclaim context.
    Compacting,
    /// Saving the session to disk.
    Saving,
    /// A worker error occurred; see the error text.
    Error,
    /// Generation was interrupted.
    Stopped,
}

/// Snapshot of worker progress shown in the footer, mirroring `agent_status`.
#[derive(Debug, Clone, Default)]
pub struct Status {
    /// Current worker state.
    pub state: WorkerState,
    /// Prefill tokens done.
    pub prefill_done: i32,
    /// Prefill tokens total.
    pub prefill_total: i32,
    /// Stable index selecting the playful spinner verb for this turn.
    pub prefill_label: u32,
    /// Seconds elapsed since the current operation started.
    pub elapsed_secs: f64,
    /// Prefill throughput, tokens per second.
    pub prefill_tps: f64,
    /// Tokens generated so far.
    pub generated: i32,
    /// Generation throughput, tokens per second.
    pub gen_tps: f64,
    /// True when sampling greedily (shown as a snowflake).
    pub greedy_sampling: bool,
    /// Context tokens in use.
    pub ctx_used: i32,
    /// Context window size.
    pub ctx_size: i32,
    /// Power limit percent; 0 or 100 hides the suffix.
    pub power_percent: i32,
    /// Reasoning level, shown as the think segment ahead of the ctx gauge.
    pub think: crate::engine::ThinkMode,
    /// Error text for the `Error` state.
    pub error: String,
}

/// Returns the input prompt text.
#[must_use]
pub fn prompt_text() -> &'static str {
    "🪵> "
}

/// Collapses `home` at the front of `path` to `~` (e.g. `/Users/x/Code` with
/// home `/Users/x` becomes `~/Code`). Returns `path` unchanged otherwise.
#[must_use]
fn collapse_home(path: &str, home: &str) -> String {
    if !home.is_empty() {
        if path == home {
            return "~".to_owned();
        }
        if let Some(rest) = path.strip_prefix(&format!("{home}/")) {
            return format!("~/{rest}");
        }
    }
    path.to_owned()
}

/// Current working directory with the home prefix collapsed to `~`; empty if
/// the cwd cannot be determined.
#[must_use]
pub fn cwd_label() -> String {
    let Ok(cwd) = std::env::current_dir() else {
        return String::new();
    };
    let home = std::env::var("HOME").unwrap_or_default();
    collapse_home(&cwd.to_string_lossy(), &home)
}

/// Current git branch, discovered from the cwd via `git2`. Returns the branch
/// name for a symbolic HEAD, a short commit hash for a detached HEAD, or
/// `None` when not inside a repo.
#[must_use]
pub fn git_branch_label() -> Option<String> {
    let repo = git2::Repository::discover(".").ok()?;
    let head = repo.head().ok()?;
    if head.is_branch() {
        return head.shorthand().ok().map(str::to_owned);
    }
    // Detached HEAD: fall back to the short commit hash.
    let oid = head.target()?;
    Some(oid.to_string().chars().take(7).collect())
}

/// Where inference is running, shown in the footer beside the directory.
///
/// A process-wide global rather than a [`Status`] field: it is decided once when
/// the engine is built and never changes, so threading it through the worker,
/// the remote-UI protocol, and both front-ends would be noise. Mirrors how
/// [`cwd_label`] and [`git_branch_label`] read the ambient environment.
static ENGINE_ORIGIN: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Shown when nothing set an origin: the local Metal-backed ds4 engine (and the
/// `EchoEngine` stub, which is equally local).
const LOCAL_ORIGIN: &str = "(local)";

/// Records the engine origin for the footer. First call wins; later calls are
/// ignored, so a single engine choice per process is the only thing shown.
pub fn set_engine_origin(label: &str) {
    let _ = ENGINE_ORIGIN.set(label.to_owned());
}

/// The engine-origin label: a provider or remote host, else [`LOCAL_ORIGIN`].
#[must_use]
pub fn engine_origin_label() -> &'static str {
    ENGINE_ORIGIN.get().map_or(LOCAL_ORIGIN, String::as_str)
}

/// Extracts the bare host from a base URL for display: `https://api.anthropic.com/v1`
/// becomes `api.anthropic.com`. Any credentials, port, and path are dropped —
/// this is a footer label, not something to reconstruct a request from. Returns
/// the trimmed input when it has no recognizable host.
#[must_use]
pub fn url_host(url: &str) -> String {
    let rest = url
        .split_once("://")
        .map_or(url, |(_, rest)| rest)
        .trim_start_matches('/');
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    // Strip `user:pass@` and any `:port`.
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        url.trim().to_owned()
    } else {
        host.to_owned()
    }
}

/// Formats a token count compactly: `8000` becomes `8k`, `2500` becomes `2.5k`.
#[must_use]
pub fn format_ctx_size(ctx_size: i32) -> String {
    if ctx_size >= 1_000_000 {
        if ctx_size % 1_000_000 == 0 {
            format!("{}M", ctx_size / 1_000_000)
        } else {
            format!("{:.1}M", f64::from(ctx_size) / 1_000_000.0)
        }
    } else if ctx_size >= 1000 {
        if ctx_size % 1000 == 0 {
            format!("{}k", ctx_size / 1000)
        } else {
            format!("{:.1}k", f64::from(ctx_size) / 1000.0)
        }
    } else {
        ctx_size.to_string()
    }
}

/// Claude-Code-style playful gerunds shown next to the spinner. One is picked
/// per turn (keyed by `Status::prefill_label`) so the footer does not visually
/// churn while progress updates stream in.
pub const SPINNER_VERBS: [&str; 200] = [
    "Accomplishing",
    "Actualizing",
    "Baking",
    "Bamboozling",
    "Beaming",
    "Befriending",
    "Bewitching",
    "Bloviating",
    "Boiling",
    "Boondoggling",
    "Bootstrapping",
    "Brainstorming",
    "Braising",
    "Brewing",
    "Burrowing",
    "Buzzing",
    "Calculating",
    "Calibrating",
    "Canoodling",
    "Caramelizing",
    "Cerebrating",
    "Channelling",
    "Churning",
    "Clauding",
    "Coalescing",
    "Cogitating",
    "Combobulating",
    "Composing",
    "Computing",
    "Concocting",
    "Conjuring",
    "Contemplating",
    "Cooking",
    "Crafting",
    "Creating",
    "Crunching",
    "Crystallizing",
    "Curating",
    "Deciphering",
    "Decoding",
    "Deliberating",
    "Discombobulating",
    "Distilling",
    "Divining",
    "Doodling",
    "Dreaming",
    "Effervescing",
    "Elaborating",
    "Elucidating",
    "Embellishing",
    "Enchanting",
    "Envisioning",
    "Extrapolating",
    "Fermenting",
    "Fiddling",
    "Finagling",
    "Flambéing",
    "Flourishing",
    "Fluttering",
    "Forging",
    "Formulating",
    "Frolicking",
    "Galloping",
    "Galvanizing",
    "Germinating",
    "Gesticulating",
    "Gitifying",
    "Grokking",
    "Guessing",
    "Gusting",
    "Hatching",
    "Herding",
    "Honking",
    "Hustling",
    "Hyperventilating",
    "Hypothesizing",
    "Ideating",
    "Illuminating",
    "Imagining",
    "Improvising",
    "Incubating",
    "Inferring",
    "Intuiting",
    "Jitterbugging",
    "Jiving",
    "Juggling",
    "Kerfuffling",
    "Kindling",
    "Kneading",
    "Levitating",
    "Lollygagging",
    "Macerating",
    "Manifesting",
    "Marinating",
    "Meandering",
    "Meditating",
    "Metabolizing",
    "Mind-melding",
    "Mixing",
    "Moseying",
    "Mulling",
    "Musing",
    "Mustering",
    "Mutating",
    "Nesting",
    "Noodling",
    "Normalizing",
    "Orbiting",
    "Orchestrating",
    "Osmosing",
    "Oxidizing",
    "Percolating",
    "Perusing",
    "Philosophising",
    "Photosynthesizing",
    "Pirouetting",
    "Polishing",
    "Pontificating",
    "Pondering",
    "Prognosticating",
    "Puttering",
    "Puzzling",
    "Quibbling",
    "Reticulating",
    "Riffing",
    "Ruminating",
    "Rustling",
    "Sautéing",
    "Scheming",
    "Schlepping",
    "Sculpting",
    "Searing",
    "Seasoning",
    "Shimmering",
    "Shimmying",
    "Shucking",
    "Simmering",
    "Sizzling",
    "Sketching",
    "Skedaddling",
    "Smooshing",
    "Snoozing",
    "Sparkling",
    "Spelunking",
    "Spinning",
    "Sprouting",
    "Squishing",
    "Steeping",
    "Stewing",
    "Stirring",
    "Strategizing",
    "Strutting",
    "Sublimating",
    "Summoning",
    "Swirling",
    "Swooshing",
    "Synthesizing",
    "Tinkering",
    "Toasting",
    "Transmuting",
    "Twirling",
    "Unfurling",
    "Unravelling",
    "Vibing",
    "Wandering",
    "Weaving",
    "Whirring",
    "Whisking",
    "Wibbling",
    "Wizarding",
    "Wobbling",
    "Wondering",
    "Wrangling",
    "Zesting",
    "Zigzagging",
    "Zooming",
    "Alchemizing",
    "Amalgamating",
    "Annealing",
    "Blossoming",
    "Bubbling",
    "Cascading",
    "Catalyzing",
    "Chiseling",
    "Deducing",
    "Digesting",
    "Dovetailing",
    "Etching",
    "Excavating",
    "Fathoming",
    "Gilding",
    "Harmonizing",
    "Infusing",
    "Interpolating",
    "Lassoing",
    "Navigating",
    "Quenching",
    "Scintillating",
    "Tessellating",
    "Vortexing",
];

/// Keep each operation on a single playful verb so the footer does not
/// visually churn while progress updates stream in.
#[must_use]
pub fn prefill_label(st: &Status) -> &'static str {
    SPINNER_VERBS[st.prefill_label as usize % SPINNER_VERBS.len()]
}

/// Picks a stable random verb index for a new turn, seeded from wall-clock.
#[must_use]
pub fn random_verb_index() -> u32 {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    #[allow(clippy::cast_possible_truncation)]
    let seed = (ms ^ (ms >> 17)) as u32;
    seed % u32::try_from(SPINNER_VERBS.len()).unwrap_or(1)
}

/// Formats elapsed seconds Claude-Code style: `12s`, `1m 2s`, `1h 4m`.
#[must_use]
pub fn format_elapsed(secs: f64) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let s = if secs.is_finite() && secs > 0.0 {
        secs as u64
    } else {
        0
    };
    if s >= 3600 {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// How long each rotating status-bar tip stays up before the next one.
pub const TIP_ROTATE_MS: u64 = 120_000;

/// Rotating one-line hints shown (in yellow) at the tail of the TUI status bar.
/// Each points at a real command or key binding; keep them short — the bar
/// truncates on narrow terminals and the tip is last.
pub const TIPS: &[&str] = &[
    "try /context to see context usage",
    "try /config to change settings interactively",
    "try /compact to summarize and free up context",
    "try /compact <instructions> to steer what the summary keeps",
    "try /usage to see token usage and cost this session",
    "try /checkpoint <name> to bookmark this point",
    "try /rollback <name> to jump back to a checkpoint",
    "try /save to persist this session to disk",
    "try /list to see your saved sessions",
    "try /switch <sha> to load another session",
    "try /resume to pick up your most recent session",
    "try /del <sha> to delete a saved session",
    "try /tag <text> to label this session",
    "try /history to reprint recent turns",
    "try /new to start a fresh session",
    "try /mcp to see connected MCP servers and their tools",
    "try /skills to list the skills available to the model",
    "try /templates to list your {{var}} prompt templates",
    "try /tasks to see the model's todo list",
    "try /agent to list sub-agents you can delegate to",
    "try /hooks to see which hooks are configured",
    "try /remember <fact> to save a note for later sessions",
    "try /init to generate an AGENTS.md for this repo",
    "try /power <1..100> to cap GPU power draw",
    "try /strip <sha> to trim a session's oldest turns",
    "try /help to see every command and flag",
    "type @ to fuzzy-complete a file path into your prompt",
    "prefix a line with ! to run a shell command yourself",
    "press Shift+Enter (or Alt+Enter) for a newline",
    "press Ctrl+C once to clear the input line",
    "press Ctrl+U to delete to the start of the line",
    "press Ctrl+W to delete the previous word",
    "press Up/Down to walk your prompt history",
    "paste an image and the model can open it with its tools",
    "scroll up with the mouse wheel to review scrollback",
    "drag with the mouse to select and copy text",
    "ask a mid-turn question with /btw <question>",
    "hide thinking text with /config ui.showThinking false",
    "let the model act mid-thought with /config engine.thinkingToolCalls true",
    "show tool calls with /config ui.showToolCalls true",
    "echo tool results with /config ui.showToolResults true",
    "settings save to ./.plank/settings.json — commit it to share",
    "flags override settings.json; settings.json overrides defaults",
    "put project notes in AGENTS.md and plank reads them at start",
    "MCP servers load from ~/.plank/.mcp.json and ./.mcp.json",
    "compaction keeps a durable summary plus the recent tail",
    "the system prompt is cached on disk so restarts are fast",
    "-sys \"...\" overrides the system prompt for one run",
    "run one prompt and exit with plank -p \"your question\"",
    "keep secrets out of settings.json; use --api-key or env vars",
    "your session is saved automatically — /resume brings it back",
];

/// How long each rotating tip stays *visible* within its [`TIP_ROTATE_MS`]
/// window before the tip section auto-hides until the next rotation.
pub const TIP_VISIBLE_MS: u64 = 10_000;

/// The tip to show for the given animation clock, or `""` when none exist or
/// the current tip's [`TIP_VISIBLE_MS`] visibility window has lapsed (the tip
/// section disappears until the next rotation).
#[must_use]
pub fn rotating_tip(tick_ms: u64) -> &'static str {
    if tick_ms % TIP_ROTATE_MS >= TIP_VISIBLE_MS {
        return "";
    }
    // TIPS is a non-empty compile-time table, so the modulo never divides by zero.
    let idx = usize::try_from(tick_ms / TIP_ROTATE_MS).unwrap_or(0) % TIPS.len();
    TIPS[idx]
}

/// How long a transient "flash" tip (e.g. a copy confirmation) stays in the
/// status bar before it reverts to the rotating tip. Fixed, not configurable.
pub const FLASH_TIP_MS: u64 = 10_000;

/// The floor on how long a tool label stays in the status bar. A tool that
/// finishes in milliseconds would otherwise flicker past unread, so the label
/// lingers this long after the run ends — unless the next tool starts first,
/// which replaces it immediately.
pub const TOOL_LINGER_MS: u64 = 4_000;

/// Full period of the tool label's blink, half on and half off.
pub const TOOL_BLINK_MS: u64 = 800;

/// The tools currently executing, as a display label (e.g. `read_file, bash`),
/// paired with the moment the run ended (`None` while still running).
///
/// Unlike a flash tip a running label has no expiry: it is posted when a DSML
/// block's tool calls start and stays up for exactly as long as they run, then
/// for [`TOOL_LINGER_MS`] more. While it is set it owns the status bar's tail
/// notification slot — no flash, no rotating tip — because that is the one
/// thing the user needs to see while waiting on a tool.
static TOOL_ACTIVITY: std::sync::Mutex<Option<(String, Option<std::time::Instant>)>> =
    std::sync::Mutex::new(None);

/// Posts the running-tools label, replacing any previous one (including one
/// still serving out its linger window — a fresh tool always wins the slot).
pub fn set_tool_activity(label: String) {
    let mut guard = TOOL_ACTIVITY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some((label, None));
}

/// Marks the tools as finished, starting the [`TOOL_LINGER_MS`] window after
/// which the label drops out of the bar.
pub fn end_tool_activity() {
    let mut guard = TOOL_ACTIVITY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, ended @ None)) = guard.as_mut() {
        *ended = Some(std::time::Instant::now());
    }
}

/// Drops the tool label immediately, linger window and all.
pub fn clear_tool_activity() {
    let mut guard = TOOL_ACTIVITY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = None;
}

/// The tool label to show: present while tools run and for [`TOOL_LINGER_MS`]
/// after they finish. Expired labels are dropped on read.
#[must_use]
pub fn tool_activity() -> Option<String> {
    let mut guard = TOOL_ACTIVITY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match guard.as_ref() {
        Some((label, None)) => Some(label.clone()),
        Some((label, Some(at))) if flash_active(at.elapsed(), TOOL_LINGER_MS) => {
            Some(label.clone())
        }
        Some(_) => {
            *guard = None;
            None
        }
        None => None,
    }
}

/// Whether the tool label is in the lit half of its blink cycle.
#[must_use]
pub fn tool_blink_on(tick_ms: u64) -> bool {
    tick_ms % TOOL_BLINK_MS < TOOL_BLINK_MS / 2
}

/// Whether the engine currently prefilling or generating runs on this machine's
/// own weights. Process-global for the same reason the tool label is: the TUI
/// draws on the UI thread while the pass runs on the worker, and the renderer
/// only ever receives the *rendered* status string, not the engine.
///
/// True for exactly the span of a local pass, which is what lets the think
/// segment's brain blink mean "the local model is working right now" — including
/// a `provider: local` sub-agent sidechain under a remote main agent, since the
/// engine swap happens before the pass begins.
static LOCAL_PASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a local-engine pass is in flight.
#[must_use]
pub fn local_pass_active() -> bool {
    LOCAL_PASS.load(std::sync::atomic::Ordering::Relaxed)
}

/// When the live local pass began, as milliseconds on [`crate::anim`]'s shared
/// epoch. Meaningless unless `LOCAL_PASS` is set.
static LOCAL_PASS_SINCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Marks a local-engine pass as running for as long as the guard lives.
///
/// A guard rather than a set/clear pair for the same reason [`ToolActivity`] is
/// one: a pass has several early returns and can panic, and a leaked `true`
/// would leave the brain blinking forever.
#[derive(Debug)]
pub struct LocalPass;

impl LocalPass {
    /// Marks a pass as local; the returned guard clears it.
    #[must_use]
    pub fn begin() -> Self {
        #[cfg(test)]
        LOCAL_PASS_OVERRIDE.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        LOCAL_PASS_SINCE.store(
            crate::anim::epoch_ms(),
            std::sync::atomic::Ordering::Relaxed,
        );
        LOCAL_PASS.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

impl Drop for LocalPass {
    fn drop(&mut self) {
        LOCAL_PASS.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Milliseconds the live local pass has been running, or `0` when none is.
///
/// This — not the free-running animation clock — is what phases the brain
/// blink, so the blink is a function of *the pass's own elapsed time*: the same
/// quantity the status bar's `9s` and `t/s` readouts are computed from. A pass
/// therefore always starts on a lit brain and blinks in step with its own
/// counters, rather than picking up whatever phase the shared clock happened to
/// be in when it started.
#[must_use]
pub fn local_pass_ms() -> u64 {
    if !local_pass_active() {
        return 0;
    }
    #[cfg(test)]
    {
        let pinned = LOCAL_PASS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if pinned != u64::MAX {
            return pinned;
        }
    }
    crate::anim::epoch_ms()
        .saturating_sub(LOCAL_PASS_SINCE.load(std::sync::atomic::Ordering::Relaxed))
}

/// Pins how long the live local pass has been running, so a test can place the
/// blink at a chosen phase without sleeping through one. Cleared by the next
/// [`LocalPass::begin`].
#[cfg(test)]
pub(crate) fn set_local_pass_ms(ms: u64) {
    LOCAL_PASS_OVERRIDE.store(ms, std::sync::atomic::Ordering::Relaxed);
}

/// Test override for [`local_pass_ms`]; `u64::MAX` means "not overridden", a
/// value no real pass can reach.
#[cfg(test)]
static LOCAL_PASS_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// How long one on/off cycle of the local-engine brain blink takes. Slower than
/// the tool blink: this is a reassurance, not an alert, and a fast pulse next to
/// the shimmering verb would be noise.
pub const BRAIN_BLINK_MS: u64 = 1200;

/// Whether the brain is in the lit half of its blink at `tick_ms`.
#[must_use]
pub fn brain_blink_on(tick_ms: u64) -> bool {
    tick_ms % BRAIN_BLINK_MS < BRAIN_BLINK_MS / 2
}

/// Sets the running-tools label for as long as the guard lives, then hands it
/// over to its linger window on drop, so no early return (or panic) can leave
/// the bar claiming a tool is still running.
#[derive(Debug)]
pub struct ToolActivity;

impl ToolActivity {
    /// Posts `label` and returns the guard that ends it.
    #[must_use]
    pub fn begin(label: String) -> Self {
        set_tool_activity(label);
        Self
    }
}

impl Drop for ToolActivity {
    fn drop(&mut self) {
        end_tool_activity();
    }
}

/// A transient status-bar message (with its display window in ms) shown in
/// place of the rotating tip until it expires, then cleared automatically on
/// the next read. Process-global so any call site (e.g. a clipboard copy in
/// the mouse handler, tool dispatch on the worker thread) can post one without
/// threading state through the draw path.
static FLASH_TIP: std::sync::Mutex<Option<(String, std::time::Instant, u64)>> =
    std::sync::Mutex::new(None);

/// Whether a flash tip posted `elapsed` ago is still within a `window_ms`
/// display window.
#[must_use]
pub fn flash_active(elapsed: std::time::Duration, window_ms: u64) -> bool {
    elapsed < std::time::Duration::from_millis(window_ms)
}

/// Posts a transient status-bar tip with the default [`FLASH_TIP_MS`] window,
/// replacing any current one and restarting the window.
pub fn set_flash_tip(msg: String) {
    set_flash_tip_for(msg, FLASH_TIP_MS);
}

/// Posts a transient status-bar tip shown for `window_ms`, replacing any
/// current one and restarting the window.
pub fn set_flash_tip_for(msg: String, window_ms: u64) {
    let mut guard = FLASH_TIP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some((msg, std::time::Instant::now(), window_ms));
}

/// Clears any active flash tip immediately.
pub fn clear_flash_tip() {
    let mut guard = FLASH_TIP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = None;
}

/// The active flash tip if one is still within its window; expired tips are
/// cleared and reported as `None` so the caller falls back to the rotating tip.
#[must_use]
pub fn flash_tip() -> Option<String> {
    let mut guard = FLASH_TIP
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match guard.as_ref() {
        Some((msg, at, window_ms)) if flash_active(at.elapsed(), *window_ms) => Some(msg.clone()),
        Some(_) => {
            *guard = None;
            None
        }
        None => None,
    }
}

fn power_suffix(st: &Status) -> String {
    if st.power_percent > 0 && st.power_percent < 100 {
        format!(" | ⚡ {}%", st.power_percent)
    } else {
        String::new()
    }
}

/// Cells in the compaction bar shown in the status bar's tail slot. Much
/// narrower than [`PROGRESS_BAR_WIDTH`]: that bar owns a full centered line on
/// the warm-up screen, this one shares a single row with the whole status bar.
pub const COMPACT_BAR_WIDTH: usize = 18;

/// Filled cell of the compaction bar.
pub const COMPACT_BAR_FILLED: char = '▰';
/// Empty cell of the compaction bar.
pub const COMPACT_BAR_EMPTY: char = '▱';

/// How much of the compaction bar the prefill phase spans. Compaction is one
/// prefill of the entire transcript followed by one summary generation, and the
/// prefill is far the longer of the two, so it owns most of the bar.
pub const COMPACT_PREFILL_SHARE: f64 = 0.8;

/// Nominal length of a compaction summary, in tokens, used to advance the bar
/// through the summarizing phase. It is an *estimate*: `n_predict` is unlimited
/// by default, so there is no real total to divide by, and the phase is clamped
/// just short of full so the bar never claims to be done before it is.
pub const COMPACT_SUMMARY_TOKENS: f64 = 1200.0;

/// Compaction progress as a 0.0..=1.0 fraction while a compaction pass is
/// running, or `None` when none is. Process-global for the same reason as
/// [`FLASH_TIP`]: compaction runs on the worker thread and the status bar is
/// drawn on the UI thread, with no shared state between them to thread it
/// through.
static COMPACT_PROGRESS: std::sync::Mutex<Option<f64>> = std::sync::Mutex::new(None);

/// Publishes compaction progress, clamped to 0.0..=1.0.
pub fn set_compact_progress(frac: f64) {
    let mut guard = COMPACT_PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some(frac.clamp(0.0, 1.0));
}

/// Drops the compaction bar from the status bar.
pub fn clear_compact_progress() {
    let mut guard = COMPACT_PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = None;
}

/// The running compaction's progress fraction, or `None` when not compacting.
#[must_use]
pub fn compact_progress() -> Option<f64> {
    *COMPACT_PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Shows the compaction bar for as long as the guard lives, then clears it on
/// drop, so an interrupt, an engine error, or a panic cannot leave the status
/// bar insisting a compaction is still in flight.
#[derive(Debug)]
pub struct CompactProgress;

impl CompactProgress {
    /// Posts an empty bar and returns the guard that clears it.
    #[must_use]
    pub fn begin() -> Self {
        set_compact_progress(0.0);
        Self
    }

    /// Reports prefill progress; maps onto the first
    /// [`COMPACT_PREFILL_SHARE`] of the bar.
    #[allow(clippy::unused_self)]
    pub fn prefill(&self, done: i32, total: i32) {
        let total = total.max(1);
        let frac = f64::from(done.clamp(0, total)) / f64::from(total);
        set_compact_progress(frac * COMPACT_PREFILL_SHARE);
    }

    /// Reports summary bytes generated so far; maps onto the remainder of the
    /// bar, approaching but never reaching full.
    #[allow(clippy::unused_self)]
    pub fn summarizing(&self, bytes: usize) {
        // ~4 bytes per token is the usual rule of thumb for English prose; the
        // phase total is an estimate either way (see COMPACT_SUMMARY_TOKENS).
        #[allow(clippy::cast_precision_loss)]
        let tokens = bytes as f64 / 4.0;
        let frac = (tokens / COMPACT_SUMMARY_TOKENS).clamp(0.0, 0.95);
        set_compact_progress(COMPACT_PREFILL_SHARE + frac * (1.0 - COMPACT_PREFILL_SHARE));
    }
}

impl Drop for CompactProgress {
    fn drop(&mut self) {
        clear_compact_progress();
    }
}

/// Splits the compaction bar into its filled and empty runs for a 0.0..=1.0
/// fraction, plus the percentage to print after it. Returned as pieces rather
/// than one string so the TUI can style the two runs differently; the total
/// width is always [`COMPACT_BAR_WIDTH`], so the status line never reflows as
/// the bar advances.
#[must_use]
pub fn compact_bar(frac: f64) -> (String, String, u16) {
    let frac = frac.clamp(0.0, 1.0);
    // `frac` is clamped to 0..=1 and both widths are small constants, so every
    // rounded product below fits its target type; the `min`/`clamp` are belts to
    // the braces against float rounding at the endpoints.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let filled = ((frac * f64::from(u16::try_from(COMPACT_BAR_WIDTH).unwrap_or(u16::MAX))).round()
        as usize)
        .min(COMPACT_BAR_WIDTH);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pct = (frac * 100.0).round() as u16;
    (
        std::iter::repeat_n(COMPACT_BAR_FILLED, filled).collect(),
        std::iter::repeat_n(COMPACT_BAR_EMPTY, COMPACT_BAR_WIDTH - filled).collect(),
        pct,
    )
}

/// Milliseconds per throbber frame (one Braille glyph of the ping-pong cycle).
pub const THROBBER_STEP_MS: u64 = 100;

/// Braille throbber frame, expressed in terms of the shared [`crate::anim`]
/// clock so there is no duplicate timing logic. Any footer repaint advances the
/// animation and a pegged progress bar still shows the worker is alive. In
/// reduced-motion mode the clock returns `None` and the throbber freezes on its
/// first frame (a stable, non-animated glyph).
#[must_use]
pub fn throbber() -> char {
    let now = crate::anim::clock_ms().unwrap_or(0);
    crate::anim::throbber_char(now, THROBBER_STEP_MS)
}

/// Renders the prefill progress bar with a t/s readout after the bar.
#[must_use]
pub fn progress_bar(done: i32, total: i32, tps: f64, color: bool) -> String {
    let total = if total <= 0 { 1 } else { total };
    let done = done.clamp(0, total);
    let width = i64::try_from(PROGRESS_BAR_WIDTH).unwrap_or(i64::MAX);
    #[allow(clippy::cast_possible_truncation)]
    let mut filled = ((i64::from(done) * width) / i64::from(total)).unsigned_abs() as usize;
    filled = filled.min(PROGRESS_BAR_WIDTH);
    if color && filled == 0 && done < total {
        filled = 1;
    }
    let mut out = String::from("[");
    if color {
        out.push_str(STATUS_BAR_FILL);
    }
    for i in 0..PROGRESS_BAR_WIDTH {
        if color && i == filled {
            out.push_str(STATUS_STYLE_START);
        }
        out.push_str(if i < filled { "▶" } else { "·" });
    }
    if color {
        out.push_str(STATUS_STYLE_START);
    }
    out.push(']');
    if tps > 0.0 {
        let _ = std::fmt::Write::write_fmt(&mut out, format_args!(" {tps:.0}t/s"));
    }
    out
}

/// The animated progress segment — throbber, spinner verb, and the
/// elapsed/tokens/throughput readout — for the prefill and generating states.
/// `None` in every other state. Split out so the TUI can render it on a line
/// below the output instead of in the footer.
#[must_use]
pub fn progress_segment(st: &Status, color: bool) -> Option<String> {
    let theme = |text: &str| {
        if color {
            format!("\x1b[38;5;{THEME_COLOR};1m{text}{STATUS_STYLE_START}")
        } else {
            text.to_owned()
        }
    };
    match st.state {
        WorkerState::Prefill => {
            let total = if st.prefill_total > 0 {
                st.prefill_total
            } else {
                1
            };
            let done = st.prefill_done.min(total);
            Some(format!(
                "{} {}… ({} · ↑ {}/{} tokens · {:.1} t/s)",
                throbber(),
                theme(prefill_label(st)),
                format_elapsed(st.elapsed_secs),
                format_ctx_size(done),
                format_ctx_size(total),
                st.prefill_tps
            ))
        }
        WorkerState::Generating => Some(format!(
            "{} {}… ({} · ↓ {} tokens{} · {:.1} t/s)",
            throbber(),
            theme(prefill_label(st)),
            format_elapsed(st.elapsed_secs),
            format_ctx_size(st.generated),
            if st.greedy_sampling { " ❄️" } else { "" },
            st.gen_tps
        )),
        _ => None,
    }
}

/// Builds the compact one-line footer shown below the prompt. When
/// `progress_in_bar` is false the animated [`progress_segment`] is omitted from
/// prefill/generating footers (the TUI renders it in the output area instead).
#[must_use]
pub fn build_status_text(st: &Status, color: bool, progress_in_bar: bool) -> String {
    // Context usage shown as a bare percentage of the window (the ctx gauge).
    let ctx = if st.ctx_size > 0 {
        format!(
            "ctx {:.0}%",
            100.0 * f64::from(st.ctx_used) / f64::from(st.ctx_size)
        )
    } else {
        "ctx 0%".to_owned()
    };
    // The think segment: the reasoning level `/think` selects, ahead of the ctx
    // gauge. Always shown — it changes how every turn is generated, so it is
    // worth a permanent slot rather than appearing only when off the default.
    // Abbreviated to a fixed three columns so changing level does not shift the
    // rest of the footer sideways.
    let think = format!("{THINK_MARK} {} | ", st.think.short_name());
    let power = power_suffix(st);
    // Theme-colored accent text; returns to the footer style (not a full
    // reset) so the status bar's background survives on color terminals.
    let theme = |text: &str| {
        if color {
            format!("\x1b[38;5;{THEME_COLOR};1m{text}{STATUS_STYLE_START}")
        } else {
            text.to_owned()
        }
    };
    let cwd = cwd_label();
    // The origin rides with the directory prefix — both answer "where am I?" —
    // but as its own bar-separated segment, like the think and ctx segments.
    let origin = format!("{} | ", engine_origin_label());
    let dir = if cwd.is_empty() {
        origin
    } else if let Some(branch) = git_branch_label() {
        format!(
            "{} {POWERLINE_BRANCH} {} | {origin}",
            theme(&cwd),
            theme(&branch)
        )
    } else {
        format!("{} | {origin}", theme(&cwd))
    };
    let ctx = format!("{think}{ctx}");
    let body = match st.state {
        WorkerState::Prefill | WorkerState::Generating => {
            match progress_segment(st, color).filter(|_| progress_in_bar) {
                Some(progress) => format!("{ctx} | {progress}{power}"),
                // Progress lifted into the output area (showThinking off).
                None => format!("{ctx}{power}"),
            }
        }
        WorkerState::Compacting => format!(
            "{ctx} | COMPACTING summary {} tokens {:.1} t/s{power}",
            st.generated, st.gen_tps
        ),
        WorkerState::Saving => format!("{ctx} | saving session{power}"),
        WorkerState::Error => format!(
            "{ctx} | error: {}{power}",
            if st.error.is_empty() {
                "unknown error"
            } else {
                &st.error
            }
        ),
        WorkerState::Stopped => format!("{ctx} | interrupted{power}"),
        WorkerState::Idle => format!("{ctx} | idle{power}"),
    };
    format!("{dir}{body}")
}

/// Formats the echoed user prompt line (`* <text>` with bold styling on TTYs).
#[must_use]
pub fn format_user_prompt_echo(text: &str, color: bool) -> String {
    if color {
        format!("\x1b[1;91m*\x1b[1;97m {text}\x1b[0m\n\n")
    } else {
        format!("* {text}\n\n")
    }
}

/// Formats a system status line, mirroring `agent_publish_system_status`:
/// a `✦` bullet followed by the message in the theme green on TTYs, with any
/// URL in the message lifted to white so the target stands out from the prose.
///
/// These are the agent talking about itself ("Searching Google for ...",
/// "Compaction interrupted; ..."), distinct from both model output and the
/// dim tool-observation log. The returned string has no trailing newline.
#[must_use]
pub fn system_line(msg: &str, color: bool) -> String {
    if !color {
        return format!("✦ {msg}");
    }
    let green = format!("\x1b[38;5;{THEME_COLOR}m");
    let mut body = String::with_capacity(msg.len());
    for (i, word) in msg.split(' ').enumerate() {
        if i > 0 {
            body.push(' ');
        }
        if word.starts_with("http://") || word.starts_with("https://") {
            body.push_str("\x1b[38;5;231m");
            body.push_str(word);
            body.push_str(&green);
        } else {
            body.push_str(word);
        }
    }
    format!("\x1b[33m✦ {green}{body}\x1b[0m")
}

/// Formats the welcome banner line, mirroring the C agent's phrasing.
#[must_use]
pub fn welcome_banner(ctx_size: i32, color: bool) -> String {
    let ctx = format_ctx_size(ctx_size);
    let ver = crate::logo::version_label();
    if color {
        format!("\x1b[1;97mpl\x1b[1;94mank\x1b[0m {ver} 🪵 Agent, context {ctx} tokens\n\n")
    } else {
        format!("plank {ver} Agent, context {ctx} tokens\n\n")
    }
}

/// Startup lines shown when the engine is the echo stub: nothing infers until
/// a model is configured, and the binary itself cannot say so any other way.
///
/// The stub is only ever selected when the binary was built without the local
/// ds4 engine and no `--provider`/`--remote` was given, so the lines list the
/// three ways out of that state.
#[must_use]
pub fn no_model_lines() -> Vec<String> {
    vec![
        "No model loaded: this build has no local engine, so replies come from the echo stub."
            .to_owned(),
        "  local  : git submodule update --init refs/ds4 && cargo build   (macOS + Metal, 96 GB RAM)"
            .to_owned(),
        "           then plank -m <model.gguf>, or plank with no -m to fetch ~/.plank/ds4flash.gguf (~87 GB)"
            .to_owned(),
        "  hosted : plank --provider anthropic --model <name>   (key from $ANTHROPIC_API_KEY)".to_owned(),
        "  remote : plank --remote <url>   (another box running plank serve)".to_owned(),
        "  persist: ~/.plank/settings.json, e.g. {\"engine\": {\"model\": \"~/models/ds4.gguf\"}}"
            .to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_line_is_theme_green_with_white_urls() {
        assert_eq!(
            system_line("Opening page x...", false),
            "✦ Opening page x..."
        );
        let s = system_line("Opening page https://example.com/a...", true);
        assert!(s.contains(&format!("\x1b[38;5;{THEME_COLOR}m")));
        assert!(s.contains("\x1b[38;5;231mhttps://example.com/a..."));
        // Prose after the URL returns to green.
        assert!(system_line("see https://x.dev now", true).ends_with(" now\x1b[0m"));
    }

    #[test]
    fn ctx_size_formatting() {
        assert_eq!(format_ctx_size(512), "512");
        assert_eq!(format_ctx_size(8000), "8k");
        assert_eq!(format_ctx_size(2500), "2.5k");
        assert_eq!(format_ctx_size(2_000_000), "2M");
        assert_eq!(format_ctx_size(1_048_576), "1.0M");
    }

    #[test]
    fn idle_status_line() {
        let st = Status {
            ctx_used: 1000,
            ctx_size: 8000,
            ..Status::default()
        };
        assert!(
            build_status_text(&st, false, true).ends_with("ctx 12% | idle"),
            "{}",
            build_status_text(&st, false, true)
        );
    }

    #[test]
    fn url_host_keeps_only_the_domain() {
        assert_eq!(
            url_host("https://api.anthropic.com/v1"),
            "api.anthropic.com"
        );
        assert_eq!(url_host("https://api.openai.com/v1"), "api.openai.com");
        assert_eq!(url_host("http://localhost:8080/v1"), "localhost");
        assert_eq!(
            url_host("https://user:pw@gw.example.com/x"),
            "gw.example.com"
        );
        assert_eq!(url_host("api.anthropic.com"), "api.anthropic.com");
        // Nothing host-shaped: echo the input rather than showing an empty slot.
        assert_eq!(url_host("://"), "://");
    }

    #[test]
    fn footer_shows_the_engine_origin_beside_the_directory() {
        // No engine set this origin (unit tests build no engine), so the local
        // ds4/echo default stands. The label sits in the dir prefix, ahead of
        // the think segment — not appended to the tail.
        let line = build_status_text(&Status::default(), false, true);
        let origin = line.find("(local)").expect("origin in footer");
        let think = line.find(THINK_MARK).expect("think segment in footer");
        assert!(origin < think, "{line}");
    }

    #[test]
    fn progress_in_bar_toggle_moves_the_segment_out_of_the_footer() {
        let st = Status {
            state: WorkerState::Generating,
            generated: 42,
            gen_tps: 9.5,
            elapsed_secs: 62.0,
            ctx_used: 1500,
            ctx_size: 8000,
            ..Status::default()
        };
        // In the bar: the footer carries the throbber/verb/stats.
        let shown = build_status_text(&st, false, true);
        assert!(shown.contains("↓ 42 tokens"), "{shown}");
        assert!(
            shown.contains(&format!("{}…", prefill_label(&st))),
            "{shown}"
        );

        // Out of the bar: footer is just the ctx gauge, no progress segment.
        let hidden = build_status_text(&st, false, false);
        assert!(!hidden.contains("↓ 42 tokens"), "{hidden}");
        assert!(!hidden.contains('…'), "{hidden}");
        assert!(hidden.contains("ctx 19%"), "{hidden}");

        // The segment is still available for the output-area line.
        let seg = progress_segment(&st, false).expect("segment present");
        assert!(seg.contains("↓ 42 tokens · 9.5 t/s"), "{seg}");

        // No progress segment outside prefill/generating.
        assert!(progress_segment(&Status::default(), false).is_none());
    }

    #[test]
    fn git_branch_label_reads_current_repo() {
        // Running under cargo, the cwd is inside the plank repo.
        let branch = git_branch_label();
        assert!(branch.is_some(), "expected a branch inside the repo");
    }

    #[test]
    fn collapse_home_variants() {
        assert_eq!(collapse_home("/Users/x/Code", "/Users/x"), "~/Code");
        assert_eq!(collapse_home("/Users/x", "/Users/x"), "~");
        assert_eq!(collapse_home("/opt/tool", "/Users/x"), "/opt/tool");
        assert_eq!(collapse_home("/Users/x", ""), "/Users/x");
    }

    #[test]
    fn generating_status_line() {
        let st = Status {
            state: WorkerState::Generating,
            generated: 42,
            gen_tps: 9.5,
            elapsed_secs: 62.0,
            ctx_used: 1500,
            ctx_size: 8000,
            ..Status::default()
        };
        let line = build_status_text(&st, false, true);
        assert!(line.contains("ctx 19% | "), "{line}");
        assert!(line.contains("(1m 2s · ↓ 42 tokens · 9.5 t/s)"), "{line}");
        assert!(line.contains(&format!("{}…", prefill_label(&st))), "{line}");
    }

    #[test]
    fn prefill_status_line() {
        let st = Status {
            state: WorkerState::Prefill,
            prefill_done: 500,
            prefill_total: 2000,
            prefill_tps: 120.0,
            elapsed_secs: 5.0,
            ctx_used: 1500,
            ctx_size: 8000,
            ..Status::default()
        };
        let line = build_status_text(&st, false, true);
        assert!(line.contains("ctx 19% | "), "{line}");
        assert!(
            line.contains("(5s · ↑ 500/2k tokens · 120.0 t/s)"),
            "{line}"
        );
        assert!(line.contains(&format!("{}…", prefill_label(&st))), "{line}");
    }

    #[test]
    fn spinner_verbs_are_200_and_unique() {
        assert_eq!(SPINNER_VERBS.len(), 200);
        let set: std::collections::HashSet<_> = SPINNER_VERBS.iter().collect();
        assert_eq!(set.len(), 200);
    }

    #[test]
    fn bar_fill_uses_theme_color() {
        assert!(STATUS_BAR_FILL.contains(&format!(";38;5;{THEME_COLOR};")));
    }

    #[test]
    fn elapsed_formatting() {
        assert_eq!(format_elapsed(0.0), "0s");
        assert_eq!(format_elapsed(12.4), "12s");
        assert_eq!(format_elapsed(62.0), "1m 2s");
        assert_eq!(format_elapsed(3845.0), "1h 4m");
    }

    #[test]
    fn progress_bar_plain() {
        let bar = progress_bar(16, 32, 0.0, false);
        assert!(bar.starts_with('[') && bar.ends_with(']'));
        assert_eq!(bar.matches('▶').count(), 16);
        assert_eq!(bar.matches('·').count(), 16);
    }

    #[test]
    fn compact_bar_splits_at_the_fraction_and_keeps_a_fixed_width() {
        for (frac, filled) in [(0.0, 0), (0.5, 9), (1.0, COMPACT_BAR_WIDTH)] {
            let (on, off, _) = compact_bar(frac);
            assert_eq!(on.chars().count(), filled, "at {frac}");
            assert_eq!(
                on.chars().count() + off.chars().count(),
                COMPACT_BAR_WIDTH,
                "width must not vary at {frac}"
            );
        }
        // Out-of-range input is clamped rather than panicking or overflowing.
        assert_eq!(compact_bar(-1.0).2, 0);
        assert_eq!(compact_bar(2.0).2, 100);
        assert_eq!(compact_bar(0.21).2, 21);
    }

    #[test]
    fn compact_progress_phases_advance_monotonically_and_the_guard_clears() {
        let p = CompactProgress::begin();
        assert_eq!(compact_progress(), Some(0.0));
        // Prefill spans the first share of the bar.
        p.prefill(50, 100);
        assert!((compact_progress().unwrap() - COMPACT_PREFILL_SHARE / 2.0).abs() < 1e-9);
        p.prefill(100, 100);
        let after_prefill = compact_progress().unwrap();
        assert!((after_prefill - COMPACT_PREFILL_SHARE).abs() < 1e-9);
        // Summarizing picks up where prefill left off and never reaches full.
        p.summarizing(400);
        let mid = compact_progress().unwrap();
        assert!(mid > after_prefill, "{mid} must exceed {after_prefill}");
        p.summarizing(1_000_000);
        let end = compact_progress().unwrap();
        assert!(
            end > mid && end < 1.0,
            "{end} must approach but not reach 1"
        );
        // Dropping the guard takes the bar down, however the pass ended.
        drop(p);
        assert!(compact_progress().is_none());
    }

    #[test]
    fn rotating_tip_advances_and_wraps() {
        assert_eq!(rotating_tip(0), TIPS[0]);
        // Within the visibility window the first tip shows.
        assert_eq!(rotating_tip(TIP_VISIBLE_MS - 1), TIPS[0]);
        // Past the visibility window the tip section auto-hides.
        assert_eq!(rotating_tip(TIP_VISIBLE_MS), "");
        assert_eq!(rotating_tip(TIP_ROTATE_MS - 1), "");
        // The next window advances by one and is visible again.
        assert_eq!(rotating_tip(TIP_ROTATE_MS), TIPS[1 % TIPS.len()]);
        // After a full cycle it wraps back to the first.
        let cycle = TIP_ROTATE_MS * TIPS.len() as u64;
        assert_eq!(rotating_tip(cycle), TIPS[0]);
    }

    #[test]
    fn flash_active_respects_window() {
        use std::time::Duration;
        assert!(flash_active(Duration::from_millis(0), FLASH_TIP_MS));
        assert!(flash_active(
            Duration::from_millis(FLASH_TIP_MS - 1),
            FLASH_TIP_MS
        ));
        assert!(!flash_active(
            Duration::from_millis(FLASH_TIP_MS),
            FLASH_TIP_MS
        ));
        assert!(!flash_active(
            Duration::from_millis(FLASH_TIP_MS + 5_000),
            FLASH_TIP_MS
        ));
    }

    #[test]
    fn tool_activity_lasts_the_run_then_lingers_until_the_next_tool() {
        clear_tool_activity();
        {
            let _running = ToolActivity::begin("read_file, bash".to_string());
            assert_eq!(tool_activity().as_deref(), Some("read_file, bash"));
        }
        // The guard is gone but the label lingers so a fast tool stays read.
        assert_eq!(tool_activity().as_deref(), Some("read_file, bash"));
        // The next tool takes the slot immediately, without waiting it out.
        let _next = ToolActivity::begin("web_search".to_string());
        assert_eq!(tool_activity().as_deref(), Some("web_search"));
        clear_tool_activity();
        assert_eq!(tool_activity(), None);
    }

    #[test]
    fn tool_blink_is_half_on_half_off() {
        assert!(tool_blink_on(0));
        assert!(tool_blink_on(TOOL_BLINK_MS / 2 - 1));
        assert!(!tool_blink_on(TOOL_BLINK_MS / 2));
        assert!(!tool_blink_on(TOOL_BLINK_MS - 1));
        assert!(tool_blink_on(TOOL_BLINK_MS));
    }

    #[test]
    fn flash_tip_round_trips_and_clears() {
        set_flash_tip("Copied 42 chars".to_string());
        assert_eq!(flash_tip().as_deref(), Some("Copied 42 chars"));
        // A fresh post replaces the previous one.
        set_flash_tip("Copied 7 chars".to_string());
        assert_eq!(flash_tip().as_deref(), Some("Copied 7 chars"));
        // Leave the process-global clean for other tests in this binary.
        clear_flash_tip();
        assert_eq!(flash_tip(), None);
    }

    /// The think segment sits immediately before the ctx gauge, in every state,
    /// and names the level `/think` selected.
    #[test]
    fn think_segment_precedes_the_ctx_gauge() {
        use crate::engine::ThinkMode;

        for (level, name) in [
            (ThinkMode::Off, "off"),
            (ThinkMode::Low, "low"),
            (ThinkMode::Medium, "med"),
            (ThinkMode::Max, "max"),
        ] {
            for state in [
                WorkerState::Idle,
                WorkerState::Saving,
                WorkerState::Stopped,
                WorkerState::Compacting,
            ] {
                let st = Status {
                    state,
                    ctx_used: 30,
                    ctx_size: 1000,
                    think: level,
                    ..Status::default()
                };
                let line = build_status_text(&st, false, true);
                assert!(
                    line.contains(&format!("{THINK_MARK} {name} | ctx 3%")),
                    "{level:?}/{state:?}: {line}"
                );
            }
        }
    }

    /// It is a body segment, not part of the dir prefix: the separator before it
    /// belongs to the directory hand-off, so the branch name must not absorb it.
    /// (The TUI splits on `THINK_MARK` for exactly this reason.)
    #[test]
    fn think_segment_follows_the_dir_prefix() {
        let st = Status {
            ctx_used: 1,
            ctx_size: 100,
            ..Status::default()
        };
        let line = build_status_text(&st, false, true);
        let mark = line.find(THINK_MARK).expect("think segment present");
        let ctx = line.find("ctx ").expect("ctx gauge present");
        assert!(mark < ctx, "{line}");
        // Nothing between the two but the separator.
        assert_eq!(&line[mark..ctx], format!("{THINK_MARK} med | "), "{line}");
    }

    #[test]
    fn power_suffix_shown_only_when_limited() {
        let mut st = Status {
            ctx_size: 100,
            power_percent: 50,
            ..Status::default()
        };
        assert!(build_status_text(&st, false, true).ends_with("⚡ 50%"));
        st.power_percent = 100;
        assert!(!build_status_text(&st, false, true).contains('⚡'));
    }

    #[test]
    fn user_echo_formats() {
        assert_eq!(format_user_prompt_echo("hi", false), "* hi\n\n");
        assert!(format_user_prompt_echo("hi", true).contains("\x1b[1;91m*"));
    }
}

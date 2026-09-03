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

/// Leading glyph of the **git stat segment** (`📄 3 · +12 -4`), shown just
/// after the branch name inside the dir prefix.
///
/// Like [`THINK_MARK`], it is also the anchor the TUI splits on: the segment
/// rides between the branch and the engine origin, so `push_dir_prefix` peels
/// it here rather than letting it be absorbed into the branch name.
pub const GIT_STAT_MARK: &str = "📄";

/// The added-lines count in the git stat segment: pale mint.
///
/// 256-color 158 (`#afffd7`), the nearest palette entry to the chosen
/// `#c3f8e8` — a palette index rather than truecolor so the segment matches
/// the rest of the bar, which is indexed throughout. Indices 16 and up are
/// fixed by the xterm cube, so unlike 0-15 they are not repainted by the
/// user's theme.
pub const GIT_ADD_COLOR: u8 = 158;
/// The deleted-lines count: pale magenta, 256-color 212 (`#ff87d7`), the
/// nearest entry to the chosen `#ee88db`. See [`GIT_ADD_COLOR`].
pub const GIT_DEL_COLOR: u8 = 212;

/// Working-tree change summary: files touched, lines added, lines deleted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GitStats {
    /// Number of files differing from `HEAD` (staged and unstaged, deduped).
    pub files: usize,
    /// Lines added across those files.
    pub added: usize,
    /// Lines deleted across those files.
    pub deleted: usize,
}

impl GitStats {
    /// True when the tree is clean, in which case the footer shows nothing.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.files == 0 && self.added == 0 && self.deleted == 0
    }
}

/// How long a computed [`GitStats`] is reused before the diff is walked again.
///
/// The footer repaints several times a second; diffing the whole working tree
/// at that rate would cost more than everything else on the line put together.
/// A second of staleness in a change counter nobody is watching frame-by-frame
/// is not a defect.
const GIT_STATS_TTL: std::time::Duration = std::time::Duration::from_millis(1000);

/// Cached result of the last working-tree diff, with the instant it was taken.
static GIT_STATS_CACHE: std::sync::Mutex<Option<(std::time::Instant, Option<GitStats>)>> =
    std::sync::Mutex::new(None);

/// Working-tree change summary against `HEAD`, discovered from the cwd via
/// `git2`. `None` outside a repo, or when the diff cannot be taken.
///
/// Cached for [`GIT_STATS_TTL`]; a poisoned cache falls through to a fresh
/// computation rather than panicking on the render path.
#[must_use]
pub fn git_stats() -> Option<GitStats> {
    let now = std::time::Instant::now();
    if let Ok(cache) = GIT_STATS_CACHE.lock()
        && let Some((taken, stats)) = *cache
        && now.duration_since(taken) < GIT_STATS_TTL
    {
        return stats;
    }
    let stats = compute_git_stats();
    if let Ok(mut cache) = GIT_STATS_CACHE.lock() {
        *cache = Some((now, stats));
    }
    stats
}

/// Walks the `HEAD`-to-working-tree diff once. See [`git_stats`] for the
/// cached entry point.
fn compute_git_stats() -> Option<GitStats> {
    let repo = git2::Repository::discover(".").ok()?;
    // Tree-to-workdir-with-index: one diff covering both staged and unstaged
    // work, so a file edited and then added is counted once, not twice.
    let tree = repo.head().ok().and_then(|h| h.peel_to_tree().ok());
    let mut opts = git2::DiffOptions::new();
    opts.include_untracked(true).recurse_untracked_dirs(true);
    let diff = repo
        .diff_tree_to_workdir_with_index(tree.as_ref(), Some(&mut opts))
        .ok()?;
    let stats = diff.stats().ok()?;
    Some(GitStats {
        files: stats.files_changed(),
        added: stats.insertions(),
        deleted: stats.deletions(),
    })
}

/// The git stat segment text, `📄 3 · +12 -4`, or `None` for a clean tree or
/// no repo. With `color`, the counts carry their own ANSI colors and return to
/// the footer style.
#[must_use]
pub fn git_stat_segment(color: bool) -> Option<String> {
    let st = git_stats()?;
    if st.is_clean() {
        return None;
    }
    if color {
        Some(format!(
            "{GIT_STAT_MARK} {} · \x1b[38;5;{GIT_ADD_COLOR};1m+{}{STATUS_STYLE_START} \x1b[38;5;{GIT_DEL_COLOR};1m-{}{STATUS_STYLE_START}",
            st.files, st.added, st.deleted
        ))
    } else {
        Some(format!(
            "{GIT_STAT_MARK} {} · +{} -{}",
            st.files, st.added, st.deleted
        ))
    }
}

/// The think segment's level name colored by *temperature*: the hotter the
/// reasoning effort, the hotter the color. Red for `max`, white for `med`, blue
/// for `low`, grey for `off` — so the level is legible at a glance, from a
/// segment only three columns wide, without reading the word.
///
/// A 256-color index rather than an RGB triple: the footer is written as
/// `38;5;<n>` escapes on the plain path and as `Color::Indexed` in the TUI, and
/// one number serves both. Grey 245 rather than the 240 used for dim text
/// elsewhere: the status bar's own background is 238, and 240 on 238 is not a
/// contrast so much as a rumor.
#[must_use]
pub fn think_color(mode: crate::engine::ThinkMode) -> u8 {
    use crate::engine::ThinkMode;
    match mode {
        ThinkMode::Max => 196,
        ThinkMode::Medium => 231,
        ThinkMode::Low => 39,
        ThinkMode::Off => 245,
    }
}

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
    /// True while the stream is inside a `<think>` block, so the spinner verb
    /// can say so. Set from the renderer alongside `greedy_sampling`.
    pub thinking: bool,
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
    /// Speculative-decoding counters (`--dspark`). Rendered only when
    /// [`active`](crate::engine::SpecStats::active); a run without a support
    /// model never shows the segment.
    pub spec: crate::engine::SpecStats,
}

/// Marks the speculative-decoding segment, mirroring how `THINK_MARK` labels
/// the reasoning level.
///
/// Deliberately not `⚡`: the power suffix already owns that glyph (`local
/// ⚡100%`), and two different meanings for one mark in a single footer is
/// exactly the sort of thing nobody notices until they misread it.
const SPEC_MARK: &str = "⏩";

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
static ENGINE_ORIGINS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Shown for an engine running on this machine's own weights: the Metal-backed
/// ds4 engine, and the `EchoEngine` stub, which is equally local.
pub const LOCAL_ORIGIN: &str = "(local)";

/// Records an engine origin for the footer, in first-seen order.
///
/// A list rather than a single value: a session can hold several engines at
/// once — a provider main agent with a `provider: local` sub-agent, or
/// definitions pointing at different hosts — and "where is inference running"
/// then has more than one answer. Repeats are dropped, so two definitions on
/// the same host name it once and the main engine stays first.
pub fn set_engine_origin(label: &str) {
    let label = label.trim();
    if label.is_empty() {
        return;
    }
    let Ok(mut origins) = ENGINE_ORIGINS.lock() else {
        return;
    };
    if !origins.iter().any(|o| o == label) {
        origins.push(label.to_owned());
    }
}

/// GPU power cap for the local engine, as a percentage. `0` or `100` means
/// unlimited and shows no badge.
///
/// Process-global beside the origin list rather than carried on [`Status`],
/// because it has to be readable from *both* sides of the footer: the builder
/// that composes the origin segment and the TUI that peels that same segment
/// back off. Two sources would let them disagree about where the segment ends.
static LOCAL_POWER: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Records the local engine's GPU power cap, from startup config or `/power`.
/// Status-bar cells contributed by WASM `segment` components, already
/// rendered.
///
/// Published rather than pulled for the same reason the roster and the power
/// share are: [`build_status_text`] is a pure function called from a dozen
/// places that have no access to a session, and threading one through all of
/// them to reach a plugin registry would be a far larger change than the
/// feature is worth. The registry re-renders on its own cadence and drops the
/// text here; the bar only ever reads it.
/// Each cell is `(text, priority)`: the priority is what `segment_render`
/// returned, kept so the bar can *elide* rather than truncate when the line
/// overflows. Dropping a whole cell by priority is what a component asked for
/// by returning one; chopping the line at the right edge instead removes the
/// power suffix, which is the bar's own anchor.
static WASM_SEGMENTS: std::sync::RwLock<Vec<Cell>> = std::sync::RwLock::new(Vec::new());

/// One contributed status-bar cell as the bar needs it.
#[derive(Debug, Clone, Default)]
pub struct Cell {
    /// Rendered text, already one line.
    pub text: String,
    /// Elision order: higher survives longer.
    pub priority: u8,
    /// Foreground, when the component asked for one.
    pub fg: Option<(u8, u8, u8)>,
    /// Background, when the component asked for one.
    pub bg: Option<(u8, u8, u8)>,
}

/// Replaces the contributed cells, highest priority first.
pub fn set_wasm_segments(segments: Vec<Cell>) {
    if let Ok(mut slot) = WASM_SEGMENTS.write() {
        *slot = segments;
    }
}

/// The contributed cells, joined for the bar. Empty when there are none, so
/// the common case adds not one byte to the line.
#[must_use]
/// The contributed cells, dropping the lowest-priority ones until at most
/// `keep` of them remain.
///
/// `usize::MAX` keeps them all, which is the ordinary path: elision only
/// happens when [`build_status_text_within`] has established the line does not
/// fit.
fn wasm_segment_text_keeping(keep: usize, color: bool) -> String {
    let Ok(slot) = WASM_SEGMENTS.read() else {
        return String::new();
    };
    elide_cells_styled(&slot, keep, color)
}

/// Joins the highest-priority `keep` cells, prefixed for the bar.
///
/// Pure, and takes the cells rather than reading the global, so it is testable
/// without a shared mutable static that parallel tests would fight over.
/// Joins the highest-priority `keep` cells, optionally painting each cell's
/// requested colours.
///
/// Pure, and takes the cells rather than reading the published global, so it is
/// testable without a shared mutable static that parallel tests would fight
/// over — the first version of its test set that static and broke an unrelated
/// one.
///
/// A cell's colour is closed by returning to the bar's own style rather than by
/// a full reset, exactly as the theme accent does: a reset here would drop the
/// status bar's background for the rest of the line.
fn elide_cells_styled(cells: &[Cell], keep: usize, color: bool) -> String {
    if cells.is_empty() || keep == 0 {
        return String::new();
    }
    // Already highest-priority-first, so keeping a prefix drops the lowest.
    let kept: Vec<String> = cells
        .iter()
        .take(keep)
        .map(|cell| {
            if !color || (cell.fg.is_none() && cell.bg.is_none()) {
                return cell.text.clone();
            }
            let mut out = String::new();
            if let Some((r, g, b)) = cell.fg {
                let _ =
                    std::fmt::Write::write_fmt(&mut out, format_args!("\x1b[38;2;{r};{g};{b}m"));
            }
            if let Some((r, g, b)) = cell.bg {
                let _ =
                    std::fmt::Write::write_fmt(&mut out, format_args!("\x1b[48;2;{r};{g};{b}m"));
            }
            out.push_str(&cell.text);
            out.push_str(STATUS_STYLE_START);
            out
        })
        .collect();
    if kept.is_empty() {
        return String::new();
    }
    format!(" | {}", kept.join(" | "))
}

pub fn set_local_power(percent: i32) {
    LOCAL_POWER.store(percent, std::sync::atomic::Ordering::Relaxed);
}

/// The local engine's origin label, carrying its power share: `(local ⚡100%)`,
/// or `(local ⚡60%)` under a cap.
///
/// Always shown, never hidden at 100%. The reading answers "how much of the GPU
/// is this engine allowed", and "all of it" is an answer — a badge that appears
/// only under a cap makes its absence ambiguous between unlimited and not
/// supported. Unset (`0`) is the default and means uncapped, so it reads 100.
///
/// The share goes *inside* the parentheses so the whole thing is one label for
/// one engine. Outside them it reads as a second segment of the bar, which is
/// what it used to be and exactly what stopped being true once the footer could
/// name more than one engine.
///
/// Derived from `LOCAL_ORIGIN` rather than spelled out again, so the two cannot
/// drift apart — and if it ever stops being parenthesized, this degrades to
/// appending rather than producing a stray bracket.
#[must_use]
fn local_origin_label() -> String {
    let pct = LOCAL_POWER.load(std::sync::atomic::Ordering::Relaxed);
    let pct = if pct <= 0 { 100 } else { pct.min(100) };
    LOCAL_ORIGIN.strip_suffix(')').map_or_else(
        || format!("{LOCAL_ORIGIN} ⚡{pct}%"),
        |head| format!("{head} ⚡{pct}%)"),
    )
}

/// The engine-origin label: every origin in play, comma-separated in first-seen
/// order, or [`LOCAL_ORIGIN`] when nothing registered one.
///
/// The local entry carries the GPU power cap, because the cap applies to *that*
/// engine and to no other. Beside a provider it would otherwise read as a
/// property of the session, which it is not — a `provider: local` sub-agent
/// under a remote main agent is exactly the case where the distinction matters.
#[must_use]
pub fn engine_origin_label() -> String {
    let local = local_origin_label();
    let Ok(origins) = ENGINE_ORIGINS.lock() else {
        return local;
    };
    if origins.is_empty() {
        return local;
    }
    origins
        .iter()
        .map(|o| if o == LOCAL_ORIGIN { &local } else { o })
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

/// Serializes tests that touch the process-global origin list or power cap.
///
/// These live in two modules (here and [`crate::tui`]), and a reader that calls
/// [`engine_origin_label`] twice — as `push_dir_prefix` does — sees a torn
/// result if another test mutates between the calls. Cheaper and more honest
/// than making every such test tolerate arbitrary global state.
#[cfg(test)]
pub(crate) fn origin_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Forgets every recorded origin. Tests only: the list is process-global, and a
/// test that registers one would otherwise leak into the next.
#[cfg(test)]
pub(crate) fn reset_engine_origins() {
    if let Ok(mut origins) = ENGINE_ORIGINS.lock() {
        origins.clear();
    }
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

/// The five kinds of work the spinner verb can describe. The footer shows one
/// verb per turn, but *which pool* it is drawn from follows what the agent is
/// actually doing at that instant, so "Chiseling" never shows up while the
/// model is still reading the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbPhase {
    /// Sampling inside a `<think>` block: reasoning, not answering yet.
    Thinking,
    /// Sampling ordinary assistant output.
    Generating,
    /// A tool dispatch is in flight — the model has stopped, the machine works.
    Tool,
    /// Prefilling the prompt: taking the context in, producing nothing yet.
    Prefill,
    /// The 1-in-[`FUN_ODDS`] escape hatch, drawn regardless of phase.
    Fun,
}

/// Reasoning verbs, shown while the stream is inside a `<think>` block.
pub const THINKING_VERBS: [&str; 45] = [
    "Brainstorming",
    "Cerebrating",
    "Cogitating",
    "Contemplating",
    "Deciphering",
    "Decoding",
    "Deducing",
    "Deliberating",
    "Digesting",
    "Divining",
    "Dreaming",
    "Elucidating",
    "Elucubrating",
    "Envisioning",
    "Extrapolating",
    "Fathoming",
    "Grokking",
    "Guessing",
    "Hypothesizing",
    "Ideating",
    "Illuminating",
    "Imagining",
    "Incubating",
    "Inferring",
    "Intuiting",
    "Meditating",
    "Mulling",
    "Musing",
    "Noodling",
    "Osmosing",
    "Percolating",
    "Philosophising",
    "Pondering",
    "Pontificating",
    "Prognosticating",
    "Puzzling",
    "Reasoning",
    "Reflecting",
    "Ruminating",
    "Scheming",
    "Speculating",
    "Strategizing",
    "Theorizing",
    "Untangling",
    "Wondering",
];

/// Producing verbs, shown while sampling ordinary (non-thinking) output.
pub const GENERATING_VERBS: [&str; 45] = [
    "Arranging",
    "Articulating",
    "Assembling",
    "Authoring",
    "Composing",
    "Conjuring",
    "Crafting",
    "Creating",
    "Curating",
    "Describing",
    "Detailing",
    "Dictating",
    "Distilling",
    "Doodling",
    "Drafting",
    "Elaborating",
    "Emitting",
    "Explaining",
    "Expounding",
    "Formulating",
    "Framing",
    "Generating",
    "Improvising",
    "Manifesting",
    "Narrating",
    "Orchestrating",
    "Outlining",
    "Penning",
    "Phrasing",
    "Producing",
    "Rendering",
    "Riffing",
    "Scribing",
    "Sketching",
    "Spinning",
    "Summoning",
    "Synthesizing",
    "Transcribing",
    "Unfurling",
    "Voicing",
    "Weaving",
    "Wording",
    "Wordsmithing",
    "Writing",
    "Yarning",
];

/// Hands-on verbs, shown while a tool dispatch is running — the phase where
/// something in the world actually changes.
pub const TOOL_VERBS: [&str; 45] = [
    "Bolting",
    "Bulldozing",
    "Carving",
    "Cementing",
    "Chiseling",
    "Chopping",
    "Drilling",
    "Etching",
    "Excavating",
    "Fanning",
    "Forging",
    "Gilding",
    "Grafting",
    "Grinding",
    "Hacking",
    "Hammering",
    "Kneading",
    "Lathing",
    "Milling",
    "Mortaring",
    "Nailing",
    "Painting",
    "Patching",
    "Planing",
    "Plumbing",
    "Polishing",
    "Prestidigitizing",
    "Rewiring",
    "Riveting",
    "Sanding",
    "Sawing",
    "Scaffolding",
    "Sculpting",
    "Smooshing",
    "Soldering",
    "Splicing",
    "Squishing",
    "Stitching",
    "Tinkering",
    "Trowelling",
    "Welding",
    "Whittling",
    "Wiring",
    "Wrangling",
    "Wrenching",
];

/// Taking-it-in verbs, shown while prefilling — reading the context, not yet
/// producing anything from it.
pub const PREFILL_VERBS: [&str; 45] = [
    "Absorbing",
    "Appraising",
    "Assessing",
    "Assimilating",
    "Auditing",
    "Cataloguing",
    "Charting",
    "Checking",
    "Collating",
    "Combing",
    "Confirming",
    "Consuming",
    "Correlating",
    "Devouring",
    "Examining",
    "Gauging",
    "Indexing",
    "Ingesting",
    "Inspecting",
    "Mapping",
    "Measuring",
    "Monitoring",
    "Noting",
    "Observing",
    "Parsing",
    "Perusing",
    "Poring",
    "Probing",
    "Reading",
    "Registering",
    "Reviewing",
    "Sampling",
    "Scanning",
    "Screening",
    "Scrutinizing",
    "Sifting",
    "Skimming",
    "Studying",
    "Surveying",
    "Tallying",
    "Tracing",
    "Tracking",
    "Verifying",
    "Watching",
    "Weighing",
];

/// The rare ones: whatever the agent is doing, roughly one turn in
/// [`FUN_ODDS`] claims to be doing this instead.
pub const FUN_VERBS: [&str; 20] = [
    "Racing 🏎",
    "Fishing 🐟",
    "Traveling ✈",
    "Eating 🍜",
    "Watching 📺",
    "Napping 😴",
    "Surfing 🏄",
    "Juggling 🤹",
    "Dancing 💃",
    "Bowling 🎳",
    "Gardening 🌱",
    "Skateboarding 🛹",
    "Rocketing 🚀",
    "Sailing ⛵",
    "Camping 🏕",
    "Grilling 🍖",
    "Skiing 🎿",
    "Yodeling 🎤",
    "Unicycling 🚲",
    "Moonwalking 🌙",
];

/// One turn in this many draws from [`FUN_VERBS`] instead of the phase pool.
/// Rare enough to stay a surprise, common enough to actually be seen.
pub const FUN_ODDS: u32 = 20;

/// Every pool in phase order — used by the tests and by anything that wants to
/// enumerate the whole vocabulary.
#[must_use]
pub fn verbs_for(phase: VerbPhase) -> &'static [&'static str] {
    match phase {
        VerbPhase::Thinking => &THINKING_VERBS,
        VerbPhase::Generating => &GENERATING_VERBS,
        VerbPhase::Tool => &TOOL_VERBS,
        VerbPhase::Prefill => &PREFILL_VERBS,
        VerbPhase::Fun => &FUN_VERBS,
    }
}

/// Which pool this status draws from.
///
/// The fun pool wins first and is decided by the turn's own seed, so a turn
/// that drew a fun verb keeps it across every phase it passes through rather
/// than flipping back to shop tools the moment a tool runs. Otherwise the
/// phase follows the live state: prefill beats everything (nothing else is
/// happening yet), then a running tool dispatch — which outlives the last
/// `Status` the engine published, so it is read from the process-global tool
/// activity rather than from `st` — then thinking, then plain generation.
#[must_use]
pub fn verb_phase(st: &Status) -> VerbPhase {
    if st.prefill_label.is_multiple_of(FUN_ODDS) {
        return VerbPhase::Fun;
    }
    if st.state == WorkerState::Prefill {
        return VerbPhase::Prefill;
    }
    if tools_running() {
        return VerbPhase::Tool;
    }
    if st.thinking {
        VerbPhase::Thinking
    } else {
        VerbPhase::Generating
    }
}

/// Keep each operation on a single playful verb so the footer does not
/// visually churn while progress updates stream in. The verb index is fixed
/// for the turn; only the pool it indexes moves with the phase.
///
/// In the fun phase the seed has already been constrained to a multiple of
/// [`FUN_ODDS`] by the roll in [`verb_phase`], so dividing that factor out
/// before the modulo is what recovers a uniform spread across the pool —
/// without it, `pool.len() == FUN_ODDS` would collapse every fun draw onto
/// index 0.
#[must_use]
pub fn prefill_label(st: &Status) -> &'static str {
    let phase = verb_phase(st);
    let pool = verbs_for(phase);
    let idx = if phase == VerbPhase::Fun {
        (st.prefill_label / FUN_ODDS) as usize % pool.len()
    } else {
        st.prefill_label as usize % pool.len()
    };
    pool[idx]
}

/// Picks a stable random verb seed for a new turn, seeded from wall-clock.
///
/// Deliberately *not* reduced modulo a pool length: the raw seed also decides
/// the [`FUN_ODDS`] roll, and each pool takes its own modulo at lookup time.
#[must_use]
pub fn random_verb_index() -> u32 {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    #[allow(clippy::cast_possible_truncation)]
    let seed = (ms ^ (ms >> 17)) as u32;
    seed
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

/// The one tip that is an advertisement, for the one product that is free.
///
/// Takes a rotation slot rather than a banner or a startup notice, so it costs
/// exactly what any other tip costs and can be ignored the same way.
///
/// Spelled with the scheme because terminals that linkify URLs mostly want one:
/// a bare host reads as prose, `https://` reads as something to click.
pub const PROMO_TIP: &str =
    "10,000,000 free tokens, already claimed — https://plank-agent.dev/free-tokens";

/// How often [`PROMO_TIP`] displaces an ordinary tip, counted in rotations.
///
/// At [`TIP_ROTATE_MS`] that is once every hundred minutes of wall clock — rare
/// enough to read as a joke rather than a nag, and never on the first rotation
/// of a session, where a useful tip earns its place more.
pub const PROMO_EVERY: u64 = 50;

/// The tip to show for the given animation clock, or `""` when none exist or
/// the current tip's [`TIP_VISIBLE_MS`] visibility window has lapsed (the tip
/// section disappears until the next rotation).
#[must_use]
pub fn rotating_tip(tick_ms: u64) -> &'static str {
    if tick_ms % TIP_ROTATE_MS >= TIP_VISIBLE_MS {
        return "";
    }
    let rotation = tick_ms / TIP_ROTATE_MS;
    if rotation > 0 && rotation.is_multiple_of(PROMO_EVERY) {
        return PROMO_TIP;
    }
    // TIPS is a non-empty compile-time table, so the modulo never divides by zero.
    let idx = usize::try_from(rotation).unwrap_or(0) % TIPS.len();
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

/// Whether a tool dispatch is *running right now* — as opposed to
/// [`tool_activity`], which keeps returning a label through the linger window
/// after the tools finished. The spinner verb needs the strict answer: once
/// the model is sampling again, a lingering label must not keep it on shop
/// verbs.
#[must_use]
pub fn tools_running() -> bool {
    let guard = TOOL_ACTIVITY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    matches!(guard.as_ref(), Some((_, None)))
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
        ROUTED_TOKEN.store(NO_ROUTED_TOKEN, std::sync::atomic::Ordering::Relaxed);
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

/// Sentinel for "no token has been decoded in this pass yet".
const NO_ROUTED_TOKEN: u64 = u64::MAX;

/// The token whose expert routing the think segment is currently drawing.
///
/// Process-global for the same reason [`LOCAL_PASS`] is: the pass runs on the
/// worker thread while the TUI draws on the UI thread, and the renderer only
/// ever receives the rendered status string.
static ROUTED_TOKEN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(NO_ROUTED_TOKEN);

/// Records the token just decoded, so [`crate::experts`] can draw its routing.
///
/// Called from the local decode loop; a pass that never reports one (prefill, or
/// any engine but the native one) falls back to the pass clock in
/// [`routing_seed`], so the glyph still animates on the `EchoEngine` path.
pub fn note_routed_token(token: i32) {
    ROUTED_TOKEN.store(
        u64::from(token.unsigned_abs()),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// How long one expert frame holds when the seed comes from the clock rather
/// than from a real token. Close to a fast decode rate: slow enough to read as
/// distinct frames, fast enough to look like work.
pub const EXPERT_FRAME_MS: u64 = 120;

/// The seed [`crate::experts`] draws the current routing from: the live token
/// when the decode loop is reporting them, else the pass clock in frame steps.
#[must_use]
pub fn routing_seed() -> u64 {
    let token = ROUTED_TOKEN.load(std::sync::atomic::Ordering::Relaxed);
    if token == NO_ROUTED_TOKEN {
        local_pass_ms() / EXPERT_FRAME_MS
    } else {
        token
    }
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

/// How long the spill splash stays in the status line.
///
/// A spill is over in milliseconds — the file is written and the turn carries
/// on — so without a window of its own the indicator would flash for one
/// frame and be gone. Three seconds is long enough to be read by someone who
/// was not already looking at the footer.
pub const SPILL_FLASH_MS: u64 = 3_000;

/// When output was last spilled to a file, for [`spill_active`].
static SPILL_AT: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

/// Marks output as just spilled, starting the [`SPILL_FLASH_MS`] window.
pub fn note_spill() {
    let mut guard = SPILL_AT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some(std::time::Instant::now());
}

/// Whether the spill splash should be shown. Expired marks are dropped on
/// read, the way [`flash_tip`] drops expired tips.
#[must_use]
pub fn spill_active() -> bool {
    let mut guard = SPILL_AT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match guard.as_ref() {
        Some(at) if flash_active(at.elapsed(), SPILL_FLASH_MS) => true,
        Some(_) => {
            *guard = None;
            false
        }
        None => false,
    }
}

/// Clears the spill splash immediately (tests, and `/clear`).
pub fn clear_spill() {
    let mut guard = SPILL_AT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = None;
}

/// The spill splash as a bar segment, when one is due.
///
/// Read by the TUI footer, which is the only front end painting while tools
/// run — and deliberately not by [`build_status_text`], which stays a pure
/// function of its [`Status`] rather than of when output was last spilled.
#[must_use]
pub fn spill_segment() -> Option<&'static str> {
    spill_active().then_some(SPILL_MARK)
}

/// Glyph shown while output is being spilled to a file.
pub const SPILL_MARK: &str = "💦";

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

/// The bar's trailing power slot, now always empty: the reading moved next to
/// the local origin, where it names the engine it actually caps. Kept as a
/// seam rather than deleted so the body arms below stay one shape.
fn power_suffix(_st: &Status) -> String {
    String::new()
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
    build_status_text_with_cells(st, color, progress_in_bar, usize::MAX)
}

/// [`build_status_text`] keeping at most `cells` contributed segments.
///
/// The elision knob for [`build_status_text_within`]. Private: callers ask for
/// a width, not for a cell count.
fn build_status_text_with_cells(
    st: &Status,
    color: bool,
    progress_in_bar: bool,
    cells: usize,
) -> String {
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
    // The level itself is temperature-colored (see `think_color`); the mark and
    // separator stay in the footer's own style.
    let level = if color {
        format!(
            "\x1b[38;5;{}m{}{STATUS_STYLE_START}",
            think_color(st.think),
            st.think.short_name()
        )
    } else {
        st.think.short_name().to_owned()
    };
    let think = format!("{THINK_MARK} {level} | ");
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
        // The git stat segment rides with the branch, inside the dir prefix:
        // it answers "what have I changed in this tree", which is the same
        // question the path and branch answer, one level down.
        let stat = git_stat_segment(color).map_or(String::new(), |s| format!(" | {s}"));
        format!(
            "{} {POWERLINE_BRANCH} {}{stat} | {origin}",
            theme(&cwd),
            theme(&branch)
        )
    } else {
        format!("{} | {origin}", theme(&cwd))
    };
    let ctx = format!("{think}{ctx}");
    // The dspark segment sits with the ctx gauge rather than in the state word:
    // it describes the whole turn, and keeping it left of the state keeps the
    // power suffix anchored on the right.
    let ctx = match spec_segment(st) {
        Some(seg) => format!("{ctx} | {}", theme(&seg)),
        None => ctx,
    };
    // Contributed cells ride between the state word and the power suffix: the
    // suffix is the line's right anchor and must stay last.
    let power = format!("{}{power}", wasm_segment_text_keeping(cells, color));
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

/// The `--dspark` segment: mean tokens committed per speculative step, then the
/// share of the offered draft capacity that survived verification.
///
/// `None` when the pass never speculated, so a plain run's footer is unchanged.
///
/// Rendered `1.5t/step`, never `1.5x`: it is a per-step token count, not a
/// wall-clock speedup, and the two diverge badly. See
/// [`SpecStats::tokens_per_step`](crate::engine::SpecStats::tokens_per_step).
#[must_use]
pub fn spec_segment(st: &Status) -> Option<String> {
    if !st.spec.active() {
        return None;
    }
    Some(format!(
        "{SPEC_MARK}{:.1}t/step {:.0}%",
        st.spec.tokens_per_step(),
        100.0 * st.spec.block_fill()
    ))
}

/// [`build_status_text`] fitted into `cols` columns.
///
/// When the line overflows, contributed cells are dropped lowest-priority
/// first until it fits, and only then is the result truncated. Plain
/// truncation alone cuts the *right* edge, which is where the power suffix
/// lives — the one segment documented as the line's anchor — so a plugin cell
/// could push the bar's own status off the screen.
///
/// The built-in segments are never elided: they are not competing for space on
/// a plugin's behalf, and a user who cannot see the context gauge because a
/// component had something to say is worse off than one who cannot see the
/// component.
#[must_use]
pub fn build_status_text_within(
    st: &Status,
    color: bool,
    progress_in_bar: bool,
    cols: usize,
) -> String {
    let full = build_status_text(st, color, progress_in_bar);
    if visible_width(&full) <= cols {
        return full;
    }
    let total = WASM_SEGMENTS.read().map_or(0, |s| s.len());
    // Drop one cell at a time rather than computing a budget: the cells are
    // already ordered, and the number of them is single digits.
    for keep in (0..total).rev() {
        let candidate = build_status_text_with_cells(st, color, progress_in_bar, keep);
        if visible_width(&candidate) <= cols {
            return candidate;
        }
    }
    build_status_text_with_cells(st, color, progress_in_bar, 0)
}

/// Visible width, ignoring ANSI escapes so a coloured line is not judged by
/// the length of its escape sequences.
fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut in_escape = false;
    for c in text.chars() {
        if in_escape {
            // A CSI sequence ends at its final byte, which is alphabetic.
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else if c == '\u{1b}' {
            in_escape = true;
        } else {
            width += 1;
        }
    }
    width
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

    /// Elision drops the lowest-priority cells first and keeps the order.
    ///
    /// Pure: it takes the cells rather than reaching for the published global,
    /// which parallel tests would otherwise fight over — the first version of
    /// this test set that static and broke an unrelated one.
    #[test]
    fn cells_are_elided_lowest_priority_first() {
        // As published: highest priority first.
        let cell = |text: &str, priority: u8| Cell {
            text: text.to_string(),
            priority,
            fg: None,
            bg: None,
        };
        let cells = vec![
            cell("IMPORTANT", 200),
            cell("middling", 128),
            cell("chatty", 10),
        ];
        assert_eq!(
            elide_cells_styled(&cells, usize::MAX, false),
            " | IMPORTANT | middling | chatty"
        );
        assert_eq!(
            elide_cells_styled(&cells, 2, false),
            " | IMPORTANT | middling"
        );
        assert_eq!(elide_cells_styled(&cells, 1, false), " | IMPORTANT");
        // Nothing kept is the empty string, not a dangling separator that
        // would leave the bar ending in " | ".
        assert_eq!(elide_cells_styled(&cells, 0, false), "");
        assert_eq!(elide_cells_styled(&[], usize::MAX, false), "");
    }

    /// A cell's requested colours are painted, and closed by returning to the
    /// bar's own style rather than by a reset — a reset would drop the status
    /// bar's background for the rest of the line.
    #[test]
    fn a_cells_colours_are_painted_and_closed_to_the_bar_style() {
        let coloured = vec![Cell {
            text: "BUSY".to_string(),
            priority: 200,
            fg: Some((255, 0, 0)),
            bg: Some((0, 0, 128)),
        }];
        // Without colour the escapes must not appear at all: a piped run gets
        // plain text.
        assert_eq!(elide_cells_styled(&coloured, 1, false), " | BUSY");
        let painted = elide_cells_styled(&coloured, 1, true);
        assert!(painted.contains("\x1b[38;2;255;0;0m"), "{painted:?}");
        assert!(painted.contains("\x1b[48;2;0;0;128m"), "{painted:?}");
        assert!(painted.ends_with(STATUS_STYLE_START), "{painted:?}");
        // Colour must not change how wide the bar thinks the cell is.
        assert_eq!(visible_width(&painted), " | BUSY".len());

        // A cell with no opinion stays untouched even in colour mode.
        let plain = vec![Cell {
            text: "quiet".to_string(),
            priority: 1,
            fg: None,
            bg: None,
        }];
        assert_eq!(elide_cells_styled(&plain, 1, true), " | quiet");
    }

    /// Widths are measured in visible columns, so a coloured line is not judged
    /// by the length of its escape sequences.
    #[test]
    fn visible_width_ignores_ansi() {
        assert_eq!(visible_width("plain"), 5);
        assert_eq!(visible_width("\x1b[1;31mred\x1b[0m"), 3);
        assert_eq!(visible_width("\x1b[38;5;120mgreen\x1b[0m!"), 6);
    }
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
    fn dspark_segment_appears_only_when_a_pass_speculated() {
        // A run without a support model must look exactly as it did before.
        let plain = Status {
            ctx_used: 1000,
            ctx_size: 8000,
            ..Status::default()
        };
        let line = build_status_text(&plain, false, true);
        assert!(line.ends_with("ctx 12% | idle"), "{line}");
        assert!(!line.contains(SPEC_MARK), "{line}");

        // 10 steps of a 4-token block, 30 committed: 3.0 per step, 50% fill.
        let spark = Status {
            spec: crate::engine::SpecStats {
                steps: 10,
                committed: 30,
                drafted: 40,
            },
            ..plain
        };
        let line = build_status_text(&spark, false, true);
        assert!(
            line.ends_with("ctx 12% | \u{23e9}3.0t/step 50% | idle"),
            "{line}"
        );
    }

    #[test]
    fn dspark_segment_survives_into_the_idle_footer() {
        // The figures are only readable after the answer lands, so an idle
        // footer carrying them is the point of the feature, not an artefact.
        let st = Status {
            state: WorkerState::Idle,
            ctx_used: 10,
            ctx_size: 100,
            spec: crate::engine::SpecStats {
                steps: 4,
                committed: 4,
                drafted: 16,
            },
            ..Status::default()
        };
        let line = build_status_text(&st, false, true);
        // Every draft rejected: 1.0 per step and 0%, still shown — "speculation is on
        // and buying nothing" is exactly what a user needs to see.
        assert!(line.contains("\u{23e9}1.0t/step 0%"), "{line}");
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

    /// Several engines can be live at once — a provider main agent with a
    /// `provider: local` sub-agent, or definitions on different hosts — so the
    /// footer names all of them rather than only whichever registered first.
    #[test]
    fn engine_origins_list_every_engine_in_play() {
        let _guard = origin_test_guard();
        reset_engine_origins();
        // Nothing registered: the local engine is the implied answer.
        assert_eq!(engine_origin_label(), local_origin_label());

        set_engine_origin("regolo.ai");
        assert_eq!(engine_origin_label(), "regolo.ai");

        // A second engine joins, in first-seen order, so the main agent leads.
        // The local one renders through `local_origin_label`, power badge and
        // all — registering `LOCAL_ORIGIN` is what selects that rendering.
        let local = local_origin_label();
        set_engine_origin(LOCAL_ORIGIN);
        assert_eq!(engine_origin_label(), format!("regolo.ai, {local}"));

        // Repeats are dropped: two definitions on one host name it once.
        set_engine_origin("regolo.ai");
        set_engine_origin("api.anthropic.com");
        set_engine_origin(LOCAL_ORIGIN);
        assert_eq!(
            engine_origin_label(),
            format!("regolo.ai, {local}, api.anthropic.com")
        );

        // An empty label is not a slot in the list.
        set_engine_origin("   ");
        assert_eq!(
            engine_origin_label(),
            format!("regolo.ai, {local}, api.anthropic.com")
        );
        reset_engine_origins();
    }

    #[test]
    fn footer_shows_the_engine_origin_beside_the_directory() {
        // No engine set this origin (unit tests build no engine), so the local
        // ds4/echo default stands. The label sits in the dir prefix, ahead of
        // the think segment — not appended to the tail.
        let _guard = origin_test_guard();
        let line = build_status_text(&Status::default(), false, true);
        let origin = line
            .find(local_origin_label().as_str())
            .expect("origin in footer");
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

    /// Every pool, in phase order. Kept here rather than in the module so the
    /// production code never invites a caller to iterate all phases blindly.
    const ALL_PHASES: [VerbPhase; 5] = [
        VerbPhase::Thinking,
        VerbPhase::Generating,
        VerbPhase::Tool,
        VerbPhase::Prefill,
        VerbPhase::Fun,
    ];

    #[test]
    fn spinner_verbs_are_200_and_unique_across_pools() {
        let all: Vec<&str> = ALL_PHASES
            .iter()
            .flat_map(|p| verbs_for(*p).iter().copied())
            .collect();
        assert_eq!(all.len(), 200);
        let set: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(set.len(), 200, "duplicate verb across pools");
    }

    #[test]
    fn every_pool_is_non_empty_so_the_modulo_cannot_divide_by_zero() {
        for phase in ALL_PHASES {
            assert!(!verbs_for(phase).is_empty(), "{phase:?}");
        }
    }

    #[test]
    fn verb_phase_follows_the_live_state() {
        clear_tool_activity();
        // 1 is coprime-ish with FUN_ODDS: any seed not divisible by it works.
        let base = Status {
            prefill_label: 1,
            ..Status::default()
        };
        assert_eq!(
            verb_phase(&Status {
                state: WorkerState::Prefill,
                ..base.clone()
            }),
            VerbPhase::Prefill
        );
        assert_eq!(
            verb_phase(&Status {
                state: WorkerState::Generating,
                thinking: true,
                ..base.clone()
            }),
            VerbPhase::Thinking
        );
        assert_eq!(
            verb_phase(&Status {
                state: WorkerState::Generating,
                ..base.clone()
            }),
            VerbPhase::Generating
        );
        {
            let _running = ToolActivity::begin("🔧 bash".to_owned());
            assert_eq!(
                verb_phase(&Status {
                    state: WorkerState::Generating,
                    thinking: true,
                    ..base.clone()
                }),
                VerbPhase::Tool,
                "a live tool dispatch outranks the last published sampling state"
            );
        }
        // Dropped: the linger window keeps the *label* up, but not the phase.
        assert!(tool_activity().is_some());
        assert_eq!(
            verb_phase(&Status {
                state: WorkerState::Generating,
                ..base
            }),
            VerbPhase::Generating
        );
        clear_tool_activity();
    }

    #[test]
    fn fun_verbs_win_regardless_of_phase() {
        clear_tool_activity();
        let st = Status {
            state: WorkerState::Prefill,
            prefill_label: 4 * FUN_ODDS,
            ..Status::default()
        };
        assert_eq!(verb_phase(&st), VerbPhase::Fun);
        assert!(FUN_VERBS.contains(&prefill_label(&st)));
    }

    #[test]
    fn fun_verbs_are_not_all_the_first_entry() {
        // The fun-odds roll constrains the seed to a multiple of FUN_ODDS, and
        // FUN_VERBS has FUN_ODDS entries, so a naive `seed % pool.len()` would
        // collapse every fun draw onto index 0. Every multiple of FUN_ODDS must
        // reach a distinct verb, and all of them must be reachable.
        clear_tool_activity();
        let drawn: std::collections::HashSet<_> = (0..FUN_VERBS.len())
            .map(|k| {
                #[allow(clippy::cast_possible_truncation)]
                let k = k as u32;
                let st = Status {
                    state: WorkerState::Prefill,
                    prefill_label: k * FUN_ODDS,
                    ..Status::default()
                };
                assert_eq!(verb_phase(&st), VerbPhase::Fun);
                prefill_label(&st)
            })
            .collect();
        assert_eq!(
            drawn.len(),
            FUN_VERBS.len(),
            "fun draws collapsed to {} distinct verb(s) instead of {}",
            drawn.len(),
            FUN_VERBS.len()
        );
    }

    #[test]
    fn the_verb_index_is_stable_while_the_phase_moves() {
        clear_tool_activity();
        let seed = 7;
        let thinking = Status {
            state: WorkerState::Generating,
            thinking: true,
            prefill_label: seed,
            ..Status::default()
        };
        let generating = Status {
            thinking: false,
            ..thinking.clone()
        };
        assert_eq!(
            prefill_label(&thinking),
            THINKING_VERBS[seed as usize % THINKING_VERBS.len()]
        );
        assert_eq!(
            prefill_label(&generating),
            GENERATING_VERBS[seed as usize % GENERATING_VERBS.len()]
        );
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

    /// The plug takes a rotation slot like any other tip — same window, same
    /// styling — and never the first one of a session.
    #[test]
    fn the_promo_takes_one_rotation_in_fifty() {
        let at = |rotation: u64| rotating_tip(TIP_ROTATE_MS * rotation);
        assert_ne!(at(0), PROMO_TIP, "a session opens on a useful tip");
        assert_eq!(at(PROMO_EVERY), PROMO_TIP);
        assert_eq!(at(PROMO_EVERY * 2), PROMO_TIP);
        assert_ne!(at(PROMO_EVERY - 1), PROMO_TIP);
        assert_ne!(at(PROMO_EVERY + 1), PROMO_TIP);

        // Exactly one slot in fifty, and it obeys the same visibility window as
        // the rest — no lingering advertisement.
        let promos = (1..=200).filter(|&r| at(r) == PROMO_TIP).count();
        assert_eq!(promos, 4, "one in {PROMO_EVERY}");
        assert_eq!(
            rotating_tip(TIP_ROTATE_MS * PROMO_EVERY + TIP_VISIBLE_MS),
            ""
        );

        // And it is a tip, not a banner: it never displaces the whole table.
        let distinct = (0..TIPS.len() as u64)
            .map(at)
            .collect::<std::collections::HashSet<_>>();
        assert!(
            distinct.len() > 40,
            "the ordinary tips still rotate: {}",
            distinct.len()
        );
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
    fn the_spill_splash_lasts_long_enough_to_be_read() {
        clear_spill();
        assert!(spill_segment().is_none(), "nothing spilled yet");
        note_spill();
        assert_eq!(spill_segment(), Some(SPILL_MARK));
        // Three seconds, so a spill that is over in milliseconds is still
        // seen by someone who was not watching the footer at the time.
        const { assert!(SPILL_FLASH_MS >= 3_000) };
        // Leave the process-global clean for other tests in this binary.
        clear_spill();
        assert!(spill_segment().is_none());
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

    /// The level is colored by temperature — red `max`, white `med`, blue `low`,
    /// grey `off` — and the four are distinct, which is the whole point: the
    /// color has to identify the level on its own, from three columns.
    ///
    /// Colored only when the caller asked for color; the TUI renders the footer
    /// plain and re-styles it span by span, so an escape leaking into that text
    /// would be drawn literally.
    #[test]
    fn the_think_level_is_colored_by_temperature() {
        use crate::engine::ThinkMode;

        let levels = [
            ThinkMode::Off,
            ThinkMode::Low,
            ThinkMode::Medium,
            ThinkMode::Max,
        ];
        let colors: std::collections::HashSet<u8> =
            levels.iter().map(|&m| think_color(m)).collect();
        assert_eq!(colors.len(), levels.len(), "a shared color hides a level");
        // Hotter effort, hotter color: max is the red one, off the grey one.
        assert_eq!(think_color(ThinkMode::Max), 196);
        assert_eq!(think_color(ThinkMode::Off), 245);

        for level in levels {
            let st = Status {
                think: level,
                ctx_size: 1000,
                ctx_used: 30,
                ..Status::default()
            };
            let want = format!("\x1b[38;5;{}m{}", think_color(level), level.short_name());
            let colored = build_status_text(&st, true, true);
            assert!(colored.contains(&want), "{level:?}: {colored:?}");
            let plain = build_status_text(&st, false, true);
            assert!(!plain.contains("\x1b["), "{level:?}: {plain:?}");
            assert!(plain.contains(level.short_name()), "{level:?}: {plain:?}");
        }
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

    /// The change counter belongs to the *location*, not the body: it describes
    /// the tree named by the path and branch, so it sits inside the dir prefix,
    /// after the branch and before the origin. (The TUI splits on
    /// `GIT_STAT_MARK` for the same reason it splits on `THINK_MARK`.)
    #[test]
    fn git_stat_segment_sits_between_the_branch_and_the_body() {
        let st = Status {
            ctx_used: 1,
            ctx_size: 100,
            ..Status::default()
        };
        let line = build_status_text(&st, false, true);
        // The repo under test may be clean, in which case the segment is absent
        // by design and there is nothing to place.
        let Some(mark) = line.find(GIT_STAT_MARK) else {
            assert!(git_stats().is_none_or(|s| s.is_clean()), "{line}");
            return;
        };
        let branch = line.find(POWERLINE_BRANCH).expect("branch glyph present");
        let think = line.find(THINK_MARK).expect("think segment present");
        assert!(branch < mark && mark < think, "{line}");
    }

    /// The counts are colored where they are read: bright green for additions,
    /// bright red for deletions, with the footer style restored after each so
    /// the bar's background survives.
    #[test]
    fn git_stat_counts_carry_their_own_colors() {
        let Some(seg) = git_stat_segment(true) else {
            return;
        };
        assert!(
            seg.contains(&format!("\x1b[38;5;{GIT_ADD_COLOR};1m+")),
            "{seg}"
        );
        assert!(
            seg.contains(&format!("\x1b[38;5;{GIT_DEL_COLOR};1m-")),
            "{seg}"
        );
        assert!(seg.ends_with(STATUS_STYLE_START), "{seg}");
    }

    /// A clean tree says nothing: the segment exists to report change, and a
    /// permanent `0 · +0 -0` would be three columns of noise.
    #[test]
    fn a_clean_tree_shows_no_git_stat_segment() {
        let clean = GitStats {
            files: 0,
            added: 0,
            deleted: 0,
        };
        assert!(clean.is_clean());
        assert!(
            !GitStats {
                files: 1,
                added: 0,
                deleted: 0
            }
            .is_clean()
        );
    }

    /// The power cap rides with the local origin, not the bar's tail: it caps
    /// that engine and no other, so beside a provider a trailing badge would
    /// read as a property of the session. Shown only when actually limiting.
    #[test]
    fn power_badge_rides_with_the_local_origin() {
        let _guard = origin_test_guard();
        reset_engine_origins();
        let st = Status {
            ctx_size: 100,
            ..Status::default()
        };

        set_local_power(50);
        let line = build_status_text(&st, false, true);
        // Inside the parentheses, so it reads as one label for one engine.
        assert!(line.contains("(local ⚡50%)"), "{line}");
        assert!(!line.contains("(local)"), "not left bare: {line}");
        assert!(!line.ends_with('%'), "not in the tail slot: {line}");

        // A provider alongside the local engine: the badge stays attached to
        // the engine it describes.
        set_engine_origin("regolo.ai");
        set_engine_origin(LOCAL_ORIGIN);
        let line = build_status_text(&st, false, true);
        assert!(line.contains("regolo.ai, (local ⚡50%)"), "{line}");

        // Uncapped still reads: "all of it" is an answer, and a badge that
        // vanished at 100% would make its absence ambiguous between unlimited
        // and unsupported.
        set_local_power(100);
        assert!(build_status_text(&st, false, true).contains("(local ⚡100%)"));
        // Unset is the default and means uncapped, so it reads the same.
        set_local_power(0);
        let line = build_status_text(&st, false, true);
        assert!(line.contains("(local ⚡100%)"), "{line}");
        assert!(!line.contains("(local)"), "never the bare label: {line}");
        set_local_power(0);
        reset_engine_origins();
    }

    #[test]
    fn user_echo_formats() {
        assert_eq!(format_user_prompt_echo("hi", false), "* hi\n\n");
        assert!(format_user_prompt_echo("hi", true).contains("\x1b[1;91m*"));
    }
}

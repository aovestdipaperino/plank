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

/// Bright highlight color for the shimmer sweeping across the spinner verb.
pub const SHIMMER_COLOR: u8 = 231;

/// Milliseconds per shimmer step (one display column of travel).
pub const SHIMMER_STEP_MS: u64 = 200;
/// ANSI style for the filled portion of the progress bar (theme color).
pub const STATUS_BAR_FILL: &str = "\x1b[48;5;238;38;5;106;1m";
/// ANSI style for the queued-prompt preview rows.
pub const QUEUE_STYLE: &str = "\x1b[38;5;87;1m";

/// Powerline branch glyph (U+E0A0), shown before the git branch name. Requires
/// a Powerline-patched or Nerd Font to render.
pub const POWERLINE_BRANCH: char = '\u{e0a0}';

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

fn power_suffix(st: &Status) -> String {
    if st.power_percent > 0 && st.power_percent < 100 {
        format!(" | ⚡ {}%", st.power_percent)
    } else {
        String::new()
    }
}

/// Braille throbber frame derived from wall-clock time, so any footer
/// repaint advances the animation and a pegged progress bar still shows
/// the worker is alive.
#[must_use]
pub fn throbber() -> char {
    const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    #[allow(clippy::cast_possible_truncation)]
    FRAMES[(ms / 100) as usize % FRAMES.len()]
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

/// Builds the compact one-line footer shown below the prompt.
#[must_use]
pub fn build_status_text(st: &Status, color: bool) -> String {
    let used = format_ctx_size(st.ctx_used);
    let total_ctx = if st.ctx_size > 0 {
        format!(
            "{} ({:.0}%)",
            format_ctx_size(st.ctx_size),
            100.0 * f64::from(st.ctx_used) / f64::from(st.ctx_size)
        )
    } else {
        format_ctx_size(st.ctx_size)
    };
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
    let dir = if cwd.is_empty() {
        String::new()
    } else if let Some(branch) = git_branch_label() {
        format!("{} {POWERLINE_BRANCH} {} | ", theme(&cwd), theme(&branch))
    } else {
        format!("{} | ", theme(&cwd))
    };
    let body = match st.state {
        WorkerState::Prefill => {
            let total = if st.prefill_total > 0 {
                st.prefill_total
            } else {
                1
            };
            let done = st.prefill_done.min(total);
            format!(
                "ctx {used}/{total_ctx} | {} {}… ({} · ↑ {}/{} tokens · {:.1} t/s){power}",
                throbber(),
                theme(prefill_label(st)),
                format_elapsed(st.elapsed_secs),
                format_ctx_size(done),
                format_ctx_size(total),
                st.prefill_tps
            )
        }
        WorkerState::Generating => format!(
            "ctx {used}/{total_ctx} | {} {}… ({} · ↓ {} tokens{} · {:.1} t/s){power}",
            throbber(),
            theme(prefill_label(st)),
            format_elapsed(st.elapsed_secs),
            format_ctx_size(st.generated),
            if st.greedy_sampling { " ❄️" } else { "" },
            st.gen_tps
        ),
        WorkerState::Compacting => format!(
            "ctx {used}/{total_ctx} | COMPACTING summary {} tokens {:.1} t/s{power}",
            st.generated, st.gen_tps
        ),
        WorkerState::Saving => format!("ctx {used}/{total_ctx} | saving session{power}"),
        WorkerState::Error => format!(
            "ctx {used}/{total_ctx} | error: {}{power}",
            if st.error.is_empty() {
                "unknown error"
            } else {
                &st.error
            }
        ),
        WorkerState::Stopped => format!("ctx {used}/{total_ctx} | interrupted{power}"),
        WorkerState::Idle => format!("ctx {used}/{total_ctx} | idle{power}"),
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

#[cfg(test)]
mod tests {
    use super::*;

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
            build_status_text(&st, false).ends_with("ctx 1k/8k (12%) | idle"),
            "{}",
            build_status_text(&st, false)
        );
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
        let line = build_status_text(&st, false);
        assert!(line.contains("ctx 1.5k/8k (19%) | "), "{line}");
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
        let line = build_status_text(&st, false);
        assert!(line.contains("ctx 1.5k/8k (19%) | "), "{line}");
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
    fn power_suffix_shown_only_when_limited() {
        let mut st = Status {
            ctx_size: 100,
            power_percent: 50,
            ..Status::default()
        };
        assert!(build_status_text(&st, false).ends_with("⚡ 50%"));
        st.power_percent = 100;
        assert!(!build_status_text(&st, false).contains('⚡'));
    }

    #[test]
    fn user_echo_formats() {
        assert_eq!(format_user_prompt_echo("hi", false), "* hi\n\n");
        assert!(format_user_prompt_echo("hi", true).contains("\x1b[1;91m*"));
    }
}

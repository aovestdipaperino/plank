//! Terminal prompt, status footer, and progress rendering.
//!
//! Port of the "Terminal Prompt, Status Footer" section of `ds4_agent.c`:
//! the compact one-line footer, the prefill progress bar with an embedded
//! t/s readout, and the user prompt echo styling.

/// ANSI style opening for the status footer row.
pub const STATUS_STYLE_START: &str = "\x1b[48;5;238;38;5;252m";
/// ANSI style reset.
pub const STATUS_STYLE_END: &str = "\x1b[0m";
/// ANSI style for the filled portion of the progress bar.
pub const STATUS_BAR_FILL: &str = "\x1b[48;5;238;38;5;201;1m";
/// ANSI style for the queued-prompt preview rows.
pub const QUEUE_STYLE: &str = "\x1b[38;5;87;1m";

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
    /// Stable index selecting the playful prefill label.
    pub prefill_label: u32,
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
    "plank> "
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

/// Keep each prefill operation on a single playful label so the footer does
/// not visually churn while progress updates stream in.
#[must_use]
pub fn prefill_label(st: &Status) -> &'static str {
    const LABELS: [&str; 6] = [
        "reading",
        "absorbing",
        "studying",
        "gathering",
        "crunching",
        "scrutinizing",
    ];
    LABELS[st.prefill_label as usize % LABELS.len()]
}

fn power_suffix(st: &Status) -> String {
    if st.power_percent > 0 && st.power_percent < 100 {
        format!(" | ⚡ {}%", st.power_percent)
    } else {
        String::new()
    }
}

/// Renders the prefill progress bar with an embedded t/s readout.
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
    let rate = if tps > 0.0 && filled < PROGRESS_BAR_WIDTH {
        format!(" {tps:.0}t/s")
    } else {
        String::new()
    };
    let rate = rate.as_bytes();

    let mut out = String::from("[");
    if color {
        out.push_str(STATUS_BAR_FILL);
    }
    for i in 0..PROGRESS_BAR_WIDTH {
        if color && i == filled {
            out.push_str(STATUS_STYLE_START);
        }
        if i >= filled && i - filled < rate.len() {
            out.push(rate[i - filled] as char);
        } else {
            out.push_str(if i < filled { "▶" } else { "·" });
        }
    }
    if color {
        out.push_str(STATUS_STYLE_START);
    }
    out.push(']');
    out
}

/// Builds the compact one-line footer shown below the prompt.
#[must_use]
pub fn build_status_text(st: &Status, color: bool) -> String {
    let used = format_ctx_size(st.ctx_used);
    let total_ctx = format_ctx_size(st.ctx_size);
    let power = power_suffix(st);
    match st.state {
        WorkerState::Prefill => {
            let total = if st.prefill_total > 0 {
                st.prefill_total
            } else {
                1
            };
            let done = st.prefill_done.min(total);
            let pct = 100.0 * f64::from(done) / f64::from(total);
            let bar = progress_bar(done, total, st.prefill_tps, color);
            format!(
                "ctx {used}/{total_ctx} | {} {bar} {done}/{total} {pct:.1}%{power}",
                prefill_label(st)
            )
        }
        WorkerState::Generating => format!(
            "ctx {used}/{total_ctx} | generation {} tokens{} {:.1} t/s{power}",
            st.generated,
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
    }
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
        assert_eq!(build_status_text(&st, false), "ctx 1k/8k | idle");
    }

    #[test]
    fn generating_status_line() {
        let st = Status {
            state: WorkerState::Generating,
            generated: 42,
            gen_tps: 9.5,
            ctx_used: 1500,
            ctx_size: 8000,
            ..Status::default()
        };
        assert_eq!(
            build_status_text(&st, false),
            "ctx 1.5k/8k | generation 42 tokens 9.5 t/s"
        );
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

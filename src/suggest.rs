//! Prompt suggestions: the part that needs no model.
//!
//! A suggestion is a guess at the next thing the *user* will want to type,
//! generated after a turn and shown as ghost text. Everything here — the
//! instruction text, turning a raw reply into something showable, and
//! deciding what the idle moment should spend itself on — is pure, so it is
//! tested with no engine present.
//!
//! Design: `docs/superpowers/specs/2026-09-23-prompt-suggestions-design.md`.

use std::time::Duration;

/// Longest suggestion that is still a glanceable hint. About one editor line
/// at a typical terminal width; past that it wraps and stops being readable
/// at a glance, so it is rejected rather than truncated — a half-sentence
/// hint is worse than none.
pub const MAX_LEN: usize = 120;

/// The instruction handed to the model.
///
/// Written as separate pushes in [`prompt`] rather than one literal because a
/// `\`-continued Rust string literal strips the next line's leading
/// whitespace, and this is model-facing text.
pub const PROMPT: &str = "Suggest the single next thing the user is most likely to type next, based on the conversation so far. Reply with that line and nothing else: no quotes, no explanation, no preamble. Write it as the user would write it, in the imperative, addressed to you. If nothing useful comes to mind, reply with nothing at all.";

/// Openings that mean the model answered as itself rather than writing as the
/// user. It was asked for a line the user would type; a first-person report
/// or an assistant-voice opener is a failed suggestion, not a short one.
const ASSISTANT_VOICE: [&str; 10] = [
    "i've ",
    "i have ",
    "i'll ",
    "i will ",
    "i cannot ",
    "i can't ",
    "i'm ",
    "sure!",
    "sure,",
    "here's ",
];

/// One suggestion, bound to the transcript it was generated against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    pub text: String,
    /// Transcript length when this was generated. A suggestion whose depth no
    /// longer matches is stale and must not be shown: it was a guess about a
    /// conversation that has since moved.
    pub depth: usize,
}

/// What the idle moment should spend itself on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleWork {
    Suggestion,
    MemoryPass,
    Nothing,
}

/// Turns a raw generation into something showable, or `None`.
///
/// `None` shows nothing. There is deliberately no fallback text and no retry:
/// a bad suggestion costs more than a missing one.
#[must_use]
pub fn sanitize(reply: &str) -> Option<String> {
    let line = reply.lines().map(str::trim).find(|l| !l.is_empty())?;

    // Strip one leading list or quote marker, then surrounding quotes.
    let line = line
        .strip_prefix("- ")
        .or_else(|| line.strip_prefix("* "))
        .or_else(|| line.strip_prefix("> "))
        .unwrap_or(line);
    let line = match line.split_once(". ") {
        // "1. add tests" — a numbered marker, not a sentence.
        Some((head, rest)) if !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()) => rest,
        _ => line,
    };
    let line = line
        .trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim();

    if line.is_empty() || line.len() > MAX_LEN {
        return None;
    }
    let lowered = line.to_ascii_lowercase();
    if ASSISTANT_VOICE.iter().any(|p| lowered.starts_with(p)) {
        return None;
    }
    Some(line.to_string())
}

/// The instruction text, as a user-turn preamble.
#[must_use]
pub fn prompt() -> String {
    let mut out = String::with_capacity(PROMPT.len() + 2);
    out.push_str(PROMPT);
    out.push('\n');
    out
}

/// Which of the two background jobs the idle moment should run.
///
/// A pending suggestion wins, because a suggestion that lands after the user
/// starts typing is wasted while a memory pass is explicitly allowed to defer
/// (`memory.minTurnSeconds`). The starvation window is what stops a fast
/// back-and-forth from locking the memory pass out forever: once the oldest
/// queued job has waited that long, it takes the slot back.
///
/// A `starvation` of zero therefore means "never give the suggestion
/// priority", not "never starve".
#[must_use]
pub fn idle_work(
    suggestion_pending: bool,
    memory_pending: bool,
    oldest_memory_wait: Option<Duration>,
    starvation: Duration,
) -> IdleWork {
    let starved = memory_pending && oldest_memory_wait.is_some_and(|w| w >= starvation);
    if suggestion_pending && !starved {
        return IdleWork::Suggestion;
    }
    if memory_pending {
        return IdleWork::MemoryPass;
    }
    if suggestion_pending {
        return IdleWork::Suggestion;
    }
    IdleWork::Nothing
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn sanitize_takes_the_first_non_empty_line() {
        assert_eq!(
            sanitize("\n\nadd tests for the parser\nand then something else"),
            Some("add tests for the parser".to_string())
        );
    }

    #[test]
    fn sanitize_strips_surrounding_quotes_and_list_markers() {
        assert_eq!(sanitize("\"add tests\""), Some("add tests".to_string()));
        assert_eq!(sanitize("- add tests"), Some("add tests".to_string()));
        assert_eq!(sanitize("> add tests"), Some("add tests".to_string()));
        assert_eq!(sanitize("1. add tests"), Some("add tests".to_string()));
    }

    #[test]
    fn sanitize_rejects_empty_and_whitespace_only() {
        assert_eq!(sanitize(""), None);
        assert_eq!(sanitize("   \n  \n"), None);
        assert_eq!(sanitize("\"\""), None);
    }

    #[test]
    fn sanitize_rejects_anything_past_the_length_cap() {
        let long = "x".repeat(MAX_LEN + 1);
        assert_eq!(sanitize(&long), None, "a wrapped hint is not glanceable");
        let ok = "y".repeat(MAX_LEN);
        assert_eq!(sanitize(&ok), Some(ok));
    }

    /// The model is asked to write AS THE USER and sometimes answers as
    /// itself instead. Those replies must not reach the input line.
    #[test]
    fn sanitize_rejects_the_model_answering_as_itself() {
        assert_eq!(sanitize("I've added the tests you asked for."), None);
        assert_eq!(sanitize("Sure! Here's what I would suggest:"), None);
        assert_eq!(sanitize("I cannot help with that."), None);
    }

    #[test]
    fn a_pending_suggestion_takes_the_slot_ahead_of_a_memory_job() {
        assert_eq!(
            idle_work(
                true,
                true,
                Some(Duration::from_secs(1)),
                Duration::from_secs(300)
            ),
            IdleWork::Suggestion
        );
    }

    #[test]
    fn a_starved_memory_job_takes_the_slot_back() {
        assert_eq!(
            idle_work(
                true,
                true,
                Some(Duration::from_secs(301)),
                Duration::from_secs(300)
            ),
            IdleWork::MemoryPass,
            "a memory must not be locked out by a fast back-and-forth"
        );
    }

    #[test]
    fn the_memory_pass_runs_when_no_suggestion_is_pending() {
        assert_eq!(
            idle_work(
                false,
                true,
                Some(Duration::from_secs(1)),
                Duration::from_secs(300)
            ),
            IdleWork::MemoryPass
        );
    }

    #[test]
    fn nothing_pending_means_nothing_runs() {
        assert_eq!(
            idle_work(false, false, None, Duration::from_secs(300)),
            IdleWork::Nothing
        );
    }

    #[test]
    fn a_zero_starvation_window_always_prefers_the_memory_pass() {
        assert_eq!(
            idle_work(true, true, Some(Duration::ZERO), Duration::ZERO),
            IdleWork::MemoryPass,
            "0 disables the suggestion's priority rather than meaning 'never starve'"
        );
    }
}

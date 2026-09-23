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
///
/// A line starting with `/` or `!` is rejected outright. A suggestion is a
/// *prompt*, and Enter over a placed suggestion submits it immediately: a
/// model that wrote `/clear the session` would run a slash command on one
/// keystroke, and `!cargo test --release` would run a shell command. One
/// keypress away from either is not a place to be relaxed, so the model's
/// output never reaches the command dispatcher or a shell.
#[must_use]
pub fn sanitize(reply: &str) -> Option<String> {
    // Drop the model's reasoning first. This family generates *inside* an
    // implicit think block and emits only the closing tag, so a reply
    // routinely arrives as `</think>Write out the …` — the transcript
    // carries `</think>` with no opening partner (see `debugmirror`'s
    // `needs_think_prefix`). Taking the first line without cutting here put
    // a literal `</think>` on the user's input line.
    //
    // Cut at the LAST close, not the first: reasoning that itself mentions
    // the tag would otherwise leave a fragment behind.
    let reply = match reply.rfind("</think>") {
        Some(at) => &reply[at + "</think>".len()..],
        None => reply,
    };
    // A think block that never closed means the whole budget went on
    // reasoning and no suggestion was reached. There is nothing to show.
    if reply.contains("<think>") {
        return None;
    }
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

    // Characters, not bytes. `len()` would reject a perfectly short
    // suggestion written in any language whose letters are multi-byte —
    // "rivedi le modifiche all'interfaccia" costs more bytes than chars, and
    // the cap is about how much fits on one line, not how much it weighs.
    if line.is_empty() || line.chars().count() > MAX_LEN {
        return None;
    }
    // `/` is a slash command and `!` is shell execution — `!cmd` runs it and
    // records it, `!!cmd` runs it silently, both checked before anything else
    // and neither asking for confirmation. Enter over a placed suggestion
    // submits immediately, so either prefix would let the model run something
    // the user never typed, on one keystroke. A suggestion is a *prompt*; the
    // model's output never reaches the command dispatcher or a shell.
    if line.starts_with('/') || line.starts_with('!') {
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

    /// The same hazard as the slash guard, on a different prefix: plank's
    /// submit arm treats `!cmd` as shell execution and runs it without
    /// confirmation, `!!cmd` silently. A suggested `!cargo test --release`
    /// would run on one keystroke.
    #[test]
    fn sanitize_refuses_a_shell_command() {
        assert_eq!(sanitize("!cargo test --release"), None);
        assert_eq!(sanitize("!!rm -rf target"), None);
    }

    #[test]
    fn sanitize_refuses_a_slash_command() {
        // Enter over a placed suggestion submits it, so a leading `/` would
        // be one keystroke from running a command the user never typed.
        assert_eq!(sanitize("/clear the session"), None);
        assert_eq!(sanitize("\"/compact\""), None);
        assert_eq!(sanitize("- /quit"), None);
        // A slash anywhere else is ordinary prose.
        assert_eq!(
            sanitize("check src/suggest.rs"),
            Some("check src/suggest.rs".to_string())
        );
    }

    /// Reported from a live session: the ghost showed a literal
    /// `</think>Write out the arena-based doubly-linked list with tests.`
    ///
    /// This model family generates inside an implicit think block and emits
    /// only the closing tag, so the reply arrives with `</think>` glued to
    /// the front of the real answer — on the same line, which is why taking
    /// the first line was not enough.
    #[test]
    fn sanitize_drops_the_reasoning_close_glued_to_the_answer() {
        assert_eq!(
            sanitize("</think>Write out the arena-based doubly-linked list with tests."),
            Some("Write out the arena-based doubly-linked list with tests.".to_string())
        );
    }

    #[test]
    fn sanitize_drops_a_whole_reasoning_block_before_the_answer() {
        assert_eq!(
            sanitize(
                "the user just added a parser\nso tests are next</think>\nadd tests for the parser"
            ),
            Some("add tests for the parser".to_string())
        );
    }

    /// Reasoning that mentions the tag must not leave a fragment: cut at the
    /// last close, not the first.
    #[test]
    fn sanitize_cuts_at_the_last_reasoning_close() {
        assert_eq!(
            sanitize("I should not write </think> here</think>run the tests"),
            Some("run the tests".to_string())
        );
    }

    /// The token budget ran out mid-thought, so no suggestion was ever
    /// reached. Showing the reasoning would be worse than showing nothing.
    #[test]
    fn sanitize_rejects_an_unclosed_reasoning_block() {
        assert_eq!(sanitize("<think>the user probably wants"), None);
    }

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

    /// The cap counts characters, not bytes: a suggestion in a language with
    /// multi-byte letters must not be rejected for a length it never had.
    #[test]
    fn the_length_cap_counts_characters_not_bytes() {
        // 120 accented chars — 240 bytes, so a byte cap would reject it.
        let accented = "à".repeat(MAX_LEN);
        assert_eq!(accented.len(), MAX_LEN * 2, "precondition: multi-byte");
        assert_eq!(sanitize(&accented), Some(accented.clone()));

        let too_long = "à".repeat(MAX_LEN + 1);
        assert_eq!(sanitize(&too_long), None, "still capped, just in chars");
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

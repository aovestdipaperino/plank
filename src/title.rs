// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal window title, kept in sync with what plank is doing.
//!
//! Three states, so the window (and tab) names plank's phase at a glance:
//! `🚀 Plank loading...` before a front end is up, `🪵 Plank - READY.` while
//! idle at the prompt, and `🚀 <prompt>` while a turn runs. Set via the OSC 0
//! escape (`ESC ] 0 ; title BEL`), written to **stderr** in a single write:
//! stderr reaches the same tty as stdout but bypasses the Ratatui frame
//! buffer, so a title change can never tear a frame even when emitted from the
//! worker thread. No-op when stderr is not a terminal (piped runs, tests,
//! `--non-interactive` under a harness).

use std::io::{IsTerminal, Write};

/// What plank is doing, as reflected in the window title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State<'a> {
    /// Starting up — no front end is accepting input yet.
    Loading,
    /// Sitting at the prompt, waiting for the user.
    Idle,
    /// Running a turn for the given user prompt.
    Busy(&'a str),
}

/// Longest prompt (in characters) kept in a [`State::Busy`] title before it is
/// truncated with an ellipsis.
const TITLE_PROMPT_MAX: usize = 20;

/// Title shown while starting up — including the KV-cache prefill, which is the
/// slowest launch step and the one most likely to be looked at.
const LOADING: &str = "🚀 Plank loading...";

/// Formats the window title for `state`. A [`State::Busy`] prompt is collapsed
/// to one line and truncated past [`TITLE_PROMPT_MAX`] characters; a
/// whitespace-only prompt degrades to the plain loading form.
#[must_use]
pub fn window_title(state: State<'_>) -> String {
    let prompt = match state {
        State::Loading => return LOADING.to_string(),
        State::Idle => return "🪵 Plank - READY.".to_string(),
        State::Busy(p) => p.split_whitespace().collect::<Vec<_>>().join(" "),
    };
    if prompt.is_empty() {
        return LOADING.to_string();
    }
    match prompt.char_indices().nth(TITLE_PROMPT_MAX) {
        Some((i, _)) => format!("🚀 {}…", prompt[..i].trim_end()),
        None => format!("🚀 {prompt}"),
    }
}

/// Sets the terminal window title to [`window_title`]`(state)`. Best-effort:
/// errors are ignored, and nothing is written when stderr is not a tty.
pub fn set(state: State<'_>) {
    let mut err = std::io::stderr();
    if !err.is_terminal() {
        return;
    }
    // OSC 0 (icon + window title), BEL-terminated — the most widely supported
    // form. One write so it cannot interleave with other stderr output.
    let seq = format!("\x1b]0;{}\x07", window_title(state));
    let _ = err.write_all(seq.as_bytes());
    let _ = err.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_and_idle_are_fixed_strings() {
        assert_eq!(window_title(State::Loading), "🚀 Plank loading...");
        assert_eq!(window_title(State::Idle), "🪵 Plank - READY.");
    }

    #[test]
    fn blank_busy_prompt_falls_back_to_loading() {
        assert_eq!(window_title(State::Busy("   ")), "🚀 Plank loading...");
        assert_eq!(window_title(State::Busy("")), "🚀 Plank loading...");
    }

    #[test]
    fn busy_prompt_is_collapsed_and_truncated() {
        assert_eq!(window_title(State::Busy("fix  the\nbug")), "🚀 fix the bug");
        let long = "a".repeat(60);
        let t = window_title(State::Busy(&long));
        assert!(t.starts_with("🚀 "));
        assert!(t.ends_with('…'));
        assert_eq!(
            t.chars().count(),
            "🚀 ".chars().count() + TITLE_PROMPT_MAX + 1
        );
    }
}

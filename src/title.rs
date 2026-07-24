// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal window title, kept in sync with what plank is working on.
//!
//! The title is `🪵 plank`, extended to `🪵 plank - <prompt>` while a turn is
//! running so the window (and tab) names the task at a glance. Set via the
//! OSC 0 escape (`ESC ] 0 ; title BEL`), written to **stderr** in a single
//! write: stderr reaches the same tty as stdout but bypasses the Ratatui
//! frame buffer, so a title change can never tear a frame even when emitted
//! from the worker thread. No-op when stderr is not a terminal (piped runs,
//! tests, `--non-interactive` under a harness).

use std::io::{IsTerminal, Write};

/// Formats the window title: `🪵 plank`, plus ` - <prompt>` when a prompt is
/// given (collapsed to one line, truncated with an ellipsis past
/// [`TITLE_PROMPT_MAX`] characters). A whitespace-only prompt is treated as
/// absent.
#[must_use]
pub fn window_title(prompt: Option<&str>) -> String {
    const TITLE_PROMPT_MAX: usize = 40;
    let collapsed = prompt.map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "));
    match collapsed.as_deref() {
        None | Some("") => "🪵 plank".to_string(),
        Some(p) => match p.char_indices().nth(TITLE_PROMPT_MAX) {
            Some((i, _)) => format!("🪵 plank - {}…", p[..i].trim_end()),
            None => format!("🪵 plank - {p}"),
        },
    }
}

/// Sets the terminal window title to [`window_title`]`(prompt)`. Best-effort:
/// errors are ignored, and nothing is written when stderr is not a tty.
pub fn set(prompt: Option<&str>) {
    let mut err = std::io::stderr();
    if !err.is_terminal() {
        return;
    }
    // OSC 0 (icon + window title), BEL-terminated — the most widely supported
    // form. One write so it cannot interleave with other stderr output.
    let seq = format!("\x1b]0;{}\x07", window_title(prompt));
    let _ = err.write_all(seq.as_bytes());
    let _ = err.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_title_without_prompt() {
        assert_eq!(window_title(None), "🪵 plank");
        assert_eq!(window_title(Some("   ")), "🪵 plank");
        assert_eq!(window_title(Some("")), "🪵 plank");
    }

    #[test]
    fn prompt_is_appended_collapsed_and_truncated() {
        assert_eq!(
            window_title(Some("fix  the\nbug")),
            "🪵 plank - fix the bug"
        );
        let long = "a".repeat(60);
        let t = window_title(Some(&long));
        assert!(t.starts_with("🪵 plank - "));
        assert!(t.ends_with('…'));
        assert_eq!(t.chars().count(), "🪵 plank - ".chars().count() + 41);
    }
}

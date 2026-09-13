// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Single-line live progress display for prefill and generation.
//!
//! Shown on the current line and rewritten in place with a carriage return, so
//! it never emits a bare newline and cannot fight the line editor's own
//! terminal control (an earlier scroll-region version corrupted the screen).
//! The bar is cleared as soon as the model starts streaming text, so generated
//! output prints cleanly from column zero.

use std::io::Write;

use crate::status::{self, Status};

/// Terminal size `(rows, cols)` from `TIOCGWINSZ`, falling back to `(24, 80)`.
#[must_use]
pub fn term_size() -> (usize, usize) {
    // SAFETY: winsize is plain-old-data; zeroed is valid and ioctl overwrites.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: stdout fd is valid and `ws` is a writable winsize buffer.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &raw mut ws) };
    if rc == 0 && ws.ws_row > 0 && ws.ws_col > 0 {
        (ws.ws_row as usize, ws.ws_col as usize)
    } else {
        (24, 80)
    }
}

/// A single-line progress indicator rewritten in place.
#[derive(Debug)]
pub struct StatusBar {
    enabled: bool,
    color: bool,
    line_open: bool,
}

impl StatusBar {
    /// Creates a bar; `enabled` should be true only for interactive TTYs.
    #[must_use]
    pub fn new(enabled: bool, color: bool) -> Self {
        Self {
            enabled,
            color,
            line_open: false,
        }
    }

    /// Draws or redraws the status on the current line (no newline emitted).
    pub fn show(&mut self, st: &Status) {
        if !self.enabled {
            return;
        }
        let (_, cols) = term_size();
        // Width-aware: contributed plugin cells are dropped lowest-priority
        // first, and only what is left gets truncated. Truncation alone cuts
        // the right edge, which is the power suffix — the bar's own anchor.
        let mut line = status::build_status_text_within(st, self.color, true, cols);
        // Keep the status within one screen row so it never wraps, for the case
        // where the built-in segments alone exceed the width. Measured the
        // same way `build_status_text_within` fits its own candidates:
        // display columns via `unicode_width`, with ANSI escapes stripped
        // before measuring and always copied through whole when truncating.
        // A plain `chars().count()`/`take()` here would both undercount wide
        // emoji (so it would fail to catch an overflow) and, once `color` is
        // on, overcount by treating every byte of an escape sequence as a
        // visible column — which could truncate mid-escape and leave a
        // dangling, unterminated sequence bleeding into whatever prints next.
        if status::visible_width(&line) > cols {
            line = status::truncate_visible(&line, cols);
        }
        let mut out = std::io::stdout();
        // Carriage-return to column 0, paint, clear to end of line. No newline.
        if self.color {
            let _ = write!(
                out,
                "\r{}{}{}\x1b[K",
                status::STATUS_STYLE_START,
                line,
                status::STATUS_STYLE_END
            );
        } else {
            let _ = write!(out, "\r{line}\x1b[K");
        }
        let _ = out.flush();
        self.line_open = true;
    }

    /// Clears the status line so following output starts at column zero.
    pub fn clear(&mut self) {
        if !self.line_open {
            return;
        }
        self.line_open = false;
        let mut out = std::io::stdout();
        let _ = write!(out, "\r\x1b[K");
        let _ = out.flush();
    }
}

impl Drop for StatusBar {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_bar_is_noop() {
        let mut bar = StatusBar::new(false, false);
        bar.show(&Status::default());
        assert!(!bar.line_open);
        bar.clear();
    }

    /// Regression test for the truncation bug in `show`: a coloured line
    /// carrying a wide emoji, sitting right at the boundary of the terminal
    /// width, must be cut by display column (never mid-escape-sequence and
    /// never off by the emoji's extra column). This exercises the exact
    /// `visible_width`/`truncate_visible` pair `show` calls, since `show`
    /// itself writes straight to stdout and can't be captured here.
    #[test]
    fn boundary_width_truncation_handles_ansi_and_wide_emoji() {
        // 8 ASCII cols of red text, a 2-col emoji, then more red text and a
        // reset — 8 + 2 + 8 = 18 true columns, but naive `chars().count()`
        // would see every escape byte as a column and wildly overcount, or
        // (without colour) undercount the emoji and miss the overflow.
        let line = format!("\x1b[31m{}🧠{}\x1b[0m", "x".repeat(8), "y".repeat(8));
        assert_eq!(status::visible_width(&line), 18);

        // Exactly at the boundary: fits untouched.
        assert_eq!(status::truncate_visible(&line, 18), line);
        assert_eq!(
            status::visible_width(&status::truncate_visible(&line, 18)),
            18
        );

        // One column short: must drop exactly the trailing "y", keeping the
        // colour escapes intact (never truncated mid-sequence) and the emoji
        // whole.
        let cut = status::truncate_visible(&line, 17);
        assert_eq!(status::visible_width(&cut), 17);
        assert_eq!(cut, format!("\x1b[31m{}🧠{}", "x".repeat(8), "y".repeat(7)));

        // Cutting right after the emoji: the emoji itself must not be sliced
        // (it is atomic — either the whole 2-column glyph or none of it).
        let cut10 = status::truncate_visible(&line, 10);
        assert_eq!(status::visible_width(&cut10), 10);
        assert_eq!(cut10, format!("\x1b[31m{}🧠", "x".repeat(8)));
    }
}

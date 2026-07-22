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
        let mut line = status::build_status_text(st, self.color, true);
        // Keep the status within one screen row so it never wraps.
        if line.chars().count() > cols {
            line = line.chars().take(cols).collect();
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
}

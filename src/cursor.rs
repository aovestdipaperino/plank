// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Cursor colour, kept in sync with whether plank wants your input.
//!
//! The block cursor sitting on the prompt row is the one piece of the TUI the
//! eye is already resting on, so it doubles as the phase indicator: the theme
//! green while plank is idle and waiting for you, red while a turn is running.
//!
//! The colour used to be *requested* with the OSC 12 escape, and a terminal is
//! free to ignore that. Warp does: it renders the cursor as a widget in its own
//! UI, coloured from its theme, so there is no grid cell for OSC 12 to recolour
//! and no escape path into that widget. The same sequence that recolours the
//! cursor in Ghostty does nothing at all there — not an error, silence, which
//! is the worst failure mode for a visual indicator, because it works on your
//! machine. So plank stops asking and paints the cursor itself, into the frame
//! it is already drawing (see [`crate::tui`] `render_input`, which calls
//! `nano_cursor`).
//!
//! This module is what is left once the escapes are gone: two process-global
//! cells that cross the `terminal.draw` boundary.
//!
//! - The **phase**, written by [`set`] at the three moments the draw loops
//!   already know it (prompt live, turn running, compacting) and read by the
//!   renderer through [`state`]. Ambient rather than a parameter, because the
//!   call sites that know it and the renderer that needs it are separated by a
//!   closure and several widget signatures.
//! - The **caret**, written by the renderer through [`set_caret`] with the
//!   position `nano_cursor` reports, and consumed by [`place`] after the frame
//!   is flushed. plank never sets a ratatui cursor position, so ratatui emits
//!   `Hide` — but hidden is not absent: terminals anchor the IME candidate
//!   window and screen-reader focus to the position they are tracking, and
//!   hiding without moving is how you break CJK input.

use ratatui::layout::Position;
use ratatui::style::Color;
use std::io::IsTerminal;
use std::sync::atomic::{AtomicU8, Ordering};

/// Cursor colour while plank is at the prompt: the theme's military green,
/// `#87af5f` — the truecolor spelling of [`crate::status::THEME_COLOR`] (256
/// colour 106), written out longhand because a [`Color::Rgb`] cannot carry a
/// palette index.
pub const IDLE_COLOR: Color = Color::Rgb(0x87, 0xaf, 0x5f);

/// Cursor colour while a turn is running: `#d75f5f` (256 colour 167). A brick
/// red rather than a pure `#ff0000`, so it sits in the same muted family as the
/// rest of the palette instead of glaring out of the prompt row.
pub const BUSY_COLOR: Color = Color::Rgb(0xd7, 0x5f, 0x5f);

/// What the cursor colour is saying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum State {
    /// At the prompt, waiting for the user.
    #[default]
    Idle,
    /// Prefilling, generating, or compacting — plank owns the turn.
    Busy,
}

impl State {
    /// The colour this state paints the cursor.
    #[must_use]
    pub fn color(self) -> Color {
        match self {
            Self::Idle => IDLE_COLOR,
            Self::Busy => BUSY_COLOR,
        }
    }

    /// Discriminant stored in [`PHASE`]; `0` is reserved for "nothing written
    /// yet".
    fn tag(self) -> u8 {
        match self {
            Self::Idle => 1,
            Self::Busy => 2,
        }
    }

    /// Inverse of [`State::tag`]. `0` — never written — reads as [`State::Idle`]:
    /// plank is waiting for you until a turn says otherwise.
    fn from_tag(tag: u8) -> Self {
        if tag == Self::Busy.tag() {
            Self::Busy
        } else {
            Self::Idle
        }
    }
}

/// The phase the next drawn cursor will wear.
static PHASE: AtomicU8 = AtomicU8::new(0);

/// Records the phase for the frames drawn from here on. Cheap enough to call
/// once per frame, which is what the draw loops do.
pub fn set(state: State) {
    PHASE.store(state.tag(), Ordering::Relaxed);
}

/// The phase the renderer should paint.
#[must_use]
pub fn state() -> State {
    State::from_tag(PHASE.load(Ordering::Relaxed))
}

/// Where the cursor was painted on the last drawn frame, or `None` when that
/// frame drew no prompt.
static CARET: std::sync::Mutex<Option<Position>> = std::sync::Mutex::new(None);

/// Records the caret `nano_cursor` reported, or `None` on a frame with no
/// prompt. Cleared alongside `tui::set_input_rect(None)`: both answer "is
/// there a live prompt on this frame?" and must not drift apart.
pub fn set_caret(pos: Option<Position>) {
    if let Ok(mut slot) = CARET.lock() {
        *slot = pos;
    }
}

/// The caret from the last drawn frame.
#[must_use]
pub fn caret() -> Option<Position> {
    CARET.lock().ok().and_then(|c| *c)
}

/// Moves the terminal's own (hidden) cursor to the painted caret, so the IME
/// candidate window and screen-reader focus land where the user is typing.
///
/// Call right after `terminal.draw` returns — after the frame is flushed, so
/// this can never interleave with ratatui's own buffered writes and tear it.
/// A no-op when the frame painted no caret, which is what makes it safe to
/// call after every draw: the cell decides, not the call site.
pub fn place() {
    let Some(p) = caret() else {
        return;
    };
    let mut out = std::io::stdout();
    if !out.is_terminal() {
        return;
    }
    let _ = ratatui::crossterm::execute!(out, ratatui::crossterm::cursor::MoveTo(p.x, p.y));
}

/// Serializes tests that touch [`PHASE`] or [`CARET`]. Both are process
/// globals and libtest runs tests in parallel, so a test that reads either
/// one has to exclude every other test that writes it — including any test
/// that draws a prompt, since `render_input` writes the caret.
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_is_the_theme_green() {
        // The 256-color theme accent is 106 = #87af5f; the drawn cursor's
        // colour must track it, and a ratatui `Color::Rgb` cannot carry the
        // palette index, so the hue is written out longhand.
        assert_eq!(crate::status::THEME_COLOR, 106);
        assert_eq!(IDLE_COLOR, Color::Rgb(0x87, 0xaf, 0x5f));
        assert_eq!(State::Idle.color(), IDLE_COLOR);
    }

    #[test]
    fn busy_is_the_muted_brick_red() {
        // 256 colour 167, not a pure #ff0000: it sits in the same muted family
        // as the rest of the palette instead of glaring out of the prompt row.
        assert_eq!(BUSY_COLOR, Color::Rgb(0xd7, 0x5f, 0x5f));
        assert_eq!(State::Busy.color(), BUSY_COLOR);
    }

    #[test]
    fn the_phase_round_trips_through_the_store() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set(State::Busy);
        assert_eq!(state(), State::Busy);
        set(State::Idle);
        assert_eq!(state(), State::Idle);
    }

    #[test]
    fn the_phase_defaults_to_idle_before_anything_sets_it() {
        // Tag 0 is "never written". plank is waiting for you until a turn says
        // otherwise, so an unwritten phase must read as idle rather than red.
        assert_eq!(State::from_tag(0), State::Idle);
    }

    #[test]
    fn the_caret_cell_round_trips_and_clears() {
        // `place` moves the real cursor to whatever is in here, so a frame
        // that draws no prompt must be able to empty it — otherwise the
        // hidden cursor is left at a position from a previous frame.
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_caret(Some(Position::new(7, 3)));
        assert_eq!(caret(), Some(Position::new(7, 3)));
        set_caret(None);
        assert_eq!(caret(), None);
    }

    #[test]
    fn place_without_a_caret_does_nothing() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        set_caret(None);
        place(); // must not panic, and must not move anything
        assert_eq!(caret(), None);
    }
}

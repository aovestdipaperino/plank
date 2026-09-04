// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal cursor color, kept in sync with whether plank wants your input.
//!
//! The block cursor sitting on the prompt row is the one piece of the TUI the
//! eye is already resting on, so it doubles as the phase indicator: the theme
//! green while plank is idle and waiting for you, red while a turn is running.
//! Set via the OSC 12 escape (`ESC ] 12 ; #rrggbb BEL`) and undone with OSC 112
//! ("reset cursor color"), which hands the color back to the user's own
//! terminal theme on exit.
//!
//! The same escape also pins the cursor to a *steady* block (DECSCUSR `ESC [
//! 2 q`), because the point of a colored cursor is to be read at a glance and a
//! blinking one makes you wait for it. Ghostty and friends blink by default and
//! nothing in plank was overriding that, so the recolor made a pre-existing
//! blink newly obvious. [`reset`] hands the style back with `ESC [ 0 q`
//! alongside the color.
//!
//! Written to **stderr** for the same reason [`crate::title`] is: stderr
//! reaches the same tty as stdout but bypasses the Ratatui frame buffer, so a
//! color change can never tear a frame even when it lands mid-draw. No-op when
//! stderr is not a terminal, and repeats are suppressed — the busy loop redraws
//! at animation rate, and re-emitting the same escape every frame is pure noise
//! on the wire (and visible flicker on terminals that repaint the cell).

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU8, Ordering};

/// Cursor color while plank is at the prompt: the theme's military green,
/// `#87af5f` — the truecolor spelling of [`crate::status::THEME_COLOR`] (256
/// color 106). OSC 12 takes a color spec, not a palette index, so the hue has
/// to be written out longhand here.
pub const IDLE_COLOR: &str = "#87af5f";

/// Cursor color while a turn is running: `#d75f5f` (256 color 167). A brick red
/// rather than a pure `#ff0000`, so it sits in the same muted family as the
/// rest of the palette instead of glaring out of the prompt row.
pub const BUSY_COLOR: &str = "#d75f5f";

/// What the cursor color is saying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// At the prompt, waiting for the user.
    Idle,
    /// Prefilling, generating, or compacting — plank owns the turn.
    Busy,
}

impl State {
    /// The OSC 12 color spec for this state.
    #[must_use]
    pub fn color(self) -> &'static str {
        match self {
            Self::Idle => IDLE_COLOR,
            Self::Busy => BUSY_COLOR,
        }
    }

    /// Discriminant used by [`LAST`]; `0` is reserved for "nothing written
    /// yet", so the first `set` always reaches the terminal.
    fn tag(self) -> u8 {
        match self {
            Self::Idle => 1,
            Self::Busy => 2,
        }
    }
}

/// The state last written, so a redraw at the same state writes nothing.
/// `0` means the cursor color is still the terminal's own.
static LAST: AtomicU8 = AtomicU8::new(0);

/// DECSCUSR "steady block": the cursor stops blinking so the color reads at a
/// glance. Emitted with every color change rather than once at startup, so a
/// full-screen app or a shell escape that reset the style gets corrected on the
/// next phase change.
pub const STEADY_BLOCK: &str = "\x1b[2 q";

/// DECSCUSR "default": the style the user's terminal config asks for.
pub const DEFAULT_STYLE: &str = "\x1b[0 q";

/// The escapes that paint the cursor for `state`: color, then steady block.
#[must_use]
pub fn sequence(state: State) -> String {
    format!("\x1b]12;{}\x07{STEADY_BLOCK}", state.color())
}

/// The escapes that hand the cursor color *and* blink style back to the
/// terminal's own theme.
pub const RESET_SEQUENCE: &str = "\x1b]112\x07\x1b[0 q";

/// Which cursor indicator plank uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CursorMode {
    /// Recolour the terminal's own cursor with OSC 12. The default: it keeps
    /// cursor presentation where the user configured it. Inert on terminals
    /// that ignore the escape, Warp among them.
    #[default]
    Terminal,
    /// Paint the cursor into the frame with `nano-cursor`. Works everywhere,
    /// at the cost of overriding the terminal's own cursor style.
    Drawn,
    /// No phase indicator; the terminal's cursor is left entirely alone.
    Off,
}

impl CursorMode {
    /// The settings-file spelling of the mode.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CursorMode::Terminal => "terminal",
            CursorMode::Drawn => "drawn",
            CursorMode::Off => "off",
        }
    }

    /// Parses a settings value.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "terminal" | "osc" | "true" => Some(CursorMode::Terminal),
            "drawn" | "block" => Some(CursorMode::Drawn),
            "off" | "none" | "false" => Some(CursorMode::Off),
            _ => None,
        }
    }
}

/// Sets the cursor color to `state`'s. Best-effort, and a no-op both when
/// stderr is not a tty and when `state` is already what was last written.
pub fn set(state: State) {
    if crate::settings::active().ui.cursor != CursorMode::Terminal {
        return;
    }
    if LAST.swap(state.tag(), Ordering::Relaxed) == state.tag() {
        return;
    }
    write(&sequence(state));
}

/// Restores the terminal's own cursor color. Called from the TUI teardown, so
/// plank never leaves a shell prompt wearing plank's colors.
pub fn reset() {
    // Unconditional rather than guarded on `LAST`: the teardown runs once, and
    // resetting a cursor that was never recolored is harmless.
    LAST.store(0, Ordering::Relaxed);
    write(RESET_SEQUENCE);
}

/// One write of an already-formed escape, so it cannot interleave with other
/// stderr output.
fn write(seq: &str) {
    let mut err = std::io::stderr();
    if !err.is_terminal() {
        return;
    }
    let _ = err.write_all(seq.as_bytes());
    let _ = err.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_is_the_theme_green() {
        // The 256-color theme accent is 106 = #87af5f; the OSC spelling must
        // track it, since OSC 12 cannot take the palette index.
        assert_eq!(crate::status::THEME_COLOR, 106);
        assert_eq!(IDLE_COLOR, "#87af5f");
    }

    #[test]
    fn sequences_carry_the_color_and_pin_a_steady_block() {
        assert_eq!(sequence(State::Idle), "\x1b]12;#87af5f\x07\x1b[2 q");
        assert_eq!(sequence(State::Busy), "\x1b]12;#d75f5f\x07\x1b[2 q");
        assert_eq!(RESET_SEQUENCE, "\x1b]112\x07\x1b[0 q");
    }

    #[test]
    fn repeats_are_suppressed_but_changes_are_not() {
        // `LAST` is process-global, so drive it through the same door `set`
        // uses rather than asserting on terminal output.
        LAST.store(0, Ordering::Relaxed);
        let busy = State::Busy.tag();
        let idle = State::Idle.tag();
        assert_ne!(LAST.swap(busy, Ordering::Relaxed), busy);
        assert_eq!(LAST.swap(busy, Ordering::Relaxed), busy);
        assert_ne!(LAST.swap(idle, Ordering::Relaxed), idle);
        LAST.store(0, Ordering::Relaxed);
    }

    #[test]
    fn cursor_mode_round_trips_through_its_settings_spelling() {
        for m in [CursorMode::Terminal, CursorMode::Drawn, CursorMode::Off] {
            assert_eq!(CursorMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CursorMode::parse("  DRAWN "), Some(CursorMode::Drawn));
        assert_eq!(CursorMode::parse("nonsense"), None);
    }

    #[test]
    fn the_default_is_the_terminal_path() {
        // OSC 12 keeps the user's own cursor on terminals that honour it; the
        // drawn block is opt-in because it takes cursor presentation away from
        // the terminal.
        assert_eq!(CursorMode::default(), CursorMode::Terminal);
    }
}

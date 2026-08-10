// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! In-process replacement for the Ctrl-G `$EDITOR` escape hatch.
//!
//! A single-buffer fork of Microsoft Edit (`refs/edit`, MIT): plank suspends
//! its Ratatui TUI, miniedit takes the raw terminal, and the user gets a real
//! editor on the half-typed prompt. Accept returns the edited text; cancel
//! returns nothing and the caller keeps what was typed. There is no file I/O:
//! the buffer starts from a string and ends as a string.

use std::io;
use std::sync::OnceLock;

/// Scratch-arena reservation for `stdext`. Virtual address space, not
/// committed memory; `bin/edit` reserves 512 MiB on 64-bit and miniedit
/// edits prompts, so a tenth of that is ample.
const SCRATCH_ARENA_CAPACITY: usize = 48 * 1024 * 1024;

/// Initializes the scratch arenas `edit`'s buffer and TUI code allocate from.
///
/// Idempotent and cheap after the first call, so every Ctrl-G can call it.
/// Must run on the thread that owns the TUI: the arenas are process-wide
/// singletons with no synchronization.
///
/// # Errors
/// Returns the OS error when the arena reservation fails.
pub fn init() -> io::Result<()> {
    static DONE: OnceLock<bool> = OnceLock::new();
    if *DONE.get_or_init(|| stdext::arena::init(SCRATCH_ARENA_CAPACITY).is_ok()) {
        Ok(())
    } else {
        Err(io::Error::other("edit scratch arena init failed"))
    }
}

pub mod draw;
pub mod state;

pub use state::{Mode, Outcome, Search, State};

/// Applies the one normalization file mode performs on the text it is seeded
/// with: appending a trailing newline when the text lacks one.
///
/// File mode sets `insert_final_newline(true)`, but `write_raw` (the raw
/// insert path `State::build` uses for a file, to avoid `write_canon`'s
/// per-line indentation re-derivation) does not apply that setting itself —
/// so `State::build` pre-computes the seed here before handing it to
/// `write_raw`. Whether the *result* looks like an edit is no longer decided
/// by comparing against this seed anywhere: `State::is_modified` reads the
/// buffer's own post-write text back into `original` at construction, so it
/// already reflects this and every other normalization the buffer applies.
///
/// An empty file must not gain a newline: there is no line to terminate.
#[must_use]
pub(crate) fn file_seed_text(initial: &str) -> String {
    if initial.is_empty() || initial.ends_with('\n') {
        initial.to_string()
    } else {
        // Match the newline convention `State::build` sets via `set_crlf`:
        // a CRLF file gets a CRLF appended, not a bare LF that would make
        // an untouched accept look edited.
        let newline = if initial.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        format!("{initial}{newline}")
    }
}

use edit::input;
use edit::sys;
use edit::tui::Tui;
use edit::vt;
use stdext::arena::scratch_arena;

/// Opens the built-in editor on `initial` and returns the edited text, or
/// `None` when the user cancels.
///
/// The caller is responsible for leaving raw mode and the alternate screen
/// first: this function takes the terminal over exactly like a child process
/// would. In `ui.rs` that is [`with_tui_suspended`](crate::ui).
///
/// Must be called from the thread that owns the TUI — see [`init`].
///
/// # Errors
/// Returns an error when the scratch arena or the text buffer cannot be
/// allocated, or when the terminal cannot be switched to raw mode.
pub fn edit_text(initial: &str) -> io::Result<Option<String>> {
    run(initial, Mode::Prompt, None)
}

/// Opens the built-in editor on a file's contents.
///
/// `title` is the display path shown in the status bar; `initial` is the file's
/// text. Returns the edited text on accept and `None` on cancel — this function
/// does no file I/O of its own, so miniedit stays a string-in/string-out
/// editor and `crate::openfile` owns reading and writing.
///
/// The same terminal-ownership and thread rules as [`edit_text`] apply.
///
/// # Errors
/// Returns an error when the scratch arena or the text buffer cannot be
/// allocated, or when the terminal cannot be switched to raw mode.
pub fn edit_file(title: &str, initial: &str) -> io::Result<Option<String>> {
    run(initial, Mode::File, Some(title))
}

/// The editor session: takes the terminal, pumps input until the user accepts
/// or cancels, and puts the terminal back.
fn run(initial: &str, mode: Mode, title: Option<&str>) -> io::Result<Option<String>> {
    debug_log("--- session start");
    init()?;

    // Terminal ownership: `sys::init` saves and restores termios, and
    // `RestoreModes` puts the screen back. Both are drop guards, and both run
    // before plank's own restore, which is a guard one stack frame up.
    let _sys_deinit = sys::init();
    sys::switch_modes()?;
    let _restore = RestoreModes;
    setup_terminal();

    let mut vt_parser = vt::Parser::new();
    let mut input_parser = input::Parser::new();
    let mut tui = Tui::new()?;
    sys::inject_window_size_into_stdin();

    let mut state = match (mode, title) {
        (Mode::File, Some(title)) => State::new_for_file(initial, tui.size().width, title)?,
        _ => State::new(initial, tui.size().width)?,
    };

    loop {
        {
            let scratch = scratch_arena(None);
            let timeout = vt_parser.read_timeout().min(tui.read_timeout());
            // `None` means stdin closed (terminal went away). Treat it as a
            // cancel: never hand back half-edited text nobody confirmed.
            let Some(bytes) = sys::read_stdin(&scratch, timeout) else {
                debug_log("read_stdin returned None (stdin closed)");
                return Ok(None);
            };
            if !bytes.is_empty() {
                debug_log(&format!("read {} bytes: {:?}", bytes.len(), &*bytes));
            }
            let vt_iter = vt_parser.parse(&bytes);
            let mut input_iter = input_parser.parse(vt_iter);
            while {
                let input = input_iter.next();
                let has_more_input = input.is_some();
                let mut ctx = tui.create_context(input);
                draw::draw(&mut ctx, &mut state);
                has_more_input
            } {}
        }

        // Keep drawing until the layout settles; focus can move between
        // controls and needs more than one pass.
        while tui.needs_settling() {
            let mut ctx = tui.create_context(None);
            draw::draw(&mut ctx, &mut state);
        }

        match state.outcome {
            Outcome::Accept => {
                debug_log("session end: accept");
                return Ok(state.accepted_text());
            }
            Outcome::Cancel => {
                debug_log("session end: cancel");
                return Ok(None);
            }
            Outcome::Running => {}
        }

        let scratch = scratch_arena(None);
        let output = tui.render(&scratch);
        sys::write_stdout(&output);
    }
}

/// Appends a line to the file named by `$PLANK_MINIEDIT_DEBUG`, and does
/// nothing when that variable is unset.
///
/// While a session runs, miniedit owns the terminal and plank's log is not on
/// screen, so a file is the only place a diagnostic can go. This is what
/// established that keystrokes were reaching the process but not the text
/// area; leaving it in place means the next such question is one env var away.
fn debug_log(msg: &str) {
    use std::io::Write as _;
    if let Ok(path) = std::env::var("PLANK_MINIEDIT_DEBUG")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    {
        let _ = writeln!(f, "{msg}");
    }
}

/// Puts the terminal into the modes the editor needs: alternate screen, cell
/// motion + SGR mouse tracking, bracketed paste, and "meta sends escape". The
/// trailing queries ask the terminal for its color table and capabilities;
/// `edit`'s VT parser consumes the replies.
fn setup_terminal() {
    sys::write_stdout(concat!(
        "\x1b[?1049h\x1b[?1002;1006;2004h\x1b[?1036h",
        "\x1b]4;0;?;1;?;2;?;3;?;4;?;5;?;6;?;7;?\x07",
        "\x1b]4;8;?;9;?;10;?;11;?;12;?;13;?;14;?;15;?\x07",
        "\x1b]10;?\x07\x1b]11;?\x07",
        "\x1b[c",
    ));
}

/// Undoes [`setup_terminal`] on the way out, including on panic.
struct RestoreModes;

impl Drop for RestoreModes {
    fn drop(&mut self) {
        sys::write_stdout("\x1b[?1002;1006;2004l\x1b[?1036l\x1b[?1049l");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edit::buffer::TextBuffer;

    #[test]
    fn file_seed_text_appends_a_missing_trailing_newline() {
        assert_eq!(file_seed_text("hello"), "hello\n");
    }

    #[test]
    fn file_seed_text_leaves_an_already_newline_terminated_file_alone() {
        assert_eq!(file_seed_text("hello\n"), "hello\n");
    }

    #[test]
    fn file_seed_text_does_not_invent_a_newline_for_an_empty_file() {
        assert_eq!(file_seed_text(""), "");
    }

    #[test]
    fn file_seed_text_matches_the_crlf_convention() {
        assert_eq!(file_seed_text("a\r\nb"), "a\r\nb\r\n");
    }

    /// The dependency is wired up and usable headlessly: text in, text out,
    /// through the same round-trip the editor session will use.
    #[test]
    fn text_round_trips_through_a_text_buffer() {
        init().unwrap();
        let mut tb = TextBuffer::new(true).unwrap();
        tb.set_insert_final_newline(false);
        tb.write_canon(b"hello\nworld");
        let mut out = String::new();
        tb.save_as_string(&mut out);
        assert_eq!(out, "hello\nworld");
    }
}

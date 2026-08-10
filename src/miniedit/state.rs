// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Session state for the built-in editor, kept free of terminal handling so
//! it can be tested headlessly.
//!
//! Layout and text editing live in `edit`'s [`TextBuffer`]; this type only
//! tracks what plank cares about: the buffer handle, whether the user has
//! asked to leave, which search panel (if any) is up, and the last error to
//! show in the status bar.

use std::fmt;
use std::io;

use edit::buffer::{RcTextBuffer, TextBuffer};
use edit::helpers::CoordType;

/// What the session should do next, checked once per frame by the run loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Keep editing.
    Running,
    /// Leave and hand the edited text back to the prompt.
    Accept,
    /// Leave and discard the edit; the prompt keeps what the user typed.
    Cancel,
}

/// Which search panel is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// No panel.
    Hidden,
    /// Find only.
    Find,
    /// Find and replace.
    Replace,
}

/// A one-shot request to move the keyboard focus on the next frame.
///
/// The two are mutually exclusive — focus is somewhere, not in two places —
/// so this is an enum rather than a pair of flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusRequest {
    /// Leave the focus where it is.
    None,
    /// Focus the search needle: the panel just opened.
    Search,
    /// Focus the text area: the panel just closed, and the widget that held
    /// the focus is no longer drawn.
    Editor,
}

/// Which confirmation dialog is up, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialog {
    /// No dialog.
    None,
    /// "Discard your edits?" — raised when a cancel is asked for and the text
    /// no longer matches what the session started with.
    ConfirmDiscard,
}

/// What the session is editing, which decides the buffer's text conventions
/// and how the footer labels its exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A half-typed plank prompt: prose, wrapped, submitted verbatim.
    Prompt,
    /// A file on disk: no wrapping, and a trailing newline on save.
    File,
}

/// One editing session.
pub struct State {
    /// The text being edited. Shared with the textarea widget, which needs an
    /// `Rc` handle of its own.
    pub buffer: RcTextBuffer,
    /// Set by the accept/cancel affordances; the run loop exits on anything
    /// but [`Outcome::Running`].
    pub outcome: Outcome,
    /// Which search panel is up.
    pub search: Search,
    /// Last error, shown in the status bar until the next keystroke. Used for
    /// the "search unavailable" case when ICU cannot be loaded.
    pub error: Option<String>,
    /// Current search needle, kept across panel toggles.
    pub needle: String,
    /// Current replacement text.
    pub replacement: String,
    /// Pending focus move, applied and cleared by the next frame's draw.
    pub focus: FocusRequest,
    /// Which dialog is up. While one is showing it owns the keyboard: the
    /// session-level shortcuts stand down so Esc dismisses the dialog.
    pub dialog: Dialog,
    /// Case-sensitive search.
    pub match_case: bool,
    /// Whole-word search.
    pub whole_word: bool,
    /// The text the session started with, kept so [`State::is_modified`] can
    /// compare against it.
    pub original: String,
    /// Whether the line-number margin is shown. Mirrors
    /// `TextBuffer::set_margin_enabled`, which has no getter, so the View menu
    /// needs its own copy to render the checkbox.
    pub line_numbers: bool,
    /// What this session is editing. Set once at construction.
    pub mode: Mode,
    /// Display name shown in the status bar — the file's path in
    /// [`Mode::File`], `None` for a prompt (there is nothing to name).
    pub title: Option<String>,
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("State")
            .field("outcome", &self.outcome)
            .field("search", &self.search)
            .field("error", &self.error)
            .field("needle", &self.needle)
            .field("replacement", &self.replacement)
            .field("text", &self.text())
            // `buffer` is an `Rc<SemiRefCell<TextBuffer>>` with no useful
            // `Debug`; the `text` field above is what a reader wants from it.
            .finish_non_exhaustive()
    }
}

impl State {
    /// Creates a session seeded with `initial`, laid out for a `width`-column
    /// terminal.
    ///
    /// # Errors
    /// Returns the OS error when the text buffer cannot allocate.
    pub fn new(initial: &str, width: CoordType) -> io::Result<Self> {
        Self::build(initial, width, Mode::Prompt, None)
    }

    /// Creates a file-editing session for `title`'s contents.
    ///
    /// Unlike [`new`](Self::new) this wraps nothing and ends the text with a
    /// newline; `title` is shown in the status bar.
    ///
    /// # Errors
    /// Returns the OS error when the text buffer cannot allocate.
    pub fn new_for_file(initial: &str, width: CoordType, title: &str) -> io::Result<Self> {
        Self::build(initial, width, Mode::File, Some(title.to_string()))
    }

    /// Shared constructor. The two modes differ only in word wrap and the
    /// final-newline convention; everything else is identical.
    fn build(
        initial: &str,
        width: CoordType,
        mode: Mode,
        title: Option<String>,
    ) -> io::Result<Self> {
        let buffer = TextBuffer::new_rc(true)?;
        let original;
        {
            let mut tb = buffer.borrow_mut();
            // plank prompts are prose, not source: wrap rather than scroll
            // sideways, and never invent a trailing newline the user did not
            // type — the prompt is submitted verbatim. A file is the other way
            // round on both counts.
            let file = matches!(mode, Mode::File);
            tb.set_insert_final_newline(file);
            tb.set_word_wrap(!file);
            tb.set_margin_enabled(true);
            tb.set_width(width);
            // Use write_canon/write_raw instead of copy_from_str: copy_from_str
            // assumes the line count doesn't change and truncates multi-line
            // content to the first line when the buffer starts empty.
            //
            // File mode must use the raw path: write_canon is the
            // human-typing insert path, and after every newline it re-derives
            // indentation from the *previous* line and stamps it onto the
            // next one — which compounds indentation down a file loaded
            // verbatim from disk (and expands tabs, and normalizes CRLF).
            // write_raw only normalizes the newline convention, which is set
            // from the input first so it round-trips exactly.
            if file {
                tb.set_crlf(initial.contains("\r\n"));
                tb.write_raw(initial.as_bytes());
            } else {
                tb.write_canon(initial.as_bytes());
            }
            tb.cursor_move_to_offset(initial.len());
            tb.mark_as_clean();
            // Read the text back rather than keeping `initial`: in file mode
            // the buffer may have appended the final newline, and that is the
            // buffer's own normalization, not an edit the user made. Comparing
            // against the raw input would show a fresh file as modified and
            // raise "discard your edits?" on an untouched Esc.
            let mut canon = String::new();
            tb.save_as_string(&mut canon);
            original = canon;
        }
        Ok(Self {
            buffer,
            outcome: Outcome::Running,
            search: Search::Hidden,
            error: None,
            needle: String::new(),
            replacement: String::new(),
            // The menubar is the first focusable widget in the layout, so it
            // takes the default focus and every keystroke lands on a closed
            // menu instead of the text. Claim the focus for the text area on
            // the very first frame.
            focus: FocusRequest::Editor,
            dialog: Dialog::None,
            original,
            match_case: false,
            whole_word: false,
            line_numbers: true,
            mode,
            title,
        })
    }

    /// The current text.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        self.buffer.borrow_mut().save_as_string(&mut out);
        out
    }

    /// True when the text differs from the prompt the session started with.
    /// Drives the modified marker and the discard confirmation.
    ///
    /// This compares against [`State::original`] rather than asking the buffer
    /// whether it is dirty: `TextBuffer::save_as_string` calls
    /// `mark_as_clean`, so the dirty flag is cleared by the very act of
    /// reading the text out, and anything that called [`State::text`] would
    /// make an edited buffer look untouched.
    #[must_use]
    pub fn is_modified(&self) -> bool {
        self.text() != self.original
    }

    /// Asks to leave without keeping the text.
    ///
    /// Discarding an untouched buffer loses nothing, so that leaves straight
    /// away. Once the text has been edited it costs the user real work, so it
    /// goes through [`Dialog::ConfirmDiscard`] first.
    pub fn request_cancel(&mut self) {
        if self.is_modified() {
            self.dialog = Dialog::ConfirmDiscard;
        } else {
            self.outcome = Outcome::Cancel;
        }
    }

    /// Dismisses the dialog and puts the focus back in the text area.
    pub const fn dismiss_dialog(&mut self) {
        self.dialog = Dialog::None;
        self.focus = FocusRequest::Editor;
    }

    /// Hides the search panel and asks for the focus to go back to the text
    /// area. Every path that closes the panel goes through here, because
    /// leaving the focus on a widget that is no longer drawn silently eats
    /// every keystroke that follows.
    pub const fn close_search(&mut self) {
        self.search = Search::Hidden;
        self.focus = FocusRequest::Editor;
    }

    /// Opens the search panel in `kind` and focuses the needle.
    pub const fn open_search(&mut self, kind: Search) {
        self.search = kind;
        self.focus = FocusRequest::Search;
    }

    /// True once, when `want` is the pending focus request; clears it so the
    /// steal happens on exactly one frame.
    pub const fn take_focus(&mut self, want: FocusRequest) -> bool {
        if matches!(
            (self.focus, want),
            (FocusRequest::Search, FocusRequest::Search)
                | (FocusRequest::Editor, FocusRequest::Editor)
        ) {
            self.focus = FocusRequest::None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_session_holds_the_initial_text_and_is_unmodified() {
        crate::miniedit::init().unwrap();
        let mut s = State::new("write a haiku about\nkv caches", 80).unwrap();
        assert!(
            s.take_focus(FocusRequest::Editor),
            "the text area must take the focus on the first frame, or typing \
             goes to the menubar and is silently dropped"
        );
        assert_eq!(s.text(), "write a haiku about\nkv caches");
        assert!(!s.is_modified());
        assert_eq!(s.outcome, Outcome::Running);
        assert_eq!(s.search, Search::Hidden);
    }

    #[test]
    fn editing_marks_the_session_modified_and_shows_in_the_text() {
        crate::miniedit::init().unwrap();
        let s = State::new("abc", 80).unwrap();
        s.buffer.borrow_mut().write_canon(b"!");
        assert!(s.is_modified(), "write_canon should mark dirty");
        assert_eq!(s.text(), "abc!");
    }

    /// The prompt is submitted verbatim, so the editor must not add the
    /// trailing newline a file-oriented editor would.
    #[test]
    fn no_trailing_newline_is_invented() {
        crate::miniedit::init().unwrap();
        let s = State::new("one line", 80).unwrap();
        assert_eq!(s.text(), "one line");
    }

    #[test]
    fn an_empty_initial_text_is_allowed() {
        crate::miniedit::init().unwrap();
        let s = State::new("", 80).unwrap();
        assert_eq!(s.text(), "");
        assert!(!s.is_modified());
    }

    /// An untouched buffer has nothing to lose, so cancel leaves at once.
    #[test]
    fn cancelling_an_unmodified_session_needs_no_confirmation() {
        crate::miniedit::init().unwrap();
        let mut s = State::new("unchanged", 80).unwrap();
        s.request_cancel();
        assert_eq!(s.outcome, Outcome::Cancel);
        assert_eq!(s.dialog, Dialog::None);
    }

    /// Once the text differs from the prompt it came from, cancel asks first
    /// and does not leave until the user says so.
    #[test]
    fn cancelling_a_modified_session_asks_before_discarding() {
        crate::miniedit::init().unwrap();
        let mut s = State::new("original", 80).unwrap();
        s.buffer.borrow_mut().write_canon(b" plus edits");
        assert!(s.is_modified());

        s.request_cancel();
        assert_eq!(s.dialog, Dialog::ConfirmDiscard);
        assert_eq!(
            s.outcome,
            Outcome::Running,
            "the session must not end before the user confirms"
        );

        // "Keep editing" puts the user back in the text area with the edits.
        s.dismiss_dialog();
        assert_eq!(s.dialog, Dialog::None);
        assert_eq!(s.outcome, Outcome::Running);
        assert!(s.take_focus(FocusRequest::Editor));
        assert_eq!(s.text(), "original plus edits");

        // Asking again and confirming is what actually ends the session.
        s.request_cancel();
        assert_eq!(s.dialog, Dialog::ConfirmDiscard);
        s.outcome = Outcome::Cancel;
        assert_eq!(s.outcome, Outcome::Cancel);
    }

    /// Regression: `TextBuffer::save_as_string` marks the buffer clean, so a
    /// dirty-flag-based check would report "unmodified" after anything read
    /// the text — and Esc would then discard real edits without asking.
    #[test]
    fn reading_the_text_does_not_make_an_edited_session_look_untouched() {
        crate::miniedit::init().unwrap();
        let s = State::new("original", 80).unwrap();
        s.buffer.borrow_mut().write_canon(b"!");
        assert_eq!(s.text(), "original!");
        assert!(s.is_modified(), "still modified after a read");
        assert_eq!(s.text(), "original!");
        assert!(s.is_modified(), "and after a second read");
    }

    /// Regression: closing the search panel must hand the focus back to the
    /// text area. It used to stay on the (no longer drawn) needle field, and
    /// every keystroke after Esc was silently dropped.
    #[test]
    fn closing_the_search_panel_asks_for_the_editor_focus() {
        crate::miniedit::init().unwrap();
        let mut s = State::new("alpha", 80).unwrap();

        s.open_search(Search::Find);
        assert_eq!(s.search, Search::Find);
        assert!(s.take_focus(FocusRequest::Search), "the needle takes focus");
        assert!(
            !s.take_focus(FocusRequest::Search),
            "the request is one-shot"
        );

        s.close_search();
        assert_eq!(s.search, Search::Hidden);
        assert!(
            !s.take_focus(FocusRequest::Search),
            "a closed panel must not claim the focus"
        );
        assert!(s.take_focus(FocusRequest::Editor), "the text area gets it");
        assert!(
            !s.take_focus(FocusRequest::Editor),
            "the request is one-shot"
        );
    }

    /// Search state survives closing and reopening the panel, so Ctrl-F after
    /// a find does not clear what the user typed.
    #[test]
    fn the_needle_survives_hiding_the_search_panel() {
        crate::miniedit::init().unwrap();
        let mut s = State::new("alpha beta", 80).unwrap();
        s.search = Search::Find;
        s.needle = "beta".to_string();
        s.search = Search::Hidden;
        s.search = Search::Find;
        assert_eq!(s.needle, "beta");
    }

    /// Prompt mode is unchanged: no trailing newline is invented, because the
    /// prompt is submitted verbatim.
    #[test]
    fn prompt_mode_does_not_add_a_final_newline() {
        crate::miniedit::init().unwrap();
        let state = State::new("hello", 80).unwrap();
        assert_eq!(state.mode, Mode::Prompt);
        assert_eq!(state.title, None);
        assert_eq!(state.text(), "hello");
        assert!(!state.is_modified());
    }

    /// File mode ends the file with a newline, the POSIX convention every tool
    /// downstream of the editor expects.
    #[test]
    fn file_mode_adds_a_final_newline() {
        crate::miniedit::init().unwrap();
        let state = State::new_for_file("hello", 80, "a.txt").unwrap();
        assert_eq!(state.mode, Mode::File);
        assert_eq!(state.title.as_deref(), Some("a.txt"));
        assert_eq!(state.text(), "hello\n");
    }

    /// The newline file mode adds must not read as a user edit: otherwise
    /// opening a file with no trailing newline and pressing Esc would raise
    /// "discard your edits?" over an edit the user never made.
    #[test]
    fn the_added_final_newline_is_not_a_modification() {
        crate::miniedit::init().unwrap();
        let state = State::new_for_file("hello", 80, "a.txt").unwrap();
        assert!(
            !state.is_modified(),
            "a freshly opened file must start clean"
        );
    }

    /// A file that already ends in a newline round-trips byte-for-byte.
    #[test]
    fn file_mode_round_trips_an_already_newline_terminated_file() {
        crate::miniedit::init().unwrap();
        let state = State::new_for_file("a\nb\n", 80, "a.txt").unwrap();
        assert_eq!(state.text(), "a\nb\n");
        assert!(!state.is_modified());
    }

    /// Regression: `write_canon` re-derives each line's indentation from the
    /// *previous* line and compounds it going down the file — fine for a
    /// human typing in the prompt editor, corruption for a file loaded
    /// verbatim from disk. File mode must use the raw insert path so
    /// indentation is preserved exactly as written.
    #[test]
    fn file_mode_preserves_brace_and_space_indentation() {
        crate::miniedit::init().unwrap();
        let src = "fn a() {\n    let x = 1;\n    let y = 2;\n}\n";
        let state = State::new_for_file(src, 80, "a.rs").unwrap();
        assert_eq!(state.text(), src);
        assert!(!state.is_modified());
    }

    /// Regression: a tab-indented Makefile line must not be expanded to
    /// spaces or have indentation compounded onto it.
    #[test]
    fn file_mode_preserves_tab_indentation() {
        crate::miniedit::init().unwrap();
        let src = "all:\n\tgcc x.c\n";
        let state = State::new_for_file(src, 80, "Makefile").unwrap();
        assert_eq!(state.text(), src);
        assert!(!state.is_modified());
    }

    /// Regression: CRLF-terminated files must round-trip with their line
    /// endings intact, not silently normalized to LF.
    #[test]
    fn file_mode_preserves_crlf_line_endings() {
        crate::miniedit::init().unwrap();
        let src = "a\r\nb\r\n";
        let state = State::new_for_file(src, 80, "a.txt").unwrap();
        assert_eq!(state.text(), src);
        assert!(!state.is_modified());
    }
}

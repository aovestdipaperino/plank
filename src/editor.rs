// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Terminal line editor with a status footer, ported from the ds4 agent.
//!
//! This is a faithful-but-tractable port of the linenoise-derived editor in
//! `refs/ds4/ds4_agent.c` ("Terminal Prompt, Status Footer, And Async Output
//! Rendering"). The pure pieces (line buffer, history ring, completion
//! cycling, paste-marker stripping) are plain data structures testable
//! without a TTY; only [`Editor`] touches the terminal.
//!
//! Deliberate simplifications versus the C reference:
//!
//! - The scroll-region optimization used by `editor_write_async` is not
//!   ported; [`Editor::write_above`] hides the prompt and footer, writes the
//!   text, and repaints instead.
//! - CPR (cursor position report) probing is not ported; the editor always
//!   repaints from column zero on its own lines.
//! - Rendering is single-visual-line with horizontal scrolling (embedded
//!   newlines from a bracketed paste are displayed as `␤`).

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Write as _};
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};

/// Maximum number of history entries kept in memory, matching the C agent.
pub const HISTORY_MAX: usize = 512;

/// Fallback terminal width when `TIOCGWINSZ` is unavailable.
const DEFAULT_COLS: usize = 80;

/// Result of one [`Editor::read_line`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    /// The user submitted a line (may contain newlines from a paste).
    Line(String),
    /// The user pressed Ctrl-C.
    Interrupted,
    /// The user pressed Ctrl-D on an empty line (end of input).
    Eof,
}

// ---------------------------------------------------------------------------
// Line buffer (pure, testable)
// ---------------------------------------------------------------------------

/// Returns the byte offset of the start of the word before `cursor`.
///
/// Readline semantics: skip any whitespace immediately left of the cursor,
/// then skip the run of non-whitespace characters. Returns `cursor` itself
/// only when already at the start of the line.
#[must_use]
pub fn prev_word_boundary(text: &str, cursor: usize) -> usize {
    let head = &text[..cursor];
    let trimmed = head.trim_end_matches(char::is_whitespace);
    match trimmed.rfind(char::is_whitespace) {
        // `rfind` gives the byte offset of a whitespace char; the word starts
        // at the next char boundary after it.
        Some(i) => i + trimmed[i..].chars().next().map_or(1, char::len_utf8),
        None => 0,
    }
}

/// Returns the byte offset just past the end of the word after `cursor`.
///
/// Mirror of [`prev_word_boundary`]: skip whitespace, then the word.
#[must_use]
pub fn next_word_boundary(text: &str, cursor: usize) -> usize {
    let tail = &text[cursor..];
    let skipped = tail.len() - tail.trim_start_matches(char::is_whitespace).len();
    let rest = &tail[skipped..];
    let word = rest.find(char::is_whitespace).unwrap_or(rest.len());
    cursor + skipped + word
}

/// Whether a CSI parameter string like `"1;3"` carries an Alt or Ctrl
/// modifier, the ones that turn an arrow key into a word-wise motion.
///
/// xterm encodes the modifier as `1 + bitmask` (shift 1, alt 2, ctrl 4), so
/// Alt is 3, Ctrl 5, Ctrl+Alt 7, and 9/13 appear when Meta is also set.
#[must_use]
fn is_word_modifier(params: &str) -> bool {
    matches!(
        params.rsplit(';').next(),
        Some("3" | "5" | "7" | "9" | "13")
    )
}

/// An editable UTF-8 line with a cursor, mirroring linenoise's edit ops.
///
/// It also carries an optional *selection anchor* (issue: Shift+arrow
/// selection). The anchor is the fixed end of a selection whose moving end is
/// the cursor, so every existing motion doubles as a selection-extending motion
/// once [`LineBuffer::anchor_here`] has pinned it. Nothing sets the anchor on
/// its own: a buffer whose caller never asks for selection behaves exactly as
/// it did before.
#[derive(Debug, Default, Clone)]
pub struct LineBuffer {
    text: String,
    /// Byte offset of the cursor; always on a char boundary.
    cursor: usize,
    /// Fixed end of an active selection, as a byte offset; `None` when nothing
    /// is selected. May sit either side of the cursor — [`LineBuffer::selection`]
    /// normalizes.
    anchor: Option<usize>,
}

impl LineBuffer {
    /// Creates an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the cursor position as a byte offset.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Pins the selection anchor at the cursor, unless one is already set.
    ///
    /// Called before a motion while Shift is held: the first Shift+motion drops
    /// the anchor where the cursor was, and every later one leaves it alone so
    /// the selection grows from the same origin.
    pub fn anchor_here(&mut self) {
        self.anchor.get_or_insert(self.cursor);
    }

    /// Drops any selection, leaving the text and cursor alone.
    pub fn clear_selection(&mut self) {
        self.anchor = None;
    }

    /// The selected byte range, ordered, or `None` when nothing is selected.
    ///
    /// An anchor that has collapsed onto the cursor counts as no selection, so
    /// a Shift+Left followed by Shift+Right does not leave an invisible
    /// zero-width selection behind to swallow the next keystroke.
    #[must_use]
    pub fn selection(&self) -> Option<(usize, usize)> {
        let a = self.anchor?;
        let (lo, hi) = if a <= self.cursor {
            (a, self.cursor)
        } else {
            (self.cursor, a)
        };
        (lo < hi).then_some((lo, hi))
    }

    /// The selected text, or `None` when nothing is selected.
    #[must_use]
    pub fn selected_text(&self) -> Option<&str> {
        self.selection().map(|(a, b)| &self.text[a..b])
    }

    /// Selects the whole buffer, leaving the cursor at the end.
    pub fn select_all(&mut self) {
        self.anchor = Some(0);
        self.cursor = self.text.len();
    }

    /// Deletes the selection, leaving the cursor where it started. Returns
    /// whether anything was removed.
    pub fn delete_selection(&mut self) -> bool {
        let Some((start, end)) = self.selection() else {
            self.anchor = None;
            return false;
        };
        self.text.replace_range(start..end, "");
        self.cursor = start;
        self.anchor = None;
        true
    }

    /// Moves the cursor to `byte`, snapped down to a char boundary and clamped
    /// to the text. Used by mouse clicks, which land on a screen cell rather
    /// than a known offset.
    pub fn set_cursor(&mut self, byte: usize) {
        let mut b = byte.min(self.text.len());
        while !self.text.is_char_boundary(b) {
            b -= 1;
        }
        self.cursor = b;
    }

    /// Replaces the whole line and puts the cursor at the end.
    pub fn set_text(&mut self, text: impl AsRef<str>) {
        text.as_ref().clone_into(&mut self.text);
        self.cursor = self.text.len();
        self.anchor = None;
    }

    /// Clears the line.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.anchor = None;
    }

    /// Inserts a string at the cursor and advances past it. An active selection
    /// is replaced, the way typing over a selection works everywhere else.
    pub fn insert(&mut self, s: impl AsRef<str>) {
        self.delete_selection();
        let s = s.as_ref();
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Moves the cursor one character left. Returns whether it moved.
    pub fn move_left(&mut self) -> bool {
        match self.prev_boundary() {
            Some(b) => {
                self.cursor = b;
                true
            }
            None => false,
        }
    }

    /// Moves the cursor one character right. Returns whether it moved.
    pub fn move_right(&mut self) -> bool {
        match self.next_boundary() {
            Some(b) => {
                self.cursor = b;
                true
            }
            None => false,
        }
    }

    /// Moves the cursor to the start of the line (Ctrl-A / Home).
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Moves the cursor to the end of the line (Ctrl-E / End).
    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    /// Deletes the character before the cursor (Backspace), or the whole
    /// selection when one is active.
    pub fn backspace(&mut self) -> bool {
        if self.delete_selection() {
            return true;
        }
        match self.prev_boundary() {
            Some(b) => {
                self.text.replace_range(b..self.cursor, "");
                self.cursor = b;
                true
            }
            None => false,
        }
    }

    /// Deletes the character under the cursor (Delete / Ctrl-D on non-empty),
    /// or the whole selection when one is active.
    pub fn delete(&mut self) -> bool {
        if self.delete_selection() {
            return true;
        }
        match self.next_boundary() {
            Some(b) => {
                self.text.replace_range(self.cursor..b, "");
                true
            }
            None => false,
        }
    }

    /// Deletes from the cursor to the end of the line (Ctrl-K).
    pub fn kill_to_end(&mut self) {
        self.anchor = None;
        self.text.truncate(self.cursor);
    }

    /// Deletes from the start of the line to the cursor (Ctrl-U).
    pub fn kill_to_start(&mut self) {
        self.anchor = None;
        self.text.replace_range(..self.cursor, "");
        self.cursor = 0;
    }

    /// Deletes the word before the cursor (Ctrl-W / Alt-Backspace).
    pub fn delete_prev_word(&mut self) {
        self.anchor = None;
        let start = prev_word_boundary(&self.text, self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Deletes the word after the cursor (Alt-D / Alt-Delete).
    pub fn delete_next_word(&mut self) {
        self.anchor = None;
        let end = next_word_boundary(&self.text, self.cursor);
        self.text.replace_range(self.cursor..end, "");
    }

    /// Moves the cursor to the start of the previous word (Alt-Left).
    /// Returns whether it moved.
    pub fn move_prev_word(&mut self) -> bool {
        let b = prev_word_boundary(&self.text, self.cursor);
        let moved = b != self.cursor;
        self.cursor = b;
        moved
    }

    /// Moves the cursor past the end of the next word (Alt-Right).
    /// Returns whether it moved.
    pub fn move_next_word(&mut self) -> bool {
        let b = next_word_boundary(&self.text, self.cursor);
        let moved = b != self.cursor;
        self.cursor = b;
        moved
    }

    /// Returns the byte range of the word ending at the cursor.
    ///
    /// Used by tab completion: the "word" is the run of non-space bytes
    /// immediately before the cursor.
    #[must_use]
    pub fn word_before_cursor(&self) -> (usize, usize) {
        let bytes = self.text.as_bytes();
        let mut start = self.cursor;
        while start > 0 && bytes[start - 1] != b' ' {
            start -= 1;
        }
        (start, self.cursor)
    }

    /// Replaces the byte range `start..end` with `s`, cursor after `s`.
    pub fn replace_range(&mut self, start: usize, end: usize, s: impl AsRef<str>) {
        self.anchor = None;
        let s = s.as_ref();
        self.text.replace_range(start..end, s);
        self.cursor = start + s.len();
    }

    /// Byte range of the *logical* line (newline-delimited) holding the cursor,
    /// plus the cursor's char offset within it.
    fn logical_line(&self) -> (usize, usize, usize) {
        let start = self.text[..self.cursor].rfind('\n').map_or(0, |i| i + 1);
        let end = self.text[self.cursor..]
            .find('\n')
            .map_or(self.text.len(), |i| self.cursor + i);
        (start, end, self.text[start..self.cursor].chars().count())
    }

    /// Byte offset `col` chars into the line spanning `start..end`, clamped to
    /// its end when the line is shorter than `col`.
    fn offset_in_line(&self, start: usize, end: usize, col: usize) -> usize {
        self.text[start..end]
            .char_indices()
            .nth(col)
            .map_or(end, |(i, _)| start + i)
    }

    /// Moves the cursor to the same column on the previous logical line
    /// (Shift+Up / Up in a multi-line prompt). A cursor already on the first
    /// line goes to the start of the buffer, which is what a single-line prompt
    /// wants too. Returns whether it moved.
    ///
    /// Logical, not visual: a long wrapped line counts as one line here, so this
    /// stays independent of the terminal width.
    ///
    /// The column is not sticky — crossing a short line and continuing keeps the
    /// column the short line clamped to, rather than restoring the original. A
    /// goal column would need state that every other motion has to remember to
    /// reset, which is not worth it for a prompt that is usually one line.
    pub fn move_line_up(&mut self) -> bool {
        let (start, _, col) = self.logical_line();
        let before = self.cursor;
        if start == 0 {
            self.cursor = 0;
        } else {
            let prev_start = self.text[..start - 1].rfind('\n').map_or(0, |i| i + 1);
            self.cursor = self.offset_in_line(prev_start, start - 1, col);
        }
        self.cursor != before
    }

    /// Mirror of [`LineBuffer::move_line_up`]: the same column on the next
    /// logical line, or the end of the buffer when already on the last.
    pub fn move_line_down(&mut self) -> bool {
        let (_, end, col) = self.logical_line();
        let before = self.cursor;
        if end == self.text.len() {
            self.cursor = self.text.len();
        } else {
            let next_start = end + 1;
            let next_end = self.text[next_start..]
                .find('\n')
                .map_or(self.text.len(), |i| next_start + i);
            self.cursor = self.offset_in_line(next_start, next_end, col);
        }
        self.cursor != before
    }

    fn prev_boundary(&self) -> Option<usize> {
        if self.cursor == 0 {
            return None;
        }
        let mut b = self.cursor - 1;
        while !self.text.is_char_boundary(b) {
            b -= 1;
        }
        Some(b)
    }

    fn next_boundary(&self) -> Option<usize> {
        if self.cursor >= self.text.len() {
            return None;
        }
        let mut b = self.cursor + 1;
        while !self.text.is_char_boundary(b) {
            b += 1;
        }
        Some(b)
    }
}

// ---------------------------------------------------------------------------
// History ring (pure, testable)
// ---------------------------------------------------------------------------

/// Field separator used to tag a saved entry with its origin directory.
///
/// `\x1f` (ASCII Unit Separator) is effectively never typed at a prompt, so a
/// legacy history file (plain lines, no separator) is unambiguously
/// distinguishable from a directory-tagged one: any line without a leading
/// separator loads as an untagged (global) entry.
const DIR_SEP: char = '\x1f';

/// One history entry: the text plus the directory it was entered in.
///
/// `dir` is `None` for entries with no directory scope — legacy entries loaded
/// from a pre-tagging history file, or entries added when the working
/// directory could not be resolved. Untagged entries are always eligible for
/// navigation (a global fallback), so upgrading never hides old history.
#[derive(Debug, Clone)]
struct HistEntry {
    text: String,
    dir: Option<String>,
}

/// Canonical tag for the current working directory, or `None` if unresolved.
#[must_use]
fn current_dir_tag() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

/// A bounded command-history ring with consecutive-duplicate suppression.
///
/// Each entry records the directory it was entered in; navigation filters to
/// the current directory (plus untagged/global entries) so a command typed in
/// one project does not surface in another (issue #49).
#[derive(Debug, Clone)]
pub struct History {
    entries: VecDeque<HistEntry>,
    /// `Some` pins the cap to an explicit value (what every existing
    /// constructor does, tests included). `None` means "consult
    /// `ui.historySize` live" — see [`History::live`] and [`effective_max`].
    max_override: Option<usize>,
    /// Directory new entries are tagged with, and the one navigation filters
    /// to. Defaults to the process working directory; overridable for tests.
    cwd: Option<String>,
}

impl Default for History {
    fn default() -> Self {
        Self::new(HISTORY_MAX)
    }
}

impl History {
    /// Creates an empty history bounded to `max` entries, tagging new entries
    /// with the process working directory.
    ///
    /// The cap is pinned at this value for the life of the `History`. The app
    /// itself does not use this constructor for that reason — see
    /// [`History::live`].
    #[must_use]
    pub fn new(max: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_override: Some(max.max(1)),
            cwd: current_dir_tag(),
        }
    }

    /// Creates an empty history whose cap tracks `ui.historySize` live.
    ///
    /// A resize of an already-full `VecDeque` mid-session is more invasive
    /// than it is worth (shrinking would mean deciding which live entries to
    /// discard, right as the user might be scrolling them); consulting the
    /// setting on every trim, the same way `complete::refresh_throttle`
    /// consults `ui.indexRefreshSecs`, gets the live behaviour a user expects
    /// — a smaller cap starts winning back entries immediately — without that
    /// complexity.
    #[must_use]
    pub fn live() -> Self {
        Self {
            entries: VecDeque::new(),
            max_override: None,
            cwd: current_dir_tag(),
        }
    }

    /// The cap trimming enforces right now: the pinned value, or the live
    /// `ui.historySize` setting.
    fn effective_max(&self) -> usize {
        self.max_override
            .unwrap_or_else(|| crate::settings::active().ui.history_size)
            .max(1)
    }

    /// Overrides the directory used for tagging and navigation filtering.
    ///
    /// Primarily for tests; the app relies on the process working directory.
    pub fn set_cwd(&mut self, dir: Option<String>) {
        self.cwd = dir;
    }

    /// Number of stored entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the entry text at `idx` (0 = oldest), if present.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.entries.get(idx).map(|e| e.text.as_str())
    }

    /// Returns the origin directory of the entry at `idx`, if tagged.
    #[must_use]
    pub fn dir_of(&self, idx: usize) -> Option<&str> {
        self.entries.get(idx).and_then(|e| e.dir.as_deref())
    }

    /// Whether the entry at `idx` belongs to the current directory scope.
    ///
    /// True for entries tagged with the current directory and for untagged
    /// (global/legacy) entries; false for entries from a different directory.
    #[must_use]
    pub fn is_eligible(&self, idx: usize) -> bool {
        match self.entries.get(idx) {
            None => false,
            Some(e) => match &e.dir {
                None => true,
                Some(d) => Some(d.as_str()) == self.cwd.as_deref(),
            },
        }
    }

    /// Appends an entry (tagged with the current directory), skipping empties
    /// and consecutive duplicates.
    pub fn add(&mut self, entry: impl AsRef<str>) {
        let dir = self.cwd.clone();
        self.add_in_dir(entry, dir);
    }

    /// Appends an entry tagged with an explicit directory (`None` = global).
    pub fn add_in_dir(&mut self, entry: impl AsRef<str>, dir: Option<String>) {
        let entry = entry.as_ref();
        if entry.is_empty() || self.entries.back().is_some_and(|last| last.text == entry) {
            return;
        }
        // `>=` rather than `==`: a live cap can drop between two calls (the
        // user just lowered `ui.historySize`), and this must catch up on the
        // very next add rather than only once length happens to re-cross it.
        while self.entries.len() >= self.effective_max() {
            self.entries.pop_front();
        }
        self.entries.push_back(HistEntry {
            text: entry.to_owned(),
            dir,
        });
    }

    /// Loads history from `path`.
    ///
    /// Each line is either a plain (legacy, untagged) entry or a
    /// directory-tagged entry of the form `\x1f<dir>\x1f<text>`. A missing file
    /// is not an error (fresh start).
    ///
    /// # Errors
    ///
    /// Returns any I/O error other than `NotFound`.
    pub fn load(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let data = match fs::read_to_string(path.as_ref()) {
            Ok(d) => d,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };
        for line in data.lines() {
            let (text, dir) = parse_line(line);
            self.add_in_dir(text, dir);
        }
        Ok(())
    }

    /// Saves the history to `path`, one entry per line, tagging each entry with
    /// its origin directory (untagged entries are written as plain lines, so a
    /// downgrade still reads them).
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing the file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut out = String::new();
        for e in &self.entries {
            if let Some(dir) = &e.dir {
                out.push(DIR_SEP);
                out.push_str(dir);
                out.push(DIR_SEP);
            }
            out.push_str(&e.text);
            out.push('\n');
        }
        fs::write(path.as_ref(), out)
    }
}

/// Splits a saved history line into `(text, dir)`.
///
/// A line starting with [`DIR_SEP`] is `\x1f<dir>\x1f<text>`; anything else is
/// a legacy untagged entry.
fn parse_line(line: &str) -> (&str, Option<String>) {
    if let Some(rest) = line.strip_prefix(DIR_SEP)
        && let Some((dir, text)) = rest.split_once(DIR_SEP)
    {
        return (text, Some(dir.to_owned()));
    }
    (line, None)
}

/// Returns the default history path: `$HOME/.plank_history`.
#[must_use]
pub fn default_history_path() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".plank_history")
}

// ---------------------------------------------------------------------------
// Completion cycling (pure, testable)
// ---------------------------------------------------------------------------

/// Tracks Tab-completion candidates and cycles through them like linenoise.
#[derive(Debug, Default)]
struct CompletionState {
    candidates: Vec<String>,
    /// Index of the candidate currently shown; `candidates.len()` shows the
    /// original word (linenoise wraps through the original).
    index: usize,
    /// Word being completed, so cycling can restore it.
    original: String,
    active: bool,
}

impl CompletionState {
    /// Starts or advances a completion cycle. Returns the text to display in
    /// place of the completed word, or `None` when there are no candidates.
    fn advance(&mut self, word: &str, candidates: Vec<String>) -> Option<&str> {
        if self.active {
            self.index = (self.index + 1) % (self.candidates.len() + 1);
        } else {
            if candidates.is_empty() {
                return None;
            }
            self.candidates = candidates;
            word.clone_into(&mut self.original);
            self.index = 0;
            self.active = true;
        }
        if self.index == self.candidates.len() {
            Some(&self.original)
        } else {
            Some(&self.candidates[self.index])
        }
    }

    /// Whether only one candidate exists (replace, don't cycle).
    fn is_single(&self) -> bool {
        self.candidates.len() == 1
    }

    fn reset(&mut self) {
        self.active = false;
        self.candidates.clear();
        self.original.clear();
        self.index = 0;
    }
}

// ---------------------------------------------------------------------------
// Bracketed paste (pure helper, testable)
// ---------------------------------------------------------------------------

/// Strips bracketed-paste start/end markers from `data`, keeping newlines.
///
/// `\r` is normalized to `\n` (terminals send CR for Enter inside a paste).
#[must_use]
pub fn strip_paste_markers(data: &str) -> String {
    let mut s = data.replace("\x1b[200~", "").replace("\x1b[201~", "");
    s = s.replace("\r\n", "\n").replace('\r', "\n");
    s
}

// ---------------------------------------------------------------------------
// Raw mode guard
// ---------------------------------------------------------------------------

/// Restores the saved termios state when dropped.
#[derive(Debug)]
struct RawModeGuard {
    fd: RawFd,
    saved: libc::termios,
    active: bool,
}

impl RawModeGuard {
    /// Puts `fd` into linenoise-style raw mode.
    fn enable(fd: RawFd) -> io::Result<Self> {
        // SAFETY: `termios` is a plain-old-data struct; zeroed is a valid
        // initial value that tcgetattr fully overwrites on success.
        let mut orig: libc::termios = unsafe { std::mem::zeroed() };
        // SAFETY: fd is a valid open descriptor owned by the process and
        // `orig` is a properly aligned, writable termios.
        if unsafe { libc::tcgetattr(fd, &raw mut orig) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut raw = orig;
        raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
        raw.c_oflag &= !libc::OPOST;
        raw.c_cflag |= libc::CS8;
        raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        // SAFETY: fd is valid and `raw` is a fully initialized termios copied
        // from the current settings.
        if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &raw const raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            fd,
            saved: orig,
            active: true,
        })
    }

    fn restore(&mut self) {
        if self.active {
            // SAFETY: fd is valid and `saved` holds the termios captured by
            // tcgetattr in `enable`.
            unsafe { libc::tcsetattr(self.fd, libc::TCSAFLUSH, &raw const self.saved) };
            self.active = false;
        }
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/// Completion callback: given the word before the cursor, return candidates.
pub type CompletionFn = Box<dyn Fn(&str) -> Vec<String>>;

/// Interactive line editor with history, completion, and a status footer.
///
/// Not `Send`: it owns terminal state and a non-`Send` completion closure by
/// design (it must live on the thread driving the TTY).
pub struct Editor {
    buf: LineBuffer,
    /// Command history (public field-style access via methods).
    history: History,
    history_index: Option<usize>,
    /// Line stashed when navigating away from the in-progress entry.
    stash: String,
    completion: Option<CompletionFn>,
    completion_state: CompletionState,
    raw: Option<RawModeGuard>,
    prompt: String,
    footer: String,
    /// Whether the prompt/footer pair is currently drawn on screen.
    painted: bool,
    in_fd: RawFd,
}

impl std::fmt::Debug for Editor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Editor")
            .field("buf", &self.buf)
            .field("history_len", &self.history.len())
            .field("raw_mode", &self.raw.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    /// Creates an editor reading from stdin and writing to stdout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: LineBuffer::new(),
            history: History::default(),
            history_index: None,
            stash: String::new(),
            completion: None,
            completion_state: CompletionState::default(),
            raw: None,
            prompt: String::new(),
            footer: String::new(),
            painted: false,
            in_fd: libc::STDIN_FILENO,
        }
    }

    /// Mutable access to the history (for load/save/add).
    pub fn history_mut(&mut self) -> &mut History {
        &mut self.history
    }

    /// Shared access to the history.
    #[must_use]
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Installs the Tab-completion callback.
    pub fn set_completion(&mut self, f: CompletionFn) {
        self.completion = Some(f);
    }

    /// Re-enables raw mode (e.g. after a shelled-out job reset the TTY).
    ///
    /// # Errors
    ///
    /// Returns the OS error when termios calls fail.
    pub fn ensure_raw_mode(&mut self) -> io::Result<()> {
        if self.raw.is_none() {
            self.raw = Some(RawModeGuard::enable(self.in_fd)?);
        }
        Ok(())
    }

    /// Restores the terminal to its original (cooked) mode.
    pub fn restore_terminal(&mut self) {
        if let Some(mut g) = self.raw.take() {
            g.restore();
        }
    }

    /// Updates the footer text and repaints if the editor is active.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing to stdout.
    pub fn set_footer(&mut self, footer: impl AsRef<str>) -> io::Result<()> {
        footer.as_ref().clone_into(&mut self.footer);
        if self.painted {
            self.redraw()?;
        }
        Ok(())
    }

    /// Repaints the prompt line and footer.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing to stdout.
    pub fn redraw(&mut self) -> io::Result<()> {
        let frame = render_frame(
            &self.prompt,
            self.buf.text(),
            self.buf.cursor(),
            &self.footer,
            terminal_cols(),
        );
        let mut out = io::stdout().lock();
        out.write_all(frame.as_bytes())?;
        out.flush()?;
        self.painted = true;
        Ok(())
    }

    /// Hides the prompt and footer, writes `text` above, then repaints.
    ///
    /// This is the essence of the C `editor_write_async`; the scroll-region
    /// optimization is intentionally not ported (see module docs).
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing to stdout.
    pub fn write_above(&mut self, text: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.painted {
            // Clear footer line then prompt line, leaving the cursor at the
            // start of the prompt line.
            out.write_all(b"\r\x1b[K\x1b[B\r\x1b[K\x1b[A")?;
        }
        // In raw mode OPOST is off, so LF does not imply CR; normalize.
        let mut normalized = text.replace('\n', "\r\n");
        if !normalized.ends_with("\r\n") {
            normalized.push_str("\r\n");
        }
        out.write_all(normalized.as_bytes())?;
        out.flush()?;
        drop(out);
        if self.painted {
            self.redraw()?;
        }
        Ok(())
    }

    /// Reads one line with full editing, history, and completion support.
    ///
    /// Bracketed paste is enabled for the duration; a multi-line paste is
    /// returned as a single submission with its newlines preserved.
    ///
    /// # Errors
    ///
    /// Returns errors from terminal setup or stdin/stdout I/O.
    pub fn read_line(&mut self, prompt: &str, footer: &str) -> io::Result<ReadOutcome> {
        prompt.clone_into(&mut self.prompt);
        footer.clone_into(&mut self.footer);
        self.buf.clear();
        self.history_index = None;
        self.stash.clear();
        self.completion_state.reset();

        self.ensure_raw_mode()?;
        write_stdout(b"\x1b[?2004h")?; // enable bracketed paste
        let outcome = self.edit_loop();
        // Best-effort cleanup even when the loop errored.
        let _ = write_stdout(b"\x1b[?2004l");
        self.painted = false;
        self.restore_terminal();
        let outcome = outcome?;
        write_stdout(b"\r\n")?;
        if let ReadOutcome::Line(line) = &outcome {
            self.history.add(line);
        }
        Ok(outcome)
    }

    fn edit_loop(&mut self) -> io::Result<ReadOutcome> {
        self.redraw()?;
        loop {
            let b = read_byte(self.in_fd)?;
            let Some(b) = b else {
                return Ok(ReadOutcome::Eof);
            };
            if b != b'\t' {
                self.completion_state.reset();
            }
            match b {
                b'\r' | b'\n' => return Ok(ReadOutcome::Line(self.buf.text().to_owned())),
                0x03 => return Ok(ReadOutcome::Interrupted), // Ctrl-C
                0x04 => {
                    // Ctrl-D: EOF on empty line, else delete-forward.
                    if self.buf.text().is_empty() {
                        return Ok(ReadOutcome::Eof);
                    }
                    self.buf.delete();
                }
                0x01 => self.buf.move_home(), // Ctrl-A
                0x05 => self.buf.move_end(),  // Ctrl-E
                0x02 => {
                    self.buf.move_left(); // Ctrl-B
                }
                0x06 => {
                    self.buf.move_right(); // Ctrl-F
                }
                0x08 | 0x7f => {
                    self.buf.backspace();
                }
                0x0b => self.buf.kill_to_end(),      // Ctrl-K
                0x15 => self.buf.kill_to_start(),    // Ctrl-U
                0x17 => self.buf.delete_prev_word(), // Ctrl-W
                0x0c => {
                    // Ctrl-L: clear screen, repaint at top.
                    write_stdout(b"\x1b[H\x1b[2J")?;
                }
                0x10 => self.history_move(-1), // Ctrl-P
                0x0e => self.history_move(1),  // Ctrl-N
                b'\t' => self.handle_tab(),
                0x1b => self.handle_escape()?,
                b if b >= 0x20 => self.insert_input_byte(b)?,
                _ => {}
            }
            self.redraw()?;
        }
    }

    /// Inserts a printable byte, gathering UTF-8 continuation bytes.
    fn insert_input_byte(&mut self, first: u8) -> io::Result<()> {
        let need = match first {
            0x00..=0x7f => 0,
            0xc0..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf7 => 3,
            _ => return Ok(()), // stray continuation byte; drop it
        };
        let mut bytes = vec![first];
        for _ in 0..need {
            match read_byte(self.in_fd)? {
                Some(b) => bytes.push(b),
                None => return Ok(()),
            }
        }
        if let Ok(s) = std::str::from_utf8(&bytes) {
            self.buf.insert(s);
        }
        Ok(())
    }

    fn handle_tab(&mut self) {
        let Some(cb) = self.completion.as_ref() else {
            return;
        };
        let (start, end) = self.buf.word_before_cursor();
        let word = self.buf.text()[start..end].to_owned();
        let candidates = if self.completion_state.active {
            Vec::new() // ignored; cycling continues on stored candidates
        } else {
            cb(&word)
        };
        // Cycling replaces the *original* word region, which currently spans
        // start..cursor (the shown candidate).
        let shown_end = self.buf.cursor();
        let cycle_word = if self.completion_state.active {
            self.completion_state.original.clone()
        } else {
            word
        };
        let Some(replacement) = self
            .completion_state
            .advance(&cycle_word, candidates)
            .map(str::to_owned)
        else {
            return;
        };
        self.buf.replace_range(start, shown_end, &replacement);
        if self.completion_state.is_single() {
            self.completion_state.reset();
        }
    }

    fn handle_escape(&mut self) -> io::Result<()> {
        let Some(b1) = read_byte(self.in_fd)? else {
            return Ok(());
        };
        if b1 == b'[' {
            let Some(b2) = read_byte(self.in_fd)? else {
                return Ok(());
            };
            match b2 {
                b'A' => self.history_move(-1),
                b'B' => self.history_move(1),
                b'C' => {
                    self.buf.move_right();
                }
                b'D' => {
                    self.buf.move_left();
                }
                b'H' => self.buf.move_home(),
                b'F' => self.buf.move_end(),
                b'0'..=b'9' => {
                    // Extended sequence: ESC [ digits ~
                    let mut num = String::from(b2 as char);
                    loop {
                        let Some(b) = read_byte(self.in_fd)? else {
                            return Ok(());
                        };
                        if b.is_ascii_digit() || b == b';' {
                            num.push(b as char);
                        } else {
                            match b {
                                b'~' => match num.as_str() {
                                    "1" | "7" => self.buf.move_home(),
                                    "3" => {
                                        self.buf.delete();
                                    }
                                    "3;3" | "3;5" => self.buf.delete_next_word(),
                                    "4" | "8" => self.buf.move_end(),
                                    "200" => self.read_paste()?,
                                    _ => {}
                                },
                                // ESC [ 1 ; <mod> C/D — Alt/Ctrl + arrow.
                                b'C' if is_word_modifier(&num) => {
                                    self.buf.move_next_word();
                                }
                                b'D' if is_word_modifier(&num) => {
                                    self.buf.move_prev_word();
                                }
                                _ => {}
                            }
                            break;
                        }
                    }
                }
                _ => {}
            }
        } else if b1 == b'O' {
            match read_byte(self.in_fd)? {
                Some(b'H') => self.buf.move_home(),
                Some(b'F') => self.buf.move_end(),
                _ => {}
            }
        } else {
            // ESC-prefixed (Meta/Alt) word operations.
            match b1 {
                b'b' => {
                    self.buf.move_prev_word();
                }
                b'f' => {
                    self.buf.move_next_word();
                }
                b'd' => self.buf.delete_next_word(),
                0x08 | 0x7f => self.buf.delete_prev_word(),
                _ => {}
            }
        }
        Ok(())
    }

    /// Consumes a bracketed paste body up to `ESC [ 201 ~`, inserting it.
    fn read_paste(&mut self) -> io::Result<()> {
        const END: &[u8] = b"\x1b[201~";
        let mut data = Vec::new();
        while let Some(b) = read_byte(self.in_fd)? {
            data.push(b);
            if data.ends_with(END) {
                data.truncate(data.len() - END.len());
                break;
            }
        }
        let text = String::from_utf8_lossy(&data);
        self.buf.insert(strip_paste_markers(&text));
        Ok(())
    }

    fn history_move(&mut self, dir: i32) {
        // Only entries scoped to the current directory (plus untagged/global
        // ones) are visited; `history_index` indexes into this eligible list,
        // not the raw history (issue #49).
        let eligible: Vec<usize> = (0..self.history.len())
            .filter(|i| self.history.is_eligible(*i))
            .collect();
        if eligible.is_empty() {
            return;
        }
        let len = eligible.len();
        let new_index = match (self.history_index, dir) {
            (None, d) if d < 0 => {
                self.stash = self.buf.text().to_owned();
                Some(len - 1)
            }
            (None, _) => None,
            (Some(0), d) if d < 0 => Some(0),
            (Some(i), d) if d < 0 => Some(i - 1),
            (Some(i), _) if i + 1 < len => Some(i + 1),
            (Some(_), _) => {
                // Past the newest entry: restore the stashed in-progress line.
                self.buf.set_text(std::mem::take(&mut self.stash));
                self.history_index = None;
                return;
            }
        };
        self.history_index = new_index;
        if let Some(i) = new_index {
            let entry = eligible
                .get(i)
                .and_then(|h| self.history.get(*h))
                .unwrap_or_default()
                .to_owned();
            self.buf.set_text(entry);
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering (pure, testable)
// ---------------------------------------------------------------------------

/// Builds the escape-sequence frame that paints prompt+line and footer.
///
/// Layout: prompt line (with horizontal scrolling so the cursor stays
/// visible), then the footer on the next line, then the cursor is moved back
/// to its position on the prompt line. Embedded newlines display as `␤`.
fn render_frame(prompt: &str, line: &str, cursor: usize, footer: &str, cols: usize) -> String {
    let cols = cols.max(2);
    let display: String = line
        .chars()
        .map(|c| if c == '\n' { '␤' } else { c })
        .collect();
    let cursor_chars = line[..cursor].chars().count();
    let prompt_chars = prompt.chars().count();

    // Horizontal scroll: drop leading chars until the cursor fits.
    let avail = cols.saturating_sub(prompt_chars).max(1);
    let mut start = 0usize; // in chars
    if cursor_chars >= avail {
        start = cursor_chars + 1 - avail;
    }
    let visible: String = display
        .chars()
        .skip(start)
        .take(avail.saturating_sub(1) + 1)
        .collect();
    // Truncate footer to the terminal width (by chars; styling is caller's).
    let footer_visible: String = footer.chars().take(cols).collect();

    let col = prompt_chars + (cursor_chars - start) + 1; // 1-based
    format!("\r{prompt}{visible}\x1b[K\r\n{footer_visible}\x1b[K\x1b[A\r\x1b[{col}G")
}

/// Terminal width from `TIOCGWINSZ`, falling back to 80.
fn terminal_cols() -> usize {
    // SAFETY: winsize is plain-old-data; zeroed is a valid value that ioctl
    // overwrites on success.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    // SAFETY: stdout fd is valid and `ws` is a properly aligned, writable
    // winsize buffer, matching the TIOCGWINSZ contract.
    let rc = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &raw mut ws) };
    if rc == 0 && ws.ws_col > 0 {
        ws.ws_col as usize
    } else {
        DEFAULT_COLS
    }
}

/// Reads one byte from `fd`; `Ok(None)` on EOF.
fn read_byte(fd: RawFd) -> io::Result<Option<u8>> {
    // Use a File-like read via libc to avoid taking StdinLock (fd may be a
    // TTY in raw mode).
    let mut byte = [0u8; 1];
    loop {
        // SAFETY: fd is a valid open descriptor and `byte` is a writable
        // 1-byte buffer whose length is passed correctly.
        let n = unsafe { libc::read(fd, byte.as_mut_ptr().cast(), 1) };
        return match n {
            1 => Ok(Some(byte[0])),
            0 => Ok(None),
            _ => {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                Err(e)
            }
        };
    }
}

fn write_stdout(bytes: &[u8]) -> io::Result<()> {
    let mut out = io::stdout().lock();
    out.write_all(bytes)?;
    out.flush()
}

// ---------------------------------------------------------------------------
// External editor escape hatch (Ctrl-G in the TUI prompt)
// ---------------------------------------------------------------------------

/// Editor fallback when neither `$EDITOR` nor `$VISUAL` is set.
const DEFAULT_EDITOR: &str = "vi";

/// A resolved external-editor invocation: the program plus any fixed arguments
/// that came with it (`EDITOR="code -w"` → `code` + `["-w"]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorCommand {
    /// Program to spawn.
    pub program: String,
    /// Arguments preceding the temp-file path.
    pub args: Vec<String>,
}

/// Resolves the external editor from `$EDITOR`, then `$VISUAL`, then `vi`.
///
/// Pure in its inputs so the precedence is testable without touching the
/// process environment. Values are split on whitespace, which covers the
/// common `EDITOR="code -w"` / `EDITOR="emacs -nw"` forms; quoting and shell
/// metacharacters are deliberately not supported (the editor is spawned
/// directly, never through a shell). Blank or whitespace-only values are
/// treated as unset.
#[must_use]
pub fn resolve_editor_command(editor: Option<&str>, visual: Option<&str>) -> EditorCommand {
    let pick = [editor, visual]
        .into_iter()
        .flatten()
        .find(|v| !v.trim().is_empty());
    let mut parts = pick
        .unwrap_or(DEFAULT_EDITOR)
        .split_whitespace()
        .map(str::to_owned);
    let program = parts.next().unwrap_or_else(|| DEFAULT_EDITOR.to_owned());
    EditorCommand {
        program,
        args: parts.collect(),
    }
}

/// Resolves the external editor from the live process environment.
#[must_use]
pub fn editor_command_from_env() -> EditorCommand {
    let editor = std::env::var("EDITOR").ok();
    let visual = std::env::var("VISUAL").ok();
    resolve_editor_command(editor.as_deref(), visual.as_deref())
}

/// Path of the scratch file handed to the external editor. The `.md` suffix
/// makes editors syntax-highlight the prompt as Markdown.
fn scratch_path(dir: &Path) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    dir.join(format!("plank-prompt-{}-{nanos}.md", std::process::id()))
}

/// Normalizes what came back from the editor into a prompt, or `None` when the
/// user saved nothing meaningful. Trailing whitespace (including the newline
/// most editors add on save) is stripped; a blank file leaves the prompt alone.
#[must_use]
fn normalize_edited(text: &str) -> Option<String> {
    let trimmed = text.trim_end();
    (!trimmed.trim().is_empty()).then(|| trimmed.to_owned())
}

/// Round-trips `initial` through a scratch `.md` file in `dir`, handing the
/// path to `run` and reading the result back.
///
/// `run` reports whether the editor exited successfully; on a non-zero exit,
/// an empty/blank file, or any I/O error the result is `None` and the caller
/// keeps the prompt it had. The scratch file is always removed, including when
/// `run` fails. Split out from [`edit_text_externally`] so the file handling
/// is testable with a stub editor.
///
/// # Errors
///
/// Returns an error when the scratch file cannot be written, or when `run`
/// itself fails (the editor could not be spawned).
pub fn edit_text_with<F>(initial: &str, dir: &Path, run: F) -> io::Result<Option<String>>
where
    F: FnOnce(&Path) -> io::Result<bool>,
{
    let path = scratch_path(dir);
    fs::write(&path, initial)?;
    let outcome = run(&path);
    let edited = match &outcome {
        Ok(true) => fs::read_to_string(&path).ok(),
        Ok(false) | Err(_) => None,
    };
    let _ = fs::remove_file(&path);
    outcome?;
    Ok(edited.as_deref().and_then(normalize_edited))
}

/// Opens `$EDITOR` on a scratch file seeded with `initial` and returns the
/// edited prompt, or `None` to keep the existing input.
///
/// The caller is responsible for leaving raw mode and the alternate screen
/// first: this function hands the terminal straight to the child process.
///
/// # Errors
///
/// Returns an error when the scratch file cannot be written or the editor
/// cannot be spawned (typically `$EDITOR` naming a program that is not on
/// `PATH`). A non-zero editor exit is not an error — it yields `None`.
pub fn edit_text_externally(initial: &str) -> io::Result<Option<String>> {
    let cmd = editor_command_from_env();
    edit_text_with(initial, &std::env::temp_dir(), |path| {
        let status = std::process::Command::new(&cmd.program)
            .args(&cmd.args)
            .arg(path)
            .status()?;
        Ok(status.success())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- word boundaries ----

    #[test]
    fn prev_word_boundary_skips_whitespace_then_word() {
        assert_eq!(prev_word_boundary("", 0), 0);
        assert_eq!(prev_word_boundary("hello", 0), 0);
        assert_eq!(prev_word_boundary("hello", 5), 0);
        assert_eq!(prev_word_boundary("hello", 3), 0);
        assert_eq!(prev_word_boundary("foo bar", 7), 4);
        // Trailing whitespace is skipped before the word is consumed.
        assert_eq!(prev_word_boundary("foo bar   ", 10), 4);
        assert_eq!(prev_word_boundary("foo   bar", 6), 0);
        // Leading whitespace collapses to the start of the line.
        assert_eq!(prev_word_boundary("   foo", 6), 3);
        assert_eq!(prev_word_boundary("   foo", 3), 0);
        // Tabs and newlines count as whitespace.
        assert_eq!(prev_word_boundary("foo\tbar", 7), 4);
        assert_eq!(prev_word_boundary("foo\nbar", 7), 4);
    }

    #[test]
    fn prev_word_boundary_lands_on_char_boundaries() {
        // "é" is two bytes; the returned offset must be sliceable.
        let s = "héllo wörld";
        let b = prev_word_boundary(s, s.len());
        assert!(s.is_char_boundary(b));
        assert_eq!(&s[b..], "wörld");
        let b2 = prev_word_boundary(s, b);
        assert_eq!(b2, 0);
        // A non-ASCII separator (NBSP) is whitespace too.
        let t = "foo\u{a0}bar";
        let b3 = prev_word_boundary(t, t.len());
        assert!(t.is_char_boundary(b3));
        assert_eq!(&t[b3..], "bar");
    }

    #[test]
    fn next_word_boundary_skips_whitespace_then_word() {
        assert_eq!(next_word_boundary("", 0), 0);
        assert_eq!(next_word_boundary("hello", 0), 5);
        assert_eq!(next_word_boundary("hello", 5), 5);
        assert_eq!(next_word_boundary("foo bar", 0), 3);
        assert_eq!(next_word_boundary("foo bar", 3), 7);
        assert_eq!(next_word_boundary("foo   bar", 3), 9);
        assert_eq!(next_word_boundary("foo   ", 3), 6);
        let s = "héllo wörld";
        let b = next_word_boundary(s, 0);
        assert!(s.is_char_boundary(b));
        assert_eq!(&s[..b], "héllo");
    }

    #[test]
    fn word_motions_and_deletes() {
        let mut b = LineBuffer::new();
        b.set_text("alpha beta gamma");
        assert!(b.move_prev_word());
        assert_eq!(b.cursor(), 11);
        assert!(b.move_prev_word());
        assert_eq!(b.cursor(), 6);
        assert!(b.move_prev_word());
        assert_eq!(b.cursor(), 0);
        assert!(!b.move_prev_word());

        assert!(b.move_next_word());
        assert_eq!(b.cursor(), 5);
        assert!(b.move_next_word());
        assert_eq!(b.cursor(), 10);
        assert!(b.move_next_word());
        assert_eq!(b.cursor(), 16);
        assert!(!b.move_next_word());

        // Alt-Backspace at end of line eats the last word.
        b.delete_prev_word();
        assert_eq!(b.text(), "alpha beta ");
        assert_eq!(b.cursor(), 11);

        // Alt-D from the start eats the word ahead, cursor unmoved.
        b.move_home();
        b.delete_next_word();
        assert_eq!(b.text(), " beta ");
        assert_eq!(b.cursor(), 0);
        b.delete_next_word();
        assert_eq!(b.text(), " ");
        // Nothing left but whitespace: the delete consumes it and stops.
        b.delete_next_word();
        assert_eq!(b.text(), "");
        b.delete_next_word();
        assert_eq!(b.text(), "");
    }

    #[test]
    fn csi_word_modifier_detection() {
        assert!(is_word_modifier("1;3")); // Alt
        assert!(is_word_modifier("1;5")); // Ctrl
        assert!(is_word_modifier("1;7")); // Ctrl+Alt
        assert!(!is_word_modifier("1;2")); // Shift
        assert!(!is_word_modifier("1"));
        assert!(!is_word_modifier(""));
    }

    // ---- LineBuffer ----

    #[test]
    fn insert_and_move() {
        let mut b = LineBuffer::new();
        b.insert("héllo");
        assert_eq!(b.text(), "héllo");
        assert!(b.move_left());
        assert!(b.move_left());
        b.insert("X");
        assert_eq!(b.text(), "hélXlo");
        b.move_home();
        assert!(!b.move_left());
        b.move_end();
        assert!(!b.move_right());
    }

    /// A buffer nobody asks to select behaves exactly as it did before the
    /// anchor existed.
    #[test]
    fn nothing_is_selected_until_the_anchor_is_pinned() {
        let mut b = LineBuffer::new();
        b.insert("hello");
        b.move_left();
        b.move_left();
        assert_eq!(b.selection(), None);
        assert_eq!(b.selected_text(), None);
        b.backspace();
        assert_eq!(b.text(), "helo");
    }

    #[test]
    fn shift_motions_grow_a_selection_from_one_anchor() {
        let mut b = LineBuffer::new();
        b.insert("hello world");
        // Shift+Left three times: the anchor stays where the first one dropped.
        for _ in 0..3 {
            b.anchor_here();
            b.move_left();
        }
        assert_eq!(b.selected_text(), Some("rld"));
        // Shifting back the other way shrinks the same selection.
        b.anchor_here();
        b.move_right();
        assert_eq!(b.selected_text(), Some("ld"));
    }

    #[test]
    fn a_selection_collapsed_back_onto_its_anchor_is_no_selection() {
        let mut b = LineBuffer::new();
        b.insert("hi");
        b.anchor_here();
        b.move_left();
        b.anchor_here();
        b.move_right();
        assert_eq!(b.selection(), None, "zero-width selection must not linger");
    }

    #[test]
    fn a_plain_motion_drops_the_selection() {
        let mut b = LineBuffer::new();
        b.insert("hello");
        b.anchor_here();
        b.move_left();
        assert!(b.selection().is_some());
        b.clear_selection();
        b.move_left();
        assert_eq!(b.selection(), None);
    }

    #[test]
    fn typing_over_a_selection_replaces_it() {
        let mut b = LineBuffer::new();
        b.insert("hello world");
        b.anchor_here();
        for _ in 0..5 {
            b.move_left();
        }
        assert_eq!(b.selected_text(), Some("world"));
        b.insert("there");
        assert_eq!(b.text(), "hello there");
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor(), b.text().len());
    }

    #[test]
    fn backspace_and_delete_take_the_whole_selection() {
        for delete in [false, true] {
            let mut b = LineBuffer::new();
            b.insert("abcdef");
            b.set_cursor(2);
            b.anchor_here();
            b.set_cursor(5);
            if delete {
                assert!(b.delete());
            } else {
                assert!(b.backspace());
            }
            assert_eq!(b.text(), "abf");
            assert_eq!(b.cursor(), 2);
            assert_eq!(b.selection(), None);
        }
    }

    #[test]
    fn select_all_then_type_replaces_the_line() {
        let mut b = LineBuffer::new();
        b.insert("throw this away");
        b.select_all();
        assert_eq!(b.selected_text(), Some("throw this away"));
        b.insert("new");
        assert_eq!(b.text(), "new");
    }

    #[test]
    fn selection_survives_utf8_and_never_splits_a_char() {
        let mut b = LineBuffer::new();
        b.insert("aé漢b");
        b.anchor_here();
        b.move_left(); // over 'b'
        b.move_left(); // over 漢
        assert_eq!(b.selected_text(), Some("漢b"));
        // A byte offset landing mid-char snaps down to the boundary.
        b.set_cursor(2); // inside 'é' (a=0, é=1..3)
        assert_eq!(b.cursor(), 1);
    }

    #[test]
    fn set_text_and_clear_drop_the_selection() {
        let mut b = LineBuffer::new();
        b.insert("hello");
        b.select_all();
        b.set_text("from history");
        assert_eq!(b.selection(), None);
        b.select_all();
        b.clear();
        assert_eq!(b.selection(), None);
    }

    #[test]
    fn line_motions_hold_the_column_across_logical_lines() {
        let mut b = LineBuffer::new();
        b.insert("alpha\nbe\ngamma");
        b.set_cursor(b.text().len()); // end of "gamma", column 5
        assert!(b.move_line_up());
        // "be" is shorter than column 5, so the cursor clamps to its end.
        assert_eq!(b.text()[..b.cursor()].to_owned(), "alpha\nbe");
        // The column is not sticky: continuing up carries column 2, not 5.
        assert!(b.move_line_up());
        assert_eq!(b.text()[..b.cursor()].to_owned(), "al");
        assert!(b.move_line_down());
        assert_eq!(b.text()[..b.cursor()].to_owned(), "alpha\nbe");
    }

    #[test]
    fn line_motions_run_to_the_ends_on_a_single_line() {
        let mut b = LineBuffer::new();
        b.insert("one line");
        assert!(b.move_line_up());
        assert_eq!(b.cursor(), 0, "up on the first line goes to the start");
        assert!(!b.move_line_up(), "already there");
        assert!(b.move_line_down());
        assert_eq!(b.cursor(), b.text().len());
        assert!(!b.move_line_down());
    }

    #[test]
    fn backspace_and_delete_utf8() {
        let mut b = LineBuffer::new();
        b.insert("aé漢b");
        b.backspace(); // remove 'b'
        assert_eq!(b.text(), "aé漢");
        b.move_left(); // before 漢
        assert!(b.delete()); // remove 漢
        assert_eq!(b.text(), "aé");
        b.move_home();
        assert!(!b.backspace());
    }

    #[test]
    fn kill_ops() {
        let mut b = LineBuffer::new();
        b.insert("one two three");
        b.move_home();
        b.move_right();
        b.move_right();
        b.move_right();
        b.kill_to_end();
        assert_eq!(b.text(), "one");
        b.insert(" two");
        b.kill_to_start();
        assert_eq!(b.text(), "");
    }

    #[test]
    fn delete_prev_word() {
        let mut b = LineBuffer::new();
        b.insert("foo bar  baz");
        b.delete_prev_word();
        assert_eq!(b.text(), "foo bar  ");
        b.delete_prev_word();
        assert_eq!(b.text(), "foo ");
        b.delete_prev_word();
        assert_eq!(b.text(), "");
    }

    #[test]
    fn word_before_cursor() {
        let mut b = LineBuffer::new();
        b.insert("git com");
        assert_eq!(b.word_before_cursor(), (4, 7));
        b.replace_range(4, 7, "commit");
        assert_eq!(b.text(), "git commit");
        assert_eq!(b.cursor(), 10);
    }

    // ---- History ----

    #[test]
    fn history_dedup_and_cap() {
        let mut h = History::new(3);
        h.add("a");
        h.add("a"); // consecutive dup skipped
        h.add("");
        h.add("b");
        h.add("c");
        h.add("d"); // evicts "a"
        assert_eq!(h.len(), 3);
        assert_eq!(h.get(0), Some("b"));
        assert_eq!(h.get(2), Some("d"));
    }

    /// `History::live` must not capture `ui.historySize` at construction: a
    /// change to the setting has to be observed on the very next trim, the
    /// same live behaviour `complete::refresh_throttle` gives
    /// `ui.indexRefreshSecs`.
    #[test]
    fn live_history_tracks_history_size_setting_changes() {
        let mut s = crate::settings::Settings::default();
        s.ui.history_size = 2;
        crate::settings::install_for_test(s);

        let mut h = History::live();
        h.add("a");
        h.add("b");
        h.add("c"); // cap is 2: evicts "a"
        assert_eq!(h.len(), 2);
        assert_eq!(h.get(0), Some("b"));

        // Lower the cap live, with no new `History` constructed.
        let mut s = crate::settings::Settings::default();
        s.ui.history_size = 1;
        crate::settings::install_for_test(s);
        h.add("d");
        assert_eq!(
            h.len(),
            1,
            "the next add catches up to the new, smaller cap"
        );
        assert_eq!(h.get(0), Some("d"));
    }

    #[test]
    fn history_load_save_roundtrip() {
        let dir = std::env::temp_dir().join(format!("plank_hist_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("hist");
        let mut h = History::new(10);
        h.add("first");
        h.add("second");
        h.save(&path).unwrap();
        let mut h2 = History::new(10);
        h2.load(&path).unwrap();
        assert_eq!(h2.len(), 2);
        assert_eq!(h2.get(1), Some("second"));
        // Missing file is fine.
        let mut h3 = History::new(10);
        h3.load(dir.join("nope")).unwrap();
        assert!(h3.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_dir_tagging_filters_eligibility() {
        let mut h = History::new(10);
        h.set_cwd(Some("/proj/a".into()));
        h.add("cargo build"); // tagged /proj/a
        h.set_cwd(Some("/proj/b".into()));
        h.add("npm run"); // tagged /proj/b
        // Viewed from /proj/a only the /proj/a entry is eligible.
        h.set_cwd(Some("/proj/a".into()));
        assert!(h.is_eligible(0));
        assert!(!h.is_eligible(1));
        // Viewed from /proj/b it flips.
        h.set_cwd(Some("/proj/b".into()));
        assert!(!h.is_eligible(0));
        assert!(h.is_eligible(1));
    }

    #[test]
    fn history_legacy_untagged_entries_are_always_eligible() {
        // A pre-tagging file is just plain lines: they load untagged and stay
        // visible from any directory (global fallback, no data lost on upgrade).
        let dir = std::env::temp_dir().join(format!("plank_hist_legacy_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("hist");
        fs::write(&path, "old one\nold two\n").unwrap();
        let mut h = History::new(10);
        h.load(&path).unwrap();
        h.set_cwd(Some("/somewhere/else".into()));
        assert_eq!(h.len(), 2);
        assert!(h.is_eligible(0));
        assert!(h.is_eligible(1));
        assert_eq!(h.dir_of(0), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_save_roundtrip_preserves_dir_tags() {
        let dir = std::env::temp_dir().join(format!("plank_hist_tag_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("hist");
        let mut h = History::new(10);
        h.add_in_dir("global cmd", None);
        h.add_in_dir("scoped cmd", Some("/proj/a".into()));
        h.save(&path).unwrap();
        let mut h2 = History::new(10);
        h2.load(&path).unwrap();
        assert_eq!(h2.len(), 2);
        assert_eq!(h2.get(0), Some("global cmd"));
        assert_eq!(h2.dir_of(0), None);
        assert_eq!(h2.get(1), Some("scoped cmd"));
        assert_eq!(h2.dir_of(1), Some("/proj/a"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_no_project_scope_when_dir_unresolved() {
        // Launched somewhere the cwd tag can't be resolved: new entries are
        // untagged and remain eligible everywhere rather than vanishing.
        let mut h = History::new(10);
        h.set_cwd(None);
        h.add("still visible");
        assert_eq!(h.dir_of(0), None);
        assert!(h.is_eligible(0));
        h.set_cwd(Some("/anywhere".into()));
        assert!(h.is_eligible(0));
    }

    // ---- Completion cycling ----

    #[test]
    fn completion_single_candidate() {
        let mut cs = CompletionState::default();
        let got = cs.advance("com", vec!["commit".into()]).unwrap().to_owned();
        assert_eq!(got, "commit");
        assert!(cs.is_single());
    }

    #[test]
    fn completion_cycles_through_original() {
        let mut cs = CompletionState::default();
        let cands = vec!["cat".into(), "car".into()];
        assert_eq!(cs.advance("ca", cands).unwrap(), "cat");
        assert_eq!(cs.advance("ca", Vec::new()).unwrap(), "car");
        assert_eq!(cs.advance("ca", Vec::new()).unwrap(), "ca"); // original
        assert_eq!(cs.advance("ca", Vec::new()).unwrap(), "cat"); // wraps
    }

    #[test]
    fn completion_no_candidates() {
        let mut cs = CompletionState::default();
        assert!(cs.advance("zz", Vec::new()).is_none());
        assert!(!cs.active);
    }

    // ---- Paste ----

    #[test]
    fn paste_markers_stripped_newlines_kept() {
        let s = strip_paste_markers("\x1b[200~line1\rline2\r\nline3\x1b[201~");
        assert_eq!(s, "line1\nline2\nline3");
    }

    // ---- Rendering ----

    #[test]
    fn render_frame_basic() {
        let f = render_frame("> ", "hi", 2, "status", 80);
        assert!(f.starts_with("\r> hi\x1b[K\r\nstatus\x1b[K"));
        assert!(f.ends_with("\x1b[5G")); // prompt(2) + cursor(2) + 1
    }

    #[test]
    fn render_frame_scrolls_horizontally() {
        let line = "abcdefghij";
        let f = render_frame("> ", line, line.len(), "s", 8);
        // avail = 8 - 2 = 6; cursor at 10 -> start = 5, visible "fghij".
        assert!(f.contains("fghij"));
        assert!(!f.contains("abcde"));
    }

    #[test]
    fn render_frame_newline_placeholder() {
        let f = render_frame("> ", "a\nb", 3, "s", 80);
        assert!(f.contains("a␤b"));
    }

    #[test]
    fn render_frame_truncates_footer() {
        let f = render_frame("> ", "", 0, "0123456789", 5);
        assert!(f.contains("\r\n01234\x1b[K"));
    }

    // ---- external editor ----

    #[test]
    fn editor_precedence_editor_then_visual_then_default() {
        assert_eq!(
            resolve_editor_command(Some("nano"), Some("emacs")).program,
            "nano"
        );
        assert_eq!(resolve_editor_command(None, Some("emacs")).program, "emacs");
        assert_eq!(resolve_editor_command(None, None).program, "vi");
        // Blank / whitespace-only values count as unset.
        assert_eq!(resolve_editor_command(Some("  "), None).program, "vi");
        assert_eq!(
            resolve_editor_command(Some(""), Some("emacs")).program,
            "emacs"
        );
    }

    #[test]
    fn editor_command_splits_arguments() {
        let cmd = resolve_editor_command(Some("code -w --new-window"), None);
        assert_eq!(cmd.program, "code");
        assert_eq!(cmd.args, vec!["-w", "--new-window"]);
        assert!(resolve_editor_command(Some("vim"), None).args.is_empty());
    }

    #[test]
    fn scratch_file_has_markdown_suffix() {
        let p = scratch_path(Path::new("/tmp"));
        assert_eq!(p.extension().and_then(|e| e.to_str()), Some("md"));
        assert!(p.starts_with("/tmp"));
    }

    #[test]
    fn edit_round_trip_seeds_and_reads_back() {
        let dir = std::env::temp_dir();
        let seen = std::cell::RefCell::new(String::new());
        let out = edit_text_with("hello", &dir, |p| {
            *seen.borrow_mut() = fs::read_to_string(p)?;
            fs::write(p, "hello world\n")?;
            Ok(true)
        })
        .unwrap();
        assert_eq!(seen.into_inner(), "hello");
        // The newline the editor appended on save is stripped.
        assert_eq!(out.as_deref(), Some("hello world"));
    }

    #[test]
    fn edit_returns_none_on_failure_or_blank() {
        let dir = std::env::temp_dir();
        // Non-zero editor exit: keep the existing prompt.
        let out = edit_text_with("keep me", &dir, |p| {
            fs::write(p, "discard me")?;
            Ok(false)
        })
        .unwrap();
        assert_eq!(out, None);
        // Emptied file: keep the existing prompt.
        let out = edit_text_with("keep me", &dir, |p| {
            fs::write(p, "   \n\n")?;
            Ok(true)
        })
        .unwrap();
        assert_eq!(out, None);
    }

    #[test]
    fn edit_cleans_up_scratch_file() {
        let dir = std::env::temp_dir();
        let path = std::cell::RefCell::new(PathBuf::new());
        let _ = edit_text_with("x", &dir, |p| {
            *path.borrow_mut() = p.to_path_buf();
            Ok(true)
        });
        assert!(!path.borrow().exists());
        // Also on an editor spawn error.
        let path = std::cell::RefCell::new(PathBuf::new());
        let err = edit_text_with("x", &dir, |p| {
            *path.borrow_mut() = p.to_path_buf();
            Err(io::Error::other("boom"))
        });
        assert!(err.is_err());
        assert!(!path.borrow().exists());
    }
}

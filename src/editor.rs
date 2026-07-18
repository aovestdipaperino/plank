//! Terminal line editor with a status footer, ported from the ds4 agent.
//!
//! This is a faithful-but-tractable port of the linenoise-derived editor in
//! `ds4-ref/ds4_agent.c` ("Terminal Prompt, Status Footer, And Async Output
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

/// An editable UTF-8 line with a cursor, mirroring linenoise's edit ops.
#[derive(Debug, Default, Clone)]
pub struct LineBuffer {
    text: String,
    /// Byte offset of the cursor; always on a char boundary.
    cursor: usize,
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

    /// Replaces the whole line and puts the cursor at the end.
    pub fn set_text(&mut self, text: impl AsRef<str>) {
        text.as_ref().clone_into(&mut self.text);
        self.cursor = self.text.len();
    }

    /// Clears the line.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    /// Inserts a string at the cursor and advances past it.
    pub fn insert(&mut self, s: impl AsRef<str>) {
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

    /// Deletes the character before the cursor (Backspace).
    pub fn backspace(&mut self) -> bool {
        match self.prev_boundary() {
            Some(b) => {
                self.text.replace_range(b..self.cursor, "");
                self.cursor = b;
                true
            }
            None => false,
        }
    }

    /// Deletes the character under the cursor (Delete / Ctrl-D on non-empty).
    pub fn delete(&mut self) -> bool {
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
        self.text.truncate(self.cursor);
    }

    /// Deletes from the start of the line to the cursor (Ctrl-U).
    pub fn kill_to_start(&mut self) {
        self.text.replace_range(..self.cursor, "");
        self.cursor = 0;
    }

    /// Deletes the word before the cursor (Ctrl-W), linenoise style.
    pub fn delete_prev_word(&mut self) {
        let bytes = self.text.as_bytes();
        let mut start = self.cursor;
        while start > 0 && bytes[start - 1] == b' ' {
            start -= 1;
        }
        while start > 0 && bytes[start - 1] != b' ' {
            start -= 1;
        }
        // `start` lands after a space (or 0), so it is a char boundary.
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
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
        let s = s.as_ref();
        self.text.replace_range(start..end, s);
        self.cursor = start + s.len();
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

/// A bounded command-history ring with consecutive-duplicate suppression.
#[derive(Debug, Clone)]
pub struct History {
    entries: VecDeque<String>,
    max: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::new(HISTORY_MAX)
    }
}

impl History {
    /// Creates an empty history bounded to `max` entries.
    #[must_use]
    pub fn new(max: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max: max.max(1),
        }
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

    /// Returns the entry at `idx` (0 = oldest), if present.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&str> {
        self.entries.get(idx).map(String::as_str)
    }

    /// Appends an entry, skipping empties and consecutive duplicates.
    pub fn add(&mut self, entry: impl AsRef<str>) {
        let entry = entry.as_ref();
        if entry.is_empty() || self.entries.back().is_some_and(|last| last == entry) {
            return;
        }
        if self.entries.len() == self.max {
            self.entries.pop_front();
        }
        self.entries.push_back(entry.to_owned());
    }

    /// Loads newline-separated history from `path`.
    ///
    /// A missing file is not an error (fresh start).
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
            self.add(line);
        }
        Ok(())
    }

    /// Saves the history to `path`, newline-separated.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing the file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut out = String::new();
        for e in &self.entries {
            out.push_str(e);
            out.push('\n');
        }
        fs::write(path.as_ref(), out)
    }
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
                        if b.is_ascii_digit() {
                            num.push(b as char);
                        } else {
                            if b == b'~' {
                                match num.as_str() {
                                    "1" | "7" => self.buf.move_home(),
                                    "3" => {
                                        self.buf.delete();
                                    }
                                    "4" | "8" => self.buf.move_end(),
                                    "200" => self.read_paste()?,
                                    _ => {}
                                }
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
        if self.history.is_empty() {
            return;
        }
        let len = self.history.len();
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
            let entry = self.history.get(i).unwrap_or_default().to_owned();
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

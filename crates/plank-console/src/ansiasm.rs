// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Stateful line assembly over `turbo_vision`'s stateless ANSI parser.

use turbo_vision::core::ansi::AnsiParser;
use turbo_vision::core::draw::Cell;
use turbo_vision::core::palette::{Attr, TvColor};

/// `\r\x1b[0K` — cursor to column 0, erase to end of line. `plank_stream`
/// emits exactly this byte sequence when it repaints a fenced code block's
/// first line once the language is known.
const ERASE_LINE: &[u8] = b"\r\x1b[0K";

/// The SGR sequence that reproduces `attr` from a fresh parser state.
///
/// `AnsiParser::parse_sgr` is private, so attribute state cannot be handed
/// from one `parse_line` call to the next directly. Re-encoding it as a
/// prefix works instead: escape sequences produce no cells, so the prefix is
/// invisible in the parsed output.
#[must_use]
pub fn attr_to_sgr(attr: Attr) -> String {
    // `AnsiParser::ansi256_to_tv_color` maps indices 0-15 in ANSI's own
    // color order (Black, Red, Green, Brown/Yellow, Blue, Magenta, Cyan,
    // LightGray, then the bright variants), which differs from `TvColor`'s
    // discriminant order (`to_index`, CGA order: Black, Blue, Green, Cyan,
    // Red, ...). Re-encoding must invert `ansi256_to_tv_color`, not
    // `to_index`.
    fn ansi256_index(color: TvColor) -> u8 {
        match color {
            TvColor::Black | TvColor::Rgb { .. } => 0,
            TvColor::Red => 1,
            TvColor::Green => 2,
            TvColor::Brown => 3,
            TvColor::Blue => 4,
            TvColor::Magenta => 5,
            TvColor::Cyan => 6,
            TvColor::LightGray => 7,
            TvColor::DarkGray => 8,
            TvColor::LightRed => 9,
            TvColor::LightGreen => 10,
            TvColor::Yellow => 11,
            TvColor::LightBlue => 12,
            TvColor::LightMagenta => 13,
            TvColor::LightCyan => 14,
            TvColor::White => 15,
        }
    }
    fn one(kind: u8, color: TvColor) -> String {
        match color {
            TvColor::Rgb { r, g, b } => format!("\x1b[{kind};2;{r};{g};{b}m"),
            other => format!("\x1b[{kind};5;{}m", ansi256_index(other)),
        }
    }
    format!("{}{}", one(38, attr.fg), one(48, attr.bg))
}

/// Feeds a byte stream into `AnsiParser` one line at a time, carrying SGR
/// state across line breaks and holding back incomplete input.
///
/// Deliberate limits, not bugs to fix: `Attr` carries only `fg` and `bg`, so
/// ANSI bold becomes a brighter foreground (that is what `AnsiParser`
/// already does) and italic is dropped entirely — plank's dim-italic
/// thinking style arrives as its 256-color grey with the italic lost.
/// Turbo Vision cells have no italic attribute. Also deliberate: the
/// `ERASE_LINE` repaint marker (`\r\x1b[0K`) is detected as a literal
/// 4-byte match anywhere it appears in `pending`, with no surrounding
/// context check. Content that happens to quote that exact byte sequence
/// (a fenced code block showing raw ANSI, a pasted terminal transcript)
/// would have everything before it on the line silently discarded, same as
/// a real repaint. This mirrors `AnsiParser`'s own no-erase-in-line design
/// and is not worth adding machinery to disambiguate.
pub struct AnsiLineAssembler {
    parser: AnsiParser,
    /// Bytes received since the last newline.
    pending: Vec<u8>,
    /// Attribute in force at the start of `pending`.
    carry: Attr,
    /// Complete lines cut but not yet taken.
    ready: Vec<Vec<Cell>>,
}

impl std::fmt::Debug for AnsiLineAssembler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `AnsiParser` does not implement `Debug`, so `parser` is omitted
        // (its state is not observable anyway; it holds only defaults).
        f.debug_struct("AnsiLineAssembler")
            .field("pending_len", &self.pending.len())
            .field("carry", &self.carry)
            .field("ready_len", &self.ready.len())
            .finish_non_exhaustive()
    }
}

impl AnsiLineAssembler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            parser: AnsiParser::new(),
            pending: Vec::new(),
            carry: Attr::new(TvColor::LightGray, TvColor::Black),
            ready: Vec::new(),
        }
    }

    /// Accepts more stream bytes, cutting any complete lines.
    pub fn push(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if b == b'\n' {
                let line = self.parse_pending();
                self.carry = self.trailing_attr_of_pending();
                self.ready.push(line);
                self.pending.clear();
            } else {
                self.pending.push(b);
                if self.pending.ends_with(ERASE_LINE) {
                    // `plank_stream`'s fence highlighter repaints a line it
                    // already wrote plain by moving the cursor home and
                    // erasing to end of line, then rewriting it highlighted.
                    // `AnsiParser` has no concept of cursor position or
                    // erase-in-line — it only understands SGR — so without
                    // this the erased text and the erase sequence itself
                    // would be parsed as literal/garbage content ahead of
                    // the repaint. Since the cursor is at column 0, "erase
                    // to end of line" here means "discard the whole line so
                    // far".
                    self.pending.clear();
                }
            }
        }
    }

    /// Removes and returns every line cut so far.
    pub fn take_complete_lines(&mut self) -> Vec<Vec<Cell>> {
        std::mem::take(&mut self.ready)
    }

    /// The line currently being assembled, rendered as far as it has arrived.
    ///
    /// A trailing incomplete escape sequence is withheld, so a half-received
    /// `\x1b[3` never appears on screen as literal `[3`.
    #[must_use]
    pub fn partial_line(&self) -> Vec<Cell> {
        self.parse_pending()
    }

    /// Emits the in-progress line as final, if there is one. Used when a
    /// stream ends without a trailing newline.
    pub fn flush(&mut self) -> Option<Vec<Cell>> {
        if self.pending.is_empty() {
            return None;
        }
        let line = self.parse_pending();
        self.carry = self.trailing_attr_of_pending();
        self.pending.clear();
        Some(line)
    }

    /// Parses `pending` with the carried attribute prefixed, minus any
    /// trailing incomplete escape.
    fn parse_pending(&self) -> Vec<Cell> {
        let usable = &self.pending[..complete_len(&self.pending)];
        let text = String::from_utf8_lossy(usable);
        let with_state = format!("{}{}", attr_to_sgr(self.carry), text);
        self.parser.parse_line(&with_state)
    }

    /// The attribute in force at the end of `pending`, for a line that
    /// produced no cells (e.g. a line holding only escape sequences).
    fn trailing_attr_of_pending(&self) -> Attr {
        let usable = &self.pending[..complete_len(&self.pending)];
        let text = String::from_utf8_lossy(usable);
        let probe = format!("{}{}X", attr_to_sgr(self.carry), text);
        self.parser
            .parse_line(&probe)
            .last()
            .map_or(self.carry, |c| c.attr)
    }
}

impl Default for AnsiLineAssembler {
    fn default() -> Self {
        Self::new()
    }
}

/// Length of `buf` excluding a trailing incomplete escape sequence.
///
/// A CSI sequence is `ESC [` then parameter bytes `0x30..=0x3f`, then a final
/// byte `0x40..=0x7e`. Anything after a bare `ESC` that has not yet reached a
/// final byte is incomplete and must be held back.
fn complete_len(buf: &[u8]) -> usize {
    let Some(esc) = buf.iter().rposition(|&b| b == 0x1b) else {
        return buf.len();
    };
    let tail = &buf[esc..];
    // `ESC` alone, or `ESC [` with no final byte yet.
    if tail.len() == 1 {
        return esc;
    }
    if tail[1] != b'[' {
        // Not a CSI introducer; the parser ignores it, let it through.
        return buf.len();
    }
    if tail[2..].iter().any(|&b| (0x40..=0x7e).contains(&b)) {
        buf.len()
    } else {
        esc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_vision::core::palette::TvColor;

    fn text(cells: &[Cell]) -> String {
        cells.iter().map(|c| c.ch).collect()
    }

    #[test]
    fn splits_on_newline_and_holds_the_tail() {
        let mut a = AnsiLineAssembler::new();
        a.push(b"one\ntwo");
        let lines = a.take_complete_lines();
        assert_eq!(lines.len(), 1);
        assert_eq!(text(&lines[0]), "one");
        assert_eq!(text(&a.partial_line()), "two");
    }

    #[test]
    fn attribute_carries_across_a_line_break() {
        let mut a = AnsiLineAssembler::new();
        a.push(b"\x1b[31mred one\nstill red");
        let lines = a.take_complete_lines();
        assert_eq!(lines[0].last().unwrap().attr.fg, TvColor::Red);
        let partial = a.partial_line();
        assert_eq!(
            partial[0].attr.fg,
            TvColor::Red,
            "SGR state must survive the newline"
        );
    }

    #[test]
    fn byte_at_a_time_matches_whole_delivery() {
        let input = b"\x1b[1;32mgreen\x1b[0m plain\nnext\n";
        let mut whole = AnsiLineAssembler::new();
        whole.push(input);
        let expected = whole.take_complete_lines();

        let mut drip = AnsiLineAssembler::new();
        let mut got = Vec::new();
        for b in input {
            drip.push(&[*b]);
            got.extend(drip.take_complete_lines());
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn escape_split_across_chunks_is_not_shown_as_text() {
        let mut a = AnsiLineAssembler::new();
        a.push(b"x\x1b[3");
        assert_eq!(
            text(&a.partial_line()),
            "x",
            "an incomplete escape must not leak as literal characters"
        );
        a.push(b"1mY");
        assert_eq!(text(&a.partial_line()), "xY");
        assert_eq!(a.partial_line()[1].attr.fg, TvColor::Red);
    }

    #[test]
    fn carriage_return_is_dropped_not_rendered() {
        let mut a = AnsiLineAssembler::new();
        a.push(b"abc\r\n");
        let lines = a.take_complete_lines();
        assert_eq!(text(&lines[0]), "abc");
    }

    #[test]
    fn fence_repaint_replaces_the_line_instead_of_appending_to_it() {
        // `plank_stream`'s fence highlighter writes a code line plain, then
        // once it learns the language, repaints it: `\r\x1b[0K` moves the
        // cursor to column 0 and erases to end of line, followed by the
        // syntax-highlighted rewrite. `AnsiParser` has no concept of cursor
        // position or erase-in-line, so without special-casing this exact
        // sequence its "invalid escape" fallback swallows the *next*
        // escape's `m` terminator, silently eating the highlight colors —
        // and without any special-casing at all, the plain and highlighted
        // text would simply concatenate on one line.
        let mut a = AnsiLineAssembler::new();
        a.push(b"fn main() {}\x1b[0m\r\x1b[0K\x1b[38;5;214mfn\x1b[0m main() {}\n");
        let lines = a.take_complete_lines();
        assert_eq!(
            text(&lines[0]),
            "fn main() {}",
            "the plain pre-repaint text must not survive alongside the repaint"
        );
        assert_ne!(
            lines[0][0].attr.fg,
            TvColor::LightGray,
            "the repainted line must carry the highlight color, not the default"
        );
    }

    #[test]
    fn trailing_sgr_after_the_last_char_does_not_bleed_into_the_next_line() {
        // `plank_stream` closes `<think>` with a reset that lands *after*
        // the last visible character on the line, e.g. `...pondering\x1b[0m`.
        // The old code took `carry` from the last cell's attribute, which
        // predates that trailing reset — dropping it and letting the
        // thinking color bleed onto the next line.
        let mut a = AnsiLineAssembler::new();
        a.push(b"\x1b[38;5;8mpondering\x1b[0m\nplain text\n");
        let lines = a.take_complete_lines();
        assert_eq!(
            lines[1][0].attr.fg,
            TvColor::LightGray,
            "the reset after the last char of line 0 must carry into line 1, \
             not line 0's last cell color"
        );
    }

    #[test]
    fn flush_emits_a_trailing_line_without_a_newline() {
        let mut a = AnsiLineAssembler::new();
        a.push(b"tail");
        assert_eq!(text(&a.flush().unwrap()), "tail");
        assert!(a.flush().is_none(), "flush must be idempotent");
    }
}

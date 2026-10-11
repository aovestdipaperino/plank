// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `/useless-buddy`: a small picture standing on the rule above the prompt.
//!
//! It does nothing, which is the point. Only terminals that speak the kitty
//! graphics protocol *with Unicode placeholders* can show it (kitty itself and
//! Ghostty); everywhere else the command says so and leaves the screen alone.
//!
//! The picture is `~/.plank/buddy.png` when that file exists, else a plank
//! character compiled in from `src/resources/buddy.png`, which is free to ship
//! with the code.
//!
//! The image is placed with the protocol's Unicode placeholders rather than at
//! a cursor position. Transmitting it once creates a *virtual* placement
//! ([`transmit_sequence`]); it then shows wherever the screen holds U+10EEEE
//! cells whose foreground color is the image id and whose two combining marks
//! name the cell's row and column of the picture ([`render`]). Those are
//! ordinary cells in ratatui's buffer, so the frame diff moves, covers and
//! erases the picture like any text: a pane drawn over it hides it, a frame
//! without the prompt drops it, and nothing outside the buffer can go stale.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

/// The compiled-in picture.
pub const DEFAULT_PNG: &[u8] = include_bytes!("resources/buddy.png");

/// The file that replaces [`DEFAULT_PNG`], under the plank home.
pub const OVERRIDE_FILE: &str = "buddy.png";

/// Size of the box the picture is fit into, in cells. The terminal scales the
/// image into it keeping its aspect ratio; with cells about twice as tall as
/// wide this box is roughly square.
pub const COLS: u16 = 20;
/// See [`COLS`].
pub const ROWS: u16 = 10;

/// Columns of bare rule kept to the buddy's right.
const RIGHT_GAP: u16 = 2;

/// The kitty image id. Placeholder cells carry it as their 24-bit foreground
/// color, so it must fit in 24 bits; plank uses no other kitty image.
const IMAGE_ID: u32 = 0x00B0_DD1E;

/// The placeholder character every picture cell holds.
const PLACEHOLDER: char = '\u{10EEEE}';

/// Base64 bytes per transmission chunk, the protocol's maximum.
const CHUNK: usize = 4096;

/// The first entries of kitty's `rowcolumn-diacritics.txt`: the combining mark
/// at index `n` names row (or column) `n` of the picture. Only as many as the
/// larger side of the box are needed.
const DIACRITICS: [char; 20] = [
    '\u{0305}', '\u{030D}', '\u{030E}', '\u{0310}', '\u{0312}', '\u{033D}', '\u{033E}', '\u{033F}',
    '\u{0346}', '\u{034A}', '\u{034B}', '\u{034C}', '\u{0350}', '\u{0351}', '\u{0352}', '\u{0357}',
    '\u{035B}', '\u{0363}', '\u{0364}', '\u{0365}',
];

/// Whether the buddy is on screen (for this run only; nothing is saved).
static SHOWN: AtomicBool = AtomicBool::new(false);

/// Whether the buddy is currently shown.
#[must_use]
pub fn shown() -> bool {
    SHOWN.load(Ordering::Relaxed)
}

/// Whether this process's terminal can show the buddy; see [`supported_by`].
#[must_use]
pub fn supported() -> bool {
    let var = |k| std::env::var(k).ok();
    supported_by(
        var("TERM").as_deref(),
        var("TERM_PROGRAM").as_deref(),
        std::env::var_os("KITTY_WINDOW_ID").is_some(),
        std::env::var_os("TMUX").is_some(),
    )
}

/// Kitty or Ghostty, the terminals that implement Unicode placeholders, and
/// not inside tmux, which would swallow the graphics escapes (and leaks
/// `KITTY_WINDOW_ID` into its panes, so that variable alone proves nothing).
#[must_use]
pub fn supported_by(
    term: Option<&str>,
    term_program: Option<&str>,
    kitty_window: bool,
    tmux: bool,
) -> bool {
    if tmux {
        return false;
    }
    let term = term.unwrap_or_default();
    let program = term_program.unwrap_or_default().to_ascii_lowercase();
    kitty_window
        || term == "xterm-kitty"
        || term == "xterm-ghostty"
        || program == "kitty"
        || program == "ghostty"
}

/// The picture to show: the override file when it holds a PNG, else
/// [`DEFAULT_PNG`]. An override that exists but is not a PNG is an error
/// rather than a silent fallback, so the user learns why theirs is not shown.
fn picture() -> Result<Vec<u8>, String> {
    let path = crate::home::plank_home().join(OVERRIDE_FILE);
    match std::fs::read(&path) {
        Ok(bytes) if is_png(&bytes) => Ok(bytes),
        Ok(_) => Err(format!("{} is not a PNG", path.display())),
        Err(_) => Ok(DEFAULT_PNG.to_vec()),
    }
}

fn is_png(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
}

/// The escapes that hand `png` to the terminal as image [`IMAGE_ID`] with a
/// virtual placement of [`COLS`] x [`ROWS`] cells (`U=1`), split into chunks
/// as the protocol requires. `q=2` keeps the terminal from answering: a reply
/// would arrive on stdin as keystrokes.
#[must_use]
pub fn transmit_sequence(png: &[u8]) -> String {
    let data = crate::tui::base64(png);
    let chunks: Vec<&str> = data
        .as_bytes()
        .chunks(CHUNK)
        .map(|c| std::str::from_utf8(c).unwrap_or_default())
        .collect();
    let mut out = String::with_capacity(data.len() + chunks.len() * 16 + 64);
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        // Writing to a `String` cannot fail.
        let _ = if i == 0 {
            write!(
                out,
                "\x1b_Ga=T,U=1,f=100,i={IMAGE_ID},c={COLS},r={ROWS},q=2,m={more};{chunk}\x1b\\"
            )
        } else {
            write!(out, "\x1b_Gm={more};{chunk}\x1b\\")
        };
    }
    out
}

/// The escape that frees image [`IMAGE_ID`] and its placements.
#[must_use]
pub fn delete_sequence() -> String {
    format!("\x1b_Ga=d,d=I,i={IMAGE_ID},q=2\x1b\\")
}

fn write_to_terminal(seq: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut out = std::io::stdout().lock();
    out.write_all(seq.as_bytes())?;
    out.flush()
}

/// `/useless-buddy [on|off]`: toggles the buddy (or sets it), returning the
/// line to show. Bare, it flips.
pub fn command(arg: &str) -> String {
    let want = match arg.trim() {
        "" => !shown(),
        "on" => true,
        "off" => false,
        other => return format!("/useless-buddy: expected on|off, got `{other}`"),
    };
    if want == shown() {
        return format!("useless buddy already {}", if want { "on" } else { "off" });
    }
    if !want {
        SHOWN.store(false, Ordering::Relaxed);
        let _ = write_to_terminal(&delete_sequence());
        return "useless buddy dismissed".to_owned();
    }
    if !supported() {
        return "/useless-buddy needs a terminal with kitty graphics (kitty or Ghostty, outside tmux)"
            .to_owned();
    }
    let png = match picture() {
        Ok(png) => png,
        Err(e) => return format!("/useless-buddy: {e}"),
    };
    if let Err(e) = write_to_terminal(&transmit_sequence(&png)) {
        return format!("/useless-buddy: {e}");
    }
    SHOWN.store(true, Ordering::Relaxed);
    "useless buddy reporting for duty (it will not help)".to_owned()
}

/// Where the buddy goes in `output`: right-aligned with [`RIGHT_GAP`] columns
/// to spare and standing on its bottom edge, which is the rule above the
/// prompt. `None` when the pane is too small to hold it with room left over.
#[must_use]
pub fn placement(output: Rect) -> Option<Rect> {
    if output.width < COLS + RIGHT_GAP + 2 || output.height < ROWS + 2 {
        return None;
    }
    Some(Rect {
        x: output.right() - RIGHT_GAP - COLS,
        y: output.bottom() - ROWS,
        width: COLS,
        height: ROWS,
    })
}

/// The symbol of the placeholder cell at `row`, `col` of the picture.
fn cell_symbol(row: u16, col: u16) -> String {
    [
        PLACEHOLDER,
        DIACRITICS[usize::from(row)],
        DIACRITICS[usize::from(col)],
    ]
    .iter()
    .collect()
}

/// Paints the placeholder cells into `buf` at [`placement`] in `output`, when
/// the buddy is on. Call it after the output pane is drawn, or the pane would
/// paint over it, and only on frames with a resting prompt, since the buddy
/// stands on the rule above it.
pub fn render(buf: &mut Buffer, output: Rect) {
    if shown() {
        paint(buf, output);
    }
}

fn paint(buf: &mut Buffer, output: Rect) {
    let Some(at) = placement(output) else {
        return;
    };
    let [_, r, g, b] = IMAGE_ID.to_be_bytes();
    for row in 0..at.height {
        for col in 0..at.width {
            if let Some(cell) = buf.cell_mut((at.x + col, at.y + row)) {
                cell.reset();
                cell.set_symbol(&cell_symbol(row, col))
                    .set_fg(Color::Rgb(r, g, b));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_kitty_and_ghostty_outside_tmux_are_supported() {
        assert!(supported_by(Some("xterm-kitty"), None, false, false));
        assert!(supported_by(Some("xterm-ghostty"), None, false, false));
        assert!(supported_by(None, Some("ghostty"), false, false));
        assert!(supported_by(Some("xterm-256color"), None, true, false));
        assert!(!supported_by(
            Some("xterm-256color"),
            Some("WarpTerminal"),
            false,
            false
        ));
        assert!(!supported_by(
            Some("xterm-256color"),
            Some("iTerm.app"),
            false,
            false
        ));
        assert!(!supported_by(Some("tmux-256color"), None, true, true));
        assert!(!supported_by(Some("xterm-kitty"), None, false, true));
    }

    #[test]
    fn the_default_picture_is_a_png() {
        assert!(is_png(DEFAULT_PNG));
        assert!(image::load_from_memory(DEFAULT_PNG).is_ok());
    }

    #[test]
    fn transmission_is_chunked_and_creates_a_virtual_placement() {
        let png = vec![7u8; CHUNK * 2];
        let seq = transmit_sequence(&png);
        let parts: Vec<&str> = seq.split("\x1b\\").filter(|p| !p.is_empty()).collect();
        assert!(parts.len() > 2, "{} chunks", parts.len());
        assert!(parts[0].starts_with(&format!(
            "\x1b_Ga=T,U=1,f=100,i={IMAGE_ID},c={COLS},r={ROWS},q=2,m=1;"
        )));
        for mid in &parts[1..parts.len() - 1] {
            assert!(mid.starts_with("\x1b_Gm=1;"));
        }
        assert!(parts.last().unwrap().starts_with("\x1b_Gm=0;"));
        let payload: String = parts.iter().map(|p| p.split_once(';').unwrap().1).collect();
        assert_eq!(payload, crate::tui::base64(&png));
        for part in &parts {
            assert!(part.split_once(';').unwrap().1.len() <= CHUNK);
        }
    }

    #[test]
    fn a_small_picture_goes_in_one_final_chunk() {
        let seq = transmit_sequence(b"\x89PNG\r\n\x1a\n");
        assert_eq!(seq.matches("\x1b_G").count(), 1);
        assert!(seq.contains(",m=0;"));
    }

    #[test]
    fn the_id_fits_the_24_bit_foreground() {
        const { assert!(IMAGE_ID < 1 << 24) };
        assert!(DIACRITICS.len() >= usize::from(COLS.max(ROWS)));
    }

    #[test]
    fn placement_stands_on_the_bottom_edge_right_aligned() {
        let output = Rect::new(0, 0, 80, 30);
        let at = placement(output).unwrap();
        assert_eq!(at.bottom(), output.bottom());
        assert_eq!(at.right(), output.right() - RIGHT_GAP);
        assert_eq!((at.width, at.height), (COLS, ROWS));
        assert!(placement(Rect::new(0, 0, 80, ROWS + 1)).is_none());
        assert!(placement(Rect::new(0, 0, COLS + 3, 30)).is_none());
    }

    #[test]
    fn painted_cells_are_one_column_placeholders_naming_row_and_column() {
        let output = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(output);
        paint(&mut buf, output);
        let at = placement(output).unwrap();
        let [_, r, g, b] = IMAGE_ID.to_be_bytes();
        for (row, col) in [(0, 0), (3, 7), (ROWS - 1, COLS - 1)] {
            let cell = &buf[(at.x + col, at.y + row)];
            let chars: Vec<char> = cell.symbol().chars().collect();
            assert_eq!(
                chars,
                [
                    PLACEHOLDER,
                    DIACRITICS[usize::from(row)],
                    DIACRITICS[usize::from(col)]
                ]
            );
            assert_eq!(cell.fg, Color::Rgb(r, g, b));
            assert_eq!(unicode_width::UnicodeWidthStr::width(cell.symbol()), 1);
        }
        // Nothing outside the box is touched.
        assert_eq!(buf[(at.x - 1, at.y)].symbol(), " ");
        assert_eq!(buf[(at.x, at.y - 1)].symbol(), " ");
    }
}

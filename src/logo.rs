// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Startup banner: the plank logo, rendered from `resources/logo.png`.
//!
//! Uses the `logo-art` crate to turn the PNG into true-color ANSI art. The
//! near-white background is keyed to transparent first so the terminal shows
//! through. The TUI converts the art into styled lines; the plain/piped path
//! prints the ANSI directly.
//!
//! The banner is rendered with the finest cell subdivision the terminal is
//! known to draw ([`logo_art::Cell::best_supported`]) rather than with
//! half-blocks, which is what lets [`DEFAULT_WIDTH`] be half what it used to
//! be: an octant cell packs 2x4 samples instead of 1x2, so the logo keeps its
//! old pixel grid in a quarter of the screen area. Set `PLANK_LOGO_CELL` to
//! `half`, `quad`, `sextant`, `octant` or `braille` to override the detection
//! — a terminal that leaves the newer block glyphs to the font will draw tofu,
//! and `half` is the universally safe answer.
//!
//! A terminal with kitty graphics and Unicode placeholders (kitty, Ghostty;
//! the same detection as `/buddy`) gets the picture itself instead: the PNG
//! is sent once as a kitty image and the banner holds placeholder cells in
//! a box of their own, so it scrolls with the text like the art does. The
//! box is narrower than the art's (`KITTY_COLS`), since a real image needs
//! fewer cells. Setting `PLANK_LOGO_CELL` keeps the art.
//!
//! Warp draws kitty images only at the cursor, outside the cell grid, so
//! there the banner holds blank marker cells instead, each naming its own
//! column and row by its colour ([`direct_art`]). After every frame
//! [`after_draw`] looks for a whole, uncovered block of them and places the
//! picture over it, moving it as the output scrolls and deleting it when the
//! block is partly off screen or anything is drawn over it.

use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;

/// The logo image, embedded at build time.
pub const LOGO_PNG: &[u8] = include_bytes!("resources/logo.png");

/// Default render width, in terminal columns.
pub const DEFAULT_WIDTH: u32 = 18;

/// Environment variable that overrides the detected cell subdivision.
pub const CELL_ENV: &str = "PLANK_LOGO_CELL";

/// Pixels with every channel at or above this level are treated as background.
const BACKGROUND_THRESHOLD: u8 = 232;

/// The kitty image id the banner logo is sent as. Placeholder cells name
/// their image by their 24-bit foreground colour, so it fits in 24 bits, and
/// it differs from the buddy's clip ids.
pub const KITTY_ID: u32 = 0x00B0_DD30;

/// The second id the Warp picture alternates with [`KITTY_ID`], so a move
/// places the new copy before deleting the old and nothing blinks.
const KITTY_ID_ALT: u32 = KITTY_ID + 1;

/// Red channel of a Warp marker cell's colour; green and blue are the cell's
/// column and row within the picture ([`direct_art`]).
const DIRECT_TAG: u8 = 0xB0;

/// Width, in columns, of the kitty picture: smaller than the art's
/// [`DEFAULT_WIDTH`], since a real image needs fewer cells to read clearly.
pub const KITTY_COLS: u32 = 10;

/// The logo for one banner: what to draw in the text, and the escapes that
/// must reach the terminal before it for a kitty picture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Logo {
    /// Escapes transmitting the picture (`None` for ANSI art).
    pub transmit: Option<String>,
    /// The banner text: true-colour art, or placeholder cells.
    pub art: String,
}

/// How the banner can show the picture: the `/buddy` detection, on a
/// terminal stdout, unless `PLANK_LOGO_CELL` asks for art.
fn kitty_mode() -> Option<crate::buddy::Mode> {
    use std::io::IsTerminal as _;
    if std::env::var_os(CELL_ENV).is_some() || !std::io::stdout().is_terminal() {
        return None;
    }
    crate::buddy::supported()
}

/// A terminal cell's size in pixels `(width, height)`, from `TIOCGWINSZ`;
/// `None` when the terminal does not report its pixel size.
fn cell_pixels() -> Option<(u32, u32)> {
    let (cols, rows) = crate::platform::stdout_size()?;
    let (width, height) = crate::platform::stdout_pixels()?;
    Some((
        u32::from(width) / u32::from(cols),
        u32::from(height) / u32::from(rows),
    ))
}

/// The box for an `image` of `(width, height)` pixels drawn `cols` cells wide
/// at native resolution, given the `cell` size in pixels: the pixel size to
/// scale it to, `(width, height)`, and the rows it covers. The width fills
/// the columns exactly and the height keeps the image's proportions, so the
/// terminal draws it pixel for pixel and never rescales it.
fn kitty_box(image: (u32, u32), cell: (u32, u32), cols: u32) -> Option<((u32, u32), u32)> {
    let ((w, h), (cw, ch)) = (image, cell);
    if w == 0 || h == 0 || cw == 0 || ch == 0 || cols == 0 {
        return None;
    }
    let width = cols * cw;
    let height = (u64::from(width) * u64::from(h) / u64::from(w)).max(1);
    let height = u32::try_from(height).ok()?;
    Some(((width, height), height.div_ceil(ch)))
}

/// `png` made ready to send at exactly `size` pixels: decoded, its
/// background keyed out when `key`, and scaled nearest-neighbour, which
/// keeps a pixel-art logo's edges hard. `None` when it does not decode.
fn kitty_png(png: &[u8], key: bool, size: (u32, u32)) -> Option<Vec<u8>> {
    let mut rgba = image::load_from_memory(png).ok()?.to_rgba8();
    if key {
        // Only the white the border reaches, as for the buddy: a threshold
        // alone would also punch out the eye highlights.
        rgba = crate::buddy::key_background(rgba);
    }
    let scaled =
        image::imageops::resize(&rgba, size.0, size.1, image::imageops::FilterType::Nearest);
    let mut out = Vec::new();
    image::DynamicImage::ImageRgba8(scaled)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .ok()?;
    Some(out)
}

/// The placeholder text for a `cols` x `rows` picture of image [`KITTY_ID`]:
/// one line per row, each cell naming its row and column by diacritics, in
/// the image id's colour. `None` past the diacritics plank carries.
fn placeholder_art(cols: usize, rows: usize) -> Option<String> {
    use crate::buddy::{DIACRITICS, PLACEHOLDER};
    if cols == 0 || rows == 0 || cols > DIACRITICS.len() || rows > DIACRITICS.len() {
        return None;
    }
    let [_, r, g, b] = KITTY_ID.to_be_bytes();
    let colour = format!("\x1b[38;2;{r};{g};{b}m");
    let mut out = String::new();
    for &row in &DIACRITICS[..rows] {
        out.push_str(&colour);
        for &col in &DIACRITICS[..cols] {
            out.extend([PLACEHOLDER, row, col]);
        }
        out.push_str("\x1b[39m\n");
    }
    Some(out)
}

/// The art `ansi` draws, swapped for a kitty picture of `png` when the
/// terminal can show one; the art otherwise, or if anything fails.
///
/// The picture is `cols` wide and drawn at native resolution: it is scaled
/// to exactly the pixels its cells cover ([`kitty_box`]), so the terminal
/// shows it pixel for pixel.
fn with_kitty(ansi: impl FnOnce() -> String, png: &[u8], key: bool, cols: u32) -> Logo {
    let picture = |mode: crate::buddy::Mode| {
        let cell = cell_pixels()?;
        let size = image::ImageReader::new(std::io::Cursor::new(png))
            .with_guessed_format()
            .ok()?
            .into_dimensions()
            .ok()?;
        let (pixels, rows) = kitty_box(size, cell, cols)?;
        if mode == crate::buddy::Mode::Direct {
            let (c, r) = (u16::try_from(cols).ok()?, u16::try_from(rows).ok()?);
            let text = direct_art(c, r)?;
            let scaled = kitty_png(png, key, pixels)?;
            let mut direct = DIRECT.lock().ok()?;
            direct.png = Some(scaled);
            direct.size = (c, r);
            return Some(Logo {
                transmit: None,
                art: text,
            });
        }
        let text = placeholder_art(usize::try_from(cols).ok()?, usize::try_from(rows).ok()?)?;
        let scaled = kitty_png(png, key, pixels)?;
        let (c, r) = (u16::try_from(cols).ok()?, u16::try_from(rows).ok()?);
        Some(Logo {
            transmit: Some(crate::buddy::transmit_sequence_sized(
                KITTY_ID, &scaled, c, r,
            )),
            art: text,
        })
    };
    if let Some(mode) = kitty_mode()
        && let Some(logo) = picture(mode)
    {
        return logo;
    }
    Logo {
        transmit: None,
        art: ansi(),
    }
}

/// The marker text for a `cols` x `rows` Warp picture: blank
/// [`crate::buddy::MARKER`] cells, each coloured with [`DIRECT_TAG`] and its
/// own column and row, so [`after_draw`] can tell a whole block from one
/// scrolled partly away or drawn over. `None` past 256 cells a side.
fn direct_art(cols: u16, rows: u16) -> Option<String> {
    if cols == 0 || rows == 0 || cols > 256 || rows > 256 {
        return None;
    }
    let mut out = String::new();
    for row in 0..rows {
        for col in 0..cols {
            // Writing to a `String` cannot fail.
            let _ = write!(
                out,
                "\x1b[38;2;{DIRECT_TAG};{col};{row}m{}",
                crate::buddy::MARKER
            );
        }
        out.push_str("\x1b[39m\n");
    }
    Some(out)
}

/// The Warp picture: what to place and where it is now.
struct Direct {
    /// The picture, scaled to its box; `None` until a banner used one.
    png: Option<Vec<u8>>,
    /// The box, in cells: `(cols, rows)`.
    size: (u16, u16),
    /// Where it is on screen, and under which id.
    placed: Option<(Rect, u32)>,
}

static DIRECT: Mutex<Direct> = Mutex::new(Direct {
    png: None,
    size: (0, 0),
    placed: None,
});

/// The first whole block of Warp marker cells `cols` x `rows` in `buf`: a
/// cell naming column 0, row 0 with every other cell of the box in place.
fn find_block(buf: &Buffer, (cols, rows): (u16, u16)) -> Option<Rect> {
    let is_marker = |x: u16, y: u16, col: u16, row: u16| {
        let (Ok(col), Ok(row)) = (u8::try_from(col), u8::try_from(row)) else {
            return false;
        };
        buf.cell((x, y)).is_some_and(|c| {
            c.symbol() == crate::buddy::MARKER && c.fg == Color::Rgb(DIRECT_TAG, col, row)
        })
    };
    let area = buf.area;
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if !is_marker(x, y, 0, 0) {
                continue;
            }
            let whole = (0..rows).all(|r| {
                (0..cols).all(|c| {
                    x.checked_add(c)
                        .zip(y.checked_add(r))
                        .is_some_and(|(cx, cy)| is_marker(cx, cy, c, r))
                })
            });
            if whole {
                return Some(Rect::new(x, y, cols, rows));
            }
        }
    }
    None
}

/// For Warp: called with every frame drawn, places, moves or removes the
/// banner picture to match its marker cells. A no-op until a banner used the
/// picture, which only happens in Warp.
pub fn after_draw(buf: &Buffer) {
    let Ok(mut direct) = DIRECT.lock() else {
        return;
    };
    let Some(png) = direct.png.as_ref() else {
        return;
    };
    let want = find_block(buf, direct.size);
    if want == direct.placed.map(|(at, _)| at) {
        return;
    }
    let mut seq = String::new();
    let mut placed = None;
    if let Some(at) = want {
        let id = match direct.placed {
            Some((_, KITTY_ID)) => KITTY_ID_ALT,
            _ => KITTY_ID,
        };
        seq.push_str(&crate::buddy::place_sequence(id, at, png));
        placed = Some((at, id));
    }
    // The old copy goes only once the new one is up.
    if let Some((_, old)) = direct.placed {
        seq.push_str(&crate::buddy::delete_sequence(old));
    }
    direct.placed = placed;
    let _ = crate::buddy::write_to_terminal(&seq);
}

/// Removes the Warp picture as plank exits: it sits outside the cell grid, so
/// leaving the alternate screen would not take it away. `try_lock`, as for
/// the buddy: a force quit must not hang on it.
pub fn hide_for_exit() {
    if let Ok(mut direct) = DIRECT.try_lock()
        && let Some((_, id)) = direct.placed.take()
    {
        let _ = crate::buddy::write_to_terminal(&crate::buddy::delete_sequence(id));
    }
}

/// The logo PNG with its near-white background made transparent (computed once).
fn transparent_png() -> &'static [u8] {
    static CACHE: OnceLock<Vec<u8>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let Ok(img) = image::load_from_memory(LOGO_PNG) else {
            return LOGO_PNG.to_vec();
        };
        let mut rgba = img.to_rgba8();
        for px in rgba.pixels_mut() {
            let [r, g, b, _] = px.0;
            if r >= BACKGROUND_THRESHOLD && g >= BACKGROUND_THRESHOLD && b >= BACKGROUND_THRESHOLD {
                px.0[3] = 0;
            }
        }
        let mut out = Vec::new();
        if image::DynamicImage::ImageRgba8(rgba)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .is_err()
        {
            return LOGO_PNG.to_vec();
        }
        out
    })
}

/// How finely to subdivide a terminal cell: `PLANK_LOGO_CELL` if it names a
/// mode, otherwise whatever the terminal is known to draw (computed once).
#[must_use]
pub fn cell() -> logo_art::Cell {
    static CACHE: OnceLock<logo_art::Cell> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var(CELL_ENV)
            .ok()
            .as_deref()
            .and_then(parse_cell)
            .unwrap_or_else(logo_art::Cell::best_supported)
    })
}

/// Parses a `PLANK_LOGO_CELL` value; `None` for anything unrecognized, so a
/// typo falls back to detection instead of failing the banner.
fn parse_cell(name: &str) -> Option<logo_art::Cell> {
    match name.trim().to_ascii_lowercase().as_str() {
        "half" | "halfblock" | "half-block" => Some(logo_art::Cell::HalfBlock),
        "quad" | "quadrant" => Some(logo_art::Cell::Quadrant),
        "sextant" => Some(logo_art::Cell::Sextant),
        "octant" => Some(logo_art::Cell::Octant),
        "braille" => Some(logo_art::Cell::Braille),
        _ => None,
    }
}

/// Renders the logo as true-color ANSI art `width` columns wide.
#[must_use]
pub fn art(width: u32) -> String {
    logo_art::image_to_ansi_with(transparent_png(), width.max(1), cell())
}

/// Renders the logo at `path`, falling back to plank's own art.
///
/// A profile's identity may fail to be pretty but must not fail to launch: an
/// absent, unreadable or undecodable PNG degrades to the built-in logo. The
/// warning is printed once by the caller, not here, so this stays pure enough
/// to test.
#[must_use]
pub fn art_from_path(path: Option<&std::path::Path>, width: u32) -> String {
    let Some(path) = path else {
        return art(width);
    };
    let Ok(bytes) = std::fs::read(path) else {
        return art(width);
    };
    if image::load_from_memory(&bytes).is_err() {
        return art(width);
    }
    logo_art::image_to_ansi_with(&bytes, width.max(1), cell())
}

/// Version label like `v2.5.0`, with ` BETA` appended for beta builds.
///
/// Channel-by-patch scheme (see VERSIONING.md): a `X.Y.0` version is a stable
/// release, any patch above 0 is a beta. A `beta` pre-release identifier or a
/// compile-time `PLANK_CHANNEL=beta` still forces the label for odd builds.
#[must_use]
pub fn version_label() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let forced = option_env!("PLANK_CHANNEL").is_some_and(|c| c.eq_ignore_ascii_case("beta"));
    if is_beta(version, env!("CARGO_PKG_VERSION_PATCH")) || forced {
        format!("v{version} BETA")
    } else {
        format!("v{version}")
    }
}

/// The git commit plank was built from: a short hash, `-dirty` when the build
/// had uncommitted changes, or `unknown` when the source tree had no git.
#[must_use]
pub fn commit_id() -> &'static str {
    env!("PLANK_GIT_COMMIT")
}

/// One-line `--version` output: `plank v2.5.0 (abc123def456)`.
#[must_use]
pub fn version_line() -> String {
    format!("plank {} ({})", version_label(), commit_id())
}

fn is_beta(version: &str, patch: &str) -> bool {
    patch != "0" || version.contains("beta")
}

/// The banner: the active profile's logo, name and plank's version when a
/// profile is running, else plank's own art at [`DEFAULT_WIDTH`] and its
/// version line alone.
///
/// A profiled build still needs its plank version visible for bug reports,
/// so the label carries both rather than the display name replacing the
/// version outright.
#[must_use]
pub fn banner() -> String {
    let title = crate::profile::active().map(|_| crate::profile::title());
    let label = banner_label(title.as_deref());
    let logo = active_logo(DEFAULT_WIDTH);
    format!(
        "{}{}      {label}\n",
        logo.transmit.unwrap_or_default(),
        logo.art
    )
}

/// The logo for this run, `width` columns wide: the active profile's `logo`
/// when a profile is running, else plank's own, as a kitty picture where the
/// terminal shows one and as art elsewhere. A profile's `logoScale`
/// multiplies both the art's `width` and the picture's [`KITTY_COLS`].
///
/// The one place both banners, plain and TUI, choose their logo, so a profile
/// cannot show its logo on one front end and plank's on the other.
#[must_use]
pub fn active_logo(width: u32) -> Logo {
    let spec = crate::profile::active().map(|a| &a.spec);
    let profile_png = spec
        .and_then(|s| s.logo.as_deref().map(std::fs::read))
        .and_then(Result::ok)
        .filter(|bytes| image::load_from_memory(bytes).is_ok());
    match profile_png {
        Some(bytes) => {
            let scale = spec.map_or(1, |s| s.logo_scale.max(1));
            with_kitty(
                || logo_art::image_to_ansi_with(&bytes, (width * scale).max(1), cell()),
                &bytes,
                false,
                KITTY_COLS * scale,
            )
        }
        None => with_kitty(|| art(width), LOGO_PNG, true, KITTY_COLS),
    }
}

/// The banner's text label: the version alone for plain plank, or the
/// profile's display name followed by the version when `profile_name` is
/// `Some`. Split out from [`banner`] so the name+version composition can be
/// tested without touching the process-global active profile.
fn banner_label(profile_name: Option<&str>) -> String {
    match profile_name {
        Some(name) => format!("{name}  {}", version_label()),
        None => version_label(),
    }
}

#[cfg(test)]
mod tests {
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    /// A buffer holding `art` (via the TUI's ANSI conversion) at `(x, y)`.
    fn buffer_with(art: &str, area: Rect, x: u16, y: u16) -> Buffer {
        let mut buf = Buffer::empty(area);
        for (i, line) in crate::tui::ansi_to_lines(art).iter().enumerate() {
            let row = y + u16::try_from(i).unwrap();
            if row < area.bottom() {
                buf.set_line(x, row, line, area.width - x);
            }
        }
        buf
    }

    #[test]
    fn a_whole_warp_block_is_found_where_it_stands() {
        let art = super::direct_art(4, 3).unwrap();
        let area = Rect::new(0, 0, 20, 10);
        let buf = buffer_with(&art, area, 5, 2);
        assert_eq!(super::find_block(&buf, (4, 3)), Some(Rect::new(5, 2, 4, 3)));
    }

    #[test]
    fn a_warp_block_scrolled_partly_away_or_covered_is_not_placed() {
        let art = super::direct_art(4, 3).unwrap();
        let area = Rect::new(0, 0, 20, 10);
        // Its top row scrolled off: the first visible row is row 1.
        let tail = art.lines().skip(1).fold(String::new(), |mut out, l| {
            out.push_str(l);
            out.push('\n');
            out
        });
        let buf = buffer_with(&tail, area, 5, 0);
        assert_eq!(super::find_block(&buf, (4, 3)), None);
        // Its bottom row cut off by the edge of the area.
        let buf = buffer_with(&art, area, 5, 8);
        assert_eq!(super::find_block(&buf, (4, 3)), None);
        // One cell drawn over.
        let mut buf = buffer_with(&art, area, 5, 2);
        buf[(7, 3)].set_symbol("x");
        assert_eq!(super::find_block(&buf, (4, 3)), None);
    }

    // The commit stamp always exists (build.rs falls back to "unknown"), and
    // the line carries both the version and the commit.
    #[test]
    fn version_line_has_version_and_commit() {
        let line = super::version_line();
        assert!(line.contains(env!("CARGO_PKG_VERSION")), "{line}");
        assert!(!super::commit_id().is_empty());
        assert!(line.contains(super::commit_id()), "{line}");
    }

    #[test]
    fn art_renders_ansi() {
        let art = super::art(24);
        // True-color cells carry SGR escapes and newlines.
        assert!(art.contains('\x1b'));
        assert!(art.contains('\n'));
    }

    // The banner has to fit its own width: one line per row, `width` cells each.
    #[test]
    fn art_is_as_wide_as_asked() {
        let art = super::art(super::DEFAULT_WIDTH);
        let mut rows = 0;
        for line in art.lines() {
            let mut cells = 0;
            let mut chars = line.chars();
            while let Some(c) = chars.next() {
                if c == '\x1b' {
                    for c in chars.by_ref() {
                        if c == 'm' {
                            break;
                        }
                    }
                } else {
                    cells += 1;
                }
            }
            assert_eq!(cells, super::DEFAULT_WIDTH, "row {rows}");
            rows += 1;
        }
        // 711x1075 source, so the aspect ratio gives 14 rows at 18 columns.
        assert_eq!(rows, 14);
    }

    #[test]
    fn placeholder_art_names_every_cell_in_the_logo_colour() {
        use crate::buddy::{DIACRITICS, PLACEHOLDER};
        let art = super::placeholder_art(3, 2).unwrap();
        let lines: Vec<&str> = art.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("\x1b[38;2;176;221;48m"), "{art:?}");
        let cell: String = [PLACEHOLDER, DIACRITICS[1], DIACRITICS[2]].iter().collect();
        assert!(lines[1].contains(&cell));
        // Each cell is one column wide, so the box is the art's own.
        assert_eq!(unicode_width::UnicodeWidthStr::width(cell.as_str()), 1);
        assert_eq!(super::placeholder_art(25, 2), None);
        assert_eq!(super::placeholder_art(0, 2), None);
    }

    #[test]
    fn the_sent_logo_is_scaled_to_its_cells_and_keyed() {
        let png = super::kitty_png(super::LOGO_PNG, true, (200, 302)).unwrap();
        let img = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(img.dimensions(), (200, 302));
        assert_eq!(img.get_pixel(0, 0).0[3], 0, "the white corner is keyed out");
        // ...but not the white inside the logo (the eye highlights).
        let opaque_white = img
            .pixels()
            .filter(|p| p.0[3] == 255 && p.0[..3].iter().all(|&c| c >= 232))
            .count();
        assert!(opaque_white > 0, "the eye highlights must stay");
    }

    #[test]
    fn the_kitty_box_fills_its_columns_at_native_resolution() {
        // The 711x1075 logo, 10 columns of 20x40-pixel cells: 200 pixels
        // wide, 302 tall, which takes 8 rows (the last one partly).
        assert_eq!(
            super::kitty_box((711, 1075), (20, 40), 10),
            Some(((200, 302), 8))
        );
        assert_eq!(super::kitty_box((711, 1075), (0, 40), 10), None);
        assert_eq!(super::kitty_box((0, 1075), (20, 40), 10), None);
    }

    #[test]
    fn cell_override_names_every_mode() {
        use logo_art::Cell;
        assert_eq!(super::parse_cell("half"), Some(Cell::HalfBlock));
        assert_eq!(super::parse_cell(" Quadrant "), Some(Cell::Quadrant));
        assert_eq!(super::parse_cell("SEXTANT"), Some(Cell::Sextant));
        assert_eq!(super::parse_cell("octant"), Some(Cell::Octant));
        assert_eq!(super::parse_cell("braille"), Some(Cell::Braille));
        // A typo falls back to detection rather than breaking the banner.
        assert_eq!(super::parse_cell("octopus"), None);
        assert_eq!(super::parse_cell(""), None);
    }

    #[test]
    fn banner_has_version() {
        assert!(super::banner().contains(env!("CARGO_PKG_VERSION")));
    }

    // The no-profile banner label must stay exactly the version line: a
    // regression here would change the byte-identical no-profile banner.
    #[test]
    fn banner_label_without_a_profile_is_just_the_version() {
        assert_eq!(super::banner_label(None), super::version_label());
    }

    // A profiled banner must carry both the display name and the version,
    // not one instead of the other: version_label() alone is the thing that
    // must never be silently dropped from a profiled build's banner.
    #[test]
    fn banner_label_with_a_profile_has_both_name_and_version() {
        let label = super::banner_label(Some("HAL"));
        assert!(label.contains("HAL"), "{label}");
        assert!(label.contains(&super::version_label()), "{label}");
    }

    // Channel-by-patch: X.Y.0 is stable, any higher patch is a beta build.
    #[test]
    fn beta_follows_patch_number() {
        assert!(!super::is_beta("2.5.0", "0"));
        assert!(super::is_beta("2.5.1", "1"));
        assert!(super::is_beta("2.5.12", "12"));
        assert!(super::is_beta("3.0.0-beta.1", "0"));
    }

    #[test]
    fn version_label_starts_with_v_and_version() {
        let label = super::version_label();
        assert!(label.starts_with('v'));
        assert!(label.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn an_unreadable_profile_logo_falls_back_to_the_plank_art() {
        let missing = std::path::Path::new("/nonexistent/hal.png");
        let art = super::art_from_path(Some(missing), super::DEFAULT_WIDTH);
        assert_eq!(
            art,
            super::art(super::DEFAULT_WIDTH),
            "a missing PNG must not change the banner"
        );
    }

    #[test]
    fn a_corrupt_profile_logo_falls_back_to_the_plank_art() {
        let dir = std::env::temp_dir().join(format!("plank-logo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("bad.png");
        std::fs::write(&path, b"not a png").expect("write");
        assert_eq!(
            super::art_from_path(Some(&path), super::DEFAULT_WIDTH),
            super::art(super::DEFAULT_WIDTH)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_profile_logo_is_the_plank_art() {
        assert_eq!(
            super::art_from_path(None, super::DEFAULT_WIDTH),
            super::art(super::DEFAULT_WIDTH)
        );
    }
}

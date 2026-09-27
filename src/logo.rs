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

use std::sync::OnceLock;

/// The logo image, embedded at build time.
pub const LOGO_PNG: &[u8] = include_bytes!("resources/logo.png");

/// Default render width, in terminal columns.
pub const DEFAULT_WIDTH: u32 = 18;

/// Environment variable that overrides the detected cell subdivision.
pub const CELL_ENV: &str = "PLANK_LOGO_CELL";

/// Pixels with every channel at or above this level are treated as background.
const BACKGROUND_THRESHOLD: u8 = 232;

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
    format!("{}      {label}\n", active_art(DEFAULT_WIDTH))
}

/// The logo for this run, `width` columns wide: the active profile's `logo`
/// when a profile is running, else plank's own art.
///
/// The one place both banners, plain and TUI, choose their art, so a profile
/// cannot show its logo on one front end and plank's on the other.
#[must_use]
pub fn active_art(width: u32) -> String {
    match crate::profile::active() {
        Some(a) => art_from_path(a.spec.logo.as_deref(), width),
        None => art(width),
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

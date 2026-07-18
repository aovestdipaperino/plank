//! Startup banner: the plank logo, rendered from `resources/logo.png`.
//!
//! Uses the `logo-art` crate to turn the PNG into true-color half-block ANSI
//! art. The near-white background is keyed to transparent first so the
//! terminal shows through. The TUI converts the art into styled lines; the
//! plain/piped path prints the ANSI directly.

use std::sync::OnceLock;

/// The logo image, embedded at build time.
pub const LOGO_PNG: &[u8] = include_bytes!("resources/logo.png");

/// Default render width, in terminal columns.
pub const DEFAULT_WIDTH: u32 = 36;

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

/// Renders the logo as true-color ANSI art `width` columns wide.
#[must_use]
pub fn art(width: u32) -> String {
    logo_art::image_to_ansi(transparent_png(), width.max(1))
}

/// Version label like `v0.9.9`, with ` BETA` appended for beta builds.
///
/// A build is beta when the version carries a `beta` pre-release identifier or
/// the release workflow sets `PLANK_CHANNEL=beta` at compile time.
#[must_use]
pub fn version_label() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let beta = version.contains("beta")
        || option_env!("PLANK_CHANNEL").is_some_and(|c| c.eq_ignore_ascii_case("beta"));
    if beta {
        format!("v{version} BETA")
    } else {
        format!("v{version}")
    }
}

/// The logo art at [`DEFAULT_WIDTH`] followed by a version line.
#[must_use]
pub fn banner() -> String {
    format!("{}      {}\n", art(DEFAULT_WIDTH), version_label())
}

#[cfg(test)]
mod tests {
    #[test]
    fn art_renders_ansi() {
        let art = super::art(24);
        // True-color half-block cells carry SGR escapes and newlines.
        assert!(art.contains('\x1b'));
        assert!(art.contains('\n'));
    }

    #[test]
    fn banner_has_version() {
        assert!(super::banner().contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn version_label_starts_with_v_and_version() {
        let label = super::version_label();
        assert!(label.starts_with('v'));
        assert!(label.contains(env!("CARGO_PKG_VERSION")));
    }
}

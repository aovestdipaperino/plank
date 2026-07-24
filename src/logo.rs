// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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

fn is_beta(version: &str, patch: &str) -> bool {
    patch != "0" || version.contains("beta")
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
}

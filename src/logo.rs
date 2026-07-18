//! Startup banner: the plank logo, rendered from `resources/logo.png`.
//!
//! Uses the `logo-art` crate to turn the PNG into true-color half-block ANSI
//! art. The TUI converts that art into styled lines; the plain/piped path
//! prints the ANSI directly.

/// The logo image, embedded at build time.
pub const LOGO_PNG: &[u8] = include_bytes!("resources/logo.png");

/// Default render width, in terminal columns.
pub const DEFAULT_WIDTH: u32 = 36;

/// Renders the logo as true-color ANSI art `width` columns wide.
#[must_use]
pub fn art(width: u32) -> String {
    logo_art::image_to_ansi(LOGO_PNG, width.max(1))
}

/// The logo art at [`DEFAULT_WIDTH`] followed by a version line.
#[must_use]
pub fn banner() -> String {
    format!(
        "{}      v{}\n",
        art(DEFAULT_WIDTH),
        env!("CARGO_PKG_VERSION")
    )
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
}

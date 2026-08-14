// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The plank arcade, as one WASM component.
//!
//! Every face is in this one module. The design doc originally called for a
//! module per game sharing a support crate; one artifact is better on every
//! axis that matters here — one build, one id in the trust store, one load,
//! one place a user approves — and with a single module there is nothing to
//! share *across*, so `Rng` and the colour helpers are simply internal.
//!
//! Which face opens is [`frame_open`]'s `arg`. Each face also gets its own
//! slash command, which is what `command_run` replying `{"open": "<face>"}`
//! is for: the component asks the host to open its own frame at that face.

mod breakout;
mod matrix;
mod shared;
mod starfield;
mod support;

use extism_pdk::*;

/// The ABI this component speaks. Checked against the host's at load.
#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

/// Which face is open, and its state.
///
/// A guest is single-threaded and the host serializes calls into one plugin,
/// so a static is the whole story. It is `None` until `frame_open`, which is
/// also what makes a `frame_step` before an open a no-op rather than a trap.
static mut FACE: Option<Face> = None;

/// The faces this component offers.
enum Face {
    /// Falling glyphs.
    Matrix(matrix::Rain),
    /// A perspective starfield.
    Starfield(starfield::Starfield),
    /// Bricks, a paddle and a ball — the first *game* to land, as opposed to
    /// a face that only has to look right.
    Breakout(breakout::Breakout),
}

impl Face {
    /// Builds a face by name. Unknown names fall back to the rain rather than
    /// failing: the host has already decided to open *something*, and a black
    /// screen would be a worse answer than the default face.
    fn new(name: &str, seed: u64) -> Self {
        match name {
            // The star count the built-in arcade opens its field with; the
            // port is only faithful if it is dealt the same number of stars.
            "starfield" | "stars" => Self::Starfield(starfield::Starfield::new(seed, 220)),
            "breakout" | "bricks" => Self::Breakout(breakout::Breakout::new(seed)),
            // Every face lands here as it is ported. Until then an unknown
            // name is the rain rather than a failure: the host has already
            // decided to open *something*, and a black screen is a worse
            // answer than the default face.
            _ => Self::Matrix(matrix::Rain::new(seed)),
        }
    }

    fn step(&mut self, dt_ms: u64) {
        match self {
            Self::Matrix(r) => r.step(dt_ms),
            Self::Starfield(s) => s.step(dt_ms),
            Self::Breakout(g) => g.step(dt_ms),
        }
    }

    fn glyphs(&self, w: u16, h: u16) -> Vec<support::Glyph> {
        match self {
            Self::Matrix(r) => r.glyphs(w, h),
            Self::Starfield(s) => s.glyphs(w, h),
            Self::Breakout(g) => g.glyphs(w, h),
        }
    }
}

/// Opens a face. `arg` names it; `seed` makes it reproducible.
#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    let seed = support::int(&input, "seed");
    let face = support::text(&input, "arg");
    unsafe {
        FACE = Some(Face::new(&face, seed));
    }
    Ok(r#"{"ok": true}"#.to_string())
}

/// Advances the open face and returns what it paints.
#[plugin_fn]
pub fn frame_step(input: String) -> FnResult<Vec<u8>> {
    let dt_ms = support::int(&input, "dt_ms");
    let w = support::num(&input, "w") as u16;
    let h = support::num(&input, "h") as u16;
    let face = unsafe { (*(&raw mut FACE)).as_mut() };
    let Some(face) = face else {
        // Stepped before opening: paint nothing rather than trap. The host
        // treats an undecodable buffer as fatal to the frame, and an empty
        // one is both decodable and honest.
        return Ok(support::encode(&[], w, h));
    };
    face.step(dt_ms);
    Ok(support::encode(&face.glyphs(w, h), w, h))
}

/// The face's controls, mirroring what the built-in arcade binds today: the
/// rain speeds up and slows down, `c` re-letters it (a font without katakana
/// draws boxes, and this is the way out), and Esc or `q` closes.
///
/// A key the face does not use closes it. That is the screensaver rule — a
/// person coming back should not have to find the right key — and it costs a
/// deliberate face nothing, since a face with real controls simply claims the
/// keys it wants.
#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    let code = support::text(&input, "code");
    let face = unsafe { (*(&raw mut FACE)).as_mut() };
    let handled = match (face, code.as_str()) {
        (Some(Face::Matrix(rain)), "up" | "k") => {
            rain.scale_speed(1.3);
            true
        }
        (Some(Face::Matrix(rain)), "down" | "j") => {
            rain.scale_speed(1.0 / 1.3);
            true
        }
        (Some(Face::Matrix(rain)), "c") => {
            rain.cycle_charset();
            true
        }
        (Some(Face::Starfield(sky)), "up" | "k") => {
            sky.scale_speed(1.3);
            true
        }
        (Some(Face::Starfield(sky)), "down" | "j") => {
            sky.scale_speed(1.0 / 1.3);
            true
        }
        // A game claims its own keys and says so; anything it does not claim
        // falls through to the close below, which is how Esc and q work
        // without every game having to spell them.
        (Some(Face::Breakout(game)), code) => game.handle_key(code),
        _ => false,
    };
    if handled {
        Ok(r#"{"stay": true}"#.to_string())
    } else {
        Ok(r#"{"close": ""}"#.to_string())
    }
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    unsafe {
        FACE = None;
    }
    Ok("{}".to_string())
}

/// One slash command per face.
#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(
        r#"[{"name": "matrix", "args": "", "desc": "falling glyphs"},
           {"name": "starfield", "args": "", "desc": "a perspective starfield"},
           {"name": "breakout", "args": "", "desc": "bricks, a paddle and a ball"}]"#
            .to_string(),
    )
}

/// Opens the named face. The host resolves `open` against *this* component's
/// frame, so a face name is all that has to travel.
#[plugin_fn]
pub fn command_run(input: String) -> FnResult<String> {
    let name = support::text(&input, "name");
    Ok(format!(r#"{{"open": "{name}"}}"#))
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Screensaver faces, as one WASM component.
//!
//! A screensaver plugin declares `"kind": "screensaver"`, which is what puts
//! it in the idle rotation beside the faces plank ships. It is a different
//! plugin from the arcade on purpose: the two want opposite behaviour — a
//! screensaver appears unasked and any key dismisses it, a game is started
//! deliberately and keeps its keys — and a user who wants ambient faces and no
//! games should be able to install exactly that.
//!
//! **plank's own matrix rain is not here.** It stays in the binary, because it
//! is the *default* screensaver: a plank with no plugins installed still has
//! one, and a default that depends on a plugin is not a default. The same goes
//! for breakout, which the download screen draws.

mod minions;
mod starfield;
mod support;

use extism_pdk::*;

/// The ABI this component speaks. Checked against the host's at load.
#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

/// The open face. A guest is single-threaded and the host serializes calls
/// into one plugin, so a static is the whole story.
static mut FACE: Option<Face> = None;

/// The faces this component offers.
enum Face {
    /// A perspective starfield.
    Starfield(starfield::Starfield),
    /// Two minions on the shore of a night lake.
    Minions(minions::Skit),
}

impl Face {
    /// Builds a face by name. An unknown name is the starfield rather than a
    /// failure: the host has already decided to open something, and a black
    /// screen is a worse answer than the wrong ambient face.
    fn new(name: &str, seed: u64) -> Self {
        match name {
            "minions" => Self::Minions(minions::Skit::new(seed)),
            _ => Self::Starfield(starfield::Starfield::new(seed, STARS)),
        }
    }

    fn step(&mut self, dt_ms: u64) {
        match self {
            Self::Starfield(s) => s.step(dt_ms),
            Self::Minions(m) => m.step(dt_ms),
        }
    }

    fn glyphs(&self, w: u16, h: u16) -> Vec<support::Glyph> {
        match self {
            Self::Starfield(s) => s.glyphs(w, h),
            Self::Minions(m) => m.glyphs(w, h),
        }
    }
}

/// Steps since the frame opened, reported on close so the achieved frame rate
/// is observable from outside.
static mut STEPS: u32 = 0;

/// The star count plank's own arcade opens its field with. Kept identical so a
/// ported face is the same face, which is what the host-side glyph-for-glyph
/// test pins.
const STARS: usize = 220;

#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    let seed = support::int(&input, "seed");
    let face = support::text(&input, "arg");
    unsafe {
        FACE = Some(Face::new(&face, seed));
        STEPS = 0;
    }
    Ok(r#"{"ok": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_step(input: String) -> FnResult<Vec<u8>> {
    let dt_ms = support::int(&input, "dt_ms");
    let w = support::num(&input, "w") as u16;
    let h = support::num(&input, "h") as u16;
    let face = unsafe { (*(&raw mut FACE)).as_mut() };
    let Some(face) = face else {
        // Stepped before opening: paint nothing rather than trap. An empty
        // buffer is both decodable and honest.
        return Ok(support::encode(&[], w, h));
    };
    face.step(dt_ms);
    unsafe {
        STEPS = STEPS.saturating_add(1);
    }
    Ok(support::encode(&face.glyphs(w, h), w, h))
}

/// A screensaver is dismissed by anything — that is what makes it a
/// screensaver — except the two keys that steer the field, which are worth
/// keeping for someone who opened it on purpose.
#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    let code = support::text(&input, "code");
    let face = unsafe { (*(&raw mut FACE)).as_mut() };
    let Some(face) = face else {
        return Ok(r#"{"close": ""}"#.to_string());
    };
    match (face, code.as_str()) {
        (Face::Starfield(sky), "up" | "k") => sky.scale_speed(1.3),
        (Face::Starfield(sky), "down" | "j") => sky.scale_speed(1.0 / 1.3),
        // The minions have no controls; anything closes them, which is what a
        // screensaver should do.
        _ => return Ok(r#"{"close": ""}"#.to_string()),
    }
    Ok(r#"{"stay": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    let steps = unsafe {
        FACE = None;
        STEPS
    };
    Ok(format!(
        r#"{{"scrollback": "screensaver: {steps} frames"}}"#
    ))
}

/// One slash command per face.
#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(
        r#"[{"name": "starfield", "args": "", "desc": "a perspective starfield"},
           {"name": "minions", "args": "", "desc": "two minions on a night lake"}]"#
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

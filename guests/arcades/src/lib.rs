// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The arcade games, as one WASM component.
//!
//! Separate from the screensavers plugin because they are separate things: a
//! game is started deliberately and owns its keys until it is quit, and
//! someone who wants ambient faces should not have to install games.
//!
//! plank's own breakout is not here. It stays in the binary because the
//! download screen draws it above the progress gauge, which runs long before
//! any plugin could be loaded.

mod centipede;
mod frogger;
mod invaders;
mod shared;
mod support;

use extism_pdk::*;

/// The ABI this component speaks. Checked against the host's at load.
#[plugin_fn]
pub fn plank_abi() -> FnResult<String> {
    Ok("1".to_string())
}

/// The open game. A guest is single-threaded and the host serializes calls
/// into one plugin, so a static is the whole story.
static mut GAME: Option<Game> = None;

/// Steps since the frame opened, reported on close so the achieved frame rate
/// is observable from outside.
static mut STEPS: u32 = 0;

enum Game {
    Centipede(centipede::Centipede),
    Frogger(frogger::Frogger),
    Invaders(invaders::Invaders),
}

impl Game {
    /// Builds a game by name. An unknown name is centipede rather than a
    /// failure: the host has already decided to open something.
    fn new(name: &str, seed: u64) -> Self {
        match name {
            "frogger" => Self::Frogger(frogger::Frogger::new(seed)),
            "invaders" => Self::Invaders(invaders::Invaders::new(seed)),
            _ => Self::Centipede(centipede::Centipede::new(seed)),
        }
    }

    fn step(&mut self, dt_ms: u64) {
        match self {
            Self::Centipede(g) => g.step(dt_ms),
            Self::Frogger(g) => g.step(dt_ms),
            Self::Invaders(g) => g.step(dt_ms),
        }
    }

    fn glyphs(&self, w: u16, h: u16) -> Vec<support::Glyph> {
        match self {
            Self::Centipede(g) => g.glyphs(w, h),
            Self::Frogger(g) => g.glyphs(w, h),
            Self::Invaders(g) => g.glyphs(w, h),
        }
    }

    fn handle_key(&mut self, code: &str) -> bool {
        match self {
            Self::Centipede(g) => g.handle_key(code),
            Self::Frogger(g) => g.handle_key(code),
            Self::Invaders(g) => g.handle_key(code),
        }
    }
}

#[plugin_fn]
pub fn frame_open(input: String) -> FnResult<String> {
    let seed = support::int(&input, "seed");
    let name = support::text(&input, "arg");
    unsafe {
        GAME = Some(Game::new(&name, seed));
        STEPS = 0;
    }
    Ok(r#"{"ok": true}"#.to_string())
}

#[plugin_fn]
pub fn frame_step(input: String) -> FnResult<Vec<u8>> {
    let dt_ms = support::int(&input, "dt_ms");
    let w = support::num(&input, "w") as u16;
    let h = support::num(&input, "h") as u16;
    let game = unsafe { (*(&raw mut GAME)).as_mut() };
    let Some(game) = game else {
        // Stepped before opening: paint nothing rather than trap.
        return Ok(support::encode(&[], w, h));
    };
    game.step(dt_ms);
    unsafe {
        STEPS = STEPS.saturating_add(1);
    }
    Ok(support::encode(&game.glyphs(w, h), w, h))
}

/// A game claims the keys it uses; anything else closes it, which is how Esc
/// and `q` work without every game spelling them out.
#[plugin_fn]
pub fn frame_key(input: String) -> FnResult<String> {
    let code = support::text(&input, "code");
    let game = unsafe { (*(&raw mut GAME)).as_mut() };
    let claimed = game.is_some_and(|g| g.handle_key(&code));
    if claimed {
        Ok(r#"{"stay": true}"#.to_string())
    } else {
        Ok(r#"{"close": ""}"#.to_string())
    }
}

#[plugin_fn]
pub fn frame_close() -> FnResult<String> {
    let steps = unsafe {
        GAME = None;
        STEPS
    };
    Ok(format!(r#"{{"scrollback": "arcade: {steps} frames"}}"#))
}

/// One slash command per game.
#[plugin_fn]
pub fn command_specs() -> FnResult<String> {
    Ok(
        r#"[{"name": "centipede", "args": "", "desc": "shoot the centipede"},
           {"name": "frogger", "args": "", "desc": "cross the road"},
           {"name": "invaders", "args": "", "desc": "hold off the invaders"}]"#
            .to_string(),
    )
}

/// Opens the named game. The host resolves `open` against *this* component's
/// frame, so a game name is all that has to travel.
#[plugin_fn]
pub fn command_run(input: String) -> FnResult<String> {
    let name = support::text(&input, "name");
    Ok(format!(r#"{{"open": "{name}"}}"#))
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Pieces the games share, ported verbatim from plank's `src/arcade.rs`.
//!
//! Verbatim because a game's physics is tuned against these exact numbers: a
//! port that quietly rounded one would play differently in a way only a
//! glyph-for-glyph comparison could find.

#![allow(dead_code)]

/// Sub-step the physics integrates at, in milliseconds.
pub const SUB_MS: u64 = 8;
/// How close two bodies must be to count as touching, in field units.
pub const CONTACT: f32 = 0.012;
/// Horizontal stretch of a ball's velocity. A character cell is about twice as
/// tall as it is wide, so an unstretched ball looks like it crawls sideways.
pub const H_GAIN: f32 = 1.15;
/// How long a key press keeps a player moving, in milliseconds. Terminal key
/// repeat re-arms this, so holding a key gives continuous travel while a single
/// tap gives a short nudge.
pub const GLIDE_MS: u64 = 190;
/// Smallest terminal a game will render into.
pub const MIN_W: u16 = 30;
pub const MIN_H: u16 = 9;

/// Where a match currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Ball parked at center, counting down to the serve.
    Serve { ms_left: u64 },
    /// A rally is live.
    Playing,
    /// Level cleared; holding the banner before the next one.
    LevelUp { ms_left: u64 },
    /// Paused by the player.
    Paused,
    /// The CPU took the level.
    GameOver,
    /// All five levels cleared.
    Won,
}

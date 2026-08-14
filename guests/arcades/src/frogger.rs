//! **Ported from plank's `src/arcade/frogger.rs`.** Physics, level table and
//! drawing are verbatim; only the imports and `handle_key` changed, the latter
//! because a guest sees the ABI's key *name* rather than a crossterm
//! `KeyEvent`. Mouse handling is dropped until the host delivers
//! `frame_mouse`.

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Frogger: cross the road, then ride the river home.
//!
//! Same contract as the rest of [`crate::arcade`] — a 1x1 normalized field,
//! state advanced only by an injected `dt`, rendering handed out as
//! [`Glyph`]s. The frog hops between whole rows (that is the feel of the game)
//! but everything it dodges or rides moves continuously.

// Every numeric conversion in this file turns a board index or a small
// dimension constant into `f32` for the geometry, or back for an array index.
// The board is a couple of dozen cells across; none of these can lose a bit.
// Declared once rather than as a per-function allow on nearly every function.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#![allow(dead_code)]

use crate::shared::{GLIDE_MS, MIN_H, MIN_W, Phase, SUB_MS};
use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng};

/// Board rows, top to bottom: home bank, five river lanes, median, five road
/// lanes, start bank.
pub const ROWS: usize = 13;
const HOME_ROW: usize = 0;
const RIVER: std::ops::Range<usize> = 1..6;
const MEDIAN: usize = 6;
const ROAD: std::ops::Range<usize> = 7..12;
const START_ROW: usize = 12;
/// Homes to fill to clear a level.
const HOMES: usize = 5;
/// Lives at the start of a game.
const LIVES: u32 = 3;
/// How long a hop takes, in milliseconds. Long enough to read as a hop, short
/// enough that you can still react.
const HOP_MS: u64 = 90;
/// Colors.
const FROG_COLOR: Rgb = (150, 245, 130);
const CAR_COLORS: [Rgb; 5] = [
    (255, 120, 120),
    (255, 190, 90),
    (200, 160, 255),
    (255, 140, 200),
    (140, 210, 255),
];
const LOG_COLOR: Rgb = (190, 140, 90);
const TURTLE_COLOR: Rgb = (120, 220, 190);
const HOME_COLOR: Rgb = (110, 255, 180);
const BANK_COLOR: Rgb = (70, 90, 80);

/// Per-level difficulty.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec {
    /// Multiplier on every lane's speed.
    pub pace: f32,
    /// Multiplier on the gap between things in a lane: below 1 the traffic
    /// thickens and the logs get shorter.
    pub density: f32,
}

/// Five levels: faster traffic, tighter gaps.
pub const LEVELS: [LevelSpec; 5] = [
    LevelSpec {
        pace: 1.0,
        density: 1.0,
    },
    LevelSpec {
        pace: 1.25,
        density: 0.9,
    },
    LevelSpec {
        pace: 1.5,
        density: 0.8,
    },
    LevelSpec {
        pace: 1.8,
        density: 0.72,
    },
    LevelSpec {
        pace: 2.1,
        density: 0.65,
    },
];

/// One lane of traffic or river craft.
#[derive(Debug, Clone, Copy)]
struct Lane {
    /// Field-widths per second; the sign is the direction.
    speed: f32,
    /// Distance between the start of one body and the next.
    gap: f32,
    /// How wide one body is.
    len: f32,
    /// Scroll offset, wrapped into `0..gap`.
    offset: f32,
    /// Rideable (a log or turtles) rather than lethal (a car).
    rideable: bool,
    /// Turtles look different from logs and are the ones that submerge.
    turtles: bool,
}

impl Lane {
    /// Whether `x` is on a body of this lane.
    fn covers(&self, x: f32) -> bool {
        let phase = (x - self.offset).rem_euclid(self.gap);
        phase < self.len
    }

    /// Walks the lane by `dt`, keeping the offset small.
    fn advance(&mut self, dt: f32) {
        self.offset = (self.offset + self.speed * dt).rem_euclid(self.gap);
    }
}

/// A frogger game.
#[derive(Debug)]
pub struct Frogger {
    level: usize,
    lives: u32,
    score: u32,
    lanes: [Lane; ROWS],
    /// Which of the five home slots are already filled.
    homes: [bool; HOMES],
    frog: (f32, usize),
    /// Time left in the current hop, for the squash-and-stretch glyph.
    hop_ms: u64,
    glide_ms: u64,
    phase: Phase,
    acc_ms: u64,
    rng: Rng,
}

impl Frogger {
    /// Starts a new game at level 1.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut me = Self {
            level: 0,
            lives: LIVES,
            score: 0,
            lanes: [Lane {
                speed: 0.0,
                gap: 1.0,
                len: 0.0,
                offset: 0.0,
                rideable: false,
                turtles: false,
            }; ROWS],
            homes: [false; HOMES],
            frog: (0.5, START_ROW),
            hop_ms: 0,
            glide_ms: 0,
            phase: Phase::Serve { ms_left: 900 },
            acc_ms: 0,
            rng: Rng::new(seed),
        };
        me.build_lanes();
        me
    }

    /// The level being played, 1-based.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level + 1
    }

    /// Homes still to fill.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.homes.iter().filter(|h| !**h).count()
    }

    /// Points so far. Only ever goes up, which is what makes it usable as the
    /// "something good happened" signal for the optional blips.
    #[must_use]
    pub const fn score(&self) -> u32 {
        self.score
    }

    /// Lives left.
    #[must_use]
    pub const fn lives(&self) -> u32 {
        self.lives
    }

    /// Current phase.
    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    /// Whether the game has ended, either way.
    #[must_use]
    pub const fn finished(&self) -> bool {
        matches!(self.phase, Phase::GameOver | Phase::Won)
    }

    const fn spec(&self) -> LevelSpec {
        LEVELS[self.level]
    }

    /// Lays out the lanes for the current level. Alternating directions is what
    /// makes the board readable: you can see at a glance which way to time it.
    fn build_lanes(&mut self) {
        let spec = self.spec();
        for row in 0..ROWS {
            let lane = &mut self.lanes[row];
            *lane = Lane {
                speed: 0.0,
                gap: 1.0,
                len: 0.0,
                offset: 0.0,
                rideable: false,
                turtles: false,
            };
            let dir = if row % 2 == 0 { 1.0 } else { -1.0 };
            if ROAD.contains(&row) {
                let base = 0.10 + 0.035 * (row - ROAD.start) as f32;
                lane.speed = dir * base * spec.pace;
                lane.gap = (0.34 - 0.02 * (row - ROAD.start) as f32) * spec.density;
                lane.len = 0.06;
            } else if RIVER.contains(&row) {
                let i = row - RIVER.start;
                let base = 0.09 + 0.03 * i as f32;
                lane.speed = dir * base * spec.pace;
                lane.rideable = true;
                // Turtles on two of the five lanes, logs on the rest.
                lane.turtles = i == 1 || i == 3;
                lane.gap = (0.40 - 0.02 * i as f32) * spec.density;
                lane.len = if lane.turtles { 0.09 } else { 0.16 };
            }
            // A random start so two levels never look the same.
            lane.offset = self.rng.range(0.0, lane.gap.max(f32::EPSILON));
        }
    }

    /// Feeds one key to the game.
    pub fn handle_key(&mut self, code: &str) -> bool {
        match code {
            "up" | "k" | "w" => self.hop(0, -1),
            "down" | "j" | "s" => self.hop(0, 1),
            "left" | "h" | "a" => self.hop(-1, 0),
            "right" | "l" | "d" => self.hop(1, 0),
            "p" => {
                self.phase = match self.phase {
                    Phase::Paused => Phase::Serve { ms_left: 500 },
                    Phase::Playing | Phase::Serve { .. } => Phase::Paused,
                    other => other,
                };
            }
            "space" | "enter" => match self.phase {
                Phase::Serve { .. } => self.phase = Phase::Playing,
                Phase::Paused => self.phase = Phase::Serve { ms_left: 500 },
                Phase::GameOver | Phase::Won => *self = Self::new(self.rng.next_u64()),
                _ => {}
            },
            _ => return false,
        }
        true
    }

    /// One hop. Horizontal hops are a fixed fraction of the width; vertical
    /// hops are a whole row, which is what makes the board legible.
    #[allow(clippy::cast_precision_loss)] // dx is -1, 0 or 1
    fn hop(&mut self, dx: i32, dy: i32) {
        self.unpause();
        if self.phase != Phase::Playing {
            return;
        }
        self.glide_ms = GLIDE_MS;
        self.hop_ms = HOP_MS;
        if dx != 0 {
            self.frog.0 = (dx as f32).mul_add(0.055, self.frog.0).clamp(0.0, 1.0);
            return;
        }
        // Saturating rather than signed casts: a hop off either end of the
        // board simply does not move the frog.
        self.frog.1 = if dy < 0 {
            self.frog.1.saturating_sub(1)
        } else {
            (self.frog.1 + 1).min(ROWS - 1)
        };
        if dy < 0 {
            // Getting further up the board is the thing worth points.
            self.score += 1;
        }
        if self.frog.1 == HOME_ROW {
            self.reach_home();
        }
    }

    fn unpause(&mut self) {
        if self.phase == Phase::Paused {
            self.phase = Phase::Serve { ms_left: 500 };
        }
    }

    /// Advances the game by `dt_ms`, in fixed sub-steps.
    pub fn step(&mut self, dt_ms: u64) {
        let dt_ms = dt_ms.min(MAX_STEP_MS);
        self.glide_ms = self.glide_ms.saturating_sub(dt_ms);
        self.hop_ms = self.hop_ms.saturating_sub(dt_ms);
        match self.phase {
            Phase::Paused | Phase::GameOver | Phase::Won => return,
            Phase::Serve { ms_left } => {
                self.phase = match ms_left.checked_sub(dt_ms) {
                    Some(0) | None => Phase::Playing,
                    Some(left) => Phase::Serve { ms_left: left },
                };
            }
            Phase::LevelUp { ms_left } => {
                self.phase = match ms_left.checked_sub(dt_ms) {
                    Some(0) | None => {
                        self.level += 1;
                        self.homes = [false; HOMES];
                        self.build_lanes();
                        self.frog = (0.5, START_ROW);
                        Phase::Serve { ms_left: 900 }
                    }
                    Some(left) => Phase::LevelUp { ms_left: left },
                };
            }
            Phase::Playing => {}
        }
        self.acc_ms += dt_ms;
        while self.acc_ms >= SUB_MS {
            self.acc_ms -= SUB_MS;
            self.tick();
        }
    }

    #[allow(clippy::cast_precision_loss)] // SUB_MS is a small constant
    fn tick(&mut self) {
        let dt = SUB_MS as f32 / 1000.0;
        for lane in &mut self.lanes {
            lane.advance(dt);
        }
        if self.phase != Phase::Playing {
            return;
        }
        let row = self.frog.1;
        if ROAD.contains(&row) {
            if self.lanes[row].covers(self.frog.0) {
                self.squashed();
            }
        } else if RIVER.contains(&row) {
            let lane = self.lanes[row];
            if lane.covers(self.frog.0) {
                // Riding: the frog drifts with whatever it is standing on, and
                // being carried off the edge drowns it just the same.
                self.frog.0 += lane.speed * dt;
                if !(0.0..=1.0).contains(&self.frog.0) {
                    self.squashed();
                }
            } else {
                self.squashed();
            }
        }
    }

    /// Lost a life: back to the near bank, or the end.
    fn squashed(&mut self) {
        self.lives = self.lives.saturating_sub(1);
        self.frog = (0.5, START_ROW);
        self.hop_ms = 0;
        self.phase = if self.lives == 0 {
            Phase::GameOver
        } else {
            Phase::Serve { ms_left: 700 }
        };
    }

    /// The frog reached the top row: fill a home, or fall in the water.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped to 0..HOMES
    fn reach_home(&mut self) {
        // The five homes are evenly spaced bays; anything between them is bank.
        let slot = (self.frog.0 * HOMES as f32)
            .floor()
            .clamp(0.0, (HOMES - 1) as f32) as usize;
        let centre = (slot as f32 + 0.5) / HOMES as f32;
        if (self.frog.0 - centre).abs() > 0.5 / HOMES as f32 * 0.7 || self.homes[slot] {
            // Missed the bay, or it is already taken.
            self.squashed();
            return;
        }
        self.homes[slot] = true;
        self.score += 50;
        if self.remaining() == 0 {
            self.phase = if self.level + 1 >= LEVELS.len() {
                Phase::Won
            } else {
                Phase::LevelUp { ms_left: 1400 }
            };
            return;
        }
        self.frog = (0.5, START_ROW);
        self.phase = Phase::Serve { ms_left: 500 };
    }

    /// Paints the board across the whole area.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )] // board coords scaled into a bounded, positive range
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w < MIN_W || h < MIN_H {
            return Vec::new();
        }
        let sy = f32::from(h - 1) / (ROWS - 1) as f32;
        let cols = f32::from(w - 1);
        let mut out = Vec::with_capacity(usize::from(w) * 4);

        for row in 0..ROWS {
            let y = (row as f32 * sy).round() as u16;
            if row == HOME_ROW {
                // The five bays, filled or waiting.
                for (slot, filled) in self.homes.iter().enumerate() {
                    let centre = (slot as f32 + 0.5) / HOMES as f32;
                    let x = (centre * cols).round() as u16;
                    out.push(Glyph {
                        x,
                        y,
                        ch: if *filled { '❁' } else { '◇' },
                        color: HOME_COLOR,
                    });
                }
                continue;
            }
            if row == MEDIAN || row == START_ROW {
                for x in 0..w {
                    out.push(Glyph {
                        x,
                        y,
                        ch: '─',
                        color: BANK_COLOR,
                    });
                }
                continue;
            }
            let lane = self.lanes[row];
            let (ch, color) = if !lane.rideable {
                ('▄', CAR_COLORS[(row - ROAD.start) % CAR_COLORS.len()])
            } else if lane.turtles {
                ('◠', TURTLE_COLOR)
            } else {
                ('▬', LOG_COLOR)
            };
            for x in 0..w {
                if lane.covers(f32::from(x) / cols) {
                    out.push(Glyph { x, y, ch, color });
                }
            }
        }

        if !matches!(self.phase, Phase::GameOver) {
            out.push(Glyph {
                x: (self.frog.0.clamp(0.0, 1.0) * cols).round() as u16,
                y: (self.frog.1 as f32 * sy).round() as u16,
                // Mid-hop the frog is stretched; at rest it sits.
                ch: if self.hop_ms > 0 { '✦' } else { '☗' },
                color: FROG_COLOR,
            });
        }
        out
    }

    /// The banner across the middle, if any.
    #[must_use]
    pub fn banner(&self) -> Option<String> {
        match self.phase {
            Phase::LevelUp { .. } => Some(format!("ALL HOME — LEVEL {}", self.level() + 1)),
            Phase::Paused => Some("PAUSED".to_string()),
            Phase::GameOver => Some(format!(
                "SPLATTED AT {} POINTS — space to restart",
                self.score
            )),
            Phase::Won => Some(format!("EVERY FROG HOME — {} points", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "level {}/{}  ·  {} points  ·  {} lives  ·  {} bays open",
            self.level(),
            LEVELS.len(),
            self.score,
            self.lives,
            self.remaining()
        )
    }
}

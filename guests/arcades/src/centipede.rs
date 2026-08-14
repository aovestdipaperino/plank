//! **Ported from plank's `src/arcade/centipede.rs`.** Physics, level table and
//! drawing are verbatim; only the imports and `handle_key` changed, the latter
//! because a guest sees the ABI's key *name* rather than a crossterm
//! `KeyEvent`. Mouse handling is dropped until the host delivers
//! `frame_mouse`.

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Centipede: shoot the thing apart before it walks into you.
//!
//! Same contract as the rest of [`crate::arcade`] — a 1x1 normalized field,
//! state advanced only by an injected `dt`, rendering handed out as
//! [`Glyph`]s — with one difference: the mushroom field and the centipede walk
//! on a **discrete grid**, because a centipede that turns on a mushroom has to
//! agree with the player about where that mushroom is.

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

/// The playfield grid. Everything that has to agree about position — the
/// centipede, the mushrooms, the shot — lives on these cells.
pub const COLS: usize = 30;
pub const ROWS: usize = 18;
/// How many bottom rows the player may move in.
const PLAYER_ROWS: usize = 5;
/// Segments the centipede starts with.
const START_LEN: usize = 10;
/// Mushrooms scattered at the start of a level.
const START_SHROOMS: usize = 30;
/// Hits a mushroom takes before it goes.
const SHROOM_HP: u8 = 4;
/// Player travel, in cells per second.
const PLAYER_SPEED: f32 = 16.0;
/// Shot travel, in cells per second. Only one shot is ever in the air.
const SHOT_SPEED: f32 = 40.0;
/// Lives at the start of a game.
const LIVES: u32 = 3;
/// Head and body colors, and the mushroom ramp from fresh to nearly gone.
const HEAD_COLOR: Rgb = (255, 120, 200);
const BODY_COLOR: Rgb = (170, 240, 130);
const SHROOM_COLORS: [Rgb; SHROOM_HP as usize] = [
    (90, 70, 50),
    (140, 110, 70),
    (190, 150, 90),
    (230, 190, 120),
];

/// Per-level difficulty.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec {
    /// Centipede steps per second.
    pub steps_per_sec: f32,
    /// Extra mushrooms seeded on top of the survivors.
    pub extra_shrooms: usize,
}

/// Five levels: a faster crawl through a thicker forest.
pub const LEVELS: [LevelSpec; 5] = [
    LevelSpec {
        steps_per_sec: 7.0,
        extra_shrooms: 0,
    },
    LevelSpec {
        steps_per_sec: 9.0,
        extra_shrooms: 6,
    },
    LevelSpec {
        steps_per_sec: 11.5,
        extra_shrooms: 12,
    },
    LevelSpec {
        steps_per_sec: 14.0,
        extra_shrooms: 18,
    },
    LevelSpec {
        steps_per_sec: 17.0,
        extra_shrooms: 24,
    },
];

/// One body segment, walking on the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Segment {
    col: usize,
    row: usize,
    /// +1 walking right, -1 walking left.
    dir: i8,
}

/// A centipede game.
#[derive(Debug)]
pub struct Centipede {
    level: usize,
    lives: u32,
    score: u32,
    /// Remaining hit points per cell, 0 for empty. Row-major `ROWS * COLS`.
    shrooms: Vec<u8>,
    /// Head first: killing a middle segment splits the train in two, and both
    /// halves keep walking, so order is the structure.
    body: Vec<Segment>,
    player: (f32, f32),
    dir: (f32, f32),
    glide_ms: u64,
    /// The single shot in the air, if any.
    shot: Option<(usize, f32)>,
    /// Time owed to the centipede's next step.
    walk_ms: f32,
    phase: Phase,
    acc_ms: u64,
    rng: Rng,
}

impl Centipede {
    /// Starts a new game at level 1.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut me = Self {
            level: 0,
            lives: LIVES,
            score: 0,
            shrooms: vec![0; ROWS * COLS],
            body: Vec::new(),
            player: (mid_col(), last_row()),
            dir: (0.0, 0.0),
            glide_ms: 0,
            shot: None,
            walk_ms: 0.0,
            phase: Phase::Serve { ms_left: 900 },
            acc_ms: 0,
            rng: Rng::new(seed),
        };
        me.seed_forest(START_SHROOMS);
        me.spawn_centipede();
        me
    }

    /// The level being played, 1-based.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level + 1
    }

    /// Segments still crawling.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.body.len()
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

    /// Scatters `n` mushrooms, keeping the bottom row clear so the player
    /// always has somewhere to stand.
    fn seed_forest(&mut self, n: usize) {
        for _ in 0..n {
            let col = self.pick(COLS);
            let row = 2 + self.pick(ROWS - 4);
            self.shrooms[row * COLS + col] = SHROOM_HP;
        }
    }

    /// A value in `0..n` from the seeded generator.
    fn pick(&mut self, n: usize) -> usize {
        usize::try_from(self.rng.next_u64() % n as u64).unwrap_or(0)
    }

    fn spawn_centipede(&mut self) {
        self.body = (0..START_LEN)
            .map(|i| Segment {
                col: START_LEN - 1 - i,
                row: 0,
                dir: 1,
            })
            .collect();
        self.walk_ms = 0.0;
    }

    /// Feeds one key to the game.
    pub fn handle_key(&mut self, code: &str) -> bool {
        match code {
            "left" | "h" | "a" => self.steer(-1.0, 0.0),
            "right" | "l" | "d" => self.steer(1.0, 0.0),
            "up" | "k" | "w" => self.steer(0.0, -1.0),
            "down" | "j" | "s" => self.steer(0.0, 1.0),
            "space" | "enter" => match self.phase {
                Phase::Playing => self.fire(),
                Phase::Serve { .. } => self.phase = Phase::Playing,
                Phase::Paused => self.phase = Phase::Serve { ms_left: 500 },
                Phase::GameOver | Phase::Won => *self = Self::new(self.rng.next_u64()),
                Phase::LevelUp { .. } => {}
            },
            "p" => {
                self.phase = match self.phase {
                    Phase::Paused => Phase::Serve { ms_left: 500 },
                    Phase::Playing | Phase::Serve { .. } => Phase::Paused,
                    other => other,
                };
            }
            _ => return false,
        }
        true
    }

    fn steer(&mut self, dx: f32, dy: f32) {
        self.dir = (dx, dy);
        self.glide_ms = GLIDE_MS;
        self.unpause();
    }

    /// Fires, if the previous shot has already landed.
    fn fire(&mut self) {
        if self.shot.is_some() {
            return;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped to the grid
        let col = self.player.0.round() as usize;
        self.shot = Some((col.min(COLS - 1), self.player.1));
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
        if self.glide_ms == 0 {
            self.dir = (0.0, 0.0);
        }
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
                        let extra = self.spec().extra_shrooms;
                        self.seed_forest(extra);
                        self.spawn_centipede();
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

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // SUB_MS and the grid are small constants
    fn tick(&mut self) {
        let dt = SUB_MS as f32 / 1000.0;
        self.player = (
            self.dir
                .0
                .mul_add(PLAYER_SPEED * dt, self.player.0)
                .clamp(0.0, COLS as f32 - 1.0),
            self.dir
                .1
                .mul_add(PLAYER_SPEED * dt, self.player.1)
                .clamp((ROWS - PLAYER_ROWS) as f32, ROWS as f32 - 1.0),
        );

        if self.phase != Phase::Playing {
            return;
        }

        self.move_shot(dt);

        // The centipede walks on its own clock, a whole cell at a time.
        let period = 1000.0 / self.spec().steps_per_sec;
        self.walk_ms += SUB_MS as f32;
        while self.walk_ms >= period {
            self.walk_ms -= period;
            self.crawl();
        }
        self.check_squash();
    }

    /// Moves the shot, resolving whatever it meets first.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // grid coords, bounds-checked
    fn move_shot(&mut self, dt: f32) {
        let Some((col, row)) = self.shot.as_mut() else {
            return;
        };
        *row -= SHOT_SPEED * dt;
        let (col, row) = (*col, *row);
        if row < 0.0 {
            self.shot = None;
            return;
        }
        let cell_row = row.round().max(0.0) as usize;
        if cell_row >= ROWS {
            return;
        }
        // A segment first: hitting the bug is the point, and it is standing on
        // the mushroom the shot would otherwise chew.
        if let Some(i) = self
            .body
            .iter()
            .position(|s| s.col == col && s.row == cell_row)
        {
            self.kill_segment(i);
            self.shot = None;
            return;
        }
        let slot = &mut self.shrooms[cell_row * COLS + col];
        if *slot > 0 {
            *slot -= 1;
            if *slot == 0 {
                self.score += 1;
            }
            self.shot = None;
        }
    }

    /// Removes segment `i`, leaving a mushroom where it fell.
    ///
    /// The tail behind the break keeps crawling as its own centipede — that is
    /// the whole shape of the game, and it falls out of keeping the body as one
    /// ordered list rather than as a rigid train.
    fn kill_segment(&mut self, i: usize) {
        let seg = self.body.remove(i);
        self.shrooms[seg.row * COLS + seg.col] = SHROOM_HP;
        self.score += 10;
        if self.body.is_empty() {
            self.phase = if self.level + 1 >= LEVELS.len() {
                Phase::Won
            } else {
                Phase::LevelUp { ms_left: 1400 }
            };
        }
    }

    /// Walks every segment one cell, turning down at a wall or a mushroom.
    #[allow(clippy::cast_possible_wrap)] // COLS is 30; nothing here can wrap
    fn crawl(&mut self) {
        for i in 0..self.body.len() {
            let seg = self.body[i];
            // Signed arithmetic on purpose: the step off column 0 going left is
            // the wall test, and it has to be representable to be tested.
            let next = isize::from(seg.dir) + seg.col as isize;
            let blocked = !(0..COLS as isize).contains(&next)
                || self.shrooms[seg.row * COLS + next.unsigned_abs()] > 0;
            if blocked {
                // Down one row and back the other way; off the bottom, wrap to
                // the top so the game never simply runs out of centipede.
                let row = if seg.row + 1 >= ROWS { 0 } else { seg.row + 1 };
                self.body[i] = Segment {
                    col: seg.col,
                    row,
                    dir: -seg.dir,
                };
            } else {
                self.body[i].col = next.unsigned_abs();
            }
        }
    }

    /// Ends the life if any segment is standing on the player.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped to the grid
    fn check_squash(&mut self) {
        let (pc, pr) = (
            self.player.0.round() as usize,
            self.player.1.round() as usize,
        );
        if !self.body.iter().any(|s| s.col == pc && s.row == pr) {
            return;
        }
        self.lives = self.lives.saturating_sub(1);
        if self.lives == 0 {
            self.phase = Phase::GameOver;
            return;
        }
        self.shot = None;
        self.spawn_centipede();
        self.player = (mid_col(), last_row());
        self.phase = Phase::Serve { ms_left: 700 };
    }

    /// Paints the forest, the centipede and the gun across the whole area.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )] // grid coords scaled into a bounded, positive range
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w < MIN_W || h < MIN_H {
            return Vec::new();
        }
        let sx = f32::from(w - 1) / (COLS as f32 - 1.0);
        let sy = f32::from(h - 1) / (ROWS as f32 - 1.0);
        let at =
            |col: usize, row: f32| ((col as f32 * sx).round() as u16, (row * sy).round() as u16);

        let mut out = Vec::with_capacity(ROWS * COLS / 2);
        for row in 0..ROWS {
            for col in 0..COLS {
                let hp = self.shrooms[row * COLS + col];
                if hp == 0 {
                    continue;
                }
                let (x, y) = at(col, row as f32);
                out.push(Glyph {
                    x,
                    y,
                    ch: if hp >= SHROOM_HP { '♣' } else { '♠' },
                    color: SHROOM_COLORS[(hp - 1) as usize],
                });
            }
        }
        for (i, seg) in self.body.iter().enumerate() {
            let (x, y) = at(seg.col, seg.row as f32);
            // The head of each run is what you want to shoot, so it is marked:
            // a segment whose predecessor is not adjacent starts a new run.
            let leads = i == 0 || !adjacent(self.body[i - 1], *seg);
            out.push(Glyph {
                x,
                y,
                ch: if leads { '◉' } else { '●' },
                color: if leads { HEAD_COLOR } else { BODY_COLOR },
            });
        }
        if let Some((col, row)) = self.shot {
            let (x, y) = at(col, row);
            out.push(Glyph {
                x,
                y,
                ch: '│',
                color: (200, 255, 220),
            });
        }
        if !matches!(self.phase, Phase::GameOver) {
            let (x, y) = at(self.player.0.round() as usize, self.player.1);
            out.push(Glyph {
                x,
                y,
                ch: '▲',
                color: (140, 235, 255),
            });
        }
        out
    }

    /// The banner across the middle, if any.
    #[must_use]
    pub fn banner(&self) -> Option<String> {
        match self.phase {
            Phase::LevelUp { .. } => {
                Some(format!("CENTIPEDE IN PIECES — LEVEL {}", self.level() + 1))
            }
            Phase::Paused => Some("PAUSED".to_string()),
            Phase::GameOver => Some(format!(
                "IT REACHED YOU AT {} POINTS — space to restart",
                self.score
            )),
            Phase::Won => Some(format!("FOREST CLEARED — {} points", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "level {}/{}  ·  {} points  ·  {} lives  ·  {} segments",
            self.level(),
            LEVELS.len(),
            self.score,
            self.lives,
            self.body.len()
        )
    }
}

/// The grid's middle column and bottom row, as floats. Named so the casts live
/// in one place instead of being sprinkled through the physics.
#[allow(clippy::cast_precision_loss)] // grid sizes are small constants
fn mid_col() -> f32 {
    COLS as f32 / 2.0
}

#[allow(clippy::cast_precision_loss)] // grid sizes are small constants
fn last_row() -> f32 {
    ROWS as f32 - 1.0
}

/// Whether `a` is the cell just behind `b` — i.e. they are one run.
fn adjacent(a: Segment, b: Segment) -> bool {
    a.row == b.row && a.col.abs_diff(b.col) == 1
}

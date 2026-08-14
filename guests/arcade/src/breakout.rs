// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! **Ported from plank's `src/arcade/breakout.rs`.** The physics, the level
//! table, the brick layout and the drawing are verbatim; only two things
//! changed. The imports, and `handle_key`, which takes the ABI's key *name*
//! instead of a crossterm `KeyEvent` — a guest never sees a `KeyEvent`, and
//! pinning the ABI to crossterm's enum is exactly what the name-based protocol
//! exists to avoid. Mouse handling is dropped for now: the host does not yet
//! deliver `frame_mouse`.
//!
//! Breakout: knock the wall down without losing the ball.
//!
//! Same contract as the rest of [`crate::arcade`] — a 1x1 normalized field,
//! state advanced only by an injected `dt`, rendering handed out as
//! [`Glyph`]s — so it fills any terminal and replays identically from a seed.

#![allow(dead_code)]

use crate::shared::{CONTACT, H_GAIN, MIN_H, MIN_W, Phase, SUB_MS};
use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng};

/// Bricks per row. Fixed, so the wall reads the same at every size.
pub const COLS: usize = 14;
/// Where the wall starts and how tall one brick row is, as fractions of the
/// field height.
///
/// The wall hangs from the very top: any gap above it is dead space the ball
/// only passes through, and on the download screen it read as a blank row
/// under the rule. The ceiling bounce still works — with the top row cleared
/// the ball simply reaches `y = 0` and reflects.
const WALL_TOP: f32 = 0.0;
const BRICK_H: f32 = 0.05;
/// The paddle's row.
const PADDLE_Y: f32 = 0.94;
/// How fast the paddle travels, in field-widths per second. Comfortably above
/// every ball speed: the ball should beat you on placement, not on pace.
const PADDLE_SPEED: f32 = 1.2;
/// How long one key press keeps the paddle moving.
///
/// Breakout keeps its own, well under the arcade-wide [`super::GLIDE_MS`]:
/// terminals give us discrete presses rather than key-down/key-up, so the
/// glide *is* the step size, and a tap that slides the paddle past its own
/// length makes fine positioning impossible. At [`PADDLE_SPEED`] this moves it
/// roughly half a paddle, while still outlasting the gap between auto-repeats
/// so holding the key reads as continuous travel.
const PADDLE_GLIDE_MS: u64 = 80;
/// Steepest angle off the paddle, in radians (~65°).
const MAX_BOUNCE: f32 = 1.13;
/// Lives at the start of a game.
const LIVES: u32 = 3;
/// Row colors, top (hardest to reach, worth most) to bottom.
const ROW_COLORS: [Rgb; 7] = [
    (255, 96, 96),
    (255, 160, 72),
    (255, 216, 88),
    (140, 224, 120),
    (110, 200, 255),
    (170, 150, 255),
    (230, 130, 220),
];

/// Per-level difficulty.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec {
    /// Brick rows in the wall.
    pub rows: usize,
    /// Ball speed, in field-heights per second.
    pub ball_speed: f32,
    /// Half the paddle, as a fraction of the field width.
    pub paddle_half: f32,
}

/// Five levels: a taller wall, a faster ball, a shorter paddle.
pub const LEVELS: [LevelSpec; 5] = [
    LevelSpec {
        rows: 3,
        ball_speed: 0.55,
        paddle_half: 0.09,
    },
    LevelSpec {
        rows: 4,
        ball_speed: 0.66,
        paddle_half: 0.08,
    },
    LevelSpec {
        rows: 5,
        ball_speed: 0.78,
        paddle_half: 0.07,
    },
    LevelSpec {
        rows: 6,
        ball_speed: 0.90,
        paddle_half: 0.06,
    },
    LevelSpec {
        rows: 7,
        ball_speed: 1.05,
        paddle_half: 0.05,
    },
];

/// A breakout game.
#[derive(Debug)]
pub struct Breakout {
    level: usize,
    lives: u32,
    score: u32,
    /// Row-major `rows * COLS`; true while the brick is still standing.
    bricks: Vec<bool>,
    rows: usize,
    ball: (f32, f32),
    vel: (f32, f32),
    paddle_x: f32,
    dir: f32,
    glide_ms: u64,
    phase: Phase,
    acc_ms: u64,
    rng: Rng,
}

impl Breakout {
    /// Starts a new game at level 1.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut me = Self {
            level: 0,
            lives: LIVES,
            score: 0,
            bricks: Vec::new(),
            rows: 0,
            ball: (0.5, 0.5),
            vel: (0.0, 0.0),
            paddle_x: 0.5,
            dir: 0.0,
            glide_ms: 0,
            phase: Phase::Serve { ms_left: 900 },
            acc_ms: 0,
            rng: Rng::new(seed),
        };
        me.build_wall();
        me.serve();
        me
    }

    /// The level being played, 1-based.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level + 1
    }

    /// Bricks still standing.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.bricks.iter().filter(|b| **b).count()
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

    fn build_wall(&mut self) {
        self.rows = self.spec().rows;
        self.bricks = vec![true; self.rows * COLS];
    }

    /// Puts the ball back on the paddle, aimed upward at a random slant.
    fn serve(&mut self) {
        let s = self.spec().ball_speed;
        let angle = self.rng.range(-0.5, 0.5);
        self.ball = (self.paddle_x, PADDLE_Y - 0.04);
        self.vel = (H_GAIN * s * angle.sin(), -s * angle.cos());
    }

    /// Feeds one key to the game.
    pub fn handle_key(&mut self, code: &str) -> bool {
        match code {
            "left" | "h" | "a" => {
                self.dir = -1.0;
                self.glide_ms = PADDLE_GLIDE_MS;
                self.unpause();
            }
            "right" | "l" | "d" => {
                self.dir = 1.0;
                self.glide_ms = PADDLE_GLIDE_MS;
                self.unpause();
            }
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
            self.dir = 0.0;
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
                        self.build_wall();
                        self.serve();
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
        let half = self.spec().paddle_half;
        self.paddle_x = self
            .dir
            .mul_add(PADDLE_SPEED * dt, self.paddle_x)
            .clamp(half, 1.0 - half);

        // Before the serve the ball rides the paddle, so aiming is aiming.
        if self.phase != Phase::Playing {
            if matches!(self.phase, Phase::Serve { .. }) {
                self.ball.0 = self.paddle_x;
            }
            return;
        }

        self.ball.0 += self.vel.0 * dt;
        self.ball.1 += self.vel.1 * dt;

        // Side and top walls.
        if self.ball.0 < 0.0 {
            self.ball.0 = -self.ball.0;
            self.vel.0 = -self.vel.0;
        } else if self.ball.0 > 1.0 {
            self.ball.0 = 2.0 - self.ball.0;
            self.vel.0 = -self.vel.0;
        }
        if self.ball.1 < 0.0 {
            self.ball.1 = -self.ball.1;
            self.vel.1 = -self.vel.1;
        }

        self.hit_brick();

        // Paddle.
        if self.vel.1 > 0.0
            && self.ball.1 >= PADDLE_Y
            && (self.ball.0 - self.paddle_x).abs() <= half + CONTACT
        {
            let off = (self.ball.0 - self.paddle_x) / (half + CONTACT);
            let s = self.spec().ball_speed;
            let angle = off.clamp(-1.0, 1.0) * MAX_BOUNCE;
            self.ball.1 = PADDLE_Y;
            self.vel = (H_GAIN * s * angle.sin(), -s * angle.cos());
        } else if self.ball.1 > 1.0 {
            self.lose_ball();
        }
    }

    /// Knocks out the brick the ball is inside, if any.
    ///
    /// The reflection just flips the vertical component. Proper edge detection
    /// would be more correct on a corner clip, but at this speed it is
    /// indistinguishable in play and the simple rule never wedges the ball.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )] // guarded by the range checks above; ROWS/COLS are small constants
    fn hit_brick(&mut self) {
        let wall_bottom = BRICK_H.mul_add(self.rows as f32, WALL_TOP);
        if self.ball.1 < WALL_TOP || self.ball.1 >= wall_bottom {
            return;
        }
        if self.ball.0 < 0.0 || self.ball.0 >= 1.0 {
            return;
        }
        let row = ((self.ball.1 - WALL_TOP) / BRICK_H) as usize;
        let col = (self.ball.0 * COLS as f32) as usize;
        let Some(slot) = self.bricks.get_mut(row * COLS + col.min(COLS - 1)) else {
            return;
        };
        if !*slot {
            return;
        }
        *slot = false;
        // Top rows are further away and worth more.
        self.score += u32::try_from(self.rows - row).unwrap_or(1);
        self.vel.1 = -self.vel.1;
        if self.remaining() == 0 {
            self.phase = if self.level + 1 >= LEVELS.len() {
                Phase::Won
            } else {
                Phase::LevelUp { ms_left: 1400 }
            };
        }
    }

    fn lose_ball(&mut self) {
        self.lives = self.lives.saturating_sub(1);
        if self.lives == 0 {
            self.phase = Phase::GameOver;
            return;
        }
        self.serve();
        self.phase = Phase::Serve { ms_left: 700 };
    }

    /// Paints the wall, paddle and ball across the whole `w` x `h` area.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )] // normalized coords scaled into a bounded, positive range
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w < MIN_W || h < MIN_H {
            return Vec::new();
        }
        let rows_f = f32::from(h - 1);
        let cols_f = f32::from(w - 1);
        let row = |v: f32| (v.clamp(0.0, 1.0) * rows_f).round() as u16;
        let col = |v: f32| (v.clamp(0.0, 1.0) * cols_f).round() as u16;

        let mut out = Vec::with_capacity(self.rows * COLS + usize::from(w));
        // The wall: each surviving brick fills its slice of the row.
        for r in 0..self.rows {
            // Drawn at the top edge of the brick's band, not its center: the
            // centering offset is half a brick, which on a tall field rounds
            // the first row down to row 1 and reopens the gap under the rule.
            let y = row(BRICK_H.mul_add(r as f32, WALL_TOP));
            let color = ROW_COLORS[r % ROW_COLORS.len()];
            for c in 0..COLS {
                if !self.bricks[r * COLS + c] {
                    continue;
                }
                let from = col(c as f32 / COLS as f32);
                let to = col((c as f32 + 1.0) / COLS as f32);
                // One blank column between bricks keeps the wall legible.
                for x in from..to.saturating_sub(1).max(from) {
                    out.push(Glyph {
                        x,
                        y,
                        ch: '▬',
                        color,
                    });
                }
            }
        }

        // Paddle. Drawn at half-cell resolution: `paddle_x` is a float that
        // moves continuously, so snapping both ends to whole columns made it
        // jump a full character at a time. Half blocks at the ends let it
        // land between columns, which is what reads as smooth motion.
        let half = self.spec().paddle_half;
        let py = row(PADDLE_Y);
        // `col` rounds, so column `i` covers `[i - 0.5, i + 0.5)`. Shifting by
        // a half puts the ends in a space where column `i` is `[i, i + 1)`,
        // which floor and ceil below can work in — and keeps the paddle
        // aligned with the ball and bricks, which are placed with `col`.
        let paddle = (
            (self.paddle_x - half).clamp(0.0, 1.0).mul_add(cols_f, 0.5),
            (self.paddle_x + half).clamp(0.0, 1.0).mul_add(cols_f, 0.5),
        );
        let mut cap = |x: f32, ch: char| {
            out.push(Glyph {
                x: x as u16,
                y: py,
                ch,
                color: (120, 220, 255),
            });
        };
        // Whole columns the paddle covers end to end.
        let (first_full, last_full) = (paddle.0.ceil(), paddle.1.floor());
        let mut x = first_full;
        while x < last_full {
            cap(x, '█');
            x += 1.0;
        }
        // A left edge landing in the back half of its column fills just that
        // half; likewise the right edge, from the other side.
        if paddle.0.fract() > 0.0 && paddle.0.fract() <= 0.5 {
            cap(paddle.0.floor(), '▐');
        }
        if paddle.1 - last_full >= 0.5 {
            cap(last_full, '▌');
        }

        // Ball — hidden while a banner is up.
        if matches!(
            self.phase,
            Phase::Playing | Phase::Serve { .. } | Phase::Paused
        ) {
            out.push(Glyph {
                x: col(self.ball.0),
                y: row(self.ball.1),
                ch: '●',
                color: (255, 255, 200),
            });
        }
        out
    }

    /// The banner across the middle, if any.
    #[must_use]
    pub fn banner(&self) -> Option<String> {
        match self.phase {
            Phase::LevelUp { .. } => Some(format!("WALL DOWN — LEVEL {}", self.level() + 1)),
            Phase::Paused => Some("PAUSED".to_string()),
            Phase::GameOver => Some(format!("OVER AT {} POINTS — space to restart", self.score)),
            Phase::Won => Some(format!("EVERY WALL CLEARED — {} points", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "level {}/{}  ·  {} points  ·  {} lives  ·  {} bricks",
            self.level(),
            LEVELS.len(),
            self.score,
            self.lives,
            self.remaining()
        )
    }
}

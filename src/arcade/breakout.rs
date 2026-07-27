// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Breakout: knock the wall down without losing the ball.
//!
//! Same contract as the rest of [`crate::arcade`] — a 1x1 normalized field,
//! state advanced only by an injected `dt`, rendering handed out as
//! [`Glyph`]s — so it fills any terminal and replays identically from a seed.

use ratatui::crossterm::event::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};

use super::{CONTACT, GLIDE_MS, Glyph, H_GAIN, MAX_STEP_MS, MIN_H, MIN_W, Phase, Rng, SUB_MS};
use crate::anim::Rgb;

/// Bricks per row. Fixed, so the wall reads the same at every size.
pub const COLS: usize = 14;
/// Where the wall starts and how tall one brick row is, as fractions of the
/// field height.
const WALL_TOP: f32 = 0.06;
const BRICK_H: f32 = 0.05;
/// The paddle's row.
const PADDLE_Y: f32 = 0.94;
/// How fast the paddle travels, in field-widths per second. Comfortably above
/// every ball speed: the ball should beat you on placement, not on pace.
const PADDLE_SPEED: f32 = 1.6;
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
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Left | KeyCode::Char('h' | 'H' | 'a' | 'A') => {
                self.dir = -1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
            KeyCode::Right | KeyCode::Char('l' | 'L' | 'd' | 'D') => {
                self.dir = 1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
            KeyCode::Char('p') => {
                self.phase = match self.phase {
                    Phase::Paused => Phase::Serve { ms_left: 500 },
                    Phase::Playing | Phase::Serve { .. } => Phase::Paused,
                    other => other,
                };
            }
            KeyCode::Char(' ') | KeyCode::Enter => match self.phase {
                Phase::Serve { .. } => self.phase = Phase::Playing,
                Phase::Paused => self.phase = Phase::Serve { ms_left: 500 },
                Phase::GameOver | Phase::Won => *self = Self::new(self.rng.next_u64()),
                _ => {}
            },
            _ => return false,
        }
        true
    }

    /// Feeds one mouse event to the game. Breakout's paddle is horizontal, so
    /// it is the pointer's **column** that matters — `w` is the play width.
    pub fn handle_mouse(&mut self, ev: MouseEvent, w: u16) -> bool {
        match ev.kind {
            MouseEventKind::ScrollUp => {
                self.dir = -1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
            MouseEventKind::ScrollDown => {
                self.dir = 1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
            MouseEventKind::Down(_) | MouseEventKind::Drag(_) | MouseEventKind::Moved => {
                if w < MIN_W {
                    return false;
                }
                let x = f32::from(ev.column) / f32::from(w - 1);
                let half = self.spec().paddle_half;
                self.paddle_x = x.clamp(half, 1.0 - half);
                self.dir = 0.0;
                self.glide_ms = 0;
                self.unpause();
            }
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
            let y = row(BRICK_H.mul_add(r as f32 + 0.5, WALL_TOP));
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

        // Paddle.
        let half = self.spec().paddle_half;
        let py = row(PADDLE_Y);
        for x in col(self.paddle_x - half)..=col(self.paddle_x + half) {
            out.push(Glyph {
                x,
                y: py,
                ch: '█',
                color: (120, 220, 255),
            });
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
            Phase::LevelUp { .. } => Some(format!("MURO ABBATTUTO — LIVELLO {}", self.level() + 1)),
            Phase::Paused => Some("PAUSA".to_string()),
            Phase::GameOver => Some(format!(
                "FINITO A {} PUNTI — spazio per ricominciare",
                self.score
            )),
            Phase::Won => Some(format!("HAI PULITO TUTTI I MURI — {} punti", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "livello {}/{}  ·  {} punti  ·  {} vite  ·  {} mattoni",
            self.level(),
            LEVELS.len(),
            self.score,
            self.lives,
            self.remaining()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyModifiers, MouseButton};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn run(b: &mut Breakout, ms: u64) {
        for _ in 0..(ms / 50) {
            b.step(50);
        }
    }

    #[test]
    fn a_new_game_starts_with_a_full_wall() {
        let b = Breakout::new(1);
        assert_eq!(b.remaining(), LEVELS[0].rows * COLS);
        assert_eq!(b.level(), 1);
    }

    #[test]
    fn the_ball_rides_the_paddle_until_it_is_served() {
        let mut b = Breakout::new(1);
        b.handle_key(key(KeyCode::Right));
        b.step(50);
        assert!(
            (b.ball.0 - b.paddle_x).abs() < f32::EPSILON,
            "ball did not follow the paddle before the serve"
        );
    }

    #[test]
    fn the_ball_reflects_off_the_side_walls() {
        let mut b = Breakout::new(1);
        b.phase = Phase::Playing;
        b.ball = (0.99, 0.5);
        b.vel = (0.8, -0.3);
        b.step(50);
        assert!(b.vel.0 < 0.0, "did not come off the right wall");
        assert!(b.ball.0 <= 1.0);
    }

    #[test]
    fn hitting_a_brick_removes_it_and_scores() {
        let mut b = Breakout::new(1);
        b.phase = Phase::Playing;
        let before = b.remaining();
        // Straight into the middle of the top row.
        b.ball = (0.5, WALL_TOP + BRICK_H * 0.5);
        b.vel = (0.0, -0.5);
        b.step(50);
        assert_eq!(b.remaining(), before - 1, "brick survived");
        assert!(b.score > 0, "no score");
        assert!(b.vel.1 > 0.0, "ball did not bounce off the brick");
    }

    #[test]
    fn the_same_brick_is_never_scored_twice() {
        let mut b = Breakout::new(1);
        b.phase = Phase::Playing;
        b.ball = (0.5, WALL_TOP + BRICK_H * 0.5);
        b.vel = (0.0, -0.01); // barely moving: many sub-steps inside one brick
        b.step(50);
        let after_first = b.score;
        assert_eq!(b.remaining(), LEVELS[0].rows * COLS - 1);
        b.step(50);
        assert_eq!(b.score, after_first, "scored the same brick again");
    }

    #[test]
    fn top_rows_are_worth_more_than_bottom_rows() {
        let hit_row = |r: usize| {
            let mut b = Breakout::new(1);
            b.phase = Phase::Playing;
            #[allow(clippy::cast_precision_loss)]
            let y = BRICK_H.mul_add(r as f32 + 0.5, WALL_TOP);
            b.ball = (0.5, y);
            b.vel = (0.0, -0.5);
            b.step(50);
            b.score
        };
        assert!(hit_row(0) > hit_row(LEVELS[0].rows - 1));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // COLS is a small constant
    fn clearing_the_wall_advances_the_level() {
        let mut b = Breakout::new(2);
        b.phase = Phase::Playing;
        // Leave one brick, then knock it out.
        b.bricks.iter_mut().for_each(|slot| *slot = false);
        b.bricks[0] = true;
        b.ball = (0.5 / COLS as f32, WALL_TOP + BRICK_H * 0.5);
        b.vel = (0.0, -0.5);
        b.step(50);
        assert!(matches!(b.phase(), Phase::LevelUp { .. }));
        run(&mut b, 2000);
        assert_eq!(b.level(), 2);
        assert_eq!(b.remaining(), LEVELS[1].rows * COLS, "wall did not rebuild");
    }

    #[test]
    fn missing_the_ball_costs_a_life_and_three_ends_it() {
        let mut b = Breakout::new(1);
        for expected in (0..LIVES).rev() {
            b.phase = Phase::Playing;
            b.ball = (0.5, 0.99);
            b.vel = (0.0, 0.9);
            // Park the paddle far from the ball so it cannot be saved.
            b.paddle_x = 0.09;
            run(&mut b, 400);
            assert_eq!(b.lives, expected, "life not deducted");
            if expected > 0 {
                assert!(!b.finished());
            }
        }
        assert_eq!(b.phase(), Phase::GameOver);
    }

    #[test]
    fn the_paddle_saves_the_ball_and_steers_it_by_where_it_lands() {
        let bounce = |off: f32| {
            let mut b = Breakout::new(1);
            b.phase = Phase::Playing;
            b.paddle_x = 0.5;
            b.ball = (0.5 + off * LEVELS[0].paddle_half, PADDLE_Y - 0.001);
            b.vel = (0.0, 0.6);
            b.step(50);
            assert!(b.vel.1 < 0.0, "paddle did not return the ball");
            b.vel.0
        };
        let flat = bounce(0.0);
        let edge = bounce(1.0);
        assert!(flat.abs() < 0.05, "center bounce was not flat: {flat}");
        assert!(edge > 0.2, "edge did not steer right: {edge}");
    }

    #[test]
    fn the_paddle_and_the_mouse_stay_on_the_field() {
        let mut b = Breakout::new(1);
        for _ in 0..200 {
            b.handle_key(key(KeyCode::Left));
            b.step(50);
        }
        assert!(b.paddle_x >= LEVELS[0].paddle_half - 0.001);
        for column in 0..200u16 {
            b.handle_mouse(
                MouseEvent {
                    kind: MouseEventKind::Down(MouseButton::Left),
                    column,
                    row: 20,
                    modifiers: KeyModifiers::NONE,
                },
                80,
            );
            let half = LEVELS[0].paddle_half;
            assert!(b.paddle_x >= half - 0.001 && b.paddle_x <= 1.0 - half + 0.001);
        }
    }

    #[test]
    fn the_ball_can_never_tunnel_through_a_brick_row() {
        #[allow(clippy::cast_precision_loss)]
        let dt = SUB_MS as f32 / 1000.0;
        let fastest = LEVELS.iter().map(|l| l.ball_speed).fold(0.0f32, f32::max);
        assert!(
            fastest * dt < BRICK_H,
            "a sub-step moves {} but a brick row is only {BRICK_H} tall",
            fastest * dt
        );
    }

    #[test]
    fn a_long_game_keeps_the_ball_on_the_field() {
        let mut b = Breakout::new(99);
        for _ in 0..3000 {
            b.step(50);
            assert!(
                b.ball.0 >= -0.01 && b.ball.0 <= 1.01,
                "ball left sideways at {}",
                b.ball.0
            );
            if b.finished() {
                break;
            }
        }
    }

    #[test]
    fn the_wall_and_paddle_fill_whatever_screen_they_get() {
        let b = Breakout::new(1);
        assert!(b.glyphs(MIN_W - 1, MIN_H).is_empty());
        for (w, h) in [(MIN_W, MIN_H), (80, 24), (200, 60)] {
            let g = b.glyphs(w, h);
            assert!(!g.is_empty(), "nothing drawn at {w}x{h}");
            assert!(g.iter().all(|g| g.x < w && g.y < h), "spilled at {w}x{h}");
            let widest = g.iter().map(|g| g.x).max().unwrap();
            assert!(widest > w - w / 6, "wall stops at {widest} on a {w} screen");
        }
    }
}

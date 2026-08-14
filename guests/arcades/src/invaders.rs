//! **Ported from plank's `src/arcade/invaders.rs`.** Physics, level table and
//! drawing are verbatim; only the imports and `handle_key` changed, the latter
//! because a guest sees the ABI's key *name* rather than a crossterm
//! `KeyEvent`. Mouse handling is dropped until the host delivers
//! `frame_mouse`.

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Space invaders: hold the line while the fleet walks down at you.
//!
//! Same contract as the rest of [`crate::arcade`] — a 1x1 normalized field,
//! state advanced only by an injected `dt`, rendering handed out as
//! [`Glyph`]s — so it fills any terminal and replays identically from a seed.

#![allow(dead_code)]

use crate::shared::{GLIDE_MS, MIN_H, MIN_W, Phase, SUB_MS};
use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng};

/// Fleet shape.
pub const COLS: usize = 11;
pub const ROWS: usize = 5;
/// Spacing between neighbouring aliens, as fractions of the field. Wide enough
/// that the two-row sprites below never touch, at any terminal size that draws
/// them (see [`Invaders::big_sprites`]).
const SPACING_X: f32 = 0.082;
const SPACING_Y: f32 = 0.13;
/// How far the fleet drops each time it reaches a wall.
const DROP: f32 = 0.055;
/// The ship's row, and the row at which the fleet has overrun you.
const SHIP_Y: f32 = 0.95;
const OVERRUN_Y: f32 = 0.88;
/// Ship travel, in field-widths per second.
const SHIP_SPEED: f32 = 1.1;
/// Shot and bomb travel, in field-heights per second.
const SHOT_SPEED: f32 = 1.5;
const BOMB_SPEED: f32 = 0.5;
/// Minimum gap between shots. Without it a held key is a laser.
const FIRE_COOLDOWN_MS: u64 = 260;
/// How near a shot has to pass to count as a hit. Sized to the drawn sprite:
/// an alien you can see is an alien you can shoot.
const HIT_X: f32 = 0.034;
const HIT_Y: f32 = 0.048;
/// Lives at the start of a game.
const LIVES: u32 = 3;
/// One alien: two poses, each two rows of three cells.
type Sprite = [[&'static str; 2]; 2];

/// Alien sprite and color per row, back rank first.
///
/// Three cells wide and two tall: a single character reads as a speck, and the
/// fleet is the thing you are supposed to be looking at. Each rank gets its own
/// shape so you can tell at a glance how deep you have cut into the formation.
///
/// The two poses alternate as the fleet walks (see [`STRIDE`]) — arms and legs
/// swapping — which is what makes them read as marching creatures rather than
/// as sliding tiles.
const ALIEN_ROWS: [(Sprite, Rgb); ROWS] = [
    ([[" ▄ ", "▝█▘"], [" ▄ ", "▘█▝"]], (255, 120, 120)),
    ([["▗▄▖", "▘▀▝"], ["▗▄▖", "▚▀▞"]], (255, 180, 100)),
    ([["▟▀▙", "▘ ▝"], ["▛▀▜", "▚ ▞"]], (250, 235, 120)),
    ([["▞█▚", "▝ ▘"], ["▚█▞", "▘ ▝"]], (140, 230, 140)),
    ([["▛▀▜", "▚ ▞"], ["▟▀▙", "▘ ▝"]], (130, 200, 255)),
];
/// The single-character stand-ins, for a terminal too narrow for sprites. Two
/// poses each, so even the speck keeps a flicker of the walk.
const ALIEN_SPECKS: [[char; 2]; ROWS] =
    [['▄', '▀'], ['▖', '▗'], ['▙', '▟'], ['▚', '▞'], ['▛', '▜']];
/// The ship, three wide and two tall. It does not animate: it is the one still
/// thing on screen, which is what makes the fleet's motion read.
const SHIP_SPRITE: [&str; 2] = [" ▲ ", "▟█▙"];
/// The bomb's two frames — a falling bomb that tumbles is easier to track.
const BOMB_FRAMES: [char; 2] = ['*', '+'];
/// How far the fleet walks between pose changes, in field-widths.
///
/// Tying the animation to distance rather than to a timer means the aliens
/// walk faster exactly when they march faster, which is the whole charm of the
/// original's accelerating shuffle.
const STRIDE: f32 = 0.022;
/// Sprites need this many columns; below it the fleet falls back to specks.
const SPRITE_MIN_W: u16 = 52;
/// ...and this many rows, so the two-row sprites do not run into each other.
/// Pinned by `sprite_ranks_never_collide`: at this height the rank spacing is
/// just over two rows, which is exactly what a two-row sprite needs.
const SPRITE_MIN_H: u16 = 21;

/// Per-level difficulty.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec {
    /// Fleet march, in field-widths per second, at full strength. The fleet
    /// speeds up as it thins — see [`Invaders::march_speed`].
    pub march: f32,
    /// Average gap between bombs, in milliseconds.
    pub bomb_ms: u64,
    /// How far down the fleet starts.
    pub start_y: f32,
}

/// Five levels: a faster fleet, heavier fire, a shorter runway.
pub const LEVELS: [LevelSpec; 5] = [
    LevelSpec {
        march: 0.055,
        bomb_ms: 1400,
        start_y: 0.10,
    },
    LevelSpec {
        march: 0.075,
        bomb_ms: 1100,
        start_y: 0.13,
    },
    LevelSpec {
        march: 0.095,
        bomb_ms: 850,
        start_y: 0.16,
    },
    LevelSpec {
        march: 0.115,
        bomb_ms: 650,
        start_y: 0.19,
    },
    LevelSpec {
        march: 0.140,
        bomb_ms: 500,
        start_y: 0.22,
    },
];

/// Stamps a sprite centered on `(x, y)`, clipped to a `w` x `h` area.
fn stamp(out: &mut Vec<Glyph>, sprite: &[&str; 2], x: u16, y: u16, color: Rgb, (w, h): (u16, u16)) {
    for (dy, line) in sprite.iter().enumerate() {
        let Some(sy) = u16::try_from(dy)
            .ok()
            .and_then(|dy| y.checked_add(dy))
            .filter(|sy| *sy < h)
        else {
            continue;
        };
        for (dx, ch) in line.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            // The sprite is three wide, so it starts one left of center.
            let Some(sx) = u16::try_from(dx)
                .ok()
                .and_then(|dx| (x + dx).checked_sub(1))
                .filter(|sx| *sx < w)
            else {
                continue;
            };
            out.push(Glyph {
                x: sx,
                y: sy,
                ch,
                color,
            });
        }
    }
}

/// A space invaders game.
#[derive(Debug)]
pub struct Invaders {
    level: usize,
    lives: u32,
    score: u32,
    /// Row-major `ROWS * COLS`; true while the alien is still flying.
    alive: Vec<bool>,
    /// Top-left corner of the fleet.
    fleet: (f32, f32),
    fleet_dir: f32,
    /// Distance walked since the last pose change, and which pose is showing.
    walked: f32,
    pose: usize,
    ship_x: f32,
    dir: f32,
    glide_ms: u64,
    shots: Vec<(f32, f32)>,
    bombs: Vec<(f32, f32)>,
    fire_ms: u64,
    bomb_ms: u64,
    phase: Phase,
    acc_ms: u64,
    rng: Rng,
}

impl Invaders {
    /// Starts a new game at level 1.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut me = Self {
            level: 0,
            lives: LIVES,
            score: 0,
            alive: Vec::new(),
            fleet: (0.0, 0.0),
            fleet_dir: 1.0,
            walked: 0.0,
            pose: 0,
            ship_x: 0.5,
            dir: 0.0,
            glide_ms: 0,
            shots: Vec::new(),
            bombs: Vec::new(),
            fire_ms: 0,
            bomb_ms: 0,
            phase: Phase::Serve { ms_left: 900 },
            acc_ms: 0,
            rng: Rng::new(seed),
        };
        me.launch_fleet();
        me
    }

    /// The level being played, 1-based.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level + 1
    }

    /// Aliens still flying.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.alive.iter().filter(|a| **a).count()
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

    fn launch_fleet(&mut self) {
        self.alive = vec![true; ROWS * COLS];
        #[allow(clippy::cast_precision_loss)] // COLS is a small constant
        let width = SPACING_X * (COLS - 1) as f32;
        self.fleet = ((1.0 - width) / 2.0, self.spec().start_y);
        self.fleet_dir = 1.0;
        self.shots.clear();
        self.bombs.clear();
        self.bomb_ms = self.spec().bomb_ms;
    }

    /// Where an alien sits, whether or not it is still alive.
    #[allow(clippy::cast_precision_loss)] // ROWS/COLS are small constants
    fn alien_at(&self, row: usize, col: usize) -> (f32, f32) {
        (
            SPACING_X.mul_add(col as f32, self.fleet.0),
            SPACING_Y.mul_add(row as f32, self.fleet.1),
        )
    }

    /// The fleet marches faster as it thins — the classic pressure curve, and
    /// the reason the last few aliens are the hard ones.
    #[allow(clippy::cast_precision_loss)] // small counts
    fn march_speed(&self) -> f32 {
        let total = (ROWS * COLS) as f32;
        let left = self.remaining().max(1) as f32;
        self.spec().march * (1.0 + 2.5 * (1.0 - left / total))
    }

    /// Feeds one key to the game.
    pub fn handle_key(&mut self, code: &str) -> bool {
        match code {
            "left" | "h" | "a" => {
                self.dir = -1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
            "right" | "l" | "d" => {
                self.dir = 1.0;
                self.glide_ms = GLIDE_MS;
                self.unpause();
            }
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

    fn aim(&mut self, column: u16, w: u16) {
        if w < MIN_W {
            return;
        }
        self.ship_x = (f32::from(column) / f32::from(w - 1)).clamp(0.02, 0.98);
        self.dir = 0.0;
        self.glide_ms = 0;
    }

    /// Fires, if the cooldown has run out.
    fn fire(&mut self) {
        if self.fire_ms > 0 {
            return;
        }
        self.fire_ms = FIRE_COOLDOWN_MS;
        self.shots.push((self.ship_x, SHIP_Y - 0.02));
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
        self.fire_ms = self.fire_ms.saturating_sub(dt_ms);
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
                        self.launch_fleet();
                        Phase::Serve { ms_left: 900 }
                    }
                    Some(left) => Phase::LevelUp { ms_left: left },
                };
            }
            Phase::Playing => {}
        }
        self.bomb_ms = self.bomb_ms.saturating_sub(dt_ms);
        if self.phase == Phase::Playing && self.bomb_ms == 0 {
            self.drop_bomb();
        }
        self.acc_ms += dt_ms;
        while self.acc_ms >= SUB_MS {
            self.acc_ms -= SUB_MS;
            self.tick();
        }
    }

    /// A random alien on the front rank of its column drops a bomb — the ones
    /// behind it would have to shoot through their own fleet.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // bomb_ms is a few hundred; the jitter is deliberately coarse
    fn drop_bomb(&mut self) {
        let spec = self.spec();
        let mean = spec.bomb_ms as f32;
        self.bomb_ms = self.rng.range(mean * 0.6, mean * 1.4) as u64;
        let mut front: Vec<(usize, usize)> = Vec::with_capacity(COLS);
        for col in 0..COLS {
            if let Some(row) = (0..ROWS).rev().find(|r| self.alive[r * COLS + col]) {
                front.push((row, col));
            }
        }
        if front.is_empty() {
            return;
        }
        let pick = usize::try_from(self.rng.next_u64() % front.len() as u64).unwrap_or(0);
        let (row, col) = front[pick];
        let (x, y) = self.alien_at(row, col);
        self.bombs.push((x, y + 0.02));
    }

    #[allow(clippy::cast_precision_loss)] // SUB_MS is a small constant
    fn tick(&mut self) {
        let dt = SUB_MS as f32 / 1000.0;
        self.ship_x = self
            .dir
            .mul_add(SHIP_SPEED * dt, self.ship_x)
            .clamp(0.02, 0.98);

        if self.phase != Phase::Playing {
            return;
        }

        self.march(dt);
        self.move_shots(dt);
        self.move_bombs(dt);
    }

    /// Walks the fleet sideways, dropping and reversing at the walls.
    fn march(&mut self, dt: f32) {
        let travel = self.march_speed() * dt;
        self.fleet.0 += self.fleet_dir * travel;
        self.walked += travel;
        if self.walked >= STRIDE {
            self.walked %= STRIDE;
            self.pose ^= 1;
        }
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for row in 0..ROWS {
            for col in 0..COLS {
                if !self.alive[row * COLS + col] {
                    continue;
                }
                let (x, _) = self.alien_at(row, col);
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo > hi {
            return; // fleet wiped this sub-step
        }
        if (self.fleet_dir < 0.0 && lo <= 0.02) || (self.fleet_dir > 0.0 && hi >= 0.98) {
            self.fleet_dir = -self.fleet_dir;
            self.fleet.1 += DROP;
            // Reaching the ship's row is an immediate loss, however many
            // lives are left: they have landed.
            if self.lowest_alien() >= OVERRUN_Y {
                self.phase = Phase::GameOver;
            }
        }
    }

    fn lowest_alien(&self) -> f32 {
        (0..ROWS)
            .rev()
            .find(|r| (0..COLS).any(|c| self.alive[r * COLS + c]))
            .map_or(f32::MIN, |r| self.alien_at(r, 0).1)
    }

    /// Moves the player's shots and resolves hits.
    fn move_shots(&mut self, dt: f32) {
        let mut hits: Vec<usize> = Vec::new();
        self.shots.retain_mut(|shot| {
            shot.1 -= SHOT_SPEED * dt;
            shot.1 >= 0.0
        });
        // Collect first, mutate after: the borrow of `alive` and of `shots`
        // cannot overlap.
        let mut spent: Vec<usize> = Vec::new();
        for (i, shot) in self.shots.iter().enumerate() {
            for row in 0..ROWS {
                for col in 0..COLS {
                    if !self.alive[row * COLS + col] {
                        continue;
                    }
                    let (x, y) = self.alien_at(row, col);
                    if (shot.0 - x).abs() <= HIT_X && (shot.1 - y).abs() <= HIT_Y {
                        hits.push(row * COLS + col);
                        spent.push(i);
                        break;
                    }
                }
                if spent.last() == Some(&i) {
                    break;
                }
            }
        }
        for idx in hits {
            if std::mem::replace(&mut self.alive[idx], false) {
                // Back ranks are further and worth more.
                self.score += u32::try_from(ROWS - idx / COLS).unwrap_or(1) * 10;
            }
        }
        for i in spent.into_iter().rev() {
            if i < self.shots.len() {
                self.shots.remove(i);
            }
        }
        if self.remaining() == 0 {
            self.phase = if self.level + 1 >= LEVELS.len() {
                Phase::Won
            } else {
                Phase::LevelUp { ms_left: 1400 }
            };
        }
    }

    /// Moves the fleet's bombs and resolves hits on the ship.
    fn move_bombs(&mut self, dt: f32) {
        let ship_x = self.ship_x;
        let mut struck = false;
        self.bombs.retain_mut(|bomb| {
            bomb.1 += BOMB_SPEED * dt;
            if bomb.1 >= SHIP_Y && (bomb.0 - ship_x).abs() <= HIT_X {
                struck = true;
                return false;
            }
            bomb.1 <= 1.0
        });
        if struck {
            self.lives = self.lives.saturating_sub(1);
            self.phase = if self.lives == 0 {
                Phase::GameOver
            } else {
                self.bombs.clear();
                Phase::Serve { ms_left: 700 }
            };
        }
    }

    /// Whether the area is big enough for the two-row sprites.
    ///
    /// Below this the fleet is drawn as single characters. Eleven three-wide
    /// sprites need real estate; on a small terminal, overlapping mush would be
    /// worse than specks.
    const fn big_sprites(w: u16, h: u16) -> bool {
        w >= SPRITE_MIN_W && h >= SPRITE_MIN_H
    }

    /// Paints the fleet, ship, shots and bombs across the whole area.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    // normalized coords scaled into a bounded, positive range; the body is one
    // straight-line pass over fleet, shots, bombs and ship
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w < MIN_W || h < MIN_H {
            return Vec::new();
        }
        let rows_f = f32::from(h - 1);
        let cols_f = f32::from(w - 1);
        let row = |v: f32| (v.clamp(0.0, 1.0) * rows_f).round() as u16;
        let col = |v: f32| (v.clamp(0.0, 1.0) * cols_f).round() as u16;
        let big = Self::big_sprites(w, h);

        let mut out = Vec::with_capacity(ROWS * COLS * 6 + 16);
        for (r, (sprite, color)) in ALIEN_ROWS.iter().enumerate() {
            let pose = &sprite[self.pose];
            for c in 0..COLS {
                if !self.alive[r * COLS + c] {
                    continue;
                }
                let (x, y) = self.alien_at(r, c);
                if !(0.0..=1.0).contains(&y) {
                    continue;
                }
                if big {
                    stamp(&mut out, pose, col(x), row(y), *color, (w, h));
                } else {
                    out.push(Glyph {
                        x: col(x),
                        y: row(y),
                        ch: ALIEN_SPECKS[r][self.pose],
                        color: *color,
                    });
                }
            }
        }
        for shot in &self.shots {
            out.push(Glyph {
                x: col(shot.0),
                y: row(shot.1),
                ch: '│',
                color: (200, 255, 220),
            });
        }
        for bomb in &self.bombs {
            out.push(Glyph {
                x: col(bomb.0),
                y: row(bomb.1),
                ch: BOMB_FRAMES[self.pose],
                color: (255, 140, 160),
            });
        }
        if !matches!(self.phase, Phase::GameOver) {
            let (x, y) = (col(self.ship_x), row(SHIP_Y));
            if big {
                // The ship is two rows tall and sits on the bottom row, so it
                // is stamped one row up from its collision line.
                stamp(
                    &mut out,
                    &SHIP_SPRITE,
                    x,
                    y.saturating_sub(1),
                    (140, 235, 255),
                    (w, h),
                );
            } else {
                out.push(Glyph {
                    x,
                    y,
                    ch: '▲',
                    color: (140, 235, 255),
                });
            }
        }
        out
    }

    /// The banner across the middle, if any.
    #[must_use]
    pub fn banner(&self) -> Option<String> {
        match self.phase {
            Phase::LevelUp { .. } => Some(format!("WAVE REPELLED — LEVEL {}", self.level() + 1)),
            Phase::Paused => Some("PAUSED".to_string()),
            Phase::GameOver => Some(format!(
                "THEY GOT YOU AT {} POINTS — space to restart",
                self.score
            )),
            Phase::Won => Some(format!("PLANET SAVED — {} points", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "level {}/{}  ·  {} points  ·  {} lives  ·  {} aliens",
            self.level(),
            LEVELS.len(),
            self.score,
            self.lives,
            self.remaining()
        )
    }
}

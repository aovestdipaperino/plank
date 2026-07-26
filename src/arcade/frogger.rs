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

use ratatui::crossterm::event::{KeyCode, KeyEvent, MouseEvent, MouseEventKind};

use super::{GLIDE_MS, Glyph, MAX_STEP_MS, MIN_H, MIN_W, Phase, Rng, SUB_MS};
use crate::anim::Rgb;

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
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Up | KeyCode::Char('k' | 'K' | 'w' | 'W') => self.hop(0, -1),
            KeyCode::Down | KeyCode::Char('j' | 'J' | 's' | 'S') => self.hop(0, 1),
            KeyCode::Left | KeyCode::Char('h' | 'H' | 'a' | 'A') => self.hop(-1, 0),
            KeyCode::Right | KeyCode::Char('l' | 'L' | 'd' | 'D') => self.hop(1, 0),
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

    /// Feeds one mouse event: a click hops toward wherever you clicked.
    pub fn handle_mouse(&mut self, ev: MouseEvent, w: u16, h: u16) -> bool {
        if !matches!(
            ev.kind,
            MouseEventKind::Down(_) | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            return false;
        }
        if ev.kind == MouseEventKind::ScrollUp {
            self.hop(0, -1);
            return true;
        }
        if ev.kind == MouseEventKind::ScrollDown {
            self.hop(0, 1);
            return true;
        }
        if w < MIN_W || h < MIN_H {
            return false;
        }
        // Click above the frog to go up, below to go down, beside to sidestep.
        let target_row = f32::from(ev.row) / f32::from(h - 1) * (ROWS - 1) as f32;
        let target_col = f32::from(ev.column) / f32::from(w - 1);
        let dy = target_row - self.frog.1 as f32;
        if dy.abs() >= 0.5 {
            self.hop(0, if dy < 0.0 { -1 } else { 1 });
        } else {
            self.hop(if target_col < self.frog.0 { -1 } else { 1 }, 0);
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
            Phase::LevelUp { .. } => Some(format!("TUTTE A CASA — LIVELLO {}", self.level() + 1)),
            Phase::Paused => Some("PAUSA".to_string()),
            Phase::GameOver => Some(format!(
                "SPIACCICATA A {} PUNTI — spazio per ricominciare",
                self.score
            )),
            Phase::Won => Some(format!("TUTTE LE RANE A CASA — {} punti", self.score)),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        format!(
            "livello {}/{}  ·  {} punti  ·  {} vite  ·  {} tane libere",
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

    fn run(g: &mut Frogger, ms: u64) {
        for _ in 0..(ms / 50) {
            g.step(50);
        }
    }

    fn playing(seed: u64) -> Frogger {
        let mut g = Frogger::new(seed);
        g.phase = Phase::Playing;
        g
    }

    /// Empties every lane so a test can place exactly what it means to test.
    fn clear_lanes(g: &mut Frogger) {
        for lane in &mut g.lanes {
            lane.len = 0.0;
            lane.speed = 0.0;
        }
    }

    #[test]
    fn a_new_game_starts_on_the_near_bank_with_every_home_open() {
        let g = Frogger::new(1);
        assert_eq!(g.frog.1, START_ROW);
        assert_eq!(g.remaining(), HOMES);
        assert_eq!(g.level(), 1);
    }

    #[test]
    fn the_banks_and_the_median_are_always_safe() {
        let mut g = playing(1);
        for row in [START_ROW, MEDIAN] {
            g.frog = (0.5, row);
            g.lives = LIVES;
            run(&mut g, 2000);
            assert_eq!(g.lives, LIVES, "row {row} killed the frog");
        }
    }

    #[test]
    fn hopping_up_scores_and_hopping_is_bounded_by_the_board() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        let before = g.score;
        g.hop(0, -1);
        assert_eq!(g.frog.1, START_ROW - 1);
        assert!(g.score > before, "no score for progress");
        for _ in 0..40 {
            g.hop(0, 1);
        }
        assert_eq!(g.frog.1, ROWS - 1, "hopped off the bottom");
    }

    #[test]
    fn a_car_squashes_the_frog() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        let row = ROAD.start;
        g.lanes[row].len = 1.0;
        g.lanes[row].gap = 1.0;
        g.lanes[row].offset = 0.0;
        g.frog = (0.5, row);
        g.step(50);
        assert_eq!(g.lives, LIVES - 1, "the car missed");
        assert_eq!(g.frog.1, START_ROW, "the frog did not go back");
    }

    #[test]
    fn open_water_drowns_the_frog() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        g.frog = (0.5, RIVER.start);
        g.step(50);
        assert_eq!(g.lives, LIVES - 1, "the frog walked on water");
    }

    #[test]
    fn a_log_carries_the_frog_along() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        let row = RIVER.start;
        g.lanes[row] = Lane {
            speed: 0.2,
            gap: 1.0,
            len: 1.0,
            offset: 0.0,
            rideable: true,
            turtles: false,
        };
        g.frog = (0.4, row);
        run(&mut g, 500);
        assert_eq!(g.lives, LIVES, "the frog drowned on a log");
        assert!(g.frog.0 > 0.45, "the log did not carry it: {}", g.frog.0);
    }

    #[test]
    fn being_carried_off_the_edge_drowns_the_frog() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        let row = RIVER.start;
        g.lanes[row] = Lane {
            speed: 0.9,
            gap: 1.0,
            len: 1.0,
            offset: 0.0,
            rideable: true,
            turtles: false,
        };
        g.frog = (0.95, row);
        run(&mut g, 1000);
        assert_eq!(g.lives, LIVES - 1, "riding off the board was survivable");
    }

    #[test]
    fn reaching_a_bay_fills_it_and_scores() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        g.frog = ((0.5) / HOMES as f32, RIVER.start);
        g.hop(0, -1);
        assert!(g.homes[0], "the bay did not fill");
        assert!(g.score >= 50);
        assert_eq!(g.frog.1, START_ROW, "did not go back for the next one");
    }

    #[test]
    fn landing_between_the_bays_is_a_miss() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        // Right on the divider between bay 0 and bay 1.
        g.frog = (1.0 / HOMES as f32, RIVER.start);
        g.hop(0, -1);
        assert!(g.homes.iter().all(|h| !*h), "a miss filled a bay");
        assert_eq!(g.lives, LIVES - 1);
    }

    #[test]
    fn a_bay_that_is_already_taken_cannot_be_taken_twice() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        g.homes[0] = true;
        g.frog = ((0.5) / HOMES as f32, RIVER.start);
        g.hop(0, -1);
        assert_eq!(g.remaining(), HOMES - 1, "the same bay counted twice");
        assert_eq!(g.lives, LIVES - 1);
    }

    #[test]
    fn filling_every_bay_advances_the_level() {
        let mut g = playing(2);
        clear_lanes(&mut g);
        for slot in 0..HOMES {
            g.phase = Phase::Playing;
            g.frog = ((slot as f32 + 0.5) / HOMES as f32, RIVER.start);
            g.hop(0, -1);
        }
        assert!(matches!(g.phase(), Phase::LevelUp { .. }));
        run(&mut g, 2000);
        assert_eq!(g.level(), 2);
        assert_eq!(g.remaining(), HOMES, "the bays did not reopen");
    }

    #[test]
    fn filling_every_bay_on_the_last_level_wins() {
        let mut g = playing(2);
        g.level = LEVELS.len() - 1;
        clear_lanes(&mut g);
        for slot in 0..HOMES {
            g.phase = Phase::Playing;
            g.frog = ((slot as f32 + 0.5) / HOMES as f32, RIVER.start);
            g.hop(0, -1);
        }
        assert_eq!(g.phase(), Phase::Won);
    }

    #[test]
    fn three_deaths_end_the_game() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        for expected in (0..LIVES).rev() {
            g.phase = Phase::Playing;
            g.frog = (0.5, RIVER.start);
            g.step(50);
            assert_eq!(g.lives, expected);
        }
        assert_eq!(g.phase(), Phase::GameOver);
    }

    #[test]
    fn lanes_alternate_direction_so_the_board_can_be_read() {
        let g = Frogger::new(1);
        for row in ROAD {
            let here = g.lanes[row].speed;
            let next = g.lanes[row + 1].speed;
            if ROAD.contains(&(row + 1)) {
                assert!(here * next < 0.0, "rows {row} and {} run together", row + 1);
            }
        }
    }

    #[test]
    fn higher_levels_run_faster_and_tighter() {
        let mut easy = Frogger::new(1);
        let mut hard = Frogger::new(1);
        hard.level = LEVELS.len() - 1;
        hard.build_lanes();
        easy.build_lanes();
        let row = ROAD.start;
        assert!(hard.lanes[row].speed.abs() > easy.lanes[row].speed.abs());
        assert!(hard.lanes[row].gap < easy.lanes[row].gap);
    }

    #[test]
    fn the_mouse_hops_toward_the_click() {
        let mut g = playing(1);
        clear_lanes(&mut g);
        let click = |column, row| MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };
        // Click near the top: the frog should hop up a row.
        g.handle_mouse(click(40, 0), 80, 24);
        assert_eq!(g.frog.1, START_ROW - 1);
        // Scroll up hops too.
        g.handle_mouse(
            MouseEvent {
                kind: MouseEventKind::ScrollUp,
                column: 40,
                row: 10,
                modifiers: KeyModifiers::NONE,
            },
            80,
            24,
        );
        assert_eq!(g.frog.1, START_ROW - 2);
    }

    #[test]
    fn a_replay_from_the_same_seed_is_identical() {
        let mut a = Frogger::new(31);
        let mut b = Frogger::new(31);
        for i in 0..500 {
            if i % 13 == 0 {
                a.handle_key(key(KeyCode::Up));
                b.handle_key(key(KeyCode::Up));
            }
            a.step(50);
            b.step(50);
        }
        assert_eq!(a.score, b.score);
        assert_eq!(a.lives, b.lives);
        assert!((a.frog.0 - b.frog.0).abs() < f32::EPSILON);
    }

    #[test]
    fn a_long_game_keeps_the_frog_on_the_board() {
        let mut g = playing(77);
        for _ in 0..3000 {
            g.step(50);
            assert!(g.frog.1 < ROWS, "frog left the board");
            if g.finished() {
                break;
            }
        }
    }

    #[test]
    fn the_board_fills_whatever_screen_it_gets() {
        let g = Frogger::new(1);
        assert!(g.glyphs(MIN_W - 1, MIN_H).is_empty());
        for (w, h) in [(MIN_W, MIN_H), (80, 24), (200, 60)] {
            let out = g.glyphs(w, h);
            assert!(!out.is_empty(), "nothing drawn at {w}x{h}");
            assert!(out.iter().all(|g| g.x < w && g.y < h), "spilled at {w}x{h}");
            assert!(out.iter().any(|g| g.y == 0), "no home row at {w}x{h}");
            assert!(out.iter().any(|g| g.y == h - 1), "no start bank at {w}x{h}");
        }
    }
}

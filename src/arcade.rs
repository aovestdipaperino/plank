// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Hidden arcade: five games, a matrix rain, a perspective starfield and two
//! minions.
//!
//! The games, the rain and the minions are easter eggs behind `/pelota` and
//! friends. None is listed in [`crate::config::usage`] nor offered by the
//! completion popup — that is the point — but all are known commands, so the
//! dispatcher runs them instead of forwarding the line to the model. The three
//! ambient screens are also what the idle screensaver puts up, which is not
//! gated at all.
//!
//! # Design
//!
//! This module follows the rule [`crate::anim`] sets for every animated piece
//! of the UI: **state advances only through an injected time delta**, and
//! rendering is a pure function of that state. Nothing here reads a clock, and
//! the only randomness comes from a seeded [`Rng`], so a whole rally replays
//! identically given the same seed and the same `dt` sequence — which is what
//! makes the physics unit-testable without a running terminal.
//!
//! Rendering stops at [`Glyph`]: this module says *what* to paint and where,
//! [`crate::tui::draw_arcade`] turns that into ratatui cells. That mirrors the
//! `/config` modal, where [`crate::configform`] owns the state and `tui.rs`
//! draws its rows.
//!
//! # Front-end boundary
//!
//! Motion is Ratatui-only, as `anim.rs` requires. The plain line REPL is told
//! the games need the TUI, rather than growing a second raw-mode game loop
//! alongside the first.

pub mod breakout;
pub mod matrix;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use crate::anim::Rgb;

/// One character to paint, in the arcade's own coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glyph {
    /// Column, relative to the drawing area.
    pub x: u16,
    /// Row, relative to the drawing area.
    pub y: u16,
    /// The character itself.
    pub ch: char,
    /// Foreground color.
    pub color: Rgb,
}

/// What a key press asks the caller to do with the open arcade modal.
#[derive(Debug)]
pub enum Outcome {
    /// Stay open; the arcade absorbed the key.
    Stay,
    /// Close, optionally leaving this line in the scrollback.
    Close(Option<String>),
}

/// Longest `dt` a single [`Arcade::step`] will integrate, in milliseconds.
///
/// The UI loop measures real elapsed time, so the first frame after the modal
/// opens — or after the terminal was suspended — can carry a large delta that
/// would teleport the ball straight through a paddle. Clamping keeps the
/// simulation continuous at the cost of running slightly slow after a stall.
pub(crate) const MAX_STEP_MS: u64 = 100;

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

/// A seeded xorshift64\* generator.
///
/// plank has no `rand` dependency and does not want one for this: an explicit
/// seed is what lets a test replay a rally exactly.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator. Zero is replaced (xorshift is stuck at zero).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Next raw 64-bit value.
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next value in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)] // 24 bits of mantissa, by construction
    pub fn next_f32(&mut self) -> f32 {
        // Take the top 24 bits: exactly the f32 mantissa width, so the divide
        // is lossless and the result never rounds up to 1.0.
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }

    /// Next value in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

// ---------------------------------------------------------------------------
// Pelota
// ---------------------------------------------------------------------------

/// Smallest terminal the field will render into.
pub const MIN_W: u16 = 30;
pub const MIN_H: u16 = 9;
/// The player's paddle plane, as a fraction of the field width.
const PLAYER_X: f32 = 0.035;
/// The CPU's paddle plane.
const CPU_X: f32 = 0.965;
/// Half the CPU's paddle, as a fraction of the field height. Constant across
/// levels: the difficulty dial is its speed and its aim, never its size.
const CPU_HALF: f32 = 0.10;
/// How fast the player's paddle travels, in field-heights per second.
/// Deliberately above every CPU speed in [`LEVELS`], and fast enough to cross
/// the field inside the ball's flight time at every level: losing should be
/// about reading the ball, never about being outrun.
const PLAYER_SPEED: f32 = 1.25;
/// Horizontal stretch of the ball's velocity. A character cell is about twice
/// as tall as it is wide, so an unstretched ball looks like it crawls sideways.
pub(crate) const H_GAIN: f32 = 1.15;
/// How long a key press keeps the paddle moving, in milliseconds. Terminal key
/// repeat re-arms this, so holding a key gives continuous travel while a single
/// tap gives a short nudge.
pub(crate) const GLIDE_MS: u64 = 190;
/// Physics sub-step, in milliseconds. Fixed and small so a fast ball cannot
/// tunnel through a paddle between frames.
pub(crate) const SUB_MS: u64 = 8;
/// Steepest angle the ball can leave a paddle at, in radians (~60°).
const MAX_BOUNCE: f32 = 1.05;
/// How much the paddle shrinks while charged.
///
/// Moving with Shift held drops into a charged stance: the paddle becomes a
/// third of its length — much easier to whiff, and the CPU gets the point when
/// you do — but a hit that lands leaves at [`POWER_GAIN`] times the speed.
/// Pure risk for pure reward, and the reason to take it is that the CPU cannot
/// cover the field in a third of the flight time.
const CHARGE_SHRINK: f32 = 3.0;
/// Speed multiplier on a charged hit. The CPU's own return is built from the
/// level's normal speed, so the boost lasts exactly one crossing — long enough
/// to win the point, not long enough to make the rally unplayable.
const POWER_GAIN: f32 = 3.0;
/// Contact tolerance around a paddle, as a fraction of the field height.
pub(crate) const CONTACT: f32 = 0.012;

/// Per-level difficulty, in **normalized field units**: the whole match is
/// played on a 1x1 field and mapped onto the terminal only when drawing, so
/// pelota fills whatever screen it is given and plays identically on all of
/// them.
///
/// The CPU is tuned by how fast it may move and how wrong it is allowed to be
/// — never by cheating on the ball. What keeps every level winnable is the
/// ratio between its speed and the ball's flight time: even at level 5 it can
/// only cover about two thirds of the field per crossing, so a flat drive to
/// the far side still beats it.
#[derive(Debug, Clone, Copy)]
pub struct LevelSpec {
    /// Ball speed, in field-heights per second.
    pub ball_speed: f32,
    /// Ceiling on the CPU paddle's travel, in field-heights per second.
    pub cpu_speed: f32,
    /// How far off the ball the CPU aims, re-rolled every rally.
    pub cpu_error: f32,
    /// Half the player's paddle, as a fraction of the field height.
    pub paddle_half: f32,
    /// Points needed to clear the level.
    pub target: u32,
}

/// The five levels. Ball speed climbs faster than CPU speed, so the CPU's
/// reachable fraction of the field per crossing actually *falls* as levels go
/// up — it gets harder to pass, never impossible.
pub const LEVELS: [LevelSpec; 5] = [
    LevelSpec {
        ball_speed: 0.45,
        cpu_speed: 0.30,
        cpu_error: 0.13,
        paddle_half: 0.16,
        target: 3,
    },
    LevelSpec {
        ball_speed: 0.55,
        cpu_speed: 0.40,
        cpu_error: 0.10,
        paddle_half: 0.135,
        target: 4,
    },
    LevelSpec {
        ball_speed: 0.66,
        cpu_speed: 0.52,
        cpu_error: 0.07,
        paddle_half: 0.11,
        target: 5,
    },
    LevelSpec {
        ball_speed: 0.78,
        cpu_speed: 0.64,
        cpu_error: 0.045,
        paddle_half: 0.09,
        target: 6,
    },
    LevelSpec {
        ball_speed: 0.92,
        cpu_speed: 0.78,
        cpu_error: 0.02,
        paddle_half: 0.075,
        target: 7,
    },
];

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

/// A pelota match against the CPU.
#[derive(Debug)]
pub struct Pelota {
    level: usize,
    score: (u32, u32),
    phase: Phase,
    ball: (f32, f32),
    vel: (f32, f32),
    player_y: f32,
    player_dir: f32,
    glide_ms: u64,
    /// Time left on the charged stance (see [`CHARGE_SHRINK`]). Armed by
    /// moving with Shift held and decayed like `glide_ms`.
    charge_ms: u64,
    cpu_y: f32,
    /// The CPU's aiming error for this rally, in field-heights.
    cpu_bias: f32,
    /// Leftover time not yet consumed by a whole physics sub-step.
    acc_ms: u64,
    rng: Rng,
}

impl Pelota {
    /// Starts a new match at level 1.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut me = Self {
            level: 0,
            score: (0, 0),
            phase: Phase::Serve { ms_left: 900 },
            ball: (0.5, 0.5),
            vel: (0.0, 0.0),
            player_y: 0.5,
            player_dir: 0.0,
            glide_ms: 0,
            charge_ms: 0,
            cpu_y: 0.5,
            cpu_bias: 0.0,
            acc_ms: 0,
            rng: Rng::new(seed),
        };
        me.park(1.0);
        me
    }

    /// The level being played, 1-based.
    #[must_use]
    pub const fn level(&self) -> usize {
        self.level + 1
    }

    /// Player and CPU score in the current level.
    #[must_use]
    pub const fn score(&self) -> (u32, u32) {
        self.score
    }

    /// Current phase.
    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    /// Whether the match has ended, either way.
    #[must_use]
    pub const fn finished(&self) -> bool {
        matches!(self.phase, Phase::GameOver | Phase::Won)
    }

    /// Ends the match, for tests that need a finished game to hand around.
    #[cfg(test)]
    pub(crate) const fn force_finished(&mut self) {
        self.phase = Phase::GameOver;
    }

    const fn spec(&self) -> LevelSpec {
        LEVELS[self.level]
    }

    /// Whether the charged stance is armed right now.
    #[must_use]
    pub const fn charged(&self) -> bool {
        self.charge_ms > 0
    }

    /// Half the player's paddle, as a fraction of the field height — a third
    /// of that while charged.
    fn player_half(&self) -> f32 {
        if self.charged() {
            self.spec().paddle_half / CHARGE_SHRINK
        } else {
            self.spec().paddle_half
        }
    }

    /// Parks the ball at center, aimed at `dir` (+1 toward the CPU).
    fn park(&mut self, dir: f32) {
        let s = self.spec().ball_speed;
        let angle = self.rng.range(-0.4, 0.4);
        self.ball = (0.5, 0.5);
        self.vel = (H_GAIN * s * angle.cos() * dir, s * angle.sin());
        self.cpu_bias = self
            .rng
            .range(-self.spec().cpu_error, self.spec().cpu_error);
    }

    /// Feeds one key to the match.
    ///
    /// Shift arms the charged stance. A terminal never reports key *release*,
    /// so "holding Shift" is only observable while it rides along with another
    /// key — which is exactly right here: the stance lasts as long as you keep
    /// moving with Shift down, and lapses on its own the moment you stop.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            // Shift+letter arrives as the uppercase char, so both cases move.
            KeyCode::Up | KeyCode::Char('k' | 'K' | 'w' | 'W') => {
                self.player_dir = -1.0;
                self.glide_ms = GLIDE_MS;
                self.set_charge(shift);
                self.unpause();
            }
            KeyCode::Down | KeyCode::Char('j' | 'J' | 's' | 'S') => {
                self.player_dir = 1.0;
                self.glide_ms = GLIDE_MS;
                self.set_charge(shift);
                self.unpause();
            }
            KeyCode::Char('p') => {
                self.phase = match self.phase {
                    Phase::Paused => Phase::Serve { ms_left: 600 },
                    Phase::Playing | Phase::Serve { .. } => Phase::Paused,
                    other => other,
                };
            }
            KeyCode::Char(' ') | KeyCode::Enter => match self.phase {
                // Space skips the countdown, and restarts a finished match.
                Phase::Serve { .. } => self.phase = Phase::Playing,
                Phase::Paused => self.phase = Phase::Serve { ms_left: 600 },
                Phase::GameOver | Phase::Won => *self = Self::new(self.rng.next_u64()),
                _ => {}
            },
            _ => return false,
        }
        true
    }

    /// Feeds one mouse event to the match.
    ///
    /// The wheel and a trackpad's two-finger swipe both arrive as scroll
    /// events, and nudge the paddle like a key press. A click or a drag places
    /// it outright at the pointer's row — `h` is the play area's height, the
    /// same mapping [`Self::glyphs`] uses in reverse.
    ///
    /// Free hover (no button down) only arrives when the caller has asked the
    /// terminal for any-motion reporting; it is handled the same way as a drag.
    pub fn handle_mouse(&mut self, ev: MouseEvent, h: u16) -> bool {
        match ev.kind {
            MouseEventKind::ScrollUp => {
                self.player_dir = -1.0;
                self.glide_ms = GLIDE_MS;
                self.set_charge(ev.modifiers.contains(KeyModifiers::SHIFT));
                self.unpause();
            }
            MouseEventKind::ScrollDown => {
                self.player_dir = 1.0;
                self.glide_ms = GLIDE_MS;
                self.set_charge(ev.modifiers.contains(KeyModifiers::SHIFT));
                self.unpause();
            }
            MouseEventKind::Down(_) | MouseEventKind::Drag(_) | MouseEventKind::Moved => {
                if h < MIN_H {
                    return false;
                }
                // Invert the render mapping: rows 1..h-2 hold the field.
                let rows = f32::from(h - 3);
                let y = (f32::from(ev.row) - 1.0) / rows;
                let half = self.player_half();
                self.player_y = y.clamp(half, 1.0 - half);
                // Pointing is aiming, not gliding: no residual drift.
                self.player_dir = 0.0;
                self.glide_ms = 0;
                self.set_charge(ev.modifiers.contains(KeyModifiers::SHIFT));
                self.unpause();
            }
            _ => return false,
        }
        true
    }

    fn unpause(&mut self) {
        if self.phase == Phase::Paused {
            self.phase = Phase::Serve { ms_left: 600 };
        }
    }

    /// Arms or drops the charged stance for this movement.
    const fn set_charge(&mut self, on: bool) {
        self.charge_ms = if on { GLIDE_MS } else { 0 };
    }

    /// Advances the match by `dt_ms`, in fixed sub-steps.
    pub fn step(&mut self, dt_ms: u64) {
        let dt_ms = dt_ms.min(MAX_STEP_MS);
        self.glide_ms = self.glide_ms.saturating_sub(dt_ms);
        self.charge_ms = self.charge_ms.saturating_sub(dt_ms);
        if self.glide_ms == 0 {
            self.player_dir = 0.0;
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
                        self.score = (0, 0);
                        self.park(-1.0);
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

    /// One fixed physics sub-step.
    #[allow(clippy::cast_precision_loss)] // SUB_MS is a small constant
    fn tick(&mut self) {
        let dt = SUB_MS as f32 / 1000.0;
        let half = self.player_half();
        self.player_y = self
            .player_dir
            .mul_add(PLAYER_SPEED * dt, self.player_y)
            .clamp(half, 1.0 - half);

        self.move_cpu(dt);

        // The paddles move even between rallies; the ball does not.
        if self.phase != Phase::Playing {
            return;
        }

        self.ball.0 += self.vel.0 * dt;
        self.ball.1 += self.vel.1 * dt;

        // Walls.
        if self.ball.1 < 0.0 {
            self.ball.1 = -self.ball.1;
            self.vel.1 = -self.vel.1;
        } else if self.ball.1 > 1.0 {
            self.ball.1 = 2.0 - self.ball.1;
            self.vel.1 = -self.vel.1;
        }

        // Paddles.
        if self.vel.0 < 0.0 && self.ball.0 <= PLAYER_X {
            let reach = half + CONTACT;
            if (self.ball.1 - self.player_y).abs() <= reach {
                let off = (self.ball.1 - self.player_y) / reach;
                let power = if self.charged() { POWER_GAIN } else { 1.0 };
                self.bounce(PLAYER_X, off, 1.0, power);
                // A fresh aiming error each time the ball comes back keeps the
                // CPU from settling into a groove during a long rally.
                self.cpu_bias = self
                    .rng
                    .range(-self.spec().cpu_error, self.spec().cpu_error);
            } else if self.ball.0 < 0.0 {
                self.concede();
            }
        } else if self.vel.0 > 0.0 && self.ball.0 >= CPU_X {
            let reach = CPU_HALF + CONTACT;
            if (self.ball.1 - self.cpu_y).abs() <= reach {
                let off = (self.ball.1 - self.cpu_y) / reach;
                // No power for the CPU: its return resets the ball to the
                // level's speed, which is what bounds a charged hit to one
                // crossing.
                self.bounce(CPU_X, off, -1.0, 1.0);
            } else if self.ball.0 > 1.0 {
                self.score_for_cpu();
            }
        }
    }

    /// Reflects the ball off a paddle at `x`, steering by where it hit.
    ///
    /// `off` is the contact point relative to the paddle center, in `[-1, 1]`:
    /// the center sends the ball back flat, the tips send it away steeply.
    /// This is the whole skill of the game — and the only way to beat level 5.
    /// `power` is [`POWER_GAIN`] on a charged hit and 1 otherwise.
    fn bounce(&mut self, x: f32, off: f32, dir: f32, power: f32) {
        let s = self.spec().ball_speed * power;
        let angle = off.clamp(-1.0, 1.0) * MAX_BOUNCE;
        self.ball.0 = x;
        self.vel = (H_GAIN * s * angle.cos() * dir, s * angle.sin());
    }

    /// The player missed.
    fn concede(&mut self) {
        self.score.1 += 1;
        if self.score.1 >= self.spec().target {
            self.phase = Phase::GameOver;
            return;
        }
        self.park(-1.0);
        self.phase = Phase::Serve { ms_left: 700 };
    }

    /// The CPU missed.
    fn score_for_cpu(&mut self) {
        self.score.0 += 1;
        if self.score.0 >= self.spec().target {
            self.phase = if self.level + 1 >= LEVELS.len() {
                Phase::Won
            } else {
                Phase::LevelUp { ms_left: 1400 }
            };
            return;
        }
        self.park(1.0);
        self.phase = Phase::Serve { ms_left: 700 };
    }

    /// Moves the CPU paddle toward where it thinks the ball will arrive.
    ///
    /// It only chases while the ball is coming at it, drifts back to center
    /// otherwise, aims at the ball plus [`Self::cpu_bias`], and is capped at
    /// `cpu_speed`. Those four limits together are what make it feel like an
    /// opponent rather than a wall.
    fn move_cpu(&mut self, dt: f32) {
        let target = if self.vel.0 > 0.0 && self.phase == Phase::Playing {
            self.ball.1 + self.cpu_bias
        } else {
            0.5
        };
        let delta = target - self.cpu_y;
        let travel = self.spec().cpu_speed * dt;
        self.cpu_y = if delta.abs() <= travel {
            target
        } else {
            travel.mul_add(delta.signum(), self.cpu_y)
        };
        self.cpu_y = self.cpu_y.clamp(CPU_HALF, 1.0 - CPU_HALF);
    }

    /// Paints the field across the whole `w` x `h` area.
    ///
    /// The normalized field is stretched to fill the terminal — walls on the
    /// first and last row, paddles on the outer columns — so pelota is as big
    /// as the window is. Returns nothing when the terminal is below
    /// [`MIN_W`] x [`MIN_H`]; the caller shows the "too small" hint instead.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // normalized coords scaled into a bounded, positive range
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w < MIN_W || h < MIN_H {
            return Vec::new();
        }
        // Rows 0 and h-1 are the walls; the field occupies everything between.
        let rows = f32::from(h - 3);
        let cols = f32::from(w - 1);
        let row = |v: f32| 1 + (v.clamp(0.0, 1.0) * rows).round() as u16;
        let col = |v: f32| (v.clamp(0.0, 1.0) * cols).round() as u16;

        let mut out = Vec::with_capacity(usize::from(w) * 2 + usize::from(h) * 3);
        let mut put = |x: u16, y: u16, ch: char, color: Rgb| out.push(Glyph { x, y, ch, color });

        // Walls and the center line.
        for x in 0..w {
            put(x, 0, '─', (70, 78, 110));
            put(x, h - 1, '─', (70, 78, 110));
        }
        for y in (2..h - 1).step_by(2) {
            put(w / 2, y, '┊', (52, 58, 82));
        }

        // Paddles. The charged stance is visibly different — you are about to
        // hit three times as hard with a third of the paddle, and you should
        // be able to see that without reading the scoreboard.
        let half = self.player_half();
        let (player_ch, player_color) = if self.charged() {
            ('▓', (255, 210, 90))
        } else {
            ('█', (120, 220, 255))
        };
        for y in row(self.player_y - half)..=row(self.player_y + half) {
            put(col(PLAYER_X), y, player_ch, player_color);
        }
        for y in row(self.cpu_y - CPU_HALF)..=row(self.cpu_y + CPU_HALF) {
            put(col(CPU_X), y, '█', (255, 150, 120));
        }

        // Ball — hidden while a banner is up.
        if matches!(
            self.phase,
            Phase::Playing | Phase::Serve { .. } | Phase::Paused
        ) {
            put(col(self.ball.0), row(self.ball.1), '●', (255, 255, 200));
        }
        out
    }

    /// The banner across the middle of the field, if any.
    #[must_use]
    pub fn banner(&self) -> Option<String> {
        match self.phase {
            Phase::LevelUp { .. } => Some(format!("LEVEL {} CLEARED", self.level())),
            Phase::Paused => Some("PAUSED".to_string()),
            Phase::GameOver => Some(format!("LOST ON LEVEL {} — space to restart", self.level())),
            Phase::Won => Some("EVERY LEVEL BEATEN".to_string()),
            _ => None,
        }
    }

    /// The scoreboard line.
    #[must_use]
    pub fn hud(&self) -> String {
        let spec = self.spec();
        format!(
            "level {}/{}  ·  you {} — {} cpu  (to {})",
            self.level(),
            LEVELS.len(),
            self.score.0,
            self.score.1,
            spec.target
        )
    }
}

// ---------------------------------------------------------------------------
// Sound
// ---------------------------------------------------------------------------

/// What just happened, for the optional blips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    /// Something was hit, killed, or bounced.
    Hit,
    /// A life was lost, or the game ended.
    Lose,
    /// A level was cleared, or a home was filled.
    Win,
}

/// Optional sound, off unless the command was given the `sound` argument.
///
/// **This adds nothing to the binary**, which is the constraint it was built
/// under: it writes `BEL` (`\x07`) to the terminal and stops there. Real audio
/// would mean a synthesis crate and a system audio dependency — megabytes, for
/// an easter egg.
///
/// What that buys, and what it does not: BEL has no pitch and no length, so
/// every cue is the same blip and the only variation possible is how many.
/// Terminals configured for a visual bell will flash instead, which during a
/// game is worse than silence — hence off by default, and hence the toggle.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sound {
    on: bool,
}

impl Sound {
    /// Whether `arg` asks for sound.
    #[must_use]
    pub fn wanted(arg: &str) -> bool {
        arg.split_whitespace()
            .any(|w| w.eq_ignore_ascii_case("sound"))
    }

    /// Whether blips are currently on.
    #[must_use]
    pub const fn is_on(&self) -> bool {
        self.on
    }

    /// Turns blips on or off.
    pub const fn set(&mut self, on: bool) {
        self.on = on;
    }

    /// Flips blips, and reports the new state.
    pub const fn toggle(&mut self) -> bool {
        self.on = !self.on;
        self.on
    }

    /// The bytes a cue writes to the terminal — empty when sound is off.
    ///
    /// Kept as a pure function so the cue mapping is testable without a
    /// terminal to listen to.
    #[must_use]
    pub const fn bytes(self, cue: Cue) -> &'static [u8] {
        if !self.on {
            return b"";
        }
        match cue {
            // Count is the only dimension BEL gives us, so it carries the
            // meaning: one blip for a hit, two for a loss, three for a win.
            Cue::Hit => b"\x07",
            Cue::Lose => b"\x07\x07",
            Cue::Win => b"\x07\x07\x07",
        }
    }

    /// Sounds a cue on the terminal. Best-effort: a write that fails is not
    /// worth interrupting a game over.
    pub fn play(self, cue: Cue) {
        use std::io::Write as _;
        let bytes = self.bytes(cue);
        if bytes.is_empty() {
            return;
        }
        let mut out = std::io::stdout();
        let _ = out.write_all(bytes);
        let _ = out.flush();
    }
}

// ---------------------------------------------------------------------------
// The modal
// ---------------------------------------------------------------------------

/// Which easter egg is open.
#[derive(Debug)]
pub enum Game {
    /// Falling glyphs: `/matrix`, and one of the screensaver's faces.
    Matrix(matrix::Rain),
    /// `/pelota`
    Pelota(Box<Pelota>),
    /// `/breakout`
    Breakout(Box<breakout::Breakout>),
}

impl Game {
    /// Whether this is weather rather than a game: something to watch, with no
    /// score, no ending and no state a player could come back to.
    ///
    /// The cabinet uses this to decide *not* to park one — resumed rain is
    /// indistinguishable from fresh rain and a resumed skit from a fresh skit,
    /// so both the slot it would take and the "resumed where you left off"
    /// notice would be lies.
    #[must_use]
    pub const fn ambient(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }

    /// Whether this game is over, either way. The ambient ones never end.
    #[must_use]
    pub fn finished(&self) -> bool {
        match self {
            Self::Matrix(_) => false,
            Self::Pelota(p) => p.finished(),
            Self::Breakout(b) => b.finished(),
        }
    }

    /// The line to leave in the scrollback when this game leaves the screen.
    #[must_use]
    pub fn closing_line(&self) -> Option<String> {
        match self {
            // Nothing happened that a scrollback line could report.
            Self::Matrix(_) => None,
            Self::Pelota(p) => Some(format!(
                "pelota: level {} — {} to {}{}",
                p.level(),
                p.score().0,
                p.score().1,
                resume_hint(p.finished(), "/pelota")
            )),
            Self::Breakout(b) => Some(format!(
                "breakout: level {} — {} bricks standing{}",
                b.level(),
                b.remaining(),
                resume_hint(b.finished(), "/breakout")
            )),
        }
    }
}

/// A game's state boiled down to what the blips care about. Score only rises,
/// lives only fall, and the level only advances — so a plain comparison across
/// one step says exactly what just happened, with no bookkeeping inside the
/// games themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Vitals {
    score: u32,
    lives: u32,
    level: usize,
    phase: Option<Phase>,
}

impl Game {
    fn vitals(&self) -> Vitals {
        match self {
            // No score, no lives, no levels: nothing that could earn a blip,
            // which is what keeps the ambient screens silent even with sound on.
            Self::Matrix(_) => Vitals {
                score: 0,
                lives: 0,
                level: 0,
                phase: None,
            },
            Self::Pelota(p) => Vitals {
                // Either side scoring is an event worth hearing.
                score: p.score().0 + p.score().1,
                lives: 0,
                level: p.level(),
                phase: Some(p.phase()),
            },
            Self::Breakout(b) => Vitals {
                score: b.score(),
                lives: b.lives(),
                level: b.level(),
                phase: Some(b.phase()),
            },
        }
    }
}

/// The cue a step deserves, given the vitals either side of it.
///
/// Ordered by what matters most: an ending outranks a level, which outranks a
/// lost life, which outranks a point. Only one blip per step, so a busy frame
/// does not turn into a rattle.
fn cue_for(before: Vitals, after: Vitals) -> Option<Cue> {
    if before.phase != after.phase {
        match after.phase {
            Some(Phase::Won) => return Some(Cue::Win),
            Some(Phase::GameOver) => return Some(Cue::Lose),
            _ => {}
        }
    }
    if after.level > before.level {
        return Some(Cue::Win);
    }
    if after.lives < before.lives {
        return Some(Cue::Lose);
    }
    if after.score > before.score {
        return Some(Cue::Hit);
    }
    None
}

/// Tells the player how to get back in, which differs depending on whether the
/// game they just left is still resumable.
fn resume_hint(finished: bool, cmd: &str) -> String {
    if finished {
        format!(" · {cmd} for a new game")
    } else {
        format!(" · {cmd} resumes, {cmd} new restarts")
    }
}

/// The arcade cabinet: at most one game on screen, the rest parked mid-play.
///
/// The TUI loops hold one of these for the whole session, the way they hold the
/// `/btw` panel — which is what lets a game survive being closed, a turn
/// starting, and being opened again. While a game is on screen it owns every
/// key, exactly like the `/config` modal.
#[derive(Debug, Default)]
pub struct Arcade {
    /// The game currently on screen, if any.
    open: Option<Game>,
    /// Which command the open game came from, so it goes back to its own slot.
    open_slot: usize,
    /// Games left mid-play, one slot per command in [`Self::COMMANDS`].
    /// Reopening that command resumes from here instead of starting over.
    parked: [Option<Game>; Self::COMMANDS.len()],
    /// Paint over the live UI instead of blanking it.
    ///
    /// A terminal cell holds one character and one pair of colors — there is
    /// no alpha to composite with, so real transparency is not on the table.
    /// What *is*: every one of these games is sparse (the rain touches maybe
    /// a quarter of the cells), so leaving the untouched cells alone and
    /// dimming what is underneath reads as a translucent layer. Switched on
    /// when the arcade opens over a running turn, so the model's output stays
    /// legible behind the glyphs.
    pub translucent: bool,
    /// Optional blips. Off unless a command was given the `sound` argument.
    pub sound: Sound,
    /// True while the open ambient face is the screensaver rather than a game.
    /// It has no command and therefore no parking slot, and any key dismisses
    /// it instead of steering.
    screensaver: bool,
}

/// How long the TUI sits idle before the screensaver comes up
/// (`ui.screensaver`).
///
/// A short fixed menu rather than a free-form number: the useful range is
/// small, and a cycling `/config` row beats typing minutes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScreensaverDelay {
    /// One minute — the default.
    #[default]
    M1,
    /// Two minutes.
    M2,
    /// Five minutes.
    M5,
    /// Never: no screensaver.
    Never,
}

/// Which ambient screen the idle screensaver puts up (`ui.screensaverFace`).
///
/// Separate from [`ScreensaverDelay`], which says *when*: the two answer
/// different questions and a user who wants the rain at two minutes should not
/// have to spell that as one combined value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ScreensaverFace {
    /// Falling glyphs — the default.
    #[default]
    Matrix,
    /// Whichever the seed lands on, decided afresh each time the screensaver
    /// opens. This was the behaviour before the setting existed, and it is
    /// worth keeping reachable: coming back to an idle terminal and finding a
    /// *different* screen is a small pleasure, and no face is doing anything
    /// you could be interrupted in the middle of.
    Random,
}

impl ScreensaverFace {
    /// The settings spellings of every built-in face, in cycle order.
    ///
    /// `random` is last because it is a choice among the others, and a picker
    /// that offers it first invites choosing it without seeing what it picks
    /// between.
    pub const NAMES: [&'static str; 2] = ["matrix", "random"];

    /// How many concrete faces plank itself ships.
    ///
    /// One: the matrix rain, which is the *default* screensaver and the reason
    /// it stayed in the binary when the rest became plugins. Used to weigh the
    /// random rotation, where every face — built-in or contributed — gets one
    /// slot.
    pub const BUILT_IN: usize = 1;

    /// The settings-file spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::Random => "random",
        }
    }

    /// Parses a settings value. `rain` and `stars` are accepted as the names
    /// people reach for first, and `either` for the draw.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "matrix" | "rain" => Some(Self::Matrix),
            "random" | "either" => Some(Self::Random),
            _ => None,
        }
    }

    /// The next value in the `/config` cycle.
    #[must_use]
    pub const fn cycle(self) -> Self {
        match self {
            Self::Matrix => Self::Random,
            Self::Random => Self::Matrix,
        }
    }

    /// Resolves [`Random`](Self::Random) against `seed`, so callers only ever
    /// deal with a concrete face.
    ///
    /// Through the Rng rather than a bare bit test: `arcade_seed` mixes the
    /// wall clock, whose low bit merely alternates second by second.
    #[must_use]
    fn resolve(self, _seed: u64) -> Self {
        // One built-in face left, so `Random` has nothing to choose between
        // *here*. It still means something: the rotation it feeds mixes this
        // face with every screensaver a plugin contributes, and that choice is
        // made by the caller, which is the only place that knows what is
        // installed.
        match self {
            Self::Random | Self::Matrix => Self::Matrix,
        }
    }
}

impl ScreensaverDelay {
    /// The settings-file spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::M1 => "1m",
            Self::M2 => "2m",
            Self::M5 => "5m",
            Self::Never => "never",
        }
    }

    /// Parses a settings value. Bare minute counts are accepted alongside the
    /// `1m` spellings, and `off`/`false`/`0` mean never.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1m" | "1" => Some(Self::M1),
            "2m" | "2" => Some(Self::M2),
            "5m" | "5" => Some(Self::M5),
            "never" | "off" | "false" | "0" => Some(Self::Never),
            _ => None,
        }
    }

    /// The idle stretch before the screensaver, or `None` when it is off.
    #[must_use]
    pub const fn duration(self) -> Option<std::time::Duration> {
        let mins = match self {
            Self::M1 => 1,
            Self::M2 => 2,
            Self::M5 => 5,
            Self::Never => return None,
        };
        Some(std::time::Duration::from_secs(mins * 60))
    }

    /// The next value in the `/config` cycle.
    #[must_use]
    pub const fn cycle(self) -> Self {
        match self {
            Self::M1 => Self::M2,
            Self::M2 => Self::M5,
            Self::M5 => Self::Never,
            Self::Never => Self::M1,
        }
    }
}

/// Whether the easter eggs exist this run (`ui.easterEggs`, on by default).
///
/// Off is stronger than hidden: the commands stop being known, so the line goes
/// to the model like any other unrecognized one. Every entry point that could
/// open a game checks this, not just the completion path — a flag that only hid
/// them would still leave them reachable by typing.
#[must_use]
pub fn enabled() -> bool {
    crate::settings::active().ui.easter_eggs
}

/// The arcade command `line` starts with, if any and if they exist at all —
/// `/pelota new` included.
#[must_use]
pub fn command_of(line: &str) -> Option<&'static str> {
    if !enabled() {
        return None;
    }
    Arcade::COMMANDS
        .into_iter()
        .find(|cmd| crate::config::slash_command_with_args(line, cmd))
}

impl Arcade {
    /// Every command this module answers to. Kept next to [`Self::open`] so the
    /// dispatcher's list and the constructors cannot drift apart.
    /// The commands plank itself answers.
    ///
    /// Three, since the rest became plugins: pelota and breakout because
    /// breakout is what the download screen draws and pelota shares its
    /// physics, and the rain because it is the default screensaver. Anything
    /// else a user types comes from an installed component.
    pub const COMMANDS: [&'static str; 3] = ["/pelota", "/breakout", "/matrix"];
    /// The argument that throws the parked game away and deals a new one.
    pub const NEW_ARGS: [&'static str; 2] = ["new", "reset"];

    /// An empty cabinet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a game is on screen right now.
    #[must_use]
    pub const fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Whether `arg` asks for a fresh game rather than the parked one.
    ///
    /// Word-wise, not whole-string, because the arguments compose:
    /// `/breakout new sound` has to mean both.
    #[must_use]
    pub fn wants_new(arg: &str) -> bool {
        arg.split_whitespace()
            .any(|w| Self::NEW_ARGS.iter().any(|a| a.eq_ignore_ascii_case(w)))
    }

    /// Puts a game on screen: the one parked under `cmd` if there is one, a
    /// fresh one otherwise. `fresh` discards whatever was parked.
    ///
    /// Returns false — changing nothing — when `cmd` is not one of ours.
    pub fn open(&mut self, cmd: &str, fresh: bool, seed: u64) -> bool {
        let Some(slot) = Self::COMMANDS.iter().position(|c| *c == cmd) else {
            return false;
        };
        self.park();
        // `take` either way: on a fresh deal the parked game is *discarded*,
        // not shelved, or the next plain `/pelota` would resume the very game
        // the player just asked to be rid of.
        let resumed = self.parked[slot].take().filter(|_| !fresh);
        self.open = Some(resumed.unwrap_or_else(|| Self::deal(cmd, seed)));
        self.open_slot = slot;
        true
    }

    /// Whether reopening `cmd` right now would resume rather than start over.
    #[must_use]
    pub fn has_parked(&self, cmd: &str) -> bool {
        Self::COMMANDS
            .iter()
            .position(|c| *c == cmd)
            .is_some_and(|slot| self.parked[slot].is_some())
    }

    /// A brand new game for `cmd`.
    fn deal(cmd: &str, seed: u64) -> Game {
        match cmd {
            "/pelota" => Game::Pelota(Box::new(Pelota::new(seed))),
            "/breakout" => Game::Breakout(Box::new(breakout::Breakout::new(seed))),
            // `open` has already rejected commands that are not ours, so the
            // rain is the only thing left to be.
            _ => Self::rain(seed),
        }
    }

    /// A field of falling glyphs. It needs no size: everything vertical is a
    /// fraction of the height and the columns outnumber any terminal's.
    fn rain(seed: u64) -> Game {
        Game::Matrix(matrix::Rain::new(seed))
    }

    /// Opens an ambient screen as a screensaver: full screen rather than
    /// translucent, dismissed by any key or mouse event rather than played.
    ///
    /// Not a command and not an easter egg — this is what `ui.screensaver`
    /// runs after an idle stretch, so it stays available with
    /// `ui.easterEggs` off.
    ///
    /// Which face comes up is [`ScreensaverFace`](crate::settings::UiSettings)
    /// (`ui.screensaverFace`), defaulting to the rain;
    /// [`ScreensaverFace::Random`] restores the draw this used to do
    /// unconditionally.
    pub fn open_screensaver(&mut self, seed: u64) {
        self.open_screensaver_as(crate::settings::active().ui.screensaver_face, seed);
    }

    /// [`open_screensaver`](Self::open_screensaver) with the face passed in, so
    /// a test can drive it without touching the process-wide settings.
    ///
    /// `face` is still taken, and still resolved, although one built-in face
    /// is left and the answer is always the rain: the parameter is what a
    /// caller *asked* for, and dropping it would make adding a second built-in
    /// face a signature change rather than a match arm.
    pub fn open_screensaver_as(&mut self, face: ScreensaverFace, seed: u64) {
        self.park();
        let _ = face.resolve(seed);
        self.open = Some(Self::rain(seed));
        self.screensaver = true;
        self.translucent = false;
    }

    /// Whether what is on screen is the screensaver rather than a game.
    #[must_use]
    pub const fn is_screensaver(&self) -> bool {
        self.screensaver
    }

    /// Takes the open game off screen and parks it for next time.
    ///
    /// A game that has already ended is dropped rather than parked: resuming
    /// onto a game-over banner is not resuming, and the next `/breakout` should
    /// deal a new wall. So is an [ambient](Game::ambient) one, which has no
    /// state worth coming back to.
    fn park(&mut self) {
        let Some(game) = self.open.take() else {
            return;
        };
        // The screensaver came from idleness, not a command: there is no slot
        // to park it in, and resuming it would defeat the point.
        if self.screensaver {
            self.screensaver = false;
            return;
        }
        if !game.finished() && !game.ambient() {
            self.parked[self.open_slot] = Some(game);
        }
    }

    /// Closes the open game, parking it, and reports what to leave in the log.
    pub fn close(&mut self) -> Option<String> {
        let line = self.open.as_ref().map(Game::closing_line);
        self.park();
        line.flatten()
    }

    /// Opens over live output: translucent, so the UI underneath stays read.
    pub const fn veil(&mut self) {
        self.translucent = true;
    }

    /// Advances the open easter egg, sounding whatever the step earned.
    pub fn step(&mut self, dt_ms: u64) {
        let before = self.open.as_ref().map(Game::vitals);
        self.advance(dt_ms);
        if !self.sound.is_on() {
            return;
        }
        let after = self.open.as_ref().map(Game::vitals);
        if let (Some(before), Some(after)) = (before, after)
            && let Some(cue) = cue_for(before, after)
        {
            self.sound.play(cue);
        }
    }

    fn advance(&mut self, dt_ms: u64) {
        let Some(game) = self.open.as_mut() else {
            return;
        };
        match game {
            Game::Matrix(m) => m.step(dt_ms),
            Game::Pelota(p) => p.step(dt_ms),
            Game::Breakout(b) => b.step(dt_ms),
        }
    }

    /// Characters to paint over a `w` x `h` area.
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        let Some(game) = self.open.as_ref() else {
            return Vec::new();
        };
        match game {
            Game::Matrix(m) => m.glyphs(w, h),
            Game::Pelota(p) => p.glyphs(w, h),
            Game::Breakout(b) => b.glyphs(w, h),
        }
    }

    /// A centered banner, if the state calls for one.
    #[must_use]
    pub fn banner(&self, w: u16, h: u16) -> Option<String> {
        let game = self.open.as_ref()?;
        // The ambient screens fill whatever they are given and have nothing to
        // announce, so they never carry a banner — not even the size warning.
        if game.ambient() {
            return None;
        }
        if w < MIN_W || h < MIN_H {
            return Some(format!("serve un terminale di almeno {MIN_W}x{MIN_H}"));
        }
        match game {
            Game::Matrix(_) => None,
            Game::Pelota(p) => p.banner(),
            Game::Breakout(b) => b.banner(),
        }
    }

    /// How to leave, kept out of [`Self::footer`] so the renderer can anchor it
    /// to the right edge. On a narrow terminal the hint line is the first thing
    /// to be truncated, and "how do I get out of this" must not be what is lost.
    pub const EXIT_HINT: &'static str = "Esc/q quit";

    /// The status and key hints along the bottom, without the exit hint.
    #[must_use]
    pub fn footer(&self) -> String {
        let Some(game) = self.open.as_ref() else {
            return String::new();
        };
        let veil = if self.translucent { "opaque" } else { "veiled" };
        let blips = if self.sound.is_on() { "mute" } else { "sound" };
        match game {
            Game::Matrix(m) => format!(
                "↑↓/wheel speed ({:.2}×) · c glyphs ({}) · t {veil} · b {blips}",
                m.speed(),
                m.charset().label()
            ),
            Game::Pelota(p) => {
                let charge = if p.charged() {
                    "⚡CHARGED"
                } else {
                    "⇧ charge"
                };
                format!(
                    "{} · ↑↓/mouse move · {charge} · p pause · t {veil} · b {blips}",
                    p.hud()
                )
            }
            Game::Breakout(b) => format!(
                "{} · ←→/mouse move · space serve · p pause · t {veil} · b {blips}",
                b.hud()
            ),
        }
    }

    /// Feeds one mouse event to the open easter egg.
    ///
    /// `w` and `h` are the play area's size, needed to turn an absolute screen
    /// position into a field one — pelota's paddle tracks the row, breakout's
    /// and the invaders' ship track the column. Wheel and trackpad arrive as
    /// scroll events; click and drag arrive as button events with a position.
    pub fn handle_mouse(&mut self, ev: MouseEvent, w: u16, h: u16) -> Outcome {
        let Some(game) = self.open.as_mut() else {
            return Outcome::Stay;
        };
        match game {
            Game::Matrix(m) => match ev.kind {
                MouseEventKind::ScrollUp => m.scale_speed(1.3),
                MouseEventKind::ScrollDown => m.scale_speed(1.0 / 1.3),
                _ => {}
            },
            Game::Pelota(p) => {
                p.handle_mouse(ev, h);
            }
            Game::Breakout(b) => {
                b.handle_mouse(ev, w);
            }
        }
        Outcome::Stay
    }

    /// Feeds one key to the open easter egg.
    ///
    /// [`Outcome::Close`] means the game has already been parked: reopening its
    /// command resumes exactly here.
    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        if !self.is_open() {
            return Outcome::Stay;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Outcome::Close(self.close());
        }
        // A screensaver is dismissed, not played: the first key of whatever
        // the user came back to type puts the UI back.
        if self.screensaver {
            return Outcome::Close(self.close());
        }
        // `t` veils or unveils the layer, whichever game is up. `s` would be a
        // natural toggle for sound but half these games steer with it, so the
        // blips are bound to `b` instead.
        if key.code == KeyCode::Char('t') {
            self.translucent = !self.translucent;
            return Outcome::Stay;
        }
        if key.code == KeyCode::Char('b') {
            self.sound.toggle();
            return Outcome::Stay;
        }
        // `q` is free in every one of these games — none of them steer with it
        // — so it can quit alongside Esc. `k`/`j`/`h`/`l` cannot.
        let quit = matches!(key.code, KeyCode::Esc | KeyCode::Char('q' | 'Q'));
        if quit {
            return Outcome::Close(self.close());
        }
        let Some(game) = self.open.as_mut() else {
            return Outcome::Stay;
        };
        match game {
            Game::Matrix(m) => match key.code {
                KeyCode::Up | KeyCode::Char('k') => m.scale_speed(1.3),
                KeyCode::Down | KeyCode::Char('j') => m.scale_speed(1.0 / 1.3),
                // `c` re-letters the rain: the katakana are the point, but a
                // font without them draws boxes, and this is the way out.
                KeyCode::Char('c') => m.cycle_charset(),
                _ => {}
            },
            Game::Pelota(p) => {
                p.handle_key(key);
            }
            Game::Breakout(b) => {
                b.handle_key(key);
            }
        }
        Outcome::Stay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::MouseButton;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Runs `ms` of simulation in 50 ms frames, the TUI's animation tick.
    fn run(p: &mut Pelota, ms: u64) {
        for _ in 0..(ms / 50) {
            p.step(50);
        }
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let v = a.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} out of range");
            assert!((v - b.next_f32()).abs() < f32::EPSILON);
        }
    }

    /// Steering keys must never be read as "quit" — losing a rally to a
    /// mis-bound key is the worst possible bug in a game. Pelota and breakout
    /// are what is left in the binary; the ported games carry the same test
    /// against their own component.
    #[test]
    fn steering_keys_never_quit() {
        for (cmd, keys) in [
            ("/pelota", ['k', 'j', 'w', 's']),
            ("/breakout", ['h', 'l', 'a', 'd']),
        ] {
            let mut a = Arcade::new();
            a.open(cmd, false, 1);
            for k in keys {
                assert!(
                    matches!(a.handle_key(key(KeyCode::Char(k))), Outcome::Stay),
                    "{cmd} quit on {k}"
                );
            }
        }
    }

    /// Switching games parks the one you left and brings it back untouched.
    #[test]
    fn every_game_keeps_its_own_slot() {
        let mut a = Arcade::new();
        a.open("/breakout", false, 1);
        for _ in 0..100 {
            a.step(50);
        }
        let bricks = match a.open.as_ref() {
            Some(Game::Breakout(b)) => b.remaining(),
            _ => panic!(),
        };
        // Switch to another game without closing first.
        a.open("/pelota", false, 2);
        assert!(matches!(a.open.as_ref(), Some(Game::Pelota(_))));
        assert!(a.has_parked("/breakout"), "breakout was not parked");
        a.open("/breakout", false, 3);
        match a.open.as_ref() {
            Some(Game::Breakout(b)) => assert_eq!(b.remaining(), bricks),
            _ => panic!("breakout did not come back"),
        }
        assert!(a.has_parked("/pelota"), "pelota was not parked");
    }

    #[test]
    fn the_closing_line_says_how_to_get_back_in() {
        let mut a = Arcade::new();
        a.open("/breakout", false, 1);
        let line = a.close().expect("no closing line");
        assert!(line.contains("/breakout resumes"), "{line}");

        a.open("/pelota", false, 1);
        match a.open.as_mut() {
            Some(Game::Pelota(p)) => p.force_finished(),
            _ => panic!(),
        }
        let line = a.close().expect("no closing line");
        assert!(line.contains("new game"), "{line}");
    }

    /// `s` steers in pelota; it must not also be the mute key.
    #[test]
    fn the_mute_key_does_not_collide_with_a_steering_key() {
        for cmd in ["/pelota", "/breakout"] {
            let mut a = Arcade::new();
            a.open(cmd, false, 1);
            let before = a.sound.is_on();
            a.handle_key(key(KeyCode::Char('s')));
            assert_eq!(a.sound.is_on(), before, "{cmd}: s muted instead of moving");
        }
    }

    /// Every spelling the settings file accepts maps back to its canonical
    /// name, so `/config` writing what it read cannot drift.
    #[test]
    fn the_face_setting_round_trips_and_cycles() {
        for face in [ScreensaverFace::Matrix, ScreensaverFace::Random] {
            assert_eq!(ScreensaverFace::parse(face.as_str()), Some(face));
        }
        assert_eq!(
            ScreensaverFace::parse("rain"),
            Some(ScreensaverFace::Matrix)
        );
        // Two faces, so the cycle is a toggle and must still come home.
        assert_eq!(
            ScreensaverFace::Matrix.cycle().cycle(),
            ScreensaverFace::Matrix
        );
        // The names the picker offers are the names the parser takes.
        for name in ScreensaverFace::NAMES {
            assert!(ScreensaverFace::parse(name).is_some(), "{name}");
        }
    }

    /// With one built-in face left, `random` resolves to it — but the constant
    /// the rotation weighs by must agree, or contributed faces get the wrong
    /// share.
    #[test]
    fn random_resolves_to_the_one_built_in_face() {
        for seed in 0..10 {
            assert_eq!(
                ScreensaverFace::Random.resolve(seed),
                ScreensaverFace::Matrix
            );
        }
        assert_eq!(ScreensaverFace::BUILT_IN, 1);
    }

    #[test]
    fn rng_survives_a_zero_seed() {
        let mut r = Rng::new(0);
        assert_ne!(r.next_u64(), 0);
    }

    #[test]
    fn ball_reflects_off_the_top_wall() {
        let mut p = Pelota::new(1);
        p.phase = Phase::Playing;
        p.ball = (0.5, 0.01);
        p.vel = (0.4, -0.5);
        p.step(50);
        assert!(p.vel.1 > 0.0, "vertical velocity did not flip");
        assert!(p.ball.1 >= 0.0, "ball went through the wall");
    }

    #[test]
    fn ball_reflects_off_the_bottom_wall() {
        let mut p = Pelota::new(1);
        p.phase = Phase::Playing;
        p.ball = (0.5, 0.99);
        p.vel = (0.4, 0.5);
        p.step(50);
        assert!(p.vel.1 < 0.0);
        assert!(p.ball.1 <= 1.0);
    }

    #[test]
    fn hitting_the_paddle_edge_steers_more_than_the_center() {
        let mut flat = Pelota::new(1);
        flat.phase = Phase::Playing;
        flat.player_y = 0.5;
        flat.ball = (PLAYER_X + 0.002, 0.5);
        flat.vel = (-0.6, 0.0);
        flat.step(50);

        let mut steep = Pelota::new(1);
        steep.phase = Phase::Playing;
        steep.player_y = 0.5;
        // Contact at the very tip of the paddle.
        steep.ball = (PLAYER_X + 0.002, 0.5 + steep.player_half());
        steep.vel = (-0.6, 0.0);
        steep.step(50);

        assert!(flat.vel.0 > 0.0 && steep.vel.0 > 0.0, "ball did not bounce");
        assert!(
            steep.vel.1.abs() > flat.vel.1.abs() + 0.1,
            "edge hit did not steer: flat {} vs edge {}",
            flat.vel.1,
            steep.vel.1
        );
    }

    #[test]
    fn missing_the_ball_gives_the_point_to_the_cpu() {
        let mut p = Pelota::new(1);
        p.phase = Phase::Playing;
        p.player_y = 0.15;
        p.ball = (0.04, 0.85);
        p.vel = (-0.9, 0.0);
        run(&mut p, 500);
        assert_eq!(p.score(), (0, 1));
    }

    #[test]
    fn the_cpu_paddle_never_exceeds_its_level_speed() {
        for (i, spec) in LEVELS.iter().enumerate() {
            let mut p = Pelota::new(9);
            p.level = i;
            p.phase = Phase::Playing;
            p.cpu_y = 0.2;
            p.cpu_bias = 0.0;
            p.ball = (0.5, 0.9);
            p.vel = (0.6, 0.0);
            let before = p.cpu_y;
            p.step(100);
            let moved = (p.cpu_y - before).abs();
            // 100 ms of travel, plus a rounding cushion.
            assert!(
                moved <= spec.cpu_speed.mul_add(0.1, 0.001),
                "level {} moved {moved}, cap is {}",
                i + 1,
                spec.cpu_speed * 0.1
            );
        }
    }

    /// The CPU must never be able to cover the whole field while the ball
    /// crosses it — that is what keeps every level passable rather than a wall.
    #[test]
    fn every_level_leaves_the_cpu_short_of_the_full_field() {
        for (i, spec) in LEVELS.iter().enumerate() {
            let transit = (CPU_X - PLAYER_X) / (H_GAIN * spec.ball_speed);
            let reach = spec.cpu_speed * transit;
            assert!(
                reach < 0.85,
                "level {} lets the cpu cover {reach} of the field",
                i + 1
            );
            // And the player must be able to cover it, or losing is unfair.
            assert!(
                PLAYER_SPEED * transit > 1.0,
                "level {} outruns the player",
                i + 1
            );
        }
    }

    #[test]
    fn the_cpu_drifts_back_to_center_when_the_ball_is_leaving() {
        let mut p = Pelota::new(2);
        p.phase = Phase::Playing;
        p.cpu_y = 0.2;
        p.ball = (0.5, 0.2);
        // Ball heading away from the CPU: it should stop chasing.
        p.vel = (-0.6, 0.0);
        run(&mut p, 1500);
        assert!(
            (p.cpu_y - 0.5).abs() < 0.05,
            "cpu parked at {} instead of center",
            p.cpu_y
        );
    }

    #[test]
    fn clearing_the_target_advances_the_level_and_resets_the_score() {
        let mut p = Pelota::new(4);
        p.phase = Phase::Playing;
        p.score = (LEVELS[0].target - 1, 1);
        p.cpu_y = 0.1;
        p.ball = (0.99, 0.9);
        p.vel = (0.9, 0.0);
        run(&mut p, 300);
        assert!(matches!(p.phase(), Phase::LevelUp { .. }));
        run(&mut p, 2000);
        assert_eq!(p.level(), 2);
        assert_eq!(p.score(), (0, 0));
    }

    #[test]
    fn losing_the_target_ends_the_match() {
        let mut p = Pelota::new(4);
        p.phase = Phase::Playing;
        p.score = (0, LEVELS[0].target - 1);
        p.player_y = 0.15;
        p.ball = (0.04, 0.85);
        p.vel = (-0.9, 0.0);
        run(&mut p, 300);
        assert_eq!(p.phase(), Phase::GameOver);
        assert!(p.finished());
    }

    #[test]
    fn clearing_the_last_level_wins_the_match() {
        let mut p = Pelota::new(4);
        p.level = LEVELS.len() - 1;
        p.phase = Phase::Playing;
        p.score = (LEVELS[LEVELS.len() - 1].target - 1, 0);
        p.cpu_y = 0.1;
        p.ball = (0.99, 0.9);
        p.vel = (0.9, 0.0);
        run(&mut p, 300);
        assert_eq!(p.phase(), Phase::Won);
    }

    #[test]
    fn space_restarts_a_finished_match() {
        let mut p = Pelota::new(4);
        p.level = 3;
        p.phase = Phase::GameOver;
        p.handle_key(key(KeyCode::Char(' ')));
        assert_eq!(p.level(), 1);
        assert!(!p.finished());
    }

    #[test]
    fn the_paddle_glides_then_stops() {
        let mut p = Pelota::new(1);
        p.player_y = 0.5;
        p.handle_key(key(KeyCode::Up));
        p.step(50);
        assert!(p.player_y < 0.5, "paddle did not move up");
        // Well past the glide window with no further key.
        run(&mut p, 500);
        let settled = p.player_y;
        run(&mut p, 500);
        assert!(
            (p.player_y - settled).abs() < f32::EPSILON,
            "paddle kept drifting"
        );
    }

    #[test]
    fn the_paddle_stays_on_the_field() {
        let mut p = Pelota::new(1);
        for _ in 0..200 {
            p.handle_key(key(KeyCode::Up));
            p.step(50);
        }
        assert!(p.player_y >= p.player_half() - 0.001);
        for _ in 0..200 {
            p.handle_key(key(KeyCode::Down));
            p.step(50);
        }
        assert!(p.player_y <= 1.0 - p.player_half() + 0.001);
    }

    #[test]
    fn pause_freezes_the_ball() {
        let mut p = Pelota::new(1);
        p.phase = Phase::Playing;
        p.ball = (0.5, 0.5);
        p.handle_key(key(KeyCode::Char('p')));
        assert_eq!(p.phase(), Phase::Paused);
        run(&mut p, 1000);
        assert!((p.ball.0 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn a_rally_replays_identically_from_the_same_seed() {
        let mut a = Pelota::new(1234);
        let mut b = Pelota::new(1234);
        for i in 0..400 {
            if i % 7 == 0 {
                a.handle_key(key(KeyCode::Up));
                b.handle_key(key(KeyCode::Up));
            }
            a.step(50);
            b.step(50);
        }
        assert_eq!(a.score(), b.score());
        assert!((a.ball.0 - b.ball.0).abs() < f32::EPSILON);
        assert!((a.ball.1 - b.ball.1).abs() < f32::EPSILON);
    }

    #[test]
    fn a_long_match_never_loses_the_ball() {
        let mut p = Pelota::new(77);
        for _ in 0..4000 {
            p.step(50);
            assert!(
                p.ball.1 >= -0.01 && p.ball.1 <= 1.01,
                "ball left the field vertically at {}",
                p.ball.1
            );
            if p.finished() {
                break;
            }
        }
    }

    #[test]
    fn the_field_fills_whatever_screen_it_is_given() {
        let p = Pelota::new(1);
        assert!(p.glyphs(MIN_W - 1, MIN_H).is_empty());
        assert!(p.glyphs(MIN_W, MIN_H - 1).is_empty());
        for (w, h) in [(MIN_W, MIN_H), (80, 24), (200, 60), (400, 100)] {
            let g = p.glyphs(w, h);
            assert!(!g.is_empty(), "nothing drawn at {w}x{h}");
            // Everything lands inside, and the field spans the full area:
            // walls edge to edge, paddles on the first and last rows' extremes.
            assert!(g.iter().all(|g| g.x < w && g.y < h), "spilled at {w}x{h}");
            assert!(
                g.iter().any(|g| g.x == 0) && g.iter().any(|g| g.x == w - 1),
                "walls do not reach both edges at {w}x{h}"
            );
            assert!(
                g.iter().any(|g| g.y == 0) && g.iter().any(|g| g.y == h - 1),
                "walls do not reach top and bottom at {w}x{h}"
            );
            // The CPU paddle sits near the right edge, not in a centered box.
            let cpu_col = g.iter().filter(|g| g.ch == '█').map(|g| g.x).max().unwrap();
            assert!(
                cpu_col > w - w / 8,
                "cpu paddle at {cpu_col} on a {w} screen"
            );
        }
    }

    #[test]
    fn esc_closes_pelota_with_the_score() {
        let mut a = Arcade::new();
        a.open("/pelota", false, 1);
        match a.handle_key(key(KeyCode::Esc)) {
            Outcome::Close(Some(line)) => assert!(line.starts_with("pelota: level 1"), "{line}"),
            other => panic!("expected a closing score line, got {other:?}"),
        }
        assert!(!a.is_open());
    }

    /// The screensaver is dismissed by whatever key the user came back to
    /// type — not just Esc — and leaves nothing in the scrollback.
    #[test]
    fn any_key_dismisses_the_screensaver_quietly() {
        for code in [
            KeyCode::Esc,
            KeyCode::Char('a'),
            KeyCode::Enter,
            KeyCode::Up,
        ] {
            let mut a = Arcade::new();
            a.open_screensaver(1);
            assert!(a.is_open() && a.is_screensaver());
            assert!(
                matches!(a.handle_key(key(code)), Outcome::Close(None)),
                "{code:?} should have dismissed the screensaver"
            );
            assert!(!a.is_open() && !a.is_screensaver());
        }
    }

    /// The screensaver has no command and so no parking slot: coming back to
    /// an idle terminal a second time deals a fresh sky rather than resuming.
    #[test]
    fn the_screensaver_is_never_parked() {
        let mut a = Arcade::new();
        a.open_screensaver(1);
        a.handle_key(key(KeyCode::Char('x')));
        assert!(!a.is_open());
        assert!(
            a.parked.iter().all(Option::is_none),
            "the screensaver must not occupy a game's slot"
        );
    }

    /// The rain is weather, not a game: closing it leaves nothing behind and
    /// takes no slot, so the next `/matrix` is a fresh downpour rather than a
    /// resume notice about a thing that has no state.
    #[test]
    fn the_rain_is_not_parked() {
        let mut a = Arcade::new();
        a.open("/matrix", false, 1);
        assert!(matches!(a.open.as_ref(), Some(Game::Matrix(_))));
        for _ in 0..200 {
            a.step(50);
        }
        assert_eq!(a.close(), None, "the rain left a line in the scrollback");
        assert!(!a.has_parked("/matrix"), "the rain took a parking slot");
        assert!(a.parked.iter().all(Option::is_none));
    }

    /// `c` re-letters the rain rather than quitting or steering — the escape
    /// hatch for a terminal font without katakana has to be reachable.
    #[test]
    fn c_cycles_the_rain_glyphs() {
        let mut a = Arcade::new();
        a.open("/matrix", false, 1);
        let charset = |a: &Arcade| match a.open.as_ref() {
            Some(Game::Matrix(m)) => m.charset(),
            _ => panic!("the rain is not open"),
        };
        let first = charset(&a);
        assert!(matches!(
            a.handle_key(key(KeyCode::Char('c'))),
            Outcome::Stay
        ));
        assert_ne!(charset(&a), first, "c did not change the alphabet");
        // ...and the whole cycle comes back around, whatever its length.
        while charset(&a) != first {
            a.handle_key(key(KeyCode::Char('c')));
        }
        assert!(a.is_open(), "cycling closed the rain");
    }

    /// The rain has no score and no lives, so it must stay as silent as the
    /// starfield even with the blips on.
    #[test]
    fn the_rain_never_blips() {
        let mut a = Arcade::new();
        a.open("/matrix", false, 1);
        a.sound.set(true);
        let before = a.open.as_ref().map(Game::vitals).unwrap();
        for _ in 0..100 {
            a.step(50);
        }
        let after = a.open.as_ref().map(Game::vitals).unwrap();
        assert_eq!(cue_for(before, after), None);
    }

    /// `/stars` is gone: it is the screensaver now, not a command.
    #[test]
    fn stars_is_no_longer_a_command() {
        assert!(!Arcade::COMMANDS.contains(&"/stars"));
        assert!(command_of("/stars").is_none());
        let mut a = Arcade::new();
        assert!(!a.open("/stars", false, 1));
    }

    #[test]
    fn ctrl_c_closes_any_easter_egg() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        for cmd in Arcade::COMMANDS {
            let mut a = Arcade::new();
            a.open(cmd, false, 1);
            assert!(matches!(a.handle_key(ctrl_c), Outcome::Close(_)), "{cmd}");
            assert!(!a.is_open(), "{cmd} stayed open");
        }
    }

    /// A key event with Shift held, the way a terminal reports Shift+arrow.
    fn shift_key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }

    #[test]
    fn shift_shrinks_the_paddle_to_a_third() {
        let mut p = Pelota::new(1);
        let full = p.player_half();
        p.handle_key(shift_key(KeyCode::Up));
        assert!(p.charged());
        assert!(
            (p.player_half() - full / CHARGE_SHRINK).abs() < f32::EPSILON,
            "charged paddle is {} not {}",
            p.player_half(),
            full / CHARGE_SHRINK
        );
        // Moving without Shift drops the stance immediately.
        p.handle_key(key(KeyCode::Up));
        assert!(!p.charged());
        assert!((p.player_half() - full).abs() < f32::EPSILON);
    }

    #[test]
    fn the_charged_stance_lapses_when_you_stop_moving() {
        let mut p = Pelota::new(1);
        p.handle_key(shift_key(KeyCode::Down));
        assert!(p.charged());
        run(&mut p, 500);
        assert!(!p.charged(), "stance outlived the movement");
    }

    #[test]
    fn a_charged_hit_leaves_at_triple_speed() {
        // Same contact, once plain and once charged.
        let hit = |charged: bool| {
            let mut p = Pelota::new(1);
            p.phase = Phase::Playing;
            p.player_y = 0.5;
            if charged {
                p.charge_ms = GLIDE_MS;
            }
            p.ball = (PLAYER_X + 0.002, 0.5);
            p.vel = (-0.6, 0.0);
            p.step(50);
            p.vel.0.hypot(p.vel.1)
        };
        let plain = hit(false);
        let charged = hit(true);
        assert!(
            (charged / plain - POWER_GAIN).abs() < 0.05,
            "charged hit was {charged} vs plain {plain}"
        );
    }

    #[test]
    fn the_cpu_return_takes_the_power_back_off_the_ball() {
        let mut p = Pelota::new(3);
        p.phase = Phase::Playing;
        p.cpu_y = 0.5;
        p.ball = (CPU_X - 0.002, 0.5);
        // Arriving three times as fast as the level's ball.
        p.vel = (H_GAIN * LEVELS[0].ball_speed * POWER_GAIN, 0.0);
        p.step(50);
        let speed = p.vel.0.hypot(p.vel.1);
        assert!(
            speed < LEVELS[0].ball_speed * H_GAIN * 1.1,
            "cpu returned it at {speed}, still boosted"
        );
    }

    /// The charged shot is the fastest the ball ever goes; if a sub-step could
    /// carry it across the whole paddle plane it would score instead of bounce.
    #[test]
    fn even_the_fastest_shot_cannot_tunnel_through_a_paddle() {
        #[allow(clippy::cast_precision_loss)]
        let dt = SUB_MS as f32 / 1000.0;
        let fastest =
            LEVELS.iter().map(|l| l.ball_speed).fold(0.0f32, f32::max) * POWER_GAIN * H_GAIN;
        assert!(
            fastest * dt < PLAYER_X,
            "a {SUB_MS}ms sub-step moves {} but the paddle plane is only {PLAYER_X} deep",
            fastest * dt
        );
    }

    #[test]
    fn the_wheel_nudges_the_paddle_and_a_click_places_it() {
        let mouse = |kind, row| MouseEvent {
            kind,
            column: 4,
            row,
            modifiers: KeyModifiers::NONE,
        };
        let mut p = Pelota::new(1);
        p.player_y = 0.5;
        p.handle_mouse(mouse(MouseEventKind::ScrollUp, 0), 24);
        p.step(50);
        assert!(p.player_y < 0.5, "wheel did not move the paddle up");

        // A click on the bottom third of a 24-row area puts it down there.
        let mut p = Pelota::new(1);
        p.handle_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 18), 24);
        assert!(p.player_y > 0.7, "click landed at {}", p.player_y);
        // Pointing is absolute: no residual glide afterwards.
        let placed = p.player_y;
        run(&mut p, 500);
        assert!((p.player_y - placed).abs() < f32::EPSILON);
    }

    #[test]
    fn a_shift_wheel_arms_the_charge_too() {
        let mut p = Pelota::new(1);
        p.handle_mouse(
            MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 4,
                row: 5,
                modifiers: KeyModifiers::SHIFT,
            },
            24,
        );
        assert!(p.charged());
    }

    #[test]
    fn the_mouse_never_puts_the_paddle_off_the_field() {
        let mut p = Pelota::new(1);
        for row in 0..40u16 {
            p.handle_mouse(
                MouseEvent {
                    kind: MouseEventKind::Moved,
                    column: 4,
                    row,
                    modifiers: KeyModifiers::NONE,
                },
                24,
            );
            let half = p.player_half();
            assert!(
                p.player_y >= half - 0.001 && p.player_y <= 1.0 - half + 0.001,
                "row {row} put the paddle at {}",
                p.player_y
            );
        }
    }

    #[test]
    fn t_toggles_the_veil_in_any_game() {
        for cmd in Arcade::COMMANDS {
            let mut a = Arcade::new();
            a.open(cmd, false, 1);
            assert!(!a.translucent);
            a.handle_key(key(KeyCode::Char('t')));
            assert!(a.translucent, "{cmd} did not veil");
            a.handle_key(key(KeyCode::Char('t')));
            assert!(!a.translucent, "{cmd} did not unveil");
        }
    }

    /// Two lists name these commands: the constructor here and the
    /// dispatcher's `slash_command_known`. If they drift, a command either
    /// stops working or gets forwarded to the model as a prompt.
    #[test]
    fn every_arcade_command_is_known_to_the_dispatcher() {
        for cmd in Arcade::COMMANDS {
            assert!(
                crate::config::slash_command_known(cmd),
                "{cmd} would be sent to the model"
            );
            // ...and with an argument, or `/pelota new` would go to the model.
            assert!(
                crate::config::slash_command_known(&format!("{cmd} new")),
                "{cmd} new would be sent to the model"
            );
            assert!(Arcade::new().open(cmd, false, 1), "{cmd} opens nothing");
        }
    }

    /// `ui.easterEggs=false` has to make the games *not exist*, not merely stay
    /// unlisted: an unknown slash command goes to the model as a prompt, which
    /// is exactly what a deployment that wants no games at all asks for. Driven
    /// through the pure form so neither answer depends on process-wide state.
    #[test]
    fn the_easter_eggs_flag_decides_whether_the_commands_exist() {
        for cmd in Arcade::COMMANDS {
            assert!(
                crate::config::slash_command_known_with(cmd, true),
                "{cmd} unknown with the flag on"
            );
            assert!(
                crate::config::slash_command_known_with(&format!("{cmd} new"), true),
                "{cmd} new unknown with the flag on"
            );
            assert!(
                !crate::config::slash_command_known_with(cmd, false),
                "{cmd} still known with the flag off"
            );
            assert!(
                !crate::config::slash_command_known_with(&format!("{cmd} new"), false),
                "{cmd} new still known with the flag off"
            );
        }
        // A real command must not be collateral damage of the gate.
        assert!(
            crate::config::slash_command_known_with("/help", false),
            "/help stopped working with the flag off"
        );
    }

    /// The flag defaults on, so the eggs are there for anyone who has not asked
    /// otherwise — and `command_of` agrees with the dispatcher about that.
    #[test]
    fn the_easter_eggs_are_on_by_default() {
        assert!(
            crate::settings::Settings::default().ui.easter_eggs,
            "easter eggs are opt-out, not opt-in"
        );
        assert!(enabled(), "default settings disabled the arcade");
        for cmd in Arcade::COMMANDS {
            assert_eq!(command_of(cmd), Some(cmd), "{cmd} not recognized");
            assert_eq!(
                command_of(&format!("{cmd} new sound")),
                Some(cmd),
                "{cmd} with arguments not recognized"
            );
        }
        assert_eq!(command_of("/helpful"), None, "matched a non-arcade command");
    }

    /// Easter eggs stay out of the help text and the completion popup. That is
    /// the whole point of an easter egg, and it is easy to undo by accident.
    #[test]
    fn no_arcade_command_is_advertised() {
        let help = crate::config::usage();
        for cmd in Arcade::COMMANDS {
            assert!(!help.contains(cmd), "{cmd} is listed in /help");
        }
    }

    #[test]
    fn every_game_survives_a_long_unattended_run() {
        for cmd in Arcade::COMMANDS {
            let mut a = Arcade::new();
            a.open(cmd, false, 7);
            for _ in 0..2000 {
                a.step(50);
                // Whatever the state, it must always have something to paint
                // and never spill outside the area it was given.
                assert!(
                    a.glyphs(80, 23).iter().all(|g| g.x < 80 && g.y < 23),
                    "{cmd} spilled outside its area"
                );
            }
        }
    }

    /// Closing and reopening must land you back in the same game — that is the
    /// whole reason the cabinet outlives the modal.
    #[test]
    fn closing_and_reopening_resumes_the_same_game() {
        let mut a = Arcade::new();
        assert!(a.open("/breakout", false, 1));
        // Knock the wall about a bit so the state is distinguishable.
        for _ in 0..200 {
            a.step(50);
        }
        let before = match a.open.as_ref() {
            Some(Game::Breakout(b)) => (b.level(), b.remaining()),
            _ => panic!("breakout not open"),
        };
        a.close();
        assert!(!a.is_open());
        assert!(a.has_parked("/breakout"));

        a.open("/breakout", false, 999);
        let after = match a.open.as_ref() {
            Some(Game::Breakout(b)) => (b.level(), b.remaining()),
            _ => panic!("breakout not open"),
        };
        assert_eq!(before, after, "the game restarted instead of resuming");
    }

    #[test]
    fn the_new_argument_deals_a_fresh_game() {
        let mut a = Arcade::new();
        a.open("/breakout", false, 1);
        for _ in 0..200 {
            a.step(50);
        }
        a.close();
        a.open("/breakout", true, 2);
        match a.open.as_ref() {
            Some(Game::Breakout(b)) => {
                assert_eq!(b.remaining(), breakout::LEVELS[0].rows * breakout::COLS);
                assert_eq!(b.level(), 1);
            }
            _ => panic!("breakout not open"),
        }
        assert!(!a.has_parked("/breakout"), "the old game was kept as well");
    }

    #[test]
    fn new_is_recognized_however_it_is_written() {
        for arg in ["new", "NEW", " reset ", "RESET"] {
            assert!(Arcade::wants_new(arg), "{arg:?} did not read as new");
        }
        for arg in ["", "  ", "newton", "1"] {
            assert!(!Arcade::wants_new(arg), "{arg:?} read as new");
        }
    }

    /// A finished game is not worth resuming onto its own game-over banner.
    #[test]
    fn a_finished_game_is_not_parked() {
        let mut a = Arcade::new();
        a.open("/pelota", false, 1);
        match a.open.as_mut() {
            Some(Game::Pelota(p)) => p.force_finished(),
            _ => panic!("pelota not open"),
        }
        a.close();
        assert!(!a.has_parked("/pelota"), "a finished match was parked");
    }

    #[test]
    fn an_unknown_command_opens_nothing() {
        let mut a = Arcade::new();
        assert!(!a.open("/nope", false, 1));
        assert!(!a.is_open());
    }

    #[test]
    fn a_closed_cabinet_paints_nothing_and_absorbs_nothing() {
        let mut a = Arcade::new();
        assert!(a.glyphs(80, 24).is_empty());
        assert!(a.banner(80, 24).is_none());
        assert!(a.footer().is_empty());
        a.step(50);
        assert!(matches!(a.handle_key(key(KeyCode::Esc)), Outcome::Stay));
    }

    #[test]
    fn sound_is_off_unless_the_argument_asks_for_it() {
        for arg in ["sound", "SOUND", "new sound", "sound new"] {
            assert!(Sound::wanted(arg), "{arg:?} did not ask for sound");
        }
        for arg in ["", "new", "soundtrack", "reset"] {
            assert!(!Sound::wanted(arg), "{arg:?} asked for sound");
        }
        assert!(!Sound::default().is_on());
    }

    /// The arguments compose: `new sound` has to mean both.
    #[test]
    fn new_and_sound_can_be_asked_for_together() {
        assert!(Arcade::wants_new("new sound"));
        assert!(Sound::wanted("new sound"));
        assert!(Arcade::wants_new("sound new"));
    }

    #[test]
    fn a_silent_arcade_writes_no_bytes_at_all() {
        let quiet = Sound::default();
        for cue in [Cue::Hit, Cue::Lose, Cue::Win] {
            assert!(quiet.bytes(cue).is_empty(), "{cue:?} made noise while off");
        }
        let mut loud = Sound::default();
        loud.set(true);
        // Count is the only thing BEL gives us to distinguish cues with.
        assert_eq!(loud.bytes(Cue::Hit).len(), 1);
        assert_eq!(loud.bytes(Cue::Lose).len(), 2);
        assert_eq!(loud.bytes(Cue::Win).len(), 3);
        assert!(loud.bytes(Cue::Hit).iter().all(|b| *b == 0x07));
    }

    #[test]
    fn b_toggles_the_blips() {
        let mut a = Arcade::new();
        a.open("/breakout", false, 1);
        assert!(!a.sound.is_on());
        a.handle_key(key(KeyCode::Char('b')));
        assert!(a.sound.is_on());
        a.handle_key(key(KeyCode::Char('b')));
        assert!(!a.sound.is_on());
    }

    #[test]
    fn cues_rank_an_ending_above_everything_else() {
        let base = Vitals {
            score: 10,
            lives: 3,
            level: 1,
            phase: Some(Phase::Playing),
        };
        // A game over outranks the point that came with it.
        let over = Vitals {
            score: 11,
            lives: 2,
            phase: Some(Phase::GameOver),
            ..base
        };
        assert_eq!(cue_for(base, over), Some(Cue::Lose));
        let won = Vitals {
            score: 11,
            phase: Some(Phase::Won),
            ..base
        };
        assert_eq!(cue_for(base, won), Some(Cue::Win));
        // A level outranks a lost life.
        let up = Vitals {
            level: 2,
            lives: 2,
            ..base
        };
        assert_eq!(cue_for(base, up), Some(Cue::Win));
        // A lost life outranks a point.
        let hurt = Vitals {
            lives: 2,
            score: 11,
            ..base
        };
        assert_eq!(cue_for(base, hurt), Some(Cue::Lose));
        assert_eq!(cue_for(base, Vitals { score: 11, ..base }), Some(Cue::Hit));
        assert_eq!(cue_for(base, base), None);
    }

    /// The starfield has no score, no lives and no levels, so it must stay
    /// silent even with sound on — a blip every frame would be unbearable.
    #[test]
    fn the_starfield_never_blips() {
        let mut a = Arcade::new();
        a.open_screensaver(1);
        a.sound.set(true);
        let before = a.open.as_ref().map(Game::vitals).unwrap();
        for _ in 0..100 {
            a.step(50);
        }
        let after = a.open.as_ref().map(Game::vitals).unwrap();
        assert_eq!(cue_for(before, after), None);
    }

    #[test]
    fn opening_over_output_starts_veiled() {
        let mut a = Arcade::new();
        a.open("/pelota", false, 1);
        a.veil();
        assert!(a.translucent);
    }

    #[test]
    fn a_too_small_terminal_explains_itself() {
        let mut a = Arcade::new();
        a.open("/pelota", false, 1);
        assert!(a.banner(10, 4).is_some_and(|b| b.contains("terminale")));
        let mut sky = Arcade::new();
        sky.open_screensaver(1);
        assert!(sky.banner(80, 24).is_none());
    }
}

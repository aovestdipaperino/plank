// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A perspective starfield.
//!
//! **Ported verbatim from plank's `src/arcade.rs`**, with one change: the
//! imports. The projection, the recycle rule, the streak length and the
//! brightness ramp are the original, because the point of the port is that the
//! sky looks the same after it — pinned by a glyph-for-glyph comparison
//! against the built-in field.

#![allow(dead_code)]

use crate::support::{Glyph, MAX_STEP_MS, Rgb, Rng, lerp_rgb};

// ---------------------------------------------------------------------------
// Starfield
// ---------------------------------------------------------------------------

/// Depth at which a star has passed the viewer and is recycled.
const Z_NEAR: f32 = 0.12;
/// Depth a star is (re)born at.
const Z_FAR: f32 = 1.0;
/// Vertical squash of the projection.
///
/// Terminal cells are roughly twice as tall as they are wide, so the vertical
/// axis is halved to keep the outward rush radial instead of egg-shaped.
const CELL_ASPECT: f32 = 0.5;
/// Half-height of the spawn box, pre-squash.
///
/// The squash is applied on the way to the screen, so spawning `y` in the same
/// `[-1, 1]` as `x` would land every new star in the middle half of the frame
/// and leave visible dead bands along the top and bottom edges. Widening the
/// box by exactly the inverse of the squash makes the projected sky uniform.
const Y_SPAWN: f32 = 1.0 / CELL_ASPECT;
/// Brightness ramp, dimmest (far) to brightest (near).
const STAR_RAMP: [char; 8] = ['.', '.', '·', '·', '+', '*', 'o', '@'];
/// Color of the most distant stars.
const STAR_FAR: Rgb = (56, 64, 104);
/// Color of a star about to sweep past the viewer.
const STAR_NEAR: Rgb = (255, 255, 255);
/// Slowest and fastest travel, in depth units per second.
const SPEED_MIN: f32 = 0.08;
const SPEED_MAX: f32 = 3.0;
/// Default travel: already clearly in motion when the sky opens.
const SPEED_DEFAULT: f32 = 0.9;
/// How much of a star's recent path is drawn behind it, in seconds.
///
/// This is what turns points into streaks. Because it multiplies the star's
/// own speed, the whole sky stretches when the speed goes up — the jump to
/// lightspeed — and collapses back to points when it comes down.
const TRAIL_SECONDS: f32 = 0.24;
/// How far the streak dims from head to tail, as a fraction of the head.
const TRAIL_FADE: f32 = 0.8;
/// Longest streak drawn, in cells. Bounds the per-star cost on a huge screen.
const MAX_TRAIL: f32 = 48.0;

/// One star in normalized eye space: `x`/`y` in `[-1, 1]`, `z` in `(0, 1]`.
#[derive(Debug, Clone, Copy)]
struct Star {
    x: f32,
    y: f32,
    z: f32,
}

/// A perspective starfield that rushes outward past the edges of the screen.
///
/// Each star travels straight toward the viewer (`z` shrinking); the
/// projection `x / z` makes it accelerate off the edge, which is the whole
/// effect. A star is recycled once it passes the viewer *or* leaves the frame,
/// so the visible density stays constant instead of thinning out.
#[derive(Debug)]
pub struct Starfield {
    stars: Vec<Star>,
    rng: Rng,
    speed: f32,
}

impl Starfield {
    /// Builds a field of `count` stars, spread over the full depth range so
    /// the first frame is already a sky rather than a distant wall.
    #[must_use]
    pub fn new(seed: u64, count: usize) -> Self {
        let mut rng = Rng::new(seed);
        let stars = (0..count)
            .map(|_| Star {
                x: rng.range(-1.0, 1.0),
                y: rng.range(-Y_SPAWN, Y_SPAWN),
                z: rng.range(Z_NEAR, Z_FAR),
            })
            .collect();
        Self {
            stars,
            rng,
            speed: SPEED_DEFAULT,
        }
    }

    /// Current travel speed, in depth units per second.
    #[must_use]
    pub const fn speed(&self) -> f32 {
        self.speed
    }

    /// Scales the travel speed, clamped to the usable range.
    pub const fn scale_speed(&mut self, factor: f32) {
        self.speed = (self.speed * factor).clamp(SPEED_MIN, SPEED_MAX);
    }

    /// Advances every star by `dt_ms` and recycles the ones that left the frame.
    #[allow(clippy::cast_precision_loss)] // dt is bounded by MAX_STEP_MS
    pub fn step(&mut self, dt_ms: u64) {
        let dt = dt_ms.min(MAX_STEP_MS) as f32 / 1000.0;
        let travel = self.speed * dt;
        for i in 0..self.stars.len() {
            self.stars[i].z -= travel;
            if Self::escaped(&self.stars[i]) {
                self.stars[i] = Star {
                    x: self.rng.range(-1.0, 1.0),
                    y: self.rng.range(-Y_SPAWN, Y_SPAWN),
                    z: Z_FAR,
                };
            }
        }
    }

    /// Whether a star has passed the viewer or swept outside the frame.
    ///
    /// The frame test is done in normalized space (the projection maps
    /// `[-1, 1]` onto the full area whatever the terminal size), so recycling
    /// does not depend on the caller's geometry.
    fn escaped(s: &Star) -> bool {
        if s.z <= Z_NEAR {
            return true;
        }
        // Generous: the head is allowed well past the edge so its streak can
        // finish sweeping out of frame instead of being cut off mid-flight.
        (s.x / s.z).abs() > 1.4 || (s.y * CELL_ASPECT / s.z).abs() > 1.4
    }

    /// Where a star sits on a `fw` x `fh` area at depth `z`.
    ///
    /// Every star lies on a ray from the center, so its whole trail projects
    /// onto that same ray — which is what makes the streaks radiate.
    fn project(s: &Star, z: f32, fw: f32, fh: f32) -> (f32, f32) {
        (
            0.5f32.mul_add(s.x / z, 0.5) * fw,
            0.5f32.mul_add(s.y * CELL_ASPECT / z, 0.5) * fh,
        )
    }

    /// Projects the field onto a `w` x `h` area, each star drawn as the streak
    /// it swept over the last [`TRAIL_SECONDS`].
    ///
    /// The streak length is the star's own motion, so it falls out of the
    /// perspective for free: near the vanishing point stars are still points,
    /// and as they accelerate outward they stretch into the lines that read as
    /// hyperspace. Turning the speed up lengthens every streak at once.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // screen coords: small non-negative integers, bounds-checked below
    #[must_use]
    pub fn glyphs(&self, w: u16, h: u16) -> Vec<Glyph> {
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let (fw, fh) = (f32::from(w - 1), f32::from(h - 1));
        let trail_depth = self.speed * TRAIL_SECONDS;
        let last = (STAR_RAMP.len() - 1) as f32;
        let mut out = Vec::with_capacity(self.stars.len() * 3);
        for star in &self.stars {
            let (head_x, head_y) = Self::project(star, star.z, fw, fh);
            // The tail is where the star was; never earlier than its birth.
            let (tail_x, tail_y) = Self::project(star, (star.z + trail_depth).min(Z_FAR), fw, fh);
            // One sample per cell along the longer axis: no gaps, no waste.
            let span = (head_x - tail_x)
                .abs()
                .max((head_y - tail_y).abs())
                .min(MAX_TRAIL);
            let steps = span.round() as usize;
            // Nearness in [0, 1]: 0 at the spawn depth, 1 at the viewer.
            let near = ((Z_FAR - star.z) / (Z_FAR - Z_NEAR)).clamp(0.0, 1.0);
            // Tail first, so the bright head wins the cell it shares.
            for i in (0..=steps).rev() {
                let along = if steps == 0 {
                    0.0
                } else {
                    i as f32 / steps as f32
                };
                let px = (tail_x - head_x).mul_add(along, head_x);
                let py = (tail_y - head_y).mul_add(along, head_y);
                if px < 0.0 || py < 0.0 || px > fw || py > fh {
                    continue;
                }
                let bright = near * along.mul_add(-TRAIL_FADE, 1.0);
                let idx = ((bright * last).round() as usize).min(STAR_RAMP.len() - 1);
                out.push(Glyph {
                    x: px.round() as u16,
                    y: py.round() as u16,
                    ch: STAR_RAMP[idx],
                    color: lerp_rgb(STAR_FAR, STAR_NEAR, bright),
                });
            }
        }
        out
    }
}

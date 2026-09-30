// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Shared-clock TUI animation subsystem.
//!
//! Every animated element in the Ratatui front-end is driven by one shared
//! 20 Hz (50 ms) clock so the busy-state UI reads as a single coherent system.
//! Each effect is a **pure function of an injected time value** (plus text and
//! colors); the only persistent state is the stall intensity, which is threaded
//! explicitly rather than stored globally. This keeps every effect
//! deterministic and unit-testable at chosen timestamps without a running clock.
//!
//! # Reduced motion
//!
//! [`set_reduced_motion`] flips one global toggle that collapses every effect to
//! a static fallback. The shared clock ([`clock_ms`]) returns `None` in that
//! mode, so a component branches **once** on the clock and renders its static
//! form (for liveness it can still fall back to the slow [`reduced_pulse`]).
//!
//! # Front-end boundary
//!
//! Motion is Ratatui-only. The plain line REPL and `--ui console` paths
//! render the static/reduced-motion form. These functions live behind the
//! `viz.rs` / `tui.rs` render sink so the plain path stays untouched.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Animation refresh rate: 20 frames per second.
pub const TICK_HZ: u64 = 20;
/// Milliseconds between animation frames (the shared clock's quantum).
pub const TICK_MS: u64 = 1000 / TICK_HZ;

/// Global reduced-motion toggle. Off by default, so existing behavior (and the
/// plain-REPL throbber) is unchanged until a front-end opts in.
static REDUCED_MOTION: AtomicBool = AtomicBool::new(false);

/// Enables or disables reduced-motion mode process-wide. When enabled the
/// shared [`clock_ms`] returns `None` and every effect collapses to its static
/// fallback.
pub fn set_reduced_motion(on: bool) {
    REDUCED_MOTION.store(on, Ordering::Relaxed);
}

/// Serialises the tests that flip the process-global reduced-motion flag.
///
/// `cargo test` runs the suite in parallel and [`set_reduced_motion`] is
/// process-wide, so without this two tests can observe each other's flag.
/// Mirrors `crate::status::origin_test_guard`.
#[cfg(test)]
pub(crate) fn reduced_motion_test_guard() -> ReducedMotionGuard {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ReducedMotionGuard(
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// Holds the reduced-motion test lock and restores the flag when it drops.
///
/// Restoring on drop rather than at the end of each test is what makes a
/// panicking test harmless: an assertion that fires while the flag is on would
/// otherwise leave every later test running in reduced motion, turning one
/// failure into a cascade that looks nothing like its cause.
#[cfg(test)]
pub(crate) struct ReducedMotionGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

#[cfg(test)]
impl Drop for ReducedMotionGuard {
    fn drop(&mut self) {
        set_reduced_motion(false);
    }
}

/// Whether reduced-motion mode is currently active.
#[must_use]
pub fn reduced_motion() -> bool {
    REDUCED_MOTION.load(Ordering::Relaxed)
}

/// The shared animation clock: milliseconds since the first read, or `None`
/// when reduced motion is active (components then render their static form).
///
/// The epoch is captured on first use, so the very first frame reads `~0`.
#[must_use]
pub fn clock_ms() -> Option<u64> {
    if reduced_motion() {
        return None;
    }
    Some(epoch_ms())
}

/// Milliseconds since the shared epoch, ignoring reduced motion.
///
/// [`clock_ms`] is the clock effects animate off, and it goes dark on purpose.
/// This is the underlying *measurement*, for callers that need to time a real
/// interval (how long a pass has been running) rather than to move something —
/// they branch on reduced motion themselves, at the point where they decide
/// whether to animate at all.
#[must_use]
pub fn epoch_ms() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(0)
}

/// An 8-bit-per-channel RGB color.
pub type Rgb = (u8, u8, u8);

/// Linearly interpolates a single channel from `a` to `b` by `t`, with `t`
/// clamped to `0.0..=1.0`. Rounds to the nearest integer.
#[must_use]
pub fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let t = t.clamp(0.0, 1.0);
    let a = f32::from(a);
    let b = f32::from(b);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = (a + (b - a) * t).round().clamp(0.0, 255.0) as u8;
    v
}

/// Linearly interpolates an RGB color from `a` to `b` by `t` (clamped), the
/// shared color utility behind every color effect.
#[must_use]
pub fn lerp_rgb(a: Rgb, b: Rgb, t: f32) -> Rgb {
    (
        lerp_u8(a.0, b.0, t),
        lerp_u8(a.1, b.1, t),
        lerp_u8(a.2, b.2, t),
    )
}

/// A sine oscillation mapped to `0.0..=1.0`: `0.5` at `t == delay_ms`, rising to
/// `1.0` a quarter period later and dipping to `0.0` at three-quarters. Before
/// `delay_ms` the value rests at `0.0` (dim). `period_ms == 0` rests at `0.0`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn sine01(now_ms: u64, period_ms: u64, delay_ms: u64) -> f32 {
    if period_ms == 0 || now_ms < delay_ms {
        return 0.0;
    }
    let phase = (now_ms - delay_ms) as f64 / period_ms as f64;
    let s = (phase * std::f64::consts::TAU).sin();
    #[allow(clippy::cast_possible_truncation)]
    let v = f64::midpoint(s, 1.0) as f32;
    v
}

/// Shimmer/Pulse: interpolates between a `dim` and `bright` color on a sine
/// period (optionally delayed). Reduces to `dim` at troughs, `bright` at peaks.
#[must_use]
pub fn pulse_color(dim: Rgb, bright: Rgb, now_ms: u64, period_ms: u64, delay_ms: u64) -> Rgb {
    lerp_rgb(dim, bright, sine01(now_ms, period_ms, delay_ms))
}

/// Flash: whole-message color interpolation between `base` and `flash` on a
/// sine period. Same math as [`pulse_color`] with no start delay.
#[must_use]
pub fn flash_color(base: Rgb, flash: Rgb, now_ms: u64, period_ms: u64) -> Rgb {
    pulse_color(base, flash, now_ms, period_ms, 0)
}

/// Half-width (in columns) of the glimmer/shimmer highlight window, so the
/// window spans `2 * SWEEP_HALF + 1` columns (3 by default).
pub const SWEEP_HALF: i64 = 1;

/// Glimmer (sweep): the inclusive `[lo, hi]` column window of the highlight for
/// text of `len` columns at `now_ms`. The center advances one column per
/// `step_ms` and travels over `len + gap` columns, resting off-text for `gap`
/// columns between sweeps. `reverse` sends the sweep left-to-right instead of
/// right-to-left. The returned window may lie partly or wholly off-text; use
/// [`sweep_contains`] to test a column.
#[must_use]
pub fn sweep_window(
    len: usize,
    now_ms: u64,
    step_ms: u64,
    gap: usize,
    reverse: bool,
) -> (i64, i64) {
    let len = i64::try_from(len).unwrap_or(i64::MAX);
    let gap = i64::try_from(gap).unwrap_or(0);
    let cycle = (len + gap).max(1);
    let step = step_ms.max(1);
    let s = i64::try_from(now_ms / step).unwrap_or(0) % cycle;
    // Right-to-left: center starts past the right edge and walks left.
    let center = if reverse {
        s - SWEEP_HALF - 1
    } else {
        (len + SWEEP_HALF) - s
    };
    (center - SWEEP_HALF, center + SWEEP_HALF)
}

/// Whether column `col` falls inside a [`sweep_window`].
#[must_use]
pub fn sweep_contains(col: i64, window: (i64, i64)) -> bool {
    col >= window.0 && col <= window.1
}

/// Braille throbber frames, shared by the footer and any other liveness cue.
pub const THROBBER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Throbber: the frame index at `now_ms`, cycling **forward then backward**
/// (ping-pong) so the animation eases at both ends instead of snapping. One
/// frame advances per `step_ms`; `n` is the frame count.
#[must_use]
pub fn throbber_index(now_ms: u64, step_ms: u64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let step = step_ms.max(1);
    let n_i = i64::try_from(n).unwrap_or(1);
    // A full ping-pong period visits each end once: 2*(n-1) frames.
    let period = 2 * (n_i - 1);
    let pos = i64::try_from(now_ms / step).unwrap_or(0) % period;
    let idx = if pos < n_i { pos } else { period - pos };
    usize::try_from(idx).unwrap_or(0)
}

/// The Braille throbber glyph at `now_ms` (ping-pong over [`THROBBER_FRAMES`]).
#[must_use]
pub fn throbber_char(now_ms: u64, step_ms: u64) -> char {
    THROBBER_FRAMES[throbber_index(now_ms, step_ms, THROBBER_FRAMES.len())]
}

/// Stalled → red: advances the smoothed stall intensity (`0.0` normal, `1.0`
/// fully error-red). When `stalled` is false the intensity resets **instantly**
/// to `0.0` (a new token or an active tool clears it); otherwise it fades toward
/// `1.0` by `dt_ms / fade_ms`. Threaded explicitly — this is the only
/// persistent animation state.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn stall_next(prev: f32, dt_ms: u64, stalled: bool, fade_ms: u64) -> f32 {
    if !stalled {
        return 0.0;
    }
    if fade_ms == 0 {
        return 1.0;
    }
    let inc = dt_ms as f32 / fade_ms as f32;
    (prev + inc).clamp(0.0, 1.0)
}

/// The color for the current stall `intensity`, fading `base` toward `red`.
#[must_use]
pub fn stall_color(base: Rgb, red: Rgb, intensity: f32) -> Rgb {
    lerp_rgb(base, red, intensity)
}

/// xterm-256 palette index to RGB: 0-15 base colors, 16-231 the 6x6x6 cube,
/// 232-255 the 24-step grayscale ramp.
///
/// Lives here beside [`lerp_rgb`] because it is pure color math with no
/// business logic, and both the status shimmer and `crate::ui` need it to turn
/// an indexed accent into something interpolable.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn indexed_to_rgb(i: u8) -> Rgb {
    const BASE: [Rgb; 16] = [
        (0, 0, 0),
        (205, 0, 0),
        (0, 205, 0),
        (205, 205, 0),
        (0, 0, 238),
        (205, 0, 205),
        (0, 205, 205),
        (229, 229, 229),
        (127, 127, 127),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (92, 92, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ];
    match i {
        0..=15 => BASE[i as usize],
        16..=231 => {
            let n = i - 16;
            let steps = [0u8, 95, 135, 175, 215, 255];
            let r = steps[(n / 36) as usize];
            let g = steps[((n / 6) % 6) as usize];
            let b = steps[(n % 6) as usize];
            (r, g, b)
        }
        232..=255 => {
            let v = 8 + 10 * (i - 232);
            (v, v, v)
        }
    }
}

/// Nearest xterm 6x6x6 colour-cube index for a 0-255 RGB triple.
#[must_use]
pub fn cube_index(red: f32, green: f32, blue: f32) -> u8 {
    const LEVELS: [f32; 6] = [0.0, 95.0, 135.0, 175.0, 215.0, 255.0];
    let quantise = |value: f32| -> u8 {
        let value = value.clamp(0.0, 255.0);
        let mut best = 0u8;
        let mut best_dist = f32::MAX;
        for (slot, level) in LEVELS.iter().enumerate() {
            let dist = (value - level).abs();
            if dist < best_dist {
                best_dist = dist;
                best = u8::try_from(slot).unwrap_or(0);
            }
        }
        best
    };
    16 + 36 * quantise(red) + 6 * quantise(green) + quantise(blue)
}

/// [`cube_index`] for an [`Rgb`] triple.
#[must_use]
pub fn cube_index_rgb(c: Rgb) -> u8 {
    cube_index(f32::from(c.0), f32::from(c.1), f32::from(c.2))
}

/// How far a derived shimmer secondary sits from its accent toward white.
///
/// A profile that sets only an `accent` still gets a sweep in its own hue
/// rather than the built-in olive, which is the whole point of deriving one.
///
/// This deliberately does *not* try to reproduce the built-in ramp. That ramp
/// lifts the theme green 106 `#87af00` toward 192 `#d7ff87` by 0.67, 1.00 and
/// 0.53 on R, G and B — it was picked by eye, not by formula, and no single
/// factor lands on it. So the built-in keeps its own hand-picked
/// `crate::status::SHIMMER_RAMP` and this constant only has to make an
/// arbitrary accent read as lit: far enough to be visibly brighter at the
/// sweep's center, short of washing the hue out to white.
pub const DERIVED_SECONDARY_T: f32 = 0.45;

/// The resting colour for a profile that declared an accent but no
/// `secondary`: the accent lightened toward white by [`DERIVED_SECONDARY_T`],
/// so the word reads in the profile's hue while its accent sweeps across.
#[must_use]
pub fn derived_secondary(accent: Rgb) -> Rgb {
    lerp_rgb(accent, (255, 255, 255), DERIVED_SECONDARY_T)
}

/// The shimmer shades sweeping the status verb, interpolated from the word's
/// resting colour to the highlight and quantized to the xterm cube.
///
/// Ordered outermost column first, matching `crate::status::SHIMMER_RAMP`'s
/// contract, so the last entry lands on the center of the sweep and is the
/// highlight exactly. `crate::profile::shimmer_ramp` passes the profile's
/// `secondary` as `rest` and its `accent` as `highlight`: the accent is the
/// thing that travels. The length is deliberately three: the sweep window is
/// `2 * (len - 1) + 1` columns, so keeping it preserves the geometry and the
/// existing shimmer tests, and only the shades differ.
#[must_use]
pub fn shimmer_ramp(rest: Rgb, highlight: Rgb) -> [u8; 3] {
    let mut out = [0u8; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = (i + 1) as f32 / 3.0;
        *slot = cube_index_rgb(lerp_rgb(rest, highlight, t));
    }
    out
}

/// Fixed cycle for the reduced-motion pulse (dim half, bright half).
pub const REDUCED_PULSE_MS: u64 = 1000;

/// Reduced-motion pulse: a static dot toggling dim/bright on a fixed slow
/// cycle. `true` (bright) for the first half of each `period_ms`, `false` (dim)
/// for the second. This is the minimal liveness fallback used when the shared
/// clock is `None`.
#[must_use]
pub fn reduced_pulse(now_ms: u64, period_ms: u64) -> bool {
    let period = period_ms.max(1);
    now_ms % period < period / 2
}

/// Milliseconds per column for the fast sweep, used while the prompt is still
/// prefilling. Fast enough to read as "taking it in quickly" against the slow
/// sweep of generation.
pub const SWEEP_FAST_MS: u64 = 50;

/// Milliseconds per column for the ordinary sweep, used while the model
/// generates or thinks. Same value as `crate::status::SHIMMER_STEP_MS`, which
/// is deliberately left in place: that constant belongs to the footer's content
/// layer and is named by existing tests, while this one belongs to the
/// animation layer.
pub const SWEEP_SLOW_MS: u64 = 200;

/// Period of the whole-word flash used while a tool dispatch is in flight.
pub const FLASH_PERIOD_MS: u64 = 2000;

/// Which animation paints the status verb.
///
/// The verb says *that* work is in flight; which effect it uses says *what
/// kind*. A sweep is the model working on the text — one highlight travelling
/// over the word. A flash is the machine working instead, so nothing travels
/// and the whole word pulses together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerbAnim {
    /// A graded highlight travels across the word, one column per `step_ms`.
    /// `reverse` sends it left-to-right instead of right-to-left.
    Sweep {
        /// Left-to-right when `true`, right-to-left when `false`.
        reverse: bool,
        /// Milliseconds of travel per display column.
        step_ms: u64,
    },
    /// Every column takes the same colour, oscillating between the resting
    /// colour and the highlight on a sine of `period_ms`.
    Flash {
        /// Milliseconds for one full dim -> bright -> dim cycle.
        period_ms: u64,
    },
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_possible_wrap)]
mod tests {
    use super::*;

    #[test]
    fn reduced_motion_stops_the_clock() {
        let _motion = crate::anim::reduced_motion_test_guard();
        set_reduced_motion(false);
        assert!(clock_ms().is_some());
        set_reduced_motion(true);
        assert_eq!(clock_ms(), None);
        set_reduced_motion(false); // leave global clean for other tests
    }

    #[test]
    fn lerp_clamps_and_rounds() {
        assert_eq!(lerp_u8(0, 100, 0.0), 0);
        assert_eq!(lerp_u8(0, 100, 1.0), 100);
        assert_eq!(lerp_u8(0, 100, 0.5), 50);
        // t out of range is clamped, not extrapolated.
        assert_eq!(lerp_u8(0, 100, -1.0), 0);
        assert_eq!(lerp_u8(0, 100, 2.0), 100);
        assert_eq!(lerp_rgb((0, 0, 0), (255, 255, 255), 0.5), (128, 128, 128));
    }

    #[test]
    fn sine01_starts_mid_and_peaks_at_quarter_period() {
        // At t == delay the sine is at 0 → midpoint 0.5.
        assert!((sine01(0, 1000, 0) - 0.5).abs() < 1e-6);
        // Quarter period → peak (1.0).
        assert!((sine01(250, 1000, 0) - 1.0).abs() < 1e-3);
        // Three-quarter period → trough (0.0).
        assert!(sine01(750, 1000, 0) < 1e-3);
        // Before the start delay it rests dim.
        assert_eq!(sine01(100, 1000, 500), 0.0);
        // Degenerate period rests dim.
        assert_eq!(sine01(100, 0, 0), 0.0);
    }

    #[test]
    fn pulse_and_flash_hit_endpoints() {
        let dim = (10, 10, 10);
        let bright = (250, 250, 250);
        // Peak at quarter period → bright.
        assert_eq!(pulse_color(dim, bright, 250, 1000, 0), bright);
        // Trough at three-quarter period → dim.
        assert_eq!(pulse_color(dim, bright, 750, 1000, 0), dim);
        // Flash is pulse with no delay.
        assert_eq!(flash_color(dim, bright, 250, 1000), bright);
    }

    #[test]
    fn sweep_travels_right_to_left_and_rests_off_text() {
        let len = 5;
        let (step, gap) = (50, 3);
        // First step: center just off the right edge → nothing lit on-text.
        let cols0: Vec<i64> = (0..len as i64)
            .filter(|&c| sweep_contains(c, sweep_window(len, 0, step, gap, false)))
            .collect();
        assert!(cols0.is_empty(), "expected off-text start: {cols0:?}");
        // Enough steps in, the highlight reaches the left edge (center → 0).
        let mid_tick = (len as u64 + 1) * step;
        let cols_mid: Vec<i64> = (0..len as i64)
            .filter(|&c| sweep_contains(c, sweep_window(len, mid_tick, step, gap, false)))
            .collect();
        assert!(
            cols_mid.contains(&0),
            "expected left edge lit: {cols_mid:?}"
        );
        // The cycle length is len + gap columns; it repeats.
        let cycle = (len as u64 + gap as u64) * step;
        assert_eq!(
            sweep_window(len, 0, step, gap, false),
            sweep_window(len, cycle, step, gap, false)
        );
    }

    #[test]
    fn sweep_reverse_goes_left_to_right() {
        let len = 5;
        let (step, gap) = (50, 3);
        // Reverse: center starts left of the text and walks right.
        let w0 = sweep_window(len, 0, step, gap, true);
        assert!(w0.1 < 0, "reverse should start off the left edge: {w0:?}");
        // A few steps in, the window overlaps the middle of the text.
        let lit: Vec<i64> = (0..len as i64)
            .filter(|&c| sweep_contains(c, sweep_window(len, 3 * step, step, gap, true)))
            .collect();
        assert!(lit.contains(&2), "expected column 2 lit: {lit:?}");
    }

    #[test]
    fn throbber_pings_forward_then_back() {
        let n = THROBBER_FRAMES.len(); // 10 → period 18
        let step = 100;
        // Forward leg 0..=9.
        assert_eq!(throbber_index(0, step, n), 0);
        assert_eq!(throbber_index(step, step, n), 1);
        assert_eq!(throbber_index(9 * step, step, n), 9);
        // Backward leg 8,7,...,1.
        assert_eq!(throbber_index(10 * step, step, n), 8);
        assert_eq!(throbber_index(17 * step, step, n), 1);
        // Then it wraps to the start of the forward leg.
        assert_eq!(throbber_index(18 * step, step, n), 0);
        // Char accessor tracks the index.
        assert_eq!(throbber_char(0, step), THROBBER_FRAMES[0]);
        assert_eq!(throbber_char(9 * step, step), THROBBER_FRAMES[9]);
    }

    #[test]
    fn stall_fades_up_and_resets_instantly() {
        // Not stalled → always 0 regardless of prior intensity.
        assert_eq!(stall_next(0.9, 50, false, 1000), 0.0);
        // Stalled → fades toward 1 by dt/fade.
        let a = stall_next(0.0, 500, true, 1000);
        assert!((a - 0.5).abs() < 1e-6, "{a}");
        let b = stall_next(a, 500, true, 1000);
        assert!((b - 1.0).abs() < 1e-6, "{b}");
        // Clamps at 1.
        assert_eq!(stall_next(1.0, 500, true, 1000), 1.0);
        // Color endpoints.
        let base = (200, 200, 200);
        let red = (255, 0, 0);
        assert_eq!(stall_color(base, red, 0.0), base);
        assert_eq!(stall_color(base, red, 1.0), red);
    }

    #[test]
    fn indexed_and_cube_round_trip_through_the_theme_green() {
        // 106 is the theme color. Note it is #87af00 — `status::SHIMMER_RAMP`
        // long documented it as #87af5f, which is really 107.
        assert_eq!(indexed_to_rgb(106), (0x87, 0xaf, 0x00));
        assert_eq!(cube_index_rgb((0x87, 0xaf, 0x00)), 106);
        assert_eq!(indexed_to_rgb(107), (0x87, 0xaf, 0x5f));
        // 192 is the built-in ramp's brightest shade, #d7ff87.
        assert_eq!(indexed_to_rgb(192), (0xd7, 0xff, 0x87));
        assert_eq!(cube_index_rgb((0xd7, 0xff, 0x87)), 192);
    }

    #[test]
    fn the_ramp_ends_on_the_secondary_and_eases_from_the_accent() {
        let accent = indexed_to_rgb(106);
        let secondary = indexed_to_rgb(192);
        let ramp = shimmer_ramp(accent, secondary);
        // The center column is the secondary exactly.
        assert_eq!(ramp[2], 192);
        // The outermost column sits nearer the accent than the center does.
        let dist = |i: u8| {
            let c = indexed_to_rgb(i);
            let d = |a: u8, b: u8| f32::from(a) - f32::from(b);
            d(c.0, accent.0).powi(2) + d(c.1, accent.1).powi(2) + d(c.2, accent.2).powi(2)
        };
        assert!(
            dist(ramp[0]) < dist(ramp[2]),
            "ramp should ease out of the accent: {ramp:?}"
        );
    }

    #[test]
    fn an_indexed_accent_gives_the_same_ramp_as_its_hex_equivalent() {
        let from_index = shimmer_ramp(indexed_to_rgb(106), indexed_to_rgb(192));
        let from_hex = shimmer_ramp((0x87, 0xaf, 0x00), (0xd7, 0xff, 0x87));
        assert_eq!(from_index, from_hex);
    }

    #[test]
    fn a_derived_secondary_is_lighter_and_keeps_the_accent_hue() {
        for accent in [(0xc0, 0x40, 0x40), (0x87, 0xaf, 0x00), (0x40, 0x40, 0xc0)] {
            let d = derived_secondary(accent);
            let lum = |c: Rgb| {
                0.2126 * f32::from(c.0) + 0.7152 * f32::from(c.1) + 0.0722 * f32::from(c.2)
            };
            assert!(
                lum(d) > lum(accent),
                "{accent:?} -> {d:?} should be lighter"
            );
            // Short of white: the hue must survive the lift, or every accent
            // would shimmer the same washed-out color.
            assert!(d != (255, 255, 255), "{accent:?} washed out to white");
            // The dominant channel stays dominant.
            let arg_max = |c: Rgb| {
                if c.0 >= c.1 && c.0 >= c.2 {
                    0
                } else if c.1 >= c.2 {
                    1
                } else {
                    2
                }
            };
            assert_eq!(arg_max(d), arg_max(accent), "hue shifted: {accent:?}");
        }
    }

    #[test]
    fn a_ramp_may_run_dark_as_well_as_light() {
        // plank's own profile secondary is #444444, darker than the accent, so
        // the sweep reads as a shadow rather than a highlight. Nothing in the
        // ramp may assume the secondary is the brighter end.
        let ramp = shimmer_ramp(indexed_to_rgb(106), (0x44, 0x44, 0x44));
        assert_eq!(ramp[2], cube_index_rgb((0x44, 0x44, 0x44)));
        assert_ne!(ramp[0], ramp[2]);
    }

    #[test]
    fn reduced_pulse_toggles_on_fixed_cycle() {
        // Bright for the first half, dim for the second, repeating.
        assert!(reduced_pulse(0, 1000));
        assert!(reduced_pulse(499, 1000));
        assert!(!reduced_pulse(500, 1000));
        assert!(!reduced_pulse(999, 1000));
        assert!(reduced_pulse(1000, 1000));
    }

    #[test]
    fn verb_flash_rests_at_the_base_and_peaks_at_the_highlight() {
        let base: Rgb = (0x87, 0xaf, 0x5f);
        let bright: Rgb = (0xd7, 0xff, 0x87);
        let period = FLASH_PERIOD_MS;
        // sine01 is 0.5 at t=0, 1.0 a quarter period later, 0.0 at three
        // quarters. So the flash is fully bright at period/4 and fully at rest
        // at 3*period/4.
        assert_eq!(flash_color(base, bright, period / 4, period), bright);
        assert_eq!(flash_color(base, bright, 3 * period / 4, period), base);
        // In between it never leaves the interval between the two endpoints.
        for step in 0..40u64 {
            let c = flash_color(base, bright, step * 50, period);
            assert!(c.0 >= base.0 && c.0 <= bright.0, "r out of range: {c:?}");
            assert!(c.1 >= base.1 && c.1 <= bright.1, "g out of range: {c:?}");
            assert!(c.2 >= base.2 && c.2 <= bright.2, "b out of range: {c:?}");
        }
    }

    #[test]
    fn verb_anim_timings_are_the_values_the_spec_names() {
        // Pinned to literals. These three numbers are the whole difference
        // between the effects, so a drift in any of them should fail here
        // rather than quietly changing how the footer moves.
        assert_eq!(SWEEP_FAST_MS, 50);
        assert_eq!(SWEEP_SLOW_MS, 200);
        assert_eq!(FLASH_PERIOD_MS, 2000);
        // The prefill sweep must outpace the one it is meant to be
        // distinguishable from. Checked at compile time, so a regression is a
        // build error rather than a test failure.
        const { assert!(SWEEP_FAST_MS < SWEEP_SLOW_MS) }
        // The fields are readable by the pattern later tasks match on.
        let VerbAnim::Sweep { reverse, step_ms } = (VerbAnim::Sweep {
            reverse: true,
            step_ms: SWEEP_FAST_MS,
        }) else {
            panic!("constructed a Sweep but it did not match as one");
        };
        assert!(reverse);
        assert_eq!(step_ms, SWEEP_FAST_MS);
    }
}

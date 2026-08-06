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
//! Motion is Ratatui-only. The plain line REPL and `--non-interactive` paths
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

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_possible_wrap)]
mod tests {
    use super::*;

    #[test]
    fn reduced_motion_stops_the_clock() {
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
    fn reduced_pulse_toggles_on_fixed_cycle() {
        // Bright for the first half, dim for the second, repeating.
        assert!(reduced_pulse(0, 1000));
        assert!(reduced_pulse(499, 1000));
        assert!(!reduced_pulse(500, 1000));
        assert!(!reduced_pulse(999, 1000));
        assert!(reduced_pulse(1000, 1000));
    }
}

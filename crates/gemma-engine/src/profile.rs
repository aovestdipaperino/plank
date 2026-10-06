//! Opt-in wall-clock breakdown of the forward pass, for `examples/bench_decode`.
//!
//! Off unless `PLANK_GEMMA_PROFILE=1`. When on, every [`region`] synchronizes
//! the device before and after its body, so GPU work queued inside it is
//! charged to it rather than to whatever region next waits on the GPU. That
//! synchronization slows the forward pass itself, so the totals are a
//! breakdown, never a throughput.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use candle_core::{Device, Result};

static TOTALS: Mutex<Vec<(&'static str, Duration)>> = Mutex::new(Vec::new());
/// 0: follow the environment; 1: on; 2: off.
static FORCED: AtomicU8 = AtomicU8::new(0);

/// Whether regions are being timed: as last set by [`set_enabled`], else
/// whether `PLANK_GEMMA_PROFILE=1` (read once).
#[must_use]
pub fn enabled() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    match FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => *ENV.get_or_init(|| std::env::var("PLANK_GEMMA_PROFILE").as_deref() == Ok("1")),
    }
}

/// Turns profiling on or off regardless of the environment.
pub fn set_enabled(on: bool) {
    FORCED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Runs `body`, charging its time (device work included) to `name` when
/// profiling is on.
pub(crate) fn region<T>(
    name: &'static str,
    device: &Device,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if !enabled() {
        return body();
    }
    device.synchronize()?;
    let started = Instant::now();
    let out = body()?;
    device.synchronize()?;
    let spent = started.elapsed();
    let mut totals = TOTALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match totals.iter_mut().find(|(n, _)| *n == name) {
        Some((_, d)) => *d += spent,
        None => totals.push((name, spent)),
    }
    Ok(out)
}

/// The accumulated totals in first-seen order, clearing them.
#[must_use]
pub fn take() -> Vec<(&'static str, Duration)> {
    std::mem::take(
        &mut *TOTALS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

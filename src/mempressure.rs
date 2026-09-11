// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Memory-pressure sensing and the hysteresis that turns raw kernel levels
//! into yield/resume decisions.
//!
//! plank yields because *another* application is thrashing the machine: the
//! live KV session is anonymous and partly Metal-wired, so the kernel cannot
//! reclaim it, unlike the file-backed GGUF weights. Freeing the session is the
//! only memory plank can return that the kernel could not have taken itself.
//!
//! The decisions are asymmetric on purpose. Yielding is urgent, so it happens
//! on the first `Critical`. Resuming is expensive — it costs a re-prefill — so
//! it waits for a sustained `Normal`, and a minimum interval between yields
//! stops a re-prefill that re-triggers pressure from becoming a livelock.

/// Seconds of `Critical` required before yielding. Zero: the whole point is to
/// return wired memory to a thrashing system as fast as possible.
pub const YIELD_DWELL_SECS: u64 = 0;

/// Seconds of continuous `Normal` required before re-acquiring the session.
/// Long, because resuming costs a re-prefill and a premature resume pays it
/// twice.
pub const RESUME_DWELL_SECS: u64 = 30;

/// Minimum seconds between two yields. The livelock guard: a resume's own
/// re-prefill can re-trigger pressure, and yielding again immediately would
/// spend the session cycling instead of working. Being killed by the OOM
/// killer is preferable to an hour of re-prefilling in a loop.
pub const MIN_YIELD_INTERVAL_SECS: u64 = 120;

/// Kernel memory-pressure level, mirroring the dispatch source's three states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    Normal,
    Warn,
    Critical,
}

/// What the policy layer should do about the level just observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Nothing to do.
    Hold,
    /// Drop pure cache (ladder rungs). Invisible: costs only a deeper
    /// re-prefill later, which may never happen.
    ShedCache,
    /// Stop at a token boundary and free the live session.
    Yield,
    /// Re-acquire the session and restore from disk.
    Resume,
}

/// Turns a stream of raw levels into decisions, applying both dwells and the
/// minimum-interval guard.
///
/// Time is passed in rather than read, so the whole state machine is testable
/// with a mock clock and carries no platform dependency.
#[derive(Debug)]
pub struct Hysteresis {
    /// True between a `Yield` and its `Resume`.
    yielded: bool,
    /// True once cache has been shed for the current non-`Normal` episode.
    shed: bool,
    /// When the current run of `Normal` began, while yielded.
    normal_since: Option<u64>,
    /// When the current run of `Critical` began, while not yielded.
    critical_since: Option<u64>,
    /// Timestamp of the last `Yield`, for the minimum-interval guard.
    last_yield: Option<u64>,
}

impl Default for Hysteresis {
    fn default() -> Self {
        Self::new()
    }
}

impl Hysteresis {
    #[must_use]
    pub fn new() -> Self {
        Self {
            yielded: false,
            shed: false,
            normal_since: None,
            critical_since: None,
            last_yield: None,
        }
    }

    /// True while plank has given its session back to the system.
    #[must_use]
    pub fn is_yielded(&self) -> bool {
        self.yielded
    }

    /// Records a yield that happened outside [`Self::observe`].
    ///
    /// A mid-generation yield is raised by the cancel callback during the
    /// pass, not by a turn-boundary poll. Without this the state machine would
    /// still believe it holds a session: it would offer a second `Yield` with
    /// nothing left to free, and never the matching `Resume`.
    pub fn note_external_yield(&mut self, now_secs: u64) {
        self.yielded = true;
        self.shed = true;
        self.normal_since = None;
        self.critical_since = None;
        self.last_yield = Some(now_secs);
    }

    /// Observes one level reading and returns the action it implies.
    pub fn observe(&mut self, level: PressureLevel, now_secs: u64) -> Decision {
        if level == PressureLevel::Normal {
            self.shed = false;
            self.critical_since = None;
            if !self.yielded {
                self.normal_since = None;
                return Decision::Hold;
            }
            let since = *self.normal_since.get_or_insert(now_secs);
            if now_secs.saturating_sub(since) > RESUME_DWELL_SECS {
                self.yielded = false;
                self.normal_since = None;
                return Decision::Resume;
            }
            return Decision::Hold;
        }

        // Any non-Normal level clears a resume dwell in progress. The clock
        // restarts from the next observed Normal, not from this reading: the
        // dwell has to be evidence that the machine was quiet, and only an
        // actual Normal observation is that evidence.
        self.normal_since = None;

        if self.yielded {
            // Already given back everything there is to give.
            return Decision::Hold;
        }

        if level == PressureLevel::Warn {
            self.critical_since = None;
            if self.shed {
                return Decision::Hold;
            }
            self.shed = true;
            return Decision::ShedCache;
        }

        // Critical, and not yet yielded.
        let since = *self.critical_since.get_or_insert(now_secs);
        // YIELD_DWELL_SECS is 0 by design (see its doc comment), so this is
        // always false today; kept so a future non-zero dwell needs no other
        // change here.
        #[allow(clippy::absurd_extreme_comparisons)]
        if now_secs.saturating_sub(since) < YIELD_DWELL_SECS {
            return Decision::Hold;
        }
        if let Some(last) = self.last_yield
            && now_secs.saturating_sub(last) < MIN_YIELD_INTERVAL_SECS
        {
            return Decision::Hold;
        }
        self.yielded = true;
        self.shed = true;
        self.critical_since = None;
        self.last_yield = Some(now_secs);
        Decision::Yield
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

impl PressureLevel {
    fn as_u8(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Warn => 1,
            Self::Critical => 2,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Warn,
            2 => Self::Critical,
            _ => Self::Normal,
        }
    }
}

/// Publishes the kernel's current memory-pressure level.
///
/// On macOS a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source writes the atomic
/// from its own queue; everywhere else the atomic simply stays at `Normal`, so
/// every consumer compiles and behaves as though the machine were never under
/// pressure.
#[derive(Debug, Clone)]
pub struct PressureSensor {
    level: Arc<AtomicU8>,
}

impl PressureSensor {
    /// Starts the sensor. Safe to call more than once; each call installs its
    /// own source, and plank installs exactly one at startup.
    #[must_use]
    pub fn start() -> Self {
        let level = Arc::new(AtomicU8::new(PressureLevel::Normal.as_u8()));
        #[cfg(target_os = "macos")]
        macos::install(&level);
        Self { level }
    }

    /// The most recent level the kernel reported.
    #[must_use]
    pub fn level(&self) -> PressureLevel {
        PressureLevel::from_u8(self.level.load(Ordering::SeqCst))
    }

    /// Drives the level directly, for tests and for `--force-pressure`-style
    /// manual exercise. Never called by the dispatch source.
    pub fn set_for_test(&self, level: PressureLevel) {
        self.level.store(level.as_u8(), Ordering::SeqCst);
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{Arc, AtomicU8, Ordering, PressureLevel};
    use std::ffi::c_void;

    // libdispatch is part of libSystem, which every Rust binary already links
    // on macOS, so these need no build.rs change.
    const DISPATCH_MEMORYPRESSURE_NORMAL: usize = 0x01;
    const DISPATCH_MEMORYPRESSURE_WARN: usize = 0x02;
    const DISPATCH_MEMORYPRESSURE_CRITICAL: usize = 0x04;

    unsafe extern "C" {
        #[link_name = "_dispatch_source_type_memorypressure"]
        static SOURCE_TYPE_MEMORYPRESSURE: c_void;

        fn dispatch_source_create(
            ty: *const c_void,
            handle: usize,
            mask: usize,
            queue: *mut c_void,
        ) -> *mut c_void;
        fn dispatch_source_set_event_handler_f(
            source: *mut c_void,
            handler: Option<unsafe extern "C" fn(*mut c_void)>,
        );
        fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
        fn dispatch_source_get_data(source: *mut c_void) -> usize;
        fn dispatch_resume(object: *mut c_void);
        fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
    }

    /// Leaked on purpose: the source lives for the process, and the handler
    /// dereferences this pointer from libdispatch's queue at arbitrary times.
    /// A drop would be a use-after-free with no upside — plank has exactly one
    /// sensor and it is meant to outlive every turn.
    struct Ctx {
        level: Arc<AtomicU8>,
        source: *mut c_void,
    }

    unsafe extern "C" fn on_event(ud: *mut c_void) {
        // SAFETY: ud is the leaked Ctx we installed with dispatch_set_context.
        let ctx = unsafe { &*ud.cast::<Ctx>() };
        // SAFETY: source is the live dispatch source that invoked us.
        let data = unsafe { dispatch_source_get_data(ctx.source) };
        let level = if data & DISPATCH_MEMORYPRESSURE_CRITICAL != 0 {
            PressureLevel::Critical
        } else if data & DISPATCH_MEMORYPRESSURE_WARN != 0 {
            PressureLevel::Warn
        } else {
            PressureLevel::Normal
        };
        ctx.level.store(level.as_u8(), Ordering::SeqCst);
    }

    pub(super) fn install(level: &Arc<AtomicU8>) {
        let mask = DISPATCH_MEMORYPRESSURE_NORMAL
            | DISPATCH_MEMORYPRESSURE_WARN
            | DISPATCH_MEMORYPRESSURE_CRITICAL;
        // SAFETY: all four calls are the documented libdispatch construction
        // sequence; the queue is a global queue that always exists, and the
        // context outlives the source because it is leaked.
        unsafe {
            let queue = dispatch_get_global_queue(0, 0);
            let source = dispatch_source_create(
                std::ptr::addr_of!(SOURCE_TYPE_MEMORYPRESSURE),
                0,
                mask,
                queue,
            );
            if source.is_null() {
                // No sensor means plank never yields, which is the old
                // behaviour. Not worth failing startup over.
                return;
            }
            let ctx = Box::into_raw(Box::new(Ctx {
                level: Arc::clone(level),
                source,
            }));
            dispatch_set_context(source, ctx.cast::<c_void>());
            dispatch_source_set_event_handler_f(source, Some(on_event));
            dispatch_resume(source);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_yields_immediately() {
        let mut h = Hysteresis::new();
        assert_eq!(h.observe(PressureLevel::Normal, 0), Decision::Hold);
        assert_eq!(
            h.observe(PressureLevel::Critical, 1),
            Decision::Yield,
            "yielding is the urgent direction; it must not wait on a dwell"
        );
    }

    #[test]
    fn warn_sheds_cache_without_yielding() {
        let mut h = Hysteresis::new();
        assert_eq!(h.observe(PressureLevel::Warn, 0), Decision::ShedCache);
        assert_eq!(
            h.observe(PressureLevel::Warn, 1),
            Decision::Hold,
            "shedding is edge-triggered: there is nothing left to drop"
        );
    }

    #[test]
    fn resume_waits_for_the_full_dwell() {
        let mut h = Hysteresis::new();
        h.observe(PressureLevel::Critical, 0);
        assert_eq!(
            h.observe(PressureLevel::Normal, 1),
            Decision::Hold,
            "the first Normal only starts the clock; it proves nothing yet"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 1 + RESUME_DWELL_SECS),
            Decision::Hold,
            "the dwell is measured from the first Normal, and is exclusive"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 2 + RESUME_DWELL_SECS),
            Decision::Resume
        );
    }

    #[test]
    fn a_dip_back_to_critical_restarts_the_resume_dwell() {
        let mut h = Hysteresis::new();
        h.observe(PressureLevel::Critical, 0);
        h.observe(PressureLevel::Normal, 1);
        assert_eq!(
            h.observe(PressureLevel::Critical, 2),
            Decision::Hold,
            "already yielded: a second Critical has nothing to free"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 2 + RESUME_DWELL_SECS),
            Decision::Hold,
            "the dwell restarts at the dip, so it has not elapsed yet"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 3 + 2 * RESUME_DWELL_SECS),
            Decision::Resume
        );
    }

    #[test]
    fn a_flap_produces_exactly_one_yield() {
        let mut h = Hysteresis::new();
        let mut yields = 0;
        for (t, level) in [
            (0, PressureLevel::Warn),
            (1, PressureLevel::Critical),
            (2, PressureLevel::Warn),
            (3, PressureLevel::Critical),
            (4, PressureLevel::Warn),
            (5, PressureLevel::Critical),
        ] {
            if h.observe(level, t) == Decision::Yield {
                yields += 1;
            }
        }
        assert_eq!(yields, 1, "a flap must not yield once per spike");
    }

    #[test]
    fn an_external_yield_is_reconciled() {
        let mut h = Hysteresis::new();
        // A mid-generation yield happens through the cancel callback, not
        // through observe(); the state machine has to be told.
        h.note_external_yield(0);
        assert!(h.is_yielded());
        assert_eq!(
            h.observe(PressureLevel::Critical, 1),
            Decision::Hold,
            "already yielded: there is nothing left to free"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 2 + RESUME_DWELL_SECS),
            Decision::Hold,
            "the first Normal after the dip only starts the clock"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 3 + 2 * RESUME_DWELL_SECS),
            Decision::Resume,
            "a mid-pass yield must still get its matching resume"
        );
    }

    #[test]
    fn min_interval_suppresses_a_second_yield() {
        let mut h = Hysteresis::new();
        assert_eq!(h.observe(PressureLevel::Critical, 0), Decision::Yield);
        // Resume needs two Normals spanning the dwell, then go critical again
        // well inside the guard window.
        assert_eq!(
            h.observe(PressureLevel::Normal, 1 + RESUME_DWELL_SECS),
            Decision::Hold
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 2 + 2 * RESUME_DWELL_SECS),
            Decision::Resume
        );
        assert_eq!(
            h.observe(PressureLevel::Critical, 3 + 2 * RESUME_DWELL_SECS),
            Decision::Hold,
            "a re-prefill that re-triggers pressure must not yield again at once"
        );
        assert_eq!(
            h.observe(PressureLevel::Critical, 1 + MIN_YIELD_INTERVAL_SECS),
            Decision::Yield,
            "past the guard window, pressure is actionable again"
        );
    }

    #[test]
    fn a_single_normal_reading_never_resumes() {
        let mut h = Hysteresis::new();
        h.observe(PressureLevel::Critical, 0);
        assert_eq!(
            h.observe(PressureLevel::Normal, 10_000),
            Decision::Hold,
            "one sample proves time passed, not that the machine was quiet"
        );
    }

    #[test]
    fn a_fresh_sensor_reports_normal() {
        let s = PressureSensor::start();
        assert_eq!(
            s.level(),
            PressureLevel::Normal,
            "no reading yet must never look like pressure"
        );
    }

    #[test]
    fn the_sensor_round_trips_every_level() {
        let s = PressureSensor::start();
        for level in [
            PressureLevel::Warn,
            PressureLevel::Critical,
            PressureLevel::Normal,
        ] {
            s.set_for_test(level);
            assert_eq!(s.level(), level);
        }
    }
}

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
        self.normal_since = Some(now_secs);
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

        // Any non-Normal level interrupts a resume dwell in progress; while
        // already yielded, a dip back into pressure restarts the dwell clock
        // rather than clearing it, since the dwell must run from *this*
        // reading, not from whenever the next Normal happens to be observed.
        self.normal_since = if self.yielded { Some(now_secs) } else { None };

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
        self.normal_since = Some(now_secs);
        self.critical_since = None;
        self.last_yield = Some(now_secs);
        Decision::Yield
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
        assert_eq!(h.observe(PressureLevel::Normal, 1), Decision::Hold);
        assert_eq!(
            h.observe(PressureLevel::Normal, RESUME_DWELL_SECS),
            Decision::Hold,
            "the dwell is measured from the first Normal, and is exclusive"
        );
        assert_eq!(
            h.observe(PressureLevel::Normal, 1 + RESUME_DWELL_SECS),
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
            Decision::Resume,
            "a mid-pass yield must still get its matching resume"
        );
    }

    #[test]
    fn min_interval_suppresses_a_second_yield() {
        let mut h = Hysteresis::new();
        assert_eq!(h.observe(PressureLevel::Critical, 0), Decision::Yield);
        // Resume, then go critical again well inside the guard window.
        assert_eq!(
            h.observe(PressureLevel::Normal, 1 + RESUME_DWELL_SECS),
            Decision::Resume
        );
        assert_eq!(
            h.observe(PressureLevel::Critical, 2 + RESUME_DWELL_SECS),
            Decision::Hold,
            "a re-prefill that re-triggers pressure must not yield again at once"
        );
        assert_eq!(
            h.observe(PressureLevel::Critical, 1 + MIN_YIELD_INTERVAL_SECS),
            Decision::Yield,
            "past the guard window, pressure is actionable again"
        );
    }
}

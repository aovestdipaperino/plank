// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Per-model peak throughput for the current session.
//!
//! Every turn reports how fast it prefilled and decoded, but those numbers
//! scroll past and are gone. This keeps the best seen for each model and
//! prints it in the session's exit message.
//!
//! Scoped to the session on purpose: nothing is written to disk and nothing
//! carries over. A peak from last week was measured on a different engine
//! build, a different context length, and a cooler machine, so comparing
//! against it silently is worse than not comparing at all.
//!
//! Records are per **model**, not per run: an 87 GB Flash quant and a
//! draft-assisted config are different machines as far as throughput goes, and
//! averaging them would describe neither.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// Seconds a phase must run before its rate is treated as representative.
/// Shared with generation, which measures its own steady rate inside the
/// engine loop — see [`crate::engine::STEADY_WARMUP_SECS`].
const WARMUP_SECS: f64 = crate::engine::STEADY_WARMUP_SECS;

/// Tokens a phase must cover *after* the warmup before its rate is believable.
/// A couple of tokens divided by a sliver of a second is noise, not a peak.
const MIN_TOKENS: i32 = 8;

/// Best throughput seen for one model, in tokens per second.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Record {
    /// Peak prefill rate, or 0 when none has been recorded.
    pub prefill_tps: f64,
    /// Peak generation rate, or 0 when none has been recorded.
    pub gen_tps: f64,
}

impl Record {
    /// True when nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.prefill_tps <= 0.0 && self.gen_tps <= 0.0
    }
}

/// Where a prefill pass stood when it crossed the warmup: tokens done and
/// seconds elapsed at that moment. Rates are measured from here, not from the
/// start of the pass.
#[derive(Debug, Clone, Copy)]
struct PrefillMark {
    done: i32,
    secs: f64,
}

/// One model's peaks plus the in-flight prefill pass, if any.
#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    peaks: Record,
    /// Set once the current prefill pass passes the warmup; cleared when a new
    /// pass starts.
    prefill_mark: Option<PrefillMark>,
    /// Tokens reported by the previous prefill sample, to spot a new pass.
    prefill_last_done: i32,
}

/// This session's peaks, keyed by model name.
static SESSION: Mutex<BTreeMap<String, Entry>> = Mutex::new(BTreeMap::new());

/// Notes one prefill progress sample: `done` tokens so far in this pass at a
/// cumulative rate of `tps`.
///
/// Called for every progress event, not only the last, because the rate that
/// matters is the one measured *after* the pass has been running for
/// [`WARMUP_SECS`] — which cannot be recovered from the final aggregate alone.
/// A `done` that goes backwards means a new pass has started.
pub fn note_prefill_progress(model: &str, done: i32, tps: f64) {
    if model.is_empty() || !tps.is_finite() || tps <= 0.0 || done <= 0 {
        return;
    }
    let secs = f64::from(done) / tps;
    with_entry(model, |e| {
        if done < e.prefill_last_done {
            // A new pass: the previous mark describes work already finished.
            e.prefill_mark = None;
        }
        e.prefill_last_done = done;
        let Some(mark) = e.prefill_mark else {
            if secs >= WARMUP_SECS {
                e.prefill_mark = Some(PrefillMark { done, secs });
            }
            return;
        };
        let tokens = done - mark.done;
        let span = secs - mark.secs;
        if tokens < MIN_TOKENS || span <= 0.0 {
            return;
        }
        let steady = f64::from(tokens) / span;
        if steady > e.peaks.prefill_tps {
            e.peaks.prefill_tps = steady;
        }
    });
}

/// Notes a generation pass's steady rate, as measured by the engine after its
/// own warmup. Zero means the pass never ran long enough to have one, which is
/// also what online providers report — they do not measure local throughput.
pub fn note_generation(model: &str, steady_tps: f64) {
    if model.is_empty() || !steady_tps.is_finite() || steady_tps <= 0.0 {
        return;
    }
    with_entry(model, |e| {
        if steady_tps > e.peaks.gen_tps {
            e.peaks.gen_tps = steady_tps;
        }
    });
}

/// Runs `f` against `model`'s entry, creating it if needed.
fn with_entry(model: &str, f: impl FnOnce(&mut Entry)) {
    if let Ok(mut map) = SESSION.lock() {
        f(map.entry(model.to_string()).or_default());
    }
}

/// This session's peaks for `model`.
#[must_use]
pub fn session_best(model: &str) -> Record {
    SESSION
        .lock()
        .ok()
        .and_then(|m| m.get(model).map(|e| e.peaks))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test uses its own model key: [`SESSION`] is process-global and the
    /// suite runs in parallel, so sharing a key makes tests clobber each other
    /// rather than test anything.
    #[test]
    fn a_prefill_pass_shorter_than_the_warmup_records_nothing() {
        // 100 tokens at 700 tok/s is 0.14s — never reaches the mark.
        note_prefill_progress("short", 100, 700.0);
        assert!(session_best("short").is_empty());
    }

    #[test]
    fn prefill_is_measured_from_the_warmup_mark_onward() {
        // Mark lands at 2000 tokens / 2.0s. Then 2000 more tokens arrive by
        // 4.0s: 2000 tokens in 2.0s is 1000 tok/s, even though the pass
        // average is only 1000 as well — here they coincide by construction.
        note_prefill_progress("mark", 2000, 1000.0);
        note_prefill_progress("mark", 4000, 1000.0);
        assert!((session_best("mark").prefill_tps - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn the_slow_opening_is_excluded_from_the_prefill_rate() {
        // 200 tokens in the first 2.0s (100 tok/s), then 1000 more in 1.0s.
        // Whole-pass average is 1200/3.0 = 400; the steady rate is 1000.
        note_prefill_progress("steady", 200, 100.0);
        note_prefill_progress("steady", 1200, 400.0);
        let got = session_best("steady").prefill_tps;
        assert!(
            (got - 1000.0).abs() < 1e-6,
            "expected the post-mark rate, got {got}"
        );
    }

    #[test]
    fn a_new_pass_resets_the_mark() {
        note_prefill_progress("reset", 2000, 1000.0);
        note_prefill_progress("reset", 4000, 1000.0);
        // `done` going backwards means a fresh pass; its own short prefix must
        // not be measured against the previous pass's mark.
        note_prefill_progress("reset", 10, 1000.0);
        assert!((session_best("reset").prefill_tps - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn a_sliver_after_the_mark_is_not_a_peak() {
        note_prefill_progress("sliver", 2000, 1000.0);
        // One token past the mark: a real rate cannot be read off that.
        note_prefill_progress("sliver", 2001, 1000.4);
        assert!(session_best("sliver").is_empty());
    }

    #[test]
    fn generation_records_the_engine_measured_steady_rate() {
        note_generation("gen", 42.0);
        note_generation("gen", 12.0);
        assert!((session_best("gen").gen_tps - 42.0).abs() < 1e-9);
    }

    #[test]
    fn a_pass_with_no_steady_rate_records_nothing() {
        // The engine reports 0 when the pass never cleared its own warmup.
        note_generation("nosteady", 0.0);
        assert!(session_best("nosteady").is_empty());
    }

    #[test]
    fn nonsense_rates_are_refused() {
        note_generation("nonsense", f64::NAN);
        note_generation("nonsense", f64::INFINITY);
        note_generation("nonsense", -5.0);
        note_prefill_progress("nonsense", 2000, f64::NAN);
        assert!(session_best("nonsense").is_empty());
    }

    #[test]
    fn the_stub_engine_records_nothing() {
        // `EchoEngine::model_name` is empty; it has no throughput to speak of.
        note_generation("", 999.0);
        note_prefill_progress("", 5000, 999.0);
        assert!(session_best("").is_empty());
    }

    #[test]
    fn models_are_recorded_separately() {
        note_generation("sep-flash", 42.0);
        note_generation("sep-other", 7.0);
        assert!((session_best("sep-flash").gen_tps - 42.0).abs() < 1e-9);
        assert!((session_best("sep-other").gen_tps - 7.0).abs() < 1e-9);
    }

    #[test]
    fn an_unseen_model_has_no_record() {
        assert!(session_best("never-seen-model").is_empty());
    }
}

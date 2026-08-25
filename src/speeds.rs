// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Per-model time and throughput for the current session.
//!
//! Every turn reports how fast it prefilled and decoded, but those numbers
//! scroll past and are gone. This keeps a running total for each model —
//! seconds and tokens spent prefilling, seconds and tokens spent generating,
//! seconds spent running tools — and prints it in the session's exit message.
//!
//! Totals, not peaks: a peak is one lucky pass and says nothing about how the
//! session actually went. An average over the whole run, shown next to the
//! time each phase cost, is the number that describes where a session spent
//! itself.
//!
//! Scoped to the session on purpose: nothing is written to disk and nothing
//! carries over. Figures from last week were measured on a different engine
//! build, a different context length, and a cooler machine, so comparing
//! against them silently is worse than not comparing at all.
//!
//! Records are per **model**, not per run: an 87 GB Flash quant and a
//! draft-assisted config are different machines as far as throughput goes, and
//! averaging them together would describe neither.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// Totals accumulated for one model, in tokens and seconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Record {
    /// Tokens prefilled across every pass.
    pub prefill_tokens: i64,
    /// Wall-clock seconds spent prefilling.
    pub prefill_secs: f64,
    /// Tokens generated across every pass.
    pub gen_tokens: i64,
    /// Wall-clock seconds spent generating.
    pub gen_secs: f64,
    /// Wall-clock seconds spent dispatching tool calls.
    pub tool_secs: f64,
}

impl Record {
    /// True when nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.prefill_secs <= 0.0 && self.gen_secs <= 0.0 && self.tool_secs <= 0.0
    }

    /// Average prefill rate over the session, or 0 when nothing prefilled.
    #[must_use]
    pub fn prefill_tps(self) -> f64 {
        rate(self.prefill_tokens, self.prefill_secs)
    }

    /// Average generation rate over the session, or 0 when nothing generated.
    #[must_use]
    pub fn gen_tps(self) -> f64 {
        rate(self.gen_tokens, self.gen_secs)
    }
}

/// Tokens over seconds, guarding the degenerate cases.
fn rate(tokens: i64, secs: f64) -> f64 {
    if tokens <= 0 || secs <= 0.0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let tokens = tokens as f64;
    tokens / secs
}

/// One model's totals plus the in-flight prefill pass, if any.
#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    totals: Record,
    /// Last prefill sample of the pass currently running: tokens done and the
    /// seconds elapsed at that moment. Deltas against it are what get folded
    /// into the totals, so a pass contributes exactly once however many
    /// progress events it emits.
    prefill_last: Option<(i32, f64)>,
}

/// This session's totals, keyed by model name.
static SESSION: Mutex<BTreeMap<String, Entry>> = Mutex::new(BTreeMap::new());

/// Notes one prefill progress sample: `done` tokens so far in this pass at a
/// cumulative rate of `tps`.
///
/// Called for every progress event, not only the last, because a pass that is
/// interrupted or ends without a final sample still cost the time it cost. A
/// `done` that goes backwards means a new pass has started.
pub fn note_prefill_progress(model: &str, done: i32, tps: f64) {
    if model.is_empty() || !tps.is_finite() || tps <= 0.0 || done <= 0 {
        return;
    }
    let secs = f64::from(done) / tps;
    with_entry(model, |e| {
        let (base_done, base_secs) = match e.prefill_last {
            // A `done` that shrank belongs to a fresh pass; measure it from
            // zero rather than against the previous pass's tail.
            Some((last_done, last_secs)) if done >= last_done => (last_done, last_secs),
            _ => (0, 0.0),
        };
        e.prefill_last = Some((done, secs));
        let tokens = i64::from(done - base_done);
        let span = secs - base_secs;
        if tokens <= 0 || span <= 0.0 {
            return;
        }
        e.totals.prefill_tokens += tokens;
        e.totals.prefill_secs += span;
    });
}

/// Notes a completed generation pass: `generated` tokens at a wall-clock rate
/// of `tps`. Zero on either means there is nothing to time — which is what
/// online providers report, since they do not measure local throughput.
pub fn note_generation(model: &str, generated: i32, tps: f64) {
    if model.is_empty() || !tps.is_finite() || tps <= 0.0 || generated <= 0 {
        return;
    }
    let secs = f64::from(generated) / tps;
    with_entry(model, |e| {
        e.totals.gen_tokens += i64::from(generated);
        e.totals.gen_secs += secs;
        // The next prefill sample starts a new pass whatever its `done`.
        e.prefill_last = None;
    });
}

/// Notes `secs` of wall-clock spent running tool calls for `model`.
pub fn note_tool_time(model: &str, secs: f64) {
    if model.is_empty() || !secs.is_finite() || secs <= 0.0 {
        return;
    }
    with_entry(model, |e| e.totals.tool_secs += secs);
}

/// Runs `f` against `model`'s entry, creating it if needed.
fn with_entry(model: &str, f: impl FnOnce(&mut Entry)) {
    if let Ok(mut map) = SESSION.lock() {
        f(map.entry(model.to_string()).or_default());
    }
}

/// This session's totals for `model`.
#[must_use]
pub fn session_totals(model: &str) -> Record {
    SESSION
        .lock()
        .ok()
        .and_then(|m| m.get(model).map(|e| e.totals))
        .unwrap_or_default()
}

/// Every model that recorded something this session, in name order.
#[must_use]
pub fn session_all() -> Vec<(String, Record)> {
    SESSION.lock().map_or_else(
        |_| Vec::new(),
        |m| {
            m.iter()
                .filter(|(_, e)| !e.totals.is_empty())
                .map(|(name, e)| (name.clone(), e.totals))
                .collect()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test uses its own model key: [`SESSION`] is process-global and the
    /// suite runs in parallel, so sharing a key makes tests clobber each other
    /// rather than test anything.
    #[test]
    fn a_prefill_pass_is_counted_once_however_many_samples_it_emits() {
        // 1000 tokens at 500 tok/s is 2.0s, reported in two samples.
        note_prefill_progress("once", 500, 500.0);
        note_prefill_progress("once", 1000, 500.0);
        let r = session_totals("once");
        assert_eq!(r.prefill_tokens, 1000);
        assert!((r.prefill_secs - 2.0).abs() < 1e-9);
        assert!((r.prefill_tps() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn the_slow_opening_is_part_of_the_average() {
        // 200 tokens in the first 2.0s, then 1000 more in 1.0s: the average is
        // the whole pass, 1200 tokens over 3.0s.
        note_prefill_progress("avg", 200, 100.0);
        note_prefill_progress("avg", 1200, 400.0);
        let got = session_totals("avg").prefill_tps();
        assert!((got - 400.0).abs() < 1e-6, "expected 400 tok/s, got {got}");
    }

    #[test]
    fn a_new_pass_is_measured_from_zero() {
        note_prefill_progress("fresh", 2000, 1000.0);
        // `done` going backwards means a fresh pass: 10 tokens in 0.01s, not a
        // negative delta against the previous pass.
        note_prefill_progress("fresh", 10, 1000.0);
        let r = session_totals("fresh");
        assert_eq!(r.prefill_tokens, 2010);
        assert!((r.prefill_secs - 2.01).abs() < 1e-9);
    }

    #[test]
    fn passes_accumulate_across_the_session() {
        note_generation("acc", 100, 50.0); // 2.0s
        note_generation("acc", 200, 40.0); // 5.0s
        let r = session_totals("acc");
        assert_eq!(r.gen_tokens, 300);
        assert!((r.gen_secs - 7.0).abs() < 1e-9);
        assert!((r.gen_tps() - 300.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn a_generation_pass_ends_the_prefill_pass() {
        note_prefill_progress("split", 1000, 500.0);
        note_generation("split", 10, 10.0);
        // Same `done` again: a new pass, not a zero-token delta.
        note_prefill_progress("split", 1000, 500.0);
        assert_eq!(session_totals("split").prefill_tokens, 2000);
    }

    #[test]
    fn tool_time_accumulates() {
        note_tool_time("tools", 1.5);
        note_tool_time("tools", 0.5);
        assert!((session_totals("tools").tool_secs - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_pass_with_no_rate_records_nothing() {
        // The engine reports 0 when the pass produced no measurable rate.
        note_generation("norate", 10, 0.0);
        note_generation("norate", 0, 10.0);
        assert!(session_totals("norate").is_empty());
    }

    #[test]
    fn nonsense_rates_are_refused() {
        note_generation("nonsense", 10, f64::NAN);
        note_generation("nonsense", 10, f64::INFINITY);
        note_generation("nonsense", 10, -5.0);
        note_prefill_progress("nonsense", 2000, f64::NAN);
        note_tool_time("nonsense", f64::NAN);
        assert!(session_totals("nonsense").is_empty());
    }

    #[test]
    fn the_stub_engine_records_nothing() {
        // `EchoEngine::model_name` is empty; it has no throughput to speak of.
        note_generation("", 500, 999.0);
        note_prefill_progress("", 5000, 999.0);
        note_tool_time("", 3.0);
        assert!(session_totals("").is_empty());
    }

    #[test]
    fn models_are_recorded_separately() {
        note_generation("sep-flash", 420, 42.0);
        note_generation("sep-other", 70, 7.0);
        assert_eq!(session_totals("sep-flash").gen_tokens, 420);
        assert_eq!(session_totals("sep-other").gen_tokens, 70);
        let all = session_all();
        assert!(all.iter().any(|(m, _)| m == "sep-flash"));
        assert!(all.iter().any(|(m, _)| m == "sep-other"));
    }

    #[test]
    fn an_unseen_model_has_no_record() {
        assert!(session_totals("never-seen-model").is_empty());
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Per-model peak throughput records.
//!
//! Every turn reports how fast it prefilled and decoded, but those numbers
//! scroll past and are gone. This keeps the best one ever seen for each model
//! in `~/.plank/model-speeds.json` and prints it in the session's exit
//! message, so "is this build faster than the last one" has an answer that
//! does not depend on remembering a number from last week.
//!
//! Records are per **model**, not per run: an 87 GB Flash quant and a 6 GB
//! draft-assisted config are different machines as far as throughput goes, and
//! averaging them would describe neither.
//!
//! Peaks are collected in memory during the session and merged into the file
//! once, at exit. Turns stay free of file IO, at the cost of losing a record
//! if the process is killed outright.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// State file under `~/.plank`.
const FILE: &str = "model-speeds.json";

/// Prefill tokens a pass must cover before its rate is believable.
///
/// A pass of a handful of tokens is mostly fixed setup cost, which divides out
/// to a rate far above anything the model sustains. Recording that as a peak
/// would poison the record permanently — it is a high-water mark, so a single
/// bad sample never washes out.
const MIN_PREFILL_TOKENS: i32 = 64;

/// Generated tokens a pass must produce before its rate is believable. Lower
/// than the prefill floor because decode is measured over its own steady loop.
const MIN_GEN_TOKENS: i32 = 32;

/// Best throughput seen for one model, in tokens per second.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Peak prefill rate, or 0 when none has been recorded.
    #[serde(default)]
    pub prefill_tps: f64,
    /// Peak generation rate, or 0 when none has been recorded.
    #[serde(default)]
    pub gen_tps: f64,
}

impl Record {
    /// Raises each field to the larger of the two.
    fn merge(&mut self, other: Self) {
        if other.prefill_tps > self.prefill_tps {
            self.prefill_tps = other.prefill_tps;
        }
        if other.gen_tps > self.gen_tps {
            self.gen_tps = other.gen_tps;
        }
    }

    /// True when nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.prefill_tps <= 0.0 && self.gen_tps <= 0.0
    }
}

/// This session's peaks, keyed by model name. Flushed by [`commit`].
static SESSION: Mutex<BTreeMap<String, Record>> = Mutex::new(BTreeMap::new());

/// Path to the records file, or `None` when `$HOME` is unset.
#[must_use]
pub fn path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".plank").join(FILE))
}

/// Notes a completed prefill pass.
///
/// `tokens` is what this pass actually evaluated, so a mostly cache-hit prompt
/// contributes only the suffix it really prefilled. Ignored below
/// [`MIN_PREFILL_TOKENS`], and for the stub engine, which has no model.
pub fn note_prefill(model: &str, tps: f64, tokens: i32) {
    if model.is_empty() || tokens < MIN_PREFILL_TOKENS || !tps.is_finite() || tps <= 0.0 {
        return;
    }
    note(
        model,
        Record {
            prefill_tps: tps,
            gen_tps: 0.0,
        },
    );
}

/// Notes a completed generation pass. Ignored below [`MIN_GEN_TOKENS`], and
/// when the engine reports no rate at all (online providers do not measure
/// local throughput).
pub fn note_generation(model: &str, tps: f64, tokens: i32) {
    if model.is_empty() || tokens < MIN_GEN_TOKENS || !tps.is_finite() || tps <= 0.0 {
        return;
    }
    note(
        model,
        Record {
            prefill_tps: 0.0,
            gen_tps: tps,
        },
    );
}

/// Folds one sample into the session's peaks.
fn note(model: &str, sample: Record) {
    if let Ok(mut map) = SESSION.lock() {
        map.entry(model.to_string()).or_default().merge(sample);
    }
}

/// This session's peak for `model`, before any merge with the stored record.
#[must_use]
pub fn session_best(model: &str) -> Record {
    SESSION
        .lock()
        .ok()
        .and_then(|m| m.get(model).copied())
        .unwrap_or_default()
}

/// Reads the stored records. A missing or unparseable file reads as empty:
/// this is a convenience log, never something to fail a session over.
#[must_use]
pub fn load() -> BTreeMap<String, Record> {
    let Some(p) = path() else {
        return BTreeMap::new();
    };
    std::fs::read_to_string(p)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// What [`commit`] observed for one model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Committed {
    /// The all-time peaks after merging this session in.
    pub best: Record,
    /// True when this session set a new prefill peak.
    pub new_prefill: bool,
    /// True when this session set a new generation peak.
    pub new_gen: bool,
}

/// Merges this session's peaks into the stored records and writes them back,
/// reporting what `model` ended up with.
///
/// Returns `None` when this session recorded nothing for `model` *and* the
/// file holds nothing either — the caller has nothing worth printing. Write
/// failures are swallowed deliberately; a read-only `$HOME` should not turn
/// into an error at the very end of a session.
pub fn commit(model: &str) -> Option<Committed> {
    let session = session_best(model);
    let mut stored = load();
    let previous = stored.get(model).copied().unwrap_or_default();

    let new_prefill = session.prefill_tps > previous.prefill_tps;
    let new_gen = session.gen_tps > previous.gen_tps;

    if !session.is_empty() {
        // Merge every model this session touched, not just the one asked
        // about: a sidechain on a second engine earned its record too.
        if let Ok(map) = SESSION.lock() {
            for (name, rec) in map.iter() {
                stored.entry(name.clone()).or_default().merge(*rec);
            }
        }
        write(&stored);
    }

    let best = stored.get(model).copied().unwrap_or_default();
    if best.is_empty() {
        return None;
    }
    Some(Committed {
        best,
        new_prefill,
        new_gen,
    })
}

/// Writes `records`, creating `~/.plank` if needed. Best-effort.
fn write(records: &BTreeMap<String, Record>) {
    let Some(p) = path() else {
        return;
    };
    if let Some(dir) = p.parent()
        && std::fs::create_dir_all(dir).is_err()
    {
        return;
    }
    if let Ok(text) = serde_json::to_string_pretty(records) {
        let _ = std::fs::write(p, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that swap `$HOME`, which is process-global.
    static HOME: Mutex<()> = Mutex::new(());

    /// Every test uses its own model key: [`SESSION`] is process-global and
    /// the suite runs in parallel, so sharing a key makes tests clobber each
    /// other rather than test anything.
    #[test]
    fn peaks_keep_the_maximum_not_the_latest() {
        note_generation("max", 40.0, 100);
        note_generation("max", 12.0, 100);
        assert!((session_best("max").gen_tps - 40.0).abs() < 1e-9);
    }

    #[test]
    fn short_passes_are_ignored_as_unrepresentative() {
        // A 3-token prefill "at" 900 tok/s is setup cost, not throughput, and
        // a high-water mark would never recover from recording it.
        note_prefill("short", 900.0, 3);
        assert_eq!(session_best("short"), Record::default());
        note_prefill("short", 700.0, MIN_PREFILL_TOKENS);
        assert!((session_best("short").prefill_tps - 700.0).abs() < 1e-9);
    }

    #[test]
    fn nonsense_rates_are_refused() {
        note_generation("nonsense", f64::NAN, 100);
        note_generation("nonsense", f64::INFINITY, 100);
        note_generation("nonsense", 0.0, 100);
        note_generation("nonsense", -5.0, 100);
        assert_eq!(session_best("nonsense"), Record::default());
    }

    #[test]
    fn the_stub_engine_records_nothing() {
        // `EchoEngine::model_name` is empty; it has no throughput to speak of.
        note_generation("", 999.0, 500);
        assert_eq!(session_best(""), Record::default());
    }

    #[test]
    fn models_are_recorded_separately() {
        note_generation("sep-flash", 42.0, 100);
        note_generation("sep-other", 7.0, 100);
        assert!((session_best("sep-flash").gen_tps - 42.0).abs() < 1e-9);
        assert!((session_best("sep-other").gen_tps - 7.0).abs() < 1e-9);
    }

    #[test]
    fn prefill_and_generation_are_tracked_independently() {
        note_prefill("both", 700.0, 500);
        note_generation("both", 40.0, 100);
        let b = session_best("both");
        assert!((b.prefill_tps - 700.0).abs() < 1e-9);
        assert!((b.gen_tps - 40.0).abs() < 1e-9);
    }

    /// Runs `f` with `$HOME` pointed at a scratch directory.
    fn with_scoped_home(tag: &str, f: impl FnOnce()) {
        let _guard = HOME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tmp = std::env::temp_dir().join(format!("plank-speeds-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        // SAFETY: the HOME lock makes this the only thread touching the var.
        let prev = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", &tmp) };
        f();
        // SAFETY: as above.
        unsafe {
            match prev {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn commit_merges_into_the_file_and_flags_new_records() {
        with_scoped_home("commit", || {
            note_generation("commit", 40.0, 100);
            let first = commit("commit").expect("a record was set");
            assert!(first.new_gen, "first sample is always a new best");
            assert!((first.best.gen_tps - 40.0).abs() < 1e-9);

            // A slower session must not lower the stored peak, nor claim a
            // record. Clearing the session map imitates a fresh run.
            SESSION.lock().unwrap().remove("commit");
            note_generation("commit", 10.0, 100);
            let second = commit("commit").expect("the file still holds a record");
            assert!(!second.new_gen);
            assert!((second.best.gen_tps - 40.0).abs() < 1e-9);

            // A faster one does both.
            SESSION.lock().unwrap().remove("commit");
            note_generation("commit", 55.0, 100);
            let third = commit("commit").expect("a record was set");
            assert!(third.new_gen);
            assert!((third.best.gen_tps - 55.0).abs() < 1e-9);
        });
    }

    #[test]
    fn a_model_with_no_history_reports_nothing_to_print() {
        with_scoped_home("empty", || {
            assert_eq!(commit("never-seen-model"), None);
        });
    }
}

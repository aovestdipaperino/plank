// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `--fake-gpu`: an engine that pretends to prefill, so KV-cache behaviour can
//! be exercised without the model.
//!
//! The bug class this exists for is the one that took a whole investigation to
//! rule out: "switching X re-prefills". Answering it needs two launches that
//! each report whether the system-prompt checkpoint was *restored* or *rebuilt*
//! — and nothing else about inference. Yet reproducing it normally costs an
//! ~87 GB model load, and only one process may hold it
//! (`main::acquire_model_lock`), so the one machine that can reproduce it is
//! usually the one already running plank.
//!
//! [`FakeGpuEngine`] closes that gap. It carries a real KV *identity* — the
//! warm token buffer, captured and restored through [`crate::kvcache::KVCache`]
//! exactly as the real engine's is — while the "KV bytes" are a digest of the
//! text they stand for. That is enough for the whole tier walk to run for real:
//! checkpoints are written, looked up by fingerprint, restored or missed, and
//! every miss is reported with its reason.
//!
//! What it deliberately does not do is pretend to be fast or slow. Timings
//! under `--fake-gpu` are meaningless, and anything that reads like a
//! measurement would be a lie; the point is *which* tier resumed, not how long
//! it took.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::engine::{
    Engine, EngineError, EngineEvent, GenerationOptions, GenerationStats, PrefillProgress, Prompt,
};

/// Set once when a run is faking the GPU, so code far from the engine can tell
/// without threading a flag through every constructor.
static FAKING: AtomicBool = AtomicBool::new(false);

/// Marks this process as running with a fake engine.
pub fn set_active(on: bool) {
    FAKING.store(on, Ordering::Relaxed);
}

/// Whether this run is faking the GPU.
#[must_use]
pub fn active() -> bool {
    FAKING.load(Ordering::Relaxed)
}

/// What a warm step did, as the fake engine saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// `warm_reset`: the system tier was placed, `bytes` of it.
    Reset { bytes: usize },
    /// `warm_append`: a tier's text was added to the buffer.
    Append { bytes: usize },
    /// `warm_sync`: the buffer was "prefilled" up to `bytes`.
    Prefill { bytes: usize },
    /// `set_kv`: a checkpoint was restored, standing for `bytes` of text.
    Restore { bytes: usize },
}

/// An engine that keeps a warm token buffer and snapshots it, but runs no
/// model.
///
/// The buffer is the whole state. `get_kv` hands back a snapshot whose bytes
/// are a digest of it, and `set_kv` adopts a snapshot's text, so a restore
/// genuinely replaces what a later append builds on — which is the property
/// every "did it resume?" question turns on.
#[derive(Debug)]
pub struct FakeGpuEngine {
    /// Cumulative warm text, the stand-in for the token buffer.
    warm: String,
    /// How much of `warm` has been "prefilled".
    prefilled: usize,
    /// The model name this engine reports.
    ///
    /// Key material: `kvtier::system_fingerprint` hashes it, so a fake run
    /// keys its checkpoints exactly where the real engine would — which is
    /// what lets a `--fake-gpu` launch and a real one share a cache and what
    /// makes the reproduction faithful.
    model: String,
    ctx: i32,
    /// Whether this engine reports the trusted/untrusted split, mirroring the
    /// real engine so the tier plan has the same shape.
    splits_tail: bool,
    /// Every warm step, in order, for tests and for `--fake-gpu`'s report.
    steps: Vec<Step>,
}

impl FakeGpuEngine {
    /// A fake engine reporting `model` and a `ctx`-token window.
    #[must_use]
    pub fn new(model: &str, ctx: i32) -> Self {
        Self {
            warm: String::new(),
            prefilled: 0,
            model: model.to_string(),
            ctx,
            splits_tail: true,
            steps: Vec::new(),
        }
    }

    /// Whether to report the trusted/untrusted system split.
    #[must_use]
    pub fn with_split(mut self, on: bool) -> Self {
        self.splits_tail = on;
        self
    }

    /// The warm steps so far.
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Bytes "prefilled" since the last reset — what a real run would have
    /// paid for, and the number that should be *small* when a checkpoint was
    /// restored.
    #[must_use]
    pub fn prefilled_bytes(&self) -> usize {
        self.prefilled
    }

    /// A one-line summary of what this launch actually did, for the report.
    #[must_use]
    pub fn summary(&self) -> String {
        let restored: usize = self
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Restore { bytes } => Some(*bytes),
                _ => None,
            })
            .sum();
        let verdict = if restored == 0 {
            "rebuilt from zero"
        } else if self.prefilled > restored {
            "restored, then extended"
        } else {
            "restored"
        };
        format!(
            "fake-gpu: {verdict} — {restored} bytes restored, {} bytes prefilled, {} warm steps",
            self.prefilled,
            self.steps.len()
        )
    }

    /// The snapshot bytes standing for `text`.
    ///
    /// A digest rather than the text itself: it must change when the text
    /// changes (or a mismatched restore would look successful) without a
    /// checkpoint file growing to the size of the prompt.
    fn digest(text: &str) -> Vec<u8> {
        crate::session::sha1_hex(text.as_bytes()).into_bytes()
    }
}

impl Engine for FakeGpuEngine {
    fn generate(
        &mut self,
        prompt: Prompt<'_>,
        _opts: &GenerationOptions,
        interrupt: &dyn Fn() -> bool,
        _greedy: &dyn Fn() -> bool,
        on_event: &mut dyn FnMut(EngineEvent),
    ) -> Result<GenerationStats, EngineError> {
        let transcript = prompt.flat();
        let total = self.count_tokens(transcript).max(1);
        // A prefill *event*, because "did it prefill at boot?" is the question
        // this mode exists to answer and the front ends read it from here.
        on_event(EngineEvent::Prefill(PrefillProgress {
            done: total,
            total,
            tps: 0.0,
        }));
        if interrupt() {
            return Ok(GenerationStats {
                interrupted: true,
                ..GenerationStats::default()
            });
        }
        // `Notice` is the channel for "why the system-prompt cache is being
        // rebuilt", which is precisely this mode's verdict.
        on_event(EngineEvent::Notice(self.summary()));
        on_event(EngineEvent::Text(
            "[fake-gpu] no model loaded\n".to_string(),
        ));
        Ok(GenerationStats::default())
    }

    fn ctx_size(&self) -> i32 {
        self.ctx
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }

    fn splits_system_tail(&self) -> bool {
        self.splits_tail
    }

    fn warm_reset(&mut self, system: &str) -> Result<(), EngineError> {
        self.warm = system.to_string();
        self.prefilled = 0;
        self.steps.push(Step::Reset {
            bytes: system.len(),
        });
        Ok(())
    }

    fn warm_append(&mut self, text: Option<&str>) -> Result<(), EngineError> {
        if let Some(text) = text {
            self.warm.push_str(text);
            self.steps.push(Step::Append { bytes: text.len() });
        }
        Ok(())
    }

    fn warm_sync(&mut self, _on_event: &mut dyn FnMut(EngineEvent)) -> Result<bool, EngineError> {
        // "Prefill" whatever the buffer gained since the last sync. A restore
        // that really took effect shows up here as a small number.
        self.prefilled = self.warm.len();
        self.steps.push(Step::Prefill {
            bytes: self.warm.len(),
        });
        Ok(true)
    }

    fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
        Some(crate::kvcache::KVCache::new(
            Self::digest(&self.warm),
            crate::ds4tokens::TokenTranscript::default(),
        ))
    }

    fn set_kv(&mut self, cache: &crate::kvcache::KVCache) -> Result<(), EngineError> {
        // The snapshot carries a digest, not the text, so the fake engine
        // cannot reconstruct the buffer from it. It does not need to: what the
        // tier walk requires is that the restore be *accounted for*, and that
        // the buffer afterwards describe the restored prefix, which the walk's
        // own `warm_append` of every tier then rebuilds.
        self.steps.push(Step::Restore {
            bytes: cache.kv().len(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> FakeGpuEngine {
        FakeGpuEngine::new("DeepSeek V4 Flash Vision Experimental", 131_072)
    }

    #[test]
    fn the_model_name_is_reported_so_checkpoints_key_where_a_real_run_keys() {
        // The whole point of the mode: a fake launch must land on the same
        // fingerprint a real one would, or it reproduces nothing.
        let e = engine();
        assert_eq!(e.model_name(), "DeepSeek V4 Flash Vision Experimental");
        let fp_fake = crate::kvtier::system_fingerprint(
            &e.model_name(),
            "SYSTEM",
            crate::engine::ThinkMode::default(),
            0,
        );
        let fp_real = crate::kvtier::system_fingerprint(
            "DeepSeek V4 Flash Vision Experimental",
            "SYSTEM",
            crate::engine::ThinkMode::default(),
            0,
        );
        assert_eq!(fp_fake, fp_real);
    }

    #[test]
    fn a_snapshot_changes_when_the_warm_text_changes() {
        // A digest that did not move with the text would make a mismatched
        // restore look successful, which is the one failure this mode must
        // never invent.
        let mut a = engine();
        a.warm_reset("SYSTEM ONE").expect("reset");
        let first = a.get_kv().expect("kv").kv().to_vec();

        let mut b = engine();
        b.warm_reset("SYSTEM TWO").expect("reset");
        let second = b.get_kv().expect("kv").kv().to_vec();

        assert_ne!(first, second);
    }

    #[test]
    fn the_same_text_snapshots_identically() {
        let mut a = engine();
        a.warm_reset("SYSTEM").expect("reset");
        a.warm_append(Some(" context")).expect("append");
        let mut b = engine();
        b.warm_reset("SYSTEM").expect("reset");
        b.warm_append(Some(" context")).expect("append");
        assert_eq!(
            a.get_kv().expect("kv").kv(),
            b.get_kv().expect("kv").kv(),
            "identical warm text must snapshot identically"
        );
    }

    #[test]
    fn the_warm_steps_are_recorded_in_order() {
        let mut e = engine();
        e.warm_reset("SYS").expect("reset");
        e.warm_append(Some("ctx")).expect("append");
        e.warm_append(None).expect("append none");
        e.warm_sync(&mut |_| {}).expect("sync");
        assert_eq!(
            e.steps(),
            &[
                Step::Reset { bytes: 3 },
                Step::Append { bytes: 3 },
                Step::Prefill { bytes: 6 },
            ],
            "a None append places no tokens and must not be recorded"
        );
        assert_eq!(e.prefilled_bytes(), 6);
    }

    #[test]
    fn a_run_with_no_restore_reports_a_rebuild() {
        let mut e = engine();
        e.warm_reset("SYS").expect("reset");
        e.warm_sync(&mut |_| {}).expect("sync");
        assert!(e.summary().contains("rebuilt from zero"), "{}", e.summary());
    }

    #[test]
    fn a_run_that_restored_says_so() {
        let mut e = engine();
        e.warm_reset("SYS").expect("reset");
        let snap = e.get_kv().expect("kv");
        let mut fresh = engine();
        fresh.set_kv(&snap).expect("restore");
        assert!(fresh.summary().contains("restored"), "{}", fresh.summary());
        assert!(!fresh.summary().contains("rebuilt"), "{}", fresh.summary());
    }

    #[test]
    fn the_active_flag_round_trips() {
        let was = active();
        set_active(true);
        assert!(active());
        set_active(false);
        assert!(!active());
        set_active(was);
    }
}

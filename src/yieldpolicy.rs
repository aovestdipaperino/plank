// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Turns memory-pressure decisions into engine actions.
//!
//! Pure policy: it touches the engine only through the [`Engine`] trait, so it
//! is always compiled and tested without a model.
//!
//! The invariant that makes the whole design work is negative — **the yield
//! path never calls `get_kv`**. Capturing a snapshot would allocate twice the
//! session's size at the exact moment the system has none, which is
//! self-defeating. The KV is not preserved; it is rebuilt from the on-disk
//! tiers, and the worst case (Tier 2 plus a full transcript re-prefill) is
//! expensive but needs no memory at yield time.

use crate::engine::Engine;
use crate::kvladder::KvLadder;

/// What a resume will cost, computed at yield time.
///
/// Computed *before* the session is freed, because afterwards the information
/// is gone: the surviving restore point and the keep set both depend on state
/// the free destroys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorePlan {
    /// Rungs discarded getting here, for the disclosure line.
    pub rungs_dropped: usize,
    /// Tokens the resume must re-prefill.
    pub reprefill_tokens: i32,
    /// Blob stems the GC must not sweep while plank is yielded. A yielded
    /// session has no live rungs, so without this the sweep is free to delete
    /// the very blob the resume needs.
    pub keep: Vec<String>,
}

/// Applies pressure decisions to the engine.
#[derive(Debug, Default)]
pub struct YieldPolicy {
    plan: Option<RestorePlan>,
}

impl YieldPolicy {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops every ladder rung, returning how many went.
    ///
    /// The `Warn` response: rungs are pure cache, already on disk, and losing
    /// them costs only a deeper re-prefill later that may never be needed. No
    /// generation is interrupted.
    pub fn shed(&mut self, ladder: &mut KvLadder) -> usize {
        ladder.truncate_to(0).len()
    }

    /// Frees the live session and records what the resume will cost.
    ///
    /// `live_tokens` is the engine's current KV depth; `keep` is the set of
    /// blob stems the resume will restore through.
    pub fn yield_now(
        &mut self,
        engine: &mut dyn Engine,
        ladder: &KvLadder,
        live_tokens: i32,
        keep: Vec<String>,
    ) -> RestorePlan {
        // Deepest surviving rung, if any, is the restore point; otherwise the
        // floor is a tier blob and the whole live prefix is rebuilt.
        let floor = ladder.rungs().last().map_or(0, |r| r.tokens);
        let plan = RestorePlan {
            rungs_dropped: ladder.rungs().len(),
            reprefill_tokens: (live_tokens - floor).max(0),
            keep,
        };
        // Order matters: the plan is computed from state the free destroys.
        engine.release_session();
        self.plan = Some(plan.clone());
        plan
    }

    /// The pinned plan, while yielded.
    #[must_use]
    pub fn plan(&self) -> Option<&RestorePlan> {
        self.plan.as_ref()
    }

    /// Takes the plan, retiring it. Called on resume.
    pub fn clear_plan(&mut self) -> Option<RestorePlan> {
        self.plan.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::kvladder::KvLadder;

    /// Records what the policy asked the engine to do.
    #[derive(Default)]
    struct YieldSpy {
        released: usize,
        captured: usize,
        restored: usize,
    }

    impl std::fmt::Debug for YieldSpy {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("YieldSpy")
        }
    }

    impl Engine for YieldSpy {
        fn generate(
            &mut self,
            _p: crate::engine::Prompt<'_>,
            _o: &crate::engine::GenerationOptions,
            _i: &dyn Fn() -> bool,
            _g: &dyn Fn() -> bool,
            _e: &mut dyn FnMut(crate::engine::EngineEvent),
        ) -> Result<crate::engine::GenerationStats, crate::engine::EngineError> {
            unreachable!("the yield path never generates")
        }
        fn ctx_size(&self) -> i32 {
            4096
        }
        fn get_kv(&mut self) -> Option<crate::kvcache::KVCache> {
            self.captured += 1;
            None
        }
        fn set_kv(
            &mut self,
            _c: &crate::kvcache::KVCache,
        ) -> Result<(), crate::engine::EngineError> {
            self.restored += 1;
            Ok(())
        }
        fn release_session(&mut self) {
            self.released += 1;
        }
    }

    fn ladder_with(spans: &[(usize, i32)]) -> KvLadder {
        let mut l = KvLadder::new();
        for (s, t) in spans {
            let _ = l.push(*s, *t);
        }
        l
    }

    #[test]
    fn yielding_never_captures_a_snapshot() {
        let mut spy = YieldSpy::default();
        let mut p = YieldPolicy::new();
        let ladder = ladder_with(&[(2, 8192)]);
        p.yield_now(&mut spy, &ladder, 20_000, vec!["tier2".to_owned()]);
        assert_eq!(
            spy.captured, 0,
            "get_kv allocates twice the session size; calling it here is the \
             one thing this design exists to avoid"
        );
        assert_eq!(spy.released, 1, "the session must actually be freed");
    }

    #[test]
    fn shedding_drops_every_rung() {
        let mut p = YieldPolicy::new();
        let mut ladder = ladder_with(&[(2, 8192), (5, 16_384)]);
        assert_eq!(p.shed(&mut ladder), 2);
        assert!(
            ladder.rungs().is_empty(),
            "rungs are pure cache and are the cheapest thing to give back"
        );
        assert_eq!(p.shed(&mut ladder), 0, "shedding twice drops nothing more");
    }

    #[test]
    fn the_restore_plan_records_the_reprefill_cost() {
        let mut spy = YieldSpy::default();
        let mut p = YieldPolicy::new();
        // No rungs: the restore floor is a tier blob, so the whole live
        // transcript must be re-prefilled.
        let ladder = KvLadder::new();
        let plan = p.yield_now(&mut spy, &ladder, 20_000, vec!["tier2".to_owned()]);
        assert_eq!(
            plan.reprefill_tokens, 20_000,
            "with no rung the entire live prefix is rebuilt"
        );
    }

    #[test]
    fn a_surviving_rung_bounds_the_reprefill() {
        let mut spy = YieldSpy::default();
        let mut p = YieldPolicy::new();
        let ladder = ladder_with(&[(2, 8192)]);
        let plan = p.yield_now(&mut spy, &ladder, 20_000, vec!["tier2".to_owned()]);
        assert_eq!(
            plan.reprefill_tokens,
            20_000 - 8192,
            "the deepest rung is the restore point, so only the suffix is rebuilt"
        );
    }

    #[test]
    fn the_plan_is_pinned_until_taken() {
        let mut spy = YieldSpy::default();
        let mut p = YieldPolicy::new();
        assert!(p.plan().is_none());
        p.yield_now(&mut spy, &KvLadder::new(), 100, vec!["tier2".to_owned()]);
        assert_eq!(
            p.plan().map(|pl| pl.keep.as_slice()),
            Some(["tier2".to_owned()].as_slice()),
            "the GC keep set has to survive the yield or there is nothing to \
             come back through"
        );
        assert!(p.clear_plan().is_some());
        assert!(p.plan().is_none(), "resuming retires the plan");
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Depth-indexed KV snapshot ladder.

/// Most rungs held at once. Each blob is 200-400 MB on disk, so this is a
/// disk-budget decision, not a latency one: capture costs ~0.4s and restore
/// ~0.1s, against the minutes of prefill a rebuild costs.
pub const LADDER_MAX_RUNGS: usize = 3;

/// A new anchor must cover at least this many tokens more than the best rung
/// that already predates the same edit point, or it is not worth capturing.
///
/// Anchors are taken at the transcript index a large tool result is about to
/// occupy, so the *first* one for a given index is free of competition and
/// always taken. This constant governs only the follow-ups: as
/// `microcompact_first_index` marches forward, a deeper anchor covers more
/// tokens and so leaves less to re-prefill, but each capture is a certain cost
/// (~0.3s and a few hundred MB written) against a probabilistic gain. At the
/// measured ~100 tok/s prefill, 8192 tokens is ~80s of prefill saved *if* the
/// pass fires — three orders of magnitude above the capture cost — while
/// halving the capture rate the old blind turn-end spacing (4096) produced.
pub const LADDER_ANCHOR_MIN_SPACING_TOKENS: i32 = 8192;

/// One stored snapshot, identified by how much of the transcript it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rung {
    /// Slot number, used to name the blob on disk.
    pub index: usize,
    /// Transcript spans covered. A rung is usable for an edit at span `k`
    /// only when `spans <= k`.
    pub spans: usize,
    /// Tokens covered, used to compare against what the engine would reuse.
    pub tokens: i32,
}

/// Depth-indexed ladder of KV snapshots, ordered shallowest-first.
#[derive(Debug, Default)]
pub struct KvLadder {
    rungs: Vec<Rung>,
    next_index: usize,
}

impl KvLadder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Rungs, shallowest first.
    #[must_use]
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }

    /// Whether an anchor for an edit at transcript index `spans`, covering
    /// `tokens`, is worth capturing.
    ///
    /// Two rules, in this order:
    /// 1. **Redundancy.** If the ladder already holds a rung shallow enough to
    ///    cover an edit at `spans` (exactly [`Self::select`]'s test, so the
    ///    answer matches what a restore would actually find), a new capture
    ///    buys nothing but depth.
    /// 2. **Spacing.** Extra depth is only bought when it is worth at least
    ///    [`LADDER_ANCHOR_MIN_SPACING_TOKENS`], so anchors cannot be taken back
    ///    to back across a run of tool calls in one turn.
    #[must_use]
    pub fn wants_anchor(&self, spans: usize, tokens: i32) -> bool {
        self.select(spans, 0)
            .is_none_or(|best| tokens - best.tokens >= LADDER_ANCHOR_MIN_SPACING_TOKENS)
    }

    /// Records a rung at `spans`/`tokens`, evicting one when full. Returns the
    /// new rung and whatever it displaced, so the caller can write the new
    /// blob and delete the old one.
    pub fn push(&mut self, spans: usize, tokens: i32) -> (Rung, Option<Rung>) {
        let rung = Rung {
            index: self.next_index,
            spans,
            tokens,
        };
        self.next_index += 1;
        self.rungs.push(rung);
        let evicted = (self.rungs.len() > LADDER_MAX_RUNGS).then(|| self.evict());
        (rung, evicted)
    }

    /// Drops the rung whose removal least widens the largest gap, never the
    /// shallowest (it is the only one that can cover an edit near the start of
    /// the transcript) and never the newest.
    fn evict(&mut self) -> Rung {
        let mut best = 1;
        let mut best_gap = i32::MAX;
        for i in 1..self.rungs.len() - 1 {
            let gap = self.rungs[i + 1].tokens - self.rungs[i - 1].tokens;
            if gap < best_gap {
                best_gap = gap;
                best = i;
            }
        }
        self.rungs.remove(best)
    }

    /// The deepest rung that predates an edit at span `max_spans` and covers
    /// more tokens than the engine would reuse unaided.
    #[must_use]
    pub fn select(&self, max_spans: usize, already_reused: i32) -> Option<&Rung> {
        self.rungs
            .iter()
            .rev()
            .find(|r| r.spans <= max_spans && r.tokens > already_reused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_anchor_is_only_wanted_when_it_is_uncovered_or_materially_deeper() {
        let mut l = KvLadder::new();
        // An empty ladder always wants its first anchor.
        assert!(l.wants_anchor(5, 1000));
        l.push(5, 14_000);
        // An edit at span 5 or later is already covered by that rung, so a new
        // anchor must buy real depth.
        assert!(!l.wants_anchor(9, 14_000 + LADDER_ANCHOR_MIN_SPACING_TOKENS - 1));
        assert!(l.wants_anchor(9, 14_000 + LADDER_ANCHOR_MIN_SPACING_TOKENS));
        // An edit *before* the rung is not covered by it at all: the spacing
        // rule must not suppress the only anchor that could ever help.
        assert!(l.wants_anchor(4, 100));
    }

    #[test]
    fn eviction_keeps_the_shallowest_rung_and_the_widest_spread() {
        let mut l = KvLadder::new();
        l.push(2, 5_000);
        l.push(6, 20_000);
        l.push(9, 25_000);
        assert_eq!(l.rungs().len(), LADDER_MAX_RUNGS);
        // Pushing a fourth evicts the interior rung whose removal creates the
        // smallest gap: dropping 25_000 leaves gaps of 15_000/10_000, while
        // dropping 20_000 would leave a 20_000 gap. Neither the shallowest nor
        // the newest is ever a candidate.
        let (_, evicted) = l.push(12, 30_000);
        assert_eq!(evicted.map(|r| r.tokens), Some(25_000));
        let kept: Vec<i32> = l.rungs().iter().map(|r| r.tokens).collect();
        assert_eq!(kept, vec![5_000, 20_000, 30_000]);
        // The shallowest rung survives: it is the only one that can cover an
        // edit near the start of the transcript.
        assert_eq!(l.rungs()[0].spans, 2);
    }

    #[test]
    fn select_takes_the_deepest_rung_at_or_below_the_edit() {
        let mut l = KvLadder::new();
        l.push(2, 5_000);
        l.push(6, 14_000);
        l.push(10, 25_000);
        // Edit at span 9: the rung covering 10 spans contains the edited span,
        // so the deepest usable one covers 6.
        assert_eq!(l.select(9, 0).map(|r| r.tokens), Some(14_000));
        // Edit at span 1: nothing is shallow enough.
        assert_eq!(l.select(1, 0), None);
    }

    #[test]
    fn select_refuses_a_rung_that_beats_nothing() {
        let mut l = KvLadder::new();
        l.push(2, 5_000);
        // The engine would already reuse 6_000 tokens unaided; restoring a
        // 5_000-token rung would make the turn worse, not better.
        assert_eq!(l.select(9, 6_000), None);
        assert_eq!(l.select(9, 4_000).map(|r| r.tokens), Some(5_000));
    }

    #[test]
    fn select_returns_rung_when_spans_exactly_equal_max_spans() {
        let mut l = KvLadder::new();
        l.push(5, 10_000);
        // Query at exactly the rung's spans: should return the rung (5 <= 5).
        assert_eq!(l.select(5, 0).map(|r| r.tokens), Some(10_000));
        // Query just below: should return None (no rung at or below 4).
        assert_eq!(l.select(4, 0), None);
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Loop guards: watch the agent loop for unproductive patterns.
//!
//! The classic local-model failure is a loop: the model reads the same file,
//! or runs the same failing `cargo test`, four times. On a hosted model that
//! costs money; on a local Metal engine it costs the user's afternoon. This
//! module detects repeated identical tool calls, and repeated *sequences* of
//! calls — a model that cycles through the same eight reads never repeats one
//! call back to back, but the eight-call period is visible in the window —
//! and nudges the model.
//!
//! The guard escalates: advisory, then a hard block of the call, then — when
//! the model answers three blocked stanzas in a row with the identical stanza
//! — [`LoopGuard::tripped`] tells the turn loop to end the turn. That last
//! rung exists because a block alone changes nothing the model sees: at
//! temperature 0 the same prompt yields the same pass forever, and a 19-minute
//! repro of exactly that is what added it. A legitimate poll of an
//! async bash job (`bash_status` with identical args) looks identical to a
//! stuck loop, so the polling path is exempted explicitly by the caller.

use std::collections::{HashMap, VecDeque};

/// Identical-call threshold: the Nth identical call gets the advisory.
const REPEAT_THRESHOLD: u32 = 3;

/// Hard block threshold: once a call's identical-repeat count exceeds this,
/// the call is refused outright instead of merely advised. Gives the model
/// `REPEAT_THRESHOLD..=BLOCK_THRESHOLD` advisory chances to self-correct
/// before the circuit breaker trips.
const BLOCK_THRESHOLD: u32 = 5;

/// Consecutive stanzas in which *every* call was refused before the turn is
/// ended. Each one is a full pass in which the model saw the refusal and
/// re-emitted the identical calls; three is deterministic-loop territory.
const STANZA_TRIP: u32 = 3;

/// How many recent calls the guard remembers before aging out. Wide enough
/// that a multi-call cycle (the two-hour repro had a period of eight) shows
/// up at least twice in full.
const MAX_WINDOW: usize = 32;

/// Shortest cycle the sequence check looks for; a period of one is the
/// identical-call check.
const MIN_PERIOD: usize = 2;

/// A tool call signature: the tool name plus a digest of its normalised args.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallSig(String, String);

/// What [`LoopGuard::observe`] decided to do about a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Nudge {
    /// No advisory; the call is not a repeat.
    None,
    /// An advisory line to append to the tool result the model receives.
    Advisory(String),
    /// A hard block: the call must be refused and must not be dispatched.
    /// The caller returns a tool error with this text instead of running
    /// the call. This is the circuit breaker an advisory-only guard cannot
    /// provide — a stuck model that reads the advisory, says "let's stop,"
    /// and re-emits the identical call is physically prevented from
    /// running it.
    Block(String),
}

/// Detects repeated identical tool calls within a bounded window.
#[derive(Debug, Clone)]
pub struct LoopGuard {
    /// Dispatched calls, oldest first. Refused calls never enter it: they
    /// did not run, so they must not push real history out of the window,
    /// and counting them here made the reported count plateau (one aged out
    /// for each one added) at a number that then never changed.
    window: VecDeque<CallSig>,
    repeats: HashMap<CallSig, u32>,
    /// Calls refused per signature; never ages, so the count the model is
    /// shown keeps rising and every refusal reads differently.
    refused: HashMap<CallSig, u32>,
    /// Consecutive stanzas in which every call was refused (see
    /// [`Self::note_stanza`]).
    blocked_stanzas: u32,
}

impl LoopGuard {
    /// A fresh guard with an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: VecDeque::new(),
            repeats: HashMap::new(),
            refused: HashMap::new(),
            blocked_stanzas: 0,
        }
    }

    /// Observes one tool call. Returns an advisory on the Nth identical call
    /// (N = [`REPEAT_THRESHOLD`]), or when the most recent calls repeat the
    /// sequence of calls immediately before them; older calls age out of the
    /// window.
    pub fn observe(&mut self, tool: &str, args_digest: String) -> Nudge {
        let sig = CallSig(tool.to_string(), args_digest);
        // Already at the block threshold: refuse without touching the window.
        if let Some(&dispatched) = self.repeats.get(&sig)
            && dispatched >= BLOCK_THRESHOLD
        {
            let refused = self.refused.entry(sig).or_insert(0);
            *refused += 1;
            let count = dispatched + *refused;
            return Nudge::Block(format!(
                "you have called this tool with these identical arguments {count} times. This call is refused. Do something different: act on what you already know, or tell the user what is missing"
            ));
        }
        // Age out the oldest call so a repeat long ago does not count forever.
        if self.window.len() >= MAX_WINDOW
            && let Some(oldest) = self.window.pop_front()
            && let Some(c) = self.repeats.get_mut(&oldest)
        {
            *c = c.saturating_sub(1);
            if *c == 0 {
                self.repeats.remove(&oldest);
            }
        }
        self.window.push_back(sig.clone());
        let count = self.repeats.entry(sig).or_insert(0);
        *count += 1;
        if *count >= REPEAT_THRESHOLD {
            return Nudge::Advisory(format!(
                "you have called this tool with these identical arguments {count} times. Do not call it again: act on what you already know, or tell the user what is missing"
            ));
        }
        match self.repeated_period() {
            Some(period) => Nudge::Advisory(format!(
                "your last {period} tool calls repeat the {period} calls before them, with the same results. You are in a loop. Do not run them again: decide now and act — emit the edit you already planned, or answer the user"
            )),
            None => Nudge::None,
        }
    }

    /// Records whether the stanza just observed was refused in full. Call once
    /// per stanza, after observing its calls; a stanza with even one call
    /// that ran resets the run, so a model that is making *some* progress is
    /// never cut off.
    pub fn note_stanza(&mut self, all_blocked: bool) {
        self.blocked_stanzas = if all_blocked {
            self.blocked_stanzas + 1
        } else {
            0
        };
    }

    /// Whether the turn should end: the last [`STANZA_TRIP`] stanzas were
    /// each refused in full. A block feeds the model a prompt that differs
    /// from the last one by a single digit, and a deterministic model answers
    /// it identically, so past this point nothing but ending the turn changes
    /// the outcome.
    #[must_use]
    pub fn tripped(&self) -> bool {
        self.blocked_stanzas >= STANZA_TRIP
    }

    /// The period of the cycle the window ends in, if its last `2 * period`
    /// calls are two identical runs: the shortest such period, or `None`. Only
    /// full repeats count, so a cycle is reported once per repetition, on its
    /// last call.
    fn repeated_period(&self) -> Option<usize> {
        let n = self.window.len();
        (MIN_PERIOD..=n / 2).find(|&period| {
            let tail = n - period;
            (0..period).all(|k| self.window[tail + k] == self.window[tail - period + k])
        })
    }
}

impl Default for LoopGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Nudge {
    /// The advisory text, if any.
    #[must_use]
    pub fn as_advisory(&self) -> Option<&str> {
        match self {
            Nudge::None | Nudge::Block(_) => None,
            Nudge::Advisory(s) => Some(s),
        }
    }

    /// The block text, if this nudge is a hard block.
    #[must_use]
    pub fn as_block(&self) -> Option<&str> {
        match self {
            Nudge::None | Nudge::Advisory(_) => None,
            Nudge::Block(s) => Some(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(s: &str) -> String {
        crate::session::sha1_hex(s.as_bytes())
    }

    #[test]
    fn the_refused_count_keeps_rising_and_refusals_stay_out_of_the_window() {
        let mut g = LoopGuard::new();
        for _ in 0..BLOCK_THRESHOLD {
            let _ = g.observe("read", digest("a"));
        }
        let n = |nudge: Nudge| -> u32 {
            let text = nudge.as_block().expect("blocked").to_owned();
            text.split("arguments ")
                .nth(1)
                .and_then(|t| t.split(' ').next())
                .and_then(|d| d.parse().ok())
                .expect("count")
        };
        assert_eq!(n(g.observe("read", digest("a"))), 6);
        // Far past the window size: the count must still climb by one per call.
        for _ in 0..(MAX_WINDOW * 2) {
            let _ = g.observe("read", digest("a"));
        }
        assert_eq!(
            n(g.observe("read", digest("a"))),
            BLOCK_THRESHOLD + 1 + u32::try_from(MAX_WINDOW * 2).unwrap() + 1
        );
        // Refused calls never entered the window, so it holds just the dispatched ones.
        assert_eq!(g.window.len(), BLOCK_THRESHOLD as usize);
    }

    #[test]
    fn trips_after_three_fully_refused_stanzas_and_resets_on_progress() {
        let mut g = LoopGuard::new();
        assert!(!g.tripped());
        g.note_stanza(true);
        g.note_stanza(true);
        assert!(!g.tripped());
        g.note_stanza(false);
        g.note_stanza(true);
        g.note_stanza(true);
        assert!(!g.tripped(), "progress in between resets the run");
        g.note_stanza(true);
        assert!(g.tripped());
    }

    #[test]
    fn an_advisory_fires_on_the_third_identical_call_and_not_the_second() {
        let mut g = LoopGuard::new();
        let d = digest("file.txt");
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        let nudge = g.observe("read", d.clone());
        assert!(matches!(nudge, Nudge::Advisory(_)));
        assert!(nudge.as_advisory().unwrap().contains("3 times"));
    }

    #[test]
    fn a_different_tool_or_args_is_a_separate_signature() {
        let mut g = LoopGuard::new();
        let d = digest("a");
        let d2 = digest("b");
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        // A different digest is its own signature; it does not advance d's count.
        assert_eq!(g.observe("read", d2.clone()), Nudge::None);
        // d's 3rd call (one d2 interleaved) -> advisory.
        let nudge = g.observe("read", d.clone());
        assert!(matches!(nudge, Nudge::Advisory(_)));
    }

    #[test]
    fn a_repeated_sequence_is_reported_even_when_no_single_call_repeats_thrice() {
        // The two-hour repro: an eight-call cycle in a window of ten never
        // had one signature three times. Two full turns of the cycle are enough.
        let mut g = LoopGuard::new();
        let cycle: Vec<String> = (0..8).map(|i| digest(&format!("f{i}"))).collect();
        for d in &cycle {
            assert_eq!(g.observe("read", d.clone()), Nudge::None);
        }
        for d in &cycle[..7] {
            assert_eq!(
                g.observe("read", d.clone()),
                Nudge::None,
                "cycle not yet closed"
            );
        }
        let nudge = g.observe("read", cycle[7].clone());
        let text = nudge
            .as_advisory()
            .expect("the closed cycle is an advisory");
        assert!(text.contains("last 8 tool calls repeat"), "{text}");
    }

    #[test]
    fn a_partial_overlap_is_not_a_cycle() {
        let mut g = LoopGuard::new();
        for name in ["a", "b", "c", "a", "b", "d"] {
            assert_eq!(g.observe("read", digest(name)), Nudge::None);
        }
    }

    #[test]
    fn old_calls_age_out_of_the_window() {
        let mut g = LoopGuard::new();
        let d = digest("x");
        // Fill the window with distinct calls, then one repeat of an old sig.
        for i in 0..MAX_WINDOW {
            let _ = g.observe("read", digest(&format!("f{i}")));
        }
        // The first sig aged out, so a single repeat is not yet an advisory.
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
    }

    #[test]
    fn a_block_fires_after_advisory_chances_are_exhausted() {
        let mut g = LoopGuard::new();
        let d = digest("file.txt");
        // Counts 1-2: no nudge.
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        assert_eq!(g.observe("read", d.clone()), Nudge::None);
        // Counts 3-5: advisory (REPEAT_THRESHOLD..=BLOCK_THRESHOLD).
        for expected_count in 3..=BLOCK_THRESHOLD {
            let nudge = g.observe("read", d.clone());
            let text = nudge.as_advisory().expect("advisory before block");
            assert!(text.contains(&format!("{expected_count} times")), "{text}");
        }
        // Count 6: block (count > BLOCK_THRESHOLD).
        let nudge = g.observe("read", d.clone());
        let text = nudge.as_block().expect("block after threshold");
        assert!(text.contains("refused"), "{text}");
    }
}

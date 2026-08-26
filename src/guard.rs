// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Loop guards: watch the agent loop for unproductive patterns.
//!
//! The classic local-model failure is a loop: the model reads the same file,
//! or runs the same failing `cargo test`, four times. On a hosted model that
//! costs money; on a local Metal engine it costs the user's afternoon. This
//! module detects repeated identical tool calls and nudges the model.
//!
//! The guard is **advisory only** — it never blocks. A legitimate poll of an
//! async bash job (`bash_status` with identical args) looks identical to a
//! stuck loop, so the polling path is exempted explicitly by the caller.

use std::collections::{HashMap, VecDeque};

/// Identical-call threshold: the Nth identical call gets the advisory.
const REPEAT_THRESHOLD: u32 = 3;

/// How many recent calls the guard remembers before aging out.
const MAX_WINDOW: usize = 10;

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
}

/// Detects repeated identical tool calls within a bounded window.
#[derive(Debug, Clone)]
pub struct LoopGuard {
    window: VecDeque<CallSig>,
    repeats: HashMap<CallSig, u32>,
}

impl LoopGuard {
    /// A fresh guard with an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: VecDeque::new(),
            repeats: HashMap::new(),
        }
    }

    /// Observes one tool call. Returns an advisory on the Nth identical call
    /// (N = [`REPEAT_THRESHOLD`]); older calls age out of the window.
    pub fn observe(&mut self, tool: &str, args_digest: String) -> Nudge {
        let sig = CallSig(tool.to_string(), args_digest);
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
            Nudge::Advisory(format!(
                "you have called this tool with these arguments {count} times; the result has not changed"
            ))
        } else {
            Nudge::None
        }
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
            Nudge::None => None,
            Nudge::Advisory(s) => Some(s),
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
}

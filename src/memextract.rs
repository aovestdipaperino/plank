// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The background memory extraction pass.
//!
//! Fires at the end of the query loop — a generation that produced a final
//! response with no tool calls — subject to three gates, in order: mutual
//! exclusion with the model's own `remember` call, a throttle over eligible
//! turns, and depth keying so the pass reads only the transcript it has not
//! already seen.
//!
//! The depth keying is the same shape as a `kvladder` rung, and carries the
//! same warning: the recorded depth is what makes the resume correct, and
//! getting it wrong degrades *silently* into reprocessing the whole
//! transcript on every pass, at exactly the moment the user is idle.
//!
//! The pass runs through the sub-agent fork, so `in_sidechain()` holds: no
//! rungs pushed, no payload stored, no checkpoint debris.

use crate::memory::{Entry, MetaStore, Scope};
use crate::session::{Message, Role};

/// Gating state for the pass, owned by the `Agent`.
#[derive(Debug, Clone, Default)]
pub struct ExtractState {
    /// Mirror of `settings.memory.auto_extract`, sampled per turn.
    pub enabled: bool,
    /// Mirror of `settings.memory.extract_every_n_turns`.
    pub every_n: u32,
    /// Transcript depth the last completed pass covered.
    processed_depth: usize,
    /// Set while a pass is in flight.
    running: bool,
    /// Depth requested while a pass was running; at most one, because a
    /// trailing run covers everything up to the newest depth anyway.
    stashed: Option<usize>,
    /// Eligible turns seen since the last run, for the throttle.
    eligible: u32,
    /// Whether the model called `remember` or `forget` this turn.
    wrote_this_turn: bool,
}

impl ExtractState {
    /// Records that the model wrote memory itself this turn, suppressing the
    /// passive pass exactly once.
    pub fn note_tool_write(&mut self) {
        self.wrote_this_turn = true;
    }

    /// Decides whether to run, given the current transcript depth. Returns
    /// the depth to start reading from, or `None` to skip.
    ///
    /// A call while a pass is already running stashes the new depth and
    /// returns `None`; the stash is a flag, not a queue — a second trigger
    /// while still running just overwrites it. Once `finish` clears
    /// `running`, the next `should_run` call resumes from
    /// `processed_depth` (which `finish` has just advanced), covering
    /// everything that arrived while the pass was busy in one trailing run.
    pub fn should_run(&mut self, depth: usize) -> Option<usize> {
        let suppressed = std::mem::take(&mut self.wrote_this_turn);
        if !self.enabled || suppressed {
            return None;
        }
        if depth <= self.processed_depth {
            return None;
        }
        if self.running {
            self.stashed = Some(depth);
            return None;
        }
        self.eligible = self.eligible.saturating_add(1);
        if self.eligible < self.every_n.max(1) {
            return None;
        }
        self.eligible = 0;
        self.running = true;
        Some(self.processed_depth)
    }

    /// Records a completed pass covering up to `depth`. `depth` is expected
    /// to be at least the depth most recently handed out by `should_run`;
    /// a caller that passes something lower would move `processed_depth`
    /// backwards and cause the next pass to reprocess transcript it has
    /// already seen — callers must always pass the depth observed at the
    /// time the pass was started, never an earlier one.
    pub fn finish(&mut self, depth: usize) {
        self.processed_depth = depth;
        self.running = false;
        // A trailing run is enabled by clearing `running`; the stash is only
        // a record that more arrived, and the next `should_run` recomputes
        // the span from `processed_depth` anyway.
        self.stashed = None;
    }

    /// Abandons an in-flight pass without recording progress, so the work is
    /// simply redone later. Nothing is half-applied: verdicts are applied
    /// only after the pass returns.
    pub fn cancel(&mut self) {
        self.running = false;
    }
}

/// Renders a message's role for the prompt. `Role` is not `Display`, so this
/// mirrors the tag words used elsewhere in the transcript.
fn role_word(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Builds the pass's prompt: the transcript slice, the current entries with
/// their ids, and the verdict contract.
#[must_use]
pub fn build_prompt(slice: &[Message], entries: &[(Scope, Entry)]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(
        "You are the memory extraction pass. Read the conversation excerpt below and \
         update persistent memory.\n\n\
         Save only facts that cannot be re-derived by reading the repository: who the \
         user is, corrections they gave about how to work, project goals and constraints \
         not in the code or git history, and pointers to external systems. Never save \
         code patterns, architecture, file paths, git history, debugging recipes, or \
         anything already stated in AGENTS.md. Convert relative dates to absolute ones.\n\n\
         Reply with a JSON array and nothing else. Each element is one of:\n\
         {\"verdict\": \"ADD\", \"text\": ..., \"type\": user|feedback|project|reference, \
         \"scope\": user|project}\n\
         {\"verdict\": \"UPDATE\", \"id\": ..., \"text\": ...}  (supersede an existing entry)\n\
         {\"verdict\": \"DELETE\", \"id\": ...}  (it is redundant or wrong)\n\
         {\"verdict\": \"USED\", \"id\": ...}  (this entry bore on the work in the excerpt)\n\
         An empty array is a valid and common answer.\n\n\
         Existing entries:\n",
    );
    if entries.is_empty() {
        out.push_str("(none)\n");
    }
    for (scope, e) in entries {
        let _ = writeln!(
            out,
            "{} [{}] {{{}}} {}",
            scope.marker_name(),
            e.kind.tag(),
            e.id(),
            e.text
        );
    }
    out.push_str("\nConversation excerpt:\n");
    for m in slice {
        let _ = writeln!(out, "{}: {}", role_word(m.role), m.text);
    }
    out
}

/// Reads every non-retracted entry from both scopes, for the prompt. A
/// retracted entry is one the model already decided was wrong via `forget`,
/// so the pass must not be shown it as still live.
#[must_use]
pub fn current_entries(cwd: &std::path::Path) -> Vec<(Scope, Entry)> {
    let mut out = Vec::new();
    for scope in [Scope::User, Scope::Project] {
        let Some(path) = crate::memory::path_for(scope, cwd) else {
            continue;
        };
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        let meta = MetaStore::load(&crate::memory::meta_path_for(&path));
        for e in crate::memory::parse_entries(&body) {
            if !meta.get(&e.id()).retracted {
                out.push((scope, e));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> ExtractState {
        ExtractState {
            enabled: true,
            every_n: 1,
            ..ExtractState::default()
        }
    }

    #[test]
    fn a_second_pass_over_an_unchanged_transcript_processes_nothing() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0), "first pass covers depth 0..10");
        s.finish(10);
        assert_eq!(s.should_run(10), None, "nothing new — no pass");
        assert_eq!(
            s.should_run(14),
            Some(10),
            "resumes from the recorded depth"
        );
    }

    #[test]
    fn a_remember_call_in_the_turn_suppresses_the_pass() {
        let mut s = state();
        s.note_tool_write();
        assert_eq!(s.should_run(10), None, "the model's own judgment wins");
        assert_eq!(
            s.should_run(10),
            Some(0),
            "the suppression lasts one turn only"
        );
    }

    #[test]
    fn the_throttle_counts_only_eligible_turns() {
        let mut s = ExtractState {
            enabled: true,
            every_n: 3,
            ..ExtractState::default()
        };
        assert_eq!(s.should_run(4), None);
        assert_eq!(s.should_run(6), None);
        assert_eq!(s.should_run(8), Some(0), "every third eligible turn");
    }

    #[test]
    fn a_trigger_while_running_stashes_and_yields_one_trailing_run() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        assert_eq!(s.should_run(12), None, "a pass is already running");
        assert_eq!(
            s.should_run(14),
            None,
            "still running; the stash is overwritten, not queued"
        );
        s.finish(10);
        assert_eq!(
            s.should_run(14),
            Some(10),
            "one trailing run picks up the stash"
        );
    }

    #[test]
    fn cancelling_leaves_the_recorded_depth_untouched() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        s.cancel();
        assert_eq!(s.should_run(10), Some(0), "the work is simply redone later");
    }

    #[test]
    fn disabled_never_runs() {
        let mut s = ExtractState {
            enabled: false,
            every_n: 1,
            ..ExtractState::default()
        };
        assert_eq!(s.should_run(100), None);
    }
}

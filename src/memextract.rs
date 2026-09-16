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
    /// Eligible turns seen since the last run, for the throttle.
    eligible: u32,
    /// Whether the model called `remember` or `forget` this turn.
    wrote_this_turn: bool,
    /// Passes whose reply yielded no verdicts; the front end announces
    /// only the first.
    unusable_replies: u32,
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
    /// A call while a pass is already running is simply dropped — nothing is
    /// recorded about it. That is not a lost trigger: once `finish` clears
    /// `running` and advances `processed_depth`, the next `should_run` call
    /// re-derives the span from `processed_depth` against whatever depth it
    /// is given then, which necessarily covers everything that arrived while
    /// the pass was busy, in one trailing run.
    pub fn should_run(&mut self, depth: usize) -> Option<usize> {
        let suppressed = std::mem::take(&mut self.wrote_this_turn);
        if !self.enabled || suppressed {
            return None;
        }
        if depth <= self.processed_depth {
            return None;
        }
        if self.running {
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

    /// Records a completed pass covering up to `depth`. Callers should
    /// always pass the depth observed at the time the pass was started,
    /// never an earlier one — but `processed_depth` is clamped to never
    /// move backwards regardless, because a clamp does not hide caller
    /// misuse, it makes it harmless. This module is supposed to guarantee
    /// the "read only above the recorded depth" invariant on its own rather
    /// than trusting every call site to get it right (two separate
    /// front-end turn loops will call this).
    pub fn finish(&mut self, depth: usize) {
        self.processed_depth = self.processed_depth.max(depth);
        self.running = false;
        // A trailing run is enabled purely by clearing `running`: the next
        // `should_run` recomputes the span from `processed_depth` against
        // whatever depth it is given then.
    }

    /// Abandons an in-flight pass without recording progress, so the work is
    /// simply redone later. Nothing is half-applied: verdicts are applied
    /// only after the pass returns.
    pub fn cancel(&mut self) {
        self.running = false;
    }

    /// Re-anchors after a rewrite that kept the transcript's *tail* but
    /// changed its length — compaction replaces the head with a summary and
    /// keeps the last messages verbatim. The number of unprocessed trailing
    /// messages is preserved: whatever sat above `processed_depth` before the
    /// rewrite is still above it afterwards, and the new head (the summary)
    /// counts as processed, since it condenses text the pass already read.
    ///
    /// Without this the shrunken transcript satisfies
    /// `depth <= processed_depth` and the pass silently stops, then skips the
    /// post-rewrite messages once the depth grows back past the stale value —
    /// the same hazard `kvladder` guards against with `truncate_to`.
    pub fn rebase(&mut self, old_len: usize, new_len: usize) {
        let unseen = old_len.saturating_sub(self.processed_depth);
        self.processed_depth = new_len.saturating_sub(unseen);
    }

    /// Clamps the recorded depth after the transcript was truncated to
    /// `depth` (fork end, rollback): messages past the cut no longer exist,
    /// so a depth pointing past it would skip whatever replaces them.
    pub fn truncate_to(&mut self, depth: usize) {
        self.processed_depth = self.processed_depth.min(depth);
    }

    /// Adopts a transcript the pass has no history with (`/clear`, `/new`,
    /// `/resume`, `/switch`): everything up to `depth` counts as processed,
    /// so a restored session is never shipped wholesale to the model on the
    /// first idle turn, and the throttle and in-flight flag start over.
    pub fn reset_to(&mut self, depth: usize) {
        self.processed_depth = depth;
        self.running = false;
        self.eligible = 0;
    }

    /// Records that a pass replied with something no verdict could be read
    /// from. Returns `true` the first time in this state's lifetime, so the
    /// front end can say so once rather than on every idle turn.
    pub fn note_unusable_reply(&mut self) -> bool {
        self.unusable_replies = self.unusable_replies.saturating_add(1);
        self.unusable_replies == 1
    }
}

/// Locates the verdict array in a model reply. A local model told to answer
/// with "a JSON array and nothing else" still tends to wrap it in a markdown
/// fence or lead in with a sentence; `memory::parse_verdicts` parses from
/// position 0 and would reject both. This returns the outermost `[` … `]`
/// span, with any surrounding code fence removed first, or `None` when the
/// reply holds no array at all. An empty `[]` is returned as is: it is the
/// common, valid answer.
#[must_use]
pub fn extract_verdict_array(reply: &str) -> Option<&str> {
    let mut body = reply.trim();
    if let Some(rest) = body.strip_prefix("```") {
        // Drop the info string (```json) up to the end of the fence line.
        let rest = rest.split_once('\n').map_or("", |(_, after)| after);
        body = rest.strip_suffix("```").unwrap_or(rest).trim();
    }
    let start = body.find('[')?;
    let end = body.rfind(']')?;
    (end > start).then(|| &body[start..=end])
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
    fn a_trigger_while_running_is_dropped_and_yields_one_trailing_run() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        assert_eq!(s.should_run(12), None, "a pass is already running");
        assert_eq!(
            s.should_run(14),
            None,
            "still running; later triggers are simply dropped"
        );
        s.finish(10);
        assert_eq!(
            s.should_run(14),
            Some(10),
            "one trailing run re-derives the span and covers everything that arrived"
        );
    }

    #[test]
    fn finish_never_regresses_processed_depth() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        s.finish(10);
        // A stale, lower depth must not move processed_depth backwards.
        s.finish(4);
        assert_eq!(
            s.should_run(12),
            Some(10),
            "processed_depth stayed at 10, not regressed to 4"
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
    fn a_rewrite_that_shrinks_the_transcript_neither_disables_nor_skips() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        s.finish(10);
        // Two more messages arrive, then compaction folds the 12-message
        // transcript into a 1-message summary plus a 4-message verbatim tail.
        s.rebase(12, 5);
        assert_eq!(
            s.should_run(5),
            Some(3),
            "the two unprocessed tail messages are still above the depth"
        );
        s.finish(5);
        assert_eq!(s.should_run(5), None);
        assert_eq!(
            s.should_run(7),
            Some(5),
            "the pass keeps running afterwards"
        );
    }

    #[test]
    fn rebase_with_nothing_unseen_marks_the_whole_rewrite_processed() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        s.finish(10);
        s.rebase(10, 3);
        assert_eq!(s.should_run(3), None, "summary only — nothing new to read");
        assert_eq!(s.should_run(4), Some(3));
    }

    #[test]
    fn truncate_to_clamps_and_reset_to_adopts() {
        let mut s = state();
        assert_eq!(s.should_run(10), Some(0));
        s.finish(10);
        s.truncate_to(6);
        assert_eq!(
            s.should_run(8),
            Some(6),
            "clamped to the cut, not left at 10"
        );
        s.finish(8);
        s.reset_to(40);
        assert_eq!(
            s.should_run(40),
            None,
            "a restored transcript is not shipped wholesale"
        );
        assert_eq!(s.should_run(42), Some(40));
        s.reset_to(0);
        assert_eq!(s.should_run(2), Some(0), "/clear starts over");
    }

    #[test]
    fn an_unusable_reply_is_reported_once() {
        let mut s = state();
        assert!(s.note_unusable_reply());
        assert!(!s.note_unusable_reply());
    }

    #[test]
    fn extract_verdict_array_tolerates_fences_and_prose() {
        assert_eq!(extract_verdict_array("[]"), Some("[]"));
        assert_eq!(extract_verdict_array("  [ ]\n"), Some("[ ]"));
        assert_eq!(
            extract_verdict_array("```json\n[{\"verdict\": \"DELETE\", \"id\": \"a\"}]\n```"),
            Some("[{\"verdict\": \"DELETE\", \"id\": \"a\"}]")
        );
        assert_eq!(
            extract_verdict_array(
                "Here are my verdicts:\n[{\"verdict\": \"USED\", \"id\": \"b\"}]\nDone."
            ),
            Some("[{\"verdict\": \"USED\", \"id\": \"b\"}]")
        );
        assert_eq!(extract_verdict_array("Nothing worth saving."), None);
        assert_eq!(extract_verdict_array("]["), None);
    }

    #[test]
    fn a_fenced_reply_round_trips_through_parse_verdicts() {
        use crate::memory::{Verdict, parse_verdicts};
        let reply = "```json\n[{\"verdict\": \"DELETE\", \"id\": \"def456\"}]\n```";
        let json = extract_verdict_array(reply).expect("array found");
        let verdicts = parse_verdicts(json).expect("parses");
        assert_eq!(
            verdicts,
            vec![Verdict::Delete {
                id: "def456".to_string()
            }]
        );
    }

    #[test]
    fn build_prompt_documents_the_four_verdict_words() {
        let out = build_prompt(&[], &[]);
        for word in ["ADD", "UPDATE", "DELETE", "USED"] {
            assert!(out.contains(word), "prompt should mention verdict {word}");
        }
    }

    #[test]
    fn the_prompts_json_contract_round_trips_through_parse_verdicts() {
        use crate::memory::{Kind, Scope, Verdict, parse_verdicts};

        let json = r#"[
            {"verdict": "ADD", "text": "prefers tabs", "type": "user", "scope": "user"},
            {"verdict": "UPDATE", "id": "abc123", "text": "updated text"},
            {"verdict": "DELETE", "id": "def456"},
            {"verdict": "USED", "id": "aaa111"}
        ]"#;

        let verdicts = parse_verdicts(json).expect("valid JSON contract parses");
        assert_eq!(verdicts.len(), 4);
        assert_eq!(
            verdicts[0],
            Verdict::Add {
                text: "prefers tabs".to_string(),
                kind: Kind::User,
                scope: Scope::User,
            }
        );
        assert_eq!(
            verdicts[1],
            Verdict::Update {
                id: "abc123".to_string(),
                text: "updated text".to_string(),
            }
        );
        assert_eq!(
            verdicts[2],
            Verdict::Delete {
                id: "def456".to_string(),
            }
        );
        assert_eq!(
            verdicts[3],
            Verdict::Used {
                id: "aaa111".to_string(),
            }
        );
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

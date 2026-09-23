// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The memory extraction pass: gating state and prompt construction.
//!
//! Fires at the end of the query loop — a generation that produced a final
//! response with no tool calls — subject to five gates, in order
//! ([`ExtractState::should_run`]): enabled and not suppressed by the model's
//! own `remember`/`forget` call this turn, depth keying so the pass reads
//! only the transcript it has not already seen, a turn-duration floor
//! (`memory.minTurnSeconds`) below which the turn's span is deferred rather
//! than dropped, not already running, and a throttle over eligible turns.
//! On by default since 5.1.7 (`memory.autoExtract`).
//!
//! The depth keying is the same shape as a `kvladder` rung, and carries the
//! same warning: the recorded depth is what makes the resume correct, and
//! getting it wrong degrades *silently* into reprocessing the whole
//! transcript on every pass, at exactly the moment the user is idle.
//!
//! The pass itself runs synchronously on the turn thread as a sidechain
//! (`Agent::maybe_extract_memories` in `ui.rs`: `begin_sidechain`, one
//! tool-free generation, `end_subagent_fork`), so `in_sidechain()` holds: no
//! rungs pushed, no payload stored, no checkpoint debris.

use crate::memory::{Entry, Scope};
use crate::session::{Message, Role};

/// One snapshotted span waiting for an idle moment: the prompt the pass
/// will run, built from the transcript as it stood when the turn ended, so
/// the pass reads exactly what the model said then whatever the session does
/// meanwhile (`/clear`, a follow-up turn, compaction).
#[derive(Debug, Clone)]
pub struct MemoryJob {
    /// The complete extraction prompt (`build_prompt`), self-contained.
    pub task: String,
    /// Transcript depth the span ran to, for the sidechain dump's label.
    pub depth: usize,
    /// Run-time faults (engine errors) seen so far; the job is dropped at
    /// [`MAX_JOB_ATTEMPTS`] so a broken engine cannot pin the idle loop.
    pub attempts: u8,
    /// The prefilled prompt of an earlier attempt, kept when that attempt
    /// was interrupted so the retry does not start from zero.
    pub resume: Option<MemoryResume>,
}

/// A local engine's KV after the pass prompt was prefilled and before a
/// single token was sampled, with the exact prompt text it covers.
///
/// Why *before* sampling: the engine reuses a live KV only when the next
/// prompt extends its end, and rebuilds from zero when the prompt is a
/// strict prefix of it. A snapshot taken at the interrupt would hold the
/// partial reply, and the retry's prompt (the same task, a fresh assistant
/// turn) would be a strict prefix of that — worthless. So the pass prefills
/// first with `n_predict: 0`, snapshots, and only then samples; a retry
/// restores the snapshot and re-issues the stored prompt, which the KV
/// covers to the last token. Held in memory only while the job waits.
#[derive(Debug, Clone)]
pub struct MemoryResume {
    pub kv: crate::kvcache::KVCache,
    pub prompt: String,
}

/// Engine failures a queued job survives before it is dropped.
pub const MAX_JOB_ATTEMPTS: u8 = 3;

/// Gating state for the pass, owned by the `Agent`.
///
/// The bools are independent switches read side by side, not a state
/// machine in disguise: two settings mirrors, an open span, a running
/// generation, and the model's own write this turn.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct ExtractState {
    /// Mirror of `settings.memory.auto_extract`, sampled per turn.
    pub enabled: bool,
    /// Mirror of `settings.memory.extract_every_n_turns`.
    pub every_n: u32,
    /// Mirror of `settings.memory.min_turn_seconds`, as a `Duration`. A turn
    /// shorter than this defers the span instead of triggering a pass.
    /// `Duration::ZERO` (the default) is no floor.
    pub min_turn: std::time::Duration,
    /// Transcript depth the last completed pass covered.
    processed_depth: usize,
    /// Set between a positive `should_run` and the `finish`/`cancel` that
    /// closes the span — with the idle queue that is only the moment the
    /// span is snapshotted into a [`MemoryJob`].
    running: bool,
    /// Set while a queued job's generation is actually running, so the
    /// footer can name the pass (`Status::memory_pass`).
    processing: bool,
    /// Eligible turns seen since the last run, for the throttle.
    eligible: u32,
    /// Whether the model called `remember` or `forget` this turn.
    wrote_this_turn: bool,
    /// Passes whose reply yielded no verdicts; the front end announces
    /// only the first.
    unusable_replies: u32,
    /// Spans retired without a run because the prompt would not fit the
    /// context; the front end announces only the first.
    oversized_spans: u32,
    /// Messages a span rejected by the System-1 gate may accumulate before an
    /// extraction pass runs anyway, bypassing the gate.
    ///
    /// `0`, the default, means no holding at all: a rejected span is finished
    /// on the spot and never reconsidered, which bounds the cost and makes a
    /// false negative permanent. A positive value trades that for recovery —
    /// the span is held and re-judged as part of a larger one — while
    /// capping how large the held span can grow.
    pub held_span_cap: usize,
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
    ///
    /// A turn shorter than `min_turn` returns `None` without recording
    /// anything: the span is deferred to the next turn that clears the
    /// floor, never discarded.
    pub fn should_run(&mut self, depth: usize, turn: std::time::Duration) -> Option<usize> {
        let suppressed = std::mem::take(&mut self.wrote_this_turn);
        if !self.enabled || suppressed {
            return None;
        }
        if depth <= self.processed_depth {
            return None;
        }
        // A turn too short to be worth a pass. Deliberately *before* the
        // `eligible` increment, so a short turn is not merely throttled but
        // never counts as an eligible turn: `every_n` keeps meaning "every
        // Nth turn worth extracting from". Just as deliberately, nothing is
        // recorded here — `processed_depth` does not move and the span stays
        // open, so the next turn that clears the floor reads these turns too.
        if turn < self.min_turn {
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

    /// Whether the next [`should_run`](Self::should_run) call at `depth`
    /// would start a pass. A pure peek: it takes nothing and counts nothing,
    /// so a front end can decide *how* to run the pass (on a worker thread,
    /// behind a status line) before committing to it. A `false` here does
    /// not excuse the caller from calling `should_run` anyway: that call is
    /// what consumes the model's own `remember` suppression and advances the
    /// throttle.
    #[must_use]
    pub fn would_run(&self, depth: usize, turn: std::time::Duration) -> bool {
        self.enabled
            && !self.wrote_this_turn
            && depth > self.processed_depth
            && turn >= self.min_turn
            && !self.running
            && self.eligible.saturating_add(1) >= self.every_n.max(1)
    }

    /// Whether a queued job's generation is running right now (between
    /// `begin_pass` and `end_pass`): the footer's cue to say `taking notes…`.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.processing
    }

    /// Marks a queued job's generation as running; see [`is_running`](Self::is_running).
    pub fn begin_pass(&mut self) {
        self.processing = true;
    }

    /// The end of that generation, however it ended.
    pub fn end_pass(&mut self) {
        self.processing = false;
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

    /// Records that a span was retired unrun because its prompt could not
    /// fit the context. Returns `true` the first time, for a one-off notice.
    pub fn note_oversized_span(&mut self) -> bool {
        self.oversized_spans = self.oversized_spans.saturating_add(1);
        self.oversized_spans == 1
    }

    /// Whether the span `from..depth` has outgrown [`held_span_cap`] and must
    /// run an extraction pass without consulting the gate. Always false when
    /// the cap is `0`, because nothing is ever held in that mode.
    ///
    /// [`held_span_cap`]: Self::held_span_cap
    #[must_use]
    pub fn gate_bypassed(&self, from: usize, depth: usize) -> bool {
        self.held_span_cap > 0 && depth.saturating_sub(from) > self.held_span_cap
    }

    /// Records that the System-1 gate judged `from..depth` not worth
    /// extracting.
    ///
    /// With no cap this is a completed pass — `processed_depth` advances and
    /// the span is gone. With a cap the span is held: `processed_depth` stays
    /// put, so the next eligible turn re-reads these messages together with
    /// the new ones.
    pub fn reject(&mut self, _from: usize, depth: usize) {
        if self.held_span_cap == 0 {
            self.finish(depth);
        } else {
            self.cancel();
        }
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

/// Upper bound, in bytes, on the rendered conversation excerpt.
///
/// The excerpt rides on top of the *whole live session*, which is already
/// resident in the context (the sidechain prompt is the transcript plus this
/// task), so it can only ever use the headroom compaction leaves. 32 KiB is
/// roughly 8k tokens at the engine's ~4-bytes-per-token estimate: dozens of
/// turns of prose — far more than one pass needs, since the pass runs every
/// eligible turn and only the messages above its recorded depth are new —
/// yet small enough that it never competes with the answer just given. The
/// newest content is kept and the oldest dropped first, because the pass
/// reads forward from its depth: the most recent turns are the ones that
/// have not been seen, and a fact stated long ago and still unsaved was
/// most likely judged not worth saving already.
pub const EXCERPT_MAX_BYTES: usize = 32 * 1024;

/// Tokens the preflight in `Agent::maybe_extract_memories` reserves for the
/// reply on top of the prompt. A verdict array is short — a handful of
/// entries at most — so 1024 tokens is ample while still refusing a span
/// that would leave the model no room to answer at all.
pub const REPLY_RESERVE_TOKENS: i32 = 1024;

/// Bytes set aside inside [`EXCERPT_MAX_BYTES`] for the omission note that
/// heads a truncated excerpt, so the note never pushes the excerpt over the
/// cap it announces.
const OMISSION_NOTE_RESERVE: usize = 96;

/// Renders a message's role for the prompt. `Role` is not `Display`, so this
/// mirrors the tag words used elsewhere in the transcript.
fn role_word(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// Renders one transcript message as an excerpt line. A tool result — a
/// user turn framed as `<tool_result>` — is replaced by a short placeholder
/// naming its size: tool output is the bulk of any tool-heavy stretch (each
/// body may run to `tools.spillMaxBytes`, 1 MiB by default), it is the least
/// useful part for extracting durable facts about the *user*, and the model
/// already has the real bytes resident in the KV cache when it reads this.
fn excerpt_line(m: &Message) -> String {
    if m.role == Role::User && m.is_tool_user() {
        let bytes = m.tool_result_payload().len();
        return format!("user: [tool result omitted, {bytes} bytes]\n");
    }
    format!("{}: {}\n", role_word(m.role), m.text)
}

/// Truncates `text` to at most `max` bytes on a char boundary, keeping the
/// head: the point of a user message is usually made at its start.
fn truncate_to(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Renders `slice` as the conversation excerpt, bounded by
/// [`EXCERPT_MAX_BYTES`]: tool-result bodies become placeholders, and when
/// the lines still exceed the cap the oldest messages are dropped first, with
/// a note saying how many. A single newest message wider than the whole cap
/// is cut to fit rather than dropped, so the excerpt is never empty for a
/// non-empty slice.
#[must_use]
pub fn render_excerpt(slice: &[Message]) -> String {
    let budget = EXCERPT_MAX_BYTES.saturating_sub(OMISSION_NOTE_RESERVE);
    let mut kept: Vec<String> = Vec::new();
    let mut used = 0usize;
    let mut omitted = 0usize;
    for m in slice.iter().rev() {
        if omitted > 0 {
            omitted += 1;
            continue;
        }
        let line = excerpt_line(m);
        if used + line.len() <= budget {
            used += line.len();
            kept.push(line);
        } else if kept.is_empty() {
            // Even the newest message does not fit: keep its head.
            let cut = truncate_to(&line, budget.saturating_sub(16));
            kept.push(format!("{}[...]\n", cut.trim_end_matches('\n')));
        } else {
            omitted = 1;
        }
    }
    let mut out = String::with_capacity(used + OMISSION_NOTE_RESERVE);
    if omitted > 0 {
        use std::fmt::Write as _;
        let _ = writeln!(out, "[{omitted} earlier message(s) omitted for length]");
    }
    for line in kept.iter().rev() {
        out.push_str(line);
    }
    out
}

/// Builds the pass's prompt: the bounded transcript excerpt
/// ([`render_excerpt`]), the current entries with their ids, and the
/// verdict contract.
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
    out.push_str(&render_excerpt(slice));
    out
}

/// Reads every entry from both scopes, for the prompt. The file is the
/// whole truth here: a model `forget` deletes its entry outright, so there
/// is no hidden state to filter against.
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
        for e in crate::memory::parse_entries(&body) {
            out.push((scope, e));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn comfortably over any floor these tests set.
    const LONG: std::time::Duration = std::time::Duration::from_secs(600);

    #[test]
    fn would_run_peeks_without_consuming() {
        let mut s = state();
        assert!(s.would_run(10, LONG), "the first eligible turn");
        assert!(s.would_run(10, LONG), "a peek changes nothing");
        assert_eq!(s.should_run(10, LONG), Some(0));
        assert!(!s.would_run(12, LONG), "a span is open");
        s.finish(10);
        assert!(!s.would_run(10, LONG), "nothing new");
        assert!(s.would_run(11, LONG));
        // The model's own write suppresses the pass, and the peek agrees
        // without clearing the suppression: `should_run` still does that.
        s.note_tool_write();
        assert!(!s.would_run(11, LONG));
        assert_eq!(s.should_run(11, LONG), None);
        assert!(
            s.would_run(11, LONG),
            "suppression consumed by should_run alone"
        );
    }

    #[test]
    fn is_running_tracks_the_generation_not_the_span() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        assert!(!s.is_running(), "an open span is not a running generation");
        s.finish(10);
        s.begin_pass();
        assert!(s.is_running());
        s.end_pass();
        assert!(!s.is_running());
    }

    #[test]
    fn would_run_agrees_with_the_throttle() {
        let mut s = ExtractState {
            enabled: true,
            every_n: 3,
            ..ExtractState::default()
        };
        assert!(!s.would_run(4, LONG));
        assert_eq!(s.should_run(4, LONG), None);
        assert!(!s.would_run(6, LONG));
        assert_eq!(s.should_run(6, LONG), None);
        assert!(s.would_run(8, LONG), "the third eligible turn");
        assert_eq!(s.should_run(8, LONG), Some(0));
    }

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
        assert_eq!(
            s.should_run(10, LONG),
            Some(0),
            "first pass covers depth 0..10"
        );
        s.finish(10);
        assert_eq!(s.should_run(10, LONG), None, "nothing new — no pass");
        assert_eq!(
            s.should_run(14, LONG),
            Some(10),
            "resumes from the recorded depth"
        );
    }

    #[test]
    fn a_remember_call_in_the_turn_suppresses_the_pass() {
        let mut s = state();
        s.note_tool_write();
        assert_eq!(
            s.should_run(10, LONG),
            None,
            "the model's own judgment wins"
        );
        assert_eq!(
            s.should_run(10, LONG),
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
        assert_eq!(s.should_run(4, LONG), None);
        assert_eq!(s.should_run(6, LONG), None);
        assert_eq!(s.should_run(8, LONG), Some(0), "every third eligible turn");
    }

    #[test]
    fn a_trigger_while_running_is_dropped_and_yields_one_trailing_run() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        assert_eq!(s.should_run(12, LONG), None, "a pass is already running");
        assert_eq!(
            s.should_run(14, LONG),
            None,
            "still running; later triggers are simply dropped"
        );
        s.finish(10);
        assert_eq!(
            s.should_run(14, LONG),
            Some(10),
            "one trailing run re-derives the span and covers everything that arrived"
        );
    }

    #[test]
    fn finish_never_regresses_processed_depth() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.finish(10);
        // A stale, lower depth must not move processed_depth backwards.
        s.finish(4);
        assert_eq!(
            s.should_run(12, LONG),
            Some(10),
            "processed_depth stayed at 10, not regressed to 4"
        );
    }

    #[test]
    fn cancelling_leaves_the_recorded_depth_untouched() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.cancel();
        assert_eq!(
            s.should_run(10, LONG),
            Some(0),
            "the work is simply redone later"
        );
    }

    #[test]
    fn a_rewrite_that_shrinks_the_transcript_neither_disables_nor_skips() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.finish(10);
        // Two more messages arrive, then compaction folds the 12-message
        // transcript into a 1-message summary plus a 4-message verbatim tail.
        s.rebase(12, 5);
        assert_eq!(
            s.should_run(5, LONG),
            Some(3),
            "the two unprocessed tail messages are still above the depth"
        );
        s.finish(5);
        assert_eq!(s.should_run(5, LONG), None);
        assert_eq!(
            s.should_run(7, LONG),
            Some(5),
            "the pass keeps running afterwards"
        );
    }

    #[test]
    fn rebase_with_nothing_unseen_marks_the_whole_rewrite_processed() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.finish(10);
        s.rebase(10, 3);
        assert_eq!(
            s.should_run(3, LONG),
            None,
            "summary only — nothing new to read"
        );
        assert_eq!(s.should_run(4, LONG), Some(3));
    }

    #[test]
    fn truncate_to_clamps_and_reset_to_adopts() {
        let mut s = state();
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.finish(10);
        s.truncate_to(6);
        assert_eq!(
            s.should_run(8, LONG),
            Some(6),
            "clamped to the cut, not left at 10"
        );
        s.finish(8);
        s.reset_to(40);
        assert_eq!(
            s.should_run(40, LONG),
            None,
            "a restored transcript is not shipped wholesale"
        );
        assert_eq!(s.should_run(42, LONG), Some(40));
        s.reset_to(0);
        assert_eq!(s.should_run(2, LONG), Some(0), "/clear starts over");
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
    fn an_oversized_span_is_reported_once() {
        let mut s = state();
        assert!(s.note_oversized_span());
        assert!(!s.note_oversized_span());
    }

    #[test]
    fn a_large_tool_result_yields_a_bounded_prompt() {
        // One tool result at the spill ceiling (`tools.spillMaxBytes`, 1 MiB)
        // plus the turn around it. Unbounded, the prompt would be ~1 MiB.
        let big = "z".repeat(1024 * 1024);
        let slice = vec![
            Message::user("read the log"),
            Message::assistant("reading"),
            Message::user(format!("<tool_result>{big}</tool_result>")),
            Message::assistant("the log says hello"),
        ];
        let out = build_prompt(&slice, &[]);
        let fixed = build_prompt(&[], &[]).len();
        assert!(
            out.len() <= fixed + EXCERPT_MAX_BYTES,
            "prompt is {} bytes, excerpt cap is {EXCERPT_MAX_BYTES}",
            out.len()
        );
        assert!(!out.contains("zzzz"), "the tool body must not be quoted");
        assert!(
            out.contains("[tool result omitted, 1048576 bytes]"),
            "the placeholder names the size: {}",
            &out[out.len().saturating_sub(300)..]
        );
        assert!(out.contains("user: read the log"));
        assert!(out.contains("assistant: the log says hello"));
    }

    #[test]
    fn the_excerpt_drops_the_oldest_messages_first() {
        // 40 prose messages of 2 KiB each: 80 KiB against a 32 KiB cap.
        let slice: Vec<Message> = (0..40)
            .map(|i| Message::user(format!("msg-{i:02} {}", "p".repeat(2048))))
            .collect();
        let out = render_excerpt(&slice);
        assert!(out.len() <= EXCERPT_MAX_BYTES, "{} bytes", out.len());
        assert!(out.contains("msg-39"), "the newest message is kept");
        assert!(!out.contains("msg-00"), "the oldest message is dropped");
        assert!(
            out.starts_with('['),
            "a truncated excerpt announces the omission: {}",
            &out[..60]
        );
        let omitted: usize = out
            .trim_start_matches('[')
            .split(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .expect("the note starts with a count");
        let kept = out.matches("user: msg-").count();
        assert_eq!(
            omitted + kept,
            40,
            "every message is either kept or counted"
        );
    }

    #[test]
    fn a_single_oversized_message_is_cut_rather_than_dropped() {
        let slice = vec![Message::user("q".repeat(EXCERPT_MAX_BYTES * 2))];
        let out = render_excerpt(&slice);
        assert!(out.len() <= EXCERPT_MAX_BYTES, "{} bytes", out.len());
        assert!(out.starts_with("user: qqq"), "the head is kept");
        assert!(out.trim_end().ends_with("[...]"), "the cut is marked");
    }

    #[test]
    fn a_short_excerpt_is_rendered_verbatim() {
        let slice = vec![Message::user("hi"), Message::assistant("hello")];
        assert_eq!(render_excerpt(&slice), "user: hi\nassistant: hello\n");
    }

    #[test]
    fn disabled_never_runs() {
        let mut s = ExtractState {
            enabled: false,
            every_n: 1,
            ..ExtractState::default()
        };
        assert_eq!(s.should_run(100, LONG), None);
    }

    #[test]
    fn a_short_turn_does_not_run_and_leaves_the_span_open() {
        let mut s = ExtractState {
            min_turn: std::time::Duration::from_secs(120),
            ..state()
        };
        assert_eq!(
            s.should_run(10, std::time::Duration::from_secs(4)),
            None,
            "a four-second turn is not worth a pass"
        );
        // Deferred, not discarded: the span is still open at depth 0, and
        // the state is not left mid-pass.
        assert_eq!(
            s.should_run(10, LONG),
            Some(0),
            "the very next long turn reads the span the short turn left"
        );
    }

    #[test]
    fn a_long_turn_reads_back_across_deferred_short_turns() {
        let mut s = ExtractState {
            min_turn: std::time::Duration::from_secs(120),
            ..state()
        };
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.finish(10);
        // Three short turns grow the transcript but trigger nothing.
        for depth in [12, 14, 16] {
            assert_eq!(s.should_run(depth, std::time::Duration::from_secs(5)), None);
        }
        // The long turn that follows starts from the last *completed* pass,
        // not from the last short turn, so nothing said in between is lost.
        assert_eq!(
            s.should_run(18, LONG),
            Some(10),
            "the deferred span must be read in full"
        );
    }

    #[test]
    fn a_short_turn_does_not_consume_a_throttle_tick() {
        let mut s = ExtractState {
            enabled: true,
            every_n: 2,
            min_turn: std::time::Duration::from_secs(120),
            ..ExtractState::default()
        };
        assert_eq!(
            s.should_run(4, std::time::Duration::from_secs(3)),
            None,
            "short: not a pass, and not an eligible turn either"
        );
        assert_eq!(
            s.should_run(6, LONG),
            None,
            "the first eligible turn — the short one did not count"
        );
        assert_eq!(s.should_run(8, LONG), Some(0), "the second eligible turn");
    }

    #[test]
    fn a_zero_floor_reproduces_the_old_behavior() {
        let mut s = state();
        assert_eq!(s.min_turn, std::time::Duration::ZERO, "the default floor");
        assert_eq!(
            s.should_run(10, std::time::Duration::ZERO),
            Some(0),
            "with no floor even an instantaneous turn runs"
        );
    }

    #[test]
    fn would_run_agrees_with_the_floor() {
        let mut s = ExtractState {
            min_turn: std::time::Duration::from_secs(120),
            ..state()
        };
        let short = std::time::Duration::from_secs(4);
        assert!(!s.would_run(10, short), "the peek must see the floor too");
        assert!(s.would_run(10, LONG));
        assert_eq!(s.should_run(10, short), None);
        assert!(
            s.would_run(10, LONG),
            "a rejected short turn leaves the peek unchanged"
        );
    }

    #[test]
    fn with_no_cap_a_rejected_span_is_finished_and_never_seen_again() {
        let mut s = state();
        s.held_span_cap = 0;
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.reject(0, 10);
        assert_eq!(
            s.should_run(10, LONG),
            None,
            "the span is done: processed_depth advanced past it"
        );
    }

    #[test]
    fn with_a_cap_a_rejected_span_is_held_for_the_next_turn() {
        let mut s = state();
        s.held_span_cap = 5;
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.reject(0, 10);
        assert_eq!(
            s.should_run(12, LONG),
            Some(0),
            "the held span is re-read as part of a larger one"
        );
    }

    #[test]
    fn the_gate_is_bypassed_once_a_held_span_outgrows_the_cap() {
        let s = {
            let mut s = state();
            s.held_span_cap = 5;
            s
        };
        assert!(!s.gate_bypassed(0, 5), "exactly at the cap still asks");
        assert!(s.gate_bypassed(0, 6), "past the cap runs unconditionally");
    }

    #[test]
    fn a_zero_cap_never_bypasses_the_gate() {
        let mut s = state();
        s.held_span_cap = 0;
        assert!(!s.gate_bypassed(0, 10_000));
        // And a rejection still finishes rather than holding.
        assert_eq!(s.should_run(10, LONG), Some(0));
        s.reject(0, 10);
        assert_eq!(s.should_run(10, LONG), None);
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Context compaction: reclaim context in escalating steps.
//!
//! Cheapest first: **microcompact** clears the bodies of old tool results
//! (keeping the newest few) without any model round-trip. When that is not
//! enough, full compaction asks the model for durable task state and rebuilds
//! the live transcript as: system prompt + summary + recent verbatim tail +
//! a budgeted re-injection of recently read files. Port of the "Context
//! Compaction" section of `ds4_agent.c`, adapted from token transcripts to
//! text messages, with the layered strategy from the reference agent.

use crate::session::{Message, Role};

/// Compact once used context reaches this percentage of the window.
pub const COMPACT_SOFT_PERCENT: i32 = 85;
/// Also compact when fewer than this many tokens remain free.
pub const COMPACT_MIN_FREE_TOKENS: i32 = 8192;
/// The verbatim tail keeps at most `ctx / TAIL_DIVISOR` tokens.
pub const COMPACT_TAIL_DIVISOR: i32 = 8;
/// Hard cap on the verbatim tail, in tokens.
pub const COMPACT_TAIL_CAP_TOKENS: i32 = 8192;
/// Newest tool results microcompact leaves intact.
pub const MICROCOMPACT_KEEP_RESULTS: usize = 3;
/// Tool-result bodies at or below this many bytes are not worth clearing.
pub const MICROCOMPACT_MIN_BYTES: usize = 256;
/// Replacement body for tool results cleared by microcompact.
pub const MICROCOMPACT_STUB: &str =
    "[old tool result cleared to reclaim context; rerun the tool if the output is needed again]";
/// An opportunistic end-of-turn microcompact only fires when it would reclaim
/// at least this many bytes: pruning mid-session rewrites transcript text in
/// place, which invalidates the KV prefix from that point, so an eager pass
/// that reclaims little costs more than it saves.
pub const MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES: usize = 4096;
/// Bytes a pass must reclaim per token it forces back through prefill.
///
/// Prefill measured at ~88 tok/s against a buffered write at GB/s, so a token
/// re-prefilled is far more expensive than a byte reclaimed; this floor is
/// deliberately conservative rather than tuned.
pub const MICROCOMPACT_BYTES_PER_TOKEN_FLOOR: f64 = 2.0;

/// The floor [`microcompact_floor`] relaxes *to* at the full-compaction
/// trigger. Deliberately not zero: see that function.
pub const MICROCOMPACT_FLOOR_EPSILON: f64 = 0.05;
/// Maximum files re-injected after a full compaction.
pub const REINJECT_MAX_FILES: usize = 5;
/// Hard cap on the post-compaction re-injection budget, in tokens.
pub const REINJECT_CAP_TOKENS: i32 = 50_000;

/// The indices of tool-result bodies microcompact would clear: the newest
/// [`MICROCOMPACT_KEEP_RESULTS`] survive, plus anything under
/// [`MICROCOMPACT_MIN_BYTES`] (never candidates) plus anything belonging to
/// the current task — a tool result that follows the last `# Task list`
/// injection is part of the active work and is kept.
fn clear_set(transcript: &[Message]) -> Vec<usize> {
    let task_inject = transcript
        .iter()
        .rposition(|m| m.text.contains("# Task list"));
    let idx: Vec<usize> = transcript
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.role == Role::User
                && m.text.starts_with("<tool_result>")
                && m.text.len() > MICROCOMPACT_MIN_BYTES
                // Image-bearing tool results are exempt: the image tokens
                // cannot be stubbed, and clearing the text would orphan them
                // from their observation context (C parity).
                && m.images.is_empty()
        })
        .map(|(i, _)| i)
        .collect();
    idx.iter()
        .enumerate()
        .filter(|(pos, i)| {
            let is_last = *pos >= idx.len().saturating_sub(MICROCOMPACT_KEEP_RESULTS);
            let is_task = task_inject.is_some_and(|t| **i > t);
            !is_last && !is_task
        })
        .map(|(_, i)| *i)
        .collect()
}

/// Bytes a microcompact would reclaim, without clearing anything. Used to gate
/// the opportunistic end-of-turn pass on [`MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES`].
#[must_use]
pub fn microcompact_reclaimable(transcript: &[Message]) -> usize {
    clear_set(transcript)
        .iter()
        .map(|&i| transcript[i].text.len())
        .sum()
}

/// The earliest transcript index [`microcompact`] would rewrite, or `None`
/// when it would clear nothing.
///
/// This is the KV divergence point: rewriting a body in place makes every
/// token from here on diverge from the live KV, and `ds4_session_sync` cannot
/// rewrite behind its live end, so the whole prefix is otherwise rebuilt from
/// zero. The agent uses this to pick a snapshot rung at or below the edit.
#[must_use]
pub fn microcompact_first_index(transcript: &[Message]) -> Option<usize> {
    clear_set(transcript).into_iter().min()
}

/// The used-token level at which [`should_compact`] starts returning true.
///
/// Both of that function's conditions expressed as one threshold, so the
/// opportunistic gate and the full-compaction decision cannot disagree about
/// where "pressure" is.
#[must_use]
pub fn compaction_trigger_used(ctx_size: i32) -> i32 {
    if ctx_size <= 0 {
        return 0;
    }
    let soft = (ctx_size * COMPACT_SOFT_PERCENT) / 100;
    let free_threshold = COMPACT_MIN_FREE_TOKENS.min(ctx_size / 4);
    soft.min(ctx_size - free_threshold).max(0)
}

/// The bytes-per-token floor an opportunistic pass must clear at this level of
/// context pressure.
///
/// Flat [`MICROCOMPACT_BYTES_PER_TOKEN_FLOOR`] until the window is half way to
/// the full-compaction trigger, then a clamped linear relaxation down to
/// [`MICROCOMPACT_FLOOR_EPSILON`] at the trigger itself: at that point full
/// compaction is imminent and will rebuild the KV anyway, so a cheap pass must
/// never be the blocker.
///
/// The floor bottoms out at a small epsilon rather than at zero, which would
/// mean "accept any pass at all". The opportunistic pass runs at the *end* of a
/// turn while `should_compact` is consulted at the *start* of the next, so a
/// pass barely clearing [`MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES`] could spend a
/// `set_kv` and an in-place rewrite moments before a full compaction discards
/// the ladder and rebuilds anyway. The waste was bounded but pointless. The
/// epsilon keeps every property the function had: strict at low pressure,
/// monotone non-increasing in pressure, monotone in bytes and tokens through
/// [`microcompact_is_worth_it`], and far too small to be the blocker at high
/// pressure.
#[must_use]
pub fn microcompact_floor(ctx_size: i32, ctx_used: i32) -> f64 {
    let trigger = compaction_trigger_used(ctx_size);
    if trigger <= 0 || ctx_used <= 0 {
        return MICROCOMPACT_BYTES_PER_TOKEN_FLOOR;
    }
    if ctx_used >= trigger {
        return MICROCOMPACT_FLOOR_EPSILON;
    }
    let relax_start = trigger / 2;
    if ctx_used <= relax_start {
        return MICROCOMPACT_BYTES_PER_TOKEN_FLOOR;
    }
    let span = f64::from(trigger - relax_start);
    let remaining = f64::from(trigger - ctx_used) / span;
    MICROCOMPACT_FLOOR_EPSILON
        + (MICROCOMPACT_BYTES_PER_TOKEN_FLOOR - MICROCOMPACT_FLOOR_EPSILON) * remaining
}

/// Whether clearing `reclaimable` bytes justifies `reprefill_tokens` of
/// prefill, at the context pressure `ctx_used`/`ctx_size` describes.
///
/// The first gate compared bytes against a flat threshold and could not see the
/// token cost at all: one measured pass reclaimed ~12 KB and paid ~14,000
/// tokens, because rewriting a body in place forces a rebuild from zero. The
/// second was token-denominated but pressure-blind: a measured 1.16 bytes per
/// token is a bad trade at 2% of the window and an obviously good one at 80%,
/// where the alternative is a full compaction (a model round-trip plus a total
/// rebuild). The floor therefore relaxes as pressure rises — see
/// [`microcompact_floor`].
#[must_use]
pub fn microcompact_is_worth_it(
    reclaimable: usize,
    reprefill_tokens: i32,
    ctx_size: i32,
    ctx_used: i32,
) -> bool {
    if reprefill_tokens <= 0 {
        return true;
    }
    let reclaimable = u32::try_from(reclaimable).unwrap_or(u32::MAX);
    f64::from(reclaimable) / f64::from(reprefill_tokens) >= microcompact_floor(ctx_size, ctx_used)
}

/// Clears the bodies of old tool results in place, keeping the newest
/// [`MICROCOMPACT_KEEP_RESULTS`] intact plus anything belonging to the current
/// task; returns `(cleared, bytes_reclaimed)`.
///
/// This is the cheap first step of compaction: no model round-trip, and the
/// conversation flow (user turns, assistant turns, tool-call structure) is
/// preserved — only bulky, stale tool output is dropped. Clearing an early
/// message invalidates the KV prefix from that point, but so would a full
/// compaction, and this one costs zero generated tokens.
pub fn microcompact(transcript: &mut [Message]) -> (usize, usize) {
    let clear = clear_set(transcript);
    let mut bytes = 0usize;
    for &i in &clear {
        bytes += transcript[i].text.len();
        transcript[i].text = format!("<tool_result>{MICROCOMPACT_STUB}</tool_result>");
    }
    (clear.len(), bytes)
}

/// Token budget for the post-compaction file re-injection.
#[must_use]
pub fn reinject_budget(ctx_size: i32) -> i32 {
    (ctx_size / 8).clamp(0, REINJECT_CAP_TOKENS)
}

/// Builds the post-compaction re-injection block: current contents of the
/// most recently read files (newest first), up to [`REINJECT_MAX_FILES`]
/// files and `budget` tokens. Files that no longer exist or would exceed the
/// remaining budget are skipped. Returns `None` when nothing fits.
pub fn build_reinjection(
    recent_reads: &[std::path::PathBuf],
    budget: i32,
    count_tokens: &mut dyn FnMut(&str) -> i32,
) -> Option<String> {
    let mut out = String::from(
        "<tool_result>Post-compaction context re-injection: current contents of recently read files.\n",
    );
    let mut remaining = budget;
    let mut included = 0;
    for path in recent_reads.iter().rev() {
        if included == REINJECT_MAX_FILES || remaining <= 0 {
            break;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        let section = format!("\n=== {} ===\n{content}\n", path.display());
        let cost = count_tokens(&section);
        if cost > remaining {
            continue;
        }
        remaining -= cost;
        out.push_str(&section);
        included += 1;
    }
    if included == 0 {
        return None;
    }
    out.push_str("</tool_result>");
    Some(out)
}

/// Decides when to compact before a turn or a large tool result.
///
/// The fixed free-token threshold is capped proportionally for smaller
/// contexts so tiny-context runs still compact rather than fail.
#[must_use]
pub fn should_compact(ctx_size: i32, ctx_used: i32) -> bool {
    if ctx_size <= 0 || ctx_used <= 0 {
        return false;
    }
    if ctx_used >= (ctx_size * COMPACT_SOFT_PERCENT) / 100 {
        return true;
    }
    let free_threshold = COMPACT_MIN_FREE_TOKENS.min(ctx_size / 4);
    ctx_size - ctx_used <= free_threshold
}

/// Token budget for the verbatim tail kept after compaction.
#[must_use]
pub fn tail_budget(ctx_size: i32) -> i32 {
    (ctx_size / COMPACT_TAIL_DIVISOR).clamp(1, COMPACT_TAIL_CAP_TOKENS)
}

/// Everything the compaction prompt says before any caller-supplied
/// instructions: what is being asked for, the fixed sections, and the
/// `<analysis>`/`<summary>` tag contract.
const PROMPT_BODY: &str = "Internal plank-agent context compaction request. This is not a user request.\n\
     Summarize the conversation so far into durable task state for continuing the work. Use exactly these numbered sections, omitting none (write \"none\" when a section is empty):\n\
     1. Primary request and intent\n\
     2. Key technical concepts\n\
     3. Files and code sections (exact paths, ranges, and why each matters)\n\
     4. Errors and fixes (including rejected approaches and known bugs)\n\
     5. All user messages (condensed, in order)\n\
     6. Pending tasks\n\
     7. Current work (what was in progress at this very moment)\n\
     8. Next step (only if one was explicitly requested by the user)\n\n\
     You may reason first inside a single <analysis>...</analysis> block; it will be discarded. Then wrap the final summary in <summary>...</summary> tags.\n\
     Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion; prefer exact paths/ranges/commands that can reload the data.\n";

/// The closing instruction, kept **last** in the prompt so it is the final thing
/// the model reads before it starts writing: any caller-supplied instructions go
/// above it and cannot displace the no-tools rule.
const PROMPT_TRAILER: &str = "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n";

/// Builds the private prompt used to ask the model for durable state.
///
/// Asks for a fixed-section summary wrapped in `<summary>` tags, with an
/// optional `<analysis>` scratch block that [`extract_summary`] strips. The
/// prompt explicitly forbids tool calls because the result is consumed
/// internally, not delivered as an assistant turn.
///
/// `instructions` carries the argument to `/compact <instructions>` (empty for
/// automatic compaction). It is inserted between the section list and the
/// closing no-tools trailer, and framed as *additional* to keep it from being
/// read as a replacement for the section contract that
/// [`extract_summary`] and the rebuild depend on.
#[must_use]
pub fn make_prompt(reason: &str, instructions: &str) -> String {
    let mut b = String::from(PROMPT_BODY);
    let instructions = instructions.trim();
    if !instructions.is_empty() {
        b.push_str("\nAdditional instructions from the user for this summary (follow them in addition to, not instead of, the sections above):\n");
        b.push_str(instructions);
        b.push('\n');
    }
    b.push_str(PROMPT_TRAILER);
    if !reason.is_empty() {
        b.push_str("\nCompaction reason: ");
        b.push_str(reason);
        b.push('\n');
    }
    b
}

/// Extracts the durable summary from a raw compaction reply: `<analysis>`
/// blocks are discarded, and the `<summary>` body is unwrapped when present.
/// Falls back to the stripped text so a model that ignores the tag contract
/// still compacts usefully.
#[must_use]
pub fn extract_summary(raw: &str) -> String {
    let mut text = raw.to_string();
    while let (Some(start), Some(end)) = (text.find("<analysis>"), text.find("</analysis>")) {
        if end < start {
            break;
        }
        text.replace_range(start..end + "</analysis>".len(), "");
    }
    if let Some(start) = text.find("<summary>") {
        let body = &text[start + "<summary>".len()..];
        let body = body.find("</summary>").map_or(body, |end| &body[..end]);
        return body.trim().to_string();
    }
    text.trim().to_string()
}

/// Banner announcing a compaction pass, mirroring the C UX string.
#[must_use]
pub fn banner(reason: &str, color: bool) -> String {
    let reason = if reason.is_empty() { "context" } else { reason };
    if color {
        format!(
            "\n\x1b[1;95mCOMPACTING\x1b[0m {reason}: summarizing durable task state\n\x1b[38;5;245m"
        )
    } else {
        format!("\nCOMPACTING {reason}: summarizing durable task state\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_percent_triggers() {
        assert!(should_compact(1000, 850));
        assert!(!should_compact(100_000, 50_000));
    }

    #[test]
    fn min_free_triggers_with_proportional_cap() {
        // Large context: 8192 free tokens left → compact.
        assert!(should_compact(100_000, 92_000));
        // Tiny context: proportional cap (ctx/4) applies.
        assert!(should_compact(400, 301));
        assert!(!should_compact(400, 200));
    }

    #[test]
    fn tail_budget_capped() {
        assert_eq!(tail_budget(100_000), 8192);
        assert_eq!(tail_budget(8000), 1000);
        assert_eq!(tail_budget(0), 1);
    }

    #[test]
    fn prompt_includes_reason() {
        assert!(make_prompt("low context", "").contains("Compaction reason: low context"));
        assert!(!make_prompt("", "").contains("Compaction reason"));
    }

    #[test]
    fn prompt_carries_custom_instructions_above_the_no_tools_trailer() {
        let p = make_prompt("user request", "focus on the parser bug");
        assert!(p.contains("focus on the parser bug"), "{p}");
        // The section contract must still be asked for: extra instructions are
        // additional, never a replacement.
        assert!(p.contains("1. Primary request and intent"));
        // The no-tools trailer stays last, so instructions cannot displace it.
        let at = p.find("focus on the parser bug").unwrap();
        let trailer = p.find("do not call tools").unwrap();
        assert!(at < trailer, "instructions must precede the trailer");
        // Blank or whitespace-only arguments add nothing at all.
        assert_eq!(
            make_prompt("user request", ""),
            make_prompt("user request", "   ")
        );
        assert!(!make_prompt("user request", "  ").contains("Additional instructions"));
    }

    #[test]
    fn prompt_asks_for_fixed_sections() {
        let p = make_prompt("", "");
        assert!(p.contains("1. Primary request and intent"));
        assert!(p.contains("8. Next step"));
        assert!(p.contains("<summary>"));
        assert!(p.contains("<analysis>"));
    }

    // The prompt and `extract_summary` are two halves of one contract: the
    // prompt names the tags, the parser looks for them. Renaming either alone
    // fails silently at runtime — an empty summary, or one still carrying the
    // model's reasoning — so assert a reply shaped exactly as the prompt
    // instructs round-trips through the parser.
    #[test]
    fn a_reply_following_the_prompt_round_trips_through_extract_summary() {
        let prompt = make_prompt("low context", "");

        // Build the reply from the tag names the prompt itself asks for, so a
        // rename in the prompt moves this test's input with it and only a
        // *mismatch* between prompt and parser can fail.
        let analysis = prompt
            .split_once("inside a single ")
            .and_then(|(_, rest)| rest.split_once("..."))
            .map(|(open, _)| open.to_owned())
            .expect("the prompt names an analysis open tag");
        let summary = prompt
            .split_once("wrap the final summary in ")
            .and_then(|(_, rest)| rest.split_once("..."))
            .map(|(open, _)| open.to_owned())
            .expect("the prompt names a summary open tag");
        let close = |open: &str| format!("</{}", open.trim_start_matches('<'));

        let reply = format!(
            "{analysis}deciding what matters{}\n{summary}\n1. Primary request and intent\nPort the thing.\n{}\n",
            close(&analysis),
            close(&summary)
        );

        let got = extract_summary(&reply);
        assert_eq!(got, "1. Primary request and intent\nPort the thing.");
        assert!(
            !got.contains("deciding what matters"),
            "reasoning must be discarded, got: {got:?}"
        );
        assert!(!got.contains('<'), "tags must be unwrapped, got: {got:?}");
    }

    fn big_tool_result(tag: &str) -> Message {
        Message::user(format!(
            "<tool_result>{tag} {}</tool_result>",
            "x".repeat(MICROCOMPACT_MIN_BYTES)
        ))
    }

    #[test]
    fn microcompact_clears_only_old_large_results() {
        let mut t = vec![
            Message::user("do the thing"),
            big_tool_result("first"),
            Message::user("<tool_result>tiny</tool_result>"),
            big_tool_result("second"),
            big_tool_result("third"),
            Message::assistant("working on it"),
            big_tool_result("fourth"),
            big_tool_result("fifth"),
        ];
        let (cleared, bytes) = microcompact(&mut t);
        // Five large results; the newest three survive.
        assert_eq!(cleared, 2);
        assert!(bytes > 0, "reclaimed bytes reported");
        assert!(t[1].text.contains(MICROCOMPACT_STUB));
        assert!(t[3].text.contains(MICROCOMPACT_STUB));
        assert!(t[4].text.contains("third"));
        assert!(t[6].text.contains("fourth"));
        assert!(t[7].text.contains("fifth"));
        // Non-tool and tiny messages untouched.
        assert_eq!(t[0].text, "do the thing");
        assert_eq!(t[2].text, "<tool_result>tiny</tool_result>");
        assert_eq!(t[5].text, "working on it");
        // Idempotent: cleared stubs are small, nothing more to do.
        assert_eq!(microcompact(&mut t), (0, 0));
    }

    #[test]
    fn reclaimable_gates_the_opportunistic_pass() {
        // The M5 gate: `try_microcompact_opportunistic` fires only when the
        // pass would reclaim at least MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES.
        // Pruning rewrites transcript text in place and invalidates the KV
        // prefix from that point, so a pass that reclaims little costs more
        // than it saves. This pins the predicate that gate reads.

        // Nothing clearable: the newest three results are all there is.
        let short = vec![
            Message::user("do the thing"),
            big_tool_result("first"),
            big_tool_result("second"),
            big_tool_result("third"),
        ];
        assert_eq!(
            microcompact_reclaimable(&short),
            0,
            "the newest three are never candidates"
        );
        assert!(microcompact_reclaimable(&short) < MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES);

        // Two clearable results, each just over the small-result floor: worth
        // reclaiming in principle, but under the opportunistic threshold.
        let mut small = vec![Message::user("do the thing")];
        for tag in ["first", "second"] {
            small.push(big_tool_result(tag));
        }
        for tag in ["third", "fourth", "fifth"] {
            small.push(big_tool_result(tag));
        }
        let reclaimable = microcompact_reclaimable(&small);
        assert!(reclaimable > 0, "two old results are clearable");
        assert!(
            reclaimable < MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES,
            "{reclaimable} bytes must not trip the {MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES}-byte gate"
        );

        // The same shape with genuinely large results clears the gate.
        let mut large = vec![Message::user("do the thing")];
        for tag in ["first", "second"] {
            large.push(Message::user(format!(
                "<tool_result>{tag} {}</tool_result>",
                "x".repeat(MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES)
            )));
        }
        for tag in ["third", "fourth", "fifth"] {
            large.push(big_tool_result(tag));
        }
        assert!(
            microcompact_reclaimable(&large) >= MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES,
            "two large old results must trip the gate"
        );

        // The predicate agrees with what the pass actually clears.
        let mut t = large.clone();
        let before = microcompact_reclaimable(&t);
        let (cleared, bytes) = microcompact(&mut t);
        assert_eq!(cleared, 2);
        assert_eq!(
            bytes, before,
            "reclaimable must predict the bytes the pass reclaims"
        );
        assert_eq!(
            microcompact_reclaimable(&t),
            0,
            "nothing left to reclaim after the pass"
        );
    }

    #[test]
    fn results_after_a_task_list_injection_are_kept() {
        // A tool result that follows the last `# Task list` injection belongs
        // to the current task and survives microcompact even when it is old.
        let mut t = vec![
            Message::user("do the thing"),
            big_tool_result("first"),
            big_tool_result("second"),
            Message::user("# Task list\n\nYour current tasks"),
            big_tool_result("third"),
            big_tool_result("fourth"),
        ];
        let (cleared, _) = microcompact(&mut t);
        // "first" is old and not part of the current task -> cleared.
        assert_eq!(cleared, 1);
        assert!(t[1].text.contains(MICROCOMPACT_STUB));
        // "second" survives as one of the newest three.
        assert!(t[2].text.contains("second"));
        // "third"/"fourth" follow the injection -> kept as current-task work.
        assert!(t[4].text.contains("third"));
        assert!(t[5].text.contains("fourth"));
    }

    #[test]
    fn extract_summary_strips_analysis_and_unwraps() {
        let raw =
            "<analysis>thinking\nmore</analysis>\n<summary>\n1. Fix the bug\n</summary>\ntrailing";
        assert_eq!(extract_summary(raw), "1. Fix the bug");
        // Missing tags: falls back to the stripped text.
        assert_eq!(extract_summary("plain text"), "plain text");
        assert_eq!(extract_summary("<analysis>x</analysis> kept"), "kept");
        // Unclosed summary tag still unwraps to the end.
        assert_eq!(extract_summary("<summary>open ended"), "open ended");
    }

    #[test]
    fn reinjection_respects_budget_and_freshness() {
        let dir = std::env::temp_dir().join(format!("plank-reinject-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        let missing = dir.join("missing.txt");
        std::fs::write(&a, "alpha contents").unwrap();
        std::fs::write(&b, "beta contents").unwrap();
        let reads = vec![a.clone(), missing, b.clone()];

        // Ample budget: both files, newest (b) first, missing skipped.
        let out = build_reinjection(&reads, 10_000, &mut |s| i32::try_from(s.len()).unwrap_or(0))
            .unwrap();
        assert!(out.starts_with("<tool_result>"));
        assert!(out.ends_with("</tool_result>"));
        let (pa, pb) = (
            out.find("alpha contents").unwrap(),
            out.find("beta contents").unwrap(),
        );
        assert!(pb < pa, "newest read comes first");

        // Tight budget (exactly the newest file's section): only it fits.
        let section_b = format!("\n=== {} ===\nbeta contents\n", b.display());
        let budget = i32::try_from(section_b.len()).unwrap();
        let out = build_reinjection(&reads, budget, &mut |s| i32::try_from(s.len()).unwrap_or(0))
            .unwrap();
        assert!(out.contains("beta contents"));
        assert!(!out.contains("alpha contents"));

        // No budget: nothing to inject.
        assert!(
            build_reinjection(&reads, 0, &mut |s| i32::try_from(s.len()).unwrap_or(0)).is_none()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn first_index_is_the_earliest_body_microcompact_would_clear() {
        let big = "<tool_result>".to_owned() + &"x".repeat(600) + "</tool_result>";
        let mut t = Vec::new();
        t.push(Message::user("hello"));
        for _ in 0..6 {
            t.push(Message::assistant("ok"));
            t.push(Message::user(big.clone()));
        }
        // The newest MICROCOMPACT_KEEP_RESULTS results survive, so the earliest
        // cleared body is the first tool result in the transcript, at index 2.
        assert_eq!(microcompact_first_index(&t), Some(2));

        // Nothing clearable => nothing to report.
        let short = vec![Message::user("hi")];
        assert_eq!(microcompact_first_index(&short), None);
    }

    #[test]
    fn the_gate_weighs_bytes_reclaimed_against_tokens_reprefilled() {
        // A near-empty 1M window: pressure is nil, the floor is the strict one.
        let (size, used) = (1_000_000, 25_000);
        // The measured regression: ~12 KB reclaimed for ~14,000 tokens of prefill.
        assert!(!microcompact_is_worth_it(12_000, 14_000, size, used));
        // With a usable rung the remainder is small and the pass is clearly worth it.
        assert!(microcompact_is_worth_it(12_000, 500, size, used));
        // Nothing to re-prefill: always worth it.
        assert!(microcompact_is_worth_it(5_000, 0, size, used));
    }

    /// The exact numbers logged by the run-3 benchmark: 12,344 bytes against
    /// 10,674 tokens re-prefilled, in a 1M window ~2% full. Ratio 1.16.
    /// Reclaiming ~3,300 tokens of context for ~98 s of prefill is a bad trade
    /// when nothing is short, and must stay refused.
    #[test]
    fn the_measured_low_pressure_trade_stays_refused() {
        assert!(!microcompact_is_worth_it(12_344, 10_674, 1_000_000, 24_958));
    }

    /// The same trade, in a window under real pressure but still short of the
    /// full-compaction trigger: now the cheap relief pre-empts the expensive
    /// kind and must be accepted.
    #[test]
    fn the_same_trade_is_accepted_under_pressure_before_full_compaction() {
        let (size, used) = (30_000, 18_000);
        // Still short of full compaction: this is the pre-emption window.
        assert!(!should_compact(size, used));
        assert!(microcompact_is_worth_it(12_344, 10_674, size, used));
    }

    /// At or past the point full compaction fires, the KV is going to be
    /// rebuilt regardless, so the opportunistic pass must never be the blocker
    /// for any pass that reclaims a real amount of context. Only a degenerate
    /// pass — near-zero yield for a whole rebuild, which a full compaction is
    /// about to perform anyway — is still refused, by
    /// [`MICROCOMPACT_FLOOR_EPSILON`].
    #[test]
    fn at_the_full_compaction_threshold_the_gate_never_blocks() {
        let (size, used) = (30_000, 23_000);
        assert!(should_compact(size, used));
        // The smallest pass the opportunistic path will even attempt, against a
        // full-window re-prefill: accepted.
        assert!(microcompact_is_worth_it(
            MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES,
            size,
            size,
            used
        ));
        // A whole rebuild for one byte: still refused, and rightly.
        assert!(!microcompact_is_worth_it(1, 100_000, size, used));
    }

    /// Monotone in each input, in the obvious direction.
    #[test]
    fn the_gate_is_monotone_in_bytes_tokens_and_pressure() {
        let (size, used) = (30_000, 17_000);
        // More bytes reclaimed => more willing.
        assert!(!microcompact_is_worth_it(8_000, 10_000, size, used));
        assert!(microcompact_is_worth_it(20_000, 10_000, size, used));
        // More tokens re-prefilled => less willing.
        assert!(microcompact_is_worth_it(12_344, 4_000, size, used));
        assert!(!microcompact_is_worth_it(12_344, 40_000, size, used));
        // More pressure => more willing; the floor never rises with pressure.
        assert!(!microcompact_is_worth_it(12_344, 10_674, size, 5_000));
        assert!(microcompact_is_worth_it(12_344, 10_674, size, 20_000));
        let mut prev = f64::MAX;
        for used in (0..=30_000).step_by(250) {
            let f = microcompact_floor(size, used);
            assert!(f <= prev, "floor rose with pressure at used={used}");
            prev = f;
        }
    }

    /// The relaxed floor is anchored to the same threshold `should_compact`
    /// uses, so the two decisions cannot contradict each other.
    #[test]
    fn the_floor_reaches_zero_exactly_where_full_compaction_starts() {
        for size in [8_192, 30_000, 128_000, 1_000_000] {
            let trigger = compaction_trigger_used(size);
            assert!(should_compact(size, trigger), "size={size}");
            assert!(!should_compact(size, trigger - 1), "size={size}");
            // Bottoms out at the epsilon, never at zero: "accept anything"
            // would let a marginal pass fire immediately before a full
            // compaction throws the work away.
            let bottom = microcompact_floor(size, trigger);
            assert!(bottom > 0.0, "size={size}");
            assert!(
                (bottom - MICROCOMPACT_FLOOR_EPSILON).abs() < 1e-9,
                "size={size} bottom={bottom}"
            );
            assert!(
                microcompact_floor(size, trigger / 2) > MICROCOMPACT_FLOOR_EPSILON,
                "size={size}"
            );
        }
    }

    #[test]
    fn microcompact_exempts_image_bearing_tool_results() {
        // A large old tool result carrying an image must survive microcompact:
        // the image tokens cannot be stubbed, and clearing the text would
        // orphan them from their observation context.
        let big = "<tool_result>".to_owned() + &"x".repeat(600) + "</tool_result>";
        let mut t = vec![
            Message::user("do the thing"),
            Message::user(big.clone()).with_images(vec![crate::engine::VisionImage {
                path: "img.png".to_string(),
                embedding: crate::engine::VisionEmbedding {
                    data: vec![0.0; 4],
                    token_count: 1,
                    layout: 0,
                    grid_width: 1,
                    grid_height: 1,
                    width: 8,
                    height: 8,
                    content_width: 8,
                    content_height: 8,
                    fingerprint: [1; 32],
                },
            }]),
        ];
        for _ in 0..4 {
            t.push(Message::user(big.clone()));
        }
        let (cleared, _) = microcompact(&mut t);
        // The image-bearing result at index 1 is exempt; only the text-only
        // old results are candidates.
        assert!(
            !t[1].text.contains(MICROCOMPACT_STUB),
            "image-bearing result must not be cleared"
        );
        // At least one text-only old result was cleared.
        assert!(cleared > 0, "text-only old results should still be cleared");
    }
}

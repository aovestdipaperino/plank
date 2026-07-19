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
/// Maximum files re-injected after a full compaction.
pub const REINJECT_MAX_FILES: usize = 5;
/// Hard cap on the post-compaction re-injection budget, in tokens.
pub const REINJECT_CAP_TOKENS: i32 = 50_000;

/// Clears the bodies of old tool results in place, keeping the newest
/// [`MICROCOMPACT_KEEP_RESULTS`] intact; returns how many were cleared.
///
/// This is the cheap first step of compaction: no model round-trip, and the
/// conversation flow (user turns, assistant turns, tool-call structure) is
/// preserved — only bulky, stale tool output is dropped. Clearing an early
/// message invalidates the KV prefix from that point, but so would a full
/// compaction, and this one costs zero generated tokens.
pub fn microcompact(transcript: &mut [Message]) -> usize {
    let idx: Vec<usize> = transcript
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.role == Role::User
                && m.text.starts_with("<tool_result>")
                && m.text.len() > MICROCOMPACT_MIN_BYTES
        })
        .map(|(i, _)| i)
        .collect();
    let clear_upto = idx.len().saturating_sub(MICROCOMPACT_KEEP_RESULTS);
    for &i in &idx[..clear_upto] {
        transcript[i].text = format!("<tool_result>{MICROCOMPACT_STUB}</tool_result>");
    }
    clear_upto
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

/// Builds the private prompt used to ask the model for durable state.
///
/// Asks for a fixed-section summary wrapped in `<summary>` tags, with an
/// optional `<analysis>` scratch block that [`extract_summary`] strips. The
/// prompt explicitly forbids tool calls because the result is consumed
/// internally, not delivered as an assistant turn.
#[must_use]
pub fn make_prompt(reason: &str) -> String {
    let mut b = String::from(
        "Internal plank-agent context compaction request. This is not a user request.\n\
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
         Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion; prefer exact paths/ranges/commands that can reload the data.\n\
         After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n",
    );
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
        assert!(make_prompt("low context").contains("Compaction reason: low context"));
        assert!(!make_prompt("").contains("Compaction reason"));
    }

    #[test]
    fn prompt_asks_for_fixed_sections() {
        let p = make_prompt("");
        assert!(p.contains("1. Primary request and intent"));
        assert!(p.contains("8. Next step"));
        assert!(p.contains("<summary>"));
        assert!(p.contains("<analysis>"));
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
        let cleared = microcompact(&mut t);
        // Five large results; the newest three survive.
        assert_eq!(cleared, 2);
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
        assert_eq!(microcompact(&mut t), 0);
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
}

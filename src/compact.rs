//! Context compaction: summarize the transcript to reclaim context.
//!
//! Compaction asks the model for durable task state, then rebuilds the live
//! transcript as: system prompt + summary + recent verbatim tail. This keeps
//! the active context usable while avoiding unbounded transcript growth.
//! Port of the "Context Compaction" section of `ds4_agent.c`, adapted from
//! token transcripts to text messages.

/// Compact once used context reaches this percentage of the window.
pub const COMPACT_SOFT_PERCENT: i32 = 85;
/// Also compact when fewer than this many tokens remain free.
pub const COMPACT_MIN_FREE_TOKENS: i32 = 8192;
/// The verbatim tail keeps at most `ctx / TAIL_DIVISOR` tokens.
pub const COMPACT_TAIL_DIVISOR: i32 = 8;
/// Hard cap on the verbatim tail, in tokens.
pub const COMPACT_TAIL_CAP_TOKENS: i32 = 8192;

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
/// The prompt explicitly forbids tool calls because the result is consumed
/// internally, not delivered as an assistant turn.
#[must_use]
pub fn make_prompt(reason: &str) -> String {
    let mut b = String::from(
        "Internal plank-agent context compaction request. This is not a user request.\n\
         Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n\
         - user goals, constraints, and preferences\n\
         - files inspected or edited\n\
         - commands run and important results\n\
         - decisions, rejected approaches, known bugs, and pending next steps\n\
         - reloadable bulky data with exact paths/ranges/commands when available\n\n\
         Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n\
         After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n\
         Output only the compact summary.\n",
    );
    if !reason.is_empty() {
        b.push_str("\nCompaction reason: ");
        b.push_str(reason);
        b.push('\n');
    }
    b
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
}

//! The `/goal` autonomous loop: state, prompt texts, and verdict parsing.
//!
//! Pure logic only — no engine, no I/O — so the whole state machine unit-tests
//! without a model. `ui.rs` owns the two front-end integrations (the plain REPL
//! loop and the `tui_turn` continuation hook).
//!
//! Nothing here touches the system prompt or the DSML tool syntax: the goal and
//! adjudication texts are ordinary *user* messages, which is what keeps
//! `tests/c_parity.rs` unaffected.
//!
//! Context cost: every iteration appends two messages to the transcript (the
//! adjudication prompt and its reply), so the default cap of 20 adds up to 40
//! beyond the goal's own work. An adjudication pass never calls `maybe_compact`
//! itself; the growth is bounded only in that `run_turn`/`worker_turn` compact
//! at the next turn boundary.

/// Iterations a goal runs before settling as [`Outcome::Cap`], absent `--max`.
pub const DEFAULT_MAX_ITERS: usize = 20;

/// Printed for `/goal` with no argument, and for any malformed argument.
pub const USAGE: &str = "usage: /goal [--max <n>] <objective>";

/// Appended as a user message after every goal iteration. Deliberately narrow:
/// the parser only trusts the two marker lines, and anything else degrades to
/// [`Verdict::Continue`], which the iteration cap then bounds.
pub const ADJUDICATION_PROMPT: &str = concat!(
    "Assess the goal now. Do not use any tools and do not do any further work ",
    "in this reply — answer with exactly these two lines and nothing else:\n",
    "GOAL_VERDICT: ATTAINED|UNATTAINABLE|NEEDS_USER|CONTINUE\n",
    "GOAL_REASON: <one line>\n",
    "\n",
    "ATTAINED — the goal is met and verified. UNATTAINABLE — it cannot be met, ",
    "and the reason says why. NEEDS_USER — progress is blocked on a decision or ",
    "information only the user has. CONTINUE — more work remains and you know ",
    "what to do next; the reason says what."
);

/// What the model said about the goal after an iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Attained,
    Unattainable,
    NeedsUser,
    Continue,
}

/// A parsed adjudication reply: the verdict plus its one-line reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjudication {
    pub verdict: Verdict,
    pub reason: String,
}

impl Adjudication {
    /// The safe default: keep going. Used when the reply is unparseable or the
    /// model answered with work instead of a verdict.
    #[must_use]
    pub fn keep_going() -> Self {
        Self {
            verdict: Verdict::Continue,
            reason: String::new(),
        }
    }
}

/// How a goal loop ended. [`Verdict::Continue`] has no outcome — it is the one
/// verdict that does not end the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Attained,
    Unattainable,
    NeedsUser,
    /// Ran out of iterations while still returning `CONTINUE`.
    Cap,
    /// The user interrupted (Esc in the TUI, Ctrl+C in the plain REPL).
    Interrupted,
}

impl Outcome {
    /// The terminal outcome for a verdict, or `None` for `CONTINUE`.
    #[must_use]
    pub fn from_verdict(v: Verdict) -> Option<Self> {
        match v {
            Verdict::Attained => Some(Self::Attained),
            Verdict::Unattainable => Some(Self::Unattainable),
            Verdict::NeedsUser => Some(Self::NeedsUser),
            Verdict::Continue => None,
        }
    }

    fn phrase(self) -> &'static str {
        match self {
            Self::Attained => "attained",
            Self::Unattainable => "proved unattainable",
            Self::NeedsUser => "needs your input",
            Self::Cap => "hit the iteration cap",
            Self::Interrupted => "interrupted",
        }
    }
}

/// Durable goal state, persisted on the session (M7) so `/resume` and
/// compaction know what the session is for. The field is the durable *fact*;
/// the transcript entry (the kickoff message) is the record of what was shown.
/// Per the log-everything invariant (M3), the field must never become a second
/// unlogged source of model-visible text — anything it renders is also pushed
/// into the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalState {
    /// The objective text, as the user typed it.
    pub objective: String,
    /// Iteration cap (`--max <n>`, default [`DEFAULT_MAX_ITERS`]).
    pub max_iters: usize,
    /// Iterations started so far; 0 before the first one.
    pub iter: usize,
    /// Lifecycle status.
    pub status: GoalStatus,
}

/// Lifecycle status of a durable goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// The goal is live and being worked.
    Active,
    /// The loop settled (attained, unattainable, needs user, cap, interrupted).
    Done,
    /// `/goal clear` ended it without settling.
    Cleared,
}

impl GoalStatus {
    /// The record tag written to the session file.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            GoalStatus::Active => "active",
            GoalStatus::Done => "done",
            GoalStatus::Cleared => "cleared",
        }
    }

    /// Parses a record tag; `None` for anything unrecognised.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "active" => Some(Self::Active),
            "done" => Some(Self::Done),
            "cleared" => Some(Self::Cleared),
            _ => None,
        }
    }
}

/// Live state of one `/goal` run. Transient: it lives on the `Agent` only while
/// the loop runs and is dropped before control returns to the prompt.
#[derive(Debug, Clone)]
pub struct GoalLoop {
    goal: String,
    max_iters: usize,
    iter: usize,
}

impl GoalLoop {
    #[must_use]
    pub fn new(goal: &str, max_iters: usize) -> Self {
        Self {
            goal: goal.to_owned(),
            max_iters,
            iter: 0,
        }
    }

    #[must_use]
    pub fn goal(&self) -> &str {
        &self.goal
    }

    #[must_use]
    pub fn max_iters(&self) -> usize {
        self.max_iters
    }

    /// Iterations started so far; 0 before the first one.
    #[must_use]
    pub fn iters_done(&self) -> usize {
        self.iter
    }

    /// Starts the next iteration and returns its 1-based number.
    pub fn next_iteration(&mut self) -> usize {
        self.iter += 1;
        self.iter
    }

    /// True once the iterations started have reached the cap.
    #[must_use]
    pub fn at_cap(&self) -> bool {
        self.iter >= self.max_iters
    }
}

/// Splits `/goal`'s argument into the objective and the iteration cap.
///
/// Accepts `<objective>` or `--max <n> <objective>`. Every malformed form
/// returns [`USAGE`] as the error, so callers have one message to print.
///
/// # Errors
/// Returns [`USAGE`] if the objective is empty or `--max` is malformed.
pub fn parse_command(arg: &str) -> Result<(String, usize), String> {
    let arg = arg.trim();
    let (max, rest) = match arg.strip_prefix("--max") {
        Some(rest) => {
            let rest = rest.trim_start();
            let mut it = rest.splitn(2, char::is_whitespace);
            let n: usize = it
                .next()
                .unwrap_or("")
                .parse()
                .map_err(|_| USAGE.to_owned())?;
            if n == 0 {
                return Err(USAGE.to_owned());
            }
            (n, it.next().unwrap_or("").trim())
        }
        None => (DEFAULT_MAX_ITERS, arg),
    };
    if rest.is_empty() {
        return Err(USAGE.to_owned());
    }
    Ok((rest.to_owned(), max))
}

/// The user message that opens a goal loop.
#[must_use]
pub fn kickoff_message(goal: &str) -> String {
    format!(
        "Work towards this goal:\n\n{goal}\n\nI will keep handing control back \
to you until the goal is settled, so do not stop to ask whether to proceed — \
take the next concrete step each turn, and verify your work before calling it \
done."
    )
}

/// Finds `tag` on `line` and returns the text after it, but only if what
/// precedes the tag on that line is empty/whitespace or ends with
/// `</think>`.
///
/// The allowance exists for one observed shape: a thinking model closes its
/// `</think>` tag on the same line as the marker (`</think>GOAL_VERDICT:
/// ATTAINED`), with no whitespace in between. Matching only at the trimmed
/// start of a line (the original rule) misses that case entirely, which is
/// the defect this function fixes. The allowance is the literal closing tag
/// rather than any `>`, so a Markdown blockquote (`> GOAL_VERDICT: …`) — a
/// model quoting the protocol back at us — is not read as a real verdict.
fn marker_suffix<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let idx = line.find(tag)?;
    let prefix = line[..idx].trim();
    if prefix.is_empty() || prefix.ends_with("</think>") {
        Some(line[idx + tag.len()..].trim())
    } else {
        None
    }
}

/// Parses an adjudication reply.
///
/// The **last** `GOAL_VERDICT:` line wins, so a reply that quotes the marker
/// while explaining itself cannot decide the loop from its preamble. Anything
/// unrecognized degrades to [`Verdict::Continue`]: turning a parse miss into a
/// premature `ATTAINED` would be the worse failure, and the cap bounds the
/// alternative.
///
/// The marker no longer has to open the line: it may follow a closing tag
/// like `</think>` with no separating whitespace (see [`marker_suffix`]).
/// This is deliberately narrower than "match the tag anywhere in the line" —
/// that laxer rule would let a reply that merely *quotes* the marker in
/// prose (e.g. "the format is `GOAL_VERDICT: ATTAINED`") be read as a real
/// verdict whenever it happened to be the last such line, with no preamble
/// after it to override it. Requiring the preceding text to be blank or end
/// in `</think>` rules out ordinary prose — and Markdown blockquotes, which
/// a laxer "ends with `>`" rule would have admitted — while still allowing
/// the tag-adjacent case that motivated this change.
///
/// The scan is per line and takes the *first* occurrence of the marker on it
/// ([`str::find`]), so a line whose first occurrence is prose-quoted and whose
/// second is genuine is rejected as prose and degrades to [`Verdict::Continue`].
/// That is the safe direction — a missed verdict costs an iteration, which the
/// cap bounds; a false `ATTAINED` ends the goal wrongly.
#[must_use]
pub fn parse_verdict(text: &str) -> Adjudication {
    let last = |tag: &str| {
        text.lines()
            .filter_map(|l| marker_suffix(l, tag))
            .next_back()
    };
    let verdict = match last("GOAL_VERDICT:")
        .map(str::to_ascii_uppercase)
        .as_deref()
    {
        Some("ATTAINED") => Verdict::Attained,
        Some("UNATTAINABLE") => Verdict::Unattainable,
        Some("NEEDS_USER") => Verdict::NeedsUser,
        _ => Verdict::Continue,
    };
    Adjudication {
        verdict,
        reason: last("GOAL_REASON:").unwrap_or_default().to_owned(),
    }
}

/// Dim notice printed before each iteration.
#[must_use]
pub fn banner(iter: usize, max: usize) -> String {
    format!("[goal {iter}/{max}]")
}

/// Dim notice printed once, when the loop ends.
#[must_use]
pub fn closing(outcome: Outcome, iters: usize, reason: &str) -> String {
    let plural = if iters == 1 {
        "iteration"
    } else {
        "iterations"
    };
    let phrase = outcome.phrase();
    if reason.is_empty() {
        format!("[goal {phrase} after {iters} {plural}]")
    } else {
        format!("[goal {phrase} after {iters} {plural} — {reason}]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_command_takes_a_bare_objective() {
        let (goal, max) = parse_command("make the parser tests pass").expect("parses");
        assert_eq!(goal, "make the parser tests pass");
        assert_eq!(max, DEFAULT_MAX_ITERS);
    }

    #[test]
    fn parse_command_takes_an_explicit_cap() {
        let (goal, max) = parse_command("--max 5 ship the thing").expect("parses");
        assert_eq!(goal, "ship the thing");
        assert_eq!(max, 5);
    }

    #[test]
    fn parse_command_rejects_a_bad_cap() {
        assert_eq!(parse_command("--max zero do it"), Err(USAGE.to_owned()));
        assert_eq!(parse_command("--max 0 do it"), Err(USAGE.to_owned()));
        assert_eq!(parse_command("--max 5"), Err(USAGE.to_owned()));
    }

    #[test]
    fn parse_command_rejects_an_empty_objective() {
        assert_eq!(parse_command(""), Err(USAGE.to_owned()));
        assert_eq!(parse_command("   "), Err(USAGE.to_owned()));
    }

    #[test]
    fn parse_verdict_reads_each_token() {
        for (tok, want) in [
            ("ATTAINED", Verdict::Attained),
            ("UNATTAINABLE", Verdict::Unattainable),
            ("NEEDS_USER", Verdict::NeedsUser),
            ("CONTINUE", Verdict::Continue),
        ] {
            let text = format!("GOAL_VERDICT: {tok}\nGOAL_REASON: because\n");
            let adj = parse_verdict(&text);
            assert_eq!(adj.verdict, want, "token {tok}");
            assert_eq!(adj.reason, "because", "token {tok}");
        }
    }

    #[test]
    fn parse_verdict_lets_the_last_marker_win() {
        let text = "I could answer `GOAL_VERDICT: ATTAINED` but I am not done.\n\
                    GOAL_VERDICT: CONTINUE\n\
                    GOAL_REASON: two tests still fail\n";
        let adj = parse_verdict(text);
        assert_eq!(adj.verdict, Verdict::Continue);
        assert_eq!(adj.reason, "two tests still fail");
    }

    #[test]
    fn parse_verdict_reads_the_marker_sharing_a_line_with_a_closing_think_tag() {
        let text = "</think>GOAL_VERDICT: ATTAINED\nGOAL_REASON: the tests pass\n";
        let adj = parse_verdict(text);
        assert_eq!(adj.verdict, Verdict::Attained);
        assert_eq!(adj.reason, "the tests pass");
    }

    #[test]
    fn parse_verdict_ignores_a_marker_inside_a_blockquote() {
        // A model quoting the protocol back at us in Markdown is not a verdict.
        let text = "> GOAL_VERDICT: ATTAINED\n> GOAL_REASON: quoted spec\n";
        let adj = parse_verdict(text);
        assert_eq!(adj.verdict, Verdict::Continue);
        assert_eq!(adj.reason, "");
    }

    #[test]
    fn parse_verdict_ignores_a_marker_quoted_mid_prose() {
        let text = "The format is `x GOAL_VERDICT: ATTAINED` as documented.\n";
        assert_eq!(parse_verdict(text).verdict, Verdict::Continue);
    }

    #[test]
    fn parse_verdict_falls_back_to_continue() {
        assert_eq!(parse_verdict("no marker here").verdict, Verdict::Continue);
        assert_eq!(
            parse_verdict("GOAL_VERDICT: MAYBE\n").verdict,
            Verdict::Continue
        );
        assert_eq!(parse_verdict("no marker here").reason, "");
    }

    #[test]
    fn parse_verdict_tolerates_a_missing_reason() {
        let adj = parse_verdict("GOAL_VERDICT: ATTAINED\n");
        assert_eq!(adj.verdict, Verdict::Attained);
        assert_eq!(adj.reason, "");
    }

    #[test]
    fn goal_loop_caps_after_max_iterations() {
        let mut g = GoalLoop::new("do it", 2);
        assert_eq!(g.iters_done(), 0);
        assert!(!g.at_cap());
        assert_eq!(g.next_iteration(), 1);
        assert!(!g.at_cap());
        assert_eq!(g.next_iteration(), 2);
        assert!(g.at_cap());
        assert_eq!(g.goal(), "do it");
        assert_eq!(g.max_iters(), 2);
    }

    #[test]
    fn outcome_maps_every_terminal_verdict() {
        assert_eq!(Outcome::from_verdict(Verdict::Continue), None);
        assert_eq!(
            Outcome::from_verdict(Verdict::Attained),
            Some(Outcome::Attained)
        );
        assert_eq!(
            Outcome::from_verdict(Verdict::Unattainable),
            Some(Outcome::Unattainable)
        );
        assert_eq!(
            Outcome::from_verdict(Verdict::NeedsUser),
            Some(Outcome::NeedsUser)
        );
    }

    #[test]
    fn banner_and_closing_read_as_notices() {
        assert_eq!(banner(3, 20), "[goal 3/20]");
        assert_eq!(
            closing(Outcome::Attained, 4, "all tests pass"),
            "[goal attained after 4 iterations — all tests pass]"
        );
        assert_eq!(
            closing(Outcome::Attained, 1, ""),
            "[goal attained after 1 iteration]"
        );
        assert_eq!(
            closing(Outcome::Cap, 20, "still failing"),
            "[goal hit the iteration cap after 20 iterations — still failing]"
        );
    }

    #[test]
    fn kickoff_message_carries_the_objective() {
        let m = kickoff_message("fix the flaky test");
        assert!(m.contains("fix the flaky test"), "{m}");
    }
}

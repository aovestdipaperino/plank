---
name: code-review
description: Use when asked to review a diff, branch, PR or file in this repository - dispatches a reviewer sub-agent briefed on plank's own invariants and acts on its Critical/Important/Minor findings
argument-hint: [diff | branch | file | PR number]
---

# Code Review (plank)

Review the work described by `$ARGUMENTS`. If that is empty, review the
uncommitted diff plus anything on this branch that is not on `main`.

**Review mode:** If the user says "as-is" or "current state", review the
snapshot as-is without comparing to a previous commit/branch. Read the
current files directly instead of using `git diff`.

**Core principle:** every finding names a file:line, says what breaks, and
says why it matters. A review with no file:line references is not a review.

## Dispatch a Reviewer Sub-Agent

Call the `agent` tool: `task` is the review brief, and the optional `name`
selects one of the configured sub-agents from the session roster (an unknown
name silently runs a general-purpose one and says so in the report). The diff
and the reading live in that sub-agent's context; only its report comes back
to you, which is the point - you stay the coordinator.

Hand it precisely crafted context, never this session's history: what was
built, the base and head SHAs, and the invariant list below. Its final report
is what you get, so ask for the output format explicitly.

**Sub-agent constraints:**

- **Stop early with findings, not a full review:** A sub-agent should report as soon as it has a handful of real findings (3-5). Don't wait to read every file or write a complete synthesis.
- **Emit items immediately:** Each finding goes to the user after `</think>`, one at a time. No accumulating items in reasoning.
- **No drafting in reasoning:** If reasoning contains numbered deliverables or fenced code, stop and emit what you have.
- **Use fanout for large diffs:** For diffs >300 lines, split by directory and use multiple sub-agents instead of one exhaustive reviewer.

For a diff too large for one brief, split it by area and issue several `agent`
calls. `fanout` takes the same subtasks as one JSON array, but it runs them
serially (and can be turned off with `tools.fanout`), so it buys a
deterministic join rather than speed.

Two cases where you review directly instead: you are already a sub-agent
(nesting is capped and will refuse), or the sub-agent came back with no
report. Say which case applies rather than going quiet.

Act on what comes back: fix Critical immediately, fix Important before
proceeding, note Minor. Push back with technical reasoning if a finding is
wrong - a reviewer that misread the code is not an instruction.

## Scope the Diff

    git status --short
    git diff                      # uncommitted
    git diff main...HEAD          # the branch's own work
    git diff --stat main...HEAD   # size first, so you know how many passes

For a PR number use `gh pr diff <n>`.

**As-is review:** If the user wants to assess the current state without
comparison, skip `git diff` and read the current files directly. Focus on
code quality, style, and correctness rather than change impact.

Read the *files* a hunk sits in, not
only the hunk: what usually goes wrong here is an invariant the change stepped
on somewhere off-screen, rather than anything visibly wrong in the added lines.
Say so in the brief, or the reviewer will read the diff and miss precisely
those.

## plank's Invariants

These are the mistakes this codebase actually makes. Check every one the diff
touches, and say explicitly which ones you checked.

**C parity.** Wire formats and prompt text answer to `refs/ds4`, not to what
reads better: the framing around tool output, the DSML call syntax and the
system prompt have to come out as the exact bytes the model was trained on, so
`tests/c_parity.rs` still has to be green. Watch for `\`-continued Rust string
literals anywhere the model will read the result - the continuation eats the
next line's indentation, so the bytes change without the diff looking like it
changed them.

**Write containment.** Every write path goes through
`ToolContext::resolve_for_write`. It is one function on purpose: the Seatbelt
profile is generated from the same `Sandbox::write_roots`, so a tool that
assembles its own destination escapes the runtime check and the sandbox in one
move. Reading is a different matter and is left open by design, so an
uncontained read is not a finding.

**KV ladder.** Look a rung up under the fingerprint of the transcript cut back
to the depth that rung recorded - never the fingerprint of the transcript as it
stands now. Beyond that: `discard_ladder` on rollback, `KvLadder::truncate_to`
on fork and `end_subagent_fork`, no stored payload and no new rungs while
`in_sidechain()`, and live rungs held by the GC keep set.

**Staged artifact swap.** `downloader::swap_staged` has to move the staged
`ds4.manifest` after everything else, because a landed manifest is what tells
the next launch the rest of the set landed too.

**Two front ends.** Slash commands are implemented twice in `ui.rs`, once for
the plain stdout REPL and once for the Ratatui TUI. Touching only one of them
is Important; if the command opens a pane, the plain path needs a text-only
equivalent as well.

**Background jobs.** Nothing but the `BashJobs` table decides what has been
reported: observing a job as done drops it from the table, which is exactly
why `take_finished` can never announce the same job twice. A notification
enters the transcript at a turn boundary and nowhere else.

**Fingerprint churn.** Editing model-facing tool descriptions or shell rules
moves `fp1`, which throws away the system-prompt KV snapshot. That is a fine
thing to do on purpose, and a surprising thing to do by accident - record it in
`docs/SYSTEM-PROMPT-OVERRIDES.md`.

**Sessions and settings.** A transcript's identity is its filename stem, and
`validate_name` will take nothing but ASCII alphanumerics and `-`. Four
sections are off-limits to a plugin's `settings.json` - `engine.*`,
`worktree.*`, `tools.*` and `pluginConfig` (`PLUGIN_REFUSED_SECTIONS`).

## Reuse (simplify pass 1)

- Search for an existing helper before accepting a newly written one. Flag any
  new function that duplicates functionality that already exists, and name the
  one to call instead.
- Flag inline logic that an existing utility already covers: hand-rolled path
  handling, ad-hoc env lookups, custom string munging, re-implemented glob or
  fingerprint logic.

## Quality (simplify pass 2)

- **Redundant state:** state duplicating existing state, cached values that
  could be derived, a channel or flag where a direct call would do.
- **Parameter sprawl:** a new parameter bolted onto a function instead of
  restructuring it.
- **Copy-paste with variation:** near-duplicate blocks that want one shared
  function - especially across the REPL and TUI paths.
- **Leaky abstractions:** reaching past `Engine`, `RenderSink` or
  `ToolContext` instead of through them.
- **Stringly-typed code:** raw strings where a constant or enum already exists.
- **Unnecessary comments:** comments restating what the code says, narrating
  the change, or naming the caller - delete. Keep only non-obvious WHY:
  hidden constraints, invariants, workarounds. This codebase's comments carry
  rationale; a new comment that carries none is noise.

## Efficiency (simplify pass 3)

- Redundant work: repeated file reads, recomputed fingerprints, re-prefilled
  prefixes that a rung already covers.
- Hot-path bloat: new blocking work in the per-token render path, the TUI tick,
  or startup.
- Recurring no-op updates: status-bar or store writes that fire every tick
  regardless of change - add a change guard.
- TOCTOU existence checks: operate and handle the error instead of pre-checking.
- Memory: unbounded buffers, missing cleanup, a transcript or log that only
  grows.

## General Checks

- **Correctness:** edge cases, error paths, `unwrap`/`expect` on input that can
  legitimately be absent, off-by-one in slicing and truncation.
- **Tests:** do they verify real behavior, or restate the implementation?
  Native-engine code must be `#[cfg(ds4_engine)]`-gated so the EchoEngine path
  still builds and tests.
- **Docs:** a new invariant belongs in `FINDINGS.md`; anything about the model
  looping goes in `docs/LOOP-FINDINGS.md` instead.

## Verify Before Reporting

Claims need evidence. The brief must tell the reviewer to run these and quote
what they actually printed:

    cargo test --lib
    cargo clippy --workspace --all-targets -- -D warnings

A clean local run proves less than it looks: clippy lints only what it
recompiled, so warnings in files the build skipped stay hidden. A report that says "tests pass" without quoting
output has not verified anything - run them yourself before acting on it.

## Output Format

### Strengths
Specific and accurate, or omit the section.

### Issues
Critical (must fix) / Important (should fix) / Minor (nice to have). For each:
`file.rs:line` - what is wrong - why it matters - how to fix if not obvious.

### Verification
The commands run and their real output.

### Assessment
**Ready to merge?** Yes | No | With fixes, plus one or two sentences.

## Loop Prevention for Large Tasks

When the review task is large or open-ended, use these patterns to avoid loops:

- **Emit deliverables immediately:** For list-shaped answers (reviews, audits, surveys), emit each item after `</think>` as soon as it's decided. Don't accumulate items in reasoning and write them at the end.
- **Draft in code, not reasoning:** Use the `edit` tool for concrete code. If reasoning contains numbered deliverables or fenced code blocks, stop and emit what you have—never continue drafting inside `<think>`.
- **Break large tasks:** For large diffs (>300 lines), split the review by directory or module and use `fanout` with multiple reviewer sub-agents.
- **Watch for intent loops:** If the reasoning repeatedly states "Let me check X" or "I will verify Y" without actually calling the tool, the model is stuck deliberating. Force a decision with `ask` on user-visible choices.

## Red Flags

| Thought | Reality |
|---------|---------|
| "I'll write the whole review in reasoning first" | A 30 KB review drafted in `<think>` and never emitted loses all work on interrupt. Emit items one at a time. |
| "I'll just read the diff inline" | That burns the context you need to keep driving the work. Dispatch a reviewer sub-agent; keep the findings, not the diff. |
| "I'll check the docs / build / tests to be thorough" | Without explicit tool calls, deliberation loops produce no edits. `ask` on user-visible design choices, not internal ones. |
| "It's a small diff, it's fine" | Small diffs break invariants too. Check the list. |
| "Tests probably pass" | Run them. Evidence before assertions, always. |
| "Improve error handling" | Vague. Name the file, the line, and the failure. |
| "This nit is Critical" | Calibrate. Miscategorized nits get the whole review ignored. |
| "The read path isn't contained!" | Reads are uncontained on purpose. Not a finding. |

---
name: code-review
description: Use when asked to review a diff, branch, PR or file in this repository - dispatches a reviewer sub-agent briefed on plank's own invariants and acts on its Critical/Important/Minor findings
argument-hint: [diff | branch | file | PR number]
---

# Code Review (plank)

Review the work described by `$ARGUMENTS`. If that is empty, review the
uncommitted diff plus anything on this branch that is not on `main`.

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

For a PR number use `gh pr diff <n>`. Read the *files* around a hunk, not just
the hunk: plank's bugs usually live in the invariant a hunk quietly broke, not
in the lines it added. Put that instruction in the brief - a reviewer that
reads only the diff misses exactly these.

## plank's Invariants

These are the mistakes this codebase actually makes. Check every one the diff
touches, and say explicitly which ones you checked.

**C parity.** `refs/ds4` is the source of truth for wire formats and prompt
text: tool-output framing, DSML tool-call syntax and the system prompt must
stay byte-for-byte identical, because that is what the model was trained on.
`tests/c_parity.rs` must still pass. Flag any `\`-continued Rust string
literal in model-facing text: the continuation strips the next line's leading
whitespace and silently alters the bytes.

**Write containment.** Any tool that writes a path must resolve it through
`ToolContext::resolve_for_write`. That is the single choke point the Seatbelt
profile shares via `Sandbox::write_roots`; a tool that builds its own path
bypasses both. Reads are deliberately uncontained - not a finding.

**KV ladder.** A rung is looked up under the fingerprint of the transcript
*truncated to the rung's own recorded depth*, never the full current
transcript. Rollback must `discard_ladder`; fork and `end_subagent_fork` must
`KvLadder::truncate_to`; sidechains (`in_sidechain()`) must neither store the
payload nor push rungs; the GC keep set must include live rungs.

**Staged artifact swap.** In `downloader::swap_staged` the staged
`ds4.manifest` moves last - its presence is the proof the whole set landed.

**Two front ends.** `ui.rs` handles slash commands on two parallel paths, the
plain stdout REPL and the Ratatui TUI. A one-sided change is Important; a
pane-based command also needs a static text equivalent on the plain path.

**Background jobs.** The `BashJobs` table is the sole source of truth: a job
the model observed done is removed at that observation, so `take_finished`
announces each job at most once. Notifications join the transcript only at a
turn boundary.

**Fingerprint churn.** A change to model-facing tool descriptions or shell
rules churns `fp1` and invalidates the system-prompt KV snapshot. That is
allowed but must be deliberate and documented in
`docs/SYSTEM-PROMPT-OVERRIDES.md`.

**Sessions and settings.** Transcript ids are the filename stem; `validate_name`
accepts only ASCII alphanumerics and `-`. A plugin `settings.json` can never
set `engine.*`, `worktree.*`, `tools.*` or `pluginConfig`
(`PLUGIN_REFUSED_SECTIONS`).

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

Clippy only re-lints crates it recompiles, so a cached clean run can miss
warnings in untouched files. A report that says "tests pass" without quoting
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

## Red Flags

| Thought | Reality |
|---------|---------|
| "I'll just read the diff inline" | That burns the context you need to keep driving the work. Dispatch a reviewer; keep the findings, not the diff. |
| "It's a small diff, it's fine" | Small diffs break invariants too. Check the list. |
| "Tests probably pass" | Run them. Evidence before assertions, always. |
| "Improve error handling" | Vague. Name the file, the line, and the failure. |
| "This nit is Critical" | Calibrate. Miscategorized nits get the whole review ignored. |
| "The read path isn't contained!" | Reads are uncontained on purpose. Not a finding. |

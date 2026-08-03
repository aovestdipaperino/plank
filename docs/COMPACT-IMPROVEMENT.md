# Compaction: gap analysis and improvement plan

A comparison of plank's context compaction against the reference agent's
`/compact`, and what is worth changing as a result.

The reference behavior is documented in `local/COMPACT-COMMAND.md` (an analysis
of the TypeScript agent, untracked). This document is the plank-side conclusion:
what already matches, where the two designs deliberately diverge, and an ordered
list of the work.

plank's implementation lives in `src/compact.rs` (pure policy and text
handling) plus these `Agent` methods in `src/ui.rs`:

| Method | Role |
| --- | --- |
| `try_microcompact` | cheap pass: clear old tool-result bodies |
| `rebuild_after_compact` | replace the transcript with summary + tail + re-injection + tasks |
| `compact` | plain-REPL orchestration (prints to stdout) |
| `maybe_compact_notify` / `do_compact_notify` | TUI orchestration (notes to a `CompactSink`) |
| `compact_trigger`, `fire_pre_compact`, `fire_post_compact` | hook dispatch shared by both orchestrators |

## What already matches

- **Microcompact before summarizing.** Both agents clear stale tool-result
  bodies first and only fall back to a model round-trip if that is not enough.
  plank's `try_microcompact` returns `None` when the cheap pass was
  insufficient, but the mutation it made is *kept*, so the summarization pass
  sees the already-shrunken transcript. That matches the reference's ordering.
- **The `<analysis>` / `<summary>` contract.** Same scratchpad-then-summary
  shape, same discard of the scratchpad (`extract_summary`), and the same
  fallback of using the stripped text when the model ignores the tags.
- **Budgeted post-compact file re-injection.** Both cap at 5 files and 50K
  tokens (`REINJECT_MAX_FILES` / `REINJECT_CAP_TOKENS` against the reference's
  `POST_COMPACT_MAX_FILES_TO_RESTORE` / `POST_COMPACT_TOKEN_BUDGET`).
- **Post-compact message order.** summary, then the verbatim tail, then
  re-injected attachments, then hook/task material. The reference's
  `buildPostCompactMessages` produces the same sequence.
- **Compaction never starts a model turn.** The reference returns
  `shouldQuery: false`; plank compacts inline at the head of a turn (or from
  `/compact`) and returns to the prompt.

## Where plank is ahead: KV reuse is free

The reference spends a whole subsystem on making the summarization request share
the main conversation's prompt cache: `getCacheSharingParams` rebuilds the exact
system prompt, `userContext` and `systemContext`, `streamCompactSummary` runs the
request through `runForkedAgent` with byte-identical cache-key params, and there
is a documented trap where setting `maxOutputTokens` clamps `budget_tokens`,
breaks the thinking-config match, and silently invalidates the cache.

plank needs none of it. The compaction prompt is
`Prompt::Flat(render_transcript(...) + make_prompt(...))`, a literal extension of
the live prefix, and `Ds4Engine` holds one session across turns, so only the
~1.5KB compaction prompt is prefilled. A local engine with a persistent KV makes
the reference's hardest piece of compaction machinery unnecessary.

**Keep it that way.** Any change to how the compaction prompt is assembled must
preserve "the live transcript bytes, then an appended suffix". Reordering or
re-rendering the prefix turns a 1.5KB prefill into a full-context one.

## The deliberate divergence: destructive rebuild vs. read-time projection

The reference deletes nothing. It appends a compact boundary marker, and every
consumer slices the message array via `getMessagesAfterCompactBoundary()`, so
pre-compaction history remains in the transcript, visible under `ctrl+o` and
`--verbose`, and relinkable on `--resume`.

plank does `session.transcript = Vec::new()` followed by `clear_branches()`.

| | plank | reference |
| --- | --- | --- |
| Pre-compaction messages | destroyed; the session file is overwritten | retained, hidden at read time |
| Off-path branches | dropped (issue #65) | survive |
| Transcript growth | bounded | unbounded per session |
| Post-compact size | structurally bounded: summary + tail (≤ ctx/8) + re-injection (≤ ctx/8) + tasks | depends on what the boundary slice yields |

This cuts both ways, and plank's choice buys something the reference has to patch
around. Because the rebuild lands near ctx/4 by construction, plank **cannot
enter a compaction loop**: the reference needs
`MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3` as a circuit breaker precisely
because its projection offers no such guarantee. plank does not need that
breaker, and should not grow one.

What plank pays is auditability and branch survival. For a tool that ships
`/tree` and `/fork`, silently discarding off-path branches at every compaction is
the sharper cost, and it is the one place where adopting the boundary-marker
model would be a real (if large) improvement rather than a stylistic one. It is
out of scope for the work below: it touches the session format, `/resume`,
`/strip`, the branch tree, and context accounting all at once.

## Defects

### 1. Compaction hooks fire on only one of the two front-ends — **fixed**

`PreCompact` and `PostCompact` were dispatched inside `Agent::compact()`, which
serves the **plain REPL only**. `do_compact_notify` serves both TUI paths
(threshold-driven compaction inside a turn, and manual `/compact`) and fired
neither.

The TUI is the default front-end whenever both ends are a TTY, so for most users
a configured compaction hook never ran at all. This is the two-parallel-paths
hazard CLAUDE.md calls out: a change landed in one path and not its mirror.

**Fixed.** The dispatch moved out of `compact()` into `Agent::fire_pre_compact`
and `Agent::fire_post_compact`, which both orchestrators call, with
`Agent::compact_trigger` deriving `manual` for `/compact` and `auto` otherwise.
The hook plumbing in `src/hooks.rs` was already correct; only the call site was
missing. Covered by `compaction_hooks_fire_on_the_tui_path`, which was confirmed
to fail before the fix.

### 2. No empty-summary guard — **fixed**

The reference throws when the summary comes back empty, throws when it starts
with an API-error prefix, and specifically checks `isApiErrorMessage` because an
ESC abort arrives as a synthetic assistant message whose text does *not* start
with `API Error`.

plank checks `stats.interrupted`, which covers the abort case, and then calls
`rebuild_after_compact(&summary)` unconditionally. A generation that yields no
text destroys the transcript and replaces it with
`<tool_result>Compacted session summary:\n</tool_result>` plus the tail. The
verbatim tail survives, so this is not total loss, but it is unforced.

**Fixed.** An empty (or whitespace-only) `extract_summary` result — including a
reply that is only a discarded `<analysis>` block — is now a failed pass:
`Compacted::NoSummary` is returned, the transcript is untouched, `PostCompact`
does not fire, and the turn is abandoned. Note it returns a *status*, not an
`Err`: the main TUI loop propagates turn errors with `?`, so erroring here would
tear down the session over a recoverable failure. `Compacted::aborted()` is what
both turn sites check, so an interrupt and an empty summary are handled
identically while only an interrupt consumes the interrupt flag. Microcompact has
already run by this point, so "leave it alone" keeps the cleared tool results —
the correct outcome, since that pass did reclaim something.

## Gaps, with a verdict on each

| Reference feature | plank today | Verdict |
| --- | --- | --- |
| `/compact [instructions]` | **done.** `make_prompt(reason, instructions)` splices the argument between the section list and the no-tools trailer; both orchestrators pass it; `/compact` moved to the with-args form in `config.rs` | Landed |
| No-tools instruction at both ends of the prompt | trailer only | **Port.** The reference measured a 2.79% stray-tool-call rate on one model generation against 0.01% on the previous one; one stray call wastes the pass and a full re-prefill |
| Per-file cap on re-injection | absent: one 16K-token file can consume an entire ctx/8 budget | **Port.** A few lines in `build_reinjection`, and it stops one large file crowding out the other four |
| Clearing the read-file set after compaction | `recent_reads` is a capped LRU (`RECENT_READS_CAP = 16`) that survives compaction, so the same top-5 files re-inject at *every* pass | **Port.** The reference clears `readFileState` after re-attaching, so the next compaction only considers files touched since. Today plank can re-inject a file the model has not looked at in many turns |
| "Session is being continued…" wrapper, plus a pointer to the on-disk transcript, plus "resume without acknowledging the summary" on auto-compaction | plain `<tool_result>Compacted session summary:` framing | **Port selectively.** Keep the `<tool_result>` framing (it is what the ds4 model was trained on and what `c_parity` pins); add the do-not-acknowledge instruction and the transcript-path pointer, which are free |
| Prompt-too-long retry (drop oldest round groups and retry) | none | **Marginal.** The 85% soft trigger plus microcompact makes overflow unlikely on a fixed local window, but a single huge tool result can still cross it with no recovery path |
| 9th prompt section ("Problem Solving") | 8 sections; this one omitted | **Marginal.** Cheap to add, unclear value over sections 4 and 7 |
| Pre/post token-count telemetry | a single `"context compacted"` line | **Marginal.** Would let the ctx/8 tail and re-injection constants be tuned against evidence rather than guessed |
| Session-memory cheap route (compaction with no model call) | none; microcompact is the only cheap route | **Skip.** Depends on an extraction pipeline plank does not have. `/remember` and layered `MEMORY.md` are durable user facts, not a compaction shortcut |
| Reactive compaction on prompt-too-long | none | **Skip.** Subsumed by the PTL-retry row above |
| Auto-compact circuit breaker | none | **Skip deliberately.** plank's bounded rebuild makes the failure mode it guards against unreachable |
| `DISABLE_COMPACT`, threshold override env var | none | **Low.** plank's percentage trigger (`COMPACT_SOFT_PERCENT` = 85, `COMPACT_MIN_FREE_TOKENS` = 8192 capped at ctx/4) already scales across window sizes better than the reference's absolute 13K buffer, which is tuned for one known model |

### On the thresholds

Worth stating explicitly, since it looks like a gap and is not one. The
reference triggers on an absolute buffer: `effectiveContextWindow − 13_000` for
auto-compaction, a 3K reserve for manual, warning bands 20K below. plank
triggers on 85% of the window *or* fewer than 8192 free tokens, whichever comes
first, with the fixed floor capped at ctx/4 so tiny-context runs still compact
rather than fail. Absolute buffers are better when the window size is known and
fixed; a percentage is better across the 8K-to-128K spread of local GGUFs plank
actually runs. Keep the percentage.

## Ordered plan

1. ~~**Hook parity** on the TUI paths (defect 1).~~ **Landed.**
2. ~~**Empty-summary guard** (defect 2).~~ **Landed.**
3. ~~**`/compact <instructions>`**~~ — **landed.** The argument is threaded into
   `make_prompt` as an additional-instructions block, above the no-tools trailer,
   and the command is registered as argument-taking. The pinned
   `compaction_prompt.txt` fixture is unchanged, so the automatic form is
   byte-identical to before. Still outstanding from the reference: merging
   instructions returned by a `PreCompact` hook, which needs item 1 first.
4. **Prompt hardening**: no-tools preamble as well as trailer; the
   do-not-acknowledge line; the transcript-path pointer.
5. **Re-injection fixes**: per-file token cap, and clear `recent_reads` after a
   successful rebuild.
6. *(optional)* pre/post token-count reporting, to make the constants tunable.

Items 1, 2, 4 and 5 are contained changes in `src/compact.rs` and the two
orchestrators. Item 3 touches command registration and completion. Nothing here
requires the boundary-marker rewrite, which stays out of scope.

Anything landed from this list must be mirrored across **both** orchestration
paths, which is the lesson of defect 1. Merging `compact()` and
`do_compact_notify()` into one implementation behind the `CompactSink` trait
would remove the class of bug entirely, and is the right shape to aim for while
doing items 1 through 3.

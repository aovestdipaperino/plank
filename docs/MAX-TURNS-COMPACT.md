# MAX_TURNS and inline compact — port plan

Status: **planned, not implemented.** This document captures the design for
porting two patterns from `local/claude-code-typescript-code` into plank's
`src/`:

1. **MAX_TURNS** — a turn budget on the *main* turn loop (currently unbounded).
2. **Inline compact** — run the cheap compaction step on *every* loop iteration,
   not just once at turn start.

Both are turn-loop concerns above the `Engine` trait, so they need no
`#[cfg(ds4_engine)]` gating and are fully exercisable with `EchoEngine`.

## Reference (TypeScript)

In `local/claude-code-typescript-code`:

- `maxTurns` is a per-agent turn budget threaded through the query stack
  (`query.ts`, `runAgent.ts`, `forkedAgent.ts`). `query.ts` checks it after
  each turn and yields a `max_turns_reached` attachment. The fork subagent's
  `FORK_AGENT.maxTurns = 200` (`forkSubagent.ts:65`).
- "Inline compact" = microcompact running every turn inside the query loop,
  before each API request (`query.ts` `query_microcompact_start`/`_end`
  checkpoints). It clears/replaces old tool results in flight, distinct from
  autocompact's heavier summarization.

## Current state in plank (verified)

- `run_turn` (`src/ui.rs:2052`, plain REPL) and `worker_turn` (`src/ui.rs:6548`,
  TUI) are flat `loop { … }` that end only on no-tool-calls / interrupt /
  hook-stop. **No turn counter, no cap.**
- `run_subagent_rounds` (`src/ui.rs:1824`) and `run_fanout_rounds`
  (`src/ui.rs:4306`) **already** bound themselves with `const MAX_ROUNDS: usize
  = 40` and push `agents::final_round_reminder()` on the last round to force a
  text conclusion. This is the in-repo template to copy.
- `compact::microcompact` (`src/compact.rs:44`) exists and is idempotent, but it
  only runs via `maybe_compact` / `maybe_compact_notify` **once, before the loop
  starts** (`ui.rs:2067` and `ui.rs:6578`). After tool results are pushed inside
  the loop, context pressure is **not re-checked** until the next turn — a
  tool-heavy turn can blow the window mid-turn with no relief.
- `SUBAGENT_DEPTH_CAP = 1` / `SKILL_DEPTH_CAP = 8` live in `src/tools/mod.rs`
  (`:149`, `:154`) alongside each other — the natural home for a new main-turn
  cap.
- `agents::final_round_reminder()` (`src/agents.rs:345`) is the existing
  "final turn, no more tools, write the report" reminder string, shared by the
  serial and parallel subagent paths.

## Change 1 — MAX_TURNS on the main turn loop

### Constant

New constant in `src/tools/mod.rs`, next to the other caps:

```rust
/// Maximum generate→tools rounds in one main turn before the model is forced
/// to conclude. Bounds a runaway turn that keeps calling tools forever.
/// Sub-agent sidechains use their own `MAX_ROUNDS` (40); the main loop gets
/// headroom because a real task legitimately runs longer.
pub const MAX_TURNS: usize = 200;
```

200 matches the TS fork-subagent default and is generous for real work; a turn
that hits it is almost certainly stuck.

### Mechanism

Copy `run_subagent_rounds`'s pattern into both main loops:

- Convert `loop { … }` to `for round in 0..MAX_TURNS { … }`.
- On `round + 1 == MAX_TURNS` (last permitted round), push
  `Message::user(agents::final_round_reminder())` before generating, and treat
  whatever text comes back as the turn's conclusion (return `Ok(())`), exactly
  like `run_subagent_rounds` does at `ui.rs:1832-1843`.
- Reuse `agents::final_round_reminder()` verbatim — its operative instruction
  ("Do not call any more tools … write your final report as plain text now") is
  correct for the main turn too; the word "report" is harmless and avoids drift
  between two near-identical strings.
- After the loop falls through (unreachable given the final-round return, but
  keep `Ok(())` like `run_subagent_rounds` does at `ui.rs:1866`).

### Edit sites (mirror change — both must move together per CLAUDE.md)

- `run_turn` (`src/ui.rs:2077`): the `loop { … }` through `return Ok(())` at
  `ui.rs:2201`.
- `worker_turn` (`src/ui.rs:6594`): the `loop { … }` through `return Ok(())` at
  `ui.rs:6727`.

### Configurability (optional)

Add `pub max_turns: usize` to the agent `Config` in `src/config.rs` (default
`MAX_TURNS`), so `/config` or a setting can raise it. The loops read
`self.cfg.max_turns` instead of the bare const. If skipped, the const is fine
for a first cut.

## Change 2 — Inline microcompact inside the turn loop

### Core move

Lift the `maybe_compact` / `maybe_compact_notify` call from *before* the loop to
the *top* of every loop iteration, so the cheapest relief runs every round (and
full compaction only when microcompact isn't enough, same as today).

### Edit sites (mirror change)

- `run_turn`: move `self.maybe_compact()?` from `ui.rs:2067-2069` to the first
  line inside `for round in …` (before `render_transcript` at `ui.rs:2078`).
  Keep the `if …aborted() { return Ok(()) }` guard.
- `worker_turn`: move the
  `self.maybe_compact_notify(&mut NoteSink(&mut note), &compact_interrupt)?`
  block from `ui.rs:6574-6586` to the top of the inner `for round in …` (before
  `render_transcript` at `ui.rs:6595`). The `note` closure and `compact_interrupt`
  closure are already in scope for the whole function, so they compose unchanged.

### Why this is safe

- `compact::microcompact` is idempotent (second call returns 0; covered by the
  `microcompact_clears_only_old_large_results` test). Running it every round is
  cheap and a no-op once context is comfortable.
- `should_compact` returns false under threshold, so the per-iteration cost when
  context is fine is one `count_tokens` call — the same thing `maybe_compact`
  already does once today.
- `Compacted::aborted()` (interrupt or no-summary) already ends the turn; that
  semantics is identical inside the loop.
- Full compaction mid-turn rebuilds the transcript via `rebuild_after_compact`,
  which already clears branches and re-injects the task block (`ui.rs:2382-2419`)
  — the same path runs at turn start today, so mid-turn invocation composes.

### Scope note

This inlines the *whole* `maybe_compact` (micro + full). A lower-risk first cut
would inline **only `try_microcompact`** every round and leave full compaction at
turn-start. The recommended path is the full move (faithful to the TS, which
runs autocompact in the loop too).

## Tests

- **MAX_TURNS**: in `src/ui.rs`'s test module, add a test using
  `ScriptedEngine`/`EchoEngine` that emits a tool call on every generation for
  >MAX_TURNS rounds and asserts (a) the loop terminates, (b) the final
  assistant message is text (no tool call), and (c) `final_round_reminder()`
  was pushed as a user message. Mirror the existing
  `worker_turn_drains_queued_user_between_tool_rounds` test style (`ui.rs:14275`).
- **Inline microcompact**: add a test where a tool returns a large result
  mid-turn that crosses the `should_compact` threshold, and assert microcompact
  runs *during* the same turn (the large result body is replaced with
  `MICROCOMPACT_STUB` before the next generation). Today this only happens on
  the *next* turn. Use the `microcompact_clears_only_old_large_results` helper
  style for the fixture.
- **Adjustment**: scan existing tests that count compaction passes or assert
  exact turn counts (e.g. around `ui.rs:9157`, `ui.rs:14275`, `ui.rs:14334`) —
  moving `maybe_compact` inside the loop may run it one extra time in tests with
  tiny (64-token) `EchoEngine` contexts. Update assertions where needed; the
  behavior change is intentional.
- Run `cargo test --lib` and `cargo clippy --workspace --all-targets -- -D
  warnings` (the CI gate).

## Ordering / risk

1. **MAX_TURNS first** (smaller, self-contained, mirrors an existing pattern).
   Land + test.
2. **Inline microcompact second** (touches the loop structure Change 1 just
   edited, and may shift test counts). Land + test.
3. Both are gated to the two parallel paths (`run_turn` + `worker_turn`) — the
   CLAUDE.md "mirror change" rule applies; make both edits in the same commit.
4. No `#[cfg(ds4_engine)]` needed — these are turn-loop concerns above the
   engine trait, fully exercisable with `EchoEngine`.

## Out of scope (explicitly not doing)

- TS-style cache-edits / time-based microcompact / cached-MC state — plank has
  no server-side cache-edit API; its microcompact is purely local
  content-clearing, which is the direct analog.
- TS-style per-agent `maxTurns` on `AgentDef` — plank's `AgentDef`
  (`src/agents.rs`) has no model/maxTurns fields and subagents already use the
  fixed `MAX_ROUNDS = 40`. Adding per-def maxTurns is a separate enhancement.
- Recursion guards (`session_memory`/`compact` querySource exclusions) — plank
  has no forked-agent compaction recursion; `run_subagent_loop` deliberately runs
  *no* compaction (`ui.rs:1792-1794`), so there's no recursion to guard.

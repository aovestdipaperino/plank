# Mid-generation /btw — Genuine Freeze / Answer / Unfreeze

Design document for answering a `/btw` side question **during an in-flight
generation pass** — literally suspending the main task's token stream, answering
the aside, then resuming the main task exactly where it left off. This lifts the
restriction `docs/BTW-DESIGN.md` §4.3 accepted (btw is answered only at
generation *boundaries*, never mid-pass) now that the engine layer is confirmed
to support it cheaply.

Status: **implemented behind `btw.suspend` (default off; `--btw-suspend`).**
Depends on the boundary-scheduled `/btw` already shipped (BTW-DESIGN steps 1–4).
Steps 1–3 (engine trait + `Ds4Engine::generate_aside` + snapshot round-trip
foundation) and steps 4–6 (worker integration, presentation, docs) have landed;
the real-model snapshot-round-trip smoke test (§5.3) remains Metal-only.

## 1. Motivation

Today a mid-turn `/btw` is queued and answered at the next generation boundary
(between tool passes), where the transcript is stable. Inside a *single* long
generation pass — a big model reply streaming for tens of seconds — the queued
question waits for the whole pass to finish. For "wait, what did that flag mean?"
mid-stream, that latency is the whole point of `/btw`, lost.

BTW-DESIGN §4.3 deferred true suspension because it looked like it required "a
transcript snapshot plus a second engine context, or aborting the pass — both
rejected." The engine investigation (see below) shows the first option is
already available through the FFI and is cheap. This document specifies it.

## 2. Why this is possible — the engine facts

Established by reading `src/ds4engine.rs` and `src/ffi.rs`:

- **Generation is a Rust-side token loop** (`Ds4Engine::generate`,
  `src/ds4engine.rs:485`): `while generated < max_tokens { sample; eval; }`,
  checking `interrupt()` each iteration. Breaking the loop leaves the
  `Ds4Session` holding a **fully valid KV state** at its write cursor
  (`ds4_session_pos`). A halt at a token boundary *is* a pause — the state is
  intact, nothing is lost by stopping.
- **The one thing a mid-pass halt holds that a boundary halt doesn't** is a
  *partial assistant reply*: tokens sampled and committed to the KV cache but
  not yet in `session.transcript`. That partial-reply KV is the state a btw
  would otherwise destroy by reusing the single live session.
- **The FFI already exposes both ways to protect it:**
  - `ds4_session_save_snapshot` / `ds4_session_load_snapshot` /
    `ds4_session_snapshot_free` (`src/ffi.rs:198`) — serialize and restore a
    session's KV state.
  - `ds4_session_create(out, engine, ctx_size)` (`src/ffi.rs:157`) — a second
    session on the same engine with its own KV, sharing the read-only weights.

Neither the single-live-session policy nor "stop, don't pause" is a Metal or
platform limitation; both are plank design choices. Metal's only real
constraint is that a second session does not buy *parallelism* — one command
queue means two generations time-slice — which is a throughput cost, not a
correctness barrier. This design therefore keeps the main task **frozen** (not
concurrently running) while the aside is answered, and pays only the cost of
preserving/restoring its KV.

## 3. Two mechanisms, and the choice

| | (A) Snapshot / restore on the single session | (B) Ephemeral second session |
|---|---|---|
| Preserve main KV | `save_snapshot` before the aside | left untouched in session A |
| Answer the aside | destructively on the same session | on a fresh session B |
| Restore main KV | `load_snapshot` after the aside | discard B; A already intact |
| Memory cost | one transient snapshot buffer (~used KV) | a second live ctx (~KV) for the aside's lifetime |
| Re-prefill on resume | none — snapshot restores exact KV incl. cursor | none — A never moved |
| Enables true concurrency later | no | yes (foundation for a real second stream) |
| New FFI surface | none | none |

**Decision: (A) snapshot/restore is the primary design.** It is lighter (a
transient buffer vs. a full second live context), needs no second ctx
allocation, and restores the main task's KV *exactly* — including the partial
reply and cursor — so resume costs zero re-prefill. (B) is documented as the
evolution path if plank ever wants the aside to run *concurrently* with the main
task (§8), which snapshot/restore cannot provide.

## 4. Detailed design

### 4.1 Trigger and scope

- Applies only to a `/btw` (or `/side`) submitted **while a generation pass is
  streaming**, in the TUI worker path (`src/worker.rs`). Idle and
  boundary-scheduled btw keep their existing, cheaper paths (BTW-DESIGN §4.2,
  §4.4) — suspension is strictly for the in-pass case.
- Off by default behind a config flag `btw.suspend` (default `false`) until
  proven on the real Metal engine; when off, an in-pass `/btw` falls back to the
  boundary queue exactly as today. No behavior change unless opted in.

### 4.2 The engine-level primitive

Add one method to the `Engine` trait (`src/engine.rs`), default-unsupported so
`EchoEngine` and remote engines need no change:

```rust
/// Answer a one-shot, tool-free prompt without disturbing the live
/// generation state, then restore it exactly. Returns the aside's text.
/// Default: `Err(EngineError::unsupported())` — callers fall back to the
/// boundary queue.
fn generate_aside(
    &mut self,
    prompt: &str,
    opts: &GenerationOptions,
    interrupt: &dyn Fn() -> bool,
    on_event: &mut dyn FnMut(EngineEvent),
) -> Result<GenerationStats, EngineError> { Err(EngineError::unsupported()) }
```

`Ds4Engine::generate_aside` (the only real impl for now):

1. **Snapshot.** `save_snapshot(session)` → owned `Ds4SessionSnapshot`. This
   captures the frozen main-task KV (transcript + partial reply + cursor).
2. **Answer destructively on the same session.** Run the standard btw execution
   model (BTW-DESIGN §4.2): `render_transcript + btw_user_message(question)` as
   the prompt. `ds4_session_sync` rolls the session's cursor back to the common
   prefix with the frozen state — which is the transcript (the partial reply
   diverges), so only the framed question is prefilled — then the normal token
   loop generates the answer with **tools denied** (drop `finished().calls`) and
   greedy off. Stream its `EngineEvent::Text` to `on_event` so the aside renders
   live.
3. **Restore.** `load_snapshot(session, &snap)` returns the session to the exact
   frozen KV and cursor; `snapshot_free`. The main task's next
   `sample`/`eval` continues as if nothing happened — **zero re-prefill**,
   because the KV positions, including the partial reply, are byte-identical to
   the pre-aside state.
4. **Accounting.** Save/restore `last_ctx_used` around the call, as the existing
   btw paths do; the aside's tokens never touch the main context estimate.

Interrupting the aside (Esc/Ctrl-C) aborts step 2's loop, then step 3 still
runs — an interrupted aside must never leave the main task's KV corrupted. The
snapshot is the safety net: restore is unconditional (RAII guard), even on the
error path.

### 4.3 Worker-loop integration

In the worker turn loop (`src/worker.rs`), the token loop that today only checks
`interrupt` between tokens gains a second check: an in-pass btw request in
`TurnShared::btw` (when `btw.suspend` is enabled). On seeing one:

1. Finish the current token (don't tear a multi-byte piece).
2. Emit a `UiEvent` marking the split (§4.4).
3. Call `engine.generate_aside(...)` for each queued question in FIFO order
   (cap 20, drop-oldest — same policy as boundary btw).
4. Resume the main token loop. The engine state is already restored; the worker
   simply continues sampling.

The worker owns the engine, so this is a straight-line call — no cross-thread
handoff, no second thread. The main generation is genuinely frozen for the
wall-clock duration of the aside (Metal time-slicing is moot because we do not
run them concurrently).

### 4.4 Presentation

Reuse the side-channel markers from BTW-DESIGN §4.5, with an in-pass twist so the
user sees the main reply visibly pause and resume:

- On suspend: end the current output line, emit `[btw — main task paused]` (dim),
  then the echoed question and the streamed answer bracketed as a side block.
- On resume: `[btw — resuming]` (dim), then the main reply continues on a fresh
  line. The already-streamed portion of the main reply stays on screen; the
  continuation appends to it, matching how the KV actually resumed.

### 4.5 Invariants added to BTW-DESIGN's set

- The main task's KV state after a suspended aside is **byte-identical** to
  before it (snapshot round-trip is lossless); resume does zero re-prefill.
- Restore is unconditional — any aside outcome (success, interrupt, error)
  leaves the main session valid.
- The aside remains ephemeral: nothing it generates enters `session.transcript`,
  including the partial main reply it was interleaved with.
- Feature-flagged off by default; when off, behavior is exactly today's
  boundary-scheduled btw.

## 5. Implementation plan

Each step independently landable; EchoEngine covers everything except the real
snapshot round-trip.

1. **Trait + fallback.** Add `Engine::generate_aside` with the default
   `unsupported` impl and `EngineError::unsupported()`. Callers detect it and
   fall back to the boundary queue. No engine work yet.
2. **`Ds4Engine::generate_aside`** (`ds4_engine` cfg): snapshot → destructive
   btw run → restore, with the unconditional-restore guard. Unit-testable only
   on a Metal box; add a `#[cfg(ds4_engine)]` integration smoke test.
3. **Snapshot round-trip test.** A focused test (real engine) that snapshots
   mid-reply, runs an aside, restores, and asserts the next N tokens are
   identical to an uninterrupted run of the same seed — proving losslessness.
4. **Worker integration + flag.** `btw.suspend` config, the in-pass check in the
   worker token loop, FIFO drain, fallback when the flag is off or the engine
   returns `unsupported`.
5. **Presentation.** Suspend/resume markers in both the split-screen renderer
   and the plain path.
6. **Docs.** This file; a note in BTW-DESIGN §4.3 that the deferral is lifted;
   FINDINGS.md entry for the snapshot-round-trip losslessness result.

## 6. Testing

- `aside_unsupported_falls_back` (EchoEngine) — `generate_aside` returns
  `unsupported`; the worker routes the question to the boundary queue; no panic.
- `aside_restores_on_interrupt` (EchoEngine stub of the guard) — the restore
  runs even when the aside loop is interrupted.
- `aside_leaves_transcript_untouched` — transcript bytes unchanged across a
  suspended aside.
- `aside_snapshot_roundtrip_lossless` (`#[cfg(ds4_engine)]`, real model) —
  §5.3: post-aside continuation matches an uninterrupted seeded run.
- `aside_fifo_cap` — >20 in-pass questions drop oldest with a notice.
- Manual (Metal): start a long reply, fire `/btw` mid-stream, confirm the reply
  visibly pauses, the aside streams, the reply resumes on the same content with
  no re-prefill flicker in the status bar, and context accounting is unchanged.

## 7. Constraints and invariants

Inherits all of BTW-DESIGN §7 (model-facing framing byte-stable, nothing enters
the transcript, prompt is `full transcript + suffix`, two UI paths), plus §4.5
above. Additionally:

- **No second live context in the primary design** — snapshot/restore only. A
  second session is §8, not this.
- **Engine-agnostic fallback** — any engine without `generate_aside` degrades to
  the boundary queue; the feature never hard-requires the primitive.

## 8. Non-goals / evolution path

- **Concurrent aside (true second stream).** Running the aside *while* the main
  task also generates needs mechanism (B) — an ephemeral second session
  (`ds4_session_create`) — and buys only time-sliced, not parallel, compute on
  Metal's single queue. Rejected here for the same cost/complexity reasons as
  BTW-DESIGN §8; §8.1 records what it would take, since it is feasible and shares
  a prerequisite with other roadmap work.
- **Suspending across tool dispatch.** Out of scope; boundary scheduling already
  covers the between-passes case.
- **Snapshotting for session save/resume.** The same FFI powers per-session KV
  payloads (#12); that is a separate feature with its own persistence format.

### 8.1 What concurrent sessions would require (and whether it's worth it)

Feasible — nothing below is blocked by Metal or the engine — but the payoff for
`/btw` specifically is small. Documented so a later version can adopt it without
re-deriving the engine facts.

Requirements, in dependency order:

1. **Split "engine" from "session" in the Rust API.** Today `Ds4Engine` owns a
   single `session: *mut Ds4Session` and `generate(&mut self, …)` conflates the
   two. The C layer already separates them (`ds4_session_create(out, engine,
   ctx_size)` against shared, read-only weights; multiple sessions per engine are
   supported). Make a session a first-class Rust object you can hold two of —
   weights in one place, KV + cursor in each session. **This is the load-bearing
   change**; everything else sits on top.
2. **Thread-safety audit of the C engine + Metal.** The real risk, not memory.
   The engine and its Metal command queue were written for one worker thread;
   two threads calling `ds4_session_eval`/`sample` concurrently is almost
   certainly unsound as-is. Either serialize every GPU-touching call behind a
   mutex and **interleave at token granularity** (a cooperative scheduler on one
   thread is cleaner than two threads contending a mutex), or prove concurrent
   submission is safe (unlikely).
3. **Cheap bootstrap of the second session.** A fresh session B would re-prefill
   the whole transcript — the exact cost the single-session design avoids. Fix:
   `save_snapshot(A)` → `load_snapshot` into a freshly-created B to clone A's KV
   rather than recompute it, then let B diverge with the framed question. The
   snapshot primitive from §4.2 is therefore a *dependency* of the concurrent
   design, not an alternative to it.
4. **Bounded second KV allocation.** Two live contexts; B must hold transcript +
   question + answer, so it approaches A's size — memory roughly doubles for the
   aside's lifetime.
5. **Two-stream multiplexing.** The split-screen renderer already shipped; the
   worker loop (`src/worker.rs`) would drive two generation loops over tagged
   events, with two interrupt flags and two `last_ctx_used` accountings.

**Verdict.** One GPU with one command queue means "concurrent" is *time-sliced,
not parallel*: the main task does not progress faster while the aside runs. The
only gain over freeze/answer/unfreeze is that the main task keeps inching forward
instead of being fully paused for the seconds an aside takes — not worth the
thread-safety audit and doubled KV for `/btw` alone. However, requirement 1 (the
session-as-first-class-object refactor) is the **same primitive** that per-session
KV payloads (#12), instant `/switch` without re-prefill, and subagent sidechains
with their own KV all need. Justify that refactor by those features; concurrent
`/btw` then falls out as a near-free consequence.

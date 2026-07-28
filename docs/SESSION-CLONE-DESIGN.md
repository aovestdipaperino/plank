# Session Cloning — Validating the Fork Primitive via Concurrent `/btw`

Status: proposal. Prerequisite for parallel subagents ([#19](https://github.com/aovestdipaperino/plank/issues/19)).
Companions: [`BTW-SUSPEND-DESIGN.md`](BTW-SUSPEND-DESIGN.md) §8.1 (which specified this),
[`SHARED-ENGINE-DESIGN.md`](SHARED-ENGINE-DESIGN.md) (the host that made it feasible).

## 1. Motivation

`/subagent` is a *transcript* fork: it appends the framed task to the live
transcript, runs the ordinary turn loop, then truncates the sidechain and keeps
only the report (`src/agents.rs:6-13`). It is cheap precisely because it never
leaves the parent's session — the per-turn common-prefix sync reuses the parent
KV in place.

That design cannot host two subagents at once: two forks are two divergent
continuations of one prefix, and there is only one KV cursor. What is missing is
a **session clone**: take a live session A, produce a second live session B that
starts from A's exact KV, and let them diverge independently.

BTW-SUSPEND-DESIGN §8.1 listed five requirements for that. Three have since
landed as a side effect of the shared-engine work:

| §8.1 requirement | Status |
| --- | --- |
| 1. Session as a first-class object, split from the engine | **Done** — `ModelHandle` (weights) vs `HostSession` (KV + cursor), `src/host.rs:137`/`:171` |
| 2. Thread-safety: cooperative scheduler, token-granularity interleave | **Done, as recommended** — `scheduler_loop` round-robins at `cfg.slice_tokens`, `src/host.rs:465` |
| 3. Cheap bootstrap: snapshot A, load into a fresh B | **Primitive exists, never used this way** — `snapshot_bytes`/`restore_bytes`, `src/host.rs:203`/`:215` |
| 4. Bounded second KV allocation | **Partly** — the host has an admission cap and KV-bytes budget, `src/host.rs:333` |
| 5. Two-stream multiplexing | **Not built** |

So exactly one load-bearing assumption is unproven: that `snapshot_bytes` on a
**still-live** session, restored into a **freshly spawned sibling**, yields a
usable clone with no cold re-prefill and no damage to the original. Everything
about parallel subagents rests on that. This document specifies how to prove it.

## 2. Why `/btw` is the vehicle

`/btw` is the smallest feature that genuinely needs a second divergent
continuation of the main task's context, and it already has all the surrounding
machinery: a framed ephemeral prompt, a split-screen renderer, a queue, and
markers. Concurrent `/btw` is therefore a near-zero-UI way to exercise cloning
end to end.

**It is a test vehicle, not a shipping goal.** §8.1's verdict stands: one Metal
command queue means concurrent is *time-sliced, not parallel*, so the main task
does not finish sooner. The user-visible gain over today's freeze/answer/resume
is that the main task keeps inching forward during the aside. That is not worth
shipping on its own. It is worth building as a throwaway proof that the clone
primitive is sound, because the clone primitive is what unblocks #19.

Today's `generate_aside` (`src/ds4engine.rs:853`) is the counterexample that
defines the gap: it captures a snapshot, answers **destructively on the same
session**, and unconditionally restores via `RestoreOnDrop`. One session, one
cursor, strictly serial. The concurrent variant keeps A running and puts the
aside on B.

## 3. What must be proven

Four properties, in dependency order. Phase 1 (§5) tests P1–P3 with no `/btw`
involvement at all; only P4 needs the feature.

- **P1 — Non-destructive capture.** `snapshot_bytes()` on a live session leaves
  A's KV and cursor unchanged. A's subsequent generation is byte-identical to a
  control run without the capture.
- **P2 — Faithful clone.** B, spawned fresh and given A's bytes, holds A's token
  transcript and cursor. `snapshot_bytes` already prepends the encoded
  `TokenTranscript` in front of the KV bytes (`src/ds4engine.rs:1249-1255`), and
  `restore_bytes` decodes it back and supersedes B's transcript
  (`:1263-1270`), so this should hold — but it has only ever been exercised
  restoring into the *same* session id after idle reclaim.
- **P3 — No cold re-prefill.** B's first generation prefills only the genuinely
  new suffix. Measured, not assumed: assert prefill tokens are a small constant,
  not proportional to transcript length.
- **P4 — Independent divergence.** A and B interleave in the rotation without
  cross-contamination. A's output is unaffected by B's existence; B's aside never
  leaks into A's transcript.

## 4. Constraints inherited from the existing code

- **Metal-gated.** Both `snapshot_bytes` and `restore_bytes` default to no-ops on
  `HostSession` (`Some`-less / `Ok(())`). Only `Ds4Engine` implements them, so
  every test here is `#[cfg(ds4_engine)]` plus a real GGUF. `EchoEngine` cannot
  exercise any of it.
- **Double-free discipline.** Snapshot bytes that have made a round trip through
  Rust-owned memory MUST restore through the non-owning `restore_bytes` FFI path,
  never the owning one (FINDINGS; `src/snapshot.rs:171` and `src/ds4engine.rs:1258-1261`).
  A clone path produces exactly such a buffer, so it uses `restore_bytes`.
- **KV budget is real.** Each clone is a full `attach` under `max_sessions` and
  the KV-bytes budget. B approaches A's size, so memory roughly doubles for the
  clone's lifetime (§8.1 requirement 4). The clone request must fail cleanly, not
  panic, when admission refuses.
- **GPU-thread affinity.** `snapshot_bytes`/`restore_bytes` run on the single
  `plank-gpu` worker thread. A clone is therefore a scheduler `Command`, not
  something a caller does behind the host's back.
- **Never clone a rotating session mid-generation** in phase 1. Reclaim already
  refuses to touch a session in the rotation (`src/host.rs:597`); the clone path
  starts with the same restriction and only relaxes it in phase 3.

## 5. Phase 1 — Prove the primitive (no `/btw`)

The cheapest possible test, and the one that carries all the risk.

### 5.1 Host API

Add one command and one method, mirroring `attach_sized`:

```rust
// src/host.rs
impl EngineHost {
    /// Attaches a new session pre-loaded with `from`'s current KV and token
    /// transcript, so it continues where `from` left off without re-prefilling.
    pub fn attach_clone(&self, from: &SessionHandle) -> Result<SessionHandle, EngineError>;
}

enum Command {
    Clone { src: u64, reply: Sender<Result<u64, EngineError>> },
    // ...
}
```

Scheduler handling, in `apply_command`:

1. Look up `src`; error if missing or currently reclaimed (`slot.session.is_none()`).
2. Refuse if `src` is in `rotation` (phase 1 restriction, §4).
3. `snapshot_bytes()` on the source. `None` means the backend cannot snapshot,
   so return an `unsupported`-style `EngineError` and let the caller fall back.
4. Run the ordinary admission checks (`max_sessions`, KV-bytes) for the new
   session at the source's `ctx_size`. On refusal, return the same error
   `attach_sized` would.
5. `model.spawn(ctx_size)`, then `restore_bytes(&bytes)` into it.
6. Insert the slot, `publish_status`, reply with the new id.

Note the bytes stay in memory here; the reclaim path writes them to disk
(`src/host.rs:606-609`) only because it needs to free the KV. A clone does not.

### 5.2 Tests

All `#[cfg(ds4_engine)]`, in `src/ds4engine.rs` next to the existing
snapshot/reclaim tests, reusing `two_sessions_no_cross_contamination` for the
two-live-sessions harness and `idle_reclaim_restore_no_cold_reprefill`
(`src/ds4engine.rs:2113`) for the prefill-accounting assertion pattern.

| Test | Proves | Shape |
| --- | --- | --- |
| `clone_capture_does_not_disturb_source` | P1 | Run A one turn; capture; run A again. Assert output and `ctx_used` match a control run with no capture. |
| `clone_carries_source_transcript` | P2 | Clone A into B; assert B's `ctx_tokens`/cursor equal A's, and B's transcript decodes to A's. |
| `clone_avoids_cold_reprefill` | P3 | Build a long transcript in A, clone, generate on B. Assert B's first-generation prefill count is a small constant, not O(transcript). This is the headline assertion. |
| `clone_diverges_independently` | P4 | Clone, drive both concurrently through the host rotation, assert A's stream matches its solo control and B's content never appears in A's transcript. |
| `clone_refused_over_kv_budget` | §4 | Set a tight `HostConfig` budget; assert a clean `EngineError`, source still usable. |
| `clone_of_reclaimed_session_errors` | §4 | Force idle reclaim, then clone; assert a clean error rather than a panic. |

**Exit criterion for phase 1.** If `clone_avoids_cold_reprefill` fails, stop —
cloning is not cheap, and parallel subagents need rethinking from §8.1 requirement 3
rather than proceeding to `/btw`.

## 6. Phase 2 — Concurrent `/btw` over the clone

Only after phase 1 is green. This is §8.1 requirement 5.

### 6.1 Engine surface

Add alongside the existing pair:

```rust
// src/engine.rs
fn generate_aside_concurrent(&mut self, /* same args as generate_aside */)
    -> Result<GenerationStats, EngineError>;
fn supports_concurrent_aside(&self) -> bool { false }
```

Default `false` so `EchoEngine` and remote engines fall through to today's
`generate_aside`, which itself falls back to the boundary queue. Three tiers,
each degrading cleanly, matching the existing fallback discipline.

Implementation for the host-backed engine: `attach_clone` the live session, issue
the framed question on the clone, stream its events tagged as aside, detach the
clone when the answer finishes or is interrupted. The main session is never
touched, so there is no `RestoreOnDrop` and no transcript save/restore — the two
steps `generate_aside` needs precisely because it is destructive.

### 6.2 Worker and UI

- A second interrupt flag and a second `last_ctx_used` accounting, per §8.1.
- Event tagging so `UiEvent::BtwStart`/`BtwEnd` route the clone's tokens to the
  side panel while main tokens keep flowing to the main log.
- `BTW_SUSPEND_MARKER` / `BTW_RESUME_MARKER` (`src/worker.rs:236`, `:240`) are
  **not** emitted on this path: nothing pauses. If the clone is refused
  (budget, unsupported), fall back to the suspend path and emit them as today.
- Both slash-command paths in `ui.rs` need the change (plain REPL and TUI), per
  the two-parallel-paths rule in `CLAUDE.md`.

### 6.3 Config

Gate behind `--enable-btw-concurrent`, defaulting **off**, mirroring
`--disable-btw-suspend`'s shape but with the opposite default: this is
experimental validation, not a feature. Doubling KV for an aside is not an
acceptable silent default.

### 6.4 Tests

- `concurrent_aside_leaves_main_transcript_clean` — the strongest regression, and
  the direct analogue of the invariant `generate_aside` protects today.
- `concurrent_aside_falls_back_when_clone_refused` — tight KV budget, assert the
  suspend path runs and the markers appear.
- `concurrent_aside_unsupported_engine_uses_suspend` — `EchoEngine` tier check,
  runnable without Metal.
- Manual: a long generation plus a `/btw`, confirming the main log keeps
  advancing while the side panel fills.

## 7. Phase 3 — What this unlocks

With P1–P4 proven and exercised by a real feature, parallel subagents become a
scheduling problem rather than an engine problem:

1. Relax §4's "never clone a rotating session": clone at a token-slice boundary
   on the GPU thread, where the session is momentarily quiescent.
2. `/subagent` grows a parallel dispatch form; each agent gets a clone and its
   own transcript (`Session` is already `#[derive(Clone)]`, `src/session.rs:231`).
3. `SUBAGENT_DEPTH_CAP` (`src/tools/mod.rs:158`) and the synchronous
   `begin → run → finish` call (`src/ui.rs:1358-1361`) are revisited; the depth
   cap likely becomes a *breadth* cap driven by the KV budget.
4. Reports rejoin the parent transcript in a deterministic order (completion
   order is nondeterministic; sort by dispatch index).

Nothing in phase 3 is committed here. The point of phases 1 and 2 is to find out
whether it is worth writing.

## 8. Risks

- **Snapshot cost is not free.** Capture serializes KV to owned bytes; restore
  reloads it. For a large transcript this is real latency and a transient double
  allocation. If capture turns out to cost seconds, cloning is not viable for
  interactive use and phase 2 should be abandoned even if phase 1 passes.
  `clone_avoids_cold_reprefill` should record wall time, not just token counts.
- **Metal-only coverage.** None of this is testable in CI. Phase 1 tests run
  locally on macOS with a model; CI keeps compiling them behind `cfg(ds4_engine)`
  without running them. Say so in the test module doc.
- **Fallback tiers multiply.** Three aside paths (concurrent, suspend, boundary
  queue) is one more than today. The `EchoEngine`-runnable tier tests in §6.4
  exist to keep that honest.
- **Scope creep into #19.** Phase 2 is deliberately gated off by default and
  deliberately small. Resist growing it into agent teams before phase 1's numbers
  are in.

## 9. Non-goals

- Shipping concurrent `/btw` on by default.
- True parallel GPU execution. One Metal queue, one worker thread; interleaving
  is time-slicing, and this document does not change that.
- Cross-process or persisted clones. `snapshot_bytes` already backs per-session
  KV payloads and `/checkpoint`; those are separate features with their own
  formats (`src/snapshot.rs:15`).

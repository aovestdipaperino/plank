# `/checkpoint` — named in-session rollback points

Implements issue #29: name a snapshot of the current conversation and roll back
to it later without leaving the session, ideally with zero re-prefill.

## Commands (UX)

- `/checkpoint` (no argument) — list the checkpoints in this session, each with
  its name, a one-line summary, age, and a `+kv` marker when an engine KV
  payload was captured.
- `/checkpoint <name>` — capture a checkpoint under `<name>`. Re-using a name
  overwrites that checkpoint in place.
- `/rollback <name>` — restore the transcript (and engine KV, when available)
  to `<name>`. The current tail is first saved as an auto-checkpoint named
  `pre-rollback`, so a rollback is itself undoable via `/rollback pre-rollback`.
- `/rollback` (no argument) — prints usage.

### Why `/rollback` and not `/checkpoint restore`

A separate verb reads more clearly at the prompt (`/rollback foo` vs.
`/checkpoint restore foo`) and mirrors the destructive/non-destructive split the
rest of the REPL already uses (`/save` vs. `/switch`).

### Why bare `/checkpoint` lists instead of auto-naming

Issue #29 sketched both "bare `/checkpoint` lists" and "omitted name
auto-generates a slug". Those conflict on the same input, so the implementation
picks the unambiguous one: bare `/checkpoint` always lists; creating a
checkpoint always requires a name. Auto-naming still happens for the one
implicit checkpoint the system creates itself (`pre-rollback`).

## Storage model

A checkpoint (`src/checkpoint.rs`, `Checkpoint`) holds:

- the **full transcript** at capture time (a `Vec<Message>` clone), and
- an optional **engine KV snapshot** (`Option<Vec<u8>>`).

Checkpoints live in an in-memory `CheckpointStore` owned by the `Agent`. They
are **per-session and ephemeral**: dropped on `/new`, `/clear`, `/switch`, and
`/resume`, and never written to disk. Disk persistence is deferred to the #12
per-session KV format and is not required by this change.

Storing the whole transcript (rather than a truncation offset) is deliberate: it
lets a rollback cross a compaction boundary. The pre-compaction transcript is
reconstructed exactly, regardless of how the live session was rewritten by
compaction in between (issue #29's compaction note).

## KV restore + transcript truncation consistency

Capture takes both halves together: `Engine::snapshot_kv()` serializes the live
session KV at the same instant the transcript is cloned, so the two describe the
same conversation state.

Rollback restores them together: `checkpoint::restore_transcript` rewinds the
`Session` transcript, then the `Agent` hands the stored KV bytes back to
`Engine::restore_kv`. The live session KV now matches the restored transcript,
so the next turn's `ds4_session_sync` common-prefix probe reuses the cached
prefix instead of re-prefilling.

One caveat on "zero" re-prefill: `restore_kv` clears `Ds4Engine::last_reply`,
because the last sampled reply no longer describes the restored tail. The bulk
of the prefix stays cached; only the final assistant turn is re-templated from
text on the next turn (a small re-prefill). Hence "near-zero" in practice.

## Engine boundary

Two defaulted methods were added to the `Engine` trait:

- `snapshot_kv(&mut self) -> Option<Vec<u8>>` — default `None`.
- `restore_kv(&mut self, &[u8]) -> Result<(), EngineError>` — default `Err`.

`Ds4Engine` implements both over the existing `ds4_session_save_snapshot` /
`ds4_session_load_snapshot` FFI (the FINDINGS.md rule applies: a snapshot loaded
from our own bytes uses a non-owning `Ds4SessionSnapshot` that is never freed).

`EchoEngine` uses the defaults, so on an engine without snapshot support a
checkpoint records only the transcript and `/rollback` restores the text and
re-prefills on the next turn — it never panics.

---
name: debug
description: Use when something in this plank session misbehaved - a tool failed with a terse one-liner, the TUI misrendered, a hook did not fire, the engine errored. Reads plank's persistent error log and diagnoses from it.
argument-hint: [issue description]
---

# Debug a plank Session

Diagnose: $ARGUMENTS (if empty, read the log and summarize what is wrong.)

## Where the Detail Is

What surfaces in the session when a tool fails is a single stripped line
(`Tool error: visit_page failed: ...`). Everything useful - the subsystem, the
input it choked on, the untruncated error - is written to
`~/.plank/errors.log` instead. Open that first.

    tail -100 ~/.plank/errors.log
    grep -n "ERROR\|panic\|failed" ~/.plank/errors.log | tail -40

Other sources, by symptom:

- **Model stream looks wrong** (thinking leaking, banners misplaced, tool call
  not detected): run with `--debug` and mirror the raw stream to a
  turbo-debug-console, then compare against what `viz::StreamRenderer`
  produced. `docs/LOOP-FINDINGS.md` catalogs the known looping shapes.
- **Turn-level behavior** (rounds, guards, compaction): `--trace`.
- **Hooks not firing:** start with `/hooks`, which lists what actually
  loaded. Two usual causes: a lifecycle matcher has to match that event's own
  discriminator, and JSON printed by a command hook is thrown away unless the
  keys are ones plank looks for (`additionalContext` and its aliases).
- **Engine refuses to start:** almost always the model lock. Only one process
  can map the weights, and the ds4 engine's answer to a second one is a bare
  `exit(2)`, which is why plank tests the lock first. Look for another plank
  still running.
- **Session or KV weirdness:** everything lives in `~/.plank/kvcache` -
  `<id>.<family>.kv` transcripts, `<stem>.kv_raw` blobs, `<stem>.json`
  sidecars, `<id>.rung-<n>.kv_raw` rungs. Trust the signature written inside a
  blob and nothing else; a sidecar is a hint, not evidence. `/kvcache` walks
  the directory for you.
- **Downloads stuck:** look in `~/.plank/downloads/`, which holds the flock'd
  `lock` that keeps the machine to a single helper and the `cancel` flag file;
  half-fetched artifacts sit under `~/.plank/staging/`.

## Method

1. Read the user's description, then the log. Do not theorize before reading.
2. Find the first failure, not the loudest one - later errors are usually
   consequences.
3. Someone may have paid for this lesson already: `FINDINGS.md` catalogs the
   parity and tooling quirks, and anything about the model looping is kept
   separately in `docs/LOOP-FINDINGS.md`. Read before deriving.
4. Explain in plain language what happened, then propose a concrete fix.
5. If you pinned down a new quirk, offer to add it to the right findings file.

State what you could not determine. A confident wrong diagnosis costs more
than an honest gap.

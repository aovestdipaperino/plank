---
name: debug
description: Use when something in this plank session misbehaved - a tool failed with a terse one-liner, the TUI misrendered, a hook did not fire, the engine errored. Reads plank's persistent error log and diagnoses from it.
argument-hint: [issue description]
---

# Debug a plank Session

Diagnose: $ARGUMENTS (if empty, read the log and summarize what is wrong.)

## Where the Detail Is

Tool failures shown to the model and the user are terse one-liners
(`Tool error: visit_page failed: ...`). The full detail - which subsystem, for
which input, with the complete error text - goes to `~/.plank/errors.log`.
That file is the first thing to read.

    tail -100 ~/.plank/errors.log
    grep -n "ERROR\|panic\|failed" ~/.plank/errors.log | tail -40

Other sources, by symptom:

- **Model stream looks wrong** (thinking leaking, banners misplaced, tool call
  not detected): run with `--debug` and mirror the raw stream to a
  turbo-debug-console, then compare against what `viz::StreamRenderer`
  produced. `docs/LOOP-FINDINGS.md` catalogs the known looping shapes.
- **Turn-level behavior** (rounds, guards, compaction): `--trace`.
- **Hooks not firing:** `/hooks` shows what loaded. Remember lifecycle events
  match on the event's own discriminator, and a command hook that prints JSON
  must use a key plank reads (`additionalContext` and friends) or its output
  is dropped.
- **Engine refuses to start:** the model lock. plank probes it before opening
  the engine because the ds4 engine would otherwise `exit(2)` outright - a
  second instance cannot map the model. Check for another running plank.
- **Session or KV weirdness:** `~/.plank/kvcache` holds `<id>.<family>.kv`
  transcripts, `<stem>.kv_raw` blobs, `<stem>.json` sidecars and
  `<id>.rung-<n>.kv_raw` ladder rungs. A blob's embedded signature is the only
  trust input; the sidecar is advisory. `/kvcache` browses it.
- **Downloads stuck:** `~/.plank/downloads/` - the `lock` file (one helper per
  machine via flock) and the `cancel` flag file; staged artifacts are in
  `~/.plank/staging/`.

## Method

1. Read the user's description, then the log. Do not theorize before reading.
2. Find the first failure, not the loudest one - later errors are usually
   consequences.
3. Check `FINDINGS.md` before deriving a quirk from scratch; it is the catalog
   of hard-won parity and tooling gotchas. Looping goes in
   `docs/LOOP-FINDINGS.md`.
4. Explain in plain language what happened, then propose a concrete fix.
5. If you pinned down a new quirk, offer to add it to the right findings file.

State what you could not determine. A confident wrong diagnosis costs more
than an honest gap.

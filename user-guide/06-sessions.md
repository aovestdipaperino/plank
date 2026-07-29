[← Tools](05-tools.md) · [Index](README.md) · Next: [Context →](07-context.md)

# 6. Sessions

A session is the whole conversation: every message, the tasks, and — where the engine supports it — a snapshot of the model's internal KV state so returning to it does not mean re-reading it from scratch.

Sessions live under `~/.plank/kvcache/` as `<name>.kv`, with a fingerprinted `<name>.payload` sidecar holding the engine state. A session id is a memorable `adjective-celebrity` name minted on first save (`deadly-einstein`), and titles derive from your first prompt, so `/list` is readable rather than a wall of hashes.

## Saving, listing, switching

Sessions save automatically; `/save` forces it.

```
/list               # most recent first
/switch <id>        # load another session
/tag reindex bug    # label this one
/del <id>           # delete
```

`/strip <id>` trims a saved session's oldest turns when one has grown unwieldy but you want to keep the recent part.

## Resuming

```
/resume             # inside plank: the most recent, or a picker
/resume dead        # by name prefix or list number
```

```sh
plank /resume       # straight from the shell
```

A resumed session replays through the same renderer as a live one, so history comes back as rendered markdown with thinking dimmed, not flat text. The KV sidecar is restored alongside the transcript — which is the difference between resuming instantly and waiting for the whole conversation to be re-read. If the sidecar does not match the current model, system prompt, and transcript, it is rebuilt rather than trusted.

## Checkpoints and rollback

A checkpoint is a named return point *inside* a session:

```
/checkpoint before-refactor
… let the model work …
/rollback before-refactor
```

Rolling back restores the transcript verbatim and hands the engine its KV bytes back, so the next turn resumes with almost no re-reading. The tail you discarded is not lost: it is saved as a checkpoint named `pre-rollback`, so a rollback is itself reversible.

Two properties worth knowing:

- A checkpoint stores the **whole transcript**, not an offset. That is what lets a rollback cross a compaction boundary — the pre-compaction conversation is reconstructed exactly, no matter how the live session was rewritten in between.
- Checkpoints are **per-session and in-memory**. They are dropped by `/new`, `/switch`, and `/resume`, and they are not written to disk.

## Branching

A session is stored as a straight line, but a conversation is really a tree: from any earlier prompt you can try a different approach without losing what you already explored.

```
/tree            # show the tree; fork points are numbered
/fork 3          # rewind to just before the 3rd prompt and go a different way
/clone           # freeze this branch and continue on a copy
```

- **`/fork n`** rewinds the live transcript to just before that prompt. Everything after it stays in the tree as a sibling branch, still visible in `/tree` and still reachable by forking again.
- **`/clone`** duplicates the current branch and makes the copy live, so the original is frozen exactly where it stands.

`/tree` collapses linear runs into one line each, so what you see is the fork structure rather than every turn; `*` marks the active branch, and a trailing section numbers the fork points `/fork` accepts.

Fork points are your *real* prompts — tool results do not count, so `/fork 2` means "the second thing I actually asked", which is how you think about it.

Branching costs nothing to keep: the off-path branches are written into the session file as extra records, and a session that never branched is byte-identical to one written before branching existed. Older session files load as single-branch trees.

There is worked-through advice on *when* to fork versus checkpoint versus start fresh in [Advanced workflows](12-advanced-workflows.md).

## Exporting

```
/export                      # markdown, auto-named in the working directory
/export html                 # standalone HTML
/export md notes/review.md   # explicit path
```

HTML output is self-contained — inline CSS, no external assets — and every byte of model and tool content is escaped, since transcripts routinely carry arbitrary code.

## Reproducing a bug

```
/repro
```

writes `~/.plank/repro/repro-<timestamp>.md`: the exact rendered prompt the engine would see, plus the model, backend, context size, sampling settings, think mode, and engine tuning. Hand that to a maintainer and the state that triggered your bug is reproducible without your live session. It is a read-only snapshot; nothing about the running session changes.

## Insights

```
/insights          # full report
/insights fast     # statistics only, no model-written prose
```

Reads every saved session and writes `~/.plank/usage-data/report.html`. Every number is computed deterministically in code; the model is used only for narrative prose. The two halves never mix, so a failed model call costs you the narrative and never the statistics.

---

Next: [Context →](07-context.md)

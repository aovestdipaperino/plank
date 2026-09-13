# Intercepting file writes

Two problems in plank would be easy if the agent could see every write a tool
makes as it happens: the no-progress guard stopping turns that did real work
through the shell, and the `~/.plank` write prompt firing on commands that only
read. This document records what is actually available on macOS, which of the
two problems that reaches, and the design for the part not yet built.

The short version: there is no usable write hook, only after-the-fact
observation. That is enough for the guard, which is why `treedigest` exists and
why it is a git diff and not a kernel callback. It is not enough for the sandbox
prompt, and no amount of engineering makes it enough, for a reason that is
structural rather than incidental.

## What macOS actually offers

Four mechanisms get proposed for this, and three of them are dead ends here.

**Endpoint Security** (`EndpointSecurity.framework`) is the real answer and the
one plank cannot have. `ES_EVENT_TYPE_AUTH_CREATE` / `_WRITE` / `_RENAME` are
genuine authorization events: the client sees the write before it lands and can
allow or deny it. It needs the `com.apple.developer.endpoint-security.client`
entitlement, which Apple grants by application, the binary must be notarized
with that entitlement embedded, and the client must run as root. A locally built
`cargo run` cannot load it at all. This is the correct architecture for a
security product and the wrong one for a coding agent someone builds from
source.

**`DYLD_INSERT_LIBRARIES` interposition** — replacing `open`, `write`, `rename`
in the dynamic linker — is the classic trick and SIP retired it. The variable is
stripped from the environment of any protected binary, which includes every
system utility a model's `bash` call reaches: `/bin/sh` itself, `sed`, `cp`,
`mv`. It would work on `cargo` and on a locally built binary and on nothing
else, which is worse than not having it, because the coverage gap is invisible
and the commands it misses are the common ones.

**DTrace** is SIP-blocked for the same reason, and `fs_usage` requires root and
produces a firehose that has to be correlated back to a process tree.

**FSEvents** (`FSEventStreamCreate`, or `notify` in Rust) is the one that works.
No entitlement, no root, no SIP interaction. It reports that a directory changed,
coalesced, after the fact, with a resumable event id — so a stream opened once
at startup can be asked "what changed since this id" at any later point. It is
not interception. That distinction decides everything below.

## Why the sandbox prompt cannot be fixed this way

`sandbox::mentions_plank_home` matches the command *text*, and its doc comment
already states the constraint: the Seatbelt profile is built before the command
runs, so there is no write to observe yet. A prompt asked after the write has
happened is not a prompt, it is a notification. FSEvents is after the fact by
construction, so it arrives too late by exactly the margin that matters.

The tempting workaround is optimistic execution: run under the default profile
with `~/.plank` unwritable, detect the kernel's denial, then prompt and re-run.
It is wrong for arbitrary shell. The denial arrives partway through a command
that has already had other effects, and the re-run repeats them. `cp a b && rm
a` denied on the second half does not survive being run twice.

The real fix for the false alarms is unrelated to hooks and much smaller.
`sandbox::is_read_only_command` is deliberately conservative — anything it does
not recognize keeps the prompt, so a miss costs one question and never a silent
grant, and the Seatbelt profile stays the boundary either way. The false alarms
are misses in that allowlist. Widening it (`jq`, `wc`, `stat`, `file`, `find`
without `-delete`/`-exec`, `sed` without `-i`, `git log`/`show`/`diff`) removes
most of them with a contained, testable change. The sharper version is to stop
matching command text at all and match resolved write targets — parse redirects
and the mutating flags of known utilities — but the allowlist gets most of the
benefit first.

## What was built: `treedigest`

The guard only ever asks its question once the tool call has returned, so
observation is all it needs. `src/treedigest.rs` fingerprints the git
working-tree state — `HEAD`, plus each dirty entry's path, status bits, length
and mtime — and `tools::dispatch` takes one fingerprint before and one after
every tool in `OPAQUE_MUTATOR_TOOLS` (the `bash` family, `run_code`, MCP and
WASM tools). A difference sets `ToolContext::touched_tree`, which the
no-progress guard reads alongside `last_written`.

Three properties are load-bearing:

- **Difference, not exit status.** A successful `cargo test` moves nothing and
  resets nothing. This is what `docs/LOOP-FINDINGS.md` ("An attempted mutation
  is not progress") was protecting, and it survives intact.
- **`write` and `edit` are excluded.** They report their own writes precisely;
  a digest would be a slower second opinion on a question already answered.
- **No repository, no witness.** `capture` returns `None`, `changed` reads a
  `None` on either side as "nothing observed", and the guard falls back to
  `last_written` exactly as it did before.

The stat is not decoration. A status-only digest misses an edit to a file that
was *already* dirty: the entry reads `WT_MODIFIED` on both sides, and only the
length and mtime separate them. That case has a regression test.

The cost is one `git status`-shaped walk per opaque call, with
`recurse_untracked_dirs(false)` and `no_refresh(true)` — the latter because a
guard has no business writing the index from inside a tool dispatch while the
user's own git may be running.

## The FSEvents widening, if it is ever needed

`treedigest` inherits git's blind spots. Ignored paths do not count (correct:
`target/` is not the work), untracked directories are not recursed, and a
directory with no repository has no witness at all. If those turn out to matter
— the likely trigger is someone running plank productively outside a checkout —
FSEvents closes them.

The shape: open one stream over `Sandbox::write_roots` at startup, keep the
latest event id, and have `dispatch` ask for the changed-paths set between the
id before the call and the id after. `notify` (not currently a dependency) wraps
this; a raw `FSEventStreamCreate` with `kFSEventStreamCreateFlagFileEvents`
avoids the dependency at the cost of writing the CoreFoundation run-loop
plumbing.

Two things would need care, and they are the reason this is documented rather
than built:

**Coalescing latency.** FSEvents batches with a configurable delay, so a write
that lands just as the call returns can be reported after the digest would have
been taken. That misattributes it to the *next* call, which for a progress
guard is a missed reset rather than a false one — tolerable, but it means the
id-range read has to be a high-water mark, not a snapshot, and the boundary
needs a test that is not timing-dependent.

**Filtering.** The stream sees plank's own writes to `~/.plank` — session
transcripts, KV blobs, the spill store — on every turn. Those are not the
model's progress and would reset the budget unconditionally, which would quietly
disable the guard. Any FSEvents implementation has to exclude the plank home and
every build directory before it counts anything, and getting that exclusion
wrong fails in the silent direction: the feature looks fully wired up and the
guard never fires again.

That last risk is why the git digest went first. It gets the common case, it
cannot accidentally count plank's own bookkeeping, and its failure mode is a
missed budget reset rather than a guard that has stopped working.

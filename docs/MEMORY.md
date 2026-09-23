# Persistent memory

plank can remember things about you and about a project across sessions, and
it can maintain that memory itself instead of relying entirely on you typing
`/remember`. This document explains what gets saved, how it is stored, why it
is stored that way, what each part costs, and every setting that controls it.

Status: shipped. The `remember`/`forget` tools are on by default
(`tools.remember: true`), and since 5.1.7 so is the automatic extraction pass
(`memory.autoExtract: true`). The pass is not free: each run stalls the front
end for a KV snapshot, a prefill and a generation after the answer — see "The
extraction pass" and "Cache accounting" for what it costs and
`memory.extractEveryNTurns` for how to thin it, or set `memory.autoExtract` to
`false` to turn it off. The design
choices below all trace back to one constraint: a memory rewrite must never
force a mid-session KV re-prefill. Part 1 covers why. The rest is mechanics.

## Part 1: why writes land on disk instead of in the conversation

`docs/KV-CACHE.md` explains plank's KV caching in full; the summary that
matters here is that plank's context is layered into tiers so that most of a
long conversation's prefill work is reused turn to turn. `AGENTS.md` and
persistent memory sit together in Tier 2, the **project-stable** tier
(`ContextContent::stable_context` in `src/context.rs`): content that rarely
changes, cached as `project.kv` and keyed by a hash of exactly that content
(`stable_hash`). Git status and the date sit in Tier 3, volatile, prefilled
fresh every session because they always differ.

That placement is what makes memory cheap: on a local Metal backend, prefill
is the slow phase, and reusing the Tier 2 prefix is what keeps a long session
from re-reading its own history every turn. It also means memory cannot
change mid-session without invalidating that prefix. If a `remember` tool
call spliced its text into the live context, every following turn would pay a
full re-prefill of everything after it — on `refs/ds4`'s Metal backend, the
single most expensive operation in the system.

So writes never touch the live context. `remember`, `forget`, `/remember`,
`/forget`, and the automatic extraction pass all write to the memory files on
disk and nothing else. The new or changed content joins the model's context
only the next time `ContextContent::new_with_agents` runs and rebuilds Tier
2 — in practice, the next session start. The tool's own observation text says
so directly ("it joins your context at the next session start"), because a
model that just called `remember` and sees no visible effect needs to be told
that is expected, not a bug.

This is also why retrieval-shaped work — ranking, budgeting, eviction — is
designed to run once at render time rather than continuously: it produces a
static block of text for that tier, not a live view that has to stay in sync
turn by turn.

## Part 2: what memory is

### Files

Memory is two flat markdown files, layered the same way as plank's other
`.plank` configs:

| Scope | Path | What belongs there |
|---|---|---|
| User | `~/.plank/MEMORY.md` | Follows you across every project: who you are, durable preferences, corrections that apply everywhere. |
| Project | `<cwd>/.plank/MEMORY.md` | Tied to this checkout: goals and constraints specific to the project. |

Both are loaded at session start (`memory::load_default`) and injected into
Tier 2 context ahead of the first user turn, user scope first. Either or both
files can be absent; `load_default` returns `None` only when neither has
renderable content, in which case no memory section appears at all.

### The four entry types, and what must never be saved

Every entry carries a type, written as a `[tag]` right after the date:

- **`[user]`** — who the user is: role, expertise, standing preferences.
- **`[feedback]`** — corrections and confirmed approaches on how to work.
- **`[project]`** — goals and constraints not derivable from the code.
- **`[reference]`** — pointers to external URLs, tickets, dashboards.

The taxonomy is deliberately closed (`memory::Kind`) and there is no
catch-all fifth type, because the discipline that keeps a memory file useful
is what it refuses to hold. Nothing goes in that the model — or a person
reading `AGENTS.md` — could reconstruct more accurately by looking: code
patterns, architecture, file paths, git history, debugging recipes, or
anything already stated in `AGENTS.md`. A memory file that accumulates
re-derivable facts rots the same way any cache rots when nothing ever
invalidates it: it goes stale, and by the time someone notices, several other
entries near it are stale too and nobody trusts the file. The extraction
pass's own prompt states this rule to the model in those terms; see Part 3.

### The file format, and that untagged entries still parse

A memory file is a template header followed by dated bullets:

```markdown
# Memory

Durable notes loaded into every session start. Keep entries to facts that
cannot be re-derived from the repository. Types: [user] who the user is,
[feedback] corrections on how to work, [project] goals and constraints,
[reference] external URLs/tickets/dashboards.

- (2026-07-19) [user] prefers tabs over spaces
- (2026-07-20) [feedback] don't force-add generated docs
- (2026-08-01) [project] target is macOS only for real inference
```

`parse_entries` recognizes a line as an entry purely by its `- (DATE)` prefix;
everything else (the header, blank lines, prose) is skipped, and a hand-typed
note that isn't a bullet is simply invisible to memory without breaking
anything.

A bullet written or edited without a `[type]` tag still parses: it falls back
to `Kind::Project`. This is the same code path every entry goes through, so
a memory file written by an older plank build, or hand-edited without the tag
syntax, is read exactly like a freshly tagged one. What it is *not* is a
promise that such a file renders as it did before this feature existed; see
"What an old file gets after the upgrade" below for what actually changes.

A bracket that isn't one of the four recognized tag words (say, a
hand-written `[WIP]`) is not stripped or treated as a parse error; it stays
in the entry's text verbatim, and the entry reads as untagged. Silently
deleting text a person typed because it happened to look like a tag would be
worse than leaving it alone.

### What the model actually sees

The text injected into Tier 2 is not the file. `load_scope` regroups the kept
entries under one `### <type>` heading per type, in the fixed order `user`,
`feedback`, `project`, `reference`, and writes each bullet back out with its
id inserted after the tag:

```markdown
### project
- (2026-01-01) [project] {3f9a1c02b7de} an old untagged fact
```

The `{id}` is there so the model can name an entry in a `forget` call or an
extraction verdict without quoting its text back; the headings make the
per-type grouping the budgets work on visible to the model. Neither is
written to `MEMORY.md`, which stays a plain list of dated bullets a person
can edit.

### The sidecar, and why it is advisory

Each memory file has a companion `<file>.meta.json` (`memory::meta_path_for`)
holding per-entry bookkeeping, keyed by [`Entry::id`](#entry-identity):

- `uses` — how many extraction passes judged the entry to have borne on the
  work.
- `last_used` — the date of the most recent such pass.
- `pinned` — ranks the entry ahead of every unpinned one in its type, so it
  is the last to be evicted. **Reserved, not yet settable**: `MetaStore` can
  read and write the flag, and the ranking honours it, but nothing in plank
  sets it — there is no `/memory pin`, no tool argument, and no verdict for
  it. Today the only way to pin an entry is to edit the sidecar by hand and
  write `"pinned": true` on its row.

A sidecar written by an earlier build of this branch may also carry a
`retracted` key, from a since-dropped design in which a model `forget` only
hid an entry. `MetaStore::load` ignores any key it does not know, so such a
file loads normally and the stray key disappears at the next save.

Every field defaults to zero/false, and `MetaStore::load` treats a missing
file, an unreadable file, malformed JSON, or the wrong shape identically: an
empty store. Nothing in `MEMORY.md` itself is ever machine-owned — a person
can edit, reorder, or delete lines by hand, and the worst that happens is an
orphaned sidecar row, cleaned up the next time anything calls `MetaStore::gc`
against the live entry ids — the pass's `apply_verdicts` and both `forget`
paths do; a plain `remember` does not touch the sidecar at all. Losing the
sidecar file
entirely — deleted, corrupted, never created — must leave memory loading and
rendering exactly as if every entry had a fresh row: every counter at zero,
nothing pinned, nothing missing. That correctness is what "advisory" means
here, and it is worth stating explicitly because it is easy to design a cache
that quietly becomes load-bearing; this one is tested not to.

#### Entry identity

`Entry::id()` is the first 12 hex characters of a SHA-256 hash of the
entry's text alone — not its date, not its type tag. That is deliberate:
re-tagging or re-dating a line keeps the same id, so it keeps its accumulated
usage. The consequence is that rewriting an entry's *text* — which is exactly
what an `UPDATE` verdict from the extraction pass does — changes its id, and
something has to carry the old row's counters onto the new one or a fact
reworded six times over a month would reset to zero usage every time.
`MetaStore::carry(old_id, new_id)` is that something; `apply_one_verdict`'s
`Update` arm calls it in the same step that rewrites the line.

Because the id is a hash of the text, two entries with identical text in the
same file are one id: an `ADD` whose text already exists is a no-op, and a
`forget` of that id removes every line carrying it.

### Budgets and eviction

Rendering is governed by per-type budgets (`memory::Budgets`), not a single
file-level cap — a runaway `project` block cannot silently crowd out the
`user` block the way one shared budget would let it:

| Type | Default budget |
|---|---|
| `user` | 4096 bytes |
| `feedback` | 4096 bytes |
| `project` | 6144 bytes |
| `reference` | 2048 bytes |

The unit is **bytes**, not characters: `select_for_render` charges each entry
`Entry::render().len()`, the UTF-8 length of its canonical
`- (date) [type] text` line. For ASCII the two coincide; an entry written in
a multi-byte script costs more than its character count. The accounting is
also slightly optimistic: what is charged is the canonical line, while what
is injected (see "What the model actually sees") carries an extra `{id} ` per
entry and a `### type` heading per block, none of which is counted. A block
that exactly fills its budget therefore overruns it by roughly 15 bytes per
entry plus the heading, which is why the budgets are best read as "about
this much", not a hard cap on the injected bytes.

Within a type, entries are ranked **pinned first, then by descending `uses`,
then by most recent `last_used`, then by most recent date, then by position
in the file (later wins)**, and kept in that order until the budget is spent;
whatever doesn't fit is reported as dropped, not silently discarded —
`load_scope` appends a line noting how many lower-ranked entries were omitted
under the type budgets, so nothing disappears without a trace. The kept
entries are then emitted in file order, so the ranking decides *what*
survives, not the order the model reads it in. The date and position
tiebreaks matter most on a file that has no sidecar yet: every entry ties on
the counters, and without them the sort would keep the *oldest* lines — the
exact inverse of the tail truncation this replaces.

The budgets can be changed only by hand, in the `memory.budgets` object of
`settings.json`; they are not in the `/config` form and `/config` does not
write them back (it rewrites only the keys it owns, so a hand-set budget
survives a `/config` save).

### What an old file gets after the upgrade

A memory file written before this feature — untagged bullets, no sidecar —
is read by the same parser as a new one, and with the settings at their
defaults **no counter is ever bumped and no pass ever runs**, so the sidecar
stays empty and ranking degrades to "newest first". Three things do change
for that file, and none of them is a setting:

1. Every entry renders as `[project]` under a `### project` heading, with
   an `{id}` after the tag, instead of as the raw line.
2. The old rule — inject the newest 16 KiB of the file and drop whatever
   older text lay above it, behind an `(older entries truncated)` line — is
   replaced by the `project` budget, 6144 bytes by default, applied to the
   canonical lines. The survivors are still the newest entries (see the
   tiebreaks above), but a legacy file between 6 KiB and 16 KiB that used
   to render whole now loses its oldest entries and gains the `omitted`
   line. Raising `memory.budgets.project` by hand restores the old
   capacity.
3. An unrecognised `[tag]` is kept in the text, as described above.

The test `an_untagged_legacy_file_renders_every_entry_when_nothing_is_configured`
in `src/memory.rs` pins the narrow guarantee that holds: a small untagged
file at default settings renders every entry and emits no `omitted` line. It
does not, and cannot, show that the injected text is byte-identical to the
previous behaviour — it is not, per points 1 and 2.

## Part 3: writing memory

### The `remember` / `forget` tools

Advertised to the model after the parity-frozen prompt region
(`sysprompt.rs`), gated on `tools.remember` (default `true`; setting it to
`false` removes both tools from the model's tool table, returning `unknown
tool` if called anyway):

- **`remember(text, type, scope)`** appends a dated, tagged bullet to the
  named scope's file (default `project`). It writes immediately and logs the
  change under the reason `remember tool`, but — per Part 1 — the entry is
  not visible in context until the next session start; the tool's own reply
  says so.
- **`forget(id)`** deletes the entry with that id from whichever scope holds
  it, through the same atomic, audited write path as `/forget`
  (`memory::forget_by_id_to`, sharing `forget_where_to` with
  `forget_matching_to`). There is no hidden "retracted" state: what is in
  `MEMORY.md` is what the model sees at the next start, and what the user
  reads in their own file is live. Recoverability comes from the audit log
  instead — the `~/.plank/memory-log.jsonl` line records the entry's full
  text under the reason `forget tool`, distinct from a user's `/forget`
  (`user /forget`), so a wrong model call can be found in `/memory log` and
  re-added by hand.

Either tool call also suppresses the extraction pass for that turn (see
below): the model's explicit judgement wins over a second reading of the same
turn.

### `/remember` and `/forget`

The user-typed equivalents, on both front ends:

- `/remember [user] <text>` — default scope is `project`; prefixing `user `
  writes to the user scope instead. Writes through `memory::remember`
  directly and is **not** logged to the audit log; only the tools and the
  pass write there.
- `/forget <pattern>` — a case-insensitive substring match against every
  entry's text in both scopes. The command previews every entry that would
  be removed and asks before touching anything (a yes/no panel in the TUI, a
  `[y/N]` prompt on the plain REPL, declined automatically when stdin is not
  a terminal), then deletes through `forget_matching`, logging each removal
  under `user /forget`.

### The extraction pass

The passive half of memory maintenance: a pass that reads the turn's
transcript and proposes changes, without the model having to think to call
`remember` itself. **On by default since 5.1.7**, so what follows is what
you are paying for unless you turn it off.

**When and where it runs.** In two steps. At the end of a turn that
produced a final response with no tool calls (`Agent::enqueue_memory_job` in
`src/ui.rs`, called from `run_turn` on the plain path and `worker_turn` in
the TUI), the span since the last pass is **snapshotted** into a
`MemoryJob`: the complete extraction prompt, built from the transcript as it
stands, pushed onto a queue on the `Agent`. Nothing is generated at that
point, so the prompt comes back the moment the answer is done, the Stop
hooks fire right then, and the idle window title is set last of all. The
span is retired (`ExtractState::finish`) as it is captured: the job carries
everything it needs, so the next turn opens a new span from there and a
`/clear`, a follow-up turn or a compaction between the turn end and the
reading cannot lose what was captured. A turn the user interrupted (Esc,
Ctrl-C) queues nothing, and its span is left for the next turn that
completes (`Agent::memory_pass_allowed`).

The **reading** happens at the next idle moment (`Agent::process_memory_job`,
one job per call, oldest first). In the TUI that idle moment is now shared
with prompt suggestions (`suggest.rs`, `docs/superpowers/specs/2026-09-23-prompt-suggestions-design.md`):
`Agent::idle_work` (`crate::suggest::idle_work`) decides which of the two
queued background jobs the slot goes to. A pending suggestion wins by
default — a suggestion that lands after the user starts typing is wasted,
while a deferred memory pass is explicitly tolerated
(`memory.minTurnSeconds`) — but `suggestions.memoryStarvationSeconds`
(default 300s) hands the slot back to the oldest queued memory job once it
has waited that long, so a fast back-and-forth cannot lock memory extraction
out forever. Once the slot is granted to the memory pass, reading proceeds
exactly as before: the idle loop's poll timeout (`tui_memory_pass`), with
the same guards as the background-job wake — no draft in the editor, no
modal pane — so a pass never starts under a keystroke. It runs on a worker
thread behind the same busy UI loop as a turn, and typing keeps working.
Its only trace while it runs is a `✍️` in
the footer (`Status::memory_pass`, `status::MEMORY_MARK`) in place of the
state word — one mark per queued span, the running one included, so
`✍️✍️` means one more is waiting — and, only with `--show-memory-stats`, the bare figures of its phase
(`↑ 3.3k/4k tokens · 392 t/s`, then `↓ 12 tokens · 20 t/s`) floated at the
right end of the rule below the prompt, where a turn's figures go; without
the flag the rule stays plain while notes are taken: no throbber or verb, no progress
line under the output, no scrollback line, no window title change, and the
JSON reply itself is never rendered (`sub_sink_render_sink` is null for the
pass), because this is housekeeping the user did not ask for, and the
conversation should not fill with it. **A prompt submitted during the pass interrupts it**
(`TurnShared::memory_pass` makes the busy loop raise the worker interrupt
as it queues the line): the job goes back to the front of the queue,
uncounted, and the typed line starts its turn at once — the user never waits
for the notes. Ctrl-D on an empty prompt during the pass quits plank, exactly
as it does at an idle prompt: the pass stops at its next token and the queued
job is dropped. Mid-*turn* Ctrl-D remains inert — nobody should lose a
generation in flight to a stray keystroke. `/btw` is refused during a pass, with a hint to just type the
prompt, because there is no main task to ask beside. The plain REPL reads
one job per 250 ms idle tick of its stdin wait (`run_repl_plain_local`); it
cannot cut a generation short on keystrokes, so a line typed meanwhile
waits out the job that is running. The headless paths have no idle loop to
come back to, so they drain the queue synchronously before exit
(`drain_memory_jobs`); the stdin protocol also reads one job per idle tick.

Each reading is one sidechain (`begin_sidechain(prompt, true)` …
`end_subagent_fork`), which means, in order: a snapshot of the whole
session's KV (`engine.get_kv()`), a prefill of the pass prompt on top of the
live context, one generation with no tool dispatch (`run_memory_round` — the
sidechain can never touch a tool), and a KV restore back to the parent
prefix. None of that time or those tokens appears in any turn's stats: the
traces are the footer mark and one dim `memory completed in 1m 12s` line
when the pass succeeds, carrying the change summary when it changed
something. A pass whose reply held no verdicts is silent: it is
recorded in the sidechain dump (`/repro`) and nowhere else. The sidechain
runs under `in_sidechain()`, so it pushes no KV ladder rungs and stores no
payload: it leaves no checkpoint debris.

**Not from zero after an interrupt.** On a local engine the pass first
prefills the prompt alone (a `n_predict: 0` generation), snapshots that KV
into the job (`MemoryResume`: the KV plus the exact prompt text it covers),
and only then samples. An interrupted attempt keeps the snapshot; the retry
restores it (`engine.set_kv`) and re-issues the stored prompt byte for byte,
so it prefills only the assistant prefix. An interrupt that lands *during*
the prefill — the long phase on a local model, so the common case — keeps a
snapshot too: the engine stops at a token boundary and leaves a valid
shorter prefix behind, and the retry continues the prefill from there rather
than from zero. The order is forced by the
engine: a live KV is reused only when the prompt *extends* it, and a prompt
that is a strict prefix rebuilds from zero, so a snapshot taken at the
interrupt (which would hold the partial reply) would be worthless. Engines
with no KV to snapshot (providers, the echo stub) skip the prefill-only
step as well. The snapshot is held in memory only while the job waits and
goes with the job when it leaves the queue.

**How a job leaves the queue:** applied; unusable (a reply that is not a
JSON array is a property of the model on this prompt, not a transient
fault, so it is not retried); too big for the remaining context (dropped,
with a one-time notice); or after `MAX_JOB_ATTEMPTS` engine errors, which
is what keeps a broken engine from pinning the idle loop on one span.

**The snapshot is gated by five checks, in this order** (`ExtractState::should_run`;
`enqueue_memory_job` also refuses to run while already inside another
sidechain, so a sub-agent's turn never triggers it):

1. **Enabled, and mutual exclusion.** `memory.autoExtract` must be on, and
   the model must not have called `remember` or `forget` itself this turn.
   A model write suppresses the passive pass for that turn only: the
   model's own judgment wins over a second reading of the same turn, and
   the suppression does not carry into the next turn. The flag is consumed
   at the top of the function, before any other check, so a turn cannot
   stay suppressed because a later gate returned first.
2. **Depth keying.** The pass tracks `processed_depth`, the transcript
   length its last completed run covered, and fires only when the current
   depth has grown past it.
3. **Turn duration floor.** The turn must have taken at least
   `memory.minTurnSeconds`. A turn that misses the floor does not advance
   `processed_depth`, so its span is deferred rather than discarded: the
   next turn that clears the floor reads it too.
4. **Not already running.** A trigger that arrives while a pass is in
   flight is dropped outright. Nothing is recorded about it, because
   nothing needs to be: once `finish` clears the flag and advances
   `processed_depth`, the next call re-derives the span from
   `processed_depth` against the then-current depth, which necessarily
   covers everything that arrived meanwhile, in one trailing run.
5. **Throttle.** The pass runs only every `memory.extractEveryNTurns`
   *eligible* turns.

The order is what makes "eligible" mean something. The floor sits above the
"already running" check and the throttle counter, so a turn skipped for
suppression, for having nothing new, for missing the duration floor, or for
arriving mid-pass does not count against the throttle. Only a turn that
genuinely had new transcript to look at, cleared the duration floor, and
could have been acted on, advances the count.

**The invariant that makes depth keying correct, stated the way the code
states it:** a pass must read only the transcript above its recorded depth,
and getting that wrong degrades silently into reprocessing everything rather
than failing loudly. There is no error state for "read too much" — it just
means every pass after the mistake re-derives verdicts for text it already
saw, wasting the model's time and risking duplicate or contradictory
verdicts, with nothing in the logs pointing at the cause. `ExtractState`
guards this from both directions: `finish(depth)` clamps `processed_depth` to
never move backward even if called with a stale depth, and transcript
rewrites re-anchor it — `rebase` after compaction, `truncate_to` after a
rollback or fork end, `reset_to` on `/clear`, `/new`, `/resume` and
`/switch`, so a restored session is never shipped wholesale to the model on
its first idle turn.

**What the pass reads.** The prompt (`memextract::build_prompt`) is the
current entries of both scopes with their ids, the verdict contract, and an
excerpt of the transcript above `processed_depth` (`render_excerpt`). The
excerpt is capped at 32 KiB (`EXCERPT_MAX_BYTES`), tool-result bodies are
replaced by a one-line size placeholder, and when the span is still too long
the *oldest* messages are dropped first with a note saying how many. Before
the KV snapshot is taken, the prompt is preflighted against the context
headroom the session leaves (`last_ctx_used` plus the prompt plus
`REPLY_RESERVE_TOKENS`, 1024); a span that does not fit is retired unrun with
a one-time notice, because nothing about it would shrink on a retry.

**How a pass ends.** Three outcomes, and they advance the depth differently:

- A clean reply, whether or not it holds verdicts, retires the span
  (`finish`). A reply that is not a JSON array — prose, or a tool call —
  also retires it, with a one-time "unusable reply" notice and a repro dump:
  re-reading the same span every idle turn would cost a growing prompt
  forever and never do better.
- An engine error or an interrupt cancels (`ExtractState::cancel`) without
  advancing `processed_depth`, so the unprocessed span is picked up by the
  next eligible turn.
- Nothing is ever half-applied: verdicts are parsed and applied only after
  the sidechain has been folded back out of the transcript and the KV
  restored.

**Verdicts.** The pass replies with a JSON array of verdicts
(`memory::parse_verdicts`), each one of:

- `ADD` — a brand-new entry, with its type and scope. Logged as `add`,
  reason `extracted`.
- `UPDATE` — supersede an existing entry's text (carries usage via
  `MetaStore::carry`, see "Entry identity" above). Logged as `update`,
  reason `reconciled`.
- `DELETE` — remove an entry outright, because it's redundant or wrong. Same
  audited outcome as a model `forget`, reached from the pass's verdicts
  rather than a tool call. Logged as `delete`, reason `reconciled`.
- `USED` — credit an existing entry with having borne on the work in this
  excerpt, bumping its `uses`/`last_used`. Logged as `used`, reason
  `reused`.

Every verdict naming an id with no live matching entry is silently
discarded: the file always wins, because a person may have hand-edited
`MEMORY.md` between the pass reading it and this write landing. Within one
batch, verdicts apply in order against the file as rewritten so far, so a
`USED` or `DELETE` that names the *pre-update* id of an entry an earlier
`UPDATE` in the same batch rewrote no longer finds it and no-ops.
`apply_verdicts` / `apply_verdicts_to` is the single write path for the
pass, and it stages audit-log lines and sidecar mutations in memory while
walking a batch, flushing both only after the scope's file write has actually
succeeded: a log that says an entry was added when the write failed would be
worse than no log.

### The System-1 gate

Behind `memory.gate` (default off), `enqueue_memory_job` asks one extra
question before a job is ever built: is this span worth extracting at all?
The check sits at `enqueue_memory_job`, after `should_run` has already
decided the turn is eligible and before the span is turned into a
`MemoryJob` — never at `process_memory_job`, which only ever reads what was
already queued. A `Yes` (or the gate being off, or the check being
bypassed) falls through to the same `build_prompt` / `finish` / enqueue path
described above; a confident `No` calls `ExtractState::reject` instead and
returns without queuing anything.

The question runs on `Engine::decide` (see `decide.rs` in `CLAUDE.md`'s
architecture list), which prefills the rendered excerpt plus a short
boolean question onto a **dedicated decision session** and reads the
logprobs of the answer letters — no tokens generated, no tool call, and
critically no contact with the live turn session: the decision session is
created lazily and torn down independently, so this check never rewinds or
otherwise disturbs the KV rung ladder or the fingerprinted live prefix.

Every uncertain outcome runs the pass rather than skipping it
(`memory_gate_says_worthy`): an engine that does not `supports_decide()`,
a `decide` that returns an error, and an abstained verdict all read as
"worth extracting". The gate can only suppress a pass it is *confident*
found nothing; it must never be the reason a memory is lost. The pass runs
only on a non-abstained `Worthy::Yes` whose probability clears
`memory.gatePercent`; everything else — a confident `No`, or a `Yes` the
model is not sure enough about — suppresses it.

One cost to know before turning the gate on: the decision is taken
*synchronously at turn exit*, in `enqueue_memory_job`, whose whole design
property otherwise is that it only snapshots and generates nothing, so the
prompt comes back the moment the answer is done. Asking the gate means
prefilling the span excerpt — up to `EXCERPT_MAX_BYTES` (32 KiB) — on the
decision session before the prompt returns. The excerpt is a rolling tail,
so the session's longest-common-prefix reuse will often miss and the prefill
is paid in full. It is still far cheaper than the generation it avoids, but
it is not free, and it lands in the one place the design says nothing
happens. This is a large part of why the gate ships off.

Three settings govern it, all under `memory` in settings.json:

- **`memory.gate`** (`bool`, default `false`) — the gate is off by default;
  every eligible turn's span goes straight to the extraction pass exactly
  as before this feature existed.
- **`memory.gatePercent`** (`u32`, default `60`) — the confidence threshold,
  as a percentage, a `Yes` verdict's probability must clear to count as a
  rejection.
- **`memory.heldSpanCap`** (`u32`, default `0`) — see below.

**What a rejection does to the span** depends on `heldSpanCap`. With the
shipped default of `0`, `ExtractState::reject` calls `finish`: the span is
retired outright, exactly like a completed pass, and its content is
permanently discarded — never re-read, never re-judged. With a positive
cap, `reject` calls `cancel` instead: `processed_depth` does not advance,
so the span stays unread and is folded into the next eligible turn's span
for a fresh judgment — unless by then it has grown past the cap
(`ExtractState::gate_bypassed`), in which case the gate is skipped
entirely and the pass runs unconditionally, so a span can never be held
forever.

### `/memory` and `/memory log`

- **`/memory`** opens every source in one editable buffer (`memory::combine`
  / `memory::apply`), marked with `<!-- plank-memory: begin/end SCOPE -->`
  comments so edits route back to the right file on save. Editing requires
  the interactive TUI's built-in editor; the plain-stdout REPL prints the
  same combined view read-only, followed by a line pointing at the TUI and
  `/memory log`.
- **`/memory log`** prints the last 20 lines of the audit log
  (`~/.plank/memory-log.jsonl`), oldest first, human-rendered, on both front
  ends. Every write by the tools (`remember`, `forget`), by `/forget`, and by
  the extraction pass's verdicts is appended there as one JSON line with five
  fields: `action` (`add`, `update`, `delete`, `used`, `forget`), `scope`,
  `id`, the entry's full `text`, and `reason`. The `reason` is **a fixed
  label chosen by the code path** — `remember tool`, `forget tool`,
  `user /forget`, `extracted`, `reconciled`, `reused` — that says *which
  mechanism* made the change. It is not the model's rationale, which is
  never captured: the pass replies with bare verdicts, and `forget` takes
  only an id. So `/memory log` answers "what changed, by which hand, and
  what did the text say", which is enough to re-add a wrongly removed entry,
  but it cannot answer "why did the model think this was wrong". It is a
  write-only trail: nothing reads it back except this command. Writing to it
  is best-effort — a failed append never fails the change it describes.

## Part 4: settings

All under the `memory` and `tools` blocks in `~/.plank/settings.json` /
`.plank/settings.json` (`src/settings.rs`):

| Setting | Default | Effect |
|---|---|---|
| `memory.autoExtract` | `true` | Whether the extraction pass runs at all. On, every eligible turn ends with a synchronous stall for a KV snapshot, a prefill of the excerpt, a generation and a restore, none of it counted in the turn stats. Off leaves the `remember`/`forget` tools and `/remember` working — only the passive pass stops, and the sidecar counters are never bumped, so eviction ranks by date alone. Also in the `/config` form. |
| `memory.extractEveryNTurns` | `1` | Run the pass every N *eligible* turns (a turn with no tool calls and no model `remember`/`forget`). `1` means every eligible turn. A configured `0` is clamped to `1`. Also in the `/config` form. |
| `memory.minTurnSeconds` | `120` | A floor on how long a turn must have taken for it to trigger the pass. The pass costs a KV snapshot, a prefill, a generation and a restore; a four-second exchange rarely produces anything worth that. Set it to `0` to remove the floor. A short turn **defers** its span rather than discarding it. The pass reads the transcript from the depth the last completed pass recorded, and a turn under the floor does not move that depth, so the next turn that does clear the floor reads the short turns too. Nothing said to the model is lost to this gate — the only effect is when the reading happens. The consequence to be aware of is the mirror image: a session made entirely of short turns accumulates an unread span and extracts nothing until one long turn arrives. The floor is checked before the `extractEveryNTurns` counter, so a short turn does not count as an eligible turn either: `extractEveryNTurns: 3` means every third turn *worth* extracting from. |
| `memory.budgets.user` | `4096` | Byte budget for `[user]` entries (see "Budgets and eviction" for what is counted). Hand-edit only. |
| `memory.budgets.feedback` | `4096` | Byte budget for `[feedback]` entries. Hand-edit only. |
| `memory.budgets.project` | `6144` | Byte budget for `[project]` entries, including every untagged legacy entry. Hand-edit only. |
| `memory.budgets.reference` | `2048` | Byte budget for `[reference]` entries. Hand-edit only. |
| `tools.remember` | `true` | Whether the `remember`/`forget` tools are advertised to the model at all. Flipping it changes the system prompt and so churns the `fp1` fingerprint once. `/remember` and `/forget` are unaffected — they are user-typed commands, not model tool calls. |
| `memory.gate` | `false` | Whether the System-1 gate (see above) runs before a span is enqueued. Off by default — no behavior change from before the gate existed. |
| `memory.gatePercent` | `60` | The confidence threshold, as a percentage, a `Yes` verdict must clear to reject a span. Out-of-range values are clamped to 0–100 rather than rejected. |
| `memory.heldSpanCap` | `0` | Transcript-depth span size past which the gate is bypassed and the pass runs unconditionally. `0` (the default) means a rejected span is finished outright rather than held for re-judging. |

## Cache accounting

How memory's cost is actually paid, tying together Part 1's placement, Part
3's pass and Part 4's budgets — the numbers that matter for judging whether
memory is cheap in practice:

| Event | What gets prefilled | Tier |
|---|---|---|
| First turn of a session | Full Tier 2 (`AGENTS.md` set + memory + agent roster) plus Tier 3 (git status, date) | Tier 2 cached to `project.kv` keyed by `stable_hash`; Tier 3 always fresh |
| Every later turn in the same session | Nothing from memory — it is part of the already-cached prefix | Reused |
| A memory file edited (`/remember`, `/forget`, the tools, the pass, a hand edit) | Nothing, until the *next session* | The new content changes `stable_hash`, so the next session's first turn re-prefills Tier 2 once |
| `/memory` opened and saved with no actual change | Nothing | `apply` only writes files whose body changed; an unchanged section reports `unchanged` and touches no cache |
| The sidecar (`MEMORY.md.meta.json`) changing | Nothing, ever | It is never part of any prompt tier — only `MEMORY.md`'s own text is |
| **The extraction pass** (`memory.autoExtract` on), once per eligible turn | The pass prompt — current entries plus up to 32 KiB of excerpt — on top of the whole live session, then one generation of up to `REPLY_RESERVE_TOKENS` | Sidechain: `get_kv()` of the full session before, restore after, so the live prefix is unchanged for the next turn. **Unmetered**: it runs after `fire_turn_end`, so its time and tokens are in neither the turn stats nor `/toks`. |

The number that governs memory's *resident* prefill cost is
`ContextTokens::memory` (`src/context.rs`), reported alongside
`git`/`agents_md`/`date` in the context-token breakdown shown by `/toks`.
Because memory lives entirely in Tier 2, that number is amortized once per
session, not once per turn — which is the whole point of keeping it out of
the live conversation in the first place. The pass is the one exception to
"once per session", and it is the exception you opt into.

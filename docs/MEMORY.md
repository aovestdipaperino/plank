# Persistent memory

plank can remember things about you and about a project across sessions, and
it can maintain that memory itself instead of relying entirely on you typing
`/remember`. This document explains what gets saved, how it is stored, why it
is stored that way, and every setting that controls it.

Status: shipped, on by default (`memory.autoExtract: true`, `tools.remember:
true`). The design choices below all trace back to one constraint: a memory
rewrite must never force a mid-session KV re-prefill. Part 1 covers why. The
rest is mechanics.

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
pass's own prompt states this rule to the model in those terms; see Part 4.

### The file format, and that untagged entries still work

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
to `Kind::Project`. This is not a fallback bolted on for compatibility — it
is the same code path every entry goes through, so a memory file written by
an older plank build, or hand-edited without the tag syntax, renders
identically to a freshly tagged one. `docs/MEMORY.md`'s companion test,
`an_untagged_legacy_file_renders_every_entry_when_nothing_is_configured` in
`src/memory.rs`, pins exactly this: an untagged file at default settings
renders every entry, with nothing evicted. That test is the project's
off-by-default equivalence proof — see "The off-by-default guarantee" below.

A bracket that isn't one of the four recognized tag words (say, a
hand-written `[WIP]`) is not stripped or treated as a parse error; it stays
in the entry's text verbatim, and the entry reads as untagged. Silently
deleting text a person typed because it happened to look like a tag would be
worse than leaving it alone.

### The sidecar, and why it is advisory

Each memory file has a companion `<file>.meta.json` (`memory::meta_path_for`)
holding per-entry bookkeeping, keyed by [`Entry::id`](#entry-identity):

- `uses` — how many extraction passes judged the entry to have borne on the
  work.
- `last_used` — the date of the most recent such pass.
- `pinned` — never evict, whatever the counters say.

A sidecar written by an earlier build of this branch may also carry a
`retracted` key, from a since-dropped design in which a model `forget` only
hid an entry. `MetaStore::load` ignores any key it does not know, so such a
file loads normally and the stray key disappears at the next save.

Every field defaults to zero/false, and `MetaStore::load` treats a missing
file, an unreadable file, malformed JSON, or the wrong shape identically: an
empty store. Nothing in `MEMORY.md` itself is ever machine-owned — a person
can edit, reorder, or delete lines by hand, and the worst that happens is an
orphaned sidecar row, cleaned up the next time anything calls `MetaStore::gc`
against the live entry ids. Losing the sidecar file entirely — deleted,
corrupted, never created — must leave memory loading and rendering exactly
as if every entry had a fresh row: every counter at zero, nothing pinned,
nothing missing. That correctness is what "advisory"
means here, and it is worth stating explicitly because it is easy to design
a cache that quietly becomes load-bearing; this one is tested not to.

#### Entry identity

`Entry::id()` is a truncated SHA-256 hash of the entry's text alone — not its
date, not its type tag. That is deliberate: re-tagging or re-dating a line
keeps the same id, so it keeps its accumulated usage. The consequence is that
rewriting an entry's *text* — which is exactly what an `UPDATE` verdict from
the extraction pass does — changes its id, and something has to carry the old
row's counters onto the new one or a fact reworded six times over a month
would reset to zero usage every time. `MetaStore::carry(old_id, new_id)` is
that something; `apply_one_verdict`'s `Update` arm calls it in the same step
that rewrites the line. See `FINDINGS.md` for the failure mode this guards
against.

### Budgets and eviction

Rendering is governed by per-type character budgets (`memory::Budgets`), not
a single file-level cap — a runaway `project` block can no longer silently
crowd out the `user` block the way one shared budget would let it:

| Type | Default budget |
|---|---|
| `user` | 4096 characters |
| `feedback` | 4096 characters |
| `project` | 6144 characters |
| `reference` | 2048 characters |

`select_for_render` applies each type's budget independently. Within a type,
entries are ranked **pinned first, then by descending `uses`, then by most recent
`last_used`**, and kept in that order until the budget is spent; whatever
doesn't fit is reported as dropped, not silently discarded — `load_scope`
appends a line noting how many older entries were omitted under the type
budget, so nothing disappears without a trace. Age is the *last* tiebreak
rather than the whole rule, on purpose: it inverts a naive tail-truncation
scheme, because the oldest facts about a user are usually the most durable
ones, not the ones most due for eviction.

### The off-by-default guarantee

The property that makes all of the above shippable on by default: an
untagged legacy memory file, with every setting left at its default,
renders every one of its entries, with nothing evicted and no `omitted`
line. Turn the tagging and budgeting machinery off — which is exactly what
an old file with no `[type]` tags does — and you are back to the plain
behavior memory had before this feature existed. This is pinned as a test in
`src/memory.rs`:
`an_untagged_legacy_file_renders_every_entry_when_nothing_is_configured`.

## Part 3: writing memory

### The `remember` / `forget` tools

Advertised to the model after the parity-frozen prompt region
(`sysprompt.rs`), gated on `tools.remember` (default `true`; setting it to
`false` removes both tools from the model's tool table, returning `unknown
tool` if called anyway):

- **`remember(text, type, scope)`** appends a dated, tagged bullet to the
  named scope's file (default `project`). It writes immediately and logs the
  change, but — per Part 1 — the entry is not visible in context until the
  next session start; the tool's own reply says so.
- **`forget(id)`** deletes the entry with that id from whichever scope holds
  it, through the same atomic, audited write path as `/forget`
  (`memory::forget_by_id_to`, sharing `forget_where_to` with
  `forget_matching_to`). There is no hidden "retracted" state: what is in
  `MEMORY.md` is what the model sees, and what the user reads in their own
  file is live. Recoverability comes from the audit log instead — the
  `~/.plank/memory-log.jsonl` line records the entry's full text under the
  reason `forget tool`, distinct from a user's `/forget` (`user /forget`),
  so a wrong model call can be found in `/memory log` and re-added.

### `/remember` and `/forget`

The user-typed equivalents, on both front ends:

- `/remember [user] <text>` — default scope is `project`; prefixing `user `
  writes to the user scope instead.
- `/forget <pattern>` — a case-insensitive substring match against every
  entry's text in both scopes. Unlike the model's `forget`, this is a real,
  confirmed deletion (`forget_matching`): the command previews every entry
  that would be removed and asks before touching anything, because a person
  asking to forget something wants it gone, not hidden.

### The extraction pass

The passive half of memory maintenance: a background pass that reads the
turn's transcript and proposes changes, without the model having to think to
call `remember` itself. It runs at the end of a turn that produced a final
response with no tool calls (`Agent::maybe_extract_memories` in `src/ui.rs`),
through the same sub-agent sidechain other background work uses, so it leaves
no KV checkpoint debris.

**The pass is gated by four checks, in this order** (`ExtractState::should_run`):

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
3. **Not already running.** A trigger that arrives while a pass is in
   flight is dropped outright. Nothing is recorded about it, because
   nothing needs to be: once `finish` clears the flag and advances
   `processed_depth`, the next call re-derives the span from
   `processed_depth` against the then-current depth, which necessarily
   covers everything that arrived meanwhile, in one trailing run.
4. **Throttle.** The pass runs only every `memory.extractEveryNTurns`
   *eligible* turns.

The order is what makes "eligible" mean something. The throttle counter is
reached last, so a turn skipped for suppression, for having nothing new, or
for arriving mid-pass does not count against it. Only a turn that genuinely
had new transcript to look at, and could have been acted on, advances the
count.

**The invariant that makes depth keying correct, stated the way the code
states it:** a pass must read only the transcript above its recorded depth,
and getting that wrong degrades silently into reprocessing everything rather
than failing loudly. There is no error state for "read too much" — it just
means every pass after the mistake re-derives verdicts for text it already
saw, wasting the model's time and risking duplicate or contradictory
verdicts, with nothing in the logs pointing at the cause. `ExtractState`
guards this from both directions: `finish(depth)` clamps `processed_depth` to
never move backward even if called with a stale depth, and a trigger that
arrives while a pass is already running is simply dropped — not queued, not
recorded — because the next `should_run` call after the running pass finishes
recomputes its span from `processed_depth` against whatever depth it is
given *then*, which necessarily covers everything that arrived in between, in
one trailing run.

If the pass is cancelled (interrupted, or its sub-agent turn errors),
`ExtractState::cancel` clears the running flag without advancing
`processed_depth`, so the unprocessed span is simply picked up by the next
eligible turn. Nothing is ever half-applied: verdicts are parsed and applied
only after the pass returns cleanly.

**Verdicts.** The pass replies with a JSON array of verdicts
(`memory::parse_verdicts`), each one of:

- `ADD` — a brand-new entry, with its type and scope.
- `UPDATE` — supersede an existing entry's text (carries usage via
  `MetaStore::carry`, see "Entry identity" above).
- `DELETE` — remove an entry outright, because it's redundant or wrong. Same
  audited outcome as a model `forget`, reached from the pass's verdicts
  rather than a tool call.
- `USED` — credit an existing entry with having borne on the work in this
  excerpt, bumping its `uses`/`last_used`.

Every verdict naming an id with no live matching entry is silently
discarded: the file always wins, because a person may have hand-edited
`MEMORY.md` between the pass reading it and this write landing.
`apply_verdicts` / `apply_verdicts_to` is the single write path for every
automatic change (from both the pass and the tools' logging), and it stages
audit-log lines and sidecar mutations in memory while walking a batch,
flushing both only after the scope's file write has actually succeeded — see
`FINDINGS.md` for why that ordering matters.

### `/memory` and `/memory log`

- **`/memory`** opens every source in one editable buffer (`memory::combine`
  / `memory::apply`), marked with `<!-- plank-memory: begin/end SCOPE -->`
  comments so edits route back to the right file on save. Requires the
  interactive TUI's built-in editor; the plain-stdout REPL prints a message
  pointing at `/memory log` instead.
- **`/memory log`** prints the last 20 entries from the audit log
  (`~/.plank/memory-log.jsonl`), human-rendered, on both front ends. Every
  automatic change — from `remember`/`forget`, from `/forget`, and from the
  extraction pass's verdicts — is appended there as one JSON line
  (`action`, `scope`, `id`, `text`, `reason`), regardless of whether `/memory
  log` is ever opened. It is a write-only audit trail: nothing reads it back
  except this command.

## Part 4: settings

All under the `memory` and `tools` blocks in `~/.plank/settings.json` /
`.plank/settings.json` (`src/settings.rs`):

| Setting | Default | Effect |
|---|---|---|
| `memory.autoExtract` | `true` | Whether the background extraction sidechain runs at all. Off leaves the `remember`/`forget` tools and `/remember` working — only the passive pass stops, and rendering falls back to plain budgeted display with counters nothing ever bumps. |
| `memory.extractEveryNTurns` | `1` | Run the pass every N *eligible* turns (a turn with no tool calls and no model `remember`). `1` means every eligible turn. A configured `0` is clamped to `1`. |
| `memory.budgets.user` | `4096` | Character budget for `[user]` entries. |
| `memory.budgets.feedback` | `4096` | Character budget for `[feedback]` entries. |
| `memory.budgets.project` | `6144` | Character budget for `[project]` entries. |
| `memory.budgets.reference` | `2048` | Character budget for `[reference]` entries. |
| `tools.remember` | `true` | Whether the `remember`/`forget` tools are advertised to the model at all. `/remember` and `/forget` are unaffected — they are user-typed commands, not model tool calls. |

## Cache accounting

How memory's cost is actually paid, tying together Part 1's placement and
Part 4's budgets — the numbers that matter for judging whether memory is
cheap in practice:

| Event | What gets prefilled | Tier |
|---|---|---|
| First turn of a session | Full Tier 2 (`AGENTS.md` set + memory + agent roster) plus Tier 3 (git status, date) | Tier 2 cached to `project.kv` keyed by `stable_hash`; Tier 3 always fresh |
| Every later turn in the same session | Nothing from memory — it is part of the already-cached prefix | Reused |
| A memory file edited (`/remember`, `/forget`, the pass, a hand edit) | Nothing, until the *next session* | The new content changes `stable_hash`, so the next session's first turn re-prefills Tier 2 once |
| `/memory` opened and saved with no actual change | Nothing | `apply` only writes files whose body changed; an unchanged section reports `unchanged` and touches no cache |
| The sidecar (`MEMORY.md.meta.json`) changing | Nothing, ever | It is never part of any prompt tier — only `MEMORY.md`'s own text is |

The number that actually governs prefill cost is `ContextTokens::memory`
(`src/context.rs`), reported alongside `git`/`agents_md`/`date` in the
context-token breakdown shown by the status bar and `/toks`. Because memory
lives entirely in Tier 2, that number is amortized once per session, not
once per turn — which is the whole point of keeping it out of the live
conversation in the first place.

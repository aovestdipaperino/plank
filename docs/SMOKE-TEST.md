# Smoke test: the `docs/dsh.md` milestones (M0–M10)

Manual verification for the eleven milestones implemented from the DeepSeek-harness
plan in [`docs/dsh.md`](dsh.md). Each section names the surface, the exact commands to
type, and what a pass looks like. Nothing here needs a GGUF model except where marked
**needs inference**; everything else works against the `EchoEngine` stub.

Order matters only in that M0 (`--dump-config`) is the fastest way to confirm a setting
actually took effect, so run it first and come back to it whenever a later step looks
inert.

## 0. Before you start

```sh
cargo build
cargo test --lib                       # pure logic + EchoEngine
cargo test --test log_invariant         # M3's invariant test
cargo clippy --workspace --all-targets -- -D warnings
```

All four must be clean. Then take a backup of your plank home if you care about it,
because several steps write settings and read/write session state:

```sh
cp -r ~/.plank ~/.plank.bak
```

Several milestones are **off by default** on purpose (parity: enabling them changes the
tools prompt and therefore the system-prompt fingerprint `fp1`). Turn on the ones you
want to exercise in `~/.plank/settings.json`:

```json
{
  "tools": {
    "recall": true,
    "fanout": true,
    "runCode": true,
    "callTimeoutSec": 10,
    "spillMaxBytes": 2048,
    "spillPreviewBytes": 256
  }
}
```

The two spill numbers above are deliberately tiny so ordinary output spills and you can
see the mechanism; the shipped defaults are 1 MiB / 4 KiB, which ordinary sessions never
hit. Remember to remove them afterwards.

---

## M0 — Configuration provenance (`--dump-config`, `/config --resolved`)

```sh
./target/debug/plank --dump-config | head -40
./target/debug/plank --dump-config -c 8192 | grep -i ctx
```

**Pass when:** every effective key prints with the layer it came from; `engine.ctx`
shows the CLI origin winning over user settings in the second command, with the shadowed
candidate listed underneath.

Then in an interactive session (both front-ends — plain REPL and TUI):

```
/config --resolved
```

**Pass when:** the same provenance dump appears in the pane/scrollback rather than the
config form opening.

The shadowed-candidate line only appears when a *settings file* set the key that the CLI
flag overrides. With nothing but the built-in default beneath it, `engine.ctx` correctly
prints one line and no `(shadowed: …)` — to exercise the shadow path, set
`{"engine":{"ctx":262144}}` in `~/.plank/settings.json` first, then pass `-c 8192`.

Keys a settings file can set but `/config` cannot edit (`kvcache.*`, `worktree.*`,
`update.check`, `pluginConfig.*`) print their origin with **no value**. That is by design,
not a truncated line.

To see the plugin-claiming rule, drop two plugins that define the same skill or agent name
under `~/.plank/plugins/dev/` and confirm the dump names the winner, the losers, and the
`<plugin>:<name>` alias each loser is still reachable under. When exactly two plugins
collide and nothing else claims the bare name, the winner line is
`skills.<name>        <- no plugin holds the bare name` with both plugins listed as
shadowed beneath it — a bare name nobody holds is the correct outcome of a tie, not a
missing winner. A `plugin warning: '<name>' is contributed by plugins a, b; use the
namespaced names` also goes to stderr at startup.

> **This part needs an interactive session.** Claiming provenance is populated during
> session construction, and `--dump-config` prints and exits before that, so
> `plank --dump-config` shows **no** `skills.*` lines however many plugins collide. Check
> the claiming rule via `/config --resolved` only.

## M1 — Loop guards (repeat advisory + call deadline)

**Repeat advisory** (`tools.repeatAdvisory`, default **on**) — needs inference, or drive
it with a scripted engine. Ask the model to read the same file several times:

> read src/lib.rs, then read it again, then read it again, then again

**Pass when:** the third identical `read` (same tool, same normalised args) carries the
line `[loop guard] you have called this tool with these arguments 3 times; the result has
not changed`, the first two do not, and nothing is ever blocked. The threshold is
`guard::REPEAT_THRESHOLD = 3`, counted over a 10-call window (`MAX_WINDOW`), so a repeat
far enough back ages out and stops counting.

**Pass when (negative case):** an async bash job polled repeatedly never produces the
advisory. Start a long job and let the model poll it; identical poll args are exempt by
design.

**Deadline** (`tools.callTimeoutSec`, default `0` = off). With `callTimeoutSec: 10` set,
have the model run `bash sleep 30`.

**Pass when:** the result carries the line
`[deadline] bash exceeded tools.callTimeoutSec=10s` (model-facing, appended to the tool
result) — not a hang and not a panic. Unset the key and confirm the same command runs to
completion with no such line (parity untouched by default).

> **The deadline does not cancel the tool.** It is post-hoc and advisory: `sleep 30` still
> runs its full 30s and exits 0, and only then is the notice appended. A run that takes
> 30s rather than being killed at 10s is correct behaviour — the budget is something the
> model is told it blew, not a kill switch. Expect the model to say so in its reply.

## M2 — Feedback sidecar (`/rate`)

Plain-stdout REPL, after at least one assistant turn:

```
/rate + this was the right fix
/rate -
/rate ?            # expect the usage line
```

Then:

```sh
cat ~/.plank/usage-data/feedback/<session-id>.jsonl
```

**Pass when:** one JSON line per rating with `ordinal`, `digest`, `positive` (the rating),
`note` and `at`; the usage line `usage: /rate [+|-] [note]` prints for the bad sign.

**The important assertion — the rating must not leak into the model's world.** Before
rating, copy the session transcript file out of `~/.plank/kvcache`; after rating, diff it.

**Pass when:** byte-identical. The rating lives only in the sidecar; it never enters the
transcript, model context, or KV.

**Orphaning:** rate a turn, then `/compact` (which renumbers the transcript), then run
`/insights`.

**Pass when:** the report carries a line
`N orphaned (the rated turn is gone; not counted above)`, the orphaned rating is **not**
folded into the `N ratings: … positive … negative` totals or the satisfaction-by-day
series, and a negative orphan does not appear under `worst-rated turns` — it is never
reattributed to whatever now sits at that index. Ratings whose session file has been
swept entirely are orphaned on the same rule.

> `/insights` only reports once there are sessions substantial enough to summarise; a
> handful of one-line toy sessions prints `No sessions substantial enough to report on
> yet.` and no feedback block at all. Use a real working session for this step, or lean
> on the unit tests `insights::tests::a_rating_orphaned_by_compaction_is_reported_and_not_counted`
> and `ratings_for_a_vanished_session_are_orphaned`.

> Known gap: `/rate` exists on the plain-stdout path only. The TUI has no rating key
> yet, so verify this one in a piped/plain session (`plank | cat`-style, or a non-TTY
> stdin).

## M3 — Log-everything invariant

```sh
cargo test --test log_invariant
```

**Pass when:** green. To confirm it actually bites, temporarily inject a line of
model-visible text into an assembled request without pushing it to the transcript (for
example an extra `push_str` in a context builder) and re-run.

**Pass when:** the test fails with
`assertion \`left == right\` failed: every span must be attributable to the transcript or
the system prompt`. Revert the injection.

Inject into `ui::render_transcript` (`src/ui.rs`) — a `push_str` between the `[system]`
block and the message loop is the shape a real regression would take. Note that only
`every_injection_site_is_reconstructible_from_the_transcript` catches it; the companion
test `the_system_prompt_is_the_fingerprinted_exception` asserts only `starts_with` and
`contains`, so it stays green under injection by design. Back the file up first — an
unreverted injection is a live prompt-corruption bug.

Manual counterpart: `/repro` and `/resume` on a session that used skills, templates,
memory, tasks and MCP tools.

**Pass when:** the **transcript messages** of the replayed request match the recorded one
byte-for-byte — no block silently gained or lost. Compare the span between
`----- BEGIN TRANSCRIPT -----` and `----- END TRANSCRIPT -----` in the two `/repro` files,
splitting the `[system]` block off at the first `[user]`/`[assistant]` marker. Note that
`## Tools`, `## Editing files` and `### Available Tool Schemas` are headings *inside* the
system prompt, not sections of the repro: splitting on them compares almost nothing.

**The system prompt is expected to differ, and that is not a failure.** Tool
advertisements are rebuilt live from the current config on every start, and the invariant
exempts the system prompt precisely because it is accounted for separately by `fp1`. To
see this: add a server to `~/.plank/.mcp.json`, start a session (its adverts include
`mcp__<server>__<tool>`), remove the server, then `/resume` and `/repro` again.

**Pass when:** the transcript messages are identical, the system prompt has legitimately
lost the advert, and a second `sysprompt-<fp1>.kv_raw` appears in `~/.plank/kvcache` —
the fingerprint changed and the snapshot was rebuilt, not silently reused. A session that
*still shows the advertisement text it originally saw* would actually be the bug, since
the prompt would no longer match `fp1`.

> **Known hazard, not yet a pass/fail criterion.** A session that actually *called* a tool
> from a since-removed server keeps those `<tool_result>` blocks in its transcript while
> the resumed system prompt no longer advertises the tool. The transcript is intact and
> `fp1` is honest, so the invariant holds, but the model sees results from a tool it is
> not offered. Worth watching if a resumed session starts behaving oddly around MCP.

## M4 — Output spill

With `spillMaxBytes: 2048` / `spillPreviewBytes: 256` set, get a large tool result. Ask
for a whole-file read of a file a few KiB long:

> Call the read tool on src/guard.rs with whole=true.

> **Do not use `bash` for this.** A large `bash` result never reaches the spill policy:
> the bash tool already writes the full output to a temp file and inlines only
> `<head -100 …>` plus an `output_path=… (N bytes, M lines)` line, so
> `bash yes plank | head -c 200000` yields a ~600-byte result and no spill at all. That is
> the C-parity framing working as intended, not a spill failure.
>
> Keep the session short — one or two tool calls. Microcompact (M5) clears all but the
> newest three tool-result bodies at end-of-turn, so in a longer session the locator you
> came to inspect is replaced by the `[old tool result cleared …]` stub before you can
> read it.

**Pass when:**
- the inline result is a bounded preview followed by
  `[Output truncated at 256 bytes of NNNNN. continue_offset=256. Call more with count=4096 to read the next chunk.]`
  (for `src/guard.rs` the observed line is `… at 256 bytes of 5262 …`)
- the preview really is `spillPreviewBytes` long, and the transcript holds **only** it —
  grep the transcript for a string from the middle of the file and expect zero hits
- `~/.plank/spill/<session-id>/0.txt` exists and holds the **full** payload, its byte
  count equal to the `of NNNNN` in the locator
- asking the model to call `more` returns the next chunk
- a `PostToolUse` hook sees the full text while the transcript sees the preview (add a
  hook that writes `$stdin` length to a file and compare)

Repeat for the other previously-dead-ended sites: a file too large for `read`, a `glob`
hitting the result cap, and an oversized **MCP** result (which had no cap at all before).
For `glob`, ask a question that needs the list ("How many Rust files are in this repo?
Use the glob tool with pattern `**/*.rs`") — a bare instruction to call it is often
answered without a tool call. Its own `... more than 100 matches; showing the first 100 ...`
cap and the spill locator both appear; that is correct, they bound different things.

Spill is applied **post-dispatch and tool-agnostic** — every arm of the `dispatch` match,
`mcp__*` included, funnels into the single `spill::apply` call — so the MCP case needs no
special handling. If no MCP server to hand returns a few KiB, the structural check plus
`spill::tests::oversized_results_spill_to_a_preview_plus_locator` (which spills under the
tool name `mcp_call`) is adequate evidence.

**Pass when:** every one of them is bounded and every bounded result has a retrieval
path.

**Then restore the defaults** and confirm an ordinary session never spills:
`ls ~/.plank/spill` stays empty for normal work.

**Parity check:** `cargo test --test c_parity` must still pass — the existing `read`
truncation fixtures are byte-for-byte.

## M5 — Microcompact cadence and keep-policy

Run a session that produces many large tool results, one `read` per turn across several
turns (a dozen whole-file reads batched into one turn will not do it — see the note
below), and watch the context gauge in the status bar across turns.

Whole-file reads of large modules are slow; prefer several small files (`src/guard.rs`,
`src/feedback.rs`, `src/kvmeta.rs`, `src/provenance.rs`) over a few big ones, or the
session outruns a ten-minute patience budget before the policy has anything to do.

**Pass when:**
- old tool-result bodies get replaced by the microcompact stub at end-of-turn without you
  typing `/compact`
- the pass only fires when it would reclaim at least 4096 bytes
  (`MICROCOMPACT_OPPORTUNISTIC_MIN_BYTES`) — short sessions with small results are left
  alone, the gate that stops eager pruning from costing more KV prefix than it saves.
  The threshold itself is pinned by
  `compact::tests::reclaimable_gates_the_opportunistic_pass`, which also asserts that
  `microcompact_reclaimable` predicts exactly the bytes the pass goes on to reclaim
- the newest three result **messages** survive (`MICROCOMPACT_KEEP_RESULTS = 3`), **and**
  any result body of 256 bytes or less survives (`MICROCOMPACT_MIN_BYTES`), **and** every
  tool result that follows the last `# Task list` injection survives regardless of
  position — that suffix is the currently active work

> **Count messages, not tool calls.** Several tool calls emitted in one assistant turn
> come back as a *single* transcript message holding `Tool result 1 (read):`,
> `Tool result 2 (read):` and so on, and microcompact clears message bodies. Four parallel
> reads are therefore one unit to the keep-policy, not four: a session of four batched
> reads leaves nothing to clear and correctly produces no stubs. To exercise the policy
> you need several separate *turns*, each with its own tool call.
- a cleared body reads exactly
  `[old tool result cleared to reclaim context; rerun the tool if the output is needed again]`

`/context` before and after gives you the numbers to compare.

## M6 — Session index and `/search`

Both front-ends:

```
/search kvcache
/search kvcache --all
/search
```

**Pass when:** hits render as a numbered list with session title, age and a snippet, with
`/resume` offered on a hit; the bare form prints `usage: /search <query> [--all]`; the
default is scoped to the current project and `--all` widens it beyond it.

The point of the feature is compaction-proof history. Re-indexing is wholesale — a session
whose size/mtime stamp changed is re-read from scratch — so the index keeps an **archive**
of conversation the transcript has since dropped (`IndexEntry::archived`), and `search`
looks at archive plus live transcript together.

**Pass when:** text that existed in a session *before* it was compacted is still findable
by `/search`, and the hit's snippet quotes the archived text.

To check it end-to-end without fighting the model, read the index entry directly rather
than trusting a marker to disappear:

```sh
A=<session-id>
python3 -c "import json;d=json.load(open('$HOME/.plank/usage-data/session-index/$A.json'));
print(len(d['messages']),'live,',len(d['archived']),'archived')"
```

**Pass when:** after a `/compact` and a re-indexing `/search`, `archived` is non-empty,
and a phrase taken from an archived message — one that `grep` no longer finds in
`~/.plank/kvcache/$A.kv` — is still returned by `/search`.

> Two traps make this look broken when it is not. A small session keeps everything in
> compaction's verbatim tail, so nothing is actually dropped and the test proves nothing —
> shrink the context (`-c 16384`) so the tail is small. And the compaction summary often
> quotes a distinctive marker verbatim, so the marker is findable via the *summary* rather
> than the archive; probe with a phrase the summary paraphrases instead.

**Deliberately not archived: tool results.** Microcompact clears large result bodies on
most turns, so archiving them would grow the index without bound and make it a second copy
of every tool output ever produced. Tool output is re-derivable by rerunning the tool; the
conversation is not. The archive is also capped at `ARCHIVE_MAX_BYTES` (64 KiB per
session), trimmed oldest-first.

**Pass when:** a session whose large tool result was cleared by microcompact does **not**
carry that result text in its index entry, while conversation that left the transcript
does.

Idempotence:

```sh
ls -l ~/.plank/*index* 2>/dev/null || true   # locate the index dir
```

Run `/search` twice with no session activity in between.

**Pass when:** the index is not rebuilt wholesale the second time, and a session
*rewritten in place* (append a turn, save) does get re-indexed.

## M7 — Durable goal state

```
/goal                      # expect "no goal set"
/goal --max 2 make the failing test pass
/goal                      # expect objective, iteration n/2, status
/goal clear
/goal                      # back to "no goal set"
```

**Pass when:** all four behave as annotated, on both front-ends.

Durability:
1. Set a goal, `/save`, quit, `/resume` the session — **pass when** the goal and its
   iteration counter come back.
2. Set a goal, `/compact` — **pass when** the goal survives (it is pinned like the durable
   summary, not dropped like a tool result).
3. Take a session file written before this feature existed, load it, and re-save with no
   goal set — **pass when** the file is byte-identical (the `goal` record is optional).

Reconciliation:

**Pass when:** `/tasks` shows the goal above the task list, and a sub-agent launched while
a goal is active receives the goal in its preamble — the delegated task message opens with
the subagent `<system-reminder>` and then a literal `Session goal: <objective>` line, ahead
of any named-agent `Instructions:` block and the `Task:` line. Check this on **both**
delegation paths, the single `agent` call and a `fanout` subtask; each builds its task
message independently, so one can regress without the other.

**The negative half matters more.** Only an *active* goal is model-facing
(`Agent::active_goal`). A goal that has settled, and one ended with `/goal clear`, must
**not** appear in a subagent preamble or in the post-compaction task block — presenting a
revoked or finished objective as the live one is worse than showing none.

**Pass when:** after `/goal clear`, a sub-agent's task message carries no `Session goal:`
line, and after the loop settles (`status = done`) neither does it. `/goal` and `/tasks`
still read `session.goal` directly, so a settled goal is *reported* with its status even
though it is no longer injected; a cleared goal is gone from both, and its `goal` record
is absent from the saved session file.

## M8 — The `recall` tool (needs `tools.recall: true`, needs inference)

With `recall` off (the default), ask the model to use it.

**Pass when:** it is not advertised in the tools prompt at all, and a hand-forced call
returns `Tool error: unknown tool: recall`.

Turn it on, then in a *fresh* session ask something only an earlier session knows:

> recall what we decided about the KV sysprompt fingerprint

**Pass when:** the tool returns hits as `<session-id> — <title> — <snippet>` lines from
prior sessions in **this project only**, plus `current session — <snippet>` lines from the
live transcript; and an oversized result is bounded by M4's spill policy.

Ask for the call explicitly — *Call the recall tool with query "…"* — rather than phrasing
it as a question. Asked "recall what we decided about X", the model tends to go and read
the repository instead of calling the tool at all.

> **Two criteria cannot be checked live.** The empty-query error needs a call the model
> will not make. And `No sessions match "…".` is unreachable in a live session: the
> request itself quotes the query, so the current transcript always self-matches and at
> least one `current session —` line always comes back. Both are covered by
> `tools::tests::recall_rejects_an_empty_query_and_reports_no_match`.

**Parity note to verify, not to be surprised by:** enabling `recall` changes the tools
prompt, which changes `fp1`, which invalidates `sysprompt-<fp1>.kv_raw`.

**Pass when:** the first turn after flipping the setting is slower (snapshot rebuild) and
a new `sysprompt-<fp1>.kv_raw` appears in `~/.plank/kvcache` — rebuilt, not broken.

## M9 — Subagent fan-out (needs `tools.fanout: true`, needs inference)

Off by default; same unknown-tool check as M8.

Turn it on and ask for two independent subtasks that map to named agents you have
defined, e.g.:

> use fanout to have the reviewer agent check src/spill.rs and the reviewer agent check src/guard.rs

**Pass when:**
- the joined result is `Subtask 1 (<name>): …` / `Subtask 2 (<name>): …` in a stable,
  deterministic order
- an agent whose def asks for `isolation: worktree` really runs in a throwaway worktree
  (`git worktree list` during the run) and it is cleaned up afterwards
- an empty or malformed `subtasks` array returns
  `fanout requires a non-empty 'subtasks' JSON array`

**Explicitly not a pass criterion: speed.** Subtasks run serially (concurrency bound 1)
because one Metal command queue buys no parallel throughput; the tool description says so
on purpose. If a fan-out of two subtasks takes about as long as two sequential agent
calls, that is correct behaviour, not a bug.

## M10 — `run_code` (needs `tools.runCode: true`)

Off by default; same unknown-tool check as M8.

Turn it on and have the model run a script — one operation per line, from
`read | glob | edit | bash`:

```
glob src/*.rs
read src/lib.rs
edit src/foo.rs old_name new_name
bash cargo fmt --check
```

**Pass when:** the result is `Step N (<tool>):` blocks in script order; an unrecognised
operation yields `Step N: unknown operation "…"` and does **not** abort the rest of the
script; an empty script returns `run_code requires a non-empty 'script'`; and oversized
combined output is bounded by M4's spill policy.

A four-operation script exercises all of it at once, e.g. `glob src/g*.rs`,
`read src/guard.rs`, `frobnicate something`, `bash echo hello-from-run-code` — expect
`Step 3: unknown operation "frobnicate"` followed by a `Step 4 (bash):` that still ran.

> **Ask for the whole script in one message.** The plain REPL submits a turn per line, so
> pasting a multi-line script types the first line into `run_code` and feeds the remaining
> lines to the model as separate user turns. Describe the operations inline instead —
> *"its script parameter must contain exactly these four operations, one per line"* — and
> let the model emit the newlines inside the DSML parameter.
>
> The empty-script error is not reachable through the model; it is covered by
> `tools::tests::run_code_is_off_by_default_and_executes_a_script_when_enabled`.

**The security assertion — this is the whole risk of the feature.** Bindings must go
*through* the tool path, not around it:

**Pass when:** a `bash` step inside `run_code` that writes outside the sandboxed cwd is
refused by the same message a bare `bash` tool call is refused with, a step touching
`~/.plank` triggers the same plank-home grant prompt, and a step needing consent prompts
exactly as the equivalent tool call would. If any of these silently succeed inside
`run_code`, stop and report it — that is a hole straight through every guard in
`consent.rs` and `sandbox.rs`.

The binding is structural, which is why it holds: each operation is turned into a real
`ToolCall` and handed to **`dispatch`**, the same entry point a model-issued call goes
through. The PreToolUse hooks live inside `dispatch`, and the sandbox and plank-home grant
live inside `tools::bash::tool_bash` below it, so `run_code` cannot reach a tool without
passing them. `tools::tests::a_bash_step_inside_run_code_is_sandboxed_like_a_bare_call`
pins this: the same escaping command is refused with `[sandbox blocked:` both as a bare
call and as a script step, and the write never lands. It is macOS-only (needs
`/usr/bin/sandbox-exec`).

---

## Cross-cutting checks

Run these once at the end, regardless of which milestones you exercised.

**Both front-ends.** Every surface above landed twice. Repeat `/config --resolved`,
`/search`, `/goal` in the plain-stdout REPL *and* the Ratatui TUI.

**Pass when:** behaviour matches, allowing for the documented gap that `/search` is static
text in the TUI (no dedicated pane yet) and `/rate` is plain-path only.

**Parity.**

```sh
cargo test --test c_parity
```

**Pass when:** green with all the new tools **off** (the default). With `recall`/`fanout`/
`runCode` on, the tools prompt legitimately differs — that is the versioned deviation
recorded in `FINDINGS.md` and `docs/SYSTEM-PROMPT-OVERRIDES.md`, not a failure.

**No second cache, no second GC.** Fill `~/.plank/spill`, the session index and the
feedback sidecars, then let the startup sweep run (or force it via `/kvcache gc`).

**Pass when:** spill blobs are swept under the same TTL and byte budget as `kvcache`, and
no new GC-looking directory appears anywhere else.

**Cleanup.**

```sh
rm -rf ~/.plank && mv ~/.plank.bak ~/.plank
```

## Reporting a failure

Include: which milestone, the exact command or prompt, the observed vs. expected text,
and the relevant settings block from `plank --dump-config` (M0 exists precisely so this
last one is one command).

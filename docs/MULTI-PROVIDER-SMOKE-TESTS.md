# Multi-provider smoke tests

Manual checks for cross-provider sub-agents: a named definition in
`~/.plank/agents/*.md` running on a different engine than the main agent.

Everything here is **manual and needs real credentials**. The unit suite covers
the logic against `EchoEngine` and stubs with no network
(`cargo test --lib` — see `agents::`, `sysprompt::`, and the `fanout_*` /
`alt_engine` tests in `ui::`). What it cannot cover is a real provider's wire
format, real latency, real rate limits, and whether the model actually chooses
to delegate. That is what these are for.

## What is and isn't supported

| Main engine | Sub-agent engine | Supported |
|---|---|---|
| local ds4 | local ds4 (no `provider:`) | yes — the pre-existing behaviour |
| local ds4 | remote provider (`provider:` set) | **yes — the point of this feature** |
| remote provider | same or another remote provider | yes |
| remote provider | local ds4 | **no — see below** |

### Why "remote main, local sub-agent" does not work

Not merely unimplemented — there is no local engine in the process to run on.
`make_host` in `src/main.rs` returns the `ProviderEngine` and **returns early**
(`src/main.rs:249`), so the ds4 branch never executes and no GGUF is ever
loaded. A definition without `provider:` runs on "the parent's engine", which in
that configuration *is* the remote provider.

Making it work would mean loading the local model on demand for a sidechain. The
default quant needs ~82 GB resident and only one process can hold it
(`require_min_ram` and the single-instance guard in the same function), so this
is a real architectural constraint rather than a missing flag. If you want a
cheap local sub-agent under an expensive remote main agent, the shape that could
work is a *local plank serving over `--serve`* with the sub-agent pointed at it
as an OpenAI-compatible endpoint — untested, and out of scope here.

**Consequence worth internalising:** with `--provider`, a definition that omits
`provider:` does not mean "local". It means "whatever the main agent is". Test 7
checks that this is at least harmless.

## Setup

```sh
export ANTHROPIC_API_KEY=sk-ant-...        # provider default
export ANTHROPIC_API_KEY_ALT=sk-ant-...    # a second key, for test 5
mkdir -p ~/.plank/agents
```

`~/.plank/agents/remote-reviewer.md`

```markdown
---
name: remote-reviewer
description: Reviews a diff for correctness and missed edge cases
provider: anthropic
model: claude-opus-5
---
Review what you are given for defects you can demonstrate. For each one, give
the input that triggers it and the wrong output. Skip style opinions. Finish
with a short report.
```

`~/.plank/agents/local-helper.md` — no `provider:`, so it runs on the parent
engine:

```markdown
---
name: local-helper
description: Summarises how a module works
---
Answer concisely from the code you read. Finish with a short report.
```

---

## 1. Registration and visibility

**Run:** `/agent`

**Expect:**

```
Agents (dispatch with /subagent <name> <task>):
  local-helper — Summarises how a module works
  remote-reviewer — Reviews a diff for correctness and missed edge cases [anthropic claude-opus-5]

Model may pick these on its own: yes (/config agents.autoRoute), up to 4 at once (/config agents.maxParallel).
```

- The engine label appears only on `remote-reviewer`.
- No `(no ANTHROPIC_API_KEY)` marker, since the key is set.

**Then** unset the key in a fresh shell and re-run: the marker appears and names
the variable. Definitions stay listed — being unusable hides a definition from
the *model*, not from you.

## 2. Explicit dispatch to a remote definition

**Run:** `/subagent remote-reviewer summarise what src/agents.rs does`

**Expect:**
- A `[sub-agent: remote-reviewer — ctrl+o to follow]` line.
- `ctrl+o` shows the sidechain; its text reads like the remote model, not the
  local one.
- Only the framed report enters the main conversation.
- The footer's engine-origin indicator still shows the **main** engine
  afterwards — the swap was restored.

**Watch for:** any sign the sidechain ran on the local model (the local model has
a recognisable voice, and a local run is far slower to first token on a cold KV).

## 3. Tools work inside a remote sidechain

This is the one most likely to break silently, because a provider engine given a
flat prompt receives an empty tool list.

**Run:** `/subagent remote-reviewer count the test functions in src/agents.rs by reading the file`

**Expect:** the sub-agent actually calls `read` (or `search`), and its report
cites a real number. If it instead says it cannot access files, the structured
prompt or tool registry is not reaching it — check
`build_structured_for`.

## 4. The model routes on its own

**Run:** a prompt that invites delegation without naming an agent:

> fix the off-by-one in `<some file>`, then have the change reviewed

**Expect:** the model calls `agent` with `name: "remote-reviewer"` of its own
accord.

**Then** run `/config agents.autoRoute false` and repeat. Expect the model to
stop selecting definitions and either do the work itself or delegate to a
general-purpose sub-agent — while `/subagent remote-reviewer ...` still works.
Restore with `/config agents.autoRoute true`.

**Also try** a plausible-but-wrong name by asking for "the code-reviewer agent".
A near-miss must produce a `note: no agent named '…'` line plus a real report,
never a bare tool error.

## 5. Two keys, two accounts

Add `~/.plank/agents/alt-reviewer.md` — identical to `remote-reviewer` except:

```markdown
name: alt-reviewer
api-key-env: ANTHROPIC_API_KEY_ALT
```

**Run:** `/subagent remote-reviewer …` then `/subagent alt-reviewer …` in one
session.

**Expect:** both succeed. They differ only in key variable, so they must get
*separate* cached engines — if they shared one, the second would run on the
first's credentials. Check the provider dashboards for both keys and confirm
each shows exactly one request's worth of usage.

**Then** unset `ANTHROPIC_API_KEY_ALT` and retry `alt-reviewer`: expect
`Tool error: agent 'alt-reviewer' engine unavailable: ANTHROPIC_API_KEY_ALT is
not set`, naming *that* variable rather than the provider default, and with no
sidechain started.

## 6. Parallel fan-out

Add a second and third remote definition (`remote-a`, `remote-b`, `remote-c`),
each with a `provider:`/`model:`, then ask for work that splits cleanly:

> review src/agents.rs, src/settings.rs and src/engine.rs — use a separate
> sub-agent for each

**Expect:**
- A single `[sub-agents: remote-a, remote-b, remote-c — ctrl+o to follow]` line,
  plural.
- Wall-clock close to the *slowest* sub-agent, not the sum. Time it — this is the
  only check that proves concurrency rather than fast serial execution.
- Reports appear in the order the model requested them, regardless of which
  finished first.
- Output is **buffered**: nothing streams during the fan-out, then each
  sub-agent's block appears labelled, in call order. This is by design; the pane
  holds one log, so live interleaving would be unreadable.

**Then** `/config agents.maxParallel 1` and repeat: expect the same reports,
serially, taking roughly the sum of the individual times.

**Then** ask for two reviews *and* a file read in the same turn. Expect no
fan-out — a mixed block stays serial so side effects keep their order.

## 7. Remote main agent

**Run:** `plank --provider anthropic --model claude-opus-5`

**Expect:**
- `/agent` lists both definitions as before.
- `/subagent remote-reviewer …` works (remote main → remote sub-agent).
- `/subagent local-helper …` **also works, but runs on the remote provider** — it
  has no `provider:`, so it inherits the parent engine. Confirm it is not
  silently doing nothing, and confirm the billing lands on the main key.

There is deliberately no test for "local sub-agent under a remote main agent";
see the table above.

## 8. Failure and interruption

| Do this | Expect |
|---|---|
| Set `model:` to a name the provider does not know | A tool error from the provider; the main session keeps working afterwards |
| Set `base-url:` to an unreachable host | A tool error after the retry budget; the main session survives |
| `ctrl+c` during a remote sidechain | The turn ends; the next turn works normally and the footer shows the main engine |
| `ctrl+c` during a fan-out | Same; partial reports for whatever finished |
| Revoke the key mid-session, then dispatch again | `engine unavailable: <VAR> is not set`, with no sidechain started |

After every one of these, `/subagent remote-reviewer ok` must still work. A
leaked engine swap would leave the whole session pointed at the wrong engine,
which is the worst failure this design can produce and the thing most worth
re-checking by hand.

### Two `claude-opus-5` behaviours worth provoking

Neither is a plank bug, and neither shows up on a happy-path run — but both
surface through the provider path as something that *looks* like a plank defect.

**A refusal is an HTTP 200, not an error.** The model's safety classifiers can
decline a request, returning success with `stop_reason: "refusal"`, a
`stop_details` category, and `content` that is empty (declined before any output)
or partial (declined mid-stream). A request touching security or life-sciences
topics is the way to provoke one; benign adjacent work sometimes trips them,
which is exactly why this matters. Ask the sub-agent to review something
security-shaped and check what comes back: a clear tool error naming the refusal
is fine, an empty report or a parse failure is not — that means the response path
reads `content` without checking `stop_reason` first.

**Thinking is on by default and shares the `max_tokens` budget.** Unlike
Opus 4.8/4.7, omitting the `thinking` parameter on `claude-opus-5` runs adaptive
thinking, and `max_tokens` caps thinking *plus* response text together. Give a
definition a deliberately small `ctx:` and a task needing a long answer; if the
report arrives truncated mid-sentence, the budget is being consumed by thinking.
Worth knowing before blaming the sidechain loop.

## 9. Cache and cost sanity

- Dispatch the same definition three times in one session. The context-window
  probe should happen **once**; watch for a single extra request beyond the three
  generations on the provider side.
- A clean-room sidechain must send only the framed task, never the parent
  conversation. Check the provider's request logs (or a proxy) and confirm the
  parent transcript does not appear. This is a privacy property, not just a cost
  one.
- Compare token counts for a long main conversation: a remote sub-agent's input
  should stay small and roughly constant as the parent conversation grows.

## 10. Regression checks after any change here

```sh
cargo test --lib            # 1235 tests, no model or network
cargo test --test c_parity  # must pass with fixtures untouched
cargo clippy --workspace --all-targets -- -D warnings
```

`tests/fixtures/` must show no diff. With an empty roster both schema paths emit
byte-identical output to the pre-roster build, which is what keeps the parity
fixtures valid; if they change, the fix belongs in the code, not in
`PLANK_REGEN_FIXTURES=1`.

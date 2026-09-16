[← Sessions](06-sessions.md) · [Index](README.md) · Next: [Configuration →](08-configuration.md)

# 7. Context

Everything the model can see at once — the system prompt, your project's instructions, the conversation so far, the files it has read, the tool output it received — lives in a fixed-size **context window**. Managing it is most of what separates a session that stays sharp from one that goes vague after an hour.

plank does the management for you. This page is about understanding it, and about the handful of levers worth pulling.

## Seeing where it went

```
/context
```

reports usage by category: the system prompt, session context, the conversation, tool results, and so on. The status bar carries a live gauge of the same number, so you normally notice pressure before you have to think about it.

## What goes in at session start

Before your first message the model receives:

- **The system prompt** — plank's instructions, tool definitions, and the environment. Override it for one run with `-sys "…"`.
- **Session context** — git status, the date, and the working directory.
- **`AGENTS.md`** — project instructions, discovered from the working tree.
- **Memory** — `~/.plank/MEMORY.md` and `./.plank/MEMORY.md`.

The system prompt is cached on disk as a fingerprinted snapshot, so restarts do not pay to re-read it.

## `AGENTS.md`

The conventional place for "things anyone working in this repo needs to know": build commands, architecture, house style, gotchas. plank finds it and injects it at session start. It is the only instructions file plank reads: a project root that has a `CLAUDE.md` but no `AGENTS.md` gets an `AGENTS.md` symlink to it the first time you start plank there, and a project with neither is asked whether to generate one. Headless runs (`--ui console`) do neither.

```
/init
```

has the model read your repository and write one. Review what it produces — it is a starting point, not a finished document.

## Memory

Memory is two plain markdown files, layered like the rest of plank's config:

- `~/.plank/MEMORY.md` — **user scope**: who you are, durable preferences.
- `./.plank/MEMORY.md` — **project scope**: goals and constraints of this checkout.

Both load at session start. Append to them from the prompt:

```
/remember prefers small commits with imperative subject lines
/remember user I work in Rust and TypeScript, mostly on macOS
```

Without `user`, the entry goes to project memory. Entries are dated bullets, each carrying a type tag:

```
- (2026-09-15) [feedback] don't force-add generated docs
```

To edit them rather than append, `/memory` opens both files as one buffer in the built-in editor, each between markers naming its scope and path; on save the buffer is split back along those markers and only the files whose text changed are written. `/forget <pattern>` removes every entry whose text contains the pattern, case-insensitively, after showing you exactly what will go.

Four kinds of entry are worth keeping, and they share one test: **facts the model cannot re-derive from the repository.**

| Type | What |
|---|---|
| `user` | who you are |
| `feedback` | corrections on how you want work done, and why |
| `project` | goals and constraints not visible in the code |
| `reference` | external URLs, tickets, dashboards |

Do not record what the code, the git history, or `AGENTS.md` already says. That is context you are paying for twice. A bullet without a tag still works and reads as `project`, so a memory file from an older plank needs no migration.

### The model can write memory too

The model has a `remember` tool and a `forget` tool (gated on `tools.remember`, on by default). `remember` appends an entry exactly as `/remember` would; `forget` deletes one by the short id shown beside it in context. Both write to the file on disk and **take effect at the next session start**, not mid-conversation. That is deliberate: memory sits in the cached, project-stable part of the prompt, and rewriting it mid-session would force a full re-prefill of everything after it, which on a local model is the most expensive thing plank can do. The tool's own reply says so, so a model that just saved something and sees no change in its context knows that is expected.

### Budgets, not truncation

Each type has its own byte budget for what renders into context (4096 for `user` and `feedback`, 6144 for `project`, 2048 for `reference`, set under `memory.budgets`). When a type is over budget, entries are dropped least-valuable first: pinned entries never go, then the least-used, then the least recently used, and among entries the counters cannot separate the newest wins. The usage counters live in a `MEMORY.md.meta.json` sidecar beside each file. It is advisory: delete it and memory loads exactly as before, with every counter at zero.

### The extraction pass

With `memory.autoExtract` on (it is **on by default** since 5.1.7), plank runs a pass at the end of any turn that produced no tool calls. It hands the model the new part of the conversation and the current entries, and asks for a JSON list of verdicts: add an entry, update one, delete one, or mark one as having been useful. plank applies the verdicts itself; the pass cannot run tools, cannot touch anything but memory, and reads only the part of the transcript it has not already seen. Every change it makes is appended to `~/.plank/memory-log.jsonl`, which `/memory log` prints.

The pass is not free, and it is worth knowing what you are paying for. It runs synchronously at the end of the turn and costs one extra generation plus a KV snapshot each time, which on a local model is a visible pause after each answer. `memory.extractEveryNTurns` thins it out, and `"memory": {"autoExtract": false}` in `settings.json` turns it off entirely. The full design, including what each failure mode looks like, is in [`docs/MEMORY.md`](https://github.com/aovestdipaperino/plank/blob/main/docs/MEMORY.md).

## Compaction

When the conversation approaches the window, plank reclaims space in escalating steps, cheapest first.

**Microcompact** clears the *bodies* of old tool results, keeping the newest few. No model round-trip, no summary, nothing lost that the model still needs — a file it read twenty turns ago and has not mentioned since is the cheapest thing in the transcript to give up.

**Full compaction** runs when that is not enough. plank asks the model for durable task state, then rebuilds the live transcript as: system prompt + summary + the recent verbatim tail + a budgeted re-injection of recently read files. You keep the shape of the conversation and the exact text of its recent part.

```
/compact
```

runs it on demand — useful just before handing the model a big new job, so it starts with room.

**Steering the summary.** Anything after `/compact` is extra instruction for that one pass:

```
/compact keep the failing test cases verbatim
/compact focus on the parser work and drop the deployment detour
```

Your instruction is *added* to what plank already asks for, not substituted for it, so the summary keeps its structure and gains your emphasis. Use it when you know which thread matters next and the default summary would flatten it. Automatic compaction has no instructions, and asks exactly what it always did.

While a pass runs the status bar shows its progress and the window title reads `🗑️ compacting...`. `Esc` interrupts it, which leaves the conversation exactly as it was.

**If it fails, nothing is lost.** An interrupted pass, or one where the model returns no usable summary, leaves the transcript untouched and abandons the turn rather than rebuilding on a bad summary. You will see `Compaction produced no summary; keeping the previous conversation state.` — retry it, or `/compact` with an instruction to nudge the model.

Two things deliberately survive compaction: the **task list** (that is what the `task` tool is for) and **memory**.

And one thing makes compaction reversible: a `/checkpoint` taken before it stores the whole transcript, so `/rollback` reconstructs the pre-compaction conversation exactly. See [Sessions](06-sessions.md).

## Context size

The window defaults to 1048576 tokens. Set it with `-c N` or `engine.ctx`. Bigger costs memory; smaller compacts sooner.

If plank feels unexpectedly slow or forgetful, check the startup line — a `settings.json` that shrank `ctx` or moved you off Metal is otherwise invisible once the UI is up, so plank names what is in force:

```
plank: settings in effect (/path/to/.plank/settings.json): threads=3, backend=cpu, ctx=65536
```

## Thinking

The model reasons before it answers. `--think-low` (the default), `--think`, `--think-max`, and `--nothink` set the effort; `ui.showThinking` controls whether you see it, and clicking the footer's brain toggles that for the session without writing the setting. On DeepSeek V4.1 Flash, which has a native reasoning-effort dial of 0 to 100, those names are labels for the numbers the model is actually told (low is 25, medium is 75, max is 100), `/think <n>` sets a number directly, and the footer shows the number in force. `off` stays `off`, because thinking-disabled is a distinct state rather than effort zero. Low is the default because on the same coding task it finished a third faster than medium with the same result, and its reasoning reads as a plan rather than a list of second thoughts. Hiding thinking does not stop it — the model still produces it, and it still occupies context.

## Token usage on hosted providers

```
/usage
```

reports billed input and output tokens for the session, including Anthropic cache reads, cache writes, and the hit rate. Local engines have nothing to bill, so it is a provider-only report.

## Generation and prefill speed

```
/toks
```

draws two braille line charts side by side in the theme green, generation speed on the left and prefill speed on the right: tokens per second, sampled once a second while the model works, newest at the right, each with its own current, average, minimum and maximum rates underneath. Two panels rather than one because the question a slow pass raises is comparative, whether it is prefill-bound or decode-bound, and that reading is only immediate when both lines sit on the same rows. The x axis is time spent in the engine, so tool rounds and turns leave no mark on it. It opens in the same dismissable panel as `/usage`, and typed during a turn it redraws on every status tick, so you can watch the rate settle as a long answer streams. On the piped REPL it prints as plain text.

---

Next: [Configuration →](08-configuration.md)

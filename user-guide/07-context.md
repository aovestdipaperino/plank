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

The conventional place for "things anyone working in this repo needs to know": build commands, architecture, house style, gotchas. plank finds it and injects it at session start. It is the only instructions file plank reads: a project root that has a `CLAUDE.md` but no `AGENTS.md` gets an `AGENTS.md` symlink to it the first time you start plank there, and a project with neither is asked whether to generate one. Headless runs (`--non-interactive`) do neither.

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

Without `user`, the entry goes to project memory. Entries are dated bullets.

To edit them rather than append, `/memory` opens both files as one buffer in the built-in editor, each between markers naming its scope and path; on save the buffer is split back along those markers and only the files whose text changed are written.

Four kinds of entry are worth keeping, and they share one test: **facts the model cannot re-derive from the repository.**

| Type | What |
|---|---|
| `user` | who you are |
| `feedback` | corrections on how you want work done, and why |
| `project` | goals and constraints not visible in the code |
| `reference` | external URLs, tickets, dashboards |

Do not record what the code, the git history, or `AGENTS.md` already says. That is context you are paying for twice.

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

The model reasons before it answers. `--think-low` (the default), `--think`, `--think-max`, and `--nothink` set the effort; `ui.showThinking` controls whether you see it. Low is the default because on the same coding task it finished a third faster than medium with the same result, and its reasoning reads as a plan rather than a list of second thoughts. Hiding thinking does not stop it — the model still produces it, and it still occupies context.

## Token usage on hosted providers

```
/usage
```

reports billed input and output tokens for the session, including Anthropic cache reads, cache writes, and the hit rate. Local engines have nothing to bill, so it is a provider-only report.

---

Next: [Configuration →](08-configuration.md)

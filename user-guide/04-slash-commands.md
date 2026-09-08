[← The interface](03-the-interface.md) · [Index](README.md) · Next: [Tools →](05-tools.md)

# 4. Slash commands

A line beginning with `/` is a command, not a prompt. Unknown slash lines are passed to the model as ordinary text. `/help` prints the full usage text; completion offers the known commands as you type.

Every command below works identically in the TUI and the plain REPL.

## Session lifecycle

| Command | What it does |
|---|---|
| `/new`, `/clear` | start a fresh session, dropping the current transcript |
| `/save` | write the current session to disk (sessions also save automatically) |
| `/list` | list saved sessions, most recent first |
| `/switch <id>` | load another saved session by name or number |
| `/resume [prefix]` | resume a saved session; no argument picks the most recent, or shows a list |
| `/del <id>` | delete a saved session |
| `/tag <text>` | label the current session so it is recognizable in `/list` |
| `/rename <name>` | change the name later saves use; what is already on disk keeps its old name |
| `/strip <id>` | drop a saved session's KV payload to reclaim disk; the transcript survives and a later resume re-prefills it |
| `/history` | reprint recent turns |
| `/quit`, `/exit` | leave (the session is saved) |

See [Sessions](06-sessions.md).

## Branching and rollback

| Command | What it does |
|---|---|
| `/checkpoint [name]` | mark a named return point in this session |
| `/rollback [name]` | return to a checkpoint; the discarded tail is kept as `pre-rollback` |
| `/tree` | draw the session tree and number its fork points |
| `/fork [n]` | rewind to fork point `n` and explore a different path; no argument shows the tree |
| `/clone` | freeze the current branch and continue on a copy |

## Context

| Command | What it does |
|---|---|
| `/context` | report context-window usage by category |
| `/compact [instructions]` | compact the conversation now, rather than waiting for the automatic pass; an argument steers what this one summary keeps |
| `/usage` | billed token counts for the session (hosted providers, including cache hit rate) |
| `/rate [+\|-] [note]` | rate the last turn (thumbs up by default) with an optional note; works in the TUI and the plain REPL alike |
| `/remember [user] <fact>` | append a dated entry to project memory (or user memory with `user`) |
| `/memory` | open user and project memory as one buffer in the built-in editor; edits are split back to the right file on save |
| `/init` | have the model read the repo and generate an `AGENTS.md` |

See [Context](07-context.md).

## Extensions and inspection

| Command | What it does |
|---|---|
| `/mcp` | connected MCP servers and the tools they expose |
| `/skills` | skills available to the model |
| `/templates` | your `{{var}}` prompt templates |
| `/agent` | named subagents you can delegate to, and which engine each runs on |
| `/hooks` | which hooks are configured and on what events |
| `/plugins` | loaded plugins, where each came from, what it contributes, and any warnings |
| `/tasks` | the model's task list |

See [Extending plank](09-extending.md).

## Asides and delegation

| Command | What it does |
|---|---|
| `/btw <question>` | ask a side question mid-task; answered in a split panel, nothing written to the conversation |
| `/subagent <task>` | delegate a task to a general-purpose subagent; only its final report enters the transcript, and the turn continues from it |
| `/subagent:<name> <task>` | the same, using a named definition. The `:<name>` shows green while you type if it exists and red if it does not |

`/btw` genuinely suspends the running generation, answers, and resumes byte-for-byte with no re-prefill (`safety.btwSuspend`, on by default). With suspend off, the question is queued and answered at the next generation boundary instead.

## Configuration and runtime

| Command | What it does |
|---|---|
| `/config` | open the interactive settings form |
| `/config <section>.<key> <value>` | set one setting, e.g. `/config ui.showThinking false` |
| `/debug [on\|off]` | override the `--debug` switch: mirror the raw model stream to a running `turbo-debug-console` (on connects and backfills at once); bare `/debug` reports the state |
| `/power <1..100>` | cap GPU power draw for this run; shown as `(local ⚡60%)` in the status bar |
| `/mtp [on\|off]` | turn speculative decoding (multi-token prediction) on or off for this session; bare `/mtp` reports the state |
| `/temp [0..100]` | set the sampling temperature; refused while `/mtp` is on, which pins it at 0 |
| `/loopguard [on\|off]`, `/lg` | arm or silence the loop guards. The one mutating command that also works mid-turn |
| `/notify <mode>` | change notification mode for this session |
| `/version` | the running version |
| `/help` | full command and flag reference |

The model can also ask for a compaction itself, with the `compact` tool, and carries on from the summary. It is told not to reach for it unprompted: only you know whether the detail a summary drops still matters, and plank compacts on its own when the window gets tight.

Speculative decoding verifies its drafts by argmax, so it only runs at temperature 0: `/mtp on` pins the temperature there and `/mtp off` gives you back the one you were sampling at. `/mtp on` is refused without loaded draft support — that is chosen at startup (`--mtp`, `--mtp-model`) and cannot be loaded into a running engine. The status bar shows `✨` while speculation is on, with its per-step figures beside it, and `🌡 0.60` while it is off.

`/loopguard off` silences every rung of every guard: the repetition guard and its think budget, the repeated-tool-call guard and its no-progress budget, and the repeat advisory. Unlike other settings commands it takes effect on a turn that is already generating, because the moment you want the guards out of the way is usually while they are firing. `🔁` in the status bar means they are armed; `♻ looping` means one has actually seen a cycle. The switch is session-only and is never written to disk (the underlying setting is `tools.loopGuards`).

`/config` changes write to `./.plank/settings.json` and apply immediately. In the TUI, a bare `/config` opens the form and `/config <key> <value>` sets and persists the value directly, the same as on the plain REPL. See [Configuration](08-configuration.md).

## Output and diagnostics

| Command | What it does |
|---|---|
| `/export [md\|html] [path]` | write the transcript to a shareable file (markdown by default, auto-named; HTML is standalone) |
| `/kvcache` | browse the KV cache as a tree: what each snapshot is, what it was built on, its size, how often it has been used, and when it expires |
| `/kvcache gc\|pin\|unpin\|rm` | sweep expired entries now, or pin, unpin or delete one by fingerprint prefix |
| `/insights [fast\|fresh]` | a usage report computed from every saved session, written to `~/.plank/usage-data/report.html` (`fast` skips the model-written prose, `fresh` forces it to be written again) |
| `/repro [note]` | dump the exact engine input and runtime knobs to `~/.plank/repro/` for a bug report; the file's path is copied to the clipboard |

`/repro` is the one to reach for when you want to report a problem: it captures the rendered prompt the engine would see plus the model, backend, context size, sampling settings and think mode, in a single self-contained file. It never touches the live session.

Under `--debug` (or after `/debug on`) plank dumps on its own as well: quitting writes `repro-quit-<timestamp>.md`, and a panic writes `repro-panic-<timestamp>.md` with whatever transcript the session had last rendered. Neither needs you to remember `/repro` before the session ends.

`/insights` computes **every number in code** and uses the model only for prose it cannot replace — a failed or skipped model call costs the report its narrative, never its statistics.

The report is **differential**. Per-session statistics have always been cached and recomputed only for sessions that changed; the written sections now work the same way. plank remembers the last report in `~/.plank/usage-data/last-report.json` and reuses its prose until ten sessions — or a tenth of your history, whichever is smaller — are new or have been written to since. A section the previous run failed to produce is not reused, so it gets written on the next run rather than staying missing. `/insights fresh` writes everything again regardless, for when the last answer was wrong rather than stale.

When there is a previous report to compare against, the new one opens with a **Since** strip: sessions new or updated, prompts, lines, files, commits, and any tool or friction category that was not there last time. It is subtraction of two deterministic aggregates, so it costs nothing and cannot be wrong the way a written summary could.

Suggestions are made to be used, not read: every snippet, prompt and instruction in the report has a **Copy** button, and the **Worth putting in AGENTS.md** list arrives as a checklist — untick what you disagree with, press **Copy all checked**, and paste the rest into plank. The report stays a single self-contained local file: the clipboard handler is inlined next to the stylesheet, with a fallback for `file://`, where the browser clipboard API is unavailable.

Among the written sections is **Features to try**: two or three of plank's own [extension points](09-extending.md) — a skill, a template, a subagent, a hook, an MCP server — chosen from what your sessions actually show, each with a ready-to-run snippet that sets it up. The report knows what you already have installed (it reads the live skill, template, subagent, hook and MCP rosters), so it recommends what is missing rather than what is already there.

## Editing a file

| Command | What it does |
|---|---|
| `/open [path]` | edit an existing file in the built-in editor; with no path, reopens the last file a tool call edited this session |

`/open` hands the terminal to the same editor `Ctrl-G` uses, with the file loaded: `Ctrl-S` saves, `Esc` discards. It is the fast way to fix up an edit the model just made — bare `/open` needs no path at all — or to look at a file without spending a turn on it.

It edits, and only edits. A path that does not exist is refused rather than created, so a typo cannot leave an empty file behind, and so are directories, binary files, and anything over 32 MB. An untouched `Ctrl-S` writes nothing. Saving follows a symlink to the file it points at instead of replacing the link. TUI only; in the plain REPL the command is not available.

## Remote control

| Command | Status |
|---|---|
| `/remote` | recognized as a command |
| `/grant [session]` | approves a remote client's control request |

Both are part of the remote-control surface described in [Remote and hosted engines](10-remote-and-providers.md). `/grant` matters only when the bridge was started with `/rc ask`, which withholds control from attaching clients until you approve each request; plain `/rc` grants it up front and leaves `/grant` with nothing to answer.

## Not in `/help`

Five arcade commands are deliberately absent from `/help` and from the completion popup. They are covered in [The arcade](11-arcade.md), and they can be removed entirely with `ui.easterEggs: false`.

## Skills and templates as commands

Any installed skill or template also becomes a slash command: `/<name> [args]`. Built-in commands always win over a template or skill with the same name. `/skills` and `/templates` list what is available.

---

Next: [Tools →](05-tools.md)

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
| `/strip <id>` | trim a saved session's oldest turns to reclaim space |
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
| `/remember [user] <fact>` | append a dated entry to project memory (or user memory with `user`) |
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
| `/power <1..100>` | cap GPU power draw for this run; shown as `(local ⚡60%)` in the status bar |
| `/notify <mode>` | change notification mode for this session |
| `/version` | the running version |
| `/help` | full command and flag reference |

`/config` changes write to `./.plank/settings.json` and apply immediately. See [Configuration](08-configuration.md).

## Output and diagnostics

| Command | What it does |
|---|---|
| `/export [md\|html] [path]` | write the transcript to a shareable file (markdown by default, auto-named; HTML is standalone) |
| `/insights [fast]` | a usage report computed from every saved session, written to `~/.plank/usage-data/report.html` (`fast` skips the model-written prose) |
| `/repro [note]` | dump the exact engine input and runtime knobs to `~/.plank/repro/` for a bug report |

`/repro` is the one to reach for when you want to report a problem: it captures the rendered prompt the engine would see plus the model, backend, context size, sampling settings and think mode, in a single self-contained file. It never touches the live session.

`/insights` computes **every number in code** and uses the model only for prose it cannot replace — a failed or skipped model call costs the report its narrative, never its statistics.

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
| `/grant` | recognized; approves a remote client's control request |

Both are part of the remote-control surface described in [Remote and hosted engines](10-remote-and-providers.md). The local `/grant` handler is not wired into the UI yet, so today a remote client that needs to take control should be started against a server run with `--control-allow`.

## Not in `/help`

Five arcade commands are deliberately absent from `/help` and from the completion popup. They are covered in [The arcade](11-arcade.md), and they can be removed entirely with `ui.easterEggs: false`.

## Skills and templates as commands

Any installed skill or template also becomes a slash command: `/<name> [args]`. Built-in commands always win over a template or skill with the same name. `/skills` and `/templates` list what is available.

---

Next: [Tools →](05-tools.md)

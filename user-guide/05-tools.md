[← Slash commands](04-slash-commands.md) · [Index](README.md) · Next: [Sessions →](06-sessions.md)

# 5. Tools

Tools are what turn a chat model into an agent. The model asks for one, plank runs it, feeds the result back, and the model continues — repeating until a generation asks for nothing more. That whole loop is one *turn*.

You do not invoke tools directly; you ask for an outcome and the model chooses. What you control is the *policy*: the sandbox, plan mode, which extensions are installed, and how much of the machinery you see.

## The built-in set

### Files

| Tool | What it does |
|---|---|
| `read` | read a file (or a slice of one) |
| `more` | continue reading where the last `read` stopped |
| `write` | create a file, or overwrite one |
| `list` | list a directory |
| `glob` | find files by name pattern across a tree — `**` crosses directories, `*` stays within one component |
| `edit` | replace a region of a file, anchored on surrounding text |
| `search` | search file contents |

`edit` and an overwriting `write` render as **diff cards** in the TUI: an `Update(path)` header, an added/removed summary, and red/green `@@` hunks. A genuinely new file streams its contents dimmed as it is written, so you can watch it appear.

### Shell

| Tool | What it does |
|---|---|
| `bash` | run a shell command |
| `bash_status` | poll a running job's output |
| `bash_stop` | stop a running job |

Bash commands are **tracked jobs**, not blocking one-shot calls: each owns a process, reader threads, and an output file, so the model can start a long build, do something else, and check back. The first observation is head-biased so headers and early errors are visible; later ones are tail-biased so you see recent output rather than the top of a log.

### Documents

`read` on a `.pdf` transparently converts the file to Markdown first, so a PDF is just a readable file:

```
summarize the first few pages of Claude-Code-Manuale-Completo.pdf
```

Conversion uses spatial text extraction, with OCR filling in pages that have no text layer — so both born-digital and scanned PDFs work. The result is cached in `~/.plank/doc-cache/`, keyed by content, so a second read of the same document costs nothing.

Everything else behaves normally: paging, line numbers, and `more` continuation all work, and the model only ever sees *your* path, never the cache path. A file that cannot be parsed comes back as a plain tool error.

PDF is currently the only converted format.

### Web

| Tool | What it does |
|---|---|
| `google_search` | search the web |
| `visit_page` | fetch and read a page |

Web access asks for your consent the first time, and you can grant it standing so it stops asking.

The pattern that works is to say what you want done with the results, not just what to look up:

```
search the web for the latest news about <topic> and summarize
```

The model searches, opens the pages worth opening, and reports. If it answers from memory when you wanted current information, say so — "it's a different one, find it online" — and it will go look.

### Agent machinery

| Tool | What it does |
|---|---|
| `task` | maintain a task list that survives compaction |
| `agent` | delegate a bounded sub-task to a fresh subagent |
| `skill` | invoke an installed skill by name |
| `ask` | ask you a multiple-choice question |
| `EnterPlanMode` / `ExitPlanMode` | enter and leave read-only plan mode |
| `mcp_describe` | fetch the full schema of a non-primary MCP tool |
| `mcp_list_resources` / `mcp_read_resource` | list and read MCP resources |
| `mcp__<server>__<tool>` | any tool from a connected MCP server |

## The bash sandbox

On macOS, model-initiated shell commands run under `/usr/bin/sandbox-exec` with a generated Seatbelt profile: **read everywhere, write only under the working directory, temp directories, and any roots you allow**. It is on by default, because a model-chosen command should not be able to write outside the project it was pointed at.

Commands *you* type with `!` or `!!` are never sandboxed — you typing the command is the authorization.

Configure it in `~/.plank/sandbox.json`, overlaid by `./.plank/sandbox.json`:

```json
{
  "enabled": true,
  "writablePaths": ["/some/extra/root"],
  "excludedCommands": ["git push*", "brew *"]
}
```

Scalars come from the most specific file; lists concatenate. `excludedCommands` glob-matches the whole command line and skips the sandbox for it — a convenience escape hatch, **not a security boundary**.

Turn it off for a run with `--no-sandbox`, or permanently with `safety.sandbox: false`.

## Plan mode

When a task is risky or ambiguous, the model can call `EnterPlanMode`. While it is active, every workspace-mutating tool is refused — `write`, `edit`, `bash` — and only read-only tools work, so the model researches and designs without touching anything.

It leaves by calling `ExitPlanMode` with a proposed plan, which you approve or reject. On approval the gate lifts and it may edit; on rejection plan mode stays on and it refines the plan.

You can also just ask for it: "plan this out before you touch anything."

## Questions

When a turn is genuinely ambiguous, the model can call `ask` instead of guessing: a multiple-choice question with a short header and two to seven options, shown as a panel in the TUI or a numbered list in the REPL. It blocks until you answer, and degrades cleanly when there is nobody to ask (headless mode).

`ask.maxOptions` caps how many options one question may offer (default 7; the minimum of 2 is fixed).

## Task lists

The `task` tool keeps a plan the model can add to and update — statuses are `pending`, `in_progress`, and done. The list is **model-visible and survives compaction**, which is the point: a long job does not lose its plan when the conversation is summarized. `/tasks` shows it, and the status bar carries a task counter.

## Subagents

The `agent` tool hands a self-contained sub-task to a fresh subagent working in its own scoped context, and returns **only its final report** to the main conversation. This is how a long research detour stays out of your transcript.

Delegation is bounded: a subagent cannot itself delegate. You can also start one yourself with `/subagent <task>`, and define named subagents with their own instructions — see [Extending plank](09-extending.md).

## Watching (or not watching) tools run

By default the UI stays clean: tool-call banners and tool output are hidden, and the status bar just names what is running. Turn them on when you want to see the mechanics:

```
/config ui.showToolCalls true
/config ui.showToolResults true
```

Neither changes what the model receives — the tools run and the results are fed back either way.

## Tool calls inside thinking

By default plank dispatches tool calls the model emits inside its thinking block. Set `engine.thinkingToolCalls` to `false` for strict parity with the ds4 reference, where such a call is instead recovered forward by force-closing the thinking block.

---

Next: [Sessions →](06-sessions.md)

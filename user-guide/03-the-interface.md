[← Getting started](02-getting-started.md) · [Index](README.md) · Next: [Slash commands →](04-slash-commands.md)

# 3. The interface

## The screen

The TUI is three regions: **scrollback** (the conversation), the **prompt**, and the **status bar** along the bottom.

Assistant replies render as markdown — headings, lists, tables, and fenced code blocks with tree-sitter syntax highlighting. The model's thinking appears dimmed above its answer (turn it off with `ui.showThinking`). File edits render as git-style **diff cards**: an `Update(path)` header, an added/removed summary, and red/green `@@` hunks. A brand-new file streams its contents dimmed as it is written.

The status bar is two rows. The top one answers "which tree am I in" and holds still: the working directory and git branch, and nothing else. The bottom one carries everything that churns — where inference is running, the reasoning level, a context-usage gauge, an activity throbber, what the model is doing, generation stats, the task counter, and the remote marker. When a tool is running its name sits in the notification slot and blinks; otherwise a rotating tip appears there.

**Where inference runs** is named on the second row, and there can be more than one answer:

```
api.regolo.ai, (local ⚡100%)
```

Every engine in play is listed in the order it first appeared, so a hosted main agent beside a `provider: local` subagent shows both. The local engine carries its GPU power share (`/power`, `--power`), because the cap applies to that engine and to no other.

**The brain blinks** while the local model is prefilling or generating. It is the one signal that says *which* engine is working right now, which is otherwise invisible when a local subagent runs under a hosted main agent. It blinks in step with the elapsed and tokens/second readouts beside it.

While plank is compacting, the throbber-and-verb line is replaced by a flashing `compacting` with a progress bar and percentage:

```
compacting ▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 21%
```

Most of the bar tracks re-reading the conversation, which is the slow part; the tail tracks the summary being written. It goes away as soon as the pass ends. See [Compaction](07-context.md#compaction).

Every turn closes with a line of its own before the prompt comes back:

```
✻ Planked for 0h 02m 20s
```

That clock covers the whole turn — every generate-and-tools round, plus any lines you typed while it was busy and it absorbed — not just the last pass. It is the number to quote when something felt slow.

## Keys

### Editing the prompt

| Key | Action |
|---|---|
| `Enter` | submit |
| `Shift+Enter`, `Alt+Enter`, `Ctrl-J` | newline instead of submitting |
| `Ctrl-A` / `Home` | start of line |
| `Ctrl-E` / `End` | end of line |
| `Ctrl-B` / `Ctrl-F` | left / right one character |
| `Alt+Left` / `Alt+Right` | left / right one word |
| `Ctrl-K` | delete to end of line |
| `Ctrl-U` | delete to start of line |
| `Ctrl-W` / `Alt+Backspace` | delete previous word |
| `Ctrl-D` | delete forward, or quit on an empty line |
| `Up` / `Down` | walk prompt history |
| `Ctrl-P` / `Ctrl-N` | history back / forward (REPL) |
| `Ctrl-L` | clear the screen (REPL) |
| `Ctrl-G` | open the current prompt in an editor |

`Shift+Enter` needs the kitty keyboard protocol to be reported, which not every terminal does — `Alt+Enter` and `Ctrl-J` work everywhere.

**`Ctrl-G`** is the escape hatch for a prompt too long to edit inline. By default it opens plank's built-in editor (a fork of Microsoft Edit, running in-process, no external dependency). Set `ui.builtinEditor` to `false` to shell out to `$EDITOR` instead. Either way, what you save comes back into the prompt; cancelling keeps what you typed.

The same editor also edits files on disk: **`/open [path]`** takes over the terminal with the file loaded, `Ctrl-S` saves and `Esc` discards. See [Slash commands](04-slash-commands.md#editing-a-file).

### During a turn

| Key | Action |
|---|---|
| `Ctrl-C` | interrupt the generation (at an idle prompt: clear the input line) |
| `Esc` | interrupt the generation |
| `←` on an empty prompt | step into the agent roster, when there is one |
| mouse wheel / trackpad | scroll the scrollback |
| click-drag | select text to copy |

With an arcade game open, the first `Ctrl-C` closes the game and a second interrupts the model — you are never locked out of stopping a turn.

`Esc` at an idle prompt dismisses a `/btw` panel left open from an earlier turn, which is the only way it closes.

### In a question panel

The `ask` tool's panel takes `Up`/`Down` to move, `Space` to toggle an option when the question is multi-select, `Enter` to answer, `Esc` to decline, and `Ctrl-C` to interrupt.

## The agent roster

The moment anything is delegated — by you with `/subagent`, or by the model with the `agent` tool — a roster appears below the status bar: `main` first, then one row per run with a state bullet (`●` working, `○` finished), the agent's name, the task it was given on one line, and, flush right, how long it has been going and what it has spent in tokens.

```
● main
● reviewer   check the diff on this branch          1m 12s  · 4.1k
○ researcher find every caller of load_config()        48s  · 9.7k
```

A fan-out gets a row each, with its own output buffer, so concurrent agents never overwrite one another. `←` on an empty prompt steps into the roster, `←`/`→` walk the rows, `Enter` expands the selected agent's output over the transcript with its own scroll position, and `Esc` comes back.

It is a live readout: it appears with the first agent and goes away with the last, staying put only while you are reading it, and `←` brings a finished roster back so a report you delegated is still reachable. The last eight runs are kept. The transcript itself gets only a one-line signpost, which is the point of delegating in the first place — see [Extending plank](09-extending.md).

The plain REPL has no roster and prints subagent output inline instead; `--non-interactive` stays silent so its stdout protocol is not corrupted.

## `@` file completion

Type `@` in the prompt and a fuzzy-completion popup offers file paths from the working tree; pick one and the path is spliced into your message. `Tab` accepts.

- `ui.popupRows` sets how many rows it offers (default 15).
- `ui.respectGitignore` decides whether untracked files that `.gitignore` excludes are offered (default `true`).
- `ui.indexRefreshSecs` is how long the file index is trusted before it is rebuilt (default 5).

## `!` and `!!` — run a shell command yourself

Prefix a line with `!` and it runs in your shell, in plank's working directory, with the output streaming into the screen as it is produced:

```
!cargo test --lib parser
```

The number of `!`s decides whether the model ever sees it:

| | What the model gets |
|---|---|
| `!command` | the command and its output are recorded in the conversation, so the model has them as history on your next message |
| `!!command` | nothing — the output is yours alone |

Neither form starts a turn. A `!` command does not make the model respond; it just means that when you *do* send your next message, the model can already see what happened. So `!cargo test` followed by "fix the failure" works without pasting anything, while `!!git log` keeps a bit of poking around out of a context you are paying for.

Recorded output is capped (200 lines, 16 KB) with a truncation marker, so one runaway command cannot flood the conversation. If you want the model to act on something *right now* rather than on your next message, ask it in a normal turn and let it run `bash` itself.

Either form is **your** command, not the model's, so it is **never sandboxed** — you typing it is the authorization. Both are also the right way to do anything interactive (a login flow, an editor, a pager) that the model's `bash` tool cannot drive.

`Esc` or `Ctrl-C` kills a running command, and `Up`/`Down` on a line that starts with `!` walks only your previous shell commands.

The marker is coloured as you type, by where the output goes: **red `!`** feeds the command and its output to the model, **green `!!`** keeps it between you and the shell. That is the only difference between the two forms and the only thing you cannot see once the line is typed. When the command finishes, plank says `done.` in green — a command that printed nothing is otherwise indistinguishable from one still running. An interrupted command says `[interrupted]` instead, and a non-zero exit adds `[exit code: N]`.

## Pasting images

Paste an image (or a path to one) into the prompt and plank attaches it: PNGs are downsampled, deduplicated by content into `~/.plank/image-cache/`, and attached to your message.

**Be aware of what this does and does not do today.** The local ds4 engine is text-only, so what reaches the model is the *path* to the image, not its pixels. A pasted screenshot of a stack trace is a filename as far as the model is concerned — it can pass the path to a tool, but it cannot see the picture. A vision model to close that gap is designed but not implemented.

If you need the model to act on what is in an image, transcribe the important part into your prompt.

## Reading PDFs

`read` converts PDFs to Markdown transparently, so you can just point at one:

```
summarize the first few pages of manual.pdf
```

See [Tools](05-tools.md).

## Notifications and the window title

Long turns end with a native macOS notification: your prompt as the headline, the tail of the answer as the body (`interrupted` for an aborted turn). The terminal title tracks the current task, e.g. `🪵 plank - fix the bug…`, and names the phase when plank is busy with something that is not your turn: `🗑️ compacting...` while it reclaims context, `👀 introspecting...` during `/insights`. The title it displaced comes back afterwards, so a compaction mid-turn returns the title to your prompt.

- `ui.notifications` — `always`, `unfocused` (only when the terminal is not focused), or `never`.
- `ui.notifyAfterSecs` — minimum turn length before a completion notification fires (default 10). Awaiting-input notifications ignore it.
- `/notify` changes the mode for the running session.

## Animation, screensaver, exit

- `ui.reducedMotion` collapses every animation — throbber, shimmer, pulse, flash, stall-fade — to a static fallback.
- `ui.screensaver` sets how long the TUI must sit idle before a screensaver takes the screen: `1m`, `2m`, `5m`, or `never`. Any key or mouse event dismisses it, and it never appears mid-turn or over a dialog. `ui.screensaverFace` picks which one — see [The arcade](11-arcade.md#the-screensaver).
- `ui.crtOff` plays a CRT power-off animation of the final frame when you exit cleanly.

Leaving prints the session's token totals and, per model, the fastest sustained rates it reached:

```
peak DeepSeek V4 Flash  prefill 167.1 tok/s  ·  generation 16.8 tok/s
```

Both figures are scoped to this session and never persisted — a peak from last week was a different engine build on a differently loaded machine. They are the quickest way to see what a flag like `--dspark` actually did on your hardware.

## Quieting the display

Three settings control how much of the machinery you see. None of them change what the *model* receives:

| Setting | Default | Off means |
|---|---|---|
| `ui.showToolCalls` | `false` | tool-call banners hidden; tools still run |
| `ui.showToolResults` | `false` | tool output not echoed; the model still gets it |
| `ui.showThinking` | `true` | thinking hidden from the display; the model still thinks |

---

Next: [Slash commands →](04-slash-commands.md)

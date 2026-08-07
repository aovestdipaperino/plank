[← Installation](01-installation.md) · [Index](README.md) · Next: [The interface →](03-the-interface.md)

# 2. Getting started

## Launching

```sh
plank              # interactive
plank --help       # every flag
```

plank picks its front end from your terminal, not from a flag:

| Situation | Front end |
|---|---|
| stdin and stdout are both a TTY | the full-screen **Ratatui TUI** |
| either end is piped | the plain **line REPL** |
| `--non-interactive` | **headless** stdin protocol |
| `-p "…"` / `--prompt "…"` | run one prompt, print the reply, exit |

The TUI and the REPL support the same slash commands and the same tools. The TUI adds markdown rendering, syntax highlighting, mouse scrollback, the status bar, panels for `/btw` and questions, and the arcade.

## Your first turn

Start plank in the project you want to work on:

```sh
cd ~/Code/my-project
plank
```

At startup plank gathers session context — git status, the date, any `AGENTS.md` it finds — and hands it to the model before your first message, so you can open with "why is the parser dropping the last token?" rather than explaining where you are.

Then type. A few things to know immediately:

- **You can keep typing while it works.** Each turn runs on a worker thread; the prompt stays live during generation and your next message queues.
- **Ctrl-C interrupts the turn**, it does not kill plank. At an empty prompt it clears the input line.
- **Esc also interrupts** a running generation.
- **The turn ends when the model stops calling tools.** plank runs generate → dispatch tools → feed results back → generate again, until a generation asks for nothing.

## It is not only for code

plank is built as a coding agent, but nothing about it is restricted to code. In practice it gets used for:

- **Questions with no repository involved** — "how do you write a lock-free queue?", "how is the diagonal function computed in a residual number system?" Ask; there is no requirement that a session touch a file.
- **Web research** — "search the web for the latest news about X and summarize". The model searches, visits pages, and reports back. See [Tools](05-tools.md).
- **Documents** — point it at a PDF and it reads it: "summarize the first few pages of manual.pdf".
- **Other languages** — the model answers in whatever language you write in.
- **Writing** — prose, posts, poems. Handing it a style file ("write a post about X using the style in `style.md`") works better than describing the style.

Short follow-ups work the way they do in a conversation: "what else", "keep going", "now one in Pascal", "summarize it here". You do not need to restate context that is already on screen.

## Working directory

plank operates in the directory it was launched from. Tools resolve relative paths against it, the bash sandbox writes only inside it, and project-scoped config (`./.plank/`, `./.mcp.json`) is read from it.

```sh
plank --chdir ~/Code/other-project
```

changes the working directory before starting. One caveat: project settings are read from the launch directory, so `./.plank/settings.json` does *not* follow `--chdir`.

## One-shot mode

```sh
plank -p "summarize what changed in the last three commits"
```

Runs a single prompt with all tools available, prints the reply, and exits. Works with hosted providers too:

```sh
plank --provider anthropic --model claude-sonnet-4-5 -p "review src/parser.rs"
```

## Headless mode

`--non-interactive` disables the interactive UI and reads a line protocol from stdin — the mode to drive plank from a script or another program. For driving the *TUI* from a test harness, see `--ui-remote` in [Remote and hosted engines](10-remote-and-providers.md).

## Leaving and coming back

Type `/quit` (or `/exit`, or Ctrl-D at an empty prompt). Your session is saved automatically.

```sh
plank /resume            # most recent session
plank /resume amused     # by name prefix
```

or from inside plank, `/resume` with no argument to pick from a list. A resumed session replays through the same renderer, so history comes back as markdown with thinking dimmed — and plank restores the engine KV alongside the transcript, so it does not have to re-read the whole conversation before your first new message.

On the way out plank prints what the run cost:

```
Session stats  ↓ 128,209 ↑ 8,269  ·  12:20
  glm5.2          ↓ 120,000 ↑ 7,900
  ds4 (local)     ↓   8,209 ↑   369
```

Tokens in and out, and the wall clock. The per-engine breakdown appears only when more than one engine served the session — which is the case worth reading, since it separates what a hosted provider billed you for from what ran on your own machine.

See [Sessions](06-sessions.md) for the full story.

## Telling plank about your project

Two files change how every session in a directory starts:

- **`AGENTS.md`** — project instructions, discovered and injected at session start. `/init` generates one by having the model read your repo.
- **`./.plank/MEMORY.md`** — durable facts about goals and constraints, appended with `/remember`.

See [Context](07-context.md).

## When something is ambiguous

The model can ask you rather than guess: the `ask` tool opens a multiple-choice panel (a numbered list in the REPL) and blocks until you pick. It degrades cleanly when there is nobody to ask.

If a task is risky, the model can enter **plan mode** — read-only until it proposes a plan and you approve it. See [Tools](05-tools.md).

---

Next: [The interface →](03-the-interface.md)

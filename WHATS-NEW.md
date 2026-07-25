# What's new in plank

A short, human-readable highlight reel per release. For the full change list
see the GitHub releases and commit history.

## 2.6.0

The 2.5 beta line, promoted. Six weeks of the cache learning to keep what it
already knows, plus session branching, a handful of commands, and a lot of
progress-reporting that finally tells the truth.

⚡ **`/new` is fast again.** Starting a fresh session used to throw away the
system-prompt cache and rebuild it from scratch — thousands of tokens, twenty
seconds on a large prompt — while the progress bar sat at 100% claiming there
was nothing to do. It looked like a hang because, from the outside, it was
indistinguishable from one. `/new` now puts the cache back to exactly the state
a cold launch has, so the next turn only evaluates your question. On DeepSeek V4
Flash the same `write a haiku` → `/new` → `write a haiku` flow went from 31.7s
to 19.7s, and the token accounting dropped from a hidden 2509-token rebuild to a
7-token prefill. While the cache is being restored the prompt hides behind a
throbber, so you can see the brief pause instead of typing into a frozen line.

📊 **The prefill bar measures the work, not the prompt.** It used to run from
the cached prefix to the end of the prompt, so a warm turn that reused 8000
tokens and prefilled 200 opened at 97% and inched along, while the tok/s figure
beside it already counted only the new tokens. Bar and throughput now describe
the same 200 tokens.

🌳 **Sessions branch.** A conversation is a tree now, not a line. `/fork [n]`
starts a new branch from an earlier prompt of yours, `/clone` duplicates the
branch you are on, and `/tree` shows the shape and which branch is live.
Existing linear sessions load unchanged.

🧠 **The cache is layered.** What rarely changes (system prompt, then your
project's AGENTS.md/CLAUDE.md and local MCP tools) is checkpointed separately
from what changes every session (git status, the date). At launch plank restores
the deepest layer still valid and prefills only from the first thing that
actually differs — and the project layer is shared across every session in that
directory. Superseded snapshots are deleted instead of accumulating by the
hundreds of megabytes.

📤 **`/export [md|html]`** writes the transcript out as Markdown or a
self-contained HTML file.

📝 **Prompt templates.** Markdown files in `~/.plank/templates` or
`./.plank/templates` become `/name` commands, with `{{var}}` interpolation.
Built-in commands can never be shadowed.

⌨️ **A real prompt line.** Ctrl+G opens `$EDITOR` on what you have typed and
brings it back. Alt/Ctrl + arrows move by word, Alt/Ctrl + Backspace/Delete kill
by word, and the emacs bindings (Alt+B/F/D) work too. Long input wraps instead
of scrolling sideways.

🧑‍🔧 **Delegate to a sub-agent.** The model can hand a bounded task to a fresh,
scoped sub-agent with the `agent` tool and get back only its conclusion, instead
of filling the main transcript with the research. It runs as a sidechain off your
conversation and rolls back out afterward. An optional `name` picks one of your
`~/.plank/agents` personas.

📋 **Plan mode.** `EnterPlanMode` puts the model in a read-only phase — it can
research with read/list/glob/search but `write`, `edit`, and `bash` are refused
— until it proposes a plan with `ExitPlanMode` and you approve it. A cheap
course-correction before any edits land. Like the `task` and `agent` tools, it
is off by default: the DS4 model was not trained on it.

🔍 **File changes show as a git diff.** Edits render as a change card — an
`Update(path)` header, an added/removed summary, `@@` hunks in red and green —
and highlighting narrows to the changed words within a line rather than painting
the whole row. A brand-new file streams its contents dimmed as it is written.

🌐 **The web tools grew a browser.** `visit_page` fetches through an embedded
headless browser rather than curl, and `google_search` runs client-side. Web
access asks for consent first, with an "Always allow" option.

🔔 **Notifications and window title.** A turn that runs past 10 seconds ends with
a macOS banner headlined by your prompt; `ui.notifications` picks `always`,
`unfocused`, or `never`, and `/notify` toggles it live. The terminal title tracks
what plank is doing.

⚙️ **`/config`** is an interactive form over every setting (or
`/config ui.showThinking false` straight from the prompt), writing
`./.plank/settings.json` and applying immediately. `ui.showThinking: false`
hides the model's reasoning; `ui.reducedMotion` turns off every animation.

✨ **Animation and polish.** A shared 20 Hz clock drives the throbber, glimmer,
and flashes; thinking text is dim italic; tool dispatches flash in the status
bar; fenced code blocks are click-to-copy and drag-selection survives scrolling;
a CRT power-off animation plays on exit. The status bar shows context as a bare
percentage, with the live progress line pinned below the output.

🔌 **MCP over HTTP.** `.mcp.json` entries with a `"url"` (and optional
`"headers"`) connect over Streamable HTTP; stdio servers work exactly as before.

🆙 **Update checks.** A once-a-day, offline-safe peek at GitHub Releases hints
when a newer plank exists. Disable with `update.check`.

🧹 **Under the hood.** Saving and restoring the KV cache had grown three
implementations of the same file header, two payload layouts, and a legacy
fallback; it is now one type, one format, one owner. Two plank instances can no
longer interleave into the same cache file. Prefill runs in chunks so Ctrl-C
interrupts it promptly. The TUI no longer wedges at 100% CPU on a streaming code
block, providers retry transient HTTP failures with backoff instead of crashing
the run, and a task-list rewrite no longer invalidated the top of the prompt and
re-prefilled the whole conversation every turn. Your existing caches are rebuilt
once on first launch of this version; they are pure caches, so the cost is one
prefill.

The Homebrew formulas are `plank-agent` and `plank-agent-beta`.

## 2.0.2

The v2 line, promoted to stable. plank stays a local agent by default, but it
can now be driven remotely, serve one model to many sessions at once, and talk
to hosted models when you want them — plus a round of TUI polish.

📁 **The status bar tells you where you are.** The footer now leads with the
working directory (home shown as `~`) and, inside a git repo, the current
branch after a powerline glyph, both in the theme green — so a resumed session
in an unexpected folder or branch can't surprise you.

🔁 **Resumed sessions look like live ones.** `/resume`, `/switch`, and `plank
/resume` at startup now replay the conversation through the same renderer a live
turn uses: assistant replies come back as rendered markdown with thinking dimmed
and tool-call banners intact, instead of a flat wall of text.

📜 **Long output scrolls all the way.** Big reports like `/context` now scroll to
the very bottom instead of stopping a few lines short.

✨ **A livelier prefill.** While the prompt is being ingested, the footer now
animates with the same spinner and verb as token decoding, so you can tell it is
working rather than staring at a frozen bar.

🎛️ **Drive plank from anywhere.** A remote-control channel lets another process
or machine attach to a running instance over a loopback WebSocket: mirror its
output, send prompts and commands, and take or hand back control. `plank remote
<url>` is a terminal client, and a small web client is served straight from the
instance. Loopback only by default, token authenticated, with an Origin
allow-list for browsers.

🌐 **Remote and hosted models.** `plank serve` turns one machine into an
inference host over HTTP, and `--remote <url>` points a thin client at it, so
the heavy Metal box does the work while you drive from a laptop. Behind the same
engine boundary, `--provider openai` and `--provider anthropic` route turns to
hosted models, with native tool calls translated back into plank's own tool
syntax so tools behave the same either way. Anthropic prompt caching is on by
default.

🧩 **One model, many sessions.** A shared, reference-counted engine
(`--shared-engine`) loads the weights once and hands out independent sessions
over a single GPU, fairly time-sliced, each with its own context. Admission caps
(`--max-sessions` and a KV-memory budget) keep it from oversubscribing the
machine, and idle sessions can be snapshotted to disk and restored on demand.

⏸️ **Side questions that truly freeze the task.** A mid-generation `/btw` now
genuinely suspends the running reply, answers the aside, and resumes byte for
byte where it left off with zero re-prefill, instead of rewinding and re-running
the step. This is the default now; `--disable-btw-suspend` falls back to the old
boundary queue.

🔖 **Checkpoints and rollback.** `/checkpoint <name>` snapshots the whole
conversation, transcript and live KV together, and `/rollback <name>` returns to
it without leaving the session, so you can explore a risky direction and step
back cleanly. The KV restore means a rollback resumes with no re-prefill, and it
is itself undoable.

💾 **Instant resume.** Sessions now persist the engine KV alongside the
transcript, so `/switch` and `/resume` restore the warm cache instead of
re-reading the whole conversation, and `/strip` reclaims that disk when you do
not need it.

⌨️ **Live command highlighting.** As you type, a valid slash command lights up
green in the prompt and the `!` shell marker turns red, so you can see a command
is recognized before you press Enter.

📁 **`@` to reference a file.** Type `@` in the prompt for a fuzzy typeahead over
your repo's files, directories, and MCP resources. Tab extends the shared
prefix, Enter drills into a directory, paths with spaces get quoted, and your
project's own files sort above vendored submodule paths.

🔍 **The model can find files.** A `glob` tool lets it locate files by pattern
(`**/*_test.rs`) directly, instead of shelling out to `find` — and it reliably
reaches for it. Alongside it, plank now speaks the MCP *resource* protocol, so
the model can read content a server publishes as resources, not just call its
tools.

⚙️ **Settings file.** Preferences you would otherwise retype — model and backend
defaults, `@`-completion tuning, sandbox and `/btw` defaults, the MCP timeout —
live in `~/.plank/settings.json`, overlaid per project. A startup line names
anything in force, so a file that quietly picks the CPU backend can't hide as
"plank got slow."

🐚 **Better `!` shell commands.** Output now streams into the view as the command
runs instead of arriving all at once at the end, and arrow-key history on a `!`
line cycles through past shell commands only. History is also scoped to the
directory you are in, so one project's commands stay out of another's.

✅ **A task list that survives compaction.** The model keeps a structured,
visible task list as working memory: it shows as a counter in the status bar
and a short strip of the active and upcoming tasks, `/tasks` prints the whole
thing, and — the point — it persists through compaction, `/resume`, and
checkpoint rollback, so a long task's plan is not the first thing lost when the
window fills.

🧑‍🔧 **Named agents.** Define specialized subagents as markdown files in
`~/.plank/agents/`, list them with `/agent`, and dispatch one with `/subagent
<name> <task>`. Skills also became something the model can reach for on its own
mid-task, not just a command you type.

🪝 **More hooks.** Hooks now fire on prompt submission, session start and end,
before and after compaction, and on tool failure — several able to inject
context into the turn. A JSON response can halt a turn, warn without blocking,
or run a hook asynchronously, and matchers can key on a command's arguments
(`bash(git *)`).

All still local first, macOS, open source.

## 1.6.0

The whole 1.x line, promoted to stable. plank is a terminal coding agent
written in Rust that runs DeepSeek V4 Flash locally on Apple Silicon through
Metal. No cloud, no API bill, the model lives on your machine. It began as a
functionality by functionality port of a C reference agent, and the last
stable was 0.9.10. Here is what the road to 1.6.0 delivered.

⌨️ **Type while it thinks.** Every turn runs on a worker thread, so the prompt
stays live during generation. Write your next message, or fire off a quick
question, without waiting for the model to finish.

💬 **Side questions that do not derail.** The `/btw` command answers from the
shared conversation context while the main task keeps running. The screen
splits, the answer streams on the right, the work continues on the left, and
none of it touches the real transcript. It stays on screen until you dismiss
it.

🤖 **Delegation.** `/subagent` hands a task to a sidechain run of the same
model with full tool access, and only the final report comes back.

💾 **Remember and resume.** Sessions now get memorable names like
`deadly-einstein` instead of a hash, save automatically on exit, and reopen
with `plank /resume`. Persistent memory carries durable notes across sessions.

🧩 **Extend it.** Skills turn markdown files into slash commands, hooks wrap
your own scripts around tool calls, and an opt in sandbox fences the shell
commands the model runs.

🧠 **Context that lasts.** Layered compaction reclaims the window in escalating
steps and re-attaches your working files across the boundary, so long sessions
keep their footing.

🛟 **Reliability.** A single-instance guard turns the old "cannot load model"
crash into a clear message, and a green rule now separates the scrollback from
the resting prompt.

## 0.x — the foundation

The pre-1.0 line, where plank became a working local agent. It was ported from
the `ds4_agent` C reference functionality by functionality, each C section
becoming an idiomatic Rust module with its own tests, and the wire formats kept
byte for byte identical to what the model was trained on.

🧠 **Real local inference.** DeepSeek V4 Flash runs on Apple Silicon through
Metal, wired in over FFI and kept behind an `Engine` trait, with an echo stub
so the whole app still builds and runs without a model.

🖥️ **A full-screen terminal UI.** A Ratatui interface (with a plain line REPL
and a headless mode) renders assistant replies as markdown with syntax
highlighted code, mouse-wheel scrollback, and a live status bar showing tokens,
throughput, and context usage.

⬇️ **One-keypress model download.** With no model on disk, plank offers to fetch
the quantized GGUF from Hugging Face. The download is resumable, guarded by a
RAM check, and keeps you company with a live progress gauge.

⚡ **Fast startup.** The system prompt is prefilled once and snapshotted to a
fingerprinted checkpoint, so a fresh launch restores the warm KV cache instead
of recomputing it, and each turn reuses the cached prefix.

🧰 **A real tool suite.** File read and edit (with `[upto]` anchored
replacements), synchronous and background shell commands, and web search, all
framed exactly like the C reference, plus a strict DSML tool-call parser with
on-screen banners.

🔌 **MCP support.** Stdio MCP servers listed in `.mcp.json` are launched at
startup and their tools exposed to the model, with a `primaryTools` list to
keep the system prompt small.

💾 **Sessions and context management.** Conversations save, list, and switch;
context compaction reclaims the window with a durable summary plus a verbatim
tail; and upgrade-time cache maintenance clears exactly what a new version can
no longer trust.

🍺 **A Homebrew hotfix (0.9.10).** The last release of the line fixed installs
from the tap that could not load any model, because the Metal kernel sources
were resolved from a compile time CI path that did not exist on your machine.
The kernels now ship inside the bottles (`share/plank/metal`) and are resolved
at runtime, and the engine-open error says plainly when they are missing
instead of blaming the model file.

All local, macOS, open source.

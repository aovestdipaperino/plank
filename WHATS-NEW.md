# What's new in plank

A short, human-readable highlight reel per release. For the full change list
see the GitHub releases and commit history.

## Beta (2.1, unreleased)

The 2.1 beta (`brew install plank-agent-beta`) is a round of "see what the agent is
doing" polish, plus the first delegation tool.

🧑‍🔧 **Delegate to a sub-agent.** The model can hand a bounded task to a fresh,
scoped sub-agent with the new `agent` tool and get back only its conclusion,
instead of filling the main transcript with the research. It runs as a sidechain
off your conversation and rolls back out afterward, so the delegation is cheap.
An optional `name` picks one of your `~/.plank/agents` personas.

📝 **Plan mode.** `EnterPlanMode` puts the model in a read-only phase — it can
research with read/list/glob/search but `write`, `edit`, and `bash` are refused
— until it proposes a plan with `ExitPlanMode` and you approve it. A cheap
course-correction before any edits land.

📊 **File changes show as a git diff.** When the model edits a file (or
overwrites one), plank now renders a git-style change card: an `Update(path)`
header, an added/removed summary, and `@@` hunks with red/green line
highlighting. A brand-new file instead streams its contents dimmed as it is
written, so you always see what is going onto disk.

⏳ **Progress you can actually see.** The system-prompt cache, if it needs
rebuilding at launch, now does so behind a simple progress bar before the full
UI appears. During a turn the spinner, verb, and token stats sit on their own
line right below the output — so even while thinking is hidden, you can tell the
turn is alive. The status bar itself is quieter: it shows the working directory,
git branch, and context as a percentage, framed above and below by a rule.

🔎 **Run reports without waiting.** `/context`, `/usage`, `/mcp`, and `/help` now
work while the model is still generating, answered from a turn-start snapshot —
no need to interrupt to check where things stand.

⌨️ **The prompt wraps.** Long input wraps onto the next line instead of scrolling
sideways, so you can see the whole thing as you type.

⚙️ **`showThinking` setting.** Set `ui.showThinking: false` to hide the model's
dimmed thinking from the display (it still produces it); the on-screen progress
line keeps the turn visible either way.

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

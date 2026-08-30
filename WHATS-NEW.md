# What's new in plank

A short, human-readable highlight reel. For the full change list see the GitHub
releases and commit history.

plank ships on two channels, and the patch number is the channel: `X.Y.0` is
stable, anything above it is a beta on the way to the next stable. This page is
grouped the same way — what is in the beta now, then the stable releases behind
it.

## In the betas

Riding the beta channel today, on top of the newest stable release. Install with
`brew install aovestdipaperino/tap/plank-agent-beta`.

### 3.4.1

Opens the new beta channel on the same code as 3.4.0.

📦 **`/install-claude-plugin` fetches a Claude Code plugin and installs it.** Plank
already knew how to *load* a Claude Code plugin; now it can also grab one from
a GitHub repo, a `/tree/...` link copied straight out of the browser, a
marketplace, a `.tar.gz`, or a local directory. It sorts out the format
differences along the way — `${CLAUDE_PLUGIN_ROOT}` becomes a real path, and
nested hook config gets unwrapped — so the plugin's hooks and MCP servers just
work. Tried end to end with `obra/superpowers`: one command, all 14 skills
showed up on the next start.

## Stable releases

### 3.4.0

Eleven features built on the DeepSeek-harness plan, and the theme is
self-knowledge: plank gets better at telling you what it is doing, remembering
what it has done, and staying inside its own guard rails while doing more.

🔍 **`plank --dump-config` tells you where every setting came from.** Settings
arrive from five layers — defaults, plugins, `~/.plank`, `./.plank`, CLI flags —
and "I set that and nothing happened" used to be a code-reading exercise. Now
each key prints with the layer that won and the layers it beat.

```
engine.ctx = 8192        <- CLI flag
                              (shadowed: ~/.plank/settings.json)
```

`/config --resolved` does the same inside a session, and additionally shows
which plugin won each contested skill or agent name.

🔁 **Loop guards.** The classic local-model failure is a loop: a stale edit
anchor sends the model into read/edit/read/edit and it does not notice. On the
third identical call it now sees *"you have called this tool with these
arguments 3 times; the result has not changed"*. Nothing is blocked — a
legitimate third read exists — and polling an async job is exempt, because a
poll looks exactly like a stuck loop. Separately, `tools.callTimeoutSec` gives
tool calls a budget and tells the model when one blew it, so a hung test suite
stops being invisible.

📄 **Big tool output is no longer a dead end.** Ask about a 5 MB build log and
`read` used to truncate with no way forward. Now the full payload goes to
`~/.plank/spill/`, the model gets a bounded preview plus a `continue_offset` it
can page with `more`, and `/export` still sees everything. The same applies to
oversized MCP results, which previously had no cap at all — one chatty server
could eat a whole context in a single call.

🧹 **Context gets reclaimed without a round-trip.** After a dozen large reads
their bodies are dead weight. Microcompact clears the old ones at end of turn,
keeping the newest three, anything small, and everything belonging to the task
you are on. It only runs when it would reclaim at least 4 KiB, because
rewriting the transcript invalidates the KV prefix and an eager pass costs more
than it saves.

🔎 **`/search` across your own history.** You remember fixing a Metal crash a
few weeks ago but not how. `/search metal` finds the session, shows the
snippet, and offers `/resume`. Crucially it is compaction-proof: long sessions
are both the ones worth searching and the ones that get compacted, so the index
keeps conversation the transcript has dropped.

🎯 **Goals survive the session.** `/goal --max 5 make the failing test pass`
runs an adjudicated loop you can walk away from. The objective is durable state
— it survives `/save`, `/resume` and `/compact`, so tomorrow you can still see
what it was pursuing and how far it got.

⭐ **`/rate` records what worked, where the model cannot see it.** A rating
lives in a sidecar, never the transcript, context or KV — if the model could
read your ratings it would optimise for them and the signal would be worthless.
`/insights` turns a week of them into satisfaction over time and the notes on
the worst turns.

🧰 **Three new tools, on by default.** `recall` lets the model search that same
session history instead of guessing or interrupting you. `fanout` runs several
independent subtasks and joins their reports in a fixed order, with optional
throwaway-worktree isolation for subtasks that edit. `run_code` batches a few
pre-decided operations into one turn instead of one round-trip each — and every
operation goes through the normal tool dispatch, so the sandbox, the `~/.plank`
grant and the `PreToolUse` hooks apply exactly as they would to a bare call.
Each can be switched off individually with `tools.recall`, `tools.fanout` or
`tools.runCode`.

🪵 **And a rule the rest of it rests on.** A test now asserts structurally that
everything reaching the model is reconstructible from the session log — either
a transcript entry or the separately-fingerprinted system prompt. Without it,
`/repro` and `/resume` are guesses.

### 3.2.0

The 3.1 beta line, promoted: the status bar learns to show what the model is
doing, the exit message learns to say where the session went, and — carried up
from the 3.0 betas — plank learns to read the screenshots you paste at it.

⏱️ **The exit summary reports where the time went, per model.** It used to print
a peak prefill rate and a peak generation rate — one lucky pass each, which
tells you nothing about the session you just had. Now every model that ran gets
a line: how long it spent prefilling and how long generating, each with the
session's average rate, and how long it spent running tools.

```
avg deepseek-v4-flash  prefill 12.3s (1420.5 tok/s)  ·  generation 45.2s (38.1 tok/s)  ·  tools 8.4s
```

That last figure is the one nobody was measuring. A turn that feels slow is
often not the model at all.

🧠 **The think segment shows the router working.** Two braille cells beside the
reasoning level re-roll on every decoded token, standing in for the mixture-of-
experts routing. Being straight about it: the real selection never leaves the
GPU on the Metal path, so plank derives the pattern from the token id. It is
honest about sparsity, about routing changing every token, and about the same
token lighting the same dots — and it does not know which experts. The reasoning
level itself is now colored by temperature, red for `max` down to grey for
`off`, so three columns are readable without reading the word.

📉 **The decode rate in the footer stopped lying on long prompts.** It was timed
from the start of the generation call, so it divided tokens by decode time plus
prefill plus the wait for the first token, and opened far below the real rate.
It is now measured from the first token out.

⚡ **Greedy chain decode on Metal.** At temperature 0 a run of argmax tokens is
decoded with the next token id kept on-device, dropping the per-token GPU sync
and logits readback. Output is bit-identical. It is off on M5, where it measures
slightly slower than the plain path — `PLANK_GREEDY_CHAIN=1` forces it on if you
want to re-measure on your own machine.

🔢 **The `--dspark` footer reads `1.5t/step`, not `1.5x`.** It was always tokens
committed per speculative step, and that is not a wall-clock speedup: on Metal
it sits above 1.0 on runs that decode *slower* than plain decode. Calling it a
multiplier was the bug.

And carried up from the 3.0 betas, which never got a stable entry of their own:
pasted screenshots stop being filenames.

👁️ **plank can finally read your screenshots.** Image pasting is on by default
now, and paired with the new [`ocr-mcp`](https://github.com/aovestdipaperino/ocr-mcp)
server the model can actually read what you paste. Screenshot a stack trace,
paste it, ask what it says. The transcription happens on your machine with a
0.9B OCR model: no cloud, no API key, no image leaving the laptop. Install it
with `brew install llama.cpp && cargo install ocr-mcp`, register it in
`.mcp.json`, and the guide walks through
[the weights](user-guide/09-extending.md#reading-images-with-ocr-mcp).

The feature had been compiled out behind `--features images` for a reason that
no longer holds. A pasted image reached the model as a path it had no way to
open, so the whole thing was a tease. Now there is a tool that opens it.

🖼️ **Pasted images are cached exactly as you pasted them.** plank used to
downsample every PNG to 2000px, a rule inherited from an image-upload API limit
that plank does not have and never did: the ds4 engine is text-only and never
uploads pixels anywhere. All the resampling did was throw away the pixel density
and the DPI metadata that an OCR tool then needs. The bytes now land in
`~/.plank/image-cache/` byte-for-byte identical to the source.

One consequence worth knowing: the cache is bounded by file count, not bytes, so
full-resolution captures make it larger than it used to be.

### 2.8.0

The 2.7 beta line, promoted: `/btw` learning to answer without stopping the
task, PDF reading, and an `/insights` report that stops overreaching — plus
shell escapes and compaction both learning a distinction they were missing.

🐚 **`!` and `!!` now differ by who sees the output.** A `!` command used to be
yours alone: it ran, you read the output, and the model never knew. That is now
`!!`. Plain `!` runs the command exactly the same way and then records it and its
result in the conversation, so `!cargo test` followed by "fix the failure" works
without you pasting anything. Neither form starts a turn, so `!` is not a way to
ask for something — it is a way to have already shown it.

🗜️ **`/compact` takes instructions.** `/compact keep the failing test cases
verbatim` steers that one summary. What you say is added to what plank already
asks for rather than replacing it, so the summary keeps its structure and gains
your emphasis. Automatic compaction asks exactly what it always did.

⏳ **Compaction says how far along it is.** It used to print one line and then go
quiet for as long as it took, which on a large conversation reads as a hang. The
throbber line now carries a flashing `compacting` with a bar and a percentage,
and the window title reads `🗑️ compacting...` until the pass ends.

🪝 **Compaction hooks fired for almost nobody.** `PreCompact` and `PostCompact`
ran only in the plain REPL, so if you use plank in a terminal — which is to say,
in the TUI — a compaction hook you had configured never ran at all. Both now
fire on both front-ends.

🛟 **A failed compaction no longer costs you the conversation.** If the model
came back with no usable summary, plank rebuilt the transcript anyway and put
that emptiness where the summary should have been. A pass with nothing to show
for it is now a failure: your conversation is left exactly as it was.

💬 **`/btw` answers beside the task instead of freezing it.** Asking a side
question mid-generation used to stop the main task dead until the answer was
done. The aside now runs on a fork of the session, interleaved with the main
generation, so the reply keeps streaming on the left while the answer fills the
panel on the right. The aside takes the larger share of the thread while it
runs, since it is the one you are waiting on.

One Metal command queue means this is time-slicing, not parallelism — nothing
finishes sooner overall. What changes is that the main task no longer stops.
Only one aside runs at a time, and questions are no longer queued behind one:
with the answer already on screen, holding the next one back bought nothing.

🧠 **The second `/btw` of a turn stopped rebuilding the whole conversation.** It
re-prefilled every token — around 14,600 of them on a moderate transcript, a
minute of dead air that read as a hang. A suspended turn generates twice into
one reply, but the cache index recorded the two halves separately while the
conversation holds them as one, so the next question no longer lined up with
what was cached and the engine started over. One reply is now one entry, and
the second aside reuses 99.9% of what it already had.

📄 **plank reads PDFs.** `read` on a `.pdf` gives you the document as Markdown
and pages through it like any other file — the same bounded chunks, the same
line numbers, `more` continuing where the last chunk stopped. Underneath it is
spatial text extraction over PDFium with OCR for pages that have no text layer,
so a scanned manual reads as well as a born-digital one, and the converted
Markdown is cached by content hash: paging through a 400-page document parses it
once. `visit_page` does the same for a PDF on the web, recognised by its URL or
by the bytes that come back, instead of handing you mojibake.

It is not a new tool and there is nothing to turn on. `read` simply stopped
finding documents unreadable — which took two goes to get right. The first
attempt worked perfectly and never ran, because nothing told the model PDFs were
readable, so it went looking for `pdftotext` instead; the prompt now says so in
one sentence. The second went looking, found the file, and refused it: the
16 MB cap on how much text a read may put in context was being applied to bytes
that never enter the context, so the 60 MB manual you actually wanted came back
as *too large to read*. Both fixed. Office formats are still a non-goal — the
converter reaches them by shelling out to LibreOffice, which plank is not going
to require behind your back.

🔍 **`/insights` tells you what it is doing, and stops.** The title reads
`introspecting...` while it runs, Esc and Ctrl-C actually take effect now
(the command runs inside the dispatch, so no key was ever being read), and a
stopped run leaves the report you already had whole rather than half-written.
It has also stopped drawing conclusions from numbers that are not there: a
history recorded before per-message timestamps sums to a near-zero hour count,
and the report used to read that back to you as "your total time spent is very
limited". Timing is now reported only when at least half your sessions carry it.

🟩 **Status lines read better.** `Opening page ...` and friends are the theme
green instead of dim pink, with any URL lifted to white so the target stands
out from the prose.

### 2.6.0

The 2.5 beta line, promoted: five betas' worth of the cache learning to keep
what it already knows, plus session branching, a handful of commands, and a lot
of progress-reporting that finally tells the truth.

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

### 2.0.2

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

### 1.6.0

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

### 0.x — the foundation

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

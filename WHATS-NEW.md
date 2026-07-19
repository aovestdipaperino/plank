# What's new in plank

A short, human-readable highlight reel per release. For the full change list
see the GitHub releases and commit history.

## 1.6.0

The whole 1.x line, promoted to stable. plank is a terminal coding agent
written in Rust that runs DeepSeek V4 Flash locally on Apple Silicon through
Metal. No cloud, no API bill, the model lives on your machine. It began as a
functionality by functionality port of a C reference agent, and the last
stable was 0.9.10. Here is what the road to 1.6.0 delivered.

**Type while it thinks.** Every turn runs on a worker thread, so the prompt
stays live during generation. Write your next message, or fire off a quick
question, without waiting for the model to finish.

**Side questions that do not derail.** The `/btw` command answers from the
shared conversation context while the main task keeps running. The screen
splits, the answer streams on the right, the work continues on the left, and
none of it touches the real transcript. It stays on screen until you dismiss
it.

**Delegation.** `/subagent` hands a task to a sidechain run of the same model
with full tool access, and only the final report comes back.

**Remember and resume.** Sessions now get memorable names like
`deadly-einstein` instead of a hash, save automatically on exit, and reopen
with `plank /resume`. Persistent memory carries durable notes across sessions.

**Extend it.** Skills turn markdown files into slash commands, hooks wrap your
own scripts around tool calls, and an opt in sandbox fences the shell commands
the model runs.

**Context that lasts.** Layered compaction reclaims the window in escalating
steps and re-attaches your working files across the boundary, so long sessions
keep their footing.

**Reliability.** A single-instance guard turns the old "cannot load model"
crash into a clear message, and a green rule now separates the scrollback from
the resting prompt.

All local, macOS, open source.

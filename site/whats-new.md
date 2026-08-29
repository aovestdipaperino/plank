# What's new

This site went up in late July 2026, around v2.5. Plank has not held still since.
Below are the changes worth knowing about if you have been away, newest first. The
[full changelog](https://github.com/aovestdipaperino/plank/blob/main/CHANGELOG.md)
has every last fix; this page has the ones you will actually notice.

## Just landed

**plank tells you where every setting came from.** Settings arrive from five
layers — built-in defaults, plugins, `~/.plank`, the project's `./.plank`, and
CLI flags — and "I set that and nothing happened" used to mean reading code.
`plank --dump-config` now prints each effective key with the layer that won and
the layers it beat, and `/config --resolved` does the same inside a session,
including which plugin won a contested skill or agent name. See
[Configuration](/guide/08-configuration.html).

**Loop guards for the failure local models actually have.** A stale edit anchor
sends a model into read/edit/read/edit, and small models often do not notice. On
the third identical call the result now carries *"you have called this tool with
these arguments 3 times; the result has not changed"*. Nothing is blocked — a
legitimate third read exists, and blocking on a guess would be worse — and
polling an async job is exempt, because a poll looks exactly like a stuck loop.
Set `tools.callTimeoutSec` and the model is also told when a call overran its
budget, so a hung test suite stops being invisible.

**Large tool output is no longer a dead end.** Ask about a 5 MB build log and
`read` used to truncate with nowhere to go. The full payload now goes to
`~/.plank/spill/`, the model gets a bounded preview plus a `continue_offset` it
can page through with `more`, and `/export` still sees everything the tool
returned. The same bound applies to oversized MCP results, which previously had
no cap at all — a single chatty server could fill your context in one call.

**Context reclaims itself between turns.** After a dozen large reads those
bodies are dead weight. plank now clears the old ones at end of turn — keeping
the newest three, anything small, and everything belonging to the task you are
on — with no model round-trip and no summarising. It holds off unless it would
reclaim at least 4 KiB, because rewriting the transcript invalidates the KV
prefix and an eager pass costs more than it saves.

**`/search` across your own sessions.** You remember fixing a Metal crash a few
weeks ago but not how. `/search metal` finds the session, shows the matching
snippet and offers `/resume`; `--all` widens beyond the current project. It is
deliberately compaction-proof — long sessions are both the ones worth searching
and the ones that get compacted, so the index keeps conversation the transcript
has since dropped.

**Goals that outlive the session.** `/goal --max 5 make the failing test pass`
runs a loop that adjudicates its own progress and stops on a verdict or the cap.
The objective is durable state: it survives `/save`, `/resume` and `/compact`,
so you can come back tomorrow and still see what it was pursuing and how far it
got.

**`/rate` records what worked, somewhere the model cannot see.** A rating goes
to a sidecar, never the transcript, context or KV — if the model could read your
ratings it would start optimising for them and the signal would be worthless.
`/insights` turns a week of them into satisfaction over time plus the notes on
the turns that went wrong.

**Three new tools, on by default.** `recall` gives the model that same session
history, so it can look up a past decision instead of guessing or interrupting
you. `fanout` runs several independent subtasks and joins their reports in a
fixed order, with optional throwaway-worktree isolation when subtasks edit
files. `run_code` batches a few pre-decided operations — `read`, `glob`, `edit`,
`bash` — into one turn instead of a round-trip each; every operation is routed
through the normal tool dispatch, so the sandbox, the `~/.plank` write grant and
your `PreToolUse` hooks apply exactly as they would to a bare call. Switch any
of them off with `tools.recall`, `tools.fanout` or `tools.runCode`.

**plank signs the commits it writes.** When the model creates a git commit, the
message now ends with a blank line and the single line `--with help from plank`,
so a `git log` months from now still says which commits came out of a session
rather than out of your fingers. It is a system-prompt instruction, not a hook,
which means the model can be told to leave it off for one commit and your
repository conventions still come first. If you would rather it never appear,
set `"git": { "signCommits": false }` in `settings.json` or run
`/config git.signCommits false`, and the instruction is gone from the prompt
entirely. See [Configuration](/guide/08-configuration.html#git).

**DSpark speculative decoding is on by default.** The auxiliary draft checkpoint
used to need `--dspark`; now it is the default and `--dspark-off` is how you get
target-only decode. Speculation only engages at temperature 0, so a bare `plank`
samples argmax: pass `--temp` if you want sampling back, and `--dspark-off`
leaves the old 0.6 default in force. The support model is still fetched on
demand the first time it is needed.

**The exit summary says where the session went.** It used to end with a peak
prefill rate and a peak generation rate — one lucky pass each, which tells you
nothing about the run you just had. Now every model that ran gets a line: time
spent prefilling and time spent generating, each with the session's average
rate, and time spent running tools.

```
avg deepseek-v4-flash  prefill 12.3s (1420.5 tok/s)  ·  generation 45.2s (38.1 tok/s)  ·  tools 8.4s
```

That last number is the one nobody was measuring, and a turn that felt slow is
often not the model at all.

**The think segment shows the router working.** Two braille cells beside the
reasoning level re-roll on every decoded token, standing in for the
mixture-of-experts routing. Being straight about it: the real selection never
leaves the GPU on the Metal path, so the pattern is derived from the token id.
It is honest about sparsity, about routing changing every token, and about the
same token lighting the same dots — and it does not know which experts. The
reasoning level itself is now colored by how hard the model is thinking, red for
`max` down to grey for `off`.

**The decode rate stopped lying on long prompts.** It was timed from the start of
the generation call, so it divided tokens by decode time plus prefill plus the
wait for the first token, and opened far below the real rate. It is measured
from the first token out now.

**Greedy chain decode on Metal.** At temperature 0 a run of argmax tokens decodes
with the next token id kept on-device, dropping the per-token GPU sync and logits
readback. Output is bit-identical. Off on M5, where it measures slightly slower
than the plain path.

**The `--dspark` footer reads `1.5t/step`, not `1.5x`.** It was always tokens
committed per speculative step, and that is not a wall-clock speedup: on Metal it
sits above 1.0 on runs that decode *slower* than plain decode. Calling it a
multiplier was the bug.

**plank can read your screenshots.** Image pasting is on by default now, and paired
with the [ocr-mcp](https://github.com/aovestdipaperino/ocr-mcp) server the model can
act on what you paste: it calls `transcribe_image` on the cached path and gets the
text back. Screenshot a stack trace, paste it, ask what it says. Transcription runs
on your own machine against a 0.9B OCR model, so no image leaves the laptop and there
is no API key. Install it with `brew install llama.cpp && cargo install ocr-mcp`,
register it in `.mcp.json`, and see
[Extending plank](/guide/09-extending.html) for the weights. Pasted images are also
cached byte-for-byte now: the old downsampling was inherited from an image-upload
limit plank never had, and it only threw away the pixel density an OCR tool needs.

**Plugins load.** A plugin is one directory bundling skills, agents, templates, hooks,
an `.mcp.json` and a `settings.json`, contributed to a session as a unit. plank picks
them up from `~/.plank/plugins/dev/`, from `./.plank/plugins/`, or from a repeatable
`--plugin-dir` for the session you are in, and reads both its own spelling and Claude
Code's. A plugin contribution is always addressable as `<plugin>:<name>` and keeps the
bare name only when nothing else claims it, so your own skills and agents never lose
theirs. Plugin settings sit below your own files, and `/plugins` shows what loaded,
what each one contributes and every warning. There is no installer and no marketplace
yet — you place the directory yourself. See
[Extending plank](/guide/09-extending.html).

**v3.3.0 is out**, and the beta channel has opened on 3.3.1. The patch number is
still the channel: `.0` is stable, anything above it is beta.

**Your session has a name from the first frame.** The memorable
`adjective-celebrity` name used to be minted when a session was first saved, so
until you quit there was nothing to call the conversation you were in. It is minted
at the start now, and floats at the right end of the rule above the prompt.
`/rename <name>` changes what later saves use without touching what is already on
disk, so the earlier file stays resumable under its old name. Resuming a session no
longer replays plank's own scaffolding at you either — the agent instructions,
memory, git status and date that open a session are sent to the model, not typed by
you, and they stay out of the transcript you read. See
[Sessions](/guide/06-sessions.html).

**`/kvcache` shows the cache as the tree it really is.** Every KV snapshot on disk
now carries a small metadata file recording what it is, which snapshot it was built
on, the model and reasoning level behind it, its size, how many times it has been
reused, and when. `/kvcache` draws that as a tree you can walk with the arrow keys:
`p` pins an entry so nothing will ever sweep it, `d` deletes one, `g` sweeps now.

**The cache expires on age and is capped on size.** Snapshots used to be kept only
for the *current* system prompt and project, and every sibling was deleted, so
switching model or reasoning level and back paid a full system-prompt re-prefill each
way. Now they expire on time since last use (14 days for a conversation, 30 for a
shared checkpoint) and a 20 GB ceiling evicts the least recently used beyond that.
Several system prompts coexist for as long as you are using them. See
[Sessions](/guide/06-sessions.html#the-kv-cache) and
[Configuration](/guide/08-configuration.html#kvcache).

**`/open [path]`** edits an existing file in plank's own editor: `Ctrl-S` saves, `Esc`
discards. Bare `/open` reopens the last file a tool call touched, which is usually the
one you wanted to look at.

## One session, several engines

**Cross-engine sub-agents.** A subagent definition can name the engine it runs on,
independently of the main agent: a `provider:` and `model:` in its frontmatter, with
an optional base URL and the *name* of the environment variable holding the key, so
the file stays committable. `provider: local` names the local engine specifically, so
a hosted main agent can delegate to the model on your own Mac. `/agent` shows each
definition's engine and the variable to set when it is missing.

**Git worktrees.** A session can move itself into an isolated worktree and back out
again, so an agent can work on a copy of the repo without touching yours. Subagents
can each get their own with `isolation: worktree`.

**DSpark speculative decoding, behind `--dspark`.** DeepSeek's auxiliary draft
checkpoint (~5.6 GB on top of the model) proposes tokens the main model verifies in
batches. It downloads and resumes the same way the model does.

**A live agent roster under the status bar.** Instead of one sub-agent's output in a
hidden pane, every run gets a row: what it is doing, how long it has been at it, and
what it has spent. `←` on an empty prompt steps into it, `Enter` opens an agent's
output in full, `Esc` comes back. A fan-out shows every agent at once, each with its
own buffer, and what comes back into your transcript is the agent's answer with its
thinking stripped out. See [The agent roster](/guide/03-the-interface.html#the-agent-roster).

## Drive it from somewhere else

**`/remote-control`, or `/rc`,** starts and stops a remote-control server from inside
a running session, and `/grant` approves a client that asks for control. The old
`--control*` flags are gone.

**The bundled web client is a real front-end now.** It wears plank's own dark theme,
streams the turn as it happens, and tells you unmistakably when the connection drops.
Attached clients get the end-of-turn notification too, so you can walk away from the
laptop and still be told when it is your move.

## Reasoning you can dial

**`/think off | low | medium | max`,** with `--think-max` and friends on the command
line. `low` is experimental and cheap; `max` prepends a reasoning-effort preamble and
is the one worth reaching for on a hard problem. The status footer shows which level
is in force as a `🧠 med` segment, because it changes both cost and answers and used
to be invisible.

## The terminal got more useful

**`!` and `!!`.** A bare `!command` runs a shell command *and hands the result to the
model*, which is what you almost always wanted; `!!command` keeps the output to
yourself.

**`/btw <question>` answers beside the running task** instead of freezing it. The
aside runs on a fork of the session in a split panel, so the real conversation is
never touched and neither side waits for the other.

**PDFs are readable.** `read` on a `.pdf` converts the document to Markdown, with OCR
for a scanned one.

**`/insights`** builds a personal usage report over every session you have ever saved
and writes it to an HTML file: where the time went, which tools you lean on, how the
model actually behaves for you.

**A prompt editor on `Ctrl-G`,** built in rather than shelling out to `$EDITOR`, for
when the thing you are about to ask has outgrown one line.

**`/compact [instructions]`** compacts the conversation now rather than waiting for
the automatic pass, and an argument steers what that one summary keeps. Compaction
shows its progress in the status bar, and `Ctrl-C` interrupts it.

**A screensaver, and an arcade.** After a few idle minutes plank goes to a starfield,
matrix rain, or a couple of minions, chosen at random and configurable. There are also
games that run over a live turn, which is either a feature or a confession
([chapter 11](/guide/11-arcade.html)).

## Sessions became a tree

**`/tree`, `/fork`, `/clone`.** A session is a tree of messages rather than a line.
`/tree` draws it and numbers the fork points; `/fork <n>` rewinds to just before one
of your prompts and keeps everything after it as a sibling branch, so you can try a
different approach without losing the first; `/clone` freezes the current branch and
continues on a copy. All of it is shaped so the cache is reused rather than rebuilt.

**`/export [md|html]`** renders the transcript to a shareable file. The HTML is
standalone.

**Prompt templates.** Markdown files in `~/.plank/templates` become commands, with
`{{variable}}` substitution.

**MCP over Streamable HTTP.** An `.mcp.json` entry with a `"url"` speaks to a remote
MCP server, alongside the stdio servers that were already supported.

## Things that were quietly broken

A resumed session used to re-prefill its whole conversation. `/new` and `/clear` used
to rebuild the system-prompt cache from scratch. A dropped network could hang a turn
forever with `Ctrl-C` doing nothing. A compaction that produced no usable summary
could destroy the transcript. Several shapes of tool call the model emits were parsed
wrongly and died. All fixed, and each one has an entry in the changelog explaining
what actually went wrong, if you like that sort of thing.

---

New here instead? Start with the [user guide](/guide/), or
[install it](/) and get going.

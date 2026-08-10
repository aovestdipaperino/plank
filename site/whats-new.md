# What's new

This site went up in late July 2026, around v2.5. Plank has not held still since.
Below are the changes worth knowing about if you have been away, newest first. The
[full changelog](https://github.com/aovestdipaperino/plank/blob/main/CHANGELOG.md)
has every last fix; this page has the ones you will actually notice.

## Just landed

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

**A live agent roster on `Ctrl-O`.** Instead of one sub-agent's output, the pane is
now a roster of every run in the session with what each is doing, how long it has
been at it, and what it has spent. The status bar became two rows to make room for
it, and the working directory and git branch moved up to the first.

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

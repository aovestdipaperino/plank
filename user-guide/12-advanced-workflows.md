[← The arcade](11-arcade.md) · [Index](README.md) · Next: [Troubleshooting →](13-troubleshooting.md)

# 12. Advanced workflows

The basics get you a good conversation. This page is about the things that make a *long* one hold together: exploring more than one approach, keeping context clean over hours, and not losing work when an approach turns out to be wrong.

## Branching: exploring more than one answer

The core insight is that a conversation is a tree, not a line. You asked for something, the model did it, and now you want to know what a different approach would have looked like — without throwing away the one you have.

```
/tree            # where am I, and what can I fork from?
/fork 3          # rewind to just before my 3rd prompt, go a different way
/clone           # freeze this branch, continue on a copy
```

### `/fork` — try a different prompt

`/fork n` rewinds the live transcript to just before your `n`-th real prompt. That prompt and everything the model did after it stay in the tree as a sibling branch, still listed in `/tree` and still reachable.

Use it when **the prompt was wrong**. You asked for a caching layer, got one, and realised the real problem was the query. Fork back to before you asked, ask the better question, and the caching detour is preserved rather than argued with.

Fork points are your *real* prompts — tool results do not count — so `/fork 2` means "the second thing I actually asked."

### `/clone` — try a different continuation

`/clone` duplicates the current branch and makes the copy live, freezing the original exactly where it stands.

Use it when **the conversation is good and the next step is risky**. You have twenty turns of hard-won context about a subsystem and you are about to ask for a refactor that might go badly. Clone first: if it goes badly, `/tree` still shows the branch where it had not happened yet.

### Fork, clone, or checkpoint?

All three let you go back. They differ in what "back" means:

| | Rewinds to | Keeps the discarded work | Survives compaction | Persisted |
|---|---|---|---|---|
| `/fork n` | before one of your prompts | yes, as a sibling branch | yes | yes, in the session file |
| `/clone` | nowhere — copies forward | yes, the frozen original | yes | yes |
| `/checkpoint` + `/rollback` | a point you marked | yes, as `pre-rollback` | yes, exactly | no, in-memory only |

The practical split: **checkpoints are for undo, branches are for comparison.** A checkpoint is the thing you take before a risky operation and forget about if it went fine. A branch is the thing you keep because you genuinely want both versions available.

Checkpoints are also dropped by `/new`, `/switch`, and `/resume`; branches are written into the session file and come back with it.

### Why branching is cheap

Both operations are shaped so the model does not have to re-read the conversation. A fork leaves the transcript a strict prefix of what it was; a clone leaves it byte-identical. Nothing about the engine's cached state is copied or reinterpreted — the next turn reconciles the prompt against what is already loaded and only pays for what genuinely differs.

Practically: forking back ten turns is not a ten-turn penalty. Cloning is nearly free.

### A worked pattern

```
… fifteen turns establishing how the scheduler works …
/checkpoint understood        (cheap insurance)
/clone                        (freeze the understanding)

  "rewrite the scheduler to be lock-free"
  … it goes badly …

/tree                         (the frozen branch is right there)
/fork 16                      (back to just after the understanding)
  "what would it take to make the existing scheduler lock-free?
   don't write anything yet"
```

You now have two branches from the same fifteen turns of context: one that tried it, one that scoped it first. Neither cost you the understanding.

## Keeping a long session sharp

### Delegate the detours

The single biggest cause of a session going vague is a research detour eating the context. Hand those to a subagent:

```
/subagent find every place we assume the config is loaded before the logger,
          and report the ones that would break if that changed
```

The subagent does the reading in its own scoped context and only its **final report** enters your transcript. Twenty file reads become one paragraph. The model does the same thing on its own with the `agent` tool when it recognises a bounded chore.

### Ask sideways with `/btw`

```
/btw what's the difference between Arc<Mutex<T>> and RwLock again?
```

The running generation genuinely suspends, answers in a split panel, and resumes byte-for-byte — and **nothing is written to the conversation**. Use it for the question you would otherwise open a browser for. The main task does not learn that you asked.

`Esc` at an idle prompt dismisses a panel left open from an earlier turn.

### Compact deliberately

Do not wait for the automatic pass to fire in the middle of something delicate. Before handing the model a large new job:

```
/context      # see what is actually taking up room
/compact      # rebuild: summary + recent tail + recently read files
```

When you already know what the next stretch of work needs, say so and the summary will keep it:

```
/compact keep the failing test cases verbatim
```

A `/checkpoint` taken beforehand makes even that reversible — a checkpoint stores the whole transcript, so `/rollback` reconstructs the pre-compaction conversation exactly.

### Give it a plan that survives

Ask the model to keep a task list. The `task` tool's list survives compaction by design, so a long job does not lose its plan when the transcript is summarized. `/tasks` shows it and the status bar counts it.

### Write down what it cannot re-derive

```
/remember the integration tests need DOCKER_HOST set; CI sets it, local shells don't
```

Anything the model would otherwise rediscover — or worse, get wrong the same way twice — belongs in memory or `AGENTS.md`. Anything it can read out of the repository does not; that is context you pay for twice.

## Working carefully

### Plan before editing

For anything risky or ambiguous, ask for a plan first:

> plan this out before you touch anything

The model enters read-only plan mode, where every mutating tool is refused until it proposes a plan and you approve it. Rejecting the plan keeps the gate on, so you can iterate on the approach with no risk of an edit landing mid-discussion.

### Phrasings that hold the model back

You do not have to name plan mode to get its effect. These work, and they are worth keeping in your fingers:

> **don't make changes, just investigate**
>
> **show me, without writing anything, what a reasonable plan would look like**
>
> **give it a fast review and save the findings in `docs/CODE-REVIEW.md`**

The last one is the general shape worth internalising: **say where the output goes.** A review that lands in a file survives the session; a review that lands in the transcript is gone at the next compaction.

The same applies to delegation and to style. "Use a subagent to …" is a valid instruction, not just something the model decides on its own. And handing over a style file — "write this using the style in `local/medium-post-style.md`" — beats describing the style in the prompt every time.

### Tighten the sandbox

The bash sandbox already limits writes to the working directory and temp. If a project needs more, name it explicitly rather than switching the sandbox off:

```json
{ "writablePaths": ["/Users/me/.cache/my-build"] }
```

`excludedCommands` is a convenience, not a boundary — a command it matches runs unsandboxed.

### Do the interactive things yourself

The model's `bash` tool cannot drive a login prompt, a pager, or an editor. Use `!`:

```
!!gh auth login
```

Your command, your shell, never sandboxed. Use `!!` when the output is only for you — a login flow is noise the model does not need — and plain `!` when you want the result recorded so the model can act on it in your next message.

## Making it repeatable

### Skills for procedures

Anything you have explained twice should be a skill. A skill is both a slash command *and* something the model can invoke itself when the task matches:

```
~/.plank/skills/release/SKILL.md   →   /release 2.8.0
```

### Templates for prompt shapes

Anything you retype with small variations should be a template — one file, `{{named}}` holes:

```
~/.plank/templates/review.md   →   /review src/parser.rs "error handling"
```

### Hooks for policy

Anything that must happen *every* time belongs in a hook, not in your memory of what to check. A `PreToolUse` hook exiting 2 blocks a tool and tells the model why, which is how you enforce "never edit generated files" without repeating it every session.

```json
{
  "PreToolUse": [
    { "matcher": "edit(*/generated/*)|write(*/generated/*)",
      "hooks": [ { "type": "command", "command": "echo 'generated code — edit the template instead' >&2; exit 2" } ] }
  ]
}
```

### Project settings in the repo

`./.plank/settings.json` is committable, and sharing it is the point: everyone on the project gets the same context size, the same sandbox policy, the same display defaults. Keep secrets out — the file is inside the working tree.

## Getting more out of the hardware

- **`/power <1..100>`** caps GPU draw mid-session, for when you want the laptop to stay cool or quiet.
- **`--ssd-streaming`** streams experts from SSD rather than loading them resident, which is how you run a model that does not fit in RAM. `--ssd-streaming-cache-experts` and `--ssd-streaming-preload-experts` trade memory for latency.
- **`--mtp PATH`** adds a multi-token-prediction draft model, with `--mtp-draft` and `--mtp-margin` tuning how aggressively drafts are accepted.
- **`plank serve` + `--remote`** puts the work on the machine with the GPU and the typing on the machine you are sitting at. See [Remote and hosted engines](10-remote-and-providers.md).

## When something goes wrong

```
/repro
```

before you change anything. It captures the exact rendered prompt the engine would see plus every runtime knob — model, backend, context size, sampling, think mode, engine tuning — in one self-contained file, without touching the live session. That is a far better bug report than a description, and it is the only artifact that makes a generation bug reproducible after the fact.

---

Next: [Troubleshooting →](13-troubleshooting.md)

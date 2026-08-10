[← Configuration](08-configuration.md) · [Index](README.md) · Next: [Remote and hosted engines →](10-remote-and-providers.md)

# 9. Extending plank

Five extension points, all discovered the same way: a global directory or file under `~/.plank/`, overlaid by a project-scoped one in the working directory, where the project version wins on a name collision.

| Extension | Global | Project |
|---|---|---|
| Skills | `~/.plank/skills/` | `./.plank/skills/` |
| Templates | `~/.plank/templates/` | `./.plank/templates/` |
| Subagents | `~/.plank/agents/*.md` | `./.plank/agents/*.md` |
| Hooks | `~/.plank/hooks.json` | `./.plank/hooks.json` |
| MCP servers | `~/.plank/.mcp.json` | `./.mcp.json` |

## Skills

A skill is a packaged procedure: a directory containing `SKILL.md`, with optional frontmatter followed by the prompt body.

```
~/.plank/skills/release/SKILL.md
```

```markdown
---
name: release
description: Cut a release — bump, tag, changelog, publish
argument-hint: <version>
---

Cut a release for version $ARGUMENTS:

1. Bump the version in Cargo.toml
2. Update CHANGELOG.md from the commits since the last tag
3. Commit, tag, and push
```

Invoking `/release 2.8.0` splices the body — with `$ARGUMENTS` substituted — in as a preamble and runs a normal turn.

Skills are **both user- and model-invoked**: you can type `/release`, and the model can reach for the `skill` tool when a task matches one. `/skills` lists what is installed.

## Templates

A template is the lighter-weight sibling of a skill: one `*.md` file whose stem is the command name, with named holes instead of a single `$ARGUMENTS` splice.

```
~/.plank/templates/review.md
```

```markdown
---
description: Review a file against a specific concern
argument-hint: <path> <concern>
---

Review {{path}}, focusing specifically on {{concern}}.
```

Interpolation is deliberately minimal: `{{name}}` and nothing else. No conditionals, no expressions. `/templates` lists them, and built-in commands always win over a template of the same name.

**Skill or template?** A skill is a directory with a procedure the model may also choose on its own; a template is a file that saves you retyping a prompt shape.

## Subagents

A subagent runs a delegated task as a fork of the current conversation: the framed task is appended, a normal turn loop runs with tools, and afterwards the fork is truncated so **only the final report** enters your transcript. Because the fork shares the parent's prefix, it costs almost nothing to enter and is rolled back cleanly on the next real turn.

```
/subagent audit every unwrap() in src/ and report the risky ones
```

Beyond the built-in general-purpose subagent, define named ones as markdown files:

```
~/.plank/agents/reviewer.md
```

Each supplies extra instructions that frame the subagent's turn. Dispatch one by name with a colon on the command itself:

```
/subagent:reviewer check the diff on this branch
```

`/agent` lists them. The name is part of the command rather than the first word of the task, so a task that happens to start with a definition's name is never quietly reinterpreted as a persona — and because the name is explicit, one that does not exist is reported rather than silently falling back to the general-purpose subagent. While you type, the TUI colours the `:<name>` **green** when it resolves and **red** when it does not, so a typo shows up before you press Enter.

The model can delegate on its own with the `agent` tool. Delegation is bounded at one level — a subagent cannot itself delegate — which keeps a runaway from spawning a tree.

What comes back is the answer, not the deliberation: the report is stripped of the subagent's thinking before it enters your transcript, and the subagent is asked to state its conclusion once with the reason attached. Its reasoning is still there to read in the roster while it works.

While agents are running, the TUI lists them in a roster below the status bar — one row each, with the task, the clock and the token spend, and `←` then `Enter` to read any agent's output in full. See [The agent roster](03-the-interface.md#the-agent-roster).

When the subagent reports back, plank runs a turn on that report: delegated work comes back into the conversation and gets acted on, rather than sitting in the transcript until you type again.

### Giving a subagent its own worktree

Add `isolation: worktree` to a definition's frontmatter and each run of that subagent gets its own throwaway checkout, so several agents working at once cannot overwrite each other's edits:

```markdown
---
name: refactorer
description: Restructures code without touching the main checkout
isolation: worktree
---
```

The subagent is told where it is and to translate inherited paths accordingly. When it finishes, a worktree it left clean is removed; one holding changes is **kept**, and its path is reported back so you can review and merge it. Setting `worktree.isolateAgents` to `true` in `settings.json` turns this on for every subagent instead of one at a time. It is off by default: a checkout per agent is not free, and the work then has to be merged back.

### Running a subagent on a different engine

A definition can name its own engine, and then its sidechain runs there instead of on whatever the main agent uses:

```markdown
---
name: cheap-local
description: grep-and-summarise work that does not need the expensive model
provider: local
---
Be terse. Report findings, not process.
```

`provider: local` means the local ds4 engine specifically. Under a hosted main agent, plank loads it alongside the provider at startup — that is a real memory cost, and it says so before it loads, so only an explicit definition triggers it. A definition can equally point at a hosted provider (`provider: anthropic`, a model, and the environment variable holding the key); `/agent` lists each one's engine and tells you when its key variable is unset.

Two consequences worth knowing:

- **A cross-engine sidechain is clean-room.** The parent transcript is hidden and only the framed task is sent, so a hosted subagent is never billed for your conversation and a local one never has to prefill it.
- **`/agent` and the status bar both show it.** Every engine in play is named in the bar, so you can see which one is working — see [The interface](03-the-interface.md#the-screen).

## Hooks

Hooks run your shell commands (or inject static prompts) at lifecycle points. Configuration is JSON, merged from `~/.plank/hooks.json` then `./.plank/hooks.json`, with both lists running.

```json
{
  "PreToolUse": [
    { "matcher": "bash|edit",
      "hooks": [ { "type": "command", "command": "check.sh", "timeout": 60 } ] }
  ]
}
```

### Events

| Event | Fires | Exit code 2 means |
|---|---|---|
| `PreToolUse` | before a tool runs | block the tool; stderr becomes the model-visible error |
| `PostToolUse` | after a tool runs | append stderr to the model's observation |
| `PostToolUseFailure` | after a tool that failed | (carries the error) |
| `Stop` | as a turn concludes | feed stderr back and continue the turn (once per turn) |
| `UserPromptSubmit` | every submitted prompt | may inject turn context |
| `SessionStart` | session begins (`startup`/`resume`/`clear`/`compact`) | may inject context |
| `SessionEnd` | session ends (with a reason) | — |
| `PreCompact` / `PostCompact` | around a compaction pass | may inject context; both carry `trigger` (`manual` for `/compact`, `auto` for a threshold pass) and `PostCompact` carries the summary. `PostCompact` does not fire for a pass that was interrupted or produced no summary, since no compaction happened |
| `WorktreeCreate` / `WorktreeRemove` | plank needs a worktree made or destroyed | (see below) |

### The protocol

Hook input is a JSON object piped to the command's stdin. Beyond exit codes, a command hook may print a JSON response envelope on stdout:

| Field | Effect |
|---|---|
| `continue: false` + `stopReason` | halt the turn |
| `systemMessage` | warn the user |
| `suppressOutput` | keep the output out of the display |
| `async: true` + `asyncTimeout` | run without blocking |

Any other nonzero exit shows stderr to **you** only, not the model.

Matchers alternate on tool name and can match arguments: `bash|edit`, `bash(git *)`, `write(*.md)`. An empty or missing matcher matches every tool. Unknown event names load with a warning rather than failing.

The worktree events are different in kind from the rest: configuring `WorktreeCreate` **replaces** git as plank's worktree backend rather than adding to it, which is how a non-git VCS can be driven. The hook is given the requested name and must print the resulting directory on stdout; `WorktreeRemove` is given a path to destroy. If you configure the first without the second, plank will refuse to remove what it cannot remove rather than guessing.

A hook can also be `{"type": "prompt", "prompt": "…"}` — static text injected to the model instead of a command to run.

`/hooks` shows what is configured.

## MCP servers

plank loads external tools from stdio and Streamable HTTP MCP servers. `~/.plank/.mcp.json` applies globally; `./.mcp.json` (or `--mcp-config FILE`) overrides same-named servers and adds new ones.

```json
{
  "mcpServers": {
    "demo": {
      "command": "some-mcp-server",
      "args": ["--flag"],
      "env": {"KEY": "value"},
      "primaryTools": ["tool_a"]
    },
    "remote": {
      "type": "http",
      "url": "http://127.0.0.1:6510/mcp",
      "headers": {"Authorization": "Bearer <token>"}
    }
  }
}
```

A `command` entry is spawned as a stdio subprocess; a `url` entry is reached over Streamable HTTP, with optional `headers` for auth.

Tools reach the model as `mcp__<server>__<tool>`. **`primaryTools` controls prompt size**: listed tools get their full schema in the system prompt, everything else appears in a compact directory and is described on demand via the built-in `mcp_describe` tool. Omit the key and every tool is primary — fine for a small server, expensive for a large one.

Servers can also publish **resources**, which the model reads with `mcp_list_resources` and `mcp_read_resource`.

Once a server is connected you just ask for what it does — you do not name its tools:

```
what's the code health score as reported by tokensave?
what was my last published article on medium?
```

`/mcp` shows connected servers and their tools. A server that misses the response deadline is dropped along with all of its tools, so check `mcp.timeoutSecs` if one is slow to start.

### When a global server fails to start

plank does **not** simply drop it from the prompt. A global server's tool schemas are part of the cached system prompt, so losing one would change that prompt and force the most expensive possible reload. Instead plank substitutes the server's **last-known-good advertisement** from `~/.plank/mcp-advert/`, keeping the prompt byte-identical and the cache valid.

The consequence to know about: after a failed start, the model still believes those tools exist, and calls to them fail at dispatch rather than being avoided. If a server's tools are erroring in a way that makes no sense, check `~/.plank/errors.log` for a line about a substituted advertisement.

This applies to global servers only. Project-local servers are cheap to rebuild, so they get no cached record — a dead project server simply is not advertised.

---

Next: [Remote and hosted engines →](10-remote-and-providers.md)

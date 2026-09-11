---
name: update-config
description: Use when the user wants to change plank's configuration - settings.json keys, permissions and sandbox roots, hooks that must fire automatically ("whenever X", "after every Y"), MCP servers, or plugin behavior. plank executes hooks, not the model, so an automated behavior can only be delivered as a hook.
argument-hint: [what to configure]
---

# Configure plank

The user wants to change plank's configuration: $ARGUMENTS

## Where Configuration Lives

| File | Scope | Git | Use for |
|------|-------|-----|---------|
| `~/.plank/settings.json` | user | n/a | preferences across all projects |
| `./.plank/settings.json` | project | inside the worktree | project-wide preferences |
| `~/.plank/hooks.json`, `./.plank/hooks.json` | user, project | same | hooks |
| `~/.plank/.mcp.json`, `./.mcp.json` | user, project | same | MCP servers |
| a plugin's `settings.json` | plugin | in the plugin | below the user file, always |

Precedence: built-in defaults < `~/.plank/settings.json` < `./.plank/settings.json`
< env < CLI flags. A missing, unreadable or malformed file falls back to
defaults per key - a broken settings file degrades preferences, never startup.

**Secrets never go here.** `./.plank/settings.json` sits inside the worktree
and is easy to commit by accident; the provider API key stays on the
environment or the command line.

## Settings Sections

    {
      "engine":  { "model", "threads", "backend", "power", "ctx", "thinkingToolCalls" },
      "ui":      { "respectGitignore", "popupRows", "indexRefreshSecs", "historySize",
                   "showToolCalls", "showToolResults", "showThinking", "crtOff",
                   "easterEggs", "screensaver", "screensaverFace" },
      "safety":  { "sandbox", "btwSuspend" },
      "mcp":     { "timeoutSecs" },
      "ask":     { "maxOptions" },
      "agents":  { "autoRoute", "maxParallel" },
      "git":     { "signCommits" },
      "context": { "microcompact" },
      "tools":   { "repeatAdvisory", "loopGuards", "callTimeoutSec", "spillMaxBytes",
                   "spillPreviewBytes", "recall", "fanout", "runCode", "bashNotify" }
    }

Read `settings.rs` for the authoritative list and each default - do not quote
a default from memory, and be aware that a doc comment there can lag the
actual `Default` impl.

Two things to check before promising an effect:

1. **Live vs restart-bound.** Most keys are read fresh at the point of use, so
   a `/config` save takes effect on the next read. A value captured once at
   startup (engine sizing, for instance) needs a restart. Say which applies.
2. **What a plugin may not set.** `engine.*`, `worktree.*`, `tools.*` and
   `pluginConfig` are dropped from a plugin's `settings.json` with a warning
   surfaced through `settings_audit_warnings` (`PLUGIN_REFUSED_SECTIONS`).

For simple toggles, point the user at the `/config` form
(`/config <section>.<key> <value>`) rather than hand-editing JSON.

## Hooks: the Only Way to Automate a Behavior

"From now on, whenever X, do Y" cannot be satisfied by memory or by a
preference: plank runs hooks, the model does not. Write a hook.

Events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `Stop`,
`UserPromptSubmit`, `SessionStart`, `SessionEnd`, `PreCompact`, `PostCompact`,
`WorktreeCreate`, `WorktreeRemove`. An unknown event name loads with a warning
rather than failing.

A hook is a `command` (a shell command) or a `prompt` (static text injected to
the model). Matchers alternate on `|`, and each alternative is either a bare
tool name (`bash`) or a name with an argument glob (`bash(git *)`,
`write(*.md)`). An empty matcher matches everything. Lifecycle events match on
the event's own discriminator: `SessionStart`'s source (`startup`, `clear`,
`compact`, `resume`), `SessionEnd`'s reason, compaction's trigger.

Exit codes: 0 succeeds (on a context-capable event its stdout is injected as
turn context), 2 blocks and feeds stderr back, anything else warns the user.
A command hook may also print a JSON envelope on stdout:
`{"continue": false, "stopReason": "..."}` halts the turn, `systemMessage`
warns the user, `suppressOutput` keeps stdout out of the transcript, and
`additionalContext` (also read from `hookSpecificOutput.additionalContext` and
`additional_context`) injects context on a context-capable event.
`async: true` makes a hook fire-and-forget.

## Workflow

1. Clarify what the user actually wants, and decide which file it belongs in -
   personal preference goes user-scope, anything the project needs goes
   project-scope.
2. Read the existing file before writing. Never clobber keys you did not come
   for.
3. Merge, write, and show the diff of what changed.
4. State plainly whether it is live now or needs a restart, and how to undo it.
5. Verify: `/config` for a settings key, `/hooks` for the hook listing.

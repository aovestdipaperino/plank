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

Later wins: built-in defaults, then `~/.plank/settings.json`, then
`./.plank/settings.json`, then the environment, then CLI flags. Absent,
unreadable and malformed files are all handled the same way, key by key,
against the defaults - so the worst a mangled settings file can do is cost you
your preferences, not the launch.

**Keep secrets out.** Anything under `./.plank/` is a `git add` away from
being published; provider API keys belong in the environment or on the command
line and nowhere near these files.

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

`settings.rs` is where the real list and the real defaults are. Never recite a
default from memory, and when you do read it, believe the `Default` impl over
a doc comment that may have been left behind.

Two things to check before promising an effect:

1. **Live vs restart-bound.** Most keys are read fresh at the point of use, so
   a `/config` save takes effect on the next read. A value captured once at
   startup (engine sizing, for instance) needs a restart. Say which applies.
2. **Plugins are not allowed everywhere.** Four sections - `engine.*`,
   `worktree.*`, `tools.*`, `pluginConfig` - are stripped out of a plugin's
   `settings.json` (`PLUGIN_REFUSED_SECTIONS`), and the user hears about it
   through `settings_audit_warnings`.

For simple toggles, point the user at the `/config` form
(`/config <section>.<key> <value>`) rather than hand-editing JSON.

## Hooks: the Only Way to Automate a Behavior

No amount of remembering delivers "from now on, whenever X, do Y" - the model
is not what runs between turns, plank is, and what plank runs is hooks. So
write one.

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

What the exit code buys you: 0 is success, and where the event can carry
context, stdout becomes that context; 2 blocks the action and returns stderr
to the model; every other code is reported to the user as a warning.
A command hook may also print a JSON envelope on stdout:
`{"continue": false, "stopReason": "..."}` halts the turn, `systemMessage`
warns the user, `suppressOutput` keeps stdout out of the transcript, and
`additionalContext` (also read from `hookSpecificOutput.additionalContext` and
`additional_context`) injects context on a context-capable event.
`async: true` makes a hook fire-and-forget.

## Workflow

1. Pin down the actual request, then pick the file by who it serves: a taste
   of the user's is user-scope, a requirement of the repo's is project-scope.
2. Open the file before you write it, and leave every key you did not come for
   exactly as it was.
3. Merge, write, and show the diff of what changed.
4. State plainly whether it is live now or needs a restart, and how to undo it.
5. Verify: `/config` for a settings key, `/hooks` for the hook listing.

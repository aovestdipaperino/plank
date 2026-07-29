# plank user guide

plank is an interactive coding agent for the terminal: a full-screen TUI, a plain line REPL, and a headless one-shot mode, driving either a local DeepSeek V4 Flash model on Apple's Metal backend or a hosted provider.

This guide is for people *using* plank. For how it is built, see [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md); for the exhaustive feature list, [`docs/FEATURES.md`](../docs/FEATURES.md).

> **macOS only.** Real inference uses the ds4 C engine with the Metal backend. Other platforms build and run against the echo stub, which is useful for development but not for real work.

## Contents

1. [Installation](01-installation.md) — Homebrew, channels, building from source, getting the model
2. [Getting started](02-getting-started.md) — your first session, the three front ends, one-shot mode
3. [The interface](03-the-interface.md) — the TUI tour, keybindings, `@` completion, `!` commands, images
4. [Slash commands](04-slash-commands.md) — the complete command reference
5. [Tools](05-tools.md) — what the model can do, the sandbox, plan mode, questions, task lists
6. [Sessions](06-sessions.md) — saving, resuming, checkpoints, branching, export
7. [Context](07-context.md) — the context window, compaction, `AGENTS.md`, memory
8. [Configuration](08-configuration.md) — `settings.json`, command-line flags, precedence
9. [Extending plank](09-extending.md) — skills, templates, subagents, hooks, MCP servers
10. [Remote and hosted engines](10-remote-and-providers.md) — `serve`, `--remote`, shared engines, providers
11. [The arcade](11-arcade.md) — the games that run over a live turn
12. [Advanced workflows](12-advanced-workflows.md) — branching, delegation, long sessions, repeatability
13. [Troubleshooting](13-troubleshooting.md) — when something goes wrong

## The five-minute version

```sh
brew install aovestdipaperino/tap/plank-agent
plank                  # first run offers to download the model
```

Then type. Ask a question, ask for a change, let it read and edit files. When you want to know what plank can do, `/help`; when you want to know where your context went, `/context`; when you want to come back tomorrow, just quit — the session saves itself and `/resume` brings it back.

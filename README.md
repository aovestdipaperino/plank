# plank

<p align="center">
  <img src="assets/logo.png" alt="Plank logo" width="300">
</p>

Plank is a fast-moving agent harness built on the [ds4](https://github.com/aovestdipaperino/ds4) C reference (`ds4_agent`). It was ported functionality-by-functionality (not line-by-line), with each C section becoming an idiomatic Rust module, so changes landing in `ds4_agent` stay easy to port over — the upstream remains the source of truth for wire formats and prompt text, while plank iterates quickly on everything around it.

Plank is an interactive coding agent with a Ratatui TUI, a plain terminal REPL, a one-shot headless mode, and a set of built-in tools (shell, file read/edit, web).

> **macOS only.** Plank targets macOS exclusively: inference uses the original ds4 C engine with the Metal backend, linked via the `ds4-ref` submodule. Other platforms are not supported.

## Installing

Homebrew is the only distribution channel (plank is not on crates.io):

```sh
brew tap aovestdipaperino/tap
brew install plank         # stable channel
brew install plank-beta    # beta channel
```

Or in one step without a prior tap: `brew install aovestdipaperino/tap/plank`. Prebuilt bottles exist for Apple Silicon and Intel Macs; on other setups Homebrew builds from source (requires Rust). Upgrade with `brew upgrade plank`.

Releases follow a two-channel scheme: the highest major version is always the beta. When a beta is promoted to stable, the next major opens as the new (initially empty) beta — e.g. promoting v8.x to stable creates v9.0.0 as the new beta. The two formulas conflict since both install a `plank` binary, so switch channels with `brew uninstall plank && brew install plank-beta` (or the reverse). See [VERSIONING.md](VERSIONING.md) for the channel model and the promote-to-stable process.

## Building

Requires macOS (Apple Silicon or Intel) with the Xcode command line tools. Clone with the submodule to get the ds4 engine:

```sh
git clone --recurse-submodules https://github.com/aovestdipaperino/plank
cd plank
cargo build --release
```

- **With `ds4-ref` present:** `build.rs` builds `libds4core.a` from the Metal-backend objects and links the required frameworks, enabling the `ds4_engine` cfg.
- **Missing submodule:** plank still builds, but without the native engine it uses the echo engine only (useful for development/CI).

You will also need a GGUF model file (e.g. `ds4flash.gguf`) for real inference; see the `download_model.sh` script in `ds4-ref`.

## Usage

```sh
plank            # interactive REPL
plank --help     # full option list
```

Run with a prompt argument for one-shot headless mode.

### Model download

Real inference needs the DeepSeek V4 Flash GGUF. You can point plank at any copy with `-m <path>`, but with no flag it looks in the default location (`~/.plank/ds4flash.gguf`) and, when nothing is there, offers to fetch the quantized model (~87 GB) from Hugging Face — one keypress and it downloads in place with live progress:

<p align="center">
  <img src="assets/model-download.gif" alt="Model download progress UI" width="700">
</p>

Details worth knowing:

- **Resumable.** The download streams to a `.part` file next to the destination; if it's interrupted (Ctrl-C, network drop), the next launch detects the partial file and resumes from where it stopped instead of starting over.
- **Guarded.** The default quant needs ~82 GB resident, so plank refuses to download or load on machines with less than 96 GB of RAM — you find out before spending hours on the transfer, not after.
- **Honest about the wait.** An 87 GB download takes a while; the progress bar keeps you company with size/rate counters and a rotation of two hundred status messages ("Almost sentient. Please hold." among them).
- **Headless-safe.** With stdin not attached to a terminal there is no prompt to answer, so plank exits with instructions instead of hanging a script.

Without a model (or on non-macOS platforms) plank still runs against a built-in echo stub — useful for developing the UI and tools, not for real inference.

### Plank-only features

Plank tracks `ds4_agent` for the core agent loop, but moves faster on the user-facing side. Features that exist only in plank:

- **Ratatui TUI** — the C reference is a plain line REPL; plank auto-selects a full-screen TUI when running on a terminal, with mouse-wheel scrollback.
- **Markdown renderer** — assistant replies render as styled markdown in the TUI via [ratatui-markdown](https://crates.io/crates/ratatui-markdown) (headings, emphasis, lists, tables), with tree-sitter syntax highlighting in fenced code blocks. The in-progress segment is re-rendered as tokens stream in, so partial emphasis and unclosed fences resolve live.
- **Animated status bar** — the spinner verb in the status bar gets a Claude-Code-style shimmer: a bright highlight sweeps across the word while the model is prefilling or generating. The `🪵>` input prompt hides entirely (cursor included) while the agent is busy, reappearing when it's ready for input.
- **`/init`** — asks the model to analyze the codebase and generate an `AGENTS.md` for future sessions (build/test commands, architecture, gotchas).
- **`/context`** — a visual breakdown of context-window usage by category (system prompt, tools, AGENTS.md, conversation), shown below.
- **Session-start context** — plank automatically injects git status, recent commits, discovered `AGENTS.md`/`CLAUDE.md` files, and the current date at the start of each session.
- **`/clear` and `/mcp`** — reset the session in place, and inspect the state of connected MCP servers.
- **`!` commands** — `!<command>` runs a shell command immediately from the prompt (no model round-trip); output stays in the UI with exit-code display, Esc/Ctrl-C kills it.
- **`/hooks`** — command hooks from `~/.plank/hooks.json` + `./.plank/hooks.json` (reference-compatible shape): `PreToolUse` (exit 2 blocks the tool, stderr goes to the model), `PostToolUse` (exit 2 appends stderr to the observation), and `Stop` (exit 2 feeds stderr back and the turn continues, once). Hook input JSON arrives on stdin; other nonzero exits warn the user only.
- **Bash sandbox** — opt-in Seatbelt sandboxing for model-initiated shell commands (`--sandbox`, or `"enabled": true` in `~/.plank/sandbox.json` / `./.plank/sandbox.json`): commands run under `sandbox-exec` with writes limited to the working directory and temp dirs, plus configured `writablePaths`; `excludedCommands` glob patterns skip the sandbox (convenience, not a security boundary). Sandbox denials get a `[sandbox blocked: ...]` hint in the tool result so the model can react. User-typed `!` commands are never sandboxed — typing the command is the authorization.
- **Persistent memory (`/remember`)** — layered memory files loaded into session-start context: `~/.plank/MEMORY.md` (user scope) and `./.plank/MEMORY.md` (project scope). `/remember <text>` appends a dated note to the project file, `/remember user <text>` to the user file; the file template documents the four entry types worth keeping (user, feedback, project, reference). Oversized files are tail-truncated at injection so memory can't crowd out the conversation.
- **`/resume` and `/tag`** — `/resume` shows the most recent sessions numbered, with tag and last prompt; `/resume <number>` (or a sha prefix) continues one. `/tag <text>` labels the current session for the listings. Listing metadata is stored in a trailer record read with bounded head/tail reads, so `/list` and `/resume` never parse whole transcripts.
- **`/subagent`** — `/subagent <task>` delegates a task to a sidechain run of the same model: the subagent works with full tool access on a fork of the conversation, and only its final report is carried back into the transcript (framed as background context). The fork shares the conversation's KV-cache prefix going in and is rolled back afterwards, so delegation costs one report instead of the whole sidechain. Named agent definitions and teams are tracked separately.
- **`/skills`** — markdown prompt templates in `~/.plank/skills/<name>/SKILL.md` (overlaid by `./.plank/skills`) become slash commands: frontmatter gives `name`/`description`/`argument-hint`, the body is injected as the user turn with `$ARGUMENTS` substituted.
- **Interruptible everywhere** — Esc or Ctrl-C stops generation *and* prompt prefill; the engine's cancel callback aborts mid-sync instead of making you wait out a long prompt.
- **Text selection and clipboard** — click-drag selects rendered output (WYSIWYG, wrapped lines included) and copies it to the system clipboard on release via `pbcopy` plus OSC 52 (so copy also works over SSH); Cmd-V pastes into the prompt through bracketed paste.
- **Hierarchical MCP configs with `primaryTools`** — user-scope plus project-scope config files, and per-server control over which tool schemas go in the system prompt (see below).

### Highlights

Assistant replies render as markdown in the TUI, with tree-sitter syntax highlighting for fenced code blocks:

<p align="center">
  <img src="assets/syntax-highlighting.png" alt="Syntax-highlighted Rust code in the plank TUI" width="700">
</p>

The `/context` command visualizes context-window usage by category:

<p align="center">
  <img src="assets/context-usage.png" alt="/context report showing token usage by category" width="700">
</p>

### MCP servers

Plank can load external tools from stdio MCP servers. Configs are hierarchical like Claude Code's user and project scopes: `~/.plank/.mcp.json` applies globally, and `./.mcp.json` in the working directory (or the file given with `--mcp-config`) overrides same-named servers and adds new ones. Both use the standard `mcpServers` format:

```json
{
  "mcpServers": {
    "demo": {
      "command": "some-mcp-server",
      "args": ["--flag"],
      "env": {"KEY": "value"},
      "primaryTools": ["tool_a"]
    }
  }
}
```

Tools are exposed to the model as `mcp__<server>__<tool>`. The optional `primaryTools` list controls prompt size: listed tools get their full schema in the system prompt, the rest appear in a compact directory and are described on demand via the built-in `mcp_describe` tool. Omit the key to make every tool primary.

## Project layout

Each module in `src/` maps to one functional section of the original `ds4_agent.c`:

- `engine.rs` / `ds4engine.rs` / `ffi.rs` — inference engine abstraction and native ds4 bindings
- `session.rs`, `compact.rs`, `sysprompt.rs` — conversation state, compaction, system prompt
- `tools/` — built-in agent tools (bash, edit, files, web) and the MCP client
- `ui.rs`, `render.rs`, `statusbar.rs`, `editor.rs`, `viz.rs` — terminal UI
- `config.rs`, `trace.rs`, `interrupt.rs`, `status.rs` — configuration, tracing, signal handling

## Star History

<!-- Chart is rendered in CI by .github/workflows/star-history.yml (the hosted
     star-history.com embed broke with GitHub's 2026-06-30 stargazers API
     restriction). The action rewrites everything between these markers. -->
<!-- star-history:start -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/star-history/star-history-dark.svg">
  <img alt="Star history" src="assets/star-history/star-history-light.svg">
</picture>
<!-- star-history:end -->

## License

[MIT](LICENSE)

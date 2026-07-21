# plank

<p align="center">
  <img src="assets/logo.png" alt="Plank logo" width="300">
</p>

Plank is a fast-moving agent harness built on the [ds4](https://github.com/aovestdipaperino/ds4) C reference (`ds4_agent`). It was ported functionality-by-functionality (not line-by-line), with each C section becoming an idiomatic Rust module, so changes landing in `ds4_agent` stay easy to port over — the upstream remains the source of truth for wire formats and prompt text, while plank iterates quickly on everything around it.

Plank is an interactive coding agent with a Ratatui TUI, a plain terminal REPL, a one-shot headless mode, and a set of built-in tools (shell, file read/edit, web).

> **macOS only.** Plank targets macOS exclusively: inference uses the original ds4 C engine with the Metal backend, linked via the `refs/ds4` submodule. Other platforms are not supported.

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

- **With `refs/ds4` present:** `build.rs` builds `libds4core.a` from the Metal-backend objects and links the required frameworks, enabling the `ds4_engine` cfg.
- **Missing submodule:** plank still builds, but without the native engine it uses the echo engine only (useful for development/CI).

You will also need a GGUF model file (e.g. `ds4flash.gguf`) for real inference; see the `download_model.sh` script in `refs/ds4`.

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
- **Animated status bar** — the spinner verb in the status bar gets a Claude-Code-style shimmer: a bright highlight sweeps across the word while the model is prefilling or generating.
- **Type while it thinks** — each turn runs on a worker thread (the C reference's model-worker architecture), so the prompt stays live during generation: type the next message and press Enter to queue it. Queued messages join the conversation at the next tool round, or start the next turn when the current one settles. Esc interrupts; Ctrl-C clears the typed line first, then interrupts on an empty line.
- **`/init`** — asks the model to analyze the codebase and generate an `AGENTS.md` for future sessions (build/test commands, architecture, gotchas).
- **`/context`** — a visual breakdown of context-window usage by category (system prompt, tools, AGENTS.md, conversation), shown below.
- **`/usage`** — cumulative billed token usage for online models this session (input, output, and — on providers that cache prompts, like Anthropic — cache read/write with hit rate). Empty on the local engine, which has no billed usage.
- **Session-start context** — plank automatically injects git status, recent commits, discovered `AGENTS.md`/`CLAUDE.md` files, and the current date at the start of each session.
- **`/clear` and `/mcp`** — reset the session in place, and inspect the state of connected MCP servers.
- **`!` commands** — `!<command>` runs a shell command immediately from the prompt (no model round-trip); output stays in the UI with exit-code display, Esc/Ctrl-C kills it.
- **`/hooks`** — command hooks from `~/.plank/hooks.json` + `./.plank/hooks.json` (reference-compatible shape): `PreToolUse` (exit 2 blocks the tool, stderr goes to the model), `PostToolUse` (exit 2 appends stderr to the observation), and `Stop` (exit 2 feeds stderr back and the turn continues, once). Hook input JSON arrives on stdin; other nonzero exits warn the user only.
- **Bash sandbox** — opt-in Seatbelt sandboxing for model-initiated shell commands (`--sandbox`, or `"enabled": true` in `~/.plank/sandbox.json` / `./.plank/sandbox.json`): commands run under `sandbox-exec` with writes limited to the working directory and temp dirs, plus configured `writablePaths`; `excludedCommands` glob patterns skip the sandbox (convenience, not a security boundary). Sandbox denials get a `[sandbox blocked: ...]` hint in the tool result so the model can react. User-typed `!` commands are never sandboxed — typing the command is the authorization.
- **Persistent memory (`/remember`)** — layered memory files loaded into session-start context: `~/.plank/MEMORY.md` (user scope) and `./.plank/MEMORY.md` (project scope). `/remember <text>` appends a dated note to the project file, `/remember user <text>` to the user file; the file template documents the four entry types worth keeping (user, feedback, project, reference). Oversized files are tail-truncated at injection so memory can't crowd out the conversation.
- **`/resume` and `/tag`** — `/resume` shows the most recent sessions numbered, with tag and last prompt; `/resume <number>` (or a sha prefix) continues one. `/tag <text>` labels the current session for the listings. Listing metadata is stored in a trailer record read with bounded head/tail reads, so `/list` and `/resume` never parse whole transcripts.
- **Session KV payloads and `/strip`** — `/save` also snapshots the engine's KV state to a fingerprinted sidecar (`<sha>.payload`) next to the transcript, so `/switch` and `/resume` restore it and skip re-prefilling the whole conversation. The fingerprint ties the payload to the exact model, system prompt, and transcript; anything stale is ignored and rebuilt by a normal prefill. `/strip <sha>` drops a session's payload to reclaim disk (the transcript survives), reporting the token count a later resume will re-prefill; `/list` shows each session's payload size or `stripped`.
- **`/checkpoint` and `/rollback`** — `/checkpoint <name>` snapshots the current conversation (transcript plus live engine KV) under a name; `/rollback <name>` truncates back to it and restores the KV, so a rollback resumes with no re-prefill. A rollback first auto-saves the current tail as `pre-rollback`, so it is itself undoable. `/checkpoint` with no argument lists the checkpoints. On an engine without KV snapshots (the echo stub) it degrades to transcript-only rollback.
- **`/subagent`** — `/subagent <task>` delegates a task to a sidechain run of the same model: the subagent works with full tool access on a fork of the conversation, and only its final report is carried back into the transcript (framed as background context). The fork shares the conversation's KV-cache prefix going in and is rolled back afterwards, so delegation costs one report instead of the whole sidechain. Named agent definitions and teams are tracked separately.
- **`/btw`** — ask a side question *while the agent is working* without disturbing the task. `/btw <question>` is answered from the shared conversation context with tools disabled and nothing written to the transcript (it never enters history, compaction, or the next turn). By default a mid-generation `/btw` genuinely **suspends** the running reply: the generation freezes, the aside is answered, and the task resumes byte-for-byte where it left off with zero re-prefill (`--disable-btw-suspend` falls back to the older pause-and-rerun boundary queue). While the answer streams the screen splits — main conversation 60% on the left, the side answer 40% on the right — and Esc cancels it and restores the full-width view. The side prompt reuses the live KV prefix, so it costs only the framed question.
- **`/skills`** — markdown prompt templates in `~/.plank/skills/<name>/SKILL.md` (overlaid by `./.plank/skills`) become slash commands: frontmatter gives `name`/`description`/`argument-hint`, the body is injected as the user turn with `$ARGUMENTS` substituted.
- **`/repro`** — dumps a diagnostic snapshot for bug reports to `~/.plank/repro/repro-<time>.md`: the exact rendered engine input (system prompt, tools, context, full transcript) plus the runtime knobs that shape generation (model, backend, context size, sampling, think mode, engine tuning). `/repro <note>` records a description of the bug. Read-only — the live session is untouched.
- **Interruptible everywhere** — Esc stops generation *and* prompt prefill; the engine's cancel callback aborts mid-sync instead of making you wait out a long prompt.
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

`/btw` answers a side question in a split panel while the main task keeps its place — here the model counts to 20 on the left while a `/btw what is 2 plus 2?` is answered on the right, with nothing written to the conversation:

<p align="center">
  <img src="assets/btw-panel.png" alt="The plank TUI split screen: a counting task on the left, a /btw side answer on the right" width="700">
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

### Remote, hosted, and shared engines (beta)

The v2 beta channel extends plank past a single local process. All of it is off by default; a plain `plank` still runs the local Metal engine exactly as before.

- **Serve and connect** — `plank serve` hosts the local ds4 engine over HTTP+SSE so another machine can use it; `plank --remote <url>` points a thin client at that host (drive from a laptop, infer on the Metal box). The transport is synchronous, adds no async runtime, and streams tokens as they generate. Token auth via `--remote-token` / `$PLANK_REMOTE_TOKEN`; keep it behind an SSH tunnel or a TLS reverse proxy.
- **Hosted providers** — behind the same `Engine` trait, `--provider openai --model <name>` targets any OpenAI-compatible endpoint (`--base-url`, `--api-key` / `$OPENAI_API_KEY`; covers vLLM, Ollama, OpenRouter, Together) and `--provider anthropic` targets the Anthropic Messages API (`$ANTHROPIC_API_KEY`). Native provider tool calls are synthesized back into plank's DSML tool syntax, so tools dispatch identically regardless of backend, and multi-turn tool-call ids are threaded through. Anthropic prompt caching (`cache_control`) is on by default (`--provider-cache`).
- **Shared engine** — `plank serve --shared-engine` loads the weights once and serves many concurrent sessions from a single cooperative GPU thread (round-robin at token granularity; the one Metal queue means time-sliced, not parallel). A freshly attached session restores the warm system-prompt prefix instead of cold-prefilling it. `--max-sessions` and `--kv-budget-bytes` cap admission, `--session-ctx-size` sizes each session's context, and `--idle-reclaim-secs` snapshots idle sessions to disk and restores them on demand; `/info` reports live-session and KV accounting.
- **Remote control** — `plank --control[=ADDR]` opens a loopback WebSocket so another process, a browser, or the `plank remote <url>` terminal client can attach to a running instance: it mirrors the output, sends prompts/commands/`/btw`/interrupts, and takes or hands back control (single controller, many mirrors, with a reconnect grace window). A self-contained web client is served at `/`. Auth is a bearer token (`--control-token`), with an `--control-origin` allow-list for browsers and `--control-queue-max` slow-client eviction.
- **`--ui-remote[=PORT]`** — for driving the TUI from a test harness: opens a `127.0.0.1`-only listener (bare form picks an ephemeral port, `=PORT` a fixed one) accepting line-delimited JSON `keypress`/`snapshot`/`uitree` commands. `snapshot`/`uitree` replies are held until the screen reflects any keys sent first, so a harness can assert without sleeping. One client at a time; a second simply queues.

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

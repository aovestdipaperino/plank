# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What plank is

A Rust port of the `ds4_agent` C reference (an interactive coding agent for the DeepSeek V4 Flash model), ported functionality-by-functionality — each C section became an idiomatic Rust module with its own tests. The C reference lives in the `ds4-ref` git submodule and is the **source of truth for wire formats and prompt text**: tool output framing, the DSML tool-call syntax, and the system prompt must stay byte-for-byte identical to the C, because that's what the model was trained on. `tests/c_parity.rs` enforces this against committed fixtures (and against the C source when the submodule is present); regenerate fixtures with `PLANK_REGEN_FIXTURES=1 cargo test`. Hard-won parity and tooling gotchas are cataloged in `FINDINGS.md` — check it before re-deriving a quirk, and add to it when you pin down a new one. Beware: a `\`-continued Rust string literal strips the next line's leading whitespace — never use continued literals for model-facing text with indentation. macOS only for real inference (Metal).

## Commands

```sh
cargo build                 # debug build (builds the C engine via build.rs when ds4-ref is present)
cargo test --lib            # unit tests — no model needed, pure logic + EchoEngine
cargo test --lib <name>     # single test by substring filter
cargo fmt
cargo clippy --workspace --all-targets   # pedantic + perf lints are warn-by-default in Cargo.toml
```

The pre-commit hook runs `cargo fmt` and the clippy command above; fix warnings rather than allowing them.

- **With the `ds4-ref` submodule present** (macOS): `build.rs` compiles `libds4core.a`, links Foundation/Metal, and emits the `ds4_engine` cfg. Real inference needs a GGUF model (see `download_model.sh` in `ds4-ref`).
- **Without it**: plank still builds and tests, using only the `EchoEngine` stub — this is the normal dev/CI path. Code touching the native engine must be gated with `#[cfg(ds4_engine)]`.

## Architecture

Read `docs/ARCHITECTURE.md` for the full picture (layer diagram, turn lifecycle, module reference). The essentials:

- **Engine trait boundary** (`engine.rs`): all inference sits behind `Engine` (`generate`, `warm_system_prompt`, `count_tokens`, `ctx_size`). `ds4engine.rs` + `ffi.rs` are the real Metal-backed implementation (cfg-gated); `EchoEngine` is the always-available stub that keeps the whole app runnable.
- **Agent core** (`ui.rs`): the `Agent` struct owns engine, session, tools, and system prompt; `run_turn`/`tui_turn` drive the generate → dispatch tools → feed results loop until a generation emits no tool calls. Slash commands are handled here, in **two parallel paths** (plain stdout REPL and Ratatui TUI) — a change to one usually needs the mirror change in the other.
- **Streaming display** (`viz.rs` → `render.rs`/`tui.rs`, `dsml.rs`): model bytes flow through `viz::StreamRenderer` (detects DSML tool calls, emits banners, splits visible vs. thinking text) into a swappable `RenderSink` — ANSI stdout or the Ratatui `OutputLog`. `dsml.rs` is the strict parser producing executable `ToolCall`s.
- **Tools** (`tools/`): `dispatch` mirrors the C tool table — files, edit (with `[upto]` anchoring), bash (sync + async jobs), web, plus the MCP stdio client (`mcp.rs`, hierarchical `~/.plank/.mcp.json` + `./.mcp.json` configs).
- **Sessions & context** (`session.rs`, `compact.rs`, `sysprompt.rs`, `context.rs`): transcript persistence under `~/.plank/kvcache` with SHA-1 identities, compaction (durable summary + verbatim tail), system prompt text, and session-start context (git status, AGENTS.md discovery, date).
- **KV-cache discipline**: `Ds4Engine` keeps one live session across turns so only the new suffix is prefilled; the system prompt has a fingerprinted disk snapshot (`sysprompt.kv`). Reuse only genuinely matching token prefixes — a stale checkpoint is rebuilt, never trusted.
- **Front-end selection** (`main.rs`): TTY on both ends → Ratatui TUI; piped → plain line REPL; `--non-interactive` → headless stdin protocol.

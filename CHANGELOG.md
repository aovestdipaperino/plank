# Changelog

All notable changes to plank are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Ratatui full-screen UI** for interactive sessions. Uses the alternate
  screen buffer so block-based terminals like Warp render plank cleanly. Draws
  a scrollback area, a pinned input line, and a reverse-video status bar, with
  the logo shown inside its own scrollback.
- **Real ds4 inference engine** via FFI (`-m/--model`), built from the
  `ds4-ref` submodule on macOS (Metal backend). Kept behind an `Engine` trait
  with an `EchoEngine` fallback when no model is loaded.
- **System-prompt KV cache** reuse across turns: the live session is kept
  alive so only the new suffix is prefilled, and the progress bar reflects the
  cached prefix.
- **System-prompt cache warm-up** at startup ("Updating system prompt cache...")
  with a disk checkpoint (`sysprompt.kv`) fingerprinted by model + system
  prompt, so a fresh launch restores the prefilled KV instead of recomputing it.
- **Live progress/status display**: a prefill progress bar (filled arrows in
  magenta, matching the C agent) and a generation status line (tokens, t/s,
  context usage).
- **Context compaction** with the durable-summary + verbatim-tail rebuild, plus
  automatic triggering under context pressure.
- **Session persistence**: save/load/list/switch/delete with SHA-1 identities
  and history rendering (`/save`, `/list`, `/switch`, `/del`, `/history`,
  `/strip`).
- **Tool suite**: file read/more/write/list, edit with `[upto]` anchoring,
  search, synchronous and async bash jobs, and browser web tools
  (`google_search`, `visit_page`).
- **Streaming DSML tool-call parser** and tool-call visualization (banners for
  bash/read/edit/diffs), suppressing raw markup from display.
- **Markdown/token rendering** with syntax highlighting and gray thinking text.
- **Trace logging** (`--trace`), SIGINT-based generation interrupt, and a
  headless mode (`--non-interactive`) with the stdin quiet-window protocol.
- Default context window of 1M tokens (`1048576`), displayed as `1.0M`.
- **Automatic model download.** With no `-m`, plank looks for
  `~/.plank/ds4flash.gguf` and, if missing, offers to fetch the DeepSeek V4
  Flash GGUF from Hugging Face. The download runs on a Ratatui alternate screen
  (so it repaints in place everywhere, including Warp) with a red gauge and a
  rotating series of 200 "downloading alien/genius intelligence" one-liners.
  Resumable via `curl -C -`; the prompt defaults to yes; curl runs in its own
  process group so cancelling never touches the parent shell.
- **RAM guard.** plank refuses to download or load the model on machines with
  less than 96 GB of physical RAM (the recommended minimum for this quant).
- **`docs/ARCHITECTURE.md`** describing the module layout and data flows.

### Notes

- Ported functionality-by-functionality from the `ds4_agent.c` reference
  (tracked as the `ds4-ref` submodule), not line-by-line.
- Web-tool approval currently reads stdin; a TUI modal is a follow-up.

## [0.1.0]

- Initial commit: plank, a Rust port of the ds4 agent, with README and logo.

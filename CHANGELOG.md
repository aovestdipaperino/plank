# Changelog

All notable changes to plank are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-07-19

Opens the v2 beta channel and promotes v1.6.0 to stable. No functional changes.

## [1.6.0] - 2026-07-19

### Added

- **Live `/btw` side panel**: the main task resumes the instant a side answer
  finishes (it keeps rendering on the left while the finished answer stays on
  the right). The panel persists across turns and closes only with Esc, and an
  idle `/btw` uses the same panel.
- **Memorable session names**: session ids are now `adjective-celebrity` names
  (e.g. `deadly-einstein`) minted on first save, drawn from 50 adjectives and
  150 celebrities (75 scientists / 75 historical-pop-sport, ~50% science), with
  a short guid on filename collision. Legacy 40-hex sessions still load and
  list.
- **Resume from the command line**: `plank /resume [name]` resumes a session at
  startup (a name, prefix, list number, or bare for the most recent), showing
  the recovered history.
- **End-of-session dump**: on exit the transcript is saved and plank prints
  where it landed and how to resume it.
- **`/repro`**: writes a diagnostic dump (the exact rendered engine input plus
  the generation knobs) to `~/.plank/repro/` for bug reports.
- A green rule now separates the scrollback from the resting prompt.

### Fixed

- The "cannot load model" crash when a second instance starts: plank probes the
  engine's single-instance lock file first and exits cleanly with a clear
  message instead of the engine's `exit(2)`.

### Changed

- `cargo update`: 12 transitive dependencies refreshed.

## [1.5.0] - 2026-07-19

### Added

- **`/btw` un-gated** (#7): a first-class command, no longer behind the `images`
  feature flag.
- **Split-screen `/btw` panel**: while a side answer streams the screen splits
  (main 60% / side 40%); Esc cancels and restores full width; nothing enters
  the transcript.
- **Priority preemption** (#18): a `/btw` submitted mid-generation pauses the
  running task, answers, then re-runs the interrupted step. Questions typed
  during tool execution answer at the next boundary; a `/btw` during a streaming
  answer joins a FIFO queue (cap 20, drop-oldest).

### Changed

- OpenClaw is vendored as a reference submodule (`refs/openclaw`, shallow,
  CI-skipped) for the side-question design.

## [1.4.0] - 2026-07-19

### Added

- **Worker-thread architecture** (#12): TUI turns run on a worker thread, so the
  prompt stays live during generation — type and queue the next message; queued
  lines join between tool rounds or start the next turn.
- **`/subagent <task>`** (#10): delegates to a sidechain run of the same model
  with full tool access; only the final report returns, and the sidechain's KV
  cost is rolled back.
- **Persistent memory** (#2): `/remember [user] <text>` appends dated notes to
  project or user `MEMORY.md`, loaded into session-start context.
- **`/resume` and `/tag`** (#2): a numbered recent-session picker with tags and
  last prompts, backed by a bounded-read session `meta` trailer (older files
  still load).

## [1.3.0] - 2026-07-19

### Added

- **`/hooks`** (#8): command hooks (PreToolUse / PostToolUse / Stop) from
  `~/.plank/hooks.json` + `./.plank/hooks.json`.
- **Bash sandbox** (#17): opt-in Seatbelt sandboxing for model-initiated shell
  commands (`--sandbox` or `sandbox.json`), writes limited to cwd/temp plus
  `writablePaths`, with `[sandbox blocked: ...]` hints on denials.
- **`/btw`** (#7): first cut, gated behind the experimental `images` flag
  pending the model-format investigation (#18).

## [1.2.1] - 2026-07-19

### Added

- README "Model download" section with an animated demo of the first-run
  download UI (resume support, the 96 GB RAM guard, headless behavior).

## [1.2.0] - 2026-07-19

### Added

- **Layered compaction** (#3): microcompact first (clear old tool-result
  bodies, zero model cost), then structured summarization, with recently read
  files re-attached across the boundary.
- **`/skills`** (#9): markdown `SKILL.md` templates become slash commands with
  `$ARGUMENTS` substitution; `~/.plank/skills` overlaid by `./.plank/skills`.

## [1.1.0] - 2026-07-19

### Added

- **`!` commands** (#4): `!<command>` runs a shell command immediately in both
  UI paths, no model round-trip, output stays in the UI.
- **MCP `instructions`** (#14): a server's initialize `instructions` are
  injected into the system prompt alongside its tool schemas.
- **Parallel git context** (#13): the five session-start git commands run
  concurrently.
- **`docs/SYSTEM-PROMPT.md`** (#5) and a static/volatile prompt-boundary guard
  (#15) that keeps per-session bytes out of the cached prefix.

## [1.0.1] - 2026-07-19

### Fixed

- **#1** Text selection copies to the clipboard (pbcopy + OSC 52); the copy
  path had read a cleared frame buffer.
- **#11** Invalid DSML tool calls no longer leak raw tags; error banners render
  bold red in both the REPL and TUI.
- **#6** The TUI output log is scrollable during generation, with a
  jump-to-bottom hint.
- Status bar: the context gauge updates live during a turn, and elapsed time
  counts the whole tool loop.

### Added

- **C-parity** (#12): the streaming `edit` old-selector preflight aborts doomed
  edits mid-generation with the C's exact error text; malformed and incomplete
  DSML tool calls feed the C's `invalid DSML tool call:` payload plus the syntax
  reminder; greedy (argmax) sampling runs inside DSML stanzas (❄️ indicator);
  and the engine tuning CLI flags are exposed (`--mtp*`, `--prefill-chunk`,
  `--quality`, `--warm-weights`, `--ssd-streaming*`, `--simulate-used-memory`,
  `--dir-steering-*`, `--backend`).

## [1.0.0] - 2026-07-19

Opens the v1 beta channel and promotes v0.9.9 to stable. No functional changes.

## [0.9.10] - 2026-07-19

### Fixed

- Homebrew installs could not load any model: the Metal kernel sources were
  resolved from a compile-time CI path. The kernels now ship in the bottles
  (`share/plank/metal`) and resolve at runtime (`DS4_METAL_DIR` override, then
  the build path, then the exe-relative share dir); the engine-open error now
  reports missing kernels instead of blaming the model file.

## [0.9.9] - 2026-07-19

### Added

- **C-parity byte-diff tests** (`tests/c_parity.rs`): the tools prompt, DSML
  syntax reminder, system-prompt reminder framing, tool-result framing, and
  datetime context line are byte-compared against committed fixtures on every
  test run, and — when the `ds4-ref` submodule is present — against the string
  constants decoded straight out of `ds4_agent.c`. Regenerate fixtures with
  `PLANK_REGEN_FIXTURES=1 cargo test`. The first run caught a real parity
  break: Rust's `\` string-literal continuation strips the next line's leading
  whitespace, which had silently deleted the indentation in the anchored-edit
  example and in every JSON tool schema of the system prompt. The schema
  section now ships as `src/resources/tools_prompt_after_edit.txt` via
  `include_str!` so the bytes are what the model was trained on.
- **`FINDINGS.md`**: a catalog of the wire-format nuances the port must
  preserve (DSML fullwidth bars, dual system-prompt tokenization, KV splice
  of sampled reply tokens, …) and the environment gotchas (macOS 15 SDK,
  Homebrew channel-by-major, download-resume 416 trap, …), so they are
  discovered once instead of per-session.
- **Upgrade cache maintenance** (`src/upgrade.rs`): on the first launch after
  a version change, plank classifies the transition from the version marker
  in `~/.plank/version` and clears exactly the caches the new binary can no
  longer trust — a minor bump drops the sysprompt KV checkpoint, a major bump
  (or downgrade, or missing marker) also drops the image cache. Session
  transcripts are never touched, and everything removed is rebuilt on demand.

- **MCP client** ported from the ds4 `mcp-support` branch: stdio MCP servers
  listed in `./.mcp.json` (or `--mcp-config FILE`) are spawned at startup and
  their tools exposed to the model as `mcp__<server>__<tool>`. A server's
  optional `primaryTools` list keeps the system prompt small: unlisted tools
  appear only in a compact directory and are described on demand via the new
  `mcp_describe` tool.
- **Ratatui full-screen UI** for interactive sessions. Uses the alternate
  screen buffer so block-based terminals like Warp render plank cleanly. Draws
  a scrollback area, a pinned input line, and a reverse-video status bar, with
  the logo shown inside its own scrollback.
- **True-color logo** rendered from `resources/logo.png` via the `logo-art`
  crate. The near-white background is keyed to transparent, and the download
  splash centers it, sized to the terminal.
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

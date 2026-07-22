# Changelog

All notable changes to plank are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Sub-agent tool (`agent`)** (#50): the model delegates a bounded task to a
  fresh scoped sub-agent (a sidechain fork of the transcript) and gets back only
  its final report; nesting is bounded (`SUBAGENT_DEPTH_CAP = 1`). An optional
  `name` selects a `~/.plank/agents` / `./.plank/agents` persona. Wired into both
  the plain-REPL and TUI/worker turn loops.
- **Plan mode (`EnterPlanMode` / `ExitPlanMode`)** (#50): a read-only
  propose-then-approve gate. While active, `write`/`edit`/`bash` are refused and
  read-only tools stay; `ExitPlanMode` presents the plan via the `ask` panel for
  approval (auto-approves in non-interactive runs).
- **Git-style diff card** for `edit` and overwriting `write`: an
  `Update`/`Create(path)` header, an added/removed summary, and `@@` hunks with
  red-background removals and green-background additions (Myers diff via the
  `similar` crate). A `write` to a new file instead streams its content as a dim
  preview while it is generated.
- **`ui.showThinking` setting** (default `true`): when `false`, thinking text is
  produced but not displayed.
- **Read-only reports run mid-turn**: `/context`, `/usage`, `/mcp`, and `/help`
  work while the model is generating, answered from a turn-start snapshot.

### Changed

- The status bar shows context as a bare percentage (`ctx N%`), and the animated
  progress (throbber + spinner verb + token stats) renders on a line pinned
  below the output rather than in the footer. The resting prompt is framed by a
  rule above and below it.
- The system-prompt KV cache, when it needs rebuilding at launch, is warmed
  behind a simple progress bar before the full UI is shown.
- The prompt input word-wraps to the next line instead of scrolling
  horizontally.

### Fixed

- Long scrollback (e.g. the `/context` report) now scrolls all the way to the
  bottom (exact wrapped-line count instead of a char-packing estimate).
- Resumed sessions (`/resume`, `/switch`, `plank /resume`) replay through the
  live renderer, so history returns as markdown with dimmed thinking and
  tool-call banners instead of flat text.

## [2.0.2] - 2026-07-21

Promotes the v2 beta line to stable. Everything accumulated on the beta channel
since v1.6.0 — remote control, remote and hosted engines, the shared engine,
mid-generation `/btw` suspend, checkpoints, per-session KV payloads — ships in
this release, alongside a batch of TUI polish.

### Added

- **Status bar shows the working directory and git branch**: the footer leads
  with the cwd (home collapsed to `~`) and, inside a repository, the current
  branch after a powerline glyph. Both are themed green; the branch is
  discovered with the `git2` crate. Detached HEAD shows a short commit hash.
- **Remote-control interface** (#25): drive a running instance from another
  process or machine over a loopback WebSocket. Mirror output and send
  `prompt`/`command`/`btw`/`interrupt` frames, with single-controller /
  many-mirror handoff and a reconnect grace window. Ships a `plank remote <url>`
  terminal client and a self-contained web client served at `/`. Token auth,
  `--control[=ADDR]`, an `--control-origin` allow-list, and
  `--control-queue-max` slow-client eviction. Also wired the server into the
  live turn loop and added plain-REPL remote drive.
- **Remote and third-party engines** (#26): `plank serve` hosts the local ds4
  engine over HTTP+SSE and `--remote <url>` selects the remote client (sync,
  no async runtime). Third-party providers behind the `Engine` trait:
  `--provider openai` (OpenAI-compatible gateways) and `--provider anthropic`,
  with native tool calls synthesized back into DSML so tools behave identically.
  Anthropic prompt caching via `cache_control` (`--provider-cache`, default on)
  and cross-turn tool-call-id threading.
- **Shared reference-counted engine** (#28): `--shared-engine` serves many
  sessions from one model over a single cooperative GPU thread (round-robin,
  non-preemptible prefill). `--max-sessions` and `--kv-budget-bytes` admission,
  per-session `--session-ctx-size`, idle KV reclamation (`--idle-reclaim-secs`),
  and live `/info` accounting.
- **Mid-generation `/btw` suspend** (#27): an in-pass `/btw` freezes the running
  generation, answers the aside, and resumes with zero re-prefill. On by
  default; `--disable-btw-suspend` restores boundary queueing.
- **`/checkpoint` and `/rollback`** (#29): name a snapshot of the conversation
  (transcript + engine KV) and roll back to it in-session with no re-prefill; a
  rollback is itself undoable via an automatic `pre-rollback` snapshot.
- **Per-session engine KV payloads and `/strip`** (#12): `/save` snapshots the
  engine KV to a fingerprinted `<sha>.payload` sidecar so `/switch` and
  `/resume` skip re-prefilling the whole conversation; `/strip <sha>` reclaims
  the disk. Stale payloads are ignored and rebuilt by a normal prefill.
- **Live command highlighting** in the TUI prompt: a valid `/command` token is
  shown green and the `!` shell-escape marker red as the user types.

### Changed

- **In-pass `/btw` now freezes and resumes by default** rather than
  preempt-and-rerun (see `--disable-btw-suspend` above).
- The session on-disk format carries an optional KV payload sidecar; older
  payload-less sessions still load and list.
- **Prefill footer** now animates with the same spinner verb and throbber as
  token decoding, replacing the static label and progress bar.

### Fixed

- **Scrollback reaches the bottom of long output** (e.g. the `/context`
  report): the view now clamps to ratatui's exact wrapped-line count instead of
  a char-packing estimate that undercounted word-wrapped rows.
- **Resumed sessions render as markdown**: `/resume`, `/switch`, and
  `plank /resume` startup now replay assistant text through the live rendering
  pipeline, so markdown, dimmed thinking, and tool-call banners come back
  instead of flat plain text.

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

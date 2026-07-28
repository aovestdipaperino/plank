# Changelog

All notable changes to plank are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.1] - unreleased

Beta channel on the 2.7 series.

### Added

- **A built-in prompt editor behind `Ctrl-G`**, replacing the shell-out to
  `$EDITOR`. It is an in-process, single-buffer fork of
  [Microsoft Edit](https://github.com/microsoft/edit) (MIT, vendored as the
  `refs/edit` submodule and used as a library): plank suspends its own TUI and
  hands over the raw terminal, exactly as it did for a child editor, but with
  no temp file and no process spawn. Undo/redo, selection, clipboard, find and
  replace, word wrap and line numbers, all reachable from an F10 menubar.
  `Ctrl-S` returns the edited text to the prompt; `Esc` discards it, asking
  first when the text actually changed. There is no Save: the buffer starts
  from a string and ends as one. `ui.builtinEditor` (default `true`) or a build
  without the `builtin_editor` feature falls back to `$EDITOR`.
- **A starfield screensaver**, `ui.screensaver`: `1m` (default), `2m`, `5m`, or
  `never`. After that much idle time at the prompt the perspective starfield
  takes the screen, and the next key, click or paste puts the UI back — the
  waking event is consumed, so it does not leave a stray character behind.
  Idleness is measured only in the idle input loop, so it never appears
  mid-turn, and focus or resize events do not count as activity: a window
  manager moving focus around would otherwise keep it from ever appearing.
  Unlike the games it is not an easter egg, so `ui.easterEggs` does not gate it.

### Changed

- **Ctrl-C now interrupts compaction** instead of being ignored until the
  summary finished. Both compaction paths passed the engine a constant
  "never interrupt" predicate, so a summary pass over a full context could not
  be stopped. An interrupted pass now discards the partial summary, leaves the
  conversation exactly as it was, reports
  `Compaction interrupted; keeping the previous conversation state.`, and ends
  the turn. Ported from the C's cooperative-interruption work.
- **The web tools say what they are doing while they do it**: `google_search`
  and `visit_page` publish `Searching Google for ...` / `Opening page ...`
  before they block, as `✦`-prefixed system status lines. Previously a web call
  looked like a hang until its result landed. The same line style now carries
  every agent-about-itself notice.
- **A tool call started inside an unclosed `<think>` is recovered forward**
  when in-think tool calls are prohibited (`engine.thinkingToolCalls: false`,
  the default). Rather than waiting for a `</think>` that never comes and
  dropping the stanza at parse time, the engine force-feeds `</think>` and lets
  the model restart the call on the executable side of it — the turn does real
  work instead of being spent on a rejected call. With `thinkingToolCalls: true`
  the stanza is dispatched as-is, so nothing is injected. Ported from the C
  server's `chat_think_tool_recovery`; per its findings the stanza opening
  itself is deliberately *not* re-emitted, since the model then reads the call
  as already made and ends the turn.
- **A tool call made inside `<think></think>` is now reported to the model as a
  placement error**, not a syntax one. It used to be fed back behind
  `invalid DSML tool call:` with the DSML syntax reminder attached — and if the
  model had stopped mid-stanza, as `incomplete DSML tool call` — both of which
  send it rewriting markup that was already correct. It now gets the same
  sentence the tools prompt gave it ("Tool calls are not allowed inside
  <think></think>; finish thinking before emitting DSML") plus a note that the
  call was not run and should be re-emitted after `</think>`. The C reference
  routes this through its malformed-tool path; this is a deliberate divergence
  from it.
- **`/stars` is gone** — the starfield is the screensaver now, not a command.
  The arcade is five games (`/pelota`, `/breakout`, `/invaders`, `/centipede`,
  `/frogger`); the plain REPL's static-sky rendering went with the command.
- **`engine.thinkingToolCalls` now defaults to `false`.** Tool calls the model
  emits inside `<think></think>` are discarded with a `[tool call ignored: ...]`
  notice, which is strict `refs/ds4` parity; turn the setting on (it is in
  `/config`) to have plank dispatch them instead.
- The KV warm-up names the tier it is prefilling ("Updating project context
  cache") rather than always claiming the system prompt is being rebuilt.
- `dsml.rs` accepts `SSML` as an alias for the `DSML` marker name. The model
  occasionally spells the marker back with the far more common pretraining
  string; without the alias the stanza parsed as nothing, printed raw, and
  ended the turn with no tool error to retry from. The prompt still teaches
  `DSML` only, so this stays a recovery path rather than a second syntax.
- **Rust 1.93 is now the minimum.** The vendored `edit` crates require it.
  CI already builds on stable; a local toolchain older than that will refuse to
  build with a clear `rust-version` error.

## [2.7.0] - 2026-07-27

Stable release: the 2.6 beta line promoted. One addition on top of it.

### Added

- **`ui.easterEggs`** (default `true`) decides whether the arcade exists. Off is
  stronger than hidden: the six commands stop being known, so `/pelota` reaches
  the model as an ordinary prompt exactly like any other unrecognized slash
  command — which is what a shared or managed install that wants no games in it
  actually needs, rather than a command that is recognized and then refused. Every
  entry point checks it, not just the completion path, since a flag that only hid
  them would leave them reachable by typing. The startup line names the setting
  when it is off, so a `settings.json` cannot quietly remove them without saying
  so, and `/config` exposes it as a toggle.

## [2.6.3] - 2026-07-27

Beta patch bump on the 2.6 series.

### Changed

- The arcade speaks English. The games shipped in 2.6.2 with Italian
  user-facing text while the rest of the UI is English; every displayed string
  is translated — the banners, the five scoreboards, the key-hint footers, the
  exit hint, and the closing and resume lines left in the scrollback. The
  `nuova` and `suono` argument aliases are dropped rather than translated, since
  `new` and `sound` were already accepted and meant the same thing; `reset`
  stays as the one real synonym for `new`. The English footers are longer, so
  they truncate sooner on a narrow terminal, but the exit hint is still the last
  thing to go.

### Added

- The README's arcade section leads with a screenshot, which carries the claim
  prose struggles with: `/breakout` running over a turn that is still streaming,
  with the model's output legible underneath the veil.

## [2.6.2] - 2026-07-27

Beta patch bump on the 2.6 series.

### Added

- Six games behind slash commands — `/stars`, `/pelota`, `/breakout`,
  `/invaders`, `/centipede`, `/frogger` — meant to be played *while the model is
  generating*, which is the point of them: waiting on a long turn is the one
  moment a coding agent has nothing for you to do. They are the only commands
  besides the read-only reports that run mid-turn, and they open as a layer over
  the live output, which keeps streaming underneath. Each keeps its own slot, so
  closing one and reopening it resumes where it was; `new` deals a fresh game and
  `sound` turns on blips, and the two compose (`/breakout new sound`). Keyboard
  and mouse both steer. While a game is up the first `Ctrl-C` closes it and a
  second interrupts the model, so a turn can always be stopped. None of them
  appear in `/help` or the completion popup — deliberately, though a test keeps
  the command list in sync with the dispatcher so one can never be forwarded to
  the model as a prompt. Two limits are worth stating plainly: "translucency" is
  not alpha (a cell holds one character and one pair of colors, so the layer
  underneath is dimmed rather than composited, and the sparse glyphs land in the
  gaps), and "sound" is the terminal bell and nothing else — chosen because it
  adds zero bytes to the binary, at the cost of having no pitch or length, so
  cues differ only in count. Physics runs on a normalized field mapped to the
  terminal at draw time, and follows the rule `anim.rs` already sets: state
  advances only through an injected delta and randomness comes from a seeded
  xorshift, so a whole rally replays identically from its seed and is testable
  without a terminal. See the README for controls.
- Tool calls the model emits inside `<think>` are now dispatched instead of
  ignored, behind `engine.thinkingToolCalls` (default on). The system prompt
  drops its in-think prohibition when the setting is on, so the prompt and the
  renderer agree about what is allowed, and the stanza's `<think>` block is
  closed before the `<tool_result>` that follows so the transcript stays
  well-formed. Turning the setting off restores C-parity behaviour, where such a
  stanza is reported as ignored and not run.
- The window title now names plank's phase rather than always reading
  `🪵 plank`: loading before a front end is up, `READY.` while idle at the
  prompt, and the prompt itself (trimmed to 20 characters) while a turn runs.
  Stamped at both front ends' ready points and all three turn-completion
  boundaries, so the TUI and the plain REPL agree.
- A globally-configured MCP server that fails to start no longer throws away the
  system-prompt cache. Tier 1 is keyed on the prompt text, and that text carries
  every connected server's tool schemas, so one flaky server used to change the
  prompt and force the most expensive re-prefill there is. plank now remembers
  each global server's last successful tool advertisement under
  `~/.plank/mcp-advert/` and renders it when that server cannot start, keeping
  the prompt byte-identical and the cache warm. Startup names the server it is
  serving from cache and warns that its tools will report it as down, `/mcp`
  shows the same alongside the cached tool count, and calling one of those tools
  reports the server as not running rather than the tool as unknown — the two
  need different recovery. Project-local servers are untouched: they key the cheap Tier 2 and
  never get cached definitions, so a project prompt cannot advertise a dead tool.
  Records are dropped when the server leaves `~/.plank/.mcp.json`, and never
  when that file is merely unreadable.
- A system-prompt cache miss now explains itself. Tier 1 is the priciest prefix
  to rebuild — everything below it re-prefills too — so instead of silently
  re-prefilling, plank reports that the system prompt changed and shows the
  first few differing lines, diffed against the prompt text behind the previous
  checkpoint. A benign cause (a ticking MCP tool count, a new date) is obvious
  at a glance. The comparison text lives in a `sysprompt-last.prompt` sidecar
  that is only ever used to explain a miss, never to validate a cache, and it
  is refreshed only after a rebuild actually completes.

### Changed

- The CRT-off exit animation lets the final phosphor dot fade instead of
  blinking out: crt-off 0.1.2 decays it on an exponential, gamma-encoded curve,
  given a 0.9s window (was 0.2s, short enough that the old linear ramp read as an
  instant cut) so the glow visibly dies away.
- A `DEADBEEF` sentinel in an API key marks a mock or stubbed endpoint rather
  than a real provider, so `top_p` is omitted from the request body. The filter
  sits at the one place that has the key, covering both providers across
  structured and flat prompts.

### Fixed

- A synthetic `</think>` is no longer appended when nothing was actually going to
  follow it. The close exists only to keep the transcript well-formed ahead of a
  `<tool_result>`, but it fired whenever the renderer's `<think>` was left open at
  stream end — including a stanza discarded in parity mode and a stream cut short
  by an interrupt, where a real abort gets no such close in the C reference. The
  gate is now the reason the pass continues, and a real interrupt is
  distinguished from an ordinary continuation identically on all three turn
  paths. Two related bugs surfaced while testing it: an ignored in-think stanza
  was synced into the renderer's call list before the ignore check ran, so it
  would have been dispatched despite the notice, and the interrupt early-return
  ran after the gate.
- `/resume` replay no longer renders a stored in-think tool call as
  `[tool call ignored]` directly above its own stored result — the replay
  renderer never received the `thinkingToolCalls` setting that the live one did.
- On a provider engine, the structured tool registry filtered servers on `alive`
  while the text prompt deliberately did not, so an offline shadow was advertised
  in the prompt but missing from the table: the model's call came back as an
  unknown tool instead of the "server is not running" message. Both paths now
  mirror each other, keeping the prompt byte-identical and fp1 stable.
- An offline shadow server now reports as offline everywhere. Reading a cached
  resource URI gave the generic "not available", and listing a shadow's resources
  validated the name and then reported zero — both now return the same offline
  sentence through one shared path so the framing cannot drift.
- Startup read `~/.plank/.mcp.json` three times, and an unreadable first read
  followed by a readable third silently yielded an empty eligible set, costing
  every global server its record refresh and its shadow. It is read once now.

## [2.6.1] - 2026-07-26

Beta channel opened on the 2.6 series. No functional changes: the tag carries
only the version bump, and the work drafted against this section during the
series shipped in 2.6.2, where it is now documented.

## [2.6.0] - 2026-07-25

Stable release: the 2.5 beta line promoted. Two visible fixes to how a turn's
progress is shown while the engine works.

### Changed

- The prefill progress bar now spans only the tokens the current pass actually
  evaluates. It previously ran from the cached prefix to the end of the prompt,
  so a warm turn reusing 8000 tokens and prefilling 200 opened at 97% and
  crawled, while the tok/s figure beside it already counted just the new
  tokens. Bar and throughput now describe the same work.
- `/new` and `/clear` hide the input prompt and show a throbber while the KV
  cache is restored, instead of letting the prompt sit frozen. Restoring the
  tier checkpoint reads a snapshot in the tens of megabytes and loads it into the
  backend, so it is brief but visible. Hiding the prompt also prevents typing
  into a session whose KV is still loading. The plain REPL, which has no
  persistent prompt, prints one transient line and erases it.

## [2.5.5] - 2026-07-25

Beta patch bump on the 2.5 series. One internal refactor with one user-visible
payoff: `/new` no longer stalls, because the KV cache now has a single owner of
its on-disk format instead of five.

### Fixed

- **`/new` and `/clear` no longer rebuild the system-prompt KV cache.** A reset
  makes the next prompt a strict *prefix* of the live KV — a fresh session's
  transcript is the head of the one it replaced — and `ds4_session_sync` cannot
  rewrite behind its live end, so it discarded the whole cache and re-prefilled
  the system prompt from scratch. Worse, `ds4_session_common_prefix` reported
  every token as matching, so the progress bar primed as complete and a
  multi-thousand-token prefill ran with no feedback at all, indistinguishable
  from a hang. A reset now restores the tier checkpoint, so the next turn extends
  it. Measured on DeepSeek V4 Flash for `haiku` → `/new` → `haiku`: a
  2509-token rebuild reported as "100% reused" became a 7-token prefill, and the
  flow went from 31.7s to 19.7s. The post-`/new` state is now identical to a cold
  launch's.
- Prefill progress and the `PLANK_KV_DEBUG` trace no longer conflate "how many
  tokens match the live KV" with "how many will actually be reused". The two
  differ precisely when the engine is about to throw the cache away, so a rebuild
  that genuinely cannot be avoided is now reported honestly instead of as fully
  cached.
- Stale system-prompt checkpoints are garbage-collected. Keying `sysprompt-*.kv`
  by content means every upgrade, global MCP change, or model switch minted a new
  multi-hundred-megabyte snapshot and orphaned the previous one forever; only the
  current one is kept now, and the legacy `sysprompt.kv` is removed.
- KV cache temp files are per-process, so two plank instances persisting the same
  session can no longer interleave into a file that passes its own signature and
  version checks with a spliced body.

### Changed

- **One KV cache format, one owner.** The system-prompt checkpoint, per-project
  tier checkpoints, and session payloads were five code paths — three
  separately reimplementing the same `<fingerprint>\n<bytes>` framing, two
  carrying different payload shapes, plus a legacy `plank-replies-v1` fallback.
  They are now a single `KVCache` value type with one on-disk format, with
  `SessionStore` owning every path and the engine no longer touching the
  filesystem at all. The `Engine` trait shrinks to `get_kv` / `set_kv` /
  `warm_reset` / `warm_append` / `warm_sync`, and startup warming is one generic
  walk over the tier chain (the system prompt is now simply tier 0) instead of
  two separate phases.
- The on-disk cache format carries a version byte, so caches written by earlier
  builds are rebuilt once on first launch. They are pure caches; the cost is a
  single re-prefill.
- Remote runs no longer issue `POST /warm` at startup; the system-prompt prefill
  happens inside the first generation instead. No work is duplicated, but the
  first remote reply shows a longer prefill phase.
- The "system prompt changed" notice no longer includes a diff of what changed.
  It depended on a sidecar file that only the removed warm path wrote.

## [2.5.4] - 2026-07-25

Beta patch bump on the 2.5 series, landing the rest of the 2.6.0 work: session
branching, the native KV cache tier loop, and the Pi-parity quality-of-life
commands.

### Added

- **Session branching** (#65): sessions are now a tree rather than a line.
  `/tree` navigates and marks the active branch, `/fork [n]` branches from an
  earlier user prompt, and `/clone` duplicates the active branch. Existing
  linear sessions load unchanged and, while they stay linear, are written
  byte-identically to before.
- **Native KV cache tiers** (#64): `warm_tiers` walks the cache tiers
  most-stable-first, restoring the deepest still-valid checkpoint and
  prefilling only from the first fingerprint mismatch. The project-stable
  context (AGENTS.md/CLAUDE.md plus local MCP tool definitions) is checkpointed
  per project at `kvcache/<project-key>/project-<fp2>.kv` and shared across
  sessions; the volatile git/date context is prefill-only and never cached.
  Superseded per-project checkpoints are garbage-collected.
- **Session export** (#66): `/export [md|html] [path]` renders the transcript to
  Markdown or a self-contained HTML file.
- **Prompt templates** (#67): Markdown files in `~/.plank/templates` and
  `./.plank/templates` become `/name` commands with `{{var}}` interpolation.
  Built-in commands can never be shadowed.
- **External editor** (#68): Ctrl+G opens `$EDITOR` on the current prompt,
  suspending and restoring the TUI around it.
- **Word-wise prompt navigation** (#73): Alt/Ctrl + Left/Right move by word,
  Alt/Ctrl+Backspace and Alt/Ctrl+Delete kill by word, plus emacs-style
  Alt+B/F/D. Word boundaries are UTF-8 safe and treat all whitespace as
  separators.
- A joke local-inference invoice for `/usage` when running without a provider.
  Token counts stay real; only the billing framing is the gag.

### Fixed

- **Prefill progress double-counted the cached prefix** (#74): the engine
  reports the absolute prompt position, but the callback added the cached base
  to it again. Warm prefills therefore overshot the total, tripped the
  progress-bar headroom clause, and displayed cumulative numbers with inflated
  tok/s. The base is now the bar's floor and the subtrahend for throughput.
- **Warm prefill was discarded on the first question** (#64): tier text was
  tokenized verbatim at warm time but trimmed on the transcript round-trip the
  turn rebuilds its tokens from, so the KV common-prefix probe diverged at the
  first tier and re-prefilled the entire context. Tier text is now canonical.
- **`/clear` and `/new` left the old conversation on screen** (#72): the TUI
  output log is cleared and the banner re-rendered so the display matches the
  fresh session.

## [2.5.3] - 2026-07-25

Beta patch bump on the 2.5 series, landing the first wave of the 2.6.0 work.

### Added

- **Update-available detection** (#56): a best-effort, once-per-day check of the
  GitHub Releases API surfaces a non-intrusive hint when a newer plank exists.
  Offline-safe (silent on failure), cached under `~/.plank`, and disableable via
  the `update.check` setting.
- **Word/character-level diff highlighting** (#62): edit diffs now highlight only
  the changed spans within a line, pairing adjacent removed/added lines and
  falling back to full-line highlighting once the change ratio exceeds ~40%.
- **TUI animation subsystem** (#61): a shared 20 Hz clock drives glimmer, pulse,
  flash, a ping-pong Braille throbber, and a stall-fade, with a hard
  reduced-motion fallback (`ui.reducedMotion`, also in `/config`).
- **Startup context warming** (#63): the session-start context is prefilled into
  the KV so the first turn prefills only the question. The TUI input prompt now
  appears only once warming completes, behind an animated "warming cache" screen.

### Changed

- **ds4 engine transcript is token-primary** (#58): the token buffer is now the
  source of truth (C-parity append-only transcript), with text derived from
  tokens, replacing the text-primary reply-splice cache.
- **Hierarchical KV cache tier foundation** (#60): the session-start context is
  split into a project-stable tier and a session-volatile tier, with
  tier-fingerprint chaining and project-scoped checkpoint paths. (Native
  restore-loop wiring tracked in #64.)

## [2.5.2] - 2026-07-25

### Fixed

- **Prefill no longer re-feeds the whole conversation** (#57): the model-visible
  task list (#35) was injected as a `[user]` block right after the system
  prompt and rebuilt every turn, so any `task` add/update rewrote the tokens at
  the top of the prompt and broke the engine's KV common-prefix reuse — the
  entire conversation re-prefilled on the next turn (accidental O(turns²)). The
  rendered transcript is now strictly append-only, matching the C reference's
  token transcript: the task list rides in the `task` tool's own observations
  and a one-time re-injection after compaction, never mid-transcript.
- **KV reuse now spans every assistant turn, not just the last one**: the engine
  keeps the exact sampled token ids of every reply still in the transcript
  (retokenizing reply *text* does not reproduce the sampled ids — BPE
  segmentation is many-to-one) and splices each back in, so only the genuinely
  new suffix prefills each turn. The token history is persisted with the KV
  payload, so `/resume` and idle reclaim keep full prefix reuse.
- **TUI no longer hangs on fenced code blocks** (#59): streaming a ```code```
  block wedged the UI at 100% CPU because the markdown segment was
  re-highlighted on every token and `ratatui-markdown`'s tree-sitter
  highlighter recompiles its query per call. Markdown re-rendering is now
  throttled to ~10/second while streaming, with a guaranteed flush at each
  segment boundary; live syntax highlighting is preserved.

### Changed

- **Thinking text is now italic** as well as dim grey, in both the Ratatui TUI
  and the plain stdout renderer, so reasoning reads as background muttering
  distinct from the assistant's real output.

## [2.5.1] - 2026-07-24

### Added

- **MCP Streamable HTTP transport**: `.mcp.json` entries with a `"url"` (plus
  optional `"headers"`, e.g. an `Authorization` token) connect over Streamable
  HTTP — each JSON-RPC message is one POST answered with plain JSON or a short
  SSE stream, and a server-assigned `Mcp-Session-Id` is echoed on later
  requests. Stdio `"command"` servers work exactly as before.
- **Native macOS desktop notifications**: a turn that ran past
  `ui.notifyAfterSecs` (default 10) ends with a banner reading
  `'<prompt...>' finished` — the prompt as the bold headline, the tail of the
  answer as the body (`'...' interrupted` / "Task interrupted" for a
  user-aborted turn) — that persists until dismissed; the `ask` tool and
  awaiting-input also notify. Banners wear the host terminal's icon with
  plank's logo as the content image. `ui.notifications` picks when they fire:
  `always` (default), `unfocused` (only while the terminal window isn't
  focused, tracked via TUI focus events), or `never`; `/notify` toggles at
  runtime. Warp gets native OSC 777 agent notifications too.
- **Window title**: the terminal title shows `🪵 plank`, extended with the
  current prompt (`🪵 plank - fix the bug…`) while a turn runs.
- **Interactive `/config` editor** (#52): a TUI form (and
  `/config <section>.<key> <value>` from the prompt) over every settings key;
  changes write `./.plank/settings.json` and apply immediately. New keys since
  2.0.2: `ui.notifications`, `ui.notifyAfterSecs`, `ui.crtOff`, and the
  `tools.task` / `tools.agent` / `tools.planMode` gates.
- **Status-bar tips and tool flash**: rotating 💡 hints at the tail of the
  status bar (auto-hiding after 10 s); dispatched tools show as a transient
  `🔧 <names>` flash for 5 s; clipboard copies confirm with 📋.
- **Mouse copy**: click-to-copy fenced code blocks (`⧉ copy`), and
  content-anchored drag selection that survives scrolling and copies the full
  underlying text (code blocks verbatim, not soft-wrapped rows). The
  jump-to-bottom hint is clickable.
- **CRT power-off exit animation** on clean TUI exit, colors included
  (`ui.crtOff`, default on) (#54).
- **Web tools**: `google_search` is a client-side DuckDuckGo search;
  `visit_page` fetches pages through the embedded obscura headless browser
  (feature `use_obscura`, statically linked — no external binary) instead of
  curl. Web access asks for consent with an "Always allow" option; failures
  dump details to `~/.plank/errors.log`.
- **System-prompt cache-miss diagnostics**: a rebuild at launch explains why
  (cache missing / prompt changed) with a sanitized red/green diff snippet
  below the warm-up progress bar; `PLANK_DEBUG_SYSPROMPT` instruments the
  cache decisions.
- **`per_project_kv` cargo feature** (off by default): keys the system-prompt
  KV checkpoint by project directory (`sysprompt-<hash>.kv`) so per-project
  prompt inputs (AGENTS.md, local MCP config) don't invalidate other projects'
  snapshots.
- The single-instance error now names the PID holding the lock.
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
- Prefill runs in chunks (fixed at 256 tokens) so Ctrl-C interrupts a long
  prefill promptly; the interim `--prefill-chunk` flag was dropped.
- Tools the DS4 model wasn't trained on (`task`, `agent`, plan mode) are gated
  behind settings and off by default.
- The Homebrew formulas were renamed `plank` → `plank-agent` and
  `plank-beta` → `plank-agent-beta`.

### Fixed

- The turn-end notification headline sometimes showed the last tool result
  instead of the user's prompt (tool results are stored as user-role
  transcript messages and were not filtered out).
- KV caches (`sysprompt.kv`, session payloads) survive plank version changes;
  co-installed stable/beta versions no longer churn the checkpoint on every
  switch — the fingerprint and payload format-version already validate them.
- Sessions with no real activity no longer leave a resume point.
- The TUI no longer freezes on the web-access approval prompt.
- **Provider engine no longer aborts on an HTTP error.** A 4xx/5xx from an
  OpenAI/Anthropic-compatible provider used to propagate out as a fatal
  `EngineError` (crashing the plain-REPL / non-interactive / `-p` paths).
  Transient failures (HTTP 408/429/5xx and connection-setup drops) now retry
  with bounded, jittered exponential backoff (up to 5 attempts, ~250ms→4s,
  honoring `Retry-After`); auth/permission errors (401/403) fail fast with the
  provider's own error message instead of a bare `http status: N`.
- **Smoother remote token streaming.** The provider request now asks for
  `Accept-Encoding: identity`; the default gzip stream was decompressed through
  `flate2`'s fixed 32 KiB buffer and arrived in chunky clumps. Identity encoding
  streams one SSE frame at a time.
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

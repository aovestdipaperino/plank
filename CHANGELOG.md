# Changelog

All notable changes to plank are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [3.2.0] - 2026-08-25

### Added

- **Per-model phase times in the exit message.** The session summary used to end
  with one peak prefill and one peak generation rate. A peak is one lucky pass
  and says nothing about where the session actually went, so it now reports, for
  each model that ran, how long that model spent prefilling and generating with
  the session average rate beside each, plus how long it spent running tools:

  ```
  avg deepseek-v4-flash  prefill 12.3s (1420.5 tok/s)  ·  generation 45.2s (38.1 tok/s)  ·  tools 8.4s
  ```

  With more than one model the rows are labeled and aligned with the per-engine
  token rows below them. Tool time is charged at the single dispatch chokepoint
  every front-end shares and attributed to whichever engine drove the turn — the
  alt engine during a sidechain, same rule as the token tally. Online providers
  report neither a token count nor a local rate, so a provider-only session's
  exit message is unchanged.

- **The think segment draws the router.** Two braille cells beside the reasoning
  level stand in for the `MoE` expert routing, re-rolled on every decoded token.
  The real selection never leaves the GPU on the Metal path, so the display is
  derived from the token id rather than read from the engine: it is honest about
  sparsity (a few of many), about routing changing per token, and about the same
  token lighting the same experts — and says nothing about *which* experts. Two
  columns wide, exactly like the 🧠 it replaces, so the footer never reflows.

- **The reasoning level is colored by temperature** — red for `max`, white for
  `med`, blue for `low`, grey for `off` — so a three-column segment is legible
  without reading the word.

- **Greedy chain decode on Metal**, via the ds4 engine's new
  `ds4_session_eval_chain_greedy`. At temperature 0 plank decodes a run of
  argmax tokens with the next token id kept on-device, removing the per-token
  host round-trip (GPU sync, 517 KiB logits readback, CPU argmax). Output is
  bit-identical to the previous path, verified by md5 over the reply.

  **Off on M5.** Upstream measured +1.75% on an M3 Ultra, but on an M5 Max it
  is ~1.3% *slower* — it lost all three interleaved pairs — so the chain is
  enabled everywhere except M5, matching where the fork gates its other Metal
  decode work. `PLANK_GREEDY_CHAIN=1` forces it on to re-measure after a kernel
  change; `DS4_DISABLE_GREEDY_CHAIN=1` turns it off everywhere.

  The engine declines the chain for any session holding a `--dspark`/MTP
  support model, so the two are mutually exclusive.

### Changed

- **The live decode rate is measured from the first token out.** It was anchored
  at the start of the `generate` call, so on a long prompt it divided the token
  count by decode time *plus* prefill *plus* time-to-first-token: the footer
  opened far below the real rate and only crept toward it as the pass ran.

- **The `--dspark` footer segment now reads `1.5t/step`, not `1.5x`.** It was
  always mean tokens committed per speculative step, which is not a wall-clock
  speedup: on Metal it reads well above 1.0 on runs that decode *slower* than
  plain decode, because a batched verify costs more per token than plain decode
  does. The accompanying percentage is likewise a lower bound — plank cannot
  see how many tokens the C actually drafted, only the block size it was
  allowed. `FINDINGS.md` has the measurements.

- **Bumped the `refs/ds4` submodule** to `ivanfioravanti/ds4-metal` for the
  chain-decode API and the pre-M5 Metal decode work.


## [3.1.0] - 2026-08-22

### Added

- **`Ctrl+W` wipes every saved session** from the `/resume` picker. Same
  two-press arming as the single-session `Ctrl+X`, but the confirmation names
  the count — `Ctrl+W again to delete ALL 362 saved sessions` — because this one
  is not undoable. Transcripts and their KV payloads go; the shared
  system-prompt and project checkpoints stay, since rebuilding those would cost
  a full prefill on the next launch for nothing.

### Changed

- **`--dspark` defaults the sampling temperature to 0.** Speculative decoding
  only engages at temperature 0, so asking for DSpark and leaving the 0.6
  default in place silently decoded target-only. An explicit `--temp` still
  wins, in either flag order.

## [3.0.4] - 2026-08-11

### Added

- **`/resume` is a picker now.** In the TUI, a bare `/resume` opens a panel over
  every saved session instead of printing ten numbered lines: type to filter on
  name, title, tag or last prompt, arrow through the results, Space to preview a
  session's last turns, `Ctrl+R` to rename, `Ctrl+X` twice to delete, Enter to
  resume. The plain-stdout path keeps the numbered listing, and `/resume <name>`
  still resumes directly from either front end.

- **`/retitle`** re-derives every saved session's title from its transcript. It
  splices the title record in place rather than re-saving, so the `last used`
  stamps the picker sorts on survive the pass.

### Fixed

- **Session titles no longer name the project instead of the conversation.** A
  title was taken from the transcript's first user message, and the first user
  message is usually the session-start context plank injects itself — so every
  session in a repo was titled `Agent instructions: --- # From: …/CLAUDE.md` and
  the new picker was a wall of identical rows. Titles now come from the first
  turn a human actually typed. Existing sessions keep their stored title until
  `/retitle` is run.

## [3.0.3] - 2026-08-11

### Added

- **plank can read your screenshots.** Image pasting is on by default now, and
  paired with the [`ocr-mcp`](https://github.com/aovestdipaperino/ocr-mcp)
  server the model can act on what you paste: it calls `transcribe_image` on the
  cached path and gets the text back. Transcription runs on your own machine
  against a 0.9B OCR model, so no image leaves the laptop and there is no API
  key. Install it with `brew install llama.cpp && cargo install ocr-mcp` and
  register it in `.mcp.json`; the guide covers fetching the weights.

  The feature had been compiled out behind `--features images` because a pasted
  image reached the model as a path nothing could open, which made the whole
  thing a tease. There is now a tool that opens it.

### Changed

- **Pasted images are cached byte-for-byte.** plank used to downsample every PNG
  to 2000px on its long edge, a rule inherited from an image-upload API limit
  that plank does not have: the ds4 engine is text-only and never uploads pixels
  anywhere. The resampling only discarded the pixel density and DPI metadata that
  an OCR tool then needs. Note that `~/.plank/image-cache/` is bounded by file
  count rather than bytes, so it now grows larger for the same number of images.

## [3.0.2] - 2026-08-11

### Added

- **Plugins load.** A plugin is one directory bundling skills, agents,
  templates, hooks, an `.mcp.json` and a `settings.json`, contributed to a
  session as a unit. plank picks them up from `~/.plank/plugins/dev/`, from
  `./.plank/plugins/`, and from a repeatable `--plugin-dir <path>` that lasts
  for the session — and it reads both its own spelling (`.plank-plugin/plugin.json`,
  `templates/`, `hooks.json`) and Claude Code's (`.claude-plugin/plugin.json`,
  `commands/`, `hooks/hooks.json`). A directory with no manifest but with
  recognizable components still loads, named after itself.

- **A plugin contribution is always addressable as `<plugin>:<name>`,** and keeps
  the bare `<name>` only when nothing else claims it, so your own skills, agents
  and templates never lose theirs to a plugin. Two plugins offering the same name
  both keep only the namespaced form. MCP servers are the exception: the
  separator is `-`, because a server name is embedded in `mcp__<server>__<tool>`
  and split at the first `__` — and a plugin server name containing `__` is
  rejected outright.

- **Plugin settings sit below yours.** The order is defaults, then plugin
  settings in load order, then `~/.plank/settings.json`, then
  `./.plank/settings.json`, then the environment and the command line, so a
  plugin can never override a setting you wrote. A plugin that sets a `safety.*`
  key still beats the built-in default, and plank warns by name when one does.
  Hooks are additive and all run, ordered `~/.plank`, then plugins, then the
  project file.

- **`/plugins`** lists what loaded, where each plugin came from, what it
  contributes, and every warning. Nothing about a malformed plugin is fatal: a
  bad manifest or component demotes itself and says so. There is no installer and
  no marketplace yet — you place the directory yourself.

- **`/goal [--max <n>] <objective>`** hands control back to the model turn after
  turn until the objective is settled, instead of you typing "continue". After
  each turn plank asks for a one-line adjudication (`ATTAINED`, `UNATTAINABLE`,
  `NEEDS_USER` or `CONTINUE`) and stops on the first terminal verdict, printing a
  closing line with the reason and the iteration count. Anything unparseable
  reads as `CONTINUE`, so a parse miss costs an iteration rather than falsely
  declaring success, and the cap (20 by default) bounds the loop. Ctrl-C in the
  plain REPL and Esc in the TUI end the goal, not just the turn in flight.

- **The sandbox can grant `~/.plank` writes on request,** per session, for the
  cases where a tool legitimately needs to write into plank's own home.

## [3.0.0] - 2026-08-10

### Added

- **A session is named the moment it starts, and the name is on screen the whole
  time.** The memorable `adjective-celebrity` name used to be minted at save
  time, so until you quit there was nothing to call the conversation you were
  having. It is now minted at session start (and again on `/new`), and the TUI
  floats it at the right end of the rule above the prompt — so the name a
  transcript will be saved under is visible from the first frame.

- **`/rename <name>`** changes the name later saves use. Nothing already on disk
  is touched: a session saved before the rename stays resumable under its old
  name, so the next save is a logical copy rather than a move. Names are
  validated rather than sanitized (letters, digits, `-`, `_`, `.`), so the name
  you see is the name you typed, and a name already on disk asks before taking
  it — the `ask` panel in the TUI, a `[y/N]` prompt on the plain path, declining
  wherever there is nobody to ask. `/save` already wrote the session without
  quitting; it now reports the name rather than eight characters of it.

- **`/kvcache`** shows the KV cache as the tree it actually is. Every persisted
  snapshot now carries a JSON metadata sidecar recording what it is, the
  fingerprint of the snapshot it extends, the model and reasoning level behind
  it, its size, its hit count, when it was last used, and whether it is pinned.
  The pane draws that tree with `↑↓` to move, `←→` to fold, `p` to pin, `d` to
  delete and `g` to sweep; the plain REPL prints the same tree and takes
  `/kvcache pin|unpin|rm|gc` by fingerprint prefix.

  The metadata is strictly advisory. The signature inside a snapshot's own body
  remains the only thing that decides whether its bytes may be restored, so a
  missing or corrupt sidecar costs display quality and never correctness.

- **`/open [path]`** edits an existing file in the built-in editor: `Ctrl-S`
  saves, `Esc` discards. Bare `/open` reopens the last file a tool call
  edited this session. TUI-only, and it never creates a file.

### Changed

- **The `/resume` picker leads with the session name.** Each row is the name and
  age, then the title, then the last prompt, and the help line says to type the
  name (a number from the list still works). The name is what you type to pick a
  session and, now that it exists from session start, what you were looking at
  on the prompt rule the whole time it ran.

- **Replayed history leaves out the session-start context.** Resuming a session
  used to replay the scaffolding plank injects for the model — agent
  instructions, persistent memory, the sub-agent roster, git status, the date —
  which on a short session was most of what you saw. Those blocks are user turns
  as far as the model is concerned but nobody typed them, so they no longer
  count as turns for the history window and are never rendered. A session that
  holds nothing else replays nothing at all.

- **The KV cache expires on age and is capped on size**, instead of keeping
  only the current fingerprints. The old garbage collector kept exactly the
  live system-prompt and project checkpoints and deleted every sibling, so
  switching model or reasoning level and back paid a full system-prompt
  re-prefill each way. Snapshots now expire on time since last use
  (`kvcache.ttlSessionDays`, default 14; `kvcache.ttlTierDays`, default 30),
  and if the survivors still exceed `kvcache.maxBytes` (default 20 GB) the
  least-recently-used are evicted until they fit. Pinned entries, the chain the
  current launch is using, and any snapshot something newer still builds on are
  exempt, and when everything remaining is protected plank stays over budget
  rather than deleting something it should not. Several system prompts now
  coexist for as long as they are all in use.

- **Every KV body is now a `.kv_raw` file**, so `.kv` means "session
  transcript" and nothing else. The two used to share an extension, which
  forced the collector to filter by filename prefix to avoid eating a
  transcript.

- **`/strip`'s description was wrong** in the guide and is now correct: it drops
  a session's KV payload to reclaim disk and leaves the transcript untouched.
  It never trimmed turns.

### Migration

On the first launch after upgrading, plank deletes the old-format KV snapshots
once and reports how much it reclaimed. **Session transcripts are not touched**,
so every saved conversation still loads; each pays one re-prefill the next time
you open it, and the shared system-prompt and project snapshots rebuild on
demand. If your cache had grown large, expect that first launch to hand back
most of it. The 20 GB ceiling then applies to what you rebuild afterwards.

### Fixed

- **A dropped network no longer freezes the turn with no way out.** Losing the
  link mid-generation — Wi-Fi off, sleep, a NAT rebind — used to hang plank
  indefinitely, with Ctrl-C and Esc doing nothing. Two faults compounded. The
  streaming HTTP agent set no timeouts at all (every `ureq` timeout defaults to
  `None`, and it was the one agent in the tree that did not override them), and
  a silent drop produces no RST or FIN: with the request already sent there is
  no unacked data for TCP to retransmit, so no kernel timeout ever fires and
  the socket sits established but black-holed forever. Meanwhile cancellation
  was polled *inside* the SSE callback, which runs per arriving event — so zero
  bytes meant zero interrupt checks, and the one situation that needed
  cancelling was the one where it could not work.

  The response body now reads on its own thread feeding a channel, and the turn
  polls it with a timeout, so the interrupt flag is checked on a **clock rather
  than on data arrival** — Ctrl-C lands within a quarter second even against a
  dead socket. Ninety seconds of silence is reported as a stalled stream
  instead of a hang, which is sound because both providers keepalive their
  streams. Connect and header timeouts cover a drop *before* the stream starts.
  Both remote engines are fixed. As a last resort, a second Ctrl-C on an
  interrupt the worker has not acknowledged within two seconds force-quits,
  with the status bar saying so.

### Changed

- **The status bar task counter reads `✓ Tasks: 2/5`.** The bare `✓ 2/5` did
  not say what it was counting.

- **The window title shows `❓ waiting for you...` while the `ask` tool has the
  turn open**, so a backgrounded window says the turn is waiting on *you*
  rather than on the model. The title it displaced — normally the `🚀` of a
  running turn — comes back however the question ends, including a declined or
  interrupted one.

### Added

- **DSpark speculative decoding, behind `--dspark`.** DeepSeek's auxiliary
  draft checkpoint for V4 Flash reads hidden states from the main model and
  proposes up to five tokens ahead; the target model verifies them and commits
  only the prefix it agrees with, so one verification pass can advance the
  stream by several tokens. Off by default. `--dspark-confidence F` sets the
  pruning threshold, `--dspark-strict` loads the drafter but keeps target-only
  decode, and sampled decoding never uses proposals — verification is argmax,
  so speculation only applies at `--temp 0`.

  The support model does not need `--mtp`: it resolves to
  `~/.plank/ds4flash.dspark.gguf` and is offered for download (~5.6 GB) through
  the same prompt, resume and progress path as the main model. An explicit
  `--mtp` still wins, since that is also how a legacy one-stage MTP drafter is
  supplied.

  Worth knowing before turning it on: the payoff moved a long way during
  development. Through the M5 decode-fusion work it was a consistent *loss* on
  an M5 Max — 0.71× on generation, because verification and replay cost more
  than the target passes they saved. Upstream then pipelined the Metal verifier
  (`42033ee`), and the same measurement flipped to **1.19×**, with wall clock
  agreeing at 0.81×. Both figures are one machine, one quant, one engine
  commit; treat any DSpark number as attached to a specific engine SHA.

- **The exit message reports the session's peak prefill and generation rates**
  per model, alongside the token totals:

  ```
  peak DeepSeek V4 Flash  prefill 167.1 tok/s  ·  generation 16.8 tok/s
  ```

  Session-scoped on purpose — nothing is written to disk. A peak from last week
  was measured on a different engine build, a different context length and a
  cooler machine, so comparing against it silently is worse than not comparing.
  Both rates exclude the first two seconds of their phase, which is where the
  bias lives: the first decode token pays one-time GPU costs, and a short pass
  divides by an elapsed dominated by fixed setup. A KV-cache restore is not
  counted as prefill, however fast it looks.

- **A third screensaver face: two minions.** `ui.screensaverFace` gains
  `minions` alongside `matrix` and `starfield`, and `random` now draws from all
  three: a pair of minions on the shore of a night lake, who walk, blink, elbow
  each other and fall about laughing, reflected in the water underneath.
  `/minions` opens the same screen on demand, gated by `ui.easterEggs` like the
  games; `↑`/`↓` and the wheel set the pace, `r` strips the scene back to the
  two of them, `t` lays them over live model output. Being weather rather than
  a game it is never parked, scores nothing and leaves no line in the
  scrollback.

  **The whole animation is 218 bytes of the binary.** Six poses of ASCII
  paint-by-numbers live in `src/resources/minions.txt` — 1 440 cells, readable
  and diffable — and that file does not ship: `build.rs` packs it with a
  three-op coder (same cell as the pose before, same ink again, something
  already drawn up to 256 cells back, anything else a literal) into `OUT_DIR`,
  a seventh of the size, and a `const` assertion fails the build if that ever
  stops being true. Nothing else about the scene is stored: the walk, the bob,
  the nudge, the laughter, the stars, the ripples and the reflection all come
  out of one seeded generator and a clock.

  It is also as close to transparency as a terminal gets, in three parts: a
  cell too faint to matter is not drawn at all, `█ ▓ ▒ ░` make the coverage
  ramp into an alpha channel that fading walks down, and glyphs that are shapes
  rather than fills fade by colour instead. Over live output that means the
  model's text survives where the minions are thin, rather than being punched
  out by cells too dark to see.

- **Git worktrees.** `EnterWorktree` / `ExitWorktree` move the session into an
  isolated second checkout of the repository — `.plank/worktrees/<name>`, on
  branch `worktree-<name>` — so a large or speculative change never lands in the
  tree you have open. Entering switches every tool's working directory; leaving
  takes `keep` (worktree and branch stay, for review or merge) or `remove`.
  Everything destructive is fail-closed. A `remove` that would take uncommitted
  files or commits reachable from no other ref is refused, and the refusal names
  what would be lost; so is a `remove` whose state git could not be asked about,
  because not knowing is not the same as knowing there is nothing there.
  `--worktree NAME` and `--worktree-pr N` start a whole session in one, and go
  further than the tool does: the worktree becomes the session's project, so the
  hooks, sub-agent definitions, and settings that apply are the ones found there.
  `isolation: worktree` on a sub-agent definition — or `worktree.isolateAgents`
  for all of them — gives each run its own throwaway worktree, which is what
  makes fanning several agents over the same files safe rather than a race; a
  clean one is removed when the run ends, one holding work is kept and its path
  reported back so it can be merged. Throwaway worktrees leaked by a killed
  process are swept at startup, and only ever ones whose names have the exact
  shape plank itself generates, so a worktree you named is never a candidate.
  `WorktreeCreate` / `WorktreeRemove` hooks replace the git backend outright, for
  a VCS that is not git. Tuned by `worktree.sparsePaths` and
  `worktree.symlinkDirectories`, plus a `.worktreeinclude` file naming gitignored
  files (a `.env`, a local build config) to carry into each new worktree — copied
  only when they are both listed there and genuinely ignored.

- **The status bar's brain blinks while the local engine works.** The 🧠 already
  in the think segment pulses for the whole span of a local prefill or
  generation, and sits still otherwise — so which engine is actually running is
  visible rather than inferred. It answers a question nothing on screen could:
  a `provider: local` sub-agent under a remote main agent looks identical to one
  running on the provider, since the engine-origin segment reports the session's
  engine and never changes. The blink dims rather than blanks, so the bar never
  changes width, and it is driven by a guard held across the pass, so an early
  return or a panic cannot leave it pulsing forever.

- **A third cross-engine sub-agent test: remote main, remote sub-agent, one key.**
  `multi-provider-tests/remote-remote/` runs the main agent on one hosted model
  and its sub-agent on another at the *same* endpoint with the *same* credential
  — the definitions differ only in the model name. That is the case the existing
  two directions cannot reach: when wire format, base URL, and key are all
  identical, a bug that collapses the two engines into one (a client cached on
  the base URL, a key lookup that memoises its first hit, a restore that compares
  the wrong field) shows up only as the sub-agent quietly answering as the
  parent's model. It loads no local model, so unlike the other two it starts
  instantly and runs anywhere. `/usage` reporting one row per model is the check
  that does not rely on a model's own account of which model it is.

### Changed

- **The matrix rain is the default screensaver, and which face you get is now a
  setting.** `ui.screensaverFace` takes `matrix` (the default), `starfield`, or
  `random`. The screensaver used to flip a coin between the two every time it
  opened, which is a nice surprise and a poor default — if you like one of them
  you had no way to say so. `random` keeps the old behaviour for anyone who
  wants it. `ui.screensaver` still says *when* (`1m`/`2m`/`5m`/`never`); the two
  are separate because wanting the rain at five minutes should not require
  spelling that as one combined value. Both cycle in `/config`.


- **Named subagents are dispatched as `/subagent:<name>`, not `/subagent <name>`.**
  The name now rides on the command token instead of being the first word of the
  argument. Two things follow from that. A task whose first word happens to match
  a definition — `/subagent reviewer notes are stale`, say — is no longer silently
  reinterpreted as that persona; the whole argument is the task. And because the
  name is explicit, one that does not resolve is an error naming what *is*
  available, rather than a quiet fallback to the general-purpose subagent that
  looks like it worked. The TUI colours the `:<name>` green when it resolves and
  red when it does not, so a typo shows while you type rather than after you press
  Enter; a half-typed `/subagent:` stays unhighlighted, like any other incomplete
  command. Bare `/subagent <task>` is unchanged.

- **The status bar is two rows.** Row one is the working directory and the git
  branch and nothing else — the answer to "which tree am I in" now holds still
  instead of being shoved around by whatever the model is doing. Row two carries
  everything volatile in the order the single row used: engine origin, think
  level, context gauge, progress or state, task counter, power suffix, remote
  marker, and the tail notification slot. The engine origin moved rows with the
  rest; it used to sit beside the path.
- **The sub-agent roster moved out of the system prompt.** It was the `agent`
  tool's `name` enum plus a per-definition description listing, inside the
  fingerprinted prefix — so editing any definition invalidated `sysprompt.kv`
  and cost a full reprefill of a 1M-token prompt. The roster is now part of the
  Tier-2 project context, which `stable_hash` already keys, so an edit rebuilds
  that far smaller cache and leaves the expensive prefix intact. `name` is a
  plain string in both prompt shapes, and both are now independent of what is on
  disk — which is also what keeps the C-parity fixtures valid, since they lock
  exactly the no-roster bytes. Dispatch is unchanged: an unmatched name still
  runs a general-purpose sub-agent with a `note:` line, and the same visibility
  gates (`auto`, `agents.autoRoute`, a present API-key variable) now filter the
  context roster instead of the schema.

### Added

- **Cross-engine sub-agents, including a local one under a remote main agent.** A
  definition in `~/.plank/agents/*.md` can name the engine its sidechain runs on:
  `provider:` plus `model:` (with optional `base-url:`, `ctx:`, `api-key-env:`)
  for a hosted model, or `provider: local` for the local ds4 engine. When the
  main agent is a provider and a definition asks for `local`, plank loads the
  local model alongside the provider one so the sidechain has something to run
  on — at startup, so a missing model or insufficient RAM fails before the
  prompt rather than mid-turn, and only when a definition actually asks, since it
  costs the full ~82 GB residency. Under a local main agent `provider: local` is
  not an override at all and the sidechain runs on the parent engine.

  Note `provider: local` and *omitting* `provider:` are deliberately different:
  omitting it means "whatever the parent is", which under `--provider` is the
  remote model. Only the explicit spelling triggers the extra load.

## [2.8.2] - 2026-08-06

Beta. The remote-control feature became usable from a browser: `/remote-control`
(`/rc`) starts the server from inside a session and prints a one-click link, and
the bundled web client grew into a real front-end. The `--control*` launch flags
are gone — see Removed.

### Added

- **A `low` reasoning level (experimental).** `/think low` and `--think-low` sit
  below `medium`, asking the model in a preamble to keep its deliberation short.
  This is a *prompt* level, not an engine level: the ds4 engine has no state
  below `DS4_THINK_HIGH`, so `low` is `HIGH` at the FFI boundary plus
  `THINK_LOW_PREFIX` prepended ahead of the system prompt — the same mechanism
  `max` uses, pulling the other way. Unlike every other model-facing string in
  plank it has **no C counterpart**, so `tests/c_parity.rs` cannot check it and
  the model was not trained on it; treat its effect as unverified until measured
  against real traces. Switching in or out of it costs one re-prefill (the prompt
  prefix changes, so the token transcript and KV are dropped), which `off` ↔
  `medium` still does not.

- **Engine origin in the status bar.** A new segment after the dir prefix says
  where inference is running: the provider or remote host's domain
  (`api.anthropic.com`, `localhost`) or `(local)` for the on-device DS4 engine,
  so a footer never leaves it ambiguous whether a turn is going over the wire.
- **Context-window discovery for `--provider anthropic`.** Without an explicit
  `-c/--ctx`, plank asks `GET /v1/models/{model}` for the model's
  `max_input_tokens` and sizes the ctx gauge from that instead of the local
  model's 1M default. Best-effort: any failure keeps the configured value, and an
  explicit `-c` is never overridden. The OpenAI path has no such field to read,
  so it is unchanged.
- **`/remote-control` (`/rc`) starts and stops the remote-control server at
  runtime**, from inside a running TUI session. Bare `/rc` toggles; `/rc on` and
  `/rc off` are explicit and case-insensitive. Turning it on always binds an
  ephemeral loopback port (never a fixed one), and a new activation mints a
  fresh token, printing `http://127.0.0.1:PORT/?t=TOKEN` plus an `ssh -L`
  tunnel hint. Opening that link auto-connects the bundled web client and
  claims control immediately, since typing `/rc` is the operator's own
  consent — no token to paste, no button to press. That link is the only way in,
  so the client's URL and token fields are gone: the token is printed nowhere
  else, and a page opened without `?t=` says so instead of offering controls
  that cannot authenticate. `/rc off`
  tells connected clients, shuts the listener down, and the token dies with it,
  so a stale link is refused; a later `/rc` mints a new port and token. The
  token still lands in browser history and any `Referer` header, an accepted
  trade for one-click attach on a loopback-only listener, not a claim that the
  link is secret.
- **The bundled web client became a real front-end.** It wears plank's own dark
  theme and logo, renders the model's markdown as it streams (keeping the
  model's line breaks, which strict markdown would fold), and shows the same
  directory / branch / engine-origin / reasoning-level segments as the TUI
  footer beside a live context gauge. The prompt is a three-row box with Enter
  to send, Shift+Enter for a new line, and ↑/↓ prompt history; `send` is
  enabled only with something to send and `stop` only while a turn is running.
  There is no frame-kind selector: everything goes as a prompt and the agent's
  own slash dispatcher routes it, so `/btw` and friends work from the browser
  without the page knowing what any of them are.
- **End-of-turn notification for attached clients.** The desktop notification
  only reaches whoever is at the machine plank runs on. A finished turn now
  crosses the wire too: a browser notification where permission allows, an
  in-page banner that needs none, a blip, and a tab-title flash; the terminal
  client prints the line with a BEL. Gated by the same `ui.notifyAfterSecs`
  threshold as the local one, so a turn shorter than it stays quiet everywhere.

- **A dropped connection is unmistakable in the web client.** `disconnected
  (1006)` renders bold in the error colour rather than as grey small print, and
  the prompt is disabled with it: nothing typed after the socket closes reaches
  the agent, and there is no reconnect to wait for, since the token died with
  the server. The placeholder says to run `/rc` again for a new link.

### Fixed

- **`/clear` now resets attached remote clients.** It replaced the session and
  cleared the local log, both local-only, so a browser kept showing a session
  that no longer existed — and the bus still held the pre-clear scrollback, so
  a client attaching afterwards was replayed the transcript that had just been
  cleared. A session reset is now an event on the bus: clients clear, and the
  scrollback goes with it. `/switch` and `/resume` send it too.
- **A turn's end reaches remote status.** Status frames came only from engine
  callbacks during a turn, so the last thing a remote client ever saw was
  `generating`: its context gauge froze and anything keyed off "a turn is
  running" stayed stuck on. A turn now publishes an idle snapshot when it ends.

### Removed

- **The `--control`, `--control-token`, `--control-allow`, `--control-origin`,
  and `--control-queue-max` launch flags.** Remote control now starts only from
  `/rc` inside a running session (see above); there is no longer a way to bring
  a session up already listening from the command line. **This is a breaking
  change** for any launcher or script passing these flags — they are no longer
  recognized. Only the *overrides* went with `--control-origin`/
  `--control-queue-max`: the browser `Origin` allow-list keeps its
  default-deny policy (loopback only) but can no longer be extended, and the
  per-client outbound queue keeps its 1 MiB default but can no longer be
  resized. The headless
  (`--non-interactive`) and piped plain-REPL remote-drive paths were also
  deleted: they were unreachable now that starting the server requires typing
  a slash command in the full-screen TUI, so `/rc` is TUI-only and a piped or
  headless plank cannot be remote-driven.

## [2.8.0] - 2026-08-03

Stable release: the 2.7 beta series promoted, plus the compaction and shell-escape
work below. Everything listed under 2.7.1 through 2.7.9 is now on the stable
channel (`brew install aovestdipaperino/tap/plank-agent`), most notably the
tool-call parsing and system-prompt tokenization fixes in 2.7.8 and 2.7.9.

### Added

- **`!` now feeds its result to the model; `!!` keeps it private.** The old `!`
  behavior (run a shell command, show the output, record nothing) moved to `!!`.
  A single `!` runs the command the same way and then records the command and its
  output in the transcript as one caveated user message, so the model has it as
  history on your next prompt without a turn being spent on it. Output is capped
  at 200 lines / 16 KB, and `<`/`&` are escaped so command output cannot forge
  the framing.
- **`/compact [instructions]`** steers a single compaction pass
  (`/compact keep the failing test cases verbatim`). The argument is added to the
  standing eight-section summary contract rather than replacing it, and sits above
  the closing no-tools instruction so it cannot displace it. Automatic compaction
  sends a byte-identical prompt to before.
- **Compaction progress in the status bar.** While a pass runs, the
  throbber/spinner-verb line below the output is replaced by a flashing
  `compacting` plus an `▰▱` bar and percentage, driven by real prefill progress
  for the bulk of the bar and the summary's length for the tail. The window title
  reads `🗑️ compacting...` for the duration and is restored afterwards, including
  on an interrupted or failed pass.
- **The status footer shows the reasoning level**, as a `🧠 med` segment just
  before the ctx gauge. The level changes how every turn is generated, so it
  keeps a permanent slot rather than appearing only when off the default. The
  three names are abbreviated to a fixed three columns (`off`, `med`, `max`) so
  switching level never shifts the rest of the footer sideways; `/think med`
  parses too, since that is the spelling on screen.

### Fixed

- **A resumed session re-prefilled its whole conversation.** The KV payload
  fingerprint covered the model, system prompt, and rendered transcript, but not
  the reasoning level or the trusted-prefix length — both of which change the
  *tokens* a prompt prefills to while leaving every byte of that text identical.
  A payload written before either changed passed the staleness gate and was
  restored over a KV it did not match. Both are now part of the fingerprint.
- **A repeated `tool_calls` wrapper opener became a phantom tool.** A stanza that
  opens the wrapper twice had the second opener read as an element name, so
  `tool_calls` itself was dispatched as a tool. The wrapper openers are now
  skipped, and the three structural element names can never be named as tools.
- **Compaction hooks fired on only one front-end.** `PreCompact` and
  `PostCompact` ran on the plain-REPL path but not the TUI path, so a hook
  configured by a TUI user — the default front-end on a TTY — silently never ran.
  Both orchestrators now dispatch both events through one shared implementation.
- **A compaction that produced no usable summary destroyed the transcript**,
  replacing it with an empty summary plus the verbatim tail. A pass that comes
  back empty (including a reply that is only a discarded `<analysis>` block) is
  now treated as a failure: the conversation is left as it was, `PostCompact`
  does not fire, and the turn is abandoned rather than continuing on a context
  that was never reclaimed.
### Changed

- **Releases are arm64 only.** Intel Macs cannot run the Metal backend and none
  has the unified memory to hold the model, so the x86_64 build could only ever
  ship the engine-less stub. The bottle and binary tarball are no longer
  produced, and the Homebrew formula requires `arm64` up front instead of
  falling back to a source build that would spend twenty minutes producing that
  same stub.

## [2.7.9] - 2026-08-03

Beta channel on the 2.7 series.

### Fixed

- **A tool call is judged when it closes, not when it opens.** Rejecting a
  stanza the moment its opening marker appeared inside `<think>` was too eager:
  a model reasoning about DSML syntax writes an opening marker mid-thought and
  often closes the thinking block before emitting the real call. The recorded
  repro shows exactly that — a correct, post-`</think>` `edit` call thrown away,
  and the model told to stop calling tools inside thinking, which it had not
  done. It rewrote correct markup and looped. An opening marker is now only a
  candidate; the verdict lands at the stop token, against the thinking state at
  that instant. `</think>` is recognized inside an open stanza too, except
  within a parameter value, where it is payload a `write` or `edit` may
  legitimately contain.

- **Mentioning the markup while thinking no longer costs the turn.** Any
  DSML-shaped bytes inside `<think>` used to raise the placement prohibition at
  stream end, even with no stanza and no stop token — so a model recalling its
  own tool-call format got a tool error back for it.

- **The tool name written as its own element is accepted.** The weights emit
  `<｜DSML｜edit>` in place of `<｜DSML｜invoke name="edit">`, closing with
  `</｜DSML｜invoke>` — their own tell that an invoke was meant. plank already
  tolerated the identical rewrite one level down, for parameters; the invoke
  form was rejected five times in one recorded session, with no recovery. Which
  shorthand a bare element is depends only on whether an invoke is open: before
  one it names the tool, inside one a parameter. Both stay narrow — DSML marker
  present, element name a plain identifier, no `name` attribute — and a name
  that is not a real tool fails at dispatch, by name, which the model can act
  on. The renderer learned both forms, so a shorthand call draws its banner
  instead of running invisibly.

- **The system prompt's tool instructions reach the model as the token they were
  trained as.** `｜DSML｜` is an entry in the model's own vocabulary, but a
  `system` message is tokenized as plain text, so every marker in the tools
  prompt arrived as spelled-out BPE pieces — and a spelled-out marker is what
  the model then reproduces letter by letter and occasionally corrupts
  (`<｜DSinvoke name="bash">`, and the `SSML` misspelling). The built-in prompt
  is now tokenized as rendered chat, as the C reference has always done, which
  turns those 16 markers into single tokens. MCP tool schemas and `-sys` text
  are deliberately left as plain content: both are third-party text, and as
  control tokens either could forge a turn boundary.

### Added

- **`/think off|medium|max` selects the reasoning level**, and `--think-max`
  now means what it says. The engine has always had three levels — the third
  adds a reasoning-effort preamble ahead of the system prompt — but plank
  collapsed the top two and never reached it. The preamble is the reference
  engine's own text, byte-for-byte. It wants at least a 384K context, and below
  that the level is refused rather than silently downgraded, so a request that
  cannot take effect says so.

## [2.7.8] - 2026-08-02

Beta channel on the 2.7 series.

### Fixed

- **Tool calls the updated model weights emit are parsed again.** Two syntax
  shapes appeared after the weights were updated, and each one cost the whole
  turn. The stanza opener now arrives as `<｜DSML｜tool_calls｜>` — the same
  optional trailing bar the closing tags have always tolerated, now on the
  opener. No opener form matched it, so the stanza never opened: the call
  streamed out as prose and the model was told only "DSML markup outside a
  valid tool_calls block", with nothing to identify which part of its syntax
  was wrong.

- **A parameter written as its own element is accepted.** The weights emit
  `<｜DSML｜command string="true">` in place of
  `<｜DSML｜parameter name="command" string="true">`, closed with
  `</｜DSML｜invoke>`. The recorded repro shows the model unable to work
  backwards from the resulting error: it blamed the marker spelling twice and
  then emitted a tool call inside `<think>`. The shorthand is now accepted, but
  narrowly — only inside an already-open invoke, only for a tag whose element
  name is a plain identifier, and only when it carries no `name` attribute. A
  canonical parameter keeps its strict terminator, so a `write` payload that
  itself contains `</｜DSML｜invoke>` is still never truncated.

### Changed

- **The model download fetches the official `-0731` Flash build** (~87 GB), the
  non-preview DeepSeek V4 Flash of 2026-07-31. The architecture is unchanged
  from the April preview — the gains are all post-training — so the engine
  loads it exactly as before. The build a model came from is now recorded
  beside it: previously a bumped default was invisible to anyone who already
  had a file, since only the path's existence was ever checked. The stamp is
  advisory, and unknown never means re-download.

- **Breakout is playable while the model downloads.** Hours is a long time to
  watch a bar fill, so a round sits above the gauge. It never delays the
  transfer; Esc puts it away, and `q` or Ctrl-C abort the download from
  anywhere, so a rally can't trap you. The wall now hangs from the very top,
  and the paddle takes a shorter glide per key press — terminals give discrete
  presses rather than key-down/key-up, so the glide is effectively the step
  size, and a tap that slid the paddle past its own length made fine
  positioning impossible.

- **The TUI banner reads as one masthead**, with the version line set to the
  right of the logo's middle row instead of stacked underneath it.

## [2.7.7] - 2026-07-29

Beta channel on the 2.7 series.

### Fixed

- **A `/btw` answer is closed out when the aside finishes, not when the main
  task does.** A one-line aside is done in a single slice — around two seconds —
  while the main task may have hundreds of tokens left. Its panel line was only
  terminated when the whole multiplexed pass returned, so the answer sat
  unfinished on screen for the rest of the turn.

### Changed

- The aside's share of the thread is now 2:1 over the main task rather than 4:1.
  Measured, the weight mostly matters for a long aside: a one-line answer
  finishes inside a single slice at any weight, and what follows is the main task
  running alone because there is nothing left to share with.
- `README.md` and the site show `/btw` multiplexing as a recording, and no longer
  describe the old freeze-and-resume behaviour. `WHATS-NEW.md` is grouped by
  channel — what is in the betas, then the stable releases — instead of one
  section per version.

## [2.7.6] - 2026-07-29

Beta channel on the 2.7 series.

### Added

- **`/btw` answers beside the running task instead of freezing it.** An in-pass
  side question used to pause the main generation for the whole answer. It now
  runs on a fork of the session, interleaved with the main task at token
  granularity, so the main reply keeps advancing while the answer streams into
  the side panel. The aside gets the larger share of the thread (4:1) — it is a
  short question with someone waiting on it, and at an even split the answer
  took twice as long as it would have alone.

  This is time-slicing, not parallelism: one Metal command queue means nothing
  finishes sooner overall. What changes is that the main task no longer stops.

  One aside runs at a time and questions are no longer queued behind one — with
  the answer already on screen, holding the next question back bought nothing.
  Engines that cannot fork (the stub, remote engines) keep the old
  freeze/answer/resume behaviour.

### Fixed

- **The second `/btw` of a turn no longer re-prefills the whole conversation.**
  It rebuilt every token — 14602 of them on a moderate transcript, around a
  minute of dead air that read as a hang. A suspended pass generates twice into
  one assistant message, but the transcript recorded a span per generation
  while the rendered conversation holds one merged section; reconciliation
  diverged there and the prompt stopped extending the live cache. One assistant
  turn is now one span, and the second aside reuses 99.9% of its prefix.
  Diagnosis in `docs/DOUBLE-BTW.md`.
- **A `/btw` answer no longer loses its last line.** The panel was never told to
  close the final line, so the tail of every answer was dropped.

## [2.7.5] - 2026-07-29

Beta channel on the 2.7 series.

### Changed

- **The status bar holds the name of the running tool for as long as it
  runs.** The label used to be a 5s flash tip, so it could expire while the
  tool was still working, and it competed with the rotating tips for the tail
  slot. It is now posted for the life of the dispatch by an RAII guard,
  lingers 4s past the end so a fast tool stays readable, and is replaced
  immediately when the next tool starts. While it is up it owns the tail
  notification slot — tips are suppressed; the task counter and the rest of
  the bar are untouched — and blinks on an 800ms cycle, alternating the
  shimmer sweep with a dimmed pass at the same width so the line doesn't
  jitter.
- **The CRT power-off animation ends on a thinner trace.** Its last two
  phases used half- and full-block glyphs, which read as a solid bar rather
  than a collapsing scanline. The line is now `▁` (a lower one-eighth block,
  which tiles without gaps where a plain `_` would not) and the final
  phosphor point is a bold `.`, via crt-off 0.1.4.

### Fixed

- **CI is green again on the PDF tests.** `liteparse-pdfium-sys` downloads
  PDFium into `~/.cache/pdfium-rs` and bakes that path into the binary, but
  `rust-cache` restores `target/` and not that directory — so on a cache hit
  the build script never re-ran, nothing re-downloaded, and the baked path
  pointed at a directory the runner didn't have. Every `doc::` test died in
  `dlopen`, which is why 2.7.3 and 2.7.4 both shipped with CI red. The cache
  directory is now cached in its own right, keyed on `Cargo.lock` so a
  dependency bump re-downloads, and the build is forced to re-run when that
  cache is evicted while `target/` stays warm (#78).

## [2.7.4] - 2026-07-29

Beta channel on the 2.7 series.

### Added

- **Invented tool markup no longer scrolls past silently.** A bare `<task>`
  block, the shape the model falls back to for tools it was never trained on
  (#51), was invisible to the streaming detectors: the turn ended with no
  tool call and no error, and the model just tried again on the next turn. A
  detector now recognizes a registered tool name, or a generic `<tool_call>` /
  `<function_call>` / `<invoke ` wrapper, opening a line in the answer region,
  and routes it into the existing correction path so the model is handed the
  DSML syntax reminder instead. It stays quiet inside `<think>`, inside
  fenced code blocks, and mid-sentence in prose; a fence opened while
  thinking and closed just after `</think>` is still recognized as a closer
  rather than mistaken for a fresh opener that would disarm the detector for
  the rest of the answer. Matching waits for the end of the line, so a line
  that opens with a tool tag and then continues in prose is left alone; only
  a line that is nothing but the tag counts, and the error quotes the line
  that was actually seen. Because such a line can only appear after
  `</think>`, this error now takes precedence over the in-think prohibition
  when a generation manages both mistakes at once.

### Fixed

- **Two more shapes of tool call that used to die outright now run.** The
  dropped-leading-bar typo on inner tags (`<DSML｜parameter …>`) is now
  accepted, matching the tolerance the stanza opener has always had; the
  previous "unexpected DSML tag" error it produced sent the model chasing
  the wrong cause, in one recorded case concluding the fullwidth bar
  character itself was wrong. A tool call whose name is the prompt's own
  `$TOOL_NAME` placeholder now gets its own error instead of being reported
  as a placement mistake.
- **In-think rejections are usually logged with the stanza that was
  rejected**, rather than an empty payload, which was the single most common
  entry in `~/.plank/tool-call-errors.log`. The record now falls back to the
  held `<`-anchored tail when the parser buffer has already been drained; when
  no such tail was held the payload is still empty, so this narrows the blind
  spot rather than closing it. A hypothesis that the empty-payload bug
  compounded into a further misdiagnosis downstream, a leaked in-think stanza
  tail being reported as bad syntax rather than misplacement, did not survive
  review: the call site that reports bad syntax outside a stanza guards on the
  same in-think state. The pseudo-tool detector added later in this release
  does report from the answer region without that guard, but only for markup
  the model invented there, and its error deliberately outranks the in-think
  one.
- **Test runs stopped writing to the real error log.** `cargo test` was
  appending its fixture failures to the developer's own
  `~/.plank/tool-call-errors.log`, mixing synthetic test data into a file
  meant to reflect real sessions. Test builds now log to nothing instead of
  a shared machine path, and each log record is written in a single call so
  a concurrent writer can't interleave with it mid-record.

## [2.7.3] - 2026-07-28

Beta channel on the 2.7 series.

### Added

- **A sub-agent output pane on Ctrl-O.** Sub-agent model text no longer
  interleaves with the parent transcript: it streams into a second `OutputLog`
  you toggle to with Ctrl-O, titled `[sub-agent: <label>]` with a
  `ctrl+o: back to main` hint drawn over it without changing the output
  geometry. Scroll, End, jump hints, mouse drag-selection and the code-copy hit
  test all follow the visible pane, and the selection is cleared on a switch so
  a highlight is never painted over the other pane's rows. `/subagent` runs
  follow into the pane too, and a nested `agent` call can no longer clear the
  buffer or end the outer run mid-stream. The pane resets on `/clear`, `/new`,
  `/resume` and `/switch`. Headless and remote paths are unaffected: the
  pane-only events have no wire frame and are never broadcast, and the plain
  REPL keeps printing sub-agent output on stdout while the machine protocols
  keep the null sink.
- **The task, agent and plan-mode tools are always on.** The `tools` opt-in
  block in settings is gone along with its three README rows; there is nothing
  left to enable.

- **PDFs are readable.** `read` on a `.pdf` converts the document to Markdown
  and serves it exactly like a text file — bounded chunks, line numbers,
  `continue_offset=`, and `more` continuing where the last chunk stopped. The
  conversion is [liteparse](https://crates.io/crates/liteparse): spatial text
  extraction over PDFium, with bundled Tesseract OCR filling in pages that
  carry no text layer, so a scan reads as well as a born-digital file.
  Converted Markdown is cached by content hash under `~/.plank/doc-cache`, so
  re-reading a document — or paging through a long one — parses nothing.
  Paragraphs are hard-wrapped to 100 columns on the way in, because liteparse
  reflows each one onto a single line and line-based paging over a 5000-column
  line is not paging at all; tables, headings and fenced code pass through
  untouched. Design and rationale in `docs/LITEPARSE.md`.

  `visit_page` routes documents through the same converter, recognised by URL
  extension or by a `%PDF-` magic in the fetched body, instead of returning
  lossy-decoded mojibake.

  Deliberately **not** a new tool: the system prompt's tool table is frozen by
  `tests/c_parity.rs`, and appending to it would churn the Tier 1 KV
  fingerprint. Extending `read` by extension costs one sentence of prompt —
  which turned out to be the sentence that matters, see *Fixed* below.

  Office formats stay a non-goal for now. liteparse accepts DOCX/XLSX/PPTX, but
  it reaches them by shelling out to LibreOffice or ImageMagick to make a PDF
  first; `DOC_EXTENSIONS` is `["pdf"]` until that dependency is detected and
  degraded from explicitly. The whole thing sits behind the default-on
  `docparse` feature (~17 MB of binary, a CMake build of libtesseract);
  `--no-default-features` gets the old behaviour, where a PDF reads as bytes.

### Changed

- **`/insights` says what it is doing and stops when asked.** The window title
  reads `introspecting...` for the duration, restored by a drop guard so no
  error path can leave it describing finished work. Esc and Ctrl-C now take
  effect: the command runs inside the dispatch, so the event loop was parked
  and no key was ever read — the repaint drains pending keys and raises the
  same shared interrupt flag a SIGINT sets. The session scan polls per session
  and the streaming callback stops the current section rather than waiting for
  the next section boundary. Cancelling is reported as cancelled, not as a
  failure. The report is written to `report.html.tmp` and renamed into place,
  and the render checks between sections rather than mid-string, so a stopped
  run leaves the report you already had intact rather than a truncated one.
- **System status lines are the theme green**, not dim pink, and any http(s)
  URL in the message is lifted to white so the target reads apart from the
  prose (`Opening page ...`, `Searching Google for ...`).

### Fixed

- **The screensaver came up the instant a long generation ended.** The idle
  clock was stamped when an event arrived, so the Enter that submitted the
  prompt started the countdown and it had long elapsed by the time the turn
  finished. `tui_loop` now also stamps after the key match, and remote-driven
  and `--prompt` startup turns stamp at their own call sites.
- **The model would not use PDF support unless the prompt said it existed.**
  `docs/LITEPARSE.md` argued the prompt cost was zero because "the model does
  not learn anything new — documents simply stop being unreadable". False in
  the one way that mattered: a model that believes a `.pdf` is unreadable never
  calls `read` on one. It searched for the file, found it, and shelled out to
  `pdftotext` — slower, unpaged, and absent on most machines. One sentence now
  sits with the other reading rules, outside the C-locked base so the parity
  fixtures stay authoritative.
- **A large PDF was rejected before it was ever converted.** The conversion
  path loaded the file through `read_file_bytes`, which enforces the 16 MB
  `FILE_MAX_BYTES` cap — and did so only to compute a cache hash. That cap
  bounds how much text a read may put in context, which is the wrong rule for a
  PDF: its bytes never enter the context, they are input to a converter whose
  Markdown is paged. A 60 MB manual, exactly the case the feature exists for,
  came back as *too large to read*. Hashing now hands the path to `shasum`,
  which streams it.
- **Tesseract's C++ diagnostics landed on the TUI prompt line.** liteparse's
  `quiet` flag gates only the crate's own logging; the bundled Tesseract writes
  `Detected N diacritics` and friends straight to fd 2 through C stdio, and
  because plank parses in-process those bytes went wherever the cursor happened
  to be. A mutex-serialized `StderrSilencer` `dup2`s fd 2 to `/dev/null` around
  the parse and hands it back afterwards. The regression test converts a noisy
  scanned fixture in a subprocess — in-process fd capture races across parallel
  test threads — and asserts nothing but a sentinel reaches stderr.
- **`/insights` no longer reports timing that does not describe your history.**
  A history recorded before per-message timestamps sums to a near-zero hour
  count, and the model was handed it as fact: "your total time spent is very
  limited, which suggests many quick interactions rather than sustained deep
  work", drawn entirely from the artifact. Timing now counts only when at least
  half the counted sessions carry it — the earlier guard asked merely whether
  the total was non-zero, which `0.3h` sailed past. Unrepresentative timing is
  omitted from the model's context, shown as an em dash in the stat tile, and
  dropped from the terminal summary; the `(unrecorded)` project placeholder is
  never sent as if it were a project name.

## [2.7.2] - 2026-07-28

Beta channel on the 2.7 series.

### Added

- **`/insights`**, a personal usage report over every saved session, written to
  `~/.plank/usage-data/report.html` (owner-readable only) with a condensed
  summary in the terminal. Adapted from Claude Code's builtin of the same name,
  keeping its central discipline: every number is computed deterministically
  from the transcripts — tool mix, languages, lines added and removed, files
  touched, commits, failure categories and per-tool failure rates, reply times,
  activity by hour, and sessions that were live at the same moment — and the
  model is asked only for the prose it cannot replace. Unlike the reference,
  plank does not ask the model to judge each session individually: that is one
  call per session, which does not scale on a local engine, so the model sees
  only the finished aggregate and costs a fixed handful of calls. `/insights
  fast` skips the written sections entirely. Per-session statistics are cached
  under `usage-data/session-meta/`, so a rerun is milliseconds rather than
  seconds; a section whose model call fails is dropped without touching the
  statistics.

  The session format grew two optional, backward-compatible records for this:
  a per-message timestamp on `msg`/`node`, and the project directory the
  session ran in. Sessions saved before this release still load, still count,
  and still contribute every statistic that does not need a clock — the report
  says how many of them there are rather than averaging over a smaller set.

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

### Fixed

- **The starfield screensaver came up on grey instead of black.** Its opaque
  background was painted with `Color::Black`, which is ANSI index 0 — a slot
  terminal themes remap freely, and most render as a dark grey. It is now an
  explicit `Rgb(0, 0, 0)`, which no theme can reinterpret. The same paint path
  backs an opaquely opened arcade game, so those get a real black too.

### Internal

- **CI's format check no longer fails on vendored submodule code.** It ran
  `cargo fmt --all`, which follows the `path =` dependencies into `refs/obscura`
  and `refs/edit` and checks upstream's source against plank's `rustfmt.toml`;
  the result was a red build on every push that said nothing about plank. It is
  `cargo fmt --check` now, which stops at the package boundary.

- **Restored the `refs/openclaw` stanza in `.gitmodules`.** It was dropped while
  `refs/edit` was added, but its gitlink stayed in the tree, so every CI and
  release checkout died at `git submodule update --recursive` before building
  anything. Its old entry carried `update = none`, which is why it had been
  skipped harmlessly until then.

- The compaction prompt is locked with a fixture (`tests/plank_prompts.rs`, the
  mirror of `c_parity.rs` for text plank deliberately does *not* share with the
  C reference). It names the `<summary>`/`<analysis>` tags that
  `compact::extract_summary` parses, so an accidental edit to either half would
  have failed silently at runtime rather than in CI.

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
progress is shown while the engine works, and one to how the spinner looks
while it works.

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
- **The shimmer sweeping across the spinner verb is shaded in the theme's own
  hue** rather than flat white. All three of its columns were painted pure
  white over the military green, which reads as a blown-out glitch and pulls
  the eye harder than the text it decorates. Each column now takes its
  lightness by distance from the center of the window — brightest in the
  middle, easing back into the theme color at the edges — and the window
  widened to match, so adding a shade softens the sweep in one edit.

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
- **`/version`**, on both the REPL and TUI paths.
- **Rejected DSML tool calls are logged** to `~/.plank/tool-call-errors.log`,
  which is the record later releases mine to find the shapes of tool call the
  model actually gets wrong.

### Changed

- **ds4 engine transcript is token-primary** (#58): the token buffer is now the
  source of truth (C-parity append-only transcript), with text derived from
  tokens, replacing the text-primary reply-splice cache.
- **Hierarchical KV cache tier foundation** (#60): the session-start context is
  split into a project-stable tier and a session-volatile tier, with
  tier-fingerprint chaining and project-scoped checkpoint paths. (Native
  restore-loop wiring tracked in #64.)

### Fixed

- **The CRT power-off animation ran over an all-black image.** Ratatui swaps
  its buffers before `draw()` returns, so the effect rasterized the blank
  post-swap buffer instead of the frame that had just been on screen. The TUI
  loop now snapshots each tick's completed frame and hands the last one back
  to the caller, gated on the effect actually being enabled so there is no
  per-tick cost when it is off.

## [2.5.2] - 2026-07-25

### Added

- **`/renotify`** re-shows the last delivered desktop notification. The
  banner is remembered after delivery, which matters on Ventura and later
  where `_showsButtons` no longer makes a banner stick around to be read.
- **A missing API key falls back to `DUMMY`** instead of erroring, so
  key-less OpenAI-compatible endpoints — a local `ollama serve`, for
  instance — work without inventing a credential for them.

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

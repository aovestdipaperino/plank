# Findings

Everything plank learned the hard way while porting `ds4_agent.c`, in two
parts:

1. **Wire-format and parity nuances** — behaviors the Rust port must
   replicate byte-for-byte because the DeepSeek V4 Flash model was trained on
   the C agent's exact output, plus the Rust-side traps that silently break
   that identity. Each item states the behavior and where it is enforced.
2. **Environment & tooling** — build, release, and terminal gotchas: the kind
   of thing that costs an hour the second time you hit it.

The enforcement mechanism for part 1 is `tests/c_parity.rs`: committed
fixtures under `tests/fixtures/` are byte-compared on every `cargo test`, and
when the `refs/ds4` submodule is checked out the C string constants are decoded
straight out of `ds4_agent.c` and compared too, so the fixtures cannot drift
from the reference. Regenerate fixtures with `PLANK_REGEN_FIXTURES=1 cargo
test` and review the diff before committing.

## Part 1 — Wire-format and parity nuances

- **Rust `\` string-literal continuation eats leading whitespace.** A
  backslash at the end of a line inside a `"..."` literal skips the newline
  *and all leading whitespace on the next line*. The tools prompt was written
  as one continued literal, which silently deleted the 4-space indentation in
  the anchored-edit example and every indent inside the JSON tool schemas —
  thousands of bytes that no longer matched what the model was trained on,
  invisible in review because the source *showed* the indentation. First
  thing the parity tests caught. The schema section now lives in
  `src/resources/tools_prompt_after_edit.txt` (included via `include_str!`),
  and any string that must survive byte-exact should either avoid continued
  literals or keep the indentation on the same physical line as the `\n`.
- **DSML markers use the fullwidth vertical bar U+FF5C (`｜`), not ASCII
  `|`.** `<｜DSML｜tool_calls>` etc. (`src/dsml.rs`). The parser is
  deliberately strict after the opening marker; typo tolerance lives in the
  streaming detector (`src/viz.rs`), never in the executable parser.
- **The system prompt is tokenized in two different ways.** The built-in
  tools prompt goes through the chat template so the DSML markers become
  control tokens; user `-sys` text is tokenized as plain content. Composing
  them as one string is fine for display but not for tokenization
  (`src/sysprompt.rs`, `build_system_prompt` doc).
- **Tool results are stored as user-role turns.** History replay detects them
  by prefix — `<tool_result>`, `Tool:`, or `Tool result` — exactly like the C
  (`src/session.rs:149`).
- **Tool-result framing.** Each call's output is prefixed with
  `Tool result N (name):\n` (1-based, `unknown` when the call has no name), a
  trailing `\n` is appended only when the output is non-empty and doesn't end
  with one, and an empty DSML block yields exactly
  `Tool error: empty tool call block\n` (`src/tools/mod.rs`,
  `dispatch_all`, mirroring `agent_execute_tool_calls`).
- **Session identity is SHA-1(title bytes ‖ created_at as little-endian
  u64).** Once assigned it never changes; listing ties break on ascending id;
  only 40-hex-stem files are considered sessions (`src/session.rs`).
- **The system-prompt reminder is pressure-based, not periodic.** It is
  re-injected only once the token-estimate distance since it was last seen
  exceeds 50,000 (`AGENT_SYSTEM_PROMPT_REMINDER_TOKENS` in the C,
  `SYSTEM_PROMPT_REMINDER_TOKENS` in `src/sysprompt.rs`).
- **The datetime context line falls back to raw Unix seconds.** Local time is
  formatted with `strftime("%Y-%m-%d %H:%M:%S %Z")`; if that fails, the raw
  seconds are printed instead — the surrounding sentence is fixed either way
  (`src/sysprompt.rs`, `datetime_context_line`; timestamp masked in the
  fixture).
- **KV-cache identity is textual, not structural.** The sysprompt checkpoint
  fingerprint is SHA-1(model name ‖ NUL ‖ system prompt text); a mismatched
  fingerprint means rebuild, never trust (`src/ds4engine.rs`,
  `checkpoint_fingerprint`). Retokenizing the previous reply's *text* does
  not reproduce its sampled token ids, so the engine remembers the exact
  sampled tokens of the last reply and splices them into the next prompt —
  otherwise the KV common-prefix probe diverges at the start of the reply and
  the whole tail re-prefills (`src/ds4engine.rs`, `build_tokens`).
- **Per-session KV payloads cannot share the transcript's `.kv` name, and a
  restored payload must drop the spliced-reply cache.** The C stores the
  engine payload inside the session `.kv` file; plank's v1 transcript format
  already owns that name, so payloads live in a `<sha>.payload` sidecar
  (fingerprint line + `ds4_session_save_snapshot` bytes, same layout as
  `sysprompt.kv`). The fingerprint covers model ‖ NUL ‖ system prompt ‖ NUL ‖
  rendered transcript, so a resave after more turns (or compaction) is
  detected as stale. And because `Ds4Engine` splices the *previous* reply's
  exact sampled tokens into the next prompt build, restoring a snapshot from
  another conversation with `last_reply` still set would splice the wrong
  tokens — `Ds4Session::restore_kv` clears it (`src/ds4engine.rs`), and the
  payload load path (`Agent::load_session_payload` in `src/ui.rs`) goes
  through `restore_kv` so it inherits that. The raw KV bytes are the shared
  `snapshot_kv`/`restore_kv` primitive (`SessionSnapshot::as_bytes` /
  `restore_bytes`); the `.payload` sidecar and fingerprint only wrap them, so
  there is no second hand-rolled KV-serialize path.
- **`count_tokens` must subtract chat-template overhead** so it reports
  text-only counts; the template wrapper is measured once at engine startup
  (`src/ds4engine.rs`).
- **Trace timestamps are byte-for-byte `agent_trace_time`**
  (`clock_gettime`-derived formatting, `src/trace.rs:127`).
- **A session snapshot owns its buffer; `ds4_session_snapshot_free` frees only
  what the engine allocated.** `ds4_session_save_snapshot` allocates the
  buffer, so the owning `SessionSnapshot` wrapper frees it on drop
  (`src/snapshot.rs`). But *loading* a snapshot read back from disk must wrap
  the caller's `Vec` in a **transient, non-owning** `Ds4SessionSnapshot` and
  never call the free — the buffer is Rust's, and freeing it via the C
  allocator double-frees. Hence `SessionSnapshot::restore_bytes` builds the
  FFI struct on the stack and drops the `Vec` itself; only `capture` produces a
  freeable snapshot. Restore itself (`ds4_session_load_snapshot`) is
  idempotent and lossless — the KV, cursor, and any partial reply come back
  byte-identical, which is what makes an unconditional-restore RAII guard
  (`RestoreOnDrop`) safe on the aside interrupt/error path.
- **Resuming a suspended pass reuses the partial via `last_reply` splicing, not
  a longer prompt string.** After an in-pass `/btw` suspend (`--btw-suspend`),
  the worker resumes the frozen main pass by re-invoking `generate` with the
  prompt `render_transcript(...) + "[assistant]\n" + partial`. That extra
  assistant section matters: `Ds4Engine::build_tokens` only splices the exact
  sampled tokens of `last_reply` when the transcript's last assistant section's
  text *equals* `last_reply.text`. Match, and `ds4_session_common_prefix` reaches
  through the partial and only the closing EOS + new assistant prefix are
  prefilled (≈2 tokens); mismatch (e.g. a trailing-whitespace drift, since
  `last_reply.text` is `trim_end`-ed), and it silently falls back to
  re-prefilling the partial's text — still correct output, just not free.
  `generate_aside` preserves `last_reply` across the aside (save/restore) so the
  splice is available on resume. The worker orchestration is straight-line in
  `Agent::worker_turn` (`src/ui.rs`): the engine owns the token loop, so
  "suspend" is `stop-at-boundary → generate_aside → resume`, not a callback
  interposed mid-loop.

## Part 2 — Environment & tooling

- **The Metal backend needs the macOS 15 SDK** (`MTLResidencySet`), so
  release builds run on `macos-15` runners and bottle as `arm64_sequoia`.
  The ds4 Makefile's `-mcpu=native` default is invalid for x86_64 clang and
  non-portable for bottles; override `NATIVE_CPU_FLAG` per arch
  (`apple-m1` / `x86-64-v3`).
- **Releases are Homebrew-only and the tag number is the channel.** The
  highest tagged major is beta (`plank-beta` formula), everything below is
  stable — there is no channel flag anywhere. See `VERSIONING.md`.
- **Upgrades run maintenance keyed on the version delta.** On first launch
  after a version change, `src/upgrade.rs` drops the sysprompt KV checkpoint
  (minor) or that plus the image cache (major / downgrade / unknown
  previous). Session transcripts are never touched. Pick release numbers
  accordingly: bump minor when the sysprompt or engine snapshot format
  moves, major when older caches must not be trusted at all.
- **Never bake filesystem paths in with `env!` for shipped binaries.** The
  Metal kernel dir compiled in via `env!("DS4_METAL_DIR")` was the CI
  runner's checkout, so every brew install failed model load with a
  misleading "failed to open model" (fixed in v0.9.10). `metal_source_dir`
  in `src/ds4engine.rs` now resolves at runtime: `DS4_METAL_DIR` env →
  compile-time path (dev builds) → `../share/plank/metal` next to the
  executable (bottles ship the kernels there). Keep any new bundled-asset
  lookup on the same pattern.
- **The default quant needs ~82 GB resident**, hence the hard 96 GB RAM
  guard before any download or model load (`src/main.rs`).
- **Download resume trap:** a `.part` file already matching the full
  `Content-Length` must be renamed, not range-requested — otherwise the
  server answers 416 forever (`src/download.rs`).
- **Two parallel slash-command paths.** The plain stdout REPL and the Ratatui
  TUI each implement slash-command handling in `src/ui.rs`; a change to one
  usually needs the mirror change in the other.
- **Terminal quirks:** block-based terminals (Warp) need the alternate-screen
  TUI rather than scroll regions; clipboard copy goes through `pbcopy` *and*
  OSC 52; the TUI ANSI parser must handle 256-color `38;5` SGR as well as
  truecolor `38;2`, or `/context` and `/mcp` render monochrome.
- **Ratatui swaps and clears buffers on every `draw()`.** After a frame is
  flushed, `terminal.current_buffer_mut()` is the *empty next-frame* buffer,
  not what's on screen. Reading rendered cells after the fact (the original
  selection-copy bug, issue #1) silently yields blank text; extract cell
  content inside the `draw` closure from `frame.buffer_mut()` while the
  frame is still being composed (`src/ui.rs`, mouse-up handler).
- **Strict provider gateways reject noisy float params.** plank's sampling
  knobs are `f32`, and serde_json widens e.g. `temperature: 0.6` to the noisy
  `f64` `0.6000000238…`, printing every digit. z.ai's Anthropic-compatible
  gateway rejects any `temperature`/`top_p` with more than two decimals
  (`400 … "temperature parameter is illegal"`). `build_anthropic_request` /
  `build_openai_request` now route both through `round2()` (`src/remote/provider.rs`).
  Also note z.ai's base URL is `https://api.z.ai/api/anthropic/**v1**` — plank
  appends `/messages` itself, so the `/v1` segment must be in `--base-url`.
- **Raw-DSML display is not parity territory.** The C agent dumps the
  rejected stanza's raw bytes on a parse error; plank deliberately diverges
  and suppresses them (issue #11) — only the bold-red
  `[invalid tool call: ...]` banner (which names the offending tag) is shown,
  routed through `RenderSink::error_text`. Byte-parity applies to what the
  *model* sees (transcript, tool results), never to the terminal projection.
- **`Agent::tui_loop` cannot be driven in-process by an integration test.**
  Its terminal parameter is `&mut ratatui::DefaultTerminal`, a type alias for
  `Terminal<CrosstermBackend<Stdout>>` — not generic over `Backend` — so a
  `TestBackend` can't be substituted without changing production code's
  signature just to make it testable. `tests/ui_remote.rs` covers the
  `uiremote` primitives it depends on (region recording, `frame_tree`,
  `buffer_to_ansi`) directly instead; the injection/deferred-reply plumbing
  in `UiRemote::drain` stays covered only by `src/ui.rs`'s unit tests.

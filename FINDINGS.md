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
  `checkpoint_fingerprint`). Retokenizing a previous reply's *text* does
  not reproduce its sampled token ids: BPE encoding is many-to-one, and the
  tokenizer picks *one* canonical segmentation (its merge order), while the
  sampler is free to emit any of the equivalent sequences — it can sample
  `"in"`+`"to"` where the encoder would produce `"into"`, split a number or
  identifier at a different boundary, or emit a rare standalone token the
  merge rules would have absorbed. Detokenize-then-retokenize is therefore
  not the identity on token ids even though it is on text (and trailing-
  whitespace trimming plus multi-byte characters split across tokens add
  further drift at the edges). One different id shifts every byte after it,
  so the KV common-prefix probe diverges at the first such reply. The C never
  faces this because it is token-first: `w->transcript` is the append-only
  *token* buffer — sampled tokens are appended to it directly during
  generation and it is what gets persisted in the session `.kv` — while text
  is always *derived* from tokens (`ds4_kvstore_render_tokens_text`) for
  display and export, never the reverse. The C retokenizes text
  (`ds4_tokenize_rendered_chat`) only when loading a *stripped* session whose
  token payload was deleted ("rebuilt from text" in `ds4_agent.c`), accepting
  the one-time full re-prefill that implies. Plank inverts the
  representation — the text transcript is primary — so it must carry the
  sampled tokens alongside as splice state instead. That inversion is
  deliberate, not an accident of porting: token ids only mean anything to the
  one ds4 vocabulary, while plank's `Engine` trait spans backends with no
  shared token space (`EchoEngine` in every CI run, remote provider engines
  that take text or structured messages and tokenize server-side), so text is
  the only representation every engine can consume. Everything above the
  engine boundary also wants text: compaction feeds the transcript back
  through the model as prose and splices summaries in, `render_transcript`
  is the C-parity surface the fixtures byte-compare, and the v1 session
  format keeps transcripts readable/greppable on disk rather than as an
  id dump that dies with its vocabulary. The cost of that choice is exactly
  this entry: the one token-exact backend must remember its sampled ids on
  the side (`replies` + the payload wrap) instead of getting exactness for
  free from an append-only token buffer. Could the token buffer be the
  source of truth instead, since token → text is deterministic and text
  could always be derived (the C's model)? For the ds4 backend alone, yes —
  that is literally the C design. As plank's global source of truth, no:
  a token transcript is bound to one vocabulary, so provider engines (which
  never see ids) and any model/engine switch would need the derived-text
  path anyway, making text the de-facto interchange format with tokens as a
  cache — which is the current design viewed from the other side. Note that
  text-shaped mutations are *not* a blocker for token-primary: rewrite ops
  (compaction, tool-result clearing) can detokenize → edit the text →
  retokenize back into the buffer, and the C does exactly this
  (`ds4_kvstore_render_tokens_text` → rewrite → `ds4_tokenize_rendered_chat`).
  The retokenized ids differ from the sampled ones, but a rewrite already
  invalidates the KV at the rewrite point, so the re-prefill this forces
  coincides with one that was due anyway — and rewrites are rare, so the
  cost amortizes to nothing. The decisive argument is the vocabulary
  binding alone: with multiple backends and no shared token space, tokens
  cannot be the interchange format, only the ds4-local cache. That is why
  the engine keeps the exact sampled tokens of *every* reply still present in
  the transcript (`replies:
  Vec<SampledReply>`, the Rust half of the C's append-only token transcript)
  and splices each matching assistant section into the next prompt — otherwise
  the KV common-prefix probe diverges at the start of the first re-templated
  reply and the whole tail re-prefills.
  **Update (issue #58): the ds4 backend is now token-primary.** `Ds4Session`
  owns an append-only token buffer (`ds4tokens::TokenTranscript`, mirroring the
  C's `w->transcript`) instead of the `replies: Vec<SampledReply>` splice cache.
  Each turn *reconciles* the UI's rendered transcript against the buffer's spans
  by a structural common-prefix on (role, text) keys (`build_prompt` →
  `reconcile` in `src/ds4engine.rs`): the matching prefix keeps its exact tokens
  verbatim; at the first divergence the tail is dropped and retokenized (user/
  system/tool retokenize deterministically; a re-appended assistant section pays
  one re-prefill the KV was already going to force). This is robust to any text
  drift instead of exact-match fragile, and replaces `build_tokens` /
  `plan_splices` / `retain_matched` (removed). The reconciliation and
  persistence logic lives FFI-free in `src/ds4tokens.rs` and is unit-tested in
  CI without the native engine. NOTE: the native/Metal runtime behavior of this
  inversion is unverified in the porting environment (no submodule/model);
  compaction/rollback rewrite ops still `truncate_spans`+re-append rather than
  detokenize→edit→`reset`, which `TokenTranscript::reset` is staged for.
- **Every persisted KV is one `KVCache` in one format, and a restored payload
  must restore the token transcript captured with it.** The C stores the engine
  payload inside the session `.kv` file; plank's v1 transcript format already
  owns that name, so session payloads live in a `<sha>.payload` sidecar. All of
  them — the system-prompt checkpoint, the per-project tier checkpoints, and
  session payloads — are written by `KVCache::persist` and read by
  `KVCache::from_file` (`src/kvcache.rs`) in a single layout:
  `<signature>\n<version:u8><encoded transcript><raw KV bytes>`. `SessionStore`
  owns every path via `KvKey` (`System` / `Project` / `Session`), so no other
  code constructs a KV filename. Reads are `Option`-valued and a miss is
  indistinguishable by design: absent, signature mismatch, truncated body, and
  an unrecognized version byte all mean "prefill instead", so nothing else in the
  tree makes a trust decision about cached bytes. Writes are best-effort — a
  failed persist costs the next launch one prefill and must never abort startup.
  For a session payload the signature is `payload_fingerprint` = model ‖ NUL ‖
  system prompt ‖ NUL ‖ rendered transcript, so a resave after more turns (or
  compaction) is detected as stale; keying on the session id alone would make a
  payload captured under a *different model* a cache hit rather than a rejection.
  The token transcript travels inside the value rather than as a hand-rolled
  prefix (it **was** `plank-replies-v1\n`, then `plank-tokens-v1\n`; issue #58),
  so a resumed session keeps the restored KV's own token buffer — never another
  conversation's — and prefills only genuinely new tokens; an empty buffer is
  still correct, it just rebuilds from text and re-prefills from the first reply.
  Tier checkpoints deliberately carry an **empty** transcript, so "the transcript
  describes this KV" holds for session payloads but *not* for tier checkpoints —
  do not start trusting it there. This one type replaced five divergent paths
  (three separately reimplementing `<fingerprint>\n<bytes>`, two different
  payload shapes, plus the legacy fallback). The only KV bytes still framed by
  hand are the host idle-reclaim swap file (`snapshot_bytes` / `restore_bytes` on
  `Ds4HostSession`): a process-scoped temp file with no signature and no
  staleness question, which is the sole remaining reason `strip_legacy` exists.
  Temp files are `<name>.tmp.<pid>` — the pid matters, because two processes
  persisting the same path would otherwise interleave into a file whose signature
  line and version byte are intact but whose KV region is spliced, which
  `decode` would accept.
- **A KV cache tier boundary must fall on a chat-template *message* boundary,
  and a tier's checkpoint must be taken while the cursor sits exactly at that
  boundary.** The tiered cache (#60/#64) makes the project-stable context a
  reusable prefix, but a snapshot only replays if the tokens ahead of it are
  reproducible: byte-level BPE merges across a mid-message seam mean
  `tokenize(stable)` is not necessarily a prefix of `tokenize(stable ‖
  volatile)`, and the per-message template wrapper closes at the end of a
  message anyway. So the session-start context enters as **two** user messages
  (stable then volatile, `ui::push_session_context`) rather than one — the
  concatenated text is unchanged, and the system prompt (the only
  parity-pinned part) is untouched, so `tests/c_parity.rs` is unaffected.
  Likewise, `SessionSnapshot::capture` serializes the *whole* session, not a
  prefix of it, so `kvtier::warm` syncs to one tier's end, writes that tier's
  checkpoint, and only then syncs the next — never build the full prefix first
  and try to checkpoint a tier retroactively. Fingerprints cannot catch a
  violation here: persisting tier *i* after prefilling tier *i+1* stores tier
  *i+1*'s KV under a key that is genuinely correct for tier *i*.
- **The token buffer handed to `ds4_session_sync` must always *extend* the live
  checkpoint's end. Anything else silently discards the entire KV.** This one
  rule caused two separate bugs, and it is the thing to check first whenever a
  turn is unexpectedly slow. `ds4_session_sync` reuses the live KV only when
  `prompt->len >= checkpoint.len && ds4_tokens_starts_with(prompt,
  &s->checkpoint)`; every other case — a divergence *or* a prompt that is a
  strict **prefix** of the live checkpoint — falls through to
  `metal_graph_reset_prefill_state` and re-prefills from zero. The C states why
  next to `ds4_session_rewrite_requires_rebuild`: "Extending exactly at the live
  end is safe; rewriting behind it is not an in-place operation" — the backend
  still holds raw SWA rows, compressed KV rows, indexer rows, and compressor
  frontiers for the old suffix, and shortening the token vector
  (`ds4_session_rewind`) cannot roll those back. **So the only safe way to move
  *back* to a shorter prefix is to restore a real frontier snapshot (`set_kv`),
  never to truncate.** `/rollback` and `/switch` are correct precisely because
  they do that.
  The corollary trap is that **`ds4_session_common_prefix` answers "how many
  tokens match", not "how many will be reused"** — the two diverge in exactly
  the reset cases, and believing the former turns a full rebuild into a silent
  one. `engine::reusable_prefix(pos, common)` is the honest predicate:
  `common == pos` or nothing.
  *Instance 1 — `/new` and `/clear`.* A fresh session's rendered transcript is
  the head of the one it replaced (same system prompt, same session context), so
  the next prompt is a strict prefix of the live KV. `common_prefix` returned the
  whole prompt while the engine threw the KV away, and because
  `PrefillProgress::primed` treats a fully-cached prompt as complete, a
  ~2500-token prefill ran with the bar already at 100% and no further event ever
  arriving — indistinguishable from a hang. `Agent::rewarm_after_reset` now
  re-runs `kvtier::warm` after a reset, restoring the tier checkpoint (a genuine
  frontier snapshot at the warm boundary) so the next turn extends it. Measured
  on DeepSeek V4 Flash for `haiku; /new; haiku`: a 2509-token rebuild reported as
  `prefill=0 (100.0% reused)` became a 7-token prefill, 31.7s → 19.7s. The
  post-`/new` trace is now byte-identical to a cold launch's first turn, which is
  what `/new` should mean.
  *Instance 2 — the tier walk.* `kvtier::warm` must call `warm_append` for
  **every** tier and skip only the *sync* for tiers below the resume point. An
  early version skipped the append too (`.skip(resume)`), so after restoring the
  project checkpoint the buffer read `[system, volatile]` while the live KV held
  `[system, project]` — no longer an extension, so `sync` discarded the
  just-restored KV and the deep-hit path became *slower* than a cold start. Hence
  the split between `warm_append` (extend the buffer, no prefill) and `warm_sync`
  (prefill to the buffer's end); a single combined call cannot express it. Note
  that a spy/echo engine which models no token buffer cannot catch this class of
  bug — it is invisible to `kvtier`'s unit tests by construction, and only the
  `PLANK_KV_DEBUG` trace or a real-model timing shows it.
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
- **Resuming a suspended pass reuses the partial via reply splicing, not
  a longer prompt string.** After an in-pass `/btw` suspend (`--btw-suspend`),
  the worker resumes the frozen main pass by re-invoking `generate` with the
  prompt `render_transcript(...) + "[assistant]\n" + partial`. That extra
  assistant section matters: `Ds4Engine::build_tokens` only splices a
  remembered reply's exact sampled tokens when an assistant section's text
  *equals* that reply's text (`plan_splices`). Match, and
  `ds4_session_common_prefix` reaches through the partial and only the closing
  EOS + new assistant prefix are prefilled (≈2 tokens); mismatch (e.g. a
  trailing-whitespace drift, since reply text is `trim_end`-ed), and it
  silently falls back to re-prefilling the partial's text — still correct
  output, just not free. After resume, the partial and its continuation sit in
  the history as two entries while the transcript shows one merged section, so
  that turn re-templates once and both entries prune — bounded, not
  compounding. `generate_aside` restores a pre-aside copy of the history (the
  aside still splices the shared prefix) so the splice is available on resume. The worker orchestration is straight-line in
  `Agent::worker_turn` (`src/ui.rs`): the engine owns the token loop, so
  "suspend" is `stop-at-boundary → generate_aside → resume`, not a callback
  interposed mid-loop.
- **Tool calls inside `<think>` are a deliberate divergence.** The C reference
  forbids them in two places: the tools-prompt line at `ds4_agent.c:718` and the
  stream-time discard at `ds4_agent.c:3107` and friends. plank can dispatch them
  instead, behind `engine.thinkingToolCalls` (`/config`), which is **off** by
  default so the shipped behavior is strict parity. Turning it on strips the
  prompt line from the built prompt — the C string constants in
  `src/sysprompt.rs` are still verbatim, so `tests/c_parity.rs` keeps passing;
  only the assembled output differs.

  A rejected call is reported to the model as a *placement* error, never a
  syntax one. The C routes it through the malformed-tool path
  (`ds4_agent.c:7853`), so it reaches the model behind an `invalid DSML tool
  call:` prefix plus the DSML syntax reminder — and when the model stopped
  mid-stanza, behind the parser's own "incomplete DSML tool call". Both tell it
  to fix markup that was correct; the actual mistake was where the call sat.
  plank overrides the parse verdict in `StreamRenderer::finished` and words it
  with the prohibition sentence the tools prompt already carries, no syntax
  reminder attached. Watch the distinction between `dsml_in_think` (a marker
  was *seen* in thinking — also true when dispatch is allowed) and
  `in_think_rejected` (a call was actually thrown away): only the second is
  worth an error, or the allow path reports failures for calls it just ran.

  A call fired mid-thought leaves the reply with an unterminated `<think>`. plank
  appends a synthetic `</think>` before the `<tool_result>` message
  (`close_open_think` in `src/ui.rs`). No engine change is needed to resume
  reasoning: the local chat template re-opens `<think>` in the prefill prefix on
  every assistant pass, so the continuation is already inside a think block.
  This is expected to be cheap in KV-cache terms since the divergence sits at
  the very tail of the reply, but that is unmeasured — worth a manual macOS
  run against a real GGUF model before release.

- **Forward recovery from an in-think tool call must not re-emit the stanza
  opening.** When the model opens a DSML stanza inside an unclosed `<think>`,
  the fix is to force-feed `</think>\n\n` and stop there: that position
  predicts a fresh stanza opening strongly enough that the model restarts the
  call on the executable side by itself. The C tried also re-emitting the
  opening after the close and found it counterproductive — with the dangling
  opening right before the close and a forced copy right after it, the model
  reads the call as already made and ends the turn. The dangling opening is
  harmless where it is, inside reasoning.

  Two details are load-bearing. Detection runs on *accumulated* text, so the
  marker's tokenization does not matter — but the scan cursor must be held back
  past the longest opening (`TOOL_START_SCAN_HOLD`) or an opening split across
  tokens is missed, and it must be snapped to a UTF-8 char boundary because the
  markers are multi-byte. And the trigger is only the `tool_calls` *wrapper*
  form, not the bare `invoke` opener the streaming detector also accepts: a
  forced injection is too expensive to spend on a weaker signal.

  This is a policy fork from the agent-side handling, not a replacement for it.
  plank enables recovery only when `engine.thinkingToolCalls` is false; with
  in-think calls allowed the stanza is dispatched where it sits and cutting
  reasoning short would be a regression.

- **An interrupted compaction must keep the old transcript, not the new KV.**
  Interrupting the summary pass leaves the live KV holding the private
  compaction prompt while the transcript still holds the real conversation. The
  C calls `ds4_session_invalidate` there; plank does not need to, because
  `build_prompt`'s common-prefix reconciliation sees the next turn's prompt as a
  strict *prefix* of the live checkpoint and rebuilds from zero anyway (the
  `reusable_prefix` rule). Correctness is the same, cost is the same full
  rebuild — but do not "optimize" that reconciliation without re-checking this
  path. Interruption is also not a failure: the turn returns to idle, and the
  latched interrupt has to be consumed (`shared.interrupt` under the TUI, the
  SIGINT flag otherwise) or the next turn starts already cancelled.

- **`cargo fmt --all` reaches into vendored path dependencies.** plank is a
  single package, but `obscura` and `edit` are `path =` dependencies pointing
  into submodules, and `--all` formats those crates too. Upstream code is not
  written to plank's `rustfmt.toml`, so CI's `cargo fmt --all -- --check`
  reported 667 diffs — none of them plank's — and failed every push while
  saying nothing about plank's own source. `cargo fmt --check` (no `--all`)
  covers this package and stops at the submodule boundary; verified by probing
  a misformatted function into `src/` and confirming it is still caught.

  The pre-commit hook hid it: it ran `cargo fmt --all` *without* `--check`, so
  locally it silently rewrote ~59 submodule files instead of failing. Nothing
  was ever committed (only already-staged files get re-staged), but it left
  `refs/obscura` permanently dirty and meant the hook could never agree with
  CI. The hook now runs `rustfmt --edition <crate edition>` on the staged files
  only. Clippy needs no equivalent change: path dependencies are built, not
  linted.

- **A gitlink without a `.gitmodules` entry breaks every CI checkout.** The
  tree can hold a submodule commit (mode `160000`) that `.gitmodules` does not
  describe; `git submodule update --recursive`, which `actions/checkout` runs,
  then dies with `fatal: No url found for submodule path '<path>'` before a
  single line is built. `refs/openclaw` sat in that state after its stanza was
  dropped while adding `refs/edit`, and it broke both CI *and* the release
  bottle build. It had survived earlier releases only because its entry carried
  `update = none`, which made recursive checkout skip it — remove the stanza and
  the exemption goes with it.

  Local builds notice none of this: the submodule is already checked out, so
  nothing re-resolves it. Check with
  `diff <(git ls-files -s | grep 160000 | awk '{print $4}' | sort) <(grep 'path = ' .gitmodules | awk '{print $3}' | sort)`
  before touching `.gitmodules`.

- **`Color::Black` is not black.** Ratatui's `Color::Black` emits ANSI index 0,
  which is a *palette slot*, not a value: terminal themes remap it freely and
  most render it as a dark grey. Painting the screensaver's night sky with it
  produced a grey background that looked like a missing fill but was the
  terminal substituting its own colour. Any surface that must be a specific
  colour rather than "whatever the theme calls this" needs an explicit
  `Color::Rgb`. The named constants are still right for text that should adapt
  to the user's theme — the distinction is whether you are asking for a role or
  a value.

- **The TUI's ANSI parser only understands `38;2` / `38;5` colours.**
  `apply_sgr` in `src/tui.rs` handles reset, `39`/`49`, and the truecolor and
  256-colour forms; the basic and bright SGR codes (`30`–`37`, `90`–`97`) fall
  through the `_ => i += 1` arm and are silently ignored, so text styled with
  them keeps whatever colour was active. Escape sequences that must survive the
  stdout *and* TUI paths — anything routed through `OutputLog::push_ansi`, e.g.
  `status::system_line` — have to use the indexed form (white is `38;5;231`,
  not `97`).

## Part 2 — Environment & tooling

- **The Metal backend needs the macOS 15 SDK** (`MTLResidencySet`), so
  release builds run on `macos-15` runners and bottle as `arm64_sequoia`.
  The ds4 Makefile's `-mcpu=native` default is invalid for x86_64 clang and
  non-portable for bottles; override `NATIVE_CPU_FLAG` per arch
  (`apple-m1` / `x86-64-v3`).
- **Releases are Homebrew-only and the tag number is the channel.** The
  highest tagged major is beta (`plank-agent-beta` formula), everything below is
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
- **A volatile byte in an MCP tool schema rebuilds `sysprompt.kv` on every
  launch.** The system-prompt KV snapshot is fingerprinted over the whole
  prompt text, which includes every connected MCP server's tool schemas. An
  MCP server that interpolates live data into a tool *description* — the
  trigger was `tokensave_context` advertising `(520445 nodes)`, a graph-size
  counter — changes the prompt bytes each run, so the fingerprint misses and
  the (~130 MB) snapshot is rebuilt cold every start. Fix is on the server
  (keep descriptions static; put counts in tool *results*). Defensively,
  `McpServer::handshake` now sorts tools by name so a server returning
  `tools/list` in a nondeterministic order can't churn the fingerprint by
  reordering alone. Diagnose with a fingerprint/prompt diff across two
  launches; the culprit is almost always a same-length change (a fixed-width
  number ticking) mid-prompt.
- **Anthropic prompt-cache breakpoints default to the 5-minute tier.** A bare
  `cache_control: {type: "ephemeral"}` expires after 5 minutes, so an
  interactive turn taken more than 5 minutes after the last one loses the
  cached system+tools prefix. `remote/provider.rs` requests the 1-hour tier
  (`ttl: "1h"` plus the `anthropic-beta: extended-cache-ttl-2025-04-11`
  header); it costs 2× base input on the cache *write* but keeps reads at
  0.1×, a clear win when the prefix is re-read far more than it changes.

- **`refs/edit` needs Rust 1.93+.** `edit`, `lsh`, and `stdext` all declare
  `rust-version = "1.93"`, and on 1.91 `stdext` genuinely fails to build
  (`maybe_uninit_slice`, `vec_into_raw_parts`). CI's `stable` toolchain is
  fine; a stale local `rustup` is not. Two further gotchas: `stdext`'s scratch
  arenas are process-wide `static mut` singletons, so miniedit may only be
  driven from the TUI thread, and `arena::init` must run once before any
  `TextBuffer` exists. Search goes through ICU loaded at runtime, so it must
  degrade to "unavailable" rather than fail the session.

- **Three `edit` TUI traps, all found by driving the real binary.** Its TUI is
  immediate mode, and each of these fails *silently* — the editor renders fine
  and simply ignores you. (1) The first focusable widget takes the focus, which
  is the menubar, so the text area has to claim it on the first frame or every
  keystroke lands on a closed menu; the input is read and parsed, it just has
  nowhere to go. (2) An `editline` collapses to zero width without an explicit
  `attr_intrinsic_size(COORD_TYPE_SAFE_MAX, 1)` after it, so a search panel
  renders as a bare label with nothing to type into. (3)
  `TextBuffer::save_as_string` calls `mark_as_clean`, so reading the text out
  clears the dirty flag — "has this been edited?" has to compare against the
  original string, or a discard prompt stops appearing the moment anything
  reads the buffer. `PLANK_MINIEDIT_DEBUG=<file>` logs session start/end and
  every stdin read, which is how (1) was pinned down: the bytes arrived, the
  screen never changed.

- **A screensaver's idle clock must ignore focus and resize events.** Treating
  every terminal event as "the user is here" means it never fires on a desktop
  where anything moves focus around — a tiling WM, a notification, an agent
  driving the terminal. Only keys, mouse and pastes count. Symptom: the idle
  timer visibly resets every few seconds with nobody touching the keyboard.

- **`ratatui-markdown` code-block headers can't be customized via the block's
  `header_override`.** When a `RenderHooks` impl (plank uses `HighlightHooks`)
  returns `Some` from `render_code_block`, the crate renders the whole block —
  header, body, footer — from that one call and `return`s before it ever
  consults `header_override`/`code_block_header`. So injecting a control (the
  `⧉ copy` affordance) into the language-label row by mutating the parsed
  `MarkdownBlock::CodeBlock` is silently ignored for any block the highlighter
  recognizes. The reliable seam is *after* `md.render`: scan the rendered
  `Line`s (`╭` header → `╰` footer, body rows carry a `│ ` gutter) and append
  the control span there. That scan also recovers the block's raw text WYSIWYG
  (strip the `│ ` gutter, trim trailing space) — the same philosophy as
  `selection_text`, and it needs no back-reference to the markdown source,
  which `OutputLog` discards once a streaming segment closes.

- **A single sampled token's detokenized bytes are not necessarily valid
  UTF-8.** DeepSeek's byte-level BPE splits multi-byte characters (emoji,
  CJK) across tokens — 🦀 (`F0 9F A6 80`) commonly arrives as `F0 9F` in one
  token and `A6 80` in the next. Calling `String::from_utf8_lossy` per token
  turns each fragment into replacement characters (rendered as `???` in the
  output window) even though the concatenated byte stream is perfectly valid.
  Decode across tokens: `ds4_token_text` output is carried through
  `engine::Utf8Stream`, which emits only the complete UTF-8 prefix and holds
  an unfinished trailing sequence (≤3 bytes) for the next token, flushing
  lossily only at end of generation. The same applies to any byte-chunked
  stream (EchoEngine's 8-byte chunking deliberately splits a 🦀 to keep the
  stub honest); `viz::StreamRenderer` already had its own carry for the same
  reason.

- **The rendered transcript must be append-only — never inject or rewrite
  anything between the system prompt and the newest message.** The C keeps
  `w->transcript` as an append-only *token* buffer, so
  `ds4_session_common_prefix` always reaches the previous turn's end and only
  the new suffix is prefilled. Plank re-renders the transcript from text each
  turn, so the same invariant must hold on the rendered bytes: any
  mid-transcript mutation moves the first divergent token to that point and
  everything after it is re-prefilled. Issue #35's task list broke this by
  injecting a fresh `[user]` task block right after `[system]` every turn — one
  `task add`/`update` changed the tokens at the very top and the *entire
  conversation* re-prefilled on every subsequent turn. The fix: mutating `task`
  ops append the current list to their tool observation (append-only by
  construction), and the block is re-injected only inside
  `rebuild_after_compact`, where the KV prefix is already invalidated. Apply
  the same rule to any future "always visible" state: piggyback it on appended
  messages, or accept a full re-prefill.

- **The TUI must not re-render streamed markdown on every token — code-block
  syntax highlighting is not free.** `OutputLog::visible_text` (`src/tui.rs`)
  re-parses and re-renders the whole in-progress segment on each append so
  partial fences/emphasis resolve as more text arrives. That is fine for prose,
  but a fenced code block routes through `ratatui-markdown`'s tree-sitter
  highlighter, whose `TreeSitterHighlighter::highlight` recompiles the
  highlight query (`HighlightConfiguration::new`) on *every* call — no
  per-language cache, ~44 ms per render for a Rust block in a debug build. Per
  token that is O(tokens²) with a large constant: the UI thread never yields
  back to paint and the whole TUI wedges at 100% CPU the instant a code block
  streams (looks like a deadlock; it is a livelock). Fix: `md_render` is
  throttled to at most once per `MD_RENDER_MIN_GAP` (100 ms) while streaming,
  with a guaranteed `flush_md` at every segment boundary (`md_close`,
  `end_line`) so no tail tokens are lost; `md_close` resets the throttle so a
  new segment's first token still renders immediately. The upstream recompile
  is the real bug (celestia-island/ratatui-markdown#18; already fixed on their
  unreleased `master`, which also relicensed to SySL-1.0 — watch that before
  upgrading). The throttle is worth keeping regardless: it bounds cost for
  large blocks even once the config is cached.

- **An offline MCP shadow must be substituted in place, not appended.**
  `append_tool_schemas` (`src/tools/mcp.rs`) iterates `servers` in order, so a
  server's index is part of the system prompt bytes and therefore part of `fp1`.
  When `start_servers_with` replaces a failed global server with a
  cached-advertisement shadow (`McpServer::offline`), pushing that shadow at the
  end of the vec instead of at the failed config's own index yields a reordered
  prompt that matches no `sysprompt-*.kv` snapshot — while every test that only
  checks "the tools are present" still passes. Verified the hard way: building
  the append-at-end variant makes
  `a_shadow_takes_the_failed_servers_place_in_order` fail with `["a", "c"]` vs
  `["a", "b", "c"]`, which is why that test asserts the name order *before* it
  compares prompt bytes.

- **`append_resource_tool_schemas` must not gate on `alive`.** The gate was
  `s.alive && !s.resources().is_empty()`. Every server is alive when the prompt
  is first built — failures are dropped before that point — so the `alive` term
  never did anything useful, but it silently removed the `mcp_list_resources` /
  `mcp_read_resource` schemas from Tier 1 whenever the prompt was rebuilt after
  the only resource-bearing server died, moving `fp1` for a reason that has
  nothing to do with the tools the model can actually use. An offline shadow
  carries its cached `resources` precisely so those two schemas stay, so the
  gate is presence of resources alone. `build_tools_prompt(&[])` is unaffected
  either way, so the C-parity fixtures do not move.

- **`｜DSML｜` is a dedicated vocab token in C, but plain characters in plank —
  hence the `SSML` misspelling.** `ds4_agent.c:986-990` tokenizes the tools
  prompt with `ds4_tokenize_rendered_chat` explicitly "so the literal ｜DSML｜
  markers in the examples become the model's dedicated DSML token"; the C
  asserts the marker is one id at `ds4_agent.c:7408`. plank has no binding for
  `ds4_tokenize_rendered_chat` (`src/ffi.rs`) and composes the tools prompt as
  an ordinary `system` message, so the marker arrives as ordinary BPE pieces.
  The model then spells it back out at generation time, and the "D" is just
  another sampled character — with `SSML` (Speech Synthesis Markup Language) a
  far more common pretraining string. Repro
  `~/.plank/repro/repro-1785161356.md`: after ~18 correct calls, one came back
  as `<｜SSML｜tool_calls>` with every other byte right. `engine.thinkingToolCalls`
  amplifies it: stripping `IN_THINK_PROHIBITION` puts every tool call
  off-distribution inside `<think>`, which flattens the distribution over those
  spelled-out pieces. `src/ds4engine.rs:517` notes that a re-appended assistant
  section retokenizes from text, so a compaction or resume converts the whole
  call history from control-token DSML to the spelled-out form at once.
  Mitigated, not fixed, by `dsml::MARKER_NAMES`: `SSML` is accepted as a parse
  alias so the call dispatches instead of printing raw and silently ending the
  turn with no error for the model to retry from, and `MARKER_SPELLING_NOTE`
  tells the model the spelling is unsupported. The real fix is binding
  `ds4_tokenize_rendered_chat` (public at `refs/ds4/ds4.h:203`) for the tools
  prompt and the reminder. Note the alias is deliberately narrow — only the one
  observed misspelling, not any four letters — so prose cannot open a stanza.

- **liteparse's `quiet` does not silence the bundled Tesseract.** The flag
  only gates the crate's own Rust-side logging; Tesseract 5.3.4's
  `tprintf("Detected %d diacritics", …)` (`textord/strokewidth.cpp:381`,
  unconditional on the `PFR_NOISE` path) writes to fd 2 through C stdio,
  which no Rust-side flag reaches. Because plank parses in-process, those
  bytes land wherever the terminal cursor happens to be — the TUI's prompt
  line. Any C library can do this; the one lever that covers both Rust and C
  writers is `dup2` on the fd itself. `StderrSilencer` in `src/doc/mod.rs`
  routes fd 2 to `/dev/null` around the parse, mutex-serialized so
  overlapping guards cannot restore in the wrong order and strand the fd.
  The regression test (`doc::tests::parser_diagnostics_never_reach_stderr`)
  converts a noisy scanned fixture (`tests/fixtures/doc_noisy.pdf`) in a
  subprocess — in-process fd capture races across parallel test threads —
  and asserts nothing but a post-conversion sentinel reaches stderr.

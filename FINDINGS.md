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
  *Instance 3 — the subagent fork.* The fork truncated the sidechain out of the
  text transcript but left the live KV holding parent+sidechain, so the very
  next turn's prompt (parent prefix + the small framed report) diverged behind
  the live end and the **whole parent context re-prefilled from zero** — the
  slowest possible turn, every time a sub-agent finished. The fix follows the
  `/btw` aside precedent instead of fighting the rule: `begin_subagent_fork`
  captures a real frontier snapshot (`engine.get_kv()`), and every fork-end
  path (`finish_subagent_fork`, plus the `agent` tool's inline truncation)
  restores it (`engine.set_kv()`), turning the post-fork sync back into an
  extend that prefills only the report. The guard is a *stack*
  (`Agent::fork_kv`) because `/subagent` turns are interactive and can nest;
  engines without snapshot support (Echo, scripted test doubles) return `None`
  from `get_kv` and keep the old re-prefill behavior. Unit tests can only
  assert the capture/restore *calls* — per the note above, the actual token
  savings need a real model or the `PLANK_KV_DEBUG` trace to observe.
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

- **The stanza opener can carry a trailing `｜` too.** Post-update weights emit
  `<｜DSML｜tool_calls｜>` where the prompt teaches `<｜DSML｜tool_calls>` — the
  same optional bar the *closing* tags have always tolerated, now on the opener.
  Because no opener form matched, the stanza never opened: the whole tool call
  streamed out as prose and the inner `<｜DSML｜invoke` tripped the loose-marker
  detector, so every turn ended with "DSML markup outside a valid tool_calls
  block" and the model had no idea which part of its syntax was rejected. It is
  a wrapper-only quirk; `invoke` and `parameter` openers were unaffected. The
  accepted forms live in two places that must stay in sync — `dsml_start_match`
  in `src/viz.rs` (the streaming detector, which seeds the parser with canonical
  bytes) and `DSML_START*` / `find_tool_start` in `src/dsml.rs`.

- **Post-update weights also write the parameter name as the element name.**
  `<｜DSML｜command string="true">ls</｜DSML｜invoke>` in place of
  `<｜DSML｜parameter name="command" string="true">ls</｜DSML｜parameter>`. The C
  reference errors on this (`unexpected DSML tag`, `ds4_agent.c`, the `else`
  arm of `agent_dsml_parse`) and has no fix upstream — checked at `80ebbc3`,
  which is `origin/main`. Rejecting is not neutral: the recorded repro shows the
  model unable to work backwards from an error that only echoes the tag, so it
  blamed the marker spelling twice and then emitted DSML inside `<think>`,
  losing the turn. plank accepts the shorthand, but narrowly — only inside an
  already-open invoke, only for a DSML-marked tag whose element name is a plain
  identifier, and only when it carries no `name` attribute (a tag with one is a
  different malformation and still errors).
  The close tag is the trap: the model ends the shorthand with
  `</｜DSML｜invoke>`, not `</｜DSML｜command>`. Widening the value terminator to
  match is why the widening is confined to shorthand parameters via
  `param_elem` — a canonical parameter keeps the strict `parameter`-only
  terminator, so a `write` payload containing `</｜DSML｜invoke>` (this repo's own
  sources and docs do) is still never truncated. Do not lift that restriction.

- **`DS4_THINK_MAX` was in the C all along; plank collapsed it away.** The C's
  `ds4_think_mode` has three levels (`NONE`/`HIGH`/`MAX`), but plank's
  `ThinkMode` had `Auto`/`On`/`Off` where `Auto` and `On` both mapped to `HIGH`
  and `MAX` was unreachable. Exposing the third level (`/think off|medium|max`)
  therefore needed **no parity break**: the preamble text is the C's own
  `DS4_REASONING_EFFORT_MAX_PREFIX`, checked byte-for-byte by `c_parity.rs`.
  Two things about it are load-bearing:

  *Position.* The preamble is tokenized as **plain text with no role wrapper**,
  immediately after `ds4_chat_begin` and **before** the system message — see
  `encode_chat_prompt` and the REPL's `repl_chat_apply_max_prefix`, which
  inserts it at transcript index 1 (past the BOS). Folding it into the system
  prompt *string* instead looks equivalent and is not: it would land after the
  system role marker and diverge from the trained prefix.

  *Cache identity.* Because it sits above the system prompt, the reasoning
  level is key material for Tier 1 — `system_fingerprint` hashes
  `model ‖ NUL ‖ think ‖ NUL ‖ system`. Without it a `max` checkpoint would be
  restored under `medium`. (Adding the field changed the layout, so existing
  `sysprompt-*.kv` files miss once and are rebuilt.) Changing the level in or
  out of `max` invalidates the live KV and the token transcript; changing
  between `off` and `medium` costs nothing, because that difference lives
  entirely in the per-turn assistant prefix.

  The C *downgrades* `max` to `high` below a 384K context
  (`ds4_think_mode_for_context`); plank refuses instead, at both `--think-max`
  and `/think max`, so the user learns the level did not take effect.

- **The in-think tool-call verdict belongs at the stop token, not the opening
  marker.** The original rule rejected a stanza the moment its opening marker
  appeared inside `<think>`, and separately raised the prohibition at stream end
  whenever `note_thinking_dsml_byte` had seen *any* DSML-shaped bytes while
  thinking. Both are too eager, because a model reasoning about its own tool
  syntax writes that syntax as part of the thought. `repro-1785754509.md` is the
  case: the model quoted an opening `<｜DSML｜tool_calls>`, closed the thinking
  block, emitted a correct call after it — and got back "tool calls are not
  allowed inside `<think>`", which is advice about something it had not done, so
  it rewrote correct markup and looped.

  An opening marker is only a *candidate*. The model has not called a tool until
  the stanza reaches its stop token, so that is where `rejects_in_think` is
  evaluated, against the think state at that instant.

  Two things had to move with it:

  *`</think>` must be recognized while a stanza is open* — it was gated on
  `!dsml_active`, so a stanza opened in thinking could never observe the close
  and stayed "in think" for the rest of the stream. But only where it cannot be
  data: inside a parameter value `</think>` is payload (a `write` of any document
  discussing thinking blocks contains it), so the gate is
  `parser.state() != ParamValue`, not `!dsml_active`. Getting this backwards
  silently corrupts written files.

  *Rendering is deferred, not just the verdict.* A stanza opening in thinking is
  tracked but not drawn; the banner starts if `</think>` arrives with it still
  open. Drawing optimistically would flash a tool banner every time the model
  quoted a marker mid-thought.

  Still unfixed, and separate: a *shorthand* opener (`<｜DSML｜invoke …>`) quoted
  inside thinking opens an implicit stanza that never closes, so the rest of the
  turn is swallowed as stanza content. That predates this change and is not
  distinguishable from a real call at the point it matters.

- **The shorthand has two levels, and only one of them was accepted.** The
  parameter form (`<｜DSML｜command …>` for `<｜DSML｜parameter name="command" …>`)
  was tolerated; the identical rewrite one level up was not. In
  `repro-1785754509.md` the model wrote the *tool* name as the element name —

  ```
  <｜DSML｜tool_calls>
  <｜DSML｜edit>
    <｜DSML｜parameter name="path" string="true">…</｜DSML｜parameter>
  </｜DSML｜invoke>
  </｜DSML｜tool_calls>
  ```

  — five times, for `write` and `edit`, and got `unexpected DSML tag` every
  time. The stanza is otherwise flawless: right wrapper, canonical parameters,
  and it closes with `</｜DSML｜invoke>`, which is the model's own tell that it
  meant an invoke. It never recovered, and eventually broke the think gate
  restating the syntax to itself.

  Telling the two shorthands apart needs exactly one bit: **is an invoke open?**
  Before one, a bare element names the tool; inside one, a parameter. So
  `shorthand_param_name` is checked first and `shorthand_invoke_name` second,
  and each guards on the opposite state of `self.current`. Both stay narrow the
  same way — DSML marker present, element name a plain identifier, no `name`
  attribute — because accepting either means *running* a call the prompt never
  taught. A bogus element name is safe: it reaches dispatch and fails there by
  tool name, which the model can act on.

  `viz.rs::scan_dsml_tag` mirrors the parser for rendering and had to learn both
  forms too, keyed on `viz.tool_announced` (that side's copy of "an invoke is
  open"). Without it a shorthand call ran with no banner, or rendered its tool
  name as a parameter — worse than the rejection it replaced.

  Still unaccepted, seen once in the same log: `<｜DSinvoke name="bash">`, the
  marker itself corrupted (`DSML｜` → `DS`). That is token damage rather than a
  syntax variant, and widening `MARKER_NAMES` to catch it would start matching
  prose.

- **The system prompt must be tokenized in two halves, and plank was doing it in
  one.** `｜DSML｜` is not a markup convention — it is a token in the model's own
  GGUF vocabulary, looked up at load time next to BOS/EOS/`<｜User｜>` and fatal
  if missing (`ds4.c:22272`). But `ds4_chat_append_message` tokenizes a `system`
  message with plain `bpe_tokenize_text`, *no* special-token splitting. So every
  marker in the tools prompt reached the model as ordinary BPE pieces, and the
  model read a spelled-out marker and wrote a spelled-out marker back — which is
  where `<｜DSinvoke name="bash">` and the `SSML` misspelling come from.

  The C already solves this. `agent_append_system_prompt` splits the prompt and
  routes the halves differently, with the rule in a comment:

  > The built-in tool prompt is trusted DS4 control text. Tokenize it like a
  > rendered chat prompt so the literal ｜DSML｜ markers in the examples become
  > the model's dedicated DSML token. Do not apply that tokenizer to user
  > supplied -sys text: arbitrary user text containing `<｜User｜>`, `<think>`,
  > or `｜DSML｜` must remain plain content, not control tokens.

  That second sentence is a **prompt-injection boundary**, and plank's is wider
  than the C's: the C's tools prompt is entirely built in, while plank appends
  MCP tool schemas and server instructions to it. Those come from third-party
  processes, so they sit outside the trusted span alongside `-sys` text. Every
  `｜DSML｜` the prompt teaches is in the built-in part, so nothing is lost by
  drawing the line there — see `SplitSystemPrompt::trusted_len`. Widening that
  span would let an MCP server forge a turn boundary.

  Two traps in implementing it:

  *Two paths tokenize the system prompt, not one.* The warm walk builds it via
  `build_system_tokens`, and `reconcile` rebuilds it every turn as section 0 of
  the rendered transcript. If they disagree by a single token the KV
  common-prefix probe stops right after the system prompt and each turn
  re-prefills the whole conversation — silently, since nothing errors. Both now
  go through `append_system_text`, and a model-gated test asserts they agree.

  *The split changes the tokens without changing the text.* `system_fingerprint`
  keys the Tier 1 checkpoint on the prompt's bytes, which are identical before
  and after, so a checkpoint written under the old scheme would have been
  restored under the new one and prefilled against a KV that no longer matches.
  `trusted_len` is therefore key material too.

  Measured on the real model (`PLANK_TEST_MODEL`): 3263 → 3213 tokens, with 16
  native `｜DSML｜` tokens where there were previously none.

- **A valid deep tier means the tier above it is never written.** `kvtier::warm`
  restores the *deepest* checkpoint that loads and then skips every tier above
  it — correct for restoring, because each tier's fingerprint chains its
  parent's, so a deep hit proves the ancestors match. The consequence is easy to
  miss: a tier that is skipped is never prefilled, and a tier that is never
  prefilled is never *persisted*. Once Tier 2 (`project-*.kv`) is valid, Tier 1
  (`sysprompt-*.kv`) stops being written, and if it was never written before
  that it never will be.

  This is invisible for the main engine, which restores Tier 2 — a superset of
  Tier 1 — and is strictly better off. It surfaced only when a `provider: local`
  sub-agent needed Tier 1 *alone*: a sidechain runs clean-room, so its prompt is
  the system prompt plus the framed task with no project or session context
  between them, and restoring Tier 2 would seed the KV with tokens its prompt
  does not contain. Every first dispatch prefilled the whole system prompt, on a
  machine carrying a healthy 210 MB Tier 2 checkpoint and not one
  `sysprompt-*.kv` anywhere.

  The fix is a tier list of **one**: warm the alt engine with only the system
  tier, so there is nothing deeper to short-circuit it and the checkpoint
  actually gets built. Still open for the main engine — if its Tier 2 is ever
  invalidated it rebuilds from token zero rather than restoring Tier 1, and
  fixing that needs a second snapshot taken at the Tier 1 boundary, since a
  snapshot captures the whole session and can only be trusted at the boundary it
  was taken on.

  Two related traps, both of which cost a debugging round here:

  *An engine's warm buffer is built from its own fields.* `warm_reset` calls
  `build_system_tokens(system, self.trusted_system_len, self.think)`, so an
  engine that never had `set_trusted_system_prefix`/`set_think_mode` applied
  tokenizes the same system text differently from the one that wrote the
  checkpoint. The restore then loads a KV that the token buffer does not
  describe, the first common-prefix probe truncates back, and the prefill you
  were avoiding happens anyway — a hit that buys nothing and reports success.
  Any engine held outside `self.engine` needs the same configuration the main
  one gets.

  *`warm_sync` cannot be interrupted* (`interrupt: &|| false`) and reports
  progress only through the sink it is handed. Warming on the thread a front end
  draws on, with a no-op sink, is a hard freeze for the length of a cold system
  prompt. Warm at startup where a prefill is expected and drawn, or restore only
  and leave the prefill to the pass that needs it.

  *The checkpoint GC assumed one engine.* `gc_system_checkpoints` deletes every
  `sysprompt-*.kv` except the fingerprint it is told to keep, and it was told
  the main engine's. With two engines live that deletes the other's on every
  launch — and under a provider main agent it deletes *all* of them, since the
  provider's own fingerprint never has a file and so nothing matches the keep.
  That, not the tier-skip alone, is why the cache directory held zero system
  checkpoints. A collector that keys on "the current one" needs every live key,
  not the first.

  Corollary for debugging any of this: a silent hit and a silent miss look
  identical. `kvtier::Restored` names which of the four things happened and the
  callers print it with the fingerprint and the exact path, because two rounds
  were spent guessing at it.

## Part 2 — Environment & tooling

- **Bumping `refs/ds4` is three coupled edits, not one.** `ds4_engine_options`
  is mirrored field-for-field by `ffi::Ds4EngineOptions`, and the mirror is
  positional: a field added mid-struct in the C shifts everything after it and
  the mismatch is silent — no compile error, just an engine configured from the
  wrong bytes. Re-read the C struct top to bottom on every bump. `build.rs`
  carries its own copy of the Makefile's `CORE_OBJS` list, so a new translation
  unit there (`ds4_tp.o`, `ds4_layer_pack.o` arrived together) surfaces only as
  undefined symbols at link time. And the prompt constants drift: `cargo test
  --test c_parity` is the check, `PLANK_REGEN_FIXTURES=1` the fix, but only
  after confirming the C's new text is text plank should actually be sending.
- **Speculation needs its own entry point; the option struct is not enough.**
  Everything `--dspark` touches in `ds4_engine_open_internal` and
  `ds4_session_create` is allocation and setup: the support GGUF loads,
  target-hidden capture turns on for layers 40-42, drafts get prepared. The
  accept/rollback loop that *consumes* drafts lives in exactly one exported
  function, `ds4_session_eval_speculative_argmax`, which takes a sampled
  `first_token` and returns the committed run with that token at index 0.
  `ds4_session_eval` advances one token and can never accept a block, so a
  loop built on it gets zero benefit no matter how the engine is configured.
  Diagnostic: `DS4_DSPARK_STATS=1` printing *nothing* means the speculative
  path never ran — it is not the same as poor acceptance, which prints
  counters. The C gates the call on `temperature <= 0` (verification is
  argmax) and `ds4_engine_mtp_draft_tokens(e) > 1`; `ds4engine.rs` mirrors
  both. The committed run is already in the KV cache when the call returns,
  so nothing downstream can reject part of it — which is why think-recovery
  now runs once per block instead of once per token.
- **`DSpark` on Metal went from 0.71x to 1.19x on one upstream commit — pin the
  engine before quoting any number.** Through the whole M5 fusion work it was a
  clear loss on M5 Max at IQ2XXS: paired plank replicas read `19.84 t/s`
  target-only against `13.99 t/s` with `--dspark`, and `DS4_DSPARK_STATS=1`
  said why — `accept_rate=74.13%` looked healthy, but `verify=107s` and
  `replay=134555ms` against `saved=148784ms` left `net_saved=-125652ms`.
  Then `42033ee metal: pipeline `DFlash` verification` (submit tiny
  target-verifier batches incrementally, and retune the Metal confidence
  default to 0.6) flipped it: `19.84` against `23.63 t/s`, 1.19x, with wall
  clock agreeing at 0.81x and `--dspark` winning every pair. Nothing in plank
  changed between those two measurements — the same binary, prompt, and model.
  Two lessons. `net_saved` is the number to read, not `accept_rate`, which
  looked fine in both regimes. And a `DSpark` verdict is only ever true of one
  engine commit, so record the SHA next to the figure.
  **Confirm against `./ds4` before suspecting plank.** Building the reference
  CLI out of the submodule (`make ds4`) and running the same prompt and model
  reproduced the loss era exactly — `21.92 / 21.78 t/s` target-only against
  `15.79 / 15.44` with `--dspark`, the same 0.71x plank showed. Two binaries
  agreeing rules the port out in one step, and it is much cheaper than
  reasoning about the FFI.
- **`--dspark-confidence` has no fixed default, so plank does not name one.**
  The engine picks `METAL ? 0.6 : 0.7` and *ignores the value plank passes*
  unless `dspark_confidence_threshold_set` is true — which is why the field is
  `Option<f32>` (`0` means "fixed five-token blocks", not "unset"). A mirrored
  constant went stale the moment upstream retuned Metal from 0.7 to 0.6, so
  there is deliberately no `DSPARK_CONFIDENCE_DEFAULT` in `config.rs`: the
  unset case passes a placeholder the engine discards.
- **Do not time this with one-shot `./ds4 -p`; use `ds4-bench`.** Ad-hoc CLI
  timings of a long generation swing wildly on a laptop — the same
  target-only command measured 34.01, 21.92 and 21.78 t/s across three runs,
  and an early cold-cache run read 11.52. Any before/after conclusion drawn
  from single runs like that is noise. The fork's contract is three
  independent `ds4-bench` processes with the *Promessi sposi* fixture at a
  fixed 2048-token context (the exact command is in
  `speed-bench/ds4_m5_fusion_port_results.md`), reporting median and min-max.
  On that instrument this machine reads decode `42.03 t/s` median
  (42.86/41.86/42.03) and steady `42.31`, against the fork's documented
  `39.39` median baseline at `caf64d1` — so the M5 fusions are engaged and
  worth about +7%. Paired A/B on one instrument beats absolute numbers across
  two.
- **plank's greedy output is not bit-reproducible, and it is not `DSpark`.**
  Across twelve `--temp 0` runs of one prompt, ten produced the same output
  and two diverged from the first token — one under `--dspark`, one under
  target-only decode, so the earlier reading that blamed speculation was an
  artifact of small samples. Both outliers were the first run after something
  invalidated a cache (a rebuild, a prompt change), which points at prefill
  chunking changing float accumulation order and flipping a near-tie token,
  not at the accept path. Divergent outputs are coherent prose, never
  corruption. Before blaming the port for any of this,
  `ffi::tests::engine_options_*_match_the_c_layout` pins all 42 field offsets
  and the struct size against `offsetof` on the C header.
- **The C's prompt constants can hide behind `#define`s.** `ds4_agent.c` split
  the editing section into `agent_tools_prompt_edit_exact` (its default) and
  `agent_tools_prompt_edit_upto`, sharing a sentence through
  `AGENT_EDIT_TARGET_RULE`. plank ships the `[upto]` variant, because its edit
  tool implements the anchor. `tests/c_parity.rs` expands object-like string
  macros before decoding literals; a literal decoder alone chokes on the bare
  macro name.
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

- **…and it must be stamped when the event finishes being handled, not when it
  arrives.** The Enter that submits a prompt is a user event, but the loop does
  not come back to it until the whole turn is done — minutes later. Stamping on
  arrival means the delay has already elapsed by then and the stars come up the
  instant the turn ends. `tui_loop` stamps twice: once on arrival (covering the
  short paths that `continue` out of the match) and once after the match, so a
  long turn restarts the countdown from idle. Remote-driven turns and the
  `--prompt` startup turn stamp at their own call sites for the same reason.

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

- **The agent's own tool harness has no write-lock around file mutations —
  concurrent tool invocations can interleave their outputs into shared state.**
  Discovered while drafting a Medium post via the `medium_create_draft` MCP tool:
  the `write` tool and `bash` fired in the same frame, and the bash output
  (`wc -w` results) landed as literal bytes inside the file content being
  written, including the DSML tool-call syntax that invoked the bash command.
  The corrupted file then caused `edit` to fail (old text anchor no longer
  matched) and `read` to return a file that contained raw DSML markers as
  content rather than Markdown. Three distinct failure modes from one root cause:

  1. **No mutual exclusion between write and bash.** The harness dispatches all
     tools in a single frame without ordering guarantees. A `write` that creates
     or overwrites a file and a `bash` that reports on it race: the bash output
     stream lands inside the write's byte stream, producing a file whose content
     is the intended text plus an embedded tool-call stanza and its output.
  2. **Bash heredoc captures DSML syntax as literal stdin.** Using
     `cat << 'POSTEOF'` heredocs to append to a file, the bash process receives
     DSML markers (`<｜DSML｜tool_calls>`, `</think>`) as literal stdin bytes.
     The system then tries to parse those stdin bytes as tool calls instead of
     file content, producing parse errors and further corrupting the file.
  3. **Write-tool content truncated by concurrent bash.** The `write` tool's
     content parameter is a single string; when a bash job in the same frame
     produces output before the write completes, the write's byte stream is
     interleaved with the bash output at the OS level, truncating the written
     content at the point of interleaving.

  The fix for the draft was to use Python for file writes (single-threaded, no
  concurrent output to corrupt) instead of the `write` tool or bash heredocs.
  The root cause — no write-lock around file mutations in the tool harness —
  is unresolved in the agent itself. Any tool that writes a file and any tool
  that reads or reports on that file must not be dispatched in the same frame,
  or the harness must serialize them. The same class of bug can affect any pair
  of tools where one produces output that another consumes as input: `edit` +
  `bash`, `write` + `read`, `write` + `edit`, and any MCP tool that writes a
  file followed by a built-in tool that reads it.

- **The empty-payload defect in `finish_ignored_dsml` is real and fixed** — in-think
  rejections are the most common entry in `~/.plank/tool-call-errors.log` (969
  recorded) and were logging `self.parser.raw()` at a point where it has
  usually been drained, so the most frequent failure carried no evidence. It
  now falls back to the held `dsml_start_tail` bytes.

  The fallback is not total: when no `<`-anchored tail was being held, the
  logged payload is still empty.

  The misdiagnosis was **hypothesized and not reproduced**. `note_plain_dsml_byte`
  guards on `dsml_in_think`, which is sticky for the rest of the stream, so
  *that* call site cannot report `malformed_dsml` again once an in-think stanza
  is seen. A guard added there was removed as dead code. The pseudo-tool
  detector (issue #51) later added a second `malformed_dsml` call site which is
  deliberately *not* guarded on `dsml_in_think`: the whole point is to catch
  the model falling back to invented `<task>` markup in its answer after its
  in-think DSML attempt was discarded. That path reports the invented markup,
  not a parse verdict on the discarded stanza, and `finished()` lets it outrank
  the in-think prohibition, so it is not a route back into the misdiagnosis.

  The methodological lesson, which is the part worth cataloguing: the
  hypothesis came from reading `~/.plank/tool-call-errors.log` and seeing
  "DSML markup outside a valid tool_calls block" adjacent to in-think
  rejections 402 times, but those clusters were `echo plank-e2e` test-fixture
  runs, not real sessions. The log was filtered for fixtures when counting,
  and not when reasoning about adjacency. Note also that the log cannot see
  hallucinated non-DSML markup at all, since `log_tool_error` is only reached
  from paths that already classified something as tool markup.

- **Two test shapes that are flaky against a live loopback server.** Both bit
  the `/remote-control` work; the second bit it twice, the second time after a
  fix agent had measured 30 clean runs of the test in isolation.

  *Never probe whether a port is still bound after the server shut down.* The
  server binds `127.0.0.1:0`, so `remote_off` releases an ephemeral port that
  another test in the parallel suite can be handed immediately — the probe
  connects, succeeds, and the assertion fails for a reason that has nothing to
  do with the code under test. It passes in isolation and on an unloaded
  machine, which is what makes it expensive: it fails in CI, or on the run
  where you are trying to diagnose something else.

  *Never assert an `Arc` refcount in a test that also opens a connection to the
  same server.* Counting references to the shared state is the right way to
  prove a shutdown actually dropped the server rather than leaking it — a
  `std::mem::forget(server)` passes every other assertion. But a connection
  makes the accept loop spawn a handler thread that clones the same `Arc`, and
  that thread can outlive `remote_off`, so the count is above 1 for a reason
  unrelated to leaking. The fix is not a sleep or a retry: put the two
  properties in separate tests, one that connects and never counts, one that
  counts and never connects (`remote_on_installs_a_bridge_and_remote_off_tears_it_down`
  and `remote_off_drops_the_server_rather_than_leaking_it` in `src/ui.rs`).

  The general shape: a test whose failure mode is *another test's timing* will
  not reproduce under `cargo test --lib <name>`. Verify this class by running
  the full suite in a loop ten-plus times, not the test alone — and verify the
  assertion still discriminates by mutating the code it guards, since the
  obvious repair for flakiness is an assertion that can no longer fail.

- **A terminal grid only survives width-1 glyphs, and the katakana come in both
  widths.** The matrix rain (`src/arcade/matrix.rs`) paints one glyph per cell,
  so a single full-width character shears every column to its right for as long
  as it is on screen. The kana used here are the *half-width* forms
  (U+FF66..U+FF9D), which `unicode_width` reports as 1 and which look identical
  at terminal sizes; the full-width block (U+30A1..) reports 2 and must not be
  used. `every_glyph_is_one_cell_wide` asserts it for all three alphabets,
  because the failure is invisible in a diff and obvious only on screen.

  The second half of the same problem is not solvable in code: a font that has
  no kana draws boxes, and nothing in the program can tell. Hence `c`, which
  cycles the rain to binary and then to ASCII — an escape hatch in the UI
  rather than a probe that cannot work.

- **The real TUI *can* be driven headlessly — with a pty, not `script`.**
  `tests/ui_remote.rs` notes that `tui_loop` is hardwired to
  `Terminal<CrosstermBackend<Stdout>>` and so cannot take a `TestBackend`, but
  the loop runs fine against a pty, and `--ui-remote=PORT` then drives it
  end-to-end. Two traps make the obvious attempts fail:

  - `script -q /dev/null plank --ui-remote=…` dies with `tcgetattr/ioctl:
    Operation not supported on socket` the moment its own stdin is not a tty —
    which it never is when launched from a tool runner. Use `pty.openpty()`
    from Python (or any direct pty spawn) instead.
  - stdin must stay *open*, not merely exist. Pointing it at `/dev/null` EOFs
    immediately, which the key loop reads as Ctrl-D and exits cleanly — the CRT
    power-off frames in the captured output are the tell.

  Also drain the pty master, or a chatty frame stream eventually blocks the
  child. With those three in place, `{"cmd":"keypress"}` → `{"cmd":"snapshot"}`
  is a genuine end-to-end check of key handling, layout and highlighting; strip
  the SGR escapes and diff the last ~20 rows, since the banner logo dominates
  the rest.

- **A terminal has no alpha, but it has three things that add up to one.** The
  minions screensaver (`src/arcade/minions.rs`) needed a sprite to fade — up
  when the screen opens, down into its reflection — over a layer that may be
  live model output. What works, in the order it matters:

  *Not drawing is the only real transparency.* Every glyph painted over live
  output **replaces** the character under it, so a cell dimmed to near-black
  does not fade, it punches a hole in the text that is invisible on black and
  obvious over a transcript. Below `FAINTEST` the cell is skipped instead.

  *The block-element ramp is an alpha channel.* `█ ▓ ▒ ░` cover a known
  fraction of a cell and the terminal composites them against whatever is
  behind for free, so an ink's place on that ramp is its opacity and fading is
  walking down it. This is what rounds a sprite's shoulders without a second
  colour, and it costs nothing.

  *Shapes cannot use it.* A goggle rim or an eye is a glyph whose identity is
  its outline; at quarter coverage it is not a fainter rim, it is a different
  character. Those fade toward the background by colour and then stop being
  drawn. Splitting the ink table into fills (which carry a ramp position) and
  shapes (which do not) is what made one `paint` function serve both.

  The same width-1 rule as the matrix rain applies to all of it: box drawing
  and block elements are one cell, and a test asserts it for every ink, because
  one double-width glyph shears the whole grid to its right.

- **Sharing one file between `build.rs` and the crate beats generating a
  format twice.** The minions sprite sheet is packed at build time and unpacked
  at runtime, which normally means an encoder in `build.rs` and a decoder in
  `src/` that agree until the day they do not. `#[path = "src/arcade/minions/
  codec.rs"] mod minions_codec;` in `build.rs` compiles the *same file* into
  both, so the format has one definition; `build.rs` then asserts the blob
  decodes back to the sheet, which fails the build rather than the screensaver.
  The sizes it measured are written out as consts (`OUT_DIR/minions_stats.rs`)
  and included by the module, so the documented footprint cannot go stale.

- **A hash that does not mix puts every ripple in a row next to the last one.**
  The lake's ripples are placed by hashing (row, index) rather than kept as
  state, so the water is a pure function of the clock — a screensaver may be up
  for hours. The obvious `row * A + i * B` is not a hash: consecutive `i` land
  a constant apart, and after `% width` that constant was *one*, so five
  ripples drew as `~~~~~`. Running the key through the splitmix64 finalizer
  first fixed it. The lesson is narrow but recurring: multiply-and-add is a
  *sequence*, not a scatter, and modulo does not rescue it.

- **`ureq` defaults *every* timeout to `None`, and a dropped network gives the
  kernel nothing to time out.** A sudden link loss (Wi-Fi off, sleep, NAT
  rebind) produces no RST and no FIN, and once the request is fully sent plank
  is purely *receiving* — so there is no unacked data for TCP to retransmit and
  therefore no kernel timeout can ever fire. The socket sits established and
  black-holed, and a blocking read on it parks forever. Every agent needs
  explicit `timeout_connect` / `timeout_recv_response`; the streaming ones need
  more than that, because `timeout_recv_body` bounds the *total* body duration
  and so cannot tell a dead socket from a long healthy generation. The body
  therefore gets an **idle** timeout instead (`remote::STREAM_IDLE_TIMEOUT`),
  which is only sound because both providers keepalive their SSE streams
  (Anthropic `event: ping`, OpenAI comment frames).

- **Never poll a cancellation flag from a data-driven callback.** The provider
  and ds4 clients used to check `interrupt()` inside the `read_sse` callback,
  which runs per arriving event. Zero bytes means zero polls, so cancellation
  died in exactly the situation it was needed. The fix is structural, not a
  timeout: the read runs on its own thread feeding a channel and the turn does
  `recv_timeout`, so the flag is polled on a *clock* (`remote::pump_sse`).
  Anything gated on "the peer is still talking to us" has this bug latent.

- **A `std::thread::scope` worker cannot be abandoned, so force-quit means
  `process::exit`.** `run_worker_ui` spawns the turn on a scoped thread holding
  `&mut Agent`; the scope cannot be left while it runs and the borrow checker
  enforces it. There is no "abandon the thread and keep going" — the UI's only
  escape from a wedged worker is restoring the terminal and exiting the
  process, which skips every destructor and loses the in-flight turn. Worth
  knowing before designing any UI affordance that promises to cancel a turn.

- **`TextBuffer::write_canon` auto-indents, so it must never seed a file you
  intend to write back.** The canonicalizing insert path expands tabs to spaces
  (gated on `indent_with_tabs`, default off) and, after every newline,
  re-emits the *previous* line's indentation on top of the incoming line's own
  leading whitespace — so indentation compounds down the file. It also rewrites
  every interior line break to the one convention `set_crlf` selected. `/open`
  inherited that seeding from the Ctrl-G prompt path, where it is harmless
  because nothing is written to disk; against a real file it turned
  `fn a() {\n    let x;\n}\n` into cascading indentation plus a junk trailing
  line, silently converted Makefile tabs to spaces, and normalized CRLF — and
  because `State::original` is read back *out of the buffer*, `is_modified()`
  reported the mangled buffer as clean, so nothing warned. File mode now uses
  `write_raw` plus `set_crlf` detection (`src/miniedit/state.rs`), with
  byte-exact round-trip tests for brace-indented, tab-indented, CRLF, empty and
  newline-less inputs. Flat unindented fixtures cannot catch this class of bug;
  any test guarding a file round-trip needs indentation and CRLF in it.
- **Never decide "did this change?" by re-deriving what an editor buffer would
  have done.** `/open` first compared the edited text against a seed computed
  outside the buffer. Each fix modelled one more of the buffer's
  normalizations (the missing trailing newline) and still missed the next
  (interior line breaks), so mixed and lone-CR files were silently rewritten on
  a Ctrl+S the user never edited into. Any independently-derived baseline drifts
  the moment the buffer normalizes something new. The fix is structural: ask the
  buffer itself, via `State::is_modified()` against the `original` it read back
  at construction (`State::accepted_text`). For a file, "accepted unchanged" and
  "cancelled" are the same outcome, so both collapse to `None`.
- **Bare `cargo fmt` reformats the vendored `refs/obscura` submodule.** obscura
  is a path dependency (`Cargo.toml`), so rustfmt walks into it and rewrites 59
  files / ~4000 lines to plank's style — churn inside a submodule we do not own,
  showing up only as ` M refs/obscura` in the parent's status. Use `cargo fmt -p
  plank`. Note the pre-commit hook still runs the bare form, so a commit made
  through it can carry the drift; recover with `git -C refs/obscura checkout --
  .`.
- **A sub-agent's report is its transcript text, and a transcript keeps
  `<think>` verbatim.** The report handed back as a tool observation came from
  `last_assistant_text`, i.e. the raw last assistant message — which still
  carries the reasoning block, because the KV prefix depends on the transcript
  holding thinking unaltered. The parent therefore received a report narrating
  half-abandoned alternatives, judged it unreliable, and redid the work by hand
  (defeating the whole point of delegating). Two fixes are needed, because they
  address different halves: `strip_thinking` removes blocks the model *marked*
  as thinking, and the `agents::task_message` envelope asks for the plain answer
  stated once, which is the only lever against reasoning narrated as ordinary
  prose in the report body — no parser can identify that. Note the emptiness
  test has to run *after* stripping, or a pass that was pure reasoning yields a
  blank report instead of falling back to the last real answer.
- **`record_usage` fires at pass *completion*, so per-pass token accounting
  shows nothing during a long pass.** A sub-agent roster row wired only to the
  per-pass tally sat blank for the minutes a local pass takes, which reads as a
  broken feature rather than as "not counted yet" — and it is intermittent,
  since a short pass populates the row almost immediately. Live counts have to
  come from the worker's `UiEvent::Status` snapshots (`prefill_done`/
  `prefill_total`/`generated`), the same source `status::progress_segment` draws
  the main progress line from. Fold the completed pass in and drop the live
  figures in the same breath, or the two double-count. The snapshot describes
  whichever pass the engine is running, so it can only be attributed when
  exactly one sub-agent is in flight; a fan-out has several, and its rows must
  stay on their own per-pass tallies.
- **A live one-line summary of an agent cannot come from the tail of its
  output.** The roster's task column was first derived from the newest non-blank
  line of the run's `OutputLog`, on the reasoning that a derived value cannot
  drift from what it summarises. It cannot drift, but it is worthless: a
  streaming line is sampled mid-statement, so the column showed fragments like
  `vals =` from whatever code the model happened to be emitting. The delegated
  task is the stable, meaningful source — flattened to one line, since a task is
  often a paragraph and a raw newline breaks the row. Cap it well below the
  available width too (`TASK_MAX_COLS`): sized only by "whatever room is left",
  prose runs to the edge on a wide terminal and buries the name and the tally
  the row exists to show.
- **`.kv` used to mean two different things, and the GC paid for it.** The
  extension named both a session transcript and a Tier 1 checkpoint, so
  `gc_system_checkpoints` had to hand-filter file names by the `sysprompt-`
  prefix to avoid eating user data. Any new species of cache file inherited the
  same obligation, and one forgotten prefix check would delete transcripts.
  Bodies are now `<stem>.kv_raw` with a `<stem>.json` sidecar, transcripts alone
  keep `.kv`, and the sweep can walk by extension with nothing to hand-filter.
  The rename is also why the migration wipes rather than adopts the old layout:
  synthesized metadata would carry no lineage and unreliable counters, and every
  tier rebuilds on demand.
- **Metadata beside a KV blob has to stay advisory, or the cache stops being
  safe.** The signature inside the body (`KVCache::decode`) is the only trust
  input for restoring cached bytes; the sidecar exists for display, counters and
  retention. A missing or corrupt sidecar therefore costs a nicer `/kvcache` row
  and some counters, never correctness, and sidecar writes are best-effort for
  the same reason. The moment anything consults a sidecar field to decide whether
  a body may be loaded, a stale or hand-edited JSON file becomes able to feed the
  model a KV built from a different prompt, which is exactly the failure every
  fingerprint in this subsystem exists to prevent. `META_VERSION` is likewise
  independent of `kvcache::FORMAT_VERSION`, and a sidecar at an unrecognized
  version is ignored rather than migrated: resetting counters is cheap, and
  guessing at a schema you do not know is not.
- **The GC's "has a surviving child" rule reads the pre-sweep node set.** Judged
  against a set mutating as files are unlinked, a sweep's outcome would depend on
  directory scan order, so the same cache could collect different files on two
  runs. Reading the set as it stood before the sweep began costs one extra run to
  collect a parent whose last child died this run, which is the intended
  bottom-up cascade: a dead chain collects one level per launch.
- **Phase 2 must read the *post*-phase-1 survivor set, the opposite of phase 1.**
  The budget pass re-derives "has a surviving child" against what phase 1 left
  alive. Reusing phase 1's pre-sweep view there would make a parent whose only
  child just expired immortal under any budget, because it would keep looking
  like it was holding a live descendant up. The two phases read different sets on
  purpose, and the reason differs on each side: phase 1 wants order independence,
  phase 2 wants an upper bound that actually binds.
- **`kvcache.maxBytes = 0` means unbounded, not "evict everything".** The
  opposite reading is the natural one for a ceiling, and it would wipe the entire
  cache on every launch for everyone who never set the key. A budget of zero is
  also the shape an absent or unparsable settings value degrades to, which is the
  worst possible moment to start deleting. Note the neighbouring TTLs go the
  other way, where `>=` makes a TTL of zero mean "collect on sight" rather than a
  silent no-op; a zero is only self-evident once you decide what it disables.
- **A sweep verdict must map to a path, not to a fingerprint.** Two bodies can
  legitimately share one fingerprint: a root `sysprompt-X.kv_raw` beside a
  `<projkey>/project-X.kv_raw`, or the same `project-X` under two project
  directories. A fingerprint-keyed delete then unlinks a file the sweep decided
  to keep, and it looks like a policy bug rather than a lookup bug. `sweep` walks
  paths and metadata as one paired list so verdict `i` names file `i`. The
  `/kvcache` mutation path cannot do that (its input is a fingerprint prefix
  typed by a user), so it refuses an ambiguous match outright instead of guessing.
- **Re-persisting an existing fingerprint must preserve `created` and `pinned`.**
  A refresh is the same blob being written again, not a new blob, and treating it
  as new silently unpins whatever the user pinned and resets the age the TTL is
  measured from. Both fields are read back off the prior sidecar before the new
  one is written, so a pin survives every subsequent store.
- **A test whose fixtures are always written fresh silently stops testing
  anything once retention becomes age-based.**
  `gc_keeps_the_alt_local_engines_system_checkpoint` was in exactly that state:
  its blobs were young, freshness alone kept them, and the keep-set it existed to
  exercise had no effect on the outcome. Deleting the entire code path it
  guarded left it green. Any test of an age-sensitive policy has to write
  explicit `last_used` values into its sidecars rather than let the filesystem
  supply "now".
- **A feature-gated dependency's binary cost is invisible until something calls
  it.** Adding Extism (and through it wasmtime) behind the `plugins` feature
  measured as +1.1 MiB, which is roughly wasmtime's *symbol table* and nothing
  else: no code in `plank` reached `wasmhost::host()`, so the linker dead-stripped
  the runtime. Forcing one reachable call from `main` put the real number at
  **+18.0 MiB**. Any "how much does this dependency cost?" measurement has to
  route through a call site the binary actually retains, or it measures the
  linker rather than the dependency.
- **`extism` does not compile with `default-features = false`.** The obvious way
  to drop its `http`/`register-http`/`register-filesystem` defaults also drops
  `wasmtime-default-features`, which is what carries the cranelift backend; the
  result is 41 trait-resolution errors inside `extism` itself, none of which name
  the missing feature. Disable the three by name and keep
  `wasmtime-default-features` on.
- **plank's release flow signs nothing.** `release.yml` is `cargo build
  --release` into a tarball plus a Homebrew bottle — no codesigning, no hardened
  runtime, no notarization. That is why a JIT-based plugin runtime is viable at
  all: the `com.apple.security.cs.allow-jit` entitlement problem a notarized
  build would have hit does not exist here. Worth re-checking before anyone adds
  signing, since it would become a blocker retroactively.
- **An animation gate keyed to one feature starves the next one.** The TUI's
  idle loop polled at 20 Hz "if the arcade is open" and at 200 ms otherwise, so
  the first WASM `frame` component ran at five frames a second. The stutter was
  the visible half; the worse half is that the frame delta is measured from real
  elapsed time and then clamped to `MAX_STEP_MS`, so at that rate half of every
  second was dropped from the simulation and the motion ran *slow* as well as
  rough. Any new thing that animates has to be added to that condition, and the
  symptom does not look like a poll-rate problem — it looks like a slow plugin.
- **A wall-clock assertion in a parallel test suite measures the machine.** A
  per-frame budget test asserted 10 ms, passed at 3.7 ms run alone, and failed
  under the full suite because everything else was competing for the CPU. Keep
  the measurement, print the number, and set the threshold to catch an
  order-of-magnitude regression rather than a busy box.
- **A discarded stderr turns every server death into a timeout.** plank spawned
  MCP stdio servers with `.stderr(Stdio::null())` so that only JSON-RPC reached
  stdout, and reported both a poll timeout and a closed pipe through one string:
  `no response from server (timeout or closed pipe)`. When `tokensave serve` is
  started outside an indexed project it prints `no TokenSave index found ...` to
  stderr and exits 1 *before* answering `initialize`, so the single actionable
  line was thrown away and the message blamed a 30-second timeout that had not
  elapsed. Piping stderr into a bounded rolling tail costs nothing and keeps
  stdout clean; the tail is what makes the failure legible. Two traps in the
  fix: EOF on stdout **races the child's exit**, so an immediate `try_wait` says
  "still running" for a process that has already died and the message reverts to
  blaming the timeout — a short grace poll is required; and a server's startup
  error is usually followed by a long usage dump, so keep the *tail* and quote
  the *last* line rather than the first.
- **`primaryTools` was honoured on the text path and ignored on the provider
  path.** `append_tool_schemas` gives a full schema only to primary tools and
  lists the rest as a one-line directory, but `provider_tool_registry` pushed a
  `ToolSpec` for *every* MCP tool. With tokensave connected that made a plank
  request 94.5 KB of which 90.2 KB was 140 tool schemas — 95% of the body,
  resent on every turn. Filtering to primary cut it to 52.6 KB / 66 tools.

  Measured on a 27B Q4 Metal server (prompt cache off, so prefill is
  deterministic): bare prompt 61 tokens / 1.4 s to first token; 66 tools 13.6k
  tokens / 80.8 s; 140 tools 24.7k tokens / 156.4 s. **The cost is
  time-to-first-token, not generation speed** — decode came out flat at
  7.07 / 7.67 / 8.16 t/s across the three, i.e. within noise, and a quieter run
  put the context penalty at roughly 15% (26.6 t/s bare vs 22.4 t/s at 13.6k).
  The intuition that a long KV slows every token is real but second-order here;
  what people report as "plank is much slower than llama-cli" is a ~160 t/s
  prefill multiplied by a prompt two orders of magnitude larger, which reads as
  catastrophic t/s only if the measurement divides tokens by wall-clock. With
  the server's prompt cache on, repeat turns prefill in 0.2 s, so the payload is
  paid on the first turn of a session and again whenever anything perturbs the
  prefix (tool list change, a server flapping, compaction rewriting early
  transcript). Beware measuring any of this on a loaded box: an orphaned
  benchmark hitting the same server concurrently moved decode between 1 and
  27 t/s and inverted the ordering.
- **A text-path tool can be undeclared; a provider-path tool cannot.** The text
  path can leave directory tools out of the prompt because the model emits DSML
  — free text that can name anything. On an OpenAI-compatible endpoint the model
  can only call a function that was declared, and llama.cpp goes further by
  building a *grammar* from the `tools` array, making an undeclared name
  literally ungeneratable. So narrowing the declared set needs an `mcp_call`
  escape hatch (full name + JSON arguments) carrying the directory in its
  description; without one, "hide the schema" silently becomes "delete the
  tool". Measuring a payload win here is easy, and it is exactly the change that
  can lose functionality without failing a single test.
- **The provider path advertised MCP tools under a name its own dispatcher
  rejects.** `provider_tool_registry` pushed `ToolSpec { name: tool.name }` —
  the bare `tokensave_status` — while `dispatch` routes MCP calls on the `mcp__`
  prefix (`tools/mod.rs`) and the text path spells them `mcp__<server>__<tool>`
  (`append_one_schema`). So on an OpenAI-compatible provider every MCP tool was
  offered to the model and then answered with `Tool error: unknown tool`. Two
  reasons it survived: no test compared the advertised name against what
  dispatch accepts (they only checked that the name was *present*), and the
  failure needs a live provider turn plus a tool the model actually decides to
  call. It stayed invisible until the `mcp_call` directory started listing the
  qualified spelling next to the bare specs — the model then said so out loud,
  reasoning that "the tool is listed in the function spec as tokensave_status,
  but it says unknown tool ... maybe it's in the MCP directory". A prompt that
  contradicts itself produces exactly that thrash, and it looks like a model
  failure rather than a naming bug. Assert routability, not presence.
- **A long prompt costs ~20% of decode and all of the TTFT.**
  `--minimal-prompt` makes the comparison cheap. Measured by warming each prefix
  once and then sampling decode four times with the server's prompt cache on,
  same question in every body (±1% repeatability): 59 tokens 6.55 t/s,
  2,863 tokens 5.37 t/s (**-18%**), 10,679 tokens 5.14 t/s (**-22%**). Cold TTFT
  over the same three: 0.5 s, 16.1 s, 54.8 s — warm, all three are 0.2-0.3 s.
  So both effects are real, they compound, and **TTFT is the one that dominates
  a cold turn** by an order of magnitude.

  Getting a decode number that means anything took three attempts, and the two
  failures are the lesson. (1) Short natural generations (48-57 tokens) are pure
  noise: one body measured 20.69 and 7.91 t/s minutes apart. (2) `ignore_eos`,
  the obvious way to force equal-length generations, **inverts the result** on a
  speculative-decoding server — past the natural end the model emits degenerate
  text, the DFlash draft model stops predicting it, acceptance collapses, and
  the *bare* prompt looks slowest (3.05 t/s) because it reaches that point
  first. Warm-cache repeat sampling is the method that works. Absolute values
  still track machine load — the same contexts gave 22-26 t/s on an idle box and
  5-7 t/s under load average 7.6 — so only ever compare figures measured back to
  back, and quote ratios rather than absolutes.
- **A provider engine can report real throughput; it just has to measure a
  different clock.** `ProviderEngine::generate` hardcoded `tps: 0.0` and
  `steady_tps: 0.0`, so every provider turn displayed `0.0 t/s` and the
  `/speeds` peak line was suppressed by its own `> 0.0` guard. Both are
  measurable from this side: `tps` from a clock started before the request,
  `steady_tps` from the arrival of the *first text event* — the local engines
  mark "steady" at `STEADY_WARMUP_SECS` into the pass, which a provider cannot
  observe, but the first byte separates exactly what that warmup exists to
  exclude (connect, queue, server-side prefill). Count tokens from the API's
  `usage`, never from SSE deltas: a delta is not a token. Validated against
  llama.cpp's own `print_timing` for the same turn — plank 8.5 tok/s vs
  llama-server 8.46 t/s over an identical 198-token count.

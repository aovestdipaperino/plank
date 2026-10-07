# What plank can take from rig

Status: proposal. Nothing here is implemented. Fact-checked against both
codebases in a second pass; corrections from that pass are folded in.

[rig](https://github.com/0xPlaygrounds/rig) (`0xPlaygrounds/rig`, MIT) is a
provider-agnostic async agent framework: `rig-core` (messages, completion
types, provider clients), `rig-agent` (tool loop, hooks), `rig-memory`
(history windows and compaction), `rig-candle` (local candle inference),
`rig-rmcp`, `rig-cassette` (record/replay). This document records what a read
of that code found worth carrying into plank, with a design for each item and
where it deliberately does *not* fit. It also records two plank bugs the
comparison surfaced, which are worth more than any of the ports.

## Why not depend on rig

Plank's transcript is text, not typed messages. `session::Message` is
`{role, text, at, images}`, tool results are `User` messages whose text starts
with `<tool_result>`, and tool calls live inside the assistant text as DSML
(or the Gemma and Qwen dialects in `crates/trace-stream`). Model-facing bytes
must match the C reference (`tests/c_parity.rs`). Rig's `Message` model, async
bus, provider clients and MCP bridge each overlap an existing plank layer
(`Engine`, `tools/`, `tools/mcp.rs`) and would fight the parity rule. The value
is in a few *algorithms, policies and test lists*, copied as small pure
functions with their own tests, the way the rest of plank was ported.

Licence: rig is MIT, as is plank. Copied snippets keep an attribution comment.

| # | Item | Plank site | Effort | Value |
|---|------|------------|--------|-------|
| 0 | **Bug:** `more` loses its spill continuation | `tools/mod.rs`, `tools/files.rs`, `spill.rs` | small | high |
| 1 | Sampler: top-k, repeat penalty, validation | `sample.rs`, `engine.rs`, `ds4engine.rs`, `config.rs` | medium | medium |
| 2 | Escalating unknown-tool feedback | `guard.rs`, `ui.rs`, `tools/mod.rs` | small | medium, parity-sensitive |
| 3 | Parser case matrix | `crates/trace-stream` tests | small | low-medium |
| 4 | Head-and-tail spill preview | `spill.rs` | small | medium |
| 5 | Pre-send transcript validator | `ui.rs` `session_to_messages` | small | low-medium |
| 6 | Orphan-aware compaction tail | `ui.rs` `rebuild_after_compact` | small | low (cosmetic) |
| 7 | GGUF load validation | `crates/gemma-engine` | unknown | to evaluate |
| 8 | Heuristic token counter | `engine.rs` | — | rejected |
| 9 | Effect-log replay | `EchoEngine` tests | large | defer |

---

## 0. Bug: `more` loses its spill continuation (and splits characters)

Not a rig port. Comparing rig's char-boundary-safe truncation with
`spill.rs` led here, and it should land before anything else.

### Defect A: continuation dies after the first `more`

`tool_more` (`tools/files.rs`, around line 250) advances `ctx.spill.offset` and
prints `continue_offset=… Call more …`. But `more` is dispatched through
`dispatch` (`tools/mod.rs`), whose tail unconditionally runs the spill policy
on every tool's output, `more` included:

```rust
let (preview, spilled) = crate::spill::apply(&policy, &ctx.session_id, &call.name, output);
ctx.spill = spilled;
```

A `more` chunk is normally below `spill_max_bytes`, so `spilled` is `None` and
the offset `tool_more` just advanced is wiped. The model was told to call
`more` again; the second call falls back to `ctx.more` (a stale file-read
continuation) or "no previous output". The existing test only exercises the
first `more` call, which is why it passes.

Fix: in `dispatch`, assign `ctx.spill` only when the policy actually spilled
(`if spilled.is_some() { ctx.spill = spilled; }`), or skip the policy for
`more`. The first is safer: a new oversized result from any other tool still
replaces the old continuation. Check that a *fresh* small result from another
tool should still leave the old spill reachable; if not, clear it explicitly
for every tool except `more`. Test: three successive `more` calls walk the
whole payload and the third reports the end.

### Defect B: preview counts characters, offset counts bytes

`spill::apply_in` builds the preview with
`result.chars().take(policy.preview_bytes)` but sets
`offset: policy.preview_bytes` and the locator says "bytes". `tool_more` slices
the file by bytes. With 3-byte characters and the default 4096:

- the preview shows 4096 characters (12288 bytes, three times the budget);
- `more` resumes at byte 4096, inside text already shown, and mid-character
  (4096 = 3·1365+1), so `from_utf8_lossy` emits U+FFFD.

No panic (`&bytes[start.min(total)..]` is always in range), but duplicated
output and replacement characters. Every later `more` chunk also cuts at an
arbitrary byte, so the fix covers both sites: cut the preview at a byte
boundary rounded down to a char boundary, and round each `more` chunk end down
the same way (rig's `truncate_summary` uses the `is_char_boundary` loop;
`str::floor_char_boundary` if the MSRV allows it). Test with a payload of
3-byte characters.

The locator text is fixture-shaped; only the numbers change, so record the
fix in `FINDINGS.md` and check the fixtures still match.

---

## 1. Sampler: top-k, repeat penalty, validation

### What rig does

`rig-candle/src/generation.rs`. `GenerationConfig` carries `max_tokens`,
`temperature`, `top_k`, `top_p`, `seed`, `repeat_penalty` (default 1.1) and
`repeat_last_n` (default 64), maps them onto candle's `Sampling` variants, and
applies `apply_repeat_penalty` over the last `repeat_last_n` tokens. Reusable
pieces:

- `validate_generation`: rejects zero `max_tokens`, non-finite or negative
  temperature, `top_k` of zero or above the vocabulary, `top_p` outside
  `(0, 1]`, non-finite or non-positive `repeat_penalty`.
- `effective_output_limit`: separates "prompt longer than the context" from
  "prompt fits, no room to generate".
- `OptionalGenerationOverride` (`Inherit | Set | Disable`).

### What plank has

- Gemma: `Sampler::sample(&mut self, logits, temperature, top_p, min_p, greedy)`
  in `crates/gemma-engine/src/sample.rs`, SplitMix64 so a seed reproduces a run
  with no `rand` dependency; called in `gemmaengine.rs` near line 531.
- DeepSeek: the C sampler is
  `ds4_session_sample(s, temperature, top_k, top_p, min_p, rng)`
  (`refs/ds4/ds4.h`), and plank passes a hard-coded `0` for `top_k`
  (`ds4engine.rs` around 1885, and again around 2880). **Top-k already exists
  on the DeepSeek path; plank just never exposes it.**
- `GenerationOptions` (`engine.rs`) has `temperature`, `top_p`, `min_p`,
  `seed`. CLI flags `--top-p`/`--min-p` exist (`config.rs`, `parse_float_range`).
  There are **no sampling keys in `settings.rs`** today.
- `GenerationOptions` is mirrored in the remote protocol
  (`src/remote/proto.rs`), so new fields touch that wire too.

Do not delegate to candle's `LogitsProcessor` (suggested in the first pass):
it takes `Tensor`s, draws from `rand`, and would change every seeded output,
breaking `greedy_is_argmax_and_seeded_sampling_is_deterministic`. Port the
semantics into the in-house sampler.

### Design

Two independent changes.

**1a. Top-k, both engines.** Add `top_k: i32` (0 = off) to `GenerationOptions`,
a `--top-k` flag beside `--top-p`, and the field in `remote/proto.rs` with a
serde default of 0 so older peers interoperate. DeepSeek passes it to
`ds4_session_sample` in place of the literal `0` (keeping 0 when greedy, as the
other parameters already switch on `g`). Gemma adds it to its sampler: after
min-p, `truncate(top_k)` on the already-sorted survivor list. Default 0 keeps
both paths byte-identical.

**1b. Repeat penalty, Gemma only.** The C sampler has no penalty argument, so
this cannot reach DeepSeek without a C change. Add `repeat_penalty: f32`
(1.0 = off) and `repeat_last_n: usize` to `GenerationOptions`; the Gemma loop
passes the tail of `self.session.tokens()` as `recent`. Pipeline order, fixed:
penalty (on a copy, touching only the distinct ids in the window, O(window))
-> temperature softmax -> min-p -> top-k -> top-p -> draw. Penalty convention
matches llama.cpp and candle: divide a positive logit, multiply a negative one.

**Default stays off.** Rig's 1.1 over 64 tokens is wrong for plank: DSML and
JSON repeat the same tags and quotes by design. The mitigation hook already
exists: `Engine::generate` receives `greedy: &dyn Fn() -> bool`, fed from the
stream renderer's `wants_greedy_sampling()`, and Gemma consults it at the
sampling call. Skip the penalty whenever `greedy()` is true; it is true while a
tool stanza is being emitted. Measure on Gemma reasoning loops before
recommending any non-default value, and record the result in
`docs/LOOP-FINDINGS.md`.

**Validation.** Port `validate_generation` as `GenerationOptions::validate()`
and call it after CLI parsing, so a bad value fails at startup naming the
field. (`parse_float_range` already bounds `top_p`/`min_p`; the new value is
the cross-field and vocabulary checks.)

**Not needed:**

- `effective_output_limit`'s "prompt too long" error: Gemma's `check_fits`
  already errors `context full: {need} tokens > {ctx}` before touching the
  session (`gemmaengine.rs` around 436). The only silent case left is a prompt
  that fits exactly and generates zero tokens; cheap to make explicit, low value.
- The tri-state override and a settings block: plank sets sampling from CLI
  flags; adding settings keys is a separate decision, not a rig port.

### Tests

- `top_k = 1` equals argmax at any temperature; `top_k` above the vocabulary
  clamps;
- penalty 1.0 leaves logits bit-identical; a positive logit shrinks and a
  negative one grows under penalty;
- the penalty is not applied while `greedy()` is true;
- a seeded run with every new option off reproduces the existing seeded test;
- `tests/gemma_parity.rs` and `tests/remote_ds4.rs` pass unchanged.

---

## 2. Escalating unknown-tool feedback

### What rig does

`rig-agent/src/run/policy.rs` classifies an undispatchable call
(`UnknownTool`, `DisallowedByToolChoice`, `MalformedArguments { error }`) and
lets a hook choose an action (`Fail`, `Retry { feedback }`,
`Repair { tool_name }`, `Skip { reason }`, `Stop { reason }`). Defaults
differ by reason: an unknown or disallowed tool **fails the run**
(`UnhandledInvalidToolCall` defaults to `Fail`, `run/spec.rs`); only
malformed arguments get a corrective tool result by default
(`answer_malformed_tool_call`). `arguments_parse_error`/`json_kind` turn a serde
error into "expected a JSON object, found an array". The hook context carries
`available_tools`.

### What plank has

- An unknown or profile-withheld tool answers
  `Tool error: unknown tool: <name>\n`. That string is in the C
  (`ds4_agent.c`, around line 8084) and pinned byte for byte by tests in
  `tools/mod.rs`. The first answer cannot change.
- Unknown-tool calls **are** seen by `LoopGuard::observe`: `run_tool_calls`
  calls `observe_calls` on every call before dispatch (`ui.rs` around 2067 and
  3947). The advisory fires at the third *identical* call (same name and args),
  the block above five; it is gated by `tools.repeat_advisory` as well as
  `guards_enabled()`, and is appended as a `[loop guard] …` line after the
  whole stanza.
- `FailedPassStreak` never sees them; it covers passes that did not dispatch.
- No error path lists the callable tools. The only "Available tools:" text is
  the system-prompt reminder, built from `sysprompt::tool_names`, which is
  **unfiltered**: it does not apply the profile allow-list or drop the gated
  tools that answer "unknown tool" when off (`recall`, `remember`, `forget`,
  `run_code`).

Gap: a model calling the same nonexistent tool with *varying* arguments never
trips the identical-call advisory, and nothing tells it what it can call.

### Design

1. A pure classifier in `guard.rs`:

   ```rust
   pub enum ToolFault {
       Unknown { name: String },  // includes profile-withheld and gated-off
       BadArgs { tool: String, why: String },
   }
   ```

   No `Withheld` variant on purpose: `tools/mod.rs` keeps withheld tools
   indistinguishable from never-offered ones, and the classifier must not
   reintroduce the difference.

2. A **post-dispatch** hook, `LoopGuard::note_fault(&mut self, fault) -> Nudge`.
   It cannot live in `observe`, which runs before dispatch and before the fault
   is known. Its counter is keyed on the tool *name* (not name plus args, which
   is what makes the gap above), separate from the dispatched-call window.

3. Thresholds are new and should be stated as such: advisory on the second
   occurrence of the same unknown name, block on the fourth. They sit below
   `REPEAT_THRESHOLD`/`BLOCK_THRESHOLD` because a call to a tool that does not
   exist can never succeed on retry.

4. The advisory reuses the existing `[loop guard]` line format, so the pinned
   error line is untouched, and is gated exactly like the existing advisory
   (`guards_enabled()` and `tools.repeat_advisory`).

5. The tool list must come from a new **dispatchable** source, not
   `sysprompt::tool_names`: builtins after the profile allow-list and the gates,
   plus live MCP and WASM tools. Testing it against `dispatch` (every listed
   name must not answer "unknown tool") is the guard, the same "assert
   routability" lesson as `FINDINGS.md` around line 1815.

6. `BadArgs`: port `arguments_parse_error`/`json_kind` only for argument
   errors that are plank-originated; leave any string that comes from the C.

Not proposed: `Repair` (silently runs a call the model did not make), and rig's
fail-the-run default for unknown tools (plank's loop recovers instead).
`Stop` already exists as `LoopGuard::tripped`.

### Tests

- first unknown call: output equals the pinned string, no advisory;
- second call to the same unknown name with *different* args: advisory present;
- the advisory never names a withheld or gated-off tool;
- different unknown names do not share a counter;
- `/loopguard off` or `tools.repeat_advisory` off: no advisory.

---

## 3. Parser case matrix

### What rig does

`rig-candle/src/protocol.rs`, `parse_qwen3_assistant`, rejects malformed
Qwen3 output with a distinct error per case: unterminated `<think>`;
`</think>` without an opener; a misspelled `<tool-call` delimiter; a close
before any open; an unterminated or nested block; invalid JSON; an empty name;
non-object arguments; duplicate or empty call ids; reserved protocol markers in
the visible text between calls (text itself is allowed).

### What plank has

`crates/trace-stream` holds three dialect parsers: `dsml.rs` (`DsmlParser`),
`gemma.rs` (`GemmaParser`) and `qwen.rs`, with the syntax selector in
`syntax.rs`. They are deliberately *lenient* where real model output was
damaged: a repeated wrapper opener is consumed idempotently; near-miss
delimiters (trailing bar, missing `>`, dropped leading bar, SSML alias, bare
tags) are salvaged; an invoke without a name errors "tool invoke without
name"; there is no `finish()`, so truncated input stays in `ParamValue` with no
error. Gemma already has `strict_rejections` and
`every_byte_split_gives_the_same_result`; DSML has
`parses_bytewise_identically`.

So rig's strict outcomes are often the *opposite* of plank's intended
behaviour, and porting them as expectations would be wrong.

### Design

Use rig's list as a coverage checklist, not as expected outcomes. For each
case and each dialect, find the existing test or add one that pins the
*current, intended* behaviour:

| rig case | DSML | Gemma | Qwen |
|----------|------|-------|------|
| unterminated reasoning block | | | |
| reasoning close without opener | | | |
| near-miss delimiter | salvaged (tested) | | |
| close before open | | | |
| second opener before close | skipped (tested) | | |
| truncated mid-call | `ParamValue`, no error (tested) | | |
| nested call | | | |
| invalid argument JSON | | rejected (`strict_rejections`) | |
| empty name | errors (tested) | rejected (`strict_rejections`) | |
| non-object arguments | | | |
| protocol markers in visible text | | | |

Fill each blank with an existing test name or a new pinning test. The Qwen
column is the closest analogue to rig's own matrix and probably has the most
blanks. Anything whose current behaviour looks accidental rather than chosen
goes to `FINDINGS.md` as a question; the success path stays byte-identical.
Rig's duplicate-id rule does not apply: plank mints ids itself
(`call_{turn}_{i}` in `session_to_messages`).

---

## 4. Head-and-tail spill preview

Depends on item 0.

### What rig does

`rig-memory`'s `truncate_summary` keeps a capped summary's first line, inserts
`[…truncated…]`, keeps the *tail*, and advances the cut to a char boundary.

### Design

`spill::apply_in` keeps only the head. For build and test output the failure
is at the end, so the model spends a `more` call to see the error it ran the
command for. Add `tail_bytes: usize` to `SpillPolicy`, default 0 so every
existing path and fixture is unchanged:

```
<head: preview_bytes - tail_bytes>
[... N bytes elided; continue_offset=H ...]
<tail: tail_bytes>
[Output truncated at ... of ...]
```

- Both cuts on char boundaries (item 0's helper).
- Enabled for `bash` only, where the tail carries the exit state; reads stay
  head-first.
- The spill file still holds the full payload; `continue_offset` points just
  after the head, so a continuation re-shows the tail at the end. Acceptable,
  and simpler than a second offset.
- A new locator shape is a new model-facing site: add a fixture and a
  `FINDINGS.md` entry rather than editing the existing one.

---

## 5. Pre-send transcript validator

### What rig does

Before sending, `rig-candle`'s protocol renderer checks that every tool result
answers an outstanding call (`UnmatchedToolResult`) and refuses otherwise.

### What plank has

`session_to_messages` (`ui.rs` around 2165) rebuilds structured messages for
provider engines, and silently emits an id-less `ChatRole::Tool` message for a
result with no pending call. That case is legitimate for the compaction
summary, the post-compaction re-injection block and stop-hook feedback. Each
provider wire format then degrades the id-less message to text
(`remote/provider.rs`: Anthropic around 1212, chat completions around 1330,
Responses around 1447, all as `"Tool result:\n…"`), so nothing is rejected.

### Design

A pure `fn validate_chat(messages: &[ChatMessage]) -> Vec<ChatIssue>` that
reports: a `tool_call_id` with no matching call earlier in the same assistant
batch; a call whose id never receives a result; duplicate ids. Id-less tool
messages are *allowed* (the degradation path is intended). Run it in tests over
`session_to_messages` output for a set of transcripts (normal batches,
compacted, hook feedback, interrupted stanza), and as a `debug_assert!` at the
send site. It is a regression guard for `split_tool_results`' pairing, which
is the code most likely to drift.

---

## 6. Orphan-aware compaction tail

### What rig does

`rig-memory`'s `split_window` scans the head of the kept window up to the first
assistant message and demotes everything through the *last* tool-result
message before it, so a provider never receives a tool result whose call was
cut.

### What plank has

`rebuild_after_compact` (`ui.rs` around 6629) walks backwards on the token
budget and never inspects the first kept message. One batch of results is one
`User` message, so the tail can begin with a `<tool_result>` whose assistant
call was summarised away. As item 5 explains, this is **not** a protocol error
for provider engines: the id-less message degrades to text, as the summary
itself already does. The effect is cosmetic: the summary is followed by an
observation the model has no call for.

### Design (low priority)

```rust
/// First tail index, advanced past leading call-result batches whose
/// assistant call precedes the cut.
pub fn tail_start_after_orphans(transcript: &[Message], raw_start: usize) -> usize
```

Advance only over messages that are genuinely a call's results. Do **not**
drop:

- stop-hook feedback (`<tool_result>` after an assistant that made no calls,
  `ui.rs` around 5945 and 15725): it carries instructions;
- image-bearing results: microcompact exempts these deliberately
  (`compact.rs` around 98).

The predicate has to agree with `session_to_messages`, which treats a trimmed
text starting with `<tool_result>`, `Tool:` or `Tool result` as a tool
message, not with `clear_set`'s narrower `starts_with("<tool_result>")`.
Distinguishing a call batch from hook feedback needs the previous message,
which the tail no longer has; the clean way is to classify on the full
transcript before cutting (does `transcript[raw_start - 1]` contain a tool
call?). This rule is plank's own (stop at the first non-orphan), not rig's
"through the last tool result" rule.

Do this only if the stray observation is shown to confuse a model; the KV
cost is nil (the rebuild already invalidates the prefix), but so is the
demonstrated benefit.

---

## 7. GGUF load validation (to evaluate)

`rig-candle/src/validation.rs` (about 870 lines) validates GGUF metadata, the
tokenizer vocabulary, token ids, stop tokens, and tensor shapes and dtypes
before running. `gemma-engine` loads arbitrary Gemma GGUFs classified by
`general.architecture`, so a malformed or mismatched file is a real input.
Not yet compared against `crates/gemma-engine/src/model.rs` and
`tokenizer.rs`; the next step is to list which of rig's checks plank lacks and
which would turn a confusing candle error into a named one.

---

## 8. Heuristic token counter — rejected

Rig's `HeuristicTokenCounter` divides bytes by 3.5 or 4, rounds up, and adds a
per-message overhead and a flat per-attachment cost. Plank already has what
this buys: `Engine::count_tokens` defaults to `text.len() / 4`, DeepSeek counts
with its real tokenizer, `GemmaEngine` uses `encode_plain`, and the remote
DeepSeek client calls `/tokenize` with a cache.

Per-message counting happens in several loops (the compaction tail walk, the
context report, microcompact and re-injection). Their cost is not strictly
bounded by the budget: the tail walk also tokenizes the one message that
crosses it, which can be up to `spill_max_bytes`. If that ever shows up in a
profile, the fix is a byte-length pre-check before tokenizing, not a rig port.
The only rounding difference (`len / 4` floors) is a one-line `div_ceil` if it
ever matters.

---

## 9. Effect-log replay — defer

`rig-cassette` records provider HTTP exchanges and an effect log of an agent
run, then replays them deterministically. Plank covers generation with
`EchoEngine` scripting and wire bytes with `PLANK_REGEN_FIXTURES`, but not a
whole turn (generate, dispatch, feed, generate) as a recorded unit. A plank
version would record the `EngineEvent` stream and each tool call with its
output, then replay through `run_turn`, turning the `repro-*` dumps in
`docs/LOOP-FINDINGS.md` into model-free regression tests. Large, touches the
central loop, nothing above depends on it. Revisit if items 1b and 2 keep
needing real-model repros to verify.

---

## Suggested order

1. Item 0, both defects: real bugs, small fixes, one test each.
2. Item 1a (top-k): a plumbing change that unlocks a parameter the C already has.
3. Item 3: tests only; fills the gaps before anyone touches parser behaviour.
4. Item 2.
5. Item 4, then item 5.
6. Item 1b, only after measuring the penalty on Gemma loops.
7. Items 6, 7 and 9 as evidence or need appears.

# Gemma 4 on the native Rust engine

Status: implemented (v1). Gemma 4 runs as plank's main agent model on
`GemmaEngine` (`src/gemmaengine.rs`), which sits over the native-Rust,
candle-based `crates/gemma-engine`. The engine needs neither the `refs/ds4`
submodule nor macOS: it runs on Metal on macOS and on the CPU elsewhere.
The tree compiles no C for it.

## How to run

```sh
plank --model gemma4-e4b          # catalog engine; downloads ~5 GB on first use
plank --model gemma4-12b          # the 12B QAT Q4_0 engine
plank -m ~/models/gemma-4-E4B-it-Q4_K_M.gguf   # any Gemma 4 GGUF by path
PLANK_NO_DS4=1 cargo build --release           # a build without the C engine
```

The cargo feature `gemma` is on by default. With `PLANK_NO_DS4=1`, `build.rs`
skips the C engine, and that build still runs Gemma for real. A Gemma-family
file goes to `GemmaEngine` under either cfg. `make_local_engine` checks for it
first, because none of the DeepSeek-sized gates apply.

`gemma4-e4b` (unsloth E4B instruct, Q4_K_M) and `gemma4-12b` (Google's 12B
instruct, QAT Q4_0) are catalog engines in `engines.json`. Each has a main role
only, and each carries a `"family": "gemma"` hint. The hint routes a catalog
engine to `GemmaEngine` and names its transcripts `.gemma.kv` before its file
has been downloaded and its header can be read. A file given by path is
classified by its own `general.architecture` (`gemma4`).

With no `-c`, Gemma picks its own context: `min(32768, context_length)`. It
does not inherit DeepSeek's 131072 default, which would cost about 14 GB of
Gemma KV and is no answer about this model. An explicit `-c` is still capped
by the GGUF's `context_length`.

## The chat format

The format follows the GGUF's own `tokenizer.chat_template`, which is the
trained format. Where the template and the plan text disagreed, the template
won.

```
<|turn>system
<|think|>
You are …<|tool>declaration:NAME{description:<|"|>…<|"|>,parameters:{…}}<tool|><turn|>
<|turn>user
…<turn|>
<|turn>model
<|channel>thought
…<channel|><|tool_call>call:NAME{key:<|"|>value<|"|>,n:3}<tool_call|><|tool_response>response:NAME{value:<|"|>…<|"|>}<tool_response|>…final answer<turn|>
```

- **Thinking** is switched on by `<|think|>\n` at the start of the system
  turn.
- **`<|"|>` delimits every string value.** There is no escaping inside it.
- **Tool results sit inside the model turn**, not in a user turn. A string
  result is `response:NAME{value:<|"|>…<|"|>}`; the key is `value`. The body
  is plank's ordinary tool-output framing, byte for byte. The agent still
  records a result as a `<tool_result>` user message. `GemmaEngine` renders
  that message as a `<|tool_response>` inside the open model turn.
- **Declarations** follow the template's field order:
  `parameters:{properties:{…},required:[…],type:<|"|>OBJECT<|"|>}`, with
  properties sorted and `type` last in every object, uppercased.
  `sysprompt::gemma_tools_prompt()` generates them from the same tool table
  as the DSML prompt, so the two tool sets cannot drift.
  `tests/gemma_parity.rs` pins the rendered prompts to fixtures, which are
  byte-identical to a jinja2 rendering of the real template.
- **Control ids** (E4B): BOS is 2. `<|turn>` is 105. `<turn|>` is 106, and it
  is also EOS. The full table is in `FINDINGS.md`.

User text, tool output and MCP schema text are tokenized plainly, so they
never produce a control id, whatever they spell. Only the template's own
markers and the trusted system prefix map to control ids.

### Span rendering: `<turn|>` belongs to the next section

`template.rs` renders each section from its own text and the previous
section's kind, and from nothing else. A section's tokens therefore never
change once the next section arrives, and `GemmaEngine` can keep a recorded
span's tokens verbatim, as the ds4 engine does.

The one context-dependent piece is the `<turn|>` that closes a model turn.
Whether a model turn closes depends on what follows: a tool result continues
the turn, and a user message ends it. So the *following* user section emits
the `<turn|>`, not the assistant span. An assistant reply is recorded open.
The next user prompt closes it and opens its own turn.

### Thoughts

The model writes its reasoning as `<|channel>thought\n…<channel|>`.
`ThinkTranslator` streams that block as `<think>…</think>`, so the renderer
sees what every other local model gives it. `Engine::emits_think_tags` tells
the pass not to pre-open a think block.

**Past-turn thoughts are kept** (Ruling 7). The template strips thoughts from
model turns before the last user message. plank keeps them, because KV reuse
needs recorded tokens reused verbatim: stripping them would change a turn's
tokens once the next user message arrived, and every turn would re-prefill
from the previous reply. The DeepSeek path keeps its thoughts too. The cost is
some context that is off the trained distribution, plus the thought tokens.

**No thought prefix after a tool result** (Ruling 15). With thinking on, the
template adds `<|channel>thought\n` as the generation prefix after a tool
response. plank adds nothing, which keeps span rendering context-free; the
model may open its own channel. In the smoke runs below, the model thought
before its first tool call and did not open a channel after any tool result.
It went straight to the next call or to the answer.

### Stop tokens

A pass stops on EOS, on `<turn|>`, or on `<|tool_response>`. EOS and `<turn|>`
are the same id (106). The template emits `<|tool_response>` after an
unanswered call, so the model writing it means the call is complete and the
result is plank's to supply.

## KV discipline

`GemmaEngine` keeps one live session across turns. It splits the rendered
transcript into sections, keeps the leading sections that match recorded
spans, and renders and tokenizes only the rest. This is the ds4 engine's
discipline (`docs/KV-CACHE.md`), with one difference that simplifies it:
**Gemma's KV is truncatable at any token.** A prompt that diverges anywhere
keeps the KV up to the first differing token and prefills only the remainder.

- **Sliding layers keep every position.** The sliding-window layers attend
  over the last 512 positions, but the cache does not use a ring buffer. It
  keeps every position, and attention narrows to the window when it reads.
  This costs memory on those layers. In return, truncation is exact at any
  depth. With a ring buffer, truncating behind the window would lose the
  positions the window needs. Exact truncation is what makes common-prefix
  reuse, ladder rungs and forks plain prefix operations.
- **`kv_reuse_probe` reports `live == common`** (Ruling 14). Because the KV
  truncates exactly, a divergence behind the live end is never the
  rebuild-from-zero shape that the agent's rung and fork rescue exist for.
  The probe therefore reports the reusable prefix as the effective live end.
  The agent truncates and re-prefills the tail instead of restoring a rung,
  which is the cheaper path anyway.
- **A warm walk keeps the warm buffer.** `warm_reset` only places tokens, and
  `warm_sync` prefills them (Ruling 13), so a checkpoint restore does not
  waste a ~20 s system-prompt prefill. During the walk, `set_kv` keeps the
  warm buffer (`warm_pending`) rather than adopting the checkpoint's
  transcript. Otherwise `kvtier::warm` would append the restored tiers again,
  and the project tier would be re-prefilled at every launch.
- The system-prompt checkpoint (`sysprompt-<fp>.kv_raw`) and per-session KV
  blobs work unchanged. A Gemma blob is about 560 MB at a 5k-token prompt.

`PLANK_KV_DEBUG=<file>` appends one `gemma reconcile:` line per render (spans
held, sections in, spans kept) and one `gemma generate:` line per pass (prompt
tokens, live KV end, tokens reused). Note that the variable names a file, not
a switch.

## Correctness

The forward pass was checked against llama.cpp (commit `d7a695e`) on E4B
Q4_K_M, with both on Metal. On all three fixture prompts the tokens are
identical and the argmax matches. The top-5 probabilities agree within
0.0008, and greedy continuations agree 16/16, 2/2 (EOS) and 16/16. The third
prompt is 729 tokens and crosses both the 512-token window and the 512-token
prefill chunk.

On the CPU, near-ties differ. candle's CPU quantized matmul rounds every
activation to `Q8_K` (as llama.cpp's CPU backend also does, rounded
differently). On a nearly tied prompt that moves the top probabilities by up
to 0.065. Compare Metal with Metal.

The reference test is opt-in, because it needs the 5 GB model:

```sh
PLANK_GEMMA_GGUF=~/.plank/models/gemma-4-E4B-it-Q4_K_M.gguf PLANK_NO_DS4=1 \
  cargo test -p gemma-engine --features candle --release --test reference -- --ignored --nocapture
```

`PLANK_GEMMA_DEVICE=cpu` forces the CPU. The fixture's `source` field records
the exact llama.cpp commands.

## Speed

E4B Q4_K_M on Apple Metal:

| | rate | source |
|---|---|---|
| decode at 32 tokens of context | ~21 tok/s | `examples/bench_decode.rs` |
| decode at 4096 tokens of context | ~19 tok/s | `examples/bench_decode.rs` |
| decode in an agent session (~5.2k context) | 16.8–17.5 tok/s | smoke run, exit stats |
| prefill | ~200 tok/s (201.1 measured) | smoke run, cold 5,250 tokens in 26.0 s |

Decode holds steady with depth because the KV view is zero-copy and
grouped-query attention reshapes the query instead of repeating K/V (Task 7b;
see `FINDINGS.md`). Before that change, decode fell to 2.8 tok/s at 4096
tokens. Prefill is MLP-dominated. A cold system prompt of 4–5k tokens
therefore takes 20–26 s once. After that, the system-prompt checkpoint
restores it from disk, and each turn prefills only its new suffix. In the
two-turn smoke run below, that suffix was 15 and 18 tokens.

## Smoke run (2026-10-06, release, `PLANK_NO_DS4=1`, temporary `HOME`)

One-shot tool use:

```sh
plank -m ~/.plank/models/gemma-4-E4B-it-Q4_K_M.gguf --ui console \
  -p "Create a file hello.txt containing 'hi', then show me its contents with bash."
```

The model thought, then issued
`<|tool_call>call:write{content:<|"|>hi<|"|>,path:<|"|>hello.txt<|"|>}<tool_call|>`.
It then issued `call:bash{command:<|"|>cat hello.txt<|"|>}` in a second pass
and answered. `hello.txt` held `hi`. The run took 37.4 s in total, including
the cold system-prompt prefill.

Two turns in the plain REPL, with `PLANK_KV_DEBUG` set and the checkpoint
warm:

```
gemma reconcile: 0 spans held, 1 sections in, kept 0
gemma reconcile: 2 spans held, 3 sections in, kept 2
gemma reconcile: 2 spans held, 3 sections in, kept 2
gemma generate: prompt=5157 live=5142 reused=5142
gemma reconcile: 4 spans held, 5 sections in, kept 4
gemma reconcile: 4 spans held, 5 sections in, kept 4
gemma generate: prompt=5204 live=5186 reused=5186
```

On turn 2, all four held spans were kept: system, session context, the first
prompt, and the first reply. So the reply's recorded span matched what the
agent rendered back, and only the 18-token suffix was prefilled.

## v1 limits

- **No MoE.** A GGUF that declares experts is refused with
  `gemma4 MoE models are not supported yet`.
- **No vision.** `view_image` is still declared in the tools prompt, and it
  is refused when called, as on a DeepSeek engine without vision.
- **No MTP** speculative decoding, and no companions of any kind.
- **No `decide`.** The System-1 gate cannot run, so `memory.gate` stays off.
  The `memory.gateBias.gemma` key exists so that a settings file round-trips.
- **`serve --shared-engine` refuses Gemma**, because the shared host is built
  on the ds4 engine's shared model.

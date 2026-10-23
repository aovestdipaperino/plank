# Flint on the native Rust engine

Status: implemented (v1). Flint, Recursive's 40B mixture-of-experts model in
NVFP4, runs as plank's main agent model on `FlintEngine`
(`src/flintengine.rs`). That engine is `NativeEngine<FlintFamily>`: the
shared native adapter in `src/nativeengine.rs`, which Gemma also uses, with
Flint's tokenizer, template and stream rules plugged in. The weights run on
`crates/flint-engine`, which adds NVFP4 kernels for Metal and the CPU on top
of gemma-engine's session, KV cache and sampler. Like Gemma, it needs neither
the `refs/ds4` submodule nor any C.

## How to run

```sh
plank --model flint                            # catalog engine; downloads ~22.9 GB on first use
plank -m ~/.plank/models/flint-nvfp4.gguf      # the same file by path
PLANK_NO_DS4=1 cargo build --release           # a build without the C engine
```

The cargo feature `flint` is on by default in plank and in
`crates/local-inference-engine`, as `gemma` is. A build without the C engine
still runs Flint for real, and `make_local_engine` routes a Flint-family file
to `FlintEngine` under either cfg, before any of the DeepSeek-sized gates.

`flint` is a catalog engine in `engines.json` with a main role only. It
carries a `"family": "flint"` hint, so an engine that has not been downloaded
yet still routes to `FlintEngine` and names its transcripts `.flint.kv`; a
file given by path is classified by its own `general.architecture`
(`flint`). The shape name the engine reports comes from `general.name`
(`Flint NVFP4 Preview` on the real file) and selects `ToolSyntax::Flint`.

With no `-c`, Flint sizes its context the way Gemma does
(`nativeengine::default_ctx`, a third of RAM worth of f32 KV in 1024-token
steps, never below 8192), and then caps it at 65,536 tokens. On the 128 GB
Mac the cap is what binds, and plank says so at startup:

```
plank: context 65536 tokens (12.2 GB of KV, this model's default cap; -c to change)
```

## The chat format

The format is the GGUF's own `tokenizer.chat_template` as Hugging Face
transformers renders it, which is `flint_engine::template`:

```
<|system|>SYSTEM<|system|>
# Tools ... <tools>{fn}{fn}</tools> ...
<|user|>USER<|assistant|><think>…</think>TEXT<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>
<|observation|><tool_response>RESULT</tool_response><|assistant|>…
```

No turn has a closing marker, so a section's tokens never depend on what
follows it, and plank can keep a recorded span's tokens verbatim without the
"close belongs to the next section" rule Gemma needs. The renderer matters
here: under transformers' jinja settings (`trim_blocks`, `lstrip_blocks` and
its own `tojson`) the function JSON lines inside `<tools>…</tools>` are
written back to back with no separator. A renderer such as minja or plain
jinja2 inserts newlines there, which is not the trained form.
`crates/flint-engine/tests/fixtures/flint_template.json` pins the
transformers form.

### The tools turn and the fence

plank builds one system text for every model, so it needs a way to hand
Flint's template a tools list it can render as its own `# Tools` turn. The
Flint tools prompt (`sysprompt::build_flint_tools_prompt_parts`) writes
plank's prose first, then the tools between two plank-internal fence lines,
`FLINT_TOOLS_OPEN` and `FLINT_TOOLS_CLOSE`, one function JSON per line.
`FlintFamily` cuts the system text at the first open marker and at the first
close marker after it, never the last, because whatever follows the close
(MCP server instructions, a user's system text) is not plank's and may spell
either marker. The prose and the text after the fence become one system turn,
and the fenced lines go to `template::render_tools`, which wraps them in the
template's scaffold. The trained turn `<|system|>\n# Tools …<tools>{…}{…}</tools>…`
is reproduced exactly, and it comes after the system text, as the template
places it before the first non-system message. The model never sees either
fence marker. `tests/flint_parity.rs` and the fixture-anchored test in
`flintengine.rs` pin this.

### Plain text is inert

Only the template's own markers and the trusted part of plank's system prompt
may produce control ids. User text, tool output, MCP schema text and the
function JSON lines are tokenized plainly, so they never produce one, whatever
they spell. This was the user's decision, and it is a deliberate difference
from llama.cpp and Hugging Face, which match Flint's type-4 user-defined
tokens (`<think>`, `<tool_call>`, `<arg_key>` and the rest) even in text
tokenized without special parsing. A tool result that contains
`</tool_call>` therefore stays characters and cannot close a call or forge a
turn. The tokenizer parity test asserts this on a corpus string where
llama.cpp emits three control ids and plank emits the same bytes with none.
On top of that, the template's `flint_text` escaping applies to every system,
user and tool section: it drops `<think>` and `</think>` and spells each turn
marker as `&lt;|…|&gt;`.

### Thinking is always on

The model opens every reply with `<think>` on its own; the generation prefix
is just `<|assistant|>`. `<think>` and `</think>` are single tokens whose
pieces spell the tags, so the stream needs no translation, unlike Gemma's
channel markers. plank's think modes do not change Flint's rendering:
`ThinkMode::Off` is a no-op. The obvious way to switch thinking off would be
to prefill an empty `<think></think>`, but nothing in the trained template
does that, and we have no evidence of how the model behaves after one, so
plank does not invent it. Past turns keep their thinking in the recorded
assistant span, as on Gemma and DeepSeek, because KV reuse needs recorded
tokens reused unchanged.

### Stop tokens

A pass stops on any of the GGUF's `eog_token_ids`: `<|end_of_text|>`,
`<|user|>`, `<|observation|>` and `<|developer|>`. The model writing
`<|observation|>` after a call is how it hands the turn back for the result.
The GGUF's `add_bos_token` is false, so no BOS is prepended.

## The tool-call dialect

Flint writes

```
<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>
```

with one key and value pair per argument, and values written raw with no
escaping, so a multi-line code body survives verbatim.
`trace-stream/src/flint.rs` parses it incrementally and shares `ToolCall` and
the parser state with the DSML, Qwen and Gemma parsers.

Arguments are typed by the Qwen parser's existing `value_is_json` rule (plan
decision 1): a value that is a JSON object, array, number, boolean or null is
non-string, and anything else is a string. The spec had suggested typing
values from the tool schemas instead. Keeping the Qwen rule means one typing
rule across the XML-ish dialects, and the parser needs no schema at parse
time, which matters because the renderer parses a stream before anything
knows which tool table applies.

A Flint renderer still adopts DSML openers, exactly as Gemma's does: the
stream renderer recognises any dialect's opener and parses that stanza in the
dialect it names, so a model that wrote DSML would still be understood. The
other direction is a known gap. Renderers that are not told the dialect, the
tdk console and plank's aside path, see Flint's `<tool_call>` opener, take it
for Qwen's identical opener, and show Flint calls as malformed Qwen calls.
That is display only, since dispatch goes through the Flint parser, and it is
a follow-up item: those renderers should disambiguate after the opener.

`edit_upto` (the `[upto]` anchored edit) defaults on for Flint, as it does
for Gemma (plan decision 6). The DeepSeek and Qwen prompts default it off
because their text is bound to the C reference, but Flint's tools prompt is
plank's own, like Gemma's, so nothing ties it to the exact-match edit; the
prompt teaches the anchor with an example written in Flint's own call
syntax. `tools.editUpto` still overrides the default.

## Sampling

`FlintConfig::sampling` reads the GGUF's suggestion: temperature 1.0, top_p
0.95 and top_k 20. The shared sampler has no top_k, so `FlintFamily` applies
the GGUF's top_k (20) to the logits itself before the sampler runs, keeping
the 20 largest and anything tied with the twentieth (plan decision 5). The
model's authors ship that value with the weights, and sampling at
temperature 1.0 without it would let the long tail through. Greedy
sampling skips the filter, since the maximum survives it anyway.

## KV discipline

`FlintEngine` keeps the native adapter's discipline, which is Gemma's: one
live session across turns, the leading sections that match recorded spans
keep their tokens, and only the rest is rendered and prefilled. Flint's KV
truncates exactly at any token, so `kv_reuse_probe` reports
`live == common`, and `Engine::kv_truncates_exactly` makes the agent skip
fork snapshots and ladder rungs. The rationale is the same as in
`docs/GEMMA.md` and is not repeated here.

The KV cache is f32, reusing gemma-engine's cache, at 200,704 bytes a token
on the real file (`FlintConfig::kv_bytes_per_token`). The spec asked for f16,
which would be 100,352. The user accepted f32: it is what the shared session
already does, and it keeps the forward pass exact against the f64 oracle
below. Snapshots go through the generic session (magic `PLFK`), and the
system-prompt checkpoint and per-session blobs work unchanged. In the smoke
run the system-prompt checkpoint for 5,059 tokens was 372 MB on disk and the
session blob 391 MB, well under the f32 KV those depths hold in memory,
because the generic session's snapshot writes only the window of each
sliding layer.

**The context cap** (plan decision 3). Flint's attention memory grows with
`t × keys`: the global layers' mask and the attention scores are each a
`heads × t × keys` f32 tensor, which at 262,144 keys and a 512-token chunk
would be about 17 GB per copy (arithmetic, not measured; see `FINDINGS.md`).
So the default context is the RAM-sized window capped at 65,536 tokens
(`FLINT_DEFAULT_CTX_CAP`). An explicit `-c` is still honoured up to the
file's 262,144, and refused when its KV would exceed half of RAM, as Gemma's
is.

**The adaptive prefill chunk.** Each prefill chunk is sized so one f32
attention score copy stays at or below 2 GiB: with `keys` positions already
cached, a chunk of `t` tokens costs `heads × t × (keys + t) × 4` bytes, and
`chunk_for` takes the largest `t` within the budget, clamped to 32..=512. At
the depths the smoke runs reached the chunk stays at 512; it only shrinks
deep into a long context.

`PLANK_KV_DEBUG=<file>` appends `flint reconcile:` and `flint generate:`
lines, with the same fields as Gemma's.

## The GPU buffer limit and the Metal lock

On Metal the whole GGUF mapping becomes one no-copy buffer, which requires
the file (rounded up to whole pages) to fit the device's `maxBufferLength`.
On this M5 Max that limit is 80.64 GiB and the 22.9 GB file fits. On a Mac
where it does not, `crates/flint-engine` would fall back to copying every
weight into a buffer of its own, which doubles resident memory. Plan
decision 4 refuses that: `FlintFamily::open` checks the limit before mapping
and fails with `Flint's N GB file is larger than this Mac's GPU buffer limit
(M GB)`, and the engine-fit rule (`enginefit.rs`) disables the catalog
engine in the menu with `needs a N GB GPU buffer (this Mac allows M GB)`, so
the menu never offers what the load would refuse.

candle 0.9.2 runs kernels out of order when two threads encode on one Metal
device (the full analysis is in `FINDINGS.md`). Flint's `Model` therefore
holds a `Mutex` on Metal for a whole `forward`, and the session takes the
same lock around `snapshot` and `restore`. The lock has limits: it is per
`Model`, so two models sharing one `Device` clone, or Flint and Gemma on the
same device, are not excluded from each other; and loading a model (its RoPE
tables and tensor uploads) runs kernels before the lock exists. plank opens
one model per process, so neither case arises in normal use.

## Correctness

`crates/flint-engine/tests/reference_model.rs` checks the tokenizer and the
forward pass on the real file, and every test there is opt-in through
`PLANK_FLINT_GGUF`. The tokenizer matches llama.cpp `970b51c` on 13 of 14
corpus strings (the 14th is the inert-text case above) and on all five
rendered prompts.

The parity criterion is an f64 NumPy oracle, by the user's decision. The
spec had asked for 64 tokens of greedy agreement with llama.cpp, and the
first replacement was a teacher-forced check against it. Both proved weaker
than llama.cpp's own noise: at that commit its NVFP4 matmuls run on the CPU
with activations rounded to Q8_0, its KV is f16, and its results change with
the batch shape, so it disagrees with itself at single positions. It does
agree with plank on the first token of all six cases (within 0.000015 on the
top five probabilities) and over the full greedy runs of three of them,
including 16 tokens after a 20,861-token prompt. The oracle
(`tools/flint_numpy_ref.py`) is an independent float64 forward pass written
from llama.cpp's graph semantics, with weights decoded by the pinned
commit's gguf-py. On all ten checked positions plank's f32 Metal engine
agrees with it on the argmax, and the largest difference on any top-20
probability is 1.8e-6 against a tolerance of 1e-3. Its prompts are short (at
most 22 positions), so the sliding window and deep RoPE positions are only
covered by the coarser llama.cpp agreement. The details are in `FINDINGS.md`
under "Flint against llama.cpp on the real file".

## Speed

Measured on an Apple M5 Max with 128 GiB, on AC power.

| | rate | source |
|---|---|---|
| decode at 32 tokens of context | 60.1 to 62.6 tok/s | `bench_decode`, three runs |
| decode at 512 tokens of context | 52.5 to 54.2 tok/s | `bench_decode`, three runs |
| prefill, 512 tokens | about 491 to 536 tok/s | `bench_decode`, three runs |
| decode in an agent session (~5.1K to 5.6K context) | 37.3 tok/s (486 tokens in 13.0 s) | smoke run, exit stats |
| decode in an agent session, two short replies | 40.1 tok/s (100 tokens in 2.5 s) | smoke run, exit stats |
| cold prefill of plank's system prompt | 366.4 tok/s (5,073 tokens in 13.8 s) | smoke run, exit stats |

For comparison, `llama-bench` at `970b51c` reported 116.42 tok/s prefill and
9.78 tok/s decode on the same machine. The session figures count the
thinking tokens along with the answer. After the first launch the
system-prompt checkpoint restores the 5,059-token prefix from disk, and each
turn prefills only its new suffix.

## Smoke run (2026-10-23, release, `PLANK_NO_DS4=1`, temporary `HOME`, AC power)

One-shot tool use:

```sh
plank -m ~/.plank/models/flint-nvfp4.gguf --ui console \
  -p "Create a file hello.txt containing 'hi', then show me its contents with bash."
```

The model thought, then called
`<tool_call>bash<arg_key>command</arg_key><arg_value>echo "hi" > hello.txt</arg_value></tool_call>`.
In a second pass it thought briefly and called `bash` with `cat hello.txt`,
and in a third it answered with the file's contents. It chose `bash` for the
write rather than the `write` tool; both calls parsed and ran, and
`hello.txt` held `hi`. The run took 28.9 s by plank's own count, including
the model load and the cold system-prompt prefill. The KV log shows every
pass reusing everything but its new suffix:

```
flint generate: prompt=5078 live=5059 reused=5059
flint generate: prompt=5257 live=5220 reused=5220
flint generate: prompt=5323 live=5284 reused=5284
```

Two turns in the plain REPL, piped, with the checkpoint warm ("Say the word
apple." then "Now say it backwards."; the answers were `Apple.` and
`The word "apple" backwards is elppa.`):

```
flint reconcile: 0 spans held, 1 sections in, kept 0
flint reconcile: 2 spans held, 3 sections in, kept 2
flint reconcile: 2 spans held, 3 sections in, kept 2
flint generate: prompt=5066 live=5059 reused=5059
flint reconcile: 4 spans held, 5 sections in, kept 4
flint reconcile: 4 spans held, 5 sections in, kept 4
flint generate: prompt=5108 live=5101 reused=5101
```

On turn 2 all four held spans were kept (system, session context, the first
prompt and the first reply), so the reply's recorded span matched what the
agent rendered back, and only a 7-token suffix was prefilled.

## v1 limits

Flint has no `decide`, so the System-1 gate cannot run and `memory.gate`
stays off; the `memory.gateBias.flint` key exists so a settings file
round-trips. There is no vision (`view_image` is declared and refused when
called, as on a DeepSeek engine without vision) and no MTP or companion of
any kind. `serve --shared-engine` refuses Flint, because the shared host is
built on the ds4 engine's shared model. The Metal lock is per model, as
described above, and a Mac whose GPU buffer limit is smaller than the file
cannot run Flint at all. Finally, nothing has been tested beyond about 21K
tokens: the longest prompt in any check is llama.cpp's 20,861-token `long`
case, and the smoke runs stayed near 5K. The context cap and the adaptive
prefill chunk are there so that deeper contexts fit in memory, but how the
model behaves past 21K is unmeasured.

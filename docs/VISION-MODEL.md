# Vision model plan: interpreting screenshots

Status: design, not implemented.

## Why

plank already accepts images. `imagepaste.rs` catches an empty bracketed paste
(clipboard image) or a pasted file path, downsamples PNGs to 2000px, dedups by
SHA-256 into `~/.plank/image-cache`, and attaches the result to the outgoing
message. But the attachment is only a *path*: the ds4 engine is text-only, so
the model is told "an image exists at `~/.plank/image-cache/<sha>.png`" and can
do nothing with it. A screenshot of a stack trace, a broken layout, or an error
dialog reaches the agent as a filename.

This plan adds a second, local, vision-capable model that turns those pixels
into text the ds4 model can reason about.

## The constraint that shapes everything

`refs/ds4` is antirez's own C kernel (`ds4.c`, `ds4_metal.m`), not llama.cpp.
There is no `mtmd` multimodal path to reuse and no vision tower in the ds4
architecture. A vision GGUF therefore cannot ride on the existing FFI; it needs
its own inference stack.

Three ways to get one were considered:

- Link llama.cpp statically as a second engine beside `ffi.rs`. Self-contained,
  but two inference stacks share one address space and one Metal device, and
  the build grows substantially.
- Use a Rust-native runtime (`mistral.rs`, `candle`). No C build, but a large
  dependency tree, thinner VLM architecture coverage, and its own Metal
  allocator sitting next to ds4's.
- **Spawn `llama-server` as a child process** and talk to it over localhost
  HTTP. Chosen.

The deciding factor is memory. ds4 Flash occupies roughly 87 GB. A vision model
in a separate process can be spawned on first use and killed when idle, and the
OS reclaims every byte. In-process, it is resident for the life of the session.
Process isolation also keeps plank's build exactly as it is today: no new
`build.rs` branch, no new `cfg` gate, no native code.

## Architecture

```mermaid
flowchart TD
    A[User pastes screenshot] --> B[imagepaste.rs<br/>downscale + cache]
    B --> C["Attachment line names the path<br/>and mentions view_image"]
    C --> D[ds4 model emits DSML tool call]
    D --> E["tools::dispatch — arm: view_image"]
    E --> F[vision::describe]
    F --> G{Server running?}
    G -- no --> H[vision::server::spawn<br/>llama-server + mmproj]
    H --> I[Poll /health until ready]
    G -- yes --> I
    I --> J["vision::client — POST /v1/chat/completions<br/>image as data URI"]
    J --> K[Description text]
    K --> L[ToolResult back to ds4 as tool-role observation]
    M[Idle timer expires] --> N[Kill child, reclaim RAM]
```

### New module: `src/vision/`

Three files, mirroring how `engine.rs` splits a trait from its backends:

- **`mod.rs`** — the `VisionEngine` trait (`fn describe(&mut self, image:
  &Path, prompt: &str) -> Result<String, VisionError>`), the `VisionError`
  enum, and a `StubVision` implementation that returns canned text. `StubVision`
  is to vision what `EchoEngine` is to inference: it keeps the whole feature
  testable and the app runnable with no model on disk.
- **`server.rs`** — child-process lifecycle. Finds the `llama-server` binary,
  picks a free localhost port, spawns with `--model`, `--mmproj`, `--port`,
  `--no-warmup`, waits on `/health`, and holds the `Child`. Owns the idle timer
  and a `Drop` impl that kills the child so no orphan survives a plank crash.
- **`client.rs`** — the OpenAI-compatible HTTP call. Builds a
  `/v1/chat/completions` body with a `content` array of one `text` part and one
  `image_url` part whose URL is a `data:image/png;base64,...` URI, parses the
  response, and extracts the message text. Pure request-building and
  response-parsing, no process management, so it unit-tests against a fake
  server.

The split matters: `client.rs` is the part with fiddly wire-format details and
it can be tested without ever spawning a process, while `server.rs` is the part
with OS-level concerns and no parsing.

### Why a trait when there is one backend

The `VisionEngine` boundary is not speculation about future backends. It is
what lets `tools::vision::tool_view_image` be tested with `StubVision` in CI,
where no GGUF and no `llama-server` exist. That is the same reason `Engine`
exists.

## Tool surface

A single new tool, `view_image`:

```
<view_image>
<path>~/.plank/image-cache/abc123.png</path>
<question>What error is shown in this dialog?</question>
</view_image>
```

`path` is required. `question` is optional; when absent the default prompt asks
for a full transcription of visible text plus a description of layout and
visual state, which is what a screenshot usually needs. Output is plain text
framed as a normal tool observation. Failures follow the C convention and start
with `Tool error:`.

Dispatch adds one arm to the match in `src/tools/mod.rs`, alongside
`google_search` and `visit_page`, guarded by a new `tools.vision` setting.

### Advertising it without breaking C parity

The system prompt is frozen by `tests/c_parity.rs` against the C source, so the
tool table in `sysprompt.rs` cannot gain a `view_image` entry. The tool is
advertised the same way MCP tools already are: as text appended after the
frozen prompt, present only when the feature is enabled.

This has a KV-cache consequence worth stating plainly. The Tier 1 fingerprint
is keyed on system prompt text, so the advert block must be byte-stable across
runs, and toggling vision on or off forces one full re-prefill. The advert must
therefore be rendered from settings alone, never from anything that varies with
the machine (no port number, no discovered binary path, no model filename).

## Model selection

### Two jobs, not one

"Interpret a screenshot" hides two different tasks, and the models that are
best at them are different models:

- **Transcription.** A terminal full of a stack trace, a log pane, a diff, a
  config file. The answer is the exact characters, and the failure that matters
  is a wrong one: `l` for `1` in a hash, `rn` for `m` in an identifier, or an
  invented stack frame that reads perfectly plausibly. Dedicated OCR models win
  here, and they win at sizes a general VLM cannot reach.
- **Description.** A broken layout, a misaligned flexbox, a dialog with a
  greyed-out button, a chart that renders wrong. The answer is spatial and
  semantic, and there is no text to extract. Only a general VLM can do this.

ds4 is a coding model, so plank's images skew hard toward the first case. But
the second is precisely the case where a screenshot is the *only* way to convey
the problem, which is why the design does not reduce to OCR alone.

### The gating constraint: llama.cpp support

The sidecar is `llama-server`, so a model is a candidate only if llama.cpp has
a `clip`/`mtmd` implementation for its vision tower **and** a published
`mmproj` GGUF. This disqualifies more models than benchmark scores do, and it
changes release to release. Any table of candidates is a snapshot; verify
`mmproj` availability on the model's Hugging Face repo before committing to
one, and treat a model with weights but no projector as unavailable rather than
as work to do.

### OCR-specialist tier

Sizes are parameter counts; on-disk figures assume Q4-class quantization plus
projector.

| Model | Size | Notes |
| --- | --- | --- |
| **GLM-OCR** | 0.9B | Tiny and fast; markdown output; tuned for documents and code. The floor of what is useful, and the cheapest thing to spawn. |
| **PaddleOCR-VL** | 0.9B | Same weight class, strong multilingual and table structure. Stronger layout handling than GLM-OCR, weaker free-form instruction following. |
| **Granite-Docling** | 258M | IBM; absurdly small, document-structure focused. Interesting for a always-resident tier; too narrow as the only model. |
| **MinerU2.5** | 1.2B | Two-stage layout-then-recognize; excellent on dense tables and formulas. |
| **dots.ocr** | 1.7B | Layout detection and recognition in one model; among the best small open document OCR available. |
| **Nanonets-OCR2** | 3B | Qwen2.5-VL derived; markdown with semantic tagging (checkboxes, signatures, LaTeX). |
| **DeepSeek-OCR** | 3B | Optical context compression; very high throughput on long documents. |
| **olmOCR-2** | 7B | AllenAI, Qwen2.5-VL derived, fully open pipeline. The accuracy ceiling of this tier, at real cost. |

For plank's actual traffic, the useful span is GLM-OCR 0.9B at the bottom and
dots.ocr 1.7B as the accuracy upgrade that still spawns in a couple of seconds.
Everything at 3B and above is buying document-conversion quality that a
terminal screenshot does not need.

### General-VLM tier

| Model | Size | Notes |
| --- | --- | --- |
| **Moondream 3** | ~2B active (MoE) | Small, fast, unusually good at pointing at UI elements. |
| **Qwen3-VL** | 4B / 8B / 30B-A3B | The default recommendation. Strong OCR *and* strong grounding, so it covers both jobs adequately; the 30B MoE runs at 3B-active cost if the RAM is there. |
| **InternVL3.5** | 2B–38B | Consistently at or above Qwen3-VL on document and chart benchmarks; llama.cpp coverage historically lags. |
| **MiniCPM-V 4.5** | 8B | High-resolution tiling, mature llama.cpp support, well-tested on dense text. |
| **Gemma 3** | 4B / 12B | Excellent general reasoning, comparatively weak on dense small text. Not the right pick for a terminal. |

### Recommendation

Ship **one** model, configurable, defaulting to **Qwen3-VL 4B at Q4**. It is
the smallest thing that does not fail the description case, and its OCR is good
enough that a second model earns its complexity only under measurement.

Then make the second model *possible* rather than mandatory: `[vision]` takes a
`model`/`mmproj` pair, so pointing plank at GLM-OCR 0.9B is a settings change,
not a code change. A user whose images are all terminals gets a 1 GB, sub-second
sidecar by editing two lines.

A two-model routing tier — OCR specialist first, general VLM on a
low-confidence signal — is explicitly deferred. There is no reliable confidence
signal out of `llama-server`, so routing would have to guess from the image or
from the question text, and guessing wrong is worse than being uniformly
mediocre. Revisit only with real usage to look at.

### Accuracy notes that matter more than the rankings

Two properties dominate real-world accuracy on console output, and neither
appears in benchmark tables:

- **Input resolution.** Monospace terminal text at 1x is where every model in
  the 1B class falls apart. Capture at 2x and do not downscale below roughly
  1600px wide. The 2000px cap already in `imagepaste.rs` is close to right, but
  it is a cap on the long edge, which means a wide-but-short terminal grab
  survives and a tall one gets crushed. Worth checking before blaming a model.
- **Contrast polarity.** Light-on-dark degrades OCR measurably across the small
  models, because their training data is overwhelmingly dark-on-light documents.
  Nothing plank can fix, but it belongs in the error text when a transcription
  comes back garbled.

And one behavioral property: at these sizes, a model asked to *reason* will
hallucinate instead of transcribing. This is why the default prompt in the tool
surface above asks for verbatim transcription plus a layout description, and
why a caller-supplied `question` is passed through rather than merged into a
combined instruction.

## Model acquisition

Two files are needed: the quantized VLM GGUF and its companion `mmproj`
projector. Both land in `~/.plank/vision/`. A Qwen3-VL 4B at Q4 is roughly 3 GB
plus a few hundred megabytes of projector, small enough to coexist with ds4
during the seconds it is alive; a GLM-OCR-class OCR specialist is closer to
1 GB.

`download.rs` already streams a Hugging Face file with a progress gauge and a
consent prompt; it gets a second entry point that fetches a named pair rather
than the single hardcoded ds4 GGUF. The prompt fires the first time
`view_image` is called with no model present, and declining is a normal
outcome: the tool returns `Tool error: no vision model installed` and the agent
carries on.

The `llama-server` binary is *not* vendored. plank looks for it on `PATH`, then
at a configured path. If it is missing, the error text names the fix
(`brew install llama.cpp`).

## Settings

A new `[vision]` section in `Settings`, following the existing shape in
`settings.rs`:

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Master switch; also gates the prompt advert |
| `model` | `~/.plank/vision/model.gguf` | VLM weights |
| `mmproj` | `~/.plank/vision/mmproj.gguf` | Projector |
| `binary` | *(PATH lookup)* | Explicit `llama-server` path |
| `idle_timeout_secs` | `300` | Shut the child down after this long unused |
| `max_tokens` | `1024` | Cap on description length |

Opt-in by default. Vision costs disk, RAM, and an external binary, and most
sessions never paste an image, so it should not switch itself on. `enabled`
belongs in `startup_note` so an active vision config is visible at launch.

## Integration with the paste path

`imagepaste.rs` needs no structural change. Only the attachment line the user
and model see changes: when vision is enabled it says the image can be read
with `view_image`, and when disabled it stays exactly as it is today. The
content-addressed cache path is already a stable handle, which is precisely
what the tool wants as its argument.

`upgrade.rs` already drops `~/.plank/image-cache` on a major version
transition. `~/.plank/vision/` should be exempt from that sweep: the weights
are large, slow to fetch, and have no format coupling to plank's caches.

## Error handling

Every failure is a tool observation, never a panic and never a crash of the
session:

- `llama-server` not found → error names the install command.
- Model or projector missing → error offers the download.
- Spawn fails or `/health` never comes up within a timeout → error reports the
  child's stderr tail, captured to `errlog.rs`.
- Request times out → the child is killed rather than left wedged, so the next
  call gets a clean server.
- Path is not an image, or is not under a readable location → rejected before
  any process work, reusing `detect_media_type` from `imagepaste.rs`.

Large images are downscaled through the same `image` crate path
`imagepaste.rs` already uses before base64 encoding, so a 5K screenshot does
not become a multi-megabyte JSON body.

## Testing

No test requires a model, a GPU, or a network:

- `client.rs`: build a request for a known image and assert the JSON shape and
  data URI; parse recorded response bodies, including error and truncated
  forms.
- `mod.rs`: `StubVision` drives `tool_view_image` end to end through
  `tools::dispatch`, covering the success arm and each error arm.
- `server.rs`: port selection and argument-vector construction are pure and
  tested directly; spawn itself is exercised with a fake binary script that
  answers `/health`, kept out of CI's default run.
- Prompt advert: a test asserts the advert text is byte-stable and absent when
  `enabled = false`, protecting the Tier 1 fingerprint.

## Non-goals

- No second *conversational* model. The vision model answers one question about
  one image and is not part of the transcript or the KV cache.
- No video, no PDF, no multi-image comparison in a single call.
- No automatic screenshot capture. The user pastes; plank does not grab the
  screen.
- No remote vision API fallback. This feature is local or it is off.
- No routing between an OCR specialist and a general VLM. One model per
  session, chosen in settings, for the reason given under Recommendation.
- No screen-reading loop. `view_image` answers one question about one file the
  user already put there; it is not a way for the agent to watch a UI.

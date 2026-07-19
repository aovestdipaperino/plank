# Remote Engines — Hosting plank's engine and driving third-party LLMs

Design document for GitHub issue #26 ("Remote LLM support via llms-sdk") and the
personal note `vault/hosting-support.md`. It covers the two distinct "remote"
flavors, why they are **two Engine implementations rather than one**, and how
each maps onto plank's existing invariants: the narrow `Engine` boundary
(`src/engine.rs`), the DSML-in-band tool protocol, KV-cache discipline, the
byte-parity DS4 system prompt, and the single streaming path through
`viz::StreamRenderer`.

Status: **flavor (a) implemented** (steps 1–4 of §5); flavor (b) not yet started.
`make_engine` (`src/main.rs`) now selects `RemoteDs4Engine` when `--remote URL`
is given (all platforms), and `plank serve` hosts any `Engine` (the Metal
`Ds4Engine` on a real box, `EchoEngine` elsewhere) over HTTP+SSE. The sync
`ureq` client matches the synchronous `Engine` contract directly, so no async
runtime is pulled in for flavor (a); the tokio bridge and structured-input
boundary remain TODO for flavor (b) (see `src/remote/mod.rs`). Modules:
`src/remote/{mod,proto,ds4_client}.rs`, `src/serve.rs`; tests in
`tests/remote_ds4.rs` (mock server + real serve↔client round-trip on EchoEngine).

## 1. Goal

Let plank run against inference that does not live in-process, in two forms
that users conflate but which are architecturally different:

- **(a) Remote-hosted ds4** — plank's own Metal/CUDA engine running on a rented
  GPU box (GCP GPU VM, RunPod pod, any reachable host), driven over the network
  by a thin plank client. This is the primary intent of
  `vault/hosting-support.md`: the model is too large to hold locally (~82 GB
  resident for the default quant, see `require_min_ram` in `src/main.rs`), so
  offload the whole engine, keep everything else identical.
- **(b) Third-party provider APIs** — OpenAI-compatible chat completions and
  Anthropic Messages, via the `llms-sdk` crate (`refs/llms-sdk`), so plank's
  agent loop and tools work against `gpt-*`, `claude-*`, and any
  OpenAI-compatible gateway (vLLM, Ollama, OpenRouter, Together, …).

Both slot behind `Engine` (`generate`, `warm_system_prompt`, `count_tokens`,
`ctx_size`, `model_name`) so the UI, session, compaction, and tool layers are
untouched — the design principle "narrow engine surface" from
`docs/ARCHITECTURE.md`. The hard part is not the trait; it is making (b)'s
native tool calls, model-specific prompt, and server-side token accounting
honor plank's DSML-and-byte-parity assumptions without a second code path in the
UI.

## 2. Prior art and context

### 2.1 The BTW "no second stream" principle applies here too

`docs/BTW-DESIGN.md` established that plank does not add a parallel generation
channel when it can reuse the existing one at near-zero cost — OpenClaw itself
degrades to a single sequential path on non-forkable runtimes. The remote-engine
analogue is **"no second tool-call source."** Issue #26's open question frames
the choice as *"translate provider tool-call deltas into the DSML stream, or
teach the dispatch layer a second tool-call source."* This design takes the
first option unconditionally: everything downstream of `Engine::generate`
(`viz::StreamRenderer` → `DsmlParser` → `dispatch_all`) stays byte-identical to
the local path. A remote engine that speaks native tool calls **re-emits them as
DSML text** into the `EngineEvent::Text` stream; `stream.finished().calls`
(`src/ui.rs:382`) is populated by the same parser for every backend. One tool
dispatch path, one renderer, one transcript format.

### 2.2 What the transcript boundary actually is

The agent flattens the session to a single text blob via `render_transcript`
(`src/ui.rs:150`) — `[system]`/`[user]`/`[assistant]` sections with the DS4
system prompt (tool instructions written *as DSML*) prepended — and passes that
one `&str` to `Engine::generate`. This is perfect for (a) (the remote ds4
tokenizes the identical bytes) but is a genuine impedance mismatch for (b),
which wants a structured `Vec<Message>` plus a `Vec<Tool>` and its own system
prompt. §4.4 addresses this head-on; it is the central design decision.

### 2.3 llms-sdk surface plank will build on

From `refs/llms-sdk/src` (`rust@0.1.0`):

- `LLM::new(RetryPolicy)`; `LLM::stream_response(LLMRequest) -> LLMStream`
  (`Pin<Box<dyn Stream<Item = Result<LLMStreamingResponse, …>>>>`) and the
  non-streaming `respond`.
- `LLMRequest { api_type: ApiType::{OpenAI,Anthropic}, base_url, api_key,
  model, messages, max_output_tokens, temperature, top_p, reasoning_effort,
  prompt_cache_ttl, stream, tools, tool_choice, parallel_tool_calls, … }`, with
  a fluent `builder()`.
- `Message { role, content: Vec<MessagePart> }`; `MessagePart::{Text, Thinking,
  ToolCall(ToolCallPart{id,name,arguments}), ToolResult(ToolResultPart), …}`.
- `Tool::from_parameters_value(name, description, json_schema)` and
  `ToolChoice`.
- Streaming variants `LLMStreamingResponse::{Delta(LLMStreamingDelta),
  ThinkingDelta(LLMThinkingDelta), ToolDelta(LLMToolDelta{tool_call_id, name,
  partial_arguments}), Complete(LLMStreamingComplete{ …, usage:
  Option<LLMUsage>, tool_calls: Option<Vec<ToolCallPart>>})}`.
- `LLMUsage { input_tokens, output_tokens, cache_read_tokens,
  cache_write_tokens, … }` — the source for exact `ctx_used`.
- `RetryPolicy` (max_retries, min/max interval, jitter) for transient retries.

The SDK is async (`tokio` + `futures::Stream`); plank's `Engine::generate` is
sync and polls an `interrupt` closure between tokens. Bridging those (§4.6) is a
per-engine runtime concern, not a trait change.

## 3. One abstraction or two? — Two impls, one runtime helper

**Decision: two Engine implementations, not one `RemoteEngine`.** They share
almost nothing at the semantic level:

| Concern | (a) `RemoteDs4Engine` | (b) `ProviderEngine` |
|---|---|---|
| Wire format | plank's own protocol (serde) over HTTP+SSE | llms-sdk / provider APIs |
| Tool calls | DSML in-band, byte-identical | native → translated to DSML |
| System prompt | byte-parity DS4 prompt (server owns it) | model-specific variant |
| KV / caching | real server-side KV session + `sysprompt.kv` | provider prompt caching or no-op |
| Tokenizer | real ds4 tokenizer via `/tokenize` | provider usage + estimate |
| Input | flat `render_transcript` text (verbatim) | structured messages + tools |

Forcing both under one struct would produce a type that is half dumb-transport
and half semantic-adapter, with a mode flag branching every method — the
anti-pattern. Instead:

- `src/remote/mod.rs` — a shared **async runtime bridge**: a lazily-created
  `tokio` current-thread runtime (or a dedicated blocking thread), an
  interrupt-polling stream-drain helper `pump_stream(stream, interrupt,
  on_event)`, reqwest client construction, and TLS/timeout config. Both engines
  call into it; neither duplicates the sync↔async plumbing.
- `src/remote/ds4_client.rs` — `RemoteDs4Engine` (flavor a).
- `src/remote/provider.rs` — `ProviderEngine` (flavor b).
- `src/remote/proto.rs` — serde wire types for flavor (a) only (shared with the
  server).
- `src/serve.rs` — the `plank serve` subcommand for flavor (a).

`make_engine` (`src/main.rs`) gains two mutually-exclusive branches selecting
these; both are available on **all platforms** (no `ds4_engine` cfg gate — they
are pure Rust + HTTP), which is a side benefit: Linux/Windows users get plank
against a provider or a remote ds4 box without building the C engine.

## 4. Detailed design

### 4.1 Flavor (a): `plank serve` + `RemoteDs4Engine`

This is `vault/hosting-support.md` realized. The insight that makes it cheap:
**the client is a dumb transport and the server is just `Ds4Engine` behind a
socket.** No protocol drift, no prompt divergence, and KV discipline —
common-prefix reuse across turns and the `sysprompt.kv` checkpoint — lives
server-side, unchanged.

**Server (`plank serve --model … --listen 0.0.0.0:PORT [--token …]`):**

- Reuses `Ds4Engine::open` and holds one live session (single-tenant v1, which
  matches the one-user plank workflow). An async HTTP server (axum/hyper) exposes:
  - `GET /info` → `{ model_name, ctx_size, protocol_version }`.
  - `POST /generate` → request `{ session_id, transcript, opts }`; response is
    an SSE stream of `Event::Prefill{done,total,tps}`, `Event::Text{s}`, and a
    terminal `Event::Done{stats}` mapping 1:1 onto `EngineEvent` +
    `GenerationStats`. `opts` serializes `GenerationOptions` (incl.
    `think_mode`, seed, sampling).
  - `POST /warm` → streams prefill progress while the server runs
    `warm_system_prompt` against its own `sysprompt.kv` (the checkpoint fast
    path stays on the box that has the KV bytes).
  - `POST /tokenize` → `{ n_tokens }` using the real ds4 tokenizer, so
    `count_tokens` is exact (matters for compaction thresholds in
    `src/compact.rs` — the open question in the vault note).
  - `DELETE /generate/{id}` → cancel signal for interrupt.
- The C engine's `greedy` closure semantics (`src/engine.rs` doc: argmax while
  tool-call stanzas stream) are reproduced server-side. Because greedy state is
  derived from the streaming parser and the server runs the *same* parser over
  its own output, this is a server-local concern; the client does not send
  greedy hints. (If we later want the client-side DSML parser to drive greedy,
  we add a `greedy` boolean to the SSE protocol — deferred; v1 lets the server
  own it, identical to in-process behavior.)

**Client (`RemoteDs4Engine`, selected by `--remote URL`):**

- `generate` POSTs `{session_id, transcript, opts}`, opens the SSE stream, and
  in `pump_stream` maps each server event onto `on_event`, polling `interrupt`
  between chunks; on interrupt it fires `DELETE /generate/{id}`, drops the
  stream, and returns `GenerationStats{ interrupted: true, .. }` — the same
  contract as local prefill interruption.
- `warm_system_prompt` → `POST /warm`; the `checkpoint: Option<&Path>` argument
  is **ignored client-side** (the checkpoint is a server file), and the method
  streams progress and returns whether a prefill happened, per the trait doc.
- `count_tokens` → `/tokenize` with a short in-memory cache; falls back to the
  trait's default `len()/4` on transport error so accounting degrades rather
  than fails.
- `ctx_size` / `model_name` → cached from the `/info` handshake at construction.

**Reconnect/resume:** dropped connections are safe by construction — the next
`generate` re-sends the full rendered transcript, and the server's
common-prefix `ds4_session_sync` re-prefills only past the divergence point,
exactly as an in-process turn would. A mid-stream drop surfaces as a normal
`EngineError` and the turn can be retried; no client-side resume token needed
for v1.

**Prompt / tools / DSML:** *nothing special.* The bytes plank renders are the
bytes ds4 tokenizes; DSML tool calls stream back as text and the existing viz
pipeline parses them. Byte parity (`tests/c_parity.rs`) still holds because the
same `render_transcript` and system prompt are in play. This is why (a) is the
low-risk flavor and ships first.

### 4.2 Flavor (b): `ProviderEngine` over llms-sdk

Selected by `--provider openai|anthropic` (+ `--model`, `--base-url`,
`--api-key`). This is a real adapter, and every one of issue #26's open
questions bites here.

**Streaming into `viz::StreamRenderer` (the "no second stream" translation):**

`generate` builds an `LLMRequest` (§4.4), calls `stream_response`, and in
`pump_stream` translates each `LLMStreamingResponse` into `EngineEvent`s so the
existing renderer needs zero changes:

- `Delta{delta: Some(s), ..}` → `EngineEvent::Text(s)` verbatim.
- `ThinkingDelta{delta: Some(s)}` → emitted as `EngineEvent::Text` **wrapped in
  synthetic `<think>…</think>`** so `StreamRenderer`'s existing thinking-split
  logic (it keys off `<think>` byte markers, `src/viz.rs`) routes it to
  `think_text`. The wrapper open/close is emitted once around a contiguous run
  of thinking deltas.
- `ToolDelta` / `Complete.tool_calls` → **synthesized DSML**. When the provider
  finalizes a tool call (`ToolCallPart{name, arguments}` where `arguments` is
  JSON), the engine renders the canonical DSML stanza
  (`<｜DSML｜tool_calls>…<｜DSML｜invoke name="…">…<｜DSML｜parameter…>` per the
  markers in `src/viz.rs`), mapping JSON argument fields to DSML `parameter`
  tags, and emits it as `EngineEvent::Text`. The renderer paints the normal
  tool banner and `DsmlParser` produces the executable `ToolCall`. The provider
  `tool_call_id` is retained in a side map only if we later need to satisfy
  providers that require echoing ids on tool-result turns (§4.4).
- `Complete.usage` → drives `GenerationStats.ctx_used`
  (`input_tokens + output_tokens`) and generated-token count; exact, no
  estimate.

Because we stream text deltas straight through and DSML is synthesized at
tool-call finalization, the output the user sees is indistinguishable in shape
from a local turn.

**Prefill progress:** provider APIs report no prefill. `generate` emits a single
synthetic `EngineEvent::Prefill{done: total, total, tps: 0}` derived from
`count_tokens(transcript)` (or the `Complete.usage.input_tokens` once known) so
the progress bar completes instead of hanging. No fake incremental animation —
one done-event, honest about the fact that prefill is opaque.

### 4.3 The tool-call format decision, stated plainly

- **Provider request:** register plank's tool table as native
  `llms-sdk::Tool`s (JSON-schema’d), with `tool_choice: auto` and
  `parallel_tool_calls: false` (plank dispatches serially and re-feeds; parallel
  batches would complicate the single-transcript reconciliation). Native tools
  give far better reliability than hoping a non-DS4 model reproduces DS4's exact
  DSML syntax.
- **Provider response:** translate native tool calls **back to DSML** (§4.2) so
  dispatch has one source. This is the issue's option 1, and it keeps the entire
  `dispatch_all` / `dsml.rs` / `viz.rs` stack backend-agnostic.
- **Rejected:** teaching `run_turn`/`worker_turn` a second tool-call source.
  That would fork the turn loop across backends and duplicate the very code the
  narrow engine boundary exists to protect.

The tool table itself must be extracted into structured form. Today the tool
schemas live only as prose inside the DS4 system prompt text (`sysprompt.rs`).
§4.4 requires a machine-readable tool registry; this is real work and is the
main reason (b) is Phase 3, not Phase 1.

### 4.4 The transcript-boundary problem (central decision)

`Engine::generate(transcript: &str, …)` hands the provider engine a flattened
blob with the DS4 system prompt embedded. Two options:

- **Option A — reparse the flat text** back into `Vec<Message>` by splitting on
  `[system]/[user]/[assistant]/[tool_result]` markers. Fragile (marker
  collisions, tool-result framing), and it cannot recover structured tool_call
  ids. Rejected as the long-term answer.
- **Option B (recommended) — widen the engine input** so structure-hungry
  backends get structure. Introduce an enum passed to `generate`:

  ```
  enum Prompt<'a> { Flat(&'a str), Structured(&'a StructuredTurn) }
  ```

  where `StructuredTurn` borrows the session's message list, the resolved
  system prompt, and the tool registry. `Ds4Engine`/`RemoteDs4Engine`/`Echo`
  only ever read `Flat` (the caller passes `render_transcript` output as today —
  zero behavior change, byte parity intact). `ProviderEngine` reads
  `Structured`. The `Agent` decides which to pass via a cheap
  `engine.wants_structured() -> bool` (default `false`).

  To keep the change small and the trait honest, `StructuredTurn` is built by a
  new `session::to_messages()` that already exists in spirit (the session *is* a
  message list; only `render_transcript` currently flattens it). This is the one
  boundary widening the design accepts, and it is additive — the local path is
  unaffected.

**System prompt for (b):** the byte-parity DS4 prompt is model-specific and
must **not** be sent to a provider model (it teaches DSML syntax and DS4-only
conventions, and would waste tokens / confuse native tool use). `sysprompt.rs`
gains a **provider prompt variant**: the same behavioral guidance
(role, safety, tool-usage norms, session context injection from `context.rs`)
minus the DSML tool-call instructions (native tools replace them) and minus any
DS4-specific framing. This variant is *not* under `c_parity.rs` — it is plank's
own text, free to evolve. Concretely: a `SystemPrompt::{Ds4, Provider}` selector
in `sysprompt.rs`, chosen by the engine kind.

**Tool results back to the provider:** when plank re-feeds tool observations,
the structured path emits `MessagePart::ToolResult{tool_call_id, result}` paired
to the retained id (Anthropic requires this; OpenAI-compatible needs the
`tool_call_id`). This is why §4.2 retains the id side-map.

### 4.5 KV / prompt-cache discipline mapping

- **(a)** real KV: server keeps one live session; `warm_system_prompt` and
  common-prefix reuse are genuine, identical to in-process. `sysprompt.kv` lives
  on the server. The client honors the same "reuse only a genuinely matching
  prefix, never trust a stale checkpoint" invariant transitively.
- **(b)** there is no client KV. Two sub-cases:
  - **Anthropic**: set `prompt_cache_ttl` and rely on cache_control over the
    stable prefix (system prompt + early transcript). Because plank always
    appends and never reorders (a BTW invariant, `docs/BTW-DESIGN.md` §7.3), the
    prefix is cache-friendly by construction — the opencode cache lesson carries
    over. `warm_system_prompt` becomes a **no-op returning `false`** (nothing to
    prefill client-side), optionally issuing a tiny priming request to populate
    the provider cache; v1 keeps it a pure no-op.
  - **OpenAI-compatible**: automatic server-side prefix caching where offered;
    plank does nothing beyond keeping the prefix stable. `warm_system_prompt`
    no-op.
  - `LLMUsage.cache_read_tokens/cache_write_tokens` are surfaced in the status
    bar so the user can see caching working.

### 4.6 Sync↔async, interrupt, timeouts, retries

- **Runtime:** `src/remote/mod.rs` owns a `tokio` current-thread runtime created
  once. `generate` calls `runtime.block_on(pump_stream(...))`. `pump_stream`
  drives the `Stream` with `futures::StreamExt::next`, and between items checks
  `interrupt()`; on true it drops the stream (cancels the request) and returns
  `interrupted: true`. This keeps the sync `Engine` contract intact — the UI
  worker thread and the SIGINT atomic (`interrupt.rs`) are untouched.
- **Timeouts:** connect + idle-read timeouts on the reqwest client; a stalled
  provider surfaces as `EngineError`, which the turn loop already renders as a
  tool/engine error (`src/ui.rs:383`).
- **Retries:** (b) uses llms-sdk `RetryPolicy` (bounded, jittered) for transient
  5xx/429 **before** the first byte streams; once streaming has begun a failure
  is non-idempotent and is surfaced, not retried (partial output already
  rendered). (a) does not retry mid-turn either; it relies on re-sync on the
  next turn (§4.1).
- **Interrupt for (a):** `DELETE /generate/{id}`; for (b): drop the stream
  (the SDK/reqwest aborts the HTTP request).

### 4.7 Auth and config surface

CLI (`src/config.rs`, mirrored in `--help` and both slash/`tui_slash` paths per
the two-UI-path rule):

- `--remote URL` → `RemoteDs4Engine` (flavor a). Requires `https://`/`wss://`
  or an explicit `--insecure` for localhost; token via `PLANK_REMOTE_TOKEN` env
  or `--remote-token`. Mutually exclusive with `-m/--model/--metal/--cuda/--cpu`
  and with `--provider`.
- `--provider openai|anthropic` (flavor b) + `--model NAME`, `--base-url URL`
  (for OpenAI-compatible gateways), `--api-key` or env
  (`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, matching llms-sdk conventions).
  Mutually exclusive with the local and `--remote` selectors.
- `plank serve --model … --listen … [--token …] [--tls-cert/--tls-key]`
  (server, flavor a).
- Config file (`~/.plank/…`, extending the existing `AgentConfig`): a
  `[remote]` / `[provider]` section with url, provider, model, base_url, and an
  env-var name for the key (never the key literal in a dotfile). Precedence:
  CLI > env > config, matching existing flag handling.

Keys are read from env by default; the config stores only the env var *name*.
TLS is required for any non-loopback host (RunPod/GCP boxes are
public-internet-facing per the vault note).

### 4.8 Status bar and model name

`model_name()` returns the `/info` name for (a) and `provider:model` for (b),
shown in the status bar (`status.rs`/`statusbar.rs`). For (a) a ping-derived
latency hint and prefill/gen t/s flow through events as usual. For (b) the
per-turn `LLMUsage` (input/output/cache tokens, and if exposed, cost) is shown
once the `Complete` frame arrives.

## 5. Implementation plan

Ordered; each step independently landable and testable with a mock (§6).

1. **Runtime bridge + wire types.** `src/remote/mod.rs` (tokio runtime,
   `pump_stream`, reqwest/TLS config) and `src/remote/proto.rs` (serde types for
   flavor a). No engine yet; unit-test `pump_stream` interrupt semantics against
   an in-process mock stream.
2. **`plank serve` (flavor a server).** `src/serve.rs` wrapping `Ds4Engine`;
   `/info`, `/generate` (SSE), `/warm`, `/tokenize`, cancel. Gated behind
   `ds4_engine` (the server needs the real engine). Manual smoke on a Metal box.
3. **`RemoteDs4Engine` client + `--remote`.** Implements `Engine`; wired into
   `make_engine` and `config.rs`. Interrupt via `DELETE`. Tested against an
   **EchoEngine-backed mock server** (§6) so it runs in CI with no model. This
   is the first end-to-end remote path and the primary deliverable of the vault
   note.
4. **Auth + TLS + reconnect polish** for (a): bearer token, https enforcement,
   graceful error surfacing in the TUI, next-turn re-sync validated.
5. **Structured input boundary.** `Prompt::{Flat,Structured}`,
   `Engine::wants_structured`, `session::to_messages`, and the
   `SystemPrompt::Provider` variant + a machine-readable tool registry extracted
   from `sysprompt.rs`. Local engines keep passing `Flat` — assert byte parity
   unchanged.
6. **`ProviderEngine` (flavor b)** over llms-sdk: request build, delta→
   `EngineEvent` translation, **native-tool → DSML synthesis**, thinking-wrap,
   usage accounting, `--provider` wiring, RetryPolicy, prompt-cache TTL.
7. **Docs:** `docs/REMOTE-ENGINE-DESIGN.md` (this), a README quickstart for
   RunPod/GCP (flavor a) and provider setup (flavor b).

Deferred to v2 (design so config can grow into them): ds4 distributed mode
(`Ds4DistributedOptions`, local plank as coordinator over layer-sharded remote
boxes); multi-tenant serve; serverless/queue-based RunPod endpoints; WebSocket
transport if same-connection cancel proves nicer than `DELETE`.

## 6. Testing

Unit / integration (`cargo test --lib` + `tests/`, no model, CI-safe):

- **Mock server (`EchoEngine`-style).** A `tests/mock_remote.rs` axum server
  backed by `EchoEngine` implementing the flavor-(a) protocol (`/info`,
  `/generate` SSE, `/tokenize`). `RemoteDs4Engine` drives it end-to-end: assert
  events map through, `ctx_used`/`model_name` come from `/info`, and a turn
  completes. This is the remote analogue of how `EchoEngine` exercises the local
  turn loop.
- `remote_ds4_interrupt`: mock streams slowly; interrupt closure flips true;
  assert the client cancels and returns `interrupted: true`.
- `remote_ds4_resync`: drop the mock mid-stream; assert the next `generate`
  re-sends the full transcript and succeeds (no client resume state).
- `pump_stream_interrupt_unit`: pure, no HTTP.
- **ProviderEngine translation (mock llms-sdk stream)** — feed a scripted
  `Vec<LLMStreamingResponse>`:
  - `provider_text_passthrough`: `Delta`s → `EngineEvent::Text` verbatim.
  - `provider_thinking_wrap`: `ThinkingDelta`s bracketed in `<think>…</think>`
    and routed to `think_text` by a real `StreamRenderer`.
  - `provider_toolcall_to_dsml`: a `ToolCallPart{name,arguments}` synthesizes a
    DSML stanza that `DsmlParser` parses into the expected executable
    `ToolCall` — the crux test proving "one tool-call source".
  - `provider_usage_accounting`: `Complete.usage` → `GenerationStats.ctx_used`.
- `structured_boundary_local_unchanged`: local engines receive `Flat`;
  `render_transcript` bytes and `c_parity` fixtures unchanged.
- `provider_system_prompt_omits_dsml`: the `Provider` variant contains none of
  the DSML tool-call instruction text.
- Config/CLI: mutual-exclusion of `--remote`/`--provider`/`-m`; env-var key
  resolution; https enforcement; both slash paths.

Parity (`tests/c_parity.rs`): unaffected — flavor (a) reuses the exact bytes;
flavor (b)'s prompt is explicitly out of parity scope. No new fixtures.

Manual (real): flavor (a) against a RunPod/GCP ds4 box — confirm prefill t/s,
interrupt, and next-turn prefix reuse (watch the status line). Flavor (b)
against Anthropic and an OpenAI-compatible gateway — confirm tool calls dispatch
and cache_read tokens appear after the first turn.

## 7. Constraints and invariants

1. **One tool-call source.** No backend adds a second dispatch path; every
   engine's tool calls reach `dispatch_all` as DSML parsed by `dsml.rs`
   (§2.1, §4.3).
2. **Local path is byte-identical.** `render_transcript` output, the DS4 system
   prompt, and `c_parity` fixtures are untouched; the `Prompt::Flat` arm is the
   old behavior exactly (§4.4).
3. **Never send the DS4 byte-parity prompt to a provider model** — use the
   `SystemPrompt::Provider` variant (§4.4).
4. **Transcript is always append-only, never reordered** — required for
   provider prompt caching and for flavor-(a) common-prefix re-sync (§4.5).
5. **Interrupt honors the sync contract**: `generate` polls `interrupt` between
   received chunks and returns `interrupted: true`, never blocking the UI
   thread; the async runtime is an implementation detail (§4.6).
6. **Keys never persisted in config** — env-var name only; TLS required for
   non-loopback (§4.7).
7. **Two UI paths**: every `config.rs`/slash change lands in both `slash` and
   `tui_slash`; engine selection is front-end-agnostic (all three front-ends
   get remote engines for free through `make_engine`).
8. **Degrade, don't fail**: `count_tokens` falls back to `len()/4` on transport
   error; a missing `/tokenize` or usage field never aborts a turn.

## 8. Non-goals

- **A concurrent second generation stream.** Same reasoning as
  `docs/BTW-DESIGN.md` §8; unrelated to remoting.
- **ds4 distributed / layer-sharded inference** (`Ds4DistributedOptions`, plank
  as coordinator). Bigger lift; v2. Config is designed not to collide with it.
- **Multi-tenant `plank serve`.** v1 is single-session, matching the one-user
  plank workflow; the vault note lists multi-client as an open question, not a
  v1 requirement.
- **Serverless / queue-based provider endpoints** (RunPod serverless): a
  different adapter shape; out of scope.
- **Reparsing flat transcripts into structured messages** (§4.4 Option A):
  rejected in favor of widening the input boundary.
- **Provider parallel tool calls** and **multimodal input** (images/audio/docs
  via llms-sdk `MessagePart`): the plumbing exists in the SDK but plank's
  tool loop and `--features images` story are a separate design.
- **Non-streaming (`respond`) mode.** plank is a streaming UI; always
  `stream: true`.

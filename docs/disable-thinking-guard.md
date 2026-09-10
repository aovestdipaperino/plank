# Closed-think recovery after a reasoning-guard stop

Status: proposed, 2026-09-10. Not implemented.

## The problem this fixes

Two sessions on 2026-09-10 ran the same request, `do a code review` on the
tommaso repository, at temperature 0 with thinking hidden
(`repro-loop-1789060243`, `repro-loop-1789062437`; both analysed in
`LOOP-FINDINGS.md`, "The review that was written in the wrong place"). In both
the reading rounds were healthy and the synthesis pass then wrote the review
inside `<think>`. In the first the user interrupted at 12m42s; in the second
the exact-cycle rung fired. Then, in both, the recovery pass did exactly what
the guard's error text asked. It opened with "I have enough to write a code
review. Let me synthesize findings." And it synthesized inside `<think>` again,
until it cycled and the trip cap ended the turn. Eighteen and fourteen minutes,
no output, twice.

The model follows the error's instruction. What it cannot do is follow it in
the right place, because every DS4 pass begins inside a `<think>` block that
the chat template opens in the assistant prefix, and closing that block is the
one step the model keeps not taking. At temperature 0 the same prompt produces
the same pass, so rewording the error will not move it. The fix is to make the
recovery pass start after `</think>`, where the only thing it can write is the
answer or the tool calls.

The draft rung and the "Write findings as you find them" prompt rule shipped
the same evening target the *first* drafting pass. This targets the pass after
a stop, and makes every reasoning-rung stop productive instead of a restart.

## Why it is cheap: the engine already splits the two prefixes

`ds4engine.rs` derives the think mode from two sources, and they are already
independent:

- The **effort preamble** (`THINK_LOW_PREFIX` / max-effort prefix, ahead of the
  system prompt) comes from the engine's own `self.think`, set by
  `Engine::set_think_mode`. It is at the head of the KV prefix; changing it
  forces a full re-prefill, which is why `set_think_mode` only rebuilds when
  `effort_prefix()` differs.
- The **assistant prefix** (`<｜Assistant｜><think>` or, for `Off`,
  `<｜Assistant｜></think>`) comes from `opts.think_mode` on the
  `GenerationOptions` passed to *that* `generate` call: `build_prompt(transcript,
  opts.think_mode)` at the top of `generate`, and `record_reply(..,
  opts.think_mode)` when the reply span is stored. It is the last few tokens of
  the prompt and is re-prefilled every pass regardless.

So a pass generated with `GenerationOptions { think_mode: ThinkMode::Off,
..self.gen_opts.clone() }` keeps the whole cached prefix, including the
low-effort preamble, and starts the reply after `</think>`. The span recorded
for that reply carries `Off` too, so the next turn's KV common-prefix probe
sees the tokens that were actually evaluated. Nothing on the engine side needs
to change for DS4.

There is precedent in the C reference: `ds4_agent.c`'s compaction pass builds
its prompt on the live transcript and appends
`ds4_chat_append_assistant_prefix(.., DS4_THINK_NONE)` regardless of the
session's think mode, for the same reason: it wants text, not deliberation.

## The change

### 1. A one-pass override on the agent

`Agent` gains `reply_only_next: bool`. It is set at the four sites that see a
reasoning-rung stop and clear it, and consumed by the three generate paths,
which pass an options copy with `think_mode: Off` when it is set and clear it
in the same breath. One pass, never two.

Set sites (all already branch on the stop and call `stub_last_reasoning`):

- `run_turn` (plain REPL) after `is_reasoning_stop(preflight_error)`.
- `worker_turn` (TUI) after `out.error.as_ref().filter(|e| e.looped)`.
- `run_subagent_rounds` after `pass.looped`.
- the fan-out fold after `pass.looped`, on the slot rather than the agent:
  `FanoutSlot` gets its own `reply_only_next`, since the slots run
  independently.

Consume sites, each of which currently reads `&self.gen_opts` or a clone:

- `stream_generation` (plain).
- `worker_generate_kind` (TUI).
- `generate_quiet` / `generate_pass` (sub-agents), through the options the
  caller already clones into the pass context (`live_opts` in
  `generate_quiet`, `opts` in the fan-out).

### 2. The renderer must agree with the prefix

Four places decide whether a pass starts inside `<think>` from `self.think`
rather than from the options actually sent:

- `stream_generation`: `if !matches!(self.think, ThinkMode::Off) && !wants_structured() { stream.begin_in_think(); debugmirror::begin_in_think(); }`
- `worker_generate_kind`: the same test.
- `generate_pass`'s context: `think_off: matches!(self.think, ThinkMode::Off)`.
- the aside path: the same test again.

Each must use the effective per-pass mode, or the renderer will treat the
reply as hidden reasoning and the user sees nothing, which is the bug in a
new costume. Factor the test into one helper, `pass_opens_in_think(&opts,
engine)`, and call it from all four. The `ThinkToolRecovery` inside
`ds4engine::generate` already reads `opts.think_mode`, so it needs nothing.

### 3. Tell the model

The recovery pass has no reasoning, and the model should not go looking for
where it went. Append one sentence to `REPEAT_LOOP_ERROR`, `THINK_BUDGET_ERROR`
and `DRAFT_ERROR`:

> Your next reply has no reasoning step: write the answer, or the tool calls,
> directly.

These are plank-owned strings, not C-parity text, so no fixture changes. The
`stub_last_reasoning` placeholder ("Previous reasoning was cut short for
looping and has been removed. Do not reconstruct it.") stays as it is.

### 4. Other engines

- **Qwen** (`ToolSyntax::Qwen`, same `Ds4Engine`): the template's no-think
  form is `<think></think>`, produced by the same `Ds4ThinkMode::None` mapping
  in `ds4_think`, so it is covered by the DS4 change. Verify with the Qwen
  fixture in `ds4engine` tests rather than assume.
- **Provider engines** (`remote/provider.rs`): Anthropic and OpenAI already
  omit sampling parameters and let the server decide reasoning; the Responses
  path leaves `reasoning.effort` to the default. Ignore the flag there in the
  first cut. If a provider model shows the same drafting pattern, the same
  option can map to the provider's own "no reasoning" knob later; do not
  guess at it now.
- **`EchoEngine`**: ignores `think_mode` already; the turn-loop tests use it.

### 5. What stays the same

- `self.think`, the footer's 🧠 segment, `/think`, and the effort preamble are
  untouched. The user's level is what every ordinary pass runs at; only the
  one pass after a stop differs.
- `MAIN_REPEAT_TRIP_CAP` and `SUBAGENT_REPEAT_TRIP_CAP` keep their values. A
  closed-think pass cannot trip a reasoning rung, so a second trip in a row
  now means the pass *after* the recovery looped, which is the case the cap
  was for.
- Visible output is still unguarded. A recovery pass that loops in its answer
  text runs to `n_predict` or the tool-call guard, exactly as visible text
  does today. If the dumps show that happening, that is a separate rung
  (visible-text cycles are legitimate more often than reasoning ones, per
  the `stream_chunk_must_stop` comment), not a reason to reopen `<think>`.
- KV cost: the reply span for the recovery pass records `Off`. Later passes
  record the session mode again. The ladder and the common-prefix probe
  compare tokens, so a mode change between spans is just a different token
  in the assistant prefix, as it already is when the user runs `/think off`
  mid-session.

## Tests

- `ds4engine`: `build_prompt` with `think_mode: Off` on an engine whose
  `self.think` is `Low` yields tokens that start with the low-effort preamble
  and end with the closed assistant prefix; `record_reply` with the same
  options stores a span whose head is that closed prefix. This is the
  property the whole plan rests on, so it gets its own test even though the
  code already behaves this way.
- `ui` (EchoEngine): after a pass whose preflight error is `REPEAT_LOOP_ERROR`,
  the next pass is generated with `think_mode: Off` and the one after with
  the session mode; the recording `EchoEngine` in the tests already keeps the
  options it was called with. One test per turn loop (plain, TUI worker,
  sub-agent, fan-out slot).
- `ui`: the renderer helper returns `false` for the recovery pass so
  `begin_in_think` is not called; assert through the rendered output that the
  recovery text lands as visible text, not dimmed reasoning.
- Repro: the `## Passes` table shows the recovery pass with `reasoning 0`.
  Add the row shape to `the_passes_table_names_each_stop` so the table reads
  correctly for a pass with no think block.

## Measurement

The benchmark is the request that failed twice: `do a code review` on the
tommaso repository, low think, thinking hidden, temperature 0, on the build
that carries this change plus the draft rung and prompt rule already in main.
Success is a review in the scrollback. The `## Passes` table in the dump, if
one is written, should show at most one reasoning-rung stop followed by a
pass with `reasoning 0` and stop `answer`. Record the outcome in
`LOOP-FINDINGS.md` either way.

## Size

Half a day. One field on `Agent` and one on `FanoutSlot`, four set sites, three
consume sites, one renderer helper replacing four inline tests, one sentence
appended to three error strings, and the tests above. No C-parity impact, no
settings, no new documentation beyond `LOOP-FINDINGS.md` and a CHANGELOG line.

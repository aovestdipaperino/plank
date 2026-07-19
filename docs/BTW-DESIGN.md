# /btw — Side Questions Done Properly

Design document for finishing plank's `/btw` feature, modeled as closely as
possible on OpenClaw's `/btw` (docs.openclaw.ai/tools/btw), the most complete
open-source implementation of the pattern. OpenClaw is vendored as a reference
submodule at `refs/openclaw` (`shallow`/`update=none`: CI skips it; fetch with
`git submodule update --init --checkout refs/openclaw`).

Status: **shipped and un-gated.** §4's mechanics landed on the #12
worker-thread architecture instead of the reverted `433fcb6` event-drain:
the busy UI loop queues `/btw <q>` into `TurnShared` (FIFO cap
20, drop-oldest — OpenClaw's bounded-buffer policy), and the worker answers at
via `Agent::drain_btw` (interrupt flushes the queue; errors log and continue;
`last_ctx_used` restored on every path). A `/btw` submitted mid-generation
**preempts** the running main pass and is answered immediately, then the pass
re-runs (§4.4a); questions queued during tool execution or after the worker's
final boundary answer at the boundary. While an answer streams the screen
splits into a main (60%) / btw (40%) panel that Esc cancels and tears down
(§4.5–4.6). Deviations: queue depth is a dim notice on push rather than a
status-bar field, and plain (non-`/btw`) lines typed while busy queue into the
*main* conversation — that shipped with #12 as the C's `queued_user_drain`,
superseding §4.4's "stays in the buffer".

§4.8 was revised: the C reference contains **no** btw framing
(`btw_user_message` came from Claude Code's pattern, not `ds4_agent.c`), so
the planned parity fixture (step 2) was impossible. Un-gating instead rested
on validating the framing against the real Metal engine, which the
split-panel smoke test confirmed answers side questions correctly from shared
context with tools disabled; `/btw` is now in the unconditional arm of
`slash_command_known` and both slash dispatchers. The `images` feature keeps
gating image pasting only.

## 1. Goal

Let the user ask the model a quick question **while the agent is working**,
answered from the shared conversation context, without interrupting, steering,
or polluting the main task. The defining properties, shared by Claude Code and
OpenClaw:

- **Ephemeral**: the question and answer never enter the session transcript,
  are never persisted, never compacted, and never influence later turns.
- **Read-only**: no tools. The model answers only from what it already knows
  from the conversation context.
- **One-shot**: a single generation pass, no follow-up loop.
- **Non-interfering**: the main turn's transcript, KV cache discipline, and
  tool loop are untouched.

## 2. Prior art

| Agent | Side question? | Mechanics |
|---|---|---|
| Claude Code | `/btw` | Separate API request reusing the prompt cache; no tools; answer in a dismissible overlay; never persisted. |
| OpenClaw | `/btw` (alias `/side`) | Snapshots the session (including in-flight run) as background context; separate one-shot query; answer delivered on a distinct `chat.side_result` event channel; never replayed from history. On forkable runtimes it forks the live thread into an ephemeral child; on plain CLI runtimes it issues a fresh one-shot call with tools and session persistence disabled. |
| Hermes | none | `/busy steer|queue|interrupt` — all feed the main conversation. |
| pi | none | Enter = steer at next tool boundary (skips remaining pending tool calls); Alt+Enter = follow-up queue. Both persisted. |
| opencode | none (declined, issue #17691) | `steer`/`queue` delivery modes; steered messages injected as **plain user messages** at provider-turn boundaries — a system-reminder wrapper was removed because it busted the prompt cache (PR #33039). |

Two lessons transfer directly:

1. **OpenClaw's runtime split legitimizes plank's single-engine answer.** Even
   OpenClaw does not run the side question concurrently on runtimes that
   cannot fork a thread — it degrades to a fresh one-shot call. Plank's local
   Metal engine with one live KV session is exactly that runtime class. A
   second concurrent stream is explicitly **not** part of this design (§8).
2. **opencode's cache lesson** confirms plank's existing choice: the side
   question is appended *after* the full transcript, so the KV prefix stays
   byte-stable and the next real turn's common-prefix sync rolls the side
   question back for free.

## 3. What exists today in plank

- `btw_user_message()` (`src/ui.rs:72`) — the system-reminder framing, copied
  from the reference agent. **Model-facing text; must not change** (see §7.1).
- `btw_plain()` (`src/ui.rs:964`) and `tui_btw()` (`src/ui.rs:1768`) — one
  generation pass over `render_transcript(session) + framed question`, tool
  calls suppressed with a printed notice, `last_ctx_used` saved/restored,
  nothing pushed to the session. The next turn's per-turn KV prefix sync
  re-prefills past the divergence point automatically.
- Both are reachable **only between turns**, and only with
  `--features images` (`slash_command_known`, `src/config.rs:291`).
- Reverted commit `433fcb6` added the missing half: a live line editor inside
  the TUI's mid-generation event drain, a `BtwPrompt { input, queue }` carried
  through `tui_turn`, and `tui_drain_btw()` answering queued questions at
  generation boundaries.

The gap between "today" and "properly" is: mid-turn availability, a distinct
side-result presentation channel, queue policy, interrupt semantics, snapshot
semantics for the in-flight pass, and un-gating.

## 4. Detailed design

### 4.1 Command surface

- `/btw <question>` — the only command. (OpenClaw offers a `/side` alias;
  plank deliberately keeps a single spelling.)
- `/btw` with no argument prints usage, as today.
- Available in both front-ends:
  - **TUI**: idle (immediate) and mid-turn (queued, §4.4).
  - **Plain REPL**: idle only. Cooked stdin cannot deliver mid-turn lines;
    document the limitation in `/help` rather than fake it.
  - **Non-interactive**: not exposed (a headless driver has no "meanwhile").

### 4.2 Execution model: fresh one-shot over the shared prefix

This is OpenClaw's CLI-runtime branch, which maps 1:1 onto plank's engine:

```
prompt = render_transcript(&session, &system)      // shared, KV-cached prefix
       + "[user]\n" + btw_user_message(question)   // divergent suffix
one pass of stream_generation(prompt), tools denied
nothing pushed to session; last_ctx_used restored
```

- **KV discipline**: the prefix is identical to the live session's tokens, so
  `ds4_session_sync` reuses the cached prefix and prefills only the framed
  question — the local analogue of Claude Code's server-side prompt-cache
  reuse. The side answer's tokens are rolled back on the next real
  generation, because that generation's prompt diverges at the question and
  the per-turn common-prefix sync re-prefills from there. No explicit
  checkpointing is needed; **do not** add snapshot/restore FFI for this.
- **Tools denied**: unchanged from today — if the finished stream contains
  calls or a DSML error, print the "tools are disabled during /btw" notice
  instead of dispatching. Nothing is fed back to the model.
- **One-shot**: no loop. `finished().calls` are dropped, not retried.
- **Context accounting**: save `last_ctx_used` before, restore after, so the
  status bar's context estimate never drifts (this is bookkeeping only; the
  KV state needs no restore, per above).

### 4.3 Snapshot semantics for the in-flight turn

OpenClaw snapshots "the current session **including any in-flight main-run
prompt**" — the side question sees what the agent is currently doing. Plank
answers queued questions at generation boundaries (§4.4), where the just
finished pass has already been pushed, so the side question naturally sees:

- every completed generation pass of the current turn (assistant text and
  tool results already in the transcript), and
- **not** the token stream of a pass still in flight.

This is the correct fidelity for a boundary-scheduled answer and requires no
extra machinery. Answering *about* an in-flight pass was said to require either
a transcript snapshot plus a second engine context (§8) or aborting the pass —
both rejected here.

**Deferral lifted (see `docs/BTW-SUSPEND-DESIGN.md`).** The first option turned
out to be available cheaply through the existing FFI: `ds4_session_save_snapshot`
/ `load_snapshot` snapshot and restore a single session's KV, so an in-pass
`/btw` can genuinely freeze the running generation, answer the aside on the same
session, and resume with zero re-prefill — no second context, no abort. This
lives behind the `btw.suspend` config flag (default off; `--btw-suspend`) and
degrades to the boundary queue described above whenever the flag is off or the
engine has no `generate_aside` support. The boundary-scheduled behavior in this
section remains the default and the fallback.

One deliberate deviation from a naive boundary drain: when the boundary is
reached **mid-turn** (more tool passes are still coming), the side question is
answered against the live transcript as-is, including the not-yet-final tool
results. That matches OpenClaw's "background context" framing — the model is
told it is a separate instance sharing context, so partial task state is
expected and harmless.

### 4.4 Mid-turn input: live prompt + FIFO queue

Re-land `433fcb6` as the base, with the policy refinements below. Mechanics
(all in `src/ui.rs`, TUI path):

- `BtwPrompt { input: String, queue: Vec<String> }` owned by `tui_turn` and
  threaded into `tui_generate`'s event drain.
- The mid-generation event drain becomes a minimal line editor: printable
  chars append, Backspace pops, Ctrl-U clears, first Ctrl-C clears a typed
  line (interrupts only on empty input, preserving today's contract), Esc
  always interrupts. The prompt row is drawn only while `input` is non-empty,
  with the cursor at the end.
- **Enter** on a line matching `/btw <q>` pushes `q` onto
  `queue` and clears the editor. Any other submitted line **stays in the
  buffer** with a status-bar hint ("only /btw <question> runs while the agent
  is working") — plank deliberately does not steer (§8).
- The in-progress line survives across generation passes within the turn.
- Status bar shows queue depth: `/btw queued: N`.

Drain points — `drain_btw` runs the queue in FIFO order:

1. **Priority preempt (default):** submitting a `/btw` while the main task is
   generating sets `TurnShared::preempt`; the running main pass stops at its
   next token, and `drain_btw` runs immediately — the side question is
   answered *now*, not at the next natural boundary. See §4.4a.
2. after each generation pass in `worker_turn`, before the next tool dispatch
   (mid-turn boundary — catches questions queued during tool execution);
3. after the turn ends normally;
4. after an interrupt (`out.interrupted`) — the user asked; answer anyway.

Because an answer is itself a generation pass with the same live prompt, the
user can queue further questions while an answer streams; the drain loops
until the queue is empty.

### 4.4a Priority preempt and restart

Queuing a `/btw` only for the *next* boundary is limiting when a single pass
is long (a big essay, a verbose reasoning block). So a `/btw` submitted while
the main task is mid-generation **preempts** it:

- The UI Enter handler, when no side panel is up (`btw.is_none()`), sets
  `TurnShared::preempt` in addition to enqueuing the question.
- `worker_generate`'s interrupt closure honors `preempt` **only for main
  passes** (`is_main`), so a side answer is never interrupted by another
  queued side question — those just join the FIFO and answer after it.
- A preempted pass returns `TurnOutput::preempted`. `worker_turn` then does
  **not** commit its partial assistant text (a pass is committed only on
  completion, so nothing durable is lost), sends `MainRollback` to discard
  the partial output already streamed into the main log (`OutputLog`
  checkpoints its line count on `MainCheckpoint` at each main pass start),
  drains the btw with priority, and `continue`s — re-running the identical
  step from the last committed boundary.
- **Cost:** the interrupted pass is regenerated, so its on-screen partial is
  replaced and the redo is fresh sampling (may differ). The KV prefix is
  reused (the transcript is unchanged), so only the pass's own tokens are
  redone. This is the accepted tradeoff for immediacy — the single-session
  engine cannot pause-and-resume a generation, only stop it.
- Once a side answer is already streaming (the split panel is up), a further
  `/btw` does **not** preempt — the main task is paused already — it just
  joins the queue and is answered after the current one.

**Queue policy** (adopted from OpenClaw's `/queue` defaults, simplified):

- FIFO, **cap 20**. A push beyond the cap drops the *oldest* entry and notes
  it in the log (OpenClaw's `drop old` policy). Silent unbounded growth and
  silent drops are both worse than a visible one-line notice.
- No debounce. OpenClaw debounces because its inbox is fed by chat channels;
  a TUI line editor already debounces by requiring Enter.

### 4.5 Presentation: a split-screen side panel

OpenClaw delivers answers via a dedicated `chat.side_result` event so clients
can render them apart from the conversation. Plank's analogue is a **live
split panel**: while a `/btw` answer streams the output area divides into two
columns — the main conversation keeps **60%** on the left, the side answer
gets **40%** on the right behind a labelled left border (`btw · Esc cancels`).
The input line and status bar span the full width below, unchanged.

Mechanics (implemented on the #12 worker-thread architecture):

- The worker brackets a drain with `UiEvent::BtwBegin` / `BtwEnd`. On
  `BtwBegin` the busy UI loop (`busy_ui_loop`) allocates a fresh side
  `OutputLog` + `OutputView` and routes every subsequent render event
  (`Visible`/`Think`/`Tool`/`Error`/`Dim`/`UserEcho`/`EndLine`) into it
  instead of the main log; on `BtwEnd` it drops the side log and the split
  collapses back to the single-column [`tui::draw`]. The side panel is
  therefore **ephemeral by construction** — its `OutputLog` is never merged
  into the main scrollback, matching the "never part of the conversation"
  invariant (§7.2).
- The panel is drawn by [`tui::draw_btw_split`], which reuses the shared
  `render_output` helper for both columns so scrolling/markdown/highlighting
  behave identically; the side column always follows the newest output.
- **Echo + markers** still stream inside the panel: `/btw <question>` in the
  user-echo style, an opening `[btw]` dim line, and the closing
  `[btw — not part of the conversation]` trailer.
- Multiple queued questions share one panel: `drain_btw` sends `BtwBegin`
  once, loops answering, and sends `BtwEnd` only when the queue empties, so
  the split stays up across a run of asides.
- `/history`, `/context`, session save/load, and `/compact` are unaffected by
  construction, since nothing enters the transcript. A test pins this (§6).

### 4.6 Interrupt semantics

- Esc (or Ctrl-C on an empty editor) **while the split panel is showing**
  cancels the side answer and **tears the panel down** (the worker still
  emits `BtwEnd` on the interrupt path, so the split always collapses). It
  also **flushes the remaining queue** (the user is saying "stop the asides"),
  with a one-line `[btw queue cleared: N]` notice. Because the panel is only
  up while `drain_btw` runs — the main turn is paused at a generation
  boundary — the shared interrupt flag reaches only the side generation;
  `worker_generate` clears it on return, so the main turn resumes cleanly (a
  mid-turn drain continues the tool loop; a post-turn drain returns to the
  prompt). The main turn is never aborted by cancelling a `/btw`.
- An interrupt during the **main pass** still ends the main turn as today;
  queued questions are then answered (drain point 3) unless the same
  interrupt already cleared them — distinguish by where the flag was raised.

### 4.7 Error handling

- A failed side generation (`Err` from the engine) logs `/btw failed: {e}`
  and continues with the next queued question; it must never abort the main
  turn (mid-turn drain errors are logged, not propagated — `?` is wrong
  there, unlike the current `btw_plain` call site which is turn-level).
- `last_ctx_used` restore must run on the error path too (RAII guard or
  explicit restore before `?`).

### 4.8 Un-gating (done)

`/btw` shared the `images` experimental gate "until the model-format
investigation lands." That investigation concerned the *framing text* the
model was trained on, not the queueing mechanics. What actually happened:

1. Landed everything above behind the gate.
2. The planned parity fixture was **impossible**: the C reference has no btw
   framing to match (`btw_user_message` follows Claude Code's pattern, not
   `ds4_agent.c`). Instead the framing was validated live — the split-panel
   smoke test confirmed the model answers side questions from shared context
   with tools disabled and without derailing the main task.
3. `/btw` moved into the unconditional arm of `slash_command_known` and both
   slash dispatchers; the gate comments were deleted. The `images` feature
   keeps gating image pasting only.

## 5. Implementation plan

Ordered, each step independently landable:

1. **Re-land `433fcb6`** (revert of `bd0adbd`), rebased: `BtwPrompt`, live
   line editor in the event drain, `tui_drain_btw`, boundary drains. Before
   re-landing, confirm the revert reason — if it was only the model-format
   gate, nothing in the mechanics needs to change; if a defect was found,
   record it in `FINDINGS.md` and fix it here.
2. **Queue policy**: cap 20 with drop-oldest notice.
3. **Side-channel markers**: `[btw]` opener, interrupt-flushes-queue
   semantics, error-path `last_ctx_used` guard, mid-turn drain errors logged
   not propagated.
4. **Parity fixture** for `btw_user_message`, then **un-gate**.

## 6. Testing

Unit tests (`cargo test --lib`, EchoEngine, no model):

- `btw_leaves_transcript_untouched`: run a turn, run `/btw`, assert transcript
  length and bytes unchanged and `last_ctx_used` restored.
- `btw_denies_tools`: EchoEngine scripted to emit a DSML call; assert the
  notice is printed and no tool dispatch happened.
- `btw_queue_fifo_and_cap`: push 22 questions mid-turn; assert order, the two
  oldest dropped with a notice.
- `btw_mid_turn_boundary`: scripted multi-pass turn; queue a question during
  pass 1; assert the answer generation's prompt contains pass 1's tool result
  and the framed question, and the *next* main pass's prompt does not contain
  the side exchange.
- `btw_interrupt_flushes_queue`: interrupt during a side answer with 2 more
  queued; assert both cleared and the notice logged.
- Editor-key tests for the busy line editor (Ctrl-C clear vs interrupt,
  buffer survival across passes) — pure-logic extraction of the key handler
  is acceptable if driving crossterm events in tests is awkward.

Parity (`tests/c_parity.rs`): `btw_user_message` fixture, regenerated with
`PLANK_REGEN_FIXTURES=1` while the submodule is present.

Manual (TUI, real model): type `/btw` while a long tool loop runs; confirm the
prompt row appears, queue depth shows, answers stream at boundaries, and the
next real turn's prefill cost reflects prefix reuse (watch the status line).

## 7. Constraints and invariants

1. **`btw_user_message` is model-facing text.** Byte-for-byte identical to
   the reference framing; never reflow, never use `\`-continued literals with
   indentation (see `FINDINGS.md`).
2. **Nothing side-channel ever enters `session.transcript`** — not the
   question, not the answer, not the tool-denied notice.
3. **The side prompt is always `full transcript + suffix`** — never reorder,
   never inject mid-transcript, or KV prefix reuse dies.
4. **Two UI paths**: every dispatcher change lands in both `slash` and
   `tui_slash`; every turn-loop change checks whether `run_turn` (plain)
   needs a mirror (here it does not — mid-turn input is TUI-only by design).

## 8. Non-goals

- **A second concurrent generation stream.** Rejected: requires either a
  second engine session (duplicate KV memory, Metal queue contention, FFI
  redesign — `Ds4Engine` is built around one live session) or interleaved
  batched decoding plus a multiplexing renderer. OpenClaw itself falls back
  to sequential one-shot on non-forkable runtimes; boundary scheduling gives
  the same UX at near-zero cost. Revisit only if the engine ever grows a
  cheap session-fork primitive.
- **Steering** (injecting mid-turn input into the *main* conversation, à la
  Hermes/pi/opencode/OpenClaw-`/queue steer`). A separate feature with its
  own design questions (plain-user-message injection for cache friendliness,
  pending-tool-call skipping). The busy line editor built here is the natural
  substrate for it later.
- **Follow-up/collect/interrupt queue modes** (OpenClaw `/queue`): plank's
  mid-turn input accepts only `/btw`; everything else stays in the buffer.
- **Multi-turn side conversations**: one-shot only, like the prior art.
- **`/subagent` changes**: the fork-and-report sidechain (`src/agents.rs`) is
  the complementary tool-using pattern and is out of scope here.

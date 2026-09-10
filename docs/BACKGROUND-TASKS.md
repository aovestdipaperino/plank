# Background Task Notifications

Implementation plan for letting a finished bash job wake the model instead of
making the model poll for it.

Status: **shipped behind `tools.bashNotify` (default off).** Steps 1 through 7
and the prompt sentence, setting, `/jobs` command and desktop notice from step
8 landed together. Deviations from the plan below are recorded in §7.

## 1. The problem

plank's `bash` tool is already asynchronous. `tool_bash` starts a job, waits
up to `refresh_sec`, and if the command is still running returns a
`status=running` observation that ends with the hint "Use bash_status job=N
to get info before refresh time". From there the model is on its own. It
either blocks a full `refresh_sec` inside `bash_status` (the C's
`agent_bash_job_tool_result` with `wait=true`), which burns wall-clock while
the engine sits idle, or it forgets the job, in which case the job runs until
`BashJobs::sweep` happens to time it out on some later bash call, or until
`BashJob::drop` kills it at session end. Nothing ever tells the model "job 3
finished, exit 0" unless the model asks.

Compare Claude Code's `run_in_background`: the harness owns the wait. The
model launches the command, ends its turn, and when the process exits the
harness appends a `<task-notification>` to the conversation and starts a new
turn. The model re-reads the output file and reports. The user gets an idle
prompt in between and can type.

That is the behavior this plan adds. It is a **harness** feature, not a tool
feature: the tool table, the DSML wire format and the observation text stay
byte-identical to the C reference, because `tests/c_parity.rs` pins them and
the model was trained on them.

## 2. What exists today

| Piece | Where | Relevant fact |
|---|---|---|
| Job table | `src/tools/bash.rs`, `BashJobs` on `ToolContext.bash` | Lives across turns. Polled only inside bash-family tool calls (`sweep`, `poll`, `refresh_for`). No reaper thread. Reader threads drain stdout/stderr continuously into a temp file whose path the model already knows as `output_path`. |
| Exit detection | `BashJob::poll` via `Child::try_wait` | Cheap and non-blocking; safe to call from any thread that owns `&mut BashJobs`. |
| Turn loop | `Agent::run_turn` (plain REPL), `Agent::tui_turn_inner` (TUI worker) | A turn ends when a pass produces no tool calls. Nothing can start a turn except a user line. |
| Mid-turn injection | `TurnShared::push_queued` / `Agent::drain_queued` | User lines typed while busy join the transcript as user messages between tool rounds (the C's `queued_user_drain`). This is the one sanctioned way to add a message mid-turn. |
| Idle loops | TUI: 200 ms redraw tick in `run_tui`. Plain REPL: blocking `read_line`. Headless: `run_non_interactive` stdin protocol. | Only the TUI loop can notice an external event without a stdin change. |
| Transcript rule | `docs/KV-CACHE.md`, `tasks.rs` header | Append-only. Never insert mid-transcript or the KV prefix is invalidated. A notification must be a new trailing user message. |
| Prompt text | `sysprompt.rs` line "For long bash jobs, pass refresh_sec and then poll with bash_status" | Byte-pinned to the C. plank's own additions go through the override mechanism in `docs/SYSTEM-PROMPT-OVERRIDES.md`. |
| Desktop notice | `notify.rs` | Already fires on turn completion; reusable for "job finished". |
| Hooks | `hooks.rs` | `PostToolUse` and `Stop` exist; no job-lifecycle events. |

The one thing missing is a path from "child exited" to "a turn starts".

## 3. Design

### 3.1 Terminology

A **background job** is a `BashJob` that was still `running` when the tool
observation that last mentioned it was returned to the model. That is the
only definition needed; there is no new tool argument. The model already
expresses "run this in the background" by passing a short `refresh_sec` and
moving on, which is what the DeepSeek prompt teaches it.

A **task notification** is a user-role transcript message, appended at a
turn boundary, telling the model that one or more background jobs finished.

### 3.2 Notification text

Mirror Claude Code's framing closely enough that the model recognizes the
pattern, but reuse plank's own observation renderer for the payload so the
model sees exactly the `bash_status` output it would have gotten by polling:

```
<system-reminder>
[BACKGROUND JOB NOTIFICATION - NOT USER INPUT]
A bash job you started earlier has finished. This is an automated event, not a
message from the user. Do not treat it as an answer to any pending question.

bash job=3 pid=48120 status=done elapsed_sec=1016.4 timed_out=0
exit_status=0
output_path=/tmp/ds4_agent_output_4711_3_0 (812 bytes, 14 lines)
<tail -20 /tmp/ds4_agent_output_4711_3_0>
...
</tail>
</system-reminder>
```

The block between the two sentences and `</system-reminder>` is
`BashJob::observation(true)` verbatim, so the model's mental model of job
output does not change and no new parity fixture is needed for it. Several
jobs finishing together produce one message with one observation each.

The message is a plain `Message::user`, exactly like a queued line or the
`AGENTS.md` context injection, so it persists, compacts and replays like any
other user turn.

### 3.3 Detection: a job watcher, not a reaper

`BashJobs` gets a new method:

```rust
/// Polls every job and returns the ids of jobs that finished since the
/// model last observed them as running.
pub fn take_finished_unannounced(&mut self) -> Vec<i64>
```

backed by a new `announced: bool` field on `BashJob`. `observation()` sets
`announced = true` when it renders a `status=done` block, so a job the model
polled itself is never announced twice. `take_finished_unannounced` sets it
too. This keeps a strict invariant: **each job produces at most one
notification, and none if the model saw the exit on its own.**

Polling still uses `Child::try_wait`, so there is no reaper thread and no
change to how `Drop` kills the process group.

### 3.4 Delivery: three front-ends, one policy

Delivery happens at two kinds of boundary:

1. **Mid-turn** (a job finishes while the model is still working). After
   each tool round, right where `drain_queued` runs, call
   `take_finished_unannounced` and, if non-empty, push the notification
   message before the next generation pass. This is free: the worker already
   owns `Agent` there, and the transcript is at a legal append point.

2. **Idle** (a job finishes after the turn ended). This is the new part and
   is where the front-ends differ.

| Front-end | Idle wait today | Change |
|---|---|---|
| TUI | `run_tui` redraws on a 200 ms tick | On each tick, if no turn is running and no modal pane is open, call `take_finished_unannounced`. If non-empty, build the notification and start a turn with it exactly as if the user had submitted that text, but rendered in the log as a dim system line instead of a user prompt. |
| Plain REPL | `read_line` blocks the thread | Move stdin reading onto a helper thread that sends lines over an mpsc channel. The REPL loop then does `recv_timeout(250 ms)`, checking jobs on timeout. Same start-a-turn path as the TUI. |
| Headless (`--non-interactive`) | stdin protocol | Do **not** auto-start a turn: the driver owns the loop. Instead emit a `{"type":"job_finished", ...}` event on stdout and let the driver decide. The notification text is still appended to the transcript so a following `run_turn` sees it. |

Both interactive paths funnel into one new `Agent` entry point:

```rust
/// Starts a turn driven by a background-job notification rather than a user
/// line. Returns without generating when no unannounced job exists.
pub fn wake_for_jobs(&mut self) -> Result<bool, String>
```

so the TUI and REPL mirror rule from `CLAUDE.md` is satisfied by construction.

### 3.5 Guards against loops and interruptions

- **Draft protection.** In the TUI, if the editor holds unsent text, the
  notification waits. Starting a generation while the user is mid-sentence is
  the failure mode people hate most about auto-wakes. The check is
  `editor.is_empty()` on the tick; the job stays pending and is announced on
  the next empty tick or at the next turn boundary.
- **One wake per job.** Enforced by `announced` (§3.3). A model that reacts
  to a notification by starting another job is fine; that job gets its own
  single notification later.
- **Wake budget.** A setting `bash.notifyMaxPerIdle` (default 3) caps how many
  auto-started turns can run back-to-back with no user input in between.
  Beyond it, notifications are appended to the transcript but no turn is
  started until the user types. This is the circuit breaker for a model that
  spawns a job every time it is woken; the loop guards in
  `docs/LOOP-FINDINGS.md` do not cover it because each turn is legitimately
  triggered.
- **Interrupt semantics.** Ctrl-C / Esc during an auto-started turn behaves
  exactly like during a user turn. It does not cancel other pending
  notifications.
- **Sidechains.** `in_sidechain()` passes never drain notifications; they
  belong to the main transcript only.

### 3.6 Telling the model

The C prompt line "poll with bash_status or stop with bash_stop" stays.
Through the override mechanism plank appends one sentence when the feature is
on:

> When a bash job is still running you may end your turn; plank will notify
> you automatically when it finishes, so do not poll or sleep to wait for it.

Without this line the model keeps polling and the feature is silent. With it,
`google_search`-style loops where the model calls `bash_status` ten times in a
row should disappear. Measure this (§5) rather than assume it.

### 3.7 User-facing surface

- Setting `bash.notify` (`true` by default once the smoke test passes; ship as
  `false` in the first beta).
- Footer: while any job is running and unannounced, the stats segment shows
  `⧗ 2 jobs`. This reuses the segment naming in the status bar rather than
  adding a new one.
- Desktop notification through `notify.rs` when a job finishes while plank is
  unfocused, gated by the existing notify settings.
- `/jobs` slash command (both dispatchers) listing running and finished jobs
  with id, pid, elapsed and exit status. Static text only, no pane, so it
  works on the plain path too.
- Hook event `JobFinished` with the same JSON the headless protocol emits, so
  users can play a sound or post to Slack without touching plank.

## 4. Implementation steps

Each step is independently testable with `cargo test --lib` and the
`EchoEngine`. No step needs a model.

1. **Track announcement state.** Add `announced` to `BashJob`, set it in
   `observation()` on `status=done`, add `take_finished_unannounced`. Unit
   tests: a job observed as done is never returned; a job that exits between
   observations is returned once; a timed-out job is returned once with
   `timed_out=1`.
2. **Render the notification.** New `bash::render_notification(&mut BashJobs,
   ids) -> String` producing §3.2. Snapshot test against a fixture in
   `tests/fixtures/`, because the exact text is what the model will learn to
   recognize and it should not drift by accident.
3. **Mid-turn delivery.** In `drain_queued` (TUI worker) and its `run_turn`
   mirror, append the notification after queued user lines. Test with
   `EchoEngine`: a tool round that leaves a job running, followed by the job
   exiting, yields a transcript with the notification before the next
   assistant message.
4. **`Agent::wake_for_jobs`.** Appends the notification and runs one turn.
   Test: returns `Ok(false)` with an empty table, `Ok(true)` and a longer
   transcript otherwise. Respects the wake budget.
5. **TUI idle hook.** Check on the redraw tick with the draft and modal
   guards. Render the trigger as a dim log line, not a user prompt echo.
6. **REPL stdin thread.** Replace the blocking `read_line` with a channel and
   `recv_timeout`. This is the riskiest step because Ctrl-C handling and EOF
   detection both live around that call; keep the existing interrupt tests
   green before adding the job check.
7. **Headless event.** Emit `job_finished` on stdout; document the shape in
   the non-interactive protocol section of `docs/ARCHITECTURE.md`.
8. **Prompt override, setting, footer, `/jobs`, hook, desktop notice.** Each
   is small; the order does not matter. Add the setting to the settings
   schema and `configform.rs`.
9. **Docs.** Add the "one rule that must never break" for this feature to
   `CLAUDE.md`'s architecture list: *a notification is appended only at a turn
   boundary and only once per job; `announced` is the sole source of truth*.
   Record any model-behavior surprises in `docs/LOOP-FINDINGS.md`.

## 5. Verification with the real engine

`EchoEngine` proves the plumbing, not the behavior. Before flipping the
default on:

- Run `bash` with `sleep 20; echo done` and `refresh_sec=2`. Expected: the
  model ends its turn after the `status=running` observation, the prompt
  becomes idle, and roughly 18 s later a turn starts on its own and the model
  reports "done" with exit 0.
- Same, but type a partial line during the wait. Expected: no wake until the
  line is sent or cleared.
- Start three jobs of different lengths. Expected: three notifications, each
  once, in exit order, and the footer count drops to zero.
- Count `bash_status` calls per session on the smoke-test corpus in
  `docs/SMOKE-TEST.md` before and after the prompt override. The feature
  earns its default-on only if that count falls.

## 6. Out of scope

- Persisting jobs across a plank restart. `BashJob::drop` kills the process
  group by design so nothing outlives plank; a restarted plank has no job to
  announce.
- A `run_in_background` tool argument. It would change the pinned tool schema
  and the model does not need it; short `refresh_sec` already expresses the
  intent.
- Notifying about MCP tool calls or sub-agents. Sub-agents already return at
  their own boundary; MCP has no long-running call model.

## 7. What shipped, and where it differs from the plan

- **No `announced` flag.** The job table itself is the source of truth:
  `job_tool_result` already removes a job the moment the model observes it as
  `status=done`, so any job still in the table and not running finished
  unseen. `BashJobs::take_finished` polls, removes and returns those. This is
  simpler than §3.3 and has the same invariant: one notification per job,
  none if the model saw the exit itself.
- **Headless protocol.** The event is a stderr marker in the existing
  `+DWARFSTAR_*` family, `+DWARFSTAR_JOBS_FINISHED <n>`, not a JSON line on
  stdout, and plank does run the turn itself after appending the
  notification. The driver still sees the marker before any output from that
  turn, so it can tell an auto-turn from a reply to its own prompt.
- **REPL stdin thread.** `run_repl_plain_local` reads stdin on a detached
  helper thread and the loop uses `recv_timeout(250 ms)`. EOF and read errors
  travel over the same channel. Ctrl-C behavior is unchanged because the
  interrupt flag was never tied to `read_line`.
- **Not shipped:** the footer job count (§3.7), the `JobFinished` hook event,
  and the per-idle wake budget (§3.5). The wake budget was dropped because a
  notification is only ever produced by a job the model itself started; a
  model that reacts to each wake by starting a new job is still bounded by
  the tool-call loop guards on the turn that starts it. Revisit if a real
  session shows otherwise.
- **Setting is off by default.** Turning it on also appends one sentence to
  the shell rules, so it churns the `fp1` system-prompt fingerprint like
  `tools.recall` does. The §5 smoke test has not been run against the real
  engine yet; flip the default only after it passes.

# Loop findings

Everything plank has learned about the model repeating itself: reasoning that
cycles, tool stanzas re-issued after a refusal, and the guards that stop each.
This is the one place for past, current and future loop findings; `FINDINGS.md`
keeps the parity and tooling gotchas and points here for loops. Every auto-saved
dump lives in `~/.plank/repro/repro-loop-<secs>.md` (see `src/repro.rs`), so an
entry should name its dump.

The guards, for orientation:

- **Reasoning repeat guard** — `insights::RepeatGuard`, fed with `<think>`
  text only through `ui::stream_chunk_must_stop`. Exact byte cycles, period at
  least 12 bytes, stop at 4 cycles (`REPEAT_CYCLES`), footer marker at 2, checked
  every window/16 bytes over an 8 KiB window (`REPEAT_LOOP_WINDOW`). Stops the
  pass through the preflight-error channel with `REPEAT_LOOP_ERROR` and writes
  a `repro-loop` dump.
- **Tool-call loop guard** — `guard::LoopGuard`: advisory on the 3rd identical
  call, refusal from the 6th, turn ended after three stanzas in a row refused
  in full (`tripped`). Also detects a repeated *sequence* of calls
  (`repeated_period`).
- **Sub-agent trip cap** — `SUBAGENT_REPEAT_TRIP_CAP = 2`: a sidechain stopped
  twice running is pushed to its report; a third loop fails it.

## Timeline: how the guards got here

Every commit that touched loop detection, from `git log`. Each rung was cut
by a dump that showed the previous rung was not enough, so the order is the
argument for the design.

| date | commit | what it added | driven by |
|---|---|---|---|
| 2026-08-26 | `3bab717` | `guard::LoopGuard`: advisory on the Nth identical tool call (name + SHA-1 of normalized args) in a bounded window; `tools.repeatAdvisory` setting; `tools.callTimeoutSec` wall-clock deadline on a single tool dispatch (off by default) | dsh.md milestone M1 |
| 2026-08-31 | `7ddb566` | `insights::RepeatGuard`: exact-cycle detector on the streamed tail (1 KiB window, checked every 64 bytes, period at least 12 bytes, 4 cycles), wired into the insights report sections only | recommendation sections repeating one clause until the section budget ran out |
| 2026-08-31 | `bbebf6a` | context-pressure precondition on the opportunistic microcompact gate | `repro-1788613069`: eight identical tool calls for two hours after every turn rewrote just-read files into stubs (see "Other findings", below) |
| 2026-09-05 | `1bb07c0` | `RepeatGuard` on every generation pass (REPL, TUI, quiet sub-agent) through `ui::stream_chunk_must_stop`, `<think>` text only, 8 KiB window, stop via `fail_preflight` and `REPEAT_LOOP_ERROR`; "Working style" prompt rules; `--think-low` as default | the two-hour session that was one 149 K-character think block cycling three paragraphs 263 times |
| 2026-09-05 | `59f0e8e`, `ef163d6` | `LoopGuard` hard block (`Nudge::Block` from the 6th identical call, refused calls get a `Tool error:` instead of running); `repeated_period` sequence detection; `[timestamp]` markers in repro dumps | `repro-1788619030`: 52 iterations of a stanza the model kept re-issuing through every advisory |
| 2026-09-05 | `dbfca93` | warn level at 2 cycles (`REPEAT_WARN_CYCLES`, `RepeatGuard::repeating`) rendered as `🔁 looping` in the TUI footer; automatic `repro-loop-<secs>.md` dump with sub-agent sidecars; per-round tool activity line; prompt asks the model to narrate | a stopped pass left no evidence unless the user typed `/repro` |
| 2026-09-06 | `6e3bc2b` | refused calls kept out of the guard window with their own monotonic counter; three stanzas in a row refused in full end the turn (`LoopGuard::tripped`, `LOOP_TRIPPED_NOTICE`) | `repro-1788676865`: the refused count plateaued at 11 and the model re-emitted the identical stanza for six minutes |
| 2026-09-06 | `966b55d` | `[status]` reminder appended when a pass emits tool calls with no visible text | long turns whose only output was tool summary lines |
| 2026-09-06 | `5d5508a` | `SUBAGENT_REPEAT_TRIP_CAP = 2` (second stop pushes the final-round reminder, third fails the sub-agent with `REPEAT_TRIPS_NOTICE`); guard stops as red `guard:` lines on the main window; live status from the quiet pass; interrupted pass keeps its partial text | `repro-1788690439`: a fan-out sub-agent looped ten minutes, was stopped, then generated thirteen more with nothing moving |
| 2026-09-07 | `aaf0f3d` | `MAIN_REPEAT_TRIP_CAP = 2` on both main-turn paths (`MAIN_REPEAT_TRIPS_NOTICE`); `Agent::repro_dir` so test dumps stay out of `~/.plank/repro`; this document | `repro-loop-1788708943`/`-1788709421`: the main turn looped, stopped, looped again, and the user quit |

Two patterns run through the table. First, every detector started advisory
or per-pass and had to grow a rung that *ends the turn*: at temperature 0 a
guard that only refuses or only feeds back an error leaves the prompt
effectively unchanged, so the next pass is the same pass. Second, each new
guard shipped with its own evidence channel (timestamps, the auto-dump, the
red lines, the footer marker), because the previous stall had been
undiagnosable from what was on disk.

Not a guard, but worth knowing: `tools.callTimeoutSec` from `3bab717` is a
per-dispatch deadline and is off by default; a hung tool is a different
failure from a looping model and is not covered by anything above.

## How to read a loop dump

The last `[assistant]` block before `</think>` is the stopped pass. Its tail is
what the guard saw; to measure how late the guard fired, find the smallest
period `p` such that the tail ends in `k` copies of the last `p` bytes, then
`k*p` is the repeated span and everything before it is pre-loop reasoning.
Measured on the dumps to date (2026-09-07):

| dump | session | pass bytes | exact cycle | repeated span | pre-loop |
|---|---|---|---|---|---|
| `repro-loop-1788675204` | cuddly-fischer | 18 095 | 787 B × 4 | 3 148 | 14.9 K |
| `repro-loop-1788708943` | bumbling-einstein | 3 700 | 123 B × 5 | 615 | 3.1 K |
| `repro-loop-1788709421` | bumbling-einstein | 20 676 | 714 B × 4 (8 lines) | 2 856 | 17.8 K |
| `repro-loop-1788693586/649` | *(test artifact)* | 18 091 | 45 B × 400 | 18 000 | 0 |

Every real dump stopped at the designed four cycles: a stop costs about four
periods, 0.6 to 3 KB, some 150 to 800 tokens, well under a minute at 25 t/s.
The latency of the guard is no longer the problem. The paragraphs below are.

## A stopped pass is regenerated verbatim: the main turn has no trip cap

`repro-loop-1788708943` and `repro-loop-1788709421` are the same session,
eight minutes apart. The guard stopped a pass, `REPEAT_LOOP_ERROR` went back as
a tool result, and the next pass opened with "Need stop repeating. We have
enough. Need implement." and then produced 20 KB of reasoning that ended in the
same shape of loop (a 29-line stutter of "Need maybe emit unresolved refs for
…", converging into an exact 8-line cycle). The session file shows a third
pass starting at 17:45 with "We need stop repeating. Let's implement now." and
ending after 697 bytes: the user quit. Nothing was edited.

This is the main-turn instance of the lesson the sub-agent path already
learned (`SUBAGENT_REPEAT_TRIP_CAP`, below): at temperature 0 the pass after
the guard's error is the same prompt plus one message, and a one-line "do not
resume" is not a material change. The main turn counts nothing across passes
(`repeat_trip_text(1)` is always reported with count 1), so it will loop, stop,
loop, stop until the user gives up. Fixed 2026-09-07: `MAIN_REPEAT_TRIP_CAP = 2` — the second stop in a row
ends the turn with a red `guard: turn stopped: the model's reasoning looped
twice in a row…` line (`MAIN_REPEAT_TRIPS_NOTICE`), on both the plain and the
TUI path, leaving the guard's tool error as the last message so the next
prompt sees why. The user, not a retry, is what changes the prompt.

## The loop is preceded by a stutter the exact-match guard cannot see

Both bumbling-einstein passes and the original two-hour stall share a shape:
before the byte-exact cycle forms, the model emits a long run of lines that
share a prefix and differ in the tail ("Need maybe update `src/extraction/
mod.rs` `LanguageRegistry::new`…", 25 consecutive lines in
`repro-loop-1788708943`, 29 in `-1788709421`). The exact guard fires only once
the variation dies out. A fuzzy detector was tried against the dumps: "a
normalized line of at least 40 bytes seen three times within the 8 KiB
window". It would have saved 0.5 to 2.6 KB per real stop and fired on 52 of 664
ordinary assistant passes in the other dumps (repeated DSML parameter lines,
repeated read-plan lines), so it is not a candidate as stated. Any drift
detector has to be think-only, line-normalized, and require several *distinct*
lines each recurring, and it still needs a false-positive pass over the
non-loop dumps before it ships. The stutter itself is a `--think-low`
artefact only in part: it appears with low thinking too.

## Test dumps leak into the real `~/.plank/repro`

`repro-loop-1788693586` and `-649` (13:19 and 13:20 on 2026-09-06, identical
apart from timestamps, `[user] do the task`, 400 copies of one 45-byte line)
are not sessions. They are `a_looping_main_pass_is_flagged_in_red_without_
naming_a_sub_agent` (`src/ui.rs`): `worker_turn` calls `loop_repro_line`,
which saves through `repro::save_loop`, whose directory is `$HOME/.plank/repro`
(`repro::repro_dir`). `test_agent` gives the agent a scratch `SessionStore` but
nothing redirects the repro root, so every run of that test writes a fake loop
dump next to the real ones. Two consequences: the folder cannot be trusted as
a record of real stalls without checking the `session:` header, and the 400×
figure is meaningless for guard latency (the scripted engine delivers the whole
reply as one chunk, so the guard sees it all at once). Fixed 2026-09-07: the agent resolves
its repro folder once at construction (`Agent::repro_dir`) and every dump goes
through `repro::save_in` with it; `test_agent` points it at a temp directory
(`test_repro_dir`), never `set_var("HOME")` (see the spill-test entry in
`FINDINGS.md`). The two stray files can be deleted.

## Other findings that end in a loop

Recorded under their own headings in `FINDINGS.md`, because the loop was the
symptom and the cause lay elsewhere:

- **Microcompact cadence (M5)** — the opportunistic gate rewrote the model's
  just-read files into stubs after every turn, so at temperature 0 the
  identical context produced the identical eight tool calls for two hours
  (`repro-1788613069`: 214 reads, 0 edits). The tool-call guard never fired
  because the cycle was eight calls and its window was ten. Fixed with the
  context-pressure precondition.
- **Rejecting DSML inside `<think>` on the opening marker** — the model quoted
  the marker in its reasoning, emitted a correct call afterwards, was told it
  had done something it had not, and rewrote correct markup in a loop
  (`repro-1785754509`). Fixed by evaluating the prohibition at the stop token.

## An hour-long turn was one thinking loop, not a hundred tool calls

A session that took plank two hours on a request Claude Code finished in
minutes had 46 tool calls and no edit to the project. The time was one
assistant message: 149K characters of `<think>` in which three paragraphs
("Need maybe add `MF_AUTO_DISMISS` to...") repeated 263 times, until the 50K
token `n_predict` cap stopped it. At 25 tokens per second that is about 50
minutes of silence, and the reply after the cap was empty.

`insights::RepeatGuard` already detected exactly this, but was wired only into
the insights sections. It now runs on every generation site (`stream_generation`,
the quiet sub-agent pass, `worker_generate_kind`) through
`ui::stream_chunk_must_stop`, with three rules that are easy to break:

- **Watch reasoning only.** The guard is fed while `stream.in_think()`.
  Visible output and tool arguments repeat legitimately — a `write` of a table
  with identical rows is four cycles of a 12-byte period — and would trip it.
- **The window must hold four cycles of a paragraph loop.** The observed
  period was about 600 bytes; the insights default of 1 KiB can never see it.
  The turn loop uses `RepeatGuard::with_window(8192)`, and the check interval
  scales with the window so the scan stays bounded on fast provider streams.
- **Stop through the preflight-error channel.** `StreamRenderer::fail_preflight`
  records the model-facing text, so the existing "engine interrupted but it is
  a tool error, not a user abort" path feeds it back as `Tool error:` and the
  turn continues. A new stop reason would have needed its own plumbing at all
  three consumers.

The guard also has a *warning* level: `RepeatGuard::repeating()` turns true at
two identical cycles (`REPEAT_WARN_CYCLES`) and stays true for the pass, while
the stop still waits for four. The TUI status snapshot copies it into
`Status::looping`, which the footer renders as `🔁 looping` after the ctx
gauge. Two consequences worth remembering: the flag is sticky on purpose, since
the tail drifts in and out of alignment between checks and a flickering marker
reads as a bug; and the plain REPL never shows it, because its status bar is
cleared the moment output starts streaming, so there is no generating footer to
carry it.


## A refused tool call is not a stopped loop

`repro-1788676865`: the loop guard blocked the model's three-read stanza at
the sixth repeat, as designed, and the model then re-emitted the identical
stanza every pass for six more minutes until the user pressed Ctrl-C. Two
things conspired. At temperature 0 the pass is a pure function of the prompt,
and a refusal changes the prompt by one digit, so the model has no reason to
behave differently. And the digit stopped changing: refused calls were pushed
into the 32-call window, so once it was full every new call aged out one
identical old call, the per-signature count went down one and up one, and
"11 times" was reported forever, making the prompt *exactly* identical across
passes. The fix keeps refused calls out of the window with their own
monotonic counter, and adds the rung the guard was missing: three stanzas in a
row refused in full end the turn (`LoopGuard::tripped`), after the automatic
loop dump. The general lesson: a guard that only refuses an action leaves a
deterministic model exactly where it was; something has to change the
prompt materially or stop the turn.


## A sub-agent that publishes no status looks hung

`repro-1788690439`: a `fanout` sub-agent looped in its reasoning for ten
minutes until the repeat guard stopped it, then generated for thirteen more
with nothing on screen moving but the roster row's clock, and the user pressed
Esc. Three separate gaps, each of which had to be closed at its own layer:

- **Only the main pass published `Status`.** The roster row's live token count
  comes from `UiEvent::Status` snapshots (`SubPane::note_status`), but those
  were built only in `worker_turn`'s engine callback; the quiet sub-agent pass
  (`generate_pass`) handled `EngineEvent::Text` alone. A row credited only by
  the per-pass `SubTokens` tally froze for the whole pass, exactly the failure
  the "record_usage fires at pass completion" entry in `FINDINGS.md` describes — that
  fix wired the row to snapshots without making the sub-agent pass emit any.
  Both passes now build their snapshots through one `LiveStatus`, so they
  cannot drift apart again; the fan-out leaves it off (several passes, no
  honest single row).
- **The guard's error was fed back and nothing else changed.** Same lesson as
  the refused-tool-call entry above: at temperature 0 the pass after
  `REPEAT_LOOP_ERROR` is the same prompt plus one message, and the sub-agent
  loop let that repeat for up to 40 rounds. Two stops in a row now push the
  final-round reminder (`SUBAGENT_REPEAT_TRIP_CAP`), which changes the prompt
  materially and turns the next pass into the report; a third loop fails the
  sub-agent with `REPEAT_TRIPS_NOTICE` rather than retrying.
- **The interrupted pass was discarded.** `generate_pass` returned a bare
  `Err("interrupted")`, so the sidechain dump ended on the guard's tool result
  and could not show what the model was doing for those thirteen minutes. The
  abort now carries the partial text (`QuietAbort`), pushed into the sidechain
  before the fork end truncates it out of the parent transcript. The main loop
  had always kept its partial text; the sidechain path was written separately
  and missed it.

Guard stops are also red `UiEvent::Error` lines on the main window now
(`Agent::report_guard`), naming the sub-agent (`ToolContext::subagent_label`,
set around the delegated run so nesting restores the outer name). The sub-agent
pane already showed the tool error, but nobody watching the parent could see
it, which is the difference between "stuck" and "looping, being handled".

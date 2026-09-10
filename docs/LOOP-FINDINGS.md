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
  a `repro-loop` dump. A cycle too long for four copies to fit the window is
  *latched* at the warn rung and then tracked forward copy by copy
  (`RepeatGuard::extend_latched`), so the period a stop can see is not capped
  by the window — see "The stop rung saw a quarter of the window", below.
- **Tool-call loop guard** — `guard::LoopGuard`: advisory on the 3rd identical
  call, refusal from the 6th, turn ended after three stanzas in a row refused
  in full (`tripped`). Also detects a repeated *sequence* of calls
  (`repeated_period`).
- **Think budget** — `repeat_think_budget(ctx_size)`: `ctx_size / 10` bytes,
  floored at 16 KiB (`REPEAT_THINK_BUDGET_FLOOR`; ~102 KB on the 1M-token
  window) of reasoning
  per pass, whatever the tail looks like. Recognises nothing, so unlike the cycle rungs
  it has no `2p` latency floor and no period ceiling, and it is the only rung
  that catches a drifting loop. Stops through the same preflight channel with
  `THINK_BUDGET_ERROR`, and counts towards `MAIN_REPEAT_TRIP_CAP`.
- **Draft rung** — `RepeatGuard::drafting`: past `DRAFT_MIN_BYTES` (8 KiB) of
  reasoning, `DRAFT_HEADINGS` (10) numbered deliverable headings
  (`**Bug 3:**`, `1. **Title**`, `### 4.`) or `DRAFT_FENCED_BYTES` (4 KiB)
  inside code fences means the reasoning is writing the answer, not deciding
  it. Checked after the cycle rungs and before the budget, stops through the
  same preflight channel with `DRAFT_ERROR` ("write this as your answer, not
  in reasoning"), counts towards `MAIN_REPEAT_TRIP_CAP`, and runs only on
  guards that have a budget, i.e. the turn guards — see "The review that was
  written in the wrong place", below.
- **No-progress budget** — `NO_PROGRESS_BYTE_BUDGET` = 32 KiB generated in one
  turn with no `PROGRESS_TOOLS` call (`write`, `edit`, `bash`, `bash_stop`).
  Turn-scale, so it is the only rung that sees a turn whose every pass is
  individually reasonable and which still changes nothing.
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
| 2026-09-07 | *(this change)* | `RepeatGuard` latches the period the warn rung matched and counts further copies forward (`Latched`, `extend_latched`, `cycle_period` replacing `has_cycles`), removing the window's cap on the period a stop can see | `repro-1788788326`: a 2434-byte cycle ran 17 times, warned in the footer the whole way, and could never be stopped |
| 2026-09-07 | *(this change)* | `REPEAT_THINK_BUDGET = 16 KiB` per-pass reasoning cap (`RepeatGuard::with_think_budget`, `THINK_BUDGET_ERROR`, counted towards `MAIN_REPEAT_TRIP_CAP`); `NO_PROGRESS_BYTE_BUDGET = 32 KiB` per-turn cap on output with no `PROGRESS_TOOLS` call | `repro-1788796284`: a 9042-byte cycle 20 times over in a sub-agent, longer than the whole window so no rung could see it, and a parent turn that looped no text at all yet edited nothing in fifty minutes |
| 2026-09-07 | `aaf0f3d` | `MAIN_REPEAT_TRIP_CAP = 2` on both main-turn paths (`MAIN_REPEAT_TRIPS_NOTICE`); `Agent::repro_dir` so test dumps stay out of `~/.plank/repro`; this document | `repro-loop-1788708943`/`-1788709421`: the main turn looped, stopped, looped again, and the user quit |
| 2026-09-08 | *(this change)* | no-progress budget resets only after a successful direct `write` or `edit`, not an attempted `edit` or arbitrary `bash` call | `repro-loop-1788833715`: 5h7m of failed edits, builds, and repeated reads kept resetting the budget |
| 2026-09-10 | *(this change)* | think budget sized from the context window: `repeat_think_budget` = `ctx_size / 10` bytes, floored at the old 16 KiB (`REPEAT_THINK_BUDGET_FLOOR`) | `repro-loop-1789051332` … `-1789053127`: seven budget stops in three sessions on one feature request, none a cycle — see "The think budget fired on reasoning that was not looping", below |
| 2026-09-10 | *(this change)* | draft rung (`RepeatGuard::drafting`, `DRAFT_ERROR`): numbered deliverable headings or fenced code accumulating inside `<think>` past 8 KiB stop the pass with "write this as your answer, not in reasoning"; and a `WORKING_STYLE` rule, "Write findings as you find them", so list-shaped answers are emitted item by item after `</think>` | `repro-loop-1789060243` and the seven 2026-09-10 dumps: deliverables drafted in reasoning, never emitted |
| 2026-09-10 | *(no code change)* | counter-case to the raised budget recorded: a 30 KB review drafted inside `<think>` under the ~102 KB budget, interrupted by the user at 12m42s; the `resume` pass redrafted and fell into a 5-line cycle the exact-cycle rung caught | `repro-loop-1789060243`: `do a code review`, 18 minutes, no visible output — see "The review that was written in the wrong place", below |
| 2026-09-08 | *(this change)* | `tools.loopGuards` and `/loopguard` (alias `/lg`): one switch over every rung — `LoopGuard::observe`/`tripped`, the gated `RepeatGuard` (cycles and think budget), the no-progress budget. Read through `guard::guards_enabled()` at each check, never captured at turn start, so the switch lands on a generation already streaming; `🔁` in the footer while armed, and the tripped marker moved to `♻ looping` | diagnosing the guards themselves, where every rung fires before the behaviour under study can be observed |

Two patterns run through the table. First, every detector started advisory
or per-pass and had to grow a rung that *ends the turn*: at temperature 0 a
guard that only refuses or only feeds back an error leaves the prompt
effectively unchanged, so the next pass is the same pass. Second, each new
guard shipped with its own evidence channel (timestamps, the auto-dump, the
red lines, the footer marker), because the previous stall had been
undiagnosable from what was on disk.

One consequence of the switch worth stating: a silenced guard keeps *feeding*.
`RepeatGuard::feed` still eats every chunk while `gated` and off, so the tail
and the byte count stay honest and `/loopguard on` mid-pass answers from real
history instead of a fresh allowance. A switch that stopped the bookkeeping
would hand a looping model a clean slate every time it was flicked.

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
| `repro-1788796284.sub-1` | cranky-watt *(sub-agent)* | 189 744 | 9 042 B × 20 | 180 840 | 8.9 K |

Every real dump stopped at the designed four cycles: a stop costs about four
periods, 0.6 to 3 KB, some 150 to 800 tokens, well under a minute at 25 t/s.
The latency of the guard is no longer the problem. The paragraphs below are.

## The stop rung saw a quarter of the window, the warn rung a half

`repro-1788788326` (loopy-napoleon, saved by hand at 15:38 on 2026-09-07 —
`note: (none)`, because nothing auto-saved it) is the cleanest loop on record
and the one the guard could not stop. The final `[assistant]` pass is 46 692
bytes of think text with no DSML stanza at all:

| | |
|---|---|
| pre-loop reasoning | 3 466 B |
| cycle | **2 434 B, byte-exact** |
| cycles | **17** (18 occurrences, every delta exactly 2434, no drift) |
| repeated span | 41 378 B, 89% of the pass |

It also skips the fuzzy stutter phase described in the section below: seven
paragraphs — "Wait, maybe the script's `out.append(line)`…", "Let's not debug;
the script is flawed…", "Given the repeated failures, I think we should stop
and report…", a fenced excerpt of the script, "Actually, the user explicitly
chose 'first option'…" — locked byte-exact from the first copy.

The user reported `🔁 looping` in the status bar, and that is the whole
diagnosis. `has_cycles` searched `REPEAT_MIN_PERIOD..=len / cycles`, so the
two rungs had *different period ceilings* off the same window:

| rung | cycles | max period at 8 KiB | 2434 B? |
|---|---|---|---|
| `repeating` → footer marker | `REPEAT_WARN_CYCLES` = 2 | 4096 | ✅ |
| `feed` → `REPEAT_LOOP_ERROR` | `REPEAT_CYCLES` = 4 | 2048 | ❌ never |

Any period in `(window/4, window/2]` lands in that gap, and the gap is not a
latency problem that more cycles eventually close: the stop condition is
*unreachable* there. The footer latches, the model runs to `n_predict`, and
the only thing that ends the turn is the user. Note how this inverts the
reassurance in "How to read a loop dump": every dump in that table stopped at
the designed four cycles because every one of them had a period under 2048.
The table was measuring the loops the guard could see.

Fixed 2026-09-07 by making the stop rung independent of the window instead of
widening it. Widening is the obvious move and it is the wrong one: at a 32 KiB
window a stop still costs four full periods (~10 KB of reasoning here) and the
ceiling merely moves, so the next loop with a longer period repeats this
finding. Instead the block the warn rung matched is kept (`Latched { block,
end, cycles }`) and each further copy is verified as it arrives
(`extend_latched`), which needs two periods of window rather than four and
caps nothing. A copy that does not match — or one that scrolled out of the
window before it could be checked — drops the latch, and the same check is
free to re-latch onto whatever is cycling now, so a loop that breaks and is
replaced by a different loop is still caught
(`a_second_long_cycle_relatches_after_the_first_one_broke`). Counting forward
without verifying each copy would turn ordinary prose after a broken cycle
into a false stop; that is what
`a_latched_cycle_that_breaks_does_not_count_toward_a_stop` pins, and it fails
if the mismatch branch is made to increment the count.

Replayed against the dump, the fixed guard stops at **12 925 of 46 692 bytes**
— the latched rung firing at its 4th cycle, ~9.5 KB into the loop — leaving
33.8 KB, some 8 500 tokens, ungenerated. The `has_cycles` in-window fast path
is unchanged and still handles short periods, and
`wide_window_catches_paragraph_loops_the_default_misses` (600-byte period)
still passes through it.

## A cycle can be longer than the whole window, and then nothing fires

`repro-1788796284` (cranky-watt, saved by hand — the parent dump shows no loop
at all, and that is the trap). The parent's four passes are 435, 1690, 2768 and
4328 bytes of think text, all well under the 8 KiB window and none of them
cyclic; the turn ends with a 48-byte assistant message after
`Tool error: sub-agent failed: interrupted`. Fifty minutes, no edit, and by
every signal the guard publishes the model never looped.

The loop is entirely in the sidecar, `repro-1788796284.sub-1.md`. Thirteen
ordinary passes, then a fourteenth of **189 744 bytes** that is one cycle
repeated twenty times:

| | |
|---|---|
| pre-loop reasoning | 8 904 B |
| cycle | **9 042 B, byte-exact** |
| cycles | **20** (21 occurrences, no drift) |
| repeated span | 180 840 B, 95% of the pass |

The cycle is a whole deliberation, not a paragraph: "This is getting complex.
Let's step back. We can use a subagent? We are already a subagent…", a
partition of the 32 leaf files into three lettered sub-agent batches, a
Python-script plan, an argument that the script breaks builder structs, and
back to "This is getting complex. Let's step back." It ran to the 50 000-token
`n_predict` cap — 189 744 bytes is about 47 K tokens — which is what the parent
saw as `interrupted`.

Both rungs are blind here, and it is one inequality:

| rung | cycles | window needed for period `p` | 9 042 B at 8 KiB? |
|---|---|---|---|
| `repeating` → footer marker | 2 | `2p` = 18 084 | ❌ |
| latched stop (`extend_latched`) | 2 to latch, then forward | `2p` = 18 084 | ❌ |

The latch fix from the section above removed the *stop* rung's dependence on
the window, but the warn rung is still `cycle_period(2)`, so it needs two
copies resident and caps the detectable period at `window / 2` = 4096. Nothing
can latch onto a 9 042-byte cycle, so nothing counts forward from it either.
The previous finding closed the gap `(window/4, window/2]`; this one is simply
`p > window/2`, and the fix that closed the first gap does not touch it.

Two lessons that generalize past this dump:

- **A sidecar loop is invisible in the parent.** The parent transcript is the
  sub-agent's *task string and final tool result*, so a sidechain that burns
  47 K tokens cycling shows up as one tool error. `Agent::report_guard` prints
  red guard lines on the main window for stops, but nothing fired here, so
  there was nothing to print. Reading only the top-level dump and concluding
  "not a loop" is the failure mode; check every `.sub-N.md` sidecar first,
  and note that a hand-saved `repro-<secs>` (rather than `repro-loop-<secs>`)
  is itself evidence that no guard fired.
- **Cycle length scales with the size of the decision, not the prose.** Every
  earlier dump cycled a paragraph (123 B to 2.4 KB). This one cycles a
  *plan*: enumerate an approach, enumerate its objection, abandon it, restart.
  A guard sized for repeated sentences is structurally the wrong size for
  repeated plans, and the request that produces repeated plans — "use
  sub-agents for the independent tasks" over a 32-file refactor — is exactly
  the kind plank is for. Expect the period to keep growing; a detector whose
  ceiling is any fixed multiple of a fixed window will keep being outrun.

Fixed 2026-09-07 by two rungs that recognise nothing, because recognition is
what has the floor. Cycle detection needs two copies before it can name a
cycle, so its latency floor is `2p` — 18 KB here, and unreachable anyway when
`2p` exceeds the window. A budget has no floor: it just stops counting.

- **`REPEAT_THINK_BUDGET` = 16 KiB** (since 2026-09-10 `repeat_think_budget`,
  a tenth of the context window with 16 KiB as the floor), a per-pass cap on reasoning bytes
  (`RepeatGuard::with_think_budget`), fed back as `THINK_BUDGET_ERROR` and
  counted towards `MAIN_REPEAT_TRIP_CAP` alongside a real loop, since both
  leave the prompt materially unchanged at temperature 0.
- **`NO_PROGRESS_BYTE_BUDGET` = 32 KiB**, a per-*turn* cap on output generated
  without a `PROGRESS_TOOLS` call (`write`, `edit`, `bash`, `bash_stop` — the
  same set plan mode blocks). This is the rung the per-pass budget cannot be:
  the parent turn above passes every per-pass check ever written.

Both were sized against the 32 dumps in `~/.plank/repro`, 939 passes, and the
numbers are the argument:

| rung | wall clock | loops caught | healthy tripped |
|---|---|---|---|
| think budget @ 7 KiB | 89 s | 7 | 27 of 923 (2.93%) |
| **think budget @ 16 KiB** | **204 s** | **5** | **1 of 923 (0.11%)** |
| no-progress @ 32 KiB | ~7 min | the non-cyclic turn | 1 of 575 runs (0.17%) |

Do not lower the think budget to buy latency. 7 KiB inverts the ratio from 5:1
to 1:3.9, and a rung that fires on 3% of good reasoning is one the user turns
off. The loops worth catching here are enormous — 190 KB, 46 KB, 45 KB — so
latency is the cheap axis and false positives are the expensive one. The
corpus is also thin in its healthy tail and drawn from two projects, so treat
0.11% as an order of magnitude, not a rate.

Two deliberate choices in the wording. A budget stop is reported as
`stopped an over-budget pass`, never as a loop: it is a weaker claim, and
sending whoever reads the dump looking for a cycle that is not there is how
the next finding gets misdiagnosed. And `THINK_BUDGET_ERROR` tells the model
to pick the option it was leaning towards rather than accusing it of
repeating itself, which would be a false statement it then has to reconcile.

Still open: five looping passes sit below 12 KB where neither rung reaches
them, and four of those are *drifting* loops with no byte-exact period (the
"stutter" section below). Lowering a budget is the wrong instrument for them,
at 27 healthy passes to catch 2. A duplicate-line ratio is the right one, and
the corpus separates on it cleanly: healthy passes sit at zero duplicate
lines, those four at 15 to 25 percent.

## An attempted mutation is not progress

`repro-loop-1788833715` (cuddly-columbus, auto-saved 2026-09-08) ran for
5h7m and reached 635K transcript tokens. It made 153 `bash`, 305 `edit`, and
201 `read` calls. Late in the turn, the model repeatedly read the same three
lines containing `popup.bounds()()` and reasoned verbatim about why its own
mechanical replacement was correct. The exact-call guard eventually refused
the identical reads and ended the turn, but only after the work had already
stalled for hours.

The no-progress budget existed but did not apply: it reset from the parsed
call name, treating every `edit`, `bash`, and `bash_stop` as a world-changing
action. Many edits had returned an anchor error, while the shell calls were
builds, searches, or other read-only commands. Invocation and exit status do
not prove that a task advanced.

Fixed by sampling `ToolContext::last_written` after each dispatch, before the
UI consumes it for `/open`. The field is assigned by `write` and `edit` only
after a successful file write, so only that evidence resets the budget. The
plain REPL and TUI paths use the same rule. Shell commands are deliberately
not assumed to have made progress: detecting arbitrary shell mutations
reliably would require a separate workspace-mutation witness, rather than an
exit-status heuristic. Regression tests cover a failed `edit`, a successful
read-only `bash`, and repeated successful writes.

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


## The trip cap stops the bleeding; it does not stop the loop

`repro-loop-1788862078` and `repro-loop-1788862135` are one session
(`zany-magellan`, 2026-09-08, 57 seconds apart) surveying an unfamiliar repo:
40 messages, 57 KB of transcript, `--think-low`, temperature 0. The two dumps
are the same transcript one guard round apart, so they read as a controlled
experiment on the recovery path fixed the day before.

What they show:

- **`MAIN_REPEAT_TRIP_CAP` works as designed.** Trip 1 stopped a pass and put
  `REPEAT_LOOP_ERROR` back as a tool result; trip 2 stopped the next pass and
  ended the turn. Two dumps, no third pass, no user Ctrl-C. That is the
  entry above ("A stopped pass is regenerated verbatim") behaving correctly.
- **The regeneration is now byte-identical.** In the `bumbling-einstein` pair
  the pass after the error at least opened differently ("Need stop repeating.
  We have enough."). Here the assistant message after the guard's error
  reproduces the poisoned reasoning *character for character*, including the
  point where it truncates. The guard's instruction not to resume that
  reasoning is in the prompt, and it loses to the 8 KiB of repetition sitting
  in front of it.

That is the sharper form of the temperature-0 lesson: it is not that the
one-line error is too weak a nudge, it is that we ask the model to continue a
context whose tail *is* the degenerate cycle. "Let me look at the docs for the
SSH section. Let me also look at the docs for the SSH section." has exactly one
likely continuation, and no appended instruction outranks it. Retaining the
looping assistant turn verbatim is what makes the next pass deterministic in
the wrong direction.

The fix is therefore to change the transcript, not the wording. Fixed
2026-09-08: after the dump is written (`loop_repro_line` runs first, so the
dump keeps the reasoning verbatim and only the model's copy loses it), the
guard rewrites the final `<think>` block of the message it just stopped down to
`STOPPED_REASONING_STUB` (`Agent::stub_last_reasoning`, both turn paths). The
cap stays as the backstop, unchanged. Anything the pass emitted *after* leaving
`<think>` is visible output the user has already been shown and is kept, so a
partial answer is not silently withdrawn from the model's own context.

The KV cost is the part that turned out to be free, and it is worth recording
why, because the obvious reading is that this is microcompact's mid-transcript
rewrite and has to pay microcompact's rung restore. It is not: the rewritten
message is the transcript's *last*, so every earlier section still matches
byte for byte, `ds4_session_common_prefix` reuses the whole prefix, and the
recovery pass prefills the stub plus the guard's tool result and nothing else.
No rung is invalidated either — every rung sits at a shallower depth and still
describes an intact prefix — so neither `discard_ladder` nor
`truncate_ladder_to` is called. A tail rewrite is cheap by construction; the
expensive rewrites are the ones with transcript behind them.

What it trades is the property that the recovery prompt can see what looped.
The guard's tool result still says *that* the reasoning was stopped and why, so
the model is not left guessing at the reason, only at the text — which is the
text we are trying to keep it from rebuilding. The regression test is
`the_pass_after_a_reasoning_stop_is_not_shown_the_loop`, which asserts the
second prompt carries the stub and the error but not the cycle; it fails with
the call to `stub_last_reasoning` commented out.

Two secondary observations from the same pair:

- **The stall was visible eight messages earlier, at the level of intent.**
  From roughly message 30 the reasoning repeats the same *plan* while the text
  varies: "let me check the docs for the SSH section and the `ssh` feature"
  recurs across four passes, three of them *after* the model has already read
  `## SSH Integration` and gotten its answer, and "let me verify the build/test
  commands work" is announced in three consecutive passes without a single
  `bash` call ever being emitted. An intent that is restated and never executed
  looks like a cheaper trip-wire than character-level repetition. **Measured
  against the corpus, it is not**, and the negative result is worth keeping so
  nobody re-derives it. Scored over the 39 dumps in `~/.plank/repro` (loop
  dumps against ordinary ones, normalized reasoning sentences of at least 25
  characters, `Agent` passes split on the transcript's role tags): a sentence
  recurring in every pass of a 3-pass window fires on 4 of 13 loop dumps and 3
  of 26 ordinary ones. Restricting it to intent phrasings (`let me`, `i need
  to`, `i will`, …), which is the shape actually observed here, makes it
  *worse* rather than better: 5/13 against 3/28 at a 2-pass window, and 2/13
  against 2/26 at three. A rung that fires as often on good turns as on bad
  ones is not a rung. It also buys almost no latency — in this session's own
  dump the first hit is at pass 19 of 22, three passes before the exact guard
  already stopped it.

  The half that would discriminate is the one the script cannot cheaply check:
  whether the announced action was ever *executed*. Every non-final pass in
  these dumps does emit some tool call, so "announced and never acted on"
  needs the announcement matched against the calls, which is semantic matching
  and not a guard rung. Left unshipped, like the fuzzy line detector in "The
  loop is preceded by a stutter the exact-match guard cannot see", and for the
  same reason.
- **Nineteen minutes bought nothing.** The turn ran 9m40s to trip 1 and ended
  at 12:08:55 having read three doc chunks and listed two directories, with no
  edit and no answer. The cap is the floor on the damage, not a fix; the cost
  of a loop is still the whole turn.

## The think budget fired on reasoning that was not looping

`repro-loop-1789051332` through `repro-loop-1789053127` (2026-09-10, sessions
`dapper-jagger`, `grumpy-churchill` at temperature 0 and `witty-jagger` at
0.6) are one request — "add the ability to expand a folder to select single
items with the '+' key" against the tommaso disk sweeper — retried three times
for about forty minutes, with zero edits. All seven stops are
`THINK_BUDGET_ERROR`; no dump has a byte cycle.

What filled the 16 KiB, pass by pass:

| dump | fenced code lines | "Actually / Wait / Hmm" | option-weighing lines |
|---|---|---|---|
| `1789051332` | 135 | 15 | 3 |
| `1789051554` (recovery) | 122 | 17 | 0 |
| `1789051976` | 39 | 0 | 2 |
| `1789052266` (recovery) | 101 | 0 | 3 |
| `1789052500` (2nd recovery) | 146 | 0 | 0 |
| `1789052875` | 53 | 16 | 4 |
| `1789053127` (recovery) | 82 | 23 | 4 |

Two distinct things happen. The **first** pass of each session stalls on
genuine design forks — sync or async child measurement, one-level or recursive
expansion, how parent and child ticks interact on delete — questions the user
could answer in seconds and which the `ask` tool exists for. Across the 78
dumps in the repro directory the model has called `ask` in exactly one session
(`repro-quit-1788876504`), both times about process, never about design.

The **recovery** passes show the larger problem. After the budget error the
model does decide in one sentence, as the error asks, then drafts the entire
implementation inside `<think>` as fenced Rust — struct fields, `expand()`,
`ticked_paths()`, the poll loop — until the budget cuts it again mid-function.
`stub_last_reasoning` then erases the draft, so the next pass restarts the
code from zero, and the third trip ends the turn. A 16 KiB budget is smaller
than a multi-file feature drafted as code, so a task this shape could never
finish through reasoning.

Fixed 2026-09-10 in the cheap direction first: the budget is now
`repeat_think_budget(ctx_size)`, `ctx_size / 10` bytes with the old 16 KiB as
the floor — ~102 KB on the 1M-token window, about six times the reasoning any
of these passes managed before the stop. The rung goes back to
being a backstop against drift rather than a ration; the exact-cycle rungs
and the no-progress budget still stop the shorter loops first. Raising it does
not teach the model to leave code for the edit tool; that, a prompt rule to
`ask` on user-visible design forks (reconciled with "Decide, do not
deliberate": decide on internal choices, ask on user-visible ones), and a
budget error that demands the first *edit* in the same pass rather than a
one-sentence decision, are the follow-ups.

## The review that was written in the wrong place

`repro-loop-1789060243` (2026-09-10, session `jazzy-koch`, DeepSeek V4 Flash
Vision, think low, `showThinking` off, temperature 0) is the counter-case to
the budget raise above, and it arrived the same evening. The binary was a
local build of the working tree, so it already carried the ~102 KB budget
despite reporting v5.0.4.

The request was `do a code review` on the tommaso repository. The first
twelve minutes were healthy: eight tool rounds reading every source file,
`Cargo.toml` and the docs, then `cargo test` and `cargo clippy`, each round
narrated with the one-line status the working-style rule asks for. The
synthesis pass then wrote the whole review inside `<think>` — thirty-one
numbered `**Bug N:**` entries, 30 KB — and never closed the block. Under the
old 16 KiB budget it would have been stopped near the seven-minute mark; under
the new one nothing stopped it, and with thinking hidden the user saw a token
counter and nothing else for 12m42s, then pressed Esc. The partial pass stayed
in the transcript.

The user typed `resume`. With its own 30 KB of analysis in context, the model
did not read it back: it restarted the synthesis inside `<think>`, produced
17 KB, and degenerated into a five-line cycle — the same "`draw`
`f.render_widget(Block::default()…)` for the X block. Good." sentence for the
header, categories, items, path and modal blocks, five times round. The
exact-cycle rung caught it after about 3 KB of repetition, as designed, and
the turn ended at 19:10:43: eighteen minutes, zero visible output, no review.

Three things follow.

**The budget size is not the lever.** The morning dumps were the fixed budget
cutting off a design that was not looping; this is the raised budget letting a
deliverable be drafted in reasoning until the user gave up. Both are the same
failure — a long, structured deliverable composed inside `<think>` and
restarted from scratch after every stop — seen from either side of a number.
Pulling the budget back down in reaction to this dump would only trade one
for the other.

**The narration rule does not reach the synthesis.** "Narrate progress
outside your thinking" is written around tool rounds — one line after
`</think>` before each stanza — and the model obeyed it there. The final pass
has no stanza to hang a line on, so the rule is silent exactly where the
output was. A review, an audit, a plan: anything whose deliverable is prose
rather than an edit ends in a pass this rule never touches.

**The `resume` restarted rather than resumed.** As in the morning's recovery
passes, prior reasoning in context was not used as a draft to finish but as
evidence to re-derive. Whatever the model is asked after a stop, it starts the
composition over.

### A prompt rule for intermediate findings

The candidate fix is in `WORKING_STYLE` (`src/sysprompt.rs`), plank's own
prompt text, not the C-parity section — so it costs a fixture regeneration
(`PLANK_REGEN_FIXTURES=1 cargo test`) and, like any system-prompt byte, a
fresh `fp1` and a sysprompt KV rebuild, but no parity concern. Something of
the shape:

> Write findings as you find them. When the answer is a list — a review, an
> audit, a survey of options — emit each item to the user as soon as you have
> it, after `</think>`, and move to the next. Do not accumulate the list in
> your thinking and write it out at the end: the user sees nothing until then,
> and a stop loses all of it. Your thinking is for deciding what the next item
> is, not for drafting it.

Why this shape rather than a generic "think less": the model already narrates
between tool rounds when told where the line goes, so the rule names the
place (`after </think>`) and the unit (one item), the same way the existing
rule does. It also gives the exact-cycle and budget rungs something to work
with: a pass that emits an item every few KB of reasoning has short think
blocks, which is where the guards have latency to spare, and a stop after
item 20 leaves twenty items on screen instead of none.

What it will not do on its own is stop the restart-after-stop pattern, and it
adds prompt bytes the model may weigh against "keep reasoning short". The
check is the one this document always uses: count, in the dumps that follow,
how many list-shaped answers reach the user before the pass ends, against
the four sessions this evening and this morning in which none did.

### The draft rung

Shipped alongside the rule rather than held back as its fallback, because the
rule only reaches a model that reads it and the dumps show what happens when
one does not. `RepeatGuard::drafting` scans the reasoning line by line: a line
that reads as a numbered deliverable heading — `**Bug 22:**`, `1. **Scan
worker panic**`, `### 3.`, `Finding 7:` — counts once, and every byte of a
line inside a ``` fence counts towards a second tally. Plain numbered
thoughts (`1. read the file`) do not count: a plan numbers its steps too, and
the difference between a plan and a written-out list is the bold title on
each item. Past `DRAFT_MIN_BYTES` = 8 KiB of reasoning, `DRAFT_HEADINGS` = 10
headings or `DRAFT_FENCED_BYTES` = 4 KiB of fenced code trips it.

The thresholds against the corpus: the review pass had thirty-one headings in
30 KB, so it would have stopped around the tenth, at roughly 9-10 KB and three
minutes rather than thirteen; the seven morning dumps carried 39-146 fenced
lines each, all past 4 KiB well before the 16 KiB budget that actually cut
them. The two-option design fork with a quoted snippet that opens most honest
passes stays under both counts, and the byte floor keeps a short outline from
tripping regardless of how it is numbered.

It sits between the cycle rungs and the budget in `stream_chunk_must_stop`:
a proven cycle still wins because it has evidence, and the draft rung goes
ahead of the budget because it can name the shape of the reasoning where the
budget can only name its size. Its message, `DRAFT_ERROR`, is the instruction
the symptom calls for — close the thinking, emit the items you already have
one at a time, make code changes with the edit tool — rather than the cycle
text's "you were repeating yourself", which would be a lie, or the budget's
"pick the option you were leaning towards", which addresses a deliberation
this pass was not having. It counts towards `MAIN_REPEAT_TRIP_CAP` like the
others, and `stub_last_reasoning` still erases the stopped reasoning, so the
restart-from-zero pattern is untouched by it; the rule is what is meant to
change where the next attempt writes.

### What the dump now records

Reading this dump meant reconstructing by hand what plank knew at the time:
whether the 30 KB pass was ended by Esc or a rung (the transcript shows an
unclosed `<think>` either way), how many reasoning bytes each pass had, and
what the guards were set to. Every repro now carries a `## Passes` table —
one row per generation pass with its end time and the gap since the previous
one, the agent that ran it, tokens and rate, the reasoning bytes the guard
saw, a latched cycle as period × copies, the draft rung's heading and
fenced-code counts, and the stop reason (`tool calls: N`, `answer`,
`interrupted by user`, `guard: cycle|draft|budget`, `tool error`) — and the
`## Generation` section states whether `tools.loopGuards` was armed and the
think budget in effect. The pass notes live on the agent
(`Agent::passes`, capped at `PASS_NOTES_CAP`), recorded at the four points a
pass's message is pushed: the plain and TUI turn loops, the serial
sub-agent loop and the fan-out fold. Future entries in this file should quote
the table rather than re-derive it.

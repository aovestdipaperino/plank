# Compaction with System-1 decisions

A review of [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction), a Claude
Code plugin that compacts context with Jev's typed decisions instead of a summary, and what its
design suggests for plank's micro-compaction. The review reads the source at commit `e3f262a`
(`src/`, `hooks/fast-jev.ts`, the tests and the plugin manifest). The test suite was not run.

Background on System-1 models (Jev, Laya, and the logit-reading open implementations) lives in
the `system-1-models` note in the shared vault. Plank's own System-1 path is `decide.rs` and
`Engine::decide`, described in `CLAUDE.md` and `docs/ARCHITECTURE.md`.

## What fast-jev-compaction does

Most compaction asks a model to summarise old turns, and a summary can quietly lose a file
path, an exact error or a constraint the user stated once. fast-jev-compaction never rewrites
anything. It deletes tool calls and tool results that Jev judges no longer needed, and returns
every user and assistant message verbatim and in order.

Every `tool_use` is paired with its `tool_result` by id. A call is pinned, and never touched,
when the call or its result sits in the first message or in the newest `preserveRecentMessages`
messages (6 by default). The state sent to Jev is the whole conversation, oldest first, with
each tool output replaced by a note such as `ok, 4213 chars (omitted)`. Tool inputs and all text
are included, and a `goal` defaults to the last three user prompts.

The state has to fit `maxStateTokens` (25k), and it shrinks in stages, each used only when the
previous one was not enough: tool inputs cut to 1,000, then 200, then 60 characters; long texts
abridged to a 400-character head and a 150-character tail, oldest first; old messages collapsed
to `[… N chars omitted …]`; old calls reduced to one line (`t12 Read file_path=src/a.ts → ok
480ch`); old messages without calls dropped; runs of call-only messages merged. If that still
does not fit, compaction throws. Tokens are estimated without a tokenizer (one per six letters,
half per digit, 0.9 per symbol), calibrated to land 2 to 18% above what Jev reports, where a
plain characters-per-token ratio undercounted the JSON-heavy states by up to 40%.

For every unpinned call Jev gets two `noul` (yes/no) questions. One asks whether the *call*
should stay, because knowing it was made, with its input, still matters. The other asks whether
the *result* should stay verbatim, because the assistant still needs its contents and re-running
the tool would not do. Questions are packed into batches so state plus questions stay under
`maxRequestTokens` (30k, below Jev's 32k request limit); the full state is resent with every
batch, and batches run concurrently.

The decision against `keepThreshold` (0.5) has three outcomes. When keep-result clears the
threshold, the call and its result both stay. Otherwise, when keep-call clears it, the call
stays and the result is cut to its first 300 characters plus a `re-run the tool if needed` note.
Otherwise the call goes, together with its result. A message left with no content is removed,
untouched messages come back as the same objects, and no result is ever left without its call.

The plugin is a thin adapter over the library, on Claude Code's early-access function-hook API
(2.1.274+, `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1`). A `session.compact` hook swaps the built-in
summary for the pruned transcript, and falls back to that summary when Jev fails, the key is
missing, the history cannot be fitted, or the reduction is below `minReductionRatio` (25%). A
`turn.complete` hook triggers compaction itself once context reaches `compactAtPercent` (60%).
Every run shows a toast and logs a per-call `decisions:` line with both probabilities.

## Review

### What it gets right

The design asks exactly the kind of question System-1 models are good at: many independent,
cheap yes/no judgements against one shared state. Because nothing is paraphrased, the failure
mode is "a tool result is missing and the agent re-runs the tool", which is recoverable, rather
than "the summary misstated a constraint", which usually is not noticed until it does damage.

Failures are loud. A missing answer, a non-finite probability, an HTTP error or malformed JSON
all throw (`noulAnswer`, `parseJevResponse`), and the hook turns every throw into the built-in
fallback with the reason in the toast. A missing answer is never quietly read as "keep" or
"drop".

The transcript stays consistent: a dropped call always takes its result with it, a pinned
result pins its call, and untouched objects keep their engine handles so the host keeps the
original content. The token estimator is calibrated against measured usage, the fitting stages
run from least to most lossy, the key is only read from plugin options, the environment or
settings, and the tests run against a fake Jev and never touch the network.

### What to watch

**Jev never sees the results it is deleting.** The state replaces every tool output with a
length note, so "should this output stay verbatim?" is answered from the call's input and from
what the conversation said afterwards. That keeps the state small, but an output whose
importance lies in its content, such as an unusual error line or a value later copied into
code, can only be protected indirectly, through later messages that mention it. Showing a short
head of each result in the state, the same head a dropped result keeps in the output, would give
Jev something to judge.

**The 60% trigger can bring lossy summaries forward.** Once context passes 60%, `turn.complete`
asks for compaction. When Jev cannot remove 25% (a short session, or a second pass over a
history it already pruned), the hook falls back to Claude Code's built-in summary, which then
runs at 60% instead of at Claude Code's own threshold. A pass that succeeds but leaves the
session above 60% triggers again on the next turn, and that second pass is the one most likely
to fall back. In the worst case, installing the plugin produces more summaries, and earlier. A
cool-down after a recent pass, or a fallback that declines instead of summarising when the
trigger was the plugin's own, would avoid it. This follows from reading the code, not from a
run.

**No timeout and no concurrency limit.** `Promise.all` over the batches uses the host's `fetch`
with no abort signal, so one hung request stalls compaction. Every batch resends the full
25k-token state, so a long history costs several full-state requests in flight at once.

**"Recent" is counted in messages, not turns.** In Claude Code transcripts a tool result is its
own user message, so the default of 6 messages protects about three tool rounds.

**Privacy and quality are unmeasured.** Every compaction sends the whole conversation text and
all tool inputs, paths and commands included, to TypeSafe. The tests cover the mechanics, not
the quality: nothing measures how often a dropped result was needed again later, which is the
number that decides whether 0.5 is the right threshold.

## What it means for plank

Plank's micro-compaction (`compact.rs`) already works on the same principle as the plugin: it
never summarises, it replaces old tool-result bodies with `MICROCOMPACT_STUB` and tells the model
to re-run the tool. The difference is how candidates are chosen. `clear_set` excludes results
under `MICROCOMPACT_MIN_BYTES`, image-bearing results, anything after the last task-list
injection and the current tool-call batch, and `microcompact` then clears the *oldest* remaining
candidates until it has reclaimed the pass's token budget. Age is the only ranking.

fast-jev-compaction suggests ranking by need instead. The plan below gets that verdict the way
the memory pass and prompt suggestions already get their answers: as a short, tool-free
**sidechain generation on the warm live session**, queued at the turn end and run at an idle
moment, and it consumes the verdict later, when a micro-compaction pass actually runs.

### Why the sidechain and not `Engine::decide`

`Engine::decide` runs on a dedicated session of `DECIDE_CTX` (16,384) tokens. For this job that
means rebuilding a fitted copy of the transcript the way the plugin does (truncated inputs,
abridged text, one line per old call), prefilling it from zero at every verdict, and throwing the
tool results away to make it fit, which recreates the plugin's worst blind spot.

The sidechain has none of those costs. `process_memory_job` and `generate_suggestion` both open
it with `begin_sidechain(task, true)`, which snapshots the live KV and pushes the task as one user
message; run one generation through `run_sidechain_quietly`; and close with `end_subagent_fork`,
which truncates the task and reply back out and restores the KV. Only the task and a few output
tokens are prefilled, and the model reads the candidates *with their full bodies*, because the
bodies are already in the live context. Running under `in_sidechain()` also means no rungs, no
payload and no transcript trace. The trade is that the answer is generated text to parse rather
than logprobs, which is also what the memory pass accepts.

### The job

**Queue at the turn end, never generate there.** A turn end sets `compact_verdict_pending` when
all of these hold: `context.microcompact` and the new setting are on, the turn was not
interrupted (`memory_pass_allowed()`), context use has reached `context.verdictAtPercent`, and
the candidate set has changed since the last verdict. The threshold sits below
`MICROCOMPACT_PRESSURE_PERCENT` (75) so the verdict is ready *before* a pass can fire; a default
of 60 leaves about three large turns of warning. Below it nothing is queued: a verdict nobody
consumes is wasted generation.

**Generate at idle, under the existing guard set.** `suggest::idle_work` gains a
`CompactVerdict` variant. Order: a pending suggestion first (it is worthless once the user
types), then a pending verdict, then a memory job, with the existing
`suggestions.memoryStarvationSeconds` guard still letting a starved memory job jump both. The
verdict ranks above memory because it has a deadline, the next pass, and the memory pass is
explicitly allowed to wait.

**Skip where the other two skip.** Inside a sidechain, while yielded to memory pressure
(`is_pressure_yielded`), and when `kv_reuse_probe` says the KV would rebuild from zero, probed
with the real rendered prefix exactly as `generate_suggestion` does. The last one keeps the cost
honest: the expensive case is precisely the one it declines. The preflight from
`process_memory_job` applies too (`last_ctx_used` plus task plus reply reserve must fit
`ctx_size`), and at 60% or more of the window it is the one most likely to bite.

**The task.** One user message listing the candidates from `clear_set`, each with a short id,
the tool, a one-line input summary and the body size, then the question:

```text
These earlier tool results may be cleared to free context. For each, decide whether its
full output is still needed for the work that remains; the tool can always be re-run.
R1  read src/compact.rs                  12.4 KB
R2  bash cargo test --lib memory         3.1 KB
R3  read docs/KV-CACHE.md                28.9 KB
Reply with exactly two lines and nothing else:
STALE: <ids no longer needed, comma separated, or none>
NEEDED: <ids still needed, comma separated, or none>
```

The rows are built by the same code that builds `clear_set`, so the ids and the pass can never
disagree about which result is which. The candidate list is capped (say 40 rows, oldest first)
so the task stays short on a long session. `n_predict` stays small, since the reply is two short
lines.

**Parse strictly, fail to age order.** Take the first `STALE:` and `NEEDED:` lines, keep only
ids that were offered, drop an id that appears in both, and treat anything else, whether prose,
an empty reply or a refusal, as no verdict. The stored result is a `CompactVerdict { depth,
entries: Vec<(fingerprint, Need)> }`, where the fingerprint is a hash of the result body at the
time of asking, so an entry dies with the body it judged. A typed prompt interrupts the
generation at its next token like the other two jobs. Nothing is requeued; the next turn end
queues a fresh verdict over the transcript as it then stands.

### Consuming it

`microcompact` keeps its budget loop and gains an order. Candidates are cleared **stale first,
then unjudged, then needed**, oldest first within each group, until the budget is met. A verdict
therefore only changes the *order*: it never adds a candidate outside `clear_set`, never shields
a result from a pass that genuinely needs the room, and never changes *when* a pass runs.

That last constraint is not theoretical. `MICROCOMPACT_PRESSURE_PERCENT` exists because an
eager pass in a 1M window that was only 7% full replaced every file the model had just read with
a stub; the model re-read them, lost them again, and cycled for two hours without an edit. That
is the same failure the plugin's 60% trigger risks, reached from the other direction. Here the
verdict can queue from 60%, but only the existing pressure gates start a pass.

One rule must not break: **the KV divergence point is the earliest result the pass actually
clears.** Today `microcompact_first_index` is the minimum of `clear_set`, which is the same thing
as the first result the age order clears. Under verdict order it is not: a pass that skips old
needed results and clears newer stale ones diverges later. The fix is to compute the ordered
plan once, and derive both the rung to restore and the rewrite from that one plan. Restoring the
rung below the old minimum stays *correct*, just wasteful (it re-prefills more than needed), so
getting this wrong degrades quietly rather than failing. It needs its own test.

Every consumer drops a verdict whose depth or fingerprints no longer match: new turns (the
depth moves), `/clear`, rollback, fork, full compaction, and micro-compaction itself, which
turns a judged body into the stub and so breaks its fingerprint. That is the same invalidation
rule `Suggestion` follows, and the same reason: a stale verdict still looks like a plausible
one.

### Settings

Under `context`, next to `microcompact`:

- `context.microcompactVerdict` (bool, **default `false`**). Unlike suggestions, this changes
  what the model loses, so it ships off, and shadow mode below is how it earns being turned on.
- `context.verdictAtPercent` (u32, default 60)

### Shadow mode, then the switch

For the first release the verdict is computed and logged but not used. At each pass, the log
records the age order the pass used, the verdict order it would have used, and which results
the two disagree on. A cleared result counts as *re-read* when a later tool call repeats the
same tool with the same input. That gives the number the plugin never measured: how often each
ordering cleared something the model then had to fetch again. The verdict order earns its place
only if it is measurably lower.

### Testing

Everything except the model's judgement is pure logic and testable with `ScriptedEngine`: the
queue conditions, the idle priority (including the starvation guard), every skip condition,
reply parsing, fingerprint invalidation, the stale-unjudged-needed order, the budget still being
met with every candidate marked needed, and the divergence-point rule above.

The lesson the System-1 branch and the suggestions spec both carry is binding here too. A
scripted reply cannot show the model answering as itself, listing ids it was never offered, or
marking everything `NEEDED`. So the plan carries a required real-model check as a deliverable
(a long live session on ds4 above 60%, with verdicts produced, parsed, and not uniformly one
answer) and a `PLANK_VERDICT_DEBUG` variable that prints the raw reply before parsing. Without
it, "the model returned nothing" and "the parser rejected everything" look identical from
outside.

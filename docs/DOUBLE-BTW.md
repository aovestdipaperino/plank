# The Second `/btw` Rebuilds the Prompt

Status: diagnosed, with a fix specified in §4. Reproduced on Metal against a
real model; **not** specific to multiplexed asides — the freeze path has it too.

Related: [`BTW-SUSPEND-DESIGN.md`](BTW-SUSPEND-DESIGN.md) §4.3 (the resume this
breaks), [`SESSION-CLONE-DESIGN.md`](SESSION-CLONE-DESIGN.md) §6.2 (the
multiplexed aside that made it easy to hit twice in one turn).

## 1. Symptom

The first in-pass `/btw` of a turn is fast. The second one stalls for a minute
or more before a single token of the answer appears, and the main task stops
making progress for the same stretch. It reads as a hang; it is not.

The footer gives it away — `↑ 5.0k/8.9k tokens` is a *prefill* counter, and it
is counting the whole conversation. With `PLANK_KV_DEBUG` set:

```
reconcile: 6 spans held, 5 sections in, kept 4
  diverged at 4: incoming Assistant len=2223 head="The user wants me to count from 1 to 80..."
              held     Assistant len=1204 head="The user wants me to count from 1 to 80..."
generate: prompt=14602 cached=0 prefill=14602 (0.0% reused)
  full rebuild: prompt is a 13758-token prefix of a 14366-token live KV;
                sync cannot rewrite behind the live end
```

Zero percent reused. Everything is recomputed.

## 2. Why the first one is fine

`ds4_session_sync` reuses the live KV only for a prompt that **extends** its
end (`engine::reusable_prefix`: `prompt_len >= pos && starts_with(prompt, kv)`).
Rewriting behind the live end is not an in-place operation for the backend's
SWA rows, compressed KV rows and indexer frontiers, so anything else takes the
reset branch.

At the first freeze the transcript holds one assistant span for the partial
reply:

| | tokens |
| --- | --- |
| span | `prefix + partial + EOS` |
| live KV | `prefix + partial` |

The EOS is sampled and recorded but never evaluated (`record_reply`), so the KV
is exactly one token shorter than the span — the `-1` the snapshot tests
already encode. The resume prompt is `…[assistant]\n{partial}`, whose sections
match the held spans exactly, so `build_prompt` keeps them and appends a fresh
assistant prefix. The prompt therefore *extends* the KV by `EOS + prefix`, and
the log shows `prefill=3 (100.0% reused)`.

## 3. Why the second one is not

When the resumed pass finishes generating, `record_reply` pushes a **second**
assistant span:

```
spans:   … , Assistant("<partial>"),  Assistant("<continuation>")
live KV: …   prefix partial EOS       prefix2 continuation
```

The UI does not model it that way. `run_turn` splices the two halves into one
string (`format!("{resumed_prefix}{}", out.assistant_text)`) and pushes a single
`Message::assistant`, so the rendered transcript the next turn reconciles
against has **one** assistant section holding the whole reply.

At the second freeze, reconciliation compares section 4 — the merged
`partial + continuation` — against held span 4, which is only `partial`. They
differ, so `common_prefix` stops at 4, `build_prompt` truncates the transcript
there and retokenizes the merged text from scratch. The resulting prompt is a
*prefix* of the live KV rather than an extension of it, and the engine rebuilds
all 14602 tokens.

Nothing about this is specific to a `/btw`. It is the general consequence of a
turn generating twice into one assistant message, which today only an in-pass
suspend causes.

### 3.1 It predates multiplexing

Confirmed, not assumed. Running the identical scenario with multiplexing
disabled — the plain freeze/answer/resume path — produces the same divergence
and the same `0.0% reused` full rebuild. The multiplexed aside only makes it
easier to reach, because the main task keeps running and a second question
becomes natural.

## 4. Fix: one assistant turn is one span

Make the transcript describe the reply the way the UI does: when a generation
continues an assistant span that is already the last span, **merge** rather than
append.

The token buffer does not change at all — the merged span covers exactly the
same ids in the same order — so the KV stays byte-identical and the `-1` EOS
convention still holds. Only the span boundary and its text key change.

```
before:  Assistant("<partial>") | Assistant("<continuation>")
after:   Assistant("<partial><continuation>")
tokens:  prefix partial EOS prefix2 continuation EOS2   (unchanged)
```

### 4.1 The whitespace trap

The merge cannot simply concatenate the two spans' stored text, because
`record_reply` trims its input: the partial's trailing whitespace is already
gone by the time it is a span. The UI concatenates the **raw** halves, so a
partial ending in a newline would produce

- UI section: `"<partial>\n<continuation>"`
- merged span: `"<partial><continuation>"`

and the divergence would simply move one span along — the same bug, harder to
see.

So the fix has two halves:

1. **Spans store raw reply text.** Stop trimming in `record_reply`; a merge then
   concatenates exactly what the UI concatenates.
2. **Comparison trims.** `TokenTranscript::common_prefix` compares
   `span.text.trim_end()` against the section key. This is not a loosening:
   `parse_sections` already trims every section's trailing whitespace, so
   trimming the other side is what makes the two comparable. Before this change
   the trim happened at record time for assistant spans and not at all for the
   rest; doing it at comparison time covers every role uniformly.

### 4.2 Where

- `ds4tokens.rs`: `TokenTranscript::merge_last_assistant` (or an
  `extend_last_span`), plus the `trim_end` in `common_prefix`.
- `ds4engine.rs`: `record_reply` merges when `spans().last()` is an assistant
  span, and stops trimming its `text` argument.

Two adjacent assistant spans can only come from a resumed pass — ordinary turns
alternate user and assistant — so "last span is Assistant" is an unambiguous
signal and needs no extra state threaded through.

## 5. Tests

| Test | Level | Proves |
| --- | --- | --- |
| `merging_the_last_assistant_span_keeps_the_token_buffer` | `ds4tokens`, CI | the merge changes spans, never ids |
| `common_prefix_ignores_trailing_whitespace` | `ds4tokens`, CI | §4.1's comparison rule |
| `two_resumes_keep_one_assistant_span` | `ds4engine`, CI | a second continuation merges rather than appends |
| `second_aside_reuses_the_prefix` | `ds4engine`, Metal | the headline: freeze, aside, resume, freeze, aside — and the second aside still reuses |

`c_parity` must pass unchanged: the merge is a span-index change, and the wire
bytes are exactly what they were.

## 6. Non-goals

- Changing when a turn generates twice. The suspend/resume shape stays as it is.
- Making the UI model an assistant turn as several messages. One reply is one
  message; this brings the engine's view into line with that, not the reverse.

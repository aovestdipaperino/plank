# Roadmap

Planned work, organized by target release. The live board is the
[plank roadmap project](https://github.com/users/aovestdipaperino/projects/1); this file is the durable
summary. Issues are the source of truth for scope and discussion —
versions here follow the rules in [`VERSIONING.md`](../VERSIONING.md)
(minor bump when the sysprompt text or engine snapshot format changes,
major when older cached state must not be trusted).

Current release line: **v1.3.x**.

## v1.4.0 — concurrency and side channels

- **Model worker thread + event multiplexing** ([#12](https://github.com/aovestdipaperino/plank/issues/12), architecture section): move the engine off the UI thread, matching the C reference's worker design, so the prompt stays live while the model runs and async job output renders mid-turn.
- **/btw side questions, done properly** ([#18](https://github.com/aovestdipaperino/plank/issues/18)): mid-turn live prompt with boundary-scheduled ephemeral answers, per [`BTW-DESIGN.md`](BTW-DESIGN.md); ends with un-gating from the `images` flag.
- **Agent teams** ([#19](https://github.com/aovestdipaperino/plank/issues/19)): named agent definitions and multi-agent orchestration, building on the `/subagent` sidechain.

## Next minor releases (unslotted)

- **! command refinements** (split from #16, follow-up to #4):
  - feed `!` output into the transcript ([#20](https://github.com/aovestdipaperino/plank/issues/20)) — blocked on the model-format investigation;
  - mode-aware history navigation ([#21](https://github.com/aovestdipaperino/plank/issues/21));
  - live output streaming ([#22](https://github.com/aovestdipaperino/plank/issues/22)) — natural follow-on to the worker thread.
- **Persistent memory across sessions** ([#23](https://github.com/aovestdipaperino/plank/issues/23))
- **Saving and restoring named sessions** ([#24](https://github.com/aovestdipaperino/plank/issues/24))
- **Remote-control interface** ([#25](https://github.com/aovestdipaperino/plank/issues/25))

## v2.0.0 — session format change

- **Per-session engine KV payloads** ([#12](https://github.com/aovestdipaperino/plank/issues/12), sessions section): persist the engine KV cache alongside transcripts so `/switch` resumes without re-prefilling; unblocks a real **`/strip`**. Major bump because the session format changes.

## Ongoing

- **C reference parity** ([#12](https://github.com/aovestdipaperino/plank/issues/12)): the tracking issue for behavior the `ds4_agent` C reference has and plank lacks; checked off item-by-item as they land.

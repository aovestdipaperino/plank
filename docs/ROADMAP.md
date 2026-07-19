# Roadmap

Planned work, organized by target release. The live board is the
[plank roadmap project](https://github.com/users/aovestdipaperino/projects/1); this file is the durable
summary. Issues are the source of truth for scope and discussion —
versions here follow the rules in [`VERSIONING.md`](../VERSIONING.md)
(minor bump when the sysprompt text or engine snapshot format changes,
major when older cached state must not be trusted).

Current release line: **v2.0.0** (beta channel).

## v1.4.0 — concurrency and side channels

- ~~Model worker thread + event multiplexing~~ (#12, architecture section) — landed in `76a6428`: TUI turns run on a scoped worker thread, the prompt stays live, and queued lines drain between tool rounds.
- **/btw side questions, done properly** ([#18](https://github.com/aovestdipaperino/plank/issues/18)): mid-turn live prompt with boundary-scheduled ephemeral answers, per [`BTW-DESIGN.md`](BTW-DESIGN.md); ends with un-gating from the `images` flag.
- **Agent teams** ([#19](https://github.com/aovestdipaperino/plank/issues/19)): named agent definitions and multi-agent orchestration, building on the `/subagent` sidechain.

## Next minor releases (unslotted)

- **! command refinements** (split from #16, follow-up to #4):
  - feed `!` output into the transcript ([#20](https://github.com/aovestdipaperino/plank/issues/20)) — blocked on the model-format investigation;
  - mode-aware history navigation ([#21](https://github.com/aovestdipaperino/plank/issues/21));
  - live output streaming ([#22](https://github.com/aovestdipaperino/plank/issues/22)) — natural follow-on to the worker thread.
- ~~Persistent memory across sessions~~ (#23) and ~~named session save/restore~~ (#24) — first cuts landed in `a203a3b` (`/remember` + layered `MEMORY.md`; session `meta` trailer, `/tag`, `/resume` picker).
- **Per-session engine KV payloads** ([#12](https://github.com/aovestdipaperino/plank/issues/12), sessions section): persist the engine KV cache alongside transcripts so `/switch` resumes without re-prefilling; unblocks a real **`/strip`**. The session format change means the release that ships it must handle (or discard) payload-less older sessions gracefully.

## v2.0.0 — remote

The **session snapshot/restore foundation** (`src/snapshot.rs`: `SessionSnapshot` + the unconditional-restore `RestoreOnDrop` guard, and `Engine::generate_aside`) landed in `3e86c83` and underpins #27, #29, #28, and #12. All items below are on `main` locally. **Runtime behavior of the `cfg(ds4_engine)` paths (snapshot round-trips for #27/#29, the #28 scheduler's token interleaving and per-session KV isolation) is compile/inspection-verified only — it still needs a manual smoke test on a Metal box with a real model.**

- ~~**Mid-generation /btw suspend**~~ ([#27](https://github.com/aovestdipaperino/plank/issues/27)) — landed in `1858635`: freeze an in-flight generation, answer the aside, resume with ~zero re-prefill via `generate_aside`. **On by default**; disable with `--disable-btw-suspend` (`18e79ef`). Falls back to the boundary queue when off or the engine lacks aside support. Design: [`BTW-SUSPEND-DESIGN.md`](BTW-SUSPEND-DESIGN.md).
- ~~**`/checkpoint` in-session rollback points**~~ ([#29](https://github.com/aovestdipaperino/plank/issues/29)) — landed in `665cfbd`: `/checkpoint [name]` captures/lists, `/rollback <name>` restores transcript + KV (itself undoable via an auto `pre-rollback` snapshot); EchoEngine falls back to transcript-only. Snapshot access is via the `Engine::snapshot_kv`/`restore_kv` methods wrapping `SessionSnapshot`. Design: [`CHECKPOINT-DESIGN.md`](CHECKPOINT-DESIGN.md).
- ~~**Remote-control interface**~~ ([#25](https://github.com/aovestdipaperino/plank/issues/25)) — drive a running plank instance from another process or machine. Design: [`REMOTE-CONTROL-DESIGN.md`](REMOTE-CONTROL-DESIGN.md). Transport foundation in `c7c37a7` (WebSocket protocol, `BroadcastBus` scrollback replay, token auth, single-controller/many-mirror policy, `--control*` flags); **live wiring landed** in `243ba8a` — the server shares the agent's `TurnShared`/bus, so remote `prompt`/`command`/`btw`/`interrupt` frames drive the real turn loop and output mirrors, with a 15s reconnect grace window. **Remaining:** plain-REPL (non-TTY) remote drive and a `plank remote <url>` CLI client (documented TODOs in `src/remote/control.rs`).
- ~~**Remote LLM support via llms-sdk**~~ ([#26](https://github.com/aovestdipaperino/plank/issues/26)) — remote-hosted ds4 and third-party providers behind the `Engine` trait. Design: [`REMOTE-ENGINE-DESIGN.md`](REMOTE-ENGINE-DESIGN.md). **Flavor (a)** (`a4da712`): `plank serve` HTTP+SSE host + `RemoteDs4Engine` client (`--remote URL`), sync `ureq`, no async runtime. **Flavor (b)** (`216dd79`): OpenAI-compatible provider end-to-end via `--provider openai` (covers vLLM/Ollama/OpenRouter/Together), through the `Prompt::{Flat,Structured}` engine-input widening, a machine-readable tool registry, and native-tool-call→DSML synthesis. **Remaining:** Anthropic Messages provider (stubbed), cross-turn tool-call-id threading.
- ~~**Shared reference-counted engine**~~ ([#28](https://github.com/aovestdipaperino/plank/issues/28)) — one long-lived model, many concurrent sessions, amortizing weights/Metal context/warm prefix. Design: [`SHARED-ENGINE-DESIGN.md`](SHARED-ENGINE-DESIGN.md). **Landed** in `1760670`: the `Ds4Engine` → `Ds4Model` + `Ds4Session` split (always on, behavior-preserving; also unblocks #12 and a future `/switch`), `Arc`-refcounted `EngineHost` + admission control, and a cooperative single-GPU-thread round-robin scheduler — behind `--shared-engine` (default off) with `--max-sessions`. v1 uses round-robin K-token slices, non-preemptible prefill, one per-session `ctx_size`. **Remaining:** `/info` live-session accounting and idle-KV reclamation (v2).

## Ongoing

- **C reference parity** ([#12](https://github.com/aovestdipaperino/plank/issues/12)): the tracking issue for behavior the `ds4_agent` C reference has and plank lacks; checked off item-by-item as they land.

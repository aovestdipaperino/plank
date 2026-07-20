# System prompt: construction, caching, and invalidation

How plank builds the system prompt, why its bytes are frozen, how its KV
prefill is cached on disk, and what invalidates that cache across versions.

## Two kinds of "system" text

Plank distinguishes text that must never change from text that changes every
session:

| | Static tools prompt | Dynamic session context |
| --- | --- | --- |
| Source | `src/sysprompt.rs` (+ `src/resources/tools_prompt_after_edit.txt`) | `src/context.rs` |
| Content | Tool descriptions, editing rules, DSML syntax, JSON schemas, MCP tool schemas and server instructions | git branch/status/commits, discovered AGENTS.md files, current date |
| Stability | Byte-for-byte identical to the C reference (`tests/c_parity.rs`) | Recomputed at every session start |
| Transcript position | The `[system]` section | The first `[user]` message of the session |
| Cacheable | Yes — this is what `sysprompt.kv` snapshots | No — deliberately kept out of the cached prefix |

### The static prefix (`sysprompt.rs`)

`build_system_prompt(user_system, mcp_servers)` composes:

1. `build_tools_prompt` — three verbatim C string constants (intro, editing
   instructions, and the schemas/rules tail), the schemas of any MCP tools
   loaded at startup, and an `# MCP Server Instructions` block for servers
   whose initialize response carried an `instructions` field. The tail lives
   in a resource file included via `include_str!` because a `\`-continued
   Rust string literal would silently strip the JSON schemas' indentation
   (see FINDINGS.md).
2. The user's `-sys`/`--system` text, appended after a blank line when
   non-empty.

The C tokenizes these two parts differently — the built-in prompt goes through
the chat template so the DSML markers become control tokens, user text is
plain content — which is why they must stay separable (see the
`build_system_prompt` doc comment).

The bytes matter because the DeepSeek V4 Flash model was trained on the C
agent's exact prompt. `tests/c_parity.rs` compares the Rust constants against
committed fixtures and, when the `refs/ds4` submodule is present, against the C
source itself.

### The dynamic context (`context.rs`)

`ContextContent::new()` collects git status (truncated to 2000 chars),
AGENTS.md contents, and the datetime line, and `combined()` renders them as
one block. The UI pushes this as the session's first *user* message, not into
the system prompt — precisely so the system prompt bytes (and therefore the
KV snapshot below) stay identical across sessions in different repos on
different days.

### The pressure-based reminder

The tools prompt is re-injected mid-session as a user-turn "system prompt
reminder" (`build_system_prompt_reminder`) once the token-estimate distance
since it was last seen exceeds `SYSTEM_PROMPT_REMINDER_TOKENS` (50,000) —
pressure-based, not periodic, mirroring the C. `/new` resets the tracker.

## The `sysprompt.kv` checkpoint

Prefilling the ~10k-token system prompt on a Metal engine takes real seconds
at every launch. `Ds4Engine::warm_system_prompt` avoids that with a disk
snapshot at `~/.plank/kvcache/sysprompt.kv`:

```
sha1(model_name \0 system_prompt_text) '\n' <engine KV snapshot bytes>
```

Warm-up flow (`src/ds4engine.rs`):

1. Compute the fingerprint for the current model + composed system prompt.
2. **Fast path** — if `sysprompt.kv` exists and its first line equals the
   fingerprint, restore the snapshot into the live session: no prefill at all.
3. **Slow path** — otherwise prefill the system-prompt tokens (streaming
   progress so the UI can paint the bar), then write a fresh checkpoint.
   Saving is best-effort: a failed write just means the next launch prefills
   again.

The fingerprint is the trust boundary: a checkpoint is *only* restored when
model name and prompt text both match, so editing `-sys` text, changing MCP
servers (their schemas and instructions are part of the prompt; servers are
started *before* the prompt is composed), or swapping models each naturally
miss the cache and rebuild. A stale checkpoint is never trusted or patched —
KV-cache discipline is "reuse genuinely matching token prefixes or rebuild".

### The static/volatile boundary, formally

The rule the table above encodes: **an input may enter the composed system
prompt only if it is stable across sessions on the same machine and config.**
Anything that changes per session — the date line, git state, AGENTS.md
contents — must be injected as the session's first *user* message
(`context::ContextContent`) instead. Violating this doesn't break
correctness; it silently makes `sysprompt.kv` single-use, rebuilding the
multi-second prefill on every launch because the fingerprint never matches.
The `fingerprinted_prompt_contains_no_volatile_bytes` test in
`src/sysprompt.rs` guards the boundary by asserting the composed prompt is
deterministic and contains no date- or git-derived markers; both code seams
carry a "cache-boundary rule" doc comment pointing here.

Within a running session, the same discipline continues past the system
prompt: the engine keeps one live session, and each turn re-syncs only the
token suffix beyond the common prefix (`ds4_session_common_prefix`), splicing
the previously *sampled* reply tokens so retokenization drift cannot poison
the prefix.

## Version-driven invalidation

The snapshot's binary format follows the engine, so an upgraded binary might
not understand an old file. Instead of asking the user to clean caches,
`src/upgrade.rs` runs maintenance on first launch after a version change,
keyed on the delta against the `~/.plank/version` marker:

| Transition | Action |
| --- | --- |
| same version | nothing |
| **patch** bump | nothing; the marker advances |
| **minor** bump | delete `kvcache/sysprompt.kv` (rebuilt on next warm-up) |
| **major** bump, downgrade, or unreadable marker | delete `sysprompt.kv` **and** `image-cache/` |

Session transcripts (`kvcache/*.session`) are user data and are never touched.
Everything removed is rebuilt on demand, so a wrong classification costs one
warm-up, never data.

The table drives release numbering, not the other way around: bump **minor**
whenever the system prompt text or the engine snapshot format changes, and
**major** when older cached state must not be trusted at all (see
`VERSIONING.md` and the roadmap).

## Quick reference

- Prompt text: `src/sysprompt.rs`, `src/resources/tools_prompt_after_edit.txt`
- Session context: `src/context.rs` (`/context` shows the token breakdown)
- Warm-up + checkpoint: `Ds4Engine::warm_system_prompt`,
  `checkpoint_fingerprint`, `save_checkpoint` in `src/ds4engine.rs`
- Invalidation: `src/upgrade.rs`
- Byte-parity enforcement: `tests/c_parity.rs`
  (`PLANK_REGEN_FIXTURES=1 cargo test` to regenerate fixtures)

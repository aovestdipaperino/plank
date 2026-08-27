# Plan: Pi-style `SYSTEM-PROMPT.md` / `SYSTEM-PROMPT-APPEND.md` overrides

Status: **plan / not implemented.** This document proposes how plank should
support the Pi Harness convention of dropping a `SYSTEM-PROMPT.md` (full
replacement) or `SYSTEM-PROMPT-APPEND.md` (additive suffix) next to a project
and having it shape the agent's system prompt.

Read `docs/SYSTEM-PROMPT.md` first: it defines the static/volatile cache
boundary that this feature has to live inside, and most of the design pressure
here comes from that boundary.

## What Pi does, and what it means here

Pi's convention is two optional Markdown files:

- `SYSTEM-PROMPT.md` — its contents *replace* the harness's built-in system
  prompt entirely.
- `SYSTEM-PROMPT-APPEND.md` — its contents are *appended* to whatever prompt is
  otherwise in effect.

Both are plain Markdown, both are read at startup, and appending is the common
case while replacement is the escape hatch for people who want the harness's
loop without its personality.

Mapping that onto plank is not a straight copy, because plank's built-in prompt
is not a personality — it is the parity-constrained DS4 tools prompt
(`build_tools_prompt`), whose bytes the DeepSeek V4 Flash model was trained on.
Deleting it does not make the model terser; it makes DSML tool calls stop
working. So replacement needs a defined scope and a guard rail, covered in §3.

## 1. Where the files are found

Exactly two paths, both in the current working directory:

- `<cwd>/SYSTEM-PROMPT.md`
- `<cwd>/SYSTEM-PROMPT-APPEND.md`

No hierarchy. Deliberately unlike `skills::load_default`, MCP config, and hooks,
which each merge a `~/.plank/` root with a project root — and unlike `AGENTS.md`,
which walks upward to the filesystem root. Those compose because they are
additive registries or documentation. A system prompt is one artifact with one
author: merging two of them produces a prompt nobody wrote, and inheriting one
silently from `~/Code` is a footgun. One file per role, in the directory plank
was started in.

Rules:

- At most one of each. Both may be present together (replacement then append).
- Unreadable or empty files are treated as absent; an unreadable-but-present
  file emits a `stderr` warning through the same path as `hooks.warnings`.

New module `src/sysoverride.rs`:

```rust
pub struct SystemPromptOverride {
    pub replace: Option<OverrideFile>,   // SYSTEM-PROMPT.md
    pub append: Option<OverrideFile>,    // SYSTEM-PROMPT-APPEND.md
}
pub struct OverrideFile { pub path: PathBuf, pub body: String }

pub fn load_default() -> SystemPromptOverride;             // uses current_dir()
pub fn load_from(dir: &Path) -> SystemPromptOverride;      // testable seam
```

`load_from` takes an explicit directory so tests never depend on the real cwd.

## 2. Where the text lands in the prompt

Composition stays in `sysprompt::build_system_prompt`, which today is:

```
build_tools_prompt(mcp, parity)  ++  ("\n\n" ++ cfg.system  if non-empty)
```

Proposed signature change:

```rust
pub fn build_system_prompt(
    user_system: &str,
    mcp_servers: &[McpServer],
    parity: bool,
    over: &SystemPromptOverride,
) -> String
```

Layering, outermost first:

| Layer | Source | Present when |
| --- | --- | --- |
| 1. Tools prompt | `build_tools_prompt` | always, unless replaced (§3) |
| 2. `-sys` / `--system` text | `cfg.system` | non-empty and not the default |
| 3. Append | `SYSTEM-PROMPT-APPEND.md` | file present |

Each layer is joined with a single blank line, exactly like the existing `-sys`
join, so a config with no override files produces **byte-identical** output to
today. That equality is the first test to write.

Precedence note: `-sys` and `SYSTEM-PROMPT-APPEND.md` are not in competition —
both are additive and both survive. `-sys` comes first because it is the more
ephemeral of the two (a flag on one invocation) and the file is the more
considered instruction, so the file gets the last word.

## 3. What replacement actually replaces

`SYSTEM-PROMPT.md` replaces layers 1 and 2 — the whole composed prompt — which
is Pi's semantics. The problem is layer 1.

**Guard rail.** When a replacement file is active and the engine is
parity-constrained (`SystemPrompt::Ds4`), plank must not silently ship a
prompt with no tools block. Two ways out, both supported:

1. **Placeholder.** If the file body contains the line `{{PLANK_TOOLS}}`, that
   line is substituted with `build_tools_prompt(mcp, parity)`. This is the
   recommended form and should be what `/init`-style docs show: authors get to
   put their framing before and after the tool schemas while keeping the
   trained bytes intact.
2. **No placeholder.** The tools prompt is genuinely dropped. plank emits a
   loud one-time startup warning through the status sink:
   `system prompt replaced by <path>; the DS4 tools prompt is not included, so
   tool calls will likely fail — add {{PLANK_TOOLS}} to keep it`. For provider
   engines (`SystemPrompt::Provider`, where tools travel as JSON schemas rather
   than prompt text) this is harmless and the warning is downgraded to a notice.

No hard refusal: an explicit file placed by hand is a decision, and §"Delivering
work" applies — warn, then do what was asked.

**Interaction with the reminder.** `build_system_prompt_reminder` re-injects the
tools prompt mid-session under context pressure. With a replacement in effect it
should re-inject *the composed prompt actually in use*, not the built-in tools
text — otherwise a session drifts back toward the prompt the user replaced.
This means the reminder builder needs the same override argument, and the
`system_prompt_reminder_framing` test grows a replacement case.

## 4. The cache boundary

Override files are **stable across sessions**, so unlike git status or the date
line they are legitimately allowed inside the fingerprinted prefix. The existing
`fingerprinted_prompt_contains_no_volatile_bytes` test keeps holding: the
composed prompt stays a pure function of its inputs.

The real cost is *sharing*. `~/.plank/kvcache/sysprompt.kv` is a single file
keyed by `sha1(model_name \0 system_prompt_text)`. A per-project
`SYSTEM-PROMPT.md` makes that text per-project, so alternating between two
projects with different overrides thrashes the snapshot: each launch misses,
rebuilds, and overwrites the other project's checkpoint.

Options considered:

- **(a) Accept it (recommended for v1).** Correctness is unaffected — the
  fingerprint already prevents restoring a mismatched checkpoint — and the cost
  is one multi-second prefill per project switch, paid only by users who opted
  into per-project prompts. Zero new moving parts.
- **(b) Fingerprint-keyed filenames.** `kvcache/sysprompt-<fp[..8]>.kv`, with a
  cap (say 4) and LRU eviction on write. Removes the thrash, but adds an
  eviction policy and touches `upgrade.rs`, whose minor/major-bump maintenance
  deletes `sysprompt.kv` by exact name and would need a glob.
- **(c) Demote appends to Tier 2.** Put append text in the project-stable KV
  tier (`kvtier::TierKind::ProjectStable`, alongside `AGENTS.md`) instead of the
  system prompt, keeping Tier 1 universal. Attractive for appends, but wrong for
  replacement (which by definition rewrites Tier 1), so it splits the feature
  into two mechanisms with different semantics.

Recommendation: ship (a), and note (b) as the follow-up if project-switching
users complain. Whichever is chosen, `upgrade.rs` needs a look: if (b) lands,
its `sysprompt.kv` deletions become a prefix match.

## 5. Surfacing it to the user

An override that is invisible is a debugging trap. Minimum surface:

- **Startup status line** when anything is active:
  `system prompt: +append (SYSTEM-PROMPT-APPEND.md)` or
  `system prompt: replaced (SYSTEM-PROMPT.md)`.
- **`/context`** (or whichever command currently breaks down token usage) gains
  a row for override bytes, so the tokens are attributable.
- **`--print-system-prompt`**-style debug output, if a flag for dumping the
  composed prompt does not already exist, is the cheapest possible answer to
  "why is it behaving like that". Worth adding with this feature.
- **Trace**: `trace.text("sysprompt-override", …)` with the resolved paths.

Reload semantics: files are read once at startup, like MCP config and hooks.
`/new` resets the session but not the process, so it does **not** re-read; if
that proves annoying, a later `/reload` is the right home for it, not a stat
loop on every turn.

## 6. Implementation order

1. `src/sysoverride.rs` with `load_from` + unit tests over a temp dir: each file
   alone, both together, missing/empty/unreadable.
2. Thread `&SystemPromptOverride` through `build_system_prompt` and
   `build_system_prompt_reminder`; assert byte-identical output for the empty
   override (this is the parity safety net — `tests/c_parity.rs` must stay green
   untouched).
3. `{{PLANK_TOOLS}}` substitution + the replacement path, with tests for
   placeholder present/absent and for provider vs. DS4 warning severity.
4. Wire into `ui.rs::new_agent` (after `mcp::load_and_start`, before the
   `build_system_prompt` call at `src/ui.rs:6631`) and the shared-model path in
   `src/main.rs:502`, which passes `cfg.system` into `Ds4Model::open_shared` and
   must see the same composed text or the two will disagree on the fingerprint.
5. Status line, `/context` row, trace entry, prompt-dump flag.
6. Docs: a short section in `docs/SYSTEM-PROMPT.md` pointing here, plus a
   `FINDINGS.md` entry if the shared-model/agent fingerprint split in step 4
   turns out to have a sharp edge.

## 7. Open questions

- Does a replacement prompt also suppress the session-start `AGENTS.md` context
  injection? Argument for: someone replacing the prompt wants full control.
  Argument against: those are different mechanisms at different transcript
  positions, and conflating them is surprising. Plan assumes **no** — Tier 2/3
  context is untouched.
- Should `SYSTEM-PROMPT.md` support the same `$ARGUMENTS`/template substitution
  the skills loader does? Plan assumes no, beyond `{{PLANK_TOOLS}}`.

## Deviations from the C reference (model-facing text)

These are deliberate, versioned deviations from `refs/ds4`'s prompt/tool text,
each gated behind a setting that defaults to off (or on with a documented
exception) so parity is untouched until a user opts in. Each changes the
system prompt, which changes the `fp1` fingerprint and invalidates the
`sysprompt-<fp1>.kv_raw` snapshots — survivable (they rebuild), but batch such
changes so the fingerprint churns once rather than once per feature.

- **`tools.repeatAdvisory` (M1, default on).** The loop-guard advisory line
  (`[loop guard] you have called this tool with these arguments N times; the
  result has not changed`) is appended to a tool result the model sees. The C
  agent has no such concept.
- **`tools.recall` (M8, default off).** The `recall` tool searches prior
  sessions and the current one's pre-compaction portion, scoped to the current
  project. The C agent has no such tool, so it is advertised in the tools
  prompt only when enabled.
- **`tools.fanout` (M9, default off).** The `fanout` tool runs a list of
  independent subtasks, each delegated to a named sub-agent, and joins their
  reports deterministically. The C agent has no such tool; the description
  deliberately promises a deterministic join, not speed — on the `ds4_engine`
  path subtasks are interleaved on one Metal queue, not parallel.

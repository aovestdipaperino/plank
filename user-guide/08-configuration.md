[← Context](07-context.md) · [Index](README.md) · Next: [Extending plank →](09-extending.md)

# 8. Configuration

## Precedence

Each layer overrides the one before it:

```
built-in defaults → ~/.plank/settings.json → ./.plank/settings.json → environment → command-line flags
```

The rule of thumb: **`settings.json` holds preferences, flags hold per-run choices.**

## `settings.json`

Hierarchical like the MCP config: `~/.plank/settings.json` applies everywhere, `./.plank/settings.json` in the working directory overrides it key by key. Everything is optional — the file need not exist, and any subset of keys works.

```json
{
  "engine": { "model": "~/models/ds4.gguf", "threads": 8,
              "backend": "metal", "power": 80, "ctx": 262144,
              "thinkingToolCalls": true },
  "ui":     { "respectGitignore": true, "popupRows": 15, "indexRefreshSecs": 5,
              "historySize": 512, "showToolCalls": false, "showToolResults": false,
              "showThinking": true, "notifications": "always", "notifyAfterSecs": 10 },
  "safety": { "sandbox": true, "btwSuspend": true },
  "mcp":    { "timeoutSecs": 30 },
  "ask":    { "maxOptions": 7 },
  "update": { "check": true },
  "agents": { "autoRoute": true, "maxParallel": 4 },
  "git":    { "signCommits": true },
  "worktree": { "sparsePaths": ["src", "docs"],
                "symlinkDirectories": ["target"], "isolateAgents": false }
}
```

Edit it in-session with `/config` (an interactive form) or one key at a time:

```
/config ui.showThinking false
```

Changes write `./.plank/settings.json` and apply immediately.

### `engine`

| Key | Default | What |
|---|---|---|
| `model` | `~/.plank/ds4flash.gguf` | model file (`~` expanded). Same as `-m`. |
| `threads` | engine default | worker threads. Same as `-t`. |
| `backend` | platform default | `metal`, `cuda`, or `cpu`. Same as `--backend`. |
| `power` | unset | GPU power cap percent. Same as `--power`. |
| `ctx` | 1048576 | context window in tokens. Same as `-c`. |
| `thinkingToolCalls` | `true` | dispatch tool calls emitted inside the thinking block. `false` for strict ds4 parity. |

### `ui`

| Key | Default | What |
|---|---|---|
| `respectGitignore` | `true` | whether `@` completion honours `.gitignore` for untracked files |
| `popupRows` | 15 | rows the `@` completion popup offers |
| `indexRefreshSecs` | 5 | how long the file index is trusted before a rebuild |
| `historySize` | 512 | prompt history entries retained |
| `showToolCalls` | `false` | show the model's `🛠️` tool-call banners |
| `showToolResults` | `false` | echo tool result text into the scrollback |
| `showThinking` | `true` | render thinking (dimmed) in the scrollback |
| `notifications` | `always` | `always`, `unfocused`, or `never` |
| `notifyAfterSecs` | 10 | minimum turn duration before a completion notification |
| `crtOff` | `true` | CRT power-off animation on clean TUI exit |
| `reducedMotion` | `false` | collapse every animation to a static fallback |
| `screensaver` | `1m` | idle delay before the screensaver: `1m`, `2m`, `5m`, `never` |
| `screensaverFace` | `matrix` | which screen it shows: `matrix`, `starfield`, `minions`, or `random` for a coin flip each time |
| `easterEggs` | `true` | whether the arcade commands exist at all |
| `builtinEditor` | `true` | `Ctrl-G` uses the built-in editor; `false` shells out to `$EDITOR` |

None of `showToolCalls`, `showToolResults`, or `showThinking` change what the model receives — only what you see.

### `safety`

| Key | Default | What |
|---|---|---|
| `sandbox` | on (macOS) | default for the bash write sandbox. Same as `--sandbox` / `--no-sandbox`. |
| `btwSuspend` | `true` | default for `/btw` mid-generation suspend |

### `mcp`, `ask`, `update`, `agents`

| Key | Default | What |
|---|---|---|
| `mcp.timeoutSecs` | 30 | how long an MCP server has to answer before it is considered dead. Raise it for a slow-starting server — one that misses the deadline is dropped along with all its tools. |
| `ask.maxOptions` | 7 | most options the `ask` tool may offer in one question (minimum is fixed at 2) |
| `update.check` | `true` | check GitHub Releases at startup for a newer version |
| `agents.autoRoute` | `true` | let the model pick a subagent definition on its own |
| `agents.maxParallel` | 4 | how many subagents may run at once (capped at 16) |

### `git`

| Key | Default | What |
|---|---|---|
| `signCommits` | `true` | ask the model to end each commit message it writes with a blank line and `--Co-Authored by Plank (https://plank-agent.dev)`. Set it to `false` and commit messages follow your repository's conventions and nothing else. |

### `kvcache`

How long plank keeps the KV snapshots under `~/.plank/kvcache/`, and how much disk they may occupy in total. Ages are measured from a blob's last use, so a checkpoint you keep hitting never expires.

| Key | Default | What |
|---|---|---|
| `ttlSessionDays` | 14 | days a session's KV payload survives after its last use. Expiring one costs a re-prefill of that conversation the next time you resume it, nothing more. |
| `ttlTierDays` | 30 | days a system-prompt or project-context checkpoint survives after its last use. These are the expensive ones to rebuild, so they get the longer window. |
| `maxBytes` | 21474836480 (20 GB) | hard ceiling on the whole cache. Once the age-based pass is done, anything still over budget is evicted least-recently-used first. `0` means no ceiling. |

Three things always survive both passes: anything you have pinned in `/kvcache`, the chain the current launch is actually using, and any checkpoint that something newer still builds on. A budget is a target rather than a licence, so if everything left is protected plank stays over budget instead of deleting something it should not.

### `worktree`

| Key | Default | What |
|---|---|---|
| `sparsePaths` | `[]` (everything) | cone-mode sparse-checkout paths for a new worktree. Worth setting for a repository large enough that a second full checkout is painful. |
| `symlinkDirectories` | `[]` | directories linked from the main checkout instead of duplicated, e.g. `target` or `node_modules`. A name that could climb out of the worktree is ignored. |
| `isolateAgents` | `false` | give every subagent its own throwaway worktree. Off because a checkout per agent costs disk and time, and the work then has to be merged back; turn it on per-definition with `isolation: worktree` instead if only some need it. |

### Two things the file deliberately will not do

- **No secrets.** `./.plank/settings.json` sits inside your working tree and is easy to commit by accident, so there is no API-key setting. Keep keys on `--api-key` or the provider's environment variable.
- **No per-run choices.** `--prompt`, `--non-interactive`, `--ui-remote`, `--trace`, `--chdir`, `--seed`, `--worktree`, and the serve/control options describe one invocation, not a preference, so they have no settings key.

### When it goes wrong

A broken settings file never stops plank from starting. Malformed JSON, a wrongly-typed value, an unknown key, an unrecognised backend name — each falls back to that key's default. (The same bad name passed to `--backend` *is* an error: a flag is an explicit instruction, a config file is a preference.)

Because a settings file can quietly move you off Metal or shrink the context — both of which show up only as "plank got slow" — plank prints one startup line naming what is in force, listing only settings actually in effect. With no settings file, or one that changes nothing, there is no line at all.

One limitation: settings come from the directory plank launches in, so project settings do not follow `--chdir`.

## Command-line flags

`plank --help` prints the full list. The ones you are most likely to want:

### Model and engine

| Flag | What |
|---|---|
| `-m, --model PATH` | load a ds4 GGUF model |
| `-t, --threads N` | worker thread count |
| `--backend NAME` | `metal`, `cuda`, or `cpu` |
| `--metal` / `--cuda` / `--cpu` | the same, as switches |
| `--power N` | GPU power cap percent (1..100) |
| `-c, --ctx N` | context window in tokens |
| `-n, --tokens N` | maximum tokens to generate (default 50000) |
| `--quality` | quality mode |
| `--warm-weights` | touch all weights at load |

### Sampling and reasoning

| Flag | What |
|---|---|
| `--temp F` | temperature (0..100) |
| `--top-p F` | nucleus threshold (0..1) |
| `--min-p F` | minimum-probability threshold (0..1) |
| `--seed N` | RNG seed |
| `--think` / `--think-max` / `--nothink` | reasoning effort |

### Session and mode

| Flag | What |
|---|---|
| `-p, --prompt TEXT` | run one prompt and exit |
| `--non-interactive` | disable the interactive UI |
| `-sys, --system TEXT` | override the system prompt |
| `--chdir PATH` | change working directory before starting |
| `--worktree NAME` | start inside an isolated git worktree of this repository |
| `--worktree-pr N` | base that worktree on pull request N (implies `--worktree`) |
| `--trace PATH` | append a trace log |
| `-h, --help [topic]` | help, optionally on one topic |

### Safety and extensions

| Flag | What |
|---|---|
| `--sandbox` / `--no-sandbox` | bash write sandbox (on by default on macOS) |
| `--disable-btw-suspend` | queue an in-pass `/btw` at the next boundary instead of suspending |
| `--mcp-config FILE` | local MCP config (default `./.mcp.json`) |

### Speculative decoding

| Flag | What |
|---|---|
| `--dspark` | DSpark speculative decoding, on by default |
| `--dspark-off` | disable DSpark speculative decoding (target-only decode) |
| `--dspark-confidence F` | pruning threshold, `0..1` (`0` forces fixed five-token blocks) |
| `--dspark-strict` | load the drafter but keep target-only decode, for comparisons |

DSpark speculative decoding is **on by default**: DeepSeek's auxiliary draft checkpoint for V4 Flash proposes up to five tokens ahead and the main model verifies them, committing only the prefix it agrees with, so one verification pass can advance the stream by several tokens. `--dspark-off` turns it off for target-only decode. The support model (~5.6 GB) needs no flag of its own — it resolves to `~/.plank/ds4flash.dspark.gguf` and is offered for download through the same resumable path as the main model, unless `--mtp` names one.

Verification is argmax, so proposals are only used at `--temp 0`; sampled decoding ignores them. Whether it pays depends on the engine build, the quant and the machine: on an M5 Max it was a 0.71× *slowdown* until the Metal verifier was pipelined upstream, after which the same measurement read 1.19×. The peak rates in the exit message are the way to check on your own hardware.

### Advanced engine tuning

`--mtp PATH`, `--mtp-draft N`, `--mtp-margin F` configure multi-token prediction with a draft model. `--ssd-streaming` and its companions (`--ssd-streaming-cold`, `--ssd-streaming-cache-experts`, `--ssd-streaming-preload-experts`) stream experts from SSD instead of loading them resident, which is how you run a model that does not fit. `--simulate-used-memory <N>GB` pretends memory is already used, for testing those paths. `--dir-steering-file`, `--dir-steering-ffn`, `--dir-steering-attn` apply directional steering vectors.

Remote, shared-engine, control, and provider flags are covered in [Remote and hosted engines](10-remote-and-providers.md).

## Environment variables

| Variable | What |
|---|---|
| `OPENAI_API_KEY` | key for `--provider openai` |
| `ANTHROPIC_API_KEY` | key for `--provider anthropic` |
| `PLANK_REMOTE_TOKEN` | bearer token for `--remote`, `--control`, and `plank remote` |
| `EDITOR` | editor for `Ctrl-G` when `ui.builtinEditor` is `false` |

---

Next: [Extending plank →](09-extending.md)

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
  "update": { "check": true }
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
| `screensaver` | `1m` | idle delay before the starfield: `1m`, `2m`, `5m`, `never` |
| `easterEggs` | `true` | whether the arcade commands exist at all |
| `builtinEditor` | `true` | `Ctrl-G` uses the built-in editor; `false` shells out to `$EDITOR` |

None of `showToolCalls`, `showToolResults`, or `showThinking` change what the model receives — only what you see.

### `safety`

| Key | Default | What |
|---|---|---|
| `sandbox` | on (macOS) | default for the bash write sandbox. Same as `--sandbox` / `--no-sandbox`. |
| `btwSuspend` | `true` | default for `/btw` mid-generation suspend |

### `mcp`, `ask`, `update`

| Key | Default | What |
|---|---|---|
| `mcp.timeoutSecs` | 30 | how long an MCP server has to answer before it is considered dead. Raise it for a slow-starting server — one that misses the deadline is dropped along with all its tools. |
| `ask.maxOptions` | 7 | most options the `ask` tool may offer in one question (minimum is fixed at 2) |
| `update.check` | `true` | check GitHub Releases at startup for a newer version |

### Two things the file deliberately will not do

- **No secrets.** `./.plank/settings.json` sits inside your working tree and is easy to commit by accident, so there is no API-key setting. Keep keys on `--api-key` or the provider's environment variable.
- **No per-run choices.** `--prompt`, `--non-interactive`, `--ui-remote`, `--trace`, `--chdir`, `--seed`, and the serve/control options describe one invocation, not a preference, so they have no settings key.

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
| `--trace PATH` | append a trace log |
| `-h, --help [topic]` | help, optionally on one topic |

### Safety and extensions

| Flag | What |
|---|---|
| `--sandbox` / `--no-sandbox` | bash write sandbox (on by default on macOS) |
| `--disable-btw-suspend` | queue an in-pass `/btw` at the next boundary instead of suspending |
| `--mcp-config FILE` | local MCP config (default `./.mcp.json`) |

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

[← Index](README.md) · Next: [Getting started →](02-getting-started.md)

# 1. Installation

## Homebrew

Homebrew is the only distribution channel — plank is not on crates.io.

```sh
brew install aovestdipaperino/tap/plank-agent        # stable
brew install aovestdipaperino/tap/plank-agent-beta   # beta
```

The formula is named `plank-agent` because a `plank` formula already exists upstream and the bare name collides. **The installed binary is still just `plank`** — only the install name carries the suffix.

Prebuilt bottles exist for Apple Silicon and Intel Macs. On anything else Homebrew builds from source, which needs a Rust toolchain.

```sh
brew upgrade plank-agent
```

## Stable and beta channels

The patch number *is* the channel:

- `vX.Y.0` — stable
- `vX.Y.1`, `vX.Y.2`, … — beta (the version banner shows ` BETA`)

A series opens with its stable `.0` and accumulates beta work as patch bumps. Promoting a beta to stable opens the next minor as a matching stable/beta pair. See [`VERSIONING.md`](../VERSIONING.md) for the full model.

Both formulas install a `plank` binary, so they conflict. Switch channels by uninstalling first:

```sh
brew uninstall plank-agent && brew install plank-agent-beta
```

plank checks GitHub Releases once a day at startup and mentions a newer version if there is one. It is best-effort and silent on failure; turn it off with `update.check: false` in [`settings.json`](08-configuration.md).

## Building from source

Requires macOS with the Xcode command line tools. Clone **with the submodule** — that is where the inference engine lives:

```sh
git clone --recurse-submodules https://github.com/aovestdipaperino/plank
cd plank
cargo build --release
```

- **With `refs/ds4` present** — `build.rs` compiles `libds4core.a` from the Metal-backend objects, links Foundation and Metal, and enables real inference.
- **Without it** — plank still builds and runs, but only against the echo stub. Fine for working on the UI and tools; useless for actual generation.

## Getting the model

Real inference needs the DeepSeek V4 Flash GGUF. On first run, with no `-m` flag and nothing at the default path (`~/.plank/ds4flash.gguf`), plank offers to fetch the quantized model (~87 GB) from Hugging Face. One keypress and it downloads in place with live progress.

Things worth knowing before you start an 87 GB transfer:

- **It resumes.** The download streams to a `.part` file beside the destination. Ctrl-C it, lose your network, close the laptop — the next launch picks up where it stopped.
- **It is guarded.** The default quant needs roughly 82 GB resident, so plank refuses to download or load on machines with less than 96 GB of RAM. You find out before the transfer, not after.
- **It is honest about the wait.** Size and rate counters, plus a rotation of two hundred status messages.
- **It is headless-safe.** With stdin not on a terminal there is nobody to answer the prompt, so plank exits with instructions rather than hanging your script.

The DSpark draft checkpoint (~5.6 GB) follows the same path — DSpark is on by default, so it resolves to `~/.plank/ds4flash.dspark.gguf` and is offered for download with the same prompt, resume and progress (`--dspark-off` skips it). See [Configuration](08-configuration.md#speculative-decoding).

## Staying on the current model

Once a model is installed, plank checks for a newer one at most once a day by fetching `ds4.manifest`, a small file that names the current main, vision and DSpark artifacts and a version number. When a newer version appears, plank starts the download in a detached background process rather than blocking the session: it keeps running even if you quit plank or close the terminal, and it resumes automatically from `~/.plank/staging/` if it was interrupted, so a dropped connection or a closed laptop lid costs nothing but time. Only one such download runs per machine, whichever plank noticed the update first.

While a download is live, a status segment shows its progress, for example `⇩ model 2/3 41% 12MB/s`. In the TUI, Alt-M opens a prompt to cancel it, either keeping the partial files so the next launch resumes them or deleting them outright. From either the TUI or the plain REPL, `/model` (or `/model status`) reports what is happening, `/model cancel` stops it (add `--delete` to also remove the partial files), and `/model download` starts one by hand.

New artifacts are verified by SHA-256 as they stream in, but they are not installed the moment the download finishes — a running plank has the current model mapped into memory, so the swap happens at the next launch instead. Expect a fresh model to be in effect the next time you start plank, not mid-session.

To use a model you already have somewhere else:

```sh
plank -m ~/models/ds4flash.gguf
```

or set it permanently with `engine.model` in `settings.json`.

## No model, no problem (sort of)

Without a model file plank runs against a built-in echo engine. Every command, tool, session feature and UI element works; the "model" just echoes. This is how the test suite runs and how UI work gets done, and it is what you will see if you launch on an unsupported platform.

## Where plank keeps its files

| Path | What |
|---|---|
| `~/.plank/ds4flash.gguf` | default model location |
| `~/.plank/ds4flash.dspark.gguf` | DSpark draft model, when `--dspark` is used |
| `~/.plank/kvcache/` | saved sessions (`<name>.kv`) plus the KV snapshots (`*.kv_raw`) and their metadata (`*.json`). Browse it with `/kvcache`. |
| `~/.plank/settings.json` | global preferences |
| `~/.plank/.mcp.json` | global MCP server config |
| `~/.plank/hooks.json` | global hooks |
| `~/.plank/sandbox.json` | global sandbox policy |
| `~/.plank/skills/`, `templates/`, `agents/` | global extensions |
| `~/.plank/MEMORY.md` | user-scope memory |
| `~/.plank/repro/` | `/repro` dumps |
| `~/.plank/usage-data/` | `/insights` reports |
| `~/.plank/doc-cache/` | PDFs converted to Markdown |
| `~/.plank/image-cache/` | pasted images, deduplicated |
| `~/.plank/mcp-advert/` | last-known-good MCP tool advertisements |
| `~/.plank/errors.log` | full detail behind terse tool errors |
| `~/.plank/tool-call-errors.log` | malformed tool calls the model emitted |
| `./.plank/` | the same set, project-scoped, overriding the global one |

The two logs are the first place to look when something failed and the on-screen message was too terse to act on. See [Troubleshooting](13-troubleshooting.md).

---

Next: [Getting started →](02-getting-started.md)

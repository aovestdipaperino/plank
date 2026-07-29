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
| `~/.plank/kvcache/` | saved sessions (`<name>.kv`) and their KV sidecars |
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

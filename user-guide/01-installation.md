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

plank knows its models as named *engines*: a main model plus the companions it runs with, a drafter for speculative decoding (`mtp`) and a vision encoder (`vision`). They are listed in `engines.json`, a small catalog that ships inside plank and is refreshed from the repository at most once a day. Three are built in:

| Engine | What |
|---|---|
| `ds4vision` | DeepSeek V4 Flash with its DSpark drafter and vision encoder. The default. |
| `ds41` | DeepSeek V4.1 Flash with its vision encoder. |
| `qwen` | Qwen3.8-Flash-Next. |

Pick one with `--model <name>`, for example `plank --model ds41`. Real inference with no choice at all runs the default engine: on first run, with nothing at its path (`~/.plank/ds4vision.gguf`), plank offers to fetch the quantized model (~87 GB) from Hugging Face. One keypress and it downloads in place with live progress, and an interruption resumes where it stopped rather than starting over. Any other engine is offered the same way the first time you pick it.

DeepSeek V4.1 Flash is a family of its own, with its own tool-call dialect and its own `.ds41.kv` transcripts. `--model ds41` fetches and runs it; pointing `--model` at a V4.1 GGUF path works too, and the family and dialect follow from the file's own architecture field. It is large enough that plank will usually turn on SSD streaming for you.

Things worth knowing before you start an 87 GB transfer:

- **It resumes.** The download streams to a `.part` file beside the destination. Ctrl-C it, lose your network, close the laptop — the next launch picks up where it stopped.
- **It is guarded.** The default quant needs roughly 82 GB resident, so plank refuses to download or load on machines with less than 96 GB of RAM. You find out before the transfer, not after.
- **It is honest about the wait.** Size and rate counters, plus a rotation of two hundred status messages.
- **It is headless-safe.** With stdin not on a terminal there is nobody to answer the prompt, so plank exits with instructions rather than hanging your script.

The DSpark draft checkpoint (~5.6 GB) follows the same path. Speculative decoding is on by default, so for `ds4vision` it resolves to `~/.plank/ds4vision.mtp.gguf` and is offered for download with the same prompt, resume and progress (`--mtp-off` skips it). See [Configuration](08-configuration.md#speculative-decoding).

## Staying on the current model

Once a model is installed, plank checks for a newer one at most once a day by fetching `engines.json`, where each engine carries its own version number and the artifacts that make it up. When a newer version of the engine you run appears, plank asks first: `Download it in the background? [y/N]`, defaulting to no. Say yes and it starts in a detached background process rather than blocking the session: it keeps running even if you quit plank or close the terminal, so closing the laptop lid mid-transfer costs nothing but time. Say no (or just press Enter) and nothing downloads yet; run `/model download` whenever you are ready to start it. A dropped connection or a stopped helper leaves verified artifacts and partial files in `~/.plank/staging/<engine>/`; nothing resumes it automatically, but accepting the next daily offer or running `/model download` picks up right where it left off, re-downloading only what wasn't finished. Only one such download runs per machine, whichever plank noticed the update first.

While a download is live, a status segment shows its progress, for example `⇩ model 2/3 41% 12MB/s`. In the TUI, Alt-M opens a prompt to cancel it: keep the partial files (resume with `/model download`, or the next daily offer) or delete them outright. From either the TUI or the plain REPL, `/model` (or `/model status`) reports what is happening, `/model cancel` stops it (add `--delete` to also remove the partial files), and `/model download` starts one by hand.

New artifacts are verified by SHA-256 as they stream in, but they are not installed the moment the download finishes — a running plank has the current model mapped into memory, so the swap happens at the next launch instead. Expect a fresh model to be in effect the next time you start plank, not mid-session.

To use a model you already have somewhere else, give `--model` (or `-m`) a path instead of a name:

```sh
plank --model ~/models/my-ds4.gguf
```

A bare path loads that file and nothing else. plank never downloads or upgrades it, and looks for no drafter or encoder beside it, so a DeepSeek model loaded this way runs without speculative decoding unless you name a drafter with `--mtp-model`. There are two exceptions: a path to a managed engine's main file under any other name (a symlink or hard link included) selects that engine with all its companions, and a `.ggd` delta built on a managed engine's main inherits that engine's companions while the patched clone itself is never upgraded. `--model:<name>` is the strict form: it must name an engine and never falls back to a path, which is what a script wants. Set either permanently with `engine.model` in `settings.json`.

To keep a model of your own together with its companions, describe it as a local engine in `~/.plank/engines.local.json`. That file is layered over the catalog, is the only place a role may name a `path`, and can also change the default:

```json
{
  "default": "mine",
  "engines": {
    "mine": {
      "main":   { "path": "~/models/my-ds4.gguf", "url": "https://huggingface.co/owner/repo/blob/main/my-ds4.gguf" },
      "mtp":    { "path": "~/models/my-ds4.dspark.gguf" },
      "vision": { "path": "~/models/my-ds4.vision.gguf" }
    }
  }
}
```

`plank --model mine` then loads all three, and with `"default": "mine"` so does a plain `plank`. A local entry with the same name as a built-in engine replaces it. A role that names a `path` may also give a `url` (an `https://` link, where a Hugging Face `/blob/` page is turned into its `/resolve/` download): when the file is missing, plank offers to download it into exactly that path, but it still never checks it for upgrades.

Upgrading from a release before engines renames the files under `~/.plank` once, at the first launch: `ds4flash.gguf` becomes `ds4vision.gguf`, its drafter and encoder become `ds4vision.mtp.gguf` and `ds4vision.vision.gguf`, and the V4.1 files become `ds41.*`. Nothing is downloaded again, but it is one way: an older plank sharing the same `~/.plank` no longer finds its model and offers to fetch it.

## No model, no problem (sort of)

Without a model file plank runs against a built-in echo engine. Every command, tool, session feature and UI element works; the "model" just echoes. This is how the test suite runs and how UI work gets done, and it is what you will see if you launch on an unsupported platform.

## Where plank keeps its files

| Path | What |
|---|---|
| `~/.plank/<engine>.gguf` | an engine's main model, for example `ds4vision.gguf` (the default), `ds41.gguf` or `qwen.gguf` |
| `~/.plank/<engine>.mtp.gguf` | an engine's drafter, for speculation (`--mtp`): `ds4vision.mtp.gguf` is DeepSeek V4's DSpark model |
| `~/.plank/<engine>.vision.gguf` | an engine's vision encoder. `qwen.vision.gguf` is installed with the engine, but plank does not load one for Qwen yet |
| `~/.plank/engines/` | one `<engine>.installed.json` per installed engine, recording which version is on disk |
| `~/.plank/engines.local.json` | your own engines and default, layered over the catalog |
| `~/.plank/staging/<engine>/` | background downloads in progress, installed at the next launch |
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

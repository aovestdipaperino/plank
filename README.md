# plank

<p align="center">
  <img src="assets/logo.png" alt="Plank logo" width="300">
</p>

A Rust port of the [ds4](https://github.com/aovestdipaperino/ds4) agent, converted functionality-by-functionality (not line-by-line) from the C reference implementation.

Plank is an interactive coding agent with a terminal REPL, a one-shot headless mode, and a set of built-in tools (shell, file read/edit, web).

> **macOS only.** Plank targets macOS exclusively: inference uses the original ds4 C engine with the Metal backend, linked via the `ds4-ref` submodule. Other platforms are not supported.

## Building

Requires macOS (Apple Silicon or Intel) with the Xcode command line tools. Clone with the submodule to get the ds4 engine:

```sh
git clone --recurse-submodules https://github.com/aovestdipaperino/plank
cd plank
cargo build --release
```

- **With `ds4-ref` present:** `build.rs` builds `libds4core.a` from the Metal-backend objects and links the required frameworks, enabling the `ds4_engine` cfg.
- **Missing submodule:** plank still builds, but without the native engine it uses the echo engine only (useful for development/CI).

You will also need a GGUF model file (e.g. `ds4flash.gguf`) for real inference; see the `download_model.sh` script in `ds4-ref`.

## Usage

```sh
plank            # interactive REPL
plank --help     # full option list
```

Run with a prompt argument for one-shot headless mode.

## Project layout

Each module in `src/` maps to one functional section of the original `ds4_agent.c`:

- `engine.rs` / `ds4engine.rs` / `ffi.rs` — inference engine abstraction and native ds4 bindings
- `session.rs`, `compact.rs`, `sysprompt.rs` — conversation state, compaction, system prompt
- `tools/` — built-in agent tools (bash, edit, files, web)
- `ui.rs`, `render.rs`, `statusbar.rs`, `editor.rs`, `viz.rs` — terminal UI
- `config.rs`, `trace.rs`, `interrupt.rs`, `status.rs` — configuration, tracing, signal handling

## License

[MIT](LICENSE)

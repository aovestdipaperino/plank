# Platforms

plank started on macOS and now builds and runs on Linux and Windows too
([#89](https://github.com/aovestdipaperino/plank/issues/89)). This page says
what works where, what each OS needs to build, and what has not been run yet.

"Untested" below means the code is written and compiles but has not run on the
hardware it needs; it is not a promise that it works.

## Support matrix

| | macOS | Linux | Windows |
|---|---|---|---|
| TUI, plain REPL, headless (`--ui console`, `-p`) | yes | yes | yes |
| `--provider openai\|anthropic`, `--remote`, `/rc` | yes | yes | yes |
| DeepSeek / Qwen (ds4 C engine) | Metal | CPU by default; CUDA or ROCm with `PLANK_DS4_BACKEND` (untested) | no native build; [use WSL2](#deepseek-and-qwen-on-windows-wsl2) |
| ds4 multi-GPU placement (`--gpu-vram`, `--gpu-devices`) | no | CUDA and ROCm builds (untested) | no |
| ds4 vision encoder, SSD streaming | Metal | GPU backends only; the CPU backend refuses both, so plank skips them | no |
| Gemma 4 | Metal or CPU | CPU; CUDA in a `--features cuda` build (untested) | CPU; CUDA as on Linux (untested) |
| Flint | Metal or CPU | CPU (no CUDA kernels yet) | CPU (no CUDA kernels yet) |
| GPU yield (`PLANK_GPU_YIELD`) | Metal | CUDA and ROCm (untested) | no local GPU engine |
| `bash` tool | `/bin/sh` | `/bin/sh` | a POSIX shell: `PLANK_SHELL`, else `sh.exe`/`bash.exe` on `PATH`, else Git for Windows |
| `bash` write sandbox | Seatbelt | Landlock (kernel 5.13+) | Low integrity token, write roots labelled Low (`sh`/`bash`; see FINDINGS.md) |
| `write`/`edit` workspace containment | yes | yes | yes |
| `read` on a PDF (`docparse`) | yes | yes | yes |
| Web tools (`use_obscura`) | yes | builds; not yet exercised | builds; not yet exercised |
| Desktop notifications | `notify-rust`, terminal's icon | `notify-send`, terminal's `.desktop` entry | PowerShell toast, under Windows Terminal's app id when inside it |
| Clipboard text and images, open, sound | yes | `wl-*`/`xclip`/`xsel`, `xdg-open`, `paplay`/`aplay`/`ffplay` | PowerShell, `explorer` |
| `--debug` console mirror | yes | yes | yes (same protocol over `std`) |
| WASM plugins | yes | yes | yes; building the guests needs a POSIX shell (Git Bash) |
| Prebuilt releases | Homebrew bottle, `.tar.gz` | `.tar.gz` (x86_64, aarch64) | `.zip` (x86_64) |

The release archives for Linux and Windows are added by
[#95](https://github.com/aovestdipaperino/plank/pull/95) and are built without
the ds4 engine (`PLANK_NO_DS4=1`); build from source for DeepSeek or Qwen on
Linux.

## Building

Every OS needs the Rust toolchain pinned in `rust-toolchain.toml`, a C and C++
compiler, and `cmake` on `PATH` (the `docparse` feature compiles libtesseract).
Clone with `--recurse-submodules`.

- **macOS:** the Xcode command line tools. With `refs/ds4` present the build
  compiles the Metal engine.
- **Linux:** `build-essential` (or your distribution's equivalent), `make` and
  `cmake`. With `refs/ds4` present the build compiles ds4's CPU backend. For a
  GPU, set `PLANK_DS4_BACKEND` before building:
  - `cuda`: the NVIDIA driver and CUDA toolkit (`nvcc`, cuBLAS). `CUDA_HOME`
    (default `/usr/local/cuda`) and `CUDA_ARCH` (default `native`) are passed
    to the ds4 Makefile.
  - `rocm`: ROCm with `hipcc`, hipBLAS, hipBLASLt and rocBLAS. `ROCM_PATH`
    (default `/opt/rocm`), `ROCM_ARCH` (the Makefile's default is `gfx1151`,
    Strix Halo) and `HIPCC` are honoured.

  Gemma on an NVIDIA GPU is a separate switch: `cargo build --features cuda`.
- **Windows:** the MSVC Rust toolchain and the Visual Studio Build Tools with
  the C++ workload, whose bundled `cmake` works. Git for Windows supplies the
  POSIX shell the `bash` tool runs. The ds4 engine is not built on Windows.

`PLANK_NO_DS4=1` skips the ds4 engine on any OS.

### `cargo install`

plank is not on crates.io, but `cargo install` builds it from the repository.
Cargo checks out the submodules of a git source, `refs/ds4` included, so this
builds the ds4 engine as a checkout would: Metal on macOS, the CPU on Linux,
and nothing on Windows.

```sh
cargo install --locked --git https://github.com/aovestdipaperino/plank plank
# ds4 on an NVIDIA GPU (Linux):
PLANK_DS4_BACKEND=cuda cargo install --locked --git https://github.com/aovestdipaperino/plank plank
# Gemma on an NVIDIA GPU (Linux, Windows):
cargo install --locked --git https://github.com/aovestdipaperino/plank plank --features cuda
# Without the ds4 engine (much faster to build):
PLANK_NO_DS4=1 cargo install --locked --git https://github.com/aovestdipaperino/plank plank
```

## Memory

DeepSeek V4 Flash needs 96 GB. plank checks that floor before downloading or
loading anything, against host RAM or, on a CUDA or ROCm build, against the
GPUs' summed VRAM (`nvidia-smi`, `rocm-smi`). When RAM is short but the GPUs
hold the model, plank places its layers across them as `--gpu-vram auto` does,
and the engine menu offers it. On unified memory (DGX Spark, Strix Halo) the
RAM figure already covers what the GPU can use.

## DeepSeek and Qwen on Windows: WSL2

There is no native Windows build of the ds4 engine: its build and parts of its
runtime (`ds4_ssd.c`, `ds4_distributed.c`, `ds4_web.c`) are POSIX. Run plank in
WSL2 instead, built as on Linux:

1. Install a WSL2 distribution (`wsl --install -d Ubuntu`) and, inside it, the
   Linux build requirements above.
2. For an NVIDIA GPU, install the Windows NVIDIA driver (it exposes CUDA to
   WSL2) and, inside WSL2, NVIDIA's CUDA toolkit for WSL. Do not install a
   Linux display driver inside WSL2. Then build with `PLANK_DS4_BACKEND=cuda`.
3. WSL2 gives the VM half of the host's RAM by default, which can put a
   128 GB machine under the 96 GB floor. Raise it in `%UserProfile%\.wslconfig`:

   ```ini
   [wsl2]
   memory=110GB
   ```

   and run `wsl --shutdown` for it to take effect.
4. Keep models and the checkout on the WSL2 filesystem (`~/`), not under
   `/mnt/c`: the engine memory-maps an ~87 GB file, and the Windows drive is
   much slower through WSL2.

A native port would start upstream in `ds4`: [docs/DS4-WINDOWS-PORT.md](DS4-WINDOWS-PORT.md) assesses what it would take (feasible, no blocker; a POSIX compatibility header, stubs for the multi-host code plank does not use, then the CPU core with `clang-cl` and CUDA with `nvcc` and MSVC).

## Not done yet

- **Flint on CUDA.** Its NVFP4 matmul and MoE kernels exist for Metal and the
  CPU only; `--cuda` refuses Flint until they are ported.
- **Confining the PowerShell dialect on Windows.** The `bash` write sandbox
  runs `sh`/`bash` at Low integrity, but Windows PowerShell 5.1 crashes at that
  level (a .NET fail-fast), so `tools.shell: powershell` cannot be confined the
  same way. FINDINGS.md has the details.
- **Terminal testing.** Windows Terminal, conhost, and the common Linux
  terminals (GNOME Terminal, Konsole, kitty, Alacritty, WezTerm, tmux) still
  need checking for key events, focus events, the OSC title, true colour and
  block-glyph width.
- **Package managers.** Homebrew on Linux, Scoop and winget manifests.

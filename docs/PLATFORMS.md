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
| WASM plugins | yes | yes | yes; the guest scripts run under Git Bash (CI builds, packages and rebuild-verifies them), but the modules embed `\`-joined paths, so their hashes match only other Windows builds |
| Prebuilt releases | Homebrew bottle, `.tar.gz` | `.tar.gz` (x86_64, aarch64) | `.zip` (x86_64) |

The release archives for Linux and Windows are added by
[#95](https://github.com/aovestdipaperino/plank/pull/95) and are built without
the ds4 engine (`PLANK_NO_DS4=1`); build from source for DeepSeek or Qwen on
Linux, linking a [pre-built ds4 core](#pre-built-ds4-cores) if you cannot
compile the C engine.

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

### Pre-built ds4 cores

The ds4 C engine is the only part of plank that needs `make`, a C compiler for
its kernels, `nvcc` or `hipcc`; Gemma and Flint are Rust and build with cargo
alone. A machine that cannot compile the C engine can link a pre-built core
instead, by setting `PLANK_DS4_PREBUILT` before building. Building from source
stays the default and needs nothing from this section.

```sh
# The core for this target from the release of the ds4 commit refs/ds4 pins:
PLANK_DS4_PREBUILT=github cargo build --release
# An NVIDIA GPU: the CUDA core for the GPU nvidia-smi reports
PLANK_DS4_PREBUILT=github PLANK_DS4_BACKEND=cuda cargo build --release
# A package you already have, as a directory or a .tar.gz:
PLANK_DS4_PREBUILT=~/Downloads/ds4core-x86_64-unknown-linux-gnu-cpu.tar.gz cargo build --release
# Any other URL needs its SHA-256:
PLANK_DS4_PREBUILT=https://example.com/ds4core.tar.gz PLANK_DS4_PREBUILT_SHA256=<hex> cargo build --release
```

The `ds4 core` workflow (`.github/workflows/ds4core.yml`) publishes the
packages to a prerelease named after the ds4 commit, `ds4core-<first 12 hex>`,
each time `refs/ds4` moves. That commit names the release, not plank's version,
because it decides whether a core fits plank's FFI:

| Package | Host code | For |
|---|---|---|
| `aarch64-apple-darwin-metal` | Apple M1 | every Apple Silicon Mac |
| `x86_64-unknown-linux-gnu-cpu` | x86-64-v3 (AVX2) | Haswell, Zen and later |
| `aarch64-unknown-linux-gnu-cpu` | ARMv8.2-A + fp16 + dotprod | Graviton2+, Grace, GB10, Ampere |
| `x86_64-unknown-linux-gnu-cuda-sm_80` / `sm_86` / `sm_89` / `sm_90` / `sm_120` | x86-64-v3 | A100; RTX 30; RTX 40, L40; H100; RTX 50 |
| `aarch64-unknown-linux-gnu-cuda-sm_90` / `sm_121` | ARMv8.2-A | GH200; DGX Spark |
| `x86_64-unknown-linux-gnu-rocm-gfx1151` | x86-64-v3 | Strix Halo |

With `PLANK_DS4_BACKEND=cuda`, plank picks the variant from the GPU's compute
capability. It takes the highest variant of the same major version that does
not exceed the GPU's, because a cubin runs on later minors of its major. A GPU
with a newer major than any variant (B200, compute 10.0) gets `sm_90`, whose
embedded PTX the driver compiles for it at first load. `PLANK_DS4_PREBUILT_ARCH`
names the variant outright, and is required on a machine without `nvidia-smi`.
For ROCm it defaults to `ROCM_ARCH`, else `gfx1151`.

**What is checked before anything links.** A package downloaded from a release
must match that release's `ds4core-<hex>-SHA256SUMS`, and a URL must match
`PLANK_DS4_PREBUILT_SHA256`. The SHA-256 of `libds4core.a` must match the hash
in the package's `ds4core.json`. The manifest must also name:
- this build's target triple,
- the backend `PLANK_DS4_BACKEND` asks for (always Metal on macOS),
- the ds4 commit `refs/ds4` pins.

The commit check matters most. plank's `ffi.rs` mirrors the C structs of the
pinned `ds4.h`, so a core from another commit can corrupt memory rather than
fail cleanly. The build refuses one unless `PLANK_DS4_PREBUILT_ALLOW_MISMATCH=1`
is set. The pin is read from the git index (`git ls-files --stage refs/ds4`), so
the submodule does not need to be checked out. A source tree without git must
set `DS4_COMMIT`. The package also carries the `ds4.h` it was compiled against,
so you can diff it against `refs/ds4/ds4.h`.

**Provenance.** Each package has a GitHub build provenance attestation tying it
to the workflow run that built it:

```sh
gh attestation verify ds4core-x86_64-unknown-linux-gnu-cpu.tar.gz --repo aovestdipaperino/plank
```

**Runtime dependencies.** The cores are static archives, so plank carries the
engine itself. What remains is what a source build links as well:
- **CPU:** libm only.
- **CUDA:** a CUDA 12 runtime, libcudart and libcublas, at link time and at run
  time (`CUDA_HOME`, default `/usr/local/cuda`), plus an NVIDIA driver that
  supports CUDA 12.9 or later. The packages are built with CUDA 12.9, and the
  build warns when `CUDA_HOME` holds another major version.
- **ROCm:** ROCm 6.4 libraries (`amdhip64`, `hipblas`, `hipblaslt`, `rocblas`)
  under `ROCM_PATH`.
- **Metal:** the kernel sources ship in the package's `metal/`, and the build
  points `DS4_METAL_DIR` at its copy in `target/`. A binary moved to another
  machine looks in `../share/plank/metal`, as a Homebrew bottle does.

The Linux packages are built on Ubuntu 22.04 (glibc 2.35) and link on it and
anything newer. There are no Windows packages, because the C engine has no
Windows build ([WSL2](#deepseek-and-qwen-on-windows-wsl2) runs the Linux ones).
Gemma on CUDA (`--features cuda`) still compiles candle's kernels with `nvcc`;
no pre-built form of it exists.

**Making a package.** `PLANK_DS4_EXPORT=<dir>` makes a source build also write
its package to `<dir>`: `libds4core.a`, `ds4core.json`, `ds4.h` and, for
Metal, `metal/`. Set `NATIVE_CPU_FLAG` to a portable baseline, because the
default `native` binds the core to the build machine's CPU, and the manifest
records it. For CUDA, set `CUDA_ARCH` to an `sm_*`. Archive the directory's
contents with `tar czf <name>.tar.gz -C <dir> .`. CI exports a CPU core this way
on every push, links it back with the ds4 sources out of reach, and checks that
a tampered library is refused.

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
# Linking a pre-built ds4 core instead of compiling it:
PLANK_DS4_PREBUILT=github PLANK_DS4_BACKEND=cuda cargo install --locked --git https://github.com/aovestdipaperino/plank plank
```

If the build reports that it cannot read the `refs/ds4` pin in cargo's checkout,
set `DS4_COMMIT` to the commit `git ls-tree <tag> refs/ds4` shows for the
revision being installed.

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

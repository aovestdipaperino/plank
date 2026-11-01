# A native Windows port of the ds4 engine: assessment

Step 5 of [#89](https://github.com/aovestdipaperino/plank/issues/89) asks
whether the ds4 C engine (DeepSeek, Qwen, GLM) can be built natively on Windows
instead of only under WSL2, and what that would take upstream in `ds4` before
plank could link it. This is that assessment. It was made against the
`refs/ds4` checkout plank pins; no code was changed.

**Verdict:** feasible, with no blocker that rules it out, but it is a real
upstream project, not a build-flag change. Most of the work is a POSIX
compatibility layer and stubbing the multi-host code plank does not use. The
recommended order is the CPU core with `clang-cl` first, then CUDA with `nvcc`
and MSVC.

## What is in the way

The core library plank links (`libds4core.a`) is these sources. None of them
mentions `_WIN32` today, so upstream has never been prepared for Windows.

| Source | Lines | POSIX it uses | Needed by plank on Windows |
|---|---|---|---|
| `ds4.c` | 87,640 | `mmap`/`munmap`/`mlock`/`madvise` (8), `pread` (3), `open`/`fstat`/`fcntl` (25), pthreads (78), `flock`, `sysconf`, `/proc/meminfo` (`ds4_linux_memory.h`). Apple-only code (GCD, `sysctl`, Metal) is already behind `__APPLE__` (222 guards). | yes |
| `ds4_cuda.cu` | 36,309 | `unistd.h`/`fcntl.h`/pthread (prefetch thread), `fstat`, `/proc/self/fd` (behind `__linux__`) | yes, for CUDA |
| `ds4_image.c`, `ds4_layer_pack.c`, `ds4_gpu_args.c` | 1,538 | none | yes |
| `ds4_ssd.c` | 215 | anonymous `mmap` + `mlock` (`--simulate-used-memory`, the memory lock) | yes (small) |
| `ds4_engram.c` | 473 | pthreads (37), `pread`, `open`/`fstat` | yes (V4.1) |
| `ds4_distributed.c` | 8,437 | sockets (42), `poll`, `signal`, pthreads (205), `getpid` | no: multi-host only |
| `ds4_tp.c` | 3,944 | sockets, `poll`, pthreads, `dlopen` of `libibverbs` (RDMA) | no: tensor parallelism across hosts |
| `ds4_web.c` | 1,454 | `fork`/`exec`/`waitpid` (5), sockets, `poll` | no: plank's default web tools use `obscura` |

So the three files the issue named split two ways. `ds4_ssd.c` needs a small
port. `ds4_distributed.c` and `ds4_web.c` are not needed at all. But
`ds4.c` calls 52 distinct `ds4_dist_*`/`ds4_tp_*` functions (111 call
sites), so the multi-host files cannot simply be left out of the link; they
need a stub build, as `DS4_NO_GPU` already provides for the GPU entry points.

### The four things that are more than a header shim

1. **Partial unmapping.** `ds4.c` maps the whole GGUF, parses it, then
   `munmap`s the disk-only tail (V4.1's Engram tables, Qwen's n-gram table) so
   warming never faults it in. `MapViewOfFile` views cannot be partly
   unmapped. The Windows path has to unmap the view and map `[0, start)`
   again before anything holds a pointer into the mapping. Two sites
   (`model_unmap_engram`, `model_unmap_qwen_ngrams`).
2. **Locking model memory.** `mlock` maps to `VirtualLock`, which is capped by
   the process working-set quota, so locking tens of GB also needs
   `SetProcessWorkingSetSizeEx`, and may need the "Lock pages in memory" right
   for very large spans. `madvise(MADV_DONTNEED)` on a locked span becomes
   `VirtualUnlock`, which is what the code wants there.
3. **Threads.** About 360 pthread call sites across `ds4.c`, `ds4_engram.c`
   and the CUDA prefetch thread. They use mutexes, condition variables,
   create and join, which a thin header maps onto `SRWLOCK`,
   `CONDITION_VARIABLE` and `_beginthreadex` (a few hundred lines). Pulling
   in pthreads4w instead is the quicker route but adds a dependency.
4. **The CUDA host compiler.** On Windows `nvcc` accepts only MSVC `cl.exe`
   as its host compiler, not `clang-cl`. So `ds4_cuda.cu` and every header it
   includes must compile with `cl.exe`: the same shims, plus a check of the
   13 `({` sites (C++ initializers or GCC statement expressions; the latter
   `cl.exe` rejects). The C sources can use `clang-cl`, which accepts the GCC
   extensions `ds4.c` uses; it is MSVC-ABI compatible, so both link together.

Everything else is mechanical: `pread` becomes `ReadFile` with an
`OVERLAPPED` offset, `sysconf(_SC_PAGESIZE)` becomes `GetSystemInfo`, `flock`
becomes `LockFileEx`, `/proc/meminfo` becomes `GlobalMemoryStatusEx`, and the
clock calls become `QueryPerformanceCounter`.

### Things that turned out not to be blockers

- **WDDM and pinned file mappings.** The CUDA backend registers the model
  mapping with `cudaHostRegister`. Windows display drivers (WDDM, as opposed
  to TCC) may refuse that for a file-backed view, but the code already falls
  back when registration fails ("CUDA host registration skipped") and
  streams weights instead. Expect lower performance on WDDM, not a failure.
- **The FFI.** `ds4.h`'s public structs use no `long`, `off_t` or `ssize_t`,
  so Windows' LLP64 model leaves plank's `ffi.rs` layout pins valid.
- **Model size.** A single ~87 GB view is fine for `MapViewOfFile` on 64-bit
  Windows.

## Proposed upstream plan

1. **`ds4_platform.h`**: one compatibility header with the `mmap`, `pread`,
   file, lock, page-size, clock and memory-query wrappers, and a thread shim.
   On POSIX it reduces to the calls used today, so Linux and macOS builds do
   not change. Fix the two partial-unmap sites through a `remap_prefix`
   helper.
2. **`DS4_NO_DIST`**: a stub translation unit for the 52 `ds4_dist_*` /
   `ds4_tp_*` entry points that reports "not built" (the `DS4_NO_GPU`
   pattern), so a single-host build links without sockets or RDMA. Useful on
   Linux and macOS too, for smaller binaries.
3. **CPU core on Windows** (`ds4_cpu.o` and the single-host objects) with
   `clang-cl`. Proves the compatibility layer without CUDA in the picture.
4. **CUDA on Windows**: `ds4_cuda.cu` and the `cuda/mmq` tier with `nvcc` and
   `cl.exe`, then a check of the `cudaHostRegister` fallback on a WDDM card.
5. **A Windows build target**: the Makefile assumes GNU make and `cc`. Either
   a CMake file or an `nmake`-friendly list of objects and flags. plank's
   `build.rs` needs only the object list and flags, so a documented list is
   enough for plank.

ROCm on Windows (HIP SDK) is a separate question, left out of this plan.

## What plank then needs

- `crates/local-inference-engine/build.rs`: a Windows arm that compiles the
  object list with the `cc` crate (`clang-cl`, and `nvcc` for CUDA) instead of
  running `make`, and links `cudart`/`cublas` from `CUDA_PATH`. Drop the early
  return for any OS other than macOS and Linux.
- Nothing in the FFI layout, per the check above; the offset pins run as
  usual against the Windows core.
- The 96 GB floor and the VRAM count already work on Windows
  (`platform::physical_memory`, `nvidia-smi`).
- The default web tools already avoid `ds4_web.c` (`use_obscura`); a build
  without `use_obscura` would need it excluded on Windows.

Until then, WSL2 is the supported route ([docs/PLATFORMS.md](PLATFORMS.md#deepseek-and-qwen-on-windows-wsl2)).

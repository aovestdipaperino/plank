// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Builds the ds4 C engine from source on macOS and Linux.
//!
//! Produces `libds4core.a` from the ds4 Makefile's objects: the Metal backend
//! on macOS (linking Foundation and Metal), the CPU backend on Linux, with
//! `PLANK_DS4_BACKEND=cuda` the CUDA backend (`CUDA_ARCH`, default `native`,
//! and `CUDA_HOME` as the Makefile reads them), or with
//! `PLANK_DS4_BACKEND=rocm` the `ROCm` backend (`ROCM_ARCH`, the Makefile's
//! default `gfx1151` when unset, and `ROCM_PATH`, default `/opt/rocm`). The
//! sources come from
//! `DS4_SRC` when set, else from plank's `refs/ds4` submodule two directories
//! up. On other platforms, when the sources are missing, or with
//! `PLANK_NO_DS4` set, nothing is compiled:
//! the `ds4_engine` cfg stays off and `Model::open` reports the engine as
//! unavailable instead of failing to link.
//!
//! Direct dependents read the outcome from their own build script through the
//! `links = "ds4core"` metadata: `DEP_DS4CORE_ENGINE` is `1` when the engine
//! was built, `DEP_DS4CORE_BACKEND` names its backend, and
//! `DEP_DS4CORE_METAL_DIR` names its Metal kernel sources on macOS. The crate
//! itself sees the backend as `DS4_BUILT_BACKEND` (`ffi::built_backend`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    println!("cargo:rustc-check-cfg=cfg(ds4_multi_gpu)");
    println!("cargo:rerun-if-env-changed=DS4_SRC");
    println!("cargo:rerun-if-env-changed=PLANK_NO_DS4");
    println!("cargo:rerun-if-env-changed=PLANK_DS4_BACKEND");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=ROCM_ARCH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "macos" && os != "linux" {
        return;
    }
    // Opt out of the native engine even when the sources are present. The
    // engine needs a multi-gigabyte GGUF to start, so a developer smoke-testing
    // anything *around* inference otherwise cannot run a dependent binary at
    // all on a machine that has the submodule checked out.
    if std::env::var_os("PLANK_NO_DS4").is_some() {
        println!("cargo:warning=PLANK_NO_DS4 set; building without the ds4 engine");
        return;
    }
    let ds4 = source_dir();
    if !ds4.join("ds4.c").exists() {
        println!(
            "cargo:warning=no ds4 sources at {} (set DS4_SRC or check out refs/ds4); building without the ds4 engine",
            ds4.display()
        );
        return;
    }
    let backend = if os == "macos" {
        Backend::Metal
    } else {
        match std::env::var("PLANK_DS4_BACKEND").as_deref() {
            Ok("" | "cpu") | Err(_) => Backend::Cpu,
            Ok("cuda") => Backend::Cuda,
            Ok("rocm") => Backend::Rocm,
            Ok(other) => panic!("PLANK_DS4_BACKEND={other}: expected cpu, cuda or rocm"),
        }
    };
    let out_dir = std::env::var("OUT_DIR").unwrap();
    forget_other_backend_objects(&ds4, backend);
    let objs = backend.objects();
    run_make(&ds4, backend, &objs, &out_dir);

    let lib = Path::new(&out_dir).join("libds4core.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(objs.iter().map(|o| ds4.join(o)))
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=ds4core");
    emit_links(&ds4, backend);
    println!("cargo:rustc-env=DS4_BUILT_BACKEND={}", backend.name());
    println!("cargo:rustc-cfg=ds4_engine");
    if backend.multi_gpu() {
        println!("cargo:rustc-cfg=ds4_multi_gpu");
    }
    println!("cargo:engine=1");
    println!("cargo:backend={}", backend.name());
    for f in [
        "Makefile",
        "ds4.c",
        "ds4.h",
        "ds4_image.c",
        "ds4_image.h",
        "ds4_metal.m",
        "ds4_cuda.cu",
        "ds4_rocm.cu",
        "ds4_rocm_compat.cu",
        "ds4_ssd.c",
        "ds4_distributed.c",
        "ds4_tp.c",
        "ds4_layer_pack.c",
        "ds4_engram.c",
        "ds4_engram.h",
        "ds4_web.c",
        "ds4_web.h",
        "ds4_gpu_args.c",
    ] {
        println!("cargo:rerun-if-changed={}", ds4.join(f).display());
    }
}

/// The ds4 GPU backend the core is built for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// macOS: the Metal objects.
    Metal,
    /// Linux without a GPU toolkit: `ds4.c` with `-DDS4_NO_GPU`.
    Cpu,
    /// Linux with the CUDA toolkit (`PLANK_DS4_BACKEND=cuda`).
    Cuda,
    /// Linux with `ROCm` (`PLANK_DS4_BACKEND=rocm`). The C runs it as its CUDA
    /// backend, so it opens with `DS4_BACKEND_CUDA`.
    Rocm,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }

    /// Whether the core places layers across GPUs: `ds4_gpu_args.o` (the
    /// `--gpu-vram`/`--gpu-devices` parser) is linked, and its `auto` probe
    /// is defined by `ds4_cuda.o` or `ds4_rocm_compat.o`. A Metal or CPU core
    /// leaves it out, since the parser would reference that probe anyway.
    fn multi_gpu(self) -> bool {
        matches!(self, Self::Cuda | Self::Rocm)
    }

    /// The objects of the ds4 Makefile's `CORE_OBJS` (`CPU_CORE_OBJS` for the
    /// CPU) for this backend, plus `ds4_web.o`, which plank links for the web
    /// tool, and `ds4_engram.o`, which the CPU list leaves out but plank's FFI
    /// calls.
    fn objects(self) -> Vec<&'static str> {
        let mut objs = match self {
            Self::Metal => vec!["ds4.o", "ds4_metal.o"],
            Self::Cpu => vec!["ds4_cpu.o"],
            Self::Cuda => vec![
                "ds4.o",
                "ds4_cuda.o",
                "cuda/mmq/ds4_ggml_stubs.o",
                "cuda/mmq/ds4_mmq.o",
                "cuda/mmq/ds4_mmq_d2r.o",
                "cuda/mmq/quantize.o",
                "cuda/mmq/mmid.o",
                "cuda/mmq/mmvq.o",
                "cuda/mmq/ds4_repack.o",
            ],
            // `make strix-halo`'s `CORE_OBJS`.
            Self::Rocm => vec![
                "ds4.o",
                "ds4_rocm.o",
                "ds4_rocm_compat.o",
                "ds4_rocm_unavailable.o",
                "cuda/mmq/ds4_ggml_stubs.rocm.o",
                "cuda/mmq/ds4_mmq.rocm.o",
                "cuda/mmq/quantize.rocm.o",
                "cuda/mmq/mmid.rocm.o",
                "cuda/mmq/mmvq.rocm.o",
                "cuda/mmq/d2r_stubs.rocm.o",
            ],
        };
        objs.extend([
            "ds4_image.o",
            "ds4_distributed.o",
            "ds4_tp.o",
            "ds4_ssd.o",
            "ds4_layer_pack.o",
            "ds4_engram.o",
            "ds4_web.o",
        ]);
        if self.multi_gpu() {
            objs.push("ds4_gpu_args.o");
        }
        objs
    }
}

/// Builds `objs` in the ds4 tree with the Makefile's rules for `backend`.
fn run_make(ds4: &Path, backend: Backend, objs: &[&str], out_dir: &str) {
    let mut make = Command::new("make");
    make.arg("-C").arg(ds4);
    match backend {
        Backend::Cuda => {
            // The Makefile refuses an empty arch; `native` targets the GPU in
            // this machine, as `make cuda-generic` does.
            let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
            make.arg(format!("CUDA_ARCH={arch}"));
        }
        Backend::Rocm => {
            // `make strix-halo` passes the host objects these flags on a
            // recursive make; appending them after the Makefile's own
            // `CFLAGS` keeps its defaults, which a command-line `CFLAGS=`
            // would replace.
            let mk = Path::new(out_dir).join("ds4-rocm.mk");
            std::fs::write(
                &mk,
                "include Makefile\nCFLAGS += $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD\n",
            )
            .expect("failed to write the ROCm makefile");
            make.arg("-f").arg(&mk);
        }
        Backend::Metal | Backend::Cpu => {}
    }
    let status = make.args(objs).status().expect("failed to run make");
    assert!(status.success(), "ds4 engine build failed");
}

/// Links the backend's system libraries, and on macOS exports the Metal
/// kernel directory.
fn emit_links(ds4: &Path, backend: Backend) {
    match backend {
        Backend::Metal => {
            let metal = ds4.join("metal");
            println!("cargo:rustc-env=DS4_METAL_DIR={}", metal.display());
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:metal_dir={}", metal.display());
        }
        Backend::Cpu => {
            println!("cargo:rustc-link-lib=m");
        }
        Backend::Cuda => {
            let home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
            for dir in ["lib64", "targets/sbsa-linux/lib"] {
                println!("cargo:rustc-link-search=native={home}/{dir}");
            }
            for lib in ["cudart", "cublas", "stdc++", "m"] {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
        Backend::Rocm => {
            // hipcc links amdhip64 on its own; plain cc needs it named.
            let home = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
            println!("cargo:rustc-link-search=native={home}/lib");
            for lib in ["amdhip64", "hipblas", "hipblaslt", "rocblas", "stdc++", "m"] {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
    }
}

/// The objects in the ds4 tree compile differently per backend: `ds4.o`
/// (CUDA, or `ROCm` with `DS4_ROCM_BUILD`) and `ds4_tp.o` (which also reads
/// `DS4_ROCM_BUILD`). make sees no flags, only timestamps, so after a switch
/// it would link the other backend's copies. A stamp beside them records the
/// backend they were last built for; when it differs they are deleted. The
/// stamp ends in `.o` so the tree's `*.o` ignore rule covers it.
fn forget_other_backend_objects(ds4: &Path, backend: Backend) {
    let stamp = ds4.join(".plank-backend.o");
    if std::fs::read_to_string(&stamp).is_ok_and(|s| s == backend.name()) {
        return;
    }
    for obj in ["ds4.o", "ds4_tp.o"] {
        let _ = std::fs::remove_file(ds4.join(obj));
    }
    std::fs::write(&stamp, backend.name()).expect("failed to write the backend stamp");
}

/// `DS4_SRC`, else `refs/ds4` in the plank checkout this crate lives in.
fn source_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("DS4_SRC") {
        return PathBuf::from(dir);
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = Path::new(&manifest).join("../../refs/ds4");
    dir.canonicalize().unwrap_or(dir)
}

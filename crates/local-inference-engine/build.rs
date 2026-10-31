// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Builds the ds4 C engine from source on macOS and Linux.
//!
//! Produces `libds4core.a` from the ds4 Makefile's objects: the Metal backend
//! on macOS (linking Foundation and Metal), the CPU backend on Linux, or with
//! `PLANK_DS4_BACKEND=cuda` the CUDA backend (`CUDA_ARCH`, default `native`,
//! and `CUDA_HOME` as the Makefile reads them). The sources come from
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
    println!("cargo:rerun-if-env-changed=DS4_SRC");
    println!("cargo:rerun-if-env-changed=PLANK_NO_DS4");
    println!("cargo:rerun-if-env-changed=PLANK_DS4_BACKEND");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
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
            Ok(other) => panic!("PLANK_DS4_BACKEND={other}: expected cpu or cuda"),
        }
    };
    let objs = backend.objects();
    let mut make = Command::new("make");
    make.arg("-C").arg(&ds4);
    if backend == Backend::Cuda {
        // The Makefile refuses an empty arch; `native` targets the GPU in
        // this machine, as `make cuda-generic` does.
        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
        make.arg(format!("CUDA_ARCH={arch}"));
    }
    let status = make
        .args(&objs)
        .status()
        .expect("failed to run make");
    assert!(status.success(), "ds4 engine build failed");

    let out_dir = std::env::var("OUT_DIR").unwrap();
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
    }
    println!("cargo:rustc-env=DS4_BUILT_BACKEND={}", backend.name());
    println!("cargo:rustc-cfg=ds4_engine");
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
        "ds4_ssd.c",
        "ds4_distributed.c",
        "ds4_tp.c",
        "ds4_layer_pack.c",
        "ds4_engram.c",
        "ds4_engram.h",
        "ds4_web.c",
        "ds4_web.h",
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
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
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
        objs
    }
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

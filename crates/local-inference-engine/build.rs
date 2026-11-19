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
//!
//! **Pre-built cores** (#89, `docs/PLATFORMS.md`): with `PLANK_DS4_PREBUILT`
//! set, nothing is compiled. The variable names a package (a directory or a
//! `.tar.gz` holding `libds4core.a` and its `ds4core.json` manifest), an
//! `https://` URL of one (checked against `PLANK_DS4_PREBUILT_SHA256`), or
//! `github`, which fetches the package for this target and backend from the
//! `ds4core-<commit>` release of the commit `refs/ds4` pins and checks it
//! against that release's `SHA256SUMS`. Either way the manifest must name this
//! target, the backend asked for, and the pinned ds4 commit, and the library
//! must match its recorded hash, before anything is linked. With
//! `PLANK_DS4_EXPORT=<dir>`, a source build also writes such a package there;
//! the `ds4core` workflow publishes those.

#[path = "build_support/prebuilt.rs"]
mod prebuilt;

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    println!("cargo:rustc-check-cfg=cfg(ds4_multi_gpu)");
    for var in [
        "DS4_SRC",
        "DS4_COMMIT",
        "PLANK_NO_DS4",
        "PLANK_DS4_BACKEND",
        "PLANK_DS4_PREBUILT",
        "PLANK_DS4_PREBUILT_SHA256",
        "PLANK_DS4_PREBUILT_ARCH",
        "PLANK_DS4_PREBUILT_REPO",
        "PLANK_DS4_PREBUILT_ALLOW_MISMATCH",
        "PLANK_DS4_EXPORT",
        "NATIVE_CPU_FLAG",
        "CUDA_ARCH",
        "CUDA_HOME",
        "ROCM_ARCH",
        "ROCM_PATH",
        "HIPCC",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let prebuilt = std::env::var("PLANK_DS4_PREBUILT")
        .ok()
        .filter(|s| !s.is_empty());
    if os != "macos" && os != "linux" {
        assert!(
            prebuilt.is_none(),
            "PLANK_DS4_PREBUILT is set, but no ds4 core exists for {os}: the C engine builds on \
             macOS and Linux only (on Windows, run plank under WSL2; see docs/PLATFORMS.md)"
        );
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
    if let Some(spec) = prebuilt {
        link_prebuilt(&spec, &os);
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
    let backend = requested_backend(&os);
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

    if let Some(dir) = std::env::var_os("PLANK_DS4_EXPORT").filter(|d| !d.is_empty()) {
        export_package(&ds4, backend, &lib, Path::new(&dir));
    }
    let metal = ds4.join("metal");
    emit_engine(Path::new(&out_dir), backend, Some(&metal));
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

/// The backend `PLANK_DS4_BACKEND` asks for: Metal on macOS, else the CPU
/// unless it names a GPU backend.
fn requested_backend(os: &str) -> Backend {
    if os == "macos" {
        return Backend::Metal;
    }
    match std::env::var("PLANK_DS4_BACKEND").as_deref() {
        Ok("" | "cpu") | Err(_) => Backend::Cpu,
        Ok("cuda") => Backend::Cuda,
        Ok("rocm") => Backend::Rocm,
        Ok(other) => panic!("PLANK_DS4_BACKEND={other}: expected cpu, cuda or rocm"),
    }
}

impl Backend {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "metal" => Self::Metal,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "rocm" => Self::Rocm,
            _ => return None,
        })
    }
}

/// Links `libds4core.a` from `lib_dir` and reports the engine: the cfgs, the
/// backend for `ffi::built_backend`, the system libraries and the `links`
/// metadata dependents read. Shared by the source and pre-built paths so the
/// two can never report differently.
fn emit_engine(lib_dir: &Path, backend: Backend, metal: Option<&Path>) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=ds4core");
    emit_links(backend, metal);
    println!("cargo:rustc-env=DS4_BUILT_BACKEND={}", backend.name());
    println!("cargo:rustc-cfg=ds4_engine");
    if backend.multi_gpu() {
        println!("cargo:rustc-cfg=ds4_multi_gpu");
    }
    println!("cargo:engine=1");
    println!("cargo:backend={}", backend.name());
}

/// Links the backend's system libraries, and for Metal exports the kernel
/// directory.
fn emit_links(backend: Backend, metal: Option<&Path>) {
    match backend {
        Backend::Metal => {
            let metal = metal.expect("a Metal core needs its kernel directory");
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
/// `DS4_ROCM_BUILD`), and every object with `NATIVE_CPU_FLAG` and the GPU
/// arch. make sees no flags, only timestamps, so after a switch it would link
/// copies built for the other backend or flags — and an exported package
/// would then record flags its objects were not built with. A stamp beside
/// them records what they were last built for; when it differs, they are
/// deleted. The stamp ends in `.o` so the tree's `*.o` ignore rule covers it.
fn forget_other_backend_objects(ds4: &Path, backend: Backend) {
    let stamp = ds4.join(".plank-backend.o");
    let var = |name: &str| std::env::var(name).unwrap_or_default();
    let arch = match backend {
        Backend::Cuda => var("CUDA_ARCH"),
        Backend::Rocm => var("ROCM_ARCH"),
        Backend::Metal | Backend::Cpu => String::new(),
    };
    let wanted = format!("{}|{}|{arch}", backend.name(), var("NATIVE_CPU_FLAG"));
    if std::fs::read_to_string(&stamp).is_ok_and(|s| s == wanted) {
        return;
    }
    for obj in backend.objects().into_iter().chain(["ds4.o", "ds4_tp.o"]) {
        let _ = std::fs::remove_file(ds4.join(obj));
    }
    std::fs::write(&stamp, wanted).expect("failed to write the backend stamp");
}

/// Writes the package for the core just built into `dir`: `libds4core.a`,
/// `ds4core.json`, `ds4.h` and, for Metal, `metal/*.metal`. `dir` is emptied
/// first so a package never carries a previous backend's kernels.
fn export_package(ds4: &Path, backend: Backend, lib: &Path, dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("failed to create PLANK_DS4_EXPORT");
    std::fs::copy(lib, dir.join(prebuilt::LIBRARY)).expect("failed to export libds4core.a");
    std::fs::copy(ds4.join("ds4.h"), dir.join("ds4.h")).expect("failed to export ds4.h");
    if backend == Backend::Metal {
        let to = dir.join("metal");
        std::fs::create_dir_all(&to).expect("failed to create the exported metal/");
        for entry in std::fs::read_dir(ds4.join("metal")).expect("no metal/ in the ds4 tree") {
            let path = entry.expect("failed to list metal/").path();
            if path.extension().is_some_and(|e| e == "metal") {
                std::fs::copy(&path, to.join(path.file_name().unwrap()))
                    .expect("failed to export a Metal kernel");
            }
        }
    }
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(ds4)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let ds4_commit = git(&["rev-parse", "HEAD"])
        .expect("PLANK_DS4_EXPORT needs the ds4 sources in a git checkout, to record the commit");
    let ds4_dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    let (variant, toolkit) = match backend {
        Backend::Metal | Backend::Cpu => (String::new(), String::new()),
        Backend::Cuda => {
            let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "native".to_string());
            let home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
            let toolkit = std::fs::read_to_string(Path::new(&home).join("version.json"))
                .ok()
                .and_then(|j| prebuilt::cuda_toolkit_version(&j))
                .unwrap_or_default();
            (prebuilt::variant_of_cuda_arch(&arch), toolkit)
        }
        Backend::Rocm => {
            let arch = std::env::var("ROCM_ARCH").unwrap_or_else(|_| "gfx1151".to_string());
            let home = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
            let toolkit = std::fs::read_to_string(Path::new(&home).join(".info/version"))
                .map(|v| v.trim().to_string())
                .unwrap_or_default();
            (arch, toolkit)
        }
    };
    assert!(
        variant != "native",
        "PLANK_DS4_EXPORT with CUDA_ARCH=native: set CUDA_ARCH to the sm_* the package is for"
    );
    let cpu_flag = std::env::var("NATIVE_CPU_FLAG").unwrap_or_else(|_| "native".to_string());
    if cpu_flag.contains("native") {
        println!(
            "cargo:warning=exporting a ds4 core built for this CPU only; set NATIVE_CPU_FLAG for a portable one"
        );
    }
    let manifest = prebuilt::Manifest {
        ds4_commit,
        ds4_dirty,
        target: std::env::var("TARGET").unwrap(),
        backend: backend.name().to_string(),
        variant,
        cpu_flag: if cpu_flag.contains("native") {
            "native".to_string()
        } else {
            cpu_flag
        },
        toolkit,
        lib_sha256: prebuilt::sha256_file(lib).expect("failed to hash libds4core.a"),
    };
    std::fs::write(dir.join(prebuilt::MANIFEST), manifest.to_json())
        .expect("failed to write ds4core.json");
    println!(
        "cargo:warning=exported the ds4 core to {} as {}",
        dir.display(),
        prebuilt::asset_name(&manifest.target, &manifest.backend, &manifest.variant)
    );
}

/// Links a pre-built core instead of compiling one; see the module docs.
fn link_prebuilt(spec: &str, os: &str) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = std::env::var("TARGET").unwrap();
    let asked = std::env::var("PLANK_DS4_BACKEND")
        .ok()
        .filter(|b| !b.is_empty());
    let backend_name = if os == "macos" {
        "metal".to_string()
    } else {
        asked.clone().unwrap_or_else(|| "cpu".to_string())
    };
    let pin = pinned_commit();

    let package = if spec == "github" {
        let archive = fetch_from_release(&out_dir, &target, &backend_name, pin.as_deref());
        unpack(&archive, &out_dir)
    } else if spec.starts_with("https://") || spec.starts_with("http://") {
        let want = std::env::var("PLANK_DS4_PREBUILT_SHA256")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                panic!("PLANK_DS4_PREBUILT is a URL: set PLANK_DS4_PREBUILT_SHA256 to the archive's SHA-256")
            });
        let name = spec.rsplit('/').next().unwrap_or("ds4core.tar.gz");
        let archive = download_verified(spec, &out_dir.join(name), &want.to_ascii_lowercase());
        unpack(&archive, &out_dir)
    } else {
        let path = PathBuf::from(spec);
        println!("cargo:rerun-if-changed={}", path.display());
        if path.is_dir() {
            path
        } else if path.is_file() {
            if let Some(want) = std::env::var("PLANK_DS4_PREBUILT_SHA256")
                .ok()
                .filter(|s| !s.is_empty())
            {
                check_sha(&path, &want.to_ascii_lowercase());
            }
            unpack(&path, &out_dir)
        } else {
            panic!(
                "PLANK_DS4_PREBUILT={spec}: no such file or directory (expected `github`, a URL, a package directory or a .tar.gz)"
            );
        }
    };

    let text = std::fs::read_to_string(package.join(prebuilt::MANIFEST)).unwrap_or_else(|e| {
        panic!(
            "{}: no {} ({e}); not a ds4 core package",
            package.display(),
            prebuilt::MANIFEST
        )
    });
    let manifest =
        prebuilt::Manifest::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", package.display()));
    let allow = std::env::var_os("PLANK_DS4_PREBUILT_ALLOW_MISMATCH").is_some_and(|v| v != "0");
    let expect = prebuilt::Expect {
        target: &target,
        backend: (os == "macos" || asked.is_some()).then_some(backend_name.as_str()),
        ds4_commit: pin.as_deref(),
        allow_commit_mismatch: allow,
    };
    let warnings = manifest
        .check(&expect)
        .unwrap_or_else(|e| panic!("{}: {e}", package.display()));
    for w in warnings {
        println!("cargo:warning={w}");
    }
    let lib = package.join(prebuilt::LIBRARY);
    check_sha(&lib, &manifest.lib_sha256);
    let backend = Backend::named(&manifest.backend).expect("Manifest::parse checks the backend");
    let metal = package.join("metal");
    if backend == Backend::Metal {
        assert!(
            metal.join("flash_attn.metal").is_file(),
            "{}: a Metal core package must carry metal/*.metal",
            package.display()
        );
    }
    if backend == Backend::Cuda {
        warn_cuda_mismatch(&manifest.toolkit);
    }
    println!(
        "cargo:warning=linking the pre-built ds4 core {} (ds4 {})",
        prebuilt::asset_name(&manifest.target, &manifest.backend, &manifest.variant),
        prebuilt::short(&manifest.ds4_commit)
    );
    let lib_dir = lib.parent().unwrap().to_path_buf();
    emit_engine(
        &lib_dir,
        backend,
        (backend == Backend::Metal).then_some(metal.as_path()),
    );
}

/// Downloads this target's package from the `ds4core-<commit>` release,
/// checked against the release's `SHA256SUMS`. A copy left in `OUT_DIR` by an
/// earlier build is reused when it still matches.
fn fetch_from_release(out_dir: &Path, target: &str, backend: &str, pin: Option<&str>) -> PathBuf {
    let commit = pin.unwrap_or_else(|| {
        panic!(
            "PLANK_DS4_PREBUILT=github needs the ds4 commit refs/ds4 pins, and this checkout has no \
             git: set DS4_COMMIT to it"
        )
    });
    let variant = match backend {
        "cuda" => std::env::var("PLANK_DS4_PREBUILT_ARCH")
            .ok()
            .filter(|a| !a.is_empty())
            .or_else(detect_cuda_variant)
            .unwrap_or_else(|| {
                panic!(
                    "no NVIDIA GPU found to pick a CUDA build for: set PLANK_DS4_PREBUILT_ARCH to one of {}",
                    prebuilt::CUDA_VARIANTS.join(", ")
                )
            }),
        "rocm" => std::env::var("PLANK_DS4_PREBUILT_ARCH")
            .or_else(|_| std::env::var("ROCM_ARCH"))
            .unwrap_or_else(|_| "gfx1151".to_string()),
        _ => String::new(),
    };
    let repo = std::env::var("PLANK_DS4_PREBUILT_REPO")
        .ok()
        .filter(|r| !r.is_empty())
        .unwrap_or_else(|| prebuilt::DEFAULT_REPO.to_string());
    let tag = prebuilt::release_tag(commit);
    let asset = prebuilt::asset_name(target, backend, &variant);
    let sums_file = out_dir.join(prebuilt::sums_name(&tag));
    curl(
        &prebuilt::release_url(&repo, &tag, &prebuilt::sums_name(&tag)),
        &sums_file,
    );
    let sums =
        std::fs::read_to_string(&sums_file).expect("failed to read the downloaded SHA256SUMS");
    let want = prebuilt::sum_for(&sums, &asset).unwrap_or_else(|| {
        panic!(
            "release {tag} of {repo} has no {asset}; the published cores are listed in its {} \
             (build from source instead, or pick another PLANK_DS4_PREBUILT_ARCH)",
            prebuilt::sums_name(&tag)
        )
    });
    download_verified(
        &prebuilt::release_url(&repo, &tag, &asset),
        &out_dir.join(&asset),
        &want,
    )
}

/// The CUDA variant for the first GPU `nvidia-smi` reports.
fn detect_cuda_variant() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let cap = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .to_string();
    let variant = prebuilt::cuda_variant_for(&cap, prebuilt::CUDA_VARIANTS)?;
    println!("cargo:warning=GPU compute capability {cap}: using the {variant} ds4 core");
    Some(variant)
}

/// Fetches `url` into `dest` unless a copy with hash `want` is already there,
/// then checks the hash.
fn download_verified(url: &str, dest: &Path, want: &str) -> PathBuf {
    let cached = dest.is_file() && prebuilt::sha256_file(dest).is_ok_and(|h| h == want);
    if !cached {
        curl(url, dest);
    }
    check_sha(dest, want);
    dest.to_path_buf()
}

fn curl(url: &str, dest: &Path) {
    let status = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .unwrap_or_else(|e| panic!("failed to run curl for {url}: {e}"));
    assert!(status.success(), "failed to download {url}");
}

fn check_sha(path: &Path, want: &str) {
    let got = prebuilt::sha256_file(path)
        .unwrap_or_else(|e| panic!("failed to hash {}: {e}", path.display()));
    assert!(
        got == want,
        "{}: SHA-256 {got} does not match the expected {want}; refusing to link it",
        path.display()
    );
}

/// Extracts a package archive into a fresh `OUT_DIR/ds4core`.
fn unpack(archive: &Path, out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("ds4core");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create the package directory");
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(&dir)
        .status()
        .expect("failed to run tar");
    assert!(status.success(), "failed to extract {}", archive.display());
    dir
}

/// Warns when the CUDA toolkit this build links against has a different
/// major version from the one the core was compiled with: cudart keeps its ABI
/// within a major only.
fn warn_cuda_mismatch(built_with: &str) {
    if built_with.is_empty() {
        return;
    }
    let home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let local = std::fs::read_to_string(Path::new(&home).join("version.json"))
        .ok()
        .and_then(|j| prebuilt::cuda_toolkit_version(&j));
    match local {
        Some(v) if prebuilt::major(&v) != prebuilt::major(built_with) => println!(
            "cargo:warning=the pre-built ds4 core was compiled with CUDA {built_with}, {home} has CUDA {v}; \
             link against a CUDA {}.x runtime",
            prebuilt::major(built_with)
        ),
        Some(_) => {}
        None => println!(
            "cargo:warning=no CUDA runtime found at {home} (set CUDA_HOME); the pre-built core needs \
             CUDA {built_with} libraries (libcudart, libcublas) to link"
        ),
    }
}

/// The ds4 commit `refs/ds4` pins: `DS4_COMMIT` when set, else the gitlink in
/// the plank checkout's index. Read from the index, not the submodule's own
/// HEAD, so it works without the submodule checked out.
fn pinned_commit() -> Option<String> {
    if let Ok(c) = std::env::var("DS4_COMMIT")
        && !c.is_empty()
    {
        return Some(c.to_ascii_lowercase());
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).join("../..");
    // A submodule bump changes the pin without touching any file this script
    // reads; without this, the old core would stay linked against the new
    // FFI layouts. (In a linked worktree `.git` is a file, and a bump there
    // needs a `cargo clean -p local-inference-engine`.)
    let index = root.join(".git").join("index");
    if index.is_file() {
        println!("cargo:rerun-if-changed={}", index.display());
    }
    let out = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["ls-files", "--stage", "--", "refs/ds4"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    prebuilt::gitlink_commit(&String::from_utf8_lossy(&out.stdout))
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

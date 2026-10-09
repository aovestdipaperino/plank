// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Builds the ds4 C engine from source on macOS.
//!
//! Produces `libds4core.a` from the Metal-backend objects and links the
//! required frameworks. The sources come from `DS4_SRC` when set, else from
//! plank's `refs/ds4` submodule two directories up. On other platforms, when
//! the sources are missing, or with `PLANK_NO_DS4` set, nothing is compiled:
//! the `ds4_engine` cfg stays off and `Model::open` reports the engine as
//! unavailable instead of failing to link.
//!
//! Direct dependents read the outcome from their own build script through the
//! `links = "ds4core"` metadata: `DEP_DS4CORE_ENGINE` is `1` when the engine
//! was built, and `DEP_DS4CORE_METAL_DIR` names its Metal kernel sources.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    println!("cargo:rerun-if-env-changed=DS4_SRC");
    println!("cargo:rerun-if-env-changed=PLANK_NO_DS4");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
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
    // Mirrors the Metal-branch `CORE_OBJS` in the ds4 Makefile, plus ds4_web.o
    // which plank links for the web tool.
    let objs = [
        "ds4.o",
        "ds4_image.o",
        "ds4_distributed.o",
        "ds4_tp.o",
        "ds4_ssd.o",
        "ds4_metal.o",
        "ds4_layer_pack.o",
        "ds4_engram.o",
        "ds4_web.o",
    ];
    let status = Command::new("make")
        .arg("-C")
        .arg(&ds4)
        .args(objs)
        .status()
        .expect("failed to run make");
    assert!(status.success(), "ds4 engine build failed");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib = Path::new(&out_dir).join("libds4core.a");
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(objs.iter().map(|o| ds4.join(o)))
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed");

    let metal = ds4.join("metal");
    println!("cargo:rustc-env=DS4_METAL_DIR={}", metal.display());
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=ds4core");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-cfg=ds4_engine");
    println!("cargo:engine=1");
    println!("cargo:metal_dir={}", metal.display());
    for f in [
        "ds4.c",
        "ds4.h",
        "ds4_image.c",
        "ds4_image.h",
        "ds4_metal.m",
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

/// `DS4_SRC`, else `refs/ds4` in the plank checkout this crate lives in.
fn source_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("DS4_SRC") {
        return PathBuf::from(dir);
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = Path::new(&manifest).join("../../refs/ds4");
    dir.canonicalize().unwrap_or(dir)
}

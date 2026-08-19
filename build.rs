// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Builds the ds4 C engine from the `refs/ds4` submodule on macOS.
//!
//! Produces `libds4core.a` from the Metal-backend objects and links the
//! required frameworks. On other platforms (or if the submodule is missing)
//! the `ds4_engine` cfg is not emitted and plank falls back to the echo
//! engine only.

use std::path::Path;
use std::process::Command;

/// The sprite codec, compiled here as well as into the crate so the encoder
/// that runs at build time and the decoder that runs in the binary are one
/// piece of code rather than two that agree today.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // Opt out of the native engine even when the submodule is present. The
    // engine needs a multi-gigabyte GGUF to start, so a developer smoke-testing
    // anything *around* inference — plugins, slash commands, the session
    // lifecycle — otherwise cannot run the binary at all on a machine that has
    // the submodule checked out. This builds the same plank CI builds.
    println!("cargo:rerun-if-env-changed=PLANK_NO_DS4");
    if std::env::var_os("PLANK_NO_DS4").is_some() {
        println!("cargo:warning=PLANK_NO_DS4 set; building without the ds4 engine");
        return;
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ds4 = Path::new(&manifest).join("refs/ds4");
    if !ds4.join("ds4.c").exists() {
        println!("cargo:warning=refs/ds4 submodule missing; building without the ds4 engine");
        return;
    }
    // Mirrors the Metal-branch `CORE_OBJS` in the ds4 Makefile, plus ds4_web.o
    // which plank links for the web tool.
    let objs = [
        "ds4.o",
        "ds4_distributed.o",
        "ds4_tp.o",
        "ds4_ssd.o",
        "ds4_metal.o",
        "ds4_layer_pack.o",
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

    println!(
        "cargo:rustc-env=DS4_METAL_DIR={}",
        ds4.join("metal").display()
    );
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=ds4core");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-cfg=ds4_engine");
    for f in [
        "ds4.c",
        "ds4.h",
        "ds4_metal.m",
        "ds4_ssd.c",
        "ds4_distributed.c",
        "ds4_tp.c",
        "ds4_layer_pack.c",
        "ds4_web.c",
        "ds4_web.h",
    ] {
        println!("cargo:rerun-if-changed={}", ds4.join(f).display());
    }
}

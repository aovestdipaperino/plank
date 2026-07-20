//! Builds the ds4 C engine from the `refs/ds4` submodule on macOS.
//!
//! Produces `libds4core.a` from the Metal-backend objects and links the
//! required frameworks. On other platforms (or if the submodule is missing)
//! the `ds4_engine` cfg is not emitted and plank falls back to the echo
//! engine only.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ds4 = Path::new(&manifest).join("refs/ds4");
    if !ds4.join("ds4.c").exists() {
        println!("cargo:warning=refs/ds4 submodule missing; building without the ds4 engine");
        return;
    }
    let objs = ["ds4.o", "ds4_distributed.o", "ds4_ssd.o", "ds4_metal.o"];
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
    ] {
        println!("cargo:rerun-if-changed={}", ds4.join(f).display());
    }
}

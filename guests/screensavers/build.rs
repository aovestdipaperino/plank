// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Packs the minions sprite sheet the same way plank's own `build.rs` does.
//!
//! The sheet travels with the plugin rather than being read out of plank's
//! source tree: a plugin that reaches into the host's repository at build time
//! is not a plugin, it is a coupled artifact that happens to compile to wasm.
//! The cost is that the art exists in two places while the face also ships as
//! a built-in; the moment it stops being a built-in, plank's copy goes.

use std::path::Path;

#[path = "src/minions/codec.rs"]
mod minions_codec;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let sheet = Path::new(&manifest).join("resources/minions.txt");
    println!("cargo:rerun-if-changed={}", sheet.display());
    println!("cargo:rerun-if-changed=src/minions/codec.rs");
    let text = std::fs::read_to_string(&sheet)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", sheet.display()));
    let cells =
        minions_codec::parse_sheet(&text).unwrap_or_else(|e| panic!("{}: {e}", sheet.display()));
    let blob = minions_codec::encode(&cells);
    assert_eq!(
        minions_codec::decode(&blob),
        cells,
        "the packed sprite sheet does not decode back to the art"
    );

    let out = std::env::var("OUT_DIR").unwrap();
    let out = Path::new(&out);
    std::fs::write(out.join("minions.bin"), &blob).expect("cannot write the sprite blob");
    std::fs::write(
        out.join("minions_stats.rs"),
        format!(
            "/// Ink codes in the sheet: one byte per cell, before packing.\n\
             pub const SHEET_BYTES: usize = {};\n\
             /// Bytes the packed sheet occupies in the executable.\n\
             pub const BLOB_BYTES: usize = {};\n",
            cells.len(),
            blob.len()
        ),
    )
    .expect("cannot write the sprite stats");
}

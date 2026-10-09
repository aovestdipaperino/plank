// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Sets the `ds4_engine` cfg when `crates/local-inference-engine` compiled the C engine,
//! and stamps the git commit for `--version`.
//!
//! The engine itself (the `make` of `refs/ds4`, `libds4core.a`, the framework
//! links) is built by that crate's own build script; it reports the outcome
//! through its `links = "ds4core"` metadata as `DEP_DS4CORE_ENGINE`, so the
//! C is compiled once and plank's cfg can never disagree with what was linked.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ds4_engine)");
    emit_git_commit();
    // Read by the engine crate's build script; listed here as well so plank's
    // cfg is re-derived in the same build that toggles it.
    println!("cargo:rerun-if-env-changed=PLANK_NO_DS4");
    if std::env::var_os("DEP_DS4CORE_ENGINE").is_some() {
        println!("cargo:rustc-cfg=ds4_engine");
    }
}

/// Emits `PLANK_GIT_COMMIT` for `--version`: the short HEAD hash, with a `-dirty`
/// suffix when the working tree has uncommitted changes. A source tree with no
/// git (a crates.io/tarball build) gets `unknown`, so the env var always exists
/// and `env!` in the crate never fails to compile.
fn emit_git_commit() {
    let commit = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .filter(|s| !s.is_empty());
    let commit = match commit {
        Some(c) => {
            let dirty = Command::new("git")
                .args(["status", "--porcelain", "--untracked-files=no"])
                .output()
                .is_ok_and(|o| o.status.success() && !o.stdout.is_empty());
            if dirty { format!("{c}-dirty") } else { c }
        }
        None => "unknown".to_owned(),
    };
    println!("cargo:rustc-env=PLANK_GIT_COMMIT={commit}");
    // Rebuild when HEAD moves (branch switch, new commit) so the stamp is not
    // frozen at whatever the first build saw.
    for p in [".git/HEAD", ".git/logs/HEAD"] {
        if Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
}

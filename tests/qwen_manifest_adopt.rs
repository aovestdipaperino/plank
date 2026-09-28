//! The compiled-in catalog's `qwen` engine, checked against the artifacts
//! actually on disk.
//!
//! The consequence of getting this wrong is a 177 GB re-download of files the
//! user already has, so it is worth checking against the real filesystem
//! rather than only against synthesized sizes. Read-only: it stats the install
//! slots and writes nothing. Self-skips when the artifacts are absent, which
//! is every machine but one.

use plank::manifest::{Decision, EngineId};

#[test]
fn the_qwen_engine_adopts_the_artifacts_on_this_disk() {
    let catalog = plank::engines::parse(
        plank::engines::COMPILED_IN,
        plank::engines::Layer::Published,
        &mut Vec::new(),
    )
    .expect("the compiled-in catalog parses");
    let entry = catalog.get("qwen").expect("the catalog declares qwen");
    let remote = entry.to_manifest().expect("qwen is downloadable");
    let expected_version = remote.version;
    let roles: Vec<String> = remote.files.keys().cloned().collect();
    let kinds: Vec<&str> = roles.iter().map(String::as_str).collect();

    let present = |kind: &str| -> Option<u64> {
        let path = plank::manifest::local_path_for(EngineId::QWEN, kind)?;
        // Follows symlinks on purpose: the install slots are expected to be
        // links to wherever the user keeps the model.
        std::fs::metadata(path).ok().map(|m| m.len())
    };

    if kinds.iter().any(|k| present(k).is_none()) {
        eprintln!("Qwen artifacts not installed; skipping the on-disk adoption check");
        return;
    }

    let sizes: Vec<_> = kinds.iter().map(|k| (*k, present(k))).collect();
    match plank::manifest::decide(remote, None, &kinds, &present) {
        Decision::Adopt(m) => assert_eq!(m.version, expected_version),
        other => panic!(
            "the catalog must adopt what is already installed, not offer a \
             re-download: got {other:?}. Sizes on disk: {sizes:?}"
        ),
    }
}

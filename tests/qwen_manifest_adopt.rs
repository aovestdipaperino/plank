//! The committed Qwen manifest, checked against the artifacts actually on
//! disk.
//!
//! The consequence of getting this wrong is a 107 GB re-download of files the
//! user already has, so it is worth checking against the real filesystem
//! rather than only against synthesized sizes. Self-skips when the artifacts
//! are absent, which is every machine but one.

use plank::manifest::{Decision, ModelSet};

#[test]
fn the_qwen_manifest_adopts_the_artifacts_on_this_disk() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("qwen.manifest"),
    )
    .expect("qwen.manifest is committed");
    let remote = plank::manifest::parse(&text).expect("it parses");

    let present = |kind: &str| -> Option<u64> {
        let path = plank::manifest::local_path_for(ModelSet::Qwen, kind)?;
        // Follows symlinks on purpose: the install slots are expected to be
        // links to wherever the user keeps the model.
        std::fs::metadata(path).ok().map(|m| m.len())
    };

    if ModelSet::Qwen.kinds().iter().any(|k| present(k).is_none()) {
        eprintln!("Qwen artifacts not installed; skipping the on-disk adoption check");
        return;
    }

    match plank::manifest::decide(remote, None, ModelSet::Qwen.kinds(), &present) {
        Decision::Adopt(m) => assert_eq!(m.version, 1),
        other => panic!(
            "the manifest must adopt what is already installed, not offer a \
             re-download: got {other:?}. Sizes on disk: {:?}",
            ModelSet::Qwen
                .kinds()
                .iter()
                .map(|k| (*k, present(k)))
                .collect::<Vec<_>>()
        ),
    }
}

//! The Qwen cache split, verified in its own process.
//!
//! `SessionStore::set_cache_leaf` writes a `OnceLock`, so the effect of
//! *setting* it cannot be observed from the unit tests: they all share one
//! process, and whichever test set the leaf first would decide the directory
//! every later test resolves. An integration test gets a process to itself,
//! which is the only place the override can be exercised end to end.

use plank::session::{QWEN_CACHE_LEAF, SessionStore, cache_leaf_for};
use std::path::Path;

#[test]
fn setting_the_qwen_leaf_moves_the_default_cache_dir() {
    let before = SessionStore::default_dir();
    assert_eq!(
        before.file_name().and_then(|n| n.to_str()),
        Some("kvcache"),
        "an unset leaf must resolve to the DeepSeek default"
    );

    SessionStore::set_cache_leaf(cache_leaf_for(Some(Path::new("ple.gguf"))));

    let after = SessionStore::default_dir();
    assert_eq!(
        after.file_name().and_then(|n| n.to_str()),
        Some(QWEN_CACHE_LEAF),
        "a Qwen run must resolve to its own leaf"
    );
    // Same parent, different leaf: this is a sibling of the DeepSeek cache
    // under ~/.plank, not a relocation somewhere else.
    assert_eq!(before.parent(), after.parent());
    assert_ne!(before, after);
}

//! Named directional-steering vectors from `~/.plank/models/vectors.json`.
//!
//! The store itself, its format and the decoding the C engine needs live in
//! `plank-lib` (`crates/plank-lib`, shared with `pt`, which writes the file
//! with `pt vectorize -n NAME` and `pt install`). This module is plank's
//! face onto it: paths under plank's own home, and errors as the messages
//! plank prints.
//!
//! An entry's `model` is the plank engine name (`ds4vision`); a run on a bare
//! GGUF path that is no engine is matched by the file's name instead
//! ([`model_keys`]). A chosen vector is decoded into
//! `~/.plank/cache/steering/<sha256>.f32`, named by its content, so the KV
//! key (which digests the file's bytes, `ds4engine::steering_key`) does not
//! depend on the name it was chosen by.

use std::path::{Path, PathBuf};

use plank_lib::vectors::VectorStore;

pub use plank_lib::profiles::Steering;
pub use plank_lib::vectors::model_keys;

/// The vectors file under the plank directory `root`.
#[must_use]
pub fn store_path_in(root: &Path) -> PathBuf {
    plank_lib::vectors::store_path_in(root)
}

/// [`store_path_in`] rooted at `~/.plank`.
#[must_use]
pub fn store_path() -> PathBuf {
    store_path_in(&crate::manifest::plank_dir())
}

/// The direction names stored for the model under `keys`, in key order then
/// file order, each name once. A missing store is no directions.
///
/// # Errors
/// Returns a message naming the file when it exists but is not a valid store.
pub fn directions_in(store: &Path, keys: &[String]) -> Result<Vec<String>, String> {
    VectorStore::new(store)
        .directions(keys)
        .map_err(|e| format!("steering: {e}"))
}

/// [`directions_in`] on the default store.
///
/// # Errors
/// As [`directions_in`].
pub fn directions(keys: &[String]) -> Result<Vec<String>, String> {
    directions_in(&store_path(), keys)
}

/// Decodes direction `name`, stored under the first of `keys` that has it,
/// into a content-named vector file under `cache_dir`, and returns its path.
///
/// # Errors
/// Returns a message when the store is invalid, the model has no direction of
/// that name (the message lists the ones it has), the value is not base64 of
/// whole `f32`s, or the file cannot be written.
pub fn materialize_in(
    store: &Path,
    cache_dir: &Path,
    keys: &[String],
    name: &str,
) -> Result<PathBuf, String> {
    VectorStore::new(store)
        .materialize(keys, name, cache_dir)
        .map_err(|e| e.to_string())
}

/// [`materialize_in`] on the default store and cache.
///
/// # Errors
/// As [`materialize_in`].
pub fn materialize(keys: &[String], name: &str) -> Result<PathBuf, String> {
    let root = crate::manifest::plank_dir();
    materialize_in(
        &store_path_in(&root),
        &root.join("cache").join("steering"),
        keys,
        name,
    )
}

/// Reads a `steering` block (`plank_lib::profiles::parse_steering`).
///
/// # Errors
/// Describes the first problem found, prefixed `steering:`.
pub fn parse_steering(v: &serde_json::Value) -> Result<Steering, String> {
    plank_lib::profiles::parse_steering(v).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plank_reads_what_the_shared_store_holds() {
        let dir = std::env::temp_dir().join(format!(
            "plank-steervec-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = store_path_in(&dir);
        VectorStore::new(&store)
            .put("ds4vision", "heretic", &[0, 0, 0x80, 0x3f], false)
            .unwrap();
        let keys = model_keys(Some("ds4vision"), Path::new("/p/ds4vision.gguf"));
        assert_eq!(directions_in(&store, &keys).unwrap(), ["heretic"]);
        let file = materialize_in(&store, &dir.join("cache"), &keys, "heretic").unwrap();
        assert_eq!(std::fs::read(file).unwrap(), [0, 0, 0x80, 0x3f]);
        let err = materialize_in(&store, &dir.join("cache"), &keys, "nope").unwrap_err();
        assert!(err.contains("it has: heretic"), "{err}");
        assert!(parse_steering(&serde_json::json!({"file": "/v"})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

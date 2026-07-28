// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Content-addressed store for converted documents.
//!
//! Mirrors [`crate::imagepaste`]'s `~/.plank/image-cache` in layout and LRU
//! pruning. The key is the *source* document's content hash, so re-reading an
//! unchanged PDF never re-parses it, and editing one converts it afresh
//! without an invalidation step.

use std::path::{Path, PathBuf};

/// Maximum converted documents kept on disk before the oldest are dropped.
const MAX_CACHED_DOCS: usize = 64;

/// `~/.plank/doc-cache`, the sibling of `~/.plank/image-cache`.
pub(crate) fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".plank").join("doc-cache"))
}

/// Path a document with content hash `hash` converts to.
pub(crate) fn entry_path(dir: &Path, hash: &str) -> PathBuf {
    dir.join(format!("{hash}.md"))
}

/// Stores `markdown` under `hash`, returning the path it can be read from.
///
/// A cache that cannot be written is not an error: the caller falls back to a
/// temporary file, so a read-only `$HOME` costs a re-parse, not a failure.
pub(crate) fn store(hash: &str, markdown: &str) -> Option<PathBuf> {
    let dir = cache_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    let path = entry_path(&dir, hash);
    if !path.exists() {
        std::fs::write(&path, markdown).ok()?;
        prune(&dir);
    }
    Some(path)
}

/// Returns the cached conversion for `hash`, if one is on disk.
///
/// Touches the entry's mtime so [`prune`] treats a re-read as recent use;
/// without this the cache would evict by age of *conversion* rather than by
/// age of last access.
pub(crate) fn lookup(hash: &str) -> Option<PathBuf> {
    let path = entry_path(&cache_dir()?, hash);
    if path.is_file() {
        let _ = std::fs::File::open(&path).map(|f| f.set_modified(std::time::SystemTime::now()));
        Some(path)
    } else {
        None
    }
}

/// Keeps the newest [`MAX_CACHED_DOCS`] entries, deleting the oldest extras.
fn prune(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            meta.is_file().then_some((meta.modified().ok()?, e.path()))
        })
        .collect();
    if files.len() <= MAX_CACHED_DOCS {
        return;
    }
    files.sort_by_key(|(when, _)| std::cmp::Reverse(*when));
    for (_, path) in files.into_iter().skip(MAX_CACHED_DOCS) {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruning keeps the newest entries and drops the rest by mtime.
    #[test]
    fn prune_keeps_newest() {
        let dir = std::env::temp_dir().join(format!("plank-doccache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = std::time::SystemTime::now();
        for i in 0..(MAX_CACHED_DOCS + 5) {
            let path = entry_path(&dir, &format!("{i:064}"));
            std::fs::write(&path, "x").unwrap();
            let when = base - std::time::Duration::from_secs((MAX_CACHED_DOCS + 5 - i) as u64);
            std::fs::File::open(&path).unwrap().set_modified(when).ok();
        }
        prune(&dir);
        let left = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(left, MAX_CACHED_DOCS);
        // The newest entry (highest index) survives; the oldest does not.
        assert!(entry_path(&dir, &format!("{:064}", MAX_CACHED_DOCS + 4)).exists());
        assert!(!entry_path(&dir, &format!("{:064}", 0)).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

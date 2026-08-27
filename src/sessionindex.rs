// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Cross-session content search (`/search`, M6): a hand-rolled inverted index
//! over the metadata-cache directory.
//!
//! plank rewrites a session file in place on every save, so caching by id
//! alone is unsound — the same reason `insights` validates its per-session
//! metadata cache by size and mtime. This index reuses that discipline
//! verbatim: a session whose source stamp changed is re-indexed wholesale.
//!
//! The backend is deliberately a hand-rolled index over
//! `~/.plank/usage-data/session-index/<id>.json` rather than SQLite: plank has
//! no SQLite dependency today, and at a few hundred sessions a per-session
//! JSON file answers a query in milliseconds without a new C dependency or a
//! `build.rs` interaction. The index is human-facing only — nothing here
//! reaches the model (M8's `recall` tool is a separate, gated milestone).

use serde::{Deserialize, Serialize};

/// One indexed session: the source stamp for validation, the project key for
/// scoping, and the transcript text to search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Size of the session file when indexed, for size+mtime validation.
    pub src_size: u64,
    /// Mtime of the session file when indexed, for size+mtime validation.
    pub src_mtime: u64,
    /// Project key (`session::project_key` of the session's `cwd`), for
    /// workspace-scoped search.
    pub project_key: String,
    /// Session title, shown in hits.
    pub title: String,
    /// Creation time in unix seconds, for the age shown in hits.
    pub created_at: u64,
    /// The full transcript text, searched by substring.
    pub text: String,
}

/// One search hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// Session id, offered for `/resume`.
    pub session_id: String,
    /// Session title.
    pub title: String,
    /// Creation time in unix seconds.
    pub created_at: u64,
    /// A snippet of the matching text around the query.
    pub snippet: String,
}

/// Root of the index files, `~/.plank/usage-data/session-index`.
#[must_use]
pub fn index_dir() -> std::path::PathBuf {
    crate::insights::usage_dir().join("session-index")
}

/// Path of one session's index file.
#[must_use]
pub fn index_path(id: &str) -> std::path::PathBuf {
    index_dir().join(format!("{id}.json"))
}

/// Mtime of a file in unix seconds, or 0 when unavailable.
fn mtime_secs(path: &std::path::Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

/// Builds or refreshes the index: a session whose source stamp changed is
/// re-indexed wholesale. Returns how many sessions were (re)indexed.
///
/// # Errors
/// Returns a message when the store cannot be listed or a session loaded.
pub fn build(
    store: &crate::session::SessionStore,
    root: &std::path::Path,
) -> Result<usize, String> {
    let entries = store.list().map_err(|e| e.to_string())?;
    let mut indexed = 0usize;
    for entry in entries {
        let mtime = mtime_secs(&entry.path);
        let path = root.join(format!("{}.json", entry.id));
        let cached = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<IndexEntry>(&s).ok())
            .filter(|e| e.src_size == entry.file_size && e.src_mtime == mtime);
        if cached.is_some() {
            continue;
        }
        let session = store.load(&entry.id).map_err(|e| e.to_string())?;
        let text = session
            .transcript
            .iter()
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let project_key = crate::session::project_key(std::path::Path::new(&session.cwd));
        let index = IndexEntry {
            src_size: entry.file_size,
            src_mtime: mtime,
            project_key,
            title: entry.title.clone(),
            created_at: entry.created_at,
            text,
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        if let Ok(json) = serde_json::to_string(&index) {
            let _ = std::fs::write(&path, json);
            indexed += 1;
        }
    }
    Ok(indexed)
}

/// Searches the index for `query`, scoped to `project_key` unless `all` is
/// true. Returns hits newest-first, each with a snippet around the match.
#[must_use]
pub fn search(
    query: &str,
    project_key: Option<&str>,
    all: bool,
    root: &std::path::Path,
) -> Vec<Hit> {
    let query = query.trim();
    let mut hits = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return hits;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(id) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(index) = serde_json::from_str::<IndexEntry>(&text) else {
            continue;
        };
        if !all && index.project_key != project_key.unwrap_or_default() {
            continue;
        }
        let Some(pos) = index.text.find(query) else {
            continue;
        };
        let snippet = snippet_of(&index.text, pos, query.len());
        hits.push(Hit {
            session_id: id,
            title: index.title,
            created_at: index.created_at,
            snippet,
        });
    }
    hits.sort_by_key(|h| std::cmp::Reverse(h.created_at));
    hits
}

/// A snippet of `text` around `pos`, clipped to a readable width.
fn snippet_of(text: &str, pos: usize, query_len: usize) -> String {
    const WIDTH: usize = 120;
    let start = pos.saturating_sub(WIDTH / 3);
    let end = (pos + query_len + WIDTH / 3).min(text.len());
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(&text[start..end]);
    if end < text.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Message, Session};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-sessionindex-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    fn write_session(store: &crate::session::SessionStore, id: &str, text: &str) {
        let mut s = Session::new();
        s.id = id.to_string();
        s.cwd = "/tmp/proj".to_string();
        s.push(Message::user(text));
        store.save(&mut s).expect("save");
    }

    #[test]
    fn indexing_is_idempotent() {
        let dir = scratch("idempotent");
        let root = dir.join("index");
        let store = crate::session::SessionStore::open(&dir).expect("open");
        write_session(&store, "s1", "hello world");
        let first = build(&store, &root).expect("build");
        assert_eq!(first, 1, "one session indexed");
        let second = build(&store, &root).expect("build");
        assert_eq!(second, 0, "unchanged stamp: nothing re-indexed");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_session_rewritten_in_place_is_reindexed() {
        let dir = scratch("reindex");
        let root = dir.join("index");
        let store = crate::session::SessionStore::open(&dir).expect("open");
        write_session(&store, "s1", "first version");
        build(&store, &root).expect("build");
        // Rewrite the same session with new content; the stamp changes.
        write_session(&store, "s1", "second version with a needle");
        let indexed = build(&store, &root).expect("build");
        assert_eq!(indexed, 1, "changed stamp: re-indexed wholesale");
        let hits = search(
            "needle",
            Some(&crate::session::project_key(std::path::Path::new(
                "/tmp/proj",
            ))),
            false,
            &root,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "s1");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_scopes_by_project_key_unless_all() {
        let dir = scratch("scope");
        let root = dir.join("index");
        let store = crate::session::SessionStore::open(&dir).expect("open");
        let mut a = Session::new();
        a.id = "a".to_string();
        a.cwd = "/proj/a".to_string();
        a.push(Message::user("shared needle"));
        store.save(&mut a).expect("save a");
        let mut b = Session::new();
        b.id = "b".to_string();
        b.cwd = "/proj/b".to_string();
        b.push(Message::user("shared needle"));
        store.save(&mut b).expect("save b");
        build(&store, &root).expect("build");
        let scoped = search(
            "needle",
            Some(&crate::session::project_key(std::path::Path::new(
                "/proj/a",
            ))),
            false,
            &root,
        );
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].session_id, "a");
        let all = search("needle", None, true, &root);
        assert_eq!(all.len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}

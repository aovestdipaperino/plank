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

/// Most bytes of archived (no-longer-live) conversation kept per session.
/// Archived text only grows when a transcript loses messages, and only
/// conversation is eligible (see [`worth_archiving`]), so this is a backstop
/// against a pathological session rather than a routine limit.
const ARCHIVE_MAX_BYTES: usize = 65_536;

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
    /// The current transcript, one entry per message.
    #[serde(default)]
    pub messages: Vec<String>,
    /// Conversation an earlier version of this transcript carried and the
    /// current one does not: compaction replaces the transcript with a summary
    /// plus a tail, and re-indexing is wholesale, so without this the dropped
    /// text would stop being findable. Keeping it is what makes `/search`
    /// compaction-proof. Oldest-first; trimmed from the front at
    /// [`ARCHIVE_MAX_BYTES`].
    #[serde(default)]
    pub archived: Vec<String>,
    /// Legacy field: the joined transcript written by index files that predate
    /// `messages`. Read for migration, never written again.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
}

impl IndexEntry {
    /// The current transcript messages, migrating a legacy `text` entry.
    fn current(&self) -> Vec<&str> {
        if self.messages.is_empty() && !self.text.is_empty() {
            return vec![self.text.as_str()];
        }
        self.messages.iter().map(String::as_str).collect()
    }

    /// Everything this entry can answer a query from: archived history first
    /// (oldest), then the live transcript.
    fn searchable(&self) -> String {
        let mut parts: Vec<&str> = self.archived.iter().map(String::as_str).collect();
        parts.extend(self.current());
        parts.join("\n")
    }
}

/// Whether a message that has left the transcript is worth keeping in the
/// archive. Tool results are excluded deliberately: microcompact clears large
/// result bodies on most turns, so archiving them would make the index grow
/// without bound and turn it into a second copy of every tool output ever
/// produced. Tool output is re-derivable by rerunning the tool; the
/// conversation is not.
fn worth_archiving(text: &str) -> bool {
    let t = text.trim_start();
    !t.starts_with("<tool_result>") && !t.is_empty()
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
        let prior = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<IndexEntry>(&s).ok());
        if prior
            .as_ref()
            .is_some_and(|e| e.src_size == entry.file_size && e.src_mtime == mtime)
        {
            continue;
        }
        let session = store.load(&entry.id).map_err(|e| e.to_string())?;
        let messages: Vec<String> = session
            .transcript
            .iter()
            .map(|m| m.text.clone())
            .collect::<Vec<_>>();
        // Compaction-proofing: anything the previous version carried that the
        // new transcript has dropped moves to the archive, so a query still
        // finds it after the text is gone from the session file.
        let archived = merge_archive(prior.as_ref(), &messages);
        let project_key = crate::session::project_key(std::path::Path::new(&session.cwd));
        let index = IndexEntry {
            src_size: entry.file_size,
            src_mtime: mtime,
            project_key,
            title: entry.title.clone(),
            created_at: entry.created_at,
            messages,
            archived,
            text: String::new(),
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

/// The archive for a re-indexed session: whatever the previous entry already
/// archived, plus any of its conversation messages the new transcript no
/// longer carries. Oldest-first, de-duplicated, and trimmed from the front to
/// [`ARCHIVE_MAX_BYTES`].
fn merge_archive(prior: Option<&IndexEntry>, current: &[String]) -> Vec<String> {
    let Some(prior) = prior else {
        return Vec::new();
    };
    let live: std::collections::HashSet<&str> = current.iter().map(String::as_str).collect();
    let mut archived: Vec<String> = prior.archived.clone();
    for old in prior.current() {
        if live.contains(old) || !worth_archiving(old) {
            continue;
        }
        if !archived.iter().any(|a| a == old) {
            archived.push(old.to_string());
        }
    }
    // Trim oldest-first until the archive fits its budget.
    let mut total: usize = archived.iter().map(String::len).sum();
    let mut drop_to = 0usize;
    while total > ARCHIVE_MAX_BYTES && drop_to < archived.len() {
        total -= archived[drop_to].len();
        drop_to += 1;
    }
    archived.drain(..drop_to);
    archived
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
        // Archived history is searched alongside the live transcript: that is
        // what keeps a hit findable after compaction dropped its text.
        let haystack = index.searchable();
        let Some(pos) = haystack.find(query) else {
            continue;
        };
        let snippet = snippet_of(&haystack, pos, query.len());
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
    // Byte offsets: the window edges can land inside a multibyte character,
    // so each is snapped to a boundary before slicing.
    let start = crate::session::floor_char_boundary(text, pos.saturating_sub(WIDTH / 3));
    let end = crate::session::ceil_char_boundary(text, pos.saturating_add(query_len + WIDTH / 3));
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
    fn text_dropped_by_compaction_is_still_findable() {
        // M6's headline claim: compaction replaces the transcript with a
        // summary plus a tail, and re-indexing is wholesale, so the index must
        // archive what the transcript lost or the history stops being findable.
        let dir = scratch("compaction-proof");
        let root = dir.join("index");
        let store = crate::session::SessionStore::open(&dir).expect("open");
        let key = crate::session::project_key(std::path::Path::new("/tmp/proj"));

        write_session(&store, "s1", "a precompactionmarker worth finding");
        build(&store, &root).expect("build");
        assert_eq!(
            search("precompactionmarker", Some(&key), false, &root).len(),
            1,
            "findable before compaction"
        );

        // Compaction: the marker survives in neither summary nor tail.
        write_session(
            &store,
            "s1",
            "Compacted session summary: the user said hello",
        );
        build(&store, &root).expect("build");
        let hits = search("precompactionmarker", Some(&key), false, &root);
        assert_eq!(hits.len(), 1, "still findable after compaction");
        assert_eq!(hits[0].session_id, "s1");
        assert!(
            hits[0].snippet.contains("precompactionmarker"),
            "the snippet quotes the archived text: {:?}",
            hits[0].snippet
        );
        assert_eq!(
            search("Compacted", Some(&key), false, &root).len(),
            1,
            "the live summary is findable too"
        );

        // A second compaction must not lose the first one's history.
        write_session(&store, "s1", "Compacted again: nothing of note");
        build(&store, &root).expect("build");
        assert_eq!(
            search("precompactionmarker", Some(&key), false, &root).len(),
            1,
            "archive survives repeated compaction"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cleared_tool_results_are_not_archived() {
        // Microcompact clears large tool-result bodies on most turns. Archiving
        // those would make the index grow without bound and duplicate every
        // tool output ever produced, so only conversation is archived.
        let dir = scratch("no-tool-archive");
        let root = dir.join("index");
        let store = crate::session::SessionStore::open(&dir).expect("open");
        let key = crate::session::project_key(std::path::Path::new("/tmp/proj"));

        let mut s = Session::new();
        s.id = "s1".to_string();
        s.cwd = "/tmp/proj".to_string();
        s.push(Message::user(
            "<tool_result>a giant toolneedle payload</tool_result>",
        ));
        s.push(Message::user("keep this conversationneedle"));
        store.save(&mut s).expect("save");
        build(&store, &root).expect("build");
        assert_eq!(search("toolneedle", Some(&key), false, &root).len(), 1);

        // Microcompact replaces the body with the stub and the turn moves on.
        let mut s2 = Session::new();
        s2.id = "s1".to_string();
        s2.cwd = "/tmp/proj".to_string();
        s2.push(Message::user(
            "<tool_result>[old tool result cleared]</tool_result>",
        ));
        store.save(&mut s2).expect("save");
        build(&store, &root).expect("build");

        assert_eq!(
            search("toolneedle", Some(&key), false, &root).len(),
            0,
            "a cleared tool result is not archived"
        );
        assert_eq!(
            search("conversationneedle", Some(&key), false, &root).len(),
            1,
            "conversation that left the transcript is archived"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_archive_is_capped() {
        // Growth backstop: the archive is trimmed oldest-first to a budget, so
        // a pathological session cannot grow its index entry without bound.
        let oldest = "x".repeat(ARCHIVE_MAX_BYTES);
        let prior = IndexEntry {
            src_size: 0,
            src_mtime: 0,
            project_key: String::new(),
            title: String::new(),
            created_at: 0,
            messages: vec!["newer conversation".to_string()],
            archived: vec![oldest],
            text: String::new(),
        };
        // The live message is gone, so it joins the archive and pushes the
        // total past the cap.
        let merged = merge_archive(Some(&prior), &["something else".to_string()]);
        let total: usize = merged.iter().map(String::len).sum();
        assert!(
            total <= ARCHIVE_MAX_BYTES,
            "archive trimmed to the budget, got {total}"
        );
        assert!(
            merged.iter().any(|m| m == "newer conversation"),
            "the newest entry survives the trim"
        );
    }

    #[test]
    fn a_legacy_index_entry_still_searches() {
        // Index files written before `messages` carry a joined `text` field.
        // They must keep answering queries and migrate on the next rebuild.
        let legacy = IndexEntry {
            src_size: 0,
            src_mtime: 0,
            project_key: String::new(),
            title: String::new(),
            created_at: 0,
            messages: Vec::new(),
            archived: Vec::new(),
            text: "a legacyneedle in the old joined field".to_string(),
        };
        assert!(legacy.searchable().contains("legacyneedle"));
        // And its content is archived when the transcript moves on.
        let merged = merge_archive(Some(&legacy), &["replacement".to_string()]);
        assert!(
            merged.iter().any(|m| m.contains("legacyneedle")),
            "legacy text is archived, not silently dropped: {merged:?}"
        );
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

    /// The snippet window is measured in bytes; when either edge lands inside
    /// an em dash or emoji it must snap to a character boundary, not panic.
    #[test]
    fn snippet_window_snaps_to_char_boundaries() {
        let text = format!("{}needle{}", "\u{2014}".repeat(30), "\u{1F600}".repeat(30));
        let pos = text.find("needle").unwrap();
        let snippet = snippet_of(&text, pos, "needle".len());
        assert!(snippet.contains("needle"));
        assert!(snippet.starts_with('\u{2026}') && snippet.ends_with('\u{2026}'));
        // And a hit at the very start or end of the text is fine too.
        let _ = snippet_of(&text, 0, 3);
        let _ = snippet_of(&text, text.len() - 4, 4);
    }
}

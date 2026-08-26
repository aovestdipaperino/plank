// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Per-message feedback ratings, stored in a sidecar.
//!
//! **This file never enters the transcript, never enters model context, and
//! never enters the KV.** A rating is a fact *about* the session, not a fact in
//! it; putting it in the transcript would both pollute the model's context and
//! break the KV prefix. Nobody should "fix" that by accident — the storage
//! lives under `~/.plank/usage-data/feedback/`, deliberately outside the
//! session file, and is consumed only by `/insights`.
//!
//! Records are append-only JSONL, one per rating, keyed by `(session id, turn
//! ordinal, digest)` where `digest` is the SHA-1 of the turn text at rating
//! time. Ordinals are append-stable but not branch-stable (a compaction or
//! rollback renumbers the transcript), so a digest mismatch means the rating
//! no longer has a subject — it is rendered as orphaned, never reattributed.

use serde::{Deserialize, Serialize};

/// One rating of an assistant turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rating {
    /// Index of the rated turn in the transcript at rating time.
    pub ordinal: usize,
    /// SHA-1 of the turn text at rating time, so a renumbered transcript can
    /// be detected rather than silently misattributed.
    pub digest: String,
    /// `true` = positive (`/rate +`), `false` = negative (`/rate -`).
    pub positive: bool,
    /// Optional free-text note.
    pub note: String,
    /// Wall-clock second the rating was recorded.
    pub at: u64,
}

/// Directory holding every session's feedback sidecar, under the same
/// `usage-data` root and sweep as `insights`'s metadata cache.
#[must_use]
pub fn feedback_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join("feedback")
}

/// Path of one session's append-only feedback file.
#[must_use]
pub fn feedback_path(root: &std::path::Path, session_id: &str) -> std::path::PathBuf {
    feedback_dir(root).join(format!("{session_id}.jsonl"))
}

/// Appends one rating to the session's sidecar, creating parents as needed.
/// Append-only: a rating is never rewritten, so a later compaction cannot
/// retroactively change what was recorded.
///
/// # Errors
/// Returns an I/O error when the sidecar cannot be created or appended to.
pub fn record(root: &std::path::Path, session_id: &str, rating: &Rating) -> std::io::Result<()> {
    use std::io::Write as _;
    let path = feedback_path(root, session_id);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::to_string(rating)?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(line.as_bytes())
}

/// Reads every rating recorded for one session, in append order.
#[must_use]
pub fn load(root: &std::path::Path, session_id: &str) -> Vec<Rating> {
    let Ok(text) = std::fs::read_to_string(feedback_path(root, session_id)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str::<Rating>(l).ok())
        .collect()
}

/// Whether a rating still has a subject: the transcript's turn at `ordinal`
/// must digest to the rating's `digest`. A compaction or rollback renumbers
/// the transcript, so a mismatch means the rating no longer points at the turn
/// it rated — it is orphaned, never reattributed.
#[must_use]
pub fn is_orphaned(transcript: &[crate::session::Message], rating: &Rating) -> bool {
    match transcript.get(rating.ordinal) {
        None => true,
        Some(m) => crate::session::sha1_hex(m.text.as_bytes()) != rating.digest,
    }
}

/// Reads every session's ratings, as `(session_id, ratings)` pairs, for
/// `/insights`. A malformed line is skipped rather than fatal.
#[must_use]
pub fn load_all(root: &std::path::Path) -> Vec<(String, Vec<Rating>)> {
    let Ok(entries) = std::fs::read_dir(feedback_dir(root)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_suffix(".jsonl") else {
            continue;
        };
        out.push((id.to_string(), load(root, id)));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-feedback-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn a_rating_survives_a_save_load_round_trip() {
        let root = scratch("roundtrip");
        let r = Rating {
            ordinal: 3,
            digest: "abc".to_string(),
            positive: true,
            note: "good".to_string(),
            at: 1_700_000_000,
        };
        record(&root, "sess1", &r).expect("record");
        let loaded = load(&root, "sess1");
        assert_eq!(loaded, vec![r]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn records_are_append_only_and_ordered() {
        let root = scratch("append");
        let a = Rating {
            ordinal: 0,
            digest: "a".to_string(),
            positive: true,
            note: String::new(),
            at: 1,
        };
        let b = Rating {
            ordinal: 1,
            digest: "b".to_string(),
            positive: false,
            note: String::new(),
            at: 2,
        };
        record(&root, "s", &a).expect("a");
        record(&root, "s", &b).expect("b");
        let loaded = load(&root, "s");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].digest, "a");
        assert_eq!(loaded[1].digest, "b");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_rating_is_orphaned_when_the_transcript_renumbers() {
        // A compaction replaces the transcript wholesale, so the turn that was
        // at `ordinal` is gone; the digest mismatch must mark it orphaned, not
        // reattribute it to whatever now sits at that index.
        let original = vec![
            crate::session::Message::user("user"),
            crate::session::Message::assistant("answer one"),
        ];
        let rating = Rating {
            ordinal: 1,
            digest: crate::session::sha1_hex("answer one".as_bytes()),
            positive: true,
            note: String::new(),
            at: 1,
        };
        assert!(
            !is_orphaned(&original, &rating),
            "same transcript: not orphaned"
        );
        // After a rewrite, index 1 holds a different turn.
        let rewritten = vec![
            crate::session::Message::user("user"),
            crate::session::Message::assistant("answer two"),
        ];
        assert!(is_orphaned(&rewritten, &rating), "renumbered: orphaned");
        // An out-of-range ordinal is orphaned too.
        assert!(is_orphaned(
            &original,
            &Rating {
                ordinal: 99,
                ..rating.clone()
            }
        ));
    }

    #[test]
    fn load_all_groups_by_session() {
        let root = scratch("all");
        record(
            &root,
            "s1",
            &Rating {
                ordinal: 0,
                digest: "x".to_string(),
                positive: true,
                note: String::new(),
                at: 1,
            },
        )
        .expect("s1");
        record(
            &root,
            "s2",
            &Rating {
                ordinal: 0,
                digest: "y".to_string(),
                positive: false,
                note: String::new(),
                at: 2,
            },
        )
        .expect("s2");
        let all = load_all(&root);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "s1");
        assert_eq!(all[1].0, "s2");
        std::fs::remove_dir_all(&root).ok();
    }
}

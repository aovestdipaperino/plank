// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Output spill: bounded preview plus a retrieval locator.
//!
//! Oversized tool output is persisted out-of-band under
//! `~/.plank/spill/<session-id>/<n>.txt`, and the inline result the model sees
//! is replaced by a bounded preview plus a locator it can use to retrieve more
//! via the existing `more` tool. This is a **post-dispatch policy**, not a
//! per-tool concern: every tool gets the behaviour, including MCP tools.
//!
//! The full payload stays durable so `/export` and post-hoc inspection can
//! still see what the tool actually returned. Spill blobs are swept by the
//! existing GC policy (`SessionStore::sweep` / `kvgc::SweepPolicy`) under the
//! same TTL and byte-budget that governs `kvcache` — there is deliberately no
//! second GC.

use std::path::PathBuf;

/// How much tool output to keep inline before spilling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillPolicy {
    /// A result larger than this many bytes is spilled.
    pub max_bytes: usize,
    /// How many bytes of the full payload stay inline as the preview.
    pub preview_bytes: usize,
}

/// One spilled payload, for the `more` continuation tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spilled {
    /// The spill id, `"<session-id>/<n>"`, shown in the locator line.
    pub id: String,
    /// Total payload size in bytes.
    pub bytes: usize,
    /// Path of the spilled file.
    pub path: PathBuf,
    /// Byte offset the next `more` chunk starts at.
    pub offset: usize,
}

/// Root of every session's spill files, `~/.plank/spill`.
#[must_use]
pub fn spill_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".plank").join("spill")
}

/// Applies the spill policy to one tool result: writes the full payload to
/// disk when it exceeds `max_bytes`, and returns a bounded preview plus a
/// locator the model can use with `more`. Below the cap the result passes
/// through untouched.
#[must_use]
pub fn apply(
    policy: &SpillPolicy,
    session_id: &str,
    tool: &str,
    result: String,
) -> (String, Option<Spilled>) {
    if result.len() <= policy.max_bytes {
        return (result, None);
    }
    let dir = spill_dir().join(session_id);
    let _ = std::fs::create_dir_all(&dir);
    // Next free index: the first `<n>.txt` that does not exist.
    let mut n = 0usize;
    let path = loop {
        let candidate = dir.join(format!("{n}.txt"));
        if !candidate.exists() {
            break candidate;
        }
        n += 1;
    };
    if std::fs::write(&path, result.as_bytes()).is_err() {
        // Spill failed; keep the full result inline rather than lose it.
        return (result, None);
    }
    let bytes = result.len();
    let preview = result
        .chars()
        .take(policy.preview_bytes)
        .collect::<String>();
    let id = format!("{session_id}/{n}");
    let spilled = Spilled {
        id: id.clone(),
        bytes,
        path,
        offset: policy.preview_bytes,
    };
    // Model-facing framing, built with `format!` (never a `\`-continued
    // literal — the leading-whitespace trap in CLAUDE.md). Reuses the
    // fixture-blessed `[Read truncated at line N of M. continue_offset=K. ...]`
    // shape; this is a deliberate new site, noted in FINDINGS.md.
    let locator = format!(
        "[Output truncated at {preview_bytes} bytes of {bytes}. continue_offset={offset}. \
         Call more with count={count} to read the next chunk.]",
        preview_bytes = policy.preview_bytes,
        bytes = bytes,
        offset = spilled.offset,
        count = 4096,
    );
    let mut out = preview;
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&locator);
    out.push('\n');
    let _ = tool;
    (out, Some(spilled))
}

/// Reads the full payload of a spilled result by id (`"<session-id>/<n>"`).
#[must_use]
pub fn read_spill(id: &str) -> Option<String> {
    let path = spill_dir().join(id).with_extension("txt");
    std::fs::read_to_string(path).ok()
}

/// Sweeps spill blobs under the existing GC policy: files older than the
/// session TTL are removed, then the byte budget is enforced by evicting the
/// oldest survivors. Returns the bytes reclaimed.
///
/// Spill files live under `~/.plank/spill/<session-id>/<n>.txt`, so the walk
/// recurses into the per-session subdirectories.
#[must_use]
pub fn sweep(policy: &crate::kvgc::SweepPolicy, now: u64) -> u64 {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_txt(&spill_dir(), &mut files);
    let mut freed = 0u64;
    let mut survivors: Vec<(u64, PathBuf, u64)> = Vec::new(); // (mtime, path, bytes)
    for path in files {
        let mtime = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
        if policy.ttl_session_secs > 0 && now.saturating_sub(mtime) > policy.ttl_session_secs {
            if std::fs::remove_file(&path).is_ok() {
                freed = freed.saturating_add(bytes);
            }
        } else {
            survivors.push((mtime, path, bytes));
        }
    }
    // Byte budget: evict oldest survivors until the total fits.
    if policy.max_bytes > 0 {
        let total: u64 = survivors.iter().map(|s| s.2).sum();
        if total > policy.max_bytes {
            survivors.sort_by_key(|s| s.0);
            let mut over = total - policy.max_bytes;
            for (_, path, bytes) in survivors {
                if over == 0 {
                    break;
                }
                if std::fs::remove_file(&path).is_ok() {
                    freed = freed.saturating_add(bytes);
                    over = over.saturating_sub(bytes);
                }
            }
        }
    }
    freed
}

/// Collects every `.txt` spill file under `dir`, recursing into subdirectories.
fn collect_txt(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_txt(&path, out);
        } else if path.extension().is_some_and(|e| e == "txt") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_results_pass_through_untouched() {
        let policy = SpillPolicy {
            max_bytes: 100,
            preview_bytes: 50,
        };
        let (out, spilled) = apply(&policy, "sess", "read", "small".to_string());
        assert_eq!(out, "small");
        assert!(spilled.is_none());
    }

    #[test]
    fn oversized_results_spill_to_a_preview_plus_locator() {
        let policy = SpillPolicy {
            max_bytes: 10,
            preview_bytes: 5,
        };
        let big = "x".repeat(100);
        let (out, spilled) = apply(&policy, "sess", "mcp_call", big.clone());
        let spilled = spilled.expect("spilled");
        assert!(out.contains("[Output truncated at 5 bytes of 100."));
        assert!(out.contains("continue_offset=5."));
        assert!(out.contains("Call more with count=4096"));
        // The full payload is recoverable by id.
        assert_eq!(read_spill(&spilled.id).as_deref(), Some(big.as_str()));
        std::fs::remove_dir_all(spill_dir().join("sess")).ok();
    }

    #[test]
    fn spill_files_are_numbered_incrementally() {
        let policy = SpillPolicy {
            max_bytes: 1,
            preview_bytes: 1,
        };
        let (_, a) = apply(&policy, "num", "read", "aaaa".to_string());
        let (_, b) = apply(&policy, "num", "read", "bbbb".to_string());
        assert_eq!(a.expect("a").id, "num/0");
        assert_eq!(b.expect("b").id, "num/1");
        std::fs::remove_dir_all(spill_dir().join("num")).ok();
    }

    #[test]
    fn sweep_removes_old_blobs_and_enforces_the_byte_budget() {
        let policy = SpillPolicy {
            max_bytes: 1,
            preview_bytes: 1,
        };
        let (_, _) = apply(&policy, "sweep", "read", "aaaa".to_string());
        let (_, _) = apply(&policy, "sweep", "read", "bbbb".to_string());
        let dir = spill_dir().join("sweep");
        let now = crate::kvmeta::now_secs();
        // TTL pass: nothing is old yet.
        let sweep_policy = crate::kvgc::SweepPolicy {
            ttl_session_secs: 3600,
            ttl_tier_secs: 3600,
            max_bytes: 0,
        };
        assert_eq!(sweep(&sweep_policy, now), 0);
        // Byte budget: cap below the total evicts the oldest.
        let tight = crate::kvgc::SweepPolicy {
            ttl_session_secs: 3600,
            ttl_tier_secs: 3600,
            max_bytes: 4,
        };
        let freed = sweep(&tight, now);
        assert!(freed > 0, "budget pass must reclaim bytes");
        let remaining = std::fs::read_dir(&dir).map_or(0, std::iter::Iterator::count);
        assert!(remaining < 2, "at least one blob evicted");
        std::fs::remove_dir_all(&dir).ok();
    }
}

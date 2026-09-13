// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! A cheap witness that the working tree changed, for tools whose writes the
//! agent cannot see directly.
//!
//! `write` and `edit` announce themselves by setting
//! [`ToolContext::last_written`](crate::tools::ToolContext::last_written), so
//! the no-progress guard can trust them. Every other way a tool call mutates
//! the workspace — `sed -i`, `cargo fmt`, a codegen script, `git apply`, an
//! MCP server's own write tool — is opaque: the agent sees an exit status and
//! some output, which says a command *ran*, not that the task advanced. That
//! is the distinction `docs/LOOP-FINDINGS.md` ("An attempted mutation is not
//! progress") was written to protect, and it is why those calls were left out
//! of the budget reset entirely.
//!
//! Leaving them out costs the honest case: a `bash` call that really does
//! rewrite twenty files counts as nothing, and a turn doing real work through
//! the shell can trip the 32 KiB guard. This module supplies the "separate
//! workspace-mutation witness" that finding asked for, without assuming
//! anything from an exit status: take a digest of the git working-tree state
//! before the opaque call and after it, and let a *difference* be the
//! evidence. No difference, no progress — a failed `sed -i` and a successful
//! `cargo test` still reset nothing.
//!
//! This is observation, not interception. macOS offers no supported
//! user-space hook on file writes (see `docs/FILE-SYSTEM-HOOK.md` for why,
//! and for the `FSEvents` design that would widen the net); an after-the-fact
//! comparison is all the guard needs, because it only ever asks its question
//! once the call has returned.
//!
//! Deliberate limits, each of which costs at most a missed budget reset:
//!
//! - **No repository, no witness.** [`capture`] returns `None` outside a git
//!   checkout, and a `None` on either side compares as "nothing observed", so
//!   the guard behaves exactly as it did before this module existed.
//! - **Ignored paths do not count.** A write to `target/` is invisible here,
//!   which is the right answer: build output is not the work.
//! - **Untracked directories are not recursed.** Creating `src/new/mod.rs`
//!   shows up as the new `src/new/` entry, which is difference enough, and
//!   skipping the walk keeps the cost near `git status`'s.
//! - **An async `bash` job that lands between polls** is attributed to
//!   whichever call happens to bracket the write, not to the call that
//!   started the job.

use std::hash::{Hash as _, Hasher as _};
use std::path::Path;

/// An opaque fingerprint of the working tree's git state at one instant.
///
/// Only ever compared against another digest from the same process: the hash
/// is not stable across builds and is not written to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeDigest(u64);

/// Fingerprints the working tree at `cwd`, or `None` when there is no
/// repository to look at (or libgit2 declines to walk it).
///
/// Folds in `HEAD` — so a `git commit` or a checkout reads as a change even
/// though it can leave the status list empty — then every dirty entry's path
/// and status bits, and for each one the file's length and modification time.
/// The stat is what makes an edit to an already-dirty file visible: its status
/// bits do not move, but its bytes do.
#[must_use]
pub fn capture(cwd: &Path) -> Option<TreeDigest> {
    let repo = git2::Repository::discover(cwd).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Ok(head) = repo.head().and_then(|h| h.peel_to_commit()) {
        head.id().as_bytes().hash(&mut hasher);
    }
    let workdir = repo.workdir()?.to_path_buf();
    let mut opts = git2::StatusOptions::new();
    opts.include_untracked(true)
        .recurse_untracked_dirs(false)
        .include_ignored(false)
        // Never write the index from a guard: this runs inside a tool
        // dispatch, and a stat-cache refresh racing the user's own git is not
        // a trade the no-progress budget is worth.
        .no_refresh(true);
    let statuses = repo.statuses(Some(&mut opts)).ok()?;
    for entry in statuses.iter() {
        // Non-UTF-8 paths are skipped rather than lossily converted: a name
        // that cannot be hashed stably is better left out than aliased onto
        // another one.
        let Ok(path) = entry.path() else { continue };
        path.hash(&mut hasher);
        entry.status().bits().hash(&mut hasher);
        if let Ok(meta) = std::fs::symlink_metadata(workdir.join(path)) {
            meta.len().hash(&mut hasher);
            if let Ok(mtime) = meta.modified() {
                mtime.hash(&mut hasher);
            }
        }
    }
    Some(TreeDigest(hasher.finish()))
}

/// True when `before` and `after` are both present and differ.
///
/// A missing digest on either side means nothing was observed, which is not
/// the same as observing no change: outside a repository this is always
/// `false`, and the guard falls back to `last_written` alone.
#[must_use]
pub fn changed(before: Option<TreeDigest>, after: Option<TreeDigest>) -> bool {
    match (before, after) {
        (Some(a), Some(b)) => a != b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repository with one committed file, so `HEAD` resolves and the tree
    /// starts clean.
    fn repo_with_commit(dir: &Path) {
        let repo = git2::Repository::init(dir).unwrap();
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("tracked.txt")).unwrap();
        index.write().unwrap();
        let tree = repo.find_tree(index.write_tree().unwrap()).unwrap();
        let sig = git2::Signature::now("t", "t@t").unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "init", &tree, &[])
            .unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-treedigest-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        // The temp dir is a symlink on macOS; canonicalize so the path the
        // test hands `capture` is the one libgit2 reports as the workdir.
        std::fs::canonicalize(&dir).unwrap()
    }

    #[test]
    fn outside_a_repository_there_is_no_witness() {
        let dir = scratch("no-repo");
        // `discover` walks upward, so only assert the contract that matters:
        // a `None` on either side never reads as a change.
        assert!(!changed(None, capture(&dir)));
        assert!(!changed(capture(&dir), None));
        assert!(!changed(None, None));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_quiet_command_leaves_the_digest_alone() {
        let dir = scratch("quiet");
        repo_with_commit(&dir);
        let before = capture(&dir);
        assert!(before.is_some());
        // Reading proves nothing happened, which is the whole point.
        let _ = std::fs::read_to_string(dir.join("tracked.txt")).unwrap();
        assert!(!changed(before, capture(&dir)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_new_file_moves_the_digest() {
        let dir = scratch("new-file");
        repo_with_commit(&dir);
        let before = capture(&dir);
        std::fs::write(dir.join("fresh.txt"), "made\n").unwrap();
        assert!(changed(before, capture(&dir)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn editing_an_already_dirty_file_moves_the_digest() {
        // The case a status-only digest misses: the entry's status bits read
        // `WT_MODIFIED` both before and after, so only the stat separates them.
        let dir = scratch("dirty-again");
        repo_with_commit(&dir);
        std::fs::write(dir.join("tracked.txt"), "two\n").unwrap();
        let before = capture(&dir);
        std::fs::write(dir.join("tracked.txt"), "three and longer\n").unwrap();
        assert!(changed(before, capture(&dir)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_commit_moves_the_digest_even_with_a_clean_status() {
        // Status is empty on both sides here; `HEAD` is what tells them apart.
        let dir = scratch("commit");
        repo_with_commit(&dir);
        let before = capture(&dir);
        let repo = git2::Repository::open(&dir).unwrap();
        std::fs::write(dir.join("tracked.txt"), "committed\n").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("tracked.txt")).unwrap();
        index.write().unwrap();
        let tree = repo.find_tree(index.write_tree().unwrap()).unwrap();
        let sig = git2::Signature::now("t", "t@t").unwrap();
        let parent = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "second", &tree, &[&parent])
            .unwrap();
        let after = capture(&dir);
        assert!(changed(before, after));
        // And the clean tree really is clean on both sides of the commit.
        assert!(!changed(after, capture(&dir)));
        std::fs::remove_dir_all(&dir).ok();
    }
}

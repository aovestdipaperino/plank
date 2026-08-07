// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! End-to-end worktree tests against a real `git` binary.
//!
//! The unit tests in `src/worktree.rs` cover the pure logic (slug rules,
//! naming, fail-closed accounting). These cover the half that only a real
//! repository can exercise: that `git worktree add` is invoked correctly, that
//! the resulting checkout is recognized as a worktree, that change accounting
//! reports what was actually done in it, and that removal cleans up both the
//! directory and the branch.

use std::path::{Path, PathBuf};
use std::process::Command;

use plank::hooks::Hooks;
use plank::worktree::{self, CreateOptions};

/// Runs a git command in `dir`, asserting it succeeded.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("spawn git {args:?}: {e}"));
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Creates a throwaway repository with one commit, named after `tag` so
/// concurrently running tests never share a directory.
fn temp_repo(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("plank-wt-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    git(&dir, &["init", "-q", "-b", "main"]);
    git(&dir, &["config", "user.email", "t@example.com"]);
    git(&dir, &["config", "user.name", "t"]);
    std::fs::write(dir.join("a.txt"), b"hello\n").unwrap();
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-qm", "init"]);
    dir
}

#[test]
fn a_worktree_is_created_resumed_and_removed() {
    let repo = temp_repo("lifecycle");
    let opts = CreateOptions::default();

    let created = worktree::get_or_create(&repo, "feature", &opts).expect("create");
    assert!(!created.existed, "first call must create");
    assert_eq!(created.path, worktree::path_for(&repo, "feature"));
    assert_eq!(created.branch, "worktree-feature");
    // The base commit came along, so this really is a checkout and not an
    // empty directory that merely has the right name.
    assert!(created.path.join("a.txt").is_file());
    assert!(worktree::worktree_head_sha(&created.path).is_some());

    // Second call takes the fast-resume path rather than creating again.
    let resumed = worktree::get_or_create(&repo, "feature", &opts).expect("resume");
    assert!(resumed.existed, "second call must resume");
    assert_eq!(resumed.path, created.path);

    // A worktree resolves back to the repository that owns it, which is what
    // keeps nested worktrees and project-scoped state out of trouble.
    assert_eq!(
        worktree::canonical_git_root(&created.path)
            .map(|p| p.canonicalize().unwrap())
            .as_deref(),
        Some(repo.canonicalize().unwrap().as_path())
    );

    let session =
        worktree::create_for_session(&repo, "feature", &Hooks::default(), &opts).expect("session");
    worktree::cleanup(&session, &Hooks::default()).expect("cleanup");
    assert!(!created.path.exists(), "removal must delete the directory");
    let branches = git(&repo, &["branch", "--list", "worktree-feature"]);
    assert!(branches.is_empty(), "removal must delete the branch");

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn change_accounting_sees_edits_and_commits() {
    let repo = temp_repo("changes");
    let opts = CreateOptions::default();
    let created = worktree::get_or_create(&repo, "work", &opts).expect("create");
    let head = worktree::worktree_head_sha(&created.path);

    // A fresh worktree holds nothing, so it is safe to throw away.
    let clean = worktree::count_changes(&created.path, head.as_deref()).expect("countable");
    assert!(
        clean.is_empty(),
        "a fresh worktree has no changes: {clean:?}"
    );
    assert!(!worktree::has_changes(&created.path, head.as_deref()));

    // An uncommitted edit is work that removal must not silently destroy.
    std::fs::write(created.path.join("a.txt"), b"edited\n").unwrap();
    let dirty = worktree::count_changes(&created.path, head.as_deref()).expect("countable");
    assert_eq!(dirty.files, vec!["a.txt".to_string()]);
    assert_eq!(dirty.commits, 0);
    assert!(worktree::has_changes(&created.path, head.as_deref()));

    // ...and so is a commit that exists nowhere else.
    git(&created.path, &["config", "user.email", "t@example.com"]);
    git(&created.path, &["config", "user.name", "t"]);
    git(&created.path, &["add", "-A"]);
    git(&created.path, &["commit", "-qm", "work"]);
    let committed = worktree::count_changes(&created.path, head.as_deref()).expect("countable");
    assert!(committed.files.is_empty(), "the edit was committed");
    assert_eq!(committed.commits, 1);
    assert!(worktree::has_changes(&created.path, head.as_deref()));

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn the_stale_sweep_spares_named_and_dirty_worktrees() {
    let repo = temp_repo("sweep");
    let opts = CreateOptions::default();
    let named = worktree::get_or_create(&repo, "keep-me", &opts).expect("named");
    let agent = worktree::get_or_create(&repo, &worktree::agent_slug(1), &opts).expect("agent");

    // Nothing is old enough yet, so a sweep at the real cutoff removes nothing.
    assert_eq!(
        worktree::cleanup_stale_worktrees(&repo, worktree::STALE_AFTER, None),
        0
    );
    assert!(named.path.exists() && agent.path.exists());

    // With a zero cutoff everything qualifies on age — but the user-named
    // worktree must still be spared, because its name is not one plank
    // generates.
    let removed = worktree::cleanup_stale_worktrees(&repo, std::time::Duration::ZERO, None);
    assert_eq!(removed, 1, "only the throwaway worktree is a candidate");
    assert!(named.path.exists(), "a user-named worktree is never swept");
    assert!(!agent.path.exists());

    // A throwaway worktree holding uncommitted work is spared too: the sweep
    // is fail-closed, not merely name-scoped.
    let dirty = worktree::get_or_create(&repo, &worktree::agent_slug(2), &opts).expect("agent");
    std::fs::write(dirty.path.join("a.txt"), b"unsaved\n").unwrap();
    assert_eq!(
        worktree::cleanup_stale_worktrees(&repo, std::time::Duration::ZERO, None),
        0,
        "a dirty throwaway worktree must survive the sweep"
    );
    assert!(dirty.path.exists());

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn a_traversing_name_never_reaches_git() {
    let repo = temp_repo("traversal");
    let err = worktree::get_or_create(&repo, "../escape", &CreateOptions::default())
        .expect_err("must refuse");
    assert!(err.contains("'.' or '..'"), "{err}");
    // The check is before any side effect, so nothing was created anywhere.
    assert!(!repo.parent().unwrap().join("escape").exists());
    assert!(!worktree::worktrees_dir(&repo).exists());
    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn the_tools_move_the_working_directory_in_and_out() {
    use plank::dsml::{ToolArg, ToolCall};
    use plank::tools::{ToolContext, dispatch};

    fn call(name: &str, args: &[(&str, &str)]) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            args: args
                .iter()
                .map(|(k, v)| ToolArg {
                    name: (*k).to_string(),
                    value: (*v).to_string(),
                    is_string: true,
                })
                .collect(),
        }
    }

    let repo = temp_repo("tools");
    let mut ctx = ToolContext::new(&repo);

    let out = dispatch(&call("EnterWorktree", &[("name", "task")]), &mut ctx).output;
    assert!(out.contains("Entered worktree 'task'"), "{out}");
    assert!(out.contains("worktree-task"), "{out}");
    // The point of the tool: every later tool now resolves paths in here.
    assert_eq!(ctx.cwd, worktree::path_for(&repo, "task"));
    assert!(ctx.cwd.join("a.txt").is_file());

    // A file written through the normal `write` tool lands in the worktree and
    // leaves the main checkout untouched — the isolation, demonstrated.
    let out = dispatch(
        &call("write", &[("path", "new.txt"), ("content", "isolated\n")]),
        &mut ctx,
    )
    .output;
    assert!(!out.contains("Tool error"), "{out}");
    assert!(worktree::path_for(&repo, "task").join("new.txt").is_file());
    assert!(
        !repo.join("new.txt").exists(),
        "the main checkout is untouched"
    );

    // That untracked file is work, so a bare remove must refuse.
    let out = dispatch(&call("ExitWorktree", &[("action", "remove")]), &mut ctx).output;
    assert!(out.contains("has unsaved work"), "{out}");
    assert!(out.contains("new.txt"), "{out}");
    assert_eq!(
        ctx.cwd,
        worktree::path_for(&repo, "task"),
        "refusal stays put"
    );

    // Keeping restores the original directory and leaves the worktree behind.
    let out = dispatch(&call("ExitWorktree", &[("action", "keep")]), &mut ctx).output;
    assert!(out.contains("still on disk"), "{out}");
    assert_eq!(ctx.cwd, repo);
    assert!(worktree::path_for(&repo, "task").exists());

    // Re-entering resumes the same worktree, with the earlier work still there.
    let out = dispatch(&call("EnterWorktree", &[("name", "task")]), &mut ctx).output;
    assert!(!out.contains("Tool error"), "{out}");
    assert!(ctx.cwd.join("new.txt").is_file(), "resumed with its work");

    // And an explicit discard finally removes it.
    let out = dispatch(
        &call(
            "ExitWorktree",
            &[("action", "remove"), ("discard_changes", "true")],
        ),
        &mut ctx,
    )
    .output;
    assert!(out.contains("Removed worktree 'task'"), "{out}");
    assert_eq!(ctx.cwd, repo);
    assert!(!worktree::path_for(&repo, "task").exists());

    let _ = std::fs::remove_dir_all(&repo);
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The `EnterWorktree` / `ExitWorktree` tools: move the session into an
//! isolated working copy of the repository and back out again.
//!
//! The mechanics live in [`crate::worktree`]; this module is the model-facing
//! half — argument handling, the safety gate that guards removal, and the
//! working-directory bookkeeping that makes every other tool operate inside
//! the worktree while one is active.
//!
//! Only one worktree may be active per session. Nesting would leave the exit
//! path guessing which directory to return to, and the isolation a second
//! worktree buys over the first is nil.

use std::fmt::Write as _;

use super::ToolContext;
use crate::dsml::ToolCall;
use crate::worktree::{self, WorktreeSession};

/// Handles `EnterWorktree`: creates (or resumes) a worktree and moves the
/// session's working directory into it.
///
/// Every subsequent tool resolves relative paths against
/// [`ToolContext::cwd`](super::ToolContext::cwd), so switching that one field
/// is what actually isolates the work.
pub fn tool_enter_worktree(ctx: &mut ToolContext, call: &ToolCall) -> String {
    if let Some(active) = &ctx.worktree {
        return format!(
            "Tool error: already in worktree '{}' at {}; call ExitWorktree before entering another\n",
            active.name,
            active.path.display()
        );
    }
    let name = call.arg_value("name").unwrap_or("").trim();
    if name.is_empty() {
        return "Tool error: EnterWorktree requires a 'name' for the worktree\n".to_string();
    }
    // Create from the *main* checkout, so calling this from inside a worktree
    // (or from a subdirectory) still lands in the repo's own worktrees dir.
    let from = worktree::canonical_git_root(&ctx.cwd).unwrap_or_else(|| ctx.cwd.clone());
    let settings = crate::settings::active();
    let opts = worktree::CreateOptions {
        pr_number: None,
        sparse_paths: settings.worktree.sparse_paths.clone(),
        symlink_dirs: settings.worktree.symlink_directories.clone(),
    };
    let mut session = match worktree::create_for_session(&from, name, &ctx.hooks, &opts) {
        Ok(s) => s,
        Err(e) => return format!("Tool error: could not create worktree '{name}': {e}\n"),
    };
    // `create_for_session` was handed the repo root, but the session has to
    // return to wherever the model actually was.
    session.original_cwd.clone_from(&ctx.cwd);
    let path = session.path.clone();
    let branch = session.branch.clone();
    ctx.cwd.clone_from(&path);
    ctx.worktree = Some(session);
    let branch_note = branch.map_or_else(String::new, |b| format!(" on branch {b}"));
    format!(
        "Entered worktree '{name}' at {}{branch_note}. The working directory for every tool is \
         now this worktree; edits here do not affect the main checkout. Call ExitWorktree with \
         action 'keep' or 'remove' when you are done.\n",
        path.display()
    )
}

/// Handles `ExitWorktree`: leaves the worktree, either keeping it on disk or
/// destroying it, and restores the session's original working directory.
///
/// The gate on `remove` is the reason this is a tool rather than a `bash` call:
/// it refuses to delete a worktree holding uncommitted files or unpushed
/// commits unless the model explicitly passes `discard_changes`, and it treats
/// an unverifiable state as "has changes".
pub fn tool_exit_worktree(ctx: &mut ToolContext, call: &ToolCall) -> String {
    if ctx.worktree.is_none() {
        return "Tool error: not in a worktree; EnterWorktree was never called in this session\n"
            .to_string();
    }
    let action = call.arg_value("action").unwrap_or("keep").trim().to_owned();
    let discard = matches!(
        call.arg_value("discard_changes").unwrap_or("").trim(),
        "true" | "yes" | "1"
    );
    match action.as_str() {
        "keep" => {
            let session = take_session(ctx);
            format!(
                "Left worktree '{}'. It is still on disk at {}{}; the working directory is back \
                 to {}.\n",
                session.name,
                session.path.display(),
                session
                    .branch
                    .map_or_else(String::new, |b| format!(" on branch {b}")),
                session.original_cwd.display()
            )
        }
        "remove" => remove_worktree(ctx, discard),
        other => format!(
            "Tool error: ExitWorktree 'action' must be 'keep' or 'remove' (got {other:?})\n"
        ),
    }
}

/// The `remove` half of [`tool_exit_worktree`]: checks for unsaved work, then
/// destroys the worktree.
fn remove_worktree(ctx: &mut ToolContext, discard: bool) -> String {
    // Borrowed rather than taken: a refusal must leave the session in the
    // worktree, exactly as it was.
    let session = ctx.worktree.as_ref().expect("checked by caller");
    if !discard {
        let changes = worktree::count_changes(&session.path, session.original_head.as_deref());
        match changes {
            None => {
                return format!(
                    "Tool error: cannot verify whether worktree '{}' holds unsaved work, so it \
                     was not removed. Inspect {} yourself, then either exit with action 'keep' \
                     or pass discard_changes true to remove it anyway.\n",
                    session.name,
                    session.path.display()
                );
            }
            Some(c) if !c.is_empty() => return refuse_with_changes(session, &c),
            Some(_) => {}
        }
    }
    let session = take_session(ctx);
    match worktree::cleanup(&session, &ctx.hooks) {
        Ok(()) => format!(
            "Removed worktree '{}' and its branch; the working directory is back to {}.\n",
            session.name,
            session.original_cwd.display()
        ),
        Err(e) => format!(
            "Tool error: left worktree '{}' but could not remove it: {e}. Its files are still at \
             {}; the working directory is back to {}.\n",
            session.name,
            session.path.display(),
            session.original_cwd.display()
        ),
    }
}

/// The refusal message for a `remove` that would throw away work, naming what
/// would be lost so the model can decide rather than guess.
fn refuse_with_changes(session: &WorktreeSession, changes: &worktree::Changes) -> String {
    const MAX_LISTED: usize = 10;
    let mut msg = format!(
        "Tool error: worktree '{}' has unsaved work, so it was not removed.\n",
        session.name
    );
    if changes.commits > 0 {
        let plural = if changes.commits == 1 { "" } else { "s" };
        let _ = writeln!(
            msg,
            "  {} commit{plural} not present on the base branch",
            changes.commits
        );
    }
    if !changes.files.is_empty() {
        let listed = changes
            .files
            .iter()
            .take(MAX_LISTED)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            msg,
            "  {} uncommitted file(s): {listed}",
            changes.files.len()
        );
        if changes.files.len() > MAX_LISTED {
            let _ = writeln!(msg, "  ... and {} more", changes.files.len() - MAX_LISTED);
        }
    }
    msg.push_str(
        "Merge or push the work first, or exit with action 'keep' to leave the worktree in \
         place. Pass discard_changes true only if this work is genuinely disposable.\n",
    );
    msg
}

/// Removes the active worktree from the context and restores the working
/// directory it was entered from.
fn take_session(ctx: &mut ToolContext) -> WorktreeSession {
    let session = ctx.worktree.take().expect("checked by caller");
    ctx.cwd.clone_from(&session.original_cwd);
    session
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsml::ToolArg;

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

    /// A context already sitting in a worktree, without touching a real repo:
    /// the tools only read the session, so a synthetic one exercises the
    /// guards faithfully.
    fn ctx_in_worktree(path: &std::path::Path, head: Option<&str>) -> ToolContext {
        let mut ctx = ToolContext::new("/orig");
        ctx.worktree = Some(WorktreeSession {
            original_cwd: "/orig".into(),
            repo_root: "/repo".into(),
            path: path.to_path_buf(),
            name: "feature".to_string(),
            branch: Some("worktree-feature".to_string()),
            original_head: head.map(str::to_string),
            hook_based: false,
        });
        ctx.cwd = path.to_path_buf();
        ctx
    }

    #[test]
    fn entering_requires_a_name() {
        let mut ctx = ToolContext::new("/tmp");
        let out = tool_enter_worktree(&mut ctx, &call("EnterWorktree", &[]));
        assert!(out.contains("requires a 'name'"), "{out}");
        assert!(ctx.worktree.is_none());
    }

    #[test]
    fn entering_twice_is_refused() {
        let mut ctx = ctx_in_worktree(std::path::Path::new("/wt"), Some("abc"));
        let out = tool_enter_worktree(&mut ctx, &call("EnterWorktree", &[("name", "other")]));
        assert!(out.contains("already in worktree 'feature'"), "{out}");
        // The refusal must not have disturbed the active session.
        assert_eq!(ctx.cwd, std::path::Path::new("/wt"));
        assert_eq!(ctx.worktree.as_ref().unwrap().name, "feature");
    }

    #[test]
    fn exiting_outside_a_worktree_is_refused() {
        let mut ctx = ToolContext::new("/tmp");
        let out = tool_exit_worktree(&mut ctx, &call("ExitWorktree", &[("action", "keep")]));
        assert!(out.contains("not in a worktree"), "{out}");
    }

    #[test]
    fn keeping_restores_the_original_directory() {
        let mut ctx = ctx_in_worktree(std::path::Path::new("/wt"), Some("abc"));
        let out = tool_exit_worktree(&mut ctx, &call("ExitWorktree", &[("action", "keep")]));
        assert!(out.contains("still on disk"), "{out}");
        assert!(ctx.worktree.is_none());
        assert_eq!(ctx.cwd, std::path::Path::new("/orig"));
    }

    #[test]
    fn an_unknown_action_changes_nothing() {
        let mut ctx = ctx_in_worktree(std::path::Path::new("/wt"), Some("abc"));
        let out = tool_exit_worktree(&mut ctx, &call("ExitWorktree", &[("action", "burn")]));
        assert!(out.contains("must be 'keep' or 'remove'"), "{out}");
        assert!(ctx.worktree.is_some());
        assert_eq!(ctx.cwd, std::path::Path::new("/wt"));
    }

    #[test]
    fn removal_refuses_when_the_state_cannot_be_verified() {
        // Not a git worktree, so `count_changes` answers "unknown" — which must
        // read as "there might be work here", not "there is none".
        let path = std::env::temp_dir().join("plank-exit-wt-unverifiable");
        let mut ctx = ctx_in_worktree(&path, Some("abc"));
        let out = tool_exit_worktree(&mut ctx, &call("ExitWorktree", &[("action", "remove")]));
        assert!(out.contains("cannot verify"), "{out}");
        // A refused removal leaves the session exactly where it was.
        assert!(ctx.worktree.is_some());
        assert_eq!(ctx.cwd, path);
    }

    #[test]
    fn the_refusal_names_what_would_be_lost() {
        let session = WorktreeSession {
            original_cwd: "/orig".into(),
            repo_root: "/repo".into(),
            path: "/wt".into(),
            name: "feature".to_string(),
            branch: None,
            original_head: None,
            hook_based: false,
        };
        let changes = worktree::Changes {
            files: vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
            commits: 1,
        };
        let msg = refuse_with_changes(&session, &changes);
        assert!(msg.contains("1 commit not present"), "{msg}");
        assert!(msg.contains("src/a.rs, src/b.rs"), "{msg}");
        assert!(msg.contains("discard_changes"), "{msg}");
    }
}

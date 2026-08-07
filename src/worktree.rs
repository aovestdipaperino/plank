// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Git-worktree isolation: an isolated working copy of the repository so a
//! session's edits don't collide with the main checkout or with other
//! concurrent sessions.
//!
//! Two mechanisms share this module, as in the reference design
//! (`local/WORKTREE-SUPPORT.md`):
//!
//! 1. **Session worktrees** — the model calls `EnterWorktree` (or the user
//!    passes `--worktree`), the session's working directory switches into
//!    `<mainRepoRoot>/.plank/worktrees/<slug>`, and `ExitWorktree` leaves
//!    (keep) or destroys (remove) it.
//! 2. **Throwaway worktrees** — created per sub-agent run. They are removed
//!    when the work finishes, or swept later by [`cleanup_stale_worktrees`] if
//!    the process died first.
//!
//! The implementation is **hook-first, git-fallback**: when `WorktreeCreate` /
//! `WorktreeRemove` hooks are configured they substitute the VCS backend
//! entirely (so a non-git VCS can be driven); otherwise it shells out to
//! `git worktree add` / `git worktree remove`.
//!
//! # Safety invariants
//!
//! - Slugs are validated against an allowlist **before any side effect**, so a
//!   name can never escape `.plank/worktrees/` (see [`validate_slug`]).
//! - [`canonical_git_root`] validates the worktree `gitdir`/`commondir`
//!   backlink structure, so a hand-crafted repo cannot redirect plank's notion
//!   of the project root somewhere it shouldn't write.
//! - Removal is **fail-closed**: [`count_changes`] returns `None` when the
//!   state cannot be verified, and callers must treat that as "has changes".
//! - Git subprocesses run with credential prompts disabled, so a `git fetch`
//!   against an unauthenticated remote can never hang the agent.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

/// Longest permitted worktree slug, in bytes.
pub const MAX_SLUG_LEN: usize = 64;

/// Directory, relative to the main repo root, holding plank's worktrees.
const WORKTREES_SUBDIR: &str = "worktrees";

/// Age past which an unused throwaway worktree is swept by
/// [`cleanup_stale_worktrees`].
// Spelled in hours because `Duration::from_days` is still unstable and clippy
// asks for the largest available unit.
pub const STALE_AFTER: Duration = Duration::from_hours(30 * 24);

// ---------------------------------------------------------------------------
// Slug validation and naming
// ---------------------------------------------------------------------------

/// Validates a worktree slug.
///
/// Slugs may nest with `/` (`user/feature`); each segment must be non-empty and
/// contain only `[A-Za-z0-9._-]`. `.` and `..` segments are rejected outright:
/// path joining would normalize `..` and let the worktree escape
/// `.plank/worktrees/`, and an absolute slug would discard the prefix entirely.
///
/// # Errors
/// Returns a model-readable message describing the first rule broken.
pub fn validate_slug(slug: &str) -> Result<(), String> {
    if slug.is_empty() {
        return Err("worktree name must not be empty".to_string());
    }
    if slug.len() > MAX_SLUG_LEN {
        return Err(format!(
            "worktree name must be at most {MAX_SLUG_LEN} characters"
        ));
    }
    for segment in slug.split('/') {
        if segment.is_empty() {
            return Err("worktree name must not contain an empty path segment".to_string());
        }
        if segment == "." || segment == ".." {
            return Err("worktree name must not contain '.' or '..' segments".to_string());
        }
        if let Some(bad) = segment
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')))
        {
            return Err(format!(
                "worktree name may only contain letters, digits, '.', '_', '-' and '/' (found {bad:?})"
            ));
        }
    }
    Ok(())
}

/// Flattens a nested slug for use as a branch name and directory name:
/// `user/feature` becomes `user+feature`.
///
/// Nesting is unsafe in both places. As a ref, `worktree-user` and
/// `worktree-user/feature` are a directory/file conflict git rejects; as a
/// directory, `git worktree remove` on the parent would take the child's
/// uncommitted work with it. `+` is legal in git refs and in paths but is not
/// in the slug allowlist, so the mapping cannot collide.
#[must_use]
pub fn flatten_slug(slug: &str) -> String {
    slug.replace('/', "+")
}

/// The git branch a worktree slug checks out.
#[must_use]
pub fn branch_name(slug: &str) -> String {
    format!("worktree-{}", flatten_slug(slug))
}

/// The directory holding all of `root`'s worktrees.
#[must_use]
pub fn worktrees_dir(root: &Path) -> PathBuf {
    root.join(".plank").join(WORKTREES_SUBDIR)
}

/// The directory a worktree slug lives in, under `root`.
#[must_use]
pub fn path_for(root: &Path, slug: &str) -> PathBuf {
    worktrees_dir(root).join(flatten_slug(slug))
}

// ---------------------------------------------------------------------------
// Git plumbing
// ---------------------------------------------------------------------------

/// Runs a git command in `cwd` with credential prompting disabled, returning
/// trimmed stdout on success.
///
/// `GIT_TERMINAL_PROMPT=0`, an empty `GIT_ASKPASS`, and a closed stdin together
/// ensure a fetch against a remote plank has no credentials for fails fast
/// instead of blocking the agent on an invisible prompt.
///
/// # Errors
/// Returns git's stderr (or the spawn error) when the command fails.
fn git(args: &[&str], cwd: &Path) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "")
        .env("SSH_ASKPASS", "")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(if err.is_empty() {
            format!("git {} failed", args.join(" "))
        } else {
            err
        })
    }
}

/// True when the git command succeeds.
fn git_ok(args: &[&str], cwd: &Path) -> bool {
    git(args, cwd).is_ok()
}

/// Like [`git`], but returns stdout **untrimmed**.
///
/// `git status --porcelain` is column-significant — an unstaged modification is
/// `" M path"`, with a leading space — so trimming the output silently eats the
/// first line's status column and corrupts the path parsed out of it.
///
/// # Errors
/// Returns git's stderr (or the spawn error) when the command fails.
fn git_raw(args: &[&str], cwd: &Path) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "")
        .env("SSH_ASKPASS", "")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Walks up from `from` looking for a `.git` entry (a directory in a normal
/// clone, a pointer *file* inside a worktree or submodule).
///
/// This is the *lexical* root: inside a worktree it returns the worktree
/// directory, not the main checkout. Use [`canonical_git_root`] when the
/// answer must identify the repository rather than the checkout.
#[must_use]
pub fn find_git_root(from: &Path) -> Option<PathBuf> {
    let mut dir = from;
    loop {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// Resolves a checkout's `.git` to the real git directory: the directory
/// itself for a normal clone, or the `gitdir:` target for a worktree.
#[must_use]
pub fn resolve_git_dir(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git);
    }
    let text = std::fs::read_to_string(&dot_git).ok()?;
    let target = text.trim().strip_prefix("gitdir:")?.trim();
    let path = PathBuf::from(target);
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

/// Reads the `commondir` file of a worktree's git directory: the shared `.git`
/// directory every worktree of the repository points at.
#[must_use]
pub fn common_dir(git_dir: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(git_dir.join("commondir")).ok()?;
    let path = PathBuf::from(text.trim());
    Some(if path.is_absolute() {
        path
    } else {
        git_dir.join(path)
    })
}

/// Resolves `from` to the **main** checkout of its repository, following a
/// worktree's `gitdir:` pointer and `commondir` back to the original.
///
/// Project-scoped state (session storage, the worktrees directory itself)
/// keys off this, so every worktree of a repo shares one identity — and so a
/// throwaway worktree created from *inside* a worktree still lands in the main
/// repo, where [`cleanup_stale_worktrees`] will later find it.
///
/// The structure is validated before it is trusted: the worktree's git dir must
/// be a direct child of `<commondir>/worktrees/`, and its `gitdir` backlink must
/// point at the checkout we started from. A repository whose pointer files do
/// not match what `git worktree add` produces is not followed, so a crafted
/// checkout cannot redirect plank's project root elsewhere.
#[must_use]
pub fn canonical_git_root(from: &Path) -> Option<PathBuf> {
    let root = find_git_root(from)?;
    let git_dir = resolve_git_dir(&root)?;
    if root.join(".git").is_dir() {
        return Some(root); // a normal clone is already canonical
    }
    // `commondir` is normally the relative `../..`, so both sides have to be
    // resolved before they can be compared — a textual match would fail on the
    // very shape git actually writes.
    let common = common_dir(&git_dir)?.canonicalize().ok()?;
    // `git worktree add` always places the per-worktree git dir at
    // <common>/worktrees/<name>; anything else is not a worktree we made.
    let parent = git_dir.parent()?.canonicalize().ok()?;
    if parent != common.join(WORKTREES_SUBDIR) {
        return None;
    }
    // ...and the backlink must come back to the checkout we started from.
    let link = std::fs::read_to_string(git_dir.join("gitdir")).ok()?;
    let link = PathBuf::from(link.trim());
    let same = match (link.canonicalize(), root.join(".git").canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same {
        return None;
    }
    // The common dir is `<mainRoot>/.git`; its parent is the main checkout.
    // A bare repo has no checkout, so the common dir is the identity itself.
    match common.file_name().and_then(std::ffi::OsStr::to_str) {
        Some(".git") => common.parent().map(Path::to_path_buf),
        _ => Some(common),
    }
}

/// Reads a ref straight off disk (loose file first, then `packed-refs`),
/// avoiding a `git rev-parse` subprocess on a hot path.
#[must_use]
fn resolve_ref(git_dir: &Path, name: &str) -> Option<String> {
    if let Ok(text) = std::fs::read_to_string(git_dir.join(name)) {
        let sha = text.trim().to_string();
        if !sha.is_empty() {
            return Some(sha);
        }
    }
    let packed = std::fs::read_to_string(git_dir.join("packed-refs")).ok()?;
    packed.lines().find_map(|line| {
        let (sha, r) = line.split_once(' ')?;
        (r.trim() == name).then(|| sha.trim().to_string())
    })
}

/// The HEAD commit of an existing worktree, read from its pointer files.
///
/// This is the fast-resume existence check: a named worktree reuses the same
/// path across invocations, and answering "does it already exist?" from disk
/// keeps a resume from paying for a `git fetch` (which can block on
/// credentials) and a `rev-parse` subprocess every time.
#[must_use]
pub fn worktree_head_sha(worktree: &Path) -> Option<String> {
    let git_dir = resolve_git_dir(worktree)?;
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    match head.strip_prefix("ref:") {
        Some(name) => {
            let name = name.trim();
            let common = common_dir(&git_dir).unwrap_or_else(|| git_dir.clone());
            resolve_ref(&git_dir, name).or_else(|| resolve_ref(&common, name))
        }
        None if !head.is_empty() => Some(head.to_string()),
        None => None,
    }
}

/// The repository's default branch: `origin/HEAD` when the remote advertises
/// one, else the configured `init.defaultBranch`, else `main`.
#[must_use]
pub fn default_branch(root: &Path) -> String {
    if let Ok(s) = git(&["symbolic-ref", "refs/remotes/origin/HEAD"], root)
        && let Some(name) = s.strip_prefix("refs/remotes/origin/")
        && !name.is_empty()
    {
        return name.to_string();
    }
    match git(&["config", "init.defaultBranch"], root) {
        Ok(s) if !s.is_empty() => s,
        _ => "main".to_string(),
    }
}

/// The branch currently checked out at `root`, or `None` when detached.
#[must_use]
pub fn current_branch(root: &Path) -> Option<String> {
    match git(&["rev-parse", "--abbrev-ref", "HEAD"], root) {
        Ok(s) if !s.is_empty() && s != "HEAD" => Some(s),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

/// Tunables for worktree creation, from `settings.json`'s `worktree` block plus
/// the caller's own choices.
#[derive(Debug, Clone, Default)]
pub struct CreateOptions {
    /// A GitHub pull request to base the worktree on, fetched from
    /// `refs/pull/<n>/head`.
    pub pr_number: Option<u32>,
    /// Cone-mode sparse-checkout paths; empty means a full checkout.
    pub sparse_paths: Vec<String>,
    /// Directories symlinked from the main checkout instead of copied, e.g.
    /// `target` or `node_modules`, to keep a worktree cheap.
    pub symlink_dirs: Vec<String>,
}

/// A worktree that now exists on disk.
#[derive(Debug, Clone)]
pub struct Created {
    /// Its working directory.
    pub path: PathBuf,
    /// The branch checked out in it.
    pub branch: String,
    /// True when it was already there and this call merely resumed it.
    pub existed: bool,
}

/// Creates the worktree for `slug` under `root`, or resumes it if it already
/// exists.
///
/// # Errors
/// Returns a message when the slug is invalid or a git command fails.
pub fn get_or_create(root: &Path, slug: &str, opts: &CreateOptions) -> Result<Created, String> {
    validate_slug(slug)?;
    let path = path_for(root, slug);
    let branch = branch_name(slug);
    // Fast resume: a readable HEAD means the worktree is already set up, so
    // skip the fetch entirely.
    if worktree_head_sha(&path).is_some() {
        return Ok(Created {
            path,
            branch,
            existed: true,
        });
    }
    let base = resolve_base(root, opts.pr_number)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create worktrees dir: {e}"))?;
    }
    let path_str = path.to_string_lossy().to_string();
    let mut args = vec!["worktree", "add"];
    if !opts.sparse_paths.is_empty() {
        args.push("--no-checkout");
    }
    // `-B` rather than `-b`: a previously removed worktree can leave its branch
    // behind, and resetting it in place saves a separate `git branch -D`.
    args.extend_from_slice(&["-B", &branch, &path_str, &base]);
    git(&args, root).map_err(|e| format!("git worktree add: {e}"))?;
    if !opts.sparse_paths.is_empty()
        && let Err(e) = apply_sparse_checkout(&path, &opts.sparse_paths)
    {
        // A half-checked-out worktree must not survive: the next call's
        // fast-resume path would find its HEAD and present it as ready.
        let _ = git(&["worktree", "remove", "--force", &path_str], root);
        return Err(e);
    }
    Ok(Created {
        path,
        branch,
        existed: false,
    })
}

/// The commit a new worktree branches from: the PR head when one is named, else
/// the default branch (fetched only when it isn't already local), else `HEAD`.
fn resolve_base(root: &Path, pr_number: Option<u32>) -> Result<String, String> {
    if let Some(pr) = pr_number {
        let refspec = format!("pull/{pr}/head");
        git(&["fetch", "origin", &refspec], root)
            .map_err(|e| format!("fetch pull request #{pr}: {e}"))?;
        return Ok("FETCH_HEAD".to_string());
    }
    let default = default_branch(root);
    let remote_ref = format!("origin/{default}");
    if let Some(git_dir) = resolve_git_dir(root)
        && resolve_ref(git_dir.as_path(), &format!("refs/remotes/{remote_ref}")).is_some()
    {
        return Ok(remote_ref);
    }
    if git_ok(&["fetch", "origin", &default], root) {
        return Ok("FETCH_HEAD".to_string());
    }
    Ok("HEAD".to_string())
}

/// Narrows a `--no-checkout` worktree to `paths` and materializes them.
fn apply_sparse_checkout(worktree: &Path, paths: &[String]) -> Result<(), String> {
    let mut args = vec!["sparse-checkout", "set", "--cone", "--"];
    args.extend(paths.iter().map(String::as_str));
    git(&args, worktree).map_err(|e| format!("git sparse-checkout: {e}"))?;
    git(&["checkout", "HEAD"], worktree).map_err(|e| format!("git checkout: {e}"))?;
    Ok(())
}

/// Wires up a freshly created worktree so it behaves like the main checkout.
///
/// Best-effort throughout: none of this is worth failing an otherwise usable
/// worktree over, so each step is attempted and skipped on error.
pub fn post_creation_setup(root: &Path, worktree: &Path, opts: &CreateOptions) {
    copy_project_settings(root, worktree);
    share_git_hooks(root);
    symlink_dirs(root, worktree, &opts.symlink_dirs);
    copy_include_files(root, worktree);
}

/// Copies the project's `.plank/settings.json` into the worktree, so a session
/// that moves in keeps the project's configured engine, sandbox, and UI
/// choices instead of falling back to the global defaults.
fn copy_project_settings(root: &Path, worktree: &Path) {
    let src = root.join(".plank").join("settings.json");
    if !src.is_file() {
        return;
    }
    let dst_dir = worktree.join(".plank");
    if std::fs::create_dir_all(&dst_dir).is_ok() {
        let _ = std::fs::copy(src, dst_dir.join("settings.json"));
    }
}

/// Points every worktree at the main checkout's git hooks.
///
/// Written to the shared config (no `--worktree`), so this is a no-op once set;
/// the existing value is read first to skip even the write.
fn share_git_hooks(root: &Path) {
    let hooks = root.join(".git").join("hooks");
    if !hooks.is_dir() {
        return;
    }
    let want = hooks.to_string_lossy().to_string();
    if git(&["config", "core.hooksPath"], root).as_deref() == Ok(want.as_str()) {
        return;
    }
    let _ = git(&["config", "core.hooksPath", &want], root);
}

/// Symlinks heavyweight build directories from the main checkout, rather than
/// leaving each worktree to rebuild (and store) its own copy.
fn symlink_dirs(root: &Path, worktree: &Path, dirs: &[String]) {
    for dir in dirs {
        // A configured name is not a path: anything that could climb out of
        // the worktree is refused rather than sanitized.
        if dir.is_empty() || dir.contains("..") || Path::new(dir).is_absolute() {
            continue;
        }
        let src = root.join(dir);
        let dst = worktree.join(dir);
        if !src.is_dir() || dst.exists() {
            continue;
        }
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        #[cfg(unix)]
        let _ = std::os::unix::fs::symlink(&src, &dst);
        #[cfg(not(unix))]
        let _ = std::os::windows::fs::symlink_dir(&src, &dst);
    }
}

/// Copies gitignored files named by `.worktreeinclude` into a new worktree.
///
/// The point is local-but-essential files — `.env`, credentials, generated
/// build config — that a fresh checkout would otherwise lack. A file is copied
/// only when it is **both** listed in `.worktreeinclude` and actually ignored
/// by git, so this can never smuggle a tracked file past review.
fn copy_include_files(root: &Path, worktree: &Path) {
    let Ok(spec) = std::fs::read_to_string(root.join(".worktreeinclude")) else {
        return;
    };
    let patterns: Vec<&str> = spec
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if patterns.is_empty() {
        return;
    }
    let Ok(listing) = git(
        &["ls-files", "--others", "--ignored", "--exclude-standard"],
        root,
    ) else {
        return;
    };
    for rel in listing.lines().map(str::trim).filter(|l| !l.is_empty()) {
        if !patterns.iter().any(|p| include_matches(p, rel)) {
            continue;
        }
        let src = root.join(rel);
        let dst = worktree.join(rel);
        if !src.is_file() || dst.exists() {
            continue;
        }
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::copy(&src, &dst);
    }
}

/// Matches one `.worktreeinclude` pattern against a repo-relative path.
///
/// A deliberately small subset of gitignore syntax: an exact path, a directory
/// prefix (`config/`), or a `*.ext` suffix. Anything richer is better expressed
/// as several lines than as a half-correct glob engine.
#[must_use]
fn include_matches(pattern: &str, rel: &str) -> bool {
    let pattern = pattern.trim_start_matches("./");
    if let Some(ext) = pattern.strip_prefix("*.") {
        return rel.ends_with(&format!(".{ext}"));
    }
    if let Some(dir) = pattern.strip_suffix('/') {
        return rel.starts_with(&format!("{dir}/"));
    }
    rel == pattern || rel.starts_with(&format!("{pattern}/"))
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

/// A live session worktree: everything needed to move back out of it again.
///
/// Held in memory for the life of the session (on
/// [`ToolContext`](crate::tools::ToolContext)); nothing is written to the
/// transcript, so a resumed session always starts in the directory it was
/// launched from.
#[derive(Debug, Clone)]
pub struct WorktreeSession {
    /// The working directory the session had before entering.
    pub original_cwd: PathBuf,
    /// The main checkout, where removal commands must run from.
    pub repo_root: PathBuf,
    /// The worktree's working directory.
    pub path: PathBuf,
    /// The slug it was created under.
    pub name: String,
    /// The branch checked out in it; `None` for a hook-created worktree, whose
    /// VCS may not have branches at all.
    pub branch: Option<String>,
    /// The commit the worktree started at, so [`count_changes`] can tell how
    /// many commits the session added.
    pub original_head: Option<String>,
    /// True when a `WorktreeCreate` hook made this instead of git.
    pub hook_based: bool,
}

/// Creates a session worktree for `slug`, hook-first and git-fallback.
///
/// `from` is the session's current directory; the worktree is created under the
/// *main* checkout even when called from inside another worktree.
///
/// # Errors
/// Returns a message when the slug is invalid, no VCS backend is available, or
/// creation fails.
pub fn create_for_session(
    from: &Path,
    slug: &str,
    hooks: &crate::hooks::Hooks,
    opts: &CreateOptions,
) -> Result<WorktreeSession, String> {
    validate_slug(slug)?;
    let root = canonical_git_root(from).or_else(|| find_git_root(from));
    if crate::hooks::has_worktree_create_hook(hooks) {
        let path = crate::hooks::run_worktree_create_hook(hooks, slug, from)?;
        return Ok(WorktreeSession {
            original_cwd: from.to_path_buf(),
            repo_root: root.unwrap_or_else(|| from.to_path_buf()),
            path,
            name: slug.to_string(),
            branch: None,
            original_head: None,
            hook_based: true,
        });
    }
    let root = root.ok_or_else(|| {
        "not inside a git repository, and no WorktreeCreate hook is configured".to_string()
    })?;
    let created = get_or_create(&root, slug, opts)?;
    if !created.existed {
        post_creation_setup(&root, &created.path, opts);
    }
    let head = worktree_head_sha(&created.path);
    Ok(WorktreeSession {
        original_cwd: from.to_path_buf(),
        repo_root: root,
        path: created.path,
        name: slug.to_string(),
        branch: Some(created.branch),
        original_head: head,
        hook_based: false,
    })
}

/// Destroys a session's worktree and its branch.
///
/// Runs from the main checkout, never from the directory being deleted. The
/// branch delete is attempted after the worktree is gone, so git has released
/// its hold on the ref.
///
/// # Errors
/// Returns a message when the removal command fails.
pub fn cleanup(session: &WorktreeSession, hooks: &crate::hooks::Hooks) -> Result<(), String> {
    if session.hook_based || crate::hooks::has_worktree_remove_hook(hooks) {
        return crate::hooks::run_worktree_remove_hook(hooks, &session.path, &session.repo_root);
    }
    let path = session.path.to_string_lossy().to_string();
    git(
        &["worktree", "remove", "--force", &path],
        &session.repo_root,
    )
    .map_err(|e| format!("git worktree remove: {e}"))?;
    if let Some(branch) = &session.branch {
        let _ = git(&["branch", "-D", branch], &session.repo_root);
    }
    Ok(())
}

/// Worktree created by `--worktree` before the agent exists, waiting to be
/// adopted by the session it was made for.
///
/// `main` has to create the worktree and `chdir` into it *before* building the
/// agent, because the agent reads its working directory, hooks, and agent
/// definitions from wherever the process currently is. This hands the resulting
/// session across that gap so `ExitWorktree` can still leave it later.
static STARTUP_SESSION: std::sync::Mutex<Option<WorktreeSession>> = std::sync::Mutex::new(None);

/// Parks the `--worktree` session for the agent to pick up.
pub fn set_startup_session(session: WorktreeSession) {
    *STARTUP_SESSION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(session);
}

/// Takes the `--worktree` session, if `main` created one. Returns `None` on
/// every call after the first.
#[must_use]
pub fn take_startup_session() -> Option<WorktreeSession> {
    STARTUP_SESSION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

// ---------------------------------------------------------------------------
// Change accounting
// ---------------------------------------------------------------------------

/// What a session did inside its worktree, as of now.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Changes {
    /// Repo-relative paths with uncommitted modifications.
    pub files: Vec<String>,
    /// Commits made since the worktree was created.
    pub commits: usize,
}

impl Changes {
    /// True when nothing would be lost by removing the worktree.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.commits == 0
    }
}

/// Counts the work a worktree holds, or `None` when that cannot be determined.
///
/// `None` is not "no changes" — it is "unknown", and every caller must treat it
/// as a refusal to remove. It arises when git fails, or when the worktree was
/// made by a hook and so has no recorded starting commit to diff against.
#[must_use]
pub fn count_changes(worktree: &Path, original_head: Option<&str>) -> Option<Changes> {
    let status = git_raw(&["status", "--porcelain"], worktree).ok()?;
    let files = status
        .lines()
        .filter_map(|l| l.get(3..).map(str::trim).filter(|s| !s.is_empty()))
        .map(str::to_string)
        .collect();
    let head = original_head?;
    let range = format!("{head}..HEAD");
    let commits = git(&["rev-list", "--count", &range], worktree)
        .ok()?
        .parse()
        .ok()?;
    Some(Changes { files, commits })
}

/// Fail-closed convenience over [`count_changes`]: unverifiable state counts as
/// having changes, so a caller can never delete work it failed to look at.
#[must_use]
pub fn has_changes(worktree: &Path, original_head: Option<&str>) -> bool {
    count_changes(worktree, original_head).is_none_or(|c| !c.is_empty())
}

// ---------------------------------------------------------------------------
// Throwaway (sub-agent) worktrees
// ---------------------------------------------------------------------------

/// The slug for the throwaway worktree of sub-agent run `id`.
#[must_use]
pub fn agent_slug(id: u64) -> String {
    format!("agent-a{id:07x}")
}

/// Creates a throwaway worktree for a sub-agent run.
///
/// Unlike [`create_for_session`] this touches no session state — it only puts a
/// directory on disk and hands back its path. It resolves the **main** checkout
/// first, so a sub-agent spawned from inside a session worktree still lands in
/// the repo's own `.plank/worktrees/`, where [`cleanup_stale_worktrees`] can
/// find it if this process dies.
///
/// # Errors
/// Returns a message when `from` is not in a git repository or git fails.
pub fn create_agent_worktree(from: &Path, slug: &str) -> Result<WorktreeSession, String> {
    let root = canonical_git_root(from)
        .or_else(|| find_git_root(from))
        .ok_or_else(|| "not inside a git repository".to_string())?;
    let opts = CreateOptions::default();
    let created = get_or_create(&root, slug, &opts)?;
    if created.existed {
        // A resumed worktree is in use again; refresh its mtime so the stale
        // sweep does not judge it by when it was first created.
        touch(&created.path);
    } else {
        post_creation_setup(&root, &created.path, &opts);
    }
    let head = worktree_head_sha(&created.path);
    Ok(WorktreeSession {
        original_cwd: from.to_path_buf(),
        repo_root: root,
        path: created.path,
        name: slug.to_string(),
        branch: Some(created.branch),
        original_head: head,
        hook_based: false,
    })
}

/// Marks a worktree as recently used, so the stale sweep leaves it alone.
fn touch(path: &Path) {
    let stamp = path.join(".plank-worktree-used");
    let _ = std::fs::write(&stamp, b"");
}

// ---------------------------------------------------------------------------
// Stale cleanup
// ---------------------------------------------------------------------------

/// True when a worktree directory name has the exact shape of a throwaway
/// worktree plank generates.
///
/// The sweep is dangerous by nature, so it recognizes only names it could have
/// produced itself: `agent-a<7 hex>`. A user-named `EnterWorktree` slug never
/// matches, so it is never a candidate for automatic deletion.
#[must_use]
pub fn is_ephemeral_slug(name: &str) -> bool {
    let Some(hex) = name.strip_prefix("agent-a") else {
        return false;
    };
    hex.len() == 7 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Removes throwaway worktrees left behind when a previous plank was killed
/// before it could clean up after itself.
///
/// Fail-closed at every step: a worktree is removed only if its name has the
/// ephemeral shape, it is older than `max_age`, it is not `keep`, and git can
/// confirm it has neither uncommitted changes nor unpushed commits. Anything
/// that cannot be checked is left alone. Returns how many were removed.
#[must_use]
pub fn cleanup_stale_worktrees(root: &Path, max_age: Duration, keep: Option<&Path>) -> usize {
    let Ok(entries) = std::fs::read_dir(worktrees_dir(root)) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if !is_ephemeral_slug(&name) || Some(path.as_path()) == keep {
            continue;
        }
        let branch = branch_name(&name);
        if !is_older_than(&path, max_age) || !is_abandonable(&path, &branch) {
            continue;
        }
        let path_str = path.to_string_lossy().to_string();
        if git(&["worktree", "remove", "--force", &path_str], root).is_ok() {
            let _ = git(&["branch", "-D", &branch], root);
            removed += 1;
        }
    }
    if removed > 0 {
        let _ = git(&["worktree", "prune"], root);
    }
    removed
}

/// True when nothing in the worktree has been touched for `max_age`.
///
/// Uses the newest of the directory's own mtime and the marker
/// [`touch`] writes on resume, so a long-lived worktree that is still being
/// used is not judged by its creation time.
fn is_older_than(path: &Path, max_age: Duration) -> bool {
    let newest = [path.to_path_buf(), path.join(".plank-worktree-used")]
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok()?.modified().ok())
        .max();
    // An unreadable mtime is not evidence of staleness.
    newest.is_some_and(|t| {
        SystemTime::now()
            .duration_since(t)
            .is_ok_and(|age| age > max_age)
    })
}

/// True when git confirms a worktree holds no uncommitted changes and no
/// commits missing from every remote. A git failure answers `false`.
fn is_abandonable(path: &Path, branch: &str) -> bool {
    // Untracked files count: a sub-agent's new source file is exactly the work
    // this sweep must not throw away, and it is untracked by definition.
    let Ok(status) = git_raw(&["status", "--porcelain"], path) else {
        return false;
    };
    if !status.trim().is_empty() {
        return false;
    }
    // "Are there commits reachable only from this worktree's own branch?" —
    // asked against every other ref rather than against `--remotes`, because a
    // repository with no remote configured would otherwise report its entire
    // history as unpushed and nothing would ever become sweepable.
    let exclude = format!("--exclude=refs/heads/{branch}");
    git(
        &[
            "rev-list",
            "--max-count=1",
            "HEAD",
            "--not",
            &exclude,
            "--all",
        ],
        path,
    )
    .is_ok_and(|out| out.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_validation_rejects_traversal_and_junk() {
        assert!(validate_slug("feature").is_ok());
        assert!(validate_slug("user/feature").is_ok());
        assert!(validate_slug("a.b_c-1").is_ok());

        assert!(validate_slug("").is_err());
        assert!(validate_slug("..").is_err());
        assert!(validate_slug("../escape").is_err());
        assert!(validate_slug("a/../b").is_err());
        assert!(validate_slug("/absolute").is_err());
        assert!(validate_slug("a//b").is_err());
        assert!(validate_slug("has space").is_err());
        assert!(validate_slug("semi;colon").is_err());
        assert!(validate_slug(&"x".repeat(MAX_SLUG_LEN + 1)).is_err());
    }

    #[test]
    fn nested_slugs_flatten_injectively() {
        assert_eq!(flatten_slug("user/feature"), "user+feature");
        assert_eq!(branch_name("user/feature"), "worktree-user+feature");
        // `+` is outside the slug allowlist, so no two slugs can collide.
        assert!(validate_slug("user+feature").is_err());
    }

    #[test]
    fn worktree_paths_stay_under_the_repo() {
        let root = Path::new("/repo");
        assert_eq!(
            path_for(root, "user/feature"),
            Path::new("/repo/.plank/worktrees/user+feature")
        );
    }

    #[test]
    fn only_generated_agent_names_are_swept() {
        assert!(is_ephemeral_slug("agent-a0000001"));
        assert!(is_ephemeral_slug("agent-adeadbee"));
        assert!(!is_ephemeral_slug("agent-a000001")); // too short
        assert!(!is_ephemeral_slug("agent-a00000012")); // too long
        assert!(!is_ephemeral_slug("agent-azzzzzzz")); // not hex
        assert!(!is_ephemeral_slug("my-feature"));
        assert!(!is_ephemeral_slug(""));
        assert!(is_ephemeral_slug(&agent_slug(0x0dea_dbee)));
    }

    #[test]
    fn unverifiable_state_counts_as_having_changes() {
        // A path that is not a git worktree: `git status` fails, so the
        // fail-closed answer is "there are changes, do not delete".
        let dir = std::env::temp_dir().join("plank-wt-not-a-repo");
        assert!(count_changes(&dir, Some("abc")).is_none());
        assert!(has_changes(&dir, Some("abc")));
    }

    #[test]
    fn changes_are_empty_only_when_both_halves_are() {
        assert!(Changes::default().is_empty());
        assert!(
            !Changes {
                files: vec!["a.rs".into()],
                commits: 0
            }
            .is_empty()
        );
        assert!(
            !Changes {
                files: vec![],
                commits: 1
            }
            .is_empty()
        );
    }

    #[test]
    fn include_patterns_cover_exact_dir_and_extension() {
        assert!(include_matches(".env", ".env"));
        assert!(!include_matches(".env", ".env.local"));
        assert!(include_matches("config/", "config/dev.toml"));
        assert!(!include_matches("config/", "configuration.toml"));
        assert!(include_matches("*.pem", "certs/key.pem"));
        assert!(!include_matches("*.pem", "key.pem.txt"));
    }

    #[test]
    fn symlink_names_that_climb_out_are_refused() {
        let tmp = std::env::temp_dir().join("plank-wt-symlink-guard");
        let _ = std::fs::create_dir_all(tmp.join("wt"));
        symlink_dirs(
            &tmp,
            &tmp.join("wt"),
            &["../outside".to_string(), "/etc".to_string()],
        );
        assert!(!tmp.join("wt").join("../outside").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

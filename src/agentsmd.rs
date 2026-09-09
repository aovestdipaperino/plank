// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Project-root `AGENTS.md` preparation at interactive startup.
//!
//! plank reads `AGENTS.md` and nothing else (see `context::discover_agents_md_files`).
//! Many projects carry a `CLAUDE.md` instead, so an interactive start with a
//! `CLAUDE.md` and no `AGENTS.md` links the one to the other; a project with
//! neither is offered `/init`. Headless runs (`--non-interactive`) do none of
//! this: they must not write into a checkout or block on a question.

use std::path::{Path, PathBuf};

/// What [`prepare`] found in the project root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Startup {
    /// `AGENTS.md` already exists (a file or a link); nothing was done.
    Present,
    /// `AGENTS.md` was missing and `CLAUDE.md` was present, so `AGENTS.md`
    /// was created as a symbolic link to it. Holds the link's path.
    Linked(PathBuf),
    /// Neither file exists; the caller may offer to generate one.
    Missing,
    /// Neither file exists, but the user chose "Don't ask for this folder"
    /// on an earlier start ([`skip`]), so no offer is made.
    Skipped,
}

/// Prepares `AGENTS.md` in `dir`, linking it to `CLAUDE.md` when only the
/// latter exists.
///
/// The link target is the bare name `CLAUDE.md`, not an absolute path, so the
/// checkout can move and the link still resolves. A dangling `AGENTS.md`
/// symlink counts as present: it is the user's to fix, and replacing it would
/// destroy their intent.
///
/// A folder the user has marked with [`skip`] reports [`Startup::Skipped`]
/// instead of `Missing`, so the offer is not repeated. The skip is consulted
/// only when there is nothing to link: a `CLAUDE.md` that appears later is
/// still linked, because that writes no prose and asks no question.
///
/// # Errors
///
/// Returns the OS error text when the link cannot be created.
pub fn prepare(dir: &Path) -> Result<Startup, String> {
    let agents = dir.join("AGENTS.md");
    if agents.symlink_metadata().is_ok() {
        return Ok(Startup::Present);
    }
    if !dir.join("CLAUDE.md").is_file() {
        return Ok(if is_skipped(dir, skip_list_path().as_deref()) {
            Startup::Skipped
        } else {
            Startup::Missing
        });
    }
    std::os::unix::fs::symlink("CLAUDE.md", &agents)
        .map_err(|e| format!("cannot link AGENTS.md to CLAUDE.md: {e}"))?;
    Ok(Startup::Linked(agents))
}

/// The question put to the user when a project has no `AGENTS.md`.
pub const OFFER_QUESTION: &str = "This project has no AGENTS.md. Generate one now? The model will read the codebase and write it.";

/// The user's answer to [`OFFER_QUESTION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offer {
    /// Run `/init` now.
    Generate,
    /// Start the session without one; ask again next time.
    NotNow,
    /// Start without one and never ask again for this folder ([`skip`]).
    DontAskHere,
}

/// The three answers to [`OFFER_QUESTION`], in display order, each as
/// `(label, description)`. Generate comes first: it is the default a stray
/// Enter picks, since writing a project note is the outcome the offer exists
/// for and it is trivially undone.
pub const OFFER_OPTIONS: [(Offer, &str, &str); 3] = [
    (
        Offer::Generate,
        "Generate",
        "run /init: the model reads the codebase and writes AGENTS.md",
    ),
    (Offer::NotNow, "Not now", "start the session without one"),
    (
        Offer::DontAskHere,
        "Don't ask for this folder",
        "start without one and never offer again in this folder",
    ),
];

/// File under `~/.plank` listing the folders whose owner answered "Don't ask
/// for this folder": one canonical absolute path per line. Deleting a line
/// (or the file) re-enables the offer.
const SKIP_LIST_FILE: &str = "agentsmd-skip";

/// Path of the skip list, or `None` when `$HOME` is unset.
#[must_use]
pub fn skip_list_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(".plank").join(SKIP_LIST_FILE))
}

/// The folder's identity in the skip list: its canonical path when the OS can
/// resolve it, the path as given otherwise.
fn skip_key(dir: &Path) -> String {
    dir.canonicalize()
        .unwrap_or_else(|_| dir.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// True when `dir` is listed in the skip list at `list` (a missing or
/// unreadable list skips nothing).
fn is_skipped(dir: &Path, list: Option<&Path>) -> bool {
    let Some(list) = list else { return false };
    let Ok(text) = std::fs::read_to_string(list) else {
        return false;
    };
    let key = skip_key(dir);
    text.lines().any(|l| l.trim() == key)
}

/// Records that the offer must not be made again for `dir`, appending it to
/// the skip list at `list` (created on first use). Idempotent.
///
/// # Errors
///
/// Returns the OS error text when the list cannot be written.
pub fn skip_in(dir: &Path, list: &Path) -> Result<(), String> {
    if is_skipped(dir, Some(list)) {
        return Ok(());
    }
    if let Some(parent) = list.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    let mut text = std::fs::read_to_string(list).unwrap_or_default();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&skip_key(dir));
    text.push('\n');
    std::fs::write(list, text).map_err(|e| format!("cannot write {}: {e}", list.display()))
}

/// [`skip_in`] against the user's own list under `~/.plank`.
///
/// # Errors
///
/// Returns a message when `$HOME` is unset or the list cannot be written.
pub fn skip(dir: &Path) -> Result<(), String> {
    let list = skip_list_path().ok_or_else(|| "HOME is not set".to_owned())?;
    skip_in(dir, &list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("plank-agentsmd-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn links_agents_md_to_claude_md_when_only_claude_exists() {
        let dir = scratch("link");
        std::fs::write(dir.join("CLAUDE.md"), "# rules\n").unwrap();
        let got = prepare(&dir).unwrap();
        assert_eq!(got, Startup::Linked(dir.join("AGENTS.md")));
        assert_eq!(
            std::fs::read_link(dir.join("AGENTS.md")).unwrap(),
            PathBuf::from("CLAUDE.md"),
            "relative target so the checkout can move"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("AGENTS.md")).unwrap(),
            "# rules\n"
        );
        // Second start: already present, nothing rewritten.
        assert_eq!(prepare(&dir).unwrap(), Startup::Present);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_agents_md_is_left_alone_even_beside_claude_md() {
        let dir = scratch("present");
        std::fs::write(dir.join("AGENTS.md"), "a\n").unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "c\n").unwrap();
        assert_eq!(prepare(&dir).unwrap(), Startup::Present);
        assert_eq!(
            std::fs::read_to_string(dir.join("AGENTS.md")).unwrap(),
            "a\n"
        );
        assert!(
            dir.join("AGENTS.md")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_file()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dangling_agents_md_link_counts_as_present() {
        let dir = scratch("dangling");
        std::os::unix::fs::symlink("gone.md", dir.join("AGENTS.md")).unwrap();
        std::fs::write(dir.join("CLAUDE.md"), "c\n").unwrap();
        assert_eq!(prepare(&dir).unwrap(), Startup::Present);
        assert_eq!(
            std::fs::read_link(dir.join("AGENTS.md")).unwrap(),
            PathBuf::from("gone.md")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skipped_folder_reports_skipped_and_the_list_is_idempotent() {
        let dir = scratch("skip");
        let list = dir.join("home").join(".plank").join("agentsmd-skip");
        let project = dir.join("proj");
        std::fs::create_dir_all(&project).unwrap();
        assert!(!is_skipped(&project, Some(&list)), "no list yet");
        skip_in(&project, &list).unwrap();
        skip_in(&project, &list).unwrap();
        assert!(is_skipped(&project, Some(&list)));
        assert_eq!(
            std::fs::read_to_string(&list).unwrap().lines().count(),
            1,
            "a second skip must not duplicate the line"
        );
        // Another folder is not affected, and no list means nothing is skipped.
        assert!(!is_skipped(&dir, Some(&list)));
        assert!(!is_skipped(&project, None));
        // `prepare` itself consults the real list, which this test must not
        // touch; the classification it feeds is covered above.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn offer_options_lead_with_generate() {
        assert_eq!(OFFER_OPTIONS[0].0, Offer::Generate);
        assert_eq!(OFFER_OPTIONS.len(), 3);
    }

    #[test]
    fn neither_file_reports_missing_and_writes_nothing() {
        let dir = scratch("missing");
        assert_eq!(prepare(&dir).unwrap(), Startup::Missing);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

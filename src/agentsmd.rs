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
}

/// Prepares `AGENTS.md` in `dir`, linking it to `CLAUDE.md` when only the
/// latter exists.
///
/// The link target is the bare name `CLAUDE.md`, not an absolute path, so the
/// checkout can move and the link still resolves. A dangling `AGENTS.md`
/// symlink counts as present: it is the user's to fix, and replacing it would
/// destroy their intent.
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
        return Ok(Startup::Missing);
    }
    std::os::unix::fs::symlink("CLAUDE.md", &agents)
        .map_err(|e| format!("cannot link AGENTS.md to CLAUDE.md: {e}"))?;
    Ok(Startup::Linked(agents))
}

/// The question put to the user when a project has no `AGENTS.md`.
pub const OFFER_QUESTION: &str = "This project has no AGENTS.md. Generate one now? The model will read the codebase and write it.";

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
    fn neither_file_reports_missing_and_writes_nothing() {
        let dir = scratch("missing");
        assert_eq!(prepare(&dir).unwrap(), Startup::Missing);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

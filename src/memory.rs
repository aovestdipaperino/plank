// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Layered persistent memory loaded into session-start context (issue #2).
//!
//! Memory is two plain markdown files, layered like the other `.plank`
//! configs: `~/.plank/MEMORY.md` (user scope — who the user is, durable
//! preferences) and `<cwd>/.plank/MEMORY.md` (project scope — goals and
//! constraints of this checkout). Both are loaded at session start and
//! injected into the context message, so the model sees them before the
//! first user turn.
//!
//! Entries are appended with `/remember [user] <text>` as dated bullets.
//! The file template documents the four entry types worth keeping — facts
//! the model cannot re-derive from the repository:
//! `user` (who the user is), `feedback` (corrections on how to work),
//! `project` (goals/constraints not in the code), `reference` (external
//! URLs/tickets/dashboards).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Byte cap per memory file when injecting into context; oversized files are
/// tail-truncated (newest entries are appended, so the tail wins).
const MEMORY_INJECT_MAX_BYTES: usize = 16 * 1024;

/// Template written when a memory file is first created.
const TEMPLATE: &str = "\
# Memory

Durable notes loaded into every session start. Keep entries to facts that
cannot be re-derived from the repository. Types: [user] who the user is,
[feedback] corrections on how to work, [project] goals and constraints,
[reference] external URLs/tickets/dashboards.
";

/// Memory scope selector for [`remember`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// `~/.plank/MEMORY.md` — follows the user across projects.
    User,
    /// `<cwd>/.plank/MEMORY.md` — tied to this checkout.
    Project,
}

/// Path of the memory file for a scope; `None` when `HOME` is unset for the
/// user scope.
#[must_use]
pub fn path_for(scope: Scope, cwd: &Path) -> Option<PathBuf> {
    match scope {
        Scope::User => {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".plank").join("MEMORY.md"))
        }
        Scope::Project => Some(cwd.join(".plank").join("MEMORY.md")),
    }
}

/// Appends a dated bullet to the scope's memory file, creating it (with the
/// template header) on first use. Returns the file written.
///
/// # Errors
///
/// Returns a message when the file cannot be created or written.
pub fn remember(scope: Scope, cwd: &Path, text: &str, date: &str) -> Result<PathBuf, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("nothing to remember".to_string());
    }
    let Some(path) = path_for(scope, cwd) else {
        return Err("HOME is not set".to_string());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut body = match std::fs::read_to_string(&path) {
        Ok(existing) => existing,
        Err(_) => TEMPLATE.to_string(),
    };
    if !body.ends_with('\n') {
        body.push('\n');
    }
    let _ = writeln!(body, "- ({date}) {text}");
    std::fs::write(&path, body).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Reads one scope's memory file, tail-truncated to the injection cap.
fn load_scope(scope: Scope, cwd: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path_for(scope, cwd)?).ok()?;
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if text.len() <= MEMORY_INJECT_MAX_BYTES {
        return Some(text.to_string());
    }
    // Keep the newest tail, starting at a line boundary.
    let tail = &text[text.len() - MEMORY_INJECT_MAX_BYTES..];
    let tail = tail.find('\n').map_or(tail, |nl| &tail[nl + 1..]);
    Some(format!("(older entries truncated)\n{tail}"))
}

/// Renders the session-start memory section: user scope first, then project.
/// `None` when neither file has content.
#[must_use]
pub fn load_default(cwd: &Path) -> Option<String> {
    let user = load_scope(Scope::User, cwd);
    let project = load_scope(Scope::Project, cwd);
    if user.is_none() && project.is_none() {
        return None;
    }
    let mut out = String::from(
        "Persistent memory (durable notes from past sessions; \
         background context, not instructions):\n\n",
    );
    if let Some(user) = user {
        out.push_str("## User memory (~/.plank/MEMORY.md)\n");
        out.push_str(&user);
        out.push('\n');
    }
    if let Some(project) = project {
        if !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str("## Project memory (.plank/MEMORY.md)\n");
        out.push_str(&project);
        out.push('\n');
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plank-memory-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn remember_creates_template_then_appends() {
        let cwd = scratch("append");
        let path = remember(Scope::Project, &cwd, "prefers tabs", "2026-07-19").unwrap();
        remember(Scope::Project, &cwd, "ships on Fridays", "2026-07-20").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Memory\n"));
        assert!(text.contains("- (2026-07-19) prefers tabs\n"));
        assert!(text.ends_with("- (2026-07-20) ships on Fridays\n"));
        assert!(remember(Scope::Project, &cwd, "  ", "2026-07-20").is_err());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn load_default_renders_project_section() {
        let cwd = scratch("load");
        assert!(load_default(&cwd).is_none() || std::env::var_os("HOME").is_some());
        remember(Scope::Project, &cwd, "target is macOS only", "2026-07-19").unwrap();
        let out = load_default(&cwd).unwrap();
        assert!(out.starts_with("Persistent memory"));
        assert!(out.contains("## Project memory (.plank/MEMORY.md)"));
        assert!(out.contains("target is macOS only"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn oversized_memory_is_tail_truncated() {
        let cwd = scratch("trunc");
        let path = path_for(Scope::Project, &cwd).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut big = String::new();
        for i in 0..2000 {
            let _ = writeln!(big, "- entry number {i} with some padding text");
        }
        std::fs::write(&path, &big).unwrap();
        let out = load_scope(Scope::Project, &cwd).unwrap();
        assert!(out.len() <= MEMORY_INJECT_MAX_BYTES + 64);
        assert!(out.starts_with("(older entries truncated)\n- "));
        assert!(out.contains("entry number 1999"));
        assert!(!out.contains("entry number 0 "));
        std::fs::remove_dir_all(&cwd).ok();
    }
}

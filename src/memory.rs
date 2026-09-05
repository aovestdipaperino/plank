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
    let cut = crate::session::ceil_char_boundary(text, text.len() - MEMORY_INJECT_MAX_BYTES);
    let tail = &text[cut..];
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

// ---------------------------------------------------------------------------
// /memory: one editable view over every source, split back on save
// ---------------------------------------------------------------------------

/// One memory file as `/memory` sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    /// Which scope the file is.
    pub scope: Scope,
    /// Where it lives.
    pub path: PathBuf,
}

impl Scope {
    /// The word used inside the section markers.
    #[must_use]
    pub fn marker_name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
        }
    }

    fn from_marker_name(name: &str) -> Option<Self> {
        match name {
            "user" => Some(Self::User),
            "project" => Some(Self::Project),
            _ => None,
        }
    }
}

/// The memory sources for a checkout, user scope first. The user scope is
/// absent only when `HOME` is unset.
#[must_use]
pub fn sources_for(cwd: &Path) -> Vec<Source> {
    [Scope::User, Scope::Project]
        .into_iter()
        .filter_map(|scope| path_for(scope, cwd).map(|path| Source { scope, path }))
        .collect()
}

/// Opening line of the combined view, explaining the markup to whoever edits it.
const COMBINED_HEADER: &str = "\
<!-- plank memory: every source in one file. Edit inside the sections; each
     section is written back to the file named on its begin marker. Text you
     add between or after sections joins the section above it. Delete a
     section's markers to leave that file untouched. -->
";

/// Builds the combined `/memory` view: every source's full text (never the
/// injection-truncated form) between markers that name its scope and file.
/// A missing file appears as an empty section, so it can be created by
/// typing into it.
#[must_use]
pub fn combine(sources: &[Source]) -> String {
    let mut out = String::from(COMBINED_HEADER);
    for src in sources {
        let body = std::fs::read_to_string(&src.path).unwrap_or_default();
        let _ = write!(
            out,
            "\n<!-- plank-memory: begin {} {} -->\n",
            src.scope.marker_name(),
            src.path.display()
        );
        out.push_str(body.trim_end_matches('\n'));
        if !body.trim().is_empty() {
            out.push('\n');
        }
        let _ = writeln!(
            out,
            "<!-- plank-memory: end {} -->",
            src.scope.marker_name()
        );
    }
    out
}

/// Parses a begin marker line into its scope; the path after the scope is
/// informational and ignored, so a hand-edited path can never redirect a write.
fn parse_begin(line: &str) -> Option<Scope> {
    let rest = line.trim().strip_prefix("<!-- plank-memory: begin ")?;
    let name = rest.split_whitespace().next()?;
    Scope::from_marker_name(name)
}

fn parse_end(line: &str) -> Option<Scope> {
    let rest = line.trim().strip_prefix("<!-- plank-memory: end ")?;
    let name = rest.strip_suffix("-->")?.trim();
    Scope::from_marker_name(name)
}

/// Splits an edited combined view back into per-scope bodies.
///
/// The smart part is what happens to text outside the markers: lines between
/// or after sections are appended to the section above them (someone typing
/// a note under the project block meant it for the project file), and lines
/// before the first section, other than the header comment, go to the first
/// section. A scope whose markers were deleted is simply absent from the
/// result and its file is left alone. Returns an error only for markup that
/// cannot be interpreted: a begin without its end, or a scope opened twice.
///
/// # Errors
///
/// See above.
pub fn split(edited: &str) -> Result<Vec<(Scope, String)>, String> {
    let mut sections: Vec<(Scope, String)> = Vec::new();
    let mut open: Option<Scope> = None;
    let mut preamble = String::new();
    let mut in_header = false;
    // Blank lines right after an end marker are spacing between sections,
    // not content; they are dropped until the first real stray line.
    let mut just_closed = false;
    for line in edited.lines() {
        if let Some(scope) = parse_begin(line) {
            if open.is_some() {
                return Err(format!(
                    "memory markup: begin {} inside another section",
                    scope.marker_name()
                ));
            }
            if sections.iter().any(|(s, _)| *s == scope) {
                return Err(format!(
                    "memory markup: section {} appears twice",
                    scope.marker_name()
                ));
            }
            sections.push((scope, String::new()));
            open = Some(scope);
            continue;
        }
        if let Some(scope) = parse_end(line) {
            if open != Some(scope) {
                return Err(format!(
                    "memory markup: end {} without its begin",
                    scope.marker_name()
                ));
            }
            open = None;
            just_closed = true;
            continue;
        }
        if open.is_some() {
            if let Some((_, body)) = sections.last_mut() {
                body.push_str(line);
                body.push('\n');
            }
        } else if let Some((_, body)) = sections.last_mut() {
            // After a closed section: the note belongs to the block above.
            if just_closed && line.trim().is_empty() {
                continue;
            }
            just_closed = false;
            body.push_str(line);
            body.push('\n');
        } else {
            // Before the first section: skip the explanatory header comment,
            // keep anything the user typed.
            let t = line.trim();
            if t.starts_with("<!-- plank memory:") {
                in_header = true;
            }
            if in_header {
                if t.ends_with("-->") {
                    in_header = false;
                }
                continue;
            }
            preamble.push_str(line);
            preamble.push('\n');
        }
    }
    if let Some(scope) = open {
        return Err(format!(
            "memory markup: section {} is never closed",
            scope.marker_name()
        ));
    }
    if let Some((_, body)) = sections.first_mut().filter(|_| !preamble.trim().is_empty()) {
        *body = format!("{preamble}{body}");
    }
    for (_, body) in &mut sections {
        let trimmed = body.trim_end_matches('\n');
        let mut t = trimmed.to_owned();
        if !t.is_empty() {
            t.push('\n');
        }
        *body = t;
    }
    Ok(sections)
}

/// Writes an edited combined view back to its sources. Only files whose body
/// actually changed are written; a file is created when its section gained
/// text. Returns one line per source describing what happened.
///
/// # Errors
///
/// Markup errors from [`split`], or the first write failure.
pub fn apply(sources: &[Source], edited: &str) -> Result<Vec<String>, String> {
    let sections = split(edited)?;
    let mut report = Vec::new();
    for src in sources {
        let Some((_, body)) = sections.iter().find(|(s, _)| *s == src.scope) else {
            report.push(format!(
                "{}: markers removed, left untouched",
                src.path.display()
            ));
            continue;
        };
        let current = std::fs::read_to_string(&src.path).unwrap_or_default();
        let unchanged = current.trim_end_matches('\n') == body.trim_end_matches('\n');
        if unchanged {
            report.push(format!("{}: unchanged", src.path.display()));
            continue;
        }
        if body.is_empty() && current.is_empty() {
            continue;
        }
        if let Some(parent) = src.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&src.path, body).map_err(|e| format!("{}: {e}", src.path.display()))?;
        report.push(format!(
            "{}: wrote {} line(s)",
            src.path.display(),
            body.lines().count()
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_sources(dir: &Path) -> Vec<Source> {
        vec![
            Source {
                scope: Scope::User,
                path: dir.join("user/MEMORY.md"),
            },
            Source {
                scope: Scope::Project,
                path: dir.join("proj/.plank/MEMORY.md"),
            },
        ]
    }

    #[test]
    fn combine_marks_each_source_and_round_trips_unchanged() {
        let dir = scratch("combine");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "# Memory\n- (2026-09-01) likes tabs\n").unwrap();
        let text = combine(&srcs);
        assert!(text.contains("<!-- plank-memory: begin user "));
        assert!(text.contains("<!-- plank-memory: begin project "));
        assert!(text.contains("- (2026-09-01) likes tabs\n<!-- plank-memory: end user -->"));
        let report = apply(&srcs, &text).unwrap();
        assert!(report[0].ends_with("unchanged"), "{report:?}");
        assert!(
            !srcs[1].path.exists(),
            "an empty untouched section creates no file"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edits_are_routed_to_the_right_file_and_strays_join_the_section_above() {
        let dir = scratch("split");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "u1\n").unwrap();
        let mut text = combine(&srcs);
        text = text.replace("u1\n", "u1\nu2\n");
        text = text.replace(
            "<!-- plank-memory: end project -->\n",
            "<!-- plank-memory: end project -->\n\n- typed below the last block\n",
        );
        let report = apply(&srcs, &text).unwrap();
        assert_eq!(std::fs::read_to_string(&srcs[0].path).unwrap(), "u1\nu2\n");
        assert_eq!(
            std::fs::read_to_string(&srcs[1].path).unwrap(),
            "- typed below the last block\n",
            "stray text after the project block lands in the project file"
        );
        assert!(report.iter().all(|l| l.contains("wrote")), "{report:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn removed_markers_leave_that_file_alone_and_broken_markup_is_refused() {
        let dir = scratch("markers");
        let srcs = two_sources(&dir);
        std::fs::create_dir_all(dir.join("user")).unwrap();
        std::fs::write(&srcs[0].path, "keep me\n").unwrap();
        let text = combine(&srcs);
        let start = text.find("<!-- plank-memory: begin user").unwrap();
        let end = text.find("<!-- plank-memory: end user -->\n").unwrap()
            + "<!-- plank-memory: end user -->\n".len();
        let without_user = format!("{}{}", &text[..start], &text[end..]);
        let report = apply(&srcs, &without_user).unwrap();
        assert!(report[0].contains("markers removed"), "{report:?}");
        assert_eq!(std::fs::read_to_string(&srcs[0].path).unwrap(), "keep me\n");
        let unterminated = text.replace("<!-- plank-memory: end project -->\n", "");
        assert!(split(&unterminated).unwrap_err().contains("never closed"));
        let twice =
            format!("{text}<!-- plank-memory: begin user x -->\n<!-- plank-memory: end user -->\n");
        assert!(split(&twice).unwrap_err().contains("twice"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hand_edited_marker_path_cannot_redirect_the_write() {
        let text = "<!-- plank-memory: begin project /etc/passwd -->\nx\n<!-- plank-memory: end project -->\n";
        assert_eq!(
            split(text).unwrap(),
            vec![(Scope::Project, "x\n".to_owned())]
        );
    }

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

    /// The tail cut lands on a byte offset; when that byte is inside a
    /// multibyte character the slice must snap forward rather than panic.
    #[test]
    fn oversized_multibyte_memory_is_truncated_on_a_char_boundary() {
        let cwd = scratch("trunc-multibyte");
        let path = path_for(Scope::Project, &cwd).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        // One long line of em dashes (3 bytes each) so every offset that is not
        // a multiple of three falls inside a character, with no newline to
        // rescue the slice.
        let mut big = String::new();
        for _ in 0..(MEMORY_INJECT_MAX_BYTES / 3 + 500) {
            big.push('\u{2014}');
        }
        // Make `len - MAX` land mid-character: the cap is 1 mod 3, so two
        // trailing ASCII bytes put the cut one byte into an em dash.
        big.push_str("xy");
        assert_eq!((big.len() - MEMORY_INJECT_MAX_BYTES) % 3, 1);
        std::fs::write(&path, &big).unwrap();
        let out = load_scope(Scope::Project, &cwd).unwrap();
        assert!(out.starts_with("(older entries truncated)\n"));
        assert!(out.ends_with("xy"));
        assert!(out.len() <= MEMORY_INJECT_MAX_BYTES + 64);
        std::fs::remove_dir_all(&cwd).ok();
    }
}

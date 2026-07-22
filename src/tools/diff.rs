// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Post-edit diff preview: a compact, git-style change card shown when the
//! `edit` or `write` tool modifies a file.
//!
//! Uses the `similar` crate's Myers diff (`WRITE-TOOL.md`) so multi-hunk edits
//! and lines that survive inside a changed region render as real context, not a
//! del-block/add-block approximation. Hunks carry `@@` headers like `git diff`.

use similar::{ChangeTag, TextDiff};

/// Unchanged context lines kept on each side of a change (git default).
const CONTEXT: usize = 3;
/// Cap on rows rendered per card, so a whole-file rewrite stays compact.
const MAX_ROWS: usize = 200;

/// One rendered diff row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffRow {
    /// A `@@ -old_start,old_len +new_start,new_len @@` hunk header.
    Hunk {
        old_start: usize,
        old_len: usize,
        new_start: usize,
        new_len: usize,
    },
    /// Unchanged line (both sides), with its 1-based old/new line numbers.
    Context {
        old_no: usize,
        new_no: usize,
        text: String,
    },
    /// Removed line, with its 1-based old line number.
    Del { old_no: usize, text: String },
    /// Added line, with its 1-based new line number.
    Add { new_no: usize, text: String },
    /// A "… N more lines …" marker when the card was capped.
    Elision(usize),
}

impl DiffRow {
    /// The line number shown in the gutter: new for context/adds, old for
    /// removals, none for structural rows.
    #[must_use]
    pub fn gutter(&self) -> Option<usize> {
        match self {
            DiffRow::Context { new_no, .. } | DiffRow::Add { new_no, .. } => Some(*new_no),
            DiffRow::Del { old_no, .. } => Some(*old_no),
            DiffRow::Hunk { .. } | DiffRow::Elision(_) => None,
        }
    }
}

/// A change card for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditPreview {
    /// Path as the tool call named it (relative when the model gave it so).
    pub path: String,
    /// True when the file did not exist before (a create), else an update.
    pub created: bool,
    /// Lines added.
    pub added: usize,
    /// Lines removed.
    pub removed: usize,
    /// New file size in bytes, shown in the header for `write` (`None` for
    /// `edit`).
    pub bytes: Option<usize>,
    /// Rows to render, in order.
    pub rows: Vec<DiffRow>,
}

/// Formats a byte count as B / KiB / MiB, like the write header in the docs.
#[must_use]
pub fn human_size(bytes: usize) -> String {
    #[allow(clippy::cast_precision_loss)]
    let b = bytes as f64;
    if bytes >= 1 << 20 {
        format!("{:.2} MiB", b / 1_048_576.0)
    } else if bytes >= 1 << 10 {
        format!("{:.2} KiB", b / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Builds a diff preview between `old` and `new` file contents.
#[must_use]
pub fn edit_preview(path: &str, old: &str, new: &str, created: bool) -> EditPreview {
    let diff = TextDiff::from_lines(old, new);
    let mut added = 0;
    let mut removed = 0;
    let mut rows = Vec::new();
    let mut capped = false;

    for group in &diff.grouped_ops(CONTEXT) {
        let (Some(first), Some(last)) = (group.first(), group.last()) else {
            continue;
        };
        let old_r = first.old_range().start..last.old_range().end;
        let new_r = first.new_range().start..last.new_range().end;
        rows.push(DiffRow::Hunk {
            old_start: old_r.start + 1,
            old_len: old_r.len(),
            new_start: new_r.start + 1,
            new_len: new_r.len(),
        });
        for op in group {
            for change in diff.iter_changes(op) {
                let text = change.value().strip_suffix('\n').unwrap_or(change.value());
                let row = match change.tag() {
                    ChangeTag::Equal => DiffRow::Context {
                        old_no: change.old_index().unwrap_or(0) + 1,
                        new_no: change.new_index().unwrap_or(0) + 1,
                        text: text.to_string(),
                    },
                    ChangeTag::Delete => {
                        removed += 1;
                        DiffRow::Del {
                            old_no: change.old_index().unwrap_or(0) + 1,
                            text: text.to_string(),
                        }
                    }
                    ChangeTag::Insert => {
                        added += 1;
                        DiffRow::Add {
                            new_no: change.new_index().unwrap_or(0) + 1,
                            text: text.to_string(),
                        }
                    }
                };
                if rows.len() < MAX_ROWS {
                    rows.push(row);
                } else {
                    capped = true;
                }
            }
        }
    }
    if capped {
        rows.push(DiffRow::Elision(added + removed + diff_equal(&diff)));
    }

    EditPreview {
        path: path.to_string(),
        created,
        added,
        removed,
        bytes: None,
        rows,
    }
}

impl EditPreview {
    /// Renders the card as an ANSI string for the plain REPL / stdout paths.
    #[must_use]
    pub fn to_ansi(&self, color: bool) -> String {
        use std::fmt::Write as _;
        let verb = if self.created { "Create" } else { "Update" };
        let mut out = String::new();
        let (bold, dim, cyan, del, add, reset) = if color {
            (
                "\x1b[1m",
                "\x1b[38;5;240m",
                "\x1b[38;5;44m",
                "\x1b[48;5;52m\x1b[38;5;224m",
                "\x1b[48;5;22m\x1b[38;5;194m",
                "\x1b[0m",
            )
        } else {
            ("", "", "", "", "", "")
        };
        let size = self
            .bytes
            .map(|b| format!(" · {}", human_size(b)))
            .unwrap_or_default();
        let _ = writeln!(out, "{bold}{verb}({}){reset}{dim}{size}{reset}", self.path);
        let _ = writeln!(
            out,
            "{dim}  └ Added {} {}, removed {} {}{reset}",
            self.added,
            plural(self.added),
            self.removed,
            plural(self.removed),
        );
        for row in &self.rows {
            match row {
                DiffRow::Hunk {
                    old_start,
                    old_len,
                    new_start,
                    new_len,
                } => {
                    let _ = writeln!(
                        out,
                        "{cyan}  @@ -{old_start},{old_len} +{new_start},{new_len} @@{reset}"
                    );
                }
                DiffRow::Context { text, .. } => {
                    let _ = writeln!(out, "{dim}{}{reset}   {text}", gutter(row.gutter()));
                }
                DiffRow::Del { text, .. } => {
                    let _ = writeln!(out, "{del}{} - {text}{reset}", gutter(row.gutter()));
                }
                DiffRow::Add { text, .. } => {
                    let _ = writeln!(out, "{add}{} + {text}{reset}", gutter(row.gutter()));
                }
                DiffRow::Elision(n) => {
                    let _ = writeln!(out, "{dim}      ⋯ {n} more lines ⋯{reset}");
                }
            }
        }
        out
    }
}

/// Pluralizes "line".
#[must_use]
pub fn plural(n: usize) -> &'static str {
    if n == 1 { "line" } else { "lines" }
}

/// Right-aligned 5-column gutter for a line number (blank when absent).
#[must_use]
pub fn gutter(n: Option<usize>) -> String {
    n.map_or_else(|| "     ".to_string(), |n| format!("{n:>5}"))
}

/// Total unchanged lines, only used to size the capped-card elision note.
fn diff_equal(diff: &TextDiff<'_, '_, str>) -> usize {
    diff.iter_all_changes()
        .filter(|c| c.tag() == ChangeTag::Equal)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_shows_del_then_add_with_context_and_a_hunk_header() {
        let old = "a\nb\nc\nd\ne\n";
        let new = "a\nb\nX\nY\nd\ne\n";
        let p = edit_preview("f.rs", old, new, false);
        assert_eq!(p.removed, 1);
        assert_eq!(p.added, 2);
        assert!(p.rows.iter().any(|r| matches!(r, DiffRow::Hunk { .. })));
        assert!(
            p.rows
                .iter()
                .any(|r| matches!(r, DiffRow::Del { old_no: 3, text } if text == "c"))
        );
        assert!(
            p.rows
                .iter()
                .any(|r| matches!(r, DiffRow::Add { new_no: 3, text } if text == "X"))
        );
        let del = p
            .rows
            .iter()
            .find(|r| matches!(r, DiffRow::Del { .. }))
            .unwrap();
        assert_eq!(del.gutter(), Some(3));
    }

    #[test]
    fn create_is_all_additions() {
        let p = edit_preview("new.rs", "", "one\ntwo\n", true);
        assert!(p.created);
        assert_eq!(p.removed, 0);
        assert_eq!(p.added, 2);
        assert!(
            p.rows
                .iter()
                .all(|r| matches!(r, DiffRow::Add { .. } | DiffRow::Hunk { .. }))
        );
    }

    #[test]
    fn distant_changes_produce_two_hunks() {
        let mut old = String::new();
        for i in 0..40 {
            use std::fmt::Write as _;
            let _ = writeln!(old, "line {i}");
        }
        let mut lines: Vec<String> = (0..40).map(|i| format!("line {i}")).collect();
        lines[2] = "CHANGED early".to_string();
        lines[37] = "CHANGED late".to_string();
        let mut new = String::new();
        for l in &lines {
            new.push_str(l);
            new.push('\n');
        }
        let p = edit_preview("f.rs", &old, &new, false);
        let hunks = p
            .rows
            .iter()
            .filter(|r| matches!(r, DiffRow::Hunk { .. }))
            .count();
        assert_eq!(hunks, 2, "far-apart edits should not merge into one hunk");
    }
}

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
/// Changed fraction of a paired line's combined length above which intra-line
/// (word/char) highlighting is skipped in favor of full-line Del/Add. Per the
/// diff-rendering spec, a line that is effectively rewritten reads better fully
/// highlighted than as noisy per-word emphasis.
const INTRA_LINE_MAX_RATIO: f64 = 0.40;

/// How a run of characters inside a paired line is classified for intra-line
/// (word-level) highlighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    /// Text present on both sides of the pair — gets the row background but no
    /// word emphasis.
    Common,
    /// Text only on the old (`-`) side — emphasized on the Del line.
    Removed,
    /// Text only on the new (`+`) side — emphasized on the Add line.
    Added,
}

/// One intra-line segment: a maximal run of characters sharing a [`SegKind`].
/// Segment text is already control-character-free (sanitize discipline), so it
/// can be rendered directly without reflowing a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub kind: SegKind,
    pub text: String,
}

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
    /// Removed line, with its 1-based old line number. `segments` is `Some` when
    /// this line was paired with an adjacent Add and the change is small enough
    /// (below [`INTRA_LINE_MAX_RATIO`]) for word-level highlighting; `None` means
    /// render the whole line as a removal.
    Del {
        old_no: usize,
        text: String,
        segments: Option<Vec<Segment>>,
    },
    /// Added line, with its 1-based new line number. `segments` mirrors
    /// [`DiffRow::Del`] for the added side.
    Add {
        new_no: usize,
        text: String,
        segments: Option<Vec<Segment>>,
    },
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
                            segments: None,
                        }
                    }
                    ChangeTag::Insert => {
                        added += 1;
                        DiffRow::Add {
                            new_no: change.new_index().unwrap_or(0) + 1,
                            text: text.to_string(),
                            segments: None,
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
    annotate_intra_line(&mut rows);

    EditPreview {
        path: path.to_string(),
        created,
        added,
        removed,
        bytes: None,
        rows,
    }
}

/// Pairs each maximal run of consecutive `Del` rows with the `Add` run that
/// immediately follows it (a line replacement, as Myers emits for edits), and
/// fills in word-level [`Segment`]s on both sides when the change is small
/// enough. Rows are paired positionally (the k-th Del with the k-th Add);
/// unpaired rows and rewrites above [`INTRA_LINE_MAX_RATIO`] keep `segments:
/// None` and render at line granularity, as before.
fn annotate_intra_line(rows: &mut [DiffRow]) {
    let mut i = 0;
    while i < rows.len() {
        if !matches!(rows[i], DiffRow::Del { .. }) {
            i += 1;
            continue;
        }
        // Extent of the Del run [i, dend) and the Add run [dend, aend).
        let mut dend = i;
        while dend < rows.len() && matches!(rows[dend], DiffRow::Del { .. }) {
            dend += 1;
        }
        let mut aend = dend;
        while aend < rows.len() && matches!(rows[aend], DiffRow::Add { .. }) {
            aend += 1;
        }
        let dels = dend - i;
        let adds = aend - dend;
        if adds == 0 {
            // A pure deletion (no following Add run) — nothing to pair.
            i = dend;
            continue;
        }
        for k in 0..dels.min(adds) {
            let old_text = match &rows[i + k] {
                DiffRow::Del { text, .. } => text.clone(),
                _ => continue,
            };
            let new_text = match &rows[dend + k] {
                DiffRow::Add { text, .. } => text.clone(),
                _ => continue,
            };
            if let Some((del_segs, add_segs)) = intra_line_segments(&old_text, &new_text) {
                if let DiffRow::Del { segments, .. } = &mut rows[i + k] {
                    *segments = Some(del_segs);
                }
                if let DiffRow::Add { segments, .. } = &mut rows[dend + k] {
                    *segments = Some(add_segs);
                }
            }
        }
        i = aend;
    }
}

/// Computes word-level intra-line highlighting for a replaced `(old, new)` line
/// pair. Both sides are first stripped of control characters (sanitize
/// discipline), then diffed at word granularity. Returns `(del_segments,
/// add_segments)` where the Del side carries `Common`/`Removed` runs and the Add
/// side `Common`/`Added` runs.
///
/// Returns `None` — signalling the caller to fall back to full-line Del/Add —
/// when the lines are identical after sanitizing, or when the changed fraction
/// of the combined length exceeds [`INTRA_LINE_MAX_RATIO`] (an effective
/// rewrite, which reads better fully highlighted than word-by-word).
///
/// Pure function of its inputs; unit-tested directly.
#[must_use]
pub fn intra_line_segments(old: &str, new: &str) -> Option<(Vec<Segment>, Vec<Segment>)> {
    let o: String = old.chars().filter(|c| !c.is_control()).collect();
    let n: String = new.chars().filter(|c| !c.is_control()).collect();
    if o == n {
        return None;
    }

    let diff = TextDiff::from_words(o.as_str(), n.as_str());
    let mut del: Vec<Segment> = Vec::new();
    let mut add: Vec<Segment> = Vec::new();
    let mut removed_chars = 0usize;
    let mut added_chars = 0usize;
    for change in diff.iter_all_changes() {
        let value = change.value();
        match change.tag() {
            ChangeTag::Equal => {
                push_segment(&mut del, SegKind::Common, value);
                push_segment(&mut add, SegKind::Common, value);
            }
            ChangeTag::Delete => {
                removed_chars += value.chars().count();
                push_segment(&mut del, SegKind::Removed, value);
            }
            ChangeTag::Insert => {
                added_chars += value.chars().count();
                push_segment(&mut add, SegKind::Added, value);
            }
        }
    }

    let combined = o.chars().count() + n.chars().count();
    if combined == 0 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio = (removed_chars + added_chars) as f64 / combined as f64;
    if ratio > INTRA_LINE_MAX_RATIO {
        return None;
    }
    Some((del, add))
}

/// Appends `text` to `segs`, coalescing with the trailing segment when it shares
/// `kind` so adjacent same-class word tokens render as one run.
fn push_segment(segs: &mut Vec<Segment>, kind: SegKind, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = segs.last_mut()
        && last.kind == kind
    {
        last.text.push_str(text);
        return;
    }
    segs.push(Segment {
        kind,
        text: text.to_string(),
    });
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
        // Emphasis variants for the changed span of a word-diffed pair: a
        // brighter background plus bold, so the common text (base bg) recedes.
        let (del_emph, add_emph) = if color {
            (
                "\x1b[48;5;88m\x1b[38;5;231m\x1b[1m",
                "\x1b[48;5;28m\x1b[38;5;231m\x1b[1m",
            )
        } else {
            ("", "")
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
                DiffRow::Del { text, segments, .. } => {
                    let _ = write!(out, "{del}{} - ", gutter(row.gutter()));
                    write_ansi_diff_line(&mut out, text, segments.as_deref(), del, del_emph, reset);
                    let _ = writeln!(out, "{reset}");
                }
                DiffRow::Add { text, segments, .. } => {
                    let _ = write!(out, "{add}{} + ", gutter(row.gutter()));
                    write_ansi_diff_line(&mut out, text, segments.as_deref(), add, add_emph, reset);
                    let _ = writeln!(out, "{reset}");
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

/// Writes the text portion of a Del/Add ANSI row into `out`. With word-level
/// `segments` and color enabled, each changed run is wrapped in `emph` while
/// common runs use `base`; without segments (or without color) the raw line is
/// written unchanged, so the plain path degrades to today's line-level output.
/// The caller has already emitted the gutter/sigil in `base` and appends the
/// final reset.
fn write_ansi_diff_line(
    out: &mut String,
    text: &str,
    segments: Option<&[Segment]>,
    base: &str,
    emph: &str,
    reset: &str,
) {
    use std::fmt::Write as _;
    match segments {
        // No word emphasis available (plain path / rewrite): line-level as before.
        Some(segs) if !base.is_empty() => {
            for seg in segs {
                let code = match seg.kind {
                    SegKind::Removed | SegKind::Added => emph,
                    SegKind::Common => base,
                };
                let _ = write!(out, "{reset}{code}{}", seg.text);
            }
        }
        _ => {
            let _ = write!(out, "{text}");
        }
    }
}

/// A compact multi-line summary of the first changes between `old` and `new`:
/// up to `max_lines` changed lines, each prefixed `-`/`+` and sanitized so the
/// warm-up display stays readable (control characters — including embedded
/// CR/NL from MCP tool schemas — are stripped, and long lines are truncated
/// with `…`). Built on the same [`edit_preview`] machinery as the code-diff
/// cards. Returns `None` when the texts are identical.
#[must_use]
pub fn first_change_snippet(old: &str, new: &str, max_lines: usize) -> Option<String> {
    const WIDTH: usize = 80;
    let preview = edit_preview("", old, new, false);
    let rows = &preview.rows;
    let mut lines = Vec::new();
    // Count only changed (Del/Add) rows against `max_lines`; the caret pointer a
    // windowed pair emits is decoration, not another change.
    let mut changed = 0;
    let mut i = 0;
    while i < rows.len() && changed < max_lines {
        match &rows[i] {
            // A Del immediately followed by an Add is a line replacement: window
            // both sides around the characters that actually differ (with a caret
            // pointing at them) so a change past the truncation column is visible
            // instead of hidden behind an identical-looking `…`.
            DiffRow::Del { text: del, .. } => {
                if let Some(DiffRow::Add { text: add, .. }) = rows.get(i + 1) {
                    let (dline, aline, caret) = windowed_change_pair(del, add, WIDTH);
                    lines.push(format!("- {dline}"));
                    lines.push(format!("+ {aline}"));
                    lines.push(caret);
                    changed += 2;
                    i += 2;
                    continue;
                }
                lines.push(format!("- {}", sanitize_line(del, WIDTH)));
                changed += 1;
            }
            DiffRow::Add { text, .. } => {
                lines.push(format!("+ {}", sanitize_line(text, WIDTH)));
                changed += 1;
            }
            _ => {}
        }
        i += 1;
    }
    if lines.is_empty() {
        return None;
    }
    let total = preview.added + preview.removed;
    if total > changed {
        let extra = total - changed;
        lines.push(format!("… (+{extra} more changed {})", plural(extra)));
    }
    Some(lines.join("\n"))
}

/// Renders a replaced line as an aligned `(old, new, caret)` triple, each window
/// centered on the characters that differ. Both sides are stripped of control
/// characters, then a common-prefix/suffix scan locates the changed span; if the
/// new line is longer than `width`, the window slides to keep that span on-screen
/// (elided head/tail marked with `…`). The caret row underlines the changed
/// characters beneath the rendered `+ ` line.
#[cfg_attr(not(ds4_engine), allow(dead_code))]
fn windowed_change_pair(old: &str, new: &str, width: usize) -> (String, String, String) {
    let o: Vec<char> = old.chars().filter(|c| !c.is_control()).collect();
    let n: Vec<char> = new.chars().filter(|c| !c.is_control()).collect();

    // Longest common prefix, then longest common suffix that does not overlap it.
    let prefix = o.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let max_suffix = o.len().min(n.len()) - prefix;
    let mut suffix = 0;
    while suffix < max_suffix && o[o.len() - 1 - suffix] == n[n.len() - 1 - suffix] {
        suffix += 1;
    }
    // Changed span in the new line: [prefix, change_end).
    let change_end = n.len() - suffix;

    // Slide the window so the change stays visible, keeping a little lead-in.
    let mut start = 0;
    if n.len() > width {
        // Keep a little lead-in context before the change.
        const LEAD: usize = 16;
        start = prefix.saturating_sub(LEAD);
        if change_end > start + width {
            start = change_end - width;
        }
        start = start.min(n.len() - width);
    }

    // Column where the change begins on the rendered `+ ` line: the "+ " gutter,
    // plus a leading `…` when the head was elided, plus the in-window offset.
    //
    // A change wider than the window slides `start` past `prefix`, so the change
    // begins *before* the visible span. Anchor the caret at the first visible
    // column in that case, and measure its length from the same point — `prefix`
    // alone would underflow the column and overstate the length.
    let gutter = 2;
    let lead_ellipsis = usize::from(start > 0);
    let caret_from = prefix.max(start);
    let caret_col = gutter + lead_ellipsis + (caret_from - start);
    let visible_change_end = change_end.min(start + width);
    let caret_len = visible_change_end.saturating_sub(caret_from).max(1);
    let caret = format!("{}{}", " ".repeat(caret_col), "^".repeat(caret_len));

    (window(&o, start, width), window(&n, start, width), caret)
}

/// Slices `width` characters of `chars` starting at `start`, marking an elided
/// head and/or tail with `…`. `start` is a character index shared with the
/// paired line so their common prefix stays column-aligned.
#[cfg_attr(not(ds4_engine), allow(dead_code))]
fn window(chars: &[char], start: usize, width: usize) -> String {
    if start >= chars.len() {
        return if start > 0 {
            "…".to_string()
        } else {
            String::new()
        };
    }
    let end = (start + width).min(chars.len());
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.extend(&chars[start..end]);
    if end < chars.len() {
        out.push('…');
    }
    out
}

/// One diff line made safe for a single terminal row: control characters
/// (CR, NL, tab, …) removed so nothing reflows or garbles the display, then
/// truncated to `width` characters with a trailing `…` when trimmed.
#[cfg_attr(not(ds4_engine), allow(dead_code))]
fn sanitize_line(s: &str, width: usize) -> String {
    let clean: String = s.chars().filter(|c| !c.is_control()).collect();
    let mut out: String = clean.chars().take(width).collect();
    if clean.chars().count() > width {
        out.push('…');
    }
    out
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
    fn first_change_snippet_is_bounded_and_sanitized() {
        // A changed line carrying embedded control chars (CR/tab) — as MCP tool
        // schemas can — must render on one clean row, not reflow the display.
        let old = "intro line\nold tool: does\ta\rthing\ntail\n";
        let new = "intro line\nnew tool: does\ta\rthing\ntail\n";
        let snip = first_change_snippet(old, new, 5).expect("differs");
        assert!(
            snip.lines().any(|l| l.starts_with("- ")),
            "shows removal: {snip:?}"
        );
        assert!(
            snip.lines().any(|l| l.starts_with("+ ")),
            "shows addition: {snip:?}"
        );
        // No control characters survive into the rendered snippet.
        assert!(
            !snip.contains('\r') && !snip.contains('\t'),
            "sanitized: {snip:?}"
        );

        // Bounded to max_lines changed rows, plus a one-line elision summary.
        let mut o = String::new();
        let mut n = String::new();
        for i in 0..20 {
            let _ = std::fmt::Write::write_fmt(&mut o, format_args!("line {i}\n"));
            let _ = std::fmt::Write::write_fmt(&mut n, format_args!("LINE {i}\n"));
        }
        let capped = first_change_snippet(&o, &n, 5).expect("differs");
        assert!(capped.lines().count() <= 6, "capped: {capped}");
        assert!(capped.contains("more changed"), "notes elision: {capped}");

        // Identical texts have no change to show.
        assert!(first_change_snippet(old, old, 5).is_none());
    }

    #[test]
    fn snippet_windows_a_change_past_the_truncation_column() {
        // A long line whose only edit sits well past column 80: the naive
        // truncate-from-start rendering showed both sides identically (the change
        // hidden behind `…`). The window must slide so the differing text is
        // visible, with a caret pointing at it.
        let head = "\"description\": \"Build an AI-ready context for a task description. Returns ";
        let old = format!("{head}relevant symbols and Xold snippets.\"\n");
        let new = format!("{head}relevant symbols and Ynew snippets.\"\n");
        let snip = first_change_snippet(&old, &new, 5).expect("differs");

        let minus = snip.lines().find(|l| l.starts_with("- ")).expect("del");
        let plus = snip.lines().find(|l| l.starts_with("+ ")).expect("add");
        // The two rendered change lines must not look identical anymore.
        assert_ne!(minus, plus, "change is visible, not hidden: {snip:?}");
        assert!(minus.contains("Xold"), "old change shown: {minus:?}");
        assert!(plus.contains("Ynew"), "new change shown: {plus:?}");

        // A caret row points at the changed characters under the `+` line.
        let caret = snip
            .lines()
            .find(|l| l.trim_start().starts_with('^'))
            .expect("caret");
        let caret_col = caret.find('^').unwrap();
        // The caret sits under the first differing char ('Y') of the `+` line.
        assert_eq!(
            plus.chars().nth(caret_col),
            Some('Y'),
            "caret under the change: plus={plus:?} caret={caret:?}"
        );
    }

    // Collapses a segment list into a `(kind, text)` vector for terse asserts.
    fn segs(v: &[Segment]) -> Vec<(SegKind, &str)> {
        v.iter().map(|s| (s.kind, s.text.as_str())).collect()
    }

    #[test]
    fn intra_line_segments_isolates_a_word_change() {
        let (del, add) =
            intra_line_segments("foo bar baz", "foo qux baz").expect("minor change highlights");
        assert_eq!(
            segs(&del),
            vec![
                (SegKind::Common, "foo "),
                (SegKind::Removed, "bar"),
                (SegKind::Common, " baz"),
            ]
        );
        assert_eq!(
            segs(&add),
            vec![
                (SegKind::Common, "foo "),
                (SegKind::Added, "qux"),
                (SegKind::Common, " baz"),
            ]
        );
        // The del line carries no Added spans and the add line no Removed spans.
        assert!(del.iter().all(|s| s.kind != SegKind::Added));
        assert!(add.iter().all(|s| s.kind != SegKind::Removed));
    }

    #[test]
    fn intra_line_segments_handles_a_pure_insertion_at_the_end() {
        let (del, add) = intra_line_segments("let x = 1;", "let x = 1; // note")
            .expect("appended text highlights");
        // Nothing removed on the old side; the whole old line is common.
        assert_eq!(segs(&del), vec![(SegKind::Common, "let x = 1;")]);
        assert_eq!(
            segs(&add),
            vec![
                (SegKind::Common, "let x = 1;"),
                (SegKind::Added, " // note"),
            ]
        );
    }

    #[test]
    fn intra_line_segments_falls_back_above_the_ratio_threshold() {
        // A near-total rewrite: the changed fraction exceeds 40%, so intra-line
        // highlighting is declined and the caller paints full Del/Add.
        assert!(intra_line_segments("abcdefghij", "0123456789").is_none());
        // Identical lines have nothing to highlight.
        assert!(intra_line_segments("same", "same").is_none());
    }

    #[test]
    fn intra_line_segments_strips_control_characters() {
        // Embedded CR/tab must never reach a rendered segment (sanitize discipline).
        let (del, add) = intra_line_segments("value\tis old\r", "value\tis new\r")
            .expect("differs after sanitizing");
        for s in del.iter().chain(add.iter()) {
            assert!(
                !s.text.contains('\t') && !s.text.contains('\r'),
                "no control chars leak: {s:?}"
            );
        }
    }

    #[test]
    fn edit_preview_annotates_a_minor_replacement_with_segments() {
        let old = "aaa\nlet total = added + removed;\nzzz\n";
        let new = "aaa\nlet total = added + kept;\nzzz\n";
        let p = edit_preview("f.rs", old, new, false);
        let del = p
            .rows
            .iter()
            .find_map(|r| match r {
                DiffRow::Del { segments, .. } => segments.as_ref(),
                _ => None,
            })
            .expect("del row carries segments");
        assert!(del.iter().any(|s| s.kind == SegKind::Removed));
        let add = p
            .rows
            .iter()
            .find_map(|r| match r {
                DiffRow::Add { segments, .. } => segments.as_ref(),
                _ => None,
            })
            .expect("add row carries segments");
        assert!(add.iter().any(|s| s.kind == SegKind::Added));
    }

    #[test]
    fn edit_preview_declines_segments_for_a_full_line_rewrite() {
        let old = "aaa\nabcdefghij\nzzz\n";
        let new = "aaa\n0123456789\nzzz\n";
        let p = edit_preview("f.rs", old, new, false);
        for r in &p.rows {
            match r {
                DiffRow::Del { segments, .. } | DiffRow::Add { segments, .. } => {
                    assert!(segments.is_none(), "rewrite stays full-line: {r:?}");
                }
                _ => {}
            }
        }
    }

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
                .any(|r| matches!(r, DiffRow::Del { old_no: 3, text, .. } if text == "c"))
        );
        assert!(
            p.rows
                .iter()
                .any(|r| matches!(r, DiffRow::Add { new_no: 3, text, .. } if text == "X"))
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

    /// A changed span wider than the window pushed `start` past `prefix`, and
    /// `prefix - start` underflowed. Reachable from the Tier-1 cache-miss
    /// system-prompt diff, where lines are long and edits are wide.
    #[test]
    fn windowed_change_pair_survives_a_change_wider_than_the_window() {
        let old = format!("{}{}", "same ", "A".repeat(400));
        let new = format!("{}{}", "same ", "B".repeat(400));
        let (o, n, caret) = windowed_change_pair(&old, &new, 80);
        assert!(!o.is_empty() && !n.is_empty());
        assert!(caret.contains('^'), "caret marks something: {caret:?}");
    }

    /// A one-sided line (the other empty) makes `max_suffix` subtract from zero.
    #[test]
    fn windowed_change_pair_survives_an_empty_side() {
        for (a, b) in [("", "added text"), ("removed text", "")] {
            let (o, n, caret) = windowed_change_pair(a, b, 80);
            let _ = (o, n, caret);
        }
    }
}

// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Edit and search tools: unique old/new replacement and recursive grep.
//!
//! Port of the "Edit And Search Tools" section of `ds4_agent.c`. The edit
//! tool is intentionally conservative: the exact old text must be unique, or
//! use one `[upto]` marker whose head is unique and whose tail is unique
//! after the head. Successful edits echo nearby post-edit context so the
//! model immediately sees shifted line numbers.

use std::fmt::Write as _;
use std::path::Path;

use crate::dsml::ToolCall;

use super::files::{LineSpan, read_file_bytes, split_lines};
use super::{ToolContext, parse_bool_default, parse_int_default};

const UPTO_MARKER: &[u8] = b"[upto]";
const CONTEXT_BEFORE: usize = 5;
const CONTEXT_AFTER: usize = 8;
const EDITED_CONTEXT_HEAD: usize = 18;
const EDITED_CONTEXT_TAIL: usize = 18;

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Finds a globally unique occurrence of `needle`, mirroring
/// `agent_find_unique`.
fn find_unique(data: &[u8], needle: &[u8], label: &str) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    let Some(first) = find_bytes(data, needle) else {
        return Err(format!("{label} anchor not found"));
    };
    if find_bytes(&data[first + 1..], needle).is_some() {
        return Err(format!("{label} anchor is not unique"));
    }
    Ok(first)
}

/// Finds a unique occurrence of `needle` at or after `start`, mirroring
/// `agent_find_unique_after`.
fn find_unique_after(
    data: &[u8],
    start: usize,
    needle: &[u8],
    label: &str,
) -> Result<usize, String> {
    if needle.is_empty() {
        return Err(format!("{label} anchor is empty"));
    }
    if start > data.len() {
        return Err(format!("{label} search starts outside file"));
    }
    let Some(off) = find_bytes(&data[start..], needle) else {
        return Err(format!("{label} anchor not found after old head"));
    };
    let first = start + off;
    if find_bytes(&data[first + 1..], needle).is_some() {
        return Err(format!("{label} anchor is not unique after old head"));
    }
    Ok(first)
}

fn span_has_nonspace(s: &[u8]) -> bool {
    s.iter().any(|b| !b.is_ascii_whitespace())
}

/// Locates the file span an edit's old text covers.
///
/// Mirrors `agent_edit_find_old_span`: without `[upto]` the exact old text
/// must be unique; with one marker the head must be unique and the tail
/// unique after the head. Returns `(offset, length, anchored)`.
///
/// # Errors
///
/// Returns the C-format anchor diagnostic when the span cannot be located.
pub fn edit_find_old_span(data: &[u8], old: &[u8]) -> Result<(usize, usize, bool), String> {
    let Some(upto) = find_bytes(old, UPTO_MARKER) else {
        let offset = find_unique(data, old, "old text")?;
        return Ok((offset, old.len(), false));
    };
    let after_marker = upto + UPTO_MARKER.len();
    if find_bytes(&old[after_marker..], UPTO_MARKER).is_some() {
        return Err("old text contains more than one [upto] marker".to_string());
    }
    let head = &old[..upto];
    let mut tail = &old[after_marker..];
    // Strip leading newline/CR from the tail before searching. The head
    // already ends with a newline, so the extra newline after [upto] must
    // not be part of the tail needle.
    while let Some((&first, rest)) = tail.split_first() {
        if first == b'\n' || first == b'\r' {
            tail = rest;
        } else {
            break;
        }
    }
    if !span_has_nonspace(tail) {
        return Err("old text after [upto] must include a unique tail anchor".to_string());
    }
    let head_pos = find_unique(data, head, "old head")?;
    let tail_pos = find_unique_after(data, head_pos + head.len(), tail, "old tail")?;
    Ok((head_pos, tail_pos - head_pos + tail.len(), true))
}

/// Preflights an edit's old text against the current file contents.
///
/// Mirrors `agent_preflight_edit_old`: silently passes while the path is
/// unknown, otherwise validates that the old text resolves to a unique span.
///
/// # Errors
///
/// Returns the anchor diagnostic that the full edit would report.
pub fn preflight_edit_old(ctx: &ToolContext, call: &ToolCall) -> Result<(), String> {
    let path = call.arg_value("path").unwrap_or("");
    if path.is_empty() {
        return Ok(()); // Cannot preflight until path is known.
    }
    let old = call.arg_value("old").unwrap_or("");
    if old.is_empty() {
        return Err("edit requires non-empty old text".to_string());
    }
    let data = read_file_bytes(&ctx.resolve(path), path)?;
    edit_find_old_span(&data, old.as_bytes()).map(|_| ())
}

fn line_for_offset(spans: &[LineSpan], offset: usize) -> usize {
    if spans.is_empty() {
        return 1;
    }
    spans
        .iter()
        .position(|sp| offset < sp.end)
        .map_or(spans.len(), |i| i + 1)
}

/// Computes the old line range an edit touched and the resulting line delta.
fn old_new_line_effect(
    old_data: &[u8],
    new_data: &[u8],
    edit_offset: usize,
    replaced_len: usize,
) -> Option<(usize, usize, i64)> {
    let old_spans = split_lines(old_data);
    let new_spans = split_lines(new_data);
    if old_spans.is_empty() {
        return None;
    }
    let mut old_last = if replaced_len > 0 {
        edit_offset + replaced_len - 1
    } else {
        edit_offset
    };
    if old_last >= old_data.len() {
        old_last = old_data.len().saturating_sub(1);
    }
    let start_line = line_for_offset(&old_spans, edit_offset);
    let end_line = line_for_offset(&old_spans, old_last);
    let delta = i64::try_from(new_spans.len()).unwrap_or(i64::MAX)
        - i64::try_from(old_spans.len()).unwrap_or(i64::MAX);
    Some((start_line, end_line, delta))
}

fn append_numbered_line(out: &mut String, data: &[u8], sp: LineSpan, line: usize) {
    let _ = write!(out, "{line} ");
    out.push_str(&String::from_utf8_lossy(&data[sp.start..sp.content_end]));
    out.push('\n');
}

/// Appends the nearby post-edit file shape, mirroring
/// `agent_edit_result_append_context`.
fn edit_result_append_context(
    out: &mut String,
    path: &str,
    data: &[u8],
    anchor_start: usize,
    anchor_end: usize,
) {
    let spans = split_lines(data);
    if spans.is_empty() {
        return;
    }
    let anchor_start = anchor_start.clamp(1, spans.len());
    let anchor_end = anchor_end.clamp(anchor_start, spans.len());
    let ctx_start = anchor_start.saturating_sub(CONTEXT_BEFORE).max(1);
    let ctx_end = (anchor_end + CONTEXT_AFTER).min(spans.len());

    let _ = writeln!(
        out,
        "Current file around edit: {path} lines {ctx_start}-{ctx_end} of {}",
        spans.len()
    );

    let edited_lines = anchor_end - anchor_start + 1;
    if edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL {
        for line in ctx_start..=ctx_end {
            append_numbered_line(out, data, spans[line - 1], line);
        }
    } else {
        let head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        let tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for line in ctx_start..=head_end {
            append_numbered_line(out, data, spans[line - 1], line);
        }
        let _ = writeln!(
            out,
            "... {} edited lines omitted ...",
            tail_start - head_end - 1
        );
        for line in tail_start..=ctx_end {
            append_numbered_line(out, data, spans[line - 1], line);
        }
    }
}

/// Builds the successful edit observation, mirroring `agent_edit_result`.
fn edit_result(
    path: &str,
    effect: Option<(usize, usize, i64)>,
    new_data: &[u8],
    kind: &str,
) -> String {
    let mut out = format!("Edited {path} using {kind}\n");
    if let Some((start_line, end_line, delta)) = effect
        && start_line > 0
        && end_line >= start_line
    {
        let _ = writeln!(
            out,
            "Touched old lines {start_line}-{end_line}; current post-edit context follows."
        );
        if delta != 0 {
            let _ = writeln!(
                out,
                "Line shift: old lines after {end_line} moved by {delta:+} \
                 (old line {} is now line {}). Re-read before relying on old \
                 line numbers there.",
                end_line + 1,
                i64::try_from(end_line + 1).unwrap_or(i64::MAX) + delta
            );
        }
        let new_anchor_end = i64::try_from(end_line).unwrap_or(i64::MAX) + delta;
        let new_anchor_end = usize::try_from(new_anchor_end.max(0))
            .unwrap_or(0)
            .max(start_line);
        edit_result_append_context(&mut out, path, new_data, start_line, new_anchor_end);
    }
    out
}

/// Implements the `edit` tool: unique old/new text replacement.
pub fn tool_edit(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let path = call.arg_value("path").unwrap_or("");
    if path.is_empty() {
        return "Tool error: edit requires path\n".to_string();
    }
    let old = call.arg_value("old").unwrap_or("");
    if old.is_empty() {
        return "Tool error: edit requires non-empty old text\n".to_string();
    }
    let Some(new_text) = call.arg_value("new") else {
        return "Tool error: edit requires new text\n".to_string();
    };

    let full_path = ctx.resolve(path);
    let data = match read_file_bytes(&full_path, path) {
        Ok(d) => d,
        Err(err) => return format!("Tool error: {err}\n"),
    };
    let (offset, remove_len, anchored) = match edit_find_old_span(&data, old.as_bytes()) {
        Ok(span) => span,
        Err(err) => return format!("Tool error: {err}\n"),
    };

    let mut out_data = Vec::with_capacity(data.len() + new_text.len());
    out_data.extend_from_slice(&data[..offset]);
    out_data.extend_from_slice(new_text.as_bytes());
    out_data.extend_from_slice(&data[offset + remove_len..]);
    if let Err(e) = std::fs::write(&full_path, &out_data) {
        return format!("Tool error: write {path}: {e}\n");
    }

    // Record a diff preview for the UI change card.
    ctx.edit_previews.push(crate::tools::diff::edit_preview(
        path,
        &String::from_utf8_lossy(&data),
        &String::from_utf8_lossy(&out_data),
        false,
    ));

    let effect = old_new_line_effect(&data, &out_data, offset, remove_len);
    let kind = if anchored {
        "anchored old/new replacement"
    } else {
        "old/new replacement"
    };
    edit_result(path, effect, &out_data, kind)
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Minimal POSIX-ERE-style matcher standing in for the C `regex_t` usage.
///
/// Supports literals, `.`, `[...]` classes with ranges and negation, the
/// `* + ?` repeats, alternation `|`, groups `(...)`, anchors `^ $`, and
/// backslash escapes.
#[derive(Debug)]
struct MiniRegex {
    alt: Alt,
    case_sensitive: bool,
}

type Alt = Vec<Vec<Piece>>;

#[derive(Debug, Clone)]
struct Piece {
    atom: Atom,
    rep: Rep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rep {
    One,
    Star,
    Plus,
    Quest,
}

#[derive(Debug, Clone)]
enum Atom {
    Char(u8),
    Any,
    Class { neg: bool, ranges: Vec<(u8, u8)> },
    Group(Alt),
    Start,
    End,
}

impl MiniRegex {
    fn compile(pattern: &str, case_sensitive: bool) -> Result<Self, String> {
        let bytes = pattern.as_bytes();
        let mut pos = 0;
        let alt = parse_alt(bytes, &mut pos)?;
        if pos != bytes.len() {
            return Err("unmatched )".to_string());
        }
        Ok(Self {
            alt,
            case_sensitive,
        })
    }

    fn is_match(&self, line: &[u8]) -> bool {
        (0..=line.len()).any(|start| self.match_alt(&self.alt, line, start).is_some())
    }

    fn match_alt(&self, alt: &Alt, line: &[u8], pos: usize) -> Option<usize> {
        alt.iter().find_map(|seq| self.match_seq(seq, line, pos))
    }

    fn match_seq(&self, seq: &[Piece], line: &[u8], pos: usize) -> Option<usize> {
        let Some((piece, rest)) = seq.split_first() else {
            return Some(pos);
        };
        match piece.rep {
            Rep::One => {
                let next = self.match_atom(&piece.atom, line, pos)?;
                self.match_seq(rest, line, next)
            }
            Rep::Quest => self
                .match_atom(&piece.atom, line, pos)
                .and_then(|next| self.match_seq(rest, line, next))
                .or_else(|| self.match_seq(rest, line, pos)),
            Rep::Star | Rep::Plus => {
                let mut ends = vec![pos];
                let mut cur = pos;
                while let Some(next) = self.match_atom(&piece.atom, line, cur) {
                    if next == cur {
                        break;
                    }
                    ends.push(next);
                    cur = next;
                }
                let min_reps = usize::from(piece.rep == Rep::Plus);
                ends.iter()
                    .enumerate()
                    .rev()
                    .filter(|(reps, _)| *reps >= min_reps)
                    .find_map(|(_, &end)| self.match_seq(rest, line, end))
            }
        }
    }

    fn match_atom(&self, atom: &Atom, line: &[u8], pos: usize) -> Option<usize> {
        match atom {
            Atom::Start => (pos == 0).then_some(pos),
            Atom::End => (pos == line.len()).then_some(pos),
            Atom::Any => (pos < line.len()).then_some(pos + 1),
            Atom::Char(c) => {
                let b = *line.get(pos)?;
                let (a, b) = if self.case_sensitive {
                    (*c, b)
                } else {
                    (c.to_ascii_lowercase(), b.to_ascii_lowercase())
                };
                (a == b).then_some(pos + 1)
            }
            Atom::Class { neg, ranges } => {
                let b = *line.get(pos)?;
                let hit = ranges.iter().any(|&(lo, hi)| {
                    if self.case_sensitive {
                        b >= lo && b <= hi
                    } else {
                        let l = b.to_ascii_lowercase();
                        let u = b.to_ascii_uppercase();
                        (l >= lo && l <= hi) || (u >= lo && u <= hi)
                    }
                });
                (hit != *neg).then_some(pos + 1)
            }
            Atom::Group(alt) => self.match_alt(alt, line, pos),
        }
    }
}

fn parse_alt(p: &[u8], pos: &mut usize) -> Result<Alt, String> {
    let mut alt = vec![parse_seq(p, pos)?];
    while p.get(*pos) == Some(&b'|') {
        *pos += 1;
        alt.push(parse_seq(p, pos)?);
    }
    Ok(alt)
}

fn parse_seq(p: &[u8], pos: &mut usize) -> Result<Vec<Piece>, String> {
    let mut seq = Vec::new();
    while let Some(&c) = p.get(*pos) {
        if c == b'|' || c == b')' {
            break;
        }
        *pos += 1;
        let atom = match c {
            b'.' => Atom::Any,
            b'^' => Atom::Start,
            b'$' => Atom::End,
            b'(' => {
                let inner = parse_alt(p, pos)?;
                if p.get(*pos) != Some(&b')') {
                    return Err("missing )".to_string());
                }
                *pos += 1;
                Atom::Group(inner)
            }
            b'[' => parse_class(p, pos)?,
            b'\\' => {
                let &esc = p
                    .get(*pos)
                    .ok_or_else(|| "trailing backslash".to_string())?;
                *pos += 1;
                Atom::Char(esc)
            }
            b'*' | b'+' | b'?' => return Err("repetition with nothing to repeat".to_string()),
            other => Atom::Char(other),
        };
        let rep = match p.get(*pos) {
            Some(b'*') => {
                *pos += 1;
                Rep::Star
            }
            Some(b'+') => {
                *pos += 1;
                Rep::Plus
            }
            Some(b'?') => {
                *pos += 1;
                Rep::Quest
            }
            _ => Rep::One,
        };
        seq.push(Piece { atom, rep });
    }
    Ok(seq)
}

fn parse_class(p: &[u8], pos: &mut usize) -> Result<Atom, String> {
    let mut neg = false;
    if p.get(*pos) == Some(&b'^') {
        neg = true;
        *pos += 1;
    }
    let mut ranges = Vec::new();
    let mut first = true;
    loop {
        let &c = p.get(*pos).ok_or_else(|| "missing ]".to_string())?;
        if c == b']' && !first {
            *pos += 1;
            break;
        }
        first = false;
        *pos += 1;
        if p.get(*pos) == Some(&b'-') && p.get(*pos + 1).is_some_and(|&n| n != b']') {
            let hi = p[*pos + 1];
            *pos += 2;
            ranges.push((c.min(hi), c.max(hi)));
        } else {
            ranges.push((c, c));
        }
    }
    Ok(Atom::Class { neg, ranges })
}

/// Matches a shell-style glob (`*`, `?`, `[set]`) against a name.
fn glob_match(pattern: &[u8], name: &[u8]) -> bool {
    match pattern.split_first() {
        None => name.is_empty(),
        Some((b'*', rest)) => (0..=name.len()).any(|skip| glob_match(rest, &name[skip..])),
        Some((b'?', rest)) => name
            .split_first()
            .is_some_and(|(_, tail)| glob_match(rest, tail)),
        Some((b'[', rest)) => {
            let Some((&c, tail)) = name.split_first() else {
                return false;
            };
            let mut i = 0;
            let neg = matches!(rest.first(), Some(&b'!' | &b'^'));
            if neg {
                i += 1;
            }
            let mut hit = false;
            let mut first = true;
            while i < rest.len() && (rest[i] != b']' || first) {
                first = false;
                if i + 2 < rest.len() && rest[i + 1] == b'-' && rest[i + 2] != b']' {
                    if c >= rest[i] && c <= rest[i + 2] {
                        hit = true;
                    }
                    i += 3;
                } else {
                    if rest[i] == c {
                        hit = true;
                    }
                    i += 1;
                }
            }
            if i >= rest.len() {
                return false; // unterminated set never matches
            }
            hit != neg && glob_match(&rest[i + 1..], tail)
        }
        Some((&c, rest)) => name
            .split_first()
            .is_some_and(|(&n, tail)| n == c && glob_match(rest, tail)),
    }
}

#[derive(Debug)]
struct SearchCtx {
    query: String,
    glob: Option<String>,
    regex: Option<MiniRegex>,
    case_sensitive: bool,
    context: usize,
    max_results: usize,
    results: usize,
    out: String,
}

fn literal_match(line: &[u8], query: &[u8], case_sensitive: bool) -> bool {
    if query.is_empty() {
        return true;
    }
    if query.len() > line.len() {
        return false;
    }
    line.windows(query.len()).any(|w| {
        w.iter().zip(query).all(|(&a, &b)| {
            if case_sensitive {
                a == b
            } else {
                a.eq_ignore_ascii_case(&b)
            }
        })
    })
}

impl SearchCtx {
    fn line_matches(&self, line: &[u8]) -> bool {
        if let Some(re) = &self.regex {
            re.is_match(line)
        } else {
            literal_match(line, self.query.as_bytes(), self.case_sensitive)
        }
    }

    fn emit_line(&mut self, data: &[u8], sp: LineSpan, line_no: usize) {
        let _ = write!(self.out, "  {line_no} ");
        self.out
            .push_str(&String::from_utf8_lossy(&data[sp.start..sp.content_end]));
        self.out.push('\n');
    }

    /// Searches one text file and emits matching lines with line numbers.
    fn search_file(&mut self, path: &Path, display: &str) {
        if self.results >= self.max_results {
            return;
        }
        if let Some(glob) = &self.glob
            && !glob.is_empty()
        {
            let base = display.rsplit('/').next().unwrap_or(display);
            if !glob_match(glob.as_bytes(), base.as_bytes())
                && !glob_match(glob.as_bytes(), display.as_bytes())
            {
                return;
            }
        }
        let Ok(data) = read_file_bytes(path, display) else {
            return;
        };
        if data.contains(&0) {
            return;
        }
        let spans = split_lines(&data);
        let mut printed_file = false;
        let mut last_context_line: Option<usize> = None;
        for i in 0..spans.len() {
            if self.results >= self.max_results {
                break;
            }
            let sp = spans[i];
            if !self.line_matches(&data[sp.start..sp.content_end]) {
                continue;
            }
            if !printed_file {
                self.out.push_str(display);
                self.out.push('\n');
                printed_file = true;
            }
            let mut from = i.saturating_sub(self.context);
            let to = (i + self.context).min(spans.len() - 1);
            if let Some(last) = last_context_line
                && from <= last
            {
                from = last + 1;
            }
            for (j, &line_sp) in spans.iter().enumerate().take(to + 1).skip(from) {
                self.emit_line(&data, line_sp, j + 1);
                last_context_line = Some(j);
            }
            self.results += 1;
        }
        if printed_file {
            self.out.push('\n');
        }
        let _ = spans;
    }

    /// Recursively searches a path, skipping `.git` and honoring the cap.
    fn search_path(&mut self, path: &Path, display: &str, depth: usize) {
        if self.results >= self.max_results || depth > 24 {
            return;
        }
        let Ok(meta) = std::fs::symlink_metadata(path) else {
            return;
        };
        if meta.file_type().is_file() {
            self.search_file(path, display);
            return;
        }
        if !meta.file_type().is_dir() {
            return;
        }
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            if self.results >= self.max_results {
                break;
            }
            let name = entry.file_name();
            if name == ".git" {
                continue;
            }
            let child_display = format!("{display}/{}", name.to_string_lossy());
            self.search_path(&entry.path(), &child_display, depth + 1);
        }
    }
}

/// Implements the `search` tool with literal or regex line matching.
pub fn tool_search(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let query = call.arg_value("query").unwrap_or("");
    if query.is_empty() {
        return "Tool error: search requires query\n".to_string();
    }
    let path = match call.arg_value("path") {
        Some(p) if !p.is_empty() => p,
        _ => ".",
    };
    let use_regex = call.arg_value("mode") == Some("regex");
    let case_sensitive = parse_bool_default(call.arg_value("case_sensitive"), true);
    let mut sctx = SearchCtx {
        query: query.to_string(),
        glob: call.arg_value("glob").map(str::to_string),
        regex: None,
        case_sensitive,
        context: usize::try_from(parse_int_default(call.arg_value("context"), 0, 0, 5))
            .unwrap_or(0),
        max_results: usize::try_from(parse_int_default(call.arg_value("max_results"), 50, 1, 500))
            .unwrap_or(50),
        results: 0,
        out: String::new(),
    };
    if use_regex {
        match MiniRegex::compile(query, case_sensitive) {
            Ok(re) => sctx.regex = Some(re),
            Err(msg) => return format!("Tool error: invalid regex: {msg}\n"),
        }
    }
    sctx.search_path(&ctx.resolve(path), path, 0);
    if sctx.out.is_empty() {
        return "No matches\n".to_string();
    }
    let header = format!(
        "{} match{} shown\n\n",
        sctx.results,
        if sctx.results == 1 { "" } else { "es" }
    );
    format!("{header}{}", sctx.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{test_call, test_ctx};

    #[test]
    fn edit_replaces_unique_old_text() {
        let (mut ctx, dir) = test_ctx();
        std::fs::write(dir.join("f.txt"), "one\ntwo\nthree\n").unwrap();
        let out = tool_edit(
            &mut ctx,
            &test_call(
                "edit",
                &[("path", "f.txt"), ("old", "two\n"), ("new", "2a\n2b\n")],
            ),
        );
        assert!(out.starts_with("Edited f.txt using old/new replacement\n"));
        assert!(out.contains("Touched old lines 2-2; current post-edit context follows.\n"));
        assert!(
            out.contains("Line shift: old lines after 2 moved by +1 (old line 3 is now line 4).")
        );
        assert!(out.contains("Current file around edit: f.txt lines 1-4 of 4\n"));
        assert!(out.contains("1 one\n2 2a\n3 2b\n4 three\n"));
        assert_eq!(
            std::fs::read_to_string(dir.join("f.txt")).unwrap(),
            "one\n2a\n2b\nthree\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn edit_uniqueness_and_argument_errors() {
        let (mut ctx, dir) = test_ctx();
        std::fs::write(dir.join("f.txt"), "dup\nx\ndup\n").unwrap();
        let out = tool_edit(
            &mut ctx,
            &test_call("edit", &[("path", "f.txt"), ("old", "dup"), ("new", "y")]),
        );
        assert_eq!(out, "Tool error: old text anchor is not unique\n");
        let out = tool_edit(
            &mut ctx,
            &test_call(
                "edit",
                &[("path", "f.txt"), ("old", "missing"), ("new", "y")],
            ),
        );
        assert_eq!(out, "Tool error: old text anchor not found\n");
        assert_eq!(
            tool_edit(&mut ctx, &test_call("edit", &[("old", "a"), ("new", "b")])),
            "Tool error: edit requires path\n"
        );
        assert_eq!(
            tool_edit(
                &mut ctx,
                &test_call("edit", &[("path", "f.txt"), ("new", "b")])
            ),
            "Tool error: edit requires non-empty old text\n"
        );
        assert_eq!(
            tool_edit(
                &mut ctx,
                &test_call("edit", &[("path", "f.txt"), ("old", "x")])
            ),
            "Tool error: edit requires new text\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    // Ports of the C DS4_AGENT_TEST cases for [upto] spans.
    #[test]
    fn upto_tail_newline_is_not_part_of_anchor() {
        let data = b"CFLAGS = -Wall -Wextra -g\nLDFLAGS =\n\nall: bc\n\nbc: main.c\n\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n\nclean:\n\trm -f bc\n";
        let old = b"CFLAGS = -Wall -Wextra -g\nLDFLAGS =\n\nall: bc\n\nbc: main.c\n\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n\n[upto]\nclean:\n";
        let (offset, len, anchored) = edit_find_old_span(data, old).unwrap();
        assert!(anchored);
        assert_eq!(offset, 0);
        assert_eq!(len, data.len() - b"\trm -f bc\n".len());
    }

    #[test]
    fn upto_requires_tail_after_newline_strip() {
        let err = edit_find_old_span(b"head\nbody\ntail\n", b"head\n[upto]\n").unwrap_err();
        assert!(err.contains("must include a unique tail anchor"));
    }

    #[test]
    fn preflight_matches_edit_semantics() {
        let (ctx, dir) = test_ctx();
        std::fs::write(dir.join("f.txt"), "alpha\nbeta\n").unwrap();
        let call = test_call("edit", &[("path", "f.txt"), ("old", "beta")]);
        assert!(preflight_edit_old(&ctx, &call).is_ok());
        let call = test_call("edit", &[("path", "f.txt"), ("old", "gamma")]);
        assert_eq!(
            preflight_edit_old(&ctx, &call).unwrap_err(),
            "old text anchor not found"
        );
        let call = test_call("edit", &[("path", "f.txt")]);
        assert_eq!(
            preflight_edit_old(&ctx, &call).unwrap_err(),
            "edit requires non-empty old text"
        );
        drop(ctx);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn search_literal_hits_with_header() {
        let (mut ctx, dir) = test_ctx();
        std::fs::write(dir.join("a.txt"), "hello\nworld\nhello again\n").unwrap();
        let out = tool_search(&mut ctx, &test_call("search", &[("query", "hello")]));
        assert!(out.starts_with("2 matches shown\n\n./a.txt\n"));
        assert!(out.contains("  1 hello\n"));
        assert!(out.contains("  3 hello again\n"));
        let out = tool_search(&mut ctx, &test_call("search", &[("query", "absent")]));
        assert_eq!(out, "No matches\n");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn search_regex_and_glob() {
        let (mut ctx, dir) = test_ctx();
        std::fs::write(dir.join("a.rs"), "fn main() {}\nlet x = 12;\n").unwrap();
        std::fs::write(dir.join("b.txt"), "fn main() {}\n").unwrap();
        let out = tool_search(
            &mut ctx,
            &test_call(
                "search",
                &[
                    ("query", "^fn [a-z]+\\("),
                    ("mode", "regex"),
                    ("glob", "*.rs"),
                ],
            ),
        );
        assert!(out.starts_with("1 match shown\n\n"));
        assert!(out.contains("./a.rs\n  1 fn main() {}\n"));
        assert!(!out.contains("b.txt"));
        let out = tool_search(
            &mut ctx,
            &test_call("search", &[("query", "["), ("mode", "regex")]),
        );
        assert!(out.starts_with("Tool error: invalid regex: "));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn mini_regex_features() {
        let re = MiniRegex::compile("a(bc|de)+f?$", true).unwrap();
        assert!(re.is_match(b"xxabcdef"));
        assert!(re.is_match(b"abc"));
        assert!(!re.is_match(b"af"));
        let re = MiniRegex::compile("[^0-9]+", false).unwrap();
        assert!(re.is_match(b"abc"));
        assert!(!re.is_match(b"123"));
        let re = MiniRegex::compile("HeLLo", false).unwrap();
        assert!(re.is_match(b"say hello there"));
    }

    #[test]
    fn glob_patterns() {
        assert!(glob_match(b"*.rs", b"main.rs"));
        assert!(!glob_match(b"*.rs", b"main.rc"));
        assert!(glob_match(b"a?c", b"abc"));
        assert!(glob_match(b"[a-c]x", b"bx"));
        assert!(!glob_match(b"[!a-c]x", b"bx"));
    }
}

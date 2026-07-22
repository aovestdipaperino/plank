// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! File tools: `read`, `more`, `write`, and `list`.
//!
//! Port of the file-tool half of "Tool Argument Parsing And File Tool
//! Helpers" in `ds4_agent.c`. Output formats and error texts replicate the C
//! agent exactly, since the model was trained on them.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::dsml::ToolCall;

use super::{MoreState, ToolContext, parse_bool_default, parse_int_default};

/// Maximum file size the file tools will load, in bytes.
pub(crate) const FILE_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Default number of lines returned by a windowed read.
pub(crate) const READ_DEFAULT_LINES: usize = 500;
/// Maximum entries shown by the `list` tool.
const LIST_MAX_ENTRIES: usize = 300;

/// One line of a text buffer; `content_end` excludes the CR/LF terminator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LineSpan {
    /// Byte offset of the first character of the line.
    pub start: usize,
    /// Byte offset just past the line content, excluding CR/LF.
    pub content_end: usize,
    /// Byte offset just past the line terminator.
    pub end: usize,
}

/// Splits a buffer into line spans, handling LF, CRLF, and lone CR.
pub(crate) fn split_lines(data: &[u8]) -> Vec<LineSpan> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        while pos < data.len() && data[pos] != b'\n' && data[pos] != b'\r' {
            pos += 1;
        }
        let content_end = pos;
        if pos < data.len() {
            if data[pos] == b'\r' && pos + 1 < data.len() && data[pos + 1] == b'\n' {
                pos += 2;
            } else {
                pos += 1;
            }
        }
        spans.push(LineSpan {
            start,
            content_end,
            end: pos,
        });
    }
    spans
}

/// Reads a whole file with the tool size cap; errors match the C texts.
///
/// `display` is the path spelling used in error messages.
///
/// # Errors
///
/// Returns the C-format message (`open ...`, `read ...`, `file too large:
/// ...`) on failure.
pub(crate) fn read_file_bytes(path: &Path, display: &str) -> Result<Vec<u8>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("open {display}: {e}"))?;
    let mut buf = Vec::new();
    let mut tmp = [0_u8; 8192];
    loop {
        match file.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                if buf.len() + n > FILE_MAX_BYTES {
                    return Err(format!(
                        "file too large: {display} exceeds {FILE_MAX_BYTES} bytes"
                    ));
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("read {display}: {e}")),
        }
    }
    Ok(buf)
}

fn push_lossy(out: &mut String, bytes: &[u8]) {
    out.push_str(&String::from_utf8_lossy(bytes));
}

/// Reads a line window from a file for the model, mirroring `agent_read_range`.
///
/// Normal mode decorates lines with plain line numbers; `bare` mode returns
/// raw content for payloads that decoration would corrupt. Truncated reads
/// record continuation state for the `more` tool when `set_more` is true.
pub(crate) fn read_range(
    ctx: &mut ToolContext,
    path: &str,
    start_line: usize,
    max_lines: usize,
    whole_file: bool,
    bare: bool,
    set_more: bool,
) -> String {
    if path.is_empty() {
        return "Tool error: read requires path\n".to_string();
    }
    let data = match read_file_bytes(&ctx.resolve(path), path) {
        Ok(d) => d,
        Err(err) => return format!("Tool error: {err}\n"),
    };

    let spans = split_lines(&data);
    let start_line = start_line.max(1);
    let start_idx = (start_line - 1).min(spans.len());
    let max_lines = if whole_file {
        spans.len() - start_idx
    } else if max_lines == 0 {
        READ_DEFAULT_LINES
    } else {
        max_lines
    };
    let end_idx = (start_idx + max_lines).min(spans.len());

    let mut out = String::new();
    if bare {
        let start = spans.get(start_idx).map_or(data.len(), |sp| sp.start);
        let end = if end_idx > start_idx {
            spans[end_idx - 1].end
        } else {
            start
        };
        push_lossy(&mut out, &data[start..end]);
        if end > start && !out.ends_with('\n') {
            out.push('\n');
        }
        if end_idx < spans.len() {
            let _ = writeln!(
                out,
                "[Read truncated at line {} of {}. continue_offset={}. \
                 Call more with count={} to read the next chunk.]",
                end_idx,
                spans.len(),
                end_idx + 1,
                max_lines
            );
        }
    } else {
        let shown_start = if spans.is_empty() { 0 } else { start_idx + 1 };
        if end_idx < spans.len() {
            let _ = writeln!(
                out,
                "{path}: lines {shown_start}-{end_idx} of {}; continue_offset={}; \
                 call more with count={} to read the next chunk",
                spans.len(),
                end_idx + 1,
                max_lines
            );
        } else {
            let _ = writeln!(
                out,
                "{path}: lines {shown_start}-{end_idx} of {}",
                spans.len()
            );
        }
        for (i, sp) in spans.iter().enumerate().take(end_idx).skip(start_idx) {
            let _ = write!(out, "{} ", i + 1);
            push_lossy(&mut out, &data[sp.start..sp.content_end]);
            out.push('\n');
        }
    }
    if set_more {
        ctx.more = if end_idx < spans.len() {
            Some(MoreState {
                path: path.to_string(),
                next_line: end_idx + 1,
                bare,
            })
        } else {
            None
        };
    }
    out
}

/// Implements the `read` tool: a windowed, line-numbered file read.
pub fn tool_read(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let path = call.arg_value("path").unwrap_or("").to_string();
    let whole = parse_bool_default(call.arg_value("whole"), false);
    let start = parse_int_default(call.arg_value("start_line"), 1, 1, i64::MAX);
    let count = parse_int_default(
        call.arg_value("max_lines"),
        READ_DEFAULT_LINES.try_into().unwrap_or(i64::MAX),
        1,
        i64::MAX,
    );
    let raw = parse_bool_default(call.arg_value("raw"), false);
    let out = read_range(
        ctx,
        &path,
        usize::try_from(start).unwrap_or(usize::MAX),
        usize::try_from(count).unwrap_or(usize::MAX),
        whole,
        raw,
        true,
    );
    if !out.starts_with("Tool error:") {
        let resolved = ctx.resolve(&path);
        ctx.note_read(resolved);
    }
    out
}

/// Implements the `more` tool: continues the previous truncated read.
pub fn tool_more(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let count = parse_int_default(
        call.arg_value("count"),
        READ_DEFAULT_LINES.try_into().unwrap_or(i64::MAX),
        1,
        i64::MAX,
    );
    let Some(more) = ctx.more.clone() else {
        return "Tool error: no previous output to continue\n".to_string();
    };
    read_range(
        ctx,
        &more.path,
        more.next_line,
        usize::try_from(count).unwrap_or(usize::MAX),
        false,
        more.bare,
        true,
    )
}

/// Implements the `write` tool: replaces a file with the given content.
pub fn tool_write(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let path = call.arg_value("path").unwrap_or("");
    if path.is_empty() {
        return "Tool error: write requires path\n".to_string();
    }
    let Some(content) = call.arg_value("content") else {
        return "Tool error: write requires content\n".to_string();
    };
    let full = ctx.resolve(path);
    // Prior content (for the diff card); absent means this write creates it.
    let prior = std::fs::read(&full).ok();
    let mut file = match std::fs::File::create(&full) {
        Ok(f) => f,
        Err(e) => return format!("Tool error: open for write failed: {e}\n"),
    };
    if let Err(e) = std::io::Write::write_all(&mut file, content.as_bytes()) {
        return format!("Tool error: write failed: {e}\n");
    }
    // A new file is shown by the streaming dim preview; only an overwrite gets
    // a post-edit diff card here.
    if let Some(prior) = &prior {
        let old = String::from_utf8_lossy(prior);
        let mut preview = crate::tools::diff::edit_preview(path, &old, content, false);
        preview.bytes = Some(content.len());
        ctx.edit_previews.push(preview);
    }
    format!("Wrote {} bytes to {}\n", content.len(), path)
}

/// Implements the `list` tool: a capped directory listing.
pub fn tool_list(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let path = match call.arg_value("path") {
        Some(p) if !p.is_empty() => p,
        _ => ".",
    };
    let entries = match std::fs::read_dir(ctx.resolve(path)) {
        Ok(it) => it,
        Err(e) => return format!("Tool error: opendir failed: {e}\n"),
    };
    let mut out = format!("{path}:\n");
    let mut shown = 0;
    for entry in entries.flatten() {
        if shown >= LIST_MAX_ENTRIES {
            out.push_str("... more entries omitted ...\n");
            break;
        }
        let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        let ft = meta.file_type();
        let kind = if ft.is_dir() {
            'd'
        } else if ft.is_symlink() {
            'l'
        } else if ft.is_file() {
            '-'
        } else {
            '?'
        };
        let name = entry.file_name();
        let _ = writeln!(
            out,
            "{kind} {:>10} {}{}",
            meta.len(),
            name.to_string_lossy(),
            if ft.is_dir() { "/" } else { "" }
        );
        shown += 1;
    }
    out
}

/// Most matches `glob` returns before truncating.
const GLOB_MAX_RESULTS: usize = 100;
/// VCS metadata directories never descended into or reported.
const GLOB_SKIP_DIRS: &[&str] = &[".git", ".hg", ".svn", ".jj"];

/// Matches one path component against a glob segment (`*` and `?`; no `**`,
/// which is handled by the walker across components).
///
/// A hand-rolled backtracking matcher rather than a dependency, matching the
/// project's style. `*` spans any run within a single component; `?` is one
/// character; every other character is literal.
fn segment_matches(seg: &[char], name: &[char]) -> bool {
    // Iterative backtracking: `star`/`mark` remember where to resume the last
    // `*` if the current attempt dead-ends.
    let (mut si, mut ni) = (0usize, 0usize);
    let (mut star, mut mark): (Option<usize>, usize) = (None, 0);
    while ni < name.len() {
        if si < seg.len() && (seg[si] == '?' || seg[si] == name[ni]) {
            si += 1;
            ni += 1;
        } else if si < seg.len() && seg[si] == '*' {
            star = Some(si);
            mark = ni;
            si += 1;
        } else if let Some(s) = star {
            si = s + 1;
            mark += 1;
            ni = mark;
        } else {
            return false;
        }
    }
    while si < seg.len() && seg[si] == '*' {
        si += 1;
    }
    si == seg.len()
}

/// Matches a path against a glob pattern split on `/`.
///
/// `**` matches zero or more whole path components (crossing directory
/// boundaries); a plain `*` stays within one component. Both sides are
/// compared as `/`-split segments.
fn glob_matches(pattern_segs: &[String], path: &str) -> bool {
    let path_segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    glob_rec(pattern_segs, &path_segs)
}

/// Recursive core of [`glob_matches`], matching pattern segments to path
/// segments; `**` consumes zero or more path segments.
fn glob_rec(pat: &[String], path: &[&str]) -> bool {
    match pat.first().map(String::as_str) {
        None => path.is_empty(),
        Some("**") => (0..=path.len()).any(|skip| glob_rec(&pat[1..], &path[skip..])),
        Some(seg) => {
            let seg_chars: Vec<char> = seg.chars().collect();
            !path.is_empty()
                && segment_matches(&seg_chars, &path[0].chars().collect::<Vec<_>>())
                && glob_rec(&pat[1..], &path[1..])
        }
    }
}

/// Whether the pattern can only match inside a fixed leading directory, and
/// that directory, so the walk can start there instead of at the root.
///
/// `src/**/mod.rs` need not walk the whole tree; `**/x` and `*.rs` must.
fn glob_walk_root(pattern_segs: &[String]) -> PathBuf {
    let mut root = PathBuf::new();
    for seg in pattern_segs {
        if seg.contains(['*', '?']) {
            break;
        }
        root.push(seg);
    }
    root
}

/// Recursively collects file paths under `dir` matching `pattern_segs`,
/// relative to `search_root`, skipping VCS metadata. Depth-first, sorted by
/// the caller.
fn glob_walk(dir: &Path, search_root: &Path, pattern_segs: &[String], out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if ft.is_dir() {
            if GLOB_SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            glob_walk(&entry.path(), search_root, pattern_segs, out);
        } else if ft.is_file()
            && let Ok(rel) = entry.path().strip_prefix(search_root)
        {
            let rel = rel.to_string_lossy().replace('\\', "/");
            if glob_matches(pattern_segs, &rel) {
                out.push(rel);
            }
        }
    }
}

/// Finds files by name pattern across a tree (issue #32).
///
/// `**` crosses directory boundaries, `*` does not; results are relative to
/// the search root, sorted, and capped at [`GLOB_MAX_RESULTS`]. VCS metadata
/// directories are skipped, and a path outside the sandbox root is refused
/// like the other file tools.
pub fn tool_glob(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let pattern = match call.arg_value("pattern") {
        Some(p) if !p.is_empty() => p,
        _ => return "Tool error: glob requires pattern\n".to_string(),
    };
    let base = match call.arg_value("path") {
        Some(p) if !p.is_empty() => p,
        _ => ".",
    };
    let search_root = ctx.resolve(base);
    let Ok(canon_root) = search_root.canonicalize() else {
        return format!("Tool error: glob path not found: {base}\n");
    };
    // Same containment rule as the other file tools: the resolved root must
    // stay under the sandbox cwd.
    if let Ok(cwd) = ctx.cwd.canonicalize()
        && !canon_root.starts_with(&cwd)
    {
        return format!("Tool error: glob path escapes workspace: {base}\n");
    }

    let pattern_segs: Vec<String> = pattern
        .split('/')
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    let walk_start = canon_root.join(glob_walk_root(&pattern_segs));
    let mut matches = Vec::new();
    glob_walk(&walk_start, &canon_root, &pattern_segs, &mut matches);
    matches.sort();

    if matches.is_empty() {
        return format!("glob: no files match {pattern}\n");
    }
    let truncated = matches.len() > GLOB_MAX_RESULTS;
    matches.truncate(GLOB_MAX_RESULTS);
    let mut out = matches.join("\n");
    out.push('\n');
    if truncated {
        let _ = writeln!(
            out,
            "... more than {GLOB_MAX_RESULTS} matches; showing the first {GLOB_MAX_RESULTS} ..."
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{test_call, test_ctx};

    /// Creates the given files (and their parent dirs) under `root`.
    fn make_tree(root: &Path, files: &[&str]) {
        for f in files {
            let p = root.join(f);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&p, "x").unwrap();
        }
    }

    #[test]
    fn segment_matcher_star_and_question() {
        let m = |seg: &str, name: &str| {
            segment_matches(
                &seg.chars().collect::<Vec<_>>(),
                &name.chars().collect::<Vec<_>>(),
            )
        };
        assert!(m("*.rs", "main.rs"));
        assert!(m("*_test.rs", "foo_test.rs"));
        assert!(!m("*.rs", "main.toml"));
        assert!(m("a?c", "abc"));
        assert!(!m("a?c", "ac"));
        assert!(m("*", "anything"));
        assert!(m("mod.rs", "mod.rs"));
    }

    #[test]
    fn star_stays_within_a_component_but_doublestar_crosses() {
        let segs = |p: &str| p.split('/').map(ToString::to_string).collect::<Vec<_>>();
        assert!(glob_matches(&segs("*.rs"), "main.rs"));
        assert!(
            !glob_matches(&segs("*.rs"), "src/main.rs"),
            "* must not cross /"
        );
        assert!(glob_matches(&segs("**/*.rs"), "src/deep/main.rs"));
        assert!(
            glob_matches(&segs("**/*.rs"), "main.rs"),
            "** matches zero dirs"
        );
        assert!(glob_matches(&segs("src/**/mod.rs"), "src/a/b/mod.rs"));
        assert!(!glob_matches(&segs("src/**/mod.rs"), "other/a/mod.rs"));
    }

    #[test]
    fn glob_lists_only_the_directory_for_a_plain_star() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["a.rs", "b.rs", "sub/c.rs", "notes.txt"]);
        let out = tool_glob(&mut ctx, &test_call("glob", &[("pattern", "*.rs")]));
        assert_eq!(out, "a.rs\nb.rs\n", "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_doublestar_reaches_every_depth() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["a.rs", "sub/b.rs", "sub/deep/c.rs"]);
        let out = tool_glob(&mut ctx, &test_call("glob", &[("pattern", "**/*.rs")]));
        assert_eq!(out, "a.rs\nsub/b.rs\nsub/deep/c.rs\n", "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_never_descends_into_git() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["src/a.rs", ".git/config.rs", ".git/hooks/x.rs"]);
        let out = tool_glob(&mut ctx, &test_call("glob", &[("pattern", "**/*.rs")]));
        assert_eq!(out, "src/a.rs\n", "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_with_no_match_is_a_clear_message_not_an_error() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["a.rs"]);
        let out = tool_glob(&mut ctx, &test_call("glob", &[("pattern", "*.zzz")]));
        assert_eq!(out, "glob: no files match *.zzz\n");
        assert!(!out.starts_with("Tool error"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_truncates_past_the_cap_and_says_so() {
        let (mut ctx, dir) = test_ctx();
        let files: Vec<String> = (0..150).map(|i| format!("f{i:03}.rs")).collect();
        make_tree(&dir, &files.iter().map(String::as_str).collect::<Vec<_>>());
        let out = tool_glob(&mut ctx, &test_call("glob", &[("pattern", "*.rs")]));
        assert_eq!(out.lines().filter(|l| l.starts_with('f')).count(), 100);
        assert!(out.contains("more than 100 matches"), "{out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_missing_pattern_matches_the_error_convention() {
        let (mut ctx, dir) = test_ctx();
        let out = tool_glob(&mut ctx, &test_call("glob", &[]));
        assert_eq!(out, "Tool error: glob requires pattern\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_refuses_a_path_outside_the_workspace() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["a.rs"]);
        let out = tool_glob(
            &mut ctx,
            &test_call("glob", &[("pattern", "*.rs"), ("path", "..")]),
        );
        assert!(
            out.contains("escapes workspace") || out.contains("not found"),
            "{out}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn glob_path_scopes_the_search() {
        let (mut ctx, dir) = test_ctx();
        make_tree(&dir, &["top.rs", "sub/inner.rs", "sub/deep/x.rs"]);
        let out = tool_glob(
            &mut ctx,
            &test_call("glob", &[("pattern", "**/*.rs"), ("path", "sub")]),
        );
        assert_eq!(out, "deep/x.rs\ninner.rs\n", "{out:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn write_file(dir: &Path, name: &str, content: &str) {
        std::fs::write(dir.join(name), content).expect("write test file");
    }

    #[test]
    fn read_window_and_more_continuation() {
        let (mut ctx, dir) = test_ctx();
        let body = (1..=10).fold(String::new(), |mut s, i| {
            let _ = writeln!(s, "line{i}");
            s
        });
        write_file(&dir, "f.txt", &body);

        let out = tool_read(
            &mut ctx,
            &test_call(
                "read",
                &[("path", "f.txt"), ("start_line", "2"), ("max_lines", "3")],
            ),
        );
        assert_eq!(
            out,
            "f.txt: lines 2-4 of 10; continue_offset=5; \
             call more with count=3 to read the next chunk\n\
             2 line2\n3 line3\n4 line4\n"
        );

        let out = tool_more(&mut ctx, &test_call("more", &[("count", "6")]));
        assert!(out.starts_with("f.txt: lines 5-10 of 10\n"));
        assert!(out.ends_with("10 line10\n"));
        assert!(ctx.more.is_none());

        let out = tool_more(&mut ctx, &test_call("more", &[]));
        assert_eq!(out, "Tool error: no previous output to continue\n");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_whole_and_bare_truncation() {
        let (mut ctx, dir) = test_ctx();
        write_file(&dir, "f.txt", "a\nb\nc\n");
        let out = tool_read(
            &mut ctx,
            &test_call("read", &[("path", "f.txt"), ("whole", "true")]),
        );
        assert_eq!(out, "f.txt: lines 1-3 of 3\n1 a\n2 b\n3 c\n");

        let out = tool_read(
            &mut ctx,
            &test_call(
                "read",
                &[("path", "f.txt"), ("raw", "true"), ("max_lines", "2")],
            ),
        );
        assert_eq!(
            out,
            "a\nb\n[Read truncated at line 2 of 3. continue_offset=3. \
             Call more with count=2 to read the next chunk.]\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn read_missing_file_and_missing_path() {
        let (mut ctx, dir) = test_ctx();
        let out = tool_read(&mut ctx, &test_call("read", &[]));
        assert_eq!(out, "Tool error: read requires path\n");
        let out = tool_read(&mut ctx, &test_call("read", &[("path", "nope.txt")]));
        assert!(out.starts_with("Tool error: open nope.txt: "));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn write_then_list() {
        let (mut ctx, dir) = test_ctx();
        let out = tool_write(
            &mut ctx,
            &test_call("write", &[("path", "out.txt"), ("content", "hello")]),
        );
        assert_eq!(out, "Wrote 5 bytes to out.txt\n");
        std::fs::create_dir(dir.join("sub")).unwrap();

        let out = tool_list(&mut ctx, &test_call("list", &[]));
        assert!(out.starts_with(".:\n"));
        assert!(out.contains("-          5 out.txt\n"));
        assert!(out.contains(" sub/\n"));

        assert_eq!(
            tool_write(&mut ctx, &test_call("write", &[("content", "x")])),
            "Tool error: write requires path\n"
        );
        assert_eq!(
            tool_write(&mut ctx, &test_call("write", &[("path", "p")])),
            "Tool error: write requires content\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn write_card_only_for_overwrites_not_new_files() {
        let (mut ctx, dir) = test_ctx();
        // New file: no diff card (the streaming preview shows it).
        tool_write(
            &mut ctx,
            &test_call("write", &[("path", "c.txt"), ("content", "one\n")]),
        );
        assert!(ctx.edit_previews.is_empty(), "new file: no card");

        // Overwrite: a card diffing old vs new, with the byte size set.
        tool_write(
            &mut ctx,
            &test_call("write", &[("path", "c.txt"), ("content", "two\n")]),
        );
        assert_eq!(ctx.edit_previews.len(), 1, "overwrite: one card");
        let card = &ctx.edit_previews[0];
        assert!(!card.created);
        assert_eq!(card.added, 1);
        assert_eq!(card.removed, 1);
        assert_eq!(card.bytes, Some(4));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn split_lines_handles_crlf() {
        let spans = split_lines(b"a\r\nb\nc");
        assert_eq!(spans.len(), 3);
        assert_eq!(
            (spans[0].start, spans[0].content_end, spans[0].end),
            (0, 1, 3)
        );
        assert_eq!(
            (spans[2].start, spans[2].content_end, spans[2].end),
            (5, 6, 6)
        );
    }
}

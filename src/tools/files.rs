//! File tools: `read`, `more`, `write`, and `list`.
//!
//! Port of the file-tool half of "Tool Argument Parsing And File Tool
//! Helpers" in `ds4_agent.c`. Output formats and error texts replicate the C
//! agent exactly, since the model was trained on them.

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;

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
    read_range(
        ctx,
        &path,
        usize::try_from(start).unwrap_or(usize::MAX),
        usize::try_from(count).unwrap_or(usize::MAX),
        whole,
        raw,
        true,
    )
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
    let mut file = match std::fs::File::create(ctx.resolve(path)) {
        Ok(f) => f,
        Err(e) => return format!("Tool error: open for write failed: {e}\n"),
    };
    if let Err(e) = std::io::Write::write_all(&mut file, content.as_bytes()) {
        return format!("Tool error: write failed: {e}\n");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{test_call, test_ctx};

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

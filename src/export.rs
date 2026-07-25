// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Session export: renders a transcript to a shareable Markdown or
//! standalone HTML document (issue #66).
//!
//! The renderers are pure `transcript -> String` functions so they stay
//! unit-testable without an engine. HTML output is self-contained (inline
//! CSS, no external assets) and every byte of model/tool content is
//! HTML-escaped — transcripts routinely carry arbitrary code.

use std::fmt::Write as _;

use crate::dsml::{DsmlParser, DsmlState, ToolCall};
use crate::session::{Message, Role};

/// Opening marker of a DSML tool-call stanza.
const DSML_OPEN: &str = "<｜DSML｜tool_calls>";
/// Closing marker of a DSML tool-call stanza.
const DSML_CLOSE: &str = "</｜DSML｜tool_calls｜>";
/// Opening tag of a thinking block.
const THINK_OPEN: &str = "<think>";
/// Closing tag of a thinking block.
const THINK_CLOSE: &str = "</think>";

/// Output format for [`render`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// GitHub-flavored Markdown.
    Markdown,
    /// Standalone HTML with inline CSS.
    Html,
}

impl Format {
    /// File extension (no dot) for this format.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Format::Markdown => "md",
            Format::Html => "html",
        }
    }
}

/// Parses a `/export` format word; `md`/`markdown` and `html`/`htm`.
#[must_use]
pub fn parse_format(s: &str) -> Option<Format> {
    match s.trim().to_ascii_lowercase().as_str() {
        "md" | "markdown" => Some(Format::Markdown),
        "html" | "htm" => Some(Format::Html),
        _ => None,
    }
}

/// One renderable piece of a transcript message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// A human turn.
    User(String),
    /// Visible assistant prose.
    Assistant(String),
    /// Assistant thinking text (contents of a `<think>` block).
    Thinking(String),
    /// A parsed tool invocation.
    ToolCall(ToolCall),
    /// A tool result fed back as a pseudo-user turn.
    ToolResult(String),
}

/// Splits a whole transcript into renderable segments.
#[must_use]
pub fn segments(transcript: &[Message]) -> Vec<Segment> {
    let mut out = Vec::new();
    for m in transcript {
        match m.role {
            Role::User if m.is_tool_user() => {
                let payload = m.tool_result_payload().trim();
                if !payload.is_empty() {
                    out.push(Segment::ToolResult(payload.to_owned()));
                }
            }
            Role::User => {
                let text = m.text.trim();
                if !text.is_empty() {
                    out.push(Segment::User(text.to_owned()));
                }
            }
            Role::Assistant => split_assistant(&m.text, &mut out),
        }
    }
    out
}

/// Splits one assistant message into thinking / tool-call / prose segments.
fn split_assistant(text: &str, out: &mut Vec<Segment>) {
    let mut rest = text;
    while !rest.is_empty() {
        let think = rest.find(THINK_OPEN);
        let dsml = rest.find(DSML_OPEN);
        let Some(at) = min_opt(think, dsml) else {
            push_prose(rest, out);
            return;
        };
        push_prose(&rest[..at], out);
        if think == Some(at) {
            let after = &rest[at + THINK_OPEN.len()..];
            let (body, tail) = match after.find(THINK_CLOSE) {
                Some(end) => (&after[..end], &after[end + THINK_CLOSE.len()..]),
                None => (after, ""),
            };
            let body = body.trim();
            if !body.is_empty() {
                out.push(Segment::Thinking(body.to_owned()));
            }
            rest = tail;
        } else {
            let after = &rest[at..];
            let (stanza, tail) = match after.find(DSML_CLOSE) {
                Some(end) => after.split_at(end + DSML_CLOSE.len()),
                None => (after, ""),
            };
            let mut parser = DsmlParser::new();
            parser.feed(stanza);
            if parser.state() == DsmlState::Done && !parser.calls().is_empty() {
                for call in parser.calls() {
                    out.push(Segment::ToolCall(call.clone()));
                }
            } else {
                push_prose(stanza, out);
            }
            rest = tail;
        }
    }
}

/// Returns the smaller of two optional indices.
fn min_opt(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Pushes non-empty prose as an assistant segment.
fn push_prose(text: &str, out: &mut Vec<Segment>) {
    let text = text.trim();
    if !text.is_empty() {
        out.push(Segment::Assistant(text.to_owned()));
    }
}

/// Renders a transcript in `format` with the given document title.
#[must_use]
pub fn render(transcript: &[Message], title: &str, format: Format) -> String {
    let segs = segments(transcript);
    match format {
        Format::Markdown => render_markdown(&segs, title),
        Format::Html => render_html(&segs, title),
    }
}

/// Renders segments as Markdown.
#[must_use]
pub fn render_markdown(segs: &[Segment], title: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}\n", title_or_default(title));
    for seg in segs {
        match seg {
            Segment::User(t) => {
                out.push_str("## User\n\n");
                push_quoted(&mut out, t);
            }
            Segment::Assistant(t) => {
                out.push_str("## Assistant\n\n");
                out.push_str(t);
                out.push_str("\n\n");
            }
            Segment::Thinking(t) => {
                out.push_str("### Thinking\n\n");
                push_quoted(&mut out, t);
            }
            Segment::ToolCall(call) => {
                let _ = writeln!(out, "### Tool call: `{}`\n", call.name);
                for arg in &call.args {
                    let _ = writeln!(out, "- **{}**:\n", arg.name);
                    push_fenced(&mut out, &arg.value);
                }
            }
            Segment::ToolResult(t) => {
                out.push_str("### Tool result\n\n");
                push_fenced(&mut out, t);
            }
        }
    }
    out
}

/// Appends `text` as a Markdown blockquote.
fn push_quoted(out: &mut String, text: &str) {
    for line in text.lines() {
        if line.is_empty() {
            out.push_str(">\n");
        } else {
            let _ = writeln!(out, "> {line}");
        }
    }
    out.push('\n');
}

/// Appends `text` inside a fence long enough to survive nested backticks.
fn push_fenced(out: &mut String, text: &str) {
    let fence = "`".repeat(longest_backtick_run(text).max(2) + 1);
    let _ = writeln!(out, "{fence}");
    out.push_str(text);
    if !text.ends_with('\n') {
        out.push('\n');
    }
    let _ = writeln!(out, "{fence}\n");
}

/// Longest run of consecutive backticks in `text`.
fn longest_backtick_run(text: &str) -> usize {
    let mut best = 0;
    let mut run = 0;
    for c in text.chars() {
        if c == '`' {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

/// Escapes `text` for safe interpolation into HTML text or attributes.
#[must_use]
pub fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Inline stylesheet for the standalone HTML export.
const CSS: &str = "\
:root{color-scheme:light dark}
body{margin:0;padding:2rem 1rem;font:16px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:#fbfbfa;color:#1c1c1e}
main{max-width:52rem;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 1.5rem}
.msg{border-radius:8px;padding:.75rem 1rem;margin:0 0 1rem;border:1px solid #e2e2df;background:#fff}
.msg>.who{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:#6b6b6b;margin:0 0 .4rem}
.user{background:#eef4ff;border-color:#cddcf7}
.thinking{background:#f6f2fb;border-color:#e0d6ef;color:#4a4458}
.toolcall{background:#f3faf4;border-color:#cfe7d4}
.toolresult{background:#f7f7f6}
pre{margin:0;white-space:pre-wrap;word-wrap:break-word;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.arg{margin-top:.5rem}
.arg>.name{font-size:.75rem;color:#6b6b6b}
@media (prefers-color-scheme:dark){
body{background:#131315;color:#e6e6e6}
.msg{background:#1c1c1f;border-color:#2e2e33}
.msg>.who{color:#9a9aa0}
.user{background:#182130;border-color:#26344a}
.thinking{background:#1e1a26;border-color:#332b40;color:#c4bcd4}
.toolcall{background:#16211a;border-color:#28382c}
.toolresult{background:#1a1a1c}
.arg>.name{color:#9a9aa0}
}";

/// Renders segments as a standalone HTML document.
#[must_use]
pub fn render_html(segs: &[Segment], title: &str) -> String {
    let title = html_escape(title_or_default(title));
    let mut out = String::new();
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    let _ = writeln!(out, "<title>{title}</title>");
    let _ = writeln!(out, "<style>{CSS}</style>");
    out.push_str("</head>\n<body>\n<main>\n");
    let _ = writeln!(out, "<h1>{title}</h1>");
    for seg in segs {
        match seg {
            Segment::User(t) => push_block(&mut out, "user", "User", t),
            Segment::Assistant(t) => push_block(&mut out, "assistant", "Assistant", t),
            Segment::Thinking(t) => push_block(&mut out, "thinking", "Thinking", t),
            Segment::ToolResult(t) => push_block(&mut out, "toolresult", "Tool result", t),
            Segment::ToolCall(call) => {
                let _ = writeln!(
                    out,
                    "<section class=\"msg toolcall\">\n<p class=\"who\">Tool call: {}</p>",
                    html_escape(&call.name)
                );
                for arg in &call.args {
                    let _ = writeln!(
                        out,
                        "<div class=\"arg\"><p class=\"name\">{}</p><pre>{}</pre></div>",
                        html_escape(&arg.name),
                        html_escape(&arg.value)
                    );
                }
                out.push_str("</section>\n");
            }
        }
    }
    out.push_str("</main>\n</body>\n</html>\n");
    out
}

/// Emits one labelled HTML block with escaped body text.
fn push_block(out: &mut String, class: &str, who: &str, text: &str) {
    let _ = writeln!(
        out,
        "<section class=\"msg {class}\">\n<p class=\"who\">{who}</p>\n<pre>{}</pre>\n</section>",
        html_escape(text)
    );
}

/// Falls back to a generic document title when the session has none.
fn title_or_default(title: &str) -> &str {
    let t = title.trim();
    if t.is_empty() { "plank session" } else { t }
}

/// Builds an auto file name like `plank-session-<slug>-<when>.md`.
#[must_use]
pub fn default_filename(title: &str, when: u64, format: Format) -> String {
    let slug = slugify(title);
    let ext = format.extension();
    if slug.is_empty() {
        format!("plank-session-{when}.{ext}")
    } else {
        format!("plank-session-{slug}-{when}.{ext}")
    }
}

/// Lowercases and hyphenates `title` into a short, filesystem-safe slug.
fn slugify(title: &str) -> String {
    let mut out = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
        if out.len() >= 40 {
            break;
        }
    }
    out.trim_matches('-').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Vec<Message> {
        vec![
            Message::user("fix the parser"),
            Message::assistant(concat!(
                "<think>needs a look</think>",
                "Reading it now.",
                "<｜DSML｜tool_calls><｜DSML｜invoke name=\"read_file\">",
                "<｜DSML｜parameter name=\"path\" string=\"true\">src/a.rs",
                "</｜DSML｜parameter｜></｜DSML｜invoke｜></｜DSML｜tool_calls｜>",
            )),
            Message::user("<tool_result>fn main() {}</tool_result>"),
            Message::assistant("Done."),
        ]
    }

    #[test]
    fn segments_cover_every_message_kind() {
        let segs = segments(&transcript());
        assert_eq!(segs[0], Segment::User("fix the parser".to_owned()));
        assert_eq!(segs[1], Segment::Thinking("needs a look".to_owned()));
        assert_eq!(segs[2], Segment::Assistant("Reading it now.".to_owned()));
        let Segment::ToolCall(call) = &segs[3] else {
            panic!("expected a tool call, got {:?}", segs[3]);
        };
        assert_eq!(call.name, "read_file");
        assert_eq!(call.arg_value("path"), Some("src/a.rs"));
        assert_eq!(segs[4], Segment::ToolResult("fn main() {}".to_owned()));
        assert_eq!(segs[5], Segment::Assistant("Done.".to_owned()));
        assert_eq!(segs.len(), 6);
    }

    #[test]
    fn markdown_has_a_heading_per_kind() {
        let md = render(&transcript(), "my session", Format::Markdown);
        assert!(md.starts_with("# my session\n"));
        assert!(md.contains("## User"));
        assert!(md.contains("### Thinking"));
        assert!(md.contains("## Assistant"));
        assert!(md.contains("### Tool call: `read_file`"));
        assert!(md.contains("### Tool result"));
        assert!(md.contains("fn main() {}"));
        assert!(md.contains("> fix the parser"));
    }

    #[test]
    fn markdown_fence_outgrows_embedded_backticks() {
        let segs = vec![Segment::ToolResult("a ``` b".to_owned())];
        let md = render_markdown(&segs, "t");
        assert!(md.contains("````\na ``` b\n````"), "{md}");
    }

    #[test]
    fn html_is_self_contained_and_escapes_everything() {
        let msgs = vec![
            Message::user("<script>alert('x&y')</script>"),
            Message::assistant("if a < b && c > d {}"),
            Message::user("<tool_result><img src=x onerror=1></tool_result>"),
        ];
        let html = render(&msgs, "t<i>", Format::Html);
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<style>"));
        assert!(!html.contains("http://"), "no external assets");
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;alert(&#39;x&amp;y&#39;)&lt;/script&gt;"));
        assert!(html.contains("if a &lt; b &amp;&amp; c &gt; d {}"));
        assert!(html.contains("&lt;img src=x onerror=1&gt;"));
        assert!(html.contains("<title>t&lt;i&gt;</title>"));
    }

    #[test]
    fn html_escapes_tool_call_names_and_args() {
        let segs = segments(&[Message::assistant(concat!(
            "<｜DSML｜tool_calls><｜DSML｜invoke name=\"bash\">",
            "<｜DSML｜parameter name=\"cmd\" string=\"true\">echo \"<b>\" & true",
            "</｜DSML｜parameter｜></｜DSML｜invoke｜></｜DSML｜tool_calls｜>",
        ))]);
        let html = render_html(&segs, "t");
        assert!(!html.contains("<b>"));
        assert!(
            html.contains("echo &quot;&lt;b&gt;&quot; &amp; true"),
            "{html}"
        );
    }

    #[test]
    fn unparsable_dsml_falls_back_to_prose() {
        let segs = segments(&[Message::assistant("<｜DSML｜tool_calls>garbage")]);
        assert!(matches!(segs.as_slice(), [Segment::Assistant(_)]));
    }

    #[test]
    fn empty_transcript_still_renders_a_document() {
        let md = render(&[], "", Format::Markdown);
        assert_eq!(md, "# plank session\n\n");
        let html = render(&[], "", Format::Html);
        assert!(html.contains("<h1>plank session</h1>"));
        assert!(html.ends_with("</html>\n"));
    }

    #[test]
    fn format_parsing_and_filenames() {
        assert_eq!(parse_format("md"), Some(Format::Markdown));
        assert_eq!(parse_format(" HTML "), Some(Format::Html));
        assert_eq!(parse_format("pdf"), None);
        assert_eq!(
            default_filename("Fix the parser!", 42, Format::Html),
            "plank-session-fix-the-parser-42.html"
        );
        assert_eq!(
            default_filename("  ", 7, Format::Markdown),
            "plank-session-7.md"
        );
    }
}

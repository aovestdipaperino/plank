//! Browser web tools: `google_search` and `visit_page`.
//!
//! Port of the "Browser Web Tools" section of `ds4_agent.c` plus the page
//! extraction logic of `ds4_web.c`. The C engine drives a visible Chrome via
//! CDP and extracts Markdown by evaluating JavaScript in the page
//! (`web_extract_search_js` / `web_extract_page_js`). This port deviates on
//! transport: pages are fetched by shelling out to `curl -sL --max-time N`
//! (std-only crate, no HTTP client), and the extraction JavaScript is ported
//! as pure Rust HTML-to-Markdown functions operating on the raw HTML. As a
//! consequence dynamic (JS-rendered) content and the post-redirect
//! `location.href` are not available: `URL:` lines show the requested URL.
//! For safety with curl, only `http://` and `https://` URLs are accepted
//! (Chrome would happily navigate anywhere; curl must not read `file://`).
//!
//! The model-visible output formats are replicated exactly: the search
//! Markdown skeleton, the link-map bullet format `- [text](url)` with the
//! same per-tool caps (20 links / 180 chars for search, 80 links / 160 chars
//! for pages), the `[Content truncated by browser extractor.]` caption, the
//! `visit_page` head-plus-temp-file framing, and the `Tool error:` texts.
//!
//! The C approval flow (`agent_web_confirm`) is ported as the
//! [`super::ToolContext::web_confirm`] hook: the first web tool call per
//! session asks for approval with the C's exact prompt; a `None` hook
//! auto-denies with the C's non-interactive refusal message.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::process::Command;

use crate::dsml::ToolCall;

use super::ToolContext;

/// Maximum bytes of the rendered page shown inline (`AGENT_WEB_HEAD_BYTES`).
const WEB_HEAD_BYTES: usize = 8 * 1024;
/// Maximum lines of the rendered page shown inline (`AGENT_WEB_HEAD_LINES`).
const WEB_HEAD_LINES: usize = 100;
/// `curl --max-time` in seconds (the C's CDP timeout is 20 s).
const CURL_TIMEOUT_SEC: u32 = 20;
/// Cap on fetched HTML, mirroring the C's 4 MiB websocket message cap.
const MAX_HTML_BYTES: usize = 4 * 1024 * 1024;
/// Page Markdown length at which extraction stops (`web_extract_page_js`).
const PAGE_CONTENT_CAP: usize = 900_000;
/// Approval prompt shown before the first network access (`web_ensure_browser`).
const CONFIRM_PROMPT: &str = "The web tool wants to start a visible Chrome browser. Allow? (y/n) ";

/// Per-session web tool state (mirrors `ds4_web.browser_allowed`).
#[derive(Debug, Default)]
pub struct WebState {
    /// True once the user approved web access for this session.
    pub allowed: bool,
}

/// Executes the `google_search` tool: fetches and summarizes Google results.
pub fn tool_google_search(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let query = call.arg_value("query").unwrap_or("");
    if query.is_empty() {
        return "Tool error: google_search requires query\n".to_string();
    }
    if let Err(e) = ensure_allowed(ctx) {
        return format!("Tool error: google_search failed: {e}\n");
    }
    let url = format!("https://www.google.com/search?q={}", url_encode(query));
    match curl_fetch(&url) {
        Ok(html) => extract_search_markdown(&url, &html),
        Err(e) => format!("Tool error: google_search failed: {e}\n"),
    }
}

/// Executes the `visit_page` tool: renders a URL to Markdown with link map.
pub fn tool_visit_page(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let url = call.arg_value("url").unwrap_or("").to_string();
    if url.is_empty() {
        return "Tool error: visit_page requires url\n".to_string();
    }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return format!("Tool error: visit_page failed: unsupported URL scheme: {url}\n");
    }
    if let Err(e) = ensure_allowed(ctx) {
        return format!("Tool error: visit_page failed: {e}\n");
    }
    let html = match curl_fetch(&url) {
        Ok(html) => html,
        Err(e) => return format!("Tool error: visit_page failed: {e}\n"),
    };
    let md = extract_page_markdown(&url, &html);
    let path = match write_temp_text("ds4_agent_web", &md) {
        Ok(path) => path,
        Err(e) => return format!("Tool error: visit_page failed: {e}\n"),
    };
    frame_visit_output(&url, &path, &md)
}

/// Runs the session approval gate, mirroring `web_ensure_browser`.
///
/// # Errors
///
/// Returns the C refusal texts when no hook is installed or the user denies.
fn ensure_allowed(ctx: &mut ToolContext) -> Result<(), String> {
    if ctx.web.allowed {
        return Ok(());
    }
    let Some(confirm) = ctx.web_confirm.as_mut() else {
        return Err("visible Chrome browser startup requires interactive approval".to_string());
    };
    if !confirm(CONFIRM_PROMPT) {
        return Err("user denied Chrome browser start".to_string());
    }
    ctx.web.allowed = true;
    Ok(())
}

/// Fetches a URL with curl, returning the response body as lossy UTF-8.
///
/// Deviation from the C: transport is `curl -sL --max-time 20` instead of a
/// visible Chrome session; redirects are followed by curl.
///
/// # Errors
///
/// Returns a message when curl cannot be spawned or exits nonzero.
fn curl_fetch(url: &str) -> Result<String, String> {
    let out = Command::new("curl")
        .args([
            "-sL",
            "--max-time",
            &CURL_TIMEOUT_SEC.to_string(),
            "--compressed",
            "-A",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) \
             Chrome/124.0 Safari/537.36",
            "--",
            url,
        ])
        .output()
        .map_err(|e| format!("failed to run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "curl exited with status {}",
            out.status.code().unwrap_or(-1)
        ));
    }
    let mut body = String::from_utf8_lossy(&out.stdout).into_owned();
    if body.len() > MAX_HTML_BYTES {
        let mut end = MAX_HTML_BYTES;
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        body.truncate(end);
    }
    Ok(body)
}

/// Writes text to a fresh temp file, mirroring `agent_write_temp_text`.
///
/// # Errors
///
/// Returns a message when the file cannot be created or written.
fn write_temp_text(prefix: &str, text: &str) -> Result<String, String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "{prefix}_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).map_err(|e| format!("failed to create temporary file: {e}"))?;
    Ok(path.to_string_lossy().into_owned())
}

/// Percent-encodes a query string, mirroring `web_url_encode`.
#[must_use]
pub fn url_encode(s: impl AsRef<str>) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::new();
    for &b in s.as_ref().as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[usize::from(b >> 4)] as char);
            out.push(HEX[usize::from(b & 15)] as char);
        }
    }
    out
}

// ============================================================================
// HTML tokenizing and text helpers (pure, unit-tested)
// ============================================================================

/// One HTML token: raw text or a tag.
#[derive(Debug)]
enum Tok<'a> {
    /// Text between tags (entities not yet decoded).
    Text(&'a str),
    /// A tag with lowercase name, raw attribute text, and close flag.
    Tag {
        name: String,
        attrs: &'a str,
        closing: bool,
    },
}

/// Tokenizes HTML into text runs and tags; comments and doctypes are dropped.
fn tokenize(html: &str) -> Vec<Tok<'_>> {
    let mut toks = Vec::new();
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if html[i..].starts_with("<!--") {
                i = html[i..].find("-->").map_or(bytes.len(), |p| i + p + 3);
                continue;
            }
            let Some(end) = html[i + 1..].find('>') else {
                break;
            };
            let inner = &html[i + 1..i + 1 + end];
            i += end + 2;
            let (closing, inner) = match inner.strip_prefix('/') {
                Some(rest) => (true, rest),
                None => (false, inner),
            };
            if inner.starts_with('!') || inner.starts_with('?') {
                continue;
            }
            let name_end = inner
                .find(|c: char| c.is_whitespace() || c == '/')
                .unwrap_or(inner.len());
            let name = inner[..name_end].to_ascii_lowercase();
            if name.is_empty() {
                continue;
            }
            toks.push(Tok::Tag {
                name,
                attrs: &inner[name_end..],
                closing,
            });
        } else {
            let end = html[i..].find('<').map_or(bytes.len(), |p| i + p);
            toks.push(Tok::Text(&html[i..end]));
            i = end;
        }
    }
    toks
}

/// Decodes the common HTML entities used by rendered pages.
#[must_use]
pub fn decode_entities(s: impl AsRef<str>) -> String {
    let s = s.as_ref();
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        rest = &rest[pos..];
        let Some(semi) = rest[..rest.len().min(12)].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..semi];
        let decoded = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            _ => ent
                .strip_prefix("#x")
                .or_else(|| ent.strip_prefix("#X"))
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        if let Some(c) = decoded {
            out.push(c);
            rest = &rest[semi + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Collapses runs of whitespace to single spaces and trims (the JS `clean`).
#[must_use]
pub fn clean(s: impl AsRef<str>) -> String {
    let mut out = String::new();
    let mut prev_space = true;
    for c in s.as_ref().chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Escapes link text like the JS `esc`: backslashes, brackets, newlines.
#[must_use]
pub fn esc_link_text(s: impl AsRef<str>) -> String {
    clean(s)
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('\n', " ")
}

/// Truncates to at most `n` characters (the JS `.slice(0, n)`).
#[must_use]
pub fn slice_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Extracts one attribute value from a raw tag attribute string.
fn attr_value(attrs: &str, name: &str) -> Option<String> {
    let lower = attrs.to_ascii_lowercase();
    let mut from = 0;
    loop {
        let pos = lower[from..].find(name)? + from;
        let after = &attrs[pos + name.len()..];
        let before_ok = pos == 0
            || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric()
                && lower.as_bytes()[pos - 1] != b'-';
        let eq = after.trim_start().strip_prefix('=');
        match (before_ok, eq) {
            (true, Some(v)) => {
                let v = v.trim_start();
                let val = if let Some(rest) = v.strip_prefix('"') {
                    &rest[..rest.find('"').unwrap_or(rest.len())]
                } else if let Some(rest) = v.strip_prefix('\'') {
                    &rest[..rest.find('\'').unwrap_or(rest.len())]
                } else {
                    &v[..v
                        .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
                        .unwrap_or(v.len())]
                };
                return Some(decode_entities(val));
            }
            _ => from = pos + name.len(),
        }
    }
}

/// Resolves a possibly-relative href against the page URL (DOM `a.href`).
#[must_use]
pub fn resolve_url(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.starts_with("http://") || href.starts_with("https://") {
        return Some(href.to_string());
    }
    let scheme_end = base.find("://")?;
    let scheme = &base[..scheme_end];
    let after = &base[scheme_end + 3..];
    let host = &after[..after.find('/').unwrap_or(after.len())];
    if let Some(rest) = href.strip_prefix("//") {
        return Some(format!("{scheme}://{rest}"));
    }
    if href.starts_with('/') {
        return Some(format!("{scheme}://{host}{href}"));
    }
    if href.is_empty() || href.starts_with('#') || href.contains(':') {
        return None;
    }
    let path = &base[scheme_end + 3 + host.len()..];
    let dir = &path[..path.rfind('/').map_or(0, |p| p + 1)];
    let dir = if dir.is_empty() { "/" } else { dir };
    Some(format!("{scheme}://{host}{dir}{href}"))
}

/// A hyperlink found in the page: cleaned text plus absolute URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// Escaped, cleaned anchor text.
    pub text: String,
    /// Absolute http(s) URL.
    pub href: String,
}

/// Extracts visible-ish links in document order, mirroring the JS walkers.
///
/// Only http(s) links with at least 3 characters of text survive; Google
/// `/url?q=` redirect wrappers are unwrapped.
#[must_use]
pub fn extract_links(page_url: &str, html: &str) -> Vec<Link> {
    let mut links = Vec::new();
    let mut cur: Option<(String, String)> = None;
    let mut skip = 0usize;
    for tok in tokenize(html) {
        match tok {
            Tok::Tag {
                name,
                attrs,
                closing,
            } => match name.as_str() {
                "script" | "style" | "noscript" => {
                    if closing {
                        skip = skip.saturating_sub(1);
                    } else {
                        skip += 1;
                    }
                }
                "a" if !closing => {
                    if let Some(href) = attr_value(attrs, "href") {
                        cur = Some((href, String::new()));
                    }
                }
                "a" => {
                    if let Some((href, text)) = cur.take() {
                        let href = unwrap_google_redirect(&href);
                        if let Some(abs) = resolve_url(page_url, &href) {
                            let text = esc_link_text(decode_entities(&text));
                            if text.chars().count() >= 3 {
                                links.push(Link { text, href: abs });
                            }
                        }
                    }
                }
                _ => {}
            },
            Tok::Text(t) => {
                if skip == 0
                    && let Some((_, buf)) = cur.as_mut()
                {
                    buf.push_str(t);
                    buf.push(' ');
                }
            }
        }
    }
    links
}

/// Unwraps Google's `/url?q=<target>` redirect links (the JS `URL` dance).
#[must_use]
pub fn unwrap_google_redirect(href: &str) -> String {
    let path_start = if href.starts_with("http://") || href.starts_with("https://") {
        let p = href.find("://").unwrap_or(0);
        href[p + 3..].find('/').map_or(href.len(), |q| p + 3 + q)
    } else {
        0
    };
    if href[path_start..].starts_with("/url?") {
        for kv in href[path_start + 5..].split('&') {
            if let Some(v) = kv.strip_prefix("q=") {
                let v = &v[..v.find('#').unwrap_or(v.len())];
                return percent_decode(v);
            }
        }
    }
    href.to_string()
}

/// Decodes percent-escapes (for redirect targets); invalid escapes pass through.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Returns true for hostnames the search extractor skips (`bad` in the JS).
#[must_use]
pub fn is_google_host(host: &str) -> bool {
    for dom in ["google.", "gstatic.", "googleusercontent."] {
        if host.starts_with(dom) || host.contains(&format!(".{dom}")) {
            return true;
        }
    }
    false
}

/// Extracts the hostname from an http(s) URL.
fn url_host(url: &str) -> &str {
    let after = url.find("://").map_or(url, |p| &url[p + 3..]);
    let end = after.find(['/', '?', '#']).unwrap_or(after.len());
    let hostport = &after[..end];
    hostport.rsplit_once(':').map_or(hostport, |(h, _)| h)
}

/// Extracts visible text from HTML, dropping script/style/noscript.
#[must_use]
pub fn html_to_text(html: &str) -> String {
    let mut out = String::new();
    let mut skip = 0usize;
    for tok in tokenize(html) {
        match tok {
            Tok::Tag { name, closing, .. } => match name.as_str() {
                "script" | "style" | "noscript" => {
                    if closing {
                        skip = skip.saturating_sub(1);
                    } else {
                        skip += 1;
                    }
                }
                _ => out.push(' '),
            },
            Tok::Text(t) => {
                if skip == 0 {
                    out.push_str(&decode_entities(t));
                }
            }
        }
    }
    out
}

/// Extracts the `<title>` text, if any.
fn html_title(html: &str) -> Option<String> {
    let mut in_title = false;
    for tok in tokenize(html) {
        match tok {
            Tok::Tag { name, closing, .. } if name == "title" => {
                if closing && in_title {
                    return None;
                }
                in_title = !closing;
            }
            Tok::Text(t) if in_title => {
                let t = clean(decode_entities(t));
                return if t.is_empty() { None } else { Some(t) };
            }
            _ => {}
        }
    }
    None
}

// ============================================================================
// Markdown builders (formats replicated from ds4_web.c's extraction JS)
// ============================================================================

/// Builds the `google_search` Markdown, mirroring `web_extract_search_js`.
#[must_use]
pub fn extract_search_markdown(url: &str, html: &str) -> String {
    let mut lines = vec![
        "# Google search results".to_string(),
        String::new(),
        format!("URL: {url}"),
        String::new(),
        "## Visible links".to_string(),
    ];
    let mut seen = HashSet::new();
    for link in extract_links(url, html) {
        if is_google_host(url_host(&link.href)) || !seen.insert(link.href.clone()) {
            continue;
        }
        lines.push(format!(
            "- [{}]({})",
            slice_chars(&link.text, 180),
            link.href
        ));
        if seen.len() >= 20 {
            break;
        }
    }
    lines.push(String::new());
    lines.push("## Text snapshot".to_string());
    let text = clean(html_to_text(html));
    lines.push(slice_chars(&text, 1200).to_string());
    lines.join("\n")
}

/// Blocks the page extractor turns into Markdown (`web_extract_page_js`).
const BLOCK_TAGS: [&str; 14] = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "pre",
    "blockquote",
    "td",
    "th",
    "dt",
    "dd",
];

/// Builds the `visit_page` Markdown, mirroring `web_extract_page_js`.
///
/// One long function on purpose: it is a line-for-line port of the C's
/// single-pass extraction script.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn extract_page_markdown(url: &str, html: &str) -> String {
    let title = html_title(html).unwrap_or_else(|| url.to_string());
    let mut lines = vec![
        format!("# {title}"),
        String::new(),
        format!("URL: {url}"),
        String::new(),
        "## Content".to_string(),
    ];
    let mut total: usize = lines.iter().map(|l| l.len() + 1).sum();
    let mut seen = HashSet::new();
    let mut skip = 0usize;
    // Current open block: (tag, accumulated inline markdown).
    let mut block: Option<(String, String)> = None;
    // Open link inside the block: (href, text).
    let mut anchor: Option<(String, String)> = None;
    let mut in_code = false;
    let mut truncated = false;

    let flush = |tag: &str,
                 body: String,
                 lines: &mut Vec<String>,
                 seen: &mut HashSet<String>,
                 total: &mut usize|
     -> bool {
        let s = match tag {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = tag[1..].parse::<usize>().unwrap_or(1);
                format!("{} {}", "#".repeat(level), clean(&body))
            }
            "li" => format!("- {}", clean(&body)),
            "pre" => {
                let inner = decode_entities(body.trim_end());
                format!("```\n{inner}\n```")
            }
            "blockquote" => format!("> {}", clean(&body)),
            _ => clean(&body),
        };
        let s = s.trim().to_string();
        if s.is_empty() || !seen.insert(s.clone()) {
            return false;
        }
        *total += s.len() + 2;
        lines.push(String::new());
        lines.push(s);
        if *total > PAGE_CONTENT_CAP {
            lines.push(String::new());
            lines.push("[Content truncated by browser extractor.]".to_string());
            return true;
        }
        false
    };

    'toks: for tok in tokenize(html) {
        match tok {
            Tok::Tag {
                name,
                attrs,
                closing,
            } => match name.as_str() {
                "script" | "style" | "noscript" => {
                    if closing {
                        skip = skip.saturating_sub(1);
                    } else {
                        skip += 1;
                    }
                }
                _ if skip > 0 => {}
                "a" if block.is_some() => {
                    if closing {
                        if let Some((href, text)) = anchor.take() {
                            let text = esc_link_text(decode_entities(&text));
                            let out = match resolve_url(url, &href) {
                                Some(abs) if !text.is_empty() => format!("[{text}]({abs})"),
                                _ => text,
                            };
                            if let Some((_, buf)) = block.as_mut() {
                                buf.push_str(&out);
                            }
                        }
                    } else if let Some(href) = attr_value(attrs, "href") {
                        anchor = Some((href, String::new()));
                    }
                }
                "code" if block.as_ref().is_some_and(|(t, _)| t != "pre") => {
                    if let Some((_, buf)) = block.as_mut() {
                        buf.push('`');
                        in_code = !closing;
                    }
                }
                tag if BLOCK_TAGS.contains(&tag) => {
                    if let Some((cur, body)) = block.take() {
                        anchor = None;
                        in_code = false;
                        if flush(&cur, body, &mut lines, &mut seen, &mut total) {
                            truncated = true;
                            break 'toks;
                        }
                    }
                    if !closing {
                        block = Some((tag.to_string(), String::new()));
                    }
                }
                _ => {
                    if let Some((tag, buf)) = block.as_mut()
                        && tag != "pre"
                    {
                        buf.push(' ');
                    }
                }
            },
            Tok::Text(t) => {
                if skip > 0 {
                    continue;
                }
                if let Some((_, buf)) = anchor.as_mut() {
                    buf.push_str(t);
                } else if let Some((tag, buf)) = block.as_mut() {
                    if tag == "pre" {
                        buf.push_str(t);
                    } else if in_code {
                        buf.push_str(&clean(decode_entities(t)).replace('`', "\\`"));
                    } else {
                        buf.push_str(&decode_entities(t));
                    }
                }
            }
        }
    }
    if !truncated && let Some((cur, body)) = block.take() {
        flush(&cur, body, &mut lines, &mut seen, &mut total);
    }

    lines.push(String::new());
    lines.push("## Visible links".to_string());
    let mut link_seen = HashSet::new();
    let mut n = 0;
    for link in extract_links(url, html) {
        if !link_seen.insert(link.href.clone()) {
            continue;
        }
        lines.push(format!(
            "- [{}]({})",
            slice_chars(&link.text, 160),
            link.href
        ));
        n += 1;
        if n >= 80 {
            break;
        }
    }
    lines.join("\n")
}

// ============================================================================
// visit_page output framing (agent side of ds4_agent.c)
// ============================================================================

/// Counts display lines like `agent_count_lines` (no trailing-newline line).
#[must_use]
pub fn count_lines(s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    let n = s.bytes().filter(|&b| b == b'\n').count();
    if s.ends_with('\n') { n } else { n + 1 }
}

/// Takes the head of a string like `agent_string_head`.
///
/// Returns `(head, lines_read, byte_limited)`; the cut is byte-based like the
/// C but backed off to the nearest UTF-8 boundary.
#[must_use]
pub fn string_head(s: &str, max_lines: usize, max_bytes: usize) -> (String, usize, bool) {
    let bytes = s.as_bytes();
    let mut used = 0;
    let mut lines = 0;
    while used < bytes.len() && used < max_bytes && lines < max_lines {
        if bytes[used] == b'\n' {
            lines += 1;
        }
        used += 1;
    }
    let byte_limited = used < bytes.len() && used >= max_bytes;
    if used > 0 && bytes[used - 1] != b'\n' && lines < max_lines {
        lines += 1;
    }
    while used > 0 && !s.is_char_boundary(used) {
        used -= 1;
    }
    (s[..used].to_string(), lines, byte_limited)
}

/// Frames the rendered page like `agent_tool_visit_page` (head + temp file).
#[must_use]
pub fn frame_visit_output(url: &str, path: &str, md: &str) -> String {
    let total_lines = count_lines(md);
    let (head, shown_lines, byte_limited) = string_head(md, WEB_HEAD_LINES, WEB_HEAD_BYTES);
    let truncated = byte_limited || shown_lines < total_lines;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "visit_page url={url}\noutput_path={path} ({} bytes, {total_lines} lines)",
        md.len()
    );
    if truncated {
        let _ = writeln!(out, "<head -{WEB_HEAD_LINES} {path}>");
        out.push_str(&head);
        if !head.is_empty() && !head.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</head>\n");
        out.push_str(
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n",
        );
    } else {
        out.push_str("<markdown>\n");
        out.push_str(&head);
        if !head.is_empty() && !head.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("</markdown>\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{dispatch, test_call, test_ctx};

    #[test]
    fn text_helpers() {
        assert_eq!(clean("  a \n\t b  "), "a b");
        assert_eq!(
            decode_entities("a &amp; b &lt;x&gt; &#65;&#x42; &bogus;"),
            "a & b <x> AB &bogus;"
        );
        assert_eq!(esc_link_text("a [b] \\c"), "a \\[b\\] \\\\c");
        assert_eq!(slice_chars("héllo", 2), "hé");
        assert_eq!(url_encode("a b/~c"), "a%20b%2F~c");
    }

    #[test]
    fn url_helpers() {
        assert_eq!(
            resolve_url("https://x.com/a/b", "/z").as_deref(),
            Some("https://x.com/z")
        );
        assert_eq!(
            resolve_url("https://x.com/a/b", "c").as_deref(),
            Some("https://x.com/a/c")
        );
        assert_eq!(
            resolve_url("https://x.com/a", "//y.com/q").as_deref(),
            Some("https://y.com/q")
        );
        assert_eq!(resolve_url("https://x.com/a", "javascript:void(0)"), None);
        assert_eq!(
            unwrap_google_redirect("https://www.google.com/url?q=https%3A%2F%2Fex.com%2Fp&sa=U"),
            "https://ex.com/p"
        );
        assert!(is_google_host("www.google.com"));
        assert!(is_google_host("gstatic.com"));
        assert!(!is_google_host("notgoogle.example.com"));
    }

    #[test]
    fn link_extraction_filters_short_and_relative() {
        let html = r#"<body><a href="https://a.com/x">Read this</a>
            <a href="/rel">Relative link</a>
            <a href="https://b.com">no</a>
            <script><a href="https://evil">Hidden link</a></script></body>"#;
        let links = extract_links("https://base.org/dir/page", html);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].href, "https://a.com/x");
        assert_eq!(links[0].text, "Read this");
        assert_eq!(links[1].href, "https://base.org/rel");
    }

    #[test]
    fn search_markdown_format() {
        let html = r#"<html><body>
            <a href="https://www.google.com/settings">Google settings link</a>
            <a href="/url?q=https://ex.com/hit&amp;sa=U">Example result title</a>
            <p>Snippet body text</p></body></html>"#;
        let md = extract_search_markdown("https://www.google.com/search?q=x", html);
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "# Google search results");
        assert_eq!(lines[2], "URL: https://www.google.com/search?q=x");
        assert_eq!(lines[4], "## Visible links");
        assert_eq!(lines[5], "- [Example result title](https://ex.com/hit)");
        assert_eq!(lines[7], "## Text snapshot");
        assert!(lines[8].contains("Snippet body text"));
    }

    #[test]
    fn page_markdown_format() {
        let html = r#"<html><head><title> My &amp; Page </title></head><body>
            <h2>Section</h2>
            <p>Hello <a href="https://l.com/d">deep link</a> world</p>
            <p>Hello <a href="https://l.com/d">deep link</a> world</p>
            <li>item one</li>
            <pre>let x = 1;
let y = 2;</pre>
            <blockquote>quoted words</blockquote>
            </body></html>"#;
        let md = extract_page_markdown("https://p.com/a", html);
        assert!(md.starts_with("# My & Page\n\nURL: https://p.com/a\n\n## Content\n"));
        assert!(md.contains("\n## Section\n"));
        assert!(md.contains("Hello [deep link](https://l.com/d) world"));
        // Duplicate block deduplicated.
        assert_eq!(md.matches("Hello [deep link]").count(), 1);
        assert!(md.contains("\n- item one\n"));
        assert!(md.contains("```\nlet x = 1;\nlet y = 2;\n```"));
        assert!(md.contains("> quoted words"));
        assert!(md.contains("\n## Visible links\n- [deep link](https://l.com/d)"));
    }

    #[test]
    fn page_markdown_truncation_caption() {
        let mut html = String::from("<body>");
        for i in 0..12000 {
            let _ = write!(html, "<p>block {i} {}</p>", "x".repeat(90));
        }
        html.push_str("</body>");
        let md = extract_page_markdown("https://p.com", &html);
        assert!(md.contains("[Content truncated by browser extractor.]"));
    }

    #[test]
    fn head_and_line_counting() {
        assert_eq!(count_lines(""), 0);
        assert_eq!(count_lines("a\nb"), 2);
        assert_eq!(count_lines("a\nb\n"), 2);
        let (head, lines, limited) = string_head("a\nb\nc\n", 2, 100);
        assert_eq!((head.as_str(), lines, limited), ("a\nb\n", 2, false));
        let (head, _, limited) = string_head("abcdef", 10, 3);
        assert_eq!((head.as_str(), limited), ("abc", true));
    }

    #[test]
    fn frame_short_output_uses_markdown_tags() {
        let out = frame_visit_output("https://u", "/tmp/f", "# T\nbody");
        assert_eq!(
            out,
            "visit_page url=https://u\noutput_path=/tmp/f (8 bytes, 2 lines)\n<markdown>\n# T\nbody\n</markdown>\n"
        );
    }

    #[test]
    fn frame_long_output_uses_head_caption() {
        let md = (0..200).fold(String::new(), |mut s, i| {
            let _ = writeln!(s, "line {i}");
            s
        });
        let out = frame_visit_output("https://u", "/tmp/f", &md);
        assert!(out.contains("<head -100 /tmp/f>\n"));
        assert!(out.contains("line 99\n</head>\n"));
        assert!(out.ends_with(
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n"
        ));
        assert!(!out.contains("line 100\n"));
    }

    #[test]
    fn missing_args_and_scheme_errors() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("google_search", &[]), &mut ctx);
        assert_eq!(res.output, "Tool error: google_search requires query\n");
        let res = dispatch(&test_call("visit_page", &[]), &mut ctx);
        assert_eq!(res.output, "Tool error: visit_page requires url\n");
        let res = dispatch(
            &test_call("visit_page", &[("url", "file:///etc/passwd")]),
            &mut ctx,
        );
        assert!(
            res.output
                .starts_with("Tool error: visit_page failed: unsupported URL scheme")
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn approval_gate_denies_without_hook() {
        let (mut ctx, dir) = test_ctx();
        let res = dispatch(&test_call("google_search", &[("query", "q")]), &mut ctx);
        assert_eq!(
            res.output,
            "Tool error: google_search failed: visible Chrome browser startup requires interactive approval\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn approval_gate_denial_and_grant() {
        let (mut ctx, dir) = test_ctx();
        ctx.web_confirm = Some(Box::new(|msg: &str| {
            assert_eq!(
                msg,
                "The web tool wants to start a visible Chrome browser. Allow? (y/n) "
            );
            false
        }));
        let res = dispatch(&test_call("visit_page", &[("url", "https://x")]), &mut ctx);
        assert_eq!(
            res.output,
            "Tool error: visit_page failed: user denied Chrome browser start\n"
        );
        assert!(!ctx.web.allowed);
        ctx.web_confirm = Some(Box::new(|_| true));
        // Grant path flips the sticky per-session flag before any fetch.
        assert!(ensure_allowed(&mut ctx).is_ok());
        assert!(ctx.web.allowed);
        std::fs::remove_dir_all(dir).ok();
    }
}

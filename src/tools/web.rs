// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

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
//! On `ds4_engine` builds the tools instead drive the C engine's real browser
//! (`ds4_web.c`, Chrome over CDP) via [`crate::ds4web`], so results come from a
//! genuine browser that dodges the bot challenges plain curl trips; the curl
//! path above is the fallback for non-`ds4` builds (CI/dev).
//!
//! With the `use_obscura` feature (default) both of those transports are
//! replaced by the embedded obscura headless browser ([`crate::obscura_web`],
//! statically linked from the `refs/obscura` submodule): pages are fetched and
//! JS-rendered in-process, then flow through the same Rust extraction below.
//!
//! The C approval flow (`agent_web_confirm`) is ported as the
//! [`super::ToolContext::web_confirm`] hook and the ask-bridge gate: the first
//! web tool call per session asks for approval, and an "Always allow" answer is
//! persisted as durable consent ([`crate::consent`]) so future sessions skip
//! the prompt. A `None` hook (non-interactive) auto-denies.

use std::collections::HashSet;
use std::fmt::Write as _;
#[cfg(not(feature = "use_obscura"))]
use std::process::Command;

use crate::dsml::ToolCall;

use super::ToolContext;

/// Maximum bytes of the rendered page shown inline (`AGENT_WEB_HEAD_BYTES`).
const WEB_HEAD_BYTES: usize = 8 * 1024;
/// Maximum lines of the rendered page shown inline (`AGENT_WEB_HEAD_LINES`).
const WEB_HEAD_LINES: usize = 100;
/// `curl --max-time` in seconds (the C's CDP timeout is 20 s).
#[cfg(not(feature = "use_obscura"))]
#[cfg_attr(ds4_engine, allow(dead_code))]
const CURL_TIMEOUT_SEC: u32 = 20;
/// Cap on fetched HTML, mirroring the C's 4 MiB websocket message cap.
#[cfg_attr(all(ds4_engine, not(feature = "use_obscura")), allow(dead_code))]
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

/// Executes the `google_search` tool.
///
/// On `ds4_engine` builds this drives the ds4 C engine's real browser (Chrome
/// over CDP) after the approval gate, returning the C extractor's Markdown. On
/// other builds it falls back to a client-side `DuckDuckGo` HTML scrape (curl,
/// no browser) with optional `allowed_domains`/`blocked_domains` filtering.
#[cfg_attr(
    all(ds4_engine, not(feature = "use_obscura")),
    allow(clippy::needless_return)
)]
pub fn tool_google_search(ctx: &mut ToolContext, call: &ToolCall) -> String {
    let query = call.arg_value("query").unwrap_or("").trim();
    if query.is_empty() {
        return "Tool error: google_search requires query\n".to_string();
    }
    #[cfg(all(ds4_engine, not(feature = "use_obscura")))]
    {
        let _ = call;
        if let Err(e) = ensure_allowed(ctx) {
            return format!("Tool error: google_search failed: {e}\n");
        }
        let query = query.to_string();
        return match browser(ctx).and_then(|b| b.google_search(&query)) {
            Ok(md) => md,
            Err(e) => format!("Tool error: google_search failed: {e}\n"),
        };
    }
    #[cfg(any(not(ds4_engine), feature = "use_obscura"))]
    {
        let _ = ctx;
        let allowed = parse_domain_list(call.arg_value("allowed_domains").unwrap_or(""));
        let blocked = parse_domain_list(call.arg_value("blocked_domains").unwrap_or(""));
        if !allowed.is_empty() && !blocked.is_empty() {
            return "Tool error: google_search failed: allowed_domains and blocked_domains are mutually exclusive\n"
                .to_string();
        }
        let url = format!("https://html.duckduckgo.com/html/?q={}", url_encode(query));
        let html = match fetch_html(&url) {
            Ok(html) => html,
            Err(e) => return format!("Tool error: google_search failed: {e}\n"),
        };
        if is_ddg_challenge(&html) {
            return "Tool error: google_search failed: DuckDuckGo returned a bot-verification challenge instead of results\n".to_string();
        }
        let hits = filter_by_domains(parse_ddg_results(&html), &allowed, &blocked);
        render_search_results(query, &hits)
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
    // On ds4_engine builds (without obscura) render the page in the C
    // engine's real browser; elsewhere fetch the HTML (obscura or curl) and
    // extract Markdown in Rust.
    #[cfg(all(ds4_engine, not(feature = "use_obscura")))]
    let md = {
        let url = url.clone();
        match browser(ctx).and_then(|b| b.visit_page(&url)) {
            Ok(md) => md,
            Err(e) => return format!("Tool error: visit_page failed: {e}\n"),
        }
    };
    #[cfg(any(not(ds4_engine), feature = "use_obscura"))]
    let md = {
        let html = match fetch_html(&url) {
            Ok(html) => html,
            Err(e) => return format!("Tool error: visit_page failed: {e}\n"),
        };
        extract_page_markdown(&url, &html)
    };
    let path = match write_temp_text("ds4_agent_web", &md) {
        Ok(path) => path,
        Err(e) => return format!("Tool error: visit_page failed: {e}\n"),
    };
    frame_visit_output(&url, &path, &md)
}

/// Splits a comma-separated domain list arg into lowercased, non-empty hosts.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn parse_domain_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Lazily creates and returns the session's C-engine browser, rooted at `$HOME`
/// (falling back to the working directory). Reused across turns.
///
/// # Errors
/// Returns a message if the browser subsystem could not be created.
#[cfg(all(ds4_engine, not(feature = "use_obscura")))]
fn browser(ctx: &mut ToolContext) -> Result<&mut crate::ds4web::WebBrowser, String> {
    if ctx.web_browser.is_none() {
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map_or_else(|| ctx.cwd.clone(), std::path::PathBuf::from);
        ctx.web_browser = Some(crate::ds4web::WebBrowser::new(&home)?);
    }
    Ok(ctx.web_browser.as_mut().expect("web_browser was just set"))
}

/// Runs the session approval gate, mirroring `web_ensure_browser`.
///
/// The Ratatui TUI runs its own event loop with the terminal in raw mode, so
/// the stdin-reading [`web_confirm`](ToolContext::web_confirm) hook would block
/// forever and freeze the agent. When an ask bridge is present (TUI only), the
/// gate is routed through the [`Asker`](crate::tools::ask::Asker) the event
/// loop already services; the plain REPL and tests keep the stdin hook.
///
/// The user may answer "Always allow", which is recorded as durable per-user
/// consent ([`crate::consent`]) so future sessions skip the prompt.
///
/// # Errors
///
/// Returns the C refusal texts when no confirmation path exists or the user denies.
fn ensure_allowed(ctx: &mut ToolContext) -> Result<(), String> {
    if ctx.web.allowed {
        return Ok(());
    }
    // Standing consent from a previous "Always allow" — never prompt again.
    // Skipped under `cfg(test)` so unit tests don't depend on the developer's
    // real `~/.plank/web-consent` marker.
    #[cfg(not(test))]
    if crate::consent::web_consent_granted() {
        ctx.web.allowed = true;
        return Ok(());
    }
    // TUI: approve via the ask bridge, never blocking stdin under raw mode.
    if ctx.ask_bridge.is_some()
        && let Some(asker) = ctx.asker.as_mut()
    {
        let req = crate::tools::ask::AskRequest {
            question: "Allow the web tool to start a visible Chrome browser?".to_string(),
            header: "Web".to_string(),
            options: vec![
                crate::tools::ask::AskOption {
                    label: "Allow".to_string(),
                    description: "Allow web access for this session".to_string(),
                },
                crate::tools::ask::AskOption {
                    label: "Always allow".to_string(),
                    description: "Allow web access now and in every future session".to_string(),
                },
                crate::tools::ask::AskOption {
                    label: "Deny".to_string(),
                    description: "Refuse web access for this session".to_string(),
                },
            ],
            multi: false,
        };
        return match asker.ask(req) {
            crate::tools::ask::AskOutcome::Answered(labels)
                if labels.iter().any(|l| l == "Always allow") =>
            {
                // Best-effort persistence; a write failure still grants the
                // session so the current call proceeds.
                if let Err(e) = crate::consent::grant_web_consent() {
                    ctx.hook_warnings
                        .push(format!("could not save web consent: {e}"));
                }
                ctx.web.allowed = true;
                Ok(())
            }
            crate::tools::ask::AskOutcome::Answered(labels)
                if labels.iter().any(|l| l == "Allow") =>
            {
                ctx.web.allowed = true;
                Ok(())
            }
            _ => Err("user denied Chrome browser start".to_string()),
        };
    }
    // Plain REPL / tests: the stdin confirm hook. Its bool cannot carry the
    // "always" choice, so persistence is only offered on the TUI path.
    let Some(confirm) = ctx.web_confirm.as_mut() else {
        return Err("visible Chrome browser startup requires interactive approval".to_string());
    };
    if !confirm(CONFIRM_PROMPT) {
        return Err("user denied Chrome browser start".to_string());
    }
    ctx.web.allowed = true;
    Ok(())
}

/// Fetches page HTML through the configured transport: the embedded obscura
/// headless browser when the `use_obscura` feature is on, else curl. The body
/// is capped at [`MAX_HTML_BYTES`] on a UTF-8 boundary either way.
///
/// # Errors
///
/// Returns a message when the fetch fails.
#[cfg_attr(all(ds4_engine, not(feature = "use_obscura")), allow(dead_code))]
fn fetch_html(url: &str) -> Result<String, String> {
    #[cfg(feature = "use_obscura")]
    let mut body = crate::obscura_web::fetch(url)?;
    #[cfg(not(feature = "use_obscura"))]
    let mut body = curl_fetch(url)?;
    if body.len() > MAX_HTML_BYTES {
        let mut end = MAX_HTML_BYTES;
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        body.truncate(end);
    }
    Ok(body)
}

/// Fetches a URL with curl, returning the response body as lossy UTF-8.
///
/// Deviation from the C: transport is `curl -sL --max-time 20` instead of a
/// visible Chrome session; redirects are followed by curl.
///
/// # Errors
///
/// Returns a message when curl cannot be spawned or exits nonzero.
#[cfg(not(feature = "use_obscura"))]
#[cfg_attr(ds4_engine, allow(dead_code))]
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
// web_search: DuckDuckGo client-side search (pure, unit-tested)
// ============================================================================

/// Max result links rendered by `web_search` (mirrors the search extractor cap).
#[cfg_attr(ds4_engine, allow(dead_code))]
const SEARCH_MAX_LINKS: usize = 20;
/// Max title characters rendered per link.
#[cfg_attr(ds4_engine, allow(dead_code))]
const SEARCH_TITLE_CAP: usize = 180;

/// One parsed `DuckDuckGo` result.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(ds4_engine, allow(dead_code))]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

/// True when the page is `DuckDuckGo`'s anti-bot challenge, not results.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn is_ddg_challenge(html: &str) -> bool {
    html.contains("anomaly-modal") || html.contains("challenge-form")
}

/// Resolves a `DuckDuckGo` result href to the real destination URL.
///
/// DDG wraps hits as `//duckduckgo.com/l/?uddg=<encoded url>&...`; this decodes
/// the `uddg` param. Scheme-relative `//host/...` becomes `https://host/...`.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn decode_ddg_href(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let rest = &href[pos + "uddg=".len()..];
        let val = rest.split('&').next().unwrap_or(rest);
        let decoded = percent_decode(val);
        if !decoded.is_empty() {
            return decoded;
        }
    }
    if let Some(rest) = href.strip_prefix("//") {
        return format!("https://{rest}");
    }
    href.to_string()
}

/// True if a raw tag's `class` attribute contains `needle` as a token.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn class_contains(attrs: &str, needle: &str) -> bool {
    attr_value(attrs, "class").is_some_and(|c| c.split_whitespace().any(|t| t == needle))
}

/// Concatenates text tokens from `start` until the matching close tag `name`.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn collect_text_until_close(toks: &[Tok<'_>], start: usize, name: &str) -> String {
    let mut out = String::new();
    for tok in &toks[start..] {
        match tok {
            Tok::Text(t) => out.push_str(t),
            Tok::Tag {
                name: n,
                closing: true,
                ..
            } if n == name => break,
            Tok::Tag { .. } => {}
        }
    }
    out
}

/// Extracts result hits from a `DuckDuckGo` HTML results page.
///
/// Walks tokens: an `<a class="result__a">` opens a hit (href → real URL via
/// [`decode_ddg_href`], inner text → title); a following
/// `<a class="result__snippet">` before the next result attaches the snippet.
/// Hits with an empty title or URL are dropped.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn parse_ddg_results(html: &str) -> Vec<SearchHit> {
    let toks = tokenize(html);
    let mut hits: Vec<SearchHit> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let Tok::Tag {
            name,
            attrs,
            closing: false,
        } = &toks[i]
        {
            if name == "a" && class_contains(attrs, "result__a") {
                let url = attr_value(attrs, "href")
                    .map(|h| decode_ddg_href(&h))
                    .unwrap_or_default();
                let title = clean(decode_entities(collect_text_until_close(&toks, i + 1, "a")));
                if !title.is_empty() && !url.is_empty() {
                    hits.push(SearchHit {
                        title,
                        url,
                        snippet: String::new(),
                    });
                }
                i += 1;
                continue;
            }
            if name == "a" && class_contains(attrs, "result__snippet") {
                let snip = clean(decode_entities(collect_text_until_close(&toks, i + 1, "a")));
                if let Some(last) = hits.last_mut()
                    && last.snippet.is_empty()
                {
                    last.snippet = snip;
                }
                i += 1;
                continue;
            }
        }
        i += 1;
    }
    hits
}

/// Lowercased host of an `http(s)` URL, or `None` if not parseable.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn host_of(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let host = host.split('@').next_back().unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        None
    } else {
        Some(host.to_ascii_lowercase())
    }
}

/// True if `host` equals `domain` or is a subdomain of it.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn host_in_domain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Keeps hits per allow/block lists. `allowed` (if non-empty) is a whitelist;
/// `blocked` is a blacklist. Callers guarantee at most one is non-empty.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn filter_by_domains(
    hits: Vec<SearchHit>,
    allowed: &[String],
    blocked: &[String],
) -> Vec<SearchHit> {
    hits.into_iter()
        .filter(|h| {
            let Some(host) = host_of(&h.url) else {
                return false;
            };
            if !allowed.is_empty() {
                return allowed.iter().any(|d| host_in_domain(&host, d));
            }
            !blocked.iter().any(|d| host_in_domain(&host, d))
        })
        .collect()
}

/// Renders hits as a plank-style link map with a query header.
#[cfg_attr(ds4_engine, allow(dead_code))]
fn render_search_results(query: &str, hits: &[SearchHit]) -> String {
    let mut out = format!("Web search results for query: \"{query}\"\n\n");
    if hits.is_empty() {
        out.push_str("No results.\n");
        return out;
    }
    for h in hits.iter().take(SEARCH_MAX_LINKS) {
        let title = slice_chars(&esc_link_text(&h.title), SEARCH_TITLE_CAP).to_string();
        if h.snippet.is_empty() {
            let _ = writeln!(out, "- [{title}]({})", h.url);
        } else {
            let snip = esc_link_text(&h.snippet);
            let _ = writeln!(out, "- [{title}]({}) — {snip}", h.url);
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

    /// Fake asker returning a fixed outcome, standing in for the TUI bridge.
    struct FixedAsker(crate::tools::ask::AskOutcome);
    impl crate::tools::ask::Asker for FixedAsker {
        fn ask(&mut self, _req: crate::tools::ask::AskRequest) -> crate::tools::ask::AskOutcome {
            self.0.clone()
        }
    }

    #[test]
    fn tui_approval_uses_asker_not_stdin() {
        // With an ask bridge present (TUI), approval must route through the
        // asker — never the stdin hook, which would deadlock under raw mode.
        let (mut ctx, dir) = test_ctx();
        ctx.ask_bridge = Some(crate::tools::ask::AskBridge::new());
        // A stdin hook that would panic if ever called proves it is bypassed.
        ctx.web_confirm = Some(Box::new(|_| panic!("stdin hook must not run in TUI mode")));

        ctx.asker = Some(Box::new(FixedAsker(
            crate::tools::ask::AskOutcome::Answered(vec!["Allow".to_string()]),
        )));
        assert!(ensure_allowed(&mut ctx).is_ok());
        assert!(ctx.web.allowed);

        // A declining asker keeps the gate closed.
        ctx.web.allowed = false;
        ctx.asker = Some(Box::new(FixedAsker(
            crate::tools::ask::AskOutcome::Declined,
        )));
        assert_eq!(
            ensure_allowed(&mut ctx),
            Err("user denied Chrome browser start".to_string())
        );
        assert!(!ctx.web.allowed);
        std::fs::remove_dir_all(dir).ok();
    }

    // ---- google_search (DuckDuckGo) ----

    const DDG_FIXTURE: &str = r##"<html><body>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F&rut=1">The <b>Rust</b> Language</a>
  <a class="result__snippet" href="#">A language empowering everyone.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://doc.rust-lang.org/book/">The Rust Book</a>
</div>
<a class="result__a" href="">   </a>
</body></html>"##;

    #[test]
    fn percent_decode_basic() {
        assert_eq!(percent_decode("a%20b%2Fc"), "a b/c");
        assert_eq!(percent_decode("bad%2"), "bad%2");
    }

    #[test]
    fn decode_ddg_href_unwraps_uddg() {
        let h = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp%3Fa%3D1&rut=xyz";
        assert_eq!(decode_ddg_href(h), "https://example.com/p?a=1");
    }

    #[test]
    fn decode_ddg_href_passthrough_and_scheme() {
        assert_eq!(
            decode_ddg_href("https://direct.example/x"),
            "https://direct.example/x"
        );
        assert_eq!(
            decode_ddg_href("//host.example/y"),
            "https://host.example/y"
        );
    }

    #[test]
    fn parse_ddg_results_extracts_hits() {
        let hits = parse_ddg_results(DDG_FIXTURE);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "The Rust Language");
        assert_eq!(hits[0].url, "https://rust-lang.org/");
        assert_eq!(hits[0].snippet, "A language empowering everyone.");
        assert_eq!(hits[1].url, "https://doc.rust-lang.org/book/");
        assert_eq!(hits[1].snippet, "");
    }

    fn hit(url: &str) -> SearchHit {
        SearchHit {
            title: "t".into(),
            url: url.into(),
            snippet: String::new(),
        }
    }

    #[test]
    fn host_of_extracts_host() {
        assert_eq!(
            host_of("https://a.example.com/p?x=1"),
            Some("a.example.com".to_string())
        );
        assert_eq!(
            host_of("http://Example.COM"),
            Some("example.com".to_string())
        );
        assert_eq!(host_of("not a url"), None);
    }

    #[test]
    fn filter_by_domains_allow_and_block() {
        let hits = vec![hit("https://a.example.com/1"), hit("https://other.org/2")];
        let allowed = vec!["example.com".to_string()];
        let kept = filter_by_domains(hits.clone(), &allowed, &[]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].url, "https://a.example.com/1");

        let blocked = vec!["example.com".to_string()];
        let kept = filter_by_domains(hits, &[], &blocked);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].url, "https://other.org/2");
    }

    #[test]
    fn render_search_results_link_map() {
        let hits = vec![
            SearchHit {
                title: "Rust".into(),
                url: "https://rust-lang.org/".into(),
                snippet: "lang".into(),
            },
            SearchHit {
                title: "Book".into(),
                url: "https://doc.rust-lang.org/book/".into(),
                snippet: String::new(),
            },
        ];
        let out = render_search_results("rust", &hits);
        assert!(out.starts_with("Web search results for query: \"rust\"\n"));
        assert!(out.contains("- [Rust](https://rust-lang.org/) — lang\n"));
        assert!(out.contains("- [Book](https://doc.rust-lang.org/book/)\n"));
    }

    #[test]
    fn render_search_results_no_results() {
        let out = render_search_results("nothing", &[]);
        assert!(out.contains("No results."));
    }

    #[test]
    fn render_search_results_caps_links() {
        let hits: Vec<SearchHit> = (0..30)
            .map(|n| SearchHit {
                title: format!("t{n}"),
                url: format!("https://e{n}.com/"),
                snippet: String::new(),
            })
            .collect();
        let out = render_search_results("q", &hits);
        assert_eq!(out.matches("\n- [").count(), SEARCH_MAX_LINKS);
    }

    #[test]
    fn ddg_challenge_detected() {
        assert!(is_ddg_challenge(
            "<form id=\"challenge-form\"><div class=\"anomaly-modal__mask\"></div></form>"
        ));
        assert!(!is_ddg_challenge(DDG_FIXTURE));
    }

    // Domain filtering is a property of the Rust fetch path (obscura or
    // curl); on ds4_engine builds without obscura google_search goes through
    // the C browser instead.
    #[cfg(any(not(ds4_engine), feature = "use_obscura"))]
    #[test]
    fn google_search_rejects_both_domain_lists() {
        let (mut ctx, dir) = test_ctx();
        let out = dispatch(
            &test_call(
                "google_search",
                &[
                    ("query", "rust"),
                    ("allowed_domains", "a.com"),
                    ("blocked_domains", "b.com"),
                ],
            ),
            &mut ctx,
        );
        assert_eq!(
            out.output,
            "Tool error: google_search failed: allowed_domains and blocked_domains are mutually exclusive\n"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

# WebSearch Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a client-side `web_search` tool that queries DuckDuckGo's HTML endpoint via curl and returns a plank-style link map, with domain filtering and no Anthropic round-trip.

**Architecture:** New pure parsing functions plus a `tool_web_search` entry point in `src/tools/web.rs`, reusing existing helpers (`curl_fetch`, `url_encode`, `tokenize`, `decode_entities`, `clean`, `esc_link_text`, `slice_chars`, `attr_value`). Registered in `src/tools/mod.rs` dispatch. Advertised only in the editable `provider_system_prompt` Web section — the DS4 trained prompt and `c_parity` fixtures are NOT touched (no C counterpart).

**Tech Stack:** Rust, std-only (no HTTP crate), curl subprocess.

## Global Constraints

- macOS-only for real inference; tests are pure logic (no network).
- `cargo clippy --workspace --all-targets -- -D warnings` must pass (pedantic/perf lints on).
- Do NOT modify `refs/ds4`, the DS4 trained system prompt, or `tests/c_parity.rs` fixtures.
- Model-facing text: never use `\`-continued Rust string literals with indentation.

---

### Task 1: DuckDuckGo redirect + percent decoding

**Files:**
- Modify: `src/tools/web.rs`

**Interfaces:**
- Produces: `fn percent_decode(s: &str) -> String`, `fn decode_ddg_href(href: &str) -> String`

- [ ] **Step 1: Write failing tests** (add to `#[cfg(test)] mod tests` in `web.rs`)

```rust
#[test]
fn percent_decode_basic() {
    assert_eq!(percent_decode("a%20b%2Fc"), "a b/c");
    assert_eq!(percent_decode("x+y"), "x y");
    assert_eq!(percent_decode("bad%2"), "bad%2");
}

#[test]
fn decode_ddg_href_unwraps_uddg() {
    let h = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp%3Fa%3D1&rut=xyz";
    assert_eq!(decode_ddg_href(h), "https://example.com/p?a=1");
}

#[test]
fn decode_ddg_href_passthrough_and_scheme() {
    assert_eq!(decode_ddg_href("https://direct.example/x"), "https://direct.example/x");
    assert_eq!(decode_ddg_href("//host.example/y"), "https://host.example/y");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib web::tests::percent_decode_basic web::tests::decode_ddg_href`
Expected: FAIL (functions not found).

- [ ] **Step 3: Implement** (add near `url_encode` in `web.rs`)

```rust
/// Percent-decodes a query-string value (`%XX` and `+` → space).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                let hi = (b[i + 1] as char).to_digit(16);
                let lo = (b[i + 2] as char).to_digit(16);
                if let (Some(h), Some(l)) = (hi, lo) {
                    out.push(((h * 16 + l) as u8) as char);
                    i += 3;
                } else {
                    out.push('%');
                    i += 1;
                }
            }
            c => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Resolves a DuckDuckGo result href to the real destination URL.
///
/// DDG wraps hits as `//duckduckgo.com/l/?uddg=<encoded url>&...`; this decodes
/// the `uddg` param. Scheme-relative `//host/...` becomes `https://host/...`.
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
```

Note: `percent_decode` produces bytes decoded as Latin-1 chars; DDG uddg values are ASCII URLs so this is safe. If multi-byte matters later, revisit.

- [ ] **Step 4: Run to verify pass** — `cargo test --lib web::tests::percent_decode_basic web::tests::decode_ddg_href` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/web.rs
git commit -m "feat(web): DuckDuckGo href + percent decoding for web_search"
```

---

### Task 2: Parse DuckDuckGo HTML results

**Files:**
- Modify: `src/tools/web.rs`

**Interfaces:**
- Consumes: `decode_ddg_href` (Task 1), `tokenize`, `decode_entities`, `clean`, `attr_value`.
- Produces: `struct SearchHit { title: String, url: String, snippet: String }`, `fn parse_ddg_results(html: &str) -> Vec<SearchHit>`

- [ ] **Step 1: Write failing test**

```rust
const DDG_FIXTURE: &str = r#"<html><body>
<div class="result results_links">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F&rut=1">The <b>Rust</b> Language</a>
  <a class="result__snippet" href="#">A language empowering everyone.</a>
</div>
<div class="result results_links">
  <a class="result__a" href="https://doc.rust-lang.org/book/">The Rust Book</a>
</div>
<a class="result__a" href="">   </a>
</body></html>"#;

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
```

- [ ] **Step 2: Run to verify failure** — `cargo test --lib web::tests::parse_ddg_results_extracts_hits` → FAIL

- [ ] **Step 3: Implement**

```rust
/// One parsed DuckDuckGo result.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

/// True if a raw tag's `class` attribute contains `needle` as a token.
fn class_contains(attrs: &str, needle: &str) -> bool {
    attr_value(attrs, "class")
        .map(|c| c.split_whitespace().any(|t| t == needle))
        .unwrap_or(false)
}

/// Extracts result hits from a DuckDuckGo HTML results page.
///
/// Walks tokens: an `<a class="result__a">` opens a hit (href → real URL via
/// `decode_ddg_href`, inner text → title); a following `<a class="result__snippet">`
/// before the next result attaches the snippet. Hits with an empty title or URL
/// are dropped.
fn parse_ddg_results(html: &str) -> Vec<SearchHit> {
    let toks = tokenize(html);
    let mut hits: Vec<SearchHit> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let Tok::Tag { name, attrs, closing: false } = &toks[i] {
            if name == "a" && class_contains(attrs, "result__a") {
                let url = attr_value(attrs, "href")
                    .map(|h| decode_ddg_href(&h))
                    .unwrap_or_default();
                let title = clean(decode_entities(collect_text_until_close(&toks, i + 1, "a")));
                i += 1;
                if !title.is_empty() && !url.is_empty() {
                    hits.push(SearchHit { title, url, snippet: String::new() });
                }
                continue;
            }
            if name == "a" && class_contains(attrs, "result__snippet") {
                let snip = clean(decode_entities(collect_text_until_close(&toks, i + 1, "a")));
                if let Some(last) = hits.last_mut() {
                    if last.snippet.is_empty() {
                        last.snippet = snip;
                    }
                }
                i += 1;
                continue;
            }
        }
        i += 1;
    }
    hits
}

/// Concatenates text tokens from `start` until the matching close tag `name`.
fn collect_text_until_close(toks: &[Tok<'_>], start: usize, name: &str) -> String {
    let mut out = String::new();
    for tok in &toks[start..] {
        match tok {
            Tok::Text(t) => out.push_str(t),
            Tok::Tag { name: n, closing: true, .. } if n == name => break,
            _ => {}
        }
    }
    out
}
```

- [ ] **Step 4: Run to verify pass** — PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/web.rs
git commit -m "feat(web): parse DuckDuckGo HTML results into SearchHit"
```

---

### Task 3: Domain filtering

**Files:**
- Modify: `src/tools/web.rs`

**Interfaces:**
- Consumes: `SearchHit` (Task 2).
- Produces: `fn host_of(url: &str) -> Option<String>`, `fn filter_by_domains(hits: Vec<SearchHit>, allowed: &[String], blocked: &[String]) -> Vec<SearchHit>`

- [ ] **Step 1: Write failing test**

```rust
fn hit(url: &str) -> SearchHit {
    SearchHit { title: "t".into(), url: url.into(), snippet: String::new() }
}

#[test]
fn host_of_extracts_host() {
    assert_eq!(host_of("https://a.example.com/p?x=1"), Some("a.example.com".to_string()));
    assert_eq!(host_of("http://Example.COM"), Some("example.com".to_string()));
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
```

- [ ] **Step 2: Run to verify failure** — FAIL

- [ ] **Step 3: Implement**

```rust
/// Lowercased host of an `http(s)` URL, or `None` if not parseable.
fn host_of(url: &str) -> Option<String> {
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
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
fn host_in_domain(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

/// Keeps hits per allow/block lists. `allowed` (if non-empty) is a whitelist;
/// `blocked` is a blacklist. Callers guarantee at most one is non-empty.
fn filter_by_domains(hits: Vec<SearchHit>, allowed: &[String], blocked: &[String]) -> Vec<SearchHit> {
    hits.into_iter()
        .filter(|h| {
            let Some(host) = host_of(&h.url) else { return false };
            if !allowed.is_empty() {
                return allowed.iter().any(|d| host_in_domain(&host, d));
            }
            !blocked.iter().any(|d| host_in_domain(&host, d))
        })
        .collect()
}
```

- [ ] **Step 4: Run to verify pass** — PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/web.rs
git commit -m "feat(web): domain allow/block filtering for web_search"
```

---

### Task 4: Render link map

**Files:**
- Modify: `src/tools/web.rs`

**Interfaces:**
- Consumes: `SearchHit` (Task 2), `esc_link_text`, `slice_chars`.
- Produces: `fn render_web_search(query: &str, hits: &[SearchHit]) -> String`; consts `WEB_SEARCH_MAX_LINKS`, `WEB_SEARCH_TITLE_CAP`.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn render_web_search_link_map() {
    let hits = vec![
        SearchHit { title: "Rust".into(), url: "https://rust-lang.org/".into(), snippet: "lang".into() },
        SearchHit { title: "Book".into(), url: "https://doc.rust-lang.org/book/".into(), snippet: String::new() },
    ];
    let out = render_web_search("rust", &hits);
    assert!(out.starts_with("Web search results for query: \"rust\"\n"));
    assert!(out.contains("- [Rust](https://rust-lang.org/) — lang\n"));
    assert!(out.contains("- [Book](https://doc.rust-lang.org/book/)\n"));
}

#[test]
fn render_web_search_no_results() {
    let out = render_web_search("nothing", &[]);
    assert!(out.contains("No results."));
}

#[test]
fn render_web_search_caps_links() {
    let hits: Vec<SearchHit> = (0..30)
        .map(|n| SearchHit { title: format!("t{n}"), url: format!("https://e{n}.com/"), snippet: String::new() })
        .collect();
    let out = render_web_search("q", &hits);
    assert_eq!(out.matches("\n- [").count(), WEB_SEARCH_MAX_LINKS);
}
```

- [ ] **Step 2: Run to verify failure** — FAIL

- [ ] **Step 3: Implement**

```rust
/// Max result links rendered by `web_search` (mirrors the search extractor cap).
const WEB_SEARCH_MAX_LINKS: usize = 20;
/// Max title characters rendered per link.
const WEB_SEARCH_TITLE_CAP: usize = 180;

/// Renders hits as a plank-style link map with a query header.
fn render_web_search(query: &str, hits: &[SearchHit]) -> String {
    let mut out = format!("Web search results for query: \"{query}\"\n\n");
    if hits.is_empty() {
        out.push_str("No results.\n");
        return out;
    }
    for h in hits.iter().take(WEB_SEARCH_MAX_LINKS) {
        let title = slice_chars(&esc_link_text(&h.title), WEB_SEARCH_TITLE_CAP);
        if h.snippet.is_empty() {
            let _ = writeln!(out, "- [{title}]({})", h.url);
        } else {
            let snip = esc_link_text(&h.snippet);
            let _ = writeln!(out, "- [{title}]({}) — {snip}", h.url);
        }
    }
    out
}
```

(`use std::fmt::Write as _;` is already imported at the top of `web.rs`.)

- [ ] **Step 4: Run to verify pass** — PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/web.rs
git commit -m "feat(web): render web_search results as link map"
```

---

### Task 5: `tool_web_search` entry point + dispatch + prompt

**Files:**
- Modify: `src/tools/web.rs`
- Modify: `src/tools/mod.rs:300` (dispatch match, after `visit_page`)
- Modify: `src/sysprompt.rs:110` (provider Web section)

**Interfaces:**
- Consumes: all Task 1-4 functions, `curl_fetch`, `url_encode`.
- Produces: `pub fn tool_web_search(ctx: &mut ToolContext, call: &ToolCall) -> String`

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn tool_web_search_requires_query() {
    let mut ctx = crate::tools::ToolContext::new(".");
    let call = ToolCall { name: "web_search".into(), args: vec![], raw: String::new() };
    let out = tool_web_search(&mut ctx, &call);
    assert_eq!(out, "Tool error: web_search requires query\n");
}

#[test]
fn tool_web_search_rejects_both_domain_lists() {
    let mut ctx = crate::tools::ToolContext::new(".");
    let call = ToolCall {
        name: "web_search".into(),
        args: vec![
            crate::dsml::ToolArg { name: "query".into(), value: "rust".into() },
            crate::dsml::ToolArg { name: "allowed_domains".into(), value: "a.com".into() },
            crate::dsml::ToolArg { name: "blocked_domains".into(), value: "b.com".into() },
        ],
        raw: String::new(),
    };
    let out = tool_web_search(&mut ctx, &call);
    assert_eq!(
        out,
        "Tool error: web_search failed: allowed_domains and blocked_domains are mutually exclusive\n"
    );
}
```

Note: verify the exact field set of `ToolCall`/`ToolArg` in `src/dsml.rs` before writing (constructor fields must match — adjust `raw`/other fields if the struct differs).

- [ ] **Step 2: Run to verify failure** — FAIL

- [ ] **Step 3: Implement in `web.rs`**

```rust
/// Splits a comma-separated domain list arg into lowercased, non-empty hosts.
fn parse_domain_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Executes the `web_search` tool: client-side DuckDuckGo search, no browser.
///
/// Deviation from the browser web tools: no approval gate — this is a plain
/// curl fetch of the DuckDuckGo HTML endpoint, not a visible Chrome session.
pub fn tool_web_search(_ctx: &mut ToolContext, call: &ToolCall) -> String {
    let query = call.arg_value("query").unwrap_or("").trim();
    if query.is_empty() {
        return "Tool error: web_search requires query\n".to_string();
    }
    let allowed = parse_domain_list(call.arg_value("allowed_domains").unwrap_or(""));
    let blocked = parse_domain_list(call.arg_value("blocked_domains").unwrap_or(""));
    if !allowed.is_empty() && !blocked.is_empty() {
        return "Tool error: web_search failed: allowed_domains and blocked_domains are mutually exclusive\n"
            .to_string();
    }
    let url = format!("https://html.duckduckgo.com/html/?q={}", url_encode(query));
    let html = match curl_fetch(&url) {
        Ok(html) => html,
        Err(e) => return format!("Tool error: web_search failed: {e}\n"),
    };
    let hits = filter_by_domains(parse_ddg_results(&html), &allowed, &blocked);
    render_web_search(query, &hits)
}
```

- [ ] **Step 4: Register in `src/tools/mod.rs`** — add after the `visit_page` arm (line ~301):

```rust
        "web_search" => web::tool_web_search(ctx, call),
```

- [ ] **Step 5: Advertise in `src/sysprompt.rs`** — replace the Web section body (line ~110) so it reads:

```rust
"Use google_search to find web pages and visit_page to read a known URL. The first web call may \
ask permission to start a browser. Use web_search for a client-side search that returns a link map \
without starting a browser; it accepts optional allowed_domains or blocked_domains (comma-separated, \
not both).\n\n\
```

- [ ] **Step 6: Run tests** — `cargo test --lib web` → PASS; then `cargo test --lib sysprompt` → PASS (the provider-prompt test at `sysprompt.rs:637` asserts a substring that is unchanged; confirm it still passes, and update the `google_search` list assertion only if needed).

- [ ] **Step 7: Full gate** — `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --lib`.

- [ ] **Step 8: Commit**

```bash
git add src/tools/web.rs src/tools/mod.rs src/sysprompt.rs
git commit -m "feat(web): add client-side web_search tool (DuckDuckGo)"
```

---

## Self-Review

- **Spec coverage:** backend (Task 5 URL + Task 2 parse), uddg decode (Task 1), schema/errors (Task 5), domain filter (Task 3), output link map (Task 4), registration + prompt (Task 5), tests (all tasks). All covered.
- **Parity:** DS4 trained prompt and `c_parity` fixtures untouched — only the editable `provider_system_prompt` changes.
- **Type consistency:** `SearchHit`, `parse_ddg_results`, `filter_by_domains`, `render_web_search`, `tool_web_search` names consistent across tasks.
- **Placeholder scan:** none — all steps carry real code. The one flagged verification (ToolCall/ToolArg field names) is an explicit check, not a placeholder.

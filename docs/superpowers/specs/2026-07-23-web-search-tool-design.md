# WebSearch tool (client-side, DuckDuckGo) — Design

Date: 2026-07-23

## Goal

Add a `web_search` tool that performs web search entirely client-side, with no
call to Anthropic services. It complements the existing `google_search` and
`visit_page` tools by exposing a structured query interface (with domain
filtering) and returning a compact link map for the model.

"Offline" here means client-side: the tool talks directly to a search backend
over the network (DuckDuckGo), never routing through Anthropic's server-side
`web_search` capability.

## Backend

- Endpoint: `https://html.duckduckgo.com/html/?q=<url-encoded query>` — the
  no-JavaScript HTML results page. No API key, no account.
- Fetched with the existing `curl_fetch` helper (`curl -sL --max-time 20
  --compressed`), reusing the same size cap and user-agent as the other web
  tools.
- Result parsing over the tokenized HTML:
  - Each result's title + link comes from `a.result__a` (anchor with class
    containing `result__a`): the anchor text is the title, `href` is the link.
  - DuckDuckGo wraps result hrefs in a redirect of the form
    `//duckduckgo.com/l/?uddg=<url-encoded real URL>[&...]`. The tool decodes
    the `uddg` query parameter back to the real destination URL. Hrefs that are
    already absolute (`http`/`https`) are used as-is.
  - The optional snippet comes from `a.result__snippet` (or `.result__snippet`
    text), cleaned and appended after the title.

## Tool schema (DSML args)

- `query` (required). Empty/missing → `Tool error: web_search requires query`.
- `allowed_domains` (optional). Comma-separated host whitelist; a result is kept
  only if its host equals or is a subdomain of one of these.
- `blocked_domains` (optional). Comma-separated host blacklist; a result is
  dropped if its host equals or is a subdomain of one of these.
- Specifying both `allowed_domains` and `blocked_domains` →
  `Tool error: web_search failed: allowed_domains and blocked_domains are mutually exclusive`.

No approval gate: unlike `google_search`/`visit_page` (which mirror the C's
visible-Chrome startup confirmation), `web_search` is a plain curl fetch and
runs without the `web_confirm` prompt.

## Output (plank link-map style)

Mirrors the existing `google_search` Markdown conventions rather than the
CC "Links: [...]" / "Sources" format:

```
Web search results for query: "<query>"

- [<title>](<url>) — <snippet>
- [<title>](<url>)
...
```

- Titles/snippets escaped via `esc_link_text` / cleaned via `clean`, matching
  the existing extractors.
- Caps consistent with the search extractor: at most 20 links; title truncated
  to ~180 chars via `slice_chars`. Results with no usable link are skipped.
- On zero results after filtering: a header with an explicit "no results" line.
- Provider/transport failure → `Tool error: web_search failed: <e>`.

## Registration

- Add `tool_web_search` to `src/tools/web.rs`.
- Add `"web_search" => web::tool_web_search(ctx, call)` to the `dispatch` match
  in `src/tools/mod.rs`.
- Add `web_search` to the tool table / name list so the model is told the tool
  exists (same place `google_search` and `visit_page` are declared).

## Testing

All parsing is pure functions taking HTML as input (mirroring how the existing
`extract_search_markdown` / `extract_page_markdown` are tested), so tests need
no network:

- Extraction from a committed DuckDuckGo HTML fixture: correct title/url/snippet
  pairs.
- `uddg` redirect decoding back to the real URL (including extra params).
- Domain filtering: `allowed_domains` keeps only matching hosts (incl.
  subdomains); `blocked_domains` drops matching hosts.
- Both-domains-specified returns the mutually-exclusive error.
- Link cap (20) and title truncation enforced.
- Empty-query returns the `requires query` error.

## Non-goals

- No JS-rendered content (same limitation as the existing curl-based tools).
- No result caching or rate limiting.
- No API-key providers (Tavily/Brave) — DuckDuckGo HTML only.

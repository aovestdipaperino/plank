# Obscura vs Lightpanda

> **Snapshot taken 2026-10-29 22:22 CET.** Obscura facts come from the
> `refs/obscura` submodule pinned at `4854973` (2026-07-23); Lightpanda facts
> come from the `lightpanda-io/browser` README on `main` as of that date. Both
> projects move fast, so re-check before acting on anything below. Neither
> browser was benchmarked for this note.

## How plank uses obscura today

Plank's surface on obscura is small. `src/obscura_web.rs` (63 lines) keeps one
session-cached stealth `obscura::Browser`, opens a page, `goto`s the URL,
`settle`s for 2 s and returns `page.content()`. The resulting HTML then flows
into plank's own pure-Rust extraction in `src/tools/web.rs`, which produces the
model-visible output of `visit_page` and `google_search`. Obscura is statically
linked from the `refs/obscura` submodule behind the default `use_obscura`
feature; without it the transport falls back to curl (or, on `ds4_engine`
builds, the C engine's browser).

So plank needs exactly one capability from a browser: post-JavaScript HTML for
a URL. CDP, Puppeteer, MCP and Markdown output are all unused.

## Side by side

| | Obscura (in plank) | Lightpanda |
|---|---|---|
| Language | Rust | Zig (needs Zig 0.17) |
| JS engine | V8 via `deno_core`, single isolate | V8 |
| HTML parsing / networking | own DOM crate, `reqwest` or `wreq` (stealth) | html5ever, libcurl |
| Graphical rendering | none | none |
| Protocols | CDP, MCP | CDP, WebDriver BiDi, MCP (stdio or HTTP) |
| Embeddable as a library | yes, a Rust crate linked into plank | no library API; CLI or server only |
| Anti-detection | `--stealth`: real Chrome TLS fingerprint, tracker blocklist | none; adblocker and `--obey-robots` |
| Built-in output | HTML, Markdown extractor | `--dump html, markdown, png, pdf, structured_data` (png/pdf are text-only renderings) |
| Web API conformance | no published suite | Web Platform Tests results published |
| Robustness | V8 termination watchdogs, per-CDP-command timeout, `catch_unwind` around DOM ops | not documented in the README |
| License | Apache-2.0 | AGPL-3.0 |
| Maturity | tagged releases, ~10k stars | nightly builds only, ~37.6k stars |
| Self-reported perf | 30 MB, 85 ms page load | ~16x less memory, ~9x faster than Chrome over 933 pages |

The two performance rows are each project's own benchmark against Chrome, with
different methodologies, so they cannot be compared with each other.

## What switching would mean for plank

**License.** Plank is MIT. Lightpanda is AGPL-3.0 and ships no library, so the
only viable integration is spawning it as a separate process
(`lightpanda fetch --dump html <url>`, or `serve` plus a CDP client). That
keeps plank's license intact but gives up the single self-contained binary.

**Build weight.** Obscura's `deno_core` is plank's heaviest dependency. It is
what turns on `serde_json/preserve_order` for the whole build (see
`FINDINGS.md`) and why a bare `cargo fmt` drifts the submodule. Moving to
Lightpanda would drop that, in exchange for a runtime binary users must install.
Its prebuilt Linux binaries are glibc-only (no Alpine/musl, no Android).

**Bot walls.** Obscura's Chrome TLS fingerprint helps against the DuckDuckGo
challenge page that `is_ddg_challenge` already detects. Lightpanda has nothing
equivalent, so `google_search` would likely be blocked more often.

**Page fidelity.** Lightpanda probably wins here: html5ever, libcurl and a WPT
conformance harness, versus obscura's hand-written DOM and `bootstrap.js`.

**Markdown output.** Not useful to plank. The `visit_page` format is defined by
plank's own extractor and must stay stable.

## Recommendation

Keep obscura as the embedded transport. If specific sites come back with thin
or broken HTML, add Lightpanda as an optional out-of-process fallback (for
example a `web.transport = "lightpanda"` setting used when the binary is on
`PATH`) rather than a replacement. Before that, bump `refs/obscura`: the pin was
three months old at the time of this note, which is the cheaper fix to try
first.

# Obscura vs Lightpanda vs Cloudflare Browser Run

> **Snapshot taken 2026-10-29 22:22 CET; Browser Run added 2026-10-29 22:26
> CET.** Obscura facts come from the `refs/obscura` submodule pinned at
> `4854973` (2026-07-23). Lightpanda facts come from the `lightpanda-io/browser`
> README on `main` as of that date. Browser Run facts come from
> cloudflare.com/products/browser-run and the developers.cloudflare.com
> `browser-rendering` docs (overview, REST API, pricing, limits, automatic
> request headers) as of that date. All three move fast, so re-check before
> acting on anything below. None of them was benchmarked for this note.

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

## The three contenders

**Obscura** is an embeddable Rust headless browser (V8 through `deno_core`,
its own DOM) that plank links in as a crate.

**Lightpanda** is a headless browser written from scratch in Zig (V8,
html5ever, libcurl), shipped as a CLI and CDP/MCP server.

**Cloudflare Browser Run** (formerly Browser Rendering) is not a browser you
run but a hosted service: headless Chrome on Cloudflare's network, driven either
by stateless "Quick Actions" REST calls or by full browser sessions over
Puppeteer, Playwright, CDP or Stagehand. Its `/content` endpoint returns the
rendered HTML of a URL, which is exactly the one thing plank needs.

## Side by side

| | Obscura (in plank) | Lightpanda | Browser Run |
|---|---|---|---|
| What it is | Rust library | Zig binary | hosted SaaS |
| Engine | own DOM + V8 (`deno_core`), single isolate | own DOM + V8, html5ever | real headless Chrome |
| Networking | `reqwest`, or `wreq` in stealth mode | libcurl | Cloudflare's network |
| Graphical rendering | none | none (png/pdf dumps are text-only) | full (real screenshots and PDFs) |
| Integration | in-process call | spawn a process or speak CDP | HTTPS + API token (`Browser Run - Edit`), or a Workers binding |
| Protocols | CDP, MCP | CDP, WebDriver BiDi, MCP | REST Quick Actions, Puppeteer, Playwright, CDP, Stagehand, Playwright MCP |
| Ready-made outputs | HTML, Markdown | html, markdown, png, pdf, structured data | `/content`, `/markdown`, `/screenshot`, `/pdf`, `/snapshot`, `/accessibilityTree`, `/scrape`, `/json` (AI), `/links`, `/crawl` |
| Bot posture | stealth: real Chrome TLS fingerprint, tracker blocklist | neutral; adblocker, `--obey-robots` | self-identifying: non-removable `cf-brapi-*` / `cf-biso-devtools` headers plus Web Bot Auth signatures |
| Page fidelity | lowest (hand-written DOM and `bootstrap.js`) | middle (WPT-tracked) | highest (it is Chrome) |
| Reaches localhost / intranet | yes | yes | no |
| Privacy | URLs stay on the machine | URLs stay on the machine | every visited URL goes to Cloudflare |
| Works offline / no account | yes | yes | no |
| Cost | free | free | Free plan: 10 browser-minutes/day, Quick Actions 1 request every 10 s. Workers Paid: 10 browser-hours/month included, then $0.09/hour, Quick Actions 30 requests/s |
| License | Apache-2.0 | AGPL-3.0 | proprietary service |
| Maturity | tagged releases, ~10k stars | nightly builds only, ~37.6k stars | GA commercial product |
| Self-reported perf | 30 MB, 85 ms page load | ~16x less memory, ~9x faster than Chrome over 933 pages | "low cold-start time", "thousands of browsers" |

The two open-source performance rows are each project's own benchmark against
Chrome, with different methodologies, so they cannot be compared with each
other. Browser Run publishes no comparable figure; its cost is wall-clock
browser time billed per millisecond (reported in the `X-Browser-Ms-Used`
response header), and failed `waitForTimeout` requests are not billed.

## What switching would mean for plank

**License.** Plank is MIT. Lightpanda is AGPL-3.0 and ships no library, so the
only viable integration is spawning it as a separate process
(`lightpanda fetch --dump html <url>`, or `serve` plus a CDP client). That
keeps plank's license intact but gives up the single self-contained binary.
Browser Run raises no license question at all, since plank would only make
HTTPS calls.

**Build weight.** Obscura's `deno_core` is plank's heaviest dependency. It is
what turns on `serde_json/preserve_order` for the whole build (see
`FINDINGS.md`) and why a bare `cargo fmt` drifts the submodule. Moving to
Lightpanda would drop that, in exchange for a runtime binary users must install.
Its prebuilt Linux binaries are glibc-only (no Alpine/musl, no Android).
Browser Run adds nothing to the build: a `/content` call is a single POST that
the existing curl path in `src/tools/web.rs` could make.

**Bot walls.** Obscura's Chrome TLS fingerprint helps against the DuckDuckGo
challenge page that `is_ddg_challenge` already detects. Lightpanda has nothing
equivalent, so `google_search` would likely be blocked more often. Browser Run
goes the other way on purpose: it announces itself with headers that cannot be
removed and signs requests with Web Bot Auth, and Cloudflare publishes bot
detection IDs so site owners can recognise it. Sites that welcome verified
bots will let it through; sites that block automation will block it reliably.
It is the honest option, not the one that gets past challenges.

**Page fidelity.** Browser Run wins outright, since it is real Chrome with real
layout. Lightpanda is likely second (html5ever, libcurl, a WPT conformance
harness), and obscura's hand-written DOM is last.

**Locality and privacy.** Obscura and Lightpanda run on the user's machine, so
they can open `localhost` dev servers and intranet pages, work without an
account, and never disclose what the agent browses. Browser Run can do none of
those: every URL the model visits is sent to Cloudflare, and plank would need
a Cloudflare account and an API token in the environment (the pattern
`agents.rs` already uses for remote providers via `api_key_env`).

**Rate limits and state.** On the Free plan, Quick Actions allow one request
every 10 seconds and 10 browser-minutes a day, which a research-heavy turn can
exhaust quickly. Quick Actions are stateless too, so obscura's session-cached
cookie jar has no equivalent unless plank moves to full browser sessions.

**Markdown output.** Lightpanda's `--dump markdown` and Browser Run's
`/markdown` are both tempting and both unusable for `visit_page`: its format is
defined by plank's own extractor and must stay stable.

## Recommendation

Keep obscura as the embedded default: it is local, private, free, needs no
account, and its stealth mode is what keeps `google_search` working.

If fidelity problems show up on specific sites, Browser Run is now the more
attractive fallback than Lightpanda: real Chrome, no license issue, no binary
to install, and `/content` drops straight into `fetch_html`. Make it opt-in
(for example a `web.transport = "cloudflare"` setting enabled only when an
account ID and API token are configured), keep it off for `localhost` and
private addresses, and say clearly in the docs that visited URLs leave the
machine. Lightpanda stays the choice for users who want better fidelity
without sending URLs to a third party.

Before any of that, bump `refs/obscura`: the pin was three months old at the
time of this note, which is the cheaper fix to try first.

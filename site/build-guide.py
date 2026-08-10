#!/usr/bin/env python3
"""Render `user-guide/*.md` into `site/guide/` as pages matching plank-agent.dev.

Lives beside the site it builds rather than in `scripts/`, which is gitignored —
the generator has to ship with its output or the next person regenerates by hand.

The guide is written once, as markdown, and read in two places: on GitHub and on
the site. This turns the second into a build step rather than a second copy —
editing a guide page and forgetting its HTML twin is the failure mode worth
designing out.

Markdown conversion is pandoc's, not a hand-rolled parser: the guide uses
tables, nested lists, and fenced blocks, and a homemade converter would get one
of them subtly wrong. Requires `pandoc` on PATH.

    python3 site/build-guide.py

Idempotent; safe to re-run. Output is checked in, so the site can be deployed
without a toolchain.
"""

import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "user-guide"
OUT = ROOT / "site" / "guide"
REPO = "https://github.com/aovestdipaperino/plank"

# Design tokens lifted from site/index.html so the guide cannot drift from the
# landing page. Extracted rather than duplicated: one edit to the palette there
# reaches here on the next build.
PALETTE = re.compile(r":root\{.*?\}|@media \(prefers-color-scheme:dark\)\{.*?\}\s*\}|:root\[data-theme=\"(?:light|dark)\"\]\{.*?\}", re.S)


def site_palette() -> str:
    css = (ROOT / "site" / "index.html").read_text()
    style = re.search(r"<style>(.*?)</style>", css, re.S).group(1)
    return "\n".join(m.group(0) for m in PALETTE.finditer(style))


PAGE_CSS = """
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%; scroll-behavior:smooth}
  body{
    margin:0; background:var(--birch); color:var(--bark);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    line-height:1.65; font-size:17px; -webkit-font-smoothing:antialiased;
  }
  .rounded{font-family:ui-rounded,"SF Pro Rounded","Hiragino Maru Gothic ProN","Segoe UI",system-ui,sans-serif}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  a{color:var(--pine);text-underline-offset:3px}
  .wrap{max-width:880px;margin:0 auto;padding:0 22px}

  header{padding:22px 0 8px}
  .bar{display:flex;align-items:center;justify-content:space-between;gap:16px}
  .brand{display:flex;align-items:center;gap:9px;font-weight:800;font-size:20px;letter-spacing:-.01em;
         color:var(--bark);text-decoration:none}
  .brand .chip{font-size:22px;filter:saturate(1.1)}
  .ghost{font-size:14px;font-weight:700;text-decoration:none;color:var(--bark-soft);
    border:2px solid var(--line);border-radius:999px;padding:7px 14px;
    transition:border-color .15s,color .15s,transform .1s;white-space:nowrap}
  .ghost:hover{color:var(--bark);border-color:var(--honey);transform:translateY(-1px)}
  .navlinks{display:flex;gap:10px;align-items:center}

  .crumb{font-size:14px;color:var(--bark-soft);margin:18px 0 0}
  .crumb a{color:var(--bark-soft)}

  article{background:var(--paper);border:2px solid var(--line);border-radius:18px;
    padding:34px 34px 26px;margin:14px 0 26px;
    box-shadow:0 10px 30px rgb(var(--shadow) / .06)}
  article h1{font-family:ui-rounded,"SF Pro Rounded","Segoe UI",system-ui,sans-serif;
    font-size:34px;line-height:1.15;letter-spacing:-.02em;margin:.1em 0 .5em}
  article h2{font-size:24px;margin:1.6em 0 .5em;letter-spacing:-.01em;
    padding-top:.5em;border-top:2px solid var(--line)}
  article h2:first-of-type{border-top:0;padding-top:0}
  article h3{font-size:19px;margin:1.4em 0 .4em}
  article p{margin:.7em 0}
  article ul,article ol{padding-left:1.3em}
  article li{margin:.3em 0}
  article strong{color:var(--bark)}
  article blockquote{margin:1em 0;padding:.6em 1em;border-left:4px solid var(--honey);
    background:var(--birch);border-radius:0 10px 10px 0;color:var(--bark-soft)}
  article blockquote p{margin:.2em 0}
  code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.9em;
    background:var(--birch);border:1px solid var(--line);border-radius:6px;padding:.08em .35em}
  pre{background:var(--birch);border:2px solid var(--line);border-radius:12px;
    padding:14px 16px;overflow-x:auto;margin:1em 0}
  pre code{background:none;border:0;padding:0;font-size:.86em;line-height:1.55}
  table{border-collapse:collapse;width:100%;margin:1em 0;display:block;overflow-x:auto}
  th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
  th{background:var(--birch);font-weight:800}
  hr{border:0;border-top:2px solid var(--line);margin:1.6em 0}
  img{max-width:100%}

  .toc{list-style:none;padding:0;margin:0}
  .toc li{margin:0 0 10px}
  .toc a{display:block;text-decoration:none;color:var(--bark);
    border:2px solid var(--line);border-radius:14px;padding:12px 15px;background:var(--paper);
    transition:border-color .15s,transform .1s}
  .toc a:hover{border-color:var(--honey);transform:translateY(-1px)}
  .toc .n{font-weight:800;color:var(--honey-ink);margin-right:.5em}
  .toc .d{display:block;color:var(--bark-soft);font-size:15px;margin-top:2px}

  .pager{display:flex;justify-content:space-between;gap:12px;margin:0 0 34px;flex-wrap:wrap}
  .pager a{font-size:15px;font-weight:700;text-decoration:none;color:var(--bark-soft);
    border:2px solid var(--line);border-radius:999px;padding:9px 16px;background:var(--paper)}
  .pager a:hover{color:var(--bark);border-color:var(--honey)}

  footer{border-top:2px solid var(--line);padding:20px 0 40px;color:var(--bark-soft);font-size:14px}
  .footgrid{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center}
  .fine{font-size:12.5px;opacity:.75;margin-top:10px}

  @media (max-width:640px){
    article{padding:24px 20px 18px;border-radius:14px}
    article h1{font-size:27px}
    body{font-size:16px}
  }
"""

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%AA%B5%3C/text%3E%3C/svg%3E">
<style>
{palette}
{css}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="bar">
      <a class="brand rounded" href="/"><span class="chip">🪵</span> plank</a>
      <div class="navlinks">
        <a class="ghost" href="/whats-new">What's new</a>
        <a class="ghost" href="/guide/">Guide</a>
        <a class="ghost" href="{repo}" target="_blank" rel="noopener">GitHub ↗</a>
      </div>
    </div>
  </header>
"""

FOOT = """
  <footer>
    <div class="footgrid">
      <a class="brand rounded" href="/"><span class="chip">🪵</span> plank</a>
      <div>
        <a href="/guide/">User guide</a> ·
        <a href="{repo}" target="_blank" rel="noopener">GitHub</a> ·
        <a href="{repo}/releases" target="_blank" rel="noopener">Releases</a>
      </div>
    </div>
    <p class="fine">macOS only — Plank is a Mac guy (it needs Metal). This guide is generated from
    <a href="{repo}/tree/main/user-guide">user-guide/</a> in the repo, so it cannot drift from the source.</p>
  </footer>
</div>
</body>
</html>
"""


def pandoc(md: str) -> str:
    return subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "html5", "--no-highlight"],
        input=md, text=True, capture_output=True, check=True,
    ).stdout


def fix_links(body: str) -> str:
    """Point intra-guide links at the built pages and repo links at GitHub."""
    # Sibling guide pages: `05-tools.md#frag` -> `05-tools.html#frag`.
    body = re.sub(r'href="(\d\d-[a-z0-9-]+)\.md(#[^"]*)?"', r'href="\1.html\2"', body)
    body = body.replace('href="README.md"', 'href="./"')
    # Repo-relative paths only exist on GitHub; send them there.
    body = re.sub(r'href="\.\./([^"]+)"', rf'href="{REPO}/blob/main/\1"', body)
    return body


def inline_md(text: str) -> str:
    """Render the inline markdown the contents list uses: `code` only.

    The chapter blurbs come from the README's contents list, which is markdown —
    rendering them as plain text leaves visible backticks on the card.
    """
    parts = html.escape(text).split("`")
    return "".join(p if i % 2 == 0 else f"<code>{p}</code>" for i, p in enumerate(parts))


def page(title: str, desc: str, body: str) -> str:
    return (
        HEAD.format(title=html.escape(title), desc=html.escape(desc),
                    palette=site_palette(), css=PAGE_CSS, repo=REPO)
        + body
        + FOOT.format(repo=REPO)
    )


def build_whats_new() -> bool:
    """Render `site/whats-new.md` into `site/whats-new.html`.

    Kept out of `user-guide/` on purpose: the guide is numbered chapters with a
    prev/next pager and a contents list, and a release-notes page is none of
    those. It still goes through [`page`] so it shares the landing page's
    palette and chrome instead of carrying a second copy of them.
    """
    src = ROOT / "site" / "whats-new.md"
    if not src.is_file():
        return False
    md = src.read_text()
    title = re.search(r"^#\s+(.+)$", md, re.M).group(1).strip()
    body = (
        f'  <p class="crumb"><a href="/">plank</a> › {html.escape(title)}</p>\n'
        f"  <article>\n{fix_links(pandoc(md))}\n  </article>\n"
    )
    (ROOT / "site" / "whats-new.html").write_text(page(
        f"{title} — plank",
        "The features worth knowing about since plank-agent.dev went up: the KV cache "
        "browser, cross-engine sub-agents, remote control, reasoning levels, and more.",
        body))
    return True


def main() -> int:
    if not GUIDE.is_dir():
        print("no user-guide/ next to this script", file=sys.stderr)
        return 1
    chapters = sorted(p for p in GUIDE.glob("*.md") if p.name != "README.md")
    OUT.mkdir(parents=True, exist_ok=True)

    # Chapter titles and one-line descriptions come from the README's contents
    # list, so the index here and the one on GitHub cannot disagree.
    readme = (GUIDE / "README.md").read_text()
    entries = {}
    for m in re.finditer(r"^(\d+)\. \[([^\]]+)\]\((\d\d-[a-z0-9-]+)\.md\) — (.+)$", readme, re.M):
        entries[m.group(3)] = (m.group(1), m.group(2), m.group(4))

    built = []
    for i, src in enumerate(chapters):
        slug = src.stem
        md = src.read_text()
        # Each chapter opens with its own prev/index/next line, for reading the
        # files on GitHub. The web build has a breadcrumb and a pager already,
        # so drop it rather than show two of them.
        md = re.sub(r"\A\[[^\]]*\]\(\d\d-[a-z0-9-]+\.md\)[^\n]*\n\s*", "", md)
        md = re.sub(r"\A\[Index\]\(README\.md\)[^\n]*\n\s*", "", md)
        # The `# N. Title` heading carries the number already.
        title = re.search(r"^#\s+(.+)$", md, re.M).group(1).strip()
        num, short, desc = entries.get(slug, ("", title, ""))
        body_html = fix_links(pandoc(md))
        prev_p = f'<a href="{chapters[i-1].stem}.html">← {entries.get(chapters[i-1].stem, ("","",""))[1] or "Previous"}</a>' if i else '<a href="./">← Guide index</a>'
        next_p = f'<a href="{chapters[i+1].stem}.html">{entries.get(chapters[i+1].stem, ("","",""))[1] or "Next"} →</a>' if i + 1 < len(chapters) else '<a href="/free-tokens">A word from our sponsor →</a>'
        body = (
            f'  <p class="crumb"><a href="/">plank</a> › <a href="./">User guide</a> › {html.escape(short)}</p>\n'
            f"  <article>\n{body_html}\n  </article>\n"
            f'  <nav class="pager">{prev_p}{next_p}</nav>\n'
        )
        (OUT / f"{slug}.html").write_text(page(f"{title} — plank user guide",
                                               desc.replace("`", "") or f"{title}, from the plank user guide.", body))
        built.append((slug, num, short, desc))

    # Index: the README's prose, then the contents as cards.
    intro = readme.split("## Contents")[0]
    tail = readme.split("## The five-minute version")
    five = "## The five-minute version" + tail[1] if len(tail) > 1 else ""
    cards = "\n".join(
        f'    <li><a href="{slug}.html"><span class="n">{num}.</span>{html.escape(short)}'
        f'<span class="d">{inline_md(desc)}</span></a></li>'
        for slug, num, short, desc in built
    )
    body = (
        f'  <p class="crumb"><a href="/">plank</a> › User guide</p>\n'
        f"  <article>\n{fix_links(pandoc(intro))}\n"
        f"    <h2>Contents</h2>\n    <ul class=\"toc\">\n{cards}\n    </ul>\n"
        f"{fix_links(pandoc(five))}\n  </article>\n"
    )
    (OUT / "index.html").write_text(page(
        "plank user guide",
        "Using plank: the TUI, slash commands, tools, sessions, context, configuration, and more.",
        body))
    print(f"built {len(built) + 1} pages into {OUT.relative_to(ROOT)}")
    if build_whats_new():
        print("built site/whats-new.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

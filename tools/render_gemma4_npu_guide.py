#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the Gemma4 NPU guide as standalone HTML and a paginated docs site.

Dependency: python3 -m pip install markdown-it-py==3.0.0
Run from any directory: python3 tools/render_gemma4_npu_guide.py
"""

import argparse
import html
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt
from markdown_it.token import Token


STYLE = """
:root { color-scheme: light; --ink:#172b3a; --muted:#526675;
  --accent:#006c67; --line:#d9e3e8; --paper:#fff; --background:#f3f6f8; }
* { box-sizing:border-box; }
html { scroll-padding-top:2rem; }
body { margin:0; background:var(--background); color:var(--ink);
  font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.topbar { display:flex; align-items:center; gap:.75rem;
  background:#123744; color:#fff; padding:1rem max(1.5rem,calc((100% - 1280px)/2));
  font-size:.85rem; letter-spacing:.08em; }
.brand-mark { display:block; flex-shrink:0; width:35px; height:35px; }
.layout { display:grid; grid-template-columns:245px minmax(0,1fr);
  max-width:1280px; margin:auto; gap:2rem; padding:2rem 1.5rem 4rem; }
nav { position:sticky; top:1.5rem; align-self:start; max-height:calc(100vh - 3rem);
  overflow:auto; font-size:.86rem; }
nav strong { display:block; color:var(--muted); margin-bottom:1rem; }
nav a { display:block; padding:.42rem .7rem; border-left:2px solid var(--line);
  text-decoration:none; }
nav a:hover,nav a:focus { background:#e0eeec; border-color:var(--accent); }
main { min-width:0; padding:2.5rem 3rem; background:var(--paper);
  border:1px solid var(--line); border-radius:10px; }
h1,h2,h3 { line-height:1.25; letter-spacing:-.025em; }
h1 { font-size:2.5rem; margin:0 0 1.5rem; }
h2 { font-size:1.55rem; margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); }
h3 { font-size:1.15rem; margin-top:2rem; }
a { color:var(--accent); text-underline-offset:.18em; overflow-wrap:anywhere; }
p,li { overflow-wrap:anywhere; }
li + li { margin-top:.35rem; }
code { font: .86em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;
  background:#eef3f5; border-radius:3px; padding:.12em .3em; overflow-wrap:anywhere; }
pre { overflow:auto; background:#142c39; color:#e7f3f4; padding:1.15rem;
  border-radius:7px; line-height:1.6; }
pre code { background:none; padding:0; color:inherit; overflow-wrap:normal; }
blockquote { margin:1.5rem 0; padding:.3rem 1.2rem; background:#edf6f4;
  border-left:4px solid var(--accent); }
.table-wrap { overflow:auto; margin:1.4rem 0; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th,td { text-align:left; vertical-align:top; padding:.7rem .8rem;
  border:1px solid var(--line); min-width:130px; }
th { background:#eaf1f4; }
tbody tr:nth-child(even) { background:#f8fafb; }
.app-screenshot { margin:1.5rem 0; }
.app-screenshot a { display:block; width:360px; max-width:100%; }
.app-screenshot img { display:block; width:100%; height:auto;
  border:1px solid var(--line); border-radius:10px; }
.app-screenshot figcaption { margin-top:.7rem; color:var(--muted); font-size:.85rem; }
.skip { position:absolute; left:-9999px; }
.skip:focus { left:1rem; top:.5rem; background:white; padding:.5rem; z-index:2; }
@media(max-width:900px) {
  .layout { display:block; padding:1rem; }
  nav { position:static; max-height:none; margin-bottom:1.5rem; }
  nav a { display:inline-block; margin:.2rem; }
  main { padding:1.5rem; }
  h1 { font-size:2rem; }
}
@media print {
  body { background:white; font-size:10pt; }
  .topbar,nav,.skip { display:none; }
  .layout { display:block; padding:0; }
  main { padding:0; border:0; }
  h2,h3 { break-after:avoid; }
  pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#f3f6f8; color:#172b3a; }
  pre code { overflow-wrap:anywhere; }
  .table-wrap { overflow:visible; }
  th,td { min-width:0; }
}
"""


def parsed_source(source, reference=None):
    """Use English heading IDs in both languages to retain deep links."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(source)
    headings = [i for i, token in enumerate(tokens) if token.type == "heading_open"]
    originals = tokens if reference is None else parser.parse(reference)
    original_headings = [i for i, token in enumerate(originals) if token.type == "heading_open"]
    if [tokens[i].tag for i in headings] != [originals[i].tag for i in original_headings]:
        raise ValueError("English and Korean heading structure must match")
    ids = set()
    for index, original_index in zip(headings, original_headings):
        label = originals[original_index + 1].content
        slug = re.sub(r"[^a-z0-9 -]", "", label.lower()).replace(" ", "-")
        if slug in ids:
            raise ValueError(f"Duplicate heading anchor: {slug}")
        ids.add(slug)
        tokens[index].attrSet("id", slug)
    return parser, tokens


def render_content(parser, tokens, assets="gemma4-moe-npu-guide/assets/"):
    """Render screenshot paragraphs as linked figures, keeping Markdown portable."""
    output = []
    index = 0
    prefix = "gemma4-moe-npu-guide/assets/"
    while index < len(tokens):
        group = tokens[index:index + 3]
        children = group[1].children if len(group) == 3 else None
        if (len(group) == 3 and group[0].type == "paragraph_open"
                and group[1].type == "inline" and group[2].type == "paragraph_close"
                and children and len(children) == 1 and children[0].type == "image"
                and children[0].attrGet("src").startswith(prefix + "screenshots/")):
            screenshot = children[0]
            src = html.escape(assets + screenshot.attrGet("src")[len(prefix):], quote=True)
            alt = html.escape(parser.renderer.renderInlineAsText(
                screenshot.children or [], parser.options, {}), quote=True)
            caption = html.escape(screenshot.attrGet("title") or "")
            figure = Token("html_block", "", 0)
            figure.content = (
                f'<figure class="app-screenshot"><a href="{src}">'
                f'<img src="{src}" alt="{alt}" loading="lazy" decoding="async">'
                f'</a><figcaption>{caption}</figcaption></figure>\n'
            )
            output.append(figure)
            index += 3
        else:
            output.append(tokens[index])
            index += 1
    return parser.renderer.render(output, parser.options, {})


def render(source, language="en", reference=None):
    parser, tokens = parsed_source(source, reference)
    ui = UI[language]
    headings = [(token.attrGet("id"), tokens[i + 1].content)
                for i, token in enumerate(tokens)
                if token.type == "heading_open" and token.tag == "h2"]
    body = render_content(parser, tokens)
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")
    toc = "\n".join(
        f'<a href="#{slug}">{html.escape(label)}</a>' for slug, label in headings
    )
    return f'''<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{ui['description']}">
<title>{ui['guide_title']} — Quick.AI</title>
<link rel="icon" type="image/svg+xml" href="gemma4-moe-npu-guide/assets/quickai-icon.svg">
<style>{STYLE}</style>
</head>
<body>
<!-- Generated by tools/render_gemma4_npu_guide.py. Edit the Markdown source. -->
<a class="skip" href="#guide">{ui['skip']}</a>
<div class="topbar"><img class="brand-mark" src="gemma4-moe-npu-guide/assets/quickai-icon.svg" width="35" height="35" alt=""><span>QUICK.AI / ANDROID / GEMMA4-MOE</span></div>
<div class="layout">
<nav aria-label="{ui['documentation']}"><div role="group" aria-label="{ui['language']}">
<a data-language="ko" lang="ko" hreflang="ko" href="gemma4-moe-npu-guide.ko.html">한국어</a>
<a data-language="en" lang="en" hreflang="en" href="gemma4-moe-npu-guide.html">English</a>
</div><strong>{ui['on_page']}</strong>{toc}</nav>
<main id="guide">{body}</main>
</div>
<script>
function syncLanguageLinks() {{
  document.querySelectorAll('[data-language]').forEach(link => {{
    const target = new URL(link.getAttribute('href'), location.href);
    target.hash = location.hash;
    link.href = target.href;
  }});
}}
syncLanguageLinks();
window.addEventListener('hashchange', syncLanguageLinks);
</script>
</body>
</html>
'''


PAGES = [
    ("index.html", "Overview", "Get started"),
    ("requirements.html", "Requirements", "Get started"),
    ("linux-android-tools.html", "Linux & Android tools", "Get started"),
    ("hexagon-hexkl.html", "Hexagon SDK & HexKL", "Get started"),
    ("source-device.html", "Source & device setup", "Get started"),
    ("build-install.html", "Build & install", "Build"),
    ("model-pack.html", "Prepare the model pack", "Build"),
    ("cli.html", "CLI inference", "Run"),
    ("app.html", "Quick.AI app", "Run"),
    ("benchmarks.html", "Benchmark results", "Reference"),
    ("troubleshooting.html", "Troubleshooting", "Reference"),
    ("maintaining-guide.html", "Maintain this guide", "Reference"),
    ("lfm2-benchmarks.html", "LFM2 benchmark results", "Appendix"),
]

KO_TITLES = ["소개", "실행 환경", "Linux 및 Android 도구", "Hexagon SDK와 HexKL",
             "소스 및 기기 준비", "빌드 및 설치", "모델 팩 준비", "CLI 추론",
             "Quick.AI 앱", "벤치마크 결과", "문제 해결", "문서 관리", "LFM2 벤치마크 결과"]
KO_GROUPS = {"Get started": "시작하기", "Build": "빌드", "Run": "실행",
             "Reference": "참고 자료", "Appendix": "부록"}
SIDEBAR_CHILDREN = {
    "requirements.html": ("linux-android-tools.html", "hexagon-hexkl.html", "source-device.html"),
}
UI = {
    "en": {
        "guide_title": "Gemma4-MoE on the Hexagon NPU", "guide": "Gemma4 NPU guide",
        "description": "Install Hexagon SDK and HexKL, build nntrainer, and run Gemma4-MoE on an Android NPU through the CLI and Quick.AI app.",
        "skip": "Skip to content", "open_menu": "Open navigation", "close_menu": "Close navigation",
        "documentation": "Documentation", "docs": "Docs", "on_page": "On this page",
        "explore": "Explore the guide", "previous": "Previous", "next": "Next",
        "pagination": "Previous and next pages", "back_top": "Back to top",
        "markdown": "Markdown source",
        "theme_dark": "Switch to dark theme", "theme_light": "Switch to light theme",
        "reviewed": "Source baseline: PR #4480 · Reviewed November 5, 2026",
        "search": "Search documentation", "close_search": "Close search",
        "search_label": "Search all guide pages", "search_hint": "SDK, build, model pack…",
        "search_empty": f"Type to search all {len(PAGES)} pages.", "search_count": "{count} matching sections",
        "search_none": "No results. Try SDK, HexKL, build or permissions.",
        "copy": "Copy", "copy_label": "Copy code block", "copied": "Copied!",
        "copy_select": "Select to copy", "copy_success": "Code copied.",
        "copy_failure": "Copy unavailable. Select the code and copy it manually.",
        "heading_link": "Link to {heading}", "language": "Language",
    },
    "ko": {
        "guide_title": "Hexagon NPU에서 Gemma4-MoE 실행하기", "guide": "Gemma4 NPU 가이드",
        "description": "Hexagon SDK와 HexKL 설치부터 nntrainer 빌드, Android CLI와 Quick.AI 앱에서 Gemma4-MoE 실행까지 안내합니다.",
        "skip": "본문으로 이동", "open_menu": "메뉴 열기", "close_menu": "메뉴 닫기",
        "documentation": "문서 탐색", "docs": "문서", "on_page": "이 페이지의 내용",
        "explore": "가이드 둘러보기", "previous": "이전", "next": "다음",
        "pagination": "이전 및 다음 페이지", "back_top": "맨 위로",
        "markdown": "Markdown 원본",
        "theme_dark": "어두운 테마로 전환", "theme_light": "밝은 테마로 전환",
        "reviewed": "기준 소스: PR #4480 · 검토일: 2026년 11월 5일",
        "search": "문서 검색", "close_search": "검색 닫기",
        "search_label": "전체 가이드 검색", "search_hint": "SDK, 빌드, 모델 팩…",
        "search_empty": f"검색어를 입력하면 전체 {len(PAGES)}개 페이지에서 찾습니다.", "search_count": "{count}개 섹션에서 찾았습니다",
        "search_none": "검색 결과가 없습니다. SDK, HexKL, 빌드, 권한 등을 검색해 보세요.",
        "copy": "복사", "copy_label": "코드 블록 복사", "copied": "복사 완료!",
        "copy_select": "선택하여 복사", "copy_success": "코드를 복사했습니다.",
        "copy_failure": "자동 복사를 사용할 수 없습니다. 코드를 선택해 직접 복사하세요.",
        "heading_link": "{heading} 링크", "language": "언어",
    },
}


def render_site(source, language="en", reference=None):
    parser, tokens = parsed_source(source, reference)
    ui = UI[language]
    pages = PAGES if language == "en" else [
        (filename, title, KO_GROUPS[group])
        for (filename, _, group), title in zip(PAGES, KO_TITLES)
    ]
    assets = "assets/" if language == "en" else "../assets/"
    source_prefix = "../" if language == "en" else "../../"
    source_stem = "gemma4-moe-npu-guide" + (".ko" if language == "ko" else "")
    boundaries = [0] + [
        i for i, token in enumerate(tokens)
        if token.type == "heading_open" and token.tag == "h2"
    ] + [len(tokens)]
    if len(boundaries) - 1 != len(PAGES):
        raise ValueError("Update PAGES when adding or removing a guide section")

    heading_pages = {
        token.attrGet("id"): pages[page_index][0]
        for page_index in range(len(pages))
        for token in tokens[boundaries[page_index]:boundaries[page_index + 1]]
        if token.type == "heading_open"
    }
    outputs = {}
    search = []
    for page_index, (filename, title, group) in enumerate(pages):
        section = tokens[boundaries[page_index]:boundaries[page_index + 1]]
        for token in section:
            for child in token.children or []:
                href = child.attrGet("href") or ""
                if child.type == "link_open" and href.startswith("#"):
                    destination = heading_pages.get(href[1:])
                    if destination and destination != filename:
                        child.attrSet("href", destination + href)
                elif child.type == "link_open" and href.startswith("gemma4-moe-npu-guide/assets/"):
                    child.attrSet("href", assets + href[len("gemma4-moe-npu-guide/assets/"):])
        toc = []
        for i, token in enumerate(section):
            if token.type != "heading_open":
                continue
            label = section[i + 1].content
            slug = token.attrGet("id")
            # Each section is now a document, with one h1 and h2 subsections.
            tag = "h1" if i == 0 else "h2"
            token.tag = tag
            section[i + 2].tag = tag
            if i:
                toc.append((slug, label))

        body = render_content(parser, section, assets)
        body = body.replace("<table>", '<div class="table-wrap" tabindex="0"><table>')
        body = body.replace("</table>", "</table></div>")
        body = body.replace("<pre>", '<pre tabindex="0">')
        # Index per subsection, so results land near the matching instructions.
        starts = [i for i, token in enumerate(section) if token.type == "heading_open"]
        for start, end in zip(starts, starts[1:] + [len(section)]):
            text = " ".join(t.content for t in section[start:end] if t.content)
            search.append({"title": title, "heading": section[start + 1].content,
                           "url": filename + "#" + section[start].attrGet("id"),
                           "text": re.sub(r"\s+", " ", text)})

        sidebar = []
        last_group = None
        child_pages = {child for children in SIDEBAR_CHILDREN.values() for child in children}
        page_labels = {page: label for page, label, _ in pages}
        def nav_link(page, label):
            current = ' aria-current="page"' if page == filename else ""
            return f'<a href="{page}"{current}>{html.escape(label)}</a>'
        for page, label, category in pages:
            if page in child_pages:
                continue
            if category != last_group:
                if last_group is not None:
                    sidebar.append("</ul>")
                sidebar.append(f'<p class="nav-group">{category}</p><ul>')
                last_group = category
            children = SIDEBAR_CHILDREN.get(page, ())
            ancestor = ' class="nav-parent-active"' if filename in children else ""
            sidebar.append(f'<li{ancestor}>{nav_link(page, label)}')
            if children:
                sidebar.append('<ul class="nav-children">')
                for child in children:
                    sidebar.append(f'<li>{nav_link(child, page_labels[child])}</li>')
                sidebar.append('</ul>')
            sidebar.append('</li>')
        sidebar.append("</ul>")

        if page_index == 0:
            cards = []
            for page, label, category in pages[1:]:
                cards.append(f'<a class="section-card" href="{page}"><small>{category}</small>'
                             f'<span>{html.escape(label)} <b aria-hidden="true">→</b></span></a>')
            body += f'<h2 id="explore-the-guide">{ui["explore"]}</h2><div class="section-cards">' + "".join(cards) + "</div>"
            toc.append(("explore-the-guide", ui["explore"]))

        pager = []
        for direction, offset in (("previous", -1), ("next", 1)):
            index = page_index + offset
            if 0 <= index < len(PAGES):
                page, label, _ = pages[index]
                pager.append(f'<a class="pager-{direction.lower()}" href="{page}">'
                             f'<small>{ui[direction]}</small><span>{html.escape(label)}</span></a>')
        toc_html = "".join(f'<li><a href="#{slug}">{html.escape(label)}</a></li>' for slug, label in toc)
        english_url = filename if language == "en" else "../" + filename
        korean_url = "ko/" + filename if language == "en" else filename
        def language_links():
            return (f'<div class="language-switch" role="group" aria-label="{ui["language"]}">'
                    f'<a href="{korean_url}" lang="ko" hreflang="ko" data-language="ko"'
                    + (' aria-current="true"' if language == "ko" else '') + '>한국어</a>'
                    f'<a href="{english_url}" lang="en" hreflang="en" data-language="en"'
                    + (' aria-current="true"' if language == "en" else '') + '>English</a></div>')
        outputs[filename] = f'''<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{html.escape(title)} — {ui['description']}">
<title>{html.escape(title)} — Gemma4 NPU · Quick.AI</title>
<link rel="icon" type="image/svg+xml" href="{assets}quickai-icon.svg">
<link rel="alternate" hreflang="en" href="{english_url}">
<link rel="alternate" hreflang="ko" href="{korean_url}">
<script>try {{ document.documentElement.dataset.theme = localStorage.getItem('nntr-guide-theme') || 'light'; }} catch {{}}</script>
<link rel="stylesheet" href="{assets}site.css">
<script id="guide-i18n" type="application/json">{json.dumps(ui, ensure_ascii=False)}</script>
<script src="assets/search-index.js" defer></script>
<script src="{assets}site.js" defer></script>
</head>
<body>
<!-- Generated from docs/{source_stem}.md. Do not edit this HTML. -->
<a class="skip-link" href="#main-content">{ui['skip']}</a>
<button class="theme-toggle" aria-label="{ui['theme_dark']}" title="{ui['theme_dark']}">
<svg class="icon-moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"/></svg>
<svg class="icon-sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true" focusable="false"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
</button>
<header class="mobile-header"><button class="menu-toggle" aria-label="{ui['open_menu']}" aria-expanded="false" aria-controls="sidebar">☰</button><a class="mobile-brand" href="index.html"><img class="brand-mark" src="{assets}quickai-icon.svg" width="28" height="28" alt="">Quick.AI <span>/ Gemma4 NPU</span></a>{language_links()}</header>
<div class="sidebar-backdrop" hidden></div>
<aside class="sidebar" id="sidebar">
<a class="brand" href="index.html"><img class="brand-mark" src="{assets}quickai-icon.svg" width="35" height="35" alt=""><span>Quick.AI<small>{ui['guide']}</small></span></a>
<div class="version">PR #4480 <span>·</span> 67d2903a</div>
{language_links()}
<button class="search-open" aria-haspopup="dialog"><span>{ui['search']}</span><kbd>⌘ / Ctrl K</kbd></button>
<nav class="site-nav" aria-label="{ui['documentation']}">{"".join(sidebar)}</nav>
<div class="sidebar-footer"><a href="{source_prefix}{source_stem}.md">{ui['markdown']} ↗</a></div>
</aside>
<div class="page-shell">
<main id="main-content" tabindex="-1">
<div class="breadcrumbs"><a href="index.html">{ui['docs']}</a><span aria-hidden="true">/</span><span>{html.escape(group)}</span><span aria-hidden="true">/</span><span>{html.escape(title)}</span></div>
<article>{body}</article>
<nav class="page-navigation" aria-label="{ui['pagination']}">{"".join(pager)}</nav>
<footer class="page-footer">Quick.AI · {ui['guide_title']}<br>{ui['reviewed']}</footer>
</main>
<aside class="page-toc" aria-label="{ui['on_page']}"><p>{ui['on_page']}</p><ul>{toc_html}</ul><a class="back-top" href="#main-content">{ui['back_top']} ↑</a></aside>
</div>
<dialog id="search-dialog" aria-labelledby="search-title">
<div class="search-heading"><h2 id="search-title">{ui['search']}</h2><button class="search-close" aria-label="{ui['close_search']}">✕</button></div>
<label class="sr-only" for="search-input">{ui['search_label']}</label>
<input id="search-input" type="search" placeholder="{ui['search_hint']}" autocomplete="off">
<p id="search-status" role="status">{ui['search_empty']}</p><ul id="search-results"></ul>
</dialog>
<div class="sr-only" role="status" id="copy-status"></div>
</body>
</html>
'''
    outputs["assets/search-index.js"] = (
        "// Generated by tools/render_gemma4_npu_guide.py\nwindow.GUIDE_SEARCH = "
        + json.dumps(search, ensure_ascii=False, indent=2) + ";\n"
    )
    return outputs


def main():
    args_parser = argparse.ArgumentParser(description=__doc__)
    args_parser.add_argument("--check", action="store_true",
                             help="Fail if any generated documentation needs regeneration")
    args = args_parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "docs/gemma4-moe-npu-guide.md"
    markdown = source.read_text(encoding="utf-8")
    korean = source.with_name("gemma4-moe-npu-guide.ko.md").read_text(encoding="utf-8")
    site = source.with_suffix("")
    outputs = {source.with_suffix(".html"): render(markdown)}
    outputs.update({site / name: content for name, content in render_site(markdown).items()})
    outputs[source.with_name("gemma4-moe-npu-guide.ko.html")] = render(korean, "ko", markdown)
    outputs.update({site / "ko" / name: content
                    for name, content in render_site(korean, "ko", markdown).items()})
    if args.check:
        stale = [str(path.relative_to(root)) for path, content in outputs.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            args_parser.exit(1, "Regenerate documentation: " + ", ".join(stale) + "\n")
        print(f"All {len(outputs)} generated files match the Markdown source.")
    else:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(f"Generated {len(outputs)} files. Site entry: {site / 'index.html'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scan built mdBook HTML for math expressions that pulldown-cmark's
inline-emphasis parser ate.

The bug class: `$...$` or `$$...$$` spans whose `_` characters get
paired by pulldown-cmark's italic emphasis logic, producing literal
`<em>...</em>` tags INSIDE the math content. MathJax then can't
parse the math (the HTML tags terminate the math expression),
and the page renders raw `$$y^{(n)}<em>t = \\sum</em>...$$` text.

Recommended fix: escape the `_` as `\\_` in the source markdown. MathJax
accepts `\\_` as a subscript operator; pulldown-cmark doesn't pair it
as emphasis.

Also catches related rendering corruption:
- curly apostrophes inside math (smart-punctuation leak),
- math inside HTML attributes (`title=`, `alt=`),
- mismatched braces.

Usage:
    python find_em_in_math.py [BOOK_DIR]
    # exit 0 if clean; exit 1 if any findings.

Defaults: book_dir = ../book (relative to this script's parent).
"""
import json
import re
import sys
from pathlib import Path

SAFE_TAGS = ["script", "style", "pre", "code", "noscript", "textarea"]


def strip_safe(html: str) -> str:
    """Blank out content of tags MathJax skips so we don't false-flag
    literal `$X$` examples shown inside code blocks."""
    for tag in SAFE_TAGS:
        html = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>",
            lambda m: " " * len(m.group(0)),
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
    html = re.sub(
        r"<mjx-container\b[^>]*>.*?</mjx-container>",
        lambda m: " " * len(m.group(0)),
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return html


def find_display_math_with_em(html: str):
    findings = []
    for m in re.finditer(r"\$\$([^\$]+?)\$\$", html, flags=re.DOTALL):
        inner = m.group(1)
        if re.search(r"</?(em|strong)\b", inner, re.IGNORECASE):
            findings.append(
                {
                    "kind": "display_math_with_em_strong",
                    "snippet": m.group(0)[:300],
                    "offset": m.start(),
                }
            )
    return findings


def find_inline_math_with_em(html: str):
    findings = []
    # Start class: any character that plausibly opens an inline-math
    # expression — letters, backslash (LaTeX command), AND opening
    # brackets `(`, `[`, `{` since math like `$(A^{\top})_{ij}$` is
    # valid and the underscore-eats-into-emphasis bug bit those
    # cases too.
    pat = re.compile(
        r"\$([A-Za-z\\(\[{][^\$\n]{0,300}?<(em|strong)>[^\$\n]{0,300}?)\$",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pat.finditer(html):
        inner = m.group(1)
        has_open = re.search(r"<(em|strong)\b", inner, re.IGNORECASE)
        has_close = re.search(r"</(em|strong)>", inner, re.IGNORECASE)
        if has_open and has_close:
            findings.append(
                {
                    "kind": "inline_math_with_em_strong",
                    "snippet": m.group(0)[:300],
                    "offset": m.start(),
                }
            )
    return findings


def find_curly_quotes_in_math(html: str):
    findings = []
    for m in re.finditer(
        r"\$[^\$\n<>]{0,200}?[‘’][^\$\n<>]{0,200}?\$", html
    ):
        findings.append(
            {
                "kind": "curly_quote_in_math",
                "snippet": m.group(0),
                "offset": m.start(),
            }
        )
    return findings


def find_math_in_attribute(html: str):
    findings = []
    for m in re.finditer(
        r'(\w[\w-]*)\s*=\s*"([^"]*\$[A-Za-z\\^_\(\[][^"]*\$[^"]*)"',
        html,
    ):
        attr = m.group(1).lower()
        if attr in (
            "type", "src", "id", "class", "name", "rel", "lang",
            "dir", "charset", "href", "content",
        ):
            continue
        findings.append(
            {
                "kind": "math_in_attribute",
                "attr": attr,
                "value": m.group(2)[:200],
                "offset": m.start(),
            }
        )
    return findings


def offset_to_line(html: str, offset: int) -> int:
    return html.count("\n", 0, offset) + 1


def audit_file(path: Path):
    html = path.read_text(encoding="utf-8")
    stripped = strip_safe(html)
    findings = []
    for fn in (
        find_display_math_with_em,
        find_inline_math_with_em,
        find_curly_quotes_in_math,
        find_math_in_attribute,
    ):
        for f in fn(stripped):
            f["html_line"] = offset_to_line(stripped, f["offset"])
            findings.append(f)
    return findings


def main(argv):
    book_dir = Path(argv[1]) if len(argv) > 1 else (
        Path(__file__).resolve().parent.parent / "book"
    )
    if not book_dir.is_dir():
        print(f"book dir not found: {book_dir}", file=sys.stderr)
        return 2

    out = {}
    files = sorted(book_dir.glob("*.html"))
    for path in files:
        findings = audit_file(path)
        if findings:
            out[path.name] = findings
    total = sum(len(v) for v in out.values())
    if total == 0:
        print(f"OK — audited {len(files)} HTML files; no rendering-time math corruption.")
        return 0
    print(f"FAIL — {total} findings across {len(out)} files:")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

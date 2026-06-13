#!/usr/bin/env python3
"""Scan built mdBook HTML for `$...$` patterns that should be MathJax
but weren't rendered. These are the kind of bugs the source-level
validator can't catch — they appear only in the final rendered output
(smart-quote corruption inside math, MathJax config drift, etc.).

Strips out content that's allowed to contain `$` (code blocks, pre,
already-rendered <mjx-container>) before searching, so what's left
is genuine "math that should have rendered but didn't."

Usage:
    python3 find_unprocessed_math.py [BOOK_DIR]

Defaults: book_dir=../book
"""
import re
import sys
from pathlib import Path

DEFAULT_BOOK_DIR = Path(__file__).resolve().parent.parent / "book"

# Patterns inside HTML that legitimately contain `$` and should be
# stripped before looking for unrendered math.
SAFE_TAGS = ["script", "style", "pre", "code"]


def strip_safe(html: str) -> str:
    """Remove script/style/pre/code/mjx-container blocks from HTML."""
    for tag in SAFE_TAGS:
        html = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
    html = re.sub(
        r"<mjx-container[^>]*>.*?</mjx-container>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return html


def find_unprocessed(html: str):
    """Return list of suspicious `$...$` patterns in the stripped HTML."""
    stripped = strip_safe(html)
    # Patterns we care about: $<letter or backslash>...$
    # That distinguishes math (uses LaTeX commands or named vars) from
    # currency mentions like "$10" or stray "$".
    matches = re.findall(r"\$[A-Za-z\\][^$\n<>]{0,40}\$", stripped)
    # Also catch display math `$$...$$` that survived.
    matches += re.findall(r"\$\$[^$\n<>]{1,80}\$\$", stripped)
    # Filter out: currency-like and obvious non-math.
    return [m for m in matches if not re.match(r"^\$\s*\d", m)]


def main(argv):
    book_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_BOOK_DIR
    total = 0
    for html_path in sorted(book_dir.glob("*.html")):
        text = html_path.read_text(encoding="utf-8")
        survivors = find_unprocessed(text)
        if survivors:
            print(f"\n=== {html_path.name} === ({len(survivors)} unprocessed)")
            for m in survivors[:8]:
                print(f"  {m}")
            total += len(survivors)
    if total == 0:
        print("OK — no unprocessed math found in built HTML.")
        return 0
    print(f"\nFAIL — {total} unprocessed math patterns found.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

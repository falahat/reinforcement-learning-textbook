#!/usr/bin/env python3
"""Fix the `\\\\` row-separator-eaten-by-pulldown-cmark bug.

Inside display math (`$$ ... $$`), pulldown-cmark consumes one of the
two backslashes in `\\\\` (the LaTeX row separator), so the rendered
HTML contains `\\` (single backslash + space) which MathJax interprets
as an escape, not a row break. Matrices, aligned blocks, and cases
environments then collapse to one row.

Fix: write `\\\\\\\\` (four backslashes) in the source. Pulldown-cmark
consumes one to produce `\\\\` in the HTML; MathJax interprets that as
the row break.

This script scans every `*.md` in `docs/textbook/`, finds every
`$$ ... $$` block, and replaces every `\\\\` row separator that is
NOT already preceded by another `\\` (i.e. not already `\\\\\\\\`).

Idempotent: running twice is a no-op.

Usage:
    python docs/textbook/lint/fix_matrix_rowsep.py
"""

import re
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # docs/textbook

# Match $$ ... $$ blocks (non-greedy, DOTALL so they cross newlines).
DISPLAY_MATH = re.compile(r"(\$\$)(.+?)(\$\$)", re.DOTALL)

# Match exactly two backslashes (`\\`), NOT preceded by another
# backslash (so we don't double-process `\\\\`).
ROW_SEP = re.compile(r"(?<!\\)\\\\(?!\\)")


def fix_block(match):
    body = match.group(2)
    # Skip if no row separator candidates.
    if "\\\\" not in body:
        return match.group(0)
    new_body = ROW_SEP.sub(r"\\\\\\\\", body)
    return f"$$ {new_body} $$" if False else match.group(1) + new_body + match.group(3)


def fix_file(path: Path) -> int:
    """Returns the number of row separators fixed in this file."""
    text = path.read_text(encoding="utf-8")
    before_count = 0

    def fix_and_count(m):
        nonlocal before_count
        body = m.group(2)
        n = len(ROW_SEP.findall(body))
        before_count += n
        new_body = ROW_SEP.sub(r"\\\\\\\\", body)
        return m.group(1) + new_body + m.group(3)

    new_text = DISPLAY_MATH.sub(fix_and_count, text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return before_count


def main():
    files = sorted(ROOT.glob("*.md"))
    files = [f for f in files if f.name not in {"SUMMARY.md", "README.md"}]
    total = 0
    for f in files:
        n = fix_file(f)
        if n:
            print(f"  {f.name}: {n} row separators fixed")
            total += n
    print(f"\nTotal: {total} substitutions across {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

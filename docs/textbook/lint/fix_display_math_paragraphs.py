#!/usr/bin/env python3
"""Insert blank lines before and after `$$…$$` display-math lines that
follow or precede a non-blank line, so GitHub recognises the math
block as its own paragraph.

Idempotent — running twice is a no-op.
"""
import os
import re

TEXTBOOK_DIR = os.path.join(os.path.dirname(__file__), '..')
# Standalone `$$…$$` on a single line. May have leading `> ` if inside
# a blockquote — we leave those to the blockquote-math fix script.
SINGLE_LINE = re.compile(r'^\s*\$\$[^\n]*\$\$\s*$')


def is_blank(line: str) -> bool:
    return line.strip() == ''


def needs_blank_before(prev: str) -> bool:
    if prev is None:
        return False
    return not is_blank(prev) and not prev.rstrip().endswith('$$') and not prev.startswith('>')


def needs_blank_after(nxt: str) -> bool:
    if nxt is None:
        return False
    return not is_blank(nxt) and not nxt.lstrip().startswith('$$') and not nxt.startswith('>')


def fix_file(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    out: list[str] = []
    n_fixes = 0
    for i, line in enumerate(lines):
        if SINGLE_LINE.match(line) and not line.lstrip().startswith('>'):
            prev = lines[i - 1] if i > 0 else None
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            if needs_blank_before(prev):
                out.append('')
                n_fixes += 1
            out.append(line)
            if needs_blank_after(nxt):
                out.append('')
                n_fixes += 1
        else:
            out.append(line)
    if n_fixes:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out))
    return n_fixes


total = 0
for fn in sorted(os.listdir(TEXTBOOK_DIR)):
    if fn.endswith('.md'):
        n = fix_file(os.path.join(TEXTBOOK_DIR, fn))
        if n:
            print(f'  {fn}: {n} blank line(s) inserted around display math')
            total += n
print(f'Total fixes: {total}')

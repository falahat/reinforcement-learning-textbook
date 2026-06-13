#!/usr/bin/env python3
"""Insert a space between a word/digit/hyphen and an opening `$` so
GitHub recognises the math as math. Pattern:

  length-$n$  →  length $n$
  constant-$\\alpha$  →  constant $\\alpha$

Idempotent.
"""
import os
import re

TEXTBOOK_DIR = os.path.join(os.path.dirname(__file__), '..')
# Match: word-char/digit, then a hyphen, then `$`, then a math-marker
# (letter or backslash). The space replaces the hyphen.
PATTERN = re.compile(r'([A-Za-z0-9])-(\$[A-Za-z\\])')


def fix_file(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    new, n = PATTERN.subn(r'\1 \2', text)
    if n:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
    return n


total = 0
for fn in sorted(os.listdir(TEXTBOOK_DIR)):
    if fn.endswith('.md'):
        n = fix_file(os.path.join(TEXTBOOK_DIR, fn))
        if n:
            print(f'  {fn}: {n} fix(es)')
            total += n
print(f'Total: {total}')

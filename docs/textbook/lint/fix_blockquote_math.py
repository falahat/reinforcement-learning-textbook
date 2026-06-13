#!/usr/bin/env python3
"""Convert single-line `> $$…$$` display math to multi-line form so
GitHub's markdown→math pipeline recognises it.

Before:  `> $$math$$`
After:   `>`
         `> $$`
         `> math`
         `> $$`
         `>`

Idempotent — running twice is a no-op.
"""
import os
import re

TEXTBOOK_DIR = os.path.join(os.path.dirname(__file__), '..')
# Match `> ` indent (any spaces between > and $$), capture the math.
PATTERN = re.compile(r'^(>\s*)\$\$(.+?)\$\$\s*$')


def fix_file(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    out: list[str] = []
    n_fixes = 0
    for line in lines:
        m = PATTERN.match(line)
        if m:
            indent = m.group(1).rstrip() + ' '  # normalise to "> "
            math = m.group(2)
            out.append(indent.rstrip())          # "> "
            out.append(indent + '$$')
            out.append(indent + math)
            out.append(indent + '$$')
            out.append(indent.rstrip())          # "> "
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
            print(f'  {fn}: {n} blockquote math block(s) split')
            total += n
print(f'Total fixes: {total}')

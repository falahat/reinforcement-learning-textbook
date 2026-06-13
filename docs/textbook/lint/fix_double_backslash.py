#!/usr/bin/env python3
"""One-shot repair: collapse `\\\\X` → `\\X` for KaTeX macros that were
corrupted by an earlier sed pass. Idempotent (running twice doesn't
double-replace).
"""
import os
import re
import sys

TEXTBOOK_DIR = os.path.join(os.path.dirname(__file__), '..')
MACROS = [
    'left', 'right',
    'Big', 'big', 'Bigg', 'bigg',
    'biggl', 'biggr', 'Biggl', 'Biggr',
    'bigl', 'bigr', 'Bigl', 'Bigr',
    'biggm', 'Biggm',
]

PATTERN = re.compile(r'\\\\(' + '|'.join(MACROS) + r')\b')

def fix_file(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    new = PATTERN.sub(lambda m: '\\' + m.group(1), text)
    if new != text:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        return text.count(r'\\') - new.count(r'\\')
    return 0

total = 0
for fn in sorted(os.listdir(TEXTBOOK_DIR)):
    if fn.endswith('.md'):
        fixed = fix_file(os.path.join(TEXTBOOK_DIR, fn))
        if fixed:
            print(f'  {fn}: {fixed} fix(es)')
            total += fixed
print(f'Total fixes: {total}')
sys.exit(0 if total >= 0 else 1)

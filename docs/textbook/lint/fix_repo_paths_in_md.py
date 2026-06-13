#!/usr/bin/env python3
"""One-shot source rewriter: convert relative repo paths in the
textbook markdown (`../../crates/...`, `../proposals/...`, etc.) to
absolute GitHub URLs.

Why a source rewrite (and not a build-time HTML post-process):
absolute URLs work identically on github.com (native markdown render)
and on github.io (mdBook output). The relative-path approach only
helps if you're reading the .md in a file browser that resolves
relative paths — a niche case not worth a custom CI step.

Idempotent. Run once after authoring new chapters that use the old
relative-path style; the resulting markdown is the canonical form.

Usage:
    python fix_repo_paths_in_md.py [TEXTBOOK_DIR] [REPO_URL]

Defaults: textbook_dir=.., repo_url=https://github.com/falahat/simulator/blob/main
"""
import re
import sys
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPO_BASE = "https://github.com/falahat/simulator/blob/main"

# Matches `](../../<root>...)` from docs/textbook/, where <root> is
# crates/, docs/, or README.md — links that escape both `textbook/`
# and `docs/`. The leading `]` anchors to a markdown link target.
PATTERN_TOPLEVEL = re.compile(
    r'\]\(((?:\.\./)+)(crates/|docs/|README\.md)([^)]*)\)'
)

# Matches `](../<dir>/...)` from docs/textbook/ where <dir> is a sibling
# of `textbook/` under docs/ (proposals/, audits/, designs/, etc.). The
# `[^./]` exclusion of the leading char keeps `../../` cases out so the
# toplevel pattern handles them without overlap.
PATTERN_SIBLING = re.compile(
    r'\]\(\.\./([^./][^)]*)\)'
)


def rewrite_toplevel(match: re.Match, repo_base: str) -> str:
    root = match.group(2)
    tail = match.group(3)
    return f']({repo_base}/{root}{tail})'


def rewrite_sibling(match: re.Match, repo_base: str) -> str:
    # `../proposals/foo.md` from `docs/textbook/` resolves to
    # `docs/proposals/foo.md` in repo terms.
    tail = match.group(1)
    return f']({repo_base}/docs/{tail})'


def main(argv):
    textbook_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR
    repo_base = argv[2] if len(argv) > 2 else DEFAULT_REPO_BASE

    if not textbook_dir.is_dir():
        print(f"textbook dir not found: {textbook_dir}", file=sys.stderr)
        return 1

    files = 0
    rewrites = 0
    for md in sorted(textbook_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        new = PATTERN_TOPLEVEL.sub(lambda m: rewrite_toplevel(m, repo_base), text)
        new = PATTERN_SIBLING.sub(lambda m: rewrite_sibling(m, repo_base), new)
        n_top = len(PATTERN_TOPLEVEL.findall(text))
        n_sib = len(PATTERN_SIBLING.findall(text))
        n = n_top + n_sib
        if n:
            md.write_text(new, encoding="utf-8")
            files += 1
            rewrites += n
            print(f"  {md.name}: {n} rewrite(s)")
    print(f"\nrewrote {rewrites} link(s) across {files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

# Phase 3 (continued) — LaTeX validation + rendering checks

The most insidious bug class in a math-heavy book is "the math
*almost* renders." The validator says everything's fine; the reader
sees raw `$` signs or mis-rendered formulas on the deployed site.

The textbook targets one rendering surface: the **mdBook output**
deployed to github.io. github.com's raw-markdown render is not a
supported target. (Trying to keep both pipelines happy at once means
you can't use techniques like `\_` escaping that work in mdBook +
MathJax but look ugly in GitHub's renderer — pick one surface and
optimise for it.)

Two layers of validation, both mandatory.

## Layer 1 — pre-build syntax check

A Node script (`docs/textbook/lint/check.js`) that:
1. Parses each `$…$` and `$$…$$` block via KaTeX in strict mode.
   Catches `R_\max`-style errors that KaTeX rejects.
2. Runs regex rules against parser-fragile patterns that
   pulldown-cmark + MathJax mishandle (display-math-in-blockquote,
   indented-display, etc.).

Install deps:

```sh
cd docs/textbook/lint
npm install --save katex@^0.16
npm install         # creates package-lock.json — COMMIT THIS
```

Run via `node check.js`. Output:

```
OK  2838 math blocks pass parse + context checks across 23 files.
```

Or on failure:

```
02_chapter.md:252  [parse:inline]  KaTeX parse error: Got '\max'...
FAIL  1 failure(s) across 23 files.
```

Key rules (all calibrated against pulldown-cmark + MathJax,
empirically derived from real bugs):

- `display-math-in-blockquote` — `> $$…$$` doesn't get recognised as math.
- `display-math-leading-indent` — 4-space indent treated as code block.
- `display-math-on-continuation-line` — needs blank line above/below.
- `pmatrix-in-inline-math` — multi-line matrix env inside `$…$`
  (markdown eats the `\\`).
- `inline-math-spans-newline` — odd count of unescaped `$` on a line.

The old `opener-without-leading-space` rule (`length-$n$` not
recognised) was github.com-specific — MathJax doesn't care about the
preceding character, so we dropped it.

## Layer 2 — rendered-HTML checks

The validator can't see what mdBook does to the markdown. mdBook
runs the source through pulldown-cmark, which has its own
transformations:

- **Smart-punctuation.** `'` → `'` (curly). MathJax can't parse
  `$s'$` (curly). Fix: `smart-punctuation = false` in book.toml.
- **Multi-line `<script>` blocks** mangled. mdBook's HTML pass
  doesn't handle `<script>` content well; it converts quotes,
  strips whitespace, breaks template literals. Fix: ship widget
  JS as separate files, two-line embed in chapter.
- **Math inside HTML attributes.** If you write
  `<span title="$\alpha$">α</span>`, the `$…$` inside the attribute
  doesn't get processed by MathJax. Avoid math in attributes.

After `mdbook build`, manually check the output. Two-pass approach:

### Spot-check pass

Open ~3 random chapters in the browser. For each:
1. Math renders as glyphs, not raw `$`. Check inline and display.
2. Greek letters look like Greek letters.
3. Code blocks have syntax highlighting.
4. Mermaid diagrams render (not as raw code).
5. Widget mount points show the widget (not blank divs).

### Grep pass

Find unrendered math survivors:

```sh
# Look for $...$ patterns INSIDE table cells, paragraph text, headers,
# and list items — anywhere MathJax should have processed but didn't.
# Exclude code blocks and pre.

grep -oE '\$[A-Za-z\\][^$<]{1,30}\$' docs/textbook/book/*.html | \
  grep -v 'class="[^"]*katex' | head -20
```

You'll find some legitimate hits (raw `$` inside `<code>` tags, e.g.
showing example math syntax to the reader). The illegitimate hits
look like `<td>$s'$</td>` (raw math in a table cell). Investigate
those.

The grep pass should be silent for the bulk of the textbook. If it
hits more than 5-10 cases per book, something is systematically
wrong.

## Source-rewrite for repo paths

Chapters that link to source code use markdown links. Two forms work:

```markdown
[file](../../crates/engine/q_learning/src/value_function.rs)
[file](https://github.com/owner/repo/blob/main/crates/engine/q_learning/src/value_function.rs)
```

The relative form resolves correctly when mdBook renders the chapter
*if* the link target lives inside the textbook's `src` tree — which
`../../crates/...` does not. The relative path therefore breaks on
the deployed github.io site (it resolves to a non-existent
`<book-url>/../../crates/...` path). The absolute form is the only
one that works in the mdBook output. Use absolute URLs everywhere.

To bulk-convert existing chapters, ship a one-shot Python rewriter
at `docs/textbook/lint/fix_repo_paths_in_md.py`. Pattern:

```python
PATTERN_TOPLEVEL = re.compile(
    r'\]\(((?:\.\./)+)(crates/|docs/|README\.md)([^)]*)\)'
)
def rewrite(match, repo_base):
    return f']({repo_base}/{match.group(2)}{match.group(3)})'
```

Run once on the source markdown; commit the rewritten files. Don't
post-process HTML — source rewrite is cleaner.

## Currency vs math conflict

Math uses `$…$`. Currency in prose also uses `$…$` (`$10`). They
collide.

Fix: in prose, use `€10` or `\$10` for currency. Keep `$X$` as
math-only. Otherwise a single-`$` line trips the
"inline-math-spans-newline" rule (odd count of `$` on a line).

For the RL textbook specifically: the "Jack's car rental example"
in Sutton & Barto uses `$10`. Convert to `€10` in restated form.

## Common KaTeX gotchas

- `\max` / `\min` / `\sup` / `\inf` without explicit subscript
  syntax: `R_\max` fails strict; use `R_{\max}` instead.
- `\\` inside inline math: markdown eats one of the backslashes.
  Use display math `$$ \\ $$` for matrices.
- `\left(` / `\right)`: KaTeX supports these but they're heavier
  than plain `(` `)`. Don't use them unless the parens grow.
- `\text{...}` inside `\frac{}{}`: works but watch for nested
  brace mismatches.

## Build-step validation in CI

Run the validator in CI as a gate:

```yaml
- name: LaTeX lint
  run: |
    cd docs/textbook/lint
    npm ci
    node check.js
```

Use `npm ci` (not `npm install`) to honor the lockfile and fail
fast if it's missing. **Commit `package-lock.json`** — `npm ci`
requires it.

If the validator fails, the build fails. This prevents bugs from
shipping to Pages.

## Output of this phase

`mdbook build` clean. Validator green (2000+ blocks across all
chapters). Spot-check pass clean. Source-paths absolute.

Next: `references/06-widgets-architecture.md` for the widget
language decision tree.

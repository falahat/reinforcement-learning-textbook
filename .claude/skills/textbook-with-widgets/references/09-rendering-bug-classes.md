# Rendering bug classes — empirically-found, file-by-file

Catalogue of math-rendering bugs that source-level KaTeX validators
do NOT catch. These show up only in the rendered HTML and were
discovered during one production textbook's build. Every entry has
a known-good detection rule, a known-good fix, and the historical
incident that surfaced it.

The take-away: source-level validation is necessary but **not
sufficient**. Always run a rendered-HTML lint pass too — see the
script at `docs/textbook/lint/find_em_in_math.py` in the source
project for a working detector covering classes A and B below.

## Class 0 (most common): pulldown-cmark eats `\\` row separators inside display math

**Symptom.** A LaTeX matrix (`pmatrix`, `bmatrix`), an `aligned` block,
or a `cases` environment renders with only **one row** instead of
multiple. All the cells end up squashed onto a single row. The KaTeX
source-level validator says it's fine, the build succeeds with no
warnings, but the visual is wrong.

**Root cause.** In source markdown, two consecutive backslashes `\\`
mean "the LaTeX row separator." Pulldown-cmark — even inside
`$$ … $$` display math — interprets `\\` as "escape backslash to a
literal backslash" and writes a single `\` into the HTML. MathJax
then sees `\ ` (one backslash + the following character) which is
parsed as an escape sequence, not a row break. So the second row
is silently merged into the first.

**Example of the bug.**

```markdown
$$
A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}.
$$
```

Renders as `( 1 0 0 1 )` on one row, not the 2×2 identity.

**Detection.** After build, post-MathJax scan: any `mjx-mtable` whose
nested `mjx-itable > mjx-mtr` count is 1 (when the source clearly
intended multiple rows) is broken. A simpler static-source check: any
`\\` inside a `$$…$$` block that is NOT already `\\\\` is suspect.

The Python fixer at
`docs/textbook/lint/fix_matrix_rowsep.py` does this scan and the
fix in one pass. Run it whenever you add new matrix/aligned/cases
content.

**Fix.** Write `\\\\` (four backslashes) in source. Pulldown-cmark
consumes one to produce `\\` in the HTML; MathJax interprets that as
the row break.

```markdown
$$
A = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \end{pmatrix}.
$$
```

**Historical incident.** Twenty-three of the textbook's matrices
shipped silently broken because the author wrote `\\` (the LaTeX-
correct way) and the source-level validator considered it valid
LaTeX. The chapter 1 expansion alone had 23 broken `\\` rows; a
single batch substitution to `\\\\` fixed all of them.

**Why this isn't the same as Class A (`_` eaten as italic).** Both
are pulldown-cmark mangling math sources, but they affect different
tokens. Class A only affects subscripts in `$X$` / `$$X$$` and is
caught by the rendered-HTML lint (`find_em_in_math.py`). Class 0
affects `\\` row separators in `$$X$$` only (inline math `$X$`
doesn't generally need row separators) and is caught by
`fix_matrix_rowsep.py`.

## Class A: pulldown-cmark eats `_` as italic emphasis inside math

**Symptom.** The page shows raw `$$y^{(n)}_t = \sum_{k=0}^{n-1}...$$`
text with the middle portion italicised. The source markdown looks
correct (`$$y^{(n)}_t = \sum_{k=0}^{n-1}...$$`), the KaTeX parse-
layer validator says it's fine, but the rendered HTML contains:

```html
$$y^{(n)}<em>t = \sum</em>{k=0}^{n-1} ...$$
```

MathJax cannot parse `<em>` tags inside math, so the entire math
expression survives as raw text on the page.

**Root cause.** mdBook uses pulldown-cmark, which does NOT treat
`$...$` / `$$...$$` spans as protected from inline emphasis. When
two `_` characters appear on the same paragraph with content between
them, pulldown-cmark pairs them as italic markers — even if both
underscores are inside math subscripts.

**Detection.** Render the markdown to HTML, strip `<script>`,
`<style>`, `<pre>`, `<code>`, and `<mjx-container>` content,
then search for `<em>` or `<strong>` tags inside the surviving
`$...$` / `$$...$$` patterns. The script
`find_em_in_math.py` does exactly this. Run it after every build.

**Fix.** Replace bare `_` with `\_` inside affected math expressions.
MathJax accepts `\_` as a subscript operator escape; pulldown-cmark
does not pair `\_` as italic markers.

Before:
```markdown
$$V^\pi(s) = \mathbb{E}_\pi[r_t + \gamma G_{t+1} \mid s_t = s]$$
```

After:
```markdown
$$V^\pi(s) = \mathbb{E}\_\pi[r\_t + \gamma G\_{t+1} \mid s\_t = s]$$
```

**Historical incident.** The textbook had 23 instances of this bug
across 9 chapters, all in display math with multi-subscript
formulas (Bellman, policy-gradient, importance-sampling, advantage
estimators). Source validator was green throughout. Discovered only
when a human spot-checked the rendered live site.

**Where it bites most.** Display-math `$$X$$` blocks with multiple
subscript patterns:
- `\mathbb{E}_\pi[\cdots]` — expectation operator subscript
- `\sum_{k} \cdots` — summation index
- `\pi_\theta` — parameterised policy
- `\nabla_\theta J(\theta)` — gradient
- `\underbrace{\cdots}_{\text{label}}` — annotation

Anywhere two `_` are on the same line, the parser pairs them.

## Class B: smart-quote substitution corrupts `'` inside math

**Symptom.** State-prime notation `$s'$` renders as raw text on the
page because the rendered HTML contains `$s'$` (with U+2019 curly
right-single-quote) which MathJax can't parse.

**Root cause.** mdBook's default `smart-punctuation = true` turns
straight apostrophes into curly quotes — INCLUDING inside math
spans.

**Detection.** Render to HTML, strip safe tags, search for `'` or
`'` inside `$...$` spans.

**Fix.** Set `smart-punctuation = false` in `book.toml` under
`[output.html]`. This disables smart-quote substitution everywhere
in the book.

**Historical incident.** The notation table in chapter 0 used
state-prime `$s'$` notation. After live-site deploy, the entire
table rendered as raw text. Fix landed in one config-line change.

## Class C: `\max` / `\min` / `\sup` / `\inf` without braced subscript

**Symptom.** KaTeX parse error: `Got '\max' with no arguments as
subscript at position N: R_\max/(1-\gamma)`. Build fails the
validator.

**Root cause.** KaTeX strict mode treats `R_\max` as "subscript R
by function-with-no-arguments \max" and errors. The intent is
"R subscript max" where `\max` is rendered as a multi-letter
operator.

**Detection.** Caught by the source-level KaTeX validator already.
No HTML pass needed.

**Fix.** Wrap the multi-letter operator in braces: `R_{\max}` instead
of `R_\max`. Same for `\min`, `\sup`, `\inf`, `\arg`, `\det`, `\lim`,
`\sin`, `\cos`, `\log`, etc.

## Class D: math inside HTML attributes

**Symptom.** A title-tooltip like `<span title="$\alpha$">α</span>`
shows literally `$\alpha$` on hover, not the rendered α.

**Root cause.** MathJax only processes text content of HTML
elements, not attribute values.

**Detection.** Search rendered HTML for attributes containing
`$...$` patterns (excluding `href`, `src`, `id`, `class` and other
attributes that legitimately may contain literal dollar signs).

**Fix.** Don't put math in attributes. For tooltips with math
content, use the chapter prose or a real popover library (e.g.
Tippy.js). The simpler workaround: rewrite the tooltip to use
plain English words instead of math.

## Class E: emphasis-inside-link or math-inside-link mangling

**Symptom.** `[some $X$ thing](url)` becomes weird in the rendered
output — sometimes the `$X$` survives as raw, sometimes the link
text is broken.

**Root cause.** pulldown-cmark's link parser interacts oddly with
math spans. Not all cases reproduce.

**Detection.** Visual spot-check. No reliable automated detector.

**Fix.** Restructure to avoid math inside link text. Replace
`[the $\gamma$-contraction](url)` with `[the γ-contraction
(gamma)](url)` — use the Unicode character or the spelled-out form.

## Class F: display math next to non-blank line

**Symptom.** Display math `$$X$$` immediately preceded or followed
by a non-blank line sometimes renders as inline. The page shows
the math on the same line as the surrounding prose.

**Root cause.** mdBook's renderer treats display math as a paragraph
break ONLY when the `$$X$$` block has blank lines before and after.

**Detection.** The existing source validator has a
`display-math-on-continuation-line` rule that catches this.

**Fix.** Always put `$$X$$` on its own paragraph with blank lines
before and after.

## Class G: opener without leading space (github.com-only)

**Symptom.** `length-$n$` renders the `$n$` as plain text rather
than math on github.com's rendering of raw markdown.

**Applies to.** github.com only. MathJax (used by mdBook output)
does a text-level scan for `$…$` and doesn't care about the
preceding character, so this bug does NOT appear on github.io.

**Status.** The skill's reference projects no longer target
github.com, so this rule has been dropped from the validator.
Listed here for completeness in case a future project DOES want
github.com rendering — in which case re-add the rule from this
skill's git history.

## Note: the "raw $X$ survived in built HTML" check is futile under runtime MathJax

It's tempting to add a layer-3 rule that scans the built HTML for
raw `$X$` patterns and flags any that survived — on the theory that
"if MathJax processed it, the dollar signs would be gone." This
intuition is wrong for the recommended stack.

Under **runtime** MathJax (the setup this skill recommends), the
math is processed *in the browser, after page load*. The build
output `book/*.html` therefore contains raw `$X$` for *every* legit
math expression — that's what feeds MathJax at runtime. A
$-survival scan of the build output fires on every math
expression. In one trial it produced 2,586 false positives across
24 chapters.

The check only makes sense for **server-side** math renderers
(`mdbook-katex`, `mdbook-pandoc-mathjax`). With those, legit math
gets pre-rendered to `<span class="katex">…</span>` at build time
and only broken math survives as raw `$…$`, making the survival
scan a meaningful signal.

What the existing layer-3 rules (A: em-in-math, B: curly-quote, D:
math-in-attribute) actually detect is the *upstream cause* that
prevents MathJax from processing a math span — not the survival
itself. Those rules are robust under runtime MathJax because they
flag the corruption pattern in the source-of-truth HTML (the
inserted `<em>` tags, curly quotes, attribute placement) regardless
of whether MathJax has run yet. Don't try to "improve" the layer
by adding a $-survival rule on top of these — it will be all noise.

## CI integration

Add to the textbook workflow:

```yaml
- name: LaTeX source lint
  run: |
    cd docs/textbook/lint
    npm ci
    node check.js

- name: Build textbook
  run: mdbook build docs/textbook

- name: LaTeX rendered-HTML lint
  run: python3 docs/textbook/lint/find_em_in_math.py
```

The source lint catches classes C, F, G. The rendered-HTML lint
catches classes A, B, D. Both are required.

## Why not just use a KaTeX preprocessor

A KaTeX-preprocessor approach (e.g., `mdbook-katex`) renders math
at build time, before pulldown-cmark sees it. This would eliminate
class A and B at the cost of an extra binary install.

For a textbook that only targets github.io / mdBook (the recommended
single-surface architecture), `mdbook-katex` is a defensible choice
— it sidesteps the pulldown-cmark italic-eating bug entirely. The
trade-off is an extra preprocessor in the toolchain plus a switch
from MathJax to KaTeX (which has slightly different supported
LaTeX). The MathJax + lint approach in this skill is the path of
least toolchain risk, but `mdbook-katex` is worth considering if
class A bugs become a recurring authoring tax.

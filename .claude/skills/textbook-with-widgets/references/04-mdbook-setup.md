# Phase 3 — mdBook setup

mdBook is the Rust-ecosystem static-site generator for technical
books. Used by *The Rust Programming Language*, the Bevy book, the
nomicon. The right choice for this stack because:

- Plain markdown source — no MDX, no React, no bundler.
- Built-in support for code highlighting (any language).
- Trivial extension via preprocessors (we use mdbook-mermaid).
- Single Rust binary; ~3 MB; installs via `cargo install`.
- HTML output deploys to GitHub Pages or any static host.

## Installation

Three binaries pinned by version:

```sh
cargo install --locked --version 0.5.3 mdbook
cargo install --locked --version 0.17.0 mdbook-mermaid
# Optional: cargo install --locked mdbook-termlink (see references/08-glossary.md)
```

Don't use unpinned versions — mdBook minor versions change CLI flags
and template structure. Cache the binaries in CI (see CI section).

## Layout

```
docs/textbook/
├── book.toml                 # mdBook config
├── SUMMARY.md                # Table of contents (mdBook reads this)
├── theme/
│   └── head.hbs              # Custom <head>: MathJax config, importmap
├── 00_index.md               # Preface
├── 01_*.md ... 18_*.md       # Chapters
├── bibliography.md
├── reference/
│   └── glossary.md           # If using mdbook-termlink
├── widgets/                  # TS widget source files
│   ├── shared/
│   │   ├── widget.js         # defineWidget scaffold
│   │   ├── stepper.js
│   │   ├── helpers.js
│   │   ├── widgets.css
│   │   └── README.md
│   ├── XXX/widget.js
│   └── ...
├── lint/                     # LaTeX validator
│   ├── check.js
│   ├── package.json
│   └── package-lock.json     # COMMIT THIS
└── book/                     # mdBook output (gitignored)
```

## book.toml

Minimum viable config:

```toml
[book]
title = "Your Book Title"
description = "One-line book description, no overclaiming."
src = "."
language = "en"

[build]
build-dir = "book"
create-missing = false

[output.html]
# We load MathJax ourselves in theme/head.hbs to support $…$
# delimiters. mdBook's built-in mathjax-support uses \(…\) only.
mathjax-support = false
# Smart-punctuation OFF or it breaks math. See
# references/05-latex-and-rendering.md.
smart-punctuation = false
default-theme = "navy"
preferred-dark-theme = "navy"
edit-url-template = "https://github.com/<owner>/<repo>/edit/main/docs/textbook/{path}"
git-repository-url = "https://github.com/<owner>/<repo>"
additional-js = ["mermaid.min.js", "mermaid-init.js"]
additional-css = ["widgets/shared/widgets.css"]

[output.html.print]
enable = false

[output.html.fold]
enable = true
level = 1

[preprocessor.mermaid]
command = "mdbook-mermaid"
```

The `additional-js` entries for Mermaid are auto-created by running
`mdbook-mermaid install docs/textbook` once.

## theme/head.hbs (MathJax + importmap)

```html
<!-- MathJax 3 with $…$ inline math, $$…$$ display math. -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
    },
    options: {
      skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    },
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        id="MathJax-script" async></script>

<!-- Importmap for D3 + Observable Plot (used by widgets). -->
<script type="importmap">
{
  "imports": {
    "d3": "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm",
    "@observablehq/plot": "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/+esm"
  }
}
</script>
```

Pin the exact MathJax / D3 / Plot versions — major releases break
APIs. Update intentionally.

## SUMMARY.md

This is mdBook's TOC source-of-truth. Format:

```markdown
# Summary

[Preface and Index](00_index.md)

# Foundations

- [Mathematical Foundations](01_mathematical_foundations.md)
- [The Reinforcement Learning Problem](02_the_rl_problem.md)
- [MDPs and Bellman Equations](03_mdps_and_bellman_equations.md)

# Tabular Methods

- [Dynamic Programming](04_dynamic_programming.md)
- ...

# Reference

- [Bibliography](bibliography.md)
```

`#` headers create unclickable section dividers; `- [name](file)`
creates clickable chapter links. Sub-bullets create nested chapters.

## Local build + serve

```sh
mdbook build docs/textbook
mdbook serve docs/textbook --open    # Live-reload preview
```

`mdbook serve` watches files and rebuilds on save. Open
`http://localhost:3000` in a browser.

## CI deploy to GitHub Pages

Use first-party `actions/*` actions only — no third-party deploy
helpers. See `.github/workflows/textbook.yml` in the source project.
Key shape:

```yaml
name: Textbook

on:
  push:
    branches: [main]
    paths: ["docs/textbook/**", ".github/workflows/textbook.yml"]
  pull_request:
    paths: ["docs/textbook/**", ".github/workflows/textbook.yml"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Cache mdBook binaries
        id: cache-mdbook
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/bin/mdbook
            ~/.cargo/bin/mdbook-mermaid
          key: mdbook-${{ runner.os }}-0.5.3-mermaid-0.17.0
      - name: Install mdBook
        if: steps.cache-mdbook.outputs.cache-hit != 'true'
        run: |
          cargo install --locked --version 0.5.3 mdbook
          cargo install --locked --version 0.17.0 mdbook-mermaid
      - name: LaTeX lint
        run: |
          cd docs/textbook/lint
          npm ci
          node check.js
      - name: Build textbook
        run: mdbook build docs/textbook
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/textbook/book

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

One-time UI step: Settings → Pages → Source: **GitHub Actions**.

The `actions/cache@v4` step makes warm CI runs ~30 seconds total
(mdBook binary restored in <1s; only build + lint actually run).

## Gotchas

- **mdBook copies *everything* under src/ into output/ except .md
  files.** If you put Cargo workspaces under `docs/textbook/`,
  `target/` ends up in `book/`. Either gitignore (`widgets/*/target/`)
  or put source crates OUTSIDE the textbook tree (we used
  `widgets/<name>/` at the repo root for Rust widgets).
- **SUMMARY.md is the source of truth.** Adding a chapter file
  without listing it in SUMMARY.md leaves it unreferenced. mdBook
  doesn't auto-discover.
- **`smart-punctuation = false` is critical for math.** Without it,
  apostrophes inside `$…$` get corrupted into curly quotes that
  MathJax can't parse. See `references/05-latex-and-rendering.md`.

## Output of this phase

A clean local build (`mdbook build` succeeds, no warnings beyond
the pre-existing mdbook-mermaid version skew). Browser preview
loads, MathJax renders, Mermaid renders, chapter cross-links work.

Next: `references/05-latex-and-rendering.md` for validation.

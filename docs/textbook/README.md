# Reading the textbook

Just open the markdown files on GitHub. Math and mermaid diagrams render
natively in the GitHub web UI and the GitHub mobile app — no build step,
no tooling, no extra dependencies.

## On your phone

Bookmark
[`00_index.md`](00_index.md)
in your phone's browser (or the GitHub mobile app). Tap chapter links
to navigate. Done.

## What renders natively on GitHub

- All headings, tables, code blocks, syntax highlighting
- LaTeX math (`$V^\pi(s)$` inline, `$$...$$` display)
- Mermaid diagrams (the chapter dependency graph in `00_index.md`, the
  agent-environment loop in Chapter 4, etc.)
- Cross-references between chapters (`[Ch 6](08_temporal_difference_learning.md)`)
- Anchor links to sections within a chapter
- The per-chapter outline (click the outline icon in the file header)
- Dark mode (GitHub's site-wide theme setting)
- Mobile-responsive layout

## Source files

- [`00_index.md`](00_index.md) — start here. Preface, table of contents,
  notation, citation conventions.
- [`01_linear_algebra.md`](01_linear_algebra.md)
  through [`20_action_spaces.md`](20_action_spaces.md) — the 18 chapters.
- [`bibliography.md`](bibliography.md) — all citations.

## Authoring conventions

**Define every symbol on first use, and Greek letters get both the
LaTeX symbol AND the English name.** A formula like

> The TD update moves $Q$ toward $r + \gamma V(s')$ with step size
> $\alpha$ (alpha), discounted by $\gamma$ (gamma).

is readable; a formula like

> The TD update moves Q toward $r + \gamma V$ with step size $\alpha$.

leaves a reader who saw $\alpha$ for the first time guessing whether
it's "alpha" or "a" (or "α", which they may not be able to type).
Spelling the letter out the first time each chapter introduces it
fixes that.

Same rule for non-Greek constants: state what the symbol means in
prose the first time it appears in a chapter, even if it appeared
elsewhere in the book. Don't make a reader skim back through previous
chapters to remember what $w_{\text{alive}}$ is.

---

## Building the mdBook static site (optional)

GitHub renders the chapters natively, so this is only needed if you
want a hosted, navigable HTML version with sidebar / search / dark
mode toggle — the live deployed site at GitHub Pages (see "Hosting"
below).

### Prerequisites

- Rust toolchain (`cargo --version`).
- `mdbook` binary:
  ```sh
  cargo install --locked mdbook
  ```
  Source build from crates.io — same supply-chain path as every other
  `cargo install` in the repo. Takes ~30 seconds on a warm cache.

### Build

From the repo root:

```sh
mdbook build docs/textbook
```

Output lands in `docs/textbook/book/` (gitignored — regenerate any
time; never commit). Open `docs/textbook/book/index.html` in a
browser to verify.

### Live preview

```sh
mdbook serve docs/textbook --open
```

Watches the source files, rebuilds on save, hot-reloads the browser.
Useful when authoring or tweaking the theme.

### Lint LaTeX before publishing

The textbook ships a Node-based KaTeX + GitHub-context validator that
catches rendering bugs (`\\left` corruption, blockquote display math,
opener-without-leading-space, etc.) before they ever reach a reader.
CI runs it automatically; to run locally:

```sh
cd docs/textbook/lint
npm ci   # first time only
node check.js
```

`OK  2612 math blocks pass …` ⇒ green.

---

## Hosting (GitHub Pages)

The textbook is deployed to GitHub Pages via GitHub's first-party
deploy actions — no `gh-pages` branch, no third-party deploy action.
Deploys are automated; humans never push the site by hand.

### How it works

`.github/workflows/textbook.yml` runs on every push to `main` that
touches `docs/textbook/**`:

1. Installs `mdbook` (cached across runs).
2. Runs the LaTeX validator.
3. Builds `docs/textbook/book/`.
4. Uploads `book/` as a Pages artifact via
   `actions/upload-pages-artifact@v3`.
5. The `deploy` job publishes that artifact via
   `actions/deploy-pages@v4` to the `github-pages` environment.

Both actions are maintained by GitHub itself. The deployed version
shows up under repo **Settings → Environments → github-pages** with
full deploy history + the live URL.

PRs that touch the textbook run the same build + lint, upload the
built site as a workflow artifact for spot-checking, but do **not**
deploy.

### One-time GitHub UI setup

1. Repository → **Settings** → **Pages**.
2. **Source:** "GitHub Actions". Save.

That's it — no branch selection. The first workflow run on `main`
creates the `github-pages` environment automatically and publishes
the site.

The live URL is `https://<owner>.github.io/<repo>/` (visible in
Settings → Pages after the first successful deploy).

### Manually trigger a deploy

If you need to redeploy without a content change (e.g., after fixing
the workflow itself), use the **Actions** tab → "Textbook" → "Run
workflow" (the workflow has `workflow_dispatch:` enabled).

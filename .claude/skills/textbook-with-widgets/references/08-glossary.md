# Phase 5 — Hover-tooltip glossary

Optional but high-value: a glossary that auto-tags terms in chapter
prose with hover-tooltip definitions. No clickable navigation needed
— the reader hovers and sees the definition inline.

## Why automate it

A 18-chapter textbook has hundreds of technical terms. Manually
wrapping each occurrence in markdown is:
- Tedious to write (every chapter).
- Tedious to maintain (renaming a term breaks dozens of links).
- Source pollution (chapter source becomes unreadable).

Better: a glossary file with all terms + definitions, and a
preprocessor that auto-wraps occurrences in HTML elements with
`title="…"` attributes (which browsers render natively as
hover-tooltips with no JavaScript).

## The tool: mdbook-termlink

`mdbook-termlink` is the strongest existing fit for this. It's
an mdBook preprocessor that:
- Reads a glossary file (Markdown definition-list format).
- Walks each chapter at build time.
- Wraps the first occurrence per chapter of each term in HTML
  with a `title` tooltip.
- No custom JavaScript — pure native HTML `title=` attribute.

### Known issue (as of 2026-05)

mdbook-termlink v0.1.1 has a Windows path-resolution bug — it fails
to find the glossary file regardless of relative or absolute path.
Works on Linux/macOS. Track upstream for fixes.

## Setup

Install:

```sh
cargo install --locked --version 0.1.1 mdbook-termlink
```

Add to `book.toml`:

```toml
[preprocessor.termlink]
command = "mdbook-termlink"
glossary-path = "reference/glossary.md"
display-mode = "tooltip"      # native hover tooltips, no link navigation
```

`display-mode = "tooltip"` shows the definition on hover without
making the term clickable. The other options (`link` / `both`)
make it click-through to a glossary page.

## Glossary format

A standard Markdown definition list at
`docs/textbook/reference/glossary.md`:

```markdown
# Glossary

α (alpha)
: The TD learning rate. How much each update moves Q toward its
  target. Higher α = faster learning but noisier. In
  `LearningConfig.alpha`.

γ (gamma)
: The discount factor, γ ∈ [0, 1). Future rewards weighted by γ^k.
  Effective horizon = 1 / (1 - γ).

TD
: Temporal-Difference learning. Updates Q toward a target that
  mixes the immediate reward and a bootstrap from the current Q
  estimate.

MDP
: Markov Decision Process. The formal model: states, actions,
  transition probabilities, rewards, discount factor.

POMDP
: Partially Observable MDP. The agent sees observations, not the
  full state.

Q-learning
: TD control where the bootstrap uses the greedy next action:
  `Q ← Q + α(r + γ max_a Q(s', a) - Q)`.
```

Notes:
- Term on one line, `: definition` on the next (with leading space
  for indent).
- Definition can span multiple lines as long as each starts with
  a space.
- Term names match exactly (case-sensitive unless configured
  otherwise via `case-sensitive = false` in book.toml).

## What gets tagged

The preprocessor auto-wraps the FIRST occurrence per chapter (so the
chapter doesn't become a sea of tooltip dots). Subsequent uses are
plain text. This is configurable via `link-first-only = true/false`
in book.toml.

A reader hovers the wrapped term → browser shows the `title="…"`
contents as a tooltip. Native HTML; no JavaScript.

## CSS for the tooltip target

The preprocessor wraps with `<span class="glossary-term" title="...">`
(or similar). Add minimal CSS to widget.css to make wrapped terms
visually discoverable (e.g., dotted underline):

```css
.glossary-term {
  border-bottom: 1px dotted #888;
  cursor: help;
}
```

Hover behavior is native browser tooltip — no custom JS, no Tippy.js.

## Alternative if you really do want JS-powered popovers

Native HTML `title=` is plain text only (no formatting, no links
inside tooltips). If you want rich popovers (formatted definitions,
multiple paragraphs, links), you need JS — typically
[Tippy.js](https://atomiks.github.io/tippyjs/) plus a small init
script.

Architecture in that case:
1. mdbook-termlink wraps terms with `data-tippy-content="…"` or
   similar attribute.
2. A small JS file (loaded via `additional-js`) reads the data
   attribute, creates Tippy instances on each.
3. CSS for the tooltip popover styling.

Trade-off: ~10 KB of JS (Tippy minified) for richer tooltips. For
most textbooks, native HTML tooltips are enough.

## Maintenance

The glossary is the source of truth. To add a term:

1. Edit `docs/textbook/reference/glossary.md`. Add the new entry.
2. `mdbook build`. The preprocessor re-tags chapters.
3. Done — no chapter edits needed.

To rename a term:

1. Update `glossary.md`.
2. Use a one-shot script to rewrite occurrences in chapter
   markdown.
3. Rebuild.

The "rename" case is the rare one. For ongoing use, you mostly
just add entries.

## Output of this phase

A `reference/glossary.md` file, a `[preprocessor.termlink]` entry
in book.toml, native hover tooltips on the first occurrence of each
term in each chapter.

Browser test: open a chapter, hover an α, see "The TD learning rate
…" pop up.

## When to skip this

Don't bother if:
- The textbook has fewer than ~20 distinct technical terms.
- Readers have the terms memorized from prior coursework.
- You haven't shipped the rest of the chapters yet (do this last,
  not first).

Glossaries are a polish step, not a correctness step. The textbook
works without it; it just makes the prose more skim-friendly for
non-experts.

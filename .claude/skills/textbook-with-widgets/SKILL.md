---
name: textbook-with-widgets
description: How to take a syllabus on a technical topic and convert it into a complete interactive textbook with step-by-step chapters, math notation, interactive widgets for every concept, and live-coding exercises that auto-check the student's implementation. Use this skill whenever the user wants to build educational content from a topic outline, expand an existing curriculum into prose, write or revise a technical book chapter, set up mdBook with math + diagrams, design interactive learning widgets (drag-this-see-that visualizations, step-by-step animations, parameter sliders), add live coding exercises where the student fills in a function and gets pass/fail feedback (backprop, value iteration, REINFORCE, optimiser steps), pick an architecture for math viz versus algorithm reuse, validate LaTeX rendering across markdown and HTML, or organize a glossary with hover tooltips. Trigger even when the user just says "I want to learn X" with a roadmap-style list — that's a syllabus in disguise. Also trigger for adjacent asks like "make this chapter interactive", "let the reader implement the algorithm", "live coding exercises in a textbook", "what's the right way to ship interactive math demos", "how do I structure an RL textbook chapter", etc.
---

# Textbook + interactive widgets

This skill captures a complete, tested workflow for converting a topic
syllabus into a published interactive textbook. The output is an
mdBook site with math, diagrams, code-grounded prose, and per-concept
interactive widgets — the kind a serious learner uses to actually
understand a hard topic.

The skill is opinionated. The opinions are earned: every pattern here
was forced by a real bug, a real authoring-friction moment, or a real
pedagogical failure during the source project's textbook build. See
`references/` for the long-form context and `examples/` for excerpted
code.

## How to use this skill

Pick the right entry point depending on where the user is:

- **They have only a topic or a one-liner ("I want to learn RL").** Go
  through `references/01-syllabus.md` and produce a chapter-by-chapter
  syllabus first. Don't skip — the syllabus drives every later choice.
- **They have a syllabus or an outline.** Go to
  `references/02-research.md` and `references/03-chapter-structure.md`
  to expand a single chapter end-to-end.
- **They have draft chapters and want to publish.** Go to
  `references/04-mdbook-setup.md` (build pipeline) and
  `references/05-latex-and-rendering.md` (avoiding the rendering bugs
  that broke our source project).
- **They want widgets / interactivity.** Read
  `references/06-widgets-architecture.md` for the language-choice
  framework and `references/07-widget-authoring.md` for the
  `defineWidget` pattern.
- **They want live coding exercises** (the student fills in a
  function, the harness auto-checks). Read
  `references/11-exercise-authoring.md` for the `defineExercise`
  pattern, the three check kinds (`fixedTable`, `gradCheck`,
  `reference`), the `signature` block convention (type hints +
  auto-derived sample-input disclosure), the CodeMirror-6
  syntax-highlighting upgrade (lazy-loaded from esm.sh with a
  textarea fallback), and the smoke-test discipline. This is the
  right surface for "implement backward for tanh," "type one
  TD(0) update," "implement Adam's bias correction" — load-bearing
  algorithms whose code IS the algorithm.
- **They want a glossary with hover-tooltips.** See
  `references/08-glossary.md`.

## The core workflow at a glance

```
syllabus (chapter list with one-line goals)
  ↓ expand each chapter
research notes (sources, key results, project tie-ins)
  ↓ structure
chapter markdown (Why-this-chapter + sections + project tie-in + exercises + citations)
  ↓ validate
mdBook build + LaTeX validator + link checker
  ↓ interactivity
per-concept widgets (TS for math, Rust+WASM for algorithm reuse)
  ↓ active recall
live coding exercises (defineExercise with auto-check) on the
  load-bearing algorithm in each chapter
  ↓ author conventions
glossary tooltips, footer cross-links, accessibility
```

Each arrow is one step. Each step has a reference file with the
operational details.

## Skill philosophy (read this once)

A few non-negotiable principles that shape every decision below.
They came out of trial-and-error; honour them and the textbook will
be navigable and trustworthy.

### 1. No overclaiming on authorship or level

If the user is compiling material from primary sources for their own
learning, **do not market the result as "master's-level" or list an
author.** The honest framing is: "a self-study compilation, content
restated from primary sources by [list authors]". Read the originals
for authority; this exists to be the explanation the author wished
they had. Cite generously. This calibration matters: it shapes the
reader's expectations and the tone you write in.

### 2. Define every symbol on first use; Greek letters get both forms

When introducing a Greek letter in prose, spell out the English
name the first time *per chapter*. Write:

> The TD update moves Q toward `r + γV(s')` with step size α (alpha),
> discounted by γ (gamma).

Don't write:

> The TD update moves Q toward `r + γV(s')` with step size α.

This lets a reader who sees α for the first time know whether it's
"alpha" or "a" or "α" (which they may not be able to type). Same rule
for non-Greek constants — state what a symbol means in prose the
first time it appears in a chapter, even if you defined it elsewhere.

### 3. Ground every concept in real artifacts

If a chapter discusses a concept the project actually implements,
**link directly to the source file**. Don't paraphrase the code —
point at it. The textbook's authority comes from "this is implemented
in [`crates/engine/q_learning/src/value_function.rs:42`](...)" not
from how authoritatively the prose claims something works. When
references go stale, fix them — don't paper over with rewording.

### 4. Mixed-language architecture for widgets

Math visualisations should be TypeScript. Algorithm reuse should be
Rust compiled to WASM. The split-by-purpose rule shipped 43 widgets
without architecture fights — see `references/06-widgets-architecture.md`
for the decision tree.

### 5. "Try it" before "what is it"

Every concept that has an obvious manipulable parameter gets a
widget. The widget appears AFTER the formula, BEFORE the prose
explanation of the formula's meaning. Order matters: the reader
sees the math, plays with it, then reads the explanation with
intuition already loaded.

### 6. Type it before you've earned it

For *load-bearing* algorithms — the lines of code the chapter is
really about — pair the widget with a live coding exercise. A
slider proves *that* backprop works; typing `dx = dy * (1 - y²)`
yourself proves *what* backprop is.

The same restraint as widgets applies: not every chapter gets one.
Pick the algorithm whose code IS the algorithm — backward for one
op, one TD(0) update, one Bellman sweep, one Adam step. Five to
twenty lines, no fancy data structures. If a reader can read the
prose, slide the widget, AND type the algorithm with their own
hands, the concept is genuinely theirs.

Practical surface: `defineExercise` in
`docs/textbook/widgets/shared/exercise.js`. Authoring guide:
`references/11-exercise-authoring.md`. Smoke-test (run after every
authoring change): `node docs/textbook/widgets/shared/smoketest.js`.

### 7. Validate the rendered output, not just the source

The source-level LaTeX validator catches most bugs. It does not
catch rendering-time corruption like smart-punctuation breaking
math inside `$…$` blocks. Always view the built HTML at least once
per chapter — search for raw `$` outside `<pre>`/`<code>` tags.
This caught real bugs in our project. See
`references/05-latex-and-rendering.md` for the full checklist.

### 8. Don't inline widget code in chapter markdown

mdBook's markdown parser mangles multi-line `<script>` blocks
(strips them, converts quotes, breaks template literals). Always
ship widgets as separate `.js` files with a two-line embed in the
chapter:

```markdown
<div id="ch3-foo-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/foo/widget.js"></script>
```

Authors who need to edit the widget go to one file; readers who
just need the prose aren't distracted by 80 lines of D3 mid-chapter.

## File map

```
SKILL.md (this file)
references/
├── 01-syllabus.md             How to design / expand a syllabus
├── 02-research.md             Source discovery + citation conventions
├── 03-chapter-structure.md    The chapter template that works
├── 04-mdbook-setup.md         book.toml, theme/head.hbs, CI deploy
├── 05-latex-and-rendering.md  Validator + rendered-HTML checks (start here)
├── 06-widgets-architecture.md TS vs Rust+WASM decision tree
├── 07-widget-authoring.md     `defineWidget`/`defineStepper` patterns
├── 08-glossary.md             Tooltip glossary via mdbook-termlink
├── 09-rendering-bug-classes.md Catalogue of 7 known bug classes
├── 10-widget-correctness-audit.md Static audit checklist for "slider
│                                   moves but plot doesn't" bugs
└── 11-exercise-authoring.md   `defineExercise` pattern, three check
                                kinds, smoke-test discipline
examples/
├── chapter-template.md        A blank chapter ready to fill in
├── widget-template.js         Smallest-possible TS widget
└── exercise-template.js       Smallest-possible live coding exercise
```

Read each reference file as the workflow takes you to it. They are
self-contained and you can skim — none is longer than ~150 lines.

## Quick start — minimum viable textbook in 60 minutes

If the user wants to ship something **today**:

1. Get the syllabus (existing or generated). Save as
   `<book>/reinforcement_learning_syllabus.md` (or equivalent).
2. Pick chapter 1 of the syllabus. Open
   `examples/chapter-template.md` and copy structure.
3. Set up mdBook: `cargo install mdbook mdbook-mermaid`, write a
   minimal `book.toml` per `references/04-mdbook-setup.md`.
4. Run `mdbook build`. Verify the chapter renders. Open the HTML in
   a browser and CHECK for surviving `$` signs in math contexts.
5. Add one widget per concept that has a parameter — start with the
   simplest formula. Use `defineWidget` per
   `references/07-widget-authoring.md`.
6. *Optional, for chapters with a load-bearing algorithm:* add one
   live coding exercise via `defineExercise`. See
   `references/11-exercise-authoring.md`. Pick the algorithm whose
   code is the chapter's whole point.

Time budget: ~10 min on setup, ~30 min on the chapter, ~20 min on
two widgets, +20 min if you add an exercise. The rest of the book is
just repetition of this loop.

## When to NOT use this skill

- Pure prose books (novels, essays). Use a normal Markdown workflow.
- Single-page documentation. mdBook is overkill; use a `README.md`.
- API references. Use the language's idiomatic doc tool (`cargo doc`,
  `rustdoc`, `typedoc`, etc.).
- Live training material that needs interactivity but no math.
  Consider Observable notebooks or Astro — mdBook's math-first
  defaults are wrong for code-first content.

If in doubt: skim `references/04-mdbook-setup.md` first. If the
build-step ergonomics don't fit the user's goal, push back and
suggest the appropriate tool.

## Anti-patterns to call out before they happen

The patterns below are seductive shortcuts that bite later. If you
catch the user (or yourself) heading toward one, redirect.

- **Hand-rolling axes/grids in every widget.** Use Observable Plot
  — it has scales, axes, ticks, and tooltips for free. Re-inventing
  these costs ~80 LOC per widget and doesn't even look as good.
- **Vendoring D3 inline as `d3.min.js`.** D3 ships as a re-export
  shim, not a bundle. Vendoring requires `esm.sh`'s bundled output
  or a build step. Use the CDN with a pinned version unless you have
  a concrete offline requirement.
- **Importing TypeScript directly in browsers.** Browsers don't
  execute `.ts`. You need a transpile step. Default to plain `.js`
  with JSDoc types for the no-build authoring story; add `tsc
  --noEmit` if you want type checking in CI.
- **Writing widget logic in chapter markdown.** mdBook's parser
  WILL mangle it. Always ship widget JS in a separate file.
- **Using `:` (colon) anywhere in chapter headers that has math
  inside it.** mdBook's anchor generator strips characters and the
  resulting anchor URL is unstable across mdBook versions. Plain text
  headers with math in subheaders works.
- **Smart-punctuation enabled.** mdBook's default turns `'` into `'`
  inside math. Set `smart-punctuation = false` in `[output.html]`.
- **Letting agents edit the same chapter file concurrently.** If you
  delegate widget implementation to parallel sub-agents, give each
  agent one chapter to own. Same-file concurrent edits race.
- **Custom JavaScript tooltip libraries for the glossary.** mdBook
  preprocessors that emit native HTML `title` attributes (e.g.
  `mdbook-termlink`) give hover tooltips for free with no JS.
- **Live coding exercises that don't verify their own solutions.**
  If an exercise ships a `fixedTable` with hand-computed `expected`
  values and no `solution`, the reader is the first person to test
  it — and reports a "broken exercise" issue when their correct
  code fails the author's wrong arithmetic. Always write a
  `solution` and run `node docs/textbook/widgets/shared/smoketest.js`
  before shipping. The smoketest runs every exercise's solution
  through its own check; if the author's reference disagrees with
  the test data, it fails loudly.
- **Live coding exercises in every chapter.** Same restraint as
  widgets: pick the load-bearing algorithm where typing it
  teaches what reading it can't. From a 21-chapter textbook, ~9
  exercises is enough. More clutters the chapter and dilutes the
  signal.
- **Hand-rolled `<textarea>` + `eval` for exercises.** Use the
  `defineExercise` scaffold — it handles localStorage draft
  persistence, the syntax-error-friendly compile path, the
  pass/fail diff display, and the "Show solution" reveal. ~30
  lines of authoring per exercise.

## Validation gate before each commit

The textbook's single rendering target is mdBook output deployed to
github.io. github.com's raw-markdown render is **not** a supported
surface — pick one target and optimise for it, otherwise you can't use
techniques like `\_` escaping (which works in mdBook+MathJax but looks
ugly in GitHub's renderer).

**Three layers. All required.** Plus a fourth that only fires if the
chapter ships a live coding exercise.

```sh
# 1. Pre-build syntax check (catches KaTeX parse errors + parser-
#    fragile patterns like display-math-in-blockquote, mismatched-brace,
#    `R_\max` strict-mode failures).
node docs/textbook/lint/check.js

# 2. mdBook build (catches config bugs, link rot, missing assets).
mdbook build docs/textbook

# 3. Rendered-HTML check (catches bugs ONLY visible in the built
#    HTML — pulldown-cmark eating `_` as italic inside math,
#    smart-quote contamination, math-in-attribute). This is what
#    the deployed site actually contains.
python3 docs/textbook/lint/find_em_in_math.py

# 4. Exercise self-check (catches typos in expected values and
#    drift between an exercise's solution and its check block).
#    Only relevant when the change touches widgets/*/exercise.js.
node docs/textbook/widgets/shared/smoketest.js
```

Layer 3 is the critical insight: **the pre-build check and the mdBook
build can BOTH be green while the deployed site has broken math.**
The author project shipped 23 broken display-math formulas under
exactly this scenario — pre-build check: clean; build: no warnings;
deployed site: raw `$$...$$` text on every affected page. Layer 3
caught all 23 the moment it was added.

For larger changes (new chapter, new widget batch), also do:

```sh
# Open a chapter in the browser, move every widget slider through
# its range, verify the plot redraws live. Static audit (Bug 5 in
# references/10-widget-correctness-audit.md) misses hardcoded
# y-domain clipping; only a visual pass catches it.
mdbook serve docs/textbook --open
```

The `references/09-rendering-bug-classes.md` file enumerates seven
known bug classes with detection rules + fixes. Read it before
debugging any "MathJax says it's broken" issue.

For widget audits, `references/10-widget-correctness-audit.md` has
the static-analysis checklist. Run it after every widget batch.

For exercise authoring, `references/11-exercise-authoring.md` has
the `defineExercise` pattern, the three check kinds, and a list of
pitfalls that bit the source project. Read it once before authoring
the first exercise; refer back per-pitfall when something fails.

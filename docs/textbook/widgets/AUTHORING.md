# Authoring widgets for the textbook

A widget is a small interactive visualisation embedded in a chapter.
The reader sees a formula, plays with sliders, watches the chart
respond. There are ~100 widgets across the 21 chapters; this guide
documents the conventions that let someone with no prior context
write one in 30-60 minutes.

If you just want to add a widget to a chapter you're writing, jump
straight to the **Quick start** below. If you want to understand
*why* we do things this way (and avoid the bug classes that bit us
multiple times), read on.

## Quick start: a minimal widget in 5 steps

1. **Pick a chapter section** where the math is interesting and the
   reader will want to "see" the formula. Examples of good targets:
   any formula with a tunable parameter; any algorithm whose
   convergence depends on a hyperparameter; any geometric object
   the reader needs to visualise.

2. **Create a widget directory**:

   ```
   docs/textbook/widgets/<your_widget_name>/widget.js
   ```

3. **Write the widget** using the `defineWidget` scaffold:

   ```js
   import * as Plot from "@observablehq/plot";
   import { defineWidget } from "../shared/widget.js";
   import { plotDefaults, palette } from "../shared/helpers.js";

   defineWidget({
     hostId: "chN-your-widget-id",            // matches chapter mount
     controls: {
       alpha: { label: "α", min: 0, max: 1, step: 0.01, default: 0.1 },
     },
     render: (host, { alpha }, slots) => {
       const data = /* ... compute from alpha ... */;
       slots.main.replaceChildren(Plot.plot({
         ...plotDefaults,
         marks: [Plot.line(data, { x: "x", y: "y", stroke: palette.primary })],
       }));
       slots.readout.textContent = `α = ${alpha.toFixed(2)}`;
     },
   });
   ```

4. **Mount in chapter markdown** (two lines + a help-text paragraph):

   ```markdown
   ### Try it: the alpha thing

   <div id="chN-your-widget-id" class="textbook-widget"></div>
   <script type="module" src="./widgets/your_widget_name/widget.js"></script>

   Slide α and watch the curve flatten. At α near 0 the update barely
   moves; at α near 1 the curve overshoots and bounces. The sweet
   spot is around 0.1 — which is why the Simulator uses that value.
   ```

5. **Verify**:

   ```bash
   mdbook serve docs/textbook
   # open http://localhost:3000 and check the widget renders.
   ```

That's it. The rest of this guide explains the building blocks.

## File layout

```
docs/textbook/widgets/
├── AUTHORING.md              ← you are here
├── README.md                 ← short pointer to this file
├── shared/                   ← the toolkit
│   ├── helpers.js            ← palette, plotDefaults, readNumber, fmt
│   ├── widget.js             ← defineWidget scaffold
│   ├── stepper.js            ← defineStepper for trajectory widgets
│   ├── matrix_editor.js      ← buildMatrixEditor, buildMatrixDisplay
│   ├── random.js             ← mulberry32, gauss, sampleDiscrete
│   ├── linalg.js             ← eig2x2, svd2x2, matvec2x2, powerIterate
│   ├── plot.js               ← bandAxis, gridAxes, unitBarsY, ribbonX
│   └── widgets.css           ← shared styles for widget DOM
└── <your_widget>/
    └── widget.js             ← one file per widget
```

The convention is **one widget per directory**, named after what it
illustrates. The directory holds at least `widget.js`; if the widget
needs additional assets (a CSV of data, a sprite sheet), they live
alongside.

## The two scaffolds

### `defineWidget` — declarative widgets

The most common pattern. Declare the controls and write a render
function; the scaffold handles wiring inputs to render calls.

```js
import { defineWidget } from "../shared/widget.js";

defineWidget({
  hostId: "ch3-bellman-widget",
  controls: {
    gamma: { label: "γ", min: 0, max: 0.99, step: 0.01, default: 0.9 },
    epsilon: {
      type: "select",
      label: "exploration",
      options: [
        { value: "greedy", label: "ε = 0 (greedy)" },
        { value: "low",    label: "ε = 0.01" },
        { value: "high",   label: "ε = 0.10" },
      ],
      default: "low",
    },
  },
  slots: ["main", "table"],   // optional; default is ["main"]
  render: (host, { gamma, epsilon }, slots) => {
    // ... compute data from params ...
    slots.main.replaceChildren(Plot.plot({ ... }));
    slots.table.replaceChildren(buildTable(...));
    slots.readout.textContent = `summary: ...`;
  },
});
```

**Control types**:
- Default (range slider): `{ label, min, max, step, default }`.
- Select dropdown: `{ type: "select", label, options: [{value, label}, ...], default }`.

**Slots**: `slots.main` is always present; declare extras by passing
`slots: ["main", "extra1", ...]`. Each becomes a `<div data-slot="...">`
inside the widget host. Always present (no need to declare):
`slots.readout` — a span inside the controls row for a short status
line.

**Render contract**:
- `render` is called once on mount and again on every control change.
- The function should be **pure**: same params → same DOM. Don't keep
  module-level state that mutates across calls.
- `params` is a fresh object on every call. Destructure freely.

### `defineStepper` — trajectory / animation widgets

For widgets that step through a discrete sequence of frames — value
iteration, cobweb iteration, episode replay, power iteration, etc.

```js
import { defineStepper } from "../shared/stepper.js";

defineStepper({
  hostId: "ch6-vi-widget",
  controls: {
    gamma: { label: "γ", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["map", "delta"],
  // Build the sequence of frames up front from the params.
  trajectory: ({ gamma }) => {
    const frames = [{ k: 0, V: initialV() }];
    for (let k = 1; k <= MAX_SWEEPS; k++) {
      frames.push({ k, V: bellmanSweep(frames[k-1].V, gamma) });
    }
    return frames;
  },
  // Called for each frame as the user steps through.
  render: (host, frame, idx, total, params, slots) => {
    slots.map.replaceChildren(buildHeatmap(frame.V));
    slots.readout.textContent = `sweep k = ${frame.k} / ${total - 1}`;
  },
  playIntervalMs: 500,  // optional auto-play speed
});
```

The scaffold provides prev/play/pause/next/reset buttons. The
trajectory is rebuilt whenever a control changes; the frame index
resets to 0.

## Available helpers (the toolkit)

### `helpers.js` — palette, defaults, formatting

```js
import {
  palette,        // { primary, secondary, warning, danger, accent, muted }
  plotDefaults,   // spread into Plot.plot({ ...plotDefaults, marks: [...] })
  dashed,         // { strokeDasharray: "4 2" } — spread into a Plot mark
  annotation,     // { fontSize: 10, fill: palette.muted } — for Plot.text
  fmt,            // pretty-format a number, handling tiny / large / NaN
} from "../shared/helpers.js";
```

**Palette colour choices** — picked for dark-mode legibility and
colour-blind safety. Use the names, not raw hex:
- `palette.primary` — main series, "successful" / dominant signal.
- `palette.secondary` — second series, "info" / comparison signal.
- `palette.warning` — orange — highlights, edge cases, annotations.
- `palette.danger` — red — thresholds, errors, divergence.
- `palette.accent` — purple — third series.
- `palette.muted` — grey — reference lines, faint guides.

### `random.js` — seeded RNGs

```js
import { mulberry32, lcg, splitmix64, gauss, sampleDiscrete } from "../shared/random.js";

const rng = mulberry32(seed);  // canonical small-state PRNG (default choice)
//        = lcg(seed)          // tiny LCG; smaller but more periodic
//        = splitmix64(seed)   // 64-bit splittable; BigInt, ~3× slower
const u  = rng();              // uniform [0, 1)
const z  = gauss(rng);         // standard normal
const i  = sampleDiscrete(rng, [0.5, 0.3, 0.2]);  // index 0, 1, or 2
```

**Always seed your noise.** If a widget calls `Math.random()` inside
`render()`, every slider tick re-rolls the noise — the user moves γ
and the noisy bars jitter for unrelated reasons (bug class 13 in the
audit checklist). Use `mulberry32(seed)` with either a fixed seed or
an explicit `seed` slider.

### `linalg.js` — 2×2 linear algebra

```js
import {
  eig2x2, svd2x2,
  det2x2, trace2x2, inv2x2, mul2x2, matvec2x2,
  powerIterate,
} from "../shared/linalg.js";

const { real, eigs } = eig2x2(A);
if (real) {
  for (const { lambda, v } of eigs) { /* ... */ }
}

const { sigma, v } = svd2x2(A);  // returns [σ1, σ2] and right-singular vectors

const trajectory = powerIterate(A, [1, 0], 20);  // sequence of unit vectors
```

Most pedagogical widgets in this textbook work on 2×2 matrices
(small Markov chains, the 2D plane). If you need larger matrices,
write the routine inline or pull in `d3-array`.

### `plot.js` — Plot-mark helpers (band scales, ribbons)

```js
import { bandAxis, gridAxes, unitBarsY, ribbonX } from "../shared/plot.js";

// Square gridworld heatmap (Plot.cell needs band scales)
Plot.plot({
  ...gridAxes(N, { axis: null }),     // x and y both band over 0..N-1
  marks: [Plot.cell(cells, { x: "c", y: "r", fill: "v" })],
});

// Geometric-decay bar chart with continuous integer x
Plot.plot({
  ...plotDefaults,
  marks: [unitBarsY(data, { x: "k", y: "y", fill: palette.primary })],
});

// 1D ribbon of coloured cells along continuous x
Plot.plot({
  ...plotDefaults,
  x: { label: "t", domain: [0.5, N + 0.5] },
  y: { axis: null, domain: [0, 1] },
  marks: [ribbonX(data, { x: "t", fill: "logw" })],
});
```

### `matrix_editor.js` — editable matrix UI

For widgets that ask the reader to *edit* a matrix's entries (rather
than just slide one parameter):

```js
import { buildMatrixEditor } from "../shared/matrix_editor.js";

const A = [[1.5, 0.5], [0.3, 1.2]];
buildMatrixEditor(host, '[data-matrix="A"]', A, render, {
  step: 0.1,
  colHeaders: ["col 1", "col 2"],
});
```

Wiring is hand-rolled inside the widget HTML; `buildMatrixEditor`
replaces the contents of the selector with a grid of `<input>`s
bound to `A[i][j]`. The widget calls `render(A)` on every keystroke.

## Authoring conventions

### 1. Mount block in chapter markdown

Always two lines, in this exact form:

```markdown
<div id="chN-your-widget-id" class="textbook-widget"></div>
<script type="module" src="./widgets/your_widget_name/widget.js"></script>
```

- The `id` matches the `hostId` passed to `defineWidget`.
- The widget id format is `ch{N}-{name}-widget` where N is the
  chapter number (1-indexed) and `name` is a few hyphen-separated
  words.
- The script `src` path is relative to the chapter file.

### 2. Help text below the widget

Every widget mount MUST be followed by a 2-5 sentence student-facing
paragraph. Tell the reader:
- What to play with (which control changes which signal).
- What to look for on screen.
- The pedagogical point — why this specific manipulation matters.

Good help text (from `01_linear_algebra.md`):

> Edit the four entries of A. The widget rotates every blue unit
> vector into the orange image vector. The green directions are the
> eigenvectors: A only stretches them (the dashed green arrow is
> Av = λv along the same line as v). Try `a=2, d=3, b=c=0` for the
> diagonal example above; try `b=-c` for a rotation — *no real
> eigenvectors*.

Generic filler is not help text:

> ❌ Explore the widget and try different settings.

### 3. Section headings

The convention is `### Try it: <short verb-phrase>`. Examples:

- `### Try it: eigenvectors of a 2×2 matrix`
- `### Try it: gradient descent on three surfaces`
- `### Try it: the contraction in 1D`

These headings populate the chapter's table of contents; keep them
short so the sidebar stays readable.

## Pitfalls (the bug catalogue)

These are real bugs that have shipped in this textbook. The skill's
[10-widget-correctness-audit.md](../../../.claude/skills/textbook-with-widgets/references/10-widget-correctness-audit.md)
has the long version with detection rules; here's a fast checklist.

0. **Matrix rows collapse to one row (`\\` eaten in display math).**
   Inside `$$…$$`, pulldown-cmark consumes one of the two backslashes
   in `\\` (the LaTeX row separator). The HTML gets `\ `, MathJax
   reads it as escape-space, and the matrix shows on a single row.
   Fix: write `\\\\` (four backslashes) in source for every row
   separator inside `pmatrix`, `bmatrix`, `aligned`, `cases`, etc.
   The script `docs/textbook/lint/fix_matrix_rowsep.py` fixes all of
   them at once and is idempotent.

1. **`Plot.barY` on a continuous numeric x → zero-width bars.**
   Use `Plot.rectY` with `interval: 1`, OR use `bandAxis(N)` for an
   ordinal x. Same trap for `Plot.barX` on continuous y.

2. **`Plot.cell` on a continuous numeric x/y → zero-size cells.**
   Use `gridAxes(N)` (the helper takes care of band scales). For
   1D ribbons where you want a continuous x, use `ribbonX(data, ...)`
   from `shared/plot.js`.

3. **Array mutation via `.reverse()` / `.sort()` on a module-level
   const.** These mutate in place and return the same reference. If
   you pass the module-level `STATE_NAMES.reverse()` as a Plot scale
   domain, every subsequent `STATE_NAMES[i]` lookup gets the
   *reversed* name. Use `.slice().reverse()` to copy first.

4. **Slider default outside `[min, max]`.** Browsers silently clamp,
   so the visible default lands at `max` and you'd be confused that
   the slider has no headroom. Always have `min ≤ default ≤ max`.

5. **Unseeded `Math.random()` inside render.** Every slider tick
   re-rolls the noise — the chart jitters for reasons the user
   didn't trigger. Seed with `mulberry32` (and either fix the seed
   or expose a `seed` slider).

6. **Convergence loop cap pegged at iteration max.** If you have a
   `delta < 1e-6` test inside a `MAX_OUTER = 200` loop, and the
   slider can push the contraction rate close to 1, you'll peg the
   cap and the displayed iteration count becomes constant in the
   slider. Set `MAX_*` to be larger than `log(EPS)/log(γ_max)` for
   the slider's high end.

7. **Hard-coded plot y-domain clipping the data.** If the slider
   moves the data, the data may exit `domain: [0, 0.3]`. Prefer
   `zero: true` (anchor at 0, Plot auto-fits the top) over a fixed
   `domain`.

8. **Polyline endpoint outside the chart's domain.** Plot clips
   segments at the domain boundary. A cobweb staircase starting at
   `(x₀, 0)` when y=0 sits below the chart will appear to "reset to
   zero" at one end. Start at a coordinate that's actually on the
   chart.

## Linting

Three layers, all required (CI runs all three on PRs):

```bash
# 1. Source-level KaTeX + pulldown-cmark context check.
node docs/textbook/lint/check.js

# 2. mdBook build (catches config bugs, link rot, missing assets).
mdbook build docs/textbook

# 3. Rendered-HTML check — catches bugs only visible after pulldown-
#    cmark runs (e.g. <em> tags inside math because of `_` pairing).
python3 docs/textbook/lint/find_em_in_math.py
```

For post-MathJax issues (raw `$X$` patterns that survived to the
client-rendered DOM), use the iframe-based scanner saved at
[`lint/post_render_scan.js`](../lint/post_render_scan.js). Paste it
into the browser console while `mdbook serve` is running.

## Authoring live coding exercises

Sibling pattern to `defineWidget`: `defineExercise` ships the reader
a textarea + a Run button + automatic check against the student's
code. Use it when "type this yourself" understands the algorithm
*differently* from "drag a slider that calls it." The scaffold lives
in `shared/exercise.js`; this section is the authoritative authoring
guide (the original design proposal has shipped and been retired).

### Quick start

```
docs/textbook/widgets/<exercise_name>/exercise.js
```

```js
import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch11-tanh-backward-exercise",
  title: "Implement backward for tanh",
  prompt: `
    The forward computes \`y = tanh(x)\`. Implement the backward.

    Chain rule: **\`dx = dy · (1 - y^2)\`**. Type it.
  `,
  // Type signature for the function args. Renders as a monospace
  // pre under the prompt. Column-align `:` and `—` for readability.
  signature: `
dy  : Float32Array(N)   — upstream gradient ∂L/∂y, one value per element
y   : Float32Array(N)   — cached forward output (already = tanh(x))
→   : Float32Array(N)   — return ∂L/∂x; same length as dy and y
  `,
  template: `function backwardTanh(dy, y) {
  const dx = new Float32Array(dy.length);
  // your code here

  return dx;
}`,
  entrypoint: "backwardTanh",
  entrypointArgs: ["dy", "y"],
  check: {
    kind: "gradCheck",
    forward: (x) => Math.tanh(x),
    inputs: { length: 16, range: [-3, 3] },
    nTests: 5,
    tolerance: 1e-3,
    seed: 7,
  },
  solution: `function backwardTanh(dy, y) {
  const dx = new Float32Array(dy.length);
  for (let i = 0; i < dy.length; i++) dx[i] = dy[i] * (1 - y[i] * y[i]);
  return dx;
}`,
});
```

The harness reads `signature` and renders it as a small type-hint
block between the prompt and the editor. It *also* auto-generates a
collapsed "▸ Show example input" disclosure that — when expanded —
dumps the actual values the harness will pass on test 1, formatted
with the typed-array constructor name (`Float32Array(16) [0.34,
-0.85, ...]`). The reader can match their function parameters
against real numbers, not just types. No authoring effort: the
sample is derived from `check.cases[0]` (fixedTable), `check.generate
(rng, 0)` (reference), or seeded synthesis through `check.forward`
(gradCheck).

Mount in chapter markdown:

```markdown
### Try implementing it: backward for tanh

<div id="ch11-tanh-backward-exercise"></div>
<script type="module" src="./widgets/tanh_backward/exercise.js"></script>

A 2-3 sentence pedagogical note about *why* this exercise matters.
```

### The three check kinds

1. **`fixedTable`** — `cases: [{ input: [...args], expected: ... }, ...]`.
   Boring-but-rock-solid. Right when the answer is a small known
   value (one TD(0) update, one Bellman sweep on a known MDP).

2. **`gradCheck`** — finite-difference numerical derivative.
   Author provides `forward: (x) => …`; the harness picks random
   `dy`, computes `(forward(x+ε) - forward(x-ε)) / 2ε` per element,
   and compares to the student's analytic backward. Perfect for
   "implement backward for foo" because the *textbook claim itself*
   ("backprop matches calculus") is the test. Author tunes
   `tolerance` (typical: 1e-3) and `epsilon` (typical: 1e-4).

3. **`reference`** — call an oracle author-supplied as `reference: (...args) => …`
   on inputs generated by `generate: (rng, testIdx) => [args...]`.
   Right when the answer is "whatever the canonical library does,"
   e.g. one Adam step or a tile-coding hash.

### Authoring discipline

- **The solution is also a test.** `node smoketest.js` (in
  `shared/`) compiles every `solution` string and runs it through
  the exercise's own `check` block. If your solution doesn't pass
  your own check, the exercise is broken — find out before a reader
  does. Run it whenever you add or change an exercise.
- **Always write a `signature` block.** Type hints + a one-line
  description per arg + the return shape. Column-align `:` and `—`
  for readability. The reader sees this *before* opening the sample
  disclosure — make it self-sufficient.
- **Keep the template a hollow scaffold, not a giveaway.** Show the
  return-vector setup if it's non-obvious (e.g. the `Float32Array`
  allocation) but never leave the algorithm in.
- **Help text below the mount.** Same convention as widgets: a 2-3
  sentence paragraph telling the reader why this specific exercise
  matters, beyond the prompt inside the exercise itself.
- **One canonical entrypoint name per exercise.** `backwardTanh`,
  not `backward_tanh` or `BackwardTanh`. Mirror the JS convention.
- **Don't ship an exercise to every chapter.** Same discipline as
  widgets: one per chapter where typing the algorithm yourself
  teaches something the prose alone can't.

### The editor: textarea by default, CodeMirror 6 if reachable

The harness ships a working `<textarea>` synchronously, then
asynchronously lazy-imports CodeMirror 6 from esm.sh
(`codemirror@6` + `@codemirror/lang-javascript` +
`@codemirror/theme-one-dark`). If the CDN is reachable, CM6 takes
over and the reader gets syntax highlighting + line numbers +
better keybindings. If not (offline, blocked CDN, strict CSP), the
textarea stays. Either way the exercise works — `editor.getValue()`
and `editor.setValue(...)` abstract the backend.

Authors don't configure this. It just happens.

### Pitfalls

1. **Template doesn't compile.** A starter that doesn't even parse
   gives the reader a syntax-error wall on first Run. Always make
   sure the template returns the right shape, even if filled with
   zeros — the reader sees green or "tests failed," not a parse
   error.
2. **Solution silently doesn't define the entrypoint.** If your
   solution does `const backwardTanh = …` but the harness looks
   for `function backwardTanh`, the smoketest catches it: both
   forms work (the compiler returns `typeof === "function"` for
   both). But if you typo the function name, the smoketest fails
   loudly.
3. **Tolerance too tight.** Float32 arithmetic accumulates error;
   `1e-6` is fine for fixed-input scalar checks but `1e-3` is the
   right floor for vector grad-checks against finite differences.
4. **`fixedTable` expected values computed wrong on paper.** The
   smoketest catches this only if you also wrote a `solution`. If
   you wrote a `fixedTable` case from hand calculation and skipped
   the solution, run it through the harness once before shipping.

## When to write a widget vs. when not to

**Write a widget** when:
- The math has a tunable parameter that *visibly* changes the picture.
- The convergence/divergence regime change is something the reader
  needs to *see* to internalize ("at γ near 1 this stops working").
- The geometric object has structure that survives only at certain
  parameter values (eigenvalues, contraction modulus, ε-thresholds).

**Don't write a widget** when:
- The picture is the same at every parameter value.
- The chapter prose already says clearly what's happening.
- The "thing to play with" is a discrete enum with no good
  pedagogical sweep (just write a table).

A textbook with 100 mediocre widgets is worse than one with 30 great
ones. Each widget should be load-bearing — pulling weight that the
prose alone can't.

## Adding to this guide

If you discover a new bug class or a useful pattern, add it here.
The 10-widget-correctness-audit reference under `.claude/skills/textbook-with-widgets/`
is the long-form skill version of this guide; keep them in sync.

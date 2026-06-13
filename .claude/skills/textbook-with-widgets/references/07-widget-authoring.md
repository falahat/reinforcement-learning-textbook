# Phase 4 (continued) — Widget authoring with `defineWidget`

This file is the practical guide to writing one TypeScript widget.
For the Rust path, see `references/06-widgets-architecture.md`.

## The two-line embed

In a chapter markdown file:

```markdown
<div id="ch3-myviz-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/myviz/widget.js"></script>
```

That's the entire chapter-side contribution. No inline JS, no big
HTML block, no template literals. The widget logic lives in its own
file and stays out of the prose.

**The `<div id="…">` is the mount point.** The script imports the
widget module, which calls `defineWidget`, which mounts itself
into the `<div>`.

ID convention: `ch<N>-<short-name>-widget`. Lowercase. Hyphen-
separated. Unique across the book (in case two widgets share a
page).

## Inside the widget file

A minimum-viable TS widget using the `defineWidget` scaffold:

```js
// docs/textbook/widgets/myviz/widget.js
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

defineWidget({
  hostId: "ch3-myviz-widget",
  controls: {
    gamma: { label: "γ (gamma)", min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
    n:     { label: "n", min: 1, max: 100, step: 1, default: 10 },
  },
  render: (host, { gamma, n }, slots) => {
    const data = d3.range(n + 1).map(k => ({ k, y: Math.pow(gamma, k) }));
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      marks: [
        Plot.line(data, { x: "k", y: "y", stroke: palette.primary }),
        Plot.ruleY([1e-7], { stroke: palette.danger, ...dashed }),
      ],
    }));
    slots.readout.textContent = `γ^${n} = ${Math.pow(gamma, n).toExponential(3)}`;
  },
});
```

That's ~25 lines of widget logic. The `defineWidget` scaffold
handles:
- Reading the controls config and rendering slider HTML.
- Wiring input/change events to re-call render.
- DOMContentLoaded guard.
- Looking up the host div by id.
- Providing `slots.main` (the plot mount) and `slots.readout`
  (the per-widget readout span).

You never write `document.getElementById`, never wire `addEventListener`,
never define a CONTROLS_HTML template literal. That's all in the scaffold.

## Control descriptors

A `controls` object maps names to descriptors. Each descriptor is
one form input:

```js
// Range slider (default):
gamma: { label: "γ", min: 0.5, max: 0.999, step: 0.005, default: 0.9 }

// Number input (typed value):
n: { type: "number", label: "n", min: 1, max: 1000, step: 1, default: 100 }

// Select (dropdown):
func: { type: "select", label: "f(x)", default: "square",
  options: [
    { value: "square", label: "x² (convex)" },
    { value: "log", label: "log(x) (concave)" },
  ]
}
```

Numeric controls (range, number) get `parseFloat`'d before being
passed to render. Select controls pass the string value.

## Multi-slot widgets

By default `defineWidget` creates one plot mount called `main`. For
widgets with two or more coordinated charts, pass `slots`:

```js
defineWidget({
  hostId: "ch7-lambda-return-widget",
  controls: { lambda: {/* ... */} },
  slots: ["weights", "cumulative"],
  render: (host, { lambda }, slots) => {
    slots.weights.replaceChildren(Plot.plot({/* bar chart */}));
    slots.cumulative.replaceChildren(Plot.plot({/* cum curve */}));
    slots.readout.textContent = "…";
  },
});
```

Each name becomes a `<div data-plot="<name>">` mount under the
controls bar, in the order declared.

## Step-by-step animation widgets

For widgets that walk through a derivation — matrix multiplication
cell-by-cell, value iteration sweep-by-sweep, Bellman backup
term-by-term — use `defineStepper` instead:

```js
import { defineStepper } from "../shared/stepper.js";

defineStepper({
  hostId: "ch1-matmul-widget",
  controls: {
    m: { label: "m", min: 1, max: 4, step: 1, default: 2 },
    n: { label: "n", min: 1, max: 4, step: 1, default: 3 },
  },
  // PURE function — params to frame array.
  trajectory: ({ m, n }) => {
    const frames = [];
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        frames.push({ i, j /* … */ });
    return frames;
  },
  render: (host, frame, idx, total, params, slots) => {
    slots.main.replaceChildren(/* draw `frame` */);
    slots.readout.textContent = `step ${idx + 1} / ${total}`;
  },
  playIntervalMs: 800,
});
```

Buttons added by the scaffold: ↺ reset, ◀ prev, ▶ play, next ▶.
Trajectory recomputes when any control changes; idx resets to 0.

Backward stepping is free because frames are pure functions of
params — no state mutation during animation, just an array index.

## The shared helpers

All in `docs/textbook/widgets/shared/helpers.js`:

```js
export const palette = {
  primary: "#4caf50",   // green — main curve
  secondary: "#42a5f5", // blue — secondary curve
  warning: "#ffb74d",   // orange — half-life, edge cases
  danger: "#e57373",    // red — reference rules, thresholds
  accent: "#ba68c8",    // purple — third series
  muted: "#888",        // grey — y=x reference, faint guides
};

export const dashed = { strokeDasharray: "4 2" };
export const annotation = { fontSize: 10, fill: palette.muted };

export const plotDefaults = {
  width: 640, height: 280, marginLeft: 50, marginBottom: 36,
  style: { background: "transparent", fontSize: "11px" },
};

export function readNumber(host, selector) { /* … */ }
export function readString(host, selector) { /* … */ }
export function setReadout(host, text) { /* … */ }
export function fmt(n) { /* nice number formatting */ }
```

Use these everywhere. **Do NOT** hardcode hex colors per widget;
the book looks like one book if every widget uses the same palette.

## CSS conventions

`.textbook-widget` is the outer class. The shared
`docs/textbook/widgets/shared/widgets.css` styles:
- The widget container (padding, border, background tint).
- The `.widget-controls` row (flexbox, gap, slider widths).
- The `[data-readout]` span (monospace, dimmed).
- The plot SVG text (dark-mode-friendly tick label colour).

Loaded site-wide via `additional-css = ["widgets/shared/widgets.css"]`
in book.toml. No widget needs its own `<style>` block.

## Authoring conventions

**Greek letters in labels.** Show the symbol AND the English name
in the slider label the first time it appears in the widget:

```js
gamma: { label: "γ (gamma)", min: 0.5, max: 0.999, step: 0.005, default: 0.9 }
```

This matches the chapter prose convention from
`references/03-chapter-structure.md`. A reader who's never seen γ
can still type it (and know how to refer to it in conversation).

**Numeric defaults that show interesting behaviour.** Don't default
sliders to 0 or the middle of the range — default to a value where
the widget *immediately* shows the pedagogical point. For γ^k decay,
default γ=0.9 and k=500 because that's where the math is most
striking (the curve crashes through 10^-7).

**Readouts that explain the formula.** Don't just show a number;
show the equation:

```js
slots.readout.textContent = `γ^k_max = ${gamma.toFixed(3)}^${kmax} = ${val.toExponential(3)}`;
```

The reader sees both the inputs (gamma, kmax) and the result, formatted
the same way the chapter formula does.

## What the importmap gives you

`theme/head.hbs` ships an importmap that resolves the bare module
specifiers `"d3"` and `"@observablehq/plot"` to pinned CDN URLs.
Widgets just `import * as Plot from "@observablehq/plot"` and the
browser handles it.

No bundler. No npm install (for authoring). No transpile step.
Widget files are plain `.js` with ES module syntax that runs
directly in modern browsers.

For type checking, optionally run `tsc --noEmit --checkJs` in CI
with JSDoc type annotations. Not required for runtime.

## Output of this phase

For each widget: one `.js` file under `docs/textbook/widgets/<name>/`,
plus a two-line embed in the chapter markdown. The widget renders
on first paint and re-renders on every input change.

Verify: `mdbook build`, open the chapter in browser, confirm the
widget mounts and the sliders drive a redraw.

Next: `references/08-glossary.md` for hover tooltips.

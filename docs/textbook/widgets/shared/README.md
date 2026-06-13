# Shared widget utilities

Lightweight JS/TS helpers + CSS shared across the textbook's interactive
math-viz widgets. All widgets live as
`docs/textbook/widgets/<name>/widget.js` and import from this directory
via relative paths (`../shared/widget.js`).

## What's here

- **`widget.js`** — `defineWidget(spec)`: universal scaffolding. Reads a
  control descriptor, renders the HTML, wires events, calls your
  `render` function on every input change. ~25 LOC widget logic vs the
  ~80 LOC of boilerplate it replaces.
- **`stepper.js`** — `defineStepper(spec)`: step-by-step animation
  harness. Takes a `trajectory(params) → frames[]` function + a
  per-frame `render`; provides prev/next/play/reset buttons. Use for
  matrix multiplication walkthroughs, MCTS expansion, Bellman backups,
  sum accumulation, value-iteration sweeps — anything sequential.
- **`helpers.js`** — `palette` (consistent colours), `dashed` (rule
  styling), `annotation` (Plot text-mark defaults), `plotDefaults`
  (small fonts, transparent background), plus the lower-level
  `readNumber`, `autoRender`, `setReadout`, `fmt` helpers used inside
  the scaffolds.
- **`widgets.css`** — site-wide styling for the `.textbook-widget`
  outer class + slider conventions. Loaded once via
  `additional-css` in book.toml.

## Quickest widget: `defineWidget`

```js
// docs/textbook/widgets/example/widget.js
import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

defineWidget({
  hostId: "ch3-example-widget",
  controls: {
    gamma: { label: "γ", min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
    n:     { label: "n", min: 1, max: 100, step: 1, default: 10 },
  },
  render: (host, { gamma, n }, slots) => {
    const data = d3.range(n + 1).map(k => ({ k, y: Math.pow(gamma, k) }));
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      marks: [Plot.line(data, { x: "k", y: "y", stroke: palette.primary })],
    }));
    slots.readout.textContent = `γ^${n} = ${Math.pow(gamma, n).toFixed(6)}`;
  },
});
```

Chapter markdown is just:

```markdown
<div id="ch3-example-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/example/widget.js"></script>
```

## Control descriptors

Each entry in `controls` maps to one slider / select / number input.
Range (the default) needs `{ label, min, max, step, default }`.
Select needs `{ type: "select", label, options: [...], default }`
where each option is either a string (value = label) or
`{ value, label }`. Number input: `{ type: "number", label, min,
max, step, default }`.

## Multi-slot plots

By default `defineWidget` creates one plot slot called `main`. For
widgets with two or more coordinated charts, pass `slots`:

```js
defineWidget({
  hostId: "ch7-lambda-return-widget",
  controls: { lambda: {...} },
  slots: ["weights", "cumulative"],
  render: (host, { lambda }, slots) => {
    slots.weights.replaceChildren(Plot.plot({...}));
    slots.cumulative.replaceChildren(Plot.plot({...}));
  },
});
```

Each name becomes a `<div data-plot="<name>">` in DOM order under the
controls bar.

## Step-by-step animations: `defineStepper`

For widgets that walk through a derivation, use `defineStepper`. It
adds prev / next / play / reset buttons below the plot, takes a pure
`trajectory(params) → frames[]` function plus a `render(host, frame,
idx, total, params, slots)` function.

```js
import { defineStepper } from "../shared/stepper.js";

defineStepper({
  hostId: "ch1-matmul-widget",
  controls: {
    m: { label: "m", min: 1, max: 4, step: 1, default: 2 },
    n: { label: "n", min: 1, max: 4, step: 1, default: 3 },
  },
  trajectory: ({ m, n }) => {
    const frames = [];
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        frames.push({ i, j, partial: ... });
    return frames;
  },
  render: (host, { i, j, partial }, idx, total, params, slots) => {
    slots.main.replaceChildren(/* draw frame */);
    slots.readout.textContent = `step ${idx + 1} / ${total}`;
  },
  playIntervalMs: 800,
});
```

The trajectory recomputes whenever a control input changes; index
resets to 0. Backward stepping is free because frames are pure.

## Authoring tips

- Use the **palette** constants (`palette.primary`, `.secondary`,
  `.warning`, `.danger`, `.muted`) rather than hardcoded hex.
- Reuse the **dashed** style for reference rules:
  `Plot.ruleY([target], { stroke: palette.danger, ...dashed })`.
- Spread **plotDefaults** into every `Plot.plot` config to keep
  fonts and margins consistent.
- For readouts, use `setReadout(host, text)` or
  `slots.readout.textContent = ...` — the scaffold provides a
  `<span data-readout></span>` automatically.

// Widget 1.I — Convexity / Jensen explorer (Chapter 1).
//
// Visualises Jensen's inequality: for convex f and λ ∈ [0,1],
//
//     f(λx + (1−λ)y)  ≤  λf(x) + (1−λ)f(y).
//
// The chord between (x, f(x)) and (y, f(y)) lies *above* the curve.
// For concave f the inequality flips; for affine f they coincide.
//
// User picks a preset function, drags x, y, and λ. The widget plots
// f over a sensible domain, the chord, the point on the curve, the
// point on the chord, and reports their gap. Sign of the gap diagnoses
// convex / concave / affine at the chosen pair (x, y).
//
// Mount: `<div id="ch1-jensen-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/jensen/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

// Preset functions: { f, domain (for plotting), needsPositive }.
// `needsPositive` clamps slider values to a small positive ε for log-like
// functions whose support is x > 0.
const PRESETS = {
  square:  { f: (x) => x * x,            domain: [-3, 3],   needsPositive: false, label: "x²" },
  log:     { f: (x) => Math.log(x),      domain: [0.1, 3],  needsPositive: true,  label: "log(x)" },
  neglog:  { f: (x) => -Math.log(x),     domain: [0.1, 3],  needsPositive: true,  label: "-log(x)" },
  relu:    { f: (x) => Math.max(0, x),   domain: [-3, 3],   needsPositive: false, label: "relu(x)" },
  maxhalf: { f: (x) => Math.max(x, 1 - x), domain: [-3, 3], needsPositive: false, label: "max(x, 1−x)" },
};

defineWidget({
  hostId: "ch1-jensen-widget",
  controls: {
    func: {
      type: "select", label: "f(x)",
      options: [
        { value: "square",  label: "x² (convex)" },
        { value: "log",     label: "log(x) (concave)" },
        { value: "neglog",  label: "−log(x) (convex)" },
        { value: "relu",    label: "relu(x) (convex)" },
        { value: "maxhalf", label: "max(x, 1−x) (convex)" },
      ],
      default: "square",
    },
    x:      { label: "x", min: -3, max: 3, step: 0.05, default: -1 },
    y:      { label: "y", min: -3, max: 3, step: 0.05, default: 2 },
    lambda: { label: "λ", min: 0,  max: 1, step: 0.01, default: 0.5 },
  },
  render: (host, { func, x, y, lambda }, slots) => {
    const preset = PRESETS[func] ?? PRESETS.square;
    // Clamp to the function's support if needed (log/-log require x > 0).
    const eps = 0.1;
    const xv = preset.needsPositive ? Math.max(x, eps) : x;
    const yv = preset.needsPositive ? Math.max(y, eps) : y;
    const [lo, hi] = preset.domain;

    // Curve samples — 240 points across the plotted domain.
    const curve = d3.range(241).map((i) => {
      const t = lo + (hi - lo) * (i / 240);
      return { x: t, y: preset.f(t) };
    });

    const fx = preset.f(xv);
    const fy = preset.f(yv);
    const mix = lambda * xv + (1 - lambda) * yv;          // λx + (1−λ)y
    const fOnCurve = preset.f(mix);                        // f(λx + (1−λ)y)
    const fOnChord = lambda * fx + (1 - lambda) * fy;     // λf(x) + (1−λ)f(y)
    const gap = fOnChord - fOnCurve;                       // > 0 ⇒ convex at (x,y)

    const chord = [
      { x: xv, y: fx },
      { x: yv, y: fy },
    ];

    // y-domain: pad based on the curve's range + the two endpoints.
    const ys = curve.map((d) => d.y).concat([fx, fy, fOnCurve, fOnChord]);
    const yLo = Math.min(...ys);
    const yHi = Math.max(...ys);
    const pad = 0.1 * (yHi - yLo || 1);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { domain: [lo, hi], label: "x", grid: true },
      y: { domain: [yLo - pad, yHi + pad], label: "f(x)", grid: true },
      marks: [
        // The function itself.
        Plot.line(curve, { x: "x", y: "y", stroke: palette.primary, strokeWidth: 2 }),
        // The chord from (x, f(x)) to (y, f(y)).
        Plot.line(chord, { x: "x", y: "y", stroke: palette.secondary, strokeWidth: 1.5, ...dashed }),
        // Endpoint dots.
        Plot.dot(chord, { x: "x", y: "y", fill: palette.secondary, r: 3 }),
        // Vertical guide at the convex-combination point.
        Plot.ruleX([mix], { stroke: palette.muted, ...dashed }),
        // Point on the curve: f(λx + (1−λ)y).
        Plot.dot([{ x: mix, y: fOnCurve }], { x: "x", y: "y", fill: palette.primary, r: 5, stroke: "white", strokeWidth: 1.5 }),
        // Point on the chord: λf(x) + (1−λ)f(y).
        Plot.dot([{ x: mix, y: fOnChord }], { x: "x", y: "y", fill: palette.warning, r: 5, stroke: "white", strokeWidth: 1.5 }),
        Plot.text(
          [{ x: mix, y: fOnCurve, label: "f(λx+(1−λ)y)" }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 8, dy: 4,
            fill: palette.primary, ...annotation },
        ),
        Plot.text(
          [{ x: mix, y: fOnChord, label: "λf(x)+(1−λ)f(y)" }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 8, dy: -4,
            fill: palette.warning, ...annotation },
        ),
      ],
    }));

    const sign = gap > 1e-6 ? "convex" : gap < -1e-6 ? "concave" : "affine";
    slots.readout.textContent =
      `chord − curve = ${fmt(fOnChord)} − ${fmt(fOnCurve)} = ${fmt(gap)} (${sign} at this pair)`;
  },
});

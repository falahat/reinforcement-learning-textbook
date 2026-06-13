// Widget 1.K — Linear combinations and span (Chapter 1, §1.1).
//
// Two vectors v₁ and v₂ are draggable via four sliders (their components).
// Two coefficient sliders α, β control the linear combination
// w = α v₁ + β v₂. The widget draws v₁, v₂, and w; toggling "show span"
// fills the entire span — a line (if v₁ and v₂ are parallel) or the
// whole plane (if they're independent).
//
// Pedagogical points:
//   - Linear combinations are vector addition + scaling, nothing more.
//   - The span is the SET of all linear combinations.
//   - Two vectors span a *line* iff they are linearly dependent.
//   - Otherwise they span the whole 2D plane.
//
// Mount: `<div id="ch1-linear-combo-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

defineWidget({
  hostId: "ch1-linear-combo-widget",
  controls: {
    v1x: { label: "v₁.x", min: -3, max: 3, step: 0.1, default: 1.5 },
    v1y: { label: "v₁.y", min: -3, max: 3, step: 0.1, default: 0.4 },
    v2x: { label: "v₂.x", min: -3, max: 3, step: 0.1, default: 0.3 },
    v2y: { label: "v₂.y", min: -3, max: 3, step: 0.1, default: 1.2 },
    alpha: { label: "α", min: -2, max: 2, step: 0.05, default: 1.0 },
    beta:  { label: "β", min: -2, max: 2, step: 0.05, default: 0.7 },
    showSpan: {
      type: "select", label: "show span",
      options: [{ value: "off", label: "off" }, { value: "on", label: "on" }],
      default: "on",
    },
  },
  render: (host, { v1x, v1y, v2x, v2y, alpha, beta, showSpan }, slots) => {
    const v1 = [v1x, v1y];
    const v2 = [v2x, v2y];
    const w  = [alpha * v1x + beta * v2x, alpha * v1y + beta * v2y];

    // Is v₂ a multiple of v₁? (i.e. v₁ × v₂ = 0)
    const cross = v1x * v2y - v1y * v2x;
    const parallel = Math.abs(cross) < 1e-6;

    // Span samples — a grid of (α, β) ∈ [-2, 2]² when independent, or
    // a parametric line when parallel.
    const spanPts = [];
    if (showSpan === "on") {
      if (parallel) {
        for (let t = -3; t <= 3; t += 0.05) {
          spanPts.push({ x: t * v1x, y: t * v1y });
        }
      } else {
        const N = 18;
        for (let i = -N; i <= N; i++) {
          for (let j = -N; j <= N; j++) {
            const a = (i / N) * 2;
            const b = (j / N) * 2;
            spanPts.push({ x: a * v1x + b * v2x, y: a * v1y + b * v2y, kind: "span" });
          }
        }
      }
    }

    // Parallelogram-of-decomposition: edges showing α·v₁ then β·v₂.
    const decomp = [
      { x: 0, y: 0 },
      { x: alpha * v1x, y: alpha * v1y },
      { x: w[0], y: w[1] },
      { x: beta * v2x, y: beta * v2y },
      { x: 0, y: 0 },
    ];

    const R = Math.max(3.5, Math.max(...[v1x, v1y, v2x, v2y, w[0], w[1]].map(Math.abs)) * 1.4);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 420,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [-R, R], label: "x", grid: true },
      y: { domain: [-R, R], label: "y", grid: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Span cloud or line.
        ...(showSpan === "on" && parallel
          ? [Plot.line(spanPts, { x: "x", y: "y", stroke: palette.accent, strokeOpacity: 0.5, strokeWidth: 8 })]
          : showSpan === "on"
            ? [Plot.dot(spanPts, { x: "x", y: "y", r: 1.5, fill: palette.accent, fillOpacity: 0.25 })]
            : []),
        // Decomposition parallelogram (α v₁ → w via β v₂).
        Plot.line(decomp, { x: "x", y: "y", stroke: palette.muted, strokeOpacity: 0.7, ...dashed }),
        // The two basis vectors.
        Plot.arrow([{ x1: 0, y1: 0, x2: v1x, y2: v1y }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.warning, strokeWidth: 2.5, headLength: 12,
        }),
        Plot.arrow([{ x1: 0, y1: 0, x2: v2x, y2: v2y }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.secondary, strokeWidth: 2.5, headLength: 12,
        }),
        // The combined vector w.
        Plot.arrow([{ x1: 0, y1: 0, x2: w[0], y2: w[1] }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.primary, strokeWidth: 3, headLength: 14,
        }),
        // Labels.
        Plot.text([{ x: v1x, y: v1y, label: "v₁" }], {
          x: "x", y: "y", text: "label", dx: 8, fill: palette.warning, ...annotation,
        }),
        Plot.text([{ x: v2x, y: v2y, label: "v₂" }], {
          x: "x", y: "y", text: "label", dx: 8, fill: palette.secondary, ...annotation,
        }),
        Plot.text([{ x: w[0], y: w[1], label: `w = αv₁ + βv₂` }], {
          x: "x", y: "y", text: "label", dx: 10, dy: -4, fill: palette.primary, ...annotation,
        }),
      ],
    }));

    slots.readout.innerHTML =
      `<strong>w</strong> = ${fmt(alpha)}·v₁ + ${fmt(beta)}·v₂ = (${fmt(w[0])}, ${fmt(w[1])})  ·  ` +
      `<small>v₁ × v₂ = ${fmt(cross)} — ` +
      (parallel
        ? `<span style="color:${palette.accent}">linearly dependent (span is a line)</span>`
        : `<span style="color:${palette.primary}">linearly independent (span is ℝ²)</span>`) +
      `</small>`;
  },
});

// Widget 1.A — Vector-norm visualiser (Chapter 1).
//
// Visualises the unit ball of the ℓ_p norm in 2D:
//
//     ‖x‖_p = (|x|^p + |y|^p)^(1/p) = 1.
//
// User picks a discrete p ∈ {1, 1.5, 2, 3, ∞} and slides a vector v
// around. The plot shows the unit ball (a diamond at p=1, a circle at
// p=2, a square at p=∞, with rounded-corner squares in between), an
// arrow from the origin to v, and the live readout of ‖v‖_p.
//
// Mount: `<div id="ch1-norm-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/norm/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

// Parse the dropdown's string value into a numeric p, honouring ∞.
const parseP = (s) => (s === "inf" ? Infinity : parseFloat(s));

// Pretty label for the readout: "∞" / "1" / "1.5".
const fmtP = (p) =>
  !isFinite(p) ? "∞" : Number.isInteger(p) ? String(p) : p.toFixed(1);

// ‖(x, y)‖_p, special-casing p = ∞.
const normP = (p, x, y) =>
  !isFinite(p)
    ? Math.max(Math.abs(x), Math.abs(y))
    : Math.pow(Math.pow(Math.abs(x), p) + Math.pow(Math.abs(y), p), 1 / p);

// Sample points on the unit ball {(x, y) : ‖(x, y)‖_p = 1}. For finite
// p we sweep θ ∈ [0, 2π] and compute r(θ) = (|cos θ|^p + |sin θ|^p)^(-1/p);
// for p = ∞ the ball is the axis-aligned unit square [-1, 1]².
function unitBall(p) {
  if (!isFinite(p)) {
    return [
      { x:  1, y:  1 },
      { x: -1, y:  1 },
      { x: -1, y: -1 },
      { x:  1, y: -1 },
      { x:  1, y:  1 },
    ];
  }
  const n = 720;
  return d3.range(n + 1).map((i) => {
    const theta = (i * 2 * Math.PI) / n;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const r = Math.pow(Math.pow(Math.abs(c), p) + Math.pow(Math.abs(s), p), -1 / p);
    return { x: r * c, y: r * s };
  });
}

defineWidget({
  hostId: "ch1-norm-widget",
  controls: {
    p: {
      type: "select", label: "p",
      options: [
        { value: "1",   label: "1 (taxicab / ℓ₁)" },
        { value: "1.5", label: "1.5" },
        { value: "2",   label: "2 (Euclidean / ℓ₂)" },
        { value: "3",   label: "3" },
        { value: "10",  label: "10" },
        // p = 50 makes the "ℓ_p → ℓ_∞ as p → ∞" limit visually
        // concrete: the unit ball at p = 50 is a square with
        // *very* faintly rounded corners — toggle between 50 and ∞
        // and the curves are nearly indistinguishable.
        { value: "50",  label: "50 (≈ ℓ∞)" },
        { value: "inf", label: "∞ (sup / ℓ∞)" },
      ],
      default: "2",
    },
    vx: { label: "vₓ", min: -1.5, max: 1.5, step: 0.01, default: 0.7 },
    vy: { label: "vᵧ", min: -1.5, max: 1.5, step: 0.01, default: 0.3 },
  },
  render: (host, { p, vx, vy }, slots) => {
    const pv = parseP(p);
    const ball = unitBall(pv);
    const len = normP(pv, vx, vy);
    // v-ball: { x : ‖x‖_p = ‖v‖_p }. Same shape as the unit ball, scaled
    // so the vector's tip lies exactly on it. Lets the reader SEE the
    // norm — dragging v grows/shrinks this curve.
    const vBall = ball.map((pt) => ({ x: pt.x * len, y: pt.y * len }));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 400,
      height: 400,
      marginLeft: 36,
      marginBottom: 28,
      x: { domain: [-1.7, 1.7], label: "x", grid: true },
      y: { domain: [-1.7, 1.7], label: "y", grid: true },
      aspectRatio: 1,
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        Plot.ruleY([0], { stroke: palette.muted }),
        // Unit ball — shape reference; depends only on p.
        Plot.line(ball, {
          x: "x", y: "y",
          stroke: palette.primary, strokeWidth: 2.5,
          strokeOpacity: 0.55,
        }),
        // v-ball — passes through the vector's tip; size = ‖v‖_p.
        Plot.line(vBall, {
          x: "x", y: "y",
          stroke: palette.secondary, strokeWidth: 2,
          strokeDasharray: "4 3",
        }),
        Plot.arrow(
          [{ x1: 0, y1: 0, x2: vx, y2: vy }],
          { x1: "x1", y1: "y1", x2: "x2", y2: "y2",
            stroke: palette.warning, strokeWidth: 2, headLength: 10 },
        ),
        Plot.dot([{ x: vx, y: vy }], {
          x: "x", y: "y", fill: palette.warning, r: 5,
          stroke: "white", strokeWidth: 1.5,
        }),
      ],
    }));

    slots.readout.textContent =
      `p = ${fmtP(pv)} · v = (${vx.toFixed(2)}, ${vy.toFixed(2)}) · ‖v‖_p = ${len.toFixed(3)}` +
      `  ·  solid: unit ball  ·  dashed: ‖x‖_p = ‖v‖_p`;
  },
});

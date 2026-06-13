// Widget 16.E — Pareto-front scalarization explorer (Chapter 16).
//
// §16.3 compares linear, Tchebysheff, and smoothed-Tchebysheff
// scalarizations. The chapter's table is in prose only; this
// widget puts five candidate policies on a 2-objective plane and
// shows which one each scalarization picks as the weight vector
// w = (cos θ, sin θ) rotates.
//
// Policies (fixed):
//   A: (1.0, 0.1)   — exclusive on objective 1
//   B: (0.1, 1.0)   — exclusive on objective 2
//   C: (0.7, 0.7)   — convex compromise (on convex hull)
//   D: (0.4, 0.4)   — concave compromise (inside hull; UNREACHABLE
//                     by any linear weight — the classic example)
//   E: (0.9, 0.4)   — asymmetric (convex hull)
//
// Three scoring functions:
//   linear:        Σ w_i r_i
//   Tchebysheff:   −max_i w_i · (z* − r_i)       where z* = (1, 1)
//   smoothed:      −(1/τ) · log Σ exp(τ · w_i · (z* − r_i))
//
// The reader rotates θ (the weight direction on the unit circle)
// and watches three highlighted dots — one per scheme — slide
// between policies. Key observation: the linear-optimum dot
// never lands on policy D (it's in the non-convex region);
// Tchebysheff at the right θ does.
//
// Mount: in §16.3 of Chapter 16. Maps to Exercises 2 and 3.
//
//     <div id="ch16-pareto-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/pareto_scalarization/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const POLICIES = [
  { name: "A", r1: 1.0, r2: 0.1 },
  { name: "B", r1: 0.1, r2: 1.0 },
  { name: "C", r1: 0.7, r2: 0.7 },
  { name: "D", r1: 0.4, r2: 0.4 },
  { name: "E", r1: 0.9, r2: 0.4 },
];
const Z_STAR_1 = 1.0;
const Z_STAR_2 = 1.0;

function linearScore(p, w1, w2) {
  return w1 * p.r1 + w2 * p.r2;
}
function tchebyScore(p, w1, w2) {
  // Maximise the *negation* of the worst-case weighted deviation.
  return -Math.max(w1 * (Z_STAR_1 - p.r1), w2 * (Z_STAR_2 - p.r2));
}
function smoothedScore(p, w1, w2, tau) {
  const a = tau * w1 * (Z_STAR_1 - p.r1);
  const b = tau * w2 * (Z_STAR_2 - p.r2);
  // log-sum-exp, numerically stable
  const m = Math.max(a, b);
  return -(m + Math.log(Math.exp(a - m) + Math.exp(b - m))) / tau;
}

function argmax(scores) {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[best]) best = i;
  }
  return best;
}

defineWidget({
  hostId: "ch16-pareto-widget",
  controls: {
    theta_deg: { label: "weight angle θ (°)", min: 0, max: 90, step: 1, default: 45 },
    tau:       { label: "smoothed-Tcheb τ",    min: 1, max: 50, step: 1, default: 10 },
  },
  slots: ["plane", "scores"],
  render: (host, { theta_deg, tau }, slots) => {
    const theta = (theta_deg * Math.PI) / 180;
    const w1 = Math.cos(theta);
    const w2 = Math.sin(theta);

    const linScores = POLICIES.map((p) => linearScore(p, w1, w2));
    const tchScores = POLICIES.map((p) => tchebyScore(p, w1, w2));
    const smScores = POLICIES.map((p) => smoothedScore(p, w1, w2, tau));

    const linPick = POLICIES[argmax(linScores)];
    const tchPick = POLICIES[argmax(tchScores)];
    const smPick = POLICIES[argmax(smScores)];

    // Plane plot: dots for policies, highlight rings for the three
    // winners, plus a weight-vector arrow from the origin.
    const policyRows = POLICIES.map((p) => ({ ...p }));
    const weightArrow = [
      { r1: 0, r2: 0 },
      { r1: w1 * 0.4, r2: w2 * 0.4 },
    ];

    const winners = [
      { x: linPick.r1, y: linPick.r2, kind: "linear",          color: palette.primary },
      { x: tchPick.r1, y: tchPick.r2, kind: "Tchebysheff",     color: palette.danger },
      { x: smPick.r1,  y: smPick.r2,  kind: "smoothed-Tcheb",  color: palette.warning },
    ];

    slots.plane.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      width: 360,
      x: { label: "objective 1 (r_1)", domain: [-0.05, 1.15], grid: true },
      y: { label: "objective 2 (r_2)", domain: [-0.05, 1.15], grid: true },
      marks: [
        // Iso-linear-score line passing through the linear-optimum
        // policy. Slope = -w1/w2 in (r1, r2) space.
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        // weight vector
        Plot.line(weightArrow, { x: "r1", y: "r2", stroke: palette.muted, strokeWidth: 1.5 }),
        Plot.dot(
          [{ r1: w1 * 0.4, r2: w2 * 0.4 }],
          { x: "r1", y: "r2", fill: palette.muted, r: 4 },
        ),
        // ideal point z*
        Plot.dot([{ r1: Z_STAR_1, r2: Z_STAR_2 }], {
          x: "r1", y: "r2", fill: "none", stroke: palette.muted, r: 8, symbol: "diamond",
        }),
        Plot.text([{ r1: Z_STAR_1, r2: Z_STAR_2, label: "z*" }], {
          x: "r1", y: "r2", text: "label", dx: 8, dy: -6, fill: palette.muted, ...annotation,
        }),
        // policies
        Plot.dot(policyRows, { x: "r1", y: "r2", fill: palette.secondary, r: 6 }),
        Plot.text(policyRows, {
          x: "r1", y: "r2", text: "name",
          dx: 8, dy: -10, fontSize: 11, fill: "#ddd",
        }),
        // winners — concentric rings in distinct hues
        ...winners.map((w, i) => Plot.dot([w], {
          x: "x", y: "y",
          fill: "none", stroke: w.color,
          strokeWidth: 2.5,
          r: 10 + i * 4,
        })),
      ],
    }));

    // Companion bar chart: side-by-side scores per scheme.
    const scoreRows = [];
    for (let i = 0; i < POLICIES.length; i++) {
      scoreRows.push({ policy: POLICIES[i].name, scheme: "linear",      value: linScores[i] });
      scoreRows.push({ policy: POLICIES[i].name, scheme: "Tchebysheff", value: tchScores[i] });
      scoreRows.push({ policy: POLICIES[i].name, scheme: "smoothed",    value: smScores[i] });
    }

    slots.scores.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      marginLeft: 52,
      x: { label: "policy", domain: POLICIES.map((p) => p.name) },
      y: { label: "score", grid: true },
      color: {
        legend: true,
        domain: ["linear", "Tchebysheff", "smoothed"],
        range: [palette.primary, palette.danger, palette.warning],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.barY(scoreRows, {
          x: "policy", y: "value", fill: "scheme",
          fx: "scheme",  // facet by scheme so bars don't overlap
        }),
      ],
      fx: { label: null },
    }));

    const reachableD = (tchPick.name === "D" || smPick.name === "D") ? "yes" : "no";
    slots.readout.innerHTML =
      `θ = ${theta_deg}° → w = (${fmt(w1)}, ${fmt(w2)}) &nbsp;|&nbsp; ` +
      `linear picks <strong>${linPick.name}</strong>, ` +
      `Tcheb picks <strong>${tchPick.name}</strong>, ` +
      `smooth picks <strong>${smPick.name}</strong> &nbsp;|&nbsp; ` +
      `policy D (concave) reachable? <strong>${reachableD}</strong>`;
  },
});

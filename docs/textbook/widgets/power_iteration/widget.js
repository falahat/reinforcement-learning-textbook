// Widget 1.H — Power iteration finds the dominant eigenvector (Chapter 1, §1.6).
//
// Pedagogical claim: repeatedly applying A and renormalising sends ANY
// starting vector toward A's dominant eigenvector. The convergence rate
// per step is |λ₂ / λ₁| — the closer the second eigenvalue to the first,
// the slower the alignment.
//
// The widget plots the trajectory of v_k = A v_{k-1} / ||A v_{k-1}|| as a
// polyline on the unit circle (well, the renormalised unit). The user
// steps through with prev/next or plays the animation. The dominant
// eigenvector is marked with a dashed orange line so the convergence
// destination is visible from the start.
//
// Companion widget at `ch1-grid-transform-widget` shows what A does to
// the whole grid in one step; this widget shows what A does to one
// vector in many steps.
//
// Mount: `<div id="ch1-power-iteration-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/power_iteration/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { eig2x2 } from "../shared/linalg.js";

const N_STEPS = 20;

// Eigen-decomposition sorted by descending |λ|. The shared eig2x2
// returns eigenvalues in `eigs[0].lambda`, `eigs[1].lambda` in the
// order (tr + √disc)/2, (tr − √disc)/2 — we re-sort so the dominant
// eigenvalue (the one power iteration converges to) is always first.
function eig2x2Sorted(a, b, c, d) {
  const M = [[a, b], [c, d]];
  const { real, eigs } = eig2x2(M);
  if (!real) return { real: false, lambdas: [], vecs: [] };
  const sorted = eigs.slice().sort((x, y) => Math.abs(y.lambda) - Math.abs(x.lambda));
  return {
    real: true,
    lambdas: sorted.map((e) => e.lambda),
    vecs: sorted.map((e) => e.v),
  };
}

// Signed angle of (x, y) on the unit circle, in [0, 2π).
function angle(x, y) {
  let a = Math.atan2(y, x);
  if (a < 0) a += 2 * Math.PI;
  return a;
}

defineStepper({
  hostId: "ch1-power-iteration-widget",
  controls: {
    a:    { label: "A[0][0]", min: -2, max: 3, step: 0.1, default: 1.5 },
    b:    { label: "A[0][1]", min: -2, max: 2, step: 0.1, default: 0.6 },
    c:    { label: "A[1][0]", min: -2, max: 2, step: 0.1, default: 0.3 },
    d:    { label: "A[1][1]", min: -2, max: 3, step: 0.1, default: 1.1 },
    theta0: { label: "v₀ angle (deg)", min: 0, max: 359, step: 1, default: 70 },
  },
  slots: ["plot"],
  trajectory: ({ a, b, c, d, theta0 }) => {
    const rad = (theta0 * Math.PI) / 180;
    let v = [Math.cos(rad), Math.sin(rad)];
    const frames = [{ k: 0, v: v.slice(), all: [v.slice()] }];
    for (let k = 1; k <= N_STEPS; k++) {
      const next = [a * v[0] + b * v[1], c * v[0] + d * v[1]];
      const n = Math.hypot(next[0], next[1]);
      if (n < 1e-12) break;
      const nv = [next[0] / n, next[1] / n];
      frames.push({ k, v: nv.slice(), all: frames[frames.length - 1].all.concat([nv.slice()]) });
      v = nv;
    }
    return frames;
  },
  playIntervalMs: 350,
  render: (host, frame, idx, total, params, slots) => {
    const { a, b, c, d } = params;
    const eig = eig2x2Sorted(a, b, c, d);

    // Trajectory points so far.
    const traj = frame.all.map((p, k) => ({ k, x: p[0], y: p[1] }));
    const curV = frame.v;

    // Dominant eigenvector arrow (both directions — power iteration
    // can converge to ±v₁ depending on sign of λ₁).
    const eigMarks = eig.real
      ? [
          { x1: 0, y1: 0, x2: eig.vecs[0][0] * 1.15, y2: eig.vecs[0][1] * 1.15 },
          { x1: 0, y1: 0, x2: -eig.vecs[0][0] * 1.15, y2: -eig.vecs[0][1] * 1.15 },
        ]
      : [];

    slots.plot.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380,
      width: 420,
      marginLeft: 40,
      marginBottom: 32,
      x: { domain: [-1.4, 1.4], label: "x", grid: true },
      y: { domain: [-1.4, 1.4], label: "y", grid: true },
      aspectRatio: 1,
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Unit circle reference.
        Plot.line(d3.range(81).map((i) => {
          const t = (i * 2 * Math.PI) / 80;
          return { x: Math.cos(t), y: Math.sin(t) };
        }), { x: "x", y: "y", stroke: palette.muted, strokeOpacity: 0.35, ...dashed }),
        // Dominant eigenvector (when real).
        ...(eig.real ? [
          Plot.line(eigMarks.flatMap((e) => [
            { x: e.x1, y: e.y1 }, { x: e.x2, y: e.y2 }, { x: NaN, y: NaN },
          ]).filter((p) => !Number.isNaN(p.x) || p === undefined), {
            x: "x", y: "y", stroke: palette.warning,
            strokeOpacity: 0.85, strokeWidth: 2, ...dashed,
          }),
          Plot.text([{ x: eig.vecs[0][0] * 1.2, y: eig.vecs[0][1] * 1.2, label: `dominant v₁ (λ₁=${fmt(eig.lambdas[0])})` }], {
            x: "x", y: "y", text: "label",
            fill: palette.warning, ...annotation, textAnchor: "start", dx: 4,
          }),
        ] : []),
        // Trajectory polyline so far.
        Plot.line(traj, {
          x: "x", y: "y",
          stroke: palette.secondary, strokeWidth: 1.5, strokeOpacity: 0.65,
        }),
        // Past iterates as dots, faded by recency.
        Plot.dot(traj.slice(0, -1), {
          x: "x", y: "y", r: 2.5, fill: palette.secondary, fillOpacity: 0.45,
        }),
        // Current iterate (highlighted).
        Plot.dot([{ x: curV[0], y: curV[1] }], {
          x: "x", y: "y", r: 5, fill: palette.primary, stroke: "white", strokeWidth: 1.5,
        }),
        Plot.text([{ x: curV[0], y: curV[1], label: `v_${frame.k}` }], {
          x: "x", y: "y", text: "label",
          fill: palette.primary, ...annotation, textAnchor: "start", dx: 8, dy: -4,
        }),
      ],
    }));

    // Readout: angle alignment with dominant eigenvector.
    let angleToEig = "—";
    if (eig.real) {
      const dot = Math.abs(curV[0] * eig.vecs[0][0] + curV[1] * eig.vecs[0][1]);
      angleToEig = `${fmt(Math.acos(Math.min(1, dot)) * 180 / Math.PI)}°`;
    }
    const ratio = eig.real && Math.abs(eig.lambdas[0]) > 1e-9
      ? Math.abs(eig.lambdas[1] / eig.lambdas[0])
      : NaN;
    slots.readout.textContent =
      `step k = ${frame.k} / ${N_STEPS}  ·  ` +
      (eig.real
        ? `λ₁ = ${fmt(eig.lambdas[0])}, λ₂ = ${fmt(eig.lambdas[1])}  ·  ` +
          `|λ₂/λ₁| = ${fmt(ratio)} (convergence rate per step)  ·  ` +
          `angle(v_k, v₁) = ${angleToEig}`
        : `complex eigenvalues — no real fixed direction; v_k rotates`);
  },
});

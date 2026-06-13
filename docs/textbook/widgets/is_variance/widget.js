// Widget 5.C — Importance-sampling variance blow-up (Chapter 5).
//
// Off-policy MC reweights returns by ρ = Π_t π(a_t|s_t)/μ(a_t|s_t).
// Each factor is a ratio of probabilities; multiply L of them and
// you get a heavy-tailed mess. As the target policy π gets sharper
// (low softmax temperature τ) the maximum ratio per step climbs,
// and the histogram of trajectory-level ρ stretches into a tail
// that dwarfs the bulk of the distribution.
//
// Set-up: a tiny chain MDP, 3 actions per state. Behaviour μ is
// uniform over actions. Target π = softmax_τ over a fixed
// preference vector (here [2, 0, −1] — left action favoured). 200
// trajectories of length L are sampled under μ; we plot the
// histogram of log10(ρ) (log scale, otherwise the tail flattens
// the bulk into one column).
//
// Two readouts: max(ρ) and ESS = (Σρ)² / Σρ². ESS collapses fast
// — the prose claim "200 samples but only ESS effective ones" is
// pointable on the screen.
//
// Mount: `<div id="ch5-is-variance-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/is_variance/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, annotation } from "../shared/helpers.js";

const N_ACTIONS = 3;
const PREFS = [2.0, 0.0, -1.0]; // target-policy preferences over actions
const N_TRAJ = 200;

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Softmax of PREFS at temperature τ. τ → 0 gives a one-hot on argmax;
// τ → ∞ gives uniform.
function targetPolicy(tau) {
  const safeTau = Math.max(tau, 1e-3);
  const logits = PREFS.map((p) => p / safeTau);
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

// One trajectory of length L under uniform μ. Returns the product ρ.
function sampleRho(pi, L, rand) {
  let logRho = 0;
  const muProb = 1 / N_ACTIONS;
  for (let t = 0; t < L; t++) {
    const a = Math.floor(rand() * N_ACTIONS);
    logRho += Math.log(pi[a] / muProb);
  }
  return Math.exp(logRho);
}

defineWidget({
  hostId: "ch5-is-variance-widget",
  controls: {
    tau:    { label: "τ (target softmax temp)", min: 0.1, max: 3, step: 0.05, default: 1.0 },
    length: { label: "trajectory length L", min: 1, max: 30, step: 1, default: 5 },
  },
  render: (host, { tau, length }, slots) => {
    const pi = targetPolicy(tau);
    const rand = rng(42);
    const rhos = new Array(N_TRAJ);
    for (let i = 0; i < N_TRAJ; i++) rhos[i] = sampleRho(pi, length, rand);

    // log10(ρ) — clip ρ ≤ 0 (shouldn't happen since pi > 0 but be safe).
    // Some rhos may be 0 (impossible under uniform μ + positive π, but
    // floating-point underflow on long trajectories can drop tail mass
    // to literal zero). Pin those to a finite floor for display only.
    const logFloor = -10;
    const logRhos = rhos.map((r) => {
      if (!(r > 0)) return logFloor;
      const v = Math.log10(r);
      return Math.max(v, logFloor);
    });

    const maxRho = Math.max(...rhos);
    const sumRho = rhos.reduce((a, b) => a + b, 0);
    const sumRho2 = rhos.reduce((a, b) => a + b * b, 0);
    // Effective sample size — exactly N if all ρ are equal, collapses
    // toward 1 as the distribution concentrates on one massive ρ.
    const ess = sumRho2 > 0 ? (sumRho * sumRho) / sumRho2 : 0;

    // Histogram bins for log10(ρ). Compute domain from the data so the
    // tail stays on screen; clamp at logFloor on the left.
    const lo = Math.min(logFloor, Math.floor(d3.min(logRhos)));
    const hi = Math.max(0.5, Math.ceil(d3.max(logRhos)));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "log₁₀(ρ) — trajectory IS ratio", grid: true, domain: [lo, hi] },
      y: { label: "count (200 trajectories under μ)", grid: true },
      marks: [
        Plot.rectY(
          logRhos.map((v) => ({ v })),
          Plot.binX({ y: "count" }, { x: "v", fill: palette.warning, fillOpacity: 0.8, thresholds: 40 }),
        ),
        Plot.ruleX([0], { stroke: palette.muted, strokeDasharray: "4 2" }),
        Plot.text(
          [{ x: 0, y: 0, label: "ρ = 1" }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4, dy: -4,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    // Bar chart of the current π — small panel above the readout shows
    // how the temperature has bent the target distribution.
    const piStr = pi.map((p, i) => `π(a${i + 1})=${p.toFixed(2)}`).join(" · ");
    slots.readout.textContent =
      `${piStr}  ·  max ρ = ${maxRho.toExponential(2)}  ·  ` +
      `ESS = ${ess.toFixed(1)} / 200 (${(100 * ess / N_TRAJ).toFixed(1)}%)`;
  },
});

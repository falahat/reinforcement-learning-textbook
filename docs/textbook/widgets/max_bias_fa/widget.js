// Widget 15.E — Maximization bias under linear function approximation
// (Chapter 15).
//
// [van Hasselt 2010]'s classical maximization-bias bandit (Chapter 6)
// is sharpened here by sharing features across arms. With pure
// tabular estimation, max bias is E[max_a Q̂(a)] − max_a E[Q̂(a)] >
// 0 because noise in each arm's estimator is independent. Under
// linear FA with overlapping features, every arm's update perturbs
// the *shared* weights — so the noise becomes correlated, and the
// bias shape changes: high overlap can amplify (every step makes
// every arm look better) or damp (regression-to-mean across arms),
// depending on whether the true Q values are well-separated or close.
//
// Three estimators on the same noisy data:
//   - vanilla Q-learning: max is biased upward.
//   - Double Q-learning: two estimators, decouple argmax from value.
//   - Optimistic init: Q starts high; max never overshoots the prior.
//
// The reader slides feature_overlap ∈ [0, 1] (0 = one-hot, no shared
// features; 1 = a single shared bias feature). The widget plots
// E[max_a Q̂(a)] − Q*(best) after `steps` pulls, averaged over
// `repeats` seeds, for all three estimators.
//
//     <div id="ch15-max-bias-fa-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/max_bias_fa/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";

const N_ARMS = 4;
// Tight cluster of true means → favourable territory for max-bias.
const TRUE_MEANS = [0.50, 0.52, 0.55, 0.60]; // arm 3 is best
const REWARD_STD = 1.0;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 0x1_0000_0000;
  };
}

function gaussian(rand) {
  // Box-Muller.
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Features: each arm has a private one-hot feature + a shared bias
// feature, weighted by `overlap`. φ_a = [(1-overlap)·e_a, overlap].
// Total features = N_ARMS + 1 (private + bias).
function makePhi(overlap) {
  const F = N_ARMS + 1;
  const M = [];
  for (let a = 0; a < N_ARMS; a++) {
    const row = new Float64Array(F);
    row[a] = 1.0 - overlap;
    row[F - 1] = overlap;
    M.push(row);
  }
  return M;
}

// Run one trajectory of the chosen estimator and return final
// argmax-Q estimate and the true mean of the selected arm.
function runOne(estimator, { steps, alpha, overlap, optInit, seed }) {
  const rand = rng(seed);
  const phi = makePhi(overlap);
  const F = N_ARMS + 1;
  const initVal = estimator === "optimistic" ? optInit : 0;
  // For dueling/Double we keep two weight vectors; otherwise one.
  const w  = new Float64Array(F);
  const w2 = new Float64Array(F);
  // Push the optimistic prior onto the bias feature so Q starts at ~initVal
  // for every arm.
  if (estimator === "optimistic" && overlap > 0) {
    w[F - 1] = initVal / overlap;
  } else if (estimator === "optimistic") {
    for (let a = 0; a < N_ARMS; a++) w[a] = initVal;
  }

  function qOf(weights, a) {
    let s = 0;
    for (let k = 0; k < F; k++) s += weights[k] * phi[a][k];
    return s;
  }

  for (let t = 0; t < steps; t++) {
    // Uniform exploration so the bias story isn't confounded by
    // policy improvement. (The classical max-bias setup is the same.)
    const a = Math.floor(rand() * N_ARMS);
    const r = TRUE_MEANS[a] + REWARD_STD * gaussian(rand);

    if (estimator === "double") {
      // Coin flip which table updates; max comes from the *other*.
      const useA = rand() < 0.5;
      const updW   = useA ? w  : w2;
      const otherW = useA ? w2 : w;
      // Double-Q bandit version: target = r (no bootstrap); the
      // standard double-Q decoupling is moot for a bandit, but to
      // demonstrate the FA story we still average the two tables at
      // argmax time. The contrast remains: optimistic Q-learning is
      // tighter than vanilla in pure-tabular, and with shared
      // features the picture is more complex.
      const q = qOf(updW, a);
      const tdError = r - q;
      for (let k = 0; k < F; k++) updW[k] += alpha * tdError * phi[a][k];
      // Track 'other' too with a soft EMA so both estimates exist.
      const q2 = qOf(otherW, a);
      for (let k = 0; k < F; k++) otherW[k] += alpha * 0.1 * (r - q2) * phi[a][k];
    } else {
      const q = qOf(w, a);
      const tdError = r - q;
      for (let k = 0; k < F; k++) w[k] += alpha * tdError * phi[a][k];
    }
  }

  // Compute final estimates per arm.
  const finalQ = [];
  for (let a = 0; a < N_ARMS; a++) {
    if (estimator === "double") {
      finalQ.push(0.5 * (qOf(w, a) + qOf(w2, a)));
    } else {
      finalQ.push(qOf(w, a));
    }
  }
  let argmax = 0;
  for (let a = 1; a < N_ARMS; a++) if (finalQ[a] > finalQ[argmax]) argmax = a;
  return {
    maxEst:  Math.max(...finalQ),
    argmax,
    finalQ,
  };
}

defineWidget({
  hostId: "ch15-max-bias-fa-widget",
  controls: {
    overlap: { label: "feature overlap", min: 0, max: 1, step: 0.05, default: 0.5 },
    steps:   { label: "pulls",           min: 100, max: 5000, step: 100, default: 1000 },
    alpha:   { label: "α",               min: 0.005, max: 0.1, step: 0.005, default: 0.02 },
    optInit: { label: "Q_optimistic",    min: 0.5, max: 3.0, step: 0.1, default: 1.5 },
    repeats: { label: "seeds (avg)",     min: 10, max: 200, step: 10, default: 50 },
  },
  slots: ["main"],
  render: (host, p, slots) => {
    const bestMu = Math.max(...TRUE_MEANS);
    const cfg = {
      steps: p.steps | 0, alpha: p.alpha, overlap: p.overlap, optInit: p.optInit,
    };
    const reps = p.repeats | 0;
    // Sweep overlap across a few points so we have a curve, but
    // highlight the user-selected slider value. The slider just sets a
    // ruled vertical line.
    const overlaps = [];
    for (let v = 0.0; v <= 1.0001; v += 0.1) overlaps.push(v);
    const rows = [];
    const estimators = ["q-learning", "double", "optimistic"];
    for (const o of overlaps) {
      const aggr = { "q-learning": 0, "double": 0, "optimistic": 0 };
      for (let r = 0; r < reps; r++) {
        for (const e of estimators) {
          const result = runOne(e, { ...cfg, overlap: o, seed: 17 + r });
          aggr[e] += result.maxEst - bestMu;
        }
      }
      for (const e of estimators) {
        rows.push({ overlap: o, estimator: e, bias: aggr[e] / reps });
      }
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "feature overlap (0 = one-hot, 1 = shared bias)", grid: true, domain: [0, 1] },
      y: { label: "E[max_a Q̂] − μ* (bias)", grid: true, zero: true },
      color: {
        domain: estimators,
        range: [palette.danger, palette.secondary, palette.primary],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.ruleX([p.overlap], { stroke: palette.warning, ...dashed }),
        Plot.line(rows, { x: "overlap", y: "bias", stroke: "estimator", strokeWidth: 2 }),
        Plot.dot(rows, { x: "overlap", y: "bias", fill: "estimator", r: 3 }),
      ],
    }));

    // Single-seed final-Q snapshot at the slider's overlap for the readout.
    const snap = runOne("q-learning", { ...cfg, seed: 7 });
    const argmaxArm = snap.argmax;
    const correct = argmaxArm === TRUE_MEANS.indexOf(bestMu);
    slots.readout.textContent =
      `at overlap=${fmt(p.overlap)}: vanilla Q final max est=${fmt(snap.maxEst)}, ` +
      `argmax arm=${argmaxArm} (best=${TRUE_MEANS.indexOf(bestMu)}, ${correct ? "correct" : "wrong"}); ` +
      `μ* = ${fmt(bestMu)}`;
  },
});

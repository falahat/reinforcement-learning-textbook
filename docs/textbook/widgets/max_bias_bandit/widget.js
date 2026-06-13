// Widget 9.C — Maximization-bias drift on a K-arm bandit (Chapter 9).
//
// Section §9.3 motivates Double DQN with the claim "the max of noisy
// variables is bigger than the max of the true values." For a K-arm
// bandit with true Q* = 0 and per-arm estimates  Q_hat_k ~ N(0, σ²/n_k),
// the expected single-network max is approximately
//
//     E[max_k Q_hat_k] ≈ σ · sqrt((2 log K) / n)   (extreme-value asym.)
//
// while the Double-DQN estimator — pick a* with one independent sample,
// evaluate Q_hat at a* with another — is unbiased: E ≈ 0.
//
// The widget runs a Monte-Carlo experiment over user-controlled n (per-
// arm sample count) and K (arm count), drawing M = 4000 trials. It plots
//   (a) bar chart of single-network vs double-network max estimate
//       at the current (K, n);
//   (b) growth curve E[max] vs K with the analytic sqrt(2 log K) overlay
//       so the reader sees the asymptotic bias.
//
// Pattern: chapter markdown contains
//
//     <div id="ch9-max-bias-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/max_bias_bandit/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const TRIALS = 4000;
const SIGMA = 1.0;

// Standard-normal sample via Box-Muller. Deterministic seed-less since
// each render runs MC fresh — variance over 4000 trials is fine.
function randn() {
  const u = Math.max(Math.random(), 1e-12);
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Per-arm estimate variance: σ² / n. So each Q_hat ~ N(0, σ²/n).
function sampleEstimate(n) {
  return (SIGMA / Math.sqrt(n)) * randn();
}

// One trial: draw K estimates from "network A", K from "network B".
// Returns { singleMax, doubleEst }.
function runTrial(K, n) {
  let singleMax = -Infinity;
  let argmaxA = 0;
  let bestA = -Infinity;
  const B = new Array(K);
  for (let k = 0; k < K; k++) {
    const a = sampleEstimate(n);
    const b = sampleEstimate(n);
    if (a > singleMax) singleMax = a;
    if (a > bestA) {
      bestA = a;
      argmaxA = k;
    }
    B[k] = b;
  }
  // Double DQN: select with A, evaluate with B at the same arm.
  return { singleMax, doubleEst: B[argmaxA] };
}

function monteCarlo(K, n, trials) {
  let sumSingle = 0;
  let sumDouble = 0;
  let sumSingleSq = 0;
  let sumDoubleSq = 0;
  for (let t = 0; t < trials; t++) {
    const { singleMax, doubleEst } = runTrial(K, n);
    sumSingle += singleMax;
    sumDouble += doubleEst;
    sumSingleSq += singleMax * singleMax;
    sumDoubleSq += doubleEst * doubleEst;
  }
  const meanSingle = sumSingle / trials;
  const meanDouble = sumDouble / trials;
  const varSingle = sumSingleSq / trials - meanSingle * meanSingle;
  const varDouble = sumDoubleSq / trials - meanDouble * meanDouble;
  return { meanSingle, meanDouble, sdSingle: Math.sqrt(Math.max(varSingle, 0)), sdDouble: Math.sqrt(Math.max(varDouble, 0)) };
}

defineWidget({
  hostId: "ch9-max-bias-widget",
  controls: {
    K: { label: "K (arms)", min: 2, max: 50, step: 1, default: 10 },
    n: { label: "n (samples / arm)", min: 1, max: 200, step: 1, default: 10 },
  },
  slots: ["bars", "growth"],
  render: (host, { K, n }, slots) => {
    // (a) Bar chart at the chosen (K, n).
    const { meanSingle, meanDouble, sdSingle, sdDouble } = monteCarlo(K, n, TRIALS);
    const bars = [
      { kind: "Single (DQN)", estimate: meanSingle, sd: sdSingle, color: palette.danger },
      { kind: "Double (DDQN)", estimate: meanDouble, sd: sdDouble, color: palette.primary },
      { kind: "Truth (Q*)", estimate: 0, sd: 0, color: palette.muted },
    ];

    const yAbs = Math.max(0.6, Math.abs(meanSingle) * 1.4);
    slots.bars.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      marginLeft: 110,
      x: { label: "E[ max-estimator ]", domain: [-yAbs, yAbs], grid: true },
      y: { label: null, domain: bars.map((b) => b.kind) },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        Plot.barX(bars, { x: "estimate", y: "kind", fill: (d) => d.color, fillOpacity: 0.85 }),
        // ±1 SD whiskers (Monte-Carlo std of the estimator value, not
        // of the mean — gives the reader a sense of trial-by-trial
        // variability, not just the bias).
        Plot.ruleY(bars, {
          y: "kind",
          x1: (d) => d.estimate - d.sd,
          x2: (d) => d.estimate + d.sd,
          stroke: "#222",
          strokeOpacity: 0.6,
        }),
        Plot.text(bars, {
          x: "estimate",
          y: "kind",
          text: (d) => fmt(d.estimate),
          dx: (d) => (d.estimate >= 0 ? 6 : -6),
          textAnchor: (d) => (d.estimate >= 0 ? "start" : "end"),
          fill: "#ddd",
          fontSize: 11,
        }),
      ],
    }));

    // (b) Growth curve: E[max] vs K at the current n. Compare MC to
    // the extreme-value asymptote σ·sqrt(2 log K / n).
    const Kgrid = [2, 3, 5, 8, 12, 20, 30, 50, 80];
    const mcRows = [];
    const analyticRows = [];
    // Cheaper trial budget for the curve.
    const SWEEP_TRIALS = 1500;
    for (const k of Kgrid) {
      const { meanSingle, meanDouble } = monteCarlo(k, n, SWEEP_TRIALS);
      mcRows.push({ K: k, series: "Single (DQN)", y: meanSingle });
      mcRows.push({ K: k, series: "Double (DDQN)", y: meanDouble });
      analyticRows.push({ K: k, y: (SIGMA / Math.sqrt(n)) * Math.sqrt(2 * Math.log(k)) });
    }
    const colourFor = (s) => (s === "Single (DQN)" ? palette.danger : palette.primary);

    slots.growth.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "K (arms)", type: "log", domain: [2, 80], grid: true },
      y: { label: "E[ max-estimator ]", grid: true },
      color: { legend: true, domain: ["Single (DQN)", "Double (DDQN)"], range: [palette.danger, palette.primary] },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(analyticRows, { x: "K", y: "y", stroke: palette.warning, strokeWidth: 1.5, ...dashed }),
        Plot.text(
          [{ K: Kgrid[Kgrid.length - 1], y: analyticRows[analyticRows.length - 1].y, label: "σ·√(2 ln K / n)" }],
          { x: "K", y: "y", text: "label", textAnchor: "end", dy: -6, fill: palette.warning, ...annotation },
        ),
        Plot.line(mcRows, { x: "K", y: "y", stroke: (d) => colourFor(d.series), strokeWidth: 2, z: "series" }),
        Plot.dot(mcRows, { x: "K", y: "y", fill: (d) => colourFor(d.series), r: 3, z: "series" }),
      ],
    }));

    slots.readout.textContent =
      `K = ${K}, n = ${n} · single-net bias ≈ ${fmt(meanSingle)} · ` +
      `double-net bias ≈ ${fmt(meanDouble)} · ` +
      `analytic σ·√(2 ln K / n) = ${fmt((SIGMA / Math.sqrt(n)) * Math.sqrt(2 * Math.log(K)))}`;
  },
});

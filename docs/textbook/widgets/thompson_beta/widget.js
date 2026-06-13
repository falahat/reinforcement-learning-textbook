// Widget 12.D — Thompson sampling Beta posterior animator (Chapter 12).
//
// 5-arm Bernoulli bandit with Beta(alpha_k, beta_k) posteriors. Each
// step:
//   1. Sample mu_k ~ Beta(alpha_k, beta_k) for every arm.
//   2. Pull arm argmax_k mu_k. Mark the sampled value as a vertical
//      tick on each posterior.
//   3. Observe Bernoulli reward; update the chosen arm's posterior
//      (success: alpha++; failure: beta++).
//
// Pedagogy: posteriors start broad (uniform = Beta(1,1)); as rounds
// accumulate, each arm's density sharpens around its true success
// probability and the bad arms' posteriors stop overlapping the best
// arm's. Watching the densities sharpen is the entire Thompson story.
//
// Mount:
//     <div id="ch12-thompson-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/thompson_beta/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";

const HOST_ID = "ch12-thompson-widget";
const N_ARMS = 5;
const TRUE_MEANS = [0.10, 0.30, 0.50, 0.70, 0.90];
// One colour per arm. Drawn from the shared palette plus a stable
// extra hue so the five posteriors are individually distinguishable.
const ARM_COLOURS = [
  palette.primary,
  palette.secondary,
  palette.warning,
  palette.accent,
  palette.danger,
];

// Beta PDF at x in (0,1). Uses log-gamma to stay numerically stable
// for large alpha+beta (the posterior gets very peaky after ~500 rounds).
function logGamma(z) {
  // Lanczos approximation (g=7, n=9). Good to ~1e-14 for z > 0.
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const tHalf = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(tHalf) - tHalf + Math.log(x);
}
function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  const logB = logGamma(a) + logGamma(b) - logGamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB);
}

// Sample from Beta(a, b) via two independent Gamma samples.
// Uses Marsaglia & Tsang for shape >= 1, a small fall-back for shape < 1.
function gammaSample(shape) {
  if (shape < 1) {
    // Johnk's algorithm boost: Gamma(s) = Gamma(s+1) * U^(1/s).
    return gammaSample(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const cc = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      // Standard normal via Box-Muller.
      const u1 = Math.random(), u2 = Math.random();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + cc * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
function betaSample(a, b) {
  const x = gammaSample(a);
  const y = gammaSample(b);
  return x / (x + y);
}

const CONTROLS_HTML = `
  <div class="widget-controls">
    <button data-button="step">step</button>
    <button data-button="step10">+10 steps</button>
    <button data-button="step100">+100 steps</button>
    <button data-button="reset">reset</button>
    <span data-readout></span>
  </div>
  <div data-plot="posteriors"></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  // Uniform Beta(1,1) priors for each arm.
  const alpha = new Array(N_ARMS).fill(1);
  const beta = new Array(N_ARMS).fill(1);
  let t = 0;
  let lastPicked = -1;
  let lastSamples = null;

  function pullOne() {
    const samples = new Array(N_ARMS);
    let best = 0;
    let bestVal = -Infinity;
    for (let k = 0; k < N_ARMS; k++) {
      samples[k] = betaSample(alpha[k], beta[k]);
      if (samples[k] > bestVal) {
        bestVal = samples[k];
        best = k;
      }
    }
    const reward = Math.random() < TRUE_MEANS[best] ? 1 : 0;
    if (reward) alpha[best] += 1;
    else beta[best] += 1;
    t += 1;
    lastPicked = best;
    lastSamples = samples;
  }

  function reset() {
    alpha.fill(1);
    beta.fill(1);
    t = 0;
    lastPicked = -1;
    lastSamples = null;
    render();
  }

  function render() {
    // PDF samples for each arm across [0, 1].
    const xs = d3.range(0, 1.0001, 0.005);
    const lines = [];
    for (let k = 0; k < N_ARMS; k++) {
      for (const x of xs) {
        lines.push({ arm: k, x, y: betaPdf(x, alpha[k], beta[k]) });
      }
    }

    // True-mean dashed ticks per arm.
    const trueTicks = TRUE_MEANS.map((m, k) => ({ arm: k, x: m }));

    // Vertical sample lines from the most recent step.
    const sampleMarks = lastSamples
      ? lastSamples.map((s, k) => ({ arm: k, x: s, picked: k === lastPicked }))
      : [];

    host.querySelector('[data-plot="posteriors"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "mu (success probability)", domain: [0, 1], grid: true },
      y: { label: "Beta posterior density", grid: true, zero: true },
      marks: [
        Plot.line(lines, {
          x: "x", y: "y",
          stroke: (d) => ARM_COLOURS[d.arm],
          z: "arm",
          strokeWidth: 2,
        }),
        // True mean tick at the bottom for each arm.
        Plot.ruleX(trueTicks, {
          x: "x",
          stroke: (d) => ARM_COLOURS[d.arm],
          strokeOpacity: 0.6,
          ...dashed,
        }),
        // Most-recent sampled value per arm.
        ...(sampleMarks.length > 0 ? [
          Plot.tickX(sampleMarks, {
            x: "x",
            stroke: (d) => d.picked ? palette.muted : ARM_COLOURS[d.arm],
            strokeWidth: (d) => d.picked ? 3 : 1.5,
            strokeOpacity: (d) => d.picked ? 1.0 : 0.7,
          }),
        ] : []),
        Plot.text(
          TRUE_MEANS.map((m, k) => ({ x: m, y: 0, label: `mu*=${m.toFixed(2)}` })),
          { x: "x", y: "y", text: "label", textAnchor: "middle", dy: 14,
            fill: (d, i) => ARM_COLOURS[i], fontSize: 9 },
        ),
      ],
    }));

    const summary = alpha.map((a, k) => `arm${k}: B(${a},${beta[k]})`).join("  ");
    const picked = lastPicked >= 0 ? `picked arm ${lastPicked}` : "(none yet)";
    host.querySelector("[data-readout]").innerHTML =
      `t=${t} &nbsp; last ${picked} &nbsp; <small>${summary}</small>`;
  }

  host.querySelector('[data-button="step"]').addEventListener("click", () => {
    pullOne();
    render();
  });
  host.querySelector('[data-button="step10"]').addEventListener("click", () => {
    for (let i = 0; i < 10; i++) pullOne();
    render();
  });
  host.querySelector('[data-button="step100"]').addEventListener("click", () => {
    for (let i = 0; i < 100; i++) pullOne();
    render();
  });
  host.querySelector('[data-button="reset"]').addEventListener("click", reset);

  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

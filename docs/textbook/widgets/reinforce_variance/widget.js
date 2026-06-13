// Widget 10.B — REINFORCE gradient-variance histogram (Chapter 10).
//
// Pulls the variance claim out of prose into a measurable picture. On a
// tiny 2-arm bandit (rewards drawn from N(μ_a, σ_a²) per arm), we run
// M = 500 single-episode REINFORCE "gradient estimates" of the same
// underlying ∇J. Each run produces a scalar gradient w.r.t. the action
// logit; we histogram |g| across the M runs.
//
// Three baseline modes — toggle to see the histogram tighten:
//   • none       : ĝ = (1 − π(a))·G              (pure REINFORCE)
//   • mean       : ĝ = (1 − π(a))·(G − G̅)        (running-mean baseline)
//   • critic V   : ĝ = (1 − π(a))·(G − V*)        (oracle state-value)
//
// All three are unbiased estimators of the same gradient; only the
// *variance* differs. The empirical mean (vertical rule) coincides
// across modes — this is the §10.4 baseline-lemma's pictorial claim.
//
// Pattern:
//
//     <div id="ch10-reinforce-variance-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/reinforce_variance/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N_RUNS = 500;
const ARM_MEANS = [0.6, 0.3];    // arm 0 is better in expectation
const ARM_STDS  = [1.0, 1.0];

// Deterministic LCG so the histogram is reproducible across renders.
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Box-Muller normal sample from two uniforms in (0, 1].
function gauss(rand) {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

defineWidget({
  hostId: "ch10-reinforce-variance-widget",
  controls: {
    theta:    { label: "θ (logit gap, arm 0 vs 1)", min: -2, max: 2, step: 0.05, default: 0.5 },
    baseline: {
      type: "select",
      label: "baseline",
      options: [
        { value: "none",   label: "none (raw REINFORCE)" },
        { value: "mean",   label: "running mean of G" },
        { value: "critic", label: "oracle V (= E[G])" },
      ],
      default: "none",
    },
  },
  render: (host, { theta, baseline }, slots) => {
    // π(a=0) = σ(θ); π(a=1) = 1 − π(0). Score grad w.r.t. θ:
    //   ∇_θ log π(a=0|·) = 1 − π(0)
    //   ∇_θ log π(a=1|·) = −π(0)
    // — symmetric, so |ĝ| is comparable across arm choices.
    const pi0 = 1 / (1 + Math.exp(-theta));
    const pi1 = 1 - pi0;
    const Vstar = pi0 * ARM_MEANS[0] + pi1 * ARM_MEANS[1]; // closed-form V^π

    const rand = rng(42);
    const Gs = new Array(N_RUNS);
    const arms = new Array(N_RUNS);
    for (let i = 0; i < N_RUNS; i++) {
      const a = rand() < pi0 ? 0 : 1;
      arms[i] = a;
      Gs[i] = ARM_MEANS[a] + ARM_STDS[a] * gauss(rand);
    }
    const meanG = Gs.reduce((a, b) => a + b, 0) / N_RUNS;

    const grads = new Array(N_RUNS);
    for (let i = 0; i < N_RUNS; i++) {
      // Score function w.r.t. θ for the action that was taken.
      const score = arms[i] === 0 ? (1 - pi0) : -pi0;
      let signal;
      if (baseline === "none") signal = Gs[i];
      else if (baseline === "mean") signal = Gs[i] - meanG;
      else signal = Gs[i] - Vstar;
      grads[i] = score * signal;
    }
    const gMean = grads.reduce((a, b) => a + b, 0) / N_RUNS;
    const gVar = grads.reduce((acc, g) => acc + (g - gMean) ** 2, 0) / N_RUNS;
    const gStd = Math.sqrt(gVar);

    // Pin domain to the largest |g| we've seen across modes (compute from
    // the *no-baseline* envelope so the histogram doesn't rescale when
    // baselines tighten — the eye reads "width shrinks" not "axis grows".
    const envelope = Math.max(...ARM_MEANS) + 3 * Math.max(...ARM_STDS);
    const xMax = Math.max(Math.abs(1 - pi0), pi0) * envelope * 1.05;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "ĝ (single-run REINFORCE gradient on θ)",
           domain: [-xMax, xMax], grid: true },
      y: { label: `count (${N_RUNS} runs)`, grid: true },
      marks: [
        Plot.rectY(
          grads.map((g) => ({ g })),
          Plot.binX(
            { y: "count" },
            { x: "g", fill: palette.primary, fillOpacity: 0.75, thresholds: 50 },
          ),
        ),
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.ruleX([gMean], { stroke: palette.danger, strokeWidth: 1.5, ...dashed }),
        Plot.text(
          [{ x: gMean, y: 0, label: `mean ĝ = ${gMean.toFixed(3)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4, dy: -6,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    slots.readout.textContent =
      `π(a=0) = ${pi0.toFixed(2)}  ·  V* = ${Vstar.toFixed(3)}  ·  ` +
      `Var(ĝ) = ${gVar.toFixed(4)}  ·  std(ĝ) = ${gStd.toFixed(3)}  ·  ` +
      `mean(ĝ) = ${gMean.toFixed(3)} (≈ true ∇J regardless of baseline)`;
  },
});

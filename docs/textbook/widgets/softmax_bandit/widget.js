// Widget 10.D — Softmax policy on a 2-arm linear bandit.
//
// Two arms with means μ₁, μ₂. Policy is π(a=1) = sigmoid(θ). Run
// REINFORCE for 500 steps; plot θ trajectory and π(a=1) trajectory.
// Pedagogy: as the μ-gap shrinks the learning slows — the policy
// gradient ∝ (μ₂ - μ₁) · π(1-π), so shrinking the gap directly shrinks
// the per-step drift.
//
// Mount:
//     <div id="ch10-softmax-bandit-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/softmax_bandit/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const T = 500;

function sigmoid(x) {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

// One REINFORCE pass over a 2-arm Gaussian bandit with softmax (=sigmoid
// on a single scalar logit). θ ← θ + α · ∇log π(a) · r.
//   π(a=1) = σ(θ); π(a=0) = 1 − σ(θ).
//   ∇log π(1)/∇θ =  (1 − σ(θ));  ∇log π(0)/∇θ = −σ(θ).
function runReinforce(mu1, mu2, alpha, sigma, seed) {
  const rng = mulberry32(seed);
  let theta = 0;
  const thetas = new Array(T + 1);
  const pis = new Array(T + 1);
  const avgR = new Array(T + 1);
  thetas[0] = theta;
  pis[0] = sigmoid(theta);
  avgR[0] = 0;
  let rsum = 0;
  for (let t = 0; t < T; t++) {
    const p1 = sigmoid(theta);
    const a = rng() < p1 ? 1 : 0;
    const mean = a === 1 ? mu2 : mu1;
    const r = mean + sigma * gauss(rng);
    const grad = a === 1 ? (1 - p1) : -p1;
    theta += alpha * grad * r;
    thetas[t + 1] = theta;
    pis[t + 1] = sigmoid(theta);
    rsum += r;
    avgR[t + 1] = rsum / (t + 1);
  }
  return { thetas, pis, avgR };
}

defineWidget({
  hostId: "ch10-softmax-bandit-widget",
  controls: {
    mu1:   { label: "μ₁ (arm 0)", min: 0, max: 1, step: 0.01, default: 0.40 },
    mu2:   { label: "μ₂ (arm 1)", min: 0, max: 1, step: 0.01, default: 0.60 },
    alpha: { label: "α (learning rate)", min: 0.001, max: 0.5, step: 0.001, default: 0.05 },
    sigma: { label: "σ (noise)", min: 0, max: 0.5, step: 0.01, default: 0.10 },
    seed:  { label: "seed", min: 1, max: 999, step: 1, default: 7 },
  },
  slots: ["main", "extra"],
  render: (host, { mu1, mu2, alpha, sigma, seed }, slots) => {
    const { thetas, pis, avgR } = runReinforce(mu1, mu2, alpha, sigma, Math.round(seed));

    // Downsample for plotting speed.
    const stride = Math.max(1, Math.floor((T + 1) / 200));
    const thetaData = [];
    const piData = [];
    for (let t = 0; t <= T; t += stride) {
      thetaData.push({ t, value: thetas[t], series: "θ" });
      piData.push({ t, value: pis[t], series: "π(a=1)" });
    }
    thetaData.push({ t: T, value: thetas[T], series: "θ" });
    piData.push({ t: T, value: pis[T], series: "π(a=1)" });

    // π trajectory + average reward.
    const rewardData = [];
    for (let t = 0; t <= T; t += stride) {
      rewardData.push({ t, value: avgR[t] });
    }
    rewardData.push({ t: T, value: avgR[T] });

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "step t", grid: true, domain: [0, T] },
      y: { label: "π(a=1)", grid: true, domain: [-0.05, 1.05] },
      marks: [
        Plot.ruleY([0.5], { stroke: palette.muted, ...dashed }),
        Plot.ruleY([1.0, 0.0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.line(piData, { x: "t", y: "value", stroke: palette.primary, strokeWidth: 2 }),
      ],
    }));

    // Bottom: θ over time, and average reward.
    slots.extra.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "step t", grid: true, domain: [0, T] },
      y: { label: "θ", grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(thetaData, { x: "t", y: "value", stroke: palette.secondary, strokeWidth: 2 }),
      ],
    }));

    const gap = mu2 - mu1;
    const finalPi = pis[T];
    const optimalArm = mu2 >= mu1 ? 1 : 0;
    slots.readout.textContent =
      `μ-gap = ${gap.toFixed(2)};  θ_final = ${thetas[T].toFixed(2)};  ` +
      `π(a=1)_final = ${finalPi.toFixed(3)};  ` +
      `avg reward = ${avgR[T].toFixed(3)};  optimal arm = ${optimalArm}.`;
  },
});

// Widget 18.D — H-PPO hybrid-head action sampler (Chapter 18).
//
// Two-head PAMDP policy from [Fan et al. 2019]:
//   - discrete head: softmax over 3 action types (logits ℓ₁, ℓ₂, ℓ₃)
//   - continuous head: per-type Gaussian over parameter ∈ [-3, 3]
//                      (means μ₁, μ₂, μ₃ at fixed σ controlled by σ slider)
// Sampling: draw type k ~ softmax(ℓ); draw x ~ N(μ_k, σ²). The widget
// draws N samples and plots:
//   - top: joint scatter (type, x), coloured by type
//   - bottom: per-type marginal histograms of x
//
// Pedagogy: the joint factors as p(k, x) = softmax(ℓ)_k · N(x; μ_k, σ²)
// — H-PPO's contribution is showing this factored form trains stably
// under PPO. Sliding logits redistributes type mass; sliding μ_k
// shifts only that type's column.
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-hppo-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/hppo_sampler/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";
import { lcg, gauss, sampleDiscrete } from "../shared/random.js";

function softmax(xs) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

const TYPE_LABELS = ["type 1", "type 2", "type 3"];
const TYPE_COLORS = [palette.primary, palette.secondary, palette.accent];

defineWidget({
  hostId: "ch18-hppo-widget",
  slots: ["scatter", "marginal"],
  controls: {
    l1: { label: "logit ℓ₁", min: -3, max: 3, step: 0.1, default: 0.5 },
    l2: { label: "logit ℓ₂", min: -3, max: 3, step: 0.1, default: 0.0 },
    l3: { label: "logit ℓ₃", min: -3, max: 3, step: 0.1, default: -0.5 },
    m1: { label: "μ₁", min: -3, max: 3, step: 0.05, default: -1.5 },
    m2: { label: "μ₂", min: -3, max: 3, step: 0.05, default: 0.0 },
    m3: { label: "μ₃", min: -3, max: 3, step: 0.05, default: 1.5 },
    sigma: { label: "σ (shared)", min: 0.1, max: 1.5, step: 0.05, default: 0.5 },
  },
  render: (host, p, slots) => {
    const logits = [p.l1, p.l2, p.l3];
    const mus = [p.m1, p.m2, p.m3];
    const probs = softmax(logits);

    const N = 800;
    const rng = lcg(42);
    const samples = [];
    for (let i = 0; i < N; i++) {
      // Sample type from the softmax distribution.
      const k = sampleDiscrete(rng, probs);
      const x = mus[k] + p.sigma * gauss(rng);
      samples.push({ type: TYPE_LABELS[k], typeIdx: k, x });
    }

    // Top: joint scatter — y-jitter by type so the columns separate
    // vertically. x is the continuous parameter; type is encoded by
    // colour AND by row band.
    const jitter = lcg(7);
    const scatter = samples.map((s) => ({
      ...s, yBand: s.typeIdx + 0.5 + (jitter() - 0.5) * 0.7,
    }));

    slots.scatter.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      marginLeft: 70,
      x: { label: "parameter x", domain: [-3.5, 3.5], grid: true },
      y: { label: "type", domain: [0, 3], ticks: [0.5, 1.5, 2.5],
           tickFormat: (d) => TYPE_LABELS[Math.floor(d)] || "" },
      color: { domain: TYPE_LABELS, range: TYPE_COLORS, legend: true },
      marks: [
        Plot.ruleY([1, 2], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.dot(scatter, {
          x: "x", y: "yBand", fill: "type",
          fillOpacity: 0.5, r: 2.5,
        }),
        // Mark each μ_k.
        Plot.tickX(mus.map((m, k) => ({ mu: m, k })), {
          x: "mu", y: (d) => d.k + 0.5,
          stroke: (d) => TYPE_COLORS[d.k], strokeWidth: 3,
        }),
      ],
    }));

    // Bottom: per-type histograms of x. Use Plot.rectY with binX.
    slots.marginal.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 70,
      x: { label: "parameter x", domain: [-3.5, 3.5], grid: true },
      y: { label: "count", grid: true },
      color: { domain: TYPE_LABELS, range: TYPE_COLORS },
      marks: [
        Plot.rectY(
          samples,
          Plot.binX(
            { y: "count" },
            { x: "x", fill: "type", fillOpacity: 0.55,
              thresholds: 40, mixBlendMode: "screen" },
          ),
        ),
        Plot.ruleY([0]),
      ],
    }));

    slots.readout.innerHTML =
      `softmax type probs = [${probs.map((q) => q.toFixed(2)).join(", ")}]   ·   ` +
      `μ = [${mus.map((m) => m.toFixed(2)).join(", ")}]   ·   σ = ${p.sigma.toFixed(2)}   ·   ` +
      `<small>N = ${N} samples drawn deterministically</small>`;
  },
});

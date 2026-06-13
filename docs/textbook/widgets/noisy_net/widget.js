// Widget 9.G — NoisyNet parameter-noise demo (Chapter 9).
//
// Section §9.8 replaces ε-greedy with **parametric noise on the
// network weights**:
//
//     w = μ + σ ⊙ ξ,   ξ ~ N(0, I)
//
// Two panels:
//
//   (a) Scatter of 200 sampled weight vectors w ∈ ℝ² in the (w₀, w₁)
//       plane. With σ → 0 the cloud collapses to the deterministic μ
//       (greedy); as σ grows the cloud expands, sampling a wider band
//       of behaviour policies.
//
//   (b) The induced action distribution. We assume a 4-action linear
//       policy with action logits z_a = w · φ_a + b_a for fixed feature
//       prototypes φ_a (the 4 corners of the unit square) and zero
//       bias. For each sampled w we softmax z_a → π(a|s), then average
//       across the 200 samples → the *marginal* policy. As σ ↑ the
//       marginal softmax flattens; that's what "uncertainty translates
//       into policy stochasticity" means in NoisyNet language.
//
// Slider for μ₀, μ₁ (the deterministic point) and σ (shared scale).
//
// Pattern: chapter markdown contains
//
//     <div id="ch9-noisy-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/noisy_net/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";

const N_SAMPLES = 200;
// 4 actions with prototype features on the unit-square corners.
const PHI = [
  [+1, +1], // a₀ NE
  [+1, -1], // a₁ SE
  [-1, -1], // a₂ SW
  [-1, +1], // a₃ NW
];
const ACTION_LABELS = ["a₀ (NE)", "a₁ (SE)", "a₂ (SW)", "a₃ (NW)"];

function randn() {
  const u = Math.max(Math.random(), 1e-12);
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function softmax(zs) {
  const m = Math.max(...zs);
  const exps = zs.map((z) => Math.exp(z - m));
  const Z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / Z);
}

defineWidget({
  hostId: "ch9-noisy-widget",
  controls: {
    mu0:   { label: "μ₀", min: -2, max: 2, step: 0.05, default: 0.6 },
    mu1:   { label: "μ₁", min: -2, max: 2, step: 0.05, default: 0.3 },
    sigma: { label: "σ (noise scale)", min: 0.0, max: 2.0, step: 0.02, default: 0.5 },
  },
  slots: ["scatter", "actions"],
  render: (host, { mu0, mu1, sigma }, slots) => {
    // Sample N_SAMPLES weight vectors and compute their induced policies.
    const samples = [];
    const marginal = new Array(PHI.length).fill(0);
    let modeAgreeCount = 0;
    const greedyZ = PHI.map((p) => p[0] * mu0 + p[1] * mu1);
    const greedyA = greedyZ.indexOf(Math.max(...greedyZ));

    for (let i = 0; i < N_SAMPLES; i++) {
      const w0 = mu0 + sigma * randn();
      const w1 = mu1 + sigma * randn();
      const z = PHI.map((p) => p[0] * w0 + p[1] * w1);
      const argmax = z.indexOf(Math.max(...z));
      const pi = softmax(z);
      for (let a = 0; a < PHI.length; a++) marginal[a] += pi[a];
      samples.push({ w0, w1, argmax, label: ACTION_LABELS[argmax] });
      if (argmax === greedyA) modeAgreeCount += 1;
    }
    for (let a = 0; a < PHI.length; a++) marginal[a] /= N_SAMPLES;

    // (a) Scatter of sampled weights, coloured by argmax action.
    const actionPalette = [palette.primary, palette.secondary, palette.danger, palette.accent];
    const colorByLabel = (label) => actionPalette[ACTION_LABELS.indexOf(label)];

    const lim = Math.max(2.5, Math.abs(mu0) + 3 * sigma, Math.abs(mu1) + 3 * sigma);
    slots.scatter.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      width: 320,
      marginLeft: 50,
      x: { label: "w₀", domain: [-lim, lim], grid: true },
      y: { label: "w₁", domain: [-lim, lim], grid: true },
      color: { legend: true, domain: ACTION_LABELS, range: actionPalette },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Greedy decision boundaries: the 4 quadrants line up with the
        // 4 prototype corners, so boundary is just the axes — already
        // drawn above.
        Plot.dot(samples, {
          x: "w0",
          y: "w1",
          fill: (d) => colorByLabel(d.label),
          fillOpacity: 0.55,
          r: 3,
        }),
        // μ as a big crosshair.
        Plot.dot([{ x: mu0, y: mu1 }], { x: "x", y: "y", fill: "white", stroke: "#111", r: 6 }),
        // 1σ confidence circle as 64-pt ring.
        Plot.line(
          Array.from({ length: 65 }, (_, k) => {
            const a = (k * 2 * Math.PI) / 64;
            return { x: mu0 + sigma * Math.cos(a), y: mu1 + sigma * Math.sin(a) };
          }),
          { x: "x", y: "y", stroke: palette.muted, ...dashed },
        ),
      ],
    }));

    // (b) Marginal action distribution as bars.
    const bars = ACTION_LABELS.map((lab, a) => ({
      action: lab,
      prob: marginal[a],
      isGreedy: a === greedyA,
    }));
    slots.actions.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      width: 320,
      marginLeft: 70,
      x: { label: "marginal π(a)", domain: [0, 1], grid: true },
      y: { label: null, domain: ACTION_LABELS },
      marks: [
        Plot.barX(bars, {
          x: "prob",
          y: "action",
          fill: (d) => colorByLabel(d.action),
          fillOpacity: (d) => (d.isGreedy ? 0.95 : 0.55),
          stroke: (d) => (d.isGreedy ? "#fff" : "transparent"),
          strokeWidth: 1.5,
        }),
        Plot.text(bars, {
          x: "prob",
          y: "action",
          text: (d) => d.prob.toFixed(3),
          dx: 6,
          textAnchor: "start",
          fill: "#ddd",
          fontSize: 11,
        }),
        Plot.ruleX([1 / PHI.length], { stroke: palette.muted, ...dashed }),
      ],
    }));

    // Entropy of marginal policy (bits) — quantifies "how exploratory."
    let H = 0;
    for (const p of marginal) if (p > 0) H -= p * Math.log2(p);
    const Hmax = Math.log2(PHI.length);
    slots.readout.textContent =
      `μ = (${mu0.toFixed(2)}, ${mu1.toFixed(2)}), σ = ${sigma.toFixed(2)} · ` +
      `argmax matches greedy on ${modeAgreeCount}/${N_SAMPLES} samples · ` +
      `H(π) = ${H.toFixed(3)} bits (max = ${Hmax.toFixed(2)})`;
  },
});

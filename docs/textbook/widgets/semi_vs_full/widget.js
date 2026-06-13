// Widget 8.E — Semi-gradient vs full-gradient TD (Chapter 8, §8.3).
//
// The "semi-" in semi-gradient TD means: differentiate the prediction
// but NOT the bootstrap target. The full-gradient variant (sometimes
// called "residual gradient") propagates ∂/∂θ through both φ(s)ᵀθ and
// γφ(s')ᵀθ. On-policy and tabular, this is a curiosity; off-policy
// linear, semi-gradient can diverge while full-gradient stays bounded.
//
// This widget runs both on a tiny 3-state Baird-style off-policy MDP
// with 2-D linear features. The semi-gradient weights blow up; the
// full-gradient version stays in a small region. We plot the (w₁, w₂)
// trajectory of each as a polyline so the divergence is visible.
//
//     <div id="ch8-semi-vs-full-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/semi_vs_full/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

// 3-state MDP: s0, s1, s2. Off-policy target policy: s_i → s2 always.
// Behaviour policy: uniformly choose any next-state.
// True V^π = 0 (reward 0 everywhere; γ < 1). Features:
//   φ(s0) = (2, 1), φ(s1) = (1, 2), φ(s2) = (1, 1).
// The features span ℝ², so V^π = 0 is exactly representable with w = 0.
const PHI = [
  [2.0, 1.0],
  [1.0, 2.0],
  [1.0, 1.0],
];
const N_STATES = 3;

function dot2(w, phi) {
  return w[0] * phi[0] + w[1] * phi[1];
}

// Off-policy ρ: behaviour is uniform-over-next-states (prob 1/3 each).
// Target is "always go to s2" — prob 1 for the s2 transition, 0 otherwise.
// Importance ratio ρ = π/b = 1 / (1/3) = 3 when we sample the s2
// transition under behaviour; we only update on those transitions.
const RHO = 3.0;

function simulate({ steps, alpha, gamma, mode, seed }) {
  const rand = mulberry32(seed | 0);
  // Initial weights: w₀ = (1.0, -0.5). Off-zero so divergence is visible.
  const w = [1.0, -0.5];
  // History: keep every-k frames so the polyline is light.
  const stride = Math.max(1, Math.floor(steps / 200));
  const hist = [{ t: 0, w0: w[0], w1: w[1] }];

  for (let t = 1; t <= steps; t++) {
    // Behaviour samples a starting state uniformly.
    const s = Math.floor(rand() * N_STATES);
    // Behaviour samples next-state uniformly.
    const sn = Math.floor(rand() * N_STATES);
    // Target policy only samples the s → s2 transition.
    if (sn !== 2) continue; // ρ = 0 for these; no update.
    const phi = PHI[s];
    const phiN = PHI[sn];
    const v  = dot2(w, phi);
    const vN = dot2(w, phiN);
    const delta = 0 /* reward */ + gamma * vN - v;
    // semi-gradient: g = φ(s)
    // full-gradient: g = φ(s) − γ·φ(s')  (chain rule through target)
    let g0, g1;
    if (mode === "semi") {
      g0 = phi[0];
      g1 = phi[1];
    } else {
      g0 = phi[0] - gamma * phiN[0];
      g1 = phi[1] - gamma * phiN[1];
    }
    w[0] += alpha * RHO * delta * g0;
    w[1] += alpha * RHO * delta * g1;
    if (t % stride === 0 || t === steps) {
      hist.push({ t, w0: w[0], w1: w[1] });
    }
  }
  return hist;
}

defineWidget({
  hostId: "ch8-semi-vs-full-widget",
  controls: {
    alpha: { label: "α",     min: 0.01, max: 0.3,  step: 0.005, default: 0.05 },
    gamma: { label: "γ",     min: 0.5,  max: 0.99, step: 0.01,  default: 0.95 },
    steps: { label: "steps", min: 100,  max: 2000, step: 100,   default: 600 },
    seed:  { label: "seed",  min: 1,    max: 50,   step: 1,     default: 7 },
  },
  slots: ["main"],
  render: (host, { alpha, gamma, steps, seed }, slots) => {
    const semi = simulate({ steps: steps | 0, alpha, gamma, mode: "semi", seed: seed | 0 });
    const full = simulate({ steps: steps | 0, alpha, gamma, mode: "full", seed: seed | 0 });

    // Choose a symmetric domain that contains both trajectories.
    let m = 1.5;
    for (const h of semi) m = Math.max(m, Math.abs(h.w0), Math.abs(h.w1));
    for (const h of full) m = Math.max(m, Math.abs(h.w0), Math.abs(h.w1));
    m = Math.min(m * 1.1, 100); // cap so divergence doesn't blow the chart out
    const domain = [-m, m];

    const data = [];
    for (const h of semi) data.push({ w0: h.w0, w1: h.w1, run: "semi-gradient" });
    for (const h of full) data.push({ w0: h.w0, w1: h.w1, run: "full-gradient" });

    const last = (a) => a[a.length - 1];
    const semiLast = last(semi);
    const fullLast = last(full);
    const endpoints = [
      { w0: semiLast.w0, w1: semiLast.w1, label: "semi" },
      { w0: fullLast.w0, w1: fullLast.w1, label: "full" },
    ];
    const start = [{ w0: semi[0].w0, w1: semi[0].w1, label: "start" }];

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 360,
      width: 420,
      marginLeft: 60,
      x: { label: "w₁", domain, grid: true },
      y: { label: "w₂", domain, grid: true },
      color: {
        legend: true,
        domain: ["semi-gradient", "full-gradient"],
        range: [palette.danger, palette.primary],
      },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, ...dashed }),
        Plot.ruleY([0], { stroke: palette.muted, ...dashed }),
        Plot.dot([{ w0: 0, w1: 0 }], { x: "w0", y: "w1", fill: palette.warning, r: 4, symbol: "diamond" }),
        Plot.text([{ w0: 0, w1: 0, label: "w* = 0" }], {
          x: "w0", y: "w1", text: "label", dx: 8, dy: -8,
          fill: palette.warning, fontSize: 10,
        }),
        Plot.line(data, { x: "w0", y: "w1", stroke: "run", strokeWidth: 1.6 }),
        Plot.dot(start, { x: "w0", y: "w1", fill: palette.muted, r: 3 }),
        Plot.dot(endpoints, { x: "w0", y: "w1", fill: ["#ff5252", "#69f0ae"], r: 4 }),
        Plot.text(endpoints, {
          x: "w0", y: "w1", text: "label", dx: 8, dy: 4,
          fontSize: 10, fill: palette.muted,
        }),
      ],
    }));

    const normSemi = Math.hypot(semiLast.w0, semiLast.w1);
    const normFull = Math.hypot(fullLast.w0, fullLast.w1);
    slots.readout.textContent =
      `final ‖w‖ — semi: ${fmt(normSemi)}  ·  full: ${fmt(normFull)}  ·  true ‖w*‖ = 0`;
  },
});

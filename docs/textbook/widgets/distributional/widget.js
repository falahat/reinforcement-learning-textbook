// Widget 9.F — Distributional return: what DQN throws away (Chapter 9, §9.7).
//
// C51 / QR-DQN learn the FULL distribution of the random return Z =
// Σ γᵗ r_t, not just its expectation E[Z]. To see why that's richer,
// we compute the exact return distribution for a tiny MDP under a
// fixed policy:
//
//   - One state with two stochastic-reward outcomes:
//     50% chance reward +1, 50% chance reward -1 (mean 0).
//   - One terminal state with reward +5 (worth +5).
//   - From the initial state, 70% transition stays self-loop (the
//     ±1 outcome), 30% terminates with +5.
//
// The return distribution is a mixture-of-shifted-and-scaled copies of
// the per-step reward distribution — it's multi-modal under nontrivial
// γ. We compute it exactly by iterating: D_{k+1} = ∑ p(r,s') (r + γ D_k(s')),
// using a 21-bin histogram with linear interpolation projection (the
// C51 trick).
//
//     <div id="ch9-distributional-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/distributional/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";
import { unitBarsY } from "../shared/plot.js";

const N_BINS = 21;
const V_MIN = -5;
const V_MAX = 10;
const DZ = (V_MAX - V_MIN) / (N_BINS - 1);
const ATOMS = Array.from({ length: N_BINS }, (_, i) => V_MIN + i * DZ);

// Project a point-mass (probability `p` at value `v`) onto the fixed
// support {z_0, ..., z_{N-1}} via linear interpolation (the C51 update).
function project(p, v, into) {
  const vClip = Math.min(V_MAX, Math.max(V_MIN, v));
  const idx = (vClip - V_MIN) / DZ;
  const lo = Math.floor(idx);
  const hi = Math.min(N_BINS - 1, lo + 1);
  const frac = idx - lo;
  into[lo] += p * (1 - frac);
  if (hi !== lo) into[hi] += p * frac;
}

// Compute the value distribution for the looping state under the
// fixed policy, by fixed-point iteration of the categorical Bellman
// operator. p_stay = 0.7 (stochastic ±1 reward, loop back); p_term =
// 0.3 (terminal reward +5).
function valueDistribution(gamma) {
  const P_STAY = 0.7;
  const P_TERM = 0.3;
  // Terminal: dist is a delta at 0 (no further return).
  // Stay: stochastic immediate reward (+1 or -1, prob 0.5 each), then
  // bootstrap by γ·D_loop.
  let dist = new Float64Array(N_BINS);
  // Initialise dist with a delta at 0.
  project(1.0, 0.0, dist);

  for (let iter = 0; iter < 200; iter++) {
    const next = new Float64Array(N_BINS);
    // Terminal branch (prob 0.3): immediate reward +5, no future.
    project(P_TERM, 5.0, next);
    // Stay branch (prob 0.7): immediate ±1, then γ·D_loop.
    // For each atom in current dist, two child atoms (±1 + γz).
    for (let i = 0; i < N_BINS; i++) {
      const pi = dist[i];
      if (pi === 0) continue;
      const z = ATOMS[i];
      project(P_STAY * 0.5, +1.0 + gamma * z, next);
      project(P_STAY * 0.5, -1.0 + gamma * z, next);
    }
    // L1 convergence check.
    let l1 = 0;
    for (let i = 0; i < N_BINS; i++) l1 += Math.abs(next[i] - dist[i]);
    dist = next;
    if (l1 < 1e-8) break;
  }
  return dist;
}

defineWidget({
  hostId: "ch9-distributional-widget",
  controls: {
    gamma: { label: "γ", min: 0.0, max: 0.95, step: 0.05, default: 0.7 },
  },
  slots: ["main"],
  render: (host, { gamma }, slots) => {
    const dist = valueDistribution(gamma);
    let mean = 0;
    for (let i = 0; i < N_BINS; i++) mean += ATOMS[i] * dist[i];
    let varZ = 0;
    for (let i = 0; i < N_BINS; i++) {
      const d = ATOMS[i] - mean;
      varZ += dist[i] * d * d;
    }
    // Standardised third moment (skewness).
    let skew = 0;
    const std = Math.sqrt(varZ);
    if (std > 1e-9) {
      for (let i = 0; i < N_BINS; i++) {
        const d = (ATOMS[i] - mean) / std;
        skew += dist[i] * d * d * d;
      }
    }

    const data = ATOMS.map((z, i) => ({ z, p: dist[i] }));
    const yMax = Math.max(0.1, Math.max(...data.map((d) => d.p)) * 1.15);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "return Z", grid: true, domain: [V_MIN - 0.5, V_MAX + 0.5] },
      y: { label: "P(Z = z)", domain: [0, yMax], grid: true },
      marks: [
        unitBarsY(data, { x: "z", y: "p", fill: palette.secondary, fillOpacity: 0.85 }),
        Plot.ruleX([mean], { stroke: palette.danger, strokeWidth: 2, ...dashed }),
        Plot.text([{ x: mean, y: yMax, label: `E[Z] = ${fmt(mean)} (what DQN learns)` }], {
          x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: 10,
          fill: palette.danger, fontSize: 10,
        }),
      ],
    }));

    slots.readout.textContent =
      `E[Z] = ${fmt(mean)}  ·  Var(Z) = ${fmt(varZ)}  ·  skew = ${fmt(skew)}  ·  γ = ${fmt(gamma)}`;
  },
});

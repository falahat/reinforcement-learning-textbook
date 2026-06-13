// Widget 3.C — Stochastic vs deterministic policy explorer (Chapter 3).
//
// 3×3 gridworld with one goal cell (top-right, reward +1) and one
// "pit" cell (middle, reward −1). Step reward 0 elsewhere. Deterministic
// transitions in NESW (bumping a wall stays). γ = 0.9. Discount-free
// terminal: goal is absorbing, but value evaluation uses the equation
//
//     V^π(s) = Σ_a π(a|s) [R(s,a,s') + γ V^π(s')]
//
// computed by iterative policy evaluation to ε = 1e-6.
//
// Two stochastic-policy presets (uniform / east-biased) plus a slider
// for "softmax temperature" τ that interpolates between uniform (τ → ∞)
// and the greedy-on-Q^π policy (τ → 0). The widget shows two heatmaps
// side by side:
//
//   - left:  V^π   (the stochastic policy)
//   - right: V^π'  where π'(s) = argmax_a Q^π(s, a)
//
// The §3.8 theorem says V^π'(s) ≥ V^π(s) pointwise — so the right
// heatmap dominates. A scalar readout shows max gap and mean gap.
//
// Mount: in §3.8 of Chapter 3.
//
//     <div id="ch3-stoch-vs-det-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/stochastic_vs_det/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const N = 3;             // grid side
const NSTATES = N * N;
const NACTIONS = 4;      // N, E, S, W
const ACTION_DX = [0, 1, 0, -1];
const ACTION_DY = [-1, 0, 1, 0];
const ACTION_LABELS = ["↑", "→", "↓", "←"];

const GOAL = { x: 2, y: 0 }; // top-right
const PIT  = { x: 1, y: 1 }; // center
const GAMMA = 0.9;

function idx(x, y) { return y * N + x; }
function coord(s) { return { x: s % N, y: Math.floor(s / N) }; }
function isTerminal(s) {
  const { x, y } = coord(s);
  return x === GOAL.x && y === GOAL.y;
}
function reward(sNext) {
  const { x, y } = coord(sNext);
  if (x === GOAL.x && y === GOAL.y) return 1;
  if (x === PIT.x && y === PIT.y) return -1;
  return 0;
}
function nextState(s, a) {
  if (isTerminal(s)) return s;
  const { x, y } = coord(s);
  const nx = Math.max(0, Math.min(N - 1, x + ACTION_DX[a]));
  const ny = Math.max(0, Math.min(N - 1, y + ACTION_DY[a]));
  return idx(nx, ny);
}

// Iterative policy evaluation. pi[s][a] is the policy as a stochastic
// matrix. Returns V[s].
function evaluatePolicy(pi, maxIter = 500, tol = 1e-6) {
  const V = new Array(NSTATES).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    let delta = 0;
    for (let s = 0; s < NSTATES; s++) {
      if (isTerminal(s)) continue;
      let v = 0;
      for (let a = 0; a < NACTIONS; a++) {
        const sp = nextState(s, a);
        v += pi[s][a] * (reward(sp) + GAMMA * V[sp]);
      }
      delta = Math.max(delta, Math.abs(V[s] - v));
      V[s] = v;
    }
    if (delta < tol) break;
  }
  return V;
}

// Q^π from V^π.
function computeQ(V) {
  const Q = Array.from({ length: NSTATES }, () => new Array(NACTIONS).fill(0));
  for (let s = 0; s < NSTATES; s++) {
    if (isTerminal(s)) continue;
    for (let a = 0; a < NACTIONS; a++) {
      const sp = nextState(s, a);
      Q[s][a] = reward(sp) + GAMMA * V[sp];
    }
  }
  return Q;
}

// Greedy (deterministic) policy from Q. Each row is one-hot at argmax.
function greedyPolicy(Q) {
  return Q.map((row, s) => {
    const out = new Array(NACTIONS).fill(0);
    if (isTerminal(s)) {
      out[0] = 1;
      return out;
    }
    let best = 0;
    for (let a = 1; a < NACTIONS; a++) if (row[a] > row[best]) best = a;
    out[best] = 1;
    return out;
  });
}

// Build stochastic policy from a preset + temperature τ. τ → ∞ ⇒ uniform,
// τ → 0 ⇒ deterministic argmax of the preset's bias scores.
function buildPolicy(preset, tau) {
  // Per-state action "bias" scores: higher = preferred. We compute these
  // from the preset shape, then softmax with temperature.
  const pi = [];
  for (let s = 0; s < NSTATES; s++) {
    const { x, y } = coord(s);
    let scores;
    if (preset === "uniform") {
      scores = [0, 0, 0, 0];
    } else if (preset === "east-biased") {
      // Strong east, mild north (toward the goal at top-right).
      scores = [0.4, 1.0, -0.4, -0.8];
    } else if (preset === "into-pit") {
      // A bad policy that prefers stepping toward the pit at (1, 1).
      const toPitX = Math.sign(PIT.x - x);
      const toPitY = Math.sign(PIT.y - y);
      scores = [
        toPitY < 0 ? 1 : -0.2,  // N
        toPitX > 0 ? 1 : -0.2,  // E
        toPitY > 0 ? 1 : -0.2,  // S
        toPitX < 0 ? 1 : -0.2,  // W
      ];
    } else {
      scores = [0, 0, 0, 0];
    }
    // Softmax with temperature τ. τ very small ⇒ argmax; τ large ⇒ uniform.
    const safeTau = Math.max(tau, 1e-3);
    const logits = scores.map((v) => v / safeTau);
    const m = Math.max(...logits);
    const ex = logits.map((v) => Math.exp(v - m));
    const z = ex.reduce((a, b) => a + b, 0) || 1;
    pi.push(ex.map((v) => v / z));
  }
  return pi;
}

// Render an N×N grid as Plot.cell heatmap.
function renderHeatmap(V, title, valueDomain) {
  const cells = [];
  for (let s = 0; s < NSTATES; s++) {
    const { x, y } = coord(s);
    cells.push({
      x, y,
      v: V[s],
      label: fmt(V[s]),
      isGoal: x === GOAL.x && y === GOAL.y,
      isPit: x === PIT.x && y === PIT.y,
    });
  }
  return Plot.plot({
    ...plotDefaults,
    width: 280,
    height: 280,
    marginTop: 30,
    marginLeft: 30,
    marginBottom: 30,
    title,
    x: { label: null, domain: [0, 1, 2], type: "band" },
    y: { label: null, domain: [0, 1, 2], type: "band", reverse: false },
    color: {
      type: "linear", scheme: "RdBu",
      domain: valueDomain, label: "V",
      legend: true,
    },
    marks: [
      Plot.cell(cells, {
        x: "x", y: "y", fill: "v", inset: 1,
      }),
      Plot.text(cells, {
        x: "x", y: "y",
        text: (d) => d.isGoal ? `★\n${d.label}` : d.isPit ? `☠\n${d.label}` : d.label,
        fill: "white", fontSize: 11,
      }),
    ],
  });
}

defineWidget({
  hostId: "ch3-stoch-vs-det-widget",
  controls: {
    preset: {
      type: "select", label: "policy preset",
      options: [
        { value: "uniform", label: "uniform random" },
        { value: "east-biased", label: "east-biased (toward goal)" },
        { value: "into-pit", label: "into-pit (bad)" },
      ],
      default: "east-biased",
    },
    tau: { label: "τ (softmax temp)", min: 0.05, max: 3.0, step: 0.05, default: 1.0 },
  },
  slots: ["stoch", "det"],
  render: (host, { preset, tau }, slots) => {
    // Build stochastic policy from preset + temperature.
    const piStoch = buildPolicy(preset, tau);
    const Vstoch = evaluatePolicy(piStoch);
    const Qstoch = computeQ(Vstoch);
    const piDet = greedyPolicy(Qstoch);
    const Vdet = evaluatePolicy(piDet);

    // Shared colour domain so the two heatmaps are comparable.
    const allV = Vstoch.concat(Vdet);
    const lo = Math.min(...allV);
    const hi = Math.max(...allV);
    const m = Math.max(Math.abs(lo), Math.abs(hi), 0.5);
    const dom = [-m, m];

    slots.stoch.replaceChildren(renderHeatmap(Vstoch, "V^π (stochastic)", dom));
    slots.det.replaceChildren(renderHeatmap(Vdet, "V^π' (deterministic, argmax Q^π)", dom));

    // Gap statistics: π' should dominate pointwise (theorem §3.8).
    let maxGap = -Infinity;
    let minGap = +Infinity;
    let sumGap = 0;
    let nonTerm = 0;
    for (let s = 0; s < NSTATES; s++) {
      if (isTerminal(s)) continue;
      const gap = Vdet[s] - Vstoch[s];
      if (gap > maxGap) maxGap = gap;
      if (gap < minGap) minGap = gap;
      sumGap += gap;
      nonTerm++;
    }
    const meanGap = sumGap / nonTerm;
    const dominates = minGap >= -1e-9;

    slots.readout.innerHTML =
      `<strong>V^π' − V^π</strong>: max = ${fmt(maxGap)}, mean = ${fmt(meanGap)}, min = ${fmt(minGap)}<br>` +
      `<small>${dominates ? "✓ deterministic dominates pointwise (theorem §3.8 holds)" : "✗ unexpected: pointwise dominance violated"} · ★ goal (+1), ☠ pit (−1), γ = ${GAMMA}</small>`;
  },
});

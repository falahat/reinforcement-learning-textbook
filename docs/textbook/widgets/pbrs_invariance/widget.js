// Widget 16.F — PBRS policy-invariance visualizer (Chapter 16).
//
// Ng-Harada-Russell 1999: for any potential Φ: S → ℝ, the shaped
// reward R̃ = R + γΦ(s') − Φ(s) yields the *same* optimal policy
// as R. This widget demonstrates the theorem on a 4×4 gridworld:
//
//   - Two side-by-side V*-heatmaps with greedy arrows. Left: raw
//     reward (step-cost = −1, goal = +1). Right: same reward
//     plus a PBRS shaping for the reader-selected potential
//     shape.
//   - Potential shapes: zero (sanity check), gradient toward
//     goal, "wrong direction" (potential rises away from goal),
//     and random (per-fixed-seed). The "gradient toward goal"
//     case is the PBRS canonical example — it speeds learning
//     without changing π*.
//   - A readout reports max-cell-deviation of |V*_raw − V*_shaped|
//     (the additive offset) and a binary "policies identical?"
//     check across all states. Both should hold: the shaped V*
//     differs from V*_raw by exactly −Φ(s) up to numerical
//     rounding, and the policies match.
//
// Mount: in §16.4 of Chapter 16. Maps to Exercise 4.
//
//     <div id="ch16-pbrs-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/pbrs_invariance/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const N = 4;
const NSTATES = N * N;
const ACTIONS = [
  { dx: 0, dy: -1, name: "↑" },
  { dx: +1, dy: 0, name: "→" },
  { dx: 0, dy: +1, name: "↓" },
  { dx: -1, dy: 0, name: "←" },
];
const GOAL_X = N - 1;
const GOAL_Y = 0;
const GOAL_S = GOAL_Y * N + GOAL_X;
const STEP_REWARD = -1;
const GOAL_REWARD = 1;

function idxOf(x, y) { return y * N + x; }
function coordOf(s)  { return { x: s % N, y: Math.floor(s / N) }; }

function nextState(s, a) {
  if (s === GOAL_S) return s;
  const { x, y } = coordOf(s);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return idxOf(nx, ny);
}

function baseReward(s, a, sp) {
  return sp === GOAL_S ? GOAL_REWARD : STEP_REWARD;
}

// Hash-based deterministic "random" so the same selection always
// produces the same heights — no global RNG state.
function deterministicRandom(s) {
  let h = (s * 2654435761) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1597334677) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

function potentialFn(shape, scale) {
  return (s) => {
    const { x, y } = coordOf(s);
    if (shape === "zero") return 0;
    // Manhattan distance to goal.
    const d = Math.abs(x - GOAL_X) + Math.abs(y - GOAL_Y);
    if (shape === "toward-goal") {
      // Higher Φ closer to goal. dmax = 2(N-1).
      return scale * (1 - d / (2 * (N - 1)));
    }
    if (shape === "away-from-goal") {
      return scale * (d / (2 * (N - 1)));
    }
    if (shape === "random") {
      return scale * deterministicRandom(s);
    }
    return 0;
  };
}

function valueIteration(rewardFn, gamma) {
  const V = new Array(NSTATES).fill(0);
  // Cap = 5000 so γ=0.99 (slider max) converges to 1e-8 before pegging.
  // At γ=0.99, log(1e-8)/log(0.99) ≈ 1834 iterations.
  for (let iter = 0; iter < 5000; iter++) {
    let delta = 0;
    for (let s = 0; s < NSTATES; s++) {
      if (s === GOAL_S) { V[s] = 0; continue; }
      let best = -Infinity;
      for (let a = 0; a < ACTIONS.length; a++) {
        const sp = nextState(s, a);
        const q = rewardFn(s, a, sp) + gamma * V[sp];
        if (q > best) best = q;
      }
      delta = Math.max(delta, Math.abs(V[s] - best));
      V[s] = best;
    }
    if (delta < 1e-8) break;
  }
  const pi = new Array(NSTATES).fill(0);
  for (let s = 0; s < NSTATES; s++) {
    if (s === GOAL_S) { pi[s] = -1; continue; }
    let bestA = 0;
    let best = -Infinity;
    for (let a = 0; a < ACTIONS.length; a++) {
      const sp = nextState(s, a);
      const q = rewardFn(s, a, sp) + gamma * V[sp];
      if (q > best) { best = q; bestA = a; }
    }
    pi[s] = bestA;
  }
  return { V, pi };
}

function renderGridPanel(title, V, pi, valueDomain) {
  const cells = [];
  for (let s = 0; s < NSTATES; s++) {
    const { x, y } = coordOf(s);
    const arrow = pi[s] >= 0 ? ACTIONS[pi[s]].name : "★";
    cells.push({
      x, y, v: V[s],
      label: s === GOAL_S ? "★ goal" : `${arrow}\n${fmt(V[s])}`,
      isGoal: s === GOAL_S,
    });
  }
  return Plot.plot({
    ...plotDefaults,
    width: 280, height: 280,
    marginTop: 32, marginLeft: 20, marginBottom: 20,
    title,
    x: { axis: null, domain: d3.range(N) },
    y: { axis: null, domain: d3.range(N) },
    color: {
      type: "linear",
      domain: valueDomain,
      range: ["#1b3a52", palette.primary],
    },
    marks: [
      Plot.cell(cells, { x: "x", y: "y", fill: "v", inset: 1 }),
      Plot.text(cells, {
        x: "x", y: "y", text: "label",
        fill: "white", fontSize: 11, lineHeight: 1.2,
      }),
    ],
  });
}

defineWidget({
  hostId: "ch16-pbrs-widget",
  controls: {
    shape: {
      type: "select",
      label: "Φ shape",
      options: [
        { value: "zero",           label: "zero (sanity check)" },
        { value: "toward-goal",    label: "gradient toward goal" },
        { value: "away-from-goal", label: "gradient AWAY from goal" },
        { value: "random",         label: "random per-cell" },
      ],
      default: "toward-goal",
    },
    scale: { label: "Φ scale c", min: 0, max: 5, step: 0.1, default: 2.0 },
    gamma: { label: "γ",         min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["panels"],
  render: (host, { shape, scale, gamma }, slots) => {
    const Phi = potentialFn(shape, scale);
    const shapedReward = (s, a, sp) =>
      baseReward(s, a, sp) + gamma * Phi(sp) - Phi(s);

    const raw     = valueIteration(baseReward,    gamma);
    const shaped  = valueIteration(shapedReward,  gamma);

    // Theoretical relation: V_shaped(s) = V_raw(s) − Φ(s). Compute
    // the residual (V_shaped − V_raw + Φ) to verify numerically.
    let maxResidual = 0;
    for (let s = 0; s < NSTATES; s++) {
      const residual = Math.abs(shaped.V[s] - raw.V[s] + Phi(s));
      if (residual > maxResidual) maxResidual = residual;
    }
    // Policy invariance check.
    let policyMatch = true;
    for (let s = 0; s < NSTATES; s++) {
      if (raw.pi[s] !== shaped.pi[s]) {
        policyMatch = false;
        break;
      }
    }

    // Shared colour domain over all V values.
    const allV = raw.V.concat(shaped.V);
    const lo = Math.min(...allV);
    const hi = Math.max(...allV);
    const dom = [lo, hi === lo ? lo + 1e-3 : hi];

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.gap = "8px";
    container.style.flexWrap = "wrap";
    container.appendChild(renderGridPanel("raw R: V* + π*",     raw.V,    raw.pi,    dom));
    container.appendChild(renderGridPanel("shaped R̃: V* + π*", shaped.V, shaped.pi, dom));
    slots.panels.replaceChildren(container);

    slots.readout.innerHTML =
      `max |V*_shaped − V*_raw + Φ| = <strong>${maxResidual.toExponential(2)}</strong> ` +
      `(should be ≈ 0) &nbsp;|&nbsp; ` +
      `optimal policies identical? <strong>${policyMatch ? "yes" : "NO"}</strong>`;
  },
});

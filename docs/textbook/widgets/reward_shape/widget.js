// Widget 3.F — Reward shape comparison: R(s) vs R(s,a) vs R(s,a,s')
// (Chapter 3).
//
// Same 2×2 gridworld in all three panels. Cells (0,0) bottom-left start,
// (1,0) bottom-right goal (+1), (0,1) top-left pit (−1), (1,1) top-right
// safe (0). Deterministic NESW transitions (bump-stays). γ = 0.9.
// Goal is absorbing.
//
// The three reward formulations score the *same* environment differently:
//
//   - R(s):       reward of arriving in s. Goal-cell only.
//   - R(s, a):    reward depends on (state, action). We add a "step
//                 toward goal" bonus for action East from start.
//   - R(s, a, s'): reward depends on (state, action, next-state). We
//                 add a "pit-aware penalty" for *attempting* a move
//                 that would step into the pit even if the agent
//                 doesn't actually move (bump). This makes "tempting
//                 but trap" cells visible to the optimal policy.
//
// Run value iteration to convergence in each variant; render the
// three (V*, π*) heatmaps side by side. The optimal policies differ:
// R(s) and R(s,a) lead to the simple "go east" policy; R(s,a,s')
// shows a different routing around the pit.
//
// Mount: in §3.1 of Chapter 3.
//
//     <div id="ch3-reward-shape-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/reward_shape/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const N = 2;
const NSTATES = N * N;
const NACTIONS = 4;        // N, E, S, W
const ACTION_DX = [0, 1, 0, -1];
const ACTION_DY = [-1, 0, 1, 0];
const ACTION_ARROWS = ["↑", "→", "↓", "←"];

// Coordinate system: y increases downward, (0,0) top-left so the goal
// is at top-right. Layout matches reading order.
const GOAL = { x: 1, y: 0 }; // top-right
const PIT  = { x: 0, y: 1 }; // bottom-left
const START = { x: 0, y: 0 }; // top-left

function idx(x, y) { return y * N + x; }
function coord(s) { return { x: s % N, y: Math.floor(s / N) }; }
function isGoal(s) { const c = coord(s); return c.x === GOAL.x && c.y === GOAL.y; }
function isPit(s)  { const c = coord(s); return c.x === PIT.x && c.y === PIT.y; }

// Next state given (s, a): bump-stays at walls. Goal is absorbing.
function nextState(s, a) {
  if (isGoal(s)) return s;
  const { x, y } = coord(s);
  const nx = x + ACTION_DX[a];
  const ny = y + ACTION_DY[a];
  if (nx < 0 || nx >= N || ny < 0 || ny >= N) return s;
  return idx(nx, ny);
}

// --- the three reward shapes ---
//
// stepCost: small negative each non-terminal step, to make policies
// prefer fewer steps. goalBonus, pitPenalty: terminal rewards.
function makeRewardFn(shape, params) {
  const { stepCost, goalBonus, pitPenalty, eastBonus, bumpPitPenalty } = params;
  if (shape === "R(s)") {
    return (s, a, sp) => {
      // Reward of arriving in sp.
      if (isGoal(sp)) return goalBonus;
      if (isPit(sp)) return pitPenalty;
      return stepCost;
    };
  }
  if (shape === "R(s,a)") {
    return (s, a, sp) => {
      // Same as R(s) for terminal arrival, plus a tiny bonus for the
      // East action regardless of outcome.
      let r = 0;
      if (isGoal(sp)) r += goalBonus;
      else if (isPit(sp)) r += pitPenalty;
      else r += stepCost;
      if (a === 1) r += eastBonus; // East
      return r;
    };
  }
  if (shape === "R(s,a,s')") {
    return (s, a, sp) => {
      // Same arrival rewards, plus a penalty for *attempting* a step
      // into the pit even if the agent bumps back. The (s, a, s')
      // formulation distinguishes attempt-vs-arrival.
      let r = 0;
      if (isGoal(sp)) r += goalBonus;
      else if (isPit(sp)) r += pitPenalty;
      else r += stepCost;
      // Aimed at the pit?
      const aimX = coord(s).x + ACTION_DX[a];
      const aimY = coord(s).y + ACTION_DY[a];
      if (aimX === PIT.x && aimY === PIT.y) r += bumpPitPenalty;
      return r;
    };
  }
  return () => 0;
}

// Value iteration to convergence. Returns { V, pi } where pi[s] = argmax_a.
// maxIter = 5000 so γ=0.99 (slider max) converges to 1e-7 (~1611 iters)
// before pegging at the cap. Pre-fix this was 500 which capped at high γ.
function valueIteration(rewardFn, gamma, maxIter = 5000, tol = 1e-7) {
  const V = new Array(NSTATES).fill(0);
  for (let iter = 0; iter < maxIter; iter++) {
    let delta = 0;
    for (let s = 0; s < NSTATES; s++) {
      if (isGoal(s)) continue;
      let best = -Infinity;
      for (let a = 0; a < NACTIONS; a++) {
        const sp = nextState(s, a);
        const q = rewardFn(s, a, sp) + gamma * V[sp];
        if (q > best) best = q;
      }
      delta = Math.max(delta, Math.abs(V[s] - best));
      V[s] = best;
    }
    if (delta < tol) break;
  }
  const pi = new Array(NSTATES).fill(0);
  for (let s = 0; s < NSTATES; s++) {
    if (isGoal(s)) { pi[s] = -1; continue; }
    let best = -Infinity;
    let bestA = 0;
    for (let a = 0; a < NACTIONS; a++) {
      const sp = nextState(s, a);
      const q = rewardFn(s, a, sp) + gamma * V[sp];
      if (q > best) { best = q; bestA = a; }
    }
    pi[s] = bestA;
  }
  return { V, pi };
}

// Render a 2x2 grid: cell colour = V, cell text = π arrow + value.
function renderPanel(title, V, pi, valueDomain) {
  const cells = [];
  for (let s = 0; s < NSTATES; s++) {
    const { x, y } = coord(s);
    const arrow = pi[s] >= 0 ? ACTION_ARROWS[pi[s]] : "★";
    cells.push({
      x, y, v: V[s], label: `${arrow}\n${fmt(V[s])}`,
      isGoal: isGoal(s), isPit: isPit(s), isStart: x === START.x && y === START.y,
    });
  }
  return Plot.plot({
    ...plotDefaults,
    width: 240, height: 240,
    marginTop: 32, marginLeft: 24, marginBottom: 24,
    title,
    x: { label: null, domain: [0, 1], type: "band" },
    y: { label: null, domain: [0, 1], type: "band", reverse: false },
    color: { type: "linear", scheme: "RdBu", domain: valueDomain },
    marks: [
      Plot.cell(cells, { x: "x", y: "y", fill: "v", inset: 1 }),
      Plot.text(cells, {
        x: "x", y: "y",
        text: (d) => d.isGoal ? "★ goal" : d.isPit ? "☠ pit" : d.label,
        fill: "white", fontSize: 12, lineHeight: 1.2,
      }),
    ],
  });
}

defineWidget({
  hostId: "ch3-reward-shape-widget",
  controls: {
    stepCost:       { label: "step cost", min: -0.5, max: 0, step: 0.01, default: -0.1 },
    goalBonus:      { label: "goal bonus", min: 0, max: 5, step: 0.1, default: 1.0 },
    pitPenalty:     { label: "pit penalty", min: -5, max: 0, step: 0.1, default: -1.0 },
    eastBonus:      { label: "east bonus (R(s,a))", min: 0, max: 1, step: 0.05, default: 0.3 },
    bumpPitPenalty: { label: "pit-attempt penalty (R(s,a,s'))", min: -2, max: 0, step: 0.05, default: -0.6 },
    gamma:          { label: "γ", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["panels"],
  render: (host, params, slots) => {
    const shapes = ["R(s)", "R(s,a)", "R(s,a,s')"];
    const results = shapes.map((shape) => {
      const rfn = makeRewardFn(shape, params);
      return { shape, ...valueIteration(rfn, params.gamma) };
    });

    // Shared colour domain over all V values.
    const allV = results.flatMap((r) => r.V);
    const lo = Math.min(0, ...allV);
    const hi = Math.max(0, ...allV);
    const m = Math.max(Math.abs(lo), Math.abs(hi), 0.5);
    const dom = [-m, m];

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.gap = "6px";
    container.style.flexWrap = "wrap";
    for (const r of results) {
      container.appendChild(renderPanel(r.shape, r.V, r.pi, dom));
    }
    slots.panels.replaceChildren(container);

    // Diff readout: do the three policies differ? Report per-state.
    const policyStrs = results.map((r) => {
      return r.pi.map((a, s) => isGoal(s) ? "★" : ACTION_ARROWS[a]).join("");
    });
    const allEqual = policyStrs.every((s) => s === policyStrs[0]);
    const diffMsg = allEqual
      ? "<small>(all three reward shapes produce the same optimal policy on this MDP — try lowering the pit-attempt penalty or raising the east bonus.)</small>"
      : "<small>(the three reward shapes produce <strong>different</strong> optimal policies — the choice of formulation matters.)</small>";

    slots.readout.innerHTML =
      `<strong>optimal policies</strong>: R(s)=<code>${policyStrs[0]}</code>, ` +
      `R(s,a)=<code>${policyStrs[1]}</code>, R(s,a,s')=<code>${policyStrs[2]}</code><br>` +
      diffMsg;
  },
});

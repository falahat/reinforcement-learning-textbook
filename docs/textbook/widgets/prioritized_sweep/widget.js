// Widget 4.D — Prioritized sweeping vs vanilla VI (Chapter 4).
//
// Sparse-reward 5x5 gridworld: only the top-right "goal" cell has
// non-zero reward (+1, terminal); every other transition is 0.
// γ < 1 ensures the value function decays away from the goal.
//
// Vanilla VI is wasteful here — most states have residual ≈ 0 each
// sweep, so most of the |S| backups per sweep are pure overhead.
// Prioritized sweeping [Moore & Atkeson 1993] keeps a max-heap keyed
// by Bellman residual and updates one state at a time, propagating
// errors backward to predecessors only when they cross threshold.
//
// Stepper trajectory:
//   The "step" is one *backup* — for VI that's one state in a fixed
//   row-major sweep; for PS that's a heap-pop. Frames advance both
//   sides simultaneously so the bookkeeping is comparable.
//
// Slot layout:
//   vi    : VI heatmap (uniform fill ordering)
//   ps    : PS heatmap with flashing on the popped state
//   queue : sidebar text view of the PS priority queue
//
// Pattern: chapter markdown contains
//
//     <div id="ch4-prioritized-sweep-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/prioritized_sweep/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const N = 5;
const NUM_S = N * N;
const ACTIONS = [
  { dx: 0, dy: -1 },
  { dx: 0, dy: +1 },
  { dx: -1, dy: 0 },
  { dx: +1, dy: 0 },
];
const NUM_A = ACTIONS.length;
const GOAL_S = 0 * N + (N - 1); // top-right
const GOAL_REWARD = 1;
const MAX_BACKUPS = 200;
const PS_THRESH = 1e-4;

function nextState(s, a) {
  if (s === GOAL_S) return s;
  const x = s % N;
  const y = Math.floor(s / N);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return ny * N + nx;
}

// reward(s, a, s') — only +1 when entering goal from outside.
function reward(s, a, sp) {
  if (s !== GOAL_S && sp === GOAL_S) return GOAL_REWARD;
  return 0;
}

// One Bellman optimality backup at s: V[s] ← max_a [r + γ V[s']].
function backup(V, s, gamma) {
  if (s === GOAL_S) return 0;
  let best = -Infinity;
  for (let a = 0; a < NUM_A; a++) {
    const sp = nextState(s, a);
    const q = reward(s, a, sp) + gamma * V[sp];
    if (q > best) best = q;
  }
  return best;
}

// Predecessors of s: all (s', a) with nextState(s', a) = s.
function predecessors(s) {
  const preds = new Set();
  for (let sp = 0; sp < NUM_S; sp++) {
    if (sp === GOAL_S) continue;
    for (let a = 0; a < NUM_A; a++) {
      if (nextState(sp, a) === s) {
        preds.add(sp);
        break;
      }
    }
  }
  return [...preds];
}

// Precompute predecessor table (independent of γ).
const PREDS = [];
for (let s = 0; s < NUM_S; s++) PREDS.push(predecessors(s));

function computeTrajectory(gamma) {
  // VI: fixed sweep over states in row-major order, one backup per step.
  const viV = new Array(NUM_S).fill(0);
  // PS: priority queue keyed by current Bellman residual; seed with
  // states that currently have non-zero residual (i.e., goal predecessors).
  const psV = new Array(NUM_S).fill(0);

  // PS initial residuals.
  const residual = (V, s) => Math.abs(backup(V, s, gamma) - V[s]);
  const pq = []; // [{ s, p }] sorted descending by p
  for (let s = 0; s < NUM_S; s++) {
    if (s === GOAL_S) continue;
    const r = residual(psV, s);
    if (r > PS_THRESH) pq.push({ s, p: r });
  }
  pq.sort((a, b) => b.p - a.p);

  const frames = [];
  frames.push({
    step: 0,
    viV: viV.slice(),
    viCurrent: -1,
    psV: psV.slice(),
    psCurrent: -1,
    pq: pq.slice(),
  });

  for (let step = 1; step <= MAX_BACKUPS; step++) {
    // --- VI: pick next state in row-major sweep ---
    const viIdx = (step - 1) % NUM_S;
    if (viIdx !== GOAL_S) {
      viV[viIdx] = backup(viV, viIdx, gamma);
    }

    // --- PS: pop highest-priority state ---
    let psIdx = -1;
    if (pq.length > 0) {
      const top = pq.shift();
      psIdx = top.s;
      psV[psIdx] = backup(psV, psIdx, gamma);
      // Recompute residual for each predecessor; push if above thresh.
      for (const sPred of PREDS[psIdx]) {
        if (sPred === GOAL_S) continue;
        const r = residual(psV, sPred);
        // Remove any existing entry for sPred.
        const existing = pq.findIndex((e) => e.s === sPred);
        if (existing >= 0) pq.splice(existing, 1);
        if (r > PS_THRESH) {
          // Insert in sorted position (descending by p).
          let lo = 0;
          while (lo < pq.length && pq[lo].p > r) lo += 1;
          pq.splice(lo, 0, { s: sPred, p: r });
        }
      }
    }

    frames.push({
      step,
      viV: viV.slice(),
      viCurrent: viIdx,
      psV: psV.slice(),
      psCurrent: psIdx,
      pq: pq.slice(),
    });

    // Stop if PS has converged (no entries left) and VI has stabilized
    // for the past full sweep — but keep at least 30 frames so the
    // student can scrub.
    if (pq.length === 0 && step >= NUM_S * 2 && step >= 30) break;
  }
  return frames;
}

function buildHeatmap(V, current) {
  const cells = [];
  for (let s = 0; s < NUM_S; s++) {
    const x = s % N;
    const y = Math.floor(s / N);
    cells.push({
      x,
      y,
      v: V[s],
      isGoal: s === GOAL_S,
      isCurrent: s === current,
    });
  }
  return cells;
}

function heatPlot(V, current, title, vMax) {
  const cells = buildHeatmap(V, current);
  return Plot.plot({
    ...plotDefaults,
    height: 240,
    width: 260,
    marginLeft: 4,
    marginRight: 4,
    marginBottom: 16,
    marginTop: 24,
    title,
    x: { axis: null, domain: d3.range(N) },
    y: { axis: null, domain: d3.range(N) },
    color: {
      type: "linear",
      domain: [0, Math.max(vMax, 0.01)],
      range: ["#1b3a52", "#8bc34a"],
    },
    marks: [
      Plot.cell(cells, {
        x: "x",
        y: "y",
        fill: (d) => (d.isGoal ? null : d.v),
        stroke: "#111",
        strokeOpacity: 0.4,
      }),
      Plot.cell(cells.filter((c) => c.isGoal), {
        x: "x",
        y: "y",
        fill: palette.warning,
        stroke: palette.danger,
        strokeWidth: 2,
      }),
      Plot.cell(cells.filter((c) => c.isCurrent && !c.isGoal), {
        x: "x",
        y: "y",
        fill: "none",
        stroke: palette.danger,
        strokeWidth: 3,
      }),
      Plot.text(cells, {
        x: "x",
        y: "y",
        text: (d) => (d.isGoal ? "G" : d.v.toFixed(2)),
        fill: "white",
        fontSize: 10,
      }),
    ],
  });
}

defineStepper({
  hostId: "ch4-prioritized-sweep-widget",
  controls: {
    gamma: { label: "γ (discount)", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["vi", "ps", "queue"],
  trajectory: ({ gamma }) => computeTrajectory(gamma),
  playIntervalMs: 250,
  render: (host, frame, idx, total, params, slots) => {
    const { step, viV, viCurrent, psV, psCurrent, pq } = frame;
    const vMax = Math.max(d3.max(viV), d3.max(psV), 0.01);
    slots.vi.replaceChildren(heatPlot(viV, viCurrent, "Vanilla VI", vMax));
    slots.ps.replaceChildren(
      heatPlot(psV, psCurrent, "Prioritized Sweeping", vMax),
    );

    // Priority-queue sidebar.
    const queueDiv = slots.queue;
    queueDiv.innerHTML = "";
    queueDiv.style.cssText =
      "font-family: monospace; font-size: 11px; line-height: 1.4; " +
      "padding: 6px 10px; border: 1px solid #444; border-radius: 4px; " +
      "min-width: 140px; background: rgba(255,255,255,0.03);";
    const header = document.createElement("div");
    header.textContent = `PQ (top 8 of ${pq.length})`;
    header.style.cssText = `font-weight: bold; color: ${palette.warning}; margin-bottom: 4px;`;
    queueDiv.appendChild(header);
    if (pq.length === 0) {
      const empty = document.createElement("div");
      empty.textContent = "(empty — PS converged)";
      empty.style.color = palette.muted;
      queueDiv.appendChild(empty);
    } else {
      for (const e of pq.slice(0, 8)) {
        const x = e.s % N;
        const y = Math.floor(e.s / N);
        const row = document.createElement("div");
        row.textContent = `s=(${x},${y})  p=${e.p.toFixed(4)}`;
        queueDiv.appendChild(row);
      }
    }

    // Count non-converged states for each side as a quick progress
    // metric. A state is "informed" if V > 1e-6.
    const informedVI = viV.filter((v) => v > 1e-6).length;
    const informedPS = psV.filter((v) => v > 1e-6).length;
    slots.readout.textContent =
      `backup ${step} / ${total - 1}  ·  ` +
      `informed states: VI=${informedVI}/${NUM_S - 1}, PS=${informedPS}/${NUM_S - 1}  ·  ` +
      `PQ size = ${pq.length}  ·  γ = ${params.gamma.toFixed(2)}`;
  },
});

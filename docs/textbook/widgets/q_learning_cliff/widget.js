// Widget 6.A — Q-learning on the cliff (Chapter 6).
//
// The classic 4x12 cliff-walking environment [S&B 2018 §6.5]: start at
// the bottom-left, goal at the bottom-right, and the entire bottom row
// in-between is a -100 "cliff" that respawns the agent at the start.
// Every step costs -1, so the agent has an incentive to find the
// shortest path — which runs along the cliff edge.
//
// We run ε-greedy Q-learning for a fixed number of episodes (capped
// per-episode so a wandering agent can't hang the page). The two slots:
//   * `main` — Q-value heatmap (max_a Q(s, a)) with greedy-policy arrows.
//   * `returns` — return-per-episode curve, smoothed with a running mean.
//
// Sliders cover ε, α, γ, episodes, and a seed so the reader can resample
// trajectories without changing the algorithm. Re-running is cheap: the
// 48-state x 4-action table converges in well under 500 episodes.
//
//     <div id="ch6-qlearning-cliff-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/q_learning_cliff/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

// 4 rows x 12 cols, row 0 at bottom. Start = (0, 0), goal = (0, 11),
// cliff = (0, 1..10).
const ROWS = 4;
const COLS = 12;
const START = { r: 0, c: 0 };
const GOAL = { r: 0, c: 11 };
const ACTIONS = [
  { dr:  1, dc:  0, name: "↑" },
  { dr:  0, dc:  1, name: "→" },
  { dr: -1, dc:  0, name: "↓" },
  { dr:  0, dc: -1, name: "←" },
];
const STEP_REWARD = -1;
const CLIFF_REWARD = -100;
const MAX_STEPS = 200; // hard cap so a bad ε-greedy walk can't hang the browser.

// A small splitmix32 RNG — deterministic, well-distributed, ~8 LOC.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 0x1_0000_0000;
  };
}

function isCliff(r, c) {
  return r === 0 && c > 0 && c < COLS - 1;
}
function isGoal(r, c) {
  return r === GOAL.r && c === GOAL.c;
}

// Q-learning: ε-greedy behaviour, greedy bootstrap. Returns the final
// Q-table (S x A) plus the return-per-episode trace.
function trainQLearning({ epsilon, alpha, gamma, episodes, seed }) {
  const rand = rng(seed);
  const Q = new Float64Array(ROWS * COLS * ACTIONS.length); // zero-init
  const idx = (r, c, a) => (r * COLS + c) * ACTIONS.length + a;
  const returns = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c;
    let totalR = 0;
    for (let step = 0; step < MAX_STEPS; step++) {
      // ε-greedy.
      let a;
      if (rand() < epsilon) {
        a = Math.floor(rand() * ACTIONS.length);
      } else {
        a = 0;
        let best = Q[idx(r, c, 0)];
        for (let k = 1; k < ACTIONS.length; k++) {
          const v = Q[idx(r, c, k)];
          if (v > best) { best = v; a = k; }
        }
      }
      // Step. Off-grid → no move.
      let nr = r + ACTIONS[a].dr;
      let nc = c + ACTIONS[a].dc;
      if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) { nr = r; nc = c; }
      let reward = STEP_REWARD;
      let done = false;
      if (isCliff(nr, nc)) {
        reward = CLIFF_REWARD;
        nr = START.r; nc = START.c; // respawn; episode continues
      } else if (isGoal(nr, nc)) {
        reward = STEP_REWARD; // goal still costs the last step
        done = true;
      }
      totalR += reward;
      // Q-learning update: bootstrap from max_a' Q(s', a').
      let maxNext = 0;
      if (!done) {
        maxNext = Q[idx(nr, nc, 0)];
        for (let k = 1; k < ACTIONS.length; k++) {
          const v = Q[idx(nr, nc, k)];
          if (v > maxNext) maxNext = v;
        }
      }
      const tdTarget = reward + gamma * maxNext;
      Q[idx(r, c, a)] += alpha * (tdTarget - Q[idx(r, c, a)]);
      r = nr; c = nc;
      if (done) break;
    }
    returns[ep] = totalR;
  }
  return { Q, returns, idx };
}

defineWidget({
  hostId: "ch6-qlearning-cliff-widget",
  controls: {
    epsilon:  { label: "ε (exploration)", min: 0,    max: 0.5,  step: 0.01, default: 0.1 },
    alpha:    { label: "α (step size)",   min: 0.05, max: 0.9,  step: 0.05, default: 0.5 },
    gamma:    { label: "γ (discount)",    min: 0.5,  max: 0.999, step: 0.01, default: 0.95 },
    episodes: { label: "episodes",        min: 50,   max: 1000, step: 50,   default: 500 },
    seed:     { label: "seed",            min: 1,    max: 50,   step: 1,    default: 7 },
  },
  slots: ["main", "returns"],
  render: (host, p, slots) => {
    const { Q, returns, idx } = trainQLearning({
      epsilon: p.epsilon, alpha: p.alpha, gamma: p.gamma,
      episodes: p.episodes | 0, seed: p.seed | 0,
    });

    // Per-cell max-Q (V) and argmax action.
    const cells = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        let best = Q[idx(r, c, 0)], aBest = 0;
        for (let k = 1; k < ACTIONS.length; k++) {
          const v = Q[idx(r, c, k)];
          if (v > best) { best = v; aBest = k; }
        }
        cells.push({
          r, c,
          v: best,
          arrow: ACTIONS[aBest].name,
          cliff: isCliff(r, c),
          start: r === START.r && c === START.c,
          goal: isGoal(r, c),
        });
      }
    }
    // Cliff cells render as a tagged tile, no Q value.
    const heatCells = cells.filter((d) => !d.cliff);
    const cliffCells = cells.filter((d) => d.cliff);
    const vMin = Math.min(...heatCells.map((d) => d.v));
    const vMax = Math.max(...heatCells.map((d) => d.v));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 22, marginRight: 22, marginTop: 12, marginBottom: 28,
      ...gridAxes({ nx: COLS, ny: ROWS }, { axis: null }, {
        x: { label: "column" },
        y: { label: null },
      }),
      color: { type: "linear", domain: [vMin, vMax], range: ["#1a1a2e", palette.primary] },
      marks: [
        Plot.cell(heatCells, { x: "c", y: "r", fill: "v", inset: 0.5 }),
        Plot.cell(cliffCells, { x: "c", y: "r", fill: palette.danger, fillOpacity: 0.5, inset: 0.5 }),
        Plot.text(cliffCells, { x: "c", y: "r", text: () => "✗", fill: "#fff", fontSize: 12 }),
        Plot.text(heatCells.filter((d) => !d.start && !d.goal), {
          x: "c", y: "r", text: "arrow", fill: "#fff", fontSize: 13,
        }),
        Plot.text(heatCells.filter((d) => d.start), {
          x: "c", y: "r", text: () => "S", fill: "#fff", fontSize: 13, fontWeight: "bold",
        }),
        Plot.text(heatCells.filter((d) => d.goal), {
          x: "c", y: "r", text: () => "G", fill: "#fff", fontSize: 13, fontWeight: "bold",
        }),
      ],
    }));

    // Smoothed return curve. Q-learning's return-per-episode is the
    // classic cliff signature: high variance from occasional cliff
    // falls, but mean settles near the optimal path's -13.
    const win = Math.max(5, Math.floor(returns.length / 25));
    const smoothed = returns.map((_, i) => {
      const lo = Math.max(0, i - win), hi = Math.min(returns.length, i + win + 1);
      let s = 0;
      for (let k = lo; k < hi; k++) s += returns[k];
      return { ep: i, ret: returns[i], smooth: s / (hi - lo) };
    });

    slots.returns.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "episode", grid: true },
      y: { label: "return / episode", domain: [-200, 0], grid: true },
      marks: [
        Plot.dot(smoothed, { x: "ep", y: "ret", fill: palette.muted, fillOpacity: 0.25, r: 1.2 }),
        Plot.line(smoothed, { x: "ep", y: "smooth", stroke: palette.primary, strokeWidth: 2 }),
        Plot.ruleY([-13], { stroke: palette.warning, strokeDasharray: "4 2" }),
        Plot.text([{ x: returns.length - 1, y: -13, label: "optimal -13" }], {
          x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
          fill: palette.warning, fontSize: 10,
        }),
      ],
    }));

    const lastK = Math.min(50, returns.length);
    const avgLast = returns.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    slots.readout.textContent =
      `avg return over last ${lastK} eps: ${fmt(avgLast)}  ·  optimal ≈ -13`;
  },
});

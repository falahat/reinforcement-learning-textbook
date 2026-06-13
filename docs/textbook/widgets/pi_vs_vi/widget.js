// Widget 4.B — Policy iteration vs value iteration (Chapter 4).
//
// Runs PI and VI side-by-side to convergence on the same 5x5
// deterministic gridworld with a top-right goal. Reports
// iterations-to-convergence and total Bellman-backup work for each, so
// the "PI: fewer but expensive iterations / VI: many but cheap" tradeoff
// from §4.4 is visible on a single γ slider.
//
// Per-iteration semantics (matching the chapter):
//   - VI iteration = one full sweep of T*V on every state.
//                    Work = |S| Bellman backups.
//   - PI iteration = (a) full policy evaluation: iterate T^π V to
//                        convergence; (b) one policy improvement.
//                    Work = (#eval-sweeps) * |S| backups + |S| backups
//                    for the improvement.
//
// Both algorithms converge to the same V* and π* by Banach
// (Theorem 4.5 in S&B).
//
// Pattern: chapter markdown contains
//
//     <div id="ch4-pi-vs-vi-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/pi_vs_vi/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const N = 5;
const NUM_S = N * N;
const ACTIONS = [
  { dx: 0, dy: -1, name: "↑" },
  { dx: 0, dy: +1, name: "↓" },
  { dx: -1, dy: 0, name: "←" },
  { dx: +1, dy: 0, name: "→" },
];
const NUM_A = ACTIONS.length;
const GOAL_X = N - 1;
const GOAL_Y = 0;
const GOAL_S = GOAL_Y * N + GOAL_X;
const STEP_REWARD = -1;
const CONV_THRESH = 1e-6;
// Caps must accommodate the slowest convergence on the slider range. For
// VI/PE the residual decays at rate γ per sweep, so the sweeps needed to
// hit CONV_THRESH scale as log(CONV_THRESH)/log(γ). At γ=0.99 (slider max)
// that's ~1374 sweeps. Caps below 2000 cause both algorithms to peg at the
// cap for high γ — making the displayed iters/backups constant in γ rather
// than tracing the contraction-rate curve the widget exists to show.
const MAX_OUTER = 5000;
const MAX_EVAL = 5000;

function nextState(s, a) {
  if (s === GOAL_S) return s;
  const x = s % N;
  const y = Math.floor(s / N);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return ny * N + nx;
}

function valueIteration(gamma) {
  let V = new Array(NUM_S).fill(0);
  let backups = 0;
  for (let k = 1; k <= MAX_OUTER; k++) {
    const Vnext = new Array(NUM_S);
    let delta = 0;
    for (let s = 0; s < NUM_S; s++) {
      if (s === GOAL_S) {
        Vnext[s] = 0;
        continue;
      }
      let best = -Infinity;
      for (let a = 0; a < NUM_A; a++) {
        const q = STEP_REWARD + gamma * V[nextState(s, a)];
        if (q > best) best = q;
      }
      Vnext[s] = best;
      const d = Math.abs(Vnext[s] - V[s]);
      if (d > delta) delta = d;
    }
    backups += NUM_S; // one backup per state in this sweep
    V = Vnext;
    if (delta < CONV_THRESH) return { V, iters: k, backups };
  }
  return { V, iters: MAX_OUTER, backups };
}

// Policy evaluation: iterate T^π V until convergence (or MAX_EVAL).
// Returns { V, sweeps }.
function policyEvaluation(pi, gamma) {
  let V = new Array(NUM_S).fill(0);
  for (let k = 1; k <= MAX_EVAL; k++) {
    const Vnext = new Array(NUM_S);
    let delta = 0;
    for (let s = 0; s < NUM_S; s++) {
      if (s === GOAL_S) {
        Vnext[s] = 0;
        continue;
      }
      const a = pi[s];
      Vnext[s] = STEP_REWARD + gamma * V[nextState(s, a)];
      const d = Math.abs(Vnext[s] - V[s]);
      if (d > delta) delta = d;
    }
    V = Vnext;
    if (delta < CONV_THRESH) return { V, sweeps: k };
  }
  return { V, sweeps: MAX_EVAL };
}

function policyImprovement(V, gamma) {
  const pi = new Array(NUM_S);
  for (let s = 0; s < NUM_S; s++) {
    if (s === GOAL_S) {
      pi[s] = 0;
      continue;
    }
    let bestA = 0;
    let best = -Infinity;
    for (let a = 0; a < NUM_A; a++) {
      const q = STEP_REWARD + gamma * V[nextState(s, a)];
      if (q > best) {
        best = q;
        bestA = a;
      }
    }
    pi[s] = bestA;
  }
  return pi;
}

function policyIteration(gamma) {
  let pi = new Array(NUM_S).fill(0); // start with "always up"
  let V = new Array(NUM_S).fill(0);
  let backups = 0;
  for (let k = 1; k <= MAX_OUTER; k++) {
    const ev = policyEvaluation(pi, gamma);
    V = ev.V;
    backups += ev.sweeps * NUM_S;
    const piNew = policyImprovement(V, gamma);
    backups += NUM_S; // improvement is one backup per state
    let stable = true;
    for (let s = 0; s < NUM_S; s++) {
      if (pi[s] !== piNew[s]) {
        stable = false;
        break;
      }
    }
    pi = piNew;
    if (stable) return { V, pi, iters: k, backups };
  }
  return { V, pi, iters: MAX_OUTER, backups };
}

function buildHeatmap(V) {
  const cells = [];
  for (let s = 0; s < NUM_S; s++) {
    const x = s % N;
    const y = Math.floor(s / N);
    cells.push({ x, y, v: V[s], isGoal: s === GOAL_S });
  }
  return cells;
}

defineWidget({
  hostId: "ch4-pi-vs-vi-widget",
  controls: {
    gamma: { label: "γ (discount)", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["vi", "pi", "summary"],
  render: (host, { gamma }, slots) => {
    const vi = valueIteration(gamma);
    const pi = policyIteration(gamma);
    const vMin = Math.min(d3.min(vi.V), d3.min(pi.V));

    const heatPlot = (V, title) =>
      Plot.plot({
        ...plotDefaults,
        height: 260,
        width: 300,
        marginLeft: 8,
        marginRight: 8,
        marginBottom: 24,
        marginTop: 28,
        title,
        x: { axis: null, domain: d3.range(N) },
        y: { axis: null, domain: d3.range(N) },
        color: {
          type: "linear",
          domain: [vMin, 0],
          range: ["#1b3a52", "#8bc34a"],
        },
        marks: [
          Plot.cell(buildHeatmap(V), {
            x: "x",
            y: "y",
            fill: (d) => (d.isGoal ? null : d.v),
            stroke: "#111",
            strokeOpacity: 0.4,
          }),
          Plot.cell(buildHeatmap(V).filter((c) => c.isGoal), {
            x: "x",
            y: "y",
            fill: palette.warning,
            stroke: palette.danger,
            strokeWidth: 2,
          }),
          Plot.text(buildHeatmap(V), {
            x: "x",
            y: "y",
            text: (d) => (d.isGoal ? "G" : d.v.toFixed(1)),
            fill: "white",
            fontSize: 10,
          }),
        ],
      });

    slots.vi.replaceChildren(heatPlot(vi.V, "Value Iteration"));
    slots.pi.replaceChildren(heatPlot(pi.V, "Policy Iteration"));

    // Side-by-side comparison bars (iterations + backups).
    const summary = [
      { algo: "VI", metric: "iterations", value: vi.iters },
      { algo: "PI", metric: "iterations", value: pi.iters },
      { algo: "VI", metric: "backups", value: vi.backups },
      { algo: "PI", metric: "backups", value: pi.backups },
    ];
    slots.summary.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 220,
        marginLeft: 60,
        x: { label: "count (log)", type: "log", grid: true },
        y: { label: null },
        fy: { label: null },
        color: {
          domain: ["VI", "PI"],
          range: [palette.primary, palette.secondary],
          legend: true,
        },
        marks: [
          Plot.barX(summary, {
            y: "algo",
            fy: "metric",
            x: "value",
            fill: "algo",
          }),
          Plot.text(summary, {
            y: "algo",
            fy: "metric",
            x: "value",
            text: (d) => d.value.toString(),
            textAnchor: "start",
            dx: 4,
            fill: "white",
            fontSize: 11,
          }),
        ],
      }),
    );

    slots.readout.textContent =
      `γ = ${gamma.toFixed(2)}  ·  ` +
      `VI: ${vi.iters} iterations / ${vi.backups} backups  ·  ` +
      `PI: ${pi.iters} iterations / ${pi.backups} backups`;
  },
});

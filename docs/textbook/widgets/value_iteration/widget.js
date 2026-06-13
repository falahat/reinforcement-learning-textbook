// Widget 4.A — Value iteration on a 5x5 gridworld (Chapter 4).
//
// Animates value iteration sweep-by-sweep on a deterministic 5x5
// gridworld with a single goal cell in the top-right corner.
// Every step costs -1; reaching the goal yields 0 and terminates.
// Actions: up / down / left / right. Bumping a wall keeps you in place
// (still pays the -1 step cost), which is the standard textbook setup.
//
// Stepper trajectory:
//   frame 0     : V_0 (all zeros)
//   frame 1..K  : V after k full Bellman-optimality sweeps
//                 (and the implicit greedy policy w.r.t. V_k)
//
// Slot layout:
//   main  : heatmap of V_k with arrows showing the greedy policy
//   delta : line plot of ||V_k - V_{k-1}||_∞ vs k (log y-axis)
//
// Slider γ controls discount; lower γ converges faster but myopically.
//
// Pattern: chapter markdown contains
//
//     <div id="ch4-value-iteration-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/value_iteration/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N = 5; // grid side
const NUM_S = N * N;
const ACTIONS = [
  { dx: 0, dy: -1, name: "↑" },
  { dx: 0, dy: +1, name: "↓" },
  { dx: -1, dy: 0, name: "←" },
  { dx: +1, dy: 0, name: "→" },
];
const GOAL_X = N - 1;
const GOAL_Y = 0;
const GOAL_S = GOAL_Y * N + GOAL_X;
const STEP_REWARD = -1;
const MAX_SWEEPS = 40;
// ε must be reachable within MAX_SWEEPS for at least part of the γ slider
// range — otherwise the ruler is decorative and the "delta crosses ε ⇒
// converged" claim never fires visually. Per-sweep contraction is rate γ,
// so the curve at sweep k is ~γ^k × (initial ||V|| ≈ 1). Reachability bounds:
//   γ=0.50 → reaches 1e-12 at k=40 (well past ε)
//   γ=0.70 → reaches 1e-6  at k=40 (just past ε)
//   γ=0.90 → reaches 1e-2  at k=40 (right at ε — pedagogically perfect)
//   γ=0.99 → reaches 0.67  at k=40 (never crosses — lesson about high-γ slowness)
// 1e-2 makes the ruler meaningful across the slider range; 1e-6 made it
// invisible to the chart at every γ.
const CONV_THRESH = 1e-2;

// Deterministic next-state under (s, a). Walls bounce; goal is absorbing.
function nextState(s, a) {
  if (s === GOAL_S) return s;
  const x = s % N;
  const y = Math.floor(s / N);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return ny * N + nx;
}

// One sweep of the Bellman optimality operator T*V.
// Returns { Vnext, delta } where delta = ||Vnext - V||_∞.
function bellmanSweep(V, gamma) {
  const Vnext = new Array(NUM_S);
  let delta = 0;
  for (let s = 0; s < NUM_S; s++) {
    if (s === GOAL_S) {
      Vnext[s] = 0;
      continue;
    }
    let best = -Infinity;
    for (let a = 0; a < ACTIONS.length; a++) {
      const sp = nextState(s, a);
      const q = STEP_REWARD + gamma * V[sp];
      if (q > best) best = q;
    }
    Vnext[s] = best;
    const d = Math.abs(Vnext[s] - V[s]);
    if (d > delta) delta = d;
  }
  return { Vnext, delta };
}

// Greedy action per state given V.
function greedyPolicy(V, gamma) {
  const pi = new Array(NUM_S);
  for (let s = 0; s < NUM_S; s++) {
    if (s === GOAL_S) {
      pi[s] = -1;
      continue;
    }
    let bestA = 0;
    let best = -Infinity;
    for (let a = 0; a < ACTIONS.length; a++) {
      const sp = nextState(s, a);
      const q = STEP_REWARD + gamma * V[sp];
      if (q > best) {
        best = q;
        bestA = a;
      }
    }
    pi[s] = bestA;
  }
  return pi;
}

defineStepper({
  hostId: "ch4-value-iteration-widget",
  controls: {
    gamma: { label: "γ (discount)", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["main", "delta"],
  trajectory: ({ gamma }) => {
    const frames = [];
    let V = new Array(NUM_S).fill(0);
    const deltaHistory = [];
    frames.push({ k: 0, V: V.slice(), delta: NaN, deltaHistory: [] });
    for (let k = 1; k <= MAX_SWEEPS; k++) {
      const { Vnext, delta } = bellmanSweep(V, gamma);
      V = Vnext;
      deltaHistory.push({ k, delta: Math.max(delta, 1e-12) });
      frames.push({
        k,
        V: V.slice(),
        delta,
        deltaHistory: deltaHistory.slice(),
      });
      if (delta < CONV_THRESH) break;
    }
    return frames;
  },
  playIntervalMs: 500,
  render: (host, frame, idx, total, params, slots) => {
    const { gamma } = params;
    const { V, k, delta, deltaHistory } = frame;
    const pi = greedyPolicy(V, gamma);

    // Build cells for the heatmap.
    const cells = [];
    for (let s = 0; s < NUM_S; s++) {
      const x = s % N;
      const y = Math.floor(s / N);
      cells.push({ x, y, v: V[s], isGoal: s === GOAL_S });
    }
    // Arrows for non-goal cells.
    const arrows = [];
    for (let s = 0; s < NUM_S; s++) {
      if (s === GOAL_S) continue;
      const x = s % N;
      const y = Math.floor(s / N);
      arrows.push({ x, y, sym: ACTIONS[pi[s]].name });
    }

    const vMin = d3.min(V);
    const vMax = 0;
    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 340,
        width: 360,
        marginLeft: 28,
        marginBottom: 28,
        x: { axis: null, domain: d3.range(N) },
        y: { axis: null, domain: d3.range(N) },
        color: {
          type: "linear",
          domain: [vMin, vMax],
          range: ["#1b3a52", "#8bc34a"],
          legend: true,
          label: "V(s)",
        },
        marks: [
          Plot.cell(cells, {
            x: "x",
            y: "y",
            fill: (d) => (d.isGoal ? null : d.v),
            stroke: "#111",
            strokeOpacity: 0.4,
          }),
          // Goal cell highlight
          Plot.cell(cells.filter((c) => c.isGoal), {
            x: "x",
            y: "y",
            fill: palette.warning,
            stroke: palette.danger,
            strokeWidth: 2,
          }),
          // Value labels
          Plot.text(cells, {
            x: "x",
            y: "y",
            text: (d) => (d.isGoal ? "G" : d.v.toFixed(2)),
            fill: "white",
            fontSize: 10,
            dy: -6,
          }),
          // Greedy-policy arrows
          Plot.text(arrows, {
            x: "x",
            y: "y",
            text: "sym",
            fill: "white",
            fontSize: 14,
            dy: 8,
          }),
        ],
      }),
    );

    // Delta plot. Hide on frame 0 (no delta yet).
    if (deltaHistory.length === 0) {
      slots.delta.replaceChildren();
    } else {
      slots.delta.replaceChildren(
        Plot.plot({
          ...plotDefaults,
          height: 180,
          x: { label: "sweep k", domain: [1, MAX_SWEEPS], grid: true },
          y: {
            type: "log",
            label: "||V_k − V_{k−1}||∞",
            grid: true,
            domain: [1e-4, Math.max(1, deltaHistory[0].delta * 1.2)],
          },
          marks: [
            Plot.ruleY([CONV_THRESH], {
              stroke: palette.danger,
              ...dashed,
            }),
            Plot.text(
              [{ x: MAX_SWEEPS, y: CONV_THRESH, label: "convergence ε" }],
              {
                x: "x",
                y: "y",
                text: "label",
                textAnchor: "end",
                dy: -4,
                fill: palette.danger,
                ...annotation,
              },
            ),
            Plot.line(deltaHistory, {
              x: "k",
              y: "delta",
              stroke: palette.primary,
              strokeWidth: 2,
            }),
            Plot.dot(deltaHistory, {
              x: "k",
              y: "delta",
              fill: palette.primary,
              r: 3,
            }),
            // Mark current sweep
            Plot.dot([{ k, delta: Math.max(delta, 1e-12) }].filter((d) => !isNaN(d.delta)), {
              x: "k",
              y: "delta",
              fill: palette.danger,
              r: 5,
            }),
          ],
        }),
      );
    }

    const deltaStr = isNaN(delta) ? "—" : delta.toExponential(2);
    slots.readout.textContent =
      `sweep k = ${k} / ${total - 1}  ·  Δ = ${deltaStr}  ·  γ = ${gamma.toFixed(2)}`;
  },
});

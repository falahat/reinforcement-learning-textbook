// Widget 6.B — SARSA vs Q-learning on the cliff (Chapter 6).
//
// The classic side-by-side from S&B §6.5. Same 4x12 cliff-walking env,
// same ε-greedy behaviour, same number of episodes — only the bootstrap
// target differs:
//
//   * SARSA bootstraps from the *sampled* next action under the behaviour
//     policy, so it learns the value of the ε-greedy policy and picks
//     the *safe* path that hugs the top wall.
//   * Q-learning bootstraps from max_a Q(s', a), so it learns the value
//     of the *greedy* policy and picks the *risky-optimal* path running
//     along the cliff edge — even though the ε-greedy behaviour
//     occasionally falls in.
//
// We render greedy policies side-by-side (arrows + cliff tags) plus a
// shared return-per-episode plot with both algorithms overlaid. SARSA's
// curve tracks roughly -17 (safe path); Q-learning's mean tracks -13
// (optimal) but with deeper occasional dips from cliff falls.
//
//     <div id="ch6-sarsa-vs-q-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/sarsa_vs_q/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

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
const MAX_STEPS = 200;

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
const isCliff = (r, c) => r === 0 && c > 0 && c < COLS - 1;
const isGoal = (r, c) => r === GOAL.r && c === GOAL.c;
const qIdx = (r, c, a) => (r * COLS + c) * ACTIONS.length + a;

function step(r, c, a) {
  let nr = r + ACTIONS[a].dr;
  let nc = c + ACTIONS[a].dc;
  if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) { nr = r; nc = c; }
  let reward = STEP_REWARD;
  let done = false;
  if (isCliff(nr, nc)) { reward = CLIFF_REWARD; nr = START.r; nc = START.c; }
  else if (isGoal(nr, nc)) { done = true; }
  return { nr, nc, reward, done };
}

function epsilonGreedy(Q, r, c, epsilon, rand) {
  if (rand() < epsilon) return Math.floor(rand() * ACTIONS.length);
  let a = 0, best = Q[qIdx(r, c, 0)];
  for (let k = 1; k < ACTIONS.length; k++) {
    const v = Q[qIdx(r, c, k)];
    if (v > best) { best = v; a = k; }
  }
  return a;
}

// Two algorithms in one driver: same env, same exploration. The only
// branch is in the TD target — bootstrap from Q[s', a'] (SARSA) or
// max_a Q[s', a] (Q-learning).
function trainCliff({ algo, epsilon, alpha, gamma, episodes, seed }) {
  const rand = rng(seed);
  const Q = new Float64Array(ROWS * COLS * ACTIONS.length);
  const returns = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c;
    let a = epsilonGreedy(Q, r, c, epsilon, rand);
    let totalR = 0;
    for (let stepIdx = 0; stepIdx < MAX_STEPS; stepIdx++) {
      const { nr, nc, reward, done } = step(r, c, a);
      totalR += reward;
      let bootstrap = 0;
      let aNext = 0;
      if (!done) {
        if (algo === "sarsa") {
          aNext = epsilonGreedy(Q, nr, nc, epsilon, rand);
          bootstrap = Q[qIdx(nr, nc, aNext)];
        } else { // q-learning
          bootstrap = Q[qIdx(nr, nc, 0)];
          for (let k = 1; k < ACTIONS.length; k++) {
            const v = Q[qIdx(nr, nc, k)];
            if (v > bootstrap) bootstrap = v;
          }
        }
      }
      const tdErr = reward + gamma * bootstrap - Q[qIdx(r, c, a)];
      Q[qIdx(r, c, a)] += alpha * tdErr;
      r = nr; c = nc;
      // SARSA carries the already-sampled aNext into the next iteration;
      // Q-learning resamples since it doesn't bind to a'.
      if (done) break;
      a = (algo === "sarsa") ? aNext : epsilonGreedy(Q, r, c, epsilon, rand);
    }
    returns[ep] = totalR;
  }
  return { Q, returns };
}

function policyCells(Q) {
  const cells = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      let best = Q[qIdx(r, c, 0)], aBest = 0;
      for (let k = 1; k < ACTIONS.length; k++) {
        const v = Q[qIdx(r, c, k)];
        if (v > best) { best = v; aBest = k; }
      }
      cells.push({
        r, c, arrow: ACTIONS[aBest].name,
        cliff: isCliff(r, c),
        start: r === START.r && c === START.c,
        goal: isGoal(r, c),
      });
    }
  }
  return cells;
}

function policyPlot(cells, color, title) {
  const cliffCells = cells.filter((d) => d.cliff);
  const open = cells.filter((d) => !d.cliff);
  return Plot.plot({
    ...plotDefaults,
    width: 320, height: 130,
    marginLeft: 8, marginRight: 8, marginTop: 24, marginBottom: 8,
    title,
    ...gridAxes({ nx: COLS, ny: ROWS }, { axis: null }),
    marks: [
      Plot.cell(open, { x: "c", y: "r", fill: color, fillOpacity: 0.15, inset: 0.5 }),
      Plot.cell(cliffCells, { x: "c", y: "r", fill: palette.danger, fillOpacity: 0.45, inset: 0.5 }),
      Plot.text(cliffCells, { x: "c", y: "r", text: () => "✗", fill: "#fff", fontSize: 11 }),
      Plot.text(open.filter((d) => !d.start && !d.goal), { x: "c", y: "r", text: "arrow", fill: "#ddd", fontSize: 12 }),
      Plot.text(open.filter((d) => d.start), { x: "c", y: "r", text: () => "S", fill: "#fff", fontSize: 12, fontWeight: "bold" }),
      Plot.text(open.filter((d) => d.goal), { x: "c", y: "r", text: () => "G", fill: "#fff", fontSize: 12, fontWeight: "bold" }),
    ],
  });
}

defineWidget({
  hostId: "ch6-sarsa-vs-q-widget",
  controls: {
    epsilon:  { label: "ε",         min: 0.01, max: 0.4,  step: 0.01, default: 0.1 },
    alpha:    { label: "α",         min: 0.05, max: 0.9,  step: 0.05, default: 0.5 },
    gamma:    { label: "γ",         min: 0.5,  max: 0.999, step: 0.01, default: 0.99 },
    episodes: { label: "episodes",  min: 100,  max: 1000, step: 50,   default: 500 },
    seed:     { label: "seed",      min: 1,    max: 50,   step: 1,    default: 7 },
  },
  slots: ["policies", "returns"],
  render: (host, p, slots) => {
    const eps = p.episodes | 0;
    const seed = p.seed | 0;
    const sarsa = trainCliff({ algo: "sarsa", epsilon: p.epsilon, alpha: p.alpha, gamma: p.gamma, episodes: eps, seed });
    const ql    = trainCliff({ algo: "qlearn", epsilon: p.epsilon, alpha: p.alpha, gamma: p.gamma, episodes: eps, seed: seed + 1000 });

    // Side-by-side policy panels.
    const policiesDiv = document.createElement("div");
    policiesDiv.style.display = "flex";
    policiesDiv.style.flexWrap = "wrap";
    policiesDiv.style.gap = "12px";
    const sarsaPanel = policyPlot(policyCells(sarsa.Q), palette.secondary, "SARSA — safe path (top)");
    const qlPanel    = policyPlot(policyCells(ql.Q),    palette.primary,   "Q-learning — risky-optimal (cliff edge)");
    policiesDiv.append(sarsaPanel, qlPanel);
    slots.policies.replaceChildren(policiesDiv);

    // Smoothed return curves for both, overlaid. Window size scales with
    // episode count so the curves stay readable across the 100-1000
    // slider range.
    const win = Math.max(5, Math.floor(eps / 25));
    const smooth = (arr) => arr.map((_, i) => {
      const lo = Math.max(0, i - win), hi = Math.min(arr.length, i + win + 1);
      let s = 0;
      for (let k = lo; k < hi; k++) s += arr[k];
      return s / (hi - lo);
    });
    const rows = [];
    const sSmooth = smooth(sarsa.returns);
    const qSmooth = smooth(ql.returns);
    for (let i = 0; i < eps; i++) {
      rows.push({ ep: i, ret: sSmooth[i], algo: "SARSA" });
      rows.push({ ep: i, ret: qSmooth[i], algo: "Q-learning" });
    }
    slots.returns.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "episode", grid: true },
      y: { label: "return / episode (smoothed)", domain: [-100, 0], grid: true },
      color: { domain: ["SARSA", "Q-learning"], range: [palette.secondary, palette.primary], legend: true },
      marks: [
        Plot.ruleY([-13], { stroke: palette.warning, strokeDasharray: "4 2" }),
        Plot.text([{ x: eps - 1, y: -13, label: "optimal -13" }], {
          x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
          fill: palette.warning, fontSize: 10,
        }),
        Plot.line(rows, { x: "ep", y: "ret", stroke: "algo", strokeWidth: 2 }),
      ],
    }));

    const lastK = Math.min(50, eps);
    const sAvg = sarsa.returns.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    const qAvg = ql.returns.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    slots.readout.textContent =
      `last ${lastK} eps · SARSA: ${fmt(sAvg)}  ·  Q-learning: ${fmt(qAvg)}`;
  },
});

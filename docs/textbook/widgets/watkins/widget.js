// Widget 6.G — Watkins' counter-example: insufficient (s, a) coverage
// (Chapter 6, §6.5).
//
// Watkins-Dayan 1992 proves Q-learning converges to Q* a.s. UNDER
// infinite visitation of every (s, a) pair. The "infinite visitation"
// condition is load-bearing — drop it and Q-learning can be stuck
// forever on a sub-optimal policy.
//
// A minimal 4-state chain illustrates this. States s0..s3; s0 is the
// start, s3 is the goal. From s0 the agent can:
//   - "trap" action → s1 (reward +1, terminal), or
//   - "explore" action → s2 (reward 0, deterministic).
// From s2 a single action reaches s3 (reward +10, terminal).
//
// Greedy-from-the-start Q-learning (ε = 0) initialises Q(s0, trap) >
// Q(s0, explore) — because the first random play happens to take the
// trap, immediately gets reward +1, and the explore action is never
// tried again. ε-greedy (ε = 0.2) eventually breaks symmetry: explore
// gets sampled, sees the +10 payoff, and Q(s0, explore) > Q(s0, trap)
// from then on.
//
// Two side-by-side return-per-episode curves: left = greedy starves
// (mean ≈ +1), right = ε-greedy finds the +10 path. Same MDP, same α,
// same γ, same seed — only the exploration rate differs.
//
//     <div id="ch6-watkins-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/watkins/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

// 4 states, 2 actions per state. Actions: 0 = "trap"/"left", 1 = "explore"/"right".
const N_STATES = 4;
const N_ACTIONS = 2;
const START = 0;

// Transition table: nextState[s][a] and reward[s][a]. -1 means terminal
// after this transition (episode ends).
const NEXT = [
  [1, 2],  // s0: trap → s1, explore → s2
  [-1, -1], // s1 terminal
  [3, 3],   // s2: either action → s3 (single useful action; we duplicate)
  [-1, -1], // s3 terminal
];
const REWARD = [
  [1.0, 0.0],
  [0.0, 0.0],
  [10.0, 10.0],
  [0.0, 0.0],
];
// Which states are terminals (no actions taken there).
const TERMINAL = [false, true, false, true];

function chooseAction(Q, s, eps, rand) {
  if (rand() < eps) return Math.floor(rand() * N_ACTIONS);
  // Greedy with deterministic tiebreak (action 0 wins ties → biases toward "trap").
  let best = 0;
  let bestQ = Q[s * N_ACTIONS + 0];
  for (let a = 1; a < N_ACTIONS; a++) {
    const v = Q[s * N_ACTIONS + a];
    if (v > bestQ) { bestQ = v; best = a; }
  }
  return best;
}

function maxQ(Q, s) {
  let m = Q[s * N_ACTIONS + 0];
  for (let a = 1; a < N_ACTIONS; a++) {
    const v = Q[s * N_ACTIONS + a];
    if (v > m) m = v;
  }
  return m;
}

function runQLearning({ epsilon, alpha, gamma, episodes, seed }) {
  const rand = mulberry32(seed | 0);
  const Q = new Float64Array(N_STATES * N_ACTIONS); // zero init
  const returns = new Array(episodes);
  // Seed: take the trap once at the start to pin the greedy bias.
  // We do this via the same RNG stream — both runs see the same first
  // sample, so the only difference is whether ε > 0 lets us escape.
  for (let ep = 0; ep < episodes; ep++) {
    let s = START;
    let total = 0;
    let steps = 0;
    while (!TERMINAL[s] && steps < 50) {
      const a = chooseAction(Q, s, epsilon, rand);
      const ns = NEXT[s][a];
      const r = REWARD[s][a];
      total += r;
      const target = r + (ns < 0 || TERMINAL[ns] ? 0 : gamma * maxQ(Q, ns));
      Q[s * N_ACTIONS + a] += alpha * (target - Q[s * N_ACTIONS + a]);
      if (ns < 0 || TERMINAL[ns]) break;
      s = ns;
      steps++;
    }
    returns[ep] = total;
  }
  return { Q, returns };
}

defineWidget({
  hostId: "ch6-watkins-widget",
  controls: {
    alpha:    { label: "α",        min: 0.05, max: 0.9,  step: 0.05, default: 0.5 },
    gamma:    { label: "γ",        min: 0.5,  max: 0.99, step: 0.01, default: 0.95 },
    episodes: { label: "episodes", min: 50,   max: 1000, step: 50,   default: 300 },
    seed:     { label: "seed",     min: 1,    max: 50,   step: 1,    default: 3 },
  },
  slots: ["main"],
  render: (host, { alpha, gamma, episodes, seed }, slots) => {
    const eps = episodes | 0;
    const greedy = runQLearning({ epsilon: 0.0, alpha, gamma, episodes: eps, seed });
    const explorer = runQLearning({ epsilon: 0.2, alpha, gamma, episodes: eps, seed });

    const data = [];
    for (let i = 0; i < eps; i++) {
      data.push({ ep: i, ret: greedy.returns[i], run: "ε = 0 (greedy)" });
      data.push({ ep: i, ret: explorer.returns[i], run: "ε = 0.2" });
    }

    // Smoothed lines so high-variance ε-greedy curve is readable.
    const win = Math.max(5, Math.floor(eps / 25));
    function smooth(arr) {
      return arr.map((_, i) => {
        const lo = Math.max(0, i - win);
        const hi = Math.min(arr.length, i + win + 1);
        let s = 0;
        for (let k = lo; k < hi; k++) s += arr[k];
        return s / (hi - lo);
      });
    }
    const gs = smooth(greedy.returns);
    const xs = smooth(explorer.returns);
    const lines = [];
    for (let i = 0; i < eps; i++) {
      lines.push({ ep: i, ret: gs[i], run: "ε = 0 (greedy)" });
      lines.push({ ep: i, ret: xs[i], run: "ε = 0.2" });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true },
      y: { label: "return / episode", domain: [-0.5, 11], grid: true },
      color: {
        legend: true,
        domain: ["ε = 0 (greedy)", "ε = 0.2"],
        range: [palette.danger, palette.primary],
      },
      marks: [
        Plot.ruleY([1, 10], { stroke: palette.muted, ...dashed }),
        Plot.text([
          { x: 0, y: 1, text: "trap return = 1" },
          { x: 0, y: 10, text: "optimal return = 10" },
        ], { x: "x", y: "y", text: "text", textAnchor: "start", dy: -4, fontSize: 10, fill: palette.muted }),
        Plot.dot(data, { x: "ep", y: "ret", fill: "run", fillOpacity: 0.18, r: 1.5 }),
        Plot.line(lines, { x: "ep", y: "ret", stroke: "run", strokeWidth: 2 }),
      ],
    }));

    const lastK = Math.min(50, eps);
    const avg = (arr) => arr.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    slots.readout.textContent =
      `last ${lastK} eps · greedy ${fmt(avg(greedy.returns))}  ·  ε=0.2 ${fmt(avg(explorer.returns))}  ·  optimal = 10`;
  },
});

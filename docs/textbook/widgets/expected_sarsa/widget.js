// Widget 6.D — Expected SARSA explorer (Chapter 6).
//
// Three-way race on the same 4x12 cliff env from 6.B. Adds Expected
// SARSA to the SARSA/Q-learning pair: bootstrap from the *expectation*
// of Q(s', ·) under the behaviour ε-greedy policy, not from a single
// sample (SARSA) or the max (Q-learning).
//
// Expected SARSA's TD target is
//
//   r + γ Σ_a π(a | s') Q(s', a)
//
// With ε-greedy π over A actions and greedy a* = argmax Q(s', ·):
//   π(a*) = 1 − ε + ε/A,   π(other) = ε/A.
//
// As ε → 0 the expectation collapses to (1)·Q(s', a*) = max Q(s', ·)
// (Q-learning). As ε → 1 the expectation is a uniform average over
// actions, which is what SARSA's *expected* update tends to — Expected
// SARSA is the variance-reduced cousin of SARSA.
//
//     <div id="ch6-expected-sarsa-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/expected_sarsa/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const ROWS = 4;
const COLS = 12;
const START = { r: 0, c: 0 };
const GOAL = { r: 0, c: 11 };
const ACTIONS = [
  { dr:  1, dc:  0 },
  { dr:  0, dc:  1 },
  { dr: -1, dc:  0 },
  { dr:  0, dc: -1 },
];
const A = ACTIONS.length;
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
const qIdx = (r, c, a) => (r * COLS + c) * A + a;

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
function epsGreedy(Q, r, c, eps, rand) {
  if (rand() < eps) return Math.floor(rand() * A);
  let a = 0, best = Q[qIdx(r, c, 0)];
  for (let k = 1; k < A; k++) {
    const v = Q[qIdx(r, c, k)];
    if (v > best) { best = v; a = k; }
  }
  return a;
}
function greedyAction(Q, r, c) {
  let a = 0, best = Q[qIdx(r, c, 0)];
  for (let k = 1; k < A; k++) {
    const v = Q[qIdx(r, c, k)];
    if (v > best) { best = v; a = k; }
  }
  return a;
}

function train({ algo, epsilon, alpha, gamma, episodes, seed }) {
  const rand = rng(seed);
  const Q = new Float64Array(ROWS * COLS * A);
  const returns = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c;
    let a = epsGreedy(Q, r, c, epsilon, rand);
    let totalR = 0;
    for (let s = 0; s < MAX_STEPS; s++) {
      const { nr, nc, reward, done } = step(r, c, a);
      totalR += reward;
      let bootstrap = 0;
      let aNext = 0;
      if (!done) {
        if (algo === "sarsa") {
          aNext = epsGreedy(Q, nr, nc, epsilon, rand);
          bootstrap = Q[qIdx(nr, nc, aNext)];
        } else if (algo === "qlearn") {
          bootstrap = Q[qIdx(nr, nc, 0)];
          for (let k = 1; k < A; k++) {
            const v = Q[qIdx(nr, nc, k)];
            if (v > bootstrap) bootstrap = v;
          }
        } else { // expected SARSA
          const aStar = greedyAction(Q, nr, nc);
          const piGreedy = 1 - epsilon + epsilon / A;
          const piOther = epsilon / A;
          bootstrap = 0;
          for (let k = 0; k < A; k++) {
            const pi = (k === aStar) ? piGreedy : piOther;
            bootstrap += pi * Q[qIdx(nr, nc, k)];
          }
        }
      }
      Q[qIdx(r, c, a)] += alpha * (reward + gamma * bootstrap - Q[qIdx(r, c, a)]);
      r = nr; c = nc;
      if (done) break;
      // Expected SARSA + Q-learning resample behaviour action at next
      // step; SARSA carries aNext forward.
      a = (algo === "sarsa") ? aNext : epsGreedy(Q, r, c, epsilon, rand);
    }
    returns[ep] = totalR;
  }
  return returns;
}

defineWidget({
  hostId: "ch6-expected-sarsa-widget",
  controls: {
    epsilon:  { label: "ε",         min: 0.01, max: 0.4,  step: 0.01, default: 0.1 },
    alpha:    { label: "α",         min: 0.05, max: 0.9,  step: 0.05, default: 0.5 },
    gamma:    { label: "γ",         min: 0.5,  max: 0.999, step: 0.01, default: 0.99 },
    episodes: { label: "episodes",  min: 100,  max: 800, step: 50,   default: 400 },
    seed:     { label: "seed",      min: 1,    max: 50,   step: 1,    default: 7 },
  },
  slots: ["main"],
  render: (host, p, slots) => {
    const eps = p.episodes | 0;
    const seed = p.seed | 0;
    const sarsa = train({ algo: "sarsa",    ...p, episodes: eps, seed: seed });
    const ql    = train({ algo: "qlearn",   ...p, episodes: eps, seed: seed + 1000 });
    const esa   = train({ algo: "expected", ...p, episodes: eps, seed: seed + 2000 });

    const win = Math.max(5, Math.floor(eps / 25));
    const smooth = (arr) => arr.map((_, i) => {
      const lo = Math.max(0, i - win), hi = Math.min(arr.length, i + win + 1);
      let s = 0;
      for (let k = lo; k < hi; k++) s += arr[k];
      return s / (hi - lo);
    });
    const sS = smooth(sarsa), qS = smooth(ql), eS = smooth(esa);

    const rows = [];
    for (let i = 0; i < eps; i++) {
      rows.push({ ep: i, ret: sS[i], algo: "SARSA" });
      rows.push({ ep: i, ret: qS[i], algo: "Q-learning" });
      rows.push({ ep: i, ret: eS[i], algo: "Expected SARSA" });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true },
      y: { label: "return / episode (smoothed)", domain: [-100, 0], grid: true },
      color: {
        domain: ["SARSA", "Q-learning", "Expected SARSA"],
        range: [palette.secondary, palette.primary, palette.accent],
        legend: true,
      },
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
    const avg = (arr) => arr.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    // The pedagogical hook: at the extremes Expected SARSA collapses to
    // a known algorithm. Spell that out in the readout so the slider
    // tells a story.
    const hint = p.epsilon < 0.03
      ? "ε ≈ 0 → Expected SARSA ≈ Q-learning"
      : p.epsilon > 0.3
        ? "large ε → Expected SARSA ≈ SARSA (lower-variance)"
        : "Expected SARSA interpolates between SARSA and Q-learning";
    slots.readout.textContent =
      `last ${lastK} eps · SARSA: ${fmt(avg(sarsa))}  ·  ` +
      `Q-learn: ${fmt(avg(ql))}  ·  Exp. SARSA: ${fmt(avg(esa))}  —  ${hint}`;
  },
});

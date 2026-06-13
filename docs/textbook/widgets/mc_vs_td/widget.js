// Widget 5.A — MC vs TD trajectory on Cliff Walking (Chapter 5).
//
// The headline contrast between Monte Carlo and TD methods is *what
// kind of noise their estimates carry*. MC waits until episode end and
// updates V from the full return G — a sum of every reward in the
// episode, so the per-episode update jitters wildly with whatever the
// agent happened to fall off (the cliff) or skirted around. TD(0)
// bootstraps after every step: each update is one reward plus a
// discounted V estimate, both small numbers, so the per-update jitter
// is small.
//
// Set-up: 4×12 cliff-walking gridworld [S&B Example 6.6]. Start (3,0),
// goal (3,11), cliff cells (3,1..10). Stepping onto the cliff returns
// to start with reward −100; every other step −1. Both agents follow
// an ε-greedy policy over a shared Q-table seeded with zeros (this is
// a *learning* curve, not a fixed-π evaluation, which matches what
// "MC vs TD" looks like in practice).
//
// The plot is return-per-episode for both runs. Hover any single seed
// and you'll see MC's spiky line and TD's smooth one; we render a
// mean curve and a 10-90 percentile band over `nSeeds` independent
// runs so the variance contrast is on screen, not just felt.
//
// Mount: `<div id="ch5-mc-vs-td-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/mc_vs_td/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const ROWS = 4;
const COLS = 12;
const START = { r: 3, c: 0 };
const GOAL  = { r: 3, c: 11 };
const ACTIONS = [
  { dr: -1, dc:  0 }, // up
  { dr:  0, dc:  1 }, // right
  { dr:  1, dc:  0 }, // down
  { dr:  0, dc: -1 }, // left
];
const MAX_STEPS = 200;

// Seeded LCG so the curves are stable across re-renders.
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function step(r, c, a) {
  const nr = Math.max(0, Math.min(ROWS - 1, r + ACTIONS[a].dr));
  const nc = Math.max(0, Math.min(COLS - 1, c + ACTIONS[a].dc));
  // Cliff: bottom row, columns 1..10. Step there → reward −100, back to start.
  if (nr === 3 && nc >= 1 && nc <= 10) {
    return { r: START.r, c: START.c, reward: -100, done: false };
  }
  const done = nr === GOAL.r && nc === GOAL.c;
  return { r: nr, c: nc, reward: -1, done };
}

function epsGreedy(Q, r, c, eps, rand) {
  if (rand() < eps) return Math.floor(rand() * 4);
  const qs = Q[r][c];
  let best = 0;
  for (let a = 1; a < 4; a++) if (qs[a] > qs[best]) best = a;
  return best;
}

function emptyQ() {
  const Q = new Array(ROWS);
  for (let r = 0; r < ROWS; r++) {
    Q[r] = new Array(COLS);
    for (let c = 0; c < COLS; c++) Q[r][c] = [0, 0, 0, 0];
  }
  return Q;
}

// One MC episode: roll out, then update Q from each first-visit (s,a)
// using its return G. Returns the undiscounted total reward (the
// y-axis return-per-episode we plot).
function mcEpisode(Q, gamma, alpha, eps, rand) {
  const traj = [];
  let r = START.r, c = START.c, total = 0;
  for (let t = 0; t < MAX_STEPS; t++) {
    const a = epsGreedy(Q, r, c, eps, rand);
    const out = step(r, c, a);
    traj.push({ r, c, a, reward: out.reward });
    total += out.reward;
    r = out.r; c = out.c;
    if (out.done) break;
  }
  // First-visit MC: walk backwards, accumulate G, update Q on first occurrence.
  let G = 0;
  const seen = new Set();
  for (let t = traj.length - 1; t >= 0; t--) {
    const { r: tr, c: tc, a, reward } = traj[t];
    G = reward + gamma * G;
    const key = `${tr},${tc},${a}`;
    if (!seen.has(key)) {
      seen.add(key);
      Q[tr][tc][a] += alpha * (G - Q[tr][tc][a]);
    }
  }
  return total;
}

// One TD(0) episode (this is SARSA on Q). Per-step bootstrap update;
// no waiting for the full return.
function tdEpisode(Q, gamma, alpha, eps, rand) {
  let r = START.r, c = START.c;
  let a = epsGreedy(Q, r, c, eps, rand);
  let total = 0;
  for (let t = 0; t < MAX_STEPS; t++) {
    const out = step(r, c, a);
    total += out.reward;
    const aNext = out.done ? 0 : epsGreedy(Q, out.r, out.c, eps, rand);
    const target = out.done ? out.reward : out.reward + gamma * Q[out.r][out.c][aNext];
    Q[r][c][a] += alpha * (target - Q[r][c][a]);
    r = out.r; c = out.c; a = aNext;
    if (out.done) break;
  }
  return total;
}

// Run `nSeeds` independent agents of `runner`, return per-episode mean
// and 10/90 percentile band across seeds.
function runMany(runner, episodes, nSeeds, gamma, alpha, eps) {
  const all = []; // [seed][episode] → return
  for (let s = 0; s < nSeeds; s++) {
    const rand = rng(s + 1);
    const Q = emptyQ();
    const returns = new Array(episodes);
    for (let ep = 0; ep < episodes; ep++) returns[ep] = runner(Q, gamma, alpha, eps, rand);
    all.push(returns);
  }
  const rows = [];
  for (let ep = 0; ep < episodes; ep++) {
    const col = all.map((seed) => seed[ep]).sort((a, b) => a - b);
    const mean = col.reduce((s, v) => s + v, 0) / nSeeds;
    const lo = col[Math.floor(0.1 * nSeeds)];
    const hi = col[Math.min(nSeeds - 1, Math.floor(0.9 * nSeeds))];
    rows.push({ ep, mean, lo, hi });
  }
  return rows;
}

defineWidget({
  hostId: "ch5-mc-vs-td-widget",
  controls: {
    episodes: { label: "episodes", min: 50, max: 500, step: 25, default: 200 },
    nSeeds:   { label: "seeds",    min: 5,  max: 50,  step: 5,  default: 20 },
    alpha:    { label: "α (step)", min: 0.05, max: 0.5, step: 0.05, default: 0.2 },
    eps:      { label: "ε (explore)", min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
  },
  render: (host, { episodes, nSeeds, alpha, eps }, slots) => {
    const gamma = 1.0; // Episodic cliff — no discounting in the canonical example.
    const mcRows = runMany(mcEpisode, episodes, nSeeds, gamma, alpha, eps);
    const tdRows = runMany(tdEpisode, episodes, nSeeds, gamma, alpha, eps);

    const tag = (rows, kind) => rows.map((r) => ({ ...r, kind }));
    const data = [...tag(mcRows, "MC"), ...tag(tdRows, "TD(0)")];

    // Per-method per-episode variance, averaged over later half of the
    // run (when both have converged). Headline number for the readout.
    const lateVar = (rows) => {
      const tail = rows.slice(Math.floor(rows.length / 2));
      return tail.reduce((s, r) => s + (r.hi - r.lo), 0) / tail.length;
    };

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      marginRight: 70,
      x: { label: "episode", grid: true },
      y: { label: "return per episode (mean over seeds)", grid: true },
      color: {
        legend: true,
        domain: ["MC", "TD(0)"],
        range: [palette.warning, palette.secondary],
      },
      marks: [
        Plot.areaY(data, {
          x: "ep",
          y1: "lo",
          y2: "hi",
          fill: "kind",
          fillOpacity: 0.18,
        }),
        Plot.line(data, {
          x: "ep",
          y: "mean",
          stroke: "kind",
          strokeWidth: 1.8,
        }),
      ],
    }));

    const mcBand = lateVar(mcRows);
    const tdBand = lateVar(tdRows);
    slots.readout.textContent =
      `late 10-90 band width: MC = ${mcBand.toFixed(1)} · TD(0) = ${tdBand.toFixed(1)}` +
      ` · MC variance ≈ ${(mcBand / tdBand).toFixed(1)}× TD's`;
  },
});

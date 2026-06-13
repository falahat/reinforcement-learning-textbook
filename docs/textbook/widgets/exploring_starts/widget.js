// Widget 5.D — Exploring starts vs ε-greedy coverage (Chapter 5).
//
// MC control needs every (s, a) pair visited infinitely often for Q
// to converge for all pairs. Exploring starts guarantees this *by
// construction* — each episode begins from a uniformly random (s, a).
// ε-soft policies guarantee it only *eventually* — coverage near the
// agent's frontier is dense, but rarely-visited corners stay starved.
//
// The widget renders two coverage heatmaps over a 5×5 gridworld
// (terminal in top-right corner with reward +1, every other step
// −0.04). Cell colour = max over actions of visit count log(1 + n)
// after `episodes` MC-control episodes. Exploring-starts paints the
// whole grid roughly uniformly; ε-greedy paints a tube from the
// fixed start (bottom-left) to the goal with thin coverage off the
// optimal corridor.
//
// Mount: `<div id="ch5-exploring-starts-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/exploring_starts/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const SIZE = 5;
const GOAL = { r: 0, c: SIZE - 1 };
const START = { r: SIZE - 1, c: 0 };
const ACTIONS = [
  { dr: -1, dc:  0 },
  { dr:  0, dc:  1 },
  { dr:  1, dc:  0 },
  { dr:  0, dc: -1 },
];
const MAX_STEPS = 80;
const GAMMA = 0.95;

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function step(r, c, a) {
  const nr = Math.max(0, Math.min(SIZE - 1, r + ACTIONS[a].dr));
  const nc = Math.max(0, Math.min(SIZE - 1, c + ACTIONS[a].dc));
  const done = nr === GOAL.r && nc === GOAL.c;
  return { r: nr, c: nc, reward: done ? 1 : -0.04, done };
}

function emptyQ() {
  const Q = [];
  for (let r = 0; r < SIZE; r++) {
    const row = [];
    for (let c = 0; c < SIZE; c++) row.push([0, 0, 0, 0]);
    Q.push(row);
  }
  return Q;
}

function emptyCounts() {
  const counts = [];
  for (let r = 0; r < SIZE; r++) {
    const row = [];
    for (let c = 0; c < SIZE; c++) row.push([0, 0, 0, 0]);
    counts.push(row);
  }
  return counts;
}

function greedy(Q, r, c) {
  const qs = Q[r][c];
  let best = 0;
  for (let a = 1; a < 4; a++) if (qs[a] > qs[best]) best = a;
  return best;
}

function epsGreedy(Q, r, c, eps, rand) {
  if (rand() < eps) return Math.floor(rand() * 4);
  return greedy(Q, r, c);
}

// One MC-control episode. `mode` ∈ {"es", "eps"}.
//   - "es": pick (s0, a0) uniformly; then follow greedy(Q) for the rest.
//   - "eps": start at fixed START; follow ε-greedy throughout.
function controlEpisode(Q, counts, mode, eps, rand) {
  let r, c, firstA;
  if (mode === "es") {
    // Sample a starting cell that is *not* the goal.
    do {
      r = Math.floor(rand() * SIZE);
      c = Math.floor(rand() * SIZE);
    } while (r === GOAL.r && c === GOAL.c);
    firstA = Math.floor(rand() * 4);
  } else {
    r = START.r; c = START.c;
    firstA = epsGreedy(Q, r, c, eps, rand);
  }

  const traj = [];
  let a = firstA;
  for (let t = 0; t < MAX_STEPS; t++) {
    counts[r][c][a] += 1;
    const out = step(r, c, a);
    traj.push({ r, c, a, reward: out.reward });
    r = out.r; c = out.c;
    if (out.done) break;
    a = mode === "es" ? greedy(Q, r, c) : epsGreedy(Q, r, c, eps, rand);
  }
  // First-visit MC update.
  let G = 0;
  const seen = new Set();
  for (let t = traj.length - 1; t >= 0; t--) {
    const { r: tr, c: tc, a: ta, reward } = traj[t];
    G = reward + GAMMA * G;
    const key = `${tr},${tc},${ta}`;
    if (!seen.has(key)) {
      seen.add(key);
      const n = ++Q[tr][tc][4 + ta] || 1;
      Q[tr][tc][4 + ta] = n; // running count for averaging
      Q[tr][tc][ta] += (G - Q[tr][tc][ta]) / n;
    }
  }
}

function runMode(mode, episodes, eps, seed) {
  const Q = emptyQ();
  // Extend Q with a per-(s,a) sample count in slots [4..7] for the
  // running average. Cleaner than carrying a second tensor.
  for (let r = 0; r < SIZE; r++)
    for (let c = 0; c < SIZE; c++)
      Q[r][c].push(0, 0, 0, 0);
  const counts = emptyCounts();
  const rand = rng(seed);
  for (let ep = 0; ep < episodes; ep++) controlEpisode(Q, counts, mode, eps, rand);
  return counts;
}

// Turn a per-(s,a) count tensor into a per-cell scalar = max over
// actions of log(1 + visit count). Picks out cells that are hit at
// least once in some direction; cells never visited stay at 0.
function cellHeat(counts) {
  const rows = [];
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const m = Math.max(...counts[r][c]);
      rows.push({ r, c, heat: Math.log(1 + m) });
    }
  }
  return rows;
}

function heatPlot(rows, title) {
  return Plot.plot({
    ...plotDefaults,
    height: 260,
    width: 320,
    marginLeft: 40,
    marginRight: 40,
    title,
    x: { label: "col", domain: d3.range(SIZE) },
    y: { label: "row", domain: d3.range(SIZE).reverse() },
    color: { type: "sqrt", scheme: "viridis", legend: false, label: "log(1+visits)" },
    marks: [
      Plot.cell(rows, { x: "c", y: "r", fill: "heat", inset: 0.5 }),
      Plot.text(rows, {
        x: "c", y: "r",
        text: (d) => (d.heat > 0 ? Math.round(Math.exp(d.heat) - 1) : ""),
        fill: "white", fontSize: 9, fontWeight: "bold",
      }),
      // Mark goal.
      Plot.text([{ r: GOAL.r, c: GOAL.c }], {
        x: "c", y: "r", text: () => "G", fill: palette.danger, fontSize: 14, fontWeight: "bold",
      }),
    ],
  });
}

defineWidget({
  hostId: "ch5-exploring-starts-widget",
  controls: {
    episodes: { label: "episodes", min: 100, max: 5000, step: 100, default: 1000 },
    eps:      { label: "ε (for ε-greedy)", min: 0.0, max: 0.5, step: 0.01, default: 0.1 },
  },
  slots: ["es", "eps"],
  render: (host, { episodes, eps }, slots) => {
    const esCounts  = runMode("es",  episodes, 0,   1);
    const epsCounts = runMode("eps", episodes, eps, 1);

    const esRows  = cellHeat(esCounts);
    const epsRows = cellHeat(epsCounts);

    slots.es.replaceChildren(heatPlot(esRows, "Exploring starts (uniform s₀, a₀)"));
    slots.eps.replaceChildren(heatPlot(epsRows, `ε-greedy from fixed start (ε = ${eps.toFixed(2)})`));

    // Pair-coverage fraction: how many (s, a) pairs have at least one
    // visit. The Watkins-style claim is "ES → 100% always, ε-greedy
    // → < 100% unless ε is large and you wait." Live readout makes it
    // visible.
    const covered = (counts) => {
      let n = 0;
      for (let r = 0; r < SIZE; r++)
        for (let c = 0; c < SIZE; c++)
          for (let a = 0; a < 4; a++)
            if (counts[r][c][a] > 0) n += 1;
      return n / (SIZE * SIZE * 4);
    };
    const esCov = covered(esCounts);
    const epsCov = covered(epsCounts);
    slots.readout.textContent =
      `(s,a) pair coverage: exploring starts = ${(100 * esCov).toFixed(1)}% · ` +
      `ε-greedy = ${(100 * epsCov).toFixed(1)}%`;
  },
});

// Widget 5.B — First-visit vs every-visit MC convergence race (Chapter 5).
//
// Both estimators converge to V^π, but with different finite-sample
// behaviour. First-visit MC averages independent returns (one per
// episode per state) — unbiased, higher variance. Every-visit MC
// averages every occurrence — slightly biased on intermediate states
// because samples within an episode are correlated, but uses more
// data so the variance falls faster.
//
// Set-up: the classic 5-state random walk MRP from S&B Example 6.2.
// States A, B, C, D, E in a chain; two terminal cells on the ends.
// Each step is left/right uniformly at random. Reward is 0
// everywhere except +1 on stepping out the right end; γ = 1. The
// true values are V*(s) = (1/6, 2/6, 3/6, 4/6, 5/6) — proportional
// to the probability of eventually exiting right from each state.
//
// The plot is RMS error over the 5 states vs episode count, averaged
// across many seeds, for both estimators. As `episodes` grows the
// every-visit curve is slightly below the first-visit one early
// (faster variance reduction) and the two converge to the same
// floor (both consistent).
//
// Mount: `<div id="ch5-mc-first-vs-every-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/mc_first_vs_every/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N_STATES = 5;
// V*(s) for the symmetric random walk: V*(A..E) = i/6 with i = 1..5.
const TRUE_V = d3.range(N_STATES).map((i) => (i + 1) / (N_STATES + 1));
const START = 2; // C — middle of the chain
const MAX_STEPS = 10000;

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Generate one episode: list of (state, reward) ending at termination.
// State -1 = left terminal (reward 0), state N_STATES = right terminal
// (reward +1). Returns only the non-terminal states and rewards.
function generateEpisode(rand) {
  const traj = [];
  let s = START;
  for (let t = 0; t < MAX_STEPS; t++) {
    const next = s + (rand() < 0.5 ? -1 : 1);
    if (next < 0) { traj.push({ s, r: 0 }); return traj; }
    if (next >= N_STATES) { traj.push({ s, r: 1 }); return traj; }
    traj.push({ s, r: 0 });
    s = next;
  }
  return traj;
}

function rmsError(V) {
  let sum = 0;
  for (let i = 0; i < N_STATES; i++) {
    const d = V[i] - TRUE_V[i];
    sum += d * d;
  }
  return Math.sqrt(sum / N_STATES);
}

// Run one seed of one variant. Returns RMS-error trace over episodes.
function runVariant(variant, episodes, alpha, seed) {
  const rand = rng(seed);
  const V = new Array(N_STATES).fill(0.5);
  const trace = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    const traj = generateEpisode(rand);
    // Compute returns from each timestep (γ = 1 so G_t = sum of future rewards).
    let G = 0;
    const seen = new Set();
    for (let t = traj.length - 1; t >= 0; t--) {
      G = traj[t].r + G;
      const s = traj[t].s;
      if (variant === "first" && seen.has(s)) continue;
      seen.add(s);
      V[s] += alpha * (G - V[s]);
    }
    trace[ep] = rmsError(V);
  }
  return trace;
}

defineWidget({
  hostId: "ch5-mc-first-vs-every-widget",
  controls: {
    episodes: { label: "episodes", min: 20, max: 500, step: 10, default: 200 },
    nSeeds:   { label: "seeds",    min: 10, max: 200, step: 10, default: 100 },
    alpha:    { label: "α (step)", min: 0.01, max: 0.3, step: 0.01, default: 0.05 },
  },
  render: (host, { episodes, nSeeds, alpha }, slots) => {
    // Average RMS error across seeds, per episode, per variant.
    const meanTrace = (variant) => {
      const sums = new Array(episodes).fill(0);
      for (let s = 0; s < nSeeds; s++) {
        const trace = runVariant(variant, episodes, alpha, s + 1);
        for (let i = 0; i < episodes; i++) sums[i] += trace[i];
      }
      return sums.map((v, i) => ({ ep: i + 1, err: v / nSeeds }));
    };

    const firstTrace = meanTrace("first").map((d) => ({ ...d, kind: "first-visit" }));
    const everyTrace = meanTrace("every").map((d) => ({ ...d, kind: "every-visit" }));
    const data = [...firstTrace, ...everyTrace];

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      x: { label: "episode", grid: true, type: "log" },
      // Auto-fit y so high-α curves (which can exceed 0.3 early on)
      // aren't visually clipped at the top of the chart.
      y: { label: "RMS error (avg over states)", grid: true, zero: true },
      color: {
        legend: true,
        domain: ["first-visit", "every-visit"],
        range: [palette.primary, palette.secondary],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.line(data, {
          x: "ep",
          y: "err",
          stroke: "kind",
          strokeWidth: 1.8,
        }),
      ],
    }));

    const finalFirst = firstTrace[firstTrace.length - 1].err;
    const finalEvery = everyTrace[everyTrace.length - 1].err;
    slots.readout.textContent =
      `RMS @ ep ${episodes}: first-visit = ${finalFirst.toFixed(4)} · ` +
      `every-visit = ${finalEvery.toFixed(4)} · ratio = ${(finalFirst / finalEvery).toFixed(2)}×`;
  },
});

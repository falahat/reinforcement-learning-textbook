// Widget 5.E — Episodic-cutoff sensitivity (Chapter 5).
//
// MC needs episode endings to compute G. On a continuing task you
// have to fake one with a hard cutoff T. Two failure modes:
//   - T too small → return underestimates the true V^π (missing tail).
//   - T too large → per-episode variance climbs, sample efficiency
//                   collapses, and you wait forever for each estimate.
//
// Set-up: a 9-state circular random walk, no terminals. State indices
// 0..8 wrap around; action = ±1 uniformly. Reward = +1 on entering
// state 0, else 0. Discount γ = 0.9. With these settings the
// stationary expected discounted return from any state has a
// closed form we can compute by linear solve once and use as
// ground truth.
//
// The widget plots ‖V̂ − V*‖₂ vs episode count for a single chosen
// cutoff T, and reports the per-episode return variance. Slide T
// up: the bias term shrinks (curve approaches truth) but the
// variance balloons (curve gets noisier and per-episode wall-clock
// scales linearly in T).
//
// Mount: `<div id="ch5-mc-cutoff-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/mc_cutoff/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N_STATES = 9;
const GAMMA = 0.9;

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Solve V = R + γ P V on the cycle directly so the curves have a
// fixed ground truth. Reward on entering state 0 (the "home"); π is
// uniform-left/right; transition is deterministic given the action.
//   P[i, (i−1) mod N] = P[i, (i+1) mod N] = 0.5
//   R[i] = 0.5 * (next=0?1:0) — the chance the NEXT state is 0
// since reward is on entering.
function trueValues() {
  const N = N_STATES;
  // Build (I − γ P) and r.
  const A = [];
  const b = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    A.push(new Array(N).fill(0));
    const next1 = (i - 1 + N) % N;
    const next2 = (i + 1) % N;
    A[i][i] = 1;
    A[i][next1] -= GAMMA * 0.5;
    A[i][next2] -= GAMMA * 0.5;
    b[i] = (next1 === 0 ? 0.5 : 0) + (next2 === 0 ? 0.5 : 0);
  }
  return solve(A, b);
}

// Gauss-Jordan elimination on a small dense system. N is 9 so this is
// trivial and we don't pull in a linear-algebra lib.
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const p = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}

const TRUE_V = trueValues();

// One MC episode with hard cutoff T starting at uniformly random state.
// Returns {s0, G_T} where G_T is the truncated discounted return.
function mcEpisode(T, rand) {
  const s0 = Math.floor(rand() * N_STATES);
  let s = s0;
  let G = 0, gpow = 1;
  for (let t = 0; t < T; t++) {
    const a = rand() < 0.5 ? -1 : 1;
    const sNext = (s + a + N_STATES) % N_STATES;
    const reward = sNext === 0 ? 1 : 0;
    G += gpow * reward;
    gpow *= GAMMA;
    s = sNext;
  }
  return { s0, G };
}

// Run MC at given cutoff T for `episodes` episodes, return RMS-error
// trace and per-episode return variance.
function runMC(T, episodes, alpha, seed) {
  const rand = rng(seed);
  const V = new Array(N_STATES).fill(0);
  const trace = new Array(episodes);
  let sumG = 0, sumG2 = 0;
  for (let ep = 0; ep < episodes; ep++) {
    const { s0, G } = mcEpisode(T, rand);
    V[s0] += alpha * (G - V[s0]);
    sumG += G;
    sumG2 += G * G;
    let sse = 0;
    for (let i = 0; i < N_STATES; i++) sse += (V[i] - TRUE_V[i]) ** 2;
    trace[ep] = Math.sqrt(sse / N_STATES);
  }
  const meanG = sumG / episodes;
  const varG = sumG2 / episodes - meanG * meanG;
  return { trace, varG };
}

defineWidget({
  hostId: "ch5-mc-cutoff-widget",
  controls: {
    T:        { label: "T (cutoff)",
                type: "select",
                options: [
                  { value: "10",   label: "T = 10" },
                  { value: "30",   label: "T = 30" },
                  { value: "100",  label: "T = 100" },
                  { value: "300",  label: "T = 300" },
                  { value: "1000", label: "T = 1000" },
                ],
                default: "100" },
    episodes: { label: "episodes", min: 50, max: 1000, step: 50, default: 400 },
    nSeeds:   { label: "seeds",    min: 5,  max: 50,   step: 5,  default: 20 },
    alpha:    { label: "α (step)", min: 0.01, max: 0.2, step: 0.005, default: 0.05 },
  },
  render: (host, { T, episodes, nSeeds, alpha }, slots) => {
    const Tn = parseInt(T, 10);
    // Average RMS-error trace + variance across seeds.
    const sums = new Array(episodes).fill(0);
    let totalVarG = 0;
    for (let s = 0; s < nSeeds; s++) {
      const { trace, varG } = runMC(Tn, episodes, alpha, s + 1);
      for (let i = 0; i < episodes; i++) sums[i] += trace[i];
      totalVarG += varG;
    }
    const rows = sums.map((v, i) => ({ ep: i + 1, err: v / nSeeds }));
    const avgVarG = totalVarG / nSeeds;

    // Truncation bias floor: at γ = 0.9 the geometric tail from T
    // onward has weight γ^T / (1 − γ). The per-state reward expectation
    // is bounded by 1; truncation discards at most that much value.
    const truncBound = Math.pow(GAMMA, Tn) / (1 - GAMMA);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      x: { label: "episode", grid: true, type: "log" },
      y: { label: "RMS error vs true Vπ", grid: true, type: "log" },
      marks: [
        Plot.ruleY([truncBound], { stroke: palette.danger, ...dashed }),
        Plot.text(
          [{ x: episodes * 0.95, y: truncBound, label: `truncation bound γᵀ/(1−γ) = ${truncBound.toExponential(2)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.danger, ...annotation },
        ),
        Plot.line(rows, { x: "ep", y: "err", stroke: palette.secondary, strokeWidth: 1.8 }),
      ],
    }));

    const finalErr = rows[rows.length - 1].err;
    slots.readout.textContent =
      `T = ${Tn}  ·  RMS @ ep ${episodes} = ${finalErr.toFixed(4)}  ·  ` +
      `per-episode Var(G) ≈ ${avgVarG.toFixed(3)}  ·  ` +
      `wall-cost ∝ T = ${Tn}×`;
  },
});

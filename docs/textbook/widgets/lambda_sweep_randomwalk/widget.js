// Widget 7.F — λ-sweep on the 5-state random walk (Chapter 7).
//
// Reproduces and extends the chapter's §7.8 worked example: the
// 5-state random walk, equal-probability moves, terminal rewards
// {0, +1}. The true V^star = (1/6, 2/6, ..., 5/6). After N episodes
// of TD(λ) (accumulating traces) at a chosen α, plot RMS error vs λ
// over a grid λ ∈ {0, 0.1, ..., 1.0}.
//
//   - Top plot:    U-curve of RMS error vs λ at the current (N, α).
//                  Best λ marker.
//   - Bottom plot: learned V vs true V for three picked λ values
//                  (0.0, current best, 1.0), bar chart per state.
//
// Sliders: episode count N, step size α. Results are averaged over
// many seeds per cell so the curve is smooth. To keep the widget
// snappy, each (λ, seed, episode-count) cell runs from scratch each
// scrub — fine because random walks are ~10 ticks each, 11 λ values
// × N_SEEDS × N episodes is well under 100k tiny updates.
//
// Pattern: chapter markdown has
//
//     <div id="ch7-lambda-sweep-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/lambda_sweep_randomwalk/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const N_STATES = 5;
const GAMMA = 1.0;          // undiscounted random walk (S&B convention)
const V_INIT = 0.5;         // S&B convention
const V_STAR = [1/6, 2/6, 3/6, 4/6, 5/6];
const LAMBDA_GRID = d3.range(0, 1.01, 0.1).map((x) => Math.round(x * 100) / 100);
const N_SEEDS = 30;         // averaged over this many random seeds per cell
const MAX_STEPS_PER_EP = 1000;  // safety cap; expected length ~25

// One TD(λ) episode with accumulating traces. Mutates V in place.
function runEpisode(V, rng, alpha, lambda) {
  const e = new Array(N_STATES).fill(0);
  let s = 2;  // start at C
  for (let step = 0; step < MAX_STEPS_PER_EP; step++) {
    const goRight = rng() < 0.5;
    const sNext = goRight ? s + 1 : s - 1;
    const terminal = sNext < 0 || sNext >= N_STATES;
    const r = sNext >= N_STATES ? 1 : 0;
    const vNext = terminal ? 0 : V[sNext];
    const delta = r + GAMMA * vNext - V[s];
    // Accumulating trace bump.
    e[s] += 1;
    // Apply δ·e to every state.
    for (let i = 0; i < N_STATES; i++) V[i] += alpha * delta * e[i];
    // Decay.
    for (let i = 0; i < N_STATES; i++) e[i] *= GAMMA * lambda;
    if (terminal) return;
    s = sNext;
  }
}

// RMS error of V against V_STAR.
function rmsError(V) {
  let sumSq = 0;
  for (let i = 0; i < N_STATES; i++) {
    const d = V[i] - V_STAR[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / N_STATES);
}

// Run a (λ, α, N_eps) experiment with `seeds` independent seeds, average
// the final RMS error. Also return the seed-averaged final V.
function sweepCell(lambda, alpha, nEps, seeds) {
  let rmsAcc = 0;
  const vAcc = new Array(N_STATES).fill(0);
  for (let s = 0; s < seeds; s++) {
    const rng = mulberry32(0xC0FFEE + s);
    const V = new Array(N_STATES).fill(V_INIT);
    for (let ep = 0; ep < nEps; ep++) runEpisode(V, rng, alpha, lambda);
    rmsAcc += rmsError(V);
    for (let i = 0; i < N_STATES; i++) vAcc[i] += V[i];
  }
  return {
    rms: rmsAcc / seeds,
    V: vAcc.map((v) => v / seeds),
  };
}

defineWidget({
  hostId: "ch7-lambda-sweep-widget",
  controls: {
    nEps: { label: "episodes N", min: 1, max: 200, step: 1, default: 10 },
    alpha: { label: "α (step size)", min: 0.01, max: 0.4, step: 0.01, default: 0.1 },
  },
  slots: ["curve", "values"],
  render: (host, { nEps, alpha }, slots) => {
    // Sweep λ grid; collect RMS per λ and the seed-averaged V.
    const cells = LAMBDA_GRID.map((lambda) => ({
      lambda,
      ...sweepCell(lambda, alpha, nEps, N_SEEDS),
    }));

    // Pick the optimal λ on this scrub.
    let best = cells[0];
    for (const c of cells) if (c.rms < best.rms) best = c;

    // -------- top plot: U-curve --------
    slots.curve.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 240,
        x: { label: "λ", domain: [0, 1], grid: true },
        y: { label: "RMS error vs V*", grid: true, zero: true },
        marks: [
          Plot.line(cells, {
            x: "lambda",
            y: "rms",
            stroke: palette.primary,
            strokeWidth: 2,
            curve: "monotone-x",
          }),
          Plot.dot(cells, {
            x: "lambda",
            y: "rms",
            fill: palette.primary,
            r: 3,
          }),
          Plot.ruleX([best.lambda], {
            stroke: palette.warning,
            ...dashed,
          }),
          Plot.dot([best], {
            x: "lambda",
            y: "rms",
            fill: palette.danger,
            r: 5,
          }),
          Plot.text(
            [{ x: best.lambda, y: best.rms, label: `best λ = ${best.lambda.toFixed(1)}, RMS = ${best.rms.toFixed(3)}` }],
            {
              x: "x",
              y: "y",
              text: "label",
              dx: 8,
              dy: -8,
              textAnchor: "start",
              fill: palette.danger,
              ...annotation,
            },
          ),
        ],
      }),
    );

    // -------- bottom plot: V vs V* for three picked λ --------
    // Pick λ ∈ {0, best, 1} and compare per-state.
    const picks = [
      { lambda: 0.0, label: "λ=0 (TD(0))", color: palette.secondary },
      { lambda: best.lambda, label: `λ=${best.lambda.toFixed(1)} (best)`, color: palette.primary },
      { lambda: 1.0, label: "λ=1 (MC)", color: palette.danger },
    ];

    const vRows = [];
    for (const p of picks) {
      const cell = cells.find((c) => Math.abs(c.lambda - p.lambda) < 1e-6);
      if (!cell) continue;
      for (let s = 0; s < N_STATES; s++) {
        vRows.push({
          state: String.fromCharCode(65 + s),    // A..E
          V: cell.V[s],
          series: p.label,
          color: p.color,
        });
      }
    }
    const truthRows = V_STAR.map((v, i) => ({
      state: String.fromCharCode(65 + i),
      V: v,
    }));

    slots.values.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 240,
        x: { label: "state" },
        y: { label: "V(s)", grid: true, domain: [0, 1.05] },
        color: {
          legend: true,
          domain: picks.map((p) => p.label),
          range: picks.map((p) => p.color),
        },
        marks: [
          Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
          Plot.barY(vRows, {
            x: "state",
            y: "V",
            fill: "series",
            fx: "series",
            fillOpacity: 0.85,
          }),
          // True V over each facet.
          Plot.ruleY(truthRows.map((r) => r.V), {
            stroke: palette.warning,
            ...dashed,
            strokeOpacity: 0.0,  // hidden — drawn per-state below
          }),
          Plot.dot(
            truthRows.flatMap((r) =>
              picks.map((p) => ({ ...r, series: p.label })),
            ),
            {
              x: "state",
              y: "V",
              fx: "series",
              fill: palette.warning,
              stroke: "white",
              strokeWidth: 1,
              r: 4,
            },
          ),
        ],
      }),
    );

    // Caption.
    slots.readout.textContent =
      `N = ${nEps} episodes  ·  α = ${alpha.toFixed(2)}  ·  ` +
      `seeds averaged: ${N_SEEDS}  ·  ` +
      `best λ = ${best.lambda.toFixed(1)} (RMS ${best.rms.toFixed(3)})  ·  ` +
      `orange dots = V* = (1/6, …, 5/6)`;
  },
});

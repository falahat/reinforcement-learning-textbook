// Widget 15.F — Catastrophic interference (Chapter 15).
//
// Neural networks (and overlapping-feature linear FA) overwrite old
// knowledge when retrained on a new region of input space. The
// canonical demonstration: train a single MLP-shaped function
// approximator on Region A for a while, then on Region B, then
// measure error on Region A again — it has collapsed.
//
// Here we use a tile-coded linear approximator on a 1-D regression
// target y = sin(2π·x) over x ∈ [0, 1]. The tile coding is
// deliberately *wide* (each tile covers `tile_width` of the input
// space, with `n_tiles` total tiles), so that updating any point
// also nudges the weights for neighbouring tiles — that is the
// generalisation that *causes* interference. With narrow tiles
// (one tile per point) there's no interference but no generalisation
// either; the slider lets the reader see the tradeoff.
//
// Schedule: split training into K phases, each of `phase_steps`. In
// odd phases, sample x ∈ [0, 0.5] (Region A). In even phases, sample
// x ∈ [0.5, 1] (Region B). The widget plots three curves on a
// shared x-axis:
//   - the true target y = sin(2π·x)
//   - the learned approximator after Region-B training (latest)
//   - a faded-out snapshot of the approximator at the end of the
//     previous Region-A phase
// plus a smaller plot of *test MSE on Region A* over the phase
// sequence — the characteristic up-down sawtooth that gives
// catastrophic interference its name.
//
//     <div id="ch15-catastrophic-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/catastrophic_interference/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const TARGET = (x) => Math.sin(2 * Math.PI * x);

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

// Overlapping tile features. Each tile is a triangular activation
// centred at c with half-width tileWidth/2. A point x activates every
// tile within range.
function makeCenters(nTiles) {
  const cs = new Float64Array(nTiles);
  for (let k = 0; k < nTiles; k++) cs[k] = (k + 0.5) / nTiles;
  return cs;
}

function phi(x, centers, tileWidth) {
  const out = new Float64Array(centers.length);
  const halfW = tileWidth / 2;
  for (let k = 0; k < centers.length; k++) {
    const d = Math.abs(x - centers[k]);
    if (d <= halfW) {
      out[k] = 1 - d / halfW; // triangular
    }
  }
  return out;
}

function predict(w, x, centers, tileWidth) {
  const p = phi(x, centers, tileWidth);
  let s = 0;
  for (let k = 0; k < centers.length; k++) s += w[k] * p[k];
  return s;
}

// Test-set MSE over a region [lo, hi].
function regionMse(w, centers, tileWidth, lo, hi, n = 50) {
  let s = 0;
  for (let i = 0; i < n; i++) {
    const x = lo + ((i + 0.5) / n) * (hi - lo);
    const y = TARGET(x);
    const yHat = predict(w, x, centers, tileWidth);
    s += (yHat - y) ** 2;
  }
  return s / n;
}

function train({ nTiles, tileWidth, phaseSteps, nPhases, alpha, seed }) {
  const rand = rng(seed);
  const centers = makeCenters(nTiles);
  const w = new Float64Array(nTiles);
  const mseTrace = [];
  // Final-state snapshots so we can plot the latest Region-A and
  // Region-B fits.
  let snapshotA = null;
  let snapshotB = null;
  for (let phase = 0; phase < nPhases; phase++) {
    const lo = phase % 2 === 0 ? 0.0 : 0.5;
    const hi = phase % 2 === 0 ? 0.5 : 1.0;
    for (let t = 0; t < phaseSteps; t++) {
      const x = lo + rand() * (hi - lo);
      const y = TARGET(x);
      const yHat = predict(w, x, centers, tileWidth);
      const err = y - yHat;
      const p = phi(x, centers, tileWidth);
      for (let k = 0; k < nTiles; k++) w[k] += alpha * err * p[k];
    }
    mseTrace.push({
      phase: phase + 1,
      region: phase % 2 === 0 ? "A" : "B",
      mseA: regionMse(w, centers, tileWidth, 0.0, 0.5),
      mseB: regionMse(w, centers, tileWidth, 0.5, 1.0),
    });
    if (phase % 2 === 0) snapshotA = new Float64Array(w);
    else snapshotB = new Float64Array(w);
  }
  return { centers, tileWidth, w, snapshotA, snapshotB, mseTrace };
}

defineWidget({
  hostId: "ch15-catastrophic-widget",
  controls: {
    tileWidth:  { label: "tile width",  min: 0.05, max: 0.5, step: 0.01, default: 0.15 },
    nTiles:     { label: "# tiles",     min: 5,    max: 80,  step: 1,    default: 20 },
    phaseSteps: { label: "phase steps", min: 100,  max: 5000, step: 100, default: 1500 },
    nPhases:    { label: "# phases",    min: 2,    max: 12,  step: 1,    default: 6 },
    alpha:      { label: "α",           min: 0.005, max: 0.2, step: 0.005, default: 0.05 },
    seed:       { label: "seed",        min: 1, max: 50, step: 1, default: 7 },
  },
  slots: ["main", "mse"],
  render: (host, p, slots) => {
    const out = train({
      nTiles: p.nTiles | 0, tileWidth: p.tileWidth,
      phaseSteps: p.phaseSteps | 0, nPhases: p.nPhases | 0,
      alpha: p.alpha, seed: p.seed | 0,
    });
    const N = 200;
    const curveRows = [];
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1);
      curveRows.push({ x, kind: "target", y: TARGET(x) });
      curveRows.push({ x, kind: "current", y: predict(out.w, x, out.centers, out.tileWidth) });
      if (out.snapshotA) {
        curveRows.push({
          x, kind: "after Region-A (snapshot)",
          y: predict(out.snapshotA, x, out.centers, out.tileWidth),
        });
      }
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "x", grid: true, domain: [0, 1] },
      y: { label: "y", grid: true, domain: [-1.5, 1.5] },
      color: {
        domain: ["target", "current", "after Region-A (snapshot)"],
        range: [palette.muted, palette.primary, palette.danger],
        legend: true,
      },
      marks: [
        Plot.ruleX([0.5], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: 0.5, y: 1.4, label: "Region A | Region B" }],
          { x: "x", y: "y", text: "label", textAnchor: "middle",
            fill: palette.warning, ...annotation },
        ),
        Plot.line(curveRows, { x: "x", y: "y", stroke: "kind", strokeWidth: 1.8 }),
      ],
    }));

    // MSE-trace over phases — long form so two series ride one chart.
    const mseLong = [];
    for (const row of out.mseTrace) {
      mseLong.push({ phase: row.phase, region: "MSE on A", value: row.mseA });
      mseLong.push({ phase: row.phase, region: "MSE on B", value: row.mseB });
    }
    slots.mse.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      x: { label: "phase", grid: true, type: "linear" },
      y: { label: "test MSE", grid: true, zero: true },
      color: {
        domain: ["MSE on A", "MSE on B"],
        range: [palette.danger, palette.secondary],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(mseLong, { x: "phase", y: "value", stroke: "region", strokeWidth: 2 }),
        Plot.dot(mseLong, { x: "phase", y: "value", fill: "region", r: 3 }),
      ],
    }));

    const last = out.mseTrace[out.mseTrace.length - 1];
    slots.readout.textContent =
      `after phase ${last.phase} (Region ${last.region}): ` +
      `MSE on A = ${fmt(last.mseA)}, MSE on B = ${fmt(last.mseB)} ` +
      `(jagged trace = interference)`;
  },
});

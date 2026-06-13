// Widget 8.B — Generalization demo for tile coding (Chapter 8).
//
// Demonstrates the central pedagogical claim of §8.2: an update at one
// query point reaches *every* nearby state that shares an active tile.
// The reader picks a "train" point, trains a single TD(0)-style update
// toward a target value, and watches a value heatmap V(s) = θ·φ(s) light
// up across a neighbourhood — wider when tiles are wider, finer when
// more tilings overlap.
//
// Construction:
//   * 2-D state space, unit square.
//   * T tilings, tile width w, Sutton-style asymmetric offsets:
//       o_t = (t/T) * (1, 3) * w  mod  w
//   * Each tile (tiling_index, ix, iy) hashes (deterministically) to a
//     slot index modulo M = 4096. θ ∈ ℝ^M, initialised to zero.
//   * Single update at the chosen train point: θ ← θ + α·(target − V(s))·φ(s).
//   * Then we evaluate V over a grid and render a heatmap. The "ball" of
//     non-zero V around the train point is the generalisation radius.
//
// What to look for:
//   * w = 0.1 + T = 4  → tight spot, sharp falloff (no generalisation).
//   * w = 0.3 + T = 16 → broad halo, smooth falloff (heavy generalisation).
//   * The halo is roughly a square of side ≈ 2w around the train point;
//     stepped because each tiling contributes a piecewise-constant lump.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const M = 4096;       // hash table size
const GRID_N = 60;    // evaluation grid resolution

// Tiny deterministic hash: mix three integers into [0, M).
function hashTile(tiling, ix, iy) {
  let h = 2166136261 ^ tiling;
  h = Math.imul(h ^ ix, 16777619);
  h = Math.imul(h ^ iy, 16777619);
  h ^= h >>> 13;
  h = Math.imul(h, 16777619);
  return ((h >>> 0) % M);
}

// Sutton-style asymmetric offsets: o_t = (t/T) * (1, 3) * w mod w.
function offset(t, T, w) {
  const ox = (((t / T) * 1 * w) % w + w) % w;
  const oy = (((t / T) * 3 * w) % w + w) % w;
  return [ox, oy];
}

function activeTiles(x, y, T, w) {
  const slots = new Array(T);
  for (let t = 0; t < T; t++) {
    const [ox, oy] = offset(t, T, w);
    const ix = Math.floor((x - ox) / w);
    const iy = Math.floor((y - oy) / w);
    slots[t] = hashTile(t, ix, iy);
  }
  return slots;
}

function valueAt(theta, x, y, T, w) {
  let v = 0;
  const slots = activeTiles(x, y, T, w);
  for (let t = 0; t < T; t++) v += theta[slots[t]];
  return v;
}

defineWidget({
  hostId: "ch8-generalization-widget",
  controls: {
    T:     { label: "T (tilings)",   min: 1,    max: 16,  step: 1,    default: 8 },
    w:     { label: "w (tile width)", min: 0.05, max: 0.4, step: 0.01, default: 0.2 },
    tx:    { label: "train x",        min: 0.0,  max: 1.0, step: 0.01, default: 0.5 },
    ty:    { label: "train y",        min: 0.0,  max: 1.0, step: 0.01, default: 0.5 },
    alpha: { label: "α · target",     min: 0.0,  max: 1.0, step: 0.01, default: 1.0 },
  },
  slots: ["main", "slice"],
  render: (host, { T, w, tx, ty, alpha }, slots) => {
    const Ti = Math.round(T);

    // One semi-gradient step toward target = alpha (since V starts at 0,
    // the update value at the train point equals alpha after one step).
    // θ ← θ + α·(target − V)·φ. With θ_0 = 0 and target=alpha, the step
    // simply writes α/T into each of the T active slots so V(train) = α.
    const theta = new Float64Array(M);
    const trainSlots = activeTiles(tx, ty, Ti, w);
    for (let t = 0; t < Ti; t++) theta[trainSlots[t]] = alpha / Ti;

    // Evaluate V over the grid.
    const cells = [];
    for (let iy = 0; iy < GRID_N; iy++) {
      const y = (iy + 0.5) / GRID_N;
      for (let ix = 0; ix < GRID_N; ix++) {
        const x = (ix + 0.5) / GRID_N;
        cells.push({ x, y, v: valueAt(theta, x, y, Ti, w) });
      }
    }

    const vMax = d3.max(cells, (d) => d.v) || 1;

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 320,
        width: 360,
        marginLeft: 36,
        marginBottom: 32,
        x: { label: "state x", domain: [0, 1] },
        y: { label: "state y", domain: [0, 1] },
        color: {
          type: "linear",
          domain: [0, vMax],
          range: ["#0d1f2c", palette.primary],
          legend: true,
          label: "V(s) after one update",
        },
        marks: [
          Plot.raster(cells, {
            x: "x",
            y: "y",
            fill: "v",
            width: GRID_N,
            height: GRID_N,
            interpolate: "nearest",
          }),
          Plot.dot([{ x: tx, y: ty }], {
            x: "x",
            y: "y",
            fill: palette.warning,
            stroke: "white",
            strokeWidth: 1.5,
            r: 6,
          }),
        ],
      }),
    );

    // Horizontal slice through y = ty: V vs x.
    const slice = [];
    for (let i = 0; i < 200; i++) {
      const x = i / 199;
      slice.push({ x, v: valueAt(theta, x, ty, Ti, w) });
    }
    slots.slice.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 160,
        x: { label: "x (slice through train y)", domain: [0, 1], grid: true },
        y: { label: "V(x, train_y)", grid: true },
        marks: [
          Plot.ruleY([0], { stroke: palette.muted }),
          Plot.line(slice, {
            x: "x",
            y: "v",
            stroke: palette.primary,
            strokeWidth: 2,
          }),
          Plot.ruleX([tx], { stroke: palette.warning, strokeDasharray: "4 2" }),
        ],
      }),
    );

    // Read off the bounding box of the non-zero region on the slice.
    const nonzero = slice.filter((d) => d.v > 1e-9);
    let radius = 0;
    if (nonzero.length > 0) {
      const xs = nonzero.map((d) => d.x);
      radius = (Math.max(...xs) - Math.min(...xs)) / 2;
    }

    slots.readout.textContent =
      `T = ${Ti}, w = ${w.toFixed(2)}  ·  ` +
      `V(train) = ${alpha.toFixed(2)}  ·  ` +
      `slice halo radius ≈ ${radius.toFixed(3)} ` +
      `(theory: ≈ w = ${w.toFixed(2)})`;
  },
});

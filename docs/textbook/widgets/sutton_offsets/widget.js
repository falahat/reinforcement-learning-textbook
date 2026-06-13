// Widget 8.H — Sutton offset asymmetry visualiser (Chapter 8, §8.6).
//
// Draws the partition lines that T overlapping tilings cut into the
// 2-D unit square, under two offset schemes:
//
//   * Aligned offsets:   o_t = (0, 0) for all t.  All tilings agree on
//     tile boundaries; effective resolution stays w (no refinement).
//   * Sutton-style offsets:  o_t = (t/T) * (1, 3) * w  mod w.
//     Tilings disagree asymmetrically; the intersection of all boundary
//     lines partitions the square into ≈ (T/w)^2 distinct *cells*, each
//     of size w/T on each axis. Effective resolution refines by T.
//
// Each tiling gets a distinct hue. The readout reports the count of
// unique tile-activation patterns (i.e. equivalence classes under
// "same active tile in every tiling") inside the unit square.
//
// Pedagogy: aligned tilings are T redundant copies of one partition.
// Sutton offsets interleave the boundaries so the joint refinement is
// the actual generalisation surface the textbook's worked example
// (§8.6) calls out.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const TILING_COLOURS = [
  "#4caf50", "#42a5f5", "#ffb74d", "#e57373",
  "#ba68c8", "#26c6da", "#ffee58", "#9ccc65",
  "#f48fb1", "#80cbc4", "#ce93d8", "#ffab91",
];

function offsetSutton(t, T, dim, w) {
  const k = 2 * dim + 1; // 1, 3, 5, ...
  return (((t / T) * k * w) % w + w) % w;
}

function offsetAligned(_t, _T, _dim, _w) {
  return 0;
}

// Collect vertical and horizontal grid lines for all T tilings.
function collectLines(T, w, offsetFn) {
  const lines = [];
  for (let t = 0; t < T; t++) {
    const ox = offsetFn(t, T, 0, w);
    const oy = offsetFn(t, T, 1, w);
    // Vertical lines: x = ox + k*w in (0, 1).
    for (let k = 0; ox + k * w < 1.0001; k++) {
      const x = ox + k * w;
      if (x > 0 && x < 1) lines.push({ t, kind: "v", a: x });
    }
    for (let k = -1; ox + k * w > -0.0001; k--) {
      const x = ox + k * w;
      if (x > 0 && x < 1) lines.push({ t, kind: "v", a: x });
    }
    // Horizontal lines: y = oy + k*w.
    for (let k = 0; oy + k * w < 1.0001; k++) {
      const y = oy + k * w;
      if (y > 0 && y < 1) lines.push({ t, kind: "h", a: y });
    }
    for (let k = -1; oy + k * w > -0.0001; k--) {
      const y = oy + k * w;
      if (y > 0 && y < 1) lines.push({ t, kind: "h", a: y });
    }
  }
  return lines;
}

// Count distinct joint-tile equivalence classes on a dense grid.
function countUniqueCells(T, w, offsetFn, gridN = 200) {
  const seen = new Set();
  for (let iy = 0; iy < gridN; iy++) {
    const y = (iy + 0.5) / gridN;
    for (let ix = 0; ix < gridN; ix++) {
      const x = (ix + 0.5) / gridN;
      const parts = new Array(T);
      for (let t = 0; t < T; t++) {
        const ox = offsetFn(t, T, 0, w);
        const oy = offsetFn(t, T, 1, w);
        const px = Math.floor((x - ox) / w);
        const py = Math.floor((y - oy) / w);
        parts[t] = `${px},${py}`;
      }
      seen.add(parts.join("|"));
    }
  }
  return seen.size;
}

function makePlot(T, w, scheme, title) {
  const offsetFn = scheme === "sutton" ? offsetSutton : offsetAligned;
  const lines = collectLines(T, w, offsetFn);

  return Plot.plot({
    ...plotDefaults,
    title,
    height: 340,
    width: 340,
    marginLeft: 32,
    marginBottom: 28,
    x: { domain: [0, 1], label: null, ticks: 5 },
    y: { domain: [0, 1], label: null, ticks: 5 },
    color: { type: "ordinal", domain: d3.range(T), range: TILING_COLOURS },
    marks: [
      Plot.frame(),
      Plot.ruleX(lines.filter((d) => d.kind === "v"), {
        x: "a", stroke: "t",
        strokeWidth: 1.2, strokeOpacity: 0.7,
      }),
      Plot.ruleY(lines.filter((d) => d.kind === "h"), {
        y: "a", stroke: "t",
        strokeWidth: 1.2, strokeOpacity: 0.7,
      }),
    ],
  });
}

defineWidget({
  hostId: "ch8-sutton-offsets-widget",
  controls: {
    T: { label: "T (tilings)",     min: 1,    max: 12,  step: 1,    default: 4 },
    w: { label: "w (tile width)",  min: 0.10, max: 0.5, step: 0.01, default: 0.25 },
  },
  slots: ["main", "aligned"],
  render: (host, { T, w }, slots) => {
    const Ti = Math.round(T);

    slots.main.replaceChildren(
      makePlot(Ti, w, "sutton", "Sutton offsets (asymmetric)"),
    );
    slots.aligned.replaceChildren(
      makePlot(Ti, w, "aligned", "Aligned offsets (all zero)"),
    );

    const sutCells = countUniqueCells(Ti, w, offsetSutton);
    const aliCells = countUniqueCells(Ti, w, offsetAligned);
    const expectedAli = Math.pow(Math.ceil(1 / w), 2);
    // Effective resolution under Sutton: cells of side ~ w/T.
    const expectedSutLin = Math.ceil(1 / (w / Ti));
    const expectedSut = expectedSutLin * expectedSutLin;

    slots.readout.textContent =
      `unique joint cells in unit square — ` +
      `Sutton: ${sutCells} (theory ≈ ${expectedSut})  ·  ` +
      `aligned: ${aliCells} (theory = ${expectedAli})  ·  ` +
      `effective resolution under Sutton: w/T = ${(w / Ti).toFixed(3)}`;
  },
});

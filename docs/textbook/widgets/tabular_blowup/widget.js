// Widget 8.D — Tabular blow-up calculator (Chapter 8, §8.1).
//
// Visualises why a table that stores one Q-value per discretised state
// is hopeless for anything but toy problems. The reader picks a state
// dimension n and bins-per-dim b; the widget plots |S| = b^n on a log
// y-axis, with reference lines at:
//   * 10^9  (a billion — fits in RAM if entries are tiny)
//   * 10^12 (a trillion — disk-scale)
//   * 10^80 (atoms in the observable universe)
//
// A second readout breaks down the Simulator's 7 observation blocks
// (drives, body, emotions, perception, world, ambient, episodic), each
// toggleable; the total |S| at b bins is shown live. With all blocks
// on at b = 10, the total is 10^251 — the chapter's headline number.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

// Simulator observation blocks, dim per block. Matches the §8.1 table.
const BLOCKS = [
  { name: "drives",       dim: 14 },
  { name: "body",         dim:  8 },
  { name: "emotions",     dim:  8 },
  { name: "perception",   dim: 64 },
  { name: "world",        dim:  5 },
  { name: "ambient hist", dim:  8 },
  { name: "episodic mem", dim: 144 },
];

// log10(b^n) = n * log10(b).
function logCount(n, b) {
  return n * Math.log10(b);
}

defineWidget({
  hostId: "ch8-tabular-blowup-widget",
  controls: {
    n: { label: "n (state dim)",  min: 1, max: 60, step: 1, default: 10 },
    b: { label: "b (bins/dim)",   min: 2, max: 20, step: 1, default: 10 },
  },
  slots: ["main", "blocks"],
  render: (host, { n, b }, slots) => {
    const nn = Math.round(n);
    const bb = Math.round(b);

    // Sweep n from 1 to 60 holding b fixed.
    const sweep = d3.range(1, 61).map((nv) => ({
      n: nv,
      log10count: logCount(nv, bb),
      count: Math.pow(bb, nv),
    }));
    const here = { n: nn, log10count: logCount(nn, bb) };

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 260,
        x: { label: "n (state dimension)", grid: true },
        y: {
          label: "|S| = b^n  (log10)",
          grid: true,
          domain: [0, Math.max(85, here.log10count + 5)],
        },
        marks: [
          // Reference horizontal rules: log10 of 10^9, 10^12, 10^80.
          Plot.ruleY([9],  { stroke: palette.muted, ...dashed }),
          Plot.ruleY([12], { stroke: palette.muted, ...dashed }),
          Plot.ruleY([80], { stroke: palette.danger, ...dashed }),
          Plot.text(
            [
              { x: 60, y: 9,  label: "10^9  (RAM)" },
              { x: 60, y: 12, label: "10^12 (disk)" },
              { x: 60, y: 80, label: "10^80 (atoms in observable universe)" },
            ],
            {
              x: "x",
              y: "y",
              text: "label",
              textAnchor: "end",
              dy: -5,
              fill: (d) => (d.y >= 80 ? palette.danger : palette.muted),
              ...annotation,
            },
          ),
          Plot.line(sweep, {
            x: "n",
            y: "log10count",
            stroke: palette.primary,
            strokeWidth: 2,
          }),
          Plot.dot([here], {
            x: "n",
            y: "log10count",
            fill: palette.warning,
            stroke: "white",
            strokeWidth: 1.5,
            r: 6,
          }),
        ],
      }),
    );

    // Block-by-block log-state-count bars at the chosen b.
    const blockData = BLOCKS.map((blk) => ({
      block: blk.name,
      dim: blk.dim,
      log10count: blk.dim * Math.log10(bb),
    }));
    blockData.push({
      block: "TOTAL",
      dim: BLOCKS.reduce((s, blk) => s + blk.dim, 0),
      log10count: BLOCKS.reduce((s, blk) => s + blk.dim, 0) * Math.log10(bb),
    });

    slots.blocks.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 240,
        marginLeft: 120,
        x: { label: `log10 of states per block (b = ${bb})`, grid: true },
        y: { label: null, domain: blockData.map((d) => d.block) },
        marks: [
          Plot.ruleX([80], { stroke: palette.danger, ...dashed }),
          Plot.barX(blockData, {
            x: "log10count",
            y: "block",
            fill: (d) => (d.block === "TOTAL" ? palette.danger : palette.secondary),
          }),
          Plot.text(blockData, {
            x: "log10count",
            y: "block",
            text: (d) => `dim=${d.dim} → 10^${d.log10count.toFixed(1)}`,
            textAnchor: "start",
            dx: 4,
            fontSize: 11,
            fill: "white",
          }),
        ],
      }),
    );

    // f32 storage cost: 4 bytes per state.
    const log10Bytes = here.log10count + Math.log10(4);
    slots.readout.textContent =
      `|S| = ${bb}^${nn} = 10^${here.log10count.toFixed(2)}  ·  ` +
      `f32 storage: 10^${log10Bytes.toFixed(2)} bytes` +
      (here.log10count > 80 ? "  ← exceeds 10^80 (atoms in the observable universe)" : "");
  },
});

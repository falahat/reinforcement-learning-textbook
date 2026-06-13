// Widget 8.C — Per-block tiling demo (Chapter 8, §8.8 / §8.9).
//
// Two synthetic 4-D observations are shown side by side, each split into
// two 2-D "blocks": Drives (Hunger, Thirst) and Perception (Food, Threat).
// The reader drags four sliders to pick (drives_A, perception_A) and
// (drives_B, perception_B), and watches the active-tile sets for both
// observations under TWO tiling schemes:
//
//   * Joint tiling: one 4-D hash namespace; tiles are 4-tuples of indices.
//     Two observations share a tile only if *all* 4 dims fall in the
//     same cell of the same tiling.
//   * Per-block tiling: drives and perception each get their own 2-D
//     hash namespace, then the active-tile set is the union. Two
//     observations share a drive tile if their drives agree, regardless
//     of perception.
//
// The headline readout is "shared active tiles" under each scheme. With
// the default sliders (drives identical, perception different), joint
// tiling shares 0 tiles, per-block tiling shares T tiles (one per tiling
// on the drives block). That's the §8.9 claim made empirically visible.

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const M = 8192; // hash namespace size per block
const T = 8;    // tilings
const W = 0.25; // tile width

function hash(...ints) {
  let h = 2166136261;
  for (const v of ints) {
    h ^= v | 0;
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 13;
  h = Math.imul(h, 16777619);
  return ((h >>> 0) % M);
}

// Sutton-style asymmetric offsets, parameterised by block index so the
// two blocks use independent offset sequences.
function offset(t, dim, blockId) {
  // odd-integer asymmetry: (1, 3, 5, ...) scaled
  const k = 2 * dim + 1 + 7 * blockId;
  return (((t / T) * k * W) % W + W) % W;
}

function tilesJoint(obs) {
  // obs = [d1, d2, p1, p2] — one tile per tiling, one big 4-D index.
  const slots = new Array(T);
  for (let t = 0; t < T; t++) {
    const ix = Math.floor((obs[0] - offset(t, 0, 0)) / W);
    const iy = Math.floor((obs[1] - offset(t, 1, 0)) / W);
    const iz = Math.floor((obs[2] - offset(t, 2, 0)) / W);
    const iw = Math.floor((obs[3] - offset(t, 3, 0)) / W);
    slots[t] = hash(0, t, ix, iy, iz, iw);
  }
  return slots;
}

function tilesPerBlock(obs) {
  // T drive-block tiles + T perception-block tiles, in two namespaces.
  const driveSlots = new Array(T);
  const percSlots = new Array(T);
  for (let t = 0; t < T; t++) {
    const dx = Math.floor((obs[0] - offset(t, 0, 1)) / W);
    const dy = Math.floor((obs[1] - offset(t, 1, 1)) / W);
    driveSlots[t] = hash(1, t, dx, dy);
    const px = Math.floor((obs[2] - offset(t, 0, 2)) / W);
    const py = Math.floor((obs[3] - offset(t, 1, 2)) / W);
    percSlots[t] = hash(2, t, px, py);
  }
  return { driveSlots, percSlots };
}

function countShared(a, b) {
  const setB = new Set(b);
  let c = 0;
  for (const s of a) if (setB.has(s)) c++;
  return c;
}

defineWidget({
  hostId: "ch8-per-block-widget",
  controls: {
    dA_x: { label: "A.drives.hunger",    min: 0, max: 1, step: 0.01, default: 0.40 },
    dA_y: { label: "A.drives.thirst",    min: 0, max: 1, step: 0.01, default: 0.60 },
    pA_x: { label: "A.percep.food",      min: 0, max: 1, step: 0.01, default: 0.20 },
    pA_y: { label: "A.percep.threat",    min: 0, max: 1, step: 0.01, default: 0.20 },
    dB_x: { label: "B.drives.hunger",    min: 0, max: 1, step: 0.01, default: 0.40 },
    dB_y: { label: "B.drives.thirst",    min: 0, max: 1, step: 0.01, default: 0.60 },
    pB_x: { label: "B.percep.food",      min: 0, max: 1, step: 0.01, default: 0.80 },
    pB_y: { label: "B.percep.threat",    min: 0, max: 1, step: 0.01, default: 0.90 },
  },
  slots: ["bars", "diagram"],
  render: (host, p, slots) => {
    const A = [p.dA_x, p.dA_y, p.pA_x, p.pA_y];
    const B = [p.dB_x, p.dB_y, p.pB_x, p.pB_y];

    const jA = tilesJoint(A);
    const jB = tilesJoint(B);
    const sharedJoint = countShared(jA, jB);

    const pbA = tilesPerBlock(A);
    const pbB = tilesPerBlock(B);
    const sharedDrives = countShared(pbA.driveSlots, pbB.driveSlots);
    const sharedPerc   = countShared(pbA.percSlots, pbB.percSlots);
    const sharedPerBlock = sharedDrives + sharedPerc;

    // Bar chart of shared-tile counts.
    const bars = [
      { scheme: "joint (4-D)",           shared: sharedJoint,   total: T,     part: "all" },
      { scheme: "per-block · drives",    shared: sharedDrives,  total: T,     part: "drives" },
      { scheme: "per-block · perception",shared: sharedPerc,    total: T,     part: "perception" },
      { scheme: "per-block · total",     shared: sharedPerBlock,total: 2 * T, part: "all" },
    ];

    const colorFor = (d) =>
      d.scheme === "joint (4-D)" ? palette.danger :
      d.part === "drives"        ? palette.secondary :
      d.part === "perception"    ? palette.accent :
                                   palette.primary;

    slots.bars.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 200,
        marginLeft: 170,
        x: { label: "shared active tiles between A and B", grid: true },
        y: { label: null },
        marks: [
          Plot.barX(bars, {
            x: "shared",
            y: "scheme",
            fill: colorFor,
          }),
          Plot.text(bars, {
            x: "shared",
            y: "scheme",
            text: (d) => `${d.shared} / ${d.total}`,
            textAnchor: "start",
            dx: 4,
            fontSize: 11,
            fill: "white",
          }),
          Plot.ruleX([0]),
        ],
      }),
    );

    // Diagram: two unit squares (Drives | Perception), pin A and B
    // positions on each block. Visual answer to "are the drives equal?"
    const driveDots = [
      { block: "Drives",     obs: "A", x: A[0], y: A[1] },
      { block: "Drives",     obs: "B", x: B[0], y: B[1] },
    ];
    const percDots = [
      { block: "Perception", obs: "A", x: A[2], y: A[3] },
      { block: "Perception", obs: "B", x: B[2], y: B[3] },
    ];
    const allDots = [...driveDots, ...percDots];

    slots.diagram.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 220,
        width: 640,
        fx: { domain: ["Drives", "Perception"], label: null },
        x: { domain: [0, 1], label: "axis 1", grid: true },
        y: { domain: [0, 1], label: "axis 2", grid: true },
        marks: [
          Plot.frame(),
          Plot.dot(allDots, {
            fx: "block",
            x: "x",
            y: "y",
            r: 8,
            fill: (d) => (d.obs === "A" ? palette.secondary : palette.accent),
            stroke: "white",
            strokeWidth: 1.5,
          }),
          Plot.text(allDots, {
            fx: "block",
            x: "x",
            y: "y",
            text: "obs",
            fill: "white",
            fontSize: 10,
            fontWeight: "bold",
          }),
        ],
      }),
    );

    const driveEqual =
      Math.abs(A[0] - B[0]) < 1e-6 && Math.abs(A[1] - B[1]) < 1e-6;
    const note = driveEqual
      ? "drives identical → per-block shares all T drive tiles, joint shares 0 (perception differs)."
      : "drives differ → per-block sharing on drives drops; joint sharing already near zero.";
    slots.readout.textContent =
      `joint: ${sharedJoint}/${T}  ·  per-block: drives ${sharedDrives}/${T}, perception ${sharedPerc}/${T}, total ${sharedPerBlock}/${2 * T}. ${note}`;
  },
});

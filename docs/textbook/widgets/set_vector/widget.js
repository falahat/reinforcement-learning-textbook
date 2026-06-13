// Widget 8.G — Set-as-vector salience-swap demo (Chapter 8, §8.10).
//
// Episodic memory: K slots × F fields per slot. The "slots" are
// semantically a set (unordered), but stored as a fixed-order vector.
// A small salience perturbation can reorder slots, producing a
// discontinuous jump in the encoded vector — and in the tile-coded
// representation. A set-invariant encoding (e.g. summed per-slot tile
// activations) sees the SAME hash after the swap.
//
// The widget displays the K-slot vector before and after a swap, plus
// a Monte Carlo experiment over many random swaps: how often does the
// positional hash change vs. the set-coded one?
//
//     <div id="ch8-set-vector-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/set_vector/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";
import { gridAxes } from "../shared/plot.js";

const K_SLOTS = 6;
const F_FIELDS = 3;
const FIELD_NAMES = ["kind", "salience", "recency"];
const NUM_SWAPS_MC = 500;

// Build a deterministic "canonical" memory set from a seed.
function buildSlots(seed) {
  const rand = mulberry32(seed | 0);
  const slots = [];
  for (let k = 0; k < K_SLOTS; k++) {
    slots.push([
      Math.floor(rand() * 4),        // kind id in 0..3
      Math.round(rand() * 100) / 100, // salience 0..1
      Math.round(rand() * 100) / 100, // recency 0..1
    ]);
  }
  // Sort descending by salience (field 1) to match the project's convention.
  slots.sort((a, b) => b[1] - a[1]);
  return slots;
}

// Positional hash: concatenate every (slot index, field index, value)
// into a fold-hash. Sensitive to slot order.
function positionalHash(slots) {
  let h = 2166136261 >>> 0;
  for (let k = 0; k < slots.length; k++) {
    for (let f = 0; f < F_FIELDS; f++) {
      // quantize to 2 decimal places so we hash integers
      const x = Math.round(slots[k][f] * 100) | 0;
      h ^= (k + 1) * 0x9e3779b1;
      h = Math.imul(h ^ x, 16777619) >>> 0;
      h ^= (f + 1) * 0x85ebca6b;
    }
  }
  return h >>> 0;
}

// Set-coded hash: per-slot tile activations summed (order-independent).
// We hash each slot independently into a small int, then XOR-aggregate.
function setCodedHash(slots) {
  let h = 0;
  for (const slot of slots) {
    let s = 2166136261 >>> 0;
    for (let f = 0; f < F_FIELDS; f++) {
      const x = Math.round(slot[f] * 100) | 0;
      s = Math.imul(s ^ x, 16777619) >>> 0;
      s ^= (f + 1) * 0x85ebca6b;
    }
    h = (h + (s >>> 0)) >>> 0; // commutative aggregation
  }
  return h >>> 0;
}

// Apply a single salience-swap perturbation: pick two adjacent slots i, i+1
// and swap them (as the project's salience-sort does when their saliences
// cross).
function swapAdjacent(slots, rand) {
  const out = slots.map((s) => s.slice());
  const i = Math.floor(rand() * (out.length - 1));
  const tmp = out[i];
  out[i] = out[i + 1];
  out[i + 1] = tmp;
  return out;
}

// Build a grid-cell array for one slot vector. Each cell carries (k, f, v).
function gridCells(slots) {
  const cells = [];
  for (let k = 0; k < slots.length; k++) {
    for (let f = 0; f < F_FIELDS; f++) {
      cells.push({ k, f, v: slots[k][f], label: fmt(slots[k][f]) });
    }
  }
  return cells;
}

function heatmap(cells, title) {
  return Plot.plot({
    ...plotDefaults,
    height: 150,
    width: 360,
    marginLeft: 60,
    marginTop: 22,
    title,
    ...gridAxes({ nx: K_SLOTS, ny: F_FIELDS }, { axis: null }, {
      x: { label: "slot k" },
      y: { label: null, tickFormat: (i) => FIELD_NAMES[i] },
    }),
    color: { type: "linear", domain: [0, 3], range: ["#1a1a2e", palette.secondary] },
    marks: [
      Plot.cell(cells, { x: "k", y: "f", fill: "v", inset: 0.5 }),
      Plot.text(cells, { x: "k", y: "f", text: "label", fill: "#fff", fontSize: 9 }),
    ],
  });
}

defineWidget({
  hostId: "ch8-set-vector-widget",
  controls: {
    seed: { label: "memory seed", min: 1, max: 50, step: 1, default: 5 },
  },
  slots: ["before", "after", "stats"],
  render: (host, { seed }, slots) => {
    const rand = mulberry32(seed | 0);
    const before = buildSlots(seed | 0);
    const after = swapAdjacent(before, rand);

    slots.before.replaceChildren(heatmap(gridCells(before), "Before swap"));
    slots.after.replaceChildren(heatmap(gridCells(after), "After swap (slots i, i+1 reordered)"));

    // Monte Carlo: how often does each hash change under random adjacent
    // swaps of an unchanged memory set?
    const mcRand = mulberry32((seed | 0) + 9999);
    let posDiff = 0;
    let setDiff = 0;
    for (let trial = 0; trial < NUM_SWAPS_MC; trial++) {
      const base = buildSlots((seed | 0) + trial * 7);
      const perturbed = swapAdjacent(base, mcRand);
      if (positionalHash(base) !== positionalHash(perturbed)) posDiff++;
      if (setCodedHash(base) !== setCodedHash(perturbed)) setDiff++;
    }

    const bars = [
      { encoding: "positional", rate: posDiff / NUM_SWAPS_MC },
      { encoding: "set-coded",  rate: setDiff / NUM_SWAPS_MC },
    ];

    slots.stats.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 170,
      width: 360,
      marginLeft: 90,
      x: { label: "hash-change rate (across 500 random adjacent swaps)", domain: [0, 1.05], grid: true },
      y: { label: null, domain: ["positional", "set-coded"] },
      marks: [
        Plot.barX(bars, {
          y: "encoding", x: "rate",
          fill: (d) => d.encoding === "positional" ? palette.danger : palette.primary,
        }),
        Plot.text(bars, {
          y: "encoding", x: "rate", text: (d) => fmt(d.rate),
          dx: 6, textAnchor: "start", fill: palette.muted, fontSize: 10,
        }),
      ],
    }));

    const beforeP = positionalHash(before);
    const afterP = positionalHash(after);
    const beforeS = setCodedHash(before);
    const afterS = setCodedHash(after);
    const posChanged = beforeP !== afterP;
    const setChanged = beforeS !== afterS;
    slots.readout.textContent =
      `this swap — positional ${posChanged ? "DIFFERS" : "matches"}  ·  set-coded ${setChanged ? "DIFFERS" : "matches"}`;
  },
});

// Widget 16.A — Homeostatic reward landscape (Chapter 16).
//
// Direct visualization of the homeostatic reward formula
//
//   R = w_alive - sum_d w_d * d^p - bio_costs
//
// The reader sets three drive levels d ∈ [0, 1] (hunger, thirst,
// fatigue), the convex exponent p ∈ [1, 3], a shared drive weight
// w_d, and the alive baseline w_alive. The widget renders a stacked
// bar chart decomposing R per term, plus the marginal cost curve
// c'(d) = p · d^{p−1} for the dominant drive — so the reader can
// connect "this drive is the painful one" to "this drive's slope
// is the steepest". The total R readout lets readers numerically
// verify the §16.1 derivation.
//
// Mount: in §16.1 of Chapter 16.
//
//     <div id="ch16-reward-landscape-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/homeo_reward_landscape/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const DRIVE_NAMES = ["hunger", "thirst", "fatigue"];
const DRIVE_COLORS = [palette.danger, palette.secondary, palette.accent];

defineWidget({
  hostId: "ch16-reward-landscape-widget",
  controls: {
    hunger:  { label: "hunger d_1",  min: 0, max: 1, step: 0.01, default: 0.6 },
    thirst:  { label: "thirst d_2",  min: 0, max: 1, step: 0.01, default: 0.3 },
    fatigue: { label: "fatigue d_3", min: 0, max: 1, step: 0.01, default: 0.2 },
    p:       { label: "convex exponent p", min: 1, max: 3, step: 0.05, default: 2 },
    w_d:     { label: "drive weight w_d",  min: 0, max: 1, step: 0.01, default: 0.15 },
    w_alive: { label: "w_alive",           min: 0, max: 2, step: 0.05, default: 1.0 },
  },
  slots: ["decomp", "marginal"],
  render: (host, params, slots) => {
    const drives = [params.hunger, params.thirst, params.fatigue];
    const { p, w_d, w_alive } = params;

    // Per-term reward decomposition. Sign convention: alive baseline
    // is positive, drive costs are negative (they reduce R).
    const driveCosts = drives.map((d) => w_d * Math.pow(d, p));
    const R = w_alive - driveCosts.reduce((s, x) => s + x, 0);

    // Long-form rows for a stacked horizontal-ish bar — we use a
    // simple bar chart with a "term" axis instead so each term is
    // visually distinct.
    const rows = [
      { term: "w_alive", value: w_alive, color: palette.primary },
      ...drives.map((d, i) => ({
        term: `−w_d·${DRIVE_NAMES[i]}^p`,
        value: -driveCosts[i],
        color: DRIVE_COLORS[i],
      })),
      { term: "R (total)", value: R, color: R >= 0 ? palette.warning : palette.danger },
    ];

    const yMax = Math.max(1.5, w_alive + 0.2, Math.abs(R) + 0.2);

    slots.decomp.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      marginLeft: 56,
      x: { label: "term", domain: rows.map((r) => r.term) },
      y: { label: "contribution to R", domain: [-yMax, yMax], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
        Plot.barY(rows, { x: "term", y: "value", fill: (d) => d.color, fillOpacity: 0.85 }),
        Plot.text(rows, {
          x: "term", y: "value",
          text: (d) => fmt(d.value),
          dy: (d) => (d.value >= 0 ? -6 : 12),
          fontSize: 10, fill: "#ddd",
        }),
      ],
    }));

    // Marginal cost curve c'(d) = p · d^{p-1}, with vertical rules
    // marking each drive's current level. This makes the convex
    // "marginal urgency rises" story (§16.1) visible.
    const xs = d3.range(101).map((i) => i / 100);
    const margCurve = xs.map((d) => ({ d, slope: p * Math.pow(d, Math.max(p - 1, 0)) }));
    const driveDots = drives.map((d, i) => ({
      d,
      slope: p * Math.pow(d, Math.max(p - 1, 0)),
      name: DRIVE_NAMES[i],
      color: DRIVE_COLORS[i],
    }));

    slots.marginal.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "drive level d", domain: [0, 1], grid: true },
      y: { label: "c'(d) = p·d^{p−1}", grid: true },
      marks: [
        Plot.line(margCurve, { x: "d", y: "slope", stroke: palette.muted, strokeWidth: 2 }),
        Plot.dot(driveDots, { x: "d", y: "slope", fill: (d) => d.color, r: 5 }),
        Plot.text(driveDots, {
          x: "d", y: "slope", text: "name",
          dx: 6, dy: -6, fontSize: 10, fill: (d) => d.color,
        }),
      ],
    }));

    // Identify the dominant drive (largest cost contribution).
    let dominantIdx = 0;
    for (let i = 1; i < drives.length; i++) {
      if (driveCosts[i] > driveCosts[dominantIdx]) dominantIdx = i;
    }
    const verdict = R >= w_alive * 0.8
      ? "comfortable"
      : R >= 0
        ? "stressed"
        : R >= -1
          ? "suffering"
          : "critical";

    slots.readout.innerHTML =
      `R = <strong>${fmt(R)}</strong> &nbsp;|&nbsp; ` +
      `Σ drive cost = ${fmt(driveCosts.reduce((s, x) => s + x, 0))} &nbsp;|&nbsp; ` +
      `dominant drive: <strong>${DRIVE_NAMES[dominantIdx]}</strong> &nbsp;|&nbsp; ` +
      `valence: <em>${verdict}</em>`;
  },
});

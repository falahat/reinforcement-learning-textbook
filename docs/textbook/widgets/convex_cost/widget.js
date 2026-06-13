// Widget 16.C — Convex-cost exponent comparator (Chapter 16).
//
// §16.1's "Why convex cost?" argument turned into a side-by-side
// curve viewer. The reader slides p ∈ [1, 3] and sees:
//
//   - c(d) = d^p              — the discomfort cost
//   - c'(d) = p · d^{p−1}     — the marginal urgency
//
// rendered as two overlaid curves on the same axes (cost in green,
// marginal in orange). At p = 1 marginal urgency is *constant* —
// the textbook's "linear cost ⇒ no alliesthesia" pathology. At
// p = 2 (Simulator default) the marginal slope grows linearly:
// at d = 0.1 the slope is 0.2; at d = 0.9 it is 1.8 — a 9× ratio
// just from the convex exponent. A second panel compares the
// per-tick reward contribution `w_d · d^p` at three reference
// drive levels (0.2, 0.5, 0.8) so the reader sees how p reshapes
// urgency across the drive range.
//
// Mount: in §16.1 of Chapter 16. Maps to Exercise 1.
//
//     <div id="ch16-convex-cost-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/convex_cost/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const REF_DRIVES = [0.2, 0.5, 0.8];

defineWidget({
  hostId: "ch16-convex-cost-widget",
  controls: {
    p:   { label: "exponent p", min: 1, max: 3, step: 0.05, default: 2.0 },
    w_d: { label: "w_d",        min: 0, max: 1, step: 0.01, default: 0.15 },
  },
  slots: ["curves", "bars"],
  render: (host, { p, w_d }, slots) => {
    // Two curves on a shared x-axis.
    const xs = d3.range(101).map((i) => i / 100);
    const rows = [];
    for (const d of xs) {
      rows.push({ d, value: Math.pow(d, p), series: "c(d) = d^p" });
      rows.push({
        d,
        value: p * Math.pow(d, Math.max(p - 1, 0)),
        series: "c'(d) = p·d^{p−1}",
      });
    }

    slots.curves.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "drive level d", domain: [0, 1], grid: true },
      y: { label: null, grid: true },
      color: {
        legend: true,
        domain: ["c(d) = d^p", "c'(d) = p·d^{p−1}"],
        range: [palette.primary, palette.warning],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([1], { stroke: palette.muted, ...dashed }),
        Plot.line(rows, { x: "d", y: "value", stroke: "series", strokeWidth: 2 }),
      ],
    }));

    // Reference-drive comparison: w_d · d^p at three drive levels.
    // Highlights how p changes the *spread* across drive levels.
    const refRows = REF_DRIVES.map((d) => ({
      d_label: `d = ${d.toFixed(1)}`,
      cost: w_d * Math.pow(d, p),
    }));
    const maxCost = Math.max(0.05, ...refRows.map((r) => r.cost));

    slots.bars.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 56,
      x: { label: "reference drive level", domain: refRows.map((r) => r.d_label) },
      y: { label: "w_d · d^p", domain: [0, maxCost * 1.2], grid: true },
      marks: [
        Plot.barY(refRows, { x: "d_label", y: "cost", fill: palette.accent, fillOpacity: 0.85 }),
        Plot.text(refRows, {
          x: "d_label", y: "cost",
          text: (r) => fmt(r.cost),
          dy: -6, fontSize: 10, fill: "#ddd",
        }),
      ],
    }));

    // Ratio readout: highest-to-lowest cost across the reference
    // drives quantifies how much p sharpens marginal urgency.
    const costs = refRows.map((r) => r.cost);
    const ratio = costs[0] > 0 ? costs[2] / costs[0] : Infinity;
    const linearNote = Math.abs(p - 1) < 1e-3
      ? " (linear — no alliesthesia)"
      : p > 1
        ? " (convex — marginal urgency rises)"
        : " (concave — marginal urgency falls)";
    slots.readout.innerHTML =
      `p = ${fmt(p)}${linearNote} &nbsp;|&nbsp; ` +
      `cost ratio c(0.8)/c(0.2) = ${fmt(ratio)} &nbsp;|&nbsp; ` +
      `marginal at d=0.5: c'(0.5) = ${fmt(p * Math.pow(0.5, Math.max(p - 1, 0)))}`;
  },
});

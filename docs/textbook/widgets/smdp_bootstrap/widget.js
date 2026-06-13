// Widget 14.C — SMDP-Q bootstrap discount illustrator (Chapter 14).
//
// Picks apart the SMDP-Q update visually:
//
//   Q(s, o) ← Q(s, o) + α [ r_cumulative + γ^τ max_o' Q(s', o') − Q(s, o) ]
//   r_cumulative = Σ_{k=0}^{τ-1} γ^k · r_{t+k+1}
//
// The reader picks an option duration τ, a per-tick reward shape
// (constant / linearly-increasing / sparse-at-end) and the magnitude
// of the bootstrap term. The widget draws a timeline of the τ
// primitive ticks; each bar's *height* is r_{t+k+1} and its *opacity*
// is the discount γ^k, so the actual contribution γ^k r_{t+k+1} is
// the visible area. A second strip shows the bootstrap γ^τ·V̂.
//
// Pedagogical payoff: see *which* primitive ticks contribute most.
// Under "sparse-at-end", the only nonzero reward is at k=τ-1, and the
// reader watches its γ^{τ-1} weight fade as τ grows — making the
// chapter's "30+ ticks needed" claim from §14.3 quantitative.
//
//     <div id="ch14-smdp-bootstrap-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/smdp_bootstrap/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

function rewardShape(shape, k, tau) {
  // r_{t+k+1} for k = 0..τ-1, with peak magnitude normalised to 1.
  if (shape === "constant") return 1.0;
  if (shape === "increasing") return (k + 1) / tau;
  if (shape === "sparse_end") return k === tau - 1 ? 1.0 : 0.0;
  if (shape === "sparse_start") return k === 0 ? 1.0 : 0.0;
  return 0;
}

defineWidget({
  hostId: "ch14-smdp-bootstrap-widget",
  controls: {
    shape: {
      type: "select",
      label: "reward shape r_{t+k+1}",
      options: [
        { value: "constant",     label: "constant (=1)" },
        { value: "increasing",   label: "linearly increasing" },
        { value: "sparse_end",   label: "sparse — at terminal step" },
        { value: "sparse_start", label: "sparse — at first step" },
      ],
      default: "constant",
    },
    tau:    { label: "τ (option duration, ticks)", min: 1, max: 50, step: 1, default: 12 },
    gamma:  { label: "γ (per-tick discount)", min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
    vNext:  { label: "max_o' Q(s', o') (bootstrap V̂)", min: 0, max: 10, step: 0.1, default: 5 },
  },
  slots: ["timeline", "decomp"],
  render: (host, p, slots) => {
    const tau = Math.round(p.tau);

    // Primitive reward strip + discounted contribution per tick.
    const rows = d3.range(tau).map((k) => {
      const r = rewardShape(p.shape, k, tau);
      const w = Math.pow(p.gamma, k);
      return { k, r, weight: w, contrib: w * r };
    });

    const rCum = rows.reduce((s, d) => s + d.contrib, 0);
    const gammaTau = Math.pow(p.gamma, tau);
    const bootstrap = gammaTau * p.vNext;
    const target = rCum + bootstrap;

    // Timeline: bars at each k. Height = r_{t+k+1}, opacity = γ^k.
    slots.timeline.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "primitive tick k inside the option", domain: [-0.5, tau - 0.5], grid: true },
      y: { label: "r_{t+k+1} · γ^k  (visible area = contribution)", domain: [0, 1.05], grid: true },
      marks: [
        // rectY + interval — barY on a continuous x renders zero-width.
        Plot.rectY(rows, {
          x: "k",
          y: "r",
          interval: 1,
          fill: palette.primary,
          fillOpacity: (d) => Math.max(0.05, d.weight),
        }),
        Plot.text(rows.filter((d) => d.contrib > 0.01), {
          x: "k", y: "r",
          text: (d) => fmt(d.contrib),
          textAnchor: "middle", dy: -4, fontSize: 9, fill: palette.muted,
        }),
        Plot.ruleY([0]),
      ],
    }));

    // Decomposition strip: r_cumulative vs γ^τ V̂ as a stacked bar,
    // plus a "target" line.
    const decomp = [
      { label: "Q(s,o) update", part: "r_cumulative",   value: rCum,       order: 0 },
      { label: "Q(s,o) update", part: "γ^τ · V̂(s')",   value: bootstrap,  order: 1 },
    ];
    slots.decomp.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 110,
      x: { label: "SMDP-Q target  =  r_cumulative  +  γ^τ · max_o' Q(s', o')", grid: true },
      y: { label: null },
      color: {
        domain: ["r_cumulative", "γ^τ · V̂(s')"],
        range: [palette.primary, palette.secondary],
        legend: true,
      },
      marks: [
        Plot.barX(decomp, {
          x: "value",
          y: "label",
          fill: "part",
          order: "order",
        }),
        Plot.text(decomp, {
          x: (d) => d.value / 2,
          y: "label",
          text: (d) => `${d.part}\n${fmt(d.value)}`,
          fill: "#fff",
          fontSize: 10,
        }),
        Plot.ruleX([0]),
      ],
    }));

    slots.readout.textContent =
      `r_cumulative = ${fmt(rCum)}  ·  γ^τ = ${fmt(gammaTau)}  ·  γ^τ·V̂ = ${fmt(bootstrap)}  ·  SMDP target = ${fmt(target)}`;
  },
});

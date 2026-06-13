// Widget 10.F — Causality-trick heatmap (Chapter 10).
//
// §10.2's claim: each action a_t at time t contributes
//
//   ∇_θ log π(a_t | s_t) · Σ_{k ≥ t} γ^{k-t} r_{k+1}
//
// to the policy gradient. The Σ has *no rewards from before t*: the
// upper-triangular structure is the substantive content of swapping
// G_0 → G_t (the "reward-to-go" form).
//
// We render a T × T grid where the cell (row t, col k) carries
//
//   c(t, k) =  γ^{k-t}   for k ≥ t   (the causal contribution)
//           =  0         for k < t   (literally zero — that's the trick)
//
// Toggle "with causality" off to show the *naive* form (G_0 multiplies
// every score), which makes every cell light up — illustrating how the
// causality trick zeroes out the lower triangle for free.
//
// Slider on γ shrinks the post-t tail (long-horizon credit dies) — a
// useful side-tie to Chapter 17's γ^k decay catastrophe.
//
// Pattern:
//
//     <div id="ch10-causality-trick-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/causality_trick/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, annotation } from "../shared/helpers.js";

const T = 10;

defineWidget({
  hostId: "ch10-causality-trick-widget",
  controls: {
    gamma:    { label: "γ (discount)", min: 0.5, max: 1.0, step: 0.01, default: 0.9 },
    causality: {
      type: "select",
      label: "estimator",
      options: [
        { value: "on",  label: "with causality (G_t)" },
        { value: "off", label: "naive (G_0 — non-causal)" },
      ],
      default: "on",
    },
  },
  render: (host, { gamma, causality }, slots) => {
    const useCausality = causality === "on";
    const cells = [];
    let causalMass = 0;
    let totalMass = 0;
    let zeroed = 0;
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < T; k++) {
        // Pre-action reward contribution we'd *like* the gradient to
        // count: γ^k from t=0 — the naive estimator counts it everywhere.
        const naive = Math.pow(gamma, k);
        const causal = k >= t ? Math.pow(gamma, k - t) : 0;
        const v = useCausality ? causal : naive;
        cells.push({ t, k, v, zeroed: useCausality && k < t });
        if (k >= t) causalMass += causal;
        totalMass += v;
        if (useCausality && k < t) zeroed++;
      }
    }
    const vMax = Math.max(...cells.map((d) => d.v));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 460, height: 360,
      marginLeft: 60, marginBottom: 50,
      x: {
        label: "k (reward time)",
        domain: Array.from({ length: T }, (_, i) => i),
        tickFormat: (d) => `r_${d + 1}`,
      },
      y: {
        label: "t (action time)  ↓",
        domain: Array.from({ length: T }, (_, i) => i),
        reverse: true,
        tickFormat: (d) => `a_${d}`,
      },
      color: {
        type: "linear", domain: [0, vMax],
        range: ["#1a1a2e", palette.primary],
        legend: true, label: "γ^(k−t)",
      },
      marks: [
        Plot.cell(cells, { x: "k", y: "t", fill: "v", inset: 0.6 }),
        // Highlight the zeroed-out lower triangle when causality is on.
        Plot.cell(
          cells.filter((d) => d.zeroed),
          { x: "k", y: "t", stroke: palette.danger, strokeOpacity: 0.4, fill: "transparent", inset: 0.6 },
        ),
        Plot.text(
          cells.filter((d) => d.zeroed),
          { x: "k", y: "t", text: () => "0", fill: palette.danger, fontSize: 10, ...annotation },
        ),
        Plot.text(
          cells.filter((d) => !d.zeroed && d.v >= 0.01),
          { x: "k", y: "t", text: (d) => d.v.toFixed(2),
            fill: "#fff", fontSize: 9 },
        ),
      ],
    }));

    const zeroPct = (100 * zeroed) / (T * T);
    const description = useCausality
      ? `lower-triangle zeroed (${zeroed} / ${T * T} cells = ${zeroPct.toFixed(0)}%)  ·  Σ causal mass = ${causalMass.toFixed(2)}`
      : `every cell active (naive G_0 form)  ·  Σ = ${totalMass.toFixed(2)}`;
    slots.readout.textContent =
      `T=${T} steps  ·  γ=${gamma.toFixed(2)}  ·  ${description}`;
  },
});

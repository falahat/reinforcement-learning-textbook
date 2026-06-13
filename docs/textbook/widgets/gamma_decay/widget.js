// Widget 17.A — γ^k decay catastrophe (Chapter 17).
//
// Demonstrates how the TD bootstrap term dies exponentially with
// horizon. For γ = 0.9, γ^500 ≈ 10⁻²³ — below single-precision float
// epsilon (~10⁻⁷), so credit from step 500 is literally
// unrepresentable. This is the central pathology Ch 17 motivates.
//
// Pattern: chapter markdown has just
//
//     <div id="ch17-gamma-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/gamma_decay/widget.js"></script>
//
// The shared `defineWidget` scaffold handles control-HTML, event
// wiring, and DOMContentLoaded mounting. ~25 LOC widget logic total.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

defineWidget({
  hostId: "ch17-gamma-widget",
  controls: {
    gamma: { label: "γ (gamma)", min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
    kmax:  { label: "k_max", min: 50, max: 1000, step: 10, default: 500 },
  },
  render: (host, { gamma, kmax }, slots) => {
    const data = d3.range(kmax + 1).map((k) => ({ k, y: Math.pow(gamma, k) }));
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "k (steps from terminal reward)" },
      y: { type: "log", domain: [1e-30, 1.5], label: "γ^k", grid: true },
      marks: [
        Plot.line(data, { x: "k", y: "y", stroke: palette.primary, strokeWidth: 2 }),
        Plot.ruleY([1e-7], { stroke: palette.danger, ...dashed }),
        Plot.text(
          [{ x: kmax * 0.95, y: 1e-7, label: "float32 ε ≈ 1e-7" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));
    const val = Math.pow(gamma, kmax);
    slots.readout.textContent =
      `γ^k_max = ${gamma.toFixed(3)}^${kmax} = ${val.toExponential(3)}`;
  },
});

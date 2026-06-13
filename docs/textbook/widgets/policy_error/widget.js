// Widget 1.M — Greedy-policy error amplifier (Chapter 1).
//
// The classic bound: if ‖V - V*‖_∞ = ε, then the value loss of the
// greedy policy derived from V is at most 2γε / (1-γ). The
// amplification factor 2γ / (1-γ) blows up as γ → 1 (≈100× at
// γ = 0.99). This is *why* high-γ regimes are so brutal: "close to V*"
// isn't close enough for the greedy policy.
//
// Pattern: chapter markdown has just
//
//     <div id="ch1-policy-error-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/policy_error/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

defineWidget({
  hostId: "ch1-policy-error-widget",
  controls: {
    epsilon: { label: "ε = ‖V - V*‖∞", min: 0, max: 1, step: 0.01, default: 0.1 },
    gamma:   { label: "γ (gamma)",      min: 0, max: 0.999, step: 0.001, default: 0.9 },
    yscale:  {
      type: "select",
      label: "y-axis",
      options: [
        { value: "linear", label: "linear" },
        { value: "log",    label: "log" },
      ],
      default: "linear",
    },
  },
  render: (host, { epsilon, gamma, yscale }, slots) => {
    // Amplification factor f(γ) = 2γ / (1-γ) over the γ domain.
    const xs = d3.range(0, 0.9991, 0.001);
    const data = xs.map((g) => ({ g, f: (2 * g) / (1 - g) }));
    const loss = (2 * gamma * epsilon) / (1 - gamma);

    // For log y, clamp the lower bound and skip the (γ=0, f=0) point.
    const yLo = yscale === "log" ? 1e-3 : 0;
    const yHi = (2 * 0.999) / (1 - 0.999); // ≈ 1998
    const plotData = yscale === "log" ? data.filter((d) => d.f > 0) : data;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "γ (discount factor)", domain: [0, 1] },
      y: {
        type: yscale,
        domain: [yLo, yHi],
        label: "amplification 2γ/(1-γ)",
        grid: true,
      },
      marks: [
        Plot.line(plotData, {
          x: "g", y: "f", stroke: palette.primary, strokeWidth: 2,
        }),
        // Horizontal reference at current ε (the input error magnitude).
        Plot.ruleY([Math.max(epsilon, yLo)], {
          stroke: palette.muted, ...dashed,
        }),
        Plot.text(
          [{ x: 0.02, y: Math.max(epsilon, yLo), label: `ε = ${epsilon.toFixed(2)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dy: -6,
            fill: palette.muted, ...annotation },
        ),
        // Marker at current (γ, 2γε/(1-γ)) — the policy-loss bound.
        Plot.dot(
          [{ g: gamma, y: Math.max(loss, yLo) }],
          { x: "g", y: "y", fill: palette.danger, r: 5 },
        ),
        Plot.text(
          [{ g: gamma, y: Math.max(loss, yLo),
             label: `γ=${gamma.toFixed(3)}` }],
          { x: "g", y: "y", text: "label", textAnchor: "end", dx: -8, dy: -4,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    const amp = (2 * gamma) / (1 - gamma);
    slots.readout.textContent =
      `2γε/(1-γ) = ${loss.toExponential(3)}   (amplification ${amp.toFixed(1)}×)`;
  },
});

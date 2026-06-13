// Widget 2.B — Discount factor explorer (Chapter 2).
//
// Visualises the geometric series Σγ^k and the "effective horizon"
// 1/(1-γ). Pedagogically: γ tunes how far the agent looks ahead.
// γ = 0.5 → horizon ~2 (myopic). γ = 0.99 → horizon ~100 (far-sighted).
//
// Pattern: chapter markdown contains a `<div id="ch2-discount-widget"
// class="textbook-widget">` mount point + a `<script type="module"
// src="./widgets/discount_factor/widget.js"></script>` reference.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import {
  readNumber,
  autoRender,
  setReadout,
  plotDefaults,
} from "../shared/helpers.js";

const HOST_ID = "ch2-discount-widget";
const K_MAX = 50;

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>γ (gamma)
      <input type="range" min="0.0" max="0.999" step="0.005"
             value="0.9" data-input="gamma">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot-weights></div>
  <div data-plot-horizon></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const gamma = readNumber(host, '[data-input="gamma"]');

    // Geometric weights γ^k for k = 0..K_MAX.
    const data = d3.range(K_MAX + 1).map((k) => ({
      k,
      y: Math.pow(gamma, k),
    }));

    // Effective horizon = 1/(1-γ). Σ_{k=0..∞} γ^k = 1/(1-γ).
    // Half-life k₁₂ such that γ^k = 0.5 → k = log(0.5)/log(γ).
    const horizon = gamma < 1 ? 1 / (1 - gamma) : Infinity;
    const halfLife =
      gamma > 0 && gamma < 1 ? Math.log(0.5) / Math.log(gamma) : Infinity;

    // Top plot — γ^k bars.
    host.querySelector("[data-plot-weights]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 220,
        x: { label: "k (steps ahead)", domain: [-0.5, K_MAX + 0.5] },
        y: { label: "γ^k", domain: [0, 1], grid: true },
        marks: [
          // Plot.barY needs an ordinal/band x-scale; on a continuous x
          // (which we want here for a 0..K_MAX axis with grid lines) it
          // renders zero-width bars and the chart looks empty. rectY with
          // interval:1 gives unit-width rects on the continuous scale.
          Plot.rectY(data, {
            x: "k",
            y: "y",
            interval: 1,
            fill: "#4caf50",
          }),
          // Half-life marker if it lands within the visible window.
          ...(halfLife <= K_MAX && isFinite(halfLife)
            ? [
                Plot.ruleX([halfLife], {
                  stroke: "#ffb74d",
                  strokeDasharray: "4 2",
                }),
                Plot.text(
                  [{ x: halfLife, y: 0.95, label: `k½ ≈ ${halfLife.toFixed(1)}` }],
                  {
                    x: "x",
                    y: "y",
                    text: "label",
                    fill: "#ffb74d",
                    textAnchor: "start",
                    dx: 4,
                    fontSize: 10,
                  }
                ),
              ]
            : []),
        ],
      })
    );

    // Bottom plot — horizon bar on a 0..K_MAX axis.
    const horizonClamped = Math.min(horizon, K_MAX);
    const horizonData = [
      { x1: 0, x2: horizonClamped, y: 0.5, label: "effective horizon" },
    ];
    host.querySelector("[data-plot-horizon]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 90,
        marginBottom: 28,
        x: { label: "k", domain: [0, K_MAX], grid: true },
        y: { axis: null, domain: [0, 1] },
        marks: [
          Plot.ruleY([0.5], { stroke: "#444" }),
          Plot.barX(horizonData, {
            x1: "x1",
            x2: "x2",
            y: "y",
            fill: "#42a5f5",
            fillOpacity: 0.7,
            insetTop: 18,
            insetBottom: 18,
          }),
          Plot.text(
            [
              {
                x: horizonClamped / 2,
                y: 0.5,
                label: isFinite(horizon)
                  ? `1/(1-γ) ≈ ${horizon.toFixed(1)}`
                  : "∞",
              },
            ],
            {
              x: "x",
              y: "y",
              text: "label",
              fill: "#fff",
              textAnchor: "middle",
              fontSize: 11,
            }
          ),
        ],
      })
    );

    const horizonStr = isFinite(horizon) ? horizon.toFixed(2) : "∞";
    const halfLifeStr = isFinite(halfLife) ? halfLife.toFixed(2) : "∞";
    setReadout(
      host,
      `γ = ${gamma.toFixed(3)}   ` +
        `effective horizon = 1/(1-γ) = ${horizonStr}   ` +
        `half-life k½ = ${halfLifeStr}   ` +
        `sum Σγ^k = ${horizonStr}`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

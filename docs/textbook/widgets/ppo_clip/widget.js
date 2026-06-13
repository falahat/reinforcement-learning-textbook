// Widget 11.A — PPO clipped surrogate objective (Chapter 11).
//
// Plots L^CLIP(r, A) = min(r·A, clip(r, 1-ε, 1+ε)·A) against the
// unclipped importance-weighted target r·A. Sliders for ε and the
// advantage A let the reader see why the clip is asymmetric: when
// A > 0 the clip flattens the right tail (over-shooting a *good*
// action), when A < 0 the clip flattens the left tail (over-shooting
// a *bad* action). The unshaded middle band is where unclipped PG
// applies; shading marks the regions where the clip is active for
// the current sign of A.
//
// Pattern: chapter markdown contains a `<div id="ch11-ppo-widget"
// class="textbook-widget">` mount point + a `<script type="module"
// src="./widgets/ppo_clip/widget.js"></script>` reference.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import {
  readNumber,
  autoRender,
  setReadout,
  plotDefaults,
} from "../shared/helpers.js";

const HOST_ID = "ch11-ppo-widget";

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>ε (epsilon)
      <input type="range" min="0.05" max="0.5" step="0.01"
             value="0.2" data-input="eps">
    </label>
    <label>A (advantage)
      <input type="range" min="-2" max="2" step="0.1"
             value="1.0" data-input="adv">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot></div>
`;

function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const eps = readNumber(host, '[data-input="eps"]');
    const adv = readNumber(host, '[data-input="adv"]');

    const rLo = 1 - eps;
    const rHi = 1 + eps;

    const rMin = 0;
    const rMax = 2.5;
    const N = 251;
    const data = d3.range(N).map((i) => {
      const r = rMin + (rMax - rMin) * (i / (N - 1));
      const unclipped = r * adv;
      const clipped = Math.min(r * adv, clip(r, rLo, rHi) * adv);
      return { r, unclipped, clipped };
    });

    // The clip is "active" (i.e. the min picks the clipped branch)
    // only on one side of the band, depending on the sign of A:
    //   A > 0: clip kicks in for r > 1 + eps  (caps gains above)
    //   A < 0: clip kicks in for r < 1 - eps  (caps gains below)
    // The other tail is left to the unclipped term — that's the
    // asymmetry the textbook discusses just above this widget.
    const shadeRegions = [];
    if (adv > 0) {
      shadeRegions.push({ x1: rHi, x2: rMax });
    } else if (adv < 0) {
      shadeRegions.push({ x1: rMin, x2: rLo });
    }

    host.querySelector("[data-plot]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        x: { label: "policy ratio r = π_θ / π_old", domain: [rMin, rMax] },
        y: { label: "objective", grid: true },
        marks: [
          Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
          Plot.ruleX([1], {
            stroke: "#888",
            strokeOpacity: 0.4,
            strokeDasharray: "2 3",
          }),
          // Shade the active-clip region.
          Plot.rect(shadeRegions, {
            x1: "x1",
            x2: "x2",
            y1: () => -1e9,
            y2: () => 1e9,
            fill: "#ffb74d",
            fillOpacity: 0.12,
          }),
          // Mark the clip thresholds.
          Plot.ruleX([rLo, rHi], {
            stroke: "#ffb74d",
            strokeOpacity: 0.6,
            strokeDasharray: "3 3",
          }),
          // Unclipped target r·A — dashed.
          Plot.line(data, {
            x: "r",
            y: "unclipped",
            stroke: "#90caf9",
            strokeWidth: 1.5,
            strokeDasharray: "5 3",
          }),
          // What PPO actually optimises — solid.
          Plot.line(data, {
            x: "r",
            y: "clipped",
            stroke: "#4caf50",
            strokeWidth: 2.2,
          }),
          Plot.text(
            [
              { x: rLo, y: 0, label: `1−ε = ${rLo.toFixed(2)}` },
              { x: rHi, y: 0, label: `1+ε = ${rHi.toFixed(2)}` },
            ],
            {
              x: "x",
              y: "y",
              text: "label",
              fill: "#ffb74d",
              textAnchor: "middle",
              dy: -6,
              fontSize: 10,
            }
          ),
        ],
      })
    );

    const sign = adv >= 0 ? "+" : "−";
    const activeSide =
      adv > 0
        ? `r > ${rHi.toFixed(2)}`
        : adv < 0
          ? `r < ${rLo.toFixed(2)}`
          : "none (A = 0)";
    setReadout(
      host,
      `at r=1: L = A = ${adv.toFixed(2)} · clip band [${rLo.toFixed(2)}, ${rHi.toFixed(2)}] · clip active when ${activeSide} (sign of A = ${sign})`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

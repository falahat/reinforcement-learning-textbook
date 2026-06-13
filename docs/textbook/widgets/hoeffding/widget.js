// Widget 1.F — Hoeffding sample-size calculator (Chapter 1).
//
// Hoeffding's inequality bounds the probability that the empirical
// mean of n bounded i.i.d. samples deviates from the true mean by
// more than ε:
//
//     P(|X̄_n - μ| ≥ ε) ≤ 2 exp(-2 n ε² / (b-a)²)
//
// Solving for n at a target failure probability δ gives the required
// sample size:
//
//     n ≥ (b-a)² · ln(2/δ) / (2 ε²)
//
// The widget plots the bound curve 2 exp(-2 n ε² / (b-a)²) vs n on a
// log-x axis, with the target δ as a horizontal rule and the required
// n as a vertical rule. Pedagogical hit: halve ε, watch n quadruple
// — the quadratic scaling Hoeffding gives you.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import {
  readNumber,
  autoRender,
  setReadout,
  plotDefaults,
} from "../shared/helpers.js";

const HOST_ID = "ch1-hoeffding-widget";

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>ε (epsilon, accuracy)
      <input type="range" min="0.01" max="0.5" step="0.01"
             value="0.1" data-input="epsilon">
    </label>
    <label>δ (delta, failure prob)
      <input type="range" min="0.001" max="0.1" step="0.001"
             value="0.05" data-input="delta">
    </label>
    <label>b − a (support width)
      <input type="range" min="0.5" max="5" step="0.1"
             value="1" data-input="width">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot></div>
`;

function bound(n, epsilon, width) {
  return 2 * Math.exp((-2 * n * epsilon * epsilon) / (width * width));
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const epsilon = readNumber(host, '[data-input="epsilon"]');
    const delta = readNumber(host, '[data-input="delta"]');
    const width = readNumber(host, '[data-input="width"]');

    // Smallest integer n satisfying the bound ≤ δ.
    const nReq = Math.ceil(
      (width * width * Math.log(2 / delta)) / (2 * epsilon * epsilon)
    );

    // Plot range: 1 .. ~10× the required n, log-spaced for a smooth curve.
    const nMax = Math.max(10, nReq * 10);
    const logMax = Math.log10(nMax);
    const data = d3.range(0, 1.0001, 0.005).map((t) => {
      const n = Math.pow(10, t * logMax);
      return { n, y: Math.min(1, bound(n, epsilon, width)) };
    });

    host.querySelector("[data-plot]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        x: {
          type: "log",
          label: "n (sample size)",
          domain: [1, nMax],
          grid: true,
        },
        y: {
          domain: [0, 1],
          label: "bound on P(|X̄ − μ| ≥ ε)",
          grid: true,
        },
        marks: [
          Plot.line(data, {
            x: "n",
            y: "y",
            stroke: "#4caf50",
            strokeWidth: 2,
          }),
          Plot.ruleY([delta], {
            stroke: "#e57373",
            strokeDasharray: "4 2",
          }),
          Plot.ruleX([nReq], {
            stroke: "#64b5f6",
            strokeDasharray: "4 2",
          }),
          Plot.text([{ x: nMax, y: delta, label: `δ = ${delta.toFixed(3)}` }], {
            x: "x",
            y: "y",
            text: "label",
            fill: "#e57373",
            textAnchor: "end",
            dy: -6,
            fontSize: 10,
          }),
          Plot.text([{ x: nReq, y: 0.95, label: `n = ${nReq}` }], {
            x: "x",
            y: "y",
            text: "label",
            fill: "#64b5f6",
            textAnchor: "start",
            dx: 6,
            fontSize: 10,
          }),
        ],
      })
    );

    setReadout(
      host,
      `required n = ⌈ (${width.toFixed(1)})² · ln(2/${delta.toFixed(3)}) / (2 · ${epsilon.toFixed(2)}²) ⌉ = ${nReq.toLocaleString()}`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

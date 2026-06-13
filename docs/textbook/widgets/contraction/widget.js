// Widget 1.E — Contraction mapping animator (Chapter 1).
//
// Visualises the Banach fixed-point theorem on the simplest non-trivial
// contraction on ℝ: T(x) = γ·x + b, with γ ∈ (0, 1). Iterating from x₀
// converges geometrically to x* = b / (1 - γ).
//
// Two plots:
//   1. Cobweb: y = T(x) overlaid with y = x; the staircase tracks
//      successive iterates x₀ → T(x₀) → T(T(x₀)) → … via alternating
//      vertical (apply T) and horizontal (reflect onto y = x) segments.
//   2. Log-error: |x_k − x*| vs k on a log y-axis. The slope is log γ;
//      a straight line is the visual signature of geometric convergence.
//
// Pattern: chapter markdown contains a `<div id="ch1-contraction-widget"
// class="textbook-widget">` mount point + a `<script type="module"
// src="./widgets/contraction/widget.js"></script>` reference.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import {
  readNumber,
  autoRender,
  setReadout,
  fmt,
  plotDefaults,
} from "../shared/helpers.js";

const HOST_ID = "ch1-contraction-widget";
const N_ITERS = 15;

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>γ (contraction factor)
      <input type="range" min="0.05" max="0.95" step="0.01"
             value="0.7" data-input="gamma">
    </label>
    <label>b (offset)
      <input type="range" min="-2" max="2" step="0.05"
             value="0.5" data-input="b">
    </label>
    <label>x₀ (start)
      <input type="range" min="-2" max="4" step="0.1"
             value="2.5" data-input="x0">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot-cobweb></div>
  <div data-plot-error></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const gamma = readNumber(host, '[data-input="gamma"]');
    const b = readNumber(host, '[data-input="b"]');
    const x0 = readNumber(host, '[data-input="x0"]');

    // Fixed point: T(x*) = x* ⇒ x* = b / (1 − γ).
    const xStar = b / (1 - gamma);

    // Iterate the contraction.
    const xs = [x0];
    for (let k = 0; k < N_ITERS; k++) {
      xs.push(gamma * xs[k] + b);
    }

    // --- Cobweb plot ---------------------------------------------------
    // Choose an x-range that comfortably contains x₀, x*, and a margin.
    const lo = Math.min(x0, xStar) - 1;
    const hi = Math.max(x0, xStar) + 1;
    const xDomain = [lo, hi];
    // Sample y = T(x) as a straight line (just two endpoints).
    const tLine = [
      { x: lo, y: gamma * lo + b },
      { x: hi, y: gamma * hi + b },
    ];
    // Reference y = x.
    const idLine = [
      { x: lo, y: lo },
      { x: hi, y: hi },
    ];
    // Build the cobweb staircase: start at (x₀, x₀) on the y=x diagonal,
    // then for each k go vertical to (xₖ, T(xₖ)) = (xₖ, x_{k+1}) and
    // horizontal to (x_{k+1}, x_{k+1}). Starting on the diagonal (rather
    // than at the artificial point (x₀, 0)) keeps every cobweb vertex
    // inside the chart's y-domain — otherwise the leading segment drops
    // from y=0 below the visible area up to T(x₀), which a reader sees as
    // a spurious "vertical line back down to zero" at the right edge.
    const cobweb = [{ x: x0, y: x0 }];
    for (let k = 0; k < N_ITERS; k++) {
      cobweb.push({ x: xs[k], y: xs[k + 1] });
      cobweb.push({ x: xs[k + 1], y: xs[k + 1] });
    }

    host.querySelector("[data-plot-cobweb]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 320,
        x: { domain: xDomain, label: "x", grid: true },
        y: { domain: xDomain, label: "T(x)", grid: true },
        marks: [
          // y = x reference (dashed grey).
          Plot.line(idLine, {
            x: "x",
            y: "y",
            stroke: "#888",
            strokeDasharray: "3 3",
          }),
          // y = T(x) (solid blue).
          Plot.line(tLine, {
            x: "x",
            y: "y",
            stroke: "#4a90e2",
            strokeWidth: 2,
          }),
          // Cobweb staircase (orange).
          Plot.line(cobweb, {
            x: "x",
            y: "y",
            stroke: "#ff9800",
            strokeWidth: 1.5,
          }),
          // Fixed point marker.
          Plot.dot([{ x: xStar, y: xStar }], {
            x: "x",
            y: "y",
            fill: "#e53935",
            r: 4,
          }),
          Plot.text([{ x: xStar, y: xStar, label: "x*" }], {
            x: "x",
            y: "y",
            text: "label",
            fill: "#e53935",
            dx: 8,
            dy: -8,
            fontSize: 11,
          }),
        ],
      })
    );

    // --- Log-error plot -----------------------------------------------
    // Floor errors at a tiny positive value so log-scale doesn't blow up
    // when an iterate lands exactly on x* (e.g. γ = 0 ⇒ one-step).
    const EPS = 1e-16;
    const errData = d3.range(N_ITERS + 1).map((k) => ({
      k,
      err: Math.max(Math.abs(xs[k] - xStar), EPS),
    }));

    host.querySelector("[data-plot-error]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 220,
        x: { label: "iteration k", grid: true },
        y: {
          type: "log",
          label: "|x_k − x*|",
          grid: true,
          domain: [Math.max(EPS, 1e-12), Math.max(errData[0].err, 1)],
        },
        marks: [
          Plot.line(errData, {
            x: "k",
            y: "err",
            stroke: "#4caf50",
            strokeWidth: 2,
          }),
          Plot.dot(errData, {
            x: "k",
            y: "err",
            fill: "#4caf50",
            r: 3,
          }),
        ],
      })
    );

    setReadout(
      host,
      `x* = b / (1 − γ) = ${fmt(xStar)}   ·   contraction γ = ${gamma.toFixed(2)}`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

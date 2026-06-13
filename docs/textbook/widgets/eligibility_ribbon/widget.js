// Widget 17.C — Eligibility-trace horizon ribbon (Chapter 17, §17.2).
//
// Visualises the joint γλ trace decay across a 100-state chain. The
// eligibility weight at lag t is (λγ)^t. The widget shades a ribbon
// over the chain coloured by this weight on a log scale, draws the
// noise-floor cut-off (single-precision ε ≈ 10⁻⁷) and reports the
// effective horizon 1/((1-γ)(1-λ)).
//
// Distinct from widget 17.A (single-parameter γ^k decay). Here both
// γ and λ are sliders; the *product* γλ governs the trace decay,
// and the central pedagogical point is that even generous (γ, λ)
// pairs only reach ~100 useful lags — well short of the L-suite's
// 500-tick credit assignment problem. That gap motivates §17.3+ as
// genuinely necessary, not a minor correction.
//
// Pattern: chapter markdown has
//
//     <div id="ch17-eligibility-ribbon-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/eligibility_ribbon/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import {
  plotDefaults,
  palette,
  dashed,
  annotation,
  fmt,
} from "../shared/helpers.js";

const CHAIN_LEN = 100;
const NOISE_FLOOR = 1e-7;

defineWidget({
  hostId: "ch17-eligibility-ribbon-widget",
  controls: {
    gamma:  { label: "γ (gamma)",  min: 0.5, max: 0.999, step: 0.005, default: 0.95 },
    lambda: { label: "λ (lambda)", min: 0,   max: 1,     step: 0.01,  default: 0.9 },
  },
  slots: ["ribbon", "curve"],
  render: (host, { gamma, lambda }, slots) => {
    const gl = gamma * lambda;

    // Trace weight at lag t = state-index from the current bump.
    const data = d3.range(CHAIN_LEN + 1).map((t) => ({
      t,
      w: Math.max(Math.pow(gl, t), 1e-30),
    }));

    // First lag where the trace crosses the noise floor — the
    // "useful-trace region" extends from lag 0 up to this t*.
    const tStar = data.findIndex((d) => d.w < NOISE_FLOOR);
    const usefulLags = tStar === -1 ? CHAIN_LEN : tStar;

    // Effective horizon from the textbook §17.2 closed form.
    const safe = gamma < 1 && lambda < 1;
    const effHorizon = safe ? 1 / ((1 - gamma) * (1 - lambda)) : Infinity;

    // ----- top: heatmap ribbon over the chain -----
    // Each state is one cell; fill encodes log10(w). Clamp at the
    // noise floor so the colour scale doesn't get dragged into the
    // subnormal tail.
    const cells = data.slice(1).map((d) => ({
      t: d.t,
      logw: Math.log10(Math.max(d.w, NOISE_FLOOR)),
    }));

    slots.ribbon.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 90,
      marginBottom: 30,
      marginTop: 24,
      x: { label: "lag t (states from current bump)", domain: [0.5, CHAIN_LEN + 0.5] },
      y: { axis: null, domain: [0, 1] },
      color: {
        type: "linear",
        scheme: "viridis",
        domain: [Math.log10(NOISE_FLOOR), 0],
        label: "log₁₀ (γλ)^t",
        legend: true,
      },
      marks: [
        // Plot.cell needs a band scale on x; using explicit Plot.rect
        // coordinates lets us keep the continuous x scale that the
        // axis ticks need.
        Plot.rect(cells, {
          x1: (d) => d.t - 0.5, x2: (d) => d.t + 0.5,
          y1: 0, y2: 1,
          fill: "logw",
        }),
      ],
    }));

    // ----- bottom: (γλ)^t curve with noise-floor rule -----
    const marks = [
      Plot.line(data, {
        x: "t", y: "w",
        stroke: palette.primary,
        strokeWidth: 2,
      }),
      Plot.ruleY([NOISE_FLOOR], { stroke: palette.danger, ...dashed }),
      Plot.text(
        [{ x: CHAIN_LEN * 0.95, y: NOISE_FLOOR, label: "noise floor ε ≈ 1e-7" }],
        { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
          fill: palette.danger, ...annotation },
      ),
    ];

    if (tStar > 0 && tStar < CHAIN_LEN) {
      marks.push(
        Plot.ruleX([tStar], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: tStar, y: NOISE_FLOOR, label: `useful up to t ≈ ${tStar}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start",
            dx: 6, dy: 12, fill: palette.warning, ...annotation },
        ),
      );
    }

    slots.curve.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "lag t", grid: true, domain: [0, CHAIN_LEN] },
      y: { type: "log", domain: [1e-30, 1.5], label: "(γλ)^t", grid: true },
      marks,
    }));

    slots.readout.textContent =
      `γλ = ${gl.toFixed(4)}; ` +
      `useful-trace lags ≈ ${usefulLags}; ` +
      `effective horizon 1/((1−γ)(1−λ)) = ${fmt(effHorizon)} steps`;
  },
});

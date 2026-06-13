// Widget 7.B — λ-return horizon mixer (Chapter 7).
//
// The λ-return G_t^λ = (1-λ) Σ_{n≥1} λ^(n-1) G_t^(n) is a weighted
// average of all n-step returns. The weights w_n = (1-λ)λ^(n-1) form a
// geometric distribution over horizons. This widget makes that mixture
// concrete:
//
//   - Top plot:    bar chart of w_n for n = 1..30.
//   - Bottom plot: cumulative mass Σ_{n≤N} w_n as a step curve, with
//                  vertical dashed rules at the 90% and 99% mass points.
//
// Slider for λ ∈ [0, 1]. Edge behaviour:
//   - λ = 0  → w_1 = 1, all others 0 (TD(0)).
//   - λ → 1  → mass spreads out; at n = 30 the cumulative is still
//              tiny, so 90%/99% markers vanish off the right edge (MC).
//
// Pattern: chapter markdown has
//
//     <div id="ch7-lambda-return-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/lambda_return/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N_MAX = 30;

defineWidget({
  hostId: "ch7-lambda-return-widget",
  controls: {
    lambda: { label: "λ (lambda)", min: 0, max: 1, step: 0.01, default: 0.9 },
  },
  slots: ["weights", "cumulative"],
  render: (host, { lambda }, slots) => {
    // Weights w_n = (1-λ)λ^(n-1) for n = 1..N_MAX. At λ = 1 every
    // finite-n weight is 0 in the limit; at λ = 0, w_1 = 1 and
    // 0^0 = 1 keeps the formula well-defined for n = 1.
    const ns = d3.range(1, N_MAX + 1);
    const weights = ns.map((n) => {
      if (lambda === 0) return n === 1 ? 1 : 0;
      return (1 - lambda) * Math.pow(lambda, n - 1);
    });
    const wRows = ns.map((n, i) => ({ n, w: weights[i] }));

    // Cumulative mass after n bars. With finite horizon N_MAX this
    // tops out at 1 − λ^N_MAX, not 1, when λ > 0.
    let acc = 0;
    const cumRows = ns.map((n, i) => {
      acc += weights[i];
      return { n, c: acc };
    });

    // First n that crosses 90% / 99%. May be undefined when λ is so
    // close to 1 that 30 bars don't carry that much mass.
    const find = (thresh) => {
      const row = cumRows.find((r) => r.c >= thresh);
      return row ? row.n : undefined;
    };
    const n90 = find(0.9);
    const n99 = find(0.99);

    // ------------- top plot: bar chart of weights -------------
    const wMax = Math.max(...weights, 1e-6);
    slots.weights.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 200,
        x: { label: "n (n-step horizon)", domain: [0.5, N_MAX + 0.5] },
        y: { label: "weight wₙ = (1−λ)λⁿ⁻¹", domain: [0, wMax * 1.1], grid: true },
        marks: [
          Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
          // rectY + interval — barY on a continuous x renders zero-width.
          Plot.rectY(wRows, {
            x: "n",
            y: "w",
            interval: 1,
            fill: palette.primary,
            fillOpacity: 0.85,
          }),
        ],
      })
    );

    // ------------- bottom plot: cumulative step curve -------------
    const cumMarks = [
      Plot.ruleY([0.9, 0.99], { stroke: palette.muted, strokeOpacity: 0.4, ...dashed }),
      Plot.line(cumRows, {
        x: "n",
        y: "c",
        stroke: palette.secondary,
        strokeWidth: 2,
        curve: "step-after",
      }),
      Plot.dot(cumRows, { x: "n", y: "c", fill: palette.secondary, r: 2 }),
    ];
    if (n90 !== undefined) {
      cumMarks.push(
        Plot.ruleX([n90], { stroke: palette.warning, ...dashed }),
        Plot.text([{ x: n90, y: 0.9, label: `90% @ n=${n90}` }], {
          x: "x",
          y: "y",
          text: "label",
          textAnchor: "start",
          dx: 4,
          dy: -4,
          fill: palette.warning,
          ...annotation,
        })
      );
    }
    if (n99 !== undefined) {
      cumMarks.push(
        Plot.ruleX([n99], { stroke: palette.danger, ...dashed }),
        Plot.text([{ x: n99, y: 0.99, label: `99% @ n=${n99}` }], {
          x: "x",
          y: "y",
          text: "label",
          textAnchor: "start",
          dx: 4,
          dy: -4,
          fill: palette.danger,
          ...annotation,
        })
      );
    }
    slots.cumulative.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 200,
        x: { label: "n", domain: [0.5, N_MAX + 0.5] },
        y: { label: "cumulative mass", domain: [0, 1.05], grid: true },
        marks: cumMarks,
      })
    );

    // ------------- readout -------------
    // Effective horizon 1/(1−λ) is the center of mass of a geometric
    // (1−λ) on {1, 2, …}: E[n] = 1/(1−λ). Diverges at λ = 1; cap
    // the displayed value so the readout stays readable.
    const horizon = lambda >= 1 ? Infinity : 1 / (1 - lambda);
    const fmtH = !isFinite(horizon)
      ? "∞"
      : horizon >= 1000
        ? horizon.toExponential(2)
        : horizon.toFixed(2);
    const fmtN = (n) => (n === undefined ? `>${N_MAX}` : `${n}`);
    slots.readout.textContent =
      `λ = ${lambda.toFixed(2)}  ·  ` +
      `90% mass by n = ${fmtN(n90)}  ·  ` +
      `99% mass by n = ${fmtN(n99)}  ·  ` +
      `effective horizon E[n] = 1/(1−λ) = ${fmtH}`;
  },
});

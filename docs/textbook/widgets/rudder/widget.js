// Widget 17.G — RUDDER return-decomposition timeline.
//
// A 30-step trajectory with reward = 1 at t = T-1, 0 elsewhere. RUDDER
// trains a model to predict the cumulative return from the prefix; the
// redistributed reward is the derivative of that predictor.
//
// Stylised RUDDER here: pretend the predictor is a saturating function
// of "credit-bearing actions so far". The trajectory has K of these
// actions placed at fixed positions; RUDDER assigns each one a fraction
// 1/K of the terminal reward, and the steps without credit get 0. A
// slider δ ∈ [0,1] interpolates between the sparse signal (δ=0) and
// fully-redistributed (δ=1).
//
// Mount:
//     <div id="ch17-rudder-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/rudder/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { unitBarsY } from "../shared/plot.js";

const T_STEPS = 30;
const CREDIT_STEPS = [4, 11, 19, 25]; // pretend these actions caused the terminal reward

function buildSeries(delta) {
  // Sparse: reward only at the final step.
  const sparse = new Array(T_STEPS).fill(0);
  sparse[T_STEPS - 1] = 1;

  // Redistributed: spread the 1.0 across the credit-bearing steps. Then
  // interpolate with δ.
  const dense = new Array(T_STEPS).fill(0);
  const share = 1 / CREDIT_STEPS.length;
  for (const idx of CREDIT_STEPS) dense[idx] = share;

  const out = new Array(T_STEPS);
  for (let t = 0; t < T_STEPS; t++) {
    out[t] = (1 - delta) * sparse[t] + delta * dense[t];
  }
  return { sparse, dense, mixed: out };
}

defineWidget({
  hostId: "ch17-rudder-widget",
  controls: {
    delta: { label: "δ (redistribution strength)", min: 0, max: 1, step: 0.01, default: 0.7 },
  },
  slots: ["main", "extra"],
  render: (host, { delta }, slots) => {
    const { sparse, mixed } = buildSeries(delta);

    const sparseData = sparse.map((r, t) => ({ t, r }));
    const mixedData = mixed.map((r, t) => ({ t, r }));

    const yMax = Math.max(1.05, ...mixedData.map((d) => d.r) , ...sparseData.map((d) => d.r)) * 1.05;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "step t", domain: [-0.5, T_STEPS - 0.5], grid: true },
      y: { label: "original r_t", domain: [0, yMax], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        unitBarsY(sparseData, { x: "t", y: "r", fill: palette.danger, fillOpacity: 0.85 }),
        Plot.text(
          [{ x: T_STEPS - 1, y: 1, label: "sparse: r = 1 only at T" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    slots.extra.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "step t", domain: [-0.5, T_STEPS - 0.5], grid: true },
      y: { label: "RUDDER r̂_t", domain: [0, yMax], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        unitBarsY(mixedData, { x: "t", y: "r", fill: palette.primary, fillOpacity: 0.85 }),
        // Mark the credit-bearing steps as small dashed verticals.
        ...CREDIT_STEPS.map((idx) =>
          Plot.ruleX([idx], { stroke: palette.warning, ...dashed, strokeOpacity: 0.5 }),
        ),
        Plot.text(
          [{ x: 0, y: yMax, label: `δ = ${delta.toFixed(2)} (0 = sparse, 1 = fully redistributed)` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dy: 10,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    const sumSparse = sparse.reduce((s, x) => s + x, 0);
    const sumMixed = mixed.reduce((s, x) => s + x, 0);
    slots.readout.textContent =
      `Σ r_t (original) = ${sumSparse.toFixed(2)}; ` +
      `Σ r̂_t (redistributed) = ${sumMixed.toFixed(2)} — ` +
      `same return, different shape. ` +
      `Credit-bearing steps: {${CREDIT_STEPS.join(", ")}}.`;
  },
});

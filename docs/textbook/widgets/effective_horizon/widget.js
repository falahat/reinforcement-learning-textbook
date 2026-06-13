// Widget 7.C — Effective-horizon calculator (γ, λ joint) — Chapter 7.
//
// The eligibility trace decays at rate γλ — not γ, not λ. This widget
// makes the joint arithmetic visible:
//
//   - half-life     = log(0.5) / log(γλ)
//   - 1%-life       = log(0.01) / log(γλ)
//   - eff. horizon  = 1 / (1 - γλ)
//   - L-suite probe = (γλ)^500
//
// Distinct from widget 17.A (single-parameter γ^k decay catastrophe).
// Here both γ and λ are sliders; the product γλ is what governs trace
// decay, so the joint surface is what matters pedagogically. Even at
// γ = λ = 0.99, γλ ≈ 0.98 and (γλ)^500 ≈ 4×10⁻⁵ — still unreachable for
// L-suite 500-tick chains. That is the §7.6 claim the widget proves.

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

const K_MAX = 1000;
const L_SUITE_HORIZON = 500;

defineWidget({
  hostId: "ch7-effective-horizon-widget",
  controls: {
    gamma:  { label: "γ (gamma)",  min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
    lambda: { label: "λ (lambda)", min: 0,   max: 1,     step: 0.01,  default: 0.9 },
  },
  render: (host, { gamma, lambda }, slots) => {
    const gl = gamma * lambda;

    // Derived quantities. Guard the degenerate cases (γλ = 0 or 1).
    const safe = gl > 0 && gl < 1;
    const halfLife = safe ? Math.log(0.5)  / Math.log(gl) : Infinity;
    const onePctLife = safe ? Math.log(0.01) / Math.log(gl) : Infinity;
    const effHorizon = gl < 1 ? 1 / (1 - gl) : Infinity;
    const lSuiteVal = Math.pow(gl, L_SUITE_HORIZON);

    // (γλ)^k curve. Clip below 1e-30 so log-y axis stays sane when γλ
    // is small and the tail collapses to subnormal floats.
    const data = d3.range(K_MAX + 1).map((k) => ({
      k,
      y: Math.max(Math.pow(gl, k), 1e-30),
    }));

    const marks = [
      Plot.line(data, {
        x: "k", y: "y",
        stroke: palette.primary,
        strokeWidth: 2,
      }),
      // y = 0.5 reference (half-life)
      Plot.ruleY([0.5], { stroke: palette.warning, ...dashed }),
      // y = 0.01 reference (1%-life)
      Plot.ruleY([0.01], { stroke: palette.danger, ...dashed }),
    ];

    if (safe && halfLife <= K_MAX) {
      marks.push(
        Plot.ruleX([halfLife], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: halfLife, y: 0.5, label: `half-life ≈ ${fmt(halfLife)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: -6,
            fill: palette.warning, ...annotation },
        ),
      );
    }
    if (safe && onePctLife <= K_MAX) {
      marks.push(
        Plot.ruleX([onePctLife], { stroke: palette.danger, ...dashed }),
        Plot.text(
          [{ x: onePctLife, y: 0.01, label: `1%-life ≈ ${fmt(onePctLife)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: -6,
            fill: palette.danger, ...annotation },
        ),
      );
    }

    // L-suite probe dot at k = 500.
    marks.push(
      Plot.dot([{ k: L_SUITE_HORIZON, y: Math.max(lSuiteVal, 1e-30) }], {
        x: "k", y: "y",
        fill: palette.secondary,
        r: 4,
      }),
      Plot.text(
        [{ x: L_SUITE_HORIZON, y: Math.max(lSuiteVal, 1e-30),
           label: `(γλ)^500 = ${lSuiteVal.toExponential(2)}` }],
        { x: "x", y: "y", text: "label", textAnchor: "start", dx: 8, dy: -6,
          fill: palette.secondary, ...annotation },
      ),
    );

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      x: { label: "k (steps from current trace bump)", grid: true },
      y: { type: "log", domain: [1e-30, 1.5], label: "(γλ)^k", grid: true },
      marks,
    }));

    slots.readout.textContent =
      `γλ = ${gl.toFixed(4)}; ` +
      `half-life = ${fmt(halfLife)} steps; ` +
      `1%-life = ${fmt(onePctLife)} steps; ` +
      `effective horizon 1/(1−γλ) = ${fmt(effHorizon)} steps; ` +
      `(γλ)^500 = ${lSuiteVal.toExponential(3)}`;
  },
});

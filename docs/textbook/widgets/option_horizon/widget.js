// Widget 14.B — Option duration → effective horizon (Chapter 14).
//
// Visualizes how wrapping primitive actions into τ-step options
// expands the effective horizon. If γ_p is the per-step discount,
// the option-level discount is γ_p^τ, the option-level effective
// horizon is 1 / (1 - γ_p^τ) option-steps, and in primitive steps
// that's τ / (1 - γ_p^τ). The ratio of option-horizon-in-primitive-
// steps to primitive horizon is the credit-assignment gain options
// give you (§14.9: "10^19× improvement").
//
// Two views: bar chart comparing primitive vs option horizons at the
// chosen (γ_p, τ), and a line plot sweeping τ to show how the gain
// scales.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import {
  readNumber,
  autoRender,
  setReadout,
  plotDefaults,
  fmt,
} from "../shared/helpers.js";

const HOST_ID = "ch14-option-horizon-widget";

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>γ_p (primitive gamma)
      <input type="range" min="0.5" max="0.999" step="0.005"
             value="0.9" data-input="gamma">
    </label>
    <label>τ (option duration, primitive steps)
      <input type="range" min="1" max="100" step="1"
             value="20" data-input="tau">
    </label>
    <span data-readout></span>
  </div>
  <div data-bars></div>
  <div data-sweep></div>
`;

function primitiveHorizon(gamma) {
  return 1 / (1 - gamma);
}

function optionHorizonPrimitiveSteps(gamma, tau) {
  return tau / (1 - Math.pow(gamma, tau));
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const gamma = readNumber(host, '[data-input="gamma"]');
    const tau = Math.round(readNumber(host, '[data-input="tau"]'));

    const Hp = primitiveHorizon(gamma);
    const Ho = optionHorizonPrimitiveSteps(gamma, tau);
    const ratio = Ho / Hp;

    const bars = [
      { label: "primitive (τ=1)", horizon: Hp, fill: "#4caf50" },
      { label: `option (τ=${tau})`, horizon: Ho, fill: "#42a5f5" },
    ];

    host.querySelector("[data-bars]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 220,
        x: { label: "effective horizon (primitive steps)", grid: true },
        y: { label: null },
        marks: [
          Plot.barX(bars, {
            x: "horizon",
            y: "label",
            fill: "fill",
          }),
          Plot.text(bars, {
            x: "horizon",
            y: "label",
            text: (d) => fmt(d.horizon),
            textAnchor: "start",
            dx: 4,
            fontSize: 11,
          }),
          Plot.ruleX([0]),
        ],
      })
    );

    const sweep = d3.range(1, 101).map((t) => ({
      tau: t,
      discount: Math.pow(gamma, t),
      horizon: optionHorizonPrimitiveSteps(gamma, t),
    }));

    host.querySelector("[data-sweep]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 240,
        x: { label: "τ (option duration, primitive steps)", grid: true },
        y: {
          type: "log",
          label: "option effective horizon (primitive steps)",
          grid: true,
        },
        marks: [
          Plot.line(sweep, {
            x: "tau",
            y: "horizon",
            stroke: "#42a5f5",
            strokeWidth: 2,
          }),
          Plot.ruleY([Hp], {
            stroke: "#4caf50",
            strokeDasharray: "4 2",
          }),
          Plot.text(
            [{ x: 95, y: Hp, label: `primitive horizon ≈ ${fmt(Hp)}` }],
            {
              x: "x",
              y: "y",
              text: "label",
              fill: "#4caf50",
              textAnchor: "end",
              dy: -6,
              fontSize: 10,
            }
          ),
          Plot.dot([{ tau, horizon: Ho }], {
            x: "tau",
            y: "horizon",
            fill: "#e57373",
            r: 5,
          }),
        ],
      })
    );

    setReadout(
      host,
      `primitive horizon = ${fmt(Hp)} steps; ` +
        `option (τ=${tau}) effective horizon = ${fmt(Ho)} primitive steps; ` +
        `ratio = ${fmt(ratio)}× (γ_p^τ = ${Math.pow(gamma, tau).toExponential(2)})`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

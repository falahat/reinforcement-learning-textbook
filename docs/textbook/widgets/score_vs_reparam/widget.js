// Widget 10.C — Score-function vs reparameterisation gradient (Chapter 10).
//
// Gaussian policy π(a; μ, σ) = N(μ, σ²) on a 1-D action with reward
// r(a) = −(a − 1)². Both estimators target the same gradient
//
//   ∇_μ E_π[r(a)] = E_π[ 2(a − 1) ]  (= 2(μ − 1) in closed form).
//
// We draw N = 400 samples and plot the *per-sample* gradient under each
// estimator, then compare their empirical variance.
//
// Score-function (REINFORCE-style):
//   g_SF = r(a) · ∇_μ log π(a; μ, σ) = r(a) · (a − μ)/σ²
//
// Reparameterised (pathwise):
//   a = μ + σ·ξ, ξ ~ N(0, 1)
//   g_RP = ∇_μ r(μ + σξ) = 2(μ + σξ − 1)
//
// The widget's central claim — exercise 6/7 — is that as σ → 0 the
// score-function variance *explodes* (a − μ blows up before r(a)
// shrinks), while the reparameterised gradient's variance shrinks
// quadratically with σ. Slider σ ∈ [0.1, 2.0]; readout reports both
// variances and their ratio.
//
// Pattern:
//
//     <div id="ch10-score-vs-reparam-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/score_vs_reparam/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N_SAMPLES = 400;

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gauss(rand) {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

defineWidget({
  hostId: "ch10-score-vs-reparam-widget",
  controls: {
    mu:    { label: "μ (policy mean)",  min: -1.0, max: 2.0, step: 0.05, default: 0.0 },
    sigma: { label: "σ (policy stddev)", min: 0.1, max: 2.0, step: 0.05, default: 0.5 },
  },
  slots: ["scatter"],
  render: (host, { mu, sigma }, slots) => {
    const trueGrad = 2 * (mu - 1); // ∇_μ E[r] in closed form

    const rand = rng(7);
    const data = [];
    let sumSF = 0, sumRP = 0;
    for (let i = 0; i < N_SAMPLES; i++) {
      const xi = gauss(rand);
      const a = mu + sigma * xi;
      const r = -((a - 1) ** 2);
      const gSF = r * (a - mu) / (sigma * sigma);
      const gRP = 2 * (a - 1);
      data.push({ idx: i, kind: "score-function", g: gSF });
      data.push({ idx: i, kind: "reparameterised", g: gRP });
      sumSF += gSF;
      sumRP += gRP;
    }

    const meanSF = sumSF / N_SAMPLES;
    const meanRP = sumRP / N_SAMPLES;
    let varSF = 0, varRP = 0;
    for (let i = 0; i < data.length; i++) {
      const d = data[i];
      if (d.kind === "score-function") varSF += (d.g - meanSF) ** 2;
      else varRP += (d.g - meanRP) ** 2;
    }
    varSF /= N_SAMPLES;
    varRP /= N_SAMPLES;

    // Pin x-axis to the widest envelope across estimators so visual
    // comparison reads cleanly when σ shrinks.
    const allG = data.map((d) => d.g);
    const xLim = Math.max(
      Math.abs(Math.min(...allG)),
      Math.abs(Math.max(...allG)),
    );

    slots.scatter.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "per-sample gradient ĝ",
           domain: [-xLim * 1.05, xLim * 1.05], grid: true },
      y: { label: "count (per estimator)", grid: true },
      color: {
        domain: ["score-function", "reparameterised"],
        range: [palette.warning, palette.secondary],
        legend: true,
      },
      marks: [
        Plot.rectY(
          data,
          Plot.binX(
            { y: "count" },
            { x: "g", fill: "kind", fillOpacity: 0.5, thresholds: 40, mixBlendMode: "normal" },
          ),
        ),
        Plot.ruleX([trueGrad], { stroke: palette.danger, strokeWidth: 1.5, ...dashed }),
        Plot.text(
          [{ x: trueGrad, y: 0, label: `true ∇J = ${trueGrad.toFixed(2)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4, dy: -6,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    const ratio = varRP > 0 ? varSF / varRP : Infinity;
    const ratioStr = isFinite(ratio) ? ratio.toFixed(1) : "∞";
    slots.readout.textContent =
      `Var(score-fn) = ${varSF.toFixed(3)}  ·  ` +
      `Var(reparam) = ${varRP.toFixed(3)}  ·  ` +
      `ratio SF/RP = ${ratioStr}×  ·  ` +
      `means: SF=${meanSF.toFixed(3)} · RP=${meanRP.toFixed(3)} (true=${trueGrad.toFixed(3)})`;
  },
});

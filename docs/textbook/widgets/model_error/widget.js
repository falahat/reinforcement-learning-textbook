// Widget 13.E — Compounding model-error visualizer (Chapter 13).
//
// A noisy linear dynamical system x_{t+1} = a · x_t + ε with
// per-step Gaussian noise N(0, σ_model²). The widget rolls the model
// forward k = 0..K and plots:
//
//   - true trajectory (deterministic a · x_t),
//   - learned-model envelope (mean ± 2σ band) under repeated rollouts,
//   - log-scale divergence ||x̂ - x||₂ vs. k.
//
// The reader sees the variance of the predicted state grow ~√k under
// additive Gaussian noise, and how the planner's belief drifts off
// the true state. A second readout shows
//
//     return_under_model - return_under_real
//
// for a fixed planner horizon H: the "looks great in sim, falls over
// in reality" gap.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-model-error-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/model_error/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const X0 = 1.0;       // initial state
const A_DYN = 0.95;   // true dynamics: x_{t+1} = a · x_t (stable)
const NUM_ROLLOUTS = 200;

defineWidget({
  hostId: "ch13-model-error-widget",
  controls: {
    sigma:   { label: "σ_model (per-step noise)", min: 0.0, max: 0.2, step: 0.005, default: 0.05 },
    kmax:    { label: "K (rollout depth)", min: 5, max: 100, step: 1, default: 50 },
    horizon: { label: "planner horizon H", min: 1, max: 50, step: 1, default: 20 },
  },
  slots: ["main", "divergence"],
  render: (host, { sigma, kmax, horizon }, slots) => {
    const rng = mulberry32(42);

    // True trajectory.
    const truth = [];
    {
      let x = X0;
      for (let k = 0; k <= kmax; k++) {
        truth.push({ k, x });
        x = A_DYN * x;
      }
    }

    // Monte-Carlo rollouts of the noisy learned model.
    const rollouts = [];
    for (let r = 0; r < NUM_ROLLOUTS; r++) {
      let x = X0;
      const traj = [{ k: 0, x }];
      for (let k = 1; k <= kmax; k++) {
        x = A_DYN * x + sigma * gauss(rng);
        traj.push({ k, x });
      }
      rollouts.push(traj);
    }

    // Per-step stats (mean, ±2σ) over rollouts.
    const bands = [];
    const divergence = [];
    for (let k = 0; k <= kmax; k++) {
      const vals = rollouts.map((tr) => tr[k].x);
      const mean = d3.mean(vals);
      const std = d3.deviation(vals) ?? 0;
      bands.push({ k, mean, lo: mean - 2 * std, hi: mean + 2 * std, std });
      // L1 divergence ||x̂ - x||
      const trueVal = truth[k].x;
      const mae = d3.mean(vals.map((v) => Math.abs(v - trueVal)));
      divergence.push({ k, mae: Math.max(mae, 1e-12), std: Math.max(std, 1e-12) });
    }

    // Sample of individual rollouts for the spaghetti plot.
    const sampleTraj = [];
    for (let r = 0; r < Math.min(20, NUM_ROLLOUTS); r++) {
      for (const pt of rollouts[r]) {
        sampleTraj.push({ k: pt.k, x: pt.x, run: r });
      }
    }

    // Planner-horizon "return gap":
    //   return_under_model = mean predicted x at H (planner believes this),
    //   return_under_real  = true x at H,
    //   gap = mean_predicted - true.
    const Hclamped = Math.min(horizon, kmax);
    const returnModel = bands[Hclamped].mean;
    const returnReal = truth[Hclamped].x;
    const gap = returnModel - returnReal;
    const stdAtH = bands[Hclamped].std;

    const allX = sampleTraj.map((d) => d.x)
      .concat(bands.map((b) => b.lo))
      .concat(bands.map((b) => b.hi));
    const xMin = d3.min(allX);
    const xMax = d3.max(allX);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "rollout step k", grid: true },
      y: { label: "state x", grid: true, domain: [xMin - 0.05, xMax + 0.05] },
      marks: [
        Plot.areaY(bands, {
          x: "k",
          y1: "lo",
          y2: "hi",
          fill: palette.secondary,
          fillOpacity: 0.2,
        }),
        Plot.line(sampleTraj, {
          x: "k",
          y: "x",
          z: "run",
          stroke: palette.secondary,
          strokeOpacity: 0.18,
          strokeWidth: 0.8,
        }),
        Plot.line(truth, {
          x: "k",
          y: "x",
          stroke: palette.primary,
          strokeWidth: 2,
        }),
        Plot.ruleX([Hclamped], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: Hclamped, y: xMax, label: `H = ${Hclamped}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4,
            fill: palette.warning, ...annotation },
        ),
      ],
    }));

    slots.divergence.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "rollout step k", grid: true },
      y: {
        type: "log",
        label: "||x̂ − x|| (log)",
        grid: true,
        domain: [1e-3, Math.max(1e-1, d3.max(divergence, (d) => d.mae) * 1.5)],
      },
      marks: [
        Plot.line(divergence, {
          x: "k",
          y: "mae",
          stroke: palette.danger,
          strokeWidth: 2,
        }),
        Plot.line(divergence, {
          x: "k",
          y: "std",
          stroke: palette.secondary,
          strokeWidth: 1.5,
          ...dashed,
        }),
        Plot.text(
          [
            { x: kmax, y: divergence[kmax].mae, label: "MAE", color: palette.danger },
            { x: kmax, y: divergence[kmax].std, label: "± std", color: palette.secondary },
          ],
          { x: "x", y: "y", text: "label", textAnchor: "end", dx: -4, dy: -6,
            fill: (d) => d.color, ...annotation },
        ),
      ],
    }));

    slots.readout.innerHTML =
      `σ_model = ${fmt(sigma)} · K = ${kmax} · H = ${Hclamped}<br>` +
      `<small>At step H: model believes x = ${fmt(returnModel)} ± ${fmt(2 * stdAtH)}; ` +
      `true x = ${fmt(returnReal)}; ` +
      `<strong style="color:${palette.danger}">gap = ${fmt(gap)}</strong>` +
      ` (rollout variance scales ≈ σ·√k)</small>`;
  },
});

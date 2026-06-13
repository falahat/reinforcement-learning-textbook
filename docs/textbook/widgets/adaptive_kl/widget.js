// Widget 11.D — Adaptive KL penalty schedule (Chapter 11).
//
// The KL-penalised variant of PPO (the predecessor to the clipped form)
// replaces the hard TRPO constraint with a soft penalty term and adapts
// the coefficient β on the fly:
//
//   L^KLPEN(θ) = E[r_t A_t]  −  β · KL(π_old ‖ π_θ)
//
// Heuristic schedule (Schulman et al. 2017, §4):
//   d = E[KL(π_old ‖ π_θ)]
//   if d < d_target / 1.5:   β ← β / 2     (too cautious — relax)
//   if d > d_target · 1.5:   β ← β · 2     (too aggressive — tighten)
//
// The widget simulates the loop: at each iteration, pick a KL value
// drawn from a distribution whose mean depends inversely on β (high β
// shrinks the actual step, low β lets it run), then update β by the
// rule above. The reader watches β and KL co-evolve into a band around
// d_target. Sliders tune d_target and the underlying step "boldness."
//
// Pedagogy: §11.4 mentions PPO-KL as the original PPO variant but the
// chapter focuses on the clip. The widget shows that the *other* PPO
// is a real control loop — β is a learned Lagrange multiplier.
//
// Pattern:
//
//     <div id="ch11-adaptive-kl-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/adaptive_kl/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { lcg, gauss } from "../shared/random.js";

// Simulate one rollout: given β, return the realised KL. Model: KL is
// roughly proportional to step-boldness² / β plus log-normal noise. The
// "true" optimal β for d_target is then determined by the boldness.
function rolloutKL(beta, boldness, rand) {
  // Deterministic part scales as boldness² / β (gradient step magnitude
  // squared, damped by penalty).
  const detKL = (boldness * boldness) / Math.max(beta, 1e-6);
  // Log-normal noise σ ≈ 0.25 — keeps things stochastic but bounded.
  const z = gauss(rand);
  const noise = Math.exp(0.25 * z);
  return detKL * noise;
}

function simulate({ dTarget, boldness, beta0, iters }) {
  const rand = lcg(42);
  let beta = beta0;
  const traj = [];
  for (let i = 0; i < iters; i++) {
    const d = rolloutKL(beta, boldness, rand);
    let action = "hold";
    if (d < dTarget / 1.5) {
      beta = beta / 2;
      action = "halve";
    } else if (d > dTarget * 1.5) {
      beta = beta * 2;
      action = "double";
    }
    traj.push({ iter: i, d, beta, action });
  }
  return traj;
}

defineWidget({
  hostId: "ch11-adaptive-kl-widget",
  controls: {
    dTarget:  { label: "d_target",        min: 0.001, max: 0.1, step: 0.001, default: 0.01 },
    boldness: { label: "step boldness",   min: 0.05,  max: 0.5, step: 0.01,  default: 0.2  },
    beta0:    { label: "β₀ (initial)",    min: 0.1,   max: 50,  step: 0.1,   default: 1.0  },
    iters:    { label: "iterations",      min: 20,    max: 200, step: 5,     default: 80   },
  },
  slots: ["kl", "beta"],
  render: (host, params, slots) => {
    const traj = simulate(params);
    const { dTarget } = params;

    // KL trajectory plot.
    const klMax = Math.max(dTarget * 3, ...traj.map((d) => d.d)) * 1.1;
    slots.kl.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "iteration", grid: true },
      y: {
        label: "realised KL", grid: true,
        type: "log",
        domain: [Math.max(1e-5, dTarget / 100), klMax],
      },
      marks: [
        // Target band [d_target/1.5, d_target·1.5].
        Plot.ruleY([dTarget], { stroke: palette.warning, strokeWidth: 1.5 }),
        Plot.ruleY([dTarget / 1.5, dTarget * 1.5], {
          stroke: palette.warning, ...dashed, strokeOpacity: 0.6,
        }),
        Plot.line(traj, {
          x: "iter", y: "d", stroke: palette.primary, strokeWidth: 1.5,
        }),
        Plot.dot(traj, {
          x: "iter", y: "d", r: 2.5,
          fill: (d) =>
            d.action === "double" ? palette.danger :
            d.action === "halve" ? palette.secondary :
            palette.muted,
          stroke: "white", strokeWidth: 0.5,
        }),
        Plot.text(
          [{ x: traj.length - 1, y: dTarget, label: `d_target = ${dTarget}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.warning, ...annotation },
        ),
      ],
    }));

    // β trajectory plot (log-y because β doubles/halves).
    const betaMax = Math.max(...traj.map((d) => d.beta));
    const betaMin = Math.min(...traj.map((d) => d.beta));
    slots.beta.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "iteration", grid: true },
      y: {
        label: "penalty β", grid: true,
        type: "log",
        domain: [Math.max(betaMin / 2, 1e-3), betaMax * 2],
      },
      marks: [
        Plot.line(traj, {
          x: "iter", y: "beta", stroke: palette.accent, strokeWidth: 2,
        }),
        Plot.dot(traj, {
          x: "iter", y: "beta", r: 2, fill: palette.accent,
        }),
      ],
    }));

    const doubled = traj.filter((d) => d.action === "double").length;
    const halved  = traj.filter((d) => d.action === "halve").length;
    const finalBeta = traj[traj.length - 1].beta;
    const meanD = traj.reduce((s, d) => s + d.d, 0) / traj.length;
    slots.readout.textContent =
      `final β = ${finalBeta.toExponential(2)}   |   mean realised KL = ${meanD.toExponential(2)}   |   ` +
      `doublings: ${doubled}   halvings: ${halved}`;
  },
});

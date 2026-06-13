// Widget 10.E — GAE(λ) bias-variance tradeoff (Chapter 10).
//
// On a synthetic 20-step MDP with deterministic state values V* known
// in closed form, we plant a biased value estimate V_hat(s) = V*(s) + b
// (uniform bias b, also a slider) and draw stochastic rewards
// r_t = V*(s_t) − γ·V*(s_{t+1}) + ε_t, ε_t ~ N(0, σ²). Under this set-up
// the GAE estimator
//
//   Â_t^GAE(λ) = Σ_{l ≥ 0} (γλ)^l δ_{t+l},   δ_t = r_{t+1} + γ V̂(s_{t+1}) − V̂(s_t)
//
// has *closed-form* expected value (so bias is computable) and we
// estimate its variance from M independent trajectory rollouts.
//
// For each λ in a fixed grid we plot (bias², var) as a scatter point.
// The classic Pareto curve — pure TD (λ=0) low-variance/biased, pure
// MC (λ=1) zero-bias/high-variance, λ≈0.95 at the elbow — emerges
// directly.
//
// Pattern:
//
//     <div id="ch10-gae-lambda-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/gae_lambda/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const T = 20;                       // trajectory length
const M = 200;                      // rollouts per λ
const LAMBDAS = [0, 0.25, 0.5, 0.7, 0.85, 0.92, 0.95, 0.97, 0.99, 1.0];

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

// True state values for our toy chain: V*(s_t) = T − t (linear walk
// toward terminal). Reproducible across renders because the chain is
// deterministic in t.
function vStar(t) {
  return T - t;
}

// One rollout: T steps. Returns the per-step Â_t^GAE(λ) at t = 0 (the
// quantity the chapter writes ∇log π(a_0|s_0) · Â_0 against).
function gaeAtZero(lambda, gamma, bias, noise, rand) {
  // Per-step TD errors δ_t = r_{t+1} + γ V̂(s_{t+1}) − V̂(s_t),
  // with V̂(s) = V*(s) + bias. The bias cancels for non-terminal
  // transitions and contributes only at the boundary.
  let A = 0;
  let weight = 1;
  for (let t = 0; t < T; t++) {
    const eps = noise * gauss(rand);
    const vNext = t + 1 >= T ? 0 : vStar(t + 1);   // terminal V*=0
    const r = vStar(t) - gamma * vNext + eps;       // rewards consistent with V*
    const VhatS = vStar(t) + bias;
    const VhatNext = (t + 1 >= T ? 0 : vStar(t + 1)) + (t + 1 >= T ? 0 : bias);
    const delta = r + gamma * VhatNext - VhatS;
    A += weight * delta;
    weight *= gamma * lambda;
    if (weight < 1e-12) break;
  }
  return A;
}

defineWidget({
  hostId: "ch10-gae-lambda-widget",
  controls: {
    gamma: { label: "γ (discount)",     min: 0.5,  max: 0.999, step: 0.005, default: 0.95 },
    bias:  { label: "V̂ bias (uniform offset)", min: -2.0, max: 2.0,  step: 0.05,  default: 0.5 },
    noise: { label: "reward noise σ",   min: 0.0,  max: 2.0,   step: 0.05,  default: 0.5 },
  },
  render: (host, { gamma, bias, noise }, slots) => {
    // True advantage at t=0 with this MDP is exactly 0 because the
    // rewards were *constructed* to make the policy optimal (V̂ = V*
    // satisfies Bellman). So bias of Â_t^GAE w.r.t. zero is just the
    // mean of our sample estimates. (No baseline-vs-MC asymmetry to
    // distract from the GAE math.)
    const A_star = 0;
    const rand = rng(123);
    const stats = LAMBDAS.map((lambda) => {
      let sum = 0, sumSq = 0;
      for (let i = 0; i < M; i++) {
        const a = gaeAtZero(lambda, gamma, bias, noise, rand);
        sum += a;
        sumSq += a * a;
      }
      const mean = sum / M;
      const variance = Math.max(0, sumSq / M - mean * mean);
      const biasSq = (mean - A_star) ** 2;
      return { lambda, mean, variance, biasSq, label: lambda.toFixed(2) };
    });

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "bias² of Â (lower ← better)", grid: true, type: "linear" },
      y: { label: "Var(Â) (lower ← better)",      grid: true, type: "linear" },
      marks: [
        Plot.line(stats, {
          x: "biasSq", y: "variance",
          stroke: palette.muted, strokeOpacity: 0.4, ...dashed,
        }),
        Plot.dot(stats, {
          x: "biasSq", y: "variance",
          fill: palette.secondary, r: 5, stroke: "#fff", strokeWidth: 1,
        }),
        Plot.text(stats, {
          x: "biasSq", y: "variance",
          text: (d) => `λ=${d.label}`,
          dy: -10, fill: palette.muted, fontSize: 10,
        }),
        // PPO's default of 0.95 — the "common elbow" callout.
        Plot.dot(
          stats.filter((d) => Math.abs(d.lambda - 0.95) < 1e-6),
          { x: "biasSq", y: "variance", fill: palette.warning, r: 7, stroke: "#fff", strokeWidth: 2 },
        ),
        Plot.text(
          stats.filter((d) => Math.abs(d.lambda - 0.95) < 1e-6),
          { x: "biasSq", y: "variance", text: () => "PPO default",
            dy: 14, fill: palette.warning, ...annotation },
        ),
      ],
    }));

    const td0 = stats[0];
    const mc1 = stats[stats.length - 1];
    const elbow = stats.find((s) => Math.abs(s.lambda - 0.95) < 1e-6);
    slots.readout.textContent =
      `λ=0 (TD): bias²=${td0.biasSq.toFixed(3)}, var=${td0.variance.toFixed(3)}  ·  ` +
      `λ=0.95: bias²=${elbow.biasSq.toFixed(3)}, var=${elbow.variance.toFixed(3)}  ·  ` +
      `λ=1 (MC): bias²=${mc1.biasSq.toFixed(3)}, var=${mc1.variance.toFixed(3)}`;
  },
});

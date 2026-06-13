// Widget 1.J — Robbins-Monro step-size race (Chapter 1, §1.5).
//
// Three SGD runs in parallel on f(θ) = ½(θ − μ)² with noisy gradient
// Y_k = (θ_k − μ) + σ·ξ_k, ξ_k ~ N(0,1):
//
//   Run A (1/k):     α_k = c / k       — Robbins-Monro compliant
//                    (Σα = ∞, Σα² < ∞).  Hugs μ asymptotically.
//   Run B (1/√k):    α_k = c / √k      — Σα = ∞ but Σα² = ∞.
//                    Compliant for some refined results, faster early,
//                    larger asymptotic jitter than 1/k.
//   Run C (const):   α_k = c           — Σα² = ∞. Bounces forever in
//                    a band of radius O(σ√α/2) around μ.
//
// We plot all three θ_k vs k. The constant-α run literally never lands;
// the 1/k run hugs the dashed μ-reference. That is the section's
// punchline made geometric.
//
// Mount: `<div id="ch1-robbins-monro-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/robbins_monro/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const N_STEPS = 500;

// Deterministic Box-Muller PRNG so the race is reproducible across
// re-renders. seed mutates a Uint32Array; very small and good enough.
function mkRng(seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  return () => {
    // Box-Muller: two uniforms → one standard normal.
    const u1 = Math.max(next(), 1e-12);
    const u2 = next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

// Simulate θ_{k+1} = θ_k − α_k · ((θ_k − μ) + σ·ξ_k).
function runSchedule(scheduleFn, c, mu, sigma, theta0, seed) {
  const xi = mkRng(seed);
  let theta = theta0;
  const traj = [{ k: 0, theta }];
  for (let k = 1; k <= N_STEPS; k++) {
    const alpha = scheduleFn(c, k);
    const grad = (theta - mu) + sigma * xi();
    theta = theta - alpha * grad;
    traj.push({ k, theta });
  }
  return traj;
}

defineWidget({
  hostId: "ch1-robbins-monro-widget",
  controls: {
    c:     { label: "c",     min: 0.05, max: 2, step: 0.01, default: 0.5 },
    sigma: { label: "σ",     min: 0,    max: 3, step: 0.05, default: 1.0 },
    mu:    { label: "μ",     min: -3,   max: 3, step: 0.1,  default: 0 },
    theta0:{ label: "θ₀",    min: -5,   max: 5, step: 0.1,  default: 4 },
    seed:  { label: "seed",  min: 1,    max: 99, step: 1,   default: 7 },
  },
  render: (host, { c, sigma, mu, theta0, seed }, slots) => {
    const trajA = runSchedule((c, k) => c / k,            c, mu, sigma, theta0, seed);
    const trajB = runSchedule((c, k) => c / Math.sqrt(k), c, mu, sigma, theta0, seed);
    const trajC = runSchedule((c, _k) => c,               c, mu, sigma, theta0, seed);

    // Tag each row with its schedule for Plot's color scale.
    const allData = [
      ...trajA.map((d) => ({ ...d, kind: "α=c/k" })),
      ...trajB.map((d) => ({ ...d, kind: "α=c/√k" })),
      ...trajC.map((d) => ({ ...d, kind: "α=c (const)" })),
    ];

    // Theoretical noise band for constant α: stationary variance is
    // approximately σ²·α / (2 − α). Use that to draw a shaded band around μ.
    const alpha = c;
    const bandRadius = alpha < 2
      ? sigma * Math.sqrt(alpha / Math.max(2 - alpha, 1e-6))
      : NaN;
    const bandData = Number.isFinite(bandRadius)
      ? d3.range(N_STEPS + 1).map((k) => ({ k, lo: mu - bandRadius, hi: mu + bandRadius }))
      : [];

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      width: 640,
      x: { label: "step k", grid: true },
      y: { label: "θ_k", grid: true },
      color: {
        domain: ["α=c/k", "α=c/√k", "α=c (const)"],
        range: [palette.primary, palette.secondary, palette.danger],
        legend: true,
      },
      marks: [
        // Noise band for constant-α run.
        ...(bandData.length ? [Plot.areaY(bandData, {
          x: "k", y1: "lo", y2: "hi", fill: palette.danger, fillOpacity: 0.08,
        })] : []),
        Plot.ruleY([mu], { stroke: palette.muted, ...dashed }),
        Plot.text(
          [{ x: N_STEPS, y: mu, label: `μ = ${fmt(mu)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.muted, ...annotation },
        ),
        Plot.line(allData, { x: "k", y: "theta", stroke: "kind", strokeWidth: 1.2, strokeOpacity: 0.9 }),
      ],
    }));

    // Tail-average θ_k for the last 100 steps — closer to μ = better.
    const tailMean = (t) => {
      const tail = t.slice(-100);
      return tail.reduce((s, d) => s + d.theta, 0) / tail.length;
    };
    const mA = tailMean(trajA), mB = tailMean(trajB), mC = tailMean(trajC);
    const bandStr = Number.isFinite(bandRadius)
      ? `±${fmt(bandRadius)}` : "blow-up (α≥2)";
    slots.readout.innerHTML =
      `<small>tail-avg (last 100): ` +
      `<span style='color:${palette.primary}'>1/k=${fmt(mA)}</span>, ` +
      `<span style='color:${palette.secondary}'>1/√k=${fmt(mB)}</span>, ` +
      `<span style='color:${palette.danger}'>const=${fmt(mC)}</span>. ` +
      `Constant-α noise band ≈ ${bandStr} (red shading).</small>`;
  },
});

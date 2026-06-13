// Widget 11.E — V-trace off-policy correction (IMPALA) (Chapter 11).
//
// IMPALA's V-trace target is an off-policy value estimator that uses
// truncated importance-sampling ratios:
//
//   v_s = V(s_t) + Σ_{k=t..} γ^(k−t) · ( Π_{i=t..k−1} c_i ) · ρ_k · δ_k
//
// where ρ_k = min(ρ̄, π(a_k|s_k)/μ(a_k|s_k))   (capped at ρ̄)
//       c_i = min(c̄,  π(a_i|s_i)/μ(a_i|s_i))   (capped at c̄)
//       δ_k = r_k + γ V(s_{k+1}) − V(s_k)
//
// Two truncations, two roles. ρ̄ controls the *fixed point* (large ρ̄
// → target closer to π's value function, small ρ̄ → biased toward μ).
// c̄ controls the *variance* of multi-step credit propagation (the
// product of c's would explode otherwise).
//
// This widget compares three estimators of E[(Π ρ)] over a k-step trace:
//   (a) uncorrected (no IS — biased toward μ),
//   (b) raw importance sampling (Π ρ — unbiased but huge variance),
//   (c) V-trace truncated (Π min(c̄, ρ) — biased but bounded variance).
//
// Sliders for ρ_mean (the average π/μ ratio) and c̄ (the trace cap)
// show the variance of (b) blowing up exponentially with horizon while
// (c) stays finite. The reader sees why IMPALA can train on stale data.
//
// Pedagogy: IMPALA is mentioned obliquely as "off-policy actor-critic";
// the widget makes the V-trace clip mechanically concrete.
//
// Pattern:
//
//     <div id="ch11-vtrace-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/vtrace_correction/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { lcg, gauss } from "../shared/random.js";

// Sample a single ratio. Mixture: with prob 0.9 a moderate ratio drawn
// from log-normal(μ = log(ρ_mean), σ²); with prob 0.1 an "outlier" ratio
// 10× larger — this is what makes raw IS so variance-y.
function sampleRho(rand, rhoMean) {
  const z = gauss(rand);
  const r = rhoMean * Math.exp(0.6 * z - 0.18);
  if (rand() < 0.1) return r * 10;
  return r;
}

// Estimate variance of Π_{k=1..H} ρ_k by Monte Carlo over `trials`
// rollouts. Returns mean and stddev. Truncated version uses min(c̄, ρ).
function traceStats(rhoMean, cBar, H, trials) {
  const rand = lcg(123);
  const productsRaw = [];
  const productsTrunc = [];
  for (let t = 0; t < trials; t++) {
    let pRaw = 1, pTrunc = 1;
    for (let k = 0; k < H; k++) {
      const r = sampleRho(rand, rhoMean);
      pRaw *= r;
      pTrunc *= Math.min(cBar, r);
    }
    productsRaw.push(pRaw);
    productsTrunc.push(pTrunc);
  }
  const mean = (arr) => arr.reduce((s, x) => s + x, 0) / arr.length;
  const variance = (arr, m) =>
    arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length;
  const mRaw = mean(productsRaw);
  const mTrunc = mean(productsTrunc);
  return {
    raw:    { mean: mRaw,    std: Math.sqrt(variance(productsRaw,   mRaw)) },
    trunc:  { mean: mTrunc,  std: Math.sqrt(variance(productsTrunc, mTrunc)) },
  };
}

defineWidget({
  hostId: "ch11-vtrace-widget",
  controls: {
    rhoMean: { label: "ρ̄ = mean π/μ ratio", min: 0.5, max: 3.0, step: 0.05, default: 1.2 },
    cBar:    { label: "c̄ (V-trace cap)",     min: 0.5, max: 3.0, step: 0.05, default: 1.0 },
    Hmax:    { label: "max horizon H",        min: 5,   max: 40,  step: 1,    default: 25  },
    trials:  { label: "Monte Carlo trials",   min: 200, max: 5000, step: 100, default: 2000 },
  },
  slots: ["variance", "samples"],
  render: (host, { rhoMean, cBar, Hmax, trials }, slots) => {
    // Sweep horizon and plot stddev of the cumulative product.
    const horizons = d3.range(1, Hmax + 1);
    const stats = horizons.map((H) => {
      const s = traceStats(rhoMean, cBar, H, trials);
      return {
        H,
        rawMean:   s.raw.mean,
        rawStd:    s.raw.std,
        truncMean: s.trunc.mean,
        truncStd:  s.trunc.std,
      };
    });

    // Long-form rows so Plot can colour by estimator.
    const rows = [];
    for (const s of stats) {
      rows.push({ H: s.H, est: "raw IS  Π ρ",          std: s.rawStd,   mean: s.rawMean });
      rows.push({ H: s.H, est: "V-trace  Π min(c̄, ρ)", std: s.truncStd, mean: s.truncMean });
    }
    const allStd = rows.map((r) => r.std).filter((v) => v > 0);
    const stdMin = Math.max(1e-3, Math.min(...allStd) * 0.5);
    const stdMax = Math.max(...allStd) * 1.5;

    slots.variance.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      x: { label: "horizon H (steps)", grid: true },
      y: {
        label: "stddev of cumulative product",
        type: "log",
        domain: [stdMin, stdMax],
        grid: true,
      },
      color: {
        domain: ["raw IS  Π ρ", "V-trace  Π min(c̄, ρ)"],
        range: [palette.danger, palette.primary],
        legend: true,
      },
      marks: [
        Plot.line(rows, {
          x: "H", y: "std", stroke: "est", strokeWidth: 2,
        }),
        Plot.dot(rows, {
          x: "H", y: "std", fill: "est", r: 2.5,
        }),
        Plot.ruleY([1], { stroke: palette.muted, ...dashed }),
        Plot.text(
          [{ x: Hmax, y: 1, label: "variance = 1" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -5,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    // Sample histogram at the largest horizon — show the per-trial
    // distribution side-by-side. Truncated truncates the right tail.
    const rand = lcg(123);
    const H = Hmax;
    const samples = [];
    for (let t = 0; t < Math.min(trials, 1500); t++) {
      let pRaw = 1, pTrunc = 1;
      for (let k = 0; k < H; k++) {
        const r = sampleRho(rand, rhoMean);
        pRaw *= r;
        pTrunc *= Math.min(cBar, r);
      }
      // Clip to avoid Plot rendering issues with huge outliers.
      samples.push({ v: Math.min(pRaw, 1e6), est: "raw IS  Π ρ" });
      samples.push({ v: Math.min(pTrunc, 1e6), est: "V-trace  Π min(c̄, ρ)" });
    }

    slots.samples.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      marginLeft: 50,
      x: {
        label: `product Π ρ over H = ${Hmax} steps  (log scale)`,
        type: "log",
        grid: true,
        domain: [1e-4, 1e4],
      },
      y: { label: "samples (count)", grid: true },
      color: {
        domain: ["raw IS  Π ρ", "V-trace  Π min(c̄, ρ)"],
        range: [palette.danger, palette.primary],
      },
      marks: [
        Plot.rectY(
          samples.filter((s) => s.v > 0 && isFinite(s.v)),
          Plot.binX(
            { y: "count" },
            { x: "v", fill: "est", thresholds: 30, fillOpacity: 0.6 },
          ),
        ),
        Plot.ruleX([1], { stroke: palette.muted, ...dashed }),
      ],
    }));

    const last = stats[stats.length - 1];
    const ratio = last.truncStd > 0 ? last.rawStd / last.truncStd : Infinity;
    slots.readout.textContent =
      `at H = ${Hmax}:  raw-IS stddev = ${last.rawStd.toExponential(2)}   ` +
      `V-trace stddev = ${last.truncStd.toExponential(2)}   ` +
      `(${isFinite(ratio) ? ratio.toFixed(1) : "∞"}× reduction)`;
  },
});

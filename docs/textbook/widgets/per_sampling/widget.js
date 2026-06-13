// Widget 9.E — Prioritized Experience Replay distribution (Chapter 9).
//
// Section §9.5 introduces PER's two hyperparameters:
//
//   p_i ∝ |δ_i|^ρ + ε     (sampling probability)
//   w_i = (1 / (N · p_i))^β   (importance-sampling weight)
//
// and the effective loss contribution per transition i is
//
//   contrib_i = p_i · w_i = p_i^{1-β} · (1/N)^β.
//
// At ρ = 0: uniform sampling, w_i = 1, contrib uniform.
// At ρ = 1, β = 1: aggressive prioritisation, but w_i corrects fully —
// so contrib is *flat again* (the IS weight cancels the priority).
// At ρ = 1, β = 0.4: prioritisation with partial correction — high-|δ|
// transitions dominate the loss.
//
// We draw a 1000-transition buffer of TD errors from a heavy-tailed
// distribution (|δ_i| = |N(0,1)|^2, so a few outliers dominate), then
// for the chosen (ρ, β) plot p_i, w_i, and contrib_i sorted by |δ_i|.
//
// Pattern: chapter markdown contains
//
//     <div id="ch9-per-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/per_sampling/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const N = 1000;
const EPS = 1e-3;

function randn() {
  const u = Math.max(Math.random(), 1e-12);
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Build a deterministic heavy-tailed |δ| distribution once on module
// load. Keeping it static lets the reader compare different (ρ, β) on
// the same buffer rather than chasing fresh randomness across slider
// scrubs.
function buildBuffer(seed) {
  // Simple LCG so the buffer is reproducible across reloads.
  let s = seed >>> 0;
  const rand = () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const r = () => {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const errs = new Array(N);
  for (let i = 0; i < N; i++) {
    // |δ| = |N(0,1)|^2 → power-law-ish right tail.
    const z = r();
    errs[i] = z * z;
  }
  // Sort ascending by |δ|. Then index i = 0 is the smallest error, i = N−1
  // is the biggest — natural "left = boring, right = high-priority" axis.
  errs.sort((a, b) => a - b);
  return errs;
}

const BUFFER = buildBuffer(42);

defineWidget({
  hostId: "ch9-per-widget",
  controls: {
    rho:  { label: "ρ (priority exponent)", min: 0.0, max: 1.0, step: 0.05, default: 0.6 },
    beta: { label: "β (IS correction)",     min: 0.0, max: 1.0, step: 0.05, default: 0.4 },
  },
  slots: ["main"],
  render: (host, { rho, beta }, slots) => {
    // Compute unnormalised priorities and normalising constant.
    let Z = 0;
    const priRaw = new Array(N);
    for (let i = 0; i < N; i++) {
      priRaw[i] = Math.pow(BUFFER[i] + EPS, rho);
      Z += priRaw[i];
    }
    const rows = [];
    for (let i = 0; i < N; i++) {
      const p = priRaw[i] / Z;
      const w = Math.pow(1.0 / (N * p), beta);
      const contrib = p * w;
      rows.push({ rank: i + 1, delta: BUFFER[i], p, w, contrib });
    }

    // Normalize w by max(w) so the "weights" curve is on [0, 1] like
    // PER does in practice — only relative weights matter for gradient
    // scaling, and normalising avoids vertical-axis chaos when β is
    // small (very large w on rare transitions).
    const wMax = rows.reduce((m, r) => Math.max(m, r.w), 0);
    for (const r of rows) r.w = r.w / wMax;

    // Same for contrib so the three curves share a [0, scale] band.
    const contribMax = rows.reduce((m, r) => Math.max(m, r.contrib), 0);
    for (const r of rows) r.contribNorm = r.contrib / contribMax;
    // p is naturally tiny (sums to 1 over N=1000), so scale to its own
    // max for visibility.
    const pMax = rows.reduce((m, r) => Math.max(m, r.p), 0);
    for (const r of rows) r.pNorm = r.p / pMax;

    // Long form for Plot — three series on a shared rank axis.
    const long = [];
    for (const r of rows) {
      long.push({ rank: r.rank, value: r.pNorm, series: "priority p_i" });
      long.push({ rank: r.rank, value: r.w, series: "IS weight w_i" });
      long.push({ rank: r.rank, value: r.contribNorm, series: "loss contrib p_i·w_i" });
    }
    const colorFor = (s) =>
      s === "priority p_i"
        ? palette.danger
        : s === "IS weight w_i"
          ? palette.secondary
          : palette.primary;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "buffer rank (sorted by |δ|, lowest → highest)", grid: true, domain: [1, N] },
      y: { label: "value (each series scaled to its own max)", grid: true, domain: [0, 1.05] },
      color: {
        legend: true,
        domain: ["priority p_i", "IS weight w_i", "loss contrib p_i·w_i"],
        range: [palette.danger, palette.secondary, palette.primary],
      },
      marks: [
        // Uniform reference line so the reader sees "what flat looks like."
        Plot.ruleY([1.0 / N * N], { stroke: palette.muted, ...dashed }), // value 1.0 (own-max scale)
        Plot.line(long, { x: "rank", y: "value", stroke: (d) => colorFor(d.series), z: "series", strokeWidth: 1.6 }),
        // Highlight the top-priority transitions visually.
        Plot.text(
          [{ rank: N, y: 1.0, label: "high-|δ| tail →" }],
          { x: "rank", y: "y", text: "label", textAnchor: "end", dy: -6, fill: palette.muted, ...annotation },
        ),
      ],
    }));

    // Compute a couple of summary numbers: what fraction of the
    // *loss* gets contributed by the top 10% of transitions under
    // current (ρ, β)?
    let topMass = 0;
    let totalMass = 0;
    const cutoff = Math.floor(0.9 * N);
    for (let i = 0; i < N; i++) {
      totalMass += rows[i].contrib;
      if (i >= cutoff) topMass += rows[i].contrib;
    }
    const topFrac = topMass / totalMass;
    // Effective sample size: N_eff = (Σ p_i w_i)^2 / Σ (p_i w_i)^2, but
    // since we want a simple "concentration" readout, report the
    // top-10% mass instead.
    slots.readout.textContent =
      `ρ = ${rho.toFixed(2)}, β = ${beta.toFixed(2)} · ` +
      `top-10% of transitions carry ${(topFrac * 100).toFixed(1)}% of loss mass · ` +
      `at ρ=0 (uniform) this would be 10.0%.`;
  },
});

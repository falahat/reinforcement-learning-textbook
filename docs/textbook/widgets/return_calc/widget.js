// Widget 2.D — Episode vs continuing return calculator (Chapter 2).
//
// Lets the student build a reward profile from presets and compute
//
//   G_0 (undiscounted) = Σ_t r_t
//   G_0 (discounted)   = Σ_t γ^t · r_t
//
// for either an episodic horizon (cut off at T) or the continuing
// case (T = 200, the full window). The student sees how a *delayed*
// reward of +10 at t=100 looks huge at γ=0.999 and tiny at γ=0.9 —
// motivating the effective-horizon discussion that opens §2.3.
//
// Pattern: chapter markdown contains
//
//     <div id="ch2-return-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/return_calc/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const T_MAX = 200;

// Reward profile presets. Each returns an array `r[t]` for t in [0, T_MAX].
const PROFILES = {
  sparse: () => {
    const r = new Array(T_MAX + 1).fill(0);
    r[100] = 10;
    return r;
  },
  constant: () => new Array(T_MAX + 1).fill(0.1),
  burst: () => {
    const r = new Array(T_MAX + 1).fill(0);
    for (let t = 5; t <= 10; t++) r[t] = 1;
    return r;
  },
  mixed: () => {
    const r = new Array(T_MAX + 1).fill(0.05);
    for (let t = 5; t <= 10; t++) r[t] = 0.5;
    r[100] = 5;
    return r;
  },
  // Seeded so the "noisy" profile produces the same bars on every render
  // — otherwise moving γ or T re-rolls the noise and the user sees the
  // chart jitter for reasons unrelated to their input.
  noisy: () => {
    let s = 0x9e3779b9;
    return d3.range(T_MAX + 1).map(() => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280 < 0.1 ? 1 : 0;
    });
  },
};

const PROFILE_LABELS = {
  sparse: "sparse: +10 at t=100",
  constant: "constant: +0.1 every step",
  burst: "burst: +1 for t=5..10",
  mixed: "mixed: 0.05 baseline + burst + sparse",
  noisy: "noisy: Bernoulli(0.1) reward",
};

defineWidget({
  hostId: "ch2-return-widget",
  controls: {
    profile: {
      type: "select",
      label: "reward profile",
      options: Object.entries(PROFILE_LABELS).map(([value, label]) => ({ value, label })),
      default: "sparse",
    },
    gamma: { label: "γ", min: 0.5, max: 0.999, step: 0.001, default: 0.95 },
    mode: {
      type: "select",
      label: "horizon",
      options: [
        { value: "continuing", label: "continuing (T = 200)" },
        { value: "episodic", label: "episodic (cut at T)" },
      ],
      default: "continuing",
    },
    T: { label: "T (episodic cutoff)", min: 10, max: T_MAX, step: 1, default: 50 },
  },
  slots: ["rewards", "weighted"],
  render: (host, { profile, gamma, mode, T }, slots) => {
    const r = PROFILES[profile]();
    const cutoff = mode === "episodic" ? Math.round(T) : T_MAX;

    // Two return computations.
    let G_undisc = 0;
    let G_disc = 0;
    const weighted = []; // for the discounted-contribution plot
    const cumDisc = [];
    let running = 0;
    for (let t = 0; t <= cutoff; t++) {
      const w = Math.pow(gamma, t) * r[t];
      G_undisc += r[t];
      G_disc += w;
      running += w;
      weighted.push({ t, contrib: w });
      cumDisc.push({ t, cum: running });
    }

    // R_max envelope = max(|r|) · 1/(1-γ).
    const rMax = Math.max(...r.map((x) => Math.abs(x)));
    const bound = gamma < 1 ? rMax / (1 - gamma) : Infinity;

    // Top plot — raw reward profile, with cutoff line.
    const rewardData = r.map((y, t) => ({ t, r: y }));
    slots.rewards.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 150,
      x: { label: "t", domain: [0, T_MAX], grid: true },
      y: { label: "r_t", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        // rectY + interval on a continuous x — Plot.barY would produce
        // zero-width bars here because the x scale isn't ordinal/band.
        Plot.rectY(rewardData, {
          x: "t", y: "r", interval: 1,
          fill: (d) => d.t <= cutoff ? palette.primary : "#555",
          fillOpacity: (d) => d.t <= cutoff ? 0.9 : 0.3,
        }),
        ...(mode === "episodic" ? [
          Plot.ruleX([cutoff], { stroke: palette.danger, ...dashed }),
          Plot.text([{ x: cutoff, y: rMax * 0.9, label: `T = ${cutoff}` }], {
            x: "x", y: "y", text: "label", textAnchor: "start", dx: 4,
            fill: palette.danger, ...annotation,
          }),
        ] : []),
      ],
    }));

    // Bottom plot — discounted contributions γ^t · r_t with cumulative overlay.
    slots.weighted.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "t", domain: [0, T_MAX], grid: true },
      y: { label: "γ^t · r_t  (bars)   cumulative G_0  (line)", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.rectY(weighted, { x: "t", y: "contrib", interval: 1, fill: palette.secondary, fillOpacity: 0.7 }),
        Plot.line(cumDisc, { x: "t", y: "cum", stroke: palette.accent, strokeWidth: 2 }),
        ...(isFinite(bound) ? [
          Plot.ruleY([bound], { stroke: palette.danger, ...dashed }),
          Plot.text(
            [{ x: T_MAX * 0.98, y: bound, label: `R_max/(1-γ) = ${bound.toFixed(2)}` }],
            { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
              fill: palette.danger, ...annotation },
          ),
        ] : []),
      ],
    }));

    const ratio = G_undisc !== 0 ? (G_disc / G_undisc) : 0;
    slots.readout.innerHTML =
      `<strong>G_0 undiscounted</strong> = <code>${G_undisc.toFixed(3)}</code> &nbsp; ` +
      `<strong>G_0 discounted</strong> = <code>${G_disc.toFixed(3)}</code> &nbsp; ` +
      `<small>γ=${gamma.toFixed(3)}, horizon≈${(1 / (1 - gamma)).toFixed(1)}, ` +
      `disc/undisc=${ratio.toFixed(3)}, cutoff t≤${cutoff}</small>`;
  },
});

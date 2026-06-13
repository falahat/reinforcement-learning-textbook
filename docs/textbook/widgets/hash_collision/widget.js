// Widget 8.F — Hash collision pressure (Chapter 8, §8.7).
//
// Visualises the §8.7 "estimating collision pressure" argument. Each of
// N distinct observations activates T tile slots, drawn (under the IHT
// hashing assumption) uniformly at random from M slots. The widget plots:
//
//   * Expected fraction of slots occupied as a function of N:
//       E[occupied / M] = 1 - (1 - 1/M)^(T·N).
//   * Expected fraction of *active* tile draws that hit an already-used
//     slot — a "collision rate" approximation:
//       collision ≈ 1 - (1 - 1/M) ^ (T·N / 2).
//   * A vertical reference line at "N where occupancy crosses 50%"
//     (the chapter's threshold for "heavy collision regime").
//
// Sliders: log2(M), T, max log10(N). A toggle for the Simulator's defaults
// (M = 2^16, T = 16) flips the sliders to those values.
//
// The headline pedagogical hit: the Simulator's M = 2^16 + T = 16 means
// the 50% occupancy point is at N ≈ 2840 — well inside a normal run.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

// 1 - (1 - 1/M)^k, numerically stable for k/M large or small.
// Uses expm1 to avoid catastrophic cancellation when k/M is tiny.
function occupied(k, M) {
  // p = 1 - exp(k * log(1 - 1/M)) = -expm1(k * log1p(-1/M)).
  const t = k * Math.log1p(-1 / M);
  return -Math.expm1(t);
}

defineWidget({
  hostId: "ch8-hash-collision-widget",
  controls: {
    log2M: { label: "log2(M) — IHT size",     min: 8,  max: 22, step: 1, default: 16 },
    T:     { label: "T (tilings)",            min: 1,  max: 64, step: 1, default: 16 },
    log10Nmax: { label: "log10(N_max)",       min: 2,  max: 8,  step: 0.1, default: 6 },
    log10N:    { label: "log10(N) — readout", min: 1,  max: 7,  step: 0.05, default: 3.5 },
  },
  slots: ["main"],
  render: (host, { log2M, T, log10Nmax, log10N }, slots) => {
    const M = Math.pow(2, Math.round(log2M));
    const Ti = Math.round(T);
    const Nmax = Math.pow(10, log10Nmax);
    const N = Math.min(Math.pow(10, log10N), Nmax);

    // Sweep N on log scale.
    const data = d3.range(0, 1.0001, 0.005).map((tFrac) => {
      const Nval = Math.max(1, Math.pow(10, 1 + (log10Nmax - 1) * tFrac));
      const k = Ti * Nval;
      const occ = occupied(k, M);
      const collision = occupied(k / 2, M);
      return { N: Nval, occupied: occ, collision };
    });

    // Find N where occupancy = 0.5 (newton on log domain).
    // p = 0.5 ⇒ k = log(0.5) / log(1 - 1/M) ⇒ N = k / T.
    const k50 = Math.log(0.5) / Math.log1p(-1 / M);
    const N50 = k50 / Ti;

    const occHere = occupied(Ti * N, M);
    const collisionHere = occupied((Ti * N) / 2, M);

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 280,
        x: { type: "log", label: "N (distinct observations visited)", grid: true,
             domain: [10, Nmax] },
        y: { label: "expected fraction", domain: [0, 1.02], grid: true },
        color: { legend: true, range: [palette.primary, palette.warning] },
        marks: [
          Plot.line(data, {
            x: "N", y: "occupied",
            stroke: palette.primary, strokeWidth: 2,
          }),
          Plot.line(data, {
            x: "N", y: "collision",
            stroke: palette.warning, strokeWidth: 2,
          }),
          Plot.ruleY([0.5], { stroke: palette.muted, ...dashed }),
          Plot.ruleX([N50], { stroke: palette.danger, ...dashed }),
          Plot.ruleX([N],   { stroke: palette.accent, ...dashed }),
          Plot.text(
            [{ x: N50, y: 0.5, label: `N₅₀ ≈ ${d3.format(",.0f")(N50)}` }],
            { x: "x", y: "y", text: "label", fill: palette.danger,
              textAnchor: "start", dx: 6, dy: -4, ...annotation },
          ),
          Plot.text(
            [
              { x: Nmax, y: 0.92, label: "fraction of slots occupied" },
              { x: Nmax, y: 0.50, label: "fraction of new-tile draws colliding (~k/2 approx.)" },
            ],
            { x: "x", y: "y", text: "label", fill: (d, i) => i === 0 ? palette.primary : palette.warning,
              textAnchor: "end", dy: -4, ...annotation },
          ),
        ],
      }),
    );

    const isSimDefaults = (Math.round(log2M) === 16) && (Ti === 16);
    const note = isSimDefaults
      ? "  (matches Simulator defaults: M = 2^16, T = 16)"
      : "";
    slots.readout.textContent =
      `M = 2^${Math.round(log2M)} = ${M.toLocaleString()}  ·  T = ${Ti}  ·  ` +
      `at N = ${d3.format(",.0f")(N)}: occupied = ${(100 * occHere).toFixed(1)}%, ` +
      `collision ≈ ${(100 * collisionHere).toFixed(1)}%  ·  ` +
      `N₅₀ ≈ ${d3.format(",.0f")(N50)}${note}`;
  },
});

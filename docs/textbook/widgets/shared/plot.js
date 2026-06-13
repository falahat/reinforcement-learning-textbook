// Plot-mark helpers for textbook widgets.
//
// These wrap a few patterns of Observable Plot that the project's
// widgets ran into repeatedly — usually because Plot defaults silently
// turn certain configurations into zero-size marks.
//
// Plot's API has two recurring traps in this codebase:
//
//   1. `Plot.barY` / `Plot.barX` require an ORDINAL/BAND scale on
//      their categorical axis. A 2-element numeric domain
//      `[min, max]` forces a linear scale, and bars render with
//      zero width. Six widgets shipped with this bug.
//   2. `Plot.cell` similarly requires band scales on both x AND y.
//      Eight more widgets shipped with the same bug for heatmaps.
//
// The helpers below produce "obviously correct" Plot config snippets
// for these cases so future widgets don't have to remember the rule.
// See `.claude/skills/textbook-with-widgets/references/10-widget-correctness-audit.md`
// (bug class 8) for the long write-up.

import * as Plot from "@observablehq/plot";

/**
 * Build the `x` (or `y`) axis config for a band scale over integer
 * coordinates `0..N-1`. Use when you want `Plot.cell` or `Plot.barY`
 * to render proper-size marks indexed by integers.
 *
 * @param {number} N — number of distinct values (band positions).
 * @param {Object} [extra] — additional axis options (label, axis, etc.).
 * @returns {Object} — a Plot scale config object.
 */
export function bandAxis(N, extra = {}) {
  // d3.range(N) returns [0, 1, ..., N-1]; Plot infers an ordinal/band
  // scale from an array domain.
  const domain = [];
  for (let i = 0; i < N; i++) domain.push(i);
  return { type: "band", domain, ...extra };
}

/**
 * Convenience for gridworld heatmaps where both x and y need band
 * scales. Returns an object with `x` and `y` Plot scale configs you
 * can spread into `Plot.plot({...})`.
 *
 * Three calling conventions:
 *   gridAxes(N)                 // both axes 0..N-1
 *   gridAxes({ nx, ny })        // x = 0..nx-1, y = 0..ny-1
 *   gridAxes(N, extra)          // extra applied to BOTH axes
 *   gridAxes({ nx, ny }, common, { x, y })  // per-axis overrides
 *
 * @returns {{ x: Object, y: Object }}
 */
export function gridAxes(spec, common = {}, perAxis = {}) {
  const nx = typeof spec === "number" ? spec : spec.nx;
  const ny = typeof spec === "number" ? spec : spec.ny;
  return {
    x: bandAxis(nx, { ...common, ...(perAxis.x || {}) }),
    y: bandAxis(ny, { ...common, ...(perAxis.y || {}) }),
  };
}

/**
 * `Plot.barY` on a continuous integer x with `interval: 1` —
 * histogram-style unit-width bars on a non-band scale. Use this when
 * you need ticks at every integer (which `band` doesn't render well
 * for large N) but still want bars to fill their unit cell.
 *
 * @param {Array<Object>} data
 * @param {Object} options — passed through to Plot.rectY (must include x, y).
 * @returns {*}
 */
export function unitBarsY(data, options) {
  return Plot.rectY(data, { interval: 1, ...options });
}

/** Same as `unitBarsY` but horizontal bars. */
export function unitBarsX(data, options) {
  return Plot.rectX(data, { interval: 1, ...options });
}

/**
 * "Strip" of coloured rectangles along a continuous x — useful for
 * 1D ribbons (eligibility traces, track-cell visualisations) where
 * `Plot.cell` won't work because we need a continuous x-axis for tick
 * labels.
 *
 * Each row in `data` becomes a rectangle from x-0.5..x+0.5 spanning
 * `y1`..`y2` on the y-axis.
 *
 * @param {Array<Object>} data — each row must have an x field.
 * @param {Object} options — pass `x: "fieldname"`, `fill: ...`, etc.
 *   `y1`, `y2` default to 0 and 1.
 * @returns {*}
 */
export function ribbonX(data, { x, y1 = 0, y2 = 1, ...rest } = {}) {
  return Plot.rect(data, {
    x1: (d) => d[x] - 0.5,
    x2: (d) => d[x] + 0.5,
    y1,
    y2,
    ...rest,
  });
}

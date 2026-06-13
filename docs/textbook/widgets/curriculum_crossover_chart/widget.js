// Curriculum crossover chart — x-axis difficulty, y-axis
// time-to-criterion, one line per variant. The **crossover point**
// (the difficulty at which time-to-criterion exceeds the ceiling) is
// the substantive variant statistic.
//
// Consumes per-variant `criterion.json` files (one per difficulty
// level) — the curriculum harness writes one experiment directory
// per level, named by axis value.

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const PALETTE = [
  palette.primary, palette.secondary, palette.tertiary, palette.warn,
  palette.muted, palette.accent,
];

defineWidget({
  hostId: "ch19-curriculum-crossover",
  controls: {},
  render: async (host, _ctrls, slots) => {
    // The curriculum manifest lists levels in order; per-level
    // results live at <base>/levels/<label>/{manifest,comparison}.json.
    const base = host.dataset.experiment ?? "../experiments/long_horizon_harvest";
    const top = await fetch(`${base}/manifest.json`).then((r) => r.json()).catch(() => null);
    if (!top) {
      slots.main.textContent = `(no curriculum manifest at ${base})`;
      return;
    }
    const levels = top.levels ?? [];
    if (levels.length === 0) {
      slots.main.textContent = "(curriculum has no levels)";
      return;
    }
    const series = [];
    let ceiling = Infinity;
    for (const level of levels) {
      const levelDir = `${base}/levels/${level.label}`;
      const levelManifest = await fetch(`${levelDir}/manifest.json`).then((r) => r.json()).catch(() => null);
      if (!levelManifest) continue;
      ceiling = Math.min(ceiling, levelManifest.ceiling_ticks ?? Infinity);
      for (let i = 0; i < levelManifest.variants.length; i++) {
        const variant = levelManifest.variants[i];
        const crit = await fetch(`${levelDir}/variants/${variant}/criterion.json`).then((r) => r.json()).catch(() => null);
        if (!crit) continue;
        series.push({
          difficulty: level.value ?? level.label,
          variant,
          time_to_criterion: crit.time_to_criterion ?? ceiling,
          passed: crit.passed,
          colour: PALETTE[i % PALETTE.length],
        });
      }
    }
    if (series.length === 0) {
      slots.main.textContent = "(no per-level criterion data — did you `.save()`?)";
      return;
    }
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "difficulty", type: "linear" },
      y: { label: "time to criterion (ticks)", grid: true, type: "linear" },
      color: { legend: true },
      marks: [
        Plot.ruleY([ceiling], { stroke: "#c00", strokeDasharray: "4 4" }),
        Plot.line(series, { x: "difficulty", y: "time_to_criterion", stroke: "variant", strokeWidth: 1.8 }),
        Plot.dot(series, {
          x: "difficulty",
          y: "time_to_criterion",
          fill: "variant",
          symbol: (d) => (d.passed ? "circle" : "cross"),
          r: 5,
        }),
      ],
    }));
    slots.readout.textContent = `${top.name ?? "curriculum"} · ${levels.length} difficulties · ceiling ${ceiling}`;
  },
});

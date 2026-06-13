// Learning-curve overlay — one line per experiment variant on shared
// axes. Reads a `curves.json` per variant from the experiment output
// tree, optionally hands the user a metric selector and a variant
// toggle. The data URL is configurable via the host's
// `data-experiment` attribute (path to the experiment directory).

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const PALETTE = [
  palette.primary, palette.secondary, palette.tertiary, palette.warn,
  palette.muted, palette.accent,
];

defineWidget({
  hostId: "ch5-learning-curve-overlay",
  controls: {
    metric: {
      type: "select",
      label: "metric",
      options: [
        "learning.learner.q_best_value",
        "learning.learner.q_global_max",
        "learning.learner.td_error",
        "learning.learner.reward",
      ],
      default: "learning.learner.q_best_value",
    },
    aggregation: {
      type: "select",
      label: "aggregation",
      options: ["mean-across-seeds", "per-seed"],
      default: "mean-across-seeds",
    },
  },
  render: async (host, { metric, aggregation }, slots) => {
    const base = host.dataset.experiment ?? "../experiments/scoring_variants";
    const manifest = await fetch(`${base}/manifest.json`).then((r) => r.json()).catch(() => null);
    if (!manifest) {
      slots.main.textContent = `(no manifest at ${base}/manifest.json)`;
      return;
    }
    const series = [];
    for (let i = 0; i < manifest.variants.length; i++) {
      const variant = manifest.variants[i];
      const curves = await fetch(`${base}/variants/${variant}/curves.json`).then((r) => r.json()).catch(() => null);
      if (!curves || !curves.curves[metric]) continue;
      const perSeed = curves.curves[metric];
      const colour = PALETTE[i % PALETTE.length];
      if (aggregation === "per-seed") {
        perSeed.forEach((seed, si) => {
          for (const [t, v] of seed) {
            series.push({ tick: t, value: v, variant: `${variant} · seed ${si}`, colour });
          }
        });
      } else {
        const len = Math.min(...perSeed.map((s) => s.length));
        for (let k = 0; k < len; k++) {
          const tick = perSeed[0][k][0];
          const mean = perSeed.reduce((acc, s) => acc + s[k][1], 0) / perSeed.length;
          series.push({ tick, value: mean, variant, colour });
        }
      }
    }
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "tick" },
      y: { label: metric, grid: true },
      color: { legend: true },
      marks: [
        Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
        Plot.line(series, { x: "tick", y: "value", stroke: "variant", strokeWidth: 1.5 }),
      ],
    }));
    slots.readout.textContent = `${manifest.experiment} · ${manifest.variants.length} variants · ${manifest.seeds} seeds`;
  },
});

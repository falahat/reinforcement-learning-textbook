// Parameter-grid heatmap — one cell per variant, colour = chosen
// metric's cross-seed mean. Used to read off the winning configuration
// at a glance from an experiment that swept across (α, γ) or
// (scorer, shaper) tuples.

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults } from "../shared/helpers.js";

defineWidget({
  hostId: "ch11-parameter-grid-heatmap",
  controls: {
    metric: {
      type: "select",
      label: "metric",
      options: [
        "learning.learner.q_best_value.final",
        "learning.learner.q_best_value.mean",
        "learning.learner.td_error.trailing_mean",
        "learning.learner.reward.final",
      ],
      default: "learning.learner.q_best_value.final",
    },
  },
  render: async (host, { metric }, slots) => {
    const base = host.dataset.experiment ?? "../experiments/scoring_variants";
    const comparison = await fetch(`${base}/comparison.json`).then((r) => r.json()).catch(() => null);
    if (!comparison) {
      slots.main.textContent = `(no comparison.json at ${base})`;
      return;
    }
    const row = comparison.metrics.find((m) => m.metric === metric);
    if (!row) {
      slots.main.textContent = `(metric '${metric}' not in comparison)`;
      return;
    }
    // Variant labels often encode their axes as `axis1=v1_axis2=v2`. If
    // no `_` present, lay them out in a single row.
    const cells = Object.entries(row.values).map(([variant, value]) => {
      const parts = variant.split("_");
      const x = parts[0] ?? variant;
      const y = parts[1] ?? "—";
      return { variant, x, y, value };
    });
    const winner = row.winner;
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "axis 1" },
      y: { label: "axis 2", reverse: true },
      color: { type: "linear", legend: true, scheme: "viridis" },
      marks: [
        Plot.cell(cells, { x: "x", y: "y", fill: "value", inset: 1 }),
        Plot.text(cells, {
          x: "x",
          y: "y",
          text: (d) => (d.value == null ? "—" : d.value.toFixed(3)),
          fill: "white",
          stroke: "black",
          strokeWidth: 2,
        }),
      ],
    }));
    slots.readout.textContent = `${metric} · winner: ${winner ?? "—"}`;
  },
});

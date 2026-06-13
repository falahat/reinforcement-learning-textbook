// Attribution-spectrum overlay — lag-vs-credit curve per variant.
//
// TD(0) shows an `(αγ)^k`-shaped exponential decay. SMDP-Q over an
// option of duration τ shows a single spike at lag τ. Each variant
// gets its own line; the qualitative shape difference is the
// substantive comparison.

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const PALETTE = [
  palette.primary, palette.secondary, palette.tertiary, palette.warn,
  palette.muted, palette.accent,
];

defineWidget({
  hostId: "ch15-attribution-spectrum-overlay",
  controls: {
    scale: {
      type: "select",
      label: "y-scale",
      options: ["linear", "log"],
      default: "linear",
    },
  },
  render: async (host, { scale }, slots) => {
    const base = host.dataset.experiment ?? "../experiments/scoring_variants";
    const manifest = await fetch(`${base}/manifest.json`).then((r) => r.json()).catch(() => null);
    if (!manifest) {
      slots.main.textContent = `(no manifest at ${base})`;
      return;
    }
    // The harness writes per-variant attribution spectra into
    // variants/<label>/metrics.json as scalar series with keys
    // matching `attribution.<agent>.spectrum.lag_<k>`. We rebuild
    // the curve by harvesting any lag-shaped keys we find.
    const series = [];
    for (let i = 0; i < manifest.variants.length; i++) {
      const variant = manifest.variants[i];
      const m = await fetch(`${base}/variants/${variant}/metrics.json`).then((r) => r.json()).catch(() => null);
      if (!m) continue;
      for (const [key, stats] of Object.entries(m.scalars ?? {})) {
        const match = key.match(/spectrum.*lag[_.](\d+)/);
        if (match) {
          series.push({
            variant,
            lag: parseInt(match[1], 10),
            credit: stats.mean,
            colour: PALETTE[i % PALETTE.length],
          });
        }
      }
    }
    if (series.length === 0) {
      slots.main.textContent =
        "(no attribution spectrum data — does the task fixture insert a `RewardAttributionLedger`?)";
      return;
    }
    series.sort((a, b) => a.variant.localeCompare(b.variant) || a.lag - b.lag);
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      x: { label: "lag (ticks back)" },
      y: { label: "credit (Σ α·δ)", type: scale, grid: true },
      color: { legend: true },
      marks: [
        Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.3 }),
        Plot.line(series, { x: "lag", y: "credit", stroke: "variant", strokeWidth: 1.8 }),
        Plot.dot(series, { x: "lag", y: "credit", fill: "variant" }),
      ],
    }));
    slots.readout.textContent = `${manifest.experiment} · attribution spectrum`;
  },
});

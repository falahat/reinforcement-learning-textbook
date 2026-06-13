// Widget 16.D — Wanting vs liking dissociation timeline.
//
// Two coupled signals over time (Berridge's incentive-sensitisation
// model, stylised):
//   - liking L_t: hedonic impact at consumption. Saturates and then
//     declines (tolerance).
//   - wanting W_t: incentive salience. Sensitises with each use,
//     climbs without bound.
// The gap (wanting − liking) is the substance-abuse signal: high pull
// to act, low actual pleasure.
//
// Mount:
//     <div id="ch16-wanting-liking-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/wanting_liking/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const T = 200;

// Use a deterministic schedule: use_t = 1 every `period` ticks.
// freq ∈ [0, 1] maps to period via period = round(20 / (1 + 19*freq)).
function periodFromFreq(freq) {
  // freq=0 → period=20 (rare); freq=1 → period=1 (every tick).
  return Math.max(1, Math.round(20 / (1 + 19 * freq)));
}

// Coupled discrete-time dynamics:
//   liking_{t+1}  = liking_t  + use_t * (L_max - liking_t) * gainL
//                 - use_t * tolerance * max(0, liking_t - L_baseline)
//                 - decay_L * (liking_t - L_baseline)
//   wanting_{t+1} = wanting_t + use_t * sensitise * (1 + wanting_t)
//                 - decay_W * (wanting_t - W_baseline)
// Liking saturates at L_max (set by hedonic ceiling); wanting sensitises
// multiplicatively (no ceiling). Tolerance reduces liking proportional
// to its current elevation.
function simulate(freq, sensitise, tolerance) {
  const period = periodFromFreq(freq);
  const L_max = 1.0;
  const L_base = 0.0;
  const W_base = 0.0;
  const gainL = 0.45;
  const decayL = 0.04;
  const decayW = 0.01;

  let L = 0, W = 0;
  const rows = new Array(T + 1);
  rows[0] = { t: 0, liking: 0, wanting: 0, gap: 0, used: 0 };
  for (let t = 0; t < T; t++) {
    const use = (t > 0 && (t % period === 0)) ? 1 : 0;
    const dL = use * gainL * (L_max - L)
             - use * tolerance * Math.max(0, L - L_base)
             - decayL * (L - L_base);
    const dW = use * sensitise * (1 + W)
             - decayW * (W - W_base);
    L = Math.max(0, L + dL);
    W = Math.max(0, W + dW);
    rows[t + 1] = { t: t + 1, liking: L, wanting: W, gap: W - L, used: use };
  }
  return rows;
}

defineWidget({
  hostId: "ch16-wanting-liking-widget",
  controls: {
    freq:      { label: "use frequency", min: 0.05, max: 1.0, step: 0.01, default: 0.30 },
    sensitise: { label: "wanting sensitisation", min: 0.0, max: 0.2, step: 0.005, default: 0.05 },
    tolerance: { label: "liking tolerance", min: 0.0, max: 0.5, step: 0.01, default: 0.15 },
  },
  slots: ["main", "extra"],
  render: (host, { freq, sensitise, tolerance }, slots) => {
    const rows = simulate(freq, sensitise, tolerance);
    const stride = 1;
    const data = [];
    for (let i = 0; i <= T; i += stride) {
      data.push({ t: rows[i].t, value: rows[i].wanting, signal: "wanting" });
      data.push({ t: rows[i].t, value: rows[i].liking, signal: "liking" });
    }

    const gapData = rows.map((r) => ({ t: r.t, gap: r.gap }));
    const useTicks = rows.filter((r) => r.used).map((r) => ({ t: r.t }));

    const yMax = Math.max(
      1.1,
      Math.max(...rows.map((r) => r.wanting)) * 1.1,
    );

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "time t", grid: true, domain: [0, T] },
      y: { label: "signal", grid: true, domain: [0, yMax] },
      color: {
        legend: true,
        domain: ["wanting", "liking"],
        range: [palette.danger, palette.primary],
      },
      marks: [
        Plot.ruleY([1.0], { stroke: palette.muted, ...dashed }),
        Plot.text(
          [{ x: T, y: 1.0, label: "liking ceiling" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.muted, ...annotation },
        ),
        // Use events along the bottom of the chart.
        Plot.tickX(useTicks, { x: "t", stroke: palette.warning, strokeOpacity: 0.4 }),
        Plot.line(data, {
          x: "t", y: "value", stroke: "signal", z: "signal", strokeWidth: 2,
        }),
      ],
    }));

    slots.extra.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      x: { label: "time t", grid: true, domain: [0, T] },
      y: { label: "wanting − liking", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.areaY(gapData, {
          x: "t", y: "gap",
          fill: palette.accent, fillOpacity: 0.45,
        }),
        Plot.line(gapData, { x: "t", y: "gap", stroke: palette.accent, strokeWidth: 1.5 }),
      ],
    }));

    const period = periodFromFreq(freq);
    const finalGap = rows[T].gap;
    slots.readout.textContent =
      `use every ${period} ticks; ` +
      `final wanting = ${rows[T].wanting.toFixed(2)}, ` +
      `liking = ${rows[T].liking.toFixed(2)}, ` +
      `gap = ${finalGap.toFixed(2)} (higher = more abuse-like).`;
  },
});

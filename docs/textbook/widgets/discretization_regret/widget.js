// Widget 18.B — Discretization-cost visualizer (Chapter 18).
//
// `Strike{force}` toy: a continuous force ∈ [0, 1] dial, with a true
// unimodal Q-curve Q*(force) = exp(−((force − peak)^2) / 2σ²). The
// reader sets a bucket count B ∈ {2, 4, 8, 16, 32}, and the widget
// draws the bucketed step-function Q̂ alongside the smooth Q* and
// marks the per-step regret = max_force Q*(force) − Q̂(argmax_bucket).
//
// Slider for the peak location lets the reader create "boundary
// unfortunate" cases where the true optimum sits in the trough between
// two bucket centroids — the headline §18.1 cost of discretization.
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-discretization-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/discretization_regret/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

// True Q*(force) — unimodal Gaussian bump on [0, 1].
function qTrue(force, peak, sigma) {
  const d = force - peak;
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

defineWidget({
  hostId: "ch18-discretization-widget",
  controls: {
    buckets: { label: "B (buckets)", min: 2, max: 32, step: 1, default: 4 },
    peak:    { label: "true peak force*", min: 0.05, max: 0.95, step: 0.01, default: 0.62 },
    sigma:   { label: "Q* width σ", min: 0.05, max: 0.4, step: 0.01, default: 0.15 },
  },
  render: (host, { buckets, peak, sigma }, slots) => {
    const B = Math.round(buckets);
    // Bucket centroids — uniformly spaced midpoints over [0, 1].
    const centroids = d3.range(B).map((i) => (i + 0.5) / B);
    const qHat = centroids.map((c) => qTrue(c, peak, sigma));
    // argmax over buckets.
    let argmaxIdx = 0;
    for (let i = 1; i < B; i++) if (qHat[i] > qHat[argmaxIdx]) argmaxIdx = i;
    const bestBucketForce = centroids[argmaxIdx];
    const bestBucketQ = qHat[argmaxIdx];
    const optimalQ = qTrue(peak, peak, sigma); // = 1
    const regret = optimalQ - bestBucketQ;

    // Smooth true curve.
    const N = 401;
    const curve = d3.range(N).map((i) => {
      const f = i / (N - 1);
      return { force: f, q: qTrue(f, peak, sigma) };
    });
    // Step-function points: for plotting, each bucket spans
    // [i/B, (i+1)/B] with constant value qHat[i].
    const stepRects = centroids.map((c, i) => ({
      x1: i / B,
      x2: (i + 1) / B,
      y: qHat[i],
      isBest: i === argmaxIdx,
    }));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "force ∈ [0, 1]", domain: [0, 1], grid: true },
      y: { label: "Q(force)", domain: [0, 1.08], grid: true },
      marks: [
        // Bucket step rectangles (filled, semi-transparent).
        Plot.rect(stepRects, {
          x1: "x1", x2: "x2",
          y1: 0, y2: "y",
          fill: (d) => d.isBest ? palette.secondary : palette.muted,
          fillOpacity: (d) => d.isBest ? 0.45 : 0.18,
          stroke: palette.muted, strokeOpacity: 0.5,
        }),
        // True continuous Q*.
        Plot.line(curve, {
          x: "force", y: "q",
          stroke: palette.primary, strokeWidth: 2.2,
        }),
        // True optimum: vertical dashed at peak.
        Plot.ruleX([peak], { stroke: palette.primary, ...dashed }),
        // Best-bucket selection: vertical dashed at centroid.
        Plot.ruleX([bestBucketForce], { stroke: palette.secondary, ...dashed }),
        // Regret arrow region: shaded slice from bestBucketQ to optimum at peak.
        Plot.ruleX([peak], {
          stroke: palette.danger, strokeWidth: 3,
          y1: bestBucketQ, y2: optimalQ,
        }),
        Plot.text(
          [{ x: peak, y: (bestBucketQ + optimalQ) / 2, label: `regret ${regret.toFixed(3)}` }],
          { x: "x", y: "y", text: "label",
            fill: palette.danger, textAnchor: "start", dx: 6, ...annotation },
        ),
        Plot.text(
          [{ x: peak, y: optimalQ, label: `force* = ${peak.toFixed(2)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "middle", dy: -8,
            fill: palette.primary, ...annotation },
        ),
        Plot.text(
          [{ x: bestBucketForce, y: bestBucketQ, label: `bucket ${argmaxIdx}` }],
          { x: "x", y: "y", text: "label", textAnchor: "middle", dy: -6,
            fill: palette.secondary, ...annotation },
        ),
      ],
    }));
    slots.readout.textContent =
      `B = ${B}   ·   bucket width = ${(1 / B).toFixed(3)}   ·   ` +
      `argmax bucket force = ${bestBucketForce.toFixed(3)}   ·   ` +
      `regret = ${regret.toFixed(4)}`;
  },
});

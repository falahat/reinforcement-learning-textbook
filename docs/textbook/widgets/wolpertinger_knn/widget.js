// Widget 18.E — Wolpertinger k-NN retrieval demo (Chapter 18).
//
// 2D action-embedding plane with N = 100 discrete actions scattered as
// points (fixed seed). The actor's "proto-action" is a point the reader
// places via two sliders (proto_x, proto_y). The widget:
//   1. computes the k nearest discrete actions (Euclidean distance in
//      the embedding plane) — Wolpertinger's stage-1 retrieval,
//   2. evaluates a critic Q(a) on the k candidates and shows the
//      argmax — stage-2 re-rank.
// The critic is rendered as a smooth background heatmap so the reader
// can see "the actor placed proto-action near here, but the critic
// pulls the choice toward this brighter region of embedding space."
//
// Slider for k lets the reader see k=1 ("greedy, rigid — always picks
// the nearest discrete action") vs. larger k ("critic can override
// the actor's nearest neighbour").
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-wolpertinger-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/wolpertinger_knn/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

// N discrete actions sprinkled across [-3, 3]² with a couple of
// clusters so the k-NN behaviour is visually meaningful.
function buildActions(n, seed) {
  const rng = lcg(seed);
  const actions = [];
  for (let i = 0; i < n; i++) {
    // 3 cluster centres + noise; or pure uniform for a fraction.
    const r = rng();
    let cx, cy;
    if (r < 0.35) { cx = -1.4; cy = -1.2; }
    else if (r < 0.65) { cx = 1.6; cy = 0.6; }
    else if (r < 0.85) { cx = -0.2; cy = 1.8; }
    else { cx = 0; cy = 0; } // diffuse background
    const x = cx + (rng() - 0.5) * 2.4;
    const y = cy + (rng() - 0.5) * 2.4;
    actions.push({ id: i, x, y });
  }
  return actions;
}

// Critic Q(a): smooth bumpy field — two positive bumps and one
// negative dip — so re-ranking actually matters.
function critic(x, y) {
  const bump = (cx, cy, s) => Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
  return 1.0 * bump(1.6, 0.6, 0.9)
       + 0.7 * bump(-0.2, 1.8, 0.7)
       - 0.5 * bump(-1.4, -1.2, 0.8);
}

const ACTIONS = buildActions(100, 12345);

defineWidget({
  hostId: "ch18-wolpertinger-widget",
  controls: {
    proto_x: { label: "proto x", min: -3, max: 3, step: 0.05, default: -1.0 },
    proto_y: { label: "proto y", min: -3, max: 3, step: 0.05, default: -0.6 },
    k:       { label: "k (neighbours)", min: 1, max: 20, step: 1, default: 5 },
  },
  render: (host, { proto_x, proto_y, k }, slots) => {
    // Stage 1: k-NN around the proto-action.
    const withDist = ACTIONS.map((a) => ({
      ...a, dist: Math.hypot(a.x - proto_x, a.y - proto_y),
    }));
    withDist.sort((a, b) => a.dist - b.dist);
    const K = Math.round(k);
    const candidates = withDist.slice(0, K);
    // Stage 2: critic re-rank.
    const scored = candidates.map((a) => ({ ...a, q: critic(a.x, a.y) }));
    let argmax = scored[0];
    for (const a of scored) if (a.q > argmax.q) argmax = a;

    // Heatmap background: dense grid of critic values.
    const G = 40;
    const heat = [];
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x = -3 + 6 * (i + 0.5) / G;
        const y = -3 + 6 * (j + 0.5) / G;
        heat.push({ x, y, q: critic(x, y) });
      }
    }

    // Selected vs candidate vs other markers.
    const candidateIds = new Set(candidates.map((a) => a.id));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 420,
      width: 480,
      marginLeft: 50,
      x: { label: "embedding dim 1", domain: [-3, 3] },
      y: { label: "embedding dim 2", domain: [-3, 3] },
      color: {
        scheme: "viridis",
        domain: [-0.5, 1.0],
        legend: true,
        label: "critic Q",
      },
      marks: [
        Plot.cell(heat, {
          x: "x", y: "y", fill: "q",
          inset: 0,
          // Plot.cell uses ordinal channels by default — coerce to
          // numeric so the grid renders continuously.
          width: 6 / G, height: 6 / G,
        }),
        // Non-candidate actions: small grey dots.
        Plot.dot(
          ACTIONS.filter((a) => !candidateIds.has(a.id)),
          { x: "x", y: "y", fill: palette.muted, fillOpacity: 0.55, r: 2 },
        ),
        // k-NN candidates: yellow circles, sized by 1/(1+dist).
        Plot.dot(candidates, {
          x: "x", y: "y", stroke: palette.warning, strokeWidth: 2,
          fill: "white", fillOpacity: 0.0, r: 6,
        }),
        // Argmax over candidates: green ring.
        Plot.dot([argmax], {
          x: "x", y: "y", stroke: palette.primary, strokeWidth: 3,
          fill: palette.primary, fillOpacity: 0.35, r: 10,
        }),
        // Proto-action: red X.
        Plot.dot([{ x: proto_x, y: proto_y }], {
          x: "x", y: "y", symbol: "cross", stroke: palette.danger,
          strokeWidth: 3, r: 8,
        }),
        // Line from proto-action to selected action.
        Plot.link([{ x1: proto_x, y1: proto_y, x2: argmax.x, y2: argmax.y }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.primary, ...dashed,
        }),
      ],
    }));

    // Distance proto -> selected; distance proto -> nearest neighbour.
    const nearest = withDist[0];
    const overrideOccurred = argmax.id !== nearest.id;
    slots.readout.innerHTML =
      `k = ${K}   ·   proto = (${proto_x.toFixed(2)}, ${proto_y.toFixed(2)})   ·   ` +
      `selected action #${argmax.id} (Q = ${argmax.q.toFixed(3)})   ·   ` +
      `nearest #${nearest.id} (Q = ${critic(nearest.x, nearest.y).toFixed(3)})   ·   ` +
      `<small>${overrideOccurred ? "critic overrode the nearest neighbour" : "critic agrees with nearest"}</small>`;
  },
});

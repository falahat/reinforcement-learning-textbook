// Widget 11.F — Actor / critic / entropy loss decomposition (Chapter 11).
//
// The PPO total loss (§11.4) is a weighted sum of three terms:
//
//   L = − L_clip(θ)  +  c_v · L_V(φ)  −  c_h · H(π_θ)
//       ────────────     ──────────       ──────────────
//        actor (PG)       critic (TD)     entropy bonus
//
// Each term pulls the parameters in a different direction:
//   • L_clip wants to *increase* expected advantage-weighted log π.
//   • L_V    wants to *decrease* the value-prediction MSE.
//   • H(π)   wants to *increase* policy entropy (encourages exploration).
//
// The widget simulates a single batch's contributions. Sliders set the
// per-term magnitudes (|L_clip|, |L_V|, H) and the weights c_v, c_h.
// Stacked bar plot shows the signed contribution of each term to the
// total — letting the reader build intuition for "if I bump c_v from 0.5
// to 5.0, the critic term will dominate and the actor will move slowly."
//
// A second panel runs a tiny gradient-descent simulation over the
// weighted loss with the chosen weights, illustrating how the total
// trajectory differs from any single-term optimisation.
//
// Pedagogy: textbook §11.4 prints the loss as one line of pseudocode.
// Readers often skip past the coefficients without internalising that
// they encode a *priority ordering* between three competing objectives.
//
// Pattern:
//
//     <div id="ch11-loss-decomp-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/ac_loss_decomp/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

defineWidget({
  hostId: "ch11-loss-decomp-widget",
  controls: {
    lClip:   { label: "|L_clip| (actor)", min: 0.0, max: 2.0, step: 0.01, default: 0.4 },
    lV:      { label: "L_V (critic MSE)", min: 0.0, max: 2.0, step: 0.01, default: 0.6 },
    H:       { label: "H(π) (entropy)",   min: 0.0, max: 2.0, step: 0.01, default: 0.8 },
    cV:      { label: "c_v (critic wt)",  min: 0.0, max: 5.0, step: 0.05, default: 0.5 },
    cH:      { label: "c_h (entropy wt)", min: 0.0, max: 0.5, step: 0.005, default: 0.01 },
  },
  slots: ["bars", "trajectory"],
  render: (host, { lClip, lV, H, cV, cH }, slots) => {
    // Signed contributions to the total objective.
    // PPO minimises  −L_clip + c_v L_V − c_h H.
    const actorContrib   = -lClip;
    const criticContrib  =  cV * lV;
    const entropyContrib = -cH * H;
    const total = actorContrib + criticContrib + entropyContrib;

    const rows = [
      { term: "−L_clip (actor)",    value: actorContrib,   colour: palette.primary },
      { term: "+c_v · L_V (critic)", value: criticContrib,  colour: palette.secondary },
      { term: "−c_h · H (entropy)",  value: entropyContrib, colour: palette.accent },
      { term: "TOTAL L",             value: total,          colour: palette.warning },
    ];

    const vals = rows.map((r) => r.value);
    const yLo = Math.min(0, ...vals) * 1.2 - 0.1;
    const yHi = Math.max(0, ...vals) * 1.2 + 0.1;

    slots.bars.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      marginBottom: 60,
      x: { label: null, domain: rows.map((r) => r.term) },
      y: { label: "contribution", domain: [yLo, yHi], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.6 }),
        Plot.barY(rows, {
          x: "term",
          y: "value",
          fill: (d) => d.colour,
          fillOpacity: 0.8,
        }),
        Plot.text(rows, {
          x: "term",
          y: "value",
          text: (d) => d.value.toFixed(3),
          dy: (d) => (d.value >= 0 ? -8 : 14),
          fill: "#ddd",
          fontSize: 11,
          fontWeight: "bold",
          textAnchor: "middle",
        }),
      ],
    }));

    // Trajectory panel: simulate optimising the weighted sum starting
    // from a 2D parameter ("actor" axis = θ_a, "critic" axis = θ_c).
    // Model each loss term as a quadratic well centred at a
    // term-specific optimum, scaled by the slider value.
    //
    //   L = −|L_clip| · exp(−((θ_a − 1)² + θ_c²)/2)    (actor pulls θ_a→1)
    //     + c_v · |L_V| · ((θ_a + 0)² + (θ_c − 1)²) / 4 (critic pulls θ_c→1)
    //     − c_h · |H|   · exp(−(θ_a² + θ_c²)/8)         (entropy pulls origin)
    //
    // Toy but captures "three forces competing in parameter space".
    const lossAt = (a, c) => {
      const actor   = -lClip * Math.exp(-((a - 1) ** 2 + c ** 2) / 2);
      const critic  =  cV * lV * ((a + 0) ** 2 + (c - 1) ** 2) / 4;
      const entropy = -cH * H * Math.exp(-(a ** 2 + c ** 2) / 8);
      return actor + critic + entropy;
    };
    const gradAt = (a, c) => {
      const h = 1e-3;
      return [
        (lossAt(a + h, c) - lossAt(a - h, c)) / (2 * h),
        (lossAt(a, c + h) - lossAt(a, c - h)) / (2 * h),
      ];
    };

    // Gradient descent from (−1.5, −1.5).
    let a = -1.5, c = -1.5;
    const lr = 0.1;
    const traj = [{ a, c, k: 0, L: lossAt(a, c) }];
    for (let k = 1; k <= 80; k++) {
      const [ga, gc] = gradAt(a, c);
      a -= lr * ga;
      c -= lr * gc;
      traj.push({ a, c, k, L: lossAt(a, c) });
    }

    // Contour grid.
    const GRID = 40;
    const lo = -2.5, hi = 2.5;
    const grid = [];
    for (let i = 0; i <= GRID; i++) {
      for (let j = 0; j <= GRID; j++) {
        const x = lo + (hi - lo) * (i / GRID);
        const y = lo + (hi - lo) * (j / GRID);
        grid.push({ x, y, z: lossAt(x, y) });
      }
    }
    const zs = grid.map((d) => d.z).sort((a, b) => a - b);
    const zLo = zs[Math.floor(zs.length * 0.05)];
    const zHi = zs[Math.floor(zs.length * 0.95)];
    const thresholds = d3.range(10).map((i) => zLo + (zHi - zLo) * (i / 9));

    slots.trajectory.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      width: 360,
      aspectRatio: 1,
      x: { label: "θ_actor", domain: [lo, hi] },
      y: { label: "θ_critic", domain: [lo, hi] },
      marks: [
        Plot.contour(grid, {
          x: "x", y: "y", value: "z",
          thresholds,
          stroke: palette.muted, strokeOpacity: 0.45, strokeWidth: 0.7,
        }),
        // Reference markers for each term's pull point.
        Plot.dot([{ x: 1, y: 0 }], { x: "x", y: "y", fill: palette.primary,   r: 4, stroke: "white" }),
        Plot.dot([{ x: 0, y: 1 }], { x: "x", y: "y", fill: palette.secondary, r: 4, stroke: "white" }),
        Plot.dot([{ x: 0, y: 0 }], { x: "x", y: "y", fill: palette.accent,    r: 4, stroke: "white" }),
        // Trajectory.
        Plot.line(traj, { x: "a", y: "c", stroke: palette.warning, strokeWidth: 1.5 }),
        Plot.dot(traj, { x: "a", y: "c", fill: palette.warning, r: 1.7 }),
        Plot.dot([traj[0]], { x: "a", y: "c", fill: "white", r: 5, stroke: palette.warning, strokeWidth: 2 }),
        Plot.dot([traj[traj.length - 1]], {
          x: "a", y: "c", fill: palette.danger, r: 5, stroke: "white",
        }),
        Plot.text(
          [
            { x: 1, y: 0, label: "actor pull" },
            { x: 0, y: 1, label: "critic pull" },
            { x: 0, y: 0, label: "entropy pull" },
          ],
          { x: "x", y: "y", text: "label", dx: 8, dy: -8,
            fill: "#bbb", ...annotation },
        ),
      ],
    }));

    const endA = traj[traj.length - 1].a;
    const endC = traj[traj.length - 1].c;
    const lastTerm =
      Math.abs(actorContrib) > Math.abs(criticContrib) &&
      Math.abs(actorContrib) > Math.abs(entropyContrib)
        ? "actor"
        : Math.abs(criticContrib) > Math.abs(entropyContrib)
          ? "critic"
          : "entropy";
    slots.readout.textContent =
      `L = ${total.toFixed(3)}   |   dominant: ${lastTerm}   |   ` +
      `convergence (θ_a, θ_c) ≈ (${endA.toFixed(2)}, ${endC.toFixed(2)})`;
  },
});

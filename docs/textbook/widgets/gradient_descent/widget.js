// Widget 1.C — Gradient descent tracer (Chapter 1, §1.3 / §1.5).
//
// Drops a marker on a 2D loss surface and runs GD x_{k+1} = x_k − η ∇f(x_k)
// for a few dozen steps. The student picks the surface (convex bowl,
// Rosenbrock "banana", saddle), the learning rate η, and the start point.
// Two plots:
//   1. Loss surface contour map with the GD trajectory overlaid as a
//      polyline + dots.
//   2. f(x_k) vs step k. Convex bowl shows monotone decrease; banana
//      shows a long zig-zag; saddle shows escape (or stuck at the saddle
//      if η = 0 or start exactly on it).
//
// All three surfaces have closed-form gradients — no autodiff needed.
//
// Mount: `<div id="ch1-gradient-descent-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/gradient_descent/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

// Each surface: { f(x,y), grad(x,y) → [gx, gy], domain: [lo, hi], note }.
const SURFACES = {
  bowl: {
    label: "Convex bowl  f = ½(x² + 4y²)",
    f: (x, y) => 0.5 * (x * x + 4 * y * y),
    grad: (x, y) => [x, 4 * y],
    domain: [-3, 3],
    note: "Convex quadratic with κ=4: zig-zags down the long axis.",
  },
  banana: {
    label: "Rosenbrock  f = (1−x)² + 5(y−x²)²",
    f: (x, y) => Math.pow(1 - x, 2) + 5 * Math.pow(y - x * x, 2),
    grad: (x, y) => [
      -2 * (1 - x) - 20 * x * (y - x * x),
      10 * (y - x * x),
    ],
    domain: [-2, 2],
    note: "Curved valley along y=x². GD crawls along the floor.",
  },
  saddle: {
    label: "Saddle  f = x² − y²",
    f: (x, y) => x * x - y * y,
    grad: (x, y) => [2 * x, -2 * y],
    domain: [-2, 2],
    note: "Min along x, max along y. GD escapes off the y-axis.",
  },
};

const MAX_STEPS = 60;

defineWidget({
  hostId: "ch1-gradient-descent-widget",
  controls: {
    surface: {
      type: "select", label: "loss surface",
      options: Object.entries(SURFACES).map(([k, s]) => ({ value: k, label: s.label })),
      default: "bowl",
    },
    eta:   { label: "η (lr)", min: 0.001, max: 0.5, step: 0.001, default: 0.1 },
    x0:    { label: "x₀",     min: -2, max: 2, step: 0.05, default: 1.5 },
    y0:    { label: "y₀",     min: -2, max: 2, step: 0.05, default: 1.2 },
    steps: { label: "steps",  min: 1, max: MAX_STEPS, step: 1, default: 30 },
  },
  slots: ["contour", "loss"],
  render: (host, { surface, eta, x0, y0, steps }, slots) => {
    const sur = SURFACES[surface] ?? SURFACES.bowl;
    const [lo, hi] = sur.domain;

    // Run GD.
    const traj = [{ x: x0, y: y0, f: sur.f(x0, y0), k: 0 }];
    let x = x0, y = y0, diverged = false;
    for (let k = 1; k <= steps; k++) {
      const [gx, gy] = sur.grad(x, y);
      x -= eta * gx;
      y -= eta * gy;
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e6) {
        diverged = true;
        break;
      }
      traj.push({ x, y, f: sur.f(x, y), k });
    }

    // Contour grid: 60×60 samples → use Plot's contour mark.
    const GRID = 60;
    const grid = [];
    for (let i = 0; i <= GRID; i++) {
      for (let j = 0; j <= GRID; j++) {
        const gx = lo + (hi - lo) * (i / GRID);
        const gy = lo + (hi - lo) * (j / GRID);
        grid.push({ x: gx, y: gy, z: sur.f(gx, gy) });
      }
    }
    // Pick contour thresholds spaced along the (clipped) z-range.
    const zs = grid.map((d) => d.z).sort((a, b) => a - b);
    const zLo = zs[Math.floor(zs.length * 0.02)];
    const zHi = zs[Math.floor(zs.length * 0.92)];
    const thresholds = d3.range(10).map((i) => zLo + (zHi - zLo) * (i / 9));

    slots.contour.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      width: 360,
      x: { domain: [lo, hi], label: "x", grid: false },
      y: { domain: [lo, hi], label: "y", grid: false },
      aspectRatio: 1,
      marks: [
        Plot.contour(grid, {
          x: "x", y: "y", value: "z",
          thresholds,
          stroke: palette.muted, strokeOpacity: 0.5, strokeWidth: 0.8,
        }),
        // Trajectory polyline.
        Plot.line(traj, { x: "x", y: "y", stroke: palette.primary, strokeWidth: 1.5 }),
        Plot.dot(traj, { x: "x", y: "y", fill: palette.primary, r: 2 }),
        // Start (warning) + end (danger) markers.
        Plot.dot([traj[0]], { x: "x", y: "y", fill: palette.warning, r: 5, stroke: "white" }),
        Plot.dot([traj[traj.length - 1]], { x: "x", y: "y", fill: palette.danger, r: 5, stroke: "white" }),
      ],
    }));

    const lossDomain = traj.map((d) => d.f);
    const lossMin = Math.min(...lossDomain);
    const lossMax = Math.max(...lossDomain);
    slots.loss.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      width: 360,
      x: { label: "step k", grid: true },
      y: {
        label: "f(x_k)", grid: true,
        // Log-scale only when all losses positive and varied.
        type: lossMin > 1e-6 && lossMax / lossMin > 50 ? "log" : "linear",
      },
      marks: [
        Plot.line(traj, { x: "k", y: "f", stroke: palette.secondary, strokeWidth: 1.5 }),
        Plot.dot(traj, { x: "k", y: "f", fill: palette.secondary, r: 2.5 }),
        Plot.ruleY([0], { stroke: palette.muted, ...dashed }),
      ],
    }));

    const last = traj[traj.length - 1];
    const tag = diverged ? " <strong style='color:#e57373'>diverged</strong>" : "";
    slots.readout.innerHTML =
      `<small>${sur.note}</small><br>` +
      `step ${last.k}: x = (${fmt(last.x)}, ${fmt(last.y)}), ` +
      `f = ${fmt(last.f)} (start f = ${fmt(traj[0].f)})${tag}`;
  },
});

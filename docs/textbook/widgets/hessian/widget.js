// Widget 1.D — Hessian curvature painter (Chapter 1).
//
// At a chosen point (x, y) on a 2-D loss surface, compute the gradient
// and Hessian analytically; visualise:
//   - the loss contour lines (background)
//   - the negative gradient -∇f (yellow arrow) — the GD direction
//   - the Newton step -H⁻¹ ∇f (blue arrow) — the Newton direction
//   - the Hessian eigenvectors as green semi-axes of the quadratic-form
//     ellipse vᵀ H v = 1, with semi-axis lengths 1/√|λᵢ|
//
// Pedagogical point: the Hessian's eigenvalues are the local curvatures
// in each principal direction; the Newton direction warps the gradient
// by H⁻¹ so the step lands at the bowl's bottom in one shot. On a saddle
// (Rosenbrock + a few points; explicit saddle preset) the Newton step
// can point UPHILL because H is indefinite — that's the bug Newton's
// method has on non-convex problems.
//
// Surfaces (all 2-D):
//   - bowl     : f = 0.5 (x² + 3 y²)              — convex, anisotropic
//   - rosenbrock : f = (1-x)² + 100 (y - x²)²     — banana valley
//   - saddle   : f = x² - y²                      — indefinite Hessian
//
// Mount:
//   <div id="ch1-hessian-widget" class="textbook-widget"></div>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, annotation, fmt } from "../shared/helpers.js";
import { eig2x2, inv2x2, matvec2x2 } from "../shared/linalg.js";

// Surfaces: return { f, grad, hess } for the chosen point.
const SURFACES = {
  bowl: {
    label: "convex bowl",
    domain: [-3, 3, -3, 3],
    f: (x, y) => 0.5 * (x * x + 3 * y * y),
    grad: (x, y) => [x, 3 * y],
    hess: (_x, _y) => [[1, 0], [0, 3]],
    note: "Diagonal Hessian; Newton direction = scaled gradient.",
  },
  rosenbrock: {
    label: "Rosenbrock banana",
    domain: [-2, 2, -1, 3],
    f: (x, y) => (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
    grad: (x, y) => [
      -2 * (1 - x) - 400 * x * (y - x * x),
      200 * (y - x * x),
    ],
    hess: (x, y) => [
      [2 - 400 * (y - x * x) + 800 * x * x, -400 * x],
      [-400 * x, 200],
    ],
    note: "Curved valley; Newton turns the corner, GD oscillates.",
  },
  saddle: {
    label: "saddle x² − y²",
    domain: [-2, 2, -2, 2],
    f: (x, y) => x * x - y * y,
    grad: (x, y) => [2 * x, -2 * y],
    hess: (_x, _y) => [[2, 0], [0, -2]],
    note: "Indefinite Hessian; Newton step is meaningless (or worse).",
  },
};

function ellipsePoints(centre, eigs, N = 80) {
  // Quadratic-form level set { v : (v-c)ᵀ H (v-c) = 1 } is an ellipse
  // with principal axes the eigenvectors of H and semi-axes 1/√|λᵢ|.
  const [cx, cy] = centre;
  const a = 1 / Math.sqrt(Math.abs(eigs[0].lambda));
  const b = 1 / Math.sqrt(Math.abs(eigs[1].lambda));
  const u = eigs[0].v;
  const w = eigs[1].v;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * 2 * Math.PI;
    const ax = a * Math.cos(t);
    const ay = b * Math.sin(t);
    const x = cx + ax * u[0] + ay * w[0];
    const y = cy + ax * u[1] + ay * w[1];
    pts.push({ x, y });
  }
  return pts;
}

defineWidget({
  hostId: "ch1-hessian-widget",
  controls: {
    surface: {
      type: "select",
      label: "surface",
      options: [
        { value: "bowl", label: "convex bowl" },
        { value: "rosenbrock", label: "Rosenbrock" },
        { value: "saddle", label: "saddle x²−y²" },
      ],
      default: "bowl",
    },
    x: { label: "x", min: -2, max: 2, step: 0.05, default: 1.0 },
    y: { label: "y", min: -2, max: 2, step: 0.05, default: 1.0 },
  },
  render: (host, { surface, x, y }, slots) => {
    const surf = SURFACES[surface];
    const [xmin, xmax, ymin, ymax] = surf.domain;
    // Clamp the point to the chosen surface's domain so the ellipse and
    // arrows stay on-frame.
    const px = Math.max(xmin + 0.1, Math.min(xmax - 0.1, x));
    const py = Math.max(ymin + 0.1, Math.min(ymax - 0.1, y));

    const g = surf.grad(px, py);
    const H = surf.hess(px, py);
    const { real, eigs } = eig2x2(H);

    // Sample f on a grid for contour rendering.
    const N = 40;
    const cells = [];
    for (let i = 0; i < N; i++) {
      const gx = xmin + (i / (N - 1)) * (xmax - xmin);
      for (let j = 0; j < N; j++) {
        const gy = ymin + (j / (N - 1)) * (ymax - ymin);
        cells.push({ x: gx, y: gy, f: surf.f(gx, gy) });
      }
    }

    // -∇f (gradient descent direction).
    const gnorm = Math.hypot(g[0], g[1]);
    const arrowScale = 0.4 * Math.min(xmax - xmin, ymax - ymin) / Math.max(gnorm, 1e-3);
    const gdEnd = [px - g[0] * arrowScale * 0.5, py - g[1] * arrowScale * 0.5];

    // Newton step -H⁻¹∇f, only when H is invertible.
    const Hinv = inv2x2(H);
    let newtonEnd = null;
    if (Hinv) {
      const step = matvec2x2(Hinv, g);
      newtonEnd = [px - step[0], py - step[1]];
    }

    // Quadratic-form ellipse (only real, well-conditioned eigs).
    const drawEllipse =
      real && Math.abs(eigs[0].lambda) > 1e-6 && Math.abs(eigs[1].lambda) > 1e-6;
    const ellPts = drawEllipse ? ellipsePoints([px, py], eigs) : [];

    // Eigenvector axes: from centre out by 1/√|λᵢ| along each.
    const axes = [];
    if (drawEllipse) {
      for (const e of eigs) {
        const s = 1 / Math.sqrt(Math.abs(e.lambda));
        axes.push({
          x1: px, y1: py,
          x2: px + e.v[0] * s, y2: py + e.v[1] * s,
        });
        axes.push({
          x1: px, y1: py,
          x2: px - e.v[0] * s, y2: py - e.v[1] * s,
        });
      }
    }

    const arrows = [
      { x1: px, y1: py, x2: gdEnd[0], y2: gdEnd[1], colour: palette.warning, label: "−∇f" },
    ];
    if (newtonEnd) {
      arrows.push({
        x1: px, y1: py, x2: newtonEnd[0], y2: newtonEnd[1],
        colour: palette.secondary, label: "−H⁻¹∇f",
      });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 460,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [xmin, xmax], label: "x", grid: true },
      y: { domain: [ymin, ymax], label: "y", grid: true },
      color: { type: "log", scheme: "blues", legend: false },
      marks: [
        Plot.contour(cells, {
          x: "x", y: "y", value: "f",
          stroke: palette.muted, strokeOpacity: 0.5, thresholds: 12,
        }),
        // Quadratic-form ellipse (green).
        drawEllipse
          ? Plot.line(ellPts, { x: "x", y: "y", stroke: palette.primary, strokeWidth: 2 })
          : null,
        // Hessian eigenvector axes (dashed green).
        axes.length
          ? Plot.link(axes, {
              x1: "x1", y1: "y1", x2: "x2", y2: "y2",
              stroke: palette.primary, strokeWidth: 1.5, strokeDasharray: "3 3",
            })
          : null,
        // Chosen point marker.
        Plot.dot([{ x: px, y: py }], { x: "x", y: "y", r: 4, fill: "white", stroke: "black" }),
        // GD + Newton arrows.
        ...arrows.flatMap((a) => [
          Plot.arrow([a], {
            x1: "x1", y1: "y1", x2: "x2", y2: "y2",
            stroke: a.colour, strokeWidth: 2.5, headLength: 10,
          }),
          Plot.text([{ x: a.x2, y: a.y2, label: a.label }], {
            x: "x", y: "y", text: "label", dx: 6, dy: -4,
            fill: a.colour, ...annotation,
          }),
        ]),
      ].filter(Boolean),
    }));

    const lam1 = real ? eigs[0].lambda : NaN;
    const lam2 = real ? eigs[1].lambda : NaN;
    const cond = real && Math.abs(lam2) > 1e-9
      ? Math.abs(lam1 / lam2)
      : NaN;
    slots.readout.textContent =
      `∇f = (${fmt(g[0])}, ${fmt(g[1])})  ·  ` +
      `H = (${fmt(H[0][0])}, ${fmt(H[0][1])}; ${fmt(H[1][0])}, ${fmt(H[1][1])})  ·  ` +
      `λ₁ = ${fmt(lam1)}, λ₂ = ${fmt(lam2)}  ·  ` +
      `κ = ${fmt(cond)}  ·  ${surf.note}`;
  },
});

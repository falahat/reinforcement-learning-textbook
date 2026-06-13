// Widget 1.N — Quadratic bowl: eigenvectors as bowl axes (Chapter 1, §1.8).
//
// For a symmetric 2×2 matrix A = [[a, b], [b, c]], the quadratic form
//     f(v) = ½ vᵀ A v
// has contour lines that are ellipses (or hyperbolae, or parallel lines)
// whose AXES are the eigenvectors of A. The semi-axis lengths are
// 1/√λ_i (for level f = ½). This widget draws:
//   - the contour grid of f
//   - the two eigenvectors as arrows from origin
//   - colour-by-sign of the eigenvalues (positive: bowl up;
//     negative: bowl down; mixed: saddle)
//
// Pedagogical points:
//   - PD matrices (both λᵢ > 0): every direction curves up → bowl.
//   - PSD with one λ = 0: bowl flat in one direction (trough).
//   - One positive, one negative λ: saddle.
//   - All negative: upside-down bowl (max instead of min).
// All three cases are reachable by edit; the readout names them.
//
// Mount: `<div id="ch1-quadratic-bowl-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor } from "../shared/matrix_editor.js";
import { eig2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-quadratic-bowl-widget";

const SCAFFOLD_HTML = `
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (symmetric; b auto-mirrored)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="readouts"></div>
    </div>
    <div data-plot="main"></div>
  </div>
  <div data-readout></div>
`;

function classifyEigs(eigs) {
  const pos = eigs.filter((e) => e.lambda > 1e-6).length;
  const neg = eigs.filter((e) => e.lambda < -1e-6).length;
  if (pos === 2) return { kind: "PD",   label: "positive-definite (bowl)",        color: palette.primary };
  if (neg === 2) return { kind: "ND",   label: "negative-definite (inverted bowl)", color: palette.warning };
  if (pos === 1 && neg === 1) return { kind: "saddle", label: "indefinite (saddle)", color: palette.danger };
  if (pos === 2 || neg === 2) return { kind: "??", label: "unknown", color: palette.muted };
  return { kind: "PSD", label: "positive-semi-definite (flat trough)", color: palette.accent };
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  // Start with PD example (rotated bowl).
  const A = [[2, 0.5], [0.5, 1]];

  function render() {
    // Auto-symmetrize: keep A[0][1] = A[1][0]. Mirror after edit by
    // taking the average.
    const mid = (A[0][1] + A[1][0]) / 2;
    A[0][1] = mid;
    A[1][0] = mid;

    const { real, eigs } = eig2x2(A);
    const cls = classifyEigs(eigs);

    // Sample f(v) on a grid for contour rendering.
    const R = 3;
    const N = 50;
    const cells = [];
    for (let i = 0; i < N; i++) {
      const x = -R + (i / (N - 1)) * 2 * R;
      for (let j = 0; j < N; j++) {
        const y = -R + (j / (N - 1)) * 2 * R;
        const f = 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y);
        cells.push({ x, y, f });
      }
    }

    // Eigenvector arrows scaled by sqrt(|λ|) for visibility.
    const eigArrows = (real ? eigs : []).map((e, i) => {
      const scale = Math.max(0.5, Math.sqrt(Math.abs(e.lambda)));
      return {
        x1: 0, y1: 0,
        x2: e.v[0] * scale, y2: e.v[1] * scale,
        label: `λ${i + 1} = ${fmt(e.lambda)}`,
        colour: e.lambda >= 0 ? palette.warning : palette.danger,
      };
    });

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 400, width: 420,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [-R, R], label: "v.x", grid: true },
      y: { domain: [-R, R], label: "v.y", grid: true },
      color: {
        type: "diverging", scheme: "rdbu",
        domain: [-Math.max(...cells.map((c) => Math.abs(c.f))), Math.max(...cells.map((c) => Math.abs(c.f)))],
        legend: true,
        label: "f(v) = ½ vᵀA v",
      },
      marks: [
        // Heatmap of f values.
        Plot.raster(cells, { x: "x", y: "y", fill: "f", interpolate: "nearest" }),
        // Contour lines.
        Plot.contour(cells, { x: "x", y: "y", value: "f", stroke: palette.muted, strokeOpacity: 0.4, thresholds: 10 }),
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Eigenvector arrows.
        ...eigArrows.flatMap((e) => [
          Plot.arrow([{ x1: e.x1, y1: e.y1, x2: e.x2, y2: e.y2 }], {
            x1: "x1", y1: "y1", x2: "x2", y2: "y2",
            stroke: e.colour, strokeWidth: 3, headLength: 12,
          }),
          Plot.text([{ x: e.x2, y: e.y2, label: e.label }], {
            x: "x", y: "y", text: "label", dx: 6, dy: -4,
            fill: e.colour, ...annotation,
          }),
        ]),
      ],
    }));

    const condition = real && Math.abs(eigs[1].lambda) > 1e-9
      ? Math.abs(eigs[0].lambda / eigs[1].lambda)
      : NaN;
    host.querySelector('[data-display="readouts"]').innerHTML = `
      <div class="widget-matrix-title">readouts</div>
      <div class="widget-matrix-display">
        ${eigs.map((e, i) => `<div>λ${i + 1} = ${fmt(e.lambda)}</div>`).join("")}
        <div>tr A = ${fmt(A[0][0] + A[1][1])}</div>
        <div>det A = ${fmt(A[0][0] * A[1][1] - A[0][1] * A[1][0])}</div>
        ${isFinite(condition) ? `<div>κ(A) = |λ₁/λ₂| = ${fmt(condition)}</div>` : ""}
        <div style="color:${cls.color}"><b>${cls.label}</b></div>
      </div>
    `;
  }

  buildMatrixEditor(host, '[data-matrix="A"]', A, render, {
    step: 0.1,
    colHeaders: ["col 1", "col 2"],
  });
  render();
}

if (document.readyState !== "loading") mount();
else document.addEventListener("DOMContentLoaded", mount);

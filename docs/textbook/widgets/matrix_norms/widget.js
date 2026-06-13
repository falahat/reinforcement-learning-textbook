// Widget 1.K — Operator vs Frobenius vs nuclear norm (Chapter 1, §1.2).
//
// Editable 2×2 matrix A. The widget plots the image of the unit circle
// under A — an ellipse whose semi-axes are σ₁ ≥ σ₂, the singular values
// of A. Live readouts of the four matrix sizes the textbook table names:
//
//     ‖A‖_op  = σ₁              (largest singular value)
//     ‖A‖_F   = √(σ₁² + σ₂²)    (Frobenius — entrywise ℓ₂)
//     ‖A‖_*   = σ₁ + σ₂         (nuclear / trace norm)
//     |det A| = σ₁ · σ₂         (volume scale)
//
// Closed-form SVD for 2×2: from A^T A = V·Σ²·V^T we have
//     σ²_{1,2} = ½ (tr(A^TA) ± √(tr(A^TA)² − 4·det(A^TA))),
//     det(A^TA) = (det A)²,
//     tr(A^TA) = a²+b²+c²+d² = ‖A‖_F².
// The right singular vectors v_i are the (orthonormal) eigenvectors of
// A^TA; the ellipse axes are A·v_i. No iterative SVD library needed.
//
// Layout: the abstract 2-D plot on the left; an editable 2×2 matrix +
// singular-values panel + right-singular-vectors panel on the right.
// The plot's σ_i arrows carry a native SVG <title> tooltip telling the
// reader which singular value they correspond to.
//
// Mount: `<div id="ch1-matrix-norms-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/matrix_norms/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import {
  buildMatrixEditor,
  buildMatrixDisplay,
  buildVectorPanel,
} from "../shared/matrix_editor.js";
import { svd2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-matrix-norms-widget";

const SCAFFOLD_HTML = `
  <div class="widget-matrix-layout">
    <div data-plot="main"></div>
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (edit cells)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="sigma"></div>
      <div data-display="vrights"></div>
      <div data-display="norms"></div>
    </div>
  </div>
  <div data-readout></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  const A = [[2, 1], [0.5, 1.5]];

  function render() {
    const [[a, b], [c, d]] = A;

    const N = 96;
    const circle = d3.range(N + 1).map((i) => {
      const t = (2 * Math.PI * i) / N;
      const u = Math.cos(t), v = Math.sin(t);
      return { u, v, x: a * u + b * v, y: c * u + d * v };
    });

    const { sigma, v } = svd2x2([[a, b], [c, d]]);
    const [s1, s2] = sigma;
    const [v1, v2] = v;
    const axis1 = { x: a * v1[0] + b * v1[1], y: c * v1[0] + d * v1[1] };
    const axis2 = { x: a * v2[0] + b * v2[1], y: c * v2[0] + d * v2[1] };

    const allMag = circle.flatMap((p) => [p.x, p.y]).concat([1, -1]);
    const R = Math.max(1.2, Math.max(...allMag.map(Math.abs)) * 1.15);

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 360,
      width: 420,
      x: { domain: [-R, R], label: "x", grid: true },
      y: { domain: [-R, R], label: "y", grid: true },
      aspectRatio: 1,
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, ...dashed }),
        Plot.ruleY([0], { stroke: palette.muted, ...dashed }),
        Plot.line(circle, {
          x: "u", y: "v", stroke: palette.secondary,
          strokeOpacity: 0.7, strokeWidth: 1.5,
        }),
        Plot.line(circle, {
          x: "x", y: "y", stroke: palette.warning, strokeWidth: 2,
        }),
        // σ₁ semi-axis (longest) — with hover-tooltip.
        Plot.arrow([{ x1: 0, y1: 0, x2: axis1.x, y2: axis1.y,
                      title: `σ₁ semi-axis · σ₁ = ${fmt(s1)}` }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.primary, strokeWidth: 2.5, headLength: 8,
          title: "title",
        }),
        Plot.text([{ x: axis1.x, y: axis1.y, label: `σ₁=${fmt(s1)}` }], {
          x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: -4,
          fill: palette.primary, ...annotation,
        }),
        // σ₂ semi-axis (shortest) — with hover-tooltip.
        Plot.arrow([{ x1: 0, y1: 0, x2: axis2.x, y2: axis2.y,
                      title: `σ₂ semi-axis · σ₂ = ${fmt(s2)}` }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.accent, strokeWidth: 2.5, headLength: 8,
          title: "title",
        }),
        Plot.text([{ x: axis2.x, y: axis2.y, label: `σ₂=${fmt(s2)}` }], {
          x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: -4,
          fill: palette.accent, ...annotation,
        }),
      ],
    }));

    // Numerical singular values.
    buildMatrixDisplay(host, '[data-display="sigma"]', [[s1, s2]], {
      title: "singular values σ",
      colHeaders: ["σ₁", "σ₂"],
    });

    // Right singular vectors (rows = v₁, v₂; columns = x, y components).
    buildVectorPanel(host, '[data-display="vrights"]', [
      { label: "v₁", values: v1 },
      { label: "v₂", values: v2 },
    ], { title: "right singular vectors v" });

    const opNorm = s1;
    const frob = Math.sqrt(s1 * s1 + s2 * s2);
    const nuc = s1 + s2;
    const detA = a * d - b * c;

    // Norms summary as a one-column display.
    buildMatrixDisplay(host, '[data-display="norms"]',
      [[opNorm], [frob], [nuc], [detA]],
      {
        title: "norms",
        rowHeaders: ["‖A‖_op", "‖A‖_F", "‖A‖_*", "det A"],
      },
    );

    const frobFromEntries = Math.sqrt(a * a + b * b + c * c + d * d);
    host.querySelector("[data-readout]").innerHTML =
      `<small>Blue = unit circle, orange = its image under A. ` +
      `Green/purple arrows = σ₁, σ₂ semi-axes (hover the arrow for σ value). ` +
      `Entrywise ‖A‖_F = ${fmt(frobFromEntries)} matches √(σ₁²+σ₂²) = ${fmt(frob)}.</small>`;
  }

  buildMatrixEditor(host, '[data-matrix="A"]', A, () => render(), {
    step: 0.1,
    rowHeaders: ["row 1", "row 2"],
    colHeaders: ["col 1", "col 2"],
  });
  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

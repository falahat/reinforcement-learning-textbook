// Widget 1.M — Characteristic polynomial explorer (Chapter 1, §1.6).
//
// "Eigenvalues are the roots of det(A − λI) = 0" is the algebraic
// definition. This widget makes it visual: edit the 2×2 matrix A, see
// the characteristic polynomial p(λ) = λ² − (tr A)λ + det A plotted as
// a curve over λ. The two roots are the eigenvalues, marked with
// vertical lines whose colours match the eigenvectors widget.
//
// Crank the matrix toward a rotation (b = −c) and watch the discriminant
// go negative — the curve no longer crosses zero, and the readout
// reports "complex eigenvalues" matching the no-real-eigenvectors story
// from the eigenvector widget.
//
// Pedagogical points:
//   - "λ is an eigenvalue" ⇔ "λ is a root of p(λ)".
//   - p(λ) is a quadratic for 2×2; opens upward (leading coeff 1).
//   - Roots from quadratic formula: λ = (tr ± √(tr² − 4·det)) / 2.
//   - Discriminant < 0 ⇒ no real eigenvalues; the curve stays above 0.
//
// Mount: `<div id="ch1-char-poly-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor } from "../shared/matrix_editor.js";
import { eig2x2, trace2x2, det2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-char-poly-widget";

const SCAFFOLD_HTML = `
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (edit cells)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="readouts"></div>
    </div>
    <div data-plot="main"></div>
  </div>
  <div data-readout></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  const A = [[3, 1], [0, 2]];

  function render() {
    const tr = trace2x2(A);
    const det = det2x2(A);
    const { real, eigs } = eig2x2(A);

    // Sample the characteristic polynomial p(λ) = λ² − tr·λ + det
    // over a window that comfortably brackets the roots.
    const center = tr / 2;
    const half = Math.max(2.5, Math.abs(tr) / 2 + Math.sqrt(Math.abs(tr * tr - 4 * det)) + 1);
    const lo = center - half;
    const hi = center + half;
    const N = 200;
    const curve = d3.range(N + 1).map((i) => {
      const lam = lo + (i / N) * (hi - lo);
      const p = lam * lam - tr * lam + det;
      return { lam, p };
    });
    const pMin = Math.min(...curve.map((d) => d.p));
    const pMax = Math.max(...curve.map((d) => d.p));
    const pad = Math.max(0.5, (pMax - pMin) * 0.1);

    const eigLines = real
      ? eigs.map((e, i) => ({ lam: e.lambda, label: `λ${i + 1} = ${fmt(e.lambda)}`, colour: i === 0 ? palette.warning : palette.accent }))
      : [];

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 460,
      marginLeft: 50, marginBottom: 36,
      x: { domain: [lo, hi], label: "λ", grid: true },
      y: { domain: [Math.min(0, pMin) - pad, pMax + pad], label: "p(λ) = det(A − λI)", grid: true },
      marks: [
        // y = 0 (the root line).
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.7 }),
        // Curve.
        Plot.line(curve, {
          x: "lam", y: "p",
          stroke: palette.primary, strokeWidth: 2,
        }),
        // Vertex of the parabola at λ = tr/2 (helpful pedagogical mark).
        Plot.dot([{ lam: center, p: center * center - tr * center + det }], {
          x: "lam", y: "p", r: 3, fill: palette.muted, fillOpacity: 0.6,
        }),
        Plot.text([{ lam: center, p: center * center - tr * center + det, label: "vertex" }], {
          x: "lam", y: "p", text: "label",
          dx: 6, dy: 12, fill: palette.muted, ...annotation,
        }),
        // Vertical rules at the roots.
        ...eigLines.flatMap((e) => [
          Plot.ruleX([e.lam], { stroke: e.colour, strokeWidth: 2, ...dashed }),
          Plot.text([{ lam: e.lam, p: 0, label: e.label }], {
            x: "lam", y: "p", text: "label",
            dy: -8, fill: e.colour, ...annotation,
          }),
          Plot.dot([{ lam: e.lam, p: 0 }], { x: "lam", y: "p", r: 5, fill: e.colour }),
        ]),
      ],
    }));

    // Right panel with formulas and numeric readouts.
    const disc = tr * tr - 4 * det;
    const status = real
      ? `<span style="color:${palette.primary}">two real eigenvalues</span>`
      : `<span style="color:${palette.danger}">complex pair (no real eigenvectors)</span>`;
    host.querySelector('[data-display="readouts"]').innerHTML = `
      <div class="widget-matrix-title">readouts</div>
      <div class="widget-matrix-display">
        <div>tr(A) = ${fmt(tr)}</div>
        <div>det(A) = ${fmt(det)}</div>
        <div>discriminant = tr² − 4·det = ${fmt(disc)}</div>
        <div>p(λ) = λ² − ${fmt(tr)}·λ + ${fmt(det)}</div>
        <div>${status}</div>
        ${real ? eigs.map((e, i) => `<div>λ${i + 1} = ${fmt(e.lambda)}</div>`).join("") : ""}
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

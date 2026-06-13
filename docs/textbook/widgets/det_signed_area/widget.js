// Widget 1.J — Determinant as signed area (Chapter 1, §1.3).
//
// The columns of a 2×2 matrix A = [c1 c2] map the standard unit square
// to a parallelogram with sides c1 and c2. The *signed* area of that
// parallelogram is det A. This widget makes that connection direct: the
// reader edits the four cells of A and watches the parallelogram morph
// while the live det A readout updates.
//
// Pedagogical points:
//   - det > 0: orientation preserved (c1 → c2 is still counter-clockwise).
//   - det < 0: orientation flipped (the parallelogram is "mirrored").
//   - det = 0: parallelogram collapses to a line — A is singular.
//   - The numeric value of |det| equals the unsigned area.
//
// Mount: `<div id="ch1-det-area-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor } from "../shared/matrix_editor.js";
import { det2x2, matvec2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-det-area-widget";

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

  const A = [[1.5, 0.5], [0.3, 1.2]];

  function render() {
    const det = det2x2(A);
    // Image of the unit square: (0,0), Ae1, Ae1+Ae2, Ae2, back to (0,0).
    const [Ae1x, Ae1y] = matvec2x2(A, [1, 0]);
    const [Ae2x, Ae2y] = matvec2x2(A, [0, 1]);
    const parallelogram = [
      { x: 0, y: 0 },
      { x: Ae1x, y: Ae1y },
      { x: Ae1x + Ae2x, y: Ae1y + Ae2y },
      { x: Ae2x, y: Ae2y },
      { x: 0, y: 0 },
    ];
    const unitSquare = [
      { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: 0, y: 0 },
    ];

    // Fit the plot to the parallelogram with some margin.
    const xs = parallelogram.flatMap((p) => [p.x, 0]).concat([1, -0.2]);
    const ys = parallelogram.flatMap((p) => [p.y, 0]).concat([1, -0.2]);
    const R = Math.max(2, Math.max(...xs.map(Math.abs)), Math.max(...ys.map(Math.abs))) * 1.15;

    // Fill colour encodes orientation.
    const fill = det >= 0 ? palette.primary : palette.danger;

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 360, width: 420,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [-R, R], label: "x", grid: true },
      y: { domain: [-R, R], label: "y", grid: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Faint unit square for reference.
        Plot.line(unitSquare, {
          x: "x", y: "y",
          stroke: palette.muted, strokeOpacity: 0.6, ...dashed,
        }),
        // Output parallelogram outline (orientation-coloured).
        Plot.line(parallelogram, {
          x: "x", y: "y",
          stroke: fill, strokeWidth: 2.5,
          fill, fillOpacity: 0.15,
        }),
        // The two column-vectors as arrows from origin.
        Plot.arrow([{ x1: 0, y1: 0, x2: Ae1x, y2: Ae1y }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.warning, strokeWidth: 2, headLength: 10,
        }),
        Plot.arrow([{ x1: 0, y1: 0, x2: Ae2x, y2: Ae2y }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.secondary, strokeWidth: 2, headLength: 10,
        }),
        Plot.text([{ x: Ae1x, y: Ae1y, label: `Ae₁ = (${fmt(Ae1x)}, ${fmt(Ae1y)})` }], {
          x: "x", y: "y", text: "label",
          dx: 6, textAnchor: "start", fill: palette.warning, ...annotation,
        }),
        Plot.text([{ x: Ae2x, y: Ae2y, label: `Ae₂ = (${fmt(Ae2x)}, ${fmt(Ae2y)})` }], {
          x: "x", y: "y", text: "label",
          dx: 6, textAnchor: "start", fill: palette.secondary, ...annotation,
        }),
      ],
    }));

    const sign = det > 1e-9 ? "+ (orientation preserved)"
              : det < -1e-9 ? "− (orientation flipped)"
              : "0 (singular — collapses to a line)";
    host.querySelector('[data-display="readouts"]').innerHTML = `
      <div class="widget-matrix-title">readouts</div>
      <div class="widget-matrix-display">
        <div>Ae₁ = (${fmt(Ae1x)}, ${fmt(Ae1y)})</div>
        <div>Ae₂ = (${fmt(Ae2x)}, ${fmt(Ae2y)})</div>
        <div><b>det A = ${fmt(det)}</b></div>
        <div style="color:${fill}">signed area: ${sign}</div>
      </div>
    `;
  }

  buildMatrixEditor(host, '[data-matrix="A"]', A, render, {
    step: 0.1,
    colHeaders: ["c₁", "c₂"],
  });
  render();
}

if (document.readyState !== "loading") mount();
else document.addEventListener("DOMContentLoaded", mount);

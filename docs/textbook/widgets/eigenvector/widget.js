// Widget 1.B — Eigenvector finder (Chapter 1, §1.2).
//
// Visualises the eigenvector definition Av = λv geometrically. The
// student edits a 2×2 matrix A directly as a grid of number inputs.
// The widget draws a ring of input unit vectors v (blue) and the
// corresponding output vectors Av (orange). Vectors whose direction is
// unchanged by A — i.e. the eigenvectors — are highlighted in green,
// together with their eigenvalue (= signed length of Av along v).
//
// Eigenvalues are computed in closed form from the characteristic
// polynomial of a 2×2 matrix:
//
//     trace = a + d,  det = a·d − b·c
//     λ = (trace ± √(trace² − 4·det)) / 2
//
// Complex eigenvalues (discriminant < 0) mean A has *no* real
// eigenvectors — the widget reports this and just shows the input/output
// ring (every vector is rotated).
//
// Layout: the abstract 2-D plot on the left; an editable 2×2 matrix +
// computed eigenvalues / eigenvectors panel on the right. Hovering a
// row in the eigenvector panel highlights the corresponding arrow in
// red so readers can see "this row of numbers IS that arrow."
//
// Custom mount (not defineWidget) so we can compose a non-flat layout
// of <input> cells, computed panels, and a Plot canvas.
//
// Mount: `<div id="ch1-eigenvector-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/eigenvector/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import {
  buildMatrixEditor,
  buildMatrixDisplay,
  buildVectorPanel,
} from "../shared/matrix_editor.js";
import { eig2x2, trace2x2, det2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-eigenvector-widget";

const SCAFFOLD_HTML = `
  <div class="widget-matrix-layout">
    <div data-plot="main"></div>
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (edit cells)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="eigvals"></div>
      <div data-display="eigvecs"></div>
    </div>
  </div>
  <div data-readout></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  // Persistent state across re-renders.
  const A = [[2, 1], [0, 3]];
  // Which eigenvector index is hovered (highlights that arrow). -1 = none.
  let hoverIdx = -1;

  function render() {
    const [[a, b], [c, d]] = A;

    // Ring of input vectors and their images.
    const N = 36;
    const ring = d3.range(N).map((i) => {
      const theta = (2 * Math.PI * i) / N;
      const vx = Math.cos(theta), vy = Math.sin(theta);
      return { vx, vy, ax: a * vx + b * vy, ay: c * vx + d * vy };
    });

    const M = [[a, b], [c, d]];
    const { real, eigs } = eig2x2(M);
    const tr = trace2x2(M);
    const det = det2x2(M);

    // Auto-fit plot box to ring + arrows.
    const allX = ring.flatMap((p) => [p.vx, p.ax]);
    const allY = ring.flatMap((p) => [p.vy, p.ay]);
    const R = Math.max(1.2, Math.max(...allX.map(Math.abs), ...allY.map(Math.abs)) * 1.15);

    const inputArrows = ring.map((p) => ({ x1: 0, y1: 0, x2: p.vx, y2: p.vy }));
    const outputArrows = ring.map((p) => ({ x1: 0, y1: 0, x2: p.ax, y2: p.ay }));

    const marks = [
      Plot.ruleX([0], { stroke: palette.muted, ...dashed }),
      Plot.ruleY([0], { stroke: palette.muted, ...dashed }),
      Plot.line(ring.concat([ring[0]]), {
        x: "vx", y: "vy", stroke: palette.secondary, strokeOpacity: 0.5,
      }),
      Plot.line(ring.concat([ring[0]]), {
        x: "ax", y: "ay", stroke: palette.warning, strokeOpacity: 0.5,
      }),
      Plot.arrow(inputArrows.filter((_, i) => i % 6 === 0), {
        x1: "x1", y1: "y1", x2: "x2", y2: "y2",
        stroke: palette.secondary, strokeWidth: 1, strokeOpacity: 0.7, headLength: 5,
      }),
      Plot.arrow(outputArrows.filter((_, i) => i % 6 === 0), {
        x1: "x1", y1: "y1", x2: "x2", y2: "y2",
        stroke: palette.warning, strokeWidth: 1, strokeOpacity: 0.7, headLength: 5,
      }),
    ];

    // Highlight eigen-directions. The hovered one is red; others green.
    eigs.forEach(({ lambda, v }, idx) => {
      const colour = idx === hoverIdx ? palette.danger : palette.primary;
      const tip = [lambda * v[0], lambda * v[1]];
      marks.push(
        Plot.line(
          [{ x: -R * v[0], y: -R * v[1] }, { x: R * v[0], y: R * v[1] }],
          { x: "x", y: "y", stroke: colour, strokeWidth: 1.5, strokeOpacity: 0.4 },
        ),
        Plot.arrow([{ x1: 0, y1: 0, x2: v[0], y2: v[1] }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: colour, strokeWidth: 2.5, headLength: 8,
        }),
        Plot.arrow([{ x1: 0, y1: 0, x2: tip[0], y2: tip[1] }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: colour, strokeWidth: 2.5, strokeDasharray: "4 2", headLength: 8,
        }),
        Plot.text([{ x: tip[0], y: tip[1], label: `λ=${fmt(lambda)}` }], {
          x: "x", y: "y", text: "label", textAnchor: "start", dx: 6, dy: -4,
          fill: colour, ...annotation,
        }),
      );
    });

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380,
      width: 420,
      x: { domain: [-R, R], label: "x", grid: true },
      y: { domain: [-R, R], label: "y", grid: true },
      aspectRatio: 1,
      marks,
    }));

    // Eigenvalue display (could be 0, 1, or 2 real eigs; or complex).
    if (real) {
      const eigvals = eigs.length
        ? [eigs.map((e) => e.lambda)]
        : [[NaN]];
      buildMatrixDisplay(host, '[data-display="eigvals"]', eigvals, {
        title: "eigenvalues λ",
        colHeaders: eigs.map((_, i) => `λ${i + 1}`),
      });
      // Eigenvectors panel (hover highlights arrow on plot).
      buildVectorPanel(
        host,
        '[data-display="eigvecs"]',
        eigs.map((e, i) => ({
          label: `v${i + 1}`,
          values: e.v,
          onEnter: () => { hoverIdx = i; render(); },
          onLeave: () => { hoverIdx = -1; render(); },
        })),
        { title: "eigenvectors v (hover to highlight)" },
      );
    } else {
      const reLam = tr / 2;
      const imLam = Math.sqrt(Math.max(0, -(tr * tr - 4 * det))) / 2;
      buildMatrixDisplay(host, '[data-display="eigvals"]', [[reLam, imLam]], {
        title: "eigenvalues (complex)",
        colHeaders: ["Re λ", "Im λ"],
      });
      const panel = host.querySelector('[data-display="eigvecs"]');
      panel.innerHTML = `
        <div class="widget-matrix-title">eigenvectors</div>
        <div style="font-size:0.85em;opacity:0.8;">No real eigenvectors —
          A rotates every direction.</div>`;
    }

    let msg;
    if (!real) {
      const reLam = tr / 2;
      const imLam = Math.sqrt(Math.max(0, -(tr * tr - 4 * det))) / 2;
      msg = `<strong>No real eigenvectors.</strong> ` +
        `λ = ${fmt(reLam)} ± ${fmt(imLam)}i — A rotates every direction. ` +
        `trace = ${fmt(tr)}, det = ${fmt(det)}.`;
    } else {
      const list = eigs.map((e) =>
        `λ=${fmt(e.lambda)} along (${fmt(e.v[0])}, ${fmt(e.v[1])})`,
      ).join("; ");
      msg = `<strong>Eigenvectors</strong> (green): ${list || "none (defective)"}. ` +
        `trace = ${fmt(tr)}, det = ${fmt(det)}.`;
    }
    host.querySelector("[data-readout]").innerHTML =
      `<small>Blue ring = input unit vectors. Orange = Av. ` +
      `Green = directions A only stretches; red = currently-hovered eigenvector.</small><br>${msg}`;
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

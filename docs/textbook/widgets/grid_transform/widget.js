// Widget 1.G — Matrix-as-grid-transformation (Chapter 1, §1.6).
//
// Side-by-side plots: the input is a square 11×11 unit grid (with the
// unit square highlighted), and the right panel shows where every grid
// line lands after a user-editable 2×2 matrix A applies. The reader
// edits A's four entries and watches:
//
//   - the grid stretch, rotate, shear, or collapse;
//   - the unit square become a parallelogram of area det(A);
//   - the eigendirections (drawn as orange arrows when real) be the
//     directions the transformed grid lines DON'T rotate off of;
//   - the eigendirections collide and disappear (complex eigenvalues)
//     when A is close to a rotation.
//
// The complementary widget at `ch1-power-iteration-widget` shows what
// happens when A is applied repeatedly to a single vector — convergence
// to the dominant eigenvector. This widget is about the geometry of one
// step; that one is about the dynamics of many steps.
//
// Mount: `<div id="ch1-grid-transform-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/grid_transform/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor } from "../shared/matrix_editor.js";
import { eig2x2, trace2x2, det2x2, matvec2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-grid-transform-widget";
const GRID_HALF = 3; // grid runs from -3 to +3 on each axis
const GRID_LINES = 13; // number of lines per axis
const N_PTS = 41; // samples per line — high enough to look smooth after A

const SCAFFOLD_HTML = `
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (edit cells)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="info"></div>
    </div>
    <div data-plot="grid"></div>
  </div>
  <div data-readout></div>
`;

// Build the input grid: GRID_LINES vertical lines and GRID_LINES horizontal
// lines, each sampled N_PTS times. Returns array of { lineId, x, y } so
// Plot.line can stroke each line as a separate series via z: "lineId".
function buildInputGrid() {
  const lines = [];
  let id = 0;
  // Vertical lines (constant x).
  for (let i = 0; i < GRID_LINES; i++) {
    const x = -GRID_HALF + (i * 2 * GRID_HALF) / (GRID_LINES - 1);
    const isAxis = Math.abs(x) < 1e-9;
    const isUnit = Math.abs(x - 1) < 1e-9 || Math.abs(x) < 1e-9; // x=0 or x=1 for unit-square edges
    for (let j = 0; j < N_PTS; j++) {
      const y = -GRID_HALF + (j * 2 * GRID_HALF) / (N_PTS - 1);
      lines.push({ lineId: `v${id}`, x, y, isAxis, kind: "vert" });
    }
    id += 1;
  }
  // Horizontal lines (constant y).
  for (let i = 0; i < GRID_LINES; i++) {
    const y = -GRID_HALF + (i * 2 * GRID_HALF) / (GRID_LINES - 1);
    const isAxis = Math.abs(y) < 1e-9;
    for (let j = 0; j < N_PTS; j++) {
      const x = -GRID_HALF + (j * 2 * GRID_HALF) / (N_PTS - 1);
      lines.push({ lineId: `h${id}`, x, y, isAxis, kind: "horiz" });
    }
    id += 1;
  }
  return lines;
}

// Apply A to a (x, y) point.
const applyMatrix = (A, x, y) => matvec2x2(A, [x, y]);

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  const A = [[1.5, 0.5], [0.3, 1.2]];
  const inputGrid = buildInputGrid();
  // Pre-compute the unit-square corners in input coords.
  const unitSquareIn = [
    { lineId: "us", x: 0, y: 0 },
    { lineId: "us", x: 1, y: 0 },
    { lineId: "us", x: 1, y: 1 },
    { lineId: "us", x: 0, y: 1 },
    { lineId: "us", x: 0, y: 0 },
  ];

  function render() {
    const transformed = inputGrid.map((p) => {
      const [tx, ty] = applyMatrix(A, p.x, p.y);
      return { ...p, x: tx, y: ty };
    });
    const unitSquareOut = unitSquareIn.map((p) => {
      const [tx, ty] = applyMatrix(A, p.x, p.y);
      return { ...p, x: tx, y: ty };
    });

    const { real, eigs } = eig2x2(A);
    const tr = trace2x2(A);
    const det = det2x2(A);

    // Plot domain: enough to fit transformed grid.
    const maxAbs = Math.max(
      GRID_HALF + 0.5,
      ...transformed.map((p) => Math.abs(p.x)),
      ...transformed.map((p) => Math.abs(p.y)),
    );
    const R = Math.min(maxAbs, 10);

    // Eigenvector arrows scaled to be visible in the chart.
    const eigArrows = real
      ? eigs.map(({ lambda, v }) => ({
          x1: 0, y1: 0,
          x2: v[0] * 1.6, y2: v[1] * 1.6,
          // Also draw the IMAGE of the eigenvector under A (which is λ * v).
          ix: v[0] * lambda, iy: v[1] * lambda,
          lambda,
          label: `λ=${fmt(lambda)}`,
        }))
      : [];

    host.querySelector('[data-plot="grid"]').replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 420,
        width: 460,
        marginLeft: 36,
        marginBottom: 32,
        x: { domain: [-R, R], label: "x", grid: false },
        y: { domain: [-R, R], label: "y", grid: false },
        aspectRatio: 1,
        marks: [
          // Faint input grid (background reference).
          Plot.line(inputGrid, {
            x: "x", y: "y", z: "lineId",
            stroke: palette.muted, strokeOpacity: 0.15, strokeWidth: 1,
          }),
          // Transformed grid (foreground).
          Plot.line(transformed, {
            x: "x", y: "y", z: "lineId",
            stroke: (d) => d.isAxis ? palette.primary : palette.secondary,
            strokeOpacity: (d) => d.isAxis ? 0.95 : 0.45,
            strokeWidth: (d) => d.isAxis ? 2 : 1,
          }),
          // Transformed unit square (filled).
          Plot.line(unitSquareOut, {
            x: "x", y: "y",
            stroke: palette.warning, strokeWidth: 2.5,
          }),
          // x=0 and y=0 reference lines (faint).
          Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.5, ...dashed }),
          Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5, ...dashed }),
          // Eigenvector arrows (real case).
          ...(real ? [
            Plot.arrow(eigArrows, {
              x1: "x1", y1: "y1", x2: "ix", y2: "iy",
              stroke: palette.warning, strokeWidth: 2.5, headLength: 10,
            }),
            Plot.text(eigArrows, {
              x: "ix", y: "iy", text: "label",
              dx: 6, textAnchor: "start", fill: palette.warning,
              ...annotation,
            }),
          ] : []),
        ],
      })
    );

    // Info panel.
    const info = host.querySelector('[data-display="info"]');
    info.innerHTML = `
      <div class="widget-matrix-title">readouts</div>
      <div class="widget-matrix-display">
        <div>tr(A) = ${fmt(tr)}</div>
        <div>det(A) = ${fmt(det)}</div>
        ${real
          ? eigs.map((e, i) => `<div>λ${i + 1} = ${fmt(e.lambda)} · v${i + 1} ≈ (${fmt(e.v[0])}, ${fmt(e.v[1])})</div>`).join("")
          : `<div style="color:${palette.danger}">complex eigenvalues — no real eigenvectors</div>`
        }
      </div>
    `;
  }

  buildMatrixEditor(host, '[data-matrix="A"]', A, render, {
    step: 0.1,
    colHeaders: ["col 1", "col 2"],
  });
  render();
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

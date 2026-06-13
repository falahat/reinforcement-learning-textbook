// Widget 1.T — Transformation zoo (Chapter 1, §1.3).
//
// The reader picks a preset transformation from a dropdown — identity,
// scale, rotation, shear-x, shear-y, reflect-x, reflect-y, reflect-y=x,
// project-x, project-y — and gets:
//
//   - the 2×2 matrix that implements it (displayed as a real matrix,
//     with the active parameter substituted in);
//   - the unit grid pushed through the transformation (faint input
//     grid overlaid with the transformed grid + unit square);
//   - three editable example vectors with their images shown as
//     arrows (input arrows faint, output arrows bold);
//   - a readout: name, det A, the canonical formula, and any
//     parameter-specific notes (e.g. "rotation preserves length").
//
// The companion `grid_transform` widget lets the reader build an
// arbitrary 2×2 matrix entry-by-entry. This widget is the
// pedagogical counterpart: the reader picks a *kind* of transformation
// and reads its matrix, rather than reading a matrix and guessing the
// kind. Both views are needed — one trains "matrix → picture", the
// other trains "picture → matrix".
//
// Mount:
//   <div id="ch1-transformation-zoo-widget" class="textbook-widget"></div>
//   <script type="module" src="./widgets/transformation_zoo/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixDisplay } from "../shared/matrix_editor.js";
import { matvec2x2, det2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-transformation-zoo-widget";
const GRID_HALF = 3;
const GRID_LINES = 13;
const N_PTS = 41;

// ─── Preset transformations ───────────────────────────────────────
//
// Each preset returns a 2×2 matrix given the slider parameter `p`
// (interpretation differs per preset — angle in degrees for rotate,
// signed scale factor for scale/shear, …).
//
// `paramLabel` / `paramRange` configure the slider; `formula` is the
// LaTeX-ish display showing the matrix's symbolic form so the reader
// can connect the picture to the algebra; `notes` is one line of
// geometric commentary that surfaces in the readout.

const PRESETS = {
  identity: {
    label: "Identity",
    paramVisible: false,
    matrix: () => [[1, 0], [0, 1]],
    formula: "I = [[1, 0], [0, 1]]",
    notes: "Every vector maps to itself. det = 1.",
  },
  scale_uniform: {
    label: "Uniform scale (×s)",
    paramVisible: true,
    paramLabel: "s",
    paramRange: { min: -2, max: 3, step: 0.1, default: 1.5 },
    matrix: (s) => [[s, 0], [0, s]],
    formula: (s) => `sI = [[${fmt(s)}, 0], [0, ${fmt(s)}]]`,
    notes: (s) =>
      `Uniform scale by ${fmt(s)}. det = ${fmt(s * s)}. ` +
      (s < 0 ? "Negative s reflects through the origin." : "Shape preserved, size changes."),
  },
  scale_x: {
    label: "Scale x-axis only (a, 1)",
    paramVisible: true,
    paramLabel: "a",
    paramRange: { min: -2, max: 3, step: 0.1, default: 2 },
    matrix: (a) => [[a, 0], [0, 1]],
    formula: (a) => `[[${fmt(a)}, 0], [0, 1]]`,
    notes: (a) =>
      `Stretch x by ${fmt(a)}, leave y alone. det = ${fmt(a)}. ` +
      "Circles become ellipses with the long axis along x.",
  },
  scale_y: {
    label: "Scale y-axis only (1, d)",
    paramVisible: true,
    paramLabel: "d",
    paramRange: { min: -2, max: 3, step: 0.1, default: 2 },
    matrix: (d) => [[1, 0], [0, d]],
    formula: (d) => `[[1, 0], [0, ${fmt(d)}]]`,
    notes: (d) =>
      `Stretch y by ${fmt(d)}, leave x alone. det = ${fmt(d)}. ` +
      "Circles become ellipses with the long axis along y.",
  },
  rotate: {
    label: "Rotation by θ°",
    paramVisible: true,
    paramLabel: "θ (theta, degrees)",
    paramRange: { min: -180, max: 180, step: 1, default: 45 },
    matrix: (deg) => {
      const r = (deg * Math.PI) / 180;
      const c = Math.cos(r), s = Math.sin(r);
      return [[c, -s], [s, c]];
    },
    formula: (deg) => {
      const r = (deg * Math.PI) / 180;
      return `R_θ = [[cos θ, -sin θ], [sin θ, cos θ]] = [[${fmt(Math.cos(r))}, ${fmt(-Math.sin(r))}], [${fmt(Math.sin(r))}, ${fmt(Math.cos(r))}]]`;
    },
    notes: (deg) =>
      `Rotate by ${fmt(deg)}°. det = 1 (lengths, angles, areas all preserved). ` +
      (Math.abs(deg) > 0.01 && Math.abs(Math.abs(deg) - 180) > 0.01
        ? "No real eigenvectors — every direction rotates."
        : "Real eigenvalues at θ = 0 (identity) or θ = ±180 (point reflection)."),
  },
  shear_x: {
    label: "Horizontal shear (k along x)",
    paramVisible: true,
    paramLabel: "k",
    paramRange: { min: -2, max: 2, step: 0.05, default: 1 },
    matrix: (k) => [[1, k], [0, 1]],
    formula: (k) => `H_x(k) = [[1, ${fmt(k)}], [0, 1]]`,
    notes: (k) =>
      `Horizontal shear by ${fmt(k)}. det = 1 (area preserved). ` +
      "x-axis fixed (λ=1); only one eigenvector direction — this is a *defective* matrix.",
  },
  shear_y: {
    label: "Vertical shear (k along y)",
    paramVisible: true,
    paramLabel: "k",
    paramRange: { min: -2, max: 2, step: 0.05, default: 1 },
    matrix: (k) => [[1, 0], [k, 1]],
    formula: (k) => `H_y(k) = [[1, 0], [${fmt(k)}, 1]]`,
    notes: (k) =>
      `Vertical shear by ${fmt(k)}. det = 1 (area preserved). ` +
      "y-axis fixed (λ=1); defective, like the horizontal shear.",
  },
  reflect_x: {
    label: "Reflect across x-axis",
    paramVisible: false,
    matrix: () => [[1, 0], [0, -1]],
    formula: "M_x = [[1, 0], [0, -1]]",
    notes:
      "Reflect across x-axis. det = -1 (orientation flipped). " +
      "Eigenvectors: e₁ at λ=+1, e₂ at λ=-1.",
  },
  reflect_y: {
    label: "Reflect across y-axis",
    paramVisible: false,
    matrix: () => [[-1, 0], [0, 1]],
    formula: "M_y = [[-1, 0], [0, 1]]",
    notes:
      "Reflect across y-axis. det = -1 (orientation flipped). " +
      "Eigenvectors: e₂ at λ=+1, e₁ at λ=-1.",
  },
  reflect_diag: {
    label: "Reflect across y = x",
    paramVisible: false,
    matrix: () => [[0, 1], [1, 0]],
    formula: "M_d = [[0, 1], [1, 0]]",
    notes:
      "Swap x and y; reflect across the y = x diagonal. det = -1. " +
      "Eigenvectors: (1,1)/√2 at λ=+1, (1,-1)/√2 at λ=-1.",
  },
  project_x: {
    label: "Project onto x-axis",
    paramVisible: false,
    matrix: () => [[1, 0], [0, 0]],
    formula: "P_x = [[1, 0], [0, 0]]",
    notes:
      "Squash everything onto the x-axis. det = 0 (singular — collapses to a line). " +
      "P² = P (idempotent). Eigenvalues 1 (image) and 0 (kernel).",
  },
  project_y: {
    label: "Project onto y-axis",
    paramVisible: false,
    matrix: () => [[0, 0], [0, 1]],
    formula: "P_y = [[0, 0], [0, 1]]",
    notes:
      "Squash everything onto the y-axis. det = 0 (singular). " +
      "P² = P (idempotent). Eigenvalues 1 (image) and 0 (kernel).",
  },
  rotate_and_scale: {
    label: "Rotate 30° + scale by 1.5",
    paramVisible: false,
    matrix: () => {
      const r = (30 * Math.PI) / 180;
      const c = Math.cos(r), s = Math.sin(r), k = 1.5;
      return [[k * c, -k * s], [k * s, k * c]];
    },
    formula: "1.5·R_30° (a spiral)",
    notes:
      "Composition: scale by 1.5 *and* rotate by 30°. det = 1.5² = 2.25. " +
      "Combining rotations with scaling is what produces spiral dynamics " +
      "in repeated application (see the power-iteration widget).",
  },
};

// Default example vectors to push through A. The reader can edit each
// one. Picked to land in three different parts of the plane so the
// transformation's effect on each is visually distinct.
const DEFAULT_VECTORS = [
  { name: "u", x: 1, y: 0, color: palette.warning },
  { name: "v", x: 0, y: 1, color: palette.accent },
  { name: "w", x: 1, y: 1, color: palette.primary },
];

const SCAFFOLD_HTML = `
  <div class="widget-controls" style="flex-wrap: wrap; align-items: center;">
    <label>Transformation
      <select data-input="preset"></select>
    </label>
    <label data-param-wrap>
      <span data-param-label></span>
      <input type="range" data-input="param">
      <input type="number" step="0.01" data-input="param-num"
             style="width: 5em; margin-left: 0.4em;">
    </label>
    <button type="button" data-action="reset-vectors"
            title="Restore the default example vectors">
      reset vectors
    </button>
  </div>
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div data-display="matrix"></div>
      <div data-display="formula" style="font-family: monospace; font-size: 0.85em; opacity: 0.85;"></div>
      <div data-display="vectors" class="widget-matrix-panel" style="gap: 0.3em;">
        <div class="widget-matrix-title">example vectors</div>
        <div data-vectors-edit></div>
      </div>
    </div>
    <div data-plot="grid"></div>
  </div>
  <div data-readout></div>
`;

// Build the input grid (vertical + horizontal lines, sampled densely
// so the curves look smooth after a strong transformation). Identical
// to the companion `grid_transform` widget; kept inline so this widget
// doesn't depend on it.
function buildInputGrid() {
  const lines = [];
  let id = 0;
  for (let i = 0; i < GRID_LINES; i++) {
    const x = -GRID_HALF + (i * 2 * GRID_HALF) / (GRID_LINES - 1);
    const isAxis = Math.abs(x) < 1e-9;
    for (let j = 0; j < N_PTS; j++) {
      const y = -GRID_HALF + (j * 2 * GRID_HALF) / (N_PTS - 1);
      lines.push({ lineId: `v${id}`, x, y, isAxis, kind: "vert" });
    }
    id += 1;
  }
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

// Render the vector-editor panel: one row per vector with x/y inputs
// and a colour swatch. Mutates `vectors` in place and re-calls
// `onChange` after every input event.
function buildVectorEditor(host, vectors, onChange) {
  const root = host.querySelector("[data-vectors-edit]");
  root.innerHTML = "";
  for (let i = 0; i < vectors.length; i++) {
    const v = vectors[i];
    const row = document.createElement("div");
    row.className = "widget-vector-row";
    row.style.display = "grid";
    row.style.gridTemplateColumns = "auto 1fr auto auto";
    row.style.alignItems = "center";
    row.style.gap = "0.35em";

    const swatch = document.createElement("span");
    swatch.style.cssText =
      `display: inline-block; width: 0.9em; height: 0.9em; background: ${v.color}; ` +
      "border-radius: 2px;";
    row.appendChild(swatch);

    const label = document.createElement("span");
    label.textContent = v.name + " =";
    label.className = "widget-vector-label";
    row.appendChild(label);

    const xIn = document.createElement("input");
    xIn.type = "number";
    xIn.step = "0.1";
    xIn.value = v.x.toFixed(2);
    xIn.style.width = "4.5em";
    xIn.className = "widget-matrix-cell";
    xIn.addEventListener("input", () => {
      const x = parseFloat(xIn.value);
      if (Number.isFinite(x)) {
        vectors[i].x = x;
        onChange();
      }
    });
    row.appendChild(xIn);

    const yIn = document.createElement("input");
    yIn.type = "number";
    yIn.step = "0.1";
    yIn.value = v.y.toFixed(2);
    yIn.style.width = "4.5em";
    yIn.className = "widget-matrix-cell";
    yIn.addEventListener("input", () => {
      const y = parseFloat(yIn.value);
      if (Number.isFinite(y)) {
        vectors[i].y = y;
        onChange();
      }
    });
    row.appendChild(yIn);

    root.appendChild(row);
  }
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  // Populate the preset selector.
  const sel = host.querySelector('[data-input="preset"]');
  for (const [key, p] of Object.entries(PRESETS)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = p.label;
    sel.appendChild(opt);
  }
  sel.value = "rotate";

  // Keep one mutable copy of the example vectors so the matrix-editor
  // events can mutate it in place.
  const vectors = DEFAULT_VECTORS.map((v) => ({ ...v }));

  const inputGrid = buildInputGrid();
  const unitSquareIn = [
    { lineId: "us", x: 0, y: 0 },
    { lineId: "us", x: 1, y: 0 },
    { lineId: "us", x: 1, y: 1 },
    { lineId: "us", x: 0, y: 1 },
    { lineId: "us", x: 0, y: 0 },
  ];

  // The param slider + number share a value — keep them in sync.
  const paramRange = host.querySelector('[data-input="param"]');
  const paramNum = host.querySelector('[data-input="param-num"]');
  const paramWrap = host.querySelector("[data-param-wrap]");
  const paramLabel = host.querySelector("[data-param-label]");

  function setParamConfig(preset) {
    if (!preset.paramVisible) {
      paramWrap.style.display = "none";
      return;
    }
    paramWrap.style.display = "";
    paramLabel.textContent = preset.paramLabel + ": ";
    const r = preset.paramRange;
    paramRange.min = r.min;
    paramRange.max = r.max;
    paramRange.step = r.step;
    paramRange.value = r.default;
    paramNum.min = r.min;
    paramNum.max = r.max;
    paramNum.step = r.step;
    paramNum.value = r.default;
  }

  function currentParam() {
    return parseFloat(paramRange.value);
  }

  function render() {
    const presetKey = sel.value;
    const preset = PRESETS[presetKey] || PRESETS.identity;
    const p = preset.paramVisible ? currentParam() : null;
    const A = preset.paramVisible ? preset.matrix(p) : preset.matrix();
    const det = det2x2(A);

    // Matrix display panel.
    buildMatrixDisplay(host, '[data-display="matrix"]', A, {
      title: "A",
      colHeaders: ["col 1", "col 2"],
    });
    host.querySelector('[data-display="formula"]').textContent =
      typeof preset.formula === "function" ? preset.formula(p) : preset.formula;

    // Push the input grid + unit square through A.
    const transformed = inputGrid.map((q) => {
      const [tx, ty] = matvec2x2(A, [q.x, q.y]);
      return { ...q, x: tx, y: ty };
    });
    const unitSquareOut = unitSquareIn.map((q) => {
      const [tx, ty] = matvec2x2(A, [q.x, q.y]);
      return { ...q, x: tx, y: ty };
    });

    // Example vectors — pre- and post-image arrows.
    const vectorMarks = [];
    for (const v of vectors) {
      const [tx, ty] = matvec2x2(A, [v.x, v.y]);
      // Pre-image: faint arrow.
      vectorMarks.push({
        kind: "in",
        x1: 0, y1: 0, x2: v.x, y2: v.y,
        label: `${v.name}`,
        color: v.color,
      });
      // Post-image: bold arrow + label "Av".
      vectorMarks.push({
        kind: "out",
        x1: 0, y1: 0, x2: tx, y2: ty,
        label: `A${v.name}`,
        color: v.color,
      });
    }
    const arrowsIn = vectorMarks.filter((m) => m.kind === "in");
    const arrowsOut = vectorMarks.filter((m) => m.kind === "out");

    // Plot domain: large enough to fit the most-extreme transformed
    // point or vector.
    const maxAbs = Math.max(
      GRID_HALF + 0.5,
      ...transformed.map((p) => Math.abs(p.x)),
      ...transformed.map((p) => Math.abs(p.y)),
      ...arrowsOut.map((m) => Math.max(Math.abs(m.x2), Math.abs(m.y2))),
    );
    const R = Math.min(Math.max(maxAbs * 1.05, 3), 10);

    host.querySelector('[data-plot="grid"]').replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 440,
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
            stroke: palette.muted, strokeOpacity: 0.18, strokeWidth: 1,
          }),
          // Transformed grid (foreground).
          Plot.line(transformed, {
            x: "x", y: "y", z: "lineId",
            stroke: (d) => (d.isAxis ? palette.primary : palette.secondary),
            strokeOpacity: (d) => (d.isAxis ? 0.95 : 0.45),
            strokeWidth: (d) => (d.isAxis ? 2 : 1),
          }),
          // Transformed unit square.
          Plot.line(unitSquareOut, {
            x: "x", y: "y",
            stroke: palette.warning, strokeWidth: 2.5,
          }),
          // x=0 and y=0 reference rules.
          Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.5, ...dashed }),
          Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5, ...dashed }),
          // Pre-image vectors (faint).
          Plot.arrow(arrowsIn, {
            x1: "x1", y1: "y1", x2: "x2", y2: "y2",
            stroke: (d) => d.color, strokeOpacity: 0.45, strokeWidth: 1.5,
            headLength: 7,
          }),
          // Post-image vectors (bold).
          Plot.arrow(arrowsOut, {
            x1: "x1", y1: "y1", x2: "x2", y2: "y2",
            stroke: (d) => d.color, strokeWidth: 2.6,
            headLength: 11,
          }),
          // Labels at the post-image arrow tips.
          Plot.text(arrowsOut, {
            x: "x2", y: "y2", text: "label",
            dx: 6, textAnchor: "start", fill: (d) => d.color,
            fontSize: 11, fontWeight: 600,
          }),
        ],
      })
    );

    // Readout: name, det, notes.
    const notes =
      typeof preset.notes === "function" ? preset.notes(p) : preset.notes;
    const detStr = Math.abs(det) < 1e-9 ? "0 (singular)" : fmt(det);
    host.querySelector("[data-readout]").innerHTML =
      `<strong>${preset.label}</strong> &mdash; det A = ${detStr}. ${notes}`;
  }

  // Wire the preset select to (re-)configure the parameter slider then
  // render.
  sel.addEventListener("change", () => {
    setParamConfig(PRESETS[sel.value]);
    render();
  });

  // Slider ↔ number-input two-way sync.
  paramRange.addEventListener("input", () => {
    paramNum.value = paramRange.value;
    render();
  });
  paramNum.addEventListener("input", () => {
    const v = parseFloat(paramNum.value);
    if (Number.isFinite(v)) {
      paramRange.value = v;
      render();
    }
  });

  // Reset-vectors button.
  host.querySelector('[data-action="reset-vectors"]').addEventListener(
    "click",
    () => {
      for (let i = 0; i < DEFAULT_VECTORS.length; i++) {
        vectors[i].x = DEFAULT_VECTORS[i].x;
        vectors[i].y = DEFAULT_VECTORS[i].y;
      }
      buildVectorEditor(host, vectors, render);
      render();
    }
  );

  setParamConfig(PRESETS[sel.value]);
  buildVectorEditor(host, vectors, render);
  render();
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

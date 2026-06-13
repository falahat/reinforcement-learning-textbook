// Shared matrix-editor helper for textbook widgets that need to expose
// a matrix or vector as an editable grid of number inputs.
//
// The textbook's `defineWidget` scaffold produces a flat row of <input>
// controls — fine for sliders, but a 2×2 matrix wants a 2-D layout with
// monospace digits and clear cell borders so readers see "this IS the
// matrix being manipulated." This helper hand-rolls that grid.
//
// Three exports:
//
//   - `buildMatrixEditor(host, selector, A, onChange, opts)` — replaces
//     the contents of `host.querySelector(selector)` with a styled grid
//     of <input type="number"> cells bound to `A[i][j]`. Mutates A in
//     place; `onChange(A)` fires after every keystroke. Returns a
//     `{ refresh, inputs }` handle so callers can re-sync values after
//     external changes (e.g. row normalisation).
//
//   - `buildMatrixDisplay(host, selector, M, opts)` — read-only sibling
//     for showing computed matrices/vectors (eigenvalues, σ, μ*, …) in
//     the same monospace style so the page looks consistent.
//
//   - `buildVectorPanel(host, selector, items, opts)` — labelled rows
//     of vectors with optional hover callback. Used by the eigenvector
//     widget to highlight an eigenvector arrow on the plot when its
//     row is hovered.
//
// All three target the `.widget-matrix` / `.widget-matrix-display` /
// `.widget-vector-panel` CSS rules in `widgets/shared/widgets.css`.

import { fmt } from "./helpers.js";

/**
 * Render an editable matrix grid. `A` is a rows×cols array-of-arrays
 * that is mutated in place; the caller's `onChange(A)` is invoked on
 * every input event so the widget can re-render plots.
 *
 * @param {Element} host
 * @param {string} selector  — child element to populate (cleared first).
 * @param {number[][]} A     — initial values, mutated on edit.
 * @param {(A: number[][]) => void} onChange
 * @param {Object} [opts]
 * @param {number} [opts.step=0.1]
 * @param {string} [opts.rowLabel] — e.g. "row {i}: " prefix
 * @param {string[]} [opts.colHeaders] — header strings above each column
 * @param {string[]} [opts.rowHeaders] — header strings beside each row
 * @returns {{ refresh: () => void, inputs: HTMLInputElement[][] }}
 */
export function buildMatrixEditor(host, selector, A, onChange, opts = {}) {
  const root = host.querySelector(selector);
  if (!root) throw new Error(`buildMatrixEditor: no element matches ${selector}`);
  const { step = 0.1, colHeaders, rowHeaders } = opts;
  const rows = A.length;
  const cols = A[0].length;

  root.classList.add("widget-matrix");
  root.innerHTML = "";

  // Use a CSS grid. If row/col headers are present, add an extra
  // leading row/column for them.
  const gridCols = (rowHeaders ? 1 : 0) + cols;
  root.style.gridTemplateColumns = `repeat(${gridCols}, auto)`;

  // Column headers row (if any).
  if (colHeaders) {
    if (rowHeaders) {
      const corner = document.createElement("div");
      corner.className = "widget-matrix-corner";
      root.appendChild(corner);
    }
    for (let j = 0; j < cols; j++) {
      const h = document.createElement("div");
      h.className = "widget-matrix-header";
      h.textContent = colHeaders[j];
      root.appendChild(h);
    }
  }

  const inputs = [];
  for (let i = 0; i < rows; i++) {
    const rowInputs = [];
    if (rowHeaders) {
      const h = document.createElement("div");
      h.className = "widget-matrix-header";
      h.textContent = rowHeaders[i];
      root.appendChild(h);
    }
    for (let j = 0; j < cols; j++) {
      const input = document.createElement("input");
      input.type = "number";
      input.step = String(step);
      input.value = formatForInput(A[i][j]);
      input.className = "widget-matrix-cell";
      input.dataset.row = String(i);
      input.dataset.col = String(j);
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        if (!Number.isFinite(v)) return;
        A[i][j] = v;
        onChange(A);
      });
      root.appendChild(input);
      rowInputs.push(input);
    }
    inputs.push(rowInputs);
  }

  const refresh = () => {
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cur = inputs[i][j];
        // Only overwrite if the user isn't actively focused (avoids
        // ripping the cursor out from under them during normalisation).
        if (document.activeElement !== cur) {
          cur.value = formatForInput(A[i][j]);
        }
      }
    }
  };

  return { refresh, inputs };
}

/**
 * Render a read-only matrix display (computed values). Same visual
 * style as the editor for consistency.
 *
 * @param {Element} host
 * @param {string} selector
 * @param {number[][]} M
 * @param {Object} [opts]
 * @param {string[]} [opts.colHeaders]
 * @param {string[]} [opts.rowHeaders]
 * @param {string} [opts.title]
 */
export function buildMatrixDisplay(host, selector, M, opts = {}) {
  const root = host.querySelector(selector);
  if (!root) return;
  const { colHeaders, rowHeaders, title } = opts;
  root.innerHTML = "";
  root.classList.add("widget-matrix-display");

  if (title) {
    const t = document.createElement("div");
    t.className = "widget-matrix-title";
    t.textContent = title;
    root.appendChild(t);
  }

  const grid = document.createElement("div");
  grid.className = "widget-matrix";
  const rows = M.length;
  const cols = M[0].length;
  const gridCols = (rowHeaders ? 1 : 0) + cols;
  grid.style.gridTemplateColumns = `repeat(${gridCols}, auto)`;

  if (colHeaders) {
    if (rowHeaders) {
      const corner = document.createElement("div");
      corner.className = "widget-matrix-corner";
      grid.appendChild(corner);
    }
    for (let j = 0; j < cols; j++) {
      const h = document.createElement("div");
      h.className = "widget-matrix-header";
      h.textContent = colHeaders[j];
      grid.appendChild(h);
    }
  }

  for (let i = 0; i < rows; i++) {
    if (rowHeaders) {
      const h = document.createElement("div");
      h.className = "widget-matrix-header";
      h.textContent = rowHeaders[i];
      grid.appendChild(h);
    }
    for (let j = 0; j < cols; j++) {
      const cell = document.createElement("div");
      cell.className = "widget-matrix-cell widget-matrix-cell-readonly";
      cell.textContent = fmt(M[i][j]);
      grid.appendChild(cell);
    }
  }

  root.appendChild(grid);
}

/**
 * Render a vertical list of vectors / scalars with labels. Optionally
 * accepts hover callbacks per row — useful for highlighting an arrow
 * on a plot when the corresponding eigenvector row is hovered.
 *
 * @param {Element} host
 * @param {string} selector
 * @param {{ label: string, values: number[], onEnter?: () => void,
 *           onLeave?: () => void }[]} items
 * @param {Object} [opts]
 * @param {string} [opts.title]
 */
export function buildVectorPanel(host, selector, items, opts = {}) {
  const root = host.querySelector(selector);
  if (!root) return;
  const { title } = opts;
  root.innerHTML = "";
  root.classList.add("widget-vector-panel");

  if (title) {
    const t = document.createElement("div");
    t.className = "widget-matrix-title";
    t.textContent = title;
    root.appendChild(t);
  }

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "widget-vector-row";
    const label = document.createElement("span");
    label.className = "widget-vector-label";
    label.textContent = item.label;
    row.appendChild(label);
    const vals = document.createElement("span");
    vals.className = "widget-vector-values";
    vals.textContent = "[" + item.values.map(fmt).join(", ") + "]";
    row.appendChild(vals);
    if (item.onEnter) row.addEventListener("mouseenter", item.onEnter);
    if (item.onLeave) row.addEventListener("mouseleave", item.onLeave);
    if (item.onEnter || item.onLeave) row.classList.add("widget-vector-row-hoverable");
    root.appendChild(row);
  }
}

// Tidy number → string for an <input type="number">: trim trailing
// zeros, avoid Infinity/NaN spilling into the field.
function formatForInput(v) {
  if (!Number.isFinite(v)) return "0";
  // Round to 4 decimal places, then strip trailing zeros.
  const s = v.toFixed(4);
  return s.replace(/\.?0+$/, "") || "0";
}

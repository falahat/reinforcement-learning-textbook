// Widget 3.E — Bellman as a linear system (Chapter 3).
//
// For a fixed policy π on a small finite MDP, the Bellman expectation
// equation is the linear system
//
//   V^π = r^π + γ P^π V^π
//     ⇒  V^π = (I − γ P^π)⁻¹ r^π
//
// where P^π(s, s') = Σ_a π(a|s) P(s'|s,a)  is the policy-induced
// transition matrix, and r^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) R(s,a,s')
// is the expected immediate reward under π.
//
// Hand-rolled linear algebra (no external libs) for n ≤ 4 states. The
// reader edits P^π directly as a grid of cells (rows auto-normalise to
// sum to 1 so P stays stochastic) and r^π as a row vector. The widget
// shows:
//
//   - P^π as an editable numerical grid AND a heatmap so readers see
//     both the literal numbers AND the spatial pattern;
//   - r^π as an editable row vector;
//   - the closed-form V^π = (I − γP)⁻¹ r computed via Gauss–Jordan;
//   - iterative T^π V_k at k = 1, 2, 5, 20 as small heatmaps that
//     converge to the closed-form answer.
//
// Custom mount (not defineWidget) because we need persistent matrix
// state across γ-slider changes and across the matrix-editor edits.
//
// Mount: in §3.6 of Chapter 3.
//
//     <div id="ch3-bellman-linear-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/bellman_linear/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import {
  buildMatrixEditor,
  buildMatrixDisplay,
} from "../shared/matrix_editor.js";

const HOST_ID = "ch3-bellman-linear-widget";
const N_FIXED = 4;
const ITER_STEPS = [1, 2, 5, 20];

const SCAFFOLD_HTML = `
  <div class="widget-controls">
    <label>γ
      <input type="range" min="0" max="0.99" step="0.01" value="0.9" data-input="gamma">
    </label>
    <button data-button="reset-P">↺ default P</button>
    <button data-button="reset-r">↺ default r</button>
    <span data-readout></span>
  </div>
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">P^π (rows auto-normalise to 1)</div>
        <div data-matrix="P"></div>
      </div>
      <div>
        <div class="widget-matrix-title">r^π (immediate reward by state)</div>
        <div data-matrix="r"></div>
      </div>
      <div data-display="vstar"></div>
    </div>
    <div style="flex:1;min-width:380px">
      <div data-plot="heatmap"></div>
      <div data-plot="iterates"></div>
    </div>
  </div>
`;

// Default P^π: forward drift with self-loop.
function defaultP(n) {
  const P = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 0.4;
    row[(i + 1) % n] = 0.35;
    row[(i + n - 1) % n] = 0.25;
    P.push(row);
  }
  return P;
}

// Default r^π: reward at the last state.
function defaultR(n) {
  const r = new Array(n).fill(0);
  r[n - 1] = 1.0;
  return r;
}

function normaliseRow(row) {
  const total = row.reduce((s, x) => s + Math.max(0, x), 0);
  if (total <= 0) return row.map(() => 1 / row.length);
  return row.map((x) => Math.max(0, x) / total);
}

function solve(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    }
    if (Math.abs(M[pivot][i]) < 1e-12) return null;
    if (pivot !== i) [M[i], M[pivot]] = [M[pivot], M[i]];
    const div = M[i][i];
    for (let j = i; j <= n; j++) M[i][j] /= div;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      if (factor === 0) continue;
      for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
    }
  }
  return M.map((row) => row[n]);
}

function applyT(P, r, V, gamma) {
  const n = P.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let acc = r[i];
    for (let j = 0; j < n; j++) acc += gamma * P[i][j] * V[j];
    out[i] = acc;
  }
  return out;
}

function renderMatrix(M, title, valueDomain, height = 220, width = 240) {
  const cells = [];
  const n = M.length;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < M[i].length; j++) {
      cells.push({ i, j, v: M[i][j] });
    }
  }
  return Plot.plot({
    ...plotDefaults,
    width, height,
    marginTop: 28,
    marginLeft: 28,
    marginBottom: 24,
    title,
    x: { label: "s'", domain: d3.range(M[0].length), type: "band" },
    y: { label: "s", domain: d3.range(n), type: "band" },
    color: { type: "linear", scheme: "Blues", domain: valueDomain },
    marks: [
      Plot.cell(cells, { x: "j", y: "i", fill: "v", inset: 0.5 }),
      Plot.text(cells, {
        x: "j", y: "i",
        text: (d) => fmt(d.v),
        fill: (d) => d.v > (valueDomain[0] + valueDomain[1]) / 2 ? "white" : "#222",
        fontSize: 9,
      }),
    ],
  });
}

function renderVector(V, title, valueDomain, height = 220) {
  const cells = V.map((v, i) => ({ i, v }));
  return Plot.plot({
    ...plotDefaults,
    width: 120,
    height,
    marginTop: 28,
    marginLeft: 28,
    marginRight: 8,
    title,
    x: { label: null, domain: [0], type: "band" },
    y: { label: "s", domain: d3.range(V.length), type: "band" },
    color: { type: "linear", scheme: "RdBu", domain: valueDomain },
    marks: [
      Plot.cell(cells, { x: () => 0, y: "i", fill: "v", inset: 0.5 }),
      Plot.text(cells, {
        x: () => 0, y: "i",
        text: (d) => fmt(d.v),
        fill: "white", fontSize: 10,
      }),
    ],
  });
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  const n = N_FIXED;
  let P = defaultP(n);
  // r as a 1×n matrix so the editor can handle it.
  let rMat = [defaultR(n)];

  let editorP, editorR;

  function readGamma() {
    const el = host.querySelector('[data-input="gamma"]');
    return el ? parseFloat(el.value) : 0.9;
  }

  function render() {
    const Pnorm = P.map(normaliseRow);
    const r = rMat[0];
    const gamma = readGamma();

    // Closed-form: solve (I − γP) V = r.
    const A = Pnorm.map((row, i) =>
      row.map((p, j) => (i === j ? 1 : 0) - gamma * p)
    );
    const Vstar = solve(A, r) ?? new Array(n).fill(NaN);

    // Iterative T^π applied to V_0 = 0; snapshot at k = 1, 2, 5, 20.
    const snapshots = {};
    let V = new Array(n).fill(0);
    let prevK = 0;
    for (const k of ITER_STEPS) {
      while (prevK < k) {
        V = applyT(Pnorm, r, V, gamma);
        prevK++;
      }
      snapshots[k] = V.slice();
    }

    const allV = Vstar.concat(...ITER_STEPS.map((k) => snapshots[k]));
    const finiteV = allV.filter(Number.isFinite);
    const lo = Math.min(0, ...finiteV);
    const hi = Math.max(0, ...finiteV);
    const m = Math.max(Math.abs(lo), Math.abs(hi), 0.5);
    const valueDomain = [-m, m];

    // Heatmap: P^π and (I − γP^π)^-1.
    const Ainv = (() => {
      const inv = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let j = 0; j < n; j++) {
        const e = new Array(n).fill(0); e[j] = 1;
        const col = solve(A, e);
        if (!col) return null;
        for (let i = 0; i < n; i++) inv[i][j] = col[i];
      }
      return inv;
    })();

    const matrixContainer = document.createElement("div");
    matrixContainer.style.display = "flex";
    matrixContainer.style.gap = "8px";
    matrixContainer.style.flexWrap = "wrap";
    matrixContainer.appendChild(renderMatrix(Pnorm, "P^π (policy-induced)", [0, 1]));
    if (Ainv) {
      let invLo = Infinity, invHi = -Infinity;
      for (const row of Ainv) for (const v of row) {
        if (v < invLo) invLo = v;
        if (v > invHi) invHi = v;
      }
      matrixContainer.appendChild(renderMatrix(Ainv, "(I − γP^π)⁻¹", [invLo, invHi]));
    } else {
      const warn = document.createElement("div");
      warn.style.color = palette.danger;
      warn.style.padding = "1em";
      warn.textContent = "(I − γP^π) is singular — try γ < 1.";
      matrixContainer.appendChild(warn);
    }
    host.querySelector('[data-plot="heatmap"]').replaceChildren(matrixContainer);

    // Iterate vectors V_1, V_2, V_5, V_20, V*.
    const vecContainer = document.createElement("div");
    vecContainer.style.display = "flex";
    vecContainer.style.gap = "4px";
    vecContainer.style.flexWrap = "wrap";
    for (const k of ITER_STEPS) {
      vecContainer.appendChild(renderVector(snapshots[k], `V_${k}`, valueDomain));
    }
    vecContainer.appendChild(renderVector(Vstar, "V^π (closed)", valueDomain));
    host.querySelector('[data-plot="iterates"]').replaceChildren(vecContainer);

    // Numerical V^π display alongside the heatmap.
    buildMatrixDisplay(host, '[data-display="vstar"]', [Vstar], {
      title: "V^π (closed-form)",
      colHeaders: Vstar.map((_, i) => `s=${i}`),
    });

    // Convergence diagnostic.
    const linfErr = (k) => {
      const Vk = snapshots[k];
      let e = 0;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(Vk[i] - Vstar[i]);
        if (d > e) e = d;
      }
      return e;
    };
    const errs = ITER_STEPS.map((k) => `‖V_${k} − V^π‖∞=${fmt(linfErr(k))}`);

    host.querySelector("[data-readout]").innerHTML =
      `<strong>closed-form</strong> V^π = (I − γP^π)⁻¹ r^π via Gauss–Jordan; ` +
      `<strong>iterative</strong> V_{k+1} = r^π + γP^π V_k.<br>` +
      `<small>γ = ${fmt(gamma)} · ${errs.join(" · ")}</small>`;

    // Sync editors with normalised values.
    P = Pnorm;
    if (editorP) editorP.refresh();
    if (editorR) editorR.refresh();
  }

  editorP = buildMatrixEditor(host, '[data-matrix="P"]', P, () => render(), {
    step: 0.05,
    rowHeaders: d3.range(n).map((i) => `s=${i}`),
    colHeaders: d3.range(n).map((j) => `s'=${j}`),
  });
  editorR = buildMatrixEditor(host, '[data-matrix="r"]', rMat, () => render(), {
    step: 0.1,
    rowHeaders: ["r^π"],
    colHeaders: d3.range(n).map((i) => `s=${i}`),
  });

  host.querySelector('[data-input="gamma"]').addEventListener("input", render);
  host.querySelector('[data-button="reset-P"]').addEventListener("click", () => {
    const d = defaultP(n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) P[i][j] = d[i][j];
    render();
  });
  host.querySelector('[data-button="reset-r"]').addEventListener("click", () => {
    const d = defaultR(n);
    for (let i = 0; i < n; i++) rMat[0][i] = d[i];
    render();
  });

  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

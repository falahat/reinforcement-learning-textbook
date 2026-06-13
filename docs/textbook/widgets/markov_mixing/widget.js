// Widget 1.H — Markov-chain mixing & second-eigenvalue race (Chapter 1, §1.2).
//
// A row-stochastic transition matrix P (default 3×3) shown as an editable
// grid of cells. Each row auto-renormalises to sum to 1 after edits so the
// matrix stays a valid stochastic matrix (and the all-ones vector remains
// a right eigenvector with λ₁ = 1 — the fact §1.2 spends a page proving).
//
// The widget iterates μ_{k+1} = μ_k P from a user-editable μ₀ and plots
// ‖μ_k − μ*‖₁ on a log y-axis. Overlay a dashed line of slope log|λ₂|
// passing through the first iterate — the empirical convergence should
// hug the dashed reference, making "mixing rate = |λ₂|" visually obvious.
//
// New: the current distribution μ_k is shown as a horizontal bar chart
// alongside the convergence plot, with the stationary μ* drawn as a
// faint reference bar so readers see μ_k physically settling onto μ*.
//
// Eigenvalues of P (numeric, for n ≤ 3) are displayed too. λ₁ is always
// 1 for a stochastic matrix; |λ₂| is the mixing rate. For 3×3 we
// compute via the cubic characteristic polynomial closed form.
//
// Custom mount (not defineWidget) so we can host an editable matrix grid
// + a step-size slider + a bar-chart slot.
//
// Mount: `<div id="ch1-markov-mixing-widget" class="textbook-widget"></div>`
//        `<script type="module" src="./widgets/markov_mixing/widget.js"></script>`

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import {
  buildMatrixEditor,
  buildMatrixDisplay,
} from "../shared/matrix_editor.js";

const HOST_ID = "ch1-markov-mixing-widget";
const N_STEPS = 40;
const EPS = 1e-16;
const N_STATES = 3;

const SCAFFOLD_HTML = `
  <div class="widget-controls">
    <label>step k to display
      <input type="range" min="0" max="${N_STEPS}" step="1" value="0" data-input="k">
    </label>
    <button data-button="reset-P">↺ default P</button>
    <button data-button="reset-mu">↺ default μ₀</button>
    <span data-readout></span>
  </div>
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">P (each row auto-normalises to 1)</div>
        <div data-matrix="P"></div>
      </div>
      <div>
        <div class="widget-matrix-title">μ₀ (initial distribution; auto-normalises)</div>
        <div data-matrix="mu0"></div>
      </div>
      <div data-display="eigvals"></div>
      <div data-display="stationary"></div>
    </div>
    <div style="flex:1;min-width:380px">
      <div data-plot="bars"></div>
      <div data-plot="dist"></div>
      <div data-plot="err"></div>
    </div>
  </div>
`;

// Default P: 3-state, biased forward drift with self-loop and small backward.
function defaultP() {
  const n = N_STATES;
  const P = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 0.5;
    row[(i + 1) % n] = 0.3;
    row[(i + n - 1) % n] = 0.2;
    P.push(row);
  }
  return P;
}

function defaultMu() {
  return [[1, 0, 0]];
}

// Normalise a row to sum to 1. Returns the row unchanged if all
// entries are zero (so the user isn't denied an "all-zero" intermediate
// editing state, but downstream math will fix this).
function normaliseRow(row) {
  const total = row.reduce((s, x) => s + Math.max(0, x), 0);
  if (total <= 0) return row.map(() => 1 / row.length);
  return row.map((x) => Math.max(0, x) / total);
}

// Matrix-vector iterate μ_{k+1} = μ_k P with μ as a row vector.
function step(P, mu) {
  const n = P.length;
  const out = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (let i = 0; i < n; i++) acc += mu[i] * P[i][j];
    out[j] = acc;
  }
  return out;
}

// Real eigenvalues of a 3×3 matrix via the cubic characteristic
// polynomial λ³ − tr·λ² + c₁·λ − det = 0 where c₁ = sum of 2×2 principal
// minors. Uses Cardano's formula. Returns 1 or 3 real values, sorted by
// |λ| descending. For n=2 we use the scalar formula. For our purposes
// (verifying λ₁=1 and reporting |λ₂|) numerical-precision real roots
// are enough.
function eigenvalues(P) {
  const n = P.length;
  if (n === 2) {
    const tr = P[0][0] + P[1][1];
    const det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
    const disc = tr * tr - 4 * det;
    if (disc < 0) {
      return [{ re: tr / 2, im: Math.sqrt(-disc) / 2 },
              { re: tr / 2, im: -Math.sqrt(-disc) / 2 }];
    }
    const s = Math.sqrt(disc);
    return [{ re: (tr + s) / 2, im: 0 }, { re: (tr - s) / 2, im: 0 }];
  }
  // n === 3.
  const tr = P[0][0] + P[1][1] + P[2][2];
  const m01 = P[0][0] * P[1][1] - P[0][1] * P[1][0];
  const m02 = P[0][0] * P[2][2] - P[0][2] * P[2][0];
  const m12 = P[1][1] * P[2][2] - P[1][2] * P[2][1];
  const c1 = m01 + m02 + m12;
  const det = P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
            - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
            + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]);
  // Cubic: λ³ − tr·λ² + c1·λ − det = 0. Depressed form via λ = t + tr/3.
  const a = -tr, b = c1, c = -det;
  const p = b - (a * a) / 3;
  const q = (2 * a * a * a) / 27 - (a * b) / 3 + c;
  const disc = (q * q) / 4 + (p * p * p) / 27;
  const shift = -a / 3;
  if (disc > 1e-12) {
    // One real root.
    const sq = Math.sqrt(disc);
    const u = Math.cbrt(-q / 2 + sq);
    const v = Math.cbrt(-q / 2 - sq);
    const t = u + v;
    const re = t + shift;
    // Two complex conjugates.
    const reC = -t / 2 + shift;
    const imC = (Math.sqrt(3) / 2) * (u - v);
    return [{ re, im: 0 }, { re: reC, im: imC }, { re: reC, im: -imC }];
  }
  // Three real roots (trigonometric form). Guard p ≈ 0 (degenerate
  // triple-root cubic) by falling back to the shift only.
  if (Math.abs(p) < 1e-12) {
    return [0, 1, 2].map(() => ({ re: shift, im: 0 }));
  }
  const r = Math.sqrt(-(p * p * p) / 27);
  const cosArg = Math.max(-1, Math.min(1, -q / (2 * r)));
  const phi = Math.acos(cosArg);
  const mag = 2 * Math.sqrt(-p / 3);
  return [0, 1, 2].map((k) => ({
    re: mag * Math.cos((phi + 2 * Math.PI * k) / 3) + shift,
    im: 0,
  }));
}

// Stationary distribution: left eigenvector of P with eigenvalue 1.
// Solve μ (P − I) = 0 with Σ μ_i = 1. Cheap Gauss-style elimination
// for n ≤ 3 — replace the last equation with the normalisation
// constraint and back-solve.
function stationary(P) {
  const n = P.length;
  // System: μ^T (P − I) = 0 → (P^T − I) μ = 0.
  const A = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) row.push(P[j][i] - (i === j ? 1 : 0));
    A.push(row);
  }
  // Replace last row with Σ μ_i = 1.
  A[n - 1] = new Array(n).fill(1);
  const b = new Array(n).fill(0);
  b[n - 1] = 1;
  // Gauss–Jordan.
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    }
    if (Math.abs(M[pivot][i]) < 1e-12) return new Array(n).fill(1 / n);
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
  return M.map((row) => Math.max(0, row[n]));
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  let P = defaultP();
  // μ₀ as a 1×n matrix so the editor can treat it as a "matrix" too.
  let muMat = defaultMu();

  let editorP, editorMu;

  function readK() {
    const el = host.querySelector('[data-input="k"]');
    return el ? Math.round(parseFloat(el.value)) : 0;
  }

  function render() {
    // Normalise rows for use in math (without writing back to the
    // editor cells — the user keeps editing raw values; we display
    // the normalised result in the readout).
    const Pnorm = P.map(normaliseRow);
    const mu0 = normaliseRow(muMat[0]);
    const muStar = stationary(Pnorm);
    const eigs = eigenvalues(Pnorm);
    // Sort by |λ| descending. λ₁ should be ≈ 1.
    eigs.sort((a, b) => Math.hypot(b.re, b.im) - Math.hypot(a.re, a.im));
    const lambda2 = eigs[1] ?? { re: 0, im: 0 };
    const absL2 = Math.hypot(lambda2.re, lambda2.im);

    // Iterate μ_{k+1} = μ_k P.
    let mu = mu0.slice();
    const trajectory = [mu.slice()];
    const distData = mu.map((p, i) => ({ k: 0, state: String(i + 1), p }));
    const errData = [{ k: 0, err: Math.max(l1(mu, muStar), EPS) }];
    for (let k = 1; k <= N_STEPS; k++) {
      mu = step(Pnorm, mu);
      trajectory.push(mu.slice());
      for (let i = 0; i < mu.length; i++) {
        distData.push({ k, state: String(i + 1), p: mu[i] });
      }
      errData.push({ k, err: Math.max(l1(mu, muStar), EPS) });
    }

    const kView = Math.min(N_STEPS, readK());
    const muView = trajectory[kView];

    // Bars: μ_k(state) and μ*(state) side by side.
    const barRows = [];
    for (let i = 0; i < muView.length; i++) {
      barRows.push({ state: String(i + 1), kind: `μ_${kView}`, p: muView[i] });
      barRows.push({ state: String(i + 1), kind: "μ*", p: muStar[i] });
    }
    host.querySelector('[data-plot="bars"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      width: 380,
      marginLeft: 36,
      x: { label: "probability", domain: [0, 1], grid: true },
      y: { label: "state", domain: barRows.map((d) => d.state) },
      fx: { domain: [`μ_${kView}`, "μ*"], label: null },
      color: {
        domain: [`μ_${kView}`, "μ*"],
        range: [palette.primary, palette.muted],
        legend: true,
      },
      marks: [
        Plot.barX(barRows, {
          x: "p", y: "state", fx: "kind", fill: "kind",
          fillOpacity: (d) => d.kind === "μ*" ? 0.45 : 0.85,
        }),
        Plot.text(barRows, {
          x: "p", y: "state", fx: "kind",
          text: (d) => fmt(d.p), dx: 6,
          textAnchor: "start", fill: "#ddd", fontSize: 10,
        }),
      ],
    }));

    // Distribution-over-time line plot.
    host.querySelector('[data-plot="dist"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      width: 380,
      x: { label: "step k", grid: true },
      y: { domain: [0, 1], label: "μ_k(state)", grid: true },
      color: {
        domain: muView.map((_, i) => String(i + 1)),
        range: [palette.primary, palette.secondary, palette.accent].slice(0, muView.length),
        legend: true,
      },
      marks: [
        Plot.line(distData, { x: "k", y: "p", stroke: "state", strokeWidth: 2 }),
        Plot.ruleX([kView], { stroke: palette.warning, ...dashed }),
        ...muStar.map((p, i) =>
          Plot.ruleY([p], {
            stroke: [palette.primary, palette.secondary, palette.accent][i],
            ...dashed, strokeOpacity: 0.5,
          })
        ),
      ],
    }));

    // Convergence on log scale.
    const yMin = Math.max(EPS, d3.min(errData, (d) => d.err) * 0.5);
    const yMax = Math.max(errData[0].err, 1);
    const refLine = [];
    if (errData[0].err > EPS && absL2 > 1e-9 && absL2 < 1) {
      for (let k = 0; k <= N_STEPS; k++) {
        refLine.push({ k, err: Math.max(errData[0].err * Math.pow(absL2, k), EPS) });
      }
    }
    host.querySelector('[data-plot="err"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      width: 380,
      x: { label: "step k", grid: true },
      y: { type: "log", label: "‖μ_k − μ*‖₁", grid: true, domain: [yMin, yMax] },
      marks: [
        Plot.line(errData, { x: "k", y: "err", stroke: palette.primary, strokeWidth: 2 }),
        Plot.dot(errData, { x: "k", y: "err", fill: palette.primary, r: 2 }),
        ...(refLine.length ? [Plot.line(refLine, {
          x: "k", y: "err", stroke: palette.danger, strokeWidth: 1.5, ...dashed,
        })] : []),
        Plot.text(
          refLine.length ? [{ x: N_STEPS, y: refLine[refLine.length - 1].err,
                              label: `slope = |λ₂|^k = ${fmt(absL2)}^k` }] : [],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    // Numerical panels.
    const eigRow = eigs.map((e) => Math.abs(e.im) < 1e-9 ? e.re : NaN);
    buildMatrixDisplay(host, '[data-display="eigvals"]', [eigRow], {
      title: "eigenvalues (real parts)",
      colHeaders: eigs.map((_, i) => `λ${i + 1}`),
    });
    buildMatrixDisplay(host, '[data-display="stationary"]', [muStar], {
      title: "stationary μ* (highlighted)",
      colHeaders: muStar.map((_, i) => `μ*(${i + 1})`),
    });

    // Readout.
    const eigStr = eigs
      .map((e) => Math.abs(e.im) < 1e-9 ? fmt(e.re) : `${fmt(e.re)}±${fmt(Math.abs(e.im))}i`)
      .join(", ");
    host.querySelector("[data-readout]").innerHTML =
      `<strong>λ = [${eigStr}]</strong> · |λ₂| = ${fmt(absL2)} (mixing rate per step) · ` +
      `<small>showing μ_${kView}</small>`;

    // Keep the editor cells in sync with normalised values when the
    // user isn't actively typing in them (so they SEE that the row
    // they typed got normalised).
    P = Pnorm;
    muMat[0] = mu0;
    if (editorP) editorP.refresh();
    if (editorMu) editorMu.refresh();
  }

  function l1(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
  }

  editorP = buildMatrixEditor(host, '[data-matrix="P"]', P, () => render(), {
    step: 0.01,
    rowHeaders: ["s=1", "s=2", "s=3"],
    colHeaders: ["s'=1", "s'=2", "s'=3"],
  });
  editorMu = buildMatrixEditor(host, '[data-matrix="mu0"]', muMat, () => render(), {
    step: 0.01,
    rowHeaders: ["μ₀"],
    colHeaders: ["s=1", "s=2", "s=3"],
  });

  host.querySelector('[data-input="k"]').addEventListener("input", render);
  host.querySelector('[data-button="reset-P"]').addEventListener("click", () => {
    const d = defaultP();
    for (let i = 0; i < P.length; i++)
      for (let j = 0; j < P[i].length; j++) P[i][j] = d[i][j];
    render();
  });
  host.querySelector('[data-button="reset-mu"]').addEventListener("click", () => {
    const d = defaultMu();
    for (let j = 0; j < muMat[0].length; j++) muMat[0][j] = d[0][j];
    render();
  });

  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

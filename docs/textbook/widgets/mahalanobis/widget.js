// Widget 1.L — Mahalanobis "weird ruler" (Chapter 1, §1.5).
//
// Two ways to measure the length of the same vector v ∈ ℝ²:
//   ‖v‖₂ = √(vᵀ v)              (Euclidean)
//   ‖v‖_S = √(vᵀ S v)            (Mahalanobis, with PD matrix S)
//
// The unit ball under each norm:
//   { v : ‖v‖₂ ≤ 1 } is the circle.
//   { v : ‖v‖_S ≤ 1 } is the ellipse with semi-axis lengths 1/√λᵢ(S)
//   along the eigenvectors of S.
//
// Edit S (symmetric, auto-symmetrised). Drag v with sliders. Watch the
// vector's two "rulers" disagree: a short Euclidean vector can be
// Mahalanobis-long if it points along a high-variance direction of S.
// That's the entire pedagogical point.
//
// Mount:
//   <div id="ch1-mahalanobis-widget" class="textbook-widget"></div>

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor } from "../shared/matrix_editor.js";
import { eig2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-mahalanobis-widget";

const SCAFFOLD_HTML = `
  <div class="widget-controls">
    <label>v.x
      <input type="range" min="-2" max="2" step="0.05" value="1.0" data-input="vx">
    </label>
    <label>v.y
      <input type="range" min="-2" max="2" step="0.05" value="0.3" data-input="vy">
    </label>
    <span data-readout></span>
  </div>
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">S (symmetric PD; off-diag auto-mirrored)</div>
        <div data-matrix="S"></div>
      </div>
      <div data-display="readouts"></div>
    </div>
    <div data-plot="main"></div>
  </div>
`;

function circlePoints(N = 80) {
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * 2 * Math.PI;
    pts.push({ x: Math.cos(t), y: Math.sin(t) });
  }
  return pts;
}

function ellipsePoints(eigs, N = 80) {
  // Unit ball of ‖·‖_S: vᵀ S v = 1.
  // Parametrise as v = (cos θ / √λ₁) u₁ + (sin θ / √λ₂) u₂ where u_i
  // are eigenvectors and λ_i eigenvalues.
  const a = 1 / Math.sqrt(Math.max(1e-9, eigs[0].lambda));
  const b = 1 / Math.sqrt(Math.max(1e-9, eigs[1].lambda));
  const u = eigs[0].v;
  const w = eigs[1].v;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = (i / N) * 2 * Math.PI;
    const ax = a * Math.cos(t);
    const ay = b * Math.sin(t);
    pts.push({ x: ax * u[0] + ay * w[0], y: ax * u[1] + ay * w[1] });
  }
  return pts;
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  const S = [[2, 0.5], [0.5, 1]];

  function readV() {
    return [
      parseFloat(host.querySelector('[data-input="vx"]').value),
      parseFloat(host.querySelector('[data-input="vy"]').value),
    ];
  }

  function render() {
    // Auto-symmetrise: S[0][1] = S[1][0] = avg.
    const mid = (S[0][1] + S[1][0]) / 2;
    S[0][1] = mid;
    S[1][0] = mid;

    const [vx, vy] = readV();
    const { real, eigs } = eig2x2(S);
    const pd = real && eigs[0].lambda > 0 && eigs[1].lambda > 0;

    const euclid = Math.hypot(vx, vy);
    const quad = S[0][0] * vx * vx + 2 * S[0][1] * vx * vy + S[1][1] * vy * vy;
    const maha = quad >= 0 ? Math.sqrt(quad) : NaN;

    const circle = circlePoints();
    const ellipse = pd ? ellipsePoints(eigs) : [];

    // Eigenvector semi-axes (only if PD; otherwise ellipse is undefined).
    const axes = [];
    if (pd) {
      for (const e of eigs) {
        const s = 1 / Math.sqrt(e.lambda);
        axes.push({
          x1: 0, y1: 0,
          x2: e.v[0] * s, y2: e.v[1] * s,
          label: `λ=${fmt(e.lambda)}`,
        });
      }
    }

    const R = 3;

    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 400, width: 420,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [-R, R], label: "v.x", grid: true },
      y: { domain: [-R, R], label: "v.y", grid: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Euclidean unit ball (blue circle).
        Plot.line(circle, {
          x: "x", y: "y", stroke: palette.secondary, strokeWidth: 1.5,
          strokeDasharray: "3 3",
        }),
        // Mahalanobis unit ball (green ellipse).
        pd
          ? Plot.line(ellipse, {
              x: "x", y: "y", stroke: palette.primary, strokeWidth: 2,
            })
          : null,
        // Eigenvector axes (dashed green).
        axes.length
          ? Plot.link(axes, {
              x1: "x1", y1: "y1", x2: "x2", y2: "y2",
              stroke: palette.primary, strokeWidth: 1.2, strokeDasharray: "2 2",
            })
          : null,
        // v as an arrow from origin.
        Plot.arrow([{ x1: 0, y1: 0, x2: vx, y2: vy }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.warning, strokeWidth: 3, headLength: 10,
        }),
        Plot.text([{ x: vx, y: vy, label: "v" }], {
          x: "x", y: "y", text: "label", dx: 8, dy: -4,
          fill: palette.warning, ...annotation,
        }),
      ].filter(Boolean),
    }));

    host.querySelector('[data-display="readouts"]').innerHTML = `
      <div class="widget-matrix-title">readouts</div>
      <div class="widget-matrix-display">
        <div>v = (${fmt(vx)}, ${fmt(vy)})</div>
        <div>‖v‖₂ = ${fmt(euclid)} <span style="color:${palette.secondary}">●</span></div>
        <div>‖v‖_S = ${fmt(maha)} <span style="color:${palette.primary}">●</span></div>
        ${real ? `<div>λ₁(S) = ${fmt(eigs[0].lambda)}</div>` : ""}
        ${real ? `<div>λ₂(S) = ${fmt(eigs[1].lambda)}</div>` : ""}
        <div style="color:${pd ? palette.primary : palette.danger}">
          <b>${pd ? "PD ✓" : "NOT positive-definite"}</b>
        </div>
      </div>
    `;
    host.querySelector("[data-readout]").textContent =
      `‖v‖₂ = ${fmt(euclid)}  ·  ‖v‖_S = ${fmt(maha)}  ·  ratio = ${fmt(maha / Math.max(euclid, 1e-9))}`;
  }

  buildMatrixEditor(host, '[data-matrix="S"]', S, render, {
    step: 0.1,
    colHeaders: ["col 1", "col 2"],
  });

  for (const el of host.querySelectorAll("input")) {
    el.addEventListener("input", render);
  }
  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

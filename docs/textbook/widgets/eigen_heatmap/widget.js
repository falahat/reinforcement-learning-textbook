// Widget 1.E — Eigendirection heatmap (Chapter 1, §1.6).
//
// For every unit vector v(θ) = (cos θ, sin θ) around the unit circle,
// compute the *line-angle deviation* between v(θ) and its image
// Av(θ): the smallest angle between the two lines through the
// origin. An eigenvector direction is exactly where this deviation
// hits zero — A maps v back onto its own line (Av = λv, so v and Av
// are parallel for λ > 0 or antiparallel for λ < 0, but in either
// case the *line* is preserved).
//
// The deviation is painted as a colour around a ring at radius ~1.1,
// dark at eigendirections and bright where the rotation is largest.
// A second sub-plot shows deviation vs θ as a line — same data,
// different geometry: zeros there mark the eigenvector angles.
//
// A user-controlled probe angle θ draws the *actual* input vector
// v(θ) (faint) and its image Av(θ) (bold) on the polar plot, so the
// reader can park the cursor on any heatmap colour and see the
// arrows it represents.
//
// Mount:
//   <div id="ch1-eigen-heatmap-widget" class="textbook-widget"></div>
//   <script type="module" src="./widgets/eigen_heatmap/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { buildMatrixEditor, buildMatrixDisplay } from "../shared/matrix_editor.js";
import { matvec2x2, eig2x2, det2x2, trace2x2 } from "../shared/linalg.js";

const HOST_ID = "ch1-eigen-heatmap-widget";
// The deviation depends only on θ (A is linear ⇒ A(rv) is parallel to
// Av for every r > 0), so we can fill the *entire* unit disk by painting
// radial spokes coloured by the deviation at their angle. The disk is
// far more readable than a thin band: the dark eigendirection spokes
// run all the way through the centre and become impossible to miss.
const N_SPOKES = 720; // angular samples — half-degree resolution
const N_LINE = 360; // angular samples for the line plot — one per degree
const SPOKE_R = 1.0; // outer radius of the disk = the unit circle
const SPOKE_WIDTH = 2.4; // SVG-pixel stroke width per spoke (overlaps)

// Unsigned line-angle deviation between v(θ) and A·v(θ), in radians,
// returned in [0, π/2]. Zero ⇔ v and Av are parallel OR antiparallel
// ⇔ v is an eigendirection (real eigenvalue, any sign).
function lineDeviation(A, theta) {
  const v = [Math.cos(theta), Math.sin(theta)];
  const w = matvec2x2(A, v);
  const wn = Math.hypot(w[0], w[1]);
  if (wn < 1e-12) return Math.PI / 2; // A·v collapses to origin → max
  // Use |dot|/|w| because v and Av point along *lines*; both v and
  // -v identify the same line.
  const cosArg = Math.abs(v[0] * w[0] + v[1] * w[1]) / wn;
  return Math.acos(Math.min(1, cosArg));
}

// Signed angle change (Av rotated by how much from v), in (-π, π].
// Useful for the second view — a pure rotation matrix shows this
// constant across all θ; the identity shows 0; a sign-flip shows ±π.
function signedRotation(A, theta) {
  const v = [Math.cos(theta), Math.sin(theta)];
  const w = matvec2x2(A, v);
  const wn = Math.hypot(w[0], w[1]);
  if (wn < 1e-12) return NaN;
  const inAngle = Math.atan2(v[1], v[0]);
  const outAngle = Math.atan2(w[1], w[0]);
  let d = outAngle - inAngle;
  // Wrap to (-π, π].
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

// Convert a unit-line angle from radians to degrees, formatted.
const deg = (rad) => `${(rad * 180 / Math.PI).toFixed(1)}°`;

const SCAFFOLD = `
  <div class="widget-controls" style="flex-wrap: wrap; align-items: center;">
    <label>probe θ (degrees)
      <input type="range" min="0" max="360" step="1" value="30"
             data-input="probe">
      <input type="number" min="0" max="360" step="1" value="30"
             data-input="probe-num" style="width: 4.5em; margin-left: 0.4em;">
    </label>
    <label>view
      <select data-input="view">
        <option value="line" selected>unsigned line deviation (eigendirs at 0)</option>
        <option value="signed">signed rotation (identity at 0)</option>
      </select>
    </label>
  </div>
  <div class="widget-matrix-layout">
    <div class="widget-matrix-panel">
      <div>
        <div class="widget-matrix-title">A (edit cells)</div>
        <div data-matrix="A"></div>
      </div>
      <div data-display="eigsumm"></div>
      <div data-display="probe"></div>
    </div>
    <div style="display: flex; flex-direction: column; gap: 0.5em;">
      <div data-plot="ring"></div>
      <div data-plot="line"></div>
    </div>
  </div>
  <div data-readout></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD;

  // Default: a non-symmetric 2×2 with two real, distinct eigenvalues.
  // The two eigendirections show as two dark spokes 180° apart on the
  // ring (each direction also passes through −v, hence 4 dark spots
  // total). Two of those are the same line as the other two — i.e. v
  // and −v are one eigenvector.
  const A = [[2, 1], [0.5, 1.5]];

  const probeRange = host.querySelector('[data-input="probe"]');
  const probeNum = host.querySelector('[data-input="probe-num"]');
  const viewSel = host.querySelector('[data-input="view"]');

  function currentProbe() {
    return (parseFloat(probeRange.value) || 0) * Math.PI / 180;
  }

  function render() {
    const view = viewSel.value;
    const spokeAngles = d3.range(N_SPOKES).map((i) => (2 * Math.PI * i) / N_SPOKES);
    const lineAngles = d3.range(N_LINE).map((i) => (2 * Math.PI * i) / N_LINE);

    // Compute the per-angle value for the chosen view.
    let spokeValues, lineValues, colorScheme;
    if (view === "line") {
      spokeValues = spokeAngles.map((t) => lineDeviation(A, t));
      lineValues = lineAngles.map((t) => lineDeviation(A, t));
      colorScheme = (v) => d3.interpolateViridis(v / (Math.PI / 2));
    } else {
      spokeValues = spokeAngles.map((t) => signedRotation(A, t));
      lineValues = lineAngles.map((t) => signedRotation(A, t));
      // Diverging colormap: red (CW, negative), white (0), blue (CCW,
      // positive). The eye reads "no rotation" as the neutral band.
      colorScheme = (v) => d3.interpolateRdBu(0.5 + v / (2 * Math.PI));
    }

    // Build the heatmap disk as N_SPOKES radial lines from origin to
    // the unit circle, each coloured by the deviation at its angle.
    // Adjacent spokes overlap (strokeWidth ≈ 2.4 px ≫ outer-edge arc
    // spacing of ~1.3 px at the default plot size) so the eye reads a
    // continuous disk, with dark eigendirection spokes running all
    // the way through the centre. Each spoke is its own z-group so
    // Plot.line draws a separate stroke per spoke.
    const spokeData = [];
    spokeAngles.forEach((t, i) => {
      const fill = colorScheme(spokeValues[i]);
      spokeData.push({ x: 0, y: 0, spoke: i, color: fill });
      spokeData.push({
        x: Math.cos(t) * SPOKE_R,
        y: Math.sin(t) * SPOKE_R,
        spoke: i,
        color: fill,
      });
    });

    // Unit circle outline (drawn on top of the disk as a thin marker).
    const unitCircle = d3.range(N_LINE + 1).map((i) => {
      const t = (i * 2 * Math.PI) / N_LINE;
      return { x: Math.cos(t), y: Math.sin(t) };
    });

    // Probe vectors at θ.
    const tp = currentProbe();
    const vIn = [Math.cos(tp), Math.sin(tp)];
    const [wx, wy] = matvec2x2(A, vIn);
    const wn = Math.hypot(wx, wy);
    const probeDev = lineDeviation(A, tp);
    const probeRot = signedRotation(A, tp);

    // Eigenvectors + closed-form summary.
    const { real, eigs } = eig2x2(A);
    const tr = trace2x2(A);
    const det = det2x2(A);

    // Eigendirection markers on the ring (4 spots — each line shows
    // up at θ and θ + π). For complex eigenvalues, no markers.
    const eigMarkers = real
      ? eigs.flatMap(({ lambda, v }) => {
          const a = Math.atan2(v[1], v[0]);
          const xa = Math.cos(a);
          const ya = Math.sin(a);
          return [
            { x: xa * 1.0, y: ya * 1.0, lambda, side: "+" },
            { x: -xa * 1.0, y: -ya * 1.0, lambda, side: "−" },
          ];
        })
      : [];

    // Probe marker at the rim of the disk.
    const probeMark = [
      { x: Math.cos(tp) * SPOKE_R, y: Math.sin(tp) * SPOKE_R },
    ];

    // The polar plot — heatmap disk + unit circle + probe arrows +
    // eigendirection markers. The viewport extends past the disk so
    // the post-image arrow A·v has room when |A·v| > 1.
    const R = Math.max(SPOKE_R + 0.35, Math.hypot(wx, wy) * 1.05);
    host.querySelector('[data-plot="ring"]').replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 360,
        width: 380,
        marginLeft: 24,
        marginBottom: 24,
        x: { domain: [-R, R], axis: null },
        y: { domain: [-R, R], axis: null },
        aspectRatio: 1,
        marks: [
          // The heatmap disk: N_SPOKES coloured radial lines from
          // origin to the unit circle. Each spoke is its own z-group
          // so Plot.line strokes them independently with the colour
          // attached to that spoke's points.
          Plot.line(spokeData, {
            x: "x", y: "y", z: "spoke",
            stroke: "color", strokeWidth: SPOKE_WIDTH,
            strokeLinecap: "butt",
          }),
          // Unit circle outline (thin white-ish boundary on top of
          // the disk — separates the heatmap from the surrounding
          // arrow viewport).
          Plot.line(unitCircle, {
            x: "x", y: "y",
            stroke: "white", strokeWidth: 1.2, strokeOpacity: 0.7,
          }),
          // Axes (centred, faint — drawn on top of the disk).
          Plot.ruleX([0], { stroke: "white", strokeOpacity: 0.35, ...dashed }),
          Plot.ruleY([0], { stroke: "white", strokeOpacity: 0.35, ...dashed }),
          // Eigendirection markers (open circles on the unit circle).
          Plot.dot(eigMarkers, {
            x: "x", y: "y", r: 5,
            fill: "white", stroke: palette.warning, strokeWidth: 2,
          }),
          Plot.text(eigMarkers.filter((m) => m.side === "+"), {
            x: (d) => d.x * 1.32, y: (d) => d.y * 1.32,
            text: (d) => `λ=${fmt(d.lambda)}`,
            fill: palette.warning, fontSize: 11, fontWeight: 600,
          }),
          // Probe arrows — v(θ) faint, A·v(θ) bold.
          Plot.arrow(
            [{ x1: 0, y1: 0, x2: vIn[0], y2: vIn[1] }],
            {
              x1: "x1", y1: "y1", x2: "x2", y2: "y2",
              stroke: palette.secondary, strokeWidth: 1.7,
              strokeOpacity: 0.7, headLength: 7,
            }
          ),
          Plot.arrow(
            [{ x1: 0, y1: 0, x2: wx, y2: wy }],
            {
              x1: "x1", y1: "y1", x2: "x2", y2: "y2",
              stroke: palette.primary, strokeWidth: 2.6, headLength: 11,
            }
          ),
          Plot.text([{ x: vIn[0], y: vIn[1], label: "v(θ)" }], {
            x: "x", y: "y", text: "label",
            dx: 4, dy: -4, textAnchor: "start",
            fill: palette.secondary, fontSize: 11,
          }),
          Plot.text([{ x: wx, y: wy, label: "A·v(θ)" }], {
            x: "x", y: "y", text: "label",
            dx: 4, dy: -4, textAnchor: "start",
            fill: palette.primary, fontSize: 11, fontWeight: 600,
          }),
          // Probe pointer on the ring itself.
          Plot.dot(probeMark, {
            x: "x", y: "y", r: 6,
            fill: palette.warning, stroke: "white", strokeWidth: 1.5,
          }),
        ],
      })
    );

    // Second plot: deviation vs θ as a smooth curve. Same data,
    // unrolled so the reader sees the wave shape — period π for the
    // line view (eigenvectors at v and −v collapse), period 2π for
    // the signed view.
    const lineData = lineAngles.map((t, i) => ({
      theta: t * 180 / Math.PI,
      value: lineValues[i] * 180 / Math.PI,
    }));
    const probeMarkerLine = [{ theta: tp * 180 / Math.PI,
                               value: (view === "line" ? probeDev : probeRot) * 180 / Math.PI }];
    const eigLineMarkers = real
      ? eigs.flatMap(({ lambda, v }) => {
          const a = Math.atan2(v[1], v[0]);
          const a2 = a + Math.PI;
          // Normalise into [0, 2π) so they show in the plot range.
          const norm = (x) => ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
          return [
            { theta: norm(a) * 180 / Math.PI, lambda },
            { theta: norm(a2) * 180 / Math.PI, lambda },
          ];
        })
      : [];

    host.querySelector('[data-plot="line"]').replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 170,
        width: 380,
        marginLeft: 44,
        marginBottom: 30,
        x: { domain: [0, 360], label: "θ (degrees)", grid: true,
             ticks: [0, 45, 90, 135, 180, 225, 270, 315, 360] },
        y: {
          domain: view === "line" ? [0, 90] : [-180, 180],
          label: view === "line"
            ? "line deviation (°)"
            : "signed rotation (°)",
          grid: true,
        },
        marks: [
          ...(view === "signed"
            ? [Plot.ruleY([0], { stroke: palette.muted, ...dashed })]
            : []),
          Plot.line(lineData, {
            x: "theta", y: "value",
            stroke: palette.secondary, strokeWidth: 2,
          }),
          // Eigendirection vertical guides.
          ...(real
            ? [Plot.ruleX(eigLineMarkers, {
                x: "theta", stroke: palette.warning,
                strokeOpacity: 0.7, ...dashed,
              })]
            : []),
          Plot.dot(probeMarkerLine, {
            x: "theta", y: "value", r: 5,
            fill: palette.warning, stroke: "white", strokeWidth: 1.4,
          }),
        ],
      })
    );

    // Eigen summary panel.
    const eigPanel = host.querySelector('[data-display="eigsumm"]');
    eigPanel.innerHTML = `
      <div class="widget-matrix-title">eigen summary</div>
      <div class="widget-matrix-display">
        <div>tr(A) = ${fmt(tr)}</div>
        <div>det(A) = ${fmt(det)}</div>
        ${real
          ? eigs.map((e, i) => {
              const a = Math.atan2(e.v[1], e.v[0]) * 180 / Math.PI;
              return `<div>λ${i + 1} = ${fmt(e.lambda)} at θ ≈ ${a.toFixed(1)}°</div>`;
            }).join("")
          : `<div style="color:${palette.danger}">complex eigenvalues — no real eigendirections (deviation never hits 0)</div>`
        }
      </div>
    `;

    // Probe readout panel.
    const probePanel = host.querySelector('[data-display="probe"]');
    const inDeg = (tp * 180 / Math.PI).toFixed(1);
    probePanel.innerHTML = `
      <div class="widget-matrix-title">probe at θ = ${inDeg}°</div>
      <div class="widget-matrix-display">
        <div>v(θ) = (${fmt(vIn[0])}, ${fmt(vIn[1])})</div>
        <div>A·v(θ) = (${fmt(wx)}, ${fmt(wy)})</div>
        <div>‖A·v‖ = ${fmt(wn)}</div>
        <div>line deviation = ${deg(probeDev)}</div>
        <div>signed rotation = ${deg(probeRot)}</div>
      </div>
    `;

    const eigCount = real ? eigs.length : 0;
    host.querySelector("[data-readout]").innerHTML =
      `<small>Each angle θ on the unit disk is a unit vector v(θ). ` +
      `The disk's colour at angle θ is how far A rotates v(θ) off ` +
      `its own line — and because A is linear, every point on the ` +
      `same ray inherits that colour. ` +
      `<strong>Dark spokes are eigendirections</strong> — A maps v ` +
      `back onto the same line. ${eigCount > 0
        ? "Open circles at the rim mark them; their labels show λ."
        : "Complex eigenvalues here, so no real eigendirection exists — the disk never goes fully dark."}` +
      ` Drag the probe slider to see the input v(θ) and output A·v(θ) arrows at any angle.</small>`;
  }

  buildMatrixEditor(host, '[data-matrix="A"]', A, render, {
    step: 0.1,
    colHeaders: ["col 1", "col 2"],
  });

  // Two-way slider/number sync for the probe angle.
  probeRange.addEventListener("input", () => {
    probeNum.value = probeRange.value;
    render();
  });
  probeNum.addEventListener("input", () => {
    const v = parseFloat(probeNum.value);
    if (Number.isFinite(v)) {
      probeRange.value = ((v % 360) + 360) % 360;
      render();
    }
  });
  viewSel.addEventListener("change", render);

  render();
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

// Widget 1.L — Dot product as projection (Chapter 1, §1.2).
//
// Two draggable vectors u and v. The widget draws:
//   - u as a yellow arrow
//   - v as a blue arrow
//   - the projection of u onto the v-direction as a green arrow
//     along v (with length (u·v)/||v||)
//   - the perpendicular from u to that projection (dashed)
//
// Live readouts: u·v, ||u||·||v||·cos θ (which must equal u·v), and the
// angle θ between them. Switch sign by dragging u to the "other side"
// of v to see u·v go negative.
//
// Pedagogical points:
//   - The dot product MEASURES ALIGNMENT.
//   - u·v = 0 ⇔ orthogonal (perpendicular).
//   - u·v > 0 ⇔ same general direction; < 0 ⇔ opposite.
//   - The "shadow" of u along v has signed length (u·v)/||v||.
//
// Mount: `<div id="ch1-dot-projection-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

defineWidget({
  hostId: "ch1-dot-projection-widget",
  controls: {
    ux: { label: "u.x", min: -3, max: 3, step: 0.1, default: 2.0 },
    uy: { label: "u.y", min: -3, max: 3, step: 0.1, default: 1.5 },
    vx: { label: "v.x", min: -3, max: 3, step: 0.1, default: 2.5 },
    vy: { label: "v.y", min: -3, max: 3, step: 0.1, default: 0.4 },
  },
  render: (host, { ux, uy, vx, vy }, slots) => {
    const dot = ux * vx + uy * vy;
    const uMag = Math.hypot(ux, uy);
    const vMag = Math.hypot(vx, vy);
    const cosTheta = (uMag * vMag > 1e-9) ? dot / (uMag * vMag) : 0;
    const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)));
    const thetaDeg = (theta * 180) / Math.PI;

    // Projection of u onto v: ((u·v)/||v||²) · v
    const projScalar = vMag > 1e-9 ? dot / (vMag * vMag) : 0;
    const projX = projScalar * vx;
    const projY = projScalar * vy;
    // Foot of perpendicular at (projX, projY).

    const R = Math.max(3, Math.max(Math.abs(ux), Math.abs(uy), Math.abs(vx), Math.abs(vy)) * 1.5);
    const projColour = projScalar >= 0 ? palette.primary : palette.danger;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 420,
      marginLeft: 36, marginBottom: 32,
      aspectRatio: 1,
      x: { domain: [-R, R], label: "x", grid: true },
      y: { domain: [-R, R], label: "y", grid: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        // Long faint line along v's direction (the projection line).
        Plot.line([{ x: -R * vx / Math.max(1e-6, vMag), y: -R * vy / Math.max(1e-6, vMag) },
                   { x:  R * vx / Math.max(1e-6, vMag), y:  R * vy / Math.max(1e-6, vMag) }], {
          x: "x", y: "y", stroke: palette.secondary, strokeOpacity: 0.18,
        }),
        // u arrow (yellow).
        Plot.arrow([{ x1: 0, y1: 0, x2: ux, y2: uy }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.warning, strokeWidth: 2.5, headLength: 12,
        }),
        // v arrow (blue).
        Plot.arrow([{ x1: 0, y1: 0, x2: vx, y2: vy }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: palette.secondary, strokeWidth: 2.5, headLength: 12,
        }),
        // Projection arrow along v (green if positive, red if negative).
        Plot.arrow([{ x1: 0, y1: 0, x2: projX, y2: projY }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: projColour, strokeWidth: 3.5, headLength: 14,
        }),
        // Perpendicular from u tip to its projection.
        Plot.line([{ x: ux, y: uy }, { x: projX, y: projY }], {
          x: "x", y: "y", stroke: palette.muted, strokeOpacity: 0.7, ...dashed,
        }),
        // Labels.
        Plot.text([{ x: ux, y: uy, label: "u" }], {
          x: "x", y: "y", text: "label", dx: 8, dy: -4, fill: palette.warning, ...annotation,
        }),
        Plot.text([{ x: vx, y: vy, label: "v" }], {
          x: "x", y: "y", text: "label", dx: 8, dy: -4, fill: palette.secondary, ...annotation,
        }),
        Plot.text([{ x: projX, y: projY, label: `proj_v(u)` }], {
          x: "x", y: "y", text: "label", dx: 8, dy: 12, fill: projColour, ...annotation,
        }),
      ],
    }));

    const orthogonal = Math.abs(dot) < 0.05;
    let interp;
    if (orthogonal) interp = `<span style="color:${palette.muted}">≈ 0 — u and v are orthogonal (perpendicular)</span>`;
    else if (dot > 0) interp = `<span style="color:${palette.primary}">> 0 — u and v point in the same general direction</span>`;
    else interp = `<span style="color:${palette.danger}">< 0 — u and v point in opposite general directions</span>`;

    slots.readout.innerHTML =
      `u·v = ${fmt(ux)}·${fmt(vx)} + ${fmt(uy)}·${fmt(vy)} = <b>${fmt(dot)}</b>  ·  ` +
      `‖u‖‖v‖cos θ = ${fmt(uMag)} · ${fmt(vMag)} · ${fmt(cosTheta)} = ${fmt(uMag * vMag * cosTheta)}  ·  ` +
      `θ = ${fmt(thetaDeg)}°  ·  ${interp}`;
  },
});

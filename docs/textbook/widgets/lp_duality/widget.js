// Widget 4.F — LP formulation of V* on a 2-state MDP (Chapter 4, §4.8).
//
// V* is the solution of the LP
//   min_V  V(0) + V(1)
//   s.t.   V(s) ≥ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')    for all (s, a)
//
// We visualise the *feasibility region* in (V₀, V₁)-space. For a tiny
// 2-state MDP with 2 actions per state, there are 4 linear constraints.
// The LP's optimum is the unique lower-left vertex of the feasible
// polytope where the "min sum" objective hits an active constraint.
//
// Editable: γ, plus the four (R, sp) pairs that define the two actions
// per state. To keep the widget tractable we hardcode deterministic
// transitions: action a from state s leads to a fixed successor with
// reward R(s, a). This isolates the LP geometry from the transition
// structure — students play with R values and watch the constraint
// half-planes pivot.
//
// "Show dual" toggle: highlights the active (binding) constraint at V*,
// which corresponds to a positive occupancy measure for that (s, a)
// pair in the dual LP. The dual variable d(s, a) ≥ 0 is non-zero
// exactly when V(s) = R(s,a) + γ V(s') — i.e., the constraint is tight.
// We mark those constraints in green.
//
// Mount:
//   <div id="ch4-lp-duality-widget" class="textbook-widget"></div>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

// Fixed deterministic transitions for a 2-state MDP. Actions per state:
//   state 0, action L → stays in 0
//   state 0, action R → goes to 1
//   state 1, action L → goes to 0
//   state 1, action R → stays in 1
// Each (s, a) has a reward parameter.
const SUCCESSOR = {
  "0L": 0, "0R": 1,
  "1L": 0, "1R": 1,
};

defineWidget({
  hostId: "ch4-lp-duality-widget",
  controls: {
    gamma: { label: "γ", min: 0.1, max: 0.95, step: 0.05, default: 0.8 },
    r0L: { label: "R(0,L)", min: -1, max: 2, step: 0.1, default: 0.1 },
    r0R: { label: "R(0,R)", min: -1, max: 2, step: 0.1, default: 1.0 },
    r1L: { label: "R(1,L)", min: -1, max: 2, step: 0.1, default: 0.0 },
    r1R: { label: "R(1,R)", min: -1, max: 2, step: 0.1, default: 0.5 },
    showDual: {
      type: "select", label: "show dual",
      options: [
        { value: "off", label: "off" },
        { value: "on", label: "highlight tight constraints" },
      ],
      default: "off",
    },
  },
  slots: ["main"],
  render: (host, { gamma, r0L, r0R, r1L, r1R, showDual }, slots) => {
    // Constraint i: V(s) ≥ R + γ V(sp).
    // In (V0, V1)-plane these are half-planes.
    const constraints = [
      { s: 0, a: "L", r: r0L, sp: SUCCESSOR["0L"] },
      { s: 0, a: "R", r: r0R, sp: SUCCESSOR["0R"] },
      { s: 1, a: "L", r: r1L, sp: SUCCESSOR["1L"] },
      { s: 1, a: "R", r: r1R, sp: SUCCESSOR["1R"] },
    ];

    // Solve the small MDP for V* by value iteration (just for the dot).
    let V = [0, 0];
    for (let k = 0; k < 2000; k++) {
      const Vn = [
        Math.max(r0L + gamma * V[SUCCESSOR["0L"]], r0R + gamma * V[SUCCESSOR["0R"]]),
        Math.max(r1L + gamma * V[SUCCESSOR["1L"]], r1R + gamma * V[SUCCESSOR["1R"]]),
      ];
      const d = Math.max(Math.abs(Vn[0] - V[0]), Math.abs(Vn[1] - V[1]));
      V = Vn;
      if (d < 1e-9) break;
    }
    const [Vs0, Vs1] = V;

    // Determine which constraints are tight at V* (within tolerance).
    const tight = constraints.map((c) => {
      const lhs = c.s === 0 ? Vs0 : Vs1;
      const rhs = c.r + gamma * (c.sp === 0 ? Vs0 : Vs1);
      return Math.abs(lhs - rhs) < 1e-6;
    });

    // Plot domain: a window around V*.
    const PAD = 2.0;
    const xMin = Math.min(Vs0 - PAD, -0.5);
    const xMax = Math.max(Vs0 + PAD, 5.5);
    const yMin = Math.min(Vs1 - PAD, -0.5);
    const yMax = Math.max(Vs1 + PAD, 5.5);

    // Sample (V0, V1) on a grid; cell is feasible if all 4 constraints hold.
    const NX = 50;
    const NY = 50;
    const cells = [];
    for (let i = 0; i < NX; i++) {
      for (let j = 0; j < NY; j++) {
        const x = xMin + ((i + 0.5) / NX) * (xMax - xMin);
        const y = yMin + ((j + 0.5) / NY) * (yMax - yMin);
        let feasible = true;
        for (const c of constraints) {
          const lhs = c.s === 0 ? x : y;
          const rhs = c.r + gamma * (c.sp === 0 ? x : y);
          if (lhs < rhs - 1e-9) { feasible = false; break; }
        }
        cells.push({ x, y, f: feasible ? 1 : 0 });
      }
    }

    // Constraint lines: V(s) = R + γ V(sp).
    // - If s = 0, sp = 0:  V0 = R + γ V0  ⇒  V0 = R / (1−γ).  Vertical line.
    // - If s = 0, sp = 1:  V0 = R + γ V1  ⇒  V0 = R + γ V1.  Line in (V0, V1).
    // - If s = 1, sp = 0:  V1 = R + γ V0.
    // - If s = 1, sp = 1:  V1 = R + γ V1 ⇒ V1 = R / (1−γ).  Horizontal.
    const lineMarks = constraints.map((c, i) => {
      const label = `(s=${c.s},a=${c.a})`;
      const stroke = (showDual === "on" && tight[i]) ? palette.primary : palette.muted;
      const strokeWidth = (showDual === "on" && tight[i]) ? 2.5 : 1;
      if (c.s === 0 && c.sp === 0) {
        const x = c.r / (1 - gamma);
        return { kind: "vertical", x, label, stroke, strokeWidth };
      } else if (c.s === 1 && c.sp === 1) {
        const y = c.r / (1 - gamma);
        return { kind: "horizontal", y, label, stroke, strokeWidth };
      } else if (c.s === 0 && c.sp === 1) {
        // V0 = R + γ V1, i.e. V0 - γ V1 = R. Plot as a polyline.
        const pts = [
          { x: c.r + gamma * yMin, y: yMin, label },
          { x: c.r + gamma * yMax, y: yMax, label },
        ];
        return { kind: "line", pts, stroke, strokeWidth, label };
      } else {
        // s=1, sp=0: V1 = R + γ V0.
        const pts = [
          { x: xMin, y: c.r + gamma * xMin, label },
          { x: xMax, y: c.r + gamma * xMax, label },
        ];
        return { kind: "line", pts, stroke, strokeWidth, label };
      }
    });

    const marks = [
      // Feasible region as a raster.
      Plot.raster(cells, {
        x: "x", y: "y", fill: "f",
        interpolate: "nearest",
      }),
    ];
    for (const m of lineMarks) {
      if (m.kind === "vertical") {
        marks.push(Plot.ruleX([m.x], { stroke: m.stroke, strokeWidth: m.strokeWidth }));
      } else if (m.kind === "horizontal") {
        marks.push(Plot.ruleY([m.y], { stroke: m.stroke, strokeWidth: m.strokeWidth }));
      } else {
        marks.push(Plot.line(m.pts, {
          x: "x", y: "y", stroke: m.stroke, strokeWidth: m.strokeWidth,
        }));
      }
    }
    // V* dot.
    marks.push(Plot.dot([{ x: Vs0, y: Vs1 }], {
      x: "x", y: "y", r: 6, fill: palette.warning, stroke: "black",
    }));
    marks.push(Plot.text([{ x: Vs0, y: Vs1, label: "V*" }], {
      x: "x", y: "y", text: "label", dx: 10, dy: -6,
      fill: palette.warning, fontSize: 12, fontWeight: "bold",
    }));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 460,
      marginLeft: 44, marginBottom: 36,
      aspectRatio: 1,
      x: { domain: [xMin, xMax], label: "V(0)", grid: true },
      y: { domain: [yMin, yMax], label: "V(1)", grid: true },
      color: {
        type: "linear", domain: [0, 1],
        range: ["#1b1b2e", "#1b3a52"],
        legend: false,
      },
      marks,
    }));

    const tightStr = constraints
      .map((c, i) => (tight[i] ? `(${c.s},${c.a})` : null))
      .filter(Boolean)
      .join(", ");
    slots.readout.textContent =
      `V* = (${fmt(Vs0)}, ${fmt(Vs1)})  ·  sum V* = ${fmt(Vs0 + Vs1)}  ·  ` +
      (showDual === "on"
        ? `tight constraints (dual support): ${tightStr || "none"}`
        : `toggle "show dual" to mark the binding constraints`);
  },
});

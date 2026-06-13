// Widget 7.D — Accumulating vs replacing vs Dutch traces (Chapter 7).
//
// Three trace-update rules raced on the same revisit pattern for a
// single state. Each tick is either a "visit" (the tracked state is the
// current state) or a "no-visit" (the trace just decays).
//
//   - Accumulating: e ← γλ·e ; if visit, e ← e + 1.
//                   Spikes stack; can climb well above 1.
//   - Replacing:    e ← γλ·e ; if visit, e ← 1.
//                   Saturates at 1 each visit.
//   - Dutch:        For a single tabular state with one-hot feature φ_t = 1
//                   on visit, 0 otherwise, the linear-FA Dutch update
//                   reduces (in scalar form) to:
//                       e ← γλ·e + (1 − α·γλ·e)·φ
//                   At φ = 1 this is e ← γλ·e + 1 − α·γλ·e =
//                   (1 − α)·γλ·e + 1 — Dutch and replacing agree at
//                   α = 1 (both → 1), and Dutch lies *between* replacing
//                   and accumulating as α shrinks toward 0.
//
// Slider for α (step size) ∈ [0, 1] shows the divergence: at α near 0
// Dutch ≈ accumulating; at α near 1 Dutch ≈ replacing. Visit-pattern
// dropdown picks "never", "twice in a row", "every 3 steps",
// "every step" so readers see the qualitative behaviour change.
//
// Pattern: chapter markdown has
//
//     <div id="ch7-trace-variants-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/trace_variants/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const T_MAX = 40;            // ticks to simulate
const GAMMA = 0.9;           // fixed discount for the demo
const LAMBDA = 0.9;          // fixed λ for the demo
const GL = GAMMA * LAMBDA;   // trace decay factor 0.81

// Build a boolean visit array of length T_MAX for the chosen pattern.
// Pattern keys must match the <select> option values below.
function buildVisits(pattern) {
  const v = new Array(T_MAX).fill(false);
  switch (pattern) {
    case "single":
      v[5] = true;
      break;
    case "twice":
      v[5] = true;
      v[6] = true;
      break;
    case "every3":
      for (let t = 3; t < T_MAX; t += 3) v[t] = true;
      break;
    case "everystep":
      for (let t = 3; t < T_MAX; t += 1) v[t] = true;
      break;
    case "burst":
      // Three-visit burst then silence — shows accumulating overshoot.
      v[4] = true;
      v[5] = true;
      v[6] = true;
      break;
  }
  return v;
}

// Run all three trace updates over the visit pattern. Returns long-form
// rows {t, kind, e} for Plot.
function runTraces(visits, alpha) {
  let eAcc = 0;
  let eRep = 0;
  let eDut = 0;
  const rows = [];
  // t = 0 frame: all zero (pre-first-step).
  rows.push({ t: 0, kind: "accumulating", e: 0 });
  rows.push({ t: 0, kind: "replacing", e: 0 });
  rows.push({ t: 0, kind: "Dutch", e: 0 });
  for (let t = 1; t <= T_MAX; t++) {
    // Decay all three.
    eAcc *= GL;
    eRep *= GL;
    eDut *= GL;
    const visit = visits[t - 1];
    if (visit) {
      // Classical updates.
      eAcc += 1;
      eRep = 1;
      // Dutch in scalar one-hot form: e ← γλ·e_prev + (1 − α·γλ·e_prev).
      // eDut already holds γλ·e_{prev}; the bump is (1 − α·γλ·e_prev),
      // i.e. (1 − α·eDut_current).
      eDut = eDut + (1 - alpha * eDut);
    }
    rows.push({ t, kind: "accumulating", e: eAcc });
    rows.push({ t, kind: "replacing", e: eRep });
    rows.push({ t, kind: "Dutch", e: eDut });
  }
  return rows;
}

defineWidget({
  hostId: "ch7-trace-variants-widget",
  controls: {
    alpha: {
      label: "α (step size)",
      min: 0.05,
      max: 1.0,
      step: 0.05,
      default: 0.5,
    },
    pattern: {
      type: "select",
      label: "visit pattern",
      options: [
        { value: "single", label: "single visit (t=5)" },
        { value: "twice", label: "twice in a row (t=5,6)" },
        { value: "burst", label: "three-in-a-row burst (t=4,5,6)" },
        { value: "every3", label: "every 3 steps" },
        { value: "everystep", label: "every step" },
      ],
      default: "burst",
    },
  },
  slots: ["main", "visits"],
  render: (host, { alpha, pattern }, slots) => {
    const visits = buildVisits(pattern);
    const rows = runTraces(visits, alpha);

    // Visit markers row (drawn as a tick chart below the trace plot).
    const visitRows = visits
      .map((v, i) => ({ t: i + 1, hit: v ? 1 : 0 }))
      .filter((r) => r.hit > 0);

    // Final trace values for the readout.
    const finalAcc = rows.filter((r) => r.kind === "accumulating").at(-1).e;
    const finalRep = rows.filter((r) => r.kind === "replacing").at(-1).e;
    const finalDut = rows.filter((r) => r.kind === "Dutch").at(-1).e;

    const colorMap = {
      "accumulating": palette.danger,
      "replacing": palette.primary,
      "Dutch": palette.secondary,
    };

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 260,
        x: { label: "tick t", domain: [0, T_MAX], grid: true },
        y: { label: "trace e_t", grid: true, zero: true },
        color: {
          legend: true,
          domain: ["accumulating", "replacing", "Dutch"],
          range: [colorMap.accumulating, colorMap.replacing, colorMap.Dutch],
        },
        marks: [
          Plot.ruleY([1], {
            stroke: palette.muted,
            strokeOpacity: 0.6,
            ...dashed,
          }),
          Plot.line(rows, {
            x: "t",
            y: "e",
            stroke: "kind",
            strokeWidth: 2,
          }),
          Plot.dot(rows, {
            x: "t",
            y: "e",
            fill: "kind",
            r: 1.6,
          }),
          Plot.text(
            [{ x: T_MAX, y: 1, label: "e = 1 (replacing cap)" }],
            {
              x: "x",
              y: "y",
              text: "label",
              textAnchor: "end",
              dy: -4,
              fill: palette.muted,
              ...annotation,
            },
          ),
        ],
      }),
    );

    // Visit-pattern tick chart.
    slots.visits.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 60,
        marginBottom: 30,
        x: { label: "visits (ticks where the tracked state was current)", domain: [0, T_MAX], grid: true },
        y: { axis: null, domain: [0, 1.4] },
        marks: [
          Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
          Plot.tickX(visitRows, {
            x: "t",
            stroke: palette.warning,
            strokeWidth: 2,
          }),
        ],
      }),
    );

    slots.readout.textContent =
      `γλ = ${GL.toFixed(2)}  ·  α = ${alpha.toFixed(2)}  ·  ` +
      `final e: acc = ${finalAcc.toFixed(3)}, ` +
      `rep = ${finalRep.toFixed(3)}, ` +
      `Dutch = ${finalDut.toFixed(3)}`;
  },
});

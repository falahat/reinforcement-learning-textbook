// Widget 18.G — Strike-force project tie-in (Chapter 18 §18.9).
//
// Simulator-flavoured toy: the agent's only action is
// `Strike{force ∈ [0, 1]}`, and the optimal force depends on the
// threat valence v ∈ [-1, +1]. The "true" force–response surface is:
//
//     force*(v) = clip(0.5 + 0.4 · v, 0.05, 0.95)
//     Q*(force | v) = 1 − (force − force*(v))²
//
// i.e. positive valence (predator nearby — flee, no strike) shifts
// the optimum toward high force; negative valence (mild irritant)
// toward low force. The optimal *response* therefore varies
// continuously with v.
//
// The widget compares three architectures on the same Q*:
//   - baseline 2-bucket Strike: Light = 0.25, Hard = 0.75
//   - MP-DQN-style continuous force (the §18.9 pivot)
//   - H-PPO with a Gaussian force head (alternative pivot)
//
// For each, the widget plots the learned `force(threat)` mapping
// over v ∈ [-1, +1] and the per-v regret = Q*(force*) − Q*(force_chosen).
// The reader picks the threat valence v with a slider; markers light
// up at that v on both panels. A "learning progress" slider t ∈
// [0, 1] interpolates each method's mapping from "naive" (constant
// force = 0.5) to "converged" (its asymptotic policy), to give the
// effect of watching learning happen.
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-strike-force-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/strike_force/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

function clip(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// True optimal force as a function of valence v.
function forceStar(v) { return clip(0.5 + 0.4 * v, 0.05, 0.95); }

// True Q under (force, v).
function qStar(force, v) {
  const d = force - forceStar(v);
  return 1 - d * d;
}

// Each method's *asymptotic* learned force(v):
//   bucket2: argmax over {0.25, 0.75} of Q*(force, v) — a step
//     function of v with the threshold at v* where the two buckets
//     tie. The two buckets tie when |0.25 − f*(v)| = |0.75 − f*(v)|,
//     i.e. f*(v) = 0.5, i.e. v = 0.
//   continuous: tracks forceStar(v) exactly (MP-DQN converges).
//   hppo: tracks forceStar(v) but with a small residual variance —
//     for the deterministic mean we just return forceStar(v); the
//     widget annotates the σ band separately.
function policyAsymptotic(method, v) {
  if (method === "bucket2") {
    // 2 buckets. argmax over {0.25, 0.75}.
    const qL = qStar(0.25, v);
    const qH = qStar(0.75, v);
    return qL >= qH ? 0.25 : 0.75;
  }
  return forceStar(v); // continuous & hppo converge to the true optimum
}

// Linear blend from a "naive" force = 0.5 to the asymptotic policy.
function policyAt(method, v, t) {
  const naive = 0.5;
  return naive * (1 - t) + policyAsymptotic(method, v) * t;
}

const METHODS = [
  { id: "bucket2", label: "baseline (2 buckets)", color: palette.danger },
  { id: "continuous", label: "MP-DQN (continuous)", color: palette.primary },
  { id: "hppo", label: "H-PPO (Gaussian)", color: palette.secondary },
];

defineWidget({
  hostId: "ch18-strike-force-widget",
  slots: ["forceMap", "regretMap"],
  controls: {
    valence: { label: "threat valence v", min: -1, max: 1, step: 0.02, default: 0.3 },
    progress: { label: "learning progress t", min: 0, max: 1, step: 0.01, default: 1.0 },
  },
  render: (host, { valence, progress }, slots) => {
    const N = 201;
    const vs = d3.range(N).map((i) => -1 + 2 * i / (N - 1));

    // Build long-form rows for both panels.
    const forceRows = [];
    const regretRows = [];
    for (const v of vs) {
      const fs = forceStar(v);
      forceRows.push({ v, force: fs, method: "optimal force*(v)" });
      for (const m of METHODS) {
        const f = policyAt(m.id, v, progress);
        forceRows.push({ v, force: f, method: m.label });
        regretRows.push({ v, regret: qStar(fs, v) - qStar(f, v), method: m.label });
      }
    }

    const colorScale = {
      domain: ["optimal force*(v)", ...METHODS.map((m) => m.label)],
      range: [palette.muted, ...METHODS.map((m) => m.color)],
    };

    // Panel 1: learned force(v) per method, with optimal as dashed grey.
    slots.forceMap.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "threat valence v", domain: [-1, 1], grid: true },
      y: { label: "force(v)", domain: [0, 1], grid: true },
      color: { legend: true, ...colorScale },
      marks: [
        Plot.ruleX([valence], { stroke: palette.warning, ...dashed }),
        Plot.line(
          forceRows.filter((r) => r.method === "optimal force*(v)"),
          { x: "v", y: "force", stroke: palette.muted, strokeWidth: 1.5, ...dashed },
        ),
        Plot.line(
          forceRows.filter((r) => r.method !== "optimal force*(v)"),
          { x: "v", y: "force", z: "method", stroke: "method", strokeWidth: 2 },
        ),
        Plot.dot(
          METHODS.map((m) => ({
            v: valence, force: policyAt(m.id, valence, progress), method: m.label,
          })),
          { x: "v", y: "force", fill: "method", stroke: "white", strokeWidth: 1, r: 4 },
        ),
      ],
    }));

    // Panel 2: per-v regret per method.
    slots.regretMap.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "threat valence v", domain: [-1, 1], grid: true },
      y: { label: "regret = Q*(f*) − Q*(f_chosen)", domain: [0, 0.3], grid: true },
      color: colorScale,
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.ruleX([valence], { stroke: palette.warning, ...dashed }),
        Plot.line(regretRows, {
          x: "v", y: "regret", z: "method", stroke: "method", strokeWidth: 2,
        }),
      ],
    }));

    const summary = METHODS.map((m) => {
      const f = policyAt(m.id, valence, progress);
      const reg = qStar(forceStar(valence), valence) - qStar(f, valence);
      return `${m.label}: force=${f.toFixed(2)}, regret=${reg.toFixed(3)}`;
    }).join("   ·   ");

    slots.readout.innerHTML =
      `v = ${valence.toFixed(2)}   ·   force*(v) = ${forceStar(valence).toFixed(2)}   ·   ` +
      `t = ${progress.toFixed(2)}<br><small>${summary}</small>`;
  },
});

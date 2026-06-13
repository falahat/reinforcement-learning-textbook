// Widget 1.G — Tower-property bouncer (Chapter 1).
//
// A toy tree: one root state s, three actions a₁/a₂/a₃ with policy
// weights π(a|s), two successor states per action with transition
// probs P(s'|s,a), and leaf rewards r(s'). The widget computes E[R]
// two ways and shows they always match:
//
//   Flat:    Σ_leaves π(a|s) · P(s'|s,a) · r(s')
//   Tower:   Σ_a π(a|s) · Q(a)        where Q(a) = Σ_{s'} P(s'|s,a) · r(s')
//
// Sliders for π (3), P (2 per action × 3 = 6), and r (6) — auto-
// normalised within each probability group. Plot is a horizontal
// stacked bar of per-action contributions π(a) · Q(a) so the student
// sees Σ_a π(a) · Q(a) literally accumulate to the total.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const ACTION_COLORS = [palette.primary, palette.secondary, palette.accent];
const ACTION_LABELS = ["a₁", "a₂", "a₃"];

const controls = {};
// Policy weights π(a|s).
for (let i = 0; i < 3; i++) {
  controls[`pi${i}`] = {
    label: `π(${ACTION_LABELS[i]}|s)`,
    min: 0.05, max: 1, step: 0.01,
    default: [0.5, 0.3, 0.2][i],
  };
}
// Transition probs P(s'|s,a) — first child of each action; second = 1 − first.
for (let i = 0; i < 3; i++) {
  controls[`p${i}`] = {
    label: `P(s'₁|s,${ACTION_LABELS[i]})`,
    min: 0.05, max: 0.95, step: 0.01,
    default: [0.6, 0.4, 0.5][i],
  };
}
// Leaf rewards r(s').
const REWARD_DEFAULTS = [2, -1, 3, 0, -2, 4];
for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 2; j++) {
    controls[`r${i}${j}`] = {
      label: `r(${ACTION_LABELS[i]}, s'${j === 0 ? "₁" : "₂"})`,
      min: -5, max: 5, step: 0.1,
      default: REWARD_DEFAULTS[i * 2 + j],
    };
  }
}

const fmt = (n) => (Number.isFinite(n) ? n.toFixed(3) : "—");

defineWidget({
  hostId: "ch1-tower-widget",
  controls,
  render: (host, params, slots) => {
    // Normalise the three π values to sum to 1.
    const piRaw = [params.pi0, params.pi1, params.pi2];
    const piSum = piRaw.reduce((s, x) => s + x, 0) || 1;
    const pi = piRaw.map((x) => x / piSum);

    // Each action's two transition probs sum to 1.
    const P = [0, 1, 2].map((i) => {
      const p0 = params[`p${i}`];
      return [p0, 1 - p0];
    });

    const r = [0, 1, 2].map((i) => [params[`r${i}0`], params[`r${i}1`]]);

    // Q(a) = Σ_{s'} P(s'|s,a) · r(s')
    const Q = [0, 1, 2].map((i) => P[i][0] * r[i][0] + P[i][1] * r[i][1]);

    // Tower: Σ_a π(a) · Q(a)
    const towerContribs = [0, 1, 2].map((i) => pi[i] * Q[i]);
    const tower = towerContribs.reduce((s, x) => s + x, 0);

    // Flat: Σ_leaves π(a) · P(s'|s,a) · r(s')
    let flat = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) flat += pi[i] * P[i][j] * r[i][j];
    }

    // Plot: horizontal stacked bar of per-action contributions, color-coded.
    const bars = [0, 1, 2].map((i) => ({
      action: ACTION_LABELS[i],
      contrib: towerContribs[i],
    }));
    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 120,
      marginLeft: 80,
      x: { label: "contribution π(a) · Q(a)", grid: true, zero: true },
      y: { label: null },
      color: { domain: ACTION_LABELS, range: ACTION_COLORS, legend: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        Plot.barX(bars, { y: "action", x: "contrib", fill: "action" }),
        Plot.text(bars, {
          y: "action", x: "contrib",
          text: (d) => fmt(d.contrib),
          textAnchor: "start", dx: 4, fontSize: 10, fill: palette.muted,
        }),
        Plot.ruleX([tower], { stroke: palette.danger, strokeDasharray: "4 2" }),
      ],
    }));

    const qStr = Q.map((q, i) => `Q(${ACTION_LABELS[i]}) = ${fmt(q)}`).join(", ");
    const piStr = pi.map((p, i) => `π(${ACTION_LABELS[i]}) = ${fmt(p)}`).join(", ");
    slots.readout.innerHTML =
      `<strong>Flat E[R]</strong> = Σ π·P·r = <code>${fmt(flat)}</code> &nbsp; ` +
      `<strong>Tower E[R]</strong> = Σ π(a)·Q(a) = <code>${fmt(tower)}</code><br>` +
      `<small>${piStr} &nbsp;|&nbsp; ${qStr} &nbsp;|&nbsp; ` +
      `|flat − tower| = ${Math.abs(flat - tower).toExponential(2)}</small>`;
  },
});

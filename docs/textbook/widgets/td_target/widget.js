// Widget 6.H — TD-target dissector (Chapter 6).
//
// One state s_t with one observed transition (s_t, a_t, r, s_{t+1}) and
// three actions available at s'. The reader picks a TD-target variant
// from a dropdown — SARSA, Q-learning, Expected SARSA, TD(0) — and
// watches all four targets recompute side-by-side as the sliders move.
//
// The point is *dissection*: each variant looks at a different subset
// of Q(s', ·). SARSA reads one entry (the sampled a'); Q-learning the
// max entry; Expected SARSA all entries under a π-weighted average;
// TD(0) folds them into V(s') = E_π[Q(s', ·)] (same expectation —
// shown for completeness so the reader sees TD(0) = Expected SARSA on
// V). Tags next to each Q slider show which variants currently look
// at that entry, so the geometry of "which Q does this target depend
// on?" becomes pointable.
//
// Pattern: chapter markdown has just
//
//     <div id="ch6-td-target-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/td_target/widget.js"></script>
//
// The shared `defineWidget` scaffold handles control-HTML, event
// wiring, and DOMContentLoaded mounting.

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

// Which action a_t+1 is sampled for SARSA. We fix this to a₂ (the
// middle action) so the reader can compare its Q value against the
// max and the expectation without juggling another control. The
// chapter prose calls out that SARSA's a' is sampled from π — we
// pick a fixed sample so the widget stays deterministic.
const SARSA_ACTION_INDEX = 1;
const ACTION_LABELS = ["a₁", "a₂", "a₃"];

const VARIANTS = [
  { value: "sarsa",    label: "SARSA: r + γ Q(s', a')" },
  { value: "qlearn",   label: "Q-learning: r + γ max Q(s', ·)" },
  { value: "expected", label: "Expected SARSA: r + γ E_π[Q(s', ·)]" },
  { value: "td0",      label: "TD(0): r + γ V(s')" },
];

defineWidget({
  hostId: "ch6-td-target-widget",
  controls: {
    variant: {
      type: "select",
      label: "Target variant",
      options: VARIANTS,
      default: "qlearn",
    },
    r:       { label: "r (reward)",  min: -2,   max: 2,    step: 0.05, default: 1 },
    gamma:   { label: "γ (discount)", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
    q1:      { label: "Q(s', a₁)",    min: -3,  max: 3,    step: 0.05, default: 0.5 },
    q2:      { label: "Q(s', a₂)",    min: -3,  max: 3,    step: 0.05, default: 1.0 },
    q3:      { label: "Q(s', a₃)",    min: -3,  max: 3,    step: 0.05, default: 1.5 },
    epsilon: { label: "ε (for π)",    min: 0,   max: 1,    step: 0.01, default: 0.1 },
  },
  slots: ["main", "uses"],
  render: (host, p, slots) => {
    const qs = [p.q1, p.q2, p.q3];
    const A = qs.length;

    // ε-greedy policy over s'. Greedy action is argmax(Q). Ties broken
    // by lowest index (stable). π(greedy) = 1 − ε + ε/A; π(other) = ε/A.
    const greedyIdx = qs.indexOf(Math.max(...qs));
    const pi = qs.map((_, i) => (i === greedyIdx ? 1 - p.epsilon + p.epsilon / A : p.epsilon / A));

    // The four TD targets. Expected SARSA collapses to Q-learning at
    // ε=0 (greedy π) and to a uniform average at ε=1.
    const qSarsa    = qs[SARSA_ACTION_INDEX];
    const qMax      = Math.max(...qs);
    const qExpected = qs.reduce((s, q, i) => s + pi[i] * q, 0);
    const vS        = qExpected; // V(s') = E_π[Q(s', ·)] under the same π

    const targets = {
      sarsa:    p.r + p.gamma * qSarsa,
      qlearn:   p.r + p.gamma * qMax,
      expected: p.r + p.gamma * qExpected,
      td0:      p.r + p.gamma * vS,
    };

    // --- Bar chart: four targets side by side ---
    const rows = VARIANTS.map((v) => ({
      variant: v.value,
      label: shortLabel(v.value),
      value: targets[v.value],
      selected: v.value === p.variant,
    }));
    const yAbsMax = Math.max(1, ...rows.map((r) => Math.abs(r.value))) * 1.2;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      marginLeft: 52,
      marginBottom: 44,
      x: { label: null, domain: rows.map((r) => r.label) },
      y: { label: "TD target value", domain: [-yAbsMax, yAbsMax], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.6 }),
        Plot.barY(rows, {
          x: "label",
          y: "value",
          fill: (d) => (d.selected ? palette.primary : palette.secondary),
          fillOpacity: (d) => (d.selected ? 0.95 : 0.55),
          stroke: (d) => (d.selected ? palette.primary : "transparent"),
          strokeWidth: 1.5,
        }),
        Plot.text(rows, {
          x: "label",
          y: "value",
          text: (d) => fmt(d.value),
          dy: (d) => (d.value >= 0 ? -8 : 14),
          fontSize: 11,
          fontWeight: (d) => (d.selected ? "bold" : "normal"),
          fill: "#ddd",
        }),
      ],
    }));

    // --- "Which Q does each variant use?" tag panel ---
    // For each action a_i, build a row showing which variants currently
    // read Q(s', a_i). SARSA reads only its sampled action; Q-learning
    // reads only the argmax; Expected SARSA / TD(0) read every entry
    // with weight π(a_i | s').
    const usesRows = qs.map((q, i) => {
      const tags = [];
      if (i === SARSA_ACTION_INDEX) tags.push(badge("SARSA", palette.warning));
      if (i === greedyIdx)          tags.push(badge("Q-learn (max)", palette.danger));
      tags.push(badge(`Exp. SARSA · π=${pi[i].toFixed(2)}`, palette.accent));
      tags.push(badge(`TD(0) · π=${pi[i].toFixed(2)}`, palette.secondary));
      return `
        <div class="td-target-row">
          <span class="td-target-q">${ACTION_LABELS[i]} = ${fmt(q)}</span>
          <span class="td-target-tags">${tags.join(" ")}</span>
        </div>`;
    }).join("");

    slots.uses.innerHTML = `
      <style>
        #${host.id} .td-target-row {
          display: flex; align-items: center; gap: 0.75em;
          margin: 0.25em 0; font-size: 0.85em; flex-wrap: wrap;
        }
        #${host.id} .td-target-q {
          min-width: 7em; font-family: ui-monospace, monospace;
          color: #ddd;
        }
        #${host.id} .td-target-tags { display: flex; gap: 0.4em; flex-wrap: wrap; }
        #${host.id} .td-target-badge {
          padding: 1px 6px; border-radius: 3px; font-size: 0.85em;
          color: #fff; opacity: 0.85; white-space: nowrap;
        }
      </style>
      ${usesRows}
    `;

    // --- Readout: the selected target's value ---
    const selVariant = VARIANTS.find((v) => v.value === p.variant);
    slots.readout.textContent = `→ ${selVariant.label.split(":")[0]} = ${fmt(targets[p.variant])}`;
  },
});

function shortLabel(v) {
  return v === "sarsa" ? "SARSA"
       : v === "qlearn" ? "Q-learning"
       : v === "expected" ? "Exp. SARSA"
       : "TD(0)";
}

function badge(text, color) {
  return `<span class="td-target-badge" style="background:${color}">${text}</span>`;
}

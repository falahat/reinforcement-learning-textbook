// Widget 15.D — Score-formula contribution decomposer (Chapter 15).
//
// The Simulator's policy picks actions by
//
//     score(a) = 0.5·Q(a) + recipe_bonus(a)
//
// (`crates/cognition/planner/src/policy.rs`). Three candidate actions
// — one "committed" with a successful history, one "novel" that has
// never been picked, one "neutral" with a small recipe_bonus edge.
// recipe_bonus is fixed per action; each action's Q-value evolves
// under TD bootstrap at rates that reflect its sampling frequency:
//
//   - committed: 90% of the time (greedy argmax winner) → fast Q rise
//   - neutral:   ~ε/2 of the time → slow Q rise
//   - novel:     ~ε/2 of the time → slow Q rise
//
// The widget is a stacked-bar visualisation per action: the bottom
// segment is `0.5·Q`, the top is `recipe_bonus`. The reader drags
// `w_alive`, `γ`, and `ε`, and scrubs through ticks 0 → tick_max.
// An arrow above the tallest bar marks the argmax winner. The
// dramatic moment is watching the committed action's Q-bar grow from
// 0 → 0.5·w_alive/(1−γ), at which point its score locks the argmax
// forever — even though "neutral" started ahead on recipe_bonus.
//
//     <div id="ch15-score-decomposer-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/score_formula_decomposer/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

// Per-action recipe_bonus stays constant; Q grows under TD.
const ACTIONS = [
  { name: "committed", recipeBonus: 0.30, sampleProb: (eps) => 1.0 - eps + eps / 3 },
  { name: "neutral",   recipeBonus: 0.65, sampleProb: (eps) => eps / 3 },
  { name: "novel",     recipeBonus: 0.60, sampleProb: (eps) => eps / 3 },
];

// Simulate the three Q-values under per-action sampling probability.
// At each tick: with prob `sampleProb` the action is committed, its
// reward is r = w_alive, and Q ← Q + α(r + γQ − Q). With prob (1 −
// sampleProb) Q is unchanged (no learning happens for actions that
// aren't picked).
function simulateAll({ ticks, alpha, gamma, wAlive, epsilon }) {
  const Q = ACTIONS.map(() => 0);
  const hist = [{ tick: 0, q: [...Q] }];
  for (let t = 1; t <= ticks; t++) {
    for (let i = 0; i < ACTIONS.length; i++) {
      const p = ACTIONS[i].sampleProb(epsilon);
      // Expected update over many ticks: scale α by sample frequency.
      const effAlpha = alpha * p;
      const target = wAlive + gamma * Q[i];
      Q[i] += effAlpha * (target - Q[i]);
    }
    hist.push({ tick: t, q: [...Q] });
  }
  return hist;
}

defineWidget({
  hostId: "ch15-score-decomposer-widget",
  controls: {
    wAlive:  { label: "w_alive",       min: 0,    max: 2.0,  step: 0.05, default: 1.0 },
    gamma:   { label: "γ",             min: 0.5,  max: 0.99, step: 0.01, default: 0.9 },
    alpha:   { label: "α (step size)", min: 0.01, max: 0.5,  step: 0.01, default: 0.1 },
    epsilon: { label: "ε (explore)",   min: 0.0,  max: 0.5,  step: 0.01, default: 0.1 },
    tick:    { label: "tick t",        min: 0,    max: 2000, step: 10,   default: 0 },
  },
  render: (host, p, slots) => {
    const ticksMax = 2000;
    const hist = simulateAll({
      ticks: ticksMax, alpha: p.alpha, gamma: p.gamma,
      wAlive: p.wAlive, epsilon: p.epsilon,
    });
    const idx = Math.min(p.tick | 0, ticksMax);
    const frame = hist[idx];

    // Build stacked-bar rows. Plot.barY stacks within same x by default.
    // Order the stack from bottom to top: 0.5·Q, recipe_bonus.
    const rows = [];
    const scores = [];
    for (let i = 0; i < ACTIONS.length; i++) {
      const a = ACTIONS[i];
      const qContrib = 0.5 * frame.q[i];
      const total = qContrib + a.recipeBonus;
      rows.push({ action: a.name, term: "0.5·Q",        value: qContrib });
      rows.push({ action: a.name, term: "recipe_bonus", value: a.recipeBonus });
      scores.push({ action: a.name, total, q: frame.q[i] });
    }
    let winnerIdx = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i].total > scores[winnerIdx].total) winnerIdx = i;
    }
    const winner = scores[winnerIdx].action;

    const fixedPoint = p.wAlive / Math.max(1e-6, 1 - p.gamma);
    const yMax = Math.max(0.5 * fixedPoint + 1, 3);

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      marginLeft: 60,
      x: { label: "action", domain: ACTIONS.map((a) => a.name) },
      y: { label: "score (stacked)", domain: [0, yMax], grid: true },
      color: {
        domain: ["0.5·Q", "recipe_bonus"],
        range: [palette.danger, palette.warning],
        legend: true,
      },
      marks: [
        Plot.barY(rows, {
          x: "action", y: "value", fill: "term",
          fillOpacity: 0.85,
        }),
        // Argmax arrow above the winner.
        Plot.text(
          [{ action: winner, label: "▼ argmax" }],
          { x: "action", y: yMax * 0.96, text: "label",
            fill: palette.warning, fontSize: 12, fontWeight: "bold",
            textAnchor: "middle" },
        ),
        // Per-action total at top of each stack.
        Plot.text(scores, {
          x: "action", y: "total", text: (d) => fmt(d.total),
          fill: "#ddd", fontSize: 11, dy: -6, textAnchor: "middle",
        }),
      ],
    }));

    const qStr = scores.map((s) => `Q(${s.action})=${fmt(s.q)}`).join("  ");
    slots.readout.textContent =
      `tick ${idx} · ${qStr} · Q* = w_alive/(1−γ) = ${fmt(fixedPoint)} · argmax: ${winner}`;
  },
});

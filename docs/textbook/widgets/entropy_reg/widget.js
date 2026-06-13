// Widget 10.D — Entropy-regularisation effect on a 3-action softmax policy
// (Chapter 10).
//
// Many policy-gradient methods (A2C, PPO, SAC) add a bonus
//
//   J_reg(θ) = J(θ) + β · H(π_θ)
//
// to keep the policy from collapsing to a one-hot before it has explored.
// The widget makes this concrete on a 3-armed bandit with fixed mean
// rewards r = [1.0, 0.7, 0.3]. Under softmax(θ) the *gradient-flow*
// fixed point of the regularised objective is the closed-form Boltzmann
// policy
//
//   π(a) ∝ exp(r_a / β)
//
// (taking β → 0 recovers a one-hot on argmax r; β → ∞ recovers uniform).
//
// Left plot: the resulting policy as a bar chart over arms. Right plot:
// entropy H(π) and expected reward E_π[r] as functions of β, with a
// vertical rule at the slider's current β. The reader sweeps and *sees*
// the exploration-vs-exploitation Pareto curve traced out.
//
// Pattern:
//
//     <div id="ch10-entropy-reg-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/entropy_reg/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const REWARDS = [1.0, 0.7, 0.3];
const ARM_LABELS = REWARDS.map((_, i) => `a${i + 1}`);

// Closed-form Boltzmann policy π(a) ∝ exp(r_a / β) — log-sum-exp safe.
function boltzmann(beta) {
  const safe = Math.max(beta, 1e-3);
  const logits = REWARDS.map((r) => r / safe);
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

function entropy(p) {
  let h = 0;
  for (const pi of p) if (pi > 0) h -= pi * Math.log(pi);
  return h;
}

function expectedReward(p) {
  let er = 0;
  for (let i = 0; i < p.length; i++) er += p[i] * REWARDS[i];
  return er;
}

defineWidget({
  hostId: "ch10-entropy-reg-widget",
  controls: {
    beta: { label: "β (entropy weight)", min: 0.02, max: 2.0, step: 0.01, default: 0.2 },
  },
  slots: ["policy", "sweep"],
  render: (host, { beta }, slots) => {
    const pi = boltzmann(beta);
    const H = entropy(pi);
    const ER = expectedReward(pi);
    const Hmax = Math.log(REWARDS.length); // uniform entropy

    // ---- Left plot: bar chart of π(a) under current β ----
    const barRows = pi.map((p, i) => ({ a: ARM_LABELS[i], p, r: REWARDS[i] }));
    slots.policy.replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 320, height: 240,
      x: { label: "action (with mean reward)" },
      y: { label: "π(a)", domain: [0, 1.05], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
        Plot.barY(barRows, { x: "a", y: "p", fill: palette.primary, fillOpacity: 0.85 }),
        Plot.text(barRows, {
          x: "a", y: "p", text: (d) => `${d.p.toFixed(2)}\nr=${d.r}`,
          dy: -14, fill: palette.muted, fontSize: 10,
        }),
      ],
    }));

    // ---- Right plot: H(π) and E[r] as β sweeps ----
    const betas = d3.range(80).map((i) => 0.02 + (2.0 - 0.02) * (i / 79));
    const sweepRows = [];
    for (const b of betas) {
      const p = boltzmann(b);
      sweepRows.push({ b, kind: "H(π) / log K", v: entropy(p) / Hmax });
      sweepRows.push({ b, kind: "E[r] (rescaled)", v: expectedReward(p) / REWARDS[0] });
    }

    slots.sweep.replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 360, height: 240,
      x: { label: "β (entropy weight)", grid: true },
      y: { label: "fraction of max", domain: [0, 1.05], grid: true },
      color: {
        domain: ["H(π) / log K", "E[r] (rescaled)"],
        range: [palette.accent, palette.secondary],
        legend: true,
      },
      marks: [
        Plot.line(sweepRows, { x: "b", y: "v", stroke: "kind", strokeWidth: 2 }),
        Plot.ruleX([beta], { stroke: palette.warning, strokeWidth: 1.5, ...dashed }),
        Plot.text(
          [{ x: beta, y: 1.02, label: `β = ${beta.toFixed(2)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "middle", dy: 4,
            fill: palette.warning, ...annotation },
        ),
      ],
    }));

    const greedyER = REWARDS[0];
    const piStr = pi.map((p, i) => `π(a${i + 1})=${p.toFixed(2)}`).join(" · ");
    slots.readout.textContent =
      `${piStr}  ·  H(π) = ${H.toFixed(3)} / log 3 = ${Hmax.toFixed(3)}  ·  ` +
      `E[r] = ${ER.toFixed(3)} (greedy would give ${greedyER.toFixed(2)})`;
  },
});

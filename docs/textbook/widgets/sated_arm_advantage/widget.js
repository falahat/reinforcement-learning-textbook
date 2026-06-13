// Widget 11.G — Advantage learning on the Simulator's "sated arm" (Chapter 11).
//
// Toy reproduction of `learning_homeostatic.rs`'s sated-arm scenario,
// which the Q-bias bootstrap pathology hits head-on. Two
// actions {Plant, Consume}; the agent is "sated" so Consume's intrinsic
// reward is 0 (consumption gives no benefit). Both actions yield the
// alive-baseline w_alive = 1.0 every tick.
//
// Two side-by-side learners on identical data:
//
//   (1) PLAIN Q:    Q(a) ← Q(a) + α[r + γ max_b Q(b) − Q(a)]
//       — both Qs drift toward w_alive/(1−γ) = 10. Whichever action
//         fires first reaches the fixed point first; the other lags.
//         The decision rule argmax Q locks in.
//
//   (2) Q − V advantage:
//       V(s) ← V(s) + α[r + γ V(s) − V(s)]        (state value)
//       Q(a) ← Q(a) + α[r + γ V(s) − Q(a)]        (action value, bootstraps off V)
//       score(a) = Q(a) − V(s)
//       — V absorbs the w_alive baseline; Q−V is the action-relative
//         advantage. Differences stay small and reflect *actual*
//         reward shape, not bootstrap drift.
//
// Toggle w_alive live; plain Q's argmax flips with the alive baseline
// (pathology), advantage learning is invariant.
//
// Pedagogy: §11.7 walks through this in prose ("V absorbs w_alive, Q−V
// retains the action ordering"). The widget makes it kinetic — pull
// the w_alive slider and watch the two learners disagree.
//
// Pattern:
//
//     <div id="ch11-sated-arm-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/sated_arm_advantage/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

// Simulate `steps` of both learners under epsilon-greedy. Returns a
// per-step trace of all four state variables.
function simulate({ wAlive, gamma, alpha, eps, steps, planterBias }) {
  const rand = lcg(7);
  // PLAIN Q: two action-values, no V.
  let qPlant = 0, qConsume = 0;
  // ADV: shared V, two Q-bootstraps-off-V.
  let aV = 0, aQPlant = 0, aQConsume = 0;

  // "true" intrinsic reward: Plant pays a tiny w_alive bonus + bias,
  // Consume pays only w_alive (sated). The whole point is that the
  // *only* relevant difference between actions is the small bias, but
  // plain Q drowns it in the w_alive bootstrap.
  const rewardPlant   = () => wAlive + planterBias;
  const rewardConsume = () => wAlive;

  const trace = [];
  for (let t = 0; t < steps; t++) {
    // Plain-Q action selection.
    const aPlain =
      rand() < eps ? (rand() < 0.5 ? 0 : 1) :
      (qPlant >= qConsume ? 0 : 1);
    const rPlain = aPlain === 0 ? rewardPlant() : rewardConsume();
    // Plain Q update (TD on max Q).
    const maxQ = Math.max(qPlant, qConsume);
    if (aPlain === 0) {
      qPlant += alpha * (rPlain + gamma * maxQ - qPlant);
    } else {
      qConsume += alpha * (rPlain + gamma * maxQ - qConsume);
    }

    // Advantage-learner action selection (argmax Q − V; V cancels for
    // argmax but is shown for completeness).
    const aPlant = aQPlant - aV;
    const aCons  = aQConsume - aV;
    const aAdv =
      rand() < eps ? (rand() < 0.5 ? 0 : 1) :
      (aPlant >= aCons ? 0 : 1);
    const rAdv = aAdv === 0 ? rewardPlant() : rewardConsume();
    // V on-policy: bootstrap from V(next) = V (single state).
    aV += alpha * (rAdv + gamma * aV - aV);
    // Q bootstraps from V, not from max Q — this is the architectural fix.
    if (aAdv === 0) {
      aQPlant += alpha * (rAdv + gamma * aV - aQPlant);
    } else {
      aQConsume += alpha * (rAdv + gamma * aV - aQConsume);
    }

    trace.push({
      t,
      qPlant, qConsume,
      qPlantMargin: qPlant - qConsume,
      aV, aQPlant, aQConsume,
      aPlant: aQPlant - aV,
      aCons:  aQConsume - aV,
      aMargin: (aQPlant - aV) - (aQConsume - aV),
    });
  }
  return trace;
}

defineWidget({
  hostId: "ch11-sated-arm-widget",
  controls: {
    wAlive:       { label: "w_alive (alive baseline)", min: 0.0,  max: 2.0,  step: 0.05, default: 1.0 },
    planterBias:  { label: "Plant bias (true Δ)",      min: -0.3, max: 0.3,  step: 0.01, default: 0.05 },
    gamma:        { label: "γ",                         min: 0.5,  max: 0.99, step: 0.01, default: 0.9 },
    alpha:        { label: "α",                         min: 0.01, max: 0.5,  step: 0.01, default: 0.1 },
    eps:          { label: "ε (explore)",               min: 0.0,  max: 0.5,  step: 0.01, default: 0.1 },
    steps:        { label: "steps",                     min: 50,   max: 2000, step: 50,   default: 600 },
  },
  slots: ["plainQ", "advantage"],
  render: (host, params, slots) => {
    const trace = simulate(params);
    const { wAlive, planterBias, gamma } = params;
    const fixedPoint = wAlive / (1 - gamma);

    // Plain-Q panel.
    const plainRows = [];
    for (const r of trace) {
      plainRows.push({ t: r.t, series: "Q(Plant)",   v: r.qPlant });
      plainRows.push({ t: r.t, series: "Q(Consume)", v: r.qConsume });
    }
    const plainMax = Math.max(fixedPoint * 1.1, ...plainRows.map((r) => r.v));

    slots.plainQ.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "step", grid: true },
      y: {
        label: "plain Q(a)",
        grid: true,
        domain: [Math.min(0, plainRows.map((r) => r.v).reduce((a, b) => Math.min(a, b), 0)), plainMax],
      },
      color: {
        domain: ["Q(Plant)", "Q(Consume)"],
        range: [palette.primary, palette.danger],
        legend: true,
      },
      marks: [
        Plot.ruleY([fixedPoint], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ t: trace.length - 1, v: fixedPoint, label: `w_alive/(1−γ) = ${fixedPoint.toFixed(2)}` }],
          { x: "t", y: "v", text: "label", textAnchor: "end", dy: -6,
            fill: palette.warning, ...annotation },
        ),
        Plot.line(plainRows, { x: "t", y: "v", stroke: "series", strokeWidth: 1.6 }),
      ],
    }));

    // Advantage-learner panel.
    const advRows = [];
    for (const r of trace) {
      advRows.push({ t: r.t, series: "V(s)",        v: r.aV });
      advRows.push({ t: r.t, series: "A(Plant)",    v: r.aPlant });
      advRows.push({ t: r.t, series: "A(Consume)",  v: r.aCons });
    }
    const advMin = Math.min(0, ...advRows.map((r) => r.v));
    const advMax = Math.max(fixedPoint * 1.1, ...advRows.map((r) => r.v));

    slots.advantage.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "step", grid: true },
      y: { label: "value", grid: true, domain: [advMin, advMax] },
      color: {
        domain: ["V(s)", "A(Plant)", "A(Consume)"],
        range: [palette.secondary, palette.primary, palette.danger],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.ruleY([fixedPoint], { stroke: palette.warning, ...dashed }),
        Plot.line(advRows, { x: "t", y: "v", stroke: "series", strokeWidth: 1.6 }),
      ],
    }));

    const last = trace[trace.length - 1];
    const plainArgmax = last.qPlant >= last.qConsume ? "Plant" : "Consume";
    const advArgmax   = last.aPlant >= last.aCons   ? "Plant" : "Consume";
    const truthful = planterBias > 0 ? "Plant" : planterBias < 0 ? "Consume" : "tie";

    slots.readout.textContent =
      `final: plain argmax = ${plainArgmax} (Q margin ${last.qPlantMargin.toFixed(3)}) | ` +
      `adv argmax = ${advArgmax} (A margin ${last.aMargin.toFixed(3)}) | ` +
      `true best = ${truthful}`;
  },
});

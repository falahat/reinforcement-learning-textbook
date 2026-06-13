// Widget 15.B — Q-bias bootstrap visualizer (Chapter 15).
//
// The Simulator's central bug. In a steady state with positive
// `w_alive`, every committed action's Q-value drifts toward the
// geometric-sum fixed point  Q* = r/(1−γ)  where r ≈ w_alive. With
// the project's defaults (w_alive = 1, γ = 0.9) that's Q* = 10. The
// untried actions stay near their prior (0), so the *gap* between
// committed and untried Q grows unboundedly.
//
// The widget runs an explicit TD bootstrap loop for one "committed"
// action — Q_{t+1} = Q_t + α(r + γ·Q_t − Q_t) — and plots the
// trajectory against the analytical fixed point. A second slot
// shows the *score-formula* implication: the policy uses score =
// 0.5·Q + recipe_bonus, so the committed action's score climbs past
// every reasonable recipe_bonus value (∈ [0, 1]) within a few dozen
// ticks. After that point ε-greedy is the only way to ever try the
// other action — and at ε = 0.1 with K candidates, untried actions
// get tried ~ε/K of the time, which is far too slow to catch up
// before the committed Q saturates argmax forever.
//
//     <div id="ch15-qbias-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/q_bias_bootstrap/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

// TD-on-committed-action: bootstrap Q_t toward r + γQ_t.
function simulateBootstrap({ steps, alpha, gamma, wAlive, qInit }) {
  const history = new Array(steps + 1);
  let q = qInit;
  const r = wAlive;
  history[0] = { t: 0, q };
  for (let t = 1; t <= steps; t++) {
    // V(s') ≈ Q(s, a) in steady state with one committed action.
    const target = r + gamma * q;
    q += alpha * (target - q);
    history[t] = { t, q };
  }
  return history;
}

defineWidget({
  hostId: "ch15-qbias-widget",
  controls: {
    wAlive:      { label: "w_alive",       min: 0,    max: 2.0,  step: 0.05, default: 1.0 },
    gamma:       { label: "γ",             min: 0.5,  max: 0.99, step: 0.01, default: 0.9 },
    alpha:       { label: "α (step size)", min: 0.01, max: 0.5,  step: 0.01, default: 0.1 },
    steps:       { label: "ticks",         min: 50,   max: 2000, step: 50,   default: 500 },
    recipeBonus: { label: "recipe_bonus(untried)", min: 0, max: 1, step: 0.05, default: 0.5 },
    qInit:       { label: "Q_init (untried prior)", min: -2, max: 5, step: 0.1, default: 0.0 },
  },
  slots: ["main", "score"],
  render: (host, p, slots) => {
    const fixedPoint = p.wAlive / Math.max(1e-6, 1 - p.gamma);
    const hist = simulateBootstrap({
      steps: p.steps | 0,
      alpha: p.alpha,
      gamma: p.gamma,
      wAlive: p.wAlive,
      qInit: p.qInit,
    });
    // Two series: committed Q (the bootstrap trajectory) and untried Q
    // (frozen at qInit because it's never sampled).
    const long = [];
    for (const f of hist) {
      long.push({ t: f.t, action: "Q(committed)",  value: f.q });
      long.push({ t: f.t, action: "Q(untried)",    value: p.qInit });
    }

    const yMax = Math.max(fixedPoint * 1.15, 12, p.qInit + 1);
    const yMin = Math.min(0, p.qInit) - 0.5;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "tick", grid: true },
      y: { label: "Q-value", domain: [yMin, yMax], grid: true },
      color: {
        domain: ["Q(committed)", "Q(untried)"],
        range: [palette.danger, palette.muted],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.ruleY([fixedPoint], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: p.steps, y: fixedPoint, label: `Q* = w_alive/(1−γ) = ${fmt(fixedPoint)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.warning, ...annotation },
        ),
        Plot.line(long, { x: "t", y: "value", stroke: "action", strokeWidth: 2 }),
      ],
    }));

    // Score-formula slot: score = 0.5·Q + recipe_bonus.
    // Plot committed-score and untried-score over the same time axis.
    const scoreRows = [];
    for (const f of hist) {
      scoreRows.push({ t: f.t, action: "score(committed)", value: 0.5 * f.q + 0.0 });
      scoreRows.push({ t: f.t, action: "score(untried)",   value: 0.5 * p.qInit + p.recipeBonus });
    }
    // Where does the committed score overtake the untried score?
    const untriedScore = 0.5 * p.qInit + p.recipeBonus;
    let cross = -1;
    for (let i = 1; i < hist.length; i++) {
      if (0.5 * hist[i].q >= untriedScore && 0.5 * hist[i - 1].q < untriedScore) {
        cross = hist[i].t;
        break;
      }
    }

    const scoreYMax = Math.max(0.5 * fixedPoint * 1.15, untriedScore + 1, 6);
    const scoreYMin = Math.min(0, untriedScore - 0.5);

    slots.score.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "tick", grid: true },
      y: { label: "score (argmax target)", domain: [scoreYMin, scoreYMax], grid: true },
      color: {
        domain: ["score(committed)", "score(untried)"],
        range: [palette.danger, palette.secondary],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.line(scoreRows, { x: "t", y: "value", stroke: "action", strokeWidth: 2 }),
        ...(cross > 0 ? [
          Plot.ruleX([cross], { stroke: palette.warning, ...dashed }),
          Plot.text([{ x: cross, y: scoreYMax * 0.95, label: `lock-in at t=${cross}` }],
            { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4,
              fill: palette.warning, ...annotation }),
        ] : []),
      ],
    }));

    const finalQ = hist[hist.length - 1].q;
    const finalScore = 0.5 * finalQ;
    const gap = finalScore - untriedScore;
    slots.readout.textContent =
      `final Q(committed) = ${fmt(finalQ)} → score = ${fmt(finalScore)}; ` +
      `untried score = ${fmt(untriedScore)}; gap = ${fmt(gap)} ` +
      `(unbridgeable by recipe_bonus ∈ [0, 1])`;
  },
});

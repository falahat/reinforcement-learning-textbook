// Widget 17.G — Hindsight Credit Assignment posterior visualiser
// (Chapter 17, §17.7).
//
// HCA [Harutyunyan et al. 2019] re-weights each historical action's
// TD update by P(a_t | outcome) — the posterior probability that
// action a_t at time t caused the observed terminal outcome. Actions
// strongly predictive of the outcome get full credit; actions
// indistinguishable from the marginal policy get none.
//
// The closed-form weight is
//
//     w_t = P(a_t | outcome) / π(a_t | s_t)
//
// (importance ratio between the hindsight posterior and the behaviour
// policy). When P(a_t | outcome) = π(a_t | s_t) the update is unchanged;
// when P(a_t | outcome) > π(a_t | s_t) the update is upweighted; when
// the action was *anti*-predictive, w_t < 1 and the update is damped.
//
// The widget walks a single trajectory of length T. The reader edits a
// fixed prior π over A actions (slider per action) and a per-step
// "evidence strength" λ_t that controls how concentrated the posterior
// is on the observed action a_t. Then we plot, for each t:
//
//   - the prior bar  π(· | s_t),
//   - the posterior bar  P(· | outcome, s_t),
//   - the resulting credit weight w_t for the observed action.
//
// At λ_t = 0 the posterior equals the prior (HCA does nothing, identical
// to plain PG). As λ_t increases, the posterior concentrates on the
// observed action and the weight grows beyond 1 — that is the
// "hindsight upweight" the theorem promises.
//
// Mount: in §17.7 of Chapter 17.
//
//     <div id="ch17-hca-posterior-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/hca_posterior/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";

const ACTION_LABELS = ["a₁", "a₂", "a₃", "a₄"];
const A = ACTION_LABELS.length;
const T = 6;                  // trajectory length
// Pre-baked "observed" action per step. Deterministic so the trajectory
// is a fixed teaching example; varying λ_t shows the hindsight effect.
const OBSERVED = [0, 2, 1, 3, 0, 2];

// Posterior: P(a | outcome) ∝ π(a) · exp(λ · 1[a = observed]). This is
// a soft Bayesian update with a per-step likelihood that concentrates
// on the observed action.
function posterior(prior, observed, lambda) {
  const unnorm = prior.map((p, a) => p * Math.exp(lambda * (a === observed ? 1 : 0)));
  const Z = unnorm.reduce((s, x) => s + x, 0);
  return unnorm.map((u) => u / Z);
}

defineWidget({
  hostId: "ch17-hca-posterior-widget",
  controls: {
    pi1:    { label: "π(a₁)", min: 0.05, max: 0.7, step: 0.01, default: 0.25 },
    pi2:    { label: "π(a₂)", min: 0.05, max: 0.7, step: 0.01, default: 0.25 },
    pi3:    { label: "π(a₃)", min: 0.05, max: 0.7, step: 0.01, default: 0.25 },
    lambda: { label: "evidence strength λ", min: 0, max: 4, step: 0.05, default: 1.5 },
    step:   { label: "highlight step t", min: 1, max: T, step: 1, default: 3 },
  },
  slots: ["bars", "weights"],
  render: (host, params, slots) => {
    const { pi1, pi2, pi3, lambda, step } = params;
    // Normalise the user prior to a valid distribution; the last action
    // takes the slack so all four slot sliders don't have to coordinate.
    const raw = [pi1, pi2, pi3, Math.max(0.01, 1 - pi1 - pi2 - pi3)];
    const Zp = raw.reduce((s, x) => s + x, 0);
    const prior = raw.map((p) => p / Zp);

    // Per-step posteriors and credit weights for the observed action.
    const post = OBSERVED.map((obs) => posterior(prior, obs, lambda));
    const weights = OBSERVED.map((obs, t) => ({
      t: t + 1,
      observed: ACTION_LABELS[obs],
      w: post[t][obs] / prior[obs],
    }));

    // --- top: prior vs posterior bars for the highlighted step ---
    const tHighlight = step - 1;
    const observedAtT = OBSERVED[tHighlight];
    const barRows = [];
    for (let a = 0; a < A; a++) {
      barRows.push({ action: ACTION_LABELS[a], kind: "π (prior)",       prob: prior[a] });
      barRows.push({ action: ACTION_LABELS[a], kind: "P (posterior)",  prob: post[tHighlight][a] });
    }

    slots.bars.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "action", domain: ACTION_LABELS },
      y: { label: "probability", domain: [0, 1], grid: true },
      color: {
        domain: ["π (prior)", "P (posterior)"],
        range: [palette.muted, palette.primary],
        legend: true,
      },
      fx: { padding: 0.1 },
      marks: [
        Plot.barY(barRows, {
          x: "action", y: "prob", fill: "kind",
          fx: "kind", fillOpacity: 0.85,
        }),
        Plot.text(barRows, {
          x: "action", y: "prob", fx: "kind",
          text: (d) => fmt(d.prob),
          dy: -6, fontSize: 9, fill: palette.muted,
        }),
        // Mark the observed action with a dot above its bar.
        Plot.dot(
          [{ action: ACTION_LABELS[observedAtT], kind: "P (posterior)", y: 1.02 }],
          { x: "action", y: "y", fx: "kind", fill: palette.danger, r: 4 },
        ),
      ],
    }));

    // --- bottom: credit weights w_t over the trajectory ---
    const wMax = Math.max(1.5, ...weights.map((w) => w.w * 1.1));
    slots.weights.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "step t", domain: [0.5, T + 0.5] },
      y: { label: "credit weight  w_t = P(a_t|outcome)/π(a_t)", domain: [0, wMax], grid: true },
      marks: [
        Plot.ruleY([1], { stroke: palette.muted, ...dashed }),
        // rectY + interval — barY on a continuous x renders zero-width.
        Plot.rectY(weights, {
          x: "t", y: "w", interval: 1,
          fill: (d) => (d.t === step ? palette.primary : palette.secondary),
          fillOpacity: 0.85,
        }),
        Plot.text(weights, {
          x: "t", y: "w",
          text: (d) => `${d.observed} · ×${fmt(d.w)}`,
          dy: -6, fontSize: 9, fill: palette.muted,
        }),
      ],
    }));

    const wt = weights[tHighlight].w;
    const direction = wt > 1.05 ? "<strong>upweighted</strong>"
                    : wt < 0.95 ? "<strong>damped</strong>"
                    : "unchanged";
    slots.readout.innerHTML =
      `<small>step ${step}: observed a = ${ACTION_LABELS[observedAtT]}; ` +
      `prior π = ${fmt(prior[observedAtT])}, posterior P = ${fmt(post[tHighlight][observedAtT])}, ` +
      `credit weight w_${step} = <strong>×${fmt(wt)}</strong> (${direction}). ` +
      `λ = 0 makes the update identical to plain PG; λ → ∞ concentrates credit on the observed action.</small>`;
  },
});

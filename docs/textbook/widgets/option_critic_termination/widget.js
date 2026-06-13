// Widget 14.D — Option-Critic termination collapse (Chapter 14).
//
// Two-room gridworld with two options, both with learnable
// termination β(s). The reader controls the **deliberation cost** η —
// a per-termination penalty subtracted from the termination
// advantage. With η = 0, the chapter's predicted pathology shows up:
// β collapses to ~1 everywhere (every step is a new option) because
// the termination gradient
//
//   ∇_θ J = − ∇β(s') · A(s', o)
//
// is dominated by stochastic A values and pushes β upward at any
// state where the option is not strictly best. With η > 0, the
// advantage term gets an offset:
//
//   A_corrected(s', o) = A(s', o) + η
//
// and β stabilises at room-boundary states (the hallway).
//
// The simulation is a minimum-viable Option-Critic linear update on a
// 9-cell 1D corridor with a hallway in the middle. We bias option 1
// toward "go right", option 2 toward "go left", then run β-gradient
// updates against synthetic Q advantages. The point is to show the
// termination dynamics, not benchmark OC; the corridor lets the
// chapter's pathology stand out in <100 LOC.
//
//     <div id="ch14-option-critic-termination-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/option_critic_termination/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const N = 9;                 // corridor cells, hallway in the middle (cell 4).
const HALLWAY = 4;
const NUM_O = 2;
const BETA_LR = 0.05;        // β step size.
const ROLLOUT_STEPS = 2000;  // total β-gradient steps over the run.

// Sigmoid used for β parameterisation: β(s) = σ(θ_β(s)).
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// Synthetic Q-advantage for option o at state s. We bake in two
// plausible options:
//   o=0 is "go right" — best in the right room (s ≥ HALLWAY).
//   o=1 is "go left"  — best in the left  room (s ≤ HALLWAY).
// Advantage A(s, o) is positive in the option's "home" room (option
// is the right choice), negative in the other room, ~0 at the
// hallway. The chapter's prediction: a well-tuned β should be ~1 at
// the hallway and ~0 elsewhere. Without deliberation cost, β instead
// goes to 1 everywhere.
function trueAdvantage(s, o) {
  if (s === HALLWAY) return 0.05;
  const inHome = (o === 0 && s > HALLWAY) || (o === 1 && s < HALLWAY);
  return inHome ? 0.5 : -0.5;
}

// Termination-gradient update with deliberation cost η. We follow
// Bacon-Harb-Precup's correction A → A + η (here η ≥ 0 *raises*
// every advantage, which makes "keep going" look better and thus
// stops β from collapsing to 1).
//
// Update: θ_β(s) ← θ_β(s) − lr · β(s)(1 − β(s)) · (A(s, o) + η)
// (the σ' factor is β(1−β); the sign comes from the OC paper's
// gradient of −β·A under the log-derivative trick).
function trainBeta({ eta, noise, seed }) {
  // splitmix32 RNG.
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 0x1_0000_0000;
  };
  // θ_β[state][option] — initialise near 0 so β(s) ≈ 0.5 at t=0.
  const theta = Array.from({ length: N }, () => new Array(NUM_O).fill(0));
  for (let t = 0; t < ROLLOUT_STEPS; t++) {
    const sCell = Math.floor(rand() * N);
    const o = Math.floor(rand() * NUM_O);
    // Noisy advantage estimate (the chapter's collapse pathology
    // depends on the *expectation* of A being small relative to
    // noise — adding η lifts the mean).
    const a = trueAdvantage(sCell, o) + (rand() * 2 - 1) * noise;
    const b = sigmoid(theta[sCell][o]);
    // ∂β/∂θ = β(1−β). Termination loss gradient: − A · ∂β/∂θ.
    // With η correction: − (A + η) · ∂β/∂θ.
    const grad = -(a + eta) * b * (1 - b);
    theta[sCell][o] -= BETA_LR * grad;
  }
  return theta.map((row) => row.map(sigmoid));
}

defineWidget({
  hostId: "ch14-option-critic-termination-widget",
  controls: {
    eta:   { label: "η (deliberation cost)", min: 0, max: 1.0, step: 0.01, default: 0.0 },
    noise: { label: "advantage noise σ_A",   min: 0, max: 2.0, step: 0.1, default: 0.8 },
    seed:  { label: "seed",                  min: 1, max: 50,  step: 1,   default: 7 },
  },
  slots: ["heatmap", "summary"],
  render: (host, p, slots) => {
    const beta = trainBeta({ eta: p.eta, noise: p.noise, seed: p.seed | 0 });
    const cells = [];
    for (let s = 0; s < N; s++) {
      for (let o = 0; o < NUM_O; o++) {
        cells.push({
          state: s,
          option: o === 0 ? "o₁ (→ right)" : "o₂ (← left)",
          beta: beta[s][o],
          hallway: s === HALLWAY,
        });
      }
    }

    slots.heatmap.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 90,
      x: { label: "corridor state s", grid: false, ticks: d3.range(N) },
      y: { label: null, type: "band", domain: ["o₁ (→ right)", "o₂ (← left)"] },
      color: {
        type: "linear",
        domain: [0, 1],
        range: ["#1a1a2e", palette.danger],
        legend: true,
        label: "β(s, o)  (1 = always terminate)",
      },
      marks: [
        Plot.cell(cells, { x: "state", y: "option", fill: "beta", inset: 0.5 }),
        Plot.text(cells, {
          x: "state", y: "option",
          text: (d) => d.beta.toFixed(2),
          fill: "#fff", fontSize: 9,
        }),
        Plot.text(cells.filter((d) => d.hallway), {
          x: "state", y: "option",
          text: () => "H",
          dy: -14, fill: palette.warning, fontWeight: "bold", fontSize: 10,
        }),
      ],
    }));

    // Diagnostic: mean β across all states, per option. With η = 0
    // both should be ~1 (collapse); with η ~ 0.5 they stabilise lower.
    const meanByOption = [0, 1].map((o) => {
      let s = 0; for (let i = 0; i < N; i++) s += beta[i][o];
      return s / N;
    });
    const meanAtHallway = (beta[HALLWAY][0] + beta[HALLWAY][1]) / 2;
    const meanElsewhere =
      cells.filter((d) => !d.hallway).reduce((s, d) => s + d.beta, 0) /
      (cells.length - 2);

    const diag = [
      { kind: "mean β · option o₁",       value: meanByOption[0] },
      { kind: "mean β · option o₂",       value: meanByOption[1] },
      { kind: "mean β · at hallway",      value: meanAtHallway },
      { kind: "mean β · elsewhere",       value: meanElsewhere },
    ];

    slots.summary.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 160,
      x: { label: "β (avg)", domain: [0, 1], grid: true },
      y: { label: null },
      marks: [
        Plot.barX(diag, { x: "value", y: "kind", fill: palette.secondary }),
        Plot.text(diag, {
          x: "value", y: "kind",
          text: (d) => d.value.toFixed(2),
          textAnchor: "start", dx: 4, fontSize: 10,
        }),
        Plot.ruleX([0.5], { stroke: palette.muted, ...dashed }),
        Plot.text([{ x: 0.5, y: diag[0].kind, label: "β = 0.5" }], {
          x: "x", y: "y", text: "label", dy: -10, fill: palette.muted, ...annotation,
        }),
        Plot.ruleX([0]),
      ],
    }));

    const collapsed = meanElsewhere > 0.85;
    slots.readout.textContent =
      `η = ${fmt(p.eta)}  ·  mean β (non-hallway) = ${fmt(meanElsewhere)}  ·  ` +
      (collapsed
        ? "termination COLLAPSE — β → 1 everywhere; options reduce to primitives."
        : "stable — β concentrates at room boundaries (hallway).");
  },
});

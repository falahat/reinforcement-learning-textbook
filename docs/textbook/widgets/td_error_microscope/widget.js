// Widget 6.F — TD-error microscope (Chapter 6, §6.2 noise floor).
//
// The Robbins-Monro "constant α" result says that under constant step
// size, TD(0) does NOT converge to V^π — it bounces around in an
// O(√α)-radius neighborhood. The per-step TD error is approximately
// δ_t ~ N(0, σ²α) once V has settled, so as α scales linearly, so does
// the noise variance.
//
// We simulate TD(0) on a 5-state symmetric random-walk chain (S&B 2018,
// Example 6.2): states 0..4, terminals at -1 (reward 0) and 5 (reward
// +1), uniform left/right policy. Pre-train V for a few thousand steps
// so we're firmly in the "noise around V*" regime, then plot δ_t for
// the next N steps. A running mean (window=20) shows the noise floor;
// the readout compares measured variance against the linear scaling
// prediction.
//
//     <div id="ch6-td-error-microscope-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/td_error_microscope/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const N_STATES = 5;            // s0..s4
const LEFT_TERMINAL_REWARD = 0;
const RIGHT_TERMINAL_REWARD = 1;
const GAMMA = 1.0;             // episodic random walk, no discount
const WARMUP_STEPS = 4000;     // get V close to V^π before measuring
const MEASURE_STEPS = 200;     // δ_t samples we display
const SMOOTH_WIN = 20;

// True V^π for this chain (S&B Example 6.2): linear ramp 1/6, 2/6, ..., 5/6.
function trueV() {
  const v = new Float64Array(N_STATES);
  for (let s = 0; s < N_STATES; s++) v[s] = (s + 1) / (N_STATES + 1);
  return v;
}

// One step on the chain. Returns { nextState, reward, done }.
// nextState = -1 means we went off the left edge (terminal, r=0).
// nextState = N_STATES means off the right edge (terminal, r=1).
function step(s, rand) {
  const goRight = rand() < 0.5;
  const ns = goRight ? s + 1 : s - 1;
  if (ns < 0) return { ns: -1, r: LEFT_TERMINAL_REWARD, done: true };
  if (ns >= N_STATES) return { ns: N_STATES, r: RIGHT_TERMINAL_REWARD, done: true };
  return { ns, r: 0, done: false };
}

// Run TD(0) for `steps` updates. If `record` is true, return the
// per-step δ_t array (length up to `steps`).
function runTD(V, alpha, rand, steps, record) {
  const out = record ? new Float64Array(steps) : null;
  let s = Math.floor(rand() * N_STATES);
  for (let t = 0; t < steps; t++) {
    const { ns, r, done } = step(s, rand);
    const vNext = done ? 0 : V[ns];
    const delta = r + GAMMA * vNext - V[s];
    V[s] += alpha * delta;
    if (record) out[t] = delta;
    s = done ? Math.floor(rand() * N_STATES) : ns;
  }
  return out;
}

defineWidget({
  hostId: "ch6-td-error-microscope-widget",
  controls: {
    alpha: { label: "α (step size)", min: 0.01, max: 0.5, step: 0.005, default: 0.1 },
    seed:  { label: "seed",          min: 1,    max: 50,  step: 1,     default: 7 },
  },
  slots: ["main"],
  render: (host, { alpha, seed }, slots) => {
    const rand = mulberry32(seed | 0);
    const V = new Float64Array(N_STATES);
    // Initialise at the standard 0.5 (S&B convention).
    for (let s = 0; s < N_STATES; s++) V[s] = 0.5;

    // Warmup so we're near V^π; then record δ_t.
    runTD(V, alpha, rand, WARMUP_STEPS, false);
    const deltas = runTD(V, alpha, rand, MEASURE_STEPS, true);

    // Running mean window.
    const data = [];
    let runSum = 0;
    const ring = new Float64Array(SMOOTH_WIN);
    for (let t = 0; t < MEASURE_STEPS; t++) {
      const d = deltas[t];
      const old = ring[t % SMOOTH_WIN];
      runSum += d - old;
      ring[t % SMOOTH_WIN] = d;
      const winLen = Math.min(t + 1, SMOOTH_WIN);
      data.push({ t, delta: d, mean: runSum / winLen });
    }

    // Sample variance of δ.
    let mean = 0;
    for (let t = 0; t < MEASURE_STEPS; t++) mean += deltas[t];
    mean /= MEASURE_STEPS;
    let varD = 0;
    for (let t = 0; t < MEASURE_STEPS; t++) {
      const e = deltas[t] - mean;
      varD += e * e;
    }
    varD /= MEASURE_STEPS;

    // V^π for this chain has true reward variance σ² = V^π(s)(1 − V^π(s))
    // for the underlying Bernoulli-style return; at the chain's centre
    // s = 2 this is 1/2·1/2 = 1/4. The convergence-noise-floor prediction
    // is Var(δ) ≈ 2 α σ² in the small-α limit (Tsitsiklis-Van Roy 1997,
    // §4). We use σ² ≈ mean over states of V^π(1-V^π) ≈ 0.222.
    const Vt = trueV();
    let sigma2 = 0;
    for (let s = 0; s < N_STATES; s++) sigma2 += Vt[s] * (1 - Vt[s]);
    sigma2 /= N_STATES;
    const predictedVar = 2 * alpha * sigma2;

    // Symmetric y-domain so the running mean (≈ 0) sits centered.
    const yMax = Math.max(0.6, 4 * Math.sqrt(varD));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      x: { label: "step t", grid: true, domain: [0, MEASURE_STEPS - 1] },
      y: { label: "TD error δₜ", domain: [-yMax, yMax], grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(data, { x: "t", y: "delta", stroke: palette.secondary, strokeWidth: 1, strokeOpacity: 0.7 }),
        Plot.line(data, { x: "t", y: "mean", stroke: palette.primary, strokeWidth: 2 }),
        Plot.ruleY([Math.sqrt(predictedVar), -Math.sqrt(predictedVar)], {
          stroke: palette.warning, ...dashed,
        }),
      ],
    }));

    slots.readout.textContent =
      `Var(δ) measured = ${fmt(varD)}  ·  predicted ≈ 2ασ² = ${fmt(predictedVar)}  ·  α = ${fmt(alpha)}`;
  },
});

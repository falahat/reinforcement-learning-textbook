// Widget 15.A — Baird's counterexample live (Chapter 15).
//
// The classic 7-state divergence MDP from [Baird 1995]. Seven states
// (six "upper" + one "lower"), one action, deterministic dynamics:
//   - "dashed" behaviour: pick the dashed action with prob 6/7 → lower
//     state, with prob 1/7 → upper state uniformly.
//   - "solid" target policy: always go to the lower state.
// Reward is 0 everywhere; γ = 0.99. True value is 0 — so the true
// weights are zero in the span of the features. Linear FA features:
//
//   φ(s_i) = 2·e_i + e_8   for i ∈ 1..6  (upper states)
//   φ(s_7) =     e_7 + 2·e_8              (lower state)
//
// (eight weights total). Off-policy semi-gradient TD with importance
// sampling ratio ρ(s_7) = 7 (target/behaviour) drives weights to ±∞
// despite the true value being representable as θ = 0.
//
// The widget plots all eight weights vs. step. A toggle switches to
// on-policy (ρ = 1, behaviour = target = "always go to s_7"); weights
// then stay bounded — the same algorithm, same features, just one
// component of the deadly triad removed.
//
//     <div id="ch15-baird-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/baird_counterexample/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const N_STATES = 7;          // upper states s_1..s_6, lower state s_7
const N_FEATURES = 8;
const LOWER = 6;             // 0-indexed lower state

// Feature row per state. Row i has 8 entries; nonzero values match
// Baird's classic encoding.
const PHI = (() => {
  const m = [];
  for (let i = 0; i < N_STATES; i++) {
    const row = new Float64Array(N_FEATURES);
    if (i < LOWER) {
      row[i] = 2.0;     // 2·e_i for upper state s_{i+1}
      row[7] = 1.0;     // + e_8
    } else {
      row[6] = 1.0;     // e_7
      row[7] = 2.0;     // + 2·e_8
    }
    m.push(row);
  }
  return m;
})();

function dot(w, phi) {
  let s = 0;
  for (let k = 0; k < N_FEATURES; k++) s += w[k] * phi[k];
  return s;
}

// splitmix32 RNG.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 0x1_0000_0000;
  };
}

// One linear-TD trajectory. Returns the history of all weights.
function simulate({ steps, alpha, gamma, offPolicy, seed }) {
  const rand = rng(seed);
  // Baird's classic init: w_1..w_6 = 1, w_7 = 10, w_8 = 1.
  const w = new Float64Array(N_FEATURES);
  for (let k = 0; k < 6; k++) w[k] = 1.0;
  w[6] = 10.0;
  w[7] = 1.0;

  // Behaviour-policy state visitation. Off-policy: 6/7 of the time we
  // *come from* an upper state (the dashed action lands there), 1/7
  // from the lower. (We pick the starting state, then the next state
  // is the lower one — that's the only transition.) On-policy: every
  // visit is from the lower state.
  const history = [];
  history.push({ step: 0, ...weightsToFrame(w) });

  for (let t = 1; t <= steps; t++) {
    let s;
    let rho;
    if (offPolicy) {
      // Behaviour: 6/7 chance to start from a uniformly-chosen upper,
      // 1/7 chance to start from lower. Target policy "solid" always
      // transitions to lower from anywhere → ρ depends on which state
      // we sampled. With deterministic next-state in both policies
      // and behaviour-dashed having p = 1/7 to upper-uniform + 6/7 to
      // lower, the IS ratio for the "go to lower" transition is
      // 1 / (6/7) = 7/6 from upper and 1 / (1/7) = 7 from lower if
      // they happen — Baird simplifies this. We use the canonical
      // (S&B Ex. 11.2) presentation: every step is from one of the
      // upper states (uniform) and the IS ratio is 7 — that's the
      // configuration that diverges.
      s = Math.floor(rand() * 6); // upper 0..5
      rho = 7.0;
    } else {
      // On-policy "solid": always from lower; ρ = 1.
      s = LOWER;
      rho = 1.0;
    }
    const sNext = LOWER;
    const phi = PHI[s];
    const phiNext = PHI[sNext];
    const v = dot(w, phi);
    const vNext = dot(w, phiNext);
    const tdError = 0 /* reward */ + gamma * vNext - v;
    // Semi-gradient: gradient is φ(s), not φ(s) − γ·φ(s').
    for (let k = 0; k < N_FEATURES; k++) {
      w[k] += alpha * rho * tdError * phi[k];
    }
    // Sample-decimated history so we keep ~200 frames at most.
    const stride = Math.max(1, Math.floor(steps / 200));
    if (t % stride === 0 || t === steps) {
      history.push({ step: t, ...weightsToFrame(w) });
    }
  }
  return history;
}

function weightsToFrame(w) {
  const f = {};
  for (let k = 0; k < N_FEATURES; k++) f[`w${k + 1}`] = w[k];
  // Also stash the max abs weight — useful for the readout.
  let m = 0;
  for (let k = 0; k < N_FEATURES; k++) m = Math.max(m, Math.abs(w[k]));
  f.maxAbs = m;
  return f;
}

defineWidget({
  hostId: "ch15-baird-widget",
  controls: {
    mode: {
      type: "select",
      label: "policy",
      options: [
        { value: "off",  label: "off-policy (Baird, diverges)" },
        { value: "on",   label: "on-policy (stable)" },
      ],
      default: "off",
    },
    steps: { label: "steps", min: 100,  max: 5000, step: 100,   default: 1000 },
    alpha: { label: "α",     min: 0.005, max: 0.2,  step: 0.005, default: 0.01 },
    gamma: { label: "γ",     min: 0.5,   max: 0.999, step: 0.005, default: 0.99 },
    seed:  { label: "seed",  min: 1,     max: 50,   step: 1,     default: 7 },
  },
  render: (host, p, slots) => {
    const offPolicy = p.mode === "off";
    const hist = simulate({
      steps: p.steps | 0,
      alpha: p.alpha,
      gamma: p.gamma,
      offPolicy,
      seed: p.seed | 0,
    });

    // Long-form for Plot.line one-per-weight.
    const long = [];
    const colours = [
      palette.primary,   // w1
      palette.secondary, // w2
      palette.warning,   // w3
      palette.danger,    // w4
      palette.accent,    // w5
      "#26c6da",         // w6 cyan
      "#ec407a",         // w7 pink — the lower-state-only weight
      "#9e9d24",         // w8 olive — the shared bias weight
    ];
    for (const frame of hist) {
      for (let k = 0; k < N_FEATURES; k++) {
        long.push({ step: frame.step, w: `w${k + 1}`, value: frame[`w${k + 1}`] });
      }
    }

    const last = hist[hist.length - 1];
    // Symmetric y-domain around max-abs so divergence is visible.
    const m = Math.max(2, last.maxAbs * 1.1);
    const ySpec = offPolicy
      ? { type: "symlog", constant: 1, domain: [-m, m], grid: true, label: "weight value (symlog)" }
      : { domain: [-m, m], grid: true, label: "weight value" };

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "TD step", grid: true },
      y: ySpec,
      color: { domain: long.map((d) => d.w).filter((v, i, a) => a.indexOf(v) === i), range: colours, legend: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(long, { x: "step", y: "value", stroke: "w", strokeWidth: 1.6 }),
      ],
    }));

    const verdict = offPolicy
      ? (last.maxAbs > 100
          ? `diverging (max|θ| = ${fmt(last.maxAbs)})`
          : `growing (max|θ| = ${fmt(last.maxAbs)}); push steps higher to see blow-up`)
      : `stable (max|θ| = ${fmt(last.maxAbs)})`;
    slots.readout.textContent =
      `${offPolicy ? "off-policy semi-gradient TD" : "on-policy semi-gradient TD"} · ${verdict}`;
  },
});

// Widget 15.C — Primacy-bias / parameter-reset experiment (Chapter 15).
//
// Live demonstration of [Nikishin et al. 2022] "The Primacy Bias in
// Deep Reinforcement Learning": early bad data has an outsized,
// hard-to-shake influence on the final policy. Resetting the
// agent's parameters periodically gives the network a chance to
// re-learn from a now-better replay distribution.
//
// Toy environment: a 2-arm bandit whose mean rewards swap halfway
// through training. Arm A is initially the better arm (μ = 0.8 vs.
// 0.3); after `swap_at` steps the means flip (now 0.3 vs. 0.8). An
// ε-greedy Q-learner sees the early data and locks in Arm A. With
// no reset it can take many thousands of steps to re-learn. With a
// reset every `reset_every` steps the Q-table zeros out and re-learns
// from the *current* replay-equivalent distribution — quick recovery.
//
// We run three policies side-by-side on the same RNG seed:
//   - never reset
//   - reset every `reset_every` steps
//   - reset every 2·reset_every steps (slower cadence)
// and plot the smoothed cumulative reward. The reader watches the
// "jagged but higher" reset curves beat the locked-in baseline.
//
//     <div id="ch15-primacy-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/primacy_bias/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

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

function trueMeans(t, swapAt) {
  return t < swapAt ? [0.8, 0.3] : [0.3, 0.8];
}

// One Q-learning run with periodic reset.
function runBandit({ steps, swapAt, epsilon, alpha, resetEvery, seed }) {
  const rand = rng(seed);
  let Q = [0, 0];
  const reward = new Array(steps);
  for (let t = 0; t < steps; t++) {
    if (resetEvery > 0 && t > 0 && t % resetEvery === 0) {
      Q = [0, 0]; // hard parameter reset
    }
    // ε-greedy argmax.
    let a;
    if (rand() < epsilon) {
      a = rand() < 0.5 ? 0 : 1;
    } else {
      a = Q[0] >= Q[1] ? 0 : 1;
    }
    const mu = trueMeans(t, swapAt);
    const r = rand() < mu[a] ? 1 : 0;
    Q[a] += alpha * (r - Q[a]);
    reward[t] = r;
  }
  return reward;
}

// Boxcar-smoothed mean reward.
function smooth(arr, win) {
  const out = new Array(arr.length);
  let sum = 0;
  const q = [];
  for (let i = 0; i < arr.length; i++) {
    q.push(arr[i]);
    sum += arr[i];
    if (q.length > win) sum -= q.shift();
    out[i] = sum / q.length;
  }
  return out;
}

defineWidget({
  hostId: "ch15-primacy-widget",
  controls: {
    steps:       { label: "steps",          min: 2000, max: 40000, step: 1000, default: 20000 },
    swapAt:      { label: "swap at step",   min: 1000, max: 30000, step: 500,  default: 5000 },
    resetEvery:  { label: "reset every",    min: 500,  max: 20000, step: 500,  default: 5000 },
    epsilon:     { label: "ε",              min: 0.0,  max: 0.3,   step: 0.01, default: 0.05 },
    alpha:       { label: "α",              min: 0.01, max: 0.5,   step: 0.01, default: 0.1 },
    seed:        { label: "seed",           min: 1,    max: 50,    step: 1,    default: 7 },
  },
  render: (host, p, slots) => {
    const steps = p.steps | 0;
    const swapAt = p.swapAt | 0;
    const resetEvery = p.resetEvery | 0;
    const cfg = { steps, swapAt, epsilon: p.epsilon, alpha: p.alpha, seed: p.seed | 0 };

    const never = runBandit({ ...cfg, resetEvery: 0 });
    const reset = runBandit({ ...cfg, resetEvery });
    const slowReset = runBandit({ ...cfg, resetEvery: resetEvery * 2 });

    // Smoothing window scales with steps.
    const win = Math.max(50, Math.floor(steps / 100));
    const s1 = smooth(never, win);
    const s2 = smooth(reset, win);
    const s3 = smooth(slowReset, win);

    // Sub-sample to keep the plot light (~600 points / series).
    const stride = Math.max(1, Math.floor(steps / 600));
    const rows = [];
    for (let i = 0; i < steps; i += stride) {
      rows.push({ t: i, policy: "never reset",         value: s1[i] });
      rows.push({ t: i, policy: `reset / ${resetEvery}`, value: s2[i] });
      rows.push({ t: i, policy: `reset / ${resetEvery * 2}`, value: s3[i] });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "step", grid: true },
      y: { label: "smoothed mean reward", domain: [0, 1], grid: true },
      color: {
        domain: ["never reset", `reset / ${resetEvery}`, `reset / ${resetEvery * 2}`],
        range: [palette.danger, palette.primary, palette.secondary],
        legend: true,
      },
      marks: [
        // Show the swap moment.
        Plot.ruleX([swapAt], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: swapAt, y: 0.05, label: "true-means swap" }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 6,
            fill: palette.warning, ...annotation },
        ),
        Plot.ruleY([0.8], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.text(
          [{ x: 0, y: 0.8, label: "best-arm mean = 0.8" }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4, dy: -4,
            fill: palette.muted, ...annotation },
        ),
        Plot.line(rows, { x: "t", y: "value", stroke: "policy", strokeWidth: 1.6 }),
      ],
    }));

    // Final-window mean reward per policy.
    const tail = Math.max(500, Math.floor(steps / 20));
    function tailMean(arr) {
      let s = 0;
      const lo = arr.length - tail;
      for (let i = lo; i < arr.length; i++) s += arr[i];
      return s / tail;
    }
    slots.readout.textContent =
      `final-${tail}-step mean reward: ` +
      `never=${fmt(tailMean(never))}, ` +
      `reset/${resetEvery}=${fmt(tailMean(reset))}, ` +
      `reset/${resetEvery * 2}=${fmt(tailMean(slowReset))}`;
  },
});

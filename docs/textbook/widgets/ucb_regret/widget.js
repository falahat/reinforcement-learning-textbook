// Widget 12.H — UCB1 regret-vs-T curve, with epsilon-greedy overlays.
//
// Runs three exploration policies on the same fixed 5-arm Bernoulli
// bandit and plots cumulative regret over T rounds:
//   - UCB1 with exploration constant c
//   - epsilon-greedy with constant epsilon
//   - epsilon-greedy with decaying epsilon = 1/sqrt(t)
//
// Pedagogy: the UCB1 curve flattens (regret = O(log T)), the constant-
// epsilon curve climbs linearly (regret = Omega(epsilon * T)), and
// decaying-epsilon sits in between. Slide c or epsilon and watch the
// asymptotic behaviour change. Add a log-T reference line so the
// O(log T) shape is visible by eye.
//
// We run all three policies in one pass per render. T up to 5000 stays
// responsive (~5 ms total).
//
// Mount:
//     <div id="ch12-ucb-regret-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/ucb_regret/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { splitmix64 as makeRng } from "../shared/random.js";

const TRUE_MEANS = [0.20, 0.40, 0.55, 0.70, 0.85];
const K_ARMS = TRUE_MEANS.length;
const BEST_MEAN = Math.max(...TRUE_MEANS);

function bernoulli(p, rng) {
  return rng() < p ? 1 : 0;
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

// Run one policy for T rounds, return per-step cumulative regret.
function runUcb(T, c, seed) {
  const rng = makeRng(seed);
  const sums = new Array(K_ARMS).fill(0);
  const counts = new Array(K_ARMS).fill(0);
  const regret = new Array(T);
  let cum = 0;
  for (let t = 0; t < T; t++) {
    let best = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < K_ARMS; k++) {
      const score = counts[k] === 0
        ? Infinity
        : sums[k] / counts[k] + c * Math.sqrt(Math.log(Math.max(t + 1, 1)) / counts[k]);
      if (score > bestScore) { bestScore = score; best = k; }
    }
    const r = bernoulli(TRUE_MEANS[best], rng);
    sums[best] += r;
    counts[best] += 1;
    cum += BEST_MEAN - TRUE_MEANS[best];
    regret[t] = cum;
  }
  return regret;
}

function runEpsilonGreedy(T, epsilonFn, seed) {
  const rng = makeRng(seed);
  const sums = new Array(K_ARMS).fill(0);
  const counts = new Array(K_ARMS).fill(0);
  const regret = new Array(T);
  let cum = 0;
  for (let t = 0; t < T; t++) {
    let arm;
    const eps = epsilonFn(t);
    if (rng() < eps) {
      arm = Math.floor(rng() * K_ARMS);
    } else {
      const means = sums.map((s, k) => counts[k] === 0 ? 0.5 : s / counts[k]);
      arm = argmax(means);
    }
    const r = bernoulli(TRUE_MEANS[arm], rng);
    sums[arm] += r;
    counts[arm] += 1;
    cum += BEST_MEAN - TRUE_MEANS[arm];
    regret[t] = cum;
  }
  return regret;
}

defineWidget({
  hostId: "ch12-ucb-regret-widget",
  controls: {
    T:       { label: "T (rounds)",     min: 200, max: 5000, step: 100, default: 2000 },
    c:       { label: "UCB c",          min: 0.1, max: 3.0,  step: 0.05, default: 1.0 },
    epsilon: { label: "epsilon (const)", min: 0.01, max: 0.3, step: 0.01, default: 0.1 },
    seed:    { label: "seed",           min: 1,   max: 999,  step: 1,   default: 17 },
  },
  render: (host, { T, c, epsilon, seed }, slots) => {
    const tInt = Math.round(T);
    const ucbR = runUcb(tInt, c, seed);
    const egConstR = runEpsilonGreedy(tInt, () => epsilon, seed + 1);
    const egDecayR = runEpsilonGreedy(tInt, (t) => 1 / Math.sqrt(t + 1), seed + 2);

    // Downsample for plotting speed (~250 points).
    const stride = Math.max(1, Math.floor(tInt / 250));
    const data = [];
    for (let t = 0; t < tInt; t += stride) {
      data.push({ t, regret: ucbR[t], policy: "UCB1" });
      data.push({ t, regret: egConstR[t], policy: "epsilon-greedy (const)" });
      data.push({ t, regret: egDecayR[t], policy: "epsilon = 1/sqrt(t)" });
    }
    // Always include the final point.
    data.push({ t: tInt - 1, regret: ucbR[tInt - 1], policy: "UCB1" });
    data.push({ t: tInt - 1, regret: egConstR[tInt - 1], policy: "epsilon-greedy (const)" });
    data.push({ t: tInt - 1, regret: egDecayR[tInt - 1], policy: "epsilon = 1/sqrt(t)" });

    // O(log T) reference shape scaled to fit the UCB curve at the end.
    const finalUcb = ucbR[tInt - 1];
    const logRef = d3.range(1, tInt + 1, stride).map((t) => ({
      t, y: finalUcb * Math.log(t + 1) / Math.log(tInt + 1),
    }));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "t (rounds)", grid: true, domain: [0, tInt] },
      y: { label: "cumulative regret", grid: true, zero: true },
      color: {
        legend: true,
        domain: ["UCB1", "epsilon-greedy (const)", "epsilon = 1/sqrt(t)"],
        range: [palette.primary, palette.danger, palette.warning],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(data, {
          x: "t", y: "regret",
          stroke: "policy", z: "policy",
          strokeWidth: 2,
        }),
        // O(log T) reference, dashed.
        Plot.line(logRef, {
          x: "t", y: "y",
          stroke: palette.muted, ...dashed,
        }),
        Plot.text(
          [{ x: tInt, y: logRef[logRef.length - 1].y, label: "O(log T) shape" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    slots.readout.textContent =
      `T=${tInt}, c=${c.toFixed(2)}, epsilon=${epsilon.toFixed(2)}: ` +
      `UCB1 regret=${ucbR[tInt - 1].toFixed(1)}; ` +
      `eps-const=${egConstR[tInt - 1].toFixed(1)}; ` +
      `eps-decay=${egDecayR[tInt - 1].toFixed(1)}`;
  },
});

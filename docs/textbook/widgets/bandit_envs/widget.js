// Widget 12.E — Bernoulli vs Gaussian vs adversarial bandit under UCB1.
//
// Same 5 arms, same UCB1 algorithm; three reward environments. UCB1
// achieves O(log T) regret on Bernoulli and Gaussian — but on the
// adversarial bandit (where the adversary picks the reward to minimise
// our payoff given our action) UCB1 fails: linear regret.
//
// Mount:
//     <div id="ch12-bandit-envs-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/bandit_envs/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const TRUE_MEANS = [0.20, 0.40, 0.55, 0.70, 0.85];
const K = TRUE_MEANS.length;
const BEST = Math.max(...TRUE_MEANS);
const T = 2000;

function argmaxUcb(sums, counts, t, c) {
  let best = 0;
  let bestScore = -Infinity;
  for (let k = 0; k < K; k++) {
    const score = counts[k] === 0
      ? Infinity
      : sums[k] / counts[k] + c * Math.sqrt(Math.log(Math.max(t + 1, 1)) / counts[k]);
    if (score > bestScore) { bestScore = score; best = k; }
  }
  return best;
}

// Bernoulli: r ∈ {0,1} with mean μ_k.
function rewardBernoulli(arm, rng) {
  return rng() < TRUE_MEANS[arm] ? 1 : 0;
}

// Gaussian: r ~ N(μ_k, 0.1²), clipped to [0,1].
function rewardGaussian(arm, rng) {
  const r = TRUE_MEANS[arm] + 0.1 * gauss(rng);
  return Math.max(0, Math.min(1, r));
}

// Adversarial: the adversary observes our last action and pays the
// MINIMUM of the (clipped) Gaussian and a punishing low reward on the
// arm we keep choosing. Specifically: every time we re-pull the most-
// pulled arm, the adversary halves its reward for that round. This
// violates UCB1's stationarity assumption — empirical means become
// biased downward on whichever arm we lean on, but UCB still grows
// √(log t/N_k), so we keep exploring while regret accumulates linearly
// against the unchanging best mean.
function makeAdversary() {
  const pullCounts = new Array(K).fill(0);
  return function rewardAdversarial(arm, rng) {
    pullCounts[arm] += 1;
    // Find which arm we pull most.
    let maxPulls = -1; let maxArm = 0;
    for (let k = 0; k < K; k++) {
      if (pullCounts[k] > maxPulls) { maxPulls = pullCounts[k]; maxArm = k; }
    }
    // The "true" reward we report against (vs. unchanging BEST mean for
    // regret accounting).
    const base = TRUE_MEANS[arm] + 0.05 * gauss(rng);
    if (arm === maxArm) {
      // Punish the leader: scale down toward 0.
      return Math.max(0, base * 0.3);
    }
    return Math.max(0, Math.min(1, base));
  };
}

function runUcb(rewardFn, c, seed) {
  const rng = mulberry32(seed);
  const sums = new Array(K).fill(0);
  const counts = new Array(K).fill(0);
  const regret = new Array(T);
  let cum = 0;
  for (let t = 0; t < T; t++) {
    const arm = argmaxUcb(sums, counts, t, c);
    const r = rewardFn(arm, rng);
    sums[arm] += r;
    counts[arm] += 1;
    // Regret is measured against the best STATIONARY arm's true mean.
    cum += BEST - TRUE_MEANS[arm];
    regret[t] = cum;
  }
  return { regret, counts };
}

defineWidget({
  hostId: "ch12-bandit-envs-widget",
  controls: {
    env: {
      type: "select",
      label: "environment",
      options: [
        { value: "bernoulli", label: "Bernoulli" },
        { value: "gaussian", label: "Gaussian" },
        { value: "adversarial", label: "Adversarial" },
      ],
      default: "bernoulli",
    },
    c:    { label: "UCB c", min: 0.1, max: 3.0, step: 0.05, default: 1.0 },
    seed: { label: "seed", min: 1, max: 999, step: 1, default: 11 },
  },
  render: (host, { env, c, seed }, slots) => {
    const sd = Math.round(seed);
    let bernR, gaussR, advR;
    bernR = runUcb(rewardBernoulli, c, sd).regret;
    gaussR = runUcb(rewardGaussian, c, sd + 100).regret;
    advR = runUcb(makeAdversary(), c, sd + 200).regret;

    // Highlight selected env, but show all three faintly for comparison.
    const stride = Math.max(1, Math.floor(T / 250));
    const data = [];
    for (let t = 0; t < T; t += stride) {
      data.push({ t, regret: bernR[t], env: "Bernoulli" });
      data.push({ t, regret: gaussR[t], env: "Gaussian" });
      data.push({ t, regret: advR[t], env: "Adversarial" });
    }
    data.push({ t: T - 1, regret: bernR[T - 1], env: "Bernoulli" });
    data.push({ t: T - 1, regret: gaussR[T - 1], env: "Gaussian" });
    data.push({ t: T - 1, regret: advR[T - 1], env: "Adversarial" });

    // Reference: O(log T) curve, scaled to Bernoulli final.
    const logRef = [];
    const target = bernR[T - 1];
    for (let t = 1; t < T; t += stride) {
      logRef.push({ t, y: target * Math.log(t + 1) / Math.log(T + 1) });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "t (rounds)", grid: true, domain: [0, T] },
      y: { label: "cumulative regret", grid: true, zero: true },
      color: {
        legend: true,
        domain: ["Bernoulli", "Gaussian", "Adversarial"],
        range: [palette.primary, palette.secondary, palette.danger],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(data, {
          x: "t", y: "regret", stroke: "env", z: "env",
          strokeWidth: (d) => d.env === envName(env) ? 2.5 : 1.2,
          strokeOpacity: (d) => d.env === envName(env) ? 1 : 0.45,
        }),
        Plot.line(logRef, {
          x: "t", y: "y", stroke: palette.muted, ...dashed,
        }),
        Plot.text(
          [{ x: T, y: logRef[logRef.length - 1].y, label: "O(log T) shape" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    const finals = {
      Bernoulli: bernR[T - 1],
      Gaussian: gaussR[T - 1],
      Adversarial: advR[T - 1],
    };
    slots.readout.textContent =
      `Final regret @ T=${T}: Bernoulli=${finals.Bernoulli.toFixed(1)}, ` +
      `Gaussian=${finals.Gaussian.toFixed(1)}, ` +
      `Adversarial=${finals.Adversarial.toFixed(1)}. ` +
      `Highlighted: ${envName(env)}.`;
  },
});

function envName(env) {
  if (env === "bernoulli") return "Bernoulli";
  if (env === "gaussian") return "Gaussian";
  return "Adversarial";
}

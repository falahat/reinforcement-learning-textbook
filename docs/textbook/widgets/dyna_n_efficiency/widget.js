// Widget 13.F — Dyna planning-step efficiency under model error (Chapter 13).
//
// One real-environment FrozenLake-style chain. Six lines per chart,
// one per n_plan ∈ {0, 1, 5, 20, 50, 200}. The reader drags a slider
// `model_noise` ∈ [0, 1] that probabilistically corrupts a model
// transition (with prob noise, the model returns a uniformly-random
// transition instead of the true one).
//
// What the reader sees:
//   - noise = 0: high n_plan dominates (Dyna's sample-efficiency win).
//   - noise ≈ 0.3: high n_plan still helps but no longer crushingly.
//   - noise ≈ 0.7: high n_plan COLLAPSES below n=0 — the model is
//     poisoning learning.
//
// The chain is a 10-state corridor with a +1 absorbing goal at state 9
// and step-cost 0; ε-greedy Q-learning with α = 0.4, γ = 0.95. Each
// "real sample" is one (s, a, r, s') taken in the env. After each
// real sample, n_plan model-based updates are performed on uniformly-
// random previously-visited (s, a) pairs; with prob `noise` the
// model's recall is a random (s', r) instead of the true transition.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-dyna-n-efficiency-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/dyna_n_efficiency/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const NUM_S = 10;
const ACTIONS = [-1, +1]; // left, right
const NUM_A = ACTIONS.length;
const GOAL_S = NUM_S - 1;
const REWARD_GOAL = 1.0;
const GAMMA = 0.95;
const ALPHA = 0.4;
const EPS = 0.1;
const REAL_STEPS = 600;
const N_PLAN_VALUES = [0, 1, 5, 20, 50, 200];
const EVAL_INTERVAL = 25; // record return every N real steps

function nextState(s, a) {
  if (s === GOAL_S) return s;
  return Math.max(0, Math.min(NUM_S - 1, s + ACTIONS[a]));
}
function reward(s, a, sp) {
  return s !== GOAL_S && sp === GOAL_S ? REWARD_GOAL : 0;
}

// Greedy-policy return from s=0 to terminal: simulate up to NUM_S * 4 steps.
function greedyReturn(Q) {
  let s = 0;
  let G = 0;
  let disc = 1;
  for (let t = 0; t < NUM_S * 4; t++) {
    if (s === GOAL_S) break;
    let bestA = 0;
    let bestQ = -Infinity;
    for (let a = 0; a < NUM_A; a++) {
      if (Q[s][a] > bestQ) { bestQ = Q[s][a]; bestA = a; }
    }
    const sp = nextState(s, bestA);
    G += disc * reward(s, bestA, sp);
    disc *= GAMMA;
    s = sp;
  }
  return G;
}

function runDyna({ nPlan, noise, seed }) {
  const rng = mulberry32(seed);
  const Q = [];
  for (let s = 0; s < NUM_S; s++) {
    Q.push(new Array(NUM_A).fill(0));
  }
  // model[s][a] = { sp, r }; null if unvisited.
  const model = [];
  for (let s = 0; s < NUM_S; s++) {
    model.push(new Array(NUM_A).fill(null));
  }
  const visited = []; // list of {s, a} unique
  const visitedSet = new Set();

  const curve = [];
  let s = 0;
  for (let step = 1; step <= REAL_STEPS; step++) {
    if (s === GOAL_S) s = 0; // reset on terminal
    // ε-greedy
    let a;
    if (rng() < EPS) {
      a = rng() < 0.5 ? 0 : 1;
    } else {
      a = Q[s][0] >= Q[s][1] ? 0 : 1;
    }
    const sp = nextState(s, a);
    const r = reward(s, a, sp);
    // Q update
    const tgt = r + (sp === GOAL_S ? 0 : GAMMA * Math.max(Q[sp][0], Q[sp][1]));
    Q[s][a] += ALPHA * (tgt - Q[s][a]);
    // Update model
    model[s][a] = { sp, r };
    const key = s * NUM_A + a;
    if (!visitedSet.has(key)) {
      visitedSet.add(key);
      visited.push({ s, a });
    }
    // Planning
    for (let p = 0; p < nPlan && visited.length > 0; p++) {
      const pick = visited[Math.floor(rng() * visited.length)];
      let mSp, mR;
      if (rng() < noise) {
        // Corrupt model recall: random transition.
        mSp = Math.floor(rng() * NUM_S);
        mR = rng() < 0.05 ? 1 : 0;
      } else {
        const m = model[pick.s][pick.a];
        mSp = m.sp;
        mR = m.r;
      }
      const tgtP = mR + (mSp === GOAL_S ? 0 : GAMMA * Math.max(Q[mSp][0], Q[mSp][1]));
      Q[pick.s][pick.a] += ALPHA * (tgtP - Q[pick.s][pick.a]);
    }
    s = sp;

    if (step % EVAL_INTERVAL === 0) {
      curve.push({ step, ret: greedyReturn(Q) });
    }
  }
  return curve;
}

defineWidget({
  hostId: "ch13-dyna-n-efficiency-widget",
  controls: {
    noise: { label: "p_model_error", min: 0, max: 1, step: 0.02, default: 0.0 },
  },
  slots: ["main", "summary"],
  render: (host, { noise }, slots) => {
    // Average over a few seeds per n_plan for smoother curves.
    const SEEDS = [7, 11, 23];
    const all = [];
    for (const n of N_PLAN_VALUES) {
      const sumByStep = new Map();
      for (const seed of SEEDS) {
        const curve = runDyna({ nPlan: n, noise, seed });
        for (const pt of curve) {
          const prev = sumByStep.get(pt.step) || 0;
          sumByStep.set(pt.step, prev + pt.ret);
        }
      }
      for (const [step, sumRet] of sumByStep.entries()) {
        all.push({
          step,
          ret: sumRet / SEEDS.length,
          nPlan: n,
          label: `n=${n}`,
        });
      }
    }
    all.sort((a, b) => a.step - b.step);

    const optimalReturn = Math.pow(GAMMA, NUM_S - 2); // greedy from s=0 reaches goal in NUM_S-1 steps

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "real env steps", grid: true, domain: [0, REAL_STEPS] },
      y: { label: "policy return (greedy)", grid: true, domain: [0, 1] },
      color: {
        domain: N_PLAN_VALUES.map((n) => `n=${n}`),
        scheme: "turbo",
        legend: true,
      },
      marks: [
        Plot.ruleY([optimalReturn], {
          stroke: palette.muted,
          ...dashed,
        }),
        Plot.text(
          [{ x: REAL_STEPS, y: optimalReturn, label: "optimum" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.muted, ...annotation },
        ),
        Plot.line(all, {
          x: "step",
          y: "ret",
          stroke: "label",
          strokeWidth: 1.8,
        }),
      ],
    }));

    // Summary: final return per n_plan.
    const finalByN = new Map();
    for (const n of N_PLAN_VALUES) {
      const lastForN = all.filter((d) => d.nPlan === n).at(-1);
      finalByN.set(n, lastForN ? lastForN.ret : 0);
    }
    const summary = [...finalByN.entries()].map(([n, r]) => ({
      n_plan: `n=${n}`,
      ret: r,
    }));

    slots.summary.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      marginLeft: 60,
      x: { label: "final greedy return", grid: true, domain: [0, 1] },
      y: { label: null, domain: summary.map((d) => d.n_plan).reverse() },
      marks: [
        Plot.barX(summary, {
          y: "n_plan",
          x: "ret",
          fill: (d) => d.ret > 0.5 ? palette.primary : palette.danger,
          fillOpacity: 0.85,
        }),
        Plot.text(summary, {
          y: "n_plan",
          x: "ret",
          text: (d) => fmt(d.ret),
          textAnchor: "start",
          dx: 4,
          fontSize: 10,
          fill: "white",
        }),
      ],
    }));

    const bestN = [...finalByN.entries()].reduce((b, e) => e[1] > b[1] ? e : b);
    slots.readout.innerHTML =
      `p_model_error = ${fmt(noise)} · best n_plan = ${bestN[0]} ` +
      `(final return ${fmt(bestN[1])})<br>` +
      `<small>at noise ≈ 0, large n_plan dominates; ` +
      `at noise → 1, large n_plan collapses below n=0.</small>`;
  },
});

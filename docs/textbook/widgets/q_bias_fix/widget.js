// Widget 15.G — Q-bias fix walkthrough (Chapter 15).
//
// Side-by-side comparison of the three fixes the chapter proposes
// for the Q-bias bootstrap pathology, on a tiny homeostatic toy:
//
//   state := drive ∈ {Sated, Hungry}
//   actions := {Wait, Consume}
//   Wait:    drive → drive   (no change)
//   Consume: drive → Sated   (relieves hunger; only useful when Hungry)
//
// Per-step reward depends on the fix:
//
//   baseline (`w_alive = 1`):
//     r = w_alive − drive_cost(drive')
//        with drive_cost(Sated) = 0, drive_cost(Hungry) = 0.5
//
//   Fix 1 (`w_alive = 0`, delta-based):
//     r = drive_cost(drive) − drive_cost(drive')   (negative delta penalty)
//
//   Fix 3 (advantage / dueling): same R as baseline but the learner
//     maintains V(s) separately from A(s, a); argmax is over A only.
//     Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a)).
//
// We run on-policy TD(0) (uniform exploration) on each variant for
// `steps` ticks and plot:
//   - Q(Hungry, Wait) and Q(Hungry, Consume) over time, per fix
//     (this is the "lock-in" check: in baseline the two curves both
//     saturate near w_alive/(1−γ) = 10 and the *gap* between them
//     is small; in Fix 1 only Consume's Q is positive; in Fix 3 the
//     argmax-relevant advantage A(Hungry, Consume) > A(Hungry, Wait)).
//   - final argmax in state Hungry, per fix.
//
//     <div id="ch15-qbias-fix-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/q_bias_fix/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

const SATED = 0, HUNGRY = 1;
const WAIT = 0, CONSUME = 1;
const STATE_NAMES = ["Sated", "Hungry"];
const ACT_NAMES = ["Wait", "Consume"];

function driveCost(s, driveCostHungry) {
  return s === HUNGRY ? driveCostHungry : 0;
}

function step(s, a) {
  if (a === CONSUME) return SATED;
  return s;
}

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

// Three fixes share most of the loop, differ only in reward + update.
function trainAll({ steps, alpha, gamma, wAlive, hungerProb, driveCostHungry, sampleEvery, seed }) {
  const rand = rng(seed);
  // For tabular Q we store Q[s][a]. For Fix 3 we additionally store V[s].
  const baselineQ = [[0, 0], [0, 0]];
  const fix1Q     = [[0, 0], [0, 0]];
  const fix3Q     = [[0, 0], [0, 0]];
  const fix3V     = [0, 0];

  const trace = [];

  for (let t = 0; t < steps; t++) {
    // Resample state: simulating the agent in a hunger-then-sated cycle.
    const s = rand() < hungerProb ? HUNGRY : SATED;
    const a = rand() < 0.5 ? WAIT : CONSUME; // uniform exploration so
                                              // every action is tried.
    const sNext = step(s, a);

    // Reward variants.
    const rBaseline = wAlive - driveCost(sNext, driveCostHungry);
    const rFix1     = driveCost(s, driveCostHungry) - driveCost(sNext, driveCostHungry);
    // Fix 3 uses the baseline reward but the dueling decomposition.
    const rFix3     = rBaseline;

    // TD-target = r + γ·max_a' Q(s', a'); on-policy expected-V works too
    // for a 2x2 toy but max keeps it close to the project's Q-learning.
    const maxQ = (Qtbl, s2) => Math.max(Qtbl[s2][0], Qtbl[s2][1]);
    baselineQ[s][a] += alpha * (rBaseline + gamma * maxQ(baselineQ, sNext) - baselineQ[s][a]);
    fix1Q[s][a]     += alpha * (rFix1     + gamma * maxQ(fix1Q,     sNext) - fix1Q[s][a]);

    // Fix 3 (dueling). Update V(s) toward V_target = r + γV(s'). Update
    // Q (= V + centered A) toward the standard TD target; the A's are
    // recovered as Q − V and renormalised on read.
    const tdV = rFix3 + gamma * fix3V[sNext] - fix3V[s];
    fix3V[s] += alpha * tdV;
    const tdQ = rFix3 + gamma * maxQ(fix3Q, sNext) - fix3Q[s][a];
    fix3Q[s][a] += alpha * tdQ;

    if (t % sampleEvery === 0 || t === steps - 1) {
      const fix3A = (st) => {
        const mean = 0.5 * (fix3Q[st][0] + fix3Q[st][1]);
        return [fix3Q[st][0] - mean, fix3Q[st][1] - mean];
      };
      const aH = fix3A(HUNGRY);
      trace.push({
        t,
        bQWait:    baselineQ[HUNGRY][WAIT],
        bQConsume: baselineQ[HUNGRY][CONSUME],
        f1QWait:   fix1Q[HUNGRY][WAIT],
        f1QConsume:fix1Q[HUNGRY][CONSUME],
        f3AWait:   aH[WAIT],
        f3AConsume:aH[CONSUME],
        f3V:       fix3V[HUNGRY],
      });
    }
  }

  function argmaxName(qHungry) {
    return qHungry[WAIT] > qHungry[CONSUME] ? "Wait (LOCK-IN)" : "Consume (correct)";
  }

  return {
    trace,
    finals: {
      baseline: { qWait: baselineQ[HUNGRY][WAIT], qConsume: baselineQ[HUNGRY][CONSUME], argmax: argmaxName(baselineQ[HUNGRY]) },
      fix1:     { qWait: fix1Q[HUNGRY][WAIT],     qConsume: fix1Q[HUNGRY][CONSUME],     argmax: argmaxName(fix1Q[HUNGRY])     },
      fix3:     {
        qWait: fix3Q[HUNGRY][WAIT], qConsume: fix3Q[HUNGRY][CONSUME],
        aWait: fix3Q[HUNGRY][WAIT] - 0.5 * (fix3Q[HUNGRY][0] + fix3Q[HUNGRY][1]),
        aConsume: fix3Q[HUNGRY][CONSUME] - 0.5 * (fix3Q[HUNGRY][0] + fix3Q[HUNGRY][1]),
        argmax: argmaxName(fix3Q[HUNGRY]),
      },
    },
  };
}

defineWidget({
  hostId: "ch15-qbias-fix-widget",
  controls: {
    wAlive:          { label: "w_alive (baseline)", min: 0,    max: 2.0,  step: 0.05, default: 1.0 },
    driveCostHungry: { label: "drive_cost(Hungry)", min: 0.0,  max: 1.0,  step: 0.05, default: 0.5 },
    gamma:           { label: "γ",                  min: 0.5,  max: 0.99, step: 0.01, default: 0.9 },
    alpha:           { label: "α",                  min: 0.01, max: 0.5,  step: 0.01, default: 0.1 },
    hungerProb:      { label: "P(Hungry)",          min: 0.1,  max: 0.9,  step: 0.05, default: 0.5 },
    steps:           { label: "steps",              min: 500,  max: 10000, step: 500, default: 4000 },
    seed:            { label: "seed",               min: 1,    max: 50,   step: 1,    default: 7 },
  },
  slots: ["main"],
  render: (host, p, slots) => {
    const steps = p.steps | 0;
    const sampleEvery = Math.max(1, Math.floor(steps / 200));
    const { trace, finals } = trainAll({
      steps, alpha: p.alpha, gamma: p.gamma, wAlive: p.wAlive,
      hungerProb: p.hungerProb, driveCostHungry: p.driveCostHungry,
      sampleEvery, seed: p.seed | 0,
    });

    // Long-form rows: one per (time, series). We faceted by fix so the
    // three panels appear side-by-side and the reader can compare
    // *the same y-axis* across fixes.
    const rows = [];
    for (const fr of trace) {
      // Baseline panel: plot Q(Wait) and Q(Consume).
      rows.push({ t: fr.t, fix: "baseline (w_alive=1)", series: "Q(Hungry, Wait)",    value: fr.bQWait });
      rows.push({ t: fr.t, fix: "baseline (w_alive=1)", series: "Q(Hungry, Consume)", value: fr.bQConsume });
      // Fix 1 panel: Q(Wait) and Q(Consume) under delta reward.
      rows.push({ t: fr.t, fix: "Fix 1 (w_alive=0)",    series: "Q(Hungry, Wait)",    value: fr.f1QWait });
      rows.push({ t: fr.t, fix: "Fix 1 (w_alive=0)",    series: "Q(Hungry, Consume)", value: fr.f1QConsume });
      // Fix 3 panel: A(Wait) and A(Consume), the actual argmax target.
      rows.push({ t: fr.t, fix: "Fix 3 (advantage)",    series: "A(Hungry, Wait)",    value: fr.f3AWait });
      rows.push({ t: fr.t, fix: "Fix 3 (advantage)",    series: "A(Hungry, Consume)", value: fr.f3AConsume });
    }

    // Y-domain: widest is baseline saturating near w_alive/(1−γ).
    const sat = p.wAlive / Math.max(1e-6, 1 - p.gamma);
    const yHi = Math.max(sat * 1.15, 2);
    const yLo = -Math.max(p.driveCostHungry / Math.max(1e-6, 1 - p.gamma), 1) - 0.5;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      marginLeft: 50,
      x: { label: "step", grid: true },
      y: { label: "value", grid: true, domain: [yLo, yHi] },
      fx: {
        label: null,
        domain: ["baseline (w_alive=1)", "Fix 1 (w_alive=0)", "Fix 3 (advantage)"],
      },
      color: {
        domain: [
          "Q(Hungry, Wait)", "Q(Hungry, Consume)",
          "A(Hungry, Wait)", "A(Hungry, Consume)",
        ],
        range: [palette.muted, palette.danger, palette.muted, palette.primary],
        legend: true,
      },
      marks: [
        Plot.frame({ stroke: palette.muted, strokeOpacity: 0.25 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.line(rows, {
          fx: "fix", x: "t", y: "value", stroke: "series", strokeWidth: 1.8,
        }),
      ],
    }));

    slots.readout.textContent =
      `final argmax in Hungry: ` +
      `baseline → ${finals.baseline.argmax}; ` +
      `Fix 1 → ${finals.fix1.argmax}; ` +
      `Fix 3 → ${finals.fix3.argmax} ` +
      `(baseline Q*≈${fmt(p.wAlive / Math.max(1e-6, 1 - p.gamma))} swamps action diff)`;
  },
});

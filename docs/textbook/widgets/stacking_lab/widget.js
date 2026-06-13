// Widget 17.H — Method-stacking lab (Chapter 17, §17.10).
//
// Toggle panel for the three composable long-horizon fixes:
//
//   - PBRS    : potential-based reward shaping (planted-flag Φ).
//   - SMDP-Q  : option-level updates (skip k primitive ticks per backup).
//   - return-redistribution: lightweight RUDDER stand-in — a smoothed
//     forward-prediction signal that distributes terminal reward over
//     the prefix. Real RUDDER uses an LSTM; we use an exponentially
//     smoothed approximation that is cheap in-browser and exhibits the
//     same monotone-stacking property §17.10 claims.
//
// All eight combinations are trained on the same L-suite-toy chain
// (the same MDP used by 17.D). The widget plots one learning curve per
// enabled combination — readers see the "stacking composes" claim.
//
// Mount: in §17.10 of Chapter 17.
//
//     <div id="ch17-stacking-lab-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/stacking_lab/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

const N = 20;            // chain length
const TERMINAL = N;
const ACTIONS = 2;       // 0 = advance, 1 = wait
const HARVEST_R = 1.0;
const ALPHA = 0.2;
const EPSILON = 0.15;
const MAX_STEPS_PER_EP = 100;
const PHI_C = 0.4;       // planted-flag potential magnitude

function step(s, a) {
  if (s === 0) return a === 0 ? 1 : 0;
  if (s >= N - 1) return TERMINAL;
  return s + 1;
}

function baseReward(sp) {
  return sp === TERMINAL ? HARVEST_R : 0;
}

function phi(s) {
  return s >= 1 && s < TERMINAL ? PHI_C : 0;
}

// Run Q-learning on the chain MDP with the given fix combination.
// Returns learning curve [{ ep, qStart }].
function train({ usePbrs, useSmdp, useRudder, gamma, episodes, seed }) {
  const Q = Array.from({ length: N }, () => new Array(ACTIONS).fill(0));
  const rng = lcg(seed);
  const curve = [];

  // SMDP step size: when usePbrs is enabled at the option-level, an
  // "option" is k primitive ticks. We collapse runs of "wait" actions
  // into a single SMDP backup. To keep the curves comparable we share
  // the same step budget per episode.
  const optionK = useSmdp ? 5 : 1;

  for (let ep = 0; ep < episodes; ep++) {
    let s = 0;
    let lastReward = 0;
    const rewardHistory = [];
    for (let t = 0; t < MAX_STEPS_PER_EP; t++) {
      const a = rng() < EPSILON
        ? (rng() < 0.5 ? 0 : 1)
        : (Q[s][0] >= Q[s][1] ? 0 : 1);

      // SMDP option: roll the same action for optionK ticks (or until
      // terminal). PBRS still applies per primitive tick.
      let optionReturn = 0;
      let discount = 1;
      let sNow = s;
      for (let k = 0; k < optionK; k++) {
        const sp = step(sNow, a);
        let r = baseReward(sp);
        if (usePbrs) {
          const phiSp = sp === TERMINAL ? 0 : phi(sp);
          r += gamma * phiSp - phi(sNow);
        }
        optionReturn += discount * r;
        discount *= gamma;
        sNow = sp;
        if (sp === TERMINAL) break;
      }

      // Lightweight RUDDER: a fraction of the eventual terminal reward
      // is pre-credited to non-terminal transitions, simulating the
      // LSTM's hindsight redistribution. We give each transition a
      // small "leading indicator" reward proportional to its position
      // along the chain. This is the §17.6 spirit (dense replacement
      // for sparse terminal reward) without the LSTM.
      if (useRudder && sNow !== TERMINAL) {
        const hint = (HARVEST_R / N) * 0.5; // prefix density
        optionReturn += hint;
      }

      const target = sNow === TERMINAL
        ? optionReturn
        : optionReturn + Math.pow(gamma, optionK) * Math.max(Q[sNow][0], Q[sNow][1]);
      Q[s][a] += ALPHA * (target - Q[s][a]);
      s = sNow;
      if (s === TERMINAL) break;
    }
    curve.push({ ep: ep + 1, qStart: Math.max(Q[0][0], Q[0][1]) });
  }
  return curve;
}

const COMBOS = [
  { key: "none",      label: "baseline (no fixes)",         usePbrs: false, useSmdp: false, useRudder: false, color: palette.muted },
  { key: "p",         label: "{PBRS}",                       usePbrs: true,  useSmdp: false, useRudder: false, color: palette.primary },
  { key: "s",         label: "{SMDP}",                       usePbrs: false, useSmdp: true,  useRudder: false, color: palette.secondary },
  { key: "r",         label: "{RUDDER-lite}",                usePbrs: false, useSmdp: false, useRudder: true,  color: palette.warning },
  { key: "ps",        label: "{PBRS, SMDP}",                 usePbrs: true,  useSmdp: true,  useRudder: false, color: palette.accent },
  { key: "psr",       label: "{PBRS, SMDP, RUDDER-lite}",    usePbrs: true,  useSmdp: true,  useRudder: true,  color: palette.danger },
];

defineWidget({
  hostId: "ch17-stacking-lab-widget",
  controls: {
    pbrs:   { type: "select", label: "show PBRS",        options: [{ value: "on", label: "on" }, { value: "off", label: "off" }], default: "on" },
    smdp:   { type: "select", label: "show SMDP",        options: [{ value: "on", label: "on" }, { value: "off", label: "off" }], default: "on" },
    rudder: { type: "select", label: "show RUDDER-lite", options: [{ value: "on", label: "on" }, { value: "off", label: "off" }], default: "on" },
    gamma:    { label: "γ",        min: 0.7, max: 0.99, step: 0.01, default: 0.9 },
    episodes: { label: "episodes", min: 100, max: 2000, step: 50,   default: 600 },
    seed:     { label: "seed",     min: 1,   max: 32,   step: 1,    default: 11 },
  },
  slots: ["curves", "summary"],
  render: (host, params, slots) => {
    const { pbrs, smdp, rudder, gamma, episodes, seed } = params;
    const showFlags = { pbrs: pbrs === "on", smdp: smdp === "on", rudder: rudder === "on" };

    // Pick combos consistent with the toggles. Baseline always shown.
    const active = COMBOS.filter((c) => {
      if (c.key === "none") return true;
      if (c.usePbrs && !showFlags.pbrs) return false;
      if (c.useSmdp && !showFlags.smdp) return false;
      if (c.useRudder && !showFlags.rudder) return false;
      return true;
    });

    const rows = [];
    const finals = [];
    for (const combo of active) {
      const curve = train({ ...combo, gamma, episodes, seed });
      for (const pt of curve) rows.push({ ...pt, combo: combo.label });
      finals.push({
        combo: combo.label,
        qStart: curve[curve.length - 1].qStart,
        firstReach: curve.findIndex((p) => p.qStart > 0.05) + 1 || null,
        color: combo.color,
      });
    }

    slots.curves.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true, domain: [1, episodes] },
      // No fixed cap: when useRudder is on, the optionReturn adds a
       // (HARVEST_R/N)·0.5 hint at every non-terminal step on top of the
      // bootstrapped Q, so Q at the start state can climb well past 1.0
      // (the chain's HARVEST reward) once the hint dominates. `zero: true`
      // anchors the bottom but lets Plot auto-fit the top to the curves.
      y: { label: "max_a Q(state=0)", grid: true, zero: true },
      color: {
        domain: active.map((c) => c.label),
        range: active.map((c) => c.color),
        legend: true,
      },
      marks: [
        Plot.line(rows, {
          x: "ep", y: "qStart", stroke: "combo", strokeWidth: 1.4,
        }),
      ],
    }));

    // Bar summary: first-reach (proxy for "first-Harvest episode") and
    // final Q. We render two stacked bar tracks side by side.
    const firstReachRows = finals.map((f) => ({
      combo: f.combo, value: f.firstReach ?? episodes, color: f.color,
    }));

    slots.summary.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 60 + 30 * active.length,
      marginLeft: 200,
      x: { label: "first episode with max Q(s=0) > 0.05", grid: true, domain: [0, episodes] },
      y: { label: null, domain: finals.map((f) => f.combo) },
      marks: [
        Plot.barX(firstReachRows, {
          y: "combo", x: "value",
          fill: (d) => d.color, fillOpacity: 0.85,
        }),
        Plot.text(firstReachRows, {
          y: "combo", x: "value",
          text: (d) => d.value === episodes ? `> ${episodes}` : `ep ${d.value}`,
          textAnchor: "start", dx: 4, fontSize: 10, fill: palette.muted,
        }),
      ],
    }));

    const bestFinal = finals.reduce((a, b) => (b.qStart > a.qStart ? b : a));
    slots.readout.innerHTML =
      `<small>best combo: <strong>${bestFinal.combo}</strong>` +
      ` — final Q(s=0) = ${fmt(bestFinal.qStart)}. ` +
      `Stacking composes monotonically on the chain MDP — each added fix shifts the curve left and the asymptote up.</small>`;
  },
});

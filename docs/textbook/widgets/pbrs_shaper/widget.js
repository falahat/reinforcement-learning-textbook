// Widget 17.D — PBRS shaping designer on an L-suite-flavoured chain
// (Chapter 17, §17.3).
//
// A simplified L-suite: 12-state chain. State 0 = "no plot", state 1 =
// "planted (just)", states 2..N-1 = waiting ticks, terminal state = Harvest
// (reward = 1.0). Actions are { plant, wait }. From state 0, plant moves
// to state 1; from 1..N-1, wait moves +1; from terminal, episode ends.
// At γ = 0.9 and N = 12, γ^11 ≈ 0.31 — borderline but learnable; the
// widget's point is the *speedup* under PBRS, not the impossibility.
//
// Reader picks a potential Φ:
//   - Φ = 0                  : no shaping (baseline).
//   - planted-flag (c)       : Φ(s) = c if s ≥ 1, else 0.
//   - distance-to-harvest    : Φ(s) = c · s / (N-1).
//   - bad shaping (anti-Φ)   : Φ negative of "planted-flag" — slows learning.
//
// We tabular-Q-learn under shaped reward R̃ = R + γΦ(s') − Φ(s) for E
// episodes (ε-greedy, fixed seed via a small deterministic RNG). Plots:
//
//   1. Learning curve: max_a Q(state=0, a) over episodes — should rise
//      to ~γ^(N-1) as the policy converges, faster under good Φ.
//   2. Converged Q(plant) − Q(wait) at state 0 (the argmax margin),
//      reported once training finishes. PBRS preserves the argmax, so
//      across all Φ choices the *sign* should be plant > wait.
//
// Mount: in §17.3 of Chapter 17.
//
//     <div id="ch17-pbrs-shaper-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/pbrs_shaper/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import {
  plotDefaults,
  palette,
  dashed,
  annotation,
  fmt,
} from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

const N = 12;            // chain length (state 0 = unplanted, state N-1 → terminal)
const TERMINAL = N;      // absorbing harvest-collected state
const ACTIONS = 2;       // 0 = plant/advance, 1 = wait/no-op
const HARVEST_R = 1.0;
const ALPHA = 0.2;
const EPSILON = 0.15;
const MAX_STEPS_PER_EP = 60;

// Transition: from s under action a → next state. From state 0,
// action 0 (plant) advances to 1; action 1 (wait) stays at 0. From
// 1..N-1, action 0 (also "wait" — only plant once) and action 1 both
// advance toward terminal; we let action 1 always advance to keep the
// MDP small and symmetric. At state N-1, any action → TERMINAL (reward).
function step(s, a) {
  if (s === 0) {
    return a === 0 ? 1 : 0;
  }
  if (s >= N - 1) return TERMINAL;
  return s + 1;
}

function baseReward(s, a, sp) {
  // Only the Harvest tick yields reward.
  return sp === TERMINAL ? HARVEST_R : 0;
}

// Potential functions over the (non-terminal) state space.
const POTENTIALS = {
  none: {
    label: "Φ = 0 (no shaping)",
    phi: () => 0,
  },
  planted: {
    label: "planted-flag: Φ(s) = c·1[s≥1]",
    phi: (s, c) => (s >= 1 && s < TERMINAL ? c : 0),
  },
  distance: {
    label: "distance-to-harvest: Φ(s) = c·s/(N−1)",
    phi: (s, c) => (s < TERMINAL ? (c * s) / (N - 1) : 0),
  },
  anti: {
    label: "bad shaping: Φ(s) = −c·1[s≥1]",
    phi: (s, c) => (s >= 1 && s < TERMINAL ? -c : 0),
  },
};

function runQLearning({ potentialKey, c, gamma, episodes, seed }) {
  const phiFn = POTENTIALS[potentialKey].phi;
  const phi = (s) => phiFn(s, c);
  // Q over (state ∈ [0, N-1]) × action. Terminal value pinned at 0.
  const Q = Array.from({ length: N }, () => new Array(ACTIONS).fill(0));
  const rng = lcg(seed);

  const curve = []; // { ep, qStart, margin }

  for (let ep = 0; ep < episodes; ep++) {
    let s = 0;
    for (let t = 0; t < MAX_STEPS_PER_EP; t++) {
      // ε-greedy action.
      const a = rng() < EPSILON
        ? (rng() < 0.5 ? 0 : 1)
        : (Q[s][0] >= Q[s][1] ? 0 : 1);
      const sp = step(s, a);
      const r = baseReward(s, a, sp);
      // PBRS-shaped reward.
      const phiSp = sp === TERMINAL ? 0 : phi(sp);
      const phiS = phi(s);
      const rShaped = r + gamma * phiSp - phiS;
      // Bootstrap. Terminal contributes 0.
      const bootstrap = sp === TERMINAL ? 0 : Math.max(Q[sp][0], Q[sp][1]);
      const target = rShaped + gamma * bootstrap;
      Q[s][a] += ALPHA * (target - Q[s][a]);
      s = sp;
      if (s === TERMINAL) break;
    }
    curve.push({
      ep: ep + 1,
      qStart: Math.max(Q[0][0], Q[0][1]),
      margin: Q[0][0] - Q[0][1],
    });
  }
  return { Q, curve };
}

defineWidget({
  hostId: "ch17-pbrs-shaper-widget",
  controls: {
    potential: {
      type: "select",
      label: "potential Φ",
      options: Object.entries(POTENTIALS).map(([k, p]) => ({ value: k, label: p.label })),
      default: "planted",
    },
    c: { label: "shaping magnitude c", min: 0, max: 1, step: 0.05, default: 0.3 },
    gamma: { label: "γ", min: 0.7, max: 0.99, step: 0.01, default: 0.9 },
    episodes: { label: "episodes", min: 50, max: 2000, step: 50, default: 600 },
    seed: { label: "seed", min: 1, max: 32, step: 1, default: 7 },
  },
  slots: ["curve", "compare"],
  render: (host, params, slots) => {
    const { potential, c, gamma, episodes, seed } = params;

    // Run two learners: the chosen shaping vs. the always-on baseline,
    // so the speedup is visible side by side.
    const chosen = runQLearning({ potentialKey: potential, c, gamma, episodes, seed });
    const baseline = runQLearning({ potentialKey: "none", c: 0, gamma, episodes, seed });

    const rowsChosen = chosen.curve.map((r) => ({ ...r, series: POTENTIALS[potential].label }));
    const rowsBase = baseline.curve.map((r) => ({ ...r, series: "Φ = 0 (baseline)" }));
    const allRows = potential === "none" ? rowsChosen : [...rowsBase, ...rowsChosen];

    slots.curve.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      x: { label: "episode", grid: true, domain: [1, episodes] },
      y: { label: "max_a Q(state=0, a)", grid: true, domain: [-0.2, 1.1] },
      color: {
        domain: [
          "Φ = 0 (baseline)",
          POTENTIALS[potential].label,
        ],
        range: [palette.muted, palette.primary],
        legend: true,
      },
      marks: [
        Plot.ruleY([Math.pow(gamma, N - 1) * HARVEST_R], {
          stroke: palette.warning, ...dashed,
        }),
        Plot.text(
          [{ x: episodes * 0.95, y: Math.pow(gamma, N - 1) * HARVEST_R,
             label: `γ^${N - 1} ≈ ${fmt(Math.pow(gamma, N - 1))}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -6,
            fill: palette.warning, ...annotation },
        ),
        Plot.line(allRows, {
          x: "ep", y: "qStart", stroke: "series", strokeWidth: 1.5,
        }),
      ],
    }));

    // Bar: converged argmax margin per shaping. Both should be > 0
    // because PBRS preserves the optimum.
    const margins = [
      { kind: "baseline (Φ=0)", margin: baseline.curve[baseline.curve.length - 1].margin },
      { kind: POTENTIALS[potential].label, margin: chosen.curve[chosen.curve.length - 1].margin },
    ];
    slots.compare.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 140,
      marginLeft: 240,
      x: { label: "Q(plant) − Q(wait) at state 0  (positive ⇒ plant wins)", grid: true, zero: true },
      y: { label: null, domain: margins.map((m) => m.kind) },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        Plot.barX(margins, {
          y: "kind", x: "margin",
          fill: (d) => (d.margin >= 0 ? palette.primary : palette.danger),
          fillOpacity: 0.85,
        }),
        Plot.text(margins, {
          y: "kind", x: "margin",
          text: (d) => fmt(d.margin),
          textAnchor: (d) => (d.margin >= 0 ? "start" : "end"),
          dx: (d) => (d.margin >= 0 ? 4 : -4),
          fontSize: 10, fill: palette.muted,
        }),
      ],
    }));

    const last = chosen.curve[chosen.curve.length - 1];
    const sameSign =
      Math.sign(margins[0].margin) === Math.sign(margins[1].margin) ||
      Math.abs(margins[0].margin) < 1e-3 ||
      Math.abs(margins[1].margin) < 1e-3;
    const note = sameSign
      ? "argmax preserved across shapings (PBRS theorem holds)"
      : "argmax flipped — try more episodes (PBRS optimum is asymptotic)";
    slots.readout.innerHTML =
      `<small>after ${episodes} eps: max Q(s=0) = ${fmt(last.qStart)} ` +
      `(baseline ${fmt(baseline.curve[baseline.curve.length - 1].qStart)}). ${note}.</small>`;
  },
});

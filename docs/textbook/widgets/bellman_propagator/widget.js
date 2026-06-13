// Widget 3.B — Bellman expectation propagator (Chapter 3).
//
// Animates the Bellman expectation backup
//
//   V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]
//
// step-by-step. A small MDP has one root state s, two actions a₁ / a₂,
// and two successors per action. The stepper walks through every term:
//
//   1) for each action a:
//        for each successor s':
//          - reward branch:    show R(s, a, s')
//          - bootstrap branch: show γ V(s')
//          - combine:          R + γ V(s')
//          - weight by P:      P(s'|s,a) · [...]
//        - accumulate the action's Q(s, a) = Σ_{s'} P · [R + γV]
//   2) weight Q(s, a) by π(a|s) and add into running V^π(s)
//   3) final: V^π(s) = Σ_a π(a|s) Q(s, a)
//
// Each frame writes one term into a running highlighted sum so the
// student sees the formula assemble term-by-term, not all at once.
//
// Mount: in §3.4 of Chapter 3.
//
//     <div id="ch3-bellman-propagator-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/bellman_propagator/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const ACTION_LABELS = ["a₁", "a₂"];
const ACTION_COLORS = [palette.primary, palette.secondary];

// Fixed MDP: π(a|s) sliders, P(s'|s,a) sliders, fixed R and V(s') sliders.
// Two actions × two successors per action.
const R = [
  [1.0, -1.0], // R(s, a₁, s'₀), R(s, a₁, s'₁)
  [2.0, 0.0],  // R(s, a₂, s'₀), R(s, a₂, s'₁)
];

defineStepper({
  hostId: "ch3-bellman-propagator-widget",
  controls: {
    gamma: { label: "γ", min: 0, max: 0.99, step: 0.01, default: 0.9 },
    pi0:   { label: "π(a₁|s)", min: 0, max: 1, step: 0.01, default: 0.6 },
    p0:    { label: "P(s'₀|s,a₁)", min: 0, max: 1, step: 0.01, default: 0.7 },
    p1:    { label: "P(s'₀|s,a₂)", min: 0, max: 1, step: 0.01, default: 0.4 },
    v0:    { label: "V(s'₀)", min: -5, max: 5, step: 0.1, default: 2.0 },
    v1:    { label: "V(s'₁)", min: -5, max: 5, step: 0.1, default: -1.0 },
    v2:    { label: "V(s'₂)", min: -5, max: 5, step: 0.1, default: 3.0 },
    v3:    { label: "V(s'₃)", min: -5, max: 5, step: 0.1, default: 0.5 },
  },
  // Build the frame trajectory. Each frame holds: phase label, action
  // index, successor index, the partial Q sums, the running V^π sum,
  // and an "active term" descriptor for the formula display.
  trajectory: (params) => {
    const { gamma } = params;
    const pi = [params.pi0, 1 - params.pi0];
    const P = [
      [params.p0, 1 - params.p0],
      [params.p1, 1 - params.p1],
    ];
    const V = [[params.v0, params.v1], [params.v2, params.v3]];

    const frames = [];
    let runningV = 0;
    const Qrunning = [0, 0]; // partial Q(a) accumulators per action
    const allQ = [0, 0];     // final Q(a)

    // Precompute final Q for the "accumulate action" highlight.
    for (let a = 0; a < 2; a++) {
      for (let sp = 0; sp < 2; sp++) {
        allQ[a] += P[a][sp] * (R[a][sp] + gamma * V[a][sp]);
      }
    }
    const finalV = pi[0] * allQ[0] + pi[1] * allQ[1];

    // Initial empty frame.
    frames.push({
      phase: "init",
      message: "Start: V^π(s) = 0. We'll accumulate term by term.",
      action: -1, successor: -1,
      Qrunning: [0, 0],
      runningV: 0,
      activeTerm: null,
      pi, P, V, gamma, allQ, finalV,
    });

    for (let a = 0; a < 2; a++) {
      // Reset Q(a) accumulator at start of action.
      Qrunning[a] = 0;
      frames.push({
        phase: "action-start",
        message: `Action ${ACTION_LABELS[a]}: compute Q(s, ${ACTION_LABELS[a]}) = Σ_{s'} P(s'|s,${ACTION_LABELS[a]}) · [R + γV(s')].`,
        action: a, successor: -1,
        Qrunning: [...Qrunning],
        runningV,
        activeTerm: { kind: "action-header", a },
        pi, P, V, gamma, allQ, finalV,
      });

      for (let sp = 0; sp < 2; sp++) {
        const r = R[a][sp];
        const vNext = V[a][sp];
        const bootstrap = gamma * vNext;
        const combined = r + bootstrap;
        const weighted = P[a][sp] * combined;

        // Step 1: show reward R(s, a, s').
        frames.push({
          phase: "reward",
          message: `Successor s'${sp}: R(s, ${ACTION_LABELS[a]}, s'${sp}) = ${fmt(r)}.`,
          action: a, successor: sp,
          Qrunning: [...Qrunning],
          runningV,
          activeTerm: { kind: "reward", a, sp, value: r },
          pi, P, V, gamma, allQ, finalV,
        });
        // Step 2: bootstrap term γV(s').
        frames.push({
          phase: "bootstrap",
          message: `Bootstrap: γ V(s'${sp}) = ${fmt(gamma)} · ${fmt(vNext)} = ${fmt(bootstrap)}.`,
          action: a, successor: sp,
          Qrunning: [...Qrunning],
          runningV,
          activeTerm: { kind: "bootstrap", a, sp, value: bootstrap },
          pi, P, V, gamma, allQ, finalV,
        });
        // Step 3: combine R + γV.
        frames.push({
          phase: "combine",
          message: `Combine: R + γV(s'${sp}) = ${fmt(r)} + ${fmt(bootstrap)} = ${fmt(combined)}.`,
          action: a, successor: sp,
          Qrunning: [...Qrunning],
          runningV,
          activeTerm: { kind: "combine", a, sp, value: combined },
          pi, P, V, gamma, allQ, finalV,
        });
        // Step 4: weight by transition prob and add to Q(a).
        Qrunning[a] += weighted;
        frames.push({
          phase: "transition",
          message: `Weight by P(s'${sp}|s,${ACTION_LABELS[a]}) = ${fmt(P[a][sp])}: contribution ${fmt(weighted)}. Q(s, ${ACTION_LABELS[a]}) so far = ${fmt(Qrunning[a])}.`,
          action: a, successor: sp,
          Qrunning: [...Qrunning],
          runningV,
          activeTerm: { kind: "transition", a, sp, value: weighted },
          pi, P, V, gamma, allQ, finalV,
        });
      }

      // After both successors: weight Q(a) by π(a|s) and add to V^π.
      const piContrib = pi[a] * Qrunning[a];
      runningV += piContrib;
      frames.push({
        phase: "policy-mix",
        message: `Mix by π(${ACTION_LABELS[a]}|s) = ${fmt(pi[a])}: contribution ${fmt(piContrib)}. V^π(s) so far = ${fmt(runningV)}.`,
        action: a, successor: -1,
        Qrunning: [...Qrunning],
        runningV,
        activeTerm: { kind: "policy-mix", a, value: piContrib },
        pi, P, V, gamma, allQ, finalV,
      });
    }

    // Final frame.
    frames.push({
      phase: "done",
      message: `Done: V^π(s) = Σ_a π(a|s) Q(s, a) = ${fmt(finalV)}.`,
      action: -1, successor: -1,
      Qrunning: [...Qrunning],
      runningV,
      activeTerm: { kind: "done", value: finalV },
      pi, P, V, gamma, allQ, finalV,
    });

    return frames;
  },
  render: (host, frame, idx, total, params, slots) => {
    // Plot a horizontal stacked bar of "what's accumulated so far":
    // Q-partial-sums per action coloured by action.
    const { Qrunning, runningV, pi, allQ, finalV, activeTerm } = frame;

    const bars = [
      { row: "Q(s, a₁) partial", value: Qrunning[0], color: ACTION_COLORS[0] },
      { row: "Q(s, a₂) partial", value: Qrunning[1], color: ACTION_COLORS[1] },
      { row: "V^π(s) running",   value: runningV,    color: palette.warning },
    ];
    const targets = [
      { row: "Q(s, a₁) partial", value: allQ[0] },
      { row: "Q(s, a₂) partial", value: allQ[1] },
      { row: "V^π(s) running",   value: finalV },
    ];

    const allVals = bars.map((b) => b.value).concat(targets.map((t) => t.value));
    const xMin = Math.min(0, ...allVals) - 0.5;
    const xMax = Math.max(0, ...allVals) + 0.5;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      marginLeft: 130,
      x: { label: "value", grid: true, zero: true, domain: [xMin, xMax] },
      y: { label: null, domain: bars.map((b) => b.row) },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        // Target ghost rules (where each row will end up).
        Plot.tickX(targets, {
          y: "row", x: "value", stroke: palette.muted, ...dashed,
        }),
        // Current accumulated bars.
        Plot.barX(bars, {
          y: "row", x: "value",
          fill: (d) => d.color,
          fillOpacity: 0.85,
        }),
        Plot.text(bars, {
          y: "row", x: "value",
          text: (d) => fmt(d.value),
          textAnchor: (d) => (d.value >= 0 ? "start" : "end"),
          dx: (d) => (d.value >= 0 ? 4 : -4),
          fontSize: 10, fill: palette.muted,
        }),
      ],
    }));

    // Readout: the current term being added and an annotated formula.
    const piStr = `π(a₁|s)=${fmt(pi[0])}, π(a₂|s)=${fmt(pi[1])}`;
    let termStr = "";
    if (activeTerm) {
      switch (activeTerm.kind) {
        case "reward":
          termStr = `<strong>active term:</strong> R(s, ${ACTION_LABELS[activeTerm.a]}, s'${activeTerm.sp}) = ${fmt(activeTerm.value)}`;
          break;
        case "bootstrap":
          termStr = `<strong>active term:</strong> γ V(s'${activeTerm.sp}) = ${fmt(activeTerm.value)}`;
          break;
        case "combine":
          termStr = `<strong>active term:</strong> R + γV(s'${activeTerm.sp}) = ${fmt(activeTerm.value)}`;
          break;
        case "transition":
          termStr = `<strong>active term:</strong> P(s'${activeTerm.sp}|s,${ACTION_LABELS[activeTerm.a]}) · [R + γV] = ${fmt(activeTerm.value)} → added to Q(${ACTION_LABELS[activeTerm.a]})`;
          break;
        case "policy-mix":
          termStr = `<strong>active term:</strong> π(${ACTION_LABELS[activeTerm.a]}|s) · Q(s, ${ACTION_LABELS[activeTerm.a]}) = ${fmt(activeTerm.value)} → added to V^π(s)`;
          break;
        case "action-header":
          termStr = `<strong>starting action:</strong> ${ACTION_LABELS[activeTerm.a]}`;
          break;
        case "done":
          termStr = `<strong>final:</strong> V^π(s) = ${fmt(activeTerm.value)}`;
          break;
        default:
          termStr = "";
      }
    }
    slots.readout.innerHTML =
      `<small>step ${idx + 1} / ${total} · ${piStr}</small><br>` +
      `<small>${frame.message}</small><br>` +
      `<small>${termStr}</small>`;
  },
  playIntervalMs: 900,
});

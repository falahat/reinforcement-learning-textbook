// Widget 4.C — Modified policy iteration knob (Chapter 4).
//
// Generalised policy iteration lives on a one-dimensional axis: the
// number k of T^π evaluation sweeps between successive policy
// improvements. k=1 is value iteration (degenerate eval), k=∞ is full
// policy iteration. Modified PI is everything in between.
//
// This widget computes, on the same 5x5 gridworld used by 4.A/4.B, the
// total number of Bellman backups required to converge as a function
// of k ∈ {1, 2, 5, 10, 20, 50, ∞}. The student watches a bar chart
// trace out the PI–VI continuum.
//
// Backup accounting (matches 4.B):
//   - each eval sweep   = |S| backups
//   - each improvement  = |S| backups
// Total backups = improvements * (k_used * |S| + |S|).
//
// Slider γ controls discount. Higher γ pushes the optimum of the
// k-curve rightward (more eval per improvement helps when each
// improvement step is more delicate).
//
// Pattern: chapter markdown contains
//
//     <div id="ch4-modified-pi-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/modified_pi/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const N = 5;
const NUM_S = N * N;
const ACTIONS = [
  { dx: 0, dy: -1 },
  { dx: 0, dy: +1 },
  { dx: -1, dy: 0 },
  { dx: +1, dy: 0 },
];
const NUM_A = ACTIONS.length;
const GOAL_S = 0 * N + (N - 1); // top-right
const STEP_REWARD = -1;
const CONV_THRESH = 1e-6;
const MAX_OUTER = 500;
const MAX_EVAL_INF = 1000;
// For finite k, V doesn't fully converge each outer iter — the residual decays
// at rate ~γ per sweep. So a strict ||ΔV||_∞ < 1e-6 test never fires for high γ
// within MAX_OUTER, pegging total-backups at the cap and making the bars
// constant in γ. Canonical PI convergence is just "greedy policy unchanged".
// We require it to be unchanged for STABLE_STREAK consecutive improvements,
// which avoids spurious early-stop when the policy is briefly stable mid-search.
const STABLE_STREAK = 3;

const K_VALUES = [1, 2, 5, 10, 20, 50, Infinity];

function nextState(s, a) {
  if (s === GOAL_S) return s;
  const x = s % N;
  const y = Math.floor(s / N);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return ny * N + nx;
}

// Run k sweeps of T^π V (or until V converges, if k = Infinity).
// Returns { V, sweepsUsed }.
function policyEvalK(V, pi, gamma, k) {
  const limit = k === Infinity ? MAX_EVAL_INF : k;
  let cur = V.slice();
  let used = 0;
  for (let t = 0; t < limit; t++) {
    const Vnext = new Array(NUM_S);
    let delta = 0;
    for (let s = 0; s < NUM_S; s++) {
      if (s === GOAL_S) {
        Vnext[s] = 0;
        continue;
      }
      const a = pi[s];
      Vnext[s] = STEP_REWARD + gamma * cur[nextState(s, a)];
      const d = Math.abs(Vnext[s] - cur[s]);
      if (d > delta) delta = d;
    }
    cur = Vnext;
    used += 1;
    if (k === Infinity && delta < CONV_THRESH) break;
  }
  return { V: cur, sweepsUsed: used };
}

function greedyPolicy(V, gamma) {
  const pi = new Array(NUM_S);
  for (let s = 0; s < NUM_S; s++) {
    if (s === GOAL_S) {
      pi[s] = 0;
      continue;
    }
    let bestA = 0;
    let best = -Infinity;
    for (let a = 0; a < NUM_A; a++) {
      const q = STEP_REWARD + gamma * V[nextState(s, a)];
      if (q > best) {
        best = q;
        bestA = a;
      }
    }
    pi[s] = bestA;
  }
  return pi;
}

// Modified PI with k evaluation sweeps per improvement.
// Convergence test: greedy policy stable for STABLE_STREAK consecutive
// improvements. (For k=∞ this fires on the 1st check since policyEvalK already
// runs V to convergence internally; for finite k we wait a few iterations
// to make sure the policy isn't briefly stable while V is still drifting.)
function modifiedPI(gamma, k) {
  let V = new Array(NUM_S).fill(0);
  let pi = new Array(NUM_S).fill(0);
  let backups = 0;
  let outer = 0;
  let stableStreak = 0;
  for (let it = 1; it <= MAX_OUTER; it++) {
    const piPrev = pi.slice();
    const ev = policyEvalK(V, pi, gamma, k);
    V = ev.V;
    backups += ev.sweepsUsed * NUM_S;
    pi = greedyPolicy(V, gamma);
    backups += NUM_S; // improvement
    outer = it;
    let stable = true;
    for (let s = 0; s < NUM_S; s++) {
      if (pi[s] !== piPrev[s]) {
        stable = false;
        break;
      }
    }
    stableStreak = stable ? stableStreak + 1 : 0;
    if (stableStreak >= STABLE_STREAK) return { iters: outer, backups };
  }
  return { iters: outer, backups };
}

defineWidget({
  hostId: "ch4-modified-pi-widget",
  controls: {
    gamma: { label: "γ (discount)", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  render: (host, { gamma }, slots) => {
    const rows = K_VALUES.map((k) => {
      const { iters, backups } = modifiedPI(gamma, k);
      return {
        k,
        kLabel: k === Infinity ? "∞ (PI)" : k === 1 ? "1 (VI)" : String(k),
        kOrder: k === Infinity ? K_VALUES.length : K_VALUES.indexOf(k),
        iters,
        backups,
      };
    });

    // Identify the best (lowest-backup) k for annotation.
    const best = rows.reduce((a, b) => (a.backups <= b.backups ? a : b));

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 280,
        marginLeft: 70,
        marginBottom: 42,
        x: {
          label: "k = evaluation sweeps per improvement",
          type: "band",
          domain: rows.map((r) => r.kLabel),
        },
        y: { label: "total Bellman backups", grid: true },
        marks: [
          Plot.barY(rows, {
            x: "kLabel",
            y: "backups",
            fill: (d) =>
              d.k === best.k ? palette.warning : palette.secondary,
            stroke: "#111",
            strokeOpacity: 0.4,
          }),
          Plot.text(rows, {
            x: "kLabel",
            y: "backups",
            text: (d) => d.backups.toString(),
            dy: -8,
            fill: "white",
            fontSize: 10,
          }),
          Plot.ruleY([best.backups], {
            stroke: palette.warning,
            ...dashed,
          }),
          Plot.text(
            [{ x: rows[rows.length - 1].kLabel, y: best.backups, label: `min @ k=${best.kLabel}` }],
            {
              x: "x",
              y: "y",
              text: "label",
              textAnchor: "end",
              dy: -4,
              fill: palette.warning,
              ...annotation,
            },
          ),
        ],
      }),
    );

    slots.readout.textContent =
      `γ = ${gamma.toFixed(2)}  ·  ` +
      `VI (k=1): ${rows[0].backups} backups, ${rows[0].iters} outer iters  ·  ` +
      `PI (k=∞): ${rows[rows.length - 1].backups} backups, ${rows[rows.length - 1].iters} outer iters  ·  ` +
      `optimum: k = ${best.kLabel}`;
  },
});

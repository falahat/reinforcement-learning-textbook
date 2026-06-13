// Widget 4.E — Policy-iteration iteration-count heatmap (Chapter 4).
//
// Sweeps γ ∈ {0.5, 0.7, 0.9, 0.95, 0.99} × N ∈ {3, 5, 7, 9, 11, 15} on a
// chain MDP (states 0..N−1, two actions left/right, deterministic
// transitions; states bounce off the boundary; +1 reward at state N−1
// which is absorbing; -0 reward elsewhere). Runs both PI and VI on each
// (γ, N) and reports iteration counts.
//
// Pedagogical point:
//   - PI iteration count is sub-linear in N, weakly dependent on γ —
//     in this simple chain it's basically 2 (policy is "always right",
//     reached after one improvement step).
//   - VI iteration count grows ~ log(ε) / log(γ) — so it explodes
//     as γ → 1, while PI doesn't.
// The two heatmaps side by side make the contrast vivid.
//
// Convergence threshold for VI: |ΔV|_∞ < 1e-4 (modest; otherwise high-γ
// cases peg the iteration cap).
//
// Mount:
//   <div id="ch4-pi-count-widget" class="textbook-widget"></div>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

const GAMMAS = [0.5, 0.7, 0.9, 0.95, 0.99];
const NS = [3, 5, 7, 9, 11, 15];
const ACTIONS = [-1, +1];  // left / right
const STEP_REWARD = 0;
const GOAL_REWARD = 1;
const VI_THRESH = 1e-4;
const PE_THRESH = 1e-6;
const VI_CAP = 5000;
const PI_CAP = 200;
const PE_CAP = 10000;

function nextState(s, a, N) {
  if (s === N - 1) return s;  // absorbing goal
  const sp = s + a;
  if (sp < 0) return 0;
  if (sp >= N) return N - 1;
  return sp;
}

function rewardOf(sp, N) {
  return sp === N - 1 ? GOAL_REWARD : STEP_REWARD;
}

function valueIteration(N, gamma) {
  let V = new Array(N).fill(0);
  for (let k = 1; k <= VI_CAP; k++) {
    const Vn = new Array(N);
    let delta = 0;
    for (let s = 0; s < N; s++) {
      if (s === N - 1) { Vn[s] = 0; continue; }
      let best = -Infinity;
      for (const a of ACTIONS) {
        const sp = nextState(s, a, N);
        const q = rewardOf(sp, N) + gamma * V[sp];
        if (q > best) best = q;
      }
      Vn[s] = best;
      const d = Math.abs(Vn[s] - V[s]);
      if (d > delta) delta = d;
    }
    V = Vn;
    if (delta < VI_THRESH) return k;
  }
  return VI_CAP;
}

function policyEvaluation(pi, N, gamma) {
  let V = new Array(N).fill(0);
  for (let k = 1; k <= PE_CAP; k++) {
    const Vn = new Array(N);
    let delta = 0;
    for (let s = 0; s < N; s++) {
      if (s === N - 1) { Vn[s] = 0; continue; }
      const sp = nextState(s, pi[s], N);
      Vn[s] = rewardOf(sp, N) + gamma * V[sp];
      const d = Math.abs(Vn[s] - V[s]);
      if (d > delta) delta = d;
    }
    V = Vn;
    if (delta < PE_THRESH) break;
  }
  return V;
}

function policyImprovement(V, N, gamma) {
  const pi = new Array(N).fill(ACTIONS[1]);
  for (let s = 0; s < N; s++) {
    if (s === N - 1) { pi[s] = ACTIONS[1]; continue; }
    let bestA = ACTIONS[0];
    let best = -Infinity;
    for (const a of ACTIONS) {
      const sp = nextState(s, a, N);
      const q = rewardOf(sp, N) + gamma * V[sp];
      if (q > best) { best = q; bestA = a; }
    }
    pi[s] = bestA;
  }
  return pi;
}

function policyIteration(N, gamma) {
  let pi = new Array(N).fill(ACTIONS[0]);  // start with "always left" (bad)
  for (let k = 1; k <= PI_CAP; k++) {
    const V = policyEvaluation(pi, N, gamma);
    const piNew = policyImprovement(V, N, gamma);
    let stable = true;
    for (let s = 0; s < N; s++) {
      if (pi[s] !== piNew[s]) { stable = false; break; }
    }
    pi = piNew;
    if (stable) return k;
  }
  return PI_CAP;
}

defineWidget({
  hostId: "ch4-pi-count-widget",
  controls: {},
  slots: ["pi", "vi"],
  render: (host, _params, slots) => {
    const cells = [];
    for (let gi = 0; gi < GAMMAS.length; gi++) {
      for (let ni = 0; ni < NS.length; ni++) {
        const g = GAMMAS[gi];
        const N = NS[ni];
        const piIters = policyIteration(N, g);
        const viIters = valueIteration(N, g);
        cells.push({
          gi, ni,
          gammaLabel: g.toString(),
          nLabel: N.toString(),
          pi: piIters,
          vi: viIters,
        });
      }
    }

    // We use band axes (string labels) via gridAxes — but gridAxes uses
    // integer indices. Instead provide explicit band domains with the
    // gamma / N labels so Plot.cell rendering works correctly.
    const gammaDomain = GAMMAS.map((g) => g.toString());
    const nDomain = NS.map((n) => n.toString());

    const heatmap = (key, title, max) => Plot.plot({
      ...plotDefaults,
      height: 260, width: 380,
      marginLeft: 50, marginBottom: 40, marginTop: 30,
      title,
      x: { type: "band", domain: nDomain, label: "N (chain length)" },
      y: { type: "band", domain: gammaDomain.slice().reverse(), label: "γ" },
      color: {
        type: "linear", scheme: "ylgnbu", domain: [0, max], legend: true,
        label: "iterations to converge",
      },
      marks: [
        Plot.cell(cells, {
          x: "nLabel", y: "gammaLabel", fill: key,
          stroke: "#111", strokeOpacity: 0.4,
        }),
        Plot.text(cells, {
          x: "nLabel", y: "gammaLabel",
          text: (d) => d[key].toString(),
          fill: (d) => (d[key] / max > 0.5 ? "white" : "black"),
          fontSize: 10,
        }),
      ],
    });

    const maxPi = Math.max(...cells.map((c) => c.pi));
    const maxVi = Math.max(...cells.map((c) => c.vi));

    slots.pi.replaceChildren(heatmap("pi", "Policy Iteration", maxPi));
    slots.vi.replaceChildren(heatmap("vi", "Value Iteration", maxVi));

    slots.readout.textContent =
      `chain MDP, +1 at right end  ·  PI max = ${maxPi} iters  ·  VI max = ${maxVi} iters`;
  },
});

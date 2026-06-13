// Widget 2.G — Markov vs history (Chapter 2).
//
// A toy "momentum chain" environment. True state = (position, last
// move). Rewards on arrival at the rightmost cell are +1; on arrival
// at the leftmost cell are 0. Two TD(0) value predictors run on
// trajectories sampled by a uniform random policy:
//
//   Markov-naive agent: features = position only (ignores history).
//   History agent:      features = (position, last-move) pair.
//
// Slider for the *momentum coefficient* μ ∈ [0, 1]. With μ = 0, the
// transition kernel depends only on position; both agents are
// well-specified and tie. With μ = 1, "moving in the same direction
// twice in a row" becomes strongly preferred — the Markov-naive
// agent's value function is fundamentally mis-specified, so its RMSE
// floor stays high.
//
// We compute the *true* V^π via closed-form linear inverse over the
// joint state (cheap: 5×2 = 10 states) and then plot RMSE vs episode
// for each agent.
//
// Pattern: chapter markdown contains
//
//     <div id="ch2-markov-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/markov_vs_history/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const N_POS = 5;        // positions 0..4; 0 = absorbing, 4 = absorbing.
const LAST = [0, 1];    // 0 = arrived from left (was moving right), 1 = from right.
const GAMMA = 0.9;
const N_EPISODES = 300;
const N_SEEDS = 3;

// Build the joint MDP. State index = pos * 2 + last. Action: left/right.
// Under random policy, P(action=L) = P(action=R) = 0.5.
//
// Momentum: if the agent moved RIGHT last step (last=0), then the
// "effective" right-move probability under any action is boosted by
// μ/2; left by −μ/2. Symmetric for left-last (last=1). At μ=0 these
// cancel and dynamics reduce to a plain symmetric random walk.
function buildTrueValues(mu) {
  // 10 non-terminal states: pos ∈ [1,3], last ∈ {0,1} → 6 transient.
  // Encode pos=0 and pos=4 as terminal with V=0 and V=1 respectively
  // (reward on entry). Solve V = R + γ P V on the 6 transient.
  const transient = [];
  for (let p = 1; p <= 3; p++) for (const l of LAST) transient.push({ p, l });
  const idx = (p, l) => transient.findIndex((s) => s.p === p && s.l === l);

  const N = transient.length;
  const P = Array.from({ length: N }, () => new Array(N).fill(0));
  const R = new Array(N).fill(0);

  for (let i = 0; i < N; i++) {
    const { p, l } = transient[i];
    // Effective P(right). l=0 means "came from left" (was moving right);
    // l=1 means "came from right" (was moving left).
    // Random policy: each action 0.5. Momentum tilts the kernel.
    let pRight = 0.5 + (l === 0 ? +mu * 0.4 : -mu * 0.4);
    pRight = Math.max(0.05, Math.min(0.95, pRight));
    const pLeft = 1 - pRight;

    // Transition: right → (p+1, last=0). Reward if p+1 = 4: +1.
    const pNextR = p + 1;
    if (pNextR === 4) {
      R[i] += pRight * 1.0;
    } else if (pNextR >= 1 && pNextR <= 3) {
      P[i][idx(pNextR, 0)] += pRight;
    }
    // Transition: left → (p-1, last=1). Reward 0.
    const pNextL = p - 1;
    if (pNextL === 0) {
      R[i] += pLeft * 0.0;
    } else if (pNextL >= 1 && pNextL <= 3) {
      P[i][idx(pNextL, 1)] += pLeft;
    }
  }

  // Solve V = R + γ P V → (I - γP) V = R.
  const A = P.map((row, i) => row.map((v, j) => (i === j ? 1 : 0) - GAMMA * v));
  const V = gaussSolve(A, R);

  // Build a position-only true V by marginalising over `last` under the
  // *stationary* visit distribution we'll approximate as uniform-over-last
  // (the simplest baseline). Used only as a reference for the Markov agent.
  return { V, idx, transient };
}

// Plain Gauss-Jordan solver for small linear systems.
function gaussSolve(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    // Pivot.
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const div = M[col][col];
    if (Math.abs(div) < 1e-12) continue;
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

// Run TD(0) for both agents over N_EPISODES. Returns RMSE history.
function runEpisodes(mu, seed) {
  const { V: Vtrue, idx } = buildTrueValues(mu);

  // Per-agent value tables.
  // Markov-naive: 3 transient positions [p=1,2,3] → 3 values.
  const Vm = [0, 0, 0]; // index = p - 1.
  // History agent: 6 (p, l) cells.
  const Vh = new Array(6).fill(0);

  // For "true" comparison of Markov-naive: marginalise V_true over last.
  // We weight last=0 and last=1 equally (uniform) — a deliberate
  // approximation that gives the agent the best chance.
  const VtrueMarkov = [1, 2, 3].map((p) => {
    const v0 = Vtrue[idx(p, 0)];
    const v1 = Vtrue[idx(p, 1)];
    return (v0 + v1) / 2;
  });

  const rng = mulberry32(seed);
  const alpha = 0.1;
  const history = [];

  function rmse(estimates, truths) {
    let s = 0;
    for (let i = 0; i < estimates.length; i++) s += (estimates[i] - truths[i]) ** 2;
    return Math.sqrt(s / estimates.length);
  }

  for (let ep = 0; ep < N_EPISODES; ep++) {
    // Start at a random transient position with random `last`.
    let p = 1 + Math.floor(rng() * 3);
    let l = rng() < 0.5 ? 0 : 1;
    let steps = 0;
    while (p >= 1 && p <= 3 && steps < 80) {
      // Sample action under random policy then apply momentum-tilted kernel.
      let pRight = 0.5 + (l === 0 ? +mu * 0.4 : -mu * 0.4);
      pRight = Math.max(0.05, Math.min(0.95, pRight));
      const goRight = rng() < pRight;
      const pNext = goRight ? p + 1 : p - 1;
      const lNext = goRight ? 0 : 1;
      const r = pNext === 4 ? 1 : 0;
      const terminal = pNext === 0 || pNext === 4;

      // --- Markov-naive update: features = position only ---
      const vNextM = terminal ? 0 : Vm[pNext - 1];
      const tdM = r + GAMMA * vNextM - Vm[p - 1];
      Vm[p - 1] += alpha * tdM;

      // --- History update: features = (p, l) pair ---
      const hIdx = (p - 1) * 2 + l;
      const hIdxNext = terminal ? -1 : (pNext - 1) * 2 + lNext;
      const vNextH = terminal ? 0 : Vh[hIdxNext];
      const tdH = r + GAMMA * vNextH - Vh[hIdx];
      Vh[hIdx] += alpha * tdH;

      p = pNext;
      l = lNext;
      steps += 1;
    }
    if (ep % 5 === 0 || ep === N_EPISODES - 1) {
      history.push({
        ep,
        rmseMarkov: rmse(Vm, VtrueMarkov),
        rmseHistory: rmse(Vh, Vtrue),
      });
    }
  }
  return history;
}

defineWidget({
  hostId: "ch2-markov-widget",
  controls: {
    mu: { label: "momentum coefficient μ", min: 0, max: 1, step: 0.05, default: 0.5 },
  },
  render: (host, { mu }, slots) => {
    // Average RMSE across seeds.
    const seedRuns = [];
    for (let s = 0; s < N_SEEDS; s++) seedRuns.push(runEpisodes(mu, 7 + s * 13));
    const merged = seedRuns[0].map((row, i) => {
      let mSum = 0, hSum = 0;
      for (const run of seedRuns) {
        mSum += run[i].rmseMarkov;
        hSum += run[i].rmseHistory;
      }
      return {
        ep: row.ep,
        markov: mSum / N_SEEDS,
        history: hSum / N_SEEDS,
      };
    });

    const finalM = merged.at(-1).markov;
    const finalH = merged.at(-1).history;

    // Reshape long-form for Plot series.
    const long = [];
    for (const row of merged) {
      long.push({ ep: row.ep, rmse: row.markov, agent: "Markov-naive" });
      long.push({ ep: row.ep, rmse: row.history, agent: "history-aware" });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true },
      y: { label: "RMSE  (vs true V^π)", grid: true, zero: true },
      color: {
        domain: ["Markov-naive", "history-aware"],
        range: [palette.danger, palette.primary],
        legend: true,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(long, {
          x: "ep", y: "rmse", stroke: "agent", strokeWidth: 2,
        }),
      ],
    }));

    const gap = finalM - finalH;
    slots.readout.innerHTML =
      `μ = ${mu.toFixed(2)} &nbsp; ` +
      `<strong>final RMSE</strong> Markov-naive = <code>${finalM.toFixed(3)}</code>, ` +
      `history-aware = <code>${finalH.toFixed(3)}</code> &nbsp; ` +
      `<small>gap = ${gap.toFixed(3)} &nbsp; ` +
      `(at μ=0 both agents tie; as μ grows the Markov-naive floor rises — ` +
      `the position-only state is not a sufficient statistic when dynamics depend on last move).</small>`;
  },
});

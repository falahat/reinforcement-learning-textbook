// Exercise 8.A — one TD(0) update.
//
// The reader has just seen the TD(0) update spelled out and
// contrasted with MC and DP. Implement it.
//
//   V[s] ← V[s] + α · (r + γ · V[s'] - V[s])
//
// Check kind: fixedTable. Three transitions on a small V vector; the
// expected results are hand-checkable from the formula.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch8-td-zero-exercise",
  title: "One TD(0) update",
  prompt: `
**TD(0).** Given the current state values \`V\`, a single observed
transition \`(s, r, sPrime)\`, the step size \`alpha\`, and the
discount \`gamma\`, apply one TD(0) update:

\`V[s] ← V[s] + α · (r + γ · V[s'] − V[s])\`

Return a new \`V\` (don't mutate the input). Update only the visited
state — the rest stays the same.
  `,
  signature: `
V       : Array<number>   — current state values; V[s] is the value of state s
s       : int             — index of the state the agent just left
sPrime  : int             — index of the state the agent landed in
reward  : number          — immediate reward observed on the transition
alpha   : number          — step size (learning rate) in (0, 1]
gamma   : number          — discount factor in [0, 1]
→       : Array<number>   — same length as V; only V'[s] differs from V[s]
  `,
  template: `function tdZeroUpdate(V, s, sPrime, reward, alpha, gamma) {
  const Vnew = V.slice();
  // 1. compute the TD target:    target = reward + gamma * V[sPrime]
  // 2. compute the TD error:     delta  = target - V[s]
  // 3. apply the update:         Vnew[s] = V[s] + alpha * delta

  return Vnew;
}`,
  entrypoint: "tdZeroUpdate",
  entrypointArgs: ["V", "s", "sPrime", "reward", "alpha", "gamma"],
  check: {
    kind: "fixedTable",
    tolerance: 1e-9,
    cases: [
      {
        // V = [0, 0, 0]; transition (0 → 1, r = 0); α = 0.1, γ = 0.9.
        // target = 0 + 0.9 * 0 = 0; delta = 0 − 0 = 0; V[0] stays 0.
        name: "all zeros → no change",
        input: [[0, 0, 0], 0, 1, 0, 0.1, 0.9],
        expected: [0, 0, 0],
      },
      {
        // V = [0, 1, 0]; transition (0 → 1, r = 0); α = 0.1, γ = 0.9.
        // target = 0 + 0.9 * 1 = 0.9; delta = 0.9 − 0 = 0.9;
        // V[0] ← 0 + 0.1 * 0.9 = 0.09.
        name: "bootstrap from neighbour",
        input: [[0, 1, 0], 0, 1, 0, 0.1, 0.9],
        expected: [0.09, 1, 0],
      },
      {
        // V = [0.5, 0.5, 0.5]; transition (2 → 2, r = 1); α = 0.5, γ = 0.5.
        // target = 1 + 0.5 * 0.5 = 1.25; delta = 1.25 − 0.5 = 0.75;
        // V[2] ← 0.5 + 0.5 * 0.75 = 0.875.
        name: "self-loop with reward",
        input: [[0.5, 0.5, 0.5], 2, 2, 1.0, 0.5, 0.5],
        expected: [0.5, 0.5, 0.875],
      },
    ],
  },
  solution: `function tdZeroUpdate(V, s, sPrime, reward, alpha, gamma) {
  const Vnew = V.slice();
  const target = reward + gamma * V[sPrime];
  const delta = target - V[s];
  Vnew[s] = V[s] + alpha * delta;
  return Vnew;
}`,
});

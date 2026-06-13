// Exercise 6.A — one sweep of Bellman optimality.
//
// The reader has just seen value iteration converge on a 4-state
// chain. Now they implement one sweep of the optimality backup:
//   V_new[s] = max_a { reward(s,a) + γ · V[s'(s,a)] }.
//
// Check kind: fixedTable. A small deterministic MDP (4 states, 2
// actions per state) with a hand-computable answer for two starting
// V vectors. The "actionsPerState" shape keeps the test data
// readable: each state lists its available (reward, nextState) pairs.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch6-bellman-sweep-exercise",
  title: "One sweep of value iteration",
  prompt: `
**Bellman optimality backup.** Given current state values \`V\`, a
discount \`gamma\`, and the per-state list of available actions, do
one sweep:

\`V_new[s] = max over actions { reward + gamma · V[next_state] }\`

Don't mutate \`V\` — return a new array.
  `,
  signature: `
V               : Array<number>(N)               — current state values; V[s] is the value of state s
gamma           : number                         — discount factor in [0, 1]
actionsPerState : Array<Array<{reward, nextState}>>(N)
                                                 — actionsPerState[s] is the list of actions
                                                   available from state s. Each action is an
                                                   object { reward: number, nextState: int }
                                                   describing a deterministic transition.
→               : Array<number>(N)               — return the swept V'; V'[s] is the new value
  `,
  template: `function bellmanOptimalSweep(V, gamma, actionsPerState) {
  const N = V.length;
  const Vnew = new Array(N);
  for (let s = 0; s < N; s++) {
    // For each available action, compute reward + gamma * V[nextState].
    // Take the max. Write it to Vnew[s].

  }
  return Vnew;
}`,
  entrypoint: "bellmanOptimalSweep",
  entrypointArgs: ["V", "gamma", "actionsPerState"],
  check: {
    kind: "fixedTable",
    tolerance: 1e-6,
    cases: (() => {
      // Tiny 4-state deterministic chain with a terminal-style goal.
      //
      //   s=0 ──Left──→ s=0  (r = 0)
      //   s=0 ──Right─→ s=1  (r = 0)
      //   s=1 ──Left──→ s=0  (r = 0)
      //   s=1 ──Right─→ s=2  (r = 0)
      //   s=2 ──Left──→ s=1  (r = 0)
      //   s=2 ──Right─→ s=3  (r = +1) ← reward on transition INTO s=3
      //   s=3 ──Stay──→ s=3  (r = 0)   ← absorbing
      const mdp = [
        [{ reward: 0, nextState: 0 }, { reward: 0, nextState: 1 }],
        [{ reward: 0, nextState: 0 }, { reward: 0, nextState: 2 }],
        [{ reward: 0, nextState: 1 }, { reward: 1, nextState: 3 }],
        [{ reward: 0, nextState: 3 }],
      ];
      const gamma = 0.9;
      // From V = [0, 0, 0, 0]:
      //   s=0: max(0 + 0.9*0, 0 + 0.9*0) = 0
      //   s=1: max(0 + 0.9*0, 0 + 0.9*0) = 0
      //   s=2: max(0 + 0.9*0, 1 + 0.9*0) = 1
      //   s=3: 0 + 0.9*0 = 0
      // From V = [0, 0, 1, 0]:
      //   s=0: max(0, 0.9*0) = 0
      //   s=1: max(0, 0.9*1) = 0.9
      //   s=2: max(0, 1 + 0.9*0) = 1
      //   s=3: 0
      // From V = [0, 0.9, 1, 0]:
      //   s=0: max(0, 0.9*0.9) = 0.81
      //   s=1: max(0, 0.9*1) = 0.9
      //   s=2: max(0.9*0.9, 1) = 1
      //   s=3: 0
      return [
        { name: "V = [0,0,0,0]",     input: [[0, 0, 0, 0],     gamma, mdp], expected: [0,    0,    1, 0] },
        { name: "V = [0,0,1,0]",     input: [[0, 0, 1, 0],     gamma, mdp], expected: [0,    0.9,  1, 0] },
        { name: "V = [0,0.9,1,0]",   input: [[0, 0.9, 1, 0],   gamma, mdp], expected: [0.81, 0.9,  1, 0] },
      ];
    })(),
  },
  solution: `function bellmanOptimalSweep(V, gamma, actionsPerState) {
  const N = V.length;
  const Vnew = new Array(N);
  for (let s = 0; s < N; s++) {
    let best = -Infinity;
    for (const { reward, nextState } of actionsPerState[s]) {
      const q = reward + gamma * V[nextState];
      if (q > best) best = q;
    }
    Vnew[s] = best;
  }
  return Vnew;
}`,
});

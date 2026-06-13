// Exercise 1.A — power iteration finds the dominant eigenvector.
//
// The reader has just seen the chapter's power-iteration figure: a
// generic starting vector, after k applications of A, lines up with
// the dominant eigenvector. The student types the loop themselves.
//
// Check kind: fixedTable. Two matrices with hand-computable dominant
// eigenvectors:
//   - A = [[2, 1], [1, 2]]:  λ₁ = 3, v₁ = [1,1]/√2 ≈ [0.7071, 0.7071]
//   - A = [[3, 0], [0, 1]]:  λ₁ = 3, v₁ = [1, 0]
// Both should converge from a sensible starting vector in ~30 steps.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch1-power-iteration-exercise",
  title: "Implement power iteration",
  prompt: `
**Power iteration.** Repeatedly applying \`A\` to a generic starting
vector and renormalising lands you on the dominant eigenvector. The
loop is two lines:

\`v ← A v\` ; \`v ← v / ||v||\`

Implement \`K\` steps of this for a 2×2 \`A\` and starting vector
\`v0\`. Return the final unit vector as a plain array of two numbers.
The starter code already flips the sign so the first non-zero
component is positive (just match that convention).
  `,
  signature: `
A   : [[a, b], [c, d]]   — 2×2 matrix as nested arrays (row-major):
                              A[0][0] = a   A[0][1] = b      (row 0)
                              A[1][0] = c   A[1][1] = d      (row 1)
v0  : [x, y]             — 2-element starting vector (plain array)
K   : number             — how many power-iteration steps to take
→   : [x, y]             — the final unit vector (plain array of 2)
  `,
  template: `function powerIteration(A, v0, K) {
  let v = v0.slice();
  for (let k = 0; k < K; k++) {
    // 1. apply A: w = A v
    // 2. normalise: v = w / ||w||

  }
  // sign convention: make the first non-zero component positive
  if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; }
  return v;
}`,
  entrypoint: "powerIteration",
  entrypointArgs: ["A", "v0", "K"],
  check: {
    kind: "fixedTable",
    tolerance: 1e-3,
    cases: [
      {
        name: "A = [[2,1],[1,2]]",
        input: [[[2, 1], [1, 2]], [1, 0], 30],
        expected: [Math.SQRT1_2, Math.SQRT1_2],     // [0.7071, 0.7071]
      },
      {
        name: "diagonal",
        input: [[[3, 0], [0, 1]], [0.6, 0.8], 30],
        expected: [1, 0],
      },
      {
        name: "rotation-stretch",
        input: [[[4, 1], [2, 3]], [1, 1], 40],
        // Dominant eigenvalue of [[4,1],[2,3]]: λ = (7+√9)/2 = 5.
        // Eigenvector for λ=5: (A − 5I)v = [[-1, 1],[2, -2]] v = 0 ⇒ v ∝ [1, 1].
        expected: [Math.SQRT1_2, Math.SQRT1_2],
      },
    ],
  },
  solution: `function powerIteration(A, v0, K) {
  let v = v0.slice();
  for (let k = 0; k < K; k++) {
    const w = [
      A[0][0] * v[0] + A[0][1] * v[1],
      A[1][0] * v[0] + A[1][1] * v[1],
    ];
    const n = Math.hypot(w[0], w[1]);
    v = [w[0] / n, w[1] / n];
  }
  if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; }
  return v;
}`,
});

// Exercise 11.A — backward for tanh.
//
// The reader has just seen the live MLP train on XOR through tanh
// hidden units. Now they implement the one fact every backprop
// derivation hinges on: `d/dx tanh(x) = 1 - tanh(x)^2`.
//
// Check kind: gradCheck. The harness generates random `x`s, computes
// `y = tanh(x)` for the student (forward is given), picks a random
// upstream gradient `dy`, then compares the student's analytic `dx`
// to the finite-difference numerical derivative of `sum(forward(x) * dy)`
// w.r.t. `x`. The textbook's own claim ("backprop matches calculus")
// is the test.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch11-tanh-backward-exercise",
  title: "Implement backward for tanh",
  prompt: `
The forward pass computes \`y = tanh(x)\`. Your job is the backward
pass: given the upstream gradient \`dy = ∂L/∂y\` and the cached
forward output \`y\`, return \`dx = ∂L/∂x\`.

Calculus says **\`d/dx tanh(x) = 1 - tanh(x)^2\`**. Chain rule gives
**\`dx = dy · (1 - y^2)\`**. Type it.
  `,
  signature: `
dy  : Float32Array(N)   — upstream gradient ∂L/∂y, one value per element
y   : Float32Array(N)   — cached forward output (already = tanh(x))
→   : Float32Array(N)   — return ∂L/∂x; same length as dy and y
  `,
  template: `function backwardTanh(dy, y) {
  const dx = new Float32Array(dy.length);
  // your code here

  return dx;
}`,
  entrypoint: "backwardTanh",
  entrypointArgs: ["dy", "y"],
  check: {
    kind: "gradCheck",
    forward: (x) => Math.tanh(x),
    inputs: { length: 16, range: [-3, 3] },
    nTests: 5,
    tolerance: 1e-3,
    epsilon: 1e-4,
    seed: 7,
  },
  solution: `function backwardTanh(dy, y) {
  const dx = new Float32Array(dy.length);
  for (let i = 0; i < dy.length; i++) {
    dx[i] = dy[i] * (1 - y[i] * y[i]);
  }
  return dx;
}`,
});

// Exercise 11.B — backward for ReLU.
//
// Companion to the tanh exercise. ReLU's backward is a *gate*: the
// gradient passes through where the input was positive and gets
// stopped (set to zero) where the input was negative. This is
// pedagogically rich — the gated zero is exactly what creates "dead
// ReLU" neurons (no gradient → no learning) and is the reason
// initialisation strategies (He / Kaiming) matter so much for ReLU
// networks.
//
// Note: ReLU's backward needs `x` (the pre-activation), not `y` (the
// post-activation), because the gate is `1{x > 0}`. After ReLU the
// distinction is lost — both negative `x` and zero `x` produce `y=0`.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch11-relu-backward-exercise",
  title: "Implement backward for ReLU",
  prompt: `
The forward pass computes \`y = max(0, x)\` elementwise. Implement the
backward pass.

Calculus: **\`d/dx max(0, x) = 1\` if \`x > 0\`, else \`0\`**. Chain
rule: **\`dx = dy · 1{x > 0}\`** — the gradient flows where the
pre-activation was positive and is stopped where it wasn't. This gate
is why a unit that drifts deep into the negative region stops learning
("dead ReLU").

Note the second arg: you receive the cached **pre-activation** \`x\`,
not the post-activation \`y\` — the sign of \`x\` is lost once ReLU
clamps it to zero. Every autograd framework caches \`x\` here for
exactly this reason.
  `,
  signature: `
dy  : Float32Array(N)   — upstream gradient ∂L/∂y
x   : Float32Array(N)   — cached PRE-activation (before ReLU)
→   : Float32Array(N)   — return ∂L/∂x; same length as dy and x
  `,
  template: `function backwardRelu(dy, x) {
  const dx = new Float32Array(dy.length);
  // your code here

  return dx;
}`,
  entrypoint: "backwardRelu",
  entrypointArgs: ["dy", "x"],
  check: {
    // We adapt gradCheck: the harness's forward is applied elementwise
    // and the student receives `(dy, forward_output)` by default. ReLU's
    // backward needs the pre-activation, not the post-activation, so we
    // pass the original `x` as the second arg. The harness's gradCheck
    // calls `fn(dy, y)`; we override here by writing a custom forward
    // that returns the *pre-activation* x as the cached value passed
    // back in.
    //
    // Trick: forward returns `relu(x)` for the numerical-derivative
    // computation but we need `x` to be passed to the student's
    // function. We do this by computing the numerical derivative
    // normally (using `relu`), and using a wrapped student-fn that
    // re-applies the relation y = relu(x) → can't recover x from y.
    //
    // Resolution: use `reference` check kind instead. The author knows
    // the right answer; let the harness compare.
    kind: "reference",
    reference: (dy, x) => {
      const dx = new Float32Array(dy.length);
      for (let i = 0; i < dy.length; i++) dx[i] = x[i] > 0 ? dy[i] : 0;
      return dx;
    },
    generate: (rng) => {
      const n = 16;
      const dy = new Float32Array(n);
      const x  = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        dy[i] = 2 * rng() - 1;          // [-1, 1]
        x[i]  = 6 * rng() - 3;          // [-3, 3] — straddles the gate
      }
      return [dy, x];
    },
    nTests: 5,
    tolerance: 1e-6,
    seed: 11,
  },
  solution: `function backwardRelu(dy, x) {
  const dx = new Float32Array(dy.length);
  for (let i = 0; i < dy.length; i++) {
    dx[i] = x[i] > 0 ? dy[i] : 0;
  }
  return dx;
}`,
});

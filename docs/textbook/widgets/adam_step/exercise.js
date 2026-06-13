// Exercise 11.C — one Adam parameter update.
//
// The reader has seen Adam train an MLP in the `live_mlp_xor` widget
// and has been told the update rule. Now they implement the rule.
//
// Adam (Kingma & Ba 2014) keeps two EMAs of the gradient and its
// square, bias-corrects them, and rescales the step by the inverse
// square root of the second moment:
//
//   m ← β₁·m + (1−β₁)·g
//   v ← β₂·v + (1−β₂)·g²
//   m̂ ← m / (1 − β₁^t)        (bias correction)
//   v̂ ← v / (1 − β₂^t)
//   θ ← θ − lr · m̂ / (√v̂ + ε)
//
// Check kind: reference. The harness compares the student's output
// to the author's reference implementation on random inputs.

import { defineExercise } from "../shared/exercise.js";

function referenceAdamStep({ params, grads, m, v, t, lr, beta1, beta2, eps }) {
  const n = params.length;
  const mNew = new Float32Array(n);
  const vNew = new Float32Array(n);
  const paramsNew = new Float32Array(n);
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  for (let i = 0; i < n; i++) {
    mNew[i] = beta1 * m[i] + (1 - beta1) * grads[i];
    vNew[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
    const mHat = mNew[i] / bc1;
    const vHat = vNew[i] / bc2;
    paramsNew[i] = params[i] - lr * mHat / (Math.sqrt(vHat) + eps);
  }
  return { params: paramsNew, m: mNew, v: vNew };
}

defineExercise({
  hostId: "ch11-adam-step-exercise",
  title: "One Adam optimiser step",
  prompt: `
**Adam.** Implement one Adam update given the current params, the
fresh gradients, and the running moment estimates (\`m\`, \`v\`).

\`m ← β₁ · m + (1 − β₁) · g\`

\`v ← β₂ · v + (1 − β₂) · g²\`

\`m̂ ← m / (1 − β₁ᵗ)\`  ;  \`v̂ ← v / (1 − β₂ᵗ)\`

\`θ ← θ − lr · m̂ / (√v̂ + ε)\`

Input is a single object with eight fields (see signature below).
Return \`{ params, m, v }\` — the *new* values; don't mutate the
inputs. The harness compares your output to a reference
implementation on five seeded inputs; match within \`1e-5\`.
  `,
  signature: `
Single arg, an object with these fields:
  params  : Float32Array(N)   — current parameter vector θ
  grads   : Float32Array(N)   — fresh gradient g_t for this step
  m       : Float32Array(N)   — running first-moment EMA (input state)
  v       : Float32Array(N)   — running second-moment EMA (input state)
  t       : int               — step counter ≥ 1 (drives bias correction)
  lr      : number            — learning rate (e.g. 0.01)
  beta1   : number            — first-moment decay (e.g. 0.9)
  beta2   : number            — second-moment decay (e.g. 0.999)
  eps     : number            — denominator floor (e.g. 1e-8)

Return:
  { params : Float32Array(N),   — updated θ' (don't mutate the input)
    m      : Float32Array(N),   — updated first-moment EMA m'
    v      : Float32Array(N) }  — updated second-moment EMA v'
  `,
  template: `function adamStep({ params, grads, m, v, t, lr, beta1, beta2, eps }) {
  const n = params.length;
  const mNew = new Float32Array(n);
  const vNew = new Float32Array(n);
  const paramsNew = new Float32Array(n);
  // your code here

  return { params: paramsNew, m: mNew, v: vNew };
}`,
  entrypoint: "adamStep",
  entrypointArgs: ["state"],
  check: {
    kind: "reference",
    reference: referenceAdamStep,
    generate: (rng, testIdx) => {
      const n = 8;
      const params = new Float32Array(n);
      const grads  = new Float32Array(n);
      const m      = new Float32Array(n);
      const v      = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        params[i] = 2 * rng() - 1;
        grads[i]  = 2 * rng() - 1;
        m[i]      = 0.5 * (2 * rng() - 1);
        v[i]      = rng() * 0.1;          // v stays non-negative
      }
      return [{
        params, grads, m, v,
        t:     testIdx + 1,              // varies bias correction per test
        lr:    0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps:   1e-8,
      }];
    },
    nTests: 5,
    tolerance: 1e-5,
    seed: 42,
  },
  solution: `function adamStep({ params, grads, m, v, t, lr, beta1, beta2, eps }) {
  const n = params.length;
  const mNew = new Float32Array(n);
  const vNew = new Float32Array(n);
  const paramsNew = new Float32Array(n);
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  for (let i = 0; i < n; i++) {
    mNew[i] = beta1 * m[i] + (1 - beta1) * grads[i];
    vNew[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];
    const mHat = mNew[i] / bc1;
    const vHat = vNew[i] / bc2;
    paramsNew[i] = params[i] - lr * mHat / (Math.sqrt(vHat) + eps);
  }
  return { params: paramsNew, m: mNew, v: vNew };
}`,
});

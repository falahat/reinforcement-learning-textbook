// docs/textbook/widgets/<name>/exercise.js
//
// Minimum-viable live-coding exercise using the defineExercise
// scaffold. Replace `<name>`, `<chN>`, the prompt, the template, and
// the check with your own. Run `node docs/textbook/widgets/shared/
// smoketest.js` after editing to verify the solution passes its
// own check.
//
// Chapter markdown contains:
//
//     ### Try implementing it: <short description>
//
//     <div id="<chN>-<name>-exercise"></div>
//     <script type="module" src="./widgets/<name>/exercise.js"></script>
//
//     2-3 sentence pedagogical note about WHY typing this teaches
//     something the prose alone can't.
//
// Author convention: keep the template a hollow scaffold. The
// signature + return-shape allocation may be filled in if non-
// obvious; the algorithm itself goes in the empty line.

import { defineExercise } from "../shared/exercise.js";

defineExercise({
  // Mount-block id — matches the chapter markdown's `<div id="…">`.
  hostId: "<chN>-<name>-exercise",

  // Short label shown above the editor.
  title: "Implement <algorithm>",

  // Student-facing instructions. Markdown-ish: `code`, **bold**,
  // blank lines for paragraph breaks. Keep to ~5 lines.
  prompt: `
**<Algorithm name>.** One-paragraph description of what to compute.

Formula: **\`y = f(x)\`**. Type it.
  `,

  // Type signature for the function args. Renders as a monospace
  // pre block between prompt and editor. Column-align `:` and `—`
  // for readability. The harness *also* auto-emits a collapsed
  // "▸ Show example input" disclosure with seeded sample values,
  // so writing this block + check.generate / check.cases is enough
  // for the reader to know exactly what they're working with.
  signature: `
x   : Float32Array(N)   — input vector
→   : Float32Array(N)   — return: same length as x
  `,

  // Starter code shown in the editor. Parses cleanly so the reader's
  // first Run reports test failures (not a syntax error). Empty line
  // where the algorithm belongs.
  template: `function entrypointName(x) {
  const out = new Float32Array(x.length);
  // your code here

  return out;
}`,

  // Function name the harness extracts from the student's code.
  entrypoint: "entrypointName",
  entrypointArgs: ["x"],

  // Pick the check kind that matches what "correct" means here.
  // See references/11-exercise-authoring.md for the full guide.
  check: {
    // ── Option A: discrete input → expected output ──────────────
    // kind: "fixedTable",
    // tolerance: 1e-6,
    // cases: [
    //   { name: "tiny case",    input: [[1, 2, 3]], expected: [...] },
    //   { name: "boundary",     input: [[0]],       expected: [...] },
    // ],

    // ── Option B: finite-difference grad-check (backprop) ───────
    kind: "gradCheck",
    forward: (x) => Math.tanh(x),          // scalar-to-scalar map
    inputs: { length: 16, range: [-3, 3] },
    nTests: 5,
    tolerance: 1e-3,
    epsilon: 1e-4,
    seed: 7,

    // ── Option C: oracle on generated inputs ────────────────────
    // kind: "reference",
    // reference: (...args) => /* canonical answer */,
    // generate: (rng, testIdx) => [/* args for this test */],
    // nTests: 5,
    // tolerance: 1e-5,
    // seed: 42,
  },

  // Reference solution. Used by the smoketest (`node smoketest.js`)
  // to verify your check block is consistent. Also displayed via the
  // "Show solution" reveal after 3 failed attempts.
  solution: `function entrypointName(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
  return out;
}`,
});

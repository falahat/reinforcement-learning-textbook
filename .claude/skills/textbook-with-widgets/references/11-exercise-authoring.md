# Phase 5 — Live coding exercises with `defineExercise`

A widget is "drag a slider, watch the picture move." An **exercise**
is "fill in this function body, watch your code get auto-checked."
Same mdBook static-site pipeline, no server. Use it when typing the
algorithm yourself teaches something the prose alone can't.

This reference is the practical authoring guide; the
`defineExercise` scaffold lives in `shared/exercise.js`.

## When to ship an exercise

The same discipline as widgets: load-bearing only. Ask yourself:

> Does typing this themselves understand the algorithm differently
> from reading prose + running a slider?

If **no**, skip the exercise. If **yes**, ship it. A textbook with 30
mediocre exercises is worse than one with 9 great ones. Good
candidates are short algorithms whose code is the algorithm — backprop
for one op, a TD(0) update, a Bellman sweep, an Adam step, a power-
iteration loop. Five to twenty lines, no fancy data structures.

Bad candidates: anything where the student would spend most of their
time on Python-vs-JS bookkeeping, or anything the prose already
captured perfectly with a five-line code block.

## The two-line embed (mirrors widgets)

In a chapter markdown file:

```markdown
### Try implementing it: backward for tanh

<div id="ch11-tanh-backward-exercise"></div>
<script type="module" src="./widgets/tanh_backward/exercise.js"></script>

A 2-3 sentence pedagogical note about *why* this exercise matters.
```

ID convention: `ch<N>-<short-name>-exercise`. Same lowercase-hyphen
style as widgets; the `-exercise` suffix replaces `-widget` so a CSS
file or a future grep can tell them apart at a glance.

## Inside the exercise file

The minimum-viable exercise uses the `defineExercise` scaffold:

```js
// docs/textbook/widgets/tanh_backward/exercise.js
import { defineExercise } from "../shared/exercise.js";

defineExercise({
  hostId: "ch11-tanh-backward-exercise",
  title: "Implement backward for tanh",
  prompt: `
The forward pass computes \`y = tanh(x)\`. Implement the backward.

Chain rule: **\`dx = dy · (1 - y^2)\`**. Type it.
  `,
  // Type-hint block for the function arguments. Column-align the
  // `:` and `—` so it reads like a real type signature. The harness
  // renders this in a monospace pre between prompt and editor.
  signature: `
dy  : Float32Array(N)   — upstream gradient ∂L/∂y
y   : Float32Array(N)   — cached forward output (= tanh(x))
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
    seed: 7,
  },
  solution: `function backwardTanh(dy, y) {
  const dx = new Float32Array(dy.length);
  for (let i = 0; i < dy.length; i++) dx[i] = dy[i] * (1 - y[i] * y[i]);
  return dx;
}`,
});
```

That's ~35 lines. The `defineExercise` scaffold handles the editor
(textarea by default, async-upgraded to CodeMirror 6 with syntax
highlighting when the esm.sh CDN is reachable), the Run button,
error reporting, localStorage persistence, the "Show solution"
reveal (collapsed by default; opens after 3 failed attempts or
with a confirm-dialog), AND an auto-generated "▸ Show example
input" disclosure that dumps the actual values the harness will
pass on test 1.

You never write DOM code, never wire `addEventListener`, never
define a CONTROLS_HTML template. That's the scaffold's job.

## Why type hints matter — the `signature` field

A prompt like "implement backward for tanh given `dy` and `y`"
leaves a reader guessing about shape: vectors? matrices? what
length? row-major or column-major? typed arrays? plain arrays?
The reader either reads the spec source to find out or types
exploratory `console.log(dy.length)` calls.

The `signature` block kills that guesswork. Author-written, ~5
lines, monospace, columns aligned:

```
signature: `
V               : Array<number>(N)               — current state values; V[s] is the value of state s
gamma           : number                         — discount factor in [0, 1]
actionsPerState : Array<Array<{reward, nextState}>>(N)
                                                 — actionsPerState[s] is the list of actions
                                                   available from state s. Each action is an
                                                   object { reward: number, nextState: int }
                                                   describing a deterministic transition.
→               : Array<number>(N)               — return the swept V'; V'[s] is the new value
`
```

Use the arrow `→` for the return type so it lines up vertically
with the args. State the index semantics (`V[s]` is the value of
state s) and any shape conventions (row-major, length N, etc.).
Pair short type-ish notation with English so an author who hasn't
seen TypeScript can still read it.

The harness *also* auto-emits a collapsed "▸ Show example input"
disclosure right under the signature. Click it: the harness runs
the spec's seeded generator and dumps real values:

```
dy = Float32Array(16) [-0.4135, 0.0695, -0.4073, 0.9559, …, -0.8510, 0.8259, 0.1074, 0.2432]
y  = Float32Array(16) [-0.9943, -0.9896,  0.9935, 0.8319, …,  0.9193, 0.1100, -0.9486, -0.6597]
```

For `fixedTable` it's `cases[0].input`. For `reference` it runs
`check.generate(rng, 0)` with the spec's seed. For `gradCheck` it
mimics the harness's input synthesis. Authors write the signature
once; the sample block writes itself.

Together: the reader sees prose (prompt) → types (signature) →
real data (sample) → editor. They never have to *guess* what the
function receives.

## The editor: textarea synchronous, CodeMirror 6 if reachable

The harness ships a working `<textarea>` synchronously so the
exercise is usable instantly, then asynchronously lazy-imports
CodeMirror 6 from esm.sh:

```
codemirror@6 + @codemirror/lang-javascript@6 + @codemirror/theme-one-dark@6
```

If the imports resolve, CM6 takes over the editor host and the
reader gets syntax highlighting, line numbers, JS-aware
indentation, search, and the rest of CM6's basicSetup. If anything
fails — offline reader, blocked CDN, strict CSP — the textarea
stays and the exercise still works.

Editor choice rationale:
- **CodeMirror 6**: ESM-native, no build step, plugin-modular
  (~200 KB unminified for our setup, ~70 KB minified). Active,
  modern, used by Quarto/Observable.
- **Monaco** (VS Code's editor): ~2 MB even for stripped builds.
  Way too heavy for a per-page widget.
- **Ace**: older, requires global script tags. CM6 is the better
  modern shape.
- **Prism / highlight.js**: view-only highlighters. Don't help
  for an *editor*.

Authors don't configure any of this. The fallback is automatic.

## The three check kinds

Pick the one that matches what "correct" means for this exercise:

### 1. `fixedTable` — discrete input → expected output

Author provides `cases: [{ input: [...args], expected: ... }, ...]`.
The harness calls the student function with each `input` (spread as
positional args) and compares to `expected`. Tolerance-aware so
floating-point arithmetic doesn't betray you.

```js
check: {
  kind: "fixedTable",
  tolerance: 1e-6,
  cases: [
    {
      name: "V = [0,0,0,0]",
      input: [[0, 0, 0, 0], 0.9, mdpDescription],
      expected: [0, 0, 1, 0],
    },
    // ...
  ],
}
```

**Right when:** the answer is a small known value. One TD(0) update,
one Bellman sweep on a 4-state chain, the first three steps of
value iteration. The reader can verify the expected on paper.

**Wrong when:** the answer is an algorithm trajectory ("converged to
the dominant eigenvector"). A reader who computes a unit vector with
slightly different precision can fail a too-tight tolerance. Either
use a `reference` check, or pad tolerance to `1e-3` for vector
results.

### 2. `gradCheck` — finite-difference numerical derivative

Author provides `forward: (scalar) => scalar`. The harness picks
seeded random `dy` and `x` vectors, computes the numerical
derivative `(forward(x + ε) - forward(x - ε)) / 2ε` element-wise,
and compares to the student's analytic backward.

```js
check: {
  kind: "gradCheck",
  forward: (x) => Math.tanh(x),
  inputs: { length: 16, range: [-3, 3] },
  nTests: 5,
  tolerance: 1e-3,
  epsilon: 1e-4,
  seed: 7,
}
```

**Right when:** the exercise is implementing backprop for one op.
The textbook's own claim ("backprop computes the same thing as
calculus, by the chain rule") becomes the *test of the student's
code*. Pedagogically perfect for the gradient chapters.

**Tuning:** tolerance ~ `1e-3` is the right floor for vector
backward-pass checks against finite differences. Going lower trips
on float32 + finite-step error. Epsilon ~ `1e-4` balances truncation
error (large ε) against catastrophic cancellation (small ε).

**Limitation:** only applies when the forward is elementwise. For
reductions (loss = sum of squared errors), the gradCheck setup needs
a custom adapter — easier to use `reference` instead.

### 3. `reference` — call an oracle on generated inputs

Author provides `reference: (...args) => …` and
`generate: (rng, testIdx) => [args...]`. The harness runs both on
each generated input and compares.

```js
check: {
  kind: "reference",
  reference: (dy, x) => {
    const dx = new Float32Array(dy.length);
    for (let i = 0; i < dy.length; i++) dx[i] = x[i] > 0 ? dy[i] : 0;
    return dx;
  },
  generate: (rng) => {
    const n = 16;
    const dy = new Float32Array(n);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      dy[i] = 2 * rng() - 1;
      x[i] = 6 * rng() - 3;       // straddles the ReLU gate
    }
    return [dy, x];
  },
  nTests: 5,
  tolerance: 1e-6,
  seed: 11,
}
```

**Right when:** the right answer is "whatever the canonical library
does" (Adam step, tile-coding hash, REINFORCE policy-gradient
update), OR when `gradCheck` doesn't apply (ReLU's backward needs
the pre-activation `x`, not `y` — finite differences using `forward`
would give a function `y → dx` that the student can't write).

**Authoring tip:** make `generate` pick inputs that exercise the
edge cases of the algorithm. ReLU's gate at zero; Adam's bias
correction at small `t`. If `nTests = 5` and all five generated
inputs avoid the edge case, a buggy student passes silently.

## Authoring discipline

### 1. The solution is also a test

`docs/textbook/widgets/shared/smoketest.js` compiles every exercise's
`solution` string and runs it through that exercise's own `check`
block. **Run `node smoketest.js` whenever you add or change an
exercise.** It catches:

- Typos in `expected` values for `fixedTable` cases.
- A reference implementation that's wrong on inputs the student
  hasn't tried yet.
- Off-by-one or wrong-tolerance errors in the check spec.
- A solution that doesn't define the entrypoint function name.

If your solution doesn't pass your own check, the exercise is
broken — find out before a reader does.

### 2. Keep the template a hollow scaffold

The starter shown in the editor should:
- Include the function signature.
- Include the return-array allocation if it's non-obvious (e.g.
  `const dx = new Float32Array(dy.length);`).
- Have an empty line where the algorithm goes.
- Already parse and return *something* of the right shape, so the
  reader's first Run shows test failures (not a parse error).

The starter should not:
- Contain the algorithm. That's the point of the exercise.
- Have a misleading comment that suggests the wrong approach.
- Be so spare that the reader has to guess the input/output shapes
  — those are author-provided context.

### 3. The prompt is *short* and *self-contained*

The reader can see the editor, the prompt, the result panel — all
at once on a normal screen. Keep the prompt to ~5 lines maximum.
Use backticks for code, `**bold**` for the formula they need to type.

Don't restate what the chapter already said. The exercise is *after*
the math derivation; assume the reader followed.

### 4. Help text below the mount

Same convention as widgets — a 2-3 sentence student-facing paragraph
in the chapter markdown below the exercise mount. Tell them:
- Why this specific exercise pulls weight beyond the prose.
- What they should notice when they get it right.

A vapid "type this and see green checks" is not help text; the
checks are visible. Tell them what *typing it* teaches:

> ❌ Type the formula and see the green checks.
>
> ✅ ReLU's backward is a *gate*: gradient passes where the
>    pre-activation was positive, zero elsewhere. Note the subtle
>    data-flow point — ReLU needs the **pre-activation** `x`, not
>    the post-activation `y`, because `y = 0` for any non-positive
>    `x` and the sign information is gone. This is exactly the
>    bookkeeping every autograd framework does behind your back.

### 5. One canonical entrypoint name per exercise

`backwardTanh`, not `backward_tanh` or `BackwardTanh`. Mirror the
JS convention. This is what the harness extracts; spelling has to
match `entrypoint` exactly.

### 6. Don't ship one to every chapter

Same rule as widgets. From the source project's 21 chapters, only
~9 got exercises in the first batch. Each exercise should answer
"this is *the* line of code the chapter is really about."

## Pitfalls

These are the exercise equivalents of `references/10-widget-correctness-
audit.md`. None of them is hypothetical — each bit the source project.

### 1. Template doesn't compile

A starter that parses to a `SyntaxError` (mismatched braces, a
missing semicolon in a chain) gives the reader a wall on first Run.
They don't know if they typed wrong or whether the harness is broken.
Always copy the template into a `node -e` quick-check before shipping.

### 2. `fixedTable` expected value computed wrong on paper

The smoketest catches this *only if you also wrote a `solution`*. If
you wrote `cases` from hand calculation and skipped the solution,
the test passes once (against itself) and fails for every reader
who got it right. Always write the solution; let the smoketest verify.

### 3. Tolerance too tight

Float32 arithmetic accumulates error. `1e-6` is fine for fixed-input
scalar checks but `1e-3` is the right floor for vector grad-checks
against finite differences. A student who got the algorithm right
shouldn't fail because of float-precision noise.

### 4. The reference oracle hides a misconception

For `reference` checks, the student's wrong code can happen to
match the oracle on the specific generated inputs while being wrong
in general. Mitigations:
- Use `nTests ≥ 5` so the generator covers more configurations.
- Generate inputs that hit the algorithm's edge cases (zero,
  boundary values, sign-changing inputs).
- For really subtle algorithms, vary the test-index in `generate`
  so e.g. Adam's bias-correction varies per test.

### 5. Help text says "easy"

A reader who couldn't get the exercise reads "easy" and feels
worse. Use "type it" / "implement it" / "the inner loop" — phrases
that respect the work without underselling it.

### 6. Solution reveals too easily

The harness lets the reader click "Show solution" after 3 attempts,
or earlier with a confirm dialog. Don't override the default
threshold lower — struggling for ~5 minutes on the algorithm is the
point. If a reader truly can't get it after 10 minutes, the solution
reveal is honest help; before that, it's bailout.

## File layout

```
docs/textbook/widgets/<exercise_name>/
└── exercise.js     ← one file, declarative spec
```

Same one-directory-per-thing convention as widgets. The `exercise.js`
filename (vs `widget.js`) makes the kind self-evident; the
`-exercise` ID suffix in the mount block matches.

## Glossary of harness internals

These are worth knowing when debugging:

- **`new Function(body)`** — the harness compiles the student's
  code via the `Function` constructor. Catches syntax errors at
  compile time. No `import`, no top-level `await` — keep template
  code self-contained.
- **localStorage key** — `exercise:<hostId>`. The student's draft
  persists across navigation. Clear it via the Reset button or
  `localStorage.clear()` in DevTools.
- **`compareValues`** — handles scalars, vectors (`Float32Array` /
  `Float64Array` / plain Array), matrices (Array of Array), and
  plain objects (recurses by key, returns worst diff). Comparing
  arbitrary user-defined classes will fall through to strict
  equality — return plain objects from the student function.

## Cross-references

- The harness implementation lives at
  `docs/textbook/widgets/shared/exercise.js`. Read it once to know
  the full check surface.
- The widget conventions in `07-widget-authoring.md` apply to
  exercises too: ID format, mount pattern, the "help text below
  the mount" rule, one-directory-per-thing.
- `examples/exercise-template.js` is the smallest exercise. Copy
  it and edit; don't write `exercise.js` from scratch.

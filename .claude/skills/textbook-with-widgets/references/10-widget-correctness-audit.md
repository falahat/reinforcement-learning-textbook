# Widget correctness audit

"Looks like it loaded but the controls don't actually do anything"
is a real bug class that you cannot catch by running `mdbook build`.
The widget mounts, the sliders move, but the plot doesn't change —
or moves but the data underneath doesn't reflect the slider.

Run this audit pass after every widget batch. It's static — just
read the code — but worth doing systematically because the bugs are
silent (no error in console, no failed build).

## Bugs to look for

### Bug 1: Declared control not used in render

```js
// BUG: kmax declared but never referenced in render's body.
controls: { gamma: {...}, kmax: {...} },
render: (host, { gamma, kmax }, slots) => {
  const data = d3.range(100).map(k => ({ k, y: Math.pow(gamma, k) }));
  //                  ^^^ hardcoded 100, kmax ignored
}
```

Detection: every control name in the `controls` object must appear
in the render body and feed into either a Plot mark, axis config,
or readout text. If a control is `readNumber`'d but never used
downstream, that's a dead slider.

Fix: use the control in the computation, or remove it from the
controls object.

### Bug 2: Wrong slot name

```js
// BUG: declared `slots: ["weights", "cumulative"]` but render writes
// to `slots.main` (which is undefined when custom slots are listed).
defineWidget({
  slots: ["weights", "cumulative"],
  render: (host, params, slots) => {
    slots.main.replaceChildren(Plot.plot({...}));
    //   ^^^^ undefined
  },
});
```

Detection: when `slots: [...]` is declared, scan the render body
for `slots.main` / `slots.X` references and verify each name appears
in the slots array.

Fix: either use the declared slot names, or remove the custom slots
list and rely on the default `["main"]`.

### Bug 3: Stale data outside render

```js
// BUG: data computed once at module load, ignores all sliders.
const fixedData = d3.range(100).map(k => ({ k, y: 0.9 ** k }));

defineWidget({
  controls: { gamma: {...} },
  render: (host, { gamma }, slots) => {
    slots.main.replaceChildren(Plot.plot({
      marks: [Plot.line(fixedData, { x: "k", y: "y" })],
      //               ^^^^^^^^ never recomputes
    }));
  },
});
```

Detection: any non-trivial computation outside the render arrow
function is suspect. Constants are fine; arrays/objects that depend
on parameters are not.

Fix: move the computation inside render so it re-runs on every input
change.

### Bug 4: Stepper trajectory not depending on all params

```js
// BUG: defineStepper trajectory ignores the `policy` control,
// so changing the policy slider doesn't change the frame sequence.
defineStepper({
  controls: { gridSize: {...}, policy: {...} },
  trajectory: ({ gridSize }) => {  // policy missing from destructure
    return buildFrames(gridSize);
  },
  render: ...,
});
```

Detection: every control name should appear in either the
trajectory's destructure list or the render's destructure list (or
both). A control that's destructured in neither is dead.

Fix: pass the control into trajectory or render where it's needed.

### Bug 5: Hardcoded plot domain clips the data

```js
// BUG: data exceeds [0, 0.3] when α is large, but plot clips.
y: { domain: [0, 0.3], grid: true },
```

The slider moves the data; the data moves *off the top of the
visible chart*. Visually looks like nothing changed.

Detection: any `domain: [low, high]` with hard-coded numbers is
suspect. Verify the data actually fits in that range across all
slider positions.

Fix: use `zero: true` (anchor at 0, auto-fit top) or omit the
domain entirely (Plot auto-fits). Hard-coded domains are fine when
the math guarantees the range (e.g. `y: { domain: [0, 1] }` for a
probability).

### Bug 6: WASM init race

For widgets that import a Rust WASM module:

```js
// BUG: defineWidget runs before await init() completes.
import init, { compute } from "./pkg/widget_X.js";

await init();  // FINE — at module top level

defineWidget({
  render: (host, params, slots) => {
    const result = compute(...);  // safe: init done before this runs
  },
});
```

vs

```js
// BUG: race — render may fire before init resolves.
import init, { compute } from "./pkg/widget_X.js";

defineWidget({
  render: async (host, params, slots) => {
    await init();  // ← fires on every render; first render race
    const result = compute(...);
  },
});
```

Detection: `init()` call inside render is a yellow flag. Top-level
await before `defineWidget` is the safe pattern.

### Bug 7: Closure-captured stale value

```js
// BUG: gamma captured in setInterval closure as 0.9; slider changes
// it but the interval keeps using the old value.
defineWidget({
  controls: { gamma: {...} },
  render: (host, { gamma }, slots) => {
    setInterval(() => {
      doSomethingWith(gamma);  // stale on subsequent ticks
    }, 100);
  },
});
```

Detection: setInterval / setTimeout / requestAnimationFrame inside
render that captures params. The params are fresh ON THAT RENDER
CALL but stale on later ticks.

Fix: read the parameter inside the callback rather than capturing.
Or use defineStepper which handles this correctly.

### Bug 8: `Plot.barY` / `Plot.barX` on a continuous scale

```js
// BUG: barY needs an ordinal/band x-scale. On a continuous numeric
// x (linear scale, 2-element [min, max] domain) Plot renders bars
// with width 0 — the chart looks empty.
x: { label: "k", domain: [0, K_MAX] },
marks: [
  Plot.barY(data, { x: "k", y: "y" }),  // ← zero-width bars
],
```

Detection: any `Plot.barY` whose `x` axis (or `Plot.barX` whose `y`
axis) has a 2-element numeric `[min, max]` domain, no `type: "band"`,
and no `interval` option on the mark.

`Plot.barY` is fine when `x` is:
- a string field (Plot auto-infers ordinal), OR
- given an *array* domain of discrete labels/values (also ordinal), OR
- explicitly `type: "band"`.

Fix: switch to `Plot.rectY` with `interval: 1` (or whatever the step
is), OR change the axis to `type: "band"` with an array domain.

```js
Plot.rectY(data, { x: "k", y: "y", interval: 1, fill: "..." }),
```

**Historical incidents.** The source project shipped this bug in
SIX widgets — `discount_factor`, `return_calc` (twice — top + bottom
plots), `lambda_return`, `hca_posterior`, `smdp_bootstrap`. Every
one looked like "slider does nothing" because the axis ticks rendered
fine, the chart frame was there, but no actual bars. Took a user
report to discover; layer-2 source validators don't catch it because
the JS code is syntactically valid and renders without errors.

### Bug 9: `Array.prototype.reverse()` / `.sort()` on a shared array

```js
const STATE_NAMES = ["A", "B", "C", "D", "E"];

defineWidget({
  render: (host, params, slots) => {
    Plot.plot({
      y: { domain: STATE_NAMES.reverse().slice() },  // BUG
      // ...
    });
    // Now STATE_NAMES is ["E","D","C","B","A"] FOREVER.
    // Subsequent STATE_NAMES[step.s] returns the *reversed* name.
  },
});
```

`Array.prototype.reverse` and `Array.prototype.sort` mutate the array
in place AND return the same reference. Calling them on a module-level
`const` array (or any array that other code reads from by index)
silently corrupts every later indexed lookup. Indices that are
*palindromic* (middle of a 5-element array) appear correct, masking
the bug. Endpoints (A/E, B/D) swap.

Detection: `\.reverse\(\)` or `\.sort\(` called on an identifier
that's declared at module level. Safe when called on a fresh array:
results of `.map(...)`, `.filter(...)`, `d3.range(...)`,
`Array.from(...)`, spread `[...x]`.

Fix: `.slice().reverse()` — slice creates a copy, reverse mutates
the copy.

**Historical incident.** The `fwd_bwd_equivalence` widget had this in
its trace plot's y-domain. The bug swapped A↔E and B↔D in the episode
strip and per-visit table, while the bars/dots themselves used the
swapped index consistently, so the equivalence theorem still appeared
to hold visually — just at the wrong state labels. Took a user
reporting the data didn't match the displayed episode to find.

### Bug 10: Convergence-loop cap pegging

Already-handled pattern in `references/06-widgets-architecture.md`
and in the fixes for `modified_pi`, `pi_vs_vi`. Recap so it stays
together with the other audit rules:

```js
const CONV_THRESH = 1e-6;
const MAX_OUTER = 200;
for (let it = 1; it <= MAX_OUTER; it++) {
  // ... update V, compute delta ...
  if (delta < CONV_THRESH) break;
}
```

When `delta` shrinks at rate γ per iteration, hitting `CONV_THRESH`
requires `log(CONV_THRESH) / log(γ)` iterations. For γ near 1
(typical slider max 0.99 → ~1374 iters), `MAX_OUTER < 2000` will
peg the cap. The displayed total-iterations / total-backups then
becomes constant in γ even though the slider visibly moves.

Detection: any `MAX_*` constant ≤ 1000 paired with a `delta < 1e-6`
test in a loop whose contraction rate depends on a slider.

Fix: raise the cap (cheap — these widgets do small grids), OR switch
to a policy-stability convergence test, OR loosen the threshold to
something reachable.

### Bug 11: Plot point outside chart y-domain creates spurious "drop" segment

```js
const yDomain = [Math.min(x0, xStar) - 1, Math.max(x0, xStar) + 1];
// ...
const cobweb = [{ x: x0, y: 0 }];  // BUG: y=0 may be outside yDomain
// build staircase from here
```

When a polyline endpoint lies outside the chart's y-domain, Plot
clips the segment at the domain boundary. The visible line then
appears to start (or end) at the chart edge, often looking like a
spurious vertical "drop to zero" that doesn't represent the math.
If the polyline goes from `(x_max, 0)` to interior points, a reader
scanning the chart left-to-right reads the rightmost segment as
"the end" — and sees the line "reset to zero" there.

Detection: polyline data whose first or last point has a coordinate
that lies outside the explicit `x.domain` / `y.domain`.

Fix: place the polyline endpoint at a coordinate that's actually
*on* the visible chart, even if that means changing the math
slightly. For the contraction-mapping cobweb, start at `(x₀, x₀)`
on the y=x diagonal rather than the artificial `(x₀, 0)`.

**Historical incident.** The `contraction` widget started the
cobweb at `(x₀, 0)` "to make the first vertical visible." y=0 was
below the chart's y-domain (which is anchored to the iterates' range,
not to 0), so the leading segment ran from below the chart up to the
first staircase vertex. Since the staircase decreases right-to-left
toward x*, the rightmost edge of the polyline was that off-domain
segment — and a left-to-right reader saw a "reset to zero" at "the
end" of the line.

### Bug 12: Slider default outside `[min, max]`

```js
gamma: { min: 0.5, max: 0.999, step: 0.01, default: 1.0 },
//                       ^^^^^                    ^^^ outside the range
```

Most browsers clamp the value to `[min, max]` when the slider
mounts, so the *displayed* slider position lands at `max`, not at
the declared default. The widget renders with γ=0.999, not the
intended γ=1.0. Usually harmless (the clamp produces a sensible
value), but worth flagging: it indicates the author intended
γ=1.0 to be valid and probably wants `max: 1.0`, OR intended γ < 1
and the `default: 1.0` is a typo.

Detection: scan every `controls: {...}` entry; verify
`min ≤ default ≤ max`.

Fix: either bump `max` to include the intended default, or correct
the default.

**Historical incident.** `expected_sarsa` and `sarsa_vs_q` both had
`gamma: { min: 0.5, max: 0.999, default: 1.0 }`. The browser
clamped to 0.999 on mount — close enough that the bug wasn't
noticed in casual use, but technically the slider was sitting at
its max from the start with no headroom to move up.

### Bug 13: Unseeded `Math.random()` inside `render`

```js
defineWidget({
  controls: { gamma: {...} },
  render: (host, { gamma }, slots) => {
    const noise = d3.range(N).map(() => Math.random() < 0.1 ? 1 : 0);
    // BUG: noise re-rolls on every render. Moving γ slider visibly
    // jitters the noise even though γ has no logical effect on it.
  },
});
```

The user moves a slider that has nothing to do with the noise and
sees the noisy bars jump. They can't tell what's slider-driven and
what's random — interpretation breaks down.

Detection: `Math.random()` called inside `render` (or inside a
helper called from render). Constructor calls outside render are
fine because they run once.

Fix: seed a PRNG once with a fixed seed, or — if the user wants
"resample" interactivity — expose an explicit `seed` slider so the
re-roll happens on a deliberate input change, not on every render.

**Historical incident.** `return_calc`'s "noisy" reward profile
re-rolled Bernoulli noise on every render. Moving γ or T (which
should not change the *raw rewards*) jittered the top chart's
bars. Fixed by seeding the PRNG with a fixed seed inside the
profile generator.

## Audit procedure

For each `docs/textbook/widgets/<name>/widget.js`:

1. Open the file.
2. Find the `controls` object — list every control name.
3. Find the `render` function (and `trajectory` if `defineStepper`).
4. Check that every control name appears in render's body AND feeds
   into a Plot mark / axis / readout.
5. If `slots: [...]` is declared, check every `slots.X` access uses
   one of those names.
6. Scan for any computation outside the render arrow function that
   depends on a slider value.
7. Scan for hardcoded `domain: [low, high]` in Plot configs; verify
   the data range fits.
8. Scan for `Plot.barY` / `Plot.barX` calls; verify the bar's
   categorical axis is band/ordinal (string field, array domain,
   or explicit `type: "band"`). 2-element numeric domain = broken.
9. Scan for `.reverse()` and `.sort(` calls; verify the receiver is
   a fresh array, not a module-level `const`.
10. Scan for `MAX_*` constants ≤ 1000 paired with `delta < <tiny>`
    breaks in loops; check whether a slider can push the contraction
    rate close enough to 1 to peg the cap.
11. Scan polyline data for endpoints that lie outside the chart's
    `y.domain` / `x.domain`; flag the spurious-clipping risk.
12. Scan slider configs for `default` outside `[min, max]`.
13. Scan render bodies for `Math.random()` calls — confirm they're
    inside render only when the widget legitimately wants
    re-roll-on-every-tick behaviour.

A static-analysis script could automate steps 4-13. The source
project's most recent audit pass took ~5 minutes per ~40 widgets
when an agent did it; it found 11 bugs that the prior "zero bugs"
audits had missed (see Historical incident below) because those
earlier passes didn't cover bug classes 8-13.

## When to run the audit

- After every batch of 3+ new widgets.
- Before any release that bumps the live-site version.
- When a user reports "the slider doesn't do anything."

A quick visual spot-check is also worth doing — open each new widget
in a browser, move every slider through its full range, watch the
plot. Anything that LOOKS static is suspect even if the audit says
clean (could be Bug 5 — hardcoded domain).

## Historical incident

The source project ran THREE audits over time. The first two only
covered bug classes 1–7 and reported "zero critical bugs":

- First pass (40 widgets, classes 1–7) — 2 instances of Bug 5
  (`mc_first_vs_every` clipped at y=0.3, `n_step_td` at y=0.55).
- Second pass (40 widgets, computation-correctness focus) — 0
  instances of "values don't update" type bugs. Confirmed the
  framework's per-render-recompute discipline prevents class 3/7
  bugs structurally.

A user then reported in a hands-on session that ~6 widgets visibly
failed to respond to controls. The third pass (classes 8–13 above,
informed by the failure modes the user surfaced) found **11
additional bugs across 11 different widgets**:

- **6 instances of Bug 8** (`Plot.barY` on continuous x):
  `discount_factor`, `return_calc` (2x — both subplots),
  `lambda_return`, `hca_posterior`, `smdp_bootstrap`. All looked
  like "slider does nothing" — the chart frame and axes rendered
  but no bars.
- **1 instance of Bug 9** (`STATE_NAMES.reverse()` mutating shared
  array): `fwd_bwd_equivalence` — episode strip displayed swapped
  state names.
- **2 instances of Bug 10** (convergence cap pegging): `modified_pi`
  and `pi_vs_vi`. Plus 3 milder-precision-loss cases at high γ
  (`pbrs_invariance`, `sr_heatmap`, `reward_shape`).
- **1 instance of Bug 11** (cobweb start outside y-domain):
  `contraction`.
- **2 instances of Bug 12** (default outside slider range):
  `expected_sarsa`, `sarsa_vs_q` (both `γ default 1.0`, max 0.999).
- **1 instance of Bug 13** (unseeded `Math.random()` in render):
  `return_calc`'s `noisy` profile.
- Plus 1 Bug 5 (`stacking_lab` y-clip at 1.1 vs RUDDER Q-curve
  that climbs past it).

**Lesson:** an audit is only as thorough as its bug-class list.
"Zero critical bugs" from the first two passes was true *for the
bug classes those passes checked*. The widget framework guards
against the canonical "control not wired through" class very well
— Bug 1/2/3/4 stayed at zero — but it doesn't guard against
Observable-Plot-API misuse (Bug 8) or JS-runtime-semantics traps
(Bug 9, 13). Future audits should cover all 13 classes.

The static-audit + visual-browser-pass combination is what eventually
caught everything: static analysis found the wiring bugs, the user
exercising widgets in a real browser found the API-misuse and
visual-clipping bugs.

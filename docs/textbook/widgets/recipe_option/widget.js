// Widget 14.G — Recipe-as-option flow (Chapter 14).
//
// The Simulator's own §14.8 mapping made operational: a small recipe
// of primitive-action steps is treated as an option ⟨I, π, β⟩ over a
// 7x7 gridworld. The reader picks:
//
//   * a precondition (`DriveAbove` / `HasItem` / `AnyState`) — sets I_o
//   * a step sequence — π_o is the deterministic walker through it
//   * an option duration (= step-list length) — β fires at exhaustion
//
// SMDP-Q learns over {4 primitives + 1 recipe-option}; the reader can
// disable the recipe to compare. The map highlights the precondition
// region (initiation mask) and the recipe's expected end-state given
// the chosen step list.
//
// This puts §14.8's "Initiation set ↔ precondition" table on-screen —
// the formal options framework is *exactly* what a recipe is.
//
//     <div id="ch14-recipe-option-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/recipe_option/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

const N = 7;
const START = { r: 0, c: 0 };
const GOAL = { r: 6, c: 6 };
const ACTIONS = [
  { dr:  1, dc:  0, name: "↑" }, // up
  { dr:  0, dc:  1, name: "→" }, // right
  { dr: -1, dc:  0, name: "↓" }, // down
  { dr:  0, dc: -1, name: "←" }, // left
];
const NUM_A = ACTIONS.length;
const MAX_STEPS = 150;

// Available "recipe shapes" — short step sequences the reader can
// pick as π_o. Each is a list of primitive-action indices.
// (These mirror the spirit of `recipe.steps` in the Simulator's
// RecipeMeme: a fixed, hand-authored ordering.)
const RECIPES = {
  go_NE:    [0, 1, 0, 1, 0, 1],        // alternate up/right toward NE
  straight_E: [1, 1, 1, 1, 1, 1],      // all-right
  L_shape:  [0, 0, 0, 1, 1, 1, 1, 1],  // up then right
  staircase:[0, 1, 0, 1, 0, 1, 0, 1],  // walks toward goal
};

// Precondition (initiation set I_o).
function inInitiationSet(r, c, kind) {
  if (kind === "any") return true;
  // "DriveAbove": only fireable in the bottom-half (rows 0..3) —
  // mimics a hunger-above-threshold gate.
  if (kind === "drive_above") return r <= 3;
  // "HasItem": only fireable once you've passed the midline col=3 —
  // mimics "carrying a tool".
  if (kind === "has_item") return c >= 3;
  return false;
}

function step(r, c, a) {
  const nr = Math.max(0, Math.min(N - 1, r + ACTIONS[a].dr));
  const nc = Math.max(0, Math.min(N - 1, c + ACTIONS[a].dc));
  return { r: nr, c: nc };
}

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 0x1_0000_0000;
  };
}

function trainSMDP({ epsilon, alpha, gamma, episodes, seed, useRecipe, recipe, preKind }) {
  const rand = rng(seed);
  const steps = RECIPES[recipe];
  const NUM_O = useRecipe ? NUM_A + 1 : NUM_A;
  const RECIPE_O = NUM_A;
  const Q = new Float64Array(N * N * NUM_O);
  const qi = (r, c, o) => (r * N + c) * NUM_O + o;
  const returns = new Array(episodes);

  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c, totalR = 0, t = 0;
    while (t < MAX_STEPS) {
      // Build availability list.
      const avail = [];
      for (let a = 0; a < NUM_A; a++) avail.push(a);
      if (useRecipe && inInitiationSet(r, c, preKind)) avail.push(RECIPE_O);
      // ε-greedy.
      let o;
      if (rand() < epsilon) o = avail[Math.floor(rand() * avail.length)];
      else {
        o = avail[0]; let best = Q[qi(r, c, o)];
        for (const cand of avail) {
          const v = Q[qi(r, c, cand)];
          if (v > best) { best = v; o = cand; }
        }
      }
      // Execute.
      const s0 = { r, c };
      let tau = 0, rCum = 0, done = false;
      if (o < NUM_A) {
        const { r: nr, c: nc } = step(r, c, o);
        tau = 1;
        if (nr === GOAL.r && nc === GOAL.c) { rCum = 1; done = true; }
        r = nr; c = nc;
      } else {
        // Recipe option: deterministic step list (π_o), β = 1 at exhaustion.
        for (const a of steps) {
          const { r: nr, c: nc } = step(r, c, a);
          tau += 1;
          if (nr === GOAL.r && nc === GOAL.c) {
            rCum += Math.pow(gamma, tau - 1) * 1;
            r = nr; c = nc; done = true;
            break;
          }
          r = nr; c = nc;
          if (t + tau >= MAX_STEPS) break;
        }
      }
      totalR += Math.pow(gamma, t) * rCum;
      t += tau;
      let maxNext = 0;
      if (!done) {
        const nextAvail = [];
        for (let a = 0; a < NUM_A; a++) nextAvail.push(a);
        if (useRecipe && inInitiationSet(r, c, preKind)) nextAvail.push(RECIPE_O);
        maxNext = Q[qi(r, c, nextAvail[0])];
        for (const cand of nextAvail) {
          const v = Q[qi(r, c, cand)];
          if (v > maxNext) maxNext = v;
        }
      }
      const target = rCum + Math.pow(gamma, tau) * maxNext;
      Q[qi(s0.r, s0.c, o)] += alpha * (target - Q[qi(s0.r, s0.c, o)]);
      if (done) break;
    }
    returns[ep] = totalR;
  }
  return returns;
}

function smooth(arr, win) {
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const lo = Math.max(0, i - win), hi = Math.min(arr.length, i + win + 1);
    let s = 0;
    for (let k = lo; k < hi; k++) s += arr[k];
    out[i] = s / (hi - lo);
  }
  return out;
}

// Simulate the recipe deterministically from START to show where its
// step list lands (i.e., π_o's trajectory).
function recipeTrajectory(recipe) {
  const steps = RECIPES[recipe];
  let r = START.r, c = START.c;
  const path = [{ r, c }];
  for (const a of steps) {
    const { r: nr, c: nc } = step(r, c, a);
    r = nr; c = nc;
    path.push({ r, c });
    if (r === GOAL.r && c === GOAL.c) break;
  }
  return path;
}

defineWidget({
  hostId: "ch14-recipe-option-widget",
  controls: {
    recipe: {
      type: "select",
      label: "recipe step list (π_o)",
      options: [
        { value: "go_NE",       label: "alternate up/right" },
        { value: "straight_E",  label: "all-right" },
        { value: "L_shape",     label: "L: up×3 then right×5" },
        { value: "staircase",   label: "staircase up-right" },
      ],
      default: "staircase",
    },
    preKind: {
      type: "select",
      label: "precondition (I_o)",
      options: [
        { value: "any",         label: "AnyState — always fireable" },
        { value: "drive_above", label: "DriveAbove — bottom half only" },
        { value: "has_item",    label: "HasItem — right half only" },
      ],
      default: "any",
    },
    alpha:    { label: "α", min: 0.05, max: 0.9, step: 0.05, default: 0.3 },
    gamma:    { label: "γ", min: 0.8, max: 0.999, step: 0.005, default: 0.95 },
    epsilon:  { label: "ε", min: 0.05, max: 0.5, step: 0.05, default: 0.2 },
    episodes: { label: "episodes", min: 50, max: 400, step: 25, default: 150 },
    seed:     { label: "seed", min: 1, max: 50, step: 1, default: 5 },
  },
  slots: ["map", "curves"],
  render: (host, p, slots) => {
    const cfg = {
      epsilon: p.epsilon, alpha: p.alpha, gamma: p.gamma,
      episodes: p.episodes | 0, seed: p.seed | 0,
      recipe: p.recipe, preKind: p.preKind,
    };
    const retNo  = trainSMDP({ ...cfg, useRecipe: false });
    const retYes = trainSMDP({ ...cfg, useRecipe: true  });
    const win = Math.max(5, Math.floor(cfg.episodes / 25));
    const smNo  = smooth(retNo,  win);
    const smYes = smooth(retYes, win);

    // Map: free cells, init-set highlight, start, goal, recipe trace.
    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const init = inInitiationSet(r, c, p.preKind);
        const start = r === START.r && c === START.c;
        const goal  = r === GOAL.r && c === GOAL.c;
        cells.push({ r, c, init, start, goal });
      }
    }
    const path = recipeTrajectory(p.recipe);
    // Pair the path cells into line segments for plotting.
    const lines = [];
    for (let i = 1; i < path.length; i++) {
      lines.push({
        r1: path[i - 1].r, c1: path[i - 1].c,
        r2: path[i].r,     c2: path[i].c,
      });
    }

    slots.map.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      width: 280,
      marginLeft: 12, marginRight: 12, marginTop: 12, marginBottom: 12,
      ...gridAxes(N, { axis: null }),
      marks: [
        Plot.cell(cells, {
          x: "c", y: "r",
          fill: (d) => d.init ? palette.warning : "#1a1a2e",
          fillOpacity: (d) => d.init ? 0.35 : 1,
          inset: 0.5,
        }),
        // Recipe path as line segments — π_o's deterministic trajectory.
        Plot.link(lines, {
          x1: "c1", y1: "r1", x2: "c2", y2: "r2",
          stroke: palette.accent, strokeWidth: 2.5,
        }),
        Plot.dot(path, {
          x: "c", y: "r", fill: palette.accent, r: 3,
        }),
        Plot.text(cells.filter((d) => d.start), {
          x: "c", y: "r", text: () => "S",
          fill: "#fff", fontWeight: "bold", fontSize: 12,
        }),
        Plot.text(cells.filter((d) => d.goal), {
          x: "c", y: "r", text: () => "G",
          fill: palette.primary, fontWeight: "bold", fontSize: 12,
        }),
      ],
    }));

    // Learning curves: with vs. without the recipe option.
    const series = [];
    for (let i = 0; i < cfg.episodes; i++) {
      series.push({ ep: i, ret: smNo[i],  kind: "primitives only" });
      series.push({ ep: i, ret: smYes[i], kind: "primitives + recipe" });
    }
    slots.curves.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true },
      y: { label: "discounted return (smoothed)", grid: true, domain: [0, 1] },
      color: {
        domain: ["primitives only", "primitives + recipe"],
        range: [palette.secondary, palette.primary],
        legend: true,
      },
      marks: [
        Plot.line(series, { x: "ep", y: "ret", stroke: "kind", strokeWidth: 2 }),
      ],
    }));

    const lastK = Math.min(20, cfg.episodes);
    const avgNo  = retNo .slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    const avgYes = retYes.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    const initCount = cells.filter((d) => d.init).length;
    const endsAtGoal = path[path.length - 1].r === GOAL.r && path[path.length - 1].c === GOAL.c;
    slots.readout.textContent =
      `|I_o| = ${initCount}/${N * N} cells  ·  π_o length τ = ${RECIPES[p.recipe].length}  ·  ` +
      `recipe ${endsAtGoal ? "reaches goal" : "stops short"}  ·  ` +
      `final-eps return: prims ${fmt(avgNo)}  vs  prims+recipe ${fmt(avgYes)}`;
  },
});

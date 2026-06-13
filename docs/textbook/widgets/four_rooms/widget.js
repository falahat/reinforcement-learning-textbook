// Widget 14.A — Four-rooms options demo (Chapter 14).
//
// The canonical [Sutton, Precup & Singh 1999] four-rooms gridworld:
// an 11x11 grid partitioned into four 5x5 rooms by horizontal and
// vertical walls. Each room connects to its two neighbours through a
// single "hallway" cell. The goal is in the top-right room; every
// step costs 0 except reaching the goal (+1, terminal).
//
// Two learners run side-by-side:
//   * primitives-only — flat Q-learning over the 4 primitive moves.
//   * options + primitives — same 4 primitives plus 4 hand-engineered
//     "go to hallway N" options (one per hallway). Each option is a
//     hand-coded greedy walker toward its target hallway; β = 1 at the
//     hallway, 0 elsewhere; I_o = the option's source room.
//
// SMDP-Q is applied at the top level (γ^τ-discounted bootstrap when
// an option terminates). The chapter's claim — options accelerate
// learning on long-horizon tasks — shows up as the green (options)
// curve climbing well before the blue (primitives) curve.
//
//     <div id="ch14-four-rooms-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/four_rooms/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

// 11x11 grid. Walls split it into four 5x5 rooms at row=5 and col=5,
// with hallway gaps at the cells listed below. Goal in top-right room.
const N = 11;
const START = { r: 0, c: 0 };
const GOAL = { r: 9, c: 9 };
const ACTIONS = [
  { dr:  1, dc:  0 }, // up
  { dr:  0, dc:  1 }, // right
  { dr: -1, dc:  0 }, // down
  { dr:  0, dc: -1 }, // left
];
const NUM_A = ACTIONS.length;
// Hallway openings in the cross-shaped wall.
const HALLWAYS = [
  { r: 5, c: 2, name: "H_W" }, // bottom-left ↔ top-left
  { r: 5, c: 8, name: "H_E" }, // bottom-right ↔ top-right
  { r: 2, c: 5, name: "H_S" }, // bottom-left ↔ bottom-right
  { r: 7, c: 5, name: "H_N" }, // top-left ↔ top-right
];
const MAX_STEPS = 300;

// Walls live on row=5 and col=5, except at the four hallway cells.
function isWall(r, c) {
  if (r < 0 || r >= N || c < 0 || c >= N) return true;
  if (r === 5 || c === 5) {
    for (const h of HALLWAYS) if (h.r === r && h.c === c) return false;
    return true;
  }
  return false;
}

function roomOf(r, c) {
  if (r < 5 && c < 5) return 0; // bottom-left
  if (r < 5 && c > 5) return 1; // bottom-right
  if (r > 5 && c < 5) return 2; // top-left
  if (r > 5 && c > 5) return 3; // top-right
  return -1; // wall
}

// splitmix32 RNG.
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

function step(r, c, a) {
  const nr = r + ACTIONS[a].dr;
  const nc = c + ACTIONS[a].dc;
  if (isWall(nr, nc)) return { r, c };
  return { r: nr, c: nc };
}

// Greedy controller for "go to hallway target" — moves one step in the
// direction that most reduces L1 distance to the target (respecting
// walls). Deterministic, hand-coded; not learned.
function optionStepToward(r, c, target) {
  const candidates = [];
  for (let a = 0; a < NUM_A; a++) {
    const { r: nr, c: nc } = step(r, c, a);
    if (nr === r && nc === c) continue; // blocked
    const d = Math.abs(nr - target.r) + Math.abs(nc - target.c);
    candidates.push({ a, d });
  }
  if (candidates.length === 0) return 0;
  candidates.sort((x, y) => x.d - y.d);
  return candidates[0].a;
}

// Pick the option(s) initiable in (r, c). An option's initiation set is
// the room it points *out of* — i.e., you can fire "go to hallway H"
// from any cell in a room that H borders.
function optionsAt(r, c) {
  const room = roomOf(r, c);
  const out = [];
  // Each hallway sits between two rooms; from each room the hallway is
  // a valid exit option.
  // (room → hallway indices reachable)
  const ROOM_HALLWAYS = {
    0: [0, 2], // bottom-left ↔ H_W, H_S
    1: [1, 2], // bottom-right ↔ H_E, H_S
    2: [0, 3], // top-left ↔ H_W, H_N
    3: [1, 3], // top-right ↔ H_E, H_N
  };
  if (room in ROOM_HALLWAYS) for (const i of ROOM_HALLWAYS[room]) out.push(i);
  return out;
}

// Train flat Q-learning. ε-greedy over 4 primitives.
function trainFlat({ epsilon, alpha, gamma, episodes, seed }) {
  const rand = rng(seed);
  const Q = new Float64Array(N * N * NUM_A);
  const qi = (r, c, a) => (r * N + c) * NUM_A + a;
  const returns = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c, totalR = 0;
    for (let t = 0; t < MAX_STEPS; t++) {
      let a;
      if (rand() < epsilon) a = Math.floor(rand() * NUM_A);
      else {
        a = 0; let best = Q[qi(r, c, 0)];
        for (let k = 1; k < NUM_A; k++) {
          const v = Q[qi(r, c, k)];
          if (v > best) { best = v; a = k; }
        }
      }
      const { r: nr, c: nc } = step(r, c, a);
      const done = nr === GOAL.r && nc === GOAL.c;
      const reward = done ? 1 : 0;
      totalR += Math.pow(gamma, t) * reward;
      let maxNext = 0;
      if (!done) {
        maxNext = Q[qi(nr, nc, 0)];
        for (let k = 1; k < NUM_A; k++) {
          const v = Q[qi(nr, nc, k)];
          if (v > maxNext) maxNext = v;
        }
      }
      Q[qi(r, c, a)] += alpha * (reward + gamma * maxNext - Q[qi(r, c, a)]);
      r = nr; c = nc;
      if (done) break;
    }
    returns[ep] = totalR;
  }
  return returns;
}

// Train SMDP-Q with both options and primitives. Top-level menu has
// NUM_A primitive options + 4 hallway options = 8 total.
function trainOptions({ epsilon, alpha, gamma, episodes, seed }) {
  const NUM_O = NUM_A + HALLWAYS.length;
  const rand = rng(seed);
  // Q is indexed by (r, c, option). But not every option is initiable
  // everywhere; we still allocate the full table and only argmax over
  // available options at each state.
  const Q = new Float64Array(N * N * NUM_O);
  const qi = (r, c, o) => (r * N + c) * NUM_O + o;
  const returns = new Array(episodes);
  for (let ep = 0; ep < episodes; ep++) {
    let r = START.r, c = START.c, totalR = 0;
    let t = 0;
    while (t < MAX_STEPS) {
      // Available options at this state: primitives (always) + hallway
      // options whose initiation set contains (r, c).
      const avail = [];
      for (let a = 0; a < NUM_A; a++) avail.push(a);
      for (const oi of optionsAt(r, c)) avail.push(NUM_A + oi);

      // ε-greedy over available options.
      let o;
      if (rand() < epsilon) o = avail[Math.floor(rand() * avail.length)];
      else {
        o = avail[0]; let best = Q[qi(r, c, o)];
        for (const cand of avail) {
          const v = Q[qi(r, c, cand)];
          if (v > best) { best = v; o = cand; }
        }
      }

      // Execute the option (or primitive). Track τ and cumulative
      // primitive-discounted reward.
      const s0 = { r, c };
      let tau = 0, rCum = 0, done = false;
      if (o < NUM_A) {
        // Primitive: τ = 1.
        const { r: nr, c: nc } = step(r, c, o);
        tau = 1;
        if (nr === GOAL.r && nc === GOAL.c) { rCum = 1; done = true; }
        r = nr; c = nc;
      } else {
        // Hallway option: walk greedily toward target until you reach
        // it (β = 1) or hit the goal.
        const target = HALLWAYS[o - NUM_A];
        while (true) {
          const a = optionStepToward(r, c, target);
          const { r: nr, c: nc } = step(r, c, a);
          tau += 1;
          if (nr === GOAL.r && nc === GOAL.c) {
            rCum += Math.pow(gamma, tau - 1) * 1;
            done = true;
            r = nr; c = nc;
            break;
          }
          r = nr; c = nc;
          if (r === target.r && c === target.c) break; // β = 1
          if (t + tau >= MAX_STEPS) break; // safety
        }
      }
      totalR += Math.pow(gamma, t) * rCum;
      t += tau;

      // SMDP-Q bootstrap: max over available options at s'.
      let maxNext = 0;
      if (!done) {
        const nextAvail = [];
        for (let a = 0; a < NUM_A; a++) nextAvail.push(a);
        for (const oi of optionsAt(r, c)) nextAvail.push(NUM_A + oi);
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

defineWidget({
  hostId: "ch14-four-rooms-widget",
  controls: {
    epsilon:  { label: "ε (exploration)", min: 0.05, max: 0.5, step: 0.05, default: 0.2 },
    alpha:    { label: "α (step size)",   min: 0.05, max: 0.9, step: 0.05, default: 0.3 },
    gamma:    { label: "γ (discount)",    min: 0.8,  max: 0.999, step: 0.005, default: 0.95 },
    episodes: { label: "episodes",        min: 50,   max: 500, step: 25, default: 200 },
    seed:     { label: "seed",            min: 1,    max: 50,  step: 1, default: 3 },
  },
  slots: ["map", "curves"],
  render: (host, p, slots) => {
    const cfg = {
      epsilon: p.epsilon,
      alpha: p.alpha,
      gamma: p.gamma,
      episodes: p.episodes | 0,
      seed: p.seed | 0,
    };
    const retFlat = trainFlat(cfg);
    const retOpt  = trainOptions(cfg);
    const win = Math.max(5, Math.floor(cfg.episodes / 25));
    const smFlat = smooth(retFlat, win);
    const smOpt  = smooth(retOpt, win);

    // Map: render the four-rooms grid with walls + hallways + start + goal.
    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const wall = isWall(r, c);
        const hall = HALLWAYS.some((h) => h.r === r && h.c === c);
        const start = r === START.r && c === START.c;
        const goal  = r === GOAL.r && c === GOAL.c;
        cells.push({ r, c, wall, hall, start, goal });
      }
    }
    const freeCells = cells.filter((d) => !d.wall);
    const wallCells = cells.filter((d) => d.wall);
    const hallCells = cells.filter((d) => d.hall);

    slots.map.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      width: 280,
      marginLeft: 12, marginRight: 12, marginTop: 12, marginBottom: 12,
      ...gridAxes(N, { axis: null }),
      marks: [
        Plot.cell(freeCells, { x: "c", y: "r", fill: "#1a1a2e", inset: 0.5 }),
        Plot.cell(wallCells, { x: "c", y: "r", fill: palette.muted, inset: 0.5 }),
        Plot.cell(hallCells, { x: "c", y: "r", fill: palette.warning, inset: 0.5 }),
        Plot.text(cells.filter((d) => d.start), {
          x: "c", y: "r", text: () => "S", fill: "#fff", fontWeight: "bold", fontSize: 12,
        }),
        Plot.text(cells.filter((d) => d.goal), {
          x: "c", y: "r", text: () => "G", fill: palette.primary, fontWeight: "bold", fontSize: 12,
        }),
      ],
    }));

    // Curves: smoothed return per episode for both learners.
    const series = [];
    for (let i = 0; i < cfg.episodes; i++) {
      series.push({ ep: i, ret: smFlat[i], kind: "primitives only" });
      series.push({ ep: i, ret: smOpt[i],  kind: "options + primitives" });
    }
    slots.curves.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "episode", grid: true },
      y: { label: "discounted return (smoothed)", grid: true, domain: [0, 1] },
      color: {
        domain: ["primitives only", "options + primitives"],
        range: [palette.secondary, palette.primary],
        legend: true,
      },
      marks: [
        Plot.line(series, { x: "ep", y: "ret", stroke: "kind", strokeWidth: 2 }),
      ],
    }));

    const lastK = Math.min(25, cfg.episodes);
    const avgFlat = retFlat.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    const avgOpt  = retOpt.slice(-lastK).reduce((s, x) => s + x, 0) / lastK;
    slots.readout.textContent =
      `avg discounted return (last ${lastK} eps) — primitives: ${fmt(avgFlat)}  ·  options: ${fmt(avgOpt)}  ·  gain: ${fmt(avgOpt / Math.max(1e-6, avgFlat))}×`;
  },
});

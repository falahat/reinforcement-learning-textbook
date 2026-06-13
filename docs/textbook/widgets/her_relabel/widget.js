// Widget 17.E — HER replay-buffer relabeling animator
// (Chapter 17, §17.4).
//
// A 6×6 goal-conditioned gridworld. The agent starts at (0,0); the
// stated goal g is in the upper-right region. We simulate a *failed*
// episode (deterministic biased random walk that never reaches g) and
// then step through the HER relabeling trick frame by frame:
//
//   - frame 0:    paint the trajectory; goal g shown as a red star.
//                 reward = 0 at every transition (never reached g).
//   - frames 1..T: walk along the trajectory, promoting each visited
//                 state s_t to "goal-of-the-day". The relabeled
//                 transition (s_{t-1}, a, s_t, g' = s_t) has reward 1,
//                 because by construction s_t = g'.
//
// A right-hand panel keeps a running buffer view: positive-example
// count under HER vs. under flat replay. Flat replay sees one episode
// with 0 positives; HER sees T positives.
//
// Mount: in §17.4 of Chapter 17.
//
//     <div id="ch17-her-relabel-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/her_relabel/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import {
  plotDefaults,
  palette,
  dashed,
  annotation,
  fmt,
} from "../shared/helpers.js";
import { lcg } from "../shared/random.js";
import { gridAxes } from "../shared/plot.js";

const GRID = 6;
const START = { x: 0, y: 0 };
const GOAL = { x: 5, y: 5 };

function generateTrajectory(seed, T, bias) {
  // Biased random walk: we want a *failed* episode, so bias slightly
  // AWAY from the goal (down-left). Reader sees the trajectory wander
  // through the lower-left quadrant without ever reaching upper-right.
  const rng = lcg(seed);
  const traj = [{ x: START.x, y: START.y }];
  let { x, y } = START;
  for (let t = 0; t < T; t++) {
    const r = rng();
    let dx = 0, dy = 0;
    // Probabilities: left bias / down bias dominate.
    if (r < 0.5 - bias) dx = -1;
    else if (r < 0.75) dx = +1;
    else if (r < 1 - bias / 2) dy = -1;
    else dy = +1;
    x = Math.max(0, Math.min(GRID - 1, x + dx));
    y = Math.max(0, Math.min(GRID - 1, y + dy));
    // Snap away from g if we accidentally land on it — keeps the
    // "failed" invariant visible.
    if (x === GOAL.x && y === GOAL.y) {
      x = Math.max(0, x - 1);
    }
    traj.push({ x, y });
  }
  return traj;
}

defineStepper({
  hostId: "ch17-her-relabel-widget",
  controls: {
    T:    { label: "trajectory length T", min: 6, max: 20, step: 1, default: 12 },
    bias: { label: "anti-goal bias",      min: 0.05, max: 0.25, step: 0.01, default: 0.15 },
    seed: { label: "seed",                min: 1, max: 32, step: 1, default: 3 },
  },
  slots: ["grid", "buffer"],
  trajectory: (params) => {
    const { T, bias, seed } = params;
    const traj = generateTrajectory(seed, T, bias);
    const frames = [];

    // Frame 0: just show the failed episode.
    frames.push({
      kind: "failed",
      idx: 0,
      relabeledThrough: 0,
      message: `Failed episode: agent never reached goal g = (${GOAL.x}, ${GOAL.y}). All ${T} transitions stored with reward 0 — flat replay sees no positive examples.`,
      traj,
    });

    // One frame per relabeled transition.
    for (let t = 1; t <= T; t++) {
      frames.push({
        kind: "relabel",
        idx: t,
        relabeledThrough: t,
        message:
          `HER: promote s_${t} = (${traj[t].x}, ${traj[t].y}) to goal-of-the-day. ` +
          `Relabeled transition (s_${t - 1} → s_${t}, g' = s_${t}) has reward 1.`,
        traj,
      });
    }

    return frames;
  },
  render: (host, frame, idx, total, params, slots) => {
    const { traj, relabeledThrough, kind, message } = frame;
    const T = traj.length - 1;

    // --- left: gridworld + trajectory ---
    const cells = [];
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) cells.push({ x, y });
    }
    // Trajectory segments (line) and dots; relabeled goals along the
    // way get a star marker.
    const pathSoFar = traj.slice(0, kind === "relabel" ? relabeledThrough + 1 : traj.length);
    const relabeledGoals = kind === "relabel"
      ? traj.slice(1, relabeledThrough + 1).map((p, i) => ({ ...p, t: i + 1 }))
      : [];
    const currentGoal = kind === "relabel" ? traj[relabeledThrough] : null;

    slots.grid.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320, width: 320,
      marginTop: 32,
      title: kind === "failed" ? "failed episode (flat replay)" : "HER relabel walk",
      ...gridAxes(GRID, { label: null, axis: null }, { y: { reverse: true } }),
      aspectRatio: 1,
      marks: [
        Plot.cell(cells, {
          x: "x", y: "y",
          fill: palette.muted, fillOpacity: 0.08,
          stroke: palette.muted, strokeOpacity: 0.3,
        }),
        // Trajectory polyline + dots.
        Plot.line(traj, { x: "x", y: "y", stroke: palette.secondary, strokeWidth: 1, strokeOpacity: 0.5 }),
        Plot.line(pathSoFar, { x: "x", y: "y", stroke: palette.primary, strokeWidth: 2 }),
        Plot.dot(traj, { x: "x", y: "y", fill: palette.secondary, r: 2, fillOpacity: 0.4 }),
        Plot.dot(pathSoFar, { x: "x", y: "y", fill: palette.primary, r: 3 }),
        // Start.
        Plot.dot([START], { x: "x", y: "y", fill: palette.warning, r: 6, stroke: "white" }),
        Plot.text([START], { x: "x", y: "y", text: () => "s₀", dy: -10, fontSize: 11, fill: palette.warning }),
        // Intended goal g (always shown).
        Plot.text([GOAL], { x: "x", y: "y", text: () => "★", fontSize: 18, fill: palette.danger }),
        Plot.text([GOAL], { x: "x", y: "y", text: () => "g", dy: 14, fontSize: 11, fill: palette.danger }),
        // Relabeled goals so far (smaller stars).
        Plot.text(relabeledGoals, {
          x: "x", y: "y", text: () => "✦",
          fontSize: 12, fill: palette.accent,
        }),
        // Current relabel highlight.
        ...(currentGoal
          ? [
              Plot.dot([currentGoal], {
                x: "x", y: "y",
                r: 9, stroke: palette.accent, strokeWidth: 2, fill: "none",
              }),
              Plot.text([currentGoal], {
                x: "x", y: "y", text: () => `g' = s_${relabeledThrough}`,
                dy: -12, fontSize: 10, fill: palette.accent,
              }),
            ]
          : []),
      ],
    }));

    // --- right: buffer composition bars ---
    const herPositives = kind === "relabel" ? relabeledThrough : 0;
    const flatPositives = 0;
    const flatNegatives = T;
    const herNegatives = T - herPositives;

    const bars = [
      { kind: "flat replay", label: "positives", n: flatPositives, color: palette.primary },
      { kind: "flat replay", label: "zero-r", n: flatNegatives, color: palette.muted },
      { kind: "HER",         label: "positives", n: herPositives, color: palette.primary },
      { kind: "HER",         label: "zero-r", n: herNegatives, color: palette.muted },
    ];

    slots.buffer.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      marginLeft: 90,
      x: { label: "examples in buffer (this episode)", grid: true },
      y: { label: null, domain: ["flat replay", "HER"] },
      color: { legend: false },
      marks: [
        Plot.barX(bars, {
          y: "kind", x: "n", fill: "label",
          fillOpacity: 0.9,
        }),
        Plot.text(bars, {
          y: "kind", x: "n",
          text: (d) => d.n > 0 ? `${d.label}: ${d.n}` : "",
          textAnchor: "start", dx: 4, fontSize: 10, fill: palette.muted,
        }),
      ],
    }));

    slots.readout.innerHTML =
      `<small>step ${idx + 1} / ${total} · ${message}</small>`;
  },
  playIntervalMs: 700,
});

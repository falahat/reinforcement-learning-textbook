// Widget 14.E — HIRO subgoal-relabeling demo.
//
// High-level policy emits subgoals g_t every k steps on a 2D plane.
// Low-level policy tries to reach g_t but drifts (imperfect goal-
// following). After the episode, HIRO relabels each subgoal with the
// state the low-level policy ACTUALLY reached — making off-policy
// updates of the high level consistent with what the low level can
// actually do.
//
// Visual: red dots = original subgoals; orange dots = actual reached
// states; green dots = relabeled subgoals (= reached states). Dashed
// lines connect each original subgoal to what was reached.
//
// Mount:
//     <div id="ch14-hiro-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/hiro/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const K = 5;          // segments
const STEPS_PER = 10; // low-level steps between subgoals

// Pre-compute a canonical "episode": fixed subgoal sequence + fixed
// low-level drift trajectory for a given seed/noise.
function buildEpisode(seed, noise, follow) {
  const rng = mulberry32(Math.round(seed));
  // Subgoals: a deliberate pattern (zig-zag toward upper right).
  const subgoals = [];
  for (let i = 0; i < K; i++) {
    const x = -1 + 2 * (i / (K - 1));
    const y = ((i % 2 === 0) ? -0.6 : 0.6) + 0.15 * (i / K);
    subgoals.push({ x, y });
  }
  // Low-level rollout: between subgoal i-1 and i, drift from current
  // pos toward subgoal with `follow` strength + gauss noise.
  let pos = { x: -1.0, y: 0.0 };
  const reached = [];
  const lowTraj = [{ x: pos.x, y: pos.y, seg: -1 }];
  for (let i = 0; i < K; i++) {
    const g = subgoals[i];
    for (let s = 0; s < STEPS_PER; s++) {
      pos = {
        x: pos.x + follow * (g.x - pos.x) / STEPS_PER + noise * gauss(rng),
        y: pos.y + follow * (g.y - pos.y) / STEPS_PER + noise * gauss(rng),
      };
      lowTraj.push({ x: pos.x, y: pos.y, seg: i });
    }
    reached.push({ x: pos.x, y: pos.y });
  }
  // Relabeled subgoals = reached states.
  const relabeled = reached.map((r) => ({ ...r }));
  return { subgoals, reached, relabeled, lowTraj };
}

defineStepper({
  hostId: "ch14-hiro-widget",
  controls: {
    noise:  { label: "low-level noise σ", min: 0.0, max: 0.30, step: 0.01, default: 0.08 },
    follow: { label: "follow strength",   min: 0.2, max: 2.0,  step: 0.05, default: 0.7 },
    seed:   { label: "seed",              min: 1,   max: 999,  step: 1,    default: 5 },
  },
  // Frames: 0 = subgoals only; 1..K = roll out segment i; K+1 = full
  // trajectory; K+2 = show relabeled (green) subgoals; K+3 = relabel
  // overlay (dashed correspondence lines).
  trajectory: (params) => {
    const ep = buildEpisode(params.seed, params.noise, params.follow);
    const frames = [];
    frames.push({ phase: "subgoals", ep, upTo: 0 });
    for (let i = 1; i <= K; i++) {
      frames.push({ phase: "rollout", ep, upTo: i });
    }
    frames.push({ phase: "rollout_done", ep, upTo: K });
    frames.push({ phase: "relabel_show", ep, upTo: K });
    frames.push({ phase: "relabel_lines", ep, upTo: K });
    return frames;
  },
  render: (host, frame, idx, total, params, slots) => {
    const { ep, phase, upTo } = frame;

    // Subgoals (red) — always shown when relevant.
    const showOrig = (phase !== "relabel_show" && phase !== "relabel_lines")
      ? ep.subgoals
      : ep.subgoals; // still show under relabel phases (with dashed lines)

    // Reached / relabeled (orange / green): only after rollout reaches them.
    const reachedSoFar = ep.reached.slice(0, upTo);
    const lowSoFar = ep.lowTraj.filter((p) => p.seg < upTo);

    const showRelabeled =
      (phase === "relabel_show" || phase === "relabel_lines")
        ? ep.relabeled : [];

    const correspond = (phase === "relabel_lines")
      ? ep.subgoals.map((s, i) => ([
          { x: s.x, y: s.y, pair: i },
          { x: ep.relabeled[i].x, y: ep.relabeled[i].y, pair: i },
        ])).flat()
      : [];

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 360,
      marginLeft: 50, marginBottom: 36,
      x: { label: "x", domain: [-1.5, 1.5], grid: true },
      y: { label: "y", domain: [-1.2, 1.2], grid: true },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        // Low-level trajectory drawn as a connected polyline.
        Plot.line(lowSoFar, {
          x: "x", y: "y", stroke: palette.warning,
          strokeWidth: 1.5, strokeOpacity: 0.55,
        }),
        // Dashed lines connecting original ↔ relabeled subgoals.
        Plot.line(correspond, {
          x: "x", y: "y", z: "pair", stroke: palette.muted,
          strokeWidth: 1, ...dashed,
        }),
        // Subgoals (red). Add small index labels.
        Plot.dot(showOrig, {
          x: "x", y: "y", fill: palette.danger, stroke: palette.danger,
          r: 7, fillOpacity: 0.7,
        }),
        Plot.text(showOrig.map((p, i) => ({ ...p, i: `g${i}` })), {
          x: "x", y: "y", text: "i", fill: "#fff",
          fontSize: 9, textAnchor: "middle",
        }),
        // Reached (orange).
        Plot.dot(reachedSoFar, {
          x: "x", y: "y", fill: palette.warning, stroke: palette.warning,
          r: 6, fillOpacity: 0.85,
        }),
        // Relabeled (green) — overlay on top of reached when shown.
        Plot.dot(showRelabeled, {
          x: "x", y: "y", fill: palette.primary, stroke: palette.primary,
          r: 5, fillOpacity: 0.9,
        }),
      ],
    }));

    // Legend / phase line.
    let phaseText;
    if (phase === "subgoals") {
      phaseText = "Step 1: high-level policy emits subgoals (red).";
    } else if (phase === "rollout") {
      phaseText = `Step 2: low-level rollout, segment ${upTo}/${K} (orange = reached).`;
    } else if (phase === "rollout_done") {
      phaseText = "Step 3: episode complete; reached states (orange) differ from subgoals (red).";
    } else if (phase === "relabel_show") {
      phaseText = "Step 4: relabel each subgoal with its reached state (green).";
    } else {
      phaseText = "Step 5: training pairs (red → green) are what HIRO actually replays.";
    }
    slots.readout.innerHTML =
      `<strong>frame ${idx + 1} / ${total}</strong> &nbsp; ` +
      `<small>${phaseText}</small>`;
  },
  playIntervalMs: 900,
});

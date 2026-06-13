// Widget 13.B — Dyna-Q planning steps (Chapter 13).
//
// FrozenLake-style 5x5 grid with a single goal cell at the top-right.
// Each real step gets one Q-update; then `n_plan` extra Q-updates
// from the learned model. The stepper animates one real step at a
// time so the reader can watch:
//
//   - the agent's trajectory through the grid,
//   - the Q-table heatmap (V(s) = max_a Q(s, a)) updating,
//   - the learning curve (steps-to-goal per episode) and current
//     episode count.
//
// Slider `n_plan` lets the reader compare convergence under
// n ∈ {0, 1, 5, 20, 50}. The stepper trajectory recomputes whenever
// n_plan changes.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-dyna-q-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/dyna_q/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const N = 5;
const NUM_S = N * N;
const ACTIONS = [
  { dx: 0, dy: -1, name: "↑" },
  { dx: 0, dy: +1, name: "↓" },
  { dx: -1, dy: 0, name: "←" },
  { dx: +1, dy: 0, name: "→" },
];
const NUM_A = ACTIONS.length;
const GOAL_X = N - 1;
const GOAL_Y = 0;
const GOAL_S = GOAL_Y * N + GOAL_X;
const START_S = (N - 1) * N + 0; // bottom-left
const GAMMA = 0.95;
const ALPHA = 0.5;
const EPS = 0.15;
const MAX_REAL_STEPS = 300;

function nextState(s, a) {
  if (s === GOAL_S) return s;
  const x = s % N;
  const y = Math.floor(s / N);
  const nx = Math.min(N - 1, Math.max(0, x + ACTIONS[a].dx));
  const ny = Math.min(N - 1, Math.max(0, y + ACTIONS[a].dy));
  return ny * N + nx;
}
function reward(s, a, sp) {
  return s !== GOAL_S && sp === GOAL_S ? 1.0 : 0.0;
}

defineStepper({
  hostId: "ch13-dyna-q-widget",
  controls: {
    nPlan: { label: "n_plan (planning steps)", min: 0, max: 50, step: 1, default: 5 },
  },
  slots: ["grid", "curve"],
  trajectory: ({ nPlan }) => {
    const rng = mulberry32(13);
    const Q = [];
    for (let s = 0; s < NUM_S; s++) {
      Q.push(new Array(NUM_A).fill(0));
    }
    const model = [];
    for (let s = 0; s < NUM_S; s++) {
      model.push(new Array(NUM_A).fill(null));
    }
    const visited = [];
    const visitedSet = new Set();

    const frames = [];
    // Frame 0: initial state.
    frames.push({
      step: 0,
      episode: 0,
      s: START_S,
      pickedA: -1,
      Vsnap: new Array(NUM_S).fill(0),
      episodeLens: [],
      stepsThisEpisode: 0,
    });

    let s = START_S;
    let episode = 1;
    let stepsInEp = 0;
    const episodeLens = [];

    for (let step = 1; step <= MAX_REAL_STEPS; step++) {
      // ε-greedy
      let a;
      if (rng() < EPS) {
        a = Math.floor(rng() * NUM_A);
      } else {
        let bestA = 0;
        let bestQ = -Infinity;
        for (let aa = 0; aa < NUM_A; aa++) {
          if (Q[s][aa] > bestQ) { bestQ = Q[s][aa]; bestA = aa; }
        }
        a = bestA;
      }
      const sp = nextState(s, a);
      const r = reward(s, a, sp);
      const tgt = r + (sp === GOAL_S ? 0 : GAMMA * Math.max(...Q[sp]));
      Q[s][a] += ALPHA * (tgt - Q[s][a]);
      model[s][a] = { sp, r };
      const key = s * NUM_A + a;
      if (!visitedSet.has(key)) {
        visitedSet.add(key);
        visited.push({ s, a });
      }
      // Planning
      for (let p = 0; p < nPlan && visited.length > 0; p++) {
        const pick = visited[Math.floor(rng() * visited.length)];
        const m = model[pick.s][pick.a];
        const tgtP = m.r + (m.sp === GOAL_S ? 0 : GAMMA * Math.max(...Q[m.sp]));
        Q[pick.s][pick.a] += ALPHA * (tgtP - Q[pick.s][pick.a]);
      }
      stepsInEp += 1;

      const V = Q.map((q) => Math.max(...q));
      const isTerminal = sp === GOAL_S;
      let nextS = sp;
      if (isTerminal) {
        episodeLens.push(stepsInEp);
        nextS = START_S;
      }

      frames.push({
        step,
        episode,
        s: nextS,
        prevS: s,
        pickedA: a,
        Vsnap: V.slice(),
        episodeLens: episodeLens.slice(),
        stepsThisEpisode: isTerminal ? 0 : stepsInEp,
        terminalReached: isTerminal,
      });

      if (isTerminal) {
        episode += 1;
        stepsInEp = 0;
      }
      s = nextS;

      // Cap frames after enough episodes to keep the stepper responsive.
      if (episodeLens.length >= 12 && step >= 60) break;
    }

    return frames;
  },
  playIntervalMs: 120,
  render: (host, frame, idx, total, params, slots) => {
    const { Vsnap, s, prevS, pickedA, episodeLens, episode, stepsThisEpisode, step } = frame;
    const cells = [];
    for (let i = 0; i < NUM_S; i++) {
      cells.push({
        x: i % N,
        y: Math.floor(i / N),
        v: Vsnap[i],
        isGoal: i === GOAL_S,
        isAgent: i === s,
        isPrev: prevS !== undefined && i === prevS,
      });
    }
    const vMax = Math.max(d3.max(Vsnap), 0.01);

    slots.grid.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      width: 280,
      marginLeft: 4,
      marginRight: 4,
      marginBottom: 4,
      marginTop: 4,
      x: { axis: null, domain: d3.range(N) },
      y: { axis: null, domain: d3.range(N) },
      color: {
        type: "linear",
        domain: [0, vMax],
        range: ["#1b3a52", "#8bc34a"],
      },
      marks: [
        Plot.cell(cells, {
          x: "x",
          y: "y",
          fill: (d) => (d.isGoal ? null : d.v),
          stroke: "#111",
          strokeOpacity: 0.4,
        }),
        Plot.cell(cells.filter((c) => c.isGoal), {
          x: "x",
          y: "y",
          fill: palette.warning,
          stroke: palette.danger,
          strokeWidth: 2,
        }),
        Plot.cell(cells.filter((c) => c.isAgent && !c.isGoal), {
          x: "x",
          y: "y",
          fill: "none",
          stroke: palette.accent,
          strokeWidth: 3,
        }),
        Plot.text(cells, {
          x: "x",
          y: "y",
          text: (d) => (d.isGoal ? "G" : (d.isAgent ? "●" : d.v.toFixed(2))),
          fill: "white",
          fontSize: 10,
        }),
      ],
    }));

    // Learning curve.
    const curveData = episodeLens.map((len, i) => ({ ep: i + 1, len }));
    const yMax = Math.max(d3.max(curveData, (d) => d.len) ?? 1, 1);
    slots.curve.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "episode", grid: true, domain: [1, Math.max(12, curveData.length)] },
      y: { label: "steps to goal", grid: true, domain: [0, yMax * 1.15] },
      marks: [
        Plot.line(curveData, {
          x: "ep",
          y: "len",
          stroke: palette.primary,
          strokeWidth: 2,
        }),
        Plot.dot(curveData, {
          x: "ep",
          y: "len",
          fill: palette.primary,
          r: 3,
        }),
        Plot.ruleY([N - 1 + N - 1], { stroke: palette.muted, ...dashed }),
        Plot.text(
          [{ x: Math.max(12, curveData.length), y: N - 1 + N - 1, label: "optimal" }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    const meanLen = curveData.length
      ? d3.mean(curveData, (d) => d.len).toFixed(1)
      : "—";
    slots.readout.innerHTML =
      `real step ${step} · episode ${episode} · this ep step ${stepsThisEpisode} ` +
      `· n_plan = ${params.nPlan}<br>` +
      `<small>completed episodes: ${curveData.length} · ` +
      `mean steps-to-goal: ${meanLen}</small>`;
  },
});

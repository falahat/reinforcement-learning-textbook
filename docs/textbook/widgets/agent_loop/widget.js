// Widget 2.A — Agent-environment loop animator (Chapter 2).
//
// Walks through a short trajectory on a tiny line-world env (4 cells:
// 0 = start, 3 = goal +1). Each frame shows the (s_t, a_t, r_{t+1},
// s_{t+1}) tuple as four boxes. The Markov toggle lets the student
// flip between (a) a Markov state (just the position) and (b) a
// non-Markov state (position seen WITHOUT direction-of-arrival,
// which the agent in this env actually needs).
//
// Pedagogy: the stepper makes "one step of the loop = one tuple"
// literal. The Markov checkbox highlights that a state is only Markov
// relative to a chosen representation.
//
// Pattern: chapter markdown contains
//
//     <div id="ch2-agent-loop-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/agent_loop/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette } from "../shared/helpers.js";

// A 4-cell chain: positions 0,1,2,3. Goal = 3 (+1 reward). Other
// transitions = 0 reward. The pre-baked trajectory is the same for
// both Markov modes — only the rendered state label changes.
const TRAJ = [
  { s: 0, a: "→", r: 0, s_next: 1 },
  { s: 1, a: "→", r: 0, s_next: 2 },
  { s: 2, a: "←", r: 0, s_next: 1 }, // a deliberate backtrack
  { s: 1, a: "→", r: 0, s_next: 2 },
  { s: 2, a: "→", r: 1, s_next: 3 }, // goal!
];

// Non-Markov state label: "pos + last-direction". An agent that only
// sees position can't distinguish "arrived at cell 1 from the left"
// from "arrived from the right" — yet that history matters if the
// dynamics depend on momentum (a standard POMDP gotcha).
function labelState(idx, useMarkov) {
  if (useMarkov) return `s=${TRAJ[idx]?.s ?? "·"}`;
  const prev = idx === 0 ? "·" : TRAJ[idx - 1].a;
  return `s=${TRAJ[idx]?.s ?? "·"}, last=${prev}`;
}

function labelNextState(idx, useMarkov) {
  if (useMarkov) return `s'=${TRAJ[idx].s_next}`;
  return `s'=${TRAJ[idx].s_next}, last=${TRAJ[idx].a}`;
}

defineStepper({
  hostId: "ch2-agent-loop-widget",
  controls: {
    markov: {
      type: "select",
      label: "state representation",
      options: [
        { value: "markov", label: "Markov (position only)" },
        { value: "history", label: "non-Markov (pos + last action)" },
      ],
      default: "markov",
    },
  },
  trajectory: () => TRAJ.map((_, i) => ({ idx: i })),
  render: (host, { idx }, frameIdx, total, params, slots) => {
    const useMarkov = params.markov === "markov";
    const step = TRAJ[idx];

    // Build a tiny 4-box "tuple card" via Plot rects + text.
    // Layout: 4 columns × 1 row. Cols are S, A, R, S'.
    const cells = [
      { col: 0, label: "state", value: labelState(idx, useMarkov), fill: palette.secondary },
      { col: 1, label: "action", value: step.a, fill: palette.primary },
      { col: 2, label: "reward", value: step.r > 0 ? `+${step.r}` : `${step.r}`,
        fill: step.r > 0 ? palette.warning : palette.muted },
      { col: 3, label: "next state", value: labelNextState(idx, useMarkov), fill: palette.accent },
    ];

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 160,
      marginLeft: 0,
      marginBottom: 0,
      marginTop: 30,
      x: { axis: null, domain: [-0.5, 3.5] },
      y: { axis: null, domain: [0, 2] },
      marks: [
        Plot.rect(cells, {
          x1: (d) => d.col - 0.4, x2: (d) => d.col + 0.4,
          y1: 0.4, y2: 1.6,
          fill: "fill", fillOpacity: 0.55, stroke: (d) => d.fill, strokeWidth: 2,
        }),
        Plot.text(cells, {
          x: "col", y: 1.85, text: "label",
          fill: "#ccc", fontSize: 11, textAnchor: "middle",
        }),
        Plot.text(cells, {
          x: "col", y: 1.0, text: "value",
          fill: "#fff", fontSize: 14, fontWeight: "bold", textAnchor: "middle",
        }),
      ],
    }));

    // Sub-plot: the line-world positions, showing s_t and s_{t+1}.
    const cells2 = [0, 1, 2, 3].map((p) => ({
      pos: p,
      role:
        p === step.s ? "s_t"
        : p === step.s_next ? "s_{t+1}"
        : p === 3 ? "goal"
        : "",
      isHere: p === step.s,
      isNext: p === step.s_next,
    }));
    slots.world.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 90,
      marginLeft: 30,
      marginBottom: 24,
      x: { label: "position", domain: [-0.5, 3.5], tickFormat: (d) => `${d}` },
      y: { axis: null, domain: [0, 1] },
      marks: [
        Plot.rect(cells2, {
          x1: (d) => d.pos - 0.45, x2: (d) => d.pos + 0.45,
          y1: 0.3, y2: 0.7,
          fill: (d) => d.pos === 3 ? palette.warning : "#222",
          fillOpacity: 0.4,
          stroke: (d) => d.isHere ? palette.secondary : d.isNext ? palette.accent : "#444",
          strokeWidth: (d) => (d.isHere || d.isNext) ? 2.5 : 1,
        }),
        Plot.text(cells2, {
          x: "pos", y: 0.5, text: "pos",
          fill: "#fff", fontSize: 12, textAnchor: "middle",
        }),
        Plot.text(cells2.filter((d) => d.role), {
          x: "pos", y: 0.92, text: "role",
          fill: (d) => d.isHere ? palette.secondary : d.isNext ? palette.accent : palette.muted,
          fontSize: 10, textAnchor: "middle",
        }),
      ],
    }));

    const modeNote = useMarkov
      ? "Markov: state = position only. P(s_{t+1}|s_t, a_t) is well-defined."
      : "Non-Markov: state includes last action — needed if dynamics depend on history.";
    slots.readout.innerHTML =
      `<strong>step ${frameIdx + 1} / ${total}</strong> &nbsp; ` +
      `tuple <code>(s, a, r, s')</code> = ` +
      `(${step.s}, ${step.a}, ${step.r}, ${step.s_next}) &nbsp; ` +
      `<small>${modeNote}</small>`;
  },
  slots: ["main", "world"],
  playIntervalMs: 900,
});

// Widget 6.E — n-step TD spectrum (Chapter 6).
//
// The S&B Ch 6 Ex 1 random walk: 5 non-terminal states A..E in a chain
// 0—A—B—C—D—E—1 with terminal rewards 0 (left) and 1 (right). The
// behaviour policy is uniform-random — equal chance of stepping left or
// right — so the true value of state i (i = 1..5) is i/6 (linear).
//
// We run n-step TD for n ∈ {1, 2, 4, 8, 16, ∞} (where ∞ ≡ Monte Carlo —
// we cap at trajectory length, so for this 7-state chain any n above ~30
// is functionally MC). For each n, plot RMS error against the analytic
// V* = (1/6 ... 5/6) vs episode count.
//
// The "U-shape" emerges across the n axis: small n bootstraps from
// initialised V → biased early but low variance; large n waits for the
// full return → unbiased but high variance. Around n = 4 sits the
// empirical sweet spot for this env.
//
//     <div id="ch6-nstep-td-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/n_step_td/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

// State indexing: 0 = left terminal, 1..N = non-terminals, N+1 = right
// terminal. With N = 5, V*(i) = i / (N+1) for i ∈ {1..N}.
const N = 5;
const N_STATES = N + 2;
const TERMINAL_L = 0;
const TERMINAL_R = N + 1;
const START = 3; // middle state (S&B convention)
const V_TRUE = Array.from({ length: N }, (_, i) => (i + 1) / (N + 1));

const N_VALUES = [1, 2, 4, 8, 16, 1024]; // 1024 = MC for any plausible episode length
const N_LABELS = ["n=1", "n=2", "n=4", "n=8", "n=16", "n=∞ (MC)"];
const N_COLORS = [
  palette.primary,
  palette.secondary,
  palette.warning,
  palette.danger,
  palette.accent,
  palette.muted,
];

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
function rmsError(V) {
  let s = 0;
  for (let i = 0; i < N; i++) {
    const d = V[i + 1] - V_TRUE[i];
    s += d * d;
  }
  return Math.sqrt(s / N);
}

// One trajectory under uniform-random walk, terminating at 0 or N+1.
function sampleEpisode(rand) {
  const states = [START];
  const rewards = [];
  let s = START;
  // Bounded by ~6N expected length for a uniform random walk on this
  // chain; cap at 4000 to be safe (return-to-origin recurrence is null
  // in the line graph but with absorbing barriers it terminates a.s.).
  for (let t = 0; t < 4000; t++) {
    s = (rand() < 0.5) ? s - 1 : s + 1;
    states.push(s);
    rewards.push(s === TERMINAL_R ? 1 : 0); // only the right end pays out
    if (s === TERMINAL_L || s === TERMINAL_R) return { states, rewards };
  }
  return { states, rewards };
}

// n-step TD applied to one episode. The textbook update is
//
//   G_{t:t+n} = R_{t+1} + R_{t+2} + ... + R_{t+n} + V(S_{t+n})   (γ = 1)
//   V(S_t) ← V(S_t) + α [G_{t:t+n} − V(S_t)]
//
// For an episode of length T (T = rewards.length), at time τ = t the
// horizon is min(t + n, T). If horizon < T we bootstrap from
// V(S_{t+n}); if horizon = T we hit a terminal and there's no bootstrap.
function nStepTD(V, episode, n, alpha) {
  const { states, rewards } = episode;
  const T = rewards.length;
  for (let t = 0; t < T; t++) {
    const horizon = Math.min(t + n, T);
    let G = 0;
    for (let k = t; k < horizon; k++) G += rewards[k]; // γ = 1
    if (horizon < T) {
      // bootstrap from V(S_{t+n}); terminals already carry V = 0 by
      // construction (V is N_STATES long, V[0] = V[N+1] = 0 always).
      G += V[states[horizon]];
    }
    const s = states[t];
    if (s !== TERMINAL_L && s !== TERMINAL_R) {
      V[s] += alpha * (G - V[s]);
    }
  }
}

defineWidget({
  hostId: "ch6-nstep-td-widget",
  controls: {
    alpha:    { label: "α (step size)", min: 0.02, max: 0.6,  step: 0.02, default: 0.1 },
    episodes: { label: "episodes",      min: 20,   max: 300,  step: 10,   default: 100 },
    runs:     { label: "averaging runs", min: 1,   max: 30,   step: 1,    default: 10 },
    seed:     { label: "seed",          min: 1,    max: 50,   step: 1,    default: 7 },
  },
  slots: ["main"],
  render: (host, p, slots) => {
    const episodes = p.episodes | 0;
    const runs = p.runs | 0;

    // For each n, average RMS-vs-episodes across `runs` independent
    // seeds. V is initialised to 0.5 (S&B convention — sits exactly
    // halfway between the two terminals, equidistant from the truth).
    const curves = N_VALUES.map(() => new Float64Array(episodes));
    for (let run = 0; run < runs; run++) {
      const rand = rng((p.seed | 0) + run * 1009);
      // Pre-sample episodes once per run; all n share the same
      // trajectories so the n-axis comparison is apples-to-apples.
      const eps = Array.from({ length: episodes }, () => sampleEpisode(rand));
      for (let ni = 0; ni < N_VALUES.length; ni++) {
        const n = N_VALUES[ni];
        const V = new Float64Array(N_STATES);
        for (let i = 1; i <= N; i++) V[i] = 0.5;
        for (let e = 0; e < episodes; e++) {
          nStepTD(V, eps[e], n, p.alpha);
          curves[ni][e] += rmsError(V);
        }
      }
    }
    for (let ni = 0; ni < N_VALUES.length; ni++) {
      for (let e = 0; e < episodes; e++) curves[ni][e] /= runs;
    }

    const rows = [];
    for (let ni = 0; ni < N_VALUES.length; ni++) {
      for (let e = 0; e < episodes; e++) {
        rows.push({ ep: e + 1, rms: curves[ni][e], n: N_LABELS[ni] });
      }
    }
    // Final-episode RMS per n — the U-shape is most visible in this
    // single slice, so we tabulate it in the readout.
    const finals = N_VALUES.map((_, ni) => curves[ni][episodes - 1]);
    const bestNi = finals.indexOf(Math.min(...finals));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      x: { label: "episode", grid: true },
      // Auto-fit y so high-α curves (which can exceed 0.55 early on)
      // aren't visually clipped at the top of the chart.
      y: { label: "RMS error vs V*", zero: true, grid: true },
      color: { domain: N_LABELS, range: N_COLORS, legend: true },
      marks: [
        Plot.line(rows, { x: "ep", y: "rms", stroke: "n", strokeWidth: 1.8 }),
      ],
    }));

    const summary = N_LABELS.map((lbl, i) =>
      `${lbl}=${fmt(finals[i])}${i === bestNi ? "★" : ""}`
    ).join("  ");
    slots.readout.textContent =
      `final RMS · ${summary}  (★ = best for this α; the U bottoms here)`;
  },
});

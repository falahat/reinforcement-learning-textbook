// Widget 12.E — Sparse-reward random walk (Chapter 12).
//
// A 1-D chain of N states with reward only at state N-1 (the right end).
// A random-walking agent (epsilon-greedy with no useful Q signal — so
// effectively uniform-random) starts at the middle. The widget plots
// the closed-form probability that the agent has reached the goal by
// step t (gambler's-ruin style hitting time), for several values of N
// and a slider over the step budget T.
//
// The intuition: hitting-time on a 1-D random walk of length n with
// absorbing boundaries scales as n^2; double the chain and the budget
// quadruples. The pedagogy ports straight to "depth-d random
// exploration succeeds with prob 1 / b^d" but on a 1-D toy that lets
// the reader read the curve.
//
// We approximate P(hit by step t) by simulating M short rollouts in JS
// rather than building the full hitting-time PMF (which would need
// arc-sine arithmetic). With M = 400 and t up to 2000 the widget stays
// responsive (~ms per scrub).
//
// Mount:
//     <div id="ch12-sparse-walk-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/sparse_reward_walk/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { splitmix64 as makeRng } from "../shared/random.js";

const M_ROLLOUTS = 400;
const CHAIN_LENGTHS = [10, 20, 40];
const COLOURS = [palette.primary, palette.secondary, palette.warning];

// One Monte-Carlo estimate of P(hit by step t) for chain length n with
// absorbing right wall at n-1, reflecting left wall at 0, agent starts
// in the middle. Returns an array [hits_by_t for t = 0..tMax].
function hitProbCurve(n, tMax, rngSeed) {
  const rng = makeRng(rngSeed);
  const hitsAt = new Array(tMax + 1).fill(0);
  for (let r = 0; r < M_ROLLOUTS; r++) {
    let pos = Math.floor(n / 2);
    let hit = -1;
    for (let t = 0; t <= tMax; t++) {
      if (pos === n - 1) { hit = t; break; }
      // Reflecting left wall, absorbing right wall.
      const step = rng() < 0.5 ? -1 : 1;
      pos += step;
      if (pos < 0) pos = 0;
    }
    if (hit >= 0) hitsAt[hit] += 1;
  }
  // Convert per-step hits into cumulative P(hit by t).
  const cdf = new Array(tMax + 1).fill(0);
  let running = 0;
  for (let t = 0; t <= tMax; t++) {
    running += hitsAt[t];
    cdf[t] = running / M_ROLLOUTS;
  }
  return cdf;
}

defineWidget({
  hostId: "ch12-sparse-walk-widget",
  controls: {
    tBudget: { label: "step budget T", min: 100, max: 4000, step: 50, default: 1000 },
    seed:    { label: "seed",          min: 1,   max: 999,  step: 1,  default: 7 },
  },
  render: (host, { tBudget, seed }, slots) => {
    const tMax = tBudget;
    // One series per chain length; the y-axis is P(hit by t).
    const series = CHAIN_LENGTHS.map((n, i) => {
      const cdf = hitProbCurve(n, tMax, seed * 1000 + n);
      return { n, cdf, colour: COLOURS[i] };
    });

    const data = [];
    for (const s of series) {
      // Downsample to ~150 points for plotting speed.
      const stride = Math.max(1, Math.floor(s.cdf.length / 150));
      for (let t = 0; t < s.cdf.length; t += stride) {
        data.push({ n: s.n, t, p: s.cdf[t] });
      }
    }

    // n^2 reference markers: the expected hitting time on a 1-D walk of
    // length n is O(n^2). Draw a vertical guide at t = n^2/4 (half-cover
    // for the middle start) per chain.
    const refs = CHAIN_LENGTHS.map((n, i) => ({
      n, x: (n * n) / 4, colour: COLOURS[i],
    }));

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      x: { label: "t (steps taken)", grid: true, domain: [0, tMax] },
      y: { label: "P(reached goal by step t)", grid: true, domain: [0, 1.02] },
      color: {
        legend: true,
        domain: CHAIN_LENGTHS.map((n) => `chain n=${n}`),
        range: COLOURS,
      },
      marks: [
        Plot.line(data, {
          x: "t", y: "p",
          stroke: (d) => `chain n=${d.n}`,
          z: "n",
          strokeWidth: 2,
        }),
        Plot.ruleY([1], { stroke: palette.muted, ...dashed }),
        Plot.ruleX(refs, {
          x: "x",
          stroke: (d) => d.colour,
          strokeOpacity: 0.4,
          ...dashed,
        }),
        Plot.text(
          refs,
          { x: "x", y: 0.05, text: (d) => `n^2/4 = ${d.x.toFixed(0)}`,
            fill: (d) => d.colour, ...annotation, textAnchor: "start", dx: 4 },
        ),
      ],
    }));

    const tail = series.map((s) => `n=${s.n}: P=${s.cdf[tMax].toFixed(3)}`).join("  ");
    slots.readout.textContent =
      `At T=${tMax}:  ${tail}  (Monte-Carlo over ${M_ROLLOUTS} rollouts)`;
  },
});

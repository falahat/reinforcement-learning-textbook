// Widget 2.F — UCB1 confidence-interval visualiser (Chapter 2).
//
// A 5-arm Bernoulli bandit. Each arm shown as a vertical thermometer:
//   - dark inner bar  = empirical mean  $\hat\mu_k$
//   - translucent top = UCB1 confidence interval $\hat\mu_k + c\sqrt{\log t / n_k}$
// The *top* of the overlay is the UCB1 score the policy argmaxes. Step
// button pulls the argmax arm, observes a Bernoulli reward, shrinks
// that arm's interval. Slider c (exploration constant); reset clears
// the agent's stats.
//
// Pedagogy: watching one arm's upper edge dip below another's as a bad
// arm gets pulled enough times is the *whole story* of "optimism in
// the face of uncertainty."
//
// Pattern: chapter markdown contains
//
//     <div id="ch2-ucb-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/ucb_bars/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";

const HOST_ID = "ch2-ucb-widget";
const N_ARMS = 5;
// Fixed true means so the visualisation is reproducible. Mix of close
// and clear-best arms — arm 3 is the optimum at 0.72.
const TRUE_MEANS = [0.30, 0.50, 0.45, 0.72, 0.60];

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>exploration c
      <input type="range" min="0.0" max="3.0" step="0.05" value="1.0" data-input="c">
    </label>
    <button data-button="step">step</button>
    <button data-button="step10">+10 steps</button>
    <button data-button="reset">↺ reset</button>
    <span data-readout></span>
  </div>
  <div data-plot="bars"></div>
  <div data-plot="regret"></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  // Per-arm running stats.
  const sums = new Array(N_ARMS).fill(0);
  const counts = new Array(N_ARMS).fill(0);
  let t = 0;
  const regretHist = [{ t: 0, regret: 0 }];
  let cumRegret = 0;
  const bestMean = Math.max(...TRUE_MEANS);

  function readC() {
    const el = host.querySelector('[data-input="c"]');
    return el ? parseFloat(el.value) : 1.0;
  }

  function ucbScore(k, c) {
    if (counts[k] === 0) return Infinity;
    const mean = sums[k] / counts[k];
    return mean + c * Math.sqrt(Math.log(Math.max(t, 1)) / counts[k]);
  }

  function pullOne() {
    const c = readC();
    // Argmax of UCB; ties broken by lower index.
    let best = 0;
    let bestScore = -Infinity;
    for (let k = 0; k < N_ARMS; k++) {
      const s = ucbScore(k, c);
      if (s > bestScore) {
        bestScore = s;
        best = k;
      }
    }
    const reward = Math.random() < TRUE_MEANS[best] ? 1 : 0;
    sums[best] += reward;
    counts[best] += 1;
    t += 1;
    cumRegret += bestMean - TRUE_MEANS[best];
    regretHist.push({ t, regret: cumRegret });
    return best;
  }

  function reset() {
    sums.fill(0);
    counts.fill(0);
    t = 0;
    cumRegret = 0;
    regretHist.length = 0;
    regretHist.push({ t: 0, regret: 0 });
    render();
  }

  function render() {
    const c = readC();

    // Build per-arm data for the thermometer plot.
    const bars = [];
    let argmax = -1;
    let argmaxScore = -Infinity;
    for (let k = 0; k < N_ARMS; k++) {
      const score = ucbScore(k, c);
      if (score > argmaxScore) {
        argmaxScore = score;
        argmax = k;
      }
    }
    for (let k = 0; k < N_ARMS; k++) {
      const n = counts[k];
      const mean = n === 0 ? 0 : sums[k] / n;
      const width = n === 0 ? 1.0 : c * Math.sqrt(Math.log(Math.max(t, 1)) / n);
      const ucbTop = Math.min(1.0, mean + width);
      bars.push({
        arm: `${k}`,
        armN: k,
        mean,
        ucbTop,
        n,
        trueMean: TRUE_MEANS[k],
        isArgmax: k === argmax,
      });
    }

    host.querySelector('[data-plot="bars"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 260,
      marginBottom: 50,
      x: { label: "arm", domain: bars.map((b) => b.arm) },
      y: { label: "mean reward / UCB1 score", domain: [0, 1.05], grid: true },
      marks: [
        // Confidence interval (translucent) — from mean to ucbTop.
        Plot.rect(bars, {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: "mean", y2: "ucbTop",
          fill: palette.warning, fillOpacity: 0.35,
          inset: 18,
        }),
        // Dark inner bar — from 0 to mean.
        Plot.rect(bars, {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: 0, y2: "mean",
          fill: (d) => d.isArgmax ? palette.secondary : palette.primary,
          fillOpacity: 0.85,
          inset: 18,
        }),
        // Highlight the argmax arm with an outline up to UCB top.
        Plot.rect(bars.filter((d) => d.isArgmax), {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: 0, y2: "ucbTop",
          stroke: palette.secondary, strokeWidth: 2, fill: "none",
          inset: 18,
        }),
        // Tick mark for the true mean (dashed reference).
        Plot.tickY(bars, {
          x: "arm", y: "trueMean",
          stroke: palette.danger, strokeWidth: 2,
          ...dashed,
        }),
        // n_k count below each bar.
        Plot.text(bars, {
          x: "arm", y: -0.04,
          text: (d) => `n=${d.n}`,
          fill: palette.muted, fontSize: 10, textAnchor: "middle",
        }),
        // UCB-score label at top.
        Plot.text(bars, {
          x: "arm", y: "ucbTop",
          text: (d) => d.n === 0 ? "∞" : d.ucbTop.toFixed(2),
          fill: (d) => d.isArgmax ? palette.secondary : palette.muted,
          fontSize: 10, textAnchor: "middle", dy: -6,
        }),
      ],
    }));

    // Regret history plot.
    host.querySelector('[data-plot="regret"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 120,
      x: { label: "t (pulls)", grid: true, domain: [0, Math.max(t, 10)] },
      y: { label: "cumulative regret", grid: true, zero: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(regretHist, { x: "t", y: "regret", stroke: palette.accent, strokeWidth: 2 }),
      ],
    }));

    const truesStr = TRUE_MEANS.map((m, k) =>
      k === argmax ? `<u>${m.toFixed(2)}</u>` : m.toFixed(2)
    ).join(", ");
    host.querySelector("[data-readout]").innerHTML =
      `t=${t} &nbsp; argmax=arm ${argmax} &nbsp; ` +
      `regret=${cumRegret.toFixed(2)} &nbsp; ` +
      `<small>true μ = [${truesStr}] &nbsp; best μ* = ${bestMean.toFixed(2)} (dashed red ticks)</small>`;
  }

  host.querySelector('[data-button="step"]').addEventListener("click", () => {
    pullOne();
    render();
  });
  host.querySelector('[data-button="step10"]').addEventListener("click", () => {
    for (let i = 0; i < 10; i++) pullOne();
    render();
  });
  host.querySelector('[data-button="reset"]').addEventListener("click", reset);
  host.querySelector('[data-input="c"]').addEventListener("input", render);

  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

// Widget 12.C — UCB1 confidence-bound thermometer (Chapter 12).
//
// Per-arm vertical bars on a 5-arm Bernoulli bandit. The dark inner bar
// is the empirical mean μ̂_k; the translucent band on top is the UCB1
// bonus c·sqrt(log t / N_k). The *top* of the band is the UCB1 score
// the policy argmaxes. The argmax arm is outlined in secondary colour.
//
// Pedagogy: watching one arm's upper edge dip below another's as a bad
// arm gets pulled enough times is the entire "optimism in the face of
// uncertainty" story. Distinct from the chapter-2 ucb_bars widget — this
// one mounts at id "ch12-ucb-bars-widget" and lives next to the
// regret-vs-T plot (12.H) in the chapter-12 narrative.
//
// Mount:
//     <div id="ch12-ucb-bars-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/ucb_bars_ch12/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";

const HOST_ID = "ch12-ucb-bars-widget";
const N_ARMS = 5;
// Fixed true means so the visualisation is reproducible. Arm 3 is the
// optimum at 0.72; arms 1 and 2 are close runners-up the algorithm
// must distinguish.
const TRUE_MEANS = [0.30, 0.50, 0.45, 0.72, 0.60];

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>exploration c
      <input type="range" min="0.0" max="3.0" step="0.05" value="1.0" data-input="c">
    </label>
    <button data-button="step">step</button>
    <button data-button="step10">+10 steps</button>
    <button data-button="step100">+100 steps</button>
    <button data-button="reset">reset</button>
    <span data-readout></span>
  </div>
  <div data-plot="bars"></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  // Per-arm running stats.
  const sums = new Array(N_ARMS).fill(0);
  const counts = new Array(N_ARMS).fill(0);
  let t = 0;

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
    return best;
  }

  function reset() {
    sums.fill(0);
    counts.fill(0);
    t = 0;
    render();
  }

  function render() {
    const c = readC();

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
        mean,
        ucbTop,
        n,
        trueMean: TRUE_MEANS[k],
        isArgmax: k === argmax,
      });
    }

    host.querySelector('[data-plot="bars"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 300,
      marginBottom: 50,
      x: { label: "arm", domain: bars.map((b) => b.arm) },
      y: { label: "mean reward / UCB1 score", domain: [0, 1.1], grid: true },
      marks: [
        // Confidence band (translucent) — from mean to UCB top.
        Plot.rect(bars, {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: "mean", y2: "ucbTop",
          fill: palette.warning, fillOpacity: 0.35,
          inset: 18,
        }),
        // Dark inner bar — empirical mean.
        Plot.rect(bars, {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: 0, y2: "mean",
          fill: (d) => d.isArgmax ? palette.secondary : palette.primary,
          fillOpacity: 0.85,
          inset: 18,
        }),
        // Argmax outline up to UCB top.
        Plot.rect(bars.filter((d) => d.isArgmax), {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: 0, y2: "ucbTop",
          stroke: palette.secondary, strokeWidth: 2, fill: "none",
          inset: 18,
        }),
        // True-mean reference tick (dashed red).
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
        // UCB score label at top.
        Plot.text(bars, {
          x: "arm", y: "ucbTop",
          text: (d) => d.n === 0 ? "inf" : d.ucbTop.toFixed(2),
          fill: (d) => d.isArgmax ? palette.secondary : palette.muted,
          fontSize: 10, textAnchor: "middle", dy: -6,
        }),
      ],
    }));

    const truesStr = TRUE_MEANS.map((m, k) =>
      k === argmax ? `<u>${m.toFixed(2)}</u>` : m.toFixed(2)
    ).join(", ");
    host.querySelector("[data-readout]").innerHTML =
      `t=${t} &nbsp; argmax=arm ${argmax} &nbsp; ` +
      `<small>true mu = [${truesStr}]; dashed red = true mean</small>`;
  }

  host.querySelector('[data-button="step"]').addEventListener("click", () => {
    pullOne();
    render();
  });
  host.querySelector('[data-button="step10"]').addEventListener("click", () => {
    for (let i = 0; i < 10; i++) pullOne();
    render();
  });
  host.querySelector('[data-button="step100"]').addEventListener("click", () => {
    for (let i = 0; i < 100; i++) pullOne();
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

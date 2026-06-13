// Widget 12.G — Bootstrapped-DQN ensemble visualisation (Chapter 12).
//
// K parallel Q-learners on a 5-arm Bernoulli bandit. Each learner sees
// a *bootstrap-resampled* subset of the data: at each pull, every
// learner k accepts the observation with probability p_mask (default
// 0.5), otherwise ignores it. Each learner maintains its own per-arm
// empirical mean Q_k(a). At each step the agent samples one head
// uniformly, acts argmax under its Q estimate — an approximate Thompson
// sample over the bootstrap ensemble.
//
// Pedagogy: the K head estimates per arm form a *spread*; wide spread
// means the ensemble disagrees (high uncertainty), tight spread means
// it agrees (low uncertainty). The widget shows the 5 arms as
// horizontal lanes; each lane has K coloured tick marks at the K head
// estimates plus a small box at their mean. As pulls accumulate, the
// per-arm boxes tighten — that visible tightening *is* the implicit
// uncertainty estimate the algorithm uses.
//
// Mount:
//     <div id="ch12-bootstrap-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/bootstrapped_dqn/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette } from "../shared/helpers.js";

const HOST_ID = "ch12-bootstrap-widget";
const N_ARMS = 5;
const TRUE_MEANS = [0.20, 0.40, 0.55, 0.70, 0.85];
const N_HEADS = 10;

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>p_mask
      <input type="range" min="0.1" max="1.0" step="0.05" value="0.5" data-input="pmask">
    </label>
    <button data-button="step">step</button>
    <button data-button="step10">+10 steps</button>
    <button data-button="step100">+100 steps</button>
    <button data-button="reset">reset</button>
    <span data-readout></span>
  </div>
  <div data-plot="ensemble"></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  // sums[k][a] / counts[k][a] = head k's empirical mean for arm a.
  const sums = Array.from({ length: N_HEADS }, () => new Array(N_ARMS).fill(0));
  const counts = Array.from({ length: N_HEADS }, () => new Array(N_ARMS).fill(0));
  let t = 0;
  let lastHead = -1;
  let lastArm = -1;

  function readPmask() {
    const el = host.querySelector('[data-input="pmask"]');
    return el ? parseFloat(el.value) : 0.5;
  }

  function headEstimate(k, a) {
    return counts[k][a] === 0 ? 0.5 : sums[k][a] / counts[k][a];
  }

  function pullOne() {
    const pmask = readPmask();
    // Sample one head uniformly.
    const head = Math.floor(Math.random() * N_HEADS);
    // Act argmax under that head's Q estimate.
    let best = 0;
    let bestVal = -Infinity;
    for (let a = 0; a < N_ARMS; a++) {
      const v = headEstimate(head, a);
      if (v > bestVal) {
        bestVal = v;
        best = a;
      }
    }
    const reward = Math.random() < TRUE_MEANS[best] ? 1 : 0;
    // Bootstrap mask: each head independently accepts with prob pmask.
    for (let k = 0; k < N_HEADS; k++) {
      if (Math.random() < pmask) {
        sums[k][best] += reward;
        counts[k][best] += 1;
      }
    }
    t += 1;
    lastHead = head;
    lastArm = best;
  }

  function reset() {
    for (let k = 0; k < N_HEADS; k++) {
      sums[k].fill(0);
      counts[k].fill(0);
    }
    t = 0;
    lastHead = -1;
    lastArm = -1;
    render();
  }

  function render() {
    // Build per-(arm, head) data points.
    const points = [];
    const summary = [];
    for (let a = 0; a < N_ARMS; a++) {
      const vals = [];
      for (let k = 0; k < N_HEADS; k++) {
        const v = headEstimate(k, a);
        points.push({ arm: `${a}`, head: k, q: v });
        vals.push(v);
      }
      vals.sort((x, y) => x - y);
      const mean = vals.reduce((s, x) => s + x, 0) / vals.length;
      const spread = vals[vals.length - 1] - vals[0];
      summary.push({
        arm: `${a}`,
        mean,
        lo: vals[0],
        hi: vals[vals.length - 1],
        trueMean: TRUE_MEANS[a],
        spread,
      });
    }

    host.querySelector('[data-plot="ensemble"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      x: { label: "arm", domain: summary.map((s) => s.arm) },
      y: { label: "head Q estimate", domain: [0, 1.05], grid: true },
      marks: [
        // Spread box from min to max head.
        Plot.rect(summary, {
          x1: (d) => d.arm, x2: (d) => d.arm,
          y1: "lo", y2: "hi",
          fill: palette.warning, fillOpacity: 0.25,
          inset: 20,
        }),
        // Each head's estimate as a coloured tick.
        Plot.tickY(points, {
          x: "arm", y: "q",
          stroke: palette.secondary,
          strokeOpacity: 0.6,
          strokeWidth: 1.5,
        }),
        // Ensemble mean.
        Plot.dot(summary, {
          x: "arm", y: "mean",
          fill: palette.primary, r: 5,
        }),
        // True mean (red).
        Plot.dot(summary, {
          x: "arm", y: "trueMean",
          fill: palette.danger, r: 4,
          symbol: "diamond",
        }),
        // Spread readout below each arm.
        Plot.text(summary, {
          x: "arm", y: -0.04,
          text: (d) => `spr=${d.spread.toFixed(2)}`,
          fill: palette.muted, fontSize: 10, textAnchor: "middle",
        }),
      ],
    }));

    const picked = lastHead >= 0 ? `head ${lastHead} -> arm ${lastArm}` : "(none yet)";
    host.querySelector("[data-readout]").innerHTML =
      `t=${t}; K=${N_HEADS} heads; last ${picked}<br>` +
      `<small>green dot = ensemble mean; red diamond = true mean; ` +
      `band = head min/max; spread shrinks with pulls</small>`;
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
  host.querySelector('[data-input="pmask"]').addEventListener("input", render);

  render();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

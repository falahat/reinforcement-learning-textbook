// Widget 11.B — MLP regression on sparse 1-D points (Chapter 11, §9.1).
//
// The "universal function approximator" pitch made visible: drop a
// handful of training points on a curve, let an MLP fit them, and
// watch *what it does between* the points. Different widths and
// depths give qualitatively different inductive biases.
//
// Pedagogical points:
//   - More hidden units = curvier interpolation (overfit risk).
//   - More depth   = sharper bends (ReLU stitches piecewise-linear
//     segments; tanh smooths them).
//   - The "between the points" curve is the part nobody specified.
//     The network is making it up — by training-error minimisation +
//     activation-shape prior. That's the whole story of generalisation.
//
// Mount: `<div id="ch11-mlp-regression-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";
import { splitmix64 } from "../shared/random.js";
import {
  Tensor, Trainer, Sequential, Linear, Relu, Tanh,
  Mse, Adam,
} from "../shared/nn.js";

const HOST_ID = "ch11-mlp-regression-widget";

const SCAFFOLD = `
  <div class="widget-controls" style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <button data-btn="playpause">▶ play</button>
    <button data-btn="step">step</button>
    <button data-btn="reset">reset</button>
    <label>target
      <select data-input="target">
        <option value="sin" selected>sin(πx)</option>
        <option value="step">step at 0</option>
        <option value="bumps">two bumps</option>
      </select>
    </label>
    <label>hidden
      <select data-input="hidden">
        <option value="8">8</option>
        <option value="32" selected>32</option>
        <option value="128">128</option>
      </select>
    </label>
    <label>depth
      <select data-input="depth">
        <option value="1">1</option>
        <option value="2" selected>2</option>
        <option value="3">3</option>
      </select>
    </label>
    <label>activation
      <select data-input="act">
        <option value="tanh" selected>tanh</option>
        <option value="relu">relu</option>
      </select>
    </label>
    <label>n train pts
      <input type="range" min="5" max="40" step="1" value="12" data-input="npts">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot="main"></div>
`;

function targetFn(name) {
  switch (name) {
    case "sin":   return (x) => Math.sin(Math.PI * x);
    case "step":  return (x) => x < 0 ? -0.5 : 0.5;
    case "bumps": return (x) => Math.exp(-25 * (x - 0.4) * (x - 0.4)) - Math.exp(-25 * (x + 0.4) * (x + 0.4));
    default: throw new Error(`unknown target '${name}'`);
  }
}

function makeData(targetName, n, seed) {
  const rng = splitmix64(seed);
  const f = targetFn(targetName);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const x = -1 + 2 * rng();
    xs.push(x); ys.push(f(x));
  }
  // Sort by x so plot lines look sensible.
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const xSorted = order.map(i => xs[i]);
  const ySorted = order.map(i => ys[i]);
  return {
    x: new Tensor(new Float32Array(xSorted), [n, 1]),
    y: new Tensor(new Float32Array(ySorted), [n, 1]),
    points: xSorted.map((xv, i) => ({ x: xv, y: ySorted[i] })),
  };
}

function makeModel(hidden, depth, actName, seed) {
  const rng = splitmix64(seed);
  const layers = [];
  // Input layer.
  layers.push(new Linear(1, hidden, "xavier", rng));
  layers.push(actName === "relu" ? Relu() : Tanh());
  // Extra hidden layers.
  for (let d = 1; d < depth; d++) {
    layers.push(new Linear(hidden, hidden, "xavier", rng));
    layers.push(actName === "relu" ? Relu() : Tanh());
  }
  // Output layer (no activation).
  layers.push(new Linear(hidden, 1, "xavier", rng));
  return new Sequential(layers);
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD;

  let trainer, data;
  let playing = false;

  function build() {
    const targetName = host.querySelector('[data-input="target"]').value;
    const hidden     = parseInt(host.querySelector('[data-input="hidden"]').value, 10);
    const depth      = parseInt(host.querySelector('[data-input="depth"]').value, 10);
    const actName    = host.querySelector('[data-input="act"]').value;
    const npts       = parseInt(host.querySelector('[data-input="npts"]').value, 10);
    data = makeData(targetName, npts, 42);
    const model = makeModel(hidden, depth, actName, 7);
    trainer = new Trainer({
      model,
      loss: new Mse(),
      optim: new Adam(0.01),
    });
  }

  function curveSamples() {
    const N = 200;
    const xs = new Float32Array(N);
    for (let i = 0; i < N; i++) xs[i] = -1.1 + (i / (N - 1)) * 2.2;
    const xt = new Tensor(xs, [N, 1]);
    const pred = trainer.predict(xt);
    const out = [];
    for (let i = 0; i < N; i++) out.push({ x: xs[i], y: pred.data[i] });
    return out;
  }

  function render() {
    const targetName = host.querySelector('[data-input="target"]').value;
    const f = targetFn(targetName);
    const truth = [];
    for (let i = 0; i <= 100; i++) {
      const xv = -1.1 + (i / 100) * 2.2;
      truth.push({ x: xv, y: f(xv) });
    }
    host.querySelector('[data-plot="main"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 720, height: 360,
      x: { domain: [-1.15, 1.15], label: "x", grid: true },
      y: { domain: [-1.5, 1.5], label: "y", grid: true },
      marks: [
        // Faint truth curve (the function we secretly know).
        Plot.line(truth, {
          x: "x", y: "y", stroke: palette.muted, strokeOpacity: 0.6, ...dashed,
        }),
        // The MLP's prediction.
        Plot.line(curveSamples(), {
          x: "x", y: "y", stroke: palette.primary, strokeWidth: 2.5,
        }),
        // Training points.
        Plot.dot(data.points, {
          x: "x", y: "y", r: 5, fill: palette.danger, stroke: "white", strokeWidth: 1,
        }),
      ],
    }));
    host.querySelector("[data-readout]").textContent =
      `step ${trainer.stepCount} | params ${countParams(trainer.model)}`;
  }

  function countParams(model) {
    let total = 0;
    model.visitParams((_path, p) => { total += p.numel(); });
    return total;
  }

  function doStep() {
    trainer.trainStep(data.x, data.y);
  }

  function tick() {
    if (!playing) return;
    for (let i = 0; i < 8; i++) doStep();
    render();
    requestAnimationFrame(tick);
  }

  host.querySelector('[data-btn="playpause"]').addEventListener("click", (e) => {
    playing = !playing;
    e.target.textContent = playing ? "⏸ pause" : "▶ play";
    if (playing) requestAnimationFrame(tick);
  });
  host.querySelector('[data-btn="step"]').addEventListener("click", () => {
    for (let i = 0; i < 50; i++) doStep();
    render();
  });
  host.querySelector('[data-btn="reset"]').addEventListener("click", () => {
    playing = false;
    host.querySelector('[data-btn="playpause"]').textContent = "▶ play";
    build();
    render();
  });
  for (const id of ["target", "hidden", "depth", "act", "npts"]) {
    host.querySelector(`[data-input="${id}"]`).addEventListener("change", () => {
      playing = false;
      host.querySelector('[data-btn="playpause"]').textContent = "▶ play";
      build();
      render();
    });
  }
  // 'npts' is a slider; refresh on input.
  host.querySelector('[data-input="npts"]').addEventListener("input", () => {
    build(); render();
  });

  build();
  render();
}

if (document.readyState !== "loading") mount();
else document.addEventListener("DOMContentLoaded", mount);

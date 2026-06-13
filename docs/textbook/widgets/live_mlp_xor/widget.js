// Widget 11.A — Live MLP overfits XOR (Chapter 11, §9.1).
//
// The reader hits "play" and watches a 2-8-1 MLP descend the loss
// landscape on the four XOR points. The decision boundary morphs in
// real time as the network finds the bent-line separator the linear
// FA chapter (Ch. 10) said it couldn't.
//
// Constitutional widget for `crates/ml/nn` — uses the JS port of the
// same API. Hot-swap controls let the reader change the activation,
// loss, or optimiser MID-TRAINING without resetting weights.
//
// Mount: `<div id="ch11-live-mlp-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette } from "../shared/helpers.js";
import { splitmix64 } from "../shared/random.js";
import {
  Tensor, Trainer, Sequential, Linear, Relu, Tanh, Sigmoid,
  Mse, Huber, Sgd, Adam,
} from "../shared/nn.js";

const HOST_ID = "ch11-live-mlp-widget";

const SCAFFOLD = `
  <div class="widget-controls" style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <button data-btn="playpause">▶ play</button>
    <button data-btn="step">step</button>
    <button data-btn="reset">reset</button>
    <label>activation
      <select data-input="act">
        <option value="tanh" selected>tanh</option>
        <option value="relu">relu</option>
        <option value="sigmoid">sigmoid</option>
      </select>
    </label>
    <label>loss
      <select data-input="loss">
        <option value="mse" selected>MSE</option>
        <option value="huber">Huber</option>
      </select>
    </label>
    <label>optim
      <select data-input="optim">
        <option value="adam" selected>Adam</option>
        <option value="sgd">SGD</option>
        <option value="momentum">SGD+momentum</option>
      </select>
    </label>
    <label>lr
      <input type="range" min="0.001" max="0.2" step="0.001" value="0.05" data-input="lr">
    </label>
    <span data-readout></span>
  </div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
    <div data-plot="boundary"></div>
    <div data-plot="loss"></div>
  </div>
`;

// XOR truth table — the four immutable training points.
const X_DATA = new Float32Array([0, 0,  0, 1,  1, 0,  1, 1]);
const Y_DATA = new Float32Array([0,       1,       1,       0]);

function makeTrainer(actName, lossName, optimName, lr) {
  const rng = splitmix64(7);
  const model = new Sequential([
    new Linear(2, 8, "xavier", rng),
    activationFor(actName),
    new Linear(8, 1, "xavier", rng),
  ]);
  return new Trainer({
    model,
    loss:  lossFor(lossName),
    optim: optimFor(optimName, lr),
  });
}

function activationFor(name) {
  switch (name) {
    case "relu":    return Relu();
    case "sigmoid": return Sigmoid();
    case "tanh":    return Tanh();
    default: throw new Error(`unknown activation '${name}'`);
  }
}
function lossFor(name) {
  switch (name) {
    case "mse":   return new Mse();
    case "huber": return new Huber(0.5);
    default: throw new Error(`unknown loss '${name}'`);
  }
}
function optimFor(name, lr) {
  switch (name) {
    case "sgd":      return new Sgd(lr);
    case "momentum": return new Sgd(lr, 0.9);
    case "adam":     return new Adam(lr);
    default: throw new Error(`unknown optim '${name}'`);
  }
}

function decisionGrid(trainer, n = 32) {
  const out = [];
  const xs = new Float32Array(n * n * 2);
  let k = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const xv = -0.2 + (i / (n - 1)) * 1.4;
      const yv = -0.2 + (j / (n - 1)) * 1.4;
      xs[k++] = xv; xs[k++] = yv;
      out.push({ x: xv, y: yv });
    }
  }
  const xt = new Tensor(xs, [n * n, 2]);
  const pred = trainer.predict(xt);
  for (let i = 0; i < out.length; i++) out[i].p = pred.data[i];
  return out;
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD;

  const x = new Tensor(X_DATA, [4, 2]);
  const y = new Tensor(Y_DATA, [4, 1]);

  let trainer = makeTrainer("tanh", "mse", "adam", 0.05);
  const lossHistory = [];
  let playing = false;
  let lastSwapStep = null;
  let lastSwapKind = null;

  const sel = (q) => host.querySelector(q);
  const btnPlay  = sel('[data-btn="playpause"]');
  const btnStep  = sel('[data-btn="step"]');
  const btnReset = sel('[data-btn="reset"]');
  const selAct   = sel('[data-input="act"]');
  const selLoss  = sel('[data-input="loss"]');
  const selOptim = sel('[data-input="optim"]');
  const selLr    = sel('[data-input="lr"]');

  // Render plots from current trainer state.
  function render() {
    const grid = decisionGrid(trainer, 32);

    sel('[data-plot="boundary"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 380, height: 320,
      x: { domain: [-0.2, 1.2], label: "x₁", grid: true },
      y: { domain: [-0.2, 1.2], label: "x₂", grid: true },
      color: { type: "diverging", scheme: "rdbu", domain: [0, 1], legend: true,
               label: "MLP output" },
      marks: [
        Plot.raster(grid, { x: "x", y: "y", fill: "p", interpolate: "nearest", opacity: 0.75 }),
        // The four XOR points, coloured by label.
        Plot.dot([
          { x: 0, y: 0, label: 0 }, { x: 0, y: 1, label: 1 },
          { x: 1, y: 0, label: 1 }, { x: 1, y: 1, label: 0 },
        ], {
          x: "x", y: "y", r: 9, stroke: "black", strokeWidth: 2,
          fill: (d) => d.label === 1 ? "white" : "black",
        }),
      ],
    }));

    sel('[data-plot="loss"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      width: 380, height: 320,
      x: { label: "step", grid: true },
      y: { label: "loss", grid: true, type: lossHistory.length > 50 ? "log" : "linear" },
      marks: [
        Plot.line(lossHistory.map((v, i) => ({ step: i, loss: Math.max(v, 1e-8) })), {
          x: "step", y: "loss", stroke: palette.primary, strokeWidth: 1.5,
        }),
        ...(lastSwapStep !== null ? [
          Plot.ruleX([lastSwapStep], { stroke: palette.danger, strokeOpacity: 0.6 }),
          Plot.text([{ step: lastSwapStep, label: `swap: ${lastSwapKind}` }], {
            x: "step", y: () => Math.max(...lossHistory.map(v => Math.max(v, 1e-8))),
            text: "label", dx: 4, fill: palette.danger,
          }),
        ] : []),
      ],
    }));

    sel("[data-readout]").textContent =
      `step ${trainer.stepCount} | loss ${(lossHistory.at(-1) ?? 0).toExponential(2)}`;
  }

  function doStep() {
    const loss = trainer.trainStep(x, y);
    lossHistory.push(loss);
    if (lossHistory.length % 5 === 0 || lossHistory.length < 50) render();
  }

  function tick() {
    if (!playing) return;
    for (let i = 0; i < 4; i++) doStep();          // 4 steps per frame
    requestAnimationFrame(tick);
  }

  // ── Wire the controls. ──────────────────────────────────────────
  btnPlay.addEventListener("click", () => {
    playing = !playing;
    btnPlay.textContent = playing ? "⏸ pause" : "▶ play";
    if (playing) requestAnimationFrame(tick);
  });

  btnStep.addEventListener("click", () => { for (let i = 0; i < 10; i++) doStep(); render(); });

  btnReset.addEventListener("click", () => {
    playing = false;
    btnPlay.textContent = "▶ play";
    trainer = makeTrainer(selAct.value, selLoss.value, selOptim.value, parseFloat(selLr.value));
    lossHistory.length = 0;
    lastSwapStep = null;
    render();
  });

  selAct.addEventListener("change", () => {
    // Hot-swap the activation layer (index 1 in Sequential).
    trainer.model.replace(1, activationFor(selAct.value));
    lastSwapStep = trainer.stepCount;
    lastSwapKind = `act → ${selAct.value}`;
    render();
  });

  selLoss.addEventListener("change", () => {
    trainer.loss = lossFor(selLoss.value);
    lastSwapStep = trainer.stepCount;
    lastSwapKind = `loss → ${selLoss.value}`;
    render();
  });

  selOptim.addEventListener("change", () => {
    trainer.optim = optimFor(selOptim.value, parseFloat(selLr.value));
    lastSwapStep = trainer.stepCount;
    lastSwapKind = `optim → ${selOptim.value}`;
    render();
  });

  selLr.addEventListener("input", () => {
    const lr = parseFloat(selLr.value);
    if ("lr" in trainer.optim) trainer.optim.lr = lr;
  });

  render();
}

if (document.readyState !== "loading") mount();
else document.addEventListener("DOMContentLoaded", mount);

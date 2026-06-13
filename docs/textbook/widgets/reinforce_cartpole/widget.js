// Widget 12.A — REINFORCE on CartPole (Chapter 12, §10.x).
//
// The first L-effort widget for `crates/ml/nn` (design doc §15,
// "First L-effort widget"). A 2-hidden-layer policy network is trained from scratch
// in the reader's browser using vanilla REINFORCE (Williams 1992) on
// the canonical CartPole-v1 dynamics (Barto, Sutton, Anderson 1983).
//
// What's on screen:
//   - Left: an SVG cart sliding under a wobbling pole. The cart
//     re-renders every animation frame from the policy's current
//     trajectory.
//   - Right: episode return (sum of rewards) over training episodes,
//     plus a faint dashed line at the solved threshold (≈ 195).
//
// Algorithm (one episode-batch update):
//   1. Run the policy until the pole drops or 200 steps elapse.
//   2. Compute discounted returns G_t = Σ γ^k r_{t+k}.
//   3. Standardise: A_t = (G_t − mean(G)) / (std(G) + 1e-8). This is
//      the simplest baseline that helps; it's variance reduction by
//      whitening, not a learned value function. (Greensmith, Bartlett
//      & Baxter 2004.)
//   4. Build a batch of (state, action, advantage) tuples and run
//      ONE gradient step on the policy-gradient loss.
//
// Mount: `<div id="ch12-reinforce-cartpole-widget" class="textbook-widget"></div>`

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";
import { splitmix64 } from "../shared/random.js";
import {
  Tensor, Sequential, Linear, Tanh, Adam,
  PolicyGradientLoss, Tape,
} from "../shared/nn.js";

const HOST_ID = "ch12-reinforce-cartpole-widget";

const SCAFFOLD = `
  <div class="widget-controls" style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
    <button data-btn="playpause">▶ play</button>
    <button data-btn="reset">reset</button>
    <label>hidden
      <select data-input="hidden">
        <option value="16">16</option>
        <option value="32" selected>32</option>
        <option value="64">64</option>
      </select>
    </label>
    <label>γ
      <input type="range" min="0.80" max="0.999" step="0.001" value="0.99" data-input="gamma">
    </label>
    <label>lr
      <input type="range" min="0.001" max="0.05" step="0.001" value="0.01" data-input="lr">
    </label>
    <span data-readout></span>
  </div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px;">
    <div data-plot="cart" style="background:#1d2935;border-radius:4px;padding:8px;"></div>
    <div data-plot="returns"></div>
  </div>
`;

// ── CartPole-v1 dynamics. Barto, Sutton & Anderson (1983) /
// OpenAI Gym `CartPole-v1`. State: [x, x_dot, theta, theta_dot]. ──
const CART = {
  g: 9.8, m_cart: 1.0, m_pole: 0.1, length: 0.5,    // half-length
  force_mag: 10.0, tau: 0.02,
  x_threshold: 2.4, theta_threshold: 12 * Math.PI / 180,
  max_steps: 200,
};

function envReset(rng) {
  // Uniform in (-0.05, 0.05) like Gym's classic-control implementation.
  return [
    (rng() * 2 - 1) * 0.05,
    (rng() * 2 - 1) * 0.05,
    (rng() * 2 - 1) * 0.05,
    (rng() * 2 - 1) * 0.05,
  ];
}

function envStep(state, action) {
  const [x, x_dot, theta, theta_dot] = state;
  const force = action === 1 ? CART.force_mag : -CART.force_mag;
  const costheta = Math.cos(theta);
  const sintheta = Math.sin(theta);
  const total_mass = CART.m_cart + CART.m_pole;
  const polemass_length = CART.m_pole * CART.length;
  const temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
  const thetaacc = (CART.g * sintheta - costheta * temp) /
    (CART.length * (4.0/3.0 - CART.m_pole * costheta * costheta / total_mass));
  const xacc = temp - polemass_length * thetaacc * costheta / total_mass;

  // Euler integration (Gym uses semi-implicit Euler; for the widget
  // the simpler form is fine and the dynamics are visually identical).
  const next = [
    x + CART.tau * x_dot,
    x_dot + CART.tau * xacc,
    theta + CART.tau * theta_dot,
    theta_dot + CART.tau * thetaacc,
  ];
  const done = Math.abs(next[0]) > CART.x_threshold ||
               Math.abs(next[2]) > CART.theta_threshold;
  return { next, reward: 1.0, done };
}

// ── Policy network: state→logits. 2 hidden layers, tanh activation. ──

function buildPolicy(hidden, seed) {
  const rng = splitmix64(seed);
  return new Sequential([
    new Linear(4, hidden, "xavier", rng),
    Tanh(),
    new Linear(hidden, hidden, "xavier", rng),
    Tanh(),
    new Linear(hidden, 2, "xavier", rng),   // 2 actions: 0=left, 1=right
  ]);
}

function softmax2(logits) {
  // Stable two-class softmax: shift, exp, normalise.
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m), e1 = Math.exp(logits[1] - m);
  const s = e0 + e1;
  return [e0 / s, e1 / s];
}

function sampleAction(probs, rng) {
  return rng() < probs[0] ? 0 : 1;
}

function runEpisode(policy, rng) {
  const states = [];
  const actions = [];
  const rewards = [];
  let s = envReset(rng);
  for (let t = 0; t < CART.max_steps; t++) {
    // Forward through policy.
    const tape = new Tape();
    const xt = new Tensor(new Float32Array(s), [1, 4]);
    const out = policy.forward(tape, xt);
    const logits = [out.data[0], out.data[1]];
    const probs = softmax2(logits);
    const a = sampleAction(probs, rng);
    states.push(s.slice());
    actions.push(a);
    const { next, reward, done } = envStep(s, a);
    rewards.push(reward);
    s = next;
    if (done) break;
  }
  return { states, actions, rewards };
}

function discountedReturns(rewards, gamma) {
  const G = new Float32Array(rewards.length);
  let g = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    g = rewards[t] + gamma * g;
    G[t] = g;
  }
  return G;
}

function standardise(arr) {
  const n = arr.length;
  let mean = 0; for (const v of arr) mean += v; mean /= n;
  let v2 = 0; for (const v of arr) v2 += (v - mean) ** 2; const std = Math.sqrt(v2 / n) + 1e-8;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (arr[i] - mean) / std;
  return out;
}

// ── Cart rendering (Observable Plot — fits the rest of the textbook). ──

function plotCart(state, episodeReward) {
  // Cart at the centre of an x ∈ [-3, 3] viewport. Pole pointing up
  // makes a positive y; we render in physical (cart-frame) coords.
  const [x, _xdot, theta, _thetadot] = state ?? [0, 0, 0, 0];
  const cartW = 0.5, cartH = 0.25;
  // Pole tip: (x + L sin θ, cartH + L cos θ) in cart-frame coords.
  const tipX = x + 1.0 * Math.sin(theta);
  const tipY = cartH + 1.0 * Math.cos(theta);
  return Plot.plot({
    width: 380, height: 280,
    style: { background: "transparent", color: "#e0e0e0" },
    x: { domain: [-3, 3], grid: true, label: "x" },
    y: { domain: [-0.5, 1.6], grid: false, label: null, ticks: 0 },
    aspectRatio: 0.5,
    marks: [
      // Track.
      Plot.ruleY([0], { stroke: palette.muted, strokeWidth: 1 }),
      // Cart body.
      Plot.rect([{ x1: x - cartW, x2: x + cartW, y1: 0, y2: cartH }], {
        x1: "x1", x2: "x2", y1: "y1", y2: "y2",
        fill: palette.primary, fillOpacity: 0.8, stroke: "white", strokeWidth: 1,
      }),
      // Pole.
      Plot.link([{ x1: x, y1: cartH, x2: tipX, y2: tipY }], {
        x1: "x1", y1: "y1", x2: "x2", y2: "y2",
        stroke: palette.warning, strokeWidth: 5,
      }),
      // Pole tip dot.
      Plot.dot([{ x: tipX, y: tipY }], { x: "x", y: "y", r: 5, fill: palette.warning }),
      // Episode-reward annotation.
      Plot.text([{ x: -2.8, y: 1.4, label: `step ${episodeReward ?? 0}` }], {
        x: "x", y: "y", text: "label", textAnchor: "start", fill: "#e0e0e0",
      }),
    ],
  });
}

function plotReturns(returns) {
  if (returns.length === 0) {
    return Plot.plot({
      ...plotDefaults, width: 380, height: 280,
      x: { label: "episode" }, y: { label: "return", domain: [0, 200] },
      marks: [],
    });
  }
  const data = returns.map((r, i) => ({ episode: i, return: r }));
  // Running mean to clarify trend.
  const window = 10;
  const smooth = [];
  for (let i = 0; i < returns.length; i++) {
    const lo = Math.max(0, i - window + 1);
    let s = 0; for (let j = lo; j <= i; j++) s += returns[j];
    smooth.push({ episode: i, return: s / (i - lo + 1) });
  }
  return Plot.plot({
    ...plotDefaults,
    width: 380, height: 280,
    x: { label: "episode", grid: true },
    y: { label: "return", domain: [0, 210], grid: true },
    marks: [
      Plot.ruleY([195], { stroke: palette.muted, strokeOpacity: 0.6, ...dashed }),
      Plot.text([{ episode: 0, return: 195, label: "solved (195)" }], {
        x: "episode", y: "return", text: "label", textAnchor: "start",
        dx: 4, dy: -4, fill: palette.muted,
      }),
      Plot.line(data, { x: "episode", y: "return",
                        stroke: palette.secondary, strokeOpacity: 0.4 }),
      Plot.line(smooth, { x: "episode", y: "return",
                          stroke: palette.primary, strokeWidth: 2 }),
    ],
  });
}

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD;

  const sel = (q) => host.querySelector(q);
  const cartSlot   = sel('[data-plot="cart"]');
  const returnSlot = sel('[data-plot="returns"]');
  const readout    = sel('[data-readout]');

  let rng, policy, optim, loss, returns, episodeCount, lastEpisode, playing;

  function reset() {
    const seed = 42;
    rng = splitmix64(seed);
    const hidden = parseInt(sel('[data-input="hidden"]').value, 10);
    const lr     = parseFloat(sel('[data-input="lr"]').value);
    policy = buildPolicy(hidden, seed + 1);
    optim = new Adam(lr);
    loss = new PolicyGradientLoss();
    returns = [];
    episodeCount = 0;
    lastEpisode = null;
    render();
  }

  function trainOneEpisode() {
    const gamma = parseFloat(sel('[data-input="gamma"]').value);
    const ep = runEpisode(policy, rng);
    const T = ep.states.length;
    const totalReward = ep.rewards.reduce((a, b) => a + b, 0);
    returns.push(totalReward);
    lastEpisode = ep;
    episodeCount++;

    // Build the (state, action, advantage) batch and do ONE gradient step.
    const states = new Float32Array(T * 4);
    for (let t = 0; t < T; t++) states.set(ep.states[t], t * 4);
    const actions = new Float32Array(ep.actions);
    const G = discountedReturns(ep.rewards, gamma);
    const A = standardise(G);

    const tape = new Tape();
    const xt = new Tensor(states,  [T, 4]);
    const at = new Tensor(actions, [T]);
    const advT = new Tensor(A, [T]);
    const logits = policy.forward(tape, xt);
    const l = loss.forward(tape, logits, at, advT);
    const grads = tape.backward(l);
    optim.step(policy, grads);
  }

  function render(progressState) {
    cartSlot.replaceChildren(plotCart(
      progressState ?? lastEpisode?.states[lastEpisode.states.length - 1] ?? [0,0,0,0],
      progressState ? null : (lastEpisode ? lastEpisode.states.length : 0),
    ));
    returnSlot.replaceChildren(plotReturns(returns));
    const lastR = returns.length > 0 ? returns[returns.length - 1] : 0;
    const recentMean = returns.length === 0 ? 0
      : returns.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, returns.length);
    readout.textContent =
      `episode ${episodeCount} | last return ${lastR.toFixed(0)} | mean(10) ${recentMean.toFixed(1)}`;
  }

  // Animation: between episodes, play the last episode back in real
  // time so the reader can SEE the cart move. After the playback we
  // train another episode in the background and queue the next playback.
  let playbackTime = 0;
  function tick() {
    if (!playing) return;
    if (!lastEpisode || playbackTime >= lastEpisode.states.length) {
      // Out of frames — train the next episode (or several, to make
      // progress in real time).
      for (let i = 0; i < 3; i++) trainOneEpisode();
      playbackTime = 0;
    } else {
      // Advance playback.
      render(lastEpisode.states[playbackTime]);
      playbackTime += 2;     // 2 sim steps per animation frame ≈ realtime
    }
    requestAnimationFrame(tick);
  }

  sel('[data-btn="playpause"]').addEventListener("click", (e) => {
    playing = !playing;
    e.target.textContent = playing ? "⏸ pause" : "▶ play";
    if (playing) requestAnimationFrame(tick);
  });
  sel('[data-btn="reset"]').addEventListener("click", () => {
    playing = false;
    sel('[data-btn="playpause"]').textContent = "▶ play";
    reset();
  });
  sel('[data-input="hidden"]').addEventListener("change", reset);
  sel('[data-input="lr"]').addEventListener("input", () => {
    if (optim) optim.lr = parseFloat(sel('[data-input="lr"]').value);
  });
  // gamma is read live in trainOneEpisode; no listener needed.

  reset();
}

if (document.readyState !== "loading") mount();
else document.addEventListener("DOMContentLoaded", mount);

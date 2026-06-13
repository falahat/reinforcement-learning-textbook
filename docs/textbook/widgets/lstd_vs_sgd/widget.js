// Widget 8.I — LSTD vs online semi-gradient TD (Chapter 8, §8.12).
//
// On a small linear-FA prediction problem we race two solvers on the
// same stream of (s_t, r_{t+1}, s_{t+1}) samples:
//
//   * Online semi-gradient TD:
//       θ ← θ + α · (r + γ · θ·φ(s') − θ·φ(s)) · φ(s)
//   * LSTD (closed-form, accumulating):
//       A ← Σ φ(s_t) (φ(s_t) − γ φ(s_{t+1}))^T,
//       b ← Σ r_t φ(s_t),
//       θ_LSTD = A^{-1} b  (recomputed every step from accumulated A, b).
//
// Plot ‖θ_t − θ*‖₂ vs step for both. LSTD's curve is sample-efficient
// (drops fast); SGD-TD lags but is O(d) per step.
//
// Problem setup: a 3-state Markov chain in a ring, transition prob 1 to
// the next state, reward r(s) = s+1 ∈ {1,2,3}, γ = 0.9. Features are
// φ(s) = (1, s/2) ∈ ℝ^2 — 2-D, so we can solve a closed-form θ*
// (Bellman fixed point under the on-policy distribution).

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const N_STATES = 3;
const GAMMA = 0.9;

function feature(s) {
  // 2-D linear feature: (1, s/2).
  return [1, s / 2];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

// Compute θ* analytically: V is 3-D, solve V = (I - γ P)^{-1} r exactly
// (chain is deterministic so closed form is easy), then least-squares fit
// to the 2-D feature space.
function trueTheta() {
  // r(s) per state at transition out of s. In the ring, P[s, (s+1)%3] = 1.
  // V(s) satisfies V(s) = r(s+1 mod 3) + γ V((s+1) mod 3). Equivalently
  // we want V(s) such that V is a fixed point. Define rewards as
  // received *on entering* a state:
  const rew = [1, 2, 3]; // r received when transitioning *from* s.
  // Iterate V := r + γ V[next] to convergence (small chain → fast).
  let V = [0, 0, 0];
  for (let iter = 0; iter < 5000; iter++) {
    const Vn = V.slice();
    let maxDelta = 0;
    for (let s = 0; s < N_STATES; s++) {
      const sp = (s + 1) % N_STATES;
      Vn[s] = rew[s] + GAMMA * V[sp];
      maxDelta = Math.max(maxDelta, Math.abs(Vn[s] - V[s]));
    }
    V = Vn;
    if (maxDelta < 1e-12) break;
  }
  // Least-squares fit V ≈ θ·φ on the 3 states, uniform stationary weight.
  // Solve normal equations: (Φ^T Φ) θ = Φ^T V.
  let M00 = 0, M01 = 0, M11 = 0, b0 = 0, b1 = 0;
  for (let s = 0; s < N_STATES; s++) {
    const phi = feature(s);
    M00 += phi[0] * phi[0];
    M01 += phi[0] * phi[1];
    M11 += phi[1] * phi[1];
    b0 += phi[0] * V[s];
    b1 += phi[1] * V[s];
  }
  const det = M00 * M11 - M01 * M01;
  return [(M11 * b0 - M01 * b1) / det, (-M01 * b0 + M00 * b1) / det];
}

const THETA_STAR = trueTheta();

function thetaErr(theta) {
  const dx = theta[0] - THETA_STAR[0];
  const dy = theta[1] - THETA_STAR[1];
  return Math.sqrt(dx * dx + dy * dy);
}

// 2x2 solve A θ = b. A is column-major-ish; return [θ0, θ1].
function solve2x2(A, b) {
  // A = [[a, b], [c, d]]
  const a = A[0][0], B = A[0][1], c = A[1][0], d = A[1][1];
  const det = a * d - B * c;
  if (Math.abs(det) < 1e-12) return [0, 0];
  return [(d * b[0] - B * b[1]) / det, (-c * b[0] + a * b[1]) / det];
}

defineWidget({
  hostId: "ch8-lstd-vs-sgd-widget",
  controls: {
    alpha: { label: "α (SGD-TD step size)", min: 0.001, max: 0.5, step: 0.001, default: 0.05 },
    steps: { label: "steps",                min: 50, max: 2000, step: 10, default: 600 },
    seed:  { label: "seed",                 min: 0,  max: 99,   step: 1,  default: 1 },
  },
  slots: ["main"],
  render: (host, { alpha, steps, seed }, slots) => {
    // Tiny LCG so the trajectory is reproducible from a seed slider.
    let rng = (Math.round(seed) * 2654435761) >>> 0;
    function rand() {
      rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
      return rng / 4294967296;
    }

    // SGD state.
    let theta = [0, 0];
    // LSTD state.
    let A = [[0, 0], [0, 0]];
    let b = [0, 0];

    const history = [];

    // Start at a random state.
    let s = Math.floor(rand() * N_STATES);

    const N = Math.round(steps);
    for (let t = 0; t < N; t++) {
      const sp = (s + 1) % N_STATES;
      const r = s + 1; // reward on transition out of s.
      const phi = feature(s);
      const phiP = feature(sp);

      // SGD-TD update.
      const delta = r + GAMMA * dot(theta, phiP) - dot(theta, phi);
      theta = [theta[0] + alpha * delta * phi[0], theta[1] + alpha * delta * phi[1]];

      // LSTD accumulators:  A += φ (φ - γ φ')^T,  b += r φ.
      const diff = [phi[0] - GAMMA * phiP[0], phi[1] - GAMMA * phiP[1]];
      A[0][0] += phi[0] * diff[0];
      A[0][1] += phi[0] * diff[1];
      A[1][0] += phi[1] * diff[0];
      A[1][1] += phi[1] * diff[1];
      b[0] += r * phi[0];
      b[1] += r * phi[1];

      let thetaLstd = [0, 0];
      if (t >= 2) {
        // Add a small ridge for invertibility in the first few samples.
        const ridge = 1e-6;
        const Ar = [
          [A[0][0] + ridge, A[0][1]],
          [A[1][0],         A[1][1] + ridge],
        ];
        thetaLstd = solve2x2(Ar, b);
      }

      history.push({
        t: t + 1,
        sgd: thetaErr(theta),
        lstd: thetaErr(thetaLstd),
      });

      s = sp;
    }

    // Reshape to long form for Plot with a colour series.
    const long = [];
    for (const h of history) {
      long.push({ t: h.t, err: h.sgd,  solver: "SGD-TD" });
      long.push({ t: h.t, err: Math.max(h.lstd, 1e-12), solver: "LSTD" });
    }

    const sgdFinal = history[history.length - 1].sgd;
    const lstdFinal = history[history.length - 1].lstd;

    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 280,
        x: { label: "samples seen", grid: true },
        y: { type: "log", label: "‖θ_t − θ*‖₂", grid: true,
             domain: [1e-6, Math.max(1, ...long.map((d) => d.err)) * 1.5] },
        color: {
          legend: true,
          domain: ["SGD-TD", "LSTD"],
          range: [palette.primary, palette.secondary],
        },
        marks: [
          Plot.line(long, {
            x: "t",
            y: "err",
            stroke: "solver",
            strokeWidth: 1.8,
          }),
        ],
      }),
    );

    // Per-step cost notes. SGD is O(d); LSTD is O(d^2) per step (rank-1
    // update + a d^3 solve at readout time). For d = 2 the wall-clock
    // ordering is dominated by per-step bookkeeping, but the scaling
    // story is the point.
    const d = 2;
    slots.readout.textContent =
      `θ* = (${THETA_STAR[0].toFixed(3)}, ${THETA_STAR[1].toFixed(3)})  ·  ` +
      `SGD-TD final err = ${sgdFinal.toExponential(2)}  ·  ` +
      `LSTD final err = ${lstdFinal.toExponential(2)}  ·  ` +
      `per-step cost: SGD = O(d) = O(${d}), LSTD = O(d²) = O(${d * d}) + O(d³) solve`;
  },
});

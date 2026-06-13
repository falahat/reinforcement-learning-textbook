// Widget 9.D — Target-network freezing demo (Chapter 9).
//
// Section §9.2's "non-stationary target" argument is theoretical. Here
// we make it visible on the minimal MDP that fails it: a 2-state MDP
// with linear FA. The state-action values are parameterised by a 2-vector
// θ ∈ ℝ² (one weight per state); features are one-hot, so Q(s, ·) = θ_s.
// We force a deterministic transition 0 → 1 → 1 (state 1 is absorbing)
// with reward r = 1 always, γ = 0.99. The fixed-point equation is
//
//     θ*_1 = 1 + γ θ*_1   ⇒   θ*_1 = 1/(1−γ) = 100
//     θ*_0 = 1 + γ θ*_1   ⇒   θ*_0 = 1 + γ · 100 ≈ 100
//
// We run three updaters in parallel, each with a different target-network
// scheme, then plot θ_0 trajectories:
//
//   (a) No target net: θ ← θ + α (r + γ θ_{s'} − θ_s)
//       Here the bootstrap uses *the same* θ we're updating. With α
//       large enough this oscillates / diverges depending on the
//       initialisation.
//   (b) Hard target net: θ̄ refreshed every C steps; in-between θ̄ is
//       frozen → updates are a proper regression toward a fixed target,
//       producing a stair-step convergence.
//   (c) Soft Polyak target net: θ̄ ← (1−τ) θ̄ + τ θ each step, giving a
//       smoothed version of (b) without the stair edges.
//
// Sliders control α, C (hard sync interval), τ (soft Polyak rate).
//
// Pattern: chapter markdown contains
//
//     <div id="ch9-target-net-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/target_network/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const GAMMA = 0.99;
const R = 1.0;
const STEPS = 300;
const THETA_STAR = R / (1 - GAMMA); // 100

// Run one updater. `targetMode` ∈ {"none", "hard", "soft"}.
// Returns [{ t, theta0, theta1, target0 }, ...] of length STEPS+1.
function run(alpha, targetMode, C, tau) {
  // Initial weights start cold.
  let theta = [0, 0];
  let bar = [0, 0]; // target weights
  const history = [{ t: 0, theta0: theta[0], target0: bar[0] }];

  for (let t = 1; t <= STEPS; t++) {
    // Alternate between updates from state 0 → 1 and state 1 → 1.
    // Both types appear so neither weight is starved.
    const s = t % 2 === 0 ? 0 : 1;
    const sp = 1;
    const tgtTheta = targetMode === "none" ? theta : bar;
    const target = R + GAMMA * tgtTheta[sp];
    const delta = target - theta[s];
    theta[s] = theta[s] + alpha * delta;

    if (targetMode === "hard") {
      if (t % C === 0) {
        bar = theta.slice();
      }
    } else if (targetMode === "soft") {
      bar[0] = (1 - tau) * bar[0] + tau * theta[0];
      bar[1] = (1 - tau) * bar[1] + tau * theta[1];
    }

    history.push({ t, theta0: theta[0], target0: bar[0] });
  }
  return history;
}

defineWidget({
  hostId: "ch9-target-net-widget",
  controls: {
    alpha: { label: "α (lr)", min: 0.05, max: 1.0, step: 0.01, default: 0.5 },
    C: { label: "C (hard sync, steps)", min: 1, max: 100, step: 1, default: 20 },
    tau: { label: "τ (soft Polyak)", min: 0.001, max: 0.5, step: 0.001, default: 0.05 },
  },
  slots: ["traj", "target"],
  render: (host, { alpha, C, tau }, slots) => {
    const none = run(alpha, "none", C, tau);
    const hard = run(alpha, "hard", C, tau);
    const soft = run(alpha, "soft", C, tau);

    const rows = [];
    for (const h of none) rows.push({ t: h.t, theta0: h.theta0, target0: h.target0, scheme: "no target net" });
    for (const h of hard) rows.push({ t: h.t, theta0: h.theta0, target0: h.target0, scheme: "hard (every C)" });
    for (const h of soft) rows.push({ t: h.t, theta0: h.theta0, target0: h.target0, scheme: "soft (τ Polyak)" });

    const colorFor = (s) =>
      s === "no target net" ? palette.danger : s === "hard (every C)" ? palette.secondary : palette.primary;

    // y-domain clamp: the no-target-net curve can blow up with large α;
    // clipping keeps the picture readable for the other two while still
    // visibly diverging.
    const allTheta = rows.map((r) => r.theta0);
    const yMax = Math.min(Math.max(...allTheta, 110), 400);
    const yMin = Math.min(Math.min(...allTheta, -10), -200);

    slots.traj.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "step t", grid: true, domain: [0, STEPS] },
      y: { label: "θ₀ (online weight)", grid: true, domain: [yMin, yMax] },
      color: {
        legend: true,
        domain: ["no target net", "hard (every C)", "soft (τ Polyak)"],
        range: [palette.danger, palette.secondary, palette.primary],
      },
      marks: [
        Plot.ruleY([THETA_STAR], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ t: STEPS, y: THETA_STAR, label: "θ* = 1/(1−γ) ≈ 100" }],
          { x: "t", y: "y", text: "label", textAnchor: "end", dy: -6, fill: palette.warning, ...annotation },
        ),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.line(rows, { x: "t", y: "theta0", stroke: (d) => colorFor(d.scheme), strokeWidth: 1.6, z: "scheme" }),
      ],
    }));

    // Show the *target* weight θ̄_0 over time, faceted by scheme. This
    // is the signal the reader should associate with stability: a smooth
    // θ̄ means a stationary regression problem; a θ̄ that tracks θ
    // step-for-step (no-target-net) is the failure mode §9.2 warns about.
    slots.target.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "step t", grid: true, domain: [0, STEPS] },
      y: { label: "θ̄₀ (target weight)", grid: true, domain: [yMin, yMax] },
      color: {
        legend: false,
        domain: ["no target net", "hard (every C)", "soft (τ Polyak)"],
        range: [palette.danger, palette.secondary, palette.primary],
      },
      marks: [
        Plot.ruleY([THETA_STAR], { stroke: palette.warning, ...dashed }),
        Plot.line(rows, { x: "t", y: "target0", stroke: (d) => colorFor(d.scheme), strokeWidth: 1.6, z: "scheme" }),
      ],
    }));

    const finalNone = none[none.length - 1].theta0;
    const finalHard = hard[hard.length - 1].theta0;
    const finalSoft = soft[soft.length - 1].theta0;
    slots.readout.textContent =
      `θ₀ final: none=${finalNone.toFixed(2)}, hard(C=${C})=${finalHard.toFixed(2)}, ` +
      `soft(τ=${tau.toFixed(3)})=${finalSoft.toFixed(2)} · target θ* = ${THETA_STAR.toFixed(2)}`;
  },
});

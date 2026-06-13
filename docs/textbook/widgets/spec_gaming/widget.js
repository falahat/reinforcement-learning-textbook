// Widget 16.H — Specification-gaming sandbox (Chapter 16).
//
// CoastRunners-flavoured toy. The agent moves around a 1-D
// circular track of L = 20 cells with two reward sources:
//
//   - finish reward F at cell 0 (also the start)
//   - bonus reward B at cell K, applied EVERY time the agent
//     visits K (the gameable loop)
//   - per-tick w_alive added unconditionally
//
// Optimal-policy choice = "race once around then collect finish"
// vs. "spin in place at the bonus tile forever" depending on the
// ratio (B + w_alive) per loop-step vs. (F + (L−1)·w_alive) per
// race. The widget computes the long-run average reward of
// each policy and reports the winner; above a threshold ratio
// (visible as a slider sweep) the bonus-spinning policy wins —
// the canonical reward-hacking failure mode.
//
// A toggle adds potential-based shaping
//   Φ(cell c) = c · w_alive · γ / (1 − γ)
// which adds a "progress-toward-finish" potential. With shaping
// enabled, the race policy regains its margin even at high B,
// demonstrating the §16.4 PBRS theorem as a *practical* fix.
//
// Mount: in §16.7 of Chapter 16. Maps to Exercise 8.
//
//     <div id="ch16-spec-gaming-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/spec_gaming/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const TRACK_LEN = 20;
const BONUS_CELL = 10;
const FINISH_CELL = 0;

// Long-run average reward under the two policies.
//   - race policy: visit cells 0 → 1 → ... → L−1 → 0, collecting
//     F at the finish and B once when passing the bonus cell.
//   - spin policy: oscillate between BONUS_CELL and BONUS_CELL+1,
//     collecting B every other tick (or every tick if we model
//     "stand still on bonus tile"). We choose the more lucrative
//     "stand still on bonus cell": B per tick.
function racePolicyMeanReward(F, B, w_alive) {
  // L ticks per loop. Reward per loop = F + B + L · w_alive.
  return (F + B + TRACK_LEN * w_alive) / TRACK_LEN;
}
function spinPolicyMeanReward(F, B, w_alive) {
  // Stand on bonus cell → +B per tick + w_alive per tick.
  return B + w_alive;
}

// Potential-based shaping: Φ(cell c) = c · scale. R̃ adds γΦ(s') − Φ(s).
// Under steady-state (loop), the potential telescopes to 0, so the
// PBRS correction integrates to 0 over a closed loop. Net effect on
// *average* reward of a closed-loop policy is therefore 0 — but the
// per-tick "spin" policy sees a *negative* per-tick shaping (Φ
// doesn't change ⇒ γΦ(s') − Φ(s) = (γ−1)Φ(c) < 0 since γ < 1).
// So PBRS penalises stationarity proportional to (1−γ)·Φ(c).
function spinShapedDelta(c, gamma, scale) {
  // The shaping at cell c, repeated: (γ − 1) · Φ(c).
  return (gamma - 1) * c * scale;
}
function raceShapedDelta() {
  // Over one closed loop the potential telescopes; net is 0.
  return 0;
}

defineWidget({
  hostId: "ch16-spec-gaming-widget",
  controls: {
    F:         { label: "finish reward F", min: 1,  max: 50, step: 1,    default: 10 },
    B:         { label: "bonus reward B",  min: 0,  max: 10, step: 0.1,  default: 1.0 },
    w_alive:   { label: "w_alive",          min: 0, max: 2,   step: 0.05, default: 0.5 },
    gamma:     { label: "γ",                min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
    shaping:   {
      type: "select",
      label: "PBRS Φ",
      options: [
        { value: "off", label: "off (raw R)" },
        { value: "on",  label: "on (Φ = c · scale)" },
      ],
      default: "off",
    },
    phi_scale: { label: "Φ scale", min: 0, max: 2, step: 0.05, default: 0.5 },
  },
  slots: ["sweep", "trajectory"],
  render: (host, params, slots) => {
    const { F, w_alive, gamma, shaping, phi_scale } = params;
    const shapingOn = shaping === "on";

    // Sweep bonus magnitude B over a range to find the crossover.
    const Bs = [];
    for (let b = 0; b <= 10; b += 0.05) Bs.push(b);
    const rows = [];
    let crossoverB = null;
    for (const b of Bs) {
      let raceR = racePolicyMeanReward(F, b, w_alive);
      let spinR = spinPolicyMeanReward(F, b, w_alive);
      if (shapingOn) {
        // The spin policy pays the negative shaping every tick;
        // race telescopes.
        spinR += spinShapedDelta(BONUS_CELL, gamma, phi_scale);
        raceR += raceShapedDelta();
      }
      rows.push({ b, policy: "race",  mean: raceR });
      rows.push({ b, policy: "spin",  mean: spinR });
      if (crossoverB === null && spinR > raceR) crossoverB = b;
    }

    slots.sweep.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "bonus reward B", domain: [0, 10], grid: true },
      y: { label: "long-run mean reward / tick", grid: true },
      color: {
        legend: true,
        domain: ["race", "spin"],
        range: [palette.primary, palette.danger],
      },
      marks: [
        Plot.line(rows, { x: "b", y: "mean", stroke: "policy", strokeWidth: 2 }),
        // Mark the configured B as a vertical rule.
        Plot.ruleX([params.B], { stroke: palette.warning, ...dashed }),
        Plot.text([{ x: params.B, y: 0, label: `B = ${fmt(params.B)}` }], {
          x: "x", y: "y", text: "label", dy: -8, textAnchor: "start",
          fill: palette.warning, ...annotation,
        }),
        // Crossover marker if any.
        ...(crossoverB !== null
          ? [
              Plot.ruleX([crossoverB], { stroke: palette.muted, ...dashed }),
              Plot.text(
                [{ x: crossoverB, y: 0, label: `crossover B ≈ ${fmt(crossoverB)}` }],
                { x: "x", y: "y", text: "label", dy: -22, textAnchor: "start",
                  fill: palette.muted, ...annotation },
              ),
            ]
          : []),
      ],
    }));

    // Trajectory cartoon — two horizontal strips showing the policy
    // each picks: race draws a left-to-right arrow across the track,
    // spin draws a small circle on the bonus cell.
    const trackCells = [];
    for (let c = 0; c < TRACK_LEN; c++) {
      trackCells.push({
        c,
        kind: c === FINISH_CELL ? "finish" : c === BONUS_CELL ? "bonus" : "track",
      });
    }
    const policyChosenIsSpin = spinPolicyMeanReward(F, params.B, w_alive)
      + (shapingOn ? spinShapedDelta(BONUS_CELL, gamma, phi_scale) : 0)
      > racePolicyMeanReward(F, params.B, w_alive);

    slots.trajectory.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 90,
      marginTop: 8, marginBottom: 28, marginLeft: 8, marginRight: 8,
      x: { label: "track cell", domain: [-0.5, TRACK_LEN - 0.5], ticks: TRACK_LEN },
      y: { axis: null, domain: [0, 1] },
      color: {
        domain: ["track", "finish", "bonus"],
        range: [palette.muted, palette.primary, palette.warning],
      },
      marks: [
        // Plot.cell needs a band scale; explicit Plot.rect keeps the
        // continuous x for tick labels and renders proper-width cells.
        Plot.rect(trackCells, {
          x1: (d) => d.c - 0.5, x2: (d) => d.c + 0.5,
          y1: 0.05, y2: 0.95,
          fill: "kind",
        }),
        Plot.text(trackCells, {
          x: "c", y: () => 0.5,
          text: (d) => d.kind === "finish" ? "F" : d.kind === "bonus" ? "B" : "",
          fill: "white", fontSize: 11, fontWeight: "bold",
        }),
        ...(policyChosenIsSpin
          ? [Plot.dot([{ c: BONUS_CELL }], {
              x: "c", y: () => 0.5,
              r: 14, fill: "none", stroke: palette.danger, strokeWidth: 2.5,
            })]
          : [Plot.arrow(
              [{ x1: FINISH_CELL, x2: TRACK_LEN - 1, y: 0.5 }],
              { x1: "x1", x2: "x2", y1: "y", y2: "y",
                stroke: palette.primary, strokeWidth: 2, headLength: 8 },
            )]),
      ],
    }));

    const winner = policyChosenIsSpin ? "spin (gaming!)" : "race (intended)";
    slots.readout.innerHTML =
      `at B = ${fmt(params.B)}, winning policy: <strong>${winner}</strong> &nbsp;|&nbsp; ` +
      `crossover B = ${crossoverB === null ? "—" : fmt(crossoverB)} &nbsp;|&nbsp; ` +
      `PBRS shaping: <strong>${shapingOn ? "on" : "off"}</strong>`;
  },
});

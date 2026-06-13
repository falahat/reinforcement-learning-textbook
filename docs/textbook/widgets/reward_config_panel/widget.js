// Widget 16.G — `RewardConfig` slider panel (Chapter 16).
//
// Live form mirroring the Simulator's `RewardConfig` (the
// reward-shaping subset that matters for the Q-bias argument).
// The reader edits five fields:
//
//   w_alive          — per-tick "just alive" baseline
//   drive_weight w_d — multiplier on each drive's d^p cost
//   cost_exponent p  — convexity of the drive cost
//   w_blood_loss     — fast-acting damage cost
//   gamma γ          — discount factor (for the Q* readout)
//
// Under three fixed scenarios:
//
//   sated-Wait     : drives ≈ 0.1 each, no blood loss, action Wait
//   hungry-Wait    : hunger 0.7 (others 0.2), no blood loss, action Wait
//   hungry-Consume : hunger drops 0.7 → 0.1 (Consume relief)
//
// The widget displays a stacked decomposition of R per scenario
// (alive baseline minus drive costs minus blood loss) and the
// implied Q* = R / (1 − γ) treating each scenario as a fixed point.
// The Consume-vs-Wait gap is the key number: when w_alive → 0 it
// flips sign and Consume wins. This is the §16.6 pathology made
// interactive.
//
// Mount: in §16.5 of Chapter 16. Maps to Exercise 6.
//
//     <div id="ch16-reward-config-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/reward_config_panel/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";

// Scenarios fixed at chapter values from §16.5 / §16.6.
const SCENARIOS = [
  {
    name: "sated-Wait",
    drives: [0.1, 0.1, 0.1, 0.1],
    blood_loss: 0,
  },
  {
    name: "hungry-Wait",
    drives: [0.7, 0.2, 0.2, 0.2],
    blood_loss: 0,
  },
  {
    name: "hungry-Consume",
    drives: [0.1, 0.2, 0.2, 0.2], // after Consume, hunger snapped 0.7 → 0.1
    blood_loss: 0,
  },
];

function rewardOf(scenario, cfg) {
  const driveCost = scenario.drives.reduce(
    (s, d) => s + cfg.w_d * Math.pow(d, cfg.p),
    0,
  );
  const bloodCost = cfg.w_blood_loss * scenario.blood_loss;
  return cfg.w_alive - driveCost - bloodCost;
}

defineWidget({
  hostId: "ch16-reward-config-widget",
  controls: {
    w_alive:      { label: "w_alive",      min: 0,    max: 2,    step: 0.05, default: 1.0 },
    w_d:          { label: "drive_weight", min: 0,    max: 0.5,  step: 0.01, default: 0.15 },
    p:            { label: "cost_exponent", min: 1,   max: 3,    step: 0.05, default: 2.0 },
    w_blood_loss: { label: "w_blood_loss", min: 0,    max: 2,    step: 0.05, default: 0.5 },
    gamma:        { label: "γ",             min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
  },
  slots: ["bars", "q"],
  render: (host, params, slots) => {
    const cfg = params;
    const results = SCENARIOS.map((sc) => {
      const R = rewardOf(sc, cfg);
      return {
        ...sc,
        R,
        Q: R / (1 - cfg.gamma),
        driveCost: sc.drives.reduce((s, d) => s + cfg.w_d * Math.pow(d, cfg.p), 0),
        bloodCost: cfg.w_blood_loss * sc.blood_loss,
      };
    });

    // Stacked bar decomposition per scenario: + alive, − drive, − blood.
    const rows = [];
    for (const r of results) {
      rows.push({ scenario: r.name, term: "+ w_alive",    value:  cfg.w_alive });
      rows.push({ scenario: r.name, term: "− drive cost", value: -r.driveCost });
      rows.push({ scenario: r.name, term: "− blood loss", value: -r.bloodCost });
    }

    slots.bars.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      marginLeft: 50,
      x: { label: "scenario", domain: SCENARIOS.map((s) => s.name) },
      y: { label: "contribution to R", grid: true },
      color: {
        legend: true,
        domain: ["+ w_alive", "− drive cost", "− blood loss"],
        range: [palette.primary, palette.danger, palette.warning],
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.barY(rows, { x: "scenario", y: "value", fill: "term" }),
        Plot.text(
          results.map((r) => ({ scenario: r.name, R: r.R })),
          {
            x: "scenario",
            y: (r) => r.R,
            text: (r) => `R = ${fmt(r.R)}`,
            dy: -10, fontSize: 11, fill: "#eee", fontWeight: "bold",
          },
        ),
      ],
    }));

    // Q* per scenario.
    const qRows = results.map((r) => ({ scenario: r.name, Q: r.Q }));

    slots.q.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      marginLeft: 56,
      x: { label: "scenario", domain: SCENARIOS.map((s) => s.name) },
      y: { label: "Q* = R / (1 − γ)", grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.5 }),
        Plot.barY(qRows, { x: "scenario", y: "Q", fill: palette.accent, fillOpacity: 0.85 }),
        Plot.text(qRows, {
          x: "scenario", y: "Q",
          text: (r) => fmt(r.Q),
          dy: (r) => (r.Q >= 0 ? -6 : 12),
          fontSize: 11, fill: "#eee",
        }),
      ],
    }));

    // Consume − Wait gap. Positive gap ⇒ Consume preferred.
    const waitQ    = results.find((r) => r.name === "hungry-Wait").Q;
    const consumeQ = results.find((r) => r.name === "hungry-Consume").Q;
    const gap = consumeQ - waitQ;
    const verdict = gap > 0
      ? "Consume preferred (healthy)"
      : "Wait preferred (Q-bias pathology)";

    slots.readout.innerHTML =
      `Consume − Wait Q* gap = <strong>${fmt(gap)}</strong> ` +
      `&nbsp;|&nbsp; ${gap > 0 ? "✓" : "⚠"} ${verdict} ` +
      `&nbsp;|&nbsp; (set w_alive = 0 to flip the sign)`;
  },
});

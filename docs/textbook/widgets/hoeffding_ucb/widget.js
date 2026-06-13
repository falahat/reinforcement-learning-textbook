// Widget 12.I — Hoeffding -> UCB derivation visualiser (Chapter 12).
//
// The widget reads two quantities at the same time:
//
//   1. The Hoeffding sample-size bound as a function of n and delta:
//        bonus(n, delta) = sqrt( log(2/delta) / (2 n) ).
//      Slide n and delta; the bonus value updates and the live formula
//      shows the substitution.
//
//   2. The *per-round* UCB1 bonus as a function of t and the chosen
//      schedule for delta_t:
//        delta_t = 1/t      -> bonus = sqrt(  log(2t) / (2 N_k))
//        delta_t = 1/t^2    -> bonus = sqrt(2 log(2t) / (2 N_k)) (UCB1)
//        delta_t = const    -> bonus = sqrt( log(2/c) / (2 N_k))
//      Plot the bonus over t for three pull-rates N_k/t in {0.01, 0.1, 0.5}
//      and the chosen delta schedule.
//
// Pedagogy: §12.3 derives UCB1's bonus from Hoeffding's inequality with
// delta = 1/t^2. The widget makes that substitution interactive — the
// reader can flip schedules and see why 1/t^2 gives sqrt(2 log t / N_k)
// while a constant delta degenerates to a flat (non-vanishing) bonus.
// Directly serves Chapter 12 exercise 7.
//
// Mount:
//     <div id="ch12-hoeffding-ucb-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/hoeffding_ucb/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

const PULL_RATES = [0.01, 0.1, 0.5];
const PULL_COLOURS = [palette.danger, palette.warning, palette.primary];

// Three delta schedules. Each returns delta_t for round t.
const SCHEDULES = {
  "1/t":   { fn: (t) => 1 / Math.max(t, 1),      label: "delta_t = 1/t" },
  "1/t^2": { fn: (t) => 1 / Math.max(t * t, 1),  label: "delta_t = 1/t^2 (UCB1)" },
  "const": { fn: () => 0.05,                     label: "delta_t = 0.05 (const)" },
};

function bonus(n, delta) {
  if (n <= 0 || delta <= 0 || delta >= 1) return NaN;
  return Math.sqrt(Math.log(2 / delta) / (2 * n));
}

defineWidget({
  hostId: "ch12-hoeffding-ucb-widget",
  controls: {
    n:        { label: "n (samples for top-of-page bound)", min: 1, max: 1000, step: 1, default: 100 },
    delta:    { label: "delta (failure prob)", min: 0.001, max: 0.2, step: 0.001, default: 0.05 },
    schedule: {
      type: "select", label: "UCB delta schedule",
      options: [
        { value: "1/t",   label: "delta_t = 1/t" },
        { value: "1/t^2", label: "delta_t = 1/t^2 (UCB1)" },
        { value: "const", label: "delta_t = 0.05 (const)" },
      ],
      default: "1/t^2",
    },
    tMax:     { label: "t_max (rounds shown)", min: 100, max: 5000, step: 100, default: 2000 },
  },
  render: (host, { n, delta, schedule, tMax }, slots) => {
    const b = bonus(n, delta);

    // Curves: bonus vs t for each pull-rate, under the chosen schedule.
    const sched = SCHEDULES[schedule];
    const data = [];
    const stride = Math.max(1, Math.floor(tMax / 200));
    for (let t = stride; t <= tMax; t += stride) {
      const d_t = sched.fn(t);
      for (let i = 0; i < PULL_RATES.length; i++) {
        const rate = PULL_RATES[i];
        const Nk = Math.max(1, Math.round(rate * t));
        const y = bonus(Nk, d_t);
        if (isFinite(y)) {
          data.push({ t, y, rate: `N_k = ${rate.toFixed(2)} t` });
        }
      }
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      x: { label: "t (round)", grid: true, domain: [stride, tMax] },
      y: { label: "UCB bonus = sqrt(log(2/delta_t) / (2 N_k))", grid: true, zero: true },
      color: {
        legend: true,
        domain: PULL_RATES.map((r) => `N_k = ${r.toFixed(2)} t`),
        range: PULL_COLOURS,
      },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted }),
        Plot.line(data, {
          x: "t", y: "y",
          stroke: "rate", z: "rate",
          strokeWidth: 2,
        }),
        Plot.text(
          [{ x: tMax, y: 0, label: sched.label }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -10,
            fill: palette.muted, ...annotation },
        ),
      ],
    }));

    const bStr = isFinite(b) ? b.toFixed(4) : "NaN";
    slots.readout.innerHTML =
      `Hoeffding: bonus(n=${n}, delta=${delta.toFixed(3)}) = ` +
      `sqrt( log(2/${delta.toFixed(3)}) / (2 * ${n}) ) = ${bStr}<br>` +
      `<small>Plot below: bonus over rounds t for three pull-rates N_k/t. ` +
      `Constant delta -> flat bonus (no learning); 1/t -> shrinks; 1/t^2 -> matches UCB1 exactly.</small>`;
  },
});

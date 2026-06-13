// Widget 16.B — Drive dynamics over time (Chapter 16).
//
// A small simulation: hunger and thirst rise linearly over T ticks
// at rates r_h and r_t. Whenever a drive crosses a Consume threshold
// θ, the agent takes a Consume action that snaps that drive back to
// 0 (instantaneous relief). The widget plots the two drive
// trajectories plus the per-tick homeostatic reward
//
//   R_t = w_alive − w_d · (d_h^p + d_t^p)
//
// over T ticks, and reports total return Σ_t R_t. Sliders let the
// reader trade "consume often, low average drive" against
// "consume rarely, high average drive" and watch the return curve
// dip when drives approach saturation. Pure JS, deterministic.
//
// Mount: in §16.1 (or §16.2) of Chapter 16.
//
//     <div id="ch16-drive-dynamics-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/drive_dynamics/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

const T_TICKS = 200;

defineWidget({
  hostId: "ch16-drive-dynamics-widget",
  controls: {
    r_h:     { label: "hunger rate",  min: 0,    max: 0.05, step: 0.001, default: 0.012 },
    r_t:     { label: "thirst rate",  min: 0,    max: 0.05, step: 0.001, default: 0.008 },
    theta:   { label: "consume θ",    min: 0.1,  max: 1,    step: 0.01,  default: 0.7 },
    w_d:     { label: "drive weight w_d", min: 0, max: 1,   step: 0.01,  default: 0.15 },
    w_alive: { label: "w_alive",      min: 0,    max: 2,    step: 0.05,  default: 1.0 },
    p:       { label: "convex exp p", min: 1,    max: 3,    step: 0.05,  default: 2.0 },
  },
  slots: ["drives", "reward"],
  render: (host, params, slots) => {
    const { r_h, r_t, theta, w_d, w_alive, p } = params;

    // Run T ticks. State = (d_h, d_t). Both drives accumulate; on
    // crossing θ, Consume snaps that drive to 0 *and* logs a
    // "consume" event for the timeline.
    let d_h = 0, d_t = 0;
    const driveRows = [];
    const rewardRows = [];
    const events = [];
    let totalReturn = 0;
    for (let t = 0; t < T_TICKS; t++) {
      d_h = Math.min(1, d_h + r_h);
      d_t = Math.min(1, d_t + r_t);
      let consumedH = false, consumedT = false;
      if (d_h >= theta) { d_h = 0; consumedH = true; }
      if (d_t >= theta) { d_t = 0; consumedT = true; }
      const R = w_alive - w_d * (Math.pow(d_h, p) + Math.pow(d_t, p));
      totalReturn += R;
      driveRows.push({ t, drive: "hunger", level: d_h });
      driveRows.push({ t, drive: "thirst", level: d_t });
      rewardRows.push({ t, R });
      if (consumedH) events.push({ t, kind: "hunger" });
      if (consumedT) events.push({ t, kind: "thirst" });
    }

    slots.drives.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "tick t", domain: [0, T_TICKS - 1], grid: true },
      y: { label: "drive level", domain: [0, 1], grid: true },
      color: {
        legend: true,
        domain: ["hunger", "thirst"],
        range: [palette.danger, palette.secondary],
      },
      marks: [
        Plot.ruleY([theta], { stroke: palette.warning, ...dashed }),
        Plot.text(
          [{ x: T_TICKS - 1, y: theta, label: `θ = ${fmt(theta)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.warning, ...annotation },
        ),
        Plot.line(driveRows, { x: "t", y: "level", stroke: "drive", strokeWidth: 1.6 }),
        Plot.dot(events, {
          x: "t",
          y: () => theta,
          fill: (e) => e.kind === "hunger" ? palette.danger : palette.secondary,
          symbol: "triangle",
          r: 4,
        }),
      ],
    }));

    slots.reward.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "tick t", domain: [0, T_TICKS - 1], grid: true },
      y: { label: "R_t", grid: true },
      marks: [
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.ruleY([w_alive], { stroke: palette.primary, ...dashed }),
        Plot.text(
          [{ x: T_TICKS - 1, y: w_alive, label: `w_alive = ${fmt(w_alive)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "end", dy: -4,
            fill: palette.primary, ...annotation },
        ),
        Plot.line(rewardRows, { x: "t", y: "R", stroke: palette.warning, strokeWidth: 1.6 }),
      ],
    }));

    const meanR = totalReturn / T_TICKS;
    const consumeCount = events.length;
    slots.readout.innerHTML =
      `Σ_t R_t = <strong>${fmt(totalReturn)}</strong> over ${T_TICKS} ticks ` +
      `&nbsp;|&nbsp; mean R = ${fmt(meanR)} ` +
      `&nbsp;|&nbsp; Consume events: ${consumeCount} ` +
      `(${events.filter((e) => e.kind === "hunger").length}H / ` +
      `${events.filter((e) => e.kind === "thirst").length}T)`;
  },
});

// Widget 13.G — Dyna-over-real-world budget calculator (Chapter 13).
//
// Turns §13.8's back-of-envelope "5 extra world-ticks per agent-tick"
// into a knob. The reader picks rollout depth k, candidate count,
// cognition cadence, and agent count, and the widget computes:
//
//     extra_ticks_per_agent_tick = k * candidate_count / cognition_cadence
//     total_extra_ticks_per_frame = agent_count * extra_ticks_per_agent_tick
//
// then renders a budget bar against a configurable per-frame ms budget
// (assuming a per-tick cost). A small heatmap shows the feasibility
// region over (k, candidate_count) for the chosen agent count.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-dyna-budget-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/dyna_budget/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

defineWidget({
  hostId: "ch13-dyna-budget-widget",
  controls: {
    k:             { label: "k (rollout depth, ticks)", min: 1, max: 30, step: 1, default: 5 },
    candidates:    { label: "candidate actions", min: 1, max: 30, step: 1, default: 10 },
    cadence:       { label: "cognition cadence (ticks)", min: 1, max: 50, step: 1, default: 10 },
    agents:        { label: "agent count", min: 1, max: 200, step: 1, default: 20 },
    tickCostMs:    { label: "per-tick cost (ms)", min: 0.01, max: 2.0, step: 0.01, default: 0.2 },
    frameBudgetMs: { label: "per-frame budget (ms)", min: 4, max: 33, step: 1, default: 16 },
  },
  slots: ["main", "heatmap"],
  render: (host, { k, candidates, cadence, agents, tickCostMs, frameBudgetMs }, slots) => {
    const extraPerAgent = (k * candidates) / cadence;
    const totalExtraTicks = agents * extraPerAgent;
    const totalExtraMs = totalExtraTicks * tickCostMs;
    const pctOfBudget = (totalExtraMs / frameBudgetMs) * 100;

    const bars = [
      { row: "extra ticks / frame", value: totalExtraTicks, unit: "ticks" },
      { row: "extra cost / frame",  value: totalExtraMs,    unit: "ms" },
      { row: "frame budget",        value: frameBudgetMs,   unit: "ms" },
    ];

    const xMax = Math.max(totalExtraMs, frameBudgetMs, 1) * 1.15;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 210,
      marginLeft: 140,
      marginBottom: 44,
      marginTop: 16,
      x: { label: "value", grid: true, domain: [0, xMax] },
      y: { label: null, domain: bars.map((b) => b.row) },
      color: {
        domain: ["ok", "over"],
        range: [palette.primary, palette.danger],
        legend: false,
      },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted }),
        Plot.ruleX([frameBudgetMs], { stroke: palette.danger, ...dashed }),
        Plot.barX(bars.filter((b) => b.row !== "frame budget"), {
          y: "row",
          x: "value",
          fill: (d) =>
            d.row === "extra cost / frame" && d.value > frameBudgetMs
              ? palette.danger
              : palette.primary,
          fillOpacity: 0.85,
        }),
        Plot.barX(bars.filter((b) => b.row === "frame budget"), {
          y: "row",
          x: "value",
          fill: palette.warning,
          fillOpacity: 0.6,
        }),
        // Value labels on the two data bars only — the budget row gets
        // a separate label below the bar so the "budget" annotation
        // doesn't overlap the value text.
        Plot.text(bars.filter((b) => b.row !== "frame budget"), {
          y: "row",
          x: "value",
          text: (d) => `${fmt(d.value)} ${d.unit}`,
          textAnchor: "start",
          dx: 4,
          fontSize: 10,
          fill: "white",
        }),
        Plot.text(
          [{ x: frameBudgetMs, y: "frame budget", label: `budget = ${fmt(frameBudgetMs)} ms` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4, dy: -12,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    // Heatmap over (k, candidate_count) at the chosen agent count + cadence.
    const K_MAX = 30;
    const C_MAX = 30;
    const cells = [];
    for (let kk = 1; kk <= K_MAX; kk++) {
      for (let cc = 1; cc <= C_MAX; cc++) {
        const ms = (agents * kk * cc * tickCostMs) / cadence;
        cells.push({
          k: kk,
          c: cc,
          ms,
          feasible: ms <= frameBudgetMs ? 1 : 0,
        });
      }
    }

    slots.heatmap.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      width: 320,
      marginLeft: 38,
      marginBottom: 38,
      // Plot.cell needs an ordinal/band scale. A 2-element numeric
      // [min, max] domain forces a linear scale and renders zero-size
      // cells. Pass an array domain of integer ticks → band scale.
      x: { label: "candidate actions", domain: d3.range(1, C_MAX + 1) },
      y: { label: "rollout depth k", domain: d3.range(1, K_MAX + 1) },
      color: {
        type: "linear",
        domain: [0, frameBudgetMs * 2],
        range: ["#1b3a52", "#b71c1c"],
        legend: true,
        label: "cost (ms)",
      },
      marks: [
        Plot.cell(cells, {
          x: "c",
          y: "k",
          fill: "ms",
          stroke: null,
        }),
        // Highlight current operating point
        Plot.dot(
          [{ k, c: candidates }],
          { x: "c", y: "k", stroke: "white", fill: "none", r: 8, strokeWidth: 2 },
        ),
        // Iso-budget contour line (approximate, by drawing a step line
        // along the feasibility frontier).
        Plot.line(
          d3.range(1, C_MAX + 1).map((cc) => {
            // largest k s.t. (agents * k * cc * tickCostMs) / cadence <= budget
            const maxK = Math.floor((frameBudgetMs * cadence) / (agents * cc * tickCostMs));
            return { c: cc, k: Math.max(0.5, Math.min(K_MAX + 0.5, maxK + 0.5)) };
          }),
          { x: "c", y: "k", stroke: palette.warning, strokeWidth: 2, ...dashed },
        ),
      ],
    }));

    const verdict = pctOfBudget <= 100
      ? `<span style="color:${palette.primary}">feasible</span>`
      : `<span style="color:${palette.danger}">over budget</span>`;
    slots.readout.innerHTML =
      `extra ticks / agent-tick = k · candidates / cadence = ` +
      `${k} · ${candidates} / ${cadence} = ${fmt(extraPerAgent)}<br>` +
      `<small>frame cost = ${fmt(totalExtraMs)} ms / ${frameBudgetMs} ms ` +
      `= ${pctOfBudget.toFixed(1)}% of budget · ${verdict}</small>`;
  },
});

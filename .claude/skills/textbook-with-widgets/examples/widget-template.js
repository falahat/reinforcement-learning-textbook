// docs/textbook/widgets/<name>/widget.js
//
// Minimum-viable TypeScript widget using the defineWidget scaffold.
// Replace `<name>`, `<chN>`, and the math with your own.
//
// Pattern: chapter markdown contains
//
//     <div id="<chN>-<name>-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/<name>/widget.js"></script>
//
// Author convention: every Greek-letter slider has both the symbol
// and the English name in the label. See `references/03-chapter-structure.md`.

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

defineWidget({
  hostId: "<chN>-<name>-widget",
  controls: {
    // Range slider: numeric input parsed as float.
    gamma: {
      label: "γ (gamma)",
      min: 0.5,
      max: 0.999,
      step: 0.005,
      default: 0.9,
    },
    n: {
      label: "n (steps)",
      min: 1,
      max: 1000,
      step: 1,
      default: 100,
    },
    // Optional dropdown:
    // mode: {
    //   type: "select",
    //   label: "Mode",
    //   default: "first",
    //   options: [
    //     { value: "first", label: "First option" },
    //     { value: "second", label: "Second option" },
    //   ],
    // },
  },
  // slots: ["main"] is the default. For multi-panel:
  // slots: ["top", "bottom"],
  render: (host, { gamma, n }, slots) => {
    // 1. Compute the data.
    const data = d3.range(n + 1).map((k) => ({
      k,
      y: Math.pow(gamma, k),
    }));

    // 2. Render the plot into slots.main.
    slots.main.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        x: { label: "k (steps)", grid: true },
        y: {
          type: "log",
          label: "γ^k",
          grid: true,
          domain: [1e-30, 1.5],
        },
        marks: [
          // Main curve.
          Plot.line(data, {
            x: "k",
            y: "y",
            stroke: palette.primary,
            strokeWidth: 2,
          }),
          // Reference threshold.
          Plot.ruleY([1e-7], {
            stroke: palette.danger,
            ...dashed,
          }),
          // Annotation.
          Plot.text(
            [{ x: n * 0.95, y: 1e-7, label: "float32 ε ≈ 1e-7" }],
            {
              x: "x",
              y: "y",
              text: "label",
              fill: palette.danger,
              textAnchor: "end",
              dy: -6,
              ...annotation,
            }
          ),
        ],
      })
    );

    // 3. Update the readout. Show the formula AND the result.
    const val = Math.pow(gamma, n);
    slots.readout.textContent =
      `γ^n = ${gamma.toFixed(3)}^${n} = ${val.toExponential(3)}`;
  },
});

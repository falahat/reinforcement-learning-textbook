// Widget 18.C — MP-DQN cross-type-bias diagnostic (Chapter 18).
//
// A 2-type PAMDP toy: action types A and B, each with a continuous
// parameter x_A, x_B ∈ [0, 1]. The TRUE Q depends only on the
// *selected* type's parameter, but the P-DQN architecture exposes
// BOTH x_A and x_B to the Q head — so the gradient flow includes
// "what if non-selected x_B were different" terms, biasing the
// learned Q-surface.
//
// We use the canonical (and tractable) shape from §18.3:
//   Q_true(s, A, x_A, x_B) = u_A(x_A)
//   Q_true(s, B, x_A, x_B) = u_B(x_B)
// where u_k is a bump centred at peak_k. P-DQN sees a confounded
// surrogate Q_PDQN that includes a coupling term c · x_other (the
// confounder slider; reflects gradient leakage from the non-selected
// branch during training). MP-DQN zeros the non-selected parameter
// before computing Q, so its surface tracks Q_true exactly.
//
// The widget plots, for the SELECTED type k ∈ {A, B}, the slice
// Q(x_k | x_other = x_other_value) as three curves:
//   - solid green: Q_true (no dependence on x_other)
//   - dashed red:  Q_PDQN (biased: shifts as x_other slides)
//   - dotted blue: Q_MP-DQN (matches green — x_other zeroed)
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-mp-dqn-bias-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/mp_dqn_bias/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed } from "../shared/helpers.js";

// Bump centred at `peak` with width σ on [0, 1].
function bump(x, peak, sigma) {
  const d = x - peak;
  return Math.exp(-(d * d) / (2 * sigma * sigma));
}

// True Q for selected type k.
function qTrue(xk, peakK, sigma) {
  return bump(xk, peakK, sigma);
}

// P-DQN's biased Q: the cross-type gradient leakage in §18.3 models
// a spurious interaction term — the Q-head learns ∂Q/∂x_other ≠ 0
// even though x_other is unused for the selected action. We model it
// as a multiplicative coupling c · (xk − 0.5) · (xOther − 0.5), which
// (i) vanishes when xOther = 0.5 (neutral), (ii) is monotone in c, and
// (iii) crucially *tilts* Q(xk) so the argmax in xk shifts toward
// whichever end of [0, 1] xOther falls on. That argmax shift is the
// observable harm MP-DQN's zeroing fixes.
function qPDQN(xk, peakK, sigma, xOther, c) {
  return bump(xk, peakK, sigma) + c * (xk - 0.5) * (xOther - 0.5);
}

// MP-DQN: zero x_other before computing Q. The interaction term
// becomes c · (xk − 0.5) · (0 − 0.5) = −0.5 c · (xk − 0.5), a
// linear tilt independent of xOther. This is a small residual
// systematic bias (the same offset would appear during training too),
// but crucially it does NOT track the slider — it is a fixed
// architectural property, exactly as in the paper.
function qMPDQN(xk, peakK, sigma, c) {
  return bump(xk, peakK, sigma) + c * (xk - 0.5) * (0 - 0.5);
}

defineWidget({
  hostId: "ch18-mp-dqn-bias-widget",
  controls: {
    type: {
      type: "select",
      label: "selected type",
      options: [{ value: "A", label: "A" }, { value: "B", label: "B" }],
      default: "A",
    },
    xOther: { label: "x_other (non-selected param)", min: 0, max: 1, step: 0.01, default: 0.2 },
    confound: { label: "c (confound coupling)", min: 0, max: 0.6, step: 0.01, default: 0.3 },
  },
  render: (host, { type, xOther, confound }, slots) => {
    const peakA = 0.30;
    const peakB = 0.75;
    const sigma = 0.15;
    const peakK = type === "A" ? peakA : peakB;

    const N = 301;
    const xs = d3.range(N).map((i) => i / (N - 1));
    const rows = [];
    for (const xk of xs) {
      const qt = qTrue(xk, peakK, sigma);
      const qp = qPDQN(xk, peakK, sigma, xOther, confound);
      const qm = qMPDQN(xk, peakK, sigma, confound);
      rows.push({ xk, q: qt, kind: "Q_true" });
      rows.push({ xk, q: qp, kind: "Q_PDQN (biased)" });
      rows.push({ xk, q: qm, kind: "Q_MP-DQN (zeroed)" });
    }

    // argmax x_k for each method (over the discretised xs grid).
    const argmaxKind = (kind) => {
      const slice = rows.filter((r) => r.kind === kind);
      let best = slice[0];
      for (const r of slice) if (r.q > best.q) best = r;
      return best;
    };
    const aTrue = argmaxKind("Q_true");
    const aP = argmaxKind("Q_PDQN (biased)");
    const aM = argmaxKind("Q_MP-DQN (zeroed)");

    const colorFor = (kind) =>
      kind === "Q_true" ? palette.primary
      : kind === "Q_PDQN (biased)" ? palette.danger
      : palette.secondary;
    const dashFor = (kind) =>
      kind === "Q_true" ? null
      : kind === "Q_PDQN (biased)" ? "5 3"
      : "2 3";

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 290,
      x: { label: `x_${type} (selected-type parameter)`, domain: [0, 1], grid: true },
      y: { label: `Q(s, ${type}, x_${type})`, grid: true },
      color: { legend: true, domain: ["Q_true", "Q_PDQN (biased)", "Q_MP-DQN (zeroed)"] },
      marks: [
        Plot.line(rows, {
          x: "xk", y: "q", z: "kind",
          stroke: (d) => colorFor(d.kind),
          strokeWidth: 2,
          strokeDasharray: (d) => dashFor(d.kind),
        }),
        // Vertical rules at each method's argmax.
        Plot.ruleX([aTrue.xk], { stroke: palette.primary, ...dashed }),
        Plot.ruleX([aP.xk], { stroke: palette.danger, ...dashed }),
        Plot.dot([aTrue, aP, aM], {
          x: "xk", y: "q",
          fill: (d) => colorFor(d.kind), r: 4,
        }),
      ],
    }));

    const argmaxGap = Math.abs(aP.xk - aTrue.xk);
    slots.readout.innerHTML =
      `peak_${type} = ${peakK.toFixed(2)}   ·   ` +
      `argmax Q_true at x = ${aTrue.xk.toFixed(3)}   ·   ` +
      `argmax Q_PDQN at x = ${aP.xk.toFixed(3)}   ·   ` +
      `bias gap = ${argmaxGap.toFixed(3)}   ·   ` +
      `<small>MP-DQN tracks Q_true regardless of x_other; P-DQN tilts with it</small>`;
  },
});

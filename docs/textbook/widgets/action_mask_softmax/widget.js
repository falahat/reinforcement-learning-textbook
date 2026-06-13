// Widget 18.F — Action-masking softmax inspector (Chapter 18).
//
// K = 6 action logits ℓ₁..ℓ₆ the reader sets via sliders. A second row
// of "legal" toggles (select dropdowns "✓ legal" / "✗ illegal") marks
// which actions are available in the current state. The widget plots
// three distributions side-by-side:
//   1. unmasked softmax — what a naive policy outputs;
//   2. masked softmax — illegal logits → −∞ before softmax (the
//      [Huang & Ontañón 2020] approach);
//   3. penalty-shaped softmax — illegal logits get −P penalty (a
//      "softer" alternative that still leaks probability mass onto
//      illegal actions, especially when P is small).
//
// A penalty-magnitude slider P lets the reader sweep "penalty = 0"
// (no constraint) through "penalty = 50" (≈ hard mask). Watching
// the residual mass on illegal actions in the penalty version is
// the §18.7 pedagogical point: masking is exact; penalty is leaky.
//
// Pattern: chapter markdown contains
//
//     <div id="ch18-action-mask-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/action_mask_softmax/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette } from "../shared/helpers.js";

const K = 6;
const LABELS = ["a₁", "a₂", "a₃", "a₄", "a₅", "a₆"];

function softmax(xs) {
  // −∞ logits represent hard mask; e^{-∞} = 0, no special-case needed
  // beyond keeping the max finite.
  const finite = xs.filter((x) => Number.isFinite(x));
  const m = finite.length > 0 ? Math.max(...finite) : 0;
  const exps = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)));
  const z = exps.reduce((a, b) => a + b, 0);
  if (z === 0) return xs.map(() => 0);
  return exps.map((e) => e / z);
}

defineWidget({
  hostId: "ch18-action-mask-widget",
  controls: {
    l1: { label: "ℓ₁", min: -4, max: 4, step: 0.1, default: 2.0 },
    l2: { label: "ℓ₂", min: -4, max: 4, step: 0.1, default: 1.0 },
    l3: { label: "ℓ₃", min: -4, max: 4, step: 0.1, default: 0.5 },
    l4: { label: "ℓ₄", min: -4, max: 4, step: 0.1, default: 0.0 },
    l5: { label: "ℓ₅", min: -4, max: 4, step: 0.1, default: -0.5 },
    l6: { label: "ℓ₆", min: -4, max: 4, step: 0.1, default: -1.0 },
    legal1: { type: "select", label: "a₁",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "0" },
    legal2: { type: "select", label: "a₂",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "1" },
    legal3: { type: "select", label: "a₃",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "1" },
    legal4: { type: "select", label: "a₄",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "0" },
    legal5: { type: "select", label: "a₅",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "1" },
    legal6: { type: "select", label: "a₆",
      options: [{ value: "1", label: "✓" }, { value: "0", label: "✗" }], default: "1" },
    penalty: { label: "penalty P (soft mask)", min: 0, max: 10, step: 0.1, default: 2.0 },
  },
  render: (host, p, slots) => {
    const logits = [p.l1, p.l2, p.l3, p.l4, p.l5, p.l6];
    const legalFlags = [p.legal1, p.legal2, p.legal3, p.legal4, p.legal5, p.legal6]
      .map((s) => s === "1");
    const P = p.penalty;

    const pUnmasked = softmax(logits);
    const pMasked = softmax(
      logits.map((l, k) => (legalFlags[k] ? l : -Infinity)),
    );
    const pPenalty = softmax(
      logits.map((l, k) => (legalFlags[k] ? l : l - P)),
    );

    // Long-form rows for the faceted plot.
    const rows = [];
    for (let k = 0; k < K; k++) {
      rows.push({ action: LABELS[k], legal: legalFlags[k],
        method: "unmasked", prob: pUnmasked[k] });
      rows.push({ action: LABELS[k], legal: legalFlags[k],
        method: "hard mask (-∞)", prob: pMasked[k] });
      rows.push({ action: LABELS[k], legal: legalFlags[k],
        method: `penalty (P=${P.toFixed(1)})`, prob: pPenalty[k] });
    }

    const colorFor = (d) =>
      d.method === "hard mask (-∞)" ? palette.primary
      : d.method === "unmasked" ? palette.muted
      : palette.warning;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      marginLeft: 50,
      marginBottom: 40,
      x: { label: "action", domain: LABELS },
      y: { label: "π(a)", domain: [0, 1], grid: true },
      fx: { label: null,
        domain: ["unmasked", "hard mask (-∞)", `penalty (P=${P.toFixed(1)})`] },
      marks: [
        Plot.frame({ stroke: palette.muted, strokeOpacity: 0.25 }),
        Plot.barY(rows, {
          fx: "method", x: "action", y: "prob",
          fill: colorFor,
          // Hatch illegal actions by lowering opacity so the
          // "residual mass on illegal" jumps out visually.
          fillOpacity: (d) => d.legal ? 0.85 : 0.45,
          stroke: (d) => d.legal ? "none" : palette.danger,
          strokeWidth: 1.5,
        }),
        Plot.text(rows, {
          fx: "method", x: "action", y: "prob",
          text: (d) => d.prob > 0.005 ? d.prob.toFixed(2) : "",
          dy: -6, fontSize: 9, fill: "#ddd",
        }),
        Plot.ruleY([0]),
      ],
    }));

    // Residual mass on illegal actions (penalty leakage).
    const illegalMassUnmasked = pUnmasked.reduce(
      (s, q, k) => s + (legalFlags[k] ? 0 : q), 0);
    const illegalMassPenalty = pPenalty.reduce(
      (s, q, k) => s + (legalFlags[k] ? 0 : q), 0);
    const illegalMassMask = pMasked.reduce(
      (s, q, k) => s + (legalFlags[k] ? 0 : q), 0);

    slots.readout.innerHTML =
      `Σ π(illegal): unmasked = ${illegalMassUnmasked.toFixed(3)}   ·   ` +
      `hard mask = ${illegalMassMask.toFixed(3)}   ·   ` +
      `penalty(P=${P.toFixed(1)}) = ${illegalMassPenalty.toFixed(3)}   ·   ` +
      `<small>red outlines mark illegal actions</small>`;
  },
});

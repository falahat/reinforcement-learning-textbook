// Widget 11.C — TRPO KL trust region visualization (Chapter 11).
//
// TRPO maximises an importance-weighted surrogate subject to a KL
// constraint:
//   max_θ E[ (π_θ / π_old) A ]   s.t.   E_s[ KL(π_old ‖ π_θ) ] ≤ δ.
//
// The widget builds a tractable 2D version. Three actions, softmax over
// logits z = (0, θ₁, θ₂); π_old = softmax(0, 0, 0) = uniform. Vary θ in
// the plane and colour each cell by KL(π_old ‖ π_θ). Overlay the trust
// region boundary KL = δ as a thick contour. An advantage vector
// A = (A₀, A₁, A₂) defines the unconstrained PG direction — drawn as
// an arrow from origin; its tip is clipped (red) when outside the trust
// region. A second panel shows the simplex with π_old (blue) and the
// candidate π_θ (orange) so the reader can read the policy change.
//
// Pedagogy: TRPO is asserted in §11.4 with formulas but no picture. The
// widget shows that "KL ≤ δ" is a real geometric region the gradient
// step must respect, and that vanilla PG happily leaves it.
//
// Pattern: chapter markdown has
//
//     <div id="ch11-kl-trust-region-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/kl_trust_region/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";

// θ-plane half-width. Logit range ±2.5 covers π in [0.04, 0.83].
const TH_MAX = 2.5;
const GRID = 50; // 50×50 KL heatmap

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const Z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / Z);
}

function kl(p, q) {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]);
  }
  return s;
}

// Project a candidate θ onto the trust region by scaling toward origin
// until KL(π_old ‖ π_θ_scaled) ≤ δ. Bisection (KL is monotone along the
// ray from the uniform reference for symmetric softmax).
function projectToTrustRegion(theta1, theta2, delta) {
  const pOld = softmax([0, 0, 0]);
  const klAt = (t) => kl(pOld, softmax([0, t * theta1, t * theta2]));
  if (klAt(1) <= delta) return { t: 1, clipped: false };
  let lo = 0, hi = 1;
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    if (klAt(mid) <= delta) lo = mid;
    else hi = mid;
  }
  return { t: lo, clipped: true };
}

defineWidget({
  hostId: "ch11-kl-trust-region-widget",
  controls: {
    delta: { label: "δ (trust radius)", min: 0.001, max: 0.5, step: 0.001, default: 0.05 },
    a1:    { label: "A₁ (adv. action 1)", min: -2, max: 2, step: 0.1, default: 1.5 },
    a2:    { label: "A₂ (adv. action 2)", min: -2, max: 2, step: 0.1, default: -0.5 },
    step:  { label: "step length η", min: 0, max: 2, step: 0.01, default: 1.0 },
  },
  slots: ["theta", "simplex"],
  render: (host, { delta, a1, a2, step }, slots) => {
    const pOld = softmax([0, 0, 0]);

    // Build KL grid over θ-plane.
    const cells = [];
    for (let i = 0; i <= GRID; i++) {
      for (let j = 0; j <= GRID; j++) {
        const t1 = -TH_MAX + (2 * TH_MAX) * (i / GRID);
        const t2 = -TH_MAX + (2 * TH_MAX) * (j / GRID);
        const pNew = softmax([0, t1, t2]);
        cells.push({ t1, t2, kl: kl(pOld, pNew) });
      }
    }

    // Vanilla PG step direction = advantage vector (here treated as
    // descent direction in logit space; gradient of E[A log π]).
    const gx = step * a1;
    const gy = step * a2;
    const tip = { t1: gx, t2: gy };
    const klAtTip = kl(pOld, softmax([0, gx, gy]));

    // Project onto trust region.
    const proj = projectToTrustRegion(gx, gy, delta);
    const projTip = { t1: gx * proj.t, t2: gy * proj.t };

    // Contour thresholds — emphasise the δ level.
    const klValues = cells.map((d) => d.kl);
    const klHi = Math.max(...klValues);
    const baseThresh = d3.range(8).map((i) => (klHi * (i + 1)) / 9);
    const thresholds = [...baseThresh, delta].sort((a, b) => a - b);

    slots.theta.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 320,
      width: 360,
      aspectRatio: 1,
      x: { label: "θ₁ (logit shift, action 1)", domain: [-TH_MAX, TH_MAX] },
      y: { label: "θ₂ (logit shift, action 2)", domain: [-TH_MAX, TH_MAX] },
      marks: [
        // Background KL heatmap via raster.
        Plot.raster(cells, {
          x: "t1", y: "t2", fill: "kl",
          interpolate: "nearest",
          fillOpacity: 0.55,
        }),
        // Faint contour lines for context.
        Plot.contour(cells, {
          x: "t1", y: "t2", value: "kl",
          thresholds: baseThresh,
          stroke: palette.muted, strokeOpacity: 0.35, strokeWidth: 0.6,
        }),
        // Trust region boundary — thick warning-coloured contour at KL=δ.
        Plot.contour(cells, {
          x: "t1", y: "t2", value: "kl",
          thresholds: [delta],
          stroke: palette.warning, strokeWidth: 2.5,
        }),
        // Origin = π_old.
        Plot.dot([{ t1: 0, t2: 0 }], {
          x: "t1", y: "t2", fill: palette.secondary, r: 5, stroke: "white",
        }),
        // Unconstrained PG step (dashed, possibly outside trust region).
        Plot.arrow([{ x1: 0, y1: 0, x2: tip.t1, y2: tip.t2 }], {
          x1: "x1", y1: "y1", x2: "x2", y2: "y2",
          stroke: proj.clipped ? palette.danger : palette.primary,
          strokeWidth: 1.5,
          ...(proj.clipped ? dashed : {}),
          headLength: 8,
        }),
        // Constrained step (projected). Only drawn when clipping happened.
        ...(proj.clipped
          ? [Plot.arrow([{ x1: 0, y1: 0, x2: projTip.t1, y2: projTip.t2 }], {
              x1: "x1", y1: "y1", x2: "x2", y2: "y2",
              stroke: palette.primary, strokeWidth: 2.4, headLength: 9,
            })]
          : []),
        // Endpoint markers.
        Plot.dot([tip], {
          x: "t1", y: "t2", fill: proj.clipped ? palette.danger : palette.primary,
          r: 4, stroke: "white",
        }),
        Plot.text(
          [{ t1: 0, t2: 0, label: "π_old" }],
          { x: "t1", y: "t2", text: "label", textAnchor: "start", dx: 6, dy: -6,
            fill: palette.secondary, ...annotation },
        ),
      ],
      color: {
        scheme: "viridis",
        legend: true,
        label: "KL(π_old ‖ π_θ)",
      },
    }));

    // Simplex panel: show π_old and projected π_θ as bars over actions.
    const pNew = softmax([0, projTip.t1, projTip.t2]);
    const pNewUnclipped = softmax([0, tip.t1, tip.t2]);
    const rows = [];
    const labels = ["a₀", "a₁", "a₂"];
    for (let i = 0; i < 3; i++) {
      rows.push({ policy: "π_old", action: labels[i], p: pOld[i] });
      rows.push({ policy: "π_θ (constrained)", action: labels[i], p: pNew[i] });
      rows.push({ policy: "π_θ (unclipped PG)", action: labels[i], p: pNewUnclipped[i] });
    }

    slots.simplex.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      width: 360,
      marginLeft: 50,
      x: { label: "action", domain: labels },
      y: { label: "π(a)", domain: [0, 1], grid: true },
      color: {
        domain: ["π_old", "π_θ (constrained)", "π_θ (unclipped PG)"],
        range: [palette.secondary, palette.primary, palette.danger],
        legend: true,
      },
      marks: [
        Plot.barY(rows, {
          x: "action",
          y: "p",
          fill: "policy",
          fillOpacity: 0.75,
          dx: (d) =>
            d.policy === "π_old" ? -10 :
            d.policy === "π_θ (constrained)" ? 0 : 10,
          width: 8,
        }),
      ],
    }));

    const note = proj.clipped
      ? `clipped: η scaled by ${proj.t.toFixed(3)} (unclipped KL = ${klAtTip.toFixed(4)} > δ)`
      : `inside trust region (KL = ${klAtTip.toFixed(4)} ≤ δ)`;
    slots.readout.textContent =
      `δ = ${delta.toFixed(3)}   |   PG step = (${gx.toFixed(2)}, ${gy.toFixed(2)})   |   ${note}`;
  },
});

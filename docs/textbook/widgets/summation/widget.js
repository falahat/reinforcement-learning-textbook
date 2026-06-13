// Widget 1.X — Summation walkthrough (Chapter 1).
//
// Steps through the partial sums of a chosen series Σ a_k, term by
// term. At each frame k:
//   - the bar chart highlights term k (orange), shows terms 0..k
//     solidly (green), and dims future terms (low opacity);
//   - the line plot draws the partial-sum curve solid up to k and
//     dashed beyond, with a dashed horizontal at the closed-form
//     limit when one exists (geometric → 1/(1-γ); p-series with
//     p=2 → π²/6).
//
// Supported formulas: geometric Σ γ^k, harmonic Σ 1/k, p-series
// Σ 1/k², and arithmetic Σ k. Harmonic and arithmetic diverge —
// the limit line is omitted in those cases.
//
// Mount in §1.4 of Chapter 1.
//
//     <div id="ch1-summation-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/summation/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";

// Per-formula configuration. `termFn(k, params)` is the k-th term;
// `limit(params)` returns the closed-form limit (Infinity if the
// series diverges); `expr(k, params)` is a short string for the
// readout showing how this particular term evaluates.
const FORMULAS = {
  geometric: {
    label: "Σ γ^k (geometric)",
    latex: "Σ_{k=0}^{N} γ^k",
    termFn: (k, p) => Math.pow(p.gamma, k),
    limit: (p) => 1 / (1 - p.gamma),
    limitLatex: (p) => `1/(1−γ) = ${fmt(1 / (1 - p.gamma))}`,
    termExpr: (k, p) => `γ^${k} = ${fmt(Math.pow(p.gamma, k))}`,
    usesGamma: true,
  },
  harmonic: {
    label: "Σ 1/k (harmonic, diverges)",
    latex: "Σ_{k=1}^{N} 1/k",
    termFn: (k) => (k === 0 ? 0 : 1 / k),
    limit: () => Infinity,
    limitLatex: () => "∞ (diverges)",
    termExpr: (k) => (k === 0 ? "0 (k=0 skipped)" : `1/${k} = ${fmt(1 / k)}`),
    usesGamma: false,
  },
  p_series: {
    label: "Σ 1/k² (p-series, p=2)",
    latex: "Σ_{k=1}^{N} 1/k²",
    termFn: (k) => (k === 0 ? 0 : 1 / (k * k)),
    limit: () => (Math.PI * Math.PI) / 6,
    limitLatex: () => `π²/6 = ${fmt((Math.PI * Math.PI) / 6)}`,
    termExpr: (k) => (k === 0 ? "0 (k=0 skipped)" : `1/${k}² = ${fmt(1 / (k * k))}`),
    usesGamma: false,
  },
  arithmetic: {
    label: "Σ k (arithmetic, diverges)",
    latex: "Σ_{k=0}^{N} k",
    termFn: (k) => k,
    limit: () => Infinity,
    limitLatex: () => "∞ (diverges)",
    termExpr: (k) => `${k}`,
    usesGamma: false,
  },
};

defineStepper({
  hostId: "ch1-summation-widget",
  controls: {
    formula: {
      label: "series",
      type: "select",
      options: [
        { value: "geometric",  label: "Σ γ^k (geometric)" },
        { value: "harmonic",   label: "Σ 1/k (harmonic)" },
        { value: "p_series",   label: "Σ 1/k² (p-series)" },
        { value: "arithmetic", label: "Σ k (arithmetic)" },
      ],
      default: "geometric",
    },
    gamma: { label: "γ (geometric only)", min: 0.1, max: 0.99, step: 0.01, default: 0.9 },
    N:     { label: "N (terms)",          min: 1,   max: 50,   step: 1,    default: 20 },
  },
  slots: ["bars", "running"],
  // Precompute every term and partial sum once per parameter change.
  trajectory: (params) => {
    const N = Math.max(1, Math.round(params.N));
    const cfg = FORMULAS[params.formula] || FORMULAS.geometric;
    const limit = cfg.limit(params);

    // Build (term, partial_sum) lists over k = 0..N.
    const terms = [];
    const partials = [];
    let acc = 0;
    for (let k = 0; k <= N; k++) {
      const t = cfg.termFn(k, params);
      acc += t;
      terms.push(t);
      partials.push(acc);
    }

    // One frame per k. Each frame snapshots the running view.
    const frames = [];
    for (let k = 0; k <= N; k++) {
      frames.push({
        k,
        N,
        term: terms[k],
        partial: partials[k],
        terms: terms.slice(),     // shared reference is fine — frozen by stepper
        partials: partials.slice(),
        formula: params.formula,
        limit,
        cfg,
        params,
      });
    }
    return frames;
  },
  playIntervalMs: 600,
  render: (host, frame, idx, total, _params, slots) => {
    const { k, N, term, partial, terms, partials, limit, cfg, params } = frame;

    // ---- Slot 1: bar chart of all terms, with current bar highlighted ----
    // Build one record per k with a color/opacity flag.
    const barData = terms.map((value, j) => {
      let state;
      if (j === k) state = "current";
      else if (j < k) state = "past";
      else state = "future";
      return { k: j, value, state };
    });

    // Dynamic y-domain: pad above the max so the highlight stays visible.
    const maxTerm = Math.max(0, ...terms);
    const minTerm = Math.min(0, ...terms);
    const pad = (maxTerm - minTerm) * 0.08 || 0.1;

    slots.bars.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 200,
        x: {
          label: "term index k",
          domain: Array.from({ length: N + 1 }, (_, j) => j),
        },
        y: {
          label: "term value a_k",
          grid: true,
          domain: [minTerm - pad, maxTerm + pad],
        },
        marks: [
          Plot.ruleY([0], { stroke: palette.muted }),
          // Bars colored by state. fillOpacity dims future terms.
          Plot.barY(barData, {
            x: "k",
            y: "value",
            fill: (d) =>
              d.state === "current" ? palette.warning : palette.primary,
            fillOpacity: (d) =>
              d.state === "future" ? 0.18 : d.state === "current" ? 1.0 : 0.85,
            stroke: (d) => (d.state === "current" ? palette.danger : "none"),
            strokeWidth: (d) => (d.state === "current" ? 1.5 : 0),
          }),
        ],
      }),
    );

    // ---- Slot 2: running partial-sum line ----
    // Split into a solid segment 0..k and a dashed segment k..N. We
    // include the joint point k in both so they meet visually.
    const line = partials.map((value, j) => ({ k: j, value }));
    const solid = line.slice(0, k + 1);
    const future = line.slice(k); // overlap at j=k

    // Compute a y-range that gracefully accommodates the limit when finite.
    const allY = partials.slice();
    if (isFinite(limit)) allY.push(limit);
    const yMin = Math.min(0, ...allY);
    const yMax = Math.max(...allY) * 1.05 + 1e-9;

    const marks = [
      Plot.ruleY([0], { stroke: palette.muted }),
      // Dashed future trajectory underneath.
      Plot.line(future, {
        x: "k",
        y: "value",
        stroke: palette.primary,
        strokeOpacity: 0.5,
        ...dashed,
      }),
      // Solid included trajectory on top.
      Plot.line(solid, {
        x: "k",
        y: "value",
        stroke: palette.primary,
        strokeWidth: 2,
      }),
      Plot.dot(solid, {
        x: "k",
        y: "value",
        fill: palette.primary,
        r: 2.5,
      }),
      // Current point — orange, larger.
      Plot.dot([{ k, value: partial }], {
        x: "k",
        y: "value",
        fill: palette.warning,
        stroke: palette.danger,
        r: 5,
      }),
      // Annotation near the current partial sum.
      Plot.text([{ k, value: partial }], {
        x: "k",
        y: "value",
        text: () => `partial sum = ${fmt(partial)}`,
        dx: 6,
        dy: -8,
        textAnchor: "start",
        ...annotation,
        fill: palette.warning,
      }),
    ];

    if (isFinite(limit)) {
      marks.unshift(
        Plot.ruleY([limit], { stroke: palette.danger, ...dashed }),
        Plot.text([{ x: N, y: limit, label: `limit = ${fmt(limit)}` }], {
          x: "x",
          y: "y",
          text: "label",
          textAnchor: "end",
          dy: -4,
          fill: palette.danger,
          ...annotation,
        }),
      );
    }

    slots.running.replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 200,
        x: { label: "k", domain: [0, N], grid: true },
        y: {
          label: "partial sum S_k",
          grid: true,
          domain: [yMin, yMax],
        },
        marks,
      }),
    );

    // ---- Readout: the formula header + per-step evaluation ----
    const gammaStr = cfg.usesGamma ? ` where γ=${fmt(params.gamma)}` : "";
    const limitStr = cfg.limitLatex(params);
    const termStr  = cfg.termExpr(k, params);
    slots.readout.innerHTML =
      `<small><strong>${cfg.latex.replace("N", String(N))}</strong>${gammaStr}` +
      ` · closed form: ${limitStr}</small><br>` +
      `<small>step ${idx + 1} / ${total} · at k=${k}: term = ${termStr}` +
      ` · partial sum S_${k} = ${fmt(partial)}</small>`;
  },
});

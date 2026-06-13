// Widget 13.D — PUCT prior-shape laboratory (Chapter 13).
//
// A toy 9-action node. The reader sees how PUCT's policy prior π_prior
// guides search and how search corrects a bad prior. Fixed true
// Q*(a) values; reader chooses one of several prior shapes (uniform,
// peaked on the true best, peaked on a wrong action, "adversarial"
// inverse-Q). Slider for the PUCT constant c and total iteration N.
//
// PUCT score:
//   U(s, a) = Q(s, a) + c · π_prior(a) · √N(s) / (1 + N(s, a))
//
// Where Q(s, a) is the empirical mean reward observed for action a
// (initialized to 0). When a is selected, we sample a reward
// Q*(a) + small Gaussian noise. A toggle replaces PUCT with plain UCT
// (no prior factor) for comparison.
//
// Plots: bar chart of visit count N(s, a) vs the prior vs Q*. Bottom
// readout: cumulative regret = Σ_t (Q*(a*) - Q*(a_t)).
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-puct-prior-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/puct_prior/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const NUM_A = 9;
// Fixed true Q*(a) values — action 4 (middle, like tic-tac-toe centre) is best.
const Q_STAR = [0.20, 0.35, 0.15, 0.40, 0.90, 0.30, 0.10, 0.50, 0.25];

function makePrior(shape) {
  const p = new Array(NUM_A).fill(1 / NUM_A);
  if (shape === "uniform") return p;
  if (shape === "peaked_good") {
    // Strongly favours the true-best action 4.
    const v = NUM_A === 9 ? [0.04, 0.04, 0.04, 0.04, 0.68, 0.04, 0.04, 0.04, 0.04] : p;
    return v;
  }
  if (shape === "peaked_bad") {
    // Strongly favours action 0 (low Q*).
    return [0.68, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04];
  }
  if (shape === "adversarial") {
    // Inverse of Q* (normalised) — actively misleading prior.
    const inv = Q_STAR.map((q) => 1 - q);
    const Z = inv.reduce((a, b) => a + b, 0);
    return inv.map((v) => v / Z);
  }
  return p;
}

function runSearch({ prior, c, N, mode }) {
  const Q = new Array(NUM_A).fill(0);
  const visits = new Array(NUM_A).fill(0);
  const rng = mulberry32(7);
  let cumRegret = 0;
  const aStar = d3.maxIndex(Q_STAR);
  for (let t = 1; t <= N; t++) {
    const Ntot = t; // visits to root before this iter
    let bestA = 0;
    let bestU = -Infinity;
    for (let a = 0; a < NUM_A; a++) {
      const exploit = Q[a];
      let explore;
      if (mode === "puct") {
        explore = c * prior[a] * Math.sqrt(Ntot) / (1 + visits[a]);
      } else {
        // UCT: c · √(ln Ntot / N(a)); use small ε to avoid div-by-zero.
        explore = c * Math.sqrt(Math.log(Math.max(1, Ntot)) / Math.max(1, visits[a]));
      }
      const u = exploit + explore;
      if (u > bestU) {
        bestU = u;
        bestA = a;
      }
    }
    // Simulate: observe r = Q*(a) + small noise.
    const r = Q_STAR[bestA] + 0.05 * gauss(rng);
    visits[bestA] += 1;
    Q[bestA] += (r - Q[bestA]) / visits[bestA];
    cumRegret += Q_STAR[aStar] - Q_STAR[bestA];
  }
  return { Q, visits, cumRegret };
}

defineWidget({
  hostId: "ch13-puct-prior-widget",
  controls: {
    shape: {
      type: "select",
      label: "π_prior shape",
      options: [
        { value: "uniform",      label: "uniform" },
        { value: "peaked_good",  label: "peaked on true best (a=4)" },
        { value: "peaked_bad",   label: "peaked on a wrong action (a=0)" },
        { value: "adversarial",  label: "adversarial (inverse Q*)" },
      ],
      default: "uniform",
    },
    c:    { label: "c (PUCT constant)", min: 0.1, max: 5.0, step: 0.1, default: 1.5 },
    N:    { label: "N (iterations)", min: 10, max: 2000, step: 10, default: 200 },
    mode: {
      type: "select",
      label: "selection rule",
      options: [
        { value: "puct", label: "PUCT (uses prior)" },
        { value: "uct",  label: "UCT (no prior)" },
      ],
      default: "puct",
    },
  },
  slots: ["main", "regret"],
  render: (host, { shape, c, N, mode }, slots) => {
    const prior = makePrior(shape);
    const { Q, visits, cumRegret } = runSearch({ prior, c, N: Math.round(N), mode });
    const aStar = d3.maxIndex(Q_STAR);
    const aPicked = d3.maxIndex(visits);

    // Normalise visit distribution for comparison with prior.
    const visitDist = visits.map((v) => v / N);

    // Build bar data: three rows per action (prior, visit, Q*).
    const rows = [];
    for (let a = 0; a < NUM_A; a++) {
      rows.push({ a, kind: "π_prior", value: prior[a] });
      rows.push({ a, kind: "N(a)/N", value: visitDist[a] });
      rows.push({ a, kind: "Q*(a)",  value: Q_STAR[a] });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 280,
      marginLeft: 38,
      marginBottom: 38,
      x: { label: "action a (0..8)", domain: d3.range(NUM_A), tickFormat: (d) => String(d) },
      y: { label: "value", grid: true, domain: [0, 1] },
      color: {
        domain: ["π_prior", "N(a)/N", "Q*(a)"],
        range: [palette.muted, palette.secondary, palette.primary],
        legend: true,
      },
      marks: [
        Plot.barY(rows, {
          x: "a",
          y: "value",
          fill: "kind",
          fx: null,
          dx: (d) =>
            d.kind === "π_prior" ? -10 :
            d.kind === "N(a)/N" ?   0 :
                                  +10,
          insetLeft: 1,
          insetRight: 1,
        }),
        // Mark a*.
        Plot.tickX([{ a: aStar }], {
          x: "a",
          stroke: palette.warning,
          strokeWidth: 2,
        }),
      ],
    }));

    // Cumulative-regret comparison: re-run with the same c but the
    // alternate selection mode, so the reader sees PUCT vs UCT side by side.
    const stride = Math.max(1, Math.floor(N / 50));
    const compare = [];
    for (const m of ["puct", "uct"]) {
      const Qc = new Array(NUM_A).fill(0);
      const Vc = new Array(NUM_A).fill(0);
      const rng = mulberry32(7);
      let regret = 0;
      for (let t = 1; t <= N; t++) {
        let bestA = 0;
        let bestU = -Infinity;
        for (let a = 0; a < NUM_A; a++) {
          let explore;
          if (m === "puct") {
            explore = c * prior[a] * Math.sqrt(t) / (1 + Vc[a]);
          } else {
            explore = c * Math.sqrt(Math.log(Math.max(1, t)) / Math.max(1, Vc[a]));
          }
          const u = Qc[a] + explore;
          if (u > bestU) { bestU = u; bestA = a; }
        }
        const r = Q_STAR[bestA] + 0.05 * gauss(rng);
        Vc[bestA] += 1;
        Qc[bestA] += (r - Qc[bestA]) / Vc[bestA];
        regret += Q_STAR[aStar] - Q_STAR[bestA];
        if (t % stride === 0 || t === N) {
          compare.push({ t, regret, mode: m === "puct" ? "PUCT" : "UCT" });
        }
      }
    }

    slots.regret.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 180,
      x: { label: "iteration t", grid: true },
      y: { label: "cumulative regret", grid: true },
      color: {
        domain: ["PUCT", "UCT"],
        range: [palette.secondary, palette.danger],
        legend: true,
      },
      marks: [
        Plot.line(compare, {
          x: "t",
          y: "regret",
          stroke: "mode",
          strokeWidth: 2,
        }),
      ],
    }));

    slots.readout.innerHTML =
      `mode = ${mode.toUpperCase()} · c = ${fmt(c)} · N = ${N} · ` +
      `argmax_a N(a) = ${aPicked} ${aPicked === aStar ? "✓ (= a*)" : `≠ a* = ${aStar}`}<br>` +
      `<small>cum. regret = ${fmt(cumRegret)} ` +
      `(per-step ≈ ${fmt(cumRegret / N)})</small>`;
  },
});

// Widget 9.H — Deadly triad live demo on Baird's 7-state environment
// (Chapter 9).
//
// §9.9 explains the deadly triad: (FA × bootstrap × off-policy). No
// single leg causes divergence; only the combination does. This widget
// makes that *jointness* visible as a 2³ = 8-cell truth table.
//
// Three switches:
//   • FA       : linear vs tabular
//   • Bootstrap: TD(0) vs Monte-Carlo (n=20 truncated rollout)
//   • Data     : on-policy vs off-policy (Baird's adversarial behaviour
//                policy that visits state 7 too rarely)
//
// We run all 8 configurations to T steps and plot ‖θ‖₂ over time, then
// summarise as a 2×2×2 truth table of "converged / diverged" tiles.
//
// Baird's 7-state counterexample (1995) is the canonical
// linear-off-policy-bootstrap failure mode; it diverges to infinity for
// linear semi-gradient TD with the standard features [1, 2, 0, ..., 0]
// type that we use here. Flipping *any* of the three legs off makes it
// converge (or at least stay bounded over the budget T).
//
// Pattern: chapter markdown contains
//
//     <div id="ch9-triad-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/deadly_triad/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, fmt } from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

const N_STATES = 7;
// Baird's classical feature design: state s ∈ [0, 5] has feature
// 2 e_s + e_7 (a "shared" component on dim 7); state 6 has feature
// 2 e_7 + e_6. So we have 8 features total.
const N_FEAT = 8;

function bairdFeatures(s) {
  const phi = new Array(N_FEAT).fill(0);
  if (s < 6) {
    phi[s] = 2;
    phi[7] = 1;
  } else {
    // s === 6
    phi[6] = 1;
    phi[7] = 2;
  }
  return phi;
}

// Reward is always 0 in Baird; the divergence is *purely* from the
// off-policy bootstrap with linear FA.
const REWARD = 0.0;
const GAMMA = 0.99;
const ALPHA = 0.01;
const N_STEPS = 400;
const MC_N = 20; // truncated Monte-Carlo rollout horizon

// Behaviour policy: with prob 6/7 take "dashed" (jump to state 6); with
// prob 1/7 take "solid" (jump uniformly into states 0..5). This is
// Baird's classical adversarial behaviour — the target policy is
// "always solid" (uniform into 0..5).
function stepBehaviour(rand) {
  if (rand() < 6 / 7) return { ns: 6, action: "dashed" };
  return { ns: Math.floor(rand() * 6), action: "solid" };
}
function stepOnPolicy(rand) {
  // Pure "solid" — uniform into 0..5.
  return { ns: Math.floor(rand() * 6), action: "solid" };
}

// IS ratio for action="solid" under (target = always solid, behaviour
// = (6/7 dashed, 1/7 solid)). On dashed: π/μ = 0/(6/7) = 0. On solid:
// π/μ = 1/(1/7) = 7. We use these inside the off-policy TD update.
function isRatio(action, isOffPolicy) {
  if (!isOffPolicy) return 1.0;
  if (action === "dashed") return 0.0;
  return 7.0;
}

// Pre-roll the trajectory once: a long sequence of (s, action, ns)
// triples under the behaviour policy. We replay it for each config.
// Uses the shared LCG so all 8 configurations see the *same* random
// trajectory — making the comparison fair.
function rollTrajectory(seed, isOffPolicy) {
  const rng = lcg(seed);
  let s = 0;
  const traj = [];
  for (let t = 0; t < N_STEPS + MC_N + 10; t++) {
    const { ns, action } = isOffPolicy ? stepBehaviour(rng) : stepOnPolicy(rng);
    traj.push({ s, action, ns, r: REWARD });
    s = ns;
  }
  return traj;
}

// Run one configuration to T steps, return ‖θ‖₂ over time.
function runConfig({ fa, boot, offPolicy }) {
  const traj = rollTrajectory(7, offPolicy);
  // θ: linear FA over 8-dim features, OR tabular (7-vector keyed by state).
  let theta = new Array(fa === "linear" ? N_FEAT : N_STATES).fill(1.0);
  // Baird's classical init: θ = [1,1,1,1,1,1,10,1] — start with a large
  // weight on the shared feature to dramatise divergence.
  if (fa === "linear") theta[6] = 10.0;

  const norms = [{ t: 0, norm: l2(theta) }];

  for (let t = 0; t < N_STEPS; t++) {
    const step = traj[t];
    // Target: TD(0) = r + γ V(s'); MC = Σ_{k=0..MC_N-1} γ^k r_{t+k}
    // + γ^MC_N V(s_{t+MC_N}). All r are 0 in Baird, so MC target is
    // just γ^MC_N V(s_{t+MC_N}) — much smaller bootstrap dependence.
    let target;
    if (boot === "td") {
      target = REWARD + GAMMA * valueAt(theta, fa, step.ns);
    } else {
      // truncated MC — γ^MC_N · V(s_{t+MC_N})
      const tail = traj[t + MC_N - 1].ns;
      target = Math.pow(GAMMA, MC_N) * valueAt(theta, fa, tail);
    }
    const v = valueAt(theta, fa, step.s);
    const rho = isRatio(step.action, offPolicy);
    const delta = rho * (target - v);

    // Semi-gradient update.
    if (fa === "linear") {
      const phi = bairdFeatures(step.s);
      for (let k = 0; k < N_FEAT; k++) theta[k] += ALPHA * delta * phi[k];
    } else {
      // tabular: gradient is e_s.
      theta[step.s] += ALPHA * delta;
    }
    // Clip extreme blow-ups to keep the chart readable while still
    // making divergence visible (and to avoid Infinity propagating).
    for (let k = 0; k < theta.length; k++) {
      if (theta[k] > 1e6) theta[k] = 1e6;
      if (theta[k] < -1e6) theta[k] = -1e6;
    }
    norms.push({ t: t + 1, norm: l2(theta) });
  }
  return norms;
}

function valueAt(theta, fa, s) {
  if (fa === "linear") {
    const phi = bairdFeatures(s);
    let v = 0;
    for (let k = 0; k < N_FEAT; k++) v += phi[k] * theta[k];
    return v;
  }
  return theta[s];
}

function l2(v) {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

defineWidget({
  hostId: "ch9-triad-widget",
  controls: {
    fa:    { type: "select", label: "FA",        options: ["linear", "tabular"], default: "linear" },
    boot:  { type: "select", label: "Bootstrap", options: [{ value: "td", label: "TD(0)" }, { value: "mc", label: "Monte-Carlo" }], default: "td" },
    data:  { type: "select", label: "Data",      options: [{ value: "off", label: "off-policy" }, { value: "on", label: "on-policy" }], default: "off" },
  },
  slots: ["traj", "table"],
  render: (host, { fa, boot, data }, slots) => {
    // Run all 8 configurations so we can fill the truth table and
    // highlight the currently-selected one.
    const configs = [];
    for (const f of ["linear", "tabular"]) {
      for (const b of ["td", "mc"]) {
        for (const d of ["off", "on"]) {
          configs.push({ fa: f, boot: b, offPolicy: d === "off" });
        }
      }
    }
    const results = configs.map((c) => ({
      ...c,
      norms: runConfig(c),
    }));
    const finalNorms = results.map((r) => r.norms[r.norms.length - 1].norm);
    const DIVERGE_THRESH = 100; // ‖θ‖ above this is "diverged" given init ‖θ‖ ≈ 10.5

    // (a) Trajectory plot — show all 8 ‖θ‖ curves, highlight current.
    const long = [];
    for (const r of results) {
      const key = `${r.fa}/${r.boot}/${r.offPolicy ? "off" : "on"}`;
      const isCurrent = r.fa === fa && r.boot === boot && (r.offPolicy ? "off" : "on") === data;
      for (const point of r.norms) {
        long.push({ ...point, key, isCurrent });
      }
    }
    // Clip norm to a viewable range on a log axis.
    const yMax = 1e5;
    slots.traj.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 240,
      x: { label: "step t", grid: true, domain: [0, N_STEPS] },
      y: { type: "log", label: "‖θ‖₂", grid: true, domain: [1, yMax] },
      marks: [
        Plot.ruleY([DIVERGE_THRESH], { stroke: palette.danger, ...dashed }),
        // Faded grey lines for non-selected configs.
        Plot.line(long.filter((d) => !d.isCurrent), {
          x: "t", y: "norm", z: "key",
          stroke: palette.muted, strokeOpacity: 0.4, strokeWidth: 1,
        }),
        // Highlighted curve for selected config.
        Plot.line(long.filter((d) => d.isCurrent), {
          x: "t", y: "norm",
          stroke: palette.primary, strokeWidth: 2.5,
        }),
      ],
    }));

    // (b) Truth-table heatmap. Lay out as 4 rows × 2 cols:
    //   rows = (FA × bootstrap) ∈ {linear-TD, linear-MC, tab-TD, tab-MC}
    //   cols = data policy ∈ {on, off}
    const cells = results.map((r, idx) => ({
      row: `${r.fa}-${r.boot}`,
      col: r.offPolicy ? "off-policy" : "on-policy",
      diverged: finalNorms[idx] > DIVERGE_THRESH || !isFinite(finalNorms[idx]),
      finalNorm: finalNorms[idx],
      isCurrent: r.fa === fa && r.boot === boot && (r.offPolicy ? "off-policy" : "on-policy") === (data === "off" ? "off-policy" : "on-policy"),
    }));

    const rowOrder = ["linear-td", "linear-mc", "tabular-td", "tabular-mc"];
    slots.table.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      width: 360,
      marginLeft: 90,
      marginBottom: 36,
      x: { label: "data", domain: ["on-policy", "off-policy"] },
      y: { label: "FA × bootstrap", domain: rowOrder },
      marks: [
        Plot.cell(cells, {
          x: "col", y: "row",
          fill: (d) => (d.diverged ? palette.danger : palette.primary),
          fillOpacity: 0.85,
          stroke: (d) => (d.isCurrent ? "#fff" : "#222"),
          strokeWidth: (d) => (d.isCurrent ? 3 : 1),
        }),
        Plot.text(cells, {
          x: "col", y: "row",
          text: (d) => (d.diverged ? `✗  ‖θ‖≈${fmt(d.finalNorm)}` : `✓  ‖θ‖≈${fmt(d.finalNorm)}`),
          fill: "white",
          fontSize: 11,
        }),
      ],
    }));

    const selected = results.find((r) =>
      r.fa === fa && r.boot === boot && (r.offPolicy ? "off" : "on") === data,
    );
    const sel = selected.norms[selected.norms.length - 1].norm;
    const verdict = sel > DIVERGE_THRESH ? "DIVERGED" : "bounded";
    slots.readout.textContent =
      `selected: FA=${fa}, boot=${boot}, data=${data} → ‖θ‖ = ${fmt(sel)} (${verdict}) · ` +
      `diverged cells: ${cells.filter((c) => c.diverged).length}/8`;
  },
});

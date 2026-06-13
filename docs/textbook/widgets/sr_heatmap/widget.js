// Widget 17.F — Successor representation heatmap (Chapter 17, §17.5).
//
// 7×7 four-rooms gridworld. The successor representation
//
//   M^π(s, s') = E^π[ Σ_t γ^t · 1[s_t = s'] | s_0 = s ]
//
// is solved in closed form for the chosen policy via the linear system
// M = I + γ P^π M  ⇒  M = (I − γ P^π)^{-1}.
//
// The reader picks:
//   - the root state s  (slider over flat index),
//   - the discount γ,
//   - the policy π     (uniform-random or down-right-biased),
//   - a reward vector  (zero everywhere except one cell, set by another
//                       slider). V^π_R(s) = Σ_s' M^π(s, s') R(s') is shown
//                       as a side readout to demonstrate "change the
//                       reward, re-compute value instantly" — the SR's
//                       central practical claim.
//
// Mount: in §17.5 of Chapter 17.
//
//     <div id="ch17-sr-heatmap-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/sr_heatmap/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { gridAxes } from "../shared/plot.js";

// Four-rooms layout on a 7×7. # = wall, . = open. Doors at the
// midpoint of each interior wall.
const LAYOUT = [
  ".......",
  "...#...",
  "...#...",
  "##.##.#",
  "...#...",
  "...#...",
  ".......",
];
const N = 7;
const ACTIONS = [
  { dx: 0,  dy: -1 }, // up
  { dx: 1,  dy: 0  }, // right
  { dx: 0,  dy: 1  }, // down
  { dx: -1, dy: 0  }, // left
];

function isOpen(x, y) {
  if (x < 0 || x >= N || y < 0 || y >= N) return false;
  return LAYOUT[y][x] === ".";
}

function openStates() {
  const out = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (isOpen(x, y)) out.push({ x, y });
    }
  }
  return out;
}

// Build the transition matrix P^π under the chosen policy. P is
// (n_open × n_open). We index states by their position in the
// openStates() list.
function transitionMatrix(policy) {
  const states = openStates();
  const indexOf = new Map(states.map((s, i) => [`${s.x},${s.y}`, i]));
  const n = states.length;
  const P = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const { x, y } = states[i];
    const weights = ACTIONS.map((_, a) => {
      if (policy === "uniform") return 0.25;
      // "down-right" policy: prefer +y and +x (down-right corner-seeking).
      if (policy === "down-right") {
        return [0.05, 0.45, 0.45, 0.05][a]; // up, right, down, left
      }
      return 0.25;
    });
    for (let a = 0; a < ACTIONS.length; a++) {
      const nx = x + ACTIONS[a].dx;
      const ny = y + ACTIONS[a].dy;
      const target = isOpen(nx, ny) ? `${nx},${ny}` : `${x},${y}`;
      P[i][indexOf.get(target)] += weights[a];
    }
  }
  return { P, states, indexOf };
}

// Compute the row M^π(root, ·) of the successor representation by
// fixed-point iteration. M_row satisfies (I − γP)·M_row = e_root, i.e.
//
//   m[s'] = δ_{root, s'} + γ Σ_{s''} P[root → s''] · M[s'', s']
//
// Backward sweep: m_t = e_root + γ · m_{t-1} · P  (treating m as a row
// vector). Equivalently per entry, m_t[s'] = δ_{root,s'} + γ Σ_k m_{t-1}[k] · P[k][s'].
// Forward backups would compute the *column* M(·, fixed), which is the
// occupancy from arbitrary starts — we want the visitation from root,
// which is the row.
// iters cap = 5000 to ensure γ=0.99 (slider max) reaches the 1e-9 tol
// before pegging. log(1e-9)/log(0.99) ≈ 2063 iters needed at γ=0.99.
function successorVector(P, gamma, rootIdx, iters = 5000) {
  const n = P.length;
  let m = new Array(n).fill(0);
  m[rootIdx] = 1;
  for (let it = 0; it < iters; it++) {
    const next = new Array(n).fill(0);
    for (let sp = 0; sp < n; sp++) {
      let acc = sp === rootIdx ? 1 : 0;
      for (let k = 0; k < n; k++) acc += gamma * m[k] * P[k][sp];
      next[sp] = acc;
    }
    let maxDiff = 0;
    for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(next[i] - m[i]));
    m = next;
    if (maxDiff < 1e-9) break;
  }
  return m;
}

defineWidget({
  hostId: "ch17-sr-heatmap-widget",
  controls: {
    rootX: { label: "root x", min: 0, max: 6, step: 1, default: 1 },
    rootY: { label: "root y", min: 0, max: 6, step: 1, default: 1 },
    gamma: { label: "γ", min: 0.5, max: 0.99, step: 0.01, default: 0.9 },
    policy: {
      type: "select", label: "policy π",
      options: [
        { value: "uniform", label: "uniform random" },
        { value: "down-right", label: "down-right biased" },
      ],
      default: "uniform",
    },
    rewardX: { label: "reward cell x", min: 0, max: 6, step: 1, default: 5 },
    rewardY: { label: "reward cell y", min: 0, max: 6, step: 1, default: 5 },
  },
  slots: ["grid"],
  render: (host, params, slots) => {
    const { rootX, rootY, gamma, policy, rewardX, rewardY } = params;
    const { P, states, indexOf } = transitionMatrix(policy);

    // Resolve root: if reader picked a wall cell, snap to nearest open.
    let rk = `${rootX},${rootY}`;
    if (!indexOf.has(rk)) {
      // Find nearest open by L1.
      let best = null, bestD = Infinity;
      for (const s of states) {
        const d = Math.abs(s.x - rootX) + Math.abs(s.y - rootY);
        if (d < bestD) { bestD = d; best = s; }
      }
      rk = `${best.x},${best.y}`;
    }
    const rootIdx = indexOf.get(rk);
    const [rootGx, rootGy] = rk.split(",").map(Number);

    const m = successorVector(P, gamma, rootIdx);

    // Build cells for the heatmap. Walls stay grey; open cells get
    // their SR value. Domain is [0, max] so high-visitation cells
    // light up; the root tends to be the brightest.
    const cells = [];
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (!isOpen(x, y)) {
          cells.push({ x, y, kind: "wall", v: null });
          continue;
        }
        const i = indexOf.get(`${x},${y}`);
        cells.push({ x, y, kind: "open", v: m[i] });
      }
    }
    const vMax = Math.max(...cells.filter((c) => c.v !== null).map((c) => c.v));

    // V^π_R = Σ_s' M(s, s') · R(s')  with R = δ_{(rewardX, rewardY)}.
    const rk2 = `${rewardX},${rewardY}`;
    const vR = indexOf.has(rk2) ? m[indexOf.get(rk2)] : 0;

    slots.grid.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380, width: 380,
      ...gridAxes(N, { label: null, axis: null }, { y: { reverse: true } }),
      aspectRatio: 1,
      color: {
        type: "sqrt", scheme: "viridis", domain: [0, vMax || 1],
        label: "M^π(s, s')", legend: true,
      },
      marks: [
        Plot.cell(cells.filter((c) => c.kind === "wall"), {
          x: "x", y: "y", fill: () => "#444", inset: 1,
        }),
        Plot.cell(cells.filter((c) => c.kind === "open"), {
          x: "x", y: "y", fill: "v", inset: 1,
        }),
        // Root marker.
        Plot.text([{ x: rootGx, y: rootGy }], {
          x: "x", y: "y", text: () => "◉",
          fontSize: 18, fill: palette.danger,
        }),
        Plot.text([{ x: rootGx, y: rootGy }], {
          x: "x", y: "y", text: () => "root", dy: 14,
          fontSize: 9, fill: palette.danger,
        }),
        // Reward cell marker.
        Plot.text([{ x: rewardX, y: rewardY }], {
          x: "x", y: "y", text: () => "R",
          fontSize: 14, fill: palette.warning,
        }),
      ],
    }));

    const rewardOpen = indexOf.has(rk2);
    slots.readout.innerHTML =
      `<small>root = (${rootGx}, ${rootGy}) · ` +
      `reward cell = (${rewardX}, ${rewardY})${rewardOpen ? "" : " <strong>(wall)</strong>"} · ` +
      `V^π_R(root) = M^π(root, reward) = <strong>${fmt(vR)}</strong>. ` +
      `Drag the reward cell — V updates without re-learning M.</small>`;
  },
});

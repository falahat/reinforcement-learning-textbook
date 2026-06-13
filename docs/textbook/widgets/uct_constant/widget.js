// Widget 13.C — UCT exploration-constant explorer (Chapter 13).
//
// Same tic-tac-toe MCTS engine as 13.A, but the *slider* is the UCT
// constant c. The reader watches the tree shape morph from deep,
// narrow, exploit-heavy at small c to shallow, wide, explore-heavy
// at large c. Two derived metrics are displayed:
//
//   - visit-count entropy at the root H = -Σ p_a log p_a
//     where p_a = N(s_0, a) / Σ_b N(s_0, b)
//   - depth of the deepest principal-variation branch (visit-greedy
//     descent from the root).
//
// We always run a fixed number of iterations (no stepper), so the
// trade-off plays out across c at constant compute budget.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-uct-constant-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/uct_constant/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const X = "X", O = "O", EMPTY = ".";
const LINES = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6],
];

function winner(board) {
  for (const [a,b,c] of LINES) {
    if (board[a] !== EMPTY && board[a] === board[b] && board[b] === board[c]) {
      return board[a];
    }
  }
  if (!board.includes(EMPTY)) return "draw";
  return null;
}
function legal(board) {
  const acts = [];
  for (let i = 0; i < 9; i++) if (board[i] === EMPTY) acts.push(i);
  return acts;
}
function apply(b, i, p) { return b.slice(0, i) + p + b.slice(i + 1); }
function other(p) { return p === X ? O : X; }

function rolloutFrom(board, player, rng) {
  let b = board, p = player;
  while (true) {
    const w = winner(b);
    if (w !== null) return w === "draw" ? 0 : (w === X ? +1 : -1);
    const acts = legal(b);
    b = apply(b, acts[Math.floor(rng() * acts.length)], p);
    p = other(p);
  }
}

function makeNode(id, parent, action, board, player) {
  return { id, parent, action, board, player,
           children: [], untried: legal(board), N: 0, W: 0 };
}

function runMCTS(iters, c, seed) {
  const nodes = [makeNode(0, null, null, EMPTY.repeat(9), X)];
  const rng = mulberry32(seed);
  for (let i = 1; i <= iters; i++) {
    let cur = nodes[0];
    const path = [0];
    while (cur.untried.length === 0 && cur.children.length > 0) {
      let bestId = -1, bestU = -Infinity;
      for (const cid of cur.children) {
        const ch = nodes[cid];
        const meanRootPOV = ch.N > 0 ? ch.W / ch.N : 0;
        const exploit = cur.player === X ? meanRootPOV : -meanRootPOV;
        const explore = c * Math.sqrt(Math.log(cur.N + 1) / ch.N);
        const u = exploit + explore;
        if (u > bestU) { bestU = u; bestId = cid; }
      }
      cur = nodes[bestId];
      path.push(cur.id);
    }
    const w = winner(cur.board);
    if (w === null && cur.untried.length > 0) {
      const a = cur.untried.shift();
      const child = makeNode(nodes.length, cur.id, a, apply(cur.board, a, cur.player), other(cur.player));
      nodes.push(child);
      cur.children.push(child.id);
      cur = child;
      path.push(cur.id);
    }
    let value;
    const w2 = winner(cur.board);
    if (w2 !== null) value = w2 === "draw" ? 0 : (w2 === X ? +1 : -1);
    else value = rolloutFrom(cur.board, cur.player, rng);
    for (const nid of path) {
      nodes[nid].N += 1;
      nodes[nid].W += value;
    }
  }
  return nodes;
}

// Visit-count entropy at the root, in bits (log2).
function rootEntropy(nodes) {
  const root = nodes[0];
  const counts = root.children.map((cid) => nodes[cid].N);
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let H = 0;
  for (const c of counts) {
    if (c === 0) continue;
    const p = c / total;
    H -= p * Math.log2(p);
  }
  return H;
}

// Principal-variation depth: walk greedy-by-visit from root until
// no children.
function pvDepth(nodes) {
  let cur = 0, depth = 0;
  while (nodes[cur].children.length > 0) {
    let bestId = -1, bestN = -1;
    for (const cid of nodes[cur].children) {
      if (nodes[cid].N > bestN) { bestN = nodes[cid].N; bestId = cid; }
    }
    if (bestId === -1) break;
    cur = bestId;
    depth += 1;
  }
  return depth;
}

// Maximum branching factor across explored nodes.
function maxBranching(nodes) {
  let m = 0;
  for (const n of nodes) if (n.children.length > m) m = n.children.length;
  return m;
}

defineWidget({
  hostId: "ch13-uct-constant-widget",
  controls: {
    c:     { label: "c (UCT constant)", min: 0.0, max: 4.0, step: 0.05, default: 1.4 },
    iters: { label: "iterations", min: 20, max: 800, step: 10, default: 200 },
  },
  slots: ["main", "sweep"],
  render: (host, { c, iters }, slots) => {
    const itersI = Math.round(iters);

    // Root-child visit distribution at the slider's c.
    const nodes = runMCTS(itersI, c, 31);
    const root = nodes[0];
    const visitData = [];
    for (let a = 0; a < 9; a++) {
      const found = root.children.find((cid) => nodes[cid].action === a);
      const n = found !== undefined ? nodes[found].N : 0;
      const q = found !== undefined && nodes[found].N > 0
        ? nodes[found].W / nodes[found].N
        : 0;
      visitData.push({ action: a, count: n, q });
    }

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 220,
      x: { label: "root action (cell index)", domain: d3.range(9),
           tickFormat: (d) => String(d) },
      y: { label: "visit count N(root, a)", grid: true },
      marks: [
        Plot.barY(visitData, {
          x: "action",
          y: "count",
          fill: (d) => d.count === 0 ? palette.muted : palette.primary,
          fillOpacity: 0.85,
        }),
        Plot.text(visitData.filter((d) => d.count > 0), {
          x: "action",
          y: "count",
          text: (d) => String(d.count),
          textAnchor: "middle",
          dy: -6,
          fill: "white",
          fontSize: 10,
        }),
      ],
    }));

    // Sweep across c to plot tree-shape metrics.
    const cs = d3.range(0.0, 3.05, 0.1);
    const sweep = [];
    for (const cv of cs) {
      const nn = runMCTS(itersI, cv, 31);
      sweep.push({
        c: cv,
        entropy: rootEntropy(nn),
        pv: pvDepth(nn),
        branching: maxBranching(nn),
        size: nn.length,
      });
    }
    // Normalise so the two y-series share a domain (entropy is in
    // [0, log2(9)] ≈ [0, 3.17]; depth grows from ~0 to ~9).
    const maxH = Math.log2(9);

    slots.sweep.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 200,
      x: { label: "c (UCT constant)", grid: true, domain: [0, 3.0] },
      y: { label: "value", grid: true },
      color: {
        domain: ["entropy (bits)", "PV depth", "max branching"],
        range: [palette.warning, palette.secondary, palette.accent],
        legend: true,
      },
      marks: [
        Plot.line(sweep, {
          x: "c",
          y: "entropy",
          stroke: palette.warning,
          strokeWidth: 2,
        }),
        Plot.line(sweep, {
          x: "c",
          y: "pv",
          stroke: palette.secondary,
          strokeWidth: 2,
        }),
        Plot.line(sweep, {
          x: "c",
          y: "branching",
          stroke: palette.accent,
          strokeWidth: 2,
          ...dashed,
        }),
        Plot.ruleX([c], { stroke: palette.danger, ...dashed }),
        Plot.text(
          [{ x: c, y: maxH * 1.05, label: `c = ${fmt(c)}` }],
          { x: "x", y: "y", text: "label", textAnchor: "start", dx: 4,
            fill: palette.danger, ...annotation },
        ),
      ],
    }));

    const Hnow = rootEntropy(nodes);
    const pvNow = pvDepth(nodes);
    slots.readout.innerHTML =
      `c = ${fmt(c)} · iters = ${itersI} · ` +
      `H(root visits) = ${fmt(Hnow)} bits / ${fmt(maxH)} max · ` +
      `PV depth = ${pvNow}<br>` +
      `<small>small c → exploit, deep narrow tree · ` +
      `large c → explore, shallow wide tree</small>`;
  },
});

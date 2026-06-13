// Widget 13.A — MCTS tree animator (Chapter 13).
//
// Animates the four MCTS phases (selection → expansion → simulation →
// backpropagation) on tic-tac-toe. Each "frame" is one full UCT
// iteration. The reader can scrub forward and watch the tree grow,
// see which child is selected at each level, watch a rollout produce
// a return, and observe the backpropagation update visit counts and Q.
//
// Tree-state representation: a node has
//   { id, parent, action, untriedActions, children, N, W, player, board }
// where `board` is a length-9 string with chars '.', 'X', 'O', and
// `player` is the side to move at this node. Q(s, a) is read off
// the *child* node as W_child / N_child (from the perspective of the
// player to move at the child, then negated when scoring as parent).
//
// Display: tree laid out top-down with d3.cluster. Selection path
// is highlighted in red; the leaf being expanded in orange; rollout
// outcome shown as a +1 / -1 / 0 annotation under the leaf.
//
// Pattern: chapter markdown contains
//
//     <div id="ch13-mcts-tree-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/mcts_tree/widget.js"></script>

import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { palette, fmt } from "../shared/helpers.js";
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

function legalActions(board) {
  const acts = [];
  for (let i = 0; i < 9; i++) if (board[i] === EMPTY) acts.push(i);
  return acts;
}

function applyMove(board, idx, player) {
  return board.slice(0, idx) + player + board.slice(idx + 1);
}

function other(p) { return p === X ? O : X; }

// Rollout policy: uniform random until terminal. Return value from
// the perspective of the root player (X).
function rolloutFrom(board, player, rng) {
  let b = board, p = player;
  while (true) {
    const w = winner(b);
    if (w !== null) {
      if (w === "draw") return 0;
      return w === X ? +1 : -1;
    }
    const acts = legalActions(b);
    const a = acts[Math.floor(rng() * acts.length)];
    b = applyMove(b, a, p);
    p = other(p);
  }
}

function makeNode(id, parent, action, board, player) {
  return {
    id,
    parent,
    action,
    board,
    player,
    children: [],
    untried: legalActions(board),
    N: 0,
    W: 0,    // total reward from root player's POV
  };
}

// One full MCTS iteration. Mutates the tree. Returns an "event log"
// for the stepper to display.
function uctIteration(nodes, rootId, c, rng) {
  // 1. Selection.
  let cur = nodes[rootId];
  const path = [cur.id];
  while (cur.untried.length === 0 && cur.children.length > 0) {
    // Pick UCT-max child. Q is from the perspective of the player to
    // move at `cur`. The root player is fixed; we compute the child's
    // value from root POV as W_child / N_child (sign already correct
    // because W stores root-POV reward).
    let bestId = -1, bestU = -Infinity;
    for (const childId of cur.children) {
      const ch = nodes[childId];
      // Exploit term: value from the perspective of the parent's
      // *mover* — i.e., we want high reward if parent is the root
      // player (X), low if parent is O.
      const meanRootPOV = ch.N > 0 ? ch.W / ch.N : 0;
      const exploit = cur.player === X ? meanRootPOV : -meanRootPOV;
      const explore = c * Math.sqrt(Math.log(cur.N + 1) / ch.N);
      const u = exploit + explore;
      if (u > bestU) { bestU = u; bestId = childId; }
    }
    cur = nodes[bestId];
    path.push(cur.id);
  }

  // 2. Expansion (if non-terminal).
  let expanded = null;
  const w = winner(cur.board);
  if (w === null && cur.untried.length > 0) {
    const a = cur.untried.shift();
    const newBoard = applyMove(cur.board, a, cur.player);
    const child = makeNode(nodes.length, cur.id, a, newBoard, other(cur.player));
    nodes.push(child);
    cur.children.push(child.id);
    expanded = child;
    cur = child;
    path.push(cur.id);
  }

  // 3. Simulation.
  let value;
  const w2 = winner(cur.board);
  if (w2 !== null) {
    value = w2 === "draw" ? 0 : (w2 === X ? +1 : -1);
  } else {
    value = rolloutFrom(cur.board, cur.player, rng);
  }

  // 4. Backpropagation.
  for (const nid of path) {
    nodes[nid].N += 1;
    nodes[nid].W += value;
  }

  return { path, expandedId: expanded ? expanded.id : null, value };
}

// Simple top-down layout: x position is interpolated within the
// parent's allocated x-range, y is depth. We re-layout after every
// frame (cheap for small trees).
function layoutTree(nodes, rootId, width, height, padTop, padBot) {
  // BFS to compute depth and per-depth count.
  const depthOf = new Map();
  depthOf.set(rootId, 0);
  const queue = [rootId];
  let maxDepth = 0;
  while (queue.length > 0) {
    const id = queue.shift();
    const d = depthOf.get(id);
    if (d > maxDepth) maxDepth = d;
    for (const cid of nodes[id].children) {
      depthOf.set(cid, d + 1);
      queue.push(cid);
    }
  }
  // Recursive in-order x assignment, weighted by subtree size.
  function subtreeSize(id) {
    const ch = nodes[id].children;
    if (ch.length === 0) return 1;
    let s = 0;
    for (const cid of ch) s += subtreeSize(cid);
    return s;
  }
  const positions = new Map();
  function assign(id, xLeft, xRight) {
    const xMid = (xLeft + xRight) / 2;
    const d = depthOf.get(id);
    const y = padTop + (maxDepth > 0 ? (d * (height - padTop - padBot)) / Math.max(1, maxDepth) : 0);
    positions.set(id, { x: xMid, y });
    const ch = nodes[id].children;
    if (ch.length === 0) return;
    const total = ch.reduce((s, cid) => s + subtreeSize(cid), 0);
    let xCur = xLeft;
    for (const cid of ch) {
      const w = (subtreeSize(cid) / total) * (xRight - xLeft);
      assign(cid, xCur, xCur + w);
      xCur += w;
    }
  }
  assign(rootId, 4, width - 4);
  return positions;
}

function renderTreeSVG(nodes, rootId, frame, c) {
  const width = 580, height = 320, padTop = 18, padBot = 30;
  const positions = layoutTree(nodes, rootId, width, height, padTop, padBot);

  const xmlns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(xmlns, "svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.style.font = "10px sans-serif";

  const onPath = new Set(frame ? frame.path : []);
  const expandedId = frame ? frame.expandedId : null;

  // Edges.
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].parent === null) continue;
    const p = positions.get(nodes[i].parent);
    const ch = positions.get(i);
    const line = document.createElementNS(xmlns, "line");
    line.setAttribute("x1", p.x);
    line.setAttribute("y1", p.y + 14);
    line.setAttribute("x2", ch.x);
    line.setAttribute("y2", ch.y - 14);
    const onSel = onPath.has(i) && onPath.has(nodes[i].parent);
    line.setAttribute("stroke", onSel ? palette.danger : "#666");
    line.setAttribute("stroke-width", onSel ? 2 : 1);
    svg.appendChild(line);
  }

  // Nodes.
  for (let i = 0; i < nodes.length; i++) {
    const pos = positions.get(i);
    const n = nodes[i];
    const g = document.createElementNS(xmlns, "g");
    g.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    const r = 12;
    const circle = document.createElementNS(xmlns, "circle");
    circle.setAttribute("r", r);
    let fill = n.player === X ? "#2c3e50" : "#444";
    if (i === expandedId) fill = palette.warning;
    if (onPath.has(i)) fill = palette.danger;
    circle.setAttribute("fill", fill);
    circle.setAttribute("stroke", "#111");
    g.appendChild(circle);
    // Player label (whose move at this node).
    const label = document.createElementNS(xmlns, "text");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("dy", "0.32em");
    label.setAttribute("fill", "white");
    label.setAttribute("font-size", "9");
    label.textContent = n.player;
    g.appendChild(label);
    // Stats below the circle.
    const stats = document.createElementNS(xmlns, "text");
    stats.setAttribute("text-anchor", "middle");
    stats.setAttribute("y", r + 12);
    stats.setAttribute("fill", "#ccc");
    stats.setAttribute("font-size", "9");
    const q = n.N > 0 ? (n.W / n.N).toFixed(2) : "—";
    stats.textContent = `N=${n.N} Q=${q}`;
    g.appendChild(stats);
    svg.appendChild(g);
  }

  // Legend.
  const legend = document.createElementNS(xmlns, "text");
  legend.setAttribute("x", 6);
  legend.setAttribute("y", height - 6);
  legend.setAttribute("fill", "#888");
  legend.setAttribute("font-size", "10");
  legend.textContent =
    `c = ${c.toFixed(2)} · red path = selection · orange = expanded ` +
    `· value reported from X's POV`;
  svg.appendChild(legend);

  return svg;
}

function renderBoardSVG(board, title) {
  const size = 140, cell = size / 3;
  const xmlns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(xmlns, "svg");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size + 18);
  svg.setAttribute("viewBox", `0 0 ${size} ${size + 18}`);
  svg.style.font = "11px sans-serif";

  if (title) {
    const t = document.createElementNS(xmlns, "text");
    t.setAttribute("x", size / 2);
    t.setAttribute("y", 12);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#aaa");
    t.setAttribute("font-size", "10");
    t.textContent = title;
    svg.appendChild(t);
  }

  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3), col = i % 3;
    const rect = document.createElementNS(xmlns, "rect");
    rect.setAttribute("x", col * cell);
    rect.setAttribute("y", 18 + r * cell);
    rect.setAttribute("width", cell);
    rect.setAttribute("height", cell);
    rect.setAttribute("fill", "#1b1b1b");
    rect.setAttribute("stroke", "#666");
    svg.appendChild(rect);
    if (board[i] !== EMPTY) {
      const tx = document.createElementNS(xmlns, "text");
      tx.setAttribute("x", col * cell + cell / 2);
      tx.setAttribute("y", 18 + r * cell + cell / 2 + 6);
      tx.setAttribute("text-anchor", "middle");
      tx.setAttribute("fill", board[i] === X ? palette.secondary : palette.danger);
      tx.setAttribute("font-size", "18");
      tx.textContent = board[i];
      svg.appendChild(tx);
    }
  }
  return svg;
}

defineStepper({
  hostId: "ch13-mcts-tree-widget",
  controls: {
    iters: { label: "iterations", min: 1, max: 50, step: 1, default: 12 },
    c:     { label: "c (UCT)", min: 0.1, max: 3.0, step: 0.05, default: 1.4 },
  },
  slots: ["tree", "board"],
  trajectory: ({ iters, c }) => {
    // Fixed starting position: empty board, X to move.
    const nodes = [];
    nodes.push(makeNode(0, null, null, EMPTY.repeat(9), X));
    const rng = mulberry32(31);

    const frames = [];
    // Frame 0: empty tree.
    frames.push({
      iter: 0,
      nodes: cloneNodes(nodes),
      path: [0],
      expandedId: null,
      value: null,
      leafBoard: nodes[0].board,
    });

    for (let i = 1; i <= iters; i++) {
      const log = uctIteration(nodes, 0, c, rng);
      const leafBoard = nodes[log.path[log.path.length - 1]].board;
      frames.push({
        iter: i,
        nodes: cloneNodes(nodes),
        path: log.path,
        expandedId: log.expandedId,
        value: log.value,
        leafBoard,
      });
    }
    return frames;
  },
  playIntervalMs: 700,
  render: (host, frame, idx, total, params, slots) => {
    slots.tree.replaceChildren(renderTreeSVG(frame.nodes, 0, frame, params.c));

    const valStr = frame.value === null
      ? "—"
      : frame.value > 0 ? "+1 (X wins)"
      : frame.value < 0 ? "−1 (O wins)"
      : "0 (draw)";
    slots.board.replaceChildren(renderBoardSVG(frame.leafBoard, "rollout leaf"));

    // Root child summary: visit counts so the reader can see the
    // recommended move.
    const root = frame.nodes[0];
    let topChild = -1, topN = -1;
    for (const cid of root.children) {
      if (frame.nodes[cid].N > topN) { topN = frame.nodes[cid].N; topChild = cid; }
    }
    const topAction = topChild >= 0 ? frame.nodes[topChild].action : null;
    const moveStr = topAction === null
      ? "—"
      : `cell ${topAction} (N=${topN}, Q=${(frame.nodes[topChild].W / Math.max(1, frame.nodes[topChild].N)).toFixed(2)})`;

    slots.readout.innerHTML =
      `iteration ${frame.iter} / ${total - 1} · rollout return = ${valStr}<br>` +
      `<small>argmax-visits root move: ${moveStr} · root N = ${root.N}</small>`;
  },
});

// Deep-clone the nodes array so frames don't share state.
function cloneNodes(nodes) {
  return nodes.map((n) => ({
    id: n.id,
    parent: n.parent,
    action: n.action,
    board: n.board,
    player: n.player,
    children: n.children.slice(),
    untried: n.untried.slice(),
    N: n.N,
    W: n.W,
  }));
}

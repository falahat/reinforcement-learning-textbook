// Widget 11.C — Interactive neural network explorer (Chapter 11, §9.1).
//
// A small MLP rendered as an SVG diagram. Every neuron and weight is
// directly clickable / annotated. The reader scrubs forward / backward
// through a single, linear timeline that walks the entire forward
// pass, then the loss, then the entire backward pass, then one SGD
// update — twelve frames total. Same pacing as the matmul widget:
// one click = one logical step, "play" auto-advances.
//
// **Uses the shared NN library** (`../shared/nn.js`). The math is the
// library's; this widget is the visualisation harness.
//
// Mount: `<div id="ch11-nn-explorer-widget" class="textbook-widget"></div>`

import { defineStepper } from "../shared/stepper.js";
import { palette } from "../shared/helpers.js";
import { splitmix64 } from "../shared/random.js";
import {
  Tensor, Tape, Sequential, Linear, Tanh, Sigmoid, Sgd,
} from "../shared/nn.js";

const LAYER_SIZES = [2, 4, 1];
const ACT_NAMES = ["tanh", "sigmoid"];

// ── Build the model + record a full forward/backward/update trace ──

function buildAndTrace({ x1, x2, target, lr, seed }) {
  const rng = splitmix64(seed);
  const model = new Sequential([
    new Linear(LAYER_SIZES[0], LAYER_SIZES[1], "xavier", rng),
    Tanh(),
    new Linear(LAYER_SIZES[1], LAYER_SIZES[2], "xavier", rng),
    Sigmoid(),
  ]);

  const tape = new Tape();
  const xt = new Tensor(new Float32Array([x1, x2]), [1, 2]);
  const targetT = new Tensor(new Float32Array([target]), [1, 1]);

  // Forward, capturing each named intermediate.
  const [linear1, tanh, linear2, sigmoid] = model.layers;
  const z1 = linear1.forward(tape, xt);
  const h  = tanh.forward(tape, z1);
  const z2 = linear2.forward(tape, h);
  const y  = sigmoid.forward(tape, z2);

  // Loss = (y - target)^2 (numel 1 → meanAll == identity).
  const d  = tape.sub(y, targetT);
  const sq = tape.square(d);
  const lossT = tape.meanAll(sq);

  const grads = tape.backwardFull(lossT);

  // Snapshot weights BEFORE the update, then compute what they'd be
  // AFTER one SGD step. We don't actually mutate the model so the
  // user can scrub backward and see the original weights.
  const beforeW = model.layers.map(l =>
    l.weight ? { W: l.weight.data.slice(), b: l.bias.data.slice() } : null);
  const sgd = new Sgd(lr);
  // Clone the model so the update is on a copy.
  const updatedModel = new Sequential(model.layers.map(l => l));   // shallow
  sgd.step(updatedModel, grads.params);
  const afterW = model.layers.map(l =>
    l.weight ? { W: l.weight.data.slice(), b: l.bias.data.slice() } : null);

  return {
    model,
    tape,
    grads,
    inputTensor: xt,
    targetTensor: targetT,
    hiddenPre: z1, hiddenAct: h,
    outputPre: z2, outputAct: y,
    lossTensor: lossT,
    beforeW, afterW,
  };
}

// ── Frame schedule (12 frames). Each frame names what's BEEN computed
//    up to and including this step. The render function then knows
//    which nodes / edges / labels to light up. ──

function makeFrames() {
  return [
    { kind: "init",     desc: "initial state — inputs only" },
    { kind: "fwd-lin",  layer: 0, desc: "forward: compute hidden pre-activation z₁ = W₁·x + b₁" },
    { kind: "fwd-act",  layer: 0, desc: "forward: apply tanh — h = tanh(z₁)" },
    { kind: "fwd-lin",  layer: 1, desc: "forward: compute output pre-activation z₂ = W₂·h + b₂" },
    { kind: "fwd-act",  layer: 1, desc: "forward: apply sigmoid — ŷ = σ(z₂)" },
    { kind: "loss",     desc: "compute loss L = (ŷ − target)²" },
    { kind: "bwd-seed", desc: "backward: seed ∂L/∂ŷ = 2(ŷ − target)" },
    { kind: "bwd-act",  layer: 1, desc: "backward: through sigmoid — ∂L/∂z₂ = ∂L/∂ŷ · ŷ(1−ŷ)" },
    { kind: "bwd-lin",  layer: 1, desc: "backward: through W₂ — ∂L/∂h = W₂ᵀ · ∂L/∂z₂  (and ∂L/∂W₂ = ∂L/∂z₂ ⊗ h)" },
    { kind: "bwd-act",  layer: 0, desc: "backward: through tanh — ∂L/∂z₁ = ∂L/∂h · (1 − h²)" },
    { kind: "bwd-lin",  layer: 0, desc: "backward: through W₁ — ∂L/∂x = W₁ᵀ · ∂L/∂z₁  (and ∂L/∂W₁ = ∂L/∂z₁ ⊗ x)" },
    { kind: "update",   desc: "apply SGD step: W ← W − lr · ∂L/∂W" },
  ];
}

// ── SVG geometry ──
const VIEW_W = 760, VIEW_H = 440;
const COLS_X = LAYER_SIZES.map((_, i) =>
  60 + i * ((VIEW_W - 120) / (LAYER_SIZES.length - 1)));
const NODE_R = 24;

function colY(layerIdx, nodeIdx) {
  const n = LAYER_SIZES[layerIdx];
  const top = 60, bottom = VIEW_H - 60;
  if (n === 1) return (top + bottom) / 2;
  return top + (bottom - top) * (nodeIdx / (n - 1));
}

function divColor(v) {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const r = 220, g = Math.round(220 - 170 * t), b = Math.round(220 - 170 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const r = Math.round(220 + 170 * t), g = Math.round(220 + 170 * t), b = 240;
  return `rgb(${r}, ${g}, ${b})`;
}

function fmt(v) { return v.toFixed(3); }

// What's "computed yet" by phase. Used to grey out future state and
// highlight the just-computed thing.
function computedAt(frame) {
  // Returns: which layer-index nodes have a known activation (so we draw values)
  //          which layers have known backward grads (so we draw ∂L)
  //          which highlight to put on the just-computed thing
  const fwdLayer = (() => {
    switch (frame.kind) {
      case "init":     return -1;            // no forward yet — inputs only
      case "fwd-lin":  return frame.layer;   // pre-activation of this layer known
      case "fwd-act":  return frame.layer;   // post-activation of this layer known
      case "loss":
      case "bwd-seed":
      case "bwd-act":
      case "bwd-lin":
      case "update":
        return LAYER_SIZES.length - 1;       // full forward done
      default: throw new Error(`unknown frame ${frame.kind}`);
    }
  })();

  // Which backward grads are known so far:
  //   bwd-seed:   ∂L/∂ŷ (output node grad)
  //   bwd-act 1:  ∂L/∂z₂
  //   bwd-lin 1:  ∂L/∂h and ∂L/∂W₂
  //   bwd-act 0:  ∂L/∂z₁
  //   bwd-lin 0:  ∂L/∂x and ∂L/∂W₁
  let bwdProgress = -1;   // 0..4 ; -1 == none
  if (frame.kind === "bwd-seed") bwdProgress = 0;
  if (frame.kind === "bwd-act" && frame.layer === 1) bwdProgress = 1;
  if (frame.kind === "bwd-lin" && frame.layer === 1) bwdProgress = 2;
  if (frame.kind === "bwd-act" && frame.layer === 0) bwdProgress = 3;
  if (frame.kind === "bwd-lin" && frame.layer === 0) bwdProgress = 4;
  if (frame.kind === "update") bwdProgress = 4;

  return { fwdLayer, bwdProgress };
}

// ── Render one frame ──

function renderFrame(host, frame, idx, total, params, slots, trace) {
  const { fwdLayer, bwdProgress } = computedAt(frame);
  const lossKnown = ["loss","bwd-seed","bwd-act","bwd-lin","update"].includes(frame.kind);
  const showUpdate = frame.kind === "update";

  // Decide which weight set to render: BEFORE for everything except
  // the final "update" frame, AFTER for that.
  const weights = showUpdate ? trace.afterW : trace.beforeW;

  // Highlight set: which edges + nodes are the "stars" of this frame.
  const highlight = activeHighlight(frame);

  const parts = [];

  // ── Edges (drawn first so nodes sit on top) ──
  for (let li = 0; li < LAYER_SIZES.length - 1; li++) {
    const Wmat = weights[li * 2].W;
    const inN = LAYER_SIZES[li], outN = LAYER_SIZES[li + 1];
    for (let j = 0; j < outN; j++) {
      for (let i = 0; i < inN; i++) {
        const w = Wmat[j * inN + i];
        const x1 = COLS_X[li] + NODE_R, y1 = colY(li, i);
        const x2 = COLS_X[li + 1] - NODE_R, y2 = colY(li + 1, j);
        const baseColor = w >= 0 ? palette.secondary : palette.danger;

        const isHot = (highlight.edges?.find(([L, jj, ii]) => L === li && jj === j && ii === i));
        let strokeW = Math.max(1, Math.min(8, Math.abs(w) * 4));
        let colour = baseColor;
        let opacity = 0.4;          // grey-ish if not on the path of this frame
        if (isHot) {
          opacity = 1.0;
          strokeW = Math.max(2.5, strokeW + 1);
        } else if (highlight.activeLayer === li || lossKnown) {
          opacity = 0.85;
        }

        // Backward weight-grad arrows: if we're past bwd-lin for this layer,
        // colour the edge by sign of the grad and add a small arrowhead.
        let wGrad = 0;
        if ((li === 1 && bwdProgress >= 2) || (li === 0 && bwdProgress >= 4)) {
          const linear = li === 0 ? trace.model.layers[0] : trace.model.layers[2];
          const g = trace.grads.params.get(linear.wId);
          wGrad = g ? g[j * inN + i] : 0;
        }

        parts.push(
          `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
                 stroke="${colour}" stroke-opacity="${opacity}"
                 stroke-width="${strokeW}">
             <title>w[${j},${i}] = ${fmt(w)}${wGrad !== 0 ? `   ∂L/∂w = ${fmt(wGrad)}` : ""}</title>
           </line>`);

        // Weight label.
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        parts.push(
          `<text x="${mx}" y="${my - 4}" text-anchor="middle"
                 font-size="9" fill="#cdd" opacity="${opacity}"
                 pointer-events="none">${fmt(w)}</text>`);
      }
    }
  }

  // ── Nodes ──
  for (let li = 0; li < LAYER_SIZES.length; li++) {
    const n = LAYER_SIZES[li];
    const known = nodeKnown(li, frame, fwdLayer);
    for (let i = 0; i < n; i++) {
      const cx = COLS_X[li], cy = colY(li, i);

      // Activation value — only show if "known" yet (else grey placeholder).
      let a = null;
      if (known) {
        if (li === 0) a = trace.inputTensor.data[i];
        else if (li === 1) a = trace.hiddenAct.data[i];
        else if (li === 2) a = trace.outputAct.data[i];
      }

      const fill = a !== null ? divColor(a) : "#2a3540";
      const stroke =
        highlight.activeNodes?.find(([L, ii]) => L === li && ii === i) ? palette.warning
        : "#000";
      const strokeW = stroke !== "#000" ? 4 : 1;

      // Compute backward grad ∂L/∂a if available for this layer at
      // this frame.
      let gradVal = null;
      if (bwdProgress >= 0) {
        if (li === 2 && bwdProgress >= 0) gradVal = trace.grads.nodeGrads[trace.outputAct.node]?.[i];
        else if (li === 1 && bwdProgress >= 2) gradVal = trace.grads.nodeGrads[trace.hiddenAct.node]?.[i];
        else if (li === 0 && bwdProgress >= 4) gradVal = trace.grads.nodeGrads[trace.inputTensor.node]?.[i];
      }

      parts.push(
        `<circle cx="${cx}" cy="${cy}" r="${NODE_R}"
                 fill="${fill}" stroke="${stroke}" stroke-width="${strokeW}">
           <title>${a !== null ? `a = ${fmt(a)}` : "not yet computed"}${gradVal != null ? `   ∂L/∂a = ${fmt(gradVal)}` : ""}</title>
         </circle>`);

      // Value text inside the circle (only if known).
      if (a !== null) {
        parts.push(
          `<text x="${cx}" y="${cy + 4}" text-anchor="middle"
                 font-size="11" font-weight="600" fill="#000"
                 pointer-events="none">${fmt(a)}</text>`);
      }

      // Gradient text below the node (only if we're in the backward path
      // and the grad for this layer is known).
      if (gradVal != null) {
        parts.push(
          `<text x="${cx}" y="${cy + NODE_R + 14}" text-anchor="middle"
                 font-size="10" fill="${palette.warning}" pointer-events="none">∂L = ${fmt(gradVal)}</text>`);
      }

      // Bias label (for hidden + output layers, always — they're constant per trace).
      if (li > 0) {
        const linear = li === 1 ? trace.model.layers[0] : trace.model.layers[2];
        const b = linear.bias.data[i];
        const dy = gradVal != null ? NODE_R + 28 : NODE_R + 14;
        parts.push(
          `<text x="${cx}" y="${cy + dy}" text-anchor="middle"
                 font-size="9" fill="#aaa" pointer-events="none">b=${fmt(b)}</text>`);
      }
    }
  }

  // ── Headers + loss callout ──
  const headers = ["input x", `hidden (${ACT_NAMES[0]})`, `output (${ACT_NAMES[1]})`];
  for (let li = 0; li < LAYER_SIZES.length; li++) {
    parts.push(
      `<text x="${COLS_X[li]}" y="${VIEW_H - 24}" text-anchor="middle"
             font-size="11" fill="#bbb">${headers[li]}</text>`);
  }

  // Loss callout.
  if (lossKnown) {
    const lossVal = trace.lossTensor.data[0];
    parts.push(
      `<g transform="translate(${VIEW_W - 140}, 30)">
         <rect x="0" y="0" width="130" height="58" rx="4"
               fill="#0c1620" stroke="${palette.warning}" stroke-width="2"/>
         <text x="65" y="18" text-anchor="middle" font-size="10" fill="#bbb">loss</text>
         <text x="65" y="36" text-anchor="middle" font-size="13" fill="#fff">target = ${fmt(params.target)}</text>
         <text x="65" y="52" text-anchor="middle" font-size="13" fill="${palette.warning}">L = ${fmt(lossVal)}</text>
       </g>`);
  }

  // ── Step description banner at top of the SVG ──
  parts.push(
    `<rect x="0" y="0" width="${VIEW_W}" height="36" fill="#0c1620"/>`);
  parts.push(
    `<text x="14" y="22" font-size="13" fill="${idx === 0 ? "#bbb" : "#fff"}">step ${idx + 1} / ${total}: ${escapeHtml(frame.desc)}</text>`);

  slots.main.innerHTML =
    `<svg viewBox="0 0 ${VIEW_W} ${VIEW_H}" style="width:100%;max-width:${VIEW_W}px;background:#1d2935;border-radius:4px;display:block;">
       ${parts.join("\n")}
     </svg>`;

  // Concise readout below the SVG.
  const pred = trace.outputAct.data[0];
  const loss = trace.lossTensor.data[0];
  slots.readout.innerHTML =
    `<span style="color:#bbb">pred</span> ${fmt(pred)} ·
     <span style="color:#bbb">target</span> ${fmt(params.target)} ·
     <span style="color:${palette.warning}">L</span> ${fmt(loss)}`;
}

// Helper: which nodes are "known" (have a value computed) by this frame.
function nodeKnown(layerIdx, frame, fwdLayer) {
  // Input is known from frame 0 onward.
  if (layerIdx === 0) return true;
  // Hidden (post-tanh) known after "fwd-act layer 0".
  if (layerIdx === 1) {
    if (frame.kind === "init") return false;
    if (frame.kind === "fwd-lin" && frame.layer === 0) return false;   // only pre-act yet
    return true;
  }
  // Output (post-sigmoid) known after "fwd-act layer 1".
  if (layerIdx === 2) {
    return ["fwd-act","loss","bwd-seed","bwd-act","bwd-lin","update"].includes(frame.kind)
      && (frame.kind !== "fwd-act" || frame.layer === 1);
  }
  return false;
}

// Active highlight for a frame: which nodes / edges are the "stars."
function activeHighlight(frame) {
  switch (frame.kind) {
    case "init":
      return { activeNodes: [[0, 0], [0, 1]] };
    case "fwd-lin":
      return {
        activeLayer: frame.layer,
        // The receiving column's nodes are the "stars".
        activeNodes: range(LAYER_SIZES[frame.layer + 1]).map(i => [frame.layer + 1, i]),
        edges: allEdgesAt(frame.layer),
      };
    case "fwd-act":
      return {
        activeNodes: range(LAYER_SIZES[frame.layer + 1]).map(i => [frame.layer + 1, i]),
      };
    case "loss":
      return { activeNodes: [[LAYER_SIZES.length - 1, 0]] };
    case "bwd-seed":
      return { activeNodes: [[LAYER_SIZES.length - 1, 0]] };
    case "bwd-act":
      return { activeNodes: range(LAYER_SIZES[frame.layer + 1]).map(i => [frame.layer + 1, i]) };
    case "bwd-lin":
      return {
        activeLayer: frame.layer,
        activeNodes: range(LAYER_SIZES[frame.layer]).map(i => [frame.layer, i]),
        edges: allEdgesAt(frame.layer),
      };
    case "update":
      return { edges: [...allEdgesAt(0), ...allEdgesAt(1)] };
    default: return {};
  }
}

function allEdgesAt(layerIdx) {
  const out = [];
  for (let j = 0; j < LAYER_SIZES[layerIdx + 1]; j++)
    for (let i = 0; i < LAYER_SIZES[layerIdx]; i++)
      out.push([layerIdx, j, i]);
  return out;
}

function range(n) { return Array.from({ length: n }, (_, i) => i); }

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

// ── Stepper wiring ──

defineStepper({
  hostId: "ch11-nn-explorer-widget",
  controls: {
    x1:     { label: "x₁", min: -1.5, max: 1.5, step: 0.05, default: 0.6 },
    x2:     { label: "x₂", min: -1.5, max: 1.5, step: 0.05, default: -0.4 },
    target: { label: "target", min: 0, max: 1, step: 0.05, default: 1.0 },
    lr:     { label: "lr", min: 0.01, max: 1.0, step: 0.01, default: 0.3 },
    seed:   { label: "seed", min: 1, max: 99, step: 1, default: 13 },
  },
  // The TRAJECTORY itself is the fixed 12-frame schedule. The
  // model + values per frame depend on the params, but the COUNT
  // of frames does not.
  trajectory: () => makeFrames(),
  render: (host, frame, idx, total, params, slots) => {
    // Lazily re-trace on each render — cheap (one forward + backward).
    // We memoise across frames using a parameter signature.
    const sig = `${params.x1}|${params.x2}|${params.target}|${params.lr}|${params.seed}`;
    if (renderFrame._sig !== sig) {
      renderFrame._trace = buildAndTrace(params);
      renderFrame._sig = sig;
    }
    renderFrame(host, frame, idx, total, params, slots, renderFrame._trace);
  },
  playIntervalMs: 1100,
});

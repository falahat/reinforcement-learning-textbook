// JS mirror of the API surface in `crates/ml/nn/`. Same names, same
// semantics, same numerical-stability tricks. The Rust crate is the
// canonical implementation (with full grad-check tests + citations);
// this port exists so widgets can run in-browser today, before the
// wasm-pack pipeline is wired up.
//
// **Mirrors:**
//   Tensor         ← crates/ml/nn/src/tensor/mod.rs
//   Tape           ← crates/ml/nn/src/autograd/{mod, op}.rs
//   Linear         ← crates/ml/nn/src/module/linear.rs
//   Sequential     ← crates/ml/nn/src/module/sequential.rs
//   Relu/Tanh/...  ← crates/ml/nn/src/activation/mod.rs
//   Mse/Huber/CE   ← crates/ml/nn/src/loss/mod.rs
//   Sgd / Adam     ← crates/ml/nn/src/optim/mod.rs
//   Trainer        ← crates/ml/nn/src/train/mod.rs
//
// **Citations** are in the Rust files; this port keeps a short ref
// at each location pointing at the source-of-truth file.

import { splitmix64, gauss } from "./random.js";

// Wrap gauss(rng) → (rng, mu, sigma) for parity with the Rust API.
function sampleNormal(rng, mu, sigma) { return mu + sigma * gauss(rng); }

// ─── Tensor ─────────────────────────────────────────────────────────

/** Pure-value tensor wrapper. Shape is a flat array; data is Float32Array. */
export class Tensor {
  constructor(data, shape) {
    if (!(data instanceof Float32Array)) data = Float32Array.from(data);
    const n = shape.reduce((a, b) => a * b, 1);
    if (data.length !== n) throw new Error(`Tensor: shape ${shape} expects ${n} elements, got ${data.length}`);
    this.data = data;
    this.shape = shape.slice();
    this.node = null;   // tape node id, set when this tensor is tape-bound
  }
  static zeros(shape) {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }
  static fromArray(arr, shape) { return new Tensor(arr, shape); }
  numel() { return this.data.length; }
  item() { return this.data.length === 1 ? this.data[0] : undefined; }
}

// ─── Stable scalar primitives ───────────────────────────────────────
// Mirror crates/ml/nn/src/math/stable.rs — see there for citations
// (log-sum-exp from Boyd & Vandenberghe 2004; branched sigmoid from
// Higham 2002; fused softmax-XE gradient from Goodfellow et al. 2016).

export function sigmoid(x) {
  if (x >= 0) { const e = Math.exp(-x); return 1 / (1 + e); }
  const e = Math.exp(x); return e / (1 + e);
}

export function logSumExp(xs) {
  if (xs.length === 0) return -Infinity;
  let m = -Infinity;
  for (const x of xs) if (x > m) m = x;
  if (!isFinite(m)) return m;
  let s = 0;
  for (const x of xs) s += Math.exp(x - m);
  return m + Math.log(s);
}

// ─── Tape + Op (mirror autograd/mod.rs + op.rs) ─────────────────────
// One Op per recorded node. Backward iterates in reverse, dispatching
// on the op type. Pattern-match on `op.kind` here mirrors the Rust
// `enum Op` pattern-match.

export class Tape {
  constructor() {
    this.nodes = [];                  // each: { op, value, shape, paramId }
    this._registered = [];            // tensors whose .node we set; cleared on .clear()
  }
  clear() {
    // Inputs reused across training steps (x, y, …) cache their tape
    // index in `.node`. Clear that here so the next forward registers
    // them fresh — otherwise the second step references nodes from
    // the cleared tape, which produces NaN gradients in seconds.
    for (const t of this._registered) t.node = null;
    this._registered.length = 0;
    this.nodes.length = 0;
  }

  _ensureNode(t) {
    if (t.node !== null) return t.node;
    const id = this.nodes.length;
    this.nodes.push({
      op: { kind: "leaf" }, value: t.data.slice(), shape: t.shape.slice(), paramId: null,
    });
    t.node = id;
    this._registered.push(t);
    return id;
  }

  /** Register a leaf tensor (input or learnable param). */
  leaf(t, paramId = null) {
    const id = this.nodes.length;
    this.nodes.push({
      op: { kind: "leaf" }, value: t.data.slice(), shape: t.shape.slice(), paramId,
    });
    const out = new Tensor(t.data.slice(), t.shape);
    out.node = id;
    return out;
  }

  _record(op, value, shape) {
    const id = this.nodes.length;
    this.nodes.push({ op, value, shape, paramId: null });
    const out = new Tensor(value.slice(), shape);
    out.node = id;
    return out;
  }

  // ── elementwise binary ────────────────────────────────────────────
  add(a, b) {
    const lhs = this._ensureNode(a), rhs = this._ensureNode(b);
    const v = new Float32Array(a.data.length);
    for (let i = 0; i < v.length; i++) v[i] = a.data[i] + b.data[i];
    return this._record({ kind: "add", lhs, rhs }, v, a.shape);
  }
  sub(a, b) {
    const lhs = this._ensureNode(a), rhs = this._ensureNode(b);
    const v = new Float32Array(a.data.length);
    for (let i = 0; i < v.length; i++) v[i] = a.data[i] - b.data[i];
    return this._record({ kind: "sub", lhs, rhs }, v, a.shape);
  }
  mul(a, b) {
    const lhs = this._ensureNode(a), rhs = this._ensureNode(b);
    const v = new Float32Array(a.data.length);
    for (let i = 0; i < v.length; i++) v[i] = a.data[i] * b.data[i];
    return this._record({ kind: "mul", lhs, rhs }, v, a.shape);
  }

  // ── unary ─────────────────────────────────────────────────────────
  scale(x, factor) {
    const input = this._ensureNode(x);
    const v = new Float32Array(x.data.length);
    for (let i = 0; i < v.length; i++) v[i] = x.data[i] * factor;
    return this._record({ kind: "scale", input, factor }, v, x.shape);
  }
  relu(x) {
    const input = this._ensureNode(x);
    const v = new Float32Array(x.data.length);
    for (let i = 0; i < v.length; i++) v[i] = Math.max(0, x.data[i]);
    return this._record({ kind: "relu", input }, v, x.shape);
  }
  sigmoid(x) {
    const input = this._ensureNode(x);
    const v = new Float32Array(x.data.length);
    for (let i = 0; i < v.length; i++) v[i] = sigmoid(x.data[i]);
    return this._record({ kind: "sigmoid", input }, v, x.shape);
  }
  tanh(x) {
    const input = this._ensureNode(x);
    const v = new Float32Array(x.data.length);
    for (let i = 0; i < v.length; i++) v[i] = Math.tanh(x.data[i]);
    return this._record({ kind: "tanh", input }, v, x.shape);
  }
  square(x) {
    const input = this._ensureNode(x);
    const v = new Float32Array(x.data.length);
    for (let i = 0; i < v.length; i++) v[i] = x.data[i] * x.data[i];
    return this._record({ kind: "square", input }, v, x.shape);
  }

  // ── matmul (2D × 2D) ──────────────────────────────────────────────
  matmul(a, b) {
    const lhs = this._ensureNode(a), rhs = this._ensureNode(b);
    const [m, k] = a.shape, [, n] = b.shape;
    const v = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let kk = 0; kk < k; kk++) {
        const aik = a.data[i * k + kk];
        for (let j = 0; j < n; j++) v[i * n + j] += aik * b.data[kk * n + j];
      }
    }
    return this._record({ kind: "matmul", lhs, rhs }, v, [m, n]);
  }
  transpose(x) {
    const input = this._ensureNode(x);
    const [rows, cols] = x.shape;
    const v = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        v[c * rows + r] = x.data[r * cols + c];
    return this._record({ kind: "transpose", input }, v, [cols, rows]);
  }

  // ── reductions ────────────────────────────────────────────────────
  meanAll(x) {
    const input = this._ensureNode(x);
    let s = 0; for (const v of x.data) s += v;
    return this._record({ kind: "meanAll", input }, new Float32Array([s / x.data.length]), []);
  }
  sumAll(x) {
    const input = this._ensureNode(x);
    let s = 0; for (const v of x.data) s += v;
    return this._record({ kind: "sumAll", input }, new Float32Array([s]), []);
  }

  broadcastRow(x, target) {
    const input = this._ensureNode(x);
    const [rows, cols] = target;
    const v = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) v.set(x.data, r * cols);
    return this._record({ kind: "broadcastRow", input, to: target.slice() }, v, target);
  }

  /** Record a custom op with a caller-provided backward closure.
   * `inputs` is a list of input Tensors; `backward(dy, grads)` writes
   * into `grads[inputs[i].node]` directly. Used by loss functions that
   * fuse their forward + backward for numerical stability. */
  customOp(inputs, value, shape, backward) {
    const inputIds = inputs.map(t => this._ensureNode(t));
    const id = this.nodes.length;
    this.nodes.push({
      op: { kind: "custom", inputIds, backward }, value, shape, paramId: null,
    });
    const out = new Tensor(value.slice(), shape);
    out.node = id;
    return out;
  }

  // ── backward ──────────────────────────────────────────────────────
  backward(loss) {
    return this.backwardFull(loss).params;
  }

  /** Variant of `backward` that also returns the per-node gradient
   * vectors keyed by tape NODE id. Used by visualization widgets
   * (e.g. `nn_explorer`) that want to display ∂L/∂a on every
   * intermediate activation, not just on learnable params. */
  backwardFull(loss) {
    if (loss.node === null) throw new Error("backward: loss not on this tape");
    const grads = this.nodes.map(n => new Float32Array(n.value.length));
    grads[loss.node][0] = 1;
    for (let i = this.nodes.length - 1; i >= 0; i--) this._backwardOne(i, grads);

    const params = new Map();
    for (let i = 0; i < this.nodes.length; i++) {
      if (this.nodes[i].paramId !== null) params.set(this.nodes[i].paramId, grads[i]);
    }
    return { params, nodeGrads: grads };
  }

  _backwardOne(i, grads) {
    const dy = grads[i];
    const node = this.nodes[i];
    const op = node.op;
    const accum = (id, src) => { const g = grads[id]; for (let k = 0; k < src.length; k++) g[k] += src[k]; };
    const accumNeg = (id, src) => { const g = grads[id]; for (let k = 0; k < src.length; k++) g[k] -= src[k]; };

    switch (op.kind) {
      case "leaf": return;

      case "add":
        accum(op.lhs, dy); accum(op.rhs, dy); return;

      case "sub":
        accum(op.lhs, dy); accumNeg(op.rhs, dy); return;

      case "mul": {
        const a = this.nodes[op.lhs].value, b = this.nodes[op.rhs].value;
        const ga = grads[op.lhs], gb = grads[op.rhs];
        for (let k = 0; k < dy.length; k++) { ga[k] += dy[k] * b[k]; gb[k] += dy[k] * a[k]; }
        return;
      }

      case "scale": {
        const g = grads[op.input];
        for (let k = 0; k < dy.length; k++) g[k] += dy[k] * op.factor;
        return;
      }

      case "relu": {
        const x = this.nodes[op.input].value;
        const g = grads[op.input];
        for (let k = 0; k < dy.length; k++) g[k] += (x[k] > 0 ? dy[k] : 0);
        return;
      }

      case "sigmoid": {
        const y = node.value;     // sigmoid output is cached on the node
        const g = grads[op.input];
        for (let k = 0; k < dy.length; k++) g[k] += dy[k] * y[k] * (1 - y[k]);
        return;
      }

      case "tanh": {
        const y = node.value;
        const g = grads[op.input];
        for (let k = 0; k < dy.length; k++) g[k] += dy[k] * (1 - y[k] * y[k]);
        return;
      }

      case "square": {
        const x = this.nodes[op.input].value;
        const g = grads[op.input];
        for (let k = 0; k < dy.length; k++) g[k] += dy[k] * 2 * x[k];
        return;
      }

      case "matmul": {
        // da = dy @ b^T ; db = a^T @ dy
        const aShape = this.nodes[op.lhs].shape, bShape = this.nodes[op.rhs].shape;
        const a = this.nodes[op.lhs].value, b = this.nodes[op.rhs].value;
        const [m, k] = aShape, [, n] = bShape;
        const ga = grads[op.lhs], gb = grads[op.rhs];
        for (let i2 = 0; i2 < m; i2++) {
          for (let kk = 0; kk < k; kk++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += dy[i2 * n + j] * b[kk * n + j];
            ga[i2 * k + kk] += s;
          }
        }
        for (let kk = 0; kk < k; kk++) {
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i2 = 0; i2 < m; i2++) s += a[i2 * k + kk] * dy[i2 * n + j];
            gb[kk * n + j] += s;
          }
        }
        return;
      }

      case "transpose": {
        const [rows, cols] = node.shape;   // these are SWAPPED relative to input
        const g = grads[op.input];
        // input shape is [cols, rows]; transposing dy back lands in input layout.
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < cols; c++)
            g[c * rows + r] += dy[r * cols + c];
        return;
      }

      case "sumAll": {
        const g = grads[op.input], s = dy[0];
        for (let k = 0; k < g.length; k++) g[k] += s;
        return;
      }

      case "meanAll": {
        const g = grads[op.input], s = dy[0] / g.length;
        for (let k = 0; k < g.length; k++) g[k] += s;
        return;
      }

      case "broadcastRow": {
        // dx = column-wise sum of dy (each row of out was the input).
        const [rows, cols] = op.to;
        const g = grads[op.input];
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < cols; c++)
            g[c] += dy[r * cols + c];
        return;
      }

      case "custom": {
        op.backward(dy, grads, op.inputIds, this);
        return;
      }

      default:
        throw new Error(`backward: unknown op kind '${op.kind}'`);
    }
  }
}

// ─── Modules ────────────────────────────────────────────────────────

let nextParamId = 1;
const newParamId = () => nextParamId++;

/** Linear layer y = x @ W^T + b. Weight is [out_dim, in_dim]. */
export class Linear {
  constructor(inDim, outDim, init = "he", rng = null) {
    this.inDim = inDim; this.outDim = outDim;
    const r = rng ?? splitmix64(42);
    const w = new Float32Array(outDim * inDim);
    const sigma = init === "he"     ? Math.sqrt(2 / inDim) :
                  init === "xavier" ? Math.sqrt(2 / (inDim + outDim)) :
                                       0.1;
    for (let i = 0; i < w.length; i++) w[i] = sampleNormal(r, 0, sigma);
    this.weight = new Tensor(w, [outDim, inDim]);
    this.bias   = Tensor.zeros([outDim]);
    this.wId = newParamId(); this.bId = newParamId();
  }
  forward(tape, x) {
    const w = tape.leaf(this.weight, this.wId);
    const b = tape.leaf(this.bias,   this.bId);
    const wt = tape.transpose(w);
    const prod = tape.matmul(x, wt);                      // [batch, out]
    const bb = tape.broadcastRow(b, prod.shape);
    return tape.add(prod, bb);
  }
  visitParams(visit) {
    visit("weight", this.weight, this.wId);
    visit("bias",   this.bias,   this.bId);
  }
}

/** No-param module: any of relu/sigmoid/tanh by name. */
export class Activation {
  constructor(kind) { this.kind = kind; }
  forward(tape, x) {
    switch (this.kind) {
      case "relu":    return tape.relu(x);
      case "sigmoid": return tape.sigmoid(x);
      case "tanh":    return tape.tanh(x);
      default: throw new Error(`Activation: unknown kind '${this.kind}'`);
    }
  }
  visitParams() { /* nothing */ }
}
export const Relu    = () => new Activation("relu");
export const Sigmoid = () => new Activation("sigmoid");
export const Tanh    = () => new Activation("tanh");

/** Sequential of layers. Hot-swap a layer with `.replace(idx, newLayer)`. */
export class Sequential {
  constructor(layers) { this.layers = layers; }
  forward(tape, x) {
    let y = this.layers[0].forward(tape, x);
    for (let i = 1; i < this.layers.length; i++) y = this.layers[i].forward(tape, y);
    return y;
  }
  visitParams(visit) {
    this.layers.forEach((l, i) => l.visitParams((path, p, id) => visit(`${i}.${path}`, p, id)));
  }
  replace(idx, m) { const old = this.layers[idx]; this.layers[idx] = m; return old; }
}

// ─── Losses ─────────────────────────────────────────────────────────

export class Mse {
  forward(tape, pred, target) {
    const d = tape.sub(pred, target);
    const s = tape.square(d);
    return tape.meanAll(s);
  }
}

/** Huber loss with smoothing parameter δ; quadratic for |r| ≤ δ,
 * linear beyond. Fused forward + backward to keep autograd simple. */
export class Huber {
  constructor(delta = 1.0) { this.delta = delta; }
  forward(tape, pred, target) {
    const delta = this.delta;
    const n = pred.numel();
    let lossSum = 0;
    const r = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      r[i] = pred.data[i] - target.data[i];
      const ar = Math.abs(r[i]);
      lossSum += ar <= delta ? 0.5 * r[i] * r[i] : delta * (ar - 0.5 * delta);
    }
    const lossVal = lossSum / n;
    return tape.customOp([pred, target], new Float32Array([lossVal]), [],
      (dy, grads, ids) => {
        // d L / d pred = (1/n) * (r if |r|≤δ else δ·sign(r))
        // d L / d target = -(same)
        const s = dy[0] / n;
        const gPred = grads[ids[0]];
        const gTarg = grads[ids[1]];
        for (let i = 0; i < n; i++) {
          const ar = Math.abs(r[i]);
          const grad = (ar <= delta) ? r[i] : delta * Math.sign(r[i]);
          gPred[i] += s * grad;
          gTarg[i] -= s * grad;
        }
      });
  }
}

/** REINFORCE-style policy gradient loss. logits: [T, n_actions];
 * actions: [T] discrete indices; advantages: [T] real-valued.
 *
 * `L = -mean_t( A_t * log_softmax(logits[t])[a_t] )`
 *
 * The fused backward gradient on logits is
 * `(softmax_t - one_hot(a_t)) * A_t / T`. Same shape as the
 * cross-entropy gradient, scaled per-row by the advantage. */
export class PolicyGradientLoss {
  forward(tape, logits, actions, advantages) {
    const [T, K] = logits.shape;
    const softmax = new Float32Array(T * K);
    let lossSum = 0;
    for (let t = 0; t < T; t++) {
      const row = logits.data.subarray(t * K, (t + 1) * K);
      const lse = logSumExp(row);
      const a = actions.data[t];
      const adv = advantages.data[t];
      // log p(a_t | s_t) = row[a] - lse
      lossSum += adv * (lse - row[a]);
      for (let c = 0; c < K; c++) softmax[t * K + c] = Math.exp(row[c] - lse);
    }
    return tape.customOp([logits, actions, advantages],
      new Float32Array([lossSum / T]), [],
      (dy, grads, ids) => {
        const s = dy[0] / T;
        const g = grads[ids[0]];
        for (let t = 0; t < T; t++) {
          const a = actions.data[t];
          const adv = advantages.data[t];
          for (let c = 0; c < K; c++) {
            let v = softmax[t * K + c];
            if (c === a) v -= 1;
            g[t * K + c] += s * adv * v;
          }
        }
      });
  }
}

/** Fused softmax + cross-entropy. logits: [batch, classes];
 * targets: [batch] of class indices. Numerically stable via LSE. */
export class CrossEntropy {
  forward(tape, logits, targets) {
    const [batch, classes] = logits.shape;
    const softmax = new Float32Array(batch * classes);
    let lossSum = 0;
    for (let b = 0; b < batch; b++) {
      const row = logits.data.subarray(b * classes, (b + 1) * classes);
      const lse = logSumExp(row);
      lossSum += lse - row[targets.data[b]];
      for (let c = 0; c < classes; c++) softmax[b * classes + c] = Math.exp(row[c] - lse);
    }
    return tape.customOp([logits, targets], new Float32Array([lossSum / batch]), [],
      (dy, grads, ids) => {
        // d L / d logits = (softmax - one_hot(target)) / batch.
        const s = dy[0] / batch;
        const g = grads[ids[0]];
        for (let b = 0; b < batch; b++) {
          const t = targets.data[b];
          for (let c = 0; c < classes; c++) {
            let v = softmax[b * classes + c];
            if (c === t) v -= 1;
            g[b * classes + c] += s * v;
          }
        }
      });
  }
}

// ─── Optimizers ─────────────────────────────────────────────────────

export class Sgd {
  constructor(lr, momentum = 0) { this.lr = lr; this.momentum = momentum; this.state = new Map(); }
  step(model, grads) {
    const { lr, momentum, state } = this;
    model.visitParams((_path, p, id) => {
      const g = grads.get(id); if (!g) return;
      if (momentum === 0) {
        for (let k = 0; k < p.data.length; k++) p.data[k] -= lr * g[k];
      } else {
        let v = state.get(id);
        if (!v) { v = new Float32Array(p.data.length); state.set(id, v); }
        for (let k = 0; k < p.data.length; k++) {
          v[k] = momentum * v[k] + g[k];
          p.data[k] -= lr * v[k];
        }
      }
    });
  }
}

export class Adam {
  constructor(lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    this.state = new Map(); this.stepCount = 0;
  }
  step(model, grads) {
    this.stepCount += 1;
    const { lr, beta1: b1, beta2: b2, eps, state } = this;
    const t = this.stepCount;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    model.visitParams((_path, p, id) => {
      const g = grads.get(id); if (!g) return;
      let s = state.get(id);
      if (!s) { s = { m: new Float32Array(p.data.length), v: new Float32Array(p.data.length) }; state.set(id, s); }
      for (let k = 0; k < p.data.length; k++) {
        s.m[k] = b1 * s.m[k] + (1 - b1) * g[k];
        s.v[k] = b2 * s.v[k] + (1 - b2) * g[k] * g[k];
        const mHat = s.m[k] / bc1;
        const vHat = s.v[k] / bc2;
        p.data[k] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    });
  }
}

// ─── Trainer ────────────────────────────────────────────────────────

export class Trainer {
  constructor({ model, loss, optim }) {
    this.model = model; this.loss = loss; this.optim = optim;
    this.tape = new Tape();
    this.stepCount = 0;
  }
  trainStep(x, y) {
    this.stepCount += 1;
    this.tape.clear();
    const pred = this.model.forward(this.tape, x);
    const lossT = this.loss.forward(this.tape, pred, y);
    const lv = lossT.item();
    const grads = this.tape.backward(lossT);
    this.optim.step(this.model, grads);
    return lv;
  }
  predict(x) {
    const t = new Tape();
    return this.model.forward(t, x);
  }
}

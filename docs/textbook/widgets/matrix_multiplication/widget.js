// Widget 1.X — Matrix multiplication step-by-step (Chapter 1, §1.2).
//
// Walks through C = A × B one term at a time. For each output cell
// C[i, j] the widget steps through the dot product
//
//     C[i, j] = Σ_k A[i, k] · B[k, j]
//
// adding one term per frame, and showing a completion frame when the
// cell is finished. The row of A used at step k is tinted with
// `palette.primary`, the column of B with `palette.secondary`, and
// the active output cell in C with `palette.warning`.
//
// Trajectory layout (per (i, j) cell, then per k term):
//     frame:  { i, j, k, partial }                  // k ∈ 0..n
//     frame:  { i, j, k: n, partial: C[i,j], done } // cell complete
// Total frames = m · p · (n + 1).
//
// Mount: in §1.2 of Chapter 1.
//
//     <div id="ch1-matmul-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/matrix_multiplication/widget.js"></script>

import { defineStepper } from "../shared/stepper.js";
import { palette } from "../shared/helpers.js";
import { lcg } from "../shared/random.js";

// --- deterministic random fill ---------------------------------------------
//
// Previously used a tiny LCG with the constants (9301 * s + 49297) % 233280;
// this widget is purely visual (a demo matrix fill), so swapping to the
// shared `lcg` (1664525 * s + 1013904223, mod 2^32) is safe — the displayed
// numbers will differ but the visualisation is unchanged in character.

// Small integers 1..15 keep the arithmetic mental: a reader can
// follow `A[i,k]·B[k,j]` and the running sum in their head, which is
// the whole point of the step-by-step animation. Floats in [0.1, 0.9]
// looked smoother but made every product an arithmetic exercise on
// its own; the visualisation should be teaching the *structure* of
// matrix multiplication, not stress-testing decimal multiplication.
function genMatrix(rows, cols, seed) {
  const rng = lcg(seed);
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => 1 + Math.floor(rng() * 15)),
  );
}

// --- rendering helpers -----------------------------------------------------

const MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

function fmtNum(x) {
  // Cells hold small integers 1..15, but partial sums and products
  // can grow into the hundreds. Render integers as integers (no
  // decimal point); only fall back to a 2-decimal form for the rare
  // non-integer that slips in (e.g. a user-edited cell, future float
  // mode).
  if (Number.isInteger(x)) return String(x);
  return x.toFixed(2);
}

// Build one matrix as a <table>. `name` is shown above the matrix
// ("A", "B", or "C"). `rowHighlight` / `colHighlight` are integer
// indices (or -1). `cellOverride` is an optional function
// `(r, c) => { text?: string, fill?: string }` used for the C matrix
// to show "?" for not-yet-computed cells and the partial sum for the
// active cell.
function renderMatrix(name, dims, values, opts = {}) {
  const {
    rowHighlight = -1,
    colHighlight = -1,
    cellOverride = null,
    rowsLabel,
    colsLabel,
  } = opts;
  const [rows, cols] = dims;

  const tbl = document.createElement("table");
  tbl.style.borderCollapse = "collapse";
  tbl.style.fontFamily = MONO;
  tbl.style.fontSize = "12px";
  tbl.style.margin = "0 auto";

  for (let r = 0; r < rows; r++) {
    const tr = document.createElement("tr");
    for (let c = 0; c < cols; c++) {
      const td = document.createElement("td");
      const ov = cellOverride ? cellOverride(r, c) : null;
      const text = ov && ov.text != null ? ov.text : fmtNum(values[r][c]);
      td.textContent = text;
      td.style.border = "1px solid #555";
      td.style.padding = "4px 8px";
      td.style.minWidth = "2.6em";
      td.style.textAlign = "center";

      // Layered highlights: row first, then column overrides, then
      // explicit cell override (so an active cell stays visible even
      // when its row/column would otherwise tint it).
      let bg = null;
      if (rowHighlight === r) bg = palette.primary;
      if (colHighlight === c) bg = palette.secondary;
      if (rowHighlight === r && colHighlight === c) {
        // Cell at the row/column intersection — blend by leaning on
        // the column colour to match the "term k" highlight in the
        // readout.
        bg = palette.secondary;
      }
      if (ov && ov.fill) bg = ov.fill;
      if (bg) {
        td.style.background = bg;
        td.style.color = "white";
        td.style.opacity = "0.95";
      }
      tr.appendChild(td);
    }
    tbl.appendChild(tr);
  }

  const wrap = document.createElement("div");
  wrap.style.display = "inline-block";
  wrap.style.margin = "0 14px";
  wrap.style.verticalAlign = "middle";
  wrap.style.textAlign = "center";

  const title = document.createElement("div");
  title.textContent = `${name}  (${rows}×${cols})`;
  title.style.fontFamily = MONO;
  title.style.fontSize = "11px";
  title.style.marginBottom = "4px";
  title.style.opacity = "0.8";
  wrap.appendChild(title);
  wrap.appendChild(tbl);

  if (rowsLabel || colsLabel) {
    const sub = document.createElement("div");
    sub.style.fontSize = "10px";
    sub.style.opacity = "0.6";
    sub.style.marginTop = "4px";
    sub.style.fontFamily = MONO;
    sub.textContent = [rowsLabel, colsLabel].filter(Boolean).join("  ");
    wrap.appendChild(sub);
  }
  return wrap;
}

// Build a formula line like
//   C[1,0] = A[1,0]·B[0,0] + A[1,1]·B[1,0] + A[1,2]·B[2,0]
//          = 0.40·0.70 + 0.10·0.20 + 0.90·0.50
//          = 0.28 + 0.02 + 0.45 = 0.75
// with the current k term in bold. `activeK` is the term being shown,
// or `n` when the cell is complete.
function renderFormula(A, B, i, j, n, activeK, done) {
  const sym = "·";
  // Line 1: symbolic
  const symParts = [];
  for (let k = 0; k < n; k++) {
    const txt = `A[${i},${k}]${sym}B[${k},${j}]`;
    symParts.push(
      k === activeK && !done
        ? `<strong style="color:${palette.warning}">${txt}</strong>`
        : txt,
    );
  }
  const line1 = `C[${i},${j}] = ${symParts.join(" + ")}`;

  // Line 2: numeric products
  const numParts = [];
  for (let k = 0; k < n; k++) {
    const txt = `${fmtNum(A[i][k])}${sym}${fmtNum(B[k][j])}`;
    numParts.push(
      k === activeK && !done
        ? `<strong style="color:${palette.warning}">${txt}</strong>`
        : txt,
    );
  }
  const line2 = `       = ${numParts.join(" + ")}`;

  // Line 3: per-term products plus running total. Only include terms
  // already accumulated (k' ≤ activeK in the in-progress case; all of
  // them once done).
  const prodParts = [];
  let runningTotal = 0;
  const upto = done ? n : Math.min(activeK + 1, n);
  for (let k = 0; k < upto; k++) {
    const v = A[i][k] * B[k][j];
    runningTotal += v;
    const txt = fmtNum(v);
    prodParts.push(
      k === activeK && !done
        ? `<strong style="color:${palette.warning}">${txt}</strong>`
        : txt,
    );
  }
  const tail = done
    ? ` = <strong>${fmtNum(runningTotal)}</strong>`
    : prodParts.length > 0
      ? `  (partial: <strong>${fmtNum(runningTotal)}</strong>)`
      : "";
  const line3 = prodParts.length > 0 ? `       = ${prodParts.join(" + ")}${tail}` : "";

  // Compose with <pre> to preserve the alignment of the leading "=".
  return `<pre style="font-family:${MONO};font-size:12px;line-height:1.5;margin:0;white-space:pre-wrap;">${line1}\n${line2}${line3 ? "\n" + line3 : ""}</pre>`;
}

// --- the widget ------------------------------------------------------------

defineStepper({
  hostId: "ch1-matmul-widget",
  controls: {
    m: { label: "m (rows of A)", min: 1, max: 4, step: 1, default: 2 },
    n: { label: "n (cols A / rows B)", min: 1, max: 4, step: 1, default: 3 },
    p: { label: "p (cols of B)", min: 1, max: 4, step: 1, default: 2 },
    seed: { label: "seed", min: 1, max: 100, step: 1, default: 1 },
  },
  slots: ["main"],
  playIntervalMs: 700,
  trajectory: (params) => {
    const m = params.m | 0;
    const n = params.n | 0;
    const p = params.p | 0;
    const seed = params.seed | 0;

    const A = genMatrix(m, n, seed);
    // Offset B's seed so A and B are not identical when m=n=p.
    const B = genMatrix(n, p, seed + 17);
    // Precompute C for the "done" frames.
    const C = Array.from({ length: m }, () => new Array(p).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += A[i][k] * B[k][j];
        C[i][j] = s;
      }
    }

    const frames = [];
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let partial = 0;
        for (let k = 0; k < n; k++) {
          frames.push({
            i,
            j,
            k,
            partial, // sum over k' < k
            done: false,
            A,
            B,
            C,
            m,
            n,
            p,
          });
          partial += A[i][k] * B[k][j];
        }
        // Completion frame: dot product done, k advances past the
        // last term, partial holds the final cell value.
        frames.push({
          i,
          j,
          k: n,
          partial: C[i][j],
          done: true,
          A,
          B,
          C,
          m,
          n,
          p,
        });
      }
    }
    return frames;
  },
  render: (host, frame, idx, total, _params, slots) => {
    const { i, j, k, partial, done, A, B, C, m, n, p } = frame;

    // Build the "C so far" view. Use the natural traversal order
    // (row-major over (i, j)) to decide which cells are already
    // complete. A cell at (r, c) is complete iff it precedes (i, j)
    // in this traversal, or (it equals (i, j) AND the current frame
    // is the done frame for that cell).
    const completedIndex = i * p + j; // current cell linear index
    const cOverride = (r, c) => {
      const lin = r * p + c;
      if (lin < completedIndex) {
        return { text: fmtNum(C[r][c]) };
      }
      if (lin === completedIndex) {
        if (done) {
          return { text: fmtNum(C[r][c]), fill: palette.warning };
        }
        // Active cell, mid-computation. Show partial in italic
        // braces so it doesn't look identical to a final value.
        return {
          text: k === 0 ? "·" : fmtNum(partial),
          fill: palette.warning,
        };
      }
      return { text: "?" };
    };

    // Render the three matrices into a wrapper flexbox.
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.flexWrap = "wrap";
    wrapper.style.justifyContent = "center";
    wrapper.style.alignItems = "flex-start";
    wrapper.style.gap = "10px";
    wrapper.style.marginBottom = "10px";

    // For Matrix A we highlight row i. For B we highlight column j.
    // We also tint the *currently active k* element in A's row and
    // B's column with palette.warning when it's a term frame (not
    // the done frame).
    const aOverride = (r, c) => {
      if (!done && r === i && c === k) return { fill: palette.warning };
      return null;
    };
    const bOverride = (r, c) => {
      if (!done && c === j && r === k) return { fill: palette.warning };
      return null;
    };

    wrapper.appendChild(
      renderMatrix("A", [m, n], A, {
        rowHighlight: i,
        cellOverride: aOverride,
      }),
    );

    // A small "×" between matrices.
    const times = document.createElement("div");
    times.textContent = "×";
    times.style.fontSize = "18px";
    times.style.fontFamily = MONO;
    times.style.alignSelf = "center";
    wrapper.appendChild(times);

    wrapper.appendChild(
      renderMatrix("B", [n, p], B, {
        colHighlight: j,
        cellOverride: bOverride,
      }),
    );

    const eq = document.createElement("div");
    eq.textContent = "=";
    eq.style.fontSize = "18px";
    eq.style.fontFamily = MONO;
    eq.style.alignSelf = "center";
    wrapper.appendChild(eq);

    wrapper.appendChild(
      renderMatrix("C", [m, p], C, {
        cellOverride: cOverride,
      }),
    );

    // Formula block.
    const formula = document.createElement("div");
    formula.style.marginTop = "6px";
    formula.style.padding = "8px 10px";
    formula.style.border = "1px solid #444";
    formula.style.borderRadius = "4px";
    formula.style.maxWidth = "min(680px, 100%)";
    formula.style.marginLeft = "auto";
    formula.style.marginRight = "auto";
    formula.innerHTML = renderFormula(A, B, i, j, n, k, done);

    const container = document.createElement("div");
    container.appendChild(wrapper);
    container.appendChild(formula);
    slots.main.replaceChildren(container);

    // Readout line.
    const totalCells = m * p;
    const cellIdx = i * p + j + 1; // 1-based
    const phase = done ? `cell C[${i},${j}] complete` : `term k=${k}`;
    slots.readout.textContent =
      `step ${idx + 1} / ${total}  ·  cell ${cellIdx} / ${totalCells}  ·  ${phase}`;
  },
});

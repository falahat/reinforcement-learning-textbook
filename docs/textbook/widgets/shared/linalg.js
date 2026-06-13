// 2×2 linear-algebra utilities for textbook widgets.
//
// Many widgets in this textbook implement the same 2×2 eigen / SVD /
// matrix-vector arithmetic — `eigenvector`, `grid_transform`,
// `power_iteration`, `matrix_norms`, and a few that build small
// Markov chains by hand. This module gives them a shared, tested
// implementation so we stop duplicating the same closed-form math.
//
// Why 2×2 only? Because most pedagogical widgets work on toy
// examples: a 2-state Markov chain, a 2×2 covariance, the 2D plane.
// Larger matrices in this textbook either come from real simulator
// data (handled elsewhere) or use Plot's own n-dimensional helpers.
//
// All matrices are passed as `[[a, b], [c, d]]` (row-major arrays of
// rows). Vectors are passed as `[x, y]`. Returns follow the same
// convention.

/** Determinant of a 2×2 matrix. */
export function det2x2([[a, b], [c, d]]) {
  return a * d - b * c;
}

/** Trace of a 2×2 matrix. */
export function trace2x2([[a, _b], [_c, d]]) {
  return a + d;
}

/** Matrix-vector product Av for a 2×2 A and a 2-vector v. */
export function matvec2x2([[a, b], [c, d]], [x, y]) {
  return [a * x + b * y, c * x + d * y];
}

/** Inverse of a 2×2 matrix. Returns `null` if det ≈ 0 (singular). */
export function inv2x2([[a, b], [c, d]]) {
  const D = a * d - b * c;
  if (Math.abs(D) < 1e-12) return null;
  return [[d / D, -b / D], [-c / D, a / D]];
}

/** Matrix-matrix product (A · B) for 2×2 matrices. */
export function mul2x2(A, B) {
  const [[a, b], [c, d]] = A;
  const [[e, f], [g, h]] = B;
  return [
    [a * e + b * g, a * f + b * h],
    [c * e + d * g, c * f + d * h],
  ];
}

/**
 * Eigenvalues + (unit) eigenvectors of a 2×2 matrix.
 *
 * Returns
 *   { real: boolean,
 *     eigs: [{ lambda: number, v: [x, y] }, ...] }
 *
 * - `real: true` — two real eigenvalues. `eigs` is non-empty. If
 *   eigenvalues coincide (defective matrix), only one eigenvector
 *   direction is returned.
 * - `real: false` — two complex-conjugate eigenvalues. `eigs` is
 *   empty (no real eigenvectors exist). The complex pair is
 *   `tr/2 ± i sqrt(4 det - tr²)/2` — recover from `trace2x2` and
 *   `det2x2` if you need it.
 *
 * The unit eigenvectors are returned as `[x, y]` arrays with
 * `||v|| = 1`. For defective matrices the single direction is
 * returned in both slots.
 *
 * @param {number[][]} A — `[[a, b], [c, d]]`.
 * @returns {{ real: boolean, eigs: Array<{ lambda: number, v: [number, number] }> }}
 */
export function eig2x2(A) {
  const [[a, b], [c, d]] = A;
  const tr = a + d;
  const det = a * d - b * c;
  const disc = tr * tr - 4 * det;
  if (disc < -1e-9) return { real: false, eigs: [] };
  const sq = Math.sqrt(Math.max(0, disc));
  const lambdas = [(tr + sq) / 2, (tr - sq) / 2];

  // Eigenvector for given λ: null vector of (A - λI). Pick the larger
  // row as the basis for the null direction to avoid degenerate row.
  const ev = (lam) => {
    const r1 = [a - lam, b];
    const r2 = [c, d - lam];
    let v;
    if (Math.hypot(r1[0], r1[1]) > Math.hypot(r2[0], r2[1])) {
      v = [-r1[1], r1[0]];
    } else {
      v = [-r2[1], r2[0]];
    }
    const n = Math.hypot(v[0], v[1]);
    if (n < 1e-9) return [1, 0];
    return [v[0] / n, v[1] / n];
  };

  return {
    real: true,
    eigs: [
      { lambda: lambdas[0], v: ev(lambdas[0]) },
      { lambda: lambdas[1], v: ev(lambdas[1]) },
    ],
  };
}

/**
 * Closed-form 2×2 singular-value decomposition.
 *
 * Returns `{ sigma: [s1, s2], v: [v1, v2] }` where `s1 ≥ s2 ≥ 0`
 * are the singular values and `v1, v2` are unit right-singular
 * vectors. Degenerate cases (s1 = s2, s2 = 0) are handled by
 * returning axis-aligned defaults.
 *
 * Use this for widgets that want to draw the ellipse `A · unit-circle`
 * with semi-axes labelled by σ_i.
 *
 * @param {number[][]} A — `[[a, b], [c, d]]`.
 * @returns {{ sigma: [number, number], v: [[number, number], [number, number]] }}
 */
export function svd2x2(A) {
  const [[a, b], [c, d]] = A;
  // SVD of A from eigendecomposition of A^T A.
  const m11 = a * a + c * c;
  const m22 = b * b + d * d;
  const m12 = a * b + c * d;
  const tr = m11 + m22;
  const det = m11 * m22 - m12 * m12;
  const disc = Math.max(0, tr * tr - 4 * det);
  const s1sq = (tr + Math.sqrt(disc)) / 2;
  const s2sq = Math.max(0, (tr - Math.sqrt(disc)) / 2);
  const s1 = Math.sqrt(s1sq);
  const s2 = Math.sqrt(s2sq);

  const eig = (lam) => {
    const r1 = [m11 - lam, m12];
    const r2 = [m12, m22 - lam];
    let v;
    if (Math.hypot(r1[0], r1[1]) > Math.hypot(r2[0], r2[1])) {
      v = [-r1[1], r1[0]];
    } else {
      v = [-r2[1], r2[0]];
    }
    const n = Math.hypot(v[0], v[1]);
    if (n < 1e-9) return [1, 0];
    return [v[0] / n, v[1] / n];
  };
  const v1 = eig(s1sq);
  const v2 = [-v1[1], v1[0]];
  return { sigma: [s1, s2], v: [v1, v2] };
}

/**
 * Power iteration. Starting from `v0`, repeatedly applies
 * `v ← A · v / ||A · v||` and returns the sequence of iterates.
 * Useful for widgets that want to show convergence to the dominant
 * eigenvector.
 *
 * @param {number[][]} A — 2×2 matrix.
 * @param {[number, number]} v0 — starting unit vector (normalised on entry).
 * @param {number} steps — number of iterations.
 * @returns {Array<[number, number]>} — `steps + 1` unit vectors starting from v0.
 */
export function powerIterate(A, v0, steps) {
  const n0 = Math.hypot(v0[0], v0[1]) || 1;
  let v = [v0[0] / n0, v0[1] / n0];
  const out = [[...v]];
  for (let k = 0; k < steps; k++) {
    const w = matvec2x2(A, v);
    const n = Math.hypot(w[0], w[1]);
    if (n < 1e-12) break;
    v = [w[0] / n, w[1] / n];
    out.push([...v]);
  }
  return out;
}

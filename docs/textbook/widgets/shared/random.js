// Seeded pseudo-random generators for textbook widgets.
//
// Widgets that simulate trajectories, sample noise, or generate a
// canonical "interesting" data set MUST seed their RNG so that
// repeated renders (e.g. on every slider tick) produce identical
// noise. Otherwise the chart jitters for reasons unrelated to the
// user's input — see bug class 13 in
// `.claude/skills/textbook-with-widgets/references/10-widget-correctness-audit.md`.
//
// Two PRNGs ship here:
//
//   - `mulberry32(seed)` — the canonical small good-enough PRNG.
//     Fast, deterministic, decent distribution. Use this by default.
//     32-bit state; ~2^32 period.
//   - `lcg(seed)` — a tiny linear-congruential generator. Cheaper but
//     visibly periodic on long runs. Useful only when you want
//     deliberately-coarse noise (e.g. matrix-fill demos).
//
// Both return a *function* `() => number` that yields a fresh uniform
// sample in [0, 1) on each call.
//
// Plus two convenience wrappers for common distributions:
//
//   - `gauss(rng)` — one standard-normal sample via Box-Muller.
//     Caches the second sample so consecutive calls amortize one
//     `Math.log` + one `Math.sqrt`.
//   - `sampleDiscrete(rng, probs)` — sample an index 0..probs.length-1
//     with given probabilities. Probabilities need not sum to 1; the
//     function normalises.
//
// Example use inside a `defineWidget` render handler:
//
//     import { mulberry32, gauss } from "../shared/random.js";
//     render: (host, { seed, noise }, slots) => {
//       const rng = mulberry32(seed);
//       const data = d3.range(N).map(() => ({ y: noise * gauss(rng) }));
//       // ...
//     }

/**
 * Mulberry32 — the standard small-state PRNG used everywhere in this
 * textbook. Returns a function yielding uniform [0, 1) samples.
 *
 * @param {number} seed — any 32-bit unsigned integer.
 * @returns {() => number}
 */
export function mulberry32(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Linear-congruential generator. Smaller and faster than mulberry32
 * but visibly periodic; prefer mulberry32 unless you specifically
 * want this.
 *
 * @param {number} seed
 * @returns {() => number}
 */
export function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Splitmix64 — Sebastiano Vigna's standard 64-bit splittable PRNG.
 * Returns a uniform [0, 1) sampler via the low 32 bits of the 64-bit
 * output. Uses BigInt arithmetic, so it's ~3× slower than mulberry32
 * — use it only when you specifically want 64-bit splittable state
 * (e.g. to match the project's main RNG convention).
 *
 * @param {number | bigint} seed
 * @returns {() => number}
 */
export function splitmix64(seed) {
  let state = BigInt(seed) & 0xffffffffffffffffn;
  return () => {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z = z ^ (z >> 31n);
    return Number(z & 0xffffffffn) / 2 ** 32;
  };
}

let _gaussSpare = null;
/**
 * One standard-normal sample using Box-Muller. The second sample
 * from each Box-Muller pair is cached in a module-level slot so the
 * amortised cost is one `Math.log` + one `Math.sqrt` per two calls.
 *
 * The cache is shared across all callers (single global slot).
 * Two widgets calling `gauss(rng1)` and `gauss(rng2)` will interleave
 * the cache; this is fine because each pair is itself i.i.d. — but
 * be aware that the second call won't necessarily use rng2.
 *
 * @param {() => number} rng — a uniform sampler (from mulberry32, lcg).
 * @returns {number}
 */
export function gauss(rng) {
  if (_gaussSpare !== null) {
    const r = _gaussSpare;
    _gaussSpare = null;
    return r;
  }
  let u, v, s;
  do {
    u = 2 * rng() - 1;
    v = 2 * rng() - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  const mul = Math.sqrt((-2 * Math.log(s)) / s);
  _gaussSpare = v * mul;
  return u * mul;
}

/**
 * Sample one index from a discrete distribution. Probabilities need
 * not sum to 1.
 *
 * @param {() => number} rng
 * @param {number[]} probs
 * @returns {number} an index in [0, probs.length).
 */
export function sampleDiscrete(rng, probs) {
  const total = probs.reduce((s, p) => s + Math.max(0, p), 0);
  if (total <= 0) return 0;
  const r = rng() * total;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += Math.max(0, probs[i]);
    if (r < acc) return i;
  }
  return probs.length - 1;
}

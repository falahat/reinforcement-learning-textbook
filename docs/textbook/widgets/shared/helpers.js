// Shared helpers for TypeScript / JavaScript math-viz widgets.
//
// Each helper is small and zero-config — widgets import only what they
// need. Designed to keep per-widget code minimal: 80% of widgets need
// `readNumber` + `autoRender`, ~30 lines of widget logic, and a Plot
// configuration.
//
// To use, mount a widget like:
//
//     import * as Plot from "@observablehq/plot";
//     import { readNumber, autoRender } from "./widgets/shared/helpers.js";
//
//     const host = document.getElementById("ch17-gamma-widget");
//     autoRender(host, () => {
//       const gamma = readNumber(host, '[data-input="gamma"]');
//       host.querySelector("[data-plot]").replaceChildren(
//         Plot.plot({ /* ... */ })
//       );
//     });

/**
 * Read a numeric value from an `<input>` matching `selector` inside
 * `host`. Returns NaN if the selector misses or the value doesn't
 * parse — the caller can guard.
 *
 * @param {Element} host
 * @param {string} selector
 * @returns {number}
 */
export function readNumber(host, selector) {
  const el = host.querySelector(selector);
  if (!el) return NaN;
  return parseFloat(el.value);
}

/**
 * Read a string value (e.g. from a `<select>`).
 *
 * @param {Element} host
 * @param {string} selector
 * @returns {string}
 */
export function readString(host, selector) {
  const el = host.querySelector(selector);
  return el ? el.value : "";
}

/**
 * Wire every `<input>` and `<select>` inside `host` to re-call
 * `render` on every change ("input" event = continuous slider update,
 * "change" event = discrete commits). Also calls `render` once
 * immediately so the widget paints on load.
 *
 * @param {Element} host
 * @param {() => void} render
 */
export function autoRender(host, render) {
  for (const el of host.querySelectorAll("input, select")) {
    el.addEventListener("input", render);
    el.addEventListener("change", render);
  }
  render();
}

/**
 * Set the text content of a `[data-readout]` element (or any matching
 * selector) inside `host`. Common widget convention.
 *
 * @param {Element} host
 * @param {string} text
 * @param {string} [selector]
 */
export function setReadout(host, text, selector = "[data-readout]") {
  const el = host.querySelector(selector);
  if (el) el.textContent = text;
}

/**
 * Format a float for display — 3-significant-digit fixed for small
 * magnitudes, scientific notation for very small / very large. Reads
 * cleaner than `n.toFixed(...)` or `n.toExponential(...)` alone.
 *
 * @param {number} n
 * @returns {string}
 */
export function fmt(n) {
  if (!isFinite(n)) return String(n);
  const abs = Math.abs(n);
  if (abs === 0) return "0";
  if (abs < 1e-3 || abs >= 1e6) return n.toExponential(3);
  if (abs < 1) return n.toFixed(4);
  if (abs < 100) return n.toFixed(3);
  return n.toFixed(1);
}

/**
 * Standard Plot styling for the textbook — small fonts, dark-mode
 * friendly colours. Spread into your Plot config:
 *
 *     Plot.plot({ ...plotDefaults, marks: [...] })
 */
export const plotDefaults = {
  width: 640,
  height: 280,
  marginLeft: 50,
  marginBottom: 36,
  style: {
    background: "transparent",
    fontSize: "11px",
  },
};

/**
 * Shared colour palette across textbook widgets. Picks are dark-mode
 * friendly and have enough hue separation for colour-blind safety on
 * a 3-series chart. Use these instead of hard-coding hex literals so
 * every widget looks like one book.
 */
export const palette = {
  primary: "#4caf50",   // green — main curve, success
  secondary: "#42a5f5", // blue — secondary curve, info
  warning: "#ffb74d",   // orange — half-life, edge cases
  danger: "#e57373",    // red — reference lines, thresholds
  accent: "#ba68c8",    // purple — third series
  muted: "#888",        // grey — y=x reference, faint guides
};

/** Common Plot mark style for dashed reference lines (rules, thresholds). */
export const dashed = { strokeDasharray: "4 2" };

/** Common Plot text-mark style for annotations on a chart. */
export const annotation = { fontSize: 10, fill: palette.muted };

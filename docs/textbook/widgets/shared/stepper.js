// `defineStepper(spec)` — step-by-step animation harness for widgets
// that walk through a derivation (matrix multiplication, summation,
// MCTS expansion, value-iteration sweeps, Bellman backups, etc.).
//
// A stepper computes its trajectory up-front as a pure function of the
// input parameters, then renders one frame at a time. Backward stepping
// is therefore free — just decrement the index. Auto-play uses a
// configurable interval.
//
// Usage:
//
//     import { defineStepper } from "../shared/stepper.js";
//
//     defineStepper({
//       hostId: "ch1-matmul-widget",
//       controls: {
//         m: { label: "m", min: 1, max: 4, step: 1, default: 2 },
//         n: { label: "n", min: 1, max: 4, step: 1, default: 3 },
//         p: { label: "p", min: 1, max: 4, step: 1, default: 2 },
//       },
//       // Pure function: params → array of frames.
//       trajectory: ({ m, n, p }) => {
//         const frames = [];
//         for (let i = 0; i < m; i++)
//           for (let j = 0; j < p; j++)
//             for (let k = 0; k < n; k++)
//               frames.push({ i, j, k });
//         return frames;
//       },
//       // Render one frame into the host. `frame` is the indexed element.
//       render: (host, frame, idx, total, params, slots) => {
//         slots.main.replaceChildren(renderFrame(frame, params));
//         slots.readout.textContent = `step ${idx + 1} / ${total}`;
//       },
//       playIntervalMs: 800,  // optional, default 800
//     });
//
// The host gets: a controls bar, prev/next/play/reset buttons, and the
// configured slot DIVs. Trajectory recomputes whenever any control input
// changes; index resets to 0 on recompute.

/**
 * @param {Object} spec
 * @param {string} spec.hostId
 * @param {Object} spec.controls — same shape as defineWidget.
 * @param {string[]} [spec.slots] — extra plot slots. Default: `["main"]`.
 * @param {(params: Object) => any[]} spec.trajectory — pure function
 *   returning the frame array. Re-run on every controls change.
 * @param {(host: HTMLElement, frame: any, idx: number, total: number, params: Object, slots: Object) => void} spec.render
 * @param {number} [spec.playIntervalMs] — auto-play tick interval, ms.
 */
export function defineStepper(spec) {
  const {
    hostId,
    controls = {},
    slots: slotNames = ["main"],
    trajectory,
    render,
    playIntervalMs = 800,
  } = spec;

  function mount() {
    const host = document.getElementById(hostId);
    if (!host) return;
    host.innerHTML = renderScaffold(controls, slotNames);
    const slots = collectSlots(host, slotNames);
    const buttons = collectButtons(host);

    let frames = [];
    let idx = 0;
    let timer = null;

    function recompute() {
      stopPlay();
      const params = readParams(host, controls);
      frames = trajectory(params);
      idx = 0;
      paint();
    }

    function paint() {
      if (frames.length === 0) return;
      const params = readParams(host, controls);
      render(host, frames[idx], idx, frames.length, params, slots);
      buttons.prev.disabled = idx <= 0;
      buttons.next.disabled = idx >= frames.length - 1;
      buttons.play.textContent = timer ? "⏸︎ pause" : "▶ play";
    }

    function step(delta) {
      const next = idx + delta;
      if (next < 0 || next >= frames.length) return;
      idx = next;
      paint();
    }

    function togglePlay() {
      if (timer) {
        stopPlay();
      } else {
        timer = setInterval(() => {
          if (idx >= frames.length - 1) {
            stopPlay();
          } else {
            idx += 1;
            paint();
          }
        }, playIntervalMs);
        paint();
      }
    }

    function stopPlay() {
      if (timer) {
        clearInterval(timer);
        timer = null;
        paint();
      }
    }

    buttons.prev.addEventListener("click", () => step(-1));
    buttons.next.addEventListener("click", () => step(+1));
    buttons.play.addEventListener("click", togglePlay);
    buttons.reset.addEventListener("click", () => {
      idx = 0;
      paint();
    });
    for (const el of host.querySelectorAll("input, select")) {
      el.addEventListener("input", recompute);
      el.addEventListener("change", recompute);
    }
    recompute();
  }

  if (typeof document !== "undefined") {
    if (document.readyState !== "loading") mount();
    else document.addEventListener("DOMContentLoaded", mount);
  }
  return mount;
}

// --- internals shared with widget.js (kept inline to avoid an extra import) -----

function renderScaffold(controls, slotNames) {
  const controlHtml = Object.entries(controls)
    .map(([name, c]) => renderControl(name, c))
    .join("");
  const plotSlots = slotNames
    .map((n) => `<div data-plot="${n}"></div>`)
    .join("");
  return `
    <div class="widget-controls">
      ${controlHtml}
      <span data-readout></span>
    </div>
    ${plotSlots}
    <div class="widget-controls" style="margin-top: 0.4em;">
      <button data-button="reset">↺ reset</button>
      <button data-button="prev">◀ prev</button>
      <button data-button="play">▶ play</button>
      <button data-button="next">next ▶</button>
    </div>
  `;
}

function renderControl(name, c) {
  const type = c.type || "range";
  if (type === "select") {
    const opts = c.options
      .map(
        (o) =>
          `<option value="${o.value ?? o}"${
            (o.value ?? o) === c.default ? " selected" : ""
          }>${o.label ?? o}</option>`
      )
      .join("");
    return `<label>${c.label}
      <select data-input="${name}">${opts}</select>
    </label>`;
  }
  return `<label>${c.label}
    <input type="${type}" min="${c.min}" max="${c.max}" step="${c.step}"
           value="${c.default}" data-input="${name}">
  </label>`;
}

function readParams(host, controls) {
  const params = {};
  for (const [name, c] of Object.entries(controls)) {
    const el = host.querySelector(`[data-input="${name}"]`);
    if (!el) continue;
    const raw = el.value;
    params[name] = c.type === "select" ? raw : parseFloat(raw);
  }
  return params;
}

function collectSlots(host, slotNames) {
  const slots = { readout: host.querySelector("[data-readout]") };
  for (const name of slotNames) {
    slots[name] = host.querySelector(`[data-plot="${name}"]`);
  }
  return slots;
}

function collectButtons(host) {
  return {
    prev: host.querySelector('[data-button="prev"]'),
    next: host.querySelector('[data-button="next"]'),
    play: host.querySelector('[data-button="play"]'),
    reset: host.querySelector('[data-button="reset"]'),
  };
}

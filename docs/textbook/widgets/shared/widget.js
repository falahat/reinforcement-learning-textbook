// `defineWidget(spec)` — universal scaffolding for textbook widgets.
//
// Eliminates the ~30 lines of boilerplate (HOST_ID const, CONTROLS_HTML
// template, mount() function, DOMContentLoaded guard, getElementById +
// innerHTML + autoRender wiring) that every textbook widget repeats.
//
// A widget is now just:
//
//     import { defineWidget } from "../shared/widget.js";
//     import * as Plot from "@observablehq/plot";
//
//     defineWidget({
//       hostId: "ch17-gamma-widget",
//       controls: {
//         gamma: { label: "γ", min: 0.5, max: 0.999, step: 0.005, default: 0.9 },
//         kmax:  { label: "k_max", min: 50, max: 1000, step: 10, default: 500 },
//       },
//       render: (host, { gamma, kmax }, slots) => {
//         slots.main.replaceChildren(Plot.plot({ ... }));
//         slots.readout.textContent = `γ^k_max = ${(gamma ** kmax).toExponential(3)}`;
//       },
//     });
//
// The DOM scaffold is built once on first paint, then `render` is called
// after every input event. `slots` is `{ main, readout, [...named slots] }`.

/**
 * Define a textbook widget. Returns the mount function (call manually if
 * needed; otherwise it auto-mounts on DOMContentLoaded).
 *
 * @param {Object} spec
 * @param {string} spec.hostId — id of the `<div>` to mount into.
 * @param {Object} spec.controls — map of `name → control descriptor`. Each
 *   descriptor needs `{ label, min, max, step, default }` for sliders, or
 *   `{ type: "select", label, options: [...], default }` for dropdowns,
 *   or `{ type: "number", label, min, max, step, default }` for number
 *   inputs. Order of keys determines order in the UI.
 * @param {string[]} [spec.slots] — names of additional plot slots. The
 *   render function receives a `slots` object with `main`, `readout`, and
 *   one entry per name. Default: `["main"]`.
 * @param {(host: HTMLElement, params: Object, slots: Object) => void} spec.render
 */
export function defineWidget(spec) {
  const { hostId, controls = {}, slots: slotNames = ["main"], render } = spec;

  function mount() {
    const host = document.getElementById(hostId);
    if (!host) return;
    host.innerHTML = renderScaffold(controls, slotNames);
    const slots = collectSlots(host, slotNames);
    const update = () => render(host, readParams(host, controls), slots);
    for (const el of host.querySelectorAll("input, select")) {
      el.addEventListener("input", update);
      el.addEventListener("change", update);
    }
    update();
  }

  if (typeof document !== "undefined") {
    if (document.readyState !== "loading") mount();
    else document.addEventListener("DOMContentLoaded", mount);
  }
  return mount;
}

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
  // range or number
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
    if (c.type === "select") {
      params[name] = raw;
    } else {
      params[name] = parseFloat(raw);
    }
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

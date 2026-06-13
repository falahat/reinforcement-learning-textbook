// Widget 3.D — POMDP belief-state demo, the Tiger problem (Chapter 3, §3.7).
//
// Classic POMDP from [Kaelbling, Littman & Cassandra 1998].
// Two doors hide a tiger and treasure. The world state s ∈ {TL, TR}
// (tiger-left / tiger-right) is hidden. Each Listen action emits an
// observation o ∈ {HL, HR} (hear-left / hear-right) that agrees with
// the true state with probability α (accuracy) and disagrees with
// probability 1 − α.
//
// Belief b = P(s = TL | history). Bayes update on observation o:
//   b' ∝ P(o | s) · b(s)
// so for o = HL:
//   b'(TL) ∝ α · b(TL)
//   b'(TR) ∝ (1 − α) · b(TR)
// and renormalise. Symmetric for HR.
//
// Opening a door resets the belief to b = 0.5 (a fresh trial starts).
//
// Pedagogical point: a single listen barely moves the belief at α ≈ 0.7;
// repeated correlated listens drive it toward 0 or 1; if accuracy is
// near 0.5 (no signal) the belief stays flat forever — that's why the
// optimal Tiger policy lists "listen at least twice before opening."
//
// We don't use defineStepper here because the user picks actions
// interactively (not "next/prev frame"); the widget mutates an explicit
// state object and re-renders.
//
// Mount:
//   <div id="ch3-pomdp-widget" class="textbook-widget"></div>

import * as Plot from "@observablehq/plot";
import { plotDefaults, palette, fmt } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const HOST_ID = "ch3-pomdp-widget";

const SCAFFOLD_HTML = `
  <div class="widget-controls">
    <label>accuracy α
      <input type="range" min="0.5" max="0.99" step="0.01" value="0.85" data-input="alpha">
    </label>
    <label>seed
      <input type="number" min="0" max="999" step="1" value="7" data-input="seed">
    </label>
    <button data-action="listen">listen</button>
    <button data-action="open-left">open ←</button>
    <button data-action="open-right">open →</button>
    <button data-action="reset">↺ reset</button>
    <span data-readout></span>
  </div>
  <div data-plot="belief"></div>
  <div data-plot="history"></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = SCAFFOLD_HTML;

  // World state hidden from the agent; we resample it on reset.
  const state = {
    belief: 0.5,         // P(tiger left)
    history: [],         // sequence of { action, obs?, truth?, belief }
    truth: null,         // hidden true side of the tiger
  };

  function seedAndReset() {
    const seed = parseInt(host.querySelector('[data-input="seed"]').value, 10) || 0;
    const rng = mulberry32(seed);
    state.belief = 0.5;
    state.truth = rng() < 0.5 ? "TL" : "TR";
    state.history = [];
    state.rng = mulberry32(seed + 1);  // separate stream for observations
    render();
  }

  function readAlpha() {
    return parseFloat(host.querySelector('[data-input="alpha"]').value);
  }

  function listen() {
    const alpha = readAlpha();
    // Emit observation: correct with prob α.
    const correct = state.rng() < alpha;
    const obs = correct
      ? (state.truth === "TL" ? "HL" : "HR")
      : (state.truth === "TL" ? "HR" : "HL");
    // Bayes update.
    const b = state.belief;
    let bNew;
    if (obs === "HL") {
      const num = alpha * b;
      const den = alpha * b + (1 - alpha) * (1 - b);
      bNew = den > 0 ? num / den : b;
    } else {
      const num = (1 - alpha) * b;
      const den = (1 - alpha) * b + alpha * (1 - b);
      bNew = den > 0 ? num / den : b;
    }
    state.belief = bNew;
    state.history.push({ t: state.history.length + 1, action: "listen", obs, belief: bNew });
    render();
  }

  function openDoor(side) {
    // Reveals truth; reset trial.
    const correct = (side === "left" && state.truth === "TR")
                 || (side === "right" && state.truth === "TL");
    state.history.push({
      t: state.history.length + 1,
      action: side === "left" ? "open ←" : "open →",
      obs: correct ? "treasure" : "TIGER",
      belief: 0.5,
    });
    state.belief = 0.5;
    // New trial: resample truth.
    state.truth = state.rng() < 0.5 ? "TL" : "TR";
  }

  function render() {
    const alpha = readAlpha();
    const b = state.belief;

    // Belief bar: a single-row bar from 0 to b, with rest gradient.
    const beliefData = [
      { label: "tiger left", p: b },
      { label: "tiger right", p: 1 - b },
    ];
    host.querySelector('[data-plot="belief"]').replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 110,
      marginLeft: 90,
      marginBottom: 28,
      x: { domain: [0, 1], grid: true, label: "P(state | history)" },
      y: { domain: ["tiger left", "tiger right"], label: null },
      color: {
        domain: ["tiger left", "tiger right"],
        range: [palette.danger, palette.primary],
      },
      marks: [
        Plot.barX(beliefData, { y: "label", x: "p", fill: "label" }),
        Plot.text(beliefData, {
          y: "label", x: "p",
          text: (d) => d.p.toFixed(3),
          textAnchor: "start", dx: 6, fill: "white", fontSize: 11,
        }),
        Plot.ruleX([0.5], { stroke: palette.muted, strokeDasharray: "3 2" }),
      ],
    }));

    // History ribbon: t on x, observation glyph as colour.
    const histData = state.history;
    if (histData.length === 0) {
      host.querySelector('[data-plot="history"]').replaceChildren();
    } else {
      // Map observation → fill colour and label.
      const colour = (h) => {
        if (h.obs === "HL") return palette.secondary;
        if (h.obs === "HR") return palette.accent;
        if (h.obs === "treasure") return palette.primary;
        return palette.danger;
      };
      const TMAX = histData.length;
      const ribbon = histData.map((h) => ({
        t: h.t,
        belief: h.belief,
        action: h.action,
        obs: h.obs,
        colour: colour(h),
      }));
      host.querySelector('[data-plot="history"]').replaceChildren(Plot.plot({
        ...plotDefaults,
        height: 220,
        marginLeft: 50,
        marginBottom: 36,
        x: { domain: [0.5, TMAX + 0.5], label: "step t", ticks: TMAX <= 20 ? TMAX : 10 },
        y: { domain: [0, 1], label: "belief b(tiger left)", grid: true },
        marks: [
          Plot.ruleY([0.5], { stroke: palette.muted, strokeDasharray: "3 2" }),
          // Observation glyphs as a coloured strip just below the line.
          Plot.rect(ribbon, {
            x1: (d) => d.t - 0.4,
            x2: (d) => d.t + 0.4,
            y1: -0.04, y2: 0.02,
            fill: "colour",
            clip: false,
          }),
          Plot.text(ribbon, {
            x: "t", y: 0.04,
            text: "obs", fill: "colour", fontSize: 10,
          }),
          // Belief trajectory.
          Plot.line(ribbon, {
            x: "t", y: "belief", stroke: palette.warning, strokeWidth: 2,
          }),
          Plot.dot(ribbon, {
            x: "t", y: "belief", fill: palette.warning, r: 3,
          }),
        ],
      }));
    }

    const recentOpen = histData.slice().reverse().find((h) => h.action.startsWith("open"));
    const status = recentOpen
      ? `last: ${recentOpen.action} → ${recentOpen.obs}`
      : `belief sharpening (α = ${alpha.toFixed(2)})`;
    host.querySelector("[data-readout]").textContent =
      `b(TL) = ${fmt(b)}  ·  ${state.history.length} steps  ·  ${status}`;
  }

  host.querySelector('[data-action="listen"]').addEventListener("click", listen);
  host.querySelector('[data-action="open-left"]').addEventListener("click", () => {
    openDoor("left");
    render();
  });
  host.querySelector('[data-action="open-right"]').addEventListener("click", () => {
    openDoor("right");
    render();
  });
  host.querySelector('[data-action="reset"]').addEventListener("click", seedAndReset);
  host.querySelector('[data-input="alpha"]').addEventListener("input", render);
  host.querySelector('[data-input="seed"]').addEventListener("change", seedAndReset);

  seedAndReset();
}

if (typeof document !== "undefined") {
  if (document.readyState !== "loading") mount();
  else document.addEventListener("DOMContentLoaded", mount);
}

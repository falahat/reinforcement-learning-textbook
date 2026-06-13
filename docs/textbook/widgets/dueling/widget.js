// Widget 9.B — Dueling DQN decomposition visualiser (Chapter 9).
//
// Visualises the Wang et al. 2016 dueling decomposition:
//   Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
// i.e. V absorbs the state-wide baseline (the "alive" value) and the
// advantages A_i = Q_i − V are zero-mean across actions.
//
// Three sliders set Q_0, Q_1, Q_2 directly; the widget plots three
// faceted bar charts side-by-side (Q | V | A) so the reader can slide
// all three Q's together and *see* V track the uniform shift while
// the advantages A stay put. That's the architectural fix for the
// Q-bias bootstrap pathology Chapter 15 develops: V absorbs the
// alive baseline, A preserves the action-relative ordering.
//
// Pattern: chapter markdown contains a `<div id="ch9-dueling-widget"
// class="textbook-widget">` mount point + a `<script type="module"
// src="./widgets/dueling/widget.js"></script>` reference.

import * as Plot from "@observablehq/plot";
import {
  readNumber,
  autoRender,
  setReadout,
  fmt,
  plotDefaults,
} from "../shared/helpers.js";

const HOST_ID = "ch9-dueling-widget";

const CONTROLS_HTML = `
  <div class="widget-controls">
    <label>Q₀
      <input type="range" min="-5" max="5" step="0.1"
             value="1" data-input="q0">
    </label>
    <label>Q₁
      <input type="range" min="-5" max="5" step="0.1"
             value="2" data-input="q1">
    </label>
    <label>Q₂
      <input type="range" min="-5" max="5" step="0.1"
             value="3" data-input="q2">
    </label>
    <span data-readout></span>
  </div>
  <div data-plot></div>
`;

function mount() {
  const host = document.getElementById(HOST_ID);
  if (!host) return;
  host.innerHTML = CONTROLS_HTML;

  autoRender(host, () => {
    const q0 = readNumber(host, '[data-input="q0"]');
    const q1 = readNumber(host, '[data-input="q1"]');
    const q2 = readNumber(host, '[data-input="q2"]');

    const qs = [q0, q1, q2];
    const v = (q0 + q1 + q2) / 3;
    const advs = qs.map((q) => q - v);

    // Build a long-form table so we can facet by stream (Q | V | A)
    // and use action index as the x channel within each facet. The
    // y-domain is shared across facets so the reader can compare
    // magnitudes directly.
    const labels = ["a₀", "a₁", "a₂"];
    const rows = [];
    for (let i = 0; i < 3; i++) {
      rows.push({ stream: "Q(s, a)", action: labels[i], value: qs[i] });
      rows.push({ stream: "V(s)", action: labels[i], value: v });
      rows.push({ stream: "A(s, a)", action: labels[i], value: advs[i] });
    }

    // Pick a y-domain that always includes 0 and gives a bit of
    // headroom. Slider range is [-5, 5] so worst-case |Q| = 5 and
    // worst-case |A| = (5 − (−5))·2/3 ≈ 6.67. Round up to 7.
    const yMax = 7;

    const colorFor = (d) =>
      d.stream === "Q(s, a)"
        ? "#90caf9"
        : d.stream === "V(s)"
          ? "#4caf50"
          : "#ffb74d";

    host.querySelector("[data-plot]").replaceChildren(
      Plot.plot({
        ...plotDefaults,
        height: 300,
        marginLeft: 44,
        marginBottom: 40,
        x: { label: "action", domain: labels },
        y: { label: "value", domain: [-yMax, yMax], grid: true },
        fx: { label: null, domain: ["Q(s, a)", "V(s)", "A(s, a)"] },
        marks: [
          Plot.frame({ stroke: "#888", strokeOpacity: 0.25 }),
          Plot.ruleY([0], { stroke: "#888", strokeOpacity: 0.5 }),
          Plot.barY(rows, {
            fx: "stream",
            x: "action",
            y: "value",
            fill: colorFor,
            fillOpacity: 0.85,
          }),
          Plot.text(rows, {
            fx: "stream",
            x: "action",
            y: "value",
            text: (d) => fmt(d.value),
            dy: (d) => (d.value >= 0 ? -6 : 12),
            fontSize: 10,
            fill: "#ddd",
          }),
        ],
      })
    );

    // Advantages sum to 0 by construction; show that explicitly so
    // the reader can verify it as they slide.
    const advSum = advs.reduce((s, x) => s + x, 0);
    setReadout(
      host,
      `V = mean(Q) = ${fmt(v)}   |   ` +
        `A = [${advs.map(fmt).join(", ")}]   |   ` +
        `Σ A = ${fmt(advSum)} (≡ 0)`
    );
  });
}

if (document.readyState !== "loading") {
  mount();
} else {
  document.addEventListener("DOMContentLoaded", mount);
}

// Widget 14.F — DIAYN skill-coverage map.
//
// Five skills z ∈ {0,1,2,3,4} each parameterised by a different drift
// direction in a 2D state space. Each skill runs for 100 steps from
// the origin under Brownian motion biased along its drift direction.
// The colour clouds visualise the state-visitation density per skill.
// Diversity = the clouds are visibly distinct.
//
// Mount:
//     <div id="ch14-diayn-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/diayn/widget.js"></script>

import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { mulberry32, gauss } from "../shared/random.js";

const N_SKILLS = 5;
const STEPS_PER = 100;

// Drift directions evenly spaced around the unit circle.
function driftFor(z, drive) {
  const theta = (2 * Math.PI * z) / N_SKILLS;
  return { dx: drive * Math.cos(theta), dy: drive * Math.sin(theta) };
}

const SKILL_COLORS = [
  palette.primary,
  palette.secondary,
  palette.warning,
  palette.danger,
  palette.accent,
];

function rolloutSkill(z, drive, noise, seed) {
  const rng = mulberry32(seed);
  const { dx, dy } = driftFor(z, drive);
  const pts = [];
  let x = 0, y = 0;
  for (let t = 0; t < STEPS_PER; t++) {
    x += dx + noise * gauss(rng);
    y += dy + noise * gauss(rng);
    pts.push({ x, y, z });
  }
  return pts;
}

defineWidget({
  hostId: "ch14-diayn-widget",
  controls: {
    drive: { label: "drift |v|",   min: 0.0, max: 0.5,  step: 0.01, default: 0.10 },
    noise: { label: "step noise σ", min: 0.0, max: 0.5, step: 0.01, default: 0.20 },
    seed:  { label: "seed", min: 1, max: 999, step: 1, default: 9 },
  },
  render: (host, { drive, noise, seed }, slots) => {
    const sd = Math.round(seed);
    let all = [];
    for (let z = 0; z < N_SKILLS; z++) {
      const pts = rolloutSkill(z, drive, noise, sd + 31 * z);
      all = all.concat(pts);
    }

    // Mark each skill's endpoint with a labelled dot (so a quick visual
    // legend is the dot at the end of each cloud).
    const endpoints = [];
    for (let z = 0; z < N_SKILLS; z++) {
      const slice = all.filter((p) => p.z === z);
      const last = slice[slice.length - 1];
      endpoints.push({ x: last.x, y: last.y, z, label: `z=${z}` });
    }

    // Compute a symmetric domain that fits all points with a bit of pad.
    let maxR = 1;
    for (const p of all) {
      maxR = Math.max(maxR, Math.abs(p.x), Math.abs(p.y));
    }
    maxR *= 1.1;

    slots.main.replaceChildren(Plot.plot({
      ...plotDefaults,
      height: 380,
      marginLeft: 50,
      x: { label: "state x", domain: [-maxR, maxR], grid: true },
      y: { label: "state y", domain: [-maxR, maxR], grid: true },
      color: {
        legend: true,
        domain: [0, 1, 2, 3, 4],
        range: SKILL_COLORS,
        tickFormat: (z) => `z=${z}`,
      },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.ruleY([0], { stroke: palette.muted, strokeOpacity: 0.3 }),
        Plot.dot(all, {
          x: "x", y: "y", fill: "z", r: 2.2, fillOpacity: 0.45,
        }),
        Plot.dot(endpoints, {
          x: "x", y: "y", fill: "z", stroke: "#fff", strokeWidth: 1.5, r: 5,
        }),
        Plot.text(endpoints, {
          x: "x", y: "y", text: "label",
          dx: 8, dy: -8, fontSize: 10, fill: "#eee", textAnchor: "start",
        }),
      ],
    }));

    const ratio = drive > 0 ? drive / Math.max(noise, 1e-6) : 0;
    slots.readout.textContent =
      `${N_SKILLS} skills × ${STEPS_PER} steps. ` +
      `drift/noise ratio = ${ratio.toFixed(2)} ` +
      `(< 1 → clouds overlap; > 1 → distinct lobes).`;
  },
});

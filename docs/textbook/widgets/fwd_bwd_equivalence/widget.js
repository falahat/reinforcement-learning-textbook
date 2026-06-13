// Widget 7.E — Forward-backward equivalence stepper (Chapter 7).
//
// The forward-backward equivalence theorem [S&B 2018, §12.4] says: the
// *sum* of backward-view TD(λ) per-step updates over one episode equals
// the forward-view offline λ-return update on the same episode. This
// stepper walks through a 5-state random-walk episode and shows both
// computations side-by-side.
//
// Setup (S&B Example 7.1):
//   - States A, B, C, D, E (indices 0..4); two terminals L=0, R=+1.
//   - Equal-prob left/right walk from a chosen start (default: C).
//   - Tabular V initialised to 0.5 for non-terminals (S&B convention).
//   - One episode, then offline batch update.
//
// Frame 0:        Episode generated. No updates applied.
// Frames 1..T:    Backward view — step t advances by one tick, showing
//                 δ_t, e(s) for every state, and the per-state running
//                 ΔV(s) = α · Σ_{τ≤t} δ_τ · e_τ(s).
// Frame T+1:      Forward view — for each visited state s_t in the
//                 episode, compute G_t^λ exactly and the forward update
//                 ΔV_fwd(s_t) = α · (G_t^λ − V(s_t)). Show the summed
//                 backward update vs the forward update per state. They
//                 match (to floating-point precision).
//
// Slider for λ ∈ [0, 1]; α and γ fixed at 0.1 and 1.0 for clarity.
//
// Pattern: chapter markdown has
//
//     <div id="ch7-fwd-bwd-widget" class="textbook-widget"></div>
//     <script type="module" src="./widgets/fwd_bwd_equivalence/widget.js"></script>

import * as Plot from "@observablehq/plot";
import * as d3 from "d3";
import { defineStepper } from "../shared/stepper.js";
import { plotDefaults, palette, dashed, annotation } from "../shared/helpers.js";
import { mulberry32 } from "../shared/random.js";

const STATE_NAMES = ["A", "B", "C", "D", "E"];
const N_STATES = STATE_NAMES.length;
const GAMMA = 1.0;          // undiscounted, S&B convention for random walk
const ALPHA = 0.1;
const V_INIT = 0.5;         // V_0 for non-terminals
const SEED = 7;             // deterministic episode for stable visuals

// Generate a random-walk episode starting at C (index 2). Returns a
// list of step records {s_t, r_t, s_next, terminal_at_next, term_kind}
// where term_kind ∈ {"L", "R", null} when the *next* state is terminal.
function generateEpisode(seed) {
  const rng = mulberry32(seed);
  const steps = [];
  let s = 2;
  for (let t = 0; t < 200; t++) {
    const goRight = rng() < 0.5;
    const sNext = goRight ? s + 1 : s - 1;
    // Terminal handling: -1 = left terminal (reward 0), 5 = right terminal (reward +1).
    let r = 0;
    let terminalNext = false;
    let termKind = null;
    if (sNext < 0) {
      terminalNext = true;
      termKind = "L";
      r = 0;
    } else if (sNext >= N_STATES) {
      terminalNext = true;
      termKind = "R";
      r = 1;
    }
    steps.push({ s, sNext, r, terminalNext, termKind });
    if (terminalNext) break;
    s = sNext;
  }
  return steps;
}

// Bootstrap V at the next state — 0 if terminal, V[s_next] otherwise.
function vNext(V, step) {
  if (step.terminalNext) return 0;
  return V[step.sNext];
}

// Per-step backward update: δ_t, then for every non-terminal state s
// accumulate ΔV[s] += α δ_t e[s]; finally bump e[s_t] (replacing-trace
// shape: e ← γλ·e + 1 for the visited state) and decay all traces.
// (We use accumulating for visual clarity; replacing would also work.)
function backwardStep(state, step, lambda) {
  const { V, e, dV } = state;
  const delta = step.r + GAMMA * vNext(V, step) - V[step.s];
  // First bump the trace for the visited state (accumulating).
  e[step.s] += 1;
  // Apply δ·e to every state.
  for (let s = 0; s < N_STATES; s++) {
    dV[s] += ALPHA * delta * e[s];
  }
  // Decay all traces.
  for (let s = 0; s < N_STATES; s++) {
    e[s] *= GAMMA * lambda;
  }
  return delta;
}

// Forward-view λ-return at time t: G_t^λ = (1-λ) Σ_{n=1..T-t-1} λ^{n-1} G_t^{(n)}
//                                          + λ^{T-t-1} G_t (the truncation
//                                          term for finite episodes).
// We compute it directly from the episode's reward/state sequence.
function forwardLambdaReturn(steps, t, V0, lambda) {
  const T = steps.length;            // number of transitions
  // n-step return G_t^{(n)} for n = 1..(T - t):
  //   G_t^{(n)} = sum_{k=0..n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) if non-terminal,
  //                                            or no bootstrap if terminal reached.
  // Build G_t^{(n)} incrementally.
  const Gn = [];
  let acc = 0;
  let stopped = false;
  for (let n = 1; n <= T - t; n++) {
    const stepIdx = t + n - 1;
    const step = steps[stepIdx];
    acc += Math.pow(GAMMA, n - 1) * step.r;
    // If the next state after this step is terminal, the n-step return
    // for this n has no bootstrap — and for all larger n it equals this
    // truncated return (the trajectory ended).
    if (step.terminalNext) {
      Gn.push({ n, G: acc, terminal: true });
      stopped = true;
      break;
    }
    // Non-terminal: bootstrap with V(s_{t+n}).
    const Vbootstrap = V0[step.sNext];
    Gn.push({ n, G: acc + Math.pow(GAMMA, n) * Vbootstrap, terminal: false });
  }
  // λ-return: explicit weighted sum over horizons.
  // For finite episode that terminates at horizon N_T = Gn.length, the
  // formula collapses to (1-λ) Σ_{n=1..N_T-1} λ^{n-1} G_t^{(n)}
  //                    + λ^{N_T-1} G_t^{(N_T)}
  // (the terminal-truncation tail carries the residual mass).
  const NT = Gn.length;
  let G_lambda = 0;
  for (let n = 1; n < NT; n++) {
    G_lambda += (1 - lambda) * Math.pow(lambda, n - 1) * Gn[n - 1].G;
  }
  G_lambda += Math.pow(lambda, NT - 1) * Gn[NT - 1].G;
  return { G_lambda, Gn };
}

defineStepper({
  hostId: "ch7-fwd-bwd-widget",
  controls: {
    lambda: { label: "λ (lambda)", min: 0, max: 1, step: 0.05, default: 0.8 },
  },
  slots: ["main", "table"],
  trajectory: ({ lambda }) => {
    const steps = generateEpisode(SEED);
    const V0 = new Array(N_STATES).fill(V_INIT);
    const e = new Array(N_STATES).fill(0);
    const dV = new Array(N_STATES).fill(0);

    // Frame 0: pre-update.
    const frames = [{
      kind: "intro",
      steps,
      V0: V0.slice(),
      e: e.slice(),
      dV: dV.slice(),
      delta: null,
      tIdx: -1,
    }];

    // Frames 1..T: one backward step each.
    const state = { V: V0.slice(), e: e.slice(), dV: dV.slice() };
    for (let t = 0; t < steps.length; t++) {
      const delta = backwardStep(state, steps[t], lambda);
      frames.push({
        kind: "backward",
        steps,
        V0: V0.slice(),
        e: state.e.slice(),
        dV: state.dV.slice(),
        delta,
        tIdx: t,
        currStep: steps[t],
      });
    }

    // Final frame: forward view & comparison.
    const fwdUpdates = new Array(N_STATES).fill(0);
    const perVisit = [];
    for (let t = 0; t < steps.length; t++) {
      const s_t = steps[t].s;
      const { G_lambda } = forwardLambdaReturn(steps, t, V0, lambda);
      const upd = ALPHA * (G_lambda - V0[s_t]);
      fwdUpdates[s_t] += upd;
      perVisit.push({ t, s_t, G_lambda, V_st: V0[s_t], upd });
    }
    frames.push({
      kind: "forward",
      steps,
      V0: V0.slice(),
      e: state.e.slice(),
      dV: state.dV.slice(),
      fwdUpdates,
      perVisit,
      tIdx: steps.length,
    });

    return frames;
  },
  playIntervalMs: 700,
  render: (host, frame, idx, total, params, slots) => {
    const { lambda } = params;

    // Always paint the trace+ΔV bar group as the "main" slot.
    const traceRows = STATE_NAMES.map((name, i) => ({
      state: name,
      e: frame.e[i],
      dV: frame.dV[i],
    }));

    // Two grouped bars per state: e (trace) in primary, ΔV in secondary.
    // Plot doesn't have a native grouped bar; render two faceted plots
    // stacked.
    const eMax = Math.max(0.05, ...traceRows.map((r) => Math.abs(r.e)));
    const dvMax = Math.max(0.02, ...traceRows.map((r) => Math.abs(r.dV)),
      ...(frame.fwdUpdates ?? []).map((v) => Math.abs(v)));

    const tracePlot = Plot.plot({
      ...plotDefaults,
      height: 150,
      marginLeft: 70,
      x: { label: "trace e(s)", domain: [0, eMax * 1.15] },
      // .slice() first so we reverse a COPY — Array.prototype.reverse
      // mutates in place. Reversing STATE_NAMES directly would pollute
      // every subsequent indexed lookup (STATE_NAMES[step.s]) used to
      // build the episode strip and the per-visit table, swapping A/E
      // and B/D in the displayed text relative to the actual episode.
      y: { label: "state", domain: STATE_NAMES.slice().reverse() },
      marks: [
        Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
        Plot.barX(traceRows, {
          x: "e",
          y: "state",
          fill: palette.secondary,
          fillOpacity: 0.85,
        }),
        Plot.text(traceRows, {
          x: "e",
          y: "state",
          text: (d) => d.e.toFixed(3),
          dx: 6,
          textAnchor: "start",
          fill: palette.muted,
          ...annotation,
        }),
      ],
    });

    // Comparison plot. Shows backward-cumulative ΔV(s); on the forward
    // frame, overlays the forward-view target.
    const compareRows = STATE_NAMES.map((name, i) => ({
      state: name,
      backward: frame.dV[i],
      forward: frame.kind === "forward" ? frame.fwdUpdates[i] : null,
    }));
    const compMarks = [
      Plot.ruleX([0], { stroke: palette.muted, strokeOpacity: 0.4 }),
      Plot.barX(compareRows, {
        x: "backward",
        y: "state",
        fill: palette.primary,
        fillOpacity: 0.7,
      }),
      Plot.text(compareRows, {
        x: "backward",
        y: "state",
        text: (d) => d.backward.toFixed(4),
        dx: (d) => (d.backward >= 0 ? 6 : -6),
        textAnchor: (d) => (d.backward >= 0 ? "start" : "end"),
        fill: palette.primary,
        ...annotation,
      }),
    ];
    if (frame.kind === "forward") {
      compMarks.push(
        Plot.dot(compareRows, {
          x: "forward",
          y: "state",
          fill: palette.danger,
          r: 5,
        }),
        Plot.text(compareRows.filter((r) => r.forward !== null), {
          x: "forward",
          y: "state",
          text: (d) => `fwd=${d.forward.toFixed(4)}`,
          dy: -10,
          textAnchor: "middle",
          fill: palette.danger,
          ...annotation,
        }),
      );
    }
    const compareDomain = [
      Math.min(0, -dvMax * 1.15),
      Math.max(0, dvMax * 1.15),
    ];
    const comparePlot = Plot.plot({
      ...plotDefaults,
      height: 150,
      marginLeft: 70,
      x: {
        label: frame.kind === "forward"
          ? "ΔV(s): backward (bar)  vs  forward (red dot)"
          : "cumulative backward ΔV(s) = α Σ δ_τ e_τ(s)",
        domain: compareDomain,
      },
      y: { label: "state", domain: STATE_NAMES.slice() },
      marks: compMarks,
    });

    const wrap = document.createElement("div");
    wrap.style.display = "grid";
    wrap.style.gridTemplateColumns = "1fr 1fr";
    wrap.style.gap = "0.5em";
    wrap.appendChild(tracePlot);
    wrap.appendChild(comparePlot);
    slots.main.replaceChildren(wrap);

    // ---------- table slot: episode trace, current step, equivalence -----
    const tbl = document.createElement("div");
    tbl.style.fontSize = "11px";
    tbl.style.lineHeight = "1.4";
    tbl.style.fontFamily = "monospace";

    // Episode strip.
    const stripParts = [];
    for (let i = 0; i < frame.steps.length; i++) {
      const st = frame.steps[i];
      let marker;
      if (frame.kind === "backward" && i === frame.tIdx) {
        marker = `<b style="color:${palette.danger}">[${STATE_NAMES[st.s]}]</b>`;
      } else if (i < (frame.tIdx ?? -1)) {
        marker = `<span style="color:${palette.muted}">${STATE_NAMES[st.s]}</span>`;
      } else {
        marker = STATE_NAMES[st.s];
      }
      stripParts.push(marker);
    }
    // Final terminal marker (no leading arrow — the join already adds one).
    const lastStep = frame.steps[frame.steps.length - 1];
    const termSym = lastStep.termKind === "R" ? "R (+1)" : "L (0)";
    stripParts.push(`<span style="color:${palette.warning}">${termSym}</span>`);
    tbl.innerHTML = `<div><b>episode:</b> ${stripParts.join(" → ")}</div>`;

    if (frame.kind === "backward") {
      const { currStep, delta } = frame;
      const Vs = frame.V0[currStep.s];
      const Vsp = currStep.terminalNext ? 0 : frame.V0[currStep.sNext];
      const target = currStep.terminalNext
        ? `r=${currStep.r}`
        : `r + γV(${STATE_NAMES[currStep.sNext] ?? currStep.termKind}) = ${currStep.r} + ${GAMMA}·${Vsp.toFixed(3)}`;
      const next = currStep.terminalNext
        ? `[terminal ${currStep.termKind}]`
        : STATE_NAMES[currStep.sNext];
      tbl.innerHTML += `
        <div style="margin-top:0.4em">
          <b>step t = ${frame.tIdx}:</b>
          ${STATE_NAMES[currStep.s]} → ${next} ;
          δ = ${target} − ${Vs.toFixed(3)} = <b style="color:${palette.danger}">${delta.toFixed(4)}</b>
        </div>
        <div style="color:${palette.muted}">
          (apply α δ to every state weighted by e(s); then decay e by γλ)
        </div>
      `;
    } else if (frame.kind === "forward") {
      // Equivalence check.
      const maxAbsErr = Math.max(
        ...STATE_NAMES.map((_, i) => Math.abs(frame.dV[i] - frame.fwdUpdates[i])),
      );
      const ok = maxAbsErr < 1e-9;
      tbl.innerHTML += `
        <div style="margin-top:0.4em">
          <b>forward view computed:</b> for every visit s_t in the
          episode, ΔV_fwd(s_t) = α(G_t^λ − V(s_t)). Per-visit table:
        </div>
        <table style="margin-top:0.3em; border-collapse:collapse">
          <thead>
            <tr style="color:${palette.muted}">
              <th style="padding:0 0.6em; text-align:right">t</th>
              <th style="padding:0 0.6em; text-align:left">s_t</th>
              <th style="padding:0 0.6em; text-align:right">G_t^λ</th>
              <th style="padding:0 0.6em; text-align:right">V(s_t)</th>
              <th style="padding:0 0.6em; text-align:right">ΔV_fwd</th>
            </tr>
          </thead>
          <tbody>
            ${frame.perVisit.map((r) => `
              <tr>
                <td style="padding:0 0.6em; text-align:right">${r.t}</td>
                <td style="padding:0 0.6em">${STATE_NAMES[r.s_t]}</td>
                <td style="padding:0 0.6em; text-align:right">${r.G_lambda.toFixed(4)}</td>
                <td style="padding:0 0.6em; text-align:right">${r.V_st.toFixed(3)}</td>
                <td style="padding:0 0.6em; text-align:right">${r.upd.toFixed(5)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
        <div style="margin-top:0.4em; color:${ok ? palette.primary : palette.danger}">
          <b>max |ΔV_back − ΔV_fwd|</b> across states = ${maxAbsErr.toExponential(2)}
          ${ok ? " ✓ equivalence holds" : " — drift!"}
        </div>
      `;
    } else {
      tbl.innerHTML += `
        <div style="margin-top:0.4em; color:${palette.muted}">
          Episode generated (seed ${SEED}). Press <b>next</b> to step the
          backward view one tick at a time. The last frame shows the
          forward-view λ-return computation and the equivalence check.
        </div>
      `;
    }
    slots.table.replaceChildren(tbl);

    // Readout.
    const kindLabel = frame.kind === "backward"
      ? `backward t = ${frame.tIdx}`
      : frame.kind === "forward"
        ? "forward view"
        : "intro";
    slots.readout.textContent =
      `λ = ${lambda.toFixed(2)}  ·  α = ${ALPHA}  ·  γ = ${GAMMA}  ·  ` +
      `frame ${idx} / ${total - 1}  ·  ${kindLabel}`;
  },
});

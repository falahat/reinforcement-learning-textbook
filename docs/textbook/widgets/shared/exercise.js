// `defineExercise(spec)` — scaffold for live-coding exercises.
//
// Sibling to `defineWidget` (see widget.js). A widget reacts to slider
// input; an exercise reacts to the reader typing a function body,
// pressing Run, and watching their code get checked against either
// a fixed-input table, finite-difference grad-check, or a reference
// implementation imported from `shared/nn.js`.
//
// Authoring shape:
//
//   import { defineExercise } from "../shared/exercise.js";
//   defineExercise({
//     hostId: "ch11-tanh-backward-exercise",
//     title:  "Implement backward for tanh",
//     prompt: `... markdown-ish prose ...`,
//     template: `function backwardTanh(dy, y) { /* ... */ }`,
//     entrypoint: "backwardTanh",                  // exported function name
//     entrypointArgs: ["dy", "y"],                 // arg names for the wrapper
//     check: {
//       kind: "gradCheck",
//       forward: (x) => Math.tanh(x),
//       inputs: { length: 32, distribution: "uniform", range: [-3, 3] },
//       nTests: 5,
//       tolerance: 1e-3,
//       seed: 7,
//     },
//   });
//
// The harness handles:
//   - rendering the editor + buttons + result panel,
//   - localStorage persistence of the student's draft,
//   - parsing the student's code via `new Function(...)`,
//   - running the configured check kind,
//   - rendering pass/fail with diff details,
//   - a solution-reveal toggle (collapsed by default; explicit click).
//
// The exercise spec is *data*. No DOM, no event wiring in the spec.

import { splitmix64, mulberry32 } from "./random.js";

// ─── Public entry point ────────────────────────────────────────────

/**
 * Define a textbook live-coding exercise. Auto-mounts on
 * DOMContentLoaded; returns the mount function for manual control.
 *
 * @param {Object} spec
 * @param {string} spec.hostId — id of the `<div>` to mount into.
 * @param {string} spec.title — short label shown above the editor.
 * @param {string} spec.prompt — student-facing instructions. Plain
 *   text with simple Markdown-ish formatting (`**bold**`, backticks
 *   for code, blank lines for paragraphs). Rendered as paragraphs.
 * @param {string} spec.template — starter code shown in the editor.
 * @param {string} spec.entrypoint — name of the function the student
 *   defines. The harness extracts this function from their code.
 * @param {string[]} [spec.entrypointArgs] — arg names. Optional;
 *   inferred from `template` if omitted.
 * @param {Object} spec.check — check descriptor. See `runCheck`.
 * @param {string} [spec.solution] — reference solution shown on
 *   reveal. Optional; the reveal button hides if absent.
 */
export function defineExercise(spec) {
  // Headless / smoketest hook — `node smoketest.js` sets this to
  // collect every spec without touching the DOM. Pure-data; never set
  // in the browser.
  if (typeof globalThis !== "undefined" && globalThis.__collectExercise) {
    globalThis.__collectExercise(spec);
  }
  function mount() {
    const host = document.getElementById(spec.hostId);
    if (!host) return;
    host.classList.add("textbook-exercise");
    host.innerHTML = scaffold(spec);
    wire(host, spec);
  }
  if (typeof document !== "undefined") {
    if (document.readyState !== "loading") mount();
    else document.addEventListener("DOMContentLoaded", mount);
  }
  return mount;
}

// ─── DOM scaffold ──────────────────────────────────────────────────

function scaffold(spec) {
  const promptHtml = renderPromptHtml(spec.prompt ?? "");
  const showSolution = spec.solution ? "" : `style="display:none"`;
  // Optional signature block — author-supplied type/shape documentation
  // for the function arguments. Renders as a monospace <pre> so column
  // alignment in the source string survives.
  const signatureHtml = spec.signature
    ? `<pre class="exercise-signature">${escapeHtml(spec.signature.trim())}</pre>`
    : "";
  // Optional sample-input disclosure — collapsed by default so the
  // reader sees prompt → signature → editor on first glance, but can
  // expand to see actual numbers the harness will pass. The harness
  // computes the sample at render time using the spec's seed.
  const sampleHtml = hasSampleInput(spec)
    ? `<details class="exercise-sample">
         <summary>▸ Show example input the harness will pass</summary>
         <pre data-sample></pre>
       </details>`
    : "";
  return `
    <div class="exercise-head">
      <span class="exercise-title">${escapeHtml(spec.title)}</span>
      <span class="exercise-status" data-status></span>
    </div>
    <div class="exercise-prompt">${promptHtml}</div>
    ${signatureHtml}
    ${sampleHtml}
    <div class="exercise-editor-host" data-editor-host></div>
    <textarea class="exercise-editor" data-editor spellcheck="false"
      autocapitalize="off" autocomplete="off" autocorrect="off">${
        escapeHtml(spec.template ?? "")
      }</textarea>
    <div class="exercise-controls">
      <button data-btn="run">▶ Run</button>
      <button data-btn="reset">↺ Reset</button>
      <label class="exercise-seed">seed
        <input type="number" data-input="seed" value="${
          spec.check?.seed ?? 7
        }" min="0" max="999999" step="1">
      </label>
      <span class="exercise-attempts" data-attempts></span>
      <button data-btn="reveal" ${showSolution}>▸ Show solution</button>
    </div>
    <div class="exercise-result" data-result></div>
    <pre class="exercise-solution" data-solution hidden>${
      escapeHtml(spec.solution ?? "")
    }</pre>
  `;
}

function hasSampleInput(spec) {
  const c = spec.check;
  if (!c) return false;
  if (c.kind === "fixedTable") return c.cases && c.cases.length > 0;
  if (c.kind === "reference") return typeof c.generate === "function";
  if (c.kind === "gradCheck") return typeof c.forward === "function";
  return false;
}

/**
 * Derive the actual input the harness will pass on test 1 for the
 * given spec. Used by the "Show example input" disclosure so the
 * reader sees concrete numbers, not a prose description.
 */
function deriveSampleInput(spec, seed) {
  const c = spec.check;
  if (c.kind === "fixedTable") {
    return c.cases[0]?.input;
  }
  if (c.kind === "reference") {
    const rng = mulberry32((seed ?? c.seed ?? 7) >>> 0);
    return c.generate(rng, 0);
  }
  if (c.kind === "gradCheck") {
    // Mirror the harness's gradCheck input synthesis.
    const rng = mulberry32((seed ?? c.seed ?? 7) >>> 0);
    const len = c.inputs?.length ?? 16;
    const range = c.inputs?.range ?? [-3, 3];
    const x = sampleVector(rng, len, range);
    const y = new Float32Array(len);
    for (let i = 0; i < len; i++) y[i] = c.forward(x[i]);
    const dy = sampleVector(rng, len, [-1, 1]);
    return [dy, y];
  }
  return null;
}

// Tiny Markdown-ish renderer for the prompt. Real markdown is overkill
// here — the prompt is short paragraphs + inline code + bold.
//
// Newline handling follows Markdown: a blank line starts a new
// paragraph; a single newline inside a paragraph is treated as a
// space so authors can wrap their source at 80 cols without
// surfacing the wrap as `<br>` mid-sentence.
function renderPromptHtml(src) {
  return src
    .trim()
    .split(/\n\s*\n/)
    .map((para) => {
      let html = escapeHtml(para.trim()).replace(/\s*\n\s*/g, " ");
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      return `<p>${html}</p>`;
    })
    .join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ─── Wiring ────────────────────────────────────────────────────────

function wire(host, spec) {
  const sel = (q) => host.querySelector(q);
  const textarea = sel("[data-editor]");
  const editorHost = sel("[data-editor-host]");
  const result  = sel("[data-result]");
  const status  = sel("[data-status]");
  const attempts = sel("[data-attempts]");
  const btnRun     = sel('[data-btn="run"]');
  const btnReset   = sel('[data-btn="reset"]');
  const btnReveal  = sel('[data-btn="reveal"]');
  const seedInput  = sel('[data-input="seed"]');
  const solutionEl = sel("[data-solution]");
  const sampleEl   = sel("[data-sample]");

  // Persisted state — draft text + attempt counter, keyed by hostId.
  const storageKey = `exercise:${spec.hostId}`;
  const state = loadState(storageKey, { code: spec.template ?? "", attempts: 0 });
  textarea.value = state.code;
  updateAttempts(attempts, state.attempts);

  // Editor abstraction — starts as the textarea, async-upgrades to
  // CodeMirror 6 if the CDN load succeeds. The harness reads/writes
  // through `editor.getValue()` / `editor.setValue(...)` so calling
  // code is identical whichever backend is live.
  const editor = createEditor({
    textarea,
    editorHost,
    initial: state.code,
    onChange: (v) => {
      state.code = v;
      saveState(storageKey, state);
    },
  });

  // Lazy-load CodeMirror in the background. If it fails (offline,
  // CDN blocked), the textarea fallback stays in place. Either way
  // the exercise runs.
  editor.upgradeToCodeMirror().catch(() => { /* stays on textarea */ });

  // Expose the editor on the host so smoke tests / DevTools can
  // poke at it ("set this value, click run, read the diff"). Not
  // part of the public authoring surface — same convention as the
  // `__runCheck` re-export from the harness.
  host.__exerciseEditor = editor;

  // Render the example-input disclosure on first paint if the spec
  // supports it. Re-render when the seed changes.
  const refreshSample = () => {
    if (!sampleEl) return;
    const seed = parseInt(seedInput.value, 10) || 0;
    try {
      const sample = deriveSampleInput(spec, seed);
      sampleEl.textContent = formatSampleInput(sample, spec.entrypointArgs);
    } catch (err) {
      sampleEl.textContent = `(could not derive sample: ${err.message})`;
    }
  };
  refreshSample();
  seedInput.addEventListener("input", refreshSample);
  seedInput.addEventListener("change", refreshSample);

  btnRun.addEventListener("click", () => {
    state.attempts += 1;
    const code = editor.getValue();
    state.code = code;
    saveState(storageKey, state);
    updateAttempts(attempts, state.attempts);
    const seed = parseInt(seedInput.value, 10) || 0;
    runAndRender(code, spec, seed, status, result);
  });

  btnReset.addEventListener("click", () => {
    if (!confirm("Reset to the starter template? Your edits will be lost.")) return;
    editor.setValue(spec.template ?? "");
    state.code = spec.template ?? "";
    saveState(storageKey, state);
    result.innerHTML = "";
    status.textContent = "";
  });

  btnReveal.addEventListener("click", () => {
    const hidden = solutionEl.hasAttribute("hidden");
    if (hidden) {
      if (state.attempts < 3) {
        if (!confirm(
          `You've only made ${state.attempts} attempt(s). Are you sure ` +
          `you want to see the solution? Struggling is the point.`,
        )) return;
      }
      solutionEl.removeAttribute("hidden");
      btnReveal.textContent = "▾ Hide solution";
    } else {
      solutionEl.setAttribute("hidden", "");
      btnReveal.textContent = "▸ Show solution";
    }
  });
}

function updateAttempts(el, n) {
  el.textContent = n === 0 ? "" : `attempts: ${n}`;
}

// ─── Editor abstraction (textarea → CodeMirror upgrade) ─────────────
//
// The harness ships a working `<textarea>` synchronously so the
// exercise is usable instantly, then asynchronously upgrades to
// CodeMirror 6 for syntax highlighting + better keybindings if the
// esm.sh CDN is reachable. If the upgrade fails (offline, CSP,
// blocked CDN), the textarea stays — the exercise still works.
//
// Both backends speak the same tiny surface:
//   getValue() : string
//   setValue(s): void
//   focus()    : void
// Calling code in `wire()` doesn't know which is live.

function createEditor({ textarea, editorHost, initial, onChange }) {
  // ── textarea-backed ───────────────────────────────────────────
  textarea.value = initial;
  // Tab key inserts two spaces instead of leaving the editor.
  textarea.addEventListener("keydown", (e) => {
    if (cm6View) return;     // CM6 handles its own indentation
    if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      const { selectionStart: a, selectionEnd: b, value } = textarea;
      textarea.value = value.slice(0, a) + "  " + value.slice(b);
      textarea.selectionStart = textarea.selectionEnd = a + 2;
      onChange(textarea.value);
    }
  });
  textarea.addEventListener("input", () => {
    if (cm6View) return;
    onChange(textarea.value);
  });

  // ── CodeMirror 6 — lazy upgrade ───────────────────────────────
  let cm6View = null;
  async function upgradeToCodeMirror() {
    let cm, jsLang, theme;
    try {
      [cm, jsLang, theme] = await Promise.all([
        import("https://esm.sh/codemirror@6.0.1"),
        import("https://esm.sh/@codemirror/lang-javascript@6.2.2"),
        import("https://esm.sh/@codemirror/theme-one-dark@6.1.2"),
      ]);
    } catch (err) {
      // CDN unreachable / module load failed. Stay on textarea.
      throw err;
    }
    const { EditorView, basicSetup } = cm;
    const update = EditorView.updateListener.of((v) => {
      if (v.docChanged) onChange(cm6View.state.doc.toString());
    });
    cm6View = new EditorView({
      doc: textarea.value,
      extensions: [
        basicSetup,
        jsLang.javascript(),
        theme.oneDark,
        EditorView.theme({
          "&": { fontSize: "0.88em", borderRadius: "3px",
                 border: "1px solid rgba(128,128,128,0.28)" },
          ".cm-content": { fontFamily: "ui-monospace, monospace",
                           padding: "0.55em 0" },
          "&.cm-focused": { outline: "none",
                            borderColor: "#5aa9e6" },
        }),
        update,
      ],
      parent: editorHost,
    });
    // Hide the textarea now that CM is live. Keep it in the DOM —
    // it remains the persistence-backing element if needed.
    textarea.style.display = "none";
  }

  return {
    getValue: () => cm6View ? cm6View.state.doc.toString() : textarea.value,
    setValue: (v) => {
      if (cm6View) {
        cm6View.dispatch({
          changes: { from: 0, to: cm6View.state.doc.length, insert: v },
        });
      }
      textarea.value = v;
      onChange(v);
    },
    focus: () => (cm6View ? cm6View.focus() : textarea.focus()),
    upgradeToCodeMirror,
  };
}

// ─── Sample-input formatter ─────────────────────────────────────────
//
// Render the actual values the harness will pass to test 1 in a form
// the reader can match against their function signature. Typed arrays
// get their constructor name + length; nested arrays/objects get
// indented; numbers get a short fixed-point form so the panel doesn't
// blow up vertically with 16 decimal places.

function formatSampleInput(input, argNames) {
  if (input == null) return "(none)";
  // For positional-args specs, render each arg as `name = value`.
  if (Array.isArray(input)) {
    const names = argNames && argNames.length === input.length
      ? argNames
      : input.map((_, i) => `arg${i}`);
    return input
      .map((v, i) => `${names[i]} = ${formatValue(v, 0)}`)
      .join("\n\n");
  }
  // Single value (object-arg spec, e.g. adamStep({...})).
  return formatValue(input, 0);
}

function formatValue(v, indent) {
  const pad = "  ".repeat(indent);
  if (v === null || v === undefined) return String(v);
  if (typeof v === "number") return formatNumber(v);
  if (typeof v === "string") return JSON.stringify(v);
  if (typeof v === "boolean") return String(v);
  if (v instanceof Float32Array || v instanceof Float64Array) {
    const name = v.constructor.name;
    const arr = Array.from(v);
    return `${name}(${arr.length}) [${formatList(arr)}]`;
  }
  if (Array.isArray(v)) {
    // Vector of numbers — single line.
    if (v.length > 0 && v.every((e) => typeof e === "number")) {
      return `Array(${v.length}) [${formatList(v)}]`;
    }
    // Otherwise — one element per line.
    if (v.length === 0) return "[]";
    const items = v.map((e) => pad + "  " + formatValue(e, indent + 1));
    return `[\n${items.join(",\n")},\n${pad}]`;
  }
  if (typeof v === "object") {
    const keys = Object.keys(v);
    if (keys.length === 0) return "{}";
    // Short, primitive-only objects render on one line — keeps the
    // sample compact for things like `{ reward: 0, nextState: 1 }`.
    const allPrimitive = keys.every((k) => {
      const x = v[k];
      return x === null || x === undefined ||
             typeof x === "number" || typeof x === "string" ||
             typeof x === "boolean";
    });
    if (allPrimitive && keys.length <= 4) {
      return `{ ${keys.map((k) => `${k}: ${formatValue(v[k], 0)}`).join(", ")} }`;
    }
    const items = keys.map((k) => `${pad}  ${k}: ${formatValue(v[k], indent + 1)}`);
    return `{\n${items.join(",\n")},\n${pad}}`;
  }
  return String(v);
}

function formatNumber(n) {
  if (Number.isInteger(n)) return String(n);
  if (Math.abs(n) >= 1000 || (Math.abs(n) < 1e-3 && n !== 0)) {
    return n.toExponential(3);
  }
  return n.toFixed(4);
}

function formatList(arr) {
  // Show up to 8 elements; elide the middle if longer.
  if (arr.length <= 8) return arr.map(formatNumber).join(", ");
  const head = arr.slice(0, 4).map(formatNumber);
  const tail = arr.slice(-4).map(formatNumber);
  return `${head.join(", ")}, …, ${tail.join(", ")}`;
}

function loadState(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return { ...fallback, ...JSON.parse(raw) };
  } catch { return fallback; }
}
function saveState(key, state) {
  try { localStorage.setItem(key, JSON.stringify(state)); } catch { /* quota */ }
}

// ─── Run + render ──────────────────────────────────────────────────

function runAndRender(source, spec, seed, statusEl, resultEl) {
  resultEl.innerHTML = "";
  statusEl.textContent = "";
  let fn;
  try {
    fn = compileEntrypoint(source, spec.entrypoint, spec.entrypointArgs);
  } catch (err) {
    statusEl.textContent = "✗ syntax error";
    statusEl.className = "exercise-status status-fail";
    resultEl.innerHTML = renderError(err);
    return;
  }
  let report;
  try {
    report = runCheck(fn, spec.check, seed);
  } catch (err) {
    statusEl.textContent = "✗ runtime error";
    statusEl.className = "exercise-status status-fail";
    resultEl.innerHTML = renderError(err);
    return;
  }
  if (report.passed) {
    statusEl.textContent = "✓ all checks passed";
    statusEl.className = "exercise-status status-pass";
  } else {
    statusEl.textContent = `✗ ${report.passes}/${report.total} passed`;
    statusEl.className = "exercise-status status-fail";
  }
  resultEl.innerHTML = renderReport(report);
}

/**
 * Compile the student's source into a callable. Extracts the function
 * named `entrypoint` (whether they wrote `function entrypoint(...)`,
 * `const entrypoint = (...) => …`, or `entrypoint = function(...) { … }`).
 *
 * Throws if the entrypoint isn't defined after running their code.
 */
function compileEntrypoint(source, entrypoint, argNames) {
  // Strategy: wrap the student's code in a thunk that returns the
  // named entrypoint after executing. `new Function` constructor
  // catches syntax errors at construct time.
  const wrapper = new Function(
    `"use strict";\n${source}\n;return typeof ${entrypoint} === "function" ? ${entrypoint} : undefined;`,
  );
  const fn = wrapper();
  if (typeof fn !== "function") {
    throw new Error(
      `Could not find function \`${entrypoint}\`. Did you spell it ` +
      `correctly? Make sure it's defined at the top level of your code.`,
    );
  }
  if (argNames && fn.length !== argNames.length) {
    // Soft warning — students can use rest args, ignore extras, etc.
    // Don't fail; the check will catch wrong behaviour.
  }
  return fn;
}

function renderError(err) {
  const msg = err && err.message ? err.message : String(err);
  return `<div class="exercise-error">${escapeHtml(msg)}</div>`;
}

// ─── Check dispatch ────────────────────────────────────────────────

/**
 * Run a check against the student's function. Returns
 * `{ passed, passes, total, cases: [...] }`. Each case is
 * `{ name, passed, detail }` where `detail` is HTML.
 */
function runCheck(fn, check, seed) {
  switch (check.kind) {
    case "fixedTable": return checkFixedTable(fn, check);
    case "gradCheck":  return checkGradCheck(fn, check, seed);
    case "reference":  return checkReference(fn, check, seed);
    default:
      throw new Error(`unknown check kind '${check.kind}'`);
  }
}

// ─── fixedTable: discrete input → expected output ──────────────────

function checkFixedTable(fn, check) {
  const tol = check.tolerance ?? 1e-6;
  const cases = check.cases.map((c, i) => {
    const name = c.name ?? `case ${i + 1}`;
    let got;
    try { got = fn(...(Array.isArray(c.input) ? c.input : [c.input])); }
    catch (err) {
      return { name, passed: false, detail: renderInline(`runtime error: ${err.message}`) };
    }
    const cmp = compareValues(got, c.expected, tol);
    if (cmp.equal) {
      return { name, passed: true, detail: renderInline(`got ${fmtShort(got)}`) };
    }
    return {
      name, passed: false,
      detail: renderDiff(cmp, got, c.expected, tol),
    };
  });
  return tally(cases);
}

// ─── gradCheck: finite-difference numerical derivative ─────────────

function checkGradCheck(fn, check, seed) {
  const rng = mulberry32(seed >>> 0);
  const eps = check.epsilon ?? 1e-4;
  const tol = check.tolerance ?? 1e-3;
  const len = check.inputs?.length ?? 16;
  const range = check.inputs?.range ?? [-3, 3];
  const nTests = check.nTests ?? 5;
  const cases = [];
  for (let t = 0; t < nTests; t++) {
    const x = sampleVector(rng, len, range);
    const y = applyMap(check.forward, x);
    const dy = sampleVector(rng, len, [-1, 1]);
    // Numerical derivative of L = sum(forward(x) * dy_frozen) w.r.t. x.
    // ∂L/∂x_i ≈ (L(x + ε e_i) - L(x - ε e_i)) / (2ε).
    const dxNumeric = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const xPlus = x.slice();   xPlus[i] += eps;
      const xMinus = x.slice();  xMinus[i] -= eps;
      const yPlus  = applyMap(check.forward, xPlus);
      const yMinus = applyMap(check.forward, xMinus);
      let lPlus = 0, lMinus = 0;
      for (let k = 0; k < len; k++) {
        lPlus  += yPlus[k]  * dy[k];
        lMinus += yMinus[k] * dy[k];
      }
      dxNumeric[i] = (lPlus - lMinus) / (2 * eps);
    }
    let dxStudent;
    try { dxStudent = fn(dy, y); }
    catch (err) {
      cases.push({ name: `test ${t + 1}`, passed: false,
        detail: renderInline(`runtime error: ${err.message}`) });
      continue;
    }
    if (!isLengthLike(dxStudent, len)) {
      cases.push({ name: `test ${t + 1}`, passed: false,
        detail: renderInline(
          `expected a Float32Array of length ${len}, got ${describe(dxStudent)}`) });
      continue;
    }
    const cmp = compareVectors(dxStudent, dxNumeric, tol);
    cases.push({
      name: `test ${t + 1}`,
      passed: cmp.equal,
      detail: cmp.equal
        ? renderInline(`max |student − numeric| = ${cmp.maxAbs.toExponential(2)} (tol ${tol})`)
        : renderGradDiff(cmp, dxStudent, dxNumeric, tol),
    });
  }
  return tally(cases);
}

function sampleVector(rng, n, [lo, hi]) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = lo + (hi - lo) * rng();
  return out;
}
function applyMap(fwd, x) {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = fwd(x[i]);
  return y;
}

// ─── reference: compare to an oracle ───────────────────────────────

function checkReference(fn, check, seed) {
  const tol = check.tolerance ?? 1e-5;
  const nTests = check.nTests ?? 5;
  const rng = mulberry32(seed >>> 0);
  const cases = [];
  // The reference callable + the input generator are both author-supplied.
  // We don't await here; if the author wants async (e.g. dynamic import),
  // they pre-resolve before defineExercise.
  const reference = check.reference;
  const generate = check.generate ?? (() => null);
  for (let t = 0; t < nTests; t++) {
    const input = generate(rng, t);
    let studentOut, referenceOut;
    try { studentOut = fn(...input); }
    catch (err) {
      cases.push({ name: `test ${t + 1}`, passed: false,
        detail: renderInline(`runtime error: ${err.message}`) });
      continue;
    }
    try { referenceOut = reference(...input); }
    catch (err) {
      cases.push({ name: `test ${t + 1}`, passed: false,
        detail: renderInline(`reference error (bug in exercise): ${err.message}`) });
      continue;
    }
    const cmp = compareValues(studentOut, referenceOut, tol);
    cases.push({
      name: `test ${t + 1}`,
      passed: cmp.equal,
      detail: cmp.equal
        ? renderInline(`matches reference (max diff ${cmp.maxAbs.toExponential(2)})`)
        : renderDiff(cmp, studentOut, referenceOut, tol),
    });
  }
  return tally(cases);
}

// ─── Comparison primitives ─────────────────────────────────────────

function compareValues(got, expected, tol) {
  // Numbers.
  if (typeof expected === "number") {
    if (typeof got !== "number") {
      return { equal: false, kind: "type", got, expected, maxAbs: Infinity };
    }
    const d = Math.abs(got - expected);
    return { equal: d <= tol, kind: "scalar", maxAbs: d, got, expected };
  }
  // Vectors (Float32Array / Array of numbers).
  if (isVector(expected)) {
    if (!isVector(got) || got.length !== expected.length) {
      return { equal: false, kind: "shape", got, expected, maxAbs: Infinity };
    }
    return compareVectors(got, expected, tol);
  }
  // Matrices (Array of Array of numbers).
  if (isMatrix(expected)) {
    if (!isMatrix(got) || got.length !== expected.length ||
        got[0].length !== expected[0].length) {
      return { equal: false, kind: "shape", got, expected, maxAbs: Infinity };
    }
    let maxAbs = 0, maxAt = [0, 0];
    for (let r = 0; r < expected.length; r++) {
      for (let c = 0; c < expected[0].length; c++) {
        const d = Math.abs(got[r][c] - expected[r][c]);
        if (d > maxAbs) { maxAbs = d; maxAt = [r, c]; }
      }
    }
    return { equal: maxAbs <= tol, kind: "matrix", maxAbs, maxAt };
  }
  // Plain objects — recurse into matching keys. The student returns
  // e.g. `{ params, m, v }`; compare each field independently and
  // surface the worst.
  if (isPlainObject(expected)) {
    if (!isPlainObject(got)) {
      return { equal: false, kind: "type", got, expected, maxAbs: Infinity };
    }
    let worst = { equal: true, kind: "object", maxAbs: 0, key: null };
    for (const k of Object.keys(expected)) {
      const sub = compareValues(got[k], expected[k], tol);
      if (sub.maxAbs > worst.maxAbs) {
        worst = { ...sub, kind: "object", maxAbs: sub.maxAbs, key: k, inner: sub };
      }
      if (!sub.equal) worst.equal = false;
    }
    return worst;
  }
  // Fallback — strict equality.
  return { equal: got === expected, kind: "strict", got, expected, maxAbs: NaN };
}

function compareVectors(got, expected, tol) {
  let maxAbs = 0, maxAt = -1;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(got[i] - expected[i]);
    if (d > maxAbs) { maxAbs = d; maxAt = i; }
  }
  return { equal: maxAbs <= tol, kind: "vector", maxAbs, maxAt };
}

function isVector(v) {
  return v instanceof Float32Array || v instanceof Float64Array ||
         (Array.isArray(v) && v.every((e) => typeof e === "number"));
}
function isMatrix(v) {
  return Array.isArray(v) && v.length > 0 && Array.isArray(v[0]) &&
         v[0].every((e) => typeof e === "number");
}
function isLengthLike(v, n) {
  return v && typeof v.length === "number" && v.length === n;
}
function isPlainObject(v) {
  return v != null && typeof v === "object" && !Array.isArray(v) &&
         !(v instanceof Float32Array) && !(v instanceof Float64Array);
}
function describe(v) {
  if (v == null) return String(v);
  if (Array.isArray(v)) return `Array(${v.length})`;
  if (v instanceof Float32Array) return `Float32Array(${v.length})`;
  if (v instanceof Float64Array) return `Float64Array(${v.length})`;
  return typeof v;
}

// ─── Pretty-printing ───────────────────────────────────────────────

function fmtShort(v) {
  if (typeof v === "number") return v.toFixed(4);
  if (isVector(v)) {
    const arr = Array.from(v);
    const head = arr.slice(0, 4).map((x) => x.toFixed(3));
    return `[${head.join(", ")}${arr.length > 4 ? ", …" : ""}]`;
  }
  if (isMatrix(v)) return `Matrix(${v.length}×${v[0].length})`;
  return String(v);
}

function renderInline(text) {
  return `<span class="exercise-detail">${escapeHtml(text)}</span>`;
}

function renderDiff(cmp, got, expected, tol) {
  if (cmp.kind === "vector") {
    return renderGradDiff(cmp, got, expected, tol);
  }
  if (cmp.kind === "scalar") {
    return renderInline(
      `your ${fmtShort(got)}, expected ${fmtShort(expected)} ` +
      `(diff ${cmp.maxAbs.toExponential(2)} > tol ${tol})`,
    );
  }
  if (cmp.kind === "shape" || cmp.kind === "type") {
    return renderInline(`shape/type mismatch: got ${describe(got)}, expected ${describe(expected)}`);
  }
  if (cmp.kind === "matrix") {
    const [r, c] = cmp.maxAt;
    return renderInline(
      `at [${r}][${c}]: your ${got[r][c].toFixed(4)}, expected ` +
      `${expected[r][c].toFixed(4)} (diff ${cmp.maxAbs.toExponential(2)} > tol ${tol})`,
    );
  }
  if (cmp.kind === "object") {
    const key = cmp.key ?? "?";
    return renderInline(
      `field \`${key}\`: max diff ${cmp.maxAbs.toExponential(2)} > tol ${tol}`,
    );
  }
  return renderInline(`mismatch (kind: ${cmp.kind})`);
}

function renderGradDiff(cmp, got, expected, tol) {
  const i = cmp.maxAt;
  return renderInline(
    `worst at i=${i}: your ${got[i].toFixed(4)}, expected ${expected[i].toFixed(4)} ` +
    `(diff ${cmp.maxAbs.toExponential(2)} > tol ${tol})`,
  );
}

function tally(cases) {
  const passes = cases.filter((c) => c.passed).length;
  return { passed: passes === cases.length, passes, total: cases.length, cases };
}

function renderReport(report) {
  const rows = report.cases.map((c) => {
    const icon = c.passed ? "✓" : "✗";
    const cls = c.passed ? "case-pass" : "case-fail";
    return `<div class="exercise-case ${cls}">
      <span class="case-icon">${icon}</span>
      <span class="case-name">${escapeHtml(c.name)}</span>
      ${c.detail}
    </div>`;
  }).join("");
  return rows;
}

// ─── Re-exported helpers for exercises that want them ──────────────

export { splitmix64, mulberry32 };

// Re-exported for the smoketest harness (`smoketest.js`). Not part of
// the public exercise-authoring surface — the leading underscore is
// the convention "internal, but available for testing."
export { runCheck as __runCheck };

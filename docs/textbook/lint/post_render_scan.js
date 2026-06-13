// Post-MathJax rendered-HTML scanner.
//
// The Python validator (find_em_in_math.py) checks the STATIC build
// output. It catches the cases where source-level markup leaks into
// HTML (`<em>` inside math, smart-quotes in math, math in attributes).
//
// This scanner is COMPLEMENTARY: it loads each chapter in an iframe,
// waits for MathJax to typeset client-side, then walks the rendered
// DOM looking for surviving `$X$` patterns — anything MathJax COULDN'T
// process. Under the recommended runtime-MathJax setup, the static
// HTML always contains raw `$...$` markers (that's what feeds MathJax),
// so this kind of check is impossible without actually running the
// browser's JavaScript.
//
// USAGE (in an interactive browser session):
//
// 1. Start mdBook serve in another shell:
//        mdbook serve docs/textbook --port 3344
//
// 2. Open http://localhost:3344/ in a browser, then open the DevTools
//    console.
//
// 3. Paste the IIFE below into the console. It iterates every chapter,
//    loads each in a hidden iframe, calls MathJax.typesetPromise(),
//    then scans for surviving `$X$` patterns plus a handful of other
//    rendering red flags (HTML entity leaks, unrendered code fences /
//    headings / mermaid sources, empty widget containers).
//
// 4. The Promise resolves with an object keyed by chapter. Empty
//    object = clean run. Any chapter with `findings` warrants a look.
//
// HEURISTIC NOTES:
//
// - `unrendered_heading` produces FALSE POSITIVES when body text or a
//   table cell starts with `# Variants` (count column) or similar.
//   Cross-check the source markdown when this fires.
// - `unrendered_markdown_link` similarly false-positives on code
//   snippets that *display* a markdown-style link as an example.
// - Genuine bugs would be: any `$X$` or `$$X$$` survivor in body text;
//   `unrendered_code_fence` (means a triple-backtick leaked); a widget
//   container with controls but no SVG/table (mount failed).

(async () => {
  const chapters = [
    "00_index.html",
    "01_linear_algebra.html",
    "02_probability_and_statistics.html",
    "03_mathematics_for_ai.html",
    "04_the_rl_problem.html",
    "05_mdps_and_bellman_equations.html",
    "06_dynamic_programming.html",
    "07_monte_carlo_methods.html",
    "08_temporal_difference_learning.html",
    "09_eligibility_traces.html",
    "10_function_approximation.html",
    "11_deep_q_learning.html",
    "12_policy_gradient.html",
    "13_actor_critic.html",
    "14_exploration.html",
    "15_model_based_rl.html",
    "16_hierarchical_rl.html",
    "17_fa_pathologies.html",
    "18_homeostatic_rl.html",
    "19_long_horizon_credit.html",
    "20_action_spaces.html",
  ];

  const SKIP_TAGS = new Set([
    "script", "style", "code", "pre", "noscript", "textarea", "mjx-container",
  ]);

  const scanOne = async (path) => new Promise((resolve) => {
    const iframe = document.createElement("iframe");
    iframe.style.cssText = "position:fixed;left:-9999px;width:1200px;height:800px;visibility:hidden;";
    iframe.src = "/" + path;
    let resolved = false;
    const finish = (res) => {
      if (resolved) return;
      resolved = true;
      document.body.removeChild(iframe);
      resolve(res);
    };
    iframe.onload = async () => {
      const doc = iframe.contentDocument;
      const win = iframe.contentWindow;
      // Wait up to 6 s for MathJax to attach and finish a typeset pass.
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 100));
        if (win.MathJax && win.MathJax.typesetPromise) {
          try { await win.MathJax.typesetPromise(); } catch (e) { /* swallow */ }
          break;
        }
      }
      await new Promise((r) => setTimeout(r, 600));

      const main = doc.querySelector("main") || doc.body;
      const findings = [];
      const walker = doc.createTreeWalker(main, NodeFilter.SHOW_TEXT, null);
      const PAT_D = /\$\$[A-Za-z\\^_({\[][^$\n]{0,200}?\$\$/g;
      const PAT_I = /(?<!\$)\$([A-Za-z\\^_({\[][^$\n<>]{0,80}?)\$(?!\$)/g;
      let node;
      while ((node = walker.nextNode())) {
        // Skip if any ancestor is a safe tag.
        let p = node.parentNode;
        let skip = false;
        while (p && p !== main) {
          if (SKIP_TAGS.has(p.tagName?.toLowerCase())) { skip = true; break; }
          p = p.parentNode;
        }
        if (skip) continue;
        const text = node.nodeValue;
        if (!text || !text.trim()) continue;
        let m;
        while ((m = PAT_D.exec(text))) findings.push({ kind: "display_math_survivor", snippet: m[0].slice(0, 120) });
        while ((m = PAT_I.exec(text))) findings.push({ kind: "inline_math_survivor", snippet: m[0].slice(0, 80) });
        if (/&(amp|lt|gt|quot|apos|nbsp);/.test(text)) findings.push({ kind: "html_entity_leak", sample: text.slice(0, 80).trim() });
        if (/\*\*[A-Za-z][^*\n]{1,50}\*\*/.test(text)) findings.push({ kind: "unrendered_bold", sample: text.match(/\*\*[A-Za-z][^*\n]{1,50}\*\*/)[0] });
        if (/^```/.test(text.trim())) findings.push({ kind: "unrendered_code_fence", sample: text.slice(0, 80).trim() });
        if (/^(graph TD|flowchart|sequenceDiagram|stateDiagram)/.test(text.trim())) findings.push({ kind: "unrendered_mermaid", sample: text.slice(0, 80).trim() });
        if (findings.length > 30) break;
      }
      // Widget mounts that look empty.
      for (const w of main.querySelectorAll(".textbook-widget")) {
        const svgs = w.querySelectorAll("svg");
        const tables = w.querySelectorAll("table");
        if (svgs.length === 0 && tables.length === 0 && w.children.length > 0) {
          const controls = w.querySelectorAll("input, select, button").length;
          if (controls > 0) findings.push({ kind: "widget_no_visual", id: w.id });
        }
      }
      finish({ chapter: path, findings: findings.slice(0, 25) });
    };
    document.body.appendChild(iframe);
    setTimeout(() => finish({ chapter: path, timeout: true }), 12000);
  });

  const out = {};
  for (const ch of chapters) {
    const r = await scanOne(ch);
    if ((r.findings && r.findings.length) || r.timeout) out[ch] = r;
  }
  console.log(JSON.stringify(out, null, 2));
  return out;
})();

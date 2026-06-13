#!/usr/bin/env node
//
// Textbook LaTeX pre-build syntax check.
//
// The textbook targets github.io (mdBook output) only — github.com's
// raw-markdown rendering is no longer a supported surface. So this
// validator is now a fast PRE-BUILD syntax check: it catches LaTeX
// errors before mdBook silently produces broken output.
//
// For the actual rendered-output verification, see
// `find_em_in_math.py`, which scans the BUILT HTML for pulldown-cmark's
// inline-emphasis bug + smart-quote contamination. That script is what
// catches "math renders raw" issues; this one catches "LaTeX won't
// parse at all."
//
// Two layers, both lightweight:
//
//   1. PARSE — every `$…$` / `$$…$$` block is rendered through KaTeX
//      in strict mode. Catches macro typos, `\\left` corruption,
//      unbraced multi-char super/subscripts (e.g. `R_\max` needs
//      `R_{\max}`).
//
//   2. CONTEXT — regex rules for source-level patterns that confuse
//      pulldown-cmark + MathJax. Most are not GitHub-specific —
//      pulldown-cmark and CommonMark-family parsers share these
//      failure modes:
//        - display-math-in-blockquote (parser doesn't recognise math)
//        - display-math-leading-indent (4-space indent = code block)
//        - display-math-on-continuation-line (needs surrounding blanks)
//        - pmatrix-in-inline-math (`\\` inside inline math gets eaten)
//        - inline-math-spans-newline (odd `$` count on a line)
//
// Exit code 0 if both layers pass; non-zero with a file:line list.

'use strict';

const fs = require('fs');
const path = require('path');
const katex = require('katex');

const TEXTBOOK_DIR = path.resolve(__dirname, '..');
const STRICT = process.argv.includes('--strict');

// ─── extraction ──────────────────────────────────────────────────────
function stripNonMath(text) {
  const blankFn = (m) => '\n'.repeat(m.split('\n').length - 1);
  return text
    .replace(/```[\s\S]*?```/g, blankFn)
    .replace(/<!--[\s\S]*?-->/g, blankFn)
    .replace(/`[^`\n]*`/g, (m) => ' '.repeat(m.length));
}

function lineOfOffset(text, offset) {
  let line = 1;
  for (let i = 0; i < offset; i++) if (text.charCodeAt(i) === 10) line++;
  return line;
}

function extractMath(rawText) {
  const text = stripNonMath(rawText);
  const blocks = [];
  const displayMarks = [];
  const displayRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  while ((m = displayRe.exec(text)) !== null) {
    blocks.push({
      kind: 'display',
      math: m[1],
      line: lineOfOffset(text, m.index),
    });
    displayMarks.push([m.index, m.index + m[0].length]);
  }
  let scratch = text.split('');
  for (const [a, b] of displayMarks) {
    for (let i = a; i < b; i++) if (scratch[i] !== '\n') scratch[i] = ' ';
  }
  scratch = scratch.join('');
  const inlineRe = /(?<![\\$])\$([^\$\n]+?)\$(?!\$)/g;
  while ((m = inlineRe.exec(scratch)) !== null) {
    blocks.push({
      kind: 'inline',
      math: m[1],
      line: lineOfOffset(text, m.index),
    });
  }
  blocks.sort((a, b) => a.line - b.line);
  return blocks;
}

// ─── layer 1: KaTeX parse ────────────────────────────────────────────
function checkParseLayer(blocks) {
  const failures = [];
  for (const b of blocks) {
    try {
      katex.renderToString(b.math, {
        strict: STRICT ? 'error' : 'warn',
        throwOnError: true,
        displayMode: b.kind === 'display',
        trust: false,
      });
    } catch (e) {
      failures.push({
        line: b.line,
        rule: 'parse:' + b.kind,
        msg: e.message.split('\n')[0],
        snippet: b.math.replace(/\s+/g, ' ').trim().slice(0, 90),
      });
    }
  }
  return failures;
}

// ─── layer 2: pulldown-cmark + MathJax context checks ────────────────
//
// Rules below catch patterns that pass KaTeX in isolation but cause
// real rendering bugs in mdBook's pipeline (pulldown-cmark + MathJax
// 3). Most are CommonMark-family-wide failure modes — they were
// originally calibrated against github.com but apply equally to
// pulldown-cmark.
//
// The `opener-without-leading-space` rule (was GitHub-specific) was
// dropped 2026-05-21 when the textbook stopped targeting github.com:
// MathJax processes `$…$` patterns by client-side text scan and
// doesn't care about the preceding character.
const CONTEXT_RULES = [
  {
    name: 'display-math-in-blockquote',
    explain: 'Single-line `> $$…$$` display math inside a blockquote. The markdown parser does not recognise this as math; the `$$` survives as raw text. Move the math outside the blockquote, OR split across multiple `>` lines.',
    test: (line) => /^>\s.*\$\$.*\$\$\s*$/.test(line),
  },
  {
    name: 'display-math-leading-indent',
    explain: 'Display math `$$…$$` preceded by 4+ spaces of indent. The parser treats indented blocks as code, so the `$$` survives.',
    test: (line) => /^ {4,}\$\$.*\$\$\s*$/.test(line),
  },
  {
    name: 'display-math-on-continuation-line',
    explain: 'Display math `$$…$$` immediately follows a non-blank line. Display math needs its own paragraph. Insert a blank line before (and after, if not present).',
    test: (line, lineIdx, lines) => {
      // Only fire on lines that are a complete $$..$$ block.
      if (!/^\s*\$\$[^\n]*\$\$\s*$/.test(line)) return false;
      // Or that open a $$..$$ block on this line.
      // The rule: line above must be blank or non-existent.
      if (lineIdx === 0) return false;
      const prev = lines[lineIdx - 1];
      return prev.trim() !== '' && !prev.trim().endsWith('$$');
    },
  },
  {
    name: 'pmatrix-in-inline-math',
    explain: 'Inline `$…\\begin{pmatrix}…\\\\…\\end{pmatrix}…$` uses `\\\\` (newline) inside inline math. Markdown can eat the `\\\\` before MathJax sees it, breaking the matrix. Convert to display math `$$…$$` on its own paragraph.',
    test: (line) => /\$[^$\n]*\\begin\{(?:p|b|v|V|B|small)?matrix\}/.test(line) && !line.trim().startsWith('$$'),
  },
  {
    name: 'inline-math-spans-newline',
    explain: 'A single `$…$` math block opens on this line but does not close — likely wraps across a newline. Inline math must stay on one line. (Plain currency mentions like `$10 each` are filtered out: a line with an odd number of `$` but no LaTeX command is treated as text, not math.)',
    test: (line) => {
      // Skip lines that are obviously a display-math fence on its own.
      if (/^\s*\$\$\s*$/.test(line)) return false;
      // Skip lines containing $$ display math (block-level — counted separately).
      if (line.includes('$$')) return false;
      // Count unescaped `$` on the line.
      const dollars = (line.match(/(?<!\\)\$/g) || []).length;
      if (dollars % 2 === 0) return false;
      // Odd count: either inline math wrapping OR a stray `$` (currency,
      // S&B notation, etc.). Treat as wrap only if the line contains a
      // LaTeX command — currency mentions like `$10` won't.
      return /\\[a-zA-Z]+/.test(line);
    },
  },
];

function checkContextLayer(rawText) {
  const failures = [];
  const lines = rawText.split('\n');

  // We only want to run context rules outside fenced code blocks /
  // inline code spans. Track fence state.
  let inFence = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;

    for (const rule of CONTEXT_RULES) {
      if (rule.test(line, i, lines)) {
        failures.push({
          line: i + 1,
          rule: 'context:' + rule.name,
          msg: rule.explain,
          snippet: line.trim().slice(0, 90),
        });
      }
    }
  }
  return failures;
}

// ─── main ────────────────────────────────────────────────────────────
//
// The dollar-survival render check that used to live here (rendered
// the source through markdown-it+texmath and grepped for surviving
// `$`) was removed 2026-05-21 when the textbook stopped targeting
// github.com. The actual render-time check is `find_em_in_math.py`,
// which scans the built mdBook HTML — exactly what gets deployed to
// github.io.

function check(relFile) {
  const fullPath = path.join(TEXTBOOK_DIR, relFile);
  const text = fs.readFileSync(fullPath, 'utf8');
  const blocks = extractMath(text);
  return {
    file: relFile,
    blockCount: blocks.length,
    failures: [
      ...checkParseLayer(blocks),
      ...checkContextLayer(text),
    ],
  };
}

const files = fs.readdirSync(TEXTBOOK_DIR)
  .filter((f) => f.endsWith('.md'))
  .sort();

let totalBlocks = 0;
let totalFailures = 0;
for (const f of files) {
  const r = check(f);
  totalBlocks += r.blockCount;
  totalFailures += r.failures.length;
  for (const fail of r.failures) {
    console.error(`${r.file}:${fail.line}  [${fail.rule}]  ${fail.msg}`);
    if (fail.snippet) console.error(`    ${fail.snippet}`);
  }
}

if (totalFailures === 0) {
  console.log(`OK  ${totalBlocks} math blocks pass parse + context checks across ${files.length} files.`);
  process.exit(0);
} else {
  console.error(`\nFAIL  ${totalFailures} failure(s) across ${files.length} files.`);
  process.exit(1);
}

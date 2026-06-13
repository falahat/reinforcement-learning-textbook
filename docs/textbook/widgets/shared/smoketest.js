// Smoke test for the exercise harness — checks that every exercise's
// own `solution` string passes its own `check` block. Catches typos
// in expected values, transposed test matrices, off-by-one errors in
// the author's reference implementation, etc., before they reach a
// reader.
//
// Run from this directory:
//
//     node smoketest.js
//
// Stand-alone — no test framework dependency.

import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const widgetsDir = path.resolve(__dirname, "..");

// Install a global hook BEFORE any exercise.js loads. defineExercise()
// publishes its spec here instead of trying to mount on a missing DOM.
const collected = [];
globalThis.__collectExercise = (spec) => collected.push(spec);

// Pull in the harness so we can use its internal runCheck.
const harness = await import("./exercise.js");

// Discover and import every exercise file.
const exerciseFiles = fs.readdirSync(widgetsDir, { withFileTypes: true })
  .filter((d) => d.isDirectory() && d.name !== "shared")
  .map((d) => path.join(widgetsDir, d.name, "exercise.js"))
  .filter((p) => fs.existsSync(p));

if (exerciseFiles.length === 0) {
  console.error("smoketest: no exercise.js files found under widgets/");
  process.exit(1);
}

for (const file of exerciseFiles) {
  await import(url.pathToFileURL(file).href);
}

if (collected.length !== exerciseFiles.length) {
  console.error(
    `smoketest: collected ${collected.length} specs from ${exerciseFiles.length} files`,
  );
  process.exit(1);
}

// Check each spec's solution against its own check block.
let failures = 0;
for (const spec of collected) {
  if (!spec.solution) {
    console.log(`SKIP  ${spec.hostId}: no solution to verify`);
    continue;
  }
  const fn = new Function(
    `"use strict";\n${spec.solution}\n;return ${spec.entrypoint};`,
  )();
  if (typeof fn !== "function") {
    console.error(`FAIL  ${spec.hostId}: solution did not define '${spec.entrypoint}'`);
    failures++;
    continue;
  }
  const report = harness.__runCheck(fn, spec.check, spec.check.seed ?? 7);
  if (report.passed) {
    console.log(`PASS  ${spec.hostId}  (${report.passes}/${report.total})`);
  } else {
    console.error(`FAIL  ${spec.hostId}  (${report.passes}/${report.total})`);
    for (const c of report.cases) {
      if (!c.passed) console.error(`        ${c.name}: ${stripHtml(c.detail)}`);
    }
    failures++;
  }
}

console.log();
if (failures > 0) {
  console.error(`smoketest: ${failures} exercise(s) failed self-check`);
  process.exit(1);
}
console.log(`smoketest: all ${collected.length} exercises pass their own check`);

function stripHtml(s) { return String(s).replace(/<[^>]+>/g, "").trim(); }

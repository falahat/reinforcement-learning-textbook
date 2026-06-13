# Phase 4 — Widget architecture: TypeScript vs Rust+WASM

The question that comes up every chapter: "Should this widget be
TS or Rust?" Decision tree below. Get it right per widget; don't
force a single-language story.

## The rule

**TypeScript for math viz. Rust+WASM for project-code reuse.**

| Widget category | Language | Why |
|---|---|---|
| Pure math visualization (norms, plots, geometric demos, animations) | TypeScript + Observable Plot + D3 | ~30-150 LOC, no build step, ergonomic authoring |
| Reuses the project's actual algorithm code | Rust + wasm-bindgen | No re-implementation, no algorithm drift |
| Interactive geometry (drag-this-point-on-a-curve) | TypeScript (optionally JSXGraph) | Mature ecosystem |
| Live training over a project env | Rust + wasm-bindgen | Algorithm + env are the project's; reuse |

## When to use Rust+WASM

Only when the widget needs to call a specific function from the
project's source. Examples:

- The bandit widget compiles `playground::run_episode` over
  `MultiArmedBandit`. The episode loop is what the focused tests
  use; running it in the browser proves the textbook claim matches
  the implementation. Re-implementing in TS would drift.
- The gridworld widget compiles `playground::run_episode` over
  `GridWorld`. Same reason.
- A future "Simulator-in-a-page" widget would compile a stripped
  subset of the actual simulator. Pure TS would mean re-implementing
  the entire RL stack.

When to use TS instead:

- Plotting `γ^k` vs k. No algorithm reuse — it's just `Math.pow`.
- Visualizing a unit ball under various norms. Pure math.
- Showing a contraction mapping cobweb. Pure iteration of a
  user-input formula.
- Drag-this-point widgets where the math is closed-form.
- Step-by-step animations of derivations (matrix multiplication
  walkthrough, Bellman backup explainer, summation accumulator).

If you find yourself writing more than ~50 lines of math in Rust
that you'd otherwise write in TS — and you're not calling any
project crate — switch to TS. The build-step overhead isn't worth
it.

## When MIXED — TS UI + thin Rust WASM module

Sometimes you want both: a TS UI for plotting + sliders + DOM
glue, but a thin Rust module that exposes ONE function from the
project. E.g., a widget that lets the reader experiment with the
project's actual TileCoder, but the visualization (active-tiles
heatmap, slider for tile_width) is TS.

Pattern:

```
widgets/<name>/
├── Cargo.toml             # cdylib + wasm-bindgen + path dep on project crate
└── src/lib.rs             # `#[wasm_bindgen] pub fn coder_active_tiles(...)`

docs/textbook/widgets/<name>/
├── pkg/                   # wasm-pack output — COMMITTED
└── widget.js              # TS UI; imports pkg/<name>.js + calls Rust fns
```

The TS widget.js looks like:

```js
import * as Plot from "@observablehq/plot";
import { defineWidget } from "../shared/widget.js";
import init, { coder_active_tiles } from "./pkg/widget_X.js";

await init();

defineWidget({
  hostId: "ch8-tile-coding-widget",
  controls: { tileWidth: { label: "tile width", min: 0.05, max: 0.5, step: 0.01, default: 0.25 } },
  render: (host, { tileWidth }, slots) => {
    const tiles = coder_active_tiles(0.5, 0.5, tileWidth); // call Rust
    slots.main.replaceChildren(Plot.plot({ marks: [/* draw `tiles` */] }));
  },
});
```

The Rust side exposes only the algorithm primitive. TS owns the UI.

## Widget directory layout

Always:

```
widgets/                          # AT REPO ROOT, not under docs/textbook/
└── <name>/                       # Rust widgets only — Cargo workspace
    ├── Cargo.toml
    └── src/lib.rs

docs/textbook/widgets/
├── shared/
│   ├── widget.js                 # defineWidget scaffold
│   ├── stepper.js
│   ├── helpers.js
│   ├── widgets.css
│   └── README.md
├── <name>/widget.js              # TS widget
└── <name>/pkg/                   # Built Rust WASM artifacts — COMMITTED
```

Rust widgets keep source OUTSIDE `docs/textbook/` because mdBook
copies its source tree verbatim — we don't want `target/` or
`Cargo.lock` ending up in the built `book/` directory. Only the
`pkg/` artifacts go inside.

Pure TS widgets live entirely in `docs/textbook/widgets/<name>/widget.js`.
No Rust source dir.

## Rust widget Cargo.toml shape

```toml
# Each widget is a standalone Cargo workspace so its wasm32 build
# cache stays separate from the simulator's native cache.
[workspace]

[package]
name = "widget-<name>"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
crate-type = ["cdylib"]
doctest = false

[dependencies]
# `default-features = false` is critical — the standalone-workspace
# pattern doesn't inherit the main workspace's Bevy-free default.
playground = { path = "../../crates/engine/playground", default-features = false }
rl_core = { path = "../../crates/engine/rl_core", default-features = false }
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
  "Document", "Element", "HtmlElement", "Window", "console",
] }
```

Build via:

```sh
cd widgets/<name>
wasm-pack build --target web --release --out-dir ../../docs/textbook/widgets/<name>/pkg
rm -f ../../docs/textbook/widgets/<name>/pkg/.gitignore
```

The `rm` is required — wasm-pack creates a `pkg/.gitignore` that
excludes everything. We commit `pkg/` so CI doesn't need wasm-pack.

## CI complexity trade-off

Including Rust widgets means CI either:
- Installs `wasm-pack`, `wasm-bindgen-cli`, the wasm32 target, and
  rebuilds widgets every time → adds ~2-3 min of CI per build, OR
- Trusts committed `pkg/` blobs → fast CI but humans must remember
  to rebuild + commit when source changes.

Our project chose option 2. Trade-off: ~250 KB committed binary
blobs (3 widgets × ~85 KB), but CI stays at ~30 seconds (just
mdbook + lint).

If you have many Rust widgets (say 10+), option 1 becomes more
attractive. Set up the wasm32 target install + caching in the
workflow; rebuild widgets in CI.

## Honest size accounting (measured)

For the source project, all 3 Rust widgets shipped at:

| Widget | What it links | Raw wasm | Gzipped |
|---|---|---|---|
| bandit | playground + MultiArmedBandit + run_episode | 85 KB | 30 KB |
| gridworld | playground + GridWorld + run_episode | 83 KB | 30 KB |
| tile_coding | rl_core::TileCoder | 84 KB | 30 KB |

The estimate in the original proposal was "~200 KB compiled WASM"
for rl_core, "+500 KB" for q_learning. Actuals: 7-30× smaller —
tile coding is a few hundred lines of arithmetic and the linker
prunes everything else. Don't over-budget.

## Output of this phase

A decision per widget: Rust+WASM, TS, or mixed. If TS, go to
`references/07-widget-authoring.md`. If Rust, follow the cdylib +
wasm-pack pattern above, then call into Rust from a TS widget.js.

Either way, every widget ends up as one of:

- `docs/textbook/widgets/<name>/widget.js` (TS), or
- `widgets/<name>/Cargo.toml` + `src/lib.rs` (Rust source) +
  `docs/textbook/widgets/<name>/pkg/` (built WASM artifacts) +
  `docs/textbook/widgets/<name>/widget.js` (TS UI calling the WASM).

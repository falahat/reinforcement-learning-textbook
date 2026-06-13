# Textbook interactive widgets

Interactive demos embedded inline in the textbook chapters. Widgets
pick their language by what they reuse:

- **TypeScript + Observable Plot + D3** — pure math visualisations.
  No build step, ~30–80 lines per widget, authors plain `.js` files
  that import from a site-wide importmap. Source files live under
  [`docs/textbook/widgets/<name>/widget.js`](../docs/textbook/widgets/).
  This is the path for the **bulk** of widgets.
- **Rust + wasm-bindgen** — only when the widget needs to reuse the
  project's actual RL algorithms (`playground::run_episode`,
  `rl_core::TileCoder`, etc.). Source crates live in this
  `widgets/<name>/` directory; built `pkg/` artefacts under
  `docs/textbook/widgets/<name>/pkg/`.

The Rust path is the more involved one (wasm-pack build, committed
`.wasm` blob); read on for it. The TS path is documented at
[`docs/textbook/widgets/shared/`](../docs/textbook/widgets/shared/) —
in short, drop a `widget.js` next to similar widgets and use the
`defineWidget` scaffold from `shared/widget.js`.

## Layout

```
widgets/                          ← Rust source for each widget
├── bandit/                       ← Cargo workspace root per widget
│   ├── Cargo.toml                ← cdylib + wasm-bindgen + playground
│   └── src/lib.rs                ← `#[wasm_bindgen] pub fn start(target: &str)`
├── gridworld/
└── tile_coding/

docs/textbook/widgets/<name>/pkg/  ← `wasm-pack` output, COMMITTED
├── widget_<name>.js              ← JS bindings (ES module)
└── widget_<name>_bg.wasm         ← compiled WASM
```

The source crates are deliberately *outside* `docs/textbook/` because
mdBook copies everything under its source root into the book output —
we don't want `target/` or `Cargo.lock` in the deployed site. Only the
`pkg/` artifacts live inside `docs/textbook/`.

## Build pipeline

Each widget is its own standalone Cargo workspace (`[workspace]` at
the top of its `Cargo.toml`) so it doesn't get pulled into the main
simulator workspace's build cache. Each `Cargo.toml` declares its
`playground` / `rl_core` path deps with `default-features = false` so
no `bevy_ecs` is dragged in.

### Building one widget

From the widget's source directory (`widgets/<name>/`):

```sh
wasm-pack build --target web --release --out-dir ../../docs/textbook/widgets/<name>/pkg
rm -f ../../docs/textbook/widgets/<name>/pkg/.gitignore
```

The `--out-dir` puts the build artifacts directly where mdBook can
pick them up. `wasm-pack` auto-creates a `.gitignore` inside `pkg/`
that excludes everything; we delete it so the artifacts get committed.

### Building all widgets

```sh
./widgets/build_all.sh    # TODO once we have more than one widget
```

### Prerequisites

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked --version 0.13.1 wasm-pack
```

`wasm-pack 0.13.1` is the last version that supports the workspace's
pinned rustc (1.90); newer versions require 1.91+.

## Embedding a widget in a chapter

In the markdown source (e.g. `docs/textbook/12_exploration.md`):

```markdown
<div id="ch12-bandit-widget"></div>
<script type="module">
  import init, { start } from './widgets/bandit/pkg/widget_bandit.js';
  await init();
  start('ch12-bandit-widget');
</script>
```

The `<div>` is the mount point; `start()` populates it. mdBook passes
raw HTML through, so the `<script type="module">` block reaches the
browser unchanged.

## Why commit the `pkg/` artifacts (for now)

CI installs mdbook + mdbook-mermaid; adding wasm-pack + the wasm32
target adds ~1 min of cold install + another minute per widget rebuild.
For now, committing the artifacts keeps CI scoped to just the book
build. A future move to CI-side wasm-pack with caching is reasonable
once widget count grows.

The `pkg/` binary blobs are small — ~85 KB raw, ~30 KB gzipped per
widget. The git repo overhead is real but not painful.

## Architecture choice: Rust owns the algorithm AND the DOM

Each widget's Rust code mounts its own HTML (sliders, canvas, buttons)
into the host `<div>` and binds its own event handlers via `web-sys`.
JS only:

1. Imports the `.js` glue file `wasm-pack` generated.
2. Awaits `init()` to load the WASM.
3. Calls `start(target_id)`.

This keeps the JS surface minimal — a couple of lines per chapter —
and lets the same Rust file own both the simulation step and the
visualisation. It does mean each widget pulls in `web-sys` features
for the DOM types it uses, which adds ~10-15 KB to the WASM bundle
over a pure-algorithm crate.

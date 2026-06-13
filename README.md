# Reinforcement Learning & Foundational AI — an interactive textbook

A self-study, master's-level RL textbook with ~140 interactive widgets and
live-coding exercises, built with [mdBook](https://rust-lang.github.io/mdBook/)
and deployed to GitHub Pages.

**Read it:** start at [`docs/textbook/00_index.md`](docs/textbook/00_index.md)
— the chapters render natively on GitHub (LaTeX + mermaid), or browse the
hosted site (the `Textbook` workflow builds + deploys it).

## Layout

- [`docs/textbook/`](docs/textbook/) — the book: 21 chapters, the widget
  toolkit (`widgets/shared/`), the LaTeX lint (`lint/`), the mdBook config
  (`book.toml`) + theme.
- [`widgets/`](widgets/) — Rust source for the three WASM widgets (bandit,
  gridworld, tile-coding). The built `pkg/` artifacts are **committed** under
  `docs/textbook/widgets/<name>/pkg/`, so the site needs no Rust build. Rebuilding
  the blobs needs the `rl_core` / `playground` crates — see [`widgets/README.md`](widgets/README.md).
- [`.github/workflows/textbook.yml`](.github/workflows/textbook.yml) — builds +
  lints the book and deploys to Pages.
- `.claude/skills/textbook-with-widgets/` — the authoring skill.

## Build locally

```sh
cargo install --locked mdbook mdbook-mermaid
mdbook serve docs/textbook --open
```

Extracted from the [`falahat/simulator`](https://github.com/falahat/simulator)
monorepo, where the textbook is anchored in a real RL simulator; the chapters'
"project tie-in" links point back there.

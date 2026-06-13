# Textbook KaTeX linter

Validates every `$…$` and `$$…$$` math block in `docs/textbook/*.md`
against the same KaTeX renderer GitHub uses server-side. A block that
fails this linter will render as plaintext on GitHub.

The linter is a self-contained Node.js project under
`docs/textbook/lint/` so it does not pollute the Rust workspace.

## One-time setup

Requires Node 18+.

```bash
cd docs/textbook/lint
npm install
```

This pulls in `katex` (≈3 MB) into `node_modules/`. Both
`node_modules/` and `package-lock.json` are gitignored.

## Run

```bash
cd docs/textbook/lint
npm run lint              # warn-mode (matches GitHub default)
npm run lint:strict       # error-mode (catches more things;
                          # use to clean up before a release)
```

Output format:

```
05_monte_carlo_methods.md:214  [display]  KaTeX parse error: …
    \mathbb{E}_\pi[X] = \sum_x x P_\pi(x) = …
```

Exit code 0 if every block renders; non-zero otherwise.

## What it catches

Real bugs the author has hit on this textbook:

- `\!` (negative thin space) followed by an operator like `\left` —
  produces "2exp!" plaintext on GitHub.
- `\\left` (double-backslash) from a botched sed pass — renders as
  literal text.
- `^{}` (empty superscript) — KaTeX strict-mode flags it.
- Newlines inside an inline `$…$` block.
- Unbalanced `$` count per file.

It does **not** check for:

- Whether the markdown surrounding the math is also rendered cleanly.
- Mermaid diagrams, GitHub-flavored extensions, or HTML embedding.
- Math inside fenced code blocks (those are deliberately stripped
  before extraction).

## Wiring into CI

Add to a pre-commit hook or GitHub Actions workflow:

```yaml
- run: cd docs/textbook/lint && npm ci && npm run lint
```

## What it does NOT include

- A markdown parser. The extractor uses regex + a fenced-code stripper.
  Robust enough for the textbook's content; would not survive
  pathological math inside HTML tags or complex nesting.
- A fixer. If a block fails, you fix the source by hand. Most failures
  are obvious from the snippet the linter prints.

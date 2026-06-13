# Phase 2 — Chapter structure & writing

A chapter template that worked across 18 chapters and ~2700 math
blocks. Copy `examples/chapter-template.md` for a blank starting
point.

## Top-level template

```markdown
# Chapter N — [Title]

## Why this chapter exists

[2-3 paragraphs. What does this chapter contribute that the others
don't? What would the reader miss if they skipped it?]

## Table of contents

- [N.1 ...](#n1-...)
- [N.2 ...](#n2-...)
- ...

## N.1 Section name

[Prose. Math. Worked examples.]

### Try it: [widget name]

<div id="chN-XXX-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/XXX/widget.js"></script>

[Brief prose describing what the widget shows.]

## N.2 ...

...

## N.X Project tie-in

[Link to specific source files that implement these concepts.]

## Exercises

1. [Theory exercise — derivation or proof.]
2. [Computation exercise — work through a numerical case.]
3. [Implementation exercise — modify the linked source code.]

## Citations

- [List of `[Author Year]` references with full bibliography entries
  inline OR pointing at `bibliography.md`.]
```

This template is descriptive, not prescriptive. Sections can be
nested, reordered, or skipped where they don't apply. The Try-it,
Project-tie-in, and Exercises sections are the ones that distinguish
this from generic prose.

## Where to place widgets

Insertion points are not random. Place each "Try it" widget at the
exact point in the prose where the reader has just *seen* the math
but does NOT yet have intuition for what it means. The widget gives
them intuition; the next paragraph capitalizes on that intuition.

Wrong: widget at end of section, after all explanation.
Right: widget right after the formula, before the explanatory text.

Reader's flow:
1. "Here's the formula `Σ_a π(a|s) Σ_{s'} P(s'|s,a) [r + γV(s')]`."
2. "Try it" widget: drag sliders, watch the sum recompute live.
3. "Notice how each term contributes — the policy weights determine
   which actions dominate; the transition probs determine which
   next-states the agent expects; the discount γ controls how much
   future value matters..."

The third part is now landing on a primed mind. Reading without the
widget, the reader has no anchor and the prose feels abstract.

## Section conventions

**Per-section length.** Each subsection should be 1-3 screens of
prose. If you find yourself at 5+ screens, split. Long undifferentiated
prose loses readers.

**Worked examples in callout blocks.** Mark numerical examples with
a blockquote so the reader can skim or pause as they prefer:

```markdown
> **Worked numerics:** At γ = 0.9 and k = 100, γ^k ≈ 2.6e-5. At
> γ = 0.99 and k = 100, γ^k ≈ 0.37 — three orders of magnitude
> larger. This is why high-γ regimes need fewer reward-shaping tricks.
```

**Definition boxes.** Mark first-encounter definitions:

```markdown
> **Definition (contraction mapping).** A function `T : X → X` on
> a metric space `(X, d)` is a γ-contraction if for all x, y ∈ X,
> `d(T(x), T(y)) ≤ γ · d(x, y)` with γ < 1.
```

**Mermaid diagrams.** Use them for graphs of objects: the agent-
environment loop, MCTS trees, MDP transition graphs. Don't use
Mermaid for charts (Plot is for that). For *static* equations,
just use math. Mermaid is for *structure*.

## Project tie-in section

Every chapter should end with a section linking to the project
implementation it discusses. This is where the textbook becomes
*grounded* — the reader can see "the textbook says X; the code
does X exactly this way."

Format:

```markdown
## Project tie-in

The Simulator implements this chapter's TD update in
[`Learner::td_update`](https://github.com/.../value_function.rs#L120).
Two notes about the implementation:

- The TD error is computed first, then the weight update is applied.
  This avoids subtle ordering bugs when α is large.
- The roster of action values is passed explicitly rather than
  enumerated — see the deterministic-RNG canary discussion in
  [`policy.rs:221`](https://...).

For the Q-bias bootstrap pathology this chapter mentions, see the
[FA pathologies chapter](https://github.com/.../17_fa_pathologies.md).
```

This section is short — usually 3-5 paragraphs. It's the bridge
between concept and artifact.

## Exercises

Three flavours per chapter:

1. **Derivation / proof.** "Prove that the Bellman optimality operator
   is a γ-contraction in the sup norm." Theoretical.
2. **Numerical computation.** "Compute V(s) for the following MDP
   with γ = 0.9." Use specific numbers; let the reader plug into a
   widget to verify if there is one.
3. **Implementation modification.** "Modify the `td_update` function
   to use Double Q-learning instead." Concrete, has a definite answer
   the reader can compare against.

Don't add exercises that have no clear right answer. "Discuss why
RL is hard" is bad. "Prove the Watkins-Dayan convergence theorem"
is good.

## Citation conventions inside a chapter

Inline citations follow the result they cite:

> The TD-error convergence follows the Watkins-Dayan theorem [Watkins
> & Dayan 1992], which requires that every (state, action) pair be
> visited infinitely often.

The reader sees the claim, sees the source, can verify. Pile-on
citations at the end of paragraphs are confusing.

## A note on tone

Avoid:
- **Hype.** "This revolutionary technique..." — claim the merit,
  don't editorialize.
- **False modesty.** "Don't worry, we'll cover the harder math
  later" — the reader is here for the math; treat them as capable.
- **Marketing-speak.** "Master's-level treatment" — let the reader
  judge the level from the content.
- **Apology.** "This is admittedly complex..." — if it's complex,
  explain it; don't pre-apologise.

Aim for the voice of a knowledgeable colleague explaining at a
whiteboard. Direct. Specific. Honest about what's hard and what isn't.

## Output of this phase

A chapter `.md` file under `docs/textbook/`. Mathjax-compatible
math via `$…$` and `$$…$$`. Mermaid diagrams where structure helps.
Widget mount points in HTML, two-line embeds per `references/07-widget-authoring.md`.

Next: validate it via `references/05-latex-and-rendering.md`.

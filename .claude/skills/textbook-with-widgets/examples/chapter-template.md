# Chapter N — [Replace with title — a topic, not a technique]

## Why this chapter exists

[Two to four short paragraphs. What gap does this chapter close
that prior chapters didn't? What does the reader need to take away
that's specific to THIS chapter — not the field at large?

Resist the urge to introduce the topic here. The point is to motivate
why the chapter is in the book at all, not to start teaching.]

## Table of contents

- [N.1 First major idea](#n1-first-major-idea)
- [N.2 Second major idea](#n2-second-major-idea)
- [N.3 Third major idea](#n3-third-major-idea)
- [N.4 Pathology / failure mode](#n4-pathology--failure-mode)
- [N.5 Project tie-in](#n5-project-tie-in)
- [N.6 Exercises](#n6-exercises)

## N.1 First major idea

[Prose. Math. Examples.]

The formula:

$$
\text{result} = \sum_{k=0}^{\infty} \gamma^k r_k
$$

[Define every symbol on first use. Greek letters get both forms.
"γ (gamma) is the discount factor; r_k is the reward at step k."]

### Try it: [widget name]

<div id="chN-XXX-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/XXX/widget.js"></script>

[One short paragraph telling the reader what to manipulate and what
to look for. Don't over-explain — the widget is the explanation.]

## N.2 Second major idea

[Same structure. Prose, math, optional widget.]

> **Definition (term).** A formal definition in a blockquote, set
> off so the reader can locate it later. Use when introducing a
> capital-letter-named concept (Contraction Mapping, Markov Decision
> Process, Hoeffding Inequality).

> **Worked numerics:** A concrete numerical instance, also in a
> blockquote so it's skimmable. Use small numbers the reader can
> follow.

## N.3 Third major idea

[…]

## N.4 Pathology / failure mode

[Every chapter that has a non-trivial theorem also has a counter-
example or failure case. Show it. It's where the concept's limits
become real.]

### Try it: [pathology widget]

<div id="chN-pathology-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/pathology/widget.js"></script>

[The widget should let the reader violate the theorem's hypothesis
and watch the algorithm break. "Watch it break" is one of the most
high-leverage teaching widgets.]

## N.5 Project tie-in

[How does the project implement this chapter's ideas?

Link to specific files + line numbers using absolute GitHub URLs:

- [`function_name`](https://github.com/owner/repo/blob/main/path/file.rs#L42)
  — the canonical implementation.
- [`test_file`](https://github.com/owner/repo/blob/main/path/test.rs)
  — the validation tests for the above.
- Any related proposal/audit docs in `docs/proposals/`.]

Two notes about the implementation:

- [One specific implementation choice and why.]
- [Another, ideally pointing out a non-obvious design decision.]

## N.6 Exercises

1. **[Derivation / proof title].** [State the problem clearly.
   Specify what the reader needs to produce. Reference any prior
   chapter results they should use.]

2. **[Numerical computation title].** [Specific numbers. Specific
   expected output. If the chapter has a widget, point the reader
   at the widget to verify their answer.]

3. **[Implementation modification title].** [Pointer to the source
   file to modify. Specific change to make. Test that would validate
   the change works.]

## Citations

- [Smith Year] *Title*. Publisher. Section X.Y.
- [Jones Year] "Paper title." *Conference / Journal*.
- [Course Name Year] Lecture notes / video URL.

[Or use a footnote-style if your bibliography is in a separate
`bibliography.md` file:

`[Sutton & Barto Ch 6.5]` references the textbook entry in the
shared bibliography.]

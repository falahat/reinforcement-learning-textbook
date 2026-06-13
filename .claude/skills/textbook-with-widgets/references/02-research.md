# Phase 1 — Research per chapter

You have a syllabus. Now expand one chapter at a time. For each
chapter, gather sources before writing prose.

## What "research" means here

Not novel research — restatement. The deliverable is a chapter that:
- Cites primary sources accurately
- Restates the main results in clear prose
- Adds tie-ins to the user's application (if any)
- Avoids overclaiming originality

Plagiarism is the wrong fear here. Restating a published result and
**citing it** is honest scholarship; restating without citation is
not. The goal is to make the existing literature legible to a
specific reader.

## Source-finding workflow

For each chapter, build a sources list in this order:

1. **The textbook-of-record for the field.** Whatever everyone cites.
   For RL: Sutton & Barto. For ML: Bishop or Murphy. For real
   analysis: Rudin. For algorithms: CLRS. Read the relevant chapter
   and pull out 1-3 result statements you'll restate.
2. **One or two academic papers for the key results.** The original
   Bellman 1957 paper, the TD-Gammon 1995 paper, the DQN 2015 paper,
   the PPO 2017 paper. These should match the textbook-of-record's
   citations.
3. **One or two course lecture notes** for the specific topic. David
   Silver's RL course, Sergey Levine's RL course, Andrew Ng's ML
   course. Often these are clearer than the original paper.
4. **Any project-specific source code** the chapter ties into. For
   our RL textbook, this was the simulator's `crates/cognition/`
   tree. Link to specific files, not just directories.

A good chapter cites 5-15 sources. Fewer than 5 suggests you're
making it up; more than 20 is graduate-thesis territory and
probably overkill for the target reader.

## Citation conventions

Use a single bibliography file at the textbook root:
`bibliography.md`. Numbered or named entries; either works. Each
chapter cites by reference like `[Sutton & Barto §6.5]` or
`[Mnih et al. 2015]`. The bibliography expands to full author /
title / publication info.

Inline citations should be **immediately after the result they cite**,
not at the end of the paragraph. Reader sees a claim, sees who said
it, decides to trust it or look it up.

For project tie-ins, link to specific file + line number using
absolute GitHub URLs:

```markdown
[`Learner::td_update` in value_function.rs:42](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/value_function.rs#L42)
```

Always use absolute URLs. Relative paths like `../../crates/...` don't
resolve correctly in mdBook output — the deployed page lives under
the book's URL prefix and `../../crates/` lands at a non-existent
path. See `references/05-latex-and-rendering.md` for the
source-rewriter script that bulk-converts relative paths.

## What to read vs what to write

Research is reading. Writing is restating. Don't blur them.

While reading, take notes on:
- The 2-3 sentences that ARE the chapter's punchline
- The single most-common pitfall that practitioners hit
- The simplest possible example that illustrates the concept
- Any pathology / counter-example that lets you say "watch it break"

Each becomes a section in the chapter. The "punchline" goes near
the top; the "pitfall" gets its own subsection; the "example" gets
a worked numerical instance; the "pathology" gets an interactive
widget (see `references/07-widget-authoring.md`).

## Anti-pattern: writing while reading

The seductive failure mode is alternating one paragraph of reading
with one paragraph of writing. The result reads like notes — choppy,
non-narrative, lots of unfollowed loose ends. Instead:

1. Read the whole chapter's worth of sources first. Take notes.
2. Then close all the source material.
3. Write the chapter from your notes.
4. Open the sources again only to fact-check specific claims.

Two-pass writing — one for structure, one for accuracy — produces
far more readable chapters than one-pass alternating.

## Time budget

Per chapter: ~6 hours of reading, ~4 hours of note synthesis,
~4 hours of first-draft writing, ~2 hours of editing. Total ~16
hours per chapter. The 18-chapter RL textbook represented a few
weeks of focused work at this pace.

Don't try to compress this. Chapters that took 2 hours read like
they took 2 hours; the reader feels it.

## Output of this phase

For each chapter, a research-notes file (you can throw it away
later) with:
- A flat list of sources, full citations
- One-line summaries of each source's main contribution
- The 2-3 "punchline" sentences for the chapter
- The "pitfall" + "example" + "pathology" trio

Then move to `references/03-chapter-structure.md` for the writing
template.

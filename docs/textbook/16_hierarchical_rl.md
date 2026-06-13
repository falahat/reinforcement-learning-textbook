# Chapter 16 — Hierarchical Reinforcement Learning and Options

> **Prerequisites:** Chapter [3](05_mdps_and_bellman_equations.md) (MDPs),
> Chapter [6](08_temporal_difference_learning.md) (Q-learning),
> Chapter [11](13_actor_critic.md) (actor-critic),
> Chapter [13](15_model_based_rl.md) §13.6 (planner deletion story).

> **Citations:** [Sutton, Precup & Singh 1999] (options framework);
> [Bradtke & Duff 1995] (SMDP-Q); [Bacon, Harb & Precup 2017]
> (Option-Critic); [Vezhnevets et al. 2017] (FeUdal Nets);
> [Nachum et al. 2018] (HIRO); [Levy et al. 2019] (HAC);
> [Eysenbach et al. 2018] (DIAYN); [Parr & Russell 1998] (HAM).

> **Learning objectives:**
> 1. State the options framework $\langle I, \pi, \beta \rangle$ and
>    work the SMDP-Q update.
> 2. Compute the effective horizon $1/(1-\gamma)$ at $\gamma = 0.9$ and
>    motivate temporal abstraction from there.
> 3. Compare hand-engineered options, learned options (Option-Critic),
>    and goal-conditioned hierarchies (HIRO, HAC).
> 4. Map the Simulator's `RecipeMeme` programs onto the options
>    framework exactly.
> 5. Sketch the gap between current recipe execution and what
>    Option-Critic would add.

## Why this chapter exists

TD with $\gamma = 0.9$ has effective horizon $1/(1 - \gamma) = 10$
steps. Past 10 steps, discounting makes signal vanish. For tasks
requiring 100, 500, or 5000 steps of coherent behavior — opening a
combination lock, building a shelter, harvesting a crop —  flat
1-step RL is structurally inadequate.

**Hierarchical RL** introduces *temporal abstraction*: high-level
decisions ("get food", "build shelter", "execute crafting recipe")
that span many low-level primitives. Each high-level decision happens
on a slower clock; the agent's value function works at the high-level
timescale where $\gamma^{H}$ stays meaningful.

The Simulator's **recipes** are exactly this. They are options
mechanically — multi-step programs with preconditions, internal
ordering, and termination — but the current architecture does not
*learn* over them (it learns Q on flat actions only). This chapter is
the bridge to making the existing recipe substrate fully RL-trainable.

## Table of contents

- [14.1 The case for hierarchy](#141-the-case-for-hierarchy)
- [14.2 The options framework](#142-the-options-framework)
- [14.3 SMDP-Q learning](#143-smdp-q-learning)
- [14.4 Option-Critic](#144-option-critic-end-to-end-option-learning)
- [14.5 FeUdal Networks](#145-feudal-networks)
- [14.6 HIRO and HAC](#146-hiro-and-hac-goal-conditioned-hierarchies)
- [14.7 DIAYN — skill discovery without rewards](#147-diayn--skill-discovery-without-rewards)
- [14.8 The Simulator's recipes *are* options](#148-the-simulators-recipes-are-options)
- [14.9 Where hierarchy could help the Simulator](#149-where-hierarchy-could-help-the-simulator)
- [14.10 Exercises](#1410-exercises)
- [14.11 References](#1411-references-cited-in-this-chapter)
- [14.12 Further reading](#1412-further-reading)

---

## 14.1 The case for hierarchy

Three reasons flat 1-step RL hits a wall:

### Reason 1: discounting kills long-horizon signal

At $\gamma = 0.9$, the contribution of a reward $r$ delivered $k$ steps
in the future to the current state's value is $\gamma^k \cdot r$:

| $k$ | $\gamma^k$ at $\gamma = 0.9$ |
|---|---|
| 10 | $\approx 0.35$ |
| 50 | $\approx 5 \times 10^{-3}$ |
| 100 | $\approx 3 \times 10^{-5}$ |
| 500 | $\approx 10^{-23}$ |

By the Simulator's $\gamma = 0.9$ (per cognition step, 10 ticks each)
and L-suite's $\sim 500$-tick delays, the signal is essentially zero.
The TD update is mathematically incapable of propagating reward back
that far. Chapter 19 covers the fixes (hindsight, successor features,
reward shaping).

But the **structural** fix is to *change the timescale*. If our
"option" lasts 50 ticks and delivers $r$ at completion, the Bellman
update at the option-level uses $\gamma^{50}$ — and we operate on
option-Q with new effective horizon $1/(1 - \gamma^{50})$ at the
*option* timescale. This is what hierarchy buys.

### Reason 2: exploration in flat action space is exponentially bad

A random walk of length $L$ in a $b$-branching state space reaches a
specific goal with probability $\sim b^{-L}$. For $L = 50, b = 4$:
$4^{-50} \approx 10^{-30}$.

A random walk over *options*, where each option corresponds to a
goal-directed multi-step behavior, has dramatically smaller effective
horizons. With 5 options each lasting 10 ticks, a 50-tick task is 5
option-steps; exploration over options is tractable.

### Reason 3: transfer and modularity

A learned option (e.g., "approach the nearest food") is reusable across
tasks (hunger-driven eating, social bonding via shared meals, ritual
offerings). Flat-Q learners must re-derive this skill in every context.

### Try it: four-rooms options demo

<div id="ch14-four-rooms-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/four_rooms/widget.js"></script>

The Sutton-Precup-Singh 11×11 four-rooms gridworld with start in
the bottom-left room and goal in the top-right. Compare flat
Q-learning (4 primitives only, blue) against SMDP-Q over
{4 primitives + 4 "go to hallway" options} (green). The options
shortcut the exploration problem from Reason 2 — the green curve
climbs many episodes before the blue one.

---

## 14.2 The options framework

### Why options matter as a formalism

§14.1 motivated hierarchy informally: agents acting on slow time
scales can learn long-horizon tasks where flat Q-learning's
$\gamma^{500} \approx 0$ kills credit assignment (Ch19). But
"hierarchy" alone is vague — what *is* a higher-level action,
mathematically? The options framework gives the precise answer:
**a higher-level action is a temporally-extended policy with an
initiation set and a termination function** — a triple
$(I_o, \pi_o, \beta_o)$. Once we have that triple, every theorem
about MDPs (Bellman equations, value iteration, Q-learning) has a
direct semi-MDP analogue.

The framework is load-bearing for three reasons:
1. **Option-Critic** (§14.4) learns $(I_o, \pi_o, \beta_o)$
   end-to-end via policy gradient — a direct generalisation of
   Ch12's PGT.
2. **FeUdal Networks, HIRO, HAC** (§§14.5–14.6) are variations on
   "high-level controller picks a sub-goal; low-level controller
   tries to reach it" — sub-goals are a *parameterised* family of
   options.
3. **The Simulator's recipes** (§14.8) are options in all but name —
   matching the framework reveals where the Simulator under-uses
   what HRL offers.

[Sutton, Precup & Singh 1999] introduced the options framework, the
canonical formalism for HRL.

> **Definition.** An **option** is a triple $o = \langle I_o, \pi_o, \beta_o \rangle$:
> - $I_o \subseteq \mathcal{S}$: the **initiation set** — states where
>   the option can be initiated.
> - $\pi_o$: the **internal policy** — a (possibly stochastic) policy
>   over primitive actions that drives execution.
> - $\beta_o: \mathcal{S} \to [0, 1]$: the **termination function** —
>   probability of ending the option at each state.

To execute option $o$: starting in some $s \in I_o$, repeatedly:
1. Sample $a \sim \pi_o(\cdot \mid s)$, apply, observe $s'$.
2. Terminate with probability $\beta_o(s')$.

The composite "policy over options" picks an option in any state where
its initiation set is satisfied, executes it to completion, then picks
another. This turns the MDP into a **Semi-Markov Decision Process**
(SMDP): decisions at irregular time intervals.

### Primitive actions as options

A primitive action $a$ is just an option with $I_a = \mathcal{S}$,
$\pi_a$ = deterministic $a$, $\beta_a \equiv 1$. So options *generalize*
primitive actions; you can mix them freely in a "menu" of options +
primitives.

### Options as policies on a slower clock

The agent has two layers:
- **Top layer**: a Q-function (or policy) over options.
- **Bottom layer**: each option's internal $\pi_o$.

The top layer sees state transitions at option-completion
granularity; the bottom layer sees primitive transitions inside an
option. The effective horizon at the top is much longer per
Bellman backup.

### Why this is the key insight for long-horizon credit

If primitive actions discount at $\gamma$ per step and a useful
behavioural unit takes $k$ steps, the credit signal at the top of
that unit reaches the start at $\gamma^k$. With $\gamma = 0.9$ and
$k = 100$ (a reasonable option length), $\gamma^{100} \approx 10^{-5}$ —
already near float epsilon. **But the top layer sees one decision
spanning $k$ primitive steps, so its effective discount is
$\gamma^k$ in one *top-level* step**, not $\gamma$. Credit
propagates *across options* at one top-level step per
option-completion.

This is the algebraic reason hierarchy helps long-horizon credit
assignment (Chapter 19): you shrink the effective horizon at the
top by exactly the option-length factor.

### What options don't say

- **They don't tell you what the options *are*.** The framework
  is a *vocabulary* for hierarchies; choosing the option set is a
  separate design problem (§14.4 option-critic learns options
  end-to-end; §14.5 FeUdal nets parameterise them; §14.8 the
  Simulator's recipes hand-author them).
- **They don't preserve all MDP properties.** The SMDP that
  results from options has irregular time intervals; many
  standard MDP results need careful re-derivation (the SMDP-Q
  update in §14.3 is the canonical example).
- **Options can hurt as well as help.** A bad option set restricts
  the policy class — the agent can be locked out of a behaviour
  no option implements. §14.4's "option-critic" learns option
  termination jointly with the option's internal policy to avoid
  this lock-in.

---

## 14.3 SMDP-Q learning

[Bradtke & Duff 1995] introduced Q-learning for SMDPs:

> **SMDP Q-update.**
>
> $$
> Q(s, o) \leftarrow Q(s, o) + \alpha\left[r_\text{cumulative} + \gamma^\tau \max_{o'} Q(s', o') - Q(s, o)\right]
> $$
>
> where $\tau$ is the option's duration (in primitive steps) and
> $r_\text{cumulative} = \sum_{k=0}^{\tau-1} \gamma^k r_{t+k+1}$ is the
> discounted sum of primitive rewards across the option's execution.

Read it: same as Q-learning, but **the next state and reward jump in
time** by the option's duration. The $\gamma^\tau$ correctly discounts
the bootstrap target.

### What changes vs. flat Q

- **The transition $(s, o, r, s', \tau)$ is now 5-tuple.** $\tau$ must
  be tracked.
- **Cumulative reward replaces single-step reward.** Must be
  accumulated during option execution.
- **The bootstrap discount is $\gamma^\tau$** — longer options
  discount more.
- **Effective horizon stretches.** For option duration ~10 ticks at
  $\gamma = 0.9$ per tick, the option-level discount is
  $\gamma^{10} \approx 0.35$ — so the option-level effective horizon is
  $1/(1-0.35) \approx 1.5$ option-steps, which is $\sim 15$ ticks. **Not
  enough.** Most options need to be ~30+ ticks long for the SMDP
  horizon to beat flat Q.

### Convergence

[Bradtke & Duff 1995] showed SMDP-Q converges under the usual
Q-learning conditions, applied to the option-level MDP. The catch: the
option-level policy is *fixed* (the inner $\pi_o$ is not learning).
Learning option-internal policies *as well* is what Option-Critic does.

### Variants for intra-option learning

- **Intra-option Q-learning**: update Q on every primitive step inside
  an option, even though only options are picked at the top. Faster
  learning but slightly biased.
- **Off-policy intra-option**: update Q on options other than the one
  currently executing, using importance sampling. Aggressive; can
  diverge.

### Try it: SMDP-Q bootstrap discount illustrator

<div id="ch14-smdp-bootstrap-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/smdp_bootstrap/widget.js"></script>

Pick a per-tick reward shape and an option duration τ. The timeline
shows each primitive tick's contribution γ^k · r_{t+k+1} as a bar
whose visible area is the discounted reward; the second strip
decomposes the SMDP-Q target into r_cumulative + γ^τ·V̂. Toggle
"sparse — at terminal step" and watch the only nonzero contribution
fade as τ grows.

---

## 14.4 Option-Critic: end-to-end option learning

[Bacon, Harb & Precup 2017] gave the first end-to-end algorithm to
**learn options from scratch** — both the internal policies and the
termination functions.

The architecture: parameterize each option's $\pi_o(a \mid s; \theta_\pi^o)$ and $\beta_o(s; \theta_\beta^o)$. Add a Q-function over $(s, o)$ pairs.
Then apply policy-gradient theorems:

> **Intra-option policy gradient.**
>
> $$
> \nabla_{\theta_\pi^o} J = \mathbb{E}\left[\nabla_{\theta_\pi^o} \log \pi_o(a \mid s) \cdot Q_U(s, o, a)\right]
> $$
>
> where $Q_U$ is the **unrolled Q-value** treating $(s, o)$ as the
> SMDP state.
>
> **Termination policy gradient.**
>
> $$
> \nabla_{\theta_\beta^o} J = -\mathbb{E}\left[\nabla_{\theta_\beta^o} \beta_o(s') \cdot A(s', o)\right]
> $$
>
> where $A(s', o) = Q(s', o) - V(s')$ is the advantage of continuing
> the option vs. terminating.

Read: terminate options when the advantage of *not* continuing exceeds
the advantage of continuing. This is automatic, learned termination —
no hand-tuning of $\beta$.

The result: agent learns a small number of options (paper used 4 or 8)
that span diverse, useful sub-behaviors. On Atari-style benchmarks,
matches or beats flat Q-learning while providing *interpretable* option
structure ("Option 3 always moves the player upward; Option 7 fires
missiles").

### Practical issues

- **Option collapse.** Without regularization, options can collapse to
  identical policies (all options become "do whatever's best").
  Add entropy bonus over $\pi_o$ or option-selection regularization.
- **Termination collapse.** $\beta$ can collapse to always-1
  (every step is a new option) or always-0 (one option runs forever).
  Add deliberation cost — a small penalty for option changes.
- **Number of options.** A hyperparameter; rarely automatically chosen.
  Papers use 4-8 typically.

### Try it: termination collapse and the deliberation cost

<div id="ch14-option-critic-termination-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/option_critic_termination/widget.js"></script>

Set η = 0 and watch β(s, o) collapse to ~1 everywhere — every step
fires a new option, so options reduce to primitives. Raise η to ~0.3
and β stabilises low across the corridor, with the hallway state
(the "room boundary") naturally getting the highest termination
probability. This is the §14.4 pathology and its remedy in one frame.

---

## 14.5 FeUdal Networks

[Vezhnevets et al. 2017]: a different hierarchy, inspired by
*feudal RL* (Dayan & Hinton 1993).

**Manager-Worker architecture:**
- **Manager**: outputs latent **goal vectors** $g_t$ at a slow tempo
  (every $k$ steps, often 10).
- **Worker**: takes the goal and the current state, outputs primitive
  actions over the next $k$ steps.

The Manager is trained to set goals that correlate with environment
reward; the Worker is trained to *follow* the Manager's goals,
measured by intrinsic reward $r^\text{int}\_t = \cos(s\_{t+c} - s\_t, g\_t)$ —
the dot product between the actual state change direction and the
Manager-set goal direction.

This is **directional**: the Manager says "increase your position in
this direction in latent space"; the Worker figures out which
primitives accomplish that.

FeUdal Networks were strong on memory-heavy 3D tasks (DMLab) and
sparse-reward Atari (Montezuma's Revenge). The decomposition handles
long horizons via the slow Manager.

---

## 14.6 HIRO and HAC: goal-conditioned hierarchies

A modern simplification: **goal-conditioned** policies. The high-level
policy outputs a target state (a "subgoal"); the low-level policy is
goal-conditioned $\pi(a \mid s, g)$ trained to reach $g$.

### HIRO ([Nachum et al. 2018])

Two levels (high, low). The high-level policy samples subgoals at
every $k$ steps. Low-level intrinsic reward is distance to the goal.
**Goal relabeling** with off-policy correction makes this work at scale.

The HIRO trick: when training the high-level policy off-policy, the
*actual* low-level behavior may differ from what the original
high-level goal would have generated (because the low-level policy has
since updated). HIRO relabels old high-level transitions with the
subgoal that, *under the current low-level policy*, would best
reproduce the observed primitive trajectory. This makes off-policy HRL
sample-efficient.

### HAC ([Levy et al. 2019])

Multi-level hierarchies ($\geq 3$ levels) with **hindsight goal
relabeling** at every level. Each level pretends it succeeded by
relabeling its goal to what it actually achieved. Works on long-horizon
robotics tasks (block stacking, pick-and-place sequences).

### Why goal-conditioned hierarchies are popular

- **Single policy, multiple skills.** The goal-conditioned low-level
  policy generalizes across goals.
- **No discrete option count to tune.** Goals are continuous vectors.
- **Hindsight relabeling for free.** Each trajectory teaches the
  agent about whichever goal it actually achieved.

### Try it: HIRO's subgoal relabelling

<div id="ch14-hiro-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hiro/widget.js"></script>

Step through one episode. First the high-level policy emits five subgoals (red dots). Then the low-level policy rolls out (orange path) — it tries to track each subgoal but drifts, so the *reached* states (orange dots) sit elsewhere. The final two frames overlay the **relabeled** subgoals (green) on top of the reached states, with dashed lines pairing each original red goal with its green replacement. The green dots are what HIRO replays during off-policy high-level training — using the *original* red goals would be inconsistent with what the current low-level policy can actually do.

---

## 14.7 DIAYN — skill discovery without rewards

[Eysenbach et al. 2018] (Diversity Is All You Need): learn a library
of distinct skills *without any extrinsic reward*. The objective is
**mutual-information** maximization:

$$\max_{\pi, q} I(s; z) = H(z) - H(z \mid s)$$

where $z$ is a discrete (or continuous) skill index and $\pi(a \mid s, z)$
is the skill-conditioned policy. Equivalently: each skill should visit
a distinguishable region of state space.

Implementation: sample $z$ uniformly at episode start; train policy
$\pi(a \mid s, z)$ and a discriminator $q(z \mid s)$. The policy's reward
is $\log q(z \mid s)$ (high when the discriminator can correctly
identify the skill from the state).

Result: agent learns 10-50 distinct behaviors that span the state
space — locomotion in different directions, gripper poses, etc. Then
these skills are reusable for downstream tasks.

This is **unsupervised pretraining of an options library**. Combined
with a downstream high-level controller trained on extrinsic reward,
DIAYN often beats from-scratch HRL on sparse-reward tasks.

### Try it: skill coverage on a 2D plane

<div id="ch14-diayn-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/diayn/widget.js"></script>

Five skills $z \in \{0,\ldots,4\}$, each with a different drift direction. Each runs 100 steps from the origin and we plot every visited state, coloured by skill. The headline picture is the *five distinct lobes*: a discriminator $q(z \mid s)$ trained on these visitations can identify the skill from the state with very low error — that is the mutual-information objective made visual. Slide the drift to zero and the lobes collapse into one Gaussian blob; slide the noise up and they overlap. The drift/noise ratio is the lever that controls how diverse the discovered skills actually are.

---

## 14.8 The Simulator's recipes *are* options

The Simulator's recipe substrate (see
[`crates/engine/q_learning/src/action_template.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/action_template.rs)
and recipe handling in
[`planner/src/lib.rs:178-272`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/lib.rs))
is mechanically the options framework:

| Options concept | Recipe equivalent |
|---|---|
| **Initiation set** $I_o$ | Recipe `precondition` (`DriveAbove`, `DriveBelow`, `HasItem`) |
| **Internal policy** $\pi_o$ | Recipe `steps` list (deterministic ordering) |
| **Termination** $\beta_o$ | Last step complete = $\beta_o(s_{\text{end}}) = 1$, else 0 |

Each `RecipeMeme` is an option. When `precondition` matches, the agent
can initiate the recipe; its `ActiveRecipe` cursor walks the step list
to completion. Steps are concrete:
- `Action(template_id)`: execute a primitive action.
- `Goto(target_item)`: walk to a target (world-state advance).
- `Acquire(target_item)`: pick up an item.
- `Communicate(speech_act_id)`: emit a speech act.
- `Think(recipe_id)`: nested recipe call.

A worked example from
[`credit_attribution.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/credit_attribution.rs):
```
recipe.consume_food:
  Precondition: DriveAbove(Hunger, 0.4)        ← I_o
  Step:         Action(Consume)                 ← π_o
  Delta:        Hunger -= 0.5                  ← (semantic effect)
```

This is a one-step option. Multi-step recipes (e.g. `recipe.cook_stew`,
in the meme content fixtures) are options with $\tau > 1$.

### What's there, what's not

**There:**
- Initiation sets (preconditions).
- Internal policies (step lists; mostly deterministic).
- Termination (step-list exhaustion).
- Recipe selection bonus in the score formula
  (`policy.rs:386-387` and `:442` — a small continuation/initiation
  bonus that gives recipes a leg up in the candidate scoring).

**Not there:**
- **SMDP-Q learning over recipes.** The Simulator does not maintain
  $Q(s, o)$ where $o$ is a recipe — the $Q$ is keyed by primitive
  action template+parameter (`policy.rs:181-199`). When a recipe
  executes, its constituent primitives each get their own Q-update;
  the recipe-as-whole does not.
- **Intra-option learning.** Recipes do not adapt their step ordering
  based on outcome. They are hand-authored programs.
- **Option discovery.** Recipes are content (memes), not learned.

The previous (deleted) forward-search planner *did* maintain SMDP
option-values
(`q_learning/src/option_values.rs`, removed in commit `0a41cef`). The
deletion was part of the broader pivot to flat policy.

### Try it: recipe-as-option flow

<div id="ch14-recipe-option-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/recipe_option/widget.js"></script>

A 7×7 gridworld with the start in the bottom-left and the goal in
the top-right. Pick a "recipe" (a step list — that's $\pi_o$) and a
precondition (that's $I_o$, shown as the orange-tinted region). SMDP-Q
runs over {4 primitives + 1 recipe-option} (green) vs. primitives
only (blue). The purple trace is the recipe's deterministic trajectory
from start — a recipe that lands close to the goal is a strong option;
a misaligned recipe is dead weight.

---

## 14.9 Where hierarchy could help the Simulator

Three angles:

### Angle 1: SMDP-Q over recipes (cheapest)

Add a second tile-coded learner keyed by `RecipeMemeId` instead of
action template. When a recipe initiates, snapshot $(s, o)$ and start
accumulating discounted reward; when it terminates, apply the SMDP-Q
update. The recipe-level Q feeds back into recipe selection.

Cost: one extra learner per agent. Benefit: $\gamma^\tau$-discount-
adjusted credit assignment to whole recipes. Concretely, the L-suite's
~500-tick plant-and-harvest sequence becomes a 1-2 option-step problem
at the recipe layer — *if* a "plant then harvest" recipe exists or is
authored.

### Angle 2: Option-Critic over learned recipe-internal policies

The current `consume_food` recipe is a single `Action(Consume)` —
trivial step list. More complex recipes (e.g., "cook stew") have a
fixed sequence. Option-Critic would *learn* the step orderings from
reward.

For the Simulator this is a larger refactor: recipes would no longer be
hand-authored deterministic step lists, but learned stochastic
internal policies. Would require:

- Replacing the step-list cursor with a learned $\pi_o(a \mid s)$ per
  recipe.
- Training termination $\beta_o$ from outcomes.
- Significant change to meme-content authoring (memes become learned
  parameter vectors, not RON-authored programs).

Probably out of scope for current project goals (where recipe-content
is part of the cultural-transmission experiment, not the learning
substrate).

### Angle 3: Skill discovery via DIAYN-like methods

Train an unsupervised skill-library at the start of every run. The
agent comes out with 10-20 distinct skills (move in this direction,
attack this kind of target, vocalize at this pitch); recipe authoring
operates over these skills instead of primitive actions.

Highly speculative for the Simulator; relevant only if hand-authored
recipes prove too limiting (no current evidence they do).

### Which angle to pursue first?

**Angle 1** (SMDP-Q over existing recipes) is the highest leverage at
lowest cost. It directly addresses the long-horizon credit-assignment
problem (Chapter 19) by changing the timescale of credit assignment.
It composes with Fix 1 (set $w_\text{alive} = 0$) or Fix 3 (advantage
learning) from the Q-bias proposal — those operate on the per-action
$Q$; SMDP-Q operates on the per-recipe $Q$.

A worked example: at $\gamma_\text{prim} = 0.9$ per tick and a recipe
that lasts ~30 ticks, the recipe-level discount is
$\gamma_\text{recipe} = 0.9^{30} \approx 0.04$. So the recipe-level
effective horizon is $1/(1 - 0.04) \approx 1.04$ recipe-steps — still
tight, but **the immediate next reward after the recipe completes is
$\gamma^{30}$-weighted into the recipe's Q.** That puts the L-suite's
Plant-then-Harvest credit at $\gamma^{30}$ instead of $\gamma^{500}$ —
a $10^{19}$x improvement.

### Try it: option duration → effective horizon

<div id="ch14-option-horizon-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/option_horizon/widget.js"></script>

Slide τ from 1 (primitives) to 50 (long options). The ratio of
option-level effective horizon to primitive horizon explodes — that's
the order-of-magnitude credit-assignment improvement options give you.

---

## 14.10 Exercises

1. **(Four-rooms options example.)** Reproduce the four-rooms problem
   from [Sutton, Precup & Singh 1999] with hand-defined "go to room N"
   options. Compare flat Q-learning, SMDP-Q over options, and a
   mixed-policy that uses both options and primitives.

2. **(SMDP-Q derivation.)** Starting from the Bellman optimality
   equation for the SMDP, derive the SMDP-Q update. Identify where
   $\gamma^\tau$ enters.

3. **(Option-Critic on a 2D gridworld.)** Implement Option-Critic with
   4 options on a 11×11 gridworld with 4 goals (one per corner). Plot
   the learned $\pi_o$ for each option as a flow field.

4. **(Termination collapse.)** Train Option-Critic without deliberation
   cost. Verify that $\beta_o \to 1$ in your run (every step terminates
   the option). Add a small cost; verify recovery.

5. **(DIAYN on a 2D gridworld.)** Train DIAYN with 8 discrete skills.
   Visualize the state-visit distribution for each. Confirm they are
   distinct.

6. **(Project — SMDP-Q over recipes.)** Sketch the code change to
   add a second learner keyed by recipe-id to the Simulator.
   Detail: where the snapshot is taken, where the cumulative reward
   accumulates, where the update fires.

7. **(Recipe-as-option, formally.)** Write out
   `recipe.consume_food` as $\langle I, \pi, \beta \rangle$ exactly.
   Then do the same for a 3-step recipe (find one in the meme content
   fixtures). Identify what $\pi_o$ would need to be for nondeterministic
   step ordering.

8. **(Horizon math.)** For options of duration
   $\tau \in \{5, 10, 30, 100\}$ at $\gamma = 0.9$ (per tick),
   compute the option-level effective horizon
   $1/(1 - \gamma^\tau)$ in option-steps and in
   primitive ticks. Where does it start to bridge the L-suite's 500-tick
   gap?

---

## 14.11 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Bacon, Harb & Precup 2017] — Option-Critic (§14.4).
- [Bradtke & Duff 1995] — SMDP-Q (§14.3).
- [Dayan & Hinton 1993] — feudal RL (§14.5).
- [Eysenbach et al. 2018] — DIAYN (§14.7).
- [Levy et al. 2019] — HAC (§14.6).
- [Nachum et al. 2018] — HIRO (§14.6).
- [Parr & Russell 1998] — HAM (§14.2).
- [Sutton, Precup & Singh 1999] — options framework (§14.2-14.3).
- [Vezhnevets et al. 2017] — FeUdal Networks (§14.5).

## 14.12 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton, Precup & Singh 1999] | Whole paper | Canonical options framework |
| Bacon thesis 2018 | Option-Critic and beyond | Deep treatment |
| [Eysenbach et al. 2018] | DIAYN | Unsupervised skill discovery |
| [Pateria et al. 2021] | Survey of HRL | Modern overview |

---

**Next:** [Chapter 17 — Function Approximation Pathologies](17_fa_pathologies.md) —
the Q-bias bootstrap pathology, in detail.

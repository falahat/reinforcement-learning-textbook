# Chapter 20 — Action Spaces Beyond Discrete

> **Prerequisites:** Chapters [6](08_temporal_difference_learning.md)
> (Q-learning), [10](12_policy_gradient.md) (policy gradient),
> [11](13_actor_critic.md) (actor-critic),
> [14](16_hierarchical_rl.md) (options).

> **Citations:** [Masson, Ranchod & Konidaris 2016] (PA-MDPs);
> [Hausknecht & Stone 2016] (PA-DDPG); [Xiong et al. 2018] (P-DQN);
> [Bester, James & Konidaris 2019] (MP-DQN); [Fan et al. 2019] (H-PPO);
> [Dulac-Arnold et al. 2015] (Wolpertinger); [Tennenholtz & Mannor 2019]
> (Act2Vec); [Bacon, Harb & Precup 2017] (Option-Critic);
> [Eysenbach et al. 2018] (DIAYN); [Huang & Ontañón 2020] (action
> masking). Full entries in [`bibliography.md`](bibliography.md).

> **Learning objectives:**
> 1. Formalize the parameterized-action MDP (PAMDP) and motivate it
>    from real-world actions.
> 2. Implement P-DQN and explain MP-DQN's bias fix.
> 3. Apply action masking; explain why it is a valid PG update.
> 4. Map the Simulator's `Strike{force}` and `Vocalize{volume}` onto
>    parameterized-action methods.
> 5. Tour large-action-space tools (Wolpertinger, Act2Vec) and place
>    them in the project's design space.

## Why this chapter exists

Almost every real action has a **continuous parameter**: how hard to
swing, how loudly to shout, how far to step. Discrete Q-learning forces
us to bucket these — `Strike{Light}` vs. `Strike{Hard}`, two
templates. We lose information: a "medium" strike is unrepresentable,
and we cannot generalize learning between adjacent buckets.

**Parameterized-action methods** (PA-DDPG, P-DQN, MP-DQN, H-PPO) treat
actions as $(\text{type}, \text{parameter})$ pairs and learn the
parameter continuously. This is the natural form for the Simulator's
action space; this chapter is the bridge.

The chapter also covers:

- **Large discrete action spaces** (Wolpertinger, Act2Vec) — for when
  $|\mathcal{A}|$ is in the thousands.
- **Skill discovery** (Option-Critic, DIAYN — quick recap from Ch. 14).
- **Action masking** for state-dependent legal-action sets.

## Table of contents

- [18.1 The parameterized-action MDP](#181-the-parameterized-action-mdp)
- [18.2 PA-DDPG](#182-pa-ddpg)
- [18.3 P-DQN and MP-DQN](#183-p-dqn-and-mp-dqn)
- [18.4 H-PPO — hybrid policy gradient](#184-h-ppo--hybrid-policy-gradient)
- [18.5 Large discrete spaces: Wolpertinger and Act2Vec](#185-large-discrete-spaces-wolpertinger-and-act2vec)
- [18.6 Skill discovery: Option-Critic and DIAYN](#186-skill-discovery-option-critic-and-diayn)
- [18.7 Action masking](#187-action-masking)
- [18.8 The Simulator's action space, dissected](#188-the-simulators-action-space-dissected)
- [18.9 Project tie-in](#189-project-tie-in)
- [18.10 Exercises](#1810-exercises)
- [18.11 References](#1811-references-cited-in-this-chapter)
- [18.12 Closing thoughts](#1812-closing-thoughts)

---

## 18.1 The parameterized-action MDP

### Why this formalism matters

Every previous chapter implicitly assumed actions are either
discrete (Q-learning, Ch8: $\arg\max_a Q(s, a)$ over a finite set)
or continuous (policy gradient, Ch12: parameterized
$\pi_\theta(a \mid s)$ over $\mathbb{R}^d$). Real action spaces are
*neither*. A robot doesn't just "move" or "grip" — it picks a
*type* of action (move vs grip) and then specifies *continuous
parameters* (which direction, how much force).

The PAMDP formalism is what unifies these. Once you have
$\mathcal{A} = \bigcup_k \{k\} \times \mathcal{X}_k$ — a union of
$K$ continuous spaces indexed by discrete type — every later
section's algorithm (PA-DDPG, P-DQN, MP-DQN, H-PPO, Wolpertinger)
is a specific decomposition of this structure. Without the
formalism the algorithms look unrelated; with it they're variants
of "how should we condition continuous parameters on discrete
type?"

For the Simulator specifically, **every Action variant is a PAMDP
type**. Understanding §18.1–18.4 is what would let us go beyond
hand-discretized force levels, and §18.8 maps the formalism back
onto `Action::Strike { force }` and friends in actual code.

[Masson, Ranchod & Konidaris 2016] formalized the **PAMDP**:

> **Definition.** A **parameterized-action MDP** has action space
> $\mathcal{A} = \bigcup_{k=1}^{K} \{k\} \times \mathcal{X}_k$
> where $k$ is one of $K$ discrete action **types** and $\mathcal{X}_k$
> is the (possibly continuous) parameter space for type $k$.

So actions are pairs $(k, x)$ where $k \in \{1, \ldots, K\}$ and $x \in \mathcal{X}_k$.

Real-world examples are everywhere:

- Football: `Pass{angle, force}`, `Shoot{angle, force}`, `Dribble{direction}`.
- Robotics: `Move{velocity_vector}`, `Grip{force, finger_angles}`.
- The Simulator: `Step{direction}`, `Strike{force_level}`,
  `Vocalize{volume}`, all in
  [`action_template.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/action_template.rs).

### Three readings of $\mathcal{A} = \bigcup_k \{k\} \times \mathcal{X}_k$

The PAMDP definition is one line, but it captures a structural choice
that downstream algorithms each interpret differently. Three useful
readings:

**Reading 1 (algebraic, as a tagged union).** $\mathcal{A}$ is
literally a disjoint sum of $K$ continuous spaces — a tagged union
where $k$ is the tag and $x \in \mathcal{X}_k$ is the payload. This
is the same structure Rust calls an `enum` with payloads
(`Action::Strike { force: f32 }`), Haskell calls a sum type, and
category theory calls a coproduct $\coprod_k \mathcal{X}_k$. The
algebra immediately suggests the *shape* of any policy on it: a
mapping $\pi: \mathcal{S} \to \mathcal{A}$ factors into
*type selection* ($\mathcal{S} \to \{1,\ldots,K\}$) followed by
*parameter selection* given the type
($\mathcal{S} \times \{k\} \to \mathcal{X}_k$). Every PAMDP algorithm
in this chapter is a different choice of how to parameterize and
train those two factors. Reading 1 is what makes the algorithms feel
like variants of the same template rather than ad-hoc tricks.

**Reading 2 (decision-theoretic, as a two-stage decision).** From the
agent's point of view, choosing $a = (k, x)$ is a two-stage decision:
*which* action to take, then *how* to take it. This is not just a
notational rearrangement — it changes what counts as exploration. In
a flat discrete space, $\epsilon$-greedy explores by replacing the
$\arg\max$ with a uniform pick. In a PAMDP, you can be greedy in type
but exploratory in parameter ("definitely Strike, but try a new
force") or exploratory in type but greedy in parameter ("try a
random action with its current best-known parameter"). The two
explorations are *qualitatively different*: parameter-exploration is
local refinement (mostly cheap and informative), type-exploration is
global search (mostly expensive and high-variance). Real PAMDP
agents need *both*; flat discretization collapses them into one
noisy mechanism.

**Reading 3 (generalization-theoretic, as inductive bias).** The
PAMDP structure imposes a *generalization assumption*: nearby
parameters $x, x'$ in $\mathcal{X}_k$ should have nearby values
$Q(s, k, x) \approx Q(s, k, x')$. Discretization throws this
assumption away — `Strike{0.25}` and `Strike{0.30}` become unrelated
labels and the learner cannot share data between them. PAMDP methods
*reintroduce* the smoothness inductive bias via a continuous critic
$Q_\phi(s, k, x_k)$ that is differentiable in $x_k$. This is the
deepest reading: PAMDP isn't just a way to represent actions, it's a
*statement about which functions on actions are smooth*, which is
what makes a few-sample training tractable in the continuous part.
Different smoothness assumptions across $k$ (a quadratic landscape
in force, a sinusoidal landscape in direction) are what motivates
*per-type* parameter heads in P-DQN rather than a shared one.

### What the PAMDP formalism doesn't say

Three things worth being explicit about, because algorithms in
§18.2–18.5 silently assume them in ways the bare definition does
not require:

- **It doesn't say type sets are state-independent.** In real
  problems (chess, StarCraft, the Simulator), the set of *legal
  types* $\{k : (s, k, x) \text{ legal for some } x\}$ depends on
  state. The PAMDP definition is a Cartesian product; the practical
  setting is a *state-dependent* union. §18.7's action masking is
  what bridges the gap — it's not a side feature, it's the
  formalism's missing piece. Methods that assume static type sets
  (vanilla P-DQN) need an explicit mask layer to work in real PAMDP
  applications.
- **It doesn't say parameter spaces have to be continuous.**
  $\mathcal{X}_k$ could be discrete (e.g., `Plant{seed_type}` with
  six seeds) or mixed (`Step{direction, distance}` with discrete
  direction and continuous distance). The algorithms in this chapter
  optimize for *continuous* parameter spaces because that's where
  bucketing hurts most; for discrete-parameter PAMDPs the
  decomposition still gives clean structure, but algorithms like
  Wolpertinger (§18.5) become the right tool rather than P-DQN.
- **It doesn't say the type and parameter are statistically
  independent under the policy.** Most algorithms factor
  $\pi(k, x \mid s) = \pi_\text{type}(k \mid s) \cdot \pi_\text{param}(x \mid s, k)$
  for tractability, but the formalism itself allows arbitrary joint
  distributions. H-PPO's factored Gaussian is the *most common*
  parameterization; PA-DDPG's deterministic actor is a degenerate
  joint ($\pi$ is a point mass). For applications where types and
  parameters are genuinely entangled (e.g., a strike that is *both*
  hard *and* angled in a coordinated way), neither factorization is
  ideal — but that case rarely appears in practice and we ignore
  it.

These caveats matter because §18.8's mapping of the Simulator's
action space onto PAMDP assumes all three: state-dependent legality
(handled by `policy.rs`'s informal masking), continuous parameters
where they exist (`force`, `volume`), and factored policies (the
parameter head doesn't condition on a stochastic type sample).
Where these hold, PAMDP methods drop in; where they don't, the
chapter's machinery needs extra steps.

### Why discretization loses information

Bucketing `force ∈ [0, 1]` into 4 levels:

- **Q does not generalize across buckets.** Learning the value of
  `Strike{0.25}` does not inform `Strike{0.30}`.
- **Bucket boundaries are arbitrary.** Why 4 buckets? Why uniformly
  spaced?
- **Argmax loses precision.** "Best force is 0.27" gets quantized to
  "Light" (0.25) or "Medium" (0.5).
- **Sample complexity is multiplied** by the bucket count.

For 2 force levels: 2 templates, $2 \cdot d$ Q-storage. For continuous:
1 template, $1 \cdot d$ Q-storage *plus* a continuous-parameter
actor head. The continuous version is more sample-efficient if the
actor head can learn quickly.

<div id="ch18-discretization-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/discretization_regret/widget.js"></script>
Smooth Q*(force) bump overlaid with its B-bucket step approximation. The regret marker shows max Q* − Q̂(argmax bucket). Slide the true peak around so it lands in a trough between two bucket centroids and watch the regret jump. Crank B from 2 to 32 and the step approximation hugs the smooth curve — but every doubling of B doubles the action-space cost of tabular learning. §18.1's bucket-count-vs-precision tradeoff in one picture.

---

## 18.2 PA-DDPG

[Hausknecht & Stone 2016] published the first deep-RL PAMDP method.
Treat $(\text{one-hot type}, \text{params})$ as a single high-dim
continuous vector. Train via DDPG (deterministic actor + Q-critic).

Two tricks the paper added:

1. **Bounded gradients**: project the actor's output into the action
   space (one-hot constraint on type, box constraint on params).
   Important because DDPG's deterministic actor will otherwise propose
   actions outside the legal set.
2. **Inverting gradients**: when the actor saturates against a
   boundary, gradients are "inverted" to keep it inside.

Cons: combines DDPG's instabilities (Chapter 11) with PAMDP's
type-vs-param structure. Empirically finicky.

---

## 18.3 P-DQN and MP-DQN

[Xiong et al. 2018] proposed **P-DQN**, the more popular PAMDP
approach. Architecture:

- **Actor** $\mu_\theta(s) \to (x_1, \ldots, x_K)$: outputs one
  continuous parameter vector per action type.
- **Q-network** $Q_\phi(s, k, x_k) \to \mathbb{R}$: scores
  $(s, \text{type}, \text{params})$ triples.
- **Selection**: $k^{\star} = \arg\max_k Q_\phi(s, k, \mu _\theta(s) _k)$;
  apply $(k^{\star}, \mu _\theta(s) _{k^{\star}})$.

Train Q via DQN-style TD; train the actor via deterministic policy
gradient through the Q-critic (similar to DDPG).

### MP-DQN's bias fix

[Bester, James & Konidaris 2019] noticed: the Q-network sees parameter
vectors $x_k$ for *all* types, but only $x_{k^{\star}}$ is used. The
gradient flow includes "what if the parameters of a non-selected type
were different" — which biases Q toward configurations involving
parameters that won't be applied.

**MP-DQN's fix**: zero out the parameters of non-selected types before
computing Q. Each action evaluation only sees the parameters of *its
own type*. Eliminates the cross-type parameter bias.

Empirically MP-DQN > P-DQN on a range of PAMDPs. Adoption-grade
improvement.

### Architecture diagram

```
state s → [trunk] → param head → x_1 ... x_K
                  → Q head → Q(s, k, x_k) for each k
                  → argmax → (k*, x_{k*})
```

Trunk shared between actor and critic. Update Q via DQN; update actor
via DPG through the trunk.

<div id="ch18-mp-dqn-bias-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/mp_dqn_bias/widget.js"></script>
For the selected type k, the Q-slice over x_k at a chosen x_other. Green (true) is flat in x_other. Dashed red (P-DQN) tilts and shifts as you slide x_other — the cross-type gradient leakage is learning a bias that doesn't exist. Dotted blue (MP-DQN) zeros x_other before evaluating Q and tracks green exactly. The whole argument for MP-DQN over P-DQN is the residual gap between the red and the green.

---

## 18.4 H-PPO — hybrid policy gradient

[Fan et al. 2019] argued that the DDPG-derived methods (PA-DDPG, P-DQN,
MP-DQN) inherit DDPG's brittleness. The simpler alternative:

- **Discrete head**: softmax over action types via PPO.
- **Continuous head**: Gaussian over parameters via PPO.
- **Shared trunk**.

Train both heads with PPO. No deterministic actor; no Q-critic — just
PPO with an advantage function.

Pros: PPO's stability inherited; no DDPG-specific tricks needed. Cons:
on-policy (worse sample efficiency than P-DQN's off-policy replay).

If you are starting fresh on a PAMDP, H-PPO is a sensible default —
PPO's empirical robustness usually wins over P-DQN's sample
efficiency in practice.

<div id="ch18-hppo-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hppo_sampler/widget.js"></script>
Top scatter: N samples drawn from the factored policy p(k, x) = softmax(ℓ)_k · N(x; μ_k, σ²). Bottom: per-type marginal histograms. Slide a logit and the column for that type gets denser; slide μ_k and only its column shifts horizontally; slide σ and every column widens or narrows together. The picture *is* the factorisation — and it's the reason H-PPO trains stably while a joint head over (k, x) doesn't.

---

## 18.5 Large discrete spaces: Wolpertinger and Act2Vec

For huge discrete action spaces ($|\mathcal{A}| = 10^4$ or more —
think movie recommendation, large-vocabulary text), enumeration argmax
is the bottleneck.

### Wolpertinger ([Dulac-Arnold et al. 2015])

Three-step:

1. **Continuous proto-action.** Actor outputs $a_\text{proto} \in \mathbb{R}^d$
   in a learned embedding space.
2. **K-nearest neighbor.** Retrieve the $k$ discrete actions closest to
   $a_\text{proto}$ (in the embedding).
3. **Critic re-rank.** Score each of the $k$ candidates with the
   Q-critic; pick the best.

Effective complexity: $O(\log |\mathcal{A}|)$ for k-NN with proper
indexing, $O(k)$ for re-ranking. For $|\mathcal{A}| = 10^6, k = 5$:
~25 ops vs. $10^6$ for brute argmax.

### Act2Vec ([Tennenholtz & Mannor 2019])

word2vec for actions: learn embeddings from demonstration sequences.
Treat trajectories as "sentences," actions as "words"; train skip-gram.

Result: semantically-related actions cluster in embedding space.
Generalization across the cluster — learning "go-to-cafeteria" helps
"go-to-cafeteria-via-stairs."

### Application to the Simulator

The Simulator's 18 action keys is *not* a large action space. But if
you ever added a recipe-as-action layer (Chapter 16 §14.9 Angle 1) and
there were hundreds of recipes, Wolpertinger-style retrieval over a
recipe embedding becomes relevant. Speculative; not on roadmap.

<div id="ch18-wolpertinger-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/wolpertinger_knn/widget.js"></script>
Place the actor's proto-action on the 2-D embedding plane. The k nearest discrete actions light up — Wolpertinger's stage-1 retrieval. The critic heatmap behind them shows Q over the plane; the chosen action is the highest-Q among the k neighbours. At k = 1 the critic has no veto — you get whichever discrete action sits closest, even if Q there is bad. Larger k lets the critic override the actor's nearest-neighbour pick, at higher cost per decision.

---

## 18.6 Skill discovery: Option-Critic and DIAYN

Recap from Chapter 16. Both treat actions as **learned macros**.

- **Option-Critic** ([Bacon, Harb & Precup 2017]): end-to-end policy
  gradient on $(\pi_o, \beta_o)$.
- **DIAYN** ([Eysenbach et al. 2018]): mutual-information maximization
  to discover skills without rewards.

For the Simulator: recipes are *hand-authored* options today.
Option-Critic could *learn* recipe-internal policies from outcomes;
DIAYN could discover novel recipes without any predefined notion of
what to do. Both are larger refactors than this chapter sketches; see
Chapter 16 §14.9 for the details.

---

## 18.7 Action masking

State-dependent legal-action sets are a special case of PAMDP: action
*availability* depends on state. Examples:

- Chess: only legal moves are available.
- The Simulator: `Consume` requires food in reach.
- StarCraft: many actions require resource thresholds.

The naive approach is to penalize illegal actions in the reward
function. This is hacky and wastes exploration on actions that should
have been ruled out.

**Action masking**: set the logits of illegal actions to $-\infty$
before softmax. The policy puts zero probability on them.

> **Theorem ([Huang & Ontañón 2020]).** Action masking is a valid
> policy-gradient update on the constrained MDP (where illegal actions
> are removed from the action space at each state). The masked policy's
> gradient equals the unmasked policy's gradient restricted to legal
> actions.

Practical implications:

- No reward-penalty hacks needed.
- The agent never wastes exploration on illegal actions.
- Implementation is trivial (5 lines of mask + softmax).
- Empirically dominates penalty-based approaches when the
  illegal-action fraction is large (>50%).

### Application to the Simulator

The flat policy in
[`policy.rs`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) already
filters candidate actions before scoring (only enumerates *applicable*
templates, e.g., recipes whose preconditions match, Consume only when
food is perceived). This is *informal* action masking. A formal
softmax-policy version (Chapter 12 §10.5) would benefit from explicit
masking — but the current argmax setup already excludes illegal
actions, so the benefit is zero today.

<div id="ch18-action-mask-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/action_mask_softmax/widget.js"></script>
Three softmax outputs over the same 6 logits: unmasked, hard-masked (illegal logits → −∞), and penalty-shaped (illegal logits get −P). Toggle some actions illegal and slide P from 0 to 50. Penalty starts as "no constraint" and approaches "hard mask" asymptotically, but residual mass always leaks onto illegal actions when P is finite. Hard masking is exact and free at inference time; penalty shaping is a smooth knob that never quite gets there.

---

## 18.8 The Simulator's action space, dissected

From [`action_template.rs:32-50`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/action_template.rs)
and [`policy.rs:169-177`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs):

| Template | Parameters | # Variants | Notes |
|---|---|---|---|
| `Step` | direction (N/E/S/W) | 4 | The only navigation primitive |
| `Strike` | force level | 2 | Light, Hard. Discrete bucketing |
| `Orient` | — | 1 | |
| `Grasp` | — | 1 | Target-conditioned via perception |
| `Release` | — | 1 | |
| `Equip` | — | 1 | |
| `Unequip` | — | 1 | |
| `Place` | — | 1 | |
| `Bind` | — | 1 | |
| `Consume` | — | 1 | Target-conditioned via perception |
| `Harvest` | — | 1 | |
| `Plant` | — | 1 | |
| `Vocalize` | volume | 1 | Currently treated as a single template; volume not parameterized in learner |
| `Gesture` | — | 1 | |
| `Wait` | — | 1 | |
| `Idle` | — | 1 | |
| **Total templates** | | **16** | |
| **Total learner keys** | | **18** | Step has 4 directional + Strike has 2 force = 16 + 2 |

### Parameterization opportunities

- **`Strike{force}`**: currently 2 buckets (Light, Hard). MP-DQN
  parameterization would learn continuous force.
- **`Vocalize{volume}`**: currently a single template. Parameterizing
  by volume would let the agent learn quiet vs. loud vocalizations
  contextually.
- **`Step{direction}` is direction-discrete and naturally so.** Not a
  candidate for parameterization (north is north).
- **Recipe parameters**: some recipes have implicit parameters
  (`Plant{which_seed_type}`). These could be exposed and learned.

---

## 18.9 Project tie-in

### Pivot proposal: Strike and Vocalize → parameterized

Current state: `Strike{Light}` and `Strike{Hard}` are two
[`WellKnownTemplate`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/action_template.rs)
ids. The learner has separate Q-tile-coder slots for each.

Pivot: introduce a single `Strike` template with a continuous
`force ∈ [0, 1]` parameter. Add an MP-DQN-style parameter actor head
that predicts a force per state. Modify
[`policy.rs:474`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) to
compute the Strike's score with the actor-predicted force:
```rust
let force = strike_actor(s);
let q_strike = q_learner.q(s, "Strike", force);
let score_strike = score_action(s, Strike { force }) + 0.5 * q_strike + recipe_bonus;
```

Cost:
- New trainable component: a small MLP or tile-coded regressor that
  outputs `force` from state. Maintained per-agent.
- DPG-style gradient update for the actor: `∂Q/∂force * ∂force/∂theta`.
  Requires Q-differentiability in force — tile-coding gives a
  step function, so the gradient is zero almost everywhere. Solution:
  use a linear interpolation within each tile (or replace force's
  tile coder with a small smooth function approximator just for that
  dimension).

Risk:
- The deterministic-RNG canary needs to bind force-actor's output
  to a deterministic seed.
- The Q-update under continuous force needs a new tile-coding scheme
  (or a smooth FA) for that dimension.
- Validation suite recalibration: `learning_threat_response.rs` uses
  Strike-class actions; the bucketing change will shift Q values.

Benefit:
- Sample efficiency: one Strike learner shared across all force levels.
- Expressiveness: continuous force enables fine-grained behavior.
- Modular: applies to any future parameterized actions (Vocalize,
  partial Gesture, recipe-step parameters).

### Where action masking already exists informally

[`policy.rs:401-449`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) —
candidate enumeration:

```rust
// (pseudocode summary)
let mut candidates = baseline_primitives(s);
candidates.extend(perceived_interactions(s));  // Consume, Grasp only if perceived
candidates.extend(recipe_head_steps(s));       // only if precondition met
candidates.retain(|c| recently_failed(c));
```

This is action masking by construction — illegal actions never enter
the scoring loop. Formal proof of validity follows
[Huang & Ontañón 2020] but the practical implementation is in place.
If we ever switch to a softmax policy (Chapter 12 §10.5), the same
filter would be applied before softmax, automatically masking.

### What test exercises this chapter

[`threat_response.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/threat_response.rs)
exercises Strike actions (the agent learns to flee or strike depending
on threat valence). With continuous force, the metric
`q_flee_margin` would be replaced by a "best force given threat" metric.

<div id="ch18-strike-force-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/strike_force/widget.js"></script>
Three Strike{force} architectures on the same Q*(force | valence): the 2-bucket baseline, MP-DQN's continuous head, and H-PPO's Gaussian head. Top panel: each method's learned force(v) curve; the bucket version is a step, the continuous methods are smooth. Bottom panel: per-v regret. Drag the learning-progress slider t to watch each method evolve from "flat 0.5 force" to its asymptote. The §18.9 pivot — why two buckets aren't enough for Strike — is the visible regret gap at every v that isn't lucky enough to sit on a bucket centroid.

---

## 18.10 Exercises

1. **(PA-DDPG on half-cheetah-discrete.)** Modify the
   HalfCheetah-v3 environment to have 3 discrete action types (move
   forward, move backward, jump), each with continuous parameter.
   Implement PA-DDPG; train; report return.

2. **(P-DQN vs. MP-DQN bias.)** Construct a 2-type PAMDP where the
   cross-type parameter bias is large. Train P-DQN and MP-DQN; report
   the gap.

3. **(H-PPO on bipedal walker.)** Implement H-PPO on a hybrid version
   of BipedalWalker (action: type ∈ {step, jump} × continuous force).
   Compare wall-clock and sample-complexity to P-DQN.

4. **(Wolpertinger on text generation.)** Use Wolpertinger on a
   500-action recommendation problem (synthetic). Embedding-based
   k-NN with $k = 5$. Compare to exhaustive argmax.

5. **(Action masking proof.)** Walk through the [Huang & Ontañón 2020]
   theorem. Identify why the masked policy gradient equals the
   unmasked-but-restricted gradient.

6. **(Project — parameterized Strike sketch.)** Pseudo-code the
   code changes to replace `Strike{Light, Hard}` with `Strike{force}`
   continuous. Detail: how the actor is parameterized, how the
   Q-learner handles continuous force, how the deterministic canary
   is preserved.

7. **(Vocalize{volume} sketch.)** Same for `Vocalize{volume}`. What
   downstream systems use `Vocalize`? Which would need recalibration?

8. **(Multi-recipe action space.)** Suppose the Simulator had 1000
   recipes. Sketch a Wolpertinger-style retrieval architecture for
   recipe selection: embedding source, retrieval mechanism, re-rank
   step.

---

## 18.11 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Bacon, Harb & Precup 2017] — Option-Critic (§18.6).
- [Bester, James & Konidaris 2019] — MP-DQN (§18.3).
- [Dulac-Arnold et al. 2015] — Wolpertinger (§18.5).
- [Eysenbach et al. 2018] — DIAYN (§18.6).
- [Fan et al. 2019] — H-PPO (§18.4).
- [Hausknecht & Stone 2016] — PA-DDPG (§18.2).
- [Huang & Ontañón 2020] — action masking (§18.7).
- [Masson, Ranchod & Konidaris 2016] — PA-MDP framework (§18.1).
- [Tennenholtz & Mannor 2019] — Act2Vec (§18.5).
- [Xiong et al. 2018] — P-DQN (§18.3).

---

## 18.12 Closing thoughts

You have reached the end of the textbook. By this point, you should be
able to:

1. **Read modern RL papers** and place them in their lineage — Sutton's
   tabular intuition through Mnih's DQN through Schulman's PPO through
   Schrittwieser's MuZero.
2. **Diagnose pathologies** in your own RL systems. The deadly triad,
   the Q-bias bootstrap, the $\gamma^{500}$ catastrophe, the set-as-vector
   problem — these are not unique to the Simulator; you will meet them
   again.
3. **Implement everything** from tabular Q-learning to PPO to RUDDER
   to DIAYN.
4. **Recognize when** a problem needs a model, hierarchy, advantage
   learning, pseudo-counts, or parameterized actions.
5. **Apply the toolkit** to the Simulator's specific open problems
   (Q-bias fix; L-suite long-horizon; per-block tile generalization;
   set-as-vector for episodic memory).

The next step is **doing the work**:

- Pick one of the open project problems (Q-bias, L-suite, navigation,
  episodic-memory permutation invariance, parameterized Strike).
- Apply what you've learned. Most pivots have been sketched in the
  per-chapter project tie-ins.
- Read the papers cited in the relevant chapter as you go. Citations
  are in [`bibliography.md`](bibliography.md).
- Write up your findings as a new design proposal in
  [`docs/proposals/`](https://github.com/falahat/simulator/blob/main/docs/proposals/).

### Acknowledging the field's limits

Reinforcement learning is one of the most theoretically rich and
practically frustrating fields in machine learning. The theory is
elegant: Banach's theorem, the Bellman optimality equations, the
Tsitsiklis–Van Roy convergence proof. The practice is full of edge
cases and pathologies you have to develop intuition for: the deadly
triad, reward hacking, the $\gamma^{500}$ catastrophe, the
$w_\text{alive}$ bootstrap.

The Simulator is a particularly useful laboratory because it surfaces
these pathologies in ways that benchmarks rarely do. Atari benchmarks
hide the deadly triad behind clean, dense rewards; the Simulator has
a homeostatic reward with $w_\text{alive}$ and a 500-tick credit gap,
and the math of those choices is *unavoidably visible*.

Good luck. See you in the proposals folder.

---

**Back to:** [Table of Contents](00_index.md)

# Chapter 10 — Linear Function Approximation and Tile Coding

> **Prerequisites:** Chapter [6](08_temporal_difference_learning.md)
> (TD / Q-learning) and [7](09_eligibility_traces.md) (eligibility traces,
> in tabular form). Chapter [1](01_linear_algebra.md)
> (dot products, eigenvalues, PSD).

> **Citations:** in-line citations use `[Author Year]` keyed to the
> master [bibliography](bibliography.md). This chapter draws on
> [Sutton & Barto 2018, Ch. 9–10] (semi-gradient methods, tile coding),
> [Tsitsiklis & Van Roy 1997] (linear TD convergence), [Albus 1975]
> (CMAC original), [Sutton 1996] (sparse coarse coding for RL), and
> [Zaheer et al. 2017] (Deep Sets, for the set-as-vector problem).

> **Learning objectives:**
> 1. Explain *concretely* why a 267-dim continuous observation kills
>    tabular methods.
> 2. Derive the linear semi-gradient TD update and read it line-by-line.
> 3. State the Tsitsiklis–Van Roy convergence theorem and its
>    on-policy-only caveat.
> 4. Build, on paper, a tile-coding feature vector from a 2-D continuous
>    state. Map this construction onto the Simulator's 16-tilings ×
>    0.25-width × $2^{16}$-IHT setup.
> 5. Recognize and diagnose the *joint-tiling collapse* and the
>    *set-as-vector* problems in the Simulator's observation pipeline.

## Why this chapter exists

So far every algorithm has assumed a finite state space small enough to
store one Q-value per (state, action) pair. That is fine for
gridworlds and toy games. It is hopeless for anything real — including
this Simulator.

This chapter is the bridge from "tabular RL works in principle" to "tabular
RL does not work on a 267-dim continuous observation; here is what
replaces it." The bridge is **function approximation**: represent the
value function as a parameterized $V(s; \theta)$ and learn $\theta$.

We will spend most of the chapter on the simplest case, **linear
function approximation**, because (i) it has a complete convergence
theory, (ii) it is what most RL classical results assume, and (iii) the
specific form the Simulator uses — *tile coding* — is a particularly
elegant linear FA. Chapter 11 generalizes to nonlinear (deep) FA, where
the theory is much weaker and the practice much more dangerous.

## Table of contents

- [8.1 Why tables fail at scale](#81-why-tables-fail-at-scale)
- [8.2 Linear function approximation: the form](#82-linear-function-approximation-the-form)
- [8.3 The semi-gradient TD update](#83-the-semi-gradient-td-update)
- [8.4 Convergence: Tsitsiklis–Van Roy](#84-convergence-of-linear-td)
- [8.5 Choosing features](#85-choosing-features)
- [8.6 Tile coding from scratch](#86-tile-coding-from-scratch)
- [8.7 Hashing tiles: the Index Hash Table](#87-hashing-tiles-the-index-hash-table-iht)
- [8.8 The Simulator's tile coder, in detail](#88-the-simulators-tile-coder-in-detail)
- [8.9 The joint-tiling collapse](#89-the-joint-tiling-collapse)
- [8.10 The set-as-vector problem](#810-the-set-as-vector-problem)
- [8.11 Linear FA vs. neural FA — a comparison](#811-linear-fa-vs-neural-fa--a-comparison)
- [8.12 Project tie-in](#812-project-tie-in)
- [8.13 Exercises](#813-exercises)
- [8.14 References](#814-references-cited-in-this-chapter)
- [8.15 Further reading](#815-further-reading)

---

## 8.1 Why tables fail at scale

A **table** is the most literal value-function representation: one stored
real number per (state, action) pair. For a finite MDP with
$|\mathcal{S}|$ states and $|\mathcal{A}|$ actions, the table has
$|\mathcal{S}| \cdot |\mathcal{A}|$ entries.

The problem is **combinatorial blow-up** in the state space. The
Simulator's observation has 8 contiguous blocks: drives (14), body (8),
emotions (8), perception (64), world (5), ambient history (8),
consolidated memory (16), and episodic memory (144) — 267 reals
([`observation.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/observation.rs)).
If we naively discretized each real into just $b = 10$ bins:

| Block | Dim | Bins per dim | States per block |
|---|---|---|---|
| Drives | 14 | 10 | $10^{14}$ |
| Body | 8 | 10 | $10^8$ |
| Emotions | 8 | 10 | $10^8$ |
| Perception | 64 | 10 | $10^{64}$ |
| World | 5 | 10 | $10^5$ |
| Ambient hist | 8 | 10 | $10^8$ |
| Consolidated | 16 | 10 | $10^{16}$ |
| Episodic mem | 144 | 10 | $10^{144}$ |
| **Total $\lvert\mathcal{S}\rvert$** | 267 | | $10^{267}$ |

$10^{267}$ is more than the number of particles in the observable
universe ($\sim 10^{80}$). Even storing one bit per state is impossible.
You can lower $b$, drop blocks, or hash — none rescue the table.

### Try it: tabular blow-up calculator

<div id="ch8-tabular-blowup-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/tabular_blowup/widget.js"></script>

Slide $n$ from 10 toward 30 with $b = 10$ and watch the log-state-count
curve cross the dashed "10^80 atoms" line near $n = 27$. The
per-block bar plot reproduces the §8.1 table for the chosen $b$;
even with $b = 2$ (binary discretisation) the 267-dim total stays
above $10^{75}$. No table fits.

But the table has a deeper flaw than just storage: **no generalization**.
Even if you could store $10^{267}$ values, you would still need to *visit*
each one to learn its Q-value. The agent never sees the same state twice
(continuous observations are almost surely distinct). Without
generalization across nearby states, learning is impossible.

### The two requirements function approximation must satisfy

1. **Compact parameterization.** Store $O(n)$ parameters for some
   tractable $n$, not $O(|\mathcal{S}|)$.
2. **Generalization.** Updating the value of $s$ must affect the value
   of nearby states $s'$. Otherwise we are still effectively tabular,
   just with a hash collision.

The simplest construction satisfying both is **linear function
approximation**: a fixed feature mapping $\phi$ plus a learned weight
vector $\theta$, with predicted value $V(s) = \theta^{\top} \phi(s)$.

---

## 8.2 Linear function approximation: the form

> **Definition.** A **feature map** is a function
> $\phi: \mathcal{S} \to \mathbb{R}^d$ (or
> $\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^d$ for Q-values).
> A **linear value function** is
>
> $$
> V(s; \theta) = \theta^{\top} \phi(s) = \sum_{i=1}^{d} \theta_i \phi_i(s),
> $$
>
> with a learnable weight vector $\theta \in \mathbb{R}^d$.

The key choice: $d$, the **feature dimension**, is fixed *a priori* — it
does not grow with the state space. We trade "one parameter per state"
for "$d$ parameters" plus the inductive bias encoded in $\phi$.

Generalization is automatic: an update to $\theta_i$ changes the value of
*every* state $s$ for which $\phi_i(s) \neq 0$. The choice of $\phi$ is
the choice of what "nearby" means.

### Try it: train one point, see the neighbourhood light up

<div id="ch8-generalization-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/generalization/widget.js"></script>

A 2-D toy tile coder starts with $\theta = 0$. One semi-gradient step
at the train point lifts $V$ to the target there — and the heatmap
shows the same update spilling into every nearby state that shares an
active tile. Widen $w$ to spread the halo; raise $T$ to smooth its
edges; the slice plot below confirms the halo radius is roughly $w$
on a single axis. With $T = 1$ the halo is a clean square of side
$w$ (one tile). With $T = 16$ the boundary becomes the staircase of
overlapping tilings — the same generalisation surface §8.6's worked
example pictures with three-of-four shared tiles.

### Action-conditioned Q-values

For Q-learning we want $Q(s, a; \theta)$. Three common encodings:

1. **Separate $\theta_a$ per action.** $Q(s, a) = \theta_a^{\top} \phi(s)$.
   Used when $|\mathcal{A}|$ is small and discrete. Storage $d \cdot |\mathcal{A}|$.
2. **Action-conditioned features.** $Q(s, a) = \theta^{\top} \phi(s, a)$,
   where $\phi$ knows about $a$. Used for parameterized actions
   (Chapter 20) or when action space is large.
3. **Action-keyed lookup.** Store $Q$ in a hash table whose key is
   $(\text{tile\_indices}(s), \text{action\_key}(a))$. This is what the
   Simulator does.

The Simulator uses option (3) for tile coding because the tile machinery
already produces a sparse hash. Each (observation, action) becomes a set
of "active tile" hash indices; the Q-value is the sum of weights at
those indices. This is mathematically equivalent to option (2) with
$\phi$ being the **indicator vector** of active tiles, but stored
sparsely.

### Tabular methods are a special case

If $\phi(s) = e_s$ (the one-hot indicator of state $s$), then
$V(s) = \theta_s$ and we recover the tabular case. Tables are LFA with
the worst possible features: no generalization (each $\phi_i$ activates
on exactly one state). Choosing good features means choosing a $\phi$
that *does* generalize — features that fire on more than one state.

---

## 8.3 The semi-gradient TD update

### Why this is the bridge from §8.2 to all of deep RL

§8.2 set up the linear approximator $V(s; \theta) = \theta^\top \phi(s)$.
That's just a parameterization — it doesn't tell us *how* to find
the right $\theta$. This section's job is the learning rule:
given a sequence of transitions $(s, r, s')$, how should $\theta$
change?

The "obvious" answer is "do gradient descent on squared error to
$V^\pi(s)$." That fails for the reason all of model-free RL has to
solve: we don't *know* $V^\pi(s)$. The TD target $r + \gamma V(s'; \theta)$
substitutes — but it's *itself a function of $\theta$*, which makes
the gradient subtle. The **semi-gradient** trick is the
pragmatically-correct shortcut: treat the target as fixed when
differentiating. The cost is that you've sacrificed a clean
gradient-descent guarantee; the gain is that the algorithm actually
works (and is exactly what every deep RL codebase ships).

This rule — $\theta \leftarrow \theta + \alpha \delta \phi(s)$ —
is the linear-FA core that scales up to neural Q-learning (Chapter
11), continuous control (Chapter 12), and the Simulator's actual
tile-coded value function. Type it once and it's burned in.

### Setting up the loss

We want $V(s; \theta) \approx V^{\pi}(s)$. The squared-error loss at
one sample is

$$L(\theta) = \tfrac{1}{2} \big(V^{\pi}(s) - V(s; \theta)\big)^2.$$

The factor of $\tfrac{1}{2}$ is there so the chain rule gives a
clean form without a stray $2$. (Cosmetic — same fixed points
either way.) The gradient with respect to $\theta$ is

$$\nabla _\theta L = -\big(V^\pi(s) - V(s; \theta)\big) \cdot \nabla _\theta V(s; \theta).$$

### The TD target substitution

We don't know $V^\pi(s)$. We do know one *sample* of the Bellman
RHS, namely $r + \gamma V^\pi(s')$ — and we don't know $V^\pi(s')$
either, so we substitute our own current estimate:

$$V^\pi(s) \;\text{(unknown)} \;\longrightarrow\; r + \gamma V(s'; \theta) \;\text{(TD target)}.$$

This substitution has *two consequences*, both worth naming:
1. The target is now a sampled-and-bootstrapped quantity, not a
   deterministic ground truth. The sample noise gets absorbed by
   the step-size discipline (Robbins-Monro, Chapter 8 §6.1).
2. The target now *depends on $\theta$* through $V(s'; \theta)$.
   If we differentiate honestly, the gradient picks up an extra
   term from that dependence.

Step 2 is where semi-gradient diverges from full gradient.

### The semi-gradient trick

The **semi-gradient** rule: treat the target as a constant when
differentiating. The "exact" gradient of the substituted loss would
be

$$\nabla _\theta L = -\big(r + \gamma V(s'; \theta) - V(s; \theta)\big) \cdot \big[ \gamma \nabla _\theta V(s'; \theta) - \nabla _\theta V(s; \theta) \big].$$

The first bracketed factor is the TD error $\delta$. The second
factor has two pieces: a gradient through the *prediction* (good,
this is what we want to move) and a gradient through the *target*
(bad, this is what causes the chase-your-tail problem). The
semi-gradient simply drops the target term:

$$\nabla _\theta L \approx -\delta \cdot \nabla _\theta V(s; \theta). \qquad \text{(semi-gradient)}$$

"Semi" because it's half the gradient — the prediction half, not
the target half.

**For linear FA**, $V(s; \theta) = \theta^\top \phi(s)$ so $\nabla _\theta V(s; \theta) = \phi(s)$. The
update becomes

> **Linear semi-gradient TD(0)**
>
> $$
> \theta \leftarrow \theta + \alpha \cdot \delta_t \cdot \phi(s_t),
> $$
>
> with TD error $\delta_t = r _{t+1} + \gamma\, \theta^{\top} \phi(s _{t+1}) - \theta^{\top} \phi(s_t)$.

Read it: scale the feature vector by the TD error, take a step.
This is the simplest learning rule with both compactness and
generalization. **The Simulator's `td_update()`** in
[`value_function.rs:126-146`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/value_function.rs)
is exactly this rule for Q-learning, with $\phi$ realized by tile
coding (§8.6).

### Why "semi" is not actually pathological — but is dangerous

The full gradient would converge to the **least-squares
projection** of $V^\pi$ onto $\text{span}(\phi)$ — the closest
representable function to the true value-function. That sounds
right, but in practice it isn't: the full-gradient method is called
**residual gradient** and it has badly-conditioned dynamics that
slow learning dramatically.

The semi-gradient converges (when it converges) to a *different*
fixed point: the **projected Bellman fixed point**, defined by
$\Pi T^\pi \theta = \theta$ where $\Pi$ is the projection onto
feature span. The semi-gradient fixed point is the right answer
for *control* — it's what value iteration would converge to if you
restricted V to live in $\text{span}(\phi)$.

So why is semi-gradient called "dangerous"? Because while
*on-policy* TD with linear features converges (the theorem in §8.4
below), the same algorithm can *diverge* off-policy with linear
features — the famous **Baird counterexample**, where weights
spiral to infinity even with bounded rewards and a stable target
policy. We name this combination (function approximation +
bootstrapping + off-policy) the **deadly triad**; Chapter 17 is
its whole-chapter treatment.

The takeaway: semi-gradient is what works in practice, and the
proofs cover most cases, but you need to know where it fails. The
widget below shows the divergence in action.

### Try it: semi-gradient vs full-gradient on a 2-D linear FA toy

<div id="ch8-semi-vs-full-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/semi_vs_full/widget.js"></script>

A 3-state Baird-style off-policy chain with 2-D linear features and
true weights $w^\star = 0$ (the orange diamond at the origin). Each
polyline is the trajectory of $(w_1, w_2)$ under one update rule.
Semi-gradient (red) spirals outward — it diverges off-policy, exactly
as [Baird 1995] predicted. Full-gradient (green) drops the deadly-triad
divergence by differentiating through the bootstrap target as well, and
stays bounded. The cost is paid elsewhere: full-gradient TD converges
to the wrong fixed point in general, which is why every modern RL
codebase (the Simulator included) ships the semi-gradient flavour.

### What this rule doesn't say

- **It's not gradient descent on any loss.** Despite the name, the
  semi-gradient direction isn't the gradient of *any* scalar
  objective. It's a fixed-point iteration in disguise. The
  consequence: standard gradient-descent guarantees (convergence
  to a local minimum, monotone loss decrease) don't apply directly.
- **It assumes the features are *good enough*.** If
  $V^\pi \notin \text{span}(\phi)$ — i.e. no linear combination of
  features represents $V^\pi$ exactly — semi-gradient TD finds the
  projected Bellman fixed point, which can be arbitrarily far from
  the least-squares projection. Section 8.5 (feature engineering)
  is the practitioner's response.
- **It's per-sample, not minibatched.** The update uses one
  transition at a time. Deep RL uses minibatches to lower variance,
  which is mechanically the same rule averaged over $B$ samples —
  Chapter 11 §9.5 (PER and minibatches).

### Try it: semi-gradient vs full-gradient on a 2-D linear FA toy

<div id="ch8-semi-vs-full-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/semi_vs_full/widget.js"></script>

A 3-state Baird-style off-policy chain with 2-D linear features and
true weights $w^\star = 0$ (the orange diamond at the origin). Each
polyline is the trajectory of $(w_1, w_2)$ under one update rule.
Semi-gradient (red) spirals outward — it diverges off-policy, exactly
as [Baird 1995] predicted. Full-gradient (green) drops the deadly-triad
divergence by differentiating through the bootstrap target as well, and
stays bounded. The cost is paid elsewhere: full-gradient TD converges
to the wrong fixed point in general, which is why every modern RL
codebase (the Simulator included) ships the semi-gradient flavour.

---

## 8.4 Convergence of linear TD

Here is the foundational result of linear function approximation in RL:

> **Theorem [Tsitsiklis & Van Roy 1997].** Suppose:
> 1. Features $\phi(s)$ are bounded and linearly independent across
>    states.
> 2. The behavior policy is the same as the policy being evaluated
>    (**on-policy**).
> 3. Step sizes satisfy Robbins-Monro.
>
> Then linear semi-gradient TD converges almost surely to a unique
> fixed point $\theta^{\star}$ satisfying $\Phi \theta^{\star} = \Pi T^{\pi} \Phi \theta^{\star}$,
> where $\Pi$ is the projection onto $\text{span}(\phi)$ under the
> stationary distribution.

In words: with linear features, on-policy data, and Robbins-Monro step
sizes, TD converges. And the result is the closest linear approximation
to $V^{\pi}$ achievable in $\text{span}(\phi)$, in a precise weighted
sense.

**Three caveats:**

- **On-policy is essential.** Off-policy linear TD can diverge — Baird's
  counterexample (Chapter 17). This is one leg of the deadly triad.
- **The fixed point is the projection, not $V^{\pi}$.** If your features
  cannot express $V^{\pi}$, you get the best linear approximation under
  the on-policy state-visit measure. The **projection error**
  $\|V^{\pi} - \Pi V^{\pi}\|$ is irreducible without changing $\phi$.
- **The constant-step-size caveat from §1.5 still applies.** Practical
  $\alpha$ gives an $O(\sqrt{\alpha})$-radius neighborhood, not a point.

These three facts shape the entire chapter: pick features that span the
true value functions you care about; stay on-policy when you can; live
with the neighborhood when you cannot decay $\alpha$.

---

## 8.5 Choosing features

The features $\phi$ are the inductive bias of linear FA. The classical
options, from worst to best for the Simulator's purposes:

### Polynomial features

$\phi(s) = (1, s, s^2, s s', s'^2, \ldots)$. Conceptually simple.
Catastrophic in high dimensions: degree $k$ polynomials in $n$
variables have $\binom{n+k}{k}$ terms. For $n = 267$, $k = 3$: over
$2.6 \times 10^6$ features. Numerical conditioning is also awful
(high powers amplify scale).

### Radial Basis Functions (RBF)

$\phi_i(s) = \exp(-\|s - c_i\|^2 / (2 \sigma^2))$ for centers $c_i$.
Smooth and intuitive. Two failure modes: **curse of dimensionality** —
covering 267-D space with RBF centers needs exponentially many; and
**bandwidth sensitivity** — too narrow $\sigma$ gives no generalization,
too wide gives no discrimination.

### Fourier basis

$\phi_i(s) = \cos(\pi c_i^{\top} s)$ for integer coefficient vectors
$c_i$ [Konidaris, Osentoski & Thomas 2011]. Works well on smooth value
functions; empirically beats RBF on classic benchmarks. Still suffers
exponential growth in dimension if you want orders >2.

### Proto-value functions (PVFs)

Eigenvectors of the graph Laplacian of the state-transition graph
[Mahadevan & Maggioni 2007]. Theoretically beautiful (these are the
"natural modes" of the state-space geometry); expensive to compute
because they require knowing the graph.

### Bellman Error Basis Functions (BEBFs)

Construct features greedily to reduce Bellman error
[Parr et al. 2007]. Elegant, optimal in a precise sense, hard to
implement efficiently.

### Tile coding (CMAC)

Sparse, hashable, distance-based. **What the Simulator uses.** Detailed
construction in §8.6. The reason it wins for the Simulator: it scales
gracefully to high dimensions because it tiles **per-block**, and the
sparse hashing makes updates $O(\text{tilings})$ regardless of nominal
dimensionality.

| Method | Smoothness | Scales to high dim | Sparse updates | Compute |
|---|---|---|---|---|
| Polynomial | $C^\infty$ | No | No | Cheap |
| RBF | $C^\infty$ | No | No | Moderate |
| Fourier | $C^\infty$ | Poorly | No | Cheap |
| PVF | Smooth | Needs graph | No | Expensive |
| BEBF | Adaptive | Yes | No | Expensive |
| **Tile coding** | Piecewise const | **Yes** | **Yes** | **Cheap** |

The trade-off for tile coding is **piecewise constancy**: the
approximation is flat inside each tile and jumps at boundaries. This is
fine for Q-learning (the policy only cares about argmax across actions),
poor for tasks needing smooth derivatives of the value.

---

## 8.6 Tile coding from scratch

> **Tile coding** [Albus 1975, "CMAC"; popularized for RL by Sutton 1996]
> tiles state space with **overlapping grids**, each grid offset from
> the others. Each grid identifies a "tile" containing the state; the
> feature vector is the indicator vector of active tiles across grids.

The construction:

1. Pick a number of **tilings** $T$ (the Simulator uses $T = 16$).
2. Pick a **tile width** $w$ per dimension (Simulator: uniform $w = 0.25$).
3. Pick offsets $o_t \in [0, w)^n$ for $t = 1, \ldots, T$ such that the
   offsets are spread asymmetrically across dimensions (Sutton's rule:
   $o_t = t/T \cdot (1, 3, 5, \ldots, 2n-1) \mod w$).
4. For each tiling $t$, the tile containing state $s$ has integer index
   $i_t(s) = \lfloor (s - o_t) / w \rfloor$ (componentwise).
5. The feature vector $\phi(s)$ has $T$ active components (one per
   tiling), each pointing to a slot identified by $(t, i_t(s))$.

The feature vector is **sparse**: exactly $T$ of its components are 1,
the rest are 0. So $\theta^{\top} \phi(s)$ is a sum of $T$ stored
weights — $O(T)$ per evaluation, not $O(d)$. With $T = 16$, that is 16
table lookups regardless of the nominal state-space size.

### What the parameters do

- **Tile width $w$ (small)**: high discrimination (nearby points fall in
  different tiles), low generalization (an update to one tile barely
  affects others).
- **Tile width $w$ (large)**: low discrimination (many states share a
  tile), high generalization.
- **Number of tilings $T$**: more tilings = smoother approximation (more
  weights average per state), no change in discrimination at the
  individual-tile level, $T \times$ more storage.

The **resolution** of the approximation is roughly $w / T$: features are
piecewise constant on tiles of width $w$, but across $T$ overlapping
tilings the effective resolution refines by $T$.

> **Worked example.** Continuous 2-D state space, $s \in [0, 1]^2$. Tile
> width $w = 0.25$, four tilings ($T = 4$), offsets
> $o_1 = (0, 0), o_2 = (0.0625, 0.1875), o_3 = (0.125, 0.0625), o_4 = (0.1875, 0.125)$.
> The state $s = (0.40, 0.30)$ activates four tiles, one per tiling. A
> nearby state $s' = (0.42, 0.31)$ shares **three** of those four tiles
> (only the one whose offset puts a boundary between $s$ and $s'$
> differs). So an update at $s$ moves the value of $s'$ by 3/4 as much
> — that's generalization, built in geometrically.

### The asymmetry matters

If all $T$ offsets were aligned, the tilings would all agree on tile
boundaries — you would have $T$ copies of the same coarse partition,
gaining nothing. The asymmetric Sutton offsets ensure each pair of
tilings has *different* boundaries in *different* places, so the
intersection of tile-activation patterns is much finer than any single
tiling. The Simulator implements this offset pattern in
[`tile_coding.rs:108-119`](https://github.com/falahat/simulator/blob/main/crates/engine/rl_core/src/tile_coding.rs).

### Try it: Sutton offsets vs aligned offsets

<div id="ch8-sutton-offsets-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/sutton_offsets/widget.js"></script>

Left: tile boundaries under the Sutton-style asymmetric offsets used
by the Simulator. Right: the same $T$ tilings stacked with zero offset.
Each tiling is one colour. Aligned tilings draw $T$ copies of the
same partition — the unique-cells readout matches $\lceil 1/w\rceil^2$
no matter how many tilings you stack. Sutton offsets interleave the
boundary lines so the joint refinement reaches the $T \times T$ grid
of side $w/T$ that the chapter's "effective resolution" claim
predicts.

### Try it: tile coding on a 2D state space

<div id="ch8-tile-coding-widget"></div>
<script type="module">
  import init, { start } from './widgets/tile_coding/pkg/widget_tile_coding.js';
  await init();
  start('ch8-tile-coding-widget');
</script>

Drag the white dot around the unit square. Each coloured rectangle is
the tile of one tiling that contains the query point — `num_tilings`
overlapping grids, all asymmetrically offset. Two things to look for:

- **Generalisation.** Nudge the dot a tiny bit. Most of the coloured
  rectangles don't move — the query stayed inside the same tile of
  every tiling whose boundary it didn't cross. An update at the old
  point reaches the new point through every shared tile.
- **Discrimination.** Drag the dot a tile-width or more away. Now the
  coloured rectangles all shift; the active tile indices in the
  readout below are completely different. The two points share no
  active features, so an update at one never touches the other.

The slider for `num_tilings` controls how many coloured rectangles
overlay; the slider for `tile_width` controls how big each one is.
Narrower tiles give sharper discrimination at the cost of less
generalisation per update.

---

## 8.7 Hashing tiles: the Index Hash Table (IHT)

Naively, the number of tiles per tiling is
$\prod_{i=1}^{n} \lceil \text{range}_i / w \rceil$. For 267 dims with
range 1 and $w = 0.25$, that's $4^{267} \approx 10^{151}$ tiles per
tiling. We cannot store one weight per tile.

Sutton's [1996] fix: **hash tiles into a fixed-size table.** Pick a
table size $M$ (the **IHT size**); map each tile $(t, i_t(s))$ to a
single slot $h(t, i_t(s)) \mod M$. Store one weight per slot.

The Simulator uses $M = 2^{16} = 65{,}536$ slots
([`learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs),
field `tile_coding.iht_size`). Hash: FNV-1a 64-bit with per-block tiling
([`tile_coding.rs:43-56`](https://github.com/falahat/simulator/blob/main/crates/engine/rl_core/src/tile_coding.rs)).

**Hash collisions are deliberate.** Two distinct tiles colliding share a
weight, so updating one updates the other. This is "free generalization"
between accidentally-similar tiles. It is also **noise** — a useful
distinction can be erased by a bad collision.

Trade-off: small $M$ = many collisions (more noise), small storage; large
$M$ = fewer collisions (cleaner signal), more storage. The Simulator's
$2^{16}$ is small for a 267-dim observation; this is one of the levers
to tune if the agent appears to under-learn.

### Estimating collision pressure

With $T = 16$ tilings, each state activates 16 slots out of $M = 65{,}536$.
If the agent visits $N$ distinct states across training, the total
distinct (tiling, tile) pairs touched is at most $16 N$. Collisions
become frequent when $16 N \gtrsim M$ — that is, $N \gtrsim 4000$
distinct states. A 40 000-tick simulation easily generates that many
distinct observations. So **the Simulator runs in a regime of heavy
hash collision** by design; the per-block tiling (next sections) is what
keeps it functional.

### Try it: hash collision pressure under varying M and T

<div id="ch8-hash-collision-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hash_collision/widget.js"></script>

The green curve is the expected fraction of $M$ slots that get touched
after the agent visits $N$ distinct observations; the orange curve
approximates the per-draw collision rate. Defaults match the
Simulator's $M = 2^{16}$, $T = 16$: the $N_{50}$ marker — where half
of the IHT slots are already occupied — sits near $N \approx 2840$,
well inside a typical training run. Slide $\log_2 M$ up to 20 and the
threshold moves out by an order of magnitude; halve $T$ and you nearly
double how many distinct observations the table can absorb before
collisions dominate.

---

## 8.8 The Simulator's tile coder, in detail

The actual implementation lives in
[`crates/engine/rl_core/src/tile_coding.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/rl_core/src/tile_coding.rs).
Key facts:

| Parameter | Value | Source |
|---|---|---|
| Tilings $T$ | per-block (default 16; drives more, memory fewer) | `tile_coding.num_tilings`, `drive_num_tilings`, `memory_num_tilings` |
| Tile width $w$ | 0.25 (uniform) | `tile_coding.tile_width` |
| IHT size $M$ | $2^{16} = 65{,}536$ | `tile_coding.iht_size` |
| Hash function | FNV-1a 64-bit | `tile_coding.rs` (`fold_block`) |
| Offsets | Sutton-style asymmetric | `tile_coding.rs` (`TileCoder::new`) |
| Per-block independence | Yes (8 blocks) | `tile_coding.rs` (`with_block_capacities`, `block_seed`) |

### Per-block independence

The 267-dim observation is tiled **block by block**, not jointly. Each of
the eight blocks gets its own hash namespace, and the active-tile set
for an observation is the **union** across blocks of each block's
active tiles.

Effect: two observations agreeing on (say) drives share their drive
tiles regardless of what the perception block looks like. A clean
signal block is not corrupted by noise in another block. This is the
project's specific answer to the **joint-tiling collapse** problem
(§8.9 next).

### Action-conditioned values via per-action keys

The `Learner`'s Q-table is keyed by
$(\text{tile\_indices}(s), \text{action\_key}(a))$ where `action_key`
folds the action template and parameter
([`policy.rs:169-177`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs)).
There are 18 distinct learner keys for the 16 primitive templates
(Step has 4 directional params, Strike has 2 force params, the other
12 are parameterless). Each (state, action-key) pair shares 16
active-tile slots; the Q-update writes to exactly 16 slots per call.

### Per-block tiling capacity

Tilings are allocated **per block**, not uniformly. The coder iterates
block-outer, tiling-inner, so each block runs its own number of tilings
(set via `with_block_capacities`). The decision-critical **drive** block
gets the full `drive_num_tilings`; the churny, high-dimensional
**episodic-memory** block gets fewer (`memory_num_tilings`); the
remaining blocks use the default `num_tilings`. Resolution is spent
where it pays off (the smooth, low-dimensional drive signal) and
withheld from the noisy memory window — which both sharpens drive
generalization and limits how much memory churn can perturb the value
sum.

(There is no `prepare_partial` / `complete` fast path. An earlier
deferred-block hash trick — which hashed the drive block last so the
forward-search planner could re-finish cheaply per leaf — was removed
along with that planner; the flat ε-greedy Q-policy never used it.)

---

## 8.9 The joint-tiling collapse

Here is the failure mode that motivated the per-block design — and which
you would re-introduce if you ever switched the tile coder back to
global tiling.

> **Joint-tiling collapse.** When all dimensions are tiled jointly, a
> "shared tile" requires *all* dimensions to agree (modulo width). For
> a 267-dim observation, the probability that two arbitrary observations
> share even one tile is essentially zero. Each visited state activates
> a fresh set of tiles never seen before. Updates do not generalize —
> the tile coder degenerates into a hash table indexed by the exact
> observation, with no generalization. We are back to the tabular
> regime, with a uniform-random replacement policy on collision.

### Why per-block helps

If we tile each of the 8 blocks separately, two observations agreeing on
*any* block (say, both have the same drives but different perception)
share that block's tiles. Generalization happens at the block level: the
drive block's weights are updated whenever an agent in a similar drive
state acts, regardless of perception.

Mathematically: the joint feature map
$\phi_\text{joint}(s) = \mathbf{1}[\text{all 267 dims hit tile } i]$ has
support measure zero across observations. The per-block feature map
$\phi_\text{block}(s) = \bigcup_{b=1}^{8} \mathbf{1}[\text{block } b \text{ hit tile } i_b]$
has support that overlaps across observations whenever any single block
matches.

The cost: we lose the ability to discriminate based on *combinations* of
blocks. A learner can no longer say "drive Hunger high AND perception
has food → high Q for Consume" — it sees those as additively decoupled
contributions. Re-coupling requires (i) cross-block features (not yet
implemented), (ii) hierarchical or attention-based aggregation
(Chapter 11 territory), or (iii) recipe memes that act as "this
specific multi-block pattern" priors
([`credit_attribution.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/credit_attribution.rs)
relies on this).

See [`observation_tile_coder.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/observation_tile_coder.rs)
and [`tile_coding/`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/tile_coding)
for the per-block implementation.

### Try it: joint vs per-block sharing on a 4-D toy observation

<div id="ch8-per-block-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/per_block_tiling/widget.js"></script>

Two synthetic 4-D observations — A and B — are split into a 2-D drives
block (hunger, thirst) and a 2-D perception block (food, threat).
Defaults put A and B with **identical drives but very different
perception**. Under joint 4-D tiling, A and B share 0 of 8 active
tiles. Under per-block tiling, A and B share all 8 drive tiles
(because drives agree exactly) while still sharing 0 perception tiles.
That is exactly the §8.9 claim: per-block tiling buys you generalisation
on whichever block matches, even when the full observation diverges.
Try dragging B's drive sliders to differ slightly — the per-block
drive count drops to a smaller, non-zero number, while the joint count
stays at zero.

---

## 8.10 The set-as-vector problem

The 144-dim episodic-memory block is the largest of the eight (54% of
the observation). It encodes **16 slots × 9 fields**: each slot is one
remembered episode with attributes (kind id, age, salience, confidence,
provenance one-hot, emotional imprint, payload).

The 16 slots are *semantically* an unordered set: "the agent's 16 most
salient memories." But the observation vector is a **fixed ordered
array** — slots are sorted by salience descending. This creates a
subtle generalization failure:

### The salience-swap discontinuity

Suppose slots 7 and 8 currently have saliences $0.51$ and $0.49$. A
small new event re-scores them, pushing slot-8's salience to $0.52$.
The agent now sorts as (slot-8-payload at position 7, slot-7-payload at
position 8) — **the encoded fields swap places** even though the
underlying memory set is virtually unchanged.

The tile coder sees this as a discontinuous jump across 9 × 2 = 18
dimensions of the observation. Tile activations change wildly; learned
weights about "memory at rank 7" are now mis-applied to a different
memory.

This is a generic problem in encoding sets as vectors: **the encoding is
not permutation-invariant**, but the underlying object is. Tile coding
specifically cannot recover from this because tiles are tied to absolute
positions.

### Fixes

1. **Deep Sets** [Zaheer et al. 2017]: encode the set as
   $\phi_\text{set}(\{x_1, \ldots, x_k\}) = \rho\big(\sum_i \psi(x_i)\big)$.
   Sum-pool to be permutation-invariant; let $\psi$ and $\rho$ be MLPs.
   Not directly compatible with tile coding; would require a learned
   embedding layer.
2. **Set Transformer** [Lee et al. 2019]: attention-based pooling.
   Same compatibility issue.
3. **Slot-position-invariant tiling**: tile each slot's 9 fields
   independently, then sum the slot tiles. The agent loses
   *cross-slot* information (e.g. "two memories of the same kind")
   but keeps absolute slot ordering invariance.

The Simulator currently does **not** fix this (status open). Empirically
the issue surfaces as TD-error spikes on ticks when a new event
displaces an older one in the salience ordering. The
[`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
`td_max < 5.0` bound is generous partly to absorb these spikes.

### Try it: swap two memory slots and watch the hash change

<div id="ch8-set-vector-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/set_vector/widget.js"></script>

Six slots, three fields each (`kind`, `salience`, `recency`). The two
heatmaps show one memory set before and after a single adjacent-slot
swap — the kind of perturbation a small new event triggers when it
re-scores salience just past a neighbour. The bottom bar chart runs
500 random adjacent swaps and counts how often each encoding scheme
sees a hash change: the positional encoding (the current Simulator) sees
near-100% change rate, while a set-invariant encoding (per-slot tile
sums) sees ~0%. That gap is exactly the TD-error spike budget the
homeostatic test has to absorb.

---

## 8.11 Linear FA vs. neural FA — a comparison

To set up Chapter 11, contrast the two flavours.

| Property | Linear FA | Neural FA |
|---|---|---|
| Representation | $\theta^{\top}\phi(s)$, $\phi$ fixed | $f_\theta(s)$, $f$ a neural net |
| Parameters $d$ | Tens to millions (sparse) | Millions to billions (dense) |
| Update | Single TD step, $O(d_\text{active})$ | Full backprop, $O(\text{net size})$ |
| Convergence (on-policy TD) | Provable [TVR97] | No general guarantee |
| Convergence (off-policy TD) | Can diverge (Baird) | Diverges spectacularly |
| Expressiveness | Capped at $\text{span}(\phi)$ | Universal approximator |
| Feature design | Manual + offline | Learned end-to-end |
| Hyperparameter sensitivity | Low (4–5 knobs) | High (dozens) |
| Computational cost per step | Cheap | Expensive |

The right choice depends on what dominates: the cost of feature
engineering vs. the cost of training stability. The Simulator picked
linear FA with tile coding because (a) the cognition cadence (≥1 per
tick per agent across many agents) demands cheap inference, (b) the
debugging story is more tractable, (c) the deadly triad is more
manageable, (d) per-agent learning rate memes
([`learning_rate.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/learning_rate.rs))
let individuals diverge in $(\alpha, \gamma)$ without separate networks.

Chapter 11 explores when one would pivot to neural FA, and the engineering
discipline that pivot demands.

---

## 8.12 Project tie-in

### Where this chapter's ideas live

- **Linear semi-gradient Q-update.** `Learner::td_update()` in
  [`value_function.rs:126-146`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/value_function.rs).
- **Tile-coding hash, asymmetric offsets, per-block independence,
  per-block tiling capacity.** [`tile_coding.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/rl_core/src/tile_coding.rs)
  — `fold_block` / `block_seed` (hash + per-block namespacing),
  `TileCoder::new` (offsets), `with_block_capacities` (per-block tile
  counts). The coder is built from the schema in
  [`value_function.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/value_function.rs)
  `coder_from_config` (`drive_num_tilings` for drives,
  `memory_num_tilings` for the memory window).
- **Hyperparameters.** [`learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs)
  — fields `alpha = 0.10`, `gamma = 0.90`, `epsilon = 0.10`,
  `tile_coding.num_tilings = 16`, `tile_coding.tile_width = 0.25`,
  `tile_coding.iht_size = 65,536`.
- **18 learner keys, per-action.** [`policy.rs:181-199`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) builds the action roster.

### Where this chapter's ideas are *not* used

- **No fitted-Q-iteration (offline batch update).** Online TD only.
- **No Gauss-Newton / LSTD.** The closed-form $\theta = (\Phi^{\top}\Phi)^{-1}\Phi^{\top}r$
  is never computed; we only do incremental SGD-style updates.
- **No experience replay** with sampling from a buffer (that arrives in
  Chapter 11 with DQN).
- **No learned features.** Every feature is hand-designed via tile
  coding. No representation learning.

### Try it: LSTD vs online semi-gradient TD on a 3-state chain

<div id="ch8-lstd-vs-sgd-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/lstd_vs_sgd/widget.js"></script>

A 3-state ring with 2-D linear features and known $\theta^\star$.
Both solvers walk the same sample stream: SGD-TD does one
$O(d)$ semi-gradient step per sample; LSTD accumulates the full
$A = \sum \phi (\phi - \gamma \phi')^\top$ and $b = \sum r\phi$ matrices
and solves $A\theta = b$ every step. LSTD's error drops by orders of
magnitude in the first dozen samples; SGD-TD lags and depends on
$\alpha$. The per-step cost readout is the trade the §8.12 bullet
calls out: LSTD is sample-efficient but $O(d^2)$ in storage and
$O(d^3)$ to read out — fine at $d = 2$, hopeless at the
Simulator's $d \approx 10^6$.

### Tests exercising this chapter

- [`navigation.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/navigation.rs)
  — directional Step preference under reward modulation. Asserts
  convergence via `q_approach_advantage` metric.
- [`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
  — hungry-arm: `td_max < 5.0` $\ell_\infty$ TD-error bound.
- [`threat_response.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/threat_response.rs)
  — learning *within* an already-explored action class (Step direction
  discrimination), exploiting the per-block tile coder's clean drive
  generalization.

### What would need to change to apply modern variants

- **Fix the set-as-vector problem.** Add a permutation-invariant
  pooling layer for the episodic-memory block. Easiest first step:
  per-slot tile coding summed across slots.
- **Learnable features end-to-end.** Replace the fixed tile coder with a
  small MLP $\phi_\theta(s)$ trained jointly with the Q-head. Promotes
  the Simulator from linear FA to neural FA — pulls in Chapter 11
  machinery (target networks, replay, etc.).
- **Adaptive tile widths.** [Whiteson, Taylor & Stone 2007] split tiles
  where Bellman error is high. The Simulator's uniform 0.25 width is
  the simplest hand-tuned choice; an adaptive version would refine
  tiles in heavily-visited regions.

---

## 8.13 Exercises

1. **(Polynomial-feature blow-up.)** For a 30-dim continuous state, how
   many features does a degree-3 polynomial basis have? Compare to the
   Simulator's $T \cdot M = 16 \cdot 65{,}536 \approx 10^6$ effective
   parameters across all tilings.

2. **(Semi-gradient on Mountain Car.)** Implement linear semi-gradient
   Q-learning on Mountain Car (2-D continuous state) with tile coding:
   $T = 8$ tilings, $w = 0.0833$ (so 12 tiles/dim/tiling), $M = 2048$.
   Report ticks-to-solve over 100 seeds. Vary $T$ in
   $\{2, 4, 8, 16, 32\}$ — what is the cost of more tilings vs. the
   discrimination/generalization trade-off?

3. **(Why on-policy matters.)** Reproduce Baird's counterexample (see
   Chapter 17 for the construction). Run linear semi-gradient TD
   on-policy and off-policy. Report the weight norm trajectory.

4. **(Hash collision pressure.)** Estimate the expected number of
   distinct slots touched at the Simulator's $M = 65{,}536$, $T = 16$
   after the agent visits $N$ distinct observations, for
   $N \in \{100, 1{,}000, 10{,}000, 100{,}000\}$. What fraction of
   slots collide?

5. **(Per-block vs. joint tiling.)** Construct a 2-block toy example
   (block 1 = drives, 2 dims; block 2 = perception, 2 dims). Show that
   joint tiling at $w = 0.25$ requires $\sim 16$ tiles to span the
   state space, but two observations agreeing on drives but differing
   in perception will *not* share tiles. Show that per-block tiling
   *does* share the drive tiles.

6. **(Set-as-vector.)** Construct a 2-memory-slot example where a tiny
   salience change swaps slots and the encoded vector jumps. Compute
   the TD error if the prior tile-coded weight assigned different
   values to the two slot positions.

7. **(Per-block tiling capacity.)** The coder runs `drive_num_tilings`
   tilings on the drive block and `memory_num_tilings` (fewer) on the
   episodic-memory block. Estimate the per-decision hashing cost as the
   sum of per-block tile counts across the 8 blocks, and argue why
   spending more tilings on the smooth drive signal — and fewer on the
   churny memory window — improves generalization where it matters.

8. **(Tile width sweep.)** On the Simulator's
   [`navigation.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/navigation.rs)
   scenario, sweep `tile_width` over
   $\{0.10, 0.15, 0.20, 0.25, 0.35, 0.50\}$. Plot
   `q_approach_advantage` at 20K ticks. Identify the width that
   minimizes time-to-convergence — does it match the project's
   default 0.25?

---

## 8.14 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Albus 1975] — CMAC; the original tile-coding paper (§8.6).
- [Baird 1995] — off-policy linear TD counterexample (§8.3, §8.4).
- [Konidaris, Osentoski & Thomas 2011] — Fourier basis (§8.5).
- [Lee et al. 2019] — Set Transformer (§8.10).
- [Mahadevan & Maggioni 2007] — proto-value functions (§8.5).
- [Parr et al. 2007] — BEBF (§8.5).
- [Sutton 1996] — sparse coarse coding for RL (§8.6, §8.7).
- [Sutton & Barto 2018, Ch. 9–10] — function approximation chapters.
- [Tsitsiklis & Van Roy 1997] — linear TD convergence theorem (§8.4).
- [Whiteson, Taylor & Stone 2007] — adaptive tile coding (§8.12).
- [Zaheer et al. 2017] — Deep Sets (§8.10).

## 8.15 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton & Barto 2018] | Ch. 9–10 (semi-gradient methods, on-policy approximation) | Canonical textbook treatment |
| [Szepesvári 2010] | Ch. 3–4 | Operator-theoretic + projection-error perspective |
| [Geist & Pietquin 2013] | LSTD survey | Closed-form least-squares variants |
| [Bertsekas 2012] | Vol. II Ch. 6 | Rigorous proofs of TD convergence |

---

**Next:** [Chapter 11 — Deep Q-Learning and Beyond](11_deep_q_learning.md) —
when linear features are not enough.

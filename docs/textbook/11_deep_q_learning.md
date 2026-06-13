# Chapter 11 — Deep Q-Learning and Beyond

> **Prerequisites:** Chapter [6](08_temporal_difference_learning.md)
> (Q-learning), [8](10_function_approximation.md) (linear function
> approximation), [3](03_mathematics_for_ai.md) (Hessian and
> non-convex optimization).

> **Citations:** [Mnih et al. 2015] (DQN); [van Hasselt 2010] (Double
> Q-learning); [van Hasselt, Guez & Silver 2016] (Double DQN);
> [Wang et al. 2016] (Dueling); [Schaul et al. 2016] (PER);
> [Bellemare, Dabney & Munos 2017] (C51); [Dabney et al. 2018] (QR-DQN);
> [Fortunato et al. 2018] (NoisyNets); [Hessel et al. 2018] (Rainbow);
> [van Hasselt et al. 2018] (deep deadly triad). Full entries in
> [`bibliography.md`](bibliography.md).

> **Learning objectives:**
> 1. Read the DQN training loop line-by-line and identify what each
>    trick is preventing.
> 2. Diagnose the maximization bias of Q-learning and apply Double DQN.
> 3. Build the dueling architecture and explain *why it is the
>    architectural fix for the Simulator's Q-bias bootstrap pathology*.
> 4. Implement prioritized experience replay and explain the importance-
>    sampling correction.
> 5. Know what Rainbow is and which of its six pieces matter most.
> 6. Articulate when (and why) deep RL diverges — the deadly triad in
>    its modern form.

## Why this chapter exists

Chapter 10 set up linear FA with hand-designed features. That works when
the right features are easy to specify (continuous low-dim states, tile
coding, Fourier basis on smooth functions). It does **not** work when
the right features are themselves what we need to learn — pixels of an
Atari frame, raw audio, a 251-dim observation whose intrinsic structure
is unknown.

The pivot is from "linear in fixed $\phi$" to "non-linear in raw $s$":

$$Q(s, a; \theta) = \text{NeuralNet}_\theta(s, a).$$

This unlocks universal approximation. It also unlocks the **deadly
triad** (Chapter 17) at its full intensity, and breaks every convergence
guarantee we built up. The story of deep Q-learning is the story of
engineering tricks that empirically tame the divergence even though no
theorem guarantees they will.

## Table of contents

- [9.1 From linear to deep](#91-from-linear-to-deep)
- [9.2 DQN: the two tricks](#92-dqn-the-two-tricks-mnih-et-al-2015)
- [9.3 Double DQN](#93-double-dqn)
- [9.4 Dueling networks](#94-dueling-networks)
- [9.5 Prioritized Experience Replay](#95-prioritized-experience-replay)
- [9.6 Multi-step targets and Retrace](#96-multi-step-targets-and-retrace)
- [9.7 Distributional RL: C51 and QR-DQN](#97-distributional-rl-c51-and-qr-dqn)
- [9.8 NoisyNets and Rainbow](#98-noisynets-and-rainbow)
- [9.9 The deadly triad in DQN](#99-the-deadly-triad-in-dqn)
- [9.10 Project tie-in: why no DQN — and what we'd port](#910-project-tie-in)
- [9.11 Exercises](#911-exercises)
- [9.12 References](#912-references-cited-in-this-chapter)
- [9.13 Further reading](#913-further-reading)

---

## 9.1 From linear to deep

In Chapter 10 the Q-function was

$$Q(s, a; \theta) = \theta_a^{\top} \phi(s)$$

with hand-designed $\phi$. The gradient was just $\phi(s)$, and TD updates
were closed-form sparse vector additions.

A **deep Q-network** replaces this with

$$Q(s, a; \theta) = f_\theta(s)[a]$$

where $f_\theta: \mathbb{R}^n \to \mathbb{R}^{|\mathcal{A}|}$ is a neural
network mapping the raw observation to one Q-value per action. For Atari
the input is a stack of game-frame pixels; the network is a convolutional
stack into a final dense head. For continuous low-dim states the input
is the state vector; the network is a small MLP.

The gradient $\nabla_\theta Q(s, a; \theta)$ is computed by
**backpropagation** through the network. The TD-target loss is

$$L(\theta) = \tfrac{1}{2}\big(r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta)\big)^2,$$

where $\theta^{-}$ is a *target* parameter vector explained in §9.2.
Stochastic gradient descent on $L$ updates $\theta$ in the direction
$-\nabla_\theta L = \delta \cdot \nabla_\theta Q(s, a; \theta)$. This is
the same *semi-gradient TD* update as linear FA — we treat the target
as fixed when differentiating. The only thing that changed is what
$\nabla_\theta Q$ is.

### Try it: live MLP learns XOR

<div id="ch11-live-mlp-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/live_mlp_xor/widget.js"></script>

Hit **play** and watch a 2-hidden-unit MLP descend from a flat
mid-grey decision surface to the bent boundary that separates the
four XOR points. This is the simplest function a linear model
*can't* learn (no straight line cuts the white points from the
black ones), and it's exactly what backprop gives you for free.

Things to try while it trains:

- **Swap the activation** mid-training (the dropdown is wired to
  `Sequential::replace` — see the constitutional pillar in
  `docs/designs/neural_network_library.md`). Switching from
  `tanh` to `relu` mid-descent often re-accelerates a stalled
  trajectory; switching to `sigmoid` slows it down because the
  gradients saturate.
- **Swap the loss** from MSE to Huber. The boundary curve barely
  changes for this problem (residuals stay small), but the loss
  axis rescales — Huber's linear tail caps the influence of any
  one residual.
- **Swap the optimiser** between Adam and SGD. The red rule on the
  loss plot marks each swap. Adam usually wins on this problem;
  SGD with momentum is close behind; plain SGD takes ~5× longer.

The library doing the work is `crates/engine/nn` (in this repo).
Every line of code in that crate is ours; borrowed *ideas* (Adam's
recurrence, log-sum-exp's shift, Box-Muller's normal sampling) are
cited inline at point of use.

### Try it: backprop with your own eyes — interactive NN explorer

<div id="ch11-nn-explorer-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/nn_explorer/widget.js"></script>

Twelve frames in a linear timeline — click **next ▶** to advance,
**◀ prev** to go back, **▶ play** to auto-step. The full pipeline:

1. **Frames 1–5** walk the forward pass: input → hidden pre-activation
   → tanh → output pre-activation → sigmoid. Each click lights up
   exactly the layer being computed; the just-computed nodes get
   an orange ring and their numeric values appear inside.
2. **Frame 6** computes the loss `L = (ŷ − target)²`. The loss
   callout in the top-right turns on.
3. **Frames 7–11** unwind the backward pass: seed `∂L/∂ŷ`, then
   through each activation and each weight matrix in reverse order.
   Gradients (`∂L = …`) appear under each node as they're computed;
   hover any edge to see its `∂L/∂w`.
4. **Frame 12** applies one SGD step. Edge thicknesses change.
   Watch the prediction `ŷ` move toward the target.

The middle column is a 4-unit `tanh` hidden layer; edges show the
actual weights (blue = positive, red = negative, thickness =
magnitude). The output node uses `sigmoid` so the prediction lives
in `[0, 1]`. Same math the Rust crate runs at 10⁵× the throughput;
seeing the scalars first helps before trusting the optimised loops.

### Try implementing it: backward for tanh

<div id="ch11-tanh-backward-exercise"></div>
<script type="module" src="./widgets/tanh_backward/exercise.js"></script>

The first backward you ever write. The harness will compare your
analytic derivative against the finite-difference numerical
derivative — i.e., the textbook's claim ("backprop computes the same
thing as calculus") becomes the test of your code.

### Try implementing it: backward for ReLU

<div id="ch11-relu-backward-exercise"></div>
<script type="module" src="./widgets/relu_backward/exercise.js"></script>

ReLU's backward is a *gate*: gradient passes where the pre-activation
was positive, zero elsewhere. Note the subtle data-flow point — ReLU
needs the **pre-activation** `x`, not the post-activation `y`, because
`y = 0` for any non-positive `x` and the sign information is gone.
This is exactly the bookkeeping every autograd framework does behind
your back.

### Try implementing it: one Adam optimiser step

<div id="ch11-adam-step-exercise"></div>
<script type="module" src="./widgets/adam_step/exercise.js"></script>

Adam is five lines once you understand the bias correction. Type them.
The harness compares your update to a reference implementation across
five random `(params, grads, m, v)` configurations, varying the step
counter `t` to exercise the `(1 − β^t)` bias-correction term — the
piece of Adam that's easy to get *almost* right.

### Try it: MLP regression on sparse 1-D points

<div id="ch11-mlp-regression-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/mlp_regression_1d/widget.js"></script>

A handful of scattered red dots is the training set. The dashed
grey curve is the function those dots were sampled from — the
network never sees that, but the reader can. The green curve is
the MLP's prediction; hit play and watch it bend.

Two things to notice:

- **Between the points, the network is improvising.** The target
  curve is unknown to the model; the green curve there is a
  consequence of (i) the loss only constraining values *at* the
  red dots, and (ii) the activation's shape priors filling in the
  rest. ReLU stitches piecewise-linear segments; tanh smooths them.
- **Capacity is not the same as fit.** Push hidden units to 128 and
  the green curve can wiggle through *every* dot — including
  noise, if the data had any — but the dashed truth and the green
  curve diverge wildly between points. The bias-variance tradeoff
  made visible.

### What goes wrong

Linear semi-gradient TD converges on-policy
([Tsitsiklis & Van Roy 1997]). Deep semi-gradient TD has no convergence
guarantee. Three things break:

1. **Non-stationary target.** $\max_{a'} Q(s', a'; \theta)$ uses the
   *same* $\theta$ that we are updating. Each gradient step changes
   the target. This is a self-referential loop.
2. **Correlated samples.** Consecutive transitions are correlated through
   the dynamics. SGD theorems assume i.i.d. samples; here they are not.
3. **Function approximation error.** A neural network's bias can move
   unpredictably as $\theta$ updates; small parameter changes can produce
   large value-function changes.

All three together are the **deadly triad** (Chapter 17). DQN's two
"tricks" attack the first two.

---

## 9.2 DQN: the two tricks ([Mnih et al. 2015])

### Why this section is THE deep-RL section

[Mnih et al. 2015] published the result that started modern deep RL:
a single agent achieves human-level performance on 49 Atari 2600
games from pixels with no game-specific feature engineering. The
architecture is a 3-layer CNN + 2-layer MLP — by modern standards,
small. The algorithm is **semi-gradient Q-learning** (Chapter 10
§8.3) with neural-net $Q(s, a; \theta)$.

So why did it work? Naive "Q-learning with a neural net" had been
tried since the 1990s — and it diverged. The deadly triad
(Chapter 17) — function approximation + bootstrapping + off-policy
data — predicts exactly this. DQN's contribution isn't the
network or even the loss: **it's two stabilising tricks that bound
the deadly-triad pathologies enough that gradient descent actually
makes progress.**

Both tricks are simple and load-bearing. *Without them, DQN does
not learn.* They are the architectural decisions every subsequent
deep-RL system inherits — Rainbow, DDPG, SAC, R2D2, all add to but
do not remove these two. Internalize what each one fixes; the rest
of the chapter is variations on them.

### The two tricks

The architecture is a 3-layer CNN + 2-layer MLP; the algorithm is
semi-gradient Q-learning. The two stabilizing tricks are
unglamorous and essential:

### Experience Replay

**The trick.** Store every transition $(s_t, a_t, r _{t+1}, s _{t+1})$
in a circular buffer (DQN paper: capacity $10^6$). On each learning
step, sample a **minibatch** of $B$ transitions uniformly at random
and compute the loss as their average — rather than using the most
recent transition directly.

**What it fixes.** Three problems with naive online TD on a neural
net, each independently fatal:

1. **Temporal correlation** — adjacent transitions
   $(s_t, \ldots, s _{t+1})$ are highly correlated (the next state is
   one step away from the previous). Mini-batch SGD assumes
   i.i.d. samples; with
   correlated samples the gradient estimator's variance balloons and
   convergence guarantees evaporate. The replay buffer interleaves
   transitions from many different points in the trajectory, so each
   minibatch looks roughly i.i.d.
2. **Sample inefficiency** — a transition seen once contributes one
   gradient. With replay, each transition is sampled and used
   $\sim B \cdot N _{\text{updates}} / |\text{buffer}|$ times before
   being overwritten — typically $10$–$100$ reuses. This matters
   enormously when environment interaction is expensive (real
   robots, Atari emulation, the Simulator's full-tick cost).
3. **Catastrophic-interference dynamics** — without replay, recent
   experiences dominate the loss landscape, and the network forgets
   what it learned from older states ([McCloskey & Cohen 1989], the
   classic catastrophic forgetting result). Replay's mixing keeps
   old and new transitions visible simultaneously.

**The hyperparameter.** Buffer size is a trade-off: too small means
insufficient decorrelation (back to problems 1 + 3); too large means
the oldest transitions reflect a policy long since abandoned (the
bootstrap target $Q(s'; \theta^-)$ no longer matches the policy that
collected $s'$). DQN's $10^6$ for Atari was empirical; modern
codebases (Rainbow, R2D2) push it higher with prioritized sampling
(§9.5) to make stale data still useful.

**What this trick *doesn't* fix.** Replay still uses *off-policy*
data — the policy that collected transitions is older than the
policy being updated. Combined with bootstrapping and function
approximation, this is the deadly triad's exact configuration. The
target network (next) is what makes the triad survivable.

### Target Network

**The trick.** Maintain *two* copies of $\theta$: the **online**
network $\theta$ that gradient descent updates each step, and the
**target** network $\theta^{-}$ used only to compute the bootstrap
target. The loss becomes

$$
L(\theta) = \mathbb{E}\big[(r + \gamma \max _{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta))^2\big].
$$

Note the key asymmetry: $\theta$ inside the prediction, $\theta^{-}$
inside the max. Gradient flows only through the prediction (we
treat the target as constant — semi-gradient, Ch10 §8.3). Every
$C$ steps (DQN: $C = 10{,}000$), copy $\theta \to \theta^{-}$ —
the target catches up to the online net in a single hard sync.

**What it fixes.** Without a target network, the bootstrap target
moves *at the same speed* as the prediction:

$$
\underbrace{Q(s, a; \theta)} _{\text{prediction}} \leftarrow r + \gamma \max _{a'} \underbrace{Q(s', a'; \theta)} _{\text{target, same } \theta}.
$$

This is the "chase your tail" problem: every step of gradient
descent toward the target *changes the target*, because the target
shares parameters with the prediction. The Bellman backup is no
longer a contraction in the function approximator's space — the
fixed point that semi-gradient TD aims at *itself moves*
non-monotonically.

The fix is brutally simple: **freeze the target's parameters for
$C$ steps**, so that within each $C$-step window, the algorithm
solves a *fixed* regression problem. The optimization theory of
regression (smooth loss, descent converges) takes over. Every $C$
steps, the freeze releases and the target catches up — but each
$C$-step "epoch" was a clean optimization.

**Two intuitions for why this works:**

- **Operator decoupling.** The fix replaces "minimize
  $\|Q_\theta - T^\star Q_\theta\|$" (a non-stationary objective)
  with "minimize $\|Q_\theta - T^\star Q _{\theta^-}\|$ for fixed
  $\theta^-$" (a stationary one), then occasionally swap
  $\theta^-$. Each epoch is one *projected* Bellman backup; the
  outer loop iterates them, mimicking value iteration in
  function-approximator space.
- **Regression problem in disguise.** Inside each $C$-step window,
  you're literally doing supervised regression: input $(s, a)$,
  target $r + \gamma \max Q(s'; \theta^-)$ (a fixed function),
  minimize MSE. All the standard convergence guarantees of SGD on
  smooth losses apply. The target update is what makes it RL again.

**The hyperparameter.** $C$ trades stability against learning speed:

- **Too small (~$10$):** target moves nearly as fast as prediction;
  the chase-your-tail pathology reappears.
- **Too large (~$10^6$):** target stays stale; the Bellman backup
  converges to a fixed point of an *old* policy, then has to
  re-converge after each sync. Convergence to $Q^\star$ slows
  proportionally.
- **Just right ($10^3$–$10^5$):** target stable enough that
  regression converges within each window, fresh enough that
  ground truth approaches $Q^\star$ over $\sim 100$ syncs.

**Soft updates** (Polyak averaging):
$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$ each step
with $\tau \approx 0.005$. Equivalent to a hard sync every
$\sim 1/\tau$ steps but smoother. DDPG and SAC (Chapter 12 §10.5
and §13.4) ship this variant.

**What this trick *doesn't* fix.** The deadly triad is *bounded*,
not eliminated. The target network gives you a window of stable
regression; once the sync fires, the target moves and the
divergence pressure returns. In tabular settings it works
beautifully; with deep networks, you also need replay (above), and
in adversarial off-policy settings even both together aren't always
enough — Rainbow (§9.8) and the algorithms after stack more fixes.

### Try it: target-network freezing

<div id="ch9-target-net-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/target_network/widget.js"></script>

Three updaters race on the minimal 2-state MDP with γ = 0.99 (so
θ\* = 1/(1−γ) = 100). The red curve (no target net) oscillates or
diverges at large α — the bootstrap uses the same θ being updated.
The blue curve (hard sync every C) stair-steps toward θ\*: each plateau
is a fixed-target regression problem. The green curve (soft Polyak τ)
is the smoothed version. Push α to 1.0 and the red curve detonates
while the other two stay tame.

### The DQN training loop

```python
for episode in range(num_episodes):
    s = env.reset()
    while not done:
        # epsilon-greedy action
        a = epsilon_greedy(Q_online(s), epsilon)
        s_next, r, done = env.step(a)
        # store in replay buffer
        buffer.add((s, a, r, s_next, done))
        # update
        batch = buffer.sample(B)
        target = r + gamma * max_a Q_target(s_next, a) * (1 - done)
        loss = (Q_online(s, a) - target).pow(2).mean()
        loss.backward(); optimizer.step()
        s = s_next
        if step % C == 0:
            theta_target = theta_online  # hard sync
```

Read it: two networks, one buffer, one $\epsilon$-greedy actor. Twelve
lines. The two tricks are *both* hyperparameters away from instability,
which is why every paper after DQN spends effort improving them.

---

## 9.3 Double DQN

Chapter 8 showed that Q-learning's $\max_{a'} Q(s', a')$ over noisy
estimates is **upward biased** — the max of noisy variables is bigger
than the max of the underlying true values. With deep approximation this
bias compounds: the network is noisy by construction (it never perfectly
fits), and the max picks the most-overestimated action.

[van Hasselt 2010] proposed Double Q-learning for the tabular case: keep
two Q-functions $Q_A$ and $Q_B$; use $Q_A$ to *select* the max action
and $Q_B$ to *evaluate* it. The expected max-over-true-values is
recovered if $Q_A$ and $Q_B$ have independent noise.

[van Hasselt, Guez & Silver 2016] ported this to DQN with **one-line
elegance**: use the online network to select, target network to evaluate.

> **Double DQN target.**
>
> $$
> y = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-}\big).
> $$
>

The DQN target was $\max_{a'} Q(s', a'; \theta^{-})$ — same network does
selection and evaluation. DDQN decouples them. The fix removes a
substantial portion of the overestimation bias and improves performance
on roughly half the Atari games tested, with no extra cost.

### Trade-off

- **Pro:** removes a known bias; one-line code change; never makes
  things worse.
- **Con:** slightly slower learning on the games where the bias was not
  the bottleneck.

Modern DQN-style agents almost always use Double-DQN target. The choice
is closer to "default on" than to "optional optimization."

### Try it: the maximization bias as a bandit

<div id="ch9-max-bias-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/max_bias_bandit/widget.js"></script>

Each "arm" has true value Q* = 0 but our estimator is noisy with
variance σ²/n. The single-network max is upward-biased — the red bar
sits above zero, tracking the analytic σ·√(2 ln K / n). The
double-network estimator (select with one head, evaluate with another)
sits at zero on average. Crank K up: the gap widens like √(ln K).

---

## 9.4 Dueling networks

[Wang et al. 2016] observed that for many states, *the choice of action
does not matter much* — what matters is the value of being in that state.
A standard DQN must learn the (state-value, action-advantage) jointly
through a single output head, which is sample-inefficient.

The **dueling architecture** decomposes the network into two streams
sharing early layers:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + A(s, a; \theta, \alpha) - \tfrac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha).$$

- **$V(s)$**: state-value (one scalar per state).
- **$A(s, a)$**: advantage of $a$ over the average action in $s$.
- **Subtraction of mean advantage**: identifies the decomposition. Without
  it $(V, A)$ are non-unique (you can add a constant to $V$ and subtract
  from $A$ without changing $Q$).

### Try it: dueling decomposition

<div id="ch9-dueling-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/dueling/widget.js"></script>

Slide all three Q values together (e.g. add the same number to each).
V tracks the shift; the advantages A stay put — they're invariant to
the uniform alive-value baseline. That's how dueling fixes the
Q-bias-bootstrap pathology Chapter 17 describes: V absorbs the baseline,
A keeps the action-relative ordering clean.

### Why this is the architectural fix for the Q-bias bootstrap pathology

The Simulator's central bug, the **Q-bias bootstrap pathology**
(analyzed in detail in [Chapter 17](17_fa_pathologies.md)), reduces to: every committed action's
$Q$ drifts toward $w_\text{alive} / (1 - \gamma) = 1.0 / 0.1 = 10.0$,
because $w_\text{alive} = 1.0$ is paid every tick regardless of which
action was committed. Linear tile-coded $Q$ cannot decouple "the agent
is alive (worth ~10)" from "the action chosen was good (small delta)."

Dueling solves this *architecturally*:

- $V(s) \approx 10$ absorbs the alive baseline.
- $A(s, a)$ has zero mean across actions (by construction of the mean
  subtraction) — only differences matter.
- The argmax over $Q(s, \cdot)$ equals the argmax over $A(s, \cdot)$ —
  the alive-baseline is gone from action selection.

This is the strongest single argument for ever pivoting the Simulator
to neural FA: a dueling head over a small MLP on the 251-dim observation
would not require $w_\text{alive} = 0$ (Fix 1 in the bug report); it
would *learn around* the bias by representing $V$ and $A$ separately.
Chapter 17 walks through the linear-FA analogue (advantage learning).

### How to read the formula

The mean-subtraction trick can be replaced by a $\max$ over advantages
instead of a mean. Both work; mean is empirically more stable. The
networks share early layers (typically the convolutional trunk on
Atari) and split into the two heads only at the top. Empirically the
dueling architecture provides 2–3× improvements on environments where
many states have similar action values — which is *most* of them.

---

## 9.5 Prioritized Experience Replay

Uniform sampling from the replay buffer treats every transition as
equally informative. It is not. A transition with large TD error is
**under-fit** — the network's prediction was far from the target. A
transition with near-zero TD error is essentially memorized — sampling
it is wasted compute.

**Prioritized Experience Replay (PER)** [Schaul et al. 2016] samples
transitions in proportion to their TD-error magnitude:

$$p_i \propto |\delta_i|^\rho + \epsilon$$

for some exponent $\rho \in [0, 1]$ (paper: $\rho = 0.6$) and small
$\epsilon$ to prevent zero-probability samples. Each transition's
priority is updated when it is sampled.

### The importance-sampling correction

Non-uniform sampling biases the SGD estimator. PER corrects this with
**importance-sampling weights**:

$$w_i = \left(\frac{1}{N\, p_i}\right)^\beta$$

where $N$ is the buffer size and $\beta \in [0, 1]$ controls correction
strength ($\beta = 0$: no correction; $\beta = 1$: full correction). The
loss for transition $i$ is multiplied by $w_i$.

Why two hyperparameters? You want most of the prioritization benefit
($\rho$ large) but not too much sampling bias ($\beta$ corrects it).
Both are annealed during training; $\beta$ is typically ramped from
0.4 to 1.0 over the run.

### Trade-off

- **Pro:** ~2× sample efficiency on Atari benchmarks. Especially helps
  when reward is sparse — high-error transitions are the rare
  reward-bearing ones.
- **Con:** ~30% wallclock overhead from priority bookkeeping (segment
  tree maintenance). Memory: extra priority per transition.
- **Failure mode:** stale priorities. A transition stored long ago may
  have priority from an old network. Periodic re-priority sweeps help.

### Try it: priority, IS weight, and the effective loss contribution

<div id="ch9-per-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/per_sampling/widget.js"></script>

Each rank on the x-axis is one transition in a 1000-step buffer, sorted
by |δ|. Three curves: red is the sampling priority p_i, blue is the
IS weight w_i, green is the *effective loss contribution* p_i · w_i.
At ρ = 0 all curves are flat (uniform replay). At ρ = 1, β = 1 the IS
weight perfectly cancels the priority — *contrib goes flat again*; PER
becomes uniform-in-expectation. The interesting regime is ρ = 0.6,
β = 0.4 (the paper's settings): high-|δ| transitions still carry most
of the gradient signal even after partial IS correction.

---

## 9.6 Multi-step targets and Retrace

DQN uses 1-step TD targets: $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$. We
saw in Chapter 9 that $n$-step returns trade bias for variance:

$$y^{(n)}\_t = \sum\_{k=0}^{n-1} \gamma^k r\_{t+k+1} + \gamma^n \max\_{a'} Q(s\_{t+n}, a').$$

At $n = 1$: pure TD (low variance, high bias from bootstrap). At $n = \infty$:
Monte Carlo (no bootstrap, high variance). The sweet spot is usually
$n \in \{3, 5\}$ for DQN-style learning.

The catch in DQN: the data is **off-policy** (collected by older policies
from the replay buffer). $n$-step returns under off-policy data are
biased; importance-sampling corrections can have huge variance.

**Retrace($\lambda$)** [Munos et al. 2016] fixes this with a clipped
trace coefficient:

$$c_t = \lambda \min\left(1, \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}\right)$$

that down-weights but never inflates importance ratios. Retrace is the
foundation for distributed RL agents like R2D2 and IMPALA, where stale
behavior data is unavoidable.

For the Simulator's flat policy, off-policy is not relevant
(no replay) — but multi-step targets *are* a candidate fix for the
long-horizon credit-assignment gap on the L-suite (Chapter 19). A 5-step
return with $\gamma = 0.9$ propagates reward 5 ticks back, partly
softening the $\gamma^{500} \approx 0$ catastrophe.

---

## 9.7 Distributional RL: C51 and QR-DQN

Why learn just the mean return? The full distribution over future returns
contains risk information, multi-modal structure, and more gradient
signal per update.

**C51** [Bellemare, Dabney & Munos 2017] models the value as a categorical
distribution over a fixed support of 51 atoms
$\{z_1, \ldots, z_{51}\} \subset [V_{\min}, V_{\max}]$. The network outputs
probabilities $p_i(s, a; \theta)$ for each atom; the expected value is
$\sum_i z_i p_i$. Updates project the Bellman target distribution
$\sum_i p_i \delta_{r + \gamma z_i}$ back onto the fixed support.

**QR-DQN** [Dabney et al. 2018] flips the parameterization: fix the
quantile *levels* $\tau_1, \ldots, \tau_N$; learn the quantile *values*.
Avoids the projection step. Uses **quantile regression** loss (asymmetric
absolute error). Provably a contraction in the 1-Wasserstein metric.

Why this matters even if you only want the mean: distributional methods
empirically outperform their non-distributional counterparts on the same
mean-reward objective. The intuition is that learning more about the
target distribution gives the network richer gradient signal — every
quantile is an independent training signal.

For the Simulator, distributional Q would be a sensible upgrade if/when
neural FA arrives — but the per-atom storage cost is multiplied across
the agent population, which can be prohibitive.

### Try it: the distribution DQN throws away

<div id="ch9-distributional-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/distributional/widget.js"></script>

A tiny MDP — 70% chance to loop (stochastic $\pm 1$ reward) and 30%
chance to terminate at $+5$ — has a return distribution we can compute
exactly by iterating the categorical Bellman operator on a 21-atom
support. The blue bars are $P(Z = z)$; the dashed red line is $E[Z]$,
the only quantity a standard DQN ever learns. Slide $\gamma$ from 0
upward and watch the distribution spread, develop a second mode, and
eventually skew. The readout reports $\operatorname{Var}(Z)$ and
skewness — both rich gradient signals C51 and QR-DQN soak up that
mean-only DQN cannot.

---

## 9.8 NoisyNets and Rainbow

### NoisyNets

[Fortunato et al. 2018] replace $\epsilon$-greedy exploration with
**parametric noise on the network weights**:

$$w = \mu_w + \sigma_w \odot \xi, \quad \xi \sim \mathcal{N}(0, I).$$

The mean $\mu_w$ and stddev $\sigma_w$ are learned. Exploration becomes
state-dependent: the policy is noisy in proportion to how uncertain the
network is. Replaces a hand-tuned $\epsilon$ schedule with learned
exploration.

### Try it: parameter noise → action noise

<div id="ch9-noisy-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/noisy_net/widget.js"></script>

The scatter on the left is 200 sampled weight vectors w = μ + σ·ξ in
the (w₀, w₁) plane, each coloured by the argmax action it induces in a
4-action softmax. Push σ to 0: the cloud collapses onto μ — pure greedy.
Push σ up: the cloud crosses the quadrant boundaries and the marginal
π(a) on the right flattens toward uniform. That's "uncertainty becomes
policy stochasticity" — no separate ε-schedule needed.

### Rainbow

[Hessel et al. 2018] combine six DQN extensions in one agent:
1. Double DQN (bias fix).
2. Dueling networks (decomposition).
3. Prioritized Experience Replay (sample efficiency).
4. Multi-step (3-step) returns (bias/variance trade).
5. Distributional Q (C51 atoms).
6. NoisyNets (learned exploration).

The ablation table is one of the cleanest contributions of the paper:
remove any one piece and median performance drops. **The two single most
important pieces are distributional Q and prioritized replay**; double
DQN and multi-step are close behind; dueling and noisy nets contribute
the least but still positively.

Rainbow is the *de facto* baseline for discrete-action deep RL. It is
also large and finicky — a small change to any one piece can interact
with the others. Modern variants (Agent-57, MEME) push further; Rainbow
remains the canonical "good DQN" benchmark.

---

## 9.9 The deadly triad in DQN

Recall Chapter 10: linear semi-gradient TD converges on-policy and *can
diverge* off-policy. DQN is **all three legs of the deadly triad
simultaneously**:

1. **Function approximation**: a neural network.
2. **Bootstrapping**: the target uses the current $Q$.
3. **Off-policy data**: the replay buffer contains transitions from
   old policies.

By the [Baird 1995] story, this *should* diverge. In practice DQN
converges (often) because:

- The target network freezes one of the legs for $C$ steps.
- Replay-buffer uniform sampling weakens the policy-mismatch effect
  (compared to pure off-policy data from a very different policy).
- The neural network's implicit regularization (architecture, SGD bias)
  shrinks weights instead of letting them blow up.

[van Hasselt et al. 2018], **"Deep Reinforcement Learning and the Deadly
Triad"**, asks *when* DQN actually diverges. Headline answers:

- **Small target networks help.** A target network is approximately a
  fixed point; the smaller the network, the more stable.
- **Larger nets diverge more.** Surprising but reproducible — bigger
  networks have more capacity to chase moving targets into divergence.
- **Long bootstrap horizons help.** $n$-step targets with $n > 1$ reduce
  bootstrap dependence per step.
- **Prioritized replay can amplify divergence** if high-priority
  transitions are also high-bootstrap-dependence.

Translation: DQN's stability is a partial-engineering, partial-empirical
matter. There is no proof that it converges. We use it because in
practice it works *enough* on a wide-enough class of problems — and
when it does not, we have a list of knobs (target-update rate, buffer
size, replay priority, $n$-step length, network width) to retry.

### Try it: the deadly-triad truth table

<div id="ch9-triad-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/deadly_triad/widget.js"></script>

Three switches: function approximation (linear/tabular), bootstrap
(TD(0)/MC), and data (off-policy/on-policy). All eight cells run
Baird's 7-state environment to T = 400. The only red (diverged) cell
is `linear-td · off-policy` — every other config stays bounded. That's
the **jointness** of the deadly triad: flip *any* of the three legs
and ‖θ‖ stops exploding. Linear FA alone is fine. Bootstrap alone is
fine. Off-policy alone is fine. The combination kills you.

For the Simulator, this is the strongest argument *against* pivoting to
deep RL right now: the linear-FA agent gives provable on-policy
convergence (within the constant $\alpha$ noise floor) for free, and
the deadly-triad failure modes are well-understood and Chapter-15-fixable
without touching the architecture.

---

## 9.10 Project tie-in

### Why the Simulator does **not** use DQN

The project deliberately uses linear tile-coded Q
([`crates/engine/q_learning/`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/)),
not a neural Q-network. The reasons:

| Reason | Detail |
|---|---|
| **Cognition cadence** | Cognition runs every 10 ticks for every agent. With $N$ agents and $T$ ticks of simulation, that is $N \cdot T / 10$ Q-evaluations and updates. Linear FA is $O(\text{tilings}) = 16$ ops per eval. A small MLP is hundreds. The cost compounds. |
| **No GPU dependency** | The project targets a single-process Rust workload. A torch/burn neural-net dependency adds significant build complexity. |
| **Determinism canary** | The Simulator asserts bit-identical reproducibility across runs. Neural nets are deterministic *in principle* but harder to keep so in practice (cuDNN nondeterminism, atomic adds, etc.). |
| **Per-agent diversity** | Per-agent learning-rate memes ([`learning_rate.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/learning_rate.rs)) let individuals carry different $(\alpha, \gamma)$ without separate networks. Per-agent neural nets are prohibitive at scale. |
| **Debuggability** | A linear tile-coded weight at slot $h$ has a single number to inspect. A neural Q-value has no such direct decomposition. |
| **Convergence theorem** | On-policy linear TD has [Tsitsiklis & Van Roy 1997]. Deep Q has no such theorem. |

### What we would port if we ever pivoted

If the project ever moves to neural FA — for instance, to fix the
[joint-tiling collapse](10_function_approximation.md#89-the-joint-tiling-collapse)
end-to-end, or to learn cross-block correlations — the priority list
of DQN tricks to bring along would be:

1. **Dueling architecture (§9.4).** This is the architectural fix for
   the Q-bias bootstrap pathology. The value head $V(s)$ absorbs the
   alive baseline; the advantage head $A(s, a)$ decides actions. **Even
   if we stay with linear FA**, the dueling decomposition can be
   implemented as a separate $V$ learner alongside the $Q$ learner —
   this is **advantage learning**, Chapter 17.
2. **Double Q (§9.3).** One-line code change; removes the max bias that
   compounds in deep settings.
3. **Replay buffer with prioritized sampling (§9.5).** Would substantially
   help the L-suite's sparse-reward setting — prioritize the rare
   harvest transitions.
4. **$n$-step targets (§9.6).** $n = 5$ would propagate reward five
   ticks back instead of one, partly mitigating the
   $\gamma^{500} \approx 0$ catastrophe of long-horizon credit
   assignment. (Full mitigation is Chapter 19 — hindsight, successor
   features.)

Target networks (§9.2) only matter once the network is non-linear; for
the linear case the semi-gradient TD update is already stable
on-policy.

### The dueling architecture for tile-coded linear FA

You can apply the dueling idea *without* neural networks:

- Train one **state-value learner** $V(\phi(s))$ using TD on
  $(s, r, s')$ alone (no action argument).
- Train one **advantage learner** $A(\phi(s, a))$ using $\delta = r + \gamma V(s') - V(s) - A(s, a)$.
- Decide actions by $\arg\max_a A(s, a)$ — the dueling output is the
  advantage; $V$ is just for credit assignment.

This is **A2C-style advantage learning** in tile-coded form. Chapter 13
develops it as part of actor-critic methods. The Q-bias bootstrap
pathology dies on this construction because $V$ absorbs the
alive-baseline and $A$ only sees differences. See the dueling code in
`policy.rs` if/when it lands.

---

## 9.11 Exercises

1. **(DQN on CartPole.)** Implement DQN in PyTorch on the CartPole-v1
   environment. Buffer size 50K, target-update $C = 100$, MLP(64, 64),
   $\alpha = 10^{-3}$, $\epsilon$ annealed 1.0 → 0.05 over 10K steps.
   Report mean episode reward over the last 100 episodes vs. wall-clock.

2. **(Ablation.)** Take your CartPole DQN and disable, one at a time:
   the target network, the replay buffer (online updates only), the
   $\epsilon$-greedy exploration. Plot learning curves for the four
   configurations. Which is most catastrophic?

3. **(Double DQN one-liner.)** Modify the target computation in your
   DQN code to use Double-DQN style action selection. Verify the change
   is *literally* one line. Re-run CartPole and report the difference.

4. **(Dueling on tile-coded LFA.)** Implement the dueling decomposition
   on the Simulator's linear tile-coded learner: train a separate
   $V$-learner and $A$-learner, decide actions by $\arg\max_a A$.
   Re-run [`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
   on the *sated* arm (which currently fails due to the Q-bias
   pathology). Does the sated-arm assertion now pass?

5. **(PER sample complexity.)** On CartPole, compare DQN with uniform
   replay vs. PER with $\rho = 0.6, \beta = 0.4 \to 1.0$. Plot
   episodes-to-solve.

6. **(Distributional intuition.)** On a 1-state MDP with reward
   $\text{Unif}(-1, 1)$ at every step, $\gamma = 0.9$: what is the
   true return distribution? Compare to what C51 with 51 atoms in
   $[-10, 10]$ would estimate.

7. **(Deadly triad.)** Reproduce Baird's counterexample (from Chapter 17)
   with a small MLP instead of linear features. Does adding (a) target
   network, (b) replay buffer, (c) both stabilize it? Compare to the
   pure linear off-policy case.

8. **(Project pivot sketch.)** Sketch the code changes required to
   port the Simulator from tile-coded linear Q to neural Q with the
   dueling architecture. What stays the same? What changes?

---

## 9.12 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Baird 1995] — off-policy linear TD counterexample (§9.1, §9.9).
- [Bellemare, Dabney & Munos 2017] — C51 (§9.7).
- [Dabney et al. 2018] — QR-DQN (§9.7).
- [Fortunato et al. 2018] — NoisyNets (§9.8).
- [Hessel et al. 2018] — Rainbow (§9.8).
- [Mnih et al. 2015] — original DQN (§9.2).
- [Munos et al. 2016] — Retrace (§9.6).
- [Schaul et al. 2016] — PER (§9.5).
- [Tsitsiklis & Van Roy 1997] — linear TD convergence (§9.10).
- [van Hasselt 2010] — Double Q (tabular) (§9.3).
- [van Hasselt, Guez & Silver 2016] — Double DQN (§9.3).
- [van Hasselt et al. 2018] — deep deadly triad (§9.9).
- [Wang et al. 2016] — Dueling (§9.4).

## 9.13 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton & Barto 2018] | Ch. 16.5 (Atari case study) | Textbook synthesis |
| [Spinning Up](https://spinningup.openai.com/) | DQN section | Clean reference implementation |
| Rainbow paper [Hessel et al. 2018] | All | Single source for all 6 extensions and ablations |
| van Hasselt et al. 2018 (deadly triad) | All | When and why DQN actually diverges |

---

**Next:** [Chapter 12 — Policy Gradient](12_policy_gradient.md) — when
estimating values is the wrong objective.

# Chapter 13 — Actor-Critic Methods

> **Prerequisites:** Chapter [10](12_policy_gradient.md) (policy
> gradient, advantage function), [6](08_temporal_difference_learning.md)
> (TD learning), [8](10_function_approximation.md) (function
> approximation), [9](11_deep_q_learning.md) §9.4 (dueling architecture).

> **Citations:** [Konda & Tsitsiklis 1999] (foundational AC convergence);
> [Mnih et al. 2016] (A3C); [Schulman et al. 2015a] (TRPO);
> [Schulman et al. 2017] (PPO); [Haarnoja et al. 2018] (SAC);
> [Lillicrap et al. 2015] (DDPG); [Fujimoto, van Hoof & Meger 2018]
> (TD3); [Silver et al. 2014] (deterministic policy gradient). Full
> entries in [`bibliography.md`](bibliography.md).

> **Learning objectives:**
> 1. Combine policy parameterization (actor) with value learning (critic);
>    derive the A2C update.
> 2. Read PPO's clipped objective and explain *why* the clip works.
> 3. Recognize SAC's entropy bonus as automatic exploration.
> 4. State the Q-bias bootstrap pathology as a missing-advantage problem,
>    and write down the *one-line* fix in the Simulator's
>    [`policy.rs`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) score
>    formula.

## Why this chapter exists

Chapter 12 introduced policy gradient and showed that subtracting an
estimate of $V^{\pi}(s)$ — using the **advantage** $A(s, a) = Q(s, a) - V(s)$ —
dramatically reduces variance without biasing the gradient. The natural
next question: where does that $V^{\pi}$ estimate come from?

**Actor-critic** methods learn $V$ (the *critic*) and $\pi$ (the *actor*)
simultaneously. The critic produces advantages that drive the actor's
gradient; the actor produces trajectories that train the critic.

This pairing has practical consequences beyond variance reduction:

- It is the only family that handles **continuous action spaces** well
  at scale (DDPG, TD3, SAC).
- It is the **modern default** for on-policy RL (PPO).
- It is the architectural template for "fix the alive-baseline" problems
  like the **Q-bias bootstrap pathology** in the Simulator — see §11.7
  and §11.8.

## Table of contents

- [11.1 The actor and the critic](#111-the-actor-and-the-critic)
- [11.2 A2C — Advantage Actor-Critic](#112-a2c--advantage-actor-critic)
- [11.3 A3C and the asynchronous trick](#113-a3c-and-the-asynchronous-trick)
- [11.4 Trust regions: TRPO and PPO](#114-trust-regions-trpo-and-ppo)
- [11.5 SAC: entropy-regularized actor-critic](#115-sac-entropy-regularized-actor-critic)
- [11.6 DDPG and TD3: deterministic policies](#116-ddpg-and-td3-deterministic-policies)
- [11.7 Advantage as the architectural Q-bias fix](#117-advantage-as-the-architectural-q-bias-fix)
- [11.8 Project tie-in: the Simulator is *almost* actor-critic](#118-project-tie-in-the-simulator-is-almost-actor-critic)
- [11.9 Exercises](#119-exercises)
- [11.10 References](#1110-references-cited-in-this-chapter)
- [11.11 Further reading](#1111-further-reading)

---

## 11.1 The actor and the critic

### Why this hybrid architecture exists

Chapter 12's REINFORCE works in theory — the policy gradient
estimator is unbiased — but in practice it's *miserable*. Returns
$G_t$ are high-variance (every random future transition contributes
noise), updates only happen once per episode (no online learning),
and a bad trajectory can swing $\theta$ wildly because there's
no baseline to compare against.

Chapter 11's deep Q-learning works in practice but inherits the
classical control problem with $Q$: continuous action spaces
require $\arg\max_a Q(s, a)$ over a continuous set, which is its
own optimization problem at decision time. And Q-learning has no
*natural* way to be stochastic — the policy is forced-deterministic
($\arg\max$), so exploration is bolted on via $\epsilon$-greedy or
noise injection.

**Actor-critic methods are the synthesis.** Keep a policy gradient
(so the policy can be naturally stochastic and continuous-action),
but replace the noisy Monte Carlo return $G_t$ with a low-variance
critic-estimated advantage $\hat A(s, a)$. The critic is what we
already learned to train in Chapter 8 (TD); the actor is what
Chapter 12 already justified (PGT). The hybrid is genuinely more
than the sum: A2C, PPO, SAC, DDPG, and TD3 are all actor-critic
variants and *all* of modern continuous-control deep RL lives here.

### The two parameterized objects

- **Actor**: $\pi_\theta(a \mid s)$ — a stochastic (or
  deterministic) policy. Updated by policy gradient (Ch12).
- **Critic**: $V_\phi(s)$ — a value function. Updated by TD or
  Monte Carlo regression (Ch8, Ch10).

The actor decides *what to do*; the critic evaluates *how good
states are*. The two parameter vectors $\theta$ and $\phi$ are
*independent* (no shared parameters in the simplest version; can
share trunk layers in deep AC for representation efficiency).

### The architecture, in one diagram

```mermaid
flowchart LR
    S[State s] --> Actor
    S --> Critic
    Actor -->|action a ~ π| Env[Environment]
    Env -->|r, s'| Critic
    Critic -->|advantage A(s,a)| Actor
    Critic -->|TD error δ| Critic
```

Two coupled learning loops:

1. **Critic self-training** — the TD error $\delta_t$ is the
   critic's regression target (Chapter 8 §6.1). Each step, $\phi$
   moves to make $V_\phi$ closer to its Bellman target.
2. **Critic feeds actor** — the advantage $A(s, a) \approx Q^\pi(s, a) - V^\pi(s)$
   is the multiplier on the actor's policy gradient (Ch12's PGT
   with a state-dependent baseline; §10.4 specifically).

Because the two loops update *different* parameters ($\phi$ vs
$\theta$), they can be co-optimized without one stomping on the
other. The coupling is through the *signal* $\delta_t$, not through
shared weights.

### Three ways the critic helps the actor

**1. Variance reduction.** REINFORCE uses sampled returns $G_t$ as
the policy-gradient weight. $G_t$'s variance scales with episode
length — every random transition in the future contributes noise to
the *current* update. The TD advantage $\delta_t = r + \gamma V(s') - V(s)$
substitutes a single-step prediction error for the whole-trajectory
return, bounding the variance to the TD-error magnitude (typically
$O(1)$, not $O(\text{horizon})$).

**2. Online updates.** REINFORCE waits for episode termination
(needs $G_t$); actor-critic updates after every step (needs only
$\delta_t$). For continuing tasks (no terminal state) this isn't an
optimization, it's a *requirement* — REINFORCE simply doesn't apply.

**3. Bias-variance tunability.** The 1-step TD advantage is biased
(it uses the approximate $V_\phi$, not $V^\pi$) but low-variance.
Monte Carlo's $G_t - V(s_t)$ is unbiased but high-variance.
GAE($\lambda$) (§10.6) interpolates between them with a single
$\lambda$ knob. The practitioner picks based on the problem; the
mathematician proves the bias bound; everyone wins.

### What this hybrid doesn't say

- **It's not free of REINFORCE's correctness condition.** The
  actor's gradient is still on-policy — trajectories collected
  under an old $\pi_\theta$ can't be used directly for the actor
  update (the critic update is fine off-policy). Importance
  sampling (PPO, V-trace) handles old data for the critic but
  not for the actor naively.
- **The critic must track $V^{\pi_\theta}$ as $\pi_\theta$ moves.**
  This is a non-stationary regression target: the critic chases a
  moving function. With slow actor updates and a well-tuned critic
  learning rate, the critic stays "ahead." With fast actor updates
  the critic lags, and the advantage signal becomes stale.
- **Convergence is local.** [Konda & Tsitsiklis 1999] proved
  convergence to a *stationary point* of the expected-return
  objective — not necessarily the global optimum. Empirically A2C
  reliably finds *good* policies, not always the best. PPO's
  trust-region machinery (§11.4) is partly an answer to this.

---

## 11.2 A2C — Advantage Actor-Critic

The cleanest realization. At each step:

> **A2C update.**
>
> **Critic** (TD on $V_\phi$):
>
> $$
> \phi \leftarrow \phi + \alpha_c \cdot \delta_t \cdot \nabla_\phi V_\phi(s_t), \quad \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
> $$
>
>
> **Actor** (policy gradient with critic-provided advantage):
>
> $$
> \theta \leftarrow \theta + \alpha_a \cdot \delta_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t).
> $$
>

Two updates, one TD error $\delta_t$ feeding both. Read it: the same
prediction error drives the critic toward better fitting and the actor
toward actions that achieve better-than-baseline outcomes.

### The "A" vs "the V": is it really an advantage?

$\delta_t = r + \gamma V(s') - V(s)$ is the **TD advantage** — a
1-step estimator of $A^{\pi}(s_t, a_t)$. Higher-quality estimators
exist (GAE($\lambda$) interpolates; full $n$-step boots beyond 1-step):

| Estimator | $\hat{A}_t$ | Bias | Variance |
|---|---|---|---|
| 1-step TD | $r_t + \gamma V(s') - V(s)$ | High | Low |
| $n$-step | $\sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$ | Medium | Medium |
| GAE($\lambda$) | $\sum_l (\gamma\lambda)^l \delta_{t+l}$ | Tunable | Tunable |
| Monte Carlo | $G_t - V(s_t)$ | Zero | High |

Modern A2C/PPO use GAE($\lambda$) with $\lambda = 0.95$.

### Convergence

[Konda & Tsitsiklis 1999] proved convergence of the linear actor-critic
under standard SGD assumptions and policy-class regularity. The conditions
are stronger than for pure value methods (the actor's policy class must
be smooth enough), and the result is *local* (a stationary point of the
expected-return objective, not necessarily the global optimum). In
practice this matches what we see: A2C reliably finds *good* policies,
not always the best.

### Synchronous vs. asynchronous

The original "AC" methods updated step-by-step, online. Modern A2C runs
**$N$ parallel environments** and aggregates gradients at fixed
intervals ("batched on-policy"). This decorrelates updates similarly to
the replay buffer in DQN, without going off-policy.

---

## 11.3 A3C and the asynchronous trick

[Mnih et al. 2016] published the asynchronous variant. Multiple workers
each run an environment copy, compute gradients on their local
trajectories, and asynchronously push updates to a shared parameter
server.

What it bought:

- **CPU-only Atari at human level.** No replay buffer, no GPU required.
  This was a big result in 2016 because DQN's replay buffer is
  memory-heavy.
- **Decorrelated gradients.** Different workers explore different parts
  of state space; their gradients average out the per-worker correlation
  inside one trajectory.
- **Asynchronous communication.** Workers do not wait for global
  synchronization; they run at full speed and push gradients when
  ready.

A3C lost ground to PPO (§11.4) because asynchronous updates have subtle
correctness issues at scale (stale-gradient effects), and synchronized
A2C is empirically as good or better on a modern multi-GPU cluster. Read
A3C for historical context; do not start a new project with it.

---

## 11.4 Trust regions: TRPO and PPO

PG methods take gradient steps. A big policy-gradient step can move
the policy *very* far from where the critic's advantage estimate is
valid — and then the new policy is worse than the old one. This is the
**collapse problem** of vanilla PG.

### TRPO ([Schulman et al. 2015a])

Constrain how far the policy is allowed to move per update, measured in
KL divergence:

$$\max\_\theta \mathbb{E}\_{s, a \sim \pi\_{\text{old}}}\left[\frac{\pi\_\theta(a \mid s)}{\pi\_{\text{old}}(a \mid s)} A(s, a)\right] \quad \text{s.t.} \quad \mathbb{E}\_s\big[\mathrm{KL}(\pi\_{\text{old}}(\cdot \mid s) \,\|\, \pi\_\theta(\cdot \mid s))\big] \leq \delta.$$

The surrogate is an importance-sampled version of the policy-gradient
objective; the KL constraint bounds the policy update.

TRPO solves this via Fisher-information-matrix linear systems
(natural gradient). Mathematically elegant. Computationally expensive
(conjugate-gradient inside each update). Numerically finicky (rank-deficient
Fisher matrices on hard problems).

### See it: the KL trust region

<div id="ch11-kl-trust-region-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/kl_trust_region/widget.js"></script>

Three-action softmax in 2D logit space. The heatmap is
$\mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$; the thick orange contour
is the trust-region boundary $\mathrm{KL} = \delta$. Drag the advantage
sliders to point the vanilla-PG arrow somewhere — when it leaves the
trust region (red dashed), TRPO would project back along the ray. The
right panel shows the resulting policy change on the simplex.

### PPO ([Schulman et al. 2017])

The brilliant simplification: replace the hard KL constraint with a
**clipped surrogate objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\Big[\min\Big(r_t(\theta) A_t,\ \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t\Big)\Big]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$
is the importance ratio (1 at $\theta = \theta_{\text{old}}$).

How to read it:

- If the policy ratio $r_t \in [1 - \epsilon, 1 + \epsilon]$ (default
  $\epsilon = 0.2$), the unclipped term applies: standard policy
  gradient with advantage weighting.
- If $r_t > 1 + \epsilon$ (we *increased* the action probability a lot)
  and $A_t > 0$: clip to $1 + \epsilon$. We do not get *more* gradient
  signal for already-overshooting in a good direction. Prevents
  runaway.
- If $r_t > 1 + \epsilon$ and $A_t < 0$: the clip does NOT activate
  (the min picks the unclipped term). We *do* get the full negative
  gradient, pushing us back. Asymmetric.
- Symmetric clipping on the $r_t < 1 - \epsilon$ side.

### Try it: PPO's clipped surrogate

<div id="ch11-ppo-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/ppo_clip/widget.js"></script>

Slide ε to widen/narrow the clip region; flip the sign of A. The dashed
line is `r · A` (the naïve importance-weighted gradient target); the
solid line is what PPO actually optimises. Notice the clip flips
direction when A < 0 — PPO clips only updates that would be "too
aggressive in the bad direction".

PPO is the **modern on-policy default**. It is robust to hyperparameter
choices, runs at full SGD speed, and works on continuous and discrete
actions alike. If you do not know what to try, try PPO first.

### How PPO is actually used

```python
for iteration in range(num_iters):
    trajectories = collect(policy_theta, N_steps)  # rollout
    A = gae(trajectories, V_phi, gamma, lam=0.95)  # advantages
    for epoch in range(K):       # K = 4-10 typically
        for minibatch in shuffle(trajectories):
            r = pi_theta(a|s) / pi_old(a|s)
            L_clip = min(r * A, clip(r, 1-eps, 1+eps) * A).mean()
            L_v = (V_phi(s) - target).pow(2).mean()
            L = -L_clip + c_v * L_v - c_h * entropy(pi_theta)
            L.backward(); optimizer.step()
    pi_old = pi_theta  # update old policy
```

K epochs of small-batch SGD on each set of rollouts. Replaces hard KL
constraint with the clip + a few epochs of reusing data. Trades
mathematical purity for engineering simplicity. Wins.

### The three-way loss tug-of-war

<div id="ch11-loss-decomp-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/ac_loss_decomp/widget.js"></script>

The PPO objective is three terms: actor (clipped policy gradient),
critic (value MSE), and entropy bonus. The signed-bar panel shows each
term's contribution to the total $L$; the contour panel runs a tiny
gradient descent on a toy 2D parameter space where each term has its
own attractor. Crank $c_v$ to watch the trajectory swing toward the
critic's optimum; crank $c_h$ to see entropy keep the policy stochastic.

### The KL-penalty variant of PPO (history)

Before settling on the clip, [Schulman et al. 2017] also proposed a
*soft* version that puts the KL constraint into the objective as a
penalty with an adaptive coefficient:

$$L^{\text{KLPEN}}(\theta) = \mathbb{E}\big[r_t(\theta) A_t\big] - \beta \cdot \mathrm{KL}(\pi_{\text{old}} \| \pi_\theta),$$

with $\beta$ doubled when realised KL is too high, halved when too low.

<div id="ch11-adaptive-kl-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/adaptive_kl/widget.js"></script>

The widget runs the schedule: each iteration samples a KL value whose
mean scales as boldness² / β, then β doubles or halves to drag the
realised KL back toward $d_{\text{target}}$. Watch β log-step its way
into a band around the target. This is a learned Lagrange multiplier —
a classical control loop tacked onto SGD.

### Off-policy correction (V-trace, IMPALA)

PPO is on-policy; rollouts must come from the current $\pi_\theta$.
IMPALA (Espeholt et al. 2018) relaxes this with **V-trace**, a
truncated importance-sampling estimator that can train on slightly
stale rollouts. The trick: cap the importance ratios at $\bar\rho$
(for the fixed point) and $\bar c$ (for the trace product), so the
variance of multi-step IS does not blow up exponentially.

<div id="ch11-vtrace-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/vtrace_correction/widget.js"></script>

Raw importance sampling (red) has stddev growing exponentially with
horizon — unusable past a handful of steps. V-trace's $\min(\bar c, \rho)$
clip (green) keeps stddev bounded. The histogram at the bottom shows
the per-trial distribution: raw IS has a long right tail (the
exponentially compounded outliers); V-trace truncates it. This is
*why* IMPALA can train on data collected by older actors.

---

## 11.5 SAC: entropy-regularized actor-critic

[Haarnoja et al. 2018]. Modify the RL objective to include a
**policy entropy bonus**:

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t \big(r_{t+1} + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\big)\right].$$

The $\alpha$ knob (the **temperature**) balances reward against
entropy. Equivalent: solve a *soft* Bellman equation where the
optimal policy is a Boltzmann distribution over soft Q-values.

What this buys:

- **Automatic exploration.** High entropy = uniform; low entropy =
  greedy. The policy spreads mass over actions that are nearly equal
  in soft Q, without manual $\epsilon$ scheduling.
- **Robust policies.** An entropy-regularized policy hedges against
  model error (it does not lock onto a single overconfident action).
- **Multimodal policies.** Soft policies can represent stochastic
  optima — useful in matching games, exploration-required tasks.

SAC is **off-policy** (uses a replay buffer like DQN/DDPG). It learns
*two* Q-functions to mitigate overestimation bias (like TD3, next).
It is the empirical winner on most continuous-control benchmarks
(MuJoCo locomotion, robotic control).

**Automatic temperature tuning.** Modern SAC variants learn $\alpha$ as
a Lagrange multiplier to satisfy a target entropy constraint
$\mathbb{E}[\mathcal{H}(\pi)] = \mathcal{H}_\text{target}$. Removes the
last manually-tuned hyperparameter.

---

## 11.6 DDPG and TD3: deterministic policies

For continuous actions, a stochastic policy adds variance that may not
be useful (if the optimal policy is in fact deterministic). The
**Deterministic Policy Gradient theorem** ([Silver et al. 2014]) shows:

$$\nabla\_\theta J(\theta) = \mathbb{E}\_{s \sim d^{\mu\_\theta}}\big[\nabla\_a Q^{\mu}(s, a)|\_{a = \mu\_\theta(s)} \cdot \nabla\_\theta \mu\_\theta(s)\big].$$

The gradient passes through $Q$'s sensitivity to action choice via the
chain rule.

### DDPG ([Lillicrap et al. 2015])

Implements DPG with deep nets, target networks (like DQN), and a
deterministic actor $\mu_\theta(s)$ trained against a Q-critic. Exploration
is via additive noise (Ornstein-Uhlenbeck or simple Gaussian) on the
deterministic action.

Pros: sample-efficient on continuous control. Cons: brittle. The
deterministic actor overconfidently follows whatever the Q-critic says,
including bias toward overestimating Q.

### TD3 ([Fujimoto, van Hoof & Meger 2018])

Three fixes to DDPG:

1. **Clipped double Q.** Train two Q-networks; use the *min* as the
   target. Reduces overestimation.
2. **Delayed policy updates.** Update actor less frequently than critic;
   gives the critic time to converge to a good estimate before the actor
   chases it.
3. **Target smoothing.** Add noise to the target action when computing
   the critic target. Smooths value-function ridges that the
   deterministic actor would otherwise exploit.

TD3 became the practical-DDPG choice; SAC then overtook it in most
benchmarks via entropy regularization. Both remain useful; SAC tends
to be more robust, TD3 cheaper.

---

## 11.7 Advantage as the architectural Q-bias fix

This section connects every previous strand to the Simulator's central
bug — the **Q-bias bootstrap pathology**.

### The pathology, restated

The reward in
[`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs) is

$$R(s, s') = w_\text{alive} - 0.15 \sum_d d^2 - \text{bio costs}, \quad w_\text{alive} = 1.0.$$

Tile-coded $Q$-learning at $\alpha = 0.10$, $\gamma = 0.90$ pushes every
committed action's $Q$ toward $w_\text{alive} / (1 - \gamma) = 10.0$.
Whatever action gets committed first reaches that fixed point first;
its tile-coded weights dominate. The agent locks in.

### The advantage observation

What the policy actually needs is *which action is better than the
state's average*, not "what is the absolute value of doing this action."
That is precisely what $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$
encodes.

If we train a separate $V^{\pi}(s)$ and decide based on
$Q(s, a) - V(s)$ instead of $Q(s, a)$:

- $V(s)$ absorbs $w_\text{alive}$. Its fixed point is $10.0$ in
  every state.
- $Q(s, a) - V(s)$ has *zero mean* across actions by construction
  (when $\pi$ is greedy w.r.t. $Q - V$, the advantage of the chosen
  action is exactly $\max - \text{mean}$, but across all actions the
  expectation is zero).
- The argmax over $Q - V$ equals the argmax over $Q$ at the optimum,
  but the *learning dynamics* are completely different — only
  inter-action differences drive updates.

This is exactly the **Dueling DQN** intuition from §9.4 ported to linear
FA. Advantage learning (A2C) is the structural way to attack the
Q-bias bootstrap pathology — though it is *not* what ultimately shipped.
The pathology was resolved more directly by removing the $w_\text{alive}$
baseline from the reward (see §11.8); A2C remains the cleaner *structural*
alternative had a per-tick floor been required.

### Why this is not "just go to A2C"

A2C uses a stochastic policy. The Simulator's `policy.rs` uses
$\epsilon$-greedy argmax. The minimal fix is **value-based advantage
learning**: keep the argmax decision rule, but replace $Q$ with $Q - V$
in the scoring formula.

Concretely, in
[`policy.rs:474`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) today:
```rust
let total = drive_score + q_bias * q_score + recipe_bonus;
```
With advantage learning:
```rust
let total = drive_score + q_bias * (q_score - v_score) + recipe_bonus;
```
where `v_score` is a per-state baseline learned by parallel TD on $V$.
One line of code, one new tile-coded learner — the rest is plumbing.

### Feel it: the sated arm under plain Q vs. advantage learning

<div id="ch11-sated-arm-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/sated_arm_advantage/widget.js"></script>

A toy reproduction of the Simulator's `learning_homeostatic` sated arm:
two actions (Plant, Consume), $w_{\text{alive}} = 1$, $\gamma = 0.9$,
$\alpha = 0.1$. Top panel is plain Q-learning: both $Q$ values drift toward
$w_{\text{alive}} / (1 - \gamma) = 10$, and the margin between them is
dominated by *which one bootstrapped first*, not by the true reward
difference. Bottom panel is the advantage variant: $V$ absorbs the alive
baseline (still climbs to 10), and the advantages $A = Q - V$ stay
small and reflect the actual reward shape (the `Plant bias` slider).
Drag `w_alive` live and watch the plain-Q margin balloon while the
advantage margin holds steady.

---

## 11.8 Project tie-in: critic-only today, actor-critic as a possible pivot

The Simulator's score formula is:

$$\text{score}(s, a) = \underbrace{0.5 \cdot Q\_\theta(s, a)}\_{\text{critic-like}} + \underbrace{\text{recipe\_bonus}(s, a)}\_{\text{prior}}.$$

This is structurally a **critic-only** system with a procedural prior:

- **`Q_theta(s, a)`** — the **tile-coded Q-learner**. Acts as a
  *critic*, except scored as a reward-component, not as an advantage.
- **`recipe_bonus`** — a procedural-content prior over multi-step
  programs.

What is missing to be a proper advantage actor-critic:

1. **A learned actor** $\pi_\psi(a \mid s)$ alongside the critic.
2. **A learned $V_\phi(s)$** alongside $Q_\theta$.
3. **Replace $Q$ with $Q - V$** in the score sum and sample the actor.

The action selection is **argmax** rather than softmax-sampled — a
deterministic policy over the critic's scores.

### Why the project pivot to actor-critic might or might not happen

| For the pivot | Against the pivot |
|---|---|
| Fixes the Q-bias bootstrap pathology architecturally. | Adds a second learner per agent — 2x state. |
| Brings advantage signal that scales to longer horizons. | Recipe-bonus already partially expresses a prior over actions. |
| Standard pattern; better-studied. | Linear TD on-policy convergence still holds; advantage variant has fewer theorems. |
| One-line change in `policy.rs` plus new tile coder. | Validation tests calibrated against current $Q$ values would shift; recalibration cost. |

The pathology has since been **resolved** by removing the
$w_\text{alive}$ baseline entirely: reward is now the per-tick
drive-delta, which addresses the cause rather than the symptom. That
fix is simpler than a full actor-critic pivot and changes the reward
function's semantics for every dependent system. The advantage-learning
pivot described here remains the structural alternative had a per-tick
floor been required — it is a discussed technique, not what shipped.

### Validation test that would exercise advantage learning

[`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
**sated arm** is the cleanest indicator. Today the sated arm's
$Q_\text{Consume}$ margin reads $\sim -4.25$ (bootstrap pathology). With
advantage learning, the margin would land near 0 because $V(s)$
absorbs the alive baseline regardless of which action is committed. The
test assertion `q_consume_margin ∈ [-0.3, 0.1]` would pass without
needing to be loosened.

---

## 11.9 Exercises

1. **(A2C on CartPole.)** Implement A2C with separate MLPs for actor
   and critic. Train on CartPole-v1. Compare to your REINFORCE +
   baseline implementation from Chapter 12 — how much faster does AC
   converge?

2. **(PPO ratio clip.)** Plot the PPO surrogate $L^{\text{CLIP}}(r)$ as a
   function of $r$ for both $A > 0$ and $A < 0$. Identify the
   discontinuous behavior at the clip boundaries. Why is the asymmetric
   clip (active for $A > 0$ at the upper boundary, but not at the lower)
   the correct choice?

3. **(GAE in A2C.)** Implement GAE($\lambda$) advantage in your A2C.
   Sweep $\lambda \in \{0, 0.5, 0.95, 1.0\}$. Plot return-vs-episode.

4. **(SAC on Pendulum.)** Implement (or use a reference) SAC. Train on
   Pendulum-v1. Inspect the learned entropy temperature $\alpha$ over
   training — does it converge to a non-zero value or shrink to zero?

5. **(TD3 vs. DDPG.)** On a continuous-control benchmark, train DDPG
   and TD3 to convergence. Report the difference in episodic return
   distribution variance.

6. **(Project — advantage learning sketch.)** Sketch the implementation
   of a $V_\phi(s)$ learner alongside the Simulator's existing
   $Q_\theta(s, a)$. Detail: which tile coder is reused, which is new;
   how the TD updates for $V$ differ from those for $Q$; what
   `policy.rs:474` would look like after the change.

7. **(Q-bias bootstrap, advantage version.)** Repeat the §1.6 bootstrap
   calculation under advantage learning. If $V(s)$ converges to 10
   everywhere, what does $Q(s, a) - V(s)$ converge to for the
   committed-action drift? Confirm the alive-baseline cancels.

8. **(Trust-region intuition.)** On a 2-state MDP, explicitly write
   out the policy ratio $\pi_\theta / \pi_{\text{old}}$ as $\theta$
   varies. Identify the range over which the clip is active. Plot
   the surrogate objective as $\theta$ moves.

---

## 11.10 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Fujimoto, van Hoof & Meger 2018] — TD3 (§11.6).
- [Haarnoja et al. 2018] — SAC (§11.5).
- [Konda & Tsitsiklis 1999] — actor-critic convergence (§11.2).
- [Lillicrap et al. 2015] — DDPG (§11.6).
- [Mnih et al. 2016] — A3C (§11.3).
- [Schulman et al. 2015a] — TRPO (§11.4).
- [Schulman et al. 2016] — GAE (§11.2).
- [Schulman et al. 2017] — PPO (§11.4).
- [Silver et al. 2014] — deterministic policy gradient theorem (§11.6).
- [Sutton & Barto 2018, Ch. 13] — actor-critic chapter (§11.1-11.2).

## 11.11 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton & Barto 2018] | Ch. 13 §13.5 | Textbook actor-critic |
| [Spinning Up](https://spinningup.openai.com/) | A2C, PPO, SAC | Reference implementations |
| [PPO paper] | All | The single most-impactful modern RL paper |
| [SAC paper] | All | Continuous-control modern default |

---

**Next:** [Chapter 14 — Exploration](14_exploration.md) — what to do when
$\epsilon$-greedy is not enough.

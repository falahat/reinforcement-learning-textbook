# Chapter 12 — Policy Gradient Methods

> **Prerequisites:** Chapter [3](05_mdps_and_bellman_equations.md)
> (MDPs and value functions), [6](08_temporal_difference_learning.md)
> (Q-learning), [8](10_function_approximation.md) (linear FA),
> [3](03_mathematics_for_ai.md) (gradient, Hessian, SGD).

> **Citations:** [Williams 1992] (REINFORCE); [Sutton et al. 2000]
> (Policy Gradient Theorem with function approximation);
> [Konda & Tsitsiklis 1999] (Actor-Critic convergence); [Schulman et al.
> 2016] (GAE); [Sutton & Barto 2018, Ch. 13]. Full entries in
> [`bibliography.md`](bibliography.md).

> **Learning objectives:**
> 1. State the Policy Gradient Theorem and derive it cleanly via the
>    log-derivative trick.
> 2. Implement REINFORCE and explain its variance problem.
> 3. Subtract a baseline; recognize that the optimal baseline is $V^{\pi}$
>    and the resulting estimator uses the **advantage function**
>    $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$.
> 4. Parameterize stochastic policies (softmax for discrete actions,
>    Gaussian for continuous).
> 5. Build the Generalized Advantage Estimator (GAE($\lambda$)) and
>    pick a $\lambda$ for a given bias-variance budget.

## Why this chapter exists

So far, every method estimated **values** ($V$, $Q$) and derived a policy
by taking the argmax. That has three structural weaknesses:

1. **Continuous action spaces.** $\arg\max_{a} Q(s, a)$ over $a \in \mathbb{R}^d$
   is itself an optimization problem at every step. Even if you can do it,
   you have replaced one continuous problem with a nested second one.
2. **Large discrete action spaces.** With $|\mathcal{A}| = 10^4$ actions
   (combinatorial, parameterized), enumerating $Q(s, a)$ for argmax is
   prohibitive every tick.
3. **Stochastic-optimal policies.** Some MDPs have *only* stochastic
   optimal policies (matching games, exploration-required tasks).
   Argmax over Q always gives a deterministic policy — wrong by
   construction.

**Policy gradient** methods skip $Q$ entirely. They parameterize the
policy $\pi(a \mid s; \theta)$ directly as a distribution, and do
gradient ascent on expected return. The price: instead of a Bellman
fixed point, we get a *local* optimum of a non-convex objective with
high-variance gradient estimates. Most of the chapter is about taming
that variance.

## Table of contents

- [10.1 The policy-gradient setup](#101-the-policy-gradient-setup)
- [10.2 The Policy Gradient Theorem](#102-the-policy-gradient-theorem)
- [10.3 REINFORCE](#103-reinforce-monte-carlo-policy-gradient)
- [10.4 Baselines and the advantage function](#104-baselines-and-the-advantage-function)
- [10.5 Parameterizing stochastic policies](#105-parameterizing-stochastic-policies)
- [10.6 Generalized Advantage Estimation](#106-generalized-advantage-estimation-gae)
- [10.7 Policy gradient vs. Q-learning — a comparison](#107-policy-gradient-vs-q-learning--a-comparison)
- [10.8 Project tie-in](#108-project-tie-in)
- [10.9 Exercises](#109-exercises)
- [10.10 References](#1010-references-cited-in-this-chapter)
- [10.11 Further reading](#1011-further-reading)

---

## 10.1 The policy-gradient setup

Let $\pi_\theta(a \mid s) = \pi(a \mid s; \theta)$ be a parameterized
policy: a probability distribution over actions, smoothly parameterized
by $\theta$. Examples below; for now, just take it as a black-box
density.

Define the **performance** objective as expected discounted return from
the start-state distribution $\rho_0$:

$$J(\theta) = \mathbb{E}\_{s\_0 \sim \rho\_0, \tau \sim \pi\_\theta}\Big[\sum\_{t=0}^{\infty} \gamma^t r\_{t+1}\Big] = \mathbb{E}\_{s\_0 \sim \rho\_0}[V^{\pi\_\theta}(s\_0)].$$

We want $\theta^{\star} = \arg\max_\theta J(\theta)$. Gradient ascent updates
$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. The whole
chapter is "what is $\nabla_\theta J$, and how do we estimate it from
sampled trajectories?"

The challenge: $J$ depends on $\theta$ through the trajectory
distribution $\tau \sim \pi_\theta$. Differentiating an expectation
whose distribution depends on the parameter is non-trivial.

---

## 10.2 The Policy Gradient Theorem

### Why we need a theorem at all

Recall the setup from §10.1: we want $\nabla_\theta J(\theta)$, the
gradient of expected return with respect to the policy parameters.
The obstacle is that $\theta$ appears not in the *integrand* of the
expectation but in the *distribution* itself — the trajectories we'd
average over are drawn from $\pi_\theta$, and $\pi_\theta$ changes as
$\theta$ moves.

In ordinary supervised learning that's not a problem: the training
distribution is fixed, the loss is a function of the parameters
through the *output*, and backprop computes the gradient by walking
the chain rule from output to parameters (Chapter 11). Here the
distribution itself is what we're learning, and you can't backprop
through a sampler in the obvious way: the sample is a discrete
choice, and you can't differentiate through "I sampled action 3."

The policy gradient theorem solves this. It rewrites
$\nabla_\theta J$ as an expectation under $\pi_\theta$ that we *can*
sample — pushing the parameter dependence from the distribution into
the integrand, where ordinary differentiation works. The whole edifice
of policy-based RL (REINFORCE, A2C/A3C, PPO, TRPO, SAC) sits on this
one identity. Read this section slowly.

If you haven't internalised the log-likelihood trick from probability
or maximum-likelihood estimation, the next subsection introduces it
from scratch. If you have, skim — but the *application* to
trajectories (a few paragraphs down) is the part that's specific to RL.

### The log-derivative trick — what it is and why it matters

**The claim.** For any family of distributions $p_\theta$ parameterized
smoothly by $\theta$, and any function $f(x)$ that does *not* depend
on $\theta$,

$$\nabla _\theta\, \mathbb{E} _{x \sim p _\theta}[f(x)] = \mathbb{E} _{x \sim p _\theta}\big[f(x)\, \nabla _\theta \log p _\theta(x)\big].$$

**In words.** The gradient of an expectation under a parameterized
distribution equals an expectation (under the *same* distribution) of
$f$ multiplied by the gradient of the log-density. The left side is
hard to compute because the *distribution itself* depends on
$\theta$. The right side is easy: it's a plain expectation over
$p_\theta$ — we sample from $p_\theta$, evaluate $f$, multiply by the
score $\nabla \log p_\theta$, and average. Monte Carlo handles the
rest.

**Why this matters.** This single identity turns "differentiating
through a sampler" into "ordinary Monte Carlo estimation of an
expectation." Without it, policy gradients would require either
(a) a closed-form for $J$ — which we don't have, since dynamics are
unknown — or (b) finite differences in $\theta$ space — which would
mean rolling out *two* trajectories per parameter dimension, which
is hopeless for million-parameter policies. The log-derivative
trick is what makes policy-based deep RL *possible*.

**Derivation, step by step.** Write the expectation as an integral
and pull the gradient inside:

$$\nabla _\theta\, \mathbb{E} _{x \sim p _\theta}[f(x)] = \nabla _\theta \int p _\theta(x)\, f(x)\, dx \stackrel{(1)}{=} \int \nabla _\theta\, p _\theta(x)\, f(x)\, dx.$$

Step (1) — *interchanging gradient and integral* — needs technical
justification (dominated convergence, etc.) but holds for every
distribution family we care about: any smoothly-parameterized neural
policy with finite-variance returns. We'll take it as given.

We now have $\nabla_\theta p_\theta$ inside the integral, but that's
not yet an expectation under $p_\theta$ — there's no $p_\theta$ factor
out front. The trick: multiply and divide by $p_\theta$,

$$\int \nabla _\theta\, p _\theta(x)\, f(x)\, dx = \int p _\theta(x) \cdot \frac{\nabla _\theta\, p _\theta(x)}{p _\theta(x)} \cdot f(x)\, dx \stackrel{(2)}{=} \int p _\theta(x) \cdot \nabla _\theta \log p _\theta(x) \cdot f(x)\, dx.$$

Step (2) uses the calculus identity
$\nabla \log p = (\nabla p) / p$ (chain rule on $\log$). The right
side is exactly the form of an expectation under $p_\theta$:

$$\int p _\theta(x) \cdot \big[\nabla _\theta \log p _\theta(x) \cdot f(x)\big]\, dx = \mathbb{E} _{x \sim p _\theta}\big[f(x)\, \nabla _\theta \log p _\theta(x)\big].$$

Done. The whole derivation is three lines and one calculus identity
(`d/dx log p = (dp/dx)/p`); the *concept* is the harder part.

**Sanity check — the constants case.** Take $f(x) = 1$ (a constant
function). Then $\mathbb{E}[f] = 1$ for any distribution, so the
gradient is zero. The right-hand side gives
$\mathbb{E}[1 \cdot \nabla \log p_\theta(x)] = \mathbb{E}[\nabla \log p_\theta(x)]$.
For this to be zero, the *expected score under the distribution*
must be zero. And it is — a classical result you may know from
maximum-likelihood theory:
$\mathbb{E} _{x \sim p}[\nabla _\theta \log p _\theta(x)] = 0$.
(Differentiate $\int p\, dx = 1$ with respect to $\theta$.) The
identity holds. ✓

**Where else this shows up.** This is the **score function
estimator**, also called the **REINFORCE estimator**, and it's
genuinely universal:
- **REINFORCE** (this chapter) — RL's policy gradient.
- **Variational inference** — gradient of the ELBO when the
  variational posterior is parameterized.
- **Evolution strategies** (NES, OpenAI ES) — search in parameter
  space using the same identity.
- **Black-box derivative-free optimization** when only function
  evaluations are available.

If you remember one identity from this chapter, make it this one.

### Stitching the log-derivative trick onto trajectories

We want $\nabla_\theta J$ where the random variable is a whole
trajectory $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$ sampled
under policy $\pi_\theta$ and environment dynamics $P$. To apply the
log-derivative trick, we need the density $p_\theta(\tau)$ and its
log. Both follow from the Markov property (Chapter 5).

The trajectory's joint density factorizes as a product of one-step
terms:

$$p _\theta(\tau) = \rho _0(s _0) \cdot \prod _t \pi _\theta(a _t \mid s _t) \cdot P(s _{t+1} \mid s _t, a _t).$$

Taking the log turns the product into a sum:

$$\log p _\theta(\tau) = \log \rho _0(s _0) + \sum _t \log \pi _\theta(a _t \mid s _t) + \sum _t \log P(s _{t+1} \mid s _t, a _t).$$

Now the *crucial* observation. The dynamics $P$ are a property of the
*environment* — they do not depend on $\theta$. Neither does the
initial-state distribution $\rho_0$. So when we differentiate with
respect to $\theta$, every term except the policy log-likelihoods
vanishes:

$$\nabla _\theta \log p _\theta(\tau) = \sum _t \nabla _\theta \log \pi _\theta(a _t \mid s _t).$$

**This is why policy-gradient methods are model-free.** Even though
the *trajectory* depends on the unknown dynamics $P$, the *gradient
of its log-likelihood* does not. We never need to know $P$ to take a
gradient step — we only need to know our own policy.

Pause on that. In Chapter 6 (DP) we needed $P$ to compute the
Bellman backup. In Chapter 8 (TD) we *implicitly* used $P$ via sampled
transitions. Here, $P$ has been algebraically eliminated — its
contribution to the gradient is exactly zero. That algebraic
cancellation is what makes policy gradients *uniquely* model-free
among gradient-based methods.

### Putting it together — the theorem

Combine the log-derivative trick (with $f(\tau) = R(\tau)$ the total
return of the trajectory) and the trajectory-log-density identity
above:

$$\nabla _\theta J(\theta) = \nabla _\theta\, \mathbb{E} _{\tau \sim \pi _\theta}[R(\tau)] = \mathbb{E} _{\tau \sim \pi _\theta}\left[R(\tau) \cdot \nabla _\theta \log p _\theta(\tau)\right] = \mathbb{E} _{\tau \sim \pi _\theta}\left[R(\tau) \cdot \sum _t \nabla _\theta \log \pi _\theta(a _t \mid s _t)\right].$$

This is *almost* the policy gradient theorem. The "almost" is that
every score $\nabla \log \pi_\theta(a_t \mid s_t)$ gets multiplied by
the *whole* return $R(\tau) = \sum_{k=0}^{\infty} \gamma^k r_{k+1}$,
including rewards from *before* time $t$. That can't be right
causally — action $a_t$ can't affect $r_1$ — and indeed those
pre-$t$ rewards contribute pure noise to the estimator. The
**causality trick** (next subsection) drops them.

> **Policy Gradient Theorem ([Williams 1992]; full FA version [Sutton et al. 2000]).**
>
> $$
> \nabla _\theta J(\theta) = \mathbb{E} _{\tau \sim \pi _\theta}\left[\sum _t \nabla _\theta \log \pi _\theta(a _t \mid s _t) \cdot G _t\right]
> $$
>
> where $G_t = \sum_{k=t}^{\infty} \gamma^{k-t} r_{k+1}$ is the
> discounted return *from time $t$ onward* (Chapter 4, §3.3).

### The causality trick — and why each step is honest

Replacing $R(\tau)$ with $G_t$ in the score-weighted sum is the
*causality trick*. Intuitively: action $a_t$ can only influence
rewards $r_{t+1}, r_{t+2}, \ldots$ — not $r_1, \ldots, r_t$. So
weighting the score at time $t$ by pre-$t$ rewards is averaging in
noise. Removing those terms gives the same expectation with lower
variance. Here's the proof, line by line.

**Claim.** For any $k \leq t$,

$$\mathbb{E} _{\tau \sim \pi _\theta}\big[\nabla _\theta \log \pi _\theta(a _t \mid s _t) \cdot r _k\big] = 0.$$

**Proof.** Condition on the partial trajectory up to time $t$ — call
it $\tau_{<t} = (s_0, a_0, r_1, \ldots, s_t)$. The reward $r_k$ with
$k \leq t$ is a function of $\tau_{<t}$ alone (it was already
realised before $a_t$ was sampled). The score
$\nabla \log \pi_\theta(a_t \mid s_t)$ is a function of $a_t$ and
$s_t$. By the tower property of conditional expectation,

$$\mathbb{E}[\nabla \log \pi _\theta(a _t \mid s _t) \cdot r _k] = \mathbb{E}\Big[\, \mathbb{E}[\nabla \log \pi _\theta(a _t \mid s _t) \cdot r _k \mid \tau _{<t}] \,\Big].$$

Since $r_k$ is $\tau_{<t}$-measurable (i.e., known given the partial
trajectory), it pulls out of the inner expectation:

$$\mathbb{E}[\nabla \log \pi _\theta(a _t \mid s _t) \cdot r _k \mid \tau _{<t}] = r _k \cdot \mathbb{E}[\nabla \log \pi _\theta(a _t \mid s _t) \mid \tau _{<t}].$$

The inner expectation is the expected score under $\pi_\theta(\cdot \mid s_t)$ —
which is zero, by the sanity-check property from the
log-derivative-trick subsection
($\mathbb{E}_{x \sim p}[\nabla \log p(x)] = 0$). So the whole
expression vanishes. ☐

This means we can subtract $\sum_{k \leq t} \gamma^{k-1} r_k$ from
the coefficient of every $\nabla \log \pi_\theta(a_t \mid s_t)$
*without changing the gradient*. What's left is exactly
$G_t = \sum_{k \geq t+1} \gamma^{k-t-1} r_k$ — the return from time
$t$ onward, normalised so $G_t$'s first term has coefficient $1$.

The **bias** of the new estimator is zero (we subtracted a
mean-zero quantity); the **variance** drops because we're no longer
multiplying the score at time $t$ by noisy pre-$t$ rewards.

<div id="ch10-causality-trick-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/causality_trick/widget.js"></script>

Each cell is action $t$'s weight on reward $k$. Toggle "with
causality" on: the lower triangle is exactly zero — $a_t$ can't
affect rewards before $t$, so they're dropped from the gradient for
free. Toggle it off and every cell lights up — that's the naive
$G_0$ form, paying full variance for nothing. Slide $\gamma$ down to
watch the post-$t$ tail shrink; long-horizon credit dies
geometrically.

### Three ways to read the theorem

The formula
$\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(a_t \mid s_t) \cdot G_t]$
admits at least three intuitive readings — each one valid, each one
useful at different moments.

**1. As policy-iteration-in-disguise.** Look at one term:
$\nabla \log \pi_\theta(a_t \mid s_t)$ is the direction in
parameter space that increases the log-probability of $a_t$ at $s_t$.
Multiplying by $G_t$ scales the step: large $G_t$ → big push toward
$a_t$, small $G_t$ → small push, negative $G_t$ → push *away*. So
the update is "look at every action you took; nudge your policy to
take *more* of the ones that worked, less of the ones that didn't."
That's policy iteration's improvement step, just on a probability
density instead of a deterministic policy.

**2. As a maximum-likelihood update weighted by return.** If
all $G_t$ were $1$, the update would be ordinary maximum-likelihood
estimation of the trajectory distribution under $\pi_\theta$ — i.e.
imitation learning where the demonstrations are your *own* rollouts.
The return $G_t$ weights *which* of your own demonstrations to
imitate more heavily. Trajectories that worked well get weighted up,
ones that didn't get weighted down (or weighted negatively →
*anti*-imitated).

**3. As a stochastic-policy generalisation of the deterministic
chain rule.** With a *deterministic* policy $a = \mu_\theta(s)$, the
ordinary chain rule gives
$\nabla_\theta J = \nabla_a Q(s, a) \cdot \nabla_\theta \mu_\theta(s)$
(deep deterministic policy gradient — DDPG, Chapter 20). The
stochastic version replaces the $\nabla_a Q$ term — which requires a
known critic — with the score-weighted *sampled* return, which
doesn't. The theorem is the missing-info analogue of straight
backprop through the action.

Different problems make different readings useful. Reading 1 is
how you remember the algorithm; reading 2 is how you remember the
variance issues (low-return trajectories add noise — Chapter
10.4); reading 3 is how the deterministic and stochastic policy-
gradient families relate.

### What the theorem doesn't say

A few traps worth flagging before §10.3 builds REINFORCE on top.

- **It's not a free lunch.** The estimator is unbiased but the
  variance can be enormous. Even one bad trajectory with a huge
  negative return can swing $\theta$ wildly. Sections 10.4 and 10.6
  (baselines, GAE) exist specifically to control that variance.
- **It doesn't say which $\pi_\theta$ to pick.** Discrete actions →
  softmax; continuous → Gaussian; structured action spaces → something
  problem-specific. §10.5 walks the catalog.
- **It's on-policy.** The expectation is *under the current $\pi_\theta$*.
  You cannot reuse trajectories collected by an older policy without
  importance-sampling corrections. Off-policy policy gradients exist
  (V-trace, Chapter 13; off-policy actor-critic, Chapter 13) but
  they're a meaningful extension, not a free generalisation.
- **It assumes the policy is differentiable in $\theta$.** A policy
  that hard-thresholds on the logit (one-hot at $\arg\max$) has
  $\nabla \log \pi_\theta = 0$ almost everywhere and the theorem
  degenerates. The smooth-parameterization assumption is what makes
  gradient ascent meaningful at all.

<div id="ch10-causality-trick-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/causality_trick/widget.js"></script>
Each cell is action t's weight on reward k. Toggle "with causality" on: the lower triangle is exactly zero — a's at time t can't have caused rewards before t, so they're dropped from the gradient for free. Toggle it off and every cell lights up: that's the naive G₀-form, paying full variance for nothing. Slide γ down to watch the post-t tail shrink — long-horizon credit dies geometrically.

### What the theorem says, in English

The gradient of expected return is the **expected weighted sum of policy-
score-function-gradients**, weighted by **how good the return turned out
to be**.

- If $G_t$ is large at $(s_t, a_t)$: push $\log \pi(a_t \mid s_t)$ up —
  make this action more likely next time.
- If $G_t$ is small (or negative): push it down.

This is just "good behaviors → more likely; bad behaviors → less likely."
The mathematical content is that the *quantity* of pushing follows a
specific formula that, on average over many trajectories, gives the
exact gradient of expected return. No bias.

---

## 10.3 REINFORCE: Monte Carlo policy gradient

[Williams 1992] gave the simplest realization. Sample a complete episode
under $\pi_\theta$; compute returns $G_t$ at each step; update:

> **REINFORCE update.**
>
> $$
> \theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t.
> $$
>

In pseudocode:

```python
for episode in range(num_episodes):
    trajectory = run_episode(policy_theta)
    G = compute_returns(trajectory, gamma)
    for t, (s, a, _) in enumerate(trajectory):
        loss = -log_prob(policy_theta, s, a) * G[t]
        loss.backward()
    optimizer.step()
```

The negative log-prob × return is the **policy-gradient surrogate loss**;
SGD on it does gradient *ascent* on $J$.

### Try it: REINFORCE on a 2-arm bandit

<div id="ch10-softmax-bandit-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/softmax_bandit/widget.js"></script>

The simplest stochastic policy: $\pi(a=1) = \sigma(\theta)$ on a 2-arm Gaussian bandit. Slide $\mu_1, \mu_2$ closer together and watch $\theta$ drift toward the better arm more slowly — the per-step expected drift is $\alpha \cdot \pi(1-\pi) \cdot (\mu_2 - \mu_1)$, so shrinking the gap directly shrinks learning. Raise $\sigma$ to see the per-step gradient noise that REINFORCE inherits from $G_t$. The curves you see *are* exercise 12.

### Why REINFORCE is hard in practice

REINFORCE is an *unbiased* gradient estimator: $\mathbb{E}[\hat{\nabla J}] = \nabla J$.
The estimator is also *catastrophically high variance*. Three reasons:

1. **Return variance.** $G_0$ depends on the entire stochastic episode;
   its variance scales with episode length.
2. **Score-function variance.** $\nabla \log \pi$ has its own variance,
   amplified for poorly-parameterized policies (e.g., low-entropy
   nearly-deterministic policies have huge $\nabla \log \pi$).
3. **No information-sharing across actions.** Each trajectory only
   updates the actions actually taken; alternative actions are not
   informed.

Empirically REINFORCE often takes 10–100× more episodes than Q-learning
on the same task. This is why nobody uses pure REINFORCE today — the
rest of the chapter is variance-reduction tricks.

<div id="ch10-reinforce-variance-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/reinforce_variance/widget.js"></script>
Same 2-arm bandit, same true ∇J — only the baseline changes. Histogram of |ĝ| across 500 single-episode estimates: "none" spreads wide, "mean" tightens noticeably, "critic V" tightens further. The vertical rule (empirical mean) lands in the same place for all three — that's the baseline lemma in pictures: subtracting any state-only function leaves the gradient unbiased but slashes its variance.

### Try it: live REINFORCE on CartPole

<div id="ch12-reinforce-cartpole-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/reinforce_cartpole/widget.js"></script>

The previous widget was a single bandit step; this one is the full
algorithm. A 2-hidden-layer policy network is trained from scratch
in your browser using the `crates/engine/nn` library (its JS port). The
cart starts wobbling; after roughly 100–300 episodes a typical run
reaches the "solved" line at 195 average return.

What you're watching:

- **Left.** A direct replay of the most recent episode. The cart
  slides; the orange pole tips around its hinge. Episodes early in
  training fail quickly (the pole tips over); later episodes can
  last the full 200-step horizon.
- **Right.** Each grey dot is one episode's total reward. The
  green curve is the rolling mean over the last 10 — that's the
  smoothing you need to *see* learning through the variance that
  pure REINFORCE has even with a baseline.

Things to try:

- **Crank `lr` to 0.05.** Watch the return curve oscillate harder —
  a bigger step size lets each gradient *win* faster but also more
  often goes past the policy that worked.
- **Drop `γ` to 0.90.** Short-horizon credit assignment: the policy
  stops valuing late-trajectory upright-ness, the average return
  plateaus below the solved threshold even after many episodes.
- **Hidden = 16.** A smaller capacity network can still solve it,
  but the variance is higher and the run-to-run spread widens.

This is the canonical first L-effort widget for the new neural-net
library; design doc §15 ("First L-effort widget").

---

## 10.4 Baselines and the advantage function

The killer trick: **subtract a state-dependent baseline from $G_t$.**

> **Baseline lemma.** For any function $b(s)$ that depends only on $s$
> (not $a$),
>
> $$
> \mathbb{E}\_{a \sim \pi}\big[\nabla\_\theta \log \pi(a \mid s) \cdot b(s)\big] = 0.
> $$
>
> So replacing $G_t$ with $G_t - b(s_t)$ in the policy gradient does
> **not** bias the estimator.

*Proof.* Pull $b(s)$ out of the inner expectation and use
$\sum_a \pi(a \mid s) \nabla \log \pi(a \mid s) = \nabla \sum_a \pi(a \mid s) = \nabla 1 = 0$.
The score function has expectation zero under its own distribution. $\blacksquare$

This means we can subtract **anything** that depends only on the state
without changing the gradient's mean — but it changes its *variance*.
The variance-minimizing choice of $b$ for the policy gradient is

$$b^{\star}(s) \approx V^{\pi}(s),$$

the state-value function under $\pi$.

> **Advantage policy gradient.**
>
> $$
> \nabla\_\theta J(\theta) = \mathbb{E}\_{\tau \sim \pi\_\theta}\left[\sum\_t \nabla\_\theta \log \pi\_\theta(a\_t \mid s\_t) \cdot A^{\pi}(s\_t, a\_t)\right]
> $$
>
> where $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$ is the **advantage
> function** — how much better $a$ is than the policy's average action
> at $s$.

### Why advantage is the right signal

$Q(s, a)$ tells you the value of an action at a state. $V(s)$ tells you
the average value at that state, regardless of action. The *difference*
$A(s, a)$ isolates the action-specific signal. Whatever $V(s)$ contributes
is policy-baseline; only $A$ should drive the policy update.

This is also why **Dueling DQN** (Chapter 11 §9.4) decomposes
$Q = V + (A - \bar{A})$ — same insight, used architecturally. And why
the **Q-bias bootstrap pathology** (Chapter 17) is fundamentally an
advantage problem: the Simulator's $w_\text{alive} = 1.0$ inflates $V$
across all states uniformly; *advantages* (which the policy gradient
needs) would not care, but the Simulator's value-based policy does. An
advantage-learning Simulator would not have the bug. Chapter 13 builds
this out as actor-critic.

### How to estimate $V^{\pi}$

You usually do not have $V^{\pi}$ analytically. Three options:

1. **Use $G_t$'s running mean as a baseline.** Crude but effective.
2. **Train a separate value network $V_\phi(s)$** by TD or Monte Carlo
   regression. This *is* actor-critic (Chapter 13).
3. **Use a learned $Q_\phi(s, a)$ and a learned $V_\phi(s)$** for $A$
   exactly. Generalized Advantage Estimation (§10.6) is a smoother
   middle ground.

---

## 10.5 Parameterizing stochastic policies

The choice of $\pi_\theta(a \mid s)$ depends on the action space.

### Discrete: softmax over logits

For $|\mathcal{A}|$ discrete actions, parameterize logits
$z_a(s; \theta) \in \mathbb{R}$ and apply the softmax:

$$\pi_\theta(a \mid s) = \frac{\exp(z_a(s; \theta))}{\sum_{a'} \exp(z_{a'}(s; \theta))}.$$

The score function has a clean closed form:

$$\nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta z_a(s; \theta) - \sum_{a'} \pi_\theta(a' \mid s) \nabla_\theta z_{a'}(s; \theta).$$

The second term is the policy-weighted average of all logit gradients.

For **linear policy** $z_a(s) = \theta_a^{\top} \phi(s)$, this becomes

$$\nabla_{\theta_a} \log \pi_\theta(a \mid s) = (1 - \pi(a \mid s)) \phi(s), \quad \nabla_{\theta_{a'}} \log \pi_\theta(a \mid s) = -\pi(a' \mid s) \phi(s).$$

Read: increase $\theta_a$ for the action that was taken (weighted by
$1 - \pi(a \mid s)$, so we push harder on already-unlikely-correct
actions), decrease $\theta_{a'}$ for the others.

### Continuous: Gaussian policies

For $a \in \mathbb{R}^d$, parameterize the mean and (log) standard
deviation:

$$\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s)).$$

For simplicity, assume diagonal $\Sigma$ with $\log \sigma_\theta(s)$
learned. Sampling: $a = \mu_\theta(s) + \sigma_\theta(s) \odot \xi$ with
$\xi \sim \mathcal{N}(0, I)$.

Score function:

$$\nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta\left[-\tfrac{1}{2} (a - \mu_\theta)^{\top} \Sigma_\theta^{-1} (a - \mu_\theta) - \tfrac{1}{2} \log |\Sigma_\theta| + \text{const}\right].$$

With diagonal $\Sigma$ this reduces to a scaled gradient of $\mu$ and
$\log \sigma$.

The **reparameterization trick** (Kingma & Welling 2013, VAE; popularized
in RL by SAC, Chapter 13) reformulates this as
$a = \mu_\theta(s) + \sigma_\theta(s) \odot \xi$ with $\xi$ external
noise, allowing low-variance pathwise gradients through the network.
This is one of the two big PG variance-reduction tricks (the other is
advantages); together they make modern continuous-control RL feasible.

<div id="ch10-score-vs-reparam-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/score_vs_reparam/widget.js"></script>
Per-sample gradient under two estimators on the same Gaussian policy and quadratic reward. As you slide σ → 0 the score-function cloud blows up — the (a − μ)/σ² factor diverges before r(a) shrinks — while the reparameterised cloud collapses to a point. The variance-ratio readout climbs into the thousands at small σ. This is exercise 6/7: low-noise policies are where reparam decisively wins.

### Other policy families

- **Categorical with Dirichlet prior** for mixture policies.
- **Beta** for bounded continuous actions.
- **Mixture of Gaussians** for multimodal continuous policies.
- **Implicit (energy-based)** policies, where $\pi(a \mid s) \propto \exp(Q(s, a) / T)$.
  Connects to soft Q-learning / SAC.

### Entropy regularization

Modern PG implementations (A2C, PPO, SAC) typically add an entropy
bonus $\beta \cdot H(\pi_\theta(\cdot \mid s))$ to the policy-gradient
objective. This keeps the policy from collapsing prematurely to a
one-hot before $V$ has stabilized. At the gradient-flow fixed point of
the regularized objective on a bandit with rewards $r_a$, the policy
takes the closed form $\pi(a) \propto \exp(r_a / \beta)$ — Boltzmann
exploration with temperature $\beta$. SAC takes this further (Chapter
11): it learns $\beta$ as a Lagrange multiplier on a target entropy.

<div id="ch10-entropy-reg-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/entropy_reg/widget.js"></script>
Left panel is the closed-form Boltzmann policy π(a) ∝ exp(r_a / β). Slide β toward 0 and the bars collapse onto argmax r (exploitation, zero entropy); slide β up and they flatten toward uniform (exploration, max entropy). The right panel traces H(π) and E_π[r] across β — the Pareto frontier of the exploration/exploitation tradeoff, drawn for you with a moving rule at your current β.

---

## 10.6 Generalized Advantage Estimation (GAE)

We have two extreme advantage estimators:

- **Monte Carlo advantage**: $A_t = G_t - V(s_t)$. Unbiased; high variance
  (full episode noise).
- **TD advantage** ($n = 1$): $A_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) = \delta_t$.
  Low variance; biased (uses approximate $V$).

[Schulman et al. 2016] introduced **GAE($\lambda$)** as the $\lambda$-
weighted exponential combination, analogous to TD($\lambda$) in
Chapter 9:

$$A^{\text{GAE}(\lambda)}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}, \quad \delta\_t = r\_{t+1} + \gamma V(s\_{t+1}) - V(s\_t).$$

- $\lambda = 0$: $A_t = \delta_t$ — pure TD (low variance, biased).
- $\lambda = 1$: $A_t = G_t - V(s_t)$ — pure Monte Carlo (unbiased,
  high variance).
- $\lambda \in (0, 1)$: convex combination, tunable.

Common choice: $\lambda \in [0.92, 0.97]$ for continuous control
(PPO defaults to $\lambda = 0.95$). The recommendation: start with
$\lambda = 0.95$ and only tune if variance dominates or bias dominates.

### Mechanical implementation

Compute backwards through a trajectory of length $T$:
```python
A = np.zeros(T)
A[T-1] = delta[T-1]
for t in range(T-2, -1, -1):
    A[t] = delta[t] + gamma * lam * A[t+1]
```
One pass, $O(T)$ work. Returns the GAE values for every timestep.

### Why GAE matters beyond REINFORCE

GAE is the standard advantage estimator in modern actor-critic methods
(A2C, PPO, TRPO). Its role is to feed the policy gradient with a
low-variance, low-bias advantage signal. Chapter 13 builds A2C around
GAE.

<div id="ch10-gae-lambda-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/gae_lambda/widget.js"></script>
Each dot is one λ on the planted-bias toy MDP: x-axis bias², y-axis estimator variance. λ = 0 (pure TD) sits low-variance/high-bias; λ = 1 (pure MC) sits zero-bias/high-variance; the Pareto elbow around λ ≈ 0.95 is what real implementations pick. Slide the planted V-bias b and watch the whole curve translate — the elbow moves rightward as the critic gets worse, which is why GAE adaptively trusts MC more when V̂ is bad.

---

## 10.7 Policy gradient vs. Q-learning — a comparison

| Property | Q-learning (value-based) | Policy gradient |
|---|---|---|
| What it learns | $Q(s, a; \theta)$ | $\pi(a \mid s; \theta)$ |
| Action selection | $\arg\max_a Q$ | Sample from $\pi$ |
| Continuous actions | Hard (nested optimization) | Native |
| Stochastic-optimal policies | Cannot represent | Native |
| Convergence (tabular) | Provable [Watkins & Dayan 1992] | Provable for entropy-regularized PG |
| Convergence (FA) | On-policy linear: provable [TVR97] | Local optima only |
| Variance | Low (single value per state) | High — $\sigma^2$ scales with episode length |
| Off-policy | Native via Q-learning | Hard (importance sampling) |
| Exploration | $\epsilon$-greedy or built-in | Built-in (stochastic policy) |
| Sample efficiency | Higher (especially with replay) | Lower (needs trajectory variance) |
| Gradient estimator | TD error × $\phi(s, a)$ | $\nabla \log \pi(a \mid s) \times A(s, a)$ |

The simple rule of thumb:

- **Use Q-learning** when actions are discrete and small in number, when
  off-policy data is plentiful (e.g., replay-rich), when you can verify
  on-policy linear convergence applies.
- **Use policy gradient** when actions are continuous, when the optimal
  policy is stochastic, when episode-level rewards are dense enough to
  reduce variance, or when you need a smoothly differentiable policy
  (e.g., for downstream control via gradients).
- **Use actor-critic (Chapter 13)** when neither extreme fits — almost
  always, in modern RL.

---

## 10.8 Project tie-in

### The Simulator currently uses **no policy gradient**

The flat policy in
[`policy.rs`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs) is
$\epsilon$-greedy argmax over a hand-scored composite:

$$\text{score}(a) = \text{score\_action}(a) + q_\text{bias} \cdot Q(s, a) + \text{recipe\_bonus},$$

with $q_\text{bias} = 0.5$. Action selection is deterministic
(splitmix64-hashed argmax — see `policy.rs:221-227`) with $\epsilon = 0.10$
random override.

This is value-based: the score is a value, and selection is argmax.
No probability distribution over actions; no $\nabla \log \pi$ machinery.

### Why no policy gradient?

| Reason | Detail |
|---|---|
| **Discrete action space** | 18 effective action keys (16 templates with Step/Strike params). Argmax is cheap. |
| **Determinism canary** | Bit-identical reproducibility. PG sampling needs deterministic-RNG seeding — doable, but adds machinery. |
| **Variance budget** | PG's high variance demands many samples per gradient. Multi-agent simulations with hundreds of agents already strain compute. |
| **Per-agent diversity** | Different agents would need different $\theta_\text{policy}$ — possible but doubles the per-agent learned state. |
| **Convergence theory** | Linear TD on-policy has [TVR97]; PG has only local-optima guarantees. |

### What we could port

The **advantage decomposition** is the most valuable PG idea for the
Simulator, *even staying value-based*. The Q-bias bootstrap pathology
is fundamentally an advantage problem:

- Train $V_\theta(s)$ separately from $Q_\theta(s, a)$.
- Use $A(s, a) = Q(s, a) - V(s)$ in `policy.rs:474` instead of $Q$.

The alive-baseline gets absorbed into $V$, and $A$ has zero mean across
actions by construction — exactly the architectural fix that Dueling DQN
applies in deep settings. Chapter 13 makes this concrete.

### Sketching a softmax policy over current candidates

The least disruptive PG would be a softmax over the *existing*
candidate-scoring pipeline:

$$\pi_\theta(a \mid s) = \frac{\exp(\beta \cdot \text{score}(s, a))}{\sum_{a'} \exp(\beta \cdot \text{score}(s, a'))}$$

with a learnable temperature $\beta$. Each tick:

1. Compute `score(s, a)` for every candidate (already done).
2. Sample from the softmax instead of $\epsilon$-greedy argmax.
3. Apply the policy gradient (`gradient = (1 - π(a|s)) * φ(s)` for
   the chosen action; `gradient -= π(a'|s) * φ(s)` for the others) to
   the tile-coded score weights.

This would be a **dramatic** change to the learner — it commits to
training the *scores* (q_bias weight, recipe bonus) directly toward
expected return. Today those scores are mostly hand-tuned constants.
Worth thinking about as a long-horizon refactor; not on any near-term
roadmap.

### What test would exercise it

If implemented, the natural test would be on
[`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
sated arm — a softmax policy would sample Consume (with non-zero
probability even if Q is low) and break the lock-in.

---

## 10.9 Exercises

1. **(Log-derivative trick.)** Verify the identity
   $\nabla\_\theta \mathbb{E}\_{x \sim p\_\theta}[f(x)] = \mathbb{E}\_{x \sim p\_\theta}[f(x) \nabla\_\theta \log p\_\theta(x)]$
   for the case $p_\theta(x) = \mathcal{N}(\theta, 1)$ and $f(x) = x$.

2. **(REINFORCE on CartPole.)** Implement REINFORCE with a softmax policy
   over a 2-action discrete space. Use a small MLP for the logits. Plot
   episode-reward vs. episode number across 5 seeds. Confirm the
   estimator is very noisy.

3. **(Baseline variance reduction.)** Add a constant baseline equal to
   the running mean return so far. Re-run CartPole. How much does the
   variance drop?

4. **(Optimal baseline.)** Train a separate $V_\phi(s)$ network by MSE
   regression on the empirical returns, and use it as the baseline. How
   does sample efficiency compare?

5. **(GAE sweep.)** Implement GAE($\lambda$). On a CartPole-style task,
   sweep $\lambda \in \{0, 0.5, 0.9, 0.95, 0.99, 1.0\}$. Plot
   return-vs-episode for each. Which $\lambda$ converges fastest? Does
   it match PPO's default of 0.95?

6. **(Score-function variance.)** Compute the variance of
   $\nabla \log \pi(a \mid s)$ for a Gaussian policy as $\sigma_\theta \to 0$.
   What does this say about near-deterministic policies in PG?

7. **(Reparameterization vs. score-function.)** For a Gaussian policy,
   compare the variance of the score-function gradient
   $\nabla \log \pi(a) (G - V)$ vs. the reparameterized pathwise
   gradient $\nabla_\theta f(a_\theta)$. Run on a simple bandit task
   with $f$ a smooth function of $a$.

8. **(Softmax over Simulator candidates — paper sketch.)** Outline
   the changes to `policy.rs` required to replace argmax-on-score
   with softmax(score). Identify which constants would become
   learnable, which stay fixed, and which validation tests would
   break first.

---

## 10.10 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Konda & Tsitsiklis 1999] — Actor-Critic convergence (§10.7).
- [Schulman et al. 2016] — GAE (§10.6).
- [Sutton et al. 2000] — Policy Gradient Theorem w/ FA (§10.2).
- [Sutton & Barto 2018, Ch. 13] — textbook policy gradient (§10.1-10.5).
- [Watkins & Dayan 1992] — Q-learning convergence (§10.7).
- [Williams 1992] — REINFORCE (§10.3).
- [Tsitsiklis & Van Roy 1997] — linear TD convergence (§10.7).

## 10.11 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton & Barto 2018] | Ch. 13 | Textbook policy gradient presentation |
| [Spinning Up](https://spinningup.openai.com/) | "Intro to Policy Optimization" | Clean reference implementation of VPG (REINFORCE + baseline) |
| Schulman 2016 PhD thesis | All | Definitive treatment of GAE, TRPO, related |
| [Williams 1992] | All | The foundational paper; readable and short |

---

**Next:** [Chapter 13 — Actor-Critic Methods](13_actor_critic.md) — when
the value baseline is itself a learned network.

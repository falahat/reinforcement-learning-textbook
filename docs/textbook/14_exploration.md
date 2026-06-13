# Chapter 14 — Exploration

> **Prerequisites:** Chapter [2](04_the_rl_problem.md) (the exploration-
> exploitation tension), Chapter [6](08_temporal_difference_learning.md)
> ($\epsilon$-greedy in Q-learning), Chapter [1-3](01_linear_algebra.md)
> §1.1 (Hoeffding inequality).

> **Citations:** [Auer, Cesa-Bianchi & Fischer 2002] (UCB1);
> [Thompson 1933] (Thompson sampling); [Russo et al. 2018] (TS tutorial);
> [Jaksch, Ortner & Auer 2010] (UCRL2); [Kakade 2003] (PAC-MDP);
> [Bellemare et al. 2016] (pseudo-counts); [Pathak et al. 2017] (ICM);
> [Burda et al. 2019] (RND); [Osband et al. 2016] (Bootstrapped DQN);
> [Lattimore & Szepesvári 2020] (textbook).

> **Learning objectives:**
> 1. Bound the regret of $\epsilon$-greedy and UCB1; understand the
>    log-vs-linear gap.
> 2. Derive UCB from Hoeffding (the bound is *literally* from §1.1).
> 3. Compare $\epsilon$-greedy, UCB, Thompson sampling.
> 4. Recognize pseudo-counts, ICM, RND as solutions to sparse-reward
>    exploration in high-dim.
> 5. Explain exactly *why* the Simulator's $\epsilon = 0.10$ does not
>    rescue the Q-bias bootstrap lock-in.

## Why this chapter exists

Every previous chapter assumed exploration was either "given" (data
collected for us) or a one-line $\epsilon$-greedy. That works on simple
problems. It fails — sometimes catastrophically — on:

- **Sparse rewards**: Montezuma's Revenge gives reward only after
  hundreds of correct steps; random exploration reaches the goal with
  probability $\sim 1 / (\text{branching factor})^{\text{depth}}$. For
  depth 50, branching 4: $4^{-50} \approx 10^{-30}$.
- **Lock-in pathologies**: the Q-bias bootstrap pathology (Chapter 17)
  is partly an exploration failure — once Plant is locked in, no
  exploration scheme reaches Consume.
- **Bandit-like problems**: even without temporal structure, picking
  the right arm out of many requires careful tracking of confidence,
  not random tries.

This chapter is the toolkit: bandit-foundational exploration (UCB,
Thompson), state-space exploration in MDPs (UCRL2, count-based),
intrinsic-motivation exploration in high-dim (pseudo-counts, ICM, RND).
It closes by diagnosing what the Simulator does, what fails, and what
the cheapest fix looks like.

## Table of contents

- [12.1 The bandit reduction](#121-the-bandit-reduction)
- [12.2 ε-greedy and its limits](#122-greedy-and-its-limits)
- [12.3 UCB and optimism](#123-ucb-and-optimism-in-the-face-of-uncertainty)
- [12.4 Thompson sampling](#124-thompson-sampling)
- [12.5 Exploration in MDPs: UCRL2 and PAC-MDP](#125-exploration-in-mdps-ucrl2-and-pac-mdp)
- [12.6 Pseudo-counts in high dimensions](#126-pseudo-counts-in-high-dimensions)
- [12.7 Curiosity: ICM and RND](#127-curiosity-icm-and-rnd)
- [12.8 Choosing a scheme — a practical comparison](#128-choosing-a-scheme--a-practical-comparison)
- [12.9 Project tie-in](#129-project-tie-in-why-greedy-fails-here-and-what-would-work)
- [12.10 Exercises](#1210-exercises)
- [12.11 References](#1211-references-cited-in-this-chapter)
- [12.12 Further reading](#1212-further-reading)

---

## 12.1 The bandit reduction

Strip RL down to its exploration core: no state, no dynamics. **A
$K$-armed bandit** has $K$ arms, each with unknown reward distribution
$\mathcal{D}_k$ with mean $\mu_k$. At each of $T$ rounds, the agent picks
an arm $A_t$, receives reward $r_t \sim \mathcal{D} _{A_t}$, and decides
the next arm.

The objective: maximize total reward. The standard performance metric is
**regret** — the cumulative gap from always playing the best arm:

$$\mathrm{Regret}(T) = T \mu^{\star} - \mathbb{E}\left[\sum_{t=1}^T r_t\right] = \sum_{k \neq k^{\star}} \Delta_k \cdot \mathbb{E}[N_k(T)],$$

where $\Delta_k = \mu^{\star} - \mu_k$ is the suboptimality gap and $N_k(T)$
is the number of times arm $k$ was pulled.

Good algorithms keep regret **logarithmic** in $T$: $\mathrm{Regret}(T) = O(\log T)$.
Bad ones are **linear**: $O(T)$. The difference is enormous.

### Why bandits matter for RL

Bandits are RL with $|\mathcal{S}| = 1$ and no transitions. Every
exploration insight in bandits ports to MDPs (with extra work for state
visit-counts and dynamics estimation). UCB, Thompson sampling, and
intrinsic motivation all originated in bandits.

---

## 12.2 ε-greedy and its limits

The simplest scheme: with probability $1 - \epsilon$ pick the empirical
best arm $\arg\max_k \hat\mu_k$; with probability $\epsilon$ pick
uniformly at random.

> **Regret of $\epsilon$-greedy on a bandit.**
>
> $$
> \mathrm{Regret}(T) = \Omega(\epsilon T).
> $$
>

Read it: regret grows *linearly* in $T$. You are exploring at rate
$\epsilon$ forever, even when the best arm is obvious.

**Decaying $\epsilon$.** With $\epsilon_t = c/t$, regret becomes $O(\log T)$
— same as UCB. But the decay schedule needs tuning, and the constant
$c$ depends on the (unknown) gaps $\Delta_k$. Mis-tuned decay either
explores too much (linear regret again) or commits prematurely (gets
stuck on a wrong arm).

### Why $\epsilon$-greedy gets used anyway

- **One hyperparameter.** Simple.
- **Deterministic-RNG-friendly.** With deterministic seeding (the
  Simulator's [splitmix64 in policy.rs:221-227](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs)),
  the $\epsilon$ draw is fully reproducible.
- **No state-tracking overhead.** UCB needs visit counts; Thompson needs
  posteriors. $\epsilon$-greedy needs only the argmax.
- **Robust in practice on dense-reward problems.** When the gap
  $\Delta_{\min}$ is large, $\epsilon$ doesn't matter much — you converge
  fast either way.

It fails on sparse rewards, lock-in pathologies, and high-confidence
estimation problems. Read on.

### Try it: how random exploration scales with chain length

A 1-D random walk on a chain of length $n$ with the goal at the far end:
no reward signal exists until the agent hits the goal cell, so an
$\epsilon$-greedy learner with uniform Q estimates explores by uniform
random walk. The hitting-time scales as $n^2$; the curves below show
$P(\text{reached goal by step } t)$ for three chain lengths. Double the
chain (orange to red) and the budget required quadruples — the same
combinatorial wall sparse-reward Atari games hit, just visible on a toy.

<div id="ch12-sparse-walk-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/sparse_reward_walk/widget.js"></script>
P(reached the goal by step t) on a 1-D chain of length N, sampled over 400 random-walk rollouts. Double N and the curve shifts rightward by a factor of four — hitting-time on a 1-D walk scales as N². The same quadratic blowup ports straight to the depth-d branching argument: random exploration is hopeless once the reward is more than a few steps away. The sparse-reward catastrophe is visible in one slider drag.

### Try it: 5-armed bandit, ε-greedy

A live ε-greedy bandit running this project's actual
[`playground::run_episode`](https://github.com/falahat/simulator/blob/main/crates/engine/playground/src/lib.rs)
loop compiled to WASM. The 5 arms have true means
`[0.10, 0.30, 0.50, 0.70, 0.90]`; the learner doesn't know which is
which. Slide ε down to see the curve climb closer to 1.0; slide it up
to see exploration cap the asymptote at $1 - \epsilon + \epsilon/K$.

<div id="ch12-bandit-widget"></div>
<script type="module">
  import init, { start } from './widgets/bandit/pkg/widget_bandit.js';
  await init();
  start('ch12-bandit-widget');
</script>

The green curve is the fraction of pulls landing on the **true** best
arm (arm 4) per 50-step block. The grey horizontal line is the
random-pick fraction $1/K = 0.2$ — what you'd see with $\epsilon = 1$
(pure exploration).

---

## 12.3 UCB and "optimism in the face of uncertainty"

### Why UCB is the canonical bandit algorithm

§12.2 showed $\epsilon$-greedy gets *linear* regret (the gap from
optimal grows like $T$). UCB gets *logarithmic* regret ($O(\log T)$).
On a $T = 10^6$-step problem the difference is six orders of
magnitude in total wasted reward — UCB is *qualitatively* better,
not incrementally.

What makes UCB work? **The Hoeffding inequality (Chapter 2 §2.4) is
literally inside the algorithm.** The "$\sqrt{2 \log t / N_k}$"
confidence bonus is the Hoeffding tail probability inverted. UCB is
what happens when you take a concentration inequality and use it
*at decision time* instead of just for proving convergence after
the fact.

That structural idea — use confidence intervals on Q-values to
drive action selection — generalises to MDPs (UCRL2, §12.5), to
pseudo-counts in high-dim (§12.6), and to the optimism-bonus
methods at the heart of most modern exploration research. Read this
section as the *prototype* for those generalisations, not as a
standalone bandit fact.

The slogan: **be optimistic about arms you have not pulled much**.
Concretely, pick the arm whose *upper confidence bound* is highest.

### Deriving UCB1 from Hoeffding

From §1.1: if $X_1, \ldots, X_n \in [0, 1]$ are i.i.d. with mean $\mu$,
Hoeffding gives

$$P\big(\bar{X}_n - \mu < -\sqrt{\log(2/\delta) / (2n)}\big) \leq \delta.$$

So with confidence $1 - \delta$, $\mu \leq \bar{X}_n + \sqrt{\log(2/\delta) / (2n)}$.

Setting $\delta = 1 / t^2$ (a per-round confidence) and accumulating
over $t = 1, \ldots, T$ rounds:

> **UCB1** ([Auer, Cesa-Bianchi & Fischer 2002]).
> At round $t$, pick
>
> $$
> A_t = \arg\max_k \left[\hat\mu_k(t) + \sqrt{\frac{2 \log t}{N_k(t)}}\right].
> $$
>

The square-root term is a **confidence bonus** that shrinks as
$N_k(t)$ grows. Arms pulled rarely have large bonuses (the agent is
uncertain about them); arms pulled often have small bonuses (the agent
knows them well). UCB plays the arm with the highest *possible* value
under the confidence bound — hence "optimism."

### Three readings of UCB

**1. As an empirical Bayes-like bound.** $\hat\mu_k$ is the
empirical mean — the maximum-likelihood estimate of $\mu_k$. The
confidence bonus is a frequentist upper bound on the true mean.
UCB acts as if every arm's *true* mean equals its upper confidence
bound. That's "optimism in the face of uncertainty" — assume the
best until evidence rules it out.

**2. As an exploration-exploitation decomposition.** The decision
rule $\arg\max_k [\hat\mu_k + \text{bonus}_k]$ splits into:
- $\hat\mu_k$ — exploit what you know.
- $\text{bonus}_k$ — explore what you don't.

A poorly-explored arm wins on the second term; a well-known
high-mean arm wins on the first. The argmax balances them
automatically; you don't tune $\epsilon$.

**3. As a regret-driven design.** The bonus is *chosen* so that the
total regret is $O(\log T)$, not as some independent intuition.
Working backwards: pick the schedule $\delta_t = 1/t^2$ in Hoeffding;
the sum $\sum_t \delta_t < \infty$ caps total mistakes; the
resulting algorithm has logarithmic regret. The "optimism" framing
is the *consequence* of the regret optimisation, not its
motivation.

### What UCB doesn't say

- **It assumes bounded rewards $X \in [0, 1]$.** Outside that range,
  the Hoeffding constant $(b - a)^2 / 2$ needs adjusting, but the
  scaling is the same.
- **It assumes the gaps $\Delta_k$ are positive.** With ties between
  the best arm and others, the regret bound becomes
  $\Delta_k$-distribution-dependent. Practically: UCB still works,
  but the $\log T$ constant gets worse.
- **It's not adaptive to easier problems.** If one arm is vastly
  better than the others, UCB still pays the same logarithmic
  exploration tax. Bayesian methods like Thompson sampling
  (§12.4) can adapt to easy problems by sharpening the posterior
  faster than UCB sharpens its frequentist interval.
- **It extends to MDPs only with extra work.** UCRL2 (§12.5)
  applies UCB to the *Bellman equation* rather than to single
  arms — much harder to get right.

### Try it: the Hoeffding → UCB substitution

The widget below makes the four-line derivation interactive: scrub $n$
and $\delta$ to read the Hoeffding bound directly; flip the $\delta_t$
schedule to see why $\delta_t = 1/t^2$ is the one that gives UCB1's
$\sqrt{2 \log t / N_k}$, while a constant $\delta_t$ degenerates to a
flat bonus (and therefore linear regret). Directly serves
[exercise 7](#1210-exercises).

<div id="ch12-hoeffding-ucb-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hoeffding_ucb/widget.js"></script>
Top: slide n and δ and watch the Hoeffding bonus √(log(2/δ)/2n) update in the live formula. Bottom: the per-round bonus as a function of t, plotted for three pull rates. Flip the δ-schedule: 1/t² gives the classic UCB1 shape √(2 log t / Nₖ), 1/t shrinks a bit faster, and constant δ degenerates to a flat non-vanishing bonus. That choice of schedule is *the* derivation step from Hoeffding to UCB1.

### Try it: per-step UCB1 decisions on a 5-arm bandit

Each arm is a thermometer: the dark inner bar is the empirical mean
$\hat\mu_k$; the translucent band on top is the confidence bonus
$c\sqrt{\log t / N_k}$. The argmax arm — outlined in blue — is the one
the policy will pull next. Click *step* repeatedly: the picked arm's
band shrinks (more pulls = sharper estimate), and the argmax migrates
between arms until UCB locks onto arm 3 (true $\mu^\star = 0.72$).

<div id="ch12-ucb-bars-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/ucb_bars_ch12/widget.js"></script>

> **Regret of UCB1.**
>
> $$
> \mathrm{Regret}(T) = O\left(\sum_{k \neq k^{\star}} \frac{\log T}{\Delta_k}\right).
> $$
>

Logarithmic. The "regret per arm" depends inversely on the gap — gaps
larger than the noise level get resolved fast.

### Try it: regret curves, side by side

UCB1 vs constant $\epsilon$-greedy vs decaying $\epsilon$ on the same
fixed bandit. The grey dashed reference is an $O(\log T)$ shape scaled
to the UCB1 endpoint — the UCB1 curve hugs it; the constant $\epsilon$
curve climbs linearly above it (the $\Omega(\epsilon T)$ from §12.2
made visible). Slide $T$ out to 5000 to watch the linear vs log gap
open up.

<div id="ch12-ucb-regret-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/ucb_regret/widget.js"></script>
Cumulative regret across T rounds on a fixed 5-arm Bernoulli bandit, with a log-T reference line for shape. UCB1 hugs the log curve (regret = O(log T)). Constant-ε climbs linearly forever — every wrong pull leaks Ω(ε·Δ). Decaying ε = 1/√t sits between them. Slide c up and UCB1's curve lifts but stays sublinear; slide ε up and the constant-ε line steepens. The asymptotic gap is the whole reason UCB matters.

### Try it: UCB1 across reward distributions

<div id="ch12-bandit-envs-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/bandit_envs/widget.js"></script>

Same 5 arms, same UCB1, three reward worlds. Bernoulli and Gaussian rewards both give the classic $O(\log T)$ regret curve — UCB1's concentration inequality holds. Switch to the adversarial environment, where the adversary punishes whichever arm we lean on, and the curve goes *linear*: Hoeffding's bound assumes i.i.d. rewards, and once an adversary can react to our action the assumption fails. The killer plot is the adversarial line — that's the visible price of violating UCB1's preconditions.

### Try it: the exploration / exploitation Pareto front

<div id="ch12-explore-exploit-pareto-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/explore_exploit_pareto/widget.js"></script>

Each dot is one algorithm's *run* on the same bandit: x = final cumulative regret, y = exploration entropy $H$ of its pull distribution. Random is far right (high entropy, terrible regret); greedy ε=0 is far left and bottom (low entropy, low regret only when it gets lucky). Thompson and the well-tuned UCB1 sit on the lower envelope — the orange dashed line is the Pareto front of algorithms that aren't dominated. Reseed and the front mostly stays put: this is the universal "you can't beat this curve with one knob" tradeoff.

### Why UCB is principled

Three nice properties:

1. **No tuning of $\epsilon$.** The exploration rate is set by the
   confidence formula.
2. **State-adaptive.** Arms with high uncertainty are explored more
   *automatically*.
3. **Provably optimal up to constants** [Lai & Robbins 1985 lower bound].

The catch: UCB1 is for stationary, bounded, scalar bandits. Real RL has
non-stationarity (the policy changes), multi-dimensional states, and
unbounded rewards. UCB-like methods exist for all these (UCRL2 for
MDPs; LinUCB for linear bandits; UCB-V for variance-aware). The slogan
generalizes; the formulas are problem-specific.

---

## 12.4 Thompson sampling

A Bayesian alternative. Maintain a posterior over each arm's reward
distribution; at each round, **sample** parameters from the posterior
and pull the arm whose sampled mean is highest.

For Bernoulli arms with Beta priors:

$$\mu_k \sim \text{Beta}(\alpha_k, \beta_k); \quad \text{Pull } \arg\max_k \mu_k; \quad \text{Update}.$$

Thompson sampling **predates UCB by 70 years** ([Thompson 1933]). It was
neglected because pre-Bayesian-RL theorists could not bound its regret;
modern analysis ([Russo et al. 2018]) shows it matches UCB's
$O(\log T)$ regret, often with better empirical constants.

### Why Thompson sampling often beats UCB in practice

- **Naturally Bayesian.** Posterior carries more information than just a
  confidence bound.
- **Smooth exploration.** UCB switches arms discretely; TS samples
  continuously.
- **Handles correlations.** With a joint posterior over arm rewards, TS
  can exploit dependencies (LinUCB-style) elegantly.

The catch: you need a posterior. For Bernoulli arms: Beta-Binomial,
trivial. For Gaussian arms with known variance: Normal-Normal,
straightforward. For neural-network value functions: **Bootstrapped DQN**
([Osband et al. 2016]) — train $K$ Q-networks on bootstrap-resampled
data; act according to a randomly selected one each episode. An
approximate Thompson sample.

### Try it: Beta posteriors sharpening over time

5-arm Bernoulli bandit, $\text{Beta}(1,1)$ priors. Each step samples
one $\mu_k$ from each arm's posterior (the thin ticks), pulls the arm
with the highest sample, and updates that arm's posterior. Run 100
steps and watch the five densities sharpen — by t = 500 the best arm's
posterior (right edge) and the worst arm's (left edge) barely overlap;
by t = 1000 the policy almost never picks the wrong arm.

<div id="ch12-thompson-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/thompson_beta/widget.js"></script>
Each curve is one arm's Beta posterior; the vertical ticks are the sample draws each round. Hit "+100 steps" and watch the densities sharpen around their true means while the worst arms' posteriors pull away from the best arm's. The pull-rate naturally concentrates on arms whose posteriors still overlap the top — no explicit bonus formula, just sampling. Thompson sampling's whole story is "draw, act, update," and the picture is exactly that.

### Try it: Bootstrapped DQN's K-head spread

K = 10 parallel Q-learners on the same 5-arm bandit, each accepting
each observation with probability $p_\text{mask}$ (bootstrap masking).
Each arm shows: K coloured ticks at the K head estimates; the green
dot at their mean; the red diamond at the true mean. The orange band
is the head min/max — the *implicit uncertainty* the algorithm
samples from. Early on the band is wide (ensemble disagrees); after
~100 pulls it tightens around the true mean (ensemble agrees). That
visible tightening is the approximate Thompson sample at work.

<div id="ch12-bootstrap-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/bootstrapped_dqn/widget.js"></script>
Five horizontal lanes, one per arm; the K coloured ticks in each lane are the bootstrap heads' empirical means, and the box is their average. Step the policy and watch the boxes tighten — the spread across heads *is* the implicit uncertainty estimate the ensemble votes with. Lower p_mask (each head sees less data) and the spreads stay wide longer; raise it and the heads agree quickly but explore less. This is Thompson sampling without a Bayesian posterior.

---

## 12.5 Exploration in MDPs: UCRL2 and PAC-MDP

Bandit insights generalize to MDPs.

### UCRL2 ([Jaksch, Ortner & Auer 2010])

Maintain confidence intervals on the *transition* and *reward* functions.
At each "phase," construct the **optimistic MDP**: the one within the
confidence set that yields the highest possible average reward. Plan
optimally in it. Execute. Update counts. Repeat.

> **Regret of UCRL2 on a finite MDP** (diameter $D$, $|\mathcal{S}|$ states, $|\mathcal{A}|$ actions):
>
> $$
> \mathrm{Regret}(T) = \tilde{O}(D \sqrt{|\mathcal{S}| |\mathcal{A}| T}).
> $$
>

Square-root in $T$, polynomial in MDP size, no explicit dependence on
gaps. UCRL2 is the canonical "rigorous MDP exploration" — rarely used
in practice (the optimistic-MDP construction is expensive) but
foundational.

### PAC-MDP framework ([Kakade 2003])

Different objective: instead of regret, count the **number of
suboptimal steps** before the agent learns an $\epsilon$-optimal policy
with confidence $1 - \delta$. This is the *PAC* sample complexity.

For tabular MDPs, R-max and MBIE-EB achieve PAC bounds polynomial in
$|\mathcal{S}|, |\mathcal{A}|, 1/\epsilon, 1/(1-\gamma)$.

Why PAC-MDP matters: it gives explicit "I have learned" guarantees,
where regret gives only "I am converging." Useful in safety-critical or
audit-required settings.

---

## 12.6 Pseudo-counts in high dimensions

Visit counts $N(s)$ work in tabular settings. For continuous or
high-dim observations, no two states are exactly equal, so true counts
are useless.

**Pseudo-counts** [Bellemare et al. 2016] generalize: define
$\tilde N(s) = \rho(s) / (1 - \rho(s)) \cdot n$, where $\rho(s)$ is the
density assigned by a learned density model trained on past observations,
and $n$ is the total step count. The pseudo-count is calibrated so that
$\tilde N(s)$ increments by 1 for each observation drawn from the model.

The exploration bonus becomes

$$\text{bonus}(s) = \beta / \sqrt{\tilde N(s) + 1}$$

— same form as UCB, but the "count" is a density-model surrogate.

[Bellemare et al. 2016] gave the **first agent to escape the first room
of Montezuma's Revenge** — a notoriously sparse-reward Atari game. The
density model was a CTS bit-plane model (lightweight); subsequent work
used VAEs and PixelCNNs.

### Why pseudo-counts feel like cheating but aren't

It looks like you are adding extra information from a *learned* model.
But the pseudo-count is provably equivalent to a tabular count in the
limit ([Bellemare et al. 2016, Theorem 1]), and the bonus formula
follows from the same Hoeffding logic as UCB1. The density model is
just an efficient way to count.

---

## 12.7 Curiosity: ICM and RND

A different angle: exploration as **prediction-error maximization**.

### ICM — Intrinsic Curiosity Module ([Pathak et al. 2017])

Train two networks:

1. **Forward model**: predict $\hat\phi(s_{t+1})$ from $\phi(s_t), a_t$,
   where $\phi$ is an internal feature representation.
2. **Inverse model**: predict $\hat a_t$ from $\phi(s_t), \phi(s_{t+1})$.

The inverse model trains $\phi$ to capture features *relevant to action
prediction* — filtering out noise that the agent cannot control.

**Intrinsic reward**:

$$r^{\text{int}}\_t = \|\hat\phi(s\_{t+1}) - \phi(s\_{t+1})\|^2$$

— prediction error of the forward model. The agent is rewarded for
ending up in states the model fails to predict.

The agent's total reward is $r^{\text{ext}}_t + \eta r^{\text{int}}_t$.

Why ICM is clever: the inverse-model regularization filters out
unpredictable-but-irrelevant noise (the agent cannot benefit from
exploring TV static). Without it, naive curiosity goes to the noisiest
parts of the observation space — the "noisy TV problem."

### RND — Random Network Distillation ([Burda et al. 2019])

Even simpler. Fix a random initialization of a network $f_\text{rnd}(s)$.
Train a second network $\hat f(s)$ to predict its output.
$r^{\text{int}}\_t = \|\hat f(s\_t) - f\_\text{rnd}(s\_t)\|^2$. The intrinsic
reward is high for novel observations (where $\hat f$ has not learned)
and low for familiar ones (where it has).

RND achieves SOTA on Montezuma's Revenge. The frozen random target
sidesteps the noisy-TV problem because random networks are sensitive to
specific input patterns, not to noise level.

### Why curiosity matters even with extrinsic reward

Pure extrinsic reward fails when reward is sparse. Curiosity provides a
**dense bootstrap**: the agent has *something* to learn from on every
step, building useful features and behaviors. By the time the
extrinsic reward arrives, the agent has already explored most of the
space.

---

## 12.8 Choosing a scheme — a practical comparison

| Method | Best for | Cost | Gotcha |
|---|---|---|---|
| $\epsilon$-greedy | Dense reward, small action space | Trivial | Linear regret without decay; never escapes lock-in |
| Decaying $\epsilon$ | Same, but with bound knowledge | One schedule | Tuning $\epsilon$ schedule |
| UCB1 | Stationary bandit, scalar | $O(K)$ memory | Stationary assumption |
| Thompson sampling | Bayesian-friendly, structured noise | Posterior maintenance | Need a posterior |
| LinUCB / Linear-TS | Linear bandit with contexts | $O(d^2)$ matrices | Linear-only |
| UCRL2 | Tabular MDP, rigorous bounds | Costly optimism step | Tabular only |
| Bootstrapped DQN | Deep RL, approximate TS | $K \cdot$ network size | $K \sim 10$ |
| Pseudo-counts | High-dim, sparse reward | Density model | Density-model quality |
| ICM | High-dim sparse reward + noisy obs | Two extra networks | Noisy-TV ($\phi$-dependent) |
| RND | Same, simpler | One extra network | Less principled |

When in doubt:

- **Bandits**: UCB1 or Thompson.
- **Tabular MDP**: UCRL2 or count-based bonus.
- **Deep RL, dense reward**: $\epsilon$-greedy with decay, or NoisyNets
  (§9.8).
- **Deep RL, sparse reward**: RND. (Then think about ICM if RND wastes
  effort on noise.)
- **Deep RL, where curiosity might over-shoot**: pseudo-counts.

---

## 12.9 Project tie-in — why $\epsilon$-greedy fails here, and what would work

### The Simulator's exploration

[`policy.rs:493-501`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs)
implements $\epsilon$-greedy at $\epsilon = 0.10$ (configurable via
[`learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs)
field `tile_coding.epsilon`). With ~10 candidate actions per
cognition step, each non-argmax action has $\sim 0.10 / 9 \approx 1.1\%$
probability per cognition step. At 1 cognition step per 10 ticks, that's
$\sim 0.001$ per tick.

The draw is a deterministic [`splitmix64`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs)
hash of `(agent.bits, tick, salt)` — preserving the determinism canary.
This is bit-identical reproducible across runs, an important property
the project requires.

### Why it does not rescue the Q-bias bootstrap lock-in

The lock-in story from Chapter 1 §1.6 / Chapter 17:

1. Tick 1: agent randomly tries Plant (one of ~10 candidates). $Q_\text{Plant}$
   starts at 0.
2. Plant returns reward $w_\text{alive} = 1.0$ (and small drive-cost).
   $Q_\text{Plant}$ ticks up by $\alpha \cdot 1 = 0.10$.
3. Ticks 2-N: Plant has the highest $Q$ now (other actions still at 0).
   Argmax picks Plant. $Q_\text{Plant}$ climbs toward 10.
4. Meanwhile $\epsilon = 0.10$ occasionally picks another action — but
   that action returns its own $w_\text{alive}$ reward, climbing
   *its* Q. Slower (rare events), but not breaking the lock-in.

The pathology is symmetric: every committed action's Q climbs to ~10.
$\epsilon$-greedy *does* explore, but the Q estimates it produces for
explored actions are not informative *because every action's Q
converges to the same value*. There is nothing for argmax to pick.

This is **not solved** by a higher $\epsilon$. It is also not solved by
UCB-style optimism on its own (the climbing Q-values would still
dominate the bonus). It *is* solved by the advantage decomposition
(Chapter 13 §11.7) — and **also** solvable by Fix 1 of the bug
report (set $w_\text{alive} = 0$, so committed actions only earn
relief-of-drive reward).

### What an exploration-side fix would look like

If we wanted to keep $w_\text{alive} = 1.0$ and the value-based decision,
exploration could attempt the fix:

#### UCB-style bonus over (tile-index, action-key)

For each tile slot $h$, maintain a count $N(h, a)$ of (slot, action)
co-occurrences. The exploration bonus on action $a$ in state $s$ is

$$\text{bonus}(s, a) = c \sqrt{\frac{\log t}{\min_{h \in \text{tiles}(s)} N(h, a)}}.$$

Add this to the score. Tile-coded actions never tried have huge
bonuses; those tried often have tiny ones.

The bonus is bounded by tile structure, not by the (intractable) state
count. The Simulator's tile coder already produces sparse $h$'s; adding
the bonus adds $O(\text{tilings})$ work per evaluation.

#### Pseudo-counts over the observation

Train a small density model on observations (an autoencoder bottleneck
suffices); use $\sqrt{1/(\tilde N(s) + 1)}$ as the bonus. Less
discrimination than the per-action UCB but cheaper.

#### Both — and add to score, not Q

Critically, the bonus belongs in the **score**, not the Q-update. The
Q-update should chase true returns; the bonus changes only which action
is selected. Concretely:
```rust
let total = drive_score + q_bias * q_score + recipe_bonus + explore_bonus;
// Argmax over total; TD update is on (s, a, r, s', a') unchanged.
```

#### How any of these unlock the L-suite

[`long_horizon_harvest.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/curricula/long_horizon_harvest.rs)
needs Consume to actually be tried even when bootstrap pushes Plant to
high Q. UCB-style bonus would assign a high bonus to Consume (low count
relative to Plant) and force exploration. After enough Consume tries,
its Q lands at $\sim 10 + r_\text{drive\_relief}$ — *higher* than
Plant's — and the lock-in flips.

This is exactly the experiment the project's Q-bias bootstrap pathology
analysis should run to validate any exploration-side fix.

### What about Thompson sampling here?

Per-tile-slot beta posteriors over Q would be a stretch but conceivable.
The cost: $O(M \cdot |\mathcal{A}_\text{keys}|)$ posterior parameters,
where $M = 65{,}536$ is the IHT size. That's ~$10^6$ extra
floats per agent. Probably too expensive at scale.

A cheaper variant: Bootstrapped Q-learners — maintain $K$ independent
$Q$-tile-coders, sampling one per cognition step for argmax. $K = 5$
seems empirically OK on Atari; would cost ~5× per-agent Q storage.

---

## 12.10 Exercises

1. **(UCB1 on 10-arm Gaussian bandit.)** Set up 10 arms with means in
   $[0, 1]$ and unit variance. Implement UCB1 and $\epsilon$-greedy
   with $\epsilon \in \{0.01, 0.05, 0.1\}$. Plot regret-vs-rounds. At
   what $T$ does UCB1 cross the best $\epsilon$-greedy variant?

2. **(Thompson sampling on Bernoulli bandit.)** 5 arms with success
   probabilities $\{0.1, 0.3, 0.5, 0.7, 0.9\}$. Implement Beta-Binomial
   Thompson sampling. Compare regret to UCB1.

3. **(Sparse-reward gridworld.)** Build a 10×10 gridworld with reward
   only at the goal. Implement $\epsilon$-greedy with $\epsilon = 0.1$.
   How many steps to first reward, averaged across 50 seeds?

4. **(Same with RND.)** Add a random network and a learner predicting
   it; use the prediction error as intrinsic reward. Re-run. How much
   faster is first-reward?

5. **(UCB MDP — UCRL2 mini.)** On a 4-state MDP with known transitions
   and unknown rewards, implement the UCRL2-style optimistic planning.
   Plot regret vs. round.

6. **(Plant-lock-in unlock — project sketch.)** Pseudo-code an
   exploration bonus added to
   [`policy.rs:474`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs):
   `total = drive_score + q_bias * q + recipe_bonus + explore_bonus`,
   where `explore_bonus = c * sqrt(log t / min_tile_action_count)`.
   Detail: where the count map lives, when it's updated, what `c` you
   would default to.

7. **(Hoeffding & UCB derivation.)** Walk through the derivation of
   UCB1's bonus from Hoeffding (§12.3). Identify exactly where the
   choice of $\delta = 1/t^2$ enters and what would change if you used
   $\delta = 1/t$ or $\delta = c$ constant.

8. **(Noisy TV.)** Construct a 2-room gridworld where one room contains
   a TV that produces unpredictable random observations. Implement
   curiosity-driven exploration (ICM and pure forward-model) without
   the inverse-model regularization. Confirm naive curiosity gets
   stuck on the TV.

---

## 12.11 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Auer, Cesa-Bianchi & Fischer 2002] — UCB1 (§12.3).
- [Bellemare et al. 2016] — pseudo-counts (§12.6).
- [Burda et al. 2019] — RND (§12.7).
- [Jaksch, Ortner & Auer 2010] — UCRL2 (§12.5).
- [Kakade 2003] — PAC-MDP (§12.5).
- [Lai & Robbins 1985] — bandit lower bound (§12.3).
- [Lattimore & Szepesvári 2020] — *Bandit Algorithms* textbook.
- [Osband et al. 2016] — Bootstrapped DQN (§12.4).
- [Pathak et al. 2017] — ICM (§12.7).
- [Russo et al. 2018] — Thompson sampling tutorial (§12.4).
- [Thompson 1933] — original Thompson sampling (§12.4).

## 12.12 Further reading

| Source | What to read | Why |
|---|---|---|
| [Lattimore & Szepesvári 2020] | Bandit textbook (free online) | The reference for bandit-style exploration |
| [Russo et al. 2018] | TS tutorial | Modern Thompson sampling treatment |
| [Sutton & Barto 2018] | §2.7 (bandit gradient), §10.1 (UCB) | Textbook coverage |
| [Bellemare et al. 2016] | All | The first deep-RL pseudo-count breakthrough |

---

**Next:** [Chapter 15 — Model-Based RL](15_model_based_rl.md) — when
*planning* in a learned model rescues sample efficiency.

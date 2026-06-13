# Chapter 19 — Long-Horizon Credit Assignment

> **Prerequisites:** Chapters [6](08_temporal_difference_learning.md)
> (TD), [7](09_eligibility_traces.md) (TD($\lambda$)),
> [14](16_hierarchical_rl.md) (HRL), [15](17_fa_pathologies.md) (FA
> pathologies).

> **Citations:** [van Seijen et al. 2016] (true online TD($\lambda$));
> [Andrychowicz et al. 2017] (HER); [Ng, Harada & Russell 1999] (PBRS);
> [Munos et al. 2016] (Retrace($\lambda$)); [Dayan 1993] (successor
> representation); [Barreto et al. 2017] (successor features + GPI);
> [Arjona-Medina et al. 2019] (RUDDER); [Harutyunyan et al. 2019]
> (Hindsight Credit Assignment); [Foerster et al. 2018] (COMA);
> [Nachum et al. 2018] (HIRO); [Levy et al. 2019] (HAC).

> **Learning objectives:**
> 1. Quantify why $\gamma^{500} \approx 0$ defeats TD on the L-suite.
> 2. Apply potential-based reward shaping to add domain knowledge
>    without changing the optimal policy.
> 3. Implement Hindsight Experience Replay for goal-conditioned RL.
> 4. Build successor features and use GPI for transfer.
> 5. Sketch RUDDER's return-decomposition LSTM as the most-promising
>    single fix for the L-suite's delayed-reward problem.
> 6. Match a tool to a problem: pick the right method from the toolbox.

## Why this chapter exists

Some tasks have reward thousands of ticks after the action that caused
them. Examples:

- **Minecraft diamond.** A multi-stage chain (tree → wood → pickaxe →
  stone → better pickaxe → iron → diamond) that takes ~10 000 steps
  end-to-end.
- **The Simulator's L-suite.** Plant a crop, wait ~500 ticks, harvest
  for a reward. The Plant action's signal is the reward at $t + 500$.
- **Locking the door.** You lock a door at $t$; the *absence* of an
  intruder at $t + 10{,}000$ is the reward.

The math is unforgiving:

$$\gamma^k \text{ at } \gamma = 0.9: \quad k=100 \to 3 \cdot 10^{-5}, \quad k = 500 \to 1.4 \cdot 10^{-23}.$$

For $k = 500$, no realistic learning rate can extract signal — the
TD update size for Plant from the eventual Harvest reward is
$\alpha \cdot \gamma^{500} \cdot \text{Harvest} \approx 10^{-24}$.

Chapter 16 (HRL) gave one answer: change the timescale so $k$ is
smaller. This chapter gives the *complementary* answers — keep the
primitive-action timescale but propagate credit *better*.

## Table of contents

- [17.1 The $\gamma$-decay catastrophe](#171-the-gamma-decay-catastrophe)
- [17.2 Eligibility traces, recap](#172-eligibility-traces-recap)
- [17.3 Potential-based reward shaping](#173-potential-based-reward-shaping)
- [17.4 Hindsight Experience Replay](#174-hindsight-experience-replay)
- [17.5 Successor features and GPI](#175-successor-features-and-gpi)
- [17.6 RUDDER — return decomposition](#176-rudder--return-decomposition)
- [17.7 Hindsight credit assignment](#177-hindsight-credit-assignment)
- [17.8 COMA — counterfactual credit in multi-agent](#178-coma--counterfactual-credit-in-multi-agent)
- [17.9 Hierarchy revisited](#179-hierarchy-revisited)
- [17.10 Project tie-in: which method fits the L-suite?](#1710-project-tie-in-which-method-fits-the-l-suite)
- [17.11 Exercises](#1711-exercises)
- [17.12 References](#1712-references-cited-in-this-chapter)
- [17.13 Further reading](#1713-further-reading)

---

## 17.1 The $\gamma$-decay catastrophe

### Why this is a *catastrophe* and not just slow learning

Chapter 8 set up TD(0) with the bootstrap target
$r + \gamma V(s')$. Section §6.1 noted that $\gamma$ controls the
effective horizon, but treated it benignly — small $\gamma$ means
faster discounting, large $\gamma$ means longer-horizon credit.
The picture is clean for $T \leq 100$.

For $T \geq 500$ — which is the Simulator's L-suite of long-horizon
scenarios — the picture is **catastrophic in a specific
mathematical sense**: the signal from a delayed reward, propagated
back through $T$ TD updates, gets multiplied by $\gamma^T$ — which
at $\gamma = 0.9, T = 500$ is $10^{-23}$. *That's below
single-precision float epsilon.* The credit signal isn't slow to
learn; it's literally unrepresentable in IEEE 754.

The downstream chapters are tools to *escape* this regime:

- **§17.2 (eligibility traces, Ch9 recap):** stretch credit
  exponentially-in-$\lambda$ instead of geometrically-in-$\gamma$.
  Helps but doesn't solve $T = 500$.
- **§17.3 (PBRS):** add reward shaping that propagates intermediate
  signal — but provably preserves the optimal policy.
- **§17.4 (HER):** relabel failed trajectories as successes for
  alternative goals; cheap synthetic credit.
- **§17.5 (successor features):** learn *what the future
  distribution looks like* separately from *what we reward*.
- **§17.6 (RUDDER):** decompose returns into per-step
  contributions via supervised learning.

Each is a different bet on what's cheap to add. This chapter
exists to make the choice; §17.10 maps the L-suite scenarios to
the cheapest fix that bridges them.

### The $\gamma$-decay numbers

For a one-shot reward at time $T$, the TD(0) backup at time $0$
multiplies it by $\gamma^T$. At $\gamma = 0.9$:

| $T$ ticks | $\gamma^T$ | Practical bound on learnable signal at $\alpha = 0.1$ |
|---|---|---|
| 10 | 0.349 | $0.0349 \cdot R$ |
| 50 | $5.15 \cdot 10^{-3}$ | $5 \cdot 10^{-4} \cdot R$ |
| 100 | $2.66 \cdot 10^{-5}$ | $3 \cdot 10^{-6} \cdot R$ |
| 200 | $7 \cdot 10^{-10}$ | $7 \cdot 10^{-11} \cdot R$ |
| 500 | $1.4 \cdot 10^{-23}$ | $1.4 \cdot 10^{-24} \cdot R$ |

For $R \sim 1$ and a numerical floor of $\sim 10^{-7}$ (single
precision), reward at $T = 200+$ is below the floor. TD updates from
Harvest cannot reach Plant; not as a small effect, but mathematically
*unrepresentable*.

### Three ways to read $\gamma^T \to 0$

The single number $\gamma^T \approx 10^{-23}$ is doing a lot of work
in the diagnosis above. Three distinct readings of *why* this number
defeats learning — they correspond to three classes of fix later in
the chapter.

**Reading 1 (algebraic, the IEEE-754 ceiling).** The TD(0) backup
chain multiplies a single learning step's update by $\gamma$ each time
it ripples backward. After $T$ hops the gradient signal at the
originating state is $\alpha \cdot \gamma^T \cdot R$ in magnitude. For
$\gamma = 0.9$, $T = 500$, this is $\sim 10^{-24}$ — *below* the
relative machine epsilon of single-precision float (≈ $10^{-7}$) and
even of double-precision ($\sim 10^{-16}$). The update isn't small;
it's *zero after rounding*. The gradient never lands. This is why no
amount of additional training samples helps: every individual update
is a no-op. Fixes here have to either *change the multiplier* (move
to option-level $\gamma_\Omega$, §17.9) or *bypass the chain
entirely* (RUDDER's LSTM, §17.6).

**Reading 2 (statistical, the variance ceiling).** Even before
floating-point precision matters, $\gamma^T$ enters the *variance* of
the value estimate. A Monte Carlo return $G_t = \sum_k \gamma^k r_{t+k}$
has variance that scales with the variance of each reward times
$\gamma^{2k}$ accumulated; for sparse delayed rewards, the *signal*
($\gamma^T R$) shrinks geometrically while the *noise* from
intermediate transitions does not. The signal-to-noise ratio at the
originating state is $\gamma^T R / \sigma$. Long before the IEEE
ceiling, the credit signal is buried in stochastic-update noise. This
is why eligibility traces (§17.2) plateau at horizon $\sim 100$ —
they smooth the bias-variance trade but cannot make the signal
amplitude exceed the noise floor of the trajectory sampling itself.
Fixes here have to *increase the signal at intermediate steps* (PBRS,
§17.3; RUDDER's redistribution, §17.6).

**Reading 3 (information-theoretic, the credit-assignment failure).**
At horizon $T$ the agent cannot tell, from local TD updates alone,
*which* of the $T$ intermediate actions was responsible for the
eventual reward. Even if the gradient signal were perfectly
representable, all $T$ states share the same $\gamma^T R$ pull-back —
the algorithm has no way to distinguish "Plant at $t = 0$ caused this"
from "Wait at $t = 250$ caused this". TD's locality is the problem:
it propagates value backward one step at a time and information about
the *cause* is lost in averaging. Fixes here have to *bring causal
structure in from outside*: HER (§17.4) relabels by visited states;
HCA (§17.7) computes a posterior over which action mattered; RUDDER
(§17.6) trains a recurrent net to find the causal step. These are
substantively different fixes from "amplify the signal" — they
recover *which step*, not just *that there was* a step.

### What this catastrophe doesn't say

The $\gamma^{500} \approx 0$ argument is sharper than its surface
reading. Three traps it does *not* license:

- **It does not say "use $\gamma = 1$".** Setting $\gamma = 1$ removes
  the discount but creates an undiscounted return whose convergence
  requires either a finite horizon (episode termination) or a proper
  policy that almost-surely terminates. Without one of these,
  $V^\pi(s)$ is undefined; with the L-suite's open-ended timeline,
  it would diverge. The $\gamma$ in TD is *not* a tunable knob; it is
  the contraction-modulus that makes the Bellman operator a
  contraction in the first place (Ch3 §3.6). Lowering it below 1 is
  what *makes TD work at all*. The catastrophe is a price paid for
  having a well-defined fixed point, not a bug to be tuned away.
- **It does not say "Monte Carlo solves the L-suite".** $\lambda = 1$
  in TD($\lambda$) recovers MC, which sidesteps the bootstrap chain.
  But MC's variance grows with the trajectory length: $\text{Var}[G_t]$
  for a 500-step return with stochastic intermediate transitions can
  be enormous, and the per-state visit count over a few seeds is too
  small to average it down. MC trades the $\gamma^T$ representation
  failure for a *sample-complexity* failure. The L-suite breaks both.
- **It does not say "deep networks fix this for free".** Nonlinear FA
  changes the function class but not the credit-assignment chain. A
  deep Q-network still backups via $r + \gamma \max_{a'} Q(s', a')$;
  the chain is still $T$-deep and $\gamma^T$ is still the multiplier.
  This is empirically known — DQN on Montezuma's Revenge (sparse,
  delayed reward) achieved near-zero score for years until
  fundamentally different techniques (curiosity, options, demos) were
  bolted on. The fix is structural, not architectural.

The chapter's framing is therefore: $\gamma^T \to 0$ is a hard
mathematical wall that motivates *qualitatively different* propagation
mechanisms — not better hyperparameters.

### Try it: how fast does the bootstrap die?

<div id="ch17-gamma-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/gamma_decay/widget.js"></script>

Slide γ (gamma) toward 1 to flatten the curve; slide toward 0 to crater
it. For the project's default γ = 0.9, the bootstrap value at step 500
is ≈ 10⁻²³ — well below single-precision float epsilon (≈ 10⁻⁷, the red
dashed line). That's why Suite-L scenarios needed reward shaping: the
algorithm literally cannot see the credit signal across the horizon.

### The Simulator's L-suite, concretely

[`long_horizon_harvest.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/curricula/long_horizon_harvest.rs)
poses (paraphrasing the scenarios):

- **L1**: Plant, wait $\sim 500$ ticks, Harvest, reward.
- **L2**: as L1 with additional drive pressure.
- **L3**: sustained planting over multiple seasons.
- **L4-L5**: variations on locality + competition.

The L1 outcome over 40 000 ticks × 4 seeds: Plant = 716 (locked in),
Consume = 0 (untried). The chapter's central worked example
([Chapter 1 project tie-in](01_linear_algebra.md#110-project-tie-in), revisited).

**Diagnosis**: the L-suite is unbridgeable by TD(0) at $\gamma = 0.9$.
The signal that Plant→500-tick wait→Harvest gives reward is
multiplied by $\gamma^{500} \approx 0$ and never reaches Plant's Q.

The remaining sections are tools to bridge this gap.

---

## 17.2 Eligibility traces, recap

Chapter 9 introduced TD($\lambda$) and eligibility traces. They are
mathematically a $\lambda$-weighted combination of $n$-step returns:

$$G_t^{(\lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}.$$

For long horizons:

- At $\lambda = 0$: TD(0), pure 1-step. As above, fails at long $T$.
- At $\lambda = 1$: Monte Carlo, full episode return. Unbiased; high
  variance.
- At intermediate $\lambda$: trade. For $\lambda = 0.9$ at horizon
  $T = 50$, eligibility decays as $(\lambda \gamma)^t = 0.81^{50} \approx 5 \cdot 10^{-5}$ —
  still small.

### True online TD($\lambda$)

[van Seijen et al. 2016] published the modern variant: equivalent to
forward-view TD($\lambda$) for *every* trajectory, not just in
expectation. Key for non-linear FA. Modest constant-factor improvement
on classical benchmarks.

### Limits

Eligibility traces help up to horizon $\sim 1/((1-\gamma)(1-\lambda)) \approx 100$
ticks at $\gamma = 0.9, \lambda = 0.9$. Past that, even traces decay
to nothing. So they are a **medium-horizon** tool, not a long-horizon
solution.

### Try it: where does the useful-trace region end?

<div id="ch17-eligibility-ribbon-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/eligibility_ribbon/widget.js"></script>

Slide γ and λ. The ribbon over a 100-state chain shades each state by
$(\gamma\lambda)^t$ on a log scale; the curve below tracks the same
quantity against the noise floor $\varepsilon \approx 10^{-7}$. At the
default γ = 0.95, λ = 0.9 (so γλ ≈ 0.855) the trace crosses ε around
lag ≈ 100 — the §17.2 claim. Pushing both sliders toward 1 grows the
useful region, but never to 500: that's why the chapter keeps going.

---

## 17.3 Potential-based reward shaping

Already met in Chapter 18: add $\gamma \Phi(s') - \Phi(s)$ to the
reward without changing the optimal policy.

### Domain-knowledge transfer via $\Phi$

The intuition: if you know that closer-to-the-food states have higher
value, encode it as $\Phi(s) = -d(s, \text{nearest food})$. Then the
shaped reward gives a positive signal at each step toward food, even
without consuming.

For the L-suite, a plausible potential is

$$\Phi(s) = \mathbf{1}[\text{plot is planted}] \cdot \alpha_\text{plant}$$

— if the plot is planted, $\Phi$ jumps up. The shaped reward
$\gamma \Phi(s') - \Phi(s)$ gives a positive signal *at the Plant
action* (going from unplanted to planted state), independent of
whether Harvest ever happens.

> **Theorem ([Ng, Harada & Russell 1999]).** For any function
> $\Phi: \mathcal{S} \to \mathbb{R}$, the shaped reward
> $\tilde R(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$
> has the same optimal policy as $R$ in the same MDP.

So the shaping is **safe**: you cannot create a wrong optimal policy
by choosing $\Phi$ badly. You can make learning faster or slower, but
not incorrect.

### Practical caution

PBRS only preserves the optimum in the limit. For practical
$\alpha, \gamma$ and finite horizons, the learning dynamics change. A poorly
chosen $\Phi$ can slow learning (the shaped reward becomes informative
in the wrong direction). The standard tip: choose $\Phi$ to approximate
$-V^{\star}$ (so the shaping turns the Bellman equation into "all states
have zero value", removing bootstrap entirely). In practice, you do not
know $V^{\star}$; the shaping is approximate.

### Application to the Simulator

A plot-planted potential could be added to
[`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs) without
breaking any existing test (PBRS preserves the optimum, so the
validation suite stays valid). Cheap, low-risk experiment.

### Try it: design a potential, watch learning accelerate

<div id="ch17-pbrs-shaper-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/pbrs_shaper/widget.js"></script>

A 12-state Plant → wait → Harvest chain. Pick a potential Φ (planted-flag,
distance-to-harvest, or anti-shaping) and a magnitude c. Tabular Q-learning
runs side by side under the chosen Φ versus the unshaped baseline. Good Φ
shifts the curve left; bad (anti) Φ shifts it right. The bar panel below
confirms the PBRS theorem: argmax(plant vs. wait) at state 0 stays positive
across shapings.

---

## 17.4 Hindsight Experience Replay

[Andrychowicz et al. 2017] for **goal-conditioned RL with sparse
binary reward**. The setup:

- Reward is $\mathbf{1}[s = g]$: 1 only if the agent reaches the
  goal $g$.
- Off-policy algorithm (DDPG, SAC, DQN).
- Replay buffer storing $(s, a, r, s', g)$.

The HER trick: when sampling from the buffer, **relabel** the goal
$g$ to be a state actually visited later in the trajectory. The
relabeled transition has reward 1 (because the "goal" was, in fact,
reached). The original failed trajectory becomes a successful
trajectory toward a *different* goal.

### Why this is brilliant

Goal-conditioned policies suffer from the sparsity problem: the agent
rarely reaches the literal goal in early training. With HER, *every*
trajectory teaches the policy something — about reaching *whatever
states were visited*. The agent learns navigation generally; later,
when the literal goal is sampled, the policy already knows how to
reach it.

Empirically: 4-10× sample-efficiency improvement on robotic
manipulation tasks (block stacking, fetch-and-place).

### Application to the Simulator?

HER requires goal-conditioned RL. The Simulator's reward is
homeostatic (drive-based), not goal-based. **No direct application.**

A speculative adaptation: treat each drive setpoint as a "goal". Train
a goal-conditioned policy "reach drive $d = 0$". HER-relabel
trajectories: an agent that started at hunger 0.6 and ended at 0.65
becomes "a successful trajectory toward hunger 0.65" — usable as
training data even though it failed the original "drive 0" goal.

This is a refactor more than a fix. Not on the project roadmap; included
for completeness.

### Try it: walk through the relabel

<div id="ch17-her-relabel-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/her_relabel/widget.js"></script>

Press **play** to step through a failed gridworld episode that never
reaches the intended goal g. HER promotes each visited state in turn to
"goal-of-the-day", relabels the transition with reward 1, and adds it to
the buffer. The right-hand bars contrast HER's positive-example count
against flat replay's zero. One failed trajectory becomes T usable
positive transitions.

---

## 17.5 Successor features and GPI

[Dayan 1993] introduced the **successor representation (SR)**.
[Barreto et al. 2017] lifted it to **successor features (SF)** and added
**Generalized Policy Improvement (GPI)** for transfer.

### The successor representation

For a tabular MDP under policy $\pi$, define

$$M^{\pi}(s, s') = \mathbb{E}^{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \mathbf{1}[s_t = s'] \mid s_0 = s\right].$$

$M^{\pi}(s, s')$ is the discounted expected number of visits to $s'$
starting from $s$ under $\pi$.

Then for *any* reward function $R: \mathcal{S} \to \mathbb{R}$:

$$V^{\pi}\_R(s) = \sum\_{s'} M^{\pi}(s, s') R(s').$$

Decomposes value into **dynamics** ($M^{\pi}$) and **reward** ($R$).
Change the reward, and you compute the new value without re-learning
the dynamics.

### Successor features

For high-dim observations, replace the indicator with a feature vector:
$\phi(s) \in \mathbb{R}^d$ (the same tile-coded features as before).
The successor feature is

$$\psi^{\pi}(s) = \mathbb{E}^{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \phi(s_t) \mid s_0 = s\right].$$

And if $R(s) = w^{\top} \phi(s)$ is linear in features, then
$V^{\pi}_R(s) = w^{\top} \psi^{\pi}(s)$ — instant value computation
under any linear reward.

### GPI

Suppose you have learned $\psi^{\pi_1}, \ldots, \psi^{\pi_K}$ for $K$
policies. For a new reward $R = w^{\top} \phi$, you can compute each
$V^{\pi_i}_R$ and take the max:

$$\pi_\text{new}(s) = \arg\max_{i, a} Q^{\pi_i}(s, a)$$

This **generalized policy improvement** (vs. classical policy
improvement which only considers one $\pi$) gives at least as good a
policy as the best of the bunch on the new task — without any
re-training.

### Application to the Simulator

The deleted forward-search planner had successor-feature leaf
bootstraps (Chapter 15 §13.6). Bringing them back could help the
L-suite if:

1. We train SFs for each drive ("the policy that minimizes hunger").
2. A new reward (Plant→Harvest) is computed as a linear combination
   of drive-relief signals.
3. GPI picks the best policy among learned ones for the new task.

Speculative; depends on the SFs being trainable on the multi-tick
horizons that defeat TD. The SFs themselves are TD-trained, so they
inherit the long-horizon problem. Probably not a silver bullet.

### Try it: SR heatmap on the four-rooms gridworld

<div id="ch17-sr-heatmap-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/sr_heatmap/widget.js"></script>

The 7×7 four-rooms layout. Slide the root cell and the discount γ to
see $M^\pi(\text{root}, \cdot)$ light up cells the policy reaches under
discount; switch π from uniform to down-right to watch the heatmap skew
toward the favoured direction. The R marker is a one-hot reward; the
readout shows $V^\pi_R(\text{root}) = M^\pi(\text{root}, R)$ updating
instantly — no re-learning the dynamics when the reward changes, which
is the SR's central practical claim.

---

## 17.6 RUDDER — return decomposition

[Arjona-Medina et al. 2019]: **train an LSTM to predict episode return
from trajectory, and use successive predictions' differences as a
dense reward signal**.

The construction:

1. Run an episode, collect $(s_t, a_t, r_t)$.
2. Train LSTM $f: (s, a)_{1:t} \to \hat R$ to predict the *total
   episode return* given the prefix.
3. Define the redistributed reward
   $\hat r_t = f(\ldots, t) - f(\ldots, t-1)$.

> **Theorem ([Arjona-Medina et al. 2019]).** Provided $f$ correctly
> predicts the return at every step, RUDDER's redistributed reward
> has the same expected return as the original, but **with
> exponentially lower variance in the long-horizon case**. The
> optimal policy is preserved.

### Why this works

The LSTM has access to *the whole prefix*. It sees the Plant action
at $t = 100$ and learns: "after Plant, expected return goes up by
0.5". That delta becomes the dense reward signal at $t = 100$. The
TD update at Plant has a clean signal *now*, not at $t = 600$.

RUDDER works because LSTMs can learn to **bridge** the long horizon
during the supervised return-prediction phase — they have the
representational capacity (recurrent state) to remember Plant at
$t=100$ and credit it for the Harvest reward at $t=600$. TD has no
such memory; that is why TD fails.

### Try it: return decomposition

<div id="ch17-rudder-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/rudder/widget.js"></script>

Top panel: the original sparse reward — one spike at $t = 29$, zero everywhere else. This is what TD has to back up across 30 steps of $\gamma$-decay. Bottom panel: the RUDDER-redistributed signal $\hat r_t = f(\ldots, t) - f(\ldots, t-1)$, with the credit-bearing steps (dashed orange lines) receiving non-zero reward. Slide $\delta$ from 0 (pure sparse) to 1 (full redistribution) and watch the spike migrate into a dense signal — the *same total return*, but each TD update now has something to chew on locally. That's the magic: short-horizon TD on the redistributed signal recovers what was originally a 30-step credit-assignment problem.

### Application to the Simulator

**This is the most promising single technique for the L-suite.** The
algorithm:

1. Collect L-suite trajectories under the current $\epsilon$-greedy
   policy.
2. Train an LSTM $f$ on (sequence → total return) pairs.
3. Compute redistributed reward $\hat r_t$ per tick.
4. Use $\hat r_t$ instead of $r_t$ in the Q-learner's TD update.

Cost: an LSTM in Rust (`burn` or `candle` crate), trained periodically
on collected trajectories. The Simulator's determinism canary becomes
trickier (LSTM inference must be deterministic), but tractable.

This is the kind of project-specific application the textbook should
motivate: a real $\gamma^{500}$ problem, a real published technique,
a path to implementation.

---

## 17.7 Hindsight credit assignment

[Harutyunyan et al. 2019] generalizes the HER intuition to non-
goal-based settings. The core idea:

For each past action $a_t$, weight its update by the **posterior
probability** that $a_t$ caused the observed outcome.

Concretely: compute $P(a_t \mid \text{outcome})$ using a learned
conditional model. The TD update for $(s_t, a_t)$ is scaled by this
posterior — actions strongly predictive of the outcome get the credit.

This is bayesian-style backward inference; RUDDER is a similar idea
implemented via LSTM. HCA is the more theoretically clean version;
RUDDER is the more directly trainable version.

### Try it: how the posterior re-weights the update

<div id="ch17-hca-posterior-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hca_posterior/widget.js"></script>

Edit the prior π over four actions and the evidence strength λ. The
top panel shows π vs. the hindsight posterior $P(\cdot \mid \text{outcome})$
for the highlighted step; the dot marks the observed action. The bottom
panel plots the per-step credit weight $w_t = P(a_t \mid \text{outcome})/\pi(a_t)$
over the trajectory. At λ = 0 every weight is 1 (HCA reduces to plain
PG); as λ grows, weights on observed actions rise above 1 and the
hindsight upweighting becomes visible.

---

## 17.8 COMA — counterfactual credit in multi-agent

[Foerster et al. 2018]. In multi-agent RL, the *joint* action of $N$
agents produced a reward. Each agent needs to know what its *own*
contribution was. The COMA trick:

For agent $i$, compute the counterfactual advantage

$$A^i(s, a^{1:N}) = Q(s, a^{1:N}) - \sum_{a^{*i}} \pi^i(a^{*i} \mid s) Q(s, a^{-i}, a^{*i})$$

— actual Q minus expected Q if agent $i$ had picked an action from its
own policy (averaging out the choice). This subtracts the
"contribution of other agents" baseline and isolates agent $i$'s
marginal effect.

For the Simulator's multi-agent social interactions (memes spreading,
trades), this is a candidate framework for credit-assigning who-did-
what. Currently not implemented; agents learn individually.

---

## 17.9 Hierarchy revisited

From Chapter 16: hierarchical RL transforms a $T$-step task into a
$T/\tau$-option-step task. At $\tau = 30$ ticks per option, the
L-suite's 500-tick chain becomes a ~17-option-step problem. With
option-level $\gamma_\text{option} \approx \gamma^{30} = 0.04$, the
option-level effective horizon is $1/(1 - 0.04) \approx 1.04$
option-steps — still tight.

So hierarchy alone does not solve the L-suite. **Hierarchy +
PBRS + RUDDER** in combination might:

- Hierarchy gives a slower clock.
- PBRS adds an immediate signal at the Plant action.
- RUDDER bridges the wait-for-Harvest gap with dense LSTM-derived
  reward.

Stacking is OK because each is policy-invariant (or
policy-preserving under conditions): PBRS by theorem, RUDDER by
construction, hierarchy by the option-bellman framework.

---

## 17.10 Project tie-in: which method fits the L-suite?

A method matrix:

| Method | Applies to L-suite? | Effort | Bridge gap | Pros | Cons |
|---|---|---|---|---|---|
| True online TD($\lambda$) | Partial | Low | $\sim 100$ ticks | Drop-in | Insufficient at 500 |
| Eligibility traces | Partial | Low | $\sim 100$ ticks | Drop-in | Same |
| PBRS with planted-flag $\Phi$ | **Yes** | Low | Provides immediate signal at Plant | Provable safety | Need to design $\Phi$ |
| HER | No | High | N/A | Not applicable (no goals) | Refactor required |
| Successor features | Maybe | Medium | Depends on SF training | Transfer to new reward | SFs themselves need long-horizon training |
| **RUDDER** | **Yes** | High | Whole episode | Direct attack on delayed reward; provable | LSTM dependency, det. canary |
| Hindsight CA | Maybe | High | Whole episode | Cleaner version of RUDDER | Theory less mature |
| COMA | No (single-agent) | N/A | N/A | Multi-agent specific | Not applicable to L-suite |
| Hierarchical (SMDP-Q) | Yes | Medium | $\tau$-fold gap | Composes with reward fixes | Insufficient alone |
| Hybrid hierarchical + PBRS + RUDDER | **Yes** | High | Provably bridges any gap | Cumulative effect | All three to implement |

### Recommended priority for the Simulator

1. **PBRS with a planted-flag $\Phi$** (low effort, low risk, partial
   gain). Provable safety; no retest of optimum-preservation. Just add
   a shaping term to `reward.rs`.
2. **SMDP-Q over existing recipes** (medium effort, medium gain).
   Chapter 16 §14.9 covered this — collapse the 500-tick gap to
   ~30 ticks at the recipe level.
3. **RUDDER** (high effort, high gain). The serious fix. Requires
   neural-net infrastructure but completely bridges the gap.

Combined, these would close out the L-suite as a tractable problem.
None is currently implemented; all three are on the menu for future
work.

### Try it: stack the fixes, see them compose

<div id="ch17-stacking-lab-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/stacking_lab/widget.js"></script>

Toggle PBRS, SMDP (option-level updates every 5 ticks), and a
lightweight RUDDER stand-in (prefix-density redistribution) on the same
20-state toy chain. The top plot overlays a learning curve per active
combination; the bottom bars report the first episode where max Q at
state 0 escapes the noise floor. Each fix shifts the curve left; turning
on all three turns the L-suite-toy problem from "unreachable in 600
episodes" into "solved in tens".

---

## 17.11 Exercises

1. **(Discount catastrophe numerics.)** Compute $\gamma^k$ for
   $k \in \{50, 100, 200, 500, 1000\}$ and $\gamma \in \{0.9, 0.95, 0.99, 0.999\}$.
   For what combinations is the value below float-32 precision
   ($\sim 10^{-7}$)?

2. **(PBRS preserves the optimum.)** Repeat the proof from Chapter 18
   §16.4 specifically for the planted-flag potential
   $\Phi(s) = \mathbf{1}[\text{planted}] \cdot c$. Identify the
   per-tick shaping contribution.

3. **(HER on FetchPush.)** Implement HER + DDPG on the FetchPush
   environment from OpenAI Gym Robotics. Compare to plain DDPG over
   1M steps.

4. **(Successor representation, 4-room MDP.)** On a 4-room gridworld,
   train the successor representation $M^{\pi}$ under uniform policy
   $\pi$. Visualize the resulting $M^{\pi}(s, \cdot)$ for a few states
   — confirm it captures connectivity.

5. **(RUDDER toy.)** Construct a delayed-reward MDP: 100-step chain
   where reward is 1 at step 100 if the agent picked action A at step
   0, else 0. Implement RUDDER with a small LSTM. Show that the
   redistributed reward at step 0 is approximately 1 (correctly
   credits the action).

6. **(L-suite reward shape paper sketch.)** Pseudo-code a
   plot-planted potential $\Phi(s)$ added to the Simulator's
   `RewardConfig`. Identify the value of $c$ (the per-plant bonus) that
   would balance against $w_\text{alive}$ over the 500-tick wait.

7. **(Combined fix paper sketch.)** Sketch the implementation of
   the recommended priority list (PBRS + SMDP-Q + RUDDER). Identify
   which validation tests would need updating, in what way.

8. **(γ-scaling experiment.)** Take the navigation test
   ([`navigation.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/navigation.rs)).
   Re-run at $\gamma \in \{0.7, 0.9, 0.99\}$. Plot
   `q_approach_advantage` vs. ticks. At what $\gamma$ does navigation
   start to fail?

---

## 17.12 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Andrychowicz et al. 2017] — HER (§17.4).
- [Arjona-Medina et al. 2019] — RUDDER (§17.6).
- [Barreto et al. 2017] — successor features + GPI (§17.5).
- [Dayan 1993] — successor representation (§17.5).
- [Foerster et al. 2018] — COMA (§17.8).
- [Harutyunyan et al. 2019] — Hindsight Credit Assignment (§17.7).
- [Levy et al. 2019] — HAC (§17.9).
- [Munos et al. 2016] — Retrace (§17.2).
- [Nachum et al. 2018] — HIRO (§17.9).
- [Ng, Harada & Russell 1999] — PBRS (§17.3).
- [van Seijen et al. 2016] — true online TD($\lambda$) (§17.2).

## 17.13 Further reading

| Source | What to read | Why |
|---|---|---|
| [Andrychowicz et al. 2017] | HER paper | The clearest exposition of goal relabeling |
| [Arjona-Medina et al. 2019] | RUDDER paper | The deepest delayed-reward attack |
| [Dayan 1993] | SR paper | Foundational, beautifully short |
| [Sutton & Barto 2018] | §17 | Off-policy + advanced topics |

---

**Next:** [Chapter 20 — Action Spaces Beyond Discrete](20_action_spaces.md) —
when discrete-action enumeration is the bottleneck.

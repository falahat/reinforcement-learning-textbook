# Chapter 2 — Probability and Statistics

> **Prerequisites:** comfort with sums and integrals; familiarity with
> "random variable" and "expected value" at the level of a first
> probability course is helpful but not assumed.

> **Citations:** in-line citations use `[Author Year]` keyed to the master
> [bibliography](bibliography.md). This chapter draws on
> [Goodfellow, Bengio & Courville 2016 (Ch. 3)] for the
> machine-learning-flavoured treatment of probability, and
> [Hoeffding 1963] for the concentration inequality that anchors
> sample-complexity arguments in RL.

> **Learning objectives:**
> 1. Compute expectations and conditional expectations; use the tower
>    property without thinking about it.
> 2. Recognise a Markov chain when you see one and articulate the
>    Markov property cleanly.
> 3. Apply Hoeffding's inequality to derive sample-complexity bounds
>    (how many samples to estimate a mean to ε accuracy with confidence
>    1-δ).
> 4. Distinguish independence from uncorrelatedness; understand how
>    trajectory data violates independence and what RL does about it.

## Why this chapter exists

RL is about uncertain outcomes — actions lead to randomly-sampled next
states with stochastic rewards, and the agent must reason about
*expected* return. So we constantly need to compute "the average of a
quantity that depends on randomness." Three ideas do most of the work:

1. **Expectation and conditional expectation.** $V^\pi(s) = \mathbb{E}[G_t \mid s_t = s]$
   is the definition of a value function. The **tower property**
   $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]$ is used implicitly in
   every Bellman derivation.
2. **The Markov property.** "The future depends on the past only through
   the present" is what makes Markov Decision Processes (Chapter 5)
   tractable. Without it, the state would need to include the entire
   history.
3. **Concentration inequalities.** When we say "after $n$ samples our
   estimate is within $\epsilon$ of the truth with probability $1 - \delta$,"
   we are applying Hoeffding or one of its cousins. This is the
   foundation of the exploration bonuses in Chapter 14.

If you've taken a probability course recently, skim. If it's been a
while, work through the concentration section carefully — that's where
RL's quantitative claims live.

## Table of contents

- [2.1 Random variables and expectation](#21-random-variables-and-expectation)
- [2.2 Conditional expectation and the tower property](#22-conditional-expectation-and-the-tower-property)
- [2.3 Markov chains](#23-markov-chains)
- [2.4 Concentration inequalities](#24-concentration-inequalities)
- [2.5 Independence and correlation](#25-independence-and-correlation)
- [2.6 Project tie-in](#26-project-tie-in)
- [2.7 Exercises](#27-exercises)
- [2.8 References and further reading](#28-references-and-further-reading)

---

## 2.1 Random variables and expectation

A **random variable** $X$ takes values from some set $\mathcal{X}$ with a
probability distribution $P$. The **expectation** of $X$ is

$$
\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \cdot P(X = x)
$$

(for discrete $X$; replace the sum with an integral and $P$ with a density
for continuous $X$). Expectation is **linear**:
$\mathbb{E}[aX + bY] = a \mathbb{E}[X] + b \mathbb{E}[Y]$ regardless of
whether $X$ and $Y$ are independent. This is what makes "the value of a
state is the average return from it" a tractable definition.

## 2.2 Conditional expectation and the tower property

The **conditional expectation** of $X$ given $Y = y$ is

$$
\mathbb{E}[X \mid Y = y] = \sum_x x \cdot P(X = x \mid Y = y)
$$

Treating $y$ as variable, $\mathbb{E}[X \mid Y]$ is itself a random variable
— a function of $Y$.

> **Tower property** (a.k.a. law of total expectation, a.k.a. law of iterated
> expectation):
>
> $$
> \mathbb{E}[X] = \mathbb{E}\big[\mathbb{E}[X \mid Y]\big]
> $$
>
>
> In words: the average of $X$ equals the average over $Y$ of the average of
> $X$ given $Y$.

This is used implicitly in *every* RL derivation. For example,
$Q^\pi(s, a) = \mathbb{E}[r + \gamma V^\pi(s')]$ is an application of the
tower property: we average over the next state $s'$, then $V^\pi(s')$ is
itself an average over the rest of the trajectory.

### Try it: the tower property in 3 lines

<div id="ch1-tower-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/tower_property/widget.js"></script>

Slide the policy weights π, the transition probabilities P, or the leaf
rewards. The "flat" expectation (sum over leaves) and the "iterated"
expectation (E[R] = Σ_a π(a) · E[R | a]) always match — that's the
tower property in action.

## 2.3 Markov chains

A sequence of random variables $X_0, X_1, X_2, \ldots$ is a **Markov chain**
if for all $t$:

$$
P(X_{t+1} = x_{t+1} \mid X_t = x_t, X_{t-1} = x_{t-1}, \ldots, X_0 = x_0) = P(X_{t+1} = x_{t+1} \mid X_t = x_t)
$$

In words: **the future depends on the past only through the present.** The
state $X_t$ contains all the information needed to predict $X_{t+1}$ — there
is no benefit to remembering older history.

This is the **Markov property**. Markov Decision Processes (Chapter 5)
extend Markov chains by adding actions and rewards. The MDP framework
assumes the state $s$ is a sufficient statistic in this sense.

The transition probabilities of a finite Markov chain form a
**stochastic matrix** $P$ (rows sum to 1). The linear-algebra side
of Markov chains — stationary distributions, mixing rates, spectral
gaps — lives in [Chapter 1, §1.9](01_linear_algebra.md#19-stochastic-matrices-and-markov-chains).

## 2.4 Concentration inequalities

### Why concentration is THE bridge from sampling to RL

Every algorithm in this book that estimates a value from samples
needs an answer to the question: **after $n$ samples, how close is
my estimate to the truth?** Monte Carlo prediction (Chapter 7),
Q-learning convergence (Chapter 8 §6.4 Watkins-Dayan), and
UCB-style bandit exploration (Chapter 14) all rest on the same
quantitative statement: *with $n$ samples you're within $\epsilon$
of the true mean, with probability at least $1 - \delta$, where
$n$ scales like $\log(1/\delta) / \epsilon^2$.*

That logarithmic-in-$\delta$, quadratic-in-$\epsilon$ scaling is
the **Hoeffding bound**. It's not the tightest concentration result
known — Bernstein, Chernoff, and empirical-Bernstein are tighter
in their respective regimes — but it's the *cleanest*, requires
the fewest assumptions, and is the one most RL theorems cite. If
you remember one concentration inequality from this book, make it
Hoeffding.

### Statement

We constantly want to say things like "the average of $n$ samples
is close to the true mean with high probability." The tool is a
**concentration inequality**.

> **Hoeffding's inequality** [Hoeffding 1963]**:** if
> $X_1, \ldots, X_n$ are independent random variables with
> $X_i \in [a, b]$, and $\bar X_n = \frac{1}{n} \sum_i X_i$ is
> their sample mean, then
>
> $$
> P\big(|\bar X_n - \mathbb{E}[X]| > \epsilon\big) \leq 2 \exp\left(\frac{-2 n \epsilon^2}{(b - a)^2}\right).
> $$

**Read this carefully — each piece matters:**
- **$|\bar X_n - \mathbb{E}[X]|$** is the deviation of the sample
  mean from the true mean. We're bounding the probability that this
  deviation exceeds $\epsilon$.
- **$\epsilon$** is the *tolerance* — how close to the truth we
  insist. Tighter (smaller $\epsilon$) requires more samples.
- **$n$** is the sample size. The bound is *exponentially* tight
  in $n$ — doubling $n$ squares the failure probability.
- **$(b - a)$** is the support width. Heavier tails (wider support)
  loosen the bound. For rewards bounded in $[0, 1]$, $(b - a) = 1$
  and the bound is at its tightest natural form.

**What it doesn't require:** Gaussianity, unimodality, finite
variance beyond boundedness. *Any* bounded i.i.d. distribution
satisfies this — it's a property of $[a, b]$-valued averages, not
of any specific distribution.

### Try it: Hoeffding sample-size calculator

<div id="ch1-hoeffding-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/hoeffding/widget.js"></script>

The solid curve is the Hoeffding bound vs sample size n. The dashed
horizontal line is your target failure probability δ; the vertical
marker is the smallest n at which the bound drops below δ. Halve ε
and watch n quadruple — the **quadratic** scaling Hoeffding gives you.

> **Notation: `exp`.** `exp(x)` is just another way to write $e^x$ —
> the exponential function, where $e \approx 2.71828$ (Euler's
> number). When the exponent is a long expression like
> $-2n\epsilon^2/(b-a)^2$, writing it as a superscript on $e$ would
> shrink it to unreadable size; `exp(...)` keeps it at full size.
> Mathematically identical. You will see this notation everywhere in
> RL — most concentration bounds, softmax probabilities, KL
> divergences, and Gaussian densities are exponential in their
> parameters.

### Solving for $n$: how many samples do we need?

The exponential decay in $n \epsilon^2$ is what makes finite-sample
guarantees in RL possible: it tells us how many samples we need to
estimate a quantity to within $\epsilon$ with confidence
$1 - \delta$. Solving
$2 \exp(-2n\epsilon^2 / (b-a)^2) \leq \delta$ algebraically:

$$
\frac{-2 n \epsilon^2}{(b-a)^2} \stackrel{(1)}{\leq} \log(\delta / 2) \stackrel{(2)}{\Longleftrightarrow} n \stackrel{(3)}{\geq} \frac{(b-a)^2 \log(2/\delta)}{2\epsilon^2}.
$$

Step (1) takes the log of both sides (log is monotone, inequality
direction preserved). Step (2) multiplies by the negative
coefficient $-(b-a)^2 / 2\epsilon^2$, which flips the inequality.
Step (3) uses $-\log(\delta/2) = \log(2/\delta)$. The final form is
the sample complexity:

$$
n \geq \frac{(b-a)^2 \log(2/\delta)}{2\epsilon^2}.
$$

**Two takeaways from the scaling:**

- **Quadratic in $\epsilon$.** Halving the tolerance *quadruples*
  the required samples. Two-decimal-place accuracy ($\epsilon = 0.01$)
  needs $\sim 10^4$ samples; four-decimal-place accuracy
  ($\epsilon = 0.0001$) needs $\sim 10^8$. RL with sample budgets in
  the millions can hit $\epsilon \approx 10^{-3}$ comfortably; can't
  hit $10^{-5}$.
- **Logarithmic in $\delta$.** Going from 95% confidence
  ($\delta = 0.05$) to 99.999% confidence ($\delta = 10^{-5}$)
  only doubles the samples. *Extreme confidence is cheap.* This is
  why bandit algorithms (Chapter 14) can afford to be paranoid
  about their confidence intervals — pushing $\delta$ tiny costs
  almost nothing.

### Three ways concentration shows up in RL

**1. Sample-complexity proofs.** Chapter 8's Q-learning
convergence (Watkins-Dayan 1992), the stochastic-approximation
bounds in linear function approximation (Chapter 10
[Tsitsiklis & Van Roy 1997]), and the regret bounds in Chapter 14
(bandits) all invoke Hoeffding-type concentration to say "after $n$
visits to a state-action pair, the empirical Bellman backup is
within $\epsilon$ of the true backup with high probability." The
$\sqrt{\log n / n}$ confidence-interval form in UCB (Chapter 14
§12.4) is literally Hoeffding inverted to solve for $\epsilon$.

**2. Stopping rules for Monte Carlo.** When evaluating a policy by
Monte Carlo rollouts (Chapter 7), Hoeffding tells you when you've
sampled enough: stop when the half-width of the empirical mean's
confidence interval drops below your tolerance. The Simulator's
calibration runs use exactly this logic to decide how many seeds
to evaluate per scenario (`crates/sim/sim_test_harness`).

**3. Exploration-vs-exploitation trade-offs.** UCB acts greedily
with respect to the *upper confidence bound* on each arm's mean:
$\mu_i + \sqrt{2 \log t / n_i}$ where the square-root term is the
Hoeffding-derived confidence margin. Arms with few samples
($n_i$ small) get optimistically inflated; arms with many samples
behave near-greedy. The whole UCB-regret theorem hinges on
Hoeffding for the deviation probability.

### What this section doesn't say

- **It doesn't apply to non-i.i.d. samples.** The bound assumes
  independence; in RL we routinely sample from correlated
  trajectories (Markov chains, replay buffers). For Markov-chain
  samples, Hoeffding generalises to a *mixing-time*-dependent
  bound (Markov-chain Hoeffding inequalities, e.g. [Paulin 2015]).
  The penalty is a factor proportional to the mixing time; not a
  fundamental obstacle.
- **It's not the tightest bound available.** When the variance is
  much smaller than $(b-a)^2$, Bernstein's inequality is
  significantly tighter (scales with the variance, not the
  support). For Gaussian-like distributions, sub-Gaussian
  concentration is even tighter. Hoeffding is the "always works,
  rarely tight" workhorse.
- **It bounds the *probability* of deviation, not the *expected*
  deviation.** $\mathbb{E}|\bar X_n - \mathbb{E}[X]|$ scales like
  $O(1/\sqrt n)$ by the CLT; Hoeffding tells you about the *tails*,
  not the bulk. For RL convergence, tails are usually what matters
  (worst-case behaviour over many states/actions).

## 2.5 Independence and correlation

Two random variables $X$ and $Y$ are **independent** if
$P(X, Y) = P(X) \cdot P(Y)$. Independence implies
$\mathbb{E}[XY] = \mathbb{E}[X] \mathbb{E}[Y]$ but the converse fails
(uncorrelated $\not\Rightarrow$ independent).

In RL, samples from a trajectory are **not independent** — consecutive
states are correlated through the dynamics. This breaks naive applications
of concentration inequalities and is one reason DQN uses **experience
replay** (Chapter 11): sampling uniformly from a buffer of past transitions
approximately restores independence.

## 2.6 Project tie-in

### What's missing — Hoeffding-driven exploration

The Simulator does not currently use Hoeffding-type concentration
inequalities to decide "have I sampled this action enough to commit to
it?" — there is no UCB-style bonus on action selection. Exploration is
plain $\epsilon$-greedy at $\epsilon = 0.10$
([`learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs)).
Chapter 14 covers what an upgrade would look like (UCB1, Thompson
sampling, count-based and curiosity bonuses). The Q-bias bootstrap
pathology analysed in Chapter 3's project tie-in is partly an
exploration failure: an action that was never tried has no $Q$
estimate to update, and $\epsilon$-greedy with deterministic
tie-breaking
([`policy.rs:493-501`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs))
keeps it that way.

## 2.7 Exercises

1. **(Tower property in action.)** Let $X$ be the return of a trajectory
   under policy $\pi$ starting at $s_0 = s$. Show that

   $$\mathbb{E}[X] = \sum_a \pi(a \mid s) \cdot \mathbb{E}[X \mid a_0 = a]$$

   What is $\mathbb{E}[X \mid a_0 = a]$ called?

2. **(Hoeffding sample complexity.)** You want to estimate the mean of a
   random variable in $[0, 1]$ to within $\epsilon = 0.01$ with confidence
   $1 - \delta = 0.99$. How many samples do you need?

3. **(Why trajectories aren't independent.)** Consider a Markov chain
   on two states $\{0, 1\}$ with transition probability 0.99 of staying
   put. You observe a trajectory of length $n = 100$ starting from
   state 0. Why is the empirical fraction of time spent in state 0 a
   bad estimate of the stationary distribution? What does experience
   replay accomplish?

4. **(Conditional vs unconditional expectation.)** Let $Y$ be uniform on
   $\{1, 2, 3\}$ and $X = Y^2$ with probability 1. Compute $\mathbb{E}[X]$
   directly. Then compute it via the tower property using $Y$ as the
   conditioning variable. (Trivial — but it makes the mechanics
   concrete.)

## 2.8 References and further reading

Full bibliographic entries in [`bibliography.md`](bibliography.md):

- [Goodfellow, Bengio & Courville 2016, Ch. 3] — probability for ML.
- [Hoeffding 1963] — the canonical concentration inequality.
- [Wasserman, *All of Statistics*] — fast, dense reference for the
  full probability + statistics toolkit.

| Source | What to read | Why |
|---|---|---|
| [Goodfellow, Bengio & Courville 2016] | Ch. 3 | Probability through an ML lens. |
| Wasserman, *All of Statistics* | Ch. 1–4 | If you want the careful version. |
| [Bertsekas & Tsitsiklis, *Introduction to Probability*] | Ch. 7 (Markov chains) | The classical treatment. |

---

**Next:** [Chapter 3 — Mathematics for AI](03_mathematics_for_ai.md) — multivariable calculus, optimization, contraction mappings, and the convergence theory that makes RL work.

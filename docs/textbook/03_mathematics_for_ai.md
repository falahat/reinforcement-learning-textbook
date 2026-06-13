# Chapter 3 — Mathematics for AI

> **Prerequisites:** [Chapter 1 (Linear Algebra)](01_linear_algebra.md)
> and [Chapter 2 (Probability)](02_probability_and_statistics.md).
> Comfort with single-variable calculus.

> **Citations:** [Banach 1922] for the fixed-point theorem,
> [Robbins & Monro 1951] for stochastic approximation,
> [Szepesvári 2010] for the proof structure we adopt,
> [Tsitsiklis 1994] for asynchronous TD convergence.

> **Learning objectives:**
> 1. Compute gradients and Hessians; predict gradient-descent behaviour
>    from the Hessian's eigenvalues.
> 2. State the Banach fixed-point theorem and prove that the Bellman
>    operator is a γ-contraction in $\ell_\infty$.
> 3. Articulate the Robbins–Monro conditions and explain why constant
>    step sizes converge to a *neighbourhood*, not a point.
> 4. Use the contraction modulus γ in both roles: discount factor for
>    return and convergence rate for value iteration.

## Why this chapter exists

Every theorem about RL convergence uses one of two ideas:

1. **Contraction mappings.** The entire reason value iteration converges,
   the entire reason Q-learning converges, the reason discounting works at
   all. One theorem ([Banach 1922]) is used everywhere.
2. **Stochastic approximation.** The reason TD learning with a step size
   $\alpha$ converges to a fixed point despite never seeing the same state
   twice ([Robbins & Monro 1951]).

Plus we need multivariable optimisation basics — gradients, Hessians,
convexity, gradient descent — to build any of the function-approximation
methods in Chapter 10 onward.

If you've seen all of this before, **skim and jump to the exercises**. If
some pieces are rusty, the relevant section will refresh you. If you've
never seen contraction mappings — read that one carefully. It's the key
theorem of the entire field.

## Table of contents

- [3.1 Univariate calculus refresher](#31-univariate-calculus-refresher)
- [3.2 Gradient: the multivariate first derivative](#32-gradient-the-multivariate-first-derivative)
- [3.3 Hessian: the multivariate second derivative](#33-hessian-the-multivariate-second-derivative)
- [3.4 Stochastic gradient descent (SGD)](#34-stochastic-gradient-descent-sgd)
- [3.5 Convexity](#35-convexity)
- [3.6 Contraction mappings — *the* theorem of RL](#36-contraction-mappings--the-theorem-of-rl)
- [3.7 Stochastic approximation](#37-stochastic-approximation)
- [3.8 Project tie-in](#38-project-tie-in)
- [3.9 Exercises](#39-exercises)
- [3.10 References and further reading](#310-references-and-further-reading)

---

## 3.1 Univariate calculus refresher

For a function $f: \mathbb{R} \to \mathbb{R}$, the **derivative** $f'(x)$
is the instantaneous slope. The **second derivative** $f''(x)$ is the rate
of change of the slope — the **curvature**.

- $f'(x) = 0$ identifies a stationary point (flat).
- $f''(x) > 0$ at a stationary point: it is a local **minimum** (curving
  upward — like a cup ∪).
- $f''(x) < 0$ at a stationary point: a local **maximum** (curving
  downward, ∩).
- $f''(x) = 0$: inconclusive (saddle or inflection).

These two facts — "slope is zero, curvature is positive" — generalize to
multiple dimensions through the **gradient** and **Hessian**. The
generalization is more than a notational change: the curvature picture is
what makes most of optimization make sense.

## 3.2 Gradient: the multivariate first derivative

For a differentiable $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is
the column vector of partial derivatives:

$$\nabla f(x) = \big(\partial f / \partial x_1, \ldots, \partial f / \partial x_n\big)^{\top}.$$

Three equivalent things to know:

1. **Direction of steepest ascent.** Among all unit vectors $u$, the one
   that maximizes $u^{\top} \nabla f(x)$ is $u = \nabla f(x) / \|\nabla f(x)\|$.
2. **Best linear approximation.** Near $x_0$,
   $f(x) \approx f(x_0) + \nabla f(x_0)^{\top} (x - x_0)$ — Taylor's
   theorem to first order.
3. **A vector that lives in the same space as $x$.** Gradients of an
   $n$-dim input are themselves $n$-dim. SGD adds $\eta \nabla f$ to $x$;
   this would not type-check otherwise.

**Gradient descent** moves opposite to the gradient:

$$x_{k+1} = x_k - \eta \nabla f(x_k)$$

with step size $\eta > 0$. For convex $f$ and small enough $\eta$, it
converges to the global minimum. For non-convex $f$ (neural-network
losses), it converges to *a* stationary point — which might be a
saddle, a local min, or a degenerate flat.

**Step-size sensitivity.** If $\eta$ is too large, the iterates overshoot
and may diverge. The safe range depends on curvature (the Hessian, next):
gradient descent on a quadratic $\tfrac{1}{2} v^{\top} A v$ converges
iff $\eta < 2 / \lambda_{\max}(A)$. Too-small $\eta$ converges, just
slowly — needing $O(\kappa / \eta)$ iterations where $\kappa$ is the
condition number. **This is the whole game in non-RL deep learning:
balance step size against curvature.**

### Try it: gradient descent on three surfaces

<div id="ch1-gradient-descent-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/gradient_descent/widget.js"></script>

Pick a surface, drop a start point, set the learning rate η. The
contour plot traces x_k as a polyline; the line chart tracks f(x_k).
The convex bowl finds the minimum cleanly when η is small; crank η
past `2/λ_max` and it diverges. The banana (Rosenbrock) makes GD
crawl along the curved valley floor — a classic ill-conditioning
demo. The saddle escapes off the y-axis once y₀ ≠ 0.

## 3.3 Hessian: the multivariate second derivative

If you take a single class away from this chapter, take this.

> **Definition.** The **Hessian** of $f: \mathbb{R}^n \to \mathbb{R}$ at
> $x$ is the $n \times n$ matrix of second partial derivatives:
>
> $$
> H f(x)_{ij} = \frac{\partial^2 f}{\partial x_i\, \partial x_j}.
> $$
>
> When $f$ is twice continuously differentiable, $H f$ is **symmetric**
> ($\partial^2 f / \partial x_i \partial x_j = \partial^2 f / \partial x_j \partial x_i$,
> a.k.a. *Clairaut's theorem*).

What does it mean? Three lenses:

1. **It is the curvature.** Just as $f''(x) > 0$ tells you a univariate
   function curves upward at $x$, the Hessian being *positive-definite*
   tells you a multivariate function curves upward in every direction.
   Eigendecompose $H f(x_0) = U \Lambda U^{\top}$: in the rotated basis,
   $f$ near $x_0$ looks like a sum of independent parabolas with
   curvatures $\lambda_1, \ldots, \lambda_n$.

2. **It governs the second-order Taylor approximation.**

   $$f(x) \approx f(x_0) + \nabla f(x_0)^{\top} (x - x_0) + \tfrac{1}{2} (x - x_0)^{\top} H f(x_0) (x - x_0).$$

   The cross term is the gradient (linear). The Hessian term (quadratic)
   is what gradient descent ignores.

3. **It is the "right" preconditioner.** **Newton's method** sets the
   update to $-[H f(x)]^{-1} \nabla f(x)$ — locally, this minimizes the
   quadratic approximation in one step. If $f$ were exactly a quadratic
   bowl, Newton would find the minimum in one iteration regardless of
   condition number. (The catch: computing and inverting $H$ in
   high dimensions is intractable. More below.)

**Second-order optimality condition.** $x^{\star}$ is a local minimum of $f$ if
$\nabla f(x^{\star}) = 0$ (stationary) **and** $H f(x^{\star})$ is positive-definite
(curving up in all directions). If the Hessian has both positive and
negative eigenvalues, $x^{\star}$ is a **saddle point**: a min along some
directions, a max along others. High-dim non-convex landscapes (neural
nets) are dominated by saddles, not local minima — this is the modern
view from [Dauphin et al. 2014]\footnote{Dauphin, Pascanu, Gulcehre, Cho, Ganguli, Bengio, *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*, NeurIPS 2014.}.

**Convexity is PSD Hessian.** $f$ is convex iff $H f(x) \succeq 0$ (positive
semi-definite) everywhere. Strictly convex iff $H f(x) \succ 0$
(positive-definite). This is the multivariate generalization of "second
derivative non-negative."

**Why we rarely use Newton in deep learning.** For $n = 10^7$ parameters,
$H$ has $10^{14}$ entries. Storing it is out, inverting it is *way* out.
The whole tower of modern optimizers can be read as **approximating the
Hessian preconditioner cheaply**:

| Method | Hessian approximation | Cost |
|---|---|---|
| Gradient descent | Identity (i.e., assume isotropic curvature) | $O(n)$ per step |
| Adam / RMSProp | Diagonal — running average of $\nabla f \odot \nabla f$ | $O(n)$, often used |
| L-BFGS | Low-rank: last $k$ gradient differences | $O(kn)$, $k \sim 20$ |
| Natural gradient | Fisher information $F = \mathbb{E}[\nabla \log p \nabla \log p^{\top}]$ — Hessian of KL | $O(n^2)$ exact, $O(kn)$ with K-FAC |
| Newton | Full $H$ | $O(n^2)$ storage, $O(n^3)$ solve |

In RL specifically, **natural policy gradient** (Chapter 13 / TRPO / PPO)
is the most common "use second-order information." The Fisher
information matrix plays the role of the Hessian; K-FAC approximates it
block-diagonally. This is why TRPO/PPO converge so much more reliably
than vanilla REINFORCE on hard problems.

## 3.4 Stochastic gradient descent (SGD)

When $f(x) = \mathbb{E}_\xi[F(x; \xi)]$ for some random $\xi$, the exact
gradient is unavailable (it is an expectation). We sample $\xi_k$ and
update

$$x_{k+1} = x_k - \eta_k \nabla F(x_k; \xi_k).$$

This is **stochastic gradient descent**. The expected update is in the
true gradient direction; per-step noise averages out across iterations
(if step sizes are right).

> **Robbins-Monro conditions** [Robbins & Monro 1951]:
>
> $$
> \sum_k \eta_k = \infty, \qquad \sum_k \eta_k^2 < \infty.
> $$
>
> Under these (and bounded variance, plus other regularity), SGD converges
> almost surely to a critical point.

Why these two conditions? The first ensures we can travel arbitrarily
far — if $\sum \eta_k$ were finite, we could get stuck before reaching
the optimum. The second ensures noise dies out — squared step sizes
control the variance of the random walk.

Common choice: $\eta_k = c/k$ satisfies both. A **constant step size**
$\eta_k = \alpha$ satisfies the first but **not** the second, so SGD with
constant $\alpha$ does *not* converge to a point — it converges to a
**neighborhood** of the optimum with radius $O(\sqrt{\alpha})$. This is
the same phenomenon we will meet as the *TD-error noise floor* in
Chapter 8, and it is what the project's TD-error calibration baselines
(`td_error_floor_*.json` artifacts) measure.

**Why almost all of practical RL uses constant $\alpha$ anyway.** Real
environments are non-stationary — the agent's own policy changes, so the
target distribution moves. A schedule $\eta_k \to 0$ would forget
non-stationarity has happened. Constant $\alpha$ tracks the moving
target at the cost of an irreducible noise floor. The Simulator's
$\alpha = 0.10$ ([`learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs))
is this trade-off explicitly chosen.

### Try it: Robbins-Monro step-size race

<div id="ch1-robbins-monro-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/robbins_monro/widget.js"></script>

Three SGD runs on f(θ) = ½(θ − μ)² with noisy gradient. Each uses a
different step-size schedule: α_k = c/k (green, RM-compliant), α_k =
c/√k (blue), α_k = c constant (red). The constant-α run never lands
— it bounces forever in a band of radius O(σ√α/2) around μ (red
shading). The 1/k schedule asymptotically hugs the dashed truth.
That is the section's punchline made geometric.

## 3.5 Convexity

A function $f$ is **convex** if its graph lies below any chord:

$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y), \quad \forall \lambda \in [0,1].$$

Equivalently (for twice-differentiable $f$), $H f(x) \succeq 0$ everywhere —
exactly the Hessian condition above. **Convex problems have no local minima
other than the global minimum**: every stationary point is the optimum.

In RL:

- The projected Bellman error is convex in **linear**-FA weights $\theta$.
  This is why LSTD has a *closed-form* solution (solve a linear system).
- Neural-network Q-values are **non-convex** in their parameters. DQN
  training is finicky because the loss landscape has saddles, plateaus,
  and degenerate basins — and bootstrapping makes the target itself a
  moving function of the same parameters (deadly triad — Chapter 17).
- **Policy gradient** objectives in the policy parameters $\theta$ are
  non-convex (the policy enters the expected return non-linearly through
  the trajectory distribution). This is why basic REINFORCE has high
  variance and why advantage methods and trust-region constraints
  (Chapter 13) exist.

### Try it: Jensen's inequality

<div id="ch1-jensen-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/jensen/widget.js"></script>

Drag x and y; slide λ between them. The dot on the curve is
`f(λx + (1-λ)y)`; the dot on the chord is `λf(x) + (1-λ)f(y)`.
For convex f, chord ≥ curve everywhere — that's Jensen. Switch to
log(x) and watch the inequality flip.

## 3.6 Contraction mappings — *the* theorem of RL

### Why this single result is the most-cited theorem in the book

Every convergence proof in Chapters 5 (Bellman expectation),
6 (DP), 8 (TD), 10 (linear FA), 11 (DQN target net), 17 (the deadly
triad's *failure* mode) reduces, ultimately, to one statement:
**a contraction on a complete metric space has a unique fixed
point, reached by iteration at geometric rate $\gamma$.** That's
the Banach fixed-point theorem.

If you walked away from this textbook with one mathematical fact,
this is the one to take. It explains *why* value iteration
converges, *why the discount factor is the convergence rate*, *why
TD bootstraps work*, and *why off-policy bootstrapping with
function approximation can diverge* (it loses the contraction
property). Master this section — every theorem that follows is a
specialised version of it.

This is the single most important section of this chapter.
Master it.

### Metric spaces, briefly

A **metric space** $(X, d)$ is a set $X$ with a distance function
$d: X \times X \to \mathbb{R}_{\geq 0}$ satisfying:

1. $d(x, y) = 0 \iff x = y$
2. $d(x, y) = d(y, x)$ (symmetry)
3. $d(x, z) \leq d(x, y) + d(y, z)$ (triangle inequality)

Examples: $\mathbb{R}^n$ with Euclidean distance; the space of bounded
functions $f: \mathcal{S} \to \mathbb{R}$ with the **sup-norm metric**
$d(f, g) = \|f - g\|_\infty = \sup_s |f(s) - g(s)|$.

The metric space of bounded functions on a state space is the one we'll
use — value functions $V: \mathcal{S} \to \mathbb{R}$ are exactly such
functions.

A metric space is **complete** if every Cauchy sequence has a limit in the
space. (A Cauchy sequence is one whose terms get arbitrarily close
together.) Both $\mathbb{R}^n$ and the bounded-function space are
complete.

### Contraction mappings

> **Definition.** Let $(X, d)$ be a metric space. A map $T: X \to X$ is a
> **contraction** with modulus $\gamma \in [0, 1)$ if
>
> $$
> d(T(x), T(y)) \leq \gamma \cdot d(x, y) \quad \forall x, y \in X.
> $$
>

In words: applying $T$ shrinks distances by at least a factor of $\gamma$.

### The Banach fixed-point theorem

> **Theorem ([Banach 1922]).** Let $(X, d)$ be a complete metric space and
> $T: X \to X$ a contraction with modulus $\gamma$. Then:
> 1. $T$ has a unique **fixed point** $x^{\star} \in X$ (i.e. $T(x^{\star}) = x^{\star}$).
> 2. For any starting point $x_0$, the iteration $x_{k+1} = T(x_k)$
>    converges to $x^{\star}$.
> 3. The convergence rate is *geometric*:
>
> $$
> d(x_k, x^{\star}) \leq \gamma^k \cdot d(x_0, x^{\star}).
> $$
>

### Try it: a contraction mapping in 1D

<div id="ch1-contraction-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/contraction/widget.js"></script>

Top plot is the cobweb iteration. Bottom plot shows `|x_k - x*|` on
a log-y scale — a straight line because convergence is geometric at
rate γ. Slide γ toward 1 and watch the rate flatten; toward 0 and
watch it crash to the fixed point in one or two steps.

### Try it: summation step-by-step

<div id="ch1-summation-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/summation/widget.js"></script>

Press Next to add the next term to the sum. The bar chart shows
each term's value; the line plot shows the running partial sum
approaching the limit (for convergent series) or growing without
bound. Compare the geometric series's geometric convergence to the
harmonic series's logarithmic divergence.

### Proof, step by step

The proof has three pieces: **uniqueness**, **existence**, and the
**geometric convergence rate**. We do them in that order because
uniqueness is the cleanest argument and warms us up for the rest.
Structure follows [Szepesvári 2010, Appendix].

**Part 1 — uniqueness.** Suppose $T$ has two fixed points $x^\star$
and $y^\star$. Then

$$
d(x^\star, y^\star) \stackrel{(1)}{=} d(T(x^\star), T(y^\star)) \stackrel{(2)}{\leq} \gamma \cdot d(x^\star, y^\star).
$$

Step (1) uses $T(x^\star) = x^\star$ and $T(y^\star) = y^\star$
(both are fixed points). Step (2) is the contraction property.
Rearranging:

$$
(1 - \gamma) \cdot d(x^\star, y^\star) \leq 0.
$$

Since $\gamma < 1$, $(1 - \gamma) > 0$, so $d(x^\star, y^\star) \leq 0$.
Combined with $d \geq 0$ (metric axiom), $d(x^\star, y^\star) = 0$,
hence $x^\star = y^\star$. **At most one fixed point exists.**

**Part 2 — existence (the iteration is Cauchy).** Pick any starting
point $x_0$ and iterate $x_{k+1} = T(x_k)$. We need to show this
sequence converges. The plan: show consecutive terms get arbitrarily
close (Cauchy), then invoke completeness to get a limit.

*Step (a): consecutive distances shrink geometrically.* Apply the
contraction property iteratively:

$$
d(x_{k+1}, x_k) = d(T(x_k), T(x_{k-1})) \leq \gamma \cdot d(x_k, x_{k-1}) \leq \gamma^2 \cdot d(x_{k-1}, x_{k-2}) \leq \cdots \leq \gamma^k \cdot d(x_1, x_0).
$$

Each iteration of $T$ contracts the gap by $\gamma$, so after $k$
applications the gap is $\gamma^k$ times the initial gap.

*Step (b): distances between far-apart terms also vanish.* For any
$k$ and $m$,

$$
d(x_{k+m}, x_k) \stackrel{(3)}{\leq} \sum_{i=0}^{m-1} d(x_{k+i+1}, x_{k+i}) \stackrel{(4)}{\leq} \sum_{i=0}^{m-1} \gamma^{k+i} \cdot d(x_1, x_0) \stackrel{(5)}{\leq} \frac{\gamma^k}{1 - \gamma} \cdot d(x_1, x_0).
$$

Step (3) is the triangle inequality applied to the $m$-link chain
$x_k \to x_{k+1} \to \cdots \to x_{k+m}$. Step (4) plugs in
step (a)'s bound. Step (5) sums the geometric series and bounds it
by the infinite sum (which is $\gamma^k / (1 - \gamma)$).

*Step (c): the sequence is Cauchy.* For any $\epsilon > 0$, pick
$k$ large enough that $\gamma^k d(x_1, x_0) / (1 - \gamma) < \epsilon$.
Then any two terms past index $k$ are within $\epsilon$ of each
other. That's the definition of Cauchy.

*Step (d): the limit exists and is a fixed point.* By completeness
of $(X, d)$, the Cauchy sequence has a limit $x^\star \in X$. By
continuity of $T$ (contractions are continuous — they're 1-Lipschitz),

$$
T(x^\star) = T\left(\lim_k x_k\right) = \lim_k T(x_k) = \lim_k x_{k+1} = x^\star.
$$

So $x^\star$ is a fixed point of $T$.

**Part 3 — geometric convergence rate.** We've shown a fixed point
$x^\star$ exists and is unique. Now the rate. By induction,

$$
d(x_k, x^\star) = d(T(x_{k-1}), T(x^\star)) \leq \gamma \cdot d(x_{k-1}, x^\star) \leq \gamma^2 \cdot d(x_{k-2}, x^\star) \leq \cdots \leq \gamma^k \cdot d(x_0, x^\star).
$$

The first equality is the contraction. Each rewriting applies the
contraction to the previous bound. After $k$ applications, error is
$\gamma^k$ times the initial error. ☐

### Three readings of Banach

**1. As a fixed-point existence theorem.** The cleanest mathematical
reading: under contraction + completeness, *something* exists that
$T$ maps to itself. This is what the theorem says — a *guarantee*
that the desired object (the optimal value function, the stationary
distribution, the equilibrium) is well-defined.

**2. As an algorithm.** The proof is *constructive*: start anywhere,
iterate $T$, the sequence converges. Value iteration, policy
evaluation, fixed-point Q-learning — all are this loop with a
problem-specific $T$. The same theorem that proves $V^\star$ exists
gives you the algorithm to compute it.

**3. As a complexity bound.** The geometric convergence rate is
exactly the number of iterations to reach $\epsilon$ accuracy:
$k \geq \log(d(x_0, x^\star) / \epsilon) / \log(1/\gamma)$. For
$\gamma = 0.9$, $\epsilon = 10^{-6}$, $d(x_0, x^\star) \sim 1$:
$k \sim 130$ iterations. *That's the entire runtime analysis of
value iteration.*

### What this theorem doesn't say

- **It doesn't tell you what $T$'s fixed point IS.** The theorem
  proves existence and gives a convergent algorithm, but doesn't
  give a closed-form. For the Bellman operator, the fixed point is
  $V^\star$ — a function we have to *compute* via iteration; the
  theorem doesn't hand it to us.
- **The convergence rate is the *worst case*.** Specific starting
  points can converge faster (e.g., starting near the fixed point).
  The bound $\gamma^k$ is tight in the sense that *some* starting
  points achieve it, not that every starting point does.
- **It requires $\gamma < 1$ strictly.** With $\gamma = 1$, the
  theorem fails. The corresponding RL claim — "value iteration
  converges" — fails too in the *undiscounted* setting. The
  average-reward formulation (Ch5 §3.10's brief mention) needs
  different tools.
- **It requires completeness.** Not every metric space is complete.
  In RL we use the space of bounded functions on a finite state
  space, which is complete; fine. In neural-net function-
  approximation settings, the space of representable functions
  isn't a metric space in any standard sense — and the theorem
  doesn't apply. This is the algebraic root of why deep RL doesn't
  inherit the same convergence guarantees (Chapter 17).

### Why this matters for RL

The **Bellman operator** $T^{\star}$ on value functions is defined by

$$
(T^{\star} V)(s) = \max_a \sum_{s'} P(s' \mid s, a) [R(s, a, s') + \gamma V(s')]
$$

**Reading this term-by-term:**

| Symbol | Meaning |
|---|---|
| $T^{\star}$ | The "Bellman optimality operator" — a *function on value functions*. Takes a value function $V$ as input, returns a new value function. |
| $(T^{\star} V)(s)$ | The new value function evaluated at state $s$. |
| $\max_a$ | Take the best action — this is what makes the operator *optimal* (vs. policy-evaluation, which uses $\mathbb{E}_{a \sim \pi}$). |
| $\sum_{s'}$ | Sum over all states the world might transition to. |
| $P(s' \mid s, a)$ | Probability of landing in $s'$ given state $s$ and action $a$. The MDP's dynamics. |
| $R(s, a, s')$ | Reward received on the transition. |
| $\gamma$ | Discount factor; future rewards are worth $\gamma$ times less per step. |
| $V(s')$ | The input value function evaluated at the next state — the *bootstrap* using the current estimate. |

**Plain English.** "The value of state $s$ is the best (max over actions)
expected immediate reward plus discounted next-state value." Iterating
this operator drives $V$ toward the optimal value function $V^{\star}$.

We claim $T^{\star}$ is a contraction with modulus $\gamma$ in the
sup-norm. Proof: for two value functions $V$ and $W$ (using $W$ instead
of $V^{\prime}$ to avoid the prime symbol),

$$
\begin{aligned}
\|T^{\star}V - T^{\star}W\|\_\infty &= \max\_s \big|(T^{\star}V)(s) - (T^{\star}W)(s)\big| \\\\
&= \max\_s \big|\max\_a \mathbb{E}[r + \gamma V(s')] - \max\_a \mathbb{E}[r + \gamma W(s')]\big| \\\\
&\leq \max\_{s, a} \big|\mathbb{E}[r + \gamma V(s')] - \mathbb{E}[r + \gamma W(s')]\big| \\\\
&= \gamma \cdot \max\_{s, a} \big|\mathbb{E}[V(s') - W(s')]\big| \\\\
&\leq \gamma \cdot \|V - W\|\_\infty.
\end{aligned}
$$

**Reading the proof line by line:**

| Step | What happened | Why allowed |
|---|---|---|
| 1 | Expand the sup-norm definition. | $\|\cdot\|_\infty$ is by definition the max over coordinates (here, over states). |
| 2 | Substitute the $T^{\star}$ definition. | Both sides are $\max_a \mathbb{E}[\ldots]$. |
| 3 | Replace $\max_s \lvert\max_a f - \max_a g\rvert$ by $\max_{s,a} \lvert f - g\rvert$. | The lemma $\lvert\max_a f - \max_a g\rvert \leq \max_a \lvert f - g\rvert$. Easy exercise: try $f = (3, 1), g = (2, 4)$. |
| 4 | The rewards $r$ cancel; $\gamma$ factors out. | Linearity of expectation. |
| 5 | The inner $\lvert\mathbb{E}[\cdot]\rvert$ is bounded by $\|\cdot\|_\infty$ of the integrand. | Worst-state bound: $\lvert\mathbb{E}[V - W]\rvert \leq \max_s \lvert V(s) - W(s)\rvert = \|V - W\|_\infty$. |

The first inequality uses $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$
(easy to verify). The last step bounds the expectation by the worst case.

**By Banach, $T^{\star}$ has a unique fixed point $V^{\star}$, and iterating $T^{\star}$
converges to it at rate $\gamma$.** This is the entire convergence theory
of value iteration in one theorem.

The same argument works for the policy-evaluation operator $T^\pi$ (just
replace $\max_a$ with $\mathbb{E}_{a \sim \pi}$). And for $Q$-value
operators on action-value functions. Banach's theorem is the engine that
makes tabular RL work.

### Try it: greedy-policy error amplification

<div id="ch1-policy-error-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/policy_error/widget.js"></script>

A tiny value-function error explodes into a huge policy-quality gap
when γ is high. Slide γ to 0.99 and watch the amplification factor
2γ/(1-γ) climb past 100×. This is why high-γ regimes are so brutal:
"close to V*" isn't close enough for the greedy policy.

### Practical interpretation of $\gamma$

We'll see $\gamma$ in two roles:

- **As a hyperparameter**: discount factor that defines what "long-term
  return" means.
- **As a contraction modulus**: rate at which planning algorithms converge.

These are the *same* number. If $\gamma = 0.9$, value iteration converges
to $\epsilon$ accuracy in roughly $\log(1/\epsilon) / \log(1/\gamma) \approx 22$
iterations for $\epsilon = 0.1$. If $\gamma = 0.99$, it takes ~459
iterations. **Higher $\gamma$ = longer effective horizon = slower convergence.**

This trade-off is unavoidable and explains why most practical RL uses
$\gamma \in [0.9, 0.99]$. Below 0.9 you can't represent long-horizon goals;
above 0.99 the algorithms become impractically slow.

## 3.7 Stochastic approximation

The Banach theorem handles deterministic iterations $x_{k+1} = T(x_k)$.
But in RL we don't apply $T$ exactly — we apply *noisy estimates of* $T$.

### The TD update as stochastic approximation

The TD update (introduced and analyzed by [Sutton 1988]; rigorous
convergence via [Tsitsiklis 1994])

$$
V(s) \leftarrow V(s) + \alpha \cdot \big[r + \gamma V(s') - V(s)\big]
$$

**Reading the formula term-by-term:**

| Symbol | Role | Typical value |
|---|---|---|
| $V(s)$ | Current estimated value of state $s$ (the *prediction*) | learned |
| $\alpha$ | **Step size / learning rate**: how much we move toward the target | $0.1$ in the Simulator |
| $r$ | Immediate reward received on this transition | from `RewardConfig` |
| $\gamma$ | **Discount factor**: how much future rewards count vs. immediate | $0.9$ in the Simulator |
| $V(s')$ | Current estimated value of the *next* state — the **bootstrap** | learned |
| $r + \gamma V(s') - V(s)$ | The **TD error** $\delta$ — how surprised we were by this transition | varies tick-to-tick |
| $\alpha \cdot \delta$ | The actual change to $V(s)$ | small (the noise floor) |

**Plain English.** "Compute the difference between what I *predicted*
the value of $s$ was ($V(s)$) and what one step of experience now tells
me it should be ($r + \gamma V(s')$). Move my prediction a fraction $\alpha$
of the way toward this new estimate."

The bracketed quantity $\delta = r + \gamma V(s') - V(s)$ is so important
it gets a name and a Greek letter. **Almost every algorithm in the
textbook is some variation on TD error**:

- Q-learning replaces $V$ with $Q$ and adds a $\max_a$ inside.
- Actor-critic uses TD error as the *advantage signal* for policy
  gradient (Chapter 13).
- Eligibility traces (Chapter 9) accumulate TD errors over time.
- The deadly triad (Chapter 17) is about when TD updates diverge.

This single equation can be rewritten as

$$
V_{k+1}(s) = (1 - \alpha) V_k(s) + \alpha \cdot \underbrace{[r + \gamma V_k(s')]}_{\text{noisy sample of } T^\pi V_k}
$$

This is a **weighted average** of the old value and a noisy sample of
$T^\pi V$ — and a weighted average is *exactly* the stochastic-approximation
form $X_{k+1} = (1-\alpha) X_k + \alpha Y_k$. With shrinking step sizes
satisfying [Robbins & Monro 1951] conditions and sufficient exploration,
this **converges almost surely to the Bellman fixed point** — the same
one the deterministic operator $T^\pi$ would converge to
[Tsitsiklis 1994].

### The constant $\alpha$ caveat

Practical RL almost always uses **constant $\alpha$** (e.g. $\alpha = 0.1$).
This violates the Robbins-Monro condition $\sum \alpha_k^2 < \infty$.
Consequence: TD doesn't converge to a point — it converges to a
**neighborhood** of $V^{\star}$ whose size is $O(\alpha)$.

This neighborhood is what the Simulator's TD-error-floor calibration
artifacts ([`td_error_floor_*.json`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/focused/baselines))
measure. The "irreducible noise floor" of about 0.006-0.011 wide at
$\alpha = 0.1$ isn't a bug — it's what the math says you get.

To converge to a *point*, you'd need $\alpha_k \to 0$. To track a
non-stationary target, constant $\alpha$ is preferred. The Simulator's
choice reflects "the environment is somewhat non-stationary, so tracking
matters more than asymptotic precision."

## 3.8 Project tie-in

### The Banach theorem in action

[`crates/engine/q_learning/src/value_function.rs:126-146`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/value_function.rs)
implements the TD update inside the `Learner` type. That this update
converges (almost surely, in a neighborhood, under sampling assumptions)
is exactly the stochastic-approximation result of §3.7 composed with
Banach (§3.6): the underlying deterministic operator is a $\gamma$-contraction;
the noisy iteration tracks its fixed point modulo $O(\alpha)$ noise.

### Constant $\alpha = 0.10$ — the trade-off explicit

[`crates/sim/sim_config/src/learning.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/learning.rs)
field `alpha`. From §3.7: this violates Robbins-Monro and lands in an
$O(\sqrt{\alpha})$-radius neighborhood. The choice reflects "the
environment (other agents' policies, meme spread) is non-stationary;
tracking matters more than asymptotic precision."

If you wanted to converge to an actual point, you would schedule
$\alpha_k \to 0$. Per-agent learning-rate memes
([`learning_rate.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/learning_rate.rs))
in principle allow this, but no current meme schedules it — it is left as
future work.

### Contraction $\gamma = 0.90$ — the effective horizon

Same file, field `gamma`. Effective horizon
$1/(1-\gamma) = 10$ *cognition steps* (each step covers ~10 ticks, so
about 100 sim ticks of meaningful foresight). **This number controls
everything** about what counts as "long-term" in the Simulator.

The L-suite (farming scenarios in
[`long_horizon_harvest.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/curricula/long_horizon_harvest.rs))
poses delays of ~500 ticks between Plant and Harvest payoff. At
$\gamma = 0.90$ per step:

$$\gamma^{500} = 0.9^{500} \approx 1.4 \times 10^{-23}$$

The TD update at the Plant decision multiplies any downstream Harvest
reward by this. **The signal is mathematically unbridgeable at this
$\gamma$** — and this directly motivates the methods in Chapters 16 and
19 (hierarchical RL, reward shaping, hindsight, successor features).

### Bootstrap arithmetic preview

The Q-bias bootstrap pathology (the canonical project bug, Chapter 17)
is a worked exercise in §3.7's "constant $\alpha$ neighborhood" point.
Given:

- Reward function ([`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)):
  $R(s, s') = w_\text{alive} - 0.15 \sum_d d^2 - \text{bio costs}$
  with $w_\text{alive} = 1.0$.
- TD update at $\alpha = 0.10$, $\gamma = 0.90$.

Any action that gets committed receives positive reward (from
$w_\text{alive}$ alone) every tick. Iterating the TD update:

$$Q \leftarrow Q + 0.1\big[1.0 + 0.9 \cdot Q - Q\big] = Q + 0.1(1.0 - 0.1 Q).$$

Setting $\Delta Q = 0$ gives the **fixed point**
$Q^{\star} = 1.0 / 0.1 = 10.0$. So *every committed action's Q-value drifts
toward 10*, regardless of whether the action solves the task. That kills
the discriminating signal across actions.

This is the Q-bias bootstrap pathology, analyzed in Chapter 17. The math
is a one-liner; the consequence is that on the L1 scenario over 40 000
ticks across 4 seeds we observed Plant committed 716 times and Consume
committed **0 times** — Plant got locked in first, then bootstrapped its Q
to 10 while Consume stayed at 0. This was fixed by removing the
$w_\text{alive}$ baseline; the reward is now the per-tick drive-delta
$R = \text{cost}(s_\text{prev}) - \text{cost}(s)$.

## 3.9 Exercises

1. **(Constant $\alpha$ doesn't converge to a point.)** Consider the
   stochastic approximation $X_{k+1} = (1 - \alpha) X_k + \alpha Y_k$ where
   $Y_k$ are i.i.d. with mean $\mu$ and variance $\sigma^2$. Compute
   $\mathbb{E}[X_k]$ and $\text{Var}[X_k]$ for the limiting distribution
   as $k \to \infty$. Note the variance is $O(\alpha \sigma^2)$, not 0.

2. **(Contraction proof.)** Let $V_1, V_2$ be value functions on a finite
   state space. Let $T^\pi$ be the Bellman expectation operator for some
   policy $\pi$. Show $\|T^\pi V_1 - T^\pi V_2\| _\infty \leq \gamma \|V_1 - V_2\| _\infty$
   in full detail. Where exactly is each property of the metric / each
   property of the operator used?

3. **(Convergence rate of value iteration.)** Given $\gamma = 0.9$ and a
   value-iteration starting point $V_0$ with $\|V_0 - V^{\star}\| _\infty = 100$,
   how many iterations $k$ are needed to guarantee $\|V_k - V^{\star}\| _\infty \leq 0.01$?

4. **(Effective horizon table.)** Fill in the effective horizon $1/(1-\gamma)$
   and the iterations-to-$0.01$-accuracy for $\gamma \in \{0.5, 0.9, 0.95, 0.99, 0.999\}$.
   What does this suggest about the relationship between $\gamma$ and the
   computational cost of planning?

5. **(Hessian of a quadratic.)** Compute the gradient and Hessian of
   $f(v) = \tfrac{1}{2} v^{\top} A v - b^{\top} v + c$ where $A$ is
   symmetric. Solve $\nabla f(v) = 0$ for the minimizer. What is the
   one-step Newton update from any starting point? What does this tell
   you about Newton's method on quadratic objectives?

6. **(Condition number and convergence.)** Gradient descent on the
   quadratic $f(v) = \tfrac{1}{2} v^{\top} A v$ with step size
   $\eta = 1 / \lambda_{\max}(A)$ has error multiplied by
   $(\kappa - 1)/(\kappa + 1)$ per step where $\kappa = \lambda_{\max} / \lambda_{\min}$.
   Compare $\kappa = 1$ (sphere) with $\kappa = 100$ (long thin valley)
   and $\kappa = 10^4$. How many steps to halve the error in each case?

7. **(Q-bias bootstrap.)** Verify the fixed-point arithmetic in §3.8:
   given $\alpha = 0.1$, $\gamma = 0.9$, and reward $r = 1$ on every tick
   (the $w_\text{alive}$ contribution), the TD recurrence
   $Q \leftarrow Q + 0.1(1.0 + 0.9 Q - Q)$ has fixed point $Q^{\star} = 10.0$.
   Show convergence is geometric and compute the iterate at $k = 50$
   starting from $Q_0 = 0$.

## 3.10 References and further reading

Full bibliographic entries in [`bibliography.md`](bibliography.md):

- [Banach 1922] — fixed-point theorem (§3.6).
- [Robbins & Monro 1951] — stochastic approximation foundation (§3.7).
- [Sutton 1988] — TD learning origin (§3.7).
- [Szepesvári 2010] — proof structure for Banach (§3.6).
- [Tsitsiklis 1994] — convergence of asynchronous Q-learning (§3.7).
- [Bertsekas 2012] — operator-theoretic DP (alternative treatment of §3.6).

| Source | What to read | Why |
|---|---|---|
| [Boyd & Vandenberghe, *Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) | Ch. 1-3 | The cleanest intro to optimization. Free PDF. |
| [Goodfellow, Bengio & Courville 2016] | Ch. 4 | Numerical computation, optimization. |
| [S&B 2018] | Appendix A | Math review S&B assumes. |
| [Bertsekas 2012] | Appendix on Banach's theorem | The rigorous version. |
| [Szepesvári 2010] | Appendices on stochastic approximation | The careful version of §3.7. |

---

**Next:** [Chapter 4 — The Reinforcement Learning Problem](04_the_rl_problem.md) — we step out of pure math into the agent-environment loop. The math we just built will return constantly.

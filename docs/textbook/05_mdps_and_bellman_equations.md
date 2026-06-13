# Chapter 5 — Markov Decision Processes and the Bellman Equations

> **Prerequisites:** [Chapters 1-3](01_linear_algebra.md) (probability,
> contraction mappings) and [Chapter 4](04_the_rl_problem.md) (the
> agent-environment loop).

> **Learning objectives:**
> 1. Write down an MDP formally and identify its components.
> 2. Compute $V^\pi(s)$ and $Q^\pi(s, a)$ for small MDPs.
> 3. Derive the Bellman expectation and optimality equations.
> 4. Prove that $V^{\star}$ and $Q^{\star}$ exist and are unique (using Chapter 1's
>    Banach theorem).
> 5. Recognize POMDPs and partial observability.

> **Citations:** the formal MDP definitions follow [Puterman 2005, Ch. 2]
> and [S&B 2018, Ch. 3]. The contraction proof for $T^{\star}$ follows
> [S&B 2018, Sec. 4.3]; the rigorous operator-theoretic treatment is in
> [Bertsekas 2012, Ch. 1]. POMDP framing follows [Kaelbling, Littman &
> Cassandra 1998].

This chapter formalizes the agent-environment loop into the **Markov
Decision Process** framework. Almost every algorithm you'll meet — DP,
MC, TD, Q-learning, DQN, policy gradient — is solving (or approximately
solving) an MDP.

## 3.1 The MDP formalism

> **Definition** (after [Puterman 2005, Ch. 2]; [S&B 2018, Sec. 3.1])**.** A
> **(finite) Markov Decision Process** is a tuple
> $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where:
> - $\mathcal{S}$ is a finite set of **states**;
> - $\mathcal{A}$ is a finite set of **actions** (possibly state-dependent
>   $\mathcal{A}(s)$, but we'll write $\mathcal{A}$ for simplicity);
> - $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ is the
>   **transition probability** function — $P(s' \mid s, a)$ is the
>   probability of landing in $s'$ when taking $a$ in $s$. (Sums to 1 over $s'$ for each
>   $(s, a)$.)
> - $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$ is
>   the **reward function**, often written as $\mathbb{E}[r_t \mid s_t = s, a_t = a, s_{t+1} = s']$
>   if rewards are random.
> - $\gamma \in [0, 1)$ is the **discount factor**.

Infinite or continuous state/action spaces add measure-theoretic
complications but don't change the core ideas. We'll stick to finite
MDPs for clarity, generalizing where it matters.

### What the Markov property buys us

The whole point: $P(s' \mid s, a)$ depends *only* on the current state and
action — not on any prior history. This is what makes the value of a
state a well-defined function: if the state captures everything relevant,
then "value of being in state $s$" doesn't depend on how we got there.

When the Markov property fails, we get a **POMDP** (partially-observable
MDP, [Section 3.7](#37-partial-observability-pomdps)).

### Example: a 4×4 gridworld

The textbook MDP. $\mathcal{S} = 16$ cells of a 4×4 grid, $\mathcal{A} = \{N, S, E, W\}$.
$P(s' \mid s, a) = 1$ for the cell in direction $a$ (or staying if blocked
by wall). $R(s, a, s') = -1$ for every step except at the terminal cell.
$\gamma$ doesn't really matter for short episodes. The optimal policy
points toward the nearest terminal cell.

This is the smallest non-trivial MDP. Every RL textbook uses some variant
because it's small enough to write all value functions out explicitly and
verify by hand.

### Try it: $R(s)$ vs $R(s, a)$ vs $R(s, a, s')$

The MDP definition writes the reward as $R(s, a, s')$ — a function of the
whole transition — but two other formulations show up in the literature:
$R(s)$ (reward of *arriving* in $s$) and $R(s, a)$ (reward of *taking* $a$
in $s$, regardless of outcome). The Simulator uses $R(s)$ specifically.

<div id="ch3-reward-shape-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/reward_shape/widget.js"></script>

Same 2×2 gridworld; three reward shapes. The optimal policies don't
always agree — the formulation is *part of the MDP*, not a notational
convenience. ($R(s, a, s')$ in particular can penalize *attempting* a
step into the pit even when the agent bumps a wall and never actually
visits it — a distinction $R(s)$ and $R(s, a)$ can't make.)

### Example: the Simulator's hunger MDP

The closest analogue in our project: an agent in an empty world with one
food container.

- $\mathcal{S}$ — the agent's `Observation` (technically POMDP, but treat
  as MDP for now). Includes drives (hunger, thirst, ...), perception of
  the food, position, etc.
- $\mathcal{A}$ — the 16 `WellKnownTemplate` action types, with parameters
  for some.
- $P$ — the Bevy world dynamics. Hunger increases by metabolism rate per
  tick; perception updates as agent moves; etc.
- $R$ — `PrimaryReward` = $w_{\text{alive}} - \text{drive\_cost} - \text{bio\_cost}$.
- $\gamma = 0.9$.

This MDP is the *abstract* model. The actual sim doesn't pre-compute $P$
— it samples transitions by running the Bevy systems. That's what makes
it a *model-free* setting from the agent's perspective.

## 3.2 Policies and trajectories revisited

Recall a **policy** is a (possibly stochastic) map from states to actions.
For a finite MDP, the space of policies is finite: there are at most
$|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies.

Given an MDP and a policy $\pi$, the trajectory becomes a random sequence
generated by:

$$
s_0 \sim \mu_0, \quad a_t \sim \pi(\cdot \mid s_t), \quad s_{t+1} \sim P(\cdot \mid s_t, a_t)
$$

where $\mu_0$ is the initial state distribution.

The **return** from state $s$ under policy $\pi$ is a random variable:

$$
G_0 = \sum_{t=0}^\infty \gamma^t r_t
$$

with randomness from $\pi$'s action draws and $P$'s transition draws (and
$R$ if rewards are stochastic).

## 3.3 The state-value function $V^\pi$

> **Definition.** The **state-value function** of policy $\pi$ is
>
> $$
> V^\pi(s) = \mathbb{E}\_\pi\big[G\_t \mid s\_t = s\big]
> $$
>
> i.e. the expected return when starting in $s$ and following $\pi$.

The expectation is over both $\pi$'s and $P$'s randomness. $V^\pi$ is a
function $\mathcal{S} \to \mathbb{R}$. In a finite MDP it's just $|\mathcal{S}|$
numbers.

### Why this object matters

Without $V^\pi$, every "what should I do?" question is a search
problem over infinite-horizon trajectories. With $V^\pi$, the same
question becomes a one-step lookahead: "what's $r + \gamma V^\pi(s')$
for each $a$?" The value function is what lets RL *not* be tree
search.

Every algorithm in the rest of the book either computes $V^\pi$
directly (DP — Chapter 6), estimates it from samples (MC, TD —
Chapters 7–8), or approximates it with a parametric function (Chapters
10–11). The structure those algorithms exploit — the *fact that
$V^\pi$ has structure at all* — is what we develop next. Pay
attention to the Bellman equation; it's the single most important
formula in the book.

### Sanity checks

- $V^\pi(s) = 0$ for terminal states (no future rewards).
- If $\pi$ is deterministic and the MDP deterministic, $V^\pi(s)$ is just
  the discounted sum of rewards on the unique trajectory.
- $V^\pi$ depends on the entire infinite future, not just the next reward.

### The recursive structure: Bellman expectation equation

The naive way to compute $V^\pi(s)$ is to enumerate every trajectory
from $s$, weight by its probability, and sum. That's infinite work
in any non-trivial MDP. The Bellman equation makes it finite by
exploiting one fact: the return decomposes recursively.

**Step 1 — the return is itself recursive.** From the definition
$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$,
factor out one step:

$$
G_t = r_t + \gamma \big( r_{t+1} + \gamma r_{t+2} + \cdots \big) = r_t + \gamma G_{t+1}.
$$

That's just algebra — splitting the head off the geometric tail.

**Step 2 — expectation under $\pi$ inherits the recursion.** Take
the conditional expectation given $s_t = s$:

$$
V^\pi(s) = \mathbb{E}\_\pi[G\_t \mid s\_t = s] \stackrel{(1)}{=} \mathbb{E}\_\pi[r\_t + \gamma G\_{t+1} \mid s\_t = s] \stackrel{(2)}{=} \mathbb{E}\_\pi[r\_t \mid s\_t = s] + \gamma \mathbb{E}\_\pi[G\_{t+1} \mid s\_t = s].
$$

Step (1) substitutes the recursive form of $G_t$ inside the
expectation. Step (2) uses linearity of expectation
($\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$).

**Step 3 — expand by conditioning on $(a_t, s_{t+1})$.** The Markov
property (§3.1) says only the current state matters, so
$\mathbb{E} _\pi[G _{t+1} \mid s _t = s, s _{t+1} = s'] = V^\pi(s')$.
Use the law of total expectation (Chapter 2) to integrate out the
action and next state:

$$
V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^\pi(s') \Big].
$$

Here $\pi(a \mid s)$ is the probability we picked action $a$,
$P(s' \mid s, a)$ is the probability we landed in $s'$, $R$ is the
reward we got, and $V^\pi(s')$ is the expected return *from $s'$
onwards* — which appears on both sides of the equation. **That
self-reference is what makes this a *Bellman* equation.** It's a
linear system: $|\mathcal{S}|$ equations in $|\mathcal{S}|$ unknowns,
one per state.

### Two ways to read the Bellman equation

**1. Mechanically: as a recursive formula.** $V^\pi(s)$ equals
average-immediate-reward plus $\gamma$ times average-next-state-value,
where averages are over the action $\pi$ would pick and the next
state $P$ would deliver. To use it: pick a state $s$, plug in the
RHS with current estimates of $V^\pi(s')$, get a new $V^\pi(s)$.
Iterating this is **policy evaluation** (Chapter 6 §4.2).

**2. Compositionally: as a fixed-point identity on a linear
operator.** Define the operator $T^\pi$ that maps a value function
$V$ to a new value function $T^\pi V$ via the right-hand side.
$V^\pi$ is then the *fixed point* of $T^\pi$ — the function such
that $T^\pi V^\pi = V^\pi$. This reframing is what unlocks Banach
(below) and the geometric-convergence story for all of DP.

The two readings are equivalent. Reading 1 is how you remember the
algorithm; reading 2 is how you remember the convergence theorem.

### Operator notation

Define the **Bellman expectation operator** $T^\pi: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$
by

$$
(T^\pi V)(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \big[ R(s, a, s') + \gamma V(s') \big].
$$

Then the Bellman expectation equation is exactly

$$
V^\pi = T^\pi V^\pi,
$$

i.e. **$V^\pi$ is a fixed point of $T^\pi$**. The right way to think
about $T^\pi$: it takes a *guess* at $V^\pi$ and improves it by one
step of look-ahead. The fixed point is where look-ahead stops
changing the guess — that must be $V^\pi$.

### $T^\pi$ is a $\gamma$-contraction

**Claim.** $\|T^\pi V_1 - T^\pi V_2\| _\infty \leq \gamma \|V_1 - V_2\| _\infty$.

**What this means in plain English.** No matter how far apart two
guesses $V_1, V_2$ are, after applying $T^\pi$ to both, the new
guesses are at most $\gamma$ times as far apart. The operator
*shrinks distances by a factor of $\gamma$ per application*. After
$k$ applications, distances are at most $\gamma^k$ of the original
— *geometric* convergence. With $\gamma = 0.99$, one application
shrinks error by 1%; with $\gamma = 0.9$, by 10%; with $\gamma = 0.5$,
by half. This is why the discount factor controls how fast
algorithms converge — it *literally is* the convergence rate.

**Proof, step by step.** Pick any state $s$ and compare the two
operator outputs:

$$
\big|(T^\pi V\_1)(s) - (T^\pi V\_2)(s)\big| \stackrel{(1)}{=} \Big|\sum\_{a, s'} \pi(a \mid s) P(s' \mid s, a) \gamma \big(V\_1(s') - V\_2(s')\big)\Big|.
$$

Step (1) expands both sides using the operator definition. The
$R(s,a,s')$ terms cancel (they don't depend on $V$), leaving the
$\gamma V(s')$ pieces. Pull the $\gamma$ outside.

$$
\stackrel{(2)}{\leq} \gamma \sum\_{a, s'} \pi(a \mid s) P(s' \mid s, a) \big|V\_1(s') - V\_2(s')\big|.
$$

Step (2) is the triangle inequality: $|\sum c_i x_i| \leq \sum |c_i| \, |x_i|$.
Since $\pi$ and $P$ are probabilities (non-negative), the
absolute value on $c_i$ disappears.

$$
\stackrel{(3)}{\leq} \gamma \sum\_{a, s'} \pi(a \mid s) P(s' \mid s, a) \cdot \|V\_1 - V\_2\|\_\infty.
$$

Step (3) replaces each $|V_1(s') - V_2(s')|$ with its largest
possible value, which is the sup-norm $\|V_1 - V_2\| _\infty$. The
inequality only goes one way (the bound is loose for states where
the per-state gap is below the max).

$$
\stackrel{(4)}{=} \gamma \|V\_1 - V\_2\|\_\infty.
$$

Step (4) pulls $\|V_1 - V_2\| _\infty$ out of the sum (it doesn't
depend on the summed variables), leaving $\sum_a \pi(a|s) \sum _{s'} P(s'|s,a) = 1$ —
sums to one because $\pi(\cdot|s)$ and $P(\cdot|s,a)$ are
probability distributions.

The bound holds for every $s$. Taking $\max_s$ on both sides gives
$\|T^\pi V_1 - T^\pi V_2\| _\infty \leq \gamma \|V_1 - V_2\| _\infty$. ☐

**Where the proof's key tricks come from.**
- **Triangle inequality + probability** is the workhorse: weighted
  averages of differences are bounded by weighted averages of
  *magnitudes* of differences, which are in turn bounded by the
  global maximum.
- **The $R$ terms cancelling** is why the contraction constant is
  $\gamma$, not $\gamma$ plus some reward-dependent quantity. It's
  the geometry of $\gamma$ multiplying *only* the value piece in
  the recursion.
- **Sup-norm specifically** — would the proof work with $\ell_2$?
  Step (3) is where it'd break. The sup-norm gives one number that
  bounds every state's gap; in $\ell_2$ you'd need a Jensen step
  that costs extra. This is why almost every Bellman convergence
  result in this book is stated in $\ell_\infty$. See Chapter 1
  §1.6 for the why-each-norm-is-for table.

**Corollary (Banach, Chapter 1, §1.5).** A $\gamma$-contraction on
a complete metric space has a unique fixed point, reached by
iteration at geometric rate $\gamma$:

- **$V^\pi$ exists** (the operator has a fixed point at all).
- **$V^\pi$ is unique** (no other $V$ satisfies $V = T^\pi V$).
- **Iteration converges:** $V_{k+1} = T^\pi V_k$ satisfies
  $\|V_k - V^\pi\| _\infty \leq \gamma^k \|V_0 - V^\pi\| _\infty$.

This is the bedrock of **policy evaluation** (Chapter 6 §4.2 — the
iterative algorithm) and of **value iteration** (§4.5 — the
analogue with $T^\star$ from §3.5 below). Without the contraction
property, neither algorithm would terminate; with it, both converge
to machine epsilon in $O(\log(1/\epsilon) / \log(1/\gamma))$
sweeps. That bound shows up in every convergence argument in
Chapters 6–11.

### What the Bellman expectation equation does NOT say

Three traps worth heading off:

- **It doesn't define what $V^\pi$ is operationally.** $V^\pi$ is
  defined by the expectation in §3.3. The Bellman equation is a
  *property* $V^\pi$ satisfies, not its definition. The distinction
  matters: you can have a function $V$ that satisfies the Bellman
  equation but isn't the value function of any policy — except
  Banach guarantees uniqueness, so this can't happen for $T^\pi$.
  (For $T^\star$ in §3.5, it can — see optimal-policy theorem in
  §3.8.)
- **The contraction proof above is *not* the contraction proof for
  $T^\star$.** The $\max$ inside the optimality operator needs an
  extra lemma (the "max-trick", §3.5). Don't reuse this proof
  blindly.
- **"Convergence at rate $\gamma$" doesn't mean the algorithm
  *terminates*.** It means error decays geometrically. Termination
  is a thresholding decision: stop when
  $\|V_{k+1} - V_k\|_\infty < \epsilon$ or after a fixed $K$
  iterations. Chapter 6 §4.5 discusses the practical stopping rules.

## 3.4 The action-value function $Q^\pi$

> **Definition.** The **action-value function** of policy $\pi$ is
>
> $$
> Q^\pi(s, a) = \mathbb{E}_\pi\big[G_t \mid s_t = s, a_t = a\big]
> $$
>
> — the expected return starting in $s$, taking action $a$, then following $\pi$.

In a finite MDP, $Q^\pi$ is $|\mathcal{S}| \times |\mathcal{A}|$ numbers.

### Relating $V^\pi$ and $Q^\pi$

$$
V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)
$$

(average over the policy's action distribution).

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \big[R(s, a, s') + \gamma V^\pi(s')\big]
$$

(immediate reward + discounted future value).

Substituting the first into the second:

$$
Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \Big[R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\Big]
$$

This is the **Bellman expectation equation for $Q^\pi$**. Same contraction
structure: $Q^\pi$ is the unique fixed point of an operator on
$\mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$.

### Try it: the Bellman expectation backup, term by term

The Bellman expectation equation
$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)[R + \gamma V(s')]$
is a *triple sum*: one over actions, one over successor states, one
hidden inside $R + \gamma V$. Watching the terms accumulate one at a
time makes the structure clear.

<div id="ch3-bellman-propagator-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/bellman_propagator/widget.js"></script>

Use prev/next/play to step through the backup. For each action $a$, the
widget walks every successor: pulls $R(s, a, s')$, pulls $\gamma V(s')$,
combines them, weights by $P(s' \mid s, a)$, and accumulates into
$Q(s, a)$. Then it weights $Q(s, a)$ by $\pi(a \mid s)$ and accumulates
into $V^\pi(s)$. The dashed rules mark where each bar will end up; the
solid bars grow toward them.

### Why $Q$ is more useful than $V$ for control

To pick the best action in state $s$:

- **With $V$:** need to compute $\sum_{s'} P(s' \mid s, a)[R + \gamma V(s')]$
  for each $a$ and argmax. **Requires knowledge of $P$.**
- **With $Q$:** just $\arg\max_a Q(s, a)$. **No model needed.**

This is why **Q-learning** (Chapter 8) doesn't need a transition model.
The Q-function absorbs the model into the value estimate.

## 3.5 The optimal value functions $V^{\star}$ and $Q^{\star}$

### Why we need a "$\star$" version at all

§3.3 gave us $V^\pi$ — the value of *a* policy. But the agent's
problem isn't "what's the value of this policy I happen to have?"
It's "**what policy should I follow?**" That requires comparing
policies — which means we need a notion of "the best one" and a
*method* to characterize / compute it. $V^\star$ and $Q^\star$ are
that notion. The Bellman optimality equation is the method.

Everything in Chapters 6–11 either *computes* something close to
$V^\star$ / $Q^\star$ (DP, Q-learning, DQN) or *learns to act
greedily without computing it* (policy gradient, Chapters 12–13).
This section's contraction proof is the same kind of bedrock-
foundation as §3.3's; it's what guarantees value iteration
converges.

### Definitions

$$
V^{\star}(s) = \max_\pi V^\pi(s), \qquad Q^{\star}(s, a) = \max_\pi Q^\pi(s, a).
$$

In English: $V^\star(s)$ is the best expected return achievable
from state $s$ under any policy. $Q^\star(s, a)$ is the best return
achievable starting with action $a$ in state $s$, then playing
optimally thereafter.

> **Fact (Optimal-policy theorem; proof in §3.8).** For any finite
> MDP there exists a *deterministic* policy $\pi^\star$ such that
> $V^{\pi^\star}(s) = V^\star(s)$ for all $s$.

This is a non-trivial result — randomized policies don't help in
fully-observed finite MDPs. (They *do* help in POMDPs, §3.7, and in
multi-agent games — Chapter 14's bandit setting hints at why.) The
practical consequence: when we hunt for $\pi^\star$, we can
restrict our search to deterministic policies. That's
$|\mathcal{A}|^{|\mathcal{S}|}$ candidates — astronomical, but
*countable*, and value iteration / policy iteration sidestep
enumeration entirely.

The optimal policy is **greedy with respect to $V^\star$ or
$Q^\star$**:

$$
\pi^\star(s) = \arg\max_a \sum_{s'} P(s' \mid s, a)\big[R + \gamma V^\star(s')\big] = \arg\max_a Q^\star(s, a).
$$

The $Q^\star$ form is the one to remember: **once you know
$Q^\star$, the optimal action at every state is whichever $a$ has
the highest $Q^\star(s, a)$ — no model needed.** That's why
Q-learning (Chapter 8) targets $Q$ rather than $V$: a learned
$Q^\star$ is *directly usable* at decision time, where a learned
$V^\star$ still requires $P$ to be greedy.

### The Bellman optimality equation

For $V^\star$:

$$
V^\star(s) = \max_a \sum_{s'} P(s' \mid s, a) \big[R(s, a, s') + \gamma V^\star(s')\big].
$$

For $Q^\star$:

$$
Q^\star(s, a) = \sum_{s'} P(s' \mid s, a) \Big[R(s, a, s') + \gamma \max_{a'} Q^\star(s', a')\Big].
$$

**The structural difference from the expectation equation.** §3.3
had $\sum_a \pi(a \mid s)$ — a policy-weighted average. §3.5 has
$\max_a$ — a hard maximum. *That's it.* Everything else (the
discount, the reward, the next-state probability) is identical.
The $\max$ turns a fixed-policy fixed-point equation into a
"best-action" fixed-point equation. The cost of that single change:
the operator is no longer linear (the $\max$ is piecewise linear,
not linear), so we can't just invert a matrix to find $V^\star$.
But it's still a $\gamma$-contraction (next subsection), so
iteration still works.

**Intuition for the $\max$.** $V^\star(s)$ asks: of all the actions
I could take here, what's the best one's expected return *assuming
I keep playing optimally afterwards*? The "assuming I keep playing
optimally" is the self-referential bit — $V^\star(s')$ appears
inside the $\max$. The equation is well-defined because the
self-reference is *contractive*: each unrolling shrinks the unknown
by $\gamma$ until it's negligible.

### Operator form: $T^\star$ is a $\gamma$-contraction

(Proof following [S&B 2018, Sec. 4.3].) Define the **Bellman
optimality operator** $T^\star$ by

$$
(T^\star V)(s) = \max_a \sum_{s'} P(s' \mid s, a)\big[R(s, a, s') + \gamma V(s')\big].
$$

**Claim.** $T^\star$ is a $\gamma$-contraction in sup-norm.

**Same plain-English meaning as §3.3:** apply $T^\star$ to two
guesses, and they get closer by a factor of $\gamma$. Iterate, and
the geometric shrinkage drives any starting guess to $V^\star$.

**Proof, step by step.** This proof is *almost* the §3.3 proof —
the only new wrinkle is the $\max$ inside the operator. Bound that
$\max$ with a small lemma (next subsection), and everything else
falls out the same way.

$$
\big|(T^\star V_1)(s) - (T^\star V_2)(s)\big| \stackrel{(1)}{=} \big|\max_a E_a^1 - \max_a E_a^2\big|
$$

where I've abbreviated $E_a^i := \sum_{s'} P(s' \mid s, a)[R + \gamma V_i(s')]$
for brevity. The two operator outputs differ in their inner
expectations only by the $V_i(s')$ pieces.

$$
\stackrel{(2)}{\leq} \max_a \big|E_a^1 - E_a^2\big|.
$$

Step (2) is the **max-trick lemma** (proved below):
$|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$. This is
the only step that differs from §3.3 — the $\max$ is *Lipschitz*
with constant 1, so it can be pulled inside the absolute value at no
cost.

$$
\stackrel{(3)}{=} \max_a \Big| \gamma \sum_{s'} P(s' \mid s, a)\big(V_1(s') - V_2(s')\big) \Big| \stackrel{(4)}{\leq} \gamma \|V_1 - V_2\|_\infty.
$$

Step (3) expands the $E_a^i$ definition; the $R$ terms cancel
(they don't depend on $V$). Step (4) is the same triangle-inequality
+ probability-distribution-sums-to-one trick from §3.3 (steps
(2)-(4) of that proof, applied inside the $\max_a$ — and since
the bound $\gamma \|V_1 - V_2\|_\infty$ doesn't depend on $a$, the
$\max_a$ is trivial).

Taking $\max_s$ on both sides gives
$\|T^\star V_1 - T^\star V_2\| _\infty \leq \gamma \|V_1 - V_2\| _\infty$. ☐

**By Banach** (Chapter 1 §1.5): $V^\star$ exists, is unique, and
value iteration $V_{k+1} = T^\star V_k$ converges geometrically at
rate $\gamma$. **This is the entire theory of value iteration.**
Chapter 6 §4.5 is the algorithm + the convergence rate
$O(\log(1/\epsilon) / \log(1/\gamma))$ sweeps to reach $\epsilon$
accuracy. Chapter 8 (TD learning) builds the same convergence on a
stochastic approximation of $T^\star$ when $P$ is unknown.

### Why the max-trick lemma works

The claim: $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$.

Without loss of generality, assume $\max_a f(a) \geq \max_a g(a)$
(if not, swap $f \leftrightarrow g$; the absolute value is
symmetric). Let $a^\star = \arg\max_a f(a)$. Then

$$
\max_a f(a) - \max_a g(a) \stackrel{(1)}{=} f(a^\star) - \max_a g(a) \stackrel{(2)}{\leq} f(a^\star) - g(a^\star) \stackrel{(3)}{\leq} \max_a |f(a) - g(a)|.
$$

Step (1) just substitutes the $\arg\max$ definition. Step (2) uses
$g(a^\star) \leq \max_a g(a)$ — i.e. $-\max g \leq -g(a^\star)$ —
to weaken the bound. Step (3) bounds $f(a^\star) - g(a^\star)$ by
the maximum-over-$a$ of $|f(a) - g(a)|$ (the single point $a^\star$
is one of the $a$'s being maxed).

By symmetry the same bound holds with $f, g$ swapped, so
$|\max f - \max g| \leq \max |f - g|$. ☐

**Why this matters beyond Bellman.** Any time you have a sup-norm
contraction argument and an operator that takes a max somewhere,
this is the lemma you reach for. It shows up in MDP convergence
proofs (here), in $\ell_\infty$-Lipschitz neural net bounds, and
in the analysis of game-theoretic equilibria. Worth remembering as
"max is 1-Lipschitz in sup-norm."

### What's *not* immediate from the contraction

Three loose ends Chapters 6–11 tighten:

- **$V^\star$ being the value of an *actual* policy.** Banach gives
  uniqueness of the fixed point, but doesn't say "this fixed point
  is the value function of some policy." That's the optimal-policy
  theorem (§3.8), which uses the explicit construction
  $\pi^\star(s) = \arg\max_a Q^\star(s,a)$.
- **Stopping rule for value iteration.** The contraction tells you
  errors shrink geometrically but not when "good enough" is reached.
  Chapter 6 §4.5 ships the practical $\|V_{k+1} - V_k\|_\infty < \epsilon$
  threshold + the $2\gamma\epsilon/(1-\gamma)$ error-to-policy-
  suboptimality conversion.
- **Stochastic approximation when $P$ is unknown.** All of §3.5
  assumes you know $P$. Real RL doesn't. Chapter 8 develops the
  TD-learning machinery that approximates $T^\star$ via sampled
  transitions; the contraction structure carries over but the
  convergence proof gets harder (Robbins-Monro conditions, etc.).

## 3.6 Summary of Bellman equations

The full set, side by side:

| | Expectation (for fixed $\pi$) | Optimality |
|---|---|---|
| $V$ | $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P [R + \gamma V^\pi(s')]$ | $V^{\star}(s) = \max_a \sum_{s'} P [R + \gamma V^{\star}(s')]$ |
| $Q$ | $Q^\pi(s, a) = \sum_{s'} P [R + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]$ | $Q^{\star}(s, a) = \sum_{s'} P [R + \gamma \max_{a'} Q^{\star}(s', a')]$ |

The **optimal policy from $Q^{\star}$**: $\pi^{\star}(s) = \arg\max_a Q^{\star}(s, a)$. No
model required at decision time.

Each version's operator ($T^\pi$, $T^{\star}$, the $Q$ versions) is a
$\gamma$-contraction. By Banach: unique fixed point, geometric
convergence under iteration. **This single fact is the foundation of all
of dynamic programming.**

### Try it: Bellman as a linear system

For a fixed policy $\pi$ on a finite MDP, the Bellman expectation
equation $V^\pi = r^\pi + \gamma P^\pi V^\pi$ is just a linear system:

$$
V^\pi = (I - \gamma P^\pi)^{-1} r^\pi.
$$

This is exercise 7 below. The widget lets you watch the closed-form
inverse and the iterative $T^\pi V_k$ sequence converge to *the same*
answer.

<div id="ch3-bellman-linear-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/bellman_linear/widget.js"></script>

Slide $n$, $\gamma$, the transition weights, and the terminal reward.
The matrix on the left is $P^\pi$; the matrix beside it is
$(I - \gamma P^\pi)^{-1}$. The right-most column of small mini-vectors
is the iterative sequence $V_1, V_2, V_5, V_{20}$ together with the
closed-form $V^\pi$ — they all match (to within $\|V_{20} - V^\pi\|_\infty$
shown in the readout). DP and TD are both approximating *this* inverse.

## 3.7 Partial observability — POMDPs

In many real settings, the agent doesn't see $s$ — only some **observation**
$o$ that's a (possibly noisy) function of $s$.

> **Definition.** A **Partially Observable MDP** (POMDP) is a tuple
> $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \mathcal{O}, O)$ where $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$
> is an MDP and $O(o \mid s, a)$ is an **observation distribution**.

Now the agent sees $o_t$ instead of $s_t$. The optimal policy is generally
a function of the agent's **belief state** — a probability distribution
over $\mathcal{S}$ given the history of observations and actions:

$$
b_t(s) = P(s_t = s \mid o_0, a_0, o_1, a_1, \ldots, o_t)
$$

POMDPs can in principle be reduced to MDPs over belief states (the
**belief MDP**), but the belief-state space is a continuous probability
simplex, making this typically intractable.

**Most practical RL silently treats POMDPs as MDPs** by either:

1. **Stacking observations** to approximate Markov-ness (e.g. DQN on Atari
   stacks 4 frames).
2. **Using a recurrent agent** (LSTM, transformer) that maintains an
   implicit belief through hidden state.
3. **Hoping the observation is rich enough** to be near-Markov.

The Simulator does option 3. The observation is a 251-dim vector
including episodic memory, which is *meant* to be a sufficient summary of
recent history. Whether it actually is depends on the scenario. For
short-term tasks (next 10 ticks) probably yes; for long-term tasks
(remembering food perceived 500 ticks ago) probably no.

### POMDP papers and tools

- **Kaelbling, Littman, Cassandra 1998**, *Planning and Acting in Partially
  Observable Stochastic Domains* — the foundational POMDP paper.
  [PDF](https://www.cs.cmu.edu/~jdc/sma/litreview/kaelbling-pomdp.pdf)
- **Bakker 2001**, *Reinforcement Learning with Long Short-Term Memory* —
  the canonical "use RNN for partial observability" paper.
- **Wayne et al. 2018**, MERLIN — modern memory-augmented agent (covered
  in Chapter 10).

### Try it: belief updates on the Tiger POMDP

<div id="ch3-pomdp-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/pomdp/widget.js"></script>

Slide the observation accuracy $\alpha$ (probability a "listen" hears the
true side correctly) and click **listen** to apply one Bayes update to
the belief $b = P(\text{tiger left})$. The top bar shows the current
belief; the bottom strip plots the belief trajectory and the sequence of
observations (HL / HR glyphs). At $\alpha = 0.85$ a single listen barely
moves $b$ from 0.5; three or four correlated listens drive it past 0.9.
At $\alpha = 0.55$ (near no signal), the belief drifts forever without
sharpening — that's the regime where "stacking observations" stops
helping. Press **open ←** or **open →** to act: a correct guess reveals
treasure (and the trial resets); a wrong guess wakes the tiger.

## 3.8 Existence and uniqueness of $V^{\star}$, $Q^{\star}$, $\pi^{\star}$

A full theorem for the record:

> **Theorem (Fundamental Theorem of MDPs).** For any finite MDP with
> $\gamma \in [0, 1)$:
> 1. $V^{\star}$ exists, is unique, and is the unique fixed point of $T^{\star}$.
> 2. $Q^{\star}$ exists, is unique, and is the unique fixed point of the analogous
>    optimality operator for $Q$.
> 3. There exists a **stationary deterministic policy** $\pi^{\star}$ that is
>    optimal: $V^{\pi^{\star}}(s) = V^{\star}(s)$ for all $s$.
> 4. The greedy policy w.r.t. $V^{\star}$ (or $Q^{\star}$) is optimal:
>    $\pi^{\star}(s) = \arg\max_a Q^{\star}(s, a)$.

The first two are Banach's theorem. The third and fourth are slightly
more delicate — they require showing that the greedy policy w.r.t. $V^{\star}$
actually *achieves* the value $V^{\star}$. (This uses a "policy improvement"
argument we'll see in Chapter 6 when we cover policy iteration.)

The upshot: **you never need stochastic policies for finite MDPs**.
Stochasticity becomes useful only in function-approximation settings
(where the policy gradient methods of Chapter 12 use them for
optimization tractability) or in adversarial / multi-agent settings (where
mixing is essential).

### Try it: try to beat a deterministic policy with a stochastic one

A direct demonstration of the theorem. Pick *any* stochastic policy
$\pi$ on a small gridworld; the deterministic policy $\pi'(s) = \arg\max_a Q^\pi(s, a)$
satisfies $V^{\pi'}(s) \geq V^\pi(s)$ for every state. The "improvement
gap" $V^{\pi'} - V^\pi$ is the *cost* of randomising — and it's
non-negative everywhere.

<div id="ch3-stoch-vs-det-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/stochastic_vs_det/widget.js"></script>

Pick a preset (uniform, east-biased, into-pit), then slide the softmax
temperature $\tau$ to interpolate between "near-deterministic" ($\tau \to 0$)
and "near-uniform" ($\tau \to \infty$). The left heatmap is $V^\pi$ for the
stochastic policy; the right is $V^{\pi'}$ for the deterministic greedy
policy derived from $Q^\pi$. The right side is *never* worse, pointwise,
and the readout reports the min/mean/max of the per-state improvement.

## 3.9 Project tie-in

### Where the Simulator implicitly defines an MDP

The MDP for one agent in the Simulator:

- $\mathcal{S}$ = the space of `Observation` vectors. Conceptually
  continuous (real-valued drives), but tile-coded to discrete tiles in
  the learner.
- $\mathcal{A}$ = `WellKnownTemplate` enum (16 templates) × parameter slots.
  Some entries (`Step{direction}`) further discretize a continuous param.
- $P$ = the deterministic-but-complex composition of all Bevy systems.
  Conceptually $P$ is a function (the world is deterministic given a
  seed); pragmatically the agent treats it as a noisy black box.
- $R$ = `PrimaryReward`. Hand-engineered: $w_{\text{alive}} - \text{drive\_cost} - \text{bio\_cost}$.
- $\gamma$ = 0.9.

This is technically a POMDP (the agent sees `Observation`, not the full
world state). The Simulator pretends it's an MDP and hopes the observation
is rich enough. For most short-horizon tasks this works; for long-horizon
ones it doesn't.

### Where this chapter's ideas appear in code

- **$Q$-values stored per-agent:** [`crates/engine/q_learning/src/`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/).
  These are estimates of $Q^{\star}$ — see [Chapter 8](08_temporal_difference_learning.md).
- **The argmax-over-Q for action selection:** [`policy.rs`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/policy.rs)'s
  `argmax_index` function. This is implementing "$\pi^{\star}(s) = \arg\max_a Q^{\star}(s, a)$"
  except $Q^{\star}$ is being learned, not given.
- **The reward as $R(s)$:** [`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)
  defines `PrimaryReward`. Note: this is $R(s)$, a function of the next
  state only — the Simulator uses state-based, not transition-based,
  rewards.

### Why the project doesn't have an explicit MDP model

The Simulator can't write down $P(s' \mid s, a)$ in closed form because:

- $\mathcal{S}$ is enormous (continuous 251-dim observation).
- $P$ depends on the entire world state, not just the agent's observation.
- $P$ is implicitly defined by the Bevy systems and isn't analytically
  available.

This is exactly the model-free RL setting — and is why we'll spend most
of Chapters 4-8 on algorithms that *don't need* explicit $P$.

## 3.10 Exercises

1. **MDP for tic-tac-toe.** Write out $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$
   for tic-tac-toe vs. a fixed random opponent. What's $|\mathcal{S}|$?
   Is $P$ deterministic? What's a good $\gamma$?

2. **Compute a Bellman expectation iteration.** A 1×3 gridworld with states
   $\{L, M, R\}$. Action $\{\text{left}, \text{right}\}$, deterministic
   transitions, $R(\cdot, \cdot, L) = -1$ except $R(R, \text{right}, R) = +10$.
   $\gamma = 0.9$. Policy: always go right. Starting from $V_0 = 0$,
   compute $V_1, V_2, V_3$. Does it look like it's converging?

3. **Compute $Q^{\star}$ by hand.** For the same MDP, compute $Q^{\star}$ for both
   actions in each state. Verify $V^{\star}(s) = \max_a Q^{\star}(s, a)$.

4. **The max-trick lemma, in detail.** Prove that for any functions $f, g$:
   $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$ and
   $|\max_a f(a) - \min_a g(a)| \leq \max_a |f(a) - g(a)|$? (Hint: the
   first is true; is the second?)

5. **Stochastic-policy lemma.** Show that for any stochastic policy $\pi$,
   there exists a deterministic policy $\pi'$ with $V^{\pi'} \geq V^\pi$
   (componentwise). (Hint: greedy on $Q^\pi$.)

6. **POMDP intuition.** In the Simulator, an agent sees food at $t=0$,
   loses sight at $t=1$, then forgets at $t=20$ (memory salience falls
   below threshold). At $t=21$, the agent's `Observation` is identical to
   what it would be if no food had ever been there. Is the Simulator's
   `Observation` a Markov state? Justify.

7. **Bellman as a linear equation.** For a fixed policy $\pi$ and finite
   $\mathcal{S}$, the Bellman expectation equation is $V^\pi = r^\pi + \gamma P^\pi V^\pi$
   where $r^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) R(s, a, s')$
   and $P^\pi(s, s') = \sum_a \pi(a \mid s) P(s' \mid s, a)$. Solve
   formally for $V^\pi$. What matrix needs to be invertible? Use Chapter 1's
   spectral-radius result to argue it always is.

## 3.11 References cited in this chapter

Full bibliographic entries in [`bibliography.md`](bibliography.md):

- [Puterman 2005] — MDP formal definition, Ch. 2-6 (§3.1, §3.8)
- [S&B 2018] — Bellman equations, contraction proofs, $T^{\star}$ analysis (§3.3-§3.7)
- [Bertsekas 2012] — operator-theoretic DP, Ch. 1 (§3.5)
- [Szepesvári 2010] — max-trick lemma proof style (§3.5)
- [Kaelbling, Littman & Cassandra 1998] — POMDP foundations (§3.7)
- [Banach 1922] — fixed-point theorem (§3.3, §3.5)

## 3.12 Further reading

| Source | What to read | Why |
|---|---|---|
| [S&B 2018] | Ch. 3 | The canonical textbook MDP chapter |
| [Bertsekas 2012] | Ch. 1 | Rigorous operator-theoretic version |
| [Puterman 2005] | Ch. 2-6 | The MDP theory reference |
| [Kaelbling, Littman & Cassandra 1998] | The paper | If POMDPs interest you |

---

**Next:** [Chapter 6 — Dynamic Programming](06_dynamic_programming.md) — when you know $P$ and $R$, you can solve the MDP exactly. The algorithms are policy iteration and value iteration, and their convergence is the Banach theorem we proved here.

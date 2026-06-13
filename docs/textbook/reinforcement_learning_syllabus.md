# Reinforcement Learning & Foundational AI — Master's-Level Self-Study Syllabus

A ~34-week, ~10-15 hr/week study plan covering reinforcement learning, the
underlying decision-theoretic and dynamic-programming foundations, the
modern deep-RL frontier, plus a 12-week applied addendum (feature
engineering, memory, action spaces, long-horizon credit assignment, and
project synthesis). Pitched at a CS bachelor's graduate with basic ML
exposure. Anchored in real graduate syllabuses from Stanford, Berkeley,
MIT, Princeton, CMU, and Georgia Tech, and the canonical RL and AI
textbooks.

**Structure.** Weeks 1–22 are the core curriculum; weeks 23–32 are the
applied extensions covering project-relevant topics.

Every module is contextualised against this project's simulator
(`crates/engine/q_learning`, `crates/cognition/planner`,
`crates/sim/app/tests/{tasks,curricula,pathologies}/`) so the abstract
theory lands on code you already understand.

---

## 0. How this syllabus is built

### Sources triangulated

Consensus topic ordering pulled from these graduate RL courses:

- **Stanford CS234** (Emma Brunskill) — [course page](https://web.stanford.edu/class/cs234/)
- **UC Berkeley CS285** (Sergey Levine) — [course page](https://rail.eecs.berkeley.edu/deeprlcourse/)
- **CMU 10-703** Deep RL & Control — [course page](https://www.andrew.cmu.edu/course/10-703/)
- **MIT 6.7920** RL Foundations & Methods — [course page](https://web.mit.edu/6.7920/www/)
- **Princeton ECE 524** Foundations of RL (Chi Jin) — [course page](https://sites.google.com/view/cjin/teaching/ece524)
- **Georgia Tech CS 7642** Reinforcement Learning — [course page](https://omscs.gatech.edu/cs-7642-reinforcement-learning)

Five of six converge on the same spine: **MDPs → DP → MC/TD/Q-learning →
function approximation → policy gradients → actor-critic → model-based →
exploration → offline RL → frontier**. The differentiators (alignment,
multi-agent, control-as-inference, theory) become electives in later
weeks.

### Textbook canon

| Tier | Book | Role |
|---|---|---|
| Universal | **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018** ([free PDF](http://incompleteideas.net/book/the-book-2nd.html)) | Primary spine — every grad course assigns this |
| Theory polish | **Szepesvári, *Algorithms for Reinforcement Learning*, 2010** ([free PDF](https://sites.ualberta.ca/~szepesva/rlbook.html)) | Tight ~100pp on convergence, contraction-mapping proofs |
| Theory depth | **Bertsekas, *Dynamic Programming and Optimal Control*, Vols I & II** | Rigorous operator-theoretic treatment; MIT's choice |
| MDP theory | **Puterman, *Markov Decision Processes*, 2005** | Definitive on MDP theory, average-reward, LP formulations |
| Modern theoretical | **Agarwal, Jiang, Kakade, Sun, *Reinforcement Learning: Theory and Algorithms*** ([draft PDF](https://rltheorybook.github.io/)) | Princeton's choice; modern theoretical foundations |
| AI breadth | **Russell & Norvig, *AI: A Modern Approach*, 4th ed., 2020** | Decision theory, MDPs, probabilistic reasoning — Parts IV, V |
| DL prereq | **Goodfellow, Bengio, Courville, *Deep Learning*, 2016** ([free PDF](https://www.deeplearningbook.org/)) | Function approximation foundations, optimization, generative models |
| Bandits | **Lattimore & Szepesvári, *Bandit Algorithms*, 2020** ([free PDF](https://tor-lattimore.com/downloads/book/book.pdf)) | The reference for the bandit/exploration thread |

**Acquire all eight as PDFs before starting.** The four free ones plus
Russell & Norvig and Bertsekas Vol II will carry you through 90% of
this course.

### Video lecture companions

- **David Silver — UCL RL** (10 lectures, [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), [slides](https://davidstarsilver.wordpress.com/teaching/)) — the polished companion to Sutton & Barto
- **Sergey Levine — CS285** ([YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfX7MaC6C3HcdOf1g337dlC9)) — the deep-RL canon
- **Emma Brunskill — CS234** ([YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)) — strong on offline RL & alignment

### Mathematical prerequisites

Comfortable with:
- Linear algebra: eigenvalues/vectors, matrix decompositions, norms
- Probability: σ-algebras, conditional expectation, Markov chains, concentration inequalities (Hoeffding, Bernstein)
- Multivariable calculus: gradients, Jacobians, Lagrange multipliers
- Real analysis at the level of "I know what a contraction mapping is"
- Convex optimization basics: gradient descent, KKT conditions
- Python + NumPy. Bonus: PyTorch or JAX.

If you're rusty, **Week 0** is a structured refresh.

---

## Week 0 — Mathematical & Software Foundations

### Objectives
Get the math toolkit and software stack into working order. By end of week, you should be able to write down the Bellman equation, derive its contraction property in `‖·‖_∞`, and run a tabular gridworld Q-learning loop in Python.

### Reading
- **Goodfellow, Ch. 2** (Linear Algebra) — skim for refresh
- **Goodfellow, Ch. 3** (Probability & Information Theory) — read carefully
- **Goodfellow, Ch. 4** (Numerical Computation) — read carefully; gradient descent, conditioning, the realities of float math
- **Boyd & Vandenberghe, *Convex Optimization*** ([free PDF](https://web.stanford.edu/~boyd/cvxbook/)) — Ch. 1-3 for orientation
- **Banach fixed-point theorem & contraction mappings** — pick any analysis textbook or Wikipedia for a careful treatment

### The math that matters
- **Banach fixed-point theorem.** If `T: X → X` is a contraction on a complete metric space (`d(Tx, Ty) ≤ γ · d(x, y)` for some `γ < 1`), then `T` has a unique fixed point `x*` and iteration `x_{k+1} = T(x_k)` converges to it geometrically. **This single theorem underlies every convergence result in tabular RL.**
- **Geometric series:** `Σ_{k=0}^∞ γ^k = 1/(1−γ)`. Memorize this — it sets the "effective horizon" of discounting.
- **Hoeffding's inequality:** for `X_i ∈ [a,b]` i.i.d., `P(|X̄_n − E[X]| > ε) ≤ 2·exp(−2nε²/(b−a)²)`. Used in every PAC RL bound.
- **Conditional expectation** `E[Y | X]` as a function of `X`. Tower property: `E[Y] = E[E[Y | X]]`.

### Project tie-in
- **`crates/engine/q_learning/src/learning_rate.rs`** — read it. The `effective_alpha_gamma` function combines per-agent meme overrides with the global `α, γ`. Now you know `α` and `γ` are *the* hyperparameters of TD-learning.
- **`crates/sim/sim_config/src/learning.rs`** — your `gamma: 0.9`. Effective horizon = 1/(1−0.9) = 10 ticks. With cognition cadence of 10 ticks/cycle, that's ~1 cognition cycle of foresight. **Reflect on whether this matches the actual learning task.**

### Exercises
1. Derive that the Bellman operator `(TQ)(s,a) = E[r + γ max_a' Q(s',a')]` is a contraction in `‖·‖_∞` with modulus `γ`. Use the fact that `|max f − max g| ≤ max|f − g|`.
2. Write a tabular Q-learning agent for the classic 4×4 gridworld in Python. Plot return-per-episode. Vary `α` ∈ {0.01, 0.1, 0.5} and `γ` ∈ {0.5, 0.9, 0.99}. Observe.
3. Read `crates/sim/sim_config/src/reward.rs` — write out the PrimaryReward formula symbolically. Compute the steady-state Q under purely `w_alive=1.0` reward at `γ=0.9` — you should get ~10. (This is the value that bootstrap-pollutes Q in the bug we documented.)

### Time
~15 hr. Adjust based on your math freshness.

---

## Part I — Foundations of AI & Decision Theory (Weeks 1-2)

The pre-RL foundation: classical AI search and the decision-theoretic framing that motivates MDPs.

### Week 1 — Search, planning, classical AI

#### Objectives
Understand the search-based formulation of AI problems and where it succeeds (deterministic, known dynamics) vs. fails (stochastic, unknown dynamics). This is the regime your project's deleted forward-search planner attempted.

#### Reading
- **Russell & Norvig, Ch. 3** — Solving Problems by Searching (uninformed, informed, A*)
- **Russell & Norvig, Ch. 4** — Search in Complex Environments (local search, gradient descent, simulated annealing, evolutionary)
- **Russell & Norvig, Ch. 5** — Adversarial Search & Games (minimax, alpha-beta, **MCTS**, stochastic games)
- **Russell & Norvig, Ch. 11** — Automated Planning (classical, hierarchical, nondeterministic)

#### Project tie-in
- Read `docs/designs/04_cognition.md` — the cognition pipeline architecture
- Read the comments at the top of `crates/cognition/planner/src/policy.rs` — note the "There is no forward search" comment and the reasoning. Then think about *why* forward search was killed (the previous session's main commit removed the planner).
- The dead forward-search code in git history: `git log --oneline --all | head -20` and find commits before `5d34d62`. The deleted `search.rs` was a classical depth-limited rollout planner.

#### Exercises
1. Implement minimax + alpha-beta for tic-tac-toe.
2. Implement A* for an 8-puzzle. Tune heuristics.
3. Read the deleted `search.rs` from git history (it's still in `git show HEAD~N`). What does it do? Where does it fail? Connect this to Russell & Norvig's discussion of "search in nondeterministic environments."

#### Time
~10 hr.

### Week 2 — Probability, utility, and decision theory

#### Objectives
Build the formal apparatus of "rational agent under uncertainty." MDPs are decision theory + Markov assumption + dynamics + reward function.

#### Reading
- **Russell & Norvig, Ch. 12-13** — Quantifying Uncertainty, Probabilistic Reasoning (Bayes nets, exact + approximate inference)
- **Russell & Norvig, Ch. 14** — Probabilistic Reasoning over Time (HMMs, Kalman filters, particle filtering) — important for partially observable settings
- **Russell & Norvig, Ch. 16** — Making Simple Decisions (utility theory, MEU, decision networks, information value)
- **Russell & Norvig, Ch. 17** — Making Complex Decisions (MDPs, value iteration, policy iteration, **POMDPs**)

#### The math that matters
- **Maximum expected utility (MEU):** the rational agent picks `argmax_a Σ_s P(s|a) · U(s)`. This is what V-based decision-making is doing.
- **Bellman equation** as the recursive form of MEU for sequential decisions.

#### Project tie-in
- Your agents are *partially* observed — the 199-dim observation is a projection of full world state. Formally this is a **POMDP**, not an MDP. Read R&N's POMDP section carefully.
- The cognition pipeline doesn't maintain a belief state explicitly (no Kalman filter, no particle filter). It treats the observation as a sufficient statistic. **This is an approximation.** Note where this could matter (e.g. agents seeing food briefly then not, with no memory).

#### Exercises
1. Write up the MDP for a hungry agent in your sim — what are `S`, `A`, `P`, `R`, `γ`? Be honest about what's POMDP rather than MDP.
2. Implement a Bayes net for "is there food in this cell?" given observations across multiple ticks. (This is what proper belief maintenance would look like.)

#### Time
~12 hr.

---

## Part II — MDPs & Dynamic Programming (Weeks 3-4)

The rigorous mathematical foundation for everything that follows.

### Week 3 — Markov Decision Processes

#### Objectives
Internalize MDPs and the Bellman equations cold. Be able to derive value iteration and policy iteration from scratch and prove they converge.

#### Reading
- **Sutton & Barto, Ch. 3** — Finite Markov Decision Processes
- **Szepesvári, Ch. 1** — MDPs (concise, more mathematical)
- **Puterman, Ch. 2-3** — Model Formulation, Examples (skim for the inventory and stopping examples — great intuitions)
- **Bertsekas Vol I, Ch. 1** — The Dynamic Programming Algorithm

#### The math that matters
- **MDP tuple:** `(S, A, P, R, γ)` where `P: S×A×S → [0,1]` is the transition kernel, `R: S×A×S → ℝ` the reward function, `γ ∈ [0,1)` the discount factor.
- **Return:** `G_t = Σ_{k=0}^∞ γ^k r_{t+k}`.
- **State-value:** `V^π(s) = E_π[G_t | s_t = s]`.
- **Action-value:** `Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]`.
- **Bellman expectation equation:** `V^π(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γ V^π(s')]`
- **Bellman optimality:** `V*(s) = max_a Σ_{s',r} P(s',r|s,a)[r + γ V*(s')]`
- **Optimal policy from V*:** `π*(s) = argmax_a Σ_{s',r} P(s',r|s,a)[r + γ V*(s')]` (needs P) or **from Q*:** `π*(s) = argmax_a Q*(s,a)` (model-free).

#### Project tie-in
- Define the MDP of `learning_homeostatic`'s hungry arm: `S` = (hunger × tick), `A` = {consume, wait, ...}, `P` = deterministic (consume → hunger drops, wait → hunger climbs), `R` = `PrimaryReward(s)`.
- Write out the Bellman equation for this concrete MDP. What does `V*` look like at `γ=0.9`?

#### Exercises
1. Prove that under `R_max = max |R|`, every `V^π(s)` is bounded by `R_max/(1−γ)`.
2. Prove that the Bellman optimality operator is a contraction in `‖·‖_∞`.
3. **Sutton & Barto Ex. 3.14, 3.17, 3.22** — work these.

#### Time
~12 hr.

### Week 4 — Dynamic Programming

#### Objectives
Master policy iteration and value iteration. Understand asynchronous DP, generalized policy iteration, and the LP formulation. **This is the regime where you have a model and can plan exactly.**

#### Reading
- **Sutton & Barto, Ch. 4** — Dynamic Programming
- **Szepesvári, Ch. 1** (DP algorithms section)
- **Bertsekas Vol II, Ch. 1-2** — Discounted Problems Theory + Computational Methods (the rigorous treatment)
- **Puterman, Ch. 6** — Discounted MDPs (contraction proofs, modified PI, LP formulation, action elimination)

#### Papers (optional)
- **Howard, *Dynamic Programming and Markov Processes*, 1960** — the original policy iteration paper. Read for history.

#### The math that matters
- **Policy evaluation:** `V_{k+1}(s) := Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R + γ V_k(s')]`. Converges to `V^π`.
- **Policy improvement:** `π'(s) := argmax_a Σ_{s'} P(s'|s,a)[R + γ V^π(s')]`. Guaranteed to be at least as good as `π`.
- **Policy iteration:** alternate evaluation and improvement. Converges in finite steps for finite MDPs.
- **Value iteration:** `V_{k+1}(s) := max_a Σ_{s'} P(s'|s,a)[R + γ V_k(s')]`. Converges geometrically to `V*`.
- **Generalized Policy Iteration (GPI):** the unifying view — any alternation of partial evaluation and partial improvement converges.

#### Project tie-in
- Your project does NOT use DP because it doesn't have a model `P` it can query. **This is a fundamental constraint.** Could you build one? The deterministic engine *is* `P` — but rolling it forward per (s,a) is expensive. The deleted forward search tried this.
- Connect to Russell & Norvig Ch. 17's discussion of online vs. offline planning.

#### Exercises
1. Implement value iteration on Sutton & Barto's gridworld (Ex. 4.1). Plot value function as a heatmap.
2. Implement policy iteration on the same gridworld. Compare convergence speed (iterations and wall time) to VI.
3. **Sutton & Barto Ex. 4.7-4.9.** Especially 4.9 (Jack's car rental).

#### Time
~15 hr.

---

## Part III — Tabular Model-Free RL (Weeks 5-6)

The transition from "I know the model" to "I only see samples."

### Week 5 — Monte Carlo & Bandits

#### Objectives
Two threads: (a) Monte Carlo as the alternative to DP/TD — wait for episodes to end, average returns. (b) Bandits as the exploration/exploitation primitive.

#### Reading
- **Sutton & Barto, Ch. 2** — Multi-armed Bandits (ε-greedy, UCB, gradient bandits, contextual)
- **Sutton & Barto, Ch. 5** — Monte Carlo Methods (first/every-visit, exploring starts, on/off-policy, importance sampling)
- **Lattimore & Szepesvári, *Bandit Algorithms*** Ch. 1-4, 7-9 (the rigorous bandit theory)

#### Papers
- **Auer, Cesa-Bianchi, Fischer 2002** — UCB1 finite-time analysis. [link.springer.com](https://link.springer.com/article/10.1023/A:1013689704352)
- **Thompson 1933** — the original posterior-sampling heuristic. [Biometrika](https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862)
- **Russo et al. 2018** — Thompson Sampling tutorial. [arxiv.org/abs/1707.02038](https://arxiv.org/abs/1707.02038)

#### The math that matters
- **MC estimate of `Q^π(s,a)`:** mean of all observed returns from `(s,a)`.
- **Importance sampling correction for off-policy MC:** `ρ = π(a|s)/μ(a|s)`, ratio of target to behavior policy.
- **UCB1:** `argmax_a (Q̂(a) + c·√(log t / N_a))`. Optimism in the face of uncertainty.
- **Thompson sampling:** maintain a posterior over arm rewards, sample, act greedily on sample.
- **Regret bounds:** UCB1 achieves `O(log T)` regret; Thompson sampling matches asymptotically.

#### Project tie-in
- Your policy uses **ε-greedy at ε=0.1** — the simplest exploration strategy. Read `policy.rs:485-488` where `hash_unit(entity, tick, EXPLORE_SALT) < epsilon` decides explore vs. exploit. The deterministic hash makes ε-greedy *reproducible*, which is critical for your determinism canary.
- **UCB/Thompson would be a real upgrade.** They explore actions in proportion to uncertainty, not uniformly. Untried Plant would get a UCB bonus that uniformly-random ε-greedy doesn't give it — could break the Q-bootstrap lock-in.

#### Exercises
1. Implement UCB1 on a 10-armed bandit. Plot regret over time.
2. Implement Thompson sampling on the same bandit. Compare.
3. Implement Monte Carlo control with ε-greedy on the gridworld.
4. **Read `policy.rs:228-235` (the deterministic hash).** Modify it conceptually to compute a UCB bonus instead of uniform exploration. What would the formula look like? (Pseudo-counts of `(observation tile, action)` co-occurrence + UCB bonus.)

#### Time
~15 hr.

### Week 6 — Temporal Difference Learning (TD, SARSA, Q-learning)

#### Objectives
The single most important week. Understand TD(0), SARSA, Q-learning, and expected SARSA cold. Be able to derive each from first principles and explain when to pick which.

#### Reading
- **Sutton & Barto, Ch. 6** — TD Learning (TD(0), SARSA, Q-learning, expected SARSA, double Q, maximization bias)
- **Sutton & Barto, Ch. 7** — n-step Bootstrapping
- **Szepesvári, Ch. 2** (value prediction problems)
- **Bertsekas Vol II, Ch. 6** (approximate DP, TD methods from operator-theoretic viewpoint)

#### Papers (essential)
- **Sutton 1988** — TD learning original. [PDF](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
- **Watkins & Dayan 1992** — Q-learning convergence proof. [PDF](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)
- **van Hasselt 2010** — Double Q-learning. [NeurIPS](https://papers.nips.cc/paper/3964-double-q-learning)

#### The math that matters
- **TD(0) update:** `V(s) ← V(s) + α [r + γ V(s') − V(s)]`. **The bracket is the TD error `δ`.**
- **SARSA (on-policy):** `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]` where `a'` is the *actual* next action.
- **Q-learning (off-policy):** `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`. The `max` makes it off-policy — it learns the value of the greedy policy regardless of what you actually did.
- **Expected SARSA:** replace `Q(s',a')` with `E_π[Q(s',a')]`. Lower variance than SARSA.
- **Maximization bias:** Q-learning's `max` overestimates because `E[max X] ≥ max E[X]`. Double Q-learning fixes this by decoupling action selection from action evaluation.

#### Project tie-in
- Read `crates/engine/q_learning/src/lib.rs` and `learning_rate.rs`. The implemented update is **TD(0) Q-learning** with optional per-agent `α` override via the `meme.learning_rate` meme.
- Note: there is no **double Q-learning** — your project has the maximization bias. Whether that matters at your scale is empirical; on Atari it mattered hugely.
- The `learning_rate_meme_divergence` test demonstrates per-agent `α` differentiation — re-read it now and confirm you understand exactly what's being asserted.

#### Exercises
1. Implement tabular Q-learning, SARSA, and Expected SARSA on the cliff-walking task (Sutton & Barto Ex. 6.6). Compare paths learned — observe Q-learning's "cliff edge" preference vs. SARSA's caution.
2. Implement double Q-learning. Show on a stochastic gridworld that it reduces overestimation.
3. **Code experiment on your sim:** add a knob to swap Q-learning for SARSA in your `Learner`. Run `learning_homeostatic` hungry arm with both. Do they differ? What does this tell you about your project's exploration-vs-exploitation regime?

#### Time
~18 hr. This is the week you have to nail.

---

## Part IV — Function Approximation (Weeks 7-8)

Tables don't scale. Function approximation is mandatory for any real system — including yours.

### Week 7 — Linear Function Approximation & Tile Coding

#### Objectives
Understand linear value function approximation — its convergence theory, its limitations, and tile coding in particular. **This is what your project uses.**

#### Reading
- **Sutton & Barto, Ch. 9** — On-policy Prediction with Approximation
- **Sutton & Barto, Ch. 10** — On-policy Control with Approximation
- **Sutton & Barto, Ch. 11** — Off-policy Methods with Approximation (introduces the deadly triad)
- **Szepesvári, Ch. 2** (FA section)
- **Bertsekas Vol II, Ch. 6** (projected Bellman equations, LSTD, LSPI)

#### Papers (essential)
- **Tsitsiklis & Van Roy 1997** — Linear TD convergence and divergence. [PDF](https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf)
- **Baird 1995** — The 7-state Baird counterexample where linear off-policy Q-learning diverges. [PDF](https://www.leemon.com/papers/1995b.pdf)
- **van Hasselt et al. 2018** — *Deep RL and the Deadly Triad*. [arxiv.org/abs/1812.02648](https://arxiv.org/abs/1812.02648)

#### The math that matters
- **Linear VFA:** `V(s; θ) = θ · φ(s)` where `φ(s)` is a feature vector.
- **Semi-gradient TD(0):** `θ ← θ + α [r + γ V(s'; θ) − V(s; θ)] ∇_θ V(s; θ)`. "Semi" because we ignore the gradient through the target.
- **Tile coding:** `φ(s)` is a one-hot over multiple overlapping discretizations. Sparse, fast, locally generalizing.
- **The Deadly Triad:** function approximation + bootstrapping + off-policy → potential divergence (Baird counterexample). Q-learning with FA is therefore not guaranteed to converge.
- **Gradient TD methods (GTD, TDC):** modifications that converge under the triad.

#### Project tie-in
- `crates/engine/q_learning/src/observation_tile_coder.rs` — your tile-coded feature builder
- `crates/engine/q_learning/src/tile_coding/` — the hash-based tile coder
- `crates/sim/sim_config/src/learning.rs:tile_coding` — `iht_size = 2^16`, number of tilings, tile widths
- **You are running off-policy (Q-learning) + bootstrap + linear FA — exactly the deadly triad.** Your project's stability depends on careful hyperparameters and the fact that the bootstrap targets stay bounded. Read the Tsitsiklis & Van Roy paper carefully — note when convergence is guaranteed (on-policy linear) and when it isn't (off-policy linear with arbitrary distribution).

#### Exercises
1. Implement linear semi-gradient Q-learning on Mountain Car using tile coding. Compare to tabular discretization.
2. Reproduce the Baird counterexample. Watch the weights diverge.
3. **Read your tile coder.** What's the tile width on each observation dimension? How many tilings? Is the hash collision rate low (`iht_size = 65536` for a 199-dim observation)? Compute the expected number of distinct tiles needed and compare.

#### Time
~15 hr.

### Week 8 — Neural Network Function Approximation, Eligibility Traces

#### Objectives
Bridge to deep RL. Understand neural network VFA, why it's harder than linear, and the eligibility-trace family that unifies MC and TD.

#### Reading
- **Sutton & Barto, Ch. 12** — Eligibility Traces (TD(λ), true online TD(λ), Sarsa(λ), Watkins's Q(λ))
- **Goodfellow Ch. 6-8** (deep feedforward nets, regularization, optimization) — skim if you've done DL
- **Sutton & Barto, Ch. 9.7** — Neural Network function approximation overview

#### Papers
- **van Seijen et al. 2016** — True Online TD(λ). [arxiv.org/abs/1512.04087](https://arxiv.org/abs/1512.04087)

#### The math that matters
- **Eligibility traces:** track "which states were responsible for the current reward" via `e(s) ← γ λ e(s) + 1(s_t = s)`. λ=0 → TD(0), λ=1 → MC.
- **TD(λ) update:** propagates δ backward through the eligibility trace, achieving multi-step credit assignment in one update.
- **Backpropagation:** chain rule on the loss `L = (r + γ V(s';θ) − V(s;θ))²`. Note that semi-gradient ignores `∇V(s';θ)` — important detail.

#### Project tie-in
- Your project does NOT use eligibility traces or neural-net function approximation. **It uses TD(0) with linear tile-coded FA.** This is a *specific* design choice with tradeoffs:
  - + Simple, fast, well-understood convergence.
  - + No GPU dependency.
  - − No multi-step credit assignment → can't bridge long delays (this is part of the L-suite failure pathology).
- The previously-deleted "successor features" code was a multi-step credit-assignment mechanism. It was removed because it failed in specific ways — but eligibility traces would be a simpler alternative.

#### Exercises
1. Implement TD(λ) for prediction on a random walk. Plot RMS error vs. λ.
2. Implement Watkins's Q(λ) on Mountain Car.
3. **Conceptual project exercise:** if you added TD(λ) to your project, where would it help? Sketch how it would change L1's Plant→Consume credit propagation. (Eligibility traces would let one Consume reward backpropagate to *every* preceding action in the trace, including the upstream Plants.)

#### Time
~15 hr.

---

## Part V — Deep Q-Learning (Weeks 9-10)

The modern value-based canon. Even if you stay linear, these papers shape the field.

### Week 9 — DQN and stabilization tricks

#### Objectives
Understand DQN — experience replay, target networks, why it works. Then the stabilization stack: double DQN, dueling networks, prioritized replay.

#### Reading
- **Sutton & Barto, Ch. 16.5** — Atari games case study
- **Levine CS285 L7-8** — Value-based RL & DQN

#### Papers (all essential)
- **Mnih et al. 2015** — DQN Nature. [Nature](https://www.nature.com/articles/nature14236)
- **van Hasselt, Guez, Silver 2016** — Double DQN. [arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)
- **Wang et al. 2016** — **Dueling DQN.** [arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581) — **READ CAREFULLY: this is the architectural fix for your Q-bias bootstrap bug.**
- **Schaul et al. 2016** — Prioritized Experience Replay. [arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)
- **Hessel et al. 2018** — Rainbow. [arxiv.org/abs/1710.02298](https://arxiv.org/abs/1710.02298)

#### The math that matters
- **DQN loss:** `L = E[(r + γ max_{a'} Q_target(s', a') − Q(s, a; θ))²]`. Target network `Q_target` updated periodically to break correlation.
- **Experience replay:** sample `(s, a, r, s')` uniformly from a buffer. Decorrelates updates.
- **Double DQN:** target = `r + γ Q_target(s', argmax_a Q(s', a; θ))`. Decouple selection from evaluation.
- **Dueling architecture:** `Q(s,a) = V(s) + A(s,a) − (1/|A|) Σ_a' A(s,a')`. Decomposes Q into state value + advantage.
- **PER:** sample with probability `p_i ∝ |δ_i|^α`, correct bias with importance-sampling weight `w_i = (N · p_i)^{-β}`.

#### Project tie-in
- **The dueling architecture is an alternative angle on the Simulator's Q-bias bootstrap bug** (Chapter 17). That bug — where every committed action's Q saturated toward `w_alive/(1−γ)` so the agent locked onto its first action — was *resolved* by removing the `w_alive` baseline (reward is now the per-tick drive-delta `R = cost(s_prev) − cost(s)`). Read the Wang paper carefully: the advantage decomposition `Q(s,a) = V(s) + A(s,a)` is a different way to strip out a state-value baseline, worth studying even though the shipped fix took the simpler route.
- Your project would benefit from PER too: TD-error-prioritized replay would help the agent revisit "surprising" transitions, breaking the Q-bootstrap monotony.
- Why is your project not using DQN? **You don't use deep nets** — your linear tile-coded learner is fundamentally simpler and avoids deep RL's instability issues. The tradeoff: less representational power.

#### Exercises
1. Implement DQN on Cartpole using PyTorch. Plot return curves.
2. Add double DQN, dueling, and PER one at a time. Ablate.
3. **Project exercise:** sketch a dueling decomposition for your linear tile-coded learner. `V(s) = θ_V · φ(s)`, `A(s, a) = θ_A^a · φ(s)`, `Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)`. What changes in the score formula at `policy.rs:466`?

#### Time
~18 hr.

### Week 10 — Distributional RL

#### Objectives
The frontier of value learning. Learn the *distribution* of returns, not just the mean.

#### Reading
- **Bellemare, Dabney, Rowland, *Distributional Reinforcement Learning*, 2023** ([MIT Press](https://www.distributional-rl.org/)) — Ch. 1-4

#### Papers
- **Bellemare, Dabney, Munos 2017** — C51. [arxiv.org/abs/1707.06887](https://arxiv.org/abs/1707.06887)
- **Dabney, Rowland, Bellemare, Munos 2018** — QR-DQN. [arxiv.org/abs/1710.10044](https://arxiv.org/abs/1710.10044)
- **Dabney, Ostrovski, Silver, Munos 2018** — Implicit Quantile Networks. [arxiv.org/abs/1806.06923](https://arxiv.org/abs/1806.06923)

#### The math that matters
- **Distributional Bellman operator:** `T^π Z(s,a) :=^D R + γ Z(s', a')` — equality in distribution.
- **C51:** discretize the return support into 51 atoms; project the Bellman update onto this support; KL-divergence loss.
- **Quantile regression DQN:** maintain quantiles of the return distribution; quantile regression loss; provably 1-Wasserstein contraction.

#### Project tie-in
- Your project doesn't use distributional RL. Worth knowing about for two reasons: (a) richer learning signal generally improves representation quality; (b) explicit distributions enable risk-aware behavior (e.g. risk-averse agents).
- A risk-averse hunting wolf might prefer reliably-small-prey over occasionally-huge-but-often-failed hunts. Distributional RL would express this; expected-Q can't.

#### Exercises
1. Implement C51 on Cartpole.
2. Read the Bellemare-Dabney-Rowland book Ch. 1-2. Work through the operator theory.

#### Time
~10 hr.

---

## Part VI — Policy Gradient & Actor-Critic (Weeks 11-12)

The alternative to value-based methods. Direct policy parameterization + advantage estimation.

### Week 11 — Policy Gradient

#### Objectives
Understand the policy gradient theorem, REINFORCE, and baselines. This is where actor-critic methods come from.

#### Reading
- **Sutton & Barto, Ch. 13** — Policy Gradient Methods
- **Levine CS285 L5** — Policy Gradients

#### Papers (essential)
- **Williams 1992** — REINFORCE. [Springer](https://link.springer.com/article/10.1007/BF00992696)
- **Sutton et al. 2000** — Policy gradient theorem. [NeurIPS](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)

#### The math that matters
- **Policy gradient theorem:** `∇_θ J(θ) = E_π[ Σ_t γ^t ∇_θ log π(a_t|s_t; θ) · Q^π(s_t, a_t)]`
- **REINFORCE:** estimate `Q^π` by Monte Carlo return `G_t`. Unbiased but high-variance.
- **Baseline subtraction:** `∇_θ J ≈ E[ ∇_θ log π · (G_t − b(s_t)) ]`. Subtracting any function of state is unbiased.
- **Optimal baseline = V^π(s_t)**. With this, the multiplier becomes the advantage `A(s,a) = Q(s,a) − V(s)`. **This is where advantage shows up.**

#### Project tie-in
- Your project does NOT use policy gradient. The flat policy uses **value-based argmax over scored candidates**. PG would parameterize π directly as a distribution over candidates.
- An interesting hybrid: use the flat policy's argmax as a deterministic policy, but learn its gradient via the DPG theorem (Silver et al. 2014). This is what DDPG does for continuous control.

#### Exercises
1. Implement REINFORCE on Cartpole. Observe the variance — return curves are jagged.
2. Add a learned V(s) baseline. Variance drops, learning speeds up.
3. **Conceptual:** could your sim use PG? What would `π(a|s; θ)` look like over your 16+ candidates? Pros vs. cons.

#### Time
~15 hr.

### Week 12 — Actor-Critic & Advantage Estimation

#### Objectives
The modern workhorse family. Combine value-based critic with policy-based actor.

#### Reading
- **Sutton & Barto, Ch. 13.5-13.8** — Actor-critic methods
- **Levine CS285 L6** — Actor-critic

#### Papers
- **Mnih et al. 2016** — A3C. [arxiv.org/abs/1602.01783](https://arxiv.org/abs/1602.01783)
- **Schulman et al. 2015** — Generalized Advantage Estimation (GAE). [arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)
- **Wang et al. 2016** — Dueling DQN (review — it's actor-critic-like).

#### The math that matters
- **A2C update:**
  - Critic: `δ = r + γ V(s'; φ) − V(s; φ)`; `φ ← φ + α_c · δ · ∇_φ V(s; φ)`.
  - Actor: `θ ← θ + α_a · δ · ∇_θ log π(a|s; θ)`.
- **GAE:** `A^GAE = Σ_l (γλ)^l δ_{t+l}`. Interpolates between TD(0) (λ=0, low variance, biased) and MC (λ=1, high variance, unbiased).
- **A3C:** run N actor-learners in parallel against environment copies. Decorrelated gradients stabilize without replay.

#### Project tie-in
- **Actor-critic is a principled alternative technique for the kind of Q-bias bootstrap bug you saw in the Simulator** (Chapter 17). The critic V(s) absorbs the state-value baseline; the advantage `Q(s,a) − V(s)` is what drives the actor — and the advantage doesn't bootstrap pollute.
- Note: the Simulator's actual bug — every committed action's Q saturating toward `w_alive/(1−γ)` — was *resolved* by simply removing the `w_alive` baseline (reward is now the per-tick drive-delta `R = cost(s_prev) − cost(s)`), not by adopting actor-critic. Advantage learning remains a worthwhile technique to study as a more sophisticated alternative.
- A linear actor-critic over your tile features would be: actor `π(a|s) ∝ exp(θ_a · φ(s))` (softmax policy), critic `V(s) = w · φ(s)`. Modest engineering lift.

#### Exercises
1. Implement A2C on Cartpole.
2. Add GAE with various λ. Compare.
3. **Project exercise (significant):** prototype a linear actor-critic for your sim. Replace the score formula in `policy.rs` with softmax over scores `recipe_bonus + A(s, a)`. Run `learning_homeostatic`. Does the Q-bootstrap pathology vanish?

#### Time
~18 hr.

---

## Part VII — Advanced Policy Optimization (Week 13)

### Week 13 — TRPO, PPO, Natural Gradient

#### Objectives
The methods that make policy gradient actually work at scale. PPO is the modern default on-policy algorithm.

#### Reading
- **Levine CS285 L9-10** — Advanced policy gradients

#### Papers (essential)
- **Kakade 2001** — Natural Policy Gradient. [NeurIPS](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
- **Schulman et al. 2015** — TRPO. [arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)
- **Schulman et al. 2017** — PPO. [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

#### The math that matters
- **Natural gradient:** `θ ← θ + α · F^{-1} ∇_θ J(θ)` where `F` is the Fisher information matrix of `π`. Steepest ascent in KL-geometry instead of Euclidean.
- **TRPO:** maximize surrogate `L(θ) = E[π_θ/π_old · A^π_old]` subject to `KL(π_old || π_θ) ≤ δ`. Monotonic improvement guarantee.
- **PPO clipped surrogate:** `L^CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]` where `r_t = π_θ/π_old`. Simpler than TRPO, comparable performance.

#### Project tie-in
- Not directly applicable to your current sim (no NN policy). But PPO is the modern default — worth knowing.

#### Exercises
1. Implement PPO on continuous-action MuJoCo (HalfCheetah, Walker2d).
2. Read OpenAI's PPO blog post for engineering details.

#### Time
~12 hr.

---

## Part VIII — Continuous Control (Week 14)

### Week 14 — DDPG, TD3, SAC

#### Objectives
Off-policy actor-critic methods for continuous actions.

#### Reading
- **Levine CS285 L17** — Q-learning with continuous actions

#### Papers
- **Lillicrap et al. 2015** — DDPG. [arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)
- **Fujimoto, van Hoof, Meger 2018** — TD3. [arxiv.org/abs/1802.09477](https://arxiv.org/abs/1802.09477)
- **Haarnoja et al. 2018** — SAC. [arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)
- **Silver et al. 2014** — Deterministic Policy Gradient theorem. [PMLR](http://proceedings.mlr.press/v32/silver14.html)

#### The math that matters
- **DPG theorem:** for a deterministic policy `μ_θ`, `∇_θ J = E[∇_θ μ_θ(s) · ∇_a Q^μ(s,a)|_{a=μ_θ(s)}]`.
- **TD3 tricks:** clipped double-Q (overestimation fix), delayed policy update, target policy smoothing.
- **SAC entropy regularization:** maximize `E[Σ γ^t (r_t + α H(π(·|s_t)))]`. Maintains exploration via stochastic policy.

#### Project tie-in
- Your sim uses discrete actions. SAC and DDPG don't apply directly. But the **entropy regularization** idea from SAC could apply: penalize a too-deterministic flat policy. Could also help break the Q-bootstrap lock-in.

#### Exercises
1. Implement SAC on MuJoCo continuous control. Compare to PPO from Week 13.
2. **Conceptual:** add a temperature parameter to your flat policy's softmax over scores. Tune so behavior is more stochastic. Observe whether this breaks the Plant-only lock-in in `l1_agent_learns_to_plant`.

#### Time
~12 hr.

---

## Part IX — Model-Based RL (Week 15)

### Week 15 — Planning, Dyna, World Models

#### Objectives
The regime your project's deleted forward search lived in. Understand when model-based methods help and when they don't.

#### Reading
- **Sutton & Barto, Ch. 8** — Planning and Learning (Dyna, prioritized sweeping, MCTS, RTDP)
- **Levine CS285 L11-12, L15-16** — Variational inference, control as inference, model-based RL

#### Papers (essential)
- **Sutton 1990** — Dyna. [ACM](https://dl.acm.org/doi/pdf/10.1145/122344.122377)
- **Browne et al. 2012** — MCTS survey. [IEEE](https://ieeexplore.ieee.org/document/6145622)
- **Silver et al. 2017** — AlphaGo Zero. [Nature](https://www.nature.com/articles/nature24270)
- **Schrittwieser et al. 2019** — MuZero. [arxiv.org/abs/1911.08265](https://arxiv.org/abs/1911.08265)
- **Ha & Schmidhuber 2018** — World Models. [arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)
- **Hafner et al. 2023** — DreamerV3. [arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

#### The math that matters
- **Dyna:** alternate (a) one real environment step → update Q; (b) k simulated steps from learned model → update Q. Effectively gets free data.
- **MCTS:** selection (UCB on tree) → expansion → simulation → backup. UCT formula: `argmax_a Q̂(s,a) + c √(log N(s) / N(s,a))`.
- **World models:** learn `s' = f(s, a, z)` with `z` a latent, then train policy in imagination.
- **MuZero:** learn a latent dynamics that only needs to predict reward, value, and policy at each step — never raw observations.

#### Project tie-in
- **Your deleted forward-search planner was a primitive MCTS-like approach.** Read `docs/designs/04_cognition.md` and any docs about the planner removal. The failure was state-propagation: drives didn't evolve in simulated rollouts. A proper MCTS implementation needs to also roll forward state.
- **Could you do MuZero-style model-based in your sim?** Yes in principle — learn a latent model of just the reward + value, plan with that. Big engineering lift but potentially powerful.

#### Exercises
1. Implement Dyna-Q on a gridworld with a learned tabular model. Plot data efficiency vs. plain Q-learning.
2. Read the MuZero paper carefully. Sketch how it would apply to your sim: what does the latent state encode? what does the dynamics network predict?
3. **Project exercise:** read the git-history `search.rs` from the deleted planner. Write up what specifically failed. Compare to a proper rollout-based approach (e.g. MCTS with a value function trained via your Q-learner).

#### Time
~15 hr.

---

## Part X — Exploration (Week 16)

### Week 16 — Exploration beyond ε-greedy

#### Objectives
Understand why ε-greedy isn't enough and the modern exploration toolkit.

#### Reading
- **Sutton & Barto, Ch. 2** (re-read — bandits as exploration primitive)
- **Lattimore & Szepesvári, *Bandit Algorithms*** — Ch. 7-9, 33-36 (Bayesian methods, RL extensions)

#### Papers (essential)
- **Bellemare et al. 2016** — Pseudo-counts. [arxiv.org/abs/1606.01868](https://arxiv.org/abs/1606.01868)
- **Pathak et al. 2017** — ICM (curiosity). [arxiv.org/abs/1705.05363](https://arxiv.org/abs/1705.05363)
- **Burda et al. 2019** — RND. [arxiv.org/abs/1810.12894](https://arxiv.org/abs/1810.12894)
- **Osband et al. 2016** — Bootstrapped DQN. [arxiv.org/abs/1602.04621](https://arxiv.org/abs/1602.04621)

#### The math that matters
- **Pseudo-counts:** `N̂(s) = ρ(s) · (1−ρ'(s))/(ρ'(s)−ρ(s))` where ρ is a density model and ρ' is its update after seeing s. Bonus = `1/√N̂(s)`.
- **ICM:** intrinsic reward = prediction error of inverse dynamics features.
- **RND:** intrinsic reward = `‖f_target(s) − f_predictor(s; θ)‖²` where target is fixed random net.

#### Project tie-in
- **ε-greedy at 0.1 is barely better than nothing for your sim's exploration needs.** With many candidate actions, ε-greedy/N chance per action per cycle is tiny. Combined with the Q-bootstrap pathology, the agent never escapes its initial policy.
- **A genuine fix would be: UCB or pseudo-counts over (observation_tile, action) co-occurrence.** Untried actions get a high exploration bonus. This breaks the lock-in.

#### Exercises
1. Implement pseudo-count exploration on a sparse-reward gridworld.
2. **Project exercise:** add a UCB-style bonus to your policy's score formula: `score = 0.5·Q + recipe_bonus + c · √(log(t)/N(obs_tile, action))`. Rerun `l1_agent_learns_to_plant`. Does the agent now Consume?

#### Time
~12 hr.

---

## Part XI — Offline RL, Imitation Learning, RLHF (Week 17)

### Week 17 — Learning from fixed datasets

#### Objectives
The setting where you can't (or shouldn't) actively explore. Increasingly important for language models, robotics, healthcare.

#### Reading
- **Levine et al. 2020** — Offline RL tutorial. [arxiv.org/abs/2005.01643](https://arxiv.org/abs/2005.01643)
- **Levine CS285 L17-18** — Offline RL

#### Papers
- **Ross et al. 2011** — DAgger (imitation learning). [arxiv.org/abs/1011.0686](https://arxiv.org/abs/1011.0686)
- **Fujimoto, Meger, Precup 2019** — BCQ. [arxiv.org/abs/1812.02900](https://arxiv.org/abs/1812.02900)
- **Kumar et al. 2020** — CQL. [arxiv.org/abs/2006.04779](https://arxiv.org/abs/2006.04779)
- **Chen et al. 2021** — Decision Transformer. [arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
- **Christiano et al. 2017** — RLHF (preference learning). [arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741)
- **Ziebart et al. 2008** — MaxEnt IRL. [PDF](https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf)

#### The math that matters
- **Extrapolation error:** off-policy Q-learning on a fixed dataset over-estimates Q on out-of-distribution actions (no data to correct).
- **BCQ:** constrain the policy's action distribution to a learned generative model of the data.
- **CQL:** add a regularizer that pushes Q down on out-of-distribution actions.
- **DAgger:** iteratively collect expert corrections on the learner's trajectories.

#### Project tie-in
- Your project doesn't use offline RL. But conceptually: if you wanted to "train" your sim's cognition from gameplay traces of a player, this is the framework.

#### Exercises
1. Implement BCQ on a D4RL Mujoco task.
2. Implement DAgger on a navigation task with a hand-coded expert.

#### Time
~10 hr.

---

## Part XII — Hierarchical & Multi-Agent RL (Week 18)

### Week 18 — Temporal & inter-agent abstraction

#### Objectives
Two complementary directions: hierarchy (options, sub-policies) and multi-agent (game theory, coordination).

#### Reading
- **Sutton & Barto, Ch. 17.5** — Options and hierarchical RL (brief)
- **Albrecht, Christianos, Schäfer, *Multi-Agent Reinforcement Learning***, 2024

#### Papers
- **Sutton, Precup, Singh 1999** — Options framework. [PDF](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
- **Parr & Russell 1998** — HAM. [PDF](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/parr+russell-1998.pdf)
- **Vezhnevets et al. 2017** — FeUdal Networks. [arxiv.org/abs/1703.01161](https://arxiv.org/abs/1703.01161)
- **Lowe et al. 2017** — MADDPG. [arxiv.org/abs/1706.02275](https://arxiv.org/abs/1706.02275)
- **Rashid et al. 2018** — QMIX. [arxiv.org/abs/1803.11485](https://arxiv.org/abs/1803.11485)
- **Foerster et al. 2018** — COMA. [arxiv.org/abs/1705.08926](https://arxiv.org/abs/1705.08926)

#### The math that matters
- **Options framework:** an option is `(I, π, β)` — initiation set, internal policy, termination probability. Reduces to SMDP.
- **Centralized training, decentralized execution:** train with full information (centralized critic), deploy with local information only.
- **Value decomposition (QMIX):** `Q_tot(s, **a**) = mix(Q_1(s, a_1), ..., Q_n(s, a_n))` with monotonic mix → argmax decomposes per-agent.

#### Project tie-in
- **Your project HAS options** — they're called recipes. A `RecipeMeme` with multiple steps is exactly an option: initiation set (precondition), internal policy (recipe step sequence), termination (last step completes). The previously-deleted SMDP-Q machinery was learning option-values.
- **You also have multiple agents.** The current cognition is fully decentralized — each agent learns independently. This is "Independent Q-Learning" — the simplest (and worst-converging) multi-agent baseline. Worth reading QMIX to understand what richer multi-agent learning looks like.

#### Exercises
1. Implement the four-rooms options example from the Sutton/Precup/Singh paper.
2. **Project exercise:** can your recipes be modeled as options under Sutton's formalism? Write up the mapping. What would it mean to learn option-values directly (`Q(recipe, ctx)`) vs. just the constituent primitive-actions' Q? (Hint: this is what the deleted SMDP-Q code was doing.)

#### Time
~12 hr.

---

## Part XIII — Theoretical Foundations (Week 19)

### Week 19 — Convergence, PAC bounds, regret

#### Objectives
The math behind the algorithms. Why does Q-learning converge? When does it diverge? How much exploration do you need?

#### Reading
- **Agarwal, Jiang, Kakade, Sun** — *RL: Theory and Algorithms* ([draft PDF](https://rltheorybook.github.io/)) — Ch. 1-6
- **Princeton ECE 524 lecture notes** (Chi Jin's course site)
- **Szepesvári Ch. 4 + Appendix** — convergence theory

#### Papers
- **Kakade 2003** — PAC RL thesis. [PDF](https://homes.cs.washington.edu/~sham/papers/thesis/sham_thesis.pdf)
- **Jaksch, Ortner, Auer 2010** — UCRL2 regret bound. [JMLR](https://jmlr.org/papers/v11/jaksch10a.html)
- **Strehl, Li, Littman 2009** — PAC analysis of model-based RL.

#### The math that matters
- **Banach fixed-point applied to Bellman:** every step of value iteration shrinks distance to V* by γ. Convergence rate `O(γ^k)`.
- **Q-learning convergence (Watkins & Dayan):** under conditions (Robbins-Monro on α, infinite visits), Q → Q* with probability 1.
- **Sample complexity (PAC):** how many environment samples to reach ε-optimal w.p. ≥ 1−δ?
- **Regret:** `R_T = E[Σ_t V*(s_t) − r_t]`. Lower bound `Ω(√DSAT)` for communicating MDPs.

#### Project tie-in
- **Your project's convergence guarantees are weak.** Off-policy + linear function approximation = deadly triad = no guarantees (Tsitsiklis & Van Roy). Practical convergence is empirical.
- **Q-learning's convergence proof assumes infinite visits to every (s, a).** With ε-greedy at 0.1, the agent might NEVER visit (Consume-context, Consume) — see the L1 failure. Not convergent in any formal sense.

#### Exercises
1. Prove the Banach fixed-point theorem.
2. Work through the Watkins-Dayan Q-learning convergence proof step by step.
3. Read the UCRL2 paper. Note the regret bound, the proof technique (optimism + concentration), and how it relates to your sim's exploration setup.

#### Time
~15 hr.

---

## Part XIV — Function Approximation Pathologies & Your Project (Week 20)

### Week 20 — The deadly triad and your sim's specific issues

#### Objectives
Understand exactly why your project's flat Q-learner has the Q-bias bootstrap pathology, and the engineering choices that lead to similar issues in real RL systems.

#### Reading
- **Sutton & Barto, Ch. 11** — Off-policy methods with approximation (the deadly triad)
- **Chapter 17 (`17_fa_pathologies.md`)** — the Simulator's Q-bias bootstrap bug (now resolved: the `w_alive` baseline was removed in favour of the per-tick drive-delta reward `R = cost(s_prev) − cost(s)`)

#### Papers
- **Baird 1995** — counterexample. [PDF](https://www.leemon.com/papers/1995b.pdf)
- **van Hasselt et al. 2018** — Deadly Triad empirical study. [arxiv.org/abs/1812.02648](https://arxiv.org/abs/1812.02648)
- **Sutton, Maei, Szepesvári 2009** — Gradient TD methods. [JMLR](http://www.jmlr.org/papers/volume17/15-076/15-076.pdf)

#### Project deep-dive (this week's main exercise)
- Read `crates/engine/q_learning/` end to end.
- Read `crates/cognition/planner/src/policy.rs` end to end. Identify every place a Q-value enters a decision.
- Read every `learning_*.rs` test in `crates/sim/app/tests/focused/`. Understand what each one was *trying* to prove and how the Q-bias bootstrap pathology affected outcomes:
  - `learning_homeostatic.rs` — partly works due to recipe_bonus head-start
  - `learning_threat_response.rs` — works because q_flee_margin explicitly subtracts the baseline
  - `learning_navigation.rs` — FAILS for the same Q-bootstrap reason as L1
  - `learning_suite_l.rs` — L1/L2 pass as smoke tests, L3/L4 deleted, L5 fails
- Review the Simulator's Q-bias bootstrap bug (Chapter 17, `17_fa_pathologies.md`) until you understand every detail: why every committed action's Q saturated toward `w_alive/(1−γ) ≈ 8.5`, why the agent locked onto its first action, and why removing the `w_alive` baseline (reward is now the per-tick drive-delta `R = cost(s_prev) − cost(s)`) resolved it.

#### Exercises
1. **The big one:** prototype Fix 1 (zero `w_alive`) in `crates/sim/sim_config/src/reward.rs`. Set `w_alive = 0.0`. Rerun `l1_agent_learns_to_plant`. Does the chain close?
2. Alternatively, prototype Fix 3 (dueling/advantage): add a `V(s)` head to the learner and use `Q − V` for argmax in `policy.rs`. Compare.
3. Write up your findings as a design doc.

#### Time
~20 hr. This week is your project's RL inflection point.

---

## Part XV — Frontier Topics & Capstone (Weeks 21-22)

### Week 21 — Frontier (elective deep dive)

Pick ONE of the following and read deeply (4-6 papers).

#### Option A: Meta-RL & Few-Shot Learning
- **Finn, Abbeel, Levine 2017** — MAML. [arxiv.org/abs/1703.03400](https://arxiv.org/abs/1703.03400)
- **Duan et al. 2016** — RL² (learning to learn). [arxiv.org/abs/1611.02779](https://arxiv.org/abs/1611.02779)
- **Wang et al. 2016** — Learning to RL. [arxiv.org/abs/1611.05763](https://arxiv.org/abs/1611.05763)

#### Option B: RLHF / Alignment
- **Christiano et al. 2017** — RLHF foundations.
- **Stiennon et al. 2020** — Learning to summarize from human feedback. [arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325)
- **Ouyang et al. 2022** — InstructGPT. [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- **Rafailov et al. 2023** — DPO. [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

#### Option C: Control as Inference & MaxEnt RL
- **Levine 2018** — RL and Control as Probabilistic Inference. [arxiv.org/abs/1805.00909](https://arxiv.org/abs/1805.00909)
- **Ziebart et al. 2008** — MaxEnt IRL.
- **Haarnoja et al. 2017** — Soft Q-learning. [arxiv.org/abs/1702.08165](https://arxiv.org/abs/1702.08165)

#### Option D: Inverse RL & Imitation
- **Abbeel & Ng 2004** — Apprenticeship learning. [PDF](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- **Ho & Ermon 2016** — GAIL. [arxiv.org/abs/1606.03476](https://arxiv.org/abs/1606.03476)
- **Fu, Luo, Levine 2018** — AIRL. [arxiv.org/abs/1710.11248](https://arxiv.org/abs/1710.11248)

### Week 22 — Capstone: Rewrite your cognition layer

The synthesis week.

#### Project deliverable
Apply what you've learned. Pick one major refactor:

1. **Fix the Q-bias bootstrap.** Implement Fix 1 (zero `w_alive` + redefine R as delta reward) or Fix 3 (dueling architecture). Rerun the L-suite. Document changes.
2. **Add proper exploration.** Implement UCB or pseudo-counts in your policy. Demonstrate it breaks the Plant-lock-in.
3. **Implement an actor-critic policy.** Linear actor over your tile features + critic for V(s). Replace the score-based argmax. Document tradeoffs.
4. **Implement eligibility traces.** TD(λ) in your learner. Document how it changes credit propagation across the Plant→Consume chain.
5. **Implement a proper option/recipe-value layer.** Restore SMDP-Q for recipes (the deleted version from earlier sessions). Document why a tabular Q over options is harder to converge than primitive Q.

Write the work up as a docs/ proposal document. Cite relevant papers from your reading. Commit and let your future self (and your collaborators) benefit from your master's-degree-level grasp of the territory.

---

## Appendix A — Glossary of every term you'll see

| Term | Definition |
|---|---|
| **MDP** | Markov Decision Process: tuple `(S, A, P, R, γ)` |
| **Bellman equation** | Recursive relation defining V or Q |
| **TD error** | `δ = r + γ·V(s') − V(s)`, the "surprise" |
| **On-policy** | Learns the value of the behavior policy |
| **Off-policy** | Learns the value of a *different* policy than the behavior one (e.g., greedy from ε-greedy data) |
| **Bootstrap** | Update an estimate using another estimate |
| **Function approximation** | Parameterize V or Q as a function instead of a table |
| **Tile coding** | Linear FA with overlapping discretizations as features |
| **Deadly triad** | Bootstrap + FA + off-policy → potential divergence |
| **Advantage** | `A(s, a) = Q(s, a) − V(s)` |
| **Policy gradient** | Direct optimization of `π_θ` via REINFORCE-like updates |
| **Actor-critic** | Hybrid: parameterize both `π` (actor) and `V` or `Q` (critic) |
| **Trust region** | Constrain policy update size via KL-divergence (TRPO, PPO) |
| **Importance sampling** | Reweight off-policy samples by `π/μ` ratio |
| **Option (HRL)** | Sub-policy with initiation set + internal π + termination |
| **POMDP** | Partially-Observed MDP: agent sees only `o ~ O(o|s)` |
| **Regret** | `Σ_t (V*(s_t) − V^π(s_t))` — how much you lose by not being optimal |
| **PAC** | Probably Approximately Correct: how many samples for ε-optimal w.p. 1-δ |
| **MaxEnt RL** | Maximize `E[Σ r + α H(π)]` for entropy-regularized policies |
| **GAE** | Generalized Advantage Estimation: λ-weighted multi-step advantage |
| **RLHF** | RL from Human Feedback: train a reward model from preferences, then RL it |

---

## Appendix B — Mapping every project file to syllabus topic

| File | Topic | Week |
|---|---|---|
| `crates/engine/q_learning/src/lib.rs` | Q-learning core | 6 |
| `crates/engine/q_learning/src/learning_rate.rs` | TD(0) with α/γ override | 6 |
| `crates/engine/q_learning/src/action_template.rs` | Action space `A` | 3 |
| `crates/engine/q_learning/src/tile_coding/` | Tile-coded features | 7 |
| `crates/engine/q_learning/src/observation_tile_coder.rs` | Feature extractor | 7 |
| `crates/cognition/planner/src/policy.rs` | Argmax + ε-greedy | 5, 6 |
| `crates/cognition/planner/src/perception.rs` | Observation construction | 3 (POMDP) |
| `crates/sim/sim_config/src/learning.rs` | Hyperparameters | 6, 7 |
| `crates/sim/sim_config/src/reward.rs` | `R(s)` | 3 |
| `crates/sim/app/tests/tasks/hungry_consume.rs` | Q-learning on the hunger MDP | 6, 12 |
| `crates/sim/app/tests/tasks/threat_response.rs` | Advantage-style margin metric | 12 |
| `crates/sim/app/tests/tasks/navigation.rs` | Sparse-reward navigation | 6, 16 |
| `crates/sim/app/tests/curricula/long_horizon_harvest.rs` | Deferred reward + Q-bootstrap pathology | 14, 20 |
| `crates/sim/app/tests/pathologies/chaos_and_sabotage.rs` | Convergence guards | 19 |
| `docs/CHANGELOG.md` ("Forward-search planner removed → flat Q-learning policy") | Model-based RL — what was tried, why it was retired in favour of model-free TD | 15 |

---

## Appendix C — Reading load summary

| Phase | Weeks | Textbook chapters | Papers | Estimated hours |
|---|---|---|---|---|
| Foundations | 0-2 | ~12 | 0 | ~37 |
| MDPs & DP | 3-4 | ~10 | 0 | ~27 |
| Tabular RL | 5-6 | ~7 | 6 | ~33 |
| Function Approx | 7-8 | ~6 | 4 | ~30 |
| Deep Q | 9-10 | ~3 | 8 | ~28 |
| Policy Gradient & Actor-Critic | 11-12 | ~4 | 4 | ~33 |
| TRPO/PPO + Continuous | 13-14 | ~2 | 7 | ~24 |
| Model-Based | 15 | ~2 | 6 | ~15 |
| Exploration | 16 | ~3 | 4 | ~12 |
| Offline / Imitation / RLHF | 17 | ~1 | 6 | ~10 |
| Hierarchical / Multi-Agent | 18 | ~2 | 6 | ~12 |
| Theory | 19 | ~6 | 3 | ~15 |
| Pathologies + Project | 20 | ~1 | 3 | ~20 |
| Frontier + Capstone | 21-22 | 0 | 5+ | ~25 |
| **Total** | **~22** | **~59 chapters** | **~62 papers** | **~321 hours** |

At 15 hr/week, this is ~21 weeks of dedicated study. At 10 hr/week, ~32
weeks. Pace as you wish.

---


---

---

## What this addendum addresses

A careful read of the project's design corpus
(`docs/designs/04_cognition.md`, `docs/designs/cognitive_decision_model.md`,
14 proposal docs, the hardcoding audit, the test audit) surfaced
~70 cognition-relevant technical topics. The core syllabus covered
maybe half. This addendum covers the rest, organized into five new
parts:

| Part | Topics | New weeks |
|---|---|---|
| **XVI** | Feature engineering & state representation | 23-25 |
| **XVII** | Memory & retrieval in RL | 26 |
| **XVIII** | State similarity, aggregation & sparse FA | 27 |
| **XIX** | Action spaces beyond discrete | 28 |
| **XX** | Long-horizon credit assignment (advanced) | 29 |
| **XXI** | Project-specific synthesis | 30-32 |

Plus updates to existing weeks 6, 7, 8, 12, 14, 15, 16, 18 with
additional readings.

---

# Part XVI — Feature Engineering & State Representation (Weeks 23-25)

The single biggest gap in the original syllabus. Function approximation
weeks (7-8) introduced tile coding but didn't go deep on basis
construction, learned representations, or the project's specific
"199-dim observation" problem.

## Week 23 — Tile Coding & CMAC deeply

### Objectives
Master tile coding as the live function approximator your project uses.
Understand its theory (CMAC origins), practical parameter choice, and
its specific failure modes on the high-dimensional, multi-block
observations the cognition pipeline assembles.

### Reading
- **Sutton & Barto Ch. 9.5** — Coarse Coding, Tile Coding (re-read)
- **Albus 1975** — Original CMAC paper. [NIST](https://www.nist.gov/publications/new-approach-manipulator-control-cerebellar-model-articulation-controller-cmac1). The cerebellum-inspired sparse hashed feature representation that tile coding is.
- **Sutton 1996** — *Generalization in RL: Successful Examples Using Sparse Coarse Coding*. [NIPS PDF](http://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding). The empirical paper that made tile coding canonical.
- **Sherstov & Stone 2005** — Function Approximation via Tile Coding: Automating Parameter Choice. [PDF](http://web.cs.ucla.edu/~sherstov/pdf/sara05-tiling.pdf).
- **Whiteson, Taylor & Stone 2007** — Adaptive Tile Coding. [PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/whitesontr07.pdf). Adaptively splits tiles where the value function is steep.

### The math that matters
- **CMAC binding:** input `x ∈ ℝ^d` → set of `k` active tiles (one per tiling). Sparse: `k` active out of `n` total ⇒ each `Q(s, a)` is the sum of `k` per-action weights.
- **Tile width vs number of tilings:** broader tiles + more tilings ⇒ stronger generalization. Narrower tiles + fewer tilings ⇒ finer discrimination.
- **Hashing rule of thumb:** "8-32 tilings, each at the resolution you'd want a single grid to have; randomize offsets; hash if the Cartesian product is too large" (Sutton 1996, Whiteson 2007).
- **Asymmetric tilings:** different dimensions get different tile widths based on importance / scale.
- **The "joint-tiling collapse":** with one tiling spanning all 199 dims, a shared tile requires *all 199 coordinates* to align. Generalization degrades to lookup. This is exactly what your sim hits (see Project tie-in).

### Project tie-in
**Study this sim's observation schema and its per-block tile coder ([`observation_tile_coder.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/observation_tile_coder.rs)).** Key facts to internalize:
- 199-dim observation = 14 drives + 8 body + 8 emotions + 12 perception + 5 world + 8 ambient + **144 episodic memory (16 slots × 9 fields)**.
- The memory block is **72% of the observation** — and it's the source of most pathologies.
- **The "set-as-vector" problem:** episodic memory slots are a *set* but stored as a salience-sorted ordered vector. A tiny salience change discontinuously swaps slots — destroys tile-coding generalization.
- **Noise contamination across blocks:** joint tiling makes the whole observation only as stable as its noisiest block. Memory churn poisons learning on the stable drive block.
- **The proposed fix:** **per-block tilings with per-block IHT slices**. Each block gets its own tile coder; hash-namespace separation prevents cross-block bleeding.

Read [`crates/engine/q_learning/src/tile_coding/`](https://github.com/falahat/simulator/blob/main/crates/engine/q_learning/src/tile_coding) — the actual implementation. Note:
- `iht_size = 2^16` (65536 slots in the hash table)
- Per-block tilings (recently implemented to address the noise-contamination issue)
- Tile widths per dimension are hand-tuned

### Exercises
1. Implement tile coding for the 2D mountain-car problem. Sweep tile widths and number of tilings. Plot value functions and learning curves.
2. **Project exercise:** add a diagnostic that reports per-block tile activation patterns over a 1000-tick run. Are some blocks doing all the discrimination work? Are some blocks invariant (uninformative)?
3. Construct the "joint-tiling collapse" empirically: take your 199-dim observation, encode it with (a) one global tiling of width 0.1 and (b) per-block tilings of width 0.1. Compute the Hamming distance of the active tile set across two ticks where only 1 drive value changes. The global version should change all `n` tilings; the per-block version only the drive's block.

### Time
~12 hr.

---

## Week 24 — Linear basis construction beyond tiles

### Objectives
The wider toolkit of linear function approximators. Understand when
Fourier basis, RBF, polynomial, BEBF, or proto-value functions
outperform tile coding.

### Reading
- **Sutton & Barto Ch. 9.5-9.6** — Fourier, RBF, polynomial features
- **Konidaris, Osentoski, Thomas 2011** — *Value Function Approximation in Reinforcement Learning Using the Fourier Basis*. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/7903). Shows a simple cosine basis dominates polynomials and RBFs on classic control.
- **Mahadevan & Maggioni 2007** — *Proto-value Functions: A Laplacian Framework for Learning Representation and Control in MDPs*. [PDF](https://scholarworks.umass.edu/cs_faculty_pubs/875/). Basis from eigenvectors of the state-transition graph's Laplacian.
- **Parr et al. 2007** — *Analyzing Feature Generation for Value-Function Approximation*. [PDF](https://users.cs.duke.edu/~parr/icml07.pdf). **Bellman-Error Basis Functions (BEBFs)** — orthogonal features that provably tighten approximation error by factor γ per addition.

### The math that matters
- **Fourier basis:** `φ_k(x) = cos(π · c_k · x)` for integer coefficient vectors `c_k`. Order-N basis has `(N+1)^d` features. Pair with low-coupling (only use `c_k` with few non-zero components for `d > 4`).
- **RBF basis:** `φ_i(s) = exp(−‖s − μ_i‖² / 2σ²)`. Smooth interpolation, sensitive to bandwidth `σ`.
- **BEBF construction:** start with arbitrary features `{φ_1}`. Compute Bellman residual under linear LSTD solution. The residual is the new feature `φ_2`. Iterate. Each new feature reduces approximation error by γ.
- **Proto-value functions:** `φ_i(s) = i`-th eigenvector of graph Laplacian `L`. Smooth, basis-aligned with the MDP's natural geometry.

### Project tie-in
- Your project's observation is **continuous + categorical mix** — well-suited to tile coding but the categorical features (memory `kind` enum, perception modality) are awkward.
- An **entity embedding** over the memory `kind` enum (Guo & Berkhahn 2016; word2vec-style) would give a learned dense representation. (This was surveyed for the project but cut as out-of-scope — the sim stays tile-coded/interpretable; it remains a worthwhile academic study topic.)
- **Fourier basis over continuous drives:** would be a small targeted experiment. Just the 14-dim drive block, Fourier order 3, vs current tile coding. Sample efficiency comparison.

### Exercises
1. Implement Fourier basis on mountain-car. Compare to tile coding from Week 23.
2. Implement BEBF construction on a 5x5 gridworld. Show error reduction per added feature.
3. **Project exercise:** prototype Fourier basis encoding for the 14-dim drive block. Hook it in behind an `ObservationEncoder` seam (a sketch — the sim itself stays tile-coded). Run `learning_homeostatic` and compare.

### Time
~12 hr.

---

## Week 25 — Learned Representations

### Objectives
Move beyond hand-engineered features. Understand auxiliary tasks,
contrastive representation learning, and bisimulation-metric
representations.

### Reading
- **Sutton & Barto Ch. 9.7-9.8** — Nonlinear function approximation
- **Jaderberg et al. 2017** — UNREAL: *Reinforcement Learning with Unsupervised Auxiliary Tasks*. [arxiv.org/abs/1611.05397](https://arxiv.org/abs/1611.05397). Augments A3C with pixel-control, reward-prediction, value-replay tasks. ~10× speedup on Atari.
- **Srinivas, Laskin, Abbeel 2020** — CURL: *Contrastive Unsupervised Representations for RL*. [arxiv.org/abs/2004.04136](https://arxiv.org/abs/2004.04136). InfoNCE contrastive loss on augmented image pairs.
- **Stooke et al. 2021** — ATC: *Decoupling Representation Learning from RL*. [arxiv.org/abs/2009.08319](https://arxiv.org/abs/2009.08319). Augmented Temporal Contrast — encoder trains separately from policy, competes with end-to-end RL.
- **Zhang et al. 2021** — DBC: *Learning Invariant Representations for RL without Reconstruction*. [arxiv.org/abs/2006.10742](https://arxiv.org/abs/2006.10742). Deep bisimulation metric — latent distance matches behavioral distance, robust to visual distractors.

### The math that matters
- **Auxiliary loss:** `L_total = L_RL + λ · L_aux`. Auxiliary task shares the encoder with the policy/value head, regularizing the representation.
- **Contrastive loss (InfoNCE):** `L = −log(exp(sim(q, k+)/τ) / Σ_k exp(sim(q, k)/τ))`. Pull positives together, push negatives apart in latent space.
- **Bisimulation metric:** `d(s, s') = max_a (|R(s,a) − R(s',a)| + γ W(P(·|s,a), P(·|s',a)))` where `W` is the Wasserstein distance between next-state distributions. **Two states are bisimilar iff they're behaviorally identical.**
- **Deep Bisimulation for Control (DBC):** train an encoder so `‖φ(s) − φ(s')‖ ≈ d(s, s')`. The latent distance matches behavioral distance.

### Project tie-in
- The learned-representation spectrum is this week's academic deep dive:
  - **General Value Functions / Horde** (Sutton 2011) — each feature is a learned predictive question
  - **Successor features** (Dayan 1993; Barreto 2017)
  - **Object-centric encoders** (Slot Attention)
  - **Concept-bottleneck / neurosymbolic** (Delfosse 2024)
  - **Bisimulation-metric** (Zhang 2021)
- For the project specifically, these *learned* directions were surveyed and **cut** as out-of-scope (the sim stays tile-coded / interpretable / deterministic). What *did* survive as a buildable idea is an **`ObservationEncoder` seam + hand-engineered typed pooling** — `FlatSlotEncoder` (baseline) vs a `PooledMemoryEncoder`, A/B-tested against the live tile coder. Such a seam is precisely what would let a learned encoder drop in later, if that decision were ever revisited.
- Determinism (the canary) is why the learned directions are risky for this sim: any learned encoder must stay a deterministic function of the seeded run — pre-trained-frozen vs. jointly-trained is the real tradeoff, and a reason the project stopped at the interpretable cut.

### Exercises
1. Implement CURL on Cartpole-from-pixels. Compare sample efficiency to plain SAC.
2. Read the DBC paper carefully. Note how it uses the Wasserstein distance to define behavioral similarity.
3. **Project exercise (significant):** prototype a `PooledMemoryEncoder` that aggregates the 16 memory slots into a kind-keyed permutation-invariant summary (count, max-salience, min-age per kind). Replace `FlatSlotEncoder` and run `learning_homeostatic`. Plot whether the noise-contamination effect (the joint-tiling collapse from Chapter 10) is reduced.

### Time
~15 hr.

---

# Part XVII — Memory & Retrieval in RL (Week 26)

## Week 26 — Episodic memory and non-parametric value

### Objectives
Understand how DRL agents use memory: parametric (DNCs, MERLIN),
non-parametric (MFEC, NEC), and retrieval-augmented architectures.
This is directly applicable to your sim's two-tier memory (Episodic +
Consolidated).

### Reading
- **Graves et al. 2016** — *Hybrid computing using a neural network with dynamic external memory* (DNC). *Nature* 538:471-476.
- **Blundell et al. 2016** — MFEC: *Model-Free Episodic Control*. [arxiv.org/abs/1606.04460](https://arxiv.org/abs/1606.04460). Nonparametric Q-table keyed by random projections or VAE features. Beats DQN early in training.
- **Pritzel et al. 2017** — NEC: *Neural Episodic Control*. [arxiv.org/abs/1703.01988](https://arxiv.org/abs/1703.01988). Differentiable-neural-dictionary version of MFEC.
- **Wayne et al. 2018** — MERLIN: *Unsupervised Predictive Memory in a Goal-Directed Agent*. [arxiv.org/abs/1803.10760](https://arxiv.org/abs/1803.10760). Memory-based predictor + policy for partial observability.
- **Lin, Zhou, Zaheer, Smola 2018** — *Episodic Memory Deep Q-Networks* (EMDQN). [arxiv.org/abs/1805.07603](https://arxiv.org/abs/1805.07603).

### Set encodings and permutation invariance
A subset of memory + perception are **sets** (a bag of percepts, a bag of recent memories). Standard tile coding treats them as ordered vectors, which breaks generalization. The fix:

- **Zaheer et al. 2017** — *Deep Sets*. [arxiv.org/abs/1703.06114](https://arxiv.org/abs/1703.06114). The fundamental theorem: any permutation-invariant function on sets has the form `ρ(Σ_i φ(x_i))`. Sum-pool the elementwise features, then map.
- **Lee et al. 2019** — *Set Transformer*. [arxiv.org/abs/1810.00825](https://arxiv.org/abs/1810.00825). Attention over set elements without positional encoding; richer than sum-pooling.

### The math that matters
- **MFEC update:** `Q^MFEC(s, a) := max(Q^MFEC(s,a), G_t)` (no averaging — always take the *best* observed return). Plus k-NN over recent (state, action) pairs for state generalization.
- **Deep Sets theorem:** `f(X) = ρ(Σ_{x∈X} φ(x))` is the universal form for permutation-invariant functions on countable sets.

### Project tie-in
- **The set-as-vector problem is exactly what Deep Sets solves.** Your 16 memory slots are a set; the current observation treats them as an ordered list (salience-sorted). A **typed permutation-invariant pooling** of the memory block is the buildable, interpretable version of this — a per-kind aggregate (count/max-salience/min-age) rather than a learned pooler. (A *learned* entity embedding over the `kind` enum was surveyed and cut as out-of-scope; the hand-engineered pooling is what's proposed for the sim.)
- The two-tier memory (Episodic + Consolidated) you have is conceptually similar to MERLIN's predictive memory + policy split.

### Exercises
1. Implement MFEC on Cartpole. Compare to DQN over the first 10k steps. (MFEC should dominate early.)
2. Implement a Deep Sets pooler in PyTorch on a synthetic "bag of objects" task. Train, then permute test inputs and verify outputs are invariant.
3. **Project exercise:** sketch a Deep-Sets-based `PooledMemoryEncoder` for your sim's memory block. The 16-slot input goes through a per-slot MLP (shared weights), sum-pool, then a final MLP produces the memory-block feature vector. This replaces the current 144-dim ordered slot encoding with a permutation-invariant fixed-size representation.

### Time
~15 hr.

---

# Part XVIII — State Similarity, Aggregation & Sparse FA (Week 27)

## Week 27 — When are two states the same? Plus feature selection.

### Objectives
The theory of "this state is like that state" and how to exploit it.
Three threads: state abstraction theory, MDP homomorphisms, and sparse
feature selection.

### Reading
- **Li, Walsh, Littman 2006** — *Towards a Unified Theory of State Abstraction for MDPs*. [PDF](http://rbr.cs.umass.edu/aimath06/proceedings/P21.pdf). Five canonical abstraction schemes (model-, Q*-, Q^π-, a*-, π-equivalent) and what each preserves.
- **Ferns, Panangaden, Precup 2004** — *Metrics for Finite MDPs*. [arxiv.org/abs/1207.4114](https://arxiv.org/abs/1207.4114). Bisimulation pseudometric, formal foundation for "behavioral similarity."
- **Givan, Dean, Greig 2003** — *Equivalence Notions and Model Minimization in MDPs*. The structural-equivalence foundation.
- **Ravindran & Barto 2003** — *SMDP Homomorphisms*. [PDF](https://www.ijcai.org/Proceedings/03/Papers/145.pdf). Structure-preserving maps justify state aggregation; foundation for relativized options.
- **Ormoneit & Sen 2002** — KBRL. [Springer](https://link.springer.com/article/10.1023/A:1017928328829). Kernel-based RL — replaces Bellman backup with kernel-weighted average over sampled transitions. Provably consistent.
- **Ernst, Geurts, Wehenkel 2005** — FQI: *Tree-Based Batch Mode RL*. [JMLR](https://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf).
- **Kolter & Ng 2009** — LARS-TD / LASSO-TD. [PDF](http://www.cs.cmu.edu/~zkolter/pubs/kolter-icml09b-full.pdf). L1-regularized linear TD — robust to thousands of irrelevant features.

### The math that matters
- **Bisimulation:** states `s, s'` are bisimilar iff for every action `a`: (1) `R(s, a) = R(s', a)`; (2) `P(B | s, a) = P(B | s', a)` for every equivalence class `B`. Recursively defined.
- **Bisimulation metric:** the continuous relaxation `d(s,s') = max_a [|R(s,a)-R(s',a)| + γ W(P(·|s,a), P(·|s',a))]` where `W` is the Kantorovich metric.
- **Approximate abstraction (Abel 2016):** if two states are ε-bisimilar, value functions differ by at most `ε/(1-γ)`. Bounds the cost of aggregation.
- **LASSO-TD:** `θ = argmin_θ ‖Aθ − b‖² + λ‖θ‖_1` where `(A, b)` are the LSTD linear system. L1 sparsity selects relevant features.

### Project tie-in
- This week's state-similarity theory was surveyed for the project; the graded-similarity directions (bisimulation metrics, learned state-abstraction) were **cut** as out-of-scope, but the one buildable idea — relational pooling of perceived objects into type histograms (see the project tie-in below) — survives.
- The bias-variance tension still applies: *over-aggressive abstraction → bias: the agent confidently learns the wrong value for both, and no amount of data fixes it* — which is why the surviving idea ships with a focused "don't collapse look-alike-but-behave-different states" test.
- **The "2 trees vs 3 trees" generalization case** (Diuk-Cohen-Littman 2008 — relational MDPs) is the motivating example: an agent that has seen 2 trees should treat 3 trees similarly. The current observation lacks this (discrete entity counts, no "trees as a class"); the buildable fix is **relational pooling of perceived objects into type histograms**, so count-generalisation becomes structural.
- LASSO-TD applied to your 199-dim observation would identify which dimensions actually carry value-relevant signal. **This is testable: implement L1-regularized linear Q-learning over your tile features, log which weights are zeroed out.** Likely candidates: most memory-slot dimensions (which is why pooling helps).

### Exercises
1. Implement bisimulation metric on a small MDP. Verify two structurally-identical states get `d = 0`.
2. Implement LARS-TD on a 5x5 gridworld with 1000 noise features. Show it identifies the few real features.
3. **Project exercise:** instrument your tile-coded Q-learner to track per-feature `‖θ‖_∞` across actions. Run for a long episode. Which tiles never accumulate weight? Those are candidates for pruning.

### Time
~15 hr.

---

# Part XIX — Action Spaces Beyond Discrete (Week 28)

## Week 28 — Parameterized, hybrid, large, and structured actions

### Objectives
Move beyond "16 discrete primitives." Real grounded agents act with
intensity, direction, magnitude. The literature on this is rich and
directly applicable to your sim's `Strike { force }`, `Vocalize { volume }`,
`Step { direction }` patterns.

### Reading
- **Masson, Ranchod, Konidaris 2016** — PA-DDPG (Q-PAMDP). [arxiv.org/abs/1509.01644](https://arxiv.org/abs/1509.01644). Foundational PAMDP paper. Alternating optimization of discrete action class + continuous params.
- **Hausknecht & Stone 2016** — *Deep RL in Parameterized Action Space*. [arxiv.org/abs/1511.04143](https://arxiv.org/abs/1511.04143). PA-DDPG — extends DDPG with bounded inverting gradients for the parameter space.
- **Xiong et al. 2018** — P-DQN: *Parametrized Deep Q-Networks*. [arxiv.org/abs/1810.06394](https://arxiv.org/abs/1810.06394). Combines DDPG and DQN: actor outputs best continuous parameters for each discrete action, Q-net scores the tuple.
- **Bester, James, Konidaris 2019** — MP-DQN: *Multi-Pass Q-Networks*. [arxiv.org/abs/1905.04388](https://arxiv.org/abs/1905.04388). Fixes a P-DQN bias by zeroing parameters of irrelevant action classes per forward pass.
- **Fan et al. 2019** — H-PPO: *Hybrid Actor-Critic RL in Parameterized Action Space*. [arxiv.org/abs/1903.01344](https://arxiv.org/abs/1903.01344). Parallel discrete + continuous heads under PPO. Simpler and often better than P-DQN.
- **Dulac-Arnold et al. 2015** — Wolpertinger: *Deep RL in Large Discrete Action Spaces*. [arxiv.org/abs/1512.07679](https://arxiv.org/abs/1512.07679). Actor outputs proto-action in embedding space, kNN retrieves discrete actions, critic re-ranks.
- **Tennenholtz & Mannor 2019** — Act2Vec: *The Natural Language of Actions*. [arxiv.org/abs/1902.01119](https://arxiv.org/abs/1902.01119). word2vec for actions; learn embeddings from demo trajectories.
- **Bacon, Harb, Precup 2017** — Option-Critic. [arxiv.org/abs/1609.05140](https://arxiv.org/abs/1609.05140). Policy gradients for both intra-option policies and termination functions.
- **Eysenbach et al. 2018** — DIAYN: *Diversity Is All You Need*. [arxiv.org/abs/1802.06070](https://arxiv.org/abs/1802.06070). Unsupervised skill discovery via mutual information.
- **Huang & Ontañón 2020** — *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. [arxiv.org/abs/2006.14171](https://arxiv.org/abs/2006.14171). Proves logits-to-−∞ masking is a valid policy gradient under a constrained MDP.

### The math that matters
- **PAMDP:** action is `(k, x_k)` where `k ∈ {1,...,K}` is discrete and `x_k ∈ ℝ^{d_k}` is continuous, with `d_k` possibly differing per `k`.
- **P-DQN forward:** `Q(s, k) = Q_θ(s, k, x_k^*(s))` where `x_k^*(s) = μ_φ(s)[k]` from the actor. Argmax over `k` after computing all `Q(s, k)`.
- **Action masking gradient:** `π(a|s) = exp(z_a − M_a) / Σ exp(z_a' − M_a')` where `M_a = 0` if legal, `∞` if illegal. Gradient through illegal actions is zero — exact policy gradient on the masked MDP.
- **Option-critic intra-option gradient:** `∇_θ J = E[∇_θ log π_o(a|s) (Q_o(s,a) − A_o(s,a))]` where `o` is the active option.

### Project tie-in
- Your project's [`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md) lists 15 primitive action templates. Several have *continuous parameters*:
  - `Strike { force }` (0..1)
  - `Vocalize { volume }`
  - `Step { direction }` (categorical — N/S/E/W — but conceptually continuous heading)
- These are **parameterized actions**. The current learner treats them as discrete (separate Q-keys for `Step_N` vs `Step_E`); a PAMDP approach would learn the parameter continuously.
- The broader project principle: action selection should be learned content, not Rust the developer writes (see [`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md) — the flat ε-greedy Q-policy embodies this). Learned policy gradient does the same job for continuous parameters.
- Your project's **recipes are options** — read carefully. `RecipeMeme` with multiple steps = `(I, π, β)` triple. The deleted SMDP-Q machinery was learning option values. Option-critic (Bacon et al. 2017) is the modern way to learn these end-to-end.
- **Action masking is needed** for your project: `Consume` requires food in reach, `Plant` requires a seed, `Strike` requires a target. These are exactly the legal-action-set constraints Huang & Ontañón formalize.

### Exercises
1. Implement P-DQN on the Goal-Soccer hybrid-action benchmark.
2. Implement DIAYN on a simple gridworld; visualize the discovered skills.
3. **Project exercise:** sketch how to convert `Strike { force }` to a parameterized action under P-DQN. Discrete head = `{Strike}`; continuous param = `force ∈ [0, 1]`. Critic Q(s, Strike, force). What changes in `policy.rs:466`?
4. **Project exercise:** implement action masking in your policy. Currently `enumerate_first_step_actions()` returns all 6 baseline actions even if some are illegal in the current state. Add a `is_legal(action, state)` check and mask invalid actions before the argmax.

### Time
~18 hr.

---

# Part XX — Long-Horizon Credit Assignment Advanced (Week 29)

## Week 29 — Beyond TD(λ) and HRL

### Objectives
The full modern toolkit for credit assignment over long horizons.
Eligibility traces are foundational but only the start. Modern methods
include hindsight relabeling, return decomposition, successor features
for transfer, and counterfactual credit assignment.

### Reading
- **van Seijen et al. 2016** — *True Online TD(λ)*. [arxiv.org/abs/1512.04087](https://arxiv.org/abs/1512.04087). Modernization of classical traces. Use this instead of accumulating/replacing traces.
- **Andrychowicz et al. 2017** — HER: *Hindsight Experience Replay*. [arxiv.org/abs/1707.01495](https://arxiv.org/abs/1707.01495). Goal relabeling — every "failed" trajectory toward `g` is a "successful" one toward some `g'`.
- **Ng, Harada, Russell 1999** — *Policy Invariance under Reward Transformations*. The PBRS theorem.
- **Munos et al. 2016** — Retrace(λ). [arxiv.org/abs/1606.02647](https://arxiv.org/abs/1606.02647). Safe efficient off-policy multi-step returns with clipped importance ratios.
- **Dayan 1993** — *Improving Generalization for TD Learning: The Successor Representation*. [MIT Press](https://direct.mit.edu/neco/article/5/4/613/5774). Originating insight: `V = M · R` where `M` is the discounted future-state-occupancy.
- **Barreto et al. 2017** — *Successor Features for Transfer in RL*. [arxiv.org/abs/1606.05312](https://arxiv.org/abs/1606.05312). Lifts SR to deep features + GPI (Generalized Policy Improvement).
- **Arjona-Medina et al. 2019** — RUDDER: *Return Decomposition for Delayed Rewards*. [arxiv.org/abs/1806.07857](https://arxiv.org/abs/1806.07857). LSTM predicts episode return; differences give dense reward decomposition with same optimal policy. Game-changer for delayed reward.
- **Harutyunyan et al. 2019** — *Hindsight Credit Assignment*. [arxiv.org/abs/1912.02503](https://arxiv.org/abs/1912.02503). Posterior credit assignment — what is the probability that action a led to this observed outcome?
- **Nachum et al. 2018** — HIRO: *Data-Efficient HRL*. [arxiv.org/abs/1805.08296](https://arxiv.org/abs/1805.08296).
- **Levy et al. 2019** — HAC: *Multi-Level Hierarchies with Hindsight*. [arxiv.org/abs/1712.00948](https://arxiv.org/abs/1712.00948).

### The math that matters
- **HER goal relabeling:** for trajectory `(s_0, a_0, ..., s_T)` toward goal `g`, replace `g` with any visited state `s_k`. The new trajectory is a "successful" example. Combined with off-policy algorithm.
- **PBRS (Ng, Harada, Russell):** `F(s,s') = γΦ(s') − Φ(s)` is the *unique* additive shaping that preserves the optimal policy. Subtract this from your rewards to provide dense signal without changing what's optimal.
- **Retrace(λ):** target = `r + γ E_a[Q(s', a)] + γ c · (Q(s',a') − E_a[Q(s',a)])` where `c = λ · min(1, π/μ)`. Clipped IS ratio.
- **Successor features:** decompose `r = φ · w` (linear). Learn `ψ^π(s,a) = E[Σ γ^t φ_t]`. Then `Q^π(s,a) = ψ^π(s,a) · w`. Changing `w` (new task) doesn't require relearning `ψ`.
- **GPI:** given a set of policies `{π_i}` and their successor features `{ψ_i}`, define a new policy `π(s) = argmax_a max_i ψ_i(s,a) · w` for any new reward weights `w`. Beats every `π_i`.
- **RUDDER:** train LSTM `g(τ)` to predict `R_total` from trajectory `τ = (s_1, a_1, ..., s_T)`. Decomposed reward `r'_t = g(τ_{1:t}) − g(τ_{1:t-1})`. Provably same optimal policy, exponentially lower variance.

### Project tie-in
- **HER is directly applicable.** Your agents pursue goals (reach food, plant seed, etc.). Failed trajectories can be relabeled. Significant engineering lift to add to your harness.
- **Potential-based reward shaping is THE TOOL for fixing your `w_alive`-bootstrap pathology** in a principled way: instead of dropping `w_alive`, treat the alive baseline as a shaping potential `Φ(s) = constant`. Then PBRS says it doesn't change optimal policy — but it DOES change Q values' baseline. **However**, the bug is precisely that the baseline pollutes argmax. PBRS only preserves optimal *value-iteration* policies, not value-bootstrapped argmax with FA. Read the Ng-Harada-Russell paper carefully and decide.
- **Successor features are conceptually relevant** — your project has multiple agents with different reward weightings (different personalities, different drives). SF + GPI would let one shared dynamics-model serve all agents. **The deleted successor-features code was an attempt at this.** Re-read the git history.
- **RUDDER for the planting chain:** if you can train an LSTM to predict "given this trajectory, total reward at the end is X", then differences give per-step rewards that propagate back to Plant. **This is the technique that mathematically might let L1 work.**

### Exercises
1. Implement HER + DDPG on the bit-flipping environment from the HER paper. Show learning emerges with sparse reward only via relabeling.
2. Implement PBRS on a long-corridor gridworld where reward is at the end. Use `Φ(s) = -dist_to_goal`. Compare convergence.
3. **Project exercise (significant):** implement RUDDER's return-redistribution on a synthetic version of L1. Train a simple LSTM `g(τ) → total_reward`. Use `r'_t = g(τ_{1:t}) − g(τ_{1:t-1})` as the dense reward. Plug into your Q-learner. Does Q(Plant) now climb?

### Time
~18 hr.

---

# Part XXI — Project-Specific Synthesis (Weeks 30-32)

The final three weeks integrate everything you've learned and apply it
to your sim's specific topics: homeostatic reward, the affect/drive
architecture, the determinism discipline, and the open Q-bias bug.

## Week 30 — Homeostatic RL + MORL + the Q-bias bootstrap

### Objectives
The project-specific RL theory: drive-based reward, multi-objective
scalarization, and the live open bug.

### Reading
- **Keramati & Gutkin 2011** — *A Homeostatic Reinforcement Learning Theory*. [PDF](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3211123/). The foundational homeostatic RL paper. Drives as convex cost functions. Wanting-vs-liking dissociation.
- **Keramati & Gutkin 2014** — Updated homeostatic RL framework. [eLife](https://elifesciences.org/articles/04811).
- **Berridge & Robinson 1998, 2016** — *Wanting vs Liking*. The neuroscience foundation behind the design: Seeking drive = wanting; Joy spike = liking.
- **Roijers, Vamplew, Whiteson, Dazeley 2013** — *A Survey of Multi-Objective Sequential Decision-Making*. [JAIR](https://www.jair.org/index.php/jair/article/view/10836). The MORL survey.
- **Pareto-front theory in MORL** — read the survey above plus your own [`pareto.rs`](https://github.com/falahat/simulator/blob/main/crates/cognition/planner/src/pareto.rs) (now removed but in git history).
- **Chapter 17 (`17_fa_pathologies.md`)** — the Simulator's Q-bias bootstrap bug, cover to cover. Now resolved: the `w_alive` baseline was removed in favour of the per-tick drive-delta reward `R = cost(s_prev) − cost(s)`.
- **[`crates/sim/sim_config/src/reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)** + the [`docs/CHANGELOG.md`](https://github.com/falahat/simulator/blob/main/docs/CHANGELOG.md) "Forward-search planner removed" entry — for the project's reward design choices (`w_alive`, MORL, level-vs-delta).

### The math that matters
- **Homeostatic reward:** `R(s) = w_alive − Σ_d w_d · cost_d(drive_d) − cost_bio`. Convex `cost_d(x) = x^p` for `p > 1`.
- **Level vs delta form:** `R(s)` (state-only) vs `cost(s) − cost(s')` (delta). Differ by potential-shaping term (Ng-Harada-Russell again).
- **MORL scalarizations:**
  - Linear: `R = w · r` for fixed weights `w`. Reaches only convex part of Pareto front.
  - Tchebysheff: `R = -max_d w_d · |r_d - z_d^*|`. Reaches all Pareto front.
  - Smoothed Tchebysheff: smooth max via softmax for differentiability.
- **The Q-bias bootstrap math:** see Chapter 17 (`17_fa_pathologies.md`) — under the old constant `w_alive` floor, every committed action's Q saturated to `w_alive / (1−γ)` regardless of which action (now resolved by switching to the per-tick drive-delta reward `R = cost(s_prev) − cost(s)`).

### Project tie-in
- This entire week is reading your own project's design. The Q-bias bootstrap bug (Chapter 17, now resolved by removing `w_alive` in favour of the drive-delta reward) + the predictive planning proposal + the deleted Pareto code are the live case study.
- **You will be able to apply Fix 1 (zero `w_alive`) by the end of this week.** Implement, run `l1_agent_learns_to_plant`, observe.

### Exercises
1. Read Keramati-Gutkin 2011 in full. Connect their model to your `PrimaryReward`.
2. Implement Fix 1 in [`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs): set `w_alive = 0.0`. Document downstream effects on every test that asserts a reward floor.
3. Alternative: implement Fix 3 (dueling/advantage architecture in your learner). Add a `V(s)` head, change argmax to use `Q − V`. Compare.
4. **Project exercise:** write up your findings as `docs/proposals/q_bias_fix_v2.md`. Cite the specific RL papers (Wang dueling, advantage learning, A2C) that ground your choice.

### Time
~20 hr.

---

## Week 31 — Cognition pipeline: affect, perception, drives, beliefs, attention

### Objectives
The non-RL machinery surrounding the learner. These are the
representations the RL operates over.

### Reading from the project corpus
- **[`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md)** — cover to cover. The cognition pipeline architecture.
- **[`docs/designs/cognitive_decision_model.md`](https://github.com/falahat/simulator/blob/main/docs/designs/cognitive_decision_model.md)** — eight cognition pathways: drive→score, emotion→score, emotion→drive accumulation, drive→precondition, drive→MORL, drive→attention, effects→bridge modulation, personality→multipliers.
- **[`docs/RESEARCH.md`](https://github.com/falahat/simulator/blob/main/docs/RESEARCH.md) §Drive taxonomy** — the 8 cognitive drives (Seeking, Lust, Vigilance, Resentment, Care, Loneliness, Boredom, Revulsion) + 6 metabolic, with citations. **Cognitively writable** drives can be modulated by memes; metabolic drives are biology-only.
- **[`docs/proposals/first_person_beliefs.md`](https://github.com/falahat/simulator/blob/main/docs/proposals/first_person_beliefs.md)** — beliefs as views over memory. Bayesian update. Backward forensic inference.
- **[`crates/cognition/attention/`](https://github.com/falahat/simulator/blob/main/crates/cognition/attention/src/salience.rs)** — the salience pipeline source (RON-loadable `SalienceConfig`). Its grounding — Bruner-Goodman 1947 value-and-need perception, Mogg-Bradley 1998 threat amplification — is in `cognitive_decision_model.md` (above).

### Academic foundations
- **Damasio 1994** — *Descartes' Error*. Somatic markers. The somatic-marker hypothesis.
- **Pessoa 2008/2013** — *On the relationship between emotion and cognition*. Emotion-cognition integration as fundamental, not modular.
- **Hull 1943** — *Principles of Behavior*. Drive-reduction theory. The foundation behind drive-modulated learning.
- **Anderson & Lebiere 1998** — *The Atomic Components of Thought* (ACT-R). Episodic memory + base-level activation.
- **Camerer, Ho, Chong 2004** — *Cognitive Hierarchy Theory*. K-level reasoning, used in your project's Theory-of-Mind layer.

### Project tie-in
- The cognition pipeline is **the representation layer the Q-learner operates over**. Every change to the pipeline changes what's tile-coded, and therefore what's learned.
- The pipeline is also **dense with hand-engineered priors** — salience tables, drive-amplification coefficients, recipe bonuses. From an RL purist's perspective these are "hardcoded answers." From a behavioral-realism perspective they're "priors that make learning tractable on realistic timescales." This is a real engineering tradeoff.
- The prior-vs-learnable boundary is surveyed in [`docs/designs/cognitive_decision_model.md`](https://github.com/falahat/simulator/blob/main/docs/designs/cognitive_decision_model.md) (the eight pathways each declare what is hand-set vs learned); the hand-set constants themselves live in `sim_config`'s rustdoc.

### Exercises
1. Map every cognition-pipeline component to an RL role: (a) representation layer (perception, memory, emotion state), (b) reward shaping (drives→cost, affect→prime), (c) policy prior (recipe bonuses), (d) exploration noise (intoxication, adrenaline).
2. Read 3 of the Panksepp drive academic papers. Write up the wanting-vs-liking dissociation in 500 words.
3. **Project exercise:** for each of the 14 drives, identify whether it should (a) appear in the observation vector (the RL sees it), (b) appear in the reward function (RL is rewarded for managing it), or (c) modulate score directly (a hand-engineered prior). Some drives appear in all three. Is this redundancy a feature or a bug?

### Time
~15 hr.

---

## Week 32 — Determinism, calibration, and the production-RL discipline

### Objectives
The engineering layer that makes RL reproducible and verifiable.

### Reading from the project corpus
- **Determinism canary code:** [`crates/engine/observability/src/canary.rs`](https://github.com/falahat/simulator/blob/main/crates/engine/observability/src/canary.rs) — byte-identical reproducibility across thread counts.
- *(Per-feature meta-learned step sizes — IDBD, Autostep, TIDBD, RProp, AdaGrad/RMSProp/Adam — were considered for the project's tile-coded learner but not pursued; the academic foundations below remain the study material.)*

### Academic foundations
- **Sutton 1992** — IDBD: *Adapting Bias by Gradient Descent*. The original meta-learned step size paper.
- **Mahmood et al. 2012** — Autostep. Online step-size adaptation that's robust.
- **Kearney et al. 2018** — TIDBD. Trace-aware IDBD.
- **Riedmiller & Braun 1993** — RProp. Resilient backpropagation.
- **Henderson et al. 2018** — *Deep RL that Matters*. [arxiv.org/abs/1709.06560](https://arxiv.org/abs/1709.06560). The reproducibility crisis paper. Seed-dependence, baseline-dependence, environment-dependence. Cited in basically every RL "best practices" discussion since.
- **Engstrom et al. 2020** — *Implementation Matters in Deep RL*. [arxiv.org/abs/2005.12729](https://arxiv.org/abs/2005.12729). Why PPO's win over TRPO is mostly about code-level tricks, not algorithm.

### The math that matters
- **IDBD update:** `β ← β + θ · δ · x_i · h_i`; `α_i ← exp(β_i)`; `h_i ← h_i (1 − α_i x_i²)_+ + α_i x_i δ`. The auxiliary trace `h_i` is what makes step-size adaptation respond to gradient *correlation*, not magnitude.
- **Determinism in stochastic algorithms:** deterministic hash-keyed RNG (your project uses splitmix64). Sorted iteration. Seeded shuffling. No process-level randomness.

### Project tie-in
- Would per-feature meta-learned step sizes apply to your linear tile-coded learner? Yes — IDBD/TIDBD are *exactly* designed for linear FA. (The project chose not to pursue this; it stays a worthwhile study/experiment topic.)
- **Constant α residual band** is what causes some test bands to be hard to set precisely (the TD-error noise floor JSONs you blessed earlier this session are exactly the "what's the residual band of constant-α TD?" measurement).
- The Henderson et al. paper is essential — many of your tests have "fail or pass depends on seed" issues. The paper explains why this is rampant in RL and how to handle it.

### Exercises
1. Read Henderson et al. 2018 carefully. Compute the worst-case "seed dependence" of your `learning_navigation` test (run with 10 different seeds, plot return distribution).
2. Implement IDBD in your learner. Compare to constant `α` on `learning_homeostatic`.
3. **Project exercise:** write a `docs/proposals/rl_engineering_discipline.md` doc that codifies the discipline. Calibration JSON artifacts. Determinism canaries. Seed sweeps. Z-tests for baseline comparisons. Etc.

### Time
~12 hr.

---

# Updates to existing weeks

Add to the original syllabus's modules:

## Week 6 (TD learning) — add multi-step returns
- **Munos et al. 2016** — Retrace(λ). [arxiv.org/abs/1606.02647](https://arxiv.org/abs/1606.02647). The unification of n-step, IS, TD(λ).
- Foundation for understanding why your project's TD(0) cannot bridge long delays.

## Week 7 (Function approximation) — add tile-coding deep references
- **Sutton 1996** — sparse coarse coding paper.
- **Whiteson 2007** — adaptive tile coding.
- **Sherstov & Stone 2005** — parameter automation.

## Week 8 (Eligibility traces) — true online + adaptive step size
- **van Seijen et al. 2016** — *True Online TD(λ)*. Modernizes traces.
- **Sutton 1992** — IDBD. (Adaptive step size, ties to Week 32.)

## Week 12 (Actor-critic) — counterfactual baselines
- **Foerster et al. 2018** — COMA. The counterfactual-baseline trick.

## Week 14 (Continuous control) — parameterized actions
- **P-DQN, MP-DQN, H-PPO** — full treatment is Week 28, but a forward reference here.

## Week 15 (Model-based) — successor features
- **Barreto et al. 2017** — Successor features + GPI for transfer.

## Week 16 (Exploration) — HER as exploration
- **Andrychowicz et al. 2017** — HER. Often classified as credit assignment but functionally an exploration enabler in sparse-reward settings.

## Week 18 (Hierarchical) — option-critic + DIAYN
- **Bacon, Harb, Precup 2017** — Option-Critic.
- **Eysenbach et al. 2018** — DIAYN.
- **Nachum et al. 2018** — HIRO.

---

# Comprehensive paper count after extensions

| Category | Original syllabus | Extensions | Total |
|---|---|---|---|
| TD foundations | 4 | 1 | 5 |
| Deep Q | 7 | 0 | 7 |
| Policy gradient & A-C | 6 | 1 | 7 |
| Advanced PG | 3 | 0 | 3 |
| Continuous control | 4 | 0 | 4 |
| Model-based | 5 | 1 | 6 |
| Exploration | 5 | 1 | 6 |
| Offline / IRL / RLHF | 7 | 0 | 7 |
| Hierarchical / MA | 6 | 3 | 9 |
| Distributional | 3 | 0 | 3 |
| Theory | 3 | 0 | 3 |
| FA pathologies | 3 | 0 | 3 |
| **Feature engineering** | 0 | 12 | **12** |
| **Memory in RL** | 0 | 5 | **5** |
| **State similarity / sparse FA** | 0 | 7 | **7** |
| **Action spaces** | 0 | 10 | **10** |
| **Long-horizon credit (advanced)** | 0 | 10 | **10** |
| **Project-specific foundations** | 0 | 15 | **15** |
| **Total** | **~56** | **~66** | **~122** |

The extension nearly doubles the paper count — but the new papers are
exactly the ones your project's problems require.

---

# Updated mapping: project files → syllabus modules

| File | Topic | Week |
|---|---|---|
| **From original syllabus** | | |
| `crates/engine/q_learning/src/lib.rs` | Q-learning core | 6 |
| `crates/engine/q_learning/src/learning_rate.rs` | TD(0) | 6 |
| `crates/engine/q_learning/src/action_template.rs` | Action space | 3 |
| `crates/engine/q_learning/src/tile_coding/` | Tile coding | 7, **23** |
| `crates/engine/q_learning/src/observation_tile_coder.rs` | Feature extraction | 7, **23-25** |
| `crates/cognition/planner/src/policy.rs` | Argmax + ε-greedy | 5, 6 |
| `crates/cognition/planner/src/perception.rs` | Observation | 3 (POMDP) |
| `crates/sim/sim_config/src/learning.rs` | α, γ, ε | 6, 7 |
| `crates/sim/sim_config/src/reward.rs` | R(s) | 3, **30** |
| **New mappings from extensions** | | |
| `docs/designs/04_cognition.md` §Affect + `docs/RESEARCH.md` §Drive taxonomy | Drive taxonomy, MORL | **30, 31** |
| `docs/proposals/first_person_beliefs.md` | Beliefs as memory views | **31** |
| `crates/sim/sim_config/src/reward.rs` + `docs/CHANGELOG.md` "Forward-search planner removed" | Reward design, MORL | **28, 30** |
| `docs/designs/cognitive_decision_model.md` | Eight cognition pathways | **31** |
| `docs/designs/cognitive_decision_model.md` + `sim_config` rustdoc | Prior-vs-learned tradeoffs | **31** |
| `docs/designs/04_cognition.md` | Pipeline overview | **31** |
| `crates/cognition/attention/src/salience.rs` | Salience pipeline | **31** |
| `crates/engine/observability/src/canary.rs` | Determinism | **32** |

---

# Reading paths by goal

If you want to focus on a specific outcome, here are recommended subsets:

### "I want to fix the Q-bias bootstrap bug as quickly as possible"
Weeks 6 (TD), 7 (linear FA), 11-12 (PG + actor-critic), 20 (pathologies), **30 (project-specific)**. ~6 weeks, ~90 hours.

### "I want to make L1-L5 actually work"
Weeks 6, 16 (exploration), 20, **27** (state similarity), **28** (action spaces), **29** (long-horizon credit), **30**. Pay attention to RUDDER + HER + PBRS as candidate fixes. ~7 weeks, ~110 hours.

### "I want to upgrade the feature representation"
Weeks 7, **23-26**. ~5 weeks, ~80 hours.

### "I want to add proper exploration"
Weeks 5, 16, **28** (action masking is a form of exploration constraint). ~3 weeks, ~50 hours.

### "I want full RL master's-level fluency before touching the sim"
The full 32 weeks. ~500 hours. ~32-50 weeks at 10-15 hr/wk.

---

# How chapters get built from this

Same as the core syllabus: each module becomes a chapter document in
`docs/textbook/`. The new modules above are particularly
amenable to **case-study chapters** because they have direct project
hooks:

- Chapter 43 (Tile coding): use your actual `observation_tile_coder.rs` + the per-block fix as the case study.
- Chapter 45 (Learned reps): walk through the `ObservationEncoder` trait + each candidate implementation.
- Chapter 48 (Action spaces): walk through your 15 primitives, identify parameterized ones, sketch the P-DQN port.
- Chapter 49 (Credit assignment): RUDDER on L1 as worked example.
- Chapter 50 (Q-bias bug): the full debug story, with the fix implemented and validated.

When you're ready to write a chapter, point me at the week + file path
and I'll synthesize from the source papers + textbook chapters + your
actual code, producing 20-50 pages of derivations + intuitions +
project-anchored worked examples.

# Chapter 15 — Model-Based Reinforcement Learning

> **Prerequisites:** Chapter [4](06_dynamic_programming.md) (DP /
> planning with a known model), Chapter [6](08_temporal_difference_learning.md)
> (Q-learning), Chapter [9](11_deep_q_learning.md) (DQN). For MCTS,
> Chapter [12](14_exploration.md) (UCB).

> **Citations:** [Sutton 1990] (Dyna); [Browne et al. 2012] (MCTS
> survey); [Silver et al. 2016] (AlphaGo); [Silver et al. 2017]
> (AlphaZero); [Schrittwieser et al. 2020] (MuZero); [Ha & Schmidhuber
> 2018] (World Models); [Hafner et al. 2020] (Dreamer); [Hafner et al.
> 2023] (DreamerV3); [Janner et al. 2019] (MBPO); [Kaiser et al. 2020]
> (SimPLe).

> **Learning objectives:**
> 1. Articulate the model-free / model-based dichotomy and the
>    sample-efficiency / robustness trade-off.
> 2. Implement Dyna-Q and explain how it amortizes a model over many
>    Q-updates.
> 3. Walk through MCTS with PUCT (the AlphaGo selection rule).
> 4. Explain MuZero's latent-model trick and why it works without
>    reconstructing observations.
> 5. Diagnose the Simulator's deleted forward-search planner: why it
>    failed, what would be required to bring it back.

## Why this chapter exists

Two strategies for "learn to act in an unknown world":

1. **Model-free.** Learn $V$, $Q$, or $\pi$ directly from data. Everything
   we have done since Chapter 7. The model of the world is implicit in
   the value or policy.
2. **Model-based.** Learn a model $\hat P, \hat R$ of the world's
   dynamics. Plan using the model to derive $V$, $Q$, or $\pi$.

The trade-off:

- **Model-based is sample-efficient.** A learned model lets you run
  unlimited "simulated rollouts" without real interaction. Crucial when
  data is expensive (robotics, drug discovery, the Simulator).
- **Model-based is fragile to model error.** A small model error
  compounds over a long rollout. The agent learns to exploit
  inaccuracies in the model that do not exist in the world.
- **Model-free is robust** but **wasteful with data**.

Modern wins are model-based: AlphaGo, MuZero, DreamerV3. The Simulator
once tried model-based (the deleted forward-search planner) and
abandoned it. This chapter is the theory + the project's specific
diagnosis.

## Table of contents

- [13.1 Model-free vs. model-based — the central distinction](#131-model-free-vs-model-based--the-central-distinction)
- [13.2 Dyna and learned-model planning](#132-dyna-and-learned-model-planning)
- [13.3 Monte Carlo Tree Search](#133-monte-carlo-tree-search)
- [13.4 AlphaGo, AlphaZero, MuZero](#134-alphago-alphazero-muzero)
- [13.5 World models and Dreamer](#135-world-models-and-the-dreamer-series)
- [13.6 The Simulator's deleted forward-search planner](#136-the-simulators-deleted-forward-search-planner-why-it-failed)
- [13.7 When model-based wins, when it fails](#137-when-model-based-wins-when-it-fails)
- [13.8 Project tie-in: hybrid paths forward](#138-project-tie-in-hybrid-paths-forward)
- [13.9 Exercises](#139-exercises)
- [13.10 References](#1310-references-cited-in-this-chapter)
- [13.11 Further reading](#1311-further-reading)

---

## 13.1 Model-free vs. model-based — the central distinction

Recall the MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Model-based
methods estimate $P$ and $R$ from data:

$$\hat P(s' \mid s, a) = \frac{N(s, a, s')}{N(s, a)}, \quad \hat R(s, a) = \frac{1}{N(s, a)} \sum_{i \in \text{visits}} r_i.$$

(In high-dim or continuous spaces these are neural-network estimators.)
Then **plan** in $(\hat P, \hat R)$ using DP, MCTS, or learned-policy
rollouts.

Model-free methods skip $\hat P, \hat R$. They estimate $V$, $Q$, or
$\pi$ directly. The model is *implicit* in the learned values.

### The sample-efficiency argument for model-based

Each real interaction $(s, a, r, s')$ contributes to **all** of $\hat P$,
$\hat R$, $\hat V$, $\hat Q$ via the model. Once you have a good model,
you can generate as much simulated experience as you want at near-zero
cost. In the limit, $N$ real samples + an accurate model gives you
$N \cdot k$ effective training samples, where $k$ is your simulation
budget.

Empirically: model-based methods often need 10-100× less real-world data
than model-free counterparts on the same task. For sample-expensive
domains (real robots, the Simulator with its full-physics simulation),
this is decisive.

### The robustness argument against

A learned model is wrong. Plan over a 50-step horizon and the error
compounds. Worse: the planner picks actions whose model-predicted
reward is high, which selects for *model error* — the planner ends up
exploiting the model's miscalibration. The result is "the agent looks
great in simulation, falls over in reality."

The whole engineering of modern model-based RL — uncertainty-aware
rollouts (MBPO), short-horizon planning (DreamerV3), latent-only
modeling (MuZero) — is about taming this failure mode.

<div id="ch13-model-error-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/model_error/widget.js"></script>
Roll a noisy linear model forward k steps. The envelope around the predicted state grows like √k (additive Gaussian noise compounds), and the log-scale divergence curve goes essentially linear in k. The "return under model − return under reality" readout is the gap behind every "great in sim, falls over in deployment" story — slide σ_model up to widen it, slide the planner horizon H to control how far you compound before paying.

---

## 13.2 Dyna and learned-model planning

[Sutton 1990] introduced **Dyna-Q**, the conceptual template for all
model-based RL:

```python
for each step:
    # 1. Real interaction
    a = epsilon_greedy(Q, s)
    s', r = env.step(a)
    # 2. Q-update (model-free part)
    Q[s, a] += alpha * (r + gamma * max(Q[s']) - Q[s, a])
    # 3. Update the model
    Model[s, a] = (s', r)
    # 4. Planning: k simulated updates
    for _ in range(n_plan):
        (s_p, a_p) = random_sample(visited_state_action_pairs)
        s'_p, r_p = Model[s_p, a_p]
        Q[s_p, a_p] += alpha * (r_p + gamma * max(Q[s'_p]) - Q[s_p, a_p])
    s = s'
```

Read it: every real step gets one Q-update; then $n$ extra Q-updates
from the learned model. The model amortizes its cost across many
synthetic updates.

### What Dyna gives you

- **Faster value-function learning per real step.** On classic
  benchmarks, Dyna-Q with $n = 50$ planning steps reaches good policies
  with $\sim 1/50$ the real-environment samples of plain Q-learning.
- **Localized planning.** Each planning step is just one Q-update; no
  long imagined rollouts. This avoids the compounding-error problem
  that bites deeper model-based methods.
- **Tabular intuition** for a much broader family.

### Variants

- **Dyna-Q+**: add an exploration bonus to rarely-tried states in the
  model. Forces the agent to revisit places its model is stale on.
- **Prioritized sweeping**: instead of sampling random model-states for
  planning, plan from states most affected by recent changes (largest
  expected Q-update).
- **Deep Dyna** (MBPO, [Janner et al. 2019]): replace tabular
  $\hat P, \hat R$ with neural nets; use short imagined rollouts to
  augment a replay buffer.

<div id="ch13-dyna-q-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/dyna_q/widget.js"></script>
Step the agent one real action at a time on the 5×5 FrozenLake. The Q-table heatmap updates after each real step *and* after n_plan extra model-based updates. Crank n_plan from 0 to 50 and rerun: the steps-to-goal curve drops sharply — same real-world samples, far fewer episodes to convergence. That sample-efficiency win is the whole reason Dyna exists.

<div id="ch13-dyna-n-efficiency-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/dyna_n_efficiency/widget.js"></script>
Six learning curves, one per n_plan ∈ {0, 1, 5, 20, 50, 200}. Set model_noise = 0 and the highest n_plan dominates — pure sample efficiency. Drag noise to ~0.3 and the high-n curves still win, but only barely. Past ~0.7 they cross *below* n = 0: planning against a corrupted model is actively poisoning the Q-table. The headline lesson: the value of planning is bounded by the quality of the model.

---

## 13.3 Monte Carlo Tree Search

### Why MCTS is the load-bearing model-based algorithm

MCTS is the algorithm that made Go beatable by computers, the
engine inside AlphaGo / AlphaZero / MuZero (§13.4), and the
template for nearly every modern "use the model to plan ahead"
approach. Its single big idea is **using UCB (Chapter 14 §12.3)
inside a tree** — every tree node is a bandit problem over its
children, and the global behaviour emerges from compositing local
exploration decisions.

That's structurally important: §13.4's deep-RL hybrids replace the
random rollouts with neural-net value estimates, but keep MCTS's
UCB-inside-tree skeleton. Understanding the four-phase loop here
is what makes §13.4 readable.

Tree search has a deep history (alpha-beta minimax in chess). MCTS
([Coulom 2006]; [Kocsis & Szepesvári 2006] introduced UCT) was the
breakthrough that made Go-class games tractable before neural nets,
and the engine that AlphaGo wraps neural nets around.

### The four phases

```
selection   →   expansion   →   simulation   →   backpropagation
```

1. **Selection.** From the root, recursively pick the child node with
   the highest UCT score:

   $$\text{UCT}(s, a) = Q(s, a) + c \sqrt{\frac{\log N(s)}{N(s, a)}}.$$

   Same as UCB1 (§12.3), applied at every tree node. Continue until
   reaching an unexpanded leaf.
2. **Expansion.** Add one or more children to the leaf (one per untried
   action).
3. **Simulation** (a.k.a. **rollout**). From the new leaf, play to
   episode end using a *rollout policy* (random, or a hand-coded
   heuristic, or a fast neural net). Get a return $G$.
4. **Backpropagation.** Walk back up the tree, updating each node's $Q$
   and $N$.

After many iterations, the action selected from the root is the
most-visited child (high $N(s, a)$ at the root = the search
confidently prefers that action).

### Why each phase is the way it is

The four phases aren't arbitrary — each one solves a specific
problem MCTS would face without it:

- **Selection** uses UCB so the tree spends compute on *promising
  and under-explored* paths. Without UCB, you'd either greedily
  exploit early estimates (locking onto wrong subtrees) or
  uniformly explore (wasting compute on bad ones). The same
  $O(\log t)$-regret machinery from §12.3 carries over: tree
  search converges to the best move at logarithmic rate.
- **Expansion** grows the tree only when needed — *lazily*. A
  full tree would have $b^d$ nodes (branching factor $b$, depth
  $d$). Expanding one leaf per iteration keeps the tree's size
  proportional to the simulation budget.
- **Simulation** uses a fast rollout policy (random / hand-coded /
  cheap neural net) precisely *because* it's a rollout, not a
  decision. The simulation only needs to be unbiased *enough* —
  variance averages out across many simulations.
- **Backpropagation** updates *every* ancestor of the visited
  leaf. This is what makes UCT propagate information: a single
  good rollout at depth $d$ improves the Q-estimate at the root.

### Why UCT works

UCT is UCB inside a tree. At each node, the algorithm balances
exploring under-tried children with exploiting known-good ones.
[Kocsis & Szepesvári 2006] showed UCT converges to the minimax-optimal
move for game trees of bounded depth, under bandit-style assumptions.

### Variants in practice

- **PUCT** (used by AlphaGo): replace $\sqrt{\log N(s)/N(s, a)}$ with
  $\pi_\text{prior}(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$. A
  learned policy prior $\pi_\text{prior}$ guides search.
- **Progressive bias**: add a hand-coded heuristic bias term that fades
  as $N(s, a)$ grows.
- **Virtual loss**: in parallel MCTS, simultaneously expanding workers
  pessimistically downweight in-flight states.

<div id="ch13-mcts-tree-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/mcts_tree/widget.js"></script>
Scrub through MCTS iterations on tic-tac-toe. Each frame is one cycle: red highlights the UCT selection path down from root, orange marks the leaf being expanded, the annotation under it shows the rollout outcome, and you can watch the visit counts and Q-values propagate back up. The tree's *shape* — deep near promising lines, shallow near losing ones — is the whole point of MCTS, and you can literally watch it form.

<div id="ch13-uct-constant-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/uct_constant/widget.js"></script>
Same MCTS engine, but a slider for c at fixed iteration budget. Small c: a deep narrow spike — search exploits the current best line. Large c: a shallow wide bush — search spreads its budget across siblings. The root-visit entropy and principal-variation depth readouts quantify what your eye is seeing. There is no "right" c — the right c depends on how trustworthy your rollout estimate is.

<div id="ch13-puct-prior-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/puct_prior/widget.js"></script>
Bar chart over 9 actions: the prior, the true Q*, and the visit counts after N iterations of PUCT. With a "peaked-good" prior, visits collapse onto the true best fast — the prior is doing real work. With "peaked-bad" or "adversarial" priors, visits start in the wrong place but search eventually corrects (the regret readout still climbs, but bounded). Drop c to zero and PUCT becomes pure prior-following; raise it and the priors get overridden faster. This is how AlphaGo's network and search cooperate.

---

## 13.4 AlphaGo, AlphaZero, MuZero

The three landmark papers, in order.

### AlphaGo ([Silver et al. 2016])

- **Supervised pretraining**: a policy network $\pi_\text{sup}$ trained
  on human expert moves.
- **RL fine-tuning**: $\pi_\text{rl}$ via REINFORCE-style self-play.
- **Value network**: $V$ trained by regression on self-play returns.
- **MCTS at decision time**: PUCT-guided tree search, with the policy
  and value networks providing priors and leaf evaluations.

This combination beat Lee Sedol (2016) and Ke Jie (2017). The key
insight: **deep networks + MCTS** outperforms either alone. The networks
guide search; search corrects network errors.

### AlphaZero ([Silver et al. 2017])

Same architecture, but **no human data**. Train policy and value
networks entirely from self-play with MCTS as the policy improvement
operator. The MCTS tree's visit distribution becomes the policy target;
the actual game outcome becomes the value target.

Mastered Go, chess, and shogi with the same algorithm and no
game-specific code (other than the rules). Generalizable, end-to-end RL.

### MuZero ([Schrittwieser et al. 2020])

The breakthrough: **learn the model.** AlphaZero used the known game
rules as its dynamics model. MuZero learns a latent-space dynamics model
$h_\theta: (s, a) \to s'_\text{latent}$ that does **not** reconstruct
observations — it only predicts reward, value, and policy. The model
trains via supervision against real trajectories.

At decision time: encode the current state into latent; run PUCT in
latent space with the learned dynamics; pick an action.

Two big wins:

- **Atari**: SOTA results from pixels using planning, not just
  model-free.
- **Generalization**: same architecture for Atari, Go, chess, shogi.

The lesson: **the model does not need to reconstruct observations** —
it only needs to support value-and-policy-relevant predictions. This
sidesteps the "compound observation error" problem because we never
decode latents.

---

## 13.5 World models and the Dreamer series

A parallel thread, more focused on continuous control and visual
domains.

### Original World Models ([Ha & Schmidhuber 2018])

Train three pieces:

1. **VAE encoder/decoder** for observations → latent $z$.
2. **MDN-RNN** for dynamics: $z_{t+1} = f(z_t, a_t, h_t)$.
3. **Policy** trained inside the "imagined" trajectories.

Beautiful proof of concept. Slow to train (sequential pipeline).

### Dreamer series ([Hafner et al. 2020]; [2023])

End-to-end joint training of encoder, dynamics, reward predictor, value,
and policy. Plan via short imagined rollouts in latent space.

**DreamerV3** ([2023]) is the current standout: **single hyperparameter
setting solves Atari, DMLab, Minecraft, and more**. Achieves SOTA on
**Minecraft diamond** (collecting a diamond in Minecraft from scratch,
without expert demonstrations) — a multi-hour multi-stage task
previously thought to require curriculum or imitation.

The recipe: latent-space rollouts ($\sim 15$ steps), entropy-regularized
policy + value learning in imagination, replay-buffered real
transitions for grounding the dynamics, careful symlog-style reward
scaling for stability.

DreamerV3 demonstrates that **model-based deep RL at scale is mature
enough to plug-and-play across diverse domains** — a milestone the
field has been chasing for years.

---

## 13.6 The Simulator's deleted forward-search planner — why it failed

Before commit `0a41cef` (May 2026), the Simulator's planner was
**depth-3 BFS forward search**:

1. Enumerate candidate actions from the current state.
2. For each, predict the (state, reward) after applying it via a
   hand-written forward model (`predict_plan_outcome` in
   `planner/src/predict.rs`).
3. Recurse: enumerate from the predicted state, depth ≤ 3.
4. Score the leaf states via successor-feature bootstraps (`q_learning/src/successor.rs`).
5. Pick the action whose root child has highest aggregate score.

Architecturally: model-based planning with a hand-designed model. Cost:
~1500 lines of code. Replaced by 543 lines of flat $\epsilon$-greedy
policy.

### Why it failed

Two pathologies, in priority order:

#### Failure 1: drives did not propagate in the rollout

The forward model treated drives as static during rollout. A hungry
agent considering "Step North, then Step North, then Consume food (50
ticks)" got back: same drives at every leaf, same predicted reward.

The agent could not represent "after 50 ticks of Step actions I will be
hungrier and therefore Consume is more valuable." The model's drives
were frozen; the relevant signal — *change in drive over the planning
horizon* — was zero.

#### Failure 2: $w_\text{alive} = 1.0$ accumulated per simulated step

The leaf score added $w_\text{alive}$ per simulated tick. A 50-tick
"fast no-op chain" (Wait, Wait, Wait, ...) got 50 ticks × 1.0 = 50
units of aliveness reward. A 50-tick "approach + Consume" chain got the
same 50 units of aliveness reward plus a Consume bonus at the leaf —
but the Consume bonus was small relative to 50× the alive baseline.

So the planner picked the fast-no-op chain. **A hungry agent on top of
food did not eat.** Test runs reproduced this consistently.

This is structurally the **same pathology** as the Q-bias bootstrap
(Chapter 17): $w_\text{alive}$ saturates whatever value is being
evaluated. The forward-search version was more visible because the
horizon was longer (50 ticks vs. effective $1/(1-\gamma) = 10$).

### Why it got deleted, not fixed

The argument from commit `0a41cef`:

- The `Learner` (tile-coded Q) **already** values actions over a long
  horizon via TD. The rollout duplicated this work.
- The rollout's pathologies (drive non-propagation, alive-baseline
  saturation) would require substantial engineering to fix — making
  drives evolve in the model, adding cost or advantage subtraction.
- Removing the planner reduced ~1500 lines to ~500 with no measurable
  regression on the validation suite at the time.

Trade-off: the Simulator lost explicit planning, gaining only what the
TD learner can express in the effective horizon (10 cognition steps).
Chapter 16 (hierarchy) and Chapter 19 (long-horizon credit) discuss the
implications.

### What would be required to bring it back, modern-style

Following MuZero's lesson:

1. **Learn the dynamics model.** Stop hand-writing it. A small
   tile-coded or MLP transition predictor over the drive vector +
   relevant world features would suffice. Train on real transitions
   from the simulation.
2. **Latent-space rollouts.** Operate on a compressed representation
   (the 14 drive + 8 body + 5 world coordinates), not the full
   251-dim observation. Avoids set-as-vector noise (§8.10).
3. **Short horizons.** $\leq 5$ rollout steps. DreamerV3 uses 15;
   the Simulator's cognition cadence (10 ticks per step) suggests
   $\leq 3$ rollout steps cover effective horizon $30$ ticks.
4. **Advantage, not Q.** Subtract $V(s)$ at the leaf (per Chapter 13)
   so $w_\text{alive}$ cancels.
5. **Hybrid: real model + learned model.** The Bevy world *is*
   actually a model (deterministic given a seed). A few-step real-world
   rollout is sometimes cheaper than calling a learned model — use the
   real one when available.

None of this is on the project roadmap, but the design space is well-
understood and Chapter-17 motivated.

---

## 13.7 When model-based wins, when it fails

### Wins

- **Sample-expensive environments.** Real robots, biology/chemistry,
  the Simulator. Each interaction costs real wall-clock or money.
- **Planning-required tasks.** Multi-step puzzles (Sokoban,
  Minecraft diamond), board games, dexterous manipulation. Pure
  reactive policies fail.
- **Transfer.** A learned model can be reused across reward functions
  ("learn the world's physics, then ask different questions"). Pure
  model-free agents must re-train from scratch.

### Fails

- **Compound model error.** Long horizons amplify small model
  inaccuracies. MuZero/Dreamer fix this with short rollouts and latent
  models.
- **Off-distribution states.** The model trained on data the agent
  visited may have no information about novel states the planner
  proposes. Uncertainty-aware planning (MBPO) helps.
- **High-dim, partial-observability.** The model needs to encode all
  task-relevant state. With POMDP-like obs, the model must learn the
  full belief — hard.
- **Exploration coupling.** The agent only sees data its policy
  generates. A poor early policy means the model never learns about
  productive parts of state space. Curiosity helps (Chapter 14).

---

## 13.8 Project tie-in: hybrid paths forward

If the Simulator pursued model-based RL again:

| Approach | Effort | Risk | Upside |
|---|---|---|---|
| **Dyna over real world** | Low | Low | $10×$ Q-update efficiency; uses Bevy world itself as the model |
| **Latent-space MuZero-style** | High | Medium | Long-horizon planning; potential L-suite breakthrough |
| **DreamerV3-style world model** | High | High | SOTA on similar tasks; full pivot from current architecture |
| **Forward-search v2 with advantage-corrected leaves** | Medium | Low | Resurrects the deleted planner without the alive-baseline failure |
| **No model-based** | Zero | None | Status quo; long horizons require Chapter-17 fixes |

### Dyna over the real world — the cheapest path

Notable property: the Bevy world is **deterministic given a seed**.
So you can run *the actual simulation* as your "model": fork the world,
play forward $k$ ticks, get the exact future. No model-error problem
because the model is the real world.

Implementation sketch:
1. Snapshot the world state at the agent's decision time.
2. For each candidate action, fork the world, apply the action,
   advance $k$ ticks, observe the future.
3. Use the future observation/reward as the "model-predicted"
   transition. Update Q on the snapshot transition.

Cost: `k × candidate-count × cognition-step world-ticks`
extra simulation per cognition step. Limited by:

- Simulation cost is dominated by physics + world systems. Forking is
  cheap; advancing is not.
- For $k = 5$ ticks, 10 candidates, 1 cognition step per agent per
  10 ticks: that's $5 \times 10 / 10 = 5$ extra world-ticks per
  agent-tick on average. Probably feasible for low-agent-count tests.

This is essentially what an MCTS-on-the-real-world would do, but without
the tree-search overhead — just 1-step rollouts per candidate.

<div id="ch13-dyna-budget-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/dyna_budget/widget.js"></script>
Plug in the Simulator's actual knobs — rollout depth k, candidate count, cognition cadence, agent count — and read the per-frame cost in milliseconds against your frame budget. The heatmap to the right shows the feasibility region over (k, candidates): green cells fit in the budget, red cells don't. This is §13.8's back-of-envelope "5 extra world-ticks per agent-tick" turned into an actual sizing tool for the project.

### What test would benefit most

[`long_horizon_harvest.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/curricula/long_horizon_harvest.rs)
— the L-suite's $\gamma^{500} \approx 0$ problem is fundamentally
about long-horizon credit assignment. A $k = 5$ Dyna rollout would
not close that gap; MuZero/Dreamer-style $\sim 20$-step latent
planning might. The tradeoff: latent models need substantial training
data before they're useful.

---

## 13.9 Exercises

1. **(Dyna-Q on FrozenLake.)** Implement Dyna-Q with
   $n \in \{0, 5, 50\}$ planning steps. Plot episodes-to-converge for
   each. Confirm the speedup.

2. **(Prioritized sweeping.)** Modify Dyna-Q so planning prioritizes
   states with large expected Bellman backups. Compare to uniform-
   sample Dyna.

3. **(MCTS on tic-tac-toe.)** Implement MCTS with UCT. Confirm it
   plays optimally after enough iterations.

4. **(PUCT with a hand-coded policy prior.)** On the same tic-tac-toe,
   add a hand-coded center-prefer prior. How does the visit
   distribution shift?

5. **(MuZero paper trace.)** Read [Schrittwieser et al. 2020]
   carefully. Identify which losses are computed at which timesteps;
   sketch the gradient flow.

6. **(Compounding error.)** Train a 1-step dynamics model on a noisy
   pendulum. Roll forward $k$ steps; measure prediction error vs. $k$.
   Plot.

7. **(Dyna over the Simulator.)** Pseudo-code the Dyna-over-real-world
   approach for the Simulator: where the snapshot is taken, how the
   fork is performed, what the planning Q-update looks like, what
   the wall-clock budget is.

8. **(Forward-search v2 plan.)** Pseudo-code a resurrected forward-
   search planner that (a) advances drives via a learned drive-
   dynamics model, (b) scores leaves by $A(s, a) = Q(s, a) - V(s)$
   instead of $Q$. Identify the changes vs. the deleted planner.

---

## 13.10 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Browne et al. 2012] — MCTS survey (§13.3).
- [Coulom 2006] — original MCTS for Go (§13.3).
- [Ha & Schmidhuber 2018] — World Models (§13.5).
- [Hafner et al. 2020] — Dreamer (§13.5).
- [Hafner et al. 2023] — DreamerV3 (§13.5).
- [Janner et al. 2019] — MBPO (§13.2).
- [Kaiser et al. 2020] — SimPLe (§13.5).
- [Kocsis & Szepesvári 2006] — UCT (§13.3).
- [Schrittwieser et al. 2020] — MuZero (§13.4).
- [Silver et al. 2016] — AlphaGo (§13.4).
- [Silver et al. 2017] — AlphaZero (§13.4).
- [Sutton 1990] — Dyna (§13.2).
- [Sutton & Barto 2018, Ch. 8] — planning and learning.

## 13.11 Further reading

| Source | What to read | Why |
|---|---|---|
| [Sutton & Barto 2018] | Ch. 8 | Textbook treatment of Dyna and planning |
| MuZero blog [DeepMind] | All | Accessible exposition of the latent-model trick |
| DreamerV3 paper | All | Current SOTA model-based agent |
| [Janner et al. 2019] | MBPO paper | Deep model-based + replay |

---

**Next:** [Chapter 16 — Hierarchical RL](16_hierarchical_rl.md) — when
temporal abstraction unlocks long-horizon planning.

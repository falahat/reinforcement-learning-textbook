# Glossary

A central glossary of terms, symbols, and acronyms used throughout the
textbook. Edit this file to add or update definitions — the
`mdbook-termlink` preprocessor scans chapter prose for matches and
auto-wraps the first occurrence per chapter with a hover tooltip.
Source markdown is never touched; the wrapping happens at build time.

Format: a [Markdown definition list](https://www.markdownguide.org/extended-syntax/#definition-lists)
— term on one line, `:` plus the definition on the next.

α (alpha)
: The TD learning rate. How much each update moves Q toward its target.
  Higher α = faster learning but noisier. In `LearningConfig.alpha`.

γ (gamma)
: The discount factor, γ ∈ [0, 1). Future rewards weighted by γ^k.
  Effective horizon = 1 / (1 - γ). Higher γ values future rewards more.
  In `LearningConfig.gamma`.

ε (epsilon)
: Exploration probability in ε-greedy action selection. With probability
  ε, pick a uniformly random action; otherwise the argmax over Q.

λ (lambda)
: Eligibility-trace decay rate, λ ∈ [0, 1]. Interpolates between TD(0)
  (λ=0) and Monte Carlo (λ=1).

δ (delta)
: The TD error: δ = reward + γ · max_a Q(s', a) - Q(s, a). Drives the
  update step. |δ| → 0 as Q converges.

θ (theta)
: A vector of learnable parameters (weights). The thing gradient descent
  is descending in.

π (pi)
: A policy — a map (or distribution) from states to actions.

V^π
: State-value function under policy π. Expected return starting from a
  given state and following π thereafter.

Q^π
: Action-value function under policy π. Expected return starting from
  (s, a) and following π thereafter.

MDP
: Markov Decision Process. The formal model of an RL problem: states,
  actions, transition probabilities, rewards, discount factor.

POMDP
: Partially Observable MDP. The agent sees observations, not the full
  state. The Simulator is technically a POMDP.

TD
: Temporal-Difference learning. Updates Q toward a target that mixes
  the immediate reward and a bootstrap from the current Q estimate.

SARSA
: TD control where the bootstrap uses the actually-taken next action:
  `Q ← Q + α(r + γ Q(s', a') - Q)`.

Q-learning
: TD control where the bootstrap uses the greedy next action:
  `Q ← Q + α(r + γ max_a Q(s', a) - Q)`.

GPI
: Generalised Policy Iteration. The general "evaluate then improve"
  scheme; most RL algorithms are GPI variants.

DP
: Dynamic Programming. Solving an MDP exactly given full knowledge of
  P and R. The textbook's Chapter 4.

MC
: Monte Carlo. Estimating value by sample episode returns. Chapter 5.

FA
: Function Approximation. Representing Q or V as a parametric function
  rather than a table. Chapters 8–11.

DQN
: Deep Q-Network. Q-learning with a neural-network value approximator.
  Chapter 9.

PPO
: Proximal Policy Optimization. Policy-gradient with a clipped
  importance ratio. Chapter 11.

MCTS
: Monte Carlo Tree Search. Plan by selectively expanding a search tree.
  Chapter 13.

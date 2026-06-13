# Interactive textbook — design record

**Status:** ✅ Shipped (moved from `proposals/` to `designs/` 2026-05-21).
The mdBook site plus its interactive widgets (138 widgets live
across chapters 1–17) are in production. This doc is kept as the
canonical record of the hosting / language / build-pipeline decisions
and per-chapter widget rationale that the textbook code points back to.
The backlog of additional widget candidates surfaced by the chapter
audits has been merged directly into the per-chapter inventory below
(rows beyond the original 1.A–18.A list).

**Cognition-side groundwork:** done (shipped 2026-05-20). The
six refactors that gave us pure-Rust cores and feature-gated Bevy
unblocked the WASM path:

- A pure-Rust [`playground` crate](../../crates/engine/playground/)
  with `run_episode` over any [`Environment`](../../crates/engine/rl_core/src/environment.rs).
- Reference toy envs ([`MultiArmedBandit`, `GridWorld`](../../crates/engine/playground/src/envs.rs))
  that learn correctly (93→186 best-arm/200; 130→8 steps-to-goal).
- A feature-gated `rl_core` that builds without `bevy_ecs` —
  `cargo check -p rl_core --no-default-features` is green. **The WASM
  path is unblocked.**

The bandit / gridworld widgets compile *this*
code directly to WASM. No rewrite, no re-translation.

## The idea

The textbook today is excellent at *explaining* RL concepts but every
intuition arrives second-hand: the reader reads a formula, looks at a
static diagram, and is asked to trust the description. An interactive
textbook would let the reader **move the parameter and see the
consequence** — the fastest known path to durable understanding of
mathematical objects.

Two motivating examples from the user:

1. **Norm visualizer.** Next to §1.2's norms section, a widget where
   the student picks a norm (`ℓ₁`, `ℓ₂`, `ℓ∞`, Mahalanobis, Frobenius,
   …), drags a vector around a 2D plane, and watches the unit ball
   redraw + the vector's length update live.

2. **Pac-Man trainer.** A live RL environment running alongside the
   text, with sliders for observation radius, algorithm choice
   (Q-learning vs SARSA vs A2C), and key hyperparameters (`α`, `γ`,
   `ε`). The student watches episodes play out and compares learning
   curves across configurations.

The bet: a learner who **builds intuition by manipulation** retains
more, faster, than one who reads + grinds exercises. Interactive RL
demos are how OpenAI Spinning Up, DeepMind's blog posts, and
Distill.pub teach.

---

## What we'd build (chapter inventory)

A walk through the existing 18 chapters, listing the highest-leverage
interactive pieces. Roughly ordered by ratio of pedagogical-value to
implementation-cost.

### Chapter 1 — Math foundations

| # | Widget | What the student does | Effort |
|---|---|---|---|
| 1.A | **Norm visualizer** | Pick `p` ∈ {1, 1.5, 2, 3, ∞}; drag a 2D vector; see unit ball, vector arrow, length. Toggle Mahalanobis with an editable `S` matrix. | S |
| 1.B | **Eigenvector finder** | Edit a 2×2 matrix; see vectors color-coded by whether they stay axis-aligned under `A`. Eigenvalues displayed; eigenvectors highlighted. | S |
| 1.C | **Gradient descent tracer** | Pick a loss surface (convex bowl, banana, saddle, multi-modal); set `η` and `x₀`; watch GD trajectory unfold. Add momentum / Adam toggles. | M |
| 1.D | **Hessian curvature painter** | At any point on a 2D loss, draw the Hessian's eigenvectors + eigenvalues as a quadratic-form ellipse. Compare GD direction vs. Newton direction. | M |
| 1.E | **Contraction mapping animator** | Pick a contraction `T(x) = γx + b` on `ℝ`. Drag `x₀`. Watch iterates converge geometrically. Plot `\|x_k - x*\|` log-scale. | S |
| 1.F | **Hoeffding sample-size calculator** | Sliders for `ε`, `δ`, `b-a`. Live readout of required `n` + a plot of the bound vs `n`. | S |
| 1.G | **Tower-property bouncer** | Drag rewards/probabilities on a tiny tree; watch `E[X]` computed two ways (flat over leaves vs iterated `Σπ Σ P[…]`) always agree. | S |
| 1.H | **Markov-chain mixing & second-eigenvalue race** | Edit a row-stochastic `P`; watch an initial distribution converge to `μ*` and see the TV-distance decay track `\|λ₂\|^k`. | S-M |
| 1.I | **Convexity / Jensen explorer** | Pick a function preset; drag two points and `λ`; see the chord vs the curve and the Jensen gap numerically. | S |
| 1.J | **Robbins-Monro step-size race** | Three parallel SGD runs (`α=c/k`, `c/√k`, constant); watch the RM-compliant iterate land on truth while constant-α bounces in an `O(√α)` band. | S |
| 1.K | **Operator vs Frobenius norm — spectral stretch** | Edit a 2×2 `A`; see the unit circle map to an ellipse with singular values as semi-axes and live operator/Frobenius/nuclear-norm readouts. | S-M |
| 1.L | **Mahalanobis "weird ruler"** | Edit a PSD `S`; draw isodistance ellipses and contrast Mahalanobis vs Euclidean distance between point clouds. | M |
| 1.M | **Greedy-policy-error amplifier** | Sliders for `ε` and `γ`; live readout of `2γε/(1-γ)` showing how tiny value error explodes into policy loss at high `γ`. | S |

### Chapter 2 — RL problem

| # | Widget | | Effort |
|---|---|---|---|
| 2.A | **Agent-environment loop animator** | Step through `(s_t, a_t, r_{t+1}, s_{t+1})`. Toggle Markov vs. non-Markov state. | S |
| 2.B | **Discount factor explorer** | Adjust `γ`; visualize the geometric series; show effective horizon. | S |
| 2.C | **Multi-armed bandit playground** | 10-arm bandit; choose ε-greedy / UCB / Thompson; watch regret curves accumulate live. | M |
| 2.D | **Episode vs continuing return calculator** | Edit a reward profile and `γ`; compare undiscounted, discounted, and averaged `G₀`, with the `R_max/(1-γ)` bound shown. | S-M |
| 2.E | **Episode vs continuing dichotomy** | Two side-by-side tickers — an episodic agent that resets vs a continuing one that carries discounted credit forever. | S |
| 2.F | **UCB confidence-interval visualiser** | 5-arm bandit; each arm a thermometer of `μ̂` plus its confidence interval; step and watch UCB1 argmax the highest upper edge. | M |
| 2.G | **Markov-vs-history regression** | A momentum gridworld where a Markov-naive agent and a 2-frame-stacking agent both predict values; measure the value-RMSE gap as history matters more. | M |

### Chapter 3 — MDPs

| # | Widget | | Effort |
|---|---|---|---|
| 3.A | **Gridworld MDP editor** | Paint walls + rewards on a 5×5 grid. Edit `γ`. Live-compute `V^π` for a chosen policy. | M |
| 3.B | **Bellman expectation propagator** | One-step backup animation: pick a state, watch `V^π(s)` = `Σ π(a|s) Σ P(s'|s,a) [r + γ V(s')]` accumulate. | S |
| 3.C | **Stochastic-vs-deterministic-policy explorer** | Edit a stochastic policy on a gridworld; watch the deterministic argmax-`Q^π` policy dominate it pointwise — you can't beat it by hand. | M |
| 3.D | **POMDP belief-state demo** | The Tiger problem: step through listen/open actions and watch the belief over which door hides the tiger update via Bayes. | M |
| 3.E | **Bellman as a linear system** | Edit `P^π` and `r^π`; see `V^π = (I − γP^π)⁻¹ r^π` recompute, compared against the iterative `T^π V_k` sequence converging. | M |
| 3.F | **Reward shape — R(s) vs R(s,a) vs R(s,a,s')** | Toggle the reward formulation on a small gridworld; solve each MDP and compare how the optimal policy and `V*` change. | M |

### Chapter 4 — Dynamic programming

| # | Widget | | Effort |
|---|---|---|---|
| 4.A | **Value iteration on a gridworld** | Watch the value function fill in tile-by-tile across iterations. Sweep `γ` and see convergence rate change. | M |
| 4.B | **Policy iteration vs value iteration** | Same gridworld, side-by-side, count iterations to convergence. | M |
| 4.C | **Modified PI knob (k evaluation steps)** | Slide `k` evaluation steps per improvement; watch total Bellman backups trace the PI–VI continuum (GPI). | M |
| 4.D | **Prioritized sweeping animator** | Sparse-reward gridworld; vanilla VI fills uniformly while prioritized sweeping pops a max-residual heap one state at a time — far fewer updates. | M-L |
| 4.E | **Policy-iteration count heatmap** | Run PI across many random MDPs; watch a `(\|S\|, γ)` heatmap of iterations-to-convergence form, sitting far below the worst-case bound. | M |
| 4.F | **LP-formulation duality demo** | A 3-state MDP showing VI converging beside the LP feasible polytope, with simplex vertex hops to the optimum (optional advanced). | L |

### Chapters 5–6 — MC and TD

| # | Widget | | Effort |
|---|---|---|---|
| 5.A | **Monte Carlo vs TD trajectory** | Cliff Walking. Watch MC's high-variance episode-end updates vs TD's per-step updates. Two learning curves overlaid. | M |
| 5.B | **First-visit vs every-visit MC race** | Run both MC variants on a random walk; overlay RMS-error curves and per-state estimates (first-visit unbiased, every-visit slightly faster but biased). | S |
| 5.C | **Importance-sampling variance blow-up** | Draw trajectories under a behaviour policy; watch the IS-ratio histogram grow heavy-tailed as the target sharpens, with ordinary vs weighted IS estimates. | M |
| 5.D | **Exploring-starts converger** | Two MC-control runs on a gridworld — exploring starts vs ε-greedy — compared by `(s,a)` coverage heatmaps after many episodes. | M |
| 5.E | **Episodic-cutoff sensitivity** | A continuing task where MC needs a cutoff `T`; slide `T` and watch the estimate approach truth as variance explodes, with TD(0) converging faster. | S-M |
| 6.A | **Q-learning on the cliff** | The textbook scenario. Watch Q-table fill in; render the greedy policy as arrows. Toggle ε-greedy / softmax. | M |
| 6.B | **SARSA vs Q-learning** | Side-by-side on the cliff. SARSA learns the safe path, Q-learning the risky-optimal. The classic. | M |
| 6.C | **Maximization-bias visualizer** | Roulette MDP from S&B. Show how `max` over noisy Q overestimates; toggle Double Q. | M |
| 6.D | **Expected SARSA explorer** | Three-way cliff race (SARSA, Q-learning, Expected SARSA); slide `ε` and watch Expected SARSA interpolate between the other two with tighter variance. | M |
| 6.E | **n-step TD spectrum** | Slide `n` on a random walk; overlay RMS-error curves until the bias-variance U-shape emerges, previewing TD(λ). | M |
| 6.F | **TD-error timeseries microscope** | Run Q-learning on a gridworld; plot `δ_t` over time decaying to an `O(√α)` noise floor that widens as `α` grows. | S-M |
| 6.G | **Watkins' coverage counter-example** | A gridworld with two goals where a low-`ε` agent never reaches the better one — dial `ε` and watch the convergence hypothesis matter. | M |
| 6.H | **TD-target dissection** | Pick a TD-target formula from a dropdown; see the target computed live and which `Q(s',·)` entries each variant reads. | S |

### Chapter 7 — Eligibility traces

| # | Widget | | Effort |
|---|---|---|---|
| 7.A | **TD(λ) trace painter** | Random walk environment. Show eligibility traces decaying as colors on a chain of states. Slider for `λ`. | M |
| 7.B | **λ-return horizon mixer** | Slide `λ`; watch the `(1-λ)λ^{n-1}` weights on each n-step return redraw, with cumulative mass markers at 90% and 99%. | S |
| 7.C | **Effective-horizon calculator** | Two sliders (`γ`, `λ`); live readouts of `γλ`, trace half-life, 1% life, and effective horizon `1/(1-γλ)`. | S |
| 7.D | **Accumulating vs replacing vs Dutch traces** | Pick a state-revisit pattern; overlay the three trace formulas and slide `α` to watch them diverge. | M |
| 7.E | **Forward-backward equivalence stepper** | One random-walk episode shown both ways — forward `G_t^λ` updates vs tick-by-tick backward trace updates — proven equal at the end. | M |
| 7.F | **λ-sweep on the random walk** | Live U-curve of RMS error vs `λ`; scrub the episode count and watch the optimum migrate from low-λ toward `λ≈0.9`. | M |

### Chapter 8 — Function approximation

| # | Widget | | Effort |
|---|---|---|---|
| 8.A | **Tile coding visualizer** | 2D continuous state. Drag a point; see which tiles light up across `T` overlapping tilings. Adjust tile width, `T`, IHT size. | S-M |
| 8.B | **Generalization demo** | Train on one state, watch the value function generalize to nearby states. Side-by-side tile coding vs tabular. | M |
| 8.C | **Per-block tiling demo** | The project's specific design. Show two observations agreeing on drives but differing on perception; see drive tiles shared, perception tiles distinct. | M |
| 8.D | **Tabular blow-up calculator** | Sliders for state dim and bins; watch `b^n` cross `10^80` (atoms in the universe) against RAM-storable reference lines. | S |
| 8.E | **Semi-gradient vs full-gradient TD** | A linear-FA toy where the `θ` trajectory shows semi-gradient converging fast to the projected (wrong) fixed point vs residual-gradient slower to the true one. | M |
| 8.F | **Hash collision pressure visualizer** | Sliders for IHT size, tilings, and states visited; live expected collision fraction and per-slot occupancy histogram against the Simulator's defaults. | S |
| 8.G | **Set-as-vector salience-swap demo** | Drag slot saliences in a 2-slot memory mock; at the swap point tile activations jump discontinuously, spiking the TD error (permutation non-invariance). | M |
| 8.H | **Sutton offset asymmetry visualizer** | Toggle aligned vs Sutton offsets across `T` tilings; see the partition refine to effective resolution `w/T` with a unique-cell count. | S-M |
| 8.I | **LSTD vs SGD-TD comparison** | Race online semi-gradient TD against incremental LSTD on a linear-FA problem, trading sample efficiency against `O(d²)` per-update cost. | M |

### Chapter 9 — Deep Q-learning

| # | Widget | | Effort |
|---|---|---|---|
| 9.A | **DQN CartPole trainer** | Live DQN. Toggle target network on/off, replay buffer size, ε-decay. Watch learning curve and Q-value heatmap. | L |
| 9.B | **Dueling decomposition** | Side-by-side `Q(s, a)` vs `V(s) + A(s, a) - mean(A)`. Show how `V` absorbs the alive baseline. | M |
| 9.C | **Maximization-bias drift visualizer** | A K-arm bandit with noisy Q estimates; watch `max_a Q̂` bias grow as `√(2 log K)`, then the Double-DQN estimator drive it near zero. | S |
| 9.D | **Target network freezing demo** | DQN-style updates on a 2-state MDP with no target net (diverges), hard target (stair-steps), or soft Polyak target (smooths). | M |
| 9.E | **Prioritized Experience Replay distribution** | Sliders for priority exponent `ρ` and IS-correction `β`; see priority, IS weight, and effective sample contribution over a replay buffer. | M |
| 9.F | **C51 / QR-DQN distributional return** | Overlay C51's atom approximation and QR-DQN's quantile approximation on a known return distribution; mean is right but shape needs enough atoms. | M |
| 9.G | **NoisyNet parameter-noise demo** | Scatter sampled weights `μ + σ⊙ξ`; slide `σ` and watch the cloud expand from greedy to exploratory, with the induced action distribution. | S |
| 9.H | **Deadly triad live demo** | Three switches (FA, bootstrapping, off-policy) over Baird's example; `\|θ\|` diverges only with all three on — a `2³` truth table. | M |

### Chapter 10–11 — Policy gradient + Actor-Critic

| # | Widget | | Effort |
|---|---|---|---|
| 10.A | **REINFORCE on CartPole** | Live training; show high variance with naked policy gradient vs. with baseline. Variance plot. | L |
| 10.B | **REINFORCE variance histogram** | Run REINFORCE many times in parallel; show the gradient-magnitude histogram tighten as you add a mean baseline, then a learned critic baseline. | M |
| 10.C | **Score-function vs reparameterization gradient** | Two clouds of gradient samples for a Gaussian policy; scrub `σ` and watch score-function variance blow up while reparameterized stays bounded. | M |
| 10.D | **Softmax policy on a 2-arm linear bandit** | Step through policy-gradient updates one click at a time; see the chosen action, reward, per-logit gradient, and policy animate toward greedy. | M |
| 10.E | **GAE(λ) bias-variance scatter** | For each `λ`, plot GAE samples on a bias-variance scatter; trace the Pareto curve from pure-TD to pure-MC with `λ=0.95` at the elbow. | M |
| 10.F | **Causality-trick visualization** | A time×reward grid where cells light by `E[∇log π · r]`; the zero lower triangle shows why `G_t` replaces `G₀`. | S |
| 11.A | **PPO clipping visualizer** | Plot the surrogate objective `L^CLIP(r)` as a function of policy ratio. Show clip region. | S |
| 11.B | **Actor + critic curves** | A2C on CartPole. Two curves: actor return, critic TD error. Toggle GAE λ. | L |
| 11.C | **TRPO KL-trust-region visualization** | A KL heatmap over a 2D policy `θ`-plane with the `δ` trust-region contour and the unconstrained vanilla-PG step arrow. | M |
| 11.D | **SAC entropy-temperature schedule** | Pick a target entropy; watch the learned `α` Lagrange-multiplier its way to satisfying the constraint over training. | M |
| 11.E | **Clipped double-Q overestimation fix (TD3)** | Compare single Q (biased), Double Q (unbiased), and clipped-min Double Q (deliberately under-biased) over samples per arm. | S |
| 11.F | **Actor-critic feedback loop animator** | Animated §11.1 diagram: arrows light up per step as the TD error flows to both actor and critic, with numerical values. | M |
| 11.G | **Advantage-learning on the Simulator's sated arm** | A toy sated-arm env where plain Q inflates toward `w_alive/(1-γ)` but `Q−V` advantage keeps `A` zero-mean — toggle `w_alive` live. | M |

### Chapter 12 — Exploration

| # | Widget | | Effort |
|---|---|---|---|
| 12.A | **Bandit algorithm race** | 10-arm bandit. Race ε-greedy / UCB1 / Thompson / decaying-ε. Cumulative regret curves overlaid. The textbook §12.3-12.4 in one widget. | M |
| 12.B | **RND/ICM curiosity demo** | Sparse-reward gridworld with a "noisy TV" cell. Show curiosity going to TV with bare ICM forward model vs. inverse-model-regularized. | L |
| 12.C | **UCB1 confidence-bound visualizer** | Per-arm bars with `±√(2 log t / N_k)` whiskers; step through rounds and watch UCB pick the highest upper whisker as whiskers shrink. | S |
| 12.D | **Thompson sampling posterior animator** | Five Beta posteriors over arms; each step samples once per arm, pulls the highest, and re-shapes that arm's posterior as it sharpens around true means. | M |
| 12.E | **Bernoulli vs Gaussian vs adversarial bandit** | Switch the bandit type under the UCB1/ε-greedy/Thompson trio; watch UCB1's regret bound break under non-stationarity while Thompson stays resilient. | M |
| 12.F | **Sparse-reward Montezuma-style gridworld** | Race ε-greedy, pseudo-count-bonus, and RND-style agents to a single reward cell; compare steps-to-first-reward and visit-distribution heatmaps. | L |
| 12.G | **Bootstrapped-DQN ensemble** | `K` bootstrap-resampled Q-learners; sample one head per episode and watch the ensemble's spread (its implicit uncertainty) shrink over time. | M |
| 12.H | **Exploration-exploitation Pareto front** | Scatter each exploration policy by exploration cost vs cumulative regret; the Pareto frontier sweeps from too-little through optimum to too-much exploration. | M |
| 12.I | **Hoeffding sample-size calculator (for UCB)** | Sliders for `n`, `δ`; live Hoeffding bound plus the round-by-round UCB bonus `√(2 log t / N_k)` under different confidence schedules. | S |

### Chapter 13 — Model-based

| # | Widget | | Effort |
|---|---|---|---|
| 13.A | **MCTS tree animator** | Tic-tac-toe. Watch the search tree grow with UCT selection / expansion / rollout / backpropagation. Step-by-step or auto-play. | M |
| 13.B | **Dyna-Q planning steps** | FrozenLake. Slide `n_plan` from 0 to 100; watch convergence accelerate. | M |
| 13.C | **MCTS exploration-constant explorer** | Slide the UCT constant `c` on the tic-tac-toe tree; watch it morph from deep-narrow-exploit to shallow-wide-explore, with visit-entropy and PV-depth readouts. | M |
| 13.D | **PUCT prior-shape laboratory** | Paint a 9-vector prior and true Q-values; run PUCT and compare the resulting visit distribution against the prior and optimal, with a UCT toggle. | M |
| 13.E | **Compounding model-error visualizer** | Set per-step model noise; roll the learned dynamics forward and watch the true-vs-predicted divergence ribbon explode, plus the sim-vs-real return gap. | M |
| 13.F | **Dyna planning-step efficiency chart** | Return curves per `n_plan`; add model error and watch high-`n_plan` collapse below `n=0`, with an optional prioritised-sweeping toggle. | M |
| 13.G | **Dyna-over-real-world budget calculator** | Inputs for rollout depth, candidates, cadence, and agent count; compute extra world-ticks per agent-tick against a per-frame budget, with a feasibility heatmap. | S |

### Chapter 14 — Hierarchical

| # | Widget | | Effort |
|---|---|---|---|
| 14.A | **Four-rooms options demo** | Classic Sutton-Precup-Singh example. Toggle primitives only / options + primitives. Compare convergence. | M |
| 14.B | **Option duration → effective horizon** | Interactive: slide option duration `τ`; see `γ^τ` and option-level effective horizon update live. Visualizes Ch 14 §14.9's "10^19× improvement" claim concretely. | S |
| 14.C | **SMDP-Q bootstrap discount illustrator** | Drag option duration `τ` and per-tick rewards on a timeline; see the cumulative `Σγ^k r` and the `γ^τ max Q` bootstrap target as stacked bars. | M |
| 14.D | **Option-Critic termination dynamics** | A 2-room gridworld; vary deliberation cost `η` and watch the learned `β(s)` per option, with the termination-collapse diagnostic at `η=0`. | L |
| 14.E | **HIRO subgoal-relabeling demo** | Scrub stored high-level transitions and toggle original vs HIRO-relabeled subgoal (the displacement actually achieved), with the off-policy TD-target before/after. | L |
| 14.F | **DIAYN skill-coverage map** | State-visitation contours per skill index; slide the mutual-information weight to shift skills from overlapping to partitioned, with discriminator accuracy. | L |
| 14.G | **Four-rooms recipe-as-option flow field** | Build a recipe by dragging primitive-action arrows into a step list, treat it as an option, and run SMDP-Q over it plus four primitives. | M |

### Chapter 15 — FA pathologies

| # | Widget | | Effort |
|---|---|---|---|
| 15.A | **Baird's counterexample live** | The 7-state diverging-weights demo. Watch `θ` blow up under off-policy linear TD. Toggle: on-policy → stable. | M |
| 15.B | **Q-bias bootstrap visualizer** | The project's central bug. Show `Q` of all committed actions drifting toward `w_alive/(1-γ) = 10`. Toggle `w_alive = 0`; watch the lock-in dissolve. | M |
| 15.C | **Deadly-triad pairwise stability matrix** | A `2×2×2` cube of FA/bootstrap/off-policy toggles; pick any corner, run the matching algorithm, and read "stable / diverges / converges-but-biased". | M |
| 15.D | **Score-formula contribution decomposer** | Bars of each `Scorer`'s contribution per action (today: just `q_bias·Q`; future: advantage / UCB / intrinsic terms once the pipeline grows). Drag `w_alive` and `γ` and scrub time to watch a committed action's Q saturate and lock in the argmax. | S-M |
| 15.E | **Maximization-bias under FA** | The 4-arm noisy-Q environment with a linear-FA estimator; vary feature overlap and see FA amplify or damp the bias across Q / Double-Q / optimistic-init. | M |
| 15.F | **Primacy-bias parameter-reset experiment** | Learning curves under never-reset vs periodic parameter resets; watch early-bad-data lock-in vs a jagger-but-higher final curve. | M |
| 15.G | **Fix-comparison playground** | Three side-by-side homeostatic learners — baseline, Fix 1 (`w_alive=0`), Fix 3 (linear dueling) — showing Fix 1 and Fix 3 both resolve the lock-in. | L |

### Chapter 16 — Homeostatic reward

| # | Widget | | Effort |
|---|---|---|---|
| 16.A | **Reward landscape editor** | Drag drive levels in `[0, 1]`. See `R` recompute. Toggle convex exponent, weights, `w_alive`. Visualizes §16.5 directly. | S |
| 16.B | **Drive dynamics simulator** | Run hunger/thirst over time; watch agent take Consume actions when drives cross thresholds. | M |
| 16.C | **Convex-cost exponent comparator** | Plot `c(d)=d^p` and its derivative for slider `p`; paint a drive trajectory and compare linear (`p=1`) vs the Simulator's `p=2` marginal urgency. | S |
| 16.D | **Wanting-vs-liking dissociation timeline** | Side-by-side `Q` (wanting) and `r` (liking) on a hunger-eat trajectory; toggle "block dopamine" or "block opioids" and watch the two curves decorrelate. | M |
| 16.E | **Pareto-front scalarization explorer** | Paint a Pareto front; drag the weight vector and watch the linear optimum slide along the convex hull (missing concave parts) while Tchebysheff reaches them. | M |
| 16.F | **PBRS-equivalence visualizer** | Paint a potential `Φ(s)` on a gridworld; compare raw `R`, shaped `R+γΦ'−Φ`, and their optimal value functions — equal up to a constant. | M |
| 16.G | **RewardConfig slider panel** | Sliders mirroring `RewardConfig`'s fields; show a stacked-bar reward decomposition across three scenarios and watch the Consume-vs-Wait gap flip when `w_alive=0`. | S-M |
| 16.H | **Specification-gaming sandbox** | A boat-style env with a bonus-tile loop; raise `w_alive`/bonus past a threshold and watch the agent loop forever instead of finishing, then fix it with PBRS. | M |

### Chapter 17 — Long-horizon credit

| # | Widget | | Effort |
|---|---|---|---|
| 17.A | **γ^k decay catastrophe** | Slider for `γ` and `k`. Plot `γ^k` log-scale. Show the 10^-23 wall at `k=500`, `γ=0.9`. | S |
| 17.B | **RUDDER reward redistribution** | Synthetic delayed-reward MDP. Show original reward at step 100; LSTM redistributes; see dense reward at step 0 that credits the causal action. | L |
| 17.C | **Eligibility-trace horizon ribbon** | Set `λ` and `γ` on a 100-state chain; watch the `(λγ)^t` decay ribbon shrink toward the noise floor, with effective horizon `1/((1-γ)(1-λ))`. | S |
| 17.D | **PBRS shaping designer** | Pick a potential `Φ(s)` on an L-suite toy; run Q-learning under the shaped reward and check episodes-to-first-Harvest plus optimum-preservation. | M |
| 17.E | **HER replay-buffer relabeling animator** | Watch a failed goal-conditioned episode, then animate HER promoting each visited state to goal-of-the-day and re-scoring rewards, vs flat replay. | M |
| 17.F | **Successor-representation heatmap** | Click a state to draw `M^π(s,·)` over a four-rooms grid; paint rewards and watch `V^π_R = Σ M R` recompute instantly under sliders for `γ` and `π`. | M |
| 17.G | **RUDDER return-decomposition timeline** | A 100-step chain showing sparse reward vs redistributed `r̂_t`; toggle a partially-trained LSTM to watch the redistribution improve across epochs. | L |
| 17.H | **Method-stacking lab** | Toggle PBRS, SMDP-Q, and RUDDER in any combination on one L-suite toy; compare learning curves and confirm the fixes compose monotonically. | L |

### Chapter 18 — Action spaces

| # | Widget | | Effort |
|---|---|---|---|
| 18.A | **Parameterized-action P-DQN demo** | A toy `Strike{force}` env. Show continuous `force` learned vs. discretized buckets. | L |
| 18.B | **Discretization-cost visualizer** | Bucket a unimodal continuous Q-curve at `B` buckets; read off the per-step regret of `argmax_bucket Q` and create "boundary unfortunate" peak placements. | S |
| 18.C | **MP-DQN cross-type-bias diagnostic** | Run P-DQN (no zeroing) beside MP-DQN (with zeroing) on a 2-type PAMDP; the P-DQN Q-surface slice is biased while MP-DQN tracks the true Q. | M |
| 18.D | **H-PPO hybrid-head action sampler** | Edit a discrete softmax head and per-type Gaussian heads; sample many actions and see per-type marginals and a (type, parameter) joint scatter. | M |
| 18.E | **Wolpertinger k-NN retrieval demo** | Drag a proto-action over 100 scattered discrete actions; highlight the k nearest and re-rank by a Q-critic heatmap, varying `k` from rigid-greedy to critic-override. | M |
| 18.F | **Action-masking softmax inspector** | Drag `K` logits and toggle legality; compare the softmax before/after masking (illegal → −∞) against the penalty-in-reward alternative's residual mass. | S |
| 18.G | **Strike-force actor-critic playground** | A threat-response toy where optimal `Strike{force}` depends on valence; compare two-bucket, MP-DQN-continuous, and H-PPO heads via learning curves and the learned `force(threat)` map. | L |

### The big project-specific demo

**The Simulator-in-a-page.** A stripped-down version of the actual
project running in the browser. The student picks:

- Scenario (homeostatic-hungry, navigation, threat-response, L-suite
  farming).
- Algorithm (tile-coded Q vs. tabular vs. advantage-learning).
- Hyperparameters (`α`, `γ`, `ε`, `q_bias_weight`, `w_alive`).
- Observation knobs (vision range, memory window).

They watch agents act and learn over an episode-stream window, with
live TD-error and reward curves. **This is the user's Pac-Man idea,
specialised to the project's actual scenarios.**

The killer feature: a student reading Chapter 15 about the Q-bias
bootstrap pathology can *toggle `w_alive = 0` and watch the
pathology dissolve*, in the project's real environment.

---

## Implementation options — how to host this

GitHub-flavored markdown is static. None of the above runs natively.
Real options:

### Option 1: `mdBook` + JS plugins (Rust-ecosystem native)

[`mdBook`](https://rust-lang.github.io/mdBook/) is the Rust-ecosystem
static-site generator. Used by *The Rust Programming Language*, the
nomicon, Bevy book, dozens of crate docs. Generates HTML from
markdown; supports JavaScript preprocessors and custom themes.

**Pros**
- Already idiomatic for Rust projects (zero learning curve for repo regulars).
- Trivial GitHub Pages deploy.
- Math via `mathjax-support` or `katex` plugins.
- Custom JS goes in a `theme/index.hbs` override or per-page `<script>` tags via raw-HTML pass-through.
- Project's own crates can compile to WASM and be imported as JS modules from chapters.

**Cons**
- Plugin ecosystem is sparse compared to Quarto / Sphinx; complex interactive components mostly DIY.
- Hot-reload during authoring is OK but not great.

### Option 2: Astro + MDX

[Astro](https://astro.build/) is a modern static-site generator with
first-class MDX support — `.mdx` files are markdown that can `import`
JavaScript / TypeScript / Svelte / React / Vue components inline.

**Pros**
- Best-in-class authoring UX. Embed `<NormVisualizer />` directly in
  the chapter source like any other JSX element.
- Hot-reload, dev server, type-checked components.
- Ecosystem of math/notebook tooling (Observable, KaTeX, MathJax,
  D3, Three.js for 3D).
- Compiles to static HTML — GitHub Pages OK.

**Cons**
- Heavier than mdBook. Node toolchain (which the textbook lint already
  has). Build step every commit.
- Markdown → MDX migration is non-trivial: ~22 files need a header
  bump and the math-block syntax may differ slightly. Tractable but real.

### Option 3: Quarto

[Quarto](https://quarto.org/) is Pandoc-based, designed for technical
publishing with Observable JS integration. Used heavily by the
data-science academic community.

**Pros**
- Built-in cross-references, citations, equation numbering.
- Observable JS for interactive widgets is *very* expressive.
- Supports static HTML output → GitHub Pages.

**Cons**
- Less common in the Rust ecosystem; learning curve for the author.
- Observable is a specific dialect of JS (D3-flavored, FRP-style).

### Option 4: Stay on GitHub markdown + iframe to interactive widgets

Keep the textbook as-is on GitHub. Add `<iframe src="...">` tags
pointing at separately-hosted interactive demos (Observable notebooks,
CodePen embeds, custom widgets on a separate static host).

**Pros**
- Zero migration cost; existing GitHub rendering continues to work.
- Pick widget hosting per chapter as convenience allows.

**Cons**
- GitHub strips iframes from rendered markdown. So this **doesn't
  work on github.com** — you'd need to host the textbook somewhere
  that doesn't strip iframes (Cloudflare Pages, Vercel, plain S3).
- Two-source-of-truth problem: markdown for prose, separate hosting
  for widgets.

### Option 5: Don't change the platform; ship a companion web app

Keep the textbook in markdown on GitHub. Build a separate
`docs/textbook/companion/` web app
(Astro/Vite/whatever) hosted at `simulator-textbook.[domain]`.
Cross-link from each chapter: "**Live widget: norm visualizer →
\<link\>**".

**Pros**
- Decoupled: chapter prose stays portable + GitHub-renderable; the
  companion app grows at its own pace.
- The companion can crash / be down without breaking the textbook.
- Lower bar to ship the first widget — write it, deploy it, link to it.

**Cons**
- The widget isn't *next to* the explanation. Slightly worse pedagogy
  than inline embedding.

### Widget language decision (2026-05-20, revised)

**TypeScript for math viz, Rust for sim-integrated widgets.** Picks
the right tool per widget rather than enforcing a single toolchain.

| Widget category | Language | Stack |
|---|---|---|
| Math viz (norms, plots, geometric demos) | TypeScript ES modules | [D3 7](https://d3js.org/) + [Observable Plot 0.6](https://observablehq.com/plot/) |
| Interactive geometry (drag-points on a math surface) | TypeScript | [JSXGraph](https://jsxgraph.org/) — opt-in only when D3+Plot is awkward |
| Widgets that reuse project Rust code (`playground`, `rl_core`) | Rust + wasm-bindgen | Existing `widgets/<name>/` Cargo workspaces |

**Why mixed:**

- **Math widgets don't reuse Rust code.** A norm visualiser, a γ^k
  decay plot, a PPO clip surrogate, a contraction-mapping cobweb —
  none of these touch the simulator. Forcing them through wasm-bindgen
  + a per-widget Cargo workspace + `pkg/` artefacts is friction with
  no offsetting benefit.
- **TS authoring is the future-user-author path.** Long-term, readers
  may submit their own widgets. Plain `<script type="module">` blocks
  authored in TS or JS are an order of magnitude lower-friction than
  authoring a wasm-bindgen crate.
- **D3 + Plot is the canonical math-viz stack.** D3 is the bedrock
  (NYT, Mozilla, Observable, gov dashboards); Plot is the same
  team's declarative wrapper. Both ISC-licensed, both ~50 KB.
- **Rust widgets keep their justification.** Anything that compiles
  `playground`'s `run_episode` (bandit, gridworld, tile coding,
  future Simulator-in-a-page) stays Rust — running the production
  RL code in the browser is the point.

**Code shape:**

The γ^k decay widget in Plot (Chapter 17):

```html
<div id="ch17-gamma-widget" class="textbook-widget">
  <div class="widget-controls">
    <label>γ <input type="range" min="0.5" max="0.999" step="0.005"
                   value="0.9" data-input="gamma"></label>
    <span data-readout></span>
  </div>
  <div data-plot></div>
</div>
<script type="module">
  import * as Plot from "@observablehq/plot";
  import * as d3 from "d3";
  import { readNumber, autoRender } from "./widgets/shared/helpers.js";

  const host = document.getElementById("ch17-gamma-widget");
  autoRender(host, () => {
    const gamma = readNumber(host, '[data-input="gamma"]');
    const data = d3.range(501).map(k => ({ k, y: gamma ** k }));
    host.querySelector("[data-plot]").replaceChildren(Plot.plot({
      y: { type: "log", domain: [1e-30, 1.5] },
      marks: [Plot.line(data, { x: "k", y: "y" })],
    }));
  });
</script>
```

~30 lines for a chart with slider + log-scale axis + live redraw. The
Rust+canvas equivalent would be ~250 lines.

### Import-map: zero-build authoring

`theme/head.hbs` injects an importmap pinning d3 + Plot to specific
versions on jsdelivr's `/+esm` endpoint:

```html
<script type="importmap">
{
  "imports": {
    "d3": "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm",
    "@observablehq/plot": "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/+esm"
  }
}
</script>
```

Widgets `import * as d3 from "d3"` like an npm package; the browser
resolves it to the pinned URL. **No bundler. No transpile. No npm
install** for authoring a widget — write the `<script type="module">`
inline in the markdown chapter and it runs.

For type checking, `tsc --noEmit --checkJs` in CI catches type errors
without a runtime build step. Optional; widgets work without it.

### Shared helpers

`docs/textbook/widgets/shared/helpers.js` lifts the few patterns every
widget repeats: `readNumber(host, selector)`, `autoRender(host, render)`
(wires every `<input>` to re-call the render function on change),
common Plot styling. Each widget imports what it needs; no widget is
forced to use the helpers.

### mdBook decision

**mdBook with KaTeX, plus the import-map above for math widgets,
plus per-widget wasm-bindgen for sim-integrated widgets.** Reasons:

1. The textbook content is already 18 hand-authored Markdown chapters
   under `docs/textbook/`. mdBook reads them with **zero migration** —
   only `book.toml` + `SUMMARY.md` get added.
2. mdBook is the Rust-ecosystem native choice — used by *The Rust
   Programming Language*, the nomicon, Bevy book. Idiomatic for repo
   regulars; no JS framework decision to lock in.
3. Inline widget embedding is awkward in mdBook compared to MDX, but
   not blocked: each widget is a small `<div id="widget-X"></div>` +
   a `<script type="module">` that imports a WASM bundle. The widgets
   live in `docs/textbook/widgets/`, built by `wasm-pack` from the
   playground crate (which is already Bevy-free). The authoring story
   is "raw HTML for the mount point + a JS module for the logic" —
   acceptable for the handful of widgets per chapter.
4. The Node toolchain stays minimal — KaTeX validator is already
   Node; mdBook itself is a single Rust binary; `wasm-pack` is a
   single Rust binary. No Astro / Vite / React tree.

Astro + MDX remains the right answer if/when we want a **dozen
widgets per chapter** with rich JSX composition — at that point the
authoring leverage starts paying for the framework cost. Until then,
mdBook + sparse raw-HTML widgets is the smaller deliverable.

**Hosting:** GitHub Pages by default — free, automatic from a push
to `gh-pages` branch. Build artefact is a single `book/` directory.
A CI workflow can deploy on every push to `main`, or we build
locally and push to `gh-pages` manually.

### Where the widgets actually run

Three tiers, picked by widget complexity:

- **Pure TypeScript** — math visualizers, simple environments
  (gridworld, bandits, CartPole). HTML canvas or `<svg>` for
  rendering. ~200-500 lines per widget. Plenty fast for these
  problem sizes (CartPole training at 60 fps in a browser tab is
  routine).
- **TypeScript + Rust WASM modules** — when an inner loop is hot
  (tile-coding hash, large matrix multiply, neural net forward
  pass). Compile specific Rust crates to WASM, call from TS. Best
  of both: TS handles UI / orchestration / animation; WASM handles
  the math.
- **WASM-first** — for the Simulator-in-a-page. Compile the actual
  simulator crates to WASM. Largest engineering lift; reserved for
  the one demo where authenticity matters.

The math-only widgets (norm, eigenvector, Hessian, contraction
mapping, Hoeffding) are pure visualization — D3 or a small canvas
animation. Each is a few hundred lines.

### Implementation details — per-widget recipes

#### Norm visualizer (Ch 1.A) — pure TS, ~200 lines

```text
companion/src/widgets/NormVisualizer.tsx
├─ <svg> 400×400 canvas
├─ State: { p: number, S: number[][], v: [number, number] }
├─ Plot points where ‖x‖_p = 1 (the unit ball)
│  └─ For each θ in 0..2π, solve numerically for r such that ‖r·dir(θ)‖_p = 1
├─ Drag handle for v
└─ Live readout of ‖v‖_p
```

Math is in TS. No WASM needed; the unit ball drawing is 360 root-finder
calls (millisecond-scale).

#### Gridworld value iteration (Ch 4.A) — pure TS, ~400 lines

```text
companion/src/widgets/ValueIteration.tsx
├─ State: { grid: Cell[][], V: number[][], gamma: number, step: number }
├─ Bellman backup as a pure function
├─ <canvas> rendering with per-cell colour by V
├─ "Step" / "Run to convergence" buttons
└─ Sliders: gamma, theta (convergence threshold)
```

Pure TS. The state space is ~100 cells, value iteration converges in
30 steps, so animation is trivially smooth.

#### DQN on CartPole (Ch 9.A) — TS env + WASM agent, ~800 + 500 lines

```text
companion/src/widgets/CartpoleDQN/
├─ env.ts              ← CartPole dynamics in TS (60 fps animation)
├─ agent-wasm.ts       ← TS bindings to the WASM agent
├─ Component.tsx       ← UI with hyperparameter sliders + learning curve
└─ wasm/
   ├─ Cargo.toml       ← crate-type = ["cdylib"]
   └─ src/lib.rs       ← Q-net (small MLP) + experience replay
                          + target network. Exposes:
                            new_agent(seed, alpha, gamma, ...) -> AgentHandle
                            act(agent, obs) -> action
                            train_step(agent, batch) -> td_error
```

`wasm-pack` builds `src/lib.rs` into `pkg/`; Vite imports it as a JS
module. The CartPole env stays in TS because (a) it's tiny, (b) the
visualization needs frame-by-frame access to the physics state, (c)
TS→WASM boundary crossings are cheap but not free.

#### The Simulator-in-a-page — full WASM, biggest lift

```text
companion/src/widgets/SimulatorPage/
├─ Component.tsx       ← UI with scenario picker, hyperparameter
│                        sliders, agent observation, world view,
│                        learning curve, TD-error timeseries.
└─ wasm/
   ├─ Cargo.toml       ← Depends on workspace crates:
   │                      rl_core, q_learning, planner (sliced)
   ├─ src/lib.rs       ← Thin wrapper exposing:
   │                      new_world(scenario, seed, cfg) -> WorldHandle
   │                      step(world) -> { obs, reward, action, td_error }
   │                      stats(world) -> { reward_curve, q_values, ... }
   └─ src/scenarios/   ← Hand-translated minimum-viable versions of
                          learning_navigation, learning_homeostatic,
                          learning_suite_l. NOT full Bevy world —
                          stripped to: agent + drives + perception +
                          one cell type for food.
```

The honest accounting on reuse (**actual numbers, measured
2026-05-20** — see [`crates/tools/wasm_size_probes/`](../../crates/tools/wasm_size_probes/)):

- **What builds clean to `wasm32-unknown-unknown` today:**
  - `rl_core` (tile coder + `TileCodedValue` + `ValueFunction`):
    **27.3 KB raw / 11.1 KB gzipped**. Bevy-free via `--no-default-features`.
  - `q_learning` (`Learner` + TD update + observation schema):
    **27.7 KB raw / 11.3 KB gzipped**. Only ~400 bytes over rl_core
    because Learner is a thin wrapper over `TileCodedValue`.
  - `playground` (the run_episode harness + `MultiArmedBandit` +
    `GridWorld`): **37.9 KB raw / 15.3 KB gzipped**. The bundle a
    bandit/gridworld widget would actually ship — smaller
    than a typical web font.
- **What doesn't build to WASM yet:**
  - `crates/cognition/planner/` — currently a Bevy ECS system. The
    pure helpers `score_candidates`, `epsilon_greedy_pick`,
    `build_observation_pure` we extracted in Item 1 port directly;
    the rest is ECS plumbing that doesn't need to.
  - The full Bevy world (`crates/world_layer/`, `crates/substrate/`)
    won't compile to WASM without significant work AND is way too
    heavy for a textbook widget. For the Simulator-in-a-page demo,
    stub it: a `sim_minimal` crate that fakes a 16×16 gridworld and
    feeds it into the real `Learner` / planner core.

The **11-15 KB gzipped** measurements blow past the original
back-of-envelope "~200 KB" + "~500 KB" estimates by **7-30×** — tile
coding is a few hundred lines of arithmetic and the linker prunes
the rest. The Rust-backed widgets are dramatically cheaper to ship than the
proposal initially assumed.

So the answer to "can we re-use the simulator code?" is: **yes for the
RL algorithms (Learner, tile coding, scoring, epsilon-greedy), no for
the world / Bevy infrastructure**. The reuse pattern is "real
algorithms running against a toy world."

### TypeScript performance — concrete bounds

| Task | TS estimate | WASM estimate | Verdict |
|---|---|---|---|
| Compute one Q-update on a 16-tile coder | ~10 µs | ~3 µs | TS fine |
| 1000 Q-updates / frame for live training | ~10 ms | ~3 ms | TS fine (60 fps) |
| Train CartPole to solve (100k steps) | ~30 s | ~10 s | TS fine but WASM nicer |
| Train DQN on Atari frames | ~hours | ~tens of minutes | WASM mandatory; even then doubtful in-browser |
| Multi-agent (50 agents × tile coder) | ~50 ms / frame | ~10 ms / frame | WASM strongly preferred |

The browser's per-frame budget is 16 ms (60 fps). TS hits that bound
on simple problems comfortably; WASM gives ~3-5× headroom. For the
math widgets and tabular gridworlds, **TS is fast enough — don't
add WASM**.

**Web Workers** are the other dial: long training runs go in a worker
thread so the UI stays responsive. Both TS and WASM work in workers.
Pattern: main thread runs UI + animation; worker runs training; they
exchange snapshots via `postMessage`.

### How to choose TS vs. WASM per widget

| If the widget… | Use |
|---|---|
| Visualizes a static math object (norm, eigenvector, GD trajectory) | TS only |
| Runs a tabular RL algorithm on a gridworld or bandit | TS only |
| Has a small neural net (≤ 64 units) with online training | TS (TensorFlow.js or hand-rolled), or WASM |
| Uses the project's actual `Learner` / tile coder code | WASM via `rl_core` |
| Approaches an actual sim scenario (Pac-Man / Simulator-in-a-page) | WASM |

Decision rule: **start in TS; rewrite the hot loop in WASM only when
you can measure a frame-budget overrun**. Premature WASM-ification
adds build complexity for no benefit.

### Build pipeline

```text
companion/                          ← Astro project
├─ astro.config.mjs                 ← Astro + MDX + Vite + WASM plugin
├─ package.json                     ← Astro, React, TS, KaTeX, D3
├─ src/
│  ├─ pages/                        ← One .mdx per textbook chapter
│  │  ├─ 01_mathematical_foundations.mdx
│  │  └─ ...
│  ├─ widgets/                      ← One .tsx per widget
│  │  ├─ NormVisualizer.tsx
│  │  ├─ ValueIteration.tsx
│  │  └─ ...
│  ├─ layouts/                      ← Chapter shell, nav
│  └─ wasm/                         ← Generated by wasm-pack
│     ├─ rl_core_bg.wasm
│     ├─ rl_core.js                 ← TS bindings
│     └─ ...
└─ rust/                            ← Crates whose code we WASM-ship
   ├─ rl_core_web/                  ← Re-exports rl_core for wasm-pack
   ├─ sim_minimal/                  ← Toy world + Q-learning wrapper
   └─ widgets_dqn/                  ← CartPole DQN agent
```

`pnpm build` runs `wasm-pack build rust/*` then `astro build`. The
WASM bundles are versioned (`?v=<hash>`) so browsers cache safely.

Each chapter `.mdx` file embeds widgets inline:

```mdx
---
title: "Chapter 1 — Math Foundations"
---

import NormVisualizer from "../widgets/NormVisualizer";

## Norms — measuring "how big"

A norm assigns each vector a non-negative size…

<NormVisualizer initialP={2} />

The taxicab, Euclidean, and sup norms are the workhorses…
```

### Hosting

`pnpm build` produces a static `dist/` folder. Deploy options:

- **GitHub Pages** (free, **default**) — push `dist/` to `gh-pages`
  via the `peaceiris/actions-gh-pages` Action. Zero ops.
- **Cloudflare Pages** (free) — Cloudflare auto-builds from a GitHub
  hook, deploys globally. Slightly better cache behaviour for the
  WASM bundles.
- **Vercel** (free for personal) — similar to CF Pages.

All three serve WASM with the right MIME type by default. GitHub
Pages is the recommended default given there's no need for
serverless functions.

### Organisation: where the code lives

The proposal is to add **one top-level folder**: `companion/`. Sibling
to `crates/`, `docs/`, etc. Self-contained — its own `package.json`,
its own Rust crates for WASM-side code, its own deploy. The textbook
prose stays under `docs/textbook/` as markdown for now; the
`companion/` project mirrors each chapter as an `.mdx` file that
imports widgets.

Migration is gradual:

1. **Bootstrap:** create `companion/` with Astro + one MDX page that
   `<iframe>` references the existing `docs/textbook/00_index.md`.
   Compile, deploy, smoke-test the build pipeline. ~1 day.
2. **One real chapter:** convert chapter 1 to MDX, embed the norm
   visualizer. ~2 days.
3. **The rest:** convert chapters incrementally; non-converted ones
   stay as markdown iframes / external links.

At any point, the `docs/textbook/*.md` files remain canonical Markdown
and continue to render on GitHub. The companion site is an
*augmented* presentation layer. If the companion is broken, the
textbook is still readable on GitHub.

---

## Rollout status

The wrong move is to commit to building all 30+ widgets up front.
Right move: pick the smallest valuable thing, ship it, see if the
authoring + hosting workflow holds up, then iterate. What follows is
the rollout in that order — shipped pieces first, then the planned
work that remains.

### mdBook static site ✅ (2026-05-20)

**Done.** Concrete artefacts:

- `docs/textbook/book.toml` — mdBook config: KaTeX preprocessor,
  GitHub-style theme, mobile-friendly defaults.
- `docs/textbook/SUMMARY.md` — table of contents over the existing 18
  chapters + bibliography.
- LaTeX validation re-run against the rendered HTML (the dollar-sign
  survival check now operates against actual `book/`-output HTML
  rather than the markdown-it-texmath approximation we had before).
- `mdbook build` produces `book/`; `mdbook serve` opens locally.

**Deliberately deferred:** the GitHub Pages CI workflow. The user
previously deleted such a workflow because it pulled mdBook
binary tarballs unsolicited. If/when we add one, we either use a
checksummed pin or build locally + push to `gh-pages` manually.

### Shipped widgets

**43 widgets live as of 2026-05-21.** Three reuse the project's Rust
code via wasm-bindgen; forty are pure TypeScript math viz using
Observable Plot + D3 via the importmap in `theme/head.hbs`.

**Per-chapter coverage:**

| Chapter | Widgets shipped | Source dirs |
|---|---|---|
| 1 Math Foundations | 10 | norm, contraction, hoeffding, tower_property, jensen, policy_error, eigenvector, gradient_descent, markov_mixing, robbins_monro, matrix_norms (11) |
| 2 RL Problem | 5 | discount_factor, agent_loop, return_calc, ucb_bars, markov_vs_history |
| 3 MDPs | 4 | bellman_propagator, stochastic_vs_det, bellman_linear, reward_shape |
| 4 Dynamic Programming | 4 | value_iteration, pi_vs_vi, modified_pi, prioritized_sweep |
| 5 Monte Carlo | 5 | mc_vs_td, mc_first_vs_every, is_variance, exploring_starts, mc_cutoff |
| 6 TD Learning | 6 | gridworld (Rust), td_target, q_learning_cliff, sarsa_vs_q, expected_sarsa, n_step_td |
| 7 Eligibility Traces | 2 | lambda_return, effective_horizon |
| 8 Function Approximation | 1 | tile_coding (Rust) |
| 9 Deep Q-Learning | 1 | dueling |
| 11 Actor-Critic | 1 | ppo_clip |
| 12 Exploration | 1 | bandit (Rust) |
| 14 Hierarchical | 1 | option_horizon |
| 17 Long-Horizon Credit | 1 | gamma_decay |

**Chapters with remaining audit candidates:** 7 (3 more), 8 (6 more),
9 (6 more), 10 (5), 11 (4), 12 (7), 13–18 (33 across 6 chapters).
These candidates are now listed as rows in the per-chapter inventory
above.

Original tables below kept for reference / sources:

**Rust + wasm-bindgen (project-code reuse):**

| # | Widget | Chapter | Source | Built artifact |
|---|---|---|---|---|
| 1.1 | Bandit ε-greedy | [14](../textbook/14_exploration.md) | [`widgets/bandit/`](../../widgets/bandit/) | [`docs/textbook/widgets/bandit/pkg/`](../textbook/widgets/bandit/pkg/) |
| 1.2 | GridWorld Q-learning | [8](../textbook/08_temporal_difference_learning.md) | [`widgets/gridworld/`](../../widgets/gridworld/) | [`docs/textbook/widgets/gridworld/pkg/`](../textbook/widgets/gridworld/pkg/) |
| 1.3 | Tile-coding visualizer | [10](../textbook/10_function_approximation.md) | [`widgets/tile_coding/`](../../widgets/tile_coding/) | [`docs/textbook/widgets/tile_coding/pkg/`](../textbook/widgets/tile_coding/pkg/) |

**TypeScript + Observable Plot + D3 (pure math viz):**

| # | Widget | Chapter | Source |
|---|---|---|---|
| 1.A | Norm visualiser | [1](../textbook/01_linear_algebra.md) | [`norm/widget.js`](../textbook/widgets/norm/widget.js) |
| 1.E | Contraction mapping animator | [3](../textbook/03_mathematics_for_ai.md) | [`contraction/widget.js`](../textbook/widgets/contraction/widget.js) |
| 1.F | Hoeffding sample-size calculator | [2](../textbook/02_probability_and_statistics.md) | [`hoeffding/widget.js`](../textbook/widgets/hoeffding/widget.js) |
| 1.G | Tower-property bouncer | [2](../textbook/02_probability_and_statistics.md) | [`tower_property/widget.js`](../textbook/widgets/tower_property/widget.js) |
| 1.I | Convexity / Jensen explorer | [3](../textbook/03_mathematics_for_ai.md) | [`jensen/widget.js`](../textbook/widgets/jensen/widget.js) |
| 1.M | Greedy-policy error amplifier | [3](../textbook/03_mathematics_for_ai.md) | [`policy_error/widget.js`](../textbook/widgets/policy_error/widget.js) |
| 2.B | Discount factor explorer | [4](../textbook/04_the_rl_problem.md) | [`discount_factor/widget.js`](../textbook/widgets/discount_factor/widget.js) |
| 6.H | TD-target dissector | [8](../textbook/08_temporal_difference_learning.md) | [`td_target/widget.js`](../textbook/widgets/td_target/widget.js) |
| 7.B | λ-return horizon mixer | [9](../textbook/09_eligibility_traces.md) | [`lambda_return/widget.js`](../textbook/widgets/lambda_return/widget.js) |
| 7.C | (γ, λ) joint effective horizon | [9](../textbook/09_eligibility_traces.md) | [`effective_horizon/widget.js`](../textbook/widgets/effective_horizon/widget.js) |
| 9.B | Dueling decomposition | [11](../textbook/11_deep_q_learning.md) | [`dueling/widget.js`](../textbook/widgets/dueling/widget.js) |
| 11.A | PPO clipping visualiser | [13](../textbook/13_actor_critic.md) | [`ppo_clip/widget.js`](../textbook/widgets/ppo_clip/widget.js) |
| 14.B | Option duration → effective horizon | [16](../textbook/16_hierarchical_rl.md) | [`option_horizon/widget.js`](../textbook/widgets/option_horizon/widget.js) |
| 17.A | γ^k decay catastrophe | [19](../textbook/19_long_horizon_credit.md) | [`gamma_decay/widget.js`](../textbook/widgets/gamma_decay/widget.js) |

**Shared TypeScript utilities** in [`docs/textbook/widgets/shared/`](../textbook/widgets/shared/):

- `widget.js` — `defineWidget({ hostId, controls, slots, render })`
  eliminates ~30 LOC of scaffolding per widget.
- `stepper.js` — `defineStepper({ trajectory, render, ... })` for
  step-by-step animation widgets (matrix multiplication, MCTS, etc.).
- `helpers.js` — `palette`, `dashed`, `annotation`, `plotDefaults`,
  `readNumber`, `autoRender`, `setReadout`, `fmt`.
- `widgets.css` — `.textbook-widget` site-wide styling.

**Architecture decisions taken with the bandit template:**

- Rust owns both the algorithm AND the DOM. Each widget's Rust code
  mounts its own HTML + canvas into the host `<div>` and binds its own
  event handlers via `web-sys`. JS in the chapter is 3 lines: import
  the glue file, await `init()`, call `start(target_id)`.
- Widget sources live OUTSIDE `docs/textbook/` (under `widgets/<name>/`)
  because mdBook copies its source tree verbatim — we don't want
  `target/` or `Cargo.lock` ending up in `book/`. Only the `pkg/`
  artifacts (~85 KB raw / ~30 KB gzipped each) land inside the
  textbook tree.
- Each widget is its own standalone Cargo workspace (`[workspace]` at
  the top of its Cargo.toml) so its wasm32 build cache doesn't
  conflict with the simulator's native build cache.
- Path deps to `playground` / `rl_core` declared explicitly with
  `default-features = false` — the standalone-workspace pattern
  doesn't inherit the main workspace's Bevy-free default, so each
  widget Cargo.toml has to opt out.
- The wasm-pack `pkg/` artifacts are **committed** to git. CI doesn't
  rebuild them; the textbook workflow just runs `mdbook build` and
  ships the existing pkg/ alongside the chapters. Rebuild locally with
  `wasm-pack build --target web --release --out-dir <…>/pkg`.

See [`widgets/README.md`](../../widgets/README.md) for the Rust-widget
build pipeline. TypeScript widgets need no build step — `<script type="module">`
+ the importmap resolves the libraries at the browser.

### Expanded backlog from chapter audit (2026-05-21)

Three parallel audit agents went through every chapter and proposed
additional widget candidates beyond the original 1.A–18.A list.
**95 new candidates** were surfaced and have now been merged into the
per-chapter inventory above:

- chapters 1–6 — 28 candidates
- chapters 7–12 — 34 candidates
- chapters 13–18 — 33 candidates

Highest-leverage picks called out per audit: **6.D**
Expected SARSA, **8.G** set-as-vector (project-specific), **11.G**
advantage learning on a "sated arm" (project-specific), **15.G** Q-bias
fix walkthrough, **17.D** PBRS on the L-suite (project-specific). All
follow the TS-vs-Rust split the proposal codifies.

### Planned: more math foundations (9 widgets)

Once the WASM-widget pipeline is proven, build out chapter 1 + chapter
2 math widgets (1.A–1.F, 2.A–2.B). These are mostly pure D3 / canvas
math visualizations — they don't even need WASM. ~1–2 days each.

### Planned: more tabular RL (gridworlds, bandits) — 6 widgets

3.A–3.B, 4.A–4.B, 5.A, 6.A–6.C, 7.A. Same pipeline as the shipped
widgets; each adds a new environment under `playground::envs::*` or in
TS, depending on whether re-use of the project's TD code matters per
chapter.

### Planned: function approximation + deep RL (live training) — 5 widgets

8.A–8.C, 9.A, 11.A. Live training widgets are the most ambitious.
Each is roughly a week. Worth doing only after the tabular-RL widgets
prove the authoring workflow.

### Planned: the Simulator-in-a-page

The user's Pac-Man-ambition target. Requires:
- WASM build of a stripped Simulator (or a JS reimplementation of the
  minimum subset — perception + planner + Q-learning).
- UI for hyperparameter selection and live result rendering.

Realistically a multi-week effort. Probably worth doing only after the
tabular-RL widgets ship and the build pipeline is well-understood.

---

## Considered & rejected: in-textbook AI companion

We considered embedding a chat sidebar (RAG-grounded, citing
textbook + codebase) and rejected it: too much architectural
burden — needs a serverless function for the LLM API key, an
embedding pipeline, rate-limiting, per-session memory — and the
*workflow* substitute is trivial: open the textbook in one tab and
your LLM of choice in another, paste the chapter text in, ask
questions.

So: hosting decision goes back to "whatever is simplest." GitHub
Pages is fine. The textbook stays a static-export problem.

## Build-step validation benefit

Once we have an Astro build pipeline producing `dist/`, the linter
can grep the **built HTML** for surviving `$` signs — exactly
what GitHub's renderer would produce but inspected locally before
pushing. That's a strict improvement over the current
markdown-it-texmath approximation: we're checking the actual output
GitHub would show.

Honest limit (user-noted): this still doesn't catch *visual*
rendering bugs (misaligned matrix columns, baseline issues, spacing
quirks). Those need a human eye or screenshot-diff testing — out of
scope for a CLI linter.

The Astro build pipeline gets us "validate built HTML" for free
once we have the site. The current Node-based lint is interim;
once we're building, we run the same `$`-survival check against
`dist/` instead of approximating with markdown-it.

---

## Open questions

1. **Authoring substrate — markdown or MDX?** MDX gives inline
   widgets but breaks raw-GitHub-render. Acceptable if we accept
   the textbook's "real home" is the hosted site, not the GitHub
   repo's rendered files.
2. **Where do the widgets live in the repo?** Options: alongside
   the chapter (`docs/textbook/widgets/`) or in their own crate-
   sibling (`docs/textbook/companion/`). Matters for the build
   pipeline.
3. **How much do we re-implement vs. import?** A pure-TS bandit demo
   is faster than a WASM build of the project's bandit code, but the
   project's code is what the textbook references. Mixing is fine
   per-widget.
4. **Is hosting on `<custom-domain>.pages.dev` or `github.io` OK?**
   The latter is free + automatic; the former needs a DNS choice.
5. **Do we localise widgets to chapter content?** Or build generic
   demos (e.g. one "bandit demo" referenced from chapters 2, 12, 15)?
   Per-chapter localisation is more work but better pedagogically.

---

## What this would cost

Honest estimate, presuming Astro + MDX:

| Phase | Scope | Effort |
|---|---|---|
| 0 | Pick stack, deploy first widget | ~1 day infra + ~0.5 day widget |
| 1 | 10 math widgets | ~10 days |
| 2 | 6 tabular-RL widgets | ~12 days |
| 3 | 5 deep-RL widgets | ~25 days |
| 4 | Simulator-in-a-page | ~30 days |

Roughly 80 working days end-to-end. Phases 0–2 (~25 days) deliver
~75% of the pedagogical value; phases 3–4 polish.

The Simulator-in-a-page alone is worth a separate proposal — it
intersects engine architecture (WASM build), content (which
scenarios to expose), and product (UI for hyperparameter sliders).

---

## Decision (2026-05-20)

**The mdBook site is done.** The first widgets (the three
bandit/gridworld/tile-coding widgets) are the next concrete step, to
be scheduled separately once the site sees real-world use and any
rough edges in the mdBook setup surface.

The remaining widget tranches stay deferred — to be picked up only
once the first widgets prove the authoring + WASM workflow at scale
and the textbook is the user's actual primary reading surface.

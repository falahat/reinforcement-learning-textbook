# Chapter 18 — Homeostatic RL and Multi-Objective Reward

> **Prerequisites:** Chapter [3](05_mdps_and_bellman_equations.md)
> (reward functions), Chapter [15](17_fa_pathologies.md) (the Q-bias
> pathology — homeostatic reward is the *cause*).

> **Citations:** [Keramati & Gutkin 2011, 2014] (homeostatic RL);
> [Berridge & Robinson 1998, 2016] (wanting vs liking);
> [Roijers et al. 2013] (MORL survey); [Ng, Harada & Russell 1999]
> (potential-based reward shaping); [Cabanac 1971] (alliesthesia).
> Full entries in [`bibliography.md`](bibliography.md).

> **Learning objectives:**
> 1. State the homeostatic RL framework and derive its drive-dependent
>    reward.
> 2. Distinguish *wanting* (motivation, dopaminergic) from *liking*
>    (hedonic, opioid) — the Berridge-Robinson dissociation — and map
>    each onto an RL quantity.
> 3. Compare linear, Tchebysheff, and smoothed-Tchebysheff multi-objective
>    scalarizations and explain which Pareto-front geometries each can
>    reach.
> 4. Dissect the Simulator's `RewardConfig` line-by-line and show how
>    `w_alive = 1.0` causes the Q-bias bootstrap pathology.
> 5. Apply potential-based reward shaping to the Simulator's
>    `level` vs. `delta` reward debate without changing the optimal
>    policy.

## Why this chapter exists

Most RL textbooks assume the reward function is *given*. In an embodied
agent — a creature with bodies, drives, needs — the reward is
**derived from physiological state**. Hunger, thirst, fatigue,
loneliness, fear: these are drives, and reward is the agent's
moment-by-moment relief from (or accumulation of) drive deviation.

This is **homeostatic RL**: the framework that makes "reward" a
function of bodily state rather than an exogenous signal. The
Simulator implements it directly
([`crates/sim/sim_config/src/reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)),
and the chapter's central object is the formula

$$R(s, s') = w_\text{alive} - \sum_d w_d \cdot d^{c} - w_\text{blood} \cdot \text{blood\_loss} - w_\text{temp} \cdot |\text{temp deviation}|.$$

Every term has biological motivation and engineering consequences. We
work both.

## Table of contents

- [16.1 Homeostatic RL — the framework](#161-homeostatic-rl--the-framework)
- [16.2 Wanting vs liking — Berridge-Robinson](#162-wanting-vs-liking--the-berridge-robinson-dissociation)
- [16.3 Multi-objective scalarization](#163-multi-objective-scalarization)
- [16.4 Level vs. delta reward and PBRS](#164-level-vs-delta-reward-and-potential-based-shaping)
- [16.5 The Simulator's `RewardConfig` dissected](#165-the-simulators-rewardconfig-dissected)
- [16.6 The Q-bias pathology, again, with reward-function eyes](#166-the-q-bias-pathology-again-with-reward-function-eyes)
- [16.7 Reward hacking, specification gaming, RLHF](#167-reward-hacking-specification-gaming-and-rlhf)
- [16.8 Project tie-in](#168-project-tie-in)
- [16.9 Exercises](#169-exercises)
- [16.10 References](#1610-references-cited-in-this-chapter)
- [16.11 Further reading](#1611-further-reading)

---

## 16.1 Homeostatic RL — the framework

### Why this framework matters

Standard RL textbooks treat the reward function as an *input*: the
environment hands you $r_t$, and your job is to learn a policy that
maximizes expected discounted return. That framing is mathematically
clean but biologically wrong, and *very* misleading for designing
embodied agents.

In a real animal — or a Simulator-grade agent with a body — the
reward is **derived from internal physiological state**. Hunger
isn't a number the world gives you; it's the gap between your
current energy level and a setpoint, and you experience that gap as
discomfort. The reward signal is a function of *how much you've
relieved (or accumulated) drive deviation*, not an arbitrary input.

This isn't just biological window-dressing. Three engineering
consequences fall out immediately:
1. **Reward shaping is implicit in the body design.** Choosing
   convex vs linear drive costs changes the optimal policy. The
   Simulator's `RewardConfig` is *literally* a hyperparameter on
   the agent's psychology.
2. **The Q-bias bootstrap pathology** (§16.6, also Ch17 §15.2) is
   caused by a specific term in the homeostatic reward formula —
   the `alive` baseline. Get the homeostatic structure wrong and
   you break learning in a way no amount of algorithmic tuning can
   fix.
3. **Multi-objective RL** (§16.3) becomes the natural framework
   because you have *many* drives, each with its own setpoint and
   cost. Scalarizing them into one reward is a design choice with
   real Pareto-front consequences.

This chapter is the design document for the Simulator's actual
reward function. Every later RL textbook implicitly assumes the
reward is given; here we derive it from first principles and
discover *what we'd want to change* about the Simulator.

[Keramati & Gutkin 2011, 2014] proposed that animal reward signals are
**derivatives of internal-state deviations** from setpoint. Each drive
$d$ has:

- A current level $d \in [0, 1]$ (0 = setpoint, 1 = lethal deviation).
- A **cost function** $c_d(d) = d^{p_d}$ (convex if $p_d > 1$).
- A weight $w_d$.

The instantaneous discomfort is $\sum_d w_d \, c_d(d)$. The reward is
**reduction** in discomfort, plus an "alive baseline" $w_\text{alive}$:

$$R(s, s') = w_\text{alive} + \big[\text{discomfort}(s) - \text{discomfort}(s')\big] - \text{bio\_costs}.$$

A neutral tick (no drive change) gives $R = w_\text{alive}$. A
food-eating tick (hunger drops from 0.6 to 0.1) gives a large positive
delta. A starvation tick (hunger creeps from 0.6 to 0.65) gives a small
negative delta.

### Why convex cost?

Three justifications:

1. **Marginal urgency.** At low hunger, an extra unit doesn't matter
   much. At high hunger, it matters disproportionately. Convex cost
   ($p > 1$) reproduces this — the slope $c'_d(d) = p \cdot d^{p-1}$
   grows with $d$.
2. **Biological evidence.** [Cabanac 1971]'s "alliesthesia": food tastes
   better when you are hungrier. The same stimulus's pleasure scales
   with the underlying need.
3. **Stability.** A convex cost is locally well-approximated by a
   quadratic (Taylor expansion at any drive level), making homeostatic
   control mathematically related to LQR (linear-quadratic regulation).

The Simulator uses $p = 2$ (quadratic, $w_d = 0.15$ uniformly across
cognitive drives; see [`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)
field `cost_exponent`).

### Three readings of the homeostatic reward

**1. As a temporal-difference signal on drive state.** The reward
$\text{discomfort}(s) - \text{discomfort}(s')$ is *by construction*
a difference between two discomfort values — a one-step TD-error
on the discomfort signal. An action that reduces discomfort
generates positive reward; an action that increases it, negative.
This makes the agent's Q-function effectively a "rate-of-relief"
predictor.

**2. As Lyapunov stability around a setpoint.** The drive vector
$d$ has a setpoint $d = 0$ (perfect homeostasis). Convex cost
$\sum_d w_d d^p$ is a Lyapunov-like function: zero at the
setpoint, positive elsewhere, increasing radially. The agent
that maximises reward is implicitly minimising this Lyapunov
function — driving the body's state back toward setpoint.

**3. As multi-objective scalarization with weights $w_d$.** Each
drive contributes its own cost; the weighted sum scalarizes a
*vector* of objectives into one number. Different $w_d$ choices
yield different Pareto-optimal policies (§16.3 develops this).
The Simulator's uniform $w_d = 0.15$ is a particular point on the
Pareto front.

### What this framework doesn't say

- **It doesn't specify the cost exponent $p$.** $p = 2$ is the
  Simulator's choice; biology argues for $p \in (1, 3)$ depending
  on the drive. The choice changes optimal-policy behaviour.
- **It doesn't tell you the right $w_\text{alive}$.** Too low and
  the agent learns to suicide on bad ticks; too high and it dilutes
  the drive signal. §16.6 traces this to the Q-bias bootstrap
  pathology.
- **It assumes drives are decoupled.** Eating raises blood-sugar
  but also takes time (raising exhaustion). The cost-sum treats
  each drive independently — *cross-drive coupling* (eating raises
  exhaustion → indirect effect on $\text{exhaustion}^2$ cost)
  isn't represented. For most agents this is acceptable; for some
  it's an over-simplification.

<div id="ch16-convex-cost-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/convex_cost/widget.js"></script>

Slide $p$ from $1$ to $3$ and watch the marginal-urgency curve
$c'(d) = p \cdot d^{p-1}$ flip from flat (no alliesthesia) to steep
(strong urgency at high drives). The reference-bar panel shows how a
single $w_d \cdot d^p$ contribution swings across three drive levels.

### Why the alive baseline?

$w_\text{alive}$ is the per-tick "just being alive is worth something"
term. Three roles:

1. **Avoiding suicide.** Without it, any drive-relief action competes
   with "die now, drive deviations end" — formally, terminal states
   give reward 0 (no discomfort), which can be better than living with
   high drive deviation. $w_\text{alive}$ makes living strictly
   preferable when drives are manageable.
2. **Default valence**. A neutral tick has positive value, encoding
   "existence is good" baseline.
3. **Bootstrap floor**: more on this in §16.6 — *this is also the cause
   of the project's Q-bias pathology*.

### Bio costs

Beyond drives, two body-level penalties:

- `w_blood_loss`: damage tracking (0.5 multiplier per unit blood loss
  per tick).
- `w_temp_deviation`: thermal homeostasis (0.3 multiplier per unit
  deviation from setpoint).

These represent fast-acting biological costs that aren't classifiable
as cognitive drives. Their convex form is the same family.

<div id="ch16-reward-landscape-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/homeo_reward_landscape/widget.js"></script>

Drag the three drive sliders and watch the per-term decomposition of
$R$ stack up: a positive $w_\text{alive}$ pillar and three negative
$-w_d \cdot d^p$ pillars. The marginal-urgency panel marks each drive
on the $c'(d)$ curve so the dominant drive is visible as the rightmost
dot.

<div id="ch16-drive-dynamics-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/drive_dynamics/widget.js"></script>

Drives rise linearly; on crossing a Consume threshold $\theta$ they
snap back to $0$. The top panel shows the two trajectories with
triangles marking Consume events; the bottom panel shows the per-tick
homeostatic reward $R_t$ dipping toward zero as drives saturate and
recovering after each Consume.

---

## 16.2 Wanting vs liking — the Berridge-Robinson dissociation

A central neuroscience finding ([Berridge & Robinson 1998, 2016])
is that **motivation and pleasure are separable systems** in the brain:

- **Wanting** ("incentive salience"): dopamine-mediated; pushes the
  organism to *seek* a reward.
- **Liking** ("hedonic impact"): opioid-mediated; the actual pleasure
  experienced *during consumption*.

The two are normally correlated (wanting food predicts liking food),
but pharmacological manipulations dissociate them: drugs that block
dopamine reduce wanting without reducing liking (a rat will eat sweet
food it does not seek out); drugs that block opioids reduce liking
without changing wanting (a rat seeks sweet food without enjoying it).

### Mapping onto RL

- **Wanting** ≈ **$Q(s, a)$** (or equivalently, *expected* future
  reward). It is the agent's pull toward an action.
- **Liking** ≈ **$r(s, a, s')$** (or specifically, the spike in reward
  during the consummatory transition). It is the actual sensation.

This is more than philosophy. It guides reward-function design:

1. **Hunger relief is hedonic** (high liking when consuming food).
2. **Going-to-find-food is motivated** (high wanting via TD-propagated
   $Q$).
3. **Both should be present** for healthy behavior.

The Simulator implements:
- *Wanting*: tile-coded $Q$ via TD learning (Chapter 8).
- *Liking*: large positive transient reward at the Consume step (from
  drive relief: hunger drops by 0.5, reward spike of
  $w_\text{drive} \cdot (0.6^2 - 0.1^2) \cdot 0.5 \approx 0.05$).

The mismatch is illustrative: the Simulator's "liking" spike is small
relative to the $w_\text{alive}$ contribution to $Q$. This is part of
the pathology (§16.6): wanting saturates to $Q \approx 10$ via the
alive baseline, drowning out the liking spike of $\sim 0.05$.

### Try it: wanting climbs, liking saturates

<div id="ch16-wanting-liking-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/wanting_liking/widget.js"></script>

A stylised Berridge model. Each tick marked on the top chart is a "use" event; between uses both signals decay. Liking is bounded — it climbs to a ceiling and then drops back as the tolerance term kicks in. Wanting *sensitises*: each use adds a fixed fraction of the current wanting, so the curve grows multiplicatively without bound. The bottom panel is the gap $W_t - L_t$ — the addiction-like signal where the agent still feels intense pull (high wanting) without proportionate pleasure (saturated/declining liking). Slide the frequency up and the dissociation widens; slide the tolerance to zero and the gap shrinks because liking stays high.

### Project tie-in: drives implement the *valence* of cognitive states

The Panksepp drive taxonomy
([`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md) §Affect)
splits cognitive drives into 8 (Seeking, Lust, Vigilance, Resentment,
Care, Loneliness, Boredom, Revulsion). Each has a setpoint, accumulates
deviation when unmet, and contributes to the reward formula. The
naming convention (Vigilance, Boredom, Loneliness) describes the
*un-met sustained state* — i.e., the negative wanting signal — making
each `DriveKind` semantically unique from its dual Plutchik prime
(Fear, Disgust, Sadness, etc.).

---

## 16.3 Multi-objective scalarization

The "reward" so far is implicitly multi-objective: it has 14 drives,
2 bio terms, and the alive baseline. Scalarizing them into a single
number is a choice with consequences.

### Linear scalarization

$$R = w_\text{alive} - \sum_d w_d \cdot c_d(d) - \text{bio\_costs}.$$

This is the Simulator's choice — linear with fixed weights.

> **Theorem.** Linear scalarization of a multi-objective MDP yields a
> reward whose optimal policy is the corresponding point on the
> *convex* part of the Pareto front. The non-convex regions are
> unreachable by any choice of $w$.

So if there is a policy that does well on drives A and B but at
some non-convex trade-off, no linear scalarization can target it.
For most practical RL this is a minor issue (the convex region usually
covers what we care about).

### Tchebysheff scalarization

$$R_\text{Tcheb} = -\max_d w_d \cdot |c_d(d) - z_d^{\star}|$$

where $z_d^{\star}$ is a target value per objective. This **does** reach
the non-convex Pareto front. The cost: not differentiable (the max is
discontinuous in $d$).

### Smoothed Tchebysheff

$$R_\text{smooth} = -\frac{1}{\tau} \log \sum_d \exp(\tau w_d c_d(d))$$

using log-sum-exp as a smooth max with sharpness $\tau$. As $\tau \to \infty$,
recovers Tchebysheff; finite $\tau$ is differentiable.

### Which to use?

| Scheme | Pareto coverage | Differentiable? | Tunability |
|---|---|---|---|
| Linear | Convex hull only | Yes | One weight per objective |
| Tchebysheff | Full Pareto front | No | Weights + target |
| Smoothed Tchebysheff | Full (in the limit) | Yes | Weights, target, $\tau$ |

The Simulator's linear choice is appropriate because: (a) the convex
hull is enough for the validation cases; (b) linear is the cheapest;
(c) Tchebysheff would require tuning a target $z^{\star}$ per drive, and
those don't have natural setpoint values in a meaningful unit.

### When linear breaks

It breaks when objectives are competitive: drive A and drive B
*cannot both be satisfied*. Linear scalarization picks a single
trade-off (whoever weighs more wins); Tchebysheff can settle at a
balanced point on the non-convex Pareto front.

The Simulator hasn't hit this problem yet — the drives are mostly
non-competitive (eating, drinking, sleeping don't conflict). It could
become relevant if Lust and Fear get into a real competition in some
content scenario.

<div id="ch16-pareto-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/pareto_scalarization/widget.js"></script>

Five candidate policies sit on the 2-objective plane; rotate the
weight vector $w = (\cos\theta, \sin\theta)$ and watch the three
highlighted rings (linear / Tchebysheff / smoothed) slide between
them. Policy D sits inside the convex hull — linear scalarization
*never* picks it, no matter the $\theta$, while Tchebysheff does.

---

## 16.4 Level vs. delta reward and potential-based shaping

A subtle but consequential design choice: should reward be the
**level** of well-being $R(s) = -\text{discomfort}(s) + w_\text{alive}$,
or the **delta** $R(s, s') = \text{discomfort}(s) - \text{discomfort}(s')$?

The Simulator uses *level* form (well-being at the new state). A delta
form is mathematically equivalent up to a potential-based shaping
correction. This is the [Ng, Harada & Russell 1999] result:

> **Theorem (PBRS).** Given any function $\Phi: \mathcal{S} \to \mathbb{R}$,
> the shaped reward
>
> $$
> \tilde R(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)
> $$
>
> yields the **same optimal policy** as $R$ on the same MDP.

If we set $\Phi(s) = -\text{discomfort}(s) / (1 - \gamma)$, then the
delta-form reward becomes equivalent to the level-form reward plus a
PBRS correction. So the choice between level and delta is *just a
shaping*: same optimal policy, different learning dynamics.

### Why the choice still matters for *learning*

PBRS preserves the optimal policy in the limit, but:

- The **learning trajectory** is different (one form may have lower
  variance per update).
- The **convergence rate** of TD differs (level-form rewards are
  bounded; delta-form can be unbounded if drives jump).
- The **fixed point** of constant $\alpha$ TD differs by $O(\alpha)$,
  even though the limit $\alpha$-zero fixed point is the same.

The level form was chosen because it's the more direct biological
interpretation ("how good is it to be here, right now?"). The delta
form would be equivalent to "alliesthesia": pleasure is
*reduction* in need. Both have biological adherents.

<div id="ch16-pbrs-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/pbrs_invariance/widget.js"></script>

Pick a potential shape $\Phi(s)$ and watch the raw and shaped $V^{\star}$
heatmaps side by side. The values change with $\Phi$ but the greedy
policy arrows stay identical, and the residual
$|V^{\star} _\text{shaped} - V^{\star} _\text{raw} + \Phi|$ stays at numerical
zero — Ng-Harada-Russell's theorem holding in real time.

### PBRS as the Q-bias fix?

You might think: "If we add the PBRS shaping that converts level to
delta, does $w_\text{alive}$ go away?" Yes, *for the optimal policy*.
But the *learning* still has the Q-bias problem because the bootstrap
target uses the un-shaped Q. Concretely, $w_\text{alive}$ remains in
the *form* of the reward; PBRS only adds another term. To kill it
properly you have to remove it from the source — which is exactly how
the pathology was fixed: set $w_\text{alive} = 0$ so reward is the
per-tick drive-delta.

---

## 16.5 The Simulator's `RewardConfig` dissected

Direct line-by-line from
[`crates/sim/sim_config/src/reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs):

```rust
pub struct RewardConfig {
    pub w_alive: f32,            // 1.0
    pub drive_weight: f32,       // 0.15
    pub cost_exponent: f32,      // 2.0
    pub w_blood_loss: f32,       // 0.5
    pub w_temp_deviation: f32,   // 0.3
    pub emotion_decay: f32,      // 0.85
    pub urgency_floor: f32,      // 0.05
    pub emotion_drive_coupling: f32,  // 0.02
    pub intox_dull_disgust: f32,      // 0.6
    pub intox_dull_sadness: f32,      // 0.5
    pub intox_courage_factor: f32,    // 0.5
    pub adrenaline_amplify_fear: f32, // 0.7
    pub adrenaline_amplify_anger: f32,// 0.5
}
```

| Field | Role | Empirical bound |
|---|---|---|
| `w_alive = 1.0` | Per-tick alive baseline | Saturates Q to 10 (Q-bias pathology) |
| `drive_weight = 0.15` | Multiplier on each drive's quadratic cost | At $d = 1.0$ contributes $0.15$ per drive, 14 drives → up to $2.1$ |
| `cost_exponent = 2.0` | Quadratic convex cost | Lower at low drives, sharply higher at high |
| `w_blood_loss = 0.5` | Damage cost | Direct multiplier |
| `w_temp_deviation = 0.3` | Thermal homeostasis | Linear in temp delta |
| `emotion_decay = 0.85` | Per-tick decay of emotion intensity | Half-life $\sim 4.3$ ticks |
| `urgency_floor = 0.05` | Below this, emotion treated as zero | Numerical clean-up |
| `emotion_drive_coupling = 0.02` | Rate emotion leaks into cognitive drives | $\dot d_\text{cog} = 0.02 \cdot \text{emotion intensity}$ |
| Intox & adrenaline mods | Pharmacological-style modulators on specific emotion channels | Used in content scenarios |

### Range of the reward signal

A "perfectly happy" tick: $R \approx 1.0$ (just alive baseline, no
drive deviation, no bio cost).

A "miserable" tick: $R \approx 1.0 - 0.15 \cdot 14 \cdot 1.0^2 - 0.5 - 0.3 \approx -1.9$
(all drives maxed, full blood loss, full temp deviation).

A "drive relief" tick (Consume, hunger 0.6 → 0.1):
$R_\text{baseline} \approx 1.0 - 0.15 \cdot (0.1^2) - \ldots \approx 1.0$
where the previous tick had $R \approx 1.0 - 0.15 \cdot (0.6^2) - \ldots \approx 0.95$.
So **the immediate reward bump from Consume is only $\sim 0.05$** —
$5\%$ of the alive baseline. This is the key number.

### The reward signal's signal-to-noise ratio

For policy gradient, what matters is the *advantage* — reward gap
between actions. The Consume advantage at hunger 0.6 is
$\sim 0.05$ (the relief from $0.6^2$ to $0.1^2$ × 0.15).
The $w_\text{alive}$ contribution to *any* action is $1.0$.

**Ratio: 1 : 20.** Consume's discriminating signal is overwhelmed by
the alive baseline. This is the homeostatic-reward-design root cause
of the Q-bias bootstrap pathology — not a tile-coder bug, not a TD
bug, but a reward-design bug.

<div id="ch16-reward-config-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/reward_config_panel/widget.js"></script>

Five sliders mirror `RewardConfig`. Three scenarios — sated-Wait,
hungry-Wait, hungry-Consume — show stacked decompositions of $R$ and
the implied $Q^* = R / (1 - \gamma)$. Drag $w_\text{alive}$ to $0$
and watch the hungry-Consume bar overtake hungry-Wait: the Q-bias
gap flips sign in front of your eyes.

---

## 16.6 The Q-bias pathology, again, with reward-function eyes

Chapter 17 (next, or already studied) covers the Q-bias bootstrap
pathology in depth. Here we look at it from the reward-design angle.

The TD recurrence at every committed action is

$$Q \leftarrow Q + 0.1 \big(R + \gamma \cdot Q' - Q\big).$$

Substitute $R = w_\text{alive} - 0.15 \sum_d d^2 - \ldots$. For a
typical mid-deviation drive vector (each drive $\sim 0.3$):

$$R \approx 1.0 - 0.15 \cdot 14 \cdot 0.09 - 0.0 - 0.0 \approx 0.81.$$

Q's fixed point (assuming $Q' = Q$) is

$$Q^{\star} = R / (1 - \gamma) = 0.81 / 0.1 = 8.1.$$

So Q drifts toward $\sim 8$ for any committed action. The same number,
the same target, for Plant, Consume, Step, Wait — whatever gets
committed.

The action-discriminating signal — the $0.05$ Consume reward bump —
contributes $0.05 / (1 - 0.9) = 0.5$ to Consume's $Q^{\star}$ relative
to a Wait action's. **But** the Wait Q also climbs to $\sim 8$ from
its own committed-tick rewards; the $0.5$ gap is within the noise
floor ($O(\sqrt{\alpha}) \approx 0.32$ at $\alpha = 0.1$).

### Three fixes, restated in reward-function language

Three fixes were considered; the pathology was resolved by the first
(remove $w_\text{alive}$, making reward the drive-delta):

1. **Fix 1 (set $w_\text{alive} = 0$).** Reward becomes pure delta:
   $R = -0.15 \sum_d d^2 - \ldots$. Now $Q^{\star} = -1.21 / 0.1 = -12.1$
   for any committed action — same problem, opposite sign. *Unless*
   we use the delta form
   $R = \text{discomfort}(s) - \text{discomfort}(s')$, which gives
   zero reward at neutral ticks and positive only at drive-relief
   moments. Then $Q^{\star}$ is dominated by drive-relief events, which
   discriminate actions cleanly.
2. **Fix 2 (per-action cost).** Add a small `action_cost` to
   $R$ for each non-Wait action, so that some actions are intrinsically
   cheaper than others. Recovers discrimination but requires hand-
   tuning the per-action costs.
3. **Fix 3 (advantage learning, Chapter 13 §11.7).** Decouple
   $V(s)$ (which absorbs $w_\text{alive}$) from $Q(s, a) - V(s)$
   (which carries action discrimination). Architectural rather than
   reward-design.

Fix 1 is the simplest and most principled, and it is the one that
shipped. This chapter sides with Fix 1 because **the alive baseline is a
reward-design choice, not a structural necessity** — removing it cleans
up the math without breaking other systems.

---

## 16.7 Reward hacking, specification gaming, and RLHF

A broader caveat: any hand-crafted reward function risks **encoding
the wrong objective**. The literature catalogues many failures:

- **Reward hacking**: agent finds an unintended way to maximize the
  proxy. The CoastRunners agent that drives in a circle hitting bonus
  targets rather than racing. [Krakovna et al. 2020] curated dozens
  of examples.
- **Specification gaming**: subtler — the reward technically captures
  intent but allows degenerate optima. A grasping robot that holds
  the box at the camera's focal point to look like it's holding it.

The Simulator is not immune. A naive agent given high $w_\text{alive}$
might learn to never act at all (the all-Wait policy maximizes
aliveness if no drives ever rise — possible if you start at setpoint).
The current `RewardConfig` mitigates this by making drives accumulate
inexorably (hunger rises over time regardless of action), forcing some
relief behavior.

**Modern mitigations**:

- **Inverse RL** [Russell 1998; Ng & Russell 2000]: infer the reward
  from expert behavior. Useful when you have demonstrations but not a
  reward.
- **RLHF** [Christiano et al. 2017; OpenAI 2022]: learn the reward
  from human preference judgments. The dominant approach in
  language-model fine-tuning.
- **Reward modeling**: train a neural network to predict reward from
  state/action, supervised by human labels.

The Simulator is closer to hand-crafted homeostatic reward (RL of the
1990s-2000s flavor) than to RLHF-style learned-reward RL. Its
biological motivation is what makes the reward design defensible
without human labels — drives are mechanistic.

<div id="ch16-spec-gaming-widget" class="textbook-widget"></div>
<script type="module" src="./widgets/spec_gaming/widget.js"></script>

A CoastRunners-flavoured 1-D track. The race policy collects finish
reward $F$ once per loop; the spin policy stands on the bonus tile
collecting $B$ every tick. The sweep panel shows the long-run mean
reward of each policy versus $B$; above the crossover marker, spin
wins and the agent games the spec. Toggling PBRS shaping on adds a
"distance-to-finish" potential that penalises stationarity and
restores the race-policy margin.

---

## 16.8 Project tie-in

### Where this chapter's ideas live

- **The reward formula**: [`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs).
- **Drive taxonomy and convex cost**: [`metabolism/src/lib.rs`](https://github.com/falahat/simulator/blob/main/crates/substrate/metabolism/src/lib.rs)
  (DriveKind), [`affect/`](https://github.com/falahat/simulator/blob/main/crates/cognition/affect) (drive
  coupling).
- **`PrimaryReward` computation**: in
  [`crates/cognition/affect/`](https://github.com/falahat/simulator/blob/main/crates/cognition/affect) — wraps
  the reward formula in the per-tick agent loop.
- **Panksepp 8 + Plutchik primes design**: [`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md) §Affect.

### Where ideas are *not* yet used

- **Tchebysheff scalarization**: not implemented. Linear-only.
- **Wanting/liking dissociation**: not architecturally encoded. Both
  $Q$ and $r$ live in the same value-based scheme.
- **Inverse RL or RLHF**: not present. Reward is hand-designed.
- **Soft-max policy** (Boltzmann over Q): not implemented. The agent
  uses argmax with $\epsilon$-greedy.

### What test exercises this chapter

[`hungry_consume.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/app/tests/tasks/hungry_consume.rs)
is named for this chapter's framework. Two arms:

- **Hungry**: high initial hunger. Agent must learn to seek food. Pass
  criterion: `food_eaten_count >= 1` over 20K ticks; metric
  `q_consume_margin > -0.3` (relaxed; was originally > 0.1 before the
  Q-bias pathology was understood).
- **Sated**: low initial hunger. Pass criterion: `q_consume_margin ∈ [-0.3, 0.1]`.
  **Currently fails** with margin ≈ $-4.25$ — the Q-bias bootstrap
  drives Wait's $Q$ to $\sim 10$ while Consume is never offered.

The sated arm is the canonical homeostatic-RL failure mode in the
Simulator: the *cognitive* drive (Seeking) was satisfied, so the
agent never *wanted* food strongly enough to try Consume; the
$w_\text{alive}$ then bootstrapped Wait's $Q$ above any future
opportunity to discover Consume.

### What would need to change for modern variants

- **Fix Q-bias** via Fix 1 ($w_\text{alive} = 0$) or Fix 3
  (advantage). Both align with this chapter's homeostatic-RL framing —
  the alive baseline shouldn't dominate consummatory discrimination.
- **Add liking spikes** as transient positive reward at moments of
  consummation, separately from drive-relief math. Berridge-Robinson
  style.
- **Multi-objective explicit**: store reward as a vector per drive;
  scalarize only when needed. Useful for MORL transfer.

---

## 16.9 Exercises

1. **(Homeostatic cost shape.)** Plot $c_d(d) = d^p$ for
   $p \in \{1, 1.5, 2, 3\}$. At what $d$ does the marginal cost
   $c'_d(d) = p d^{p-1}$ exceed 1 unit per unit drive change?

2. **(Convex hull of Pareto front.)** Construct a 2-objective MDP with
   3 deterministic policies achieving returns $(0, 1)$, $(1, 0)$, and
   $(0.4, 0.4)$. Show that no linear scalarization can prefer the
   third policy.

3. **(Tchebysheff scalarization.)** With $z^{\star} = (1, 1)$ and
   $w = (1, 1)$, compute the Tchebysheff scalarization
   $-\max_d |r_d - z_d^{\star}|$ for each policy in Exercise 2. Which
   policy does Tchebysheff prefer?

4. **(PBRS preserves optimum.)** Prove the PBRS theorem: for any
   $\Phi$, the shaped reward $\tilde R = R + \gamma \Phi(s') - \Phi(s)$
   yields the same optimal policy. (Hint: write out the Bellman
   equation for $V_R$ and $V_{\tilde R}$; identify the relation.)

5. **(Level vs. delta equivalence.)** Show explicitly that the
   level-form and delta-form rewards differ by a PBRS shaping with
   $\Phi(s) = -\text{discomfort}(s) / (1 - \gamma)$.

6. **(Numerical Q-bias.)** Compute the Q-bias bootstrap fixed point
   $Q^{\star}$ given mean reward $\bar r = 0.81$ and $\gamma = 0.9$.
   Compute the noise floor $\sqrt{\alpha \sigma^2_r}$ at $\alpha = 0.1$
   and $\sigma_r = 0.5$. Compare to the inter-action gap (the $0.5$
   from §16.6). Conclude.

7. **(Wanting-liking experiment.)** Sketch how you would measure
   "wanting" (TD-propagated Q) vs. "liking" (immediate consummatory
   reward) in the Simulator. Which fields in
   [`reward.rs`](https://github.com/falahat/simulator/blob/main/crates/sim/sim_config/src/reward.rs)
   would you log? What would the dissociation look like?

8. **(Specification gaming sketch.)** Identify three ways the
   Simulator's reward formula could be "gamed" by an over-clever
   agent. Propose one safeguard per gaming strategy.

---

## 16.10 References cited in this chapter

Full entries in [`bibliography.md`](bibliography.md):

- [Berridge & Robinson 1998] — incentive sensitization (§16.2).
- [Berridge & Robinson 2016] — wanting vs liking review (§16.2).
- [Cabanac 1971] — alliesthesia (§16.1).
- [Christiano et al. 2017] — RLHF foundations (§16.7).
- [Keramati & Gutkin 2011] — homeostatic RL (§16.1).
- [Keramati & Gutkin 2014] — homeostatic RL update (§16.1).
- [Krakovna et al. 2020] — specification-gaming examples (§16.7).
- [Ng, Harada & Russell 1999] — potential-based reward shaping (§16.4).
- [Ng & Russell 2000] — inverse RL (§16.7).
- [Roijers et al. 2013] — MORL survey (§16.3).

## 16.11 Further reading

| Source | What to read | Why |
|---|---|---|
| [Keramati & Gutkin 2014] | All | The cleanest homeostatic-RL theory paper |
| [Roijers et al. 2013] | MORL survey | Complete coverage of scalarization choices |
| [Berridge & Robinson 2016] | Review | Modern wanting-liking neuroscience |
| [Christiano et al. 2017] | RLHF | Modern preference-based reward learning |
| [`docs/designs/04_cognition.md`](https://github.com/falahat/simulator/blob/main/docs/designs/04_cognition.md) | §Affect | Project drive taxonomy + citations |

---

**Next:** [Chapter 19 — Long-Horizon Credit Assignment](19_long_horizon_credit.md) —
when discounting kills signal entirely.

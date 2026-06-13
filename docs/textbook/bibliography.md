# Bibliography

Master reference list for the textbook. Every claim, derivation, example,
algorithm, or exercise the textbook borrows from a specific source is
keyed back here.

> **Validation status:** every entry was verified against its primary
> source (publisher page, arxiv abstract, JSTOR/Springer/JMLR DOI, or
> course-website URL) on 2026-05-20 via WebFetch + Web search. All
> URLs in this file resolve to the cited resource. Issues found during
> validation were fixed (incorrect author order for [Sutton,
> Szepesvári & Maei 2009]; missing co-author Brantley for the
> [AJKS 2024] textbook; expanded title for [Sutton 1990] Dyna). A
> small number of citations use the customary abbreviation "RL" rather
> than expanded "Reinforcement Learning" in paper titles — these are
> noted but kept in short form for readability; the underlying papers
> are verified by arxiv ID / DOI which uniquely identifies them.

## Citation conventions

In-chapter inline citations use the form **`[Author Year]`** or
**`[Author Year, Sec. X]`** linking to entries here. For example:

> Q-learning's tabular convergence is proved by [Watkins & Dayan 1992].

For multiple authors:

- One author: `[Sutton 1988]`
- Two authors: `[Watkins & Dayan 1992]`
- Three or more: `[Mnih et al. 2015]`

Specific examples and exercises reproduced from Sutton & Barto are
labeled `[S&B 2018, Example 6.6]` or `[S&B 2018, Ex. 4.7]`.

The textbook's chapter ordering, paper canon, and "consensus structure"
narrative are synthesized from publicly available syllabuses of six
graduate RL courses, with academic-research textbooks providing the
mathematical content. **Where structural choices follow a specific
course, that course is cited.** See [Section: Course syllabuses
consulted](#course-syllabuses-consulted) below.

---

## Textbooks

These are the canonical references. The textbook draws on each at the
chapters listed.

**[S&B 2018]** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement
Learning: An Introduction* (2nd ed.). MIT Press. Free PDF:
http://incompleteideas.net/book/the-book-2nd.html
- Used throughout. Specifically:
  - Ch. 1 (intro) → our Ch. 2 framing.
  - Ch. 2 (bandits) → our Ch. 2 §2.5.
  - Ch. 3 (MDPs) → our Ch. 3.
  - Ch. 4 (DP) → our Ch. 4.
  - Ch. 5 (MC) → our Ch. 5.
  - Ch. 6 (TD) → our Ch. 6.
  - Ch. 7 (n-step) → our Ch. 6 §6.7.
  - Ch. 9-10 (FA) → our Ch. 8.
  - Ch. 11 (off-policy with FA, deadly triad) → our Ch. 15.
  - Ch. 12 (eligibility traces) → our Ch. 7.
  - Ch. 13 (policy gradient) → our Ch. 10.
  - Cliff walking example (Sec. 6.5), Jack's car rental (Ex. 4.2),
    Blackjack (Sec. 5.1), random walk (Ex. 6.2 / 7.1) — all reproduced
    in our exercises with `[S&B 2018, ...]` attribution.

**[Bertsekas 2017]** Bertsekas, D. P. (2017). *Dynamic Programming and
Optimal Control, Volume I* (4th ed.). Athena Scientific. ISBN:
978-1-886529-43-4.
**[Bertsekas 2012]** Bertsekas, D. P. (2012). *Dynamic Programming and
Optimal Control, Volume II* (4th ed.). Athena Scientific. ISBN:
978-1-886529-44-1.
- Operator-theoretic treatment of dynamic programming. Used in our Ch. 1
  §1.4 (Banach fixed-point theorem framing) and Ch. 4 §4.2 (policy
  evaluation rigor). Vol II Ch. 1-2 directly underpins our Ch. 4
  convergence proofs.

**[Puterman 2005]** Puterman, M. L. (2005). *Markov Decision Processes:
Discrete Stochastic Dynamic Programming*. Wiley. (Original 1994.)
- The canonical MDP theory reference. Used in our Ch. 3 (formal
  definition + theorem statements) and Ch. 4 (LP formulation reference).
  Ch. 6 of Puterman is the definitive proof source for discounted-MDP
  contraction.

**[Szepesvári 2010]** Szepesvári, C. (2010). *Algorithms for
Reinforcement Learning*. Morgan & Claypool / free PDF:
https://sites.ualberta.ca/~szepesva/rlbook.html
- Concise theory-focused treatment. Used in our Ch. 1 §1.4 (operator
  contraction style), Ch. 3 (formal MDPs), Ch. 6 (TD convergence theory).

**[AJKS 2024]** Agarwal, A., Jiang, N., Kakade, S. M., & Sun, W. (2024).
*Reinforcement Learning: Theory and Algorithms* (draft). https://rltheorybook.github.io/
- Modern theoretical foundations. Princeton ECE 524 uses this as primary
  reading. Our Ch. 15 (pathologies) borrows framing from their Ch. 4-6
  (off-policy + FA).

**[R&N 2020]** Russell, S., & Norvig, P. (2020). *Artificial Intelligence:
A Modern Approach* (4th ed.). Pearson. http://aima.cs.berkeley.edu/
- The breadth-first AI reference. Used in our Ch. 2 §2.6 (MDP as
  decision-theoretic framing) and Ch. 3 §3.7 (POMDPs).

**[GBC 2016]** Goodfellow, I., Bengio, Y., & Courville, A. (2016).
*Deep Learning*. MIT Press. Free PDF: https://www.deeplearningbook.org/
- Used in our Ch. 1 (math review framing) and as background for Ch. 9
  (deep Q-learning).

**[L&S 2020]** Lattimore, T., & Szepesvári, C. (2020). *Bandit
Algorithms*. Cambridge University Press. Free PDF:
https://tor-lattimore.com/downloads/book/book.pdf
- Bandit theory reference. Used in our Ch. 12 (exploration) and Ch. 2
  §2.5 (bandit foundations of exploration).

**[Bellman 1957]** Bellman, R. (1957). *Dynamic Programming*. Princeton
University Press.
- Historical reference for the origin of dynamic programming. Cited in
  our Ch. 4 §4.1 framing.

**[Howard 1960]** Howard, R. A. (1960). *Dynamic Programming and Markov
Processes*. MIT Press.
- The original policy iteration paper. Cited in our Ch. 4 §4.4.

---

## Foundational papers

### Stochastic approximation and contraction

**[Robbins & Monro 1951]** Robbins, H., & Monro, S. (1951). A Stochastic
Approximation Method. *Annals of Mathematical Statistics*, 22(3),
400–407. https://doi.org/10.1214/aoms/1177729586
- Foundational stochastic approximation conditions. Used in our Ch. 1
  §1.5 and Ch. 6 §6.2 (TD convergence requirements).

**[Banach 1922]** Banach, S. (1922). Sur les opérations dans les
ensembles abstraits et leur application aux équations intégrales.
*Fundamenta Mathematicae*, 3, 133–181.
- The fixed-point theorem. Our Ch. 1 §1.4 proves it; every RL
  convergence proof in the textbook is an application.

### Temporal difference learning

**[Sutton 1988]** Sutton, R. S. (1988). Learning to Predict by the
Methods of Temporal Differences. *Machine Learning*, 3(1), 9–44.
https://doi.org/10.1007/BF00115009 · Erratum-corrected PDF:
http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
- Introduces TD(λ). Cited extensively in our Ch. 6 and Ch. 7.

**[Watkins & Dayan 1992]** Watkins, C. J. C. H., & Dayan, P. (1992).
Q-Learning. *Machine Learning*, 8, 279–292.
https://doi.org/10.1007/BF00992698 · PDF:
https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf
- The Q-learning convergence theorem. Stated and used in our Ch. 6 §6.4.

**[Tsitsiklis 1994]** Tsitsiklis, J. N. (1994). Asynchronous Stochastic
Approximation and Q-Learning. *Machine Learning*, 16(3), 185–202.
https://doi.org/10.1007/BF00993306
- Convergence of asynchronous Q-learning. Cited in our Ch. 1 §1.5 and
  Ch. 6.

**[Tsitsiklis & Van Roy 1997]** Tsitsiklis, J. N., & Van Roy, B. (1997).
An Analysis of Temporal-Difference Learning with Function Approximation.
*IEEE Transactions on Automatic Control*, 42(5), 674–690.
https://doi.org/10.1109/9.580874 · PDF:
https://www.mit.edu/~jnt/Papers/J063-97-bvr-td.pdf
- Linear TD convergence + the foundation for the deadly triad. Cited in
  our Ch. 8 and Ch. 15.

**[Watkins 1989]** Watkins, C. J. C. H. (1989). *Learning from Delayed
Rewards*. PhD thesis, King's College, Cambridge.
- Original Q-learning thesis. Historical attribution in our Ch. 6.

**[Singh & Sutton 1996]** Singh, S. P., & Sutton, R. S. (1996).
Reinforcement learning with replacing eligibility traces.
*Machine Learning*, 22, 123–158.
- Replacing traces. Cited in our Ch. 7 §7.3.

**[van Seijen et al. 2016]** van Seijen, H., Mahmood, A. R., Pilarski,
P. M., Machado, M. C., & Sutton, R. S. (2016). True Online Temporal-
Difference Learning. *JMLR*, 17.
https://arxiv.org/abs/1512.04087
- Dutch traces for true forward-backward equivalence. Cited in our
  Ch. 7 §7.4.

**[van Hasselt 2010]** van Hasselt, H. (2010). Double Q-learning.
*Advances in Neural Information Processing Systems*, 23.
https://papers.nips.cc/paper/3964-double-q-learning
- The original double-Q paper. Cited in our Ch. 6 §6.6.

### Deep Q-learning

**[Mnih et al. 2015]** Mnih, V., et al. (2015). Human-level control
through deep reinforcement learning. *Nature*, 518, 529–533.
https://doi.org/10.1038/nature14236
- The DQN paper. Cited in our Ch. 9.

**[Mnih et al. 2013]** Mnih, V., et al. (2013). Playing Atari with Deep
Reinforcement Learning. arXiv:1312.5602.
- DQN preprint. Cited in our Ch. 9.

**[Hasselt, Guez, Silver 2016]** van Hasselt, H., Guez, A., & Silver, D.
(2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
arXiv:1509.06461.
- Double DQN. Cited in our Ch. 9.

**[Wang et al. 2016]** Wang, Z., Schaul, T., Hessel, M., van Hasselt, H.,
Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for
Deep Reinforcement Learning. *ICML*. arXiv:1511.06581.
- Dueling DQN. Cited in our Ch. 9 and used in our Ch. 15 §15.3 (Fix 3 —
  advantage learning).

**[Schaul et al. 2016]** Schaul, T., Quan, J., Antonoglou, I., & Silver,
D. (2016). Prioritized Experience Replay. *ICLR*. arXiv:1511.05952.
- Cited in our Ch. 9.

**[Hessel et al. 2018]** Hessel, M., et al. (2018). Rainbow: Combining
Improvements in Deep Reinforcement Learning. *AAAI*. arXiv:1710.02298.
- Cited in our Ch. 9.

**[Bellemare et al. 2017]** Bellemare, M. G., Dabney, W., & Munos, R.
(2017). A Distributional Perspective on Reinforcement Learning. *ICML*.
arXiv:1707.06887.
- C51. Cited in our Ch. 9.

**[Dabney et al. 2018]** Dabney, W., Rowland, M., Bellemare, M. G., &
Munos, R. (2018). Distributional Reinforcement Learning with Quantile
Regression. *AAAI*. arXiv:1710.10044.
- QR-DQN. Cited in our Ch. 9.

**[van Hasselt et al. 2018]** van Hasselt, H., Doron, Y., Strub, F.,
Hessel, M., Sonnerat, N., & Modayil, J. (2018). Deep Reinforcement
Learning and the Deadly Triad. arXiv:1812.02648.
- The deadly triad empirically. Cited in our Ch. 15.

### Function approximation pathologies

**[Baird 1995]** Baird, L. (1995). Residual Algorithms: Reinforcement
Learning with Function Approximation. *ICML*.
http://www.leemon.com/papers/1995b.pdf
- Baird's counterexample. Cited in our Ch. 15 §15.1.

**[Sutton, Szepesvári & Maei 2009]** Sutton, R. S., Szepesvári, C., &
Maei, H. R. (2009). A Convergent O(n) Temporal-Difference Algorithm for
Off-Policy Learning with Linear Function Approximation. *Advances in
Neural Information Processing Systems 21* (NIPS 2008 conference,
proceedings published 2009), pp. 1609–1616.
- GTD methods. Cited in our Ch. 15 §15.9.

### Function approximation, basis construction, tile coding

**[Albus 1975]** Albus, J. S. (1975). A New Approach to Manipulator
Control: The Cerebellar Model Articulation Controller (CMAC). *Journal
of Dynamic Systems, Measurement, and Control*, 97(3), 220–227.
https://doi.org/10.1115/1.3426922 · NIST:
https://www.nist.gov/publications/new-approach-manipulator-control-cerebellar-model-articulation-controller-cmac1
- The CMAC origin paper. Foundation of tile coding. Cited in our Ch. 8.

**[Sutton 1996]** Sutton, R. S. (1996). Generalization in Reinforcement
Learning: Successful Examples Using Sparse Coarse Coding. *NIPS*.
http://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding
- The empirical paper that made tile coding canonical. Cited in our Ch. 8.

**[Sherstov & Stone 2005]** Sherstov, A. A., & Stone, P. (2005). Function
Approximation via Tile Coding: Automating Parameter Choice. *SARA*.
http://web.cs.ucla.edu/~sherstov/pdf/sara05-tiling.pdf
- Cited in our Ch. 8.

**[Whiteson, Taylor, Stone 2007]** Whiteson, S., Taylor, M. E., & Stone,
P. (2007). *Adaptive Tile Coding for Value Function Approximation*. UT
Tech Report. https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/whitesontr07.pdf
- Cited in our Ch. 8.

**[Konidaris, Osentoski, Thomas 2011]** Konidaris, G., Osentoski, S., &
Thomas, P. (2011). Value Function Approximation in Reinforcement
Learning Using the Fourier Basis. *AAAI*.
https://ojs.aaai.org/index.php/AAAI/article/view/7903
- Cited in our Ch. 8.

**[Mahadevan & Maggioni 2007]** Mahadevan, S., & Maggioni, M. (2007).
Proto-value Functions: A Laplacian Framework for Learning Representation
and Control in MDPs. *JMLR*, 8, 2169–2231.
- Cited in our Ch. 8.

**[Parr et al. 2007]** Parr, R., Painter-Wakefield, C., Li, L., &
Littman, M. L. (2007). Analyzing Feature Generation for Value-Function
Approximation. *ICML*. https://users.cs.duke.edu/~parr/icml07.pdf
- BEBFs. Cited in our Ch. 8.

**[Zaheer et al. 2017]** Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos,
B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. *NIPS*.
arXiv:1703.06114.
- Cited in our Ch. 8 (permutation-invariant pooling).

**[Lee et al. 2019]** Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi,
S., & Teh, Y. W. (2019). Set Transformer. *ICML*. arXiv:1810.00825.
- Cited in our Ch. 8.

### Representation learning in RL

**[Jaderberg et al. 2017]** Jaderberg, M., et al. (2017). Reinforcement
Learning with Unsupervised Auxiliary Tasks. *ICLR*. arXiv:1611.05397.
- UNREAL. Cited in our Ch. 8 §8.4.

**[Srinivas, Laskin, Abbeel 2020]** Srinivas, A., Laskin, M., & Abbeel,
P. (2020). CURL: Contrastive Unsupervised Representations for
Reinforcement Learning. *ICML*. arXiv:2004.04136.
- Cited in our Ch. 8.

**[Stooke et al. 2021]** Stooke, A., Lee, K., Abbeel, P., & Laskin, M.
(2021). Decoupling Representation Learning from Reinforcement Learning.
*ICML*. arXiv:2009.08319.
- ATC. Cited in our Ch. 8.

**[Zhang et al. 2021]** Zhang, A., McAllister, R., Calandra, R., Gal, Y.,
& Levine, S. (2021). Learning Invariant Representations for
Reinforcement Learning without Reconstruction. *ICLR*. arXiv:2006.10742.
- DBC. Cited in our Ch. 8.

**[Li, Walsh, Littman 2006]** Li, L., Walsh, T. J., & Littman, M. L.
(2006). Towards a Unified Theory of State Abstraction for MDPs. *ISAIM*.
http://rbr.cs.umass.edu/aimath06/proceedings/P21.pdf
- State abstraction taxonomy. Cited in our Ch. 8 §8.9.

**[Ferns, Panangaden, Precup 2004]** Ferns, N., Panangaden, P., & Precup,
D. (2004). Metrics for Finite MDPs. *UAI*. arXiv:1207.4114.
- Bisimulation metrics. Cited in our Ch. 8.

### Policy gradient and actor-critic

**[Williams 1992]** Williams, R. J. (1992). Simple Statistical
Gradient-Following Algorithms for Connectionist Reinforcement Learning.
*Machine Learning*, 8, 229–256. https://doi.org/10.1007/BF00992696
- REINFORCE. Cited in our Ch. 10.

**[Sutton et al. 2000]** Sutton, R. S., McAllester, D., Singh, S., &
Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning
with Function Approximation. *NIPS*.
- The policy gradient theorem. Cited in our Ch. 10.

**[Konda & Tsitsiklis 1999]** Konda, V. R., & Tsitsiklis, J. N. (1999).
Actor-Critic Algorithms. *NIPS*.
- Foundational actor-critic. Cited in our Ch. 11.

**[Mnih et al. 2016]** Mnih, V., et al. (2016). Asynchronous Methods for
Deep Reinforcement Learning. *ICML*. arXiv:1602.01783.
- A3C. Cited in our Ch. 11.

**[Kakade 2001]** Kakade, S. (2001). A Natural Policy Gradient. *NIPS*.
- Natural gradient. Cited in our Ch. 11.

**[Schulman et al. 2015a]** Schulman, J., Levine, S., Moritz, P.,
Jordan, M., & Abbeel, P. (2015). Trust Region Policy Optimization.
*ICML*. arXiv:1502.05477.
- TRPO. Cited in our Ch. 11.

**[Schulman et al. 2015b]** Schulman, J., Moritz, P., Levine, S.,
Jordan, M. I., & Abbeel, P. (2015). High-Dimensional Continuous Control
Using Generalized Advantage Estimation. arXiv:1506.02438.
- GAE. Cited in our Ch. 10.

**[Schulman et al. 2017]** Schulman, J., Wolski, F., Dhariwal, P.,
Radford, A., & Klimov, O. (2017). Proximal Policy Optimization
Algorithms. arXiv:1707.06347.
- PPO. Cited in our Ch. 11.

**[Lillicrap et al. 2015]** Lillicrap, T. P., et al. (2015). Continuous
Control with Deep Reinforcement Learning. arXiv:1509.02971.
- DDPG. Cited in our Ch. 11.

**[Fujimoto, van Hoof, Meger 2018]** Fujimoto, S., van Hoof, H., &
Meger, D. (2018). Addressing Function Approximation Error in
Actor-Critic Methods. *ICML*. arXiv:1802.09477.
- TD3. Cited in our Ch. 11.

**[Haarnoja et al. 2018]** Haarnoja, T., Zhou, A., Abbeel, P., & Levine,
S. (2018). Soft Actor-Critic. *ICML*. arXiv:1801.01290.
- SAC. Cited in our Ch. 11.

**[Silver et al. 2014]** Silver, D., Lever, G., Heess, N., Degris, T.,
Wierstra, D., & Riedmiller, M. (2014). Deterministic Policy Gradient
Algorithms. *ICML 2014*. http://proceedings.mlr.press/v32/silver14.html
- DPG theorem. Cited in our Ch. 11.

### Model-based RL

**[Sutton 1990]** Sutton, R. S. (1990). Integrated Architectures for
Learning, Planning, and Reacting Based on Approximating Dynamic
Programming. *ICML*. https://dl.acm.org/doi/10.1145/122344.122377
- Dyna. Cited in our Ch. 13.

**[Browne et al. 2012]** Browne, C. B., et al. (2012). A Survey of Monte
Carlo Tree Search Methods. *IEEE TCIAIG*.
- MCTS survey. Cited in our Ch. 13.

**[Silver et al. 2016]** Silver, D., et al. (2016). Mastering the game
of Go with deep neural networks and tree search. *Nature*, 529, 484–489.
- AlphaGo. Cited in our Ch. 13.

**[Silver et al. 2017]** Silver, D., et al. (2017). Mastering Chess and
Shogi by Self-Play with a General Reinforcement Learning Algorithm.
arXiv:1712.01815.
- AlphaZero. Cited in our Ch. 13.

**[Schrittwieser et al. 2020]** Schrittwieser, J., et al. (2020).
Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.
*Nature*, 588, 604–609. arXiv:1911.08265.
- MuZero. Cited in our Ch. 13.

**[Ha & Schmidhuber 2018]** Ha, D., & Schmidhuber, J. (2018). World
Models. arXiv:1803.10122.
- Cited in our Ch. 13.

**[Hafner et al. 2020]** Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M.
(2020). Dream to Control: Learning Behaviors by Latent Imagination.
*ICLR*. arXiv:1912.01603.
- Dreamer. Cited in our Ch. 13.

**[Hafner et al. 2023]** Hafner, D., Pasukonis, J., Ba, J., & Lillicrap,
T. (2023). Mastering Diverse Domains through World Models.
arXiv:2301.04104. (Subsequently published in *Nature* 640, 647–653,
2025.)
- DreamerV3. Cited in our Ch. 13.

### Exploration

**[Auer, Cesa-Bianchi, Fischer 2002]** Auer, P., Cesa-Bianchi, N., &
Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit
Problem. *Machine Learning*, 47, 235–256.
https://doi.org/10.1023/A:1013689704352
- UCB1. Cited in our Ch. 12 and Ch. 2 §2.5.

**[Thompson 1933]** Thompson, W. R. (1933). On the Likelihood that One
Unknown Probability Exceeds Another in View of the Evidence of Two
Samples. *Biometrika*, 25(3-4), 285–294.
- Original posterior-sampling heuristic. Cited in our Ch. 12.

**[Russo et al. 2018]** Russo, D., Van Roy, B., Kazerouni, A., Osband,
I., & Wen, Z. (2018). A Tutorial on Thompson Sampling. arXiv:1707.02038.
- Cited in our Ch. 12.

**[Jaksch, Ortner, Auer 2010]** Jaksch, T., Ortner, R., & Auer, P.
(2010). Near-optimal Regret Bounds for Reinforcement Learning. *JMLR*,
11, 1563–1600.
- UCRL2. Cited in our Ch. 12.

**[Kakade 2003]** Kakade, S. M. (2003). *On the Sample Complexity of
Reinforcement Learning*. PhD thesis, University College London.
- PAC RL framework. Cited in our Ch. 12.

**[Bellemare et al. 2016]** Bellemare, M. G., Srinivasan, S., Ostrovski,
G., Schaul, T., Saxton, D., & Munos, R. (2016). Unifying Count-Based
Exploration and Intrinsic Motivation. *NIPS*. arXiv:1606.01868.
- Pseudo-counts. Cited in our Ch. 12.

**[Pathak et al. 2017]** Pathak, D., Agrawal, P., Efros, A. A., &
Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised
Prediction. *ICML*. arXiv:1705.05363.
- ICM. Cited in our Ch. 12.

**[Burda et al. 2019]** Burda, Y., Edwards, H., Storkey, A., & Klimov,
O. (2019). Exploration by Random Network Distillation. *ICLR*.
arXiv:1810.12894.
- RND. Cited in our Ch. 12.

**[Osband et al. 2016]** Osband, I., Blundell, C., Pritzel, A., & Van
Roy, B. (2016). Deep Exploration via Bootstrapped DQN.
arXiv:1602.04621.
- Cited in our Ch. 12.

**[Brafman & Tennenholtz 2002]** Brafman, R. I., & Tennenholtz, M.
(2002). R-MAX – A General Polynomial Time Algorithm for Near-Optimal
Reinforcement Learning. *JMLR*, 3, 213–231.
- R-MAX. Cited in our Ch. 2.

### Offline RL, IRL, imitation, RLHF

**[Ross, Gordon, Bagnell 2011]** Ross, S., Gordon, G. J., & Bagnell, J.
A. (2011). A Reduction of Imitation Learning and Structured Prediction
to No-Regret Online Learning (DAgger). *AISTATS*. arXiv:1011.0686.
- Cited in our Ch. 17 reading.

**[Fujimoto, Meger, Precup 2019]** Fujimoto, S., Meger, D., & Precup,
D. (2019). Off-Policy Deep Reinforcement Learning without Exploration.
*ICML*. arXiv:1812.02900.
- BCQ. Cited in our Ch. 17 reading.

**[Kumar et al. 2020]** Kumar, A., Zhou, A., Tucker, G., & Levine, S.
(2020). Conservative Q-Learning for Offline Reinforcement Learning.
*NeurIPS*. arXiv:2006.04779.
- CQL. Cited in our Ch. 17.

**[Chen et al. 2021]** Chen, L., et al. (2021). Decision Transformer:
Reinforcement Learning via Sequence Modeling. *NeurIPS*. arXiv:2106.01345.
- Cited in our Ch. 17.

**[Christiano et al. 2017]** Christiano, P., et al. (2017). Deep
reinforcement learning from human preferences. *NIPS*. arXiv:1706.03741.
- RLHF foundation. Cited in our Ch. 17 reading.

**[Ziebart et al. 2008]** Ziebart, B. D., Maas, A., Bagnell, J. A., &
Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning.
*AAAI*.
- MaxEnt IRL. Cited in our Ch. 17 reading.

**[Levine 2018]** Levine, S. (2018). Reinforcement Learning and Control
as Probabilistic Inference: Tutorial and Review. arXiv:1805.00909.
- Cited in our Ch. 17.

### Hierarchical RL and options

**[Sutton, Precup, Singh 1999]** Sutton, R. S., Precup, D., & Singh, S.
(1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction
in Reinforcement Learning. *Artificial Intelligence*, 112, 181–211.
https://doi.org/10.1016/S0004-3702(99)00052-1
- Options framework. Cited in our Ch. 14.

**[Bradtke & Duff 1995]** Bradtke, S. J., & Duff, M. O. (1995).
Reinforcement Learning Methods for Continuous-Time Markov Decision
Problems. *NIPS*.
- SMDP-Q. Cited in our Ch. 14.

**[Parr & Russell 1998]** Parr, R., & Russell, S. (1998). Reinforcement
Learning with Hierarchies of Machines. *NIPS*.
- HAM. Cited in our Ch. 14 reading.

**[Bacon, Harb, Precup 2017]** Bacon, P.-L., Harb, J., & Precup, D.
(2017). The Option-Critic Architecture. *AAAI*. arXiv:1609.05140.
- Cited in our Ch. 14.

**[Vezhnevets et al. 2017]** Vezhnevets, A. S., et al. (2017). FeUdal
Networks for Hierarchical Reinforcement Learning. *ICML*. arXiv:1703.01161.
- Cited in our Ch. 14.

**[Nachum et al. 2018]** Nachum, O., Gu, S., Lee, H., & Levine, S.
(2018). Data-Efficient Hierarchical Reinforcement Learning (HIRO).
*NeurIPS*. arXiv:1805.08296.
- Cited in our Ch. 14.

**[Levy et al. 2019]** Levy, A., Konidaris, G., Platt, R., & Saenko, K.
(2019). Learning Multi-Level Hierarchies with Hindsight. *ICLR*.
arXiv:1712.00948.
- HAC. Cited in our Ch. 14.

**[Eysenbach et al. 2018]** Eysenbach, B., Gupta, A., Ibarz, J., &
Levine, S. (2018). Diversity Is All You Need: Learning Skills without a
Reward Function. arXiv:1802.06070.
- DIAYN. Cited in our Ch. 14.

### Multi-agent

**[Lowe et al. 2017]** Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P.,
& Mordatch, I. (2017). Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments. *NIPS*. arXiv:1706.02275.
- MADDPG. Cited in our extensions.

**[Rashid et al. 2018]** Rashid, T., Samvelyan, M., Schroeder de Witt,
C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic
Value Function Factorisation for Deep Multi-Agent Reinforcement
Learning. *ICML*. arXiv:1803.11485.
- Cited in our extensions.

**[Foerster et al. 2018]** Foerster, J., Farquhar, G., Afouras, T.,
Nardelli, N., & Whiteson, S. (2018). Counterfactual Multi-Agent Policy
Gradients (COMA). *AAAI*. arXiv:1705.08926.
- Cited in our Ch. 17.

### Long-horizon credit assignment

**[Ng, Harada, Russell 1999]** Ng, A. Y., Harada, D., & Russell, S.
(1999). Policy Invariance under Reward Transformations. *ICML*.
- Potential-based reward shaping. Cited in our Ch. 17.

**[Andrychowicz et al. 2017]** Andrychowicz, M., et al. (2017).
Hindsight Experience Replay. *NeurIPS*. arXiv:1707.01495.
- HER. Cited in our Ch. 17.

**[Munos et al. 2016]** Munos, R., Stepleton, T., Harutyunyan, A., &
Bellemare, M. G. (2016). Safe and Efficient Off-Policy Reinforcement
Learning. *NeurIPS*. arXiv:1606.02647.
- Retrace(λ). Cited in our Ch. 17.

**[Dayan 1993]** Dayan, P. (1993). Improving Generalization for
Temporal Difference Learning: The Successor Representation.
*Neural Computation*, 5(4), 613–624.
- Successor representation. Cited in our Ch. 17.

**[Barreto et al. 2017]** Barreto, A., Dabney, W., Munos, R., Hunt, J.
J., Schaul, T., van Hasselt, H., & Silver, D. (2017). Successor Features
for Transfer in Reinforcement Learning. *NeurIPS*. arXiv:1606.05312.
- Cited in our Ch. 17.

**[Arjona-Medina et al. 2019]** Arjona-Medina, J. A., Gillhofer, M.,
Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019).
RUDDER: Return Decomposition for Delayed Rewards. *NeurIPS*.
arXiv:1806.07857.
- Cited in our Ch. 17.

**[Harutyunyan et al. 2019]** Harutyunyan, A., et al. (2019). Hindsight
Credit Assignment. *NeurIPS*. arXiv:1912.02503.
- Cited in our Ch. 17.

### Action spaces

**[Masson, Ranchod, Konidaris 2016]** Masson, W., Ranchod, P., &
Konidaris, G. (2016). Reinforcement Learning with Parameterized Actions.
*AAAI*. arXiv:1509.01644.
- Cited in our Ch. 18.

**[Hausknecht & Stone 2016]** Hausknecht, M., & Stone, P. (2016). Deep
Reinforcement Learning in Parameterized Action Space. *ICLR*.
arXiv:1511.04143.
- PA-DDPG. Cited in our Ch. 18.

**[Xiong et al. 2018]** Xiong, J., et al. (2018). Parametrized Deep
Q-Networks Learning: Reinforcement Learning with Discrete-Continuous
Hybrid Action Space. arXiv:1810.06394.
- P-DQN. Cited in our Ch. 18.

**[Bester, James, Konidaris 2019]** Bester, C. J., James, S. D., &
Konidaris, G. D. (2019). Multi-Pass Q-Networks for Deep Reinforcement
Learning with Parameterised Action Spaces. arXiv:1905.04388.
- MP-DQN. Cited in our Ch. 18.

**[Fan et al. 2019]** Fan, Z., Su, R., Zhang, W., & Yu, Y. (2019).
Hybrid Actor-Critic Reinforcement Learning in Parameterized Action
Space. arXiv:1903.01344.
- H-PPO. Cited in our Ch. 18.

**[Dulac-Arnold et al. 2015]** Dulac-Arnold, G., et al. (2015). Deep
Reinforcement Learning in Large Discrete Action Spaces. arXiv:1512.07679.
- Wolpertinger. Cited in our Ch. 18.

**[Tennenholtz & Mannor 2019]** Tennenholtz, G., & Mannor, S. (2019).
The Natural Language of Actions. *ICML*. arXiv:1902.01119.
- Act2Vec. Cited in our Ch. 18.

**[Huang & Ontañón 2020]** Huang, S., & Ontañón, S. (2020). A Closer
Look at Invalid Action Masking in Policy Gradient Algorithms.
arXiv:2006.14171.
- Action masking. Cited in our Ch. 18.

### Memory in deep RL

**[Graves et al. 2016]** Graves, A., et al. (2016). Hybrid computing
using a neural network with dynamic external memory. *Nature*, 538,
471–476.
- DNC. Cited in our Ch. 8.

**[Blundell et al. 2016]** Blundell, C., et al. (2016). Model-Free
Episodic Control. arXiv:1606.04460.
- MFEC. Cited in our Ch. 8.

**[Pritzel et al. 2017]** Pritzel, A., et al. (2017). Neural Episodic
Control. *ICML*. arXiv:1703.01988.
- NEC. Cited in our Ch. 8.

**[Wayne et al. 2018]** Wayne, G., et al. (2018). Unsupervised Predictive
Memory in a Goal-Directed Agent. arXiv:1803.10760.
- MERLIN. Cited in our Ch. 8.

### State abstraction

**[Ravindran & Barto 2003]** Ravindran, B., & Barto, A. G. (2003). SMDP
Homomorphisms: An Algebraic Approach to Abstraction in Semi-Markov
Decision Processes. *IJCAI*.
- Cited in our extensions.

**[Ormoneit & Sen 2002]** Ormoneit, D., & Sen, Ś. (2002). Kernel-Based
Reinforcement Learning. *Machine Learning*, 49, 161–178.
- KBRL. Cited in our extensions.

**[Ernst, Geurts, Wehenkel 2005]** Ernst, D., Geurts, P., & Wehenkel, L.
(2005). Tree-Based Batch Mode Reinforcement Learning. *JMLR*, 6, 503–556.
- FQI. Cited in our extensions.

**[Kolter & Ng 2009]** Kolter, J. Z., & Ng, A. Y. (2009). Regularization
and Feature Selection in Least-Squares Temporal Difference Learning.
*ICML*.
- LARS-TD. Cited in our extensions.

### POMDPs

**[Kaelbling, Littman, Cassandra 1998]** Kaelbling, L. P., Littman, M.
L., & Cassandra, A. R. (1998). Planning and Acting in Partially
Observable Stochastic Domains. *Artificial Intelligence*, 101, 99–134.
- Cited in our Ch. 3 §3.7.

### Adaptive step sizes (project-specific area)

**[Sutton 1992]** Sutton, R. S. (1992). Adapting Bias by Gradient
Descent: An Incremental Version of Delta-Bar-Delta. *AAAI*.
- IDBD. Cited in our extensions.

**[Mahmood et al. 2012]** Mahmood, A. R., Sutton, R. S., Degris, T., &
Pilarski, P. M. (2012). Tuning-free step-size adaptation. *ICASSP*.
- Autostep. Cited in our extensions.

**[Kearney et al. 2018]** Kearney, A., Veeriah, V., Travnik, J. B.,
Sutton, R. S., & Pilarski, P. M. (2018). TIDBD: Adapting Temporal-
difference Step-sizes Through Stochastic Meta-descent. arXiv:1804.03334.
- TIDBD. Cited in our extensions.

### Bandit-and-RL convergence and PAC

**[Strehl, Li, Littman 2009]** Strehl, A. L., Li, L., & Littman, M. L.
(2009). Reinforcement Learning in Finite MDPs: PAC Analysis. *JMLR*, 10,
2413–2444.
- Cited in our extensions.

### Reproducibility

**[Henderson et al. 2018]** Henderson, P., Islam, R., Bachman, P.,
Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning
that Matters. *AAAI*. arXiv:1709.06560.
- Cited in our extensions.

**[Engstrom et al. 2020]** Engstrom, L., Ilyas, A., Santurkar, S.,
Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation
Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. *ICLR*.
arXiv:2005.12729.
- Cited in our extensions.

### Homeostatic RL (project-specific)

**[Keramati & Gutkin 2011]** Keramati, M., & Gutkin, B. S. (2011). A
Reinforcement Learning Theory for Homeostatic Regulation. *NeurIPS*.
- Foundational homeostatic RL. Cited in our Ch. 16.

**[Keramati & Gutkin 2014]** Keramati, M., & Gutkin, B. (2014).
Homeostatic Reinforcement Learning for Integrating Reward Collection
and Physiological Stability. *eLife*, 3, e04811.
- Updated framework. Cited in our Ch. 16.

**[Berridge & Robinson 1998]** Berridge, K. C., & Robinson, T. E. (1998).
What is the role of dopamine in reward: hedonic impact, reward learning,
or incentive salience? *Brain Research Reviews*, 28(3), 309–369.
- Wanting vs liking. Cited in our Ch. 16.

**[Berridge & Robinson 2016]** Berridge, K. C., & Robinson, T. E. (2016).
Liking, wanting, and the incentive-sensitization theory of addiction.
*American Psychologist*, 71(8), 670–679.
- Modern statement. Cited in our Ch. 16.

**[Roijers et al. 2013]** Roijers, D. M., Vamplew, P., Whiteson, S., &
Dazeley, R. (2013). A Survey of Multi-Objective Sequential Decision-
Making. *JAIR*, 48, 67–113.
- MORL survey. Cited in our Ch. 16.

---

## Course syllabuses consulted

These graduate-level RL courses informed the structure, topic ordering,
and consensus narrative. The textbook does not lift slide content
verbatim from any of them, but the "spine" of topic ordering (MDPs → DP
→ MC/TD/Q-learning → FA → DQN → policy gradient → actor-critic →
model-based → exploration → offline RL → frontier) reflects the
consensus across the six:

1. **Stanford CS234 — Reinforcement Learning** (Emma Brunskill, current
   instructor). https://web.stanford.edu/class/cs234/
2. **UC Berkeley CS285 — Deep Reinforcement Learning** (Sergey Levine).
   https://rail.eecs.berkeley.edu/deeprlcourse/
3. **CMU 10-703 — Deep Reinforcement Learning & Control**.
   https://www.andrew.cmu.edu/course/10-703/
4. **MIT 6.7920 — Reinforcement Learning: Foundations and Methods**.
   https://web.mit.edu/6.7920/www/
5. **Princeton ECE 524 — Foundations of Reinforcement Learning** (Chi Jin).
   https://sites.google.com/view/cjin/teaching/ece524
6. **Georgia Tech CS 7642 — Reinforcement Learning** (OMSCS).
   https://omscs.gatech.edu/cs-7642-reinforcement-learning

## Video lecture series consulted

- **David Silver — UCL RL Lectures** (10 lectures, 2015). YouTube
  playlist: https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
- **Sergey Levine — CS285 Lectures** (2023 iteration on YouTube).
  https://www.youtube.com/playlist?list=PL_iWQOsE6TfX7MaC6C3HcdOf1g337dlC9
- **Emma Brunskill — CS234 Lectures**.
  https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u

## Other consulted sources

- The Simulator's own design corpus (`docs/` tree) for project-specific
  framing in tie-in sections. Internal documents are linked inline as
  `[`docs/...`](...)`.
- arxiv.org for paper PDFs.
- The free Sutton & Barto PDF at incompleteideas.net for verbatim
  algorithm pseudocode and example structure.

---

## Attribution notes

The textbook **reproduces** the following from primary sources:

| Item | Source | Where used |
|---|---|---|
| Banach fixed-point theorem statement and proof structure | Standard analysis result; this proof follows [Szepesvári 2010, Appendix] | Ch. 1 §1.4 |
| Hoeffding's inequality | Standard result | Ch. 1 §1.1 |
| Bellman expectation/optimality equations (formal statements) | [Puterman 2005, Ch. 6], [S&B 2018, Ch. 3] | Ch. 3 |
| Contraction proof for $T^{\star}$ | [S&B 2018, Ch. 4.3] | Ch. 3 §3.5 |
| The max-trick lemma | Standard; proof follows [Szepesvári 2010] | Ch. 3 §3.5 |
| Policy improvement theorem | [S&B 2018, Sec. 4.2] | Ch. 4 §4.3 |
| Cliff walking example (4×12 grid) | [S&B 2018, Sec. 6.5] | Ch. 6 §6.9 |
| Jack's car rental example | [S&B 2018, Example 4.2] | Ch. 4 §4.9 |
| Blackjack example | [S&B 2018, Sec. 5.1] | Ch. 5 §5.8 |
| Five-state random walk | [S&B 2018, Example 7.1] | Ch. 7 §7.8 |
| Q-learning convergence theorem (statement) | [Watkins & Dayan 1992] | Ch. 6 §6.4 |
| TD convergence (statement) | [Tsitsiklis 1994] | Ch. 6 §6.2 |
| Linear TD convergence + deadly triad theory | [Tsitsiklis & Van Roy 1997] | Ch. 8, Ch. 15 |
| Baird's counterexample | [Baird 1995] | Ch. 15 §15.1 |
| GPI framing | [S&B 2018, Sec. 4.6] | Ch. 4 §4.6 |
| Effective horizon table | Standard derivation from $1/(1-\gamma)$ | Ch. 2 §2.3 |

Exercises **inspired by** but not lifted from [S&B 2018]: Ch. 1 Ex 6-7,
Ch. 3 Ex 5, Ch. 4 Ex 4-5.

Project-specific examples (the hunger MDP, the Q-bias bootstrap debug
story, the L-suite failure analysis) are this textbook's original
contributions, derived from the Simulator's actual code; the Q-bias
bootstrap pathology is documented in depth in Chapter 17.

---

**Back to:** [Table of Contents](00_index.md)

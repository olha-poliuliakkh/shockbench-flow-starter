# Beyond the rolling LP: which paradigm shift can pay on ShockBench-Flow

An assessment of eight families of methods: learning to optimize, approximate dynamic programming, decomposition,
SDDP, robust optimization, model-free and offline RL, and simulator lookahead. Each is judged against what this
repository has measured, not against its general reputation. It ends with one flagship prototype and its
engineering pipeline.

Section numbers like §5.15 refer to [dataset_and_rss_mechanics.md](dataset_and_rss_mechanics.md).

---

## 0. Bottom line

1. **The ceiling is not the 0.834.** That number is our rolling planner given every future disruption (§5.9). The
   ceilings that bind are elsewhere:
   - The MILP dual bound caps **any** policy at **0.973** on Small dev (§3.3).
   - The public board's top score is 0.882.
   - On Full (the prize board) our best agent stands at **0.670** (§5.15), so Full has three to four times Small's headroom.
2. **The largest gain this project has measured came from valuing the future, not from a better solver:**
   +0.253 on Full from a constant terminal credit. A planner that is short-sighted on a 104-week episode is the
   dominant defect, and it is a value-function problem.
3. **The flagship is therefore approximate dynamic programming inside the LP.** A learned, state-conditioned,
   piecewise-linear concave value of what the window leaves behind (Powell's lookahead-plus-VFA hybrid; PARL; hindsight
   learning). The network sets only coefficients of the LP, so the LP stays an LP and inference costs milliseconds.
4. **Its expected gain is modest and must be gated.** The credit's fraction sweep was flat (0.3 to 1.0 within 0.5 %
   on 7 of 8 episodes), which says the LP already got most of the value from "leftovers are worth something". A
   one-hour headroom probe (§5.6 below) decides go or no-go before any training.
5. **Five of the eight families are ruled out by measurement here:**
   - learning to optimize, which buys speed the agent does not need;
   - decomposition (ADMM, Lagrangian), the same;
   - robust LP, because over-hedging loses in level 1, which carries 44 % of the weight;
   - end-to-end RL, which cannot beat the LP teacher in the time left;
   - simulator lookahead, which measured −0.097.

   SDDP is sound but does not fit the hackathon's timeline.

---

## 1. Where the score is lost

| Quantity | Small dev | Full (8 private) | Source |
| --- | ---: | ---: | --- |
| Our best agent | 0.783 (`milp48t`) | **0.670** (`twopass48_credit`) | §5.12, §5.15 |
| Her compact LP (`mpc_nobuf`) | 0.821 | not measured | §5.16 |
| Package planner, 48-week window, persistence, no credit | 0.738 | 0.432 | `oracle_h.json`, `oracle_full_h.json` |
| Same, true disruptions in the window | 0.805 | 0.549 | same |
| Rolling planner, true disruptions, two passes | 0.834 | not measured | §5.9 |
| Clairvoyant base-load-first plan, replayed open loop | 0.852 | not measured | §1 |
| Same plan, in the LP's own model | 0.958 | not measured | §5.10 |
| Any policy (MILP dual bound) | 0.973 | not measured | §3.3 |
| Board, top three (Small) | 0.882, 0.871, 0.867 | not public | §3.5 |

The gaps between rows have different natures:

- **Formulation fidelity** (0.783 → 0.821 on Small). Her LP models the grid and fuel more faithfully, and that
  is worth +0.04 (§5.16). The binaries are not the source: relaxed, they score the same.
- **Information** (+0.067 on Small at 48 weeks, +0.117 on Full without credit). This is the cap on anything that
  forecasts or hedges disruptions better. Every attempt to collect it has measured about 0:
  - a calibrated belief: −0.004;
  - generator scenarios: +0.002 on top of two passes;
  - risk-averse settings: −0.012.
- **Myopia** (Full only). A 48-week window on a 104-week episode runs stock down toward a cliff that does not exist.
  A constant credit took Full from 0.417 to 0.670. On Small the window reaches T from week 5, so this gap is zero there.
- **Execution of a perfect plan** (0.958 → 0.852). 72 % of it is energy the fabs did not get and 28 % is wafers
  not on hand (§5.10). Our own window plans already respect the production rules, so this gap is the clairvoyant
  plan's, not ours.

**Implication.** A paradigm shift pays only where one of these gaps is large and not yet collected. On Full, myopia
was the largest gap and is the least collected beyond the credit's first step. On Small, the remaining gap is
formulation (her LP) plus information, and information has resisted every method tried.

---

## 2. The constraints a method must meet

- **Online:** at most 1.8 s CPU per week on Small and 3.5 s on Full (margins under 2 s and 4 s). Single-threaded,
  in the scoring container.
- **Imports on the server:** Python 3.13's standard library, numpy, SciPy and PyTorch (CPU) only. Anything trained
  elsewhere (Julia, Gurobi, JAX) must ship as arrays.
- **Offline:** anything goes. Local cores, cloud nodes, any solver, the package's simulator and its generator.
  The episodes' disruptions are exogenous: nothing the agent does changes them, so hindsight solves on sampled
  futures are legitimate training data.
- **Evaluation noise:** one standard error is 0.17 to 0.20 on 20 episodes, and paired comparisons are required.
  MILP agents vary by about ±0.008 between runs, and her agent by 0.008.

---

## 3. The candidates

### A. Learning to optimize: GNN-predicted bases, active sets, flow patterns

**Idea.** A graph network maps the LP (or the network state) to an initial basis or a set of likely-active
constraints, and the solver starts there.
- Smart initial basis selection (Fan et al., ICML 2023) and its tripartite-graph successor report fewer simplex
  iterations and less solve time.
- Chen et al. show GNNs can represent LP solutions.

**Fit here.** It buys speed, and speed is not what binds:
- `twopass48` already warm-starts HiGHS from the previous week's shifted basis (§5.6).
- Her LP solves in 0.07 s a week with the binaries relaxed (§5.16).
- Small's horizon is saturated (§5.9), and more MILP time bought nothing measurable (§5.12).

A faster solve would matter only if a larger model were waiting for the time. None is, unless the flagship's
headroom probe finds the full-horizon LP valuable on Full. Even then, the time-aggregated tail (§5.2) is a cheaper
way to fit it.

**Verdict:** no. Ceiling about 0, high complexity (basis repair, a training set of LPs, distribution shift between
regimes).

### B. Learned value functions as the terminal valuation (the flagship, §5)

**Idea.** Replace the constant credit $0.7\,v_k$ with $\hat V(s_{t+H})$: a value of the window's end state learned
from data. This is Powell's hybrid class, a deterministic lookahead with a value function approximation at its end.

**Lineage.**
- **Lookahead plus learned terminal value** in model-predictive control: POLO (Lowrey et al. 2019), TD-MPC and
  TD-MPC2 (Hansen et al. 2022, 2024).
- **A value network inside a math program:** PARL (Harsha et al., M&SOM 2025) puts a neural value function into the
  per-step MIP for multi-echelon inventory. It reports +44.7 % over base stock and +12.1 % over the best RL method.
- **Training from hindsight solves:** hindsight learning (Sinclair et al., ICML 2023) learns from them when, as here,
  the uncertainty is exogenous.
- **Concave piecewise-linear approximations updated from sampled duals** (SPAR/CAVE; Topaloglu and Powell): the
  classical form for resource-allocation problems, which keeps the per-step problem an LP.

**Fit here.** It is the only family that attacks the measured dominant gap (myopia on Full). It also costs nothing
online: the network outputs coefficients, so the LP gains a few segment columns and stays an LP.

**Verdict:** flagship. Its expected gain is bounded by the headroom probe.

### C. Decomposition: Lagrangian relaxation, ADMM, Benders by sector

**Idea.** Split grids, shipping and fabs, price the coupling constraints with multipliers, and solve the pieces
in parallel or faster.

**Fit here.** Decomposition is a solving technique: at convergence it returns the same optimum as the monolithic
LP, which already solves in time. ADMM converges slowly on LPs, and stopped early its primal is infeasible. That
is worse than the monolithic solve that finishes in 0.07 to 1.3 s. The server gives one thread, so no
parallelism is gained either.

**Verdict:** no. Ceiling 0, high complexity, and a risk to latency.

### D. SDDP: multi-stage stochastic programming with Markovian uncertainty

**Idea.** Build Benders cuts of each stage's cost-to-go from forward simulations and backward dual passes.
Markovian policy graphs handle persistent regimes such as a closed strait (Philpott and de Matos 2012; Dowson's
SDDP.jl). Recent work: Markov-chain-based policies for disaster-relief logistics (2022); MDP modelling of
multi-stage programs (Morton, Dowson and Pagnoncelli, 2025).

**Fit here.** The cuts could be trained offline in SDDP.jl and shipped as arrays, and online they cost one LP with
cuts. The obstacles are elsewhere:
- **State dimension.** The resource state is 84 (Small) or 155 (Full) stock slots, plus pipelines on 115 or 369
  edges. Cut quality degrades with dimension; Al-Kanj, Bouzaiene-Ayari and Powell compare SDDP with ADP on energy storage exactly on
  this point.
- **Markov state.** The joint disruption state (straits, sanctions, tariffs, fabs, grids) needs a hand-reduced chain.
- **Measured ceiling.** The stochastic gain is capped by the information gap, and the generator's own scenarios
  collected none of it once the rule was enforced (§5.14).

**Verdict:** not in the hackathon. Very high complexity; ceiling +0.00 to +0.03.

### E. Robust optimization: polyhedral and ellipsoidal sets, adjustable robust

**Idea.** Plan against a budget of disruptions (Bertsimas–Sim style) for bounded worst-case cost without scenarios.

**Fit here.** The score punishes exactly what robust plans do:
- Level 1 carries 44 % of the weight and level 4 only 4 % (§3.4).
- Precautions that move scarce supply earlier cost whenever a warning is a decoy (§4.1).
- Measured: risk-averse settings cost −0.012 (−0.020 to −0.006), all of it in levels 1 and 2.

**Verdict:** no. Expected ≤ 0.

### F. Model-free, offline and generative RL: PPO, SAC, Decision Transformer, diffusion policies

**Idea.** Learn the policy end to end from simulator rollouts, or imitate logged trajectories with sequence or
diffusion models.

**Fit here.** It cannot beat the LP teacher in the time left:
- **Action space.** Each week is a flow per route and commodity over 115 (Small) or 369 (Full) edges. Feasibility
  (stock on hand, capacities) is enforced by the simulator's clipping, not by the network.
- **Sample cost.** One Full episode takes about 135–220 s for an LP agent here. Recent benchmarks where deep RL
  beats (s, Q) policies are much smaller networks.
- **Imitation.** Imitation (Decision Transformer, diffusion) is capped by its teacher. RL fine-tuning against a
  noisy score (SE 0.17 per 20 episodes) needs thousands of episodes.

**Variant worth keeping in reserve: RL over the LP's parameters.** A small network outputs per-week coefficients of
the LP (safety stock levels, credit slopes) from the observation, and the LP stays the actor (PARL-lite; Powell's
parametric cost-function approximation). That is the flagship trained by policy search instead of regression. It is
noisier and slower, so it is the fallback if regression targets fail.

**Verdict:** no as a replacement; backup as a parameter policy.

### G. Simulator lookahead, MuZero-style planning

**Idea.** Roll the simulator forward over candidate actions and pick the best, with a learned or LP value at the leaves.

**Fit here.** Measured with full access to the true simulator state (a cheating bound; §5.4):
- one-week depth: +0.0002;
- fuel perturbations at three weeks: −0.097, because the LP's cost-to-go is biased.

The agent has no simulator snapshot online, and a learned world model would be less accurate than the one measured.

**Verdict:** no.

### H. Hindsight learning: imitation of clairvoyant solutions

**Idea.** Because disruptions are exogenous, sample futures offline, solve each in hindsight, and learn from the
solutions (Sinclair et al. 2023).

**Fit here.** Imitating the clairvoyant's **flows** inherits the clairvoyance: it would move supply ahead of
disruptions the agent cannot see. Learning the clairvoyant's **duals** (the marginal value of stock at a future
week) is better behaved, since they are averaged over futures sharing the same observation.

**Verdict:** the data source of the flagship, not a separate policy.

---

## 4. Trade-off matrix

Ceilings are gains over the current best agent on each board (Full 0.670, Small 0.821 for her LP). They are
estimates from the measurements cited, not measured results.

| Paradigm | (1) Expected RSS ceiling, Full / Small | (2) Implementation complexity | (3) Inference latency | (4) Main failure modes | Verdict |
| --- | --- | --- | --- | --- | --- |
| A. Learning to optimize (GNN bases) | ≈ 0 / ≈ 0 | High | Lower, but not binding | Invalid basis, slower than HiGHS's own warm start; regime shift | No |
| **B. Learned concave VFA in the LP** | **+0.01 to +0.05 / 0 to +0.02** | **Medium** | **+ < 5 ms, a few columns** | Hindsight bias (over-values stock); extrapolation to unseen regimes; wrong slopes make plans hoard | **Flagship** |
| C. Lagrangian / ADMM | 0 / 0 | High | Worse | Early stop gives an infeasible primal within the deadline | No |
| D. Markovian SDDP | 0 to +0.03 / 0 to +0.02 | Very high | Low (LP plus cuts) | Dimension; Markov-chain design; model mismatch with the simulator's rules | Not in time |
| E. Robust LP | ≤ 0 / ≤ 0 | Low to medium | Same | Over-hedging in levels 1 and 2 | No |
| F. End-to-end RL / DT / diffusion | Below the LP / below the LP | Very high | Low | Action dimension, sample cost, capped by its teacher | No |
| F′. RL over LP parameters | 0 to +0.03 / 0 to +0.02 | Medium to high | Same as the LP | Noisy reward, overfitting the training root | Backup |
| G. Simulator lookahead | Measured ≤ 0 | Medium | High | LP cost-to-go bias (−0.097 measured) | No |
| H. Hindsight imitation of flows | Below the LP | Medium | Low | Clairvoyance leaks into actions | Use duals, not flows |

---

## 5. Flagship: a learned, state-conditioned concave value of the window's end

### 5.1 The policy

Each week the agent solves the same window LP as `twopass48_credit`, with one change: the constant credit is
replaced by a learned value. The value is concave and piecewise linear in each slot's end-of-window stock, with
slopes predicted from what the agent observes now:

$$\max\;\ldots + \sum_{j}\sum_{m=1}^{M} \theta_{j,m}(\phi_t)\,u_{j,m},\qquad
I_{j,t+H} + X_{j,t+H} = \sum_m u_{j,m},\quad 0\le u_{j,m}\le w_{j,m},\quad
\theta_{j,1}\ge\theta_{j,2}\ge\dots\ge\theta_{j,M}\ge \nu_j .$$

- $j$ ranges over stock slots: 84 on Small, 155 on Full.
- $X_{j,t+H}$ is the cargo bound for $j$ that is still travelling at the window's end. `_terminal_topup` already
  credits exactly these terms (`I`, `Q`, in-transit `x`).
- $\nu_j$ is the scorer's salvage.
- $M = 3$ segments, with breakpoints at fractions of the slot's storage.
- The slopes $\theta$ depend on the observation $\phi_t$ and on weeks remaining, never on decisions. So the problem
  stays an LP, and concavity (decreasing slopes) is enforced by construction (a cumulative-softplus head).
- Columns added: $155 \times 3 = 465$ on Full, about 1 % of a 48-week window.

### 5.2 A no-training baseline of the same paradigm: a time-aggregated tail

Before learning anything, the deterministic version of the same object can be computed online. Append to the
48-week window a coarse tail to T, in blocks of 4 weeks:
- 14 blocks on Full at week 1, fewer later;
- persistence arrays;
- block capacities 4× weekly;
- base-load-first as in pass 2.

Its duals at the seam are a VFA under the persistence forecast. This is the teacher and the probe below, and if it
fits the budget it is also a candidate agent.

### 5.3 Training targets

Per training episode, three kinds of target are available, all offline:

| Target | How | Bias | Cost |
| --- | --- | --- | --- |
| **Hindsight duals** | One clairvoyant LP over the whole episode (base-load-first by two passes) on the realized disruptions. The duals of slot $j$'s balance at week $\tau$ give $\lambda_{j,\tau}$ for every $\tau$ from **one** solve | Optimistic: a clairvoyant needs no buffer (information-relaxation bias; Brown, Smith and Sun 2010) | 1 LP per episode |
| **Bootstrapped duals** | At week t, the agent's own planner with the window extended to T (§5.2), on the persistence forecast. The duals at week t+H | The agent's own model; no clairvoyance | 1 large LP per (episode, week): about 56 per Full episode |
| **Perturbation returns** | Add δ to slot $j$ at t+H in the simulator and continue with the current agent | Unbiased for this policy | 2 rollouts per sample: too slow except to audit |

The regression target is the bootstrapped dual, with the hindsight dual as an auxiliary input. Perturbation
returns are used only on a few hundred samples to measure the other two's bias.

### 5.4 Model

- **Inputs** $\phi_t$, computed from the observation and from the frame's `State`:
  - weeks remaining $T - (t+H)$, and t;
  - the slot's commodity, node type and storage;
  - each strait's open fraction and announced reopening;
  - pending prohibitions;
  - tariff levels on the slot's inbound routes;
  - the slot's grid ($\bar G/\bar y$) or fab (cap_eff/cap0) ratio;
  - the demand forecast's level;
  - the slot's stock and pipeline now, divided by storage.
- **Architecture:** one MLP shared across slots of the same commodity class (fuel, wafer, chip). Two hidden layers
  of 64 units, outputting M slopes as $\nu_j + \text{cumsum}(\text{softplus})$ read from the last segment upward,
  so the slopes are decreasing and $\ge \nu_j$. A cap at $v_k$ keeps the plan from hoarding.
- **Training:** PyTorch offline. Loss: Huber on $\lambda$ at the observed end stock's segment, plus a monotonicity
  penalty.
- **Export:** weights to a `.npz` beside `agent.py`. Inference is numpy matmuls (155 slots × 3 layers, under 1 ms),
  so torch is not needed on the server.

### 5.5 Data generation and compute

| Step | Size | Cost (one core) |
| --- | --- | --- |
| Episodes | 128 Full and 128 Small from a private root (`EpisodeSet.build(task, 128, entropy=…)`), never root 0 or 20261006 | — |
| Closed-loop states | `twopass48_credit` played, logging $\phi_t$ and the window-end state every week where t+H < T: about 56 per Full episode, about 7,000 samples | About 6 h |
| Bootstrapped duals | One extended-window LP per logged week, no CPU limit | About 7,000 × 3–6 s ≈ 6–12 h; 4 cores ≈ 2–3 h |
| Hindsight duals | One clairvoyant LP per episode | Minutes |
| Training | 7,000 × 155 slot-rows | Minutes on CPU |

Everything goes under `scripts/research/vfa/`, writing to `outputs/research/vfa/<date_time>/`, as the other research
scripts do. The agent is `agents/vfa48`: a copy of `twopass48_credit` whose `LPEdits.terminal_credit` becomes
`terminal_values(theta, widths)`, with the weights in `vfa_weights.npz`.

### 5.6 Gates (each one decides whether to continue)

| Gate | Question | Measurement | Continue if |
| --- | --- | --- | --- |
| **0. Headroom** (1–2 h) | How much does a better terminal value buy at all? | 8 Full private episodes, serial, no CPU limit: (a) the 48-week window with the tail of §5.2; (b) a 104-week window; both with two passes. Compare with stored `twopass48_credit` 0.670 | (a) or (b) beats 0.670 by **+0.015** or more, with the paired interval excluding 0 |
| 1. Fit (offline) | Do the slopes generalize? | Held-out root: rank correlation of predicted to bootstrapped duals; bias against perturbation returns on 200 samples | Rank correlation ≥ 0.6; bias below 30 % of $v_k$ |
| 2. Closed loop | Does it pay? | `agent_sweep.py` on the 8 Full private episodes, serial, paired against stored `twopass48_credit` | +0.01 with the interval excluding 0 |
| 3. Confirmation | Is it the training root? | 16 fresh Full episodes, paired | Same sign and interval |
| 4. Deployability | Does it fit? | `sbf check vfa48 --task=full --docker` and `--task=small` | Worst week ≤ 3.5 s (Full), ≤ 1.8 s (Small) |

If Gate 0 fails, the constant credit already collects the tail's value. Stop this line and put the time into
formulation fidelity on Full instead: porting `mpc_nobuf`'s LP, whose +0.04 on Small is the largest unexplained
gain we know of (§5.16).

### 5.7 Failure modes and guards

- **Hoarding.** Over-valued slopes buy stock the episode never uses. Guard: slopes capped at $v_k$, and Gate 2's
  paired test covers levels 1 and 2, where hoarding costs.
- **The cliff at week 57.** Today the credit switches off when the window first reaches T. A learned
  weeks-remaining input tapers it instead. Check the weeks around 57 in Gate 2's per-week costs.
- **Regimes unseen in training.** Clip $\phi$ to the training range, and fall back to the constant 0.7 credit when
  the input is out of range.
- **Latency.** Inference is under 1 ms, plus 465 columns. Only Gate 0's extended-window variant could threaten the
  4 s budget, and it is a probe, not the agent.
- **Small.** The window reaches T from week 5, so the terminal value matters only in weeks 1–4. A Small gain would
  need the same network inside the window, as risk-conditioned safety stocks. Her safety stocks are worth −0.012
  when removed (§5.16), which bounds what to expect there.

### 5.8 Sequence

1. Gate 0: the tail LP and the 104-week window on 8 Full episodes. This is the only step before a go/no-go.
2. Data generation on the training root: states, bootstrapped and hindsight duals.
3. The model and Gate 1.
4. `agents/vfa48` and Gates 2–4.
5. If time remains: the same slopes inside the window on Small (risk-conditioned safety stocks), measured against
   her `mpc_nobuf`.

---

## 6. What this assessment does not claim

- The ceilings in §4 are estimates. Only the rows marked "measured" in §1 and §3 are measurements.
- Nothing here measures the board. Dev and private-pool scores are not board scores (§3.5).
- The flagship's case rests on Full's myopia gap. If Gate 0 shows that gap already closed by the constant credit,
  the flagship is not worth building, and §5.6 says where to go instead.

---

## References

- Harsha, Jagmohan, Kalagnanam, Quanz, Singhvi. *Math programming based reinforcement learning for multi-echelon
  inventory management* (PARL). NeurIPS 2021 workshop; M&SOM 2025. [arXiv:2112.02215](https://arxiv.org/abs/2112.02215v2)
- Sinclair et al. *Hindsight learning for MDPs with exogenous inputs.* ICML 2023.
  [PMLR v202](https://proceedings.mlr.press/v202/sinclair23a.html)
- Topaloglu, Powell. *An algorithm for approximating piecewise linear concave functions from sample gradients.*
  [Princeton](https://collaborate.princeton.edu/en/publications/an-algorithm-for-approximating-piecewise-linear-concave-functions/);
  Powell, *Reinforcement Learning and Stochastic Optimization*, ch. 18 (convex resource allocation).
  [O'Reilly](https://oreilly.com/library/view/reinforcement-learning-and/9781119815037/c18.xhtml)
- Al-Kanj, Bouzaiene-Ayari, Powell. *SDDP vs. ADP: the effect of dimensionality in multistage stochastic optimization for grid
  level energy storage.* [arXiv:1605.01521](https://arxiv.org/pdf/1605.01521)
- Dowson. SDDP.jl and policy graphs. [JuMP-dev 2025 talk](https://jump.dev/assets/jump-dev-workshops/2025/talk-oscar-dowson-sddp.pdf);
  *MDP modeling for multi-stage stochastic programs.* [arXiv:2509.22981](https://arxiv.org/pdf/2509.22981)
- *Markov chain-based policies for multi-stage stochastic integer linear programming with an application to
  disaster relief logistics.* [arXiv:2207.14779](https://arxiv.org/pdf/2207.14779)
- Fan et al. *Smart initial basis selection for linear programs.* ICML 2023.
  [mlanthology](https://mlanthology.org/icml/2023/fan2023icml-smart/); GNN initial-basis follow-up:
  [arXiv:2502.02446](https://www.arxiv.org/pdf/2502.02446)
- Lowrey et al. *Plan online, learn offline* (POLO). [arXiv:1811.01848](https://arxiv.org/pdf/1811.01848);
  Hansen et al. *TD-MPC* [PMLR v162](https://proceedings.mlr.press/v162/hansen22a), *TD-MPC2*
  [arXiv:2310.16828](https://arxiv.org/pdf/2310.16828)
- Input convex neural networks as LP-embeddable value functions: *ICNN-enhanced 2SP*
  [arXiv:2505.05261](https://arxiv.org/html/2505.05261v2); *ICNNs as surrogates in mathematical optimisation*
  [arXiv:2608.09707](https://arxiv.org/pdf/2608.09707)
- Deep RL for supply chain inventory, benchmark against (s, Q): [arXiv:2306.11246](https://arxiv.org/abs/2306.11246)
- Brown, Smith, Sun. *Information relaxations and duality in stochastic dynamic programs.* Operations Research 2010.

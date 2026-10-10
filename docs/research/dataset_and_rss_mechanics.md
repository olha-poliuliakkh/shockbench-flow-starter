# ShockBench-Flow: simulator mechanics, RSS and where the score is lost

Status: 2026-10-08. Package `shockbench-flow` 0.1.2 (the version in `uv.lock`). Unless stated otherwise, numbers
come from the **20 Small dev episodes** (root 0, 5 per harm level). Gaps between two policies are **paired** on the
same episodes, with a 90 % bootstrap interval. Code citations are to the installed package under
`.venv/lib/python3.13/site-packages/shockbench_flow/` (abbreviated `sbf/`), and to this repository. The measurement
scripts are in [`scripts/research/`](../../scripts/research/). The JSON outputs this document cites are in
[`scripts/research/results/`](../../scripts/research/results/); a bare file name in a *Source* column refers to that
folder. New runs write to `outputs/research/<script>/<date_time>/`. See [Reproducibility](#8-reproducibility).

> **Corrections to claims made earlier in the project.** This document supersedes them.
>
> 1. *"The reference lets grids shed residential load to power fabs, so a ~0.21 gap is unreachable."* **False.** An
>    exact mixed-integer bound that forbids it still allows **RSS ≤ 0.973** (§3.3). The rule is a **planning-model
>    error that can be fixed**. Replaying a plan that respects the rule, open loop, scores **0.852** against 0.765
>    for the LP plan (paired **+0.087**, interval +0.056 to +0.124).
> 2. *"~0.805 is the LP ceiling on Small dev."* **False.** 0.805 is the closed-loop perfect-information planner with a
>    48-week window. A plan that respects the rule beats it **open loop** (0.852, §5.1).
> 3. *"Disposal (4.65M units) is a primary loss."* **False.** Disposal costs **$2.0B to $3.2B per episode, about
>    0.1 % of cost** (§2.4).
> 4. *"Lot starts are discrete batches (`queue_lots`)."* **False.** Lot starts are continuous. `queue_lots` is cargo
>    waiting at straits (§2.1, §2.3).
> 5. *"The LP assumes no energy throttling."* **False.** The LP has the row $e_f p_f \le R_f E_f$. What it relaxes is
>    the **priority** of base load over fabs after week 1 (§2.2).

---

## 1. Executive summary

| Quantity (Small dev, 20 episodes) | RSS |
| --- | ---: |
| `mpc_det`, the package's rolling LP, persistence forecast, 24-week window | 0.729 |
| Our agent `agents/mine` (vendored `mpc_det`, SciPy HiGHS) | 0.727 |
| Perfect disruption information in the window, 24 / 48 weeks | 0.786 / 0.805 |
| The clairvoyant LP's flows replayed open loop | 0.765 |
| **A clairvoyant plan that respects base-load-first, replayed open loop** | **0.852** |
| **`mpc_det` with the two-pass base-load-first solve, closed loop** (§5.5) | **0.748** |
| The same with perfect disruption information, 24 weeks | 0.804 |
| **`agents/twopass48`: two passes, warm-started 48-week window** (§5.6) | **0.764** (Full, 8 private: **0.417**) |
| **`agents/milp48`: plus the exact base-load-first window MILP, 0.85 s per week** (§5.11) | **0.775 to 0.783** (2 runs) |
| **`agents/milp48t`: tuned MILP (1.05 s, interrupt, light, gap 0.01, sparse)** (§5.12) | **0.783** |
| Persistence planner with a 10 s window MILP per week (offline, 8 episodes; two-pass 0.769 there) | 0.832 |
| Upper bound for **any** policy under base-load-first (MILP dual bound) | 0.973 |
| The clairvoyant reference itself | 1 by definition |

1. **What limits the score is the model, not the information.**
   - Perfect knowledge of every disruption adds **+0.057** to the 24-week planner (0.729 → 0.786).
   - Perfect demand on top adds **+0.0002**.
   - Belief and forecast work (Phase 1, Optuna) measured **≤ +0.0002**, an interval holding 0.
2. **One relaxed rule explains most of the plan-to-execution loss.**
   - **The rule.** Each grid serves its residential base load before any fab (`base_first`). The planner's LP enforces this only in **week 1**.
   - **What the LP plans.** It schedules lot starts for weeks 2..H that draw energy the simulator will give to households instead. On average the LP plans **9.38M** lots per episode, and the simulator executes **4.53M**.
   - **The fix in plan space.** A plan that enforces the rule in every week plans 7.71M lots and executes 6.55M. It replays at **+0.087** RSS.
   - **The fix in closed loop** (§5.5). A second LP solve per week, with the rule's violations bounded out, gives `mpc_det` **+0.019** on Small dev (interval +0.009 to +0.031) and **+0.014** on 8 Full episodes (+0.008 to +0.018). The submission candidate `agents/twopass` gains **+0.013** over `agents/mine` on Small dev (+0.008 to +0.019).
   - **Closed loop does not reach the open-loop 0.852.** With perfect information, two passes reach 0.804 at 24 weeks. Where the remaining gap sits is not yet measured (§6).
3. **RSS = 1 is out of reach, but only narrowly so.**
   - On dev, base-load-first alone caps any policy at **0.973**.
   - Other relaxed rules (§2) may lower that cap further. This is not yet measured.
4. **Comparing to the board's 0.88: the data do not support a confident comparison.**
   - One standard error of a 20-episode score is 0.17 to 0.20 (`docs/GUIDE.md:293-298`).
   - Our own agent scored 0.749 to 0.805 on two private held-out pools (§3.5).
   - No board score of ours is recorded in this repository. One calibration upload (the participant's decision) is the only way to know where we stand.

---

## 2. Anatomy of the simulator

One week runs ten steps in a fixed order (`sbf/dynamics/sim.py`). The ones below decide the loss.

### 2.1 Fabs and OSATs

**Lot starts** (`sbf/dynamics/sim.py:322-326, 349-354`). For fab $f$ in week $t$:

$$
\hat p_f^t = \min\!\big(\bar\alpha_f^t R_f^t \,\mathrm{cap}^0_f,\; I^t_{f,\mathrm{wafer}}\big),
\qquad
\hat E_f = \frac{e_f\,\hat p_f^t}{R_f^t},
\qquad
p_f^t = \min\!\Big(\hat p_f^t,\; \frac{R_f^t E_f^t}{e_f}\Big),
$$

The symbols:
- $\bar\alpha$ is the power multiplier, $R$ the restoration factor and $\mathrm{cap}^0$ the nominal wafer capacity.
- $I^t_{f,\mathrm{wafer}}$ is the wafer stock after this week's arrivals.
- $E_f^t$ is the energy the grid allocates to the fab (§2.2).

**What this means for control:**
- Lot starts are **continuous**, and the agent does not choose them. The simulator **pushes**: every wafer on hand starts, up to capacity and energy.
- Delivering fewer wafers changes nothing when a fab already holds more than it can start. A one-episode test throttled wafer flows to 0.5×, and lot starts stayed at exactly 960,896 (`scripts/bound_hybrid_limit.py`, first version).

**Work in progress** (`sim.py:364-376`). Starts leave the wafer stock and are booked as WIP. Fab hits scrap a fraction $\sigma$ of starts in their window. The product arrives after $\tau_f$ weeks.

**OSAT packaging** (`sbf/dynamics/production.py:41-49`). With raw-chip stocks $r_k$ and throughput $\mathrm{thr} = \mathrm{thr}_i R^{\mathrm{osat}}_i(t)$:

$$
\xi_k = \begin{cases} r_k & \text{if } \sum_k r_k \le \mathrm{thr} \\ \mathrm{thr}\cdot r_k/\sum_j r_j & \text{otherwise.} \end{cases}
$$

This is continuous and pro rata, again a push rule.

**Sizes.** Small has 6 fabs, 3 OSATs and 4 grids. Full has 16 fabs, 7 OSATs and 8 grids (`docs/fields/full.md:30-35`).

### 2.2 Energy grids and the base-load-first invariant

**Available output** (`sim.py:330-347`). Grid $g$ has fuel segments $k$ with shares $\zeta_{g,k}$, deliverable generation $\bar G_g^t$ and fuel stock $I_{g,k}$:

$$
\mathrm{av}_k = \min\!\big(\zeta_{g,k}\bar G_g^t\,\rho_k,\; I_{g,k}\big),\qquad
\rho_k = \begin{cases}\min\!\big(1, I^{t-1}_{g,k}/(\psi \bar I_{g,k})\big) & k = \text{the rationed fuel}\\ 1 & \text{otherwise}\end{cases},
\qquad
G^{\mathrm{av}}_g = \sum_k \mathrm{av}_k + \zeta_{g,0}\bar G_g^t .
$$

**Allocation** (`sbf/dynamics/production.py:24-27`). Every grid of Small (4) and Full (8) is `base_first` (checked on the instances):

$$
y_g = \min(\bar y_g^t,\; G^{\mathrm{av}}_g),\qquad
E_f = \hat E_f \cdot \min\!\Big(1, \frac{G^{\mathrm{av}}_g - y_g}{\sum_{f'\in F_g}\hat E_{f'}}\Big),
\qquad \text{shed}_g = \bar y_g^t - y_g .
$$

**The invariant.** Every trajectory of every policy satisfies

$$
\text{shed}_g^t > 0 \;\Longrightarrow\; \sum_{f\in F_g} E_f^t = 0 .
$$

Fabs receive energy only after base load is fully served.

**How throttling propagates.** Fuel arrives at a grid from its terminal ($\texttt{term\_X}\to\texttt{grid\_X}$) or directly from a source. Fuel on hand caps each segment, rationing cuts the rationed fuel below $\psi\bar I$, and base load takes the first $\bar y$ of output. Fab energy, and through it $p_f$, is the residual.

**What the planning LP does with this** (`sbf/oracle/lp.py:3-4, 44-55, 600-708`):
- **Energy.** The LP *does* model it: $e_f p_f \le R_f E_f$, $\sum_f E_f + y_g \le \sum_k G_{g,k}$, rationing as a linear row.
- **The priority.** It relaxes the priority to a feasible set. `mpc_det`'s planning rules restore it only as a **price on shed base load in week 1** (`lp.py:702-705`: `self.priority[jsh] = price * first`).
- **Other week-1-only rules.** Lot starts at the push rule (`lot_start`) and OSAT starts at the packaging rule (`osat_start`). Pro-rata segment loading holds in every week; base-load-first does not.

### 2.3 Ports, chokepoints and queues

- **Cargo at straits.** `queue_lots.qty` has shape (lot keys, T). Row $i$ is a (strait node, commodity, lane, next edge) tuple; column $w-1$ holds the quantity that reached the strait in week $w$ and still waits, as first-in-first-out cohorts (`docs/GUIDE.md:123-131`, `docs/fields/small.md:15`). It has nothing to do with fab lots.
- **Transit.** Each edge has a transit time $\tau_e \ge 1$ week, so a dispatch in week $t$ affects stocks no earlier than week $t+1$.
- **Orders.** Requests above route capacity or stock are **clipped**, not refused (`docs/GUIDE.md:21-23`).

### 2.4 Disposal

Disposal is overflow (`sim.py:407-411`): at non-chokepoint, non-supply nodes, $O_s^t = \max(0,\, I_s^t - I^{\max}_s)$. It is a symptom of stock piling up, for instance wafers a fab cannot start.

| Run | Disposal, $B per episode | Shortage | Shed |
| --- | ---: | ---: | ---: |
| Clairvoyant LP flows, open loop | 3.2 | 1,047.8 | 1,703.3 |
| Clairvoyant plan under base-load-first, open loop | 2.5 | 962.6 | 1,716.3 |
| Perfect-information planner, 24 weeks, closed loop | 2.0 | 1,038.1 | 1,699.3 |

The earlier "4.65M units" counted units, which mixes commodities. In dollars, disposal is about 0.1 % of an episode's roughly $2.8T. **Shed and shortage are about 99 % of cost** (`mentor_sync_summary.md` §1).

---

## 3. The RSS metric

### 3.1 Definition (`docs/GUIDE.md:244-263`)

For episode $n$: $g_n = J^{\text{naive}}_n - J_n$ (the saving) and $D_n = J^{\text{naive}}_n - J^{\text{clair}}_n$ (the attainable saving). With harm levels $s\in\{1,2,3,4\}$, level means $\bar g_s,\bar D_s$ and weights $p=(0.50,0.30,0.15,0.05)$:

$$
\mathrm{RSS} = \frac{\sum_s p_s\,\bar g_s}{\sum_s p_s\,\bar D_s}.
$$

The naive rule scores 0 and the clairvoyant plan scores 1; below 0 is not clipped. Episode sets hold equal numbers per level, and $p$ restores the generator's mix (level 1 = the calmer half, level 4 = the worst 5 %).

### 3.2 The reference

$J^{\text{clair}}$ is the optimum of the time-expanded LP `build_lp(inst, marks)` **without** planning rules. It is "the simulator's equations with the clip, the default container release, the energy priority rule and the allocation (18) replaced by the feasible sets they project onto" (`lp.py:3-4`). The organisers keep it a bound: "J^oracle stays a bound (V5)" (`lp.py:48-50`). Every simulator trajectory is feasible for this LP, so no policy beats it; it relaxes **every** rule of §2.

### 3.3 How far below 1 is attainable: an exact base-load-first bound

Take the reference LP and add, for every grid $g$ and week $t$, a binary $z_{g,t}$:

$$
\sum_{f\in F_g} E_{f,t} \le M_{g,t}\, z_{g,t},\qquad
\mathrm{ysh}_{g,t} \le \bar y_{g,t}\,(1 - z_{g,t}),\qquad
M_{g,t} = \bar G_{g,t}\textstyle\sum_k \zeta_{g,k}.
$$

The simulator satisfies these rows (§2.2), so the MILP is still a relaxation. Its dual bound is an upper bound on **any** policy's RSS on these episodes. Small has 208 binaries per episode. With a 60 s limit, HiGHS stopped at the limit on 15 of 20 episodes; the relative gap between incumbent and bound had a median of $1.3\times10^{-3}$ and a maximum of $4\times10^{-2}$.

| Dev, 20 episodes | RSS | L1 | L2 | L3 | L4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MILP dual bound (upper bound under base-load-first) | **0.973** | 0.979 | 0.949 | 1.000 | 0.978 |
| MILP incumbent (a plan respecting the rule, in plan space) | 0.958 | 0.958 | 0.936 | 0.995 | 0.959 |

A 120 s per-episode run gave 0.9725 and 0.9623 (`base_first_milp_120s.json`); the table shows the 60 s run (`base_first_exec.json`).

**How much the rule matters:**
- **The reference.** Its LP diverts only **1.7 %** of its shed energy from base load to fabs, which is worth **≤ 0.03** of RSS.
- **Planners.** It matters far more for planning (§5.1), because a planner that assumes the diversion schedules lots that never start.

**A bad approximation.** Imposing the rule as a **price** on shed base load in every week, `mpc_det`'s week-1 price $V/\min_f e_f$ extended to all weeks, does **not** represent it. The price also penalises shed that cannot be avoided, and it distorts the whole plan. The "clairvoyant LP with the price" scores 0.786; that is not a bound, and its closeness to the 24-week planner is a coincidence.

### 3.4 The value of a dollar by level, and why over-hedging loses

$\partial\,\mathrm{RSS}/\partial \bar g_s = p_s / \sum_{s'} p_{s'}\bar D_{s'}$, so a dollar of mean saving in level 1 is worth $p_1/p_4 = 10$ dollars in level 4.

| Level | Mean attainable saving $\bar D_s$ (dev, $T per episode) | Effective weight $p_s\bar D_s/\sum p\bar D$ |
| --- | ---: | ---: |
| 1 | 0.820 | 0.44 |
| 2 | 1.018 | 0.33 |
| 3 | 1.125 | 0.18 |
| 4 | 0.757 | 0.04 |

- **Where the score lives.** Ordinary weeks of calm episodes carry most of it.
- **Why hedging loses.** A precaution that moves scarce supply earlier or elsewhere costs shed and shortage whenever the threat is a decoy (§4.1).
- **Measured.** Risk-averse belief settings cost **−0.012** (interval −0.020 to −0.006) on 32 private episodes, all of it in levels 1-2 (`mentor_sync_summary.md` §3).
- **Minimax.** Planning against the worst case optimises the level carrying 4 % of the weight.

### 3.5 Dev, private pools and the board

| Agent and pool | Episodes | RSS |
| --- | ---: | ---: |
| `mine`, Small dev (root 0) | 20 | 0.727 (`mine_dev.json`) |
| `mine`, private held-out pool, root 20261005, run 1 | see note | 0.805 (`tune_local_2026-10-05_17-27-12.json`) |
| `mine`, private held-out pool, root 20261005, run 2 | 64 | 0.749 (`tune_local_2026-10-05_17-40-28.json`) |
| Leaderboard top three (reported by the participant) | 200 hidden | 0.882, 0.871, 0.867 |

The summaries do not record run 1's pool size, so the two rows may not be the same pool.

- **Dev noise.** One standard error of a 20-episode score is 0.17 to 0.20 (`docs/GUIDE.md:293-298`).
- **What we cannot say.** Nothing here shows that our agent reaches 0.88, nor that it doesn't.
- **What the data show is possible.** A planner that respects base-load-first can sit well above 0.80 (§5.1). That gives a plausible mechanism for 0.88, not a proof.

---

## 4. Signals and information

### 4.1 Exogenous signals in the `standard` regime

| Signal | What it is | Measured value |
| --- | --- | --- |
| `warning.score` | Latent hazard with a one-week lag, loading $a=0.604$ (`sbf/information/theta.py:53,80`) | AUROC **0.41 to 0.55** for onset within the episode: no usable signal |
| `messages` threads | Tariff, sanction and military threats; decoys carry final notices and legal publications too, until withdrawn at their effect week | Decoy shares $\varphi$: tariff_formal 0.25, tariff_informal 0.36, ties_threat **0.538**, mid_threat **0.298** (`theta.py:55-58, 203`) |
| `mid_threat` on a fully open strait | Military threat before a closure | 20 of 21 followed by a closure, median lead **3.5 weeks** |
| `pending_prohibitions` | Announced sanctions with their effect week | Median lead **5 weeks**; decoys exist |
| Closure end | Hidden in the scored regime | — |

Source: traces recorded by [`scripts/research/signals.py`](../../scripts/research/signals.py) on 52 Small episodes (32 of private root 20261004 and the 20 dev). The statistics were computed from those traces interactively and are **not scripted yet**, so treat them as provisional.

**Closure durations:**
- **72 %** of closures end within the episode, with a median length of **1 week**.
- The rest last a median **35 weeks**, censored at the episode's end.
- New onsets average **1.17 per episode**, similar across levels.

### 4.2 Demand

The window's demand comes from `demand_forecast` for weeks $t..t+7$, then the seasonal mean $\bar d\, m^{\text{sea}}(t+h)$ (`sbf/policies/lp_common.py:696-728`, `point_demand`). With perfect disruption information, replacing the forecast with the true demand changes RSS by **+0.0002**: 0.7861 → 0.7863 (`oracle_window.json`). Demand error is not a lever.

### 4.3 The value of information

| Network | Persistence forecast | Perfect disruption information | Gain |
| --- | ---: | ---: | ---: |
| Small dev, 24-week window | 0.729 | 0.786 | **+0.057** |
| Small dev, 48-week window | 0.738 | 0.805 | +0.068 |
| Full, 8 private episodes, window L | 0.332 | 0.408 | +0.075 |

These gains cap what any belief model can add at a given horizon. Our belief blocks measured **−0.0014** (interval −0.0033 to +0.0001). Optuna over 11 parameters measured **+0.0002** (interval −0.0010 to +0.0013) on 64 held-out episodes (`mentor_sync_summary.md` §3-4).

---

## 5. Benchmarks and post-mortems

### 5.1 Small dev (20 episodes)

| Policy | Info | Window | Loop | RSS | L1 | L2 | L3 | L4 | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mpc_det` | persistence | 24 | closed | 0.729 | 0.710 | 0.752 | 0.730 | 0.758 | `base_first.json` |
| `mpc_det`, base-load price in all weeks | persistence | 24 | closed | 0.713 | 0.686 | 0.736 | 0.726 | 0.756 | `base_first.json` |
| `mpc_det` | persistence | 32 / 48 / 52 | closed | 0.735 / 0.738 / 0.736 | | | | | `oracle_h.json` |
| `mpc_scen` (scenario LP) | generator | 24 | closed | 0.747 | | | | | `oracle_window.json` |
| Oracle window | true disruptions | 24 | closed | 0.786 | 0.779 | 0.802 | 0.769 | 0.807 | `base_first.json` |
| Oracle window, base-load price in all weeks | true disruptions | 24 | closed | 0.765 | 0.748 | 0.786 | 0.762 | 0.799 | `base_first.json` |
| Oracle window | true disruptions | 48 | closed | 0.805 | 0.803 | 0.817 | 0.780 | 0.836 | `oracle_h.json` |
| `mpc_det`, **two-pass "fix"** (§5.5) | persistence | 24 | closed | **0.748** | 0.735 | 0.765 | 0.743 | 0.766 | `two_pass_small.json` |
| Oracle window, **two-pass "fix"** (§5.5) | true disruptions | 24 | closed | 0.804 | 0.804 | 0.814 | 0.782 | 0.816 | `two_pass_small.json` |
| Oracle window | true disruptions | 52 | closed | 0.802 | 0.802 | 0.814 | 0.775 | 0.833 | `oracle_h.json` |
| Clairvoyant LP flows | everything | 52 | **open** | 0.765 | 0.753 | 0.781 | 0.757 | 0.802 | `base_first_exec.json` |
| **Clairvoyant MILP (base-load-first) flows** | everything | 52 | **open** | **0.852** | 0.867 | 0.810 | 0.899 | 0.812 | `base_first_exec.json` |
| Clairvoyant MILP, in plan space | everything | 52 | — | 0.958 | 0.958 | 0.936 | 0.995 | 0.959 | `base_first_exec.json` |
| MILP dual bound | everything | 52 | — | 0.973 | 0.979 | 0.949 | 1.000 | 0.978 | `base_first_exec.json` |
| Send the maximum / heuristic | — | — | — | 0.409 / 0.400 | | | | | earlier runs |

More measurements:
- **Paired gap, MILP flows vs LP flows** (open loop): **+0.0865**, interval +0.0558 to +0.1236.
- **Lots per episode:**

  | Plan | Planned | Executed |
  | --- | ---: | ---: |
  | Clairvoyant LP | 9.38M | 4.53M |
  | Clairvoyant MILP | 7.71M | 6.55M |
- **`hindsight_consensus`** (an organiser baseline), on 7 dev episodes: it saves 0.635 of the attainable saving, against 0.657 for `mpc_det` on the same episodes. Median 3.3 s per step, maximum 69 s.

### 5.2 Full (private root 20261006, 8 episodes, 2 per level; `oracle_full.json`, `oracle_full_h.json`)

| Policy | Window | RSS | L1 | L2 | L3 | L4 | Max s/week (local) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mpc_det` persistence | L | 0.332 | 0.432 | 0.230 | 0.141 | 0.477 | 0.77 |
| `mpc_det` persistence | L+8 | 0.365 | 0.465 | 0.262 | 0.179 | 0.487 | 2.03 |
| `mpc_det` persistence | 48 | **0.432** | 0.523 | 0.337 | 0.283 | 0.500 | **4.46** |
| Oracle window | L | 0.408 | 0.515 | 0.297 | 0.213 | 0.527 | 0.79 |
| Oracle window | 48 | 0.549 | 0.639 | 0.464 | 0.393 | 0.559 | **4.69** |

On 32 Full episodes, `mine` scores 0.3313 against `mpc_det`'s 0.3327 (paired −0.0013, interval −0.0035 to +0.0008). Base-load-first bounds and replays on Full are **not measured** yet.

### 5.3 CPU (our SciPy `highs-ds` port, cold solve)

| Network | LP columns | Median s/week | Max s/week | Container worst week | Budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small, 24-week window | ~10k | 0.09 to 0.12 | 0.14 | 0.37 | 2 s |
| Full, window L | ~27.8k | 0.57 | 0.80 | 1.40 | 4 s |
| Small, `agents/twopass` (two passes) | ~10k | 0.17 to 0.19 | 0.27 | **0.27** | 2 s |
| Full, `agents/twopass` (two passes) | ~27.8k | 0.64 to 0.85 | 1.65 | **1.74** | 4 s |
| Small, `agents/twopass48` (warm, 24 + 48 weeks, two passes each) | 10k + 19k | 0.19 | 0.41 | **0.42** | 2 s |
| Full, `agents/twopass48` (warm, 25 + 48 weeks, two passes each) | 27.8k + 55.5k | 1.09 to 2.16 | 3.46 | **3.45** | 4 s |

`agents/twopass48`'s week is bounded by its deadlines (§5.6). Its worst Full week leaves 0.55 s of margin in the
container, where SciPy's bundled HiGHS imports and runs (`sbf check twopass48 --docker`: 1 Small and 2 Full episodes,
0 weeks given to naive). The `agents/twopass` rows come from `sbf check twopass --docker`: 1 Small episode and 2 Full episodes, both locally and
in the scoring container, with 0 weeks given to naive. The CPU guard of §5.5 (1.0 s / 2.5 s) was never reached.

Peak memory is 489 MB of 4 GB. The interior-point method is too slow on Full.

### 5.4 Post-mortem: lookahead in the simulator (`scripts/bound_hybrid_limit.py`)

**Setup.** The base is the oracle-window planner. `Env.snapshot()` and `Env.restore()` give an exact rollback (`sbf/dynamics/env.py:676-708`). Each candidate's score is its real cost over the simulated weeks plus the LP's estimate of the remaining cost.

| Variant | Episodes | Base RSS | Lookahead RSS | Paired gap | Predicted gain | Realised gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Throttle wafers into fabs and raw chips into OSATs; 1-week depth; 14 candidates | 20 | 0.7861 | 0.7863 | +0.0002 (interval −0.0000 to +0.0004) | — | ≈ 0 |
| Rescale fuel to grids; 3-week depth; 12 candidates | 3 | 0.8294 | 0.7328 | **−0.0965** (interval −0.154 to −0.048) | $790.5B | **−$322.8B** |

1. **Chip-side throttles move the wrong lever.** Fabs hold more wafers than they start, and the binding limit is fab energy after base load (§2.2).
2. **The LP's estimate of remaining cost is biased, and taking the minimum over candidates exploits that bias.**
   - The LP estimates the cost after the simulated weeks while still assuming base load can be diverted.
   - Cutting fuel now and repairing it later therefore looks cheap to the LP, and the minimum over candidates picks whichever action the LP undervalues most.
   - The simulated weeks themselves are exact, so the false savings come from the LP's estimate.
   - A longer simulated stretch doesn't help while it still ends on that estimate. A rollout to the episode's end (`--depth=0`) avoids it, but costs about 25 LP solves per candidate per week.

### 5.5 Two-pass base-load-first solve, closed loop (`scripts/research/two_pass.py`, `agents/twopass`)

**Method.** Each week, pass 1 is the planner's own window LP. A grid-week $(g,t)$ with $t\ge 2$ **violates** the rule
when the plan sheds base load there while the grid's fabs draw energy: $\mathrm{ysh}_{g,t} > 0$ and
$\sum_{f\in F_g}E_{f,t} > 0$. Plans violate it constantly: about **12 grid-weeks per window on Small and 36 on Full,
every week**. Pass 2 re-solves the same LP with **column bounds only** (no new rows):

| Mode | Bound for each grid-week of pass 1's plan |
| --- | --- |
| zero | Every grid-week that sheds base load: $E_{f,t} \le 0$ for the grid's fabs |
| fix | Only violated grid-weeks: $\mathrm{ysh}_{g,t} \le 0$ where the fabs' energy would have covered the shed ($\sum_f E_{f,t} \ge \mathrm{ysh}_{g,t}$), else $E_{f,t} \le 0$ |

Pass 1's plan stands when pass 2 is skipped or returns no optimum. "fix" mirrors the simulator: base load is fully
served whenever the energy suffices, and fabs get energy only then. "zero" also darkens fabs in grid-weeks that
could have served base load and still had energy left over.

**Small dev, 20 episodes**, package solver with pass 2 warm-started from pass 1's basis (`two_pass_small.json`,
`two_pass_small_3passes.json`):

| Planner | Mode | RSS | L1 | L2 | L3 | L4 | Paired gap vs as shipped (90 %) | s/week median / max | Weeks with pass 2 | Violations left per pass-2 week | Lots vs shipped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `mpc_det` | as shipped | 0.7294 | 0.710 | 0.752 | 0.730 | 0.758 | — | 0.053 / 0.244 | 0 % | — | 1.000× |
| `mpc_det` | zero | 0.7343 | 0.714 | 0.752 | 0.743 | 0.766 | +0.0049 (−0.0062 to +0.0155) | 0.074 / 0.190 | 98 % | 0.14 | 1.050× |
| `mpc_det` | **fix** | **0.7481** | 0.735 | 0.765 | 0.743 | 0.766 | **+0.0187 (+0.0091 to +0.0312)** | 0.075 / 0.174 | 73 % | 2.11 | 1.093× |
| `mpc_det` | fix, 3 passes | 0.7503 | 0.738 | 0.767 | 0.746 | 0.766 | +0.0209 (+0.0104 to +0.0341) | 0.076 / 0.192 | 73 % | 0.56 | 1.099× |
| Oracle window | as shipped | 0.7861 | 0.779 | 0.802 | 0.769 | 0.807 | — | 0.051 / 0.164 | 0 % | — | 1.000× |
| Oracle window | zero | 0.7907 | 0.785 | 0.800 | 0.783 | 0.816 | +0.0046 (−0.0073 to +0.0167) | 0.074 / 0.198 | 98 % | 0.16 | 1.044× |
| Oracle window | **fix** | **0.8038** | 0.804 | 0.814 | 0.782 | 0.816 | **+0.0177 (+0.0085 to +0.0300)** | 0.075 / 0.212 | 73 % | 1.94 | 1.083× |
| Oracle window | fix, 3 passes | 0.8047 | 0.804 | 0.816 | 0.782 | 0.816 | +0.0186 (+0.0085 to +0.0318) | 0.079 / 0.225 | 73 % | 0.52 | 1.086× |

**Full, 8 private episodes** (root 20261006, 2 per level; `two_pass_full.json`):

| Planner | Mode | RSS | L1 | L2 | L3 | L4 | Paired gap (90 %) | s/week median / p99 / max | Weeks with pass 2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `mpc_det` | as shipped | 0.3324 | 0.432 | 0.230 | 0.141 | 0.477 | — | 0.199 / 0.512 / 0.870 | 0 % |
| `mpc_det` | **fix** | **0.3461** | 0.446 | 0.239 | 0.166 | 0.487 | **+0.0137 (+0.0080 to +0.0181)** | 0.377 / 0.845 / 1.390 | 87 % |

**Reading:**
- "fix" helps in every level, on both networks, with and without perfect information. "zero" does not measurably help.
- A third pass, which bounds the violations pass 2 leaves, adds about +0.002, inside the intervals.
- Closed loop stays far below the open-loop replay (0.804 against 0.852 with perfect information). A whole-horizon plan that respects the rule executes better than re-planning a 24-week window each week with this heuristic. The cause is not measured (§6).

**The submission candidate** `agents/twopass` is `agents/mine` (SciPy dual simplex, the Phase 1 blocks and
`params.json`) with `frame.Planner(base_first="fix")`, two passes:
- **Dev.** `sbf compare twopass mine --task=small`: **0.7400 against 0.7268, paired +0.0132 (+0.0083 to +0.0191)**, A above B on 100 % of resampled episode sets.
- **Why it gains less than the research planner (+0.0187).** SciPy re-solves pass 2 cold rather than warm, so on a degenerate LP it can land on a different optimal plan. The blocks also differ from the bare package. Neither cause is measured.
- **CPU guard.** Pass 2 gets HiGHS's time limit of 1.0 s (Small) or 2.5 s (Full) minus the week's CPU so far. It is skipped below 0.05 s, and pass 1's plan stands. The guard can only change a plan when a week runs long, so scores reproduce except in such weeks.

### 5.6 Warm-started HiGHS and a 48-week window (`agents/twopass48`, `scripts/research/agent_sweep.py`)

**The engine.** SciPy 1.18.1, the server's version, ships HiGHS's own bindings (`scipy.optimize._highspy._core._Highs`,
with `passModel`, `setBasis`, `getBasis`). `agents/twopass48/frame.py` (`WarmLP`) runs the vendored planner's LP on
them, warm-started from last week's basis, shifted one week.

| Window | Cold `linprog` per week | Warm per week (after week 1) | Optimum vs cold |
| --- | ---: | ---: | --- |
| Small, own window (24 weeks, 10k columns) | 0.11–0.12 s | 0.04 s | same (relative gap ~1e-15) |
| Full, own window (25 weeks, 27.8k columns) | 0.57–0.64 s | 0.13–0.18 s | same |
| Full, 48 weeks (55.5k columns) | 2.9–3.8 s | 0.36–0.53 s | same |

**The week, as built:**
1. **Own window.** The planner's 24/25-week window, two passes, under deadlines of 0.8/1.0 s (Small) and 1.6/2.0 s (Full). This plan exists every week.
2. **48-week window.** Two passes with the CPU left, under deadlines of 1.5/1.7 s and 3.1/3.4 s. Its plan replaces the first when pass 1 finishes. If it times out, pass 1's plan of the long window stands, or the own window's plan if pass 1 timed out too.
3. **Warm starts.**
   - Each window keeps its own warm-start chain. The long chain is seeded once, in week 2, from the short basis (about 2.5 s on Full, once per episode).
   - Pass 2 starts from pass 1's basis, but the chain keeps pass 1's. Starting next week's pass 1 from a pass-2 basis doubled its time.
   - Deadlines bound every week. Timing decides only which plan a long-running week plays.

**An earlier design that failed.** That build tried the long window first, kept one shared chain, and fell back to the
own window. On Full it played the long window in only **0–29 of 104 weeks**, and some weeks ended with no plan at all
(+0.0048, interval −0.0019 to +0.0124).

**Results, paired, 0 weeks over the CPU budget in every row** (`agent_sweep_small.json`, `agent_sweep_full.json`):

| Network | Agent | RSS | L1 | L2 | L3 | L4 | vs `twopass` (90 %) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Small dev (20) | `twopass` | 0.7464 | 0.733 | 0.763 | 0.744 | 0.764 | — |
| Small dev (20) | warm engine, own window only | 0.7478 | 0.735 | 0.765 | 0.743 | 0.765 | +0.0014 (−0.0003 to +0.0035) |
| Small dev (20) | **`twopass48`** | **0.7642** | 0.757 | 0.780 | 0.749 | 0.782 | **+0.0178 (+0.0096 to +0.0268)** |
| Small dev (20) | `twopass48`, one pass | 0.7379 | 0.719 | 0.761 | 0.735 | 0.770 | −0.0084 (−0.0179 to +0.0009) |
| Full (8 private) | `twopass` | 0.3414 | 0.445 | 0.240 | 0.135 | 0.484 | — |
| Full (8 private) | warm engine, own window only | 0.3433 | 0.445 | 0.238 | 0.151 | 0.486 | +0.0038 (−0.0007 to +0.0089) † |
| Full (8 private) | **`twopass48`** | **0.4170** | 0.506 | 0.340 | 0.229 | 0.499 | **+0.0756 (+0.0632 to +0.0908)** |
| Full (8 private) | `twopass48`, one pass | 0.3762 | 0.432 | 0.336 | 0.220 | 0.498 | +0.0348 (+0.0216 to +0.0520) |

† From the earlier sweep, against `twopass` at 0.3396 there.

`twopass` is itself time-guarded, so its Full score moved by 0.0018 between two sweeps of the same episodes.

**The second pass at 48 weeks**, paired: +0.0263 (+0.0154 to +0.0383) on Small, +0.0408 (+0.0353 to +0.0459) on Full.
On both networks the second pass and the longer window add up.

### 5.7 A calibrated strait belief (`agents/belief48`, `scripts/research/belief_calibrate.py`)

**Calibration.** Episodes 0–255 of private root 20261004 (never the dev split), fitted on the even episodes and scored
on the odd ones (`belief_params.json`):

| Quantity | Fitted value |
| --- | --- |
| Closure runs (fit half) | 230, 77 censored at the episode's end |
| Duration mixture $P(L\ge n)=\pi_0(1-q)^{n-1}+(1-\pi_0)(1-q_2)^{n-1}$ | $\pi_0 = 0.53$, $q = 0.59$, $q_2 = 0.010$ |
| Mean open fraction while closed | 0.16 |
| Onset within 1 / 2 / 4 / 8 / 12 weeks, live mid_threat aged 0–1 weeks | 0.29 / 0.38 / 0.54 / 0.58 / 0.60 |
| … aged 2–3 weeks | 0.19 / 0.35 / 0.38 / 0.46 / 0.54 |
| … aged 4–7 weeks | 0.02 / 0.04 / 0.08 / 0.14 / 0.22 |
| … aged 8+ weeks | 0.02 / 0.03 / 0.07 / 0.14 / 0.18 |
| … no mid_threat on the strait | 0.003 / 0.006 / 0.013 / 0.026 / 0.039 |

**What the data say against two common assumptions:**
- **The transient share is about 0.53, not 0.72.**
- **Fresh threats act fast, not after a 2–4-week lag.** A mid_threat aged 0–1 weeks is followed by a closure within 1 week 29 % of the time.

Mid_threat threads name the strait's **node index**, not its position in `graph_now.open`; map them through `State.strait_pos`.

**Forecast quality.** Held out, the belief's squared error on $o_{c,t+h}$ is **16–22 % below persistence** at every
horizon $h$ = 1 to 12 (h = 4: 0.0079 against 0.0104).

**Decisions** (paired):

| Network | Agent | RSS | vs `twopass48` (90 %) |
| --- | --- | ---: | --- |
| Small dev | `belief48` (recovery and threats) | 0.7606 | −0.0036 (−0.0101 to +0.0016) |
| Small dev | recovery only | 0.7604 | −0.0037 (−0.0102 to +0.0014) |
| Small dev | threats only | 0.7644 | +0.0002 (+0.0000 to +0.0005) |
| Full (8 private) | `belief48` | 0.4078 | −0.0092 (−0.0305 to +0.0167) |

**Reading:**
- **A better forecast did not give better decisions.** The loss sits in level 1, and it comes from the recovery part.
- **Why: planning on the mean of a bimodal outcome.** A closed strait reopens fast or stays shut for months, and the LP plans on the mean ("half open"). That routes cargo into capacity that will be either all there or not there at all.
- **Threats are too rare to matter** (about 0.4 live threads per episode).
- **Routing is not where the money is** (`route_churn_small.json`). Detour costs (freight, war risk, tariff, strait queues) are $6–11B per Small episode, against $2,300–3,200B of shed and shortage. In level 1 the belief cut routing cost by $0.26B but raised week-to-week churn from 0.43 to 0.46, and shed plus shortage by $8B.

### 5.8 Observation audit: the deterministic fields are already in the LP

Checked against the vendored planner (`agents/twopass48/sbfplan/policies/lp_common.py`) and `frame.wire_from_dict`:

| Field | Where it enters the planner |
| --- | --- |
| `pending_prohibitions.{edge, k, effective_week}` | `persistence_arrays`: `prohibited = rep(...) \| pending_mask(memory, t, H, ...)`, switched on from the stated week. Our agents soften it to a 67 % capacity cut (`params.json` `pending_trust` 0.67) |
| `graph_now.{u, c, open, kappa, supply, tariff, war_risk, prohibited, fab.*, osat.*, grid.*}` | `persistence_arrays`: the window's capacity, unit cost, open fractions, throughput, supply, tariffs, war-risk class (with its transit surcharge and strait queue holding), prohibitions, plant and grid factors |
| `wip.*`, `pipeline.*`, `queue_lots.qty` | `rolled_lp`'s initial state (`lp_common.py:326-361`): arrivals by week, mass-balanced |
| `override_qty`, `release_mode` | The LP's release columns for queued tanker cargo, turned into overrides and holds by `week1_action` and `frame.to_dict_action` |
| End-of-window value | `build_lp`'s terminal credit in the window's last week: in-transit flows, stock, strait queues, fab and OSAT work in process |
| `closure_end.*` | **Never observed in the scored regime**: `chi = False` (`theta.py:59, 206`); 0 entries over 416 dev weeks, 368 of them with a strait closed |

Never read: `closure_end` (always empty), the warning `dyads`, and last week's realised outcomes (`last_week.*`).

**Clamping pending prohibitions to 0** (`pending_trust` 1.0 against 0.67) changes nothing on Small dev: 0.7642 against
0.7642, paired 0.0000 (`pending_trust_small.json`).

**Where `twopass48`'s cost sits** ($B per Small dev episode):

| Component | $B | Share |
| --- | ---: | ---: |
| Shed | 1,734.9 | 62.3 % |
| Shortage | 1,023.2 | 36.8 % |
| Holding | 6.4 | |
| Tariff | 6.2 | |
| Disposal | 1.7 | |
| Freight | 1.2 | |
| Queue holding | 0.11 | |
| War risk | 0.07 | |

Holding through war risk together are under 1 %.

**What that bounds:**
- The attainable saving is about $930B per episode, so removing every tariff and queue-holding dollar would add at most +0.007 RSS.
- Against the clairvoyant LP's plan (shortage about $827B, shed about $1,714B per episode), `twopass48`'s remaining gap is **chip shortage, about $196B** per episode, then shed, about $21B.

### 5.9 Probing the rolling planner at 48 weeks (`scripts/research/window_probe.py`)

**End of the window.** On Small the 48-week window reaches the episode's last week from week 5 on (H = 48, 38 and 28
at weeks 5, 15 and 25). Its end-of-horizon credit is therefore the scorer's own salvage at T.
- **What the plans do at the end.** Lot starts and wafer deliveries go to 0 in the window's last 4 weeks (fabs take 6–8 weeks). Fuel deliveries fall to 40–49 % of mid-window, and stock value to 29–42 %.
- **The scorer credits little.** Leftovers are worth 1.5–3.6 % of their value for fuel and about 0 for wafers and chips.
- **So the run-down is optimal under the scored objective.** An added terminal credit would buy stock the scorer does not pay for (`window_probe_small.json`).
  This holds only for a window that reaches T. On Full, the window ends before T until week 57, and there a credit is worth +0.25 (§5.15).
- **Small's horizon is saturated:** `twopass48` plans to the episode's end from week 5. On Full (T = 104), the window ends before T until week 57.

**What pass 2 leaves.**
- The leftover violations are grid-weeks pass 2 did not bound: **0 of them sit on a bounded grid-week**, over 9 sampled windows.
- They are small: shed 0.01–12 % of base load, fab energy 0.01–6 %.
- They are re-optimisation, not degeneracy. A third pass bounds them (§5.5: +0.002).

**Information against formulation, at 48 weeks with two passes** (Small dev, package solver; `two_pass_small_h48.json`):

| Planner | RSS | L1 | L2 | L3 | L4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence, one pass | 0.7375 | 0.719 | 0.760 | 0.734 | 0.770 |
| **Persistence, two passes** (≈ `twopass48`, 0.7642) | **0.7666** | 0.762 | 0.780 | 0.749 | 0.784 |
| Perfect disruption information, one pass | 0.8051 | 0.803 | 0.817 | 0.780 | 0.836 |
| **Perfect disruption information, two passes** | **0.8340** | 0.841 | 0.844 | 0.794 | 0.849 |

Information is worth **+0.067** at full horizon; two passes add +0.029 with or without it.

**Where the rest goes.**
- With every future disruption known, the rolling planner reaches 0.834.
- The clairvoyant plan that respects base-load-first replays at 0.852, and scores 0.958 in the LP's own model.
- The 0.834 → 0.958 gap is the LP's other relaxed rules and closed-loop re-planning. Any forecast work is capped at the +0.067 above.

**CPU** (`sbf check twopass48 --task=small --episodes=20`, both windows, two passes each):
- Median week per episode: 0.11–0.23 s.
- Worst week per episode: 0.26–0.64 s (8 of 20 episodes have a week over 0.40 s).
- Budget: 2 s.

### 5.10 The production rules: what breaks a plan in execution, and whether our plans break them

**The simulator's week-7 rule, exactly** (`sim.py:322-354`, `production.py:24-27`). For fab $f$ on grid $g$, with $W_f$ the wafers on hand:

$$\hat p_f=\min(\bar\alpha_fR_f\,\mathrm{cap}^0_f,\;W_f),\qquad p_f=\rho_g\,\hat p_f,\qquad \rho_g=\min\Big(1,\frac{G^{av}_g-y_g}{\sum_{f'\in g}e_{f'}\hat p_{f'}/R_{f'}}\Big).$$

Fabs share the grid's leftover energy in proportion to their **requested draw**, not `alpha_bar`. Push (rule 1) and
the pro-rata split (rule 2) are therefore one coupled rule. An exact MILP of it needs a product of two decision
variables and a min() per fab-week: about 900 binaries on Small.

**Attribution in the open-loop replay of the clairvoyant base-load-first MILP plan** (plan 0.9587, executed 0.8573;
`production_gap_small.json`):

| Mechanism | Lots | Share of the shortfall |
| --- | ---: | ---: |
| Planned 154.4M lots, executed 134.0M | | |
| Shortfall weeks: the grid's fabs got less energy in total | 41.7M | 72 % |
| Shortfall weeks: fewer wafers on hand than planned | 16.1M | 28 % |
| Shortfall weeks: the pro-rata split | 0.43M | **1 %** |
| Weeks starting more than planned (push) | 37.8M | |

OSAT packaging is off both ways: 97M short, 63M over. Shortage is +$90B and shed +$11B per episode against the plan.

**The plan held wafers; the simulator started them**, burning fuel and wafers early and leaving later weeks short.
That plan comes from the reference LP, which has none of the planning rules.

**Our planner's own window plans** (mpc_det, two passes, 48 weeks; 12 windows of 4 dev episodes; `rule_probe_small.json`):

| Rule | Plan breaks it in weeks 2..H |
| --- | --- |
| Push: wafers held with spare capacity and energy | **0** fab-weeks |
| Pro-rata split: unequal start ratios on one grid | **0** grid-weeks |
| OSAT: raw chips held below throughput | 4 plant-weeks, **0.2 %** of packaging |

The planning rules' prices on held wafers and raw chips, charged in every week, already make the window LP plan the
way the simulator executes. **Adding the three rules as constraints would not change our plans.**

### 5.11 The exact base-load-first rule as a window MILP (`agents/milp48`)

**Offline, 8 dev episodes (2 per level), 48-week window, a 10 s MILP per week** (`milp_rolling_oracle_8.json`,
`milp_rolling_det_8.json`). The MILP is the window LP plus `base_first_milp.py`'s binaries:

| Forecast | As shipped | Two-pass "fix" | **Window MILP** | MILP vs as shipped (90 %) |
| --- | ---: | ---: | ---: | --- |
| Perfect disruption information | 0.8135 | 0.8335 | **0.9005** | +0.0869 (+0.0592 to +0.1176) |
| Persistence | 0.7452 | 0.7688 | **0.8324** | +0.0872 (+0.0596 to +0.1156) |

- **The gain comes from enforcing the rule exactly, not from information.** Two-pass's bounds are fixed from pass 1's plan, so they cannot reach plans that serve base load in full *and* run the fabs. The MILP's plans start 1.5× the lots of the shipped planner.
- **The linear relaxation of the MILP does not help** (`milp_rolling_det_cut_8.json`). Its rows are $\mathrm{ysh}/\bar y + \sum E/M \le 1$; added alone they score −0.008, with two-pass +0.018 against "fix" alone +0.024, because the big-M rows hardly bind.

**Under a CPU budget** (persistence, same 8 episodes; `milp_budget_*.json`):

| MILP per week | RSS | vs as shipped (90 %) |
| --- | ---: | --- |
| 1 s, every week of the window | 0.8048 | +0.0596 (+0.0444 to +0.0766) |
| 1 s, binaries for the first 12 weeks | 0.8070 | +0.0618 (+0.0537 to +0.0703) |
| 0.5 s, the first 8 weeks | 0.7912 | +0.0460 (+0.0432 to +0.0490) |

**Engineering the MILP into a 2 s week** (SciPy's bundled HiGHS, `frame.window_mip`). Each measured on Small windows:
- **One thread.** HiGHS sizes its thread pool at a process's first solve. The agent resets the pool (`resetGlobalScheduler`) and pins one thread at import, as the server's single CPU gives; otherwise the MILP's CPU runs at 1.4–2.1× its wall-clock limit.
- **Time limit.** One-thread HiGHS overshoots its limit by a median 0.05–0.08 s at 0.9 s, but by about 0.66 s past 1.1 s. The MILP's deadline is 0.85 s of the week's CPU.
- **A start.** Without one, 0.3–0.6 s often finds no plan (4–5 of 6 windows). Seeded with the two-pass plan's yes/no values (z = 1 where it serves base load in full, else fabs dark, always feasible), HiGHS finds a plan in 6 of 6 windows at 0.3 s, better than two-pass in 5.
- **Carried forward.** Each week's start is last week's MILP choices, shifted by a week, so the search accumulates over the episode.
- **The agent's own CPU.** The agent times itself on its own thread (`time.thread_time`). In local multi-process sweeps the harness's machine-sized HiGHS pool spins into process CPU: up to 0.45 s per week, 30 weeks handed to naive. That artefact is absent on the server's single CPU, and it also explains `twopass48`'s drift between parallel sweeps (0.7585–0.7642). Serial sweeps reproduce exactly.

**Results, Small dev, serial, paired against `twopass48`** (`agent_sweep_milp48_small.json`):

| Agent | RSS | L1 | L2 | L3 | L4 | vs `twopass48` (90 %) | MILP plan played |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `twopass48` | 0.7642 | 0.757 | 0.780 | 0.749 | 0.782 | — | — |
| `milp48`, no start | 0.7623 | 0.752 | 0.777 | 0.756 | 0.778 | −0.0020 (−0.0075 to +0.0027) | 11 episodes with 0 weeks |
| `milp48`, two-pass start | 0.7727 | 0.761 | 0.792 | 0.763 | 0.784 | +0.0085 (−0.0030 to +0.0231) | 765 of 766 weeks |
| **`milp48`, start carried forward** | **0.7832** | 0.769 | 0.804 | 0.779 | 0.785 | **+0.0190 (+0.0060 to +0.0359)** | 752 of 764 weeks |

**Scoring container** (`sbf check milp48 --docker`, 2 Small episodes): worst week **1.68 s** of 2 s, median 0.93–0.95 s,
0 weeks given to naive.

On Full the MILP is off (`MIP[4.0] = None`): Full has 8 grids and 55.5k-column windows, and is not measured yet.
Even at 1 s, the budgeted MILP keeps only part of the offline gain: +0.019 in the agent against +0.064 at 10 s.

### 5.12 Tuning the MILP's budget (`agents/milp48t`), and Full

**Timing** (`sbf check`, isolated and serial, 3 Small dev episodes each; local CPU):

| Configuration | Median week | Worst week | Weeks over 2 s |
| --- | --- | ---: | ---: |
| `milp48` (24-week insurance solve, 0.85 s MILP deadline) | 0.79–1.05 s | 1.53 s | 0 |
| No insurance solve | 0.79–1.04 s | 1.97 s | 0 |
| No insurance, deadline 1.05 s | 1.18–1.21 s | 2.24 s | **8** |
| No insurance, 1.05 s, light heuristics (RINS, RENS, symmetry off), gap 0.01 | 0.45–1.21 s | 2.31 s | 4 |
| No insurance, 1.05 s, interrupt | 1.08–1.14 s | 1.90 s | 0 |
| No insurance, 1.3 s, interrupt | 1.34–1.37 s | 2.14 s | 11 |
| **No insurance, 1.05 s, interrupt, light** | 1.08–1.10 s | **1.80 s** | 0 |
| Same, presolve off | 1.09–1.11 s | 1.85 s | 0 |

- **The time limit is not respected.** One-thread HiGHS overruns its own limit in phases that do not check the clock, and RINS/RENS are not the cause.
- **The interrupt.** `frame._run_interruptible` runs the solve on a worker thread and stops it from HiGHS's `kCallbackMipInterrupt` and `kCallbackSimplexInterrupt` callbacks at the CPU deadline. The raw binding cannot call back into Python on the caller's thread: run there, it hangs. With the interrupt, the tail shrinks but does not vanish (callbacks are sparse in some phases).

**Score** (serial, paired, 20 Small dev episodes; `agent_sweep_milp48t_small.json`, `agent_sweep_milp48_small.json`):

| Agent | RSS | L1 | L2 | L3 | L4 | vs `twopass48` 0.7642 (90 %) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `milp48`, run 1 | 0.7832 | 0.769 | 0.804 | 0.779 | 0.785 | +0.0190 (+0.0060 to +0.0359) |
| `milp48`, run 2 | 0.7753 | 0.769 | 0.790 | 0.762 | 0.783 | +0.0111 (−0.0001 to +0.0250) |
| T1: no insurance, 1.05 s, interrupt, light | 0.7726 | 0.760 | 0.800 | 0.751 | 0.782 | +0.0084 (+0.0004 to +0.0185) |
| **`milp48t`** = T1 + gap 0.01 + sparse binaries | **0.7827** | 0.759 | 0.807 | 0.796 | 0.781 | **+0.0185 (+0.0062 to +0.0344)** |

**Reading:**
- **The MILP agents are not reproducible.** A clock decides how far each week's search gets, so the same agent scores differently between runs, even serially: `milp48` scored 0.7832 and 0.7753. Treat about ±0.008 as noise between runs of one MILP agent; `twopass48` reproduces exactly.
- **Run 2 against the tuned versions:** `milp48t` +0.0074 (+0.0006 to +0.0147); T1 −0.0027.
- **More MILP time bought nothing measurable** (T1). The tuned agent's edge comes from gap 0.01 and sparse binaries, which leave a smaller search.
- **Of the offline +0.064 over two-pass (10 s per week), the online agents recover about +0.011 to +0.019 over `twopass48`**, a quarter of it.

**Scoring container** (`sbf check milp48t --docker`):

| Network | Episodes | Worst week | Median | Budget | Weeks to naive |
| --- | ---: | ---: | --- | ---: | ---: |
| Small | 2 | **1.56 s** | 0.31–0.78 s | 2 s | 0 |
| Full (MILP off) | 1 | 2.86 s | 1.50 s | 4 s | 0 |

**Full: a window MILP does not fit** (`full_mip_probe`, private root 20261006, 2 worst grids, weeks 2–12, 6 s limit).
- On the 48-week window, the MILP's root LP alone takes 1.7–3.2 s, and the MILP 5.4–7.6 s.
- On the 25-week window it takes 0.7–6.3 s, and its plan would replace the 48-week plan, which is worth +0.076.
- HiGHS solves a MIP's root LP from scratch, so warm starts do not help.

### 5.13 Gating the MILP, and deterministic stops

**How often the MILP runs, by harm level** (`milp48t`'s counters, 20 Small dev episodes):

| Level | Weeks the MILP ran |
| --- | --- |
| 1 | 192 of 255 (75 %) |
| 2 | 193 of 255 (76 %) |
| 3 | 191 of 255 (75 %) |
| 4 | 192 of 255 (75 %) |

- **Gating on calm weeks would change nothing.** The MILP already runs only when pass 1 sheds base load while fabs draw energy, and the persistence window has such conflicts in three weeks of four, in calm episodes as much as in crises. A "zero shedding" fast path would almost never trigger.
- **Level 1 is not a collapse** in these runs: `milp48t` 0.759 against L4 0.781; `twopass48` 0.757 against 0.782.
- **"Unconstrained" plans are one-pass plans,** which measured worse in every level (§5.9: L1 0.719 against 0.762 with two passes).

**Where the MILP's time goes** (one Small episode, 39 MILPs at the 1.05 s deadline):
- **Branching is almost nil:** median **0** branch-and-bound nodes, maximum 23.
- **The root node takes the time:** its LP, cuts and heuristics. A node limit cannot bound that.
- **Root-only without a clock is too slow:** `mip_max_nodes = 1` took weeks of up to 2.4 s.
- **Iterations track CPU closely:** 0.26 s per 1,000 simplex iterations at the median, 0.35 s at worst.

**A deterministic stop.** `frame._run_interruptible(iters=…)` interrupts at a simplex-iteration count (HiGHS on one thread runs the same way every time), with the CPU deadline kept as a backstop. Two consecutive serial sweeps (`agent_sweep_milp48t_iters_run1.json`, `_run2.json`):

| Agent | Run 1 | Run 2 | vs `twopass48` 0.7642 (run 2, 90 %) | Episodes that differ | Weeks over budget | Worst week |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `twopass48` | 0.7642 | 0.7642 | — | 0 of 20 | 0 | — |
| 2,500 iterations | 0.7678 | 0.7688 | +0.0046 (−0.0000 to +0.0096) | 1 of 20 (0.4 %) | 0 | 0.88 s, 3 episodes |
| 5,000 iterations | 0.7784 | 0.7808 | +0.0166 (+0.0086 to +0.0257) | 4 of 20 (up to 2 %) | 5, then 2 | — |
| Clock, 1.05 s (`milp48t`, §5.12) | 0.7827 | — | +0.0185 (+0.0062 to +0.0344) | about ±0.008 between runs | 0 | 1.65 s (container) |

- **Determinism costs about 0.01 to 0.015 RSS** against the clock-limited agent.
- **At 2,500 iterations the MILP's plan was played in only about half the weeks it ran**, because the cap stops it before it completes its start.
- **At 5,000** the clock backstop fires in some weeks, so the agent is neither deterministic nor within budget.
- `milp48t` keeps the clock (`MIP_ITERS = 0`) for the best expected score. Set `MIP_ITERS = 2500` for reproducible experiments.

### 5.14 A shed penalty, and scenario planning at 48 weeks

**Shed base load × 1000 in weeks 2..H** (`agents/scen48_penalized`: `twopass48` with `LPEdits.scale_cost("ysh", 1000,
start=1)`, no MILP; `agent_sweep_scen48_penalized_small.json`):

| Agent | RSS | L1 | L2 | L3 | L4 | vs `twopass48` 0.7642 (stored, 90 %) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `scen48_penalized` | 0.7044 | 0.683 | 0.720 | 0.717 | 0.754 | **−0.0598 (−0.0835 to −0.0408)** |

As with the planning-rule price in every week (§3.3, −0.017), the penalty charges shed that no plan can avoid as
heavily as the diversions it targets. That distorts the whole plan, and it costs in every level. The rule needs the
exact form (two passes or the MILP), not a price.

**The package's scenario planner, 48-week window** (`oracle_window.py`, variants `scen_H48`, `scenfix_H48`; package
solver; `scen_h48_small.json`, `scenfix_h48_small.json`). `mpc_scen` plans on S = 3 scenarios of the public generator:
- each observed impairment ends after a residual duration drawn from the generator's duration law, conditioned on its age;
- future onsets come from one generator draw per scenario;
- one LP shares the week-1 decisions across the scenarios.

`scenfix` adds the two-pass solve, with each scenario block bounded by "fix" from its own pass-1 plan.

| Planner | RSS | L1 | L2 | L3 | L4 | s/week median / max | Paired gap (90 %) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Scenarios, one pass | 0.7514 | 0.729 | 0.785 | 0.737 | 0.785 | 0.39 / 7.9 | vs persistence one pass 0.7375: +0.0139 (+0.0031 to +0.0279) |
| Scenarios, two passes | 0.7684 | 0.755 | 0.794 | 0.749 | 0.798 | 0.57 / 13.0 | vs persistence two passes 0.7666: **+0.0018 (−0.0118 to +0.0179)** |

- **Scenarios and two passes do not add up.** On one pass, scenarios gain +0.014; on top of two passes they add nothing measurable. The scenarios mostly correct the same plan errors two-pass does.
- **They are not deployable as they stand:** 14 of 20 episodes have a week over 2 s even on the package's warm solver (worst 13 s).
- **Of the +0.067 that perfect information is worth at this horizon (§5.9), the generator's scenarios capture about none once the rule is enforced.**

### 5.15 A conditional terminal credit on Full (`agents/twopass48_credit`, `agents/milp48_eco`)

**The defect.** On Full the 48-week window ends before the episode does until week 57. The LP values what is left in
the window's last week at the scorer's salvage, a few percent of $v_k$. So every week's plan runs its stocks and
pipelines down toward a cliff that, in the episode, is not there.

**The credit** (`frame.LPEdits.terminal_credit`, `Planner._terminal_topup`):
- **What is credited:** the window's last-week stock (I) and strait queues (Q), and shipments (x) still travelling past the window's end.
- **How much:** each is credited at frac × $v_k$, never below its salvage.
- **When:** only when the window ends before T, so the last windows keep the scorer's own salvage (§5.9).
- **Cost:** none. It changes only the objective's coefficients, so the warm-started basis stays valid.

**Score** (Full, private root 20261006, 8 episodes, 2 per level; serial, under the 4 s budget, 0 weeks over budget or
to naive; `twopass48_credit_full.json`, `eco_full_credit.json`, `eco_full_nocredit.json`; `twopass48` stored, not
re-run):

| Agent | RSS | L1 | L2 | L3 | L4 | vs `twopass48` 0.4170 (90 %) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `twopass48` (stored) | 0.4170 | 0.506 | 0.340 | 0.229 | 0.499 | — |
| `milp48_eco` = `milp48t` frame, no credit | 0.3523 | 0.456 | 0.238 | 0.168 | 0.512 | −0.0647 (−0.0835 to −0.0488) |
| `milp48_eco`, credit 0.5 | 0.6650 | 0.739 | 0.577 | 0.606 | 0.552 | +0.2481 (+0.2246 to +0.2727) |
| `twopass48_credit`, credit 0.3 | 0.6653 | 0.730 | 0.586 | 0.625 | 0.540 | +0.2484 (+0.2224 to +0.2763) |
| `twopass48_credit`, credit 0.5 | 0.6614 | 0.734 | 0.589 | 0.577 | 0.543 | +0.2444 (+0.2188 to +0.2700) |
| **`twopass48_credit`, credit 0.7** (shipped) | **0.6701** | 0.735 | 0.588 | 0.635 | 0.543 | **+0.2531 (+0.2289 to +0.2787)** |
| `twopass48_credit`, credit 1.0 | 0.6610 | 0.732 | 0.588 | 0.583 | 0.545 | +0.2441 (+0.2187 to +0.2715) |

**Reading:**
- **The credit is the largest single gain measured on Full**, in every level. On the `milp48t` frame it is worth +0.3127 (+0.2880 to +0.3429) against the same frame without it.
- **The fraction hardly matters.** Paired against 0.3: 0.5 −0.0039 (−0.0143 to +0.0046), 0.7 +0.0047 (+0.0018 to +0.0071), 1.0 −0.0043 (−0.0123 to +0.0030). On 7 of the 8 episodes, 0.5, 0.7 and 1.0 cost the same within 0.2 %, and 0.3 within 1 %.
- **One episode decides the order, and it is a timing effect.** On episode 6 (L3), 0.5 and 1.0 cost 5 % more than 0.7. In that episode's runs at 0.3 and 0.7, the 48-week window hit its CPU deadline in 36–39 weeks and the 25-week plan was played instead. With 0.5 and 1.0 it hit the deadline in 2–3 weeks. So on that episode the short window's plans were cheaper, and which window runs depends on the clock. 0.7 is the middle of a flat range, shipped as such.
- **The frame does not matter once the credit is in:** `twopass48_credit` 0.7 against `milp48_eco` 0.5: +0.0050 (−0.0008 to +0.0101). Without the credit, the `milp48t` frame on Full (MILP off there) is −0.065 against `twopass48`. That regression is not explained, and the credit removes it.
- **Small is unchanged:** the 48-week window reaches T from week 5, and the credit applies only on Full (`s.T > 52`).

**Container** (`sbf check twopass48_credit --task=full --docker`, 1 episode): ready in 0.7 s, week 1 0.95 s, median
1.27 s, worst **2.69 s** of 4 s, 0 weeks to naive. The isolated local run: median 1.20 s, worst 2.65 s.

### 5.16 Ablation of `mpc_nobuf` on Small (branch `mpc-evolve`, `agents/mpc_nobuf`)

**The agent.** `mpc_nobuf` is a compact 16-week LP written from scratch, not the package's planner. It scores
**0.8214** on Small dev, +0.039 over `milp48t`. Its parts:
- base-load-first as binaries $z$ in the first 4 weeks (`milp_weeks`, 0.8 s MILP limit), with a tight big-M ($\sum_f e_f\,\mathrm{cap}_f$) and $z$ continuous after;
- grid and fab capacity recovering toward nominal (`tau_grid` 6, `tau_fab` 12 weeks);
- fuel safety stocks at half of storage, priced at 5 % of $v_k$;
- `short_price` 10, `terminal_frac` 0.5, tanker control.

**Method.** One change per run, through `params.json` only (`agent.py` unchanged). The 20 Small dev episodes, serial,
under the 2 s budget (`scripts/research/play_folder.py`; `mpc_nobuf_ablation_small.json`). Her agent is
clock-limited, so it was run twice unchanged: run 1 scored 0.8214 and run 2 0.8130, −0.0084 (−0.0208 to +0.0047).
Each change is paired against the mean of the two runs per episode.

| Change (`params.json`) | RSS | L1 | L2 | L3 | L4 | vs mean of both runs (90 %) | s/week |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| None, run 1 / run 2 | 0.8214 / 0.8130 | 0.825 / 0.810 | 0.816 / 0.813 | 0.833 / 0.831 | 0.767 / 0.767 | — | 0.33 / 0.25 |
| (1) Binaries relaxed, $z \in [0, 1]$ everywhere (`milp_weeks` 0) | 0.8225 | 0.838 | 0.812 | 0.818 | 0.761 | +0.0053 (−0.0041 to +0.0163) | **0.07** |
| (1b) Rule off (`fab_threshold` 0) | 0.8062 | 0.797 | 0.816 | 0.820 | 0.771 | −0.0110 (−0.0397 to +0.0104) | 0.07 |
| (2) No recovery forecasts (`tau_grid`, `tau_fab` 0) | 0.8166 | 0.812 | 0.820 | 0.832 | 0.770 | −0.0006 (−0.0064 to +0.0058) | 0.24 |
| (3) No safety stocks (`safety_frac` 0) | 0.8048 | 0.812 | 0.797 | 0.808 | 0.772 | **−0.0124 (−0.0225 to −0.0026)** | 0.23 |
| (4) `short_price` 1 (from 10) | 0.8173 | 0.820 | 0.811 | 0.835 | 0.759 | +0.0001 (−0.0038 to +0.0042) | 0.24 |
| (5) 24-week window (from 16) | 0.8275 | 0.831 | 0.822 | 0.840 | 0.778 | **+0.0103 (+0.0001 to +0.0224)** | 0.35 |

No run had a week over budget or played by the naive rule. The s/week column is wall time per week averaged over
the episodes (local, serial).

**Reading:**
- **Only the safety stocks are a measured part of her edge (−0.012).** Removing them costs in L2 and L3.
- **The binaries buy nothing measurable.** The relaxed $z$ still imposes shed/$\bar y$ + E/E_max ≤ 1, a strong relaxation with the tight big-M. It scores the same at a quarter of the CPU (0.07 s a week) and is deterministic.
- **The rule itself, relaxed or exact, is worth −0.011 when removed, not significant on 20 episodes** (wide interval). In her LP it matters far less than two-pass does in `mpc_det` (+0.019 to +0.029, §5.5). Her grid model already carries much of it.
- **Recovery forecasts and `short_price` 10 measure zero.** On the package frame, `short_price` 10 measured −0.0142 (−0.0239 to −0.0054) against `milp48t` (`milp48_eco` on Small, `eco_small_shortprice10.json`), so `milp48_eco` ships with 1.
- **A longer window gains (+0.010), as on the package planner (§5.6).** With the binaries relaxed, the CPU freed would pay for it.
- **So her +0.039 over `milp48t` is not in the parts tested here, beyond the safety stocks' 0.012.** It is in the compact formulation itself: her own grid model, fuel segments and tanker control. Those were outside this ablation's five changes.

### 5.17 `compact_hierarchical`: aligning production with the simulator (Fix A) and base load first (Fix B)

**What was tried** (`compact_hier/compact_lp.py`, toggles in `params.json`; the chip-chain diagnostic of
`scripts/research/chip_chain_diagnostic.py` motivated both):
- **Fix A, OSAT side:** week-1 packaging fixed at the simulator's rule; raw chips held at OSATs priced at v_k.
- **Fix A, fab side:** week-1 lot starts split as the simulator does, with base load first in week 1 (a second
  solve); wafers held at fabs priced at v_wafer.
- **Fix B:** base load first in every window week (two passes).

**Score.** Small, root 202610 (the tuning set), 64 episodes, paired against the tuned agent (0.7953), serial, under
the CPU budget (`outputs/compare_*.log`):

| Variant | RSS | Paired gap (90 %) |
| --- | ---: | --- |
| Fix A, all parts | 0.7718 | −0.0235 (−0.0311 to −0.0162) |
| without the wafer price | 0.7858 | −0.0096 (−0.0146 to −0.0048) |
| without the week-1 second solve | 0.7812 | −0.0142 (−0.0205 to −0.0082) |
| without the week-1 fab split (and its second solve) | 0.7845 | −0.0108 (−0.0168 to −0.0051) |
| without the OSAT price | 0.7671 | −0.0283 (−0.0361 to −0.0206) |
| OSAT parts only | 0.7953 | −0.0000 (−0.0040 to +0.0042) |
| Fix B alone | 0.7487 | −0.0466 (−0.0575 to −0.0362) |
| Fix B with the OSAT parts | 0.7524 | −0.0429 (−0.0545 to −0.0315) |
| Fix B with all of Fix A | 0.7355 | −0.0598 (−0.0726 to −0.0471) |

**Reading:**
- **The OSAT side is neutral.** It cut packaged-chip disposal at OSATs by more than half (L1 0.81M → 0.31M per
  episode) and made week-1 packaging exact, but the score did not move.
- **Every change that makes the fab and energy model stricter loses.** In Fix A's diagnostic the plan expected less
  fab output and the agent shipped fewer wafers: L1 wafers lifted 12.22M → 10.68M, lots started 10.17M → 8.83M, fab
  lot-weeks lost for want of wafers 1.29M → 2.30M. Fix B's chain was not traced; the same mechanism is the likely
  one.
- **The relaxed base-load-first rule is load-bearing here.** Its optimism about fab energy, under a persistence
  forecast, keeps wafers at the fabs for the weeks that energy turns out to be there. This agrees with §5.16, where
  binaries and the relaxation scored the same. The chip gap of Gate 0 (89 % of 1 - RSS) is not closed by fidelity on
  the energy side.
- **Both fixes stay in the code behind their toggles, off by default.**

---

## 6. Roadmap

Ordered by measured leverage per unit of CPU.

| # | Direction | Evidence | Expected cost per week |
| --- | --- | --- | --- |
| 1 | ~~Enforce base-load-first in weeks 2..H of the planner's window~~ **done** (two-pass "fix", §5.5) | +0.019 closed loop on Small dev, +0.014 on 8 Full episodes; `agents/twopass` +0.013 | 2 LP solves in 73 % (Small) to 87 % (Full) of weeks |
| 1a | **The exact rule as a window MILP** (§5.11) | Offline +0.064 over two-pass at 10 s; `milp48` **+0.019** (+0.006 to +0.036) at 0.85 s on Small | More MILP time per week is the lever (1 s, multi-threaded: +0.036 offline); Full not built |
| 1b | Close the gap between closed loop (0.804) and the open-loop replay (0.852), both with perfect information | Candidates: the exact MILP in the window with a time cap (option (c) below); a longer window with two passes; the other week-1-only rules | Offline first |
| 2 | Find the remaining rule gap: MILP plan 0.958 in plan space vs 0.852 executed | Same replay method, one rule at a time: push lot starts, pro-rata fab energy, OSAT packaging | Offline only |
| 3 | ~~48-week window on Full~~ **done** (warm-started HiGHS, §5.6) | `twopass48`: +0.076 on 8 Full episodes, +0.018 on Small dev, 0 weeks over budget | No coarsening needed; the long window's pass 2 is the main CPU cost |
| 3a | ~~Terminal credit where the window ends before T~~ **done** (`twopass48_credit`, §5.15) | **+0.253** on 8 Full episodes (0.4170 → 0.6701); the fraction from 0.3 to 1.0 hardly matters | None (objective coefficients only) |
| 3b | Port what `mpc_nobuf`'s ablation found (§5.16) | Safety stocks −0.012 when removed; a 24-week window +0.010; binaries relaxed: same score at a quarter of the CPU | Not measured on the package frame |
| 4 | Calibrated beliefs (messages, pending prohibitions). Scenarios of the public generator (`mpc_scen`) add +0.002 on top of two passes and cost up to 13 s a week (§5.14) | Ceiling +0.057 to +0.075 at fixed horizon (§4.3). A calibrated strait belief planned on its mean measured **−0.004 / −0.009** (§5.7) | The next try must not plan on means: scenarios (closed vs reopened) or a hedge that does not depend on how long a closure lasts |
| 5 | **One calibration upload** of the current agent | The only way to put dev numbers next to the board's 0.88 (§3.5) | One of the day's 3 submissions; the participant decides |

**Direction 1, three options** (the binaries $z_{g,t}$ of §3.3, restricted to the window):

| Option | Size or method | Status |
| --- | --- | --- |
| (a) Exact MILP | 4 × 23 = 92 binaries on Small at H=24; 8 × 23 = 184 on Full (376 at H=48) | The 52-week clairvoyant MILP had not closed a $10^{-4}$ gap after **60 to 120 s** on most episodes (median gap $1.3\times10^{-3}$ at 60 s), so an exact solve per week does not fit the budget without a loose gap, a time cap and an LP fallback |
| (b) Fix and re-solve | Solve the LP, bound the violated grid-weeks (§5.5), solve again | Measured: §5.3, §5.5 |
| (c) MILP with a cap | HiGHS MIP warm-started from (b)'s solution, with a time cap | — |

Option (b) is measured (§5.5): +0.018 for the oracle window, +0.019 for `mpc_det`, both significant. (a) and (c) are
not measured closed loop.

**Not recommended, by measurement:**
- Simulator lookahead scored by the LP's estimate of the remaining cost (§5.4).
- Price surrogates of the rule (§3.3).
- More parameter search on the belief blocks (§4.3).
- The "zero" two-pass mode (§5.5).
- Replacing the LP with a nonlinear or learned planner. The relaxed rule was cheap to encode (§5.5), and the LP is otherwise exact up to the rules of §2.

---

## 7. Glossary

| Term | Meaning |
| --- | --- |
| Persistence forecast | `mpc_det`'s window: observed disruptions last the whole window; pending prohibitions take effect at their stated week |
| Oracle window | `mpc_det` with the true disruption marks of its window; demand stays the forecast |
| Open loop | Executing a plan's week-$t$ flows from a single solve, without re-planning |
| Closed loop | Re-solving every week from the observed state (what an agent does) |
| Planning rules | `build_lp(..., planning_rules=True)`: the simulator's rules restored in the planner's LP; most of them in week 1 only |

## 8. Reproducibility

The scripts are in [`scripts/research/`](../../scripts/research/) and share
[`common.py`](../../scripts/research/common.py): the episodes, the scorer's worlds, the planner variants and the
base-load-first bounds. Run them from the repository root. Each writes `outputs/research/<script>/<date_time>/`, and
each takes `--workers` (3 by default; more than 4 risks running out of memory on 16 GB). The first run on a network
computes and caches the references (`~/.cache/shockbench-flow`).

| Result | Command | Approx. time |
| --- | --- | --- |
| Oracle-window and horizon variants (§4.3, §5.1) | `uv run python scripts/research/oracle_window.py --variants=det_L,oracle_L,oracle_all,det_H48,oracle_H48` | 5 min |
| The same on Full (§5.2) | `uv run python scripts/research/oracle_window.py --task=full --root=20261006 --per_level=2 --variants=det_L,det_H48` | 15 min |
| Base-load price variants (§3.3, §5.1) | `uv run python scripts/research/base_first.py` | 5 min |
| Exact bound (§3.3) | `uv run python scripts/research/base_first_milp.py --time_limit=120` | 15 min |
| Open-loop replays (§5.1) | `uv run python scripts/research/base_first_exec.py --time_limit=60` | 15 min |
| Two-pass closed loop (§5.5) | `uv run python scripts/research/two_pass.py` | 10 min |
| Signal traces (§4.1) | `uv run python scripts/research/signals.py` | 5 min |
| Agent variants, paired, under the CPU budget (§5.6, §5.7) | `uv run python scripts/research/agent_sweep.py --variants="twopass;twopass48;belief48"` (Full: `--task=full --root=20261006 --per_level=2`) | 10 min (Small), 50 min (Full) |
| Strait belief calibration (§5.7) | `uv run python scripts/research/belief_calibrate.py --episodes=256` | 5 min |
| Routing cost and churn (§5.7) | `uv run python scripts/research/route_churn.py --a=belief48 --b=twopass48` | 5 min |
| Lookahead post-mortem (§5.4) | `uv run python scripts/bound_hybrid_limit.py` (`--depth`, `--limit`) | 15 min to hours |
| Terminal credit fractions on Full (§5.15) | `uv run python scripts/research/agent_sweep.py --variants="twopass48_credit:TERMINAL_FRAC_FULL=0.3;twopass48_credit:TERMINAL_FRAC_FULL=0.7" --task=full --root=20261006 --per_level=2 --workers=1` | 20 min per fraction |
| One folder, paired against a stored run (§5.16) | `uv run python scripts/research/play_folder.py --folder=PATH --stored=scripts/research/results/mpc_nobuf_small.json:KEY` (each ablation: `mpc_nobuf` with one key changed in `params.json`) | 1–6 min (Small) |

Results reproduce to the cent on the same machine type, not across CPU types (`docs/GUIDE.md:304-306`). The MILP's
incumbents and bounds also depend on its time limit and on the machine's speed.

# Threat-conditioned hedging for `compact_hierarchical`: formulation, ceilings and roadmap

A design to validate before any code. It covers three requests:
1. Safety floors that scale with threat signals.
2. Routing that acts before a strait closes.
3. The changes to `compact_lp.py` and `safety.py` that would carry them.

Before the formulation, it checks the premise against what this repository has measured, because the premise decides
where the design can pay.

Section numbers like §5.16 refer to [dataset_and_rss_mechanics.md](dataset_and_rss_mechanics.md).

---

## 0. Summary

1. **The static floors are not what L1 loses.**
   - A floor costs only physical holding: the simulator charges no purchase cost for fuel (its cost components are
     freight, war risk, tariff, holding, queue holding, shortage, disposal and shed).
   - At the tuned `safety_frac` 0.59, holding the floors for a whole Small episode costs about **$0.18B**.
   - L1's mean attainable saving is **$820B** per episode (§3.4). Removing every floor in every calm week could
     therefore return about **0.0002 RSS**.
   - Two measurements agree:
     - Removing the floors from `mpc_nobuf` left L1 unchanged (0.812 against 0.810 and 0.825 for two unchanged
       runs) and cost L2 −0.017 and L3 −0.024 (§5.16).
     - Optuna *raised* `safety_frac` from 0.50 to 0.59.
2. **L1's gap is shed and shortage, as everywhere else.**
   - At L1 0.78 the agent leaves about 22 % of $820B (the dev figure), roughly **$180B per episode**, against the clairvoyant plan.
   - Detours (freight, war risk, tariff, queues) total $6–11B per episode, against $2,300–3,200B of shed and
     shortage (§5.7).
   - Calm episodes are not disruption-free: closure onsets average 1.17 per episode, about equal across levels (§4.1).
   - Knowing every disruption in advance is worth **+0.079 in L1** on the package planner at 48 weeks (0.762 → 0.841,
     §5.9). That is the measured ceiling for anything that acts on forecasts.
3. **The version of the idea that can pay is pre-positioning, not lowering calm buffers.**
   - **In scarcity (L2, L3).** A floor's price trades shed now against expected shed later. Pricing it by a calibrated
     hazard instead of a constant is principled, and it targets the levels where the floors were measured to pay.
   - **In L1.** It acts only through better-sized buffers and earlier stocking ahead of credible threats.
   - **The signals cap it.** The only strong strait signal (a military threat on an open strait) precedes about a third
     of closures: 0.4 live threads per episode against 1.17 onsets. `warning.score` has no predictive value
     (AUROC 0.41–0.55).
4. **Expected value: +0.005 to +0.02 (estimate, not measured).** That is not enough alone to go from 0.795 to 0.89.
   Three cheap diagnostics (§5) decide whether to build it and where the rest of L1 sits.
5. **The 0.7953 is in-sample.** It is the best of a search on these 64 episodes, and board scores are not on the same
   footing as local ones (§3.5). Confirm it on root 12345 and dev, and make one calibration upload of the tuned agent,
   before sizing the gap to 0.89.

---

## 1. Why the static floor is cheap, and what it really trades

### 1.1 The constants (Small instance)

| Quantity | Value |
| --- | --- |
| VOLL, every grid | $4.125M per unit of energy (1 fuel unit burnt = 1 unit of energy in the LP's grid model) |
| $v_k$: LNG / crude | $41,253 / $38,456 per unit |
| Holding: LNG / crude | $33.4 / $31.1 per unit-week |
| Floor price today: `safety_price` 0.05 × $v_k$ | $2,063 per unit-week (LNG): 62 × the holding cost |
| Fuel storage under floors (terminals and grids, LNG and crude) | about 172,800 units; at 0.59: about 102,000 units held |
| Their holding over 52 weeks | about $0.18B per episode |

### 1.2 The newsvendor view of one contingency

Take a store $j$ and a contingency $e$ that cuts some of its inbound supply. If $e$ starts in week $h$, the store
faces a supply gap of expected volume $E_{j,e}$. Stock on hand covers $\min(I_{j,h}, E_{j,e})$ of it, and the rest
becomes shortfall worth $c_j$ per unit. With $\lambda_e(h)$ the probability that $e$ starts in week $h$ given what is
observed now, the expected cost of holding $I_{j,h}$ is

$$\lambda_e(h)\,c_j\,\big(E_{j,e}-I_{j,h}\big)^+ \;+\; \eta_j\, I_{j,h},$$

where $\eta_j$ is the marginal cost of holding a unit for a week. Keeping a unit of buffer pays while
$\lambda_e(h)\,c_j > \eta_j$.

- **In a calm week, $\eta_j$ is the holding cost.** The break-even hazard is $33.4 / 4.125\text{M} \approx 8\times10^{-6}$
  per week (with $c_j$ = VOLL). The calibrated hazard of a strait closing with no warning is about
  $3.3\times10^{-3}$ per week (§5.7: 0.013 within 4 weeks), 400 times higher. So in calm weeks the risk-neutral
  answer is to fill the buffer. **The decision that matters is its size $E_{j,e}$, not its price.**
- **In scarcity, $\eta_j$ is the value of burning the unit now:** up to VOLL, when holding it means shedding now.
  The floor's price then decides how much current shed is accepted to guard against future shed. That is a
  probability-weighted trade, and a constant price gets it right only on average.

### 1.3 What today's floor amounts to

- **Implied hazard.** Today's price $0.05\,v_k$ corresponds to $\lambda c = 2{,}063$, i.e. an implied hazard of
  $5.0\times10^{-4}$ per week at $c$ = VOLL: about 7 times below the calibrated no-warning hazard of a single strait.
- **Implied size.** At `grid_tw`, LNG burn is about 2,000 units a week (share 0.4 of a 5,400 deliverable at a 0.93 load factor), and
  0.59 × 8,647 = 5,100 units is about 2.5 weeks of burn. A 3-week re-sourcing gap under the bimodal closure law
  (§2.2) exposes about 2.2 weeks of inflow.

So the tuned static floor is about the risk-neutral buffer for a typical gap. That fits Optuna moving it up, not
down (illustrative arithmetic with an assumed 3-week gap, not computed from the routes).

---

## 2. Formulation: exposure-sized, hazard-priced floors

### 2.1 The LP rows

For each fuel store $j$, each contingency $e \in \mathcal{E}_j$ and each window week $h$, one slack $s_{j,e,h} \ge 0$:

$$I_{j,h} + s_{j,e,h} \;\ge\; \tau_h\,E_{j,e}(h), \qquad \text{cost } \pi_{j,e,h} = \lambda_e(h)\,c_j,$$

with $\tau_h$ the end-of-episode taper (`floor_taper_weeks`, already built).

- **Separate rows, not stacked segments.** Each contingency alone must be covered, and two rarely coincide, so the
  stock needed is the largest exposure, not the sum. The sum of the rows' costs is convex piecewise-linear in
  $I_{j,h}$, so the LP stays an LP.
- **Today's floor is the special case** of one contingency per store with $E = $ `safety_frac` × storage and
  $\pi = $ `safety_price` × $v_k$.
- **$\mathcal{E}_j$ holds:**
  - a base contingency: the unsignalled closure of the strait carrying most of $j$'s inflow, at hazard $\lambda_0$;
  - one contingency per live credible threat on a strait $j$'s inflow crosses;
  - one per pending prohibition that cuts one of $j$'s inbound routes.

### 2.2 Exposure $E_{j,e}$: the re-sourcing gap

For each inbound route $r$ of $j$ cut by $e$:
- $\phi_{j,r}$ is $j$'s weekly inflow on $r$ (the last plan's flows, averaged over the last 4 weeks);
- $L_r$ is the route's lead, and $a_{r,c}$ the time from origin to the strait $c$ that $e$ closes;
- $L^{\text{alt}}_{r}$ is the lead of the fastest route into $j$ not cut by $e$ with spare capacity.

When $c$ closes, cargo past the strait still arrives until $L_r - a_{r,c}$, and new cargo on the alternative arrives
from $L^{\text{alt}}_{r}$. The gap is

$$g_r = \max\big(0,\; L^{\text{alt}}_r - (L_r - a_{r,c})\big) \quad (\text{the remaining horizon if no alternative}).$$

A closure is still on in its $n$-th week with probability (calibrated on 256 private episodes, §5.7)

$$S(n) = \pi_0(1-q)^{n-1} + (1-\pi_0)(1-q_2)^{n-1}, \qquad \pi_0 = 0.53,\; q = 0.59,\; q_2 = 0.010.$$

The expected shortfall volume, capped by storage, is

$$E_{j,e} = \min\Big(\text{storage}_j,\; \sum_{r \text{ cut by } e} \phi_{j,r} \sum_{n=1}^{g_r} S(n)\Big),
\qquad \sum_{n=1}^{g} S(n) = \pi_0\frac{1-(1-q)^g}{q} + (1-\pi_0)\frac{1-(1-q_2)^g}{q_2}.$$

For $g$ = 3 this is 0.84 + 1.40 = 2.2 weeks of inflow. Straits without an alternative (Hormuz for Gulf cargo) have an
exposure capped by storage: a buffer can bridge their short closures and the start of the long ones, nothing more.

### 2.3 Hazard $\lambda_e(h)$: calibrated, never a trigger

From the calibration already fitted (`belief_params.json`, §5.7), the cumulative probability $F_a(\tau)$ that a
closure starts within $\tau$ weeks, given the age $a$ of the live `mid_threat` on that strait:

| Threat age | 1 w | 2 w | 4 w | 8 w | 12 w |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–1 weeks | 0.29 | 0.38 | 0.54 | 0.58 | 0.60 |
| 2–3 weeks | 0.19 | 0.35 | 0.38 | 0.46 | 0.54 |
| 4–7 weeks | 0.02 | 0.04 | 0.08 | 0.14 | 0.22 |
| 8+ weeks | 0.02 | 0.03 | 0.07 | 0.14 | 0.18 |
| No threat | 0.003 | 0.006 | 0.013 | 0.026 | 0.039 |

Interpolate monotonically in $\tau$, then $\lambda_e(h) = F_a(h+1) - F_a(h)$, with the age advancing with $h$.

- **Decoys are inside these numbers.** They were fitted on every threat, real or not. A fresh threat buys at most
  about 60 % of a buffer's value, and an old one decays to the base rate by itself. No threshold has to be tuned.
- **A withdrawal message** (kind 4) returns the strait to the no-threat row.
- **`warning.score` is excluded:** AUROC 0.41–0.55 (§4.1).
- **Pending prohibitions** stay as the LP treats them now: certain from their effect week (§5.8). Their decoy rate is
  measured in diagnostic D4 before any change.

### 2.4 Contingency value $c_j$

- **A grid's fuel store:** $c_j = \kappa\,\text{VOLL}_g$. In the LP's grid model a unit of the fuel short is a unit of
  load shed, since each fuel's segment is capped at its share.
- **A terminal:** the smallest VOLL of the grids it feeds, times $\kappa$.
- **$\kappa \in (0, 1]$** is one knob per network: the share of a supply gap that actually becomes shed after the plan
  re-routes. It is the only number to tune; $\lambda$ and $E$ come from calibration and geometry.

### 2.5 Realized disruptions

A buffer exists to be consumed. Once a contingency has happened, its row for the cut store is removed: the store
draws down, and shed prices the drawdown. The existing crisis rule does the opposite: it raises the floor of a
cut-off store. In the tuned configuration it is already inert (`crisis_frac` 0.32 is below `safety_frac` 0.59, so
the larger floor wins), leaving only its price multiplier. In this design the crisis controller's role becomes the
hazard update; its floor raise is retired.

### 2.6 Extension to wafers and chips

The same rows apply to wafer and chip stores, with $c_j$ the sinks' shortage penalty $\pi$ (and $\kappa$). The Taiwan
strait carries much of that flow. Build this only if D1 (§5) shows shortage, not shed, dominates L1's gap.

---

## 3. Preemptive routing without being faked out

### 3.1 A risk premium on the dispatch, not a forecast of capacity

The calibrated belief lost value (§5.7) because it planned capacity on the *mean* of a bimodal outcome: a
"half-open" strait. The design principle here is to keep capacity at persistence (deterministic) and put the risk
on the decisions it would strand.

A dispatch on route $r$ in window week $h$ reaches strait $c$ in week $h + a_{r,c}$. It finds $c$ closed with
probability

$$\rho_{r}(h) = \sum_{u \le h + a_{r,c}} \lambda_c(u)\,S\big(h + a_{r,c} - u + 1\big).$$

Given that, it waits $W = \pi_0'/q + (1-\pi_0')\min(T - t - h,\; 1/q_2)$ weeks in expectation. Here $\pi_0'$ is the
transient share given the closure is still on at arrival; Bayes on $S$ gives it. Its LP cost becomes

$$\text{cost}(x_{r,h}) \mathrel{+}= \rho_r(h)\, W\, \mu_{d(r)},$$

with $\mu_{d(r)}$ the value per unit-week of earlier delivery at the route's destination. Take it from the previous
week's solve: the dual of that store's balance row, shifted by one week. Tanker releases ($y$) whose path reaches a
threatened strait get the same premium.

**Properties:**
- **Calm destinations shrug off threats.** When the destination has no scarcity its dual is small, the premium is
  small, and the plan keeps the cheap route. Detours (about $6–11B per episode in total, §5.7) are spent only
  where delay costs shed or shortage.
- **A decoy costs only its premium-weighted detour,** in the weeks its calibrated probability is high, and fades with
  its age.
- **Rush, then divert.** The floors of §2 raise the reward for stock at exposed stores while the threatened strait is
  still open, so the plan ships more through it now. The premiums move later shipments, those that would arrive after
  the likely onset, to the alternative. That is what the clairvoyant does, and it comes out of the two pieces without
  a rule.

### 3.2 Optional: a two-scenario contingency LP for credible threats

When a strait's onset probability within the window exceeds $p_{\min}$, the planner can solve instead:

$$\min\;(1-p)\,J_A + p\,J_B$$

over two copies of the window: scenario A (no closure) and scenario B (closure from the median onset week, long
duration). Week-1 flows and releases are shared between the copies.

- **Cost:** twice the LP, only in threat weeks (about 0.4 live threads per episode).
- **Evidence so far:** the package's generic scenarios gained +0.002 over two passes and were too slow (§5.14). This
  is targeted and cheaper, but build it only if §3.1 shows the signal is worth something.

---

## 4. Changes to the code (for validation; not written)

| Module | Change |
| --- | --- |
| `compact_hier/threat.py` (new) | `ThreatModel.update(o, t)`: live `mid_threat` threads per strait (channel 5, target kind 0; the target is the strait's node index), their age, withdrawals. `hazard(c, H)` returns $\lambda_c(0..H-1)$ from the calibration table; `survival(n)` returns $S(n)$. The calibration ships as a JSON beside `agent.py`, fitted with `belief_calibrate.py` on root 20261004 (never dev or a tuning root) |
| `compact_hier/network.py` | Per store and strait: the inbound routes it cuts, $a_{r,c}$, and the fastest alternative into the store (leads recomputed weekly from `graph_now.tau`) |
| `compact_hier/safety.py` | `floors()` returns rows `(j, floor[H], price[H])`, one per contingency, built from §2 ($E$, $\lambda$, $c$, $\kappa$). The crisis floor raise is retired; a realized cut removes its row. `safety_frac` / `safety_price` remain as the fallback when the hazard model is off |
| `compact_hier/compact_lp.py` | The safety block takes per-week floors and prices: `le(-I[h,j] - s[h,i] <= -floor_i[h] * taper[h])`, cost `price_i[h]`; columns stay $H\times$rows. A premium array `prem_x[S, H]` (and `prem_y[NO, H]`) is added to the dispatch costs. The balance rows' duals (`res.eqlin.marginals`) are kept by $(h, j)$ for next week's $\mu$ |
| `compact_hier/maritime.py` | Unchanged for closed straits (the bars stay). Supplies the route geometry to `threat.py` |
| `params.json` | New knobs: `hedge_model` (0 static floors, 1 §2), `kappa_small`, `kappa_full`, `route_premium` (0/1), `premium_cap` (× $v_k$), `scenario_threats` (0/1), `scenario_p_min`. Defaults reproduce today's behaviour (`hedge_model` 0) |

**Cost per week:** a few hundred extra slack columns, an $S\times H$ cost array, and table lookups. It is negligible
next to the 0.07 s solve.

---

## 5. Roadmap with gates

**G0: diagnostics (offline, about a day; decides everything after).**

| # | Question | Measurement | Decision |
| --- | --- | --- | --- |
| D1 | What is L1's gap made of? | The tuned agent, naive and the clairvoyant on the L1 and L2 episodes of the tuning root: cost by component (the eight above), and week by week aligned to disruption onsets | If holding is under 1 % of the gap, the floors are confirmed not to be the bleed. If shortage dominates, §2.6 comes before §2.1–2.5 |
| D2 | How much would perfect information give this agent? | `compact_hierarchical` with the true disruption marks in `forecast.window` (as `oracle_window.py` does for the package planner), by level | The ceiling for §2 and §3 together |
| D3 | How much would perfect hedging give? | Floors sized as §2.2, switched on only in the weeks before a true onset that cuts the store (oracle hazard), against today's static floors | If below +0.005, do not build §2's signal part |
| D4 | How much can the signals see? | The share of onsets, by disruption type, preceded by a usable signal (threat, pending prohibition, message) with a lead at least the re-sourcing gap; the decoy rate of pending prohibitions | Signal coverage × D2 bounds what §2 and §3 can reach |

**Then, in order, each paired against the tuned agent on a fresh root and confirmed on dev:**
1. **P1, exposure-sized floors without signals** (§2.1, 2.2, 2.4; $\lambda = \lambda_0$). Tests whether geometry-sized
   buffers beat a storage share. 2 days.
2. **P2, threat hazards** (§2.3, 2.5). 2 days.
3. **P3, route premiums** (§3.1). 1–2 days.
4. **P4, the two-scenario LP** (§3.2), only if P2 or P3 measure a signal gain. 2–3 days.

**If D2 × D4 caps the predictive layer below +0.01,** L1's gap is not information. The next lever is then the
plan-versus-execution mismatch of this LP: replay its own plans through the simulator, as §5.10 did for the package
planner.

---

## 6. Notes on the current state

- **`short_price` 1.018 sits on the search's lower bound (1.0).** Before calling the search exhausted, widen it (for
  example 0.1–30). Below 1, the LP plans to hold fuel back from its segments; whether that pays is untested.
- **`crisis_frac` 0.32 is below `safety_frac` 0.59,** so the crisis floor raise is inert in the tuned configuration
  (§2.5).
- **0.7953 is the maximum over the search's trials on its own episodes.** Expect less out of sample.

# ShockBench-Flow: mentor sync summary

Status as of 2026-10-05. Scores are RSS (0 = the naive rule, 1 = the clairvoyant plan). Unless noted, they come from the
**Small** network. Gaps between two agents are **paired** (same episodes) with a 90 % bootstrap interval. Episode pools
differ between rows: compare numbers only within one row.

## 1. The core problem: the RSS dilemma

### What the score rewards
- RSS = Σ_s p_s·ḡ_s / Σ_s p_s·D̄_s with p = (0.50, 0.30, 0.15, 0.05). The denominator is fixed, so RSS ranks agents by
  **expected dollars saved against naive**: a dollar lost in a level-1 (calmest) episode costs as much score as
  **10 dollars** gained in a level-4 episode.
- **Shed and shortage are about 99 % of naive's cost in every harm level** (power shed at the grids and unserved chip
  demand). Freight, tariffs, holding, queueing and disposal together are under 1 %.
- Attainable savings are about the same in every level ($0.76T to $1.12T per episode), so the levels' effective
  weights are close to p (44 / 33 / 18 / 4 %). **Ordinary weeks carry most of the score.**

### The baseline: the package's `mpc_det` planner
- A rolling LP: each week it solves the clairvoyant oracle's own model over a **24-week window** (Small) and executes
  week 1. **0.729 on the dev split**, against 0.41 for send-the-maximum.
- **Its forecast is persistence.** Observed disruptions are assumed to last the whole window, and pending prohibitions
  are switched on at their stated week **as certain**. It reads **no warning score and no threat message**.
- It does not stockpile because its forecast never predicts a new disruption, not because of holding costs (under 1 %
  of cost).

### The dilemma in this network
- A precaution here moves scarce supply earlier or elsewhere. When the threat is a decoy, the cost appears as **shed
  and shortage in calm episodes (levels 1-2)**, at the 10:1 exchange rate.
- **Few signals are certain.** Decoy threads carry every message a real thread carries, pending prohibitions and
  final tariff notices included, until withdrawn at their effect week. Decoy shares are 25-54 % by channel. Closure
  end weeks are hidden in the scored regime.

## 2. Phase 0: the LP agent (`agents/mine`)
- **Vendored `mpc_det`** (MIT) into the submission, solved with **SciPy's HiGHS dual simplex**, because the server has
  no `highspy`. A tested adapter rebuilds the planner's input from the agent's observation.
- **Verified equivalent to the package:**
  - same LP optimum week by week (relative gap < 1e-9);
  - dev split **0.7266** vs the package's 0.7294;
  - Full **0.3313** vs 0.3327 on 32 private episodes (paired gap −0.0013, interval −0.0035 to +0.0008).
- **Within the server's limits** in the scoring container: worst week **0.37 s / 2 s** CPU (Small) and
  **1.40 s / 4 s** (Full), no naive weeks, peak memory **489 MB / 4 GB**.
- A frozen frame with four **EVOLVE blocks** (`update_belief`, `forecast`, `objective`, `settings`). A deterministic
  LP-size cap replaces timing-based guards, so scores reproduce.

## 3. Phase 1: probabilistic risk and pre-positioning (in `agent.py`)

### `update_belief` and `forecast`
- **Decoy discounting:** each live thread adds risk to its strait with weight `1 − s.decoy_share(channel)`. Signals
  combine by noisy-OR.
- **EMA** of the noisy `warning.score`, a one-week-lagged reading of the latent hazard (loading 0.604). A smooth
  logistic maps score to probability.
- **Temporal decay** of carried risk, so the belief recovers from false alarms.
- The forecast scales each strait's expected open fraction by `1 − severity_scale × risk`. Pending prohibitions can
  be discounted to an expected capacity cut (`pending_trust`).

### `objective`
- **Stock floors** at the markets (`edits.stock_floor`): weekly demand × `buffer_weeks` × global risk, capped at
  current stock. That cap matters: a floor is a hard constraint, and the LP can meet it by withholding supply.
- **Frame fix:** `s.stock_slot(node, k)` maps a market to the LP's stock column. The LP numbers 84 slots, straits
  included, while the observation lists 59.

### Measured effect against the seed (32 private training episodes)

| Setting | Gap | 90 % interval |
| --- | --- | --- |
| severity 0.2, warning center 1.5 | **−0.0120** | −0.0203 to −0.0056 |
| same, floors off | −0.0115 | −0.0206 to −0.0048 |
| severity 0.05, center 2.5, slope 3 (adopted defaults) | −0.0014 | −0.0033 to +0.0001 |

- Losses sit in **levels 1-2**: the false-alarm cost predicted in section 1. Levels 3-4 gained nothing measurable.
- The **stock floors moved the score by about 0.0005**: the JIT hypothesis is not supported so far.

## 4. Phase 2: hyperparameter optimization (11 parameters)

### Attempt 1: AlphaEvolve loop (`examples/08_evolve.py`)
- LLM mutation of the blocks (Claude), CMA-ES on declared constants, paired and stratified evaluation, a held-out
  promotion gate and a Full gate.
- Offline dry runs completed end to end. A cache-key bug (pools sharing a name) crashed the first full run; it is
  fixed with a regression test.
- The live LLM mutator was stopped by **API billing limits**, so no LLM-written candidate has been evaluated.

### Attempt 2: local Optuna (`scripts/tune_local.py`)
- TPE (multivariate), **20 trials**, 32 private training episodes per trial, in parallel worker processes. The current
  params run as trial 0, and the winner must beat them on **64 held-out episodes** (paired) before `params.json` is
  written.

| Measure | Value |
| --- | --- |
| Trial 0, the untuned params (training) | **0.7539** |
| Best trial (training) | **0.7559** |
| Held-out gap, best vs untuned | **+0.0002** (−0.0010 to +0.0013) |

- **Reading:** tuning brought **no measurable gain**. The params were written because the point estimate was
  positive, but the interval holds 0. Twenty trials on small pools do not prove a ceiling. They are consistent with
  Phase 1 adding little over the seed planner.

## 5. Phase 3: strategic pivot to data-driven EDA
Parameter search and new rules have not moved the score, so the next step is to measure where the planner's remaining
**0.27 RSS gap** to the clairvoyant plan comes from. On the dev split that gap splits about **0.13 / 0.08 / 0.05 /
0.01** across levels 1-4.

### Plan
- An extraction script replays training episodes and dumps weekly telemetry to a DataFrame (`.parquet` / `.csv`): cost
  components, shed by grid, unserved demand by market, warnings, live threads, pending prohibitions, the forecast
  window and what actually happened. The replay machinery already exists (`src/sbf_starter/evolve/trace.py`). pandas
  is a training-only dependency.

### Questions to answer
1. **Signal distribution:** the timelines of `s.warnings` and `s.threads` before real disruptions vs before decoys.
   What lead time and real share does each channel give in practice?
2. **Cost dynamics:** holding vs shed and shortage, per episode and week. Already known: logistics is under 1 % of
   naive's cost. Open: the marginal value of a unit of buffer, to size floors from data.
3. **Topology and lead times:** alternative-route capacity and transit time per strait, to bound useful
   `buffer_weeks`.
4. **Forecast error:** accuracy of `s.demand_forecast` and of the persistence window at weeks 1-4 vs 20-24. If error
   in ordinary weeks dominates, levels 1-2 (0.21 of the gap) are the target, not crises.

### Discussion points for the mentor
- Is the remaining gap mostly **forecast error in ordinary weeks**, or **disruption response**? The level split
  suggests the former.
- Is a **scenario-based LP** (two windows, closure vs no closure, blended) worth its CPU (about 0.6 s per Full week
  now, against a 4 s budget)?

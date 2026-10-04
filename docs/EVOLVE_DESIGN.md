# AlphaEvolve loop for ShockBench-Flow: system design

Status: proposal for review, 2026-10-04. Scope: the automated search that writes, evaluates and selects
`agent.py` files. The agent itself is not written here.

Figures marked *measured* were taken on the development machine (12 cores, 31 GB, shockbench-flow 0.1.2
from `uv.lock`, other work running), mostly on the public dev split of Small, which has only 5 episodes per
harm level. Appendix D says how each was taken. Figures marked *estimate* are assumptions for the pilot
(Phase 4) to replace.

## Implementation status (2026-10-04)

Built and tested: the seed agent `agents/mine` (the package's planner vendored as `sbfplan/`, SciPy dual simplex,
the Dict-to-wire adapter, the four blocks, the deterministic LP-size cap), `scripts/vendor_planner.py`, the loop in
`src/sbf_starter/evolve/` with `examples/08_evolve.py`, the stress script `examples/09_stress.py`, and
`tests/test_planner_port.py` and `tests/test_evolve.py`. Measured: the port reaches the package's LP optimum week by week
(relative gap below 1e-9) and scores 0.7266 on the Small dev split against the package's 0.7294. `sbf check` passes on
Small (worst week 0.25 s of CPU) and Full (worst week 0.94 s).

Not built yet: the forecast audit and the signal audit of the report (section 4.3, parts 4 and 5), the literal-index
lint of S0, batched API requests, and the Neyman re-sizing of pools after a pilot. The mutator has run only in its
offline scripted mode: no API key was available.

Contents: [1 Decisions](#1-decisions-in-brief) ·
[2 What the repository establishes](#2-what-the-repository-establishes) ·
[3 Architecture](#3-architecture) · [4 Components](#4-components) ·
[5 The RSS dilemma](#5-the-rss-dilemma-metric-optimization-strategy) ·
[6 Execution pipeline](#6-execution-pipeline) · [7 Budgets](#7-budgets) ·
[8 Action plan](#8-action-plan) · [9 Risks](#9-risks-and-mitigations) · Appendices A to D

## 1. Decisions in brief

- **The seed is an LP planner, not a heuristic.** The package ships `mpc_det`, a rolling linear program
  over a forecast in which observed disruptions persist. It scores 0.729 on the Small dev split, against
  0.409 for send-the-maximum and 0.400 for the shipped heuristic (measured). Its code is MIT-licensed and
  needs only numpy and SciPy once its solver call is switched to SciPy's HiGHS and one schema check is
  dropped. GUIDE.md's "Approaches" section names this route.
- **The LLM evolves what the planner sees.** Four EVOLVE blocks set the planner's beliefs, its forecast of
  capacities, costs and demand over the planning window, edits to its objective (risk premiums, stock
  floors) and its horizon. This is the brief's "dynamic graph weight optimizer", with the LP doing routing
  and allocation.
- **Fitness is the board's formula, computed in-process.** The evaluator calls
  `shockbench_flow_agent.EpisodeSet` and keeps per-episode costs. It never parses printed output.
- **Selection is paired, stratified and held out.** Candidates play the same cached episodes of a private
  training root. Promotion needs a positive lower confidence bound on a second private root and a Full
  check. The public dev split confirms a release and never selects.
- **The RSS dilemma is real, but it is not about freight.** In every harm level about 99 % of the naive
  rule's cost is power shed and unmet chip demand. Freight, tariffs, holding, queueing and disposal are
  under 1 % (measured). A false alarm is expensive when it diverts scarce supply. The exchange rate still
  holds: a dollar lost in a level-1 episode costs ten level-4 dollars.
- **Almost no announcement is certain.** In the scored regime a decoy thread sends the same messages as a
  real one, final tariff notices and legal sanction publications (`pending_prohibitions`) included, until it
  is withdrawn at the week it would have taken effect, and the end of a closure is never announced. Only
  what is observed now is certain. The package's planner applies every pending prohibition as certain.
- **Most of the remaining score is in ordinary weeks.** Attainable savings are about the same in every
  level, so the levels' effective weights are close to p: 44 %, 33 %, 18 % and 4 % (measured). Of the
  0.27 RSS the planner leaves, 0.21 is in levels 1 and 2.
- **The LLM writes structure; an optimizer tunes numbers.** The mutator declares constants with ranges and
  CMA-ES tunes them on training episodes.
- **Full decides the prize.** The private board re-scores the same zip on 400 Full episodes. Code may not
  hard-code anything about Small, and every promotion passes a Full gate. A cold SciPy solve of the
  planner's LP took at most 0.8 s of CPU per Full week against a 4 s budget (measured on 26 weeks of one
  episode).

## 2. What the repository establishes

### 2.1 Corrections to the brief

| The brief assumes | What the repository and package say | Consequence for this design |
| --- | --- | --- |
| `claude.md` holds the rules and the graph | CLAUDE.md only includes AGENTS.md. The rules are in docs/GUIDE.md, "Rules"; the graph is `config["static"]`, documented in docs/fields/ | Prompts quote GUIDE.md and the field tables |
| 1 CPU, 4 GB, no GPU, offline | Confirmed (`shockbench_flow_agent.LIMITS`). Also 2 s of CPU per week on Small and 4 s on Full, 10 s of wall clock per week, 60 s to start, and an episode kill timer of 193 s (Small) and 500 s (Full) | The evaluator meters CPU per week; the frame shortens the horizon when CPU runs high |
| Harm weights 0.50, 0.30, 0.15, 0.05 | Confirmed (`shockbench_flow.scoring.rss.STRATUM_WEIGHTS`) | None |
| 50 % of the evaluation weight is on s = 1 | p is the share of scenarios. RSS pools dollars, Σ p_s ḡ_s / Σ p_s D̄_s, and level 1 holds 44 % of the denominator (measured) because attainable savings are similar in every level | Fitness and feedback use p-weighted dollars; per-level RSS is context only |
| The cost of caution is excess freight on false alarms | Logistics is under 1 % of naive's cost in every level; shed plus shortage is about 99 % (measured) | The false-alarm cost that matters is diverted supply |
| 128 observation arrays | 128 on Tiny (64 fields and their `.observed` masks), 112 on Small (measured); Full differs | Shapes come from `config["spaces"]` and `config["layout"]` |
| `early_warning_scores` telemetry | The field is `warning.score`: 22 units on Small (14 regions, 1 rival pair, 7 straits), lagged one week, loading 0.604 on the latent hazard. Two more signals exist: `messages.*`, with a 25 % to 54 % decoy share by channel, and `pending_prohibitions.*`, announced sanctions with their effective week, which decoy threads also produce | The belief block uses all three with their published reliabilities |
| 03 cuts flows by warnings during closures | The heuristic scales a lane's flow by the strait's observed open fraction. It ignores warnings, announcements and pending sanctions | None |
| The shipped rules are the baseline to evolve | send-the-maximum 0.409, the heuristic 0.400, the package's planner 0.729 on the Small dev split (measured) | The seed is the planner |
| Parse the stdout of `sbf evaluate --task=small` | `EpisodeSet` returns per-episode costs, harm levels, fallback weeks and CPU weeks; `sbf evaluate --out=x.json` writes JSON | The evaluator calls `EpisodeSet` in-process |
| 04 caches N_s and C_s | `EpisodeSet` caches per-episode naive and clairvoyant costs under `~/.cache/shockbench-flow`, keyed by package version, root and episode | Pools are built once; a candidate then costs only its own rollouts |
| Optimize on Small | The public board plays 200 Small episodes. The final ranking re-scores the same zip on 400 Full episodes of 104 weeks | Network-agnostic code and a Full gate |
| No neural networks | The rules allow PyTorch (CPU, one thread); excluding it is this design's choice | numpy and SciPy only: predictable CPU, reviewable code |

### 2.2 Baselines on the Small dev split (measured)

| Agent | RSS | 90 % interval | Level 1 | Level 2 | Level 3 | Level 4 |
| --- | --- | --- | --- | --- | --- | --- |
| naive rule | 0 by definition | | 0 | 0 | 0 | 0 |
| shipped heuristic (`agents/heuristic`) | 0.400 | 0.294 to 0.502 | 0.31 | 0.40 | 0.58 | 0.57 |
| send-the-maximum (`agents/template`) | 0.409 | 0.297 to 0.511 | 0.30 | 0.43 | 0.59 | 0.56 |
| the package's planner (`mpc_det`) | 0.729 | 0.666 to 0.787 | 0.71 | 0.75 | 0.73 | 0.76 |

The planner beats send-the-maximum by +0.321, with a paired 90 % interval of 0.256 to 0.395.
Send-the-maximum and the heuristic differ by +0.008, with an interval of −0.012 to 0.030, so these episodes
cannot tell them apart.

### 2.3 Other facts the design relies on

All were read in the package source or measured.

- **The naive rule is an order-up-to inventory policy.** At reset it plans routes by a min-cost flow and
  sets a target level per destination and commodity. Each week it orders the gap between target and
  inventory position, splits it across planned routes, switches to a backup when the primary is observed
  closed and costs at most twice as much, and stops shipping what cannot arrive before the horizon. It reads
  no warning, message or pending sanction (`shockbench_flow/policies/naive.py`).
- **The planner re-solves a window LP every week.** Its window is H = L weeks, the longest lead time of the
  naive plan's supply chains. Its forecast keeps the week's observed state for the whole window, switches
  pending prohibitions on at their effective week, and takes demand from the 8-week forecast, then the
  seasonal mean. The LP is the clairvoyant oracle's own model on the rolled-forward instance. The planner
  executes the plan's first week, tanker releases included, and plays naive's action if the solver fails
  (`shockbench_flow/policies/lp_common.py`, `mpc_det.py`).
- **The planner uses one signal, as if it were certain.** Pending prohibitions are its only forward-looking
  input, applied from their effective week although decoy threads publish them too. It reads no warning
  score and no announcement message (threats, tariff proposals, final notices), and it assumes an observed
  closure lasts the whole window. In the scored regime the end of a closure is never shown: `closure_end`
  is hidden because the regime's `chi` is false.
- **The window LP fits the budget with SciPy's solver.** On Small it has about 10,000 columns and 6,100
  rows, and a cold `scipy.optimize.linprog(method="highs-ds")` solve takes 0.09 to 0.12 s of CPU at the
  median and 0.14 s at worst. On Full it has about 27,800 columns and 14,600 rows and takes 0.57 s at the
  median and 0.80 s at worst. The interior-point method takes up to 3.1 s on Full, too close to 4 s. Cold
  solves reach the same optimum as the package's warm-started ones (measured).
- **Vendoring is feasible.** The planner's import closure is numpy, SciPy and the standard library, apart
  from the `highspy` solver binding (imported inside the solve function) and a JSON-schema validator in the
  instance loader. The package is MIT-licensed, so a trimmed copy may ship in the submission with its
  notice.
- **The agent sees a converted observation.** The planner reads the server's internal observation. On Small
  and Full the agent's Dict observation groups in-transit shipments by (edge, commodity, lane, arrival
  week) and queued cargo by (lot key, arrival week), and the package has no inverse conversion. The frame
  needs a tested adapter (Phase 1).
- **Harm levels come from the event list alone.** A harm value is computed without the LP. Level 1 is at
  or below the generator's median and level 4 its top 5 %. Ties at a cut go to the lower level, so episodes
  without active events are level 1 (`shockbench_flow/disruption/strata.py`).
- **Pools can be stratified like the board.** `EpisodeSet.build(task, [indices], entropy=root)` scores any
  list of episodes, and `shockbench_flow.hosting.split.fill_strata` picks the first N episodes of each
  level on a root, which is the dev split's own rule.
- **The organisers' variance model is importable.** `shockbench_flow.evaluation.split_size` (`spread`,
  `se`, `neyman`, `size_for_gap`) gives the standard error of a paired gap at any episode count and the
  Neyman allocation across levels.
- **A rollout records cost by component.** Each week's `StepRecord` holds freight, war risk, tariff,
  holding, queue holding, shortage, disposal and shed, plus requested and executed flows and queues.
- **Disruptions are self-exciting.** The generator is a Hawkes process, so one event raises the near-term
  rate of others and risk should decay after an event, not reset.
- **The signal model is published at runtime.** In the scored `standard` regime, `config["static"]["regime"]`
  holds the warning lag (1 week), the loading (0.604), the skill (0.69 to 0.71) and each announcement
  channel's decoy share. A decoy thread carries every message of its type (a tariff's proposal and final
  notice, a sanction's threat and legal publication) and is withdrawn at the week it would have taken
  effect (`shockbench_flow/disruption/announce.py`, `information/messages.py`).

## 3. Architecture

```
 ┌──────────────────────── examples/08_evolve.py (orchestrator, one run folder) ─────────────────────────┐
 │                                                                                                        │
 │  Program database ──sample──► Prompt builder ─────────► Mutator ────────────────► Patch applier        │
 │  SQLite + folders     parent,   cached static prefix     Claude Opus 5.5           replaces the named  │
 │  4 islands x          2 inspira- + parent blocks         structured JSON:          EVOLVE blocks,      │
 │  MAP-Elites grid      tions,     + its report            hypothesis, USD effect    writes params.json  │
 │         ▲             directive  + directive             per level, blocks,              │             │
 │         │                                                params                          ▼             │
 │         │       Evaluator cascade (process pool; per-episode results cached by file hash)              │
 │         │       S0 static ─► S1 smoke ─► S2 screen ─► S3 train ─► S4 tune (top programs only)          │
 │         │       AST, imports  4 episodes   32 episodes   +96 episodes  CMA-ES on declared params       │
 │         │       frame hash    isolated,    paired vs     RSS, dollars by level, descriptors,           │
 │         │                     CPU, determ. parent        trace replay of 6 episodes                    │
 │         └──────────── fitness, descriptors, report ◄───────────┘                                       │
 │                                                                                                        │
 │  Champion challenge (S5): valid pool, 128 Small episodes, paired lower bound > 0                       │
 │                           + full_valid pool, 32 Full episodes, no regression, CPU inside allowance     │
 │  Release (S6, run by the participant): sbf check --task=small|full [--docker] ► dev report ► sbf pack  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 Inside every candidate's agent.py, each week:
   observation ─► adapter ─► State ─► [belief] ─► [settings: H] ─► persistence window ─► [forecast]
               ─► window LP + [objective edits] ─► SciPy HiGHS ─► first-week action ─► Dict action
   ([...] = an EVOLVE block; everything else is the frozen frame)
```

- The **program database** stores every candidate with its stage, scores and report, and keeps one elite
  per cell of a behaviour grid on each island (section 4.6).
- The **prompt builder** joins a cached static system prompt (rules, scoring economics, block contracts,
  helper reference) with the parent's blocks, its report, two inspirations and a directive (section 4.4).
- The **mutator** returns JSON validated against a schema. The **patch applier** replaces whole named
  blocks, so no fuzzy diff matching is needed.
- The **evaluator** runs a cascade in which each stage costs more than the last and stops at the first
  failure. Results are cached per (file hash, pool, episode), so any two evaluated programs compare without
  replaying (section 4.2).
- The **champion** changes only through the challenge stage. Release steps stay manual, and nothing in the
  loop uploads.

**Why a custom loop.** Open-source AlphaEvolve implementations exist and were not evaluated for this design.
The hard parts here are specific to the benchmark: board-faithful fitness on cached episodes, paired
statistics under heavy noise, and the Full gate. The generic parts (archive, sampling, prompt assembly) are
a few hundred lines.

## 4. Components

### 4.1 The candidate program and the `agent.py` skeleton

A candidate is an ordinary submission folder, so every repository tool (`sbf check`, `sbf evaluate`,
`sbf pack`) works on it unchanged:

```
<candidate>/
  agent.py        frozen frame + four EVOLVE blocks (the only text the mutator edits)
  params.json     tunable constants {"name": value}; the numeric tuner writes it
  frame.py        the frozen frame (adapter, State, Window, LPEdits, Planner)
  sbfplan/        trimmed copy of shockbench_flow's planner modules, solved with SciPy (MIT notice kept);
                  identical in every candidate
```

**The frozen frame** is written in Phase 1, tested and hashed. S0 rejects a candidate whose frame or
`sbfplan/` hash differs from the seed's. The frame has six parts:

- **Observation adapter.** It turns the Dict observation into the fields the vendored planner reads. Phase 1
  proves that it gives the same LP, matrix for matrix, as the package builds from the server's own
  observation.
- **`State`.** It exposes stock positions by destination, routes with their open fraction, capacity, freight,
  tariff and war-risk cost, the demand forecast, and the parsed signals: warning by unit, live threads by
  channel with their stated weeks, and pending prohibitions with their effective weeks. Lookups are by name, so no
  block needs a literal index.
- **Planner.** It loads the instance from `config["static"]["instance"]`, computes L once, builds the
  persistence window, builds the window LP, solves it with `linprog(method="highs-ds")` and maps the
  first week of the solution to the Dict action (`flows`, `override_qty`, `release_mode`).
- **Window helpers.** These keep the forecast consistent when a block edits it: a strait's open fraction
  and its pool throughputs change together, a war-risk class change updates queue holding and transit
  war-risk costs, and prohibitions and tariffs switch on from a stated week.
- **Objective helpers (`LPEdits`).** They add a per-unit premium to an edge's flows over a range of weeks,
  set a floor on the end-of-week stock of a node and commodity, and scale the shortage penalty of a market or
  the value of lost load of a grid. Phase 2 builds them from the LP's column and row keys.
- **LP-size cap and safety.** The window is capped by the LP's column count (26,000 on Small, 40,000 on Full,
  from the measured solve times), never by measured time, so a plan never depends on machine load and cached scores
  reproduce. HiGHS stops at the week's whole budget, a week that is lost to naive anyway. A block
  exception re-plans the week on the seed's inputs. A failed solve sends the maximum (the package's planner
  had no failed solve in 20 dev episodes). The frame never raises.

**The EVOLVE blocks** have fixed signatures. The mutator rewrites whole functions, and a block may define
helpers inside its own region.

| Block | Signature | Seed behaviour | What evolution is expected to find |
| --- | --- | --- | --- |
| `belief` | `update_belief(mem, s) -> mem` | no risk | per strait, edge and region: `warning.score` as a decaying hazard; every thread, pending prohibitions and final notices included, weighted by its channel's real share and dropped on withdrawal; how long closures last, learned from their age |
| `forecast` | `forecast(w, s, mem) -> w` | persistence | expected reopening of closed straits; pending prohibitions as expected rather than certain losses; expected open fractions and capacities under threat; expected tariffs; demand and supply outlooks |
| `objective` | `objective(edits, s, mem) -> edits` | no edits | premiums on routes exposed to risk; stock floors at grids and markets behind a threatened strait; penalty scales |
| `settings` | `settings(s, mem, default_H) -> H` | H = L | longer windows when a disruption is near, shorter ones when calm or when CPU is tight |

Rules stated in the frame's docstring, repeated in the prompt and enforced at S0 and S1:

- Blocks import only the standard library and numpy. SciPy is used through the frame.
- No file access, no randomness, and no logic that depends on the clock.
- No literal slot, node, edge or strait indices or names. Code goes through `State` and the helpers.
- A block's own code stays under 20 ms of CPU per week on Small. The LP's time is metered separately.

The built skeleton is `agents/mine/agent.py` (the blocks and `Agent`) with `agents/mine/frame.py` (the frozen frame: adapter, `State`, `Window`, `LPEdits`, `Planner`) and `agents/mine/sbfplan/` (the vendored planner). `update_belief` takes `(mem, s)`; the published regime is `s.regime`.

### 4.2 The evaluator (fitness function)

**Episode pools.** A script builds them once per package version. `EpisodeSet` caches their references on
disk, so loading a pool later takes seconds.

| Pool | Task | Root | Episodes | Used by |
| --- | --- | --- | --- | --- |
| `train_screen` | small | 20261004 | 8 per level (32) | S2 screen |
| `train` | small | 20261004 | 32 per level (128), a superset of `train_screen` | S3 fitness, S4 tuning, archive |
| `valid` | small | 20261005 | 32 per level (128) | S5 challenge only |
| `full_valid` | full | 20261006 | 8 per level (32) | S5 challenge only |
| `dev` | small | 0 | the dev split (20) | the release report only; it never selects |

Each pool takes the first N episodes of every level on its root with `fill_strata`, and harm comes from the
event list without solving the LP. The counts per level are starting values. After the pilot they are reset
by Neyman allocation (`split_size.neyman`) from the measured spread of paired gaps between real candidates.

**The cascade.** Each stage costs more than the last, and a candidate stops at its first failure.

| Stage | What runs | Rejects when | Cost |
| --- | --- | --- | --- |
| S0 static | AST parse; `frame.py` and `sbfplan/` hashes; imports (`shockbench_flow_agent.submission.missing_imports`, `agent_warnings`); banned calls (`open`, `eval`, `exec`, `subprocess`, `socket`, `random`, clock reads); literal-index lint; size | any violation | milliseconds |
| S1 smoke | `play_isolated` on two Small episodes (one calm, one level 4), then the same two from a byte-different copy | an exception, a naive substitution, invalid entries, costs that differ between the copies, block CPU over 20 ms or a week over 0.5 s (p99) | 4 rollouts |
| S2 screen | `train_screen` with `cpu_budget=True`; paired gap to the parent | the upper end of the 90 % paired interval is below 0 | 32 rollouts |
| S3 train | the other 96 `train` episodes: RSS, dollars by level, descriptors, a trace replay of 6 episodes for the report | any fallback week, invalid entry or failed solve | 96 rollouts |
| S4 tune | CMA-ES on the declared params, for programs in their island's top decile only | the tuned point is not better than the untuned one on `train` | 20 to 40 x 32 rollouts |
| S5 challenge | `valid` and `full_valid`, each paired against the champion | the lower 90 % bound of the gap on `valid` is at or below 0; the Full gap is below −0.02; any Full fallback week; a Full week over 1.5 s (p99) | 128 Small and 32 Full rollouts |
| S6 release | `sbf check --task=small` and `--task=full` (`--docker` where Docker exists), the dev report, `sbf pack` | any check fails | minutes |

**Caching and determinism.** Per-episode rows are stored under (SHA-256 of the candidate's files, pool,
episode, package version). S1 proves that a candidate's cost does not depend on its policy seed, which the
server salts with the zip's hash, so comparisons between any two evaluated programs cost nothing. Folders are
scored the way the server scores them: validated as a zip and seeded by its SHA-256.

**Statistics.**

- RSS of any cost vector comes from `EpisodeSet.rss(J)`, the scorer's exact integer-cent formula.
- A paired gap uses a stratified bootstrap with the same resampled episodes for both programs (2,000
  resamples), which is the method of `EpisodeSet.compare` applied to stored vectors.
- `split_size.spread(..., kind="gap")` gives the analytic standard error used to size the pools. For the
  planner against send-the-maximum, two very different programs, it is 0.048 at 5 episodes per level and
  0.019 at 32 (measured). Gaps between a parent and a small mutation vary less per episode, so their errors
  will be smaller; the pilot measures them.
- The archive stores RSS on `train`. Promotion uses the lower 90 % bound of the paired gap on `valid`, never
  a point estimate.

Interface sketch:

```python
# src/sbf_starter/evolve/evaluate.py (sketch)
from shockbench_flow.scoring.rss import STRATUM_WEIGHTS
from shockbench_flow_agent import EpisodeSet


def load_pool(task, root, indices, workers=8):
    """An EpisodeSet over fixed episodes of a private root; references come from the disk cache."""
    return EpisodeSet.build(task, indices, entropy=root, n_jobs=workers)


def rows(pool, name, folder, store, workers=8):
    """Per-episode rows of a candidate: J_policy_cents, fallback_weeks, cpu_weeks, invalid_entries, first_error.
    Played once as the server plays it (validated zip, weeks over the CPU budget given to naive), then cached."""
    key = (files_sha256(folder), name)
    if key not in store:
        store[key] = pool.play(str(folder), cpu_budget=True, n_jobs=workers)
    return store[key]


def dollars_by_level(pool, J):
    """What RSS actually sums: per level, the mean saving g, the attainable saving D and p_s * g, in USD."""
    out = {}
    for s, p in enumerate(STRATUM_WEIGHTS, start=1):
        kept = [(r, j) for r, j in zip(pool.references, J) if r["stratum"] == s and r["excluded"] is None]
        g = sum(r["J_naive_cents"] - j for r, j in kept) / len(kept) / 100
        D = sum(r["J_naive_cents"] - r["J_oracle_cents"] for r, _ in kept) / len(kept) / 100
        out[s] = {"g": g, "D": D, "p_g": p * g}
    return out


def fitness(pool, J):
    return pool.rss(J)["rss"]  # pooled over the harm levels with the board's weights
```

### 4.3 Diagnostics and feedback

Each evaluated candidate gets a `report.json` and a Markdown rendering of about 2,000 to 3,000 tokens for
the prompt. It has eight parts:

1. **Headline.** Train RSS with its interval, the paired gap to the parent, and the share of bootstrap
   draws in which the child is better.
2. **Dollars by level.** Mean saving per episode ḡ_s, attainable saving D̄_s, the contribution p_s·ḡ_s,
   and its change against the parent. Per-level RSS appears as a column labelled "ratio".
3. **Cost components by level, against naive and against the parent, in USD per episode.** Shed and
   shortage are broken down by grid and by market, because they carry the score; the six logistics
   components are summed. A false alarm shows here as shed or shortage rising somewhere in level 1.
4. **Forecast audit.** For each week the window the planner used, after the forecast block, is compared
   with what actually happened over the same weeks: mean absolute error by horizon for strait open
   fractions, edge capacities, demand and supply, against the persistence forecast's error. This shows
   whether a belief made the forecast better or worse.
5. **Signal audit.** For each signal kind (warning above a threshold per unit kind, military threat,
   sanction threat, tariff proposal, final notice, pending prohibition, observed closure): how often it
   occurred in training episodes, how often a matching disruption followed, and how the candidate's cost
   moved against the parent in the following weeks.
6. **Worst episodes.** The 3 with the largest loss against the parent and the 3 with the largest remaining
   gap to the clairvoyant plan. Each gets a ten-line narrative: level, harm, event timeline
   (`shockbench_flow_gym.timeline.episode_events`), the weeks where cost diverged by component, and the
   planner's window and action in those weeks.
7. **Health.** Fallback weeks, invalid entries, failed solves, window lengths used, CPU per week (p50, p99,
   max, block code and LP separately), the first error, and the line count of each block.
8. **Calibration.** The mutator's own predicted dollar effect per level, from its output, next to the
   measured effect, for this candidate and as a running average on its island.

The trace replay re-runs the 6 selected episodes through the package's `rollout` behind the agent shim, the
path `EpisodeSet.play` uses, to keep each week's `StepRecord`. A test asserts that the replayed cost equals
the cached one to the cent. The realized marks of each pool episode (what really happened each week) come
from `shockbench_flow.marks.compute_marks` and are computed once, like naive's records.

### 4.4 The mutator

**Model and request.**

- **Model.** `claude-opus-5-5` through the Messages API with the `anthropic` Python SDK, added as a
  training-only dependency (`uv add anthropic`). Thinking is always on for this model and its default
  effort is `medium`, so the request sets effort per directive: `high` by default, `xhigh` for `explore`.
- **Structured output.** `output_config.format` carries the JSON schema of Appendix B, so the response
  parses without regular expressions. The schema asks for a short hypothesis, not for written-out
  reasoning, which would add cost and can trigger refusals.
- **Streaming** with `get_final_message()` and `max_tokens` 64,000, since rewriting a block after thinking
  can pass the 16,000 that suits a non-streaming call.
- **Refusal fallback** with `fallbacks="default"` and the beta `server-side-fallback-2026-07-01`. A
  response that ends in `refusal` or `max_tokens`, or fails the schema, is logged and its slot skipped.
- **Prompt caching.** The static system prompt (Appendix A plus the helper and field reference, about 12,000
  to 15,000 tokens) carries `cache_control`. The parent, report, inspirations and directive follow it,
  uncached. From the second call on, `usage.cache_read_input_tokens` must be above 0. The default 5-minute
  lifetime suits a continuous loop; use `"ttl": "1h"` if calls are more than 5 minutes apart.
- **Batches.** An overnight run can send each generation's requests as one Message Batch at half price;
  most batches finish within an hour. Batches reject `fallbacks`, so a refusal there is skipped.

```python
# src/sbf_starter/evolve/mutate.py (sketch; check the fallbacks keyword against the installed SDK in Phase 3)
import json

import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from .env (gitignored); removed from evaluator processes


def propose(system_prompt: str, user_prompt: str, schema: dict, effort: str = "high") -> dict | None:
    with client.beta.messages.stream(
        model="claude-opus-5-5",
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        message = stream.get_final_message()
    record_usage(message.usage)  # input, cache write, cache read and output tokens feed the budget guard
    if message.stop_reason in ("refusal", "max_tokens"):
        return None
    return json.loads(next(b.text for b in message.content if b.type == "text"))
```

**Dynamic part of the prompt** (the user message; the static part is Appendix A):

````
## Directive
{directive}: {directive_text}
You may rewrite: {allowed_blocks}.

## Parent {parent_id} (island {island}, generation {generation})
Training RSS {rss} (90 % interval {lo} to {hi}); paired gap to its own parent {gap}.

### Its blocks
```python
{blocks}
```
### Its params.json
{params}

## Evaluation report of the parent
{report_markdown}

## Inspirations: what differs from the parent
### {inspiration_id} (RSS {rss}, cell {cell}, island {island})
```python
{differing_blocks}
```

## Recent attempts on this island (do not repeat a failed idea without a new reason)
| hypothesis | directive | outcome | measured USD per episode by level (1, 2, 3, 4) |
{attempt_rows}
````

**Directives.** Each call carries one directive, drawn by a bandit that favours directives with a high
recent acceptance rate on the island.

| Directive | Blocks it may rewrite | Drawn more often when |
| --- | --- | --- |
| `fix` | the failing block | the child failed S0 or S1; the traceback is included |
| `announcements` | belief, forecast | early in a run: pending prohibitions and tariff notices weighted by their thread's real share, withdrawn threads dropped |
| `risk_forecast` | belief, forecast | shed and shortage in levels 3 and 4 dominate the remaining gap |
| `false_alarms` | belief, forecast, objective | the level 1 or 2 contribution fell against the parent |
| `ordinary_weeks` | forecast, settings | the level 1 and 2 gap to the clairvoyant plan dominates (demand outlook, horizon, end of episode) |
| `objective_shaping` | objective | the report shows a grid or market starved behind a risky strait |
| `cpu` | settings, forecast | block or LP CPU is near its allowance, on Small or Full |
| `crossover` | any | two strong programs sit in distant cells |
| `simplify` | any | the parent is long; costs must stay identical on `train_screen` |
| `explore` | any | an island stagnates (for example a second window for a closure scenario, with the two plans blended) |
| `params_only` | none: it declares params for the tuner | a hypothesis depends on thresholds |

Two directive texts, as examples:

- `false_alarms`: "The parent loses money in level 1 or 2 against its own parent. Use the signal audit and
  the worst episodes to find the precautions that fire in calm episodes, and make them proportional to the
  evidence or remove them. Keep what it gains in levels 3 and 4 unless the trade is worth it at the
  exchange rate."
- `ordinary_weeks`: "Most of the remaining score is in levels 1 and 2, where few disruptions happen. Use the
  forecast audit to find where the persistence forecast misleads the planner in ordinary weeks (demand
  beyond the 8-week forecast, the window length, the last weeks of the episode) and improve it."

### 4.5 The numeric tuner

The LLM is good at structure and poor at fine numbers on a noisy objective, so constants are tuned
separately.

- A block reads constants with `p("name", default)`. The mutator declares each with a range, at most 8 per
  candidate. Only scalars that mean the same on Small and Full are allowed (decay rates, the weight of a
  channel's evidence, demand and throughput safety factors, horizon offsets), never per-slot vectors. The
  organisers' own planner variants in `shockbench_flow.policies.field` tune the same kind of numbers (a
  demand scale, a throughput scale, a fixed window).
- CMA-ES runs on the normalized ranges with a population of 8 for 3 to 5 generations. Members of one
  generation play the same 32-episode stratified subset of `train`, so they compare paired, and the subset
  rotates between generations. The `cma` package is a training-only dependency.
- The best point is checked on the full `train` pool. It replaces the untuned values only if its paired gap
  is positive.
- Tuning costs 20 to 40 times an S2 screen, so it runs only for programs in the top decile of their island.

### 4.6 The program database

- **Storage.** SQLite from the standard library plus one folder per candidate inside the run folder. A row
  holds the id, parent, island, generation, directive, file hash, the stage reached and why it stopped,
  train RSS and interval, descriptors, cell, tokens and dollars. A run resumes from it.
- **Behaviour grid.** Each island keeps a MAP-Elites grid of 5 x 5 cells over two descriptors measured on
  `train`: *calm RSS* (levels 1 and 2 pooled) and *crisis RSS* (levels 3 and 4 pooled). A cell keeps its
  highest train RSS. Bin edges come from the pilot's distribution.
- **Islands.** There are four. `forecast` may rewrite belief and forecast. `objective` may rewrite belief
  and objective. `lean` may rewrite any block but runs with half the CPU allowance on Full, as insurance
  against a slower scoring server. `open` has no restriction. Every 25 accepted children the global best is
  offered to every island as an inspiration, and an island without improvement for 40 children is reseeded
  from another island's best.
- **Sampling.** The parent comes from the island's top 5 by train RSS with probability 0.7 (softmax on RSS)
  and from a random occupied cell otherwise. The inspirations are the island's best and an elite from a
  distant cell.

The grid keeps crisis readiness alive. A program that saves in levels 3 and 4 but loses a little in level 1
keeps its own cell even while a calm-optimized program leads the island, so a later crossover can combine
the two.

### 4.7 The orchestrator

`examples/08_evolve.py` follows the repository's pattern for examples: its options are the keyword
arguments of `main`, exposed through `fire.Fire(main)`, and its runs write to
`outputs/08_evolve/<date_time>/`. The modules live in `src/sbf_starter/evolve/` so tests can import them.

- **Options.** `task`, `train_root`, `valid_root`, `full_root`, `per_level`, `workers` (8 by default on
  this machine), `llm_concurrency` (4), `max_candidates`, `budget_usd`, `effort`, `islands`, `batch`,
  `resume` and `out`.
- **Concurrency.** Mutator calls run in a thread pool, since they wait on the network. Evaluation runs in one
  queue served by a joblib process pool, since it is CPU-bound. Workers used about 230 MB each while
  building Small references (measured); planner rollouts will need somewhat more, and 8 workers still fit.
- **Budget guard.** Each response's `usage` is priced (input, cache writes, cache reads, output). The loop
  stops at `budget_usd`, at `max_candidates`, or at a wall-clock limit.
- **Logging.** Each candidate appends one JSON line to `events.jsonl`. Every 10 candidates a summary line
  gives the best RSS per island, the acceptance rate per directive, tokens, dollars and the evaluator's
  queue length.
- **Secrets.** Evaluation processes start without `ANTHROPIC_API_KEY` in their environment, and the package
  hides secret-like variables while an agent plays. `.env` is never printed or committed.

## 5. The RSS dilemma: metric optimization strategy

### 5.1 What the score rewards

The score is RSS = Σ_s p_s ḡ_s / Σ_s p_s D̄_s. Its denominator, Δ = Σ_s p_s D̄_s, is fixed by the
episodes and their references, and no agent changes it. So RSS = G / Δ, where G = Σ_s p_s ḡ_s is the
expected dollar saving per episode against naive when scenarios arrive in the generator's natural mix. The
equal-per-level episode sets and the weights p restore that mix.

The exchange rate between levels is therefore p_s: a dollar gained or lost in level s moves RSS by p_s / Δ.

| Level | p_s | One dollar here is worth |
| --- | --- | --- |
| 1 (the calmest half of scenarios) | 0.50 | 10 level-4 dollars |
| 2 | 0.30 | 6 level-4 dollars |
| 3 | 0.15 | 3 level-4 dollars |
| 4 (the most harmful 5 %) | 0.05 | 1 level-4 dollar |

The per-level score in the report, ḡ_s / D̄_s, has a different denominator in each level, so it says little
about how much a level matters. The loop optimizes G and shows per-level ratios only as context.

### 5.2 What the measurements show

Measured on the Small dev split (5 episodes per level, so the figures are rough). Δ is $0.92 trillion per
episode; naive costs $3.49 trillion and the clairvoyant plan $2.56 trillion on average.

| Level | p_s | D̄_s, USD per episode | Share of Δ | Shed and shortage, share of naive's cost | Logistics in naive's cost, USD per episode | What the planner leaves, RSS points |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.82 trillion | 44.5 % | 99.3 % | 23 billion | 0.129 |
| 2 | 0.30 | 1.02 trillion | 33.1 % | 99.3 % | 29 billion | 0.082 |
| 3 | 0.15 | 1.12 trillion | 18.3 % | 99.4 % | 22 billion | 0.049 |
| 4 | 0.05 | 0.76 trillion | 4.1 % | 99.2 % | 31 billion | 0.010 |

Logistics is freight, war risk, tariffs, holding, queue holding and disposal. The last column is
(1 − the planner's level RSS) × the level's share of Δ, and it sums to the planner's 0.27 gap to the
clairvoyant plan.

Three readings follow.

1. **Attainable savings are about as large in calm episodes as in harmful ones.** The effective weights are
   therefore close to p, and ordinary weeks carry most of the score.
2. **Logistics is too small to decide anything.** Doubling every logistics cost in every level-1 episode,
   about $23 billion, costs 0.5 × 23 / 922 ≈ 0.012 RSS. A 1 % change in level-1 shed, about $19 billion,
   moves RSS by about 0.010. Send-the-maximum earns its 0.41 the same way: in level 1 it cuts shed by $269
   billion per episode against naive and pays $28 billion more in logistics.
3. **Level 4 holds 0.010 of the 0.27 the planner leaves; levels 1 and 2 hold 0.21.** Crisis cleverness
   pays only when it costs little in calm episodes.

### 5.3 The dilemma restated for this network

- **Supply is scarce.** The clairvoyant plan still costs $2.56 trillion per episode, so much of the shed and
  shortage cannot be avoided by anyone. The clairvoyant plan's cost was not split by component; that most of
  it is shed and shortage is an inference from naive's split. Gains come from putting scarce fuel and chips
  in the right place at the right time.
- **A precaution moves supply.** Pre-positioning fuel at grids behind a threatened strait, or routing chips
  around it, takes supply from somewhere it would have been used. When the threat is a false alarm, that
  supply was missing elsewhere, and the loss appears as shed or shortage at value-of-lost-load prices, not
  as freight.
- **The exchange rate applies to those losses.** A precaution that adds $X of shed in a calm episode must
  save $10X in a level-4 episode to break even.
- **The planner makes graded hedging natural.** Lowering a strait's expected open fraction in the forecast
  in proportion to the evidence makes the LP pre-position in proportion. No if-then rule is needed; sizing
  the hedge is the forecast's job.
- **Almost nothing announced is certain.** Decoy threads send final notices and legal publications too,
  and are withdrawn only at the week they would have taken effect, so a pending prohibition is a
  probability, not a fact. The package's planner treats it as a fact and pays for every decoy sanction. Only
  what is observed now (open fractions, capacities, prohibitions in force) is certain.

### 5.4 How the loop prices the dilemma

1. **Fitness is RSS on the training pool, which is G / Δ.** The exchange rate is built into the objective,
   and nothing needs tuning.
2. **Feedback is in p-weighted dollars.** Report parts 2 to 5 show where each dollar went, with shed and
   shortage by grid and market, a forecast audit and a signal audit.
3. **The prompt states the economics and the decision rules** (Appendix A, "Scoring economics").
4. **The mutator predicts its own dollar effect by level.** The report shows predicted against measured, so
   an island that overstates crisis savings sees it.
5. **Promotion uses a lower confidence bound.** Level-4 gains come with wide intervals, because harmful
   episodes differ a lot from one another; level-1 losses come with narrow ones. A child that trades a
   certain calm-period loss for an uncertain crisis gain fails S5 unless the gain is real.
6. **The behaviour grid keeps crisis-ready programs alive.** Its cells span calm RSS against crisis RSS, so
   readiness survives even while a calm-optimized program leads.

### 5.5 Signal reliability and the cost–loss rule

A precaution that diverts supply worth C is worth taking when q · L > C, where q is the probability of the
disruption given the evidence and L is the loss it avoids if the disruption happens. In the planner the rule
takes the form of expected values: a strait's forecast open fraction becomes its current value times
(1 − q × severity) from the week the threat could act, and the LP sizes the hedge. The evidence differs in
reliability, and the environment publishes it:

| Signal | Field | Reliability in the scored regime | Default use in the forecast |
| --- | --- | --- | --- |
| Pending sanction | `pending_prohibitions.*` | from legal publications, which decoy sanction threads also send; withdrawn at the stated week if a decoy. Its real share should follow the sanction threads' (an inference from the generator's design; the signal audit measures it) | the planner applies it as certain; weight it by the real share |
| Final tariff notice | `messages.*`, channel `tariff_final` | part of a tariff thread, real or decoy; states its effective week | an expected tariff from that week |
| Tariff proposal | channels `tariff_formal`, `tariff_informal` | 25 % and 36 % of threads are decoys | expected tariff, weighted by the real share |
| Sanction threat | channel `ties_threat` | 54 % decoys; no date stated | weak evidence: a small expected capacity cut |
| Military threat | channel `mid_threat` | 30 % decoys; no date stated | moderate evidence: an expected open-fraction cut on the named strait |
| Withdrawal | `messages.kind` = 4 | marks the thread as a decoy | drop what that thread added |
| Warning score | `warning.score` | one-week lag; loading 0.604 on the latent hazard; skill 0.69 to 0.71 | a hazard proxy: smooth it, combine it with threads, apply it gradually |
| Observed closure | `graph_now.open` | certain now; its end is not announced (`closure_end` is hidden in the scored regime) | an expected remaining duration, estimated from closures in training episodes |

The agent reads the decoy shares and warning parameters from `config["static"]["regime"]` at runtime, so
the code holds no hard-coded reliabilities.

## 6. Execution pipeline

**Bootstrap, once per package version:**

1. The orchestrator (`src/sbf_starter/evolve/pools.py`) computes the cut points, picks the stratified indices of `train`,
   `valid` and `full_valid` with `fill_strata`, builds their `EpisodeSet`s (references cached on disk), and
   stores each pool episode's realized marks and naive's per-week records.
2. `uv run pytest tests/test_planner_port.py` checks the seed in `agents/evo_seed/`: the same window LP as
   the package for the same weeks, the same optima, and RSS on `train` within the paired interval of the
   package's planner.
3. The orchestrator evaluates the seed on every pool and makes it the champion and the first elite of every
   island.

**The loop, for each candidate** (`uv run python examples/08_evolve.py --task=small --budget_usd=...`):

1. Pick an island (weighted by its recent improvement), a parent and two inspirations.
2. Draw a directive from the island's bandit.
3. Render the prompt: the cached static system prompt, then the dynamic user message.
4. Call the mutator. A refusal, truncation or schema failure is logged, and the slot is skipped.
5. Apply the edits: replace the named blocks, write the declared params' defaults into `params.json`,
   write the folder, run S0.
6. Run S1, S2 and S3 in order, stopping at the first failure and recording the stage and reason.
7. If the program is in its island's top decile and declared params, run S4.
8. Insert it into the archive (cell elite rule), update the bandit, write `report.json`.
9. If its train RSS beats the champion's by at least 0.01, queue S5. A win makes it the champion, copied to
   `agents/evo_champion/`.
10. Every 25 accepted children, offer migrations, check stagnation and the budget, and log a summary.

**Release, run by the participant and never by the loop:**

```bash
uv run sbf check agents/evo_champion --task=small
uv run sbf check agents/evo_champion --task=full
uv run sbf check agents/evo_champion --task=small --docker     # where Docker is available
uv run sbf evaluate agents/evo_champion --task=small           # the dev confirmation, reported once
uv run sbf compare agents/evo_champion <current submission> --task=small
uv run sbf pack agents/evo_champion
```

Uploading spends one of the day's three submissions and happens only when the participant decides.

## 7. Budgets

Measured on this machine:

| Item | Measured |
| --- | --- |
| Naive demand model for Small, once per package version (4 workers) | 289 s |
| Harm cut points and dev split selection for Small, once (4 workers) | about 11 minutes |
| Clairvoyant references, 20 Small episodes (4 workers) | 28 s; 5.0 to 5.8 s per episode |
| One Small rollout in one process: naive, send-the-maximum, the package's planner | 0.11–0.18 s, 0.23–0.37 s, 3.6–4.3 s |
| Window LP on Small: size; cold SciPy dual-simplex CPU per week (median, worst) | 10,000 x 6,100; 0.09–0.12 s, 0.14 s |
| Window LP on Full: size; cold SciPy dual-simplex CPU per week (median, p90, worst) | 27,800 x 14,600; 0.57 s, 0.75 s, 0.80 s |
| Worker memory while building Small references | about 230 MB |
| Paired-gap standard error, planner against send-the-maximum, at 5 and 32 episodes per level | 0.048 and 0.019 |

Derived for planner candidates, with 8 evaluation workers (estimates):

| Item | Estimate |
| --- | --- |
| CPU per Small week (window build about 0.04 s, solve about 0.12 s) | about 0.16 s, 8 % of the 2 s budget |
| CPU per Full week | about 0.6 s at the median, 0.85 s at worst: 15 % to 21 % of 4 s |
| One rollout: Small, Full | about 9 s, about 65 s |
| S1 to S3, 132 Small rollouts | about 2.5 minutes |
| S5 challenge, 128 Small and 32 Full rollouts | about 7 minutes |
| `full_valid` pool, built once (naive model, cut points, 32 clairvoyant plans of over a minute each) | about an hour |
| Throughput, evaluator-bound | about 20 candidates per hour when all reach S3; more with S2 rejections |

Mutator cost per candidate at Claude Opus 5.5 prices ($4 per million input tokens, $20 per million output
tokens, $0.20 per million cache-read tokens). The token counts are estimates:

| Part | Tokens | Cost |
| --- | --- | --- |
| Static system prompt, read from cache | 12,000 | $0.002 |
| Parent, report, inspirations, directive | 15,000 | $0.06 |
| Output: thinking and JSON | 8,000 | $0.16 |
| **Per candidate** | | **about $0.22, or $0.11 batched** |

At 20 candidates an hour, a 10-hour run is about 200 candidates and about $44 with interactive calls. The
cache write happens about once per 5 minutes of continuous running and is negligible.

## 8. Action plan

| Phase | Work | Done when | Effort (estimate) |
| --- | --- | --- | --- |
| 0. Pools | `src/sbf_starter/evolve/pools.py`, built on demand by the loop, for Small and Full: cut points, stratified indices, references, realized marks, naive records | the pools load from cache in seconds; timings recorded | 0.5 day, plus about an hour of compute for Full |
| 1. Seed | Vendor the planner into `agents/mine/sbfplan/` (`scripts/vendor_planner.py`): trim it, replace `highspy` with `linprog(method="highs-ds")`, drop the schema validation, keep the MIT notice. Write the observation adapter, `State`, `send_max`, `to_dict_action` and `tests/test_planner_port.py` | the window LP equals the package's for every week of 4 Small and 1 Full episodes; optima equal; RSS on `train` within the paired interval of the package's planner; `sbf check --task=small` and `--task=full` pass; CPU per week recorded | 2 to 3 days |
| 2. Evaluator | `src/sbf_starter/evolve/evaluate.py` and `trace.py`; the window and objective helpers | the seed beats send-the-maximum with a positive lower bound on `valid` (a known gap of about 0.32); a byte-different copy of the seed shows a gap of exactly 0; a negative control (every strait closed for the whole window) is rejected at S2; a nondeterministic agent fails S1; replayed costs equal cached ones to the cent | 2 days |
| 3. Loop | `prompts.py`, `mutate.py`, `tune.py`, `archive.py`, `examples/08_evolve.py`; `uv add anthropic cma` | a 20-candidate dry run with a fake mutator (random param perturbations, no API calls) completes, survives a kill and resumes | 1.5 days |
| 4. Pilot | about 40 real candidates with a $25 budget | tokens per call, cache hits, acceptance per directive and paired-gap spreads measured; pool sizes reset by Neyman allocation; section 7 updated | 0.5 day |
| 5. Runs | overnight runs, batched requests optional | the champion beats the seed on `valid` with a positive lower bound and passes the Full gate | ongoing |
| 6. Release | the commands of section 6, run by the participant | every check passes; the dev report is recorded | 1 hour per release |

Phase 1 is a milestone on its own: once its checks pass, the seed is already a candidate submission, at
about 0.73 on the dev split against 0.41 for send-the-maximum. Whether to submit it is the participant's
call.

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The adapter loses a detail the LP uses (grouped shipments, dense queued cargo) | Phase 1's matrix-equality test finds it. If exact equality is impossible, accept RSS-level equivalence on `train` and document the difference |
| The vendored planner drifts from the package | Keep the package pinned. After `uv sync --upgrade-package shockbench-flow`, re-vendor and rerun the port test; references recompute because they are keyed by package version |
| Evolved forecasts make the LP infeasible or slow | The frame re-plans on the seed's inputs; S1 and S5 measure CPU; the LP-size cap bounds the window |
| The scoring server is slower than this machine | Full's worst solve is 0.80 s against 4 s; the `lean` island; `sbf check --docker` before release |
| Selecting noise (the winner's curse) | Paired comparisons on shared episodes, promotion on a lower bound, disjoint pools, a dev split that never selects |
| `valid` slowly overfits through repeated challenges | Count challenges; after 20 promotions or a week of runs, draw a new validation root |
| Small-specific code fails or underperforms on Full | The literal-index lint at S0, the Full gate at S5, `sbf check --task=full` at release |
| Generated code misbehaves | S0 bans, isolated evaluation processes, no API key in their environment, time limits |
| API spend runs away | Every response priced from its `usage`; a hard budget stop; a cache-hit check on the second call |
| Unusable mutator output | Schema validation, one retry quoting the validation error, then skip |
| Exceptions used on purpose to borrow the server's naive fallback | The frame never raises. Fallback weeks appear publicly on the board, and any candidate with one is rejected |
| Results do not reproduce on another machine | Costs are exact only on one CPU type (docs/GUIDE.md). Compare candidates on one machine and keep each run's file hashes |
| Licence | The vendored copy keeps the package's LICENSE and THIRD_PARTY_NOTICES files |

## Appendix A: static system prompt (cached)

The orchestrator assembles this text once per run. It is followed by a reference generated from the
frame's docstrings (`State`, the window helpers, `LPEdits`) and the observation fields of
docs/fields/small.md.

```text
You improve a control policy for ShockBench-Flow, a supply-network benchmark, by rewriting parts of its
Python source. An automatic evaluator runs your code on simulated episodes and reports the result.

# The task
Each week of an episode (52 weeks on the Small network, 104 on Full) the agent chooses how much of each
commodity to ship on each route and what tanker cargo queued at a strait does. Disruptions (strait
closures, sanctions, tariffs, conflicts, factory outages) are drawn before the episode, and the agent never
changes them. An episode's cost in USD is freight + war-risk surcharges + tariffs + holding (higher for
cargo queued at a strait) + a penalty for each unit of unserved chip demand + disposal + power shed at the
grids (at the value of lost load), minus the value of stock left at the end.

# How a program is scored
For each episode n, g_n = J_naive - J is your saving against the naive rule and D_n = J_naive -
J_clairvoyant is the saving of a plan that knew the future. Episodes fall into four harm levels s, from 1
(calmest) to 4 (most harmful), with probabilities p = (0.50, 0.30, 0.15, 0.05). The score is
    RSS = sum_s p_s * mean_s(g) / sum_s p_s * mean_s(D).
The denominator does not depend on your program, so the score is proportional to the expected dollars
saved per episode, sum_s p_s * mean_s(g).

# Scoring economics (measured on the Small network)
- Power shed and unmet chip demand are about 99 % of the naive rule's cost in every harm level. Freight,
  tariffs, holding, queue holding and disposal together are under 1 %. Supply decides the score.
- The attainable saving is about as large in a calm episode as in a harmful one, so ordinary weeks carry
  most of the score: levels 1 and 2 hold about 78 % of it.
- A dollar lost in a level-1 episode costs as much score as 10 dollars gained in a level-4 episode, 3.3 in
  level 3 or 1.7 in level 2.
- A precaution rarely costs freight here. It moves scarce fuel or chips earlier or elsewhere, and when the
  feared disruption does not come, that supply is missing where it was needed: that is the expensive false
  alarm. Take a precaution only if P(disruption | evidence) * (loss avoided) > (value of the supply moved).
- Only what is observed now is certain: open fractions, capacities and prohibitions in force. Every
  announcement can be a decoy: a decoy thread sends the same messages as a real one, final tariff notices
  and legal publications (pending prohibitions) included, until it is withdrawn at the week it would have
  taken effect. The planner applies pending prohibitions as certain; weigh them instead. Closure end weeks
  are not announced. Respond to each signal in proportion to its channel's real share.
- The decoy share of each announcement channel and the warning model's parameters are in the `regime`
  argument of update_belief; read them there.

# The program
A frozen frame runs a linear-programming planner each week. It forecasts the next H weeks (by default
every observed disruption persists, pending prohibitions switch on at their effective week, and demand
follows the 8-week forecast, then the seasonal mean), solves the cost-minimizing plan over that window and
executes the plan's first week. You change what the planner sees through four blocks:
- update_belief(mem, s) -> mem: risk state carried across weeks.
- forecast(w, s, mem) -> w: edit the window through its helpers (see the reference below).
- objective(edits, s, mem) -> edits: risk premiums on routes, floors on stock, penalty scales.
- settings(s, mem, default_H) -> H: the window length this week.
If every block returns its input unchanged, the program is the seed planner, about 0.73 RSS on Small.
The forecast is a point forecast and the LP treats it as certain. Express a probability as an expected
value (an expected open fraction, an expected capacity) so that the LP's hedge grows with the evidence.

Rules for every block:
- Python 3.13 with the standard library and numpy; no other imports.
- Deterministic: no random numbers and no logic that depends on the clock.
- Network-agnostic: the same code runs on Small and on Full, and the final ranking uses Full. Never write
  literal slot, node, edge or strait indices or names; use the lookups in `s` and `w`.
- Fast: a block's own code under 20 ms per week. The LP's time grows with H, and the frame caps H by the
  LP's size.
- Keep each block's name and signature. Helper functions go inside the block's own region.
- Read tunable constants with p("name", default) and declare each one in "params" with a range.

# What you receive
A directive, the parent's blocks and params, its evaluation report (dollars by level, cost components,
a forecast audit, a signal audit, its worst episodes, health), up to two inspiration programs, and recent
attempts on the same island.

# What you return
JSON matching the schema: a hypothesis of at most five sentences (what the report shows, what you change,
why it saves money), your expected change in USD per episode for each harm level, the rewritten blocks, and
the declared params. Rewrite only the blocks the directive allows, and make one coherent change.
```

## Appendix B: mutator output schema

```json
{
  "type": "object",
  "properties": {
    "hypothesis": {"type": "string"},
    "expected_usd_per_episode": {
      "type": "object",
      "properties": {
        "level_1": {"type": "number"}, "level_2": {"type": "number"},
        "level_3": {"type": "number"}, "level_4": {"type": "number"}
      },
      "required": ["level_1", "level_2", "level_3", "level_4"],
      "additionalProperties": false
    },
    "blocks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string", "enum": ["belief", "forecast", "objective", "settings"]},
          "source": {"type": "string"}
        },
        "required": ["name", "source"],
        "additionalProperties": false
      }
    },
    "params": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "default": {"type": "number"},
          "low": {"type": "number"},
          "high": {"type": "number"}
        },
        "required": ["name", "default", "low", "high"],
        "additionalProperties": false
      }
    }
  },
  "required": ["hypothesis", "expected_usd_per_episode", "blocks", "params"],
  "additionalProperties": false
}
```

The applier also checks that each `source` parses, defines the block's function with its signature, and
replaces only the text between that block's markers.

## Appendix C: files this plan adds

```
scripts/vendor_planner.py         copies the package's planner into an agent as sbfplan/ and prunes it
agents/mine/                      the frame, the seed blocks and sbfplan/ (the vendored planner)
examples/09_stress.py             stress tests on stratified private episodes
agents/evo_champion/              the current champion, written by the loop
src/sbf_starter/evolve/
    pools.py                      pool definitions and loading
    evaluate.py                   the cascade, the cache, paired statistics
    trace.py                      replays with per-week records, forecast and signal audits, narratives
    prompts.py                    static prompt assembly, dynamic prompt rendering
    mutate.py                     the Anthropic client, the schema, patch application
    tune.py                       CMA-ES over declared params
    archive.py                    SQLite, islands, the behaviour grid, sampling, the champion
examples/08_evolve.py             the orchestrator; runs write outputs/08_evolve/<date_time>/
tests/test_planner_port.py        the seed's differential test against the package
tests/test_evolve.py              the evaluator's controls, patching, archive and resume
```

## Appendix D: how the measurements were taken

- **Shipped agents.** `EpisodeSet.build("small", "dev", n_jobs=4)`, then `score` on `agents/template` and
  `agents/heuristic` with 4 workers.
- **The package's planner.** `make_policy("mpc_det", PolicyContext(fq_quantile=generator_quantiles(inst,
  params)))` played by `shockbench_flow.dynamics.env.rollout` in the `standard` regime on each dev episode,
  with the scenario, marks, policy seed and naive fallback `EpisodeSet` uses. RSS by `EpisodeSet.rss`;
  intervals by the stratified paired bootstrap of `EpisodeSet.compare`, applied to the stored costs.
- **Level economics.** D̄_s and the shares from the cached references. Cost components by summing
  `StepRecord.costs` over rollouts of naive's anchor (`naive_fq.anchor_policy`) and of the template behind
  the agent shim.
- **LP timings.** A subclass of `MpcDet` whose `_record` hook re-solves each week's window model with
  `scipy.optimize.linprog` (`highs-ds` and `highs-ipm`) through the oracle's own solve function, CPU by
  `time.process_time`, with `OMP_NUM_THREADS=1`. Small dev episodes 0 and 80 every week; Full dev episode 0
  every fourth week.
- **Standard errors.** `split_size.spread` and `split_size.se` on the stored costs.
- **Build timings.** `EpisodeSet.build`'s progress lines and the run's total wall time; other work was
  running on the machine.

Private package functions (`_world`, `_policy_seed`, `_boot_stats`, `_linprog_solve`) were used for these
measurements only.

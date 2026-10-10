# AGENTS.md

What a coding agent (or you) needs to work in this repository: the task, the
layout, the commands, the rules that decide a score, and how to evaluate
reliably.

## The task

ShockBench-Flow is a Gymnasium control task. Every week of an episode an agent
decides how much of each good to send along each route of a supply network.
Disruptions (a strait closes, a route is sanctioned, a tariff jumps, a factory
goes down) are drawn before the episode starts and nothing the agent does
changes them. The agent sees the network as it is this week, its stock and
shipments, a demand forecast, and noisy early warnings and announcements. An
episode costs money (USD); lower is better.

The score compares an agent's cost with two references on the same scenarios:
**0** is the naive rule (keep shipping the normal plan), **1** is the
clairvoyant plan (it knew every disruption in advance); below 0 is worse than
naive. The package and the board call it RSS.

- Networks: `tiny` (practice, 26 weeks, every tool's default), `small` (the
  public board's, 52 weeks), `full` (the private board's, 104 weeks). Shapes
  differ: read them from `config["spaces"]`, never hard-code Tiny's.
- A submission is a zip with `agent.py` at its root, defining
  `class Agent: __init__(self, config)` (once per episode) and
  `act(self, observation) -> {"flows": ..., ...}` (once per week).
- The details (every field, the RL wrappers, the scoring formula, the noise):
  [docs/GUIDE.md](docs/GUIDE.md) and [docs/fields/](docs/fields/).

## Layout

The repository is the participant's fork: any file may change. The benchmark
itself is the `shockbench-flow` package from PyPI (`>= 0.1.2` in
`pyproject.toml`, the exact version in `uv.lock`).

```
agents/<name>/     # one submission folder per agent: agent.py and the files it loads (weights are committed)
                   #   shipped: template (send the maximum), random, heuristic (reads params.json if present)
                   #   mine: the LP planner (package's mpc_det, vendored as sbfplan/ by scripts/vendor_planner.py,
                   #   solved by SciPy) in a frozen frame.py, with four EVOLVE blocks in agent.py
                   #   evo_champion: the evolution loop's current champion (written by examples/08_evolve.py)
                   #   twopass: mine plus a two-pass solve that enforces the grids' base-load-first rule in weeks 2..H
                   #   twopass48: twopass on SciPy's bundled HiGHS, warm-started, adding a 48-week window each week
                   #   belief48: twopass48 with a calibrated strait belief (belief.py; measured no gain, research doc 5.7)
                   #   milp48: twopass48 plus the exact base-load-first window MILP on Small (research doc 5.11)
                   #   milp48t: milp48 with the MILP's budget and options tuned (research doc 5.12)
                   #   scen48_penalized: twopass48 with shed base load x1000 in weeks 2..H (measured -0.06, doc 5.14)
                   #   twopass48_credit: twopass48 plus a terminal credit on Full when the window ends before T (doc 5.15)
                   #   milp48_eco: milp48t plus the same credit on Full (doc 5.15)
                   #   compact_hierarchical: mpc_nobuf's compact LP (z continuous, credit 0.7) with a pre-solve maritime
                   #   reroute controller and crisis-scaled fuel safety floors (compact_hier/; knobs in params.json)
                   #   (SBF_PARAM_<KEY>=... or SBF_PARAMS_FILE=params_baseline.json override; tuned by
                   #   scripts/research/tune_compact_optuna.py; Fix A/B toggles off by default, measured negative, doc 5.17)
                   #   final_dispatch: built by scripts/build_dispatch_agent.py: compact_hierarchical on Small,
                   #   twopass48_credit on Full, in one submission (rebuild after editing either source)
examples/          # 01_quickstart.py ... 07_dashboard.py, each self-contained; ppo_agent.py is the PPO submission's agent.py
                   #   08_evolve.py: the AlphaEvolve loop (docs/EVOLVE_DESIGN.md); 09_stress.py: stress tests on
                   #   stratified private episodes (levels, Full, paired, CPU and memory)
src/sbf_starter/evolve/  # the loop's pools, evaluator, feedback, prompts, mutator, CMA-ES tuner, archive
src/sbf_starter/   # the `sbf` CLI (cli.py), scoring.py, check.py (isolated timed run), container.py (--docker),
                   #   codabench.py (token, upload, status), agents.py (names -> folders), play.py (closures)
scripts/           # fields_docs.py: regenerates docs/fields/ from the installed shockbench-flow;
                   #   vendor_planner.py: copies the package's planner into an agent folder as sbfplan/
                   #   bound_hybrid_limit.py: the simulator-lookahead bound; research/: the offline experiments of
                   #   docs/research/dataset_and_rss_mechanics.md (results/ holds the outputs it cites)
docs/              # GUIDE.md, fields/ (every observation and action field), img/, research/ (findings and benchmarks)
tests/             # uv run pytest -n 3
outputs/           # run folders outputs/<example>/<date_time>/ and packed zips (gitignored)
```

## Commands

Always run Python through `uv run` (the locked environment). `AGENT` is a name
(a folder of `agents/`) or a path to a folder, a zip or an `agent.py`.

| Task                          | Command                                                                  |
| ----------------------------- | ------------------------------------------------------------------------ |
| Install (the rl extra too)    | `uv sync` (`uv sync --extra rl`)                                         |
| A new agent                   | `cp -r agents/template agents/mine`                                      |
| Score locally                 | `uv run sbf evaluate mine` (`--task=small`, `--quick`, `--episodes=...`) |
| Compare two agents            | `uv run sbf compare mine template` (a paired interval)                   |
| Check as the server does      | `uv run sbf check mine --task=small` (`--docker`: the real container)    |
| Pack a zip                    | `uv run sbf pack mine` (to `outputs/mine.zip`)                           |
| Codabench token, once         | `uv run sbf token` (the participant types the password; not an agent)   |
| Upload (only when asked)      | `uv run sbf upload mine --dry_run`, then without it                      |
| Submissions and scores        | `uv run sbf status`                                                      |
| Run an example                | `uv run python examples/0N_name.py --task=small --key=value`             |
| Add a dependency              | `uv add <package>` (training only: agent.py cannot import it)            |
| Evolve the LP agent           | `uv sync --extra evolve`, then `uv run python examples/08_evolve.py` (`--dry_run` offline) |
| Stress-test an agent          | `uv run python examples/09_stress.py --agent=mine --levels=3,4 --root=123` |
| Re-vendor the planner         | `uv run python scripts/vendor_planner.py agents/mine`, then `uv run pytest tests/test_planner_port.py` |
| Update shockbench-flow        | `uv sync --upgrade-package shockbench-flow` (never edit a version)       |
| Tests                         | `uv run pytest -n 3`                                                     |
| Lint / format                 | `make lint`                                                              |
| Every CLI command and option  | `uv run sbf --help`, `uv run sbf <command> --help`                       |

## Rules that decide a score

The full list is [docs/GUIDE.md, "Rules"](docs/GUIDE.md#rules). The ones code
must respect:

- **Imports**: only Python 3.13's standard library, numpy, SciPy and PyTorch
  (CPU) exist on the server, in `agent.py` and in every module it imports.
  Anything else in this environment (gymnasium, Stable-Baselines3, pandas,
  `sbf_starter`) is for training only. `sbf check` fails a violation.
- **CPU per week**: 2 s on the public board (Small), 4 s on the private board
  (Full). `Agent(config)` counts toward week 1; load weights at module level. A
  week over budget, crashed or malformed is played by the naive rule.
- **Seeding**: seed every random generator from `config["policy_seed"]`.
- **Files**: load them relative to `Path(__file__).parent`; at most 500 MB
  unpacked and 1,000 files; `print` goes to stderr.
- **Never upload unless the participant asks**: each upload spends one of the
  day's 3 submissions. Use `--dry_run` to test. Never print or commit `.env`
  (it holds `CODABENCH_TOKEN`).

## Evaluating reliably

- A score on 20 episodes is noisy (one standard error 0.17 to 0.20). To decide
  whether a change helped, run `uv run sbf compare new old`: a paired interval
  that holds 0 means the episodes cannot tell them apart.
- Tune on a root of your own (`--entropy=12345 --episodes=64`, or
  `EpisodeSet.build(task, 64, entropy=12345)` in Python) and keep the dev
  episodes (root 0) for confirmation; a search that sees only the dev episodes
  fits them.
- `--quick` is for smoke tests: a rough naive rule, no harm levels, not the
  board's numbers.
- `sbf check` times `act` on this machine and `--cpu_budget` meters it here; the
  server meters its own (`sbf check --docker` is the closest local copy).

## Working in this repository

- Examples are self-contained scripts: the options are the keyword arguments of
  `main`, exposed as `--flags` by `fire.Fire(main)`, with defaults that run on
  Tiny. A new experiment starts as a copy of the closest example and keeps that
  pattern; files it writes go under `outputs/<name>/<date_time>/`. Fire reads
  `--x=False` as False but `--x=false` as the string "false": pass `--nox`.
- An agent is a folder of `agents/` with `agent.py` at its root. The
  `heuristic` agent reads its numbers from a `params.json` beside it when there
  is one (`examples/06_policy_search.py` writes one).
- Draw randomness from local seeded generators (`np.random.default_rng(seed)`).
- `agents/mine/frame.py` and `agents/mine/sbfplan/` are frozen: the loop rejects candidates whose copies differ,
  and ruff skips `sbfplan/`. Change the frame deliberately, then start a new evolution run.
- Verify before reporting done: `uv run sbf check <agent> --task=small` for an
  agent, a run with small settings for an example, `uv run pytest -n 3` for code.
  If a check cannot run (no Docker, no network), say so.

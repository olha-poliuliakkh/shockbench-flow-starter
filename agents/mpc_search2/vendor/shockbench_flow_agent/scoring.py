"""Local scoring for evaluation and search loops: a fixed episode set, its references cached, ``score``, ``compare``.

::

    from shockbench_flow_agent import EpisodeSet

    episodes = EpisodeSet.build("tiny", "dev")      # the leaderboard's local dev split; references cached on disk
    result = episodes.score(MyAgent)                # an Agent class, a factory, a folder, a zip or an agent.py
    print(result)                                   # plain-language report
    result.rss, result.interval, result.fallback_weeks, result.cost_usd
    print(episodes.compare(MyAgent, OtherAgent))    # paired bootstrap on the same episodes

    EpisodeSet.build("tiny", "dev", quick=True)     # seconds instead of minutes: not the leaderboard's numbers

What a score is. On every episode the agent plays in this process, as ``shockbench_flow_agent.evaluate`` plays an
``Agent`` class (``mode='in_process'``, whose trajectory the scoring container reproduces bit for bit): the same
scenario, the same policy seed, the same naive fallback for a week whose ``act`` fails. Its cost J is compared with two
references that do not depend on the agent: the naive rule (score 0) and the clairvoyant plan, which knows the whole
scenario (score 1). The score is the leaderboard's: the scorer's own function and harm-level weights, pooled over the
harm levels when every level has an episode, else over all episodes (``Score.pooled`` says which). An episode whose
clairvoyant plan is not solved to optimality is left out, as on the leaderboard.

- **Episodes**: ``"dev"`` is the leaderboard's local dev split exactly as ``evaluate`` and the scorer define it (the
  harm cut points from 2,000 draws on their own public root, then episodes n = 0, 1, ... of the public dev root,
  keeping the first 5 of each harm level in index order; 20 on every task, ``[0..12, 15, 17, 20, 22, 23, 45, 48]`` on
  `tiny`). A count k is episodes 0..k-1 and a list names them. ``entropy`` other than the dev root (0) is a training
  root of your own (any integer below 2**119).
- **Quick** (``quick=True``): a rough naive rule (``QUICK``: 2 replications of its demand model instead of
  ``NAIVE_REPLICATIONS``) and no harm levels, so ``"dev"`` becomes the first ``QUICK_EPISODES`` episodes. Seconds
  instead of minutes, but not the leaderboard's numbers (``Score.quick`` says so, and the report).
- **References**: naive's and the clairvoyant plan's costs per episode, computed once (in parallel, ``n_jobs``) and
  kept on disk under ``<cache>/references/`` (``SBF_CACHE_DIR``, else ``~/.cache/shockbench-flow``), keyed by the
  package version and source digest, the generator, the root and naive's replications; ``cache_dir=False`` keeps them
  in memory only. The dev split's indices are cached beside them.
- **Interval**: a percentile bootstrap over the choice of episodes (resampled within each harm level, the same
  weights): how far the score would move on another draw of as many episodes. ``compare`` resamples both agents on the
  same draws (paired), so a difference is judged against its own noise.
- **CPU budget** (``cpu_budget``): optional. ``True`` meters each week at the task's budget (``CPU_BUDGET_S``, the
  server's CPU seconds per week, ``LIMITS.cpu_budget_s``) and a float at that many seconds: a week whose CPU time in
  this process (``time.process_time``, the agent's ``act`` and the kit's conversions, ``Agent(config)`` charged to
  week 1) is over the budget is played by the naive rule and counted, as the server does; the excess is charged to the
  next week, as the server's meter charges an ``act`` still running when its week is cut. The server meters the
  container on its own machine: a local meter is a guide, not the server's count.

Folders and zips are checked by the server's own validator first and their policy seed is salted by the zip's
SHA-256, so a changed file changes the seed; a class, a factory or an ``agent.py`` file (loaded into this process, so
a debugger works) is salted by the empty file's. Nothing here runs with the scorer's secrets set.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from shockbench_flow.hosting.limits import LIMITS as _LIMITS


DEV = "dev"  # the episodes keyword of the leaderboard's local dev split
LEVEL = 0.90  # the bootstrap interval's coverage (not a model value)
N_BOOT = 2000  # bootstrap resamples (not a model value)
# CPU seconds per week the server meters (``LIMITS.cpu_budget_s``): the Development phase plays `small` at 2 s, the
# Final `full` at 4 s; `tiny` is not hosted and takes the Development budget
CPU_BUDGET_S = dict(_LIMITS.cpu_budget_s)
# ``quick=True``: a rough naive rule and no harm levels (module docstring), and the dev split's stand-in
QUICK = MappingProxyType({"fq_replications": 2, "cut_draws": 0})
QUICK_EPISODES = 4
QUICK_NOTE = "quick: a rough naive rule and no harm levels, so these are not the leaderboard's numbers"
REFERENCE_DIR = "references"  # under the package cache directory
NO_CACHE = False


# ----- quiet logs ---------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def quiet_logs(verbose: bool = False) -> Iterator[None]:
    """Kept for code written for 0.1.1: the packages are quiet by default since 0.1.2.

    ``shockbench_flow`` disables its loguru lines when it is imported (loguru's convention for libraries); an
    application that wants them calls ``loguru.logger.enable("shockbench_flow")``. ``quiet_logs()`` leaves that choice
    alone; ``quiet_logs(verbose=True)`` lets the lines through inside the block and turns them off again after it.
    """
    if not verbose:
        yield
        return
    from loguru import logger

    logger.enable("shockbench_flow")
    try:
        yield
    finally:
        logger.disable("shockbench_flow")


def _say(verbose: bool, text: str) -> None:
    if verbose:
        print(text, flush=True)


# ----- the world of an episode --------------------------------------------------------------------------------------
def _label(entropy: int) -> str:
    from shockbench_flow.hosting.tasks import split_label

    return split_label(entropy)


@functools.lru_cache(maxsize=256)  # the worlds of recent episodes (not a model value)
def _world(task: str, entropy: int, n: int, fq_replications: int, cache: str | None) -> tuple:
    """(instance, omega, marks, naive fallback) of episode n, as ``local_eval.score_agent`` builds them; per process."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.evaluation.cache import fq_quantiles
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.policies.naive_fq import fallback_spec

    inst, params = task_generator(task)
    if cache is not None:
        fq_quantiles(inst, params, fq_replications, cache_dir=cache)  # a disk hit fills this process's memo
    fallback = fallback_spec(inst, params, fq_replications)
    omega = sample_omega(inst, params, entropy, n, _label(entropy))
    return inst, omega, compute_marks(inst, omega), fallback


def _policy_seed(entropy: int, n: int, sha256: str) -> int:
    from shockbench_flow.omega.seeds import policy_seed

    return policy_seed(entropy, _label(entropy), n, sha256)


def _reference_row(task: str, entropy: int, n: int, fq_replications: int, cache: str | None) -> dict:
    """Naive's and the clairvoyant plan's costs of episode n with its harm and exclusion (``score_agent``'s rule)."""
    from shockbench_flow.disruption.harm import episode_harm
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.results import exclusion_cause, oracle_optimal
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.omega.injected import event_free
    from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, solve_oracle
    from shockbench_flow.policies.naive_fq import anchor_policy
    from shockbench_flow_agent.local_eval import ANCHOR_REGIME, NO_ZIP_SHA256

    start = time.perf_counter()
    inst, omega, marks, fallback = _world(task, entropy, n, fq_replications, cache)
    _inst, params = task_generator(task)
    anchor = anchor_policy(inst, params, fq_replications)
    pseed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    naive = rollout(inst, anchor, omega, ANCHOR_REGIME, pseed, marks=marks, fallback=fallback)
    oracle = solve_oracle(build_lp(inst, marks), method=ORACLE_METHOD)
    oracle0 = solve_oracle(build_lp(inst, compute_marks(inst, event_free(omega, inst))), method=ORACLE_METHOD)
    return {
        "episode": n,
        "omega_hash": omega.hash,
        "generator_id": omega.generator_id,
        "harm_usd": episode_harm(inst, omega),
        "J_naive_cents": naive.J_cents,
        "J_oracle_cents": oracle.J_cents if oracle_optimal(oracle) else None,
        "excluded": exclusion_cause(oracle, oracle0),
        "seconds": round(time.perf_counter() - start, 3),
    }


def _cache_base(cache: Path, task: str, entropy: int, fq_replications: int) -> Path:
    import shockbench_flow
    from shockbench_flow.disruption.params import generator_id
    from shockbench_flow.evaluation.cache import package_sha256
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.oracle.lp import ORACLE_METHOD

    inst, params = task_generator(task)
    version = f"v{shockbench_flow.__version__}-{package_sha256()[:12]}"
    gid = generator_id(params, inst)[:16]
    return cache / REFERENCE_DIR / version / task / gid / f"entropy-{entropy}" / f"fq-{fq_replications}-{ORACLE_METHOD}"


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _episode_indices(episodes: str | int | Sequence[int]) -> list[int] | None:
    from shockbench_flow_agent.local_eval import episode_list

    return episode_list(episodes)


# ----- the bootstrap ------------------------------------------------------------------------------------------------
def _kept(rows: Sequence[dict]) -> list[int]:
    return [i for i, r in enumerate(rows) if r["excluded"] is None]


def _groups(rows: Sequence[dict], kept: Sequence[int], pooled: bool) -> list[tuple[float, np.ndarray]]:
    """(weight, row indices) per stratum when ``pooled``, else one group of weight 1."""
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    if not pooled:
        return [(1.0, np.array(kept, dtype=int))]
    out = []
    for s, w in enumerate(STRATUM_WEIGHTS, start=1):
        idx = np.array([i for i in kept if rows[i]["stratum"] == s], dtype=int)
        if w > 0 and idx.size:
            out.append((float(w), idx))
    return out


def _boot_stats(
    rows: Sequence[dict], costs: Sequence[Sequence[int]], pooled: bool, n_boot: int, seed: int
) -> list[np.ndarray]:
    """For each cost vector in ``costs``, its RSS on ``n_boot`` stratified resamples of the kept episodes (paired)."""
    kept = _kept(rows)
    groups = _groups(rows, kept, pooled)
    rng = np.random.default_rng(seed)
    naive = np.array([r["J_naive_cents"] for r in rows], dtype=float)
    oracle = np.array([r["J_oracle_cents"] if r["J_oracle_cents"] is not None else np.nan for r in rows], dtype=float)
    picks = [(w, idx[rng.integers(0, idx.size, size=(n_boot, idx.size))]) for w, idx in groups]
    out = []
    for J in costs:
        J = np.array(J, dtype=float)
        num = sum(w * (naive[p] - J[p]).mean(axis=1) for w, p in picks)
        den = sum(w * (naive[p] - oracle[p]).mean(axis=1) for w, p in picks)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(den > 0, num / den, np.nan))
    return out


def _interval(values: np.ndarray, level: float) -> tuple[float, float] | None:
    v = values[np.isfinite(values)]
    if v.size < 2:
        return None
    lo, hi = np.quantile(v, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


# ----- results ------------------------------------------------------------------------------------------------------
def _usd(cents: float | None) -> str:
    return "-" if cents is None else f"${cents / 100:,.0f}"


def _fmt(x: float | None, digits: int = 4) -> str:
    return "undefined" if x is None else f"{x:.{digits}f}"


@dataclass(frozen=True)
class Score:
    """One agent's score on an ``EpisodeSet``; ``str(score)`` is the plain-language report.

    ``rss`` is the leaderboard's score (0 the naive rule, 1 the clairvoyant plan), pooled over the harm levels when
    ``pooled`` (the leaderboard's weights), else over all episodes; ``interval`` its bootstrap interval at ``level``;
    ``fallback_weeks`` the weeks the naive rule played for the agent (``cpu_weeks`` of them over the CPU budget);
    ``cost_usd`` the agent's mean cost per episode in dollars, beside ``naive_cost_usd`` and
    ``clairvoyant_cost_usd`` (``oracle_cost_usd``, its older name). ``rows`` has one dict per episode: ``episode``,
    ``stratum`` (the harm level), ``J_policy_cents``, ``J_naive_cents``, ``J_clairvoyant_cents`` (``J_oracle_cents``,
    the same), ``fallback_weeks``, ``cpu_weeks``, ``first_error`` and more. ``quick`` is True when the episode set is
    not on the leaderboard's settings (``EpisodeSet.build(quick=True)``, or other replications or cut draws).
    """

    agent: str
    task: str
    regime: str
    rss: float | None
    pooled: bool
    rss_by_stratum: dict[int, float | None]
    rss_all: float | None
    interval: tuple[float, float] | None
    level: float
    episodes: int
    excluded: tuple[int, ...]
    fallback_weeks: int
    cpu_weeks: int
    weeks: int
    invalid_entries: int
    cost_usd: float
    naive_cost_usd: float
    oracle_cost_usd: float | None
    cpu_budget_s: float | None
    rows: tuple[dict, ...] = field(repr=False)
    boot: np.ndarray | None = field(default=None, repr=False, compare=False)
    quick: bool = False

    @property
    def J_cents(self) -> list[int]:
        """The agent's cost per episode in integer cents, in the set's order."""
        return [r["J_policy_cents"] for r in self.rows]

    @property
    def clairvoyant_cost_usd(self) -> float | None:
        """The clairvoyant plan's mean cost per episode in dollars (``oracle_cost_usd``, its older name)."""
        return self.oracle_cost_usd

    def __str__(self) -> str:
        return score_report(self)


@dataclass(frozen=True)
class Comparison:
    """Two agents on the same episodes: ``diff`` = a's score minus b's, a paired bootstrap interval, ``p_a_better``.

    ``p_a_better`` is the share of bootstrap resamples on which a scores above b (not a p-value; near 0.5 means the
    episodes cannot tell them apart).
    """

    a: Score
    b: Score
    diff: float | None
    interval: tuple[float, float] | None
    level: float
    p_a_better: float | None

    def __str__(self) -> str:
        return compare_report(self)


# ----- the CPU meter ------------------------------------------------------------------------------------------------
def _metered_shim(agent_class: Callable[[dict], object], budget_s: float | None):
    """``AgentShim`` whose weeks over ``budget_s`` CPU seconds give a null action (module docstring, 'CPU budget')."""
    from shockbench_flow_agent.shim import AgentShim

    class MeteredShim(AgentShim):
        def __init__(self) -> None:
            super().__init__(agent_class)
            self.carry, self.cpu_weeks = 0.0, []

        def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
            start = time.process_time()
            super().reset(static, obs, policy_seed)
            self.carry = time.process_time() - start  # Agent(config) and the kit's reading: week 1's

        def act(self, obs: dict) -> dict | None:
            start = time.process_time()
            action = super().act(obs)
            used = self.carry + time.process_time() - start
            if budget_s is not None and used > budget_s:
                self.cpu_weeks.append(obs["week"])
                self.carry = used - budget_s
                return None
            self.carry = 0.0
            return action

    return MeteredShim()


def _play(
    agent: object, root: str | None, sha256: str, spec: tuple, episodes: Sequence[int], budget_s: float | None
) -> list[dict]:
    """The agent's rows on ``episodes`` in this process: a class or factory, or ``root`` (a folder or an agent file)."""
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq_replications, cache = spec

    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq_replications, cache)
        with without_secret_like():  # the agent sees no API key or token (local_eval.play)
            factory = agent if root is None else load_agent_class(root, f"submission_{Path(root).stem}")
            shim = _metered_shim(factory, budget_s)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, sha256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        if root is not None:
            unload_agent()
        fell = sum(took_fallback(r) for r in traj.records)
        rows.append(
            {
                "episode": n,
                "J_policy_cents": traj.J_cents,
                "fallback_weeks": fell,
                "cpu_weeks": len(shim.cpu_weeks),
                "invalid_entries": sum(len(r.invalid) for r in traj.records if not took_fallback(r)),
                "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                "weeks": inst.T,
                "seconds": round(seconds, 3),
                "omega_hash": omega.hash,
            }
        )
    return rows


# ----- the episode set ----------------------------------------------------------------------------------------------
@dataclass
class EpisodeSet:
    """A fixed set of episodes of one task with naive's and the clairvoyant plan's costs (module docstring).

    Build it once with ``EpisodeSet.build``; then ``score`` costs one rollout of the agent per episode and ``compare``
    two. ``references`` has one dict per episode (``episode``, ``stratum``, ``harm_usd``, ``J_naive_cents``,
    ``J_oracle_cents`` and its alias ``J_clairvoyant_cents``, ``excluded``). ``quick`` is True when these are not the
    leaderboard's settings (naive's replications and the harm cut points' draws).
    """

    task: str
    regime: str
    entropy: int
    episodes: tuple[int, ...]
    references: tuple[dict, ...]
    fq_replications: int
    cut_draws: int
    cache_dir: Path | None
    verbose: bool = False
    quick: bool = False

    @classmethod
    def build(
        cls,
        task: str = "tiny",
        episodes: str | int | Sequence[int] = DEV,
        *,
        regime: str = "standard",
        entropy: int = 0,
        fq_replications: int | None = None,
        cut_draws: int | None = None,
        n_jobs: int = -1,
        cache_dir: str | Path | bool | None = None,
        verbose: bool = False,
        quick: bool = False,
    ) -> EpisodeSet:
        """The episodes of ``task``, their references read from the cache or computed once (``n_jobs`` workers).

        Args:
            task: a public task (``tiny``, ``small``, ``full``).
            episodes: ``"dev"``, the leaderboard's local dev split (the dev root only); a count k, episodes 0..k-1;
                or a list of episode indices.
            regime: the information regime the agent plays (``standard``, the scored one).
            entropy: the root the scenarios are drawn from: 0 is the public dev root; any other integer below 2**119
                is a training root of your own.
            fq_replications: naive's demand-model replications (default ``NAIVE_REPLICATIONS``, 1,000, the
                leaderboard's; ``QUICK``'s 2 with ``quick``).
            cut_draws: the harm cut points' draws (default 2,000, the leaderboard's; 0 for no harm levels, so the
                score is over all episodes and ``"dev"`` is not available; ``QUICK``'s 0 with ``quick``).
            n_jobs: joblib workers of a first run (-1 all cores); no number depends on it.
            cache_dir: the cache directory (None: ``SBF_CACHE_DIR``, else ``~/.cache/shockbench-flow``; False: none).
            verbose: print progress lines.
            quick: seconds instead of minutes (module docstring, 'Quick'): ``QUICK``'s replications and cut draws
                where ``fq_replications`` and ``cut_draws`` are not given, and ``"dev"`` the first ``QUICK_EPISODES``
                episodes of ``entropy``. Not the leaderboard's numbers.

        Raises:
            ValueError: while a scorer secret is set, on an unknown task or regime, ``episodes`` outside its forms,
                ``"dev"`` with another root or with ``cut_draws=0`` (without ``quick``), or a root of the hidden
                split's size.

        """
        from shockbench_flow.disruption.strata import stratum
        from shockbench_flow.evaluation.cache import cut_points_cached, default_cache_dir, fq_quantiles
        from shockbench_flow.hosting.split import CUT_ENTROPY, DEV_ENTROPY
        from shockbench_flow.hosting.tasks import CUT_DRAWS, DEV_PER_STRATUM, get_task, task_generator
        from shockbench_flow.information.theta import REGIME_NAMES
        from shockbench_flow.policies.naive_fq import REPLICATIONS
        from shockbench_flow_agent.local_eval import SPLIT, check_local_run, dev_episodes

        defaults = QUICK if quick else {"fq_replications": REPLICATIONS, "cut_draws": CUT_DRAWS}
        fq_replications = defaults["fq_replications"] if fq_replications is None else fq_replications
        cut_draws = defaults["cut_draws"] if cut_draws is None else cut_draws
        if quick and isinstance(episodes, str) and episodes == DEV:
            episodes = QUICK_EPISODES
        check_local_run(SPLIT, entropy)
        get_task(task)
        if regime not in REGIME_NAMES:
            raise ValueError(f"regime must be one of {REGIME_NAMES}, got {regime!r}")
        ns = _episode_indices(episodes)
        if ns is None and entropy != DEV_ENTROPY:
            raise ValueError("episodes='dev' is the dev split of the public dev root: pass a count or a list")
        if ns is None and not cut_draws:
            raise ValueError("episodes='dev' fills the harm strata: it needs cut_draws >= 1")
        cache = None if cache_dir is False else (default_cache_dir() if cache_dir is None else Path(cache_dir))
        inst, params = task_generator(task)
        t = time.perf_counter()
        fq_quantiles(inst, params, fq_replications, n_jobs=n_jobs, cache_dir=cache)
        _say(verbose, f"naive rule's quantiles ready ({time.perf_counter() - t:.1f} s)")
        cuts = cut_points_cached(inst, params, cut_draws, CUT_ENTROPY, n_jobs, cache) if cut_draws else None
        base = None if cache is None else _cache_base(cache, task, entropy, fq_replications)
        if ns is None:
            key = f"dev-{cut_draws}-{CUT_ENTROPY}-{'-'.join(map(str, DEV_PER_STRATUM))}.json"
            ns = _read_json(base / key) if base is not None else None
            if not isinstance(ns, list):
                ns = dev_episodes(task, cuts, n_jobs)
                if base is not None:
                    _write_json(base / key, ns)
        rows = cls._references(task, entropy, ns, fq_replications, cache, base, n_jobs, verbose)
        for row in rows:
            row["stratum"] = None if cuts is None else stratum(row["harm_usd"], cuts, generator_id=row["generator_id"])
            row["J_clairvoyant_cents"] = row["J_oracle_cents"]
        board = (fq_replications, cut_draws) == (REPLICATIONS, CUT_DRAWS)
        return cls(task, regime, entropy, tuple(ns), tuple(rows), fq_replications, cut_draws, cache, verbose, not board)

    @staticmethod
    def _references(task, entropy, ns, fq_replications, cache, base, n_jobs, verbose) -> list[dict]:
        from joblib import Parallel, delayed

        found = {}
        if base is not None:
            for n in ns:
                row = _read_json(base / f"{n}.json")
                if isinstance(row, dict):
                    found[n] = row
        missing = [n for n in ns if n not in found]
        if missing:
            _say(verbose, f"computing the naive rule and the clairvoyant plan on {len(missing)} episode(s) ...")
            t = time.perf_counter()
            key = None if cache is None else str(cache)
            jobs = 1 if (len(missing) == 1 or cache is None) else n_jobs  # workers read the quantiles from disk
            rows = Parallel(n_jobs=jobs)(
                delayed(_reference_row)(task, entropy, n, fq_replications, key) for n in missing
            )
            for row in rows:
                found[row["episode"]] = row
                if base is not None:
                    _write_json(base / f"{row['episode']}.json", row)
            _say(verbose, f"  done in {time.perf_counter() - t:.1f} s" + (f", cached in {base}" if base else ""))
        return [dict(found[n]) for n in ns]

    # ----- scores -----------------------------------------------------------------------------------------------
    @property
    def _spec(self) -> tuple:
        return (
            self.task,
            self.entropy,
            self.regime,
            self.fq_replications,
            None if self.cache_dir is None else str(self.cache_dir),
        )

    def _budget(self, cpu_budget: bool | float | None) -> float | None:
        if cpu_budget is None or cpu_budget is False:
            return None
        if cpu_budget is True:
            return CPU_BUDGET_S[self.task]
        if isinstance(cpu_budget, (int, float)) and math.isfinite(cpu_budget) and cpu_budget > 0:
            return float(cpu_budget)
        raise ValueError(
            f"cpu_budget must be False, True (the task's) or a positive number of seconds, got {cpu_budget!r}"
        )

    def play(self, agent: object, *, cpu_budget: bool | float | None = False, n_jobs: int = 1) -> list[dict]:
        """The agent's per-episode rows (``J_policy_cents``, ``fallback_weeks``, ``cpu_weeks``, ...), in order.

        ``agent`` as ``score``'s. ``n_jobs`` > 1 plays episodes in joblib workers (a class or factory must pickle; a
        folder, a zip or an agent file always does).

        Raises:
            SubmissionError: when the server's validator refuses a folder or zip.
            ValueError: while a scorer secret is set, or on a ``cpu_budget`` outside its forms.

        """
        from joblib import Parallel, delayed

        from shockbench_flow_agent.local_eval import NO_ZIP_SHA256, SPLIT, agent_file, check_local_run, resolve_agent
        from shockbench_flow_agent.submission import Submission

        check_local_run(SPLIT, self.entropy)
        budget = self._budget(cpu_budget)
        with tempfile.TemporaryDirectory(prefix="sbf-score-") as work:
            file = agent_file(agent)
            resolved = None if file is not None else resolve_agent(agent, Path(work))[0]
            if file is not None:  # loaded in each process that plays it, seeded as a class is
                target, root, sha = None, str(file), NO_ZIP_SHA256
            elif isinstance(resolved, Submission):
                target, root, sha = None, str(resolved.root), resolved.sha256
            else:
                target, root, sha = resolved, None, NO_ZIP_SHA256
            ns = list(self.episodes)
            jobs = 1 if self.cache_dir is None else n_jobs
            if jobs == 1 or len(ns) == 1:
                rows = _play(target, root, sha, self._spec, ns, budget)
            else:
                chunks = [c for c in np.array_split(np.array(ns), min(len(ns), _workers(jobs))) if c.size]
                parts = Parallel(n_jobs=jobs)(
                    delayed(_play)(target, root, sha, self._spec, [int(n) for n in c], budget) for c in chunks
                )
                rows = [r for part in parts for r in part]
        for ref, row in zip(self.references, rows):
            if ref["omega_hash"] != row["omega_hash"]:
                raise RuntimeError(f"episode {ref['episode']}: the cached reference was computed on another scenario")
        return rows

    def rss(self, J_cents: Sequence[int]) -> dict:
        """The leaderboard's RSS table of per-episode costs in integer cents (``evaluation.results.episode_rss``)."""
        from shockbench_flow.evaluation.results import episode_rss

        records = [{**ref, "J_policy_cents": int(j)} for ref, j in zip(self.references, J_cents, strict=True)]
        stratified = all(r["stratum"] is not None for r in records)
        one = episode_rss([{**r, "stratum": 1} for r in records], (1.0,))
        table = episode_rss(records) if stratified else None
        pooled = table is not None and table["pooled"] is not None
        return {
            "rss": table["pooled"] if pooled else one["strata"]["1"]["rss"],
            "pooled": pooled,
            "rss_all": one["strata"]["1"]["rss"],
            "rss_by_stratum": {int(s): v["rss"] for s, v in table["strata"].items()} if table else {},
        }

    def _score(self, name: str, rows: list[dict], budget: float | None, level: float, boot: np.ndarray) -> Score:
        table = self.rss([r["J_policy_cents"] for r in rows])
        merged = tuple(
            {**ref, **row, "J_clairvoyant_cents": ref["J_oracle_cents"]} for ref, row in zip(self.references, rows)
        )
        oracle = [r["J_oracle_cents"] for r in merged if r["J_oracle_cents"] is not None]
        return Score(
            agent=name,
            task=self.task,
            regime=self.regime,
            rss=table["rss"],
            pooled=table["pooled"],
            rss_by_stratum=table["rss_by_stratum"],
            rss_all=table["rss_all"],
            interval=_interval(boot, level),
            level=level,
            episodes=len(merged),
            excluded=tuple(r["episode"] for r in merged if r["excluded"] is not None),
            fallback_weeks=sum(r["fallback_weeks"] for r in merged),
            cpu_weeks=sum(r["cpu_weeks"] for r in merged),
            weeks=sum(r["weeks"] for r in merged),
            invalid_entries=sum(r["invalid_entries"] for r in merged),
            cost_usd=float(np.mean([r["J_policy_cents"] for r in merged])) / 100,
            naive_cost_usd=float(np.mean([r["J_naive_cents"] for r in merged])) / 100,
            oracle_cost_usd=float(np.mean(oracle)) / 100 if len(oracle) == len(merged) else None,
            cpu_budget_s=budget,
            rows=merged,
            boot=boot,
            quick=self.quick,
        )

    def score(
        self,
        agent: object,
        *,
        name: str | None = None,
        cpu_budget: bool | float | None = False,
        n_jobs: int = 1,
        level: float = LEVEL,
        n_boot: int = N_BOOT,
        seed: int = 0,
    ) -> Score:
        """Score ``agent``: an ``Agent`` class, a factory ``config -> agent``, a folder, a zip or an ``agent.py``.

        Args:
            agent: what to score. A folder (holding ``agent.py``) or a zip, as a ``str`` or a ``pathlib.Path``, is
                checked by the server's validator first and seeded by its zip's SHA-256; a path to a ``.py`` file is
                that file's ``Agent`` class, loaded into this process (a debugger works) and seeded as a class is.
            name: the name the report and ``Score.agent`` give it (default: the path, or the class's name).
            cpu_budget: False (no meter), True (the task's budget, ``CPU_BUDGET_S``) or CPU seconds per week.
            n_jobs: joblib workers for the agent's episodes (1: in this process).
            level: the bootstrap interval's coverage.
            n_boot: bootstrap resamples.
            seed: the bootstrap's seed (the scores themselves do not depend on it).

        Raises:
            SubmissionError: when the validator refuses a folder or zip.
            ValueError: while a scorer secret is set, or on a ``cpu_budget`` outside its forms.

        """
        from shockbench_flow_agent.local_eval import _agent_name

        rows = self.play(agent, cpu_budget=cpu_budget, n_jobs=n_jobs)
        pooled = self.rss([r["J_policy_cents"] for r in rows])["pooled"]
        (boot,) = _boot_stats(self.references, [[r["J_policy_cents"] for r in rows]], pooled, n_boot, seed)
        if name is None:
            name = str(agent) if isinstance(agent, (str, os.PathLike)) else _agent_name(agent)
        return self._score(str(name), rows, self._budget(cpu_budget), level, boot)

    def compare(
        self,
        a: object,
        b: object,
        *,
        names: Sequence[str] | None = None,
        cpu_budget: bool | float | None = False,
        n_jobs: int = 1,
        level: float = LEVEL,
        n_boot: int = N_BOOT,
        seed: int = 0,
    ) -> Comparison:
        """Score ``a`` and ``b`` on these episodes and compare them with a paired bootstrap (``Comparison``).

        Both resample the same episodes on every draw, so the interval of a's score minus b's leaves out the noise
        both share. ``names``: the two names the report gives them (default: as ``score`` names each). The other
        arguments as ``score``'s.

        Raises:
            ValueError: on ``names`` that is not two names, and as ``score``.

        """
        if names is not None and (isinstance(names, str) or len(names) != 2):
            raise ValueError(f"names must be two names, one per agent, got {names!r}")
        na, nb = (None, None) if names is None else names
        sa = self.score(a, name=na, cpu_budget=cpu_budget, n_jobs=n_jobs, level=level, n_boot=n_boot, seed=seed)
        sb = self.score(b, name=nb, cpu_budget=cpu_budget, n_jobs=n_jobs, level=level, n_boot=n_boot, seed=seed)
        pooled = sa.pooled and sb.pooled
        ba, bb = _boot_stats(self.references, [sa.J_cents, sb.J_cents], pooled, n_boot, seed)
        d = ba - bb
        ok = d[np.isfinite(d)]
        diff = None if sa.rss is None or sb.rss is None or sa.pooled != sb.pooled else sa.rss - sb.rss
        return Comparison(sa, sb, diff, _interval(d, level), level, float((ok > 0).mean()) if ok.size else None)


def _workers(n_jobs: int) -> int:
    from joblib import effective_n_jobs

    return max(1, effective_n_jobs(n_jobs))


# ----- the reports --------------------------------------------------------------------------------------------------
SCALE = "0 = the naive rule, 1 = the clairvoyant plan; higher is better"


def _weights_text() -> str:
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    return ", ".join(f"{w:.0%}" for w in STRATUM_WEIGHTS)


def _strata_text(by: dict) -> str:
    return ", ".join(f"{s}: {_fmt(v)}" for s, v in by.items())


def score_report(s: Score) -> str:
    """The plain-language report of a ``Score``."""
    how = "each harm level weighted as on the leaderboard" if s.pooled else "all episodes counted alike"
    lines = [
        f"{s.agent} on {s.task}, {s.episodes} episodes, information regime {s.regime}",
        f"Score: {_fmt(s.rss)}  ({SCALE})",
    ]
    if s.quick:
        lines.insert(1, f"NOTE: {QUICK_NOTE}")
    if s.interval:
        lines.append(
            f"  {s.level:.0%} interval over the choice of episodes: {s.interval[0]:.4f} to {s.interval[1]:.4f}"
        )
    lines.append(f"  {how}")
    if s.rss_by_stratum:
        lines.append(
            f"  by harm level (1 calmest .. 4 most harmful; weights {_weights_text()}): "
            f"{_strata_text(s.rss_by_stratum)}"
        )
    lines.append(
        f"Mean cost per episode: your agent {_usd(s.cost_usd * 100)}, naive rule {_usd(s.naive_cost_usd * 100)}, "
        f"clairvoyant plan {_usd(None if s.oracle_cost_usd is None else s.oracle_cost_usd * 100)} (lower is better)"
    )
    lines.append(f"Weeks the naive rule played for your agent: {s.fallback_weeks} of {s.weeks}")
    if s.cpu_budget_s is not None:
        lines.append(f"  of which over the CPU budget of {s.cpu_budget_s:g} s per week: {s.cpu_weeks}")
    errors = [r for r in s.rows if r["first_error"]]
    if errors:
        lines.append(f"  first error: episode {errors[0]['episode']}, {errors[0]['first_error']}")
    if s.excluded:
        lines.append(
            f"Left out of the score (the clairvoyant plan was not solved exactly): episodes {list(s.excluded)}"
        )
    if s.invalid_entries:
        lines.append(f"{s.invalid_entries} action entries were ignored as invalid (see the action contract)")
    return "\n".join(lines)


def compare_report(c: Comparison) -> str:
    """The plain-language report of a ``Comparison``."""
    lines = [
        f"A: {c.a.agent}  score {_fmt(c.a.rss)}",
        f"B: {c.b.agent}  score {_fmt(c.b.rss)}",
        f"A - B: {_fmt(c.diff)}",
    ]
    if c.interval:
        lines.append(
            f"  {c.level:.0%} paired interval over the choice of episodes: {c.interval[0]:.4f} to {c.interval[1]:.4f}"
        )
    if c.p_a_better is not None:
        lines.append(f"  A scores above B on {c.p_a_better:.0%} of the resampled episode sets")
    if c.interval and c.interval[0] <= 0 <= c.interval[1]:
        lines.append("  the interval holds 0: these episodes cannot tell the two apart")
    if c.a.quick or c.b.quick:
        lines.append(f"NOTE: {QUICK_NOTE}")
    return "\n".join(lines)


__all__ = [
    "CPU_BUDGET_S",
    "QUICK",
    "QUICK_EPISODES",
    "QUICK_NOTE",
    "Comparison",
    "EpisodeSet",
    "Score",
    "compare_report",
    "quiet_logs",
    "score_report",
]

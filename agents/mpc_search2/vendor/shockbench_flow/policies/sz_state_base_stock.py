"""`sz_state_base_stock`: base stock conditioned on the observed chokepoint state and queue (M4, "heuristics").

A base-stock vector per risk bucket, (R2.7) conditioned on the observed chokepoint state and queue (design §8.2
table; Song-Zipkin 1996; R2 §2.5; Q17, Q100 (3)).

The rule as read (design §12 M4 rows "sz_state_base_stock" and "`sz_state_base_stock` as built"): naive's plan,
steps 4-6, cover cap and end-aware cut (71)-(72), with the closure term of (67) on each lane route through (c, b)
replaced each week by the critical-ratio quantile of (66) run forward from the observed pool queue:

- bucket of c at week t: "closed" if ``graph_now.open[c]`` <= ``closed_threshold`` (0.6, naive's cut), "open" above
  it, "unobserved" when null (blackout, coverage);
- Q_obs = the summed qty of ``queue_lots`` at c whose commodity is in pool b;
- the conditional law pools ``naive_fq.backlog_path(Q_obs, d_cb m^sea(t), k_c mu_cb, o^tr[r, s:s+H])`` over the
  replications r and start weeks s whose o^tr_{c, s-1} lies in the week's bucket, H = tau_l + 1 truncated at T; o^tr
  are the transient paths of naive's own stream-22 replications (``transient_paths``: ``naive_fq.replication_opens``,
  one ``sampler.sample_closures`` draw per replication read by every pair, owner queue M5-O21 (b), bit-identical to the
  ones behind F_Q);
- a route into a fab or an OSAT carries no closure term, as in naive's (67) (1^cl = 0,
  ``naive.carries_closure_term``; owner queue M5-O33 (b)), since sz replaces naive's term route by route;
- the closure term is (d_lk / d_cb) x ``naive_fq.quantile(law, CR_jk)``, capped by w^max as naive's level; an
  unobserved bucket, or one matching no replication week, falls back to naive's unconditional quantile
  (``context.fq_quantile``); on an injected list (no generator, no paths) sz equals naive's actions.

Prediction-free observables only, so it acts alike in every regime; warning-score buckets are ``hamilton_filter``'s
role, not M4's. The transient paths are never constructor state (1,000 x 26 x 2 floats pickle to about 416 kB, near
macOS's argument limit as hex): ``reset`` reads them from the per-process cache ``_PATHS`` that the runner seeds
(``remember_paths``).

Reset on a cache miss (design §12 M4 row "sz_state_base_stock"): with ``context.generator`` set, ``reset`` looks the key
(instance content digest, ``generator_id(context.generator.params(inst), inst)``, ``context.fq_replications``) up in
the ``_PATHS`` dict directly, never through ``transient_paths``, so a runner task whose parent seeded the cache never
calls ``transient_paths`` (``tests/test_m4_runner.py`` patches it to raise inside tasks). On a miss it reads the file
``context.paths_file`` names when one is given (``load_paths``: the parent's paths as public data, ``save_paths``),
which is how the wire entry point's ``child_argv`` child gets them (INT-M4-02: drawing 1,000 stream-22 replications per
pair in the child took minutes, all counted against week 1, and naive played every week); otherwise it calls
``transient_paths(inst, context.generator.params(inst), context.fq_replications)`` (serial), which fills the cache: the
consumers that seed nothing and name no file (``scripts/python/run_episode.py``, V24 replays, the joblib workers of a
generated V5 test) compute the paths in the process, bit-identical to the seeded ones. With ``context.generator`` None
nothing is read or computed and sz equals naive.
"""

import functools
import hashlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shockbench_flow.disruption.params import generator_id
from shockbench_flow.instance.schema import POOLS, Instance, frozen_map
from shockbench_flow.parallel import ordered_map
from shockbench_flow.policies import naive_parts
from shockbench_flow.policies.base import params_run_name
from shockbench_flow.policies.naive import (
    CLOSED_THRESHOLD,
    LastUseful,
    NaivePlan,
    end_aware_action,
    lane_demand,
    last_useful_weeks,
    level,
)
from shockbench_flow.policies.naive_derived import ThresholdParams
from shockbench_flow.policies.naive_fq import (
    backlog_path,
    check_replications,
    critical_quantiles,
    critical_ratio,
    laws_from_paths,
    quantile,
    replication_opens,
    routed_demand,
    rung_index,
)
from shockbench_flow.policies.registry import PathsFile, PolicyContext


BUCKETS = ("closed", "open", "unobserved")
# per-process path cache: (instance content digest, generator_id, replications) -> {(c, pool): (R, T) read-only array};
# filled by ``transient_paths`` and ``remember_paths``, read directly by ``SzPolicy.reset`` (module docstring)
_PATHS: dict[tuple[str, str, int], Mapping[tuple[int, str], np.ndarray]] = {}


@dataclass(frozen=True)
class SzParams(ThresholdParams):
    """``sz_state_base_stock``'s parameters: naive's step-6 thresholds, the closed cut also its bucket cut.

    Raises:
        ValueError: as ``naive.check_thresholds``.

    """


def paths_key(inst: Instance, params, replications: int) -> tuple[str, str, int]:
    """The ``_PATHS`` key: (instance family digest, ``generator_id(params, inst)``, replications), as naive's F_Q.

    Raises:
        TypeError, ValueError: as ``naive_fq.check_replications``.

    """
    return inst.family_digest, generator_id(params, inst), check_replications(replications)


def _pairs(inst: Instance) -> list[tuple[int, str]]:
    """The (chokepoint node, pool name) pairs a lane of naive's plan passes (``naive_fq.routed_demand``), sorted."""
    return sorted(routed_demand(inst))


def _replicate(inst: Instance, params, pairs: list[tuple[int, str]], r: int) -> list[np.ndarray]:
    """o^tr_{c,1..T} of replication r for every pair: the one draw ``naive_fq`` makes for F_Q (stream 22, episode 0).

    One sample per replication serves every pair (``naive_fq.replication_opens``; owner queue M5-O21 (b)), so the two
    pools of a chokepoint get the same row.
    """
    opens = replication_opens(inst, params, r)
    return [opens[c] for c, _pool in pairs]


def transient_paths(inst: Instance, params, replications: int, n_jobs: int = 1) -> Mapping[tuple[int, str], np.ndarray]:
    """o^tr of (66) per replication for every (chokepoint, pool) naive's plan passes: (R, T) read-only arrays.

    The same replications as naive's F_Q (stream 22 on the public root), cached per process by (instance content
    digest, ``generator_id``, replications) like ``naive_fq.generator_quantiles``; joblib only through
    ``parallel.ordered_map``.

    Raises:
        ValueError: as ``naive_fq.check_replications`` (TypeError for a non-integer), or if gamma is not a rung.

    """
    key = paths_key(inst, params, replications)
    held = _PATHS.get(key)
    if held is not None:
        return held
    rung_index(params)  # refuse an unkeyed rung before any draw
    pairs = _pairs(inst)
    per_rep = ordered_map(functools.partial(_replicate, inst, params, pairs), range(key[2]), n_jobs)
    out = {}
    for i, pair in enumerate(pairs):
        arr = np.array([rep[i] for rep in per_rep], dtype=np.float64).reshape(key[2], inst.T)
        arr.setflags(write=False)
        out[pair] = arr
    _PATHS[key] = frozen_map(out)
    return _PATHS[key]


def fq_quantiles_from_paths(inst: Instance, paths: Mapping[tuple[int, str], np.ndarray]) -> dict[tuple, float]:
    """Naive's F_Q quantiles from the paths, drawing nothing: ``naive_fq.critical_quantiles`` of the laws (66).

    The paths are the draws behind F_Q, so ``naive_fq.laws_from_paths`` (66) over each replication, a tandem pair's
    from its pool's network on the replication's joint paths (§8.1, Q54), pooled and sorted, is ``naive_fq.fq_laws``
    bit for bit and these quantiles are ``naive_fq.generator_quantiles``'s: a caller that needs both (the runner's
    parent) draws the replications once, then seeds ``naive_fq.remember_quantiles`` with these and ``remember_paths``
    with the paths. Before M5 this ran (66) with constant inflow at every pair, which differs on `small` and `full`'s
    tandem pairs (none on `tiny`).
    """
    return critical_quantiles(inst, laws_from_paths(inst, paths))


def remember_paths(inst: Instance, params, replications: int, paths: Mapping[tuple[int, str], np.ndarray]) -> None:
    """Seed this process's path cache with paths computed elsewhere (the runner's parent), checked, never recomputed.

    The arrays are copied (float64, read-only); an entry the process already holds is kept when the arrays are equal.

    Raises:
        ValueError: if the keys are not the (c, pool) pairs naive's plan passes, an array is not (R, T) of finite
            values in [0, 1], or the process already holds other paths for the key.

    """
    key = paths_key(inst, params, replications)
    pairs = _pairs(inst)
    if set(paths) != set(pairs):
        raise ValueError(f"sz paths are keyed by the (chokepoint, pool) pairs of naive's plan {pairs}: {list(paths)}")
    out = {}
    for pair in pairs:
        try:
            arr = np.array(paths[pair], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sz paths at {pair} are not an array of open fractions") from exc
        if arr.shape != (key[2], inst.T):
            raise ValueError(f"sz paths at {pair} must be (replications, T) = {(key[2], inst.T)}, got {arr.shape}")
        if not (np.all(np.isfinite(arr)) and np.all(arr >= 0.0) and np.all(arr <= 1.0)):
            raise ValueError(f"sz paths at {pair} must be finite open fractions in [0, 1]")
        arr.setflags(write=False)
        out[pair] = arr
    held = _PATHS.get(key)
    if held is not None:
        if any(not np.array_equal(held[p], out[p]) for p in pairs):
            raise ValueError("the process already holds other sz paths for this instance, generator and count")
        return
    _PATHS[key] = frozen_map(out)


def save_paths(
    path: str | Path, inst: Instance, params, replications: int, paths: Mapping[tuple[int, str], np.ndarray]
) -> PathsFile:
    """Write paths computed here as a public data file for a process that cannot share this one's cache (INT-M4-02).

    An ``.npz`` without pickles: one float64 array per (c, pool) pair of naive's plan (``o0``, ``o1``, ... in pair
    order) and ``meta``, a JSON string of the ``paths_key`` (instance content digest, ``generator_id``, replications)
    and the pairs. Returns the ``PathsFile`` naming it (absolute path) and the SHA-256 of its bytes, which a context
    carries so the reader refuses a file of other content.

    Raises:
        ValueError: if the keys are not the pairs of naive's plan, or as ``paths_key``.

    """
    key = paths_key(inst, params, replications)
    pairs = _pairs(inst)
    if set(paths) != set(pairs):
        raise ValueError(f"sz paths are keyed by the (chokepoint, pool) pairs of naive's plan {pairs}: {list(paths)}")
    meta = {
        "instance_digest": key[0],
        "generator_id": key[1],
        "replications": key[2],
        "pairs": [list(p) for p in pairs],
    }
    arrays = {f"o{i}": np.asarray(paths[p], dtype=np.float64) for i, p in enumerate(pairs)}
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    data = buf.getvalue()
    target = Path(path).resolve()
    target.write_bytes(data)
    return PathsFile(str(target), hashlib.sha256(data).hexdigest())


def load_paths(ref: PathsFile, inst: Instance, params, replications: int) -> Mapping[tuple[int, str], np.ndarray]:
    """The paths of the file ``ref`` names, checked and put in this process's cache (``remember_paths``); no draw.

    The bytes must hash to ``ref.sha256`` and the file's key must be ``paths_key(inst, params, replications)``; the
    arrays then pass ``remember_paths``'s checks (the pairs of naive's plan, (R, T) open fractions in [0, 1]).

    Raises:
        ValueError: if the digest or the key differs, or as ``remember_paths``.
        OSError: if the file cannot be read.

    """
    data = Path(ref.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError(f"sz paths file {ref.path}: its bytes are not the SHA-256 its context names")
    key = paths_key(inst, params, replications)
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        if (meta["instance_digest"], meta["generator_id"], meta["replications"]) != key:
            raise ValueError(f"sz paths file {ref.path} holds the paths of another instance, generator or count")
        paths = {(int(c), str(pool)): npz[f"o{i}"] for i, (c, pool) in enumerate(meta["pairs"])}
    remember_paths(inst, params, replications, paths)
    return _PATHS[key]


def bucket(open_fraction: float | None, closed_threshold: float = CLOSED_THRESHOLD) -> str:
    """The risk bucket of an observed open fraction: "closed" (<= cut), "open" (> cut) or "unobserved" (None)."""
    if open_fraction is None:
        return "unobserved"
    return "closed" if open_fraction <= closed_threshold else "open"


def conditional_quantile(
    paths: np.ndarray, which: str, q_obs: float, d: float, kappa: float, horizon: int, p: float, threshold: float
) -> float | None:
    """The p-quantile of (66) run forward from ``q_obs`` over the replication weeks of bucket ``which``; None if none.

    ``paths`` (R, T) are o^tr of one (c, b); a start week s qualifies when o^tr_{c, s-1} is in the bucket (s = 1 reads
    the stationary transient state, the path's first column); the window runs H = ``horizon`` weeks, truncated at T;
    the quantile rule is ``naive_fq.quantile`` (the ceil(pn)-th of the pooled samples). Every week of every qualifying
    window enters the pool; the windows of full length go to ``backlog_path`` in one call, the shorter ones near T one
    call each (elementwise the same arithmetic, so the pooled law does not depend on the batching).

    Raises:
        ValueError: if ``paths`` is not two-dimensional, ``which`` is not a bucket, ``horizon`` is not an integer >= 1,
            or as ``naive_fq.backlog_path``.

    """
    o = np.asarray(paths, dtype=np.float64)
    if o.ndim != 2:
        raise ValueError(f"sz paths are (replications, T), got shape {o.shape}")
    if which not in BUCKETS:
        raise ValueError(f"bucket must be one of {BUCKETS}, got {which!r}")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"the window H is an integer >= 1, got {horizon!r}")
    R, T = o.shape
    if which == "unobserved" or R == 0 or T == 0:
        return None  # a replication path is always observed
    state = np.concatenate([o[:, :1], o[:, :-1]], axis=1)  # column s - 1: o^tr_{s-1}, o^tr_1 at s = 1
    closed = state <= threshold
    match = closed if which == "closed" else ~closed
    parts = []
    full = T - horizon + 1  # start weeks s = 1..full have the whole window
    if full >= 1:
        windows = np.lib.stride_tricks.sliding_window_view(o, horizon, axis=1)[:, :full]  # (R, full, H)
        chosen = windows[match[:, :full]]
        if chosen.size:
            parts.append(backlog_path(q_obs, d, kappa, chosen).ravel())
    for s in range(max(full, 0) + 1, T + 1):  # the windows cut at T
        rows = np.flatnonzero(match[:, s - 1])
        if rows.size:
            parts.append(backlog_path(q_obs, d, kappa, o[rows, s - 1 :]).ravel())
    if not parts:
        return None
    return float(quantile(np.sort(np.concatenate(parts), kind="stable"), p))


def state_targets(
    inst: Instance,
    plan: NaivePlan,
    obs: dict,
    paths: Mapping[tuple[int, str], np.ndarray],
    closed_threshold: float = CLOSED_THRESHOLD,
) -> dict[tuple[int, int], float]:
    """S_jk of week t for sz: naive's step 3 with each lane route's closure term conditioned on the observation.

    Per lane route l into (j, k) and chokepoint c on it (pool b of k): the bucket of ``graph_now.open`` at c, Q_obs the
    ``math.fsum`` of the queue lots at c in pool b, and ``conditional_quantile(paths[(c, b)], bucket, Q_obs, d_cb(t),
    k_c mu_cb, tau_l + 1, CR_jk, cut)``, d_cb(t) naive's ``lane_demand`` of the week's d_lk m^sea(t); naive's quantile
    ``plan.fq_quantile[(c, b, j, k)]`` (0 when absent) where the bucket is unobserved, no week qualifies or the pair has
    no paths. The level is ``naive.level`` (the cover cap included) and the targets ``destination_targets``.
    """
    m = naive_parts.seasonal(inst, obs["week"])
    ds = [r.d * m.get((r.dest, r.k), 1.0) for r in plan.routes]
    d_cb = lane_demand(inst, plan.routes, ds)
    g = obs.get("graph_now") or {}
    opens = g.get("open")
    queued: dict[tuple[int, str], list[float]] = defaultdict(list)
    ql = obs["queue_lots"]
    for c, k, q in zip(ql["chokepoint"], ql["k"], ql["qty"]):
        queued[(c, inst.commodities[k].pool)].append(q)
    levels = []
    for r, d in zip(plan.routes, ds):
        closure = []
        if r.lane is not None and naive_parts.carries_closure_term(inst, r.dest):  # 1^cl of (67)
            pool = inst.commodities[r.k].pool
            for c in inst.lanes[r.lane].chokepoints:
                which = bucket(None if opens is None else opens[inst.chokepoint_ordinal[c]], closed_threshold)
                law_paths = paths.get((c, pool))
                q = None
                if which != "unobserved" and law_paths is not None:
                    q = conditional_quantile(
                        law_paths,
                        which,
                        math.fsum(queued.get((c, pool), ())),
                        d_cb[(c, pool)],
                        inst.nodes[c].chokepoint.kappa0[POOLS.index(pool)],
                        r.tau + 1,
                        critical_ratio(inst, r.dest, r.k),
                        closed_threshold,
                    )
                closure.append((d_cb[(c, pool)], plan.fq_quantile.get((c, pool, r.dest, r.k), 0.0) if q is None else q))
        levels.append(level(d, r.tau, r.w_max, closure, offset=r.offset))
    return naive_parts.destination_targets(inst, plan.routes, levels)


class SzPolicy:
    """`sz_state_base_stock` as a ``Policy``: stateless across weeks; paths from the process cache at reset."""

    name = "sz_state_base_stock"

    def __init__(self, params: SzParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = SzParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = params_run_name(type(self).name, self.params)  # its parameters in its name (SPEC-M4R2-01)
        self._inst: Instance | None = None
        self._plan: NaivePlan | None = None
        self._last: LastUseful | None = None
        self._paths: Mapping[tuple[int, str], np.ndarray] | None = None

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """The instance, naive's plan with ``context.fq_quantile``, the last useful weeks and the transient paths.

        The paths come from ``_PATHS`` read directly; on a miss from the file ``context.paths_file`` names
        (``load_paths``), else ``transient_paths`` draws them (module docstring).
        """
        self._inst, self._plan = naive_parts.cached_plan(static, self.context.fq_quantile)
        self._last = last_useful_weeks(self._inst, self._plan)
        self._paths = None
        gen = self.context.generator
        if gen is not None:
            params = gen.params(self._inst)
            reps = self.context.fq_replications
            held = _PATHS.get(paths_key(self._inst, params, reps))
            if held is None:
                ref = self.context.paths_file
                held = (
                    transient_paths(self._inst, params, reps)
                    if ref is None
                    else load_paths(ref, self._inst, params, reps)
                )
            self._paths = held

    def act(self, obs: dict) -> dict:
        """The end-aware naive's steps 4-6 on the week's state-conditioned targets (module docstring)."""
        if self._inst is None or self._plan is None or self._last is None:
            raise RuntimeError("SzPolicy.act called before reset")
        targets = None
        if self._paths is not None:
            targets = state_targets(self._inst, self._plan, obs, self._paths, self.params.closed_threshold)
        return end_aware_action(
            self._inst,
            self._plan,
            self._last,
            obs,
            closed_threshold=self.params.closed_threshold,
            give_up_ratio=self.params.give_up_ratio,
            targets=targets,
        )

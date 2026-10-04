"""Stratified episode pools on private roots (docs/EVOLVE_DESIGN.md, section 4.2).

A pool takes the first N episodes of each harm level on a root, which is the rule the public dev split uses
(``shockbench_flow.hosting.split.fill_strata``; harm comes from the event list, no LP). Its indices are cached in
``outputs/evolve_pools/`` and its references (naive's and the clairvoyant plan's costs) in shockbench-flow's own cache,
so a pool is computed once per package version and loads in seconds afterwards.
"""

from __future__ import annotations

import json
import time

from sbf_starter import ROOT, check_task


POOL_DIR = ROOT / "outputs" / "evolve_pools"
LEVELS = (1, 2, 3, 4)


def stratified(task: str, root: int, per_level, n_jobs: int = -1, verbose: bool = True) -> dict[int, list[int]]:
    """{level: episode indices} with the first ``per_level[s - 1]`` episodes of level s on ``root``, in index order."""
    check_task(task)
    if root == 0:
        raise ValueError("root 0 holds the public dev episodes: pools use roots of your own")
    per_level = [int(n) for n in per_level]
    path = POOL_DIR / f"{task}-{root}-{'-'.join(map(str, per_level))}.json"
    if path.is_file():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    from shockbench_flow.evaluation.cache import cut_points_cached, default_cache_dir
    from shockbench_flow.hosting.split import CUT_ENTROPY, fill_strata
    from shockbench_flow.hosting.tasks import CUT_DRAWS, MAX_CANDIDATES, split_label, task_generator

    t = time.perf_counter()
    inst, params = task_generator(task)
    cuts = cut_points_cached(inst, params, CUT_DRAWS, CUT_ENTROPY, n_jobs, default_cache_dir())
    scenarios, drawn, unfilled = fill_strata(
        inst, params, root, split_label(root), cuts, per_level, max_candidates=MAX_CANDIDATES, n_jobs=n_jobs
    )
    if unfilled:
        raise ValueError(f"{task} root {root}: levels {list(unfilled)} unfilled after {drawn} draws")
    out = {s: sorted(sc.episode for sc in scenarios if sc.stratum == s) for s in LEVELS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    if verbose:
        print(f"pool {task} root {root}: {per_level} per level from {drawn} draws ({time.perf_counter() - t:.0f} s)")
    return out


def episode_set(task: str, root: int, episodes: list[int], n_jobs: int = -1, verbose: bool = True):
    """The EpisodeSet of these episodes (references computed once, then read from the cache)."""
    from shockbench_flow_agent import EpisodeSet

    return EpisodeSet.build(task, list(episodes), entropy=root, n_jobs=n_jobs, verbose=verbose)


def take(levels: dict[int, list[int]], first: int = 0, count: int | None = None, only=LEVELS) -> list[int]:
    """Episodes ``first`` .. ``first + count - 1`` of each level in ``only``, level by level."""
    return [n for s in only for n in levels[s][first : None if count is None else first + count]]

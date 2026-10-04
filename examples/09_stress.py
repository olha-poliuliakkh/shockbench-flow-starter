"""Stress-test an agent on stratified episodes of a private root: chosen harm levels, Small or Full, paired against
another agent or the package's original mpc_det, with CPU and memory measured in a server-like process.

    uv run python examples/09_stress.py --agent=mine --root=918273645 --per_level=16
    uv run python examples/09_stress.py --agent=evo_champion --against=mine --levels=3,4 --per_level=24
    uv run python examples/09_stress.py --agent=mine --against=mpc_det --root=271828 --per_level=12
    uv run python examples/09_stress.py --agent=mine --task=full --per_level=4 --isolated=2

The pool takes the first ``per_level`` episodes of each chosen level on ``root`` (the dev split's rule); its references
are computed once and cached. ``--against=mpc_det`` plays the package's own planner (warm-started highspy) on the same
episodes in this machine's environment, so the gap measures the port and its evolved blocks against the original.
"""

import resource
import time

import fire
import numpy as np

from sbf_starter.agents import resolve
from sbf_starter.evolve import pools
from sbf_starter.evolve.evaluate import dollars_by_level, paired_gap


def _mpc_det_cost(task: str, root: int, n: int) -> int:
    """The package's mpc_det on episode n of ``root``, as EpisodeSet plays an agent (same scenario and fallback)."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS, generator_quantiles
    from shockbench_flow.policies.registry import PolicyContext, make_policy
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    _i, params = task_generator(task)
    pol = make_policy("mpc_det", PolicyContext(fq_quantile=generator_quantiles(inst, params)))
    traj = rollout(inst, pol, omega, "standard", _policy_seed(root, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    return int(traj.J_cents)


def _play(es, agent: str, task: str, root: int, workers: int, cpu_budget: bool) -> tuple[list[int], dict]:
    if agent == "mpc_det":
        from joblib import Parallel, delayed

        J = Parallel(n_jobs=workers)(delayed(_mpc_det_cost)(task, root, r["episode"]) for r in es.references)
        return list(J), {"fallback_weeks": 0, "cpu_weeks": 0, "invalid_entries": 0, "first_error": None}
    rows = es.play(str(resolve(agent)), cpu_budget=cpu_budget, n_jobs=workers)
    health = {k: sum(r[k] for r in rows) for k in ("fallback_weeks", "cpu_weeks", "invalid_entries")}
    health["first_error"] = next((r["first_error"] for r in rows if r["first_error"]), None)
    return [int(r["J_policy_cents"]) for r in rows], health


def main(
    agent: str = "mine",
    against: str | None = None,
    task: str = "small",
    root: int = 918273645,
    levels: str | tuple = "1,2,3,4",
    per_level: int = 16,
    workers: int = 8,
    cpu_budget: bool = True,
    isolated: int = 0,
) -> None:
    """Score ``agent`` (and ``against``, paired) on stratified episodes; optionally time it in isolated processes.

    Args:
        agent: an agent name, folder or zip.
        against: another agent, or ``mpc_det`` for the package's original planner.
        task: small or full.
        root: a private root (any integer but 0); a new integer is a scenario set no tuning has seen.
        levels: the harm levels to include, e.g. ``3,4`` for the severe crises only.
        per_level: episodes per included level.
        workers: processes that play episodes.
        cpu_budget: hand weeks over the task's CPU budget to naive, as the server does.
        isolated: dev episodes to also play in a process holding only the scoring image's packages (CPU per week,
            naive substitutions, peak memory of that process).

    """
    chosen = [int(s) for s in (levels.split(",") if isinstance(levels, str) else levels)]
    counts = [per_level if s in chosen else 0 for s in (1, 2, 3, 4)]
    t = time.perf_counter()
    eps = pools.take(pools.stratified(task, root, counts), only=chosen)
    es = pools.episode_set(task, root, eps)
    print(f"{task} root {root}: {len(eps)} episodes, levels {chosen} ({time.perf_counter() - t:.0f} s to load)")
    names = [agent] + ([against] if against else [])
    results = {}
    for name in names:
        t = time.perf_counter()
        J, health = _play(es, name, task, root, workers, cpu_budget)
        results[name] = J
        pooled = es.rss(J)
        how = "the board's level weights" if pooled["pooled"] else "these levels' episodes counted alike"
        print(f"\n{name}: score {pooled['rss']:.4f} ({how}) in {time.perf_counter() - t:.0f} s; health {health}")
        for s, d in dollars_by_level(es, J).items():
            print(
                f"  level {s}: n {d['n']}, saving against naive ${d['g'] / 1e9:,.1f}B of ${d['D'] / 1e9:,.1f}B "
                f"attainable per episode (ratio {d['ratio']:.3f})"
            )
    if against:
        g = paired_gap(es, results[agent], results[against])
        print(
            f"\n{agent} - {against}: {g['diff']:+.4f}, 90 % paired interval {g['lo']:+.4f} to {g['hi']:+.4f}; "
            f"better on {g['p_better']:.0%} of resampled sets"
            + ("; the interval holds 0: these episodes cannot tell them apart" if g["lo"] <= 0 <= g["hi"] else "")
        )
        for s in chosen:
            idx = [i for i, r in enumerate(es.references) if r["stratum"] == s]
            d = np.array([results[against][i] - results[agent][i] for i in idx]) / 100
            print(
                f"  level {s}: {agent} saves ${d.mean() / 1e9:,.1f}B per episode against {against} "
                f"(wins {np.mean(d > 0):.0%} of episodes)"
            )
    if isolated:
        from shockbench_flow_agent import LIMITS, play_isolated

        budget = LIMITS.cpu_budget_s[task]
        for n in range(isolated):
            row = play_isolated(resolve(agent).resolve(), task, n)
            cpu = [c for c in row["cpu_s"] if c is not None]
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
            print(
                f"\nisolated dev episode {n} of {task}: imported {row['imported']}, week 1 {cpu[0]:.3f} s, median "
                f"{np.median(cpu):.3f} s, p99 {np.quantile(cpu, 0.99):.3f} s, max {max(cpu):.3f} s of CPU (budget "
                f"{budget} s); naive substitutions {row['substitutions'][:10]}; peak memory of a child process "
                f"{peak:,.0f} MB (limit 4,096 MB)"
            )


if __name__ == "__main__":
    fire.Fire(main)

"""Closed loop: the planner as shipped against a two-pass solve that enforces base-load-first in weeks 2..H.

    uv run python scripts/research/two_pass.py                               # Small dev: oracle window and mpc_det
    uv run python scripts/research/two_pass.py --planners=oracle_H48 --modes=fix
    uv run python scripts/research/two_pass.py --task=full --root=20261006 --per_level=2 --planners=det_L

Pass 1 is the planner's own window LP. Its plan violates the rule where a grid sheds base load while its fabs draw
energy in a week >= 2 (the simulator never does). Pass 2 re-solves with column bounds (``common.base_first_bounds``),
warm from pass 1's basis:
- zero: every grid-week that sheds base load gets fab energy <= 0;
- fix: only violated grid-weeks change: shed base load <= 0 where the fabs' energy would have covered it, else fab
  energy <= 0.
Pass 1's action stands when pass 2 fails. Planners: det_<H> (persistence forecast) and oracle_<H> (true disruption
marks in the window), H in L, H48. Results: outputs/research/two_pass/<date_time>/results.json.
"""

import json

import fire
import numpy as np
from common import episodes, fmt_row, make_planner, run_dir, world
from joblib import Parallel, delayed


HORIZON = {"L": "L", "H48": "max(26,2L)"}


def play(
    task: str,
    root: int,
    n: int,
    planner: str,
    mode: str | None,
    passes: int = 2,
    milp_time: float = 10.0,
    milp_weeks: int | None = None,
) -> dict:
    from shockbench_flow.dynamics.env import rollout

    inst, omega, marks, fallback, seed = world(task, root, n)
    kind, h = planner.split("_", 1)
    pol = make_planner(
        inst,
        marks,
        task=task,
        horizon=HORIZON[h],
        oracle=kind == "oracle",
        two_pass=mode,
        passes=passes,
        milp_time=milp_time,
        milp_weeks=milp_weeks,
    )
    traj = rollout(inst, pol, omega, "standard", seed, marks=marks, fallback=fallback)
    log = pol.log
    return {
        "episode": n,
        "J": int(traj.J_cents),
        "seconds": [r["seconds"] for r in log],
        "pass2_weeks": sum(r["pass2"] is not None for r in log),
        "pass2_failed": sum(r["pass2"] is not None and r["after"] is None for r in log),
        "violations": sum(r["violations"] for r in log),
        "violations_after": sum(r["after"] or 0 for r in log if r["pass2"] is not None),
        "extra_solves": sum(r["extra"] for r in log),
        "weeks": len(log),
        "lp_failed": sum(not r["ok"] for r in log),
        "lots": float(sum(np.sum(r.lots_started) for r in traj.records)),
    }


def main(
    planners: str = "oracle_L,det_L",
    modes: str = "zero,fix",
    task: str = "small",
    root: int = 0,
    per_level: int = 0,
    passes: int = 2,
    milp_time: float = 10.0,
    milp_weeks: int = 0,
    dev_per_level: int = 0,
    workers: int = 3,
) -> None:
    """Each planner as shipped and in each two-pass mode, paired on the same episodes.

    Args:
        planners: comma-separated det_<H> or oracle_<H>, H in L, H48.
        modes: comma-separated two-pass modes (zero, fix); the planner as shipped always runs first.
        task: small or full.
        root: 0 with ``per_level`` 0 is the Small dev split; otherwise a private root.
        per_level: episodes per harm level of a stratified private pool (0: the dev split).
        passes: the most LP solves per week in a two-pass mode (3: one more pass for the violations pass 2 left).
        milp_time: seconds per window MILP in the "milp" mode (the exact rule; an offline test).
        milp_weeks: binaries only for the window's weeks 2..1+milp_weeks, "fix" bounds after (0: every week).
        dev_per_level: play only the first ``dev_per_level`` dev episodes of each harm level (0: all 20).
        workers: processes.

    """
    from sbf_starter.evolve.evaluate import paired_gap

    def split(v):
        return [x for x in (v if isinstance(v, (list, tuple)) else str(v).split(",")) if x]

    es = episodes(task, root, per_level, workers)
    if dev_per_level:
        from shockbench_flow_agent import EpisodeSet

        keep = [
            r["episode"] for s in (1, 2, 3, 4) for r in [r for r in es.references if r["stratum"] == s][:dev_per_level]
        ]
        es = EpisodeSet.build(task, keep, n_jobs=workers)
    ns = [r["episode"] for r in es.references]
    run = run_dir("two_pass")
    print(f"{len(ns)} {task} episodes (root {root}); run folder {run}")
    print(
        "| planner | RSS | L1 | L2 | L3 | L4 | paired gap vs as shipped [90 %] | s/week med / p99 / max "
        "| weeks with pass 2 | violations per week before / after | lots vs shipped |"
    )
    out = {}
    for planner in split(planners):
        base = None
        for mode in [None] + split(modes):
            rows = Parallel(n_jobs=workers)(
                delayed(play)(task, root, n, planner, mode, passes, milp_time, milp_weeks or None) for n in ns
            )
            J = [r["J"] for r in rows]
            secs = np.concatenate([r["seconds"] for r in rows])
            weeks = sum(r["weeks"] for r in rows)
            name = f"{planner} {mode or 'as shipped'}"
            if mode == "milp":
                name += (
                    f" ({milp_time:g} s"
                    + (f", {milp_weeks} weeks)" if milp_weeks else ")")
                    + (f" ({passes} passes)" if mode and passes != 2 else "")
                )
            if base is None:
                base = rows
                gap = "-"
            else:
                g = paired_gap(es, J, [r["J"] for r in base])
                gap = f"{g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"
            p2 = sum(r["pass2_weeks"] for r in rows)
            before = sum(r["violations"] for r in rows) / weeks
            after = sum(r["violations_after"] for r in rows) / max(p2, 1)
            lots = sum(r["lots"] for r in rows) / sum(r["lots"] for r in base)
            print(
                f"{fmt_row(name, es.rss(J))} {gap} | {np.median(secs):.3f} / {np.percentile(secs, 99):.3f} / "
                f"{secs.max():.3f} | {p2 / weeks:.0%} ({sum(r['pass2_failed'] for r in rows)} failed) | "
                f"{before:.2f} / {after:.2f} | {lots:.3f}x |",
                flush=True,
            )
            out[name] = {"rss": es.rss(J)["rss"], "rows": rows}
    (run / "results.json").write_text(json.dumps(out, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

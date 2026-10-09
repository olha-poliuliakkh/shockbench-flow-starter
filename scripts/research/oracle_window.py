"""Closed-loop planner variants: the package's mpc_det with a longer window, the true disruptions, or both.

    uv run python scripts/research/oracle_window.py                                  # Small dev, the 24-week pair
    uv run python scripts/research/oracle_window.py --variants=det_H48,oracle_H48    # 48-week windows
    uv run python scripts/research/oracle_window.py --task=full --root=20261006 --per_level=2 --variants=det_L,det_H48
    uv run python scripts/research/oracle_window.py --variants=mpc_scen              # the package's scenario LP

Variants: det_<H> is mpc_det on its persistence forecast, oracle_<H> on the true disruption marks of its window
(demand stays the forecast), oracle_all the true demand too; H is L (the planner's own; 24 weeks on Small), H32 (L+8),
H48 (max(26, 2L)) or Hfull. Results: outputs/research/oracle_window/<date_time>/results.json.
"""

import json
import time

import fire
import numpy as np
from common import components, episodes, fmt_row, make_planner, run_dir, world
from joblib import Parallel, delayed


HORIZON = {"L": "L", "H32": "L+8", "H48": "max(26,2L)", "Hfull": "full"}


def _planner(name, inst, marks, task):
    if name in ("mpc_det", "mpc_scen"):
        from shockbench_flow.hosting.tasks import task_generator
        from shockbench_flow.policies.naive_fq import generator_quantiles
        from shockbench_flow.policies.registry import GeneratorRef, PolicyContext, make_policy

        _i, gen = task_generator(task)
        ctx = PolicyContext(fq_quantile=generator_quantiles(inst, gen))
        if name == "mpc_scen":
            ctx = PolicyContext(fq_quantile=ctx.fq_quantile, generator=GeneratorRef(task, 0.62))
        return make_policy(name, ctx)
    if name.startswith("scenfix_"):  # mpc_scen with another window and the two-pass base-load-first solve
        return _scen_two_pass(inst, task, HORIZON[name.split("_", 1)[1]])
    if name.startswith("scen_"):  # mpc_scen with another window: scen_H48
        from shockbench_flow.hosting.tasks import task_generator
        from shockbench_flow.policies.mpc_scen import MpcScen, MpcScenParams
        from shockbench_flow.policies.naive_fq import generator_quantiles
        from shockbench_flow.policies.registry import GeneratorRef, PolicyContext

        _i, gen = task_generator(task)
        ctx = PolicyContext(fq_quantile=generator_quantiles(inst, gen), generator=GeneratorRef(task, 0.62))
        return MpcScen(MpcScenParams(H=HORIZON[name.split("_", 1)[1]]), ctx)
    if name == "oracle_all":
        return make_planner(inst, marks, task=task, oracle=True, true_demand=True)
    if name == "oracle_disruptions":
        name = "oracle_L"
    kind, h = name.split("_", 1)
    return make_planner(inst, marks, task=task, horizon=HORIZON[h], oracle=kind == "oracle")


def _scen_two_pass(inst, task: str, horizon: str):
    """mpc_scen whose SAA gets a second pass: each scenario block bounded by "fix" from its own pass-1 plan."""
    from dataclasses import replace

    from common import base_first_bounds, base_first_grids
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies import scenarios as S_
    from shockbench_flow.policies.mpc_scen import MpcScen, MpcScenParams
    from shockbench_flow.policies.naive_fq import generator_quantiles
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext

    grids = base_first_grids(inst)

    class ScenTwoPass(MpcScen):
        def act(self, obs):
            t0 = time.perf_counter()
            inst_, t = self._inst, int(obs["week"])
            self._memory.update(inst_, obs)
            self._ages.update(inst_, obs, self._memory)
            H_t = L.window_length(self._H, t, inst_.T)
            windows = S_.scenario_windows(
                inst_, self._gen, obs, self._memory, self._library, S_.TAG_MPC_SCEN, H_t, ages=self._ages
            )
            window = L.rolled_window(inst_, obs, H_t)
            models = [
                L.rolled_lp(inst_, obs, arr, H_t, fab_hits=hits, window=window, planning_rules=True)
                for arr, hits in windows
            ]
            shared = L.action_columns(inst_, models[0])
            lp, shape = L.saa_lp(models, shared)
            res = self._session.solve(lp, shape)
            if not res.ok:
                self.log.append(time.perf_counter() - t0)
                return self._fallback.act(obs)
            n = len(models[0].lb)
            bounded = []
            for b, m in enumerate(models):
                ub, _v, _c = base_first_bounds(m, res.x[b * n : (b + 1) * n], grids, "fix")
                bounded.append(replace(m, ub=ub))
            kept = self._session.state()
            lp2, _shape2 = L.saa_lp(bounded, shared)
            res2 = self._session.solve(lp2, None)  # warm from pass 1's basis
            self._session.load_state(kept)
            if res2.ok:
                models, res = bounded, res2
            action = L.week1_action(inst_, models[0], res.x[:n], obs, L.prohibited_now(self._memory, t))
            self.log.append(time.perf_counter() - t0)
            return action

    _i, gen = task_generator(task)
    ctx = PolicyContext(fq_quantile=generator_quantiles(inst, gen), generator=GeneratorRef(task, 0.62))
    pol = ScenTwoPass(MpcScenParams(H=horizon), ctx)
    pol.log = []
    return pol


def play(task: str, root: int, n: int, name: str) -> dict:
    from shockbench_flow.dynamics.env import rollout

    inst, omega, marks, fallback, seed = world(task, root, n)
    pol = _planner(name, inst, marks, task)
    start = time.perf_counter()
    traj = rollout(inst, pol, omega, "standard", seed, marks=marks, fallback=fallback)
    if hasattr(pol, "log"):
        secs = [r if isinstance(r, float) else r["seconds"] for r in pol.log]
    else:
        secs = [s.seconds for s in pol.telemetry]
    return {
        "episode": n,
        "J": int(traj.J_cents),
        "wall": time.perf_counter() - start,
        "week_med": float(np.median(secs)),
        "week_max": float(max(secs)),
        "comps": components(traj.records),
    }


def main(
    variants: str = "det_L,oracle_L", task: str = "small", root: int = 0, per_level: int = 0, workers: int = 3
) -> None:
    """Play each variant on the episodes and print RSS by level and the planner's seconds per week.

    Args:
        variants: comma-separated names (module docstring); scen_<H> is mpc_scen with window H.
        task: small or full.
        root: 0 with ``per_level`` 0 is the Small dev split; otherwise a private root.
        per_level: episodes per harm level of a stratified private pool (0: the dev split).
        workers: processes.

    """
    names = [v for v in (variants if isinstance(variants, (list, tuple)) else str(variants).split(",")) if v]
    es = episodes(task, root, per_level, workers)
    ns = [r["episode"] for r in es.references]
    run = run_dir("oracle_window")
    print(f"{len(ns)} {task} episodes (root {root}); run folder {run}")
    print("| variant | RSS | L1 | L2 | L3 | L4 | s/week med | max |")
    out = {}
    for name in names:
        rows = Parallel(n_jobs=workers)(delayed(play)(task, root, n, name) for n in ns)
        score = es.rss([r["J"] for r in rows])
        med, mx = float(np.median([r["week_med"] for r in rows])), max(r["week_max"] for r in rows)
        print(f"{fmt_row(name, score)} {med:.3f} | {mx:.3f} |", flush=True)
        out[name] = {"rss": score["rss"], "score": score, "rows": rows}
    (run / "results.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    fire.Fire(main)

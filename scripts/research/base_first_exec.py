"""Open-loop replays: the clairvoyant LP's flows against the base-load-first MILP plan's flows (research doc 5.1).

    uv run python scripts/research/base_first_exec.py                    # 60 s per MILP, about 15 minutes

Each dev episode: solve the reference LP and the base_first_milp.py MILP, then play each plan's week-t flows in the
simulator without re-planning (``lp_common.week1_action`` of the plan's week-t columns). If the MILP plan executes
far better, base-load-first is what breaks execution of the LP's plans. Results:
outputs/research/base_first_exec/<date_time>/results.json (the cited copy: results/base_first_exec.json).
"""

import dataclasses
import json

import fire
import numpy as np
from base_first_milp import solve_milp
from common import components, episodes, fmt_row, run_dir
from joblib import Parallel, delayed


def replay(n: int, time_limit: float) -> dict:
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.oracle.lp import solve_oracle
    from shockbench_flow.policies import lp_common as L

    row, model, x_bf, (inst, omega, marks, _fb, seed) = solve_milp(n, time_limit)
    nc = len(model.columns)
    week_model = dataclasses.replace(model, lb=model.lb[:nc])
    x_lp = solve_oracle(model).x

    def execute(x):
        class OpenLoop:
            name = "open_loop"

            def reset(self, *a):
                pass

            def act(self, obs):
                t = int(obs["week"])
                return L.week1_action(inst, week_model, x[(t - 1) * nc : t * nc], obs, marks.prohibited[t - 1])

        traj = rollout(inst, OpenLoop(), omega, "standard", seed, marks=marks, fallback=None)
        return int(traj.J_cents), components(traj.records), float(sum(np.sum(r.lots_started) for r in traj.records))

    planned = [j for k, j in model.index.items() if k[0] == "p"]
    row["J_exec_lp"], row["comps_lp"], row["lots_exec_lp"] = execute(x_lp)
    row["J_exec_bf"], row["comps_bf"], row["lots_exec_bf"] = execute(x_bf)
    row["lots_plan_lp"], row["lots_plan_bf"] = float(x_lp[planned].sum()), float(x_bf[planned].sum())
    return row


def main(time_limit: float = 60.0, workers: int = 3) -> None:
    """Replay both plans on every dev episode; print RSS by level, the paired gap and the lots planned and executed."""
    from sbf_starter.evolve.evaluate import paired_gap

    es = episodes("small", 0, 0, workers)
    ns = [r["episode"] for r in es.references]
    run = run_dir("base_first_exec")
    rows = {r["episode"]: r for r in Parallel(n_jobs=workers)(delayed(replay)(n, time_limit) for n in ns)}

    def J(key):
        return [rows[n][key] for n in ns]

    print("| plan | RSS | L1 | L2 | L3 | L4 |")
    print(fmt_row("clairvoyant LP flows, open loop", es.rss(J("J_exec_lp"))))
    print(fmt_row("base-load-first MILP flows, open loop", es.rss(J("J_exec_bf"))))
    print(fmt_row("base-load-first MILP, plan space", es.rss(J("J_inc"))))
    g = paired_gap(es, J("J_exec_bf"), J("J_exec_lp"))
    print(f"paired gap, MILP flows - LP flows: {g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    mean = {k: np.mean(J(k)) / 1e6 for k in ("lots_plan_lp", "lots_exec_lp", "lots_plan_bf", "lots_exec_bf")}
    print(
        f"lots per episode, planned / executed: LP {mean['lots_plan_lp']:.2f}M / {mean['lots_exec_lp']:.2f}M, "
        f"MILP {mean['lots_plan_bf']:.2f}M / {mean['lots_exec_bf']:.2f}M"
    )
    for key in ("shortage", "shed", "disposal"):
        a = np.mean([rows[n]["comps_lp"][key] for n in ns]) / 1e9
        b = np.mean([rows[n]["comps_bf"][key] for n in ns]) / 1e9
        print(f"  {key}: LP flows {a:,.1f} $B per episode, MILP flows {b:,.1f}")
    (run / "results.json").write_text(json.dumps(list(rows.values()), default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

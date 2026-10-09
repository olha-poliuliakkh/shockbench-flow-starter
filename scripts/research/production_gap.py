"""Where does a base-load-first plan lose production when the simulator executes it? (research doc 5.10)

    uv run python scripts/research/production_gap.py                     # Small dev, 60 s per MILP, about 10 minutes

For each dev episode: the clairvoyant MILP plan of base_first_milp.py, replayed open loop (base_first_exec.py). Every
fab-week compares the plan's lot starts p and fab energy E with the simulator's, whose rule is (sim.py, step 7)

    p-hat_f = min(alpha-bar R cap0, W_f),   p_f = p-hat_f * rho_g,   rho_g = min(1, (G-av - y) / sum_f' e p-hat_f' / R),

W_f the wafers on hand. A week that starts fewer lots than planned is attributed to:
- wafers: fewer wafers on hand than the plan starts (W_f < p_plan);
- split: enough wafers, the grid gave its fabs at least the planned energy in total, but this fab got less (the
  pro-rata split by requested draw);
- generation: enough wafers, the grid's fabs got less energy in total than planned (fuel, rationing, base load);
and a week that starts more than planned to the push rule (all wafers on hand start when energy allows). OSAT weeks
compare packaged chips alike (the simulator packages all raw chips up to throughput, pro rata). Lots are counted in
units, summed over episodes; shortage and shed in USD per episode, plan (LP model) against execution.
"""

import dataclasses
import json

import fire
from base_first_milp import solve_milp
from common import components, episodes, run_dir
from joblib import Parallel, delayed


def attribute(n: int, time_limit: float) -> dict:
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.oracle.lp import lp_costs
    from shockbench_flow.policies import lp_common as L

    _row, model, x, (inst, omega, marks, _fb, seed) = solve_milp(n, time_limit)
    nc = len(model.columns)
    week_model = dataclasses.replace(model, lb=model.lb[:nc])

    class OpenLoop:
        name = "open_loop"

        def reset(self, *a):
            pass

        def act(self, obs):
            t = int(obs["week"])
            return L.week1_action(inst, week_model, x[(t - 1) * nc : t * nc], obs, marks.prohibited[t - 1])

    traj = rollout(inst, OpenLoop(), omega, "standard", seed, marks=marks, fallback=None)
    idx = model.index
    out = {k: 0.0 for k in ("planned", "executed", "short_wafers", "short_split", "short_generation", "over_push")}
    out.update({"osat_planned": 0.0, "osat_executed": 0.0, "osat_short": 0.0, "osat_over": 0.0})
    for t, rec in enumerate(traj.records, start=1):
        for gi, g in enumerate(inst.grids):
            fos = list(inst.grid_fabs[gi])
            E_plan_g = sum(x[idx[("E", t, fo)]] for fo in fos if ("E", t, fo) in idx)
            E_exec_g = float(sum(rec.energy[fo] for fo in fos))
            for fo in fos:
                f = inst.fabs[fo]
                fab = inst.nodes[f].fab
                s_in = inst.slot_index[(f, fab.input)]
                p_plan, p_exec = float(x[idx[("p", t, fo)]]), float(rec.lots_started[fo])
                W = float(rec.stock[s_in] + p_exec + rec.disposal[s_in])
                out["planned"] += p_plan
                out["executed"] += p_exec
                gap = p_plan - p_exec
                if gap > 1e-6 * max(1.0, p_plan):
                    if W < p_plan - 1e-9:
                        out["short_wafers"] += min(gap, p_plan - W)
                        gap -= min(gap, p_plan - W)
                    if gap > 0:
                        out["short_split" if E_exec_g >= E_plan_g - 1e-9 else "short_generation"] += gap
                elif gap < 0:
                    out["over_push"] += -gap
        for (oo, kp), q in rec.packaged.items():
            j = idx.get(("xi", t, oo, kp))
            plan = float(x[j]) if j is not None else 0.0
            out["osat_planned"] += plan
            out["osat_executed"] += float(q)
            d = plan - float(q)
            out["osat_short" if d > 0 else "osat_over"] += abs(d)
    plan_costs = {}
    for w in lp_costs(model, x)[0]:
        for k, v in w.as_dict().items():
            plan_costs[k] = plan_costs.get(k, 0.0) + v
    exec_costs = components(traj.records)
    out.update(
        {
            "episode": n,
            "J_plan": int(_row["J_inc"]),
            "J_exec": int(traj.J_cents),
            **{f"plan_{k}": plan_costs[k] for k in ("shortage", "shed")},
            **{f"exec_{k}": exec_costs[k] for k in ("shortage", "shed")},
        }
    )
    return out


def main(time_limit: float = 60.0, workers: int = 3) -> None:
    """Attribute the production gap on every dev episode and print the totals."""
    es = episodes("small", 0, 0, workers)
    ns = [r["episode"] for r in es.references]
    rows = Parallel(n_jobs=workers)(delayed(attribute)(n, time_limit) for n in ns)
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("episode",) and not k.startswith("J_")}
    rss_plan, rss_exec = (es.rss([r[k] for r in rows])["rss"] for k in ("J_plan", "J_exec"))
    print(f"RSS: plan {rss_plan:.4f}, executed {rss_exec:.4f}")
    print(f"lots: planned {tot['planned'] / 1e6:.2f}M, executed {tot['executed'] / 1e6:.2f}M (20 episodes)")
    short = tot["short_wafers"] + tot["short_split"] + tot["short_generation"]
    for k in ("short_wafers", "short_split", "short_generation"):
        share = tot[k] / max(short, 1e-9)
        print(f"  fewer than planned, {k[6:]}: {tot[k] / 1e6:.2f}M lots ({share:.0%} of the shortfall)")
    print(f"  more than planned (push rule): {tot['over_push'] / 1e6:.2f}M lots")
    print(
        f"OSAT packaged: planned {tot['osat_planned'] / 1e6:.2f}M, executed {tot['osat_executed'] / 1e6:.2f}M; "
        f"short {tot['osat_short'] / 1e6:.2f}M, over {tot['osat_over'] / 1e6:.2f}M"
    )
    for k in ("shortage", "shed"):
        a, b = tot[f"plan_{k}"] / len(rows) / 1e9, tot[f"exec_{k}"] / len(rows) / 1e9
        print(f"{k}: plan {a:,.1f} $B per episode, executed {b:,.1f} ({b - a:+,.1f})")
    run = run_dir("production_gap")
    (run / "results.json").write_text(json.dumps(rows, indent=1))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

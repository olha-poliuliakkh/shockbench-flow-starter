"""Base-load-first as a PRICE: how much the clairvoyant LP diverts, and what the price does to plans (research doc 3.3).

    uv run python scripts/research/base_first.py            # Small dev, about 5 minutes with 3 workers

(a) the clairvoyant LP: energy it diverts from base load to fabs at the bound grids, min(shed, sum E) per week, as a
    share of its shed base load;
(b) the clairvoyant LP with the planning rules' base-load price in EVERY week: its true J (the price excluded). Not a
    bound: the price also charges shed that no plan can avoid (the exact bound is base_first_milp.py);
(c) the oracle-window planner and (d) mpc_det, each with that price in every window week, paired against the
    planner as shipped.
Results: outputs/research/base_first/<date_time>/results.json (the cited copy: results/base_first.json).
"""

import json

import fire
import numpy as np
from common import base_first_grids, base_first_price, components, episodes, fmt_row, make_planner, run_dir, world
from joblib import Parallel, delayed


def lp_part(n: int) -> dict:
    from scipy.optimize import linprog
    from shockbench_flow.oracle.lp import build_lp, lp_cents, solve_oracle

    inst, _omega, marks, _fb, _seed = world("small", 0, n)
    model = build_lp(inst, marks)
    x = solve_oracle(model).x
    idx, T = model.index, model.T
    grids = base_first_grids(inst)
    diverted = shed = 0.0
    for t in range(1, T + 1):
        for go, fos in grids.items():
            ysh = x[idx[("ysh", t, go)]]
            E = sum(x[idx[("E", t, fo)]] for fo in fos if ("E", t, fo) in idx)
            diverted += min(ysh, E)
            shed += ysh
    c = model.objective().copy()
    for go, price in base_first_price(inst, T).items():
        for t in range(1, T + 1):
            c[idx[("ysh", t, go)]] += price
    bounds = list(zip(model.lb, np.where(np.isinf(model.ub), None, model.ub)))
    r = linprog(c, A_ub=model.A_ub, b_ub=model.b_ub, A_eq=model.A_eq, b_eq=model.b_eq, bounds=bounds, method="highs-ds")
    if r.status != 0:
        raise RuntimeError(f"episode {n}: {r.message}")
    return {
        "episode": n,
        "J_oracle": int(lp_cents(model, x)),
        "J_bf": int(lp_cents(model, r.x)),
        "diverted": float(diverted),
        "base_shed": float(shed),
    }


def mpc_part(n: int, oracle: bool, extend: bool) -> dict:
    from shockbench_flow.dynamics.env import rollout

    inst, omega, marks, fb, seed = world("small", 0, n)
    pol = make_planner(inst, marks, oracle=oracle, price_all_weeks=extend)
    traj = rollout(inst, pol, omega, "standard", seed, marks=marks, fallback=fb)
    return {
        "episode": n,
        "J": int(traj.J_cents),
        "fails": sum(1 for r in pol.log if not r["ok"]),
        "comps": components(traj.records),
        "lots": float(sum(np.sum(r.lots_started) for r in traj.records)),
    }


def main(workers: int = 3) -> None:
    """Print (a) to (d) on the Small dev split."""
    from sbf_starter.evolve.evaluate import paired_gap

    es = episodes("small", 0, 0, workers)
    ns = [r["episode"] for r in es.references]
    refs = {r["episode"]: r for r in es.references}
    run = run_dir("base_first")
    lp = Parallel(n_jobs=workers)(delayed(lp_part)(n) for n in ns)
    same = all(r["J_oracle"] == refs[r["episode"]]["J_oracle_cents"] for r in lp)
    share = sum(r["diverted"] for r in lp) / sum(r["base_shed"] for r in lp)
    print(
        f"(a) the clairvoyant LP diverts {share:.1%} of its shed base load to fabs (J matches the references: {same})"
    )
    by = {r["episode"]: r for r in lp}
    print("| policy | RSS | L1 | L2 | L3 | L4 |")
    print(fmt_row("(b) clairvoyant LP, price in every week (not a bound)", es.rss([by[n]["J_bf"] for n in ns])))
    out = {"lp": lp}
    for name, oracle in (("oracle_window", True), ("mpc_det", False)):
        rows = {e: Parallel(n_jobs=workers)(delayed(mpc_part)(n, oracle, e) for n in ns) for e in (False, True)}
        Jb, Je = [r["J"] for r in rows[False]], [r["J"] for r in rows[True]]
        g = paired_gap(es, Je, Jb)
        print(fmt_row(f"{name} as shipped", es.rss(Jb)))
        print(
            f"{fmt_row(f'{name}, price in every week', es.rss(Je))} gap {g['diff']:+.4f} "
            f"[{g['lo']:+.4f}, {g['hi']:+.4f}]; failures {sum(r['fails'] for r in rows[True])}",
            flush=True,
        )
        out[name] = {str(k): v for k, v in rows.items()}
    (run / "results.json").write_text(json.dumps(out, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

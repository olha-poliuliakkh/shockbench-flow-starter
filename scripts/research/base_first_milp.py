"""The exact base-load-first bound: an upper bound on any policy's RSS on the Small dev split (research doc 3.3).

    uv run python scripts/research/base_first_milp.py                    # 120 s per MILP, about 15 minutes
    uv run python scripts/research/base_first_milp.py --time_limit=60

The clairvoyant reference LP plus, per bound grid g and week t, a binary z with
    sum_f E_ft <= M_gt z      and      ysh_gt <= ybar_gt (1 - z),      M_gt = G-bar_gt sum_k zeta_gk:
fabs draw energy only when base load is fully served. The simulator satisfies these rows, so the MILP is still a
relaxation of every policy and its dual bound bounds any policy's RSS. Prints the RSS of the dual bound and of the
incumbent (a plan that respects the rule). Results: outputs/research/base_first_milp/<date_time>/results.json.
"""

import json
import time

import fire
import numpy as np
import scipy.sparse as sp
from common import base_first_grids, episodes, fmt_row, run_dir, world
from joblib import Parallel, delayed


def solve_milp(n: int, time_limit: float):
    """(summary row, the reference LP model, the incumbent's LP columns or None, the world) of dev episode ``n``."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from shockbench_flow.oracle.lp import build_lp, lp_cents
    from shockbench_flow.policies.lp_common import model_offset

    w = world("small", 0, n)
    inst, _omega, marks = w[:3]
    model = build_lp(inst, marks)
    idx, T, n0 = model.index, model.T, len(model.lb)
    grids = base_first_grids(inst)
    z_of = {(go, t): n0 + i for i, (go, t) in enumerate((go, t) for go in grids for t in range(1, T + 1))}
    nz = len(z_of)
    rows, cols, vals, rhs = [], [], [], []
    for (go, t), jz in z_of.items():
        r = len(rhs)
        M = float(marks.G_bar[t - 1, go]) * sum(inst.nodes[inst.grids[go]].grid.shares.values())
        for fo in grids[go]:
            if ("E", t, fo) in idx:
                rows.append(r)
                cols.append(idx[("E", t, fo)])
                vals.append(1.0)
        rows.append(r)
        cols.append(jz)
        vals.append(-M)
        rhs.append(0.0)
        yb = float(marks.y_bar[t - 1, go])
        rows += [r + 1, r + 1]
        cols += [idx[("ysh", t, go)], jz]
        vals += [1.0, yb]
        rhs.append(yb)
    N = n0 + nz
    A_ub = sp.vstack(
        [
            sp.hstack([model.A_ub, sp.csr_matrix((model.A_ub.shape[0], nz))]),
            sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), N)),
        ]
    ).tocsr()
    A_eq = sp.hstack([model.A_eq, sp.csr_matrix((model.A_eq.shape[0], nz))]).tocsr()
    start = time.perf_counter()
    res = milp(
        np.concatenate([model.objective(), np.zeros(nz)]),
        integrality=np.concatenate([np.zeros(n0), np.ones(nz)]),
        bounds=Bounds(np.concatenate([model.lb, np.zeros(nz)]), np.concatenate([model.ub, np.ones(nz)])),
        constraints=[
            LinearConstraint(A_ub, -np.inf, np.concatenate([model.b_ub, rhs])),
            LinearConstraint(A_eq, model.b_eq, model.b_eq),
        ],
        options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "disp": False},
    )
    row = {"episode": n, "status": int(res.status), "seconds": time.perf_counter() - start, "binaries": nz}
    x = None if res.x is None else res.x[:n0]
    if x is not None:
        row["J_inc"] = int(lp_cents(model, x))
    if getattr(res, "mip_dual_bound", None) is not None:
        row["J_bound"] = int(round((res.mip_dual_bound + model_offset(model)) * 100))
    return row, model, x, w


def _row(n: int, time_limit: float) -> dict:
    return solve_milp(n, time_limit)[0]


def main(time_limit: float = 120.0, workers: int = 3) -> None:
    """Solve the MILP of every dev episode and print the RSS of the bound and of the incumbents."""
    es = episodes("small", 0, 0, workers)
    ns = [r["episode"] for r in es.references]
    run = run_dir("base_first_milp")
    rows = {r["episode"]: r for r in Parallel(n_jobs=workers)(delayed(_row)(n, time_limit) for n in ns)}
    gaps = [(r["J_inc"] - r["J_bound"]) / abs(r["J_bound"]) for r in rows.values() if "J_inc" in r and "J_bound" in r]
    print(
        f"{sum(r['status'] == 1 for r in rows.values())} of {len(ns)} MILPs stopped at the time limit; "
        f"relative gap median {np.median(gaps):.1e}, max {max(gaps):.1e}\n| plan | RSS | L1 | L2 | L3 | L4 |"
    )
    if all("J_bound" in r for r in rows.values()):
        print(fmt_row("MILP dual bound (upper bound under base-load-first)", es.rss([rows[n]["J_bound"] for n in ns])))
    if all("J_inc" in r for r in rows.values()):
        print(fmt_row("MILP incumbent (respects the rule; plan space)", es.rss([rows[n]["J_inc"] for n in ns])))
    (run / "results.json").write_text(json.dumps(list(rows.values()), indent=1))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

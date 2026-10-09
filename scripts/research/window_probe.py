"""Probe the two-pass window plans of mpc_det at 48 weeks: end-of-window starvation and the violations pass 2 leaves.

    uv run python scripts/research/window_probe.py                     # Small dev episodes 0, 3, 6; weeks 5, 15, 25
    uv run python scripts/research/window_probe.py --episodes=0,1 --weeks=10,20 --horizon=L

Plays each episode with the package's mpc_det (persistence forecast, two-pass "fix", the package's warm solver) and,
at the sampled weeks, reads the window's pass-1 and pass-2 plans:

- starvation: by window week h, the planned lot starts, wafer deliveries into fabs, fuel deliveries into grids, the
  value of stock (sum of I times its terminal credit) and the shortage planned; the last 4 weeks against weeks
  2..H-5, and the terminal credit per unit against the commodity's value v_k;
- residual violations: the grid-weeks where pass 2's plan sheds base load while fabs draw energy, whether pass 2
  bounded them (it cannot violate a bounded one), and their size: the shed and fab energy as shares of base load.
"""

import json
from dataclasses import replace

import fire
import numpy as np
from common import base_first_bounds, base_first_grids, make_planner, run_dir, world


def _by_tag(model, x, tag, pred=lambda key: True):
    """Sum of x over the columns of ``tag`` per window week (keys are one week's template)."""
    nc, T = len(model.columns), int(model.T)
    js = [j for j, key in enumerate(model.columns) if key[0] == tag and pred(key)]
    return np.array([float(x[t * nc + np.array(js)].sum()) if js else 0.0 for t in range(T)])


def _violations(model, x, grids):
    """[(week, grid, shed share of base load, fab energy share of base load)] of a plan, weeks >= 2."""
    cols = {key: j for j, key in enumerate(model.columns)}
    nc, out = len(model.columns), []
    for t in range(2, int(model.T) + 1):
        off = (t - 1) * nc
        for go, fos in grids.items():
            shed, served = x[off + cols[("ysh", go)]], x[off + cols[("y", go)]]
            E = sum(x[off + cols[("E", fo)]] for fo in fos if ("E", fo) in cols)
            base = max(shed + served, 1e-9)
            if shed > 1e-6 * max(base, 1.0) and E > 1e-6 * max(base, 1.0):
                out.append((t, go, float(shed / base), float(E / base)))
    return out


def probe(n: int, weeks: list[int], horizon: str) -> dict:
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.base import reset_policy

    inst, omega, marks, fallback, seed = world("small", 0, n)
    pol = make_planner(inst, marks, horizon=horizon, two_pass="fix")
    grids = base_first_grids(inst)
    fabs, gridnodes = set(inst.fabs), set(inst.grids)
    terminals = {e.tail for e in inst.edges if e.head in gridnodes}
    env = Env(fallback=fallback)
    obs, info = env.reset(inst, "standard", omega, seed, marks=marks)
    reset_policy(pol, info["static"], obs, seed, info.get("omega"))
    out, done = [], False
    while not done:
        t = int(obs["week"])
        if t in weeks:
            mem = pol._memory.__class__.from_state(inst, pol._memory.state())
            mem.update(inst, obs)
            H = L.window_length(pol._H, t, inst.T)
            model = L.rolled_lp(inst, obs, pol._window_arrays(inst, obs, H), H, planning_rules=True)
            res1 = L.LPSession().solve(L.to_highs_lp(model), L.WindowShape.of(model))
            ub, v1, _ = base_first_bounds(model, res1.x, grids, "fix")
            m2 = replace(model, ub=ub)
            res2 = L.LPSession().solve(L.to_highs_lp(m2), L.WindowShape.of(m2))
            x = res2.x if res2.ok else res1.x
            cols, nc = {k: j for j, k in enumerate(model.columns)}, len(model.columns)

            def is_bounded(tt, go):
                off = (tt - 1) * nc
                js = [cols[("ysh", go)]] + [cols[("E", fo)] for fo in grids[go] if ("E", fo) in cols]
                return any(ub[off + j] == 0 for j in js)

            left = _violations(m2, x, grids)
            p = _by_tag(model, x, "p")
            into_fab = _by_tag(model, x, "x", lambda key: inst.edges[key[1]].head in fabs)
            into_grid = _by_tag(model, x, "x", lambda key: inst.edges[key[1]].head in gridnodes | terminals)
            unserved = _by_tag(model, x, "U") + _by_tag(model, x, "B")
            is_I = np.array([k[0] == "I" for k in model.columns])
            credit = np.where(is_I, model.salvage[(H - 1) * nc : H * nc], 0.0)  # terminal credit per unit of stock
            stock_val = np.array([float(np.dot(x[h * nc : (h + 1) * nc], credit)) for h in range(H)])
            values = {}
            for j, key in enumerate(model.columns):
                if key[0] == "I":
                    k = inst.stock_slots[key[1]].k
                    v = inst.commodities[k].v
                    if v > 0:
                        values.setdefault(inst.commodities[k].id, []).append(credit[j] / v)
            mid, tail = slice(1, max(2, H - 4)), slice(H - 4, H)

            def ratio(a):
                m = float(np.mean(a[mid]))
                return round(float(np.mean(a[tail]) / m), 3) if m > 0 else None

            out.append(
                {
                    "episode": n,
                    "week": t,
                    "H": H,
                    "tail_vs_mid": {
                        "lot_starts": ratio(p),
                        "wafers_into_fabs": ratio(into_fab),
                        "fuel_into_grids": ratio(into_grid),
                        "stock_value": ratio(stock_val),
                        "unserved": ratio(unserved),
                    },
                    "terminal_credit_over_value": {k: round(float(np.mean(v)), 3) for k, v in values.items()},
                    "violations_pass1": v1,
                    "violations_pass2": len(left),
                    "pass2_on_bounded": sum(is_bounded(tt, go) for tt, go, *_ in left),
                    "pass2_sizes": [(tt, round(sh, 4), round(e, 4)) for tt, _g, sh, e in left][:12],
                }
            )
        obs, _r, done, _tr, _i = env.step(pol.act(obs))
    return out


def main(episodes: str = "0,3,6", weeks: str = "5,15,25", horizon: str = "max(26,2L)") -> None:
    def ints(v):
        return [int(a) for a in (v if isinstance(v, (list, tuple)) else str(v).split(","))]

    rows = [r for n in ints(episodes) for r in probe(n, ints(weeks), horizon)]
    for r in rows:
        print(json.dumps(r))
    (run_dir("window_probe") / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    fire.Fire(main)

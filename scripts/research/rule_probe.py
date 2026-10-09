"""Do the planner's own window plans (two-pass, 48 weeks) break the simulator's production rules in weeks 2..H?

    uv run python scripts/research/rule_probe.py              # Small dev episodes 0, 3, 6, 9; weeks 5, 15, 25

Plays each episode with mpc_det (persistence forecast, two-pass "fix", 48 weeks; the package's warm solver) and reads
pass 2's window plan at the sampled weeks. With W the wafers a fab holds before starting (end stock + starts +
disposal) and cap = alpha-bar R cap0, the simulator's week-7 rules are broken where a plan:
- push (rule 1): holds wafers at a fab with spare capacity and spare energy: W - p > 0, p < cap, R E - e p > 0
  (the simulator would start them);
- split (rule 2): gives two fabs of one grid different start ratios p / min(cap, W) (the simulator's ratio rho_g is
  common to the grid's fabs);
- OSAT (rule 3): holds raw chips at an OSAT below its throughput (the simulator packages them all, pro rata).
Counts are grid- or plant-weeks per window, sizes in lots (or chips) per window, against the plan's total starts.
"""

import json

import fire
import numpy as np
from common import base_first_bounds, base_first_grids, make_planner, run_dir, world


def broken(inst, model, x, arrays) -> dict:
    """The rule breaks of one window plan ``x`` (module docstring); ``arrays`` the window's marks (R, R_osat)."""
    from shockbench_flow.marks import osat_throughput

    cols = {k: j for j, k in enumerate(model.columns)}
    nc, T = len(model.columns), int(model.T)
    out = {"starts": 0.0, "push_weeks": 0, "push_lots": 0.0, "split_weeks": 0, "split_lots": 0.0}
    out.update({"osat_weeks": 0, "osat_chips": 0.0, "packaged": 0.0})

    def val(key, off):
        return float(x[off + cols[key]]) if key in cols else 0.0

    for t in range(2, T + 1):
        off = (t - 1) * nc
        ratios = {}
        for fo, f in enumerate(inst.fabs):
            fab = inst.nodes[f].fab
            s_in = inst.slot_index[(f, fab.input)]
            p = val(("p", fo), off)
            cap = float(model.ub[off + cols[("p", fo)]])
            held = val(("I", s_in), off) + val(("O", s_in), off)
            out["starts"] += p
            tol = 1e-6 * max(1.0, cap)
            spare_E = True
            if ("E", fo) in cols and fab.e > 0:
                spare_E = float(arrays["R"][t - 1, fo]) * val(("E", fo), off) - fab.e * p > tol
            if held > tol and p < cap - tol and spare_E:
                out["push_weeks"] += 1
                out["push_lots"] += min(held, cap - p)
            phat = min(cap, held + p)
            if fab.grid is not None and fab.e > 0 and phat > tol:
                ratios.setdefault(inst.grid_ordinal[fab.grid], []).append((p / phat, phat))
        for rs in ratios.values():
            if len(rs) > 1 and max(a for a, _ in rs) - min(a for a, _ in rs) > 1e-4:
                out["split_weeks"] += 1
                mean = sum(a * b for a, b in rs) / sum(b for _, b in rs)
                out["split_lots"] += sum(abs(a - mean) * b for a, b in rs) / 2
        thr = osat_throughput(inst, np.asarray(arrays["R_osat"][t - 1]))
        for oo, o in enumerate(inst.osats):
            osat = inst.nodes[o].osat
            held = sum(val(("I", inst.slot_index[(o, kr)]), off) for kr in osat.packages)
            xi = sum(val(("xi", oo, kp), off) for kp in osat.packages.values())
            out["packaged"] += xi
            if held > 1e-6 and xi < float(thr[oo]) - 1e-6:
                out["osat_weeks"] += 1
                out["osat_chips"] += min(held, float(thr[oo]) - xi)
    return out


def probe(n: int, weeks: list[int]) -> list[dict]:
    from dataclasses import replace

    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.base import reset_policy

    inst, omega, marks, fallback, seed = world("small", 0, n)
    pol = make_planner(inst, marks, horizon="max(26,2L)", two_pass="fix")
    grids = base_first_grids(inst)
    env = Env(fallback=fallback)
    obs, info = env.reset(inst, "standard", omega, seed, marks=marks)
    reset_policy(pol, info["static"], obs, seed, info.get("omega"))
    rows, done = [], False
    while not done:
        t = int(obs["week"])
        if t in weeks:
            mem = pol._memory.__class__.from_state(inst, pol._memory.state())
            mem.update(inst, obs)
            H = L.window_length(pol._H, t, inst.T)
            arrays = pol._window_arrays(inst, obs, H)
            model = L.rolled_lp(inst, obs, arrays, H, planning_rules=True)
            res = L.LPSession().solve(L.to_highs_lp(model), L.WindowShape.of(model))
            ub, _v, _c = base_first_bounds(model, res.x, grids, "fix")
            m2 = replace(model, ub=ub)
            res2 = L.LPSession().solve(L.to_highs_lp(m2), L.WindowShape.of(m2))
            rows.append({"episode": n, "week": t, "H": H, **broken(inst, m2, res2.x if res2.ok else res.x, arrays)})
        obs, _r, done, _tr, _i = env.step(pol.act(obs))
    return rows


def main(episodes: str = "0,3,6,9", weeks: str = "5,15,25") -> None:
    def ints(v):
        return [int(a) for a in (v if isinstance(v, (list, tuple)) else str(v).split(","))]

    rows = [r for n in ints(episodes) for r in probe(n, ints(weeks))]
    for r in rows:
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}))
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("episode", "week", "H")}
    push = tot["push_lots"] / tot["starts"]
    split = tot["split_lots"] / tot["starts"]
    osat = tot["osat_chips"] / tot["packaged"]
    print(f"over {len(rows)} windows: push {tot['push_weeks']} fab-weeks, {push:.1%} of planned starts")
    print(f"  split {tot['split_weeks']} grid-weeks, {split:.1%} of planned starts")
    print(f"  OSAT {tot['osat_weeks']} plant-weeks, {osat:.1%} of planned packaging")
    (run_dir("rule_probe") / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    fire.Fire(main)

"""Shared pieces of the research scripts: episodes, worlds, the planner variants, base-load-first bounds, reports.

The scripts of this folder are offline experiments (they read the episodes' true future through the scorer's own
internals), never agents. They import each other as siblings: run them as ``uv run python scripts/research/<name>.py``.
"""

import os
import time
from dataclasses import replace
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process, before numpy loads

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"  # the reference outputs the research document cites


def run_dir(name: str) -> Path:
    """outputs/research/<name>/<date_time>/, created; a run started in the same second gets <date_time>_<pid>."""
    path = ROOT_DIR / "outputs" / "research" / name / time.strftime("%Y-%m-%d_%H-%M-%S")
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        path = path.with_name(f"{path.name}_{os.getpid()}")
        path.mkdir(parents=True)
    return path


def episodes(task: str = "small", root: int = 0, per_level: int = 0, workers: int = 3):
    """The Small dev split (root 0, ``per_level`` 0), else a stratified private pool of ``per_level`` per level."""
    from shockbench_flow_agent import EpisodeSet

    if root == 0 and not per_level:
        return EpisodeSet.build(task, "dev", n_jobs=workers)
    from sbf_starter.evolve import pools

    levels = pools.stratified(task, root, [per_level] * 4, n_jobs=workers)
    return pools.episode_set(task, root, pools.take(levels), n_jobs=workers)


def world(task: str, root: int, n: int):
    """(instance, omega, true marks, fallback spec, policy seed) of episode ``n``, as the scorer builds them."""
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    return inst, omega, marks, fallback, _policy_seed(root, n, NO_ZIP_SHA256)


def components(records) -> dict[str, float]:
    """USD per cost component summed over a trajectory's records."""
    out: dict[str, float] = {}
    for r in records:
        for k, v in r.costs.as_dict().items():
            out[k] = out.get(k, 0.0) + v
    return out


def base_first_grids(inst) -> dict[int, list[int]]:
    """{grid ordinal: fab ordinals} of the base_first grids whose fabs draw energy (the grids the rule binds)."""
    out = {}
    for go, g in enumerate(inst.grids):
        fos = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if fos and inst.nodes[g].grid.priority == "base_first":
            out[go] = fos
    return out


def base_first_price(inst, T: int) -> dict[int, float]:
    """``oracle.lp._Builder.base_first_price`` per bound grid: V / min e_f, V the largest pi_d (x T at backlog) or nu.

    A price, so the planning rules' week-1 surrogate of the rule; ``base_first_bounds`` is the two-pass alternative.
    """
    V = max(
        max((d.pi * (T if d.backlog else 1) for d in inst.demands), default=0.0),
        max((st.salvage for st in inst.stock_slots), default=0.0),
    )
    return {go: V / min(inst.nodes[inst.fabs[fo]].fab.e for fo in fos) for go, fos in base_first_grids(inst).items()}


def base_first_bounds(model, x, grids: dict[int, list[int]], mode: str, first_week: int = 2):
    """Pass 2's column upper bounds from pass 1's solution ``x``, and (violations, bounds changed).

    A violation is a grid-week of the window (week >= ``first_week``) that sheds base load while its fabs draw energy,
    which the simulator never does (``dynamics.production.allocate_energy``, base_first). ``mode``:
    - "zero": every grid-week that sheds base load gets its fabs' energy capped at 0 (fabs dark that week).
    - "fix": only violations change: if the week's energy served base load and fabs together covers the base load,
      shed base load is capped at 0 (base served first, fabs get the rest); otherwise the fabs' energy is capped at 0.
    The model keys are one week's template (``model.columns``); week t's column j is (t - 1) * nc + j.
    """
    cols = {key: j for j, key in enumerate(model.columns)}
    nc = len(model.columns)
    ub = np.array(model.ub, dtype=float)
    violations = changed = 0
    for t in range(first_week, int(model.T) + 1):
        off = (t - 1) * nc
        for go, fos in grids.items():
            jsh, jy = cols.get(("ysh", go)), cols.get(("y", go))
            if jsh is None or jy is None:
                continue
            jE = [off + cols[("E", fo)] for fo in fos if ("E", fo) in cols]
            shed, served = float(x[off + jsh]), float(x[off + jy])
            energy = float(sum(x[j] for j in jE))
            tol = 1e-6 * max(shed + served, 1.0)
            if shed <= tol:
                continue
            violated = energy > tol
            violations += violated
            if mode == "zero":
                ub[jE] = 0.0
                changed += 1
            elif mode == "fix" and violated:
                if energy >= shed - tol:  # the fabs' energy would have covered the shed base load
                    ub[off + jsh] = 0.0
                else:
                    ub[jE] = 0.0
                changed += 1
    return ub, violations, changed


def add_cuts(model, arrays, inst, grids: dict[int, list[int]], first_week: int = 2):
    """The window model with the base-load-first MILP's LP-relaxation rows, one per bound grid-week >= ``first_week``:

        ysh_gt / ybar_gt + sum_f E_ft / M_gt <= 1,   M_gt = min(G-bar_gt sum_k zeta_gk, sum_f e_f alpha-bar_ft cap0_f),

    implied by sum_f E <= M z and ysh <= ybar (1 - z) with z in [0, 1]: a plan cannot shed a share of base load and
    still give the fabs more than the rest of their largest draw. Valid for every simulator week (the rule holds).
    """
    import scipy.sparse as sp

    cols = {k: j for j, k in enumerate(model.columns)}
    nc, H = len(model.columns), int(model.T)
    rows, cidx, vals = [], [], []
    r = 0
    for t in range(first_week, H + 1):
        off = (t - 1) * nc
        for go, fos in grids.items():
            yb = float(arrays["y_bar"][t - 1, go])
            gen = float(arrays["G_bar"][t - 1, go]) * sum(inst.nodes[inst.grids[go]].grid.shares.values())
            draw = sum(
                inst.nodes[inst.fabs[fo]].fab.e
                * float(model.ub[off + cols[("p", fo)]])
                / max(float(arrays["R"][t - 1, fo]), 1e-9)
                for fo in fos
                if ("p", fo) in cols
            )
            M = min(gen, draw)
            if yb <= 0 or M <= 0:
                continue
            rows.append(r)
            cidx.append(off + cols[("ysh", go)])
            vals.append(1.0 / yb)
            for fo in fos:
                if ("E", fo) in cols:
                    rows.append(r)
                    cidx.append(off + cols[("E", fo)])
                    vals.append(1.0 / M)
            r += 1
    C = sp.csr_matrix((vals, (rows, cidx)), shape=(r, len(model.lb)))
    return replace(model, A_ub=sp.vstack([model.A_ub, C]).tocsr(), b_ub=np.concatenate([model.b_ub, np.ones(r)]))


def window_milp(
    model, arrays, inst, grids: dict[int, list[int]], time_limit: float, first_week: int = 2, last_week=None, ub=None
):
    """The window LP plus base_first_milp.py's exact rule rows for weeks >= ``first_week``: (x or None, status).

    One binary z per bound grid-week: sum_f E <= M z and ysh <= ybar (1 - z), M = G-bar sum_k zeta, from the window's
    own marks (``arrays``). The objective is the window LP's (planning-rule prices included). HiGHS's best incumbent
    stands when the time limit stops it. ``last_week`` limits the binaries to weeks first_week..last_week; ``ub``
    replaces the model's column upper bounds (the "fix" bounds of the weeks after ``last_week``).
    """
    import scipy.sparse as sp
    from scipy.optimize import Bounds, LinearConstraint, milp

    cols = {k: j for j, k in enumerate(model.columns)}
    nc, H, n0 = len(model.columns), int(model.T), len(model.lb)
    rows, cidx, vals, rhs, zs = [], [], [], [], []
    for t in range(first_week, min(H, last_week or H) + 1):
        off = (t - 1) * nc
        for go, fos in grids.items():
            jz = n0 + len(zs)
            zs.append((t, go))
            M = float(arrays["G_bar"][t - 1, go]) * sum(inst.nodes[inst.grids[go]].grid.shares.values())
            r = len(rhs)
            for fo in fos:
                if ("E", fo) in cols:
                    rows.append(r)
                    cidx.append(off + cols[("E", fo)])
                    vals.append(1.0)
            rows.append(r)
            cidx.append(jz)
            vals.append(-M)
            rhs.append(0.0)
            yb = float(arrays["y_bar"][t - 1, go])
            rows += [r + 1, r + 1]
            cidx += [off + cols[("ysh", go)], jz]
            vals += [1.0, yb]
            rhs.append(yb)
    nz, N = len(zs), n0 + len(zs)
    A_ub = sp.vstack(
        [
            sp.hstack([model.A_ub, sp.csr_matrix((model.A_ub.shape[0], nz))]),
            sp.csr_matrix((vals, (rows, cidx)), shape=(len(rhs), N)),
        ]
    ).tocsr()
    A_eq = sp.hstack([model.A_eq, sp.csr_matrix((model.A_eq.shape[0], nz))]).tocsr()
    res = milp(
        np.concatenate([model.objective(), np.zeros(nz)]),
        integrality=np.concatenate([np.zeros(n0), np.ones(nz)]),
        bounds=Bounds(
            np.concatenate([model.lb, np.zeros(nz)]), np.concatenate([model.ub if ub is None else ub, np.ones(nz)])
        ),
        constraints=[
            LinearConstraint(A_ub, -np.inf, np.concatenate([model.b_ub, rhs])),
            LinearConstraint(A_eq, model.b_eq, model.b_eq),
        ],
        options={"time_limit": time_limit, "mip_rel_gap": 1e-4, "disp": False},
    )
    return (None if res.x is None else np.asarray(res.x[:n0], dtype=float)), int(res.status)


def make_planner(
    inst,
    marks,
    *,
    task="small",
    horizon="L",
    oracle=False,
    true_demand=False,
    two_pass=None,
    passes=2,
    milp_time=10.0,
    milp_weeks=None,
    price_all_weeks=False,
):
    """The package's ``mpc_det`` with optional true window marks, a horizon label and a two-pass base-load-first solve.

    ``horizon``: L (the planner's own), "L+8", "max(26,2L)" (48 on Small) or "full". ``oracle``: the window's
    disruption fields are the true marks (demand too with ``true_demand``). ``two_pass``: None, "zero" or "fix"
    (``base_first_bounds``); pass 2 warm-starts from pass 1's basis and the session's weekly chain keeps pass 1's.
    ``two_pass`` "cut" adds ``add_cuts``'s rows to the window LP, "cutfix" then the "fix" pass on what they leave.
    ``two_pass`` "milp" solves the window as the exact base-load-first MILP (``window_milp``, ``milp_time`` seconds,
    an offline test: far over the CPU budget), pass 1's plan standing when it has no incumbent.
    ``passes``: the most solves per week; each pass after the second adds the bounds of the violations its
    predecessor's plan still has, keeping the earlier ones.
    ``price_all_weeks``: the planning rules' week-1 price on shed base load charged in every window week (the
    rejected surrogate of the research document, section 3.3).
    ``planner.log`` holds one dict per week: seconds, violations before and after, bounds changed, pass 2's status.
    """
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
    from shockbench_flow.policies.naive_fq import generator_quantiles
    from shockbench_flow.policies.registry import PolicyContext

    grids = base_first_grids(inst)

    class Planner(MpcDet):
        def _horizon(self, inst_, plan):
            return inst_.T if horizon == "full" else super()._horizon(inst_, plan)

        def _window_arrays(self, inst_, obs, H_t):
            arr = dict(L.persistence_arrays(inst_, obs, self._memory, H_t))
            if oracle:
                sl = slice(int(obs["week"]) - 1, int(obs["week"]) - 1 + H_t)
                for k in L.WINDOW_FIELDS:
                    if k != "demand" or true_demand:
                        arr[k] = np.array(getattr(marks, k)[sl]).copy()
            return L.read_only(arr)

        def act(self, obs):
            start = time.perf_counter()
            inst_, t = self._inst, int(obs["week"])
            self._memory.update(inst_, obs)
            H_t = L.window_length(self._H, t, inst_.T)
            arrays_t = self._window_arrays(inst_, obs, H_t)
            model = L.rolled_lp(inst_, obs, arrays_t, H_t, planning_rules=True)
            if two_pass in ("cut", "cutfix"):
                model = add_cuts(model, arrays_t, inst_, grids)
            if price_all_weeks:
                pr = np.array(model.meta["priority"], dtype=float)
                nc = len(model.columns)
                for go in grids:
                    j = model.columns.index(("ysh", go))
                    pr[j::nc] = pr[j]
                model.meta["priority"] = pr
            res = self._session.solve(L.to_highs_lp(model), L.WindowShape.of(model))
            row = {"week": t, "violations": 0, "changed": 0, "after": None, "pass2": None, "extra": 0}
            if res.ok:
                mode = {"milp": None, "cut": None, "cutfix": "fix"}.get(two_pass, two_pass)
                ub, row["violations"], changed = base_first_bounds(model, res.x, grids, mode or "count")
                row["changed"] = changed
                if two_pass == "milp" and row["violations"]:
                    ub_mix = None
                    if milp_weeks:  # binaries for weeks 2..1+milp_weeks, "fix" bounds after them
                        ub_fix = base_first_bounds(model, res.x, grids, "fix")[0]
                        cut = (1 + milp_weeks) * len(model.columns)
                        ub_mix = np.concatenate([np.asarray(model.ub)[:cut], ub_fix[cut:]])
                    xm, row["pass2"] = window_milp(
                        model,
                        arrays_t,
                        inst_,
                        grids,
                        milp_time,
                        last_week=1 + milp_weeks if milp_weeks else None,
                        ub=ub_mix,
                    )
                    if xm is not None:
                        res = replace(res, x=xm)
                        row["after"] = base_first_bounds(model, xm, grids, "count")[1]
                kept = self._session.state()
                while mode and changed and row["extra"] < passes - 1:  # each pass keeps the earlier bounds
                    model2 = replace(model, ub=ub)
                    res2 = self._session.solve(L.to_highs_lp(model2), None)  # warm from the last pass's basis
                    row["extra"] += 1
                    row["pass2"] = row["pass2"] or res2.status
                    if not res2.ok:
                        break
                    model, res = model2, res2
                    ub, row["after"], changed = base_first_bounds(model, res.x, grids, mode)
                self._session.load_state(kept)  # next week's warm start chains from pass 1, as mpc_det's
            if res.ok:
                action = L.week1_action(inst_, model, res.x, obs, L.prohibited_now(self._memory, t))
            else:
                action = self._fallback.act(obs)
            row["ok"] = bool(res.ok)
            row["seconds"] = time.perf_counter() - start
            self.log.append(row)
            return action

    _i, gen = task_generator(task)
    params = MpcDetParams(H=horizon) if horizon not in ("L", "full") else MpcDetParams()
    planner = Planner(params, PolicyContext(fq_quantile=generator_quantiles(inst, gen)))
    planner.log = []
    return planner


def by_level(score: dict) -> dict[str, float]:
    """{level: RSS} from ``EpisodeSet.rss``'s result."""
    if "rss_by_stratum" in score:
        return {str(k): v for k, v in score["rss_by_stratum"].items()}
    return {str(k): v["rss"] for k, v in score["strata"].items()}


def fmt_row(name: str, score: dict) -> str:
    levels = by_level(score)
    cells = " | ".join("-" if levels.get(str(s)) is None else f"{levels[str(s)]:.3f}" for s in (1, 2, 3, 4))
    return f"| {name} | {score['rss']:.4f} | {cells} |"

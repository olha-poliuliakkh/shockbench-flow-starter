"""The frozen frame of the evolvable LP agent (docs/EVOLVE_DESIGN.md, section 4.1). The mutator never edits this file.

Each week: the Dict observation is turned back into the fields of the server's wire observation that the vendored
planner reads (``wire_from_dict``), the planner's memory is updated, ``State`` gives the blocks name-based views, the
persistence window of the package's ``mpc_det`` is built (``Window``, writable through consistency-keeping helpers),
the window LP is built by the oracle's own builder, edited (``LPEdits``) and solved by HiGHS's dual simplex through
the copy of HiGHS's bindings that SciPy ships (the server has no highspy), warm-started week to week (``WarmLP``), and
its first week becomes the Dict action. Each week solves the planner's own window first, then a 48-week window with
the CPU left, whose plan replaces the first when it finishes; each solve is two-pass when ``base_first`` is set
(``base_first_bounds``), and each window keeps its own warm-start chain. On Small the long window's second solve is
the exact base-load-first MILP of its first weeks (``window_mip``), the two-pass solve its fallback. ``sbfplan`` is
shockbench_flow 0.1.2, vendored by ``scripts/vendor_planner.py`` (MIT, see sbfplan/LICENSE).

Imports: the standard library, numpy, SciPy and the vendored package; no randomness anywhere.
"""

import math
import time
import types
from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from sbfplan.instance.schema import WAR_RISK_CLASSES
from sbfplan.policies import lp_common as L
from sbfplan.policies.naive_parts import cached_plan
from scipy.optimize import linprog


try:  # SciPy's own copy of HiGHS's bindings: a Highs object with bases and warm starts (linprog hides them)
    from scipy.optimize._highspy import _core as _HS
except ImportError:  # another SciPy build: cold linprog solves, the planner's own window only
    _HS = None
if _HS is not None:  # the vendored builder's model and basis helpers run on SciPy's bindings
    _SHIM = types.SimpleNamespace(**{k: getattr(_HS, k) for k in dir(_HS) if not k.startswith("__")})
    _SHIM.Highs = _HS._Highs
    L._highspy = lambda: _SHIM


CHANNELS = ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
KINDS = ("proposal", "final_notice", "threat", "publication", "withdrawal")
TARGET_KINDS = ("chokepoint", "edge", "node", "region")
# The window's LP size is capped deterministically (timing never changes a plan, so scores reproduce): measured
# cold dual-simplex solves took 0.12 s at 10,000 columns (Small, H = L) and 0.57 s at 27,800 (Full, H = L), growing
# about as columns^1.5, so these caps keep a solve near a quarter of the 2 s (Small) or 4 s (Full) week.
MAX_COLUMNS = {2.0: 26_000, 4.0: 40_000}
# CPU deadlines, in seconds of the week's own CPU, of (pass 1, pass 2) of each window: the planner's own window first
# (the plan of every week), then the long window with what is left (its plan replaces the first when it finishes). A
# solve gets what is left before its deadline as HiGHS's time limit and is skipped below 0.05 s; pass 1's plan stands
# when pass 2 has no time or no optimum. Week budgets: 2 s (Small) and 4 s (Full). Timing decides only which plan a
# long-running week plays, so scores reproduce elsewhere.
DEADLINES = {2.0: {"short": (0.5, 0.6), "long": (1.0, 1.5)}, 4.0: {"short": (1.6, 2.0), "long": (3.1, 3.4)}}
# The exact base-load-first window MILP (research doc 5.11): on the week's final window, after its two passes, one
# binary per bound grid-week for the window's weeks 2..1 + ``weeks`` (pass 1's "fix" bounds after them), seeded by the
# two-pass plan, solved by HiGHS's MIP until ``deadline`` (the week's CPU); its plan replaces the two-pass plan whenever
# it has one. None: no MILP.
# One-thread HiGHS overshoots its time limit by a median 0.05-0.08 s at a 0.9 s deadline, by 0.66 s past 1.1 s (a
# phase that does not check the clock): 0.85 keeps measured weeks at or under 1.7 s of Small's 2 s.
MIP = {2.0: {"weeks": 12, "deadline": 0.85}, 4.0: None}
THREADS = {"threads": 1}  # HiGHS on one thread, as on the server's single CPU; emptied if this process refuses it
LONG_H = "max(26,2L)"  # the long window's label: 48 weeks on Small and Full
TIME_LIMIT = "Time limit reached"  # HiGHS's model status of a solve its time limit stopped


def _highs(options: dict):
    """A fresh HiGHS object with ``options`` and ``THREADS``."""
    h = _HS._Highs()
    for k, v in {**options, **THREADS}.items():
        h.setOptionValue(k, v)
    return h


def _run(h, lp, options: dict, basis=None, start=None):
    """Pass ``lp`` (a starting basis, or a MIP ``start`` (columns, values)) and run, on one thread when HiGHS agrees."""
    h.passModel(lp)
    if basis is not None:
        h.setBasis(basis)  # a basis HiGHS refuses leaves the run cold
    if start is not None:
        h.setSolution(len(start[0]), start[0], start[1])
    if h.run() == _HS.HighsStatus.kError and THREADS:  # another solve started HiGHS's pool at another size
        _HS._Highs.resetGlobalScheduler(True)
        h = _highs(options)
        h.passModel(lp)
        if basis is not None:
            h.setBasis(basis)
        if start is not None:
            h.setSolution(len(start[0]), start[0], start[1])
        if h.run() != _HS.HighsStatus.kError:
            return h
        THREADS.clear()  # still refused: keep HiGHS's default
        h = _highs(options)
        h.passModel(lp)
        if basis is not None:
            h.setBasis(basis)
        if start is not None:
            h.setSolution(len(start[0]), start[0], start[1])
        h.run()
    return h


class WarmLP:
    """HiGHS's dual simplex on SciPy's bindings, warm-started from the last optimal basis, under a CPU deadline.

    The kept basis is shifted by the weeks since it was found (0 for a second pass in the same week) to the new
    window's shape (``lp_common._shift_arrays``: first weeks dropped, last week repeated) and is never dropped by a
    failed solve, so a timed-out attempt costs no later warm start. A warm solve that fails for another reason than
    time is retried cold. A solve with ``keep`` False (pass 2) starts from the chain's basis without replacing it, so
    each week's pass 1 starts from last week's pass 1, not from a basis bent by pass 2's bounds.
    """

    def __init__(self, cpu):
        self.cpu = cpu  # the week's CPU so far
        self.col = self.row = self.shape = None
        self.week = 0

    def solve(self, lp, shape, week: int, deadline: float, keep: bool = True) -> tuple[np.ndarray | None, str]:
        basis = None
        if self.col is not None:
            try:
                col, row = L._shift_arrays(self.col, self.row, self.shape, shape, max(0, week - self.week))
                if len(col) == lp.num_col_ and len(row) == lp.num_row_:
                    basis = L._to_basis(col, row, alien=True)
            except ValueError:  # another template (never expected within an episode): cold
                basis = None
        status = "not run"
        for start in (basis, None) if basis is not None else (None,):
            left = deadline - self.cpu()
            if left < 0.05:
                return None, "no time"
            options = {**L.HIGHS_OPTIONS, "time_limit": float(left)}
            h = _run(_highs(options), lp, options, start)
            status = h.modelStatusToString(h.getModelStatus())
            if status == L.OPTIMAL:
                x = np.array(h.getSolution().col_value, dtype=float)
                if x.shape == (lp.num_col_,) and np.all(np.isfinite(x)):
                    b = h.getBasis()
                    if keep and b.valid:
                        self.col, self.row = L._from_basis(b)
                        self.shape, self.week = shape, week
                    return x, status
            if status == TIME_LIMIT:
                break
        return None, status


def _pin_threads() -> None:
    """Fix HiGHS's thread pool at one thread before any other solve in the process starts it at the machine's count.

    HiGHS sizes its pool at the first run in a process (the nominal plan's ``linprog`` at ``Planner`` init, or the
    harness's own solves, start it with every core); a later run asking for one thread then fails. The pool is reset
    first (``resetGlobalScheduler``; later default-sized solves still run). One thread is what the server's single CPU
    gives anyway, and keeps the MILP's CPU equal to its wall-clock time limit.
    """
    if _HS is None or not THREADS:
        return
    _HS._Highs.resetGlobalScheduler(True)
    lp = _HS.HighsLp()
    lp.num_col_, lp.num_row_ = 1, 0
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = np.array([1.0]), np.zeros(1), np.ones(1)
    lp.a_matrix_.format_ = _HS.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = 1, 0
    lp.a_matrix_.start_ = np.array([0, 0], dtype=np.int32)
    h = _highs({"output_flag": False})
    h.passModel(lp)
    if h.run() != _HS.HighsStatus.kOk:
        THREADS.clear()


_pin_threads()


def window_mip(
    model,
    c,
    lb,
    ub,
    arrays,
    inst,
    grids: dict[int, list[int]],
    weeks: int,
    time_limit: float,
    x0=None,
    z_prev=None,
    t0=1,
):
    """The window's plan under the exact base-load-first rule for weeks 2..1 + ``weeks``, or None.

    The window LP (cost ``c``, bounds [lb, ub]) plus, per bound grid g and week t, a binary z with
    sum_f E_ft <= M z and ysh_gt <= ybar_gt (1 - z), M = G-bar_gt sum_k zeta_gk from the window's marks (``arrays``):
    fabs draw energy only when base load is fully served, as the simulator's allocation does. HiGHS's MIP runs until
    ``time_limit`` (seconds). Returns (plan or None, {(absolute week, grid): z} of the plan). The start: ``z_prev``
    (last week's MILP, by absolute week; window week t is week t0 + t - 1), else from ``x0`` (the two-pass plan):
    z = 1 where x0 serves the base load in full, 0 elsewhere (fabs dark, always feasible). HiGHS completes the start
    into a first incumbent with one LP, so the search carries over from week to week.
    """
    cols = {k: j for j, k in enumerate(model.columns)}
    nc, H, n0 = len(model.columns), int(model.T), len(lb)
    rows, idx, vals, rhs, z0, zkeys = [], [], [], [], [], []
    nz = 0
    for t in range(2, min(H, 1 + weeks) + 1):
        off = (t - 1) * nc
        for go, fos in grids.items():
            jz, r = n0 + nz, len(rhs)
            nz += 1
            M = float(arrays["G_bar"][t - 1, go]) * sum(inst.nodes[inst.grids[go]].grid.shares.values())
            for fo in fos:
                if ("E", fo) in cols:
                    rows.append(r)
                    idx.append(off + cols[("E", fo)])
                    vals.append(1.0)
            yb = float(arrays["y_bar"][t - 1, go])
            if z_prev is not None and (t0 + t - 1, go) in z_prev:
                z0.append(z_prev[(t0 + t - 1, go)])
            elif x0 is not None:
                z0.append(1.0 if float(x0[off + cols[("ysh", go)]]) <= 1e-6 * max(1.0, yb) else 0.0)
            else:
                z0.append(0.0)
            zkeys.append((t0 + t - 1, go))
            rows += [r, r + 1, r + 1]
            idx += [jz, off + cols[("ysh", go)], jz]
            vals += [-M, 1.0, yb]
            rhs += [0.0, yb]
    if not nz:
        return None, {}
    N, n_ub = n0 + nz, model.A_ub.shape[0]
    A = sp.vstack(
        [
            sp.hstack([model.A_ub, sp.csr_matrix((n_ub, nz))]),
            sp.csr_matrix((vals, (rows, idx)), shape=(len(rhs), N)),
            sp.hstack([model.A_eq, sp.csr_matrix((model.A_eq.shape[0], nz))]),
        ]
    ).tocsc()
    inf = _HS.kHighsInf
    lo, hi = np.concatenate([lb, np.zeros(nz)]), np.concatenate([ub, np.ones(nz)])
    lp = _HS.HighsLp()
    lp.num_col_, lp.num_row_ = N, A.shape[0]
    lp.col_cost_ = np.concatenate([c, np.zeros(nz)])
    lp.col_lower_ = np.where(np.isinf(lo), -inf, lo)
    lp.col_upper_ = np.where(np.isinf(hi), inf, hi)
    lp.row_lower_ = np.concatenate([np.full(n_ub + len(rhs), -inf), model.b_eq])
    lp.row_upper_ = np.concatenate([model.b_ub, rhs, model.b_eq])
    lp.offset_ = float(L.model_offset(model))
    lp.a_matrix_.format_ = _HS.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = N, A.shape[0]
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.integrality_ = [_HS.HighsVarType.kContinuous] * n0 + [_HS.HighsVarType.kInteger] * nz
    options = {"output_flag": False, "time_limit": float(time_limit), "mip_rel_gap": 1e-3}
    start = (np.arange(n0, N, dtype=np.int32), np.array(z0, dtype=float))
    h = _run(_highs(options), lp, options, start=start)
    if h.getInfo().primal_solution_status != _HS.kSolutionStatusFeasible:
        return None, {}
    full = np.array(h.getSolution().col_value, dtype=float)
    x = full[:n0]
    if not np.all(np.isfinite(x)):
        return None, {}
    return x, {key: float(round(v)) for key, v in zip(zkeys, full[n0:])}


def base_first_grids(inst) -> dict[int, list[int]]:
    """{grid ordinal: fab ordinals} of the base_first grids whose fabs draw energy (the grids the rule binds)."""
    out = {}
    for go, g in enumerate(inst.grids):
        fos = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if fos and inst.nodes[g].grid.priority == "base_first":
            out[go] = fos
    return out


def base_first_bounds(model, x, grids: dict[int, list[int]], mode: str, ub) -> tuple[np.ndarray, int]:
    """Pass 2's upper bounds from pass 1's solution ``x`` (a copy of ``ub``), and the number of grid-weeks changed.

    The simulator serves a grid's base load before any fab (``dynamics.production.allocate_energy``, base_first): a
    week that sheds base load gives its fabs no energy. The window LP prices that only in week 1, so its plan for
    weeks 2..H may shed base load while powering fabs, and schedule lots the simulator will not start. ``mode``:
    "zero" caps the fabs' energy at 0 in every grid-week >= 2 that sheds base load; "fix" changes only the violated
    grid-weeks: shed base load capped at 0 where the fabs' energy would have covered it, else the fabs' energy.
    """
    cols = {key: j for j, key in enumerate(model.columns)}
    nc = len(model.columns)
    ub = np.array(ub, dtype=float)
    changed = 0
    for t in range(2, int(model.T) + 1):
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
            if mode == "zero":
                ub[jE] = 0.0
                changed += 1
            elif mode == "fix" and energy > tol:
                if energy >= shed - tol:  # the fabs' energy would have covered the shed base load
                    ub[off + jsh] = 0.0
                else:
                    ub[jE] = 0.0
                changed += 1
    return ub, changed


def _entries(obs: dict, block: str, fields: tuple[str, ...]) -> list[tuple]:
    """The live entries of a padded list block, as tuples of ``fields`` (None where a field is unobserved)."""
    seen = obs[f"{block}.{fields[0]}.observed"].astype(bool)
    cols = []
    for f in fields:
        vals, ok = obs[f"{block}.{f}"], obs[f"{block}.{f}.observed"].astype(bool)
        cols.append([v.item() if o else None for v, o in zip(vals[seen], ok[seen])])
    return list(zip(*cols))


def _cols(rows: list[tuple], fields: tuple[str, ...]) -> dict:
    return {f: [r[i] for r in rows] for i, f in enumerate(fields)}


def _vals(a: np.ndarray, seen: np.ndarray) -> list:
    return [x.item() if s else None for x, s in zip(a, seen.astype(bool))]


def wire_from_dict(obs: dict, cfg: dict, entry_edge: dict) -> dict:
    """The wire-observation fields the planner reads, rebuilt from the agent's Dict observation.

    On Small and Full the pipeline arrives grouped by (edge, k, lane, arrival week) and the queued lots as a dense
    (lot key, arrival week) block: a group is one shipment or lot here, which the LP treats alike. A lot's entry edge
    is its lane's edge into the strait and its dispatch week that edge's nominal lead before arrival; the LP reads
    neither (tests/test_planner_port.py checks the objective week by week).
    """
    lay, edges = cfg["layout"], cfg["static"]["edges"]
    week = int(obs["week"][0])
    stock = [(n, k, float(q)) for (n, k), q in zip(lay["stock_slots"], obs["stock.qty"])]
    backlog = [(n, k, float(q)) for (n, k), q in zip(lay["demands"], obs["backlog.qty"])]
    pfields = ("edge", "k", "lane", "qty", "arrival_week")
    if "lot_keys" in lay:
        lots = []
        qty, seen = obs["queue_lots.qty"], obs["queue_lots.qty.observed"]
        for i, j in zip(*np.nonzero(seen)):
            c, k, lane, nxt = lay["lot_keys"][i]
            e = entry_edge[(c, lane)]
            arr = int(j) + 1
            lots.append((int(c), int(k), float(qty[i, j]), int(lane), int(nxt), arr, arr - int(edges["tau0"][e]), e))
    else:  # tiny: the lots as listed
        lots = _entries(
            obs,
            "queue_lots",
            ("chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge"),
        )
    lfields = ("chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge")
    wire = {
        "week": week,
        "stock": _cols(stock, ("node", "k", "qty")),
        "backlog": _cols(backlog, ("node", "k", "qty")),
        "pipeline": _cols(_entries(obs, "pipeline", pfields), pfields),
        "queue_lots": _cols(lots, lfields),
        "wip": _cols(_entries(obs, "wip", ("node", "k", "qty", "out_week")), ("node", "k", "qty", "out_week")),
    }
    if not (obs["graph_now.open.observed"].any() or obs["graph_now.u.observed"].any()):
        wire["graph_now"] = None  # a blackout week (none in the scored regime)
    else:
        pe, pk = np.nonzero(obs["graph_now.prohibited"])
        te, tk = np.nonzero(obs["graph_now.tariff"])
        wr = obs["graph_now.war_risk"]
        wseen = obs["graph_now.war_risk.observed"].astype(bool)
        supply = [
            (n, k, float(v))
            for (n, k), v, s in zip(
                lay["supply_slots"], obs["graph_now.supply.avail"], obs["graph_now.supply.avail.observed"]
            )
            if s
        ]
        wire["graph_now"] = {
            "u": _vals(obs["graph_now.u"], obs["graph_now.u.observed"]),
            "c": _vals(obs["graph_now.c"], obs["graph_now.c.observed"]),
            "open": _vals(obs["graph_now.open"], obs["graph_now.open.observed"]),
            "kappa": {
                b: _vals(obs[f"graph_now.kappa.{b}"], obs[f"graph_now.kappa.{b}.observed"]) for b in ("tb", "ct")
            },
            "war_risk": [WAR_RISK_CLASSES[int(v)] if s else None for v, s in zip(wr, wseen)],
            "supply": _cols(supply, ("node", "k", "avail")),
            "fab": {
                "node": list(lay["fabs"]),
                **{
                    f: _vals(obs[f"graph_now.fab.{f}"], obs[f"graph_now.fab.{f}.observed"])
                    for f in ("R", "alpha_bar", "cap_eff")
                },
            },
            "osat": {
                "node": list(lay["osats"]),
                **{f: _vals(obs[f"graph_now.osat.{f}"], obs[f"graph_now.osat.{f}.observed"]) for f in ("R", "thr_eff")},
            },
            "grid": {
                "node": list(lay["grids"]),
                **{
                    f: _vals(obs[f"graph_now.grid.{f}"], obs[f"graph_now.grid.{f}.observed"])
                    for f in ("G_bar", "y_bar")
                },
            },
            "prohibited": {"edge": pe.tolist(), "k": pk.tolist()},
            "tariff": {"edge": te.tolist(), "k": tk.tolist(), "rate": obs["graph_now.tariff"][te, tk].tolist()},
        }
    pend = _entries(obs, "pending_prohibitions", ("edge", "k", "effective_week"))
    wire["pending_prohibitions"] = _cols(pend, ("edge", "k", "effective_week"))
    fq, fs = obs["demand_forecast.qty"], obs["demand_forecast.qty.observed"]
    fc = [(n, k, h, float(fq[d, h])) for d, (n, k) in enumerate(lay["demands"]) for h in range(fq.shape[1]) if fs[d, h]]
    wire["demand_forecast"] = _cols(fc, ("node", "k", "h", "qty"))
    return wire


class State:
    """What the blocks read: the week, the planner's memory, positions, routes and signals, by name.

    Attributes: ``week``, ``T``, ``obs`` (the Dict observation), ``wire`` (the rebuilt wire fields), ``chokepoints``
    (node indices, the window's column order), ``chokepoint_names``, ``open_now`` (open fraction per strait),
    ``closed_for`` (weeks each strait has been below fully open), ``warnings`` ({(kind, id): score}), ``threads``
    (live announcement threads, one dict per message: msg_id, channel, kind, region, target_kind, target, k,
    announced_week, stated_effective_week), ``pending`` ((edge, k, effective week) triples), ``regime`` (the published
    decoy shares ``phi`` and warning parameters), ``lanes_through`` ({strait node: [lane indices]}), ``edges_of_lane``,
    ``edge_names``, ``node_names``, ``node_region``, ``commodity_names``, ``demands`` ((node, k) per demand row),
    ``grids`` (node indices), ``demand_forecast`` ((demands, 8) array).
    """

    def __init__(self, planner: "Planner", obs: dict, wire: dict):
        cfg, st = planner.cfg, planner.cfg["static"]
        self.week, self.T, self.obs, self.wire = wire["week"], planner.inst.T, obs, wire
        self.regime = cfg.get("regime") or st.get("regime") or {}
        self.chokepoints = list(cfg["layout"]["chokepoints"])
        self.node_names = list(st["nodes"]["id"])
        self.node_region = list(st["nodes"]["region"])
        self.chokepoint_names = [self.node_names[c] for c in self.chokepoints]
        self.edge_names = list(st["edges"]["id"])
        self.commodity_names = list(st["commodities"]["id"])
        self.demands = [tuple(d) for d in cfg["layout"]["demands"]]
        self.grids = list(cfg["layout"]["grids"])
        self.open_now = np.asarray(obs["graph_now.open"], dtype=float)
        self.closed_for = planner.closed_for.copy()
        self.warnings = {
            (kind, int(unit)): float(v)
            for (kind, unit), v, s in zip(
                cfg["layout"]["warning_units"], obs["warning.score"], obs["warning.score.observed"]
            )
            if s
        }
        fields = (
            "msg_id",
            "channel",
            "kind",
            "region",
            "target_kind",
            "target",
            "k",
            "announced_week",
            "stated_effective_week",
        )
        self.threads = []
        for row in _entries(obs, "messages", fields):
            m = dict(zip(fields, row))
            m["channel"] = CHANNELS[m["channel"]] if m["channel"] is not None else None
            m["kind"] = KINDS[m["kind"]] if m["kind"] is not None else None
            m["target_kind"] = TARGET_KINDS[m["target_kind"]] if m["target_kind"] is not None else None
            self.threads.append(m)
        self.pending = list(planner.memory.pending)
        self.lanes_through = planner.lanes_through
        self.edges_of_lane = planner.edges_of_lane
        self.demand_forecast = np.asarray(obs["demand_forecast.qty"], dtype=float)
        self._slot_index = planner.inst.slot_index

    def decoy_share(self, channel: str) -> float:
        """The published share of decoy threads on ``channel`` (0 for channels without a shadow process)."""
        return float((self.regime.get("phi") or {}).get(channel, 0.0))

    def stock_slot(self, node: int, k: int) -> int | None:
        """The LP's stock-slot index of (node, commodity), for ``LPEdits.stock_floor``; None if it has no stock.

        It is the instance's own slot ordinal (straits included), not the position in ``stock.qty``.
        """
        return self._slot_index.get((int(node), int(k)))

    def strait_pos(self, node: int) -> int | None:
        """The window column of strait ``node``, or None if it is no strait."""
        return self.chokepoints.index(node) if node in self.chokepoints else None


class Window:
    """The H-week forecast the LP plans on (row h is week t + h). Edit it only through the helpers.

    Arrays: ``a[name]`` for every field of ``lp_common.WINDOW_FIELDS`` (u, c, o, kappa, supply, G_bar, y_bar, R,
    alpha_bar, sigma_scr, R_osat, demand, prohibited, tariff, wr_class, h_queue, c_wr and the ``*_now`` copies).
    Strait columns follow ``State.chokepoints``; edges, commodities and stock slots follow Static's tables.
    """

    def __init__(self, arrays: dict, t: int, inst):
        self.a = {k: np.array(v) for k, v in arrays.items()}  # writable copies
        self.t, self.H, self._inst = t, int(self.a["o"].shape[0]), inst
        self._kappa_full = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=float)
        self._wr_changed = False

    def _rows(self, start: int, end: int | None) -> slice:
        return slice(max(0, int(start)), self.H if end is None else max(0, min(self.H, int(end))))

    def scale_open(self, pos: int, factor: float, start: int = 0, end: int | None = None) -> None:
        """Multiply strait ``pos``'s open fraction (and its pools' throughput) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        self.a["o"][r, pos] = np.clip(self.a["o"][r, pos] * max(0.0, float(factor)), 0.0, 1.0)
        self.a["kappa"][r, pos, :] = self._kappa_full[pos] * self.a["o"][r, pos][:, None]

    def set_open(self, pos: int, value: float, start: int = 0, end: int | None = None) -> None:
        """Set strait ``pos``'s open fraction to ``value`` (0..1) in rows start..end-1, throughput alike."""
        r = self._rows(start, end)
        self.a["o"][r, pos] = float(np.clip(value, 0.0, 1.0))
        self.a["kappa"][r, pos, :] = self._kappa_full[pos] * self.a["o"][r, pos][:, None]

    def scale_capacity(self, edge: int, factor: float, start: int = 0, end: int | None = None) -> None:
        """Multiply edge ``edge``'s weekly capacity by ``factor`` (>= 0) in rows start..end-1."""
        r = self._rows(start, end)
        self.a["u"][r, edge] = self.a["u"][r, edge] * max(0.0, float(factor))

    def prohibit(self, edge: int, k: int, on: bool = True, start: int = 0, end: int | None = None) -> None:
        """Switch the prohibition of (edge, k) on or off in rows start..end-1."""
        self.a["prohibited"][self._rows(start, end), edge, k] = bool(on)

    def set_tariff(self, edge: int, k: int, rate: float, start: int = 0, end: int | None = None) -> None:
        """Set the ad valorem tariff rate of (edge, k) in rows start..end-1."""
        self.a["tariff"][self._rows(start, end), edge, k] = max(0.0, float(rate))

    def scale_demand(self, factor: float, row: int | None = None, start: int = 0, end: int | None = None) -> None:
        """Multiply demand (one demand row, or all) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        cols = slice(None) if row is None else row
        self.a["demand"][r, cols] = self.a["demand"][r, cols] * max(0.0, float(factor))

    def scale_supply(self, factor: float, slot: int | None = None, start: int = 0, end: int | None = None) -> None:
        """Multiply supply availability (one stock slot, or all) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        cols = slice(None) if slot is None else slot
        self.a["supply"][r, cols] = self.a["supply"][r, cols] * max(0.0, float(factor))

    def set_war_risk(self, pos: int, cls: int, start: int = 0, end: int | None = None) -> None:
        """Set strait ``pos``'s war-risk class (0 none, 1 red_sea, 2 hormuz_2026) in rows start..end-1.

        Queue holding and transit war-risk costs follow the class.
        """
        self.a["wr_class"][self._rows(start, end), pos] = int(cls)
        self._wr_changed = True

    def arrays(self) -> dict:
        """The window as the LP builder takes it: ``*_now`` copies and war-risk costs brought in line, read-only."""
        if self._wr_changed:
            hq, cwr = L.window_queue_and_transit(self._inst, self.a["wr_class"])
            self.a["h_queue"], self.a["c_wr"] = np.array(hq), np.array(cwr)
        out = L.with_now(dict(self.a))
        return L.read_only({k: np.array(v) for k, v in out.items()})


class LPEdits:
    """Edits of the window LP applied after it is built. Weeks are window rows 0..H-1 (row 0 is this week).

    Column tags of the oracle's model: x (edge, k, lane) flows, I (stock slot) end-of-week stock, Q (strait, k, lane)
    queues, O (slot) disposal, ysh (grid ordinal) power shed, U (demand ordinal) unserved demand, D served demand, and
    production and energy columns (p, E, xi, lam, G, short, y, rho, lift).
    """

    def __init__(self):
        self.premiums: list[tuple] = []  # (edge, k or None, usd per unit, start, end)
        self.floors: list[tuple] = []  # (stock slot, qty, start, end)
        self.scales: list[tuple] = []  # (tag, first key field or None, factor, start, end)

    def premium(self, edge: int, usd_per_unit: float, k: int | None = None, start: int = 0, end: int | None = None):
        """Add a cost per unit shipped on ``edge`` (commodity ``k`` or all): a risk premium on that route."""
        self.premiums.append((int(edge), k, float(usd_per_unit), start, end))
        return self

    def stock_floor(self, slot: int, qty: float, start: int = 0, end: int | None = None):
        """Ask the LP to keep at least ``qty`` in stock slot ``slot`` (``s.stock_slot(node, k)``) at the end of rows
        start..end-1. A hard bound: the LP may hold stock back from demand to meet it, or find no plan at all."""
        self.floors.append((int(slot), max(0.0, float(qty)), start, end))
        return self

    def scale_cost(self, tag: str, factor: float, first: int | None = None, start: int = 0, end: int | None = None):
        """Multiply the objective coefficient of columns ``tag`` (whose first key field is ``first``, or all)."""
        self.scales.append((tag, first, float(factor), start, end))
        return self


class Planner:
    """The package's mpc_det (horizon H = L, persistence forecast, oracle LP, week-1 action), solved by SciPy.

    ``base_first`` None solves once, as mpc_det; "zero" or "fix" adds the two-pass base-load-first solve
    (``base_first_bounds``). ``long_horizon`` tries a ``LONG_H`` window first each week (``attempts``).
    """

    def __init__(self, config: dict, base_first: str | None = None, long_horizon: bool = True):
        if base_first not in (None, "zero", "fix"):
            raise ValueError(f"base_first must be None, 'zero' or 'fix', got {base_first!r}")
        self.cfg = config
        self.inst, plan = cached_plan(config["static"], None)
        self.L = L.lead_time_L(self.inst, plan)
        self.H = L.horizon_length(L.CANONICAL_H, self.L)
        self.memory = L.ObservedGraph.nominal(self.inst)
        self.budget = 2.0 if self.inst.T <= 52 else 4.0
        self.base_first = base_first
        self.grids = base_first_grids(self.inst)
        self.pass2 = {"weeks": 0, "used": 0}  # weeks pass 2 ran, weeks its plan was played (diagnostics only)
        self.mip = {"weeks": 0, "used": 0}  # weeks the window MILP ran, weeks its plan was played (diagnostics only)
        self._final_kind = "short"
        self._z_prev = None  # last week's MILP yes/no values by (absolute week, grid): next week's start
        self._week = 1
        self.H_long = L.horizon_length(LONG_H, self.L) if long_horizon and _HS is not None else None
        self.lp = {"short": WarmLP(self.cpu), "long": WarmLP(self.cpu)} if _HS is not None else None
        self.skip_long_until = 0  # after a long window ran out of time, the next week skips it
        self.horizons = {"long": 0, "short": 0, "long_failed": 0, "none": 0}  # weeks by window played (diagnostics)
        self.week_columns = None  # LP columns per window week, known after the first build
        self.closed_for = np.zeros(len(self.inst.chokepoints), dtype=int)
        st = config["static"]
        lanes = st["lanes"]
        self.edges_of_lane = [list(e) for e in lanes["edges"]]
        self.lanes_through = {}
        self.entry_edge = {}
        heads = st["edges"]["head"]
        for li, (es, cs) in enumerate(zip(lanes["edges"], lanes["chokepoints"])):
            for c in cs:
                self.lanes_through.setdefault(c, []).append(li)
                self.entry_edge[(c, li)] = next(e for e in es if heads[e] == c)
        a = config["spaces"]["action"]
        self.n_slots = a["flows"]["shape"][0]
        self.n_override = a["override_qty"]["shape"][0]
        self.pairs = {tuple(p): i for i, p in enumerate(config["layout"]["release_pairs"])}
        ov = st["override_slots"]
        self.override_pair = [self.pairs[(c, k)] for c, k in zip(ov["chokepoint"], ov["k"])]
        u0 = st["edges"]["u0"]
        self.capacity = np.array([u0[e] or 0.0 for e in st["action_slots"]["edge"]], dtype=float)
        self._week_start = None
        self.last_cpu = 0.0
        self.model = None
        self.last_status = None
        self.last_objective = None

    # ----- the week -------------------------------------------------------------------------------------------------
    def observe(self, obs: dict) -> State:
        """Rebuild the wire fields, update the memory, and return the week's ``State``."""
        self._week_start = time.thread_time()  # this thread's CPU this week (HiGHS runs on it), closed by _done()
        self._week = int(np.asarray(obs["week"]).ravel()[0])
        wire = wire_from_dict(obs, self.cfg, self.entry_edge)
        self.memory.update(self.inst, wire)
        open_now = np.asarray(obs["graph_now.open"], dtype=float)
        self.closed_for = np.where(open_now < 1.0, self.closed_for + 1, 0)
        return State(self, obs, wire)

    def cpu(self) -> float:
        """This thread's CPU seconds this week so far (0 before ``observe``).

        The thread's, not the process's: a harness's idle solver threads (a HiGHS pool sized to the machine's cores)
        can spin into the process's CPU between our solves. HiGHS runs on this thread (``THREADS``); on the server's
        single CPU the two agree.
        """
        return 0.0 if self._week_start is None else time.thread_time() - self._week_start

    def attempts(self, s: State, H0: int) -> list[tuple[int, str]]:
        """The week's windows, in order: (H0, "short"), then (H, "long") when a long window is worth trying.

        The long window starts in week 2 (its chain is seeded from the short window's basis, a cold 48-week solve on
        Full taking about 3 s), is skipped near the episode's end where it would be no longer than H0, and for the
        week after one where it ran out of time.
        """
        out = [(H0, "short")]
        if self.H_long and s.week >= max(2, self.skip_long_until):
            H = L.window_length(self.H_long, s.week, self.inst.T)
            if H > H0:
                out.append((H, "long"))
        self._final_kind = out[-1][1]  # the window whose plan the week plays when it solves: the MILP's
        return out

    def note(self, s: State, kind: str | None, failed_long: bool) -> None:
        """Count the window played this week; a long window that failed skips the next week's long attempt."""
        self.horizons[kind or "none"] += 1
        if failed_long:
            self.horizons["long_failed"] += 1
            self.skip_long_until = s.week + 2

    def max_H(self) -> int:
        """The longest window the LP-size cap allows (H itself before the first build)."""
        if not self.week_columns:
            return self.H
        return max(1, MAX_COLUMNS[self.budget] // self.week_columns)

    def default_H(self, s: State) -> int:
        return L.window_length(min(self.H, self.max_H()), s.week, self.inst.T)

    def clamp_H(self, s: State, H) -> int:
        """A block's H, kept in [ceil(L/2), the LP-size cap] and within the episode."""
        lo = math.ceil(self.L / 2)
        hi = max(lo, self.max_H())
        return L.window_length(min(max(int(round(float(H))), lo), hi), s.week, self.inst.T)

    def window(self, s: State, H: int) -> Window:
        return Window(L.persistence_arrays(self.inst, s.wire, self.memory, H), s.week, self.inst)

    def solve(self, s: State, w: Window, edits: LPEdits | None = None, kind: str = "short"):
        """Build the window LP, apply ``edits``, solve (two passes with ``base_first``) under ``kind``'s deadlines.

        Returns the wire action, or None when no pass returned an optimum in time.
        """
        arrays = w.arrays()
        model = L.rolled_lp(self.inst, s.wire, arrays, w.H, planning_rules=True)
        c = np.array(model.objective(), dtype=float)
        lb, ub = np.array(model.lb, dtype=float), np.array(model.ub, dtype=float)
        if edits is not None and (edits.premiums or edits.floors or edits.scales):
            self._apply(model, edits, c, lb, ub)
        self.model = model
        self.week_columns = self.week_columns or len(model.columns)
        d1, d2 = DEADLINES[self.budget][kind]
        x = self._solve(model, c, lb, ub, s.week, d1, kind)
        if x is None:
            return None
        x1 = x
        if self.base_first and self.grids:
            try:  # pass 2 can only replace pass 1's plan, never lose it
                ub2, changed = base_first_bounds(model, x, self.grids, self.base_first, ub)
                if changed:
                    self.pass2["weeks"] += 1
                    x2 = self._solve(model, c, lb, np.maximum(ub2, lb), s.week, d2, kind, keep=False)
                    if x2 is not None:
                        self.pass2["used"] += 1
                        x = x2
            except Exception:
                pass
        xm = self._mip(model, x1, x, c, lb, ub, arrays, kind)
        if xm is not None:  # rule-exact for its weeks; the two-pass plan's objective can hide leftover violations
            self.mip["used"] += 1
            x = xm
        self.last_objective = float(model.objective() @ x) + L.model_offset(model)
        return L.week1_action(self.inst, model, x, s.wire, L.prohibited_now(self.memory, s.week))

    def _mip(self, model, x1, x2, c, lb, ub, arrays, kind: str) -> np.ndarray | None:
        """The window MILP's plan (``MIP``) on this week's final window, seeded by the two-pass plan ``x2``, or None.

        ``x1`` is pass 1's plan (its "fix" bounds hold after the MILP's weeks). Never raises.
        """
        mip = MIP.get(self.budget)
        if not mip or kind != self._final_kind or _HS is None or not (self.base_first and self.grids):
            return None
        try:
            ub_fix, changed = base_first_bounds(model, x1, self.grids, "fix", ub)
            left = mip["deadline"] - self.cpu()
            if not changed or left < 0.1:
                return None
            self.mip["weeks"] += 1
            cut = (1 + mip["weeks"]) * len(model.columns)  # binaries for weeks 2..1 + weeks, pass 1's bounds after
            ub_mix = np.maximum(np.concatenate([ub[:cut], ub_fix[cut:]]), lb)
            x, z = window_mip(
                model, c, lb, ub_mix, arrays, self.inst, self.grids, mip["weeks"], left, x2, self._z_prev, self._week
            )
            if z:  # a MILP with no plan keeps last week's values for the next start
                self._z_prev = z
            return x
        except Exception:
            return None

    def _solve(
        self, model, c, lb, ub, week: int, deadline: float, kind: str = "short", keep: bool = True
    ) -> np.ndarray | None:
        """One LP solve of ``model`` with cost ``c``, bounds [lb, ub]: warm on ``kind``'s chain, else cold linprog."""
        if self.lp is not None:
            chain = self.lp[kind]
            if chain.col is None and self.lp["short"].col is not None:  # the long chain starts from the short basis
                short = self.lp["short"]
                chain.col, chain.row, chain.shape, chain.week = short.col, short.row, short.shape, short.week
            lp = L.to_highs_lp(replace(model, lb=lb, ub=ub), objective=c)
            x, self.last_status = chain.solve(lp, L.WindowShape.of(model), week, deadline, keep)
            return x
        left = deadline - self.cpu()
        if left < 0.05:
            self.last_status = "no time"
            return None
        kw = {}
        if model.A_ub.shape[0]:
            kw.update(A_ub=model.A_ub, b_ub=model.b_ub)
        if model.A_eq.shape[0]:
            kw.update(A_eq=model.A_eq, b_eq=model.b_eq)
        res = linprog(c, bounds=np.column_stack([lb, ub]), method="highs-ds", options={"time_limit": left}, **kw)
        self.last_status = int(res.status)
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        return np.asarray(res.x, dtype=float)

    def solve_safely(self, s: State, w: Window, edits: LPEdits | None = None, kind: str = "short"):
        try:
            return self.solve(s, w, edits, kind)
        except Exception:
            return None

    def _apply(self, model, edits: LPEdits, c, lb, ub) -> None:
        nc = len(model.columns)
        cols = model.columns
        for edge, k, usd, a, b in edits.premiums:
            js = [j for j, key in enumerate(cols) if key[0] == "x" and key[1] == edge and (k is None or key[2] == k)]
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                c[[h * nc + j for j in js]] += usd
        slot_col = {key[1]: j for j, key in enumerate(cols) if key[0] == "I"}
        for slot, qty, a, b in edits.floors:
            j = slot_col.get(slot)
            if j is None:
                continue
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                lb[h * nc + j] = min(max(lb[h * nc + j], qty), ub[h * nc + j])
        for tag, first, factor, a, b in edits.scales:
            js = [j for j, key in enumerate(cols) if key[0] == tag and (first is None or key[1] == first)]
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                c[[h * nc + j for j in js]] *= factor

    # ----- actions --------------------------------------------------------------------------------------------------
    def _done(self) -> None:
        """Record the week's own CPU (``last_cpu``, for diagnostics; it never changes a plan)."""
        if self._week_start is not None:
            self.last_cpu = time.thread_time() - self._week_start
            self._week_start = None

    def send_max(self, obs: dict) -> dict:
        self._done()
        return {"flows": self.capacity * obs["action_mask"]}

    def to_dict_action(self, wire: dict) -> dict:
        """The Dict action of a wire action: flows by slot, override quantities and a release mode per pair."""
        self._done()
        flows = np.zeros(self.n_slots)
        f = wire.get("flows") or {}
        for sl, q in zip(f.get("slot", []), f.get("qty", [])):
            flows[sl] = q
        override = np.zeros(self.n_override)
        modes = np.zeros(len(self.pairs), dtype=np.int64)
        ov = wire.get("overrides") or {}
        for o, q in zip(ov.get("slot", []), ov.get("qty", [])):
            override[o] = q
            modes[self.override_pair[o]] = 1
        hold = wire.get("hold") or {}
        for c, k in zip(hold.get("chokepoint", []), hold.get("k", [])):
            modes[self.pairs[(c, k)]] = 2
        flows = np.where(np.isfinite(flows), np.maximum(flows, 0.0), 0.0)
        override = np.where(np.isfinite(override), np.maximum(override, 0.0), 0.0)
        return {"flows": flows, "override_qty": override, "release_mode": modes}

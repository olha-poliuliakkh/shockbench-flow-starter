"""The time-expanded oracle LP (51)-(52) and its solve with SciPy's HiGHS (design §6.1, §6.3; Q1, Q58, Q73).

The LP is the simulator's equations with the clip, the default container release, the energy priority rule and the
allocation (18) replaced by the feasible sets they project onto, in the simulator's order (§6.1), built from the same
instance and the same ``WeeklyMarks`` (V3). No cap row at any chokepoint (Q58 M4); queued cargo of commodities without
override is lane-indexed (52) (containers; nucfuel would be too), tanker cargo K^ov is aggregated per (c, k) since
overrides make its release free (Q34, Q68), and is released on the out-edges that carry an override slot for (c, k).
Terminal credit and salvage as in (23) (Q93): a flow column still in transit at T credits the nu of its edge's head,
a queue column at T its chokepoint's nu, and an initial shipment still out at T its head's nu, as a constant. Rows are
stored with names so the replay test (53) can report the worst one.
The builder, the solve and the cost helpers run under the fixed NumPy error state ``marks.FP_ERRORS``, never the
caller's (DET-P3-2).
Costs are formed as the simulator forms them (§12, oracle cents), so J_LP^¢(z^pi) = J^pi¢ for every trajectory, not
only within (53): flow terms per (edge, commodity) on x_ek = ``cost.edge_flows`` of the week's lane columns; queue
holding per (chokepoint, commodity) on I_ck = ``cost.queue_totals`` of its lane columns; the WIP credit of a start
still out at T as nu times its net WIP, the start times each (1 - sigma_q) of its hits in the simulator's order
(``scrap_factors``), and initial WIP entries of one (fab, out week) or (OSAT, commodity, out week) added first; the
in-transit credit per (edge, commodity, week) as nu_head x_ek and the queue credit as nu_ck I_ck, on the same lane sums.

Variable keys (tuples, week t first after the tag):
  ("x", t, e, k, lane)       executed flow; lane set for dispatches into a chokepoint and for releases of
                             lane-indexed cargo out of one, None otherwise (tanker releases aggregate over lanes)
  ("I", t, slot)             end-of-week stock of a non-chokepoint slot
  ("Q", t, c, k, lane)       end-of-week queue at chokepoint node c (lane None for K^ov cargo)
  ("O", t, slot)             disposal at non-chokepoint, non-supply slots
  ("lift", t, slot)          supply lift at supply slots, 0 <= lift <= varsigma-bar^t
  ("p", t, f)                lots started (fab ordinal)
  ("xi", t, o, k)            OSAT starts (osat ordinal, packaged k)
  ("G", t, g, k)             segment output (grid ordinal, fuel k or None)
  ("E", t, f)                energy to fab f (fabs with a grid)
  ("y", t, g), ("ysh", t, g) served and shed base load
  ("D", t, d), ("U", t, d), ("B", t, d)   served, lost, backlog per demand ordinal (U only at lost-sales sinks,
                             B only at backlog sinks)

Row names (week t second), in the simulator's order of §3.2 within each week. Equalities: ("queue", t, c, k, lane)
(52), ("balance", t, slot) (8), ("baseload", t, g) (17), ("demand", t, d) or ("backlog", t, d) (20). Rows <=:
("edge_cap", t, e) (4) where an edge has more than one variable (otherwise the cap is the variable's bound),
("dispatch", t, slot) (5), ("throughput", t, c, pool) (9), ("fleet", t, pool) (7) over the terms of E^dup
(``Instance.dup_items``: every flow on an edge-level duplicate or turn-back, a lane term's flows on its entry edge),
("fab_energy", t, f) (12), ("ration", t, g) (15), ("energy", t, g) (17), ("osat", t, o) (19) for OSATs with several
products (one product: a bound), its right-hand side thr_i R_osat_i(t) under an OSAT restoration (13) (§4.4), and
("serve", t, d) (20) at backlog sinks.

Planning rules (``build_lp(..., planning_rules=True)``; design §12 M5 row "Planning rules", owner queue M5-O19 (b)).
The window LPs of ``mpc_det`` and ``mpc_scen`` add the simulator's rules that (51) leaves free (§6.1 "Remaining
relaxation"): a proportional split as rows, a priority as a price on the column the rule empties (the prices are in
``meta["priority"]``, which ``LPModel.objective`` adds; they never enter (23)'s components, ``lp_costs``, ``lp_cents``
or a replay). The oracle and every other caller keep the switch off, and the model is then byte-identical to the one
without it, so J^oracle stays a bound (V5) and the two share one row table (V3). With the switch on:
  ("lam", t, g)              the grid's load factor in [0, 1] (18)
  ("short", t, g, k)         a fuel segment's shortfall below its share of the load, charged v_k (18)
  ("rho", t, g)              the ratio of its feasible lots a grid's fabs start (12), (18)
  rows ("load", t, g, None): G_g0 = lam zeta_g0 G-bar; ("load_hi", t, g, k): G_gk <= lam zeta_gk G-bar; ("load_lo",
  t, g, k): lam zeta_gk G-bar - short_gk <= G_gk (every week); ("lot_start", t, f): p_f = rho_g p-hat_f, or
  p_f = p-hat_f at a fab without a grid or energy, p-hat_f = min(alpha-bar R cap0, wafers on hand after steps 5-6), the
  first week only; ("osat_start", t, o, k): xi_ok at (19)'s rule on the raw stock, the first week only
  prices: ysh_g in the first week at base_first grids with a fab that draws energy (base load first), V / min e_f over
  the grid's fabs, V the largest pi_d (times the horizon at a backlog sink) or nu of the instance; a fab's wafer stock
  and disposal, v_wafer; an OSAT's raw-chip stock and disposal, v_k; a lane-indexed queue in the weeks its pool's
  throughput is open, v_k (10).
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from sbfplan.dynamics.clip import dup_terms, fleet_caps
from sbfplan.dynamics.cost import edge_flows, queue_totals, transit_pieces
from sbfplan.dynamics.production import package
from sbfplan.dynamics.sim import initial_stock
from sbfplan.dynamics.state import COST_COMPONENTS, CostComponents, cents
from sbfplan.instance.schema import POOLS, Instance
from sbfplan.marks import WeeklyMarks, fixed_fp_errors, osat_throughput
from sbfplan.oracle._template import Columns, Rows, all_with_week, with_week


ORACLE_METHOD = "highs-ipm"  # linprog method of the oracle solve (§6.3); the LP baselines solve with highspy (Q95)
ORACLE_FALLBACK = "highs-ds"  # re-solves, cold, an ORACLE_METHOD solve of status RESOLVE_STATUS (§6.3 fallback)
RESOLVE_STATUS = 4  # linprog's "numerical difficulties", the one status the fallback re-solves (§6.3)


@dataclass(frozen=True)
class LPModel:
    """A built LP: columns with bounds, weekly cost and salvage coefficients, and named equality and <= rows.

    Keys and row names are stored as one week's templates; ``keys``, ``index``, ``eq_names`` and ``ub_names`` decode
    every week on first use, ``key``, ``eq_name`` and ``ub_name`` one index.
    """

    columns: tuple[tuple, ...]  # column keys of one week, without the week
    eq_rows: tuple[tuple, ...]  # equality row names of one week, without the week
    ub_rows: tuple[tuple, ...]  # <= row names of one week, without the week
    lb: np.ndarray
    ub: np.ndarray  # np.inf where unbounded
    cost: np.ndarray  # USD per unit, charged in week ``week[j]``
    salvage: np.ndarray  # terminal credit per unit (S_T of (23))
    week: np.ndarray  # int, the week whose C_t the column's cost enters
    A_eq: csr_matrix
    b_eq: np.ndarray
    A_ub: csr_matrix
    b_ub: np.ndarray
    T: int
    instance_hash: str
    omega_hash: str
    instance_digest: str = ""  # content digest of the instance the rows were built from (V3)
    marks_digest: str = ""  # content digest of the marks the rows were built from (V3)
    meta: dict = field(default_factory=dict)

    def objective(self) -> np.ndarray:
        """J of (23) as a linear objective, cost minus salvage, plus the planning rules' prices when built with them."""
        priority = self.meta.get("priority")
        if priority is None:
            return self.cost - self.salvage
        return self.cost - self.salvage + priority

    @cached_property
    def keys(self) -> tuple[tuple, ...]:
        """Column keys with the week after the tag, week-major."""
        return all_with_week(self.columns, self.T)

    @cached_property
    def index(self) -> dict[tuple, int]:
        return {key: i for i, key in enumerate(self.keys)}

    @cached_property
    def eq_names(self) -> tuple[tuple, ...]:
        return all_with_week(self.eq_rows, self.T)

    @cached_property
    def ub_names(self) -> tuple[tuple, ...]:
        return all_with_week(self.ub_rows, self.T)

    def key(self, i: int) -> tuple:
        return with_week(self.columns, i)

    def eq_name(self, r: int) -> tuple:
        return with_week(self.eq_rows, r)

    def ub_name(self, r: int) -> tuple:
        return with_week(self.ub_rows, r)


@dataclass(frozen=True)
class LinprogCall:
    """One ``linprog`` call of an oracle solve: its method, status, message and wall seconds."""

    method: str
    status: int
    message: str
    seconds: float


@dataclass(frozen=True)
class OracleResult:
    """A solve of (51): J^oracle in integer cents by (24), weekly then minus the rounded credit, and in USD by (23).

    ``J_usd`` is J of (23) in USD formed as the simulator forms it (``lp_usd``), the value the oracle bound of (53)
    compares (``replay.oracle_bound_holds``); ``J_cents`` is the scored value. Both and ``x`` are None when the solver
    returns no point (infeasible, unbounded, error); a non-optimal status is never zero-filled (§6.3).

    ``method`` is the method the caller asked for; ``solver`` the method whose result this is (status, message, point
    and J): ``ORACLE_FALLBACK`` when §6.3's fallback re-solved a status-4 ``ORACLE_METHOD`` solve, else ``method``.
    ``attempts`` are the linprog calls in order and ``seconds`` their sum. A result built by hand is ``method``'s one
    call (both filled in).
    """

    J_cents: int | None
    status: int  # scipy.optimize.linprog status; 0 = optimal
    message: str
    x: np.ndarray | None
    seconds: float
    method: str
    J_usd: float | None = None  # J of (23) in USD, for the oracle bound of (53)
    solver: str = ""  # the linprog method whose result this is ("" on a hand-built result: ``method``)
    attempts: tuple[LinprogCall, ...] = ()  # the linprog calls in order (() on a hand-built result: one)

    def __post_init__(self) -> None:
        if not self.solver:
            object.__setattr__(self, "solver", self.method)
        if not self.attempts:
            object.__setattr__(self, "attempts", (LinprogCall(self.method, self.status, self.message, self.seconds),))


def lp_flow_key(inst: Instance, e: int, k: int, lane: int | None) -> tuple[int, int, int | None]:
    """The LP flow (e, k, lane) that carries the simulator's x^t_ekl (§6.1).

    Tanker cargo K^ov released out of a chokepoint aggregates over lanes (lane None), since overrides make its release
    free and its queue is kept per (c, k) (Q34, Q68); every other flow keeps its lane.
    """
    if inst.commodities[k].override and inst.edges[e].tail in inst.chokepoint_ordinal:
        return (e, k, None)
    return (e, k, lane)


@fixed_fp_errors
def build_lp(inst: Instance, marks: WeeklyMarks, *, planning_rules: bool = False) -> LPModel:
    """Build (51)-(52) for one episode from the instance and its weekly marks.

    ``planning_rules`` adds the simulator's rules that (51) leaves free, for the MPCs' window LPs only (module
    docstring, "Planning rules"); the oracle keeps it off (V5).

    Raises:
        ValueError: if the marks belong to another instance or horizon (V3), or the instance lacks a stock slot the
            rows need (a flow's tail or head, a fab's input or product, an OSAT's raw or packaged chip, a grid fuel, a
            demand), or declares an initial queue lot whose next edge is not e_l(c) of a lane that permits its
            commodity (its default release would have no flow variable), or a duplicate-lane term of (7) on an edge
            whose tanker releases the LP aggregates over lanes (the term could not tell its lane apart); with
            ``planning_rules``, if a fab's wafers or an OSAT's raw chips can arrive in the week of their dispatch
            (tau_e = 0) or leave the node, so that the first week's stock after steps 5-6 is not known at the start.

    """
    inst = inst.at_digest(marks.instance_digest)  # the marks' rung: its warm start block as I^0 (§2.3; M5-O37 (b))
    if marks.instance_hash != inst.hash or marks.T != inst.T or marks.instance_digest != inst.content_digest:
        raise ValueError("marks were computed for another instance or horizon (V3)")
    return _Builder(inst, marks, planning_rules=planning_rules).build()


@fixed_fp_errors
def lp_costs(model: LPModel, z: np.ndarray) -> tuple[list[CostComponents], float]:
    """The components of each week's C_t of (23) and the credit S_T for a column vector, formed as the simulator does.

    Each component is the fsum of its terms, the flow terms per (edge, commodity) on x_ek = ``edge_flows`` of the
    week's lane columns and the queue terms per (chokepoint, commodity) on I_ck = ``queue_totals`` of its lane columns
    (§12, cost terms (23)); S_T is the fsum of the salvage terms (in transit per (edge, commodity, week) on x_ek and the
    queue per (chokepoint, commodity) on I_ck, Q93; a scrapped start's WIP credit formed as the simulator forms it) and
    the initial-state constants. For a trajectory's z^pi they equal the records' ``costs`` and the
    trajectory's ``salvage`` bit for bit.
    """
    weekly, _, salvage, _ = _terms(model, np.asarray(z, dtype=float))
    return [CostComponents(**{p: math.fsum(terms) for p, terms in w.items()}) for w in weekly], math.fsum(salvage)


@fixed_fp_errors
def lp_cents(model: LPModel, z: np.ndarray) -> int:
    """J^¢ of a column vector by (24), summed exactly as the simulator sums its cost.

    Each week's C_t is the fsum of the per-component fsums of (23) (``CostComponents.total``), converted once to
    cents; minus cents(fsum of the salvage terms).
    """
    weekly, salvage = lp_costs(model, z)
    return sum(cents(w.total()) for w in weekly) - cents(salvage)


@fixed_fp_errors
def lp_usd(model: LPModel, z: np.ndarray) -> tuple[float, float]:
    """J of (23) in USD for a column vector, and the sum of the absolute values of its terms (the scale of (53))."""
    weekly, weekly_abs, salvage, salvage_abs = _terms(model, np.asarray(z, dtype=float))
    J = math.fsum(math.fsum(math.fsum(c) for c in w.values()) for w in weekly) - math.fsum(salvage)
    return J, math.fsum(weekly_abs) + math.fsum(salvage_abs)


@fixed_fp_errors
def solve_oracle(model: LPModel, method: str = ORACLE_METHOD, *, fallback: bool = True) -> OracleResult:
    """Solve with ``scipy.optimize.linprog``, cold (``highs-ipm`` for the oracle; §6.3 as amended for Q95).

    The LP baselines never come here: they solve with highspy through ``policies.lp_common`` (warm starts, the
    lexicographic solve (70)); no sealed value comes from a warm solve.

    J^oracle in cents (24) and in USD (23) from the solution; a non-optimal status is returned as is, never zero-filled.
    §6.3's fallback (the owner, 2026-09-29): an ``ORACLE_METHOD`` solve of status ``RESOLVE_STATUS`` is re-solved once
    by ``ORACLE_FALLBACK``, cold, on the same LP, and highs-ds's result is returned (its status, point and J, ``solver``
    highs-ds, both calls in ``attempts`` and ``seconds``); only a failure of both excludes the episode. Any other
    status, and any status of another ``method``, is returned as it is; ``fallback=False`` is the raw one call.
    """
    first = _linprog_solve(model, method)
    if not fallback or method != ORACLE_METHOD or first.status != RESOLVE_STATUS:
        return first
    second = _linprog_solve(model, ORACLE_FALLBACK)
    return replace(
        second, method=method, seconds=first.seconds + second.seconds, attempts=first.attempts + second.attempts
    )


def _linprog_solve(model: LPModel, method: str) -> OracleResult:
    """One cold ``linprog`` call of (51) by ``method``: its status, message, point, J and wall seconds."""
    kw = {}
    if model.A_ub.shape[0]:
        kw.update(A_ub=model.A_ub, b_ub=model.b_ub)
    if model.A_eq.shape[0]:
        kw.update(A_eq=model.A_eq, b_eq=model.b_eq)
    bounds = np.column_stack([model.lb, model.ub])
    start = time.perf_counter()
    res = linprog(model.objective(), bounds=bounds, method=method, **kw)
    seconds = time.perf_counter() - start
    x = None if res.x is None else np.asarray(res.x, dtype=float)
    point = x is not None and bool(np.all(np.isfinite(x)))
    return OracleResult(
        J_cents=lp_cents(model, x) if point else None,
        status=int(res.status),
        message=str(res.message),
        x=x,
        seconds=seconds,
        method=method,
        J_usd=lp_usd(model, x)[0] if point else None,
    )  # solver and attempts: this one call


_LANE_SUM = {"x": edge_flows, "Q": queue_totals}  # the simulator's formation of x_ek and I_ck from lane pieces (23)


def _lane_sum_vector(model: LPModel, z: np.ndarray) -> np.ndarray:
    """``z`` with the lane columns of each group summed onto its first column (0 elsewhere), for the cost terms.

    A group is an (edge, commodity) with several flow columns in a week template (two lanes dispatched on one edge,
    lane-indexed releases of several lanes on one e_l(c)), summed into x_ek by ``edge_flows``, or a (chokepoint,
    commodity) with several lane-indexed queue columns, summed into I_ck by ``queue_totals``; a single column already
    is the sum. The cost coefficients of (23) are per group and week, the same on every lane column (checked by the
    builder), so a times the sum is the simulator's term.
    """
    groups = model.meta["lane_groups"]
    if not groups:
        return z
    nc = model.meta["nc"]
    zx = z.copy()
    rest = np.array([j for _g, cols in groups for _lane, j in cols[1:]], dtype=int)
    by_tag = [(lane_sum, [(g[1:], cols) for g, cols in groups if g[0] == tag]) for tag, lane_sum in _LANE_SUM.items()]
    for off in range(0, model.T * nc, nc):
        for lane_sum, members in by_tag:
            if members:
                sums = lane_sum({(a, b, lane): float(z[off + j]) for (a, b), cols in members for lane, j in cols})
                for ab, cols in members:
                    zx[off + cols[0][1]] = sums[ab]
        zx[off + rest] = 0.0
    return zx


def _net_credit(nu: float, gross: float, factors: tuple[float, ...]) -> float:
    """The credit nu times the net WIP of one start: the gross quantity times each factor (1 - sigma_q) in turn."""
    net = gross
    for f in factors:
        net = net * f
    return nu * net


def scrap_factors(marks: WeeklyMarks, fab: int, start_week: int, w_scr: int) -> tuple[float, ...]:
    """The factors (1 - sigma_q) the simulator applies to the WIP of fab ordinal ``fab`` started in ``start_week``.

    The hits on the fab with onset week in (start_week, start_week + w_scr] (14), in the order the simulator books
    them: by onset week, and within a week in the order of ``marks.fab_hits``. Every hit has its onset week in 1..T,
    and w_scr <= tau (loader), so each one finds the lot still in WIP. The product 1 - sigma^scr_f of ``scrap_share``
    runs over the same hits in ``fab_hits`` order; the credit (23) takes these factors one by one, as the simulator
    nets its ledger.
    """
    hits = [h for h in marks.fab_hits if h.fab == fab and start_week < h.onset_week <= start_week + w_scr]
    return tuple(1.0 - h.severity for h in sorted(hits, key=lambda h: h.onset_week))


def _terms(model: LPModel, z: np.ndarray) -> tuple[list[dict[str, list[float]]], list[float], list[float], list[float]]:
    """Per week, the explicit cost terms of each component of (23); the salvage terms; and their absolute values.

    Flow terms are per (edge, commodity) on x_ek and queue terms per (chokepoint, commodity) on I_ck
    (``_lane_sum_vector``); so are the salvage terms of flow and queue columns, nu_head x_ek for goods in transit at T
    and nu_ck I_ck for the queue at T (Q93; the coefficient is the same on every lane column of a group), as the
    simulator's S_T forms them; stock and starts stay per column: a start with scrap and still out at T is credited
    ``_net_credit`` of its factors (``meta["wip_credit"]``), not its objective coefficient nu (1 - sigma^scr) times p;
    plus the initial-state constants.
    """
    parts = model.meta["cost_parts"]
    zx = _lane_sum_vector(model, z)
    weekly: list[dict[str, list[float]]] = [{} for _ in range(model.T)]
    absolute: list[float] = []
    for p, a in parts.items():
        prods = a * zx
        keep = prods != 0.0
        prods, weeks = prods[keep], model.week[keep]
        order = np.argsort(weeks, kind="stable")
        prods, weeks = prods[order], weeks[order]
        cut = np.searchsorted(weeks, np.arange(1, model.T + 2))
        for i in range(model.T):
            weekly[i][p] = prods[cut[i] : cut[i + 1]].tolist()
        absolute.extend(np.abs(prods).tolist())
    sal = model.salvage * zx
    credit = model.meta["wip_credit"]
    if credit:
        sal[[i for i, _nu, _factors in credit]] = 0.0
    salvage = sal[sal != 0.0].tolist() + [_net_credit(nu, float(z[i]), f) for i, nu, f in credit]
    salvage += list(model.meta["salvage_const"])
    return weekly, absolute, salvage, [abs(s) for s in salvage]


class _Builder:
    """Assembles (51)-(52) from week templates (``_template``): one week's columns and rows, replicated over t."""

    def __init__(self, inst: Instance, marks: WeeklyMarks, planning_rules: bool = False) -> None:
        self.inst, self.marks, self.T = inst, marks, inst.T
        self.rules = planning_rules
        self.priority: dict[int, float | np.ndarray] = {}  # column template -> the planning rules' price (per week)
        self.t = np.arange(1, inst.T + 1)
        self.cols, self.eq, self.ub = Columns(inst.T), Rows(inst.T), Rows(inst.T)
        self.chk = set(inst.chokepoints)
        self.salvage_const: list[float] = []  # credit of initial shipments and WIP still out at T (23)
        self.supply = set(inst.supply_nodes)
        self.wip_credit: list[tuple[int, int, float, tuple[float, ...]]] = []  # (week, template column, nu, factors)
        self.qrow: dict[tuple[int, int, int | None], int] = {}  # queue -> row template, set by ``build``
        # queues: lane-indexed for commodities without override (52, §12), aggregated for K^ov
        self.queues: list[tuple[int, int, int | None]] = []
        for s in inst.stock_slots:
            if s.node not in self.chk:
                continue
            if inst.commodities[s.k].override:
                self.queues.append((s.node, s.k, None))
            else:
                self.queues.extend(
                    (s.node, s.k, li)
                    for li, ln in enumerate(inst.lanes)
                    if s.node in ln.chokepoints and s.k in inst.lane_K[li]
                )

    # ----- helpers ---------------------------------------------------------------------------------------------------
    def slot(self, node: int, k: int, what: str) -> int:
        try:
            return self.inst.slot_index[(node, k)]
        except KeyError:
            raise ValueError(
                f"{what}: node {self.inst.nodes[node].id} has no stock of {self.inst.commodities[k].id}"
            ) from None

    def nu(self, node: int, k: int, what: str) -> float:
        """nu_ik of (23): the slot's salvage, a chokepoint's included (Q93), 0 at supply nodes (Q79)."""
        return 0.0 if node in self.supply else self.inst.stock_slots[self.slot(node, k, what)].salvage

    def queue_of(self, c: int, k: int, lane: int | None, what: str) -> tuple[int, int, int | None]:
        q = (c, k, None if self.inst.commodities[k].override else lane)
        if q not in self.qrow:
            raise ValueError(f"{what}: no queue for ({self.inst.nodes[c].id}, {self.inst.commodities[k].id}, {lane})")
        return q

    def flows(self) -> list[tuple[int, int, int | None]]:
        """Flow variables (e, k, lane): the action slots on edges out of non-chokepoint nodes, and chokepoint releases.

        A flow exists exactly where the simulator can move goods (§12, oracle variables and rows). A release of K^ov
        cargo (lane None) exists on each out-edge e of c with an override slot (c, k, e, .): the continuation of every
        lane through c that permits k, where the default release (10) also goes, and a turn-back into a non-chokepoint
        on no such lane; cargo never enters a chokepoint laneless. A release of lane-indexed cargo exists only on
        e_l(c) of each lane l through c (52).
        """
        inst = self.inst
        slots = defaultdict(list)
        for a in inst.action_slots:
            slots[a[0]].append(a)
        lanes_out = defaultdict(list)  # (c, k, e_l(c)) -> the lanes of the lane-indexed queues of k at c, queue order
        for c, k, lane in self.queues:
            if lane is not None:
                lanes_out[(c, k, inst.lane_through[(lane, c)][1])].append(lane)
        overridable = {(c, k, e) for c, k, e, _lane in inst.override_slots}  # where tanker cargo can leave c
        out = []
        for e, ed in enumerate(inst.edges):
            if ed.coupling:
                continue
            if ed.tail not in self.chk:
                out.extend(slots[e])
                continue
            for k in ed.K:
                if inst.commodities[k].override:
                    if (ed.tail, k, e) in overridable and (ed.tail, k) in inst.slot_index:
                        out.append(lp_flow_key(inst, e, k, None))  # one flow over every lane
                else:
                    out.extend((e, k, lane) for lane in lanes_out.get((ed.tail, k, e), ()))
        return out

    def fleet_weights(self, flows: list[tuple[int, int, int | None]]) -> dict[tuple[int, int, int | None], float]:
        """Delta tau of each flow in the fleet row (7) of its commodity's pool, from the terms of E^dup (§3.3).

        An edge-level term (e, None, Delta tau) weighs every flow on e, any lane and commodity; a duplicate-lane term
        (e, l, Delta tau) weighs only the flows of lane l on its entry edge e, which are dispatches into a chokepoint
        and so lane-indexed (``Instance.dup_items``). A flow matched by several terms carries their sum; flows matched
        by none are left out.
        """
        terms = dup_terms(self.inst)  # the simulator's term map of (7), edge -> ((lane or None, Delta tau), ...)
        out = {}
        for e, k, lane in flows:
            if lane is None and any(lt is not None for lt, _ in terms.get(e, ())):
                # a K^ov release out of a chokepoint aggregates its lanes (Q34, Q68): a lane term cannot pick its lane
                raise ValueError(f"duplicate-lane term of (7) on {self.inst.edges[e].id}, whose releases are laneless")
            w = [d for lt, d in terms.get(e, ()) if lt is None or lt == lane]
            if w:
                out[(e, k, lane)] = float(sum(w))
        return out

    # ----- assembly --------------------------------------------------------------------------------------------------
    def build(self) -> LPModel:
        inst, marks, T, t = self.inst, self.marks, self.T, self.t
        cols, eq, ub = self.cols, self.eq, self.ub
        S, K, E = inst.stock_slots, inst.commodities, inst.edges
        flows = self.flows()
        nvar = defaultdict(int)
        for e, _, _ in flows:
            nvar[e] += 1
        plain = [s for s, st in enumerate(S) if st.node not in self.chk]
        supply_nodes = set(inst.supply_nodes)
        i0 = initial_stock(inst)  # I^0; a declared chokepoint stock must equal its queue lots (§2.3)

        # rows in the simulator's order (§3.2): forward, clip, balance and production, serve --------------------------
        self.qrow = {q: eq.row(("queue", *q)) for q in self.queues}  # (52)
        cap = {e: ub.row(("edge_cap", e), marks.u[:, e]) for e in sorted(nvar) if nvar[e] > 1}  # (4)
        outs = {self.slot(E[e].tail, k, f"flow on {E[e].id}") for e, k, _ in flows if E[e].tail not in self.chk}
        disp = {s: ub.row(("dispatch", s)) for s in plain if s in outs}  # (5)
        # (9): throughput per chokepoint and pool; no cap row on the queue (Q58 M4)
        pool = inst.commodity_pool
        released = sorted({(E[e].tail, pool[k]) for e, k, _ in flows if E[e].tail in self.chk})
        thr = {
            (c, b): ub.row(("throughput", c, POOLS[b]), marks.kappa[:, inst.chokepoint_ordinal[c], b])
            for c, b in released
        }
        fcap = fleet_caps(inst)
        dtau = self.fleet_weights(flows)
        fleet = {b: ub.row(("fleet", POOLS[b]), fcap[b]) for b in sorted({pool[k] for _, k, _ in dtau})}  # (7)
        bal = {s: eq.row(("balance", s)) for s in plain}  # (8)
        fab_en = {
            fo: ub.row(("fab_energy", fo)) for fo, f in enumerate(inst.fabs) if inst.nodes[f].fab.grid is not None
        }  # (12)
        ration = {
            go: ub.row(("ration", go)) for go, g in enumerate(inst.grids) if inst.nodes[g].grid.rationed is not None
        }  # (15)
        energy = {go: ub.row(("energy", go)) for go in range(len(inst.grids))}  # (17) as <= (§12)
        base = {go: eq.row(("baseload", go), marks.y_bar[:, go]) for go in range(len(inst.grids))}  # (17)
        # (19) under (13): thr_i R_osat_i(t), the simulator's product (§4.4; V3), as a row with several products
        thr_R = osat_throughput(inst, marks.R_osat)  # (T, O), the simulator's products, one home (SIMP-M2R2-10)
        osat_cap = {oo: np.ascontiguousarray(thr_R[:, oo]) for oo in range(len(inst.osats))}
        osat = {
            oo: ub.row(("osat", oo), osat_cap[oo])
            for oo, o in enumerate(inst.osats)
            if len(inst.nodes[o].osat.packages) > 1
        }
        dem, serve = {}, {}
        for do, d in enumerate(inst.demands):  # (20)
            dem[do] = eq.row(("backlog" if d.backlog else "demand", do), marks.demand[:, do])
            if d.backlog:
                serve[do] = ub.row(("serve", do), marks.demand[:, do])

        # executed flows x ------------------------------------------------------------------------------------------
        for e, k, lane in flows:
            ed = E[e]
            banned = marks.prohibited[:, e, k]  # Z_t (3); k not in K_e has no variable at all
            bound = np.where(banned, 0.0, marks.u[:, e] if nvar[e] == 1 else np.inf)  # (4) as a bound
            nu_head = self.nu(ed.head, k, f"arrival on {ed.id}")  # a chokepoint head at its slot's nu (Q93)
            j = cols.add(
                ("x", e, k, lane),
                ub=bound,
                salvage=np.where(t + ed.tau > T, nu_head, 0.0),  # P^T of (23): in transit at T, at the head's nu (Q93)
                freight=marks.c[:, e],
                war_risk=marks.c_wr[:, e, k],  # (11)
                tariff=marks.tariff[:, e, k] * K[k].v,  # tau^tar v_k (23)
            )
            if ed.tail in self.chk:
                eq.add(self.qrow[self.queue_of(ed.tail, k, lane, f"release on {ed.id}")], j, 1.0)
                ub.add(thr[(ed.tail, pool[k])], j, 1.0)
            else:
                s = self.slot(ed.tail, k, f"flow on {ed.id}")
                eq.add(bal[s], j, 1.0)
                ub.add(disp[s], j, 1.0)
            if ed.head in self.chk:  # arrivals A^t_ckl (52), lane-labelled at the lane's entry
                if not K[k].override and inst.lane_through[(lane, ed.head)][0] != e:
                    raise ValueError(f"flow on {ed.id} with lane {lane} does not enter {inst.nodes[ed.head].id}")
                eq.add(self.qrow[self.queue_of(ed.head, k, lane, f"arrival on {ed.id}")], j, -1.0, lag=ed.tau)
            else:
                eq.add(bal[self.slot(ed.head, k, f"arrival on {ed.id}")], j, -1.0, lag=ed.tau)  # A^t (2)
            if e in cap:
                ub.add(cap[e], j, 1.0)
            if (e, k, lane) in dtau:
                ub.add(fleet[pool[k]], j, dtau[(e, k, lane)])

        # stock I, queue Q, disposal O, lift -------------------------------------------------------------------------
        jI = {}
        for s in plain:
            st = S[s]
            jI[s] = cols.add(("I", s), ub=st.storage, salvage=np.where(t == T, st.salvage, 0.0), holding=st.holding)
            eq.add(bal[s], jI[s], 1.0)
            eq.add(bal[s], jI[s], -1.0, lag=1)
            eq.add_rhs(bal[s], 1, i0[s])
            if s in disp:
                ub.add(disp[s], jI[s], -1.0, lag=1)
                ub.add_rhs(disp[s], 1, i0[s])
        for q, r in self.qrow.items():  # no cap (Q58 M4); the queue at T at the chokepoint's nu (Q93)
            c, k, lane = q
            j = cols.add(
                ("Q", *q),
                salvage=np.where(t == T, self.nu(c, k, "queue"), 0.0),
                queue_holding=marks.h_queue[:, inst.chokepoint_ordinal[c], k],
            )
            eq.add(r, j, 1.0)
            eq.add(r, j, -1.0, lag=1)
            if self.rules and lane is not None:  # (10): the default release sends what the pool's throughput allows
                self.priority[j] = np.where(marks.kappa[:, inst.chokepoint_ordinal[c], pool[k]] > 0.0, K[k].v, 0.0)
        jO = {}
        for s in plain:
            if S[s].node not in supply_nodes:
                jO[s] = cols.add(("O", s), disposal=K[S[s].k].disposal_cost)
                eq.add(bal[s], jO[s], 1.0)
        for s in plain:
            if S[s].node in supply_nodes:  # availability cap (Q79)
                eq.add(bal[s], cols.add(("lift", s), ub=marks.supply[:, s]), -1.0)

        # fabs (12)-(14) -----------------------------------------------------------------------------------------------
        stock1 = self.first_week_stock(i0) if self.rules else {}
        first = (t == 1).astype(float)  # the planning rules' first-week rows
        jrho: dict[int, int] = {}
        for fo, f in enumerate(inst.fabs):
            fa = inst.nodes[f].fab
            s_in = self.slot(f, fa.input, "fab input")
            s_out = self.slot(f, fa.product, "fab product")
            keep = 1.0 - marks.sigma_scr[:, fo]
            j = cols.add(
                ("p", fo),
                ub=marks.alpha_bar[:, fo] * marks.R[:, fo] * fa.cap0,
                salvage=np.where(t + fa.tau > T, S[s_in].salvage * keep, 0.0),  # W^T of (23), net of scrap
            )
            if self.rules:  # (12), (18): the first week's p = rho_g p-hat; kept wafers priced every week
                phat = min(float(marks.alpha_bar[0, fo] * marks.R[0, fo] * fa.cap0), stock1[s_in])
                r = eq.row(("lot_start", fo))
                eq.add(r, j, first)
                if fa.grid is not None and fa.e > 0:
                    go = inst.grid_ordinal[fa.grid]
                    if go not in jrho:
                        jrho[go] = cols.add(("rho", go), ub=1.0)
                    eq.add(r, jrho[go], -phat * first)
                else:
                    eq.add_rhs(r, 1, phat)
                self.priority[jI[s_in]] = K[fa.input].v
                if s_in in jO:
                    self.priority[jO[s_in]] = K[fa.input].v
            for week in range(max(1, T - fa.tau + 1), T + 1):  # out at T: its credit term as the simulator forms it
                factors = scrap_factors(marks, fo, week, fa.w_scr)
                if factors:
                    self.wip_credit.append((week, j, S[s_in].salvage, factors))
            eq.add(bal[s_in], j, 1.0)  # W = p
            out = np.zeros(T)
            out[fa.tau :] = -keep[: max(T - fa.tau, 0)]  # P^t = (1 - sigma_scr(t - tau)) p^{t - tau}
            eq.add(bal[s_out], j, out, lag=fa.tau)
            if fa.grid is not None:
                ub.add(fab_en[fo], j, fa.e)
                jE = cols.add(("E", fo))
                ub.add(fab_en[fo], jE, -marks.R[:, fo])  # e_f p <= R E
                ub.add(energy[inst.grid_ordinal[fa.grid]], jE, 1.0)

        # OSATs (19) ---------------------------------------------------------------------------------------------------
        raw_of = {}
        for oo, o in enumerate(inst.osats):
            oa = inst.nodes[o].osat
            xi1 = self.first_week_packaging(o, stock1, osat_cap[oo][0]) if self.rules else {}
            for kr, kp in sorted(oa.packages.items(), key=lambda item: item[1]):
                raw_of[(o, kp)] = kr
                s_raw, s_pk = self.slot(o, kr, "OSAT raw chip"), self.slot(o, kp, "OSAT packaged chip")
                j = cols.add(
                    ("xi", oo, kp),
                    ub=np.inf if oo in osat else osat_cap[oo],  # one product: thr R_osat(t) as the bound (19)
                    salvage=np.where(t + oa.tau > T, S[s_raw].salvage, 0.0),  # W^T of (23)
                )
                eq.add(bal[s_raw], j, 1.0)
                eq.add(bal[s_pk], j, -1.0, lag=oa.tau)
                if oo in osat:
                    ub.add(osat[oo], j, 1.0)
                if self.rules:  # (19): the first week's starts at the simulator's rule; kept raw chips priced
                    r = eq.row(("osat_start", oo, kp))
                    eq.add(r, j, first)
                    eq.add_rhs(r, 1, xi1[kp])
                    self.priority[jI[s_raw]] = K[kr].v
                    if s_raw in jO:
                        self.priority[jO[s_raw]] = K[kr].v

        # grids (15)-(17) ----------------------------------------------------------------------------------------------
        psi = inst.params.psi
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            Gbar = marks.G_bar[:, go]
            jlam = cols.add(("lam", go), ub=1.0) if self.rules else None
            for k in ga.fuels + ((None,) if None in ga.shares else ()):
                zeta = ga.shares[k]
                j = cols.add(("G", go, k), ub=zeta * Gbar)
                ub.add(energy[go], j, -1.0)
                if jlam is not None:  # (18): segments loaded pro rata; a fuel short of its share burns what it has
                    if k is None:
                        r = eq.row(("load", go, None))
                        eq.add(r, j, 1.0)
                        eq.add(r, jlam, -zeta * Gbar)
                    else:
                        r = ub.row(("load_hi", go, k))
                        ub.add(r, j, 1.0)
                        ub.add(r, jlam, -zeta * Gbar)
                        js = cols.add(("short", go, k))
                        self.priority[js] = K[k].v
                        r = ub.row(("load_lo", go, k))
                        ub.add(r, j, -1.0)
                        ub.add(r, js, -1.0)
                        ub.add(r, jlam, zeta * Gbar)
                if k is None:
                    continue
                s = self.slot(g, k, "grid fuel")
                eq.add(bal[s], j, 1.0)  # W = G
                if k == ga.rationed:  # psi I-bar G <= zeta G-bar^t I^{t-1}
                    ub.add(ration[go], j, psi * ga.ibar[k])
                    ub.add(ration[go], jI[s], -zeta * Gbar, lag=1)
                    ub.add_rhs(ration[go], 1, zeta * Gbar[0] * i0[s])
            jy = cols.add(("y", go))
            jsh = cols.add(("ysh", go), shed=ga.voll)
            if self.rules and ga.priority == "base_first":  # (18): no first-week plan sheds base load to power a fab
                price = self.base_first_price(go)
                if price:
                    self.priority[jsh] = price * first
            ub.add(energy[go], jy, 1.0)
            eq.add(base[go], jy, 1.0)
            eq.add(base[go], jsh, 1.0)

        # demand (20) --------------------------------------------------------------------------------------------------
        for do, d in enumerate(inst.demands):
            jD = cols.add(("D", do))
            eq.add(bal[self.slot(d.node, d.k, "demand")], jD, 1.0)
            eq.add(dem[do], jD, 1.0)
            if d.backlog:  # B^t = B^{t-1} + d - D, B^0 = 0; D <= d + B^{t-1}
                jB = cols.add(("B", do), shortage=d.pi)
                eq.add(dem[do], jB, 1.0)
                eq.add(dem[do], jB, -1.0, lag=1)
                ub.add(serve[do], jD, 1.0)
                ub.add(serve[do], jB, -1.0, lag=1)
            else:
                eq.add(dem[do], cols.add(("U", do), shortage=d.pi), 1.0)

        self.initial_constants(bal, raw_of)
        return self.finish()

    # ----- planning rules (module docstring) ----------------------------------------------------------------------
    def first_week_stock(self, i0: np.ndarray) -> dict[int, float]:
        """The stock after steps 5-6 of week 1 at every fab's wafer slot and every OSAT's raw-chip slot, by slot.

        I^0 plus the initial shipments that arrive in week 1, added one at a time in pipeline order from I^0, as the
        simulator's step 6 adds them, so the first week's p-hat and xi equal the simulator's bit for bit.

        Raises:
            ValueError: if such a slot can receive in the week of dispatch (tau_e = 0) or send on an edge, so that its
                stock after steps 5-6 of the first week is not known at the start.

        """
        inst = self.inst
        watched = {(f, inst.nodes[f].fab.input): self.slot(f, inst.nodes[f].fab.input, "fab input") for f in inst.fabs}
        for o in inst.osats:
            for kr in inst.nodes[o].osat.packages:
                watched[(o, kr)] = self.slot(o, kr, "OSAT raw chip")
        for ed in inst.edges:
            for k in ed.K:
                if ((ed.head, k) in watched and ed.tau == 0) or (ed.tail, k) in watched:
                    raise ValueError(
                        f"planning rules: edge {ed.id} moves {inst.commodities[k].id} within the week into or out of a "
                        "fab's wafers or an OSAT's raw chips, so the first week's stock is not known at the start"
                    )
        out = {s: float(i0[s]) for s in watched.values()}
        for sh in inst.initial_state.pipeline:
            s = watched.get((inst.edges[sh.edge].head, sh.k))
            if s is not None and sh.arrival_week == 1:
                out[s] += sh.qty
        return out

    def first_week_packaging(self, o: int, stock1: dict[int, float], thr: float) -> dict[int, float]:
        """(19)'s starts xi_ok of the first week by packaged k: ``production.package`` in the simulator's order."""
        pairs = sorted(self.inst.nodes[o].osat.packages.items())
        raw = [stock1[self.slot(o, kr, "OSAT raw chip")] for kr, _kp in pairs]
        return {kp: q for (_kr, kp), q in zip(pairs, package(raw, float(thr)), strict=True)}

    def base_first_price(self, go: int) -> float:
        """The first week's price on shed base load above VOLL (18): V / min e_f over the grid's energy-drawing fabs.

        V is the largest pi_d of the instance (times the horizon at a backlog sink) or salvage nu, a bound on what a
        unit of energy is worth at a fab (a lot takes e_f / R_f >= e_f of it and yields at most one chip); 0 when no
        fab of the grid draws energy.
        """
        inst = self.inst
        es = [inst.nodes[inst.fabs[fo]].fab.e for fo in inst.grid_fabs[go]]
        es = [e for e in es if e > 0]
        if not es:
            return 0.0
        V = max(
            max((d.pi * (self.T if d.backlog else 1) for d in inst.demands), default=0.0),
            max((st.salvage for st in inst.stock_slots), default=0.0),
        )
        return V / min(es)

    def initial_constants(self, bal: dict[int, int], raw_of: dict[tuple[int, int], int]) -> None:
        """Right-hand sides from the initial state (2), (12), (19), (52), and the credit of what is still out at T."""
        inst, marks, T, S, E = self.inst, self.marks, self.T, self.inst.stock_slots, self.inst.edges
        # shipments still out at T: nu_head x_ek per (edge, commodity, dispatch week), x_ek the fsum over lanes of each
        # lane's shipments in entry order, as the simulator's credit forms them from its pipeline (Q93)
        late = transit_pieces(sh for sh in inst.initial_state.pipeline if sh.arrival_week > T)
        for pieces in late.values():
            for (e, k), x in edge_flows(pieces).items():
                self.salvage_const.append(self.nu(E[e].head, k, "initial shipment") * x)
        for sh in inst.initial_state.pipeline:
            ed = E[sh.edge]
            if sh.arrival_week > T:
                continue
            if ed.head in self.chk:
                q = self.queue_of(ed.head, sh.k, sh.lane, f"initial shipment on {ed.id}")
                self.eq.add_rhs(self.qrow[q], sh.arrival_week, sh.qty)
            else:
                self.eq.add_rhs(bal[self.slot(ed.head, sh.k, "initial shipment")], sh.arrival_week, sh.qty)
        for lot in inst.initial_state.queue_lots:
            # the default release (10) sends a lot onto its next edge; the LP releases it only on e_l(c) of a lane
            # permitting k (52), where K^ov cargo has an override slot, so any other next edge has no flow variable
            through = inst.lane_through.get((lot.lane, lot.chokepoint))
            if through is None or through[1] != lot.next_edge or lot.k not in inst.lane_K[lot.lane]:
                raise ValueError(
                    f"initial queue lot of {inst.commodities[lot.k].id} at {inst.nodes[lot.chokepoint].id}: next edge "
                    f"{E[lot.next_edge].id} is not e_l(c) of its lane {inst.lanes[lot.lane].id} permitting it (52)"
                )
            q = self.queue_of(lot.chokepoint, lot.k, lot.lane, "initial queue lot")
            self.eq.add_rhs(self.qrow[q], 1, lot.qty)
        # WIP entries of one (fab, out week) or (OSAT, commodity, out week) add up first, in entry order, as the
        # simulator's initial state adds them; the gross sum is then scrapped by hits after reset (14)
        fab_wip: dict[tuple[int, int], float] = {}
        for w in inst.initial_state.fab_wip:
            fab_wip[(w.node, w.out_week)] = fab_wip.get((w.node, w.out_week), 0.0) + w.qty
        for (node, out_week), qty in fab_wip.items():
            fo, fa = inst.fab_ordinal[node], inst.nodes[node].fab
            if out_week > T:  # credit at the wafer nu, net of scrap as the simulator nets it (23)
                factors = scrap_factors(marks, fo, out_week - fa.tau, fa.w_scr)
                nu = S[self.slot(node, fa.input, "fab input")].salvage
                self.salvage_const.append(_net_credit(nu, qty, factors))
            else:  # P^w = (1 - sigma^scr_f(w - tau)) times the gross sum (12)
                keep = 1.0 - marks.scrap_share(fo, out_week - fa.tau)
                self.eq.add_rhs(bal[self.slot(node, fa.product, "fab WIP")], out_week, qty * keep)
        osat_wip: dict[tuple[int, int, int], float] = {}
        for w in inst.initial_state.osat_wip:
            osat_wip[(w.node, w.k, w.out_week)] = osat_wip.get((w.node, w.k, w.out_week), 0.0) + w.qty
        for (node, k, out_week), qty in osat_wip.items():
            if out_week > T:
                kr = raw_of[(node, k)]
                self.salvage_const.append(S[self.slot(node, kr, "OSAT raw chip")].salvage * qty)
            else:
                self.eq.add_rhs(bal[self.slot(node, k, "OSAT WIP")], out_week, qty)

    def flow_aliases(self) -> dict[tuple[int, int, int], tuple[int, int, None]]:
        """Simulator flows (e, k, lane) whose LP flow ``lp_flow_key`` drops the lane: K^ov releases on a lane.

        A release carries the lane of its override slot, or of its lot on the default release (10) onto e_l(c) of a
        lane permitting k (initial lots are checked in ``initial_constants``); either is an override slot (c, k, e, l).
        """
        inst = self.inst
        return {
            (e, k, lane): lp_flow_key(inst, e, k, lane) for _c, k, e, lane in inst.override_slots if lane is not None
        }

    def lane_groups(self, parts: dict[str, np.ndarray]) -> tuple:
        """Columns that (23) charges summed over lanes: ((tag, a, b), ((lane, j), ...)), columns in template order.

        ("x", e, k): an (edge, commodity) with several flow columns, charged on x_ek; ("Q", c, k): a (chokepoint,
        commodity) with several lane-indexed queue columns, charged on I_ck. Single columns are left out.

        Raises:
            ValueError: if a cost or salvage coefficient of (23) differs between the lane columns of one group in a
                week, since (23) charges and credits the lanes summed (§12, cost terms (23); Q93).

        """
        cols: dict[tuple, list[tuple[int | None, int]]] = defaultdict(list)
        for j, key in enumerate(self.cols.keys):
            if key[0] in ("x", "Q"):
                cols[key[:3]].append((key[3], j))
        nc = len(self.cols.keys)
        out = []
        for g, members in cols.items():
            if len(members) < 2:
                continue
            for p in ("freight", "war_risk", "tariff", "salvage") if g[0] == "x" else ("queue_holding", "salvage"):
                a = parts[p].reshape(self.T, nc)
                if any(not np.array_equal(a[:, members[0][1]], a[:, j]) for _lane, j in members[1:]):
                    where = self.inst.edges[g[1]].id if g[0] == "x" else self.inst.nodes[g[1]].id
                    raise ValueError(f"{p} of (23) differs between the lanes of {where}")
            out.append((g, tuple(members)))
        return tuple(out)

    def finish(self) -> LPModel:
        cols, T = self.cols, self.T
        nc = len(cols.keys)
        parts = {p: cols.stack(v) for p, v in cols.parts.items()}
        cost = np.zeros(T * nc)
        for p in COST_COMPONENTS:
            cost = cost + parts[p]
        A_eq, b_eq = self.eq.build(nc)
        A_ub, b_ub = self.ub.build(nc)
        priority = None
        if self.priority:  # the planning rules' prices, week-major like the columns
            table = np.zeros((T, nc))
            for j, price in self.priority.items():
                table[:, j] = price
            priority = table.ravel()
        return LPModel(
            columns=tuple(cols.keys),
            eq_rows=tuple(self.eq.names),
            ub_rows=tuple(self.ub.names),
            lb=np.zeros(T * nc),
            ub=cols.stack(cols.ub),
            cost=cost,
            salvage=cols.stack(cols.salvage),
            week=np.repeat(self.t, nc),
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            T=T,
            instance_hash=self.inst.hash,
            omega_hash=self.marks.omega_hash,
            instance_digest=self.inst.content_digest,
            marks_digest=self.marks.digest,
            meta={
                "nc": nc,
                "template": {key: j for j, key in enumerate(cols.keys)},
                "cost_parts": {p: a for p, a in parts.items() if np.any(a)},
                "lane_groups": self.lane_groups({**parts, "salvage": cols.stack(cols.salvage)}),  # x_ek, I_ck (23)
                "wip_credit": tuple(  # (week-major column, nu, factors) of each scrapped start still out at T
                    ((week - 1) * nc + j, nu, factors) for week, j, nu, factors in self.wip_credit
                ),
                "salvage_const": tuple(self.salvage_const),
                "flow_key": self.flow_aliases(),  # simulator (e, k, lane) -> LP flow, where they differ
                "slots": tuple((st.node, st.k) for st in self.inst.stock_slots),
                "chokepoints": frozenset(self.chk),
                "override": frozenset(k for k, c in enumerate(self.inst.commodities) if c.override),
                **({"planning_rules": True, "priority": priority} if self.rules else {}),
            },
        )

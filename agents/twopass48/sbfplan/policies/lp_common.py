"""The LP baselines' shared core (design §8.2, §6.3; Q17, Q67, Q95; milestone M4, stream "lp-baselines").

The only module under ``src/`` that imports highspy (1.12.0 at githash 755a8e0: the HiGHS release SciPy 1.18.1 vendors,
in highspy's own build, a library apart from SciPy's copy at 4f96ee8; design §6.3 as amended for Q95, §12 "HiGHS
builds (M4 re-gate)"), and only inside the functions that solve: importing the module, the package or an
evidence script's modules never needs it (``tests/test_m2_v12.py`` checks the evidence commands against the package's
module-level imports). ``greedy_lp``, ``mpc_det``, ``mpc_scen`` and ``hindsight_consensus`` solve through it;
the oracle, naive's reset-time flow (22) and every sealed value stay on ``scipy.optimize.linprog`` cold solves
(``oracle.lp``, ``instance.nominal``). Readings of design §12 (rows marked "M4 reading" and the stream's rows):

- **One row table** (V3, "no drift from (51)/(52)"). Every window LP except ``greedy_lp``'s is
  ``oracle.lp.build_lp(rolled_instance(inst, S_t, H_t), window_marks(...))``: zero new row code. S_t is the start
  state rebuilt from the wire observation of week t alone (``observed_start``), weeks renumbered t -> 1; the rolled
  instance is ``dataclasses.replace(inst, T=H_t, initial_state=S_t)`` (a fresh memo; its hash label is the original's,
  so it never reaches ``Env``, a replay of a real trajectory, a seal or Static; V3 compares the content digest, which
  the window marks carry). Backlog B^{t-1} is folded into week-1 demand of backlog sinks, since the builder fixes
  B^0 = 0 (exact for rows "backlog" and "serve" of (20)). WIP in the observation is net of booked scrap, so a window
  carries only fab hits with onset >= t (none under persistence; a scenario's future regional conflicts pass theirs
  through ``rolled_lp(..., fab_hits=)``, since ``build_lp`` scraps the observed initial WIP by ``marks.fab_hits``).
- **Horizon** (Q17, Q67). H_t = min(H, T - t + 1), H in {L, L+4, L+8, max(26, 2L)}; L = the largest, over the chains
  of naive's reset-time plan, of route transits plus fab and OSAT processing (22 on `tiny`; design §12 "Lead time L as
  built"). The terminal value V_H is the builder's own credit (23) at the window end (Q93), so at H_t = T - t + 1 the
  MPC is end-aware by construction; at H_t < T - t + 1 the builder's ``salvage_const`` credits what is still out at
  the window end.
- **Point forecast of ``mpc_det``** (§8.2 "Gate baselines"). The marks at the instant t - 1 persist over the window;
  pending prohibitions take effect from their effective week; h^Q and c^wr follow the persisted war-risk class by the
  marks' own rule (``marks.queue_and_transit``); sigma^scr = 0 and no fab hits; demand is (48) for h = 0..7, then
  the seasonal mean d-bar m^sea(t + h). Under blackout (``graph_now`` null) or a coverage rung (null entries) each
  entry keeps its last observed value, else the nominal instance value (``ObservedGraph``).
- **Action mapping** (``week1_action``). Requests: the week-1 LP columns of the action slots, max(x, 0), slots whose
  own (edge, k) is prohibited this week skipped: observed in ``graph_now.prohibited``, forecast under blackout (§9.3's
  dispatch rule, the first-edge rule ``slot_mask`` stated before Q111; the whole-route ``slot_mask`` of Q111 is not
  read, so the LP's choice to send into a lane whose later edge is prohibited now stands, as before). Tanker releases:
  the laneless LP release on (c, k, e) is sent on the first override slot of (c, k, e) in instance order, every week,
  qty 0 included, so the LP, not the default release, controls the tanker release;
  an override slot whose (out-edge, k) is observed or forecast prohibited this week is not sent (§9.3 would drop it
  as masked, and a dropped override does not switch the default release off), and where no override slot of (c, k)
  can be sent a hold for (c, k) stands for the LP's zero release (a hold is never masked and wins over overrides,
  §3.4 step 2). The release lane is exact when every lane through (c, e) shares the rest of its path, as on `tiny`
  (E1 ends both L0 and L2); tandem lanes (M5) need the slot whose lane matches the LP's downstream column.
  Container releases, lot starts, OSAT starts and the energy split are simulator rules and never sent.
- **highspy model.** Rows [A_ub (-inf, b_ub]; A_eq [b_eq, b_eq]] in column-wise CSC; columns [lb, ub] with inf as
  kHighsInf; the model's constant J terms (``-sum(meta["salvage_const"])``) as the HighsLp offset, so HiGHS's
  objective is the full J of (23) and a relative tolerance is relative to it; options ``HIGHS_OPTIONS`` (dual simplex,
  presolve on). ``threads`` is never set: highspy's scheduler fixes it at highspy's first run in the process (the
  oracle's linprog runs SciPy's copy, whose scheduler is its own). The objective is passed unscaled (V11 on the LP
  baselines is reported, not gated: Q100 (2)).
- **Warm start** (Q95). Cold at reset; from week t - 1's basis after that, weeks shifted by one (the dropped week
  removed, the last week repeated when H is fixed), passed as an alien basis that HiGHS repairs. A warm solve's vertex
  depends on the solve history, so an LP baseline's trajectory is deterministic across processes and workers on one
  platform (V1), not across platforms or on a wire resume, and never a sealed value; a wire resume restarts cold
  (``information.runner`` "Resume"). Every rung attempt runs on a fresh ``Highs`` object, so a solve is a function of
  the model, the options and the basis passed, and an in-process resume from ``LPSession.state`` is bit for bit
  (design §12 "LP sessions as built").
- **Status ladder** (§12 "Solver status on generated omega"). Every solve runs ``LADDER``: warm dual with presolve ->
  presolve off -> ipm with crossover -> primal simplex; if every rung fails the policy plays naive's action from the
  same observation (``internal_fallback``: ``NaivePolicy(fq_quantile, end_aware=True)``), counted in
  ``StepTelemetry.fallback``; no solve is ever zero-filled (§6.3).
- **Lexicographic solve (70)** (``hindsight_consensus``). One call of ``addLinearObjective`` with priority 2 = J and
  priority 1 = w.x, ``blend_multi_objectives`` off, ``abs_tolerance`` kHighsInf and ``rel_tolerance`` 1e-9 set
  explicitly (HiGHS's slack is min(abs, rel |J*|); the defaults -1 enforce nothing): J <= J* + 1e-9 |J*|, the reading
  of (70) that stays feasible where J* < 0 (design §12 M4 row "(70) slack"). Ladder ``LEX_LADDER`` (M5 order, design
  §12 M5 row "Lexicographic ladder order"): two calls by dual (J*, then min w.x s.t. J <= J* + 1e-9 |J*|, warm from
  the kept basis) -> one call -> one call with presolve off -> two calls by ipm -> the stage-1 point (tie-break
  skipped, counted). highspy 1.12 returns no valid basis after a multi-objective run, so a one-call success leaves the
  session's chain empty, while a two-call success keeps its stage-1 basis (design §12 "Lexicographic solve as
  built"); hence two calls first.
"""

import dataclasses
import functools
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from sbfplan.instance.schema import (
    WAR_RISK_CLASSES,
    FrozenDict,
    InitialLot,
    InitialShipment,
    InitialState,
    InitialWip,
    Instance,
)
from sbfplan.marks import FabHit, WeeklyMarks, queue_and_transit
from sbfplan.oracle.lp import LPModel, build_lp
from sbfplan.policies.naive import NaivePlan, NaivePolicy


if TYPE_CHECKING:  # annotations only: the solving functions import highspy themselves (module docstring)
    highspy = None  # not on the scoring server

    from sbfplan.policies.registry import PolicyContext

# ----- horizon (Q17, Q67) --------------------------------------------------------------------------------------------
H_SWEEP = ("L", "L+4", "L+8", "max(26,2L)")  # Q67's sweep, each a separate policy name (it enters the trajectory hash)
CANONICAL_H = "L"  # the mpc_det of the §2.6 separation pair and of the M4 report (Q17's floor; design §12 M4 row)

# ----- the window's marks --------------------------------------------------------------------------------------------
# every array field of WeeklyMarks: a window's arrays dict holds each, shape (H, ...) (``persistence_arrays``)
WINDOW_FIELDS = tuple(f.name for f in dataclasses.fields(WeeklyMarks) if f.type is np.ndarray)
WINDOW_LABEL = "window"  # the omega_hash label of window marks and models: never a real omega's hash
# the instantaneous marks of a window, (field, week-average field): each a copy of its week average (``with_now``)
NOW_FIELDS = (
    ("u_now", "u"),
    ("o_now", "o"),
    ("kappa_now", "kappa"),
    ("supply_now", "supply"),
    ("G_bar_now", "G_bar"),
    ("y_bar_now", "y_bar"),
    ("R_now", "R"),
    ("alpha_now", "alpha_bar"),
)

# ----- highspy model and the status ladder (design §12 M4 rows "highspy model", "Status ladder") ---------------------
HIGHS_OPTIONS: Mapping[str, object] = FrozenDict(
    output_flag=False,
    presolve="on",
    solver="simplex",
    simplex_strategy=1,  # serial dual simplex, as linprog's highs-ds
)
LADDER = ("dual", "dual_nopresolve", "ipm", "primal")
RUNG_OPTIONS: Mapping[str, Mapping[str, object]] = FrozenDict(
    dual=FrozenDict(),  # HIGHS_OPTIONS, warm-started where a basis is known
    dual_nopresolve=FrozenDict(presolve="off"),
    ipm=FrozenDict(solver="ipm", run_crossover="on"),
    primal=FrozenDict(simplex_strategy=4),
)
OPTIMAL = "Optimal"  # HiGHS's model status string of an optimal solve; any other status is a failed rung

# ----- the lexicographic solve (70) ----------------------------------------------------------------------------------
LEX_REL = 1e-9  # the (70) slack, relative to |J*| (J <= J* + 1e-9 |J*|; design §12 M4 row "(70) slack")
LEX_ABS = math.inf  # abs_tolerance = kHighsInf, so HiGHS's slack min(abs, rel |J*|) is the relative one
LEX_PRIORITY_J = 2  # addLinearObjective priority of J (solved first)
LEX_PRIORITY_W = 1  # priority of the stream-20 weights w.x
# two calls by dual first (M5): a two-call success keeps its stage-1 basis for next week's warm start, which a one-call
# multi-objective run cannot (design §12 M5 row "Lexicographic ladder order")
LEX_LADDER = ("two_call_dual", "one_call", "one_call_nopresolve", "two_call_ipm", "stage1")

_BASIC = 1  # HighsBasisStatus.kBasic: the status of a padded link row or of the two-call rung's added row


def _highspy():
    """highspy, imported where a solve needs it (module docstring: never at module level)."""
    highspy = None  # not on the scoring server

    return highspy


@functools.cache
def _basis_statuses() -> dict[int, "highspy.HighsBasisStatus"]:
    """HighsBasisStatus code -> enum member: a lookup, about 20x cheaper per status than the pybind11 constructor."""
    return {int(m): m for m in _highspy().HighsBasisStatus.__members__.values()}


# ----- horizon -------------------------------------------------------------------------------------------------------
def _chain_extreme(inst: Instance, plan: NaivePlan, pick: Callable) -> int | None:
    """``pick`` (max or min) over the chains of naive's plan of route transits plus processing (design §12 row).

    D(j, k) is the remaining time from an arrival of k at j to a sink: 0 at a sink demanding k; tau^fab plus the best
    route of the fab's product at a fab (its wafers); tau^osat plus the best route of the packaged chip at an OSAT
    (a raw chip it packages); the best coupling edge tau_e + D(f, wafers) into a fab f at a grid (a fuel it burns); the
    best route out elsewhere; None where no chain reaches a sink.
    """
    out: dict[tuple[int, int], list] = defaultdict(list)
    for r in plan.routes:
        out[(inst.edges[r.first_edge].tail, r.k)].append(r)
    demanded = {(d.node, d.k) for d in inst.demands}
    memo: dict[tuple[int, int], int | None] = {}

    def best(values) -> int | None:
        vals = [v for v in values if v is not None]
        return pick(vals) if vals else None

    def routes_from(j: int, k: int, path: frozenset) -> int | None:
        return best((None if (d := remaining(r.dest, r.k, path)) is None else r.tau + d) for r in out.get((j, k), ()))

    def remaining(j: int, k: int, path: frozenset) -> int | None:
        if (j, k) in memo:
            return memo[(j, k)]
        if (j, k) in path:
            raise ValueError(f"naive's planned routes form a cycle through {inst.nodes[j].id}")
        path = path | {(j, k)}
        node = inst.nodes[j]
        if node.type == "sink" and (j, k) in demanded:
            val: int | None = 0
        elif node.fab is not None and k == node.fab.input:
            rest = routes_from(j, node.fab.product, path)
            val = None if rest is None else node.fab.tau + rest
        elif node.osat is not None and k in node.osat.packages:
            rest = routes_from(j, node.osat.packages[k], path)
            val = None if rest is None else node.osat.tau + rest
        elif node.grid is not None and k in node.grid.fuels:
            cands = []
            for e in inst.out_edges[j]:
                edge = inst.edges[e]
                fab = inst.nodes[edge.head].fab
                if edge.coupling and fab is not None:
                    d = remaining(edge.head, fab.input, path)
                    cands.append(None if d is None else edge.tau + d)
            val = best(cands)
        else:
            val = routes_from(j, k, path)
        memo[(j, k)] = val
        return val

    supply = set(inst.supply_nodes)
    starts = [r for r in plan.routes if inst.edges[r.first_edge].tail in supply]
    return best((None if (d := remaining(r.dest, r.k, frozenset())) is None else r.tau + d) for r in starts)


def lead_time_L(inst: Instance, plan: NaivePlan) -> int:
    """L of Q17/Q67: the largest, over the chains of naive's plan, of route transits plus fab and OSAT processing.

    A chain runs from a supply node through naive's planned routes to a sink, entering fabs through their wafers; a
    grid's fuel continues through its coupling edges (tau 0) into the fabs it powers, so LNG chains run on to the
    fab's chips and their sink (Q17); 22 on `tiny` (LNG via L1 4 + grid -> coupling E16 -> fab_us 8 + E20 4 + OSAT 2
    + L3 4), 14 on its fastest chain (design §12 M4 row "Horizon"; ``fastest_chain``).

    Raises:
        ValueError: if the plan has no chain to a sink.

    """
    L = _chain_extreme(inst, plan, max)
    if L is None:
        raise ValueError("naive's plan has no chain from a supply node to a sink (Q17's L is undefined)")
    return int(L)


def fastest_chain(inst: Instance, plan: NaivePlan) -> int:
    """The smallest chain length of ``lead_time_L``'s recursion (reported beside L; 14 on `tiny`).

    Raises:
        ValueError: if the plan has no chain to a sink.

    """
    L = _chain_extreme(inst, plan, min)
    if L is None:
        raise ValueError("naive's plan has no chain from a supply node to a sink")
    return int(L)


def horizon_length(label: str, L: int) -> int:
    """H of a sweep label of ``H_SWEEP`` for this L: L, L + 4, L + 8 or max(26, 2L).

    Raises:
        ValueError: on a label outside ``H_SWEEP`` or L < 1.

    """
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError(f"L must be an integer >= 1, got {L!r}")
    L = int(L)
    table = {"L": L, "L+4": L + 4, "L+8": L + 8, "max(26,2L)": max(26, 2 * L)}
    if label not in table:
        raise ValueError(f"horizon label must be one of {H_SWEEP} (Q67), got {label!r}")
    return table[label]


def window_length(H: int, t: int, T: int) -> int:
    """H_t = min(H, T - t + 1), the weeks the window of week t covers (Q67).

    Raises:
        ValueError: if H < 1 or t is not in 1..T.

    """
    for name, v in (("H", H), ("t", t), ("T", T)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {v!r}")
    if H < 1 or not 1 <= t <= T:
        raise ValueError(f"need H >= 1 and 1 <= t <= T, got H={H}, t={t}, T={T}")
    return int(min(H, T - t + 1))


# ----- start state and the rolled window (V3: one row table) ---------------------------------------------------------
def observed_week(obs: dict, T: int) -> int:
    """The observation's week t, refused unless an integer in 1..T."""
    t = obs.get("week")
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 1 <= t <= T:
        raise ValueError(f"the observation's week must be an integer in 1..{T}, got {t!r}")
    return int(t)


def _index(value, bound: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
        raise ValueError(f"the observation names {what} {value!r}, which the instance does not have")
    return int(value)


def _demand_ordinal(inst: Instance) -> dict[tuple[int, int], int]:
    return {(d.node, d.k): i for i, d in enumerate(inst.demands)}


def observed_start(inst: Instance, obs: dict) -> tuple[InitialState, np.ndarray]:
    """The start state of week t's window from the wire observation alone, weeks shifted by t - 1, and B^{t-1}.

    Stock from ``obs["stock"]`` (non-chokepoint slots; a chokepoint's stock is the sum of its lots, rebuilt by the
    simulator's ``initial_stock``); each pipeline entry an ``InitialShipment`` with ``arrival_week`` = arr - (t - 1)
    and ``dispatch_week`` = that minus tau_e (the builder reads only the arrival); each queue lot an ``InitialLot``
    with its weeks shifted; fab and OSAT WIP ``InitialWip`` with ``out_week`` - (t - 1) (net of booked scrap). The
    second value is B^{t-1} per demand ordinal, (D,), folded into week-1 demand by ``window_marks``. Entries keep the
    observation's order (design §12 "Rolled start as built").

    Raises:
        ValueError: if the observation names a slot, lane or node the instance does not have, or its week is not an
            integer in 1..T.

    """
    t = observed_week(obs, inst.T)
    shift = t - 1
    chk = set(inst.chokepoints)
    n_nodes, n_edges, n_lanes, n_k = len(inst.nodes), len(inst.edges), len(inst.lanes), len(inst.commodities)
    stock = []
    st = obs["stock"]
    for node, k, q in zip(st["node"], st["k"], st["qty"], strict=True):
        key = (_index(node, n_nodes, "node"), _index(k, n_k, "commodity"))
        if key not in inst.slot_index or key[0] in chk:
            raise ValueError(f"the observation's stock names slot {key}, not a non-chokepoint slot of the instance")
        stock.append((key[0], key[1], float(q)))
    pipeline = []
    pp = obs["pipeline"]
    for e, k, lane, q, arr in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], pp["arrival_week"], strict=True):
        e, k = _index(e, n_edges, "edge"), _index(k, n_k, "commodity")
        lane = None if lane is None else _index(lane, n_lanes, "lane")
        a = int(arr) - shift
        pipeline.append(InitialShipment(e, k, lane, float(q), a - inst.edges[e].tau, a))
    lots = []
    ql = obs["queue_lots"]
    for c, k, q, lane, nxt, arr, disp, entry in zip(
        ql["chokepoint"],
        ql["k"],
        ql["qty"],
        ql["lane"],
        ql["next_edge"],
        ql["arrival_week"],
        ql["dispatch_week"],
        ql["entry_edge"],
        strict=True,
    ):
        c = _index(c, n_nodes, "node")
        if c not in chk:
            raise ValueError(f"the observation's queue lot sits at node {c}, which is no chokepoint")
        lots.append(
            InitialLot(
                chokepoint=c,
                k=_index(k, n_k, "commodity"),
                qty=float(q),
                lane=_index(lane, n_lanes, "lane"),
                next_edge=_index(nxt, n_edges, "edge"),
                dispatch_week=int(disp) - shift,
                entry_edge=_index(entry, n_edges, "edge"),
                arrival_week=int(arr) - shift,
            )
        )
    fab_wip, osat_wip = [], []
    wp = obs["wip"]
    for node, k, q, out in zip(wp["node"], wp["k"], wp["qty"], wp["out_week"], strict=True):
        node, k = _index(node, n_nodes, "node"), _index(k, n_k, "commodity")
        w = InitialWip(node, k, float(q), int(out) - shift)
        if node in inst.fab_ordinal:
            fab_wip.append(w)
        elif node in inst.osat_ordinal:
            osat_wip.append(w)
        else:
            raise ValueError(f"the observation's WIP sits at node {node}, which is neither a fab nor an OSAT")
    backlog = np.zeros(len(inst.demands))
    ordinal = _demand_ordinal(inst)
    bl = obs.get("backlog") or {"node": [], "k": [], "qty": []}
    for node, k, q in zip(bl["node"], bl["k"], bl["qty"], strict=True):
        key = (_index(node, n_nodes, "node"), _index(k, n_k, "commodity"))
        if key not in ordinal:
            raise ValueError(f"the observation's backlog names {key}, which is no demand of the instance")
        backlog[ordinal[key]] += float(q)
    state = InitialState(
        stock=tuple(stock),
        pipeline=tuple(pipeline),
        fab_wip=tuple(fab_wip),
        osat_wip=tuple(osat_wip),
        queue_lots=tuple(lots),
    )
    return state, backlog


def rolled_instance(inst: Instance, start: InitialState, H: int) -> Instance:
    """``dataclasses.replace(inst, T=H, initial_state=start)``: the window's instance, for window LPs only.

    It keeps the original hash label with other content and a stale ``raw``: it never reaches ``Env``, a replay of a
    real trajectory, a seal or Static; ``build_lp``'s V3 check compares the content digest the window marks carry.

    Raises:
        ValueError: if H is not an integer >= 1.

    """
    if isinstance(H, bool) or not isinstance(H, (int, np.integer)) or H < 1:
        raise ValueError(f"a window needs H >= 1 weeks, got {H!r}")
    return dataclasses.replace(inst, T=int(H), initial_state=start)


def _field_shapes(inst: Instance) -> dict[str, tuple[int, ...]]:
    """The per-week shape of every ``WINDOW_FIELDS`` array of ``inst`` (``marks.WeeklyMarks`` axes)."""
    E, K, C = len(inst.edges), len(inst.commodities), len(inst.chokepoints)
    S, G, F, O, D = len(inst.stock_slots), len(inst.grids), len(inst.fabs), len(inst.osats), len(inst.demands)
    base = {
        "u": (E,),
        "c": (E,),
        "o": (C,),
        "kappa": (C, 2),
        "supply": (S,),
        "G_bar": (G,),
        "y_bar": (G,),
        "R": (F,),
        "alpha_bar": (F,),
        "sigma_scr": (F,),
        "R_osat": (O,),
        "demand": (D,),
        "prohibited": (E, K),
        "tariff": (E, K),
        "wr_class": (C,),
        "h_queue": (C, K),
        "c_wr": (E, K),
    }
    base |= {now: base[week] for now, week in NOW_FIELDS}
    missing = set(WINDOW_FIELDS) - set(base)
    if missing:  # a WeeklyMarks array field this module does not know (a schema change): refuse, never guess
        raise ValueError(f"window_marks does not know the WeeklyMarks fields {sorted(missing)}")
    return base


_DTYPES = {"prohibited": np.bool_, "wr_class": np.int8}


def window_marks(
    inst_r: Instance, arrays: Mapping[str, np.ndarray], backlog: np.ndarray, *, fab_hits=()
) -> WeeklyMarks:
    """``WeeklyMarks`` of the window from forecast or scenario ``arrays`` (each field of ``WINDOW_FIELDS``, (H, ...)).

    ``instance_hash`` is ``inst_r.hash`` (the label), ``instance_digest`` ``inst_r.content_digest``, ``omega_hash``
    ``WINDOW_LABEL``; ``backlog`` (D,) is added to week-1 demand of backlog sinks (the builder fixes B^0 = 0);
    ``fab_hits`` are hits with onset in the window, weeks already renumbered.

    Raises:
        ValueError: if an array is missing or not (inst_r.T, ...) in the field's shape, ``backlog`` is not (D,) or is
            nonzero at a lost-sales sink, or a fab hit names no fab or has its onset week outside 1..inst_r.T.

    """
    H = inst_r.T
    shapes = _field_shapes(inst_r)
    kw: dict[str, np.ndarray] = {}
    for name in WINDOW_FIELDS:
        if name not in arrays:
            raise ValueError(f"window arrays lack {name!r}")
        a = np.asarray(arrays[name])
        if a.shape != (H, *shapes[name]):
            raise ValueError(f"window array {name!r} has shape {a.shape}, expected {(H, *shapes[name])}")
        kw[name] = a.astype(_DTYPES.get(name, np.float64), copy=False)
    b = np.asarray(backlog, dtype=np.float64)
    if b.shape != (len(inst_r.demands),):
        raise ValueError(f"backlog must be (D,) = ({len(inst_r.demands)},), got {b.shape}")
    if b.any():
        demand = kw["demand"].copy()
        for do, d in enumerate(inst_r.demands):
            if b[do] == 0.0:
                continue
            if not d.backlog:
                raise ValueError(f"a backlog of {b[do]!r} at demand {do}, a lost-sales sink (20)")
            demand[0, do] = demand[0, do] + b[do]  # B^{t-1} folded into week 1: exact for (20)'s rows
        kw["demand"] = demand
    hits = tuple(fab_hits)
    for h in hits:
        if not isinstance(h, FabHit) or not 0 <= h.fab < len(inst_r.fabs) or not 1 <= h.onset_week <= H:
            raise ValueError(f"a window fab hit must name a fab and an onset week in 1..{H}, got {h!r}")
    return WeeklyMarks(
        T=H,
        instance_hash=inst_r.hash,
        omega_hash=WINDOW_LABEL,
        instance_digest=inst_r.content_digest,
        fab_hits=hits,
        w_scr=tuple(inst_r.nodes[f].fab.w_scr for f in inst_r.fabs),
        **kw,
    )


def rolled_window(inst: Instance, obs: dict, H: int) -> tuple[Instance, np.ndarray]:
    """The rolled instance of week ``obs["week"]``'s window and B^{t-1}: what every window model of the week shares.

    Raises:
        ValueError: as ``observed_start``, ``check_window`` and ``rolled_instance``.

    """
    start, backlog = observed_start(inst, obs)
    check_window(inst, int(obs["week"]), H)
    return rolled_instance(inst, start, H), backlog


def rolled_lp(
    inst: Instance,
    obs: dict,
    arrays: Mapping[str, np.ndarray],
    H: int,
    *,
    fab_hits=(),
    window: tuple[Instance, np.ndarray] | None = None,
    planning_rules: bool = False,
) -> LPModel:
    """``build_lp(rolled_instance(inst, S_t, H), window_marks(...))`` of week ``obs["week"]`` (no new row code, V3).

    ``fab_hits`` (a scenario's hits with onset in the window, weeks renumbered; () under persistence) are forwarded to
    ``window_marks``, so ``build_lp`` scraps the observed initial WIP by them (``scrap_share``). At t = 1 with the true
    marks (and their fab hits) the model equals ``build_lp(inst, marks)`` array for array (the no-drift test).
    ``window`` is ``rolled_window(inst, obs, H)`` when the caller builds one model per scenario of the same week (the
    rolled instance and its content digest are then computed once); None builds it here. ``planning_rules`` is
    ``build_lp``'s switch of the simulator's rules in the window (``mpc_det`` and ``mpc_scen``; design §12 M5 row
    "Planning rules"), off by default, so ``hindsight_consensus`` and the no-drift test build (51) as the oracle does.

    Raises:
        ValueError: as ``rolled_window``, ``window_marks`` and ``build_lp``.

    """
    inst_r, backlog = rolled_window(inst, obs, H) if window is None else window
    marks = window_marks(inst_r, arrays, backlog, fab_hits=fab_hits)
    return build_lp(inst_r, marks, planning_rules=planning_rules)


# ----- mpc_det's point forecast (§8.2 "Gate baselines") --------------------------------------------------------------
def _take(target: np.ndarray, values, index=None) -> None:
    """Copy the non-null entries of ``values`` into ``target`` (at ``index[i]`` when given): null keeps the last."""
    if values is None:
        return
    for i, v in enumerate(values):
        if v is not None:
            target[i if index is None else index[i]] = v


@dataclass
class ObservedGraph:
    """The last observed value of every ``graph_now`` entry, per episode (mutable; created at reset, never pickled).

    ``update(inst, obs)`` takes every non-null entry of week t's ``graph_now`` (``u``, ``c``, ``open``, ``kappa``,
    ``supply``, ``fab``, ``grid``, ``osat``, ``prohibited``, ``tariff``, ``war_risk``) and keeps the previous value
    where the entry, or the whole ``graph_now`` (blackout), is null; before any observation an entry holds its nominal
    value of the instance (u^0, c^0, open 1, k_c mu_cb, supply caps, R = alpha-bar = 1, G-bar^0, y-bar^0, no prohibition
    beyond Z_0, no tariff, class none). ``pending`` holds the latest shown ``pending_prohibitions`` as (edge, k,
    effective week) triples (design §12 "Observed graph and persistence as built").
    """

    values: dict
    pending: list

    @classmethod
    def nominal(cls, inst: Instance) -> "ObservedGraph":
        """The memory before any observation: every entry at its nominal instance value."""
        E, K = len(inst.edges), len(inst.commodities)
        Z = np.zeros((E, K), dtype=bool)
        for e, k in inst.prohibitions_at_reset:
            Z[e, k] = True
        values = {
            "u": np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges], dtype=np.float64),
            "c": np.array([e.c0 for e in inst.edges], dtype=np.float64),
            "open": np.ones(len(inst.chokepoints)),
            "kappa": np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=np.float64).reshape(
                len(inst.chokepoints), 2
            ),
            "supply": np.array([s.supply for s in inst.stock_slots], dtype=np.float64),
            "fab_R": np.ones(len(inst.fabs)),
            "fab_alpha": np.ones(len(inst.fabs)),
            "osat_R": np.ones(len(inst.osats)),
            "grid_G": np.array([inst.nodes[g].grid.deliverable for g in inst.grids], dtype=np.float64),
            "grid_y": np.array([inst.nodes[g].grid.base_load for g in inst.grids], dtype=np.float64),
            "prohibited": Z,
            "tariff": np.zeros((E, K)),
            "war_risk": np.zeros(len(inst.chokepoints), dtype=np.int8),
        }
        return cls(values=values, pending=[])

    def update(self, inst: Instance, obs: dict) -> None:
        """Take the non-null entries of week t's ``graph_now`` and its ``pending_prohibitions`` (docstring)."""
        pend = obs.get("pending_prohibitions")
        if pend is not None:
            self.pending = [
                (int(e), int(k), int(w))
                for e, k, w in zip(pend["edge"], pend["k"], pend["effective_week"], strict=True)
            ]
        g = obs.get("graph_now")
        if g is None:
            return
        v = self.values
        _take(v["u"], g.get("u"))  # a coupling edge's u is always null: it keeps its nominal inf
        _take(v["c"], g.get("c"))
        _take(v["open"], g.get("open"))
        kappa = g.get("kappa")
        if kappa is not None:
            for bi, pool in enumerate(("tb", "ct")):
                _take(v["kappa"][:, bi], kappa.get(pool))  # a column view of the (C, 2) memory: written in place
        sup = g.get("supply")
        if sup is not None:
            slots = [inst.slot_index[(int(n), int(k))] for n, k in zip(sup["node"], sup["k"], strict=True)]
            _take(v["supply"], sup["avail"], slots)
        fab = g.get("fab")
        if fab is not None:
            idx = [inst.fab_ordinal[int(n)] for n in fab["node"]]
            _take(v["fab_R"], fab.get("R"), idx)
            _take(v["fab_alpha"], fab.get("alpha_bar"), idx)
        osat = g.get("osat")
        if osat is not None:
            _take(v["osat_R"], osat.get("R"), [inst.osat_ordinal[int(n)] for n in osat["node"]])
        grid = g.get("grid")
        if grid is not None:
            idx = [inst.grid_ordinal[int(n)] for n in grid["node"]]
            _take(v["grid_G"], grid.get("G_bar"), idx)
            _take(v["grid_y"], grid.get("y_bar"), idx)
        pr = g.get("prohibited")
        if pr is not None:  # the whole Z_t of week t (Z_0 included), never masked by a coverage rung
            Z = np.zeros_like(v["prohibited"])
            for e, k in zip(pr["edge"], pr["k"], strict=True):
                Z[int(e), int(k)] = True
            v["prohibited"] = Z
        tar = g.get("tariff")
        if tar is not None:
            rates = np.zeros_like(v["tariff"])
            for e, k, r in zip(tar["edge"], tar["k"], tar["rate"], strict=True):
                rates[int(e), int(k)] = r
            v["tariff"] = rates
        wr = g.get("war_risk")
        if wr is not None:
            _take(v["war_risk"], [None if x is None else WAR_RISK_CLASSES.index(x) for x in wr])

    def state(self) -> dict:
        """Plain data for an in-process resume (lists, and the pending triples)."""
        return {"values": {k: np.asarray(a).tolist() for k, a in self.values.items()}, "pending": list(self.pending)}

    @classmethod
    def from_state(cls, inst: Instance, state: dict) -> "ObservedGraph":
        """The memory ``state()`` wrote, with the nominal memory's dtypes."""
        mem = cls.nominal(inst)
        for k, nominal in mem.values.items():
            mem.values[k] = np.array(state["values"][k], dtype=nominal.dtype)
        mem.pending = [tuple(p) for p in state["pending"]]
        return mem


def point_demand(inst: Instance, obs: dict, H: int) -> np.ndarray:
    """(H, D) demand of weeks t..t + H - 1: the shown forecast (48) at h <= 7, else d-bar m^sea(t + h)."""
    t = int(obs["week"])
    out = np.array([[d.dbar * d.m_sea(t + h) for d in inst.demands] for h in range(H)], dtype=np.float64)
    fc = obs.get("demand_forecast")
    if fc is not None:
        ordinal = _demand_ordinal(inst)
        for node, k, h, q in zip(fc["node"], fc["k"], fc["h"], fc["qty"], strict=True):
            if q is not None and 0 <= int(h) < H and math.isfinite(q):
                out[int(h), ordinal[(int(node), int(k))]] = q
    return out


def pending_mask(memory: ObservedGraph, t: int, H: int, shape: tuple[int, int]) -> np.ndarray:
    """(H, E, K) the pending prohibitions of ``memory`` switched on from their effective week (weeks t..t + H - 1)."""
    out = np.zeros((H, *shape), dtype=bool)
    for e, k, w in memory.pending:
        start = max(w, t) - t
        if start < H:
            out[start:, e, k] = True
    return out


def window_queue_and_transit(inst: Instance, wr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h^Q and c^wr of the window weeks from their war-risk classes (H, C), by ``marks.queue_and_transit``."""
    wr = np.asarray(wr, dtype=np.int8)
    return queue_and_transit(dataclasses.replace(inst, T=int(wr.shape[0])), wr)


def check_window(inst: Instance, t: int, H: int) -> None:
    """Refuse a window of week t unless 1 <= H <= T - t + 1 (H an integer)."""
    if isinstance(H, bool) or not isinstance(H, (int, np.integer)) or H < 1 or t + H - 1 > inst.T:
        raise ValueError(f"a window of week {t} needs 1 <= H <= T - t + 1 = {inst.T - t + 1}, got {H!r}")


def with_now(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``arrays`` with each instantaneous field of ``NOW_FIELDS`` set to a copy of its week-average field."""
    for now, week in NOW_FIELDS:
        arrays[now] = arrays[week].copy()
    return arrays


def read_only(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``arrays`` with every array's WRITEABLE flag cleared (window arrays are handed out read-only)."""
    for a in arrays.values():
        a.flags.writeable = False
    return arrays


def persistence_arrays(inst: Instance, obs: dict, memory: ObservedGraph, H: int) -> dict[str, np.ndarray]:
    """``mpc_det``'s point forecast of weeks t..t + H - 1: every field of ``WINDOW_FIELDS``, shape (H, ...), read-only.

    Persistence of the instant t - 1 from ``memory`` (week-average and instantaneous fields alike; ``memory`` already
    holds week t's observation), pending prohibitions switched on from their effective week, h^Q and c^wr by
    ``marks.queue_and_transit`` of the persisted class, sigma^scr 0, demand (48) from ``obs["demand_forecast"]`` for
    h = 0..7 (null past T) and d-bar m^sea(t + h) beyond or when the forecast is null.

    Raises:
        ValueError: if H < 1 or t + H - 1 > T.

    """
    t = observed_week(obs, inst.T)
    check_window(inst, t, H)
    v = memory.values

    def rep(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, (H, *a.shape)).copy()

    wr = rep(v["war_risk"])
    hq, cwr = window_queue_and_transit(inst, wr)
    out = {
        "u": rep(v["u"]),
        "c": rep(v["c"]),
        "o": rep(v["open"]),
        "kappa": rep(v["kappa"]),
        "supply": rep(v["supply"]),
        "G_bar": rep(v["grid_G"]),
        "y_bar": rep(v["grid_y"]),
        "R": rep(v["fab_R"]),
        "alpha_bar": rep(v["fab_alpha"]),
        "sigma_scr": np.zeros((H, len(inst.fabs))),
        "R_osat": rep(v["osat_R"]),
        "demand": point_demand(inst, obs, H),
        "prohibited": rep(v["prohibited"]) | pending_mask(memory, t, H, v["prohibited"].shape),
        "tariff": rep(v["tariff"]),
        "wr_class": wr,
        "h_queue": hq,
        "c_wr": cwr,
    }
    return read_only(with_now(out))


def prohibited_now(memory: ObservedGraph, t: int) -> np.ndarray:
    """(E, K) week t's Z_t as observed (or remembered under blackout) with the pending pairs due at t."""
    return memory.values["prohibited"] | pending_mask(memory, t, 1, memory.values["prohibited"].shape)[0]


# ----- the action mapping --------------------------------------------------------------------------------------------
def action_columns(inst: Instance, model: LPModel) -> np.ndarray:
    """Indices of the week-1 action coordinates of a window model (int, sorted): the domain averaged by (70).

    The columns ("x", 1, e, k, lane) with a non-chokepoint tail (exactly the action slots) and the laneless tanker
    releases ("x", 1, e, k, None) out of chokepoints on out-edges with override slots: 20 + 3 on `tiny`. Edge tails,
    action slots and override slots come from ``inst`` (the model's columns are keyed ("x", e, k, lane) and its meta
    names only the chokepoint nodes).

    Raises:
        ValueError: if an action slot has no column in the model.

    """
    tmpl = model.meta["template"]
    cols = set()
    for e, k, lane in inst.action_slots:
        j = tmpl.get(("x", e, k, lane))
        if j is None:
            raise ValueError(f"action slot ({inst.edges[e].id}, {k}, {lane}) has no column in the window model")
        cols.add(j)
    for _c, k, e, _lane in inst.override_slots:
        j = tmpl.get(("x", e, k, None))
        if j is not None:
            cols.add(j)
    return np.array(sorted(cols), dtype=np.intp)


def week1_action(inst: Instance, model: LPModel, x: np.ndarray, obs: dict, prohibited_now: np.ndarray) -> dict:
    """The wire action of week t from a window solution ``x`` (module docstring, "Action mapping").

    ``prohibited_now`` (E, K) bool is week t's Z_t as observed (``graph_now.prohibited``) or, under blackout, as
    forecast (the window's week-1 ``prohibited``). Flows are max(x, 0) on the action slots, a slot whose own (edge, k)
    is prohibited skipped (``graph_now.prohibited`` when shown, else ``prohibited_now``; not ``slot_mask``, Q111);
    overrides carry the laneless releases on the first sendable override slot of each (c, k, e), qty 0 included; a hold
    for (c, k) where no override slot of it is sendable. No negative qty, no NaN (design §12 "Action mapping as
    built").

    Raises:
        ValueError: if ``x`` is not the model's column vector or holds a non-finite value.

    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != model.lb.shape or not bool(np.all(np.isfinite(x))):
        raise ValueError(f"x must be the model's finite column vector of shape {model.lb.shape}")
    tmpl = model.meta["template"]
    g = obs.get("graph_now")
    shown = None if g is None else set(zip(g["prohibited"]["edge"], g["prohibited"]["k"], strict=True))
    flows_s, flows_q = [], []
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if (e, k) in shown if shown is not None else bool(prohibited_now[e, k]):
            continue
        flows_s.append(s)
        flows_q.append(max(float(x[tmpl[("x", e, k, lane)]]), 0.0))
    first: dict[tuple[int, int, int], int] = {}
    for o, (c, k, e, _lane) in enumerate(inst.override_slots):
        first.setdefault((c, k, e), o)
    sendable: dict[tuple[int, int], bool] = {}
    ov = []
    for (c, k, e), o in first.items():
        j = tmpl.get(("x", e, k, None))
        if j is None:
            continue
        ok = not bool(prohibited_now[e, k])
        sendable[(c, k)] = sendable.get((c, k), False) or ok
        if ok:
            ov.append((o, max(float(x[j]), 0.0)))
    ov.sort()
    holds = [ck for ck, ok in sendable.items() if not ok]
    return {
        "week": int(obs["week"]),
        "flows": {"slot": flows_s, "qty": flows_q},
        "overrides": {"slot": [o for o, _ in ov], "qty": [q for _, q in ov]} if ov else None,
        "hold": {"chokepoint": [c for c, _ in holds], "k": [k for _, k in holds]} if holds else None,
    }


# ----- the highspy model ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowShape:
    """The row and column template of a window model: per-week columns and rows, weeks, blocks and link rows.

    A plain window has ``blocks`` 1 and ``n_link`` 0; ``saa_lp``'s model is ``blocks`` = S copies of one block's
    template side by side, columns and rows block-major (each block week-major, rows [ub; eq]), followed by ``n_link``
    nonanticipativity rows (the basis shift's input).
    """

    nc: int  # columns per week (LPModel.meta["nc"])
    n_ub: int  # <= rows per week
    n_eq: int  # equality rows per week
    H: int  # weeks
    blocks: int = 1  # window blocks side by side (S for saa_lp)
    n_link: int = 0  # nonanticipativity rows after the blocks' rows

    @classmethod
    def of(cls, model: LPModel) -> "WindowShape":
        """The shape of a plain window model (its week templates are the same in every week): one block, no links."""
        return cls(nc=int(model.meta["nc"]), n_ub=len(model.ub_rows), n_eq=len(model.eq_rows), H=int(model.T))

    @property
    def n_col(self) -> int:
        return self.blocks * self.H * self.nc

    @property
    def n_row(self) -> int:
        return self.blocks * self.H * (self.n_ub + self.n_eq) + self.n_link


def model_offset(model: LPModel) -> float:
    """The constant J terms of a model: ``-math.fsum(meta["salvage_const"])`` (credit of what is out at the end)."""
    return -math.fsum(model.meta["salvage_const"])


def highs_lp(
    cost: np.ndarray, lb: np.ndarray, ub: np.ndarray, A, row_lb: np.ndarray, row_ub: np.ndarray, offset: float = 0.0
) -> "highspy.HighsLp":
    """A ``HighsLp`` from plain arrays (``A`` a scipy sparse matrix, stored CSC), for ``greedy_lp``'s one-week model.

    Raises:
        ValueError: if the array lengths do not match ``A``.

    """
    hs = _highspy()
    A = sp.csc_matrix(A)
    n_row, n_col = A.shape
    cost, lb, ub = (np.asarray(a, dtype=np.float64) for a in (cost, lb, ub))
    row_lb, row_ub = np.asarray(row_lb, dtype=np.float64), np.asarray(row_ub, dtype=np.float64)
    if cost.shape != (n_col,) or lb.shape != (n_col,) or ub.shape != (n_col,):
        raise ValueError(f"cost and bounds must hold one entry per column ({n_col})")
    if row_lb.shape != (n_row,) or row_ub.shape != (n_row,):
        raise ValueError(f"row bounds must hold one entry per row ({n_row})")
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_ = n_col
    lp.num_row_ = n_row
    lp.col_cost_ = cost
    lp.col_lower_ = np.where(np.isinf(lb), -inf, lb)
    lp.col_upper_ = np.where(np.isinf(ub), inf, ub)
    lp.row_lower_ = np.where(np.isinf(row_lb), -inf, row_lb)
    lp.row_upper_ = np.where(np.isinf(row_ub), inf, row_ub)
    lp.offset_ = float(offset)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n_col
    lp.a_matrix_.num_row_ = n_row
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    return lp


def _stacked(model: LPModel) -> tuple:
    """The model's rows [A_ub; A_eq] with their bounds, and its column bounds (``to_highs_lp``, ``saa_lp``)."""
    A = sp.vstack([model.A_ub, model.A_eq], format="csr")
    row_lb = np.concatenate([np.full(model.A_ub.shape[0], -np.inf), model.b_eq])
    row_ub = np.concatenate([model.b_ub, model.b_eq])
    return A, row_lb, row_ub


def to_highs_lp(model: LPModel, objective: np.ndarray | None = None, offset: float | None = None) -> "highspy.HighsLp":
    """The model as a ``HighsLp``: CSC rows [A_ub; A_eq], bounds with inf as kHighsInf, J's constants as offset.

    ``objective`` defaults to ``model.objective()`` (cost - salvage), ``offset`` to ``model_offset(model)``.
    """
    A, row_lb, row_ub = _stacked(model)
    c = model.objective() if objective is None else objective
    return highs_lp(c, model.lb, model.ub, A, row_lb, row_ub, model_offset(model) if offset is None else offset)


def saa_lp(blocks: Sequence[LPModel], shared: np.ndarray) -> tuple["highspy.HighsLp", WindowShape]:
    """``mpc_scen``'s two-stage SAA: S window blocks side by side, objective the mean of their J (offsets included).

    Nonanticipativity rows x^(s)_a - x^(0)_a = 0 for every block s >= 1 and every week-1 action column a of
    ``shared`` (``action_columns`` of block 0; every block has the same template). The shape returned is the whole
    model's: one block's template with ``blocks`` = S and ``n_link`` = (S - 1) len(shared), so ``LPSession.solve`` can
    shift the basis of every block and keep the link rows.

    Raises:
        ValueError: if the blocks' templates differ or fewer than 2 blocks are given.

    """
    if len(blocks) < 2:
        raise ValueError(f"saa_lp needs at least 2 blocks, got {len(blocks)}")
    b0 = blocks[0]
    for b in blocks[1:]:
        if (b.columns, b.eq_rows, b.ub_rows, b.T) != (b0.columns, b0.eq_rows, b0.ub_rows, b0.T):
            raise ValueError("saa_lp's blocks must share one window template (columns, rows, weeks)")
    S, n = len(blocks), len(b0.lb)
    shared = np.asarray(shared, dtype=np.intp)
    if shared.size and (shared.min() < 0 or shared.max() >= n):
        raise ValueError("shared columns must index one block's columns")
    stacks = [_stacked(b) for b in blocks]
    n_link = (S - 1) * shared.size
    rows = np.repeat(np.arange(n_link), 2)
    cols = np.concatenate([[s * n + a, a] for s in range(1, S) for a in shared]) if n_link else np.zeros(0, int)
    vals = np.tile([1.0, -1.0], n_link)
    link = sp.csr_matrix((vals, (rows, cols)), shape=(n_link, S * n))
    A = sp.vstack([sp.block_diag([st[0] for st in stacks], format="csr"), link], format="csr")
    row_lb = np.concatenate([st[1] for st in stacks] + [np.zeros(n_link)])
    row_ub = np.concatenate([st[2] for st in stacks] + [np.zeros(n_link)])
    cost = np.concatenate([b.objective() / S for b in blocks])  # the mean of the blocks' J
    offset = math.fsum(model_offset(b) for b in blocks) / S
    lb = np.concatenate([b.lb for b in blocks])
    ub = np.concatenate([b.ub for b in blocks])
    shape = WindowShape.of(b0)
    shape = dataclasses.replace(shape, blocks=S, n_link=n_link)
    return highs_lp(cost, lb, ub, A, row_lb, row_ub, offset), shape


# ----- warm start and the status ladder ------------------------------------------------------------------------------
@dataclass(frozen=True)
class SolveResult:
    """One solve through the ladder. ``x`` is None whenever ``ok`` is False: never zero-filled (§6.3)."""

    x: np.ndarray | None
    ok: bool
    status: str  # the model status of the last rung tried
    trail: tuple[tuple[str, str], ...]  # (rung, model status) per attempt, in ladder order
    iterations: int  # simplex plus IPM iterations over the attempts
    seconds: float
    objective: float | None  # J of the point, offset included; None when not ok


def _check_shapes(prev: WindowShape, new: WindowShape) -> None:
    if (prev.nc, prev.n_ub, prev.n_eq, prev.blocks) != (new.nc, new.n_ub, new.n_eq, new.blocks):
        raise ValueError(f"basis shift between different templates: {prev} -> {new}")


def _shift_weeks(a: np.ndarray, H_prev: int, per: int, H_new: int, shift: int) -> np.ndarray:
    """A week-major status block moved by ``shift`` weeks: the first weeks dropped, the last repeated to H_new weeks."""
    if per == 0:
        return np.zeros(0, dtype=a.dtype)
    m = a.reshape(H_prev, per)
    kept = m[shift:] if shift < H_prev else m[-1:]
    if len(kept) < H_new:
        kept = np.concatenate([kept, np.repeat(kept[-1:], H_new - len(kept), axis=0)])
    return kept[:H_new].ravel()


def _shift_arrays(
    col: np.ndarray, row: np.ndarray, prev: WindowShape, new: WindowShape, shift: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """``shift_basis`` on status arrays (int8, HighsBasisStatus codes)."""
    _check_shapes(prev, new)
    if shift < 0:
        raise ValueError(f"a basis shift is >= 0 weeks, got {shift}")
    if len(col) != prev.n_col or len(row) != prev.n_row:
        raise ValueError(f"the basis ({len(col)} columns, {len(row)} rows) is not of the shape {prev}")
    ncb, nub, neq = prev.H * prev.nc, prev.H * prev.n_ub, prev.H * prev.n_eq
    cols, rows = [], []
    for b in range(prev.blocks):
        cols.append(_shift_weeks(col[b * ncb : (b + 1) * ncb], prev.H, prev.nc, new.H, shift))
        r0 = b * (nub + neq)
        rows.append(_shift_weeks(row[r0 : r0 + nub], prev.H, prev.n_ub, new.H, shift))
        rows.append(_shift_weeks(row[r0 + nub : r0 + nub + neq], prev.H, prev.n_eq, new.H, shift))
    link = row[prev.blocks * (nub + neq) :]
    if len(link) >= new.n_link:
        link = link[: new.n_link]
    else:
        link = np.concatenate([link, np.full(new.n_link - len(link), _BASIC, dtype=row.dtype)])
    rows.append(link)
    return np.concatenate(cols).astype(np.int8), np.concatenate(rows).astype(np.int8)


def _to_basis(col: np.ndarray, row: np.ndarray, alien: bool) -> "highspy.HighsBasis":
    hs = _highspy()
    members = _basis_statuses()

    def statuses(codes: np.ndarray) -> list:
        return [members[v] if v in members else hs.HighsBasisStatus(v) for v in np.asarray(codes).tolist()]

    b = hs.HighsBasis()
    b.col_status = statuses(col)
    b.row_status = statuses(row)
    b.valid = True
    b.alien = alien
    return b


def _from_basis(basis: "highspy.HighsBasis") -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([int(s) for s in basis.col_status], dtype=np.int8),
        np.array([int(s) for s in basis.row_status], dtype=np.int8),
    )


def shift_basis(
    basis: "highspy.HighsBasis", prev: WindowShape, new: WindowShape, shift: int = 1
) -> "highspy.HighsBasis":
    """Week t - 1's basis moved to week t's window: the first ``shift`` weeks dropped, the last week repeated to fill.

    Column and row statuses are week-major (``oracle._template``), rows ordered [ub; eq] as in ``to_highs_lp``; with
    ``blocks`` > 1 each block is shifted alike and the ``n_link`` link rows keep their statuses (their count follows
    ``new``); the result is marked alien (``HighsBasis.alien = True``) for HiGHS to repair.

    Raises:
        ValueError: if the shapes' templates (nc, n_ub, n_eq) or block counts differ, or the basis is not of ``prev``.

    """
    col, row = _from_basis(basis)
    col, row = _shift_arrays(col, row, prev, new, shift)
    return _to_basis(col, row, alien=True)


@dataclass
class _Attempt:
    status: str
    iterations: int
    x: np.ndarray | None
    objective: float | None
    basis: tuple[np.ndarray, np.ndarray] | None


def _iterations(info) -> int:
    counts = (info.simplex_iteration_count, info.ipm_iteration_count, info.crossover_iteration_count)
    return sum(max(0, int(c)) for c in counts)


def _set_options(h, options: Mapping[str, object]) -> None:
    hs = _highspy()
    for k, v in options.items():
        if h.setOptionValue(k, v) != hs.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS refused the option {k}={v!r}")


def internal_fallback(context: "PolicyContext", static: dict, obs: dict, policy_seed: int) -> NaivePolicy:
    """The LP baselines' fallback when every rung fails: the end-aware naive with the context's F_Q, reset (§6.3)."""
    fb = NaivePolicy(fq_quantile=context.fq_quantile, end_aware=True)
    fb.reset(static, obs, policy_seed)
    return fb


class LPSession:
    """One basis chain of one policy episode: the options, the last basis and its shape (created in reset).

    ``hindsight_consensus`` keeps one session per scenario; ``mpc_det`` and ``mpc_scen`` one each; ``greedy_lp`` one,
    whose model keeps its structure every week, so its basis is reused as is (not alien). ``state()`` and
    ``load_state()`` carry the chain for an in-process resume (a wire resume restarts cold). Every rung attempt runs on
    a fresh ``Highs`` object (design §12 "LP sessions as built").
    """

    def __init__(self, options: Mapping[str, object] = HIGHS_OPTIONS) -> None:
        self.options = dict(options)
        self._col: np.ndarray | None = None
        self._row: np.ndarray | None = None
        self._shape: WindowShape | None = None

    # ----- the chain ----------------------------------------------------------------------------------------------
    def _warm(self, lp, shape: WindowShape | None):
        """The basis passed to the first rung: the kept one shifted (``shape`` given) or as is; None when unusable."""
        if self._col is None or self._row is None:
            return None
        if shape is None:
            if len(self._col) != lp.num_col_ or len(self._row) != lp.num_row_:
                return None
            return _to_basis(self._col, self._row, alien=False)
        if self._shape is None:
            return None
        try:
            col, row = _shift_arrays(self._col, self._row, self._shape, shape)
        except ValueError:
            return None
        if len(col) != lp.num_col_ or len(row) != lp.num_row_:
            return None
        return _to_basis(col, row, alien=True)

    def _keep(self, basis, shape: WindowShape | None) -> None:
        if basis is None:
            self._col = self._row = self._shape = None
        else:
            self._col, self._row = basis
            self._shape = shape

    # ----- one attempt --------------------------------------------------------------------------------------------
    def _attempt(
        self, rung: str, lp, rung_options: Mapping[str, object], basis, objectives=(), row=None, cost=None
    ) -> _Attempt:
        """One run on a fresh ``Highs``: options, model, optional extra row and costs, objectives, basis, run.

        ``rung`` names the attempt ("dual", ..., "one_call", "two_call_dual:stage2", ...): the seam the ladder tests
        force a failure through; the run does not read it.
        """
        hs = _highspy()
        h = hs.Highs()
        _set_options(h, {**self.options, **rung_options})
        h.passModel(lp)
        if row is not None:  # the two-call rung's row J <= J* + rel |J*| (constants moved to the right-hand side)
            idx, val, upper = row
            h.addRow(-hs.kHighsInf, upper, len(idx), idx, val)
        if cost is not None:
            h.changeColsCost(len(cost), np.arange(len(cost), dtype=np.int32), cost)
            h.changeObjectiveOffset(0.0)
        for obj in objectives:
            h.addLinearObjective(obj)
        if basis is not None:
            h.setBasis(basis)  # a basis HiGHS refuses leaves the run cold
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        it = _iterations(info)
        if status != OPTIMAL:
            return _Attempt(status, it, None, None, None)
        x = np.array(h.getSolution().col_value, dtype=np.float64)
        if x.shape != (lp.num_col_,) or not bool(np.all(np.isfinite(x))):
            return _Attempt(f"{status} (no finite point)", it, None, None, None)
        b = h.getBasis()
        return _Attempt(status, it, x, float(info.objective_function_value), _from_basis(b) if b.valid else None)

    # ----- solves -------------------------------------------------------------------------------------------------
    def solve(self, lp: "highspy.HighsLp", shape: WindowShape | None = None, *, warm: bool = True) -> SolveResult:
        """Solve through ``LADDER``; warm from the last basis (shifted by ``shift_basis`` when ``shape`` changes).

        ``shape`` None means the model keeps the last structure (``greedy_lp``): the last basis is passed as is.
        """
        start = time.perf_counter()
        basis = self._warm(lp, shape) if warm else None
        trail, iters = [], 0
        status = "Not run"
        for i, rung in enumerate(LADDER):
            a = self._attempt(rung, lp, RUNG_OPTIONS[rung], basis if i == 0 else None)
            trail.append((rung, a.status))
            iters += a.iterations
            status = a.status
            if a.x is not None:
                self._keep(a.basis, shape)
                return SolveResult(a.x, True, status, tuple(trail), iters, time.perf_counter() - start, a.objective)
        self._keep(None, None)
        return SolveResult(None, False, status, tuple(trail), iters, time.perf_counter() - start, None)

    def lexicographic(
        self, lp: "highspy.HighsLp", w: np.ndarray, shape: WindowShape | None = None, *, rel: float = LEX_REL
    ) -> SolveResult:
        """The (70) solve through ``LEX_LADDER``: min J, then min w.x s.t. J <= J* + rel |J*| (module docstring).

        A result from the "stage1" rung is the primary optimum with the tie-break skipped (its trail says so).

        Raises:
            ValueError: if ``w`` is not one weight per column, finite and >= 0, or ``rel`` is not in (0, 1).

        """
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (lp.num_col_,) or not bool(np.all(np.isfinite(w))) or bool(np.any(w < 0)):
            raise ValueError(f"w must hold one finite weight >= 0 per column ({lp.num_col_})")
        if isinstance(rel, bool) or not 0.0 < rel < 1.0:
            raise ValueError(f"rel must lie in (0, 1), got {rel!r}")
        hs = _highspy()
        start = time.perf_counter()
        basis = self._warm(lp, shape)
        c = np.asarray(lp.col_cost_, dtype=np.float64)
        offset = float(lp.offset_)
        trail, iters = [], 0
        status = "Not run"

        def done(x: np.ndarray, st: str, kept) -> SolveResult:
            self._keep(kept, shape)
            J = math.fsum((c * x).tolist()) + offset
            return SolveResult(x, True, st, tuple(trail), iters, time.perf_counter() - start, J)

        for rung in LEX_LADDER:
            if rung in ("one_call", "one_call_nopresolve"):
                o_j = hs.HighsLinearObjective()
                o_j.weight, o_j.offset, o_j.coefficients = 1.0, offset, c.tolist()
                o_j.abs_tolerance, o_j.rel_tolerance, o_j.priority = LEX_ABS, rel, LEX_PRIORITY_J
                o_w = hs.HighsLinearObjective()
                o_w.weight, o_w.offset, o_w.coefficients, o_w.priority = 1.0, 0.0, w.tolist(), LEX_PRIORITY_W
                opts = {"blend_multi_objectives": False}
                if rung == "one_call_nopresolve":
                    opts["presolve"] = "off"
                a = self._attempt(rung, lp, opts, basis if rung == "one_call" else None, objectives=(o_j, o_w))
                trail.append((rung, a.status))
                iters += a.iterations
                status = a.status
                if a.x is not None:
                    return done(a.x, status, None)  # highspy returns no valid basis after a multi-objective run
                continue
            if rung in ("two_call_dual", "two_call_ipm"):
                opts = RUNG_OPTIONS["dual" if rung == "two_call_dual" else "ipm"]
                s1 = self._attempt(f"{rung}:stage1", lp, opts, basis if rung == "two_call_dual" else None)
                iters += s1.iterations
                if s1.x is None:
                    trail.append((rung, s1.status))
                    status = s1.status
                    continue
                J_star = s1.objective
                nz = np.flatnonzero(c)
                row = (nz.astype(np.int32), c[nz], J_star - offset + rel * abs(J_star))
                b2 = None
                if s1.basis is not None:
                    b2 = _to_basis(s1.basis[0], np.concatenate([s1.basis[1], [_BASIC]]).astype(np.int8), alien=True)
                s2 = self._attempt(f"{rung}:stage2", lp, opts, b2, row=row, cost=w)
                trail.append((rung, s2.status))
                iters += s2.iterations
                status = s2.status
                if s2.x is not None:
                    return done(s2.x, status, s1.basis)
                continue
            res = self.solve(lp, shape, warm=True)  # "stage1": the primary optimum, tie-break skipped (counted)
            trail.append(("stage1", res.status))
            iters += res.iterations
            status = res.status
            if res.ok:
                return SolveResult(res.x, True, status, tuple(trail), iters, time.perf_counter() - start, res.objective)
        self._keep(None, None)
        return SolveResult(None, False, status, tuple(trail), iters, time.perf_counter() - start, None)

    # ----- resume -------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        """The chain's cross-week state (last basis as arrays, its shape) as plain data, for an in-process resume."""
        return {
            "col_status": None if self._col is None else self._col.tolist(),
            "row_status": None if self._row is None else self._row.tolist(),
            "shape": None if self._shape is None else dataclasses.asdict(self._shape),
        }

    def load_state(self, state: dict) -> None:
        """Restore ``state()``; the next solve warm-starts from it."""
        col, row, shape = state["col_status"], state["row_status"], state["shape"]
        self._col = None if col is None else np.array(col, dtype=np.int8)
        self._row = None if row is None else np.array(row, dtype=np.int8)
        self._shape = None if shape is None else WindowShape(**shape)

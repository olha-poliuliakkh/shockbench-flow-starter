"""Static tables and the observation of week t at the instant t-1 (design §5.1, §9.2; Q68, Q69, Q86).

Both are plain JSON-able dicts with the field names of the wire schema (§9.2): lists, ``None`` for a missing value,
integers indexing the static tables. ``observe`` is the environment core's observation, the same in every regime:
own state, the present graph ``graph_now`` (instantaneous values at t-1 from ``WeeklyMarks.*_now``; binary marks in
force in week t), ``slot_mask`` and ``last_week``, with ``demand_forecast``, ``warning``, ``messages``,
``pending_prohibitions`` and ``closure_end`` None; the observation wrapper outside the core,
``information.observe.wrap``, fills and degrades it per regime theta (design §5.4 "Rules"), and ``dynamics.env``
hands out only wrapped observations (``Env._observe``).

Obs = {
  "week": t,
  "stock":      {"node": [...], "k": [...], "qty": [...]},       # I^{t-1}, non-chokepoint slots with qty != 0
  "backlog":    {"node": [...], "k": [...], "qty": [...]},       # B^{t-1}, nonzero entries
  "pipeline":   {"edge", "k", "lane", "qty", "arrival_week"},     # post-clip quantities, lane None off lanes
  "queue_lots": {"lot_id", "chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week",
                 "entry_edge"},
  "wip":        {"node", "k", "qty", "out_week"},                  # gross WIP (Q69): fab raw chips, OSAT packaged chips
  "graph_now":  {"u": [E], "c": [E], "tau": [E], "prohibited": {"edge", "k"}, "tariff": {"edge", "k", "rate"},
                 "open": [C], "kappa": {"tb": [C], "ct": [C]}, "war_risk": [C] (vocabulary names),
                 "supply": {"node", "k", "avail"}, "fab": {"node", "R", "alpha_bar", "cap_eff"},
                 "grid": {"node", "G_bar", "y_bar"}, "osat": {"node", "R", "thr_eff"}},
  "slot_mask":  [bool per action slot],                           # True = an edge of the slot's route is prohibited
  "last_week":  {"clip": {"slot", "requested", "executed"}, "cost_components": {...8 floats},
                 "sinks": {"node", "k", "demand", "served", "lost"}, "shed": {"node", "qty"}} or None at week 1,
  "demand_forecast": None, "warning": None, "messages": None, "pending_prohibitions": None, "closure_end": None,
}
``u`` of a coupling edge is None (no capacity); ``kappa`` is the instantaneous throughput (k_c mu_cb) o at t-1.

Choices within the schema: ``c`` is the week-t freight ``WeeklyMarks.c`` (a piracy surcharge is a binary mark, in force
in week t; §3.1 table); ``pipeline`` lists every shipment, ``queue_lots`` every lot in book order and ``wip`` every
ledger entry (fabs by ordinal and start week, then OSATs by ordinal, out week and commodity); ``supply`` lists the stock
slots of supply nodes; ``clip`` lists the requested slots of last week in slot order. ``graph_now.osat`` shows each
OSAT's restoration R^osat at the instant t - 1 and its effective throughput thr_i R^osat_i (Q97; §5.1, §9.2, design
§12 "OSAT observation", decided): the core lists the key and its OSAT nodes with ``R`` and ``thr_eff`` None, and the
wrapper fills both from the view's R^osat at the instant t - 1 (``marks.osat_restoration_now`` of omega's events,
``marks.osat_throughput``; design §12 row "graph_now.osat in the wrapper"), so every direct caller of the core is
unchanged. After week T the environment's final
observation (week T + 1) keeps the own-state fields and sets ``graph_now`` and ``slot_mask`` to None, since no marks
exist beyond T. ``slot_mask[s]`` is True when any edge of slot s's route is prohibited for its commodity in week t
(Q111; design §12 "Slot mask (Q111)"): the route is the slot's edge on a laneless slot, else every edge of its lane
(``Static.lanes.edges``, whose first edge is the slot's edge). It is built here once, so the wire, the flat view's
``action_mask``, the gymnasium adapter and the agent kit read one rule; the dispatch rule of §9.3 still drops only an
entry whose own edge is prohibited (``dynamics.env``). ``Static`` carries ``dyads`` (the §9.2 M3 table, empty
without omega's ``dyad_regions``) and ``regime``, the one publisher ``information.theta.static_regime``:
prediction-free gives ``{"name": "prediction_free", "L": None, "a": {}, "phi": {}, "chi": False, "h_cov": None,
"skill": {}, "blackout": None}``, since no warning, messages or closure end are shown and none of its parameters
applies.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from operator import attrgetter

import numpy as np

from shockbench_flow.dynamics.state import State, StepRecord
from shockbench_flow.information.theta import Theta, resolve_regime, static_regime
from shockbench_flow.information.view import static_dyads
from shockbench_flow.instance.io import canonical_json
from shockbench_flow.instance.schema import POOLS, WAR_RISK_CLASSES, AltRef, Instance
from shockbench_flow.marks import WeeklyMarks


def _alt(ref: AltRef | None) -> dict | None:
    return None if ref is None else {ref.kind: ref.index}


def _cols(rows: Iterable[Sequence], names: Sequence[str]) -> dict[str, list]:
    """Rows of values as the wire schema's columns: ``{name: [row[i] for each row]}``."""
    rows = list(rows)
    return {n: [r[i] for r in rows] for i, n in enumerate(names)}


def _fields(objs: Iterable, names: tuple[str, ...]) -> dict[str, list]:
    """Attributes ``names`` of each object as columns named like the attributes."""
    return _cols(map(attrgetter(*names), objs), names)


_PIPELINE = ("edge", "k", "lane", "qty", "arrival_week")
_LOT = ("lot_id", "chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge")


def _canonical_text(inst: Instance) -> str:
    return canonical_json(inst.raw)


def static_view(
    inst: Instance, regime: str | Theta = "prediction_free", dyads: tuple[tuple[int, int], ...] = ()
) -> dict:
    """Build the ``Static`` tables of §9.2.

    They hold the full canonical instance JSON of §2.3 (a fresh parse of the text the hash reads, so every object's keys
    are sorted and two files with one hash give one Static, key order included), the index tables, action and override
    slots, sinks, the dyad table (``dyads``: omega's ``dyad_regions`` rows, () without them) and the published regime
    parameters (``theta.static_regime`` of ``resolve_regime(regime)``, a registry name or a theta of (47), with the
    chokepoint skill declared for the instance kind, M5-O16).

    Raises:
        ValueError: on an unknown regime name or a theta ``resolve_regime`` refuses.

    """
    theta = resolve_regime(regime)
    return {
        "instance": json.loads(inst.memo("dynamics.observe.canonical", _canonical_text)),
        "instance_id": inst.instance_id,
        "instance_hash": inst.hash,
        "T": inst.T,
        "units": dict(inst.units),
        "regions": list(inst.regions),
        "nodes": _fields(inst.nodes, ("id", "type", "region")),
        "commodities": _fields(inst.commodities, ("id", "v", "pool", "override")),
        "edges": {
            "id": [e.id for e in inst.edges],
            "tail": [e.tail for e in inst.edges],
            "head": [e.head for e in inst.edges],
            "mode": [e.mode for e in inst.edges],
            "tau0": [e.tau for e in inst.edges],
            "c0": [e.c0 for e in inst.edges],
            "u0": [e.u0 for e in inst.edges],
            "K": [list(e.K) for e in inst.edges],
            "alt_of": [_alt(e.alt_of) for e in inst.edges],
            "pool": [e.pool for e in inst.edges],
        },
        "lanes": {
            "id": [ln.id for ln in inst.lanes],
            "edges": [list(ln.edges) for ln in inst.lanes],
            "chokepoints": [list(ln.chokepoints) for ln in inst.lanes],
            "alt_of": [_alt(ln.alt_of) for ln in inst.lanes],
        },
        "action_slots": _cols(inst.action_slots, ("edge", "k", "lane")),
        "override_slots": _cols(inst.override_slots, ("chokepoint", "k", "out_edge", "lane")),
        "sinks": _fields(inst.demands, ("node", "k", "backlog", "pi")),
        "dyads": static_dyads(dyads),
        "regime": static_regime(theta, inst.kind),
    }


@dataclass(frozen=True)
class _Tables:
    """Week-invariant parts of one instance's observation (kept in ``Instance.memo``), handed out as fresh lists."""

    plain_slots: tuple[tuple[int, int, int], ...]  # (slot, node, k) of the non-chokepoint stock slots
    supply_slots: tuple[tuple[int, int, int], ...]  # (slot, node, k) of the supply nodes' stock slots
    demands: tuple[tuple[int, int], ...]  # (sink node, k) per demand
    coupling: tuple[bool, ...]  # per edge: a coupling edge shows u = None
    tau: tuple[int, ...]  # tau_e per edge
    cap0: tuple[float, ...]  # cap^0_f per fab ordinal
    route_edge: np.ndarray  # the edges of every action slot's route, slot after slot (Q111)
    route_k: np.ndarray  # the slot's commodity, once per route edge
    route_start: np.ndarray  # index of each slot's first route edge in route_edge (every route has one edge or more)

    @staticmethod
    def build(inst: Instance) -> "_Tables":
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        slots = [(s, sl.node, sl.k) for s, sl in enumerate(inst.stock_slots)]
        # each action slot's route (Q111): its edge off a lane, else every edge of its lane (the first is the slot's)
        routes = [((e,) if lane is None else inst.lanes[lane].edges, k) for e, k, lane in inst.action_slots]
        return _Tables(
            plain_slots=tuple(x for x in slots if x[1] not in chk),
            supply_slots=tuple(x for x in slots if x[1] in supply),
            demands=tuple((d.node, d.k) for d in inst.demands),
            coupling=tuple(e.coupling for e in inst.edges),
            tau=tuple(e.tau for e in inst.edges),
            cap0=tuple(inst.nodes[f].fab.cap0 for f in inst.fabs),
            route_edge=np.array([e for route, _k in routes for e in route], dtype=np.intp),
            route_k=np.array([k for route, k in routes for _e in route], dtype=np.intp),
            route_start=np.cumsum([0] + [len(route) for route, _k in routes], dtype=np.intp)[:-1],
        )


def _slot_mask(tb: _Tables, prohibited_t: np.ndarray) -> list[bool]:
    """``slot_mask`` of week t from Z_t (E, K): True where an edge of the slot's route is prohibited for it (Q111)."""
    return np.logical_or.reduceat(prohibited_t[tb.route_edge, tb.route_k], tb.route_start).tolist()


def _graph_now(inst: Instance, tb: _Tables, marks: WeeklyMarks, t: int) -> dict:
    ti = t - 1
    tar = marks.tariff[ti]
    z_e, z_k = np.nonzero(marks.prohibited[ti])  # row-major: edge, then commodity
    r_e, r_k = np.nonzero(tar)
    kappa_now = marks.kappa_now[ti]
    supply_now = marks.supply_now[ti].tolist()
    R, alpha = marks.R_now[ti].tolist(), marks.alpha_now[ti].tolist()
    return {
        "u": [None if cp else u for cp, u in zip(tb.coupling, marks.u_now[ti].tolist())],
        "c": marks.c[ti].tolist(),
        "tau": list(tb.tau),
        "prohibited": {"edge": z_e.tolist(), "k": z_k.tolist()},
        "tariff": {"edge": r_e.tolist(), "k": r_k.tolist(), "rate": tar[r_e, r_k].tolist()},
        "open": marks.o_now[ti].tolist(),
        "kappa": {b: kappa_now[:, bi].tolist() for bi, b in enumerate(POOLS)},
        "war_risk": [WAR_RISK_CLASSES[x] for x in marks.wr_class[ti].tolist()],
        "supply": _cols(((n, k, supply_now[s]) for s, n, k in tb.supply_slots), ("node", "k", "avail")),
        "fab": {
            "node": list(inst.fabs),
            "R": R,
            "alpha_bar": alpha,
            "cap_eff": [alpha[fi] * R[fi] * cap0 for fi, cap0 in enumerate(tb.cap0)],
        },
        "grid": {
            "node": list(inst.grids),
            "G_bar": marks.G_bar_now[ti].tolist(),
            "y_bar": marks.y_bar_now[ti].tolist(),
        },
        # Q97: the key and its nodes; R and thr_eff are the wrapper's, from the view (module docstring)
        "osat": {"node": list(inst.osats), "R": [None] * len(inst.osats), "thr_eff": [None] * len(inst.osats)},
    }


def _last_week(inst: Instance, tb: _Tables, rec: StepRecord | None) -> dict | None:
    if rec is None:
        return None
    slots = sorted(rec.requested)
    return {
        "clip": {
            "slot": slots,
            "requested": [float(rec.requested[s]) for s in slots],
            "executed": [float(rec.executed.get(s, 0.0)) for s in slots],
        },
        "cost_components": rec.costs.as_dict(),
        "sinks": {
            **_cols(tb.demands, ("node", "k")),
            "demand": rec.demand.tolist(),
            "served": rec.served.tolist(),
            "lost": rec.lost.tolist(),
        },
        "shed": {"node": list(inst.grids), "qty": rec.shed.tolist()},
    }


def observe(inst: Instance, marks: WeeklyMarks, state: State) -> dict:
    """The core observation of week ``state.week + 1`` at the instant ``state.week`` (1), the five feeds None."""
    tb: _Tables = inst.memo("dynamics.observe", _Tables.build)
    t = state.week + 1
    stock = state.stock.tolist()
    backlog = state.backlog.tolist()
    wip = []
    for fi, f in enumerate(inst.fabs):
        fab = inst.nodes[f].fab
        for start, q in sorted(state.fab_wip.get(fi, {}).items()):
            wip.append((f, fab.product, float(q), start + fab.tau))
    for oi, o in enumerate(inst.osats):
        for out, book in sorted(state.osat_wip.get(oi, {}).items()):
            for k, q in sorted(book.items()):
                wip.append((o, k, float(q), out))
    in_horizon = t <= marks.T
    return {
        "week": t,
        "stock": _cols(((n, k, stock[s]) for s, n, k in tb.plain_slots if stock[s] != 0.0), ("node", "k", "qty")),
        "backlog": _cols(((n, k, b) for (n, k), b in zip(tb.demands, backlog) if b != 0.0), ("node", "k", "qty")),
        "pipeline": _fields(state.pipeline, _PIPELINE),
        "queue_lots": _fields(state.lots, _LOT),
        "wip": _cols(wip, ("node", "k", "qty", "out_week")),
        "graph_now": _graph_now(inst, tb, marks, t) if in_horizon else None,
        "slot_mask": _slot_mask(tb, marks.prohibited[t - 1]) if in_horizon else None,
        "last_week": _last_week(inst, tb, state.last),
        "demand_forecast": None,
        "warning": None,
        "messages": None,
        "pending_prohibitions": None,
        "closure_end": None,
    }

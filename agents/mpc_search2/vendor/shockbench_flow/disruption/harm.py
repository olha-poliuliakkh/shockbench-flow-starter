"""Harm of an event and of an episode, (41)-(43) (design §4.5; Q41, Q50, Q61, Q76 owner D7).

H_q = u-bar_q T^in_q x (sigma_q v-bar_q for capacity events and prohibitions: lost capacity-weeks valued in USD;
Delta c_q for cost events: tariffs tau^tar v_k, piracy 0.08-0.12 c^0_e), summed over the event's targets; T^in_q is
the part of its window D_q inside the episode [0, T) (for a war profile W0 u W1 of (40), Q87); u-bar_q is the target's
nominal weekly capacity (sum_b k_c mu_cb at a chokepoint with each pool valued at its own v-bar; u^0_e on an edge or
for a prohibited pair; cap^0_f at a fab; thr_i at an OSAT; the supply rate at a supply node; G-bar^0_g at a grid);
v-bar_q is the mean customs value v_k of the commodities the target carries at reset. H(omega) = sum_q H_q (42),
capacity and cost harm 1:1 in USD; H_ty (43) per type label (``totals`` gives all three from one pass). The affected
targets are those of ``marks`` (one rule table, V3), read through its public API (``read_events``, ``graph_marks``,
``event_capacity_marks``). V17's capacity-loss share reads the same valuation week by week (``capacity_loss_share``,
re-exported by ``realism``), the node targets by the same loss rule (``_node_losses``).

How the rule table is reused, event by event (each event alone, so no other event's product (37) enters):

- Capacity: ``marks``' own week averages of the event over the weeks 1..T: an edge, open fraction, supply slot or
  grid loses sigma^t f^t per week by (37) and (40), so the sum over weeks of (nominal - marked) is u-bar sigma T^in,
  the war profile's two severities included; a fab's loss is the sum over weeks of 1 - R_f(t) of (13), the dead time
  and the restoration tail, the same integral of sigma theta_q(s) over [0, T); an OSAT's loss is the sum over weeks
  of 1 - R_osat(t) of (13) on the OSATs of m_q of a regional conflict (§4.4), u-bar = thr_i, by the same rule.
  Friendly fire (37) is a capacity cut.
- Prohibitions and cost marks (binary marks (1)): the pairs, tariff rates and piracy surcharges that ``marks`` puts
  in force for the event at full force (a probe week inside its window), times the continuous T^in of D_q =
  [onset, onset + T_q), as (41) is written. Only three types put these marks in force (§4.4 type table): sanctions
  and export controls (Z_t), tariffs and piracy (``BINARY_TYPES``); the probe runs for them alone, since for every
  other type it finds nothing.

Valuation, where §4.5 leaves the reading open (reported for design rows): the commodities an edge carries at reset are
K_e less its pairs in Z_0 (so a pair prohibited from week 0 harms nothing, and an edge carrying nothing has no
harm); a prohibited or taxed pair (e, k) takes the equal share u^0_e / |K^0_e| of the edge's capacity at v_k, so a
whole-edge prohibition is u^0_e v-bar_e; piracy's u-bar is u^0_e on an edge that carries something at reset; the
commodities a node carries are those of its stock slots (a chokepoint pool's v-bar is the mean over its slots of
that pool, a fab's over its wafer input and raw-chip output); an OSAT's v-bar is the mean v_k of the raw chips it
packages (the keys of ``packages``; on `tiny` raw and packaged chips are both 20,000 USD); a supply slot is a target
of its own, supply rate x v_k.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from shockbench_flow import marks as _marks
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import PIRACY, SANCTION, TARIFF, MarkParams
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega


BINARY_TYPES = frozenset({SANCTION, TARIFF, PIRACY})  # the types whose operation sets Z_t, tariffs or freight (§4.4)


@dataclass(frozen=True)
class _Values:
    """Nominal capacities and valuations of one instance's targets (§4.5), per index of the ``WeeklyMarks`` axes."""

    u0: np.ndarray  # (E,) u^0_e, inf on coupling edges
    finite: np.ndarray  # (E,) bool, edges with a capacity
    edge_vbar: np.ndarray  # (E,) mean v_k over K^0_e (0 if the edge carries nothing at reset)
    edge_cap: np.ndarray  # (E,) u^0_e if the edge carries something at reset, else 0
    pair_value: np.ndarray  # (E, K) u^0_e / |K^0_e| x v_k for k in K^0_e, else 0
    pair_cap: np.ndarray  # (E, K) u^0_e / |K^0_e| for k in K^0_e, else 0
    v: np.ndarray  # (K,) v_k
    chokepoint: np.ndarray  # (C,) sum_b k_c mu_cb v-bar_cb
    supply0: np.ndarray  # (S,) supply rate per stock slot
    slot_v: np.ndarray  # (S,) v_k of the slot's commodity
    grid0: np.ndarray  # (G,) G-bar^0_g
    grid_vbar: np.ndarray  # (G,) mean v_k of the grid's slots
    fab: np.ndarray  # (F,) cap^0_f x mean v_k of the fab's slots
    osat: np.ndarray  # (O,) thr_i x mean v_k of the raw chips the OSAT packages


def _build_values(inst: Instance) -> _Values:
    E, K = len(inst.edges), len(inst.commodities)
    v = np.array([c.v for c in inst.commodities], dtype=np.float64)
    z0 = set(inst.prohibitions_at_reset)
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges], dtype=np.float64)
    edge_vbar, edge_cap = np.zeros(E), np.zeros(E)
    pair_value, pair_cap = np.zeros((E, K)), np.zeros((E, K))
    for e, edge in enumerate(inst.edges):
        carried = [k for k in edge.K if (e, k) not in z0]
        if edge.coupling or not carried:
            continue
        edge_vbar[e] = float(np.mean(v[carried]))
        edge_cap[e] = u0[e]
        for k in carried:
            pair_cap[e, k] = u0[e] / len(carried)
            pair_value[e, k] = pair_cap[e, k] * v[k]

    def node_mean(n: int, pool: int | None = None) -> float:
        ks = [s.k for s in inst.stock_slots if s.node == n and (pool is None or inst.commodity_pool[s.k] == pool)]
        return float(np.mean(v[ks])) if ks else 0.0

    chk = np.array(
        [
            math.fsum(kap * node_mean(c, b) for b, kap in enumerate(inst.nodes[c].chokepoint.kappa0))
            for c in inst.chokepoints
        ]
    )
    return _Values(
        u0=u0,
        finite=np.isfinite(u0),
        edge_vbar=edge_vbar,
        edge_cap=edge_cap,
        pair_value=pair_value,
        pair_cap=pair_cap,
        v=v,
        chokepoint=chk.reshape(len(inst.chokepoints)),
        supply0=np.array([s.supply for s in inst.stock_slots], dtype=np.float64),
        slot_v=np.array([v[s.k] for s in inst.stock_slots], dtype=np.float64),
        grid0=np.array([inst.nodes[g].grid.deliverable for g in inst.grids], dtype=np.float64),
        grid_vbar=np.array([node_mean(g) for g in inst.grids], dtype=np.float64),
        fab=np.array([inst.nodes[f].fab.cap0 * node_mean(f) for f in inst.fabs], dtype=np.float64),
        osat=np.array([_osat_value(inst, o, v) for o in inst.osats], dtype=np.float64),
    )


def _osat_value(inst: Instance, o: int, v: np.ndarray) -> float:
    """thr_i x v-bar_i: the OSAT's nominal throughput valued at the mean v_k of the raw chips it packages (§4.5)."""
    attrs = inst.nodes[o].osat
    raw = sorted(attrs.packages)
    return attrs.thr * float(np.mean(v[raw])) if raw else 0.0


def _resolve(inst: Instance, omega, params: MarkParams | None) -> tuple[Mapping[str, np.ndarray], MarkParams]:
    """The arrays of omega and the MarkParams to read them under: omega's own when it states them (V3)."""
    arrays = omega.arrays if isinstance(omega, Omega) else omega
    if "meta_instance_hash" in arrays and str(arrays["meta_instance_hash"]) != inst.hash:
        raise ValueError("omega's meta_instance_hash is not the instance hash (V3)")
    if "meta_mark_params" in arrays:
        own = _marks.mark_params_from_json(str(arrays["meta_mark_params"]))
        if params is not None and params != own:
            raise ValueError("MarkParams differ from the ones omega was built under (meta_mark_params, V3)")
        return arrays, own
    if params is None:
        raise ValueError("the events state no meta_mark_params; pass the MarkParams they were built under (V3)")
    return arrays, params


def _t_in(start: float, end: float, T: int) -> float:
    """|[start, end) ∩ [0, T)|, the part of a window inside the episode (41)."""
    return max(0.0, min(end, float(T)) - max(start, 0.0))


def _values(inst: Instance) -> _Values:
    """The instance's valuation table (§4.5), built once per instance (``Instance.memo``)."""
    return inst.memo("harm_values", _build_values)


def _node_losses(
    vals: _Values,
    o: np.ndarray,
    supply: np.ndarray,
    G_bar: np.ndarray,
    R_f: np.ndarray | None = None,
    R_osat: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray | float, np.ndarray], ...]:
    """(value, nominal, lost) of each node target type: the one loss rule of harm (41) and V17's share.

    Per target type, its value per unit of weekly capacity (``_Values``), its nominal weekly capacity and the (T, n)
    capacity the week's marks (37) remove from it, nominal - marked: 1 - o^t_c at a chokepoint, supply^0 - supply^t
    of a supply slot, G-bar^0_g - G-bar^t_g of a grid, and when their restoration factors of (13) are given, 1 - R_f(t)
    at a fab and 1 - R_osat(t) at an OSAT. Edges and prohibited pairs are each reading's own (``_capacity_terms``,
    ``capacity_loss_share``).
    """
    rows = [
        (vals.chokepoint, 1.0, 1.0 - o),
        (vals.slot_v, vals.supply0, vals.supply0 - supply),
        (vals.grid_vbar, vals.grid0, vals.grid0 - G_bar),
    ]
    if R_f is not None:
        rows += [(vals.fab, 1.0, 1.0 - R_f), (vals.osat, 1.0, 1.0 - R_osat)]
    return tuple(rows)


def _capacity_terms(vals: _Values, g: Mapping[str, np.ndarray]) -> list[float]:
    """Lost capacity-weeks of one event alone, valued in USD: its week averages ``g`` summed over the episode.

    ``g`` is ``marks.event_capacity_marks`` of the event; it holds ``R_f`` and ``R_osat`` of (13) when q hits a fab or
    an OSAT (the fabs and OSATs of m_q only; the others keep R = 1: the dead time and the restoration tail).
    """
    fin = vals.finite
    terms = list(vals.edge_vbar[fin] * np.sum(vals.u0[fin] - g["u"][:, fin], axis=0))
    for value, _, lost in _node_losses(vals, g["o"], g["supply"], g["G_bar"], g.get("R_f"), g.get("R_osat")):
        terms += list(value * np.sum(lost, axis=0))
    return [float(x) for x in terms]


def _binary_terms(inst: Instance, q: _marks.Event, params: MarkParams, vals: _Values, base: dict) -> list[float]:
    """Prohibited pairs, tariff duties and piracy surcharges of q at full force per week, times T^in of D_q (41).

    ``base`` holds the event-free marks; the probe moves q to [0, 1), in force in week 1 exactly by (1), row 0.
    """
    t_in = _t_in(q.onset, q.onset + q.duration, inst.T)
    if t_in == 0.0:
        return []
    probe = dataclasses.replace(q, onset=0.0, duration=1.0)
    g = _marks.graph_marks(inst, (probe,), params)
    new = g["prohibited"][0] & ~base["prohibited"][0]  # pairs of Z_0 were never available
    terms = list(vals.pair_value[new])
    terms += list((vals.pair_cap * vals.v[None, :] * g["tariff"][0])[g["tariff"][0] != 0.0])
    terms += list((vals.edge_cap * (g["c"][0] - base["c"][0]))[g["c"][0] != base["c"][0]])
    return [float(x) * t_in for x in terms]


@_marks.fixed_fp_errors
def event_harm(
    inst: Instance, omega, params: MarkParams | None = None, *, capacity: Sequence[Mapping] | None = None
) -> np.ndarray:
    """H_q of (41) per stored event of ``omega``, in omega's event order (USD).

    Args:
        inst: the instance omega was built on.
        omega: an ``Omega``, or a mapping holding its ``ev_*`` arrays (the event-list draws of the strata, (44)).
        params: the MarkParams of the event-to-graph rules; omega's ``meta_mark_params`` when it states them (a
            different ``params`` then raises), required otherwise.
        capacity: ``marks.event_capacity_marks`` of each stored event under those MarkParams, in omega's event order,
            when the caller already holds them (``realism.episode_realism`` shares them with its overlap count); None
            computes them here.

    Returns:
        (events,) float64, every entry finite and >= 0.

    Raises:
        ValueError: on another instance's omega, MarkParams that differ from omega's or are missing, an event the
            rule table refuses (``marks``), or ``capacity`` of another length than the event list.

    """
    arrays, params = _resolve(inst, omega, params)
    events = _marks.read_events(inst, arrays)
    if capacity is not None and len(capacity) != len(events):
        raise ValueError(f"capacity holds the marks of {len(capacity)} events, omega stores {len(events)}")
    vals = _values(inst)
    base: dict | None = None  # the event-free marks, formed for the first event with binary marks
    out = np.zeros(len(events), dtype=np.float64)
    for i, q in enumerate(events):
        g = _marks.event_capacity_marks(inst, q, params) if capacity is None else capacity[i]
        terms = _capacity_terms(vals, g)
        if q.type in BINARY_TYPES:  # every other type puts no binary mark in force: its probe finds no term
            if base is None:
                base = _marks.graph_marks(inst, (), params)
            terms += _binary_terms(inst, q, params, vals, base)
        out[i] = math.fsum(terms) + 0.0
    return out


def totals(
    inst: Instance, omega, params: MarkParams | None = None, *, capacity: Sequence[Mapping] | None = None
) -> tuple[np.ndarray, float, dict[int, float]]:
    """(H_q of (41) per event, H(omega) of (42), H_ty of (43) per type code), from one pass of ``event_harm``.

    H(omega) is the exactly rounded sum of the events' harms; H_ty covers every code of ``codes.EVENT_TYPES`` (0.0 when
    absent), each the exactly rounded sum over its events. Arguments and errors as ``event_harm``.
    """
    h = event_harm(inst, omega, params, capacity=capacity)
    arrays = omega.arrays if isinstance(omega, Omega) else omega
    types = np.asarray(arrays["ev_type"]) if "ev_type" in arrays else np.zeros(0, dtype=np.int16)
    by_type = {code: math.fsum(h[types == code].tolist()) for code in range(len(codes.EVENT_TYPES))}
    return h, math.fsum(h.tolist()), by_type


def episode_harm(inst: Instance, omega, params: MarkParams | None = None) -> float:
    """H(omega) of (42), the exactly rounded sum of the events' harms (``totals``)."""
    return totals(inst, omega, params)[1]


def harm_by_type(inst: Instance, omega, params: MarkParams | None = None) -> dict[int, float]:
    """H_ty of (43) per event type code (display labels), every code of ``codes.EVENT_TYPES`` (``totals``)."""
    return totals(inst, omega, params)[2]


def capacity_loss_share(inst: Instance, marks: _marks.WeeklyMarks) -> np.ndarray:
    """(T,) the capacity-loss share of every week: the share of the nominal weekly capacity the week's marks remove.

    The V17 row compares the "capacity-loss share in week 1 against week ceil(T/2)" and the design defines it nowhere
    else; the reading fixed here (reported for a design row) is harm's valuation (41) of §4.5 read week by week:

    - every capacity target is valued at its nominal weekly capacity times the mean customs value v-bar of what it
      carries at reset (the one table, ``_Values``): a chokepoint's k_c mu_cb per pool, each pool at its own v-bar;
      each pair (e, k) of a finite edge, k in K_e less Z_0, at u^0_e / |K^0_e| times v_k; a supply slot's rate at v_k;
      a grid's G-bar^0_g; a fab's cap^0_f; an OSAT's thr_i;
    - the week's capacity marks (37) remove from each its lost part (the node targets by harm's one loss rule,
      ``_node_losses``): 1 - o^t_c at a chokepoint; 1 - u^t_e / u^0_e of a pair, or the whole pair while (e, k) is
      prohibited in week t (the weeks it is in force by (1), where harm prices a prohibition by the continuous T^in
      of D_q, so the two part for a prohibition whose duration is not a whole number of weeks); supply^0 - supply^t
      of a slot; G-bar^0_g - G-bar^t_g of a grid; 1 - R_f(t) and 1 - R_osat(t) of (13) at a fab and an OSAT;
    - the share is the lost value over the total value: 0 without events, 1 when nothing is left. Cost marks
      (tariffs, piracy surcharges, war-risk costs) remove no capacity and are not counted. On a finite edge with
      u^0_e = 0 (loader-valid) u^t_e / u^0_e is 0/0 := 0, the design's rule; its pairs carry no value either way.

    Args:
        inst: the instance.
        marks: the episode's ``WeeklyMarks`` (``marks.compute_marks``).

    Returns:
        (T,) float64 shares in [0, 1].

    """
    vals = _values(inst)
    fin = vals.finite
    frac = np.ones_like(marks.u)
    u0 = vals.u0[fin]
    # u^t_e / u^0_e with 0/0 := 0 (§12 row 'Production order and divisions'): an edge with u^0_e = 0 carries no value
    frac[:, fin] = np.divide(marks.u[:, fin], u0, out=np.zeros_like(marks.u[:, fin]), where=u0 > 0.0)
    pairs = vals.pair_value.sum()
    lost = pairs - (vals.pair_value[None, :, :] * frac[:, :, None] * ~marks.prohibited).sum(axis=(1, 2))
    total = pairs
    for value, nominal, loss in _node_losses(vals, marks.o, marks.supply, marks.G_bar, marks.R, marks.R_osat):
        lost = lost + (value * loss).sum(axis=1)
        total = total + (value * nominal).sum()
    return lost / total

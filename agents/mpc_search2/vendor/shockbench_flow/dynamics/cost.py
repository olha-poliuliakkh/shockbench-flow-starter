"""The weekly cost C_t of (23) and the terminal credit S_T (design §3.6; Q3, Q56, Q58, Q68, Q73, Q79, Q93).

Every component is a ``math.fsum`` over explicit terms, with no ``@`` or ``np.dot`` (§3.6); C_t is the fsum of the
components and becomes integer cents once, by (24), in the caller. Flow terms are per (edge, commodity), with x_ek of
(52) the fsum of the week's lane flows x_ekl (``edge_flows``): exact and order-free, so the LP, which holds the same
pieces as lane columns (one aggregated column for a K^ov release), forms the same x_ek and the same cents (53). Tariffs
are charged in the dispatch week at the rate in force then, as ``(rate * v_k) * x_ek`` (Q58 E24, M8).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shockbench_flow.dynamics.state import CostComponents, State
from shockbench_flow.instance.schema import Instance


@dataclass(frozen=True)
class _Tables:
    """Week-invariant cost coefficients of one instance (kept in ``Instance.memo``), in the term order of (23)."""

    v: tuple[float, ...]  # customs value v_k per commodity
    holding: tuple[tuple[int, float], ...]  # (slot, h_ik) at non-chokepoint slots
    queue: tuple[tuple[int, int, int], ...]  # (slot, chokepoint ordinal, k) at chokepoint slots
    disposal: tuple[float, ...]  # c^disp_k per stock slot
    pi: tuple[float, ...]  # pi_ik per demand
    voll: tuple[float, ...]  # VOLL_g per grid ordinal

    @staticmethod
    def build(inst: Instance) -> "_Tables":
        chk = inst.chokepoint_ordinal
        slots = list(enumerate(inst.stock_slots))
        return _Tables(
            v=tuple(com.v for com in inst.commodities),
            holding=tuple((s, sl.holding) for s, sl in slots if sl.node not in chk),
            queue=tuple((s, chk[sl.node], sl.k) for s, sl in slots if sl.node in chk),
            disposal=tuple(inst.commodities[sl.k].disposal_cost for _s, sl in slots),
            pi=tuple(d.pi for d in inst.demands),
            voll=tuple(inst.nodes[g].grid.voll for g in inst.grids),
        )


def edge_flows(x: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """x_ek of (23), (52): the week's lane flows x_ekl of each (edge, commodity), summed by ``math.fsum``.

    The one formation of x_ek, shared by the simulator (over its lane flows) and the LP (over its lane columns, replay's
    aggregated K^ov column included): fsum is exact and independent of the order of the pieces, so both sides charge
    the same C_t for the same lane flows (design §12, cost terms (23) and oracle cents).

    Args:
        x: (edge, commodity, lane or None) -> flow, in any order.

    Returns:
        (edge, commodity) -> x_ek, in order of first appearance.

    """
    return _fsum_over_lanes(x)


def queue_totals(queue: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """I_ck at a chokepoint (23), (52): the lane queues Q_ckl of each (chokepoint, commodity), summed by ``math.fsum``.

    The one formation of the chokepoint stock: the simulator applies it to its lane queues (each lane's lots added in
    book order, the values ``StepRecord.queue`` records), the LP to its lane columns and replay to the recorded lane
    queues, so h^Q_ck I_ck is the same float on every side (design §12, cost terms (23)).

    Args:
        queue: (chokepoint, commodity, lane) -> Q_ckl, in any order.

    Returns:
        (chokepoint, commodity) -> I_ck, in order of first appearance.

    """
    return _fsum_over_lanes(queue)


def _fsum_over_lanes(pieces: Mapping[tuple[int, int, int | None], float]) -> dict[tuple[int, int], float]:
    """(a, b, lane) -> q summed over the lanes by ``math.fsum`` per (a, b), exact and independent of the order."""
    groups: dict[tuple[int, int], list[float]] = {}
    for (a, b, _lane), q in pieces.items():
        groups.setdefault((a, b), []).append(float(q))
    return {ab: math.fsum(qs) for ab, qs in groups.items()}


def weekly_costs(
    inst: Instance,
    c: Sequence[float],
    c_wr,
    tariff,
    h_queue,
    x: Mapping[tuple[int, int, int | None], float],
    stock: Sequence[float],
    disposal: Sequence[float],
    lost: Sequence[float],
    backlog: Sequence[float],
    shed: Sequence[float],
) -> CostComponents:
    """The eight components of C_t in (23) for one week.

    Args:
        inst: the instance.
        c: unit freight c^t_e per edge.
        c_wr: per-transit war-risk cost, ``c_wr[e][k]`` (11).
        tariff: ad valorem rate, ``tariff[e][k]``.
        h_queue: queue holding, ``h_queue[chokepoint ordinal][k]``.
        x: (edge, commodity, lane or None) -> executed flow x_ekl of the week, dispatches and chokepoint releases
            (the record's ``x``); the flow terms take x_ek = ``edge_flows(x)``.
        stock: end-of-week I^t per stock slot (queue totals at chokepoints).
        disposal: O^t per stock slot.
        lost: U^t per demand.
        backlog: B^t per demand.
        shed: y^sh,t per grid ordinal.

    """
    tb: _Tables = inst.memo("dynamics.cost", _Tables.build)
    v = tb.v
    x_ek = edge_flows(x)
    return CostComponents(
        freight=math.fsum(c[e] * q for (e, _k), q in x_ek.items()),
        war_risk=math.fsum(c_wr[e][k] * q for (e, k), q in x_ek.items()),
        tariff=math.fsum((tariff[e][k] * v[k]) * q for (e, k), q in x_ek.items()),  # (tau^tar v_k) x_ek (23)
        holding=math.fsum(h * stock[s] for s, h in tb.holding),
        queue_holding=math.fsum(h_queue[ci][k] * stock[s] for s, ci, k in tb.queue),
        shortage=math.fsum(pi * (lost[i] + backlog[i]) for i, pi in enumerate(tb.pi)),
        disposal=math.fsum(dc * disposal[s] for s, dc in enumerate(tb.disposal)),
        shed=math.fsum(voll * shed[i] for i, voll in enumerate(tb.voll)),
    )


def transit_pieces(pipeline) -> dict[int, dict[tuple[int, int, int | None], float]]:
    """P^T of (23) by dispatch week: week -> (edge, commodity, lane) -> the lane's shipments added in pipeline order.

    Each lane piece is added from 0.0 in the order the simulator appended the shipments (dispatches in slot order, then
    chokepoint releases in release order), the formation of the week's x_ekl (``StepRecord.x``); ``edge_flows`` of a
    week's pieces is then the x_ek its flow terms charged, which the LP holds in its week's lane columns (Q93).
    """
    weeks: dict[int, dict[tuple[int, int, int | None], float]] = {}
    for sh in pipeline:
        book = weeks.setdefault(sh.dispatch_week, {})
        key = (sh.edge, sh.k, sh.lane)
        book[key] = book.get(key, 0.0) + sh.qty
    return weeks


def salvage(inst: Instance, state: State) -> float:
    """S_T of (23) from the end-of-horizon state (Q3, Q56, Q79, Q93).

    Stock at nu_ik, a chokepoint's queue total (the fsum of its lane queues, ``queue_totals``) at the chokepoint slot's
    nu = c^min x the commodity's share (Q93), 0 at supply nodes (Q79); shipments still in transit at the nu of their
    edge's head, where they are going (Q93), one term per (edge, commodity, dispatch week): nu_head times x_ek, the
    fsum over lanes of the week's lane pieces (``transit_pieces``, ``edge_flows``), the x_ek its flow terms charged, so
    the LP forms the same float from its lane columns; fab WIP at the salvage of its wafer input and OSAT WIP at the
    salvage of its raw-chip input, both net of the scrap booked by T (the ledger is net once booked, Q56).
    """
    supply = set(inst.supply_nodes)

    def nu(node: int, k: int) -> float:
        if node in supply:
            return 0.0
        s = inst.slot_index.get((node, k))
        return 0.0 if s is None else inst.stock_slots[s].salvage

    terms = [nu(sl.node, sl.k) * float(state.stock[s]) for s, sl in enumerate(inst.stock_slots)]
    for pieces in transit_pieces(state.pipeline).values():
        terms += [nu(inst.edges[e].head, k) * x for (e, k), x in edge_flows(pieces).items()]
    for fi, f in enumerate(inst.fabs):
        rate = nu(f, inst.nodes[f].fab.input)
        terms += [rate * q for q in state.fab_wip.get(fi, {}).values()]
    for oi, o in enumerate(inst.osats):
        raw_of = {pk: raw for raw, pk in inst.nodes[o].osat.packages.items()}
        for book in state.osat_wip.get(oi, {}).values():
            terms += [nu(o, raw_of[k]) * q for k, q in book.items()]
    return math.fsum(terms)

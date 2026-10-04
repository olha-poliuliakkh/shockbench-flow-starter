"""The chokepoint lot book and its weekly release, design §3.4 pseudocode steps 1-3, (9)-(10) (Q4, Q25, Q34, Q68, Q86).

Step 1: shipments arriving at a chokepoint in week t become lots under the lot id they received at dispatch (in
action-slot order, then release order for tandem-lane releases; initial shipments at reset in pipeline order).

Step 2, overrides first (tanker cargo K^ov only): any override slot or hold for (c, k) turns the default release of k at
c off this week. A hold releases nothing; when a hold and override slots name the same (c, k), the hold wins. Override
quantities pass (3) (K_e and Z_t), (4) the out-edge capacity per edge over its override slots, (5) the queue content of
k at c and (6) the pool's throughput kappa_cb, each factor pro rata across the override slots it covers. On an out-edge
whose float factor (4) is subnormal, the override slots then pass the edge clamp of the dispatch clip in slot order,
before (6) (``clip.edge_clamp``, Q95): a residual of u'_e left in [-b, 0), with b = 1e-12 max(u'_e, |x|, q, smallest
normal float) and q the slot's override request, lowers that slot to the residual before it and is logged as
(override slot, value) (``StepRecord.override_clamps``); anything lower raises ``RuntimeError``. Units are then taken
FIFO within k by (arrival_week, dispatch_week, entry_edge, lane, lot_id); the taken part ships with the slot's lane
(None on a turn-back on no lane) and the remainder of a split lot keeps its id and its place in the book.

Step 3, the default release (10) on the residual kappa'_cb and u'_e: arrival cohorts in increasing order and, inside a
cohort, the pools (tb, ct); a lot whose next edge is prohibited for its commodity releases nothing; next-edge capacity
pro rata inside the cohort, then throughput pro rata inside the cohort. The arithmetic follows the design's reference
code (``tiny_fixture.py``, ``Sim.step`` step 3) exactly, including its residual updates ``u' -= X`` per released lot and
``kappa' -= eta_kappa * total``, so the package reproduces its cents.

Releases returned here are tentative: the simulator applies the fleet slack (7) last, jointly with the week's duplicate
dispatches, and only then takes the released quantities out of the lots; the unreleased share stays in the lot it came
from, which keeps its key (§3.4 step 4).
"""

from collections import defaultdict
from dataclasses import dataclass

from sbfplan.dynamics.clip import edge_stock_clip
from sbfplan.dynamics.state import Lot, State
from sbfplan.instance.schema import Instance


@dataclass
class Release:
    """A tentative release of part of one lot onto an out-edge of its chokepoint."""

    lot: Lot
    edge: int
    k: int
    lane: int | None
    qty: float
    override_slot: int | None  # None for the default release


def fifo_key(lot: Lot) -> tuple[int, int, int, int, int]:
    """The FIFO order of overrides (§3.4 step 2)."""
    return (lot.arrival_week, lot.dispatch_week, lot.entry_edge, lot.lane, lot.lot_id)


def arrivals_to_lots(inst: Instance, state: State, t: int) -> None:
    """Step 1: shipments arriving at a chokepoint in week t leave the pipeline and join its lot book."""
    chk = inst.chokepoint_ordinal
    keep = []
    for s in state.pipeline:
        head = inst.edges[s.edge].head
        if s.arrival_week == t and head in chk:
            nxt = None if s.lane is None else inst.lane_next_edge(s.lane, s.edge)
            if nxt is None:
                raise ValueError(
                    f"shipment on edge {inst.edges[s.edge].id} reaches a chokepoint without a lane to follow"
                )
            if s.lot_id is None:
                raise ValueError(f"shipment on edge {inst.edges[s.edge].id} reaches a chokepoint without a lot id")
            state.lots.append(Lot(s.lot_id, head, s.k, s.qty, s.lane, nxt, s.dispatch_week, s.edge, t))
        else:
            keep.append(s)
    state.pipeline = keep


def release(
    inst: Instance,
    lots: list[Lot],
    u: list[float],
    kappa: list[list[float]],
    prohibited,
    overrides: dict[int, float],
    holds: frozenset[tuple[int, int]],
    clamps: list[tuple[int, float]] | None = None,
) -> list[Release]:
    """Steps 2-3 at every chokepoint (node order): overrides first, then the default release (10).

    Args:
        inst: the instance.
        lots: the lot book after this week's arrivals (not modified here).
        u: u^t_e per edge.
        kappa: kappa^t_cb per chokepoint ordinal and pool.
        prohibited: Z_t, indexable as ``prohibited[e][k]``.
        overrides: override slot -> quantity (validated); every slot turns the default release of its (c, k) off.
        holds: (chokepoint node, k) held this week.
        clamps: if given, each edge clamp of an override slot (Q95; ``_overrides``) is appended to it as (override
            slot, value), value < 0, per chokepoint in slot order. The clamp acts whether or not it is logged.

    Returns:
        Tentative releases in release order: per chokepoint, override pieces in slot order, then default releases by
        cohort, pool and book order.

    Raises:
        RuntimeError: an out-edge residual below the clamp band (``clip.edge_clamp``).

    """
    pool = inst.commodity_pool
    off = set(holds) | {inst.override_slots[s][:2] for s in overrides}
    books: dict[int, list[Lot]] = defaultdict(list)  # chokepoint node -> its lots in book order
    for lt in lots:
        books[lt.chokepoint].append(lt)
    override_slots: dict[int, list[int]] = defaultdict(list)  # chokepoint node -> its override slots in slot order
    for s in sorted(overrides):
        override_slots[inst.override_slots[s][0]].append(s)
    ures = list(u)
    out: list[Release] = []
    for ci, c in enumerate(inst.chokepoints):
        kap = list(kappa[ci])
        book = books.get(c, [])
        # ----- step 2: overrides ---------------------------------------------------------------------------------
        if slots := override_slots.get(c):
            _overrides(inst, c, book, slots, overrides, holds, prohibited, ures, kap, out, clamps)
        # ----- step 3: default release (10), FIFO across cohorts, pro rata inside a cohort ------------------------
        cohorts: dict[tuple[int, int], list[Lot]] = defaultdict(list)  # (arrival week, pool) -> lots in book order
        for lt in book:
            if lt.qty > 0 and (c, lt.k) not in off:
                cohorts[(lt.arrival_week, pool[lt.k])].append(lt)
        for a, b in sorted(cohorts):
            coh = cohorts[(a, b)]
            # Y-tilde of (10): a lot releases nothing onto a next edge that is prohibited for it or does not permit it
            y = [
                0.0 if prohibited[lt.next_edge][lt.k] or lt.k not in inst.edges[lt.next_edge].K else lt.qty
                for lt in coh
            ]
            tot: dict[int, float] = {}
            for lt, yv in zip(coh, y):
                tot[lt.next_edge] = tot.get(lt.next_edge, 0.0) + yv
            eta_u = {e: (min(1.0, ures[e] / s) if s > 0 else 0.0) for e, s in tot.items()}
            tot2 = 0.0
            for lt, yv in zip(coh, y):
                tot2 += eta_u[lt.next_edge] * yv
            eta_k = min(1.0, kap[b] / tot2) if tot2 > 0 else 0.0
            for lt, yv in zip(coh, y):
                q = eta_k * eta_u[lt.next_edge] * yv
                if q > 0:
                    out.append(Release(lt, lt.next_edge, lt.k, lt.lane, q, None))
                    ures[lt.next_edge] -= q
            kap[b] -= eta_k * tot2
    return out


def _overrides(
    inst: Instance,
    c: int,
    book: list[Lot],
    slots: list[int],
    overrides: dict[int, float],
    holds: frozenset[tuple[int, int]],
    prohibited,
    ures: list[float],
    kap: list[float],
    out: list[Release],
    clamps: list[tuple[int, float]] | None = None,
) -> None:
    """Step 2 at chokepoint c: override pieces in slot order, FIFO within k; updates ``ures``, ``kap`` and ``out``.

    (3)-(5) are the clip of the policy's requests (``clip.edge_stock_clip``) with the queue content of k at c, this
    week's arrivals included, as the stock, and its edge clamp on the out-edges whose float factor (4) is subnormal
    (Q95), logged in ``clamps`` (if given) as (override slot, value); (6) then scales each pool to the residual
    throughput. (6) and the fleet slack (7) only lower a slot's quantity, so the clamp's bound holds after them.
    """
    pool = inst.commodity_pool
    live = []  # (slot, k, edge, lane) of the entries that pass the mask (3)
    items = []
    for s in slots:
        _c, k, e, lane = inst.override_slots[s]
        q = overrides[s]
        if (c, k) not in holds and q > 0 and k in inst.edges[e].K and not prohibited[e][k]:  # (3)
            live.append((s, k, e, lane))
            items.append((e, k, (c, k), q))
    content: dict[tuple[int, int], float] = {}
    for key in {key for _e, _k, key, _q in items}:
        content[key] = sum(lt.qty for lt in book if lt.k == key[1])
    log: list[tuple[int, float]] = []
    clipped = edge_stock_clip(items, ures, content, log)  # (4) with the edge clamp (Q95), (5)
    if clamps is not None:
        clamps.extend((live[i][0], value) for i, value in log)
    by_b = [0.0, 0.0]
    for (_s, k, _e, _lane), x in zip(live, clipped):
        by_b[pool[k]] += x
    f6 = [min(1.0, max(0.0, kap[b]) / by_b[b]) if by_b[b] > 0 else 0.0 for b in (0, 1)]  # (6)
    req = [(s, k, e, lane, x * f6[pool[k]]) for (s, k, e, lane), x in zip(live, clipped)]
    left = {lt.lot_id: lt.qty for lt in book}
    fifo = sorted(book, key=fifo_key)
    for s, k, e, lane, take in req:
        if take <= 0:
            continue
        ures[e] -= take
        kap[pool[k]] -= take
        for lt in fifo:
            if take <= 0:
                break
            if lt.k != k or left[lt.lot_id] <= 0:
                continue
            piece = min(take, left[lt.lot_id])
            left[lt.lot_id] -= piece
            take -= piece
            out.append(Release(lt, e, k, lane, piece, s))

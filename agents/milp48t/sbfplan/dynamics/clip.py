"""The clip of the policy's requests, (3)-(5), and the fleet slack (7) (design §3.3; Q2, Q42, Q49, Q68 E3).

Requests are indexed by action slot (edge, commodity, lane). (3) masks slots whose commodity the edge does not permit or
whose pair is in Z_t; (4) caps each edge's total over its requests at the residual capacity u'_e; (5) scales every
(edge, commodity) total drawing on one stock (tail node, commodity) to that stock, and applies the per-(edge, commodity)
factors pro rata to every lane sub-request. Requests are processed in slot order, so the result is independent of the
order in which the policy listed them (F1 evidence ``f1_clip_order.py``). Each factor is at most 1.

Floating-point order follows the design's reference code (``scripts/python/evidence/tiny_fixture.py``, ``Sim.step``
step 4): sequential sums starting from 0.0, the edge factor ``min(1, u / total)``, the stock factor
``min(1, A / total)`` and the executed quantity ``(q * f_edge) * f_stock``; the fleet factor is applied as
``q * cap / total`` when the pool binds (7). When a sum of requests overflows the float range (requests near the float
maximum, which no reference run makes), (4)-(5) are computed in exact rational arithmetic and each executed quantity is
rounded once (``_exact_clip``); so is (7) for a pool whose total overflows or whose binding product ``q * cap`` leaves
the normal float range (``fleet_slack``; M1 pre-gate 3, DOC-1).

Edge clamp (Q95): with a request near the float maximum on an edge of small capacity, the edge factor of (4) is
subnormal and errs by up to 2^-1075 absolutely, so ``q * f_edge`` can put the edge's executed total above u'_e by far
more than 1e-12 relative (docs/004 §3.1: 1.1955e307 on E11 at 1200 x 2^-52 executed 2.664535259103041e-13 against
2.6645352591003757e-13). On an edge whose factor (4) is subnormal, ``clip_requests`` therefore takes the executed
quantities from u'_e in slot order under the stock clamp rule of ``sim._take`` (Q58 M11, design §12 'Stock clamps and
dust lots'): a residual left in [-b, 0), with b = 1e-12 max(u'_e, |x|, q, smallest normal float) and q the slot's
request (the draw from which its share of (4) was computed), sets that slot to the residual before it and is logged as
(slot, value); anything lower raises ``RuntimeError`` (``edge_clamp``). A normal factor errs by a few units in the last
place, as it always did (inside the tolerance of the replay (53)), and is left alone, so no run with a normal factor
moves; nor does the exact path, which rounds each quantity once. The override step at chokepoints
(``chokepoint._overrides``) computes its (4)-(5) with ``edge_stock_clip`` too and passes a log, so its override slots
pass the same clamp on an out-edge whose factor (4) is subnormal (a probe released 1.28e-3 relative above u'_e there
before), logged by override slot in ``StepRecord.override_clamps``.

The fleet slack (7) sums ``Delta tau * x`` over the terms of ``Instance.dup_items`` (edge, lane or None, Delta tau): a
term with lane None, an edge-level sea duplicate or the turn-back where a duplicate lane leaves its route at a
chokepoint, matches every flow on its edge; a term with a lane, a sea duplicate lane leaving its route at its entry
edge, matches only the flow that edge carries on that lane (design §3.3: "Cape, Lombok-Sunda and east-of-Taiwan lanes
and turn-back edges into them"). Every matching release, dispatch or chokepoint release, is scaled when its pool binds.

Residual edge capacity (general rule): u'_e = u^t_e minus the chokepoint releases on e this week. Action slots never
leave a chokepoint (§12, action slots), so on every instance u'_e = u^t_e for every edge that carries a request; the
reference computes its chokepoint residuals separately for the same reason.
"""

import math
import sys
from collections.abc import Container, Iterable, Mapping, Sequence
from fractions import Fraction

from sbfplan.instance.schema import Instance


CLAMP_TOL = 1e-12  # relative clamp band [-1e-12 max(A-bar, draw), 0) of Q58 M11, for stocks (sim) and edges (Q95)
FLOAT_MIN = sys.float_info.min  # the band's floor: below the smallest normal float, errors are absolute (2^-1074)


def edge_stock_clip(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    cap: Sequence[float],
    avail: Mapping[tuple[int, int], float],
    clamps: list[tuple[int, float]] | None = None,
) -> list[float]:
    """(4)-(5) for requests that already passed the mask (3): joint edge cap, then shared stock pro rata.

    Shared by the clip of the policy's requests and the override step at chokepoints (§3.4 step 2), so both apply the
    same arithmetic.

    Args:
        items: (edge, commodity, stock key, quantity) in processing order; the stock key identifies the stock A-bar the
            request draws on, (tail node, commodity).
        cap: residual capacity u'_e per edge.
        avail: A-bar per stock key.
        clamps: if given, the edges whose float factor (4) is subnormal pass the edge clamp (``edge_clamp``, Q95),
            whose entries (item index, value) are appended to it; if None, nothing is clamped.

    Returns:
        The executed quantity of each item, ``(q * f_edge) * f_stock``, after the edge clamp when ``clamps`` is given.

    Raises:
        RuntimeError: an edge residual below the clamp band (``edge_clamp``).

    """
    by_e: dict[int, float] = {}
    by_ek: dict[tuple[int, int], float] = {}
    for e, k, _key, q in items:
        by_e[e] = by_e.get(e, 0.0) + q
        by_ek[(e, k)] = by_ek.get((e, k), 0.0) + q
    if not all(math.isfinite(v) for v in by_e.values()):
        return _exact_clip(items, cap, avail)
    f_edge = {e: (min(1.0, max(0.0, cap[e]) / tot) if tot > 0 else 0.0) for e, tot in by_e.items()}  # (4)
    key_of = {(e, k): key for e, k, key, _q in items}
    by_key: dict[tuple[int, int], float] = {}
    for (e, k), tot in by_ek.items():
        key = key_of[(e, k)]
        by_key[key] = by_key.get(key, 0.0) + tot * f_edge[e]
    if not all(math.isfinite(v) for v in by_key.values()):
        return _exact_clip(items, cap, avail)
    f_stock = {key: (min(1.0, avail.get(key, 0.0) / tot) if tot > 0 else 0.0) for key, tot in by_key.items()}  # (5)
    x = [q * f_edge[e] * f_stock[key] for e, _k, key, q in items]
    if clamps is not None and (sub := {e for e, f in f_edge.items() if 0.0 < f < FLOAT_MIN}):
        x = edge_clamp(items, x, cap, sub, clamps)
    return x


def _exact_clip(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    cap: Sequence[float],
    avail: Mapping[tuple[int, int], float],
) -> list[float]:
    """(4)-(5) in exact rational arithmetic, for requests near the float maximum whose sums overflow.

    Every factor is the exact ``min(1, a / total)`` and every executed quantity ``q f_edge f_stock`` is rounded once
    to the nearest float, so the executed total on an edge or a stock exceeds its bound by at most half a unit in the
    last place per item. An infinite capacity never binds, nor does an infinite stock, as on the float path: a queue
    content of the override step whose float sum overflowed (lots near the float maximum; it raised ``OverflowError``
    here before, Q95). Scaling all quantities by a power of two instead rounded small (subnormal) stocks and requests,
    by up to 2^17 units in the last place (M1 pre-gate, INT-1 follow-up).
    """
    by_e: dict[int, Fraction] = {}
    by_ek: dict[tuple[int, int], Fraction] = {}
    for e, k, _key, q in items:
        by_e[e] = by_e.get(e, 0) + Fraction(q)
        by_ek[(e, k)] = by_ek.get((e, k), 0) + Fraction(q)
    f_edge: dict[int, Fraction] = {}
    for e, tot in by_e.items():  # (4)
        if tot <= 0:
            f_edge[e] = Fraction(0)
        elif cap[e] == math.inf:
            f_edge[e] = Fraction(1)
        else:
            f_edge[e] = min(Fraction(1), Fraction(max(0.0, cap[e])) / tot)
    key_of = {(e, k): key for e, k, key, _q in items}
    by_key: dict[tuple[int, int], Fraction] = {}
    for (e, k), tot in by_ek.items():
        key = key_of[(e, k)]
        by_key[key] = by_key.get(key, 0) + tot * f_edge[e]
    f_stock: dict[tuple[int, int], Fraction] = {}
    for key, tot in by_key.items():  # (5)
        a = avail.get(key, 0.0)
        if tot <= 0:
            f_stock[key] = Fraction(0)
        elif a == math.inf:
            f_stock[key] = Fraction(1)
        else:
            f_stock[key] = min(Fraction(1), Fraction(a) / tot)
    return [float(Fraction(q) * f_edge[e] * f_stock[key]) for e, _k, key, q in items]


def edge_clamp(
    items: Sequence[tuple[int, int, tuple[int, int], float]],
    executed: Sequence[float],
    cap: Sequence[float],
    edges: Container[int],
    clamps: list[tuple[int, float]],
) -> list[float]:
    """The edge clamp of (4) (Q95): on each edge in ``edges``, the executed quantities never overdraw u'_e.

    The rule of ``sim._take`` with u'_e in place of a stock: the residual starts at ``max(0, cap[e])`` and each item on
    the edge takes its executed quantity x from it in the given order; a result in [-b, 0), with
    b = ``1e-12 max(u'_e, |x|, q, smallest normal float)`` and q the item's request (the draw its share of (4) was
    computed from; a subnormal edge factor errs by up to ``q 2^-1075``), sets x to the residual before the take (so the
    residual becomes exactly 0) and is logged in ``clamps`` as (item index, value), value < 0. An infinite capacity
    never binds. Items on other edges, and every item where no residual goes negative, keep ``executed`` bit for bit.

    Args:
        items: (edge, commodity, stock key, request q) in processing order, as ``edge_stock_clip`` takes them.
        executed: the executed quantity of each item after (4)-(5).
        cap: residual capacity u'_e per edge.
        edges: the edges to clamp (``edge_stock_clip``: those whose float factor (4) is subnormal).
        clamps: list the clamps are appended to.

    Returns:
        The executed quantity of each item after the clamp.

    Raises:
        RuntimeError: a residual below the band (a clip bug; no valid request reaches it).

    """
    out = list(executed)
    resid: dict[int, float] = {}
    for i, ((e, _k, _key, q), x) in enumerate(zip(items, executed)):
        if e not in edges:
            continue
        bound = max(0.0, cap[e])
        r = resid.get(e, bound)
        new = r - x
        if new < 0.0:
            if new < -CLAMP_TOL * max(bound, abs(x), q, FLOAT_MIN):
                raise RuntimeError(f"edge {e} would be left at {new!r} of its capacity: more than the clamp band (Q95)")
            clamps.append((i, new))
            out[i], new = r, 0.0
        resid[e] = new
    return out


def clip_requests(
    inst: Instance,
    requests: Mapping[int, float],
    prohibited,
    cap: Sequence[float],
    avail: Sequence[float],
    clamps: list[tuple[int, float]] | None = None,
) -> dict[int, float]:
    """(3)-(5): mask, joint edge cap, shared stock pro rata over the lane sub-requests of each (edge, commodity).

    Edges whose factor (4) is subnormal then pass the edge clamp in slot order (``edge_clamp``, Q95).

    Args:
        inst: the instance.
        requests: action slot -> requested quantity (already validated, §9.3).
        prohibited: Z_t of the week, indexable as ``prohibited[e][k]`` (bool).
        cap: residual capacity u'_e per edge.
        avail: A-bar of (5) per stock slot: I^{t-1} at the tail nodes (never chokepoints, which have no action slot).
        clamps: if given, each edge clamp is appended to it as (action slot, value), value < 0, in slot order.

    Returns:
        Action slot -> executed quantity for every requested slot, in slot order (0 for masked or non-positive ones).

    Raises:
        RuntimeError: an edge residual below the clamp band (``edge_clamp``).

    """
    out: dict[int, float] = {}
    live: list[int] = []
    items: list[tuple[int, int, tuple[int, int], float]] = []
    for s in sorted(requests):
        q = requests[s]
        e, k, _lane = inst.action_slots[s]
        out[s] = 0.0
        if q > 0 and k in inst.edges[e].K and not prohibited[e][k]:  # (3)
            live.append(s)
            items.append((e, k, (inst.edges[e].tail, k), q))
    stock = {key: avail[inst.slot_index[key]] for _e, _k, key, _q in items if key in inst.slot_index}
    log: list[tuple[int, float]] = []
    for s, x in zip(live, edge_stock_clip(items, cap, stock, log)):
        out[s] = x
    if clamps is not None:
        clamps.extend((live[i], value) for i, value in log)
    return out


def fleet_caps(inst: Instance) -> tuple[float, float]:
    """s^b_fl F^b per pool (tb, ct), the right-hand side of the LP row of (7)."""
    return tuple(share * measure for share, measure in zip(inst.params.fleet_share, inst.params.fleet_measure))


def dup_terms(inst: Instance) -> Mapping[int, tuple[tuple[int | None, int], ...]]:
    """The fleet-slack terms of (7) by edge, from ``inst.dup_items``: edge -> ((lane or None, Delta tau), ...).

    A term with lane None (an edge-level sea duplicate or turn-back) matches every flow on its edge; a term with a lane
    (a sea duplicate lane leaving its route at its entry edge) matches only that lane's flow on it (design §3.3).
    """
    return inst.memo("dynamics.clip.dup_terms", _build_dup_terms)


def _build_dup_terms(inst: Instance) -> dict[int, tuple[tuple[int | None, int], ...]]:
    terms: dict[int, list[tuple[int | None, int]]] = {}
    for e, lane, dtau in inst.dup_items:
        terms.setdefault(e, []).append((lane, dtau))
    return {e: tuple(v) for e, v in terms.items()}


def on_dup(inst: Instance, e: int, lane: int | None) -> bool:
    """Whether a release on edge ``e`` carrying ``lane`` matches a fleet-slack term of (7), so the pool scales it."""
    return any(ln is None or ln == lane for ln, _dtau in dup_terms(inst).get(e, ()))


def fleet_totals(inst: Instance, items: Iterable[tuple[int, int, int | None, float]]) -> list[float]:
    """Sum of Delta tau x over the fleet-slack terms of (7) per pool, accumulated in the given order.

    Args:
        inst: the instance.
        items: (edge, commodity, lane or None, quantity) of every release of the week: dispatches first (slot order),
            then chokepoint releases (release order). An item adds ``Delta tau * q`` for every term of
            ``inst.dup_items`` it matches (``dup_terms``); items that match none are ignored.

    """
    terms = dup_terms(inst)
    tot = [0.0, 0.0]
    for e, k, lane, q in items:
        for ln, dtau in terms.get(e, ()):
            if ln is None or ln == lane:
                tot[inst.commodities[k].pool_index] += dtau * q
    return tot


def fleet_scaled(q: float, total: float, cap: float) -> float:
    """One duplicate release after (7): unchanged unless its pool binds, else ``q * cap / total`` (reference order)."""
    return q * cap / total if total > cap else q


def fleet_slack(
    inst: Instance, items: Sequence[tuple[int, int, int | None, float]], caps: Sequence[float]
) -> list[float]:
    """(7): every release of the week after the fleet slack, in the order given.

    Each pool sums ``Delta tau * q`` over the terms its releases match (``fleet_totals``, in the given order); when the
    total exceeds ``caps[b]`` = s^b_fl F^b, every matching release of the pool becomes ``q * cap / total`` in the
    reference order (``fleet_scaled``). A pool whose float total is not finite, or whose binding product ``q * cap``
    leaves the normal float range (overflows, or underflows from nonzero factors), is computed in exact rational
    arithmetic instead: the exact total, binding when it exceeds a finite cap, and each scaled release
    ``q * cap / total`` rounded once to the nearest float, so no release is ever scaled up. Floats cannot do either
    there: a total of inf executed 0, a product of inf an infinite shipment, and a subnormal product rounds up to twice
    ``q`` (M1 pre-gate 3, DOC-1, ORACLE-PRE3-2).

    Args:
        inst: the instance.
        items: (edge, commodity, lane or None, quantity) of every tentative release of the week: dispatches first (slot
            order), then chokepoint releases (release order), as ``fleet_totals`` takes them.
        caps: s^b_fl F^b per pool (``fleet_caps``).

    Returns:
        The quantity of each item after (7); items that match no term of ``inst.dup_items`` are unchanged.

    """
    terms = dup_terms(inst)
    totals = fleet_totals(inst, items)
    out = [q for _e, _k, _lane, q in items]
    for b, (total, cap) in enumerate(zip(totals, caps)):
        pool = inst.commodity_pool
        members = [i for i, (e, k, lane, _q) in enumerate(items) if pool[k] == b and on_dup(inst, e, lane)]
        if math.isfinite(total) and (total <= cap or all(_normal_product(out[i], cap) for i in members)):
            for i in members:  # the reference arithmetic
                out[i] = fleet_scaled(out[i], total, cap)
            continue
        exact = Fraction(0)  # the same terms as fleet_totals, summed exactly
        for i in members:
            e, _k, lane, q = items[i]
            exact += sum(dtau for ln, dtau in terms[e] if ln is None or ln == lane) * Fraction(q)
        if not math.isfinite(cap) or exact <= Fraction(cap):
            continue  # an infinite cap never binds, as in (4)
        for i in members:
            out[i] = float(Fraction(out[i]) * Fraction(cap) / exact)
    return out


def _normal_product(q: float, cap: float) -> bool:
    """Whether ``q * cap`` keeps the float relative accuracy: finite, and normal unless a factor is 0."""
    p = q * cap
    return math.isfinite(p) and (p >= sys.float_info.min or q == 0.0 or cap == 0.0)

"""The weekly simulator in the order of design §3.2 (V4), on one instance and one ``WeeklyMarks``.

Steps of week t: 3 forward at chokepoints (§3.4 pseudocode: arrivals join the lot book; overrides first; default release
by arrival cohort, FIFO across cohorts, pro rata inside a cohort (10); fleet slack last), 4 clip of the requests (3)-(7)
(mask, joint edge cap, shared stock pro rata over the lane sub-requests of each (edge, commodity), fleet slack per pool
over every release that matches a term of ``Instance.dup_items``, chokepoint releases included: an edge-level sea
duplicate or turn-back counts all its flow, a lane term the flow dispatched on its lane), 5 dispatch (2),
6 arrive, 7 produce (supply lift refilling to the cap (Q79); grid segments with gas rationing on fuel on hand (15)-(18)
under pri_g; fab lots (12) with gross WIP; scrap booked in onset weeks (14); OSAT packaging (19) up to
thr_i R_osat_i(t), the OSAT restoration of (13) under a regional conflict (§4.4)), 8 serve (20),
9 charge (disposal above I^max at non-chokepoint, non-supply nodes; C_t (23) and C^¢_t (24)). Step 10 reads the next
week's marks; no random draw happens anywhere.

Floating-point order: the design's reference code ``scripts/python/evidence/tiny_fixture.py`` (class ``Sim``) fixes the
operation order that reproduces the frozen integer cents of §2.4; this module follows it: dispatch draws each executed
slot from its stock in slot order; shipments are appended to the pipeline as dispatches (slot order) then chokepoint
releases (release order), and arrivals are added one shipment at a time in pipeline order; each lane queue Q_ckl is
recomputed from its lots in book order and the chokepoint stock is their fsum over lanes (``cost.queue_totals``, the
LP's formation, §12 'Cost terms (23)'); supply lifts, grids, fabs and OSATs run in instance order; the rationing
factor reads I^{t-1}, the available gas is ``min((zeta G-bar) ration, fuel on hand after steps 5-6)``, the segment
load factor is ``(sum E + y) / G-av``; holding is charged on end-of-week stock after disposal.

Rules the design leaves open, with the reading chosen here (conservative; design §12 'Phase-3 implementation notes'):

- Stock clamps (Q58 M11): a stock that a dispatch, a grid burn or packaging leaves in [-band, 0) is set to 0 and
  logged in ``StepRecord.clamps`` as (slot, value), with band = 1e-12 max(A-bar, |q|, draw, smallest normal float),
  q the quantity taken; A-bar is what bounded it (I^{t-1} for a dispatch, the segment's G^av for a burn, the raw stock
  for packaging) and the draw what the factor that scaled it was computed from (a dispatch's request capped at u'_e,
  the grid's requested load y-bar + sum E-hat of (18) for a burn, the OSAT throughput thr R_osat for packaging). With
  a subnormal A-bar that factor ((5), the served share of (18), (19)) is subnormal and errs by up to 2^-1075
  absolutely, so the draw, not A-bar, bounds the excess; below the smallest normal float every operation errs by up to
  2^-1074. Anything more negative is a bug and raises ``RuntimeError``; a valid action never does (M1 pre-gate
  ORACLE-PRE-1, INT-1, DET-1).
- Edge clamps (Q95): on an edge whose factor (4) is subnormal, the clip takes the executed requests from u'_e in slot
  order under the same rule (``clip.edge_clamp``): a residual left in [-band, 0), with band = 1e-12 max(u'_e, |x|, q,
  smallest normal float) and q the slot's request, lowers that slot to the residual before it and is logged in
  ``StepRecord.edge_clamps`` as (action slot, value); the subnormal factor errs by up to q 2^-1075, which the band
  bounds. The override step at chokepoints applies the same rule to the override slots on an out-edge whose factor (4)
  is subnormal, before (6) (``chokepoint._overrides``), and logs it apart, in ``StepRecord.override_clamps`` as
  (override slot, value), q the override request. Anything lower raises ``RuntimeError``. A week without an edge clamp
  is unchanged bit for bit, and so is its record's hash.
- Dust lots: every week, after the releases, every queue lot with 0 < qty <= 1e-12 leaves the book, whether or not
  anything was released from it (a held lot, a lot on a prohibited next edge or one that arrived this week alike), as
  the reference does (``tiny_fixture.py`` filters the whole book), and is logged in ``clamps`` with its (positive)
  quantity.
- Scrap (14) is booked for every hit in ``marks.fab_hits`` whose onset week is t, on the gross WIP of start weeks
  [t - w_scr, t - 1] (initial WIP included), before this week's output matures, so (12) holds with the same totals.
  Whether a carried-in event (onset <= 0) is a fab hit is the marks module's decision, not the simulator's.
- A hold and override slots on the same (c, k) in one week: the hold wins (nothing is released).
- Supply lift is ``max(0, min(avail, I^max - stock))``; the guard only matters if a supply node ever receives
  shipments.
- The rationing factor min{1, I^{t-1} / (psi I-bar_g)} of (15), (18) is 1 whenever I^{t-1} >= psi I-bar_g, else
  I^{t-1} / (psi I-bar_g): the LP's row psi I-bar_g G <= zeta G-bar I^{t-1} read as a factor, so psi I-bar_g = 0 (0/0
  included) does not ration, and for psi I-bar_g > 0 the float is the reference's ``min`` (M1 pre-gate DC-1).
"""

from dataclasses import dataclass

import numpy as np

from sbfplan.dynamics import chokepoint as chk_mod
from sbfplan.dynamics.clip import CLAMP_TOL, FLOAT_MIN, clip_requests, fleet_caps, fleet_slack
from sbfplan.dynamics.cost import queue_totals, salvage, weekly_costs
from sbfplan.dynamics.production import allocate_energy, package
from sbfplan.dynamics.state import Lot, Shipment, State, StepRecord, cents
from sbfplan.instance.schema import FabAttrs, Instance
from sbfplan.marks import WeeklyMarks, osat_throughput


LOT_EPS = 1e-12  # a queue lot at or below this quantity leaves the book (reference rule)


@dataclass(frozen=True)
class _Tables:
    """Week-invariant tables of one instance for ``step`` (kept in ``Instance.memo``)."""

    fleet_caps: tuple[float, float]  # s^b_fl F^b per pool (7)
    fab_attrs: tuple[FabAttrs, ...]  # per fab ordinal
    chokepoint_slots: tuple[tuple[int, int, int], ...]  # (slot, chokepoint node, k), slot order
    supply_slots: tuple[tuple[int, float], ...]  # (slot, I^max) at supply nodes, slot order
    disposal_slots: tuple[tuple[int, float], ...]  # (slot, I^max) at non-chokepoint, non-supply nodes, slot order

    @staticmethod
    def build(inst: Instance) -> "_Tables":
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        slots = list(enumerate(inst.stock_slots))
        return _Tables(
            fleet_caps=fleet_caps(inst),
            fab_attrs=tuple(inst.nodes[f].fab for f in inst.fabs),
            chokepoint_slots=tuple((s, sl.node, sl.k) for s, sl in slots if sl.node in chk),
            supply_slots=tuple((s, sl.storage) for s, sl in slots if sl.node in supply),
            disposal_slots=tuple((s, sl.storage) for s, sl in slots if sl.node not in chk and sl.node not in supply),
        )


def initial_stock(inst: Instance) -> np.ndarray:
    """I^0 per stock slot (§2.3), shared by the simulator and the oracle LP.

    Repeated initial-stock entries of a slot add up, in entry order. A non-chokepoint slot holds that sum; a chokepoint
    slot holds the fsum over lanes of its lane queues, each the lane's lots added in lot order (``lane_queues``, as at
    the end of every week), and its declared stock, the sum of its entries, must equal it within 1e-9 max(1, declared),
    the loader's check (§12 'Loader checks').

    Raises:
        ValueError: if a declared stock or a queue lot has no stock slot, or a declared chokepoint stock differs from
            its queue lots.

    """
    stock = np.zeros(len(inst.stock_slots))
    chk = set(inst.chokepoints)

    def slot(node: int, k: int) -> int:
        s = inst.slot_index.get((node, k))
        if s is None:
            raise ValueError(f"initial state: node {inst.nodes[node].id} has no stock of {inst.commodities[k].id}")
        return s

    for (c, k), q in queue_totals(lane_queues(inst.initial_state.queue_lots)).items():
        stock[slot(c, k)] = q
    declared: dict[int, float] = {}
    for node, k, qty in inst.initial_state.stock:
        s = slot(node, k)
        if node not in chk:
            stock[s] += qty
        else:
            declared[s] = declared.get(s, 0.0) + qty
    for s, qty in declared.items():
        if abs(stock[s] - qty) > 1e-9 * max(1.0, qty):
            node = inst.nodes[inst.stock_slots[s].node].id
            raise ValueError(f"initial stock at {node} ({qty!r}) differs from its queue lots ({float(stock[s])!r})")
    return stock


def initial_state(inst: Instance) -> State:
    """The declared initial state (§2.3): stock, pipeline, fab and OSAT WIP (gross), queue lots, zero backlog."""
    ist = inst.initial_state
    lots = [
        Lot(i, q.chokepoint, q.k, q.qty, q.lane, q.next_edge, q.dispatch_week, q.entry_edge, q.arrival_week)
        for i, q in enumerate(ist.queue_lots)
    ]
    next_id = len(lots)
    pipeline = []
    for s in ist.pipeline:  # initial shipments into a chokepoint get their lot ids at reset, in pipeline order
        into_chk = inst.edges[s.edge].head in inst.chokepoint_ordinal
        pipeline.append(
            Shipment(s.edge, s.k, s.lane, s.qty, s.dispatch_week, s.arrival_week, next_id if into_chk else None)
        )
        next_id += into_chk
    fab_wip: dict[int, dict[int, float]] = {fi: {} for fi in range(len(inst.fabs))}
    for w in ist.fab_wip:
        fi = inst.fab_ordinal[w.node]
        start = w.out_week - inst.nodes[w.node].fab.tau
        fab_wip[fi][start] = fab_wip[fi].get(start, 0.0) + w.qty
    osat_wip: dict[int, dict[int, dict[int, float]]] = {oi: {} for oi in range(len(inst.osats))}
    for w in ist.osat_wip:
        book = osat_wip[inst.osat_ordinal[w.node]].setdefault(w.out_week, {})
        book[w.k] = book.get(w.k, 0.0) + w.qty
    return State(
        week=0,
        stock=initial_stock(inst),
        pipeline=pipeline,
        lots=lots,
        fab_wip=fab_wip,
        osat_wip=osat_wip,
        backlog=np.zeros(len(inst.demands)),
        next_lot_id=next_id,
        last=None,
    )


def lane_queues(lots) -> dict[tuple[int, int, int], float]:
    """Q_ckl of (52): (chokepoint, commodity, lane) -> the lane's lots added in book order, from 0.0.

    ``StepRecord.queue`` records these values, and the chokepoint stock is ``cost.queue_totals`` of them.
    """
    queue: dict[tuple[int, int, int], float] = {}
    for lt in lots:
        key = (lt.chokepoint, lt.k, lt.lane)
        queue[key] = queue.get(key, 0.0) + lt.qty
    return queue


def _new_lot_id(inst: Instance, state: State, e: int) -> int | None:
    """The next lot id for a shipment dispatched on edge e into a chokepoint (§3.4 step 1: at dispatch), else None."""
    if inst.edges[e].head not in inst.chokepoint_ordinal:
        return None
    state.next_lot_id += 1
    return state.next_lot_id - 1


def _take(stock: list[float], s: int, q: float, avail: float, clamps: list, draw: float = 0.0) -> None:
    """stock[s] -= q with the clamp rule of Q58 M11: a result in [-band, 0) becomes 0 and is logged in ``clamps``.

    The band is ``1e-12 max(A-bar, |q|, draw, smallest normal float)`` with A-bar = ``avail``. ``draw`` is what the
    factor that scaled ``q`` was computed from (a dispatch's request capped at the edge's capacity, a grid's requested
    load, an OSAT's throughput): when that factor is subnormal its error is absolute, so ``q`` errs by at most
    ``draw`` 2^-1075, which 1e-12 ``draw`` bounds; below the smallest normal float every operation errs by up to
    2^-1074, so the band never falls below 1e-12 of it (M1 pre-gate ORACLE-PRE-1, INT-1, DET-1). A result below the
    band is a simulator bug.

    Raises:
        RuntimeError: the result is below the band.

    """
    new = stock[s] - q
    if new < 0.0:
        if new < -CLAMP_TOL * max(avail, abs(q), draw, FLOAT_MIN):
            raise RuntimeError(f"stock slot {s} would go to {new!r}: more than the clamp band (Q58 M11)")
        clamps.append((s, new))
        new = 0.0
    stock[s] = new


def step(
    inst: Instance,
    marks: WeeklyMarks,
    state: State,
    flows: dict[int, float],
    overrides: dict[int, float] | None = None,
    holds: frozenset[tuple[int, int]] = frozenset(),
    invalid: tuple[str, ...] = (),
) -> StepRecord:
    """Simulate week ``state.week + 1`` in place and return its record.

    Args:
        inst: the instance.
        marks: the episode's weekly marks.
        state: the state at the end of the previous week; mutated to the end of this week.
        flows: action slot -> requested quantity, already validated (§9.3): finite, >= 0, in range, not masked.
        overrides: override slot -> quantity for tanker cargo K^ov. Any valid override slot or hold for (c, k) turns the
            default release of k at c off this week; a qty-0 slot releases nothing on that slot (Q86).
        holds: (chokepoint node, k) pairs held this week (no release at all).
        invalid: entries the validator dropped, copied into the record's log.

    """
    t = state.week + 1
    if not 1 <= t <= inst.T or marks.T != inst.T:
        raise ValueError(f"week {t} is outside the horizon 1..{inst.T}")
    ti = t - 1
    overrides = {int(s): float(q) for s, q in (overrides or {}).items()}
    flows = {int(s): float(q) for s, q in flows.items()}
    S, F, G, D = len(inst.stock_slots), len(inst.fabs), len(inst.grids), len(inst.demands)
    E = len(inst.edges)
    slot_index = inst.slot_index
    tables: _Tables = inst.memo("dynamics.sim", _Tables.build)
    I = [float(x) for x in state.stock]
    I_prev = list(I)
    u = marks.u[ti].tolist()
    prohibited = marks.prohibited[ti].tolist()
    clamps: list[tuple[int, float]] = []
    edge_clamps: list[tuple[int, float]] = []
    override_clamps: list[tuple[int, float]] = []

    # ----- 3 forward: arrivals join the lot book; overrides first (edge clamp of (4)); default release (10) --------
    chk_mod.arrivals_to_lots(inst, state, t)
    rel = chk_mod.release(inst, state.lots, u, marks.kappa[ti].tolist(), prohibited, overrides, holds, override_clamps)

    # ----- 4 clip (3)-(5), edge clamp on the residual u' (no request edge leaves a chokepoint), fleet slack (7) ----
    released_on: dict[int, float] = {}
    for r in rel:
        released_on[r.edge] = released_on.get(r.edge, 0.0) + r.qty
    cap = [u[e] - released_on[e] if e in released_on else u[e] for e in range(E)]
    x_req = clip_requests(inst, flows, prohibited, cap, I_prev, edge_clamps)
    items = [(*inst.action_slots[s], q) for s, q in x_req.items()]  # (edge, commodity, lane, quantity)
    after = fleet_slack(inst, items + [(r.edge, r.k, r.lane, r.qty) for r in rel], tables.fleet_caps)  # (7)
    for s, q in zip(list(x_req), after[: len(items)]):
        x_req[s] = q
    for r, q in zip(rel, after[len(items) :]):
        r.qty = q

    # ----- 5 dispatch (2): executed requests leave stock, then chokepoint releases leave their lots -----------------
    x_lane: dict[tuple[int, int, int | None], float] = {}
    for s, q in x_req.items():
        if q <= 0:
            continue
        e, k, lane = inst.action_slots[s]
        edge = inst.edges[e]
        tail_slot = slot_index[(edge.tail, k)]
        _take(I, tail_slot, q, I_prev[tail_slot], clamps, min(flows[s], cap[e]))  # the draw: request capped by (4)
        state.pipeline.append(Shipment(e, k, lane, q, t, t + edge.tau, _new_lot_id(inst, state, e)))
        x_lane[(e, k, lane)] = x_lane.get((e, k, lane), 0.0) + q
    ov_exec = {s: 0.0 for s in overrides}
    for r in rel:
        if r.qty <= 0:
            continue
        r.lot.qty -= r.qty
        lid = _new_lot_id(inst, state, r.edge)  # a release onto a lane edge into the next chokepoint (tandem lanes)
        state.pipeline.append(Shipment(r.edge, r.k, r.lane, r.qty, t, t + inst.edges[r.edge].tau, lid))
        x_lane[(r.edge, r.k, r.lane)] = x_lane.get((r.edge, r.k, r.lane), 0.0) + r.qty
        if r.override_slot is not None:
            ov_exec[r.override_slot] += r.qty
    kept = []
    for lt in state.lots:
        if lt.qty > LOT_EPS:
            kept.append(lt)
        elif lt.qty != 0.0:
            clamps.append((slot_index[(lt.chokepoint, lt.k)], lt.qty))
    state.lots = kept
    queue = lane_queues(state.lots)  # Q^t_ckl (52); no lot moves again this week (arrivals at step 6 skip chokepoints)
    chk_stock = queue_totals(queue)  # I^t_ck = fsum over lanes of Q^t_ckl, as the LP forms it (23)
    for s, c, k in tables.chokepoint_slots:
        I[s] = chk_stock.get((c, k), 0.0)

    # ----- 6 arrive at every other node, one shipment at a time in pipeline order ----------------------------------
    keep = []
    for sh in state.pipeline:
        if sh.arrival_week == t:
            I[slot_index[(inst.edges[sh.edge].head, sh.k)]] += sh.qty  # chokepoint arrivals left at step 3 (tau >= 1)
        else:
            keep.append(sh)
    state.pipeline = keep

    # ----- 7 produce ---------------------------------------------------------------------------------------------
    supply = marks.supply[ti].tolist()
    lift = [0.0] * S
    for s, storage in tables.supply_slots:  # refill up to the cap (Q79); the rest of the availability is lost
        q = max(0.0, min(supply[s], storage - I[s]))
        lift[s] = q
        I[s] += q
    alpha, R = marks.alpha_bar[ti].tolist(), marks.R[ti].tolist()
    fab_attrs = tables.fab_attrs
    phat = [
        min(alpha[fi] * R[fi] * fab.cap0, I[slot_index[(f, fab.input)]])
        for fi, (f, fab) in enumerate(zip(inst.fabs, fab_attrs))
    ]
    p = list(phat)
    energy = [0.0] * F
    served_load, shed = [0.0] * G, [0.0] * G
    segment: dict[tuple[int, int | None], float] = {}
    G_bar, y_bar = marks.G_bar[ti].tolist(), marks.y_bar[ti].tolist()
    psi = inst.params.psi
    for gi, g in enumerate(inst.grids):
        grid = inst.nodes[g].grid
        av: dict[int, float] = {}
        for k in grid.fuels:  # (15), (18): segment output after rationing, bounded by fuel on hand after steps 5-6
            s = slot_index[(g, k)]
            if k == grid.rationed:  # min{1, I^{t-1} / (psi I-bar)} as the LP's row psi I-bar G <= zeta G-bar I^{t-1}
                threshold = psi * grid.ibar[k]
                ration = 1.0 if I_prev[s] >= threshold else I_prev[s] / threshold  # psi I-bar = 0 never rations
                av[k] = min(grid.shares[k] * G_bar[gi] * ration, I[s])
            else:
                av[k] = min(grid.shares[k] * G_bar[gi], I[s])
        av_null = grid.shares.get(None, 0.0) * G_bar[gi]
        g_av = 0.0
        for k in grid.fuels:
            g_av += av[k]
        g_av += av_null
        members = inst.grid_fabs[gi]
        e_hat = [(fab_attrs[fi].e * phat[fi] / R[fi] if R[fi] > 0 else 0.0) for fi in members]
        y, E_alloc = allocate_energy(grid.priority, g_av, y_bar[gi], e_hat)
        for fi, ef in zip(members, E_alloc):
            energy[fi] = ef
            if fab_attrs[fi].e > 0:
                p[fi] = min(phat[fi], R[fi] * ef / fab_attrs[fi].e)
        load = (sum(E_alloc) + y) / g_av if g_av > 0 else 0.0
        for k in grid.fuels:
            s = slot_index[(g, k)]
            burn = av[k] * load
            segment[(gi, k)] = burn
            _take(I, s, burn, av[k], clamps, y_bar[gi] + sum(e_hat))  # the draw: the grid's requested load (18)
        segment[(gi, None)] = av_null * load
        served_load[gi] = y
        shed[gi] = y_bar[gi] - y
    scrapped = [0.0] * F
    for fi, (f, fab) in enumerate(zip(inst.fabs, fab_attrs)):
        I[slot_index[(f, fab.input)]] -= p[fi]  # p <= wafers on hand
        wip = state.fab_wip.setdefault(fi, {})
        wip[t] = wip.get(t, 0.0) + p[fi]  # (12): gross WIP booked at start
        for hit in marks.fab_hits:  # (14): scrap of this week's onsets on start weeks [t - w_scr, t - 1]
            if hit.fab == fi and hit.onset_week == t:
                for s0 in range(t - fab.w_scr, t):
                    if s0 in wip:
                        old = wip[s0]
                        wip[s0] = old * (1.0 - hit.severity)
                        scrapped[fi] += old - wip[s0]
        I[slot_index[(f, fab.product)]] += wip.pop(t - fab.tau, 0.0)
    packaged: dict[tuple[int, int], float] = {}
    osat_cap = osat_throughput(inst, marks.R_osat[ti]).tolist()  # (19) under (13): thr_i R_osat_i(t), one home
    for oi, o in enumerate(inst.osats):
        osat = inst.nodes[o].osat
        wip = state.osat_wip.setdefault(oi, {})
        for k, q in sorted(wip.pop(t, {}).items()):  # packaged output joins stock before xi is computed (Q58 M6)
            I[slot_index[(o, k)]] += q
        pairs = sorted(osat.packages.items())
        raw_slots = [slot_index[(o, raw)] for raw, _pk in pairs]
        thr = osat_cap[oi]  # (19) under (13): thr_i R_osat_i(t), the LP's product (§4.4; thr x 1.0 = thr)
        xi = package([I[s] for s in raw_slots], thr)
        book = wip.setdefault(t + osat.tau, {})
        for (_raw, pk), s, q in zip(pairs, raw_slots, xi):
            _take(I, s, q, I[s], clamps, thr)  # A-bar the raw stock; (19) computes xi from thr R_osat
            book[pk] = book.get(pk, 0.0) + q
            packaged[(oi, pk)] = q

    # ----- 8 serve (20) ------------------------------------------------------------------------------------------
    demand = marks.demand[ti].tolist()
    served, lost, backlog = [0.0] * D, [0.0] * D, [0.0] * D
    for di, d in enumerate(inst.demands):
        s = slot_index[(d.node, d.k)]
        want = demand[di] + float(state.backlog[di]) if d.backlog else demand[di]
        served[di] = min(want, I[s])
        I[s] -= served[di]
        if d.backlog:
            backlog[di] = want - served[di]
        else:
            lost[di] = demand[di] - served[di]

    # ----- 9 charge: disposal above I^max at non-chokepoint, non-supply nodes; C_t (23), C^¢_t (24) ------------------
    disposal = [0.0] * S
    for s, storage in tables.disposal_slots:
        if I[s] > storage:
            disposal[s] = I[s] - storage
            I[s] = storage
    costs = weekly_costs(
        inst,
        marks.c[ti].tolist(),
        marks.c_wr[ti].tolist(),
        marks.tariff[ti].tolist(),
        marks.h_queue[ti].tolist(),
        x_lane,  # x_ek = edge_flows(x_lane) (23)
        I,
        disposal,
        lost,
        backlog,
        shed,
    )
    rec = StepRecord(
        week=t,
        requested=dict(sorted(flows.items())),
        executed=dict(x_req),
        override_requested=dict(sorted(overrides.items())),
        override_executed=dict(sorted(ov_exec.items())),
        x=x_lane,
        stock=np.array(I),
        queue=queue,
        disposal=np.array(disposal),
        lift=np.array(lift),
        lots_started=np.array(p),
        scrapped=np.array(scrapped),
        packaged=packaged,
        segment=segment,
        energy=np.array(energy),
        served_load=np.array(served_load),
        shed=np.array(shed),
        demand=np.array(demand),
        served=np.array(served),
        lost=np.array(lost),
        backlog=np.array(backlog),
        costs=costs,
        cost_cents=cents(costs.total()),
        clamps=tuple(clamps),
        invalid=tuple(invalid),
        edge_clamps=tuple(edge_clamps),
        override_clamps=tuple(override_clamps),
    )
    state.stock = rec.stock.copy()
    state.backlog = rec.backlog.copy()
    state.week = t
    state.last = rec
    return rec


def terminal_salvage(inst: Instance, state: State) -> float:
    """Terminal credit S_T of (23) at the end of week T.

    Stock at nu (0 at supply nodes, Q79; a chokepoint's queue at its slot's nu, Q93), shipments in transit at the nu of
    their edge's head, per (edge, commodity, dispatch week) on the week's x_ek (Q93), and fab and OSAT WIP at the
    salvage of their input net of scrap booked by T (Q56).
    """
    if state.week != inst.T:
        raise ValueError(f"the terminal credit is taken at T = {inst.T}, not at week {state.week}")
    return salvage(inst, state)

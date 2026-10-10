"""`greedy_lp`: a per-week min-cost flow toward naive's deterministic cover (milestone M4, stream "lp-baselines").

Design §8.2 table and "Gate baselines"; Q8, Q84, Q100 (3).

Each week one small LP over the edges observed open, not (51): minimise freight c^t_e (week t's, observed), war-risk
cost c^wr and tariff rate x v_k over each route's full path, plus a penalty times the shortfall of each destination's
inventory position below naive's deterministic cover sum_l (tau_l + 1 + o_l) d_lk (plus I-bar_g at grids): pi of the
packaged chip at sinks, fabs and OSATs (one wafer makes one chip) and VOLL_g per GWh at grids (one GWh of fuel delivers
one GWh, Q76) (§8.2, a proposal). Readings (design §12 M4 rows "greedy_lp" and "`greedy_lp` as built"):

- destinations, routes, route demand d_lk (m^sea(t) into sinks, ``naive_parts.seasonal``), transits, offsets and the
  step-4 inventory position are naive's plan (``naive_parts.cached_plan``); the cover is ``naive.level(d, tau, None, (),
  offset)`` per route (no closure term, no cap) summed by ``naive_parts.destination_targets``;
- rows: the clip's (4) edge capacity on the observed u, (5) stock on hand I^{t-1} at each tail and (7) the fleet
  slack, so executed equals requested when the observation equals the week's marks; and (M5, design §12 M5 row
  "`greedy_lp`'s lane rows") (4) on the observed u of every lane edge after the first, and the throughput (6) of each
  chokepoint and pool a candidate lane passes on the observed kappa_cb less the content queued there in the pool, so
  the week's dispatch into a lane never exceeds what its later edges and chokepoints carry in a week (``_lane_rows``);
- a route counts as open by naive's step-6 test (``naive_parts.observed_closed`` at 0.6); an ``alt_of`` duplicate is
  used only within ``give_up_ratio`` x the route's base freight (the 2x give-up rule); a slot flagged in ``slot_mask``
  is closed too, which since Q111 (the whole-route mask) adds nothing to the step-6 test: a slot masked for a later
  edge of its lane is on a route that test already calls closed;
- the end-aware cut (Q100 (3)): no flow on an action slot s in a week t > W_s of (72) (``naive.last_useful_weeks``),
  and the cover less the cut's phantom pipeline, the target S^end of (72) (``naive.end_aware_targets``; Q109), so
  ``greedy_lp`` differs from the anchor by its named idea only;
- its own highspy model through ``lp_common`` (built at reset, structure fixed; costs, bounds and right-hand sides
  re-passed each week, the previous basis reused as is), the status ladder, and naive's action from the same
  observation when every rung fails (counted in ``telemetry``).
"""

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from shockbench_flow.dynamics.clip import dup_terms, fleet_caps
from shockbench_flow.instance.schema import POOLS, Instance
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.base import StepTelemetry, params_run_name
from shockbench_flow.policies.naive import end_aware_targets, last_useful_weeks, level
from shockbench_flow.policies.naive_derived import ThresholdParams, step4_books, step4_ip
from shockbench_flow.policies.naive_parts import (
    cached_plan,
    destination,
    destination_targets,
    observed_closed,
    route_edges,
    seasonal,
)
from shockbench_flow.policies.registry import PolicyContext


@dataclass(frozen=True)
class GreedyLPParams(ThresholdParams):
    """``greedy_lp``'s parameters: naive's step-6 thresholds (the 0.6 cut is SYNTHETIC, §11 row 83).

    Raises:
        ValueError: as ``naive.check_thresholds``.

    """


def penalty(inst: Instance, j: int, k: int) -> float:
    """The shortfall penalty of destination (j, k) (§8.2 "Gate baselines"; design §12 "`greedy_lp` as built").

    pi of the sink demand at a sink; at a fab (its wafers) or an OSAT (a raw chip it packages) the largest pi over the
    sinks demanding the packaged chip made (one wafer makes one chip); VOLL_g at a grid (one GWh of fuel delivers one
    GWh, Q76); at a terminal the largest VOLL of the grids its out-edges feed: the penalties behind naive's critical
    ratio (``naive_fq.critical_ratio``).

    Raises:
        ValueError: if the destination's type has no penalty rule, or no sink demands what it makes.

    """
    node = inst.nodes[j]

    def packaged(chips: set[int]) -> float:
        pis = [d.pi for d in inst.demands if d.k in chips]
        if not pis:
            raise ValueError(f"no sink demands the packaged chip {node.id} makes")
        return max(pis)

    if node.type == "sink":
        pis = [d.pi for d in inst.demands if d.node == j and d.k == k]
        if pis:
            return pis[0]
    elif node.grid is not None and k in node.grid.fuels:
        return node.grid.voll
    elif node.fab is not None and k == node.fab.input:
        raw = node.fab.product
        return packaged({inst.nodes[o].osat.packages[raw] for o in inst.osats if raw in inst.nodes[o].osat.packages})
    elif node.osat is not None and k in node.osat.packages:
        return packaged({node.osat.packages[k]})
    elif node.type == "terminal":
        volls = [
            inst.nodes[inst.edges[e].head].grid.voll for e in inst.out_edges[j] if inst.nodes[inst.edges[e].head].grid
        ]
        if volls:
            return max(volls)
    raise ValueError(f"greedy_lp has no shortfall penalty for {inst.commodities[k].id} into {node.type} {node.id}")


def inventory_position(inst: Instance, obs: dict) -> dict[tuple[int, int], float]:
    """Naive's step-4 inventory position of every (j, k): on hand + bound pipeline + bound queued lots - backlog.

    Summed in naive's order (``naive.naive_action``): on hand, then the pipeline shipments whose route ends at j, then
    the queued lots whose lane ends at j, less the backlog (``naive_derived.step4_books`` and ``step4_ip``).
    """
    books = step4_books(inst, obs, strict=True)
    keys = set(books.on_hand) | set(books.backlog) | set(books.in_transit) | set(books.queued)
    return {key: step4_ip(books, key) for key in keys}


class GreedyLP:
    """`greedy_lp` as a ``Policy``; ``telemetry`` holds one ``StepTelemetry`` per week of the current episode."""

    name = "greedy_lp"

    def __init__(self, params: GreedyLPParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = GreedyLPParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = params_run_name(type(self).name, self.params)  # its parameters in its name (SPEC-M4R2-01)
        self.telemetry: list[StepTelemetry] = []

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Load the instance and naive's plan (strict provenance), the last useful weeks, and the one-week model."""
        self._inst, self._plan = cached_plan(static, self.context.fq_quantile)
        inst, plan = self._inst, self._plan
        self._last = last_useful_weeks(inst, plan)
        self._fallback = L.internal_fallback(self.context, static, obs, policy_seed)
        self._memory = L.ObservedGraph.nominal(inst)
        self._session = L.LPSession()
        self.telemetry = []
        slot_of = inst.action_slot_index
        roles: dict[int, list[tuple[int, bool]]] = defaultdict(list)  # slot -> (route index, as duplicate)
        for ri, r in enumerate(plan.routes):
            roles[slot_of[(r.first_edge, r.k, r.lane)]].append((ri, False))
            if r.alt_edge is not None:
                roles[slot_of[(r.alt_edge, r.k, r.alt_lane)]].append((ri, True))
        self._slots = sorted(roles)
        self._roles = {s: tuple(v) for s, v in roles.items()}
        self._dests = list(plan.groups)
        ns, nd = len(self._slots), len(self._dests)
        dest_row = {d: i for i, d in enumerate(self._dests)}
        rows, cols, vals = [], [], []
        self._slot_dest = []
        for j, s in enumerate(self._slots):
            e, k, lane = inst.action_slots[s]
            d = (destination(inst, e, lane), k)
            if d not in dest_row:
                raise ValueError(f"greedy_lp: slot {s} delivers to {d}, which naive's plan does not feed")
            self._slot_dest.append(d)
            rows.append(dest_row[d])
            cols.append(j)
            vals.append(1.0)
        for i in range(nd):  # the shortfall column of each destination
            rows.append(i)
            cols.append(ns + i)
            vals.append(1.0)
        r0 = nd
        edges = sorted({inst.action_slots[s][0] for s in self._slots})
        self._edge_rows = edges
        for i, e in enumerate(edges):  # (4)
            for j, s in enumerate(self._slots):
                if inst.action_slots[s][0] == e:
                    rows.append(r0 + i)
                    cols.append(j)
                    vals.append(1.0)
        r0 += len(edges)
        tails = sorted({(inst.edges[inst.action_slots[s][0]].tail, inst.action_slots[s][1]) for s in self._slots})
        self._tail_rows = tails
        for i, tk in enumerate(tails):  # (5)
            for j, s in enumerate(self._slots):
                e, k, _lane = inst.action_slots[s]
                if (inst.edges[e].tail, k) == tk:
                    rows.append(r0 + i)
                    cols.append(j)
                    vals.append(1.0)
        r0 += len(tails)
        terms = dup_terms(inst)
        weights: dict[int, dict[int, float]] = defaultdict(dict)  # pool -> column -> Delta tau (7)
        for j, s in enumerate(self._slots):
            e, k, lane = inst.action_slots[s]
            w = sum(dtau for ln, dtau in terms.get(e, ()) if ln is None or ln == lane)
            if w:
                weights[inst.commodity_pool[k]][j] = float(w)
        caps = fleet_caps(inst)
        self._fleet = [(b, caps[b]) for b in sorted(weights)]
        for i, (b, _cap) in enumerate(self._fleet):
            for j, w in sorted(weights[b].items()):
                rows.append(r0 + i)
                cols.append(j)
                vals.append(w)
        r0 += len(self._fleet)
        self._lane_edge_rows, self._through_rows = self._lane_rows(inst)
        for i, (_row, members) in enumerate((*self._lane_edge_rows, *self._through_rows)):  # (4) lane edges, then (6)
            for j in members:
                rows.append(r0 + i)
                cols.append(j)
                vals.append(1.0)
        r0 += len(self._lane_edge_rows) + len(self._through_rows)
        self._A = sp.csc_matrix((vals, (rows, cols)), shape=(r0, ns + nd))
        self._penalty = np.array([penalty(inst, j, k) for j, k in self._dests], dtype=np.float64)

    def _lane_rows(self, inst: Instance) -> tuple[tuple, tuple]:
        """The M5 rows over the candidate lane slots (design §12 M5 row "`greedy_lp`'s lane rows").

        Returns (lane-edge rows, throughput rows): (4) per edge a candidate lane passes after its first edge, as (edge,
        the columns whose route passes it), in edge order; (6) per chokepoint c and pool b some candidate lane of b
        passes, as ((c, pool index), the columns of those slots), in (node, pool) order. A lane's first edge keeps its
        (4) row of the M4 build; a column enters each row once, in column order.
        """
        firsts = set(self._edge_rows)
        lane_edges: dict[int, set[int]] = defaultdict(set)
        through: dict[tuple[int, int], set[int]] = defaultdict(set)
        for j, s in enumerate(self._slots):
            _e, k, lane = inst.action_slots[s]
            if lane is None:
                continue
            ln = inst.lanes[lane]
            for x in ln.edges[1:]:
                lane_edges[x].add(j)
            for c in ln.chokepoints:
                through[(c, POOLS.index(inst.commodities[k].pool))].add(j)
        for x in lane_edges:  # a later lane edge that is also a first edge (none on the built instances) counts both
            if x in firsts:
                lane_edges[x] |= {j for j, s in enumerate(self._slots) if inst.action_slots[s][0] == x}
        return (
            tuple((x, tuple(sorted(lane_edges[x]))) for x in sorted(lane_edges)),
            tuple((cb, tuple(sorted(through[cb]))) for cb in sorted(through)),
        )

    def _queued(self, obs: dict) -> dict[tuple[int, int], float]:
        """The content queued at each chokepoint per pool index at t - 1, summed over ``queue_lots`` in book order."""
        inst = self._inst
        out: dict[tuple[int, int], float] = defaultdict(float)
        ql = obs["queue_lots"]
        for c, k, q in zip(ql["chokepoint"], ql["k"], ql["qty"], strict=True):
            out[(int(c), POOLS.index(inst.commodities[int(k)].pool))] += q
        return out

    # ----- one week -------------------------------------------------------------------------------------------------
    def _slot_allowed(self, s: int, t: int, obs: dict, g: dict, prohibited: set) -> bool:
        inst, plan, p = self._inst, self._plan, self.params
        if not self._last.useful(s, t):  # the end-aware cut (72)
            return False
        mask = obs.get("slot_mask")
        e, k, lane = inst.action_slots[s]
        if mask is not None:
            if mask[s]:
                return False
        elif self._memory.values["prohibited"][e, k]:
            return False
        for ri, dup in self._roles[s]:
            r = plan.routes[ri]
            if not dup:
                if not observed_closed(inst, r.edges, r.lane, r.k, g, prohibited, p.closed_threshold):
                    return True
            elif r.alt_cost <= p.give_up_ratio * r.base_cost and not observed_closed(
                inst, route_edges(inst, r.alt_edge, r.alt_lane), r.alt_lane, r.k, g, prohibited, p.closed_threshold
            ):
                return True
        return False

    def _model(self, obs: dict):
        """Week t's costs, bounds and right-hand sides on the fixed structure, as a ``HighsLp``; and the open slots."""
        inst, plan, t = self._inst, self._plan, int(obs["week"])
        v = self._memory.values
        g = obs.get("graph_now") or {}
        pr = g.get("prohibited") or {"edge": [], "k": []}
        prohibited = set(zip(pr["edge"], pr["k"], strict=True))
        _hq, cwr = L.window_queue_and_transit(inst, v["war_risk"][None, :])
        cwr = cwr[0]
        ns, nd = len(self._slots), len(self._dests)
        cost = np.zeros(ns + nd)
        ub = np.zeros(ns + nd)
        allowed = []
        for j, s in enumerate(self._slots):
            e, k, lane = inst.action_slots[s]
            path = route_edges(inst, e, lane)
            cost[j] = sum(v["c"][x] + cwr[x, k] + v["tariff"][x, k] * inst.commodities[k].v for x in path)
            if self._slot_allowed(s, t, obs, g, prohibited):
                ub[j] = np.inf
                allowed.append(j)
        cost[ns:] = self._penalty
        ub[ns:] = np.inf
        m = seasonal(inst, t)
        ds = [r.d * m.get((r.dest, r.k), 1.0) for r in plan.routes]
        cover = destination_targets(
            inst, plan.routes, [level(d, r.tau, None, (), r.offset) for r, d in zip(plan.routes, ds)]
        )
        cover = end_aware_targets(inst, plan, self._last, t, cover)  # less the cut's phantom pipeline (72), Q109
        books = step4_books(inst, obs, strict=True)
        queued = self._queued(obs)
        n_lane = len(self._lane_edge_rows) + len(self._through_rows)
        row_lb = np.concatenate(
            [
                [cover[d] - step4_ip(books, d) for d in self._dests],
                np.full(len(self._edge_rows) + len(self._tail_rows) + len(self._fleet) + n_lane, -np.inf),
            ]
        )
        row_ub = np.concatenate(
            [
                np.full(nd, np.inf),
                [v["u"][e] for e in self._edge_rows],
                [max(books.on_hand.get(tk, 0.0), 0.0) for tk in self._tail_rows],
                [cap for _b, cap in self._fleet],
                [v["u"][x] for x, _members in self._lane_edge_rows],  # (4) on the observed u of each later lane edge
                [  # (6): the observed kappa_cb less the content queued at c in pool b, at least 0
                    max(v["kappa"][inst.chokepoint_ordinal[c], b] - queued.get((c, b), 0.0), 0.0)
                    for (c, b), _members in self._through_rows
                ],
            ]
        )
        lp = L.highs_lp(cost, np.zeros(ns + nd), ub, self._A, row_lb, row_ub)
        return lp, allowed

    def act(self, obs: dict) -> dict:
        """Solve week t's LP through the ladder and return its requests (never overrides or holds)."""
        start = time.perf_counter()
        t = int(obs["week"])
        self._memory.update(self._inst, obs)
        lp, allowed = self._model(obs)
        res = self._session.solve(lp)
        if res.ok:  # ``allowed`` ascends and ``self._slots`` is sorted: the flows are in slot order
            slots = [self._slots[j] for j in allowed]
            qty = [max(float(res.x[j]), 0.0) for j in allowed]
            action = {"week": t, "flows": {"slot": slots, "qty": qty}, "overrides": None, "hold": None}
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action

    def state(self) -> dict:
        """The cross-week state (the session's basis, the observed-graph memory) for an in-process resume."""
        return {"session": self._session.state(), "memory": self._memory.state()}

    def load_state(self, state: dict) -> None:
        """Restore ``state()`` after a reset of the same episode."""
        self._session.load_state(state["session"])
        self._memory = L.ObservedGraph.from_state(self._inst, state["memory"])

"""The naive anchor, the numbered rule of design §8.1 (Q8, Q54, Q84, Q98): RSS = 0 and the D9 fallback.

A function of the current prediction-free observation and the instance only; no state across weeks (the plan is a
function of the instance). Reset: the plan from the reset-time min-cost flow (22) (``instance.nominal``), route demand
by node type (step 2), order-up-to levels S_jk = sum of S^nv_lk (67) plus I-bar_g at grids, offset 0 except into a
terminal (step 3), and the last useful weeks (71)-(72) of the end-aware cut. Week t:
inventory position (step 4), order-up-to (5), route choice with the 2x give-up rule on routes observed closed (6), and
no flow on a slot past its last useful week (the end-aware cut, Q98; below); the requests then pass the clip like any
policy's (7). On an injected event list without a generator F_Q is the point mass at 0, so the closure term of (67)
vanishes (§8.1). ``NaivePolicy(end_aware=False)`` (named ``naive_plain``) is the horizon-blind naive without the cut,
kept for tests and evidence ("plain naive"); it is neither the anchor nor the fallback.

Readings where §8.1 leaves a choice (the conservative one, fixed here):

- **Route demand** (step 2), d_lk per week: into a sink upsilon F_lk m^sea(t) (F_lk the route's nominal flow, its share
  of F-bar_ik times F-bar_ik); into a fab upsilon cap^0_f times the route's share of the fab's wafer inflow; into a grid
  zeta_gk G-bar^0_g times the route's share of the fuel inflow; into an OSAT upsilon F_lk; into a terminal F_lk (the
  route's nominal flow, no utilisation; tested on a terminal variant of `tiny`, which has none). A route into any other
  node type has no rule and raises. ``NaiveRoute.d`` holds the value at m^sea = 1; the seasonal factor of week t is
  m^sea[(t - 1) mod 52] (episode week 1 = week 1 of the profile) and applies to both terms of (67) and to the cap in
  the week-t target. The order split uses the same d_lk, which is proportional to the nominal flow within one
  destination (Q84's "in proportion to nominal flow"), m^sea cancelling there.
- **Duplicate**: the first edge (instance order), else the first lane, whose ``alt_of`` names the route (``{"lane": l}``
  for a lane route, ``{"edge": e}`` for an edge route), permits k and has an action slot; its first edge, lane and
  base freight (sum of c^0) are kept.
- **Offset** (step 3, red-team CMP2-1): step 3 derives offset 0 from the step order (arrivals at step 6 precede
  consumption at steps 7-8). A terminal consumes by dispatching to its grid at step 5, which draws on I^{t-1} before
  the week's arrivals, so by the same derivation an order on a route into a terminal serves the dispatch of week
  t + tau_l + 1: o_l = 1 there, S*_lk = (tau_l + 2) d_lk + closure term, and the cover cap shifts by the same week;
  o_l = 0 into every other node type (fabs, OSATs, grids and sinks consume their inflow at steps 7-8). The terminal's
  end-of-week stock then settles at S_jk - sum_l (tau_l + 1) b (b its weekly dispatch), leaving S_jk - sum_l
  (tau_l + 2) b >= 0 after the next week's dispatch.
- **Cover cap** (§8.1, §11 row 37): w_max = max(tau_l, tau_alt) + 1 + o_l with alt the duplicate, else the fastest
  other route into j for k open at reset (an action-slot edge or lane into j permitting k, no (edge, k) prohibited at
  reset, u^0 > 0); uncapped (None) with neither.
- **Closure term** of (67): the quantile F^{-1}_{Q_cb}(pi/(pi + h)) of a route into (j, k) through chokepoint c in
  pool b is ``fq_quantile[(c, pool, j, k)]`` (the critical ratio is the destination's, §2.4 profile row;
  ``naive_fq.critical_quantiles`` builds these from the laws of ``naive_fq.fq_laws``), else 0; pool names are "tb",
  "ct", and a key that names no lane route of the plan raises. A route into a fab or an OSAT carries no term (1^cl =
  0 of (67), ``carries_closure_term``; owner queue M5-O33 (b)): lot starts (12) and packaging (19) take that stock
  each week, so it is never held; its quantile stays in the map (published per lane) and is not read. d_cb of (66)
  sums d_lk over the plan's lanes through c in pool b (``lane_demand``), the plants' lanes included (their cargo
  queues at c). A lane through several chokepoints adds one term per chokepoint
  (provisional: §8.1 prescribes a reset-time simulation of naive's routed network for tandem lanes; `tiny` has one
  chokepoint). ``fq_quantile=None`` gives 0 (the point mass at 0 of injected lists). On a generated omega the
  quantiles are the generator's, ``naive_fq.generator_quantiles(inst, params)`` (cached per instance content digest,
  ``generator_id`` and replication count, never per hash label alone; DET-M2-3), for the anchor of (55)
  (``naive_fq.anchor_policy``) and the D9 fallback alike (``NaiveFallback``, the ``Env`` fallback spec, which carries
  the generator id and the instance content digest). The quantiles are converted to floats before the plan and its
  cache key see them (``canonical_quantiles``; DET-P3-4): a finite real number >= 0 (F^{-1} of a queue length), a
  ``Decimal`` included; any other value raises.
- **Grid buffer** (step 3): I-bar_gk of (16) is added to S_jk when j is grid g and k one of its buffered fuels (the
  grid's ``ibar`` entries: the gas rationing buffer, and on `small` and `full` the nuclear fuel cover, design §3.5);
  other fuels get none. On `tiny` the only buffered fuel is the rationed gas.
- **Observation** (step 4, 6): IP_jk sums on-hand stock, pipeline shipments of k whose route ends at j (the lane's
  destination when the shipment carries a lane, else the edge head), queued lots of k whose lane ends at j, minus
  backlog. A route is observed closed if a chokepoint on it shows ``graph_now.open`` <= the threshold, or an edge on it
  shows u <= 0 or is prohibited for k; a null (unobserved) value never closes a route. A request on a slot whose own
  (edge, k) ``graph_now.prohibited`` lists is not sent: it would be dropped as invalid (§9.3), which equals a zero
  request, so omitting it changes no dynamics and keeps naive's invalid count at 0. That is the dispatch rule, not
  ``slot_mask``, which since Q111 also masks a lane slot for a later edge of its lane (a request the environment would
  carry out); ``slot_mask`` equalled the rule before Q111 and is null exactly when ``graph_now`` is, so naive, the
  anchor and the D9 fallback play as before. Other requests are sent even at qty 0.
- **Immutability** (DET-3): the plan is deeply immutable (tuples and ``FrozenDict``, coerced on construction, so a write
  raises TypeError), and the per-instance cache keeps the instance and plan pickled: every reset unpickles a private
  copy, so no policy (a subclass included, even one bypassing FrozenDict) can change J^naive or the D9 fallback of a
  later episode.
- **Provenance** (DC-3, DET-2): every reset hashes the Static's instance JSON as received, keys the cache by that hash
  and raises when it is not the declared ``instance_hash``, so the same Static raises or not whatever ran before.

**End-aware cut** (Q98, design (71)-(72); the rule "R7" as measured on branch ``q96-r7``, b0cc92f). Naive starts no
work whose goods cannot reach a point of use by T, judged from T and Static's nominal lead times only (no forecast, no
observed event): its action is naive's (steps 4-6 unchanged) with every flow on an action slot past that slot's last
useful week removed (``last_useful_weeks``, ``end_aware_action``). The plan (22), F_Q (66)-(67) and the load (69) do
not read it. The rule, with tau_e the nominal lead of edge e in Static, tau_l the sum over a route's edges, R+(i, k)
naive's planned routes (step 1) out of node i carrying k and head(l) a route's destination (a lane's last head):

- *Last useful arrival week* A_jk (71), the last week in which a unit of k arriving at node j can still reach a point
  of use by T through naive's planned routes (arrivals join stock at step 6 of §3.2):
  - A_jk = T at a point of use: a sink with demand for k (served at step 8 of its arrival week) and a grid burning k
    (burned at step 7);
  - A_fw = max_{l in R+(f, r)} (A_{head(l) r} - tau_l) - 1 - tau^fab_f at a fab f with input w and product r: a lot
    starts in its arrival week (step 7), its raw chips join the fab's stock at step 7 of week start + tau^fab and leave
    one week later at the earliest, since a dispatch draws on last week's stock (5);
  - A_or = max_{l in R+(o, p)} (A_{head(l) p} - tau_l) - 1 - tau^osat_o at an OSAT o packaging raw r into p: packaging
    starts in the arrival week (step 7), its output joins stock at week start + tau^osat and leaves a week later;
  - A_ik = max_{l in R+(i, k)} (A_{head(l) k} - tau_l) - 1 at any other node with a planned route out (a terminal,
    and a supply node: on `tiny` naive's routes leave src_gulf on E0 and mat_jp on E9, E11 and E13, so A is 22 for LNG
    at src_gulf and 9 for wafers at mat_jp; no action slot and no planned route ends at a supply node, so no W reads
    it);
  - none (no week is useful) at a node with no planned route out that is not a point of use (a chokepoint, which a
    lane passes through, or a node naive's plan does not feed).
- *Last useful week* (72) of an action slot s = (e, k, lane) bound for j (the lane's destination, else the edge head):
  W_s = A_jk - tau_s, tau_s the slot's nominal transit. A lane through chokepoints counts its edges' leads only: a
  chokepoint passes cargo on in its arrival week (§3.4), and the rule reads neither the queue nor the open fraction, so
  a queue wait is never assumed; the chokepoint itself is no point of use (a lane's cargo is bound for its
  destination). An ``alt_of`` duplicate naive switches to (step 6) is judged by its own transit: the Cape duplicate E3
  of L1 on `tiny` has W = T - 6, where L1 has T - 4. Slots naive never sends get a W too, from their destination.
- *Last useful start week* of a plant: A_fw at fab f (its wafers), A_or at OSAT o (each raw chip it packages).
- *The target* of week t (Q109, amending Q98's "bit for bit"): S^end_jk(t) = S_jk(t) less the cut's phantom pipeline,
  sum over naive's routes l into (j, k) of sum_{u = max(W_l + 1, t - tau_l, 1)}^{t - 1} d_lk m^sea(u)
  (``phantom_pipeline``, ``end_aware_targets``; W_l of the route's own slot): the dispatches the cut withheld would be
  in the pipeline, and without the correction their absence reads as a deficit that step 5 re-splits onto the
  destination's faster routes (the week-103 loss of `full`, 2,726 chips at the anchor before Q109). A naive-derived
  baseline's own targets pass through the same correction (``end_aware_action(..., targets=)``).
- *The action* of week t: naive's steps 4-6 on S^end, on every slot s with t <= W_s, and no entry on any other slot.
  Nothing else changes: no re-split of a dropped share (the next week's inventory position is lower, and naive orders
  again on its routes), no override and no hold (naive sends none, §8.1 step 6; the default release at a chokepoint is
  the environment's rule, §3.4, not naive's dispatch, so queue lots already at c are stock on hand), and stock on hand
  is untouched: a fab still starts, and an OSAT still packages, the wafers and raw chips on hand after A (a simulator
  rule, §3.5), which only a dispatch naive does not send can prevent.
- *Fab conversion* (wafer to raw chip): judged at nominal, with tau^fab and no capacity, scrap, restoration or energy
  limit, so a lot that waits for capacity starts later than A_fw assumes; the rule never reads those.

On `tiny` at T = 26: A = 26 at `sink_us` and both grids, 19 at `osat_sea` (raw), 9 / 11 / 6 at `fab_tw` / `fab_cn` /
`fab_us` (wafers); W = 23 on L0, 22 on L1, 20 on E3, 8 / 10 / 3 on E9 / E11 / E13, 18 / 18 / 15 on E17 / E19 / E20 and
22 on L3 (``tests/test_naive_end_aware.py``). The D9 fallback is the same rule: the ``Env`` spec ``"naive"`` (injected
lists) or a ``NaiveFallback`` (``naive_fq.naive_fallback``), both end-aware by default; ``"naive_plain"`` and
``NaiveFallback(..., end_aware=False)`` play plain naive (tests and evidence only).
"""

import math
import numbers
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal

from sbfplan.instance.io import load_instance, sha256_hex
from sbfplan.instance.nominal import nominal_flow
from sbfplan.instance.schema import Instance, frozen_map
from sbfplan.policies.base import run_name


CLOSED_THRESHOLD = 0.6  # "observed closed" open fraction, the realism band's impairment cut, SYNTHETIC (§11 row 83)
GIVE_UP_RATIO = 2.0  # a duplicate is used only within 2x the route's base freight (DisruptSC, R2 P10)
# destinations whose processing takes their stock each week, lot starts (12) and packaging (19): 1^cl = 0 in (67)
PROCESSING_TYPES = frozenset({"fab", "osat"})


def carries_closure_term(inst: Instance, j: int) -> bool:
    """1^cl of (67) for a route into node j: False into a fab or an OSAT, True elsewhere (owner queue M5-O33 (b)).

    A fab starts its wafers up to cap^0 (12) and an OSAT packages its whole raw stock up to thr (19) every week, so a
    closure term there is never held: it is processed on arrival, naive re-orders the gap every week, and the sinks
    lose sales (§8.1). The safety stock stays at sinks, grids and terminals; d_cb, F_Q and the load do not read this.
    """
    return inst.nodes[j].type not in PROCESSING_TYPES


def check_thresholds(closed_threshold: float, give_up_ratio: float) -> None:
    """Refuse naive's step-6 thresholds outside their domain (red-team INT-5).

    ``closed_threshold`` is the open-fraction cut of step 6, in (0, 1]: a NaN never compares true, so naive would never
    observe a closure, and at 1 every route through a chokepoint is observed closed at nominal, the most a cut can do.
    ``give_up_ratio`` is a finite freight ratio >= 1: below 1 a duplicate would be refused even at the route's own
    freight. Booleans and non-real values are refused too.

    Raises:
        ValueError: if either threshold is outside its domain.

    """
    for name, value in (("closed_threshold", closed_threshold), ("give_up_ratio", give_up_ratio)):
        try:
            finite = not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(value)
        except OverflowError:  # a Real whose float() overflows (e.g. a huge Fraction)
            finite = False
        if not finite:
            raise ValueError(f"naive {name} must be a finite real number, got {value!r}")
    if not 0.0 < closed_threshold <= 1.0:
        raise ValueError(f"naive closed_threshold is an open fraction in (0, 1], got {closed_threshold!r}")
    if not give_up_ratio >= 1.0:
        raise ValueError(f"naive give_up_ratio is a freight ratio >= 1, got {give_up_ratio!r}")


def canonical_quantiles(fq_quantile: Mapping | None) -> dict:
    """The F_Q quantiles of (67) as floats: the one form the plan and naive's plan cache see (DET-P3-4).

    A quantile F^{-1}_{Q_cb}(pi/(pi+h)) of a queue length is a finite real number >= 0. A real number of any type (int,
    float, Fraction, NumPy real) or a ``Decimal`` is converted with ``float()``, so equal quantiles of different types
    give one plan and one cache entry, in a fresh process or after caching (V1). Keys are kept as given.

    Raises:
        TypeError: if a quantile is not a real number (a bool, a str, a complex or None).
        ValueError: if a quantile is NaN, infinite, beyond the float range or negative.

    """
    out = {}
    for key, value in (fq_quantile or {}).items():
        if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
            raise TypeError(f"naive F_Q quantile at {key!r} must be a real number, got {type(value).__name__}")
        try:
            q = float(value)
        except OverflowError:  # an int or Fraction beyond the float range
            q = math.inf
        if not math.isfinite(q) or q < 0.0:
            raise ValueError(f"naive F_Q quantile at {key!r} must be a finite number >= 0 (a queue length), got {q!r}")
        out[key] = q
    return out


@dataclass(frozen=True)
class NaiveParams:
    """Naive's ``policy/`` parameters: its step-6 "observed closed" cut (M1), the one field the group reads for naive.

    The registry builds naive from it (``registry.make_policy("naive", context, NaiveParams(...))``), so every entry
    point plays the cut a config names (SIM-M4-01); the default is the anchor's 0.6 (SYNTHETIC, §11 row 83). The 2x
    give-up ratio is the rule's own, not a parameter of the group (design §12 row "Policy group (M4 runner)").

    Raises:
        ValueError: as ``check_thresholds``.

    """

    closed_threshold: float = CLOSED_THRESHOLD

    def __post_init__(self) -> None:
        check_thresholds(self.closed_threshold, GIVE_UP_RATIO)


_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class NaiveFallback:
    """The D9 fallback of a generated episode: naive with the F_Q quantiles of the generator that drew omega (§9.3).

    Naive is a function of the instance and the generator only (§8.1): ``fq_quantile`` holds the quantiles of (67)
    that ``naive_fq.generator_quantiles`` computes for the generator whose ``generator_id`` (§4.1) is stored here, on
    the instance whose family digest (``Instance.family_digest``: its content at its file's own rung) is
    ``instance_digest``; ``dynamics.env.Env`` plays
    naive with them and refuses an omega drawn by another generator, and an instance of other content (the hash label
    alone, which ``generator_id`` reads, is carried over by a ``dataclasses.replace`` variant; REG-M2R2-01). Build it
    with ``naive_fq.naive_fallback(inst, params)``, which sets both; a spec built by hand without ``instance_digest``
    (None) is bound to no content, and ``Env`` checks its generator only. An ``Env`` snapshot carries it like any
    fallback spec (B2). ``end_aware`` (default True) plays the end-aware naive of the module docstring (Q98), the
    anchor's rule; False plays plain naive (tests and evidence only).

    Raises:
        ValueError: if ``generator_id``, or ``instance_digest`` when given, is not a SHA-256 hex digest, a quantile
            is refused by ``canonical_quantiles`` (TypeError for a non-real one), or ``end_aware`` is not a bool.

    """

    generator_id: str
    fq_quantile: Mapping[tuple, float]
    instance_digest: str | None = None
    end_aware: bool = True  # the end-aware cut of the module docstring (Q98); False is plain naive

    def __post_init__(self) -> None:
        for name in ("generator_id", "instance_digest"):
            value = getattr(self, name)
            if name == "instance_digest" and value is None:
                continue
            if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
                raise ValueError(f"{name} must be a SHA-256 hex digest (§4.1, V3), got {value!r}")
        if type(self.end_aware) is not bool:
            raise ValueError(f"end_aware must be a bool, got {self.end_aware!r}")
        object.__setattr__(self, "fq_quantile", frozen_map(canonical_quantiles(self.fq_quantile)))


@dataclass(frozen=True)
class NaiveRoute:
    """One route l of naive's plan into destination (j, k): a lane through chokepoints or one internal edge."""

    dest: int  # node j
    k: int
    first_edge: int
    lane: int | None
    edges: tuple[int, ...]
    d: float  # route demand d_lk per week (step 2); into sinks at m^sea = 1 (week t multiplies by m^sea(t))
    tau: int  # tau_l, sum of edge leads
    offset: int  # o_l of step 3: 0, or 1 into a terminal (its consumption, the dispatch to its grid, precedes arrivals)
    base_cost: float  # sum of c^0 over the route's edges
    alt_edge: int | None  # first edge of the `alt_of` duplicate, if any
    alt_lane: int | None  # its lane, if the duplicate rides one
    alt_cost: float | None  # the duplicate's base freight
    w_max: float | None  # cover cap w^max_l in weeks (§8.1, §11 row 37); None = uncapped
    level: float  # S^nv_lk of (67) (with F_Q the point mass at 0 on injected lists)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))  # immutable whatever sequence was passed (DET-3)


@dataclass(frozen=True)
class NaivePlan:
    """Naive's plan (reset steps 1-3): routes grouped by destination in node order, and the order-up-to targets.

    Deeply immutable (DET-3): ``routes`` becomes a tuple and every mapping a ``FrozenDict`` copy on construction, so
    ``dataclasses.replace`` with a plain dict still yields a frozen plan and a write raises TypeError.
    """

    routes: tuple[NaiveRoute, ...]
    targets: Mapping[tuple[int, int], float]  # (j, k) -> S_jk (step 3), at m^sea = 1
    # (c, pool, j, k) -> the F_Q quantile of (67) (see the module docstring, "Closure term")
    fq_quantile: Mapping[tuple, float] = field(default_factory=dict)
    # (j, k) -> its routes, destinations in plan order; derived from ``routes``
    groups: Mapping[tuple[int, int], tuple[NaiveRoute, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "targets", frozen_map(self.targets))
        object.__setattr__(self, "fq_quantile", frozen_map(self.fq_quantile))
        groups: dict[tuple[int, int], list[NaiveRoute]] = {}
        for r in self.routes:
            groups.setdefault((r.dest, r.k), []).append(r)
        object.__setattr__(self, "groups", frozen_map({jk: tuple(rs) for jk, rs in groups.items()}))


def level(
    d: float,
    tau: int,
    w_max: float | None = None,
    closure: Iterable[tuple[float, float]] = (),
    offset: int = 0,
) -> float:
    """S^nv_lk of (67): min{S*_lk, w_max d_lk}, S*_lk = (tau_l + 1 + o_l) d_lk + sum_c (d_lk / d_cb) F^{-1}(pi/(pi+h)).

    Args:
        d: the route demand d_lk per week.
        tau: the route transit tau_l in weeks.
        w_max: the cover cap in weeks, or None (uncapped); the caller shifts it by the offset.
        closure: one (d_cb, quantile) pair per chokepoint on the lane, d_cb of (66); empty off chokepoints.
        offset: o_l of §8.1 step 3, 0 where the destination consumes after the week's arrivals, 1 into a terminal.

    """
    s_star = (tau + 1 + offset) * d
    for d_cb, quantile in closure:
        if quantile:
            s_star += d / d_cb * quantile
    return s_star if w_max is None else min(s_star, w_max * d)


def _route_edges(inst: Instance, edge: int, lane: int | None) -> tuple[int, ...]:
    return inst.lanes[lane].edges if lane is not None else (edge,)


def _destination(inst: Instance, edge: int, lane: int | None) -> int:
    """The node a flow on ``edge`` (riding ``lane``, if any) is bound for."""
    return inst.lane_destination(lane) if lane is not None else inst.edges[edge].head


def _open_at_reset(inst: Instance, z0: set[tuple[int, int]], edges: tuple[int, ...], k: int) -> bool:
    """No (edge, k) of the route in Z_0 and every edge with u^0 > 0."""
    return all((e, k) not in z0 and (inst.edges[e].u0 or 0.0) > 0.0 for e in edges)


def _duplicate(inst: Instance, k: int, first_edge: int, lane: int | None) -> tuple[int, int | None] | None:
    """The route's `alt_of` duplicate as (first edge, lane), or None (first edge match, then first lane match)."""
    ref = ("lane", lane) if lane is not None else ("edge", first_edge)
    slots = inst.action_slot_index
    for e, edge in enumerate(inst.edges):
        if edge.alt_of is not None and (edge.alt_of.kind, edge.alt_of.index) == ref and (e, k, None) in slots:
            return e, None
    for li, ln in enumerate(inst.lanes):
        if ln.alt_of is not None and (ln.alt_of.kind, ln.alt_of.index) == ref and (ln.edges[0], k, li) in slots:
            return ln.edges[0], li
    return None


def _route_demand(inst: Instance, j: int, k: int, flow: float, total: float) -> float:
    """d_lk of §8.1 step 2 at m^sea = 1 (see the module docstring for the rule per node type)."""
    node = inst.nodes[j]
    ups = inst.params.upsilon
    if node.type in ("sink", "osat"):
        return ups * flow
    if node.type == "fab" and k == node.fab.input:
        return ups * node.fab.cap0 * (flow / total)
    if node.type == "grid" and k in node.grid.fuels:
        return node.grid.shares[k] * node.grid.deliverable * (flow / total)
    if node.type == "terminal":
        return flow
    raise ValueError(f"naive step 2 has no route-demand rule for {inst.commodities[k].id} into {node.type} {node.id}")


def _offset(inst: Instance, j: int) -> int:
    """o_l of §8.1 step 3 for a route into node j: 1 into a terminal, else 0.

    Step 3's offset 0 holds because arrivals (§3.2 step 6) precede consumption (steps 7-8), so an order of week t is
    the last to arrive before the consumption of week t + tau_l. A terminal consumes by dispatching to its grid at step
    5, which draws on I^{t-1} before the week's arrivals (dispatch precedence, (5)), so an order of week t serves the
    dispatch of week t + tau_l + 1: one more week of cover, (tau_l + 2) d_lk, and a cover cap one week longer.
    """
    return 1 if inst.nodes[j].type == "terminal" else 0


def _seasonal(inst: Instance, week: int) -> dict[tuple[int, int], float]:
    """m^sea of week t per sink demand with a seasonal profile (``Demand.m_sea``, profile index (t - 1) mod 52)."""
    return {(dem.node, dem.k): dem.m_sea(week) for dem in inst.demands if dem.seasonal is not None}


def lane_demand(inst: Instance, routes: Iterable[NaiveRoute], ds: Iterable[float]) -> dict[tuple[int, str], float]:
    """d_cb of (66): the route demands ``ds`` summed over the routes whose lane passes chokepoint c, per pool name.

    Keys are (chokepoint node, pool name) for every pair a lane of ``routes`` passes; routes off chokepoints add
    nothing. The sum runs in route order.
    """
    d_cb: dict[tuple[int, str], float] = defaultdict(float)
    for r, d in zip(routes, ds):
        if r.lane is not None:
            for c in inst.lanes[r.lane].chokepoints:
                d_cb[(c, inst.commodities[r.k].pool)] += d
    return dict(d_cb)


def _closures(inst: Instance, routes: tuple[NaiveRoute, ...], ds: list[float], fq: Mapping) -> list[list]:
    """Per route, the (d_cb, quantile) pairs of (67), one per chokepoint on its lane.

    Empty without F_Q, off lanes, and on a route into a fab or an OSAT (1^cl = 0, ``carries_closure_term``); d_cb of
    (66) still sums every lane route of ``routes``, those into plants included.
    """
    if not fq:
        return [[] for _ in routes]
    d_cb = lane_demand(inst, routes, ds)
    out = []
    for r in routes:
        closure = []
        if r.lane is not None and carries_closure_term(inst, r.dest):
            pool = inst.commodities[r.k].pool
            closure = [(d_cb[(c, pool)], fq.get((c, pool, r.dest, r.k), 0.0)) for c in inst.lanes[r.lane].chokepoints]
        out.append(closure)
    return out


def _levels(inst: Instance, routes: tuple[NaiveRoute, ...], ds: list[float], fq: Mapping) -> list[float]:
    """S^nv_lk of (67) for every route given the route demands ``ds`` (d_cb of (66) summed over the plan's lanes)."""
    closures = _closures(inst, routes, ds, fq)
    return [level(d, r.tau, r.w_max, cl, offset=r.offset) for r, d, cl in zip(routes, ds, closures)]


def star_levels(inst: Instance, plan: "NaivePlan") -> tuple[float, ...]:
    """S*_lk of (67) per plan route at m^sea = 1, before the cover cap (published per lane with each split, §8.1)."""
    ds = [r.d for r in plan.routes]
    closures = _closures(inst, plan.routes, ds, plan.fq_quantile)
    return tuple(level(d, r.tau, None, cl, offset=r.offset) for r, d, cl in zip(plan.routes, ds, closures))


def _targets(inst: Instance, routes: tuple[NaiveRoute, ...], levels: list[float]) -> dict[tuple[int, int], float]:
    """S_jk of step 3: the sum of the route levels into (j, k), plus I-bar_gk for a buffered fuel k of grid j."""
    grouped: dict[tuple[int, int], list[float]] = {}
    for r, lv in zip(routes, levels):
        grouped.setdefault((r.dest, r.k), []).append(lv)
    targets = {}
    for (j, k), lvs in grouped.items():
        s = sum(lvs)
        grid = inst.nodes[j].grid
        if grid is not None and k in grid.ibar:
            s += grid.ibar[k]  # the grid's buffer of k (16) is in the target: gas's rationing buffer, a nuclear cover
        targets[(j, k)] = s
    return targets


def naive_plan(inst: Instance, fq_quantile: dict | None = None) -> NaivePlan:
    """Reset steps 1-3 of §8.1, from the instance only.

    ``fq_quantile`` maps (chokepoint node, pool, destination node, commodity) of the plan's lane routes to the
    critical-ratio quantile of F_Q; None means the point mass at 0 (injected event lists). The quantiles are taken as
    floats (``canonical_quantiles``, which raises).

    Raises:
        ValueError: on a key that names no lane route of the plan through its chokepoint and pool (and as
            ``canonical_quantiles``).

    """
    fq = frozen_map(canonical_quantiles(fq_quantile))
    nf = nominal_flow(inst)
    totals: dict[tuple[int, int], float] = defaultdict(float)
    for r in nf.routes:
        totals[(r.dest, r.k)] += r.flow
    z0 = set(inst.prohibitions_at_reset)
    into: dict[tuple[int, int], list[tuple[int, int | None]]] = defaultdict(list)  # (j, k) -> (first edge, lane)
    for e, k, lane in inst.action_slots:  # every action slot that delivers k to j, in slot order
        into[(_destination(inst, e, lane), k)].append((e, lane))
    draft = []
    for r in nf.routes:
        dup = _duplicate(inst, r.k, r.first_edge, r.lane)
        tau = sum(inst.edges[e].tau for e in r.edges)
        offset = _offset(inst, r.dest)
        if dup is not None:
            alt_edges = _route_edges(inst, *dup)
            tau_alt = sum(inst.edges[e].tau for e in alt_edges)
            alt_cost = sum(inst.edges[e].c0 for e in alt_edges)
        else:
            others = [
                sum(inst.edges[e].tau for e in _route_edges(inst, e0, ln))
                for e0, ln in into.get((r.dest, r.k), ())
                if (e0, ln) != (r.first_edge, r.lane) and _open_at_reset(inst, z0, _route_edges(inst, e0, ln), r.k)
            ]
            tau_alt = min(others) if others else None
            alt_cost = None
        draft.append(
            NaiveRoute(
                dest=r.dest,
                k=r.k,
                first_edge=r.first_edge,
                lane=r.lane,
                edges=r.edges,
                d=_route_demand(inst, r.dest, r.k, r.flow, totals[(r.dest, r.k)]),
                tau=tau,
                offset=offset,
                base_cost=sum(inst.edges[e].c0 for e in r.edges),
                alt_edge=None if dup is None else dup[0],
                alt_lane=None if dup is None else dup[1],
                alt_cost=alt_cost,
                w_max=None if tau_alt is None else float(max(tau, tau_alt) + 1 + offset),  # shifted by o_l (step 3)
                level=0.0,
            )
        )
    routes = tuple(draft)
    keys = {
        (c, inst.commodities[r.k].pool, r.dest, r.k)
        for r in routes
        if r.lane is not None
        for c in inst.lanes[r.lane].chokepoints
    }
    if unknown := set(fq) - keys:
        raise ValueError(f"naive F_Q quantiles are keyed (c, pool, j, k) by the plan's lane routes; not {unknown!r}")
    levels = _levels(inst, routes, [r.d for r in routes], fq)
    routes = tuple(replace(r, level=lv) for r, lv in zip(routes, levels))
    return NaivePlan(routes=routes, targets=_targets(inst, routes, levels), fq_quantile=fq)


def _targets_at(inst: Instance, plan: NaivePlan, week: int) -> dict[tuple[int, int], float]:
    """S_jk of week t: the plan's targets, recomputed with d_lk m^sea(t) into seasonal sinks."""
    m = _seasonal(inst, week)
    if not m:
        return plan.targets
    ds = [r.d * m.get((r.dest, r.k), 1.0) for r in plan.routes]
    return _targets(inst, plan.routes, _levels(inst, plan.routes, ds, plan.fq_quantile))


def _observed_closed(
    inst: Instance, edges: tuple[int, ...], lane: int | None, k: int, g: dict, prohibited: set, threshold: float
) -> bool:
    """Step 6: a chokepoint on the route shows open <= threshold, or an edge on it shows u <= 0 or a prohibition.

    A null list or value is unobserved and closes nothing.
    """
    opens, caps = g.get("open"), g.get("u")
    if lane is not None and opens is not None:
        for c in inst.lanes[lane].chokepoints:
            o = opens[inst.chokepoint_ordinal[c]]
            if o is not None and o <= threshold:
                return True
    for e in edges:
        u = None if caps is None else caps[e]
        if (u is not None and u <= 0.0) or (e, k) in prohibited:
            return True
    return False


def naive_action(
    inst: Instance,
    plan: NaivePlan,
    obs: dict,
    closed_threshold: float = CLOSED_THRESHOLD,
    give_up_ratio: float = GIVE_UP_RATIO,
    *,
    targets: Mapping[tuple[int, int], float] | None = None,
) -> dict:
    """Week steps 4-6 from the prediction-free observation; returns a wire-schema action (no overrides, no holds).

    ``targets`` (M4) replaces the week's S_jk of step 3 for a naive-derived baseline (``nd``, ``sz_state_base_stock``,
    ``human_ref``; design §8.2), which then differs from naive by its level alone; None is naive's own.

    Raises:
        ValueError: if a threshold is outside its domain (``check_thresholds``).

    """
    check_thresholds(closed_threshold, give_up_ratio)
    week = obs["week"]
    targets = _targets_at(inst, plan, week) if targets is None else targets
    on_hand: dict[tuple[int, int], float] = defaultdict(float)
    st = obs["stock"]
    for i, k, q in zip(st["node"], st["k"], st["qty"]):
        on_hand[(i, k)] += q
    backlog: dict[tuple[int, int], float] = defaultdict(float)
    bl = obs.get("backlog") or {"node": [], "k": [], "qty": []}
    for i, k, q in zip(bl["node"], bl["k"], bl["qty"]):
        backlog[(i, k)] += q
    in_transit: dict[tuple[int, int], list[float]] = defaultdict(list)
    pp = obs["pipeline"]
    for e, k, lane, q in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"]):
        in_transit[(_destination(inst, e, lane), k)].append(q)
    queued: dict[tuple[int, int], list[float]] = defaultdict(list)
    ql = obs["queue_lots"]
    for lane, k, q in zip(ql["lane"], ql["k"], ql["qty"]):
        queued[(inst.lane_destination(lane), k)].append(q)

    g = obs.get("graph_now") or {}
    pr = g.get("prohibited") or {"edge": [], "k": []}
    prohibited = set(zip(pr["edge"], pr["k"]))
    slot_of = inst.action_slot_index
    flows: dict[int, float] = {}
    for (j, k), routes in plan.groups.items():
        ip = on_hand.get((j, k), 0.0)  # step 4
        ip += sum(in_transit.get((j, k), []))
        ip += sum(queued.get((j, k), []))
        ip -= backlog.get((j, k), 0.0)
        q = max(0.0, targets[(j, k)] - ip)  # step 5
        dsum = sum(r.d for r in routes)
        if dsum <= 0.0:  # no route demand to split the order by (e.g. upsilon = 0): nothing is requested
            continue
        for r in routes:
            share = q * r.d / dsum
            slot = (r.first_edge, r.k, r.lane)
            if (  # step 6: switch to the duplicate if the route is closed and the duplicate open and within 2x
                r.alt_edge is not None
                and _observed_closed(inst, r.edges, r.lane, r.k, g, prohibited, closed_threshold)
                and r.alt_cost <= give_up_ratio * r.base_cost  # "at most 2x": equality switches
                and not _observed_closed(
                    inst, _route_edges(inst, r.alt_edge, r.alt_lane), r.alt_lane, r.k, g, prohibited, closed_threshold
                )
            ):
                slot = (r.alt_edge, r.k, r.alt_lane)
            s = slot_of[slot]
            if (slot[0], r.k) in prohibited:
                continue  # its edge is prohibited this week: dropped as invalid if sent (§9.3), a zero request (Q111)
            flows[s] = flows.get(s, 0.0) + share
    slots = sorted(flows)
    return {"week": week, "flows": {"slot": slots, "qty": [flows[s] for s in slots]}, "overrides": None, "hold": None}


@dataclass(frozen=True)
class LastUseful:
    """The last useful weeks (71)-(72) of the end-aware cut (module docstring), from the instance and plan.

    Attributes:
        T: the horizon they were derived for.
        arrival: (node j, commodity k) -> A_jk of (71), the last week a unit of k arriving at j can still reach a point
            of use by T through naive's planned routes; None when no week is. Keys: every (j, k) some action slot or
            planned route delivers to, and every (j, k) the recursion visited.
        slot: W_s of (72) per action slot index, A_jk - tau_s of the slot's destination; None when A_jk is None.

    """

    T: int
    arrival: Mapping[tuple[int, int], int | None]
    slot: tuple[int | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrival", frozen_map(self.arrival))
        object.__setattr__(self, "slot", tuple(self.slot))

    def useful(self, s: int, week: int) -> bool:
        """Whether a dispatch on action slot ``s`` in ``week`` can still reach a point of use by T (week <= W_s)."""
        w = self.slot[s]
        return w is not None and week <= w

    def start(self, inst: Instance, node: int) -> dict[int, int | None]:
        """A plant's last useful start week per input: A of a fab's wafers, A of each raw chip an OSAT packages."""
        n = inst.nodes[node]
        if n.fab is not None:
            return {n.fab.input: self.arrival.get((node, n.fab.input))}
        if n.osat is not None:
            return {raw: self.arrival.get((node, raw)) for raw in sorted(n.osat.packages)}
        raise ValueError(f"{n.id} is neither a fab nor an OSAT: it starts no work")


def _slot_transit(inst: Instance, s: int) -> int:
    """tau_s: the nominal transit of action slot s, the sum of Static's leads over its lane's edges, else its edge's."""
    e, _k, lane = inst.action_slots[s]
    return sum(inst.edges[x].tau for x in _route_edges(inst, e, lane))


def last_useful_weeks(inst: Instance, plan: NaivePlan) -> LastUseful:
    """A_jk (71) and W_s (72) of the end-aware cut (module docstring), from T, Static's nominal leads and the plan only.

    Raises:
        ValueError: if naive's planned routes form a cycle (the recursion over destinations would not end).

    """
    T = inst.T
    out: dict[tuple[int, int], list[NaiveRoute]] = defaultdict(list)  # R+(i, k): planned routes out of i carrying k
    for r in plan.routes:
        out[(inst.edges[r.first_edge].tail, r.k)].append(r)
    demanded = {(d.node, d.k) for d in inst.demands}
    arrival: dict[tuple[int, int], int | None] = {}

    def a(j: int, k: int, path: frozenset) -> int | None:
        if (j, k) in arrival:
            return arrival[(j, k)]
        if (j, k) in path:
            raise ValueError(f"naive's planned routes form a cycle through {inst.nodes[j].id}")
        node = inst.nodes[j]
        if (node.type == "sink" and (j, k) in demanded) or (node.grid is not None and k in node.grid.fuels):
            arrival[(j, k)] = T  # a point of use: sold at step 8, burned at step 7 of the arrival week
            return T
        if node.fab is not None and k == node.fab.input:
            out_k, lag = node.fab.product, node.fab.tau + 1  # start on arrival, out at + tau^fab, shipped a week later
        elif node.osat is not None and k in node.osat.packages:
            out_k, lag = node.osat.packages[k], node.osat.tau + 1  # packaged on arrival, out at + tau^osat, shipped
        else:
            out_k, lag = k, 1  # stock of week a leaves at a + 1 at the earliest (dispatch draws on I^{t-1}, (5))
        best = None
        for r in out.get((j, out_k), ()):
            ahead = a(r.dest, r.k, path | {(j, k)})
            if ahead is not None and (best is None or ahead - r.tau > best):
                best = ahead - r.tau
        arrival[(j, k)] = None if best is None else best - lag
        return arrival[(j, k)]

    slot = []
    for s, (e, k, lane) in enumerate(inst.action_slots):
        aj = a(_destination(inst, e, lane), k, frozenset())
        slot.append(None if aj is None else aj - _slot_transit(inst, s))
    for r in plan.routes:
        a(r.dest, r.k, frozenset())
    return LastUseful(T=T, arrival=arrival, slot=tuple(slot))


def phantom_pipeline(inst: Instance, plan: NaivePlan, last: LastUseful, week: int) -> tuple[float, ...]:
    """The cut's phantom pipeline of each plan route at the observation of ``week``, in route order ((72); Q109).

    Route l into (j, k), with W_l the last useful week (72) of its own action slot (first edge, k, lane) and tau_l its
    nominal transit, would carry in the pipeline at the instant t - 1 its dispatches of weeks t - tau_l .. t - 1; the
    cut sent none in the weeks after W_l, so the missing part is sum_{u = max(W_l + 1, t - tau_l, 1)}^{t - 1} d_lk
    m^sea(u). Weeks u <= 0 are the instance's initial pipeline, which the cut never touched; a slot with no useful week
    (W None) counts every week from 1. Stateless: t, T, Static's leads and the plan only, like the cut itself.
    """
    m_sea = {(dem.node, dem.k): dem for dem in inst.demands if dem.seasonal is not None}
    out = []
    for r in plan.routes:
        w = last.slot[inst.action_slot_index[(r.first_edge, r.k, r.lane)]]
        lo = max(1, week - r.tau) if w is None else max(w + 1, week - r.tau, 1)
        dem = m_sea.get((r.dest, r.k))
        weeks = range(lo, week)
        out.append(r.d * math.fsum(1.0 if dem is None else dem.m_sea(u) for u in weeks) if weeks else 0.0)
    return tuple(out)


def end_aware_targets(
    inst: Instance,
    plan: NaivePlan,
    last: LastUseful,
    week: int,
    targets: Mapping[tuple[int, int], float] | None = None,
) -> dict[tuple[int, int], float]:
    """S^end_jk(t) of (72): the week's targets less the cut's phantom pipeline of every plan route into (j, k) (Q109).

    ``targets`` are the week's S_jk (naive's own of step 3 when None; a naive-derived baseline's level otherwise);
    each route's ``phantom_pipeline`` is subtracted from its destination's target in route order. Without it the
    missing pipeline of a route the cut stops reads as a deficit that step 5 re-splits onto the destination's other
    routes, the week-103 loss on `full` (design §12 M5 row "Phantom-pipeline correction").
    """
    out = dict(_targets_at(inst, plan, week) if targets is None else targets)
    for r, phantom in zip(plan.routes, phantom_pipeline(inst, plan, last, week), strict=True):
        if phantom:
            out[(r.dest, r.k)] -= phantom
    return out


def end_aware_action(
    inst: Instance,
    plan: NaivePlan,
    last: LastUseful,
    obs: dict,
    *,
    targets: Mapping[tuple[int, int], float] | None = None,
    **kwargs,
) -> dict:
    """The end-aware naive's action: steps 4-6 on the targets of (72), less every flow past its last useful week.

    Step 5 orders up to ``end_aware_targets`` (the week's targets, naive's own or the given ``targets`` of a
    naive-derived baseline, less the cut's phantom pipeline; Q109), and the kept entries are those flows, in naive's
    order (module docstring, "End-aware cut"); ``kwargs`` are ``naive_action``'s thresholds.

    Raises:
        ValueError: if the week is not an integer, or as ``naive_action``.

    """
    week = obs["week"]
    if isinstance(week, bool) or not isinstance(week, int):
        raise ValueError(f"the end-aware naive needs an integer week, got {week!r}")
    action = naive_action(inst, plan, obs, targets=end_aware_targets(inst, plan, last, week, targets), **kwargs)
    fl = action["flows"]
    keep = [i for i, s in enumerate(fl["slot"]) if last.useful(s, week)]
    action["flows"] = {"slot": [fl["slot"][i] for i in keep], "qty": [fl["qty"][i] for i in keep]}
    return action


# (hash of the instance JSON, F_Q quantiles) -> the pickled instance and plan: the plan is a function of the instance,
# so episodes share the solve, but never the objects (bytes cannot be altered; each reset unpickles its own, DET-3)
_PLANS: dict[tuple, tuple[bytes, bytes]] = {}


def _cached_plan(static: dict, fq_quantile: dict | None) -> tuple[Instance, NaivePlan]:
    """A private copy of the instance of ``static`` (strict provenance) and its plan, solved once per instance hash.

    Every call hashes ``static["instance"]`` as received (the full instance JSON of §9.2) and keys the cache by that
    hash and the quantiles as floats (``canonical_quantiles``: equal values of other types share one entry and one
    plan, DET-P3-4), so the answer depends on the input only, never on what ran earlier in the process (V1): a body
    whose hash is not the declared ``static["instance_hash"]`` raises ValueError on every call, warm cache or cold (a
    Static without the declared hash is keyed alike), and so does a quantile ``canonical_quantiles`` refuses. The cache
    holds pickled bytes and each call returns freshly unpickled objects (floats round-trip exactly), so whatever a
    caller does to them, bypassing the frozen types included, reaches no other episode.

    Raises:
        TypeError: if ``static["instance"]`` is not an instance JSON object (a name or path would be keyed by its text),
            or a quantile is not a real number.
        ValueError: if the hash of the instance is not the declared ``instance_hash``, or a quantile is not finite or
            is negative.

    """
    body = static["instance"]
    if not isinstance(body, dict):
        raise TypeError(f"static instance must be the instance JSON object (§9.2), got {type(body).__name__}")
    h = sha256_hex(body)
    declared = static.get("instance_hash")
    if declared is not None and declared != h:
        raise ValueError(f"static instance_hash {str(declared)[:12]}... is not the hash of its instance {h[:12]}...")
    fq = canonical_quantiles(fq_quantile)  # before the key: keys compare by ==, so types must not differ (DET-P3-4)
    key = (h, tuple(sorted(fq.items())))
    if key not in _PLANS:
        inst = load_instance(body)
        if inst.hash != h:  # the loader hashes its own copy of the same JSON; a mismatch is a loader fault
            raise ValueError(f"loaded instance hash {inst.hash[:12]}... is not the hash of its JSON {h[:12]}...")
        inst_blob = pickle.dumps(inst, protocol=pickle.HIGHEST_PROTOCOL)  # before the solve builds memo tables on it
        _PLANS[key] = inst_blob, pickle.dumps(naive_plan(inst, fq), protocol=pickle.HIGHEST_PROTOCOL)
    inst_blob, plan_blob = _PLANS[key]
    return pickle.loads(inst_blob), pickle.loads(plan_blob)


class NaivePolicy:
    """`naive` as a ``Policy``: loads the instance from ``static["instance"]`` at reset (plan solved once per instance).

    Each reset receives private copies of the instance and plan from the cache, so a subclass that alters them changes
    only its own episode. By default (``end_aware=True``) it is the anchor, with the end-aware cut of the module
    docstring (Q98), its last useful weeks derived at reset from the instance and plan; ``end_aware=False`` is plain
    naive, named ``naive_plain`` (tests and evidence only). Thresholds off naive's own carry themselves in the name
    (``naive[closed_threshold=0.3]``, ``base.run_name``), which enters the trajectory hash (SPEC-M4R2-01). The name's
    base is the class's ``name``, so a subclass that declares its own keeps it (``<name>_plain`` when plain).
    """

    name = "naive"

    def __init__(
        self,
        closed_threshold: float = CLOSED_THRESHOLD,
        fq_quantile: dict | None = None,
        give_up_ratio: float = GIVE_UP_RATIO,
        end_aware: bool = True,
    ) -> None:
        """Keep the step-6 thresholds and a float copy of the F_Q quantiles, each refused outside its domain.

        ``check_thresholds`` raises ValueError; ``canonical_quantiles`` raises TypeError or ValueError (DET-P3-4). The
        copy is the policy's own, so a later edit of the caller's mapping changes nothing. ``end_aware`` must be a bool
        (ValueError otherwise).
        """
        check_thresholds(closed_threshold, give_up_ratio)
        if type(end_aware) is not bool:
            raise ValueError(f"end_aware must be a bool, got {end_aware!r}")
        self.closed_threshold = closed_threshold
        self.give_up_ratio = give_up_ratio
        self.fq_quantile = None if fq_quantile is None else frozen_map(canonical_quantiles(fq_quantile))
        self.end_aware = end_aware
        thresholds = {"closed_threshold": closed_threshold, "give_up_ratio": give_up_ratio}
        defaults = {"closed_threshold": CLOSED_THRESHOLD, "give_up_ratio": GIVE_UP_RATIO}
        self.name = run_name(type(self).name + ("" if end_aware else "_plain"), thresholds, defaults)
        self._inst: Instance | None = None
        self._plan: NaivePlan | None = None
        self._last: LastUseful | None = None

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Load the instance (strict provenance), its plan (solved once per instance) and, end-aware, (71)-(72)."""
        self._inst, self._plan = _cached_plan(static, self.fq_quantile)
        self._last = last_useful_weeks(self._inst, self._plan) if self.end_aware else None

    def act(self, obs: dict) -> dict:
        """Naive's action for the observed week, a function of ``obs`` and the plan (and its last useful weeks) only."""
        if self._inst is None or self._plan is None:
            raise RuntimeError("NaivePolicy.act called before reset")
        if self._last is not None:
            return end_aware_action(
                self._inst,
                self._plan,
                self._last,
                obs,
                closed_threshold=self.closed_threshold,
                give_up_ratio=self.give_up_ratio,
            )
        return naive_action(self._inst, self._plan, obs, self.closed_threshold, self.give_up_ratio)

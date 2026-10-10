"""Naive's finite-horizon backlog law F_{Q_cb} (66)-(67) from reset-time replications (design §8.1; Q54, Q84).

Q_{cb,t} = (Q_{cb,t-1} + d_cb - k_c mu_cb o^tr_{c,t})^+ (66), d_cb the naive plan's routed demand through c in pool b,
o^tr the open fraction (37) of the transient component only (transient militarised closures and weather closures,
severities included; the persistent component of (39) is excluded, Q30/Q54), from the episode's initial queue and the
stationary transient state. F_{Q_cb} is the empirical law of Q_{cb,t} pooled over t = 1..T of ``replications``
reset-time replications (design 1,000, SYNTHETIC, §11 row 38). One set of replications per rung serves every
(c, b) (owner queue M5-O21 (b), 2026-09-28): replication r samples the generator once with the entropy derived from
stream 22, key (instance kind, rung index, r), on the public root (outside omega; ``PUBLIC_ROOT``), and every pair
reads its chokepoint's closures from that one sample, so the pairs' paths are joint; naive is a function of the
instance and the generator only; the replications of rung gamma run the generator at gamma (``params`` itself, never
the anchor's). A replication draws only the closures it reads (``sampler.sample_closures``: the closures of
``sample_events``' stored list at every chokepoint, the same events in the same order, without W, strikes or other
types' marks).

**Tandem lanes** (§8.1, Q54; M5, design §12 M5 generator rows "Tandem lanes' F_Q" and "Shared stream-22
replications"): where a lane of the plan reaches chokepoint c from another chokepoint, the inflow at c is not constant,
so the pair (c, b) takes Q from a reset-time simulation of naive's own routed network in pool b (``pool_networks``,
``network_backlog``): (66) at every chokepoint of the pool, the week's inflow d_l at a lane's first chokepoint and,
downstream, the reset pipeline plus the upstream release of the lane after the edge's lead, releases leaving FIFO by
cohort and pro rata inside a cohort (§3.4). The network runs once per pool and replication on the shared sample's joint
closure paths (``pair_backlogs``); pairs without upstream inflow (``tandem_pairs``; all of `tiny`'s) keep (66).

Readings where the design leaves a choice (fixed here, reported for design §12):

- **Replication entropy**: ``SeedSequence(PUBLIC_ROOT, spawn_key=(0, 22, kind, rung, r))`` by (27) with the episode
  component 0 (stream 22 has no episode), its ``generate_state(2, uint64)`` read as one 128-bit integer (word 0 low);
  replication r runs the generator as ``sample_events(inst, params, that entropy, episode 0)``. The burn-in of (38)
  gives the stationary transient state, carried-in transient closures included (§4.3). Until M5-O21 the key also
  carried (c, pool), one set of replications per pair.
- **Transient closures** are the stored events of the sample that close chokepoint c: militarised closures whose
  persistent flag of (39) is off, and weather or accident closures; o^tr multiplies their (37) factors in stored order.
- **Pairs**: one law per (chokepoint, pool) that a lane of naive's plan passes (d_cb > 0), keyed (node, pool name);
  d_cb at m^sea = 1, as the plan's route demands. Q_{cb,0} is the exact sum of the reset lots at c whose commodity is
  in pool b; a carried-in closure starts from it (the empty-queue simplification of §4.3, Q87).
- **(68)**: at k = inf with full two-state closures, a replication started from the stationary queue (d times the
  stationary closure age) pools to (68)'s F, so (67) equals (68) (V6-V8, the T -> inf law); started from the episode's
  empty queue, the finite-horizon law counts fewer closed weeks and sits below it.
- **Critical ratio** pi / (pi + h) of a route into (j, k), with h = h_jk the destination's holding and pi its penalty:
  sink pi_jk; VOLL_g at a grid; at a fab (its wafers) or an OSAT (its raw chips) the pi of the packaged chip they make
  ("one wafer makes one chip", as `greedy_lp`, §8.2), the largest over the sinks that demand it; at a terminal the
  largest VOLL of the grids its edges feed. Any other destination raises.
- **Quantile**: F^{-1}(p) = the ceil(p n)-th smallest of n pooled samples, with p n evaluated exactly on the decimal
  ``repr`` of p (as the cents of (24)), so 0.07 x 100 is 7, not the 8 of a float product.
- **Load (69)**: varrho_cb = d_cb / (k_c mu_cb E[o^tr_c]) per pair (``naive_load``), E[o^tr_c] the exact mean of
  o^tr_{c,t} over t = 1..T and the replications that F_Q pools (the two pools of one chokepoint share the rung's
  replications, so they share E[o^tr_c]); ``fq_laws_and_load`` gives both from one pass. A chokepoint
  that never opens has load inf; varrho_cb >= 1 (``OVERLOADED``) flags an overloaded rung (§8.1).
- **Once per generator**: ``generator_quantiles`` computes ``critical_quantiles(inst, fq_laws(inst, params))`` once per
  process for each (instance family digest, ``generator_id``, replication count) and keeps it in an immutable map, so
  naive stays a function of the instance and the generator only (§8.1). The key is the instance's *content* at its
  file's own rung (``Instance.family_digest``: V3's ``Instance.content_digest``, which covers its hash label too, with
  the on-hand stock of the file's own rung, since F_Q reads no stock and the instances of one file at every rung share
  it, §12 'Warm start per rung'), never the label alone, which ``dataclasses.replace`` carries over to a variant of
  other content (DET-M2-3); the replication count is validated before the key is built,
  so ``True`` or ``1.0`` never reads the entry of 1 (DET-M2-4). ``remember_quantiles`` is the one trusted seam that
  seeds that map with quantiles computed elsewhere, the scripts layer's disk cache, whose key binds the same triple
  and the package source; it refuses a map that ``generator_quantiles`` could not have returned (keys other than
  ``quantile_keys``, values that are not floats >= 0) and one that differs from the entry the process already holds
  (REG-3). ``anchor_policy`` gives the anchor of (55) and ``fallback_spec`` the D9 fallback of an episode: with the
  generator's quantiles when omega is generated, the point mass at 0 (``params=None``) on injected lists; both are the
  end-aware naive (``policies.naive``, Q98) unless ``end_aware=False`` asks for plain naive. The end-aware cut reads T,
  Static's leads and the plan's routes, which F_Q does not change; the laws, quantiles and load here serve both. The
  fallback (``NaiveFallback``) carries the instance's content digest as well as ``generator_id``, so ``Env`` refuses it
  on an instance of other content that keeps the hash label (REG-M2R2-01).
"""

import functools
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from shockbench_flow.disruption.params import RUNGS, GeneratorParams, generator_id  # RUNGS: §2.6, re-exported
from shockbench_flow.disruption.strata import inverse_cdf
from shockbench_flow.instance.schema import POOLS, Instance, frozen_map
from shockbench_flow.marks import K_CHOKEPOINT, MILITARISED_CLOSURE, WEATHER_CLOSURE, open_fraction
from shockbench_flow.omega import codes, seeds
from shockbench_flow.parallel import ordered_map
from shockbench_flow.policies.naive import NaiveFallback, NaivePolicy, canonical_quantiles, lane_demand, naive_plan


PUBLIC_ROOT = 0  # naive's replications are public and reproducible by anyone (§8.1); never a split's entropy
REPLICATION_EPISODE = 0  # the episode component of the stream-22 key (27), and the episode each replication samples
REPLICATIONS = 1000  # T-week replications per (chokepoint, pool, rung), SYNTHETIC (§8.1, §2.4 profile, §11 row 38)


# ----- keys (27), stream 22 ----------------------------------------------------------------------------------------
def rung_index(params: GeneratorParams) -> int:
    """The index of the policy-block rung gamma in ``RUNGS`` (§2.6), the second component of the stream-22 key.

    Raises:
        ValueError: if gamma is not one of the rungs.

    """
    gamma = params.hawkes.gamma
    if gamma not in RUNGS:
        raise ValueError(f"stream 22 keys a rung of {RUNGS}; gamma {gamma!r} is none of them")
    return RUNGS.index(gamma)


def stream_key(inst: Instance, params: GeneratorParams, r: int) -> tuple[int, ...]:
    """The stream-22 key (instance kind, rung index, replication) of §4.1: one set per rung for every (c, b), M5-O21."""
    return (codes.INSTANCE_KINDS.index(inst.kind), rung_index(params), r)


def replication_entropy(inst: Instance, params: GeneratorParams, r: int) -> int:
    """The 128-bit generator entropy of replication r of the rung, from stream 22 on ``PUBLIC_ROOT`` (27)."""
    ss = seeds.seed_sequence(PUBLIC_ROOT, REPLICATION_EPISODE, codes.STREAM_NAIVE_FQ, stream_key(inst, params, r))
    words = ss.generate_state(2, np.uint64)
    return int(words[0]) | (int(words[1]) << 64)


# ----- (37) restricted to the transient component, and (66) --------------------------------------------------------
def _transient(ev, c: int) -> bool:
    """A stored event that closes chokepoint c in the transient component: weather, or a non-persistent closure."""
    if ev.target_kind != K_CHOKEPOINT or ev.target != c:
        return False
    return ev.type == WEATHER_CLOSURE or (ev.type == MILITARISED_CLOSURE and not ev.persistent)


def transient_open(events, c: int, T: int) -> np.ndarray:
    """o^tr_{c,t} for t = 1..T: the product of (37)'s 1 - sigma_q f^t_q over the transient closures of c, in order.

    Read from the marks module (``marks.open_fraction``, the product ``compute_marks`` forms for a chokepoint's o, one
    rule table with V3; SIMP-M2R2-02), so on the same closures o^tr equals the marks' o bit for bit.
    """
    return open_fraction(T, [(ev.onset, ev.onset + ev.duration, ev.severity) for ev in events if _transient(ev, c)])


def backlog_path(q0, d: float, kappa: float, open_frac: np.ndarray) -> np.ndarray:
    """Q_{cb,t} of (66) for t = 1..T: (Q_{t-1} + d - kappa o_t)^+ from Q_0 = ``q0``, kappa = k_c mu_cb.

    ``open_frac`` holds o^tr over its last axis (T weeks); leading axes are independent replications, ``q0`` broadcast
    over them. A fully closed week serves nothing, also at kappa = inf (no inf x 0).

    Raises:
        ValueError: if q0 or d is negative or not finite, kappa is NaN or negative, or an open fraction lies
            outside [0, 1].

    """
    o = np.asarray(open_frac, dtype=np.float64)
    q = np.array(q0, dtype=np.float64)
    if not (np.all(np.isfinite(q)) and np.all(q >= 0.0)):
        raise ValueError("(66) needs a finite initial queue >= 0")
    if not (math.isfinite(d) and d >= 0.0):
        raise ValueError(f"(66) needs a finite routed demand d_cb >= 0, got {d!r}")
    if not kappa >= 0.0:  # NaN fails too
        raise ValueError(f"(66) needs a capacity k_c mu_cb >= 0, got {kappa!r}")
    if not (np.all(o >= 0.0) and np.all(o <= 1.0)):
        raise ValueError("(66) needs open fractions in [0, 1]")
    q = np.broadcast_to(q, o.shape[:-1]).copy()
    out = np.empty(o.shape, dtype=np.float64)
    for t in range(o.shape[-1]):
        ot = o[..., t]
        served = np.zeros_like(ot)
        np.multiply(kappa, ot, out=served, where=ot > 0.0)
        q = np.maximum(q + d - served, 0.0)
        out[..., t] = q
    return out


def check_replications(replications: object) -> int:
    """``replications`` as an int if it is an integer >= 1: the one check of every F_Q entry point (DET-M2-4).

    Raises:
        TypeError: if it is not an integer (a bool, a float such as 1.0 or a str included).
        ValueError: if it is < 1.

    """
    if isinstance(replications, (bool, np.bool_)) or not isinstance(replications, numbers.Integral):
        raise TypeError(f"replications must be an integer, got {replications!r}")
    if replications < 1:
        raise ValueError(f"replications must be >= 1, got {replications}")
    return int(replications)


# ----- the plan's inputs to (66) -----------------------------------------------------------------------------------
def initial_queue(inst: Instance, c: int, pool: str) -> float:
    """Q_{cb,0}: the episode's initial queue at c in pool b, the reset lots of §2.3 (exact sum)."""
    return math.fsum(
        lot.qty for lot in inst.initial_state.queue_lots if lot.chokepoint == c and inst.commodities[lot.k].pool == pool
    )


def routed_demand(inst: Instance) -> dict[tuple[int, str], float]:
    """d_cb of (66) per (chokepoint node, pool name) that a lane of naive's plan passes (route demands at m^sea = 1)."""
    plan = naive_plan(inst)
    return {key: d for key, d in lane_demand(inst, plan.routes, [r.d for r in plan.routes]).items() if d > 0.0}


# ----- tandem lanes (§8.1; Q54): a reset-time simulation of naive's own routed network -----------------------------
@dataclass(frozen=True)
class TandemLane:
    """One lane of naive's plan in one pool: its route demand and path, for the tandem simulation.

    ``d`` is d_l = sum of the plan's route demands d_lk on the lane over k in the pool (m^sea = 1), the lane's part of
    d_cb at each of its chokepoints (66); ``legs[i]`` is the lead of the edge from ``chokepoints[i]`` to
    ``chokepoints[i + 1]`` (a lane's interior nodes are chokepoints, so consecutive ones share one edge).
    """

    lane: int
    d: float
    chokepoints: tuple[int, ...]
    legs: tuple[int, ...]
    leg_edges: tuple[int, ...]


@dataclass(frozen=True)
class PoolNetwork:
    """Naive's routed network of one pool: the plan's lanes through its chokepoints and the reset state on them."""

    pool: str
    chokepoints: tuple[int, ...]  # every chokepoint a lane of the pool passes, node order
    lanes: tuple[TandemLane, ...]  # lane order
    kappa: Mapping[int, float]  # k_c mu_cb of the pool per chokepoint
    queue: Mapping[tuple[int, int], float]  # (lane, chokepoint) -> the reset queue lots of the pool (exact sum)
    pipeline: Mapping[tuple[int, int, int], float]  # (lane, chokepoint, week >= 1) -> reset shipments arriving there

    def upstream(self, c: int) -> bool:
        """Whether some lane of the pool reaches c from another chokepoint (its inflow then follows a release)."""
        return any(c in ln.chokepoints[1:] for ln in self.lanes)


def pool_networks(inst: Instance) -> Mapping[str, PoolNetwork]:
    """The tandem networks of naive's plan per pool name, for the pools whose lanes pass a chokepoint (§8.1; Q54).

    From ``naive_plan``'s lane routes: d_l per (lane, pool), in route order as ``lane_demand`` sums d_cb; each lane's
    chokepoints in path order and the leads of the edges between them; the reset queue lots per (lane, chokepoint) and
    the reset pipeline on those edges (``InitialShipment`` labelled with the lane, arriving at the next chokepoint in
    week >= 1), commodities of the pool only.

    Raises:
        ValueError: if an edge between two chokepoints of a lane has a lead below 1 week (a release would reach the next
            queue in its own week, which the week-by-week simulation cannot order; §12 'Routing and lead times').

    """
    plan = naive_plan(inst)
    demand: dict[tuple[int, str], float] = {}
    for r in plan.routes:
        if r.lane is not None and inst.lanes[r.lane].chokepoints:
            key = (r.lane, inst.commodities[r.k].pool)
            demand[key] = demand.get(key, 0.0) + r.d
    out: dict[str, PoolNetwork] = {}
    for pool in POOLS:
        lanes = []
        for (li, p), d in sorted(demand.items()):
            if p != pool or d <= 0.0:
                continue
            lane = inst.lanes[li]
            legs, leg_edges = [], []
            for a, b in zip(lane.chokepoints, lane.chokepoints[1:]):
                e = next(e for e in lane.edges if inst.edges[e].tail == a and inst.edges[e].head == b)
                if inst.edges[e].tau < 1:
                    raise ValueError(
                        f"tandem lane {lane.id}: the edge {inst.nodes[a].id} -> {inst.nodes[b].id} has lead 0"
                    )
                legs.append(int(inst.edges[e].tau))
                leg_edges.append(e)
            lanes.append(TandemLane(li, d, tuple(lane.chokepoints), tuple(legs), tuple(leg_edges)))
        if not lanes:
            continue
        chks = tuple(sorted({c for ln in lanes for c in ln.chokepoints}))
        in_pool = {k for k, com in enumerate(inst.commodities) if com.pool == pool}
        queue: dict[tuple[int, int], list[float]] = {}
        for lot in inst.initial_state.queue_lots:
            if lot.k in in_pool:
                queue.setdefault((lot.lane, lot.chokepoint), []).append(lot.qty)
        pipe: dict[tuple[int, int, int], list[float]] = {}
        heads = {(ln.lane, e): inst.edges[e].head for ln in lanes for e in ln.leg_edges}
        for sh in inst.initial_state.pipeline:
            c = heads.get((sh.lane, sh.edge))
            if c is not None and sh.k in in_pool and sh.arrival_week >= 1:
                pipe.setdefault((sh.lane, c, sh.arrival_week), []).append(sh.qty)
        out[pool] = PoolNetwork(
            pool=pool,
            chokepoints=chks,
            lanes=tuple(lanes),
            kappa=frozen_map({c: inst.nodes[c].chokepoint.kappa0[POOLS.index(pool)] for c in chks}),
            queue=frozen_map({key: math.fsum(v) for key, v in sorted(queue.items())}),
            pipeline=frozen_map({key: math.fsum(v) for key, v in sorted(pipe.items())}),
        )
    return frozen_map(out)


def tandem_pairs(inst: Instance) -> frozenset[tuple[int, str]]:
    """The (chokepoint node, pool name) pairs whose inflow follows an upstream release: F_Q from the network (§8.1).

    Every other pair keeps the constant-inflow recursion (66), bit for bit (`tiny`, one chokepoint, has none).
    """
    return frozenset((c, pool) for pool, net in pool_networks(inst).items() for c in net.chokepoints if net.upstream(c))


def _release_fifo(cohorts: list[dict[int, float]], amount: float) -> dict[int, float]:
    """Release ``amount`` from a chokepoint's queue, cohorts FIFO and pro rata inside a cohort (§3.4); per lane.

    ``cohorts`` (oldest first, each lane -> qty) is consumed in place: whole cohorts while ``amount`` covers them, then
    a share of the next; the per-lane releases are returned.
    """
    out: dict[int, float] = {}
    left = amount
    while cohorts and left > 0.0:
        head = cohorts[0]
        total = math.fsum(head.values())
        if total <= left:
            for lane, q in head.items():
                out[lane] = out.get(lane, 0.0) + q
            left -= total
            cohorts.pop(0)
            continue
        f = left / total
        for lane, q in head.items():
            out[lane] = out.get(lane, 0.0) + f * q
            head[lane] = q * (1.0 - f)
        left = 0.0
    return out


def network_backlog(net: PoolNetwork, opens: Mapping[int, np.ndarray], T: int) -> dict[int, np.ndarray]:
    """Q_{cb,1..T} at every chokepoint of the pool's network: (66) with the inflow of naive's own routed network.

    Week by week, at each chokepoint c (node order): the week's inflow is, per lane through c, d_l when c is the lane's
    first chokepoint (naive's steady dispatch, as (66)), else the reset pipeline arriving that week plus what the
    previous chokepoint of the lane released ``legs`` weeks before; Q_t = (Q_{t-1} + A_t - k_c mu_cb o^tr_{c,t})^+ (66)
    with A_t the ``math.fsum`` of that inflow; the release Q_{t-1} + A_t - Q_t leaves FIFO by arrival cohort and pro
    rata inside a cohort (§3.4, the lane share of §8.1), each lane's part travelling on to its next chokepoint. Leads
    between chokepoints are >= 1 week (``pool_networks``), so every inflow is known when its week is served. A fully
    closed week serves nothing, also at k = inf. ``opens[c]`` is o^tr_{c,1..T} (``transient_open``).
    """
    through: dict[int, list[tuple[TandemLane, int]]] = {c: [] for c in net.chokepoints}
    for ln in net.lanes:
        for i, c in enumerate(ln.chokepoints):
            through[c].append((ln, i))
    arrivals: dict[tuple[int, int, int], float] = dict(net.pipeline)
    Q = {c: math.fsum(q for (_, cc), q in net.queue.items() if cc == c) for c in net.chokepoints}
    cohorts: dict[int, list[dict[int, float]]] = {c: [] for c in net.chokepoints}
    for (lane, c), q in net.queue.items():
        if c in cohorts and q > 0.0:
            if not cohorts[c]:
                cohorts[c].append({})
            cohorts[c][0][lane] = cohorts[c][0].get(lane, 0.0) + q
    out = {c: np.empty(T, dtype=np.float64) for c in net.chokepoints}
    for t in range(1, T + 1):
        for c in net.chokepoints:
            parts: dict[int, float] = {}
            for ln, i in through[c]:
                a = ln.d if i == 0 else arrivals.pop((ln.lane, c, t), 0.0)
                if a > 0.0:
                    parts[ln.lane] = parts.get(ln.lane, 0.0) + a
            A = math.fsum(parts.values())
            o = float(opens[c][t - 1])
            served = net.kappa[c] * o if o > 0.0 else 0.0
            q_new = max(Q[c] + A - served, 0.0)
            if parts:
                cohorts[c].append(parts)
            if q_new == 0.0:
                released = {}
                for coh in cohorts[c]:
                    for lane, q in coh.items():
                        released[lane] = released.get(lane, 0.0) + q
                cohorts[c].clear()
            else:
                released = _release_fifo(cohorts[c], Q[c] + A - q_new)
            for ln, i in through[c]:
                q = released.get(ln.lane, 0.0)
                if q > 0.0 and i + 1 < len(ln.chokepoints):
                    key = (ln.lane, ln.chokepoints[i + 1], t + ln.legs[i])
                    arrivals[key] = arrivals.get(key, 0.0) + q
            Q[c] = q_new
            out[c][t - 1] = q_new
    return out


Pair = tuple[int, str, float, float, float]  # (c, pool, d_cb, kappa = k_c mu_cb, Q_{cb,0}) of (66)


def pairs_of(inst: Instance) -> list[Pair]:
    """(c, pool, d_cb, k_c mu_cb, Q_{cb,0}) for every (chokepoint, pool) a lane of naive's plan passes, sorted."""
    return [
        (c, pool, d_cb, inst.nodes[c].chokepoint.kappa0[POOLS.index(pool)], initial_queue(inst, c, pool))
        for (c, pool), d_cb in sorted(routed_demand(inst).items())
    ]


def replication_opens(inst: Instance, params: GeneratorParams, r: int, sampler=None) -> dict[int, np.ndarray]:
    """o^tr_{c,1..T} of replication r at every chokepoint a lane of naive's plan passes, from the rung's one sample.

    The sample is ``sampler(inst, params, replication_entropy(inst, params, r), REPLICATION_EPISODE)`` (default
    ``sampler.sample_closures``): one draw per replication, read by every (c, b) (owner queue M5-O21 (b)).
    """
    if sampler is None:  # imported here, so the policies <-> disruption imports form no cycle
        from shockbench_flow.disruption.sampler import sample_closures as sampler
    s = sampler(inst, params, replication_entropy(inst, params, r), REPLICATION_EPISODE)
    return {c: transient_open(s.stored, c, inst.T) for c in sorted({c for c, _pool in routed_demand(inst)})}


def pair_backlogs(inst: Instance, pairs: list[Pair], opens: Mapping[int, np.ndarray]) -> list[np.ndarray]:
    """Q_{cb,1..T} of every pair on one replication's joint open fractions ``opens`` (c -> o^tr_{c,1..T}).

    A pair of ``tandem_pairs`` reads Q at c from ``network_backlog`` of its pool's network (run once per pool) on the
    open fractions of every chokepoint of that network (their joint paths; §8.1, Q54); every other pair runs the
    recursion (66) from Q_{cb,0}.
    """
    tandem = tandem_pairs(inst)
    nets = pool_networks(inst) if tandem else {}
    runs: dict[str, dict[int, np.ndarray]] = {}
    out = []
    for c, pool, d_cb, kappa, q0 in pairs:
        if (c, pool) in tandem:
            if pool not in runs:
                net = nets[pool]
                runs[pool] = network_backlog(net, {cc: opens[cc] for cc in net.chokepoints}, inst.T)
            out.append(runs[pool][c])
        else:
            out.append(backlog_path(q0, d_cb, kappa, opens[c]))
    return out


def _replicate(
    inst: Instance, params: GeneratorParams, pairs: list[Pair], sampler, r: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(Q_{cb,1..T}, o^tr_{c,1..T}) of replication r for every pair (c, pool, d_cb, kappa, q0) in ``pairs``.

    Replication r runs the generator once at ``params``, the rung's own (the stream-22 key carries its index), and every
    pair reads that sample (``replication_opens``, ``pair_backlogs``; owner queue M5-O21 (b)).
    """
    opens = replication_opens(inst, params, r, sampler)
    return list(zip(pair_backlogs(inst, pairs, opens), (opens[c] for c, *_ in pairs)))


def laws_from_paths(inst: Instance, paths: Mapping[tuple[int, str], np.ndarray]) -> dict[tuple[int, str], np.ndarray]:
    """F_{Q_cb} of (66)-(67) from stored transient paths, drawing nothing: ``fq_laws`` of the replications they hold.

    ``paths`` maps every pair (c, pool) of naive's plan to an (R, T) array whose row r is o^tr_{c,1..T} of replication
    r (the paths ``sz_state_base_stock.transient_paths`` keeps, the two pools of one chokepoint equal row for row); the
    laws are ``pair_backlogs`` on each row's joint paths, pooled and sorted, which is ``fq_laws`` bit for bit when the
    paths are the draws behind it.

    Raises:
        ValueError: if the keys are not the pairs of naive's plan, the arrays are not (R, T) of one shape, or the two
            pools of one chokepoint carry different paths (not one shared replication per row).

    """
    pairs = pairs_of(inst)
    if set(paths) != {(c, pool) for c, pool, *_ in pairs}:
        raise ValueError("F_Q paths must be keyed by the (chokepoint, pool) pairs of naive's plan")
    shapes = {np.shape(a) for a in paths.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[1] != inst.T:
        raise ValueError(f"F_Q paths must be (R, T) arrays of one shape, got {sorted(shapes)}")
    by_c: dict[int, np.ndarray] = {}
    for (c, _pool), arr in sorted(paths.items()):
        a = np.asarray(arr, dtype=np.float64)
        if c in by_c and not np.array_equal(by_c[c], a):
            raise ValueError(f"the pools of chokepoint {inst.nodes[c].id} carry different paths (M5-O21: one sample)")
        by_c.setdefault(c, a)
    R = next(iter(shapes))[0]
    per_rep = [pair_backlogs(inst, pairs, {c: a[r] for c, a in by_c.items()}) for r in range(R)]
    return {
        (c, pool): np.sort(np.concatenate([rep[i] for rep in per_rep]), kind="stable")
        for i, (c, pool, *_) in enumerate(pairs)
    }


OVERLOADED = 1.0  # varrho_cb >= 1 flags an overloaded rung (§8.1 after (69))


def load_ratio(d_cb: float, kappa: float, open_mean: float) -> float:
    """The load (69) varrho_cb = d_cb / (k_c mu_cb E[o^tr_c]), with kappa = k_c mu_cb and ``open_mean`` = E[o^tr_c].

    A chokepoint that never opens (E[o^tr] = 0) serves nothing, also at kappa = inf (as (66): no inf x 0), so its load
    is inf; a zero capacity gives inf too, and kappa = inf with E[o^tr] > 0 gives 0.

    Raises:
        ValueError: if d_cb or ``open_mean`` is not finite or is negative, ``open_mean`` exceeds 1, or kappa is NaN or
            negative.

    """
    if not (math.isfinite(d_cb) and d_cb >= 0.0):
        raise ValueError(f"(69) needs a finite routed demand d_cb >= 0, got {d_cb!r}")
    if not kappa >= 0.0:  # NaN fails too
        raise ValueError(f"(69) needs a capacity k_c mu_cb >= 0, got {kappa!r}")
    if not 0.0 <= open_mean <= 1.0:
        raise ValueError(f"(69) needs a mean open fraction in [0, 1], got {open_mean!r}")
    served = kappa * open_mean if open_mean > 0.0 else 0.0
    return d_cb / served if served > 0.0 else math.inf


def fq_laws_and_load(
    inst: Instance,
    params: GeneratorParams,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    *,
    sampler: Callable | None = None,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[tuple[int, str], float]]:
    """F_{Q_cb} of (66)-(67) and the load (69) per (chokepoint node, pool name), from one pass over the replications.

    Args:
        inst: the instance.
        params: the public generator at the rung (``params.hawkes.gamma`` one of ``RUNGS``).
        replications: T-week replications per (c, b) (design 1,000, SYNTHETIC, §11 row 38).
        n_jobs: joblib workers over the replications (imported only when not 1); nothing returned depends on it.
        sampler: ``(inst, params, entropy, episode)`` returning an object with ``stored`` (whose transient closures
            are read); default ``sampler.sample_closures``, which gives the closures of ``sampler.sample_events``.

    Returns:
        (laws, load): per pair (c, pool) a lane of naive's plan passes, the T x ``replications`` pooled values of
        Q_{cb,t}, sorted ascending, and ``load_ratio(d_cb, k_c mu_cb, E[o^tr_c])`` with E[o^tr_c] the exact mean
        (``math.fsum``) of o^tr_{c,t} over t = 1..T and the same replications of (c, b) that F_Q pools.

    Raises:
        TypeError: if ``replications`` is not an integer (a bool included).
        ValueError: if ``replications`` < 1 or gamma is not a rung.

    """
    replications = check_replications(replications)
    rung_index(params)  # refuse an unkeyed rung before any draw
    if sampler is None:
        from shockbench_flow.disruption.sampler import sample_closures as sampler
    pairs = pairs_of(inst)
    per_rep = ordered_map(functools.partial(_replicate, inst, params, pairs, sampler), range(replications), n_jobs)
    laws, load = {}, {}
    for i, (c, pool, d_cb, kappa, _q0) in enumerate(pairs):
        laws[(c, pool)] = np.sort(np.concatenate([rep[i][0] for rep in per_rep]), kind="stable")
        opens = np.concatenate([rep[i][1] for rep in per_rep])
        load[(c, pool)] = load_ratio(d_cb, kappa, math.fsum(opens.tolist()) / opens.size)
    return laws, load


def fq_laws(
    inst: Instance,
    params: GeneratorParams,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    *,
    sampler: Callable | None = None,
) -> dict[tuple[int, str], np.ndarray]:
    """F_{Q_cb} as sorted samples per (chokepoint node, pool name), pooled over weeks and replications (66).

    The laws of ``fq_laws_and_load`` (its arguments, errors and pairs): (c, pool) -> the T x ``replications`` pooled
    values of Q_{cb,t}, sorted ascending, for every pair a lane of naive's plan passes.
    """
    return fq_laws_and_load(inst, params, replications, n_jobs, sampler=sampler)[0]


def naive_load(
    inst: Instance,
    params: GeneratorParams,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    *,
    sampler: Callable | None = None,
) -> dict[tuple[int, str], float]:
    """Naive's load (69) per (chokepoint node, pool name) at the rung of ``params``, published per instance and rung.

    The load of ``fq_laws_and_load`` (its arguments, errors and pairs), from the same stream-22 replications as F_Q;
    a value >= ``OVERLOADED`` flags an overloaded rung (§8.1), where naive stays defined because the cover cap binds.
    """
    return fq_laws_and_load(inst, params, replications, n_jobs, sampler=sampler)[1]


# ----- the critical ratio and the quantiles of (67) ----------------------------------------------------------------
def _packaged_pi(inst: Instance, packaged: set[int]) -> float:
    pis = [dem.pi for dem in inst.demands if dem.k in packaged]
    if not pis:
        raise ValueError("no sink demands the packaged chip this destination makes")
    return max(pis)


def critical_ratio(inst: Instance, j: int, k: int) -> float:
    """The critical ratio pi / (pi + h) of (67) for destination (j, k): h = h_jk, pi its penalty (module docstring).

    Raises:
        ValueError: if node j has no stock of k, or its type has no penalty rule (a source, material or chokepoint).

    """
    node = inst.nodes[j]
    slot = inst.slot_index.get((j, k))
    if slot is None:
        raise ValueError(f"{node.id} holds no {inst.commodities[k].id}")
    h = inst.stock_slots[slot].holding
    if node.type == "sink":
        pis = [dem.pi for dem in inst.demands if dem.node == j and dem.k == k]
        if not pis:
            raise ValueError(f"sink {node.id} has no demand for {inst.commodities[k].id}")
        pi = pis[0]
    elif node.type == "grid" and k in node.grid.fuels:
        pi = node.grid.voll
    elif node.type == "fab" and k == node.fab.input:
        raw = node.fab.product
        pi = _packaged_pi(
            inst, {inst.nodes[o].osat.packages[raw] for o in inst.osats if raw in inst.nodes[o].osat.packages}
        )
    elif node.type == "osat" and k in node.osat.packages:
        pi = _packaged_pi(inst, {node.osat.packages[k]})
    elif node.type == "terminal":
        volls = [
            inst.nodes[inst.edges[e].head].grid.voll for e in inst.out_edges[j] if inst.nodes[inst.edges[e].head].grid
        ]
        if not volls:
            raise ValueError(f"terminal {node.id} feeds no grid")
        pi = max(volls)
    else:
        raise ValueError(f"(67) has no penalty for {inst.commodities[k].id} into {node.type} {node.id}")
    return pi / (pi + h)


def _route_keys(inst: Instance):
    """(c, pool, j, k) for every chokepoint c of every lane route of naive's plan into (j, k), in plan order."""
    for r in naive_plan(inst).routes:
        if r.lane is None:
            continue
        pool = inst.commodities[r.k].pool
        for c in inst.lanes[r.lane].chokepoints:
            yield c, pool, r.dest, r.k


def critical_quantiles(inst: Instance, laws: Mapping[tuple[int, str], np.ndarray]) -> dict[tuple, float]:
    """Naive's ``fq_quantile``: F^{-1}_{Q_cb}(pi/(pi+h)) per (c, pool, j, k) of every plan lane through c in pool b.

    ``laws`` are sorted samples per (chokepoint node, pool name), as ``fq_laws`` returns; a pair without a law is left
    out (its closure term is then 0).
    """
    out: dict[tuple, float] = {}
    for c, pool, j, k in _route_keys(inst):
        law = laws.get((c, pool))
        if law is not None:
            out[(c, pool, j, k)] = quantile(law, critical_ratio(inst, j, k))
    return out


def quantile_keys(inst: Instance) -> frozenset[tuple]:
    """The keys (c, pool, j, k) of ``generator_quantiles``: the plan's lane routes through a pair with a law of (66).

    ``fq_laws`` returns one law per (c, pool) that a lane of naive's plan passes (``routed_demand``), so these are the
    keys ``critical_quantiles`` gives on its laws, for every generator.
    """
    pairs = routed_demand(inst)
    return frozenset(key for key in _route_keys(inst) if key[:2] in pairs)


def quantile(law: np.ndarray, p: float) -> float:
    """F^{-1}(p) = the smallest sample x with F(x) >= p (empirical inverse CDF of a sorted sample); 0 when empty.

    The one quantile rule of the package, ``strata.inverse_cdf``: the index is ceil(p n) - 1 with p n exact on the
    decimal ``repr`` of p (module docstring), clamped to the sample.
    """
    return 0.0 if len(law) == 0 else inverse_cdf(law, p)


# ----- once per (instance, generator): the anchor of (55) and the D9 fallback ---------------------------------------
_QUANTILES: dict[tuple[str, str, int], Mapping[tuple, float]] = {}


def _cache_key(inst: Instance, params: GeneratorParams, replications: int) -> tuple[str, str, int]:
    """(instance family digest, ``generator_id``, replications): what the F_Q quantiles are a function of (§8.1).

    The content digest at the file's own rung (``Instance.family_digest``: F_Q reads no stock, so every rung's instance
    of one file shares it; §12 'Warm start per rung'), not the hash label a ``dataclasses.replace`` copy carries over
    (DET-M2-3); ``replications`` is already checked (``check_replications``), so ``True`` or ``1.0`` never stands for 1
    (DET-M2-4).
    """
    return inst.family_digest, generator_id(params, inst), replications


def generator_quantiles(
    inst: Instance, params: GeneratorParams, replications: int = REPLICATIONS, n_jobs: int = 1
) -> Mapping[tuple, float]:
    """Naive's F_Q quantiles for one generator: ``critical_quantiles(inst, fq_laws(inst, params, replications))``.

    Computed once per process and kept, as an immutable map of floats (``canonical_quantiles``), under the key
    (instance content digest, ``generator_id(params, inst)``, replications): naive is a function of the instance and
    the generator only (§8.1), and the replication count is part of the estimate (1,000 by design). ``replications``
    is validated before the key is built. ``n_jobs`` only spreads the work (the laws do not depend on it). A call that
    raises caches nothing.

    Raises:
        TypeError: if ``replications`` is not an integer (a bool or a float included), in a fresh process or not.
        ValueError: if ``replications`` < 1 or gamma is not a rung (``fq_laws``).

    """
    key = _cache_key(inst, params, check_replications(replications))
    if key not in _QUANTILES:
        _QUANTILES[key] = _quantile_map(inst, fq_laws(inst, params, key[2], n_jobs))
    return _QUANTILES[key]


def _quantile_map(inst: Instance, laws: Mapping[tuple[int, str], np.ndarray]) -> Mapping[tuple, float]:
    """The immutable map of floats ``generator_quantiles`` keeps: ``critical_quantiles`` of the laws, as floats."""
    return frozen_map(canonical_quantiles(critical_quantiles(inst, laws)))


def generator_quantiles_and_load(
    inst: Instance, params: GeneratorParams, replications: int = REPLICATIONS, n_jobs: int = 1
) -> tuple[Mapping[tuple, float], Mapping[tuple[int, str], float]]:
    """``generator_quantiles`` and ``naive_load`` of one generator from one pass over its replications.

    The quantiles enter the per-process cache through ``remember_quantiles``, so an entry the process already holds is
    returned and must equal them (the same draws); the load, a diagnostic that naive never reads, is not kept.

    Raises:
        TypeError, ValueError: as ``fq_laws_and_load``, and ``remember_quantiles`` if the process holds other quantiles
            under the key.

    """
    laws, load = fq_laws_and_load(inst, params, check_replications(replications), n_jobs)
    return remember_quantiles(inst, params, replications, dict(_quantile_map(inst, laws))), frozen_map(load)


def remember_quantiles(
    inst: Instance, params: GeneratorParams, replications: int, quantiles: Mapping[tuple, float]
) -> Mapping[tuple, float]:
    """Seed ``generator_quantiles``' per-process cache with the F_Q quantiles of one generator computed elsewhere.

    The trusted seam of the scripts layer's F_Q disk cache (``sbf_boundary.fq_quantiles``), whose key binds the same
    instance content, ``generator_id`` and replications plus the package source and library versions: the caller
    vouches that the values are ``generator_quantiles(inst, params, replications)``, which are not recomputed here.
    What a call can check without recomputing, it checks (REG-3): the keys must be exactly ``quantile_keys(inst)`` (a
    missing key would read as 0), every value a float (the type ``generator_quantiles`` returns and the disk cache
    stores), finite and >= 0, and an entry the process already holds is never replaced by other values. Kept as
    ``generator_quantiles`` keeps its own, an immutable map of floats under (instance content digest,
    ``generator_id(params, inst)``, replications), keys in plan order, so ``anchor_policy``, ``naive_fallback`` and
    ``fallback_spec`` of that generator then take them without recomputing. A call that raises caches nothing.

    Returns:
        The map now cached under the key (the one already held, when the values equal it).

    Raises:
        TypeError: if ``replications`` is not an integer (a bool included) or a quantile is not a float.
        ValueError: if ``replications`` < 1, the keys are not ``quantile_keys(inst)``, a quantile is not finite or is
            negative, or the process already holds other quantiles under the key.

    """
    key = _cache_key(inst, params, check_replications(replications))
    expected = quantile_keys(inst)
    if set(quantiles) != expected:
        raise ValueError(
            "F_Q quantiles must hold exactly the route keys (c, pool, j, k) of naive_fq.quantile_keys(instance), as "
            f"generator_quantiles returns them; got {len(quantiles)} keys for {len(expected)} expected"
        )
    for k, v in quantiles.items():
        if type(v) is not float:
            raise TypeError(f"naive F_Q quantile at {k!r} must be a float, as generator_quantiles returns, got {v!r}")
    q = frozen_map(canonical_quantiles({k: quantiles[k] for k in _route_keys(inst) if k in expected}))
    held = _QUANTILES.get(key)
    if held is not None:
        if dict(held) != dict(q):
            raise ValueError("the process already holds other F_Q quantiles for this instance, generator and count")
        return held
    _QUANTILES[key] = q
    return q


def naive_fallback(
    inst: Instance,
    params: GeneratorParams,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    end_aware: bool = True,
) -> NaiveFallback:
    """The D9 fallback spec of the episodes ``params`` generates: naive with ``generator_quantiles`` (§8.1, §9.3).

    The spec carries ``inst.family_digest``, the content the quantiles were computed for (F_Q reads no stock, so the
    instance at any rung of the file), so ``Env`` refuses it on an instance of other content with the same hash label
    (REG-M2R2-01), as well as on an omega of another generator.
    It plays the end-aware naive (``policies.naive``, Q98), the anchor's rule; ``end_aware=False`` plays plain naive
    (tests and evidence only).
    """
    return NaiveFallback(
        generator_id(params, inst),
        generator_quantiles(inst, params, replications, n_jobs),
        instance_digest=inst.family_digest,
        end_aware=end_aware,
    )


def anchor_policy(
    inst: Instance,
    params: GeneratorParams | None = None,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    end_aware: bool = True,
) -> NaivePolicy:
    """Naive as the anchor of (55), with the F_Q of the generator that drew omega (§8.1), end-aware (Q98).

    ``params`` None means an injected list: F_Q is then the point mass at 0 and the policy is ``NaivePolicy()``, the
    frozen cents of the fixture. ``end_aware=False`` gives plain naive (``naive_plain``; tests and evidence only).
    """
    fq = None if params is None else generator_quantiles(inst, params, replications, n_jobs)
    return NaivePolicy(fq_quantile=fq, end_aware=end_aware)


def fallback_spec(
    inst: Instance,
    params: GeneratorParams | None = None,
    replications: int = REPLICATIONS,
    n_jobs: int = 1,
    end_aware: bool = True,
) -> NaiveFallback | str:
    """The ``Env`` fallback spec of an episode: ``naive_fallback`` for a generated omega, ``"naive"`` for injected.

    Both play the end-aware naive (Q98). With ``end_aware=False``, plain naive's: ``"naive_plain"`` for injected, a
    ``NaiveFallback`` with ``end_aware=False`` for generated (tests and evidence only).
    """
    if params is None:
        return "naive" if end_aware else "naive_plain"
    return naive_fallback(inst, params, replications, n_jobs, end_aware)

"""Public names of naive's own steps for the M4 naive-derived baselines (nd, sz_state_base_stock, human_ref, greedy_lp).

Design §8.1, §8.2; Q100 (3). Aliases of naive's private helpers, no new code: a baseline built from them differs from
the anchor by its named idea only, and naive's arithmetic, its plan cache and every frozen cent stay in
``policies.naive``. They live here, not at the end of ``naive.py``, so the M4 interface touches naive only by the
``targets`` keyword of ``naive_action``.
"""

from sbfplan.policies.naive import _cached_plan as cached_plan  # (static, fq) -> private instance and plan
from sbfplan.policies.naive import _closures as route_closures  # (inst, routes, ds, fq) -> (d_cb, quantile)s
from sbfplan.policies.naive import _destination as destination  # (inst, edge, lane) -> the node bound for
from sbfplan.policies.naive import _levels as route_levels  # (inst, routes, ds, fq) -> S^nv_lk of (67)
from sbfplan.policies.naive import _observed_closed as observed_closed  # step 6's "observed closed" test
from sbfplan.policies.naive import _route_edges as route_edges  # (inst, first edge, lane) -> route's edges
from sbfplan.policies.naive import _seasonal as seasonal  # (inst, week) -> m^sea(t) per sink demand (j, k)
from sbfplan.policies.naive import _slot_transit as slot_transit  # (inst, action slot) -> tau_s of (72)
from sbfplan.policies.naive import _targets as destination_targets  # (inst, routes, levels) -> S_jk, step 3
from sbfplan.policies.naive import _targets_at as targets_at  # (inst, plan, week) -> naive's S_jk of week t
from sbfplan.policies.naive import carries_closure_term  # (inst, j) -> 1^cl of (67): False into plants


__all__ = [
    "cached_plan",
    "carries_closure_term",
    "destination",
    "destination_targets",
    "observed_closed",
    "route_closures",
    "route_edges",
    "route_levels",
    "seasonal",
    "slot_transit",
    "targets_at",
]

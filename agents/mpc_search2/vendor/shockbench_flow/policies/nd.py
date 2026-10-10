"""`nd`: naive with its closure term removed and a demand safety stock at sinks (M4, stream "heuristics").

Design §8.2 table, §7.2 (57); Q55, Q84, Q100 (3), Q100 (4).

S*_lk = (tau_l + 1 + o_l) d_lk + z_{pi/(pi+h)} sd_lk into a sink, sd_lk the SD of lead-time demand under (21) without
Xi; F_Q = 0 on every route; every other node, step and threshold is naive's (§8.1), the end-aware cut (71)-(72)
included, so nd differs from the anchor by its demand safety stock alone and (57) stays exact. Readings (design §12 M4
rows "nd" and "`nd` as built"):

- the sink level is the critical-ratio quantile of lead-time demand as §7.2 and §8.2 write it, without (67)'s cover
  cap (the owner's "Uncap nd's demand term", Q100 (4)): on `tiny`, where naive's cap binds at `sink_us` ((tau + 1) d
  = w^max d = 11,250), nd holds 14,233.91 there, above naive's level; every other route's level is naive's with
  F_Q = 0, cap included;
- sd_lk = (d_lk / d-bar) x the unconditional stationary SD of sum_{s < tau_l + 1 + o_l} d-bar exp(eps_s) under
  lognormal (21) without Xi, exact by the lognormal covariance, scaled by m^sea of the ordering week as naive's
  (tau + 1) d term is; z = ``statistics.NormalDist().inv_cdf`` of ``naive_fq.critical_ratio(inst, j, k)``;
- the week's targets reach naive's steps 4-6 through ``naive_action(..., targets=)`` (via ``end_aware_action``):
  naive's inventory position, split and route choice, bit for bit; nd is stateless.
"""

import math
import numbers
from dataclasses import dataclass
from statistics import NormalDist

from shockbench_flow.instance.schema import Demand, Instance
from shockbench_flow.policies import naive_parts
from shockbench_flow.policies.base import params_run_name
from shockbench_flow.policies.naive import LastUseful, NaivePlan, end_aware_action, last_useful_weeks
from shockbench_flow.policies.naive_derived import ThresholdParams
from shockbench_flow.policies.naive_fq import critical_ratio
from shockbench_flow.policies.registry import PolicyContext


@dataclass(frozen=True)
class NdParams(ThresholdParams):
    """``nd``'s parameters: naive's step-6 thresholds (nd's only change is its level, §8.2).

    Raises:
        ValueError: as ``naive.check_thresholds``.

    """


def demand_sd(dem: Demand, weeks: int) -> float:
    """SD of sum_{s < weeks} exp(eps_s) at d-bar = 1 under stationary AR(1) (21) without Xi (module docstring).

    With v = sigma^2 / (1 - phi^2): sqrt(e^v sum_{i,j} (e^{v phi^{|i-j|}} - 1)); at phi = 0 the sum of ``weeks``
    independent lognormals. The double sum is taken by lag, n (e^v - 1) + 2 sum_{m >= 1} (n - m)(e^{v phi^m} - 1), with
    ``math.expm1`` and ``math.fsum``.

    Raises:
        ValueError: if weeks < 1 or |phi| >= 1.

    """
    if isinstance(weeks, bool) or not isinstance(weeks, numbers.Integral) or weeks < 1:
        raise ValueError(f"lead-time demand needs an integer number of weeks >= 1, got {weeks!r}")
    phi, sigma = float(dem.phi), float(dem.sigma)
    if not abs(phi) < 1.0:
        raise ValueError(f"stationary AR(1) demand (21) needs |phi| < 1, got {dem.phi!r}")
    n = int(weeks)
    v = sigma * sigma / (1.0 - phi * phi)
    terms = [n * math.expm1(v)] + [2.0 * (n - m) * math.expm1(v * phi**m) for m in range(1, n)]
    return math.sqrt(math.exp(v) * math.fsum(terms))


def nd_targets(inst: Instance, plan: NaivePlan, week: int) -> dict[tuple[int, int], float]:
    """S_jk of week t for nd: naive's step 3 with F_Q = 0 and the sink routes' safety stock, uncapped (Q100 (4)).

    ``plan`` is naive's plan with F_Q = 0 (its quantiles are not read). Every route's level is naive's with F_Q = 0 on
    the week's route demands d_lk m^sea(t), except a route into a sink with demand for k: (tau + 1 + o) d +
    z d ``demand_sd``(tau + 1 + o), d = d_lk m^sea(t), without naive's cover cap w^max d. At a zero penalty
    (pi = 0, critical ratio 0, which the loader accepts) z is -inf and so is the level, its limit: naive's step 5
    orders max(0, -inf - IP) = 0 for the destination (ORACLE-M4-04; ``NormalDist.inv_cdf`` refuses 0 itself).

    Raises:
        statistics.StatisticsError: at a critical ratio of 1 (h = 0 with pi > 0: an unbounded level, owner queue M4).

    """
    m = naive_parts.seasonal(inst, week)
    ds = [r.d * m.get((r.dest, r.k), 1.0) for r in plan.routes]
    levels = naive_parts.route_levels(inst, plan.routes, ds, {})
    demands = {(dem.node, dem.k): dem for dem in inst.demands}
    for i, (r, d) in enumerate(zip(plan.routes, ds)):
        dem = demands.get((r.dest, r.k))
        if dem is None:
            continue
        cr = critical_ratio(inst, r.dest, r.k)
        if cr == 0.0:  # pi = 0: z_0 = -inf, so S* is -inf, the level's limit, which orders nothing (ORACLE-M4-04)
            levels[i] = -math.inf
            continue
        n = r.tau + 1 + r.offset
        z = NormalDist().inv_cdf(cr)
        s_star = n * d + z * d * demand_sd(dem, n)
        levels[i] = s_star  # §7.2's critical-ratio quantile of lead-time demand, uncapped (Q100 (4))
    return naive_parts.destination_targets(inst, plan.routes, levels)


class NdPolicy:
    """`nd` as a ``Policy``: the instance and naive's plan with F_Q = 0 (``naive_parts.cached_plan``) at reset."""

    name = "nd"

    def __init__(self, params: NdParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = NdParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = params_run_name(type(self).name, self.params)  # its parameters in its name (SPEC-M4R2-01)
        self._inst: Instance | None = None
        self._plan: NaivePlan | None = None
        self._last: LastUseful | None = None

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """The instance, naive's plan with F_Q = 0 and the last useful weeks (71)-(72)."""
        self._inst, self._plan = naive_parts.cached_plan(static, None)
        self._last = last_useful_weeks(self._inst, self._plan)

    def act(self, obs: dict) -> dict:
        """The end-aware naive's steps 4-6 on nd's targets of the week."""
        if self._inst is None or self._plan is None or self._last is None:
            raise RuntimeError("NdPolicy.act called before reset")
        return end_aware_action(
            self._inst,
            self._plan,
            self._last,
            obs,
            closed_threshold=self.params.closed_threshold,
            give_up_ratio=self.params.give_up_ratio,
            targets=nd_targets(self._inst, self._plan, obs["week"]),
        )

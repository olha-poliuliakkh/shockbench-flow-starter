"""`human_ref`: Sterman's anchoring-and-adjustment ordering rule at its fitted means (M4, stream "heuristics").

Design §8.2 table; Sterman 1989; Q100 (3).

Per destination (j, k) of naive's plan, each week:

    L-hat_t = theta L_{t-1} + (1 - theta) L-hat_{t-1},    O = max(0, L-hat_t + alpha_S (S'_jk - S - beta SL))

with S = on-hand - backlog, SL = pipeline plus queued lots bound for j, and L the observed demand at a sink
(``last_week.sinks.demand``; d-bar m^sea(1) in week 1) and naive's route demand sum_l d_lk m^sea(t) elsewhere.
Readings (design §12 M4 rows "human_ref" and "`human_ref` as built"):

- S'_jk = sum_l d_lk (``desired_weeks(params)`` + beta tau_l): Sterman's desired stock S' = 17 cases over the beer
  game's flow of 4 cases a week and supply line of 16 cases, (S' - beta x 16) / 4 = (17 - 0.34 x 16) / 4 = 2.89 weeks
  of cover at the fitted means (``DESIRED_WEEKS``) plus beta x the route's transit (the alternative S'_jk = 4.25 d_jk
  would sit routes with tau >= 13 at negative equilibrium stock); the mapping is a new number, SYNTHETIC(placeholder),
  so V23 blocks sealing on it; ``s_prime`` and ``beta`` of ``HumanRefParams`` feed it; fixed at reset from d_lk at
  m^sea = 1, without naive's I-bar_g or offset (the target is Sterman's, not naive's);
- L-hat starts at the first observation (the first ``act`` after a reset sets L-hat_t = L_{t-1}, Sterman's equilibrium
  start), then follows the update above; L_{t-1} at a sink is the realised demand d^{t-1} of ``last_week.sinks``
  (served or not), d-bar m^sea(t) when ``last_week`` is None (week 1);
- the order is split over naive's routes in proportion to d_lk with naive's route choice (steps 5-6), through
  ``naive_action(..., targets=IP + O)``, then the end-aware cut (71)-(72), so human_ref differs from the anchor by its
  ordering rule alone; a slot whose own edge is prohibited is never sent (naive's rule, Q111). human_ref sums IP in
  naive's exact order (step 4 of ``naive_action``: 0.0 plus the stock entries of (j, k) in observation order, then
  ``+= sum`` of the pipeline list bound for j, then ``+= sum`` of the queued lots bound for j, then ``-=`` the summed
  backlog), so naive orders
  max(0, (IP + O) - IP): exactly 0 when O = 0, and O up to the rounding of IP + O (a few ulps of max(IP, O)) otherwise;
- L-hat is reset at every ``reset`` (V1: an object reused over two episodes gives the fresh object's trajectories);
- L-hat is the rule's only cross-week state, so ``state()`` returns it as plain data and ``load_state`` restores it
  after a reset of the same episode: an in-process resume (B2) continues bit for bit, as the LP baselines' does (design
  §12 "Warm start (M4 reading)"; DET-M4-1: without it a restore re-seeded L-hat from the first observed demand, which a
  blackout hides). A wire resume (a fresh Reset at week t) starts L-hat afresh, documented, not asserted equal.
"""

import math
from dataclasses import dataclass

from shockbench_flow.instance.schema import Instance
from shockbench_flow.policies import naive_parts
from shockbench_flow.policies.base import params_run_name
from shockbench_flow.policies.naive import LastUseful, NaivePlan, end_aware_action, last_useful_weeks
from shockbench_flow.policies.naive_derived import step4_books, step4_ip
from shockbench_flow.policies.registry import PolicyContext


# the beer-game mapping of S' (module docstring): SYNTHETIC(placeholder) until the owner confirms it (owner queue, M4)
BEER_FLOW = 4.0  # cases per week in Sterman's game (the flow S' is measured against)
BEER_SUPPLY_LINE = 16.0  # cases in the game's supply line at equilibrium (4 weeks x 4 cases)


@dataclass(frozen=True)
class HumanRefParams:
    """Sterman's (1989) fitted means over 44 subjects: theta 0.36, alpha_S 0.26, beta 0.34, S' 17 (design §8.2).

    Raises:
        ValueError: if theta, alpha_s or beta is outside [0, 1] or s_prime is not finite and > 0.

    """

    theta: float = 0.36
    alpha_s: float = 0.26
    beta: float = 0.34
    s_prime: float = 17.0

    def __post_init__(self) -> None:
        for name in ("theta", "alpha_s", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"human_ref {name} must be in [0, 1], got {value!r}")
        s = self.s_prime
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not (0.0 < s < float("inf")):
            raise ValueError(f"human_ref s_prime must be finite and > 0, got {s!r}")


def desired_weeks(params: HumanRefParams) -> float:
    """Weeks of flow in S'_jk: (s_prime - beta x ``BEER_SUPPLY_LINE``) / ``BEER_FLOW`` (module docstring)."""
    return (params.s_prime - params.beta * BEER_SUPPLY_LINE) / BEER_FLOW


DESIRED_WEEKS = desired_weeks(HumanRefParams())  # 2.89 weeks at Sterman's fitted means, SYNTHETIC(placeholder)


def desired_stocks(plan: NaivePlan, params: HumanRefParams) -> dict[tuple[int, int], float]:
    """S'_jk = sum_l d_lk (``desired_weeks(params)`` + beta tau_l) per destination of naive's plan (m^sea = 1)."""
    weeks = desired_weeks(params)
    return {jk: sum(r.d * (weeks + params.beta * r.tau) for r in routes) for jk, routes in plan.groups.items()}


def expectation(l_hat: float | None, observed: float, theta: float) -> float:
    """L-hat_t = theta L_{t-1} + (1 - theta) L-hat_{t-1}; the observation itself when there is no L-hat yet."""
    return observed if l_hat is None else theta * observed + (1.0 - theta) * l_hat


def sterman_order(l_hat: float, s_prime: float, stock: float, supply_line: float, params: HumanRefParams) -> float:
    """Sterman's order O = max(0, L-hat + alpha_S (S' - S - beta SL)) (module docstring)."""
    return max(0.0, l_hat + params.alpha_s * (s_prime - stock - params.beta * supply_line))


@dataclass(frozen=True)
class Position:
    """One destination's step-4 books from an observation, summed as ``naive_action`` sums them."""

    on_hand: float
    backlog: float
    pipeline: float  # the pipeline shipments bound for j
    queued: float  # the queue lots whose lane ends at j
    ip: float  # IP_jk of step 4, in naive's order of operations

    @property
    def stock(self) -> float:
        """Sterman's S: on-hand minus backlog."""
        return self.on_hand - self.backlog

    @property
    def supply_line(self) -> float:
        """Sterman's SL: the pipeline plus the queued lots bound for j."""
        return self.pipeline + self.queued


def positions(inst: Instance, plan: NaivePlan, obs: dict) -> dict[tuple[int, int], Position]:
    """Every destination's books of naive's step 4 (``naive_derived.step4_books`` and ``step4_ip``: IP is naive's)."""
    books = step4_books(inst, obs)
    out = {}
    for jk in plan.groups:
        pipe, que = sum(books.in_transit.get(jk, [])), sum(books.queued.get(jk, []))
        out[jk] = Position(books.on_hand.get(jk, 0.0), books.backlog.get(jk, 0.0), pipe, que, step4_ip(books, jk))
    return out


def observed_demand(inst: Instance, plan: NaivePlan, obs: dict) -> dict[tuple[int, int], float]:
    """L_{t-1} per destination: last week's realised sink demand, d-bar m^sea(t) without it, route demand elsewhere."""
    week = obs["week"]
    demands = {(dem.node, dem.k): dem for dem in inst.demands}
    seen: dict[tuple[int, int], float] = {}
    lw = obs.get("last_week")
    sinks = None if lw is None else lw.get("sinks")
    if sinks is not None:
        for n, k, d in zip(sinks["node"], sinks["k"], sinks["demand"]):
            if d is not None:
                seen[(n, k)] = d
    m = naive_parts.seasonal(inst, week)
    out = {}
    for jk, routes in plan.groups.items():
        dem = demands.get(jk)
        if dem is None:
            out[jk] = sum(r.d * m.get(jk, 1.0) for r in routes)
        else:
            out[jk] = seen[jk] if jk in seen else dem.dbar * dem.m_sea(week)
    return out


class HumanRefPolicy:
    """`human_ref` as a ``Policy``: L-hat per destination is its only cross-week state, cleared at reset."""

    name = "human_ref"

    def __init__(self, params: HumanRefParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = HumanRefParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = params_run_name(type(self).name, self.params)  # its parameters in its name (SPEC-M4R2-01)
        self._inst: Instance | None = None
        self._plan: NaivePlan | None = None
        self._last: LastUseful | None = None
        self._s_prime: dict[tuple[int, int], float] = {}
        self._l_hat: dict[tuple[int, int], float] = {}

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """The instance, naive's plan, the last useful weeks, S'_jk per destination, and L-hat cleared."""
        self._inst, self._plan = naive_parts.cached_plan(static, None)  # routes, d_lk and tau_l do not read F_Q
        self._last = last_useful_weeks(self._inst, self._plan)
        self._s_prime = desired_stocks(self._plan, self.params)
        self._l_hat = {}

    def act(self, obs: dict) -> dict:
        """Update L-hat, order O per destination, and send it through naive's steps 5-6 and the end-aware cut."""
        if self._inst is None or self._plan is None or self._last is None:
            raise RuntimeError("HumanRefPolicy.act called before reset")
        p = self.params
        demand = observed_demand(self._inst, self._plan, obs)
        targets = {}
        for jk, pos in positions(self._inst, self._plan, obs).items():
            self._l_hat[jk] = expectation(self._l_hat.get(jk), demand[jk], p.theta)
            targets[jk] = pos.ip + sterman_order(self._l_hat[jk], self._s_prime[jk], pos.stock, pos.supply_line, p)
        return end_aware_action(self._inst, self._plan, self._last, obs, targets=targets)

    def state(self) -> dict:
        """L-hat per destination as plain data (``[j, k, L-hat]`` rows in insertion order), for an in-process resume."""
        return {"l_hat": [[j, k, v] for (j, k), v in self._l_hat.items()]}

    def load_state(self, state: dict) -> None:
        """Restore ``state()`` after a reset of the same episode (the next ``act`` updates the restored L-hat).

        Raises:
            RuntimeError: if called before ``reset``.
            ValueError: if a row names a destination outside naive's plan or holds a non-finite L-hat.

        """
        if self._plan is None:
            raise RuntimeError("HumanRefPolicy.load_state called before reset")
        l_hat = {}
        for j, k, v in state["l_hat"]:
            jk = (int(j), int(k))
            if jk not in self._plan.groups or not math.isfinite(v):
                raise ValueError(f"human_ref L-hat row {[j, k, v]!r}: not a finite value of a destination of the plan")
            l_hat[jk] = float(v)
        self._l_hat = l_hat

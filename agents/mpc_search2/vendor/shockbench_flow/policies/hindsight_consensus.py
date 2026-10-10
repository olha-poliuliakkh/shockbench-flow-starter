"""`hindsight_consensus`: the consensus week-1 action of lexicographic scenario LPs (M4, stream "lp-baselines").

Design (70), §8.2 table and "Hindsight consensus"; Q17, Q38, Q58, Q76, Q95, Q100 (1).

Week t: for each of N_HC scenarios (``scenarios.scenario_library`` with ``scenarios.TAG_HINDSIGHT``, drawn once per
process; windows and their fab hits by ``scenarios.scenario_windows``) the rolled LP from the current state, lot book
included, to T (H_t = T - t + 1, ``lp_common.rolled_lp(..., fab_hits=)``) is solved lexicographically by
``LPSession.lexicographic``: min J, then min w.x s.t. J <= J* + 1e-9 |J*| (design §12 M4 row "(70) slack"), w the
stream-20 weights of the window's calendar weeks (``scenarios.stream20_weights``); x^t_HC is the mean over the scenarios
that succeeded of the week-1 action coordinates only (``lp_common.action_columns``: requests out of non-chokepoint nodes
and tanker releases; container releases, lot starts, OSAT starts and the energy split are dropped), mapped by
``lp_common.week1_action``. A failed scenario is excluded and counted (``StepTelemetry.scenarios_failed``); if every
scenario fails the week plays naive's action from the same observation, counted. One ``LPSession`` per scenario keeps
its basis chain (design §12 "`hindsight_consensus` as built").

Never a ``ClairvoyantPolicy``: it samples G_pub given the observations and never receives omega (``policies.base``).
Labelled "not budget-compliant" in every report (§8.2 table; ``registry.NOT_BUDGET_COMPLIANT``).
"""

import time
from dataclasses import dataclass

import numpy as np

from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies import scenarios as S
from shockbench_flow.policies.base import StepTelemetry
from shockbench_flow.policies.naive_parts import cached_plan
from shockbench_flow.policies.registry import PolicyContext


N_HC_DEFAULT = 5  # N_HC of §8.2 "Hindsight consensus" (2.8-22 s per step at 5 on the realistic full LP)


@dataclass(frozen=True)
class HindsightParams:
    """``hindsight_consensus``'s scenario count N_HC.

    Raises:
        ValueError: if n_hc is not an integer >= 1.

    """

    n_hc: int = N_HC_DEFAULT

    def __post_init__(self) -> None:
        if isinstance(self.n_hc, bool) or not isinstance(self.n_hc, int) or self.n_hc < 1:
            raise ValueError(f"hindsight_consensus n_hc must be an integer >= 1, got {self.n_hc!r}")


def policy_name(params: HindsightParams) -> str:
    """``hindsight_consensus`` for N_HC = 5, else ``hindsight_consensus[N=<n>]``."""
    return "hindsight_consensus" if params.n_hc == N_HC_DEFAULT else f"hindsight_consensus[N={params.n_hc}]"


def consensus(xs: list[np.ndarray], columns: np.ndarray, n_col: int) -> np.ndarray:
    """x_HC of (70): the mean of the succeeding scenarios' points on the action columns, 0 elsewhere.

    Summed in scenario order and divided by the count (design §12 "`hindsight_consensus` as built").

    Raises:
        ValueError: if ``xs`` is empty.

    """
    if not xs:
        raise ValueError("the consensus needs at least one succeeding scenario")
    total = np.zeros(len(columns))
    for x in xs:
        total = total + x[columns]
    out = np.zeros(n_col)
    out[columns] = total / len(xs)
    return out


class HindsightConsensus:
    """`hindsight_consensus` as a ``Policy``; one basis chain per scenario, per episode."""

    def __init__(self, params: HindsightParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = HindsightParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = policy_name(self.params)
        self.telemetry: list[StepTelemetry] = []

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Load the instance, the library (cached), the stream-20 weights, one ``LPSession`` per scenario."""
        self._inst, _plan = cached_plan(static, self.context.fq_quantile)
        self._gen = S.generator_params(self._inst, self.context.generator)
        self._library = S.scenario_library(self._inst, self._gen, S.TAG_HINDSIGHT, self.params.n_hc)
        self._weights: np.ndarray | None = None  # stream 20, drawn at the first solve (nc of the window template)
        self._sessions = [L.LPSession() for _ in self._library]
        self._memory = L.ObservedGraph.nominal(self._inst)
        self._ages = S.ImpairmentAges()
        self._fallback = L.internal_fallback(self.context, static, obs, policy_seed)
        self.telemetry = []

    def act(self, obs: dict) -> dict:
        """Week t's consensus of the scenarios' lexicographic week-1 actions (module docstring)."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        self._ages.update(inst, obs, self._memory)
        H = inst.T - t + 1
        windows = S.scenario_windows(
            inst, self._gen, obs, self._memory, self._library, S.TAG_HINDSIGHT, H, ages=self._ages
        )
        window = L.rolled_window(inst, obs, H)  # one rolled instance for every scenario of the week
        xs, trail, iters, model0 = [], [], 0, None
        for (arrays, hits), session in zip(windows, self._sessions, strict=True):
            model = L.rolled_lp(inst, obs, arrays, H, fab_hits=hits, window=window)
            model0 = model if model0 is None else model0
            if self._weights is None:
                self._weights = S.stream20_weights(inst, int(model.meta["nc"]))
            w = self._weights[t - 1 : inst.T].ravel()
            res = session.lexicographic(L.to_highs_lp(model), w, L.WindowShape.of(model))
            trail.extend(res.trail)
            iters += res.iterations
            if res.ok:
                xs.append(res.x)
        failed = len(windows) - len(xs)
        if xs:
            x = consensus(xs, L.action_columns(inst, model0), len(model0.lb))
            action = L.week1_action(inst, model0, x, obs, L.prohibited_now(self._memory, t))
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, iters, tuple(trail), fallback=not xs, scenarios_failed=failed)
        )
        return action

    def state(self) -> dict:
        """The cross-week state (every scenario's basis chain, observed memory and ages) for an in-process resume."""
        return {
            "sessions": [s.state() for s in self._sessions],
            "memory": self._memory.state(),
            "ages": self._ages.state(),
        }

    def load_state(self, state: dict) -> None:
        """Restore ``state()`` after a reset of the same episode."""
        for session, st in zip(self._sessions, state["sessions"], strict=True):
            session.load_state(st)
        self._memory = L.ObservedGraph.from_state(self._inst, state["memory"])
        self._ages = S.ImpairmentAges.from_state(state["ages"])

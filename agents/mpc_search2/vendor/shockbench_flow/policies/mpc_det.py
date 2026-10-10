"""`mpc_det`: rolling (51) on the persistence point forecast, horizon from the instance (M4, stream "lp-baselines").

Design §8.2 table, "Gate baselines" and "MPC horizon"; Q17, Q67, Q84, Q95.

Week t: update the observed-graph memory; H_t = min(H, T - t + 1) with H from the sweep label and L of the instance
(``lp_common.lead_time_L``); the window arrays are ``lp_common.persistence_arrays``; the model is
``lp_common.rolled_lp`` (``oracle.lp.build_lp`` on the rolled instance: one row table, V3); solve it through the
status ladder, warm from last week's basis (cold at reset); the action is ``lp_common.week1_action`` of the solution.
Its terminal value V_H is the builder's credit (23) at the window end, so at H_t = T - t + 1 it is end-aware by
construction (no Q98 cut). Every rung failing plays naive's action from the same observation, counted.

The window carries the planning rules (``oracle.lp.build_lp(..., planning_rules=True)``: segment loading pro rata, base
load first, lot and OSAT starts, container releases; design §12 M5 row "Planning rules", owner queue M5-O19 (b)) unless
``MpcDetParams.planning_rules`` is False, M4's window, a policy name of its own.

Each H of the sweep is its own policy name (the name enters ``Trajectory.sha256``, ``Env.reset(policy_name=)`` and the
conformance records): ``mpc_det`` for the canonical H = L (the §2.6 separation pair; design §12 M4 row "Horizon"),
``mpc_det[H=<label>]`` for the others.
"""

import time
from dataclasses import dataclass

from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.base import StepTelemetry
from shockbench_flow.policies.lp_common import CANONICAL_H, H_SWEEP
from shockbench_flow.policies.naive_parts import cached_plan
from shockbench_flow.policies.registry import PolicyContext


@dataclass(frozen=True)
class MpcDetParams:
    """``mpc_det``'s horizon, a label of the Q67 sweep resolved against L of the instance at reset, and its window.

    ``planning_rules`` False builds M4's window, (51) as the oracle's (design §12 M5 row "Planning rules").

    Raises:
        ValueError: on a label outside ``H_SWEEP`` or a non-bool ``planning_rules``.

    """

    H: str = CANONICAL_H
    planning_rules: bool = True

    def __post_init__(self) -> None:
        if self.H not in H_SWEEP:
            raise ValueError(f"mpc_det H must be a label of {H_SWEEP} (Q67), got {self.H!r}")
        if not isinstance(self.planning_rules, bool):
            raise ValueError(f"mpc_det planning_rules must be a bool, got {self.planning_rules!r}")


def policy_name(params: MpcDetParams) -> str:
    """``mpc_det`` for H = L with the planning rules, else ``mpc_det[H=<label>]``, ``[rules=off]`` or both."""
    parts = ([] if params.H == CANONICAL_H else [f"H={params.H}"]) + ([] if params.planning_rules else ["rules=off"])
    return f"mpc_det[{','.join(parts)}]" if parts else "mpc_det"


class MpcDet:
    """`mpc_det` as a ``Policy``: cross-week state (basis, observed graph) only within an episode, cleared at reset."""

    def __init__(self, params: MpcDetParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = MpcDetParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = policy_name(self.params)
        self.telemetry: list[StepTelemetry] = []

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Load the instance and naive's plan (for L and the fallback), a fresh ``LPSession`` and observed memory."""
        self._inst, plan = cached_plan(static, self.context.fq_quantile)
        self._H = self._horizon(self._inst, plan)
        self._session = L.LPSession()
        self._memory = L.ObservedGraph.nominal(self._inst)
        self._fallback = L.internal_fallback(self.context, static, obs, policy_seed)
        self.telemetry = []

    def act(self, obs: dict) -> dict:
        """Week t's window solve and its week-1 action (module docstring)."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        H_t = L.window_length(self._H, t, inst.T)
        arrays = self._window_arrays(inst, obs, H_t)
        model = L.rolled_lp(inst, obs, arrays, H_t, planning_rules=self.params.planning_rules)
        res = self._session.solve(L.to_highs_lp(model), L.WindowShape.of(model))
        self._record(model, res)
        if res.ok:
            action = L.week1_action(inst, model, res.x, obs, L.prohibited_now(self._memory, t))
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action

    def _horizon(self, inst, plan) -> int:
        """H of the window: the sweep label's against L of the instance (Q67).

        The seam of the leaderboard field's same-family variants (``policies.field``: a fixed horizon in weeks);
        ``mpc_det`` itself resolves ``params.H``.
        """
        return L.horizon_length(self.params.H, L.lead_time_L(inst, plan))

    def _window_arrays(self, inst, obs: dict, H_t: int) -> dict:
        """Week t's window arrays: the persistence point forecast (``lp_common.persistence_arrays``).

        The seam of the leaderboard field's safety variants (``policies.field``: a scaled forecast); ``mpc_det`` itself
        plans on the persistence forecast unchanged.
        """
        return L.persistence_arrays(inst, obs, self._memory, H_t)

    def _record(self, model, res) -> None:
        """The seam of the plan-against-execution trace: ``mpc_det`` itself keeps nothing.

        A tracing subclass keeps the week's window model and its solve (M5; design §12 M5 row "MPC plan against
        execution"; ``scripts/python/evidence/m5_base_mpc_trace.py``).
        """

    def state(self) -> dict:
        """The cross-week state (the session's basis chain, the observed-graph memory) for an in-process resume."""
        return {"session": self._session.state(), "memory": self._memory.state()}

    def load_state(self, state: dict) -> None:
        """Restore ``state()`` after a reset of the same episode."""
        self._session.load_state(state["session"])
        self._memory = L.ObservedGraph.from_state(self._inst, state["memory"])

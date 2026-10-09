"""The policy protocol: what a participant's policy sees and returns (design §9.1-9.2; Q68, Q86).

A policy receives the ``Static`` tables and the week-1 observation at reset, then one observation per week, and returns
an action dict of the wire schema: ``{"week": t, "flows": {"slot": [...], "qty": [...]}, "overrides": ... | None,
"hold": ... | None}``. It never sees omega, the simulator or ``info``, except that under the clairvoyant regime, and
only there, in local and dev runs, a policy that asks for it receives the omega payload at reset (§5.1 'full omega at
reset'; ``ClairvoyantPolicy``). Whether a policy asks is one predicate, ``wants_omega``, which ``dynamics.env.rollout``
and ``information.runner.serve`` both read, so the in-process and the wire episode call ``reset`` alike (W2): a
policy that declares ``wants_omega = True`` always gets the keyword ``omega`` (the payload under clairvoyant, None in
every other regime), any other policy never does (design §12 M3 rows).
"""

import dataclasses
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class Policy(Protocol):
    name: str

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None: ...

    def act(self, obs: dict) -> dict: ...


class ClairvoyantPolicy(Protocol):
    """A policy that asks for omega at reset (``wants_omega = True``), which it receives under clairvoyant only.

    Its reset takes the keyword ``omega``: the omega payload (``information.wire.omega_payload``, every array but
    ``meta_episode`` and ``meta_split``) under clairvoyant, None in every other regime. No M4 baseline is one:
    ``hindsight_consensus`` samples from G_pub given the observations (70) and never receives omega.
    """

    name: str
    wants_omega: bool  # True: the class attribute ``wants_omega`` reads

    def reset(self, static: dict, obs: dict, policy_seed: int, *, omega: dict | None) -> None: ...

    def act(self, obs: dict) -> dict: ...


def wants_omega(policy) -> bool:
    """Whether ``policy`` asks for omega at reset: its attribute ``wants_omega`` is the bool True (module docstring).

    The one rule ``rollout`` and ``runner.serve`` apply; a truthy non-bool (1, a str) is not a request.
    """
    return getattr(policy, "wants_omega", False) is True


def reset_policy(policy, static: dict, obs: dict, policy_seed: int, omega: dict | None = None) -> None:
    """``policy.reset`` by the one rule: with the keyword ``omega`` exactly when ``wants_omega(policy)``.

    ``omega`` is the episode's payload (``info.get('omega')``, None outside clairvoyant).
    """
    if wants_omega(policy):
        policy.reset(static, obs, policy_seed, omega=omega)
    else:
        policy.reset(static, obs, policy_seed)


def _value_text(value: object) -> str:
    """A parameter value in a run name: a real as the ``repr`` of its float when that float equals it, else exactly."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        as_float = float(value)
        return repr(as_float) if as_float == value else str(value)
    return str(value)


def run_name(base: str, values: Mapping[str, object], defaults: Mapping[str, object]) -> str:
    """A baseline's run name: ``base`` at its defaults, else ``base[<field>=<value>,...]`` over the fields off them.

    The name enters the trajectory hash (``Env.reset(policy_name=)``) and names the run in every report, so two
    parameterisations that can play apart never share one (design §12 rows "Horizon (M4 reading)" and "RSS table (M4
    runner)"; SPEC-M4R2-01): the fields in the order of ``values``, each value by ``_value_text``, so 1 and 1.0 (one
    parameter, compared exactly) give one name and a Fraction off every float keeps its exact text (``3/10``). The
    canonical parameterisation keeps the bare name, which V30 and the §2.6 pairs read. Callers pass their class's
    ``name`` as ``base``, so a subclass that declares its own name keeps it.
    """
    off = [f"{k}={_value_text(v)}" for k, v in values.items() if v != defaults[k]]
    return f"{base}[{','.join(off)}]" if off else base


def params_run_name(base: str, params) -> str:
    """``run_name`` of a baseline's frozen parameter dataclass against its class defaults, the fields in class order."""
    names = [f.name for f in dataclasses.fields(params)]
    default = type(params)()
    return run_name(base, {k: getattr(params, k) for k in names}, {k: getattr(default, k) for k in names})


def empty_action(week: int) -> dict:
    """The action that requests nothing (a missing slot is 0)."""
    return {"week": week, "flows": {"slot": [], "qty": []}, "overrides": None, "hold": None}


class ZeroPolicy:
    """`zero`: ship nothing (sanity baseline, §8.2)."""

    name = "zero"

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        pass

    def act(self, obs: dict) -> dict:
        return empty_action(obs["week"])


@dataclass(frozen=True)
class StepTelemetry:
    """One week of a baseline's solver telemetry (M4; design §12 M4 rows "Runner" and "Status ladder").

    An optional extension of the protocol: a policy that keeps a list ``telemetry`` of these (one per ``act``, cleared
    at ``reset``) has it read by the evaluation runner (``getattr(policy, "telemetry", None)``) into its timing and
    substitution tables, beside the environment's D9 counts. Telemetry is outside every hash (V1) and never reaches the
    simulator.
    """

    week: int
    seconds: float  # wall seconds of the week's solves (the runner times ``act`` itself, Q13)
    iterations: int  # simplex plus IPM iterations over the week's solves
    trail: tuple[tuple[str, str], ...]  # (ladder rung, HiGHS model status) per attempt, in order
    fallback: bool  # every rung failed: the week played naive's action from the same observation (never zero-filled)
    scenarios_failed: int = 0  # scenario solves excluded this week (hindsight_consensus, mpc_scen)

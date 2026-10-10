"""`mpc_scen`: two-stage SAA MPC on scenarios of the public generator (milestone M4, stream "lp-baselines").

Design §8.2 table; Q17, Q95, Q100 (1).

Week t: S window blocks, one per scenario of ``scenarios.scenario_windows`` (library ``scenarios.scenario_library`` with
``scenarios.TAG_MPC_SCEN``, drawn once per process), each ``lp_common.rolled_lp`` with its draw's fab hits on the same
start state and horizon H_t = min(H, T - t + 1) as ``mpc_det`` (so the F-size pair mpc_scen > mpc_det isolates scenarios
against the point forecast; design §12 M4 row "Horizon"); ``lp_common.saa_lp`` joins them with nonanticipativity rows on
the week-1 action coordinates and minimises the mean J; one ``LPSession`` warm-starts week to week; the action is the
shared week-1 coordinates (``lp_common.week1_action``). Every rung failing plays naive's action from the same
observation.

S is the step budget's (M5-O6 (a); design §12 M5 row "`mpc_scen`'s S by the step budget"): ``S_BUDGET`` holds, per
instance kind, the largest S of ``S_BUDGET_CANDIDATES`` whose measured per-week CPU stays within the size's budget (the
smallest when none does), and ``MpcScenParams.S`` None (the default) takes it at reset; ``tiny`` keeps M4's
``S_DEFAULT``, SYNTHETIC(placeholder).
``calibrate_S`` (the smallest S of ``S_CANDIDATES`` whose week-1 action vector changes by less than ``S_TOLERANCE``,
relative L1, between S and 2S; Mak-Morton-Wood's stability reading of Q17) stays an offline diagnostic (design §12 M4
row "mpc_scen scenario count"). A set S, another H or ``planning_rules`` False is its own policy name,
``mpc_scen[S=<S>,H=<label>]``, ``mpc_scen[H=<label>]`` and ``...,rules=off``.

The windows carry the planning rules (``oracle.lp.build_lp(..., planning_rules=True)``; design §12 M5 row "Planning
rules", owner queue M5-O19 (b)) unless ``MpcScenParams.planning_rules`` is False, M4's windows.
"""

import functools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from shockbench_flow.instance.schema import FrozenDict
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies import scenarios as S_
from shockbench_flow.policies.base import StepTelemetry
from shockbench_flow.policies.lp_common import CANONICAL_H, H_SWEEP
from shockbench_flow.policies.naive_parts import cached_plan
from shockbench_flow.policies.registry import GeneratorRef, PolicyContext


S_CANDIDATES = (2, 4, 8, 16, 32)  # the calibration's grid of scenario counts (design §12 M4 row)
S_BUDGET_CANDIDATES = (2, 3, 4, 6, 8)  # the step budget's grid (M5-O6 (a); design §12 M5 row "`mpc_scen`'s S")
S_DEFAULT = 8  # SYNTHETIC(placeholder): tiny's S (V23 blocks sealing on it); small and full take the budget's S
# the owner's budgets per step, CPU seconds per week: `small` 1 s (M5-O6 (a)), `full` 4 s (Q115, 2026-09-29; was 2 s)
STEP_BUDGET_SECONDS: Mapping[str, float] = FrozenDict(small=1.0, full=4.0)
# the reading of design §12 M5 row "`mpc_scen`'s S by the step budget", which the step-budget evidence states
S_BUDGET_RULE = (
    "a size's S is the largest of S_BUDGET_CANDIDATES whose act CPU p95 (numpy's linear percentile of an episode's "
    "weekly act CPU seconds) is at most the size's STEP_BUDGET_SECONDS in every measured episode at every measured "
    "gamma, at the scoring worker's load; when none is, the smallest, over its budget"
)
# S_BUDGET_RULE's reading at gamma 0.62 and 0.97 on the AMD scoring host (step_budget_reading; the re-fit
# docs/evidence/m5_base_scen_budget.txt and the check docs/evidence/m5_base_scen_budget_check.txt of
# scripts/python/evidence/m5_base_scen_budget.py, whose verdict lines say whether each size's S is within); tiny keeps
# S_DEFAULT
S_BUDGET: Mapping[str, int] = FrozenDict(tiny=S_DEFAULT, small=3, full=2)
S_TOLERANCE = 0.05  # SYNTHETIC(placeholder): relative L1 change of the week-1 action between S and 2S
CALIBRATION_REPLICATIONS = 2  # F_Q replications of the calibration's D9 fallback, which a solving policy never takes


@dataclass(frozen=True)
class MpcScenParams:
    """``mpc_scen``'s scenario count (None: the step budget's, ``budget_S``), horizon label and window.

    Raises:
        ValueError: if S is neither None nor an integer >= 2, H is not a label of ``H_SWEEP`` or ``planning_rules`` is
            not a bool.

    """

    S: int | None = None
    H: str = CANONICAL_H
    planning_rules: bool = True

    def __post_init__(self) -> None:
        if self.S is not None and (isinstance(self.S, bool) or not isinstance(self.S, int) or self.S < 2):
            raise ValueError(f"mpc_scen S must be None or an integer >= 2, got {self.S!r}")
        if self.H not in H_SWEEP:
            raise ValueError(f"mpc_scen H must be a label of {H_SWEEP} (Q67), got {self.H!r}")
        if not isinstance(self.planning_rules, bool):
            raise ValueError(f"mpc_scen planning_rules must be a bool, got {self.planning_rules!r}")


def budget_S(kind: str) -> int:
    """The step budget's S of an instance kind (``S_BUDGET``).

    Raises:
        ValueError: for a kind without a measured budget.

    """
    if kind not in S_BUDGET:
        raise ValueError(f"mpc_scen has no step-budget S for instance kind {kind!r}: one of {tuple(S_BUDGET)}")
    return S_BUDGET[kind]


def step_budget_reading(p95: Mapping[tuple[float, int], Sequence[float]], budget: float) -> tuple[int, bool]:
    """``S_BUDGET_RULE`` on measured episodes: (the size's S, whether it is within ``budget``).

    ``p95`` maps each measured (gamma, S) to its episodes' act CPU p95 in seconds; a candidate missing at a gamma that
    another candidate was measured at is not within.
    """
    gammas = {g for g, _S in p95}
    within = [s for s in S_BUDGET_CANDIDATES if all((g, s) in p95 and max(p95[g, s]) <= budget for g in gammas)]
    return (max(within), True) if within else (S_BUDGET_CANDIDATES[0], False)


def policy_name(params: MpcScenParams) -> str:
    """``mpc_scen`` for the budget's S, the canonical H and the planning rules; else the set fields (module doc)."""
    if params.S is not None:
        parts = [f"S={params.S}", f"H={params.H}"]
    else:
        parts = [] if params.H == CANONICAL_H else [f"H={params.H}"]
    if not params.planning_rules:
        parts.append("rules=off")
    return f"mpc_scen[{','.join(parts)}]" if parts else "mpc_scen"


def saa_solve(
    inst,
    gen,
    obs: dict,
    memory: L.ObservedGraph,
    ages: S_.ImpairmentAges,
    library,
    H_t: int,
    session: L.LPSession,
    planning_rules: bool = True,
):
    """Week t's SAA over ``library`` on ``session``: (the week-1 action or None, the solve's result)."""
    windows = S_.scenario_windows(inst, gen, obs, memory, library, S_.TAG_MPC_SCEN, H_t, ages=ages)
    window = L.rolled_window(inst, obs, H_t)  # one rolled instance for every block of the week
    models = [
        L.rolled_lp(inst, obs, arrays, H_t, fab_hits=hits, window=window, planning_rules=planning_rules)
        for arrays, hits in windows
    ]
    lp, shape = L.saa_lp(models, L.action_columns(inst, models[0]))
    res = session.solve(lp, shape)
    if not res.ok:
        return None, res
    x0 = res.x[: len(models[0].lb)]
    return L.week1_action(inst, models[0], x0, obs, L.prohibited_now(memory, int(obs["week"]))), res


class MpcScen:
    """`mpc_scen` as a ``Policy``; the scenario library is per process, the basis chain per episode."""

    def __init__(self, params: MpcScenParams | None = None, context: PolicyContext | None = None) -> None:
        self.params = MpcScenParams() if params is None else params
        self.context = PolicyContext() if context is None else context
        self.name = policy_name(self.params)
        self.telemetry: list[StepTelemetry] = []

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Load the instance, the generator's library (cached), a fresh ``LPSession`` and observed memory."""
        self._inst, plan = cached_plan(static, self.context.fq_quantile)
        self._H = L.horizon_length(self.params.H, L.lead_time_L(self._inst, plan))
        self._gen = S_.generator_params(self._inst, self.context.generator)
        self._S = budget_S(self._inst.kind) if self.params.S is None else self.params.S
        self._library = S_.scenario_library(self._inst, self._gen, S_.TAG_MPC_SCEN, self._S)
        self._session = L.LPSession()
        self._memory = L.ObservedGraph.nominal(self._inst)
        self._ages = S_.ImpairmentAges()
        self._fallback = L.internal_fallback(self.context, static, obs, policy_seed)
        self.telemetry = []

    def act(self, obs: dict) -> dict:
        """Week t's SAA solve and its shared week-1 action (module docstring)."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        self._ages.update(inst, obs, self._memory)
        H_t = L.window_length(self._H, t, inst.T)
        action, res = saa_solve(
            inst,
            self._gen,
            obs,
            self._memory,
            self._ages,
            self._library,
            H_t,
            self._session,
            self.params.planning_rules,
        )
        if action is None:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action

    def state(self) -> dict:
        """The cross-week state (basis chain, observed-graph memory, impairment ages) for an in-process resume."""
        return {"session": self._session.state(), "memory": self._memory.state(), "ages": self._ages.state()}

    def load_state(self, state: dict) -> None:
        """Restore ``state()`` after a reset of the same episode."""
        self._session.load_state(state["session"])
        self._memory = L.ObservedGraph.from_state(self._inst, state["memory"])
        self._ages = S_.ImpairmentAges.from_state(state["ages"])


# ----- the offline calibration of S (design §12 "`mpc_scen` calibration as built") ---------------------------------
def action_vector(inst, action: dict) -> np.ndarray:
    """The week-1 action as one vector: the request of every action slot, then the override of every override slot."""
    out = np.zeros(len(inst.action_slots) + len(inst.override_slots))
    for s, q in zip(action["flows"]["slot"], action["flows"]["qty"], strict=True):
        out[s] = q
    ov = action.get("overrides")
    if ov is not None:
        for o, q in zip(ov["slot"], ov["qty"], strict=True):
            out[len(inst.action_slots) + o] = q
    return out


def _calibration_episode(inst, generator: GeneratorRef, entropy: int, S: int, episode: int) -> tuple[float, float, int]:
    """(sum |a_S - a_2S|_1, sum |a_2S|_1, weeks left out) over one dev episode played by mpc_scen at 2S."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies.naive_fq import naive_fallback

    params = generator.params(inst)
    omega = sample_omega(inst, params, entropy, episode)
    env = Env(fallback=naive_fallback(inst, params, CALIBRATION_REPLICATIONS))
    ctx = PolicyContext(generator=generator)
    policy = MpcScen(MpcScenParams(S=2 * S), ctx)
    obs, info = env.reset(inst, "prediction_free", omega, 0, policy_name=policy.name)
    policy.reset(info["static"], obs, 0)
    small = S_.scenario_library(policy._inst, policy._gen, S_.TAG_MPC_SCEN, S)
    session = L.LPSession()
    num = den = 0.0
    skipped = 0
    done = False
    while not done:
        big = policy.act(obs)
        H_t = L.window_length(policy._H, int(obs["week"]), policy._inst.T)
        few, _res = saa_solve(
            policy._inst,
            policy._gen,
            obs,
            policy._memory,
            policy._ages,
            small,
            H_t,
            session,
            policy.params.planning_rules,
        )
        if few is None or policy.telemetry[-1].fallback:
            skipped += 1
        else:
            a_big, a_few = action_vector(policy._inst, big), action_vector(policy._inst, few)
            num += float(np.abs(a_few - a_big).sum())
            den += float(np.abs(a_big).sum())
        obs, _r, done, _trunc, _info = env.step(big)
    return num, den, skipped


def calibration_statistic(
    inst, generator: GeneratorRef, entropy: int, episodes: range, S: int, *, n_jobs: int = 1
) -> tuple[float, int]:
    """(pooled relative L1 change of the week-1 action between S and 2S, weeks left out) over the dev episodes."""
    from shockbench_flow.parallel import ordered_map

    fn = functools.partial(_calibration_episode, inst, generator, entropy, S)
    parts = ordered_map(fn, list(episodes), n_jobs)
    num = math.fsum(p[0] for p in parts)
    den = math.fsum(p[1] for p in parts)
    return (num / den if den > 0 else 0.0), sum(p[2] for p in parts)


def calibrate_S(inst, generator: GeneratorRef, entropy: int, episodes: range, *, n_jobs: int = 1) -> int:
    """The smallest S of ``S_CANDIDATES`` whose week-1 action is stable to ``S_TOLERANCE`` between S and 2S.

    Offline, per instance, on the dev omegas (``entropy``, ``episodes``) of ``generator``, at every week of each
    episode played by ``mpc_scen`` at 2S; the command and its output go to ``docs/results.md``. Candidates run in
    increasing order (design §12 "`mpc_scen` calibration as built").

    Raises:
        ValueError: if no candidate is stable (the largest is reported, not returned).

    """
    stat = None
    for S in S_CANDIDATES:
        stat, _skipped = calibration_statistic(inst, generator, entropy, episodes, S, n_jobs=n_jobs)
        if stat < S_TOLERANCE:
            return S
    raise ValueError(
        f"no S of {S_CANDIDATES} is stable to {S_TOLERANCE}: the largest, {S_CANDIDATES[-1]}, changes by {stat:.4f}"
    )

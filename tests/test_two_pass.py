"""agents/twopass: the base-load-first bounds of pass 2, and the two-pass solve on a few Small weeks."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import ROOT


AGENT = ROOT / "agents" / "twopass"


@pytest.fixture
def frame():
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    load_agent_class(AGENT, "twopass_agent")
    try:
        yield sys.modules["frame"]
    finally:
        unload_agent()


def test_bounds_follow_the_simulator_rule(frame):
    # one grid (0) with two fabs (0, 1); columns of one week: shed base load, served base load, the fabs' energy
    model = SimpleNamespace(columns=(("ysh", 0), ("y", 0), ("E", 0), ("E", 1)), T=4, ub=np.full(16, np.inf))
    x = np.array(
        [5, 10, 3, 4]  # week 1: violated, but week 1 follows the planning rules already: never bounded
        + [5, 10, 3, 4]  # week 2: the fabs' 7 would have covered the 5 shed: serve base load first
        + [5, 10, 1, 0]  # week 3: the fabs' 1 would not: the fabs go dark
        + [5, 10, 0, 0],  # week 4: shed without fab energy, as the simulator does: no violation
        dtype=float,
    )
    grids = {0: [0, 1]}
    ub, changed = frame.base_first_bounds(model, x, grids, "fix", model.ub)
    assert changed == 2
    assert ub[4] == 0 and np.isinf(ub[6:8]).all()  # week 2: shed base load <= 0, fab energy free
    assert np.isinf(ub[8]) and (ub[10:12] == 0).all()  # week 3: fab energy <= 0
    assert np.isinf(ub[:4]).all() and np.isinf(ub[12:]).all()
    ub, changed = frame.base_first_bounds(model, x, grids, "zero", model.ub)
    assert changed == 3 and (ub[[6, 7, 10, 11, 14, 15]] == 0).all() and np.isinf(ub[[4, 8, 12]]).all()


def test_two_pass_runs_and_only_adds_bounds(frame):
    """On Small dev episode 0, pass 2 runs in some week and never lowers the window's optimum (it only adds bounds).

    The CPU guard is lifted: the package's own planner runs in this process, and its HiGHS threads can spin into the
    process CPU the guard reads (a local artefact; the test is about the bounds, not the timing).
    """
    import copy

    frame.TWO_PASS_GUARD = {2.0: 1e9, 4.0: 1e9}

    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.information.flat import FlatLayout
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
    from shockbench_flow.policies.registry import PolicyContext
    from shockbench_flow_agent.convert import agent_config, observation_dict

    weeks, pairs = 6, []

    class Stop(Exception):
        pass

    class Both(MpcDet):
        def reset(self, static, obs, policy_seed):
            super().reset(static, obs, policy_seed)
            self.layout = FlatLayout.from_static(static, None)
            cfg = agent_config(copy.deepcopy(static), policy_seed, self.layout, observation_dict(self.layout, obs))
            self.one, self.two = frame.Planner(cfg), frame.Planner(cfg, base_first="fix")

        def act(self, obs):
            action = super().act(obs)
            J = []
            for p in (self.one, self.two):
                s = p.observe(observation_dict(self.layout, copy.deepcopy(obs)))
                assert p.solve(s, p.window(s, p.default_H(s))) is not None
                J.append(p.last_objective)
            pairs.append(J)
            if len(pairs) >= weeks:
                raise Stop
            return action

    inst, _params = task_generator("small")
    omega = scenario("small", 0)
    policy = Both(MpcDetParams(), PolicyContext())
    with pytest.raises(Stop):
        rollout(inst, policy, omega, "standard", 0, marks=compute_marks(inst, omega), fallback=None)
    assert policy.two.pass2["used"] >= 1, policy.two.pass2
    assert all(j2 >= j1 - 1e-7 * max(1.0, abs(j1)) for j1, j2 in pairs), pairs
    with pytest.raises(ValueError):
        frame.Planner(policy.one.cfg, base_first="both")

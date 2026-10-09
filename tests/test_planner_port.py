"""The planner vendored in agents/mine (and its copies in the other LP agents, extras off) reaches mpc_det's optimum.

The agent receives the Dict observation; the package's planner the server's wire observation. Both build the window
LP of the same week and must reach the same optimum (the plans may differ where the LP has several optimal plans).
After ``scripts/vendor_planner.py`` or an update of shockbench-flow, this is the test to rerun.
"""

import pytest

from tests.conftest import ROOT


AGENTS = ("mine", "twopass", "twopass48", "belief48", "milp48", "milp48t", "twopass48_credit", "milp48_eco")


class _Stop(Exception):
    pass


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize(("task", "episode", "weeks"), [("small", 0, 5), ("tiny", 29, 8)])
def test_same_optimum_as_the_package(task, episode, weeks, agent):
    import copy
    import sys

    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.information.flat import FlatLayout
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
    from shockbench_flow.policies.registry import PolicyContext
    from shockbench_flow_agent.convert import agent_config, observation_dict
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    load_agent_class(ROOT / "agents" / agent, "port_agent")
    Planner = sys.modules["frame"].Planner
    gaps = []

    class Both(MpcDet):
        def reset(self, static, obs, policy_seed):
            super().reset(static, obs, policy_seed)
            self.layout = FlatLayout.from_static(static, None)
            cfg = agent_config(copy.deepcopy(static), policy_seed, self.layout, observation_dict(self.layout, obs))
            self.port = Planner(cfg)

        def _record(self, model, res):
            self.ref = res.objective

        def act(self, obs):
            mine = observation_dict(self.layout, copy.deepcopy(obs))
            action = super().act(obs)
            s = self.port.observe(mine)
            assert self.port.solve(s, self.port.window(s, self.port.default_H(s))) is not None
            gaps.append(abs(self.port.last_objective - self.ref) / max(1.0, abs(self.ref)))
            if len(gaps) >= weeks:
                raise _Stop
            return action

    inst, _params = task_generator(task)
    omega = scenario(task, episode)
    try:
        with pytest.raises(_Stop):
            rollout(
                inst,
                Both(MpcDetParams(), PolicyContext()),
                omega,
                "standard",
                0,
                marks=compute_marks(inst, omega),
                fallback=None,
            )
    finally:
        unload_agent()
    assert len(gaps) == weeks and max(gaps) < 1e-9, gaps

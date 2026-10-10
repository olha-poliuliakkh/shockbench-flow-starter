"""scripts/build_dispatch_agent.py: one submission, a different agent per network."""

import importlib.util
import json

import numpy as np
import pytest

from tests.conftest import ROOT


@pytest.fixture
def build():
    spec = importlib.util.spec_from_file_location("build_dispatch_agent", ROOT / "scripts/build_dispatch_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build


def _reset(task: str):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow_gym import agent_config_from_reset

    from sbf_starter import env_id

    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": 0})
    return obs, agent_config_from_reset(env, obs, info)


def test_each_network_gets_its_agent(build, tmp_path):
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    folder = build(out=str(tmp_path / "dispatch"))
    sources = json.loads((folder / "SOURCES.json").read_text())
    assert sources["sources"] == {"small": "agents/compact_hierarchical", "full": "agents/twopass48_credit"}
    assert (folder / "small" / "params.json").read_text() == (
        ROOT / "agents/compact_hierarchical/params.json"
    ).read_text()
    assert (folder / "compact_hier").is_dir() and (folder / "frame.py").is_file() and (folder / "sbfplan").is_dir()
    try:
        cls = load_agent_class(folder, "dispatch_agent")
        for task, impl in (("small", "final_dispatch_small"), ("full", "final_dispatch_full")):
            obs, config = _reset(task)
            agent = cls(config)
            assert type(agent.impl).__module__ == impl
            flows = agent.act(obs)["flows"]
            S = config["spaces"]["action"]["flows"]["shape"][0]
            assert flows.shape == (S,) and np.isfinite(flows).all() and (flows >= 0).all()
    finally:
        unload_agent()


def test_two_sources_with_different_same_named_modules_stop_the_build(build, tmp_path):
    for name, text in (("a", "X = 1\n"), ("b", "X = 2\n")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "agent.py").write_text("class Agent:\n    pass\n")
        (tmp_path / name / "helper.py").write_text(text)
    with pytest.raises(ValueError, match="helper.py"):
        build(small=str(tmp_path / "a"), full=str(tmp_path / "b"), out=str(tmp_path / "out"))

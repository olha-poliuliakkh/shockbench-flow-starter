"""ShockBench-Flow agent: a different planner per network, each the best measured on it.

- Small (T <= 52): ``small/agent.py``, a copy of ``agents/compact_hierarchical``.
- Full: ``full/agent.py``, a copy of ``agents/twopass48_credit``.

Built by ``scripts/build_dispatch_agent.py`` (``SOURCES.json`` lists the sources); rebuild instead of editing the
copies. Both planners are imported here, at module level, so their imports never count toward a week's CPU.
"""

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SMALL_MAX_T = 52


def _load(sub: str, name: str):
    if str(HERE) not in sys.path:  # the shared modules sit beside this file
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / sub / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Agent


SmallAgent = _load("small", "final_dispatch_small")
FullAgent = _load("full", "final_dispatch_full")


class Agent:
    def __init__(self, config):
        impl = SmallAgent if int(config["T"]) <= SMALL_MAX_T else FullAgent
        self.impl = impl(config)

    def act(self, observation):
        return self.impl.act(observation)

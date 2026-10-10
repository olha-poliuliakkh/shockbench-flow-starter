"""The agent kit of ShockBench-Flow: the submission contract, the local evaluation and the scoring API.

A submission is a folder (zipped for upload) holding ``agent.py``, which defines ``class Agent`` with
``__init__(self, config)`` and ``act(self, observation)``. The scoring server runs it in a container, one per
episode, behind this package's shim; the same shim runs it here, so a local score is the server's on the same
episodes.

- **Rules**: ``LIMITS``, the limits the server applies (the per-week wall clock and CPU budget, the start-up, the
  reply size, the per-episode kill timer, the container and the submission size).
- **Check**: ``submission.check_zip`` (or ``python -m shockbench_flow_agent.submission my_agent.zip``) checks a zip as
  the server does, without importing it; ``build_submission`` zips a folder.
- **Score**: ``EpisodeSet.build(task, "dev")`` fixes a set of episodes and caches their reference costs, then
  ``score`` and ``compare`` agents on it (``Score``, ``Comparison``; ``quick=True`` for seconds instead of minutes);
  ``evaluate`` and ``report`` give one summary dict and its report.
- **Time**: ``play_isolated`` plays an episode in a child process that has only the scoring image's packages and
  reports each week's CPU seconds; ``play_container`` plays it in a local copy of the scoring container
  (``build_image``) under the server's CPU meter.
- **The contract's pieces**: ``agent_config`` (the ``config`` of ``Agent(config)``), ``observation_dict`` and
  ``action_to_wire`` (the conversions), ``AgentShim`` and ``load_agent_class``; ``spaces`` documents every
  observation and action field.

The scoring API, the timed runs and the image are loaded on first use: importing the package loads numpy and the
flat view only, as the scoring container does.
"""

import importlib

from shockbench_flow_agent.convert import action_to_wire, agent_config, observation_dict
from shockbench_flow_agent.shim import (
    AGENT_FILE,
    CPU_LOG_VAR,
    IMPORT_FAILED,
    AgentShim,
    load_agent_class,
    serve,
    serve_submission,
    unload_agent,
    warm_up,
)


# names imported on first use (module docstring): name -> (module, attribute)
_LAZY = {
    "evaluate": ("shockbench_flow_agent.local_eval", "evaluate"),
    "report": ("shockbench_flow_agent.local_eval", "report"),
    "EpisodeSet": ("shockbench_flow_agent.scoring", "EpisodeSet"),
    "Score": ("shockbench_flow_agent.scoring", "Score"),
    "Comparison": ("shockbench_flow_agent.scoring", "Comparison"),
    "CPU_BUDGET_S": ("shockbench_flow_agent.scoring", "CPU_BUDGET_S"),
    "QUICK": ("shockbench_flow_agent.scoring", "QUICK"),
    "QUICK_EPISODES": ("shockbench_flow_agent.scoring", "QUICK_EPISODES"),
    "NAIVE_REPLICATIONS": ("shockbench_flow.policies.naive_fq", "REPLICATIONS"),
    "LIMITS": ("shockbench_flow.hosting.limits", "LIMITS"),
    "Limits": ("shockbench_flow.hosting.limits", "Limits"),
    "build_submission": ("shockbench_flow.hosting.submission", "build_submission"),
    "play_isolated": ("shockbench_flow_agent.isolated", "play_isolated"),
    "play_container": ("shockbench_flow_agent.isolated", "play_container"),
    "build_image": ("shockbench_flow_agent.image", "build_image"),
    "build_context": ("shockbench_flow_agent.image", "build_context"),
    "image_distributions": ("shockbench_flow_agent.image", "image_distributions"),
}

__all__ = [
    "AGENT_FILE",
    "CPU_LOG_VAR",
    "IMPORT_FAILED",
    "AgentShim",
    "action_to_wire",
    "agent_config",
    "load_agent_class",
    "observation_dict",
    "serve",
    "serve_submission",
    "unload_agent",
    "warm_up",
    *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))

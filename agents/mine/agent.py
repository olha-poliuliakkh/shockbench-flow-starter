"""ShockBench-Flow agent: the package's LP planner (mpc_det) around four evolvable blocks (docs/EVOLVE_DESIGN.md).

Each week the frozen frame (frame.py, with the vendored planner in sbfplan/, MIT) forecasts the next H weeks, builds
the oracle's LP over that window, solves it with SciPy's HiGHS and executes the plan's first week. The four blocks
below change what the LP sees: the belief carried across weeks, the forecast, edits of the LP's objective, and the
window length. The evolution loop (pipeline/evolve.py) rewrites only the text between the EVOLVE-BLOCK markers and
params.json. A block error re-plans the week on the seed's inputs; a failed solve sends the maximum.
Standard library, numpy and SciPy only; no randomness, so config["policy_seed"] is not needed.
"""

import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401 - for the blocks
from frame import LPEdits, Planner


HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "params.json").read_text()) if (HERE / "params.json").is_file() else {}


def p(name, default):
    """A tunable constant: params.json overrides the default its block declares (the CMA-ES tuner writes it)."""
    return float(PARAMS.get(name, default))


# EVOLVE-BLOCK-START belief
def update_belief(mem, s):
    """Risk carried across weeks (a dict), from s.warnings, s.threads, s.pending, s.closed_for. Seed: no risk."""
    return mem


# EVOLVE-BLOCK-END belief


# EVOLVE-BLOCK-START forecast
def forecast(w, s, mem):
    """Edit the H-week window through its helpers (w.scale_open, w.set_open, w.scale_capacity, w.prohibit,
    w.set_tariff, w.scale_demand, w.scale_supply, w.set_war_risk). Seed: the persistence forecast, unchanged."""
    return w


# EVOLVE-BLOCK-END forecast


# EVOLVE-BLOCK-START objective
def objective(edits, s, mem):
    """Risk premiums (edits.premium), stock floors (edits.stock_floor), cost scales (edits.scale_cost).
    Seed: none."""
    return edits


# EVOLVE-BLOCK-END objective


# EVOLVE-BLOCK-START settings
def settings(s, mem, default_H):
    """The window length for this week. Seed: the package's H = L."""
    return default_H


# EVOLVE-BLOCK-END settings


class Agent:
    def __init__(self, config):
        self.planner = Planner(config)  # instance, L, observed-graph memory, LP-size cap
        self.mem = {}

    def act(self, observation):
        try:
            s = self.planner.observe(observation)
        except Exception as err:  # the frame itself failed: never raise to the scorer
            print(f"observe failed: {type(err).__name__}: {err}; sending the maximum", file=sys.stderr)
            return self.planner.send_max(observation)
        H0 = self.planner.default_H(s)
        try:
            self.mem = update_belief(self.mem, s)
            H = self.planner.clamp_H(s, settings(s, self.mem, H0))
            w = forecast(self.planner.window(s, H), s, self.mem)
            plan = self.planner.solve(s, w, objective(LPEdits(), s, self.mem))
        except Exception as err:  # a block bug costs one re-plan on the seed's inputs
            print(f"week {s.week}: {type(err).__name__}: {err}; seed inputs this week", file=sys.stderr)
            plan = self.planner.solve_safely(s, self.planner.window(s, H0), LPEdits())
        if plan is None:
            print(
                f"week {s.week}: no optimal plan (status {self.planner.last_status}); sending the maximum",
                file=sys.stderr,
            )
            return self.planner.send_max(observation)
        return self.planner.to_dict_action(plan)

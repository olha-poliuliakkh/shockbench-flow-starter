"""ShockBench-Flow agent: agents/milp48 with the MILP's budget and options as frame constants (tuning).

The package's LP planner (mpc_det) around four evolvable blocks (docs/EVOLVE_DESIGN.md). The planner's LP prices the
simulator's base-load-first rule only in week 1, so its plans for weeks 2..H power fabs with energy the grids give to
households first; frame.Planner(base_first=...) re-solves each week with that rule enforced (BASE_FIRST below).

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


BASE_FIRST = "fix"  # the two-pass mode of frame.base_first_bounds: "fix", "zero", or None for agents/mine's planner
LONG_HORIZON = True  # also solve frame.LONG_H's window each week (False: the planner's own window only)


def p(name, default):
    """A tunable constant: params.json overrides the default its block declares (the CMA-ES tuner writes it)."""
    return float(PARAMS.get(name, default))


# EVOLVE-BLOCK-START belief
def update_belief(mem, s):
    """Per-strait probability of a disruption, fused from smoothed warnings and decoy-discounted threads, decayed
    week by week; plus ``global_risk`` (their mean) for the objective block."""
    alpha = p("warning_alpha", 0.5)
    decay = p("risk_decay", 0.9)
    center, slope = p("warn_center", 2.5), p("warn_slope", 3.0)
    n = len(s.chokepoints)
    ema = mem.get("ema", {})
    for key, score in s.warnings.items():  # EMA against the sensor noise
        ema[key] = alpha * score + (1.0 - alpha) * ema.get(key, score)
    region_of = {pos: s.node_region[node] for pos, node in enumerate(s.chokepoints)}
    now = np.zeros(n)
    for pos, node in enumerate(s.chokepoints):
        z = max(ema.get(("chokepoint", node), 0.0), ema.get(("region", region_of[pos]), 0.0))
        now[pos] = 1.0 / (1.0 + np.exp(-slope * (z - center)))  # a smooth score-to-probability map
    for m in s.threads:  # live threads: each raises its strait's risk by its channel's real share
        pos = s.strait_pos(m["target"]) if m["target_kind"] == "chokepoint" and m["target"] is not None else None
        if pos is None or m["channel"] is None:
            continue
        real = 1.0 - s.decoy_share(m["channel"])
        now[pos] = 1.0 - (1.0 - now[pos]) * (1.0 - real)  # noisy-OR fusion
    prev = np.asarray(mem.get("risk", np.zeros(n)), dtype=float)
    risk = 1.0 - (1.0 - decay * prev) * (1.0 - now)  # carried risk decays, fresh evidence adds
    return {"ema": ema, "risk": risk, "global_risk": float(risk.mean())}


# EVOLVE-BLOCK-END belief


# EVOLVE-BLOCK-START forecast
def forecast(w, s, mem):
    """Expected open fractions: each strait's forecast is scaled by (1 - severity_scale * risk) from risk_start on.
    Pending prohibitions are switched on at their stated week with weight pending_trust (1 = the seed's certainty):
    below 1, the prohibition becomes an expected capacity cut on its edge."""
    sev = p("severity_scale", 0.05)
    start = int(round(p("risk_start", 1)))
    for pos, r in enumerate(mem.get("risk", [])):
        w.scale_open(pos, 1.0 - sev * float(r), start=start)
    trust = p("pending_trust", 1.0)
    if trust < 1.0:
        for edge, k, week in s.pending:
            h = max(0, week - s.week)
            w.prohibit(edge, k, on=False, start=h)
            w.scale_capacity(edge, 1.0 - trust, start=h)
    return w


# EVOLVE-BLOCK-END forecast


# EVOLVE-BLOCK-START objective
def objective(edits, s, mem):
    """Floors on stock at the markets, in proportion to global risk: weekly demand x buffer_weeks x global_risk, never
    above floor_cap x what the market holds now (a floor it cannot reach would leave the LP without a plan)."""
    g = float(mem.get("global_risk", 0.0))
    weeks, cap = p("buffer_weeks", 2.0), p("floor_cap", 1.0)
    start = int(round(p("floor_start", 1)))
    end = start + int(round(p("floor_duration", 4)))
    held = dict(zip(zip(s.wire["stock"]["node"], s.wire["stock"]["k"]), s.wire["stock"]["qty"]))
    for d, (node, k) in enumerate(s.demands):
        slot = s.stock_slot(node, k)
        qty = min(float(s.demand_forecast[d, 0]) * weeks * g, cap * float(held.get((node, k), 0.0)))
        if slot is not None and qty > 0.0:
            edits.stock_floor(slot, qty, start=start, end=end)
    return edits


# EVOLVE-BLOCK-END objective


# EVOLVE-BLOCK-START settings
def settings(s, mem, default_H):
    """The window length for this week. Seed: the package's H = L."""
    return default_H


# EVOLVE-BLOCK-END settings


class Agent:
    def __init__(self, config):
        self.planner = Planner(config, base_first=BASE_FIRST, long_horizon=LONG_HORIZON)
        self.mem = {}

    def act(self, observation):
        try:
            s = self.planner.observe(observation)
        except Exception as err:  # the frame itself failed: never raise to the scorer
            print(f"observe failed: {type(err).__name__}: {err}; sending the maximum", file=sys.stderr)
            return self.planner.send_max(observation)
        H0 = self.planner.default_H(s)
        plans, failed_long = {}, False
        try:
            self.mem = update_belief(self.mem, s)
            H0 = self.planner.clamp_H(s, settings(s, self.mem, H0))
            for H, kind in self.planner.attempts(s, H0):  # the planner's own window, then the long one
                w = forecast(self.planner.window(s, H), s, self.mem)
                plan = self.planner.solve(s, w, objective(LPEdits(), s, self.mem), kind)
                if plan is not None:
                    plans[kind] = plan
                failed_long = failed_long or (kind == "long" and plan is None)
                if plan is not None and self.planner.first_success_wins:
                    break
        except Exception as err:  # a block bug costs one re-plan on the seed's inputs
            print(f"week {s.week}: {type(err).__name__}: {err}; seed inputs this week", file=sys.stderr)
            if "short" not in plans:
                plans["short"] = self.planner.solve_safely(
                    s, self.planner.window(s, self.planner.default_H(s)), LPEdits()
                )
        kind = "long" if plans.get("long") is not None else "short" if plans.get("short") is not None else None
        plan = plans.get(kind) if kind else None
        self.planner.note(s, kind if plan is not None else None, failed_long)
        if s.week == self.planner.inst.T:  # one line per episode on stderr: the windows played, pass 2's use
            print(
                f"windows {self.planner.horizons}; pass 2 {self.planner.pass2}; MILP {self.planner.mip}",
                file=sys.stderr,
            )
        if plan is None:
            print(
                f"week {s.week}: no optimal plan (status {self.planner.last_status}); sending the maximum",
                file=sys.stderr,
            )
            return self.planner.send_max(observation)
        return self.planner.to_dict_action(plan)

"""ShockBench-Flow agent: a maritime controller and crisis-aware safety floors over a compact window LP.

Each week, in order (modules in ``compact_hier/``):

1. ``forecast.window``: what the window expects (persistence of this week's network, announced sanctions and
   reopenings, grids and fabs recovering toward nominal).
2. ``maritime.MaritimeController.decide``: before the LP is built, bar dispatches and tanker releases that would wait
   at a closed strait longer than ``reroute_max_wait`` weeks (all of them when no reopening is announced), and cut the
   terminal credit of tanker cargo stranded at a strait closed past the window.
3. ``safety.SafetyFloors.floors``: fuel safety floors, raised at stores whose inbound routes are interrupted (and
   ``crisis_hops`` downstream of them).
4. ``compact_lp.CompactLP.solve``: ``mpc_nobuf``'s compact LP (branch ``mpc-evolve``) with the controller's bounds,
   the week's floors (fading over the episode's last ``floor_taper_weeks``), base-load-first as continuous z in
   [0, 1], and a terminal credit of ``terminal_frac_small`` or ``terminal_frac_full`` x v_k only when the window
   ends before the episode. With ``rule_chips`` 1 a base-stock rule ships wafers and chips (default 0: the LP).

Without an optimum before the CPU deadline, the week sends last week's flows (the maximum in week 1), the rule's
wafer and chip shipments, with every barred route's quantity moved to its open alternatives.

The knobs are ``compact_hier/config.py``'s ``DEFAULTS``, overridden by ``params.json`` beside this file, then by a
preset file (``SBF_PARAMS_FILE=params_baseline.json``) and by ``SBF_PARAM_<KEY>`` environment variables (see
``config.py``). The agent draws no random numbers.
"""

import time
from pathlib import Path

import numpy as np
from compact_hier import config as _config
from compact_hier.compact_lp import CompactLP
from compact_hier.forecast import window
from compact_hier.maritime import MaritimeController
from compact_hier.network import Network
from compact_hier.safety import SafetyFloors


HERE = Path(__file__).resolve().parent
PARAMS = _config.load(HERE)


class Agent:
    def __init__(self, config, params=None):
        """``params``: a complete knob dict in place of ``PARAMS`` (tuning scripts; the server passes none)."""
        start = time.process_time()
        self.params = dict(PARAMS if params is None else params)
        self.net = Network(config, self.params)
        self.maritime = MaritimeController(self.net, self.params)
        self.safety = SafetyFloors(self.net, self.params)
        self.lp = CompactLP(self.net, self.params)
        self.deadline = float(self.params["deadline_small" if self.net.T <= 52 else "deadline_full"])
        self.last_flows = np.zeros(self.net.S)
        self.log = []  # per week: what ran and what it decided (diagnostics only)
        self._init_cpu = time.process_time() - start  # counts toward week 1's budget

    def act(self, observation):
        start = time.process_time()
        o, net = observation, self.net
        t = int(o["week"][0])
        H = max(1, min(net.H, net.T - t + 1))
        mask = o["action_mask"].astype(float) if o["action_mask.observed"][0] else np.ones(net.S)
        self.lp.rule_x, self.lp.status = None, None
        bounds, plan, error = None, None, None
        try:
            win = window(o, t, H, net, self.params)
            bounds = self.maritime.decide(o, t, H)
            floors = self.safety.floors(o, t, H, bounds)
            spent = time.process_time() - start + (self._init_cpu if t == 1 else 0.0)
            if self.deadline - spent > 0.05:
                plan = self.lp.solve(o, t, H, win, bounds, floors, self.deadline - spent)
        except Exception as exc:  # noqa: BLE001 - a failed week falls back here, never to the naive rule
            error = f"{type(exc).__name__}: {exc}"
            plan = None
        release = None
        if plan is not None:
            flows, release = plan
        else:
            flows = self._fallback(o, t, bounds, mask)
        flows = np.nan_to_num(np.maximum(np.asarray(flows, dtype=float), 0.0)) * mask
        self.last_flows = flows
        action = {"flows": flows}
        if release is not None:
            qty, mode = release
            om = o["override_mask"].astype(float) if o["override_mask.observed"][0] else np.ones(len(qty))
            action["override_qty"] = np.nan_to_num(np.maximum(qty, 0.0)) * om
            action["release_mode"] = mode
        self.log.append(
            {
                "week": t,
                "H": H,
                "plan": plan is not None,
                "status": self.lp.status,
                "error": error,
                "cpu": time.process_time() - start,
                "blocked_straits": self.maritime.last.get("blocked_straits", []),
                "barred_slots": self.maritime.last.get("barred_slots", 0),
                "crisis_stores": len(self.safety.last.get("crisis_stores", [])),
                "align": dict(self.lp.last_align),
            }
        )
        return action

    def _fallback(self, o, t, bounds, mask):
        """Last week's flows (the maximum in week 1), the rule's wafer and chip shipments when it ran, and every
        barred route's quantity moved to its open alternatives."""
        net = self.net
        flows = self.last_flows.copy() if t > 1 else net.u0[net.first_edge].copy()
        if self.lp.rule_x is not None:
            flows = np.where(net.rule_slot, self.lp.rule_x, flows)
        if bounds is not None:
            flows = self.maritime.reroute(flows, bounds, mask, o)
        return flows

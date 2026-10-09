"""Fuel safety floors: ``mpc_nobuf``'s soft safety stocks, raised where a store's inbound fuel routes are cut.

The static floor of each fuel store (terminals and grids, nuclear fuel excluded, finite storage) is ``safety_frac`` of
its storage. Each unit-week below it costs ``safety_price`` x v_k through a slack column.

Crisis scaling. A store is in crisis when more than ``crisis_share`` of its inbound routes (every route ending at the
store, tanker lanes included) are interrupted this week. A route is interrupted when:
- the maritime controller bars it;
- it is newly prohibited (masked this week, allowed in an earlier observed week), or an announced prohibition on one
  of its edges takes effect within the window;
- one of its edges has less than ``crisis_cut_below`` of the largest capacity that edge has shown this episode.
Both references are learnt during the episode: ``graph_now.u`` and the instance's ``u0`` are not on one scale, and
sanctions in force from week 1 are the network's normal state, not a crisis. A disruption already acting in week 1
is therefore seen only through the maritime controller and announced prohibitions.
Stores ``crisis_hops`` shipments downstream of a store in crisis are in crisis too: a grid fed by a cut-off terminal
needs its own reserve. A store in crisis keeps ``crisis_frac`` of its storage at ``crisis_price_mult`` times the
price. At a grid, the price over a whole window stays below ``crisis_voll_cap`` of the grid's VOLL, so keeping fuel
back never costs more than the shed it guards against.
"""

import numpy as np


class SafetyFloors:
    def __init__(self, net, params: dict):
        self.net = net
        self.frac = float(params["safety_frac"])
        self.price = float(params["safety_price"])
        self.crisis_on = float(params["crisis"]) >= 0.5 and self.frac > 0
        self.crisis_frac = float(params["crisis_frac"])
        self.mult = float(params["crisis_price_mult"])
        self.share = float(params["crisis_share"])
        self.cut_below = float(params["crisis_cut_below"])
        self.hops = int(params["crisis_hops"])
        self.voll_cap = float(params["crisis_voll_cap"])
        self.base = []  # (j, floor, USD per unit-week short)
        if self.frac > 0:
            for j, (n, k) in enumerate(net.stock):
                if net.node_type[n] not in ("terminal", "grid") or net.k_id[k] == "nucfuel":
                    continue
                if not np.isfinite(net.storage[j]) or self.frac * net.storage[j] <= 0:
                    continue
                self.base.append((j, self.frac * net.storage[j], self.price * net.v[k]))
        stores = {j for j, _f, _p in self.base}
        self.inbound = {j: [s for s in range(net.S) if net.dest_stock[s] == j] for j in stores}
        self.downstream = {
            j: sorted({net.dest_stock[s] for s in net.jout_x.get(j, []) if net.dest_stock[s] in stores} - {j})
            for j in stores
        }
        self.voll = {}  # grid stores: the grid's VOLL
        for j in stores:
            n = net.stock[j][0]
            if n in net.grid_of_node:
                self.voll[j] = net.grid_voll[net.grid_of_node[n]]
        self.u_ref = None  # per edge: the largest observed capacity this episode
        self.allowed_ever = np.zeros(net.S, dtype=bool)  # per slot: allowed by some observed mask this episode
        self.last = {}  # the week's diagnostics

    def interrupted(self, o: dict, t: int, H: int, bounds) -> np.ndarray:
        """Per slot: is the route interrupted this week (module docstring)? Updates the episode's references."""
        net = self.net
        out = bounds.barred_now.copy()
        if o["action_mask.observed"][0]:
            allowed = np.asarray(o["action_mask"]) > 0
            out |= ~allowed & self.allowed_ever
            self.allowed_ever |= allowed
        pending = {
            (int(e), int(k))
            for e, k, w in zip(
                o["pending_prohibitions.edge"], o["pending_prohibitions.k"], o["pending_prohibitions.effective_week"]
            )
            if t <= int(w) < t + H
        }
        u = np.nan_to_num(np.asarray(o["graph_now.u"], dtype=float), nan=0.0, posinf=1e18)
        seen = np.asarray(o["graph_now.u.observed"]) > 0 if "graph_now.u.observed" in o else np.ones(len(u), bool)
        if self.u_ref is None:
            self.u_ref = np.where(seen, u, 0.0)
        cut = seen & (self.u_ref > 0) & (u < self.cut_below * self.u_ref)
        self.u_ref = np.where(seen, np.maximum(self.u_ref, u), self.u_ref)
        if pending or cut.any():
            for s in range(net.S):
                if out[s]:
                    continue
                es, k = net.route_edges[s], int(net.k_s[s])
                out[s] = bool(cut[es].any()) or any((e, k) in pending for e in es)
        return out

    def floors(self, o: dict, t: int, H: int, bounds) -> list:
        """This week's (j, floor, price) for every fuel store."""
        self.last = {"crisis_stores": []}
        if not self.crisis_on:
            return list(self.base)
        net = self.net
        interrupted = self.interrupted(o, t, H, bounds)
        crisis = set()
        for j, routes in self.inbound.items():
            if routes and np.mean(interrupted[routes]) > self.share:
                crisis.add(j)
        frontier = set(crisis)
        for _hop in range(self.hops):
            frontier = {d for j in frontier for d in self.downstream[j]} - crisis
            crisis |= frontier
        out = []
        for j, floor, price in self.base:
            if j in crisis:
                floor = max(floor, min(self.crisis_frac, 1.0) * net.storage[j])
                raised = price * self.mult
                if j in self.voll:
                    raised = min(raised, self.voll_cap * self.voll[j] / max(H, 1))
                price = max(price, raised)  # a crisis never lowers the static price
            out.append((j, floor, price))
        self.last["crisis_stores"] = sorted(crisis)
        return out

"""The maritime controller: routes the week's plan may not use because cargo would strand at a closed strait.

Run before the LP is built. A strait counts as blocked while it is open ``reroute_open_below`` or less. Cargo
dispatched in window week h on a route reaches each strait on it after the route's lead time up to that strait. The
dispatch is barred (an upper bound of 0 on its LP column) when it would wait at a blocked strait longer than
``reroute_max_wait`` weeks: always when no end of closure is announced, and until the announced end is near enough
otherwise. The same holds for tanker releases (override slots) whose path runs into a blocked strait.

Tanker cargo already queued at a strait blocked past the window keeps only ``strand_credit`` of its terminal credit,
so the plan does not count stranded cargo as worth its value.

The LP then routes the fuel over the lanes left open; when no plan comes back, ``reroute`` moves the fallback's flows
off barred routes onto the other routes from the same stock to the same destination.
"""

import numpy as np

from .forecast import announced_ends


class RouteBounds:
    """One week's decisions: per slot (and override slot), the first window week a dispatch is allowed again (0: not
    barred, H: barred all window); the queue pairs whose terminal credit is cut; the blocked straits."""

    def __init__(self, S: int, NO: int):
        self.x_until = np.zeros(S, dtype=int)
        self.y_until = np.zeros(NO, dtype=int)
        self.stranded = set()  # queue pairs at a strait blocked past the window's end
        self.blocked = {}  # strait -> absolute week it is announced to reopen (inf: not announced)

    @property
    def barred_now(self) -> np.ndarray:
        return self.x_until > 0


class MaritimeController:
    def __init__(self, net, params: dict):
        self.net = net
        self.on = float(params["reroute"]) >= 0.5
        self.open_below = float(params["reroute_open_below"])
        self.max_wait = float(params["reroute_max_wait"])
        self.fallback_on = float(params["reroute_fallback"]) >= 0.5
        self.last = {}  # the week's diagnostics

    def decide(self, o: dict, t: int, H: int) -> RouteBounds:
        net = self.net
        out = RouteBounds(net.S, net.NO)
        self.last = {"week": t, "blocked_straits": [], "barred_slots": 0, "barred_releases": 0}
        if not self.on:
            return out
        open_ = o["graph_now.open"]
        ends = announced_ends(o, net, t)
        out.blocked = {
            c: float(ends.get(c, np.inf)) for c, row in net.chk_row.items() if float(open_[row]) <= self.open_below
        }
        if not out.blocked:
            return out
        tau = np.asarray(o["graph_now.tau"], dtype=int)

        def until(positions, edges):
            """First window week from which cargo dispatched on ``edges`` waits at most max_wait at blocked straits."""
            first = 0
            for c, i in positions:
                if c not in out.blocked:
                    continue
                lead = int(tau[edges[: i + 1]].sum())
                reopen = out.blocked[c]
                # dispatched in window week h, the cargo reaches c in week t + h + lead and waits reopen - that
                if not np.isfinite(reopen):
                    return H
                first = max(first, int(np.ceil(reopen - t - lead - self.max_wait)))
            return min(H, max(0, first))

        for s in range(net.S):
            if net.route_chk_pos[s]:
                out.x_until[s] = until(net.route_chk_pos[s], net.route_edges[s])
        for i in range(net.NO):
            if net.ov[i][0] >= 0 and net.ov_chk_pos[i]:
                out.y_until[i] = until(net.ov_chk_pos[i], net.ov[i][2])
        for p, (c, _k) in enumerate(net.pairs[: net.P]):
            if c in out.blocked and out.blocked[c] >= t + H:
                out.stranded.add(p)
        self.last.update(
            blocked_straits=sorted(out.blocked),
            barred_slots=int((out.x_until > 0).sum()),
            barred_releases=int((out.y_until > 0).sum()),
        )
        return out

    def reroute(self, flows: np.ndarray, bounds: RouteBounds, mask: np.ndarray, o: dict) -> np.ndarray:
        """The fallback's flows with every barred route's quantity moved to its alternatives (same stock, destination
        and commodity; fastest first; within this week's edge capacities), or dropped when none is open."""
        net = self.net
        out = np.asarray(flows, dtype=float).copy()
        barred = np.flatnonzero(bounds.barred_now & (out > 0))
        if not self.fallback_on or barred.size == 0:
            return out
        u_left = np.nan_to_num(np.asarray(o["graph_now.u"], dtype=float), nan=0.0, posinf=1e18)
        if "graph_now.u.observed" in o:
            u_left = np.where(np.asarray(o["graph_now.u.observed"]) > 0, u_left, 1e18)
        barred_set = set(barred.tolist())
        for s in np.flatnonzero(out > 0):
            if s not in barred_set:
                for e in net.cap_edges[s]:
                    u_left[e] -= out[s]
        tau = np.asarray(o["graph_now.tau"], dtype=int)
        moved = 0.0
        for s in barred:
            q, out[s] = float(out[s]), 0.0
            alts = [a for a in net.alternatives[s] if bounds.x_until[a] == 0 and mask[a] > 0]
            alts.sort(key=lambda a: (int(tau[net.route_edges[a]].sum()), len(net.route_chk[a])))
            for a in alts:
                room = min(u_left[e] for e in net.cap_edges[a])
                q_a = min(q, max(room, 0.0))
                if q_a <= 0:
                    continue
                out[a] += q_a
                for e in net.cap_edges[a]:
                    u_left[e] -= q_a
                q -= q_a
                moved += q_a
                if q <= 1e-9:
                    break
        self.last["fallback_moved"] = moved
        return out

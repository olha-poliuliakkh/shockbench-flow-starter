"""What the window expects in each of its weeks: ``mpc_nobuf``'s persistence forecast, unchanged.

Capacities, costs and closures persist as observed this week. Announced sanctions take effect in their week. A strait
with an announced end of closure reopens then, at ``reopen_trust`` of its nominal throughput. Cut grid deliverables and
fab capacities drift back toward nominal with time constants ``tau_grid`` and ``tau_fab``.
"""

import numpy as np


def window(o: dict, t: int, H: int, net, params: dict) -> dict:
    """Arrays with a leading axis of H, plus ``closed`` (straits closed now) and ``reopen`` (strait -> first window
    week it is open again, from the announcements; at most H)."""

    def rep(x):
        return np.repeat(np.asarray(x)[None], H, axis=0)

    proh = rep(o["graph_now.prohibited"]).astype(bool)
    for e, k, w in zip(
        o["pending_prohibitions.edge"], o["pending_prohibitions.k"], o["pending_prohibitions.effective_week"]
    ):
        if w >= t and int(w) - t < H:
            proh[int(w) - t :, int(e), int(k)] = True
    open_ = o["graph_now.open"]
    closed_below = float(params["closed_below"])
    reopen = {}
    for c, w in announced_ends(o, net, t).items():
        reopen[c] = min(H, w - t)
    closed = {c for c, row in net.chk_row.items() if float(open_[row]) <= closed_below}
    chk_cap = {}  # pool -> (H, chokepoint row) throughput
    for pool in ("tb", "ct"):
        cap = np.zeros((H, len(net.chk_row)))
        for c, row in net.chk_row.items():
            cap[:, row] = 0.0 if open_[row] <= closed_below else max(0.0, float(o[f"graph_now.kappa.{pool}"][row]))
            if c in reopen:
                p = net.chk_params[c]
                nominal = float(p.get("mu", {}).get(pool, 0.0)) * float(p.get("k_c", 1.0))
                cap[reopen[c] :, row] = np.maximum(cap[reopen[c] :, row], float(params["reopen_trust"]) * nominal)
        chk_cap[pool] = cap
    forecast = o["demand_forecast.qty"]
    demand = np.stack([forecast[:, min(h, forecast.shape[1] - 1)] for h in range(H)])

    def recover(now, nominal, tau):  # a cut level drifts back toward nominal: nom - (nom - now) exp(-h / tau)
        now = np.asarray(now, dtype=float)
        if tau <= 0:
            return rep(now)
        gap = np.maximum(nominal - now, 0.0)
        return now[None] + gap[None] * (1.0 - np.exp(-np.arange(H)[:, None] / tau))

    return {
        "c": rep(o["graph_now.c"]),
        "tariff": rep(o["graph_now.tariff"]),
        "proh": proh,
        "war": rep(o["graph_now.war_risk"]),
        "u": rep(o["graph_now.u"]),
        "chk_cap": chk_cap,
        "supply": rep(o["graph_now.supply.avail"]),
        "fab_cap": recover(o["graph_now.fab.cap_eff"], net.cap_nom, float(params["tau_fab"])),
        "osat_thr": rep(o["graph_now.osat.thr_eff"]),
        "y_bar": rep(o["graph_now.grid.y_bar"]),
        "G_bar": recover(o["graph_now.grid.G_bar"], net.G_nom, float(params["tau_grid"])),
        "demand": demand,
        "closed": closed,
        "reopen": reopen,
    }


def announced_ends(o: dict, net, t: int) -> dict:
    """Strait -> the earliest announced end-of-closure week after t (absolute weeks; straits without one are absent)."""
    out = {}
    if "closure_end.chokepoint" not in o:
        return out
    ends = zip(o["closure_end.chokepoint"], o["closure_end.end_week"], o["closure_end.end_week.observed"])
    for c, w, ok in ends:
        if ok and int(c) in net.chk_row and int(w) > t:
            out[int(c)] = min(out.get(int(c), int(w)), int(w))
    return out

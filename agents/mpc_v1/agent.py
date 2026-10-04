"""MPC: every week, solve a small linear program over the next H weeks and send its first week.

A simplified copy of the organisers' ``mpc_det`` in the server's packages (numpy, scipy). The model: flows on the
action slots (a slot is a route: an edge, or a lane through straits, with its lead time, freight, tariffs and
war-risk surcharge), stock per (node, commodity), supply lifts at sources, lot starts at fabs and packaging at OSATs
(fixed lead times), fuel burnt by the grids, demand at the sinks (served or lost at the penalty), disposal above
storage. Capacities and closures persist over the window as observed this week; announced sanctions take effect in
their week. Stock left at the window's end is credited a share of its customs value, so the plan keeps shipping.
Numbers in ``PARAMS`` (overridden by a ``params.json`` beside this file) are the knobs a search can turn.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
PARAMS = {
    "horizon": 16,  # weeks of the window
    "terminal_frac": 0.5,  # end-of-window value of stock and of goods still travelling, as a share of v_k
    "shortage_weight": 1.0,  # multiplies the shortage penalty pi
    "shed_weight": 1.0,  # multiplies the grids' value of lost load
    "closed_below": 0.05,  # a strait this open or less carries nothing in the forecast
    "holding_weight": 1.0,  # multiplies holding costs
    "_bounds": {"horizon": [4, 30], "terminal_frac": [0, 1.5], "closed_below": [0, 0.9]},
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())


class Agent:
    def __init__(self, config=None):
        st, lay = config["static"], config["layout"]
        inst = st["instance"]
        self.T = int(config["T"])
        self.H = int(PARAMS["horizon"])
        nodes, edges, lanes, slots = st["nodes"], st["edges"], st["lanes"], st["action_slots"]
        by_id = {n["id"]: n for n in inst["nodes"]}
        self.node_id, self.k_id = nodes["id"], st["commodities"]["id"]
        self.v = np.array(st["commodities"]["v"], dtype=float)
        self.disposal = np.array([c.get("disposal_cost", 0.0) for c in inst["commodities"]], dtype=float)
        self.pool = st["commodities"]["pool"]

        # stock slots: (node, k) -> j, with holding, storage and salvage from the instance
        self.stock = [tuple(x) for x in lay["stock_slots"]]
        self.j_of = {nk: j for j, nk in enumerate(self.stock)}
        J = len(self.stock)
        self.holding, self.storage, self.salvage = np.zeros(J), np.full(J, np.inf), np.zeros(J)
        for j, (n, k) in enumerate(self.stock):
            s = by_id[self.node_id[n]].get("stock", {}).get(self.k_id[k], {})
            self.holding[j] = s.get("holding_cost") or 0.0
            self.storage[j] = s.get("storage") if s.get("storage") is not None else np.inf
            self.salvage[j] = s.get("salvage") or 0.0
        self.value = np.maximum(self.salvage, PARAMS["terminal_frac"] * self.v[[k for _, k in self.stock]])
        self.holding *= PARAMS["holding_weight"]

        # routes: one per action slot
        S = len(slots["edge"])
        self.S = S
        self.route_edges, self.route_chk, self.j_out, self.j_in, self.k_s = [], [], [], [], []
        for s in range(S):
            e, k, lane = slots["edge"][s], slots["k"][s], slots["lane"][s]
            es = list(lanes["edges"][lane]) if lane is not None else [e]
            cs = list(lanes["chokepoints"][lane]) if lane is not None else []
            self.route_edges.append(es)
            self.route_chk.append(cs)
            self.k_s.append(k)
            self.j_out.append(self.j_of.get((edges["tail"][es[0]], k), -1))
            self.j_in.append(self.j_of.get((edges["head"][es[-1]], k), -1))
        self.k_s = np.array(self.k_s)
        self.used_edges = sorted({e for es in self.route_edges for e in es})
        self.chk_row = {c: i for i, c in enumerate(lay["chokepoints"])}
        self.chk_pairs = sorted({(c, self.pool[k]) for cs, k in zip(self.route_chk, self.k_s) for c in cs})
        self.lane_edges = lanes["edges"]
        self.edge_head, self.edge_tail = edges["head"], edges["tail"]

        # supply, fabs, OSATs, grids, demands
        self.supply_j = [self.j_of.get(tuple(x), -1) for x in lay["supply_slots"]]
        self.fabs = []
        for f, n in enumerate(lay["fabs"]):
            p = by_id[self.node_id[n]]["fab"]
            k_in, k_out = self.k_id.index(p["input"]), self.k_id.index(p["product"])
            self.fabs.append((self.j_of.get((n, k_in), -1), self.j_of.get((n, k_out), -1), int(p["tau"])))
        self.osats, self.osat_of = [], []  # one entry per (osat, raw k): (j_raw, j_packaged, tau, osat ordinal)
        for o, n in enumerate(lay["osats"]):
            p = by_id[self.node_id[n]]["osat"]
            for raw, packaged in p["packages"].items():
                j_raw, j_pk = (
                    self.j_of.get((n, self.k_id.index(raw)), -1),
                    self.j_of.get((n, self.k_id.index(packaged)), -1),
                )
                self.osats.append((j_raw, j_pk, int(p["tau"]), o))
        self.grid_burn = []  # (grid ordinal, j of the fuel slot, share of the base load, voll)
        for g, n in enumerate(lay["grids"]):
            p = by_id[self.node_id[n]]["grid"]
            for fuel, share in p["shares"].items():
                if fuel in self.k_id and (n, self.k_id.index(fuel)) in self.j_of:
                    self.grid_burn.append((g, self.j_of[(n, self.k_id.index(fuel))], float(share), float(p["voll"])))
        sinks = st["sinks"]
        pi = {(sinks["node"][i], sinks["k"][i]): sinks["pi"][i] for i in range(len(sinks["node"]))}
        self.demands = [
            (self.j_of.get(tuple(x), -1), pi.get(tuple(x), 0.0) * PARAMS["shortage_weight"]) for x in lay["demands"]
        ]
        self.lot_keys = [tuple(x) for x in lay.get("lot_keys", [])]
        self.chk_params = {c: by_id[self.node_id[c]].get("chokepoint", {}) for c in lay["chokepoints"]}
        self.n_override = config["spaces"]["action"]["override_qty"]["shape"]
        self.n_pairs = config["spaces"]["action"]["release_mode"]["shape"]
        self.u0 = np.array([u if u is not None else 0.0 for u in edges["u0"]], dtype=float)
        self.last_flows = np.zeros(S)

    # ------------------------------------------------------------------------------------------------ the week

    def act(self, observation):
        o = observation
        t = int(o["week"][0])
        H = max(1, min(self.H, self.T - t + 1))
        mask = o["action_mask"].astype(float)
        try:
            flows = self._solve(o, t, H)
        except Exception:  # noqa: BLE001 - a failed solve must not hand the week to the naive rule
            flows = None
        if flows is None:
            flows = self.last_flows if t > 1 else self.u0[[es[0] for es in self.route_edges]]
        flows = np.nan_to_num(np.maximum(flows, 0.0)) * mask
        self.last_flows = flows
        return {"flows": flows}

    def _solve(self, o, t, H):
        S, J = self.S, len(self.stock)
        tau, c_e, u = o["graph_now.tau"], o["graph_now.c"], o["graph_now.u"]
        proh, tariff = o["graph_now.prohibited"], o["graph_now.tariff"]
        open_, war = o["graph_now.open"], o["graph_now.war_risk"]
        kappa = {"tb": o["graph_now.kappa.tb"], "ct": o["graph_now.kappa.ct"]}
        v = self.v

        # routes this week: lead, cost, blocked weeks
        lead = np.array([int(sum(tau[e] for e in es)) for es in self.route_edges])
        cost = np.zeros(S)
        blocked_from = np.full(S, H)  # first window week a route is prohibited (H: never)
        pend = {}
        for e, k, w in zip(
            o["pending_prohibitions.edge"], o["pending_prohibitions.k"], o["pending_prohibitions.effective_week"]
        ):
            if w >= t:
                pend[(int(e), int(k))] = min(pend.get((int(e), int(k)), 10**9), int(w) - t)
        for s, (es, cs, k) in enumerate(zip(self.route_edges, self.route_chk, self.k_s)):
            cost[s] = sum(c_e[e] + tariff[e, k] * v[k] for e in es)
            for c in cs:
                cls = int(war[self.chk_row[c]])
                wr = self.chk_params[c].get("war_risk_cost", {}).get(self.k_id[k])
                if wr:
                    cost[s] += wr[min(cls, len(wr) - 1)]
            if any(proh[e, k] for e in es):
                blocked_from[s] = 0
            else:
                blocked_from[s] = min([H] + [pend[(e, k)] for e in es if (e, k) in pend])
        if o["action_mask.observed"][0]:
            blocked_from = np.where(o["action_mask"] == 0, np.minimum(blocked_from, 0), blocked_from)

        # known arrivals into stock slots per window week
        arr = np.zeros((H, J))

        def add(j, h, q):
            if j >= 0 and 0 <= h < H and q > 0:
                arr[h, j] += q

        for e, k, lane, q, w, ok in zip(
            o["pipeline.edge"],
            o["pipeline.k"],
            o["pipeline.lane"],
            o["pipeline.qty"],
            o["pipeline.arrival_week"],
            o["pipeline.qty.observed"],
        ):
            if not ok or q <= 0:
                continue
            e, k = int(e), int(k)
            if (
                o["pipeline.lane.observed"] is not None
                and lane >= 0
                and int(lane) < len(self.lane_edges)
                and e in self.lane_edges[int(lane)]
            ):
                es = self.lane_edges[int(lane)]
                rest = es[es.index(e) + 1 :]
                add(
                    self.j_of.get((self.edge_head[es[-1]], k), -1),
                    int(w) - t + int(sum(tau[x] for x in rest)),
                    float(q),
                )
            else:
                add(self.j_of.get((self.edge_head[e], k), -1), int(w) - t, float(q))
        if self.lot_keys and "queue_lots.qty" in o:
            qty = o["queue_lots.qty"].sum(axis=1)
            for (c, k, lane, nxt), q in zip(self.lot_keys, qty):
                if q <= 0 or open_[self.chk_row[c]] <= PARAMS["closed_below"]:
                    continue
                es = self.lane_edges[lane]
                rest = es[es.index(nxt) :] if nxt in es else [nxt]
                add(self.j_of.get((self.edge_head[es[-1]], k), -1), int(sum(tau[x] for x in rest)), float(q))
        for n, k, q, w, ok in zip(o["wip.node"], o["wip.k"], o["wip.qty"], o["wip.out_week"], o["wip.qty.observed"]):
            if ok and q > 0:
                add(self.j_of.get((int(n), int(k)), -1), int(w) - t, float(q))

        # columns
        F, M, D, G = len(self.fabs), len(self.osats), len(self.demands), len(self.grid_burn)
        U_ = len(self.supply_j)
        sizes = {"x": S, "I": J, "O": J, "lift": U_, "p": F, "q": M, "U": D, "shed": G}
        off, n = {}, 0
        for name, size in sizes.items():
            off[name] = n
            n += H * size

        def col(name, h, i):
            return off[name] + h * sizes[name] + i

        lb, ub = np.zeros(n), np.full(n, np.inf)
        cost_vec = np.zeros(n)
        I0 = o["stock.qty"].astype(float)
        forecast = o["demand_forecast.qty"]
        rows_eq, cols_eq, vals_eq, b_eq = [], [], [], []
        rows_ub, cols_ub, vals_ub, b_ub = [], [], [], []
        r_eq = [0]
        r_ub = [0]

        def eq(entries, rhs):
            for cidx, val in entries:
                rows_eq.append(r_eq[0])
                cols_eq.append(cidx)
                vals_eq.append(val)
            b_eq.append(rhs)
            r_eq[0] += 1

        def le(entries, rhs):
            for cidx, val in entries:
                rows_ub.append(r_ub[0])
                cols_ub.append(cidx)
                vals_ub.append(val)
            b_ub.append(rhs)
            r_ub[0] += 1

        balance = [[[] for _ in range(J)] for _ in range(H)]  # entries per (h, j)
        rhs = arr.copy()
        for h in range(H):
            for j in range(J):
                balance[h][j].append((col("I", h, j), 1.0))
                if h > 0:
                    balance[h][j].append((col("I", h - 1, j), -1.0))
                else:
                    rhs[0, j] += I0[j]
                balance[h][j].append((col("O", h, j), 1.0))
                ub[col("I", h, j)] = self.storage[j]
                cost_vec[col("I", h, j)] = self.holding[j]
                cost_vec[col("O", h, j)] = self.disposal[self.stock[j][1]]
            for s in range(S):
                cx = col("x", h, s)
                if h >= blocked_from[s] or self.j_out[s] < 0:
                    ub[cx] = 0.0
                cost_vec[cx] = cost[s]
                balance[h][self.j_out[s]].append((cx, 1.0))
                ha = h + lead[s]
                if self.j_in[s] >= 0:
                    if ha < H:
                        balance[ha][self.j_in[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= self.value[self.j_in[s]]
            for i, j in enumerate(self.supply_j):
                if j >= 0:
                    ub[col("lift", h, i)] = max(0.0, float(o["graph_now.supply.avail"][i]))
                    balance[h][j].append((col("lift", h, i), -1.0))
            for f, (j_in, j_out, tau_f) in enumerate(self.fabs):
                cp = col("p", h, f)
                ub[cp] = max(0.0, float(o["graph_now.fab.cap_eff"][f])) if j_in >= 0 else 0.0
                if j_in >= 0:
                    balance[h][j_in].append((cp, 1.0))
                if j_out >= 0:
                    if h + tau_f < H:
                        balance[h + tau_f][j_out].append((cp, -1.0))
                    else:
                        cost_vec[cp] -= self.value[j_out]
            for m, (j_raw, j_pk, tau_o, osat) in enumerate(self.osats):
                cq = col("q", h, m)
                if j_raw < 0:
                    ub[cq] = 0.0
                    continue
                balance[h][j_raw].append((cq, 1.0))
                if j_pk >= 0:
                    if h + tau_o < H:
                        balance[h + tau_o][j_pk].append((cq, -1.0))
                    else:
                        cost_vec[cq] -= self.value[j_pk]
            for osat in range(len(o["graph_now.osat.thr_eff"])):
                le(
                    [(col("q", h, m), 1.0) for m, e in enumerate(self.osats) if e[3] == osat],
                    max(0.0, float(o["graph_now.osat.thr_eff"][osat])),
                )
            for d, (j, pi) in enumerate(self.demands):
                dem = max(0.0, float(forecast[d, min(h, forecast.shape[1] - 1)]))
                cu = col("U", h, d)
                ub[cu] = dem
                cost_vec[cu] = pi
                if j >= 0:
                    rhs[h, j] -= dem
                    balance[h][j].append((cu, -1.0))
            for g, (grid, j, share, voll) in enumerate(self.grid_burn):
                burn = share * max(0.0, float(o["graph_now.grid.y_bar"][grid]))
                cs_ = col("shed", h, g)
                ub[cs_] = burn
                cost_vec[cs_] = voll * PARAMS["shed_weight"]
                rhs[h, j] -= burn
                balance[h][j].append((cs_, -1.0))
            # dispatch draws on the stock on hand at the start of the week
            for j in set(self.j_out):
                if j < 0:
                    continue
                ent = [(col("x", h, s), 1.0) for s in range(S) if self.j_out[s] == j]
                if h > 0:
                    le(ent + [(col("I", h - 1, j), -1.0)], 0.0)
                else:
                    le(ent, I0[j])
            for e in self.used_edges:
                ent = [(col("x", h, s), 1.0) for s in range(S) if e in self.route_edges[s]]
                le(ent, max(0.0, float(u[e])))
            for c, pool in self.chk_pairs:
                row = self.chk_row[c]
                cap = 0.0 if open_[row] <= PARAMS["closed_below"] else max(0.0, float(kappa[pool][row]))
                ent = [
                    (col("x", h, s), 1.0) for s in range(S) if c in self.route_chk[s] and self.pool[self.k_s[s]] == pool
                ]
                le(ent, cap)
        for h in range(H):
            for j in range(J):
                eq(balance[h][j], rhs[h, j])
        for j in range(J):
            cost_vec[col("I", H - 1, j)] -= self.value[j]

        A_eq = sp.csr_matrix((vals_eq, (rows_eq, cols_eq)), shape=(r_eq[0], n))
        A_ub = sp.csr_matrix((vals_ub, (rows_ub, cols_ub)), shape=(r_ub[0], n))
        ub = np.maximum(ub, lb)
        res = linprog(
            cost_vec,
            A_ub=A_ub,
            b_ub=np.array(b_ub),
            A_eq=A_eq,
            b_eq=np.array(b_eq),
            bounds=np.c_[lb, ub],
            method="highs",
        )
        if res.status != 0 or res.x is None:
            return None
        return res.x[off["x"] : off["x"] + S]

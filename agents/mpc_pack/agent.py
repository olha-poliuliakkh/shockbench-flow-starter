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
    "reopen_trust": 0.8,  # share of a strait's nominal throughput expected from its announced end-of-closure week
    "queue_scale": 0.0,  # queue delay and holding expected at a partly open strait (1.0 scored worse on Small dev)
    "recover_weeks": 0.0,  # a cut edge capacity expected back at nominal with this time constant (4.0 scored worse)
    "buffer_weight": 1.0,  # weight of the rationing buffer: a grid's rationed fuel is kept at psi x ibar (0: ignore)
    "fab_energy_weight": 1.0,  # fuel the grid burns for its fabs' lot starts, share_k x e_f per wafer (0: ignore)
    "fab_shed_tol": 0.0,  # base load first: fab energy only while shed < this share of load (0 off; 0.01 scored worse)
    "fab_w_cost": 1e8,  # USD per GWh of fab energy the plan would take while shedding (above any lot's worth)
    "pipeline_closure": 1.0,  # 1: cargo bound for a closed strait arrives after its announced reopening, else never
    "tanker_control": 1.0,  # 1: the LP releases tanker cargo queued at straits itself (release_mode 1, override_qty)
    "osat_auto_pack": 1.0,  # price per unit-week of raw chips held at an OSAT, as a share of v (the simulator packages
    # every raw chip on hand up to the throughput, so a plan that keeps raw chips there is not what happens; 0: off)
    "_fixed": ["fab_shed_tol", "fab_w_cost", "pipeline_closure", "tanker_control"],  # a search leaves these as they are
    "_bounds": {
        "horizon": [4, 30],
        "shortage_weight": [0.2, 3],
        "shed_weight": [0.2, 3],
        "holding_weight": [0, 3],
        "terminal_frac": [0, 1.5],
        "closed_below": [0, 0.9],
        "reopen_trust": [0, 1],
        "queue_scale": [0, 3],
        "recover_weeks": [0, 30],
        "buffer_weight": [0, 3],
        "fab_energy_weight": [0, 2],
        "fab_shed_tol": [0, 0.2],
        "pipeline_closure": [0, 1],
        "tanker_control": [0, 1],
    },
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
        self.chk_row = {c: i for i, c in enumerate(lay["chokepoints"])}
        self.lane_edges = lanes["edges"]
        self.edge_head, self.edge_tail = edges["head"], edges["tail"]

        # supply, fabs, OSATs, grids, demands
        self.supply_j = [self.j_of.get(tuple(x), -1) for x in lay["supply_slots"]]
        self.fabs = []
        for f, n in enumerate(lay["fabs"]):
            p = by_id[self.node_id[n]]["fab"]
            k_in, k_out = self.k_id.index(p["input"]), self.k_id.index(p["product"])
            self.fabs.append((self.j_of.get((n, k_in), -1), self.j_of.get((n, k_out), -1), int(p["tau"])))
        # each fab's grid ordinal and energy per wafer: the grid burns share_k x e_f x p_f of each of its fuels
        self.fab_grid, self.fab_e = [], []
        for n in lay["fabs"]:
            p = by_id[self.node_id[n]]["fab"]
            gnode = self.node_id.index(p["grid"]) if p.get("grid") in self.node_id else -1
            self.fab_grid.append(lay["grids"].index(gnode) if gnode in lay["grids"] else -1)
            self.fab_e.append(float(p.get("e") or 0.0))
        self.osats, self.osat_of = [], []  # one entry per (osat, raw k): (j_raw, j_packaged, tau, osat ordinal)
        for o, n in enumerate(lay["osats"]):
            p = by_id[self.node_id[n]]["osat"]
            for raw, packaged in p["packages"].items():
                j_raw, j_pk = (
                    self.j_of.get((n, self.k_id.index(raw)), -1),
                    self.j_of.get((n, self.k_id.index(packaged)), -1),
                )
                self.osats.append((j_raw, j_pk, int(p["tau"]), o))
        # raw chips cannot wait at an OSAT: the simulator packages all of them (19); price holding them like losing them
        for j_raw, _j_pk, _tau, _o in self.osats:
            if j_raw >= 0:
                self.holding[j_raw] += PARAMS["osat_auto_pack"] * self.v[self.stock[j_raw][1]]
        # gas rationing: below psi x ibar of its rationed fuel at the end of a week, a grid's segment output of that
        # fuel next week scales by stock / threshold; one unit short costs about voll x share x G_bar / threshold
        self.ration = []  # (j of the rationed fuel's slot, threshold, USD per unit below it)
        psi = float(inst.get("params", {}).get("psi", 0.0))
        for n in lay["grids"]:
            p = by_id[self.node_id[n]]["grid"]
            fuel = p.get("rationed")
            ibar = (p.get("ibar") or {}).get(fuel)
            if fuel in self.k_id and ibar and psi > 0 and (n, self.k_id.index(fuel)) in self.j_of:
                threshold = psi * float(ibar)
                marginal = float(p["voll"]) * float(p["shares"].get(fuel, 0.0)) * float(p["deliverable"]) / threshold
                self.ration.append((self.j_of[(n, self.k_id.index(fuel))], threshold, marginal))
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

        # tanker cargo (commodities with an override): under tanker_control a lane dispatch only reaches the lane's
        # first strait, where it joins the queue (c, k); the LP then releases it on the override slots of (c, k),
        # each a path to the next strait on its lane or to the lane's end
        chk_set = set(lay["chokepoints"])
        self.tanker = [bool(f) for f in st["commodities"]["override"]]
        self.pairs = [tuple(x) for x in lay.get("release_pairs", [])]
        self.pair_of = {p: i for i, p in enumerate(self.pairs)}
        self.tk_on = PARAMS["tanker_control"] >= 0.5 and bool(self.pairs)
        self.tk_dest = [-1] * S  # slot -> queue pair its dispatch joins (-1: a stock slot, as j_in)
        self.cap_edges = [list(es) for es in self.route_edges]  # edges a dispatch occupies this week
        self.cap_chk = [list(cs) for cs in self.route_chk]  # straits whose throughput a dispatch uses
        if self.tk_on:
            for s in range(S):
                es, k = self.route_edges[s], int(self.k_s[s])
                if not self.tanker[k] or slots["lane"][s] is None:
                    continue
                first = next((i for i, e in enumerate(es) if edges["head"][e] in chk_set), None)
                if first is None or (edges["head"][es[first]], k) not in self.pair_of:
                    continue
                self.tk_dest[s] = self.pair_of[(edges["head"][es[first]], k)]
                self.j_in[s] = -1
                self.cap_edges[s] = es[: first + 1]
                self.cap_chk[s] = []  # the throughput is used when the LP releases, not at dispatch
        ov = st["override_slots"]
        self.ov = []  # (pair from, out edge, path edges, stop pair or -1, stop stock slot or -1, straits passed)
        for o in range(len(ov["chokepoint"])):
            c, k, e, lane = ov["chokepoint"][o], ov["k"][o], ov["out_edge"][o], ov["lane"][o]
            full = list(lanes["edges"][lane]) if lane is not None and e in lanes["edges"][lane] else [e]
            path = full[full.index(e) :]
            stop = next((i for i, x in enumerate(path) if edges["head"][x] in chk_set), None)
            if stop is not None:
                path = path[: stop + 1]
                stop_pair, stop_j = self.pair_of.get((edges["head"][path[-1]], k), -1), -1
            else:
                stop_pair, stop_j = -1, self.j_of.get((edges["head"][path[-1]], k), -1)
            passed = [edges["head"][x] for x in path if edges["head"][x] in chk_set]
            self.ov.append((self.pair_of.get((c, k), -1), e, path, stop_pair, stop_j, passed))
        self.used_edges = sorted({e for es in self.cap_edges for e in es} | {ov_[1] for ov_ in self.ov})
        self.chk_pairs = sorted(
            {(c, self.pool[k]) for cs, k in zip(self.cap_chk, self.k_s) for c in cs}
            | ({(c, self.pool[k]) for c, k in self.pairs} if self.tk_on else set())
        )

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
        release = None
        if isinstance(flows, tuple):
            flows, release = flows
        if flows is None:
            flows = self.last_flows if t > 1 else self.u0[[es[0] for es in self.route_edges]]
        flows = np.nan_to_num(np.maximum(flows, 0.0)) * mask
        self.last_flows = flows
        action = {"flows": flows}
        if release is not None:
            qty, mode = release
            om = o["override_mask"].astype(float)
            action["override_qty"] = np.nan_to_num(np.maximum(qty, 0.0)) * om
            action["release_mode"] = mode
        return action

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
        # straits: the weeks a partly open one delays cargo, and the queue holding that costs
        delay, queue_cost = {}, {}  # chokepoint -> extra weeks; (chokepoint, k) -> USD per unit
        for c, row in self.chk_row.items():
            op = float(open_[row])
            if PARAMS["closed_below"] < op < 1.0:
                extra = (1.0 / op - 1.0) * PARAMS["queue_scale"]
                delay[c] = int(np.ceil(extra))
                for k, name in enumerate(self.k_id):
                    qh = self.chk_params[c].get("queue_holding", {}).get(name)
                    if qh:
                        cls = int(war[row])
                        queue_cost[(c, k)] = qh[min(cls, len(qh) - 1)] * extra
        # announced ends of closures: the strait reopens in the window from that week
        reopen = {}  # chokepoint -> first window week it is open again
        if "closure_end.chokepoint" in o:
            ends = zip(o["closure_end.chokepoint"], o["closure_end.end_week"], o["closure_end.end_week.observed"])
            for c, w, ok in ends:
                if ok and int(c) in self.chk_row and int(w) > t:
                    reopen[int(c)] = min(reopen.get(int(c), H), int(w) - t)
        closed = {c for c, row in self.chk_row.items() if float(open_[row]) <= PARAMS["closed_below"]}
        use_pc = PARAMS["pipeline_closure"] >= 0.5

        for s, (es, cs, k) in enumerate(zip(self.cap_edges, self.cap_chk, self.k_s)):
            if self.tk_dest[s] >= 0:  # a tanker lane: the dispatch only runs to the first strait
                lead[s] = int(sum(tau[e] for e in es))
                cs = [self.edge_head[es[-1]]]
            cost[s] = sum(c_e[e] + tariff[e, k] * v[k] for e in es)
            for c in cs:
                cls = int(war[self.chk_row[c]])
                wr = self.chk_params[c].get("war_risk_cost", {}).get(self.k_id[k])
                if wr:
                    cost[s] += wr[min(cls, len(wr) - 1)]
                cost[s] += queue_cost.get((c, k), 0.0)
                lead[s] += delay.get(c, 0)
            if any(proh[e, k] for e in es):
                blocked_from[s] = 0
            else:
                blocked_from[s] = min([H] + [pend[(e, k)] for e in es if (e, k) in pend])
        if o["action_mask.observed"][0]:
            blocked_from = np.where(o["action_mask"] == 0, np.minimum(blocked_from, 0), blocked_from)

        # override slots this week: lead, cost and the first blocked week of each release path
        P, NO = (len(self.pairs), len(self.ov)) if self.tk_on else (0, 0)
        ov_lead, ov_cost, ov_blocked = np.zeros(NO, dtype=int), np.zeros(NO), np.full(NO, H)
        for i, (p_from, e_out, path, _sp, _sj, passed) in enumerate(self.ov[:NO]):
            k = self.pairs[p_from][1] if p_from >= 0 else 0
            ov_lead[i] = int(sum(tau[x] for x in path))
            ov_cost[i] = sum(c_e[x] + tariff[x, k] * v[k] for x in path)
            for c in passed:
                cls = int(war[self.chk_row[c]])
                wr = self.chk_params[c].get("war_risk_cost", {}).get(self.k_id[k])
                if wr:
                    ov_cost[i] += wr[min(cls, len(wr) - 1)]
            if p_from < 0 or any(proh[x, k] for x in path):
                ov_blocked[i] = 0
            else:
                ov_blocked[i] = min([H] + [pend[(x, k)] for x in path if (x, k) in pend])
            if o["override_mask.observed"][0] and o["override_mask"][i] == 0:
                ov_blocked[i] = min(ov_blocked[i], 0)
        q_hold = np.zeros(P)  # queue holding per unit-week at each pair
        for i, (c, k) in enumerate(self.pairs[:P]):
            qh = self.chk_params[c].get("queue_holding", {}).get(self.k_id[k])
            if qh:
                q_hold[i] = qh[min(int(war[self.chk_row[c]]), len(qh) - 1)]

        # known arrivals into stock slots (and tanker queues) per window week
        arr = np.zeros((H, J))
        arrQ = np.zeros((H, max(P, 1)))

        def add(j, h, q):
            if j >= 0 and 0 <= h < H and q > 0:
                arr[h, j] += q

        def add_q(p, h, q):
            if p >= 0 and 0 <= h < H and q > 0:
                arrQ[h, p] += q

        for e, k, lane, lane_ok, q, w, ok in zip(
            o["pipeline.edge"],
            o["pipeline.k"],
            o["pipeline.lane"],
            o["pipeline.lane.observed"],
            o["pipeline.qty"],
            o["pipeline.arrival_week"],
            o["pipeline.qty.observed"],
        ):
            if not ok or q <= 0:
                continue
            e, k = int(e), int(k)
            if self.tk_on and self.tanker[k]:  # tanker cargo: it stops at the next strait, else at the edge's head
                head = self.edge_head[e]
                if (head, k) in self.pair_of:
                    add_q(self.pair_of[(head, k)], int(w) - t, float(q))
                else:
                    add(self.j_of.get((head, k), -1), int(w) - t, float(q))
            elif lane_ok and 0 <= int(lane) < len(self.lane_edges) and e in self.lane_edges[int(lane)]:
                es = self.lane_edges[int(lane)]
                i0 = es.index(e)
                h_arr = int(w) - t + int(sum(tau[x] for x in es[i0 + 1 :]))
                lost = False
                if use_pc:  # a closed strait still ahead holds the cargo until its announced reopening
                    for idx in range(i0, len(es) - 1):
                        c = int(self.edge_head[es[idx]])
                        if c in closed:
                            if c in reopen:
                                h_arr = max(h_arr, reopen[c] + int(sum(tau[x] for x in es[idx + 1 :])))
                            else:
                                lost = True
                if not lost:
                    add(self.j_of.get((self.edge_head[es[-1]], k), -1), h_arr, float(q))
            else:
                add(self.j_of.get((self.edge_head[e], k), -1), int(w) - t, float(q))
        Q0 = np.zeros(max(P, 1))
        if self.tk_on and not self.lot_keys and "queue_lots.chokepoint" in o:  # Tiny: one entry per lot
            for c, k, q, ok in zip(
                o["queue_lots.chokepoint"], o["queue_lots.k"], o["queue_lots.qty"], o["queue_lots.qty.observed"]
            ):
                if ok and q > 0 and (int(c), int(k)) in self.pair_of:
                    Q0[self.pair_of[(int(c), int(k))]] += float(q)
        if self.lot_keys and "queue_lots.qty" in o:
            qty = o["queue_lots.qty"].sum(axis=1)
            for (c, k, lane, nxt), q in zip(self.lot_keys, qty):
                if q <= 0:
                    continue
                if self.tk_on and (c, k) in self.pair_of:
                    Q0[self.pair_of[(c, k)]] += float(q)
                    continue
                es = self.lane_edges[lane]
                rest = es[es.index(nxt) :] if nxt in es else [nxt]
                h_arr = int(sum(tau[x] for x in rest)) + delay.get(c, 0)
                if c in closed:
                    if not (use_pc and c in reopen):
                        continue
                    h_arr += reopen[c]
                add(self.j_of.get((self.edge_head[es[-1]], k), -1), h_arr, float(q))
        for n, k, q, w, ok in zip(o["wip.node"], o["wip.k"], o["wip.qty"], o["wip.out_week"], o["wip.qty.observed"]):
            if ok and q > 0:
                add(self.j_of.get((int(n), int(k)), -1), int(w) - t, float(q))

        # columns
        F, M, D, G = len(self.fabs), len(self.osats), len(self.demands), len(self.grid_burn)
        U_ = len(self.supply_j)
        R = len(self.ration)
        NG = len(o["graph_now.grid.y_bar"])
        sizes = {"x": S, "I": J, "O": J, "lift": U_, "p": F, "q": M, "U": D, "shed": G, "buf": R, "w": NG}
        sizes |= {"Q": P, "y": NO}
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
        qbal = [[[] for _ in range(P)] for _ in range(H)]  # tanker queue balance entries per (h, pair)
        qrhs = arrQ[:, :P].copy()
        for h in range(H):
            for p in range(P):
                cq = col("Q", h, p)
                qbal[h][p].append((cq, 1.0))
                if h > 0:
                    qbal[h][p].append((col("Q", h - 1, p), -1.0))
                else:
                    qrhs[0, p] += Q0[p]
                cost_vec[cq] = q_hold[p]
                if h == H - 1:
                    cost_vec[cq] -= PARAMS["terminal_frac"] * v[self.pairs[p][1]]
            for i, (p_from, e_out, path, stop_p, stop_j, _passed) in enumerate(self.ov[:NO]):
                cy = col("y", h, i)
                if h >= ov_blocked[i] or p_from < 0:
                    ub[cy] = 0.0
                cost_vec[cy] = ov_cost[i]
                qbal[h][p_from].append((cy, 1.0))
                ha = h + ov_lead[i]
                if stop_p >= 0:
                    if ha < H:
                        qbal[ha][stop_p].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= PARAMS["terminal_frac"] * v[self.pairs[stop_p][1]]
                elif stop_j >= 0:
                    if ha < H:
                        balance[ha][stop_j].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= self.value[stop_j]
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
                if self.tk_dest[s] >= 0:
                    if ha < H:
                        qbal[ha][self.tk_dest[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= PARAMS["terminal_frac"] * v[self.k_s[s]]
                elif self.j_in[s] >= 0:
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
                if PARAMS["fab_energy_weight"] > 0:  # the fabs' draw on this fuel, per lot started
                    for f, (fg, e_f) in enumerate(zip(self.fab_grid, self.fab_e)):
                        if fg == grid and e_f > 0:
                            balance[h][j].append((col("p", h, f), share * e_f * PARAMS["fab_energy_weight"]))
                burn = share * max(0.0, float(o["graph_now.grid.y_bar"][grid]))
                cs_ = col("shed", h, g)
                ub[cs_] = burn
                cost_vec[cs_] = voll * PARAMS["shed_weight"]
                rhs[h, j] -= burn
                balance[h][j].append((cs_, -1.0))
            # base load first: a grid's fabs share only the headroom G_bar - y_bar, and get nothing once the grid
            # sheds more than a small tolerance:  sum_f e_f p_f + (headroom / tol) sum_k shed_gk - w <= headroom,
            # with w >= 0 priced above any gain from a lot, so the LP cuts the starts instead of paying it
            if PARAMS["fab_shed_tol"] > 0:
                for grid in range(NG):
                    fabs = [f for f, fg in enumerate(self.fab_grid) if fg == grid and self.fab_e[f] > 0]
                    if not fabs:
                        continue
                    y_bar = max(0.0, float(o["graph_now.grid.y_bar"][grid]))
                    headroom = max(0.0, float(o["graph_now.grid.G_bar"][grid]) - y_bar)
                    tol = max(PARAMS["fab_shed_tol"] * y_bar, 1e-6)
                    cw = col("w", h, grid)
                    cost_vec[cw] = PARAMS["fab_w_cost"]
                    ent = [(col("p", h, f), self.fab_e[f]) for f in fabs] + [(cw, -1.0)]
                    ent += [
                        (col("shed", h, g), headroom / tol)
                        for g, (gr, _j, _s, _v) in enumerate(self.grid_burn)
                        if gr == grid
                    ]
                    le(ent, headroom)
            for i, (j, threshold, marginal) in enumerate(self.ration):  # I[h, j] + buf >= threshold
                cb = col("buf", h, i)
                cost_vec[cb] = marginal * PARAMS["buffer_weight"]
                le([(col("I", h, j), -1.0), (cb, -1.0)], -threshold)
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
                ent = [(col("x", h, s), 1.0) for s in range(S) if e in self.cap_edges[s]]
                ent += [(col("y", h, i), 1.0) for i in range(NO) if self.ov[i][1] == e]
                cap_e = max(0.0, float(u[e]))
                if PARAMS["recover_weeks"] > 0 and cap_e < self.u0[e]:  # a cut recovers toward nominal
                    cap_e += (self.u0[e] - cap_e) * (1.0 - np.exp(-h / PARAMS["recover_weeks"]))
                le(ent, cap_e)
            for c, pool in self.chk_pairs:
                row = self.chk_row[c]
                cap = 0.0 if open_[row] <= PARAMS["closed_below"] else max(0.0, float(kappa[pool][row]))
                if c in reopen and h >= reopen[c]:
                    p = self.chk_params[c]
                    nominal = float(p.get("mu", {}).get(pool, 0.0)) * float(p.get("k_c", 1.0))
                    cap = max(cap, PARAMS["reopen_trust"] * nominal)
                ent = [
                    (col("x", h, s), 1.0) for s in range(S) if c in self.cap_chk[s] and self.pool[self.k_s[s]] == pool
                ]
                ent += [
                    (col("y", h, i), 1.0)
                    for i in range(NO)
                    if self.ov[i][0] >= 0
                    and self.pairs[self.ov[i][0]][0] == c
                    and self.pool[self.pairs[self.ov[i][0]][1]] == pool
                ]
                le(ent, cap)
        for h in range(H):
            for j in range(J):
                eq(balance[h][j], rhs[h, j])
            for p in range(P):
                eq(qbal[h][p], qrhs[h, p])
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
        flows = res.x[off["x"] : off["x"] + S]
        if not self.tk_on:
            return flows
        # the week's tanker releases: release_mode 1 on every pair with a sendable override slot, its quantities
        # from the plan (0 holds); pairs without a sendable slot keep the default release
        qty = np.zeros(self.n_override)
        qty[:NO] = res.x[off["y"] : off["y"] + NO]
        mode = np.zeros(self.n_pairs, dtype=np.int64)
        for i, (p_from, *_rest) in enumerate(self.ov[:NO]):
            if p_from >= 0 and ov_blocked[i] > 0:
                mode[p_from] = 1
        return flows, (qty, mode)

"""The static network, read once from the agent's config: routes, stock slots, plants, grids and tanker queues.

Ported from ``mpc_nobuf``'s ``Agent.__init__``, with the incidence lists the LP builder needs precomputed (which
routes use an edge, a strait's pool, a stock slot), so a week's build costs the size of the LP, not slots x edges.
Nothing here changes during an episode.
"""

import numpy as np


RULE_COMMODITIES = ("wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat")


class Network:
    def __init__(self, config: dict, params: dict):
        st, lay = config["static"], config["layout"]
        inst = st["instance"]
        self.T = int(config["T"])
        self.budget = 2.0 if self.T <= 52 else 4.0
        self.H = int(params["horizon_small"] if self.T <= 52 else params["horizon_full"])
        nodes, edges, lanes, slots = st["nodes"], st["edges"], st["lanes"], st["action_slots"]
        by_id = {n["id"]: n for n in inst["nodes"]}
        self.node_id, self.k_id = nodes["id"], st["commodities"]["id"]
        self.node_type = nodes["type"]
        self.v = np.array(st["commodities"]["v"], dtype=float)
        self.disposal = np.array([c.get("disposal_cost", 0.0) for c in inst["commodities"]], dtype=float)
        self.pool = st["commodities"]["pool"]

        # stock slots: (node, k) -> j, with holding, storage and salvage from the instance
        self.stock = [tuple(x) for x in lay["stock_slots"]]
        self.j_of = {nk: j for j, nk in enumerate(self.stock)}
        J = self.J = len(self.stock)
        self.holding, self.storage, self.salvage = np.zeros(J), np.full(J, np.inf), np.zeros(J)
        for j, (n, k) in enumerate(self.stock):
            s = by_id[self.node_id[n]].get("stock", {}).get(self.k_id[k], {})
            self.holding[j] = s.get("holding_cost") or 0.0
            self.storage[j] = s.get("storage") if s.get("storage") is not None else np.inf
            self.salvage[j] = s.get("salvage") or 0.0
        self.k_of_j = np.array([k for _, k in self.stock], dtype=int)
        # the end-of-window credit while the window ends before T: this network's fraction of v_k, never below salvage
        self.terminal_frac = float(params["terminal_frac_small" if self.T <= 52 else "terminal_frac_full"])
        self.value = np.maximum(self.salvage, self.terminal_frac * self.v[self.k_of_j])
        self.holding *= float(params["holding_weight"])

        # routes: one per action slot (an edge, or a lane through straits)
        S = self.S = len(slots["edge"])
        self.edge_head, self.edge_tail = edges["head"], edges["tail"]
        self.lane_edges = lanes["edges"]
        self.u0 = np.array([u if u is not None else 0.0 for u in edges["u0"]], dtype=float)
        self.route_edges, self.route_chk, self.j_out, self.j_in, self.k_s = [], [], [], [], []
        for s in range(S):
            e, k, lane = slots["edge"][s], slots["k"][s], slots["lane"][s]
            es = list(lanes["edges"][lane]) if lane is not None else [e]
            self.route_edges.append(es)
            self.route_chk.append(list(lanes["chokepoints"][lane]) if lane is not None else [])
            self.k_s.append(k)
            self.j_out.append(self.j_of.get((edges["tail"][es[0]], k), -1))
            self.j_in.append(self.j_of.get((edges["head"][es[-1]], k), -1))
        self.k_s = np.array(self.k_s, dtype=int)
        self.is_lane = [slots["lane"][s] is not None for s in range(S)]
        self.dest_stock = list(self.j_in)  # the stock slot a route ends at, whatever tanker control does to j_in
        self.chk_row = {c: i for i, c in enumerate(lay["chokepoints"])}
        chk_set = set(lay["chokepoints"])
        # the straits each route reaches, with the index of the edge whose head is the strait (lead = edges up to it)
        self.route_chk_pos = [
            [(int(edges["head"][e]), i) for i, e in enumerate(es) if edges["head"][e] in chk_set]
            for es in self.route_edges
        ]
        self.first_edge = np.array([es[0] for es in self.route_edges], dtype=int)

        # supply, fabs, OSATs
        self.supply_j = [self.j_of.get(tuple(x), -1) for x in lay["supply_slots"]]
        self.fabs, self.fab_grid, self.fab_e = [], [], []
        for n in lay["fabs"]:
            p = by_id[self.node_id[n]]["fab"]
            k_in, k_out = self.k_id.index(p["input"]), self.k_id.index(p["product"])
            self.fabs.append((self.j_of.get((n, k_in), -1), self.j_of.get((n, k_out), -1), int(p["tau"])))
            gnode = self.node_id.index(p["grid"]) if p.get("grid") in self.node_id else -1
            self.fab_grid.append(lay["grids"].index(gnode) if gnode in lay["grids"] else -1)
            self.fab_e.append(float(p.get("e") or 0.0))
        self.osats = []  # one entry per (osat, raw k): (j_raw, j_packaged, tau, osat ordinal)
        for o, n in enumerate(lay["osats"]):
            p = by_id[self.node_id[n]]["osat"]
            for raw, packaged in p["packages"].items():
                j_raw = self.j_of.get((n, self.k_id.index(raw)), -1)
                j_pk = self.j_of.get((n, self.k_id.index(packaged)), -1)
                self.osats.append((j_raw, j_pk, int(p["tau"]), o))
        self.n_osat = len(lay["osats"])
        self.osat_ms = [[m for m, e in enumerate(self.osats) if e[3] == o] for o in range(self.n_osat)]

        # grids: gas rationing below psi x ibar; fuel burnt per segment; the share needing no modelled fuel; VOLL
        self.ration = []  # (j of the rationed fuel's slot, threshold, USD per unit below it)
        psi = float(inst.get("params", {}).get("psi", 0.0))
        self.grid_burn = []  # (grid ordinal, j of the fuel slot, share of the base load, voll)
        self.grid_null, self.grid_voll = [], []
        self.grid_node = list(lay["grids"])
        for g, n in enumerate(lay["grids"]):
            p = by_id[self.node_id[n]]["grid"]
            fuel = p.get("rationed")
            ibar = (p.get("ibar") or {}).get(fuel)
            if fuel in self.k_id and ibar and psi > 0 and (n, self.k_id.index(fuel)) in self.j_of:
                threshold = psi * float(ibar)
                marginal = float(p["voll"]) * float(p["shares"].get(fuel, 0.0)) * float(p["deliverable"]) / threshold
                self.ration.append((self.j_of[(n, self.k_id.index(fuel))], threshold, marginal))
            for fuel_k, share in p["shares"].items():
                if fuel_k in self.k_id and (n, self.k_id.index(fuel_k)) in self.j_of:
                    self.grid_burn.append((g, self.j_of[(n, self.k_id.index(fuel_k))], float(share), float(p["voll"])))
            self.grid_null.append(sum(float(x) for f, x in p["shares"].items() if f not in self.k_id))
            self.grid_voll.append(float(p["voll"]))
        self.NG = len(lay["grids"])
        self.ration_thr = {j: threshold for j, threshold, _m in self.ration}
        self.grid_gk = [[g for g, e in enumerate(self.grid_burn) if e[0] == grid] for grid in range(self.NG)]
        self.grid_fabs = [
            [(f, e_f) for f, (fg, e_f) in enumerate(zip(self.fab_grid, self.fab_e)) if fg == grid and e_f > 0]
            for grid in range(self.NG)
        ]
        self.G_nom = np.array([float(by_id[self.node_id[n]]["grid"]["deliverable"]) for n in lay["grids"]])
        self.cap_nom = np.array([float(by_id[self.node_id[n]]["fab"]["cap0"]) for n in lay["fabs"]])
        self.grid_of_node = {n: g for g, n in enumerate(lay["grids"])}

        # demands, and the rule's destinations
        sinks = st["sinks"]
        pi = {(sinks["node"][i], sinks["k"][i]): sinks["pi"][i] for i in range(len(sinks["node"]))}
        self.demands = [
            (self.j_of.get(tuple(x), -1), pi.get(tuple(x), 0.0) * float(params["shortage_weight"]))
            for x in lay["demands"]
        ]
        self.demand_k = [int(x[1]) for x in lay["demands"]]
        self.lot_keys = [tuple(x) for x in lay.get("lot_keys", [])]
        self.chk_params = {c: by_id[self.node_id[c]].get("chokepoint", {}) for c in lay["chokepoints"]}
        self.n_override = config["spaces"]["action"]["override_qty"]["shape"]
        self.n_pairs = config["spaces"]["action"]["release_mode"]["shape"]

        self.rule_on = float(params["rule_chips"]) >= 0.5
        self.rule_slot = np.array([self.rule_on and self.k_id[k] in RULE_COMMODITIES for k in self.k_s])
        pk_pi = {}  # packaged chip -> its largest penalty pi over the sinks
        for x in lay["demands"]:
            pk_pi[x[1]] = max(pk_pi.get(x[1], 0.0), float(pi.get(tuple(x), 0.0)))
        self.dest = {}  # stock slot j -> ("fab", f) | ("osat", osat ordinal, packaged k) | ("sink", demand row)
        for f, (j_in, _j_out, _tau) in enumerate(self.fabs):
            if j_in >= 0:
                self.dest[j_in] = ("fab", f)
        self.lot_value = np.zeros(len(self.fabs))  # USD per lot started: lot_value x pi of the chip it becomes
        for o, n in enumerate(lay["osats"]):
            for raw, packaged in by_id[self.node_id[n]]["osat"]["packages"].items():
                j = self.j_of.get((n, self.k_id.index(raw)), -1)
                if j >= 0:
                    self.dest[j] = ("osat", o, self.k_id.index(packaged))
                for f, n_f in enumerate(lay["fabs"]):
                    if by_id[self.node_id[n_f]]["fab"]["product"] == raw:
                        self.lot_value[f] = float(params["lot_value"]) * pk_pi.get(self.k_id.index(packaged), 0.0)
        for d, (j, _pi) in enumerate(self.demands):
            if j >= 0:
                self.dest[j] = ("sink", d)
        self.osat_pk = {}  # osat ordinal -> packaged commodities it makes
        for x in self.dest.values():
            if x[0] == "osat":
                self.osat_pk.setdefault(x[1], []).append(x[2])

        # tanker cargo (commodities with an override): under tanker control a lane dispatch only reaches the lane's
        # first strait, where it joins the queue (c, k); the LP then releases it on the override slots of (c, k),
        # each a path to the next strait on its lane or to the lane's end
        self.tanker = [bool(f) for f in st["commodities"]["override"]]
        self.pairs = [tuple(x) for x in lay.get("release_pairs", [])]
        self.pair_of = {p: i for i, p in enumerate(self.pairs)}
        self.tk_on = float(params["tanker_control"]) >= 0.5 and bool(self.pairs)
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
        self.ov_chk_pos = []  # per override slot: (strait, index of the path edge reaching it)
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
            reached = [(int(edges["head"][x]), i) for i, x in enumerate(path) if edges["head"][x] in chk_set]
            self.ov_chk_pos.append(reached)
        self.NO = len(self.ov) if self.tk_on else 0
        self.P = len(self.pairs) if self.tk_on else 0

        # incidence lists for the LP's capacity rows
        self.used_edges = sorted({e for es in self.cap_edges for e in es} | {o_[1] for o_ in self.ov[: self.NO]})
        self.edge_x = {e: [] for e in self.used_edges}
        for s, es in enumerate(self.cap_edges):
            for e in es:
                self.edge_x[e].append(s)
        self.edge_y = {e: [] for e in self.used_edges}
        for i in range(self.NO):
            self.edge_y[self.ov[i][1]].append(i)
        self.jout_x = {}
        for s, j in enumerate(self.j_out):
            if j >= 0:
                self.jout_x.setdefault(j, []).append(s)
        self.chk_pairs = sorted(
            {(c, self.pool[k]) for cs, k in zip(self.cap_chk, self.k_s) for c in cs}
            | ({(c, self.pool[k]) for c, k in self.pairs} if self.tk_on else set())
        )
        self.chk_x = {cp: [] for cp in self.chk_pairs}
        for s, (cs, k) in enumerate(zip(self.cap_chk, self.k_s)):
            for c in cs:
                self.chk_x[(c, self.pool[k])].append(s)
        self.chk_y = {cp: [] for cp in self.chk_pairs}
        for i in range(self.NO):
            p_from = self.ov[i][0]
            if p_from >= 0:
                c, k = self.pairs[p_from]
                if (c, self.pool[k]) in self.chk_y:
                    self.chk_y[(c, self.pool[k])].append(i)

        # for the fallback's rerouting: the other routes from the same stock to the same destination
        by_key = {}
        for s in range(S):
            key = (self.j_out[s], self.dest_stock[s], int(self.k_s[s]))
            if key[0] >= 0 and key[1] >= 0:
                by_key.setdefault(key, []).append(s)
        self.alternatives = [
            [a for a in by_key.get((self.j_out[s], self.dest_stock[s], int(self.k_s[s])), []) if a != s]
            for s in range(S)
        ]

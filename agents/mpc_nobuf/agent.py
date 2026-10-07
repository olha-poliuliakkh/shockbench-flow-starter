"""Hybrid: a base-stock rule ships wafers and chips, the MILP of ``mpc_milp`` plans fuel, grids and tankers.

Chips and wafers are local decisions: each destination (a fab's wafers, an OSAT's raw chips, a sink's chips) is kept
at a target of its weekly use times (expected lead + 1 + cover weeks), capped so that a delivery does not overflow its
storage; scarce stock goes first to the destination with the fewest weeks of cover, over the fastest open route. The
MILP keeps only the fuel side: its chip and wafer routes are closed, a fab's wafer stock receives the rule's shipments
(and its weekly use from then on), and a lot started is credited ``lot_value`` x the penalty pi of its chip.

Also: safety stocks on the fuel stores (as ``mpc_safe``), a grid's deliverable G_bar and a fab's capacity recover
toward nominal over the window (time constants measured on the marks), and stock left after the episode's last week
is worth its salvage only.

Below, the base MPC with the grids' base-load-first rule: every week, solve a small mixed-integer program over the next
H weeks and send its first week.

The simulator gives a grid's fabs energy only after the grid's whole base load is served (``base_first``): a grid
that sheds any load starts no lots. A binary z per grid with fabs and window week encodes it: z = 1 lets the fabs
draw energy and forbids shed; z = 0 forbids fab energy. The plan can then concentrate scarce fuel (LNG) on the grids
whose fabs are worth running. Solved by ``scipy.optimize.milp`` (HiGHS) under a time limit; without a solution in
time the week falls back to the linear program without the rule.

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
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


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
    "fab_threshold": 1.0,  # 1: base load first as binaries (fab energy only while the grid sheds nothing); 0: off
    "milp_weeks": 16,  # window weeks whose z is binary; later weeks keep z continuous in [0, 1]
    "milp_time": 0.8,  # seconds the MILP may take before the week falls back to the LP
    "milp_gap": 0.002,  # relative optimality gap at which the MILP stops
    "grid_model": 1.0,  # 1: the simulator's grid (segments up to share x G_bar, one load factor); 0: share x y_bar each
    "short_price": 1.0,  # times v_k: a fuel segment burning below the grid's load factor (fuel kept back)
    "rule_chips": 1.0,  # 1: the base-stock rule ships wafers and chips, the MILP only fuel; 0: the MILP ships all
    "rule_cover": 3.0,  # weeks of use a destination holds beyond lead + 1 (capped by its storage)
    "lot_value": 0.8,  # a lot started is worth this share of its chip's penalty pi (rule_chips 1)
    "safety_frac": 0.5,  # share of each fuel store's storage kept as a safety stock (0: off)
    "safety_price": 0.05,  # USD per unit-week below the safety stock, as a share of the commodity's value v_k
    "tau_grid": 6.0,  # weeks: a cut G_bar recovers toward the grid's deliverable (0: persists)
    "tau_fab": 12.0,  # weeks: a fab's cut capacity recovers toward cap0 (0: persists)
    "end_salvage": 1.0,  # 1: stock left after the episode's last week is worth its salvage only
    "_fixed": [
        "fab_threshold",
        "fab_shed_tol",
        "fab_w_cost",
        "pipeline_closure",
        "tanker_control",
    ],  # a search leaves these as they are
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
        self.grid_null, self.grid_voll = [], []  # per grid ordinal: the share needing no modelled fuel, VOLL
        for n in lay["grids"]:
            p = by_id[self.node_id[n]]["grid"]
            self.grid_null.append(sum(float(x) for f, x in p["shares"].items() if f not in self.k_id))
            self.grid_voll.append(float(p["voll"]))
        self.ration_thr = {j: threshold for j, threshold, _m in self.ration}
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
        self.milp_log = []  # per week: (milp status, a solution came back) or None
        self.last_plan = None  # the last solved window: columns, offsets and objective (offline analysis only)

        # nominal levels the window recovers toward: each grid's deliverable G_bar, each fab's capacity cap0
        self.G_nom = np.array([float(by_id[self.node_id[n]]["grid"]["deliverable"]) for n in lay["grids"]])
        self.cap_nom = np.array([float(by_id[self.node_id[n]]["fab"]["cap0"]) for n in lay["fabs"]])

        # the rule's commodities and slots, and what each destination stock slot uses per week
        rule_names = ("wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat")
        self.rule_on = PARAMS["rule_chips"] >= 0.5
        self.rule_slot = np.array([self.rule_on and self.k_id[k] in rule_names for k in self.k_s])
        pk_pi = {}  # packaged chip -> its largest penalty pi over the sinks
        for d, x in enumerate(lay["demands"]):
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
                        self.lot_value[f] = PARAMS["lot_value"] * pk_pi.get(self.k_id.index(packaged), 0.0)
        for d, (j, _pi) in enumerate(self.demands):
            if j >= 0:
                self.dest[j] = ("sink", d)
        self.osat_pk = {}  # osat ordinal -> packaged commodities it makes
        for j, x in self.dest.items():
            if x[0] == "osat":
                self.osat_pk.setdefault(x[1], []).append(x[2])
        self.demand_k = [int(x[1]) for x in lay["demands"]]
        self.rule_x = None  # this week's rule flows (S,), kept when the MILP fails

        # safety stocks on the fuel stores (terminals, grids): (j, target, USD per unit-week short)
        self.safe = []
        if PARAMS["safety_frac"] > 0:
            ntype = nodes["type"]
            for j, (n, k) in enumerate(self.stock):
                kid = self.k_id[k]
                if ntype[n] not in ("terminal", "grid") or kid == "nucfuel" or not np.isfinite(self.storage[j]):
                    continue
                target = PARAMS["safety_frac"] * self.storage[j]
                if target > 0:
                    self.safe.append((j, target, PARAMS["safety_price"] * self.v[k]))

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
        self.rule_x = None
        try:
            flows = self._solve(o, t, H)
        except Exception:  # noqa: BLE001 - a failed solve must not hand the week to the naive rule
            flows = None
        release = None
        if isinstance(flows, tuple):
            flows, release = flows
        if flows is None:
            flows = self.last_flows if t > 1 else self.u0[[es[0] for es in self.route_edges]]
            if self.rule_x is not None:  # the rule's shipments stand without the MILP
                flows = np.where(self.rule_slot, self.rule_x, flows)
        flows = np.nan_to_num(np.maximum(flows, 0.0)) * mask
        self.last_flows = flows
        action = {"flows": flows}
        if release is not None:
            qty, mode = release
            om = o["override_mask"].astype(float)
            action["override_qty"] = np.nan_to_num(np.maximum(qty, 0.0)) * om
            action["release_mode"] = mode
        return action

    def _window(self, o, t, H):
        """What the plan expects in each window week (arrays with a leading axis of H): persistence of ``graph_now``,
        with announced sanctions from their week, announced reopenings and the demand forecast. ``closed`` holds the
        straits closed now and ``reopen`` the first window week each is open again (from the announcements)."""
        rep = lambda x: np.repeat(np.asarray(x)[None], H, axis=0)  # noqa: E731
        proh = rep(o["graph_now.prohibited"]).astype(bool)
        for e, k, w in zip(
            o["pending_prohibitions.edge"], o["pending_prohibitions.k"], o["pending_prohibitions.effective_week"]
        ):
            if w >= t and int(w) - t < H:
                proh[int(w) - t :, int(e), int(k)] = True
        open_ = o["graph_now.open"]
        reopen = {}  # chokepoint -> first window week it is open again
        if "closure_end.chokepoint" in o:
            ends = zip(o["closure_end.chokepoint"], o["closure_end.end_week"], o["closure_end.end_week.observed"])
            for c, w, ok in ends:
                if ok and int(c) in self.chk_row and int(w) > t:
                    reopen[int(c)] = min(reopen.get(int(c), H), int(w) - t)
        closed = {c for c, row in self.chk_row.items() if float(open_[row]) <= PARAMS["closed_below"]}
        chk_cap = {}  # pool -> (H, chokepoint row) throughput
        for pool in ("tb", "ct"):
            cap = np.zeros((H, len(self.chk_row)))
            for c, row in self.chk_row.items():
                cap[:, row] = (
                    0.0 if open_[row] <= PARAMS["closed_below"] else max(0.0, float(o[f"graph_now.kappa.{pool}"][row]))
                )
                if c in reopen:
                    p = self.chk_params[c]
                    nominal = float(p.get("mu", {}).get(pool, 0.0)) * float(p.get("k_c", 1.0))
                    cap[reopen[c] :, row] = np.maximum(cap[reopen[c] :, row], PARAMS["reopen_trust"] * nominal)
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
            "fab_cap": recover(o["graph_now.fab.cap_eff"], self.cap_nom, PARAMS["tau_fab"]),
            "osat_thr": rep(o["graph_now.osat.thr_eff"]),
            "y_bar": rep(o["graph_now.grid.y_bar"]),
            "G_bar": recover(o["graph_now.grid.G_bar"], self.G_nom, PARAMS["tau_grid"]),
            "demand": demand,
            "closed": closed,
            "reopen": reopen,
        }

    def _rule_flows(self, o, t, H, win, lead, blocked_from, closed, arr):
        """Base-stock shipments of wafers and chips. Each destination stock slot j uses d_j a week (a fab its expected
        capacity, an OSAT its throughput share of the demand for what it packages, a sink its forecast demand) and is
        kept at min(d_j (L_j + 1 + cover), storage_j + d_j L_j), L_j its fastest open route's lead; what is on hand or
        on its way counts. Stock at the origins goes, half a week of use at a time, to the destination with the fewest
        weeks of cover, over its fastest open route with capacity left. Returns the flows and, per fab wafer slot, its
        lead and weekly use (the wafers the MILP may count on)."""
        S, cover = self.S, PARAMS["rule_cover"]
        x = np.zeros(S)
        I0 = o["stock.qty"].astype(float)
        u_left = np.nan_to_num(np.asarray(win["u"][0], dtype=float), nan=0.0, posinf=1e18).copy()
        ct_left = {c: float(win["chk_cap"]["ct"][0, row]) for c, row in self.chk_row.items()}
        usable = {}  # destination j -> its open rule slots
        for s in np.flatnonzero(self.rule_slot):
            j_out, j_in = self.j_out[s], self.j_in[s]
            if j_out < 0 or j_in < 0 or j_in not in self.dest or blocked_from[s] <= 0 or t + lead[s] > self.T:
                continue
            if any(c in closed for c in self.route_chk[s]):
                continue
            usable.setdefault(j_in, []).append(s)

        # weekly use per destination over the weeks its next delivery covers
        dem_k = {}  # packaged chip -> its forecast demand per week, all sinks
        for d, k in enumerate(self.demand_k):
            dem_k[k] = dem_k.get(k, 0.0) + float(np.mean(win["demand"][:, d]))
        thr = np.asarray(win["osat_thr"][0], dtype=float)
        share = {}  # (osat, packaged k) -> share of its throughput
        for osat, pks in self.osat_pk.items():
            tot = sum(dem_k.get(k, 0.0) for k in pks)
            for k in pks:
                share[(osat, k)] = dem_k.get(k, 0.0) / tot if tot > 0 else 1.0 / len(pks)
        cap_k = {}  # packaged chip -> OSAT throughput the shares give it, all OSATs
        for (osat, k), sh in share.items():
            cap_k[k] = cap_k.get(k, 0.0) + thr[osat] * sh
        IP = I0 + arr.sum(axis=0)
        d_of, target = {}, {}
        fab_feed = {}
        for j, kind in self.dest.items():
            L = int(min(lead[s] for s in usable[j])) if j in usable else 0
            a, b = min(L, H - 1), min(H, L + int(cover) + 1)
            if kind[0] == "fab":
                d = float(np.mean(win["fab_cap"][a:b, kind[1]]))
                if j in usable:
                    fab_feed[j] = (L, np.asarray(win["fab_cap"][:, kind[1]], dtype=float))
            elif kind[0] == "osat":
                osat, k = kind[1], kind[2]
                scale = min(1.0, dem_k.get(k, 0.0) / cap_k[k]) if cap_k.get(k, 0.0) > 0 else 0.0
                d = thr[osat] * share[(osat, k)] * scale
            else:
                d = float(np.mean(win["demand"][a:b, kind[1]]))
            d_of[j] = max(d, 0.0)
            target[j] = min(d * (L + 1 + cover), self.storage[j] + d * L)
        if not usable:
            return x, fab_feed

        avail = I0.copy()
        sent = {j: 0.0 for j in usable}
        order = sorted(usable, key=lambda j: -max((pi for jj, pi in self.demands if jj == j), default=0.0))

        def slot_cap(s):
            cap = min(u_left[e] for e in self.cap_edges[s])
            for c in self.cap_chk[s]:
                cap = min(cap, ct_left[c])
            return max(cap, 0.0)

        for _step in range(4000):
            best, best_cov = None, np.inf
            for j in order:
                d = d_of[j]
                if d <= 0 or IP[j] + sent[j] >= target[j] - 1e-6:
                    continue
                cov = (IP[j] + sent[j]) / d
                if cov < best_cov and any(avail[self.j_out[s]] > 1e-6 and slot_cap(s) > 1e-6 for s in usable[j]):
                    best, best_cov = j, cov
            if best is None:
                break
            j = best
            s = min(
                (s for s in usable[j] if avail[self.j_out[s]] > 1e-6 and slot_cap(s) > 1e-6),
                key=lambda s: (lead[s], len(self.route_chk[s])),
            )
            q = min(target[j] - IP[j] - sent[j], avail[self.j_out[s]], slot_cap(s), max(0.5 * d_of[j], 1.0))
            x[s] += q
            sent[j] += q
            avail[self.j_out[s]] -= q
            for e in self.cap_edges[s]:
                u_left[e] -= q
            for c in self.cap_chk[s]:
                ct_left[c] -= q
        return x, fab_feed

    def _route_costs(self, c_e, tariff, war, queue_cost, P, NO):
        """Per unit cost of every route, override slot and queue pair in one week: freight, tariffs, war risk."""
        v = self.v
        cost = np.zeros(self.S)
        for s, (es, cs, k) in enumerate(zip(self.cap_edges, self.cap_chk, self.k_s)):
            if self.tk_dest[s] >= 0:
                cs = [self.edge_head[es[-1]]]
            cost[s] = sum(c_e[e] + tariff[e, k] * v[k] for e in es)
            for c in cs:
                cls = int(war[self.chk_row[c]])
                wr = self.chk_params[c].get("war_risk_cost", {}).get(self.k_id[k])
                if wr:
                    cost[s] += wr[min(cls, len(wr) - 1)]
                cost[s] += queue_cost.get((c, k), 0.0)
        ov_cost = np.zeros(NO)
        for i, (p_from, e_out, path, _sp, _sj, passed) in enumerate(self.ov[:NO]):
            k = self.pairs[p_from][1] if p_from >= 0 else 0
            ov_cost[i] = sum(c_e[x] + tariff[x, k] * v[k] for x in path)
            for c in passed:
                cls = int(war[self.chk_row[c]])
                wr = self.chk_params[c].get("war_risk_cost", {}).get(self.k_id[k])
                if wr:
                    ov_cost[i] += wr[min(cls, len(wr) - 1)]
        q_hold = np.zeros(P)  # queue holding per unit-week at each pair
        for i, (c, k) in enumerate(self.pairs[:P]):
            qh = self.chk_params[c].get("queue_holding", {}).get(self.k_id[k])
            if qh:
                q_hold[i] = qh[min(int(war[self.chk_row[c]]), len(qh) - 1)]
        return cost, ov_cost, q_hold

    def _solve(self, o, t, H):
        S, J = self.S, len(self.stock)
        tau = o["graph_now.tau"]
        win = self._window(o, t, H)
        v = self.v
        open_, war0 = o["graph_now.open"], o["graph_now.war_risk"]
        end = PARAMS["end_salvage"] >= 0.5 and t + H - 1 >= self.T  # the window holds the episode's last week
        tf = 0.0 if end else PARAMS["terminal_frac"]  # nothing travelling or stored is worth more after the end
        value = self.salvage if end else self.value
        closed, reopen = win["closed"], win["reopen"]

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
                        cls = int(war0[row])
                        queue_cost[(c, k)] = qh[min(cls, len(qh) - 1)] * extra
        use_pc = PARAMS["pipeline_closure"] >= 0.5

        # routes: lead (this week's), blocked weeks; costs per window week (recomputed only where they change)
        lead = np.array([int(sum(tau[e] for e in es)) for es in self.route_edges])
        blocked_from = np.full(S, H)  # first window week a route is prohibited (H: never)
        proh = win["proh"]
        for s, (es, cs, k) in enumerate(zip(self.cap_edges, self.cap_chk, self.k_s)):
            if self.tk_dest[s] >= 0:  # a tanker lane: the dispatch only runs to the first strait
                lead[s] = int(sum(tau[e] for e in es))
                cs = [self.edge_head[es[-1]]]
            for c in cs:
                lead[s] += delay.get(c, 0)
            hit = np.flatnonzero(proh[:, es, k].any(axis=1))
            blocked_from[s] = hit[0] if hit.size else H
        if o["action_mask.observed"][0]:
            blocked_from = np.where(o["action_mask"] == 0, np.minimum(blocked_from, 0), blocked_from)

        # override slots: lead and the first blocked week of each release path
        P, NO = (len(self.pairs), len(self.ov)) if self.tk_on else (0, 0)
        ov_lead, ov_blocked = np.zeros(NO, dtype=int), np.full(NO, H)
        for i, (p_from, e_out, path, _sp, _sj, passed) in enumerate(self.ov[:NO]):
            k = self.pairs[p_from][1] if p_from >= 0 else 0
            ov_lead[i] = int(sum(tau[x] for x in path))
            hit = np.flatnonzero(proh[:, path, k].any(axis=1))
            ov_blocked[i] = 0 if p_from < 0 else (hit[0] if hit.size else H)
            if o["override_mask.observed"][0] and o["override_mask"][i] == 0:
                ov_blocked[i] = min(ov_blocked[i], 0)
        costs = []  # per window week: (route cost, override cost, queue holding)
        for h in range(H):
            same = h > 0 and all(np.array_equal(win[f][h], win[f][h - 1]) for f in ("c", "tariff", "war"))
            costs.append(
                costs[-1]
                if same
                else self._route_costs(win["c"][h], win["tariff"][h], win["war"][h], queue_cost, P, NO)
            )

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

        # the rule ships wafers and chips; the MILP sees the wafers reach the fabs (this week's shipments at their
        # lead, then each fab's weekly use), and plans the rest
        if self.rule_on:
            self.rule_x, fab_feed = self._rule_flows(o, t, H, win, lead, blocked_from, closed, arr)
            for s in np.flatnonzero(self.rule_x > 0):
                add(self.j_in[s], int(lead[s]), float(self.rule_x[s]))
            for j, (L, use) in fab_feed.items():
                for h in range(1, H):
                    add(j, h + L, use[min(h + L, H - 1)])

        # columns
        F, M, D, G = len(self.fabs), len(self.osats), len(self.demands), len(self.grid_burn)
        U_ = len(self.supply_j)
        R = len(self.ration)
        NG = len(o["graph_now.grid.y_bar"])
        new_grid = PARAMS["grid_model"] >= 0.5
        sizes = {"x": S, "I": J, "O": J, "lift": U_, "p": F, "q": M, "U": D, "shed": 0 if new_grid else G, "buf": R}
        sizes |= {"w": NG, "z": NG}
        sizes |= {"gk": G, "short": G, "lam": NG, "sh": NG} if new_grid else {"gk": 0, "short": 0, "lam": 0, "sh": 0}
        sizes |= {"Q": P, "y": NO, "safe": len(self.safe)}
        off, n = {}, 0
        for name, size in sizes.items():
            off[name] = n
            n += H * size

        def col(name, h, i):
            return off[name] + h * sizes[name] + i

        def grid_sheds(h, grid):  # a grid's shed columns: one per grid (new model), one per fuel (old)
            if new_grid:
                return [(col("sh", h, grid), 1.0)]
            return [(col("shed", h, g), 1.0) for g, e in enumerate(self.grid_burn) if e[0] == grid]

        lb, ub = np.zeros(n), np.full(n, np.inf)
        integer = []  # the binary columns (z of the first milp_weeks weeks)
        cost_vec = np.zeros(n)
        I0 = o["stock.qty"].astype(float)
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
                cost_vec[cq] = costs[h][2][p]
                if h == H - 1:
                    cost_vec[cq] -= tf * v[self.pairs[p][1]]
            for i, (p_from, e_out, path, stop_p, stop_j, _passed) in enumerate(self.ov[:NO]):
                cy = col("y", h, i)
                if h >= ov_blocked[i] or p_from < 0:
                    ub[cy] = 0.0
                cost_vec[cy] = costs[h][1][i]
                qbal[h][p_from].append((cy, 1.0))
                ha = h + ov_lead[i]
                if stop_p >= 0:
                    if ha < H:
                        qbal[ha][stop_p].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= tf * v[self.pairs[stop_p][1]]
                elif stop_j >= 0:
                    if ha < H:
                        balance[ha][stop_j].append((cy, -1.0))
                    else:
                        cost_vec[cy] -= value[stop_j]
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
                if h >= blocked_from[s] or self.j_out[s] < 0 or self.rule_slot[s]:  # the rule's slots: fixed outside
                    ub[cx] = 0.0
                cost_vec[cx] = costs[h][0][s]
                balance[h][self.j_out[s]].append((cx, 1.0))
                ha = h + lead[s]
                if self.tk_dest[s] >= 0:
                    if ha < H:
                        qbal[ha][self.tk_dest[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= tf * v[self.k_s[s]]
                elif self.j_in[s] >= 0:
                    if ha < H:
                        balance[ha][self.j_in[s]].append((cx, -1.0))
                    else:
                        cost_vec[cx] -= value[self.j_in[s]]
            for i, j in enumerate(self.supply_j):
                if j >= 0:
                    ub[col("lift", h, i)] = max(0.0, float(win["supply"][h, i]))
                    balance[h][j].append((col("lift", h, i), -1.0))
            for f, (j_in, j_out, tau_f) in enumerate(self.fabs):
                cp = col("p", h, f)
                ub[cp] = max(0.0, float(win["fab_cap"][h, f])) if j_in >= 0 else 0.0
                if j_in >= 0:
                    balance[h][j_in].append((cp, 1.0))
                if self.rule_on:  # the chips' way to the sinks is the rule's: a lot is worth its share of pi
                    cost_vec[cp] -= self.lot_value[f]
                elif j_out >= 0:
                    if h + tau_f < H:
                        balance[h + tau_f][j_out].append((cp, -1.0))
                    else:
                        cost_vec[cp] -= value[j_out]
            for m, (j_raw, j_pk, tau_o, osat) in enumerate(self.osats):
                cq = col("q", h, m)
                if j_raw < 0 or self.rule_on:
                    ub[cq] = 0.0
                    continue
                balance[h][j_raw].append((cq, 1.0))
                if j_pk >= 0:
                    if h + tau_o < H:
                        balance[h + tau_o][j_pk].append((cq, -1.0))
                    else:
                        cost_vec[cq] -= value[j_pk]
            for osat in range(0 if self.rule_on else len(o["graph_now.osat.thr_eff"])):
                le(
                    [(col("q", h, m), 1.0) for m, e in enumerate(self.osats) if e[3] == osat],
                    max(0.0, float(win["osat_thr"][h, osat])),
                )
            for d, (j, pi) in enumerate([] if self.rule_on else self.demands):
                dem = max(0.0, float(win["demand"][h, d]))
                cu = col("U", h, d)
                ub[cu] = dem
                cost_vec[cu] = pi
                if j >= 0:
                    rhs[h, j] -= dem
                    balance[h][j].append((cu, -1.0))
            # the simulator's grid (segments): fuel k yields up to share_k G_bar (rationed below psi ibar), the
            # unmodelled share needs no fuel, and every segment runs at the grid's one load factor lam:
            #   sum_k G_k + lam share_0 G_bar + shed - sum_f e_f p_f = y_bar,  G_k <= lam share_k G_bar,
            #   G_k >= lam share_k G_bar - short_k (short priced at v_k: no fuel kept back while it is on hand)
            for g, (grid, j, share, voll) in enumerate(self.grid_burn if new_grid else []):
                cap = share * max(0.0, float(win["G_bar"][h, grid]))
                cg, cs_, cl = col("gk", h, g), col("short", h, g), col("lam", h, grid)
                balance[h][j].append((cg, 1.0))
                le([(cg, 1.0), (cl, -cap)], 0.0)
                le([(cl, cap), (cg, -1.0), (cs_, -1.0)], 0.0)
                cost_vec[cs_] = v[self.stock[j][1]] * PARAMS["short_price"]
                if j in self.ration_thr and cap > 0:  # (15): G_k <= share_k G_bar I^{t-1} / (psi ibar)
                    r = cap / self.ration_thr[j]
                    if h > 0:
                        le([(cg, 1.0), (col("I", h - 1, j), -r)], 0.0)
                    else:
                        le([(cg, 1.0)], r * I0[j])
            for grid in range(NG if new_grid else 0):
                cl, csd = col("lam", h, grid), col("sh", h, grid)
                y_bar = max(0.0, float(win["y_bar"][h, grid]))
                ub[cl], ub[csd] = 1.0, y_bar
                cost_vec[csd] = self.grid_voll[grid] * PARAMS["shed_weight"]
                ent = [(col("gk", h, g), 1.0) for g, e in enumerate(self.grid_burn) if e[0] == grid]
                ent += [(cl, self.grid_null[grid] * max(0.0, float(win["G_bar"][h, grid]))), (csd, 1.0)]
                ent += [
                    (col("p", h, f), -e_f)
                    for f, (fg, e_f) in enumerate(zip(self.fab_grid, self.fab_e))
                    if fg == grid and e_f > 0
                ]
                eq(ent, y_bar)
            for g, (grid, j, share, voll) in enumerate([] if new_grid else self.grid_burn):
                if PARAMS["fab_energy_weight"] > 0:  # the fabs' draw on this fuel, per lot started
                    for f, (fg, e_f) in enumerate(zip(self.fab_grid, self.fab_e)):
                        if fg == grid and e_f > 0:
                            balance[h][j].append((col("p", h, f), share * e_f * PARAMS["fab_energy_weight"]))
                burn = share * max(0.0, float(win["y_bar"][h, grid]))
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
                    y_bar = max(0.0, float(win["y_bar"][h, grid]))
                    headroom = max(0.0, float(win["G_bar"][h, grid]) - y_bar)
                    tol = max(PARAMS["fab_shed_tol"] * y_bar, 1e-6)
                    cw = col("w", h, grid)
                    cost_vec[cw] = PARAMS["fab_w_cost"]
                    ent = [(col("p", h, f), self.fab_e[f]) for f in fabs] + [(cw, -1.0)]
                    ent += [(c_, headroom / tol) for c_, _one in grid_sheds(h, grid)]
                    le(ent, headroom)
            # base load first: shed_g <= y_bar (1 - z), fab energy_g <= E_max z (z binary in the first weeks)
            for grid in range(NG):
                cz = col("z", h, grid)
                fabs = [f for f, fg in enumerate(self.fab_grid) if fg == grid and self.fab_e[f] > 0]
                if PARAMS["fab_threshold"] < 0.5 or not fabs:
                    ub[cz] = 0.0
                    continue
                ub[cz] = 1.0
                y_bar = max(0.0, float(win["y_bar"][h, grid]))
                le(grid_sheds(h, grid) + [(cz, y_bar)], y_bar)
                e_max = sum(self.fab_e[f] * max(0.0, float(win["fab_cap"][h, f])) for f in fabs)
                le([(col("p", h, f), self.fab_e[f]) for f in fabs] + [(cz, -e_max)], 0.0)
                if h < PARAMS["milp_weeks"]:
                    integer.append(cz)
            for i, (j, threshold, marginal) in enumerate(self.ration):  # I[h, j] + buf >= threshold
                cb = col("buf", h, i)
                cost_vec[cb] = marginal * PARAMS["buffer_weight"]
                le([(col("I", h, j), -1.0), (cb, -1.0)], -threshold)
            for i, (j, target, price) in enumerate(self.safe):  # I[h, j] + safe >= target
                cs_ = col("safe", h, i)
                cost_vec[cs_] = price
                le([(col("I", h, j), -1.0), (cs_, -1.0)], -target)
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
                cap_e = max(0.0, float(win["u"][h, e]))
                if PARAMS["recover_weeks"] > 0 and cap_e < self.u0[e]:  # a cut recovers toward nominal
                    cap_e += (self.u0[e] - cap_e) * (1.0 - np.exp(-h / PARAMS["recover_weeks"]))
                le(ent, cap_e)
            for c, pool in self.chk_pairs:
                row = self.chk_row[c]
                cap = float(win["chk_cap"][pool][h, row])
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
            cost_vec[col("I", H - 1, j)] -= value[j]

        A_eq = sp.csr_matrix((vals_eq, (rows_eq, cols_eq)), shape=(r_eq[0], n))
        A_ub = sp.csr_matrix((vals_ub, (rows_ub, cols_ub)), shape=(r_ub[0], n))
        ub = np.maximum(ub, lb)
        res = None
        if integer:
            kind = np.zeros(n)
            kind[integer] = 1
            try:
                res = milp(
                    cost_vec,
                    constraints=[
                        LinearConstraint(A_ub, -np.inf, np.array(b_ub)),
                        LinearConstraint(A_eq, np.array(b_eq), np.array(b_eq)),
                    ],
                    integrality=kind,
                    bounds=Bounds(lb, ub),
                    options={"time_limit": float(PARAMS["milp_time"]), "mip_rel_gap": float(PARAMS["milp_gap"])},
                )
            except Exception:  # noqa: BLE001 - the LP below plays the week
                res = None
            self.milp_log.append(None if res is None else (int(res.status), res.x is not None))
            if res is not None and res.x is None:
                res = None
        if res is None:
            ub[off["z"] : off["z"] + H * NG] = np.where(ub[off["z"] : off["z"] + H * NG] > 0, 1.0, 0.0)
            res = linprog(
                cost_vec,
                A_ub=A_ub,
                b_ub=np.array(b_ub),
                A_eq=A_eq,
                b_eq=np.array(b_eq),
                bounds=np.c_[lb, ub],
                method="highs",
            )
        if res.x is None or (getattr(res, "status", 0) != 0 and not integer):
            return None
        self.last_plan = {"x": res.x, "off": off, "sizes": sizes, "H": H, "cost": cost_vec}  # for offline analysis
        flows = res.x[off["x"] : off["x"] + S].copy()
        if self.rule_x is not None:
            flows = np.where(self.rule_slot, self.rule_x, flows)
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

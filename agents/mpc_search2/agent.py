"""mpc_tuned plus one simulated correction a week: the MPC proposes, a vendored copy of the simulator judges.

Every week (when the CPU budget allows) the agent takes the MPC's action, scales one decision group's flows by
``SEARCH["level"]`` (the fuel bound for a grid, a strait's LNG releases, the wafers for a fab, the raw chips for an
OSAT, cycled in the order the offline search found them to pay most often) and plays both branches to the episode's end
on a rebuilt copy of the simulator state under a persistence forecast: the first ``head_weeks`` weeks by a
deterministic copy of this same MPC, the rest by a cheap configuration of it (an LP over an 8-week window, ~0.04 s a
week). The correction is kept when its branch saves more than ``min_gain`` of the base branch's cost. Offline this
search gained +0.023 RSS over the MPC on Small dev with a 4-week head (6 candidates a week); the budget allows one.
numpy/scipy only: ``vendor/`` holds the simulator package (its schema validator stubbed).
"""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

from shockbench_flow.dynamics.env import validate_action  # noqa: E402
from shockbench_flow.dynamics.observe import observe  # noqa: E402
from shockbench_flow.dynamics.sim import step  # noqa: E402
from shockbench_flow.dynamics.state import Lot, Shipment, State  # noqa: E402
from shockbench_flow.information.flat import FlatLayout  # noqa: E402
from shockbench_flow.instance.io import load_instance  # noqa: E402
from shockbench_flow.marks import WeeklyMarks  # noqa: E402
from shockbench_flow_agent.convert import action_to_wire, observation_dict  # noqa: E402


SEARCH = {
    "head_weeks": 4,  # weeks the true MPC copy plays after the correction before the cheap LP takes over
    "head": {"milp_nodes": 100, "milp_time": 0.8},  # the head: this MPC, deterministic
    "tail": {"fab_threshold": 0.0, "horizon": 8, "milp_time": 0.1},  # the tail: the same model as an 8-week LP
    "level": 1.5,  # the candidate multiplies the group's flows by this (capped at capacity)
    "min_gain": 0.001,  # keep a correction only if it saves this share of the base branch's cost
    "cpu_budget": 1.7,  # seconds of CPU this week (Small); the search is skipped when it would not fit
    "cpu_budget_full": 3.4,
}
if (HERE / "search.json").is_file():
    SEARCH.update(json.loads((HERE / "search.json").read_text()))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mpc = _load(HERE / "mpc.py", "mpc_search2_mpc")
PARAMS = mpc.PARAMS  # the MPC's numbers (examples/10_foresight.py overrides them through --params)


def mpc_variant(name, over):
    mod = _load(HERE / "mpc.py", f"mpc_search2_{name}")
    mod.PARAMS.update(over)
    return mod


head_mod = mpc_variant("head", SEARCH["head"])
tail_mod = mpc_variant("tail", SEARCH["tail"])


def rebuild_state(o, inst, lay, t):
    """The simulator's State at the start of week t from the observation (exact: stock, WIP, queues, pipeline)."""
    stock = np.zeros(len(inst.stock_slots))
    for j, (n, k) in enumerate([tuple(x) for x in lay.stock_slots]):
        stock[inst.slot_index[(n, k)]] = o["stock.qty"][j]
    pipe = []
    live = o["pipeline.qty.observed"] == 1
    for e, k, lane, lok, q, w in zip(
        o["pipeline.edge"][live],
        o["pipeline.k"][live],
        o["pipeline.lane"][live],
        o["pipeline.lane.observed"][live],
        o["pipeline.qty"][live],
        o["pipeline.arrival_week"][live],
    ):
        pipe.append(Shipment(int(e), int(k), int(lane) if lok else None, float(q), t - 1, int(w)))
    lots, nid = [], 0
    chk = set(inst.chokepoints)
    for s in pipe:
        if inst.edges[s.edge].head in chk:
            s.lot_id = nid
            nid += 1
    for i, (c, k, lane, nxt) in enumerate([tuple(x) for x in lay.lot_keys]):
        row = o["queue_lots.qty"][i]
        for w in np.flatnonzero(row > 0):
            lots.append(Lot(nid, int(c), int(k), float(row[w]), int(lane), int(nxt), int(w), int(nxt), int(w) + 1))
            nid += 1
            stock[inst.slot_index[(int(c), int(k))]] += float(row[w])
    fab_wip, osat_wip = {}, {}
    fab_of = {inst.fabs[i]: i for i in range(len(inst.fabs))}
    osat_of = {inst.osats[i]: i for i in range(len(inst.osats))}
    live = o["wip.qty.observed"] == 1
    for n, k, q, w in zip(o["wip.node"][live], o["wip.k"][live], o["wip.qty"][live], o["wip.out_week"][live]):
        n, k, w, q = int(n), int(k), int(w), float(q)
        if n in fab_of:
            d = fab_wip.setdefault(fab_of[n], {})
            s0 = w - inst.nodes[n].fab.tau
            d[s0] = d.get(s0, 0.0) + q
        elif n in osat_of:
            d = osat_wip.setdefault(osat_of[n], {}).setdefault(w, {})
            d[k] = d.get(k, 0.0) + q
    return State(t - 1, stock, pipe, lots, fab_wip, osat_wip, np.asarray(o["backlog.qty"], float).copy(), nid, None)


def persistence_marks(o, inst, lay, static, w_scr):
    """WeeklyMarks holding the instant of week t - 1 for every week: what the simulator needs, nothing more."""
    T = inst.T
    rep = lambda x: np.repeat(np.asarray(x, dtype=float)[None], T, axis=0)  # noqa: E731
    E, K, C = len(inst.edges), len(inst.commodities), len(inst.chokepoints)
    u = np.where(o["graph_now.u.observed"] == 1, o["graph_now.u"], np.inf)
    kappa = np.stack([o["graph_now.kappa.tb"], o["graph_now.kappa.ct"]], axis=1)
    proh = rep(o["graph_now.prohibited"]).astype(bool)
    live = o["pending_prohibitions.edge.observed"] == 1
    for e, k, w in zip(
        o["pending_prohibitions.edge"][live],
        o["pending_prohibitions.k"][live],
        o["pending_prohibitions.effective_week"][live],
    ):
        if 1 <= int(w) <= T:
            proh[int(w) - 1 :, int(e), int(k)] = True
    war = np.asarray(o["graph_now.war_risk"], int)
    h_queue, c_wr = np.zeros((T, C, K)), np.zeros((T, E, K))
    by_id = {n["id"]: n for n in inst.raw["nodes"]}
    for ci, c in enumerate(inst.chokepoints):
        attrs = by_id[inst.nodes[c].id].get("chokepoint") or {}
        qh, wr, cls = attrs.get("queue_holding") or {}, attrs.get("war_risk_cost") or {}, int(war[ci])
        for k, kid in enumerate(static["commodities"]["id"]):
            if kid in qh:
                h_queue[:, ci, k] = qh[kid][min(cls, len(qh[kid]) - 1)]
            if kid in wr:
                for e in inst.out_edges[c]:
                    c_wr[:, e, k] = wr[kid][min(cls, len(wr[kid]) - 1)]
    supply = np.zeros(len(inst.stock_slots))
    for i, (n, k) in enumerate([tuple(x) for x in lay.supply_slots]):
        supply[inst.slot_index[(n, k)]] = o["graph_now.supply.avail"][i]
    thr = np.array([inst.nodes[oo].osat.thr for oo in inst.osats], dtype=float)
    R_osat = np.where(thr > 0, np.asarray(o["graph_now.osat.thr_eff"], float) / np.maximum(thr, 1e-12), 1.0)
    forecast = o["demand_forecast.qty"]
    demand = np.stack([forecast[:, min(h, forecast.shape[1] - 1)] for h in range(T)]).astype(float)
    return WeeklyMarks(
        T=T,
        instance_hash=inst.hash,
        omega_hash="persistence",
        u=rep(u),
        c=rep(o["graph_now.c"]),
        o=rep(o["graph_now.open"]),
        kappa=rep(kappa),
        supply=rep(supply),
        G_bar=rep(o["graph_now.grid.G_bar"]),
        y_bar=rep(o["graph_now.grid.y_bar"]),
        R=rep(o["graph_now.fab.R"]),
        alpha_bar=rep(o["graph_now.fab.alpha_bar"]),
        sigma_scr=np.zeros((T, len(inst.fabs))),
        R_osat=rep(R_osat),
        demand=demand,
        prohibited=proh,
        tariff=rep(o["graph_now.tariff"]),
        wr_class=np.repeat(war[None], T, axis=0).astype(np.int8),
        h_queue=h_queue,
        c_wr=c_wr,
        u_now=rep(u),
        o_now=rep(o["graph_now.open"]),
        kappa_now=rep(kappa),
        supply_now=rep(supply),
        G_bar_now=rep(o["graph_now.grid.G_bar"]),
        y_bar_now=rep(o["graph_now.grid.y_bar"]),
        R_now=rep(o["graph_now.fab.R"]),
        alpha_now=rep(o["graph_now.fab.alpha_bar"]),
        fab_hits=(),
        w_scr=w_scr,
        instance_digest=inst.content_digest,
        generator_id="persistence",
        generated=False,
    )


def decision_groups(static, layout):
    """(kind, index, slot indices): fuel->grid, tanker (strait, k) releases, wafer->fab, raw->OSAT."""
    slots, lanes, edges, nodes = static["action_slots"], static["lanes"], static["edges"], static["nodes"]
    com = static["commodities"]["id"]
    dest = []
    for e, lane in zip(slots["edge"], slots["lane"]):
        last = lanes["edges"][lane][-1] if lane is not None else e
        dest.append(edges["head"][last])
    feeds = {}
    for t_, h in zip(edges["tail"], edges["head"]):
        if nodes["type"][t_] == "terminal" and nodes["type"][h] == "grid":
            feeds.setdefault(t_, set()).add(h)
    kind = [com[k] for k in slots["k"]]
    groups = []
    for i, g in enumerate(layout["grids"]):
        s = [
            j
            for j, (d, k) in enumerate(zip(dest, kind))
            if k in ("lng", "crude", "nucfuel") and (d == g or g in feeds.get(d, ()))
        ]
        groups.append(("fuel", i, np.array(s, dtype=int)))
    ov = static["override_slots"]
    for i, (c, k) in enumerate(tuple(x) for x in layout.get("release_pairs", [])):
        s = [j for j, (cc, kk) in enumerate(zip(ov["chokepoint"], ov["k"])) if (cc, kk) == (c, k)]
        groups.append(("tanker", i, np.array(s, dtype=int)))
    for i, f in enumerate(layout["fabs"]):
        s = [j for j, (d, k) in enumerate(zip(dest, kind)) if d == f and k == "wafer"]
        groups.append(("wafer", i, np.array(s, dtype=int)))
    for i, oo in enumerate(layout["osats"]):
        s = [j for j, (d, k) in enumerate(zip(dest, kind)) if d == oo and k.endswith("_raw")]
        groups.append(("raw", i, np.array(s, dtype=int)))
    return [g for g in groups if len(g[2])]


class Agent:
    def __init__(self, config=None):
        self.mpc = mpc.Agent(config)
        static, layout = config["static"], config["layout"]
        self.T = int(config["T"])
        self.inst = load_instance(copy.deepcopy(static["instance"]), strict=False)
        self.lay = FlatLayout.from_static(copy.deepcopy(static), None, inst=None)
        self.static = static
        self.groups = decision_groups(static, layout)
        edges, slots = static["edges"], static["action_slots"]
        self.u0 = np.array([edges["u0"][e] or np.inf for e in slots["edge"]], dtype=float)
        self.ov_u0 = np.array([edges["u0"][e] or np.inf for e in static["override_slots"]["out_edge"]], dtype=float)
        self.w_scr = tuple(int(self.inst.nodes[f].fab.w_scr) for f in self.inst.fabs)
        pref = []  # the groups the offline search corrected most often, first
        for want in (
            ("tanker", 0),
            ("wafer", 0),
            ("raw", 0),
            ("fuel", 0),
            ("fuel", 3),
            ("fuel", 1),
            ("tanker", 9),
            ("wafer", 2),
        ):
            for g, (kind, i, _idx) in enumerate(self.groups):
                if (kind, i) == want and g not in pref:
                    pref.append(g)
        self.order = pref + [g for g in range(len(self.groups)) if g not in pref]
        self.cursor = 0
        self.head = head_mod.Agent(config)
        self.tail = tail_mod.Agent(config)
        big = len(slots["edge"]) > 200
        self.budget = SEARCH["cpu_budget_full"] if big else SEARCH["cpu_budget"]
        self.searched = self.applied = 0

    def _branch_cost(self, o, t, action, marks, state):
        """Cost to the episode's end: ``action`` now, the head MPC for head_weeks weeks, the LP tail after."""
        st = copy.deepcopy(state)
        head, tail = copy.deepcopy(self.head), copy.deepcopy(self.tail)
        total, act = 0.0, action
        for h in range(self.T - t + 1):
            week = t + h
            try:
                flows, ov, holds, inv = validate_action(self.inst, marks, week, action_to_wire(self.lay, week, act))
            except ValueError:
                return np.inf
            rec = step(self.inst, marks, st, flows, ov, holds, inv)
            total += rec.costs.total()
            if week >= self.T:
                break
            obs_flat = observation_dict(self.lay, observe(self.inst, marks, st))
            act = head.act(obs_flat) if h < SEARCH["head_weeks"] else tail.act(obs_flat)
        return total

    def act(self, observation):
        t0 = time.process_time()
        o = observation
        t = int(o["week"][0])
        action = {k: np.array(v, copy=True) for k, v in self.mpc.act(o).items()}
        if t >= self.T - 1 or not self.groups or time.process_time() - t0 > self.budget * 0.4:
            return action
        try:
            marks = persistence_marks(o, self.inst, self.lay, self.static, self.w_scr)
            state = rebuild_state(o, self.inst, self.lay, t)
            g = self.order[self.cursor % len(self.order)]
            self.cursor += 1
            kind, _i, idx = self.groups[g]
            cand = {k: np.array(v, copy=True) for k, v in action.items()}
            if kind == "tanker":
                if "override_qty" not in cand:
                    return action
                q = cand["override_qty"]
                q[idx] = np.minimum(q[idx] * SEARCH["level"], self.ov_u0[idx])
            else:
                f = cand["flows"]
                f[idx] = np.minimum(f[idx] * SEARCH["level"], self.u0[idx] * o["action_mask"][idx])
            t1 = time.process_time()
            base = self._branch_cost(o, t, action, marks, state)
            branch = time.process_time() - t1
            if time.process_time() - t0 + branch * 1.1 > self.budget:
                return action
            alt = self._branch_cost(o, t, cand, marks, state)
            self.searched += 1
            if alt < base - 1e-6 - SEARCH["min_gain"] * abs(base):
                self.applied += 1
                return cand
        except Exception:  # noqa: BLE001 - the search must never cost the week
            pass
        return action

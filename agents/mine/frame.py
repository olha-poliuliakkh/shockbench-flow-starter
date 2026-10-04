"""The frozen frame of the evolvable LP agent (docs/EVOLVE_DESIGN.md, section 4.1). The mutator never edits this file.

Each week: the Dict observation is turned back into the fields of the server's wire observation that the vendored
planner reads (``wire_from_dict``), the planner's memory is updated, ``State`` gives the blocks name-based views, the
persistence window of the package's ``mpc_det`` is built (``Window``, writable through consistency-keeping helpers),
the window LP is built by the oracle's own builder, edited (``LPEdits``) and solved with SciPy's HiGHS dual simplex
(the server has no highspy), and its first week becomes the Dict action. ``sbfplan`` is shockbench_flow 0.1.2,
vendored by ``scripts/vendor_planner.py`` (MIT, see sbfplan/LICENSE).

Imports: the standard library, numpy, SciPy and the vendored package; no randomness anywhere.
"""

import math
import time

import numpy as np
from sbfplan.instance.schema import WAR_RISK_CLASSES
from sbfplan.policies import lp_common as L
from sbfplan.policies.naive_parts import cached_plan
from scipy.optimize import linprog


CHANNELS = ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
KINDS = ("proposal", "final_notice", "threat", "publication", "withdrawal")
TARGET_KINDS = ("chokepoint", "edge", "node", "region")
# The window's LP size is capped deterministically (timing never changes a plan, so scores reproduce): measured
# cold dual-simplex solves took 0.12 s at 10,000 columns (Small, H = L) and 0.57 s at 27,800 (Full, H = L), growing
# about as columns^1.5, so these caps keep a solve near a quarter of the 2 s (Small) or 4 s (Full) week.
MAX_COLUMNS = {2.0: 26_000, 4.0: 40_000}


def _entries(obs: dict, block: str, fields: tuple[str, ...]) -> list[tuple]:
    """The live entries of a padded list block, as tuples of ``fields`` (None where a field is unobserved)."""
    seen = obs[f"{block}.{fields[0]}.observed"].astype(bool)
    cols = []
    for f in fields:
        vals, ok = obs[f"{block}.{f}"], obs[f"{block}.{f}.observed"].astype(bool)
        cols.append([v.item() if o else None for v, o in zip(vals[seen], ok[seen])])
    return list(zip(*cols))


def _cols(rows: list[tuple], fields: tuple[str, ...]) -> dict:
    return {f: [r[i] for r in rows] for i, f in enumerate(fields)}


def _vals(a: np.ndarray, seen: np.ndarray) -> list:
    return [x.item() if s else None for x, s in zip(a, seen.astype(bool))]


def wire_from_dict(obs: dict, cfg: dict, entry_edge: dict) -> dict:
    """The wire-observation fields the planner reads, rebuilt from the agent's Dict observation.

    On Small and Full the pipeline arrives grouped by (edge, k, lane, arrival week) and the queued lots as a dense
    (lot key, arrival week) block: a group is one shipment or lot here, which the LP treats alike. A lot's entry edge
    is its lane's edge into the strait and its dispatch week that edge's nominal lead before arrival; the LP reads
    neither (tests/test_planner_port.py checks the objective week by week).
    """
    lay, edges = cfg["layout"], cfg["static"]["edges"]
    week = int(obs["week"][0])
    stock = [(n, k, float(q)) for (n, k), q in zip(lay["stock_slots"], obs["stock.qty"])]
    backlog = [(n, k, float(q)) for (n, k), q in zip(lay["demands"], obs["backlog.qty"])]
    pfields = ("edge", "k", "lane", "qty", "arrival_week")
    if "lot_keys" in lay:
        lots = []
        qty, seen = obs["queue_lots.qty"], obs["queue_lots.qty.observed"]
        for i, j in zip(*np.nonzero(seen)):
            c, k, lane, nxt = lay["lot_keys"][i]
            e = entry_edge[(c, lane)]
            arr = int(j) + 1
            lots.append((int(c), int(k), float(qty[i, j]), int(lane), int(nxt), arr, arr - int(edges["tau0"][e]), e))
    else:  # tiny: the lots as listed
        lots = _entries(
            obs,
            "queue_lots",
            ("chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge"),
        )
    lfields = ("chokepoint", "k", "qty", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge")
    wire = {
        "week": week,
        "stock": _cols(stock, ("node", "k", "qty")),
        "backlog": _cols(backlog, ("node", "k", "qty")),
        "pipeline": _cols(_entries(obs, "pipeline", pfields), pfields),
        "queue_lots": _cols(lots, lfields),
        "wip": _cols(_entries(obs, "wip", ("node", "k", "qty", "out_week")), ("node", "k", "qty", "out_week")),
    }
    if not (obs["graph_now.open.observed"].any() or obs["graph_now.u.observed"].any()):
        wire["graph_now"] = None  # a blackout week (none in the scored regime)
    else:
        pe, pk = np.nonzero(obs["graph_now.prohibited"])
        te, tk = np.nonzero(obs["graph_now.tariff"])
        wr = obs["graph_now.war_risk"]
        wseen = obs["graph_now.war_risk.observed"].astype(bool)
        supply = [
            (n, k, float(v))
            for (n, k), v, s in zip(
                lay["supply_slots"], obs["graph_now.supply.avail"], obs["graph_now.supply.avail.observed"]
            )
            if s
        ]
        wire["graph_now"] = {
            "u": _vals(obs["graph_now.u"], obs["graph_now.u.observed"]),
            "c": _vals(obs["graph_now.c"], obs["graph_now.c.observed"]),
            "open": _vals(obs["graph_now.open"], obs["graph_now.open.observed"]),
            "kappa": {
                b: _vals(obs[f"graph_now.kappa.{b}"], obs[f"graph_now.kappa.{b}.observed"]) for b in ("tb", "ct")
            },
            "war_risk": [WAR_RISK_CLASSES[int(v)] if s else None for v, s in zip(wr, wseen)],
            "supply": _cols(supply, ("node", "k", "avail")),
            "fab": {
                "node": list(lay["fabs"]),
                **{
                    f: _vals(obs[f"graph_now.fab.{f}"], obs[f"graph_now.fab.{f}.observed"])
                    for f in ("R", "alpha_bar", "cap_eff")
                },
            },
            "osat": {
                "node": list(lay["osats"]),
                **{f: _vals(obs[f"graph_now.osat.{f}"], obs[f"graph_now.osat.{f}.observed"]) for f in ("R", "thr_eff")},
            },
            "grid": {
                "node": list(lay["grids"]),
                **{
                    f: _vals(obs[f"graph_now.grid.{f}"], obs[f"graph_now.grid.{f}.observed"])
                    for f in ("G_bar", "y_bar")
                },
            },
            "prohibited": {"edge": pe.tolist(), "k": pk.tolist()},
            "tariff": {"edge": te.tolist(), "k": tk.tolist(), "rate": obs["graph_now.tariff"][te, tk].tolist()},
        }
    pend = _entries(obs, "pending_prohibitions", ("edge", "k", "effective_week"))
    wire["pending_prohibitions"] = _cols(pend, ("edge", "k", "effective_week"))
    fq, fs = obs["demand_forecast.qty"], obs["demand_forecast.qty.observed"]
    fc = [(n, k, h, float(fq[d, h])) for d, (n, k) in enumerate(lay["demands"]) for h in range(fq.shape[1]) if fs[d, h]]
    wire["demand_forecast"] = _cols(fc, ("node", "k", "h", "qty"))
    return wire


class State:
    """What the blocks read: the week, the planner's memory, positions, routes and signals, by name.

    Attributes: ``week``, ``T``, ``obs`` (the Dict observation), ``wire`` (the rebuilt wire fields), ``chokepoints``
    (node indices, the window's column order), ``chokepoint_names``, ``open_now`` (open fraction per strait),
    ``closed_for`` (weeks each strait has been below fully open), ``warnings`` ({(kind, id): score}), ``threads``
    (live announcement threads, one dict per message: msg_id, channel, kind, region, target_kind, target, k,
    announced_week, stated_effective_week), ``pending`` ((edge, k, effective week) triples), ``regime`` (the published
    decoy shares ``phi`` and warning parameters), ``lanes_through`` ({strait node: [lane indices]}), ``edges_of_lane``,
    ``edge_names``, ``node_names``, ``node_region``, ``commodity_names``, ``demands`` ((node, k) per demand row),
    ``grids`` (node indices), ``demand_forecast`` ((demands, 8) array).
    """

    def __init__(self, planner: "Planner", obs: dict, wire: dict):
        cfg, st = planner.cfg, planner.cfg["static"]
        self.week, self.T, self.obs, self.wire = wire["week"], planner.inst.T, obs, wire
        self.regime = cfg.get("regime") or st.get("regime") or {}
        self.chokepoints = list(cfg["layout"]["chokepoints"])
        self.node_names = list(st["nodes"]["id"])
        self.node_region = list(st["nodes"]["region"])
        self.chokepoint_names = [self.node_names[c] for c in self.chokepoints]
        self.edge_names = list(st["edges"]["id"])
        self.commodity_names = list(st["commodities"]["id"])
        self.demands = [tuple(d) for d in cfg["layout"]["demands"]]
        self.grids = list(cfg["layout"]["grids"])
        self.open_now = np.asarray(obs["graph_now.open"], dtype=float)
        self.closed_for = planner.closed_for.copy()
        self.warnings = {
            (kind, int(unit)): float(v)
            for (kind, unit), v, s in zip(
                cfg["layout"]["warning_units"], obs["warning.score"], obs["warning.score.observed"]
            )
            if s
        }
        fields = (
            "msg_id",
            "channel",
            "kind",
            "region",
            "target_kind",
            "target",
            "k",
            "announced_week",
            "stated_effective_week",
        )
        self.threads = []
        for row in _entries(obs, "messages", fields):
            m = dict(zip(fields, row))
            m["channel"] = CHANNELS[m["channel"]] if m["channel"] is not None else None
            m["kind"] = KINDS[m["kind"]] if m["kind"] is not None else None
            m["target_kind"] = TARGET_KINDS[m["target_kind"]] if m["target_kind"] is not None else None
            self.threads.append(m)
        self.pending = list(planner.memory.pending)
        self.lanes_through = planner.lanes_through
        self.edges_of_lane = planner.edges_of_lane
        self.demand_forecast = np.asarray(obs["demand_forecast.qty"], dtype=float)

    def decoy_share(self, channel: str) -> float:
        """The published share of decoy threads on ``channel`` (0 for channels without a shadow process)."""
        return float((self.regime.get("phi") or {}).get(channel, 0.0))

    def strait_pos(self, node: int) -> int | None:
        """The window column of strait ``node``, or None if it is no strait."""
        return self.chokepoints.index(node) if node in self.chokepoints else None


class Window:
    """The H-week forecast the LP plans on (row h is week t + h). Edit it only through the helpers.

    Arrays: ``a[name]`` for every field of ``lp_common.WINDOW_FIELDS`` (u, c, o, kappa, supply, G_bar, y_bar, R,
    alpha_bar, sigma_scr, R_osat, demand, prohibited, tariff, wr_class, h_queue, c_wr and the ``*_now`` copies).
    Strait columns follow ``State.chokepoints``; edges, commodities and stock slots follow Static's tables.
    """

    def __init__(self, arrays: dict, t: int, inst):
        self.a = {k: np.array(v) for k, v in arrays.items()}  # writable copies
        self.t, self.H, self._inst = t, int(self.a["o"].shape[0]), inst
        self._kappa_full = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=float)
        self._wr_changed = False

    def _rows(self, start: int, end: int | None) -> slice:
        return slice(max(0, int(start)), self.H if end is None else max(0, min(self.H, int(end))))

    def scale_open(self, pos: int, factor: float, start: int = 0, end: int | None = None) -> None:
        """Multiply strait ``pos``'s open fraction (and its pools' throughput) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        self.a["o"][r, pos] = np.clip(self.a["o"][r, pos] * max(0.0, float(factor)), 0.0, 1.0)
        self.a["kappa"][r, pos, :] = self._kappa_full[pos] * self.a["o"][r, pos][:, None]

    def set_open(self, pos: int, value: float, start: int = 0, end: int | None = None) -> None:
        """Set strait ``pos``'s open fraction to ``value`` (0..1) in rows start..end-1, throughput alike."""
        r = self._rows(start, end)
        self.a["o"][r, pos] = float(np.clip(value, 0.0, 1.0))
        self.a["kappa"][r, pos, :] = self._kappa_full[pos] * self.a["o"][r, pos][:, None]

    def scale_capacity(self, edge: int, factor: float, start: int = 0, end: int | None = None) -> None:
        """Multiply edge ``edge``'s weekly capacity by ``factor`` (>= 0) in rows start..end-1."""
        r = self._rows(start, end)
        self.a["u"][r, edge] = self.a["u"][r, edge] * max(0.0, float(factor))

    def prohibit(self, edge: int, k: int, on: bool = True, start: int = 0, end: int | None = None) -> None:
        """Switch the prohibition of (edge, k) on or off in rows start..end-1."""
        self.a["prohibited"][self._rows(start, end), edge, k] = bool(on)

    def set_tariff(self, edge: int, k: int, rate: float, start: int = 0, end: int | None = None) -> None:
        """Set the ad valorem tariff rate of (edge, k) in rows start..end-1."""
        self.a["tariff"][self._rows(start, end), edge, k] = max(0.0, float(rate))

    def scale_demand(self, factor: float, row: int | None = None, start: int = 0, end: int | None = None) -> None:
        """Multiply demand (one demand row, or all) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        cols = slice(None) if row is None else row
        self.a["demand"][r, cols] = self.a["demand"][r, cols] * max(0.0, float(factor))

    def scale_supply(self, factor: float, slot: int | None = None, start: int = 0, end: int | None = None) -> None:
        """Multiply supply availability (one stock slot, or all) by ``factor`` in rows start..end-1."""
        r = self._rows(start, end)
        cols = slice(None) if slot is None else slot
        self.a["supply"][r, cols] = self.a["supply"][r, cols] * max(0.0, float(factor))

    def set_war_risk(self, pos: int, cls: int, start: int = 0, end: int | None = None) -> None:
        """Set strait ``pos``'s war-risk class (0 none, 1 red_sea, 2 hormuz_2026) in rows start..end-1.

        Queue holding and transit war-risk costs follow the class.
        """
        self.a["wr_class"][self._rows(start, end), pos] = int(cls)
        self._wr_changed = True

    def arrays(self) -> dict:
        """The window as the LP builder takes it: ``*_now`` copies and war-risk costs brought in line, read-only."""
        if self._wr_changed:
            hq, cwr = L.window_queue_and_transit(self._inst, self.a["wr_class"])
            self.a["h_queue"], self.a["c_wr"] = np.array(hq), np.array(cwr)
        out = L.with_now(dict(self.a))
        return L.read_only({k: np.array(v) for k, v in out.items()})


class LPEdits:
    """Edits of the window LP applied after it is built. Weeks are window rows 0..H-1 (row 0 is this week).

    Column tags of the oracle's model: x (edge, k, lane) flows, I (stock slot) end-of-week stock, Q (strait, k, lane)
    queues, O (slot) disposal, ysh (grid ordinal) power shed, U (demand ordinal) unserved demand, D served demand, and
    production and energy columns (p, E, xi, lam, G, short, y, rho, lift).
    """

    def __init__(self):
        self.premiums: list[tuple] = []  # (edge, k or None, usd per unit, start, end)
        self.floors: list[tuple] = []  # (stock slot, qty, start, end)
        self.scales: list[tuple] = []  # (tag, first key field or None, factor, start, end)

    def premium(self, edge: int, usd_per_unit: float, k: int | None = None, start: int = 0, end: int | None = None):
        """Add a cost per unit shipped on ``edge`` (commodity ``k`` or all): a risk premium on that route."""
        self.premiums.append((int(edge), k, float(usd_per_unit), start, end))
        return self

    def stock_floor(self, slot: int, qty: float, start: int = 0, end: int | None = None):
        """Ask the LP to keep at least ``qty`` in stock slot ``slot`` at the end of rows start..end-1."""
        self.floors.append((int(slot), max(0.0, float(qty)), start, end))
        return self

    def scale_cost(self, tag: str, factor: float, first: int | None = None, start: int = 0, end: int | None = None):
        """Multiply the objective coefficient of columns ``tag`` (whose first key field is ``first``, or all)."""
        self.scales.append((tag, first, float(factor), start, end))
        return self


class Planner:
    """The package's mpc_det (horizon H = L, persistence forecast, oracle LP, week-1 action), solved by SciPy."""

    def __init__(self, config: dict):
        self.cfg = config
        self.inst, plan = cached_plan(config["static"], None)
        self.L = L.lead_time_L(self.inst, plan)
        self.H = L.horizon_length(L.CANONICAL_H, self.L)
        self.memory = L.ObservedGraph.nominal(self.inst)
        self.budget = 2.0 if self.inst.T <= 52 else 4.0
        self.week_columns = None  # LP columns per window week, known after the first build
        self.closed_for = np.zeros(len(self.inst.chokepoints), dtype=int)
        st = config["static"]
        lanes = st["lanes"]
        self.edges_of_lane = [list(e) for e in lanes["edges"]]
        self.lanes_through = {}
        self.entry_edge = {}
        heads = st["edges"]["head"]
        for li, (es, cs) in enumerate(zip(lanes["edges"], lanes["chokepoints"])):
            for c in cs:
                self.lanes_through.setdefault(c, []).append(li)
                self.entry_edge[(c, li)] = next(e for e in es if heads[e] == c)
        a = config["spaces"]["action"]
        self.n_slots = a["flows"]["shape"][0]
        self.n_override = a["override_qty"]["shape"][0]
        self.pairs = {tuple(p): i for i, p in enumerate(config["layout"]["release_pairs"])}
        ov = st["override_slots"]
        self.override_pair = [self.pairs[(c, k)] for c, k in zip(ov["chokepoint"], ov["k"])]
        u0 = st["edges"]["u0"]
        self.capacity = np.array([u0[e] or 0.0 for e in st["action_slots"]["edge"]], dtype=float)
        self._week_start = None
        self.last_cpu = 0.0
        self.model = None
        self.last_status = None
        self.last_objective = None

    # ----- the week -------------------------------------------------------------------------------------------------
    def observe(self, obs: dict) -> State:
        """Rebuild the wire fields, update the memory, and return the week's ``State``."""
        self._week_start = time.process_time()  # the agent's own CPU this week, closed by _done()
        wire = wire_from_dict(obs, self.cfg, self.entry_edge)
        self.memory.update(self.inst, wire)
        open_now = np.asarray(obs["graph_now.open"], dtype=float)
        self.closed_for = np.where(open_now < 1.0, self.closed_for + 1, 0)
        return State(self, obs, wire)

    def max_H(self) -> int:
        """The longest window the LP-size cap allows (H itself before the first build)."""
        if not self.week_columns:
            return self.H
        return max(1, MAX_COLUMNS[self.budget] // self.week_columns)

    def default_H(self, s: State) -> int:
        return L.window_length(min(self.H, self.max_H()), s.week, self.inst.T)

    def clamp_H(self, s: State, H) -> int:
        """A block's H, kept in [ceil(L/2), the LP-size cap] and within the episode."""
        lo = math.ceil(self.L / 2)
        hi = max(lo, self.max_H())
        return L.window_length(min(max(int(round(float(H))), lo), hi), s.week, self.inst.T)

    def window(self, s: State, H: int) -> Window:
        return Window(L.persistence_arrays(self.inst, s.wire, self.memory, H), s.week, self.inst)

    def solve(self, s: State, w: Window, edits: LPEdits | None = None):
        """Build the window LP, apply ``edits``, solve; the wire action, or None when HiGHS returns no optimum."""
        arrays = w.arrays()
        model = L.rolled_lp(self.inst, s.wire, arrays, w.H, planning_rules=True)
        c = np.array(model.objective(), dtype=float)
        lb, ub = np.array(model.lb, dtype=float), np.array(model.ub, dtype=float)
        if edits is not None and (edits.premiums or edits.floors or edits.scales):
            self._apply(model, edits, c, lb, ub)
        kw = {}
        if model.A_ub.shape[0]:
            kw.update(A_ub=model.A_ub, b_ub=model.b_ub)
        if model.A_eq.shape[0]:
            kw.update(A_eq=model.A_eq, b_eq=model.b_eq)
        # a safety net only: a solve this long loses the week to naive anyway
        res = linprog(c, bounds=np.column_stack([lb, ub]), method="highs-ds", options={"time_limit": self.budget}, **kw)
        self.model, self.last_status = model, int(res.status)
        self.week_columns = self.week_columns or len(model.columns)
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        x = np.asarray(res.x, dtype=float)
        self.last_objective = float(model.objective() @ x) + L.model_offset(model)
        return L.week1_action(self.inst, model, x, s.wire, L.prohibited_now(self.memory, s.week))

    def solve_safely(self, s: State, w: Window, edits: LPEdits | None = None):
        try:
            return self.solve(s, w, edits)
        except Exception:
            return None

    def _apply(self, model, edits: LPEdits, c, lb, ub) -> None:
        nc = len(model.columns)
        cols = model.columns
        for edge, k, usd, a, b in edits.premiums:
            js = [j for j, key in enumerate(cols) if key[0] == "x" and key[1] == edge and (k is None or key[2] == k)]
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                c[[h * nc + j for j in js]] += usd
        slot_col = {key[1]: j for j, key in enumerate(cols) if key[0] == "I"}
        for slot, qty, a, b in edits.floors:
            j = slot_col.get(slot)
            if j is None:
                continue
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                lb[h * nc + j] = min(max(lb[h * nc + j], qty), ub[h * nc + j])
        for tag, first, factor, a, b in edits.scales:
            js = [j for j, key in enumerate(cols) if key[0] == tag and (first is None or key[1] == first)]
            for h in range(max(0, a), model.T if b is None else min(model.T, b)):
                c[[h * nc + j for j in js]] *= factor

    # ----- actions --------------------------------------------------------------------------------------------------
    def _done(self) -> None:
        """Record the week's own CPU (``last_cpu``, for diagnostics; it never changes a plan)."""
        if self._week_start is not None:
            self.last_cpu = time.process_time() - self._week_start
            self._week_start = None

    def send_max(self, obs: dict) -> dict:
        self._done()
        return {"flows": self.capacity * obs["action_mask"]}

    def to_dict_action(self, wire: dict) -> dict:
        """The Dict action of a wire action: flows by slot, override quantities and a release mode per pair."""
        self._done()
        flows = np.zeros(self.n_slots)
        f = wire.get("flows") or {}
        for sl, q in zip(f.get("slot", []), f.get("qty", [])):
            flows[sl] = q
        override = np.zeros(self.n_override)
        modes = np.zeros(len(self.pairs), dtype=np.int64)
        ov = wire.get("overrides") or {}
        for o, q in zip(ov.get("slot", []), ov.get("qty", [])):
            override[o] = q
            modes[self.override_pair[o]] = 1
        hold = wire.get("hold") or {}
        for c, k in zip(hold.get("chokepoint", []), hold.get("k", [])):
            modes[self.pairs[(c, k)]] = 2
        flows = np.where(np.isfinite(flows), np.maximum(flows, 0.0), 0.0)
        override = np.where(np.isfinite(override), np.maximum(override, 0.0), 0.0)
        return {"flows": flows, "override_qty": override, "release_mode": modes}

"""Frozen dataclasses of an instance (design §2.1, §2.3, §2.4; Q58, Q70, Q79, Q85).

An instance is one canonical JSON file (`instance/io.py`), loaded into the dataclasses below. Integer codes follow
design §4.1 (Q87): regions, nodes, edges, lanes and commodities are indexed by their position in the file's lists, and
every cross-reference below is such an index. Chokepoints, grids, fabs, OSATs and sink demands also have *ordinals*,
their order of appearance among the nodes (``Instance.chokepoints`` etc.); arrays over chokepoints (marks, the omega
container's ``wr_class``) are indexed by that ordinal.

Money is in USD per unit here; costs become integer cents only in the cost function (24).
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from sbfplan.digest import semantic_digest


SCHEMA_VERSION = "1"

# Vocabularies, frozen with schema_version (design §9.2; Q86)
NODE_TYPES = ("source", "terminal", "grid", "chokepoint", "material", "fab", "osat", "sink")
MODES = ("sea", "air", "pipeline", "grid")
POOLS = ("tb", "ct")  # tanker-and-bulk (energy, GWh) and container (semiconductor goods, wafer-eq) (Q49, Q76)
WAR_RISK_CLASSES = ("none", "red_sea", "hormuz_2026")  # codes 0, 1, 2 of the omega container's wr_class
PRIORITIES = ("base_first", "proportional", "industrial_first")  # energy priority pri_g (18) (Q57)
FAB_CLASSES = ("leading", "mature", "memory")
SUPPLY_TYPES = ("source", "material")  # V^sup (Q79)
UNMODELLED = "unmodelled"  # JSON key of the unmodelled generation segment k = ∅ in (15), (17), (18)
WEEKS_PER_YEAR = 52  # length of a seasonal profile m^sea (§2.1)
# instance kinds, whose index is the instance-kind code of the seed keys (design §4.1; Q87); defined here, next to the
# other vocabularies, so that the loader checks `kind` without importing omega (omega/codes.py re-exports it)
INSTANCE_KINDS = ("tiny", "small", "full", "abstract")
FLAGSHIP_KINDS = ("small", "full")  # the sizes built from spec §4.6-4.7 (design §2.2; Q83)


class FrozenDict(dict):
    """An immutable dict for the mapping fields of frozen dataclasses.

    Unlike ``MappingProxyType`` it pickles, so instances, marks and environment snapshots cross process boundaries
    (joblib workers, resume, B2).
    """

    def _readonly(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __reduce__(self):
        return (FrozenDict, (dict(self),))

    def __hash__(self):  # frozen dataclasses holding one stay hashable when their values are
        return hash(tuple(sorted(self.items(), key=repr)))


def frozen_map(d: Mapping) -> Mapping:
    """An immutable (and picklable) copy of a dict, for mapping fields of frozen dataclasses."""
    return FrozenDict(d)


@dataclass(frozen=True)
class Provenance:
    """Tag and source of one leaf parameter (design conventions; V20, V23)."""

    tag: str  # REAL, DERIVED(...), CALIBRATED(...), SYNTHETIC(prior...), SYNTHETIC, SYNTHETIC(placeholder)
    source: str
    lo: float | None = None
    hi: float | None = None

    @property
    def placeholder(self) -> bool:
        return self.tag.startswith("SYNTHETIC(placeholder")


@dataclass(frozen=True)
class Commodity:
    """A commodity of §2.1; ``override`` marks tanker cargo K^ov (Q34)."""

    id: str
    unit: str
    pool: str  # "tb" or "ct"
    v: float  # customs value v_k, USD per unit
    override: bool
    disposal_cost: float  # c^disp_k, USD per unit (Q24)

    @property
    def pool_index(self) -> int:
        return POOLS.index(self.pool)


@dataclass(frozen=True)
class StockSlot:
    """One stock I_ik of (8): a (node, commodity) pair the node can hold.

    At a chokepoint the slot is the queue: ``storage`` is None (no cap, Q58 M4), ``holding`` is unused (queue holding
    h^Q by war-risk class is in ``ChokepointAttrs``) and ``salvage`` is the queue's nu = c^min x the commodity's share,
    valued like every other node (Q93).
    """

    node: int
    k: int
    storage: float | None  # I^max_ik; None only at chokepoints
    holding: float  # h_ik, USD per unit-week; 0 at supply nodes (Q79)
    salvage: float  # nu_ik, USD per unit; 0 at supply nodes (Q58 E18, Q79), c^min x share elsewhere (chokepoints: Q93)
    supply: float = 0.0  # availability varsigma-bar^0_ik per week at supply nodes (Q79); 0 elsewhere


@dataclass(frozen=True)
class ChokepointAttrs:
    """Chokepoint attributes (§2.1, §3.4): throughput kappa_cb = k_c mu_cb o^t_c per pool (9)."""

    mu: tuple[float, float]  # normal traffic mu_cb per pool, (tb, ct)
    k_c: float  # clearance ratio (Q7)
    cls: str  # chokepoint class (free text in v1)
    queue_holding: Mapping[int, tuple[float, float, float]]  # k -> h^Q_ck by war-risk class (none, red_sea, hormuz)
    war_risk_cost: Mapping[int, tuple[float, float, float]]  # k -> per-transit cost c-bar^wr_k by class (11)

    @property
    def kappa0(self) -> tuple[float, float]:
        """Nominal throughput k_c mu_cb per pool (tb, ct), the factor of o^t_c in (9)."""
        return (self.k_c * self.mu[0], self.k_c * self.mu[1])


@dataclass(frozen=True)
class GridAttrs:
    """Grid attributes (§2.1, §3.5): segments (15), buffer (16), allocation (17)-(18)."""

    base_load: float  # y-bar_g, GWh/wk
    deliverable: float  # G-bar^0_g, GWh/wk
    shares: Mapping[int | None, float]  # zeta_gk by fuel commodity index; None = the unmodelled segment ∅
    days_cover: Mapping[int, float]  # D_g per fuel, days
    ibar: Mapping[int, float]  # reference buffer I-bar_g per fuel, GWh (16), as the instance declares it
    rationed: int | None  # commodity index of the gas segment, the only one rationed (Q27); None if no gas
    voll: float  # VOLL_g, USD/GWh (Q28)
    priority: str  # pri_g, one of PRIORITIES (Q57)

    @property
    def fuels(self) -> tuple[int, ...]:
        """K^E_g: the fuel commodities of the grid, in commodity order."""
        return tuple(sorted(k for k in self.shares if k is not None))


@dataclass(frozen=True)
class FabAttrs:
    """Fab attributes (§2.1, §3.5): (12)-(14)."""

    cap0: float  # wafers per week
    e: float  # GWh per wafer; 0 without a grid
    grid: int | None  # node index of the fab's grid, or None (energy outside the model)
    cls: str  # leading, mature or memory
    input: int  # commodity index of the wafer input
    product: int  # commodity index of the raw chip k^raw(f)
    tau: int  # tau^fab_f, weeks
    w_scr: int  # scrap window w^scr_f, 1 <= w_scr <= tau (14)


@dataclass(frozen=True)
class OsatAttrs:
    """OSAT attributes (§2.1, §3.5): (19)."""

    thr: float  # throughput thr_i per week over all packaged commodities
    tau: int  # tau^osat, weeks
    packages: Mapping[int, int]  # raw chip commodity -> packaged chip commodity


@dataclass(frozen=True)
class TerminalAttrs:
    """Terminal attributes (§2.1); the throughput enters as the capacity of the terminal's grid edge (§2.2)."""

    throughput: float


@dataclass(frozen=True)
class Node:
    """A node of §2.1 with its type-specific block (exactly one is set for chokepoints, grids, fabs, OSATs)."""

    id: str
    type: str  # one of NODE_TYPES
    region: int  # region index
    chokepoint: ChokepointAttrs | None = None
    grid: GridAttrs | None = None
    fab: FabAttrs | None = None
    osat: OsatAttrs | None = None
    terminal: TerminalAttrs | None = None


@dataclass(frozen=True)
class AltRef:
    """A tagged `alt_of` reference, ``{"lane": id}`` or ``{"edge": id}`` (Q85)."""

    kind: str  # "lane" or "edge"
    index: int


@dataclass(frozen=True)
class Edge:
    """An edge of §2.1; coupling edges (mode grid) carry no commodity and accept no request (Q58 M5)."""

    id: str
    tail: int
    head: int
    mode: str
    K: tuple[int, ...]  # permitted commodities K_e; empty on coupling edges
    tau: int  # lead time in weeks (fixed in v1, Q87)
    c0: float  # unit freight c^0_e, USD
    u0: float | None  # capacity u^0_e per week; None on coupling edges
    alt_of: AltRef | None
    pool: str | None  # one pool per edge (§2.1 proposal); None on coupling edges

    @property
    def coupling(self) -> bool:
        return self.mode == "grid"


@dataclass(frozen=True)
class Lane:
    """A lane l = (i, c_1, ..., c_r, j): a path whose interior nodes are chokepoints (Q4)."""

    id: str
    edges: tuple[int, ...]
    chokepoints: tuple[int, ...]  # interior node indices, in path order
    alt_of: AltRef | None


@dataclass(frozen=True)
class Demand:
    """Demand of packaged chip k at sink i, (20)-(22)."""

    node: int
    k: int
    dbar: float  # base demand d-bar_ik = upsilon F-bar_ik (22); the lognormal's median (§12)
    pi: float  # shortage penalty pi_ik, USD per unit (Q85)
    backlog: bool  # True: backlog sink; False: lost sales (20)
    phi: float  # AR(1) coefficient phi^d of (21)
    sigma: float  # innovation SD sigma^d of (21)
    seasonal: tuple[float, ...] | None  # m^sea_ik by week of year (52 values), None = identically 1
    shock: tuple[float, float, float]  # Xi_ik by the sink region's conflict layer (none, minor, war)

    def m_sea(self, t: int) -> float:
        """m^sea(t) of (21) for episode week t: ``seasonal[(t - 1) % 52]``, 1 without a profile."""
        return 1.0 if self.seasonal is None else self.seasonal[(t - 1) % WEEKS_PER_YEAR]


@dataclass(frozen=True)
class InitialShipment:
    """A pipeline shipment x^{t'<=0}_ek of the initial state (2), carrying its lane when it rides one."""

    edge: int
    k: int
    lane: int | None
    qty: float
    dispatch_week: int  # t' <= 0
    arrival_week: int  # t' + tau_e >= 1


@dataclass(frozen=True)
class InitialWip:
    """Gross work in process maturing at ``out_week`` (fab: raw chips; OSAT: packaged chips)."""

    node: int
    k: int  # the output commodity
    qty: float
    out_week: int  # 1 <= out_week <= tau


@dataclass(frozen=True)
class InitialLot:
    """A queue lot at a chokepoint at reset (§3.4 lot fields; Q58 M13)."""

    chokepoint: int  # node index
    k: int
    qty: float
    lane: int
    next_edge: int
    dispatch_week: int
    entry_edge: int
    arrival_week: int  # <= 0


@dataclass(frozen=True)
class InitialState:
    """The declared initial state (§2.3, §2.4): stock, pipeline, WIP and queue lots."""

    stock: tuple[tuple[int, int, float], ...]  # (node, k, qty) for every nonzero stock, supply nodes included
    pipeline: tuple[InitialShipment, ...]
    fab_wip: tuple[InitialWip, ...]
    osat_wip: tuple[InitialWip, ...]
    queue_lots: tuple[InitialLot, ...]


@dataclass(frozen=True)
class Params:
    """Instance-wide parameters (§2.4, §3.3, §3.5)."""

    psi: float  # rationing threshold psi (Q36)
    alpha_max: float  # overproduction ceiling alpha_max (13)
    tau_alpha: float  # ceiling lag tau_alpha in weeks (13)
    upsilon: float  # utilisation (22) (Q76 owner D2)
    fleet_share: tuple[float, float]  # s^b_fl per pool (tb, ct) (7)
    fleet_measure: tuple[float, float]  # F^b per pool (tb, ct), GWh-wk and wafer-eq-wk (7); declared, checked (§3.3)
    top_tariff: float  # the instance's top tariff rate, which sets pi (Q85)
    forecast_shares: tuple[float, ...]  # w^f_j, j = 0..8 (48)


@dataclass(frozen=True)
class Instance:
    """A loaded instance (§2.3). Build it with ``instance.io.load_instance``; never mutate it."""

    schema_version: str
    instance_id: str
    kind: str  # tiny, small, full or abstract (instance-kind code of §4.1)
    T: int
    units: Mapping[str, str]
    regions: tuple[str, ...]
    region_class: tuple[str, ...]  # per region (Q59)
    commodities: tuple[Commodity, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    lanes: tuple[Lane, ...]
    stock_slots: tuple[StockSlot, ...]
    demands: tuple[Demand, ...]
    routing_table: tuple[tuple[int, int, tuple[int, ...]], ...]  # (origin region, dest region, lanes)
    compatibility: tuple[tuple[int, int], ...]  # (source node, terminal node)
    use: tuple[tuple[int, int], ...]  # (material node, fab node)
    chokepoint_adjacency: Mapping[int, tuple[int, ...]]  # chokepoint node -> adjacent regions (Q58)
    trade_adjacency: tuple[tuple[int, int, float], ...]  # (region, region, weight) entries of A (R3 §3)
    initial_state: InitialState
    prohibitions_at_reset: tuple[tuple[int, int], ...]  # (edge, k) pairs of Z_0, in force all episode
    params: Params
    provenance: Mapping[str, Provenance]
    hash: str  # SHA-256 of the canonical JSON (§2.3)
    raw: Mapping = field(repr=False, compare=False)  # the parsed canonical JSON, shipped in Static (Q86)

    # ----- derived index tables, filled by the loader -------------------------------------------------------------
    node_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    edge_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    lane_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    commodity_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    region_index: Mapping[str, int] = field(repr=False, compare=False, default_factory=dict)
    out_edges: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # delta+(i)
    in_edges: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # delta-(i)
    chokepoints: tuple[int, ...] = field(repr=False, compare=False, default=())  # node indices, ordinal order
    grids: tuple[int, ...] = field(repr=False, compare=False, default=())
    fabs: tuple[int, ...] = field(repr=False, compare=False, default=())
    osats: tuple[int, ...] = field(repr=False, compare=False, default=())
    sinks: tuple[int, ...] = field(repr=False, compare=False, default=())
    supply_nodes: tuple[int, ...] = field(repr=False, compare=False, default=())
    slot_index: Mapping[tuple[int, int], int] = field(repr=False, compare=False, default_factory=dict)  # (i,k)->slot
    # the fleet-slack terms of (7), the one table of E^dup: (edge, lane or None, Delta tau). An edge-level sea
    # duplicate counts all its flow (lane None); a sea duplicate lane is charged its extra transit over the route it
    # replaces, counted from where it diverges, on the edge where it diverges: on its entry edge, the flow dispatched
    # on that lane; on a turn-back out of a chokepoint, all the flow of that edge (design §3.3: "Cape, Lombok-Sunda
    # and east-of-Taiwan lanes and turn-back edges into them")
    dup_items: tuple[tuple[int, int | None, int], ...] = field(repr=False, compare=False, default=())
    action_slots: tuple[tuple[int, int, int | None], ...] = field(repr=False, compare=False, default=())
    override_slots: tuple[tuple[int, int, int, int | None], ...] = field(repr=False, compare=False, default=())
    action_slot_index: Mapping[tuple, int] = field(repr=False, compare=False, default_factory=dict)  # slot -> index
    override_slot_index: Mapping[tuple, int] = field(repr=False, compare=False, default_factory=dict)
    # ordinals of chokepoints, grids, fabs and OSATs by node index
    chokepoint_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    grid_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    fab_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    osat_ordinal: Mapping[int, int] = field(repr=False, compare=False, default_factory=dict)
    commodity_pool: tuple[int, ...] = field(repr=False, compare=False, default=())  # pool index per commodity
    grid_fabs: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # fab ordinals per grid ord.
    lane_K: tuple[tuple[int, ...], ...] = field(repr=False, compare=False, default=())  # K on every edge, per lane
    # (lane, chokepoint node) -> (edge into c, edge out of c = e_l(c)) of (52), for every interior chokepoint
    lane_through: Mapping[tuple[int, int], tuple[int, int]] = field(repr=False, compare=False, default_factory=dict)
    # (chokepoint node, out-edge) -> the lanes the out-edge continues, in lane order (out-edges on some lane only)
    lanes_continuing: Mapping[tuple[int, int], tuple[int, ...]] = field(repr=False, compare=False, default_factory=dict)
    # out-edges of chokepoints that continue some lane (they carry the war-risk transit cost (11))
    continuing_edges: frozenset[int] = field(repr=False, compare=False, default=frozenset())
    # the warm start per rung of §2.3 (owner queue M5-O37 (b)): the γ rung whose block ``initial_state.stock`` holds
    # (None: one start for every rung, `tiny` and a point-mass build) and every rung's block, that one included, as
    # (node, k, qty) entries; outside the content digest, which covers the block in use (``at_rung`` swaps it in)
    start_rung: float | None = field(repr=False, compare=False, default=None)
    rung_stock: Mapping[float, tuple[tuple[int, int, float], ...]] = field(
        repr=False, compare=False, default_factory=dict
    )
    # week-invariant tables of other modules (simulator, cost, observation), built on first use by ``memo``; not an
    # init field, so ``dataclasses.replace`` starts with an empty memo and never carries tables of another instance
    _memo: dict = field(init=False, repr=False, compare=False, default_factory=dict)

    def memo(self, key: str, build: Callable[["Instance"], Any]) -> Any:
        """The table ``build(self)``, built once per instance object and kept under ``key``."""
        if key not in self._memo:
            self._memo[key] = build(self)
        return self._memo[key]

    @property
    def content_digest(self) -> str:
        """SHA-256 of the instance's content (every compared field), the V3 identity the simulator and LP check."""
        return self.memo("content_digest", semantic_digest)

    @property
    def rungs(self) -> tuple[float, ...]:
        """The γ rungs with a stock block of their own, ascending; empty with one start for every rung (§2.3)."""
        return tuple(sorted(self.rung_stock))

    def at_rung(self, gamma: float) -> "Instance":
        """The instance at γ rung ``gamma``: that rung's block as ``initial_state.stock`` (§2.3; M5-O37 (b)).

        The file's hash label and raw JSON, its own content digest; everything but the on-hand stock is shared. The
        instance itself at its own rung and on an instance with one start for every rung (`tiny`, a point-mass build),
        so those never move; built once per instance and rung (``memo``).

        Raises:
            ValueError: if the instance keeps a block per rung and none for ``gamma``.

        """
        if not self.rung_stock or gamma == self.start_rung:
            return self
        if gamma not in self.rung_stock:
            raise ValueError(f"{self.instance_id} has no warm start at rung {gamma!r} (its rungs: {self.rungs}; §2.3)")
        return self.memo(f"at_rung:{gamma!r}", lambda inst: _at_rung(inst, gamma))

    def at_digest(self, digest: str) -> "Instance":
        """The instance at the rung whose content digest is ``digest`` (V3); the instance itself if none is.

        What an omega's ``meta_instance_digest`` or a record's ``instance_digest`` names: the episode's block.
        """
        if self.content_digest == digest:
            return self
        return next((r for r in map(self.at_rung, self.rungs) if r.content_digest == digest), self)

    @property
    def family_digest(self) -> str:
        """The content digest at the file's own rung: what naive's F_Q and the D9 fallback are a function of.

        F_Q reads no stock (§12 'Warm start per rung'), so the instances of one file at every rung share it; it is the
        content digest itself on an instance with one start and on the file as loaded.
        """
        return self.memo("family_digest", _family_digest)

    def ordinal(self, kind: str, node: int) -> int:
        """Ordinal of a node among ``kind`` ∈ {chokepoints, grids, fabs, osats}."""
        return getattr(self, _ORDINAL_FIELD[kind])[node]

    def lane_next_edge(self, lane: int, edge: int) -> int | None:
        """The lane's edge after ``edge`` (e_l(c) when ``edge`` enters chokepoint c), or None at the lane's end.

        None also when ``edge`` is not on the lane.
        """
        through = self.lane_through.get((lane, self.edges[edge].head))
        return through[1] if through is not None and through[0] == edge else None

    def lane_destination(self, lane: int) -> int:
        """Head node of the lane's last edge."""
        return self.edges[self.lanes[lane].edges[-1]].head

    def continues_lane(self, edge: int) -> bool:
        """True if ``edge`` leaves a chokepoint as the continuation of some lane (war-risk transit cost, (11))."""
        return edge in self.continuing_edges


def _at_rung(inst: Instance, gamma: float) -> Instance:
    """``inst`` with the block of rung ``gamma`` as its initial stock (a fresh memo; ``Instance.at_rung``)."""
    start = dataclasses.replace(inst.initial_state, stock=inst.rung_stock[gamma])
    return dataclasses.replace(inst, initial_state=start, start_rung=gamma)


def _family_digest(inst: Instance) -> str:
    """``Instance.family_digest``: the content digest of ``inst`` at its file's own rung (``raw`` names it)."""
    own = inst.raw["initial_state"].get("rung") if inst.rung_stock else None
    return inst.content_digest if own is None else inst.at_rung(own).content_digest


_ORDINAL_FIELD = {
    "chokepoints": "chokepoint_ordinal",
    "grids": "grid_ordinal",
    "fabs": "fab_ordinal",
    "osats": "osat_ordinal",
}

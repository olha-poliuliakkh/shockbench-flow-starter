"""`chokepoint-small` and `chokepoint-full` built from spec §4.6-4.7 (design §2.1-2.3, §2.2; Q83, Q95; §12 M5 rows).

One table set describes `full`: spec §4.6's node rows (fictional firms at the spec's sites and classes, D5: ids name a
site and a class, never a company), region-pair routing rules, the compatibility, use and fab -> OSAT tables, planned
flow shares and scales, each scale with its tag. `small` is a named subset of the same tables (``SIZES``). The builder
turns the tables into a networkx ``MultiDiGraph`` (``build_graph``: parallel arcs are distinct keys, the edge ids),
then derives the instance dict of design §2.3 from it (``derive_instance_dict``): the reset-time flow (22) and the
chokepoint traffic mu_cb at a fixed point, d-bar = upsilon F-bar (22), pi by the Q85 rule, nu by the c^min rule of §11
row 1 (Q93), h = v r / 52, F^b (proposal K3), and the warm initial state of §2.3 (``warm_initial_state``). Every leaf
gets a provenance entry (V20): an explicit tag where the value is sourced or derived, else SYNTHETIC(placeholder), so
V23 refuses to seal a split of either size until the placeholders are replaced (§11 row 61).

The sourced scales (design §12 'M5 stream "instances"' rows) come from the committed M5 evidence, each constant naming
its script (``EVIDENCE``): the crude heat content (EIA MER), the terminals' LNG nameplates and the gas pipelines (GEM,
Q104), the fab capacities and the fab -> OSAT shares (SIA/BCG, the ATP shares by the owner's M5-O24 (b)) and the energy
source shares (BACI). mu_cb stays the instance's own traffic, the modelled share of the strait's volume (EIA,
PortWatch), which its provenance records (option 1, owner queue M5-O23). The chokepoint adjacency is the owner's M5-O8
(b) with the Turkish Straits weather-only (M5-O25 (b)); the turn-backs follow ``TURNBACK_CHOKEPOINTS`` (M5-O22's
default).

The warm start (§2.3) is built on naive's plan with the F_Q quantiles of each γ rung (design §12 'Warm start per rung',
owner queue M5-O37 (b)): ``data/warm_fq/<size>_g<rung>.json`` holds a rung's, computed on the point-mass build by
``rung_fq`` and checked against it by ``check_warm_fq``; without any file the builder takes the point mass (one start
for every rung). The file keeps one stock block per rung (``initial_state.stock`` the anchor rung's, named by
``initial_state.rung``, and ``initial_state.stock_by_rung`` the others'), one pipeline, WIP and queue block, and storage
caps sized on the largest block; ``Instance.at_rung`` puts a rung's block in the initial state. F_Q reads naive's plan,
mu and the reset pipeline and queues, never the warm stock, so one pass reaches the fixed point at every rung, which the
regeneration command checks by recomputing each rung's F_Q on the file it wrote. Each block is naive's event-free steady
state at its rung: (67) puts no closure term on a route into an OSAT or a fab, whose processing (12), (19) would take it
in week 1 (owner queue M5-O33 (b)).

Every grid buffers its gas by (16) with the days of cover D_g; every grid with a nuclear share also buffers its
nuclear fuel, 364 days of its requirement zeta G-bar^0 (``NUCLEAR_COVER_DAYS``; design §3.5 'Nuclear fuel cover',
the owner's answer of 2026-09-28), a buffer in naive's target (§8.1 step 3) and so in the warm stock; gas stays the
only rationed fuel (15).

The builder makes no random draw (stream 8, instance kind by seed index, stays reserved; naive's replications of the
F_Q file are stream 22's, drawn by the regeneration command, never by a rebuild). The committed ``data/small.json`` and
``data/full.json`` are the artefacts; the package never rebuilds them at runtime. Values that come from HiGHS (the
nominal flow and everything derived from it) may differ in their last bits across machines (Q94), so a rebuild
elsewhere is compared within a relative tolerance, never by hash.

Regenerate a file with ``uv run python -m shockbench_flow.instance.flagship small`` (or ``full``), which reads the
committed F_Q files; ``--warm-fq`` first recomputes them (1,000 replications per the design, ``--warm-fq=R`` for R,
``--rungs=g,...`` only those rungs, ``--n-jobs=N`` joblib workers) and checks the fixed point at every rung. The command
prints the instance hash. The readings this module implements are the design §12 rows marked "M5 reading".
"""

from __future__ import annotations

import json
import math
import platform
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from shockbench_flow.instance.io import DATA_DIR, leaf_paths, load_instance, placeholder_leaves, write_instance
from shockbench_flow.instance.nominal import nominal_flow
from shockbench_flow.instance.schema import SCHEMA_VERSION, SUPPLY_TYPES, UNMODELLED, Instance


if TYPE_CHECKING:  # networkx is imported where the builder uses it, so loading the package never imports it (Q95)
    import networkx as nx


def _nx():
    """The networkx module, imported on first use by the builder and the structure helpers."""
    import networkx

    return networkx


PLACEHOLDER = "SYNTHETIC(placeholder)"
M5_SOURCE = "design §12 M5 readings; §11 row 61"


@dataclass(frozen=True)
class ScaleValue:
    """A number with its provenance (tag, source, lo, hi), as the instance file records it (V20)."""

    value: float
    tag: str = PLACEHOLDER
    source: str = M5_SOURCE
    lo: float | None = None
    hi: float | None = None


def _ph(value: float, source: str = M5_SOURCE) -> ScaleValue:
    """A SYNTHETIC(placeholder) value: a runnable default no source supports yet (§2.3 "Placeholders")."""
    return ScaleValue(float(value), PLACEHOLDER, source)


@dataclass(frozen=True)
class NodeRow:
    """One node of spec §4.6: id, type, region (spec §4.1), fab class, the commodities a supply node offers."""

    id: str
    type: str
    region: str
    cls: str | None = None
    commodities: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class SizeSpec:
    """A size of spec §4.8: its kind, instance id, horizon and the node ids it keeps (None: every node)."""

    kind: str
    instance_id: str
    T: int
    keep: frozenset[str] | None


@dataclass(frozen=True)
class FlagshipTables:
    """The one table set of `full` (spec §4.6-4.7); `small` keeps a subset of its nodes (``SizeSpec.keep``).

    Energy plan: per grid region and fuel, (source, share, how) with how "sea" (a sea connection to the region's
    terminal), "domestic" (a pipeline to the terminal) or "pipeline" (straight to the grid). Use: material -> fabs.
    Fab -> OSAT: fab -> its 2-3 OSATs, its output split by their regions' ATP capacity (``osat_atp``, M5-O24 (b)).
    Shares split planned flows (BACI's for the fuels, design §12 'Energy source shares (BACI)'; SIA/BCG's ATP for the
    fabs' output, §12 'Fab -> OSAT shares (M5-O24 (b))'; placeholders elsewhere), from which capacities follow by
    headroom (§12 "Scales and planned flows").
    """

    regions: tuple[str, ...]
    nodes: tuple[NodeRow, ...]
    commodities: tuple[tuple[str, str, bool], ...]  # id, pool, override (K^ov)
    values: Mapping[str, ScaleValue]  # v_k, USD per unit
    energy_plan: Mapping[str, Mapping[str, tuple[tuple[str, float, str], ...]]]
    grid_shares: Mapping[str, Mapping[str, ScaleValue]]  # region -> fuel -> zeta_gk
    grid_deliverable: Mapping[str, ScaleValue]  # region -> G-bar^0_g, GWh/wk
    days_cover: Mapping[str, Mapping[str, ScaleValue]]  # region -> buffered fuel -> D_gk of (16), days
    fab_cap0: Mapping[str, ScaleValue]  # fab -> cap^0_f, wafer-eq/wk
    use: tuple[tuple[str, tuple[str, ...]], ...]  # material -> fabs using it
    fab_osat: tuple[tuple[str, tuple[str, ...]], ...]  # fab -> OSATs
    osat_atp: Mapping[str, ScaleValue]  # OSAT -> the ATP capacity of its region, wafer-eq/wk (M5-O24 (b))
    sink_share: Mapping[str, ScaleValue]  # sink -> share of packaged chips
    prohibited: tuple[tuple[str, str, str], ...]  # (tail, head, commodity) links in Z_0 from week 0 (spec §6.3)
    chokepoint_adjacency: Mapping[str, tuple[str, ...]]
    chokepoint_class: Mapping[str, str]


# ----- vocabularies and nodes (spec §4.1, §4.6; design §2.1-2.2) ---------------------------------------------------
REGIONS = ("TW", "KR", "JP", "CN", "US", "EU", "GULF", "RU", "AU", "SEA", "IN", "UA", "KZ", "ROW")  # spec §4.1
COMMODITIES = (  # design §2.1: eight commodities on small and full (Q58, M6); override = K^ov (Q34)
    ("lng", "tb", True),
    ("crude", "tb", True),
    ("nucfuel", "tb", False),
    ("wafer", "ct", False),
    ("chip_le_raw", "ct", False),
    ("chip_mat_raw", "ct", False),
    ("chip_le", "ct", False),
    ("chip_mat", "ct", False),
)
POOL_OF = {k: b for k, b, _ in COMMODITIES}
OVERRIDE = {k: ov for k, _, ov in COMMODITIES}  # K^ov, the cargo a turn-back can carry (Q34)
FUELS = ("lng", "crude", "nucfuel")
PACKAGED = {"chip_le_raw": "chip_le", "chip_mat_raw": "chip_mat"}  # (19)
PRODUCT = {"leading": "chip_le_raw", "memory": "chip_le_raw", "mature": "chip_mat_raw"}  # §2.1
GRID_REGIONS = ("TW", "KR", "JP", "CN", "US", "EU", "SEA", "IN")  # spec §4.6: one terminal and one grid each

# The M5 evidence behind the sourced constants below; each script's docstring holds its command (design §12 'M5 stream
# "instances"' rows). Provenance sources name the script, so every sourced leaf carries its command (V20).
EVIDENCE = {
    "eia": "scripts/python/evidence/m5_eia_chokepoints.py",
    "gem": "scripts/python/evidence/m5_lng_terminals.py",
    "sia_bcg": "scripts/python/evidence/m5_fab_capacity.py",
    "baci": "scripts/python/evidence/m5_baci_flows.py",
    "portwatch": "scripts/python/evidence/m5_portwatch_chokepoints.py",
}

RATE = 0.0421  # r, Fed funds 2025 mean, REAL (W2-maritime §2.1)
MMBTU_PER_GWH = 3412.14  # unit definition
LNG_USD_MMBTU = 12.09  # IMF PNGASJPUSDM 2025 mean, via W2-maritime §2.1
BRENT_USD_BBL = 68.32  # IMF POILBREUSDM 2025 mean, via W2-maritime §2.1
MMBTU_PER_BBL = 6.062  # crude imports 2025 (COIMKUS), REAL(EIA MER Table A2) (§11 row 62; EVIDENCE["eia"])
MMBTU_PER_BBL_RANGE = (5.689, 6.085)  # COPRKUS 2025 (production) to COIMKUS 2022, the same table (§12 M5 instances)
V_LNG = LNG_USD_MMBTU * MMBTU_PER_GWH  # 41,253 USD/GWh, as tiny
V_CRUDE = BRENT_USD_BBL / MMBTU_PER_BBL * MMBTU_PER_GWH  # 38,455.53 USD/GWh, DERIVED(IMF, EIA MER A2)
V_CRUDE_RANGE = tuple(BRENT_USD_BBL / h * MMBTU_PER_GWH for h in reversed(MMBTU_PER_BBL_RANGE))
HQ_RATIO = {"lng": (8.2, 60.0, 215.0), "crude": (4.7, 17.0, 52.0)}  # h^Q/h by war-risk class, DERIVED(W2-maritime)
WR_LNG = (0.0, 1723.0, 6893.0)  # per-transit war risk, USD/GWh, DERIVED(W2-maritime), as tiny
WR_CRUDE_BBL = (0.0, 0.656, 2.63)  # USD/bbl, DERIVED(W2-maritime); per GWh at EIA's heat content
WR_CRUDE = tuple(x / MMBTU_PER_BBL * MMBTU_PER_GWH for x in WR_CRUDE_BBL)
WR_CONTAINER = (0.0, 0.5, 2.0)  # USD/wafer-eq, SYNTHETIC(placeholder), tiny's (§11 row 65)
TOP_TARIFF = 0.25  # the spec's chip_tariff_25 (§11 row 25)
UPSILON = 0.9  # utilisation (22), SYNTHETIC (Q76 owner D2; §11 row 60)
K_C = 1.3  # clearance ratio, DERIVED(PortWatch) at Suez 2021 (Q7), transferred to the other chokepoints

# freight and lead times (design §12 "Routing and lead times"): USD per unit and week of transit, tiny's scale
FREIGHT_PER_WEEK = {"tb": 500.0, "ct": 1.0}  # SYNTHETIC(placeholder)
PIPE_TAU = 1  # gas and domestic pipelines, weeks, SYNTHETIC(placeholder)
NUC_TAU = 8  # nucfuel to grids, weeks (spec §4.3 "long lead time"), SYNTHETIC(placeholder)
TG_TAU, TG_C0 = 0, 0.0  # terminal -> grid: lead 0 (spec §4.7), no freight (SYNTHETIC(placeholder))
AIR_COST, AIR_CAP, AIR_TAU = 4.0, 0.25, 1  # spec §4.7 material air duplicate, SYNTHETIC(prior)
CAPE_COST = 1.4  # spec §4.4, SYNTHETIC(prior); +2 weeks by the legs of cape_tau
DETOUR = {"chk_malacca": ("lombok", 1, 1.1), "chk_taiwan": ("east", 1, 1.1)}  # kind, weeks, freight: placeholders
BYPASS_SHARE = ScaleValue(0.124, "REAL(EIA 2025)", "unused bypass 2.6 of 20.9 mb/d through Hormuz (§11 row 63)")
BYPASS_RANGE = (0.124, 0.225)  # REAL(EIA 2025, 2026), §11 row 63: one point per split
BYPASS_COST = 1.2  # freight of the bypass over the replaced segment, SYNTHETIC(placeholder)
DUP_OWN_SHARE = 0.25  # own pre-crisis capacity of a sea duplicate over the replaced route's (tiny's 4 of 16)
ALPHA_SP = {"tb": 0.8, "ct": 2.3}  # spare capacity, DERIVED(PortWatch) at the Cape (§11 row 8), reading (A)
TURNBACK_DTAU = 2  # turn-back to the Cape, weeks, as tiny's E4 (Q4)
# M5-O22 (owner queue 2026-09-28), default (a): turn-backs at Suez/Red Sea only (spec; §11 row 41). Answer (b) adds
# "chk_hormuz", "chk_malacca" and "chk_taiwan" here, at the same lead +2 and 1.4x freight (§12 'Turn-backs (M5-O22)').
TURNBACK_CHOKEPOINTS = ("chk_suez",)
H_ROUTE, H_SUPPLY, H_THR = 1.2, 1.25, 1.25  # headroom over planned flows, SYNTHETIC(placeholder)
PROHIBITED_SHARE = 0.25  # capacity of a link in Z_0 over its head's requirement, SYNTHETIC(placeholder)
H_SINK = 1.0  # OSAT -> sink capacity over the planned flow: 1 keeps every sink's share in (22) (§12 M5)
STORAGE_WEEKS = 4.0  # storage caps: weeks of a slot's steady throughput, at least twice its warm stock
MU_UNBOUNDED = 1.0e9  # mu of the first fixed-point solve: throughput rows that never bind (§12 "Scales")
FIXED_POINT_RTOL, FIXED_POINT_ITERATIONS = 1e-9, 8

NODES = (
    # sources (spec §4.6 row 1)
    NodeRow("src_qa_lng", "source", "GULF", commodities=("lng",)),
    NodeRow("src_gulf_crude", "source", "GULF", commodities=("crude",)),
    NodeRow("src_us_lng", "source", "US", commodities=("lng",)),
    NodeRow("src_us_crude", "source", "US", commodities=("crude",)),
    NodeRow("src_au_lng", "source", "AU", commodities=("lng",)),
    NodeRow("src_ru_gas", "source", "RU", commodities=("lng",), note="pipeline gas and sea LNG, one commodity (GWh)"),
    NodeRow("src_ru_crude", "source", "RU", commodities=("crude",)),
    NodeRow("src_no_gas", "source", "EU", commodities=("lng",), note="Norway, region EU (§12 'Energy paths')"),
    NodeRow("src_kz_uranium", "source", "KZ", commodities=("nucfuel",)),
    NodeRow("src_ru_enrichment", "source", "RU", commodities=("nucfuel",)),
    # chokepoints (row 2)
    NodeRow("chk_hormuz", "chokepoint", "GULF"),
    NodeRow("chk_malacca", "chokepoint", "SEA"),
    NodeRow("chk_suez", "chokepoint", "ROW", note="Suez / Red Sea, Bab el-Mandeb included"),
    NodeRow("chk_cape", "chokepoint", "ROW"),
    NodeRow("chk_taiwan", "chokepoint", "TW"),
    NodeRow("chk_panama", "chokepoint", "ROW"),
    NodeRow("chk_turkish", "chokepoint", "ROW"),
    # terminals and grids (row 3)
    *(NodeRow(f"term_{r.lower()}", "terminal", r) for r in GRID_REGIONS),
    *(NodeRow(f"grid_{r.lower()}", "grid", r) for r in GRID_REGIONS),
    # materials (row 4): all make the aggregated `wafer` (§2.1)
    NodeRow("mat_jp_wafer", "material", "JP", commodities=("wafer",)),
    NodeRow("mat_jp_resist", "material", "JP", commodities=("wafer",)),
    NodeRow("mat_kr_wafer", "material", "KR", commodities=("wafer",)),
    NodeRow("mat_de_wafer", "material", "EU", commodities=("wafer",)),
    NodeRow("mat_ua_neon", "material", "UA", commodities=("wafer",)),
    NodeRow("mat_cn_neon", "material", "CN", commodities=("wafer",)),
    NodeRow("mat_helium", "material", "GULF", commodities=("wafer",), note="Qatar/US helium, one node in GULF"),
    NodeRow("mat_cn_gage", "material", "CN", commodities=("wafer",)),
    # fabs (row 5): fictional firms at the spec's sites and classes, in spec order (D5)
    NodeRow("fab_tw_leading_1", "fab", "TW", "leading", note="spec site 1"),
    NodeRow("fab_tw_mature_1", "fab", "TW", "mature", note="spec site 2"),
    NodeRow("fab_us_leading_1", "fab", "US", "leading", note="spec site 3"),
    NodeRow("fab_kr_leading_1", "fab", "KR", "leading", note="spec site 4"),
    NodeRow("fab_kr_memory_1", "fab", "KR", "memory", note="spec site 5"),
    NodeRow("fab_us_leading_2", "fab", "US", "leading", note="spec site 6"),
    NodeRow("fab_jp_memory_1", "fab", "JP", "memory", note="spec site 7"),
    NodeRow("fab_us_leading_3", "fab", "US", "leading", note="spec site 8"),
    NodeRow("fab_eu_leading_1", "fab", "EU", "leading", note="spec site 9"),
    NodeRow("fab_row_leading_1", "fab", "ROW", "leading", note="spec site 10, on the EU grid (§12 'Energy paths')"),
    NodeRow("fab_cn_mature_1", "fab", "CN", "mature", note="spec site 11"),
    NodeRow("fab_us_mature_1", "fab", "US", "mature", note="spec site 12"),
    NodeRow("fab_eu_mature_1", "fab", "EU", "mature", note="spec site 13"),
    NodeRow("fab_sea_mature_1", "fab", "SEA", "mature", note="spec site 14"),
    NodeRow("fab_tw_mature_2", "fab", "TW", "mature", note="spec site 15"),
    NodeRow("fab_us_mature_2", "fab", "US", "mature", note="spec site 16"),
    # OSATs (row 6)
    NodeRow("osat_my", "osat", "SEA"),
    NodeRow("osat_vn", "osat", "SEA"),
    NodeRow("osat_ph", "osat", "SEA"),
    NodeRow("osat_cn", "osat", "CN"),
    NodeRow("osat_tw", "osat", "TW"),
    NodeRow("osat_kr", "osat", "KR"),
    NodeRow("osat_sg", "osat", "SEA"),
    # sinks (row 7): eight regional sinks with a demand map (§12 'Sinks and standing prohibitions')
    NodeRow("sink_us", "sink", "US"),
    NodeRow("sink_eu", "sink", "EU"),
    NodeRow("sink_cn", "sink", "CN"),
    NodeRow("sink_jp", "sink", "JP"),
    NodeRow("sink_kr", "sink", "KR"),
    NodeRow("sink_sea", "sink", "SEA"),
    NodeRow("sink_in", "sink", "IN"),
    NodeRow("sink_row", "sink", "ROW"),
)

VALUES = {
    "lng": ScaleValue(V_LNG, "DERIVED(IMF PNGASJPUSDM 2025 via W2-maritime)", "12.09 $/MMBtu x 3,412.14 MMBtu/GWh"),
    "crude": ScaleValue(
        V_CRUDE,
        "DERIVED(IMF, EIA MER A2)",
        f"Brent 68.32 $/bbl REAL(IMF via W2-maritime) at 6.062 MMBtu/bbl REAL(EIA MER Table A2, COIMKUS 2025); lo and"
        f" hi at 6.085 and 5.689 (§11 row 62; {EVIDENCE['eia']})",
        *V_CRUDE_RANGE,
    ),
    "nucfuel": _ph(5.0 * V_LNG, "spec §4.3 relative value 5.0 x lng (§11 row 62)"),
    "wafer": _ph(3000.0, "tiny's 3,000 (§11 row 62)"),
    "chip_le_raw": _ph(20000.0, "spec §4.3 wafer:chip_le 3:20, tiny's 20,000 (§11 row 62)"),
    "chip_mat_raw": _ph(4000.0, "spec §4.3 wafer:chip_mat 3:4 (§11 row 62)"),
    "chip_le": _ph(20000.0, "spec §4.3 wafer:chip_le 3:20, tiny's 20,000 (§11 row 62)"),
    "chip_mat": _ph(4000.0, "spec §4.3 wafer:chip_mat 3:4 (§11 row 62)"),
}

# grid energy (design §12 "Scales and planned flows"): TW REAL, the rest placeholders of the right order of magnitude
TW_GWH_2025 = 283009.0  # Taiwan electricity consumption 2025, REAL(MOEA via W2-industrial §2.3)
GRID_DELIVERABLE = {
    "TW": ScaleValue(TW_GWH_2025 / 52, "DERIVED(MOEA via W2-industrial)", "283,009 GWh in 2025 / 52 (§2.3 W2)"),
    "KR": _ph(11000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "JP": _ph(18000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "CN": _ph(180000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "US": _ph(80000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "EU": _ph(52000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "SEA": _ph(25000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
    "IN": _ph(36000.0, "order of magnitude only; primary statistics per grid (§11 row 14)"),
}
_ZETA = {  # generation shares zeta_gk (lng, crude, nucfuel), SYNTHETIC(placeholder) (§11 row 15); TW has no nuclear
    "TW": (0.40, 0.02, 0.0),
    "KR": (0.25, 0.02, 0.30),
    "JP": (0.30, 0.05, 0.08),
    "CN": (0.03, 0.01, 0.05),
    "US": (0.40, 0.01, 0.18),
    "EU": (0.17, 0.02, 0.23),
    "SEA": (0.30, 0.03, 0.0),
    "IN": (0.03, 0.01, 0.03),
}
GRID_SHARES = {
    r: {k: _ph(z, "generation share, none sourced (§11 row 15)") for k, z in zip(FUELS, zs) if z > 0}
    for r, zs in _ZETA.items()
}
DAYS_COVER = {
    "TW": ScaleValue(11.0, "REAL", "Taiwan legal floor 11 d, 2025-26 (W2-industrial §2.2)"),
    "KR": ScaleValue(10.0, "REAL", "mandatory 7 + preventive 3 d (W2-industrial §2.2)"),
    "JP": ScaleValue(14.0, "REAL", "about two weeks held by utilities (W2-industrial §2.2)"),
    **{r: _ph(14.0, "none sourced; tiny's EU 14 (§11 row 12)") for r in ("CN", "US", "EU", "SEA", "IN")},
}
# the nuclear fuel cover (design §3.5 'Nuclear fuel cover', §11 row 12; the owner's answer of 2026-09-28): D of (16)
# for nucfuel at every grid with a nuclear share, 52 weeks of its requirement, inside the refuelling intervals
NUCLEAR_COVER_DAYS = ScaleValue(
    364.0,
    "SYNTHETIC(prior: reactor refuelling intervals of 12-24 months)",
    "52 weeks of the grid's nuclear fuel requirement, the owner's cover (2026-09-28): reactors refuel at intervals"
    " of 12, 18 or 24 months with the loaded core and the next reload on site (World Nuclear Association, Nuclear Power"
    " Reactors, updated 2026-08-05; EIA Today in Energy 2018-10-15: 18 to 24 months); lo and hi the intervals"
    " (§3.5, §11 row 12; scripts/python/evidence/m5_nuclear_cover.py)",
    lo=364.0,
    hi=728.0,
)
BUFFER_DAYS = {  # region -> buffered fuel -> D_gk (16): gas at every grid, nuclear fuel where the grid burns it
    r: {"lng": DAYS_COVER[r], **({"nucfuel": NUCLEAR_COVER_DAYS} if zs[FUELS.index("nucfuel")] > 0 else {})}
    for r, zs in _ZETA.items()
}
# fab capacity (design §12 'Fab capacities (SIA/BCG)'): 2022 capacity per region and fab class, wafer-eq/wk, the points
# of EVIDENCE["sia_bcg"] §3 (global point x NNLS point), DERIVED(SIA/BCG 2024), as that file prints its rows
SIA_BCG_REGIONS = ("US", "EU", "JP", "KR", "TW", "CN", "Other")
SIA_BCG_ROWS = {
    "leading (logic <10 nm)": (0.0, 0.0, 0.0, 28128.0, 62608.0, 0.0, 0.0),
    "mature (logic 10-22 nm, 28 nm+, DAO)": (231144.0, 204776.0, 291054.0, 102990.0, 311148.0, 448029.0, 154101.0),
    "memory (DRAM, NAND)": (22623.0, 0.0, 143894.0, 304995.0, 87445.0, 167430.0, 27728.0),
    "alternative leading (logic <10 and 10-22 nm)": (52957.0, 24587.0, 0.0, 35693.0, 138261.0, 11348.0, 15131.0),
}
SIA_BCG_REGION = {
    "US": "US",
    "EU": "EU",
    "JP": "JP",
    "KR": "KR",
    "TW": "TW",
    "CN": "CN",
    "ROW": "Other",
    "SEA": "Other",
}


def sia_bcg_cells() -> dict[tuple[str, str], float]:
    """(SIA/BCG region, instance fab class) -> 2022 capacity, wafer-eq/wk (design §12 'Fab capacities (SIA/BCG)').

    Leading is logic below 22 nm (the file's "alternative leading" row), mature logic of 28 nm and above and DAO (its
    mature row less the 10-22 nm part), memory DRAM and NAND. DERIVED(SIA/BCG 2024), from ``EVIDENCE["sia_bcg"]``.
    """
    rows = {name.split(" ", 1)[0]: dict(zip(SIA_BCG_REGIONS, values)) for name, values in SIA_BCG_ROWS.items()}
    out = {}
    for r in SIA_BCG_REGIONS:
        lead, alt = rows["leading"][r], rows["alternative"][r]
        out[(r, "leading")] = alt
        out[(r, "mature")] = rows["mature"][r] - (alt - lead)
        out[(r, "memory")] = rows["memory"][r]
    return out


def _fab_cap0() -> dict[str, ScaleValue]:
    """cap^0_f: the fab's SIA/BCG cell over the number of `full`'s fabs in that cell (`small` keeps these values)."""
    cells = sia_bcg_cells()
    fabs = [r for r in NODES if r.type == "fab"]
    cell_of = {r.id: (SIA_BCG_REGION[r.region], r.cls) for r in fabs}
    n = defaultdict(int)
    for cell in cell_of.values():
        n[cell] += 1
    out = {}
    for f, cell in cell_of.items():
        region, cls = cell
        source = (
            f"SIA/BCG 2022 {region} {cls} capacity {cells[cell]:,.0f} wafer-eq/wk over {n[cell]} fab(s) of full"
            f" ({EVIDENCE['sia_bcg']} §3; leading = logic below 22 nm, §12 'Fab capacities (SIA/BCG)')"
        )
        out[f] = ScaleValue(cells[cell] / n[cell], "DERIVED(SIA/BCG 2024)", source)
    return out


FAB_CAP0 = _fab_cap0()  # wafer-eq/wk
E_F = {  # GWh per wafer (W2-industrial §2.3)
    "leading": ScaleValue(0.002, "SYNTHETIC(prior: W2-industrial §2.3)", "2,000 kWh/wafer [1,650, 2,600]"),
    "mature": ScaleValue(0.0009, "DERIVED(W2-industrial §2.3, median of three)", "900 kWh per 300 mm-eq [690, 990]"),
    "memory_kr": ScaleValue(0.00126, "DERIVED(W2-industrial §2.3, DRAM)", "1,260 kWh/wafer"),
    "memory_jp": _ph(0.00126, "NAND e_f not sourced (§11 row 13 ‡); the DRAM value transferred"),
}
TAU_FAB = {  # weeks: spec §4.5 for leading and mature; memory a placeholder (§12 'Memory fabs')
    "leading": ScaleValue(8.0, "SYNTHETIC(prior: spec §4.5)", "leading-edge lag 8 weeks"),
    "mature": ScaleValue(6.0, "SYNTHETIC(prior: spec §4.5)", "mature lag 6 weeks"),
    "memory": _ph(8.0, "no spec value for memory; as leading-edge"),
}
TAU_OSAT = 2  # spec §4.5
W_SCR = 2  # scrap window, SYNTHETIC(placeholder) (§11 row 49)

# BACI 2022-2024 mean bilateral flows between the plan's sources and regions, GWh/wk, DERIVED(BACI, EIA): gas at the
# 13-exporter median 52.2 MMBtu/t, crude at EIA's energy per tonne (EVIDENCE["baci"] §2; src_ru_gas's sea LNG is the
# file's "Russia LNG" row, the Gulf's crude includes Oman's). Keyed (source, region, how) as ENERGY_PLAN routes them.
BACI_GWH_WK = {
    ("src_qa_lng", "TW", "sea"): 776.0,
    ("src_us_lng", "TW", "sea"): 291.0,
    ("src_us_crude", "TW", "sea"): 2933.0,
    ("src_qa_lng", "KR", "sea"): 2658.0,
    ("src_au_lng", "KR", "sea"): 3269.0,
    ("src_gulf_crude", "KR", "sea"): 23375.0,
    ("src_au_lng", "JP", "sea"): 8146.0,
    ("src_us_lng", "JP", "sea"): 1537.0,
    ("src_ru_gas", "JP", "sea"): 1825.0,
    ("src_gulf_crude", "JP", "sea"): 28311.0,
    ("src_ru_gas", "CN", "pipeline"): 2780.0,
    ("src_ru_gas", "CN", "sea"): 2230.0,
    ("src_au_lng", "CN", "sea"): 6994.0,
    ("src_gulf_crude", "CN", "sea"): 62012.0,
    ("src_ru_crude", "CN", "sea"): 24062.0,
    ("src_no_gas", "EU", "pipeline"): 21638.0,
    ("src_us_lng", "EU", "sea"): 9308.0,
    ("src_qa_lng", "EU", "sea"): 3128.0,
    ("src_us_crude", "EU", "sea"): 17250.0,
    ("src_au_lng", "SEA", "sea"): 2040.0,
    ("src_us_crude", "SEA", "sea"): 4384.0,
    ("src_qa_lng", "IN", "sea"): 3211.0,
    ("src_gulf_crude", "IN", "sea"): 28599.0,
    ("src_ru_crude", "IN", "sea"): 17821.0,
}
_NUC = (("src_kz_uranium", 0.5, "pipeline"), ("src_ru_enrichment", 0.5, "pipeline"))  # SYNTHETIC(placeholder)


def _baci(region: str, *entries: tuple[str, str]) -> tuple[tuple[str, float, str], ...]:
    """(source, planned share, how) over the plan's sources of one region and fuel, shares by BACI's flows (§12)."""
    total = math.fsum(BACI_GWH_WK[(src, region, how)] for src, how in entries)
    return tuple((src, BACI_GWH_WK[(src, region, how)] / total, how) for src, how in entries)


# region -> fuel -> (source, planned share, how): the source sets are the builder's (spec §4.7's compatibility
# table), the shares BACI's (DERIVED(BACI); design §12 'Energy source shares (BACI)'), nuclear fuel's placeholders
# (_NUC); how is "sea" (to the region's terminal), "domestic" (a pipeline to the terminal) or "pipeline" (straight to
# the grid); US gas goes straight to the grid (§12 'Terminal throughput (Q104)')
ENERGY_PLAN = {
    "TW": {"lng": _baci("TW", ("src_qa_lng", "sea"), ("src_us_lng", "sea")), "crude": (("src_us_crude", 1.0, "sea"),)},
    "KR": {
        "lng": _baci("KR", ("src_qa_lng", "sea"), ("src_au_lng", "sea")),
        "crude": (("src_gulf_crude", 1.0, "sea"),),
        "nucfuel": _NUC,
    },
    "JP": {
        "lng": _baci("JP", ("src_au_lng", "sea"), ("src_us_lng", "sea"), ("src_ru_gas", "sea")),
        "crude": (("src_gulf_crude", 1.0, "sea"),),
        "nucfuel": _NUC,
    },
    "CN": {
        "lng": _baci("CN", ("src_ru_gas", "pipeline"), ("src_ru_gas", "sea"), ("src_au_lng", "sea")),
        "crude": _baci("CN", ("src_gulf_crude", "sea"), ("src_ru_crude", "sea")),
        "nucfuel": _NUC,
    },
    "US": {
        "lng": (("src_us_lng", 1.0, "pipeline"),),
        "crude": (("src_us_crude", 1.0, "domestic"),),
        "nucfuel": _NUC,
    },
    "EU": {
        "lng": _baci("EU", ("src_no_gas", "pipeline"), ("src_us_lng", "sea"), ("src_qa_lng", "sea")),
        "crude": (("src_us_crude", 1.0, "sea"),),
        "nucfuel": _NUC,
    },
    "SEA": {"lng": (("src_au_lng", 1.0, "sea"),), "crude": (("src_us_crude", 1.0, "sea"),)},
    "IN": {
        "lng": (("src_qa_lng", 1.0, "sea"),),
        "crude": _baci("IN", ("src_gulf_crude", "sea"), ("src_ru_crude", "sea")),
        "nucfuel": _NUC,
    },
}
# Q104: GEM Global Gas Infrastructure Tracker, operating capacity in GWh/wk at MER's 2025 dry-gas heat content, point
# with lo and hi at its bracketing contents (EVIDENCE["gem"] §1 and §3), DERIVED(GEM GGIT)
GEM_LNG_IMPORT = {  # a terminal region's LNG regasification nameplate: the LNG part of the terminal's throughput
    "TW": (6009.0, 5852.0, 6304.0),
    "KR": (43334.0, 42205.0, 45462.0),
    "JP": (67702.0, 65937.0, 71026.0),
    "EU": (50892.0, 49566.0, 53391.0),
    "CN": (43749.0, 42609.0, 45897.0),
    "US": (20773.0, 20232.0, 21793.0),
    "SEA": (17625.0, 17166.0, 18490.0),
    "IN": (14730.0, 14346.0, 15453.0),
}
GEM_PIPELINE = {  # (source, grid) -> u^0 of the gas pipeline
    ("src_ru_gas", "grid_eu"): (3791.0, 3693.0, 3977.0),  # operating routes only; in Z_0 from week 0
    ("src_no_gas", "grid_eu"): (19470.0, 18963.0, 20426.0),  # with the Baltic Pipe tie-in
    ("src_ru_gas", "grid_cn"): (7809.0, 7605.0, 8192.0),  # Power of Siberia phase I
}
GEM_TAG = "DERIVED(GEM GGIT)"
TERMINAL_TAG = "SYNTHETIC(placeholder: the crude part; the LNG part DERIVED(GEM GGIT))"
PROHIBITED = (  # spec §4.7 and §6.3: standing in Z_0 from week 0 (§12 'Sinks and standing prohibitions')
    ("src_ru_gas", "grid_eu", "lng"),
    ("src_ru_crude", "term_eu", "crude"),
)
USE = (  # material -> fabs using it (spec §4.7 "use table"), SYNTHETIC(placeholder)
    (
        "mat_jp_wafer",
        (
            "fab_tw_leading_1",
            "fab_tw_mature_1",
            "fab_tw_mature_2",
            "fab_kr_leading_1",
            "fab_kr_memory_1",
            "fab_jp_memory_1",
            "fab_cn_mature_1",
            "fab_us_leading_1",
            "fab_us_leading_3",
        ),
    ),
    (
        "mat_jp_resist",
        (
            "fab_tw_leading_1",
            "fab_kr_leading_1",
            "fab_kr_memory_1",
            "fab_jp_memory_1",
            "fab_us_leading_1",
            "fab_us_leading_2",
            "fab_us_leading_3",
            "fab_eu_leading_1",
            "fab_row_leading_1",
        ),
    ),
    ("mat_kr_wafer", ("fab_kr_leading_1", "fab_kr_memory_1", "fab_us_leading_2", "fab_cn_mature_1")),
    (
        "mat_de_wafer",
        (
            "fab_eu_leading_1",
            "fab_eu_mature_1",
            "fab_row_leading_1",
            "fab_us_mature_1",
            "fab_us_mature_2",
            "fab_sea_mature_1",
            "fab_tw_mature_2",
        ),
    ),
    ("mat_ua_neon", ("fab_eu_leading_1", "fab_eu_mature_1", "fab_tw_leading_1", "fab_kr_memory_1", "fab_us_mature_2")),
    ("mat_cn_neon", ("fab_cn_mature_1", "fab_tw_mature_1", "fab_kr_leading_1", "fab_sea_mature_1")),
    ("mat_helium", ("fab_tw_leading_1", "fab_kr_leading_1", "fab_kr_memory_1", "fab_sea_mature_1", "fab_eu_leading_1")),
    ("mat_cn_gage", ("fab_cn_mature_1", "fab_tw_mature_1", "fab_tw_mature_2", "fab_us_mature_2", "fab_jp_memory_1")),
)
FAB_OSAT = (  # fab -> 2-3 OSATs (spec §4.7); no leading or memory fab ships to the CN OSAT (§12 'Sinks')
    ("fab_tw_leading_1", ("osat_tw", "osat_my", "osat_sg")),
    ("fab_tw_mature_1", ("osat_tw", "osat_cn", "osat_ph")),
    ("fab_us_leading_1", ("osat_tw", "osat_my")),
    ("fab_kr_leading_1", ("osat_kr", "osat_vn")),
    ("fab_kr_memory_1", ("osat_kr", "osat_vn", "osat_my")),
    ("fab_us_leading_2", ("osat_kr", "osat_vn")),
    ("fab_jp_memory_1", ("osat_ph", "osat_my")),
    ("fab_us_leading_3", ("osat_my", "osat_vn")),
    ("fab_eu_leading_1", ("osat_my", "osat_vn")),
    ("fab_row_leading_1", ("osat_my", "osat_vn")),
    ("fab_cn_mature_1", ("osat_cn", "osat_my")),
    ("fab_us_mature_1", ("osat_ph", "osat_sg")),
    ("fab_eu_mature_1", ("osat_my", "osat_sg")),
    ("fab_sea_mature_1", ("osat_sg", "osat_my")),
    ("fab_tw_mature_2", ("osat_tw", "osat_cn")),
    ("fab_us_mature_2", ("osat_ph", "osat_my", "osat_cn")),
)
# M5-O24 (b), the owner's answer of 2026-09-28: a fab's output splits over its OSATs of the size in proportion to
# SIA/BCG's 2022 ATP capacity of each OSAT's region, in place of equal shares (design §12 'Fab -> OSAT shares (M5-O24
# (b))'). The rows are EVIDENCE["sia_bcg"] §4's (Exhibit 13 share x the global capacity of its rule 1, packaged units
# per week at one per wafer), DERIVED(SIA/BCG 2024); Vietnam takes the exhibit's one band for Vietnam and Mexico
SIA_BCG_ATP = {
    "TW": 696079.0,
    "CN": 773421.0,
    "KR": 232026.0,
    "Malaysia": 180465.0,
    "Philippines": 154684.0,
    "Singapore": 51561.0,
    "Vietnam and Mexico": 25781.0,
}
OSAT_ATP_ROW = {  # OSAT -> its row of SIA_BCG_ATP
    "osat_my": "Malaysia",
    "osat_vn": "Vietnam and Mexico",
    "osat_ph": "Philippines",
    "osat_cn": "CN",
    "osat_tw": "TW",
    "osat_kr": "KR",
    "osat_sg": "Singapore",
}
OSAT_ATP = {
    o: ScaleValue(
        SIA_BCG_ATP[row],
        "DERIVED(SIA/BCG 2024)",
        f"SIA/BCG 2022 ATP capacity of {row}, {SIA_BCG_ATP[row]:,.0f} wafer-eq/wk ({EVIDENCE['sia_bcg']} §4; each"
        " fab's output splits over its OSATs of the size by it, M5-O24 (b))",
    )
    for o, row in OSAT_ATP_ROW.items()
}
SINK_SHARE = {  # share of packaged chips by sink, SYNTHETIC(placeholder); BACI HS 8542 imports to check (§2.3)
    s: _ph(v, "regional share of chip demand, none sourced (§2.3 table: BACI as a check)")
    for s, v in (
        ("sink_us", 0.25),
        ("sink_eu", 0.09),
        ("sink_cn", 0.30),
        ("sink_jp", 0.07),
        ("sink_kr", 0.05),
        ("sink_sea", 0.12),
        ("sink_in", 0.04),
        ("sink_row", 0.08),
    )
}
# M5-O8 (b), the owner's answer of 2026-09-28: the regions whose conflict layer and militarised events act on each
# chokepoint, SYNTHETIC(prior) (§11 row 56; design §12 'Chokepoint adjacency (M5-O8 (b))'); the Cape, Panama and, by
# M5-O25 (b) (§12 'Turkish Straits weather-only (M5-O25 (b))'), the Turkish Straits have none, so only weather closes
# them
CHOKEPOINT_ADJACENCY = {
    "chk_hormuz": ("GULF",),
    "chk_malacca": ("SEA",),
    "chk_suez": ("GULF",),  # Suez / Red Sea: §11 row 56's GULF alternative
    "chk_cape": (),
    "chk_taiwan": ("CN",),
    "chk_panama": (),
    "chk_turkish": (),  # M5-O25 (b); ROW before
}
# The strait's whole tanker volume, crude and LNG, GWh/wk, 2023 point: a diagnostic of the modelled share mu_cb / V_cb,
# never the scale (design §12 'Chokepoint traffic, the modelled share'); EIA's World Oil Transit Chokepoints at MER
# heat contents (EVIDENCE["eia"]), and for the Taiwan Strait PortWatch's tanker tonnage at EIA's energy per tonne,
# products included (EVIDENCE["portwatch"])
STRAIT_VOLUME_TB = {
    "chk_hormuz": (219086.0, "EIA Strait of Hormuz crude 196,557 + LNG 22,529", EVIDENCE["eia"]),
    "chk_malacca": (232707.0, "EIA Strait of Malacca crude 212,729 + LNG 19,978", EVIDENCE["eia"]),
    "chk_suez": (
        64695.0,
        "EIA Suez Canal and SUMED crude 55,981 + LNG 8,714 (Bab el-Mandeb 58,469 + 8,926)",
        EVIDENCE["eia"],
    ),
    "chk_cape": (67908.0, "EIA Cape of Good Hope crude 63,445 + LNG 4,463", EVIDENCE["eia"]),
    "chk_panama": (1244.0, "EIA Panama Canal crude 1,244 (LNG not tabled; below 638 in FY2025)", EVIDENCE["eia"]),
    "chk_turkish": (23637.0, "EIA Bosporus crude 23,637 (Dardanelles 23,637 + LNG 1,063)", EVIDENCE["eia"]),
    "chk_taiwan": (
        76889.0,
        "PortWatch Taiwan Strait tanker volume 5.899 Mt/wk x EIA's median 13.03 MWh/t [70,103, 97,648]",
        EVIDENCE["portwatch"],
    ),
}
CHOKEPOINT_CLASS = {
    "chk_hormuz": "hormuz",
    "chk_malacca": "malacca",
    "chk_suez": "suez_red_sea",
    "chk_cape": "cape",
    "chk_taiwan": "taiwan_strait",
    "chk_panama": "panama",
    "chk_turkish": "turkish_straits",
}

# ----- routing (design §12 "Routing and lead times") ----------------------------------------------------------------
TB_ROUTES = {  # tanker lanes, (source region, terminal region) -> chokepoints in path order
    ("GULF", "EU"): ("chk_hormuz", "chk_suez"),
    ("GULF", "IN"): ("chk_hormuz",),
    ("GULF", "TW"): ("chk_hormuz", "chk_malacca"),
    ("GULF", "SEA"): ("chk_hormuz", "chk_malacca"),
    ("GULF", "KR"): ("chk_hormuz", "chk_malacca", "chk_taiwan"),
    ("GULF", "JP"): ("chk_hormuz", "chk_malacca", "chk_taiwan"),
    ("GULF", "CN"): ("chk_hormuz", "chk_malacca", "chk_taiwan"),
    ("US", "EU"): (),
    ("US", "JP"): ("chk_panama",),
    ("US", "KR"): ("chk_panama",),
    ("US", "CN"): ("chk_panama",),
    ("US", "TW"): ("chk_panama",),
    ("US", "SEA"): ("chk_panama",),
    ("AU", "JP"): (),
    ("AU", "KR"): (),
    ("AU", "CN"): (),
    ("AU", "TW"): (),
    ("AU", "SEA"): (),
    ("RU", "JP"): (),
    ("RU", "CN"): (),
    ("RU", "IN"): ("chk_turkish", "chk_suez"),
    ("RU", "EU"): ("chk_turkish",),
}
_ZONE = {"JP": "N", "KR": "N", "CN": "N", "TW": "T", "SEA": "S", "IN": "I", "EU": "W", "ROW": "W", "US": "U"}
_ZONE_ROUTE = {  # container lanes between zones, in the listed direction (reversed for the other)
    ("N", "S"): ("chk_taiwan",),
    ("N", "I"): ("chk_taiwan", "chk_malacca"),
    ("N", "W"): ("chk_taiwan", "chk_malacca", "chk_suez"),
    ("T", "I"): ("chk_malacca",),
    ("T", "W"): ("chk_malacca", "chk_suez"),
    ("S", "I"): ("chk_malacca",),
    ("S", "W"): ("chk_malacca", "chk_suez"),
    ("I", "W"): ("chk_suez",),
    ("I", "U"): ("chk_malacca",),
}
_GULF_CT = {"W": ("chk_hormuz", "chk_suez"), "U": ("chk_hormuz", "chk_suez"), "I": ("chk_hormuz",)}
_GULF_CT |= {"T": ("chk_hormuz", "chk_malacca"), "S": ("chk_hormuz", "chk_malacca")}
_GULF_CT |= {"N": ("chk_hormuz", "chk_malacca", "chk_taiwan")}

# sea leg lead times in weeks, symmetric; a location is a region or a chokepoint id; SYNTHETIC(placeholder), with spec
# §14's Asia-Europe 30 d via Suez and 42 d via the Cape as anchors (4 and 6 weeks)
_LEGS = {
    ("GULF", "chk_hormuz"): 1,
    ("chk_hormuz", "chk_malacca"): 1,
    ("chk_hormuz", "chk_suez"): 1,
    ("chk_hormuz", "IN"): 1,
    ("chk_malacca", "TW"): 1,
    ("chk_malacca", "SEA"): 1,
    ("chk_malacca", "IN"): 1,
    ("chk_malacca", "chk_taiwan"): 1,
    ("chk_malacca", "chk_suez"): 2,
    ("chk_malacca", "US"): 3,
    ("chk_taiwan", "JP"): 1,
    ("chk_taiwan", "KR"): 1,
    ("chk_taiwan", "CN"): 1,
    ("chk_taiwan", "SEA"): 1,
    ("chk_suez", "EU"): 1,
    ("chk_suez", "ROW"): 1,
    ("chk_suez", "IN"): 1,
    ("chk_suez", "US"): 2,
    ("US", "chk_panama"): 1,
    ("chk_panama", "JP"): 3,
    ("chk_panama", "KR"): 3,
    ("chk_panama", "CN"): 3,
    ("chk_panama", "TW"): 3,
    ("chk_panama", "SEA"): 3,
    ("RU", "chk_turkish"): 1,
    ("UA", "chk_turkish"): 1,
    ("chk_turkish", "chk_suez"): 1,
    ("chk_turkish", "EU"): 1,
    ("chk_turkish", "US"): 2,
    ("chk_turkish", "ROW"): 1,
}
LEGS = {frozenset(k): v for k, v in _LEGS.items()}
_DIRECT = {  # direct sea connections without a chokepoint, weeks, symmetric; same region 1; SYNTHETIC(placeholder)
    ("JP", "TW"): 1,
    ("JP", "KR"): 1,
    ("JP", "CN"): 1,
    ("KR", "TW"): 1,
    ("KR", "CN"): 1,
    ("CN", "TW"): 1,
    ("TW", "SEA"): 1,
    ("US", "JP"): 2,
    ("US", "KR"): 2,
    ("US", "CN"): 2,
    ("US", "TW"): 2,
    ("US", "SEA"): 3,
    ("US", "EU"): 2,
    ("US", "ROW"): 2,
    ("EU", "ROW"): 1,
    ("AU", "JP"): 2,
    ("AU", "KR"): 2,
    ("AU", "CN"): 2,
    ("AU", "TW"): 2,
    ("AU", "SEA"): 1,
    ("RU", "JP"): 1,
    ("RU", "CN"): 1,
}
DIRECT = {frozenset(k): v for k, v in _DIRECT.items()}

SMALL_KEEP = frozenset(
    (
        "src_qa_lng",
        "src_gulf_crude",
        "src_us_lng",
        "src_us_crude",
        "src_au_lng",
        "src_ru_gas",
        "src_kz_uranium",
        *(row.id for row in NODES if row.type == "chokepoint"),
        *(f"{t}_{r}" for t in ("term", "grid") for r in ("tw", "kr", "jp", "eu")),
        "mat_jp_wafer",
        "mat_de_wafer",
        "mat_ua_neon",
        "fab_tw_leading_1",
        "fab_tw_mature_1",
        "fab_kr_memory_1",
        "fab_jp_memory_1",
        "fab_eu_leading_1",
        "fab_eu_mature_1",
        "osat_my",
        "osat_tw",
        "osat_kr",
        "sink_us",
        "sink_eu",
        "sink_cn",
        "sink_jp",
    )
)
SIZES = {  # spec §4.8, design §2.2
    "small": SizeSpec("small", "chokepoint-small", 52, SMALL_KEEP),
    "full": SizeSpec("full", "chokepoint-full", 104, None),
}


def flagship_tables() -> FlagshipTables:
    """The one table set of `full` (spec §4.6-4.7; design §12 M5 rows)."""
    return FlagshipTables(
        regions=REGIONS,
        nodes=NODES,
        commodities=COMMODITIES,
        values=VALUES,
        energy_plan=ENERGY_PLAN,
        grid_shares=GRID_SHARES,
        grid_deliverable=GRID_DELIVERABLE,
        days_cover=BUFFER_DAYS,
        fab_cap0=FAB_CAP0,
        use=USE,
        fab_osat=FAB_OSAT,
        osat_atp=OSAT_ATP,
        sink_share=SINK_SHARE,
        prohibited=PROHIBITED,
        chokepoint_adjacency=CHOKEPOINT_ADJACENCY,
        chokepoint_class=CHOKEPOINT_CLASS,
    )


def fab_osat_shares(tables: FlagshipTables, size: SizeSpec) -> dict[str, dict[str, float]]:
    """Fab -> OSAT -> the share of the fab's output planned there (M5-O24 (b); design §12 'Fab -> OSAT shares').

    Over the fab's OSATs in ``tables.fab_osat`` that ``size`` keeps, in table order: ATP_o / the sum of ATP over them,
    ATP the SIA/BCG capacity of each OSAT's region (``tables.osat_atp``). Only the fabs the size keeps appear; one
    that keeps no OSAT maps to an empty dict (the builder refuses it).
    """
    keep = size.keep
    out: dict[str, dict[str, float]] = {}
    for f, osats in tables.fab_osat:
        if keep is not None and f not in keep:
            continue
        mine = [o for o in osats if keep is None or o in keep]
        total = math.fsum(tables.osat_atp[o].value for o in mine)
        out[f] = {o: tables.osat_atp[o].value / total for o in mine}
    return out


def route_chokepoints(origin: str, dest: str, pool: str) -> tuple[str, ...]:
    """The chokepoints a sea connection from region ``origin`` to ``dest`` passes, in path order (spec §4.7).

    Tanker cargo follows ``TB_ROUTES``; container cargo the zone table, with UA's cargo leaving through the Turkish
    Straits and GULF's through Hormuz first. An empty tuple is a direct edge.

    Raises:
        KeyError: for a tanker pair the routing table does not list.

    """
    if pool == "tb":
        return TB_ROUTES[(origin, dest)]
    if origin == dest:
        return ()
    if origin == "UA":
        return ("chk_turkish",) + route_chokepoints("EU", dest, pool) if dest != "EU" else ("chk_turkish",)
    if origin == "GULF":
        return _GULF_CT[_ZONE[dest]]
    a, b = _ZONE[origin], _ZONE[dest]
    if (a, b) in _ZONE_ROUTE:
        return _ZONE_ROUTE[(a, b)]
    if (b, a) in _ZONE_ROUTE:
        return tuple(reversed(_ZONE_ROUTE[(b, a)]))
    return ()


def leg_tau(a: str, b: str) -> int:
    """Lead time in weeks of the sea leg between two locations (regions or chokepoint ids); Cape legs by rule.

    A leg into or out of the Cape takes one week more than the same leg to Suez, so a Cape duplicate, which replaces
    the Suez node by the Cape node, is tau + 2 on every lane (spec §4.4; §12 'Duplicates').
    """
    if "chk_cape" in (a, b):
        other = b if a == "chk_cape" else a
        return leg_tau(other, "chk_suez") + 1
    if a.startswith("chk_") or b.startswith("chk_"):
        return LEGS[frozenset((a, b))]
    return 1 if a == b else DIRECT[frozenset((a, b))]


# ----- the graph --------------------------------------------------------------------------------------------------
@dataclass
class _Route:
    """A route of the builder: a primary connection or a duplicate, as an edge or a lane (``lane`` set)."""

    tail: str
    head: str
    pool: str
    mode: str
    K: tuple[str, ...]
    edges: list[str]
    cap: float
    kind: str  # primary, cape, lombok, east, bypass, air, turnback
    alt: dict | None = None  # the tagged alt_of of the edge or lane
    own: list[str] = field(default_factory=list)  # the edges its capacity counts on
    lane: str | None = None
    chokepoints: tuple[str, ...] = ()
    planned: dict = field(default_factory=dict)


class _Builder:
    """Assembles nodes, planned flows, routes and edges for one size (``build_graph`` is the public entry)."""

    def __init__(self, tables: FlagshipTables, size: SizeSpec):
        self.t, self.size = tables, size
        self.rows = [r for r in tables.nodes if size.keep is None or r.id in size.keep]
        self.row = {r.id: r for r in self.rows}
        self.order = {r.id: i for i, r in enumerate(self.rows)}
        self.edges: dict[str, dict] = {}
        self.routes: list[_Route] = []
        self.dup_mu: dict[tuple[str, str], float] = defaultdict(float)
        self.planned_mu: dict[tuple[str, str], float] = defaultdict(float)
        self.bypass: list[tuple[str, str]] = []  # (bypass edge id, the lane it replaces)

    # -- locations and edges ------------------------------------------------------------------------------------
    def loc(self, node: str) -> str:
        return node if node.startswith("chk_") else self.row[node].region

    def edge(self, eid: str, tail: str, head: str, mode: str, pool: str | None, tau: int, c0: float, K=()) -> str:
        """Register edge ``eid`` or check that a second use agrees on its ends, mode, lead and freight."""
        if eid in self.edges:
            e = self.edges[eid]
            if (e["tail"], e["head"], e["mode"], e["pool"], e["tau"], e["c0"]) != (tail, head, mode, pool, tau, c0):
                raise ValueError(f"edge {eid} built twice with different attributes")
        else:
            self.edges[eid] = {
                "tail": tail,
                "head": head,
                "mode": mode,
                "pool": pool,
                "tau": tau,
                "c0": c0,
                "K": set(),
                "u0": 0.0,
                "alt_of": None,
            }
        self.edges[eid]["K"].update(K)
        return eid

    def sea_leg(self, kind: str, a: str, b: str, pool: str, K) -> str:
        tau = leg_tau(self.loc(a), self.loc(b))
        return self.edge(f"{kind}.{pool}.{a}.{b}", a, b, "sea", pool, tau, FREIGHT_PER_WEEK[pool] * tau, K)

    def route_cost(self, route: _Route) -> float:
        return math.fsum(self.edges[e]["c0"] for e in route.edges)

    # -- planned flows (design §12 "Scales and planned flows") ----------------------------------------------------
    def planned(self) -> dict[tuple[str, str, str], dict[str, float]]:
        """Planned weekly flows per connection (tail, head, how) and commodity; capacities follow by headroom."""
        t, row = self.t, self.row
        conn: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.requirement: dict[tuple[str, str], float] = {}
        for g in (r for r in self.rows if r.type == "grid"):
            region = g.region
            for k, zeta in t.grid_shares[region].items():
                req = zeta.value * t.grid_deliverable[region].value
                self.requirement[(g.id, k)] = req
                plan = [p for p in t.energy_plan[region][k] if p[0] in row]
                total = math.fsum(s for _, s, _ in plan)
                if not plan:
                    raise ValueError(f"{self.size.kind}: grid {g.id} keeps no source of {k}")
                for src, share, how in plan:
                    q = req * share / total
                    term = f"term_{region.lower()}"
                    if how == "pipeline":
                        conn[(src, g.id, "pipeline")][k] += q
                    else:
                        conn[(src, term, "sea" if how == "sea" else "domestic")][k] += q
                        conn[(term, g.id, "tg")][k] += q
        for tail, head, k in t.prohibited:
            if tail in row and head in row:
                conn[(tail, head, "pipeline" if row[head].type == "grid" else "sea")][k] += 0.0
        use = defaultdict(list)
        for m, fabs in t.use:
            for f in fabs:
                if m in row and f in row:
                    use[f].append(m)
        osats = fab_osat_shares(t, self.size)  # M5-O24 (b): each fab's output by its OSATs' ATP capacity
        self.inflow: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))  # OSAT raw inflow
        for f in (r for r in self.rows if r.type == "fab"):
            cap = t.fab_cap0[f.id].value
            if not use[f.id] or not osats.get(f.id):
                raise ValueError(f"{self.size.kind}: fab {f.id} keeps no material or no OSAT")
            for m in use[f.id]:
                conn[(m, f.id, "sea")]["wafer"] += cap / len(use[f.id])
            product = PRODUCT[f.cls]
            for o, share in osats[f.id].items():
                conn[(f.id, o, "sea")][product] += cap * share
                self.inflow[o][product] += cap * share
        sinks = [r for r in self.rows if r.type == "sink"]
        mat_total = math.fsum(t.sink_share[s.id].value for s in sinks)
        le_total = math.fsum(t.sink_share[s.id].value for s in sinks if s.region != "CN")
        self.ban_cap: dict[tuple[str, str], float] = {}  # capacity of an air edge into CN, all its cargo in Z_0
        for o in (r for r in self.rows if r.type == "osat"):  # chip_le by air, chip_mat by sea (§12 'Duplicates')
            le, mat = self.inflow[o.id].get("chip_le_raw", 0.0), self.inflow[o.id].get("chip_mat_raw", 0.0)
            for s in sinks:
                share = t.sink_share[s.id].value
                if mat > 0:
                    conn[(o.id, s.id, "sea")]["chip_mat"] += mat * share / mat_total
                if le > 0:
                    air = conn[(o.id, s.id, "air")]
                    if s.region == "CN":  # chip_le into CN is in Z_0 from week 0 (spec §4.7): the edge carries only it
                        air["chip_le"] += 0.0
                        self.ban_cap[(o.id, s.id)] = H_SINK * le * share / mat_total
                    else:
                        air["chip_le"] += le * share / le_total
                        if mat > 0:
                            air["chip_mat"] += 0.0  # air admits mature chips, planned by sea
        return conn

    # -- routes -------------------------------------------------------------------------------------------------
    def primary(self, tail: str, head: str, how: str, flows: dict[str, float]) -> _Route:
        K = tuple(k for k, _, _ in COMMODITIES if k in flows)
        pool = POOL_OF[K[0]]
        total = math.fsum(flows.values())
        cap = (H_SINK if self.row[head].type == "sink" else H_ROUTE) * total
        if how == "air" and (tail, head) in self.ban_cap:
            cap = self.ban_cap[(tail, head)]
        if (tail, head) in {(a, b) for a, b, _ in self.t.prohibited}:
            k = next(k for a, b, k in self.t.prohibited if (a, b) == (tail, head))
            cap = PROHIBITED_SHARE * self.requirement[(head if head.startswith("grid_") else self._grid(head), k)]
        if how == "pipeline" and (tail, head) in GEM_PIPELINE:  # Q104: the gas pipeline's operating capacity
            cap = GEM_PIPELINE[(tail, head)][0]
        if how in ("pipeline", "domestic"):
            tau = NUC_TAU if K == ("nucfuel",) else PIPE_TAU
            eid = self.edge(
                f"pipe.{pool}.{tail}.{head}", tail, head, "pipeline", pool, tau, FREIGHT_PER_WEEK[pool] * tau, K
            )
            return _Route(tail, head, pool, "pipeline", K, [eid], cap, "primary", own=[eid], planned=dict(flows))
        if how == "tg":
            eid = self.edge(f"tg.{pool}.{tail}.{head}", tail, head, "pipeline", pool, TG_TAU, TG_C0, K)
            return _Route(tail, head, pool, "pipeline", K, [eid], cap, "primary", own=[eid], planned=dict(flows))
        if how == "air":
            sea = self.sea_cost(tail, head, pool)
            eid = self.edge(f"air.{pool}.{tail}.{head}", tail, head, "air", pool, AIR_TAU, AIR_COST * sea, K)
            return _Route(tail, head, pool, "air", K, [eid], cap, "primary", own=[eid], planned=dict(flows))
        chks = route_chokepoints(self.row[tail].region, self.row[head].region, pool)
        seq = (tail, *chks, head)
        edges = [self.sea_leg("sea", a, b, pool, K) for a, b in zip(seq, seq[1:])]
        r = _Route(tail, head, pool, "sea", K, edges, cap, "primary", own=list(edges), chokepoints=chks)
        r.planned = dict(flows)
        if chks:
            r.lane = f"lane.{tail}.{head}"
        return r

    def _grid(self, terminal: str) -> str:
        return "grid_" + terminal.removeprefix("term_")

    def sea_cost(self, tail: str, head: str, pool: str) -> float:
        """Freight of the sea route between two nodes, from its legs (the base of an air edge's 4x, spec §4.7)."""
        seq = (tail, *route_chokepoints(self.row[tail].region, self.row[head].region, pool), head)
        return math.fsum(FREIGHT_PER_WEEK[pool] * leg_tau(self.loc(a), self.loc(b)) for a, b in zip(seq, seq[1:]))

    def duplicates(self, p: _Route) -> list[_Route]:
        """The sea duplicates of primary lane ``p``: Cape, Lombok-Sunda, east of Taiwan, and the Hormuz bypass."""
        out = []
        seq = (p.tail, *p.chokepoints, p.head)
        own = DUP_OWN_SHARE * p.cap
        for pos in range(1, len(seq) - 1):
            c, prev, nxt = seq[pos], seq[pos - 1], seq[pos + 1]
            if c == "chk_suez":
                kind, new = "cape", seq[:pos] + ("chk_cape",) + seq[pos + 1 :]
                cap = (1 + ALPHA_SP[p.pool]) * own
                e1 = self._scaled_leg("cape", prev, "chk_cape", p.pool, p.K, self.edges[p.edges[pos - 1]]["c0"])
                e2 = self._scaled_leg("cape", "chk_cape", nxt, p.pool, p.K, self.edges[p.edges[pos]]["c0"])
                mine, mode, K = [e1, e2], "sea", p.K
                self.dup_mu[("chk_cape", p.pool)] += own
            elif c in DETOUR:
                kind, dtau, mult = DETOUR[c]
                new = seq[:pos] + seq[pos + 1 :]
                tau = leg_tau(self.loc(prev), self.loc(c)) + leg_tau(self.loc(c), self.loc(nxt)) + dtau
                c0 = mult * (self.edges[p.edges[pos - 1]]["c0"] + self.edges[p.edges[pos]]["c0"])
                mine = [self.edge(f"{kind}.{p.pool}.{prev}.{nxt}", prev, nxt, "sea", p.pool, tau, c0, p.K)]
                cap, mode, K = (1 + ALPHA_SP[p.pool]) * own, "sea", p.K
            elif c == "chk_hormuz" and pos == 1 and "crude" in p.K and p.tail == "src_gulf_crude":
                kind, new = "bypass", seq[:1] + seq[2:]
                tau = sum(self.edges[e]["tau"] for e in p.edges[:2])
                c0 = BYPASS_COST * (self.edges[p.edges[0]]["c0"] + self.edges[p.edges[1]]["c0"])
                mine = [self.edge(f"bypass.tb.{prev}.{nxt}", prev, nxt, "pipeline", "tb", tau, c0, ("crude",))]
                cap, mode, K = 0.0, "pipeline", ("crude",)  # set from the lane's nominal crude flow (22)
                self.bypass.append((mine[0], p.lane))
            else:
                continue
            edges = list(p.edges[: pos - 1]) + mine + list(p.edges[pos + 1 :])
            d = _Route(p.tail, p.head, p.pool, mode, K, edges, cap, kind, {"lane": p.lane}, own=list(mine))
            d.chokepoints = tuple(new[1:-1])
            if d.chokepoints:
                d.lane = f"lane.{p.tail}.{p.head}.{kind}"
            else:
                self.edges[mine[0]]["alt_of"] = {"lane": p.lane}  # an edge-level duplicate of a whole lane (Q85)
            out.append(d)
        return out

    def _scaled_leg(self, kind: str, a: str, b: str, pool: str, K, replaced_c0: float) -> str:
        tau = leg_tau(self.loc(a), self.loc(b))
        return self.edge(f"{kind}.{pool}.{a}.{b}", a, b, "sea", pool, tau, CAPE_COST * replaced_c0, K)

    def build(self) -> nx.MultiDiGraph:
        conn = self.planned()
        primaries: dict[tuple[str, str, str], _Route] = {}
        for (tail, head, how), flows in sorted(conn.items(), key=lambda kv: self._conn_key(kv[0])):
            primaries[(tail, head, how)] = r = self.primary(tail, head, how, flows)
            self.routes.append(r)
        # terminal -> grid capacity is the terminal's throughput (§2.2): GEM's LNG nameplate (Q104) plus a crude part
        self.throughput, self.crude_part = {}, {}
        for r in self.rows:
            if r.type == "terminal":
                inflow = defaultdict(float)
                for (_a, b, _h), f in conn.items():
                    if b == r.id:
                        for k, q in f.items():
                            inflow[k] += q
                if inflow["lng"] > GEM_LNG_IMPORT[r.region][0]:
                    raise ValueError(f"{r.id}: planned LNG {inflow['lng']} above GEM's nameplate (Q104)")
                self.crude_part[r.id] = H_THR * inflow["crude"]
                self.throughput[r.id] = GEM_LNG_IMPORT[r.region][0] + self.crude_part[r.id]
                primaries[(r.id, self._grid(r.id), "tg")].cap = self.throughput[r.id]
        # duplicates of primary sea lanes, never of a link in Z_0 or of a duplicate
        banned = {(a, b) for a, b, _ in self.t.prohibited}
        for r in list(self.routes):
            to_sink = self.row[r.head].type == "sink"  # a sink-bound lane's duplicate is its air edge (§12 M5)
            if r.lane is not None and (r.tail, r.head) not in banned and not to_sink:
                for d in self.duplicates(r):
                    self.routes.append(d)
        # air duplicates of material -> fab and fab -> OSAT sea routes; OSAT -> sink air marked alt_of the sea route
        for (tail, head, how), r in primaries.items():
            kinds = (self.row[tail].type, self.row[head].type)
            ref = {"lane": r.lane} if r.lane else {"edge": r.edges[0]}
            if how == "sea" and kinds in (("material", "fab"), ("fab", "osat")):
                c0 = AIR_COST * self.route_cost(r)
                eid = self.edge(f"air.{r.pool}.{tail}.{head}", tail, head, "air", r.pool, AIR_TAU, c0, r.K)
                self.edges[eid]["alt_of"] = ref
                self.routes.append(_Route(tail, head, r.pool, "air", r.K, [eid], AIR_CAP * r.cap, "air", ref, [eid]))
            if how == "air" and (tail, head, "sea") in primaries:
                sea = primaries[(tail, head, "sea")]
                self.edges[r.edges[0]]["alt_of"] = {"lane": sea.lane} if sea.lane else {"edge": sea.edges[0]}
        # mu's floor: the planned traffic of the primary lanes through each chokepoint, per pool (§12 'Scales')
        for r in primaries.values():
            for c in r.chokepoints:
                self.planned_mu[(c, r.pool)] += math.fsum(r.planned.values())
        # turn-backs at the chokepoints of TURNBACK_CHOKEPOINTS (M5-O22; Suez to the Cape by default), tanker cargo
        # only, on no lane (Q4; §12 'Turn-backs' and 'Turn-backs (M5-O22 default)')
        for c in TURNBACK_CHOKEPOINTS:
            for r in [x for x in self.routes if x.kind == "primary" and x.lane and x.pool == "tb"]:
                if (r.tail, r.head) in banned or c not in r.chokepoints:
                    continue
                pos = r.chokepoints.index(c)
                e = r.edges[pos + 1]
                if self.edges[e]["head"] != r.head:
                    continue
                K = tuple(k for k in r.K if OVERRIDE[k])
                src = self.edges[e]
                eid = self.edge(
                    f"turnback.tb.{c}.{r.head}",
                    c,
                    r.head,
                    "sea",
                    "tb",
                    src["tau"] + TURNBACK_DTAU,
                    CAPE_COST * src["c0"],
                    K,
                )
                self.edges[eid]["alt_of"] = {"edge": e}
                cap = (1 + ALPHA_SP["tb"]) * DUP_OWN_SHARE * r.cap
                self.routes.append(_Route(c, r.head, "tb", "sea", K, [eid], cap, "turnback", {"edge": e}, [eid]))
        for r in self.routes:
            for e in r.own:
                self.edges[e]["u0"] += r.cap
        for r in self.rows:
            if r.type == "terminal":
                self.edges[f"tg.tb.{r.id}.{self._grid(r.id)}"]["u0"] = self.throughput[r.id]
        return self.graph(conn)

    def _conn_key(self, key: tuple[str, str, str]) -> tuple:
        tail, head, how = key
        return (self.order[tail], self.order[head], ("sea", "air", "pipeline", "domestic", "tg").index(how))

    # -- the networkx graph -------------------------------------------------------------------------------------
    def graph(self, conn) -> nx.MultiDiGraph:
        """The MultiDiGraph: nodes with their §2.1 blocks (before the derived values), edges keyed by id, lanes."""
        t = self.t
        g = _nx().MultiDiGraph(kind=self.size.kind, instance_id=self.size.instance_id, T=self.size.T)
        supply = defaultdict(float)
        for (tail, _head, _how), flows in conn.items():
            if self.row[tail].type in SUPPLY_TYPES:
                for k, q in flows.items():
                    supply[(tail, k)] += q
        for r in self.rows:
            attrs = {"type": r.type, "region": r.region, "row": r}
            if r.type in SUPPLY_TYPES:
                attrs["supply"] = {k: H_SUPPLY * supply[(r.id, k)] for k in r.commodities}
            elif r.type == "terminal":
                attrs["throughput"] = self.throughput[r.id]
            elif r.type == "grid":
                fabs = [f for f in self.rows if f.type == "fab" and self.fab_grid(f) == r.id]
                deliverable = t.grid_deliverable[r.region].value
                attrs["deliverable"] = deliverable
                attrs["base_load"] = deliverable - math.fsum(self.fab_e(f).value * t.fab_cap0[f.id].value for f in fabs)
                attrs["shares"] = {k: v.value for k, v in t.grid_shares[r.region].items()}
            elif r.type == "fab":
                attrs["cap0"], attrs["grid"], attrs["e"] = t.fab_cap0[r.id].value, self.fab_grid(r), self.fab_e(r).value
                attrs["product"] = PRODUCT[r.cls]
            elif r.type == "osat":
                attrs["thr"] = H_THR * math.fsum(self.inflow[r.id].values())
                attrs["packages"] = {raw: PACKAGED[raw] for raw in PACKAGED if self.inflow[r.id].get(raw, 0.0) > 0}
            g.add_node(r.id, **attrs)
        for eid, e in sorted(self.edges.items(), key=lambda kv: self._edge_key(kv[0], kv[1])):
            K = tuple(k for k, _, _ in COMMODITIES if k in e["K"])
            g.add_edge(e["tail"], e["head"], key=eid, **{**e, "K": K, "id": eid})
        lanes = [r for r in self.routes if r.lane is not None]
        rank = {"primary": 0, "cape": 1, "lombok": 2, "east": 3, "bypass": 4}
        lanes.sort(key=lambda r: (rank[r.kind], self.order[r.tail], self.order[r.head]))
        g.graph["lanes"] = [
            {"id": r.lane, "edges": list(r.edges), "chokepoints": list(r.chokepoints), "alt_of": r.alt, "kind": r.kind}
            for r in lanes
        ]
        g.graph["dup_mu"] = dict(self.dup_mu)
        g.graph["planned_mu"] = dict(self.planned_mu)
        g.graph["bypass"] = list(self.bypass)
        g.graph["requirement"] = dict(self.requirement)
        g.graph["crude_part"] = dict(self.crude_part)  # the placeholder part of each terminal's throughput (Q104)
        g.graph["prohibited"] = self.prohibitions(g)
        return g

    def _edge_key(self, eid: str, e: dict) -> tuple:
        kinds = ("sea", "cape", "lombok", "east", "bypass", "turnback", "air", "pipe", "tg", "cpl")
        return (self.order[e["tail"]], self.order[e["head"]], kinds.index(eid.split(".")[0]), eid)

    def fab_grid(self, f: NodeRow) -> str:
        region = "EU" if f.region == "ROW" else f.region  # the ROW fab draws on the EU grid (§12 'Energy paths')
        return f"grid_{region.lower()}"

    def fab_e(self, f: NodeRow) -> ScaleValue:
        if f.cls == "memory":
            return E_F["memory_kr" if f.region == "KR" else "memory_jp"]
        return E_F[f.cls]

    def prohibitions(self, g: nx.MultiDiGraph) -> list[tuple[str, str]]:
        """Z_0 at reset: the table's links on their last edge, and chip_le on every edge into CN (spec §4.7, §6.3)."""
        out = []
        for tail, head, k in self.t.prohibited:
            if tail in self.row and head in self.row:
                r = next(x for x in self.routes if (x.tail, x.head, x.kind) == (tail, head, "primary") and k in x.K)
                out.append((r.edges[-1], k))
        for a, b, eid, data in g.edges(keys=True, data=True):
            if g.nodes[b]["region"] == "CN" and "chip_le" in data["K"]:
                out.append((eid, "chip_le"))
        return sorted(set(out), key=lambda p: (list(self.edges).index(p[0]), p[1]))


def build_graph(tables: FlagshipTables, size: SizeSpec) -> nx.MultiDiGraph:
    """The size's graph from the tables (spec §4.6-4.7): nodes with their blocks, arcs keyed by edge id, lanes.

    Coupling edges (grid -> fab) are added here too, with an empty K and no capacity (Q58 M5). The lanes are the
    graph attribute ``lanes`` (paths whose interior nodes are chokepoints, Q4), in file order: primary lanes, then the
    Cape, Lombok-Sunda, east-of-Taiwan and bypass duplicates.
    """
    b = _Builder(tables, size)
    g = b.build()
    for f in [n for n, d in g.nodes(data=True) if d["type"] == "fab"]:
        grid = g.nodes[f]["grid"]
        eid = f"cpl.{grid}.{f}"
        g.add_edge(grid, f, key=eid, id=eid, tail=grid, head=f, mode="grid", pool=None, tau=0, c0=0.0, K=(), u0=None)
        g.edges[grid, f, eid]["alt_of"] = None
    return g


# ----- the instance dict ------------------------------------------------------------------------------------------
def _h(k: str) -> float:
    return VALUES[k].value * RATE / 52  # h_k = v_k r / 52 (§11 row 3)


def _slots(g: nx.MultiDiGraph, lanes: list[dict]) -> dict[str, list[str]]:
    """The commodities each node holds: its role's (§2.1) and every K_e of its non-chokepoint edges (8)."""
    order = [k for k, _, _ in COMMODITIES]
    held: dict[str, set[str]] = defaultdict(set)
    for n, d in g.nodes(data=True):
        row = d["row"]
        if d["type"] in SUPPLY_TYPES:
            held[n].update(row.commodities)
        elif d["type"] == "grid":
            held[n].update(d["shares"])
        elif d["type"] == "fab":
            held[n].update(("wafer", d["product"]))
        elif d["type"] == "osat":
            held[n].update(d["packages"])
            held[n].update(d["packages"].values())
        elif d["type"] == "sink":
            held[n].update(("chip_le", "chip_mat"))
    edge_K = {eid: data["K"] for _, _, eid, data in g.edges(keys=True, data=True)}
    for a, b, eid, data in g.edges(keys=True, data=True):
        for end in (a, b):
            if g.nodes[end]["type"] != "chokepoint":
                held[end].update(data["K"])
    for ln in lanes:
        K = set(edge_K[ln["edges"][0]])
        for e in ln["edges"][1:]:
            K &= set(edge_K[e])
        for c in ln["chokepoints"]:
            held[c].update(K)
    return {n: [k for k in order if k in held[n]] for n in g.nodes}


def _salvage(inst: Instance) -> dict[tuple[int, int], float]:
    """Salvage nu = c^min x share 1 (§11 row 1; Q93): per-edge relaxation from 0 at supply nodes over open edges.

    The relaxation runs to its fixed point, so nu_head <= nu_tail + c0 holds on every non-coupling edge and commodity
    open at reset, exactly in floats (the loader's salvage invariant); a fab's raw chip is valued at its wafer, an
    OSAT's packaged chip at its raw chip. A slot no open edge reaches keeps 0.
    """
    z0 = set(inst.prohibitions_at_reset)
    supply = set(inst.supply_nodes)
    nu: dict[tuple[int, int], float] = {
        (s.node, s.k): (0.0 if s.node in supply else math.inf) for s in inst.stock_slots
    }
    for _ in range(4 * len(inst.nodes) + 8):
        changed = False
        for j, e in enumerate(inst.edges):
            if e.coupling:
                continue
            for k in e.K:
                if (j, k) in z0 or (e.tail, k) not in nu or (e.head, k) not in nu:
                    continue
                cand = nu[(e.tail, k)] + e.c0
                if cand < nu[(e.head, k)] and e.head not in supply:
                    nu[(e.head, k)] = cand
                    changed = True
        for f in inst.fabs:
            fab = inst.nodes[f].fab
            if nu[(f, fab.input)] < nu[(f, fab.product)]:
                nu[(f, fab.product)] = nu[(f, fab.input)]
                changed = True
        for o in inst.osats:
            for raw_k, pk in inst.nodes[o].osat.packages.items():
                if nu[(o, raw_k)] < nu[(o, pk)]:
                    nu[(o, pk)] = nu[(o, raw_k)]
                    changed = True
        if not changed:
            return {key: (0.0 if math.isinf(v) else v) for key, v in nu.items()}
    raise ValueError("salvage relaxation did not converge")


def _routes(inst: Instance, open_only: bool) -> list[tuple[int, int, int, float]]:
    """Every route (tail, head, k, cost): edges between non-chokepoint nodes and whole lanes, freight plus war risk.

    A lane's cost adds, for each chokepoint it passes, the highest war-risk class's transit cost of the commodity (11).
    With ``open_only``, routes with an (edge, k) in Z_0 are left out.
    """
    chk, z0 = set(inst.chokepoints), set(inst.prohibitions_at_reset)
    out = []
    for j, e in enumerate(inst.edges):
        if e.coupling or e.tail in chk or e.head in chk:
            continue
        for k in e.K:
            if not (open_only and (j, k) in z0):
                out.append((e.tail, e.head, k, e.c0))
    for li, ln in enumerate(inst.lanes):
        for k in inst.lane_K[li]:
            if open_only and any((j, k) in z0 for j in ln.edges):
                continue
            war = math.fsum(max(inst.nodes[c].chokepoint.war_risk_cost[k]) for c in ln.chokepoints)
            freight = math.fsum(inst.edges[j].c0 for j in ln.edges)
            out.append((inst.edges[ln.edges[0]].tail, inst.lane_destination(li), k, freight + war))
    return out


def _landed_cost(inst: Instance, sink: int, k: int) -> float:
    """The largest landed cost of packaged chip k at ``sink`` over supply chains open at reset (Q85; §12 M5).

    A chain is a material route, a fab route, an OSAT route and a sink route; a demand no open chain reaches takes
    the maximum over chains through Z_0.
    """
    for open_only in (True, False):
        best = {}  # (node, k) -> the dearest cost of reaching it
        routes = _routes(inst, open_only)
        for m in inst.supply_nodes:
            for s in inst.stock_slots:
                if s.node == m and inst.nodes[m].type == "material":
                    best[(m, s.k)] = 0.0
        stages = (("material", "fab"), ("fab", "osat"), ("osat", "sink"))
        for ta, tb in stages:
            for a, b, kk, cost in routes:
                if inst.nodes[a].type == ta and inst.nodes[b].type == tb and (a, kk) in best:
                    best[(b, kk)] = max(best.get((b, kk), -math.inf), best[(a, kk)] + cost)
            if tb == "fab":
                for f in inst.fabs:
                    fab = inst.nodes[f].fab
                    if (f, fab.input) in best:
                        best[(f, fab.product)] = best[(f, fab.input)]
            if tb == "osat":
                for o in inst.osats:
                    for raw_k, pk in inst.nodes[o].osat.packages.items():
                        if (o, raw_k) in best:
                            best[(o, pk)] = max(best.get((o, pk), -math.inf), best[(o, raw_k)])
        if (sink, k) in best:
            return best[(sink, k)]
    raise ValueError(f"no supply chain reaches {inst.nodes[sink].id} with {inst.commodities[k].id}")


def warm_initial_state(inst: Instance, quantiles: Mapping | None = None) -> tuple[dict, dict[tuple[str, str], float]]:
    """The warm initial state of §2.3 on naive's plan (design §12 'Warm start'), and each slot's steady throughput.

    ``quantiles`` is naive's F_Q at one rung, keyed as ``naive_plan`` takes it; None is the point mass at 0. Returns
    the ``initial_state`` block of the instance file and, per (node id, commodity id), the slot's steady weekly
    throughput (for the storage rule). Only the stock depends on ``quantiles`` (the closure terms of the levels): the
    pipeline, WIP, queue lots and throughputs read the flows, so they are one block for every rung (§2.3).
    """
    from shockbench_flow.policies.naive import naive_plan  # the warm start is naive's steady state (§2.3)

    plan = naive_plan(inst, quantiles)
    nodes, K, E, L = inst.nodes, inst.commodities, inst.edges, inst.lanes
    ups = inst.params.upsilon
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for r in plan.routes:
        groups[(r.dest, r.k)].append(r)
    fab_start = {f: ups * nodes[f].fab.cap0 for f in inst.fabs}  # steady lots p_f = upsilon cap^0 (step 2)
    consumption: dict[tuple[int, int], float] = {}
    for dem in inst.demands:
        consumption[(dem.node, dem.k)] = dem.dbar
    for f in inst.fabs:
        consumption[(f, nodes[f].fab.input)] = fab_start[f]
    for g in inst.grids:
        grid = nodes[g].grid
        load = grid.base_load + math.fsum(nodes[f].fab.e * fab_start[f] for f in inst.fabs if nodes[f].fab.grid == g)
        for k in grid.fuels:
            consumption[(g, k)] = grid.shares[k] * load  # the actual burn b_gk (Q91)
    flow: dict[int, float] = {}  # plan route index -> steady flow x_lk

    def split(key: tuple[int, int], total: float) -> None:
        rs = groups[key]
        dsum = math.fsum(r.d for r in rs)
        for r in rs:
            flow[plan.routes.index(r)] = 0.0 if dsum == 0 else total * r.d / dsum

    for key in list(groups):
        j = key[0]
        if nodes[j].type in ("sink", "fab", "grid"):
            split(key, consumption[key])
        elif nodes[j].type == "osat":
            for r in groups[key]:
                flow[plan.routes.index(r)] = r.d  # an OSAT packages what arrives: its steady inflow is d (§8.1)
    for key in list(groups):  # terminals: their grid's pull through the terminal edge
        j, k = key
        if nodes[j].type == "terminal":
            out = math.fsum(
                flow.get(i, 0.0) for i, r in enumerate(plan.routes) if r.k == k and r.first_edge in inst.out_edges[j]
            )
            consumption[key] = out
            split(key, out)
    stock: dict[tuple[int, int], float] = defaultdict(float)
    for (j, k), s in plan.targets.items():
        idx = [plan.routes.index(r) for r in groups[(j, k)]]
        stock[(j, k)] = s - math.fsum((plan.routes[i].tau + 1) * flow[i] for i in idx)
    outflow: dict[tuple[int, int], float] = defaultdict(float)
    for i, r in enumerate(plan.routes):
        outflow[(E[r.first_edge].tail, r.k)] += flow[i]
    for f in inst.fabs:  # one week of steady output, which next week's dispatch draws (5)
        stock[(f, nodes[f].fab.product)] += outflow[(f, nodes[f].fab.product)]
    for o in inst.osats:
        for pk in nodes[o].osat.packages.values():
            stock[(o, pk)] += outflow[(o, pk)]
    for s in inst.stock_slots:
        if s.node in inst.supply_nodes:
            stock[(s.node, s.k)] = s.supply  # one week of availability (§2.3)
    pipeline = []
    for i, r in enumerate(plan.routes):
        if flow[i] <= 0.0:
            continue
        for e in r.edges:
            for a in range(1, E[e].tau + 1):
                pipeline.append(
                    {
                        "edge": E[e].id,
                        "k": K[r.k].id,
                        "lane": None if r.lane is None else L[r.lane].id,
                        "qty": flow[i],
                        "dispatch_week": a - E[e].tau,
                        "arrival_week": a,
                    }
                )
    fab_wip = [
        {"node": nodes[f].id, "k": K[nodes[f].fab.product].id, "qty": fab_start[f], "out_week": w}
        for f in inst.fabs
        for w in range(1, nodes[f].fab.tau + 1)
    ]
    osat_in: dict[tuple[int, int], float] = defaultdict(float)
    for i, r in enumerate(plan.routes):
        if nodes[r.dest].type == "osat":
            osat_in[(r.dest, r.k)] += flow[i]
    osat_wip = [
        {"node": nodes[o].id, "k": K[pk].id, "qty": osat_in[(o, raw_k)], "out_week": w}
        for o in inst.osats
        for raw_k, pk in nodes[o].osat.packages.items()
        for w in range(1, nodes[o].osat.tau + 1)
    ]
    throughput = defaultdict(float)
    for (j, k), q in consumption.items():
        throughput[(nodes[j].id, K[k].id)] = max(throughput[(nodes[j].id, K[k].id)], q)
    for (j, k), q in outflow.items():
        throughput[(nodes[j].id, K[k].id)] = max(throughput[(nodes[j].id, K[k].id)], q)
    for (j, k), q in osat_in.items():
        throughput[(nodes[j].id, K[k].id)] = max(throughput[(nodes[j].id, K[k].id)], q)
    for s in inst.stock_slots:
        if s.supply:
            throughput[(nodes[s.node].id, K[s.k].id)] = max(throughput[(nodes[s.node].id, K[s.k].id)], s.supply)
    block = {
        "stock": [
            {"node": nodes[j].id, "k": K[k].id, "qty": max(0.0, q)}
            for (j, k), q in sorted(stock.items())
            if q > 0.0 and j not in inst.chokepoint_ordinal
        ],
        "pipeline": pipeline,
        "fab_wip": fab_wip,
        "osat_wip": osat_wip,
        "queue_lots": [],
    }
    return block, dict(throughput)


def _tags(size: SizeSpec, warm_note: str | Mapping[float, str]) -> list[tuple[str, str, str]]:
    """Provenance by path pattern, first match wins; anything unmatched is SYNTHETIC(placeholder) (V20, V23).

    ``warm_note`` is the source of the initial-state leaves: the F_Q the warm start was built on, one note per rung for
    a warm start per rung (the anchor rung's for ``stock``, ``rung``, the pipeline, WIP and queues).
    """
    from shockbench_flow.disruption.params import RUNGS

    notes = {RUNGS[0]: warm_note} if isinstance(warm_note, str) else dict(warm_note)
    return [
        *(
            (f"initial_state/stock_by_rung/{g!r}/*", "DERIVED(instance)", note)
            for g, note in notes.items()
            if g != RUNGS[0]
        ),
        ("T", "SYNTHETIC(prior: spec §4.8)", f"T = {size.T} on {size.kind} (spec §4.8; design §2.2)"),
        *((f"commodities/{k}/v", v.tag, v.source) for k, v in VALUES.items()),
        ("commodities/*/disposal_cost", PLACEHOLDER, "0.1 v_k (§11 row 5)"),
        (
            "nodes/*/chokepoint/mu/*",
            "DERIVED(instance)",
            "nominal routed flow (22) + own capacity of duplicates (§12 M5)",
        ),
        ("nodes/chk_suez/chokepoint/k_c", "DERIVED(PortWatch)", "Suez 2021 realised 1.24-1.28 (Q7)"),
        ("nodes/*/chokepoint/k_c", "SYNTHETIC(prior: Suez 2021 transferred)", "k_c 1.3 of Suez 2021 (Q7)"),
        ("nodes/*/chokepoint/queue_holding/lng/*", "DERIVED(W2-maritime)", "h^Q/h 8.2 / 60 / 215 x h_lng (§11 row 2)"),
        ("nodes/*/chokepoint/queue_holding/crude/*", "DERIVED(W2-maritime)", "VLCC h^Q/h 4.7 / 17 / 52 x h_crude"),
        (
            "nodes/*/chokepoint/queue_holding/*",
            "SYNTHETIC(prior: liner tariff structure)",
            "containers h^Q = h (§11 row 2)",
        ),
        ("nodes/*/chokepoint/war_risk_cost/lng/*", "DERIVED(W2-maritime)", "0.505 and 2.02 $/MMBtu (§3.4)"),
        (
            "nodes/*/chokepoint/war_risk_cost/crude/*",
            "DERIVED(W2-maritime, EIA MER A2)",
            f"0.656 / 2.63 $/bbl DERIVED(W2-maritime) at 6.062 MMBtu/bbl REAL(EIA MER Table A2; {EVIDENCE['eia']})",
        ),
        ("nodes/*/chokepoint/war_risk_cost/*", PLACEHOLDER, "containers: tiny's 0.5 / 2 USD/wafer-eq (§11 row 65)"),
        ("nodes/*/grid/base_load", "DERIVED(instance)", "deliverable less the grid's fabs' draw at cap^0 (§12 M5)"),
        ("nodes/grid_tw/grid/deliverable", GRID_DELIVERABLE["TW"].tag, GRID_DELIVERABLE["TW"].source),
        *(
            (f"nodes/grid_{r.lower()}/grid/days_cover/lng", v.tag, v.source)
            for r, v in DAYS_COVER.items()
            if v.tag != PLACEHOLDER
        ),
        ("nodes/*/grid/ibar/*", "DERIVED(instance)", "(16) D_g zeta_gk G-bar^0_g / 7"),
        ("nodes/*/osat/tau", "SYNTHETIC(prior: spec §4.5)", "OSAT lag 2 weeks"),
        (
            "nodes/*/osat/thr",
            PLACEHOLDER,
            "1.25 x the planned inflow (placeholder headroom); the inflow is each fab's SIA/BCG capacity split over its"
            f" OSATs by the ATP capacity of their regions (DERIVED(SIA/BCG 2024), {EVIDENCE['sia_bcg']} §4; §12 'Fab"
            " -> OSAT shares (M5-O24 (b))')",
        ),
        ("nodes/*/sink/demand/*/dbar", "DERIVED(instance)", "(22) upsilon x F-bar from the nominal flow"),
        ("nodes/*/sink/demand/*/pi", "SYNTHETIC(prior: spec §8.1)", "10 x the largest landed cost at the sink (Q85)"),
        ("nodes/*/stock/*/salvage", "DERIVED(instance)", "c^min rule of §11 row 1 (Q93); 0 at supply nodes (Q79)"),
        (
            "nodes/*/stock/*/holding_cost",
            "DERIVED(instance)",
            "h = v r / 52, r REAL(Fed funds 4.21 %); 0 at supply nodes",
        ),
        ("edges/tg.*/tau", "SYNTHETIC(prior: spec §4.7)", "terminal -> grid lead 0"),
        *(
            (
                pattern,
                PLACEHOLDER,
                "nuclear fuel: headroom over placeholder planned shares 0.5 / 0.5 (§12 M5 instances)",
            )
            for src in ("src_kz_uranium", "src_ru_enrichment")
            for pattern in (f"edges/pipe.tb.{src}.*/u0", f"nodes/{src}/stock/*/supply_rate")
        ),
        (
            "edges/*.tb.src_*/u0",
            PLACEHOLDER,
            "1.2 x the planned flows it carries: the grid's requirement zeta G (placeholder) x the source's BACI share"
            f" (DERIVED(BACI), {EVIDENCE['baci']}; §12 'Energy source shares (BACI)')",
        ),
        (
            "nodes/src_*/stock/*/supply_rate",
            PLACEHOLDER,
            "1.25 x the planned outflow, planned flows at the BACI shares of placeholder requirements"
            f" ({EVIDENCE['baci']}; §12 'Energy source shares (BACI)')",
        ),
        (
            "nodes/mat_*/stock/*/supply_rate",
            PLACEHOLDER,
            "1.25 x the planned outflow: the fabs' SIA/BCG capacities split equally over their materials (placeholder)",
        ),
        ("edges/air.*/tau", "SYNTHETIC(prior: spec §4.7)", "air duplicate lead 1"),
        ("edges/cpl.*", "DERIVED(instance)", "coupling edge: lead 0, no freight (§2.2)"),
        ("initial_state/*", "DERIVED(instance)", notes[RUNGS[0]]),
        ("params/psi", "SYNTHETIC(prior: onset evidence)", "range DERIVED(MOEA, IEA, METI) 0.3-0.8 (Q36)"),
        ("params/alpha_max", "SYNTHETIC(prior: WPS 6047)", "§11 row 11"),
        ("params/tau_alpha", "SYNTHETIC(prior: WPS 6047)", "1 yr = 52 weeks, §11 row 11"),
        ("params/upsilon", "SYNTHETIC", "utilisation 0.9 (Q76 owner D2; §11 row 60)"),
        ("params/top_tariff", "SYNTHETIC(prior)", "spec chip_tariff_25 (§11 row 25)"),
        ("params/fleet_measure/*", "DERIVED(instance)", "F^b = sum of tau_e u0_e over the pool's sea edges (K3)"),
        ("params/fleet_share/ct", "SYNTHETIC(prior: Alphaliner via gCaptain, Clarksons)", "0.06-0.12, lower end (Q49)"),
    ]


def _sourced_provenance(g: nx.MultiDiGraph, mu: Mapping[tuple[str, str], float]) -> dict[str, tuple]:
    """Provenance of the leaves the M5 evidence sets or bounds (design §12 'M5 stream "instances"' rows; V20).

    Terminal throughputs and their grid edges (GEM's LNG nameplate plus a placeholder crude part, Q104), the GEM gas
    pipelines, the values with a range, and every mu_cb with its strait's volume and the modelled share (option 1).
    """
    out: dict[str, tuple] = {}
    for k, v in VALUES.items():
        if v.lo is not None:
            out[f"commodities/{k}/v"] = (v.tag, v.source, v.lo, v.hi)
    for term, crude in g.graph["crude_part"].items():
        region = g.nodes[term]["region"]
        point, lo, hi = GEM_LNG_IMPORT[region]
        source = (
            f"GEM operating LNG regasification nameplate of {region}, {point:,.0f} GWh/wk (Q104; {EVIDENCE['gem']}),"
            f" plus 1.25 x the planned crude inflow, {crude:,.2f} (placeholder; §12 'Terminal throughput')"
        )
        entry = (TERMINAL_TAG, source, lo + crude, hi + crude)
        out[f"nodes/{term}/terminal/throughput"] = entry
        out[f"edges/tg.tb.{term}.grid_{region.lower()}/u0"] = entry
    for (tail, head), (point, lo, hi) in GEM_PIPELINE.items():
        eid = f"pipe.tb.{tail}.{head}"
        if g.has_edge(tail, head, key=eid):
            out[f"edges/{eid}/u0"] = (
                GEM_TAG,
                f"GEM operating capacity {point:,.0f} GWh/wk (Q104; {EVIDENCE['gem']})",
                lo,
                hi,
            )
    rule = "nominal routed flow (22), floored at the planned traffic, + the duplicates' own capacity (§12 M5)"
    for (c, b), value in mu.items():
        if b == "tb":
            volume, what, script = STRAIT_VOLUME_TB[c]
            share = f"{value / volume:.4f}"
            note = f"the strait's crude and LNG {volume:,.0f} GWh/wk ({what}; {script}), modelled share {share}"
        else:
            note = "the strait's container traffic has no wafer-eq measure (PortWatch counts calls and tonnes): the"
            note += " modelled share is not measured"
        out[f"nodes/{c}/chokepoint/mu/{b}"] = ("DERIVED(instance)", f"{rule}; option 1, {note}", None, None)
    return out


def _buffer_days(tables: FlagshipTables, d: Mapping) -> dict[str, ScaleValue]:
    """D_gk of (16) for grid node attributes ``d``: its region's buffered fuels that the grid burns (§3.5)."""
    return {k: v for k, v in tables.days_cover[d["region"]].items() if d["shares"].get(k, 0.0) > 0.0}


def _buffer_provenance(g: nx.MultiDiGraph, tables: FlagshipTables) -> dict[str, tuple]:
    """Provenance of the buffers beyond gas (the nuclear fuel cover, §3.5): D_gk as tabled, I-bar_gk by (16).

    The gas buffer's leaves keep their pattern tags (``_tags``); lo and hi of I-bar follow those of D.
    """
    out = {}
    for n, d in g.nodes(data=True):
        if d["type"] != "grid":
            continue
        for k, v in _buffer_days(tables, d).items():
            if k == "lng":
                continue
            per_day = d["shares"][k] * d["deliverable"] / 7
            out[f"nodes/{n}/grid/days_cover/{k}"] = (v.tag, v.source, v.lo, v.hi)
            out[f"nodes/{n}/grid/ibar/{k}"] = (
                "DERIVED(instance)",
                f"(16) D_gk zeta_gk G-bar^0_g / 7 at D = {v.value:g} days, {v.tag} (§3.5 'Nuclear fuel cover')",
                None if v.lo is None else v.lo * per_day,
                None if v.hi is None else v.hi * per_day,
            )
    return out


def derive_instance_dict(
    g: nx.MultiDiGraph,
    tables: FlagshipTables,
    size: SizeSpec,
    warm_fq: Mapping[FqKey, float] | Mapping[float, Mapping[FqKey, float]] | None = None,
    warm_note: str | Mapping[float, str] | None = None,
) -> dict:
    """The §2.3 instance dict of the graph, with every derived value and provenance for every leaf (V20).

    Steps (design §12 M5 rows): mu at a fixed point of the reset-time flow (22), the bypass capacities from the lanes'
    nominal crude flow, salvage by the c^min relaxation, d-bar = upsilon F-bar (22), pi (Q85), then the warm start on
    naive's plan with the F_Q quantiles ``warm_fq`` and the storage caps. ``warm_fq`` keyed by ids (``fq_by_id``) is one
    start for every rung, None the point mass's; keyed by γ rung (the anchor rung among them) it is a warm start per
    rung (§2.3; M5-O37 (b)): the anchor rung's block in ``stock`` with ``rung``, the others' in ``stock_by_rung``, and
    the storage caps on the largest block. ``warm_note`` is the initial state's provenance source (default: the point
    mass's; one note per rung for a warm start per rung). Each step loads the dict with the package's loader, so every
    intermediate instance obeys §2.3.

    Raises:
        ValueError: if ``warm_fq``'s keys are not the plan's lane routes through a pair with a law (``fq_by_index``),
            a warm start per rung lacks the anchor rung, or a rung's warm start moves more than the stock.

    """
    lanes = g.graph["lanes"]
    slots = _slots(g, lanes)
    nodes_in_order = list(g.nodes)
    chk_nodes = [n for n in nodes_in_order if g.nodes[n]["type"] == "chokepoint"]
    explicit: dict[str, tuple[str, str, float | None, float | None]] = {}
    ends = {key: (a, b) for a, b, key in g.edges(keys=True)}

    def assemble(mu, bypass_u0, salvage, dbar, pi, initial, storage) -> dict:
        nodes = []
        for n in nodes_in_order:
            d, row = g.nodes[n], g.nodes[n]["row"]
            entry: dict = {"id": n, "type": d["type"], "region": d["region"]}
            stock = {}
            for k in slots[n]:
                if d["type"] == "chokepoint":
                    stock[k] = {"storage": None, "holding_cost": 0.0, "salvage": salvage.get((n, k), 0.0)}
                elif d["type"] in SUPPLY_TYPES:
                    stock[k] = {"storage": storage.get((n, k), 1.0e12), "supply_rate": d["supply"][k]}
                else:
                    stock[k] = {
                        "storage": storage.get((n, k), 1.0e12),
                        "holding_cost": _h(k),
                        "salvage": salvage.get((n, k), 0.0),
                    }
            entry["stock"] = stock
            if d["type"] == "chokepoint":
                hq, wr = {}, {}
                for k in slots[n]:
                    hq[k] = [x * _h(k) for x in HQ_RATIO[k]] if k in HQ_RATIO else [_h(k)] * 3
                    wr[k] = list(WR_LNG if k == "lng" else WR_CRUDE if k == "crude" else WR_CONTAINER)
                entry["chokepoint"] = {
                    "mu": {"tb": mu.get((n, "tb"), 0.0), "ct": mu.get((n, "ct"), 0.0)},
                    "k_c": K_C,
                    "class": tables.chokepoint_class[n],
                    "queue_holding": hq,
                    "war_risk_cost": wr,
                }
            elif d["type"] == "terminal":
                entry["terminal"] = {"throughput": d["throughput"]}
            elif d["type"] == "grid":
                shares = dict(d["shares"])
                shares[UNMODELLED] = 1.0 - math.fsum(shares.values())
                days = _buffer_days(tables, d)
                entry["grid"] = {
                    "base_load": d["base_load"],
                    "deliverable": d["deliverable"],
                    "shares": shares,
                    "days_cover": {k: v.value for k, v in days.items()},
                    "ibar": {k: v.value * d["shares"][k] * d["deliverable"] / 7 for k, v in days.items()},  # (16)
                    "rationed": "lng",
                    "voll": 100 * V_LNG,
                    "priority": "base_first",
                }
            elif d["type"] == "fab":
                entry["fab"] = {
                    "cap0": d["cap0"],
                    "e": d["e"],
                    "grid": d["grid"],
                    "class": row.cls,
                    "input": "wafer",
                    "product": d["product"],
                    "tau": int(TAU_FAB[row.cls].value),
                    "w_scr": W_SCR,
                }
            elif d["type"] == "osat":
                entry["osat"] = {"thr": d["thr"], "tau": TAU_OSAT, "packages": dict(d["packages"])}
            elif d["type"] == "sink":
                entry["sink"] = {
                    "demand": {
                        k: {
                            "dbar": dbar.get((n, k), 0.0),
                            "pi": pi.get((n, k), 0.0),
                            "backlog": False,
                            "phi": 0.5,
                            "sigma": 0.1,
                            "seasonal": None,
                            "shock": [1.0, 1.0, 1.0],
                        }
                        for k in ("chip_le", "chip_mat")
                    }
                }
            nodes.append(entry)
        edges = []
        for a, b, eid, data in g.edges(keys=True, data=True):
            u0 = data["u0"]
            if eid in bypass_u0:
                u0 = bypass_u0[eid]
            edges.append(
                {
                    "id": eid,
                    "tail": a,
                    "head": b,
                    "mode": data["mode"],
                    "K": list(data["K"]),
                    "tau": int(data["tau"]),
                    "c0": float(data["c0"]),
                    "u0": None if data["mode"] == "grid" else float(u0),
                    "alt_of": None if data.get("alt_of") is None else dict(data["alt_of"]),
                    "pool": data["pool"],
                }
            )
        fleet = {"tb": 0.0, "ct": 0.0}
        for e in edges:
            if e["mode"] == "sea":
                fleet[e["pool"]] += e["tau"] * e["u0"]  # F^b of (7), proposal K3
        routing: dict[tuple[str, str], list[str]] = {}
        for ln in lanes:
            tail, head = ends[ln["edges"][0]][0], ends[ln["edges"][-1]][1]
            routing.setdefault((g.nodes[tail]["region"], g.nodes[head]["region"]), []).append(ln["id"])
        compat = [{"source": a, "terminal": b} for a, b in _pairs(g, lanes, "source", "terminal")]
        use = [{"material": a, "fab": b} for a, b in _pairs(g, lanes, "material", "fab")]
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_id": size.instance_id,
            "kind": size.kind,
            "T": size.T,
            "units": {
                **{k: ("GWh fuel" if b == "tb" else "wafer-eq 300 mm") for k, b, _ in COMMODITIES},
                "cost": "USD",
            },
            "regions": list(REGIONS),
            "region_class": {r: "default" for r in REGIONS},
            "params": {
                "psi": 0.6,
                "alpha_max": 1.25,
                "tau_alpha": 52.0,
                "upsilon": UPSILON,
                "fleet_share": {"tb": 0.05, "ct": 0.06},
                "fleet_measure": fleet,
                "top_tariff": TOP_TARIFF,
                "forecast_shares": [1.0] + [0.0] * 8,
            },
            "commodities": [
                {
                    "id": k,
                    "unit": "GWh fuel" if b == "tb" else "wafer-eq 300 mm",
                    "pool": b,
                    "v": VALUES[k].value,
                    "override": ov,
                    "disposal_cost": 0.1 * VALUES[k].value,
                }
                for k, b, ov in COMMODITIES
            ],
            "nodes": nodes,
            "edges": edges,
            "lanes": [
                {
                    "id": ln["id"],
                    "edges": list(ln["edges"]),
                    "chokepoints": list(ln["chokepoints"]),
                    "alt_of": ln["alt_of"],
                }
                for ln in lanes
            ],
            "routing_table": [{"origin": o, "dest": d, "lanes": ls} for (o, d), ls in routing.items()],
            "compatibility": compat,
            "use": use,
            "chokepoint_adjacency": {c: list(tables.chokepoint_adjacency[c]) for c in chk_nodes},
            "trade_adjacency": [],
            "initial_state": initial,
            "prohibitions_at_reset": [{"edge": e, "k": k} for e, k in g.graph["prohibited"]],
            "provenance": {},
        }

    empty = {"stock": [], "pipeline": [], "fab_wip": [], "osat_wip": [], "queue_lots": []}
    mu = {(c, b): MU_UNBOUNDED for c in chk_nodes for b in ("tb", "ct")}
    bypass_u0 = {eid: 0.0 for eid, _ in g.graph["bypass"]}
    previous = None
    for _ in range(FIXED_POINT_ITERATIONS):  # mu and the bypass capacities at a fixed point of (22) (§12 M5)
        inst = load_instance(assemble(mu, bypass_u0, {}, {}, {}, empty, {}), strict=False)
        nf = nominal_flow(inst)
        routed: dict[tuple[str, str], float] = defaultdict(float)
        for r in nf.routes:
            if r.lane is not None:
                for c in inst.lanes[r.lane].chokepoints:
                    routed[(inst.nodes[c].id, inst.commodities[r.k].pool)] += r.flow
        crude = inst.commodity_index["crude"]
        lane_crude = defaultdict(float)
        for r in nf.routes:
            if r.lane is not None and r.k == crude:
                lane_crude[inst.lanes[r.lane].id] += r.flow
        new_mu = {
            (c, b): max(routed.get((c, b), 0.0), g.graph["planned_mu"].get((c, b), 0.0))
            + g.graph["dup_mu"].get((c, b), 0.0)
            for c in chk_nodes
            for b in ("tb", "ct")
        }
        new_bypass = dict.fromkeys(bypass_u0, 0.0)
        for eid, lane in g.graph["bypass"]:  # a bypass edge shared by several lanes carries the sum of their shares
            new_bypass[eid] += BYPASS_SHARE.value * lane_crude.get(lane, 0.0)
        state = (new_mu, new_bypass)
        if previous is not None and _close(previous, state):
            break
        previous = state
        mu, bypass_u0 = new_mu, new_bypass
    else:
        raise ValueError(f"{size.kind}: mu did not reach a fixed point of (22) in {FIXED_POINT_ITERATIONS} solves")
    for eid in bypass_u0:
        lanes_of = ", ".join(lane for e, lane in g.graph["bypass"] if e == eid)
        crude_flow = bypass_u0[eid] / BYPASS_SHARE.value
        explicit[f"edges/{eid}/u0"] = (
            "DERIVED(instance)",
            f"12.4 % REAL(EIA 2025) x the nominal crude flow (22) of {lanes_of}; range 12.4-22.5 % (§11 row 63)",
            BYPASS_RANGE[0] * crude_flow,
            BYPASS_RANGE[1] * crude_flow,
        )
    inst = load_instance(assemble(mu, bypass_u0, {}, {}, {}, empty, {}), strict=False)
    nu = {(inst.nodes[i].id, inst.commodities[k].id): v for (i, k), v in _salvage(inst).items()}
    nf = nominal_flow(inst)
    dbar = {(inst.nodes[i].id, inst.commodities[k].id): UPSILON * f for (i, k), f in nf.sink_flow.items()}  # (22)
    inst = load_instance(assemble(mu, bypass_u0, nu, dbar, {}, empty, {}), strict=False)
    pi = {
        (inst.nodes[d.node].id, inst.commodities[d.k].id): 10
        * (_landed_cost(inst, d.node, d.k) + TOP_TARIFF * inst.commodities[d.k].v)  # Q85
        for d in inst.demands
    }
    inst = load_instance(assemble(mu, bypass_u0, nu, dbar, pi, empty, {}), strict=False)
    initial, throughput, blocks = _warm_blocks(inst, size, warm_fq)
    warm: dict[tuple[str, str], float] = defaultdict(float)  # the largest block per slot (§2.3; M5-O37 (b))
    for block in blocks:
        held: dict[tuple[str, str], float] = defaultdict(float)
        for s in block:
            held[(s["node"], s["k"])] += s["qty"]
        for key, q in held.items():
            warm[key] = max(warm[key], q)
    storage = {}
    for n in nodes_in_order:
        if g.nodes[n]["type"] == "chokepoint":
            continue
        for k in slots[n]:
            storage[(n, k)] = max(STORAGE_WEEKS * throughput.get((n, k), 0.0), 2.0 * warm.get((n, k), 0.0), 1.0)
    raw = assemble(mu, bypass_u0, nu, dbar, pi, initial, storage)
    explicit.update(_sourced_provenance(g, mu))
    explicit.update(_buffer_provenance(g, tables))
    tags = _tags(size, POINT_MASS_NOTE if warm_note is None else warm_note)
    for f in (n for n in nodes_in_order if g.nodes[n]["type"] == "fab"):
        row = g.nodes[f]["row"]
        e = E_F[
            "memory_kr"
            if row.cls == "memory" and row.region == "KR"
            else "memory_jp"
            if row.cls == "memory"
            else row.cls
        ]
        explicit[f"nodes/{f}/fab/e"] = (e.tag, e.source, None, None)
        explicit[f"nodes/{f}/fab/tau"] = (TAU_FAB[row.cls].tag, TAU_FAB[row.cls].source, None, None)
        cap = tables.fab_cap0[f]
        explicit[f"nodes/{f}/fab/cap0"] = (cap.tag, cap.source, cap.lo, cap.hi)
    for path in leaf_paths(raw):
        if path in explicit:
            tag, source, lo, hi = explicit[path]
        else:
            tag, source = next(((t, s) for p, t, s in tags if fnmatch(path, p)), (PLACEHOLDER, M5_SOURCE))
            lo = hi = None
        raw["provenance"][path] = {"tag": tag, "source": source, "lo": lo, "hi": hi}
    return raw


def _per_rung(warm_fq: Mapping | None) -> dict[float, Mapping[FqKey, float]] | None:
    """``warm_fq`` as rung -> quantiles when it is keyed by γ rung (a warm start per rung), else None."""
    if not warm_fq or not all(isinstance(key, float) for key in warm_fq):
        return None
    return {float(gamma): q for gamma, q in warm_fq.items()}


def _warm_blocks(inst: Instance, size: SizeSpec, warm_fq: Mapping | None) -> tuple[dict, dict, list[list[dict]]]:
    """(``initial_state``, slot throughputs, every stock block) of the warm start (§2.3), one start or one per rung.

    Per rung (``_per_rung``), ``warm_initial_state`` runs once on each rung's quantiles: the anchor rung's block is
    ``stock`` with ``rung``, the others go to ``stock_by_rung`` by the rung's repr (M5-O37 (b)).

    Raises:
        ValueError: if a warm start per rung lacks the anchor rung, or a rung's pipeline, WIP, queues or throughputs
            differ from the anchor rung's (F_Q moves only the closure terms of naive's levels).

    """
    from shockbench_flow.disruption.params import RUNGS

    per_rung = _per_rung(warm_fq)
    if per_rung is None:
        initial, throughput = warm_initial_state(inst, None if warm_fq is None else fq_by_index(inst, warm_fq))
        return initial, throughput, [initial["stock"]]
    anchor = RUNGS[0]
    if anchor not in per_rung:
        raise ValueError(f"{size.kind}: a warm start per rung needs the anchor rung {anchor}'s F_Q (§2.3)")
    states = {gamma: warm_initial_state(inst, fq_by_index(inst, q)) for gamma, q in sorted(per_rung.items())}
    initial, throughput = states[anchor]
    for gamma, (block, thr) in states.items():
        if thr != throughput or any(block[p] != initial[p] for p in ("pipeline", "fab_wip", "osat_wip", "queue_lots")):
            raise ValueError(
                f"{size.kind}: the warm start at rung {gamma!r} moves more than the stock (F_Q moves only the closure"
                " terms of naive's levels, §2.3)"
            )
    initial = {
        **initial,
        "rung": anchor,
        "stock_by_rung": {repr(gamma): block["stock"] for gamma, (block, _) in states.items() if gamma != anchor},
    }
    return initial, throughput, [block["stock"] for block, _ in states.values()]


def _pairs(g: nx.MultiDiGraph, lanes: list[dict], ta: str, tb: str) -> list[tuple[str, str]]:
    """(tail, head) pairs of type ``ta`` -> ``tb`` joined by an edge or a lane, in first-seen order (§2.3 tables)."""
    seen: dict[tuple[str, str], None] = {}
    ends = {key: (a, b) for a, b, key in g.edges(keys=True)}
    for a, b, key in g.edges(keys=True):
        if g.nodes[a]["type"] == ta and g.nodes[b]["type"] == tb:
            seen.setdefault((a, b), None)
    for ln in lanes:
        a, b = ends[ln["edges"][0]][0], ends[ln["edges"][-1]][1]
        if g.nodes[a]["type"] == ta and g.nodes[b]["type"] == tb:
            seen.setdefault((a, b), None)
    return list(seen)


def _close(a, b) -> bool:
    """Two (mu, bypass) states agree within the fixed point's relative tolerance."""
    for da, db in zip(a, b):
        for key in da:
            x, y = da[key], db[key]
            if abs(x - y) > FIXED_POINT_RTOL * max(1.0, abs(x), abs(y)):
                return False
    return True


# ----- the warm start's F_Q (design §12 'Warm start on the anchor rung's F_Q', 'Warm start per rung') ----------------
WARM_FQ_DIR = DATA_DIR / "warm_fq"  # <size>_g<rung>.json: naive's F_Q quantiles at one γ rung, the warm start's input
WARM_FQ_FORMAT = "shockbench-flow warm-start F_Q v2"  # v2: the profile's Hawkes baselines recorded (DET-M5-1)
# the recorded Hawkes baselines against this machine's (``recorded_profile``): another machine's LAPACK rounds the (32)
# calibration's last bits otherwise (about 1.8e-14 relative on the AMD VM, DET-M5-1; Q94); SYNTHETIC, the tolerance of
# the builder's own reproduction test (tests/test_flagship_instance.py)
WARM_FQ_RTOL = 1e-9
WARM_FQ_COMMAND = "uv run python -m shockbench_flow.instance.flagship --warm-fq small full"
FqKey = tuple[str, str, str, str]  # (chokepoint id, pool, destination node id, commodity id)
POINT_MASS_NOTE = (
    "warm start of §2.3 on naive's plan with F_Q the point mass at 0 (§12 'Warm start (M5 reading)'; the anchor"
    f" rung's F_Q replaces it once `{WARM_FQ_COMMAND}` writes data/warm_fq/, §12 'Warm start on the anchor rung's F_Q')"
)
GIVEN_NOTE = "warm start of §2.3 on naive's plan with F_Q quantiles given to the builder (no committed F_Q file)"


def fq_by_id(inst: Instance, quantiles: Mapping[tuple, float]) -> dict[FqKey, float]:
    """Naive's F_Q quantiles keyed (c, pool, j, k) by node and commodity index, re-keyed by ids, order kept."""
    nodes, K = inst.nodes, inst.commodities
    return {(nodes[c].id, pool, nodes[j].id, K[k].id): q for (c, pool, j, k), q in quantiles.items()}


def fq_by_index(inst: Instance, quantiles: Mapping[FqKey, float]) -> dict[tuple, float]:
    """``quantiles`` keyed by ids (``fq_by_id``) re-keyed by index on ``inst``, as ``naive_plan`` takes them.

    Raises:
        ValueError: if an id is not in ``inst``, or the keys are not exactly ``naive_fq.quantile_keys(inst)``: the
            plan's lane routes through a (chokepoint, pool) pair with a law of (66).

    """
    from shockbench_flow.policies.naive_fq import quantile_keys

    try:
        out = {
            (inst.node_index[c], pool, inst.node_index[j], inst.commodity_index[k]): q
            for (c, pool, j, k), q in quantiles.items()
        }
    except KeyError as err:
        raise ValueError(f"warm-start F_Q: {inst.instance_id} has no {err.args[0]!r}") from err
    expected = quantile_keys(inst)
    if set(out) != expected:
        raise ValueError(
            f"warm-start F_Q: the keys are not the plan's lane routes through a pair with a law of {inst.instance_id}"
            f" ({len(set(out) - expected)} unknown, {len(expected - set(out))} missing); rerun `{WARM_FQ_COMMAND}`"
        )
    return out


def warm_fq_path(size: str, rung: float) -> Path:
    """The F_Q file of ``size`` at γ rung ``rung``: ``WARM_FQ_DIR/<size>_g<repr(rung)>.json`` (§12)."""
    return WARM_FQ_DIR / f"{size}_g{rung!r}.json"


def rung_fq(
    inst: Instance, rung: float, replications: int | None = None, n_jobs: int = 1, *, sampler: Callable | None = None
) -> dict[FqKey, float]:
    """Naive's F_Q quantiles (67) at γ rung ``rung`` of the flagship profile, keyed by ids, in plan order (§8.1).

    ``naive_fq.generator_quantiles(inst, flagship_profile(inst, rung), replications, n_jobs)``: the quantiles the
    anchor of (55) takes on generated omega at that rung, from stream 22's public replications (``replications`` None:
    the design's ``naive_fq.REPLICATIONS``). ``sampler`` stands in for the generator (tests: the same critical quantiles
    of ``naive_fq.fq_laws`` with that sampler).
    """
    from shockbench_flow.disruption.profiles import flagship_profile
    from shockbench_flow.policies import naive_fq
    from shockbench_flow.policies.naive import canonical_quantiles

    replications = naive_fq.REPLICATIONS if replications is None else replications
    params = flagship_profile(inst, rung)
    if sampler is None:
        q = naive_fq.generator_quantiles(inst, params, replications, n_jobs)
    else:
        laws = naive_fq.fq_laws(inst, params, replications, n_jobs, sampler=sampler)
        q = canonical_quantiles(naive_fq.critical_quantiles(inst, laws))
    return fq_by_id(inst, q)


def anchor_fq(
    inst: Instance, replications: int | None = None, n_jobs: int = 1, *, sampler: Callable | None = None
) -> dict[FqKey, float]:
    """``rung_fq`` at the anchor rung (``profiles.ANCHOR_GAMMA``): the quantiles of the anchor rung's block (§8.1)."""
    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA

    return rung_fq(inst, ANCHOR_GAMMA, replications, n_jobs, sampler=sampler)


def warm_fq_record(inst0: Instance, fq: Mapping[FqKey, float], replications: int, rung: float | None = None) -> dict:
    """The F_Q file's content: ``fq`` computed on the point-mass build ``inst0``, with what ``check_warm_fq`` checks.

    Records the format, the kind, the rung (``rung`` None: the anchor rung), the replication count, ``inst0``'s hash,
    that rung's flagship generator's ``generator_id`` on ``inst0`` and its Hawkes baselines (region, policy,
    militarised: the values the (32) calibration sets, ``recorded_profile``), the command and this machine's platform
    (Q94), and the quantiles in plan order.
    """
    from shockbench_flow.disruption.params import generator_id
    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA, flagship_profile

    rung = ANCHOR_GAMMA if rung is None else rung
    params = flagship_profile(inst0, rung)
    return {
        "format": WARM_FQ_FORMAT,
        "kind": inst0.kind,
        "rung": rung,
        "replications": replications,
        "point_mass_hash": inst0.hash,
        "generator_id": generator_id(params, inst0),
        "hawkes_baselines": [[r, bp, bm] for r, bp, bm in params.hawkes.baselines],
        "command": WARM_FQ_COMMAND,
        "platform": platform.platform(),
        "quantiles": [{"chokepoint": c, "pool": b, "node": j, "k": k, "q": q} for (c, b, j, k), q in fq.items()],
    }


def recorded_profile(record: Mapping, inst0: Instance, rung: float | None = None):
    """This machine's flagship profile of ``inst0`` at ``rung`` with the Hawkes baselines an F_Q file records.

    The (32) calibration of the chokepoint-adjacent militarised baselines (``profiles.calibrate_flagship_closures``) is
    the one value of the profile another machine's LAPACK rounds differently in its last bits (Q94; DET-M5-1: about
    1.8e-14 relative on the AMD VM, the scoring host's type); every other value is exact. The recorded baselines
    must name this machine's regions in its order and lie within a relative ``WARM_FQ_RTOL`` of its values; the
    profile with them in place is the generator the file was computed on, so its ``generator_id`` is the record's on
    the machine that wrote it and on any other.

    Raises:
        ValueError: if the record carries no baselines, other regions, or values beyond the tolerance.

    """
    import dataclasses

    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA, flagship_profile

    rung = ANCHOR_GAMMA if rung is None else rung
    params = flagship_profile(inst0, rung)
    here, got = params.hawkes.baselines, record.get("hawkes_baselines")
    rerun = f"; rerun `{WARM_FQ_COMMAND}`"
    if (
        not isinstance(got, list)
        or len(got) != len(here)
        or any(not isinstance(g, list) or len(g) != 3 or g[0] != h[0] for g, h in zip(got, here))
    ):
        raise ValueError(f"warm-start F_Q: Hawkes baselines {got!r} are not the profile's regions in order{rerun}")
    for (r, *vals), (_, *mine) in zip(got, here):
        for v, w in zip(vals, mine):
            if type(v) is not float or not abs(v - w) <= WARM_FQ_RTOL * max(abs(v), abs(w)):
                raise ValueError(
                    f"warm-start F_Q: Hawkes baselines of {r} {vals} beyond a relative {WARM_FQ_RTOL:g} of this"
                    f" machine's {mine} (the calibration (32) changed){rerun}"
                )
    baselines = tuple((r, bp, bm) for r, bp, bm in got)
    return dataclasses.replace(params, hawkes=dataclasses.replace(params.hawkes, baselines=baselines))


def check_warm_fq(record: Mapping, inst0: Instance, rung: float | None = None) -> dict[FqKey, float]:
    """The quantiles of an F_Q file (``warm_fq_record``) once checked against the point-mass build ``inst0``.

    The file must be of this format, kind and rung (``rung`` None: the anchor rung), name an integer count >= 1, carry
    ``inst0``'s hash (a change of any table moves it: the file is then stale) and that rung's flagship generator: its
    Hawkes baselines within ``WARM_FQ_RTOL`` of this machine's and its ``generator_id`` that of this machine's profile
    with them in place (``recorded_profile``; Q94, DET-M5-1), hold a float >= 0 per key, and its keys must be the
    plan's lane routes (``fq_by_index``).

    Raises:
        ValueError: on any failed check, naming the regeneration command.

    """
    from shockbench_flow.disruption.params import generator_id
    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA

    rung = ANCHOR_GAMMA if rung is None else rung
    rerun = f"; rerun `{WARM_FQ_COMMAND}`"
    if record.get("format") != WARM_FQ_FORMAT:
        raise ValueError(f"warm-start F_Q: format {record.get('format')!r}, not {WARM_FQ_FORMAT!r}{rerun}")
    if record.get("kind") != inst0.kind:
        raise ValueError(f"warm-start F_Q: kind {record.get('kind')!r} for a {inst0.kind!r} instance{rerun}")
    if record.get("rung") != rung:
        raise ValueError(f"warm-start F_Q: rung {record.get('rung')!r} in the file of rung {rung!r}{rerun}")
    n = record.get("replications")
    if type(n) is not int or n < 1:
        raise ValueError(f"warm-start F_Q: replications must be an integer >= 1, got {n!r}{rerun}")
    if record.get("point_mass_hash") != inst0.hash:
        raise ValueError(
            f"warm-start F_Q: computed on a point-mass build of hash {record.get('point_mass_hash')!r}, this one is"
            f" {inst0.hash} (a table changed, or this machine's HiGHS rounds differently, Q94){rerun}"
        )
    gid = generator_id(recorded_profile(record, inst0, rung), inst0)
    if record.get("generator_id") != gid:
        raise ValueError(
            f"warm-start F_Q: another generator than the flagship profile's at {rung!r}, {gid[:12]}...{rerun}"
        )
    fq: dict[FqKey, float] = {}
    for e in record.get("quantiles", ()):
        q = e["q"]
        if type(q) is not float or not math.isfinite(q) or q < 0.0:
            raise ValueError(f"warm-start F_Q: quantile {q!r} at {e!r} is not a finite float >= 0{rerun}")
        fq[(e["chokepoint"], e["pool"], e["node"], e["k"])] = q
    if len(fq) != len(record.get("quantiles", ())):
        raise ValueError(f"warm-start F_Q: a key appears twice among the quantiles{rerun}")
    fq_by_index(inst0, fq)
    return fq


def read_warm_fq(size: str, rung: float | None = None) -> dict | None:
    """The committed F_Q file of ``size`` at ``rung`` (None: the anchor rung; ``warm_fq_path``), or None without one."""
    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA

    path = warm_fq_path(size, ANCHOR_GAMMA if rung is None else rung)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def warm_fq_note(record: Mapping, size: str) -> str:
    """The provenance source of the block built on the F_Q file ``record`` (its rung's)."""
    from shockbench_flow.disruption.profiles import ANCHOR_GAMMA

    n, rung = record["replications"], record["rung"]
    which = "the anchor rung" if rung == ANCHOR_GAMMA else "rung"
    return (
        f"warm start of §2.3 on naive's plan with the F_Q quantiles of {which} γ {rung!r}, {n:,}"
        f" replication{'s' if n != 1 else ''} of the flagship profile on stream 22 (data/warm_fq/{size}_g{rung!r}.json,"
        f" by `{WARM_FQ_COMMAND}`; §12 'Warm start per rung')"
    )


def warm_quantiles(inst: Instance, rung: float | None = None) -> dict[tuple, float] | None:
    """The F_Q quantiles the committed warm start of ``inst``'s kind was built on at a rung, keyed by index.

    ``rung`` None is the instance's own (``Instance.start_rung``; the anchor rung on an instance with one start); None
    is returned without that rung's file (the point mass). Naive with these quantiles starts in its event-free steady
    state from that rung's block (§2.3); the file's keys are checked against ``inst``'s plan (``fq_by_index``), not its
    hash (``inst`` is the F_Q build, not the point-mass one).
    """
    record = read_warm_fq(inst.kind, inst.start_rung if rung is None else rung)
    if record is None:
        return None
    return fq_by_index(inst, {(e["chokepoint"], e["pool"], e["node"], e["k"]): e["q"] for e in record["quantiles"]})


def build_flagship(
    size: str, warm_fq: Mapping[FqKey, float] | Mapping[float, Mapping[FqKey, float]] | str | None = "committed"
) -> dict:
    """The instance dict of `small` or `full` under the closed schema of §2.3, provenance on every leaf (V20).

    ``warm_fq`` sets the F_Q of the warm start: "committed" reads every rung's ``warm_fq_path(size, rung)`` and checks
    each against the point-mass build (``check_warm_fq``), a warm start per rung (§2.3; M5-O37 (b)), or takes the point
    mass when there is no file; None is the point mass; a mapping keyed by ids (``fq_by_id``) gives one start's
    quantiles directly, and one keyed by γ rung each rung's.

    Raises:
        KeyError: for a size other than ``small`` and ``full``.
        ValueError: for a stale or malformed committed F_Q file, a set of files missing a rung, or quantiles with other
            keys (``fq_by_index``).

    """
    from shockbench_flow.disruption.params import RUNGS

    spec = SIZES[size]
    tables = flagship_tables()
    g = build_graph(tables, spec)
    if warm_fq is None:
        return derive_instance_dict(g, tables, spec)
    if isinstance(warm_fq, str):
        if warm_fq != "committed":
            raise ValueError(f"warm_fq must be 'committed', None or a mapping, got {warm_fq!r}")
        records = {rung: read_warm_fq(size, rung) for rung in RUNGS}
        raw0 = derive_instance_dict(g, tables, spec)
        if all(record is None for record in records.values()):
            return raw0
        missing = [rung for rung, record in records.items() if record is None]
        if missing:
            raise ValueError(
                f"warm-start F_Q: no file for rung{'s' if len(missing) > 1 else ''} {missing} of {size}; a warm start"
                f" per rung needs every rung's (§2.3); rerun `{WARM_FQ_COMMAND}`"
            )
        inst0 = load_instance(raw0)
        fq = {rung: check_warm_fq(record, inst0, rung) for rung, record in records.items()}
        notes = {rung: warm_fq_note(record, size) for rung, record in records.items()}
        return derive_instance_dict(g, tables, spec, fq, notes)
    per_rung = _per_rung(warm_fq)
    if per_rung is not None:
        return derive_instance_dict(g, tables, spec, per_rung, dict.fromkeys(per_rung, GIVEN_NOTE))
    return derive_instance_dict(g, tables, spec, dict(warm_fq), GIVEN_NOTE)


def write_flagship(size: str) -> str:
    """Write ``data/<size>.json`` (on the committed F_Q files, if any) and return its instance hash (§2.3)."""
    return write_instance(build_flagship(size), DATA_DIR / f"{size}.json")


def write_warm_fq(
    size: str,
    replications: int | None = None,
    n_jobs: int = 1,
    *,
    sampler: Callable | None = None,
    rungs: tuple[float, ...] | None = None,
) -> dict[float, dict]:
    """Compute each rung's F_Q on the point-mass build of ``size`` and write ``warm_fq_path(size, rung)``.

    ``rungs`` None is every rung of ``params.RUNGS``. Returns the records written by rung (``warm_fq_record``);
    ``replications`` None is the design's count, ``sampler`` as in ``rung_fq``.
    """
    from shockbench_flow.disruption.params import RUNGS
    from shockbench_flow.policies.naive_fq import REPLICATIONS

    replications = REPLICATIONS if replications is None else replications
    inst0 = load_instance(build_flagship(size, warm_fq=None))
    records = {}
    for rung in RUNGS if rungs is None else rungs:
        fq = rung_fq(inst0, rung, replications, n_jobs, sampler=sampler)
        records[rung] = warm_fq_record(inst0, fq, replications, rung)
        WARM_FQ_DIR.mkdir(parents=True, exist_ok=True)
        text = json.dumps(records[rung], sort_keys=True, indent=1) + "\n"
        warm_fq_path(size, rung).write_text(text, encoding="utf-8")
    return records


def regenerate_warm_start(
    size: str,
    replications: int | None = None,
    n_jobs: int = 1,
    *,
    sampler: Callable | None = None,
    rungs: tuple[float, ...] | None = None,
) -> tuple[str, dict[float, dict]]:
    """``write_warm_fq`` of ``rungs``, ``write_flagship`` on every rung's file, then the one-pass fixed point.

    The fixed point is checked at every rung: F_Q of the written instance at the rung (``Instance.at_rung``, at the
    count its file names) equals that file's. Returns (the instance hash, every rung's F_Q record).

    Raises:
        ValueError: if a rung's file is missing, or F_Q recomputed on the written instance differs from the F_Q a block
            was built on.

    """
    from shockbench_flow.disruption.params import RUNGS

    write_warm_fq(size, replications, n_jobs, sampler=sampler, rungs=rungs)
    digest = write_flagship(size)
    inst1 = load_instance(DATA_DIR / f"{size}.json")
    records = {rung: read_warm_fq(size, rung) for rung in RUNGS}
    for rung, record in records.items():
        want = {(e["chokepoint"], e["pool"], e["node"], e["k"]): e["q"] for e in record["quantiles"]}
        if rung_fq(inst1.at_rung(rung), rung, record["replications"], n_jobs, sampler=sampler) != want:
            raise ValueError(
                f"{size}: F_Q of the warm start at rung {rung!r} is not the F_Q it was built on (no one-pass fixed"
                " point)"
            )
    return digest, records


# ----- structure: networkx view and report ------------------------------------------------------------------------
def to_networkx(inst: Instance) -> nx.MultiDiGraph:
    """A loaded instance as a MultiDiGraph (plots, structure checks): parallel arcs are distinct keys, the edge ids."""
    g = _nx().MultiDiGraph(instance_id=inst.instance_id, kind=inst.kind, T=inst.T)
    for n in inst.nodes:
        g.add_node(n.id, type=n.type, region=inst.regions[n.region])
    for e in inst.edges:
        g.add_edge(
            inst.nodes[e.tail].id,
            inst.nodes[e.head].id,
            key=e.id,
            mode=e.mode,
            K=tuple(inst.commodities[k].id for k in e.K),
            tau=e.tau,
            c0=e.c0,
            u0=e.u0,
            pool=e.pool,
            alt_of=None if e.alt_of is None else (e.alt_of.kind, e.alt_of.index),
        )
    g.graph["lanes"] = [
        {
            "id": ln.id,
            "edges": [inst.edges[j].id for j in ln.edges],
            "chokepoints": [inst.nodes[c].id for c in ln.chokepoints],
        }
        for ln in inst.lanes
    ]
    return g


def structure_report(inst: Instance) -> dict:
    """Counts and structural facts of an instance (§2.2): nodes and edges by type and mode, lanes, reachability."""
    g = to_networkx(inst)
    by_type: dict[str, int] = defaultdict(int)
    for n in inst.nodes:
        by_type[n.type] += 1
    by_mode: dict[str, int] = defaultdict(int)
    for e in inst.edges:
        by_mode[e.mode] += 1
    lanes_by_chokepoints: dict[int, int] = defaultdict(int)
    dup_lanes: dict[str, int] = defaultdict(int)
    for ln in inst.lanes:
        lanes_by_chokepoints[len(ln.chokepoints)] += 1
        dup_lanes[ln.id.split(".")[3] if ln.id.count(".") >= 3 else "primary"] += 1
    shipping = _nx().MultiDiGraph()
    shipping.add_nodes_from(g.nodes)
    shipping.add_edges_from((a, b, k) for a, b, k, d in g.edges(keys=True, data=True) if d["mode"] != "grid")
    compat: dict[str, set[str]] = defaultdict(set)
    for s, t in inst.compatibility:
        compat[inst.nodes[s].id].add(inst.regions[inst.nodes[t].region])
    return {
        "instance_id": inst.instance_id,
        "hash": inst.hash,
        "nodes": len(inst.nodes),
        "edges": len(inst.edges),
        "lanes": len(inst.lanes),
        "commodities": len(inst.commodities),
        "T": inst.T,
        "nodes_by_type": dict(by_type),
        "edges_by_mode": dict(by_mode),
        "lanes_by_chokepoint_count": dict(sorted(lanes_by_chokepoints.items())),
        "lanes_by_kind": dict(dup_lanes),
        "weakly_connected_without_coupling": _nx().is_weakly_connected(shipping),
        "fabs_with_grid": sum(inst.nodes[f].fab.grid is not None for f in inst.fabs),
        "fabs": len(inst.fabs),
        "pools_per_edge": sorted({1 if e.pool else 0 for e in inst.edges}),
        "compatibility_regions": {s: sorted(r) for s, r in compat.items()},
        "prohibitions_at_reset": len(inst.prohibitions_at_reset),
        "placeholder_leaves": len(placeholder_leaves(dict(inst.raw))),
        "leaves": len(inst.provenance),
    }


USAGE = (
    "python -m shockbench_flow.instance.flagship [--warm-fq[=REPLICATIONS] [--rungs=G,...]] [--n-jobs=N] [small] [full]"
)


def main(argv: list[str]) -> None:
    """``python -m shockbench_flow.instance.flagship``: write each size's file and print its hash.

    Without ``--warm-fq`` the files are rebuilt on the committed F_Q files; with it, ``regenerate_warm_start`` first
    recomputes each size's F_Q at every rung, or at the rungs of ``--rungs=G,...`` (the design's 1,000 replications, or
    R of ``--warm-fq=R``, over ``--n-jobs`` joblib workers) and checks the one-pass fixed point at every rung. No size
    named: both.

    Raises:
        SystemExit: on an unknown argument, a malformed count, a rung outside ``params.RUNGS``, or ``--rungs`` without
            ``--warm-fq``.

    """
    from shockbench_flow.disruption.params import RUNGS

    sizes, replications, warm, n_jobs, rungs = [], None, False, 1, None
    try:
        for a in argv:
            if a == "--warm-fq":
                warm = True
            elif a.startswith("--warm-fq="):
                warm, replications = True, int(a.removeprefix("--warm-fq="))
            elif a.startswith("--n-jobs="):
                n_jobs = int(a.removeprefix("--n-jobs="))
            elif a.startswith("--rungs="):
                rungs = tuple(float(x) for x in a.removeprefix("--rungs=").split(","))
            elif a in SIZES:
                sizes.append(a)
            else:
                raise SystemExit(f"unknown argument {a!r}; usage: {USAGE}")
    except ValueError as err:
        raise SystemExit(f"{err}; usage: {USAGE}") from err
    if rungs is not None and (not warm or any(g not in RUNGS for g in rungs) or len(set(rungs)) != len(rungs)):
        raise SystemExit(f"--rungs names distinct rungs of {RUNGS} and needs --warm-fq; usage: {USAGE}")
    for size in sizes or ["small", "full"]:
        if not warm:
            print(size, write_flagship(size))
            continue
        digest, records = regenerate_warm_start(size, replications, n_jobs, rungs=rungs)
        print(size, digest, f"warm start per rung, point-mass build {records[RUNGS[0]]['point_mass_hash'][:12]}...:")
        for rung, record in records.items():
            positive = sum(e["q"] > 0.0 for e in record["quantiles"])
            print(
                f"  rung {rung!r}{' (written)' if rungs is None or rung in rungs else ''}: {record['replications']}"
                f" replications, {positive} of {len(record['quantiles'])} quantiles positive; one pass is the fixed"
                " point"
            )


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = [
    "BUFFER_DAYS",
    "NUCLEAR_COVER_DAYS",
    "SIZES",
    "WARM_FQ_DIR",
    "FlagshipTables",
    "NodeRow",
    "ScaleValue",
    "SizeSpec",
    "anchor_fq",
    "build_flagship",
    "build_graph",
    "check_warm_fq",
    "derive_instance_dict",
    "fab_osat_shares",
    "flagship_tables",
    "fq_by_id",
    "fq_by_index",
    "leg_tau",
    "read_warm_fq",
    "recorded_profile",
    "regenerate_warm_start",
    "route_chokepoints",
    "rung_fq",
    "sia_bcg_cells",
    "structure_report",
    "to_networkx",
    "warm_fq_record",
    "warm_initial_state",
    "warm_fq_path",
    "warm_quantiles",
    "write_flagship",
    "write_warm_fq",
]

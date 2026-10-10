"""`chokepoint-tiny` written out as an instance dict (design §2.4; Q70, Q82, Q85).

`build_tiny()` returns the instance dict whose canonical JSON is ``data/tiny.json``; a test keeps the two identical.
Every value follows design §2.4, computed with the same floating-point expressions as the design's reference code
(`scripts/python/evidence/tiny_fixture.py`, `tiny_derived.py`), so that the package reproduces the fixture's integer
cents. `tiny` is a correctness fixture, not a scoring benchmark (Q82): every value is SYNTHETIC(placeholder) unless its
provenance says otherwise.

Regenerate the file with ``uv run python -m shockbench_flow.instance.tiny``.
"""

from fnmatch import fnmatch

from shockbench_flow.instance.io import DATA_DIR, leaf_paths, write_instance
from shockbench_flow.instance.schema import SCHEMA_VERSION, UNMODELLED


T = 26
RATE = 0.0421  # Fed funds 2025 mean, REAL (W2-maritime §2.1)
MMBTU_PER_GWH = 3412.14  # unit definition
LNG_USD_MMBTU = 12.09  # IMF PNGASJPUSDM 2025 mean, via W2-maritime §2.1
V = {"lng": LNG_USD_MMBTU * MMBTU_PER_GWH, "wafer": 3000.0, "chip_le_raw": 20000.0, "chip_le": 20000.0}
H = {k: v * RATE / 52 for k, v in V.items()}  # h_k = v_k r / 52
HQ_RATIO_LNG = (8.2, 60.0, 215.0)  # h^Q/h for LNG by class none / Red Sea / Hormuz-2026, DERIVED(W2-maritime)
WR_LNG = (0.0, 1723.0, 6893.0)  # per-transit war risk, USD/GWh, DERIVED(W2-maritime)
WR_CHIP = (0.0, 0.5, 2.0)  # USD/wafer-eq, SYNTHETIC(placeholder) (§11 row 65)
TOP_TARIFF = 0.25  # the spec's chip_tariff_25 (§11 row 25)
PI = 10 * (36.0 + TOP_TARIFF * V["chip_le"])  # 10 x (air freight E14+E21+E24 + duty) = 50,360 (Q85)
# salvage at chk by the c^min rule of §11 row 1 (Q93): nu_src_gulf + c0 of E0, nu_osat_sea + c0 of E22
NU_CHK = {"lng": 0.0 + 1000.0, "chip_le": 4.0 + 1.0}
VOLL = 100 * V["lng"]  # SYNTHETIC(placeholder) (Q28, Q82)
REGIONS = ["TW", "KR", "JP", "CN", "US", "EU", "GULF", "RU", "AU", "SEA", "IN", "UA", "KZ", "ROW"]  # spec §4.1

# edges of §2.4: id, tail, head, mode, K, tau, c0, u0, alt_of
EDGES = [
    ("E0", "src_gulf", "chk", "sea", "lng", 1, 1000.0, 30.0, None),
    ("E1", "chk", "grid_tw", "sea", "lng", 2, 1000.0, 20.0, None),
    ("E2", "chk", "grid_eu", "sea", "lng", 3, 1500.0, 16.0, None),
    ("E3", "src_gulf", "grid_eu", "sea", "lng", 6, 3500.0, 4 * 1.8, {"lane": "L1"}),
    ("E4", "chk", "grid_eu", "sea", "lng", 5, 2100.0, 4 * 1.8, {"edge": "E2"}),
    ("E5", "src_usau", "grid_tw", "sea", "lng", 3, 2500.0, 12.0, None),
    ("E6", "src_usau", "grid_eu", "sea", "lng", 2, 3000.0, 12.0, None),
    ("E7", "src_ru", "grid_eu", "pipeline", "lng", 1, 500.0, 12.0, None),
    ("E8", "src_ru", "chk", "sea", "lng", 1, 1200.0, 8.0, None),
    ("E9", "mat_jp", "fab_tw", "sea", "wafer", 1, 2.0, 1200.0, None),
    ("E10", "mat_jp", "fab_tw", "air", "wafer", 1, 8.0, 300.0, {"edge": "E9"}),
    ("E11", "mat_jp", "fab_cn", "sea", "wafer", 1, 2.0, 1200.0, None),
    ("E12", "mat_jp", "fab_cn", "air", "wafer", 1, 8.0, 300.0, {"edge": "E11"}),
    ("E13", "mat_jp", "fab_us", "sea", "wafer", 3, 3.0, 600.0, None),
    ("E14", "mat_jp", "fab_us", "air", "wafer", 1, 12.0, 150.0, {"edge": "E13"}),
    ("E15", "grid_tw", "fab_tw", "grid", None, 0, 0.0, None, None),
    ("E16", "grid_eu", "fab_us", "grid", None, 0, 0.0, None, None),
    ("E17", "fab_tw", "osat_sea", "sea", "chip_le_raw", 1, 2.0, 1200.0, None),
    ("E18", "fab_tw", "osat_sea", "air", "chip_le_raw", 1, 8.0, 300.0, {"edge": "E17"}),
    ("E19", "fab_cn", "osat_sea", "sea", "chip_le_raw", 1, 2.0, 1200.0, None),
    ("E20", "fab_us", "osat_sea", "sea", "chip_le_raw", 4, 3.0, 600.0, None),
    ("E21", "fab_us", "osat_sea", "air", "chip_le_raw", 1, 12.0, 150.0, {"edge": "E20"}),
    ("E22", "osat_sea", "chk", "sea", "chip_le", 1, 1.0, 3000.0, None),
    ("E23", "chk", "sink_us", "sea", "chip_le", 3, 2.0, 3000.0, None),
    ("E24", "osat_sea", "sink_us", "air", "chip_le", 1, 12.0, 750.0, {"lane": "L3"}),
]
LANES = [("L0", ["E0", "E1"]), ("L1", ["E0", "E2"]), ("L2", ["E8", "E1"]), ("L3", ["E22", "E23"])]
POOL = {"lng": "tb", "wafer": "ct", "chip_le_raw": "ct", "chip_le": "ct"}
LOTS = {"fab_tw": 900.0, "fab_cn": 900.0, "fab_us": 450.0}  # nominal lots at upsilon cap^0
PIPE0 = [  # per week of dispatch, for tau_e weeks (§2.4 table)
    ("E0", "L0", 11.0),
    ("E0", "L1", 12.3),
    ("E1", "L0", 11.0),
    ("E2", "L1", 12.3),
    ("E9", None, 900.0),
    ("E11", None, 900.0),
    ("E13", None, 450.0),
    ("E17", None, 900.0),
    ("E19", None, 900.0),
    ("E20", None, 450.0),
    ("E22", "L3", 2250.0),
    ("E23", "L3", 2250.0),
]

# provenance: explicit tags by path pattern (first match wins); everything else SYNTHETIC(placeholder), §2.4
_TAGS = [
    ("T", "SYNTHETIC(prior: spec §4.8)", "design §2.4: T = 26 on tiny"),
    ("commodities/lng/v", "DERIVED(IMF PNGASJPUSDM 2025 via W2-maritime)", "12.09 $/MMBtu x 3,412.14 MMBtu/GWh"),
    ("commodities/lng/disposal_cost", "SYNTHETIC(placeholder)", "0.1 v_k (§11 row 5)"),
    ("nodes/chk/chokepoint/k_c", "DERIVED(PortWatch)", "Suez 2021 realised 1.24-1.28 (Q7)"),
    ("nodes/chk/chokepoint/mu/*", "DERIVED(instance)", "nominal routed flow through chk (§2.4)"),
    (
        "nodes/chk/chokepoint/queue_holding/lng/*",
        "DERIVED(W2-maritime)",
        "h^Q/h 8.2 / 60 / 215 x h_lng (§2.4; W2-maritime §2.1)",
    ),
    ("nodes/chk/chokepoint/war_risk_cost/lng/*", "DERIVED(W2-maritime)", "0.505 and 2.02 $/MMBtu (§2.4)"),
    ("nodes/grid_tw/grid/days_cover/lng", "REAL", "Taiwan legal floor 11 d (W2-industrial §2.2)"),
    ("nodes/grid_tw/grid/ibar/lng", "DERIVED(instance)", "(16) D_g zeta G0 / 7 = 17.2857, written 17.29 (§2.4)"),
    ("nodes/grid_eu/grid/ibar/lng", "DERIVED(instance)", "(16) D_g zeta G0 / 7 = 24.60 (§2.4)"),
    ("nodes/fab_cn/fab/e", "DERIVED(instance)", "no grid: the fab's energy is outside the model (design §2.4)"),
    ("nodes/fab_*/fab/e", "SYNTHETIC(prior: imec N3 REAL, TSMC fleet DERIVED)", "2,000 kWh/wafer, W2-industrial §2.3"),
    ("nodes/sink_us/sink/demand/chip_le/pi", "SYNTHETIC(prior: spec §8.1)", "10 x (36 + 0.25 x 20,000) (Q85)"),
    ("nodes/sink_us/sink/demand/chip_le/dbar", "DERIVED(instance)", "(22) upsilon x 2,500 nominal delivered flow"),
    ("nodes/chk/stock/lng/salvage", "DERIVED(instance)", "c^min rule of §11 row 1 (Q93): src_gulf 0 + E0 1,000"),
    ("nodes/chk/stock/chip_le/salvage", "DERIVED(instance)", "c^min rule of §11 row 1 (Q93): osat_sea 4 + E22 1"),
    ("nodes/*/stock/*/salvage", "DERIVED(instance)", "c^min rule of §11 row 1; 0 at supply nodes (Q79)"),
    ("nodes/src_*/stock/lng/holding_cost", "DERIVED(instance)", "0 at supply nodes (Q79)"),
    (
        "nodes/chk/stock/*/holding_cost",
        "DERIVED(instance)",
        "0: queued cargo pays h^Q, not a holding cost ((23), §2.1)",
    ),
    ("nodes/mat_jp/stock/wafer/holding_cost", "DERIVED(instance)", "0 at supply nodes (Q79)"),
    ("nodes/*/stock/lng/holding_cost", "DERIVED(instance)", "h_lng = v_lng r / 52, r REAL(Fed funds 4.21 %)"),
    ("params/psi", "SYNTHETIC(prior: onset evidence)", "range DERIVED(MOEA, IEA, METI) 0.3-0.8 (Q36)"),
    ("params/alpha_max", "SYNTHETIC(prior: WPS 6047)", "§11 row 11"),
    ("params/tau_alpha", "SYNTHETIC(prior: WPS 6047)", "1 yr = 52 weeks, §11 row 11"),
    ("params/upsilon", "SYNTHETIC", "utilisation 0.9 (Q76 owner D2; §11 row 60)"),
    ("params/top_tariff", "SYNTHETIC(prior)", "spec chip_tariff_25 (§11 row 25)"),
    ("params/fleet_measure/*", "DERIVED(instance)", "F^b = sum of tau_e u0_e over the pool's sea edges (proposal K3)"),
    ("params/fleet_share/ct", "SYNTHETIC(prior: Alphaliner via gCaptain, Clarksons)", "0.06-0.12, lower end (Q49)"),
]


def _tag(path: str) -> tuple[str, str]:
    for pat, tag, source in _TAGS:
        if fnmatch(path, pat):
            return tag, source
    return "SYNTHETIC(placeholder)", "design §2.4 (Q85); tiny is a correctness fixture (Q82)"


def _stock(storage, holding, salvage, supply=None) -> dict:
    s = {"storage": storage, "holding_cost": holding, "salvage": salvage}
    if supply is not None:
        s["supply_rate"] = supply
    return s


def build_tiny() -> dict:
    """The `chokepoint-tiny` instance dict of design §2.4, with provenance for every leaf (V20).

    Every call returns fresh containers, none shared with the module's tables, so a caller's edit of one dict never
    reaches the next ``build_tiny()``.
    """
    lng, wafer, raw, chip = "lng", "wafer", "chip_le_raw", "chip_le"
    nodes = [
        {"id": "src_gulf", "type": "source", "region": "GULF", "stock": {lng: _stock(50.0, 0.0, 0.0, 25.0)}},
        {"id": "src_usau", "type": "source", "region": "US", "stock": {lng: _stock(50.0, 0.0, 0.0, 12.0)}},
        {"id": "src_ru", "type": "source", "region": "RU", "stock": {lng: _stock(50.0, 0.0, 0.0, 12.0)}},
        {
            "id": "chk",
            "type": "chokepoint",
            "region": "GULF",
            # nu = c^min x share 1 (§11 row 1; Q93): lng from src_gulf over E0 (1,000 < E8's 1,200 from src_ru), chips
            # from osat_sea (4) over E22 (1); holding 0, since queued cargo pays h^Q
            "stock": {lng: _stock(None, 0.0, NU_CHK[lng]), chip: _stock(None, 0.0, NU_CHK[chip])},
            "chokepoint": {
                "mu": {"tb": 23.3, "ct": 2250.0},
                "k_c": 1.3,
                "class": "tiny-aggregate",
                "queue_holding": {lng: [r * H[lng] for r in HQ_RATIO_LNG], chip: [H[chip]] * 3},
                "war_risk_cost": {lng: list(WR_LNG), chip: list(WR_CHIP)},
            },
        },
        {
            "id": "grid_tw",
            "type": "grid",
            "region": "TW",
            "stock": {lng: _stock(40.0, H[lng], 2000.0)},
            "grid": {
                "base_load": 20.0,
                "deliverable": 22.0,
                "shares": {lng: 0.5, UNMODELLED: 1 - 0.5},
                "days_cover": {lng: 11.0},
                "ibar": {lng: 17.29},
                "rationed": lng,
                "voll": VOLL,
                "priority": "base_first",
            },
        },
        {"id": "mat_jp", "type": "material", "region": "JP", "stock": {wafer: _stock(5000.0, 0.0, 0.0, 2750.0)}},
        {
            "id": "grid_eu",
            "type": "grid",
            "region": "EU",
            "stock": {lng: _stock(60.0, H[lng], 2500.0)},
            "grid": {
                "base_load": 40.0,
                "deliverable": 41.0,
                "shares": {lng: 0.3, UNMODELLED: 1 - 0.3},
                "days_cover": {lng: 14.0},
                "ibar": {lng: 24.60},
                "rationed": lng,
                "voll": VOLL,
                "priority": "base_first",
            },
        },
    ]
    fabs = [  # id, region, cap0, e, grid, class, tau, salvage
        ("fab_tw", "TW", 1000.0, 0.002, "grid_tw", "leading", 8, 2.0),
        ("fab_cn", "CN", 1000.0, 0.0, None, "mature", 6, 2.0),
        ("fab_us", "US", 500.0, 0.002, "grid_eu", "leading", 8, 3.0),
    ]
    for fid, region, cap, e, grid, cls, tau, nu in fabs:
        nodes.append(
            {
                "id": fid,
                "type": "fab",
                "region": region,
                "stock": {wafer: _stock(5000.0, H[wafer], nu), raw: _stock(5000.0, H[raw], nu)},
                "fab": {
                    "cap0": cap,
                    "e": e,
                    "grid": grid,
                    "class": cls,
                    "input": wafer,
                    "product": raw,
                    "tau": tau,
                    "w_scr": 2,
                },
            }
        )
    nodes.append(
        {
            "id": "osat_sea",
            "type": "osat",
            "region": "SEA",
            "stock": {raw: _stock(5000.0, H[raw], 4.0), chip: _stock(5000.0, H[chip], 4.0)},
            "osat": {"thr": 2500.0, "tau": 2, "packages": {raw: chip}},
        }
    )
    nodes.append(
        {
            "id": "sink_us",
            "type": "sink",
            "region": "US",
            "stock": {chip: _stock(10000.0, H[chip], 7.0)},
            "sink": {
                "demand": {
                    chip: {
                        "dbar": 0.9 * 2500,
                        "pi": PI,
                        "backlog": False,
                        "phi": 0.5,
                        "sigma": 0.1,
                        "seasonal": None,
                        "shock": [1.0, 1.0, 1.0],
                    }
                }
            },
        }
    )
    edges = []
    for eid, tail, head, mode, k, tau, c0, u0, alt in EDGES:
        edges.append(
            {
                "id": eid,
                "tail": tail,
                "head": head,
                "mode": mode,
                "K": [] if k is None else [k],
                "tau": tau,
                "c0": c0,
                "u0": u0,
                "alt_of": None if alt is None else dict(alt),
                "pool": None if k is None else POOL[k],
            }
        )
    tau_of = {e[0]: e[5] for e in EDGES}
    k_of = {e[0]: e[4] for e in EDGES}
    pipeline = []
    for eid, lane, q in PIPE0:
        tau = tau_of[eid]
        for a in range(1, tau + 1):
            pipeline.append(
                {"edge": eid, "k": k_of[eid], "lane": lane, "qty": q, "dispatch_week": a - tau, "arrival_week": a}
            )
    fab_tau = {fid: tau for fid, _region, _cap, _e, _grid, _cls, tau, _nu in fabs}
    raw_dict = {
        "schema_version": SCHEMA_VERSION,
        "instance_id": "chokepoint-tiny",
        "kind": "tiny",
        "T": T,
        "units": {
            lng: "GWh fuel",
            wafer: "wafer-eq 300 mm",
            raw: "wafer-eq 300 mm",
            chip: "wafer-eq 300 mm",
            "cost": "USD",
        },
        "regions": list(REGIONS),
        "region_class": {r: "default" for r in REGIONS},
        "params": {
            "psi": 0.6,
            "alpha_max": 1.25,
            "tau_alpha": 52.0,
            "upsilon": 0.9,
            "fleet_share": {"tb": 0.05, "ct": 0.06},
            "fleet_measure": {"tb": 265.2, "ct": 21000.0},
            "top_tariff": TOP_TARIFF,
            "forecast_shares": [1.0] + [0.0] * 8,
        },
        "commodities": [
            {
                "id": k,
                "unit": "GWh fuel" if k == lng else "wafer-eq 300 mm",
                "pool": POOL[k],
                "v": V[k],
                "override": k == lng,
                "disposal_cost": 0.1 * V[k],
            }
            for k in (lng, wafer, raw, chip)
        ],
        "nodes": nodes,
        "edges": edges,
        "lanes": [{"id": lid, "edges": list(es), "chokepoints": ["chk"], "alt_of": None} for lid, es in LANES],
        "routing_table": [
            {"origin": "GULF", "dest": "TW", "lanes": ["L0"]},
            {"origin": "GULF", "dest": "EU", "lanes": ["L1"]},
            {"origin": "RU", "dest": "TW", "lanes": ["L2"]},
            {"origin": "SEA", "dest": "US", "lanes": ["L3"]},
        ],
        "compatibility": [],
        "use": [{"material": "mat_jp", "fab": f} for f in ("fab_tw", "fab_cn", "fab_us")],
        "chokepoint_adjacency": {"chk": ["GULF"]},
        "trade_adjacency": [],
        "initial_state": {
            "stock": [
                {"node": n, "k": k, "qty": q}
                for n, k, q in (
                    ("src_gulf", lng, 25.0),
                    ("src_usau", lng, 12.0),
                    ("src_ru", lng, 12.0),
                    ("grid_tw", lng, 17.29),
                    ("mat_jp", wafer, 2750.0),
                    ("grid_eu", lng, 24.60),
                    ("fab_tw", wafer, 1800.0),
                    ("fab_cn", wafer, 1800.0),
                    ("fab_us", wafer, 900.0),
                    ("sink_us", chip, 4500.0),
                )
            ],
            "pipeline": pipeline,
            "fab_wip": [
                {"node": f, "k": raw, "qty": LOTS[f], "out_week": w} for f in LOTS for w in range(1, fab_tau[f] + 1)
            ],
            "osat_wip": [{"node": "osat_sea", "k": chip, "qty": 2250.0, "out_week": w} for w in (1, 2)],
            "queue_lots": [],
        },
        "prohibitions_at_reset": [{"edge": "E7", "k": lng}],
    }
    raw_dict["provenance"] = {}
    for path in leaf_paths(raw_dict):
        tag, source = _tag(path)
        raw_dict["provenance"][path] = {"tag": tag, "source": source, "lo": None, "hi": None}
    return raw_dict


def write_tiny() -> str:
    """Write ``data/tiny.json`` and return its instance hash."""
    return write_instance(build_tiny(), DATA_DIR / "tiny.json")


if __name__ == "__main__":
    print(write_tiny())

"""Generator parameters: the frozen tree that, with the instance and the entropy, fixes omega (design §4; Q18, Q87).

``GeneratorParams`` holds every knob of the disruption process of §4.2-4.5. Its canonical JSON with the instance hash
and ``schema_version`` is ``generator_id`` (§4.1 rule table), stored in omega's ``meta_generator_id`` and bound to the
omega hash. Field defaults are the design's generic values where it states one; everything instance-specific (active
regions, baselines, adjacency) comes from a profile (``disruption.profiles``).

Provenance is in field metadata (§4.1 rule table: tag, source, lo, hi): every leaf field of the tree carries
``provenance`` (the tag and the source in one string), ``tag``, ``source``, and ``lo`` and ``hi`` where the design
states a range for the value (None otherwise); ``provenance(params)`` lists them by leaf path. A leaf that bundles
values of several tags (a per-type law table, say) carries the weakest of them, SYNTHETIC(placeholder) below
SYNTHETIC(prior) below the sourced tags, so a seal that refuses placeholders (V23) refuses the bundle. The latent units
of the three signal-unit kinds are the classes ``RegionLatent``, ``DyadLatent`` and ``ChokepointLatent``, each with the
tags of its §2.4 row. ``MarkParams`` (of ``shockbench_flow.marks``) carries ``provenance`` only.

Region, node, edge and commodity references are instance *ids* (strings), resolved at sampling time, so a params tree
is readable and portable across instance files.
"""

import dataclasses
import math
from dataclasses import dataclass, field

from shockbench_flow.disruption.laws import Law, lead_law
from shockbench_flow.instance.io import sha256_hex
from shockbench_flow.instance.schema import SCHEMA_VERSION, Instance
from shockbench_flow.marks import MarkParams
from shockbench_flow.omega.codes import CHANNELS, TYPE_CHANNELS


# MID5 use-of-force durations (§4.4 "Material outage"): 25 % one day, else LN(ln 52 + 2 x 0.4307, 2.0) days, so the
# overall median is 52 days; SYNTHETIC(prior: MID5 dispute durations), as scripts/python/evidence/budget_rejection.py
MID5_FORCE_MU = math.log(52.0) + 2 * 0.4307
WEEKS_PER_YEAR = 52.0  # the design's year: Pi^c = exp(log P^yr / 52) of (28); the §2.4 rates per year over 52 weeks
RUNGS = (0.62, 0.79, 0.95, 0.97)  # the difficulty ladder of §2.6 (Q39): gamma of the policy block, anchor first


def _p(tag: str, source: str, lo: float | None = None, hi: float | None = None) -> dict:
    """The provenance record of one leaf field (§4.1 rule table): tag, source, lo, hi, and the two joined."""
    return {"provenance": f"{tag} {source}", "tag": tag, "source": source, "lo": lo, "hi": hi}


leaf_provenance = _p  # the public name of ``_p`` for other packages' leaf trees (``information.theta``; frozen for M3)


@dataclass(frozen=True)
class RegimeParams:
    """The two regime layers (28), the region classes and the dyads (Q32, Q59)."""

    P_yr: tuple[tuple[float, float, float], ...] = field(
        default=((0.9647, 0.0308, 0.0045), (0.1975, 0.7118, 0.0907), (0.0584, 0.2680, 0.6735)),
        metadata=_p("DERIVED(UCDP/PRIO v25.1)", "country-year chain none/minor/war (28)"),
    )
    class_multiplier: tuple[tuple[str, float], ...] = field(
        default=(("default", 1.0),),
        metadata=_p("SYNTHETIC(placeholder)", "m_cls per region class (§11 row 69; §2.4 'Region classes')"),
    )
    dyads: tuple[tuple[str, str], ...] = field(
        default=(("CN", "TW"),), metadata=_p("Q59", "CN-TW adopted (§11 row 69; §2.4 'Dyads')")
    )
    tension_uptime: float = field(
        default=0.8,
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "u_p, the share of time in the normal state (Tomlin uptime, (28)); tension about 20 % of the time at"
            " z_c = none (§11 row 22)",
        ),
    )
    tension_spell: float = field(default=26.0, metadata=_p("SYNTHETIC(placeholder)", "T-bar_p in weeks (§11 row 22)"))
    tension_entry_minor: float = field(default=2.0, metadata=_p("SYNTHETIC(placeholder)", "c_minor (§11 row 22)"))
    tension_entry_war: float = field(default=5.0, metadata=_p("SYNTHETIC(placeholder)", "c_war (§11 row 22)"))


_CALLER = "a latent unit built by the caller, no source (the profile's kinds are RegionLatent, DyadLatent, "


@dataclass(frozen=True)
class LatentParams:
    """One signal-unit kind's latent risk X and score noise W, (29) and (45) (Q51).

    Built directly, the values are the caller's (tagged SYNTHETIC(placeholder)); the profile uses the per-kind classes
    below, whose fields carry the tags of their §2.4 rows.
    """

    rho: float = field(metadata=_p("SYNTHETIC(placeholder)", _CALLER + "ChokepointLatent): rho_X (29)"))
    k: float = field(metadata=_p("SYNTHETIC(placeholder)", _CALLER + "ChokepointLatent): k_X (29)"))
    x_cap: float = field(metadata=_p("SYNTHETIC(placeholder)", _CALLER + "ChokepointLatent): x_cap (29)"))
    # the normaliser c_X; None = closed form E[exp(k min(X, x_cap))] under X ~ N(0, 1)
    c: float | None = field(metadata=_p("SYNTHETIC(placeholder)", _CALLER + "ChokepointLatent): c_X (29)"))


_MRS = "CALIBRATED(MRS 2024 via W2-information)"
_DYAD = "the region value transferred to dyads (§2.4 'Latent risk, regions and dyads')"
_CHK = "SYNTHETIC(prior: §5.2 illustrative persistence, region tilt)"


@dataclass(frozen=True)
class RegionLatent(LatentParams):
    """The region units' latent risk (§2.4 "Latent risk, regions and dyads"; (29))."""

    rho: float = field(default=0.9991, metadata=_p(_MRS, "rho_X of the region units (§2.4; §11 row 31)"))
    k: float = field(default=2.151, metadata=_p(_MRS, "k_X of the region units (§2.4)"))
    x_cap: float = field(default=1.128, metadata=_p(_MRS, "x_cap of the region units (§2.4)"))
    c: float | None = field(default=2.340, metadata=_p(_MRS, "c_X of the region units (§2.4)"))


@dataclass(frozen=True)
class DyadLatent(LatentParams):
    """The dyad units' latent risk: the region values transferred (§2.4 "Latent risk, regions and dyads"; Q59)."""

    rho: float = field(default=0.9991, metadata=_p("SYNTHETIC(prior)", "rho_X: " + _DYAD))
    k: float = field(default=2.151, metadata=_p("SYNTHETIC(prior)", "k_X: " + _DYAD))
    x_cap: float = field(default=1.128, metadata=_p("SYNTHETIC(prior)", "x_cap: " + _DYAD))
    c: float | None = field(default=2.340, metadata=_p("SYNTHETIC(prior)", "c_X: " + _DYAD))


@dataclass(frozen=True)
class ChokepointLatent(LatentParams):
    """The chokepoint units' latent risk (§2.4 "Latent risk, chokepoints"; §11 row 31)."""

    rho: float = field(default=0.95, metadata=_p(_CHK, "rho_X of the chokepoint units (§2.4; §11 row 31)"))
    k: float = field(default=2.151, metadata=_p(_CHK, "k_X of the chokepoint units, the region tilt (§2.4)"))
    x_cap: float = field(default=1.128, metadata=_p(_CHK, "x_cap of the chokepoint units, the region cap (§2.4)"))
    c: float | None = field(
        default=None,
        metadata=_p("DERIVED", "c_X = E exp(k_X min(X, x_cap)) under X ~ N(0, 1) in closed form (29), when None"),
    )


REGION_LATENT = RegionLatent()
DYAD_LATENT = DyadLatent()
CHOKEPOINT_LATENT = ChokepointLatent()


@dataclass(frozen=True)
class HawkesParams:
    """The two-block Hawkes process (30)-(31) and its baselines (Q5, Q39, Q52)."""

    gamma: float = field(
        default=RUNGS[0],
        metadata=_p("SYNTHETIC(prior)", "rung: spectral radius of the policy block, one of RUNGS (§2.6; §11 row 18)"),
    )
    gamma_M_max: float = field(
        default=0.79,
        metadata=_p(
            "SYNTHETIC(prior)",
            "cap of the militarised block (Q52; §11 row 18), inside the subcritical conflict fits, default 0.6-0.85,"
            " range 0.32-0.88 (R3 §3; §4.2)",
            0.32,
            0.88,
        ),
    )
    cross_PM: float = field(
        default=0.10, metadata=_p("SYNTHETIC(prior)", "policy children of a militarised event (31) (§11 row 18)")
    )
    cross_MP: float = field(
        default=0.02, metadata=_p("SYNTHETIC(prior)", "militarised children of a policy event (31) (§11 row 18)")
    )
    beta_inv: float = field(
        default=1.0,
        metadata=_p("SYNTHETIC(prior)", "kernel time 1/beta in weeks, 1-4 weeks (§11 row 19; §2.6)", 1.0, 4.0),
    )
    r_cross: float = field(
        default=0.15, metadata=_p("SYNTHETIC(prior: Tench Model 3)", "r_x, 0.15 (0.03-0.61) (§11 row 20)", 0.03, 0.61)
    )
    trade_adjacency: tuple[tuple[str, str, float], ...] = field(
        default=(),
        metadata=_p(
            "SYNTHETIC(prior)",
            "A, the instance's unless a profile sets it; `tiny`: 1 between regions an edge joins, DERIVED(instance) as"
            " a trade-and-tie measure (§2.4 'Trade adjacency'; R3 §3; §11 row 56)",
        ),
    )
    # per region: (policy baseline, militarised baseline) in events per week at the neutral regime (multiplier 1)
    baselines: tuple[tuple[str, float, float], ...] = field(
        default=(),
        metadata=_p(
            "SYNTHETIC(prior: budget_rejection.py, f5_realism.py)",
            "set by the profile: `tiny` 12/52 (1 - 0.62) events per week split policy 0.70 / militarised 0.30"
            " uniformly over the active regions; the militarised baseline of the chokepoint-adjacent regions DERIVED"
            " by the (32) calibration, run at every build by profiles.calibrate_closure_baselines"
            " (§2.4 'Baselines', 'Closure calibration'; §11 row 21)",
        ),
    )
    militarised_by_conflict: tuple[float, float, float] = field(
        default=(1.0, 2.0, 4.0),
        metadata=_p("SYNTHETIC(placeholder)", "lambda0_M multiplier by z^c none/minor/war (§2.4 'Baselines')"),
    )
    policy_by_tension: tuple[float, float] = field(
        default=(1.0, 2.0),
        metadata=_p("SYNTHETIC(placeholder)", "lambda0_P multiplier by z^p normal/tension (§2.4 'Baselines')"),
    )


@dataclass(frozen=True)
class TypeLaws:
    """Type shares per block (§4.4 table; SYNTHETIC(prior: budget_rejection.py)), renormalised per event (§2.4)."""

    policy: tuple[tuple[str, float], ...] = field(
        default=(("tariff", 0.50), ("sanction", 0.4167), ("material_outage", 0.0833)),
        metadata=_p("SYNTHETIC(prior: budget_rejection.py)", "policy-block type shares (§4.4; §2.4 'Type laws')"),
    )
    militarised: tuple[tuple[str, float], ...] = field(
        default=(
            ("militarised_closure", 0.375),
            ("regional_conflict", 0.125),
            ("piracy", 0.25),
            ("energy_shock", 0.25),
        ),
        metadata=_p(
            "SYNTHETIC(prior: budget_rejection.py)",
            "militarised-block type shares, energy shocks in M (§4.4; §2.4 'Type laws'; §11 row 21)",
        ),
    )


@dataclass(frozen=True)
class MarkLaws:
    """Durations, severities and rates per event type (§4.4; §2.4 `tiny` profile)."""

    duration: tuple[tuple[str, Law], ...] = field(
        default=(
            ("militarised_closure", Law("persistent_mix", (0.40, 4.519, 1.251, math.log(104.0), 1.0))),  # (39)
            ("regional_conflict", Law("lognormal_days", (5.531, 1.453))),
            ("tariff", Law("lognormal_days", (math.log(365.0), 1.0))),
            ("sanction", Law("one_day_or_lognormal_days", (0.25, MID5_FORCE_MU, 2.0))),
            ("material_outage", Law("one_day_or_lognormal_days", (0.25, MID5_FORCE_MU, 2.0))),
            ("piracy", Law("lognormal_days", (math.log(90.0), 1.0))),
            ("energy_shock", Law("lognormal_days", (math.log(14.0), 1.0))),
            ("weather_closure", Law("lognormal_days", (1.79, 0.80))),
        ),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "per type (§4.4; §2.4 'Durations'): closure (39) transient LN(4.519, 1.251) d DERIVED(MID5) prior only"
            " (§11 row 7), persistent weight 0.40 in [0.33, 0.50] (§2.6), median 104 wk (Q76 owner D4), spread 1.0"
            " SYNTHETIC(placeholder) (§11 row 67); regional conflict, sanction and material outage"
            " SYNTHETIC(prior: MID5 dispute durations) (§11 row 24); tariff, piracy, energy"
            " SYNTHETIC(prior: budget_rejection.py); weather LN(1.79, 0.80) d (Verschuur et al. 2020, §4.4)",
        ),
    )
    severity: tuple[tuple[str, Law], ...] = field(
        default=(
            ("militarised_closure", Law("uniform", (0.27, 0.97))),
            ("regional_conflict", Law("constant", (0.60,))),  # the fab sigma (§11 row 53, placeholder)
            ("tariff", Law("constant", (1.0,))),  # tariffs act through their rate, not a severity
            ("sanction", Law("constant", (1.0,))),
            ("material_outage", Law("constant", (1.0,))),
            ("piracy", Law("uniform", (0.08, 0.12))),
            ("energy_shock", Law("uniform", (0.1, 0.5))),
            ("weather_closure", Law("constant", (1.0,))),
        ),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "per type (§4.4; §2.4 'Severities'): closure U(0.27, 0.97) (PortWatch transit ratios), piracy"
            " U(0.08, 0.12) (Besley et al. 2015), energy U(0.1, 0.5) SYNTHETIC(prior: budget_rejection.py); sanction,"
            " material outage and weather 1; the regional-conflict fab sigma 0.60 SYNTHETIC(placeholder) (§11 row 53)",
        ),
    )
    tariff_rate_by_tension: tuple[float, float] = field(
        default=(0.10, 0.25),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "tariff rate by z^p: normal 0.10 SYNTHETIC(placeholder), tension 0.25 = chip_tariff_25 SYNTHETIC(prior)"
            " (§11 row 25; §2.4 'Severities')",
        ),
    )
    conflict_dead_time: Law = field(
        default=Law("uniform", (6.5, 12.5)),
        metadata=_p("SYNTHETIC(prior: W2-industrial §2.1)", "regime C dead time in weeks, 6.5-12.5 (§11 row 10)"),
    )
    conflict_tau_rho: Law = field(
        default=Law("uniform", (2.7, 14.0)),
        metadata=_p("SYNTHETIC(prior: W2-industrial §2.1)", "regime C tau_rho in weeks, 2.7-14 (§11 row 10)"),
    )
    conflict_restoration_regime: int = field(
        default=2,
        metadata=_p(
            "SYNTHETIC(prior: W2-industrial §2.1)",
            "regime C (code 2) for a regional conflict hitting a fab or an OSAT (§2.4 'Restoration'; §11 row 10)",
        ),
    )


@dataclass(frozen=True)
class PoissonParams:
    """Unexcited Poisson components outside Gamma (Q50): weather and accident closures, port strikes."""

    weather_per_chokepoint_year: float = field(
        default=0.2,
        metadata=_p(
            "SYNTHETIC(prior: ports, transferred to straits)",
            "weather closures per chokepoint-year (§4.2 table, Verschuur et al. 2023; §11 row 23)",
        ),
    )
    strike_per_region_year: float = field(
        default=0.1,
        metadata=_p("SYNTHETIC(placeholder)", "port strikes per active region-year (§11 row 51; §2.4 'Poisson')"),
    )
    stoppage_share: float = field(
        default=12 / 17,
        metadata=_p("DERIVED(W2 event list counts)", "12 stoppages, 5 slowdowns (§2.4 'Poisson components')"),
    )
    stoppage_duration: Law = field(
        default=Law("lognormal_weeks", (0.48, 0.67)),
        metadata=_p("CALIBRATED(W2 event list)", "full stoppage LN(0.48, 0.67) wk, fitted median 11.3 d (§4.4)"),
    )
    stoppage_severity: Law = field(
        default=Law("constant", (0.93,)),
        metadata=_p("SYNTHETIC(prior: W2-maritime §2.4)", "full stoppage sigma about 0.93 (§4.4 'Port strike')"),
    )
    slowdown_duration: Law = field(
        default=Law("lognormal_weeks", (1.93, 0.77)),
        metadata=_p("SYNTHETIC(prior: W2 event list)", "slowdown LN(1.93, 0.77) wk, fitted median 48.2 d (§4.4)"),
    )
    slowdown_severity: Law = field(
        default=Law("uniform", (0.2, 0.6)),
        metadata=_p("SYNTHETIC(prior: W2-maritime §2.4)", "slowdown sigma 0.2-0.6 (§4.4 'Port strike')"),
    )


LEAD_LAWS_COMMAND = (
    "uv run --no-project --with numpy==2.4.5 --with scipy==1.18.1 --with xlrd==2.0.2 python"
    " scripts/python/evidence/lead_laws.py"
)
_Q97_LEADS = (
    "l_q = F^{-1}_ch(V_q) of (49), one law per channel of codes.CHANNELS, in weeks: the atom_or_knots_days law of each"
    " channel's lead record (Q97), derived by `" + LEAD_LAWS_COMMAND + "` (docs/evidence/lead_laws.txt, its 'LAW"
    " channel' blocks) from the records the fetch scripts of docs/evidence/README.md fetch (Federal Register, the"
    " non-US tariff instruments, TIES 4.0, COW MID 5.0); samples: final notices all 76, formal 24 and informal 16"
    " proposals, BIS and OFAC legal publications pooled (493, negative leads kept), TIES threats on calendar dates"
    " (452), MID threats MIDI 5.0's within-dispute gap TBI (3,574, the §11 row 27 sample, whose use as a threat lead"
    " §11 row 27 tags SYNTHETIC(prior: MID5)) (design §12 'Lead laws'); the six laws are one leaf, tagged by its"
    " weakest part, the MID threat law's SYNTHETIC(prior: MID5), as the other bundles are (Q99 (e); the other five"
    " laws are DERIVED from their records)"
)

# The lead records of design §5.3 as their knots (Q97): per channel of codes.CHANNELS, the counts of the <= 0 and > 0
# parts of the sample and each part's kept order statistics (rank i, days), p = i / (m - 1), exactly as
# scripts/python/evidence/lead_laws.py prints them in docs/evidence/lead_laws.txt ('LAW channel' blocks, whose p it
# prints to six digits, which round to these ranks); ``laws.lead_law`` builds each law from them
# fmt: off
_LEAD_RECORDS: tuple[tuple[str, int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], ...] = (
    (
        "tariff_formal",
        0,
        24,
        (),
        (
            (0, 1), (1, 10), (2, 14), (3, 30), (4, 31), (5, 31), (6, 32), (7, 33), (8, 37), (9, 49), (10, 52),
            (11, 64), (12, 68), (13, 69), (14, 91), (15, 93), (16, 99), (17, 107), (18, 136), (19, 200), (20, 229),
            (21, 231), (22, 232), (23, 394),
        ),
    ),
    (
        "tariff_informal",
        0,
        16,
        (),
        (
            (0, 1), (1, 2), (2, 5), (3, 5), (4, 6), (5, 8), (6, 19), (7, 22), (8, 23), (9, 28), (10, 31), (11, 37),
            (12, 71), (13, 99), (14, 99), (15, 309),
        ),
    ),
    (
        "tariff_final",
        3,
        73,
        (
            (0, -4), (1, 0), (2, 0),
        ),
        (
            (0, 1), (15, 1), (16, 2), (17, 2), (18, 3), (22, 3), (23, 4), (26, 4), (27, 6), (30, 6), (31, 7), (37, 7),
            (38, 8), (39, 9), (40, 9), (41, 10), (42, 12), (43, 12), (44, 14), (45, 15), (53, 15), (54, 16), (55, 16),
            (56, 19), (57, 20), (58, 21), (59, 21), (60, 24), (61, 29), (62, 29), (63, 30), (66, 30), (67, 32),
            (68, 36), (69, 38), (70, 105), (71, 120), (72, 174),
        ),
    ),
    (
        "sanction_legal",
        446,
        47,
        (
            (0, -278), (1, -98), (2, -83), (3, -55), (4, -33), (5, -27), (6, -7), (7, -6), (14, -6), (15, -5),
            (23, -5), (24, -4), (41, -4), (42, -3), (58, -3), (59, -2), (78, -2), (79, -1), (98, -1), (99, 0),
            (445, 0),
        ),
        (
            (0, 2), (1, 3), (2, 5), (3, 7), (4, 14), (5, 18), (6, 20), (7, 21), (8, 23), (9, 23), (10, 26), (11, 30),
            (21, 30), (22, 31), (23, 32), (24, 32), (25, 33), (26, 40), (27, 40), (28, 42), (29, 46), (30, 47),
            (31, 50), (32, 60), (33, 60), (34, 61), (35, 62), (36, 62), (37, 80), (38, 90), (42, 90), (43, 120),
            (44, 120), (45, 156), (46, 180),
        ),
    ),
    (
        "ties_threat",
        255,
        197,
        (
            (0, -25), (1, 0), (254, 0),
        ),
        (
            (0, 1), (6, 1), (7, 2), (8, 3), (12, 3), (13, 4), (17, 4), (18, 5), (19, 5), (20, 6), (21, 6), (22, 7),
            (23, 7), (24, 8), (27, 8), (28, 9), (30, 9), (31, 10), (32, 10), (33, 11), (34, 13), (35, 15), (39, 15),
            (40, 17), (41, 18), (43, 18), (44, 21), (45, 21), (46, 22), (48, 22), (49, 23), (50, 24), (52, 24),
            (53, 26), (54, 29), (55, 33), (56, 34), (57, 34), (58, 38), (59, 39), (60, 43), (61, 49), (62, 50),
            (63, 51), (66, 51), (67, 60), (68, 70), (69, 71), (70, 72), (71, 73), (72, 74), (73, 74), (74, 80),
            (75, 83), (76, 83), (77, 91), (79, 91), (80, 107), (81, 110), (82, 119), (87, 119), (88, 125), (89, 129),
            (90, 131), (91, 139), (92, 140), (93, 141), (94, 144), (95, 156), (96, 158), (97, 161), (98, 164),
            (99, 176), (100, 178), (101, 200), (103, 200), (104, 202), (105, 203), (106, 203), (107, 210), (108, 216),
            (109, 217), (111, 217), (112, 225), (113, 225), (114, 230), (115, 237), (116, 243), (117, 258),
            (118, 285), (119, 288), (121, 288), (122, 290), (123, 291), (124, 295), (125, 299), (126, 310),
            (127, 315), (128, 330), (129, 343), (130, 351), (131, 368), (133, 368), (134, 371), (135, 384),
            (136, 392), (137, 403), (138, 403), (139, 405), (140, 416), (141, 425), (142, 446), (144, 446),
            (145, 452), (146, 457), (147, 457), (148, 467), (149, 469), (150, 481), (151, 503), (152, 504),
            (153, 511), (154, 511), (155, 517), (156, 525), (157, 531), (158, 547), (159, 566), (163, 566),
            (164, 579), (165, 597), (166, 614), (167, 636), (168, 693), (169, 701), (172, 701), (173, 710),
            (174, 743), (175, 778), (176, 812), (177, 882), (178, 890), (179, 946), (180, 981), (181, 1028),
            (182, 1049), (183, 1057), (184, 1082), (185, 1082), (186, 1105), (187, 1112), (190, 1112), (191, 1253),
            (192, 1277), (193, 1296), (194, 1433), (195, 1526), (196, 3617),
        ),
    ),
    (
        "mid_threat",
        567,
        3007,
        (
            (0, 0), (566, 0),
        ),
        (
            (0, 1), (346, 1), (347, 2), (562, 2), (563, 3), (702, 3), (703, 4), (852, 4), (853, 5), (992, 5),
            (993, 6), (1116, 6), (1117, 7), (1223, 7), (1224, 8), (1305, 8), (1306, 9), (1398, 9), (1399, 10),
            (1476, 10), (1477, 11), (1538, 11), (1539, 12), (1590, 12), (1591, 13), (1655, 13), (1656, 14),
            (1709, 14), (1710, 15), (1753, 15), (1754, 16), (1800, 16), (1801, 17), (1828, 17), (1829, 18),
            (1862, 18), (1863, 19), (1905, 19), (1906, 20), (1936, 20), (1937, 21), (1979, 21), (1980, 22),
            (2011, 22), (2012, 23), (2043, 23), (2044, 24), (2075, 24), (2076, 25), (2103, 25), (2104, 26),
            (2135, 26), (2136, 27), (2167, 27), (2168, 28), (2188, 28), (2189, 29), (2203, 29), (2204, 30),
            (2226, 30), (2227, 31), (2246, 31), (2247, 32), (2261, 32), (2262, 33), (2281, 33), (2282, 34),
            (2293, 34), (2294, 35), (2308, 35), (2309, 36), (2323, 36), (2324, 37), (2343, 37), (2344, 38),
            (2358, 38), (2359, 39), (2373, 39), (2374, 40), (2390, 40), (2391, 41), (2399, 41), (2400, 42),
            (2411, 42), (2412, 43), (2424, 43), (2425, 44), (2438, 44), (2439, 45), (2449, 45), (2450, 46),
            (2454, 46), (2455, 47), (2459, 47), (2460, 48), (2477, 48), (2478, 49), (2489, 49), (2490, 50),
            (2498, 50), (2499, 51), (2511, 51), (2512, 52), (2525, 52), (2526, 53), (2534, 53), (2535, 54),
            (2546, 54), (2547, 55), (2553, 55), (2554, 56), (2560, 56), (2561, 57), (2570, 57), (2571, 58),
            (2575, 58), (2576, 59), (2581, 59), (2582, 60), (2589, 60), (2590, 61), (2593, 61), (2594, 62),
            (2604, 62), (2605, 63), (2610, 63), (2611, 64), (2617, 64), (2618, 65), (2623, 65), (2624, 66),
            (2632, 66), (2633, 67), (2637, 67), (2638, 68), (2640, 68), (2641, 69), (2645, 69), (2646, 70),
            (2651, 70), (2652, 71), (2657, 71), (2658, 72), (2663, 72), (2664, 73), (2666, 73), (2667, 74),
            (2673, 74), (2674, 75), (2679, 75), (2680, 76), (2687, 76), (2688, 77), (2690, 77), (2691, 78),
            (2694, 78), (2695, 79), (2697, 79), (2698, 80), (2701, 80), (2702, 81), (2707, 81), (2708, 82),
            (2711, 82), (2712, 83), (2719, 83), (2720, 84), (2724, 84), (2725, 85), (2733, 85), (2734, 86),
            (2736, 86), (2737, 87), (2741, 87), (2742, 88), (2748, 88), (2749, 89), (2751, 89), (2752, 90),
            (2756, 90), (2757, 91), (2761, 91), (2762, 92), (2767, 92), (2768, 93), (2771, 93), (2772, 94),
            (2776, 94), (2777, 95), (2783, 95), (2784, 96), (2788, 96), (2789, 97), (2791, 97), (2792, 98),
            (2796, 98), (2797, 99), (2799, 99), (2800, 100), (2801, 101), (2802, 101), (2803, 102), (2805, 102),
            (2806, 103), (2809, 103), (2810, 104), (2811, 105), (2813, 105), (2814, 106), (2818, 106), (2819, 107),
            (2822, 107), (2823, 108), (2825, 108), (2826, 109), (2830, 109), (2831, 110), (2832, 111), (2833, 112),
            (2838, 112), (2839, 113), (2842, 113), (2843, 114), (2844, 115), (2845, 115), (2846, 116), (2849, 116),
            (2850, 117), (2852, 117), (2853, 118), (2855, 118), (2856, 119), (2857, 119), (2858, 120), (2860, 120),
            (2861, 121), (2862, 121), (2863, 122), (2866, 122), (2867, 123), (2868, 123), (2869, 124), (2872, 124),
            (2873, 125), (2874, 125), (2875, 126), (2879, 126), (2880, 127), (2882, 127), (2883, 128), (2884, 129),
            (2890, 129), (2891, 130), (2892, 131), (2893, 131), (2894, 132), (2897, 132), (2898, 133), (2900, 133),
            (2901, 134), (2904, 134), (2905, 135), (2907, 135), (2908, 136), (2912, 136), (2913, 137), (2914, 137),
            (2915, 138), (2916, 139), (2917, 140), (2918, 140), (2919, 141), (2921, 141), (2922, 142), (2924, 142),
            (2925, 143), (2927, 143), (2928, 144), (2929, 144), (2930, 145), (2932, 145), (2933, 146), (2937, 146),
            (2938, 147), (2939, 147), (2940, 149), (2942, 149), (2943, 150), (2944, 151), (2945, 151), (2946, 152),
            (2947, 153), (2949, 153), (2950, 154), (2951, 154), (2952, 155), (2953, 155), (2954, 156), (2956, 156),
            (2957, 159), (2958, 159), (2959, 160), (2962, 160), (2963, 161), (2965, 161), (2966, 164), (2968, 164),
            (2969, 165), (2970, 165), (2971, 166), (2972, 166), (2973, 167), (2974, 168), (2975, 168), (2976, 169),
            (2977, 169), (2978, 170), (2982, 170), (2983, 171), (2984, 173), (2985, 173), (2986, 174), (2988, 174),
            (2989, 175), (2990, 175), (2991, 179), (2992, 180), (2994, 180), (2995, 181), (2996, 181), (2997, 183),
            (2998, 202), (2999, 207), (3000, 211), (3001, 223), (3002, 235), (3003, 354), (3004, 394), (3005, 735),
            (3006, 1156),
        ),
    ),
)
# fmt: on
LEAD_LAWS: tuple[tuple[str, Law], ...] = tuple((ch, lead_law(*rec)) for ch, *rec in _LEAD_RECORDS)
assert tuple(ch for ch, _ in LEAD_LAWS) == CHANNELS  # one law per channel, in the codes' order


@dataclass(frozen=True)
class InformationParams:
    """The announcement stage of the generator (M3): leads (stream 4), shadows (stream 7), blackout spells (stream 11).

    Implements the draws behind (49) and the blackout rung's S^blk of (47) (design §4.1 streams 4, 7, 11; §5.3;
    Q51, Q89, Q97). The regimes read their theta (``information.theta``), never this tree; what the observation needs
    of it (phi-bar) reaches the view through omega's ``meta_information_params`` (``omega.assembly.InformationMeta``).
    Every leaf carries its provenance; a leaf without a source is SYNTHETIC(placeholder), so V23 refuses a seal.
    """

    lead: tuple[tuple[str, Law | None], ...] = field(
        default=LEAD_LAWS,
        metadata=_p("SYNTHETIC(prior: MID5)", _Q97_LEADS),
    )
    informal_share: float = field(
        default=25 / 57,
        metadata=_p(
            "DERIVED(W2-information §2.5)",
            "an announced tariff's proposal is informal with probability 25/57, else formal (§5.3 inputs; stream 4)",
        ),
    )
    phi_bar: tuple[tuple[str, float], ...] = field(
        default=(("tariff_formal", 0.25), ("tariff_informal", 0.36), ("ties_threat", 0.538), ("mid_threat", 0.298)),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "phi-bar_ch of (49), release-level, the largest phi used in the release (§5.3); set to the record phi in M3"
            " (formal 8/32 and informal 9/25 DERIVED(W2-information §2.5), TIES 567/1,053 DERIVED(R3 §2.5), MID 0.298"
            " SYNTHETIC(prior: MID5)); a placeholder until the decoy-flood magnitudes of §11 row 33 (phase 4) fix the"
            " largest phi; a larger phi-bar redraws Y (design §12 M3 rows)",
        ),
    )
    channels: tuple[tuple[str, tuple[str, ...]], ...] = field(
        default=TYPE_CHANNELS,
        metadata=_p(
            "Q97",
            "the channel set of each announced event type, real and shadow alike (symmetric decoys, Q97); 'proposal'"
            " is tariff_formal or tariff_informal by the informal share; every other type is unannounced; must equal"
            " the frozen omega.codes.TYPE_CHANNELS",
        ),
    )
    blackout_lengths: tuple[tuple[str, int | None], ...] = field(
        default=(("3w", 3), ("11w", 11), ("to_end", None)),
        metadata=_p(
            "SYNTHETIC(prior: JMIC, BEA, UN Comtrade)",
            "spell length in weeks per codes.BLACKOUT_SPELLS kind, None to the episode end (§11 row 34; W2-information"
            " §2.4)",
        ),
    )
    blackout_onset: str = field(
        default="uniform",
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "onset week uniform over the admissible weeks (1..T - length + 1; 1..T to the end) by one stream-11 uniform"
            " at key (spell kind); no onset rate measured (§11 row 34; M3 reading, design §12)",
        ),
    )

    def __post_init__(self) -> None:
        """Refuse channel sets other than Q97's (``codes.TYPE_CHANNELS``), which ``information.messages`` reads.

        Raises:
            ValueError: if ``channels`` is not ``codes.TYPE_CHANNELS`` (the messages module reads the frozen sets from
                (inst, omega), so another value here would never be read; design §12 "Messages"), ``lead`` does not
                give a law or None per channel of ``codes.CHANNELS`` in order, or ``informal_share`` is not a share.

        """
        if self.channels != TYPE_CHANNELS:
            raise ValueError(
                f"InformationParams.channels must be the Q97 symmetric sets {TYPE_CHANNELS} (omega.codes), got "
                f"{self.channels!r}"
            )
        if tuple(ch for ch, _ in self.lead) != CHANNELS or not all(
            law is None or isinstance(law, Law) for _, law in self.lead
        ):
            raise ValueError(f"InformationParams.lead must give one Law (or None) per channel of {CHANNELS}")
        if not 0.0 <= self.informal_share <= 1.0:
            raise ValueError(f"InformationParams.informal_share must be a share in [0, 1], got {self.informal_share}")


_ROUND = "owner 2026-09-29, 'V3: fix both (Recommended)', for small and full; docs/decisions.md Q114; design §12 row"
_ROUND += " 'Sanction target rules V3 (Q114)'; docs/099-m5-sanction-rules.md V3 (branch m5-sanction-rules)"


@dataclass(frozen=True)
class TargetingRules:
    """The sanction target rules of `small` and `full`: the M5 sanction-rules round's V3, decided as Q114.

    ``sanction_no_transit`` (V1): a sanction never targets an edge whose tail is a chokepoint, since a transit leg out
    of a chokepoint is not an export of the chokepoint's coastal region (§2.4 "Targets"; ``targets.candidate_targets``).
    ``policy_no_op`` (V2, the TIES round's broad reading): in the policy block, a type with no target in the region
    keeps its share as one no-op part instead of being renormalised onto the feasible types (§2.4 "Type laws per
    block"; ``targets.type_weights``); the militarised block keeps its renormalisation. Either flag alone is option 2
    or V2 of the round, which Q114 did not choose. Set by ``profiles.flagship_table`` only (`small` and `full`);
    `tiny` leaves ``GeneratorParams.targeting`` None and keeps §2.4's rules as written.
    """

    sanction_no_transit: bool = field(
        default=True,
        metadata=_p(
            "Q114", f"V1: no sanction on an edge out of a chokepoint, a transit leg being no export ({_ROUND})"
        ),
    )
    policy_no_op: bool = field(
        default=True,
        metadata=_p(
            "Q114",
            f"V2: the policy block's infeasible share is a no-op part, not renormalised ({_ROUND})",
        ),
    )


@dataclass(frozen=True)
class GeneratorParams:
    """The whole disruption generator of §4 for one (size, rung): with the instance and E_split it fixes omega."""

    profile: str = field(metadata=_p("DERIVED(profile)", "the profile's name, the instance kind it is written for"))
    active_regions: tuple[str, ...] = field(
        metadata=_p("DERIVED(instance)", "regions hosting a node; only they have a baseline (§2.4 'Regions')")
    )
    burn_in: int = field(
        default=1680,
        metadata=_p(
            "DERIVED((38), Q87)",
            "1,680 weeks, one per size and generator family (§4.3); profiles.check_burn_in checks (38) over RUNGS",
        ),
    )
    regime: RegimeParams = RegimeParams()
    latent_region: LatentParams = REGION_LATENT
    latent_dyad: LatentParams = DYAD_LATENT
    latent_chokepoint: LatentParams = CHOKEPOINT_LATENT
    hawkes: HawkesParams = HawkesParams()
    types: TypeLaws = TypeLaws()
    laws: MarkLaws = MarkLaws()
    poisson: PoissonParams = PoissonParams()
    marks: MarkParams = MarkParams()
    # M3: the announcement stage (streams 4, 7, 11), on by default since the M3 merge; None is the M2 generator, which
    # draws no lead, shadow or blackout spell and is left out of generator_id (``params_dict``), so an M2 omega keeps
    # its M2 generator_id
    information: InformationParams | None = field(
        default=InformationParams(),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "read only when None (then a leaf): the M2 generator, whose streams 4, 7 and 11 draw nothing, so no release"
            " omega has messages, shadows or spells; set, its own leaves carry their provenance (§5.3, (49))",
        ),
    )
    # Q114's scope: None is §2.4's target rules as written, which `tiny` keeps and which ``params_dict`` leaves out of
    # generator_id, so `tiny`'s id and pins stand; `small` and `full` set the V3 rules (``profiles.FLAGSHIP_TARGETING``)
    targeting: TargetingRules | None = field(
        default=None,
        metadata=_p(
            "Q114",
            "read only when None (then a leaf): the §2.4 target rules as written, `tiny`'s (sanctions on any edge out"
            f" of the region, infeasible types renormalised); set by profiles.flagship_table only ({_ROUND})",
        ),
    )

    def with_rung(self, gamma: float) -> "GeneratorParams":
        """The same generator at another policy-block rung; baselines and burn-in stay fixed (Q39, Q87)."""
        return dataclasses.replace(self, hawkes=dataclasses.replace(self.hawkes, gamma=gamma))


# ----- M5: the `small` and `full` profile's own values and provenance (design §12 M5 stream "generator" rows) -------
# Subclasses in the ``RegionLatent`` pattern: a field whose value or source differs from `tiny`'s is redeclared with its
# own metadata, so ``provenance`` never reports `tiny`'s strings for it; ``params_dict`` (and so ``generator_id``) sees
# the same field names, plus the strait-closure fields of ``FlagshipMarkLaws``, which only the flagship tree carries.
FLAGSHIP_CROSS_MP = 0.01  # M5-O7 (a), owner 2026-09-28: Gamma^MP on `small` and `full`, re-parameterisation attempt 1
TIES_WEIBULL = Law("weibull_days", (0.6590, 1523.17))  # Q106: TIES 4.0 sample A (profiles.TIES_WEIBULL_* constants)
_TIES_COMMAND = (
    "uv run --no-project --with numpy==2.4.5 --with scipy==1.18.1 --with xlrd==2.0.2 python"
    " scripts/python/evidence/m5_sanction_law.py"
)


@dataclass(frozen=True)
class FlagshipHawkesParams(HawkesParams):
    """The Hawkes block of the `small` and `full` profile (§4.2; M5-O7, M5-O12; design §12 M5 generator rows)."""

    cross_MP: float = field(
        default=FLAGSHIP_CROSS_MP,
        metadata=_p(
            "SYNTHETIC(prior)",
            "Gamma^MP, militarised children of a policy event (31), 0.01 on `small` and `full`: re-parameterisation"
            " attempt 1 of §2.6 for the stress ceiling (the owner's M5-O7 answer (a), 2026-09-28; 0.02 gave 2.76 and"
            " 5.01 closures/yr on `full` at 0.95 and 0.97 against 2.55); §11 row 18",
        ),
    )
    trade_adjacency: tuple[tuple[str, str, float], ...] = field(
        default=(),
        metadata=_p(
            "SYNTHETIC(prior)",
            "A = 1 between regions an edge of the instance joins (either direction), DERIVED(instance) as a"
            " trade-and-tie measure, the §2.4 rule applied to `small` and `full` (R3 §3; §11 row 56)",
        ),
    )
    baselines: tuple[tuple[str, float, float], ...] = field(
        default=(),
        metadata=_p(
            "SYNTHETIC(prior: budget_rejection.py, f5_realism.py)",
            "set by profiles.flagship_table: per active region 12/52 (1 - 0.62) / 14 events per week (`full`'s 14"
            " active regions, held on `small` too: M5-O12 (a)), split policy 0.70 / militarised 0.30; the militarised"
            " baseline of the chokepoint-adjacent regions DERIVED by the one-scale (32) calibration to 0.43 closures"
            " per year on the seven chokepoints (M5-O8 (b)), run at every build by profiles.calibrate_flagship_closures"
            " (design §12 M5 generator rows; §11 row 21)",
        ),
    )


_DURATION_FLAGSHIP = (
    "per type (§4.4; §2.4 'Durations'), as `tiny`'s except sanction and material outage: the Weibull of TIES 4.0"
    " imposed spells, shape 0.6590, scale 1,523.17 d, DERIVED(TIES 4.0) (Q106; `" + _TIES_COMMAND + "`, sample A);"
    " closure (39) transient LN(4.519, 1.251) d DERIVED(MID5) prior only (§11 row 7), persistent weight 0.40 in"
    " [0.33, 0.50] (§2.6), median 104 wk (Q76 owner D4), spread 1.0 SYNTHETIC(placeholder) (§11 row 67); regional"
    " conflict SYNTHETIC(prior: MID5 dispute durations) (§11 row 24); tariff, piracy, energy"
    " SYNTHETIC(prior: budget_rejection.py); weather LN(1.79, 0.80) d (Verschuur et al. 2020, §4.4)"
)


@dataclass(frozen=True)
class FlagshipMarkLaws(MarkLaws):
    """The mark laws of the `small` and `full` profile: the TIES Weibull (Q106) and the dyad strait closures (M5-O13).

    ``strait_closures`` lists (region a, region b, chokepoint id): a regional conflict of region a with counterpart b,
    or of b with a, also closes that chokepoint (§4.4 "a CN-TW event also closes the Taiwan Strait"), by the rule of
    ``events.strait_closures``: onset the conflict's, window W0 = max(T_q, ``strait_closure_weeks``) of (40), severity
    ``strait_closure_severity``, exciting nothing in (30), counted by V22 and the (32) calibration, left out of naive's
    transient F_Q (the owner queue's M5-O13 default (a); option (b) is ``strait_closure_weeks`` = 0).
    """

    duration: tuple[tuple[str, Law], ...] = field(
        default=(
            ("militarised_closure", Law("persistent_mix", (0.40, 4.519, 1.251, math.log(104.0), 1.0))),  # (39)
            ("regional_conflict", Law("lognormal_days", (5.531, 1.453))),
            ("tariff", Law("lognormal_days", (math.log(365.0), 1.0))),
            ("sanction", TIES_WEIBULL),  # Q106
            ("material_outage", TIES_WEIBULL),  # Q106: the sanction and export-control law (§4.4)
            ("piracy", Law("lognormal_days", (math.log(90.0), 1.0))),
            ("energy_shock", Law("lognormal_days", (math.log(14.0), 1.0))),
            ("weather_closure", Law("lognormal_days", (1.79, 0.80))),
        ),
        metadata=_p("SYNTHETIC(placeholder)", _DURATION_FLAGSHIP),
    )
    strait_closures: tuple[tuple[str, str, str], ...] = field(
        default=(("CN", "TW", "chk_taiwan"),),
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "(region a, region b, chokepoint id): a regional conflict between a and b closes the chokepoint (§4.4;"
            " the owner queue's M5-O13 default (a), 2026-09-28, until answered; CN-TW the only dyad, M5-O14 (a))",
        ),
    )
    strait_closure_weeks: float = field(
        default=52.0,
        metadata=_p(
            "SYNTHETIC(placeholder)",
            "the strait closure's window W0 = max(T_q, 52) weeks, the war profile's W0 of (40) (M5-O13 default (a))",
        ),
    )
    strait_closure_severity: float = field(
        default=1.0, metadata=_p("SYNTHETIC(placeholder)", "the strait closure's severity (M5-O13 default (a))")
    )


@dataclass(frozen=True)
class FlagshipGeneratorParams(GeneratorParams):
    """The generator of `small` and `full` (design §12 M5 stream "generator" rows): its own burn-in provenance.

    Built by ``profiles.flagship_profile``; every other field as ``GeneratorParams``, with the flagship subclasses
    above in ``hawkes`` and ``laws``.
    """

    burn_in: int = field(
        default=0,
        metadata=_p(
            "DERIVED(Q97, Q106, (33))",
            "one per size and generator family (Q87): max over the family's rungs of t_95 of (33) plus the largest"
            " run-up of Q97's 1 % rule over every event type the generator draws (M5-O9 (a), the owner queue's"
            " default until answered), set by profiles.burn_in_family (design §12 M5 row 'Burn-in of `small` and"
            " `full`'); profiles.check_burn_in checks (38) too",
        ),
    )


def provenance(params) -> dict[str, dict]:
    """The provenance metadata of every leaf of a params tree, by dotted path (§4.1 rule table; V20, V23).

    A leaf is a field whose value is a ``Law`` or not a dataclass; nested dataclasses are walked. Each record is a copy
    of the field's metadata ({} for a field without any).
    """
    out: dict[str, dict] = {}

    def walk(obj, path: str) -> None:
        for f in dataclasses.fields(obj):
            value, name = getattr(obj, f.name), f"{path}{f.name}"
            if dataclasses.is_dataclass(value) and not isinstance(value, Law):
                walk(value, name + ".")
            else:
                out[name] = dict(f.metadata)

    walk(params, "")
    return out


def params_dict(params: GeneratorParams) -> dict:
    """The resolved dataclass tree as a JSON-able dict (floats by repr through canonical_json).

    An absent announcement stage (``information`` None, the M2 generator) is left out, like ``Trajectory.failed``
    when unset, so the M2 ``generator_id`` values stand until the stage is built; so are absent target rules
    (``targeting`` None, every profile but `small` and `full`'s), so `tiny`'s ids stand.
    """
    out = dataclasses.asdict(params)
    for name in ("information", "targeting"):
        if out.get(name) is None:
            out.pop(name, None)
    return out


def generator_id(params: GeneratorParams, inst: Instance) -> str:
    """SHA-256 of the canonical JSON of the resolved tree with schema_version and the instance hash (§4.1)."""
    return sha256_hex({"schema_version": SCHEMA_VERSION, "instance_hash": inst.hash, "params": params_dict(params)})

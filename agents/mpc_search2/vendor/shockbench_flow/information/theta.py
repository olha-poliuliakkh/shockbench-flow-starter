"""The information regime theta of (47) and its registry (design §5.1, §5.3; Q10-Q12, Q46, Q51, Q55, Q80).

theta = (L_theta, {a_theta x}_x, phi, l-mode, chi, h_cov, S^blk) (47), plus the §5.1 feed switches (pending
prohibitions, the exact forecast, full omega at reset). One class, ``Theta``; each named regime and each rung kind is a
subclass whose fields carry the provenance of its own §5.3 row (``disruption.params.leaf_provenance``: tag, source,
lo, hi), as the latent kinds of ``disruption.params`` do, so V23 refuses a seal while a leaf is SYNTHETIC(placeholder)
and informed's CALIBRATED region a is not tagged down by another leaf of its regime. Fields are scalars (per unit
kind of ``codes.UNIT_KINDS``, per decoy-bearing channel of ``codes.DECOY_CHANNELS``), never mappings: OmegaConf and the
canonical JSON of a digest take no FrozenDict.

- Warning (45): ``L`` the signal lag, ``a_<kind>`` the separation per unit kind; ``L`` None means no warning feed.
  a in [-1, 1]. a < 0 is not the sign-flipped member S -> -S of the adversarial set (50): with the same W the two agree
  in distribution only, so the path-level pairing (R5.11) needs that member as a factor -1 on S (phase 4).
- Declared skill (46): ``skill_<kind>`` the AUROC_13 published in ``Static.regime.skill``; for chokepoint units the
  total extractable skill that V27's chokepoint script measures (Q80), None until it has; for standard's region and
  dyad units likewise the V27 battery maximum (Q99 (a)), not the 0.70 target its a_std was derived for.
- Messages (49): ``phi_<channel>`` the false share of the four decoy-bearing channels, ``lead_mode`` 'record'; every
  phi None means no messages feed. Clairvoyant's phi is 0 (real messages only).
- ``chi``: closure_end shown; ``pending``: pending_prohibitions shown; ``forecast``: 'mmfe' (48) or 'exact' (the
  realised d, clairvoyant); ``forecast_shock``: whether last week's conflict state enters (48)'s shock factor (§11 row
  70; yes by default, SYNTHETIC(placeholder)): the Xi term is on when it and omega's ``InformationMeta.xi`` are both
  true; ``omega_at_reset``: the omega payload in ``info['omega']`` and ``Reset.omega`` (local and dev runs only,
  design §12 M3 rows).
- ``h_cov``: the coverage rung's hop radius (prediction-free otherwise); ``blackout``: the blackout rung's spell kind,
  a name of ``codes.BLACKOUT_SPELLS`` (prediction-free outside its spell).

``Trajectory.regime`` hashes a label (``regime_label``): a registry theta's name, else its name, '#' and the first 16
hex digits of the SHA-256 of its canonical JSON, so two different custom thetas never share an identity; a theta that
is not the registry's is refused under a registry name (``resolve_regime``). The registry (``REGISTRY``) is the named
regimes, then ``coverage_rung(h)`` per level and ``blackout_rung(name)`` per spell kind, in ``REGIME_NAMES`` order.
"""

import dataclasses
import math
from dataclasses import dataclass, field

from shockbench_flow.disruption.params import leaf_provenance as _p
from shockbench_flow.disruption.params import provenance as _provenance
from shockbench_flow.instance.io import sha256_hex
from shockbench_flow.omega import codes


FORECASTS = ("mmfe", "exact")  # (48) in every regime but clairvoyant; the realised d under clairvoyant (§5.1)
LEAD_MODES = ("record",)  # l-mode of (47): leads invert the record by common random numbers (§5.3); phase 4 adds more
COVERAGE_LEVELS = (0, 1, 2)  # h_cov of the spatial-coverage rung (§5.3 table), SYNTHETIC(prior: CTP convention)
DECOY_CHANNELS = codes.DECOY_CHANNELS  # the channels with a shadow process (49), re-exported
LABEL_HEX = 16  # hex digits of a custom theta's digest in its label (64 bits: an identifier, not a model value)

_CALLER = "a theta built by the caller, no source (the named regimes and rungs are this module's subclasses)"
_VIA_OMEGA = "§5.1 table 'via omega', read as"  # the clairvoyant cells the design leaves to omega (M3 reading, §12)
_NO_FEED = "DERIVED(§5.1 table)"
_MRS = "CALIBRATED(MRS 2024 via W2-information)"
_A_STD = "a_std = 0.604 for the target AUROC_13 0.70 at L = 1 (owner D6; f3_a_std.py; §5.2 'Values', §11 row 31)"
_PLACEHOLDER = "SYNTHETIC(placeholder)"
_PHI_FORMAL = ("DERIVED(W2-information §2.5)", "phi of tariff formal proposals, 8/32 = 0.25 (§5.3 inputs; Q46)")
_PHI_INFORMAL = ("DERIVED(W2-information §2.5)", "phi of tariff informal proposals, 9/25 = 0.36 (§5.3 inputs; Q46)")
_PHI_TIES = ("DERIVED(R3 §2.5)", "phi of TIES sanction threats, 567/1,053 = 0.538 (§5.3 inputs; Q12)")
_PHI_MID = ("SYNTHETIC(prior: MID5)", "phi of MID threats, 727/2,436 = 0.298, a lower bound (§5.3 inputs; Q58)")
_CHI = "chi of informed and standard: no value (§11 row 32); false, closure_end null (M3 reading, design §12)"
_CHK_SKILL = (
    "the chokepoint units' total extractable skill (Q80), {value}: the DECLARED mean of the two seeds' battery maxima"
    " at V27's size with the announcement stage, `uv run --no-project --with numpy==2.4.5 --with scipy==1.18.1 --with"
    " fastjsonschema~=2.22.2 --with joblib --with scikit-learn~=1.8 python scripts/python/evidence/v27_chokepoint.py 0`"
    " (and `... 2`, then the same command without an argument for the summary; docs/evidence/v27_chokepoint.txt;"
    " design §12 row 'V27 chokepoint script')"
)
_STD_SKILL = (
    "standard's {kind} units' total extractable skill (Q80, Q99 (a)), 0.712: the mean of the two seeds' V27 battery"
    " maxima at V27's size, 0.7106 and 0.7140 (seeds 0 and 2; gradient boosting on 26 lags and the public history,"
    " which at a_std carries information on X the score lacks); region and dyad units share one world (§2.4; Q59), so"
    " one run measures both: `uv run pytest -q -s -m slow tests/test_m3_regimes.py -k v27_region` (design §12 row"
    " 'V27 (ii) on standard region units')"
)
_SHOCK = (
    "whether last week's conflict state enters the forecast's shock factor (48): yes by default (§11 row 70); the Xi"
    " term is on when this and omega's InformationMeta.xi are both true"
)
_A_INF_CHK = "a = 1 for every unit kind of informed (§5.1 'a=1, L=0'); chokepoints SYNTHETIC(prior) (§5.3; §11 row 31)"
_A_STD_KIND = (
    "a_std = 0.604 for every unit kind of standard (§5.1 'a_std, L=1'); {} units SYNTHETIC(prior) (§11 row 31)"
)


def _off(what: str, value=None):
    """A field of a feed this regime does not show: its value is the §5.1 table's 'none' (no parameter applies)."""
    return field(default=value, metadata=_p(_NO_FEED, f"{what}: not shown in this regime, no parameter applies"))


@dataclass(frozen=True)
class Theta:
    """One information regime theta of (47); the base class of every named regime and rung.

    Built directly, every value is the caller's (tagged SYNTHETIC(placeholder)); the registry's regimes are the
    subclasses below, whose fields carry the tags of their §5.3 rows. ``check_theta`` states the invariants.
    """

    name: str = field(metadata=_p(_PLACEHOLDER, "the regime's name, hashed into Trajectory.regime: " + _CALLER))
    L: int | None = field(default=None, metadata=_p(_PLACEHOLDER, "L_theta of (45), None: no warning: " + _CALLER))
    a_region: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "a of region units (45): " + _CALLER))
    a_dyad: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "a of dyad units (45): " + _CALLER))
    a_chokepoint: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "a of chokepoints (45): " + _CALLER))
    skill_region: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "AUROC_13 of (46): " + _CALLER))
    skill_dyad: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "AUROC_13 of (46): " + _CALLER))
    skill_chokepoint: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "AUROC_13 (Q80): " + _CALLER))
    phi_tariff_formal: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "phi of (49): " + _CALLER))
    phi_tariff_informal: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "phi of (49): " + _CALLER))
    phi_ties_threat: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "phi of (49): " + _CALLER))
    phi_mid_threat: float | None = field(default=None, metadata=_p(_PLACEHOLDER, "phi of (49): " + _CALLER))
    lead_mode: str | None = field(default=None, metadata=_p(_PLACEHOLDER, "l-mode of (47): " + _CALLER))
    chi: bool = field(default=False, metadata=_p(_PLACEHOLDER, "chi of (47), closure_end shown: " + _CALLER))
    pending: bool = field(default=False, metadata=_p(_PLACEHOLDER, "pending_prohibitions shown: " + _CALLER))
    forecast: str = field(default="mmfe", metadata=_p(_PLACEHOLDER, "the forecast of FORECASTS: " + _CALLER))
    forecast_shock: bool = field(default=True, metadata=_p(_PLACEHOLDER, "the Xi switch of (48): " + _CALLER))
    omega_at_reset: bool = field(default=False, metadata=_p(_PLACEHOLDER, "full omega at reset: " + _CALLER))
    h_cov: int | None = field(default=None, metadata=_p(_PLACEHOLDER, "h_cov of (47): " + _CALLER))
    blackout: str | None = field(default=None, metadata=_p(_PLACEHOLDER, "S^blk of (47), a spell kind: " + _CALLER))


@dataclass(frozen=True)
class PredictionFree(Theta):
    """The prediction-free regime: the no-prediction reference and the naive anchor's regime.

    prediction-free (§5.3 table; Q10, Q55): own state, the present graph and the forecast (48); the naive anchor's
    regime, whatever regime the policy plays (Q10).
    """

    name: str = field(default="prediction_free", metadata=_p("Q10", "the naive anchor's regime (§5.3 table)"))
    L: int | None = _off("warning (45)")
    a_region: float | None = _off("warning (45)")
    a_dyad: float | None = _off("warning (45)")
    a_chokepoint: float | None = _off("warning (45)")
    skill_region: float | None = _off("warning (45)")
    skill_dyad: float | None = _off("warning (45)")
    skill_chokepoint: float | None = _off("warning (45)")
    phi_tariff_formal: float | None = _off("messages (49)")
    phi_tariff_informal: float | None = _off("messages (49)")
    phi_ties_threat: float | None = _off("messages (49)")
    phi_mid_threat: float | None = _off("messages (49)")
    lead_mode: str | None = _off("messages (49)")
    h_cov: int | None = _off("the coverage rung's radius")
    blackout: str | None = _off("the blackout rung's spell")
    chi: bool = field(default=False, metadata=_p(_NO_FEED, "closure_end: 'no' in prediction-free"))
    pending: bool = field(default=False, metadata=_p(_NO_FEED, "pending_prohibitions: 'none' in prediction-free"))
    forecast: str = field(default="mmfe", metadata=_p(_NO_FEED, "the same forecast (48) as informed (Q55)"))
    forecast_shock: bool = field(default=True, metadata=_p(_PLACEHOLDER, _SHOCK))
    omega_at_reset: bool = field(default=False, metadata=_p(_NO_FEED, "full omega at reset: 'no'"))


@dataclass(frozen=True)
class Clairvoyant(Theta):
    """The clairvoyant regime: full omega at reset, the consistency endpoint.

    clairvoyant (§5.3 table; R1 A2; Q46): full omega at reset, the only regime with phi = 0; the consistency
    endpoint. Its 'via omega' cells are filled so a policy reads one code path (M3 reading, design §12).
    """

    name: str = field(default="clairvoyant", metadata=_p("Q46", "the consistency endpoint (§5.3 table)"))
    L: int | None = field(default=0, metadata=_p(_NO_FEED, _VIA_OMEGA + " L = 0"))
    a_region: float | None = field(default=1.0, metadata=_p(_NO_FEED, _VIA_OMEGA + " a = 1 for every unit kind"))
    a_dyad: float | None = field(default=1.0, metadata=_p(_NO_FEED, _VIA_OMEGA + " a = 1 for every unit kind"))
    a_chokepoint: float | None = field(default=1.0, metadata=_p(_NO_FEED, _VIA_OMEGA + " a = 1 for every unit kind"))
    skill_region: float | None = field(default=None, metadata=_p(_NO_FEED, "not declared: omega is seen whole"))
    skill_dyad: float | None = field(default=None, metadata=_p(_NO_FEED, "not declared: omega is seen whole"))
    skill_chokepoint: float | None = field(default=None, metadata=_p(_NO_FEED, "not declared: omega is seen whole"))
    phi_tariff_formal: float | None = field(default=0.0, metadata=_p(_NO_FEED, "phi = 0: real messages only"))
    phi_tariff_informal: float | None = field(default=0.0, metadata=_p(_NO_FEED, "phi = 0: real messages only"))
    phi_ties_threat: float | None = field(default=0.0, metadata=_p(_NO_FEED, "phi = 0: real messages only"))
    phi_mid_threat: float | None = field(default=0.0, metadata=_p(_NO_FEED, "phi = 0: real messages only"))
    lead_mode: str | None = field(default="record", metadata=_p(_NO_FEED, "the record leads of the real messages"))
    chi: bool = field(default=True, metadata=_p(_NO_FEED, "closure_end: 'yes' in clairvoyant"))
    pending: bool = field(
        default=True, metadata=_p(_NO_FEED, _VIA_OMEGA + " every future prohibition of omega's events (exact)")
    )
    forecast: str = field(default="exact", metadata=_p(_NO_FEED, "the realised series (§5.3; Q55)"))
    forecast_shock: bool = field(default=True, metadata=_p(_NO_FEED, "not read: the exact forecast is the realised d"))
    omega_at_reset: bool = field(default=True, metadata=_p(_NO_FEED, "full omega at reset: 'yes'"))
    h_cov: int | None = _off("the coverage rung's radius")
    blackout: str | None = _off("the blackout rung's spell")


@dataclass(frozen=True)
class Informed(Theta):
    """The informed regime: good forecasting.

    informed (§5.3 table; v0.2 §6.2; Q12, Q37, Q46, Q51): L = 0, a = 1 for regions and dyads, record phi and leads
    (good forecasting).
    """

    name: str = field(default="informed", metadata=_p("Q51", "good forecasting (§5.3 table)"))
    L: int | None = field(default=0, metadata=_p("DERIVED(§5.3 table)", "L = 0: good forecasting"))
    a_region: float | None = field(default=1.0, metadata=_p(_MRS, "a_informed = 1 by definition (§5.2 'Values')"))
    a_dyad: float | None = field(default=1.0, metadata=_p(_MRS, "a = 1 for regions and dyads (§5.3 table)"))
    a_chokepoint: float | None = field(default=1.0, metadata=_p("SYNTHETIC(prior)", _A_INF_CHK))
    skill_region: float | None = field(default=0.84, metadata=_p(_MRS, "AUROC_13 0.84 (0.82-0.87) (§5.2 'Values')"))
    skill_dyad: float | None = field(
        default=0.84, metadata=_p("SYNTHETIC(prior)", "the region value transferred to dyads (§5.2; Q80)")
    )
    skill_chokepoint: float | None = field(
        default=0.691, metadata=_p("DERIVED", _CHK_SKILL.format(value="0.691 (seeds 0 and 2: 0.6893, 0.6917)"))
    )
    phi_tariff_formal: float | None = field(default=0.25, metadata=_p(*_PHI_FORMAL))
    phi_tariff_informal: float | None = field(default=0.36, metadata=_p(*_PHI_INFORMAL))
    phi_ties_threat: float | None = field(default=0.538, metadata=_p(*_PHI_TIES))
    phi_mid_threat: float | None = field(default=0.298, metadata=_p(*_PHI_MID))
    lead_mode: str | None = field(default="record", metadata=_p("Q51", "record leads (§5.3 table)"))
    chi: bool = field(default=False, metadata=_p(_PLACEHOLDER, _CHI))
    pending: bool = field(default=True, metadata=_p(_NO_FEED, "pending_prohibitions: 'yes' in informed"))
    forecast: str = field(default="mmfe", metadata=_p(_NO_FEED, "the forecast (48) (Q55)"))
    forecast_shock: bool = field(default=True, metadata=_p(_PLACEHOLDER, _SHOCK))
    omega_at_reset: bool = field(default=False, metadata=_p(_NO_FEED, "full omega at reset: 'no'"))
    h_cov: int | None = _off("the coverage rung's radius")
    blackout: str | None = _off("the blackout rung's spell")


@dataclass(frozen=True)
class Standard(Theta):
    """The standard regime: the hackathon default and the proposed ranked regime (Q13).

    standard (§5.3 table; Q12, Q13, Q46, Q51, Q76): L = 1, a_std = 0.604, record phi and leads; the proposed
    ranked regime (Q13).
    """

    name: str = field(default="standard", metadata=_p("Q13", "hackathon default, proposed ranked regime (§5.3)"))
    L: int | None = field(default=1, metadata=_p("SYNTHETIC(prior)", "L = 1 (§5.3 table; Q76)"))
    a_region: float | None = field(default=0.604, metadata=_p("SYNTHETIC(prior)", _A_STD))
    a_dyad: float | None = field(default=0.604, metadata=_p("SYNTHETIC(prior)", _A_STD_KIND.format("dyad")))
    a_chokepoint: float | None = field(default=0.604, metadata=_p("SYNTHETIC(prior)", _A_STD_KIND.format("chokepoint")))
    skill_region: float | None = field(default=0.712, metadata=_p("DERIVED", _STD_SKILL.format(kind="region")))
    skill_dyad: float | None = field(default=0.712, metadata=_p("DERIVED", _STD_SKILL.format(kind="dyad")))
    skill_chokepoint: float | None = field(
        default=0.688, metadata=_p("DERIVED", _CHK_SKILL.format(value="0.688 (seeds 0 and 2: 0.6880, 0.6887)"))
    )
    phi_tariff_formal: float | None = field(default=0.25, metadata=_p(*_PHI_FORMAL))
    phi_tariff_informal: float | None = field(default=0.36, metadata=_p(*_PHI_INFORMAL))
    phi_ties_threat: float | None = field(default=0.538, metadata=_p(*_PHI_TIES))
    phi_mid_threat: float | None = field(default=0.298, metadata=_p(*_PHI_MID))
    lead_mode: str | None = field(default="record", metadata=_p("Q51", "record leads (§5.3 table)"))
    chi: bool = field(default=False, metadata=_p(_PLACEHOLDER, _CHI))
    pending: bool = field(default=True, metadata=_p(_NO_FEED, "pending_prohibitions: 'yes' in standard"))
    forecast: str = field(default="mmfe", metadata=_p(_NO_FEED, "the forecast (48) (Q55)"))
    forecast_shock: bool = field(default=True, metadata=_p(_PLACEHOLDER, _SHOCK))
    omega_at_reset: bool = field(default=False, metadata=_p(_NO_FEED, "full omega at reset: 'no'"))
    h_cov: int | None = _off("the coverage rung's radius")
    blackout: str | None = _off("the blackout rung's spell")


@dataclass(frozen=True)
class CoverageRung(PredictionFree):
    """The spatial-coverage rung of prediction-free.

    The spatial-coverage rung (§5.3 table; Q10, Q38; W2-information §2.3): prediction-free with the graph state
    shown only within ``h_cov`` hops of the policy's goods (``information.coverage``). Built by ``coverage_rung``.
    """

    name: str = field(default="coverage_0", metadata=_p("Q38", "coverage_<h_cov> (M3 registry)"))
    h_cov: int | None = field(
        default=0,
        metadata=_p("SYNTHETIC(prior: CTP convention; McKinsey 2021)", "h_cov in {0, 1, 2} (§5.3 table)", 0, 2),
    )


@dataclass(frozen=True)
class BlackoutRung(PredictionFree):
    """The blackout rung of prediction-free.

    The blackout rung (§5.3 table; Q10, Q68 E13; W2-information §2.4): prediction-free outside its spell; in the
    weeks of the spell of kind ``blackout`` every external feed is null and the own state is kept. Built by
    ``blackout_rung``.
    """

    name: str = field(default="blackout_3w", metadata=_p("Q10", "blackout_<spell kind name> (M3 registry)"))
    blackout: str | None = field(
        default="3w",
        metadata=_p(
            "SYNTHETIC(prior: JMIC, BEA, UN Comtrade)", "a spell kind of codes.BLACKOUT_SPELLS (§11 row 34; Q10)"
        ),
    )


# the four named regimes of the §5.3 table, with the design's values (the rungs come from coverage_rung, blackout_rung)
NAMED_REGIMES: dict[str, Theta] = {
    "clairvoyant": Clairvoyant(),
    "informed": Informed(),
    "standard": Standard(),
    "prediction_free": PredictionFree(),
}
# every name resolve_regime accepts: the named regimes, then coverage_<h> and blackout_<kind> for each level and kind
REGIME_NAMES: tuple[str, ...] = (
    *NAMED_REGIMES,
    *(f"coverage_{h}" for h in COVERAGE_LEVELS),
    *(f"blackout_{k}" for k in codes.BLACKOUT_SPELLS),
)


def coverage_rung(h: int) -> Theta:
    """The coverage rung ``coverage_<h>``: prediction-free with ``h_cov = h`` (§5.3 table; (47)).

    Raises:
        ValueError: if ``h`` is not in ``COVERAGE_LEVELS`` (a bool included).

    """
    if isinstance(h, bool) or not isinstance(h, int) or h not in COVERAGE_LEVELS:
        raise ValueError(f"h_cov of the coverage rung must be one of {COVERAGE_LEVELS} (§5.3 table), got {h!r}")
    return CoverageRung(name=f"coverage_{int(h)}", h_cov=int(h))


def blackout_rung(kind: str) -> Theta:
    """The blackout rung ``blackout_<kind>``: prediction-free with the spell kind named ``kind`` (47).

    ``kind`` is a name of ``codes.BLACKOUT_SPELLS``, as ``Theta.blackout``, ``view.blackout_weeks`` and the wire's
    ``BlackoutSpell`` carry it.

    Raises:
        ValueError: if ``kind`` is not a name of ``codes.BLACKOUT_SPELLS`` (a code, a bool included, is not).

    """
    if not isinstance(kind, str) or kind not in codes.BLACKOUT_SPELLS:
        raise ValueError(f"the blackout rung's spell kind must be one of {codes.BLACKOUT_SPELLS}, got {kind!r}")
    return BlackoutRung(name=f"blackout_{kind}", blackout=str(kind))


# every theta resolve_regime accepts by name, in REGIME_NAMES order
REGISTRY: dict[str, Theta] = {
    **NAMED_REGIMES,
    **{f"coverage_{h}": coverage_rung(h) for h in COVERAGE_LEVELS},
    **{f"blackout_{k}": blackout_rung(k) for k in codes.BLACKOUT_SPELLS},
}
assert tuple(REGISTRY) == REGIME_NAMES  # one order for the names and the registry


def resolve_regime(regime: str | Theta) -> Theta:
    """The theta of a registry name (``REGIME_NAMES``), or ``regime`` itself if it is a theta.

    A theta whose name is a registry name must equal the registry's theta of that name (class and fields), since
    ``Trajectory.regime`` would record the registry's name for it (a custom theta named 'standard' would be scored as
    standard); any other theta is recorded under ``regime_label``. ``check_theta`` runs on the result.

    Raises:
        ValueError: on an unknown name, a non-registry theta under a registry name, or a theta ``check_theta`` refuses.
        TypeError: if ``regime`` is neither a str nor a ``Theta``.

    """
    if isinstance(regime, str):
        theta = REGISTRY.get(str(regime))
        if theta is None:
            raise ValueError(f"unknown regime {regime!r}: one of {REGIME_NAMES}, or a Theta")
    elif isinstance(regime, Theta):
        theta = regime
        own = REGISTRY.get(theta.name) if isinstance(theta.name, str) else None
        if own is not None and theta != own:
            raise ValueError(
                f"a theta named {theta.name!r} must be the registry's (Trajectory.regime records the name): rename "
                "the custom theta"
            )
    else:
        raise TypeError(f"regime must be a registry name or a Theta, got {type(regime).__name__}")
    check_theta(theta)
    return theta


def _real(x) -> bool:
    """A finite real number that is not a bool (a NumPy bool included)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_theta(theta: Theta) -> None:
    """Refuse a theta outside (47)'s domain.

    ``name`` a non-empty str; L None or an int >= 0 (a bool is not); each a_<kind> None or a real in [-1, 1], every a
    set exactly when L is set; each skill None or a real in [0, 1], and None without a warning feed; each phi None or
    a real in [0, 1), all four None or none, and ``lead_mode`` in ``LEAD_MODES`` exactly when they are set (None
    otherwise); ``chi``, ``pending``, ``forecast_shock`` and ``omega_at_reset`` bools; ``forecast`` in ``FORECASTS``;
    ``h_cov`` None or in ``COVERAGE_LEVELS``; ``blackout`` None or a name of ``codes.BLACKOUT_SPELLS``.

    Raises:
        TypeError: if ``theta`` is not a ``Theta``.
        ValueError: naming the first field that breaks its rule.

    """
    if not isinstance(theta, Theta):
        raise TypeError(f"not a Theta: {type(theta).__name__}")
    if not isinstance(theta.name, str) or not theta.name:
        raise ValueError(f"Theta.name must be a non-empty str, got {theta.name!r}")
    L = theta.L
    if L is not None and (isinstance(L, bool) or not isinstance(L, int) or L < 0):
        raise ValueError(f"Theta.L must be None or an int >= 0 (45), got {L!r}")
    for kind in codes.UNIT_KINDS:
        a = getattr(theta, f"a_{kind}")
        if a is not None and not (_real(a) and -1.0 <= a <= 1.0):
            raise ValueError(f"Theta.a_{kind} must be None or a real number in [-1, 1] (45), got {a!r}")
        if (a is None) != (L is None):
            raise ValueError(f"Theta.a_{kind} must be set exactly when L is set (a warning feed on every unit kind)")
        skill = getattr(theta, f"skill_{kind}")
        if skill is not None and not (_real(skill) and 0.0 <= skill <= 1.0):
            raise ValueError(f"Theta.skill_{kind} must be None or an AUROC in [0, 1] (46), got {skill!r}")
        if skill is not None and L is None:
            raise ValueError(f"Theta.skill_{kind}: a declared skill needs a warning feed (L set)")
    phis = [getattr(theta, f"phi_{ch}") for ch in DECOY_CHANNELS]
    for ch, phi in zip(DECOY_CHANNELS, phis):
        if phi is not None and not (_real(phi) and 0.0 <= phi < 1.0):
            raise ValueError(f"Theta.phi_{ch} must be None or a real number in [0, 1) (49), got {phi!r}")
    if len({phi is None for phi in phis}) > 1:
        raise ValueError("Theta.phi_*: all four decoy shares are set, or none (no messages feed)")
    if (theta.lead_mode is None) != (phis[0] is None) or (
        theta.lead_mode is not None and theta.lead_mode not in LEAD_MODES
    ):
        raise ValueError(
            f"Theta.lead_mode must be one of {LEAD_MODES} exactly when the phi are set, got {theta.lead_mode!r}"
        )
    for name in ("chi", "pending", "forecast_shock", "omega_at_reset"):
        if not isinstance(getattr(theta, name), bool):
            raise ValueError(f"Theta.{name} must be a bool, got {getattr(theta, name)!r}")
    if theta.forecast not in FORECASTS:
        raise ValueError(f"Theta.forecast must be one of {FORECASTS}, got {theta.forecast!r}")
    h = theta.h_cov
    if h is not None and (isinstance(h, bool) or not isinstance(h, int) or h not in COVERAGE_LEVELS):
        raise ValueError(f"Theta.h_cov must be None or one of {COVERAGE_LEVELS}, got {h!r}")
    b = theta.blackout
    if b is not None and (not isinstance(b, str) or b not in codes.BLACKOUT_SPELLS):
        raise ValueError(f"Theta.blackout must be None or a spell kind of {codes.BLACKOUT_SPELLS}, got {b!r}")


def regime_label(theta: Theta) -> str:
    """The name ``Trajectory.regime`` records for ``theta`` (module docstring; (27) identity).

    A registry theta's name; any other theta's name, '#' and the first ``LABEL_HEX`` hex digits of the SHA-256 of the
    canonical JSON of its fields (floats by repr), so two custom thetas with different fields never share a label.
    """
    own = REGISTRY.get(theta.name) if isinstance(theta.name, str) else None
    if own is not None and theta == own:
        return theta.name
    return f"{theta.name}#{sha256_hex(dataclasses.asdict(theta))[:LABEL_HEX]}"


def a_by_kind(theta: Theta) -> dict[str, float]:
    """{unit_kind: a} for the kinds of ``codes.UNIT_KINDS`` whose a is set, in that order; {} without a warning."""
    return {k: getattr(theta, f"a_{k}") for k in codes.UNIT_KINDS if getattr(theta, f"a_{k}") is not None}


def phi_by_channel(theta: Theta) -> dict[str, float]:
    """{channel name: phi} for ``DECOY_CHANNELS`` whose phi is set, in that order; {} without a messages feed.

    A final notice and a legal publication belong to their thread, so their decoy share is their parent channel's
    (tariff proposal, TIES threat): they carry no phi of their own (symmetric channel sets, Q97).
    """
    return {ch: getattr(theta, f"phi_{ch}") for ch in DECOY_CHANNELS if getattr(theta, f"phi_{ch}") is not None}


# M5-O16 (a), the owner queue's default until answered (2026-09-28): the declared chokepoint skill per instance kind.
# `tiny` keeps V27's measurement (the ``skill_chokepoint`` defaults of Informed and Standard); on `small` and `full`,
# which carry seven chokepoint units per world, V27 has not measured it yet, so each value is `tiny`'s carried over as
# SYNTHETIC(placeholder) (``CHOKEPOINT_SKILL_PROVENANCE``), which keeps V23 from sealing a split that publishes it.
CHOKEPOINT_SKILL_BY_KIND: dict[str, dict[str, float]] = {
    "small": {"informed": 0.691, "standard": 0.688},
    "full": {"informed": 0.691, "standard": 0.688},
}
CHOKEPOINT_SKILL_PROVENANCE = _p(
    "SYNTHETIC(placeholder)",
    "the declared chokepoint skill on `small` and `full` (Q80): `tiny`'s V27 value carried over until V27's chokepoint"
    " units are re-measured per size with seven units per world (owner queue M5-O16 (a); design §12 M5 generator rows)",
)
CUSTOM_CHOKEPOINT_SKILL_PROVENANCE = _p(
    "SYNTHETIC(placeholder)",
    "a custom theta's own declared chokepoint skill on `small` or `full` (Q80): the caller's value, published as the"
    " theta holds it; V27 has measured the chokepoint skill on `tiny` only (M5-O16 (a); design §12 row 'Declared"
    " chokepoint skill per kind')",
)


def declared_skill(theta: Theta, kind: str = "tiny") -> tuple[float | None, dict]:
    """(value, provenance record) of theta's declared chokepoint skill on an instance of ``kind`` (Q80; M5-O16 (a)).

    A registry regime that declares one (informed, standard: equal to ``REGISTRY[theta.name]``, the identity of
    ``regime_label``) takes ``CHOKEPOINT_SKILL_BY_KIND`` on `small` and `full`, tagged ``CHOKEPOINT_SKILL_PROVENANCE``;
    any other theta of (47) keeps its own value there, tagged ``CUSTOM_CHOKEPOINT_SKILL_PROVENANCE``, so the one
    publisher never shows a value the theta does not hold (INT-M5-01); `tiny`, and every theta without a declared
    value, keep the field and its own provenance.

    Raises:
        ValueError: on a flagship kind whose table lacks a registry regime that declares a skill.

    """
    own = (theta.skill_chokepoint, _provenance(theta)["skill_chokepoint"])
    if kind == "tiny" or theta.skill_chokepoint is None or kind not in CHOKEPOINT_SKILL_BY_KIND:
        return own
    if REGISTRY.get(theta.name) != theta:
        return theta.skill_chokepoint, dict(CUSTOM_CHOKEPOINT_SKILL_PROVENANCE)
    table = CHOKEPOINT_SKILL_BY_KIND[kind]
    if theta.name not in table:
        raise ValueError(f"no declared chokepoint skill for regime {theta.name!r} on kind {kind!r} (M5-O16)")
    return table[theta.name], dict(CHOKEPOINT_SKILL_PROVENANCE)


def static_regime(theta: Theta, kind: str = "tiny") -> dict:
    """``Static.regime`` of §9.2: the published theta (Q51 X10), the one publisher.

    {"name", "L", "a": a_by_kind, "phi": phi_by_channel, "chi", "h_cov", "skill": {unit_kind: float | None} for the
    kinds with a warning ({} without one), "blackout": the spell kind or None}. prediction-free gives the §12 form plus
    the added keys: {"name": "prediction_free", "L": None, "a": {}, "phi": {}, "chi": False, "h_cov": None,
    "skill": {}, "blackout": None}; clairvoyant publishes every skill null (it declares none, design §12). A runner
    that scores several regimes in one run sends null instead (its ``hide_regime`` flag, design §12 M3 rows). The
    chokepoint skill is the instance kind's (``declared_skill``; `tiny`'s by default, M5-O16), a custom theta's own.
    """
    a = a_by_kind(theta)
    skill = {k: getattr(theta, f"skill_{k}") for k in a}
    if "chokepoint" in skill:
        skill["chokepoint"] = declared_skill(theta, kind)[0]
    return {
        "name": theta.name,
        "L": theta.L,
        "a": a,
        "phi": phi_by_channel(theta),
        "chi": theta.chi,
        "h_cov": theta.h_cov,
        "skill": skill,
        "blackout": theta.blackout,
    }

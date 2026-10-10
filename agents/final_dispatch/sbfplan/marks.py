"""Weekly marks: omega -> the arrays the simulator and the oracle LP both read (design §3.1, §3.4, §3.5, §4.4; V3).

This is the **one module** that turns the exogenous scenario into weekly edge weights and capacities (V3, R1 practice
2): the simulator (`dynamics/`) and the LP (`oracle/`) never read events themselves; they read a `WeeklyMarks`.

Timing (1), Q69: week t covers [t-1, t). *Week averages* over [t-1, t) (capacities, open fractions, availability,
deliverable energy, base load, restoration, ceiling) are pro-rated by the overlap rule (37); *binary marks*
(prohibitions, tariffs, piracy surcharge, war-risk class) are in force in week t iff w(s) <= t < w(s_end), with
w(s) = ceil(s) + 1; *instantaneous* values at the instant t-1 feed the observation of week t (`*_now`).

Array convention: axis 0 is the week, index t-1 for week t = 1..T. Other axes use the instance's integer codes: edges
E, commodities K, stock slots S (``Instance.stock_slots`` order), and the ordinals of chokepoints C, grids G, fabs F and
demands D (``Instance.demands`` order). Pools are (tb, ct) = (0, 1).

Readings of the §4.4 type table where the design leaves the rule open (conservative choices, milestone M1):

- Regions of nodes are their instance regions, chokepoints included (`chk` is GULF). "An edge between regions A and B"
  has one end in A and the other in B, either direction; coupling edges (mode grid) take no mark and keep u = inf.
- Time-varying severity (war profile (40)): the week factor is 1 - ∫σ_q(s) ds over the week's part of D_q, the (37)
  pro-rating with σ^t_q f^t_q read as an integral, so a week straddling W0 and W1 loses 0.60 f0 + 0.73 f1.
- Regional conflict: a region target only (m_q is that region; any other target kind raises); edge operations only
  with a drawn counterpart (none if -1); the war-risk class uses the event's own duration window [on, on + T_q) and the
  fab hits the dead-time window [on, on + T0) (T0 falling back to T_q when NaN), both on the event region m_q only
  (not the counterpart); the fab severity is the event's ``severity``. The CN–TW Taiwan Strait closure and the
  demand shock Xi are not marks here
  (the closure is its own event; Xi is in omega's ``d``).
- Fab events E_f: only regional conflicts act on fabs, on the fabs of m_q (§4.4; no other type has a fab operation).
  An event with target kind node on a fab applies its own graph operation, or raises if its type defines none. Dead
  time T0 falls back to the event duration when NaN.
- OSAT restoration (§4.4: "fabs and OSATs in m_q get (13)"; milestone M2, design §12 'OSAT restoration under a
  regional conflict'): ``R_osat`` (T, O) is (13)'s restoration factor on every OSAT of m_q of every regional conflict,
  by the rule of R_f and the same helpers (the event's severity, dead time T0 falling back to T_q, tail tau_rho, shape
  g_rho; several conflicts multiply in row order, (37)), with or without a counterpart; no other type acts on an OSAT.
  It scales the throughput of (19), thr_i R_osat_i(t), in the simulator and the LP alike. OSATs get no ceiling
  alpha-bar and no scrap (14): the design gives OSATs neither.
- Scrap (14): only hits inside the episode scrap, onset instant in (0, T): carried-in events (onset <= 0) scrap
  nothing, not even initial WIP (§2.3), and no scrap is booked after T, so the WIP credit of (23) is net of scrap
  booked by T in the simulator and the LP alike (Q56). ``fab_hits`` holds exactly those hits.
- War-risk class: from active events only (Q43): militarised closures on c and regional conflicts in a region adjacent
  to c; the conflict layer z_c alone sets no class, so omega^0 has class none throughout (57).
- Friendly fire: the dyad is (m_q, counterpart) for the region variant, (target node's region, counterpart) for the
  node variant (counterpart -1: every other region; the event's own region plays no part there) and (tail region, head
  region) for the edge variant; the sanctioned set is the event commodity, else K_e of the target edge (edge variant)
  or every commodity (then no edge escapes and no friendly fire applies).
- Tariffs and sanctions mark (e, k) only for k in K_e; with chokepoints in their node region, a TW tariff on GULF also
  taxes `chk` -> `grid_tw` cargo that entered `chk` from RU (edge-based tariffs do not track origin). A node sanction
  covers the node's out-edges into the counterpart, or into every *other* region when it is -1 (as the region variant
  and the friendly-fire dyad read -1), so an out-edge inside the node's own region stays open. A tariff rate is >= 0.
- Piracy multiplies c on every edge into or out of each targeted chokepoint (the region's chokepoints by
  ``chokepoint_adjacency``), so a whole transit costs (1 + rate) more; when the rate is NaN the factor is 1 + severity.
  The surcharge (the rate, or the severity without one) is >= 0, so no freight turns negative.
- Event fields: NaN means absent where a rule reads it so (the piracy rate, T0, tau_rho, and any field a type does not
  read); onset and duration are required. An infinite field raises ValueError instead of being read as NaN: the design
  gives infinity no reading, and the injected builder cannot store it.
- ``MarkParams`` refuses values outside their ranges when built (ValueError naming the field): the Hormuz-2026
  threshold, the war-profile losses (40) and sigma_ff lie in [0, 1], the war-profile window is a finite number of
  weeks > 0, the restoration shape is 'exp' or 'linear'; NaN, bools and strings in numeric fields are refused.
- Port strike: sea edges with a non-chokepoint end in the struck region (region target) or at the struck node.
- Restoration (13): the closed form of the week average theta-bar is clipped to [0, 1], the range of theta_q, since
  the dead-time share plus the tail rounds one or two ulps above 1 for a huge tau_rho (about 1e15 weeks and more); so
  every factor 1 - sigma_q theta-bar, R_f, R_osat and R_now included, lies in [0, 1] (ORACLE-M2-1).
- The stored ``R_f``, ``R_osat``, ``alpha_bar`` and ``sigma_scr`` of omega (26), when present, are the realisation the
  marks carry (owner decision Q91, option (a)): ``compute_marks`` reads them bit for bit and checks each entry against
  its recomputation from the events under omega's ``MarkParams``, |stored - recomputed| <= 1e-9 max(1, |recomputed|),
  the row tolerance of (53) (``STORED_MARK_TOL``), so an omega built where libm rounds exp or expm1 of (13) one ulp
  apart is accepted as stored, while arrays stored under other parameters (off far beyond it) are refused with
  ValueError (V3) and never mix with ``R_now`` and the fab hits recomputed here. A stored ``R_f`` or ``R_osat`` must
  also lie in [0, 1], the range of (13)'s factors, however close to its recomputation. The integer ``wr_class`` must
  equal its recomputation exactly. ``alpha_now`` is the week array shifted by one week. Without stored arrays the
  recomputation is the realisation.
- A target kind that the type table does not define for a type raises ValueError rather than being ignored.
- omega must carry ``meta_instance_hash``, ``meta_instance_digest`` and ``meta_mark_params`` (every builder writes
  them): the marks are bound to the instance content and to omega's own parameters, never to a caller's pick (V3).
- ``WeeklyMarks`` are immutable: every array is a private read-only copy over an immutable buffer, so the cached
  ``digest`` always states the content the simulator and the LP read (V3 by content).

The disruption stage reads events through the same rules, never a copy of them: ``read_events`` (omega's rows as
``Event``), ``graph_marks`` and ``restoration_factors`` (the marks of a set of events over weeks 1..T),
``event_capacity_marks`` (the capacity marks of one event alone), ``fab_targets`` and ``osat_targets`` (who a
restoration event hits), ``restoration_regions`` (the regions whose conflicts get (13)), ``strike_edges`` (where a port
strike acts) and ``war_windows`` (the windows of (40)). Harm (41), the realism overlap count, the event stage and the
Poisson components' strike rule use them.
"""

import dataclasses
import functools
import json
import math
import numbers
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from sbfplan.digest import semantic_digest
from sbfplan.instance.schema import FrozenDict, Instance
from sbfplan.omega import codes
from sbfplan.omega.container import EVENT_FLOATS, Omega, immutable_copy


# Floating-point error handling of every marks computation, fixed here so that NumPy's process-global state
# (np.seterr) never changes a result or raises one (DET-7). Underflow is expected and ignored: the restoration tail
# exp(-x) of (13) reaches 0 through the subnormals for x > 708, which is its value. Division by zero, overflow and
# invalid operations do not occur on accepted inputs, so they raise rather than yield inf or NaN marks; the one
# exception is x / tau_rho of (13), whose overflow for a subnormal tau_rho is its exact limit (``_over_tau``).
FP_ERRORS: Mapping[str, str] = FrozenDict(divide="raise", over="raise", under="ignore", invalid="raise")

# The row tolerance of (53), the value of oracle.replay.ROW_TOL (stated here, since the oracle imports this module): a
# stored R_f, alpha_bar or sigma_scr of omega is accepted when |stored - recomputed| <= STORED_MARK_TOL
# max(1, |recomputed|) entry by entry (owner decision Q1, option (a); design §12 'Mark parameters in omega').
STORED_MARK_TOL = 1e-9


def fixed_fp_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Run ``fn`` under ``np.errstate(**FP_ERRORS)``, whatever the caller's NumPy error state (DET-7)."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with np.errstate(**FP_ERRORS):
            return fn(*args, **kwargs)

    return wrapper


_RESTORATION_SHAPES = ("exp", "linear")  # g_rho of (13): exp(-x), the DIIM form, or (1 - x)^+ (§3.5 table)
_UNIT_FIELDS = ("hormuz_threshold", "war_profile_belligerent", "war_profile_third_party", "friendly_fire")


def _real(name: str, value: object) -> float:
    """``value`` as a float if it is a real number; a bool, a string or any other type raises TypeError naming it.

    A real number beyond the float range (an integer such as 10**400) lies outside every field's range, so it raises
    ValueError naming the field rather than the bare OverflowError of ``float`` (DOC-4).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"MarkParams.{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except OverflowError as err:
        raise ValueError(f"MarkParams.{name} lies beyond the float range, outside its range: {value!r}") from err


@dataclass(frozen=True)
class MarkParams:
    """Parameters of the event-to-graph rules of §4.4 that the instance does not carry (generator family values).

    Each field carries its tag in ``metadata["provenance"]``; they enter ``generator_id`` with the rest of the
    resolved generator config (§4.1). Ranges, checked when built: ``hormuz_threshold`` in [0, 1], a threshold on
    sigma_q (§3.4 table); ``war_profile_belligerent`` (two losses) and ``war_profile_third_party`` (one) in [0, 1],
    the severities of (40); ``war_profile_window`` a finite length > 0 in weeks (§11 row 57); ``friendly_fire`` in
    [0, 1], a severity of (37); ``restoration_shape`` 'exp' or 'linear' (§3.5 table).
    """

    hormuz_threshold: float = field(
        default=0.8,
        metadata={
            "provenance": "SYNTHETIC(prior: PortWatch transit ratios) — Hormuz-2026 class iff sigma >= 0.8 (Q43)"
        },
    )
    war_profile_belligerent: tuple[float, float] = field(
        default=(0.60, 0.73),
        metadata={"provenance": "DERIVED(Glick–Taylor 2010 App. Table A1): 1-0.40, 1-0.27 on W0, W1 (40)"},
    )
    war_profile_third_party: float = field(
        default=0.10, metadata={"provenance": "DERIVED(Glick–Taylor 2010): 1-0.90 on W0 (40)"}
    )
    war_profile_window: float = field(
        default=52.0, metadata={"provenance": "SYNTHETIC(prior: Glick–Taylor lags) §11 row 57"}
    )
    friendly_fire: float = field(
        default=0.75,
        metadata={"provenance": "SYNTHETIC(placeholder): midpoint of REAL range 0.65-0.85 (v0.2), §11 row 26"},
    )
    restoration_shape: str = field(
        default="exp",
        metadata={"provenance": "SYNTHETIC(placeholder): g_rho(x) = exp(-x) (DIIM form), or 'linear' (1-x)^+ (§3.5)"},
    )

    def __post_init__(self) -> None:
        """Canonical numeric types (DET-9) and the ranges of the class docstring (INT-5).

        Float fields become floats and tuple fields tuples of floats, so ``friendly_fire=1`` states the config
        ``friendly_fire=1.0`` does and equal parameters give one ``meta_mark_params`` and one ``generator_id``.

        Raises:
            TypeError: if a numeric field or an entry of a tuple field is not a real number (a bool included), a tuple
                field is not a tuple or list, or ``restoration_shape`` is not a string; the message names the field.
            ValueError: if a value lies outside its range (NaN and a number beyond the float range included) or a tuple
                field has the wrong length; the message names the field.

        """
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.type is float:
                object.__setattr__(self, f.name, _real(f.name, v))
            elif typing.get_origin(f.type) is tuple:
                if not isinstance(v, (tuple, list)):
                    raise TypeError(f"MarkParams.{f.name} must be a tuple of real numbers, got {v!r}")
                n = len(typing.get_args(f.type))
                if len(v) != n:
                    raise ValueError(f"MarkParams.{f.name} must hold exactly {n} values, got {len(v)}")
                object.__setattr__(self, f.name, tuple(_real(f.name, x) for x in v))
        for name in _UNIT_FIELDS:
            v = getattr(self, name)
            if not all(0.0 <= x <= 1.0 for x in (v if isinstance(v, tuple) else (v,))):  # NaN fails too
                raise ValueError(f"MarkParams.{name} must lie in [0, 1], got {v}")
        if not 0.0 < self.war_profile_window < math.inf:  # NaN fails too
            raise ValueError(
                f"MarkParams.war_profile_window must be a finite number of weeks > 0, got {self.war_profile_window}"
            )
        if not isinstance(self.restoration_shape, str):
            raise TypeError(f"MarkParams.restoration_shape must be a string, got {self.restoration_shape!r}")
        if self.restoration_shape not in _RESTORATION_SHAPES:
            raise ValueError(
                f"MarkParams.restoration_shape must be one of {_RESTORATION_SHAPES}, got {self.restoration_shape!r}"
            )


def mark_params_json(params: MarkParams) -> str:
    """Canonical JSON of MarkParams (design §2.3 canonical form), as omega's ``meta_mark_params`` stores it."""
    return json.dumps(dataclasses.asdict(params), sort_keys=True, separators=(",", ":"), allow_nan=False)


def mark_params_from_json(text: str) -> MarkParams:
    """Inverse of ``mark_params_json``; list fields become the dataclass's tuples.

    Raises:
        ValueError: if ``text`` is not a JSON object with exactly the fields of ``MarkParams``, or holds a value that
            ``MarkParams`` refuses (out of range, or of the wrong type).

    """
    raw = json.loads(text)
    names = {f.name for f in dataclasses.fields(MarkParams)}
    if not isinstance(raw, dict) or set(raw) != names:
        raise ValueError(f"meta_mark_params must be a JSON object with exactly the MarkParams fields {sorted(names)}")
    kw = {
        f.name: tuple(raw[f.name]) if isinstance(raw[f.name], list) else raw[f.name]
        for f in dataclasses.fields(MarkParams)
    }
    try:
        return MarkParams(**kw)
    except TypeError as err:  # a stored value of the wrong type makes omega malformed, as a missing field does
        raise ValueError(f"meta_mark_params: {err}") from err


@dataclass(frozen=True)
class FabHit:
    """An event acting on a fab (E_f of (13), (14)): the simulator books its scrap in the onset week (Q44, Q69)."""

    fab: int  # fab ordinal
    onset: float  # t^on_q in weeks (continuous)
    severity: float  # sigma_q

    @property
    def onset_week(self) -> int:
        """Onset week t_q = floor(t^on_q) + 1 of (14)."""
        return int(np.floor(self.onset)) + 1


@dataclass(frozen=True)
class WeeklyMarks:
    """Every exogenous weekly quantity of one episode (omega + instance), shared by simulator and LP (V3).

    Immutable: construction replaces every array field by a private read-only copy over an immutable buffer (the
    caller's arrays stay writable and unshared) and every tuple field by a tuple, so no field can change after the
    ``digest`` is cached. ``dataclasses.replace`` builds new marks with their own digest; a copy or an unpickled object
    is rebuilt through the constructor, so it carries no cached digest over (V3 by content). ``generator_id`` and
    ``generated`` record the generator that drew omega (``compute_marks``), so a reset from the marks alone runs the
    same fallback guard as one from omega (§8.1; INT-M2-R1-1); they are outside the digest (``compare=False``).
    """

    T: int
    instance_hash: str
    omega_hash: str
    # ----- week averages over [t-1, t), pro-rated by (37) ---------------------------------------------------------
    u: np.ndarray  # (T, E) capacity u^t_e; np.inf on coupling edges
    c: np.ndarray  # (T, E) unit freight c^t_e (c^0_e times piracy surcharges); tariffs are separate
    o: np.ndarray  # (T, C) open fraction o^t_c (37)
    kappa: np.ndarray  # (T, C, 2) throughput kappa^t_cb = (k_c mu_cb) o^t_c per pool (9)
    supply: np.ndarray  # (T, S) availability varsigma-bar^t_ik per stock slot; 0 at non-supply slots (Q79)
    G_bar: np.ndarray  # (T, G) deliverable energy G-bar^t_g
    y_bar: np.ndarray  # (T, G) base load y-bar^t_g
    R: np.ndarray  # (T, F) restoration factor R_f(t) (13)
    alpha_bar: np.ndarray  # (T, F) overproduction ceiling alpha-bar_f(t) (13)
    sigma_scr: np.ndarray  # (T, F) scrap share of lots started in week t (14)
    R_osat: np.ndarray  # (T, O) OSAT restoration factor (13) of §4.4, scaling thr_i in (19) (milestone M2)
    demand: np.ndarray  # (T, D) demand d^t_ik (21), from omega["d"]
    # ----- binary marks in force in week t: w(s) <= t < w(s_end) ---------------------------------------------------
    prohibited: np.ndarray  # (T, E, K) bool, Z_t including prohibitions_at_reset (3)
    tariff: np.ndarray  # (T, E, K) ad valorem rate tau^tar,t_ek; the cost is rate * v_k per unit (23)
    wr_class: np.ndarray  # (T, C) int8 war-risk class 0 none, 1 red_sea, 2 hormuz_2026 (§3.4 table)
    h_queue: np.ndarray  # (T, C, K) queue holding h^Q,t_ck by the week's war-risk class (§3.4)
    c_wr: np.ndarray  # (T, E, K) per-transit war-risk cost (11) on lane-continuing out-edges of chokepoints
    # ----- instantaneous values at the instant t-1 (observation of week t, §3.1 table) -----------------------------
    u_now: np.ndarray  # (T, E)
    o_now: np.ndarray  # (T, C)
    kappa_now: np.ndarray  # (T, C, 2) throughput (k_c mu_cb) o_c(t-1) per pool (9)
    supply_now: np.ndarray  # (T, S)
    G_bar_now: np.ndarray  # (T, G)
    y_bar_now: np.ndarray  # (T, G)
    R_now: np.ndarray  # (T, F)
    alpha_now: np.ndarray  # (T, F) alpha-bar_f(t-1), the ceiling at the end of the previous week (alpha-bar(0) = 1)
    # ----- per-event data the simulator needs -----------------------------------------------------------------------
    fab_hits: tuple[FabHit, ...] = ()  # hits that scrap (onset instant in (0, T)), booked in the onset week (14)
    w_scr: tuple[int, ...] = ()  # scrap window per fab ordinal (copied from the instance)
    instance_digest: str = ""  # content digest of the instance the marks were computed for (V3)
    # ----- the generator that drew omega (§4.1), for the D9 fallback guard of ``dynamics.env`` (§8.1) ----------------
    # Bookkeeping, not content: outside ``digest`` (compare=False), so the simulator and the LP read the same numbers
    # whatever they say and no digest moves; ``compute_marks`` sets them from omega, hand-built marks keep the defaults
    generator_id: str = field(default="", compare=False)  # omega's ``meta_generator_id``; "" without an omega
    generated: bool = field(default=False, compare=False)  # ``Omega.generated``: omega stores regime paths

    def __post_init__(self) -> None:
        """Freeze the content (V3 by content): read-only private copies of the arrays, tuples for the tuple fields.

        Raises:
            TypeError: if a field declared as an array does not hold a NumPy array.

        """
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.type is np.ndarray:
                if not isinstance(v, np.ndarray):
                    raise TypeError(f"WeeklyMarks.{f.name} must be a NumPy array, got {type(v).__name__}")
                object.__setattr__(self, f.name, immutable_copy(v))
            elif typing.get_origin(f.type) is tuple:
                object.__setattr__(self, f.name, tuple(v))

    def __reduce__(self):
        """Pickle and copy as the field values; the rebuild goes through the constructor (re-frozen, no digest)."""
        return (type(self), tuple(getattr(self, f.name) for f in dataclasses.fields(self)))

    @cached_property
    def digest(self) -> str:
        """SHA-256 of every field of the marks (V3: simulator and LP must read the same marks).

        Cached: the fields are frozen and every array is read-only over an immutable buffer, so it cannot go stale.
        """
        return semantic_digest(self)

    def scrap_share(self, fab: int, start_week: int) -> float:
        """sigma^scr_f(start_week) of (14) for any start week, including the initial WIP's start weeks <= 0."""
        return _scrap_share(self.fab_hits, self.w_scr, fab, start_week)


def _scrap_share(hits, w_scr, fab: int, start_week: int) -> float:
    """sigma^scr_f(s) = 1 - prod(1 - sigma_q) over the hits on fab f with onset week in (s, s + w^scr_f] (14).

    ``hits`` holds only the hits that scrap (``FabHit`` construction in ``_fab_marks``).
    """
    keep = 1.0
    for h in hits:
        if h.fab == fab and start_week < h.onset_week <= start_week + w_scr[fab]:
            keep *= 1.0 - h.severity
    return 1.0 - keep


def overlap(t: int, start: float, end: float) -> float:
    """f^t_q = |[t-1, t) ∩ [start, end)|, the share of week t inside an event window (37)."""
    return max(0.0, min(float(t), end) - max(float(t - 1), start))


def open_fraction(T: int, windows) -> np.ndarray:
    """(T,) the week average o^t, t = 1..T, of closures ``windows`` = (onset, end, sigma) on one chokepoint (37).

    The product of each closure's surviving share 1 - sigma f^t_q in the order given, formed exactly as
    ``compute_marks`` forms a chokepoint's ``o`` (``_factors``, one rule table with V3), so on the same closures in the
    same order the two agree bit for bit. Naive's o^tr of (66) reads it for the transient closures (SIMP-M2R2-02).
    Nothing is validated here (the marks check the severities when they read events).
    """
    t = _weeks(T)
    o = np.ones(T, dtype=np.float64)
    for start, end, sig in windows:
        o *= _factors(t, [(start, end, sig)])[0]
    return o


def osat_throughput(inst: Instance, R: np.ndarray) -> np.ndarray:
    """thr_i R^osat_i of (19) under (13), elementwise over OSAT ordinals on the last axis of ``R`` ((T, O) or (O,)).

    The one home of the OSAT capacity (SIMP-M2R2-10): the simulator's packaging (19) and the LP's OSAT bound or row
    read it on the week average ``R_osat`` (thr_i R_osat_i(t), V3), and the observation's ``graph_now.osat.thr_eff`` on
    the instant ``R_osat_now`` (Q97). One IEEE product per entry, so thr x 1.0 = thr exactly.
    """
    thr = inst.memo("marks.osat_thr", _osat_thr)
    return thr * np.asarray(R, dtype=np.float64)


def _osat_thr(inst: Instance) -> np.ndarray:
    """(O,) thr_i per OSAT ordinal, read-only (kept in ``Instance.memo``)."""
    return immutable_copy(np.array([inst.nodes[o].osat.thr for o in inst.osats], dtype=np.float64))


@fixed_fp_errors
def event_free_marks(inst: Instance) -> WeeklyMarks:
    """Marks of the event-free scenario with constant demand d-bar.

    A fixture helper, and omega^0's marks when the demand noise is zero: nominal capacities, open chokepoints, class
    none, Z_t = Z_0 every week.
    """
    T, E, K = inst.T, len(inst.edges), len(inst.commodities)
    C, F, O = len(inst.chokepoints), len(inst.fabs), len(inst.osats)
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    c0 = np.array([e.c0 for e in inst.edges])
    k_mu = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints])
    sup = np.array([s.supply for s in inst.stock_slots])
    Gb = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
    yb = np.array([inst.nodes[g].grid.base_load for g in inst.grids])
    Z = np.zeros((T, E, K), dtype=bool)
    for e, k in inst.prohibitions_at_reset:
        Z[:, e, k] = True
    hq = np.zeros((T, C, K))
    for ci, c in enumerate(inst.chokepoints):
        for k, row in inst.nodes[c].chokepoint.queue_holding.items():
            hq[:, ci, k] = row[0]
    rep = lambda a: np.broadcast_to(a, (T,) + a.shape).copy()  # noqa: E731
    ones_c, ones_f = np.ones((T, C)), np.ones((T, F))
    kappa = rep(k_mu.reshape(C, 2)) * 1.0  # (k_c mu_cb) o with o = 1 (9)
    return WeeklyMarks(
        T=T,
        instance_hash=inst.hash,
        instance_digest=inst.content_digest,
        omega_hash="",
        u=rep(u0),
        c=rep(c0),
        o=ones_c.copy(),
        kappa=kappa,
        supply=rep(sup),
        G_bar=rep(Gb),
        y_bar=rep(yb),
        R=ones_f.copy(),
        alpha_bar=ones_f.copy(),
        sigma_scr=np.zeros((T, F)),
        R_osat=np.ones((T, O)),
        demand=rep(np.array([d.dbar for d in inst.demands])),
        prohibited=Z,
        tariff=np.zeros((T, E, K)),
        wr_class=np.zeros((T, C), dtype=np.int8),
        h_queue=hq,
        c_wr=np.zeros((T, E, K)),
        u_now=rep(u0),
        o_now=ones_c.copy(),
        kappa_now=kappa.copy(),
        supply_now=rep(sup),
        G_bar_now=rep(Gb),
        y_bar_now=rep(yb),
        R_now=ones_f.copy(),
        alpha_now=ones_f.copy(),
        fab_hits=(),
        w_scr=tuple(inst.nodes[f].fab.w_scr for f in inst.fabs),
    )


# ----- event rows (§4.1 table) --------------------------------------------------------------------------------------
(
    TARIFF,
    SANCTION,
    MATERIAL_OUTAGE,
    MILITARISED_CLOSURE,
    REGIONAL_CONFLICT,
    PIRACY,
    ENERGY_SHOCK,
    WEATHER_CLOSURE,
    PORT_STRIKE,
) = (
    codes.EVENT_TYPES.index(name)
    for name in (
        "tariff",
        "sanction",
        "material_outage",
        "militarised_closure",
        "regional_conflict",
        "piracy",
        "energy_shock",
        "weather_closure",
        "port_strike",
    )
)
K_CHOKEPOINT, K_EDGE, K_NODE, K_REGION = (codes.TARGET_KINDS.index(k) for k in ("chokepoint", "edge", "node", "region"))


@dataclass(frozen=True)
class Event:
    """One row of omega's ``ev_*`` arrays as the rule table reads it, with the region m_q resolved.

    A region target is its own region. Public so that the disruption stage (harm (41), the strike rule of the Poisson
    components) reads events through this module's rules (``read_events``) rather than re-coding them.
    """

    row: int
    type: int
    region: int  # m_q, -1 if neither ev_region nor a region target gives it
    counterpart: int  # -1 if none (tariffs, sanctions: every other region)
    target_kind: int
    target: int
    commodity: int  # -1 if all
    onset: float
    duration: float
    severity: float
    rate: float
    T0: float
    tau_rho: float

    @property
    def end(self) -> float:
        """End of the window D_q = [onset, onset + T_q) of (37)."""
        return self.onset + self.duration

    @property
    def name(self) -> str:
        return _event_name(self.row, self.type, self.target_kind)


def _event_name(row: int, type_: int, target_kind: int) -> str:
    """How an error message names an event: its row, type and target kind."""
    return f"event {row} ({codes.EVENT_TYPES[type_]}, target kind {codes.TARGET_KINDS[target_kind]})"


def read_events(inst: Instance, arrays: Mapping[str, np.ndarray], prefix: str = "ev") -> tuple[Event, ...]:
    """The event list E of omega (carried-in events included), in row order, with codes checked against the instance.

    ``arrays`` is omega's array mapping or just its ``ev_*`` group (no ``ev_type``: no events). ``prefix`` 'sh' reads
    the shadow group Y by the same rules instead (M3, SIMP-M2R2-01: one decoder), for the trusted side's message
    table; the marks, the simulator and the LP read ``ev_*`` only, so a shadow never acts (V3).

    Raises:
        ValueError: on an unknown type or target kind, an index out of range, a region target that disagrees with
            ``ev_region``, a counterpart equal to the event region, an infinite field, a non-finite onset or a NaN or
            negative duration.
        ValueError: on a ``prefix`` other than 'ev' or 'sh'.

    """
    if prefix not in ("ev", "sh"):
        raise ValueError(f"read_events reads the ev_ or the sh_ group, got prefix {prefix!r}")
    if f"{prefix}_type" not in arrays:
        return ()
    n_nodes, n_edges, n_regions, n_k = len(inst.nodes), len(inst.edges), len(inst.regions), len(inst.commodities)
    col = lambda name, i: arrays[f"{prefix}_{name}"][i]  # noqa: E731
    out = []
    for i in range(int(arrays[f"{prefix}_type"].shape[0])):
        ty, kind, target = int(col("type", i)), int(col("target_kind", i)), int(col("target", i))
        region, cp, com = int(col("region", i)), int(col("counterpart", i)), int(col("commodity", i))
        fl = {name: float(col(name, i)) for name in EVENT_FLOATS}  # the float columns compute_marks reads
        for name, v in fl.items():  # NaN may mean absent (§12 rows); infinity has no reading, so it is refused
            if math.isinf(v):
                raise ValueError(
                    f"event {i}: {prefix}_{name} is {v}; an event field is finite, or NaN where it may be absent"
                )
        onset, duration = fl["onset"], fl["duration"]
        if not 0 <= ty < len(codes.EVENT_TYPES):
            raise ValueError(f"event {i}: unknown event type {ty}")
        if not 0 <= kind < len(codes.TARGET_KINDS):
            raise ValueError(f"event {i}: unknown target kind {kind}")
        if ty == REGIONAL_CONFLICT and kind != K_REGION:  # §4.4: it acts on its region m_q, which the target names
            raise ValueError(
                f"event {i}: a regional conflict targets a region, got target kind {codes.TARGET_KINDS[kind]!r}"
            )
        bound = {K_CHOKEPOINT: n_nodes, K_NODE: n_nodes, K_EDGE: n_edges, K_REGION: n_regions}[kind]
        if not 0 <= target < bound:
            raise ValueError(f"event {i}: target {target} out of range for kind {codes.TARGET_KINDS[kind]}")
        if kind == K_REGION:
            if region not in (-1, target):
                raise ValueError(f"event {i}: region {region} differs from its region target {target}")
            region = target
        if not -1 <= region < n_regions or not -1 <= cp < n_regions or not -1 <= com < n_k:
            raise ValueError(f"event {i}: region, counterpart or commodity code out of range")
        if cp >= 0 and cp == region:
            raise ValueError(f"event {i}: counterpart equals the event region {region}")
        if not math.isfinite(onset) or math.isnan(duration) or duration < 0:
            raise ValueError(f"event {i}: onset must be finite and duration >= 0 (got {onset}, {duration})")
        out.append(
            Event(
                row=i,
                type=ty,
                region=region,
                counterpart=cp,
                target_kind=kind,
                target=target,
                commodity=com,
                onset=onset,
                duration=duration,
                severity=fl["severity"],
                rate=fl["rate"],
                T0=fl["T0"],
                tau_rho=fl["tau_rho"],
            )
        )
    return tuple(out)


# ----- timing (1) and overlap (37), vectorised over the weeks t = 1..T ----------------------------------------------
def _weeks(T: int) -> np.ndarray:
    return np.arange(1, T + 1, dtype=np.float64)


def _share(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """f^t_q = |[t-1, t) ∩ [start, end)| of (37), the vector form of ``overlap``."""
    return np.maximum(0.0, np.minimum(t, end) - np.maximum(t - 1.0, start))


def _active(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Instant t-1 inside [start, end): the instantaneous value observed in week t (1)."""
    s = t - 1.0
    return (start <= s) & (s < end)


def _in_force(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Binary mark in force in week t iff w(start) <= t < w(end), w(s) = ceil(s) + 1 (1); ``end`` may be inf."""
    return (t >= np.ceil(start) + 1.0) & (t < np.ceil(end) + 1.0)


def _factors(t: np.ndarray, segments: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Surviving share of one event: week average 1 - Σ σ f^t (37) and instantaneous 1 - σ(t-1).

    ``segments`` are (start, end, σ) pieces of the window D_q; one piece is the plain (37) factor 1 - σ f^t_q, two are
    the war profile (40) with its σ^t_q integrated over the week.
    """
    loss, loss_now = np.zeros_like(t), np.zeros_like(t)
    for start, end, sig in segments:
        loss = loss + sig * _share(t, start, end)
        loss_now = loss_now + sig * _active(t, start, end)
    return 1.0 - loss, 1.0 - loss_now


def _severity(q: Event, value: float | None = None) -> float:
    sig = q.severity if value is None else value
    if not 0.0 <= sig <= 1.0:  # NaN fails too
        raise ValueError(f"{q.name}: capacity severity must lie in [0, 1], got {sig}")
    return sig


# ----- geometry --------------------------------------------------------------------------------------------------
class _Geo:
    """Region look-ups of one instance (node regions, chokepoints included)."""

    def __init__(self, inst: Instance) -> None:
        self.inst = inst
        self.region = tuple(n.region for n in inst.nodes)
        self.chokepoints = frozenset(inst.chokepoints)
        self.fabs = frozenset(inst.fabs)

    def between(self, a: int, b: int | None, exclude: frozenset[int] = frozenset()) -> list[int]:
        """Non-coupling edges with one end in region ``a``, the other in ``b`` (None: any region not a or excluded)."""
        out = []
        for e, edge in enumerate(self.inst.edges):
            if edge.coupling:
                continue
            ends = (self.region[edge.tail], self.region[edge.head])
            for x, y in (ends, ends[::-1]):
                if x == a and y != a and (y == b if b is not None else y not in exclude):
                    out.append(e)
                    break
        return out

    def chokepoint_ordinal(self, q: Event) -> int:
        if q.target_kind != K_CHOKEPOINT or q.target not in self.chokepoints:
            raise ValueError(f"{q.name}: the type needs a chokepoint target")
        return self.inst.chokepoint_ordinal[q.target]

    def region_of(self, q: Event) -> int:
        if q.region < 0:
            raise ValueError(f"{q.name}: the operation needs the event region m_q")
        return q.region

    def fab_targets(self, q: Event) -> tuple[int, ...]:
        """Ordinals of the fabs event q acts on (E_f of (13), (14)): a regional conflict only, on the fabs of m_q."""
        if q.type != REGIONAL_CONFLICT:
            return ()
        m = self.region_of(q)
        return tuple(i for i, f in enumerate(self.inst.fabs) if self.region[f] == m)

    def osat_targets(self, q: Event) -> tuple[int, ...]:
        """Ordinals of the OSATs event q acts on through (13): a regional conflict only, on the OSATs of m_q (§4.4)."""
        if q.type != REGIONAL_CONFLICT:
            return ()
        m = self.region_of(q)
        return tuple(i for i, o in enumerate(self.inst.osats) if self.region[o] == m)


# ----- week averages, binary cost and prohibition marks (37), (40), §4.4 type table ---------------------------------
def _graph_marks(inst: Instance, events: tuple[Event, ...], params: MarkParams, t: np.ndarray) -> dict[str, np.ndarray]:
    """Capacities, open fractions, availability and energy (week averages and instants), freight, tariffs, Z_t."""
    geo = _Geo(inst)
    T, E, K = inst.T, len(inst.edges), len(inst.commodities)
    rep = lambda a: np.broadcast_to(a, (T,) + a.shape).copy()  # noqa: E731
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    sup = np.array([s.supply for s in inst.stock_slots])
    Gb = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
    out = dict(
        u=rep(u0),
        u_now=rep(u0),
        c=rep(np.array([e.c0 for e in inst.edges])),
        o=np.ones((T, len(inst.chokepoints))),
        o_now=np.ones((T, len(inst.chokepoints))),
        supply=rep(sup),
        supply_now=rep(sup),
        G_bar=rep(Gb),
        G_bar_now=rep(Gb),
        prohibited=np.zeros((T, E, K), dtype=bool),
        tariff=np.zeros((T, E, K)),
    )
    for e, k in inst.prohibitions_at_reset:  # Z_0, in force all episode (3)
        out["prohibited"][:, e, k] = True

    def cut(key: str, idx: list[int], segments: list[tuple[float, float, float]]) -> None:
        """Multiply the surviving share of one event into ``key`` and ``key_now`` at ``idx`` (overlap rule (37))."""
        fac, fac_now = _factors(t, segments)
        for i in idx:
            out[key][:, i] *= fac
            out[f"{key}_now"][:, i] *= fac_now

    for q in events:
        if q.type in (MILITARISED_CLOSURE, WEATHER_CLOSURE):  # o^t_c by (37)
            cut("o", [geo.chokepoint_ordinal(q)], [(q.onset, q.end, _severity(q))])
        elif q.type == PORT_STRIKE:
            cut(
                "u",
                _strike_edges(inst, geo, q.target_kind, q.target, q.region, q.name),
                [(q.onset, q.end, _severity(q))],
            )
        elif q.type == MATERIAL_OUTAGE:  # availability of the target node's supply slots (Q79)
            if q.target_kind != K_NODE or q.target not in inst.supply_nodes:
                raise ValueError(f"{q.name}: a material outage targets a supply node")
            idx = [
                s
                for s, slot in enumerate(inst.stock_slots)
                if slot.node == q.target and (q.commodity < 0 or slot.k == q.commodity)
            ]
            cut("supply", idx, [(q.onset, q.end, _severity(q))])
        elif q.type == ENERGY_SHOCK:  # G-bar^t_g x (1 - σ f)
            if q.target_kind != K_NODE or q.target not in inst.grids:
                raise ValueError(f"{q.name}: an energy shock targets a grid node")
            cut("G_bar", [inst.grid_ordinal[q.target]], [(q.onset, q.end, _severity(q))])
        elif q.type == REGIONAL_CONFLICT:  # war profile (40) on D_q = W0 ∪ W1, only with a drawn counterpart
            m, cp = geo.region_of(q), q.counterpart
            if cp >= 0:
                s0, s1 = params.war_profile_belligerent
                w0_end, w1_end = war_windows(q.onset, q.duration, params.war_profile_window)
                cut("u", geo.between(m, cp), [(q.onset, w0_end, s0), (w0_end, w1_end, s1)])
                cut("u", geo.between(m, None, frozenset({cp})), [(q.onset, w0_end, params.war_profile_third_party)])
        elif q.type == PIRACY:
            _piracy(inst, geo, q, t, out)
        elif q.type == TARIFF:
            _tariff(inst, geo, q, t, out)
        elif q.type == SANCTION:
            _sanction(inst, geo, q, params, t, out, cut)
    return out


def war_windows(onset: float, duration: float, width: float) -> tuple[float, float]:
    """The ends of the war profile's windows (40): W0 = [s, s + max(T_q, width)) and W1 = [end of W0, end + width)."""
    w0_end = onset + max(duration, width)
    return w0_end, w0_end + width


def _strike_edges(inst: Instance, geo: _Geo, target_kind: int, target: int, region: int, name: str) -> list[int]:
    """Port strike: the sea edges into and out of the struck ports (§4.4), a region's non-chokepoint nodes or a node.

    ``region`` is m_q (a region target's own region); ``name`` names the event in an error message.
    """
    if target_kind == K_REGION:
        if region < 0:
            raise ValueError(f"{name}: the operation needs the event region m_q")
        ports = {i for i in range(len(inst.nodes)) if i not in geo.chokepoints and geo.region[i] == region}
    elif target_kind == K_NODE and target not in geo.chokepoints:
        ports = {target}
    else:
        raise ValueError(f"{name}: a port strike targets a region or a non-chokepoint node")
    return [e for e, ed in enumerate(inst.edges) if ed.mode == "sea" and (ed.tail in ports or ed.head in ports)]


def _piracy(inst: Instance, geo: _Geo, q: Event, t: np.ndarray, out: dict) -> None:
    """Piracy: c^t_e x (1 + rate) on edges through the targeted chokepoints, a binary cost mark (1); capacity kept."""
    if q.target_kind == K_EDGE:
        idx = [q.target]
    else:
        if q.target_kind == K_REGION:
            m = geo.region_of(q)
            chks = [c for c in inst.chokepoints if m in inst.chokepoint_adjacency.get(c, ())]
        elif q.target_kind == K_CHOKEPOINT and q.target in geo.chokepoints:
            chks = [q.target]
        else:
            raise ValueError(f"{q.name}: piracy targets a region, a chokepoint or an edge")
        idx = sorted({e for c in chks for e in inst.out_edges[c] + inst.in_edges[c]})
    surcharge = q.severity if math.isnan(q.rate) else q.rate  # 1 + sigma_q when the rate is NaN (§12)
    if math.isnan(surcharge):
        raise ValueError(f"{q.name}: piracy needs a rate or a severity")
    if surcharge < 0.0:  # c x (1 + rate) is a surcharge (§4.4): a negative one could turn freight negative
        raise ValueError(f"{q.name}: the piracy surcharge (rate, else severity) must be >= 0, got {surcharge}")
    mask = _in_force(t, q.onset, q.end)
    for e in idx:
        if not inst.edges[e].coupling:
            out["c"][mask, e] *= 1.0 + surcharge  # surcharges multiply (Q50)


def _tariff(inst: Instance, geo: _Geo, q: Event, t: np.ndarray, out: dict) -> None:
    """Tariff: ad valorem rate on edges from the counterpart (every other region if -1) into m_q; rates add (Q50)."""
    if not q.rate >= 0.0:  # NaN fails too: a tariff adds tau v_k per unit to (23) (§4.4), never a subsidy
        raise ValueError(f"{q.name}: a tariff needs a finite rate >= 0, got {q.rate}")
    if q.target_kind == K_EDGE:
        idx = [q.target]
    elif q.target_kind == K_REGION:
        m, cp = q.region, q.counterpart
        idx = [
            e
            for e, ed in enumerate(inst.edges)
            if geo.region[ed.head] == m and geo.region[ed.tail] != m and (cp < 0 or geo.region[ed.tail] == cp)
        ]
    else:
        raise ValueError(f"{q.name}: a tariff targets a region or an edge")
    mask = _in_force(t, q.onset, q.end)
    for e in idx:
        for k in inst.edges[e].K:
            if q.commodity < 0 or k == q.commodity:
                out["tariff"][mask, e, k] += q.rate


def _sanction(inst: Instance, geo: _Geo, q: Event, params: MarkParams, t: np.ndarray, out: dict, cut) -> None:
    """Sanction or export control: (e, k) in Z_t (binary) and friendly fire on the dyad's other commodities (37)."""
    cp = q.counterpart
    if q.target_kind == K_EDGE:
        edge = inst.edges[q.target]
        idx = [q.target]
        dyad = (geo.region[edge.tail], geo.region[edge.head])
        sanctioned = {q.commodity} if q.commodity >= 0 else set(edge.K)
    elif q.target_kind == K_NODE:  # the node's out-edges into the counterpart, or into every other region if -1
        own = geo.region[q.target]
        idx = [
            e
            for e in inst.out_edges[q.target]
            if (geo.region[inst.edges[e].head] == cp if cp >= 0 else geo.region[inst.edges[e].head] != own)
        ]
        dyad = (own, cp if cp >= 0 else None)
        sanctioned = {q.commodity} if q.commodity >= 0 else None
    elif q.target_kind == K_REGION:
        m = q.region
        idx = [
            e
            for e, ed in enumerate(inst.edges)
            if geo.region[ed.tail] == m and (geo.region[ed.head] == cp if cp >= 0 else geo.region[ed.head] != m)
        ]
        dyad = (m, cp if cp >= 0 else None)
        sanctioned = {q.commodity} if q.commodity >= 0 else None
    else:
        raise ValueError(f"{q.name}: a sanction targets an edge, a node or a region")
    mask = _in_force(t, q.onset, q.end)
    for e in idx:
        for k in inst.edges[e].K:
            if sanctioned is None or k in sanctioned:
                out["prohibited"][mask, e, k] = True  # prohibitions unite (Q50)
    a, b = dyad
    if sanctioned is None or a == b:  # every commodity sanctioned: no edge of the dyad escapes
        return
    ff = [e for e in geo.between(a, b) if inst.edges[e].K and not sanctioned.intersection(inst.edges[e].K)]
    cut("u", ff, [(q.onset, q.end, _severity(q, params.friendly_fire))])


# ----- war-risk class (§3.4 table), queue holding and transit cost (11) ---------------------------------------------
def _war_risk_class(inst: Instance, events: tuple[Event, ...], params: MarkParams, t: np.ndarray) -> np.ndarray:
    """wr^t_c (T, C) int8: 2 Hormuz-2026, 1 Red Sea, 0 none; binary marks in force from w(s) (1).

    The class follows the active events of omega (Q43; §3.4 table): a militarised closure on c (Hormuz-2026 at or above
    the threshold, else Red Sea) or a war-state regional conflict event in a region adjacent to c. The conflict layer
    alone sets no class, so the event-free twin omega^0 has class none throughout (57).
    """
    geo = _Geo(inst)
    T, C = inst.T, len(inst.chokepoints)
    wr = np.zeros((T, C), dtype=np.int8)
    adjacent = [inst.chokepoint_adjacency.get(c, ()) for c in inst.chokepoints]
    for q in events:
        if q.type == MILITARISED_CLOSURE:
            ci = geo.chokepoint_ordinal(q)
            cls = 2 if _severity(q) >= params.hormuz_threshold else 1
            mask = _in_force(t, q.onset, q.end)
            wr[mask, ci] = np.maximum(wr[mask, ci], cls)
        elif q.type == REGIONAL_CONFLICT:
            m = geo.region_of(q)
            mask = _in_force(t, q.onset, q.end)
            for ci in range(C):
                if m in adjacent[ci]:
                    wr[mask, ci] = 2
    return wr


def _queue_and_transit(inst: Instance, wr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h^Q,t_ck by the week's class (§3.4) and c^wr,t_ek (11) on lane-continuing out-edges; c-bar^wr(none) = 0."""
    T, C, E, K = inst.T, len(inst.chokepoints), len(inst.edges), len(inst.commodities)
    hq, cwr = np.zeros((T, C, K)), np.zeros((T, E, K))
    for ci, c in enumerate(inst.chokepoints):
        attrs = inst.nodes[c].chokepoint
        cls = wr[:, ci].astype(np.intp)
        for k, row in attrs.queue_holding.items():
            hq[:, ci, k] = np.asarray(row, dtype=np.float64)[cls]
        for e in inst.out_edges[c]:
            if not inst.continues_lane(e):  # turn-backs carry no transit premium (§3.4)
                continue
            for k in inst.edges[e].K:
                if k in attrs.war_risk_cost:
                    cwr[:, e, k] = np.where(cls > 0, np.asarray(attrs.war_risk_cost[k], dtype=np.float64)[cls], 0.0)
    return hq, cwr


# public name for the M4 LP baselines' forecast and scenario windows (persisted war-risk class -> h^Q and c^wr by the
# marks' own rule, one home; design §12 M4 rows), an alias so ``compute_marks`` and every marks digest are unchanged
queue_and_transit = _queue_and_transit


# ----- fab restoration, ceiling and scrap (13), (14) ----------------------------------------------------------------
def _over_tau(x: np.ndarray, tau: float) -> np.ndarray:
    """The ratio x / τ_ρ of (13) for durations x >= 0; a subnormal τ_ρ overflows to +inf, the exact limit.

    g_ρ(+inf) = 0 for both shapes, so the overflow is the value, not an error of the fixed state ``FP_ERRORS``.
    """
    with np.errstate(over="ignore"):
        return x / tau


def _theta_bar(t: np.ndarray, on: float, t_end: float, tau: float | None, shape: str) -> np.ndarray:
    """Week average of the instantaneous loss θ_q(s) of (13), integrated analytically, in [0, 1].

    For g = exp(-x) the tail τ(e^{-x_a} - e^{-x_b}) is evaluated as τ e^{-x_a} (-expm1(-(x_b - x_a))), the same closed
    form without cancellation, both for a huge τ_ρ (where e^{-x_a} and e^{-x_b} both round to 1) and far into the tail
    (where e^{-x_a} is tiny); x_b - x_a is the week's length after t_end over τ. θ_q(s) lies in [0, 1] (an indicator,
    or g_ρ of an argument >= 0), so its week average does; the sum of the dead-time share and the tail rounds one or
    two ulps above 1 in the week containing t_end when τ_ρ is huge (about 1e15 weeks and more, both shapes), which
    would make 1 - σ_q θ-bar negative at σ_q = 1, so the sum is clipped to [0, 1] (ORACLE-M2-1): a bit-for-bit no-op
    wherever the sum already lies there.
    """
    theta = _share(t, on, t_end)  # dead time [on, t_end)
    if tau is None:
        return theta
    xa = _over_tau(np.maximum(t - 1.0, t_end) - t_end, tau)
    xb = _over_tau(np.maximum(t - t_end, 0.0), tau)
    if shape == "exp":  # ∫ exp(-(s - t_end)/τ) ds over the week's part after t_end
        after = np.maximum(t - t_end, 0.0) - (np.maximum(t - 1.0, t_end) - t_end)  # |[t-1, t) ∩ [t_end, ∞)|
        tail = tau * np.exp(-xa) * -np.expm1(-_over_tau(after, tau))
    else:  # ∫ (1 - (s - t_end)/τ)^+ ds
        ca, cb = np.minimum(xa, 1.0), np.minimum(xb, 1.0)
        tail = tau * ((cb - 0.5 * cb * cb) - (ca - 0.5 * ca * ca))
    return np.clip(theta + tail, 0.0, 1.0)  # θ-bar in [0, 1] of (13), whatever the rounding of the sum


def _theta_now(t: np.ndarray, on: float, t_end: float, tau: float | None, shape: str) -> np.ndarray:
    """θ_q(t-1), the instantaneous loss at the instant t-1 (1), (13)."""
    theta = _active(t, on, t_end).astype(np.float64)
    if tau is None:
        return theta
    s = t - 1.0
    x = _over_tau(np.maximum(s - t_end, 0.0), tau)
    g = np.exp(-x) if shape == "exp" else np.maximum(1.0 - x, 0.0)
    return np.where(s >= t_end, g, theta)


def _restoration_window(q: Event) -> tuple[float, float, float | None]:
    """(sigma_q, t^end_q, tau_rho or None) of a restoration event of (13), for fabs and OSATs alike (§4.4).

    The dead time T0 is the event's T_q, T_q itself when ``ev_T0`` is NaN (Q58, §12); a NaN or non-positive tau_rho
    means no tail, g_rho = 0 (§12); an infinite one never reaches here (``read_events``).

    Raises:
        ValueError: if the severity lies outside [0, 1] or the dead time is NaN or negative.

    """
    sig = _severity(q)
    dead = q.duration if math.isnan(q.T0) else q.T0
    if math.isnan(dead) or dead < 0:
        raise ValueError(f"{q.name}: dead time must be >= 0, got {dead}")
    t_end = q.onset + dead
    tau = q.tau_rho if q.tau_rho > 0 and math.isfinite(t_end) else None
    return sig, t_end, tau


def _restoration(inst: Instance, events: tuple[Event, ...], params: MarkParams, t: np.ndarray, full: bool = True):
    """(13)-(14) of the restoration events, one pass in event row order (§4.4: a regional conflict hits m_q).

    Returns R_f(t), R_f(t-1) instantaneous, alpha-bar_f(t) (13) and sigma^scr_f(t) (14) as (T, F) arrays, the fab hits,
    R_osat(t) (T, O) and R_osat(t-1) instantaneous (T, O). Each event removes sigma_q times its week-averaged loss
    theta-bar^t_q of (13) from the R_f of its fabs and the R_osat of its OSATs (its instantaneous loss theta_q(t-1) from
    R_f(t-1) and R_osat(t-1)), and several multiply in event row order, so equal parameters give equal factors for fabs
    and OSATs bit for bit. The design gives OSATs no ceiling alpha-bar and no scrap (14); their observation field is
    ``graph_now.osat`` = {node, R^osat(t-1), thr R^osat(t-1)} (Q97). An event that hits OSATs only and is refused by
    ``_restoration_window`` raises after every fab-hitting event has been read (the fab factors were once formed in a
    pass of their own, first). With ``full`` False only the restoration factors are formed (``restoration_factors``,
    ``event_capacity_marks``): R_f(t) and R_osat(t) are the same bits, the same events are refused, and R_f(t-1),
    alpha-bar, sigma^scr and R_osat(t-1) are None and the hits empty.
    """
    if params.restoration_shape not in _RESTORATION_SHAPES:
        raise ValueError(f"restoration_shape must be one of {_RESTORATION_SHAPES}, got {params.restoration_shape!r}")
    geo = _Geo(inst)
    T, F = inst.T, len(inst.fabs)
    R, R_now, keep = np.ones((T, F)), np.ones((T, F)), np.ones((T, F))
    R_osat, R_osat_now = np.ones((T, len(inst.osats))), np.ones((T, len(inst.osats)))
    hits: list[FabHit] = []
    refused: ValueError | None = None  # the first refusal of an event that hits OSATs only
    for q in events:
        fabs, osats = geo.fab_targets(q), geo.osat_targets(q)
        if not (fabs or osats):
            continue
        try:
            sig, t_end, tau = _restoration_window(q)
        except ValueError as err:
            if fabs:
                raise
            refused = refused or err
            continue
        theta = _theta_bar(t, q.onset, t_end, tau, params.restoration_shape)
        for oi in osats:
            R_osat[:, oi] *= 1.0 - sig * theta
        for fi in fabs:
            R[:, fi] *= 1.0 - sig * theta
        if not full:
            continue
        theta_now = _theta_now(t, q.onset, t_end, tau, params.restoration_shape)
        for oi in osats:  # R^osat at the instant t - 1, like R_now (Q97; design §12 "OSAT observation")
            R_osat_now[:, oi] *= 1.0 - sig * theta_now
        if not fabs:
            continue
        f_dead = _share(t, q.onset, t_end)
        for fi in fabs:
            R_now[:, fi] *= 1.0 - sig * theta_now
            keep[:, fi] *= 1.0 - f_dead  # a^t_f = 1 - Π(1 - f^t_q) over the windows [on, on + T0)
            # (14) scraps only for hits inside the episode: carried-in events (onset <= 0) scrap nothing (§2.3), and
            # nothing is booked after T, so the WIP credit of (23) is net of scrap booked by T in simulator and LP alike
            if q.onset > 0 and math.floor(q.onset) + 1 <= T:
                hits.append(FabHit(fab=fi, onset=q.onset, severity=sig))
    if refused is not None:
        raise refused
    if not full:
        return R, None, None, None, (), R_osat, None
    decay = math.exp(-1.0 / inst.params.tau_alpha)
    gain = (inst.params.alpha_max - 1.0) * (1.0 - decay)
    active = 1.0 - keep
    alpha = np.ones((T, F))
    for fi in range(F):
        prev = 1.0  # alpha-bar_f(0) = 1
        for ti in range(T):
            prev = 1.0 + (prev - 1.0) * decay + gain * active[ti, fi]
            alpha[ti, fi] = prev
    w_scr = [inst.nodes[f].fab.w_scr for f in inst.fabs]
    scr = np.zeros((T, F))
    for fi in range(F):
        for ti in range(T):  # start week ti + 1 (14)
            scr[ti, fi] = _scrap_share(hits, w_scr, fi, ti + 1)
    return R, R_now, alpha, scr, tuple(hits), R_osat, R_osat_now


# ----- public API for the disruption stage: the rule table read event by event, never re-coded -----------------------
@fixed_fp_errors
def graph_marks(inst: Instance, events: tuple[Event, ...], params: MarkParams) -> dict[str, np.ndarray]:
    """The graph marks of ``events`` alone over the weeks t = 1..T, exactly as ``compute_marks`` forms them.

    Week averages by (37) and (40) (``u``, ``o``, ``supply``, ``G_bar`` and their instants ``*_now``), the binary marks
    of (1) (``prohibited`` (T, E, K) with Z_0 of (3), ``tariff`` (T, E, K), freight ``c`` (T, E) with piracy), for the
    events given (an empty tuple: the event-free marks). Runs under the fixed error state ``FP_ERRORS``.

    Raises:
        ValueError: on an event the §4.4 type table refuses (a target kind its type does not define, a severity
            outside [0, 1], a negative tariff rate or piracy surcharge).

    """
    return _graph_marks(inst, tuple(events), params, _weeks(inst.T))


@fixed_fp_errors
def restoration_factors(inst: Instance, events: tuple[Event, ...], params: MarkParams) -> tuple[np.ndarray, np.ndarray]:
    """(R_f (T, F), R_osat (T, O)): the restoration factors (13) of ``events`` alone over the weeks t = 1..T.

    The factors ``compute_marks`` forms for fabs and OSATs (§4.4: a regional conflict acts on the fabs and OSATs of
    m_q), in event order. Runs under the fixed error state ``FP_ERRORS``.

    Raises:
        ValueError: on a restoration event with a severity outside [0, 1] or a negative dead time.

    """
    R_f, _, _, _, _, R_osat, _ = _restoration(inst, tuple(events), params, _weeks(inst.T), full=False)
    return R_f, R_osat


@fixed_fp_errors
def osat_restoration_now(inst: Instance, events: tuple[Event, ...], params: MarkParams) -> np.ndarray:
    """(T, O) read-only R^osat_i at the instant t - 1, t = 1..T: (13)'s OSAT factor as ``R_now`` is the fab's (Q97).

    The observation's ``graph_now.osat.R`` (design §12 "OSAT observation"): each regional conflict removes sigma_q
    theta_q(t - 1) from the OSATs of m_q, several multiplying in event row order, formed in the pass that forms
    ``compute_marks``' R_f, R_now and R_osat (``_restoration``). Not a ``WeeklyMarks`` field, so no marks digest moves
    (the simulator and the LP read the week average ``R_osat`` only); ``information.view`` forms it from omega's events
    once per episode. Runs under the fixed error state ``FP_ERRORS``.

    Raises:
        ValueError: on a restoration event with a severity outside [0, 1] or a negative dead time.

    """
    *_, R_osat_now = _restoration(inst, tuple(events), params, _weeks(inst.T))
    return immutable_copy(R_osat_now)


@fixed_fp_errors
def event_capacity_marks(inst: Instance, q: Event, params: MarkParams) -> dict[str, np.ndarray]:
    """The capacity marks of event q alone over the weeks t = 1..T, exactly as ``compute_marks`` forms them.

    The week averages ``o``, ``u``, ``supply`` and ``G_bar`` of (37) and (40) (``graph_marks``) and, when q hits a
    fab or an OSAT, ``R_f`` and ``R_osat`` of (13) (``restoration_factors``; absent otherwise): the marks that (37)
    multiplies, which harm (41) values and the realism overlap count reads. Runs under the fixed error state
    ``FP_ERRORS``.

    Raises:
        ValueError: on an event the §4.4 type table refuses (``graph_marks``, ``restoration_factors``).

    """
    t = _weeks(inst.T)
    g = _graph_marks(inst, (q,), params, t)
    out = {name: g[name] for name in ("o", "u", "supply", "G_bar")}
    if fab_targets(inst, q) or osat_targets(inst, q):
        out["R_f"], _, _, _, _, out["R_osat"], _ = _restoration(inst, (q,), params, t, full=False)
    return out


def restoration_regions(inst: Instance) -> frozenset[int]:
    """The regions hosting a fab or an OSAT: a regional conflict there gets the restoration (13) of §4.4."""
    return frozenset(inst.nodes[n].region for n in inst.fabs + inst.osats)


def fab_targets(inst: Instance, q: Event) -> tuple[int, ...]:
    """Ordinals of the fabs event q acts on through (13) and (14): a regional conflict only, the fabs of m_q (§4.4)."""
    return _Geo(inst).fab_targets(q)


def osat_targets(inst: Instance, q: Event) -> tuple[int, ...]:
    """Ordinals of the OSATs event q acts on through (13): a regional conflict only, the OSATs of m_q (§4.4)."""
    return _Geo(inst).osat_targets(q)


def strike_edges(inst: Instance, target_kind: int, target: int) -> list[int]:
    """The sea edges a port strike on ``target`` stops (§4.4): a region's ports (``K_REGION``) or one port node.

    The rule of ``compute_marks`` (sea edges with an end at a non-chokepoint node of the struck region, or at the
    struck node), so a caller can tell where a strike has a graph operation without re-coding it.

    Raises:
        ValueError: on a target kind other than region or a non-chokepoint node.

    """
    region = target if target_kind == K_REGION else -1
    return _strike_edges(inst, _Geo(inst), target_kind, target, region, _event_name(0, PORT_STRIKE, target_kind))


def _stored(arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, int], dtype) -> np.ndarray | None:
    """Omega's stored (rows, T) array ``name`` as a (T, rows) week array, or None if omega does not store it."""
    if name not in arrays:
        return None
    a = np.asarray(arrays[name])
    if a.shape != shape:
        raise ValueError(f"omega array {name!r} has shape {a.shape}, expected {shape}")
    return np.ascontiguousarray(a.T, dtype=dtype)


# the stored marks that are restoration factors of (13): each a product of factors 1 - sigma_q theta-bar in [0, 1]
_UNIT_MARKS = frozenset({"R_f", "R_osat"})


def _realised(arrays: Mapping[str, np.ndarray], name: str, recomputed: np.ndarray, exact: bool) -> np.ndarray:
    """The week array the marks carry for omega's ``name``: the stored one, checked against ``recomputed``.

    A stored array is the realisation (owner decision Q91, option (a)). ``R_f``, ``R_osat`` and ``alpha_bar`` go
    through libm's exp/expm1 in (13), so each entry must be finite and within ``STORED_MARK_TOL`` max(1,
    |recomputed|) of the recomputation, the row tolerance of (53), and a last-bit difference of another platform's libm
    is accepted. ``R_f`` and ``R_osat`` (``_UNIT_MARKS``) must also lie in [0, 1], the range of (13)'s factors: a
    stored -1e-10 where the recomputation is 0 lies within the tolerance, yet made the simulator package negative raw
    chips and the oracle LP infeasible (ORACLE-M2-1).
    ``sigma_scr`` (products of (14), no libm) and the integer ``wr_class`` are ``exact``: equal on every IEEE platform,
    and the LP reads ``sigma_scr`` while the simulator books scrap from the hits, so a tolerance there could part them.
    Without a stored array the recomputation is the realisation.

    Raises:
        ValueError: if the stored array has the wrong shape, a non-finite entry, an ``R_f`` or ``R_osat`` entry outside
            [0, 1], or an entry off its recomputation by more than the tolerance (``wr_class``: any difference), as
            arrays stored under other MarkParams are (V3).

    """
    stored = _stored(arrays, name, recomputed.shape[::-1], recomputed.dtype)
    if stored is None:
        return recomputed
    if name in _UNIT_MARKS and not bool(np.all((stored >= 0.0) & (stored <= 1.0))):  # NaN fails too
        raise ValueError(f"omega's stored {name!r} has an entry outside [0, 1], the range of (13)'s factors")
    if not exact:
        ok = bool(np.isfinite(stored).all()) and bool(
            np.all(np.abs(stored - recomputed) <= STORED_MARK_TOL * np.maximum(1.0, np.abs(recomputed)))
        )
    else:
        ok = np.array_equal(stored, recomputed)
    if not ok:
        raise ValueError(f"omega's stored {name!r} differs from its recomputation under these MarkParams (V3)")
    return stored


@fixed_fp_errors
def compute_marks(inst: Instance, omega: Omega, params: MarkParams | None = None) -> WeeklyMarks:
    """Weekly marks of one episode from its omega container (§4.1) and the instance (V3).

    Implements (1), (11), (13)-(14), (37), (40) and the graph operations of the §4.4 type table for every event in
    ``omega`` (carried-in ones included), plus the instance's ``prohibitions_at_reset``. Demand comes from
    ``omega["d"]``. omega must state the instance it was built on (``meta_instance_hash`` and the content digest
    ``meta_instance_digest``) and its mark parameters (``meta_mark_params``, bound to its hash), as every builder does;
    ``params``, if given, must equal omega's. omega's stored ``R_f``, ``R_osat`` and ``alpha_bar``, when present, are
    the realisation the marks carry, each entry within ``STORED_MARK_TOL`` max(1, |recomputed|) of its recomputation
    (the (53) row tolerance; owner decision Q91, option (a)); a stored ``sigma_scr`` or ``wr_class`` must equal its
    recomputation. The marks record omega's ``meta_generator_id`` and ``Omega.generated`` (outside their digest), for
    the D9 fallback guard of a reset from the marks alone (§8.1). ``inst`` may be the file's instance at another rung
    than omega's: the marks are then the instance's at omega's rung (``Instance.at_digest``: the block of the warm
    start per rung omega was drawn on, §2.3; M5-O37 (b)). Runs under the fixed error state ``FP_ERRORS``.

    Raises:
        ValueError: if ``omega`` lacks ``meta_instance_hash``, ``meta_instance_digest`` or ``meta_mark_params``, was
            built on another instance hash or content, ``params`` differs from omega's, or a stored mark array has the
            wrong shape, a non-finite entry, an ``R_f`` or ``R_osat`` entry outside [0, 1], or differs from its
            recomputation beyond the tolerance (V3).

    """
    arrays = omega.arrays
    for name in ("meta_instance_hash", "meta_instance_digest", "meta_mark_params"):
        if name not in arrays:  # never defaulted: one omega hash must give one realisation for one instance (V3)
            raise ValueError(f"omega has no {name!r}; every omega builder sets it (V3)")
    if str(arrays["meta_instance_hash"]) != inst.hash:
        raise ValueError("omega's meta_instance_hash is not the instance hash (V3)")
    inst = inst.at_digest(str(arrays["meta_instance_digest"]))  # omega's rung: its warm start block (§2.3; M5-O37)
    if str(arrays["meta_instance_digest"]) != inst.content_digest:
        raise ValueError("omega was built on other instance content than this instance (V3, content digest)")
    own = mark_params_from_json(str(arrays["meta_mark_params"]))
    if params is not None and params != own:
        raise ValueError("MarkParams differ from the ones omega was built under (meta_mark_params, V3)")
    params = own
    T, F, C = inst.T, len(inst.fabs), len(inst.chokepoints)
    t = _weeks(T)
    events = read_events(inst, arrays)
    g = _graph_marks(inst, events, params, t)
    R, R_now, alpha, scr, hits, R_osat, _ = _restoration(inst, events, params, t)
    wr = _war_risk_class(inst, events, params, t)
    R, R_osat, alpha, scr, wr = (
        _realised(arrays, name, week_array, exact)
        for name, week_array, exact in (
            ("R_f", R, False),
            ("R_osat", R_osat, False),  # through libm's exp/expm1 in (13), like R_f
            ("alpha_bar", alpha, False),
            ("sigma_scr", scr, True),
            ("wr_class", wr, True),
        )
    )
    hq, cwr = _queue_and_transit(inst, wr)
    k_mu = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints]).reshape(C, 2)
    if "d" not in arrays or np.shape(arrays["d"]) != (len(inst.demands), T):
        raise ValueError(f"omega needs demand 'd' of shape {(len(inst.demands), T)}")
    yb = np.array([inst.nodes[gi].grid.base_load for gi in inst.grids])
    y_bar = np.broadcast_to(yb, (T,) + yb.shape).copy()  # no v1 event type changes base load
    alpha_now = np.ones((T, F))
    alpha_now[1:] = alpha[:-1]  # alpha-bar_f(t-1), alpha-bar_f(0) = 1
    return WeeklyMarks(
        T=T,
        instance_hash=inst.hash,
        instance_digest=inst.content_digest,
        omega_hash=omega.hash,
        u=g["u"],
        c=g["c"],
        o=g["o"],
        kappa=k_mu[None, :, :] * g["o"][:, :, None],  # (k_c mu_cb) o^t_c (9)
        supply=g["supply"],
        G_bar=g["G_bar"],
        y_bar=y_bar,
        R=R,
        alpha_bar=alpha,
        sigma_scr=scr,
        R_osat=R_osat,
        demand=np.ascontiguousarray(np.asarray(arrays["d"], dtype=np.float64).T),
        prohibited=g["prohibited"],
        tariff=g["tariff"],
        wr_class=wr,
        h_queue=hq,
        c_wr=cwr,
        u_now=g["u_now"],
        o_now=g["o_now"],
        kappa_now=k_mu[None, :, :] * g["o_now"][:, :, None],  # (k_c mu_cb) o_c(t-1) (9)
        supply_now=g["supply_now"],
        G_bar_now=g["G_bar_now"],
        y_bar_now=y_bar.copy(),
        R_now=R_now,
        alpha_now=alpha_now,
        fab_hits=hits,
        w_scr=tuple(inst.nodes[f].fab.w_scr for f in inst.fabs),
        generator_id=str(arrays["meta_generator_id"]) if "meta_generator_id" in arrays else "",
        generated=omega.generated,
    )


@fixed_fp_errors
def stored_mark_arrays(
    inst: Instance, arrays: Mapping[str, np.ndarray], params: MarkParams = MarkParams()
) -> dict[str, np.ndarray]:
    """The per-episode arrays omega stores beside its events (§4.1 table).

    ``R_f``, ``alpha_bar``, ``sigma_scr`` (F, T) by (13)-(14), ``R_osat`` (O, T) by (13) on the OSATs of regional
    conflicts (§4.4; milestone M2), and ``wr_class`` (C, T) int8 by the §3.4 war-risk rule, all from the ``ev_*`` arrays
    of ``arrays`` (no other array is read). Runs under the fixed error state ``FP_ERRORS``.
    """
    t = _weeks(inst.T)
    events = read_events(inst, arrays)
    R, _, alpha, scr, _, R_osat, _ = _restoration(inst, events, params, t)
    wr = _war_risk_class(inst, events, params, t)
    return {
        "R_f": np.ascontiguousarray(R.T),
        "R_osat": np.ascontiguousarray(R_osat.T),
        "alpha_bar": np.ascontiguousarray(alpha.T),
        "sigma_scr": np.ascontiguousarray(scr.T),
        "wr_class": np.ascontiguousarray(wr.T, dtype=np.int8),
    }

"""The V22 realism band and its reported items on generated omegas (design §4.4, §2.4, §2.6, §10 V22).

Sources: §4.4 "Realism band", the §2.4 `tiny` profile row, §2.6 "The realism band ... holds on every rung"; Q39, Q40,
Q52, Q68, Q94. The band, per rung (§4.4 table, SYNTHETIC(prior: the W2-maritime §2.2 record, counted in F5 §4.2)):
militarised closures per year at the anchor rung in [0.073, 0.43]; the mean number of chokepoints cut to <= 60 % at the
anchor rung in [0.15, 0.73]; closures per year at the stress rungs <= 3 x 0.85; every chokepoint's mean open fraction
>= 0.5 at every rung. On `tiny` (§2.4 profile row; Q94 (a), (d)) it is the §4.4 band scaled by 4/7, since `chk` stands
for four of the seven flagship chokepoints: closures per year at the anchor in [0.042, 0.246], mean impaired
chokepoints in [0.086, 0.417], and the open-fraction item read per constituent (Q94 (a)): o_chk^(1/4) >= 0.5, i.e. the
mean open fraction o_chk >= 0.5^4 = 0.0625 (``RealismBand.constituents``, 4 for `chk`; the flagship band keeps 1 per
chokepoint). No stress ceiling on `tiny` (Q94 (d)), so ``TINY_BAND`` has none; ``stress_ceiling`` reports the
stationary closure rates of (32) beside the ceiling scaled by 4/7 (3 x 0.85 x 4/7 = 1.457) and unscaled (2.55).

Reported, not gated (V22): harm (41)-(42) per episode (and (43) per type); the share of target-weeks where (37)
combines two or more events; tanker overrides that starve containers of throughput (§3.4, Q68).

Readings where the design leaves the rule open (fixed here, reported for design rows):

- **Closures per year**, two estimators of one stationary rate: the published one counts the militarised closures of
  omega's event list with onset in [0, T), per T / 52 years; the record one counts the militarised closures of the
  episode's whole generated process (``sampler.EventSample.marked``, burn-in included) with onset in
  [``record_start``, T), where ``record_start`` = min(-B_burn + 3 t95, 0) and t95 is (33): the regime layers and X
  start at their stationary laws (§4.3), so onsets are stationary once the Hawkes mean intensity has relaxed (its gap
  is then 0.05^3 of the stationary rate). On `tiny` at the anchor the record holds about 50 times the information of
  the episode's 26 weeks, so the band test reads it; the published estimator is reported beside it.
- **Impaired chokepoints**: per week, the number of chokepoints whose week-average open fraction o^t_c of (37)
  (``marks.compute_marks``) is <= the impairment cut 0.6 (§8.1 step 6, SYNTHETIC, §11 row 83), averaged over the weeks
  1..T. o^t_c includes weather and accident closures, which (37) combines with the militarised ones.
- **Open fraction**: per chokepoint, the mean of o^t_c over the weeks 1..T.
- **Independence**: episodes are independent draws (distinct episode indices n of (27)), so each item is a mean over
  episodes with the standard error of the per-episode values.
- **Band test** (burden of proof on failing, Q94 (b)): an item fails when a one-sided t test at level ``alpha`` (default
  0.001, the V9 and V12 level) rejects "the generator's value lies in the band" at the violated edge,
  (mean - hi) / SE > t_{1-alpha, n-1} or (lo - mean) / SE > t_{1-alpha, n-1}. A point-estimate reading would fail
  about half of all seeds on `tiny`, whose anchor calibration, 0.43 x 4/7 closures per year (§2.4), is the band's
  upper edge. Each chokepoint's open fraction is tested at alpha / C (Bonferroni over the C chokepoints). A chokepoint
  standing for k constituents is tested on the aggregate scale against floor^k: o^(1/k) >= floor and o >= floor^k are
  one event, since x -> x^(1/k) increases; ``BandCheck.per_constituent`` reports o^(1/k) beside it, with the
  delta-method SE.
- **Stated power** (Q94 (b)): each gated item at each rung carries power ``POWER`` = 0.8 against a stated alternative
  past its violated edge, and that power sets its episode count (``band_episodes``; a rung runs the largest count of
  its items). The alternatives: an open fraction ``OPEN_FRACTION_DISTANCE`` = 0.1 per constituent below the floor, i.e.
  the aggregate (floor - 0.1)^k (0.4^4 = 0.0256 on `chk` against 0.0625), at alpha / C (the owner queue's example for
  option (b) 1, 2026-09-26); closures per year ``CLOSURES_DISTANCE`` = 50 % above the upper edge (the anchor band, and
  the stress ceiling where the band has one); impaired chokepoints ``IMPAIRED_DISTANCE`` = 0.25 above the upper edge.
  The design states all three (Q94 (b); §10 "V22, band", §11 row 86): on `tiny` the alternatives are 0.3686
  closures per year and 0.6671 impaired chokepoints, the closure and impaired distances SYNTHETIC there (the M2
  build's earlier post-hoc power statements), the open-fraction distance the owner queue's example. The power is that
  of ``check_item`` itself. The per-episode SD at the alternative: for a bounded item (an open fraction in
  [0, 1], the impaired count in [0, C]) the largest SD any law of that mean can have on that range, sqrt((b - m)(m - a))
  of the two-point law on {a, b} (Bhatia-Davis); for the closure rate, unbounded, a pilot dispersion kappa = SD /
  sqrt(rate) from the caller's pilot on another entropy, times the root of the alternative rate (the SD scaled as the
  root of the rate, as for counts at fixed clustering). The power is the noncentral t's (``band_power``, exact for
  normal per-episode values); for a bounded item also the exact power under that two-point law (``two_point_power``,
  whose sample mean is far from normal near an end of the range), and the smaller of the two is stated. The count is
  the smallest n >= 2 whose stated power reaches ``POWER`` (``episodes_for_power``). The lower edges carry no stated
  power: the stated alternatives lie past the upper edges, where `tiny`'s anchor calibration sits (§2.4).
- **Target-weeks of (37)**: the capacity marks (37) multiplies (open fractions, edge capacities, supply rates,
  deliverable energy, and the fab and OSAT restoration factors of (13)); an event acts on a target in week t when its
  own mark there, from ``marks.event_capacity_marks`` of the event alone, is below the event-free value; a target-week
  counts when at least one event acts, and combines when two or more do. Cost marks and prohibitions are not (37) and
  are not counted.
- **Capacity-loss share** (V17, which compares it in week 1 against week ceil(T/2)): per week, the value of the
  capacity the week's marks remove over the value of all nominal capacity, harm's (41) valuation read week by week
  (``harm.capacity_loss_share``, re-exported here); 0 without events, 1 when nothing is left.
- **Starvation** (§3.4: "Tanker overrides that starve containers of throughput are reported as telemetry"): in every
  week with an override (an override slot that passed the validity rules, ``StepRecord.override_requested``) or a hold,
  and no D9 fallback, the container-pool units the chokepoint step releases at each chokepoint, against the units it
  releases from the same state with the week's overrides and holds dropped; the starved units are the positive part
  of the difference, summed over those weeks and chokepoints. The replay plays the episode's own D9 fallback
  (``naive_fq.fallback_spec``: naive with the generator's F_Q on a generated omega, the point mass at 0 on an injected
  list), so its fallback weeks replay identically. Naive never overrides (§8.1 step 6), so its telemetry is
  zero; with throughput per pool (9) and one pool per edge (§12 defaults) an override takes no container throughput,
  so the telemetry is zero on every such instance, `tiny` included.
"""

import functools
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from shockbench_flow import marks as _marks
from shockbench_flow.disruption import harm as _harm
from shockbench_flow.disruption.gate import P_MIN
from shockbench_flow.disruption.hawkes import branching_matrix, relaxation_time
from shockbench_flow.disruption.params import WEEKS_PER_YEAR, GeneratorParams
from shockbench_flow.disruption.profiles import (
    ANCHOR_GAMMA,
    FLAGSHIP_CLOSURES_PER_YEAR,
    FLAGSHIP_KINDS,
    TINY_CHOKEPOINT_SHARE,
    closure_rate,
    flagship_closure_rates,
)
from shockbench_flow.disruption.sampler import EventSample, sample_events, sample_omega
from shockbench_flow.disruption.strata import QUANTILES, inverse_cdf
from shockbench_flow.instance.schema import POOLS, Instance
from shockbench_flow.marks import K_CHOKEPOINT, MILITARISED_CLOSURE
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega
from shockbench_flow.parallel import ordered_map
from shockbench_flow.policies import naive_fq
from shockbench_flow.policies.naive import CLOSED_THRESHOLD


IMPAIRMENT_CUT = CLOSED_THRESHOLD  # §11 row 83: the band's cut, <= 60 %, is naive's closed-route threshold (§8.1)
STRESS_RUNGS = (0.95, 0.97)  # "stress, no source" (§4.2)
STRESS_MULTIPLIER = 3.0  # SYNTHETIC (§4.4 band)
W2_CLOSURES_PER_YEAR = 0.85  # the whole W2 list 2022-26 (§4.4 band record)
ALPHA = P_MIN  # the band test's level, the V9 and V12 level (p > 0.001 at a fixed seed)
POWER = 0.8  # Q94 (b): the stated power of every gated item at every rung (module docstring)
OPEN_FRACTION_DISTANCE = 0.1  # the open-fraction alternative, per constituent below the floor (Q94 (b))
CLOSURES_DISTANCE = 0.5  # the closure-rate alternative, relative, above the upper edge (Q94 (b))
IMPAIRED_DISTANCE = 0.25  # the impaired alternative, absolute, above the upper edge (Q94 (b))
RECORD_T95 = 3.0  # the record starts 3 t95 (33) after -B_burn
HARM_QUANTILES = QUANTILES  # the cut levels of (44), for the harm display
FLAGSHIP_CHOKEPOINTS = 7  # the flagship's chokepoints (§2.4: `chk` carries 4/7 of their closure rate)
# the flagship chokepoints `chk` stands for, its share of them (§2.4): Hormuz, Malacca, Suez/Red Sea, Cape reroute
TINY_CHK_CONSTITUENTS = round(TINY_CHOKEPOINT_SHARE * FLAGSHIP_CHOKEPOINTS)
_CONTAINER = POOLS.index("ct")


# ----- the band ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RealismBand:
    """The realism band of one instance family (§4.4; `tiny` §2.4, Q94), per rung."""

    closures_per_year: tuple[float, float]  # militarised closures per year, anchor rung
    impaired: tuple[float, float]  # mean number of chokepoints cut to <= the impairment cut, anchor rung
    min_open_fraction: float  # every constituent chokepoint's mean open fraction, every rung
    stress_closures_per_year: float | None = None  # ceiling at the stress rungs; None when the band states none
    anchor: float = ANCHOR_GAMMA
    stress_rungs: tuple[float, ...] = STRESS_RUNGS
    constituents: tuple[int, ...] = ()  # per chokepoint ordinal, the flagship chokepoints it stands for; () = 1 each

    def constituent_counts(self, n_chokepoints: int) -> tuple[int, ...]:
        """The constituent count k of each of the instance's ``n_chokepoints`` chokepoints, in ordinal order.

        Raises:
            ValueError: if the band states counts for another number of chokepoints, or a count below 1.

        """
        if not self.constituents:
            return (1,) * n_chokepoints
        if len(self.constituents) != n_chokepoints or min(self.constituents) < 1:
            raise ValueError(f"the band states constituents {self.constituents} for {n_chokepoints} chokepoints")
        return self.constituents


# §4.4 table: 0.073/yr 1945-2026.7 and 0.43/yr 2022-26.7; impaired 0.15 (1945-2026) and 0.73 (2022-26)
FLAGSHIP_BAND = RealismBand(
    closures_per_year=(0.073, FLAGSHIP_CLOSURES_PER_YEAR),
    impaired=(0.15, 0.73),
    min_open_fraction=0.5,
    stress_closures_per_year=STRESS_MULTIPLIER * W2_CLOSURES_PER_YEAR,
)
# §2.4 profile row: the §4.4 band scaled by 4/7 (displayed there as [0.042, 0.246] and [0.086, 0.417]); `chk`, the
# only chokepoint, read per constituent (Q94 (a)); no stress ceiling (Q94 (d))
TINY_BAND = RealismBand(
    closures_per_year=(0.073 * TINY_CHOKEPOINT_SHARE, FLAGSHIP_CLOSURES_PER_YEAR * TINY_CHOKEPOINT_SHARE),
    impaired=(0.15 * TINY_CHOKEPOINT_SHARE, 0.73 * TINY_CHOKEPOINT_SHARE),
    min_open_fraction=0.5,
    stress_closures_per_year=None,
    constituents=(TINY_CHK_CONSTITUENTS,),
)
BANDS = {"tiny": TINY_BAND, "small": FLAGSHIP_BAND, "full": FLAGSHIP_BAND}  # by instance kind (§2.4, §4.4; Q94)


def band_for(inst: Instance) -> RealismBand:
    """The realism band of the instance's kind: `tiny`'s scaled band (§2.4, Q94), the §4.4 band on `small` and `full`.

    The flagship band gates the stress ceiling 3 x 0.85 at 0.95 and 0.97 and reads each of the seven chokepoints as one
    constituent (§4.4 table; Q94 (a), (d) exempted `tiny` only).

    Raises:
        ValueError: on a kind without a band, or a flagship instance whose chokepoints are not the seven of §2.2.

    """
    band = BANDS.get(inst.kind)
    if band is None:
        raise ValueError(f"no realism band for instance kind {inst.kind!r} (bands: {sorted(BANDS)})")
    counts = band.constituent_counts(len(inst.chokepoints))
    if inst.kind != "tiny" and sum(counts) != FLAGSHIP_CHOKEPOINTS:
        raise ValueError(f"{inst.instance_id}: the §4.4 band reads {FLAGSHIP_CHOKEPOINTS} chokepoints, got {counts}")
    return band


def chokepoint_share(inst: Instance) -> float:
    """The share of the seven flagship chokepoints the instance's chokepoints stand for: 4/7 on `tiny`, 1 otherwise.

    The sum of ``band_for(inst)``'s constituent counts over ``FLAGSHIP_CHOKEPOINTS`` (§2.4: `chk` carries four).
    """
    return sum(band_for(inst).constituent_counts(len(inst.chokepoints))) / FLAGSHIP_CHOKEPOINTS


# ----- one episode -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeRealism:
    """The V22 measures of one episode (module docstring)."""

    weeks: int  # T
    closures: int  # militarised closures with onset in [0, T), every chokepoint (omega's event list)
    record_closures: int  # the same over the record [record_start, T) of the generated process; 0 without a record
    record_weeks: float  # T - record_start; 0.0 without a record
    impaired: float  # mean over the weeks 1..T of the number of chokepoints with o^t_c <= the cut
    open_fraction: tuple[float, ...]  # per chokepoint ordinal, the mean of o^t_c over the weeks 1..T
    harm: float  # H(omega) of (42), USD
    harm_by_type: tuple[float, ...]  # H_ty of (43) per event type code, USD
    target_weeks: int  # capacity target-weeks where at least one event acts (37)
    combined_target_weeks: int  # capacity target-weeks where two or more events act (37)


def record_start(inst: Instance, params: GeneratorParams) -> float:
    """min(-B_burn + 3 t95, 0): the first onset instant of the stationary record, t95 of (33) at the params' rung."""
    t95 = relaxation_time(branching_matrix(inst, params), params.hawkes.beta_inv)
    return min(-float(params.burn_in) + RECORD_T95 * t95, 0.0)


def record_closures(inst: Instance, params: GeneratorParams, sample: EventSample) -> tuple[int, float]:
    """(count, weeks): the militarised closures of the whole generated process with onset in [record_start, T).

    The cluster process's closures (``sample.marked``) and the derived ones (``sample.derived``: the dyad strait
    closures of `small` and `full`, which V22 counts, M5-O13 default).
    """
    lo = record_start(inst, params)
    n = sum(
        1
        for e in (*sample.marked, *sample.derived)
        if e is not None and e.type == MILITARISED_CLOSURE and e.target_kind == K_CHOKEPOINT and lo <= e.onset < inst.T
    )
    return n, float(inst.T) - lo


def _mark_params(omega: Omega) -> _marks.MarkParams:
    if "meta_mark_params" not in omega.arrays:
        raise ValueError("omega has no meta_mark_params; every omega builder sets it (V3)")
    return _marks.mark_params_from_json(str(omega.arrays["meta_mark_params"]))


def event_capacity(inst: Instance, omega: Omega) -> tuple[dict[str, np.ndarray], ...]:
    """``marks.event_capacity_marks`` of each stored event of omega (carried-in ones included) under its MarkParams.

    Formed once per episode and shared by harm (``harm.totals``) and the overlap count (``target_weeks``).

    Raises:
        ValueError: if omega states no ``meta_mark_params`` or an event is refused by the rule table (``marks``).

    """
    params = _mark_params(omega)
    return tuple(_marks.event_capacity_marks(inst, q, params) for q in _marks.read_events(inst, omega.arrays))


@_marks.fixed_fp_errors
def target_weeks(
    inst: Instance, omega: Omega, *, capacity: Sequence[Mapping[str, np.ndarray]] | None = None
) -> tuple[int, int]:
    """(target-weeks with >= 1 event acting, those with >= 2): the capacity marks (37) combines (module docstring).

    Each stored event (carried-in ones included) is read alone through the rule table (``marks.event_capacity_marks``;
    ``capacity`` holds them in event order when the caller has them, ``event_capacity``); it acts on a target in week t
    when its mark there is below the event-free one.

    Raises:
        ValueError: if omega states no ``meta_mark_params``, an event is refused by the rule table, or ``capacity`` is
            of another length than omega's event list.

    """
    params = _mark_params(omega)
    events = _marks.read_events(inst, omega.arrays)
    if capacity is None:
        capacity = tuple(_marks.event_capacity_marks(inst, q, params) for q in events)
    elif len(capacity) != len(events):
        raise ValueError(f"capacity holds the marks of {len(capacity)} events, omega stores {len(events)}")
    if not events:
        return 0, 0
    T, F, O = inst.T, len(inst.fabs), len(inst.osats)
    base = _marks.graph_marks(inst, (), params)
    count: np.ndarray | None = None
    for g in capacity:
        acts = [g[name] < base[name] for name in ("o", "u", "supply", "G_bar")]
        if "R_f" in g:  # q hits a fab or an OSAT (13)
            acts += [g["R_f"] < 1.0, g["R_osat"] < 1.0]
        else:
            acts += [np.zeros((T, F), dtype=bool), np.zeros((T, O), dtype=bool)]
        a = np.concatenate([x.reshape(T, -1) for x in acts], axis=1).astype(np.int64)
        count = a if count is None else count + a
    return int(np.count_nonzero(count >= 1)), int(np.count_nonzero(count >= 2))


def episode_realism(
    inst: Instance, omega: Omega, sample: EventSample | None = None, params: GeneratorParams | None = None
) -> EpisodeRealism:
    """The V22 measures of one omega; the record count too when its ``sample`` and ``params`` are given.

    Raises:
        ValueError: if only one of ``sample`` and ``params`` is given, or omega is refused by ``marks`` (V3).

    """
    if (sample is None) != (params is None):
        raise ValueError("the record count needs both the episode's EventSample and its GeneratorParams")
    a = omega.arrays
    wm = _marks.compute_marks(inst, omega)
    onset = np.asarray(a["ev_onset"])
    closing = (np.asarray(a["ev_type"]) == MILITARISED_CLOSURE) & (np.asarray(a["ev_target_kind"]) == K_CHOKEPOINT)
    closures = int(np.count_nonzero(closing & (onset >= 0.0) & (onset < inst.T)))
    rec, rec_weeks = (0, 0.0) if sample is None else record_closures(inst, params, sample)
    o = np.asarray(wm.o)  # (T, C) week averages of (37)
    capacity = event_capacity(inst, omega)  # each event's marks once, for harm and the overlap count
    _, harm, by_type = _harm.totals(inst, omega, capacity=capacity)
    tw, combined = target_weeks(inst, omega, capacity=capacity)
    return EpisodeRealism(
        weeks=inst.T,
        closures=closures,
        record_closures=rec,
        record_weeks=rec_weeks,
        impaired=float(np.count_nonzero(o <= IMPAIRMENT_CUT)) / inst.T,
        open_fraction=tuple(float(math.fsum(o[:, c].tolist())) / inst.T for c in range(o.shape[1])),
        harm=harm,
        harm_by_type=tuple(by_type[code] for code in range(len(codes.EVENT_TYPES))),
        target_weeks=tw,
        combined_target_weeks=combined,
    )


def measure_episode(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> EpisodeRealism:
    """Draw episode n of the generator (``sampler.sample_events``, ``sample_omega``) and measure it."""
    s = sample_events(inst, params, entropy, episode)
    return episode_realism(inst, sample_omega(inst, params, entropy, episode, sample=s), s, params)


def measure(
    inst: Instance, params: GeneratorParams, entropy: int, episodes: int | range, n_jobs: int = 1
) -> tuple[EpisodeRealism, ...]:
    """``measure_episode`` for the episodes n in ``episodes`` (an int m means 0..m-1), in episode order.

    Args:
        inst: the instance.
        params: the generator at the rung.
        entropy: E_split of (27).
        episodes: the episode indices.
        n_jobs: joblib workers over chunks of episodes (imported only when not 1); the result does not depend on it.

    Raises:
        TypeError: if ``episodes`` is neither an integer nor a range.

    """
    if isinstance(episodes, numbers.Integral) and not isinstance(episodes, bool):
        episodes = range(int(episodes))
    if not isinstance(episodes, range):
        raise TypeError(f"episodes must be an integer or a range, got {episodes!r}")
    return tuple(ordered_map(functools.partial(measure_episode, inst, params, entropy), episodes, n_jobs))


# ----- over episodes -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Estimate:
    """A mean over independent episodes and its standard error."""

    mean: float
    se: float
    n: int


def estimate(values: Sequence[float]) -> Estimate:
    """Mean and standard error (sample SD with n - 1, over sqrt n) of per-episode values.

    Raises:
        ValueError: with fewer than two values or a non-finite one.

    """
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("an estimate needs at least two finite per-episode values")
    return Estimate(float(x.mean()), float(x.std(ddof=1)) / math.sqrt(x.size), int(x.size))


@dataclass(frozen=True)
class RealismSummary:
    """The V22 items over a set of episodes of one rung."""

    episodes: int
    closures_per_year: Estimate  # omega's window [0, T), the published estimator
    record_closures_per_year: Estimate | None  # the stationary record, None when no episode carries one
    impaired: Estimate
    open_fraction: tuple[Estimate, ...]  # per chokepoint ordinal
    harm: Estimate  # (42), USD
    harm_quantiles: tuple[float, ...]  # at HARM_QUANTILES, the inverse CDF of (44)'s rule
    harm_zero_share: float  # episodes with H = 0
    harm_by_type: tuple[float, ...]  # mean H_ty of (43) per type code
    combined_share: float  # sum of combined target-weeks over sum of target-weeks (0 when there are none)


def summarize(episodes: Sequence[EpisodeRealism]) -> RealismSummary:
    """The V22 items of ``episodes`` (module docstring).

    Raises:
        ValueError: with fewer than two episodes, or episodes of different horizons or chokepoint counts.

    """
    if len(episodes) < 2:
        raise ValueError("a summary needs at least two episodes")
    if len({(e.weeks, len(e.open_fraction), len(e.harm_by_type)) for e in episodes}) != 1:
        raise ValueError("the episodes differ in horizon, chokepoints or event types")
    T = episodes[0].weeks
    with_record = [e for e in episodes if e.record_weeks > 0.0]
    if with_record and len(with_record) != len(episodes):
        raise ValueError("either every episode carries its record count or none does")
    harms = np.sort(np.array([e.harm for e in episodes], dtype=np.float64), kind="stable")
    tw = sum(e.target_weeks for e in episodes)
    return RealismSummary(
        episodes=len(episodes),
        closures_per_year=estimate([e.closures / T * WEEKS_PER_YEAR for e in episodes]),
        record_closures_per_year=(
            estimate([e.record_closures / e.record_weeks * WEEKS_PER_YEAR for e in episodes]) if with_record else None
        ),
        impaired=estimate([e.impaired for e in episodes]),
        open_fraction=tuple(
            estimate([e.open_fraction[c] for e in episodes]) for c in range(len(episodes[0].open_fraction))
        ),
        harm=estimate(harms.tolist()),
        harm_quantiles=tuple(inverse_cdf(harms, j) for j in HARM_QUANTILES),
        harm_zero_share=float(np.count_nonzero(harms == 0.0)) / len(episodes),
        harm_by_type=tuple(
            math.fsum(e.harm_by_type[i] for e in episodes) / len(episodes) for i in range(len(codes.EVENT_TYPES))
        ),
        combined_share=sum(e.combined_target_weeks for e in episodes) / tw if tw else 0.0,
    )


# ----- the band test -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BandCheck:
    """One gated item of the band: its estimate, its edges and the one-sided test (module docstring)."""

    item: str
    estimate: Estimate
    lo: float
    hi: float
    statistic: float  # max((mean - hi) / SE, (lo - mean) / SE): above 0 outside the band
    critical: float  # t_{1 - alpha, n - 1}
    passed: bool
    constituents: int = 1  # k: the flagship chokepoints an open-fraction item stands for (1 for every other item)

    @property
    def per_constituent(self) -> Estimate:
        """The estimate read per constituent: mean^(1/k), with the delta-method SE (the estimate itself at k = 1).

        A zero mean has no delta-method SE; it is reported as infinite.
        """
        k, est = self.constituents, self.estimate
        if k == 1:
            return est
        if est.mean <= 0.0:
            return Estimate(max(est.mean, 0.0), math.inf, est.n)
        return Estimate(est.mean ** (1.0 / k), est.se * est.mean ** (1.0 / k - 1.0) / k, est.n)

    @property
    def per_constituent_floor(self) -> float:
        """lo^(1/k): the lower edge read per constituent (lo itself at k = 1)."""
        return self.lo if self.constituents == 1 else self.lo ** (1.0 / self.constituents)


def check_item(
    item: str, est: Estimate, lo: float, hi: float, alpha: float = ALPHA, constituents: int = 1
) -> BandCheck:
    """The one-sided band test of one item: it fails when the estimate lies beyond an edge by more than t SE.

    ``constituents`` only labels the result (``BandCheck.per_constituent``); the test reads ``lo`` and ``hi`` as given.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if est.n < 2 or not math.isfinite(est.mean) or not est.se >= 0.0:
        raise ValueError(f"a band test needs n >= 2, a finite mean and an SE >= 0, got {est!r}")
    crit = float(stats.t.ppf(1.0 - alpha, est.n - 1))
    gaps = (est.mean - hi, lo - est.mean)
    if est.se > 0.0:
        stat = max(g / est.se for g in gaps)
    else:  # every episode gave the same value: outside means certainly outside
        stat = math.inf if max(gaps) > 0.0 else -math.inf
    return BandCheck(item, est, lo, hi, stat, crit, not stat > crit, constituents)


def band_checks(
    summary: RealismSummary, band: RealismBand, gamma: float, alpha: float = ALPHA
) -> tuple[BandCheck, ...]:
    """The gated items of ``band`` at rung ``gamma`` (§4.4), each by ``check_item``.

    At the anchor closures per year (the record estimator when the summary has it) and impaired chokepoints; at the
    stress rungs the closure ceiling if the band has one; at every rung each chokepoint's open fraction (at alpha / C),
    against the floor^k of its k constituents (Q94 (a); module docstring).

    Raises:
        ValueError: if the band states constituents for another number of chokepoints.

    """
    closures = summary.record_closures_per_year or summary.closures_per_year
    out = []
    if gamma == band.anchor:
        out.append(check_item("closures_per_year", closures, *band.closures_per_year, alpha))
        out.append(check_item("impaired", summary.impaired, *band.impaired, alpha))
    if band.stress_closures_per_year is not None and gamma in band.stress_rungs:
        out.append(check_item("stress_closures_per_year", closures, -math.inf, band.stress_closures_per_year, alpha))
    C = len(summary.open_fraction)
    for c, (est, k) in enumerate(zip(summary.open_fraction, band.constituent_counts(C), strict=True)):
        out.append(check_item(f"open_fraction[{c}]", est, band.min_open_fraction**k, math.inf, alpha / C, k))
    return tuple(out)


# ----- the stated power (Q94 (b)) ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Alternative:
    """The alternative against which one gated item's power is stated (module docstring, "Stated power")."""

    item: str
    lo: float  # the item's band edges, as ``band_checks`` tests them
    hi: float
    value: float  # the alternative mean: the stated distance past the violated edge
    sd: float  # the per-episode SD at the alternative
    alpha: float  # the item's level (alpha / C for an open fraction)
    support: tuple[float, float] | None = None  # the range of a bounded item's per-episode value; None when unbounded

    @property
    def gap(self) -> float:
        """How far the alternative lies past the violated edge."""
        return max(self.value - self.hi, self.lo - self.value)


def bounded_sd(mean: float, a: float, b: float) -> float:
    """sqrt((b - mean)(mean - a)): the largest SD of a law on [a, b] with this mean (Bhatia-Davis; the two-point law).

    Raises:
        ValueError: if the mean lies outside [a, b].

    """
    if not a <= mean <= b:
        raise ValueError(f"a mean {mean!r} outside [{a!r}, {b!r}]")
    return math.sqrt((b - mean) * (mean - a))


def band_power(gap: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    """Power of the one-sided band test on n episodes against a mean ``gap`` past an edge, per-episode SD ``sd``.

    The test fails when the mean lies past the edge by more than t_{1-alpha, n-1} SE, with the SE the sample's; for
    normal per-episode values the statistic is noncentral t with n - 1 degrees of freedom and noncentrality
    gap sqrt(n) / sd, so this is exact for them and the normal approximation otherwise.

    Raises:
        ValueError: if n < 2, the gap is not positive or the SD is not.

    """
    if n < 2 or not gap > 0.0 or not sd > 0.0:
        raise ValueError(f"power needs n >= 2, a positive gap and a positive SD, got {n!r}, {gap!r}, {sd!r}")
    crit = float(stats.t.ppf(1.0 - alpha, n - 1))
    return float(stats.nct.sf(crit, n - 1, gap * math.sqrt(n) / sd))


def two_point_power(alt: Alternative, n: int) -> float:
    """The exact power of ``check_item`` on n episodes of the two-point law on ``alt.support`` with mean ``alt.value``.

    x of the n episodes take the upper point b and the rest the lower point a, x ~ Binomial(n, (value - a) / (b - a));
    the mean, the SE and the statistic of each x are ``check_item``'s, and the power sums the binomial weights of the
    x it fails. This law has the largest SD of any law of that mean on the range (``bounded_sd``).

    Raises:
        ValueError: if the alternative has no support, its mean lies outside it, or n < 2.

    """
    if alt.support is None or n < 2:
        raise ValueError("the two-point power needs a bounded item and n >= 2")
    a, b = alt.support
    if not a <= alt.value <= b or not a < b:
        raise ValueError(f"the alternative mean {alt.value!r} lies outside the support {alt.support!r}")
    x = np.arange(n + 1)
    mean = a + (b - a) * x / n
    se = (b - a) * np.sqrt(x * (n - x) / (n * (n - 1.0))) / math.sqrt(n)
    gap = np.maximum(mean - alt.hi, alt.lo - mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(se > 0.0, gap / se, np.where(gap > 0.0, np.inf, -np.inf))
    crit = float(stats.t.ppf(1.0 - alt.alpha, n - 1))
    return float(math.fsum(stats.binom.pmf(x[stat > crit], n, (alt.value - a) / (b - a)).tolist()))


def item_power(alt: Alternative, n: int) -> float:
    """The stated power on n episodes: ``band_power``, or for a bounded item its minimum with ``two_point_power``."""
    p = band_power(alt.gap, alt.sd, n, alt.alpha)
    return p if alt.support is None else min(p, two_point_power(alt, n))


def episodes_for_power(alt: Alternative, power: float = POWER, n_max: int = 100_000) -> int:
    """The smallest n >= 2 whose ``item_power`` reaches ``power`` (a scan: the two-point power is not monotone in n).

    Raises:
        ValueError: if no n up to ``n_max`` reaches it.

    """
    for n in range(2, n_max + 1):
        if item_power(alt, n) >= power:
            return n
    raise ValueError(f"no episode count up to {n_max} gives {alt.item} power {power}")


def band_alternatives(
    band: RealismBand,
    gamma: float,
    n_chokepoints: int,
    closures_dispersion: float | None = None,
    alpha: float = ALPHA,
) -> tuple[Alternative, ...]:
    """The stated alternatives of the items ``band_checks`` gates at rung ``gamma`` (module docstring), in its order.

    Args:
        band: the band.
        gamma: the rung.
        n_chokepoints: C, the instance's chokepoints.
        closures_dispersion: kappa = SD / sqrt(rate) of the per-episode closure estimator at the rung, from a pilot;
            needed only where a closure item is gated (the anchor; a stress rung of a band with a ceiling).
        alpha: the test level.

    Raises:
        ValueError: if a closure item is gated and no positive dispersion is given, or the band's constituents do
            not fit C.

    """

    def closure_alternative(item: str, lo: float, hi: float) -> Alternative:
        if not (closures_dispersion is not None and closures_dispersion > 0.0):
            raise ValueError(f"rung {gamma} gates {item}: pass its pilot dispersion kappa = SD / sqrt(rate)")
        rate = (1.0 + CLOSURES_DISTANCE) * hi
        return Alternative(item, lo, hi, rate, closures_dispersion * math.sqrt(rate), alpha)

    out = []
    if gamma == band.anchor:
        out.append(closure_alternative("closures_per_year", *band.closures_per_year))
        lo, hi = band.impaired
        m, box = hi + IMPAIRED_DISTANCE, (0.0, float(n_chokepoints))
        out.append(Alternative("impaired", lo, hi, m, bounded_sd(m, *box), alpha, box))
    if band.stress_closures_per_year is not None and gamma in band.stress_rungs:
        out.append(closure_alternative("stress_closures_per_year", -math.inf, band.stress_closures_per_year))
    a = alpha / n_chokepoints
    for c, k in enumerate(band.constituent_counts(n_chokepoints)):
        floor, m = band.min_open_fraction**k, (band.min_open_fraction - OPEN_FRACTION_DISTANCE) ** k
        out.append(Alternative(f"open_fraction[{c}]", floor, math.inf, m, bounded_sd(m, 0.0, 1.0), a, (0.0, 1.0)))
    return tuple(out)


def band_episodes(
    band: RealismBand,
    gamma: float,
    n_chokepoints: int,
    closures_dispersion: float | None = None,
    power: float = POWER,
    alpha: float = ALPHA,
) -> dict[str, int]:
    """The episode count each gated item's stated power sets at rung ``gamma``; the rung runs the largest.

    Each count is ``episodes_for_power`` of the item's ``band_alternatives`` entry.

    Raises:
        ValueError: as ``band_alternatives``.

    """
    alts = band_alternatives(band, gamma, n_chokepoints, closures_dispersion, alpha)
    return {alt.item: episodes_for_power(alt, power) for alt in alts}


# ----- the stress ceiling (§4.4) against the stationary rates of (32) --------------------------------------------
@dataclass(frozen=True)
class StressCeiling:
    """The stationary militarised-closure rate of (32) at one rung beside the §4.4 stress ceiling."""

    gamma: float
    stationary_closures_per_year: float  # ``profiles.closure_rate`` x 52 (mean regime multipliers)
    scaled_ceiling: float  # 3 x 0.85 x share
    unscaled_ceiling: float  # 3 x 0.85


def stress_ceiling(
    inst: Instance, params: GeneratorParams, gammas: Sequence[float], share: float | None = None
) -> tuple[StressCeiling, ...]:
    """The stationary closure rate per rung (``params.with_rung``) beside the stress ceiling scaled by ``share``.

    ``share`` None reads it from the instance (``chokepoint_share``: 4/7 on `tiny`, 7/7 on `small` and `full`). The
    rate is (32)'s by the instance's own calibration rule: ``profiles.closure_rate`` on `tiny`,
    ``profiles.flagship_closure_rates`` summed on `small` and `full` (effective conflict laws, strait closures).
    Reported, not gated, on `tiny`, which has no stress ceiling (Q94 (d)); the flagship sizes gate the §4.4 ceiling
    on generated episodes (V22), for which this is the stationary reference.
    """
    if share is None:
        share = chokepoint_share(inst)

    def rate(p: GeneratorParams) -> float:
        if inst.kind in FLAGSHIP_KINDS:
            return float(flagship_closure_rates(inst, p).sum())
        return closure_rate(inst, p)

    ceiling = STRESS_MULTIPLIER * W2_CLOSURES_PER_YEAR
    return tuple(StressCeiling(g, rate(params.with_rung(g)) * WEEKS_PER_YEAR, ceiling * share, ceiling) for g in gammas)


# ----- V17's capacity-loss share ------------------------------------------------------------------------------------
capacity_loss_share = _harm.capacity_loss_share  # harm's valuation (41) read week by week, at home in harm


# ----- tanker overrides starving containers (§3.4, Q68) ------------------------------------------------------------
EPISODE_FALLBACK = "episode"  # ``override_starvation``'s default: the D9 fallback spec the episode itself ran with


@dataclass(frozen=True)
class Starvation:
    """Container throughput lost to tanker overrides and holds in one trajectory (module docstring)."""

    override_weeks: tuple[int, ...]  # weeks with a valid override slot or a hold, and no D9 fallback
    released: float  # container-pool units released at chokepoints in those weeks
    counterfactual: float  # the same from the same states with those weeks' overrides and holds dropped
    starved: float  # the positive shortfalls, summed over those weeks and chokepoints


def _sends_override(action) -> bool:
    return isinstance(action, Mapping) and (action.get("overrides") is not None or action.get("hold") is not None)


def container_release(inst: Instance, record) -> dict[int, float]:
    """Container-pool units released at each chokepoint node in one week (``StepRecord.x`` on its out-edges)."""
    parts: dict[int, list[float]] = {c: [] for c in inst.chokepoints}
    for (e, k, _lane), q in record.x.items():
        tail = inst.edges[e].tail
        if tail in parts and inst.commodity_pool[k] == _CONTAINER:
            parts[tail].append(q)
    return {c: math.fsum(v) for c, v in parts.items()}


def override_starvation(
    inst: Instance,
    trajectory,
    omega=None,
    *,
    marks=None,
    params: GeneratorParams | None = None,
    replications: int = naive_fq.REPLICATIONS,
    fallback=EPISODE_FALLBACK,
) -> Starvation:
    """The starvation telemetry of ``trajectory``, replayed from its stored actions (V24) on ``omega`` or ``marks``.

    Each week with an override or a hold (module docstring) is also stepped, from a snapshot of the same state,
    without them; the container releases of the two weeks are compared per chokepoint. The replay plays the D9
    fallback the episode ran with, so its fallback weeks replay identically.

    Args:
        inst: the instance.
        trajectory: a finished trajectory that did not fail.
        omega: the omega it ran on (anything ``Env.reset`` takes), or None with ``marks``.
        marks: the ``WeeklyMarks`` it ran on, instead of omega or beside it (``Env.reset``).
        params: the generator that drew omega, for a generated omega (``Omega.generated``); None for an injected list.
        replications: naive's F_Q replications of a generated omega's fallback (design 1,000; ``naive_fq``).
        fallback: the ``Env`` fallback spec of the rollout; by default (``EPISODE_FALLBACK``) the episode's own,
            ``naive_fq.fallback_spec(inst, params, replications)``: naive with the generator's F_Q for a generated
            omega (§8.1, §9.3), ``"naive"`` (F_Q the point mass at 0) for an injected list.

    Raises:
        ValueError: if the trajectory failed or is unfinished, a generated omega comes without ``params`` (or an
            explicit ``fallback``), the fallback belongs to another generator (``Env.reset``), or the replay does not
            reproduce the trajectory (SHA-256).

    """
    from shockbench_flow.dynamics.env import Env, took_fallback

    if trajectory.failed is not None or trajectory.salvage_cents is None:
        raise ValueError("the telemetry reads a finished trajectory that did not fail")
    if isinstance(fallback, str) and fallback == EPISODE_FALLBACK:
        if params is None and isinstance(omega, Omega) and omega.generated:
            raise ValueError(
                "a generated omega's episode plays naive with its generator's F_Q as the D9 fallback (§8.1, §9.3):"
                " pass the generator's params (or fallback=naive_fq.fallback_spec(inst, params, replications))"
            )
        fallback = naive_fq.fallback_spec(inst, params, replications)
    env = Env(fallback)
    env.reset(inst, trajectory.regime, omega, marks=marks, policy_name=trajectory.policy)
    weeks, released, counterfactual, starved = [], [], [], []
    for t, action in enumerate(trajectory.actions, start=1):
        snap = env.snapshot() if _sends_override(action) else None
        env.step(action)
        rec = env.trajectory.records[-1]
        if snap is None or took_fallback(rec) or not (rec.override_requested or action.get("hold") is not None):
            continue  # nothing overridden or held: a D9 week plays naive, which never overrides (§8.1, §9.3)
        twin = Env(fallback)
        twin.restore(snap)
        twin.step({**action, "overrides": None, "hold": None})
        real, cf = container_release(inst, rec), container_release(inst, twin.trajectory.records[-1])
        weeks.append(t)
        for c in inst.chokepoints:
            released.append(real[c])
            counterfactual.append(cf[c])
            starved.append(max(0.0, cf[c] - real[c]))
    if env.trajectory.sha256() != trajectory.sha256():
        raise ValueError("the replay of the stored actions does not reproduce the trajectory (V24)")
    return Starvation(tuple(weeks), math.fsum(released), math.fsum(counterfactual), math.fsum(starved))

"""Event types, targets and marks (design §4.4), the unexcited Poisson components (Q50), and omega's event arrays.

Every draw is keyed by the event's genealogical key (§4.1 streams), by inversion of keyed uniforms, so the same
event has the same marks at every rung (V2) and whatever the policy does:
- type and target: stream 9, key (*event key): uniforms (type, target). Immigrants are typed over the same
  X-modulated parts as their intensity (``targets.week_parts`` at the key's week, the one rule of the cluster
  sampler's rate); children over ``targets.type_weights``. A block with no feasible type makes a no-op event, which
  is not stored in omega.
- duration: stream 2, key (*event key) (two uniforms for the persistent mixture (39) or the one-day branch);
  severity: stream 3, key (*event key); counterpart of a regional conflict: stream 21, key (*event key), not drawn for
  a partnerless one (counterpart -1, Q94 (f)); tariff rate by the imposing region's tension state at onset: stream 16,
  key (imposing region, *event key); restoration of a regional conflict's fab and OSAT hits: stream 10, key (*event
  key, 0) -> (dead time T0, tau_rho) (regime C, §2.4).
- weather and accident closures: stream 14, key (chokepoint node, week + B_burn) count by Poisson inversion, marks key
  (chokepoint node, week + B_burn, rank): (onset offset, duration, severity by its law, 1 in the §2.4 profile).
- port strikes: stream 15, key (region, week + B_burn) count, marks key (region, week + B_burn, rank): (onset offset,
  stoppage-or-slowdown, duration, severity), on the struck region (active regions only).
- dyad strait closures (`small` and `full`, M5-O13 default): no draw; a regional conflict between the two regions of a
  ``params.laws.strait_closures`` entry derives a militarised closure of its chokepoint, key (*conflict key, 2, node)
  (``strait_closures``).
Events enter omega when they have a graph operation, their window reaches into the episode (end > 0) and their onset
is before T (carried-in events included, §4.3). Weather events use block -1 and keys (7, chokepoint, week + B_burn,
rank); strikes (8, region, week + B_burn, rank).

Readings where the design leaves the rule open (reported for design rows):
- An event of onset s reads the regime of its onset week t = floor(s) + 1 (column t + B_burn); an immigrant reads the
  week of its key (block, region, t + B_burn, rank), the same week unless the onset rounds onto the week's end: the
  conflict state and the dyad chains (``targets.TypeRules.state``) that set the feasible types and name a conflict's
  counterpart, the tension state that sets a tariff's rate, and (immigrants) the chokepoints' g(X^t_c)/c_X.
- A regional conflict takes the partner of a dyad at war that contains its region as counterpart (uniform among
  such partners, stream 21), else a region drawn with weight A_{m .}; a war from a dyad takes precedence over the
  region's own chain. Both are the conflict's candidates (``targets.candidate_targets``), so a dyad war makes the
  conflict feasible in a region with no A partner, against the dyad partner (CMP-1). A war-state region with neither
  gets its conflict with counterpart -1 (Q94 (f)), drawn on no stream 21 uniform (the draws are keyed, so no other
  draw moves); the marks give it its restoration (13) and war-risk class only, with no (40) edge operation.
- The regime-C restoration (stream 10) is stored for a regional conflict whose region hosts a fab or an OSAT, the
  nodes that get (13) under §4.4, with or without a counterpart; elsewhere restoration is -1 and T0, tau_rho NaN.
- omega keeps every regional conflict while D_q = W0 u W1 of (40) reaches past the instant 0, counterpart -1
  included (``window_end``), so a dropped one began at least 104 weeks before 0 and the bound on its dropped (13) tail
  holds for both kinds.
- Piracy's ``rate`` is its drawn surcharge (the severity, U(0.08, 0.12)); tariffs carry the rate and severity 1.
- The weekly Poisson rate is the yearly rate / 52 (the design's year of 52 weeks, (28), §2.4); strikes are drawn
  only for active regions whose ports have a sea edge (``marks``' strike rule), since elsewhere a strike has no
  graph operation.
"""

import math
from dataclasses import dataclass

import numpy as np

from shockbench_flow import marks as _marks
from shockbench_flow.disruption import targets
from shockbench_flow.disruption.hawkes import RawEvent, check_regimes, onset_in_week
from shockbench_flow.disruption.laws import Law, categorical_ppf, poisson_ppf
from shockbench_flow.disruption.params import WEEKS_PER_YEAR, GeneratorParams
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import (
    MILITARISED_CLOSURE,
    PIRACY,
    PORT_STRIKE,
    REGIONAL_CONFLICT,
    TARIFF,
    WEATHER_CLOSURE,
)
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import EVENT_FLOATS, EVENT_INTS, chokepoint_region, event_float, event_group
from shockbench_flow.omega.seeds import SPAWN_BOUND, KeyedStream, check_key


_NAN = float("nan")


@dataclass(frozen=True)
class MarkedEvent:
    """A typed, targeted, marked event, one row of omega's ``ev_*`` arrays (§4.1 table)."""

    key: tuple[int, ...]
    type: int  # codes.EVENT_TYPES index
    block: int  # 0 P, 1 M, -1 Poisson
    region: int
    counterpart: int  # region or -1
    target_kind: int  # 0 chokepoint, 1 edge, 2 node, 3 region
    target: int
    commodity: int  # -1 = every commodity
    onset: float
    duration: float  # T_q in weeks (for a regional conflict, the conflict's window; war-profile windows by (40))
    severity: float
    rate: float  # tariff rate or piracy surcharge; NaN otherwise
    restoration: int = -1  # 0-3 = regimes A-D for fab-hitting events, else -1
    T0: float = float("nan")
    tau_rho: float = float("nan")
    persistent: bool = False  # the persistent component of (39)
    parent: int = -1  # index of the parent in the full (unfiltered) list, -1 for immigrants and Poisson events


@dataclass(frozen=True)
class _Context:
    """Per-call look-ups of the event stage: the type rules (memoised, dyads included), laws by name, restoration."""

    inst: Instance
    params: GeneratorParams
    regimes: object
    rules: targets.TypeRules  # regime states, candidates, type weights and week parts, each computed once per call
    duration: dict[str, Law]
    severity: dict[str, Law]
    restoration_regions: frozenset[int]  # regions hosting a fab or an OSAT: a conflict there gets (13) (§4.4)

    @classmethod
    def build(cls, inst: Instance, params: GeneratorParams, regimes, rules: targets.TypeRules | None) -> "_Context":
        if rules is None:
            rules = targets.TypeRules(inst, params, regimes, targets.adjacency_dict(inst, params))
        return cls(
            inst=inst,
            params=params,
            regimes=regimes,
            rules=rules.check(inst, params, regimes),
            duration=dict(params.laws.duration),
            severity=dict(params.laws.severity),
            restoration_regions=_marks.restoration_regions(inst),
        )


class _Streams:
    """The event stage's keyed streams of (27) for one (E_split, n) (module docstring), validated once.

    ``check`` builds them at the first draw, where E_split and n are checked (``seeds.KeyedStream``), so an input the
    seed rule refuses raises where the first keyed draw would raise it, and never for an empty event list.
    """

    __slots__ = ("entropy", "episode", "type_target", "duration", "severity", "counterpart", "tariff", "restoration")

    def __init__(self, entropy: int, episode: int) -> None:
        self.entropy, self.episode = entropy, episode
        self.type_target: KeyedStream | None = None

    def check(self) -> None:
        if self.type_target is None:
            entropy, episode = self.entropy, self.episode
            self.type_target = KeyedStream(entropy, episode, codes.STREAM_TYPE_TARGET)
            self.duration = KeyedStream(entropy, episode, codes.STREAM_DURATION)
            self.severity = KeyedStream(entropy, episode, codes.STREAM_SEVERITY)
            self.counterpart = KeyedStream(entropy, episode, codes.STREAM_COUNTERPART)
            self.tariff = KeyedStream(entropy, episode, codes.STREAM_TARIFF)
            self.restoration = KeyedStream(entropy, episode, codes.STREAM_RESTORATION)


def _counterpart(ctx: _Context, s: _Streams, ev: RawEvent, key: tuple[int, ...], col: int) -> int:
    """A regional conflict's counterpart (stream 21): a partner of a dyad at war in the onset week, else by A_{m .}.

    The inversion of the stream-21 uniform over the conflict's candidates at the region's state in column ``col``
    (``targets.candidate_targets``: the dyad partners at war, weight 1 each, which take precedence (Q59), else A), so
    there is always one. A region with neither has the single candidate of counterpart -1 (Q94 (f)): it is returned
    without a draw, since every draw is keyed (27) and skipping one moves no other.
    """
    _, partners = ctx.rules.state(ev.region, col)
    cands = ctx.rules.candidates(REGIONAL_CONFLICT, ev.region, partners)
    if cands[0][0].counterpart < 0:  # partnerless: restoration (13) and war-risk class only, nothing to draw
        return -1
    u = s.counterpart.uniform(key)
    return cands[categorical_ppf(u, [w for _, w in cands])][0].counterpart


def regime_week(ev: RawEvent | MarkedEvent, burn_in: int) -> int:
    """The week whose regime an event reads: an immigrant its key's week, any other event floor(onset) + 1 (§4.4).

    The onset week t_q = floor(onset) + 1; an immigrant of the cluster process (block >= 0, no parent) reads its key's
    (block, region, week + B_burn, rank), the week whose intensity drew it, so its type law and its intensity read one
    week whatever the onset's rounding. Children and the unexcited Poisson events read their onset week.
    """
    if ev.block >= 0 and ev.parent < 0:
        return int(ev.key[2]) - burn_in
    return math.floor(ev.onset) + 1


def type_parts(rules: targets.TypeRules, regimes, ev: RawEvent, burn_in: int) -> tuple[int, list]:
    """(onset column, typed parts) one raw event is typed over: the one typing rule of (30) (§2.4, Q51).

    Immigrants are typed over the parts of their intensity (``TypeRules.week_parts`` at their key's week, closures
    modulated by g(X^t_c)/c_X); children over the block's renormalised type law at their own region and week, at the
    region's effective z^c there, or the single no-op part (-1, None, 1.0) where it is empty. A
    closure immigrant's part fixes its chokepoint. Real events (stream 9) and shadow events (stream 7, Q101) alike.
    """
    week = regime_week(ev, burn_in)
    col = regimes.col(week)
    if ev.parent < 0:
        return col, rules.week_parts(ev.block, ev.region, week)
    tw = rules.type_weights(ev.block, ev.region, int(regimes.z_c[ev.region, col]))
    return col, [(code, None, w) for code, w in tw] or [(-1, None, 1.0)]


def _typed(ctx: _Context, s: _Streams, ev: RawEvent) -> tuple:
    """The stream-9 draw of one raw event: (key, onset column, type code or -1, part target or None, target uniform).

    The key is checked once here (``seeds.check_key``) and drawn from as it is on every stream after; the parts are
    ``type_parts``'.
    """
    s.check()
    key = check_key(ev.key)
    u_type, u_target = s.type_target.uniforms(key, 2)
    col, parts = type_parts(ctx.rules, ctx.regimes, ev, ctx.params.burn_in)
    ty, target, _ = parts[categorical_ppf(float(u_type), [p[2] for p in parts])]
    return key, col, ty, target, float(u_target)


def _marked(
    ctx: _Context, s: _Streams, ev: RawEvent, key: tuple[int, ...], col: int, ty: int, target, u_target: float
) -> MarkedEvent:
    """Target, marks and row of one typed raw event (type ``ty`` >= 0), every draw keyed by its genealogical key."""
    params, reg = ctx.params, ctx.regimes
    name = codes.EVENT_TYPES[ty]
    counterpart = -1
    if ty == REGIONAL_CONFLICT:  # the event acts on its own region m_q, with a counterpart for (40)
        target = targets.Target(targets.REGION, ev.region)
        counterpart = _counterpart(ctx, s, ev, key, col)
    elif target is None:
        cands = ctx.rules.candidates(ty, ev.region)
        target = cands[categorical_ppf(u_target, [w for _, w in cands])][0]
    if ty == TARIFF:
        counterpart = target.counterpart  # the exporter
    dur_law = ctx.duration[name]
    u_dur = s.duration.uniforms(key, dur_law.uniforms)
    duration = dur_law.ppf(u_dur)  # (39) for closures; T_q in weeks (days / 7)
    severity = ctx.severity[name].ppf(s.severity.uniform(key))
    rate = _NAN
    if ty == TARIFF:  # the imposing region's tension state at onset (Q32); the uniform keys a per-state law
        z_p = int(reg.z_p[ev.region, col])
        u_rate = s.tariff.uniform(check_key((ev.region, *key)))
        rate = Law("constant", (params.laws.tariff_rate_by_tension[z_p],)).ppf(u_rate)
    elif ty == PIRACY:
        rate = severity  # the surcharge c x (1 + rate) of §4.4
    restoration, T0, tau_rho = -1, _NAN, _NAN
    if ty == REGIONAL_CONFLICT and ev.region in ctx.restoration_regions:
        u_rho = s.restoration.uniforms((*key, 0), 2)
        restoration = params.laws.conflict_restoration_regime
        T0 = params.laws.conflict_dead_time.ppf(u_rho[0])
        tau_rho = params.laws.conflict_tau_rho.ppf(u_rho[1])
    return MarkedEvent(
        key=key,
        type=ty,
        block=int(ev.block),
        region=int(ev.region),
        counterpart=int(counterpart),
        target_kind=int(target.kind),
        target=int(target.index),
        commodity=int(target.commodity),
        onset=float(ev.onset),
        duration=float(duration),
        severity=float(severity),
        rate=float(rate),
        restoration=int(restoration),
        T0=float(T0),
        tau_rho=float(tau_rho),
        persistent=bool(dur_law.is_persistent(u_dur)),
        parent=int(ev.parent),
    )


def _mark(ctx: _Context, s: _Streams, ev: RawEvent) -> MarkedEvent | None:
    """Type, target and marks of one raw event, every draw keyed by its genealogical key (§4.1 streams)."""
    key, col, ty, target, u_target = _typed(ctx, s, ev)
    if ty < 0:  # no feasible type: a no-op event, kept in the genealogy, never stored
        return None
    return _marked(ctx, s, ev, key, col, ty, target, u_target)


def mark_events(
    inst: Instance,
    params: GeneratorParams,
    regimes,
    raw: list[RawEvent],
    entropy: int,
    episode: int,
    *,
    rules: targets.TypeRules | None = None,
) -> list[MarkedEvent | None]:
    """Type, target and mark every raw event (None for a no-op event), in the order of ``raw``.

    Each event's marks depend only on its key, block, region, onset and the regime paths (never on other events or on
    the rung), so a lower rung's events keep identical marks at a higher rung (V2, §2.6). The type rules are memoised
    for the call (``targets.TypeRules``) and E_split and n checked once, before the first draw.

    Args:
        inst: the instance.
        params: the generator parameters (type laws, mark laws, dyads, adjacency).
        regimes: ``regime.RegimePaths`` of the episode (z_c, z_p, z_dyad, X; ``col(week)``).
        raw: the events of the cluster process (``hawkes.sample_cluster``).
        entropy: E_split of (27).
        episode: n of (27).
        rules: the episode's ``targets.TypeRules`` (built once per episode by ``sampler.sample_events``,
            SIMP-M2R2-05), or None to build them here.

    Returns:
        One ``MarkedEvent`` per raw event, or None for a no-op event (no feasible type in its block and region).

    Raises:
        ValueError: regime paths whose burn-in, T or shapes do not match the params and instance
            (``hawkes.check_regimes``).

    """
    check_regimes(inst, params, regimes)  # an immigrant's week is key[2] - params.burn_in, column week + burn-in
    ctx = _Context.build(inst, params, regimes, rules)
    s = _Streams(entropy, episode)
    return [_mark(ctx, s, ev) for ev in raw]


def mark_closures(
    inst: Instance,
    params: GeneratorParams,
    regimes,
    raw: list[RawEvent],
    entropy: int,
    episode: int,
    *,
    rules: targets.TypeRules | None = None,
) -> list[MarkedEvent]:
    """The militarised closures among ``raw``, each the ``MarkedEvent`` ``mark_events`` gives it, in ``raw`` order.

    Naive's F_Q (66) reads only the chokepoint closures of an episode, so this draws what they need and nothing else:
    every event's stream-9 type draw (its type decides whether it is a closure), then a closure's target, duration
    (39) and severity from the same keys by the same code as ``mark_events``; the other events are typed only.
    Arguments and errors as ``mark_events``.
    """
    check_regimes(inst, params, regimes)
    ctx = _Context.build(inst, params, regimes, rules)
    s = _Streams(entropy, episode)
    out: list[MarkedEvent] = []
    for ev in raw:
        key, col, ty, target, u_target = _typed(ctx, s, ev)
        if ty == MILITARISED_CLOSURE:
            out.append(_marked(ctx, s, ev, key, col, ty, target, u_target))
    return out


STRAIT_KEY = 2  # the key tag of a strait closure: (*conflict key, 2, chokepoint node); children append (0 | 1, m, r)


def strait_closures(
    inst: Instance, params: GeneratorParams, marked: list[MarkedEvent | None] | tuple[MarkedEvent | None, ...]
) -> list[MarkedEvent]:
    """The dyad strait closures of one episode (§4.4 "a CN-TW event also closes the Taiwan Strait"; M5-O13 default).

    For each regional conflict of ``marked`` (aligned with the cluster process, None for a no-op event) between the two
    regions of a ``params.laws.strait_closures`` entry (region a with counterpart b, or b with a), one militarised
    closure of that chokepoint: key (*the conflict's key, ``STRAIT_KEY``, chokepoint node), unique since a cluster
    child's key appends three components; block M; the conflict's region and onset; window W0 = max(T_q,
    ``strait_closure_weeks``) of (40) as its duration; severity ``strait_closure_severity``; not persistent; parent the
    conflict's index. It draws no uniform (the conflict's marks fix it), so no other draw moves; it is added after the
    cluster process, so it excites nothing in (30); omega stores it by the one storage rule (``in_episode``), so V22
    and the (32) calibration count it (``profiles.flagship_closure_rates``), while ``sampler.sample_closures``, which
    feeds naive's transient F_Q, leaves it out. A params tree without ``strait_closures`` (`tiny`'s) gives none.

    Raises:
        ValueError: as ``targets.strait_chokepoints`` (a named region or chokepoint the instance lacks).

    """
    straits = targets.strait_chokepoints(inst, params)
    if not straits:
        return []
    laws = params.laws
    out: list[MarkedEvent] = []
    for i, ev in enumerate(marked):
        if ev is None or ev.type != REGIONAL_CONFLICT or ev.counterpart < 0:
            continue
        c = straits.get(frozenset((ev.region, ev.counterpart)))
        if c is None:
            continue
        out.append(
            MarkedEvent(
                key=check_key((*ev.key, STRAIT_KEY, c)),
                type=MILITARISED_CLOSURE,
                block=1,
                region=ev.region,
                counterpart=-1,
                target_kind=targets.CHOKEPOINT,
                target=c,
                commodity=-1,
                onset=ev.onset,
                duration=max(float(ev.duration), float(laws.strait_closure_weeks)),  # W0 of (40)
                severity=float(laws.strait_closure_severity),
                rate=_NAN,
                parent=i,
            )
        )
    return out


def _strike_regions(inst: Instance, params: GeneratorParams) -> list[int]:
    """Active regions whose ports have a sea edge, in region order: ``marks.strike_edges`` on a region target."""
    active = sorted(inst.region_index[r] for r in params.active_regions)
    return [m for m in active if _marks.strike_edges(inst, targets.REGION, m)]


def _unit_week_events(
    stream: int, units, lam: float, k: int, B: int, T: int, entropy: int, episode: int, make
) -> list[MarkedEvent]:
    """The events of one unexcited Poisson stream (Q50): per unit, then per week -B_burn + 1 .. T, in that order.

    The week's count is Poisson(lam) by inversion of the uniform of key (unit, week + B_burn); rank r's ``k`` mark
    uniforms are those of key (unit, week + B_burn, r), and ``make(unit, week, week + B_burn, r, u)`` builds its event.
    E_split and n are checked at the first draw and each unit once; every week key lies in [1, B_burn + T], checked
    against 2**32 once (the message the first key beyond it would give).
    """
    out: list[MarkedEvent] = []
    draw = None
    for unit in units:
        weeks = range(-B + 1, T + 1)
        if weeks:
            if draw is None:
                draw = KeyedStream(entropy, episode, stream)
            check_key((unit, min(B + T, SPAWN_BOUND)))
        for week in weeks:
            wk = week + B
            n = poisson_ppf(draw.uniform((unit, wk)), lam)
            for rank in range(n):
                out.append(make(unit, week, wk, rank, draw.uniforms((unit, wk, rank), k)))
    return out


def weather_events(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> list[MarkedEvent]:
    """Weather and accident closures (stream 14) with onset in [-B_burn, T), chokepoint by chokepoint.

    Counts per (chokepoint, week) at the yearly rate / 52 (Q50, §2.4); marks (onset offset, duration, severity) from
    the (chokepoint, week + B_burn, rank) key. A closure acts on its chokepoint with its first adjacent region as m_q
    (the injected rule, §12). A zero rate draws nothing (V6-V8).
    """
    lam = params.poisson.weather_per_chokepoint_year / WEEKS_PER_YEAR
    d_law = dict(params.laws.duration)["weather_closure"]
    s_law = dict(params.laws.severity)["weather_closure"]

    def make(c: int, week: int, wk: int, rank: int, u: np.ndarray) -> MarkedEvent:
        return MarkedEvent(
            key=(WEATHER_CLOSURE, c, wk, rank),
            type=WEATHER_CLOSURE,
            block=-1,
            region=chokepoint_region(inst, c),
            counterpart=-1,
            target_kind=targets.CHOKEPOINT,
            target=c,
            commodity=-1,
            onset=onset_in_week(week, float(u[0])),
            duration=d_law.ppf(u[1]),
            severity=s_law.ppf(u[2]),
            rate=_NAN,
        )

    units = inst.chokepoints if lam > 0 else ()
    return _unit_week_events(codes.STREAM_WEATHER, units, lam, 3, params.burn_in, inst.T, entropy, episode, make)


def strike_events(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> list[MarkedEvent]:
    """Port strikes (stream 15) with onset in [-B_burn, T), struck region by region (``_strike_regions``).

    Counts per (region, week) at the yearly rate / 52 (Q50, §2.4); marks (onset offset, stoppage or slowdown, duration,
    severity) from the (region, week + B_burn, rank) key. A zero rate draws nothing (V6-V8).
    """
    pp = params.poisson
    lam = pp.strike_per_region_year / WEEKS_PER_YEAR

    def make(m: int, week: int, wk: int, rank: int, u: np.ndarray) -> MarkedEvent:
        stoppage = float(u[1]) < pp.stoppage_share
        dl, sl = (
            (pp.stoppage_duration, pp.stoppage_severity) if stoppage else (pp.slowdown_duration, pp.slowdown_severity)
        )
        return MarkedEvent(
            key=(PORT_STRIKE, m, wk, rank),
            type=PORT_STRIKE,
            block=-1,
            region=m,
            counterpart=-1,
            target_kind=targets.REGION,
            target=m,
            commodity=-1,
            onset=onset_in_week(week, float(u[0])),
            duration=dl.ppf(u[2]),
            severity=sl.ppf(u[3]),
            rate=_NAN,
        )

    units = _strike_regions(inst, params) if lam > 0 else ()
    return _unit_week_events(codes.STREAM_STRIKE, units, lam, 4, params.burn_in, inst.T, entropy, episode, make)


def poisson_events(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> list[MarkedEvent]:
    """Weather and accident closures (stream 14) then port strikes (stream 15) with onset in [-B_burn, T).

    Counts per (unit, week) by Poisson inversion of the week key's uniform at the yearly rate / 52 (Q50, §2.4); each
    event's marks from its (unit, week + B_burn, rank) key (``weather_events``, ``strike_events``). Weather closures
    act on the chokepoint with its first adjacent region as m_q (the injected rule, §12); strikes on their region. Zero
    rates draw nothing (V6-V8); a weekly rate beyond the exact range of the inversion (above about 708, i.e. about
    36,800 a year) raises ValueError (``laws.poisson_ppf``).
    """
    return weather_events(inst, params, entropy, episode) + strike_events(inst, params, entropy, episode)


def window_end(ev: MarkedEvent, window: float) -> float:
    """End of the event's storage window: D_q = W0 u W1 of (40) for every regional conflict, else T_q.

    A conflict with counterpart -1 (Q94 (f)) has no (40) edge operation, yet it is kept by the same D_q as one with a
    counterpart: its (13) dead time T0 may outlast T_q, and D_q ends at least 2 w after its onset (w =
    ``war_profile_window``), so a dropped conflict of either kind began at least 2 w before the instant 0 and the
    bound on its dropped (13) tail holds (design §12, "Restoration tail of dropped carried-in conflicts"). The burn-in
    (38) reads the same window (``profiles.duration_quantiles``).
    """
    if ev.type == REGIONAL_CONFLICT:
        return _marks.war_windows(ev.onset, ev.duration, window)[1]
    return ev.onset + ev.duration


def in_episode(ev: MarkedEvent, T: int, window: float) -> bool:
    """Stored in omega: onset before T and the storage window (``window_end``) reaching past the instant 0 (§4.3)."""
    return ev.onset < T and window_end(ev, window) > 0.0


def _row(inst: Instance, i: int, ev: MarkedEvent, burn_in: int) -> dict[str, object]:
    """The event-group row of one generated event, after the generator's own rules (the decoder's are read_events').

    A generated event has a block of (30) or -1 (the unexcited components), a region m_q, a restoration code -1..3 and
    an onset in the sampled window [-B_burn, T) (§4.3); every other rule (codes, index ranges, a region target naming
    its region, the counterpart, a finite onset and a duration >= 0) is ``marks.read_events``', run on the written
    group, so the generator and the decoder cannot disagree (SIMP-M2R2-01).

    Raises:
        ValueError: on a block, region or restoration code outside those sets, or an onset outside [-burn_in, T).
        TypeError: if the onset is not a real number (``container.event_float``).

    """
    if ev.block not in (-1, 0, 1) or not 0 <= ev.region < len(inst.regions) or ev.restoration not in (-1, 0, 1, 2, 3):
        raise ValueError(f"event {i}: block {ev.block}, region {ev.region} or restoration {ev.restoration} invalid")
    onset = event_float(i, "onset", ev.onset)  # an infinite onset raises
    if not -burn_in <= onset < inst.T:  # NaN fails too: onsets lie in [-B_burn, T) (§4.3)
        raise ValueError(f"event {i}: onset {ev.onset} outside [-{burn_in}, {inst.T})")
    return {"key": ev.key, **{n: getattr(ev, n) for n in EVENT_INTS}, **{n: getattr(ev, n) for n in EVENT_FLOATS}}


def event_arrays(inst: Instance, events: list[MarkedEvent], burn_in: int, prefix: str = "ev") -> dict[str, np.ndarray]:
    """Omega's ``ev_*`` group (and CSR keys) for the stored events, in list order (§4.1 table).

    One row per event in the dtypes of ``container.EVENT_FIELDS``, written by the one event-group writer
    (``container.event_group``, which the injected builder uses too; SIMP-M2R2-01): leads NaN on every channel (the
    announcement stage writes them, ``disruption.announce``); float fields by ``container.event_float`` (every NaN
    the one quiet NaN, -0.0 as 0.0), so ``marks.compute_marks`` reads generated and injected events alike.
    Genealogical keys go to ``<prefix>_key`` in CSR form. The written group is checked by the one decoder,
    ``marks.read_events``, besides the generator's own rules (``_row``). ``prefix`` 'sh' writes the shadow group Y
    of (26) the same way (M3).

    Raises:
        ValueError: on an unknown code, an index out of range, an infinite field, an onset outside [-burn_in, T), a
            negative or NaN duration, a key component outside [0, 2**32), or a prefix other than 'ev' or 'sh'.
        TypeError: if a float field is not a real number (a bool included; ``container.event_float``).

    """
    if prefix not in ("ev", "sh"):
        raise ValueError(f"event_arrays writes the ev_ or the sh_ group, got prefix {prefix!r}")
    out = event_group([_row(inst, i, ev, burn_in) for i, ev in enumerate(events)], prefix)
    _marks.read_events(inst, out, prefix)  # the one decoder's rules on what was written
    return out

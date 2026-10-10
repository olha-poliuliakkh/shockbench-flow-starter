"""The announcement stage of the generator: leads, shadow events and blackout spells (design §4.1, §5.3; (26), (49)).

Three keyed draws (27), all into omega, none at reset or in a week (so every regime of an episode shares them:
pairing at the path level, R5.11), and none inside ``sampler.sample_closures`` (naive's F_Q stays bit-identical):

- **Leads** (stream 4, key (*event key, channel)): every stored event of an announced type gets a lead on each
  channel of its type's set (``codes.TYPE_CHANNELS``, which ``InformationParams.channels`` equals; symmetric sets,
  Q97) by inversion, l_q = F^{-1}_ch(V_q) (49), in weeks: a tariff a proposal (key (*key, 0), two uniforms: informal
  iff the first is < ``informal_share`` = 25/57, its lead in column tariff_informal, else tariff_formal; the second its
  lead uniform) and a final notice (key (*key, 2)); a sanction or export control a TIES threat (key (*key, 4)) and a
  legal publication (key (*key, 3)); a militarised closure a MID threat (key (*key, 5)). ``ev_lead`` (events, 6)
  holds them, NaN on every other channel; ``V_lead`` (events,) keeps the signed-off §4.1 shape and holds the uniform of
  the event's decoy-bearing channel (proposal, TIES or MID), NaN for other events. A tariff's final notice never comes
  before its proposal (Q12.3's order): its lead is min(F^{-1}_final(its own uniform), the proposal's lead); a
  sanction thread's two leads are independent draws (design §12 "Messages", "Stream 4, the leads"). A lead mode other
  than 'record' cannot re-invert the second notice or a shadow lead from omega, so it redraws omega (design §12 M3
  row). Leads never enter the marks (``marks.read_events`` reads the scalar columns only), so J, the oracle and every
  frozen cent are unchanged (V3).
- **Shadows** Y (stream 7; (49) as Q101 amends it): a second cluster process of (30), the real one's law with every
  immigrant rate times k*, the largest shadow threads per real event of an announced type (``type_odds``): the same
  immigrant parts (the Q51 closure modulation g(X_c)/c_X included), branching matrix Gamma (31), kernel beta
  e^{-beta s} and typing rule (``events.type_parts``, the one composition of ``targets.TypeRules``, built once per
  episode, SIMP-M2R2-05), over the burn-in and the episode. Shadows excite shadows only, and real events never
  excite shadows: the real genealogy does not enter Y, so a shadow thread's children are announced and thinned like a
  real thread's (``shadow_cascade``). Keys are stream 1's layout on stream 7 (``SH_COUNT`` .. ``SH_DELAY``; a
  shadow's key is its genealogy, as a real event's). A shadow of an announced type y with its effect instant in
  [0, T) becomes a decoy thread when its cluster's uniform U_c (key (``SH_CLUSTER``, *immigrant key), shared by every
  member of the cluster) times k*/k_y is below 1; that product is U_decoy, the thinning uniform of (49), so a cluster
  keeps its members of one type together and each channel keeps its phi-bar in expectation (a tariff's proposal
  channel is drawn with the informal odds p k_informal/k_T, which keeps both proposal channels at their own
  phi-bar under one threshold). Every other shadow is latent: never stored, never shown. The marks come from key
  (``SH_MARKS``, *shadow key), a fixed vector of ``SHADOW_UNIFORMS`` uniforms: the type, the target (a real event's
  rule: a closure immigrant's part fixes its chokepoint, else ``TypeRules.candidates``), the proposal channel, the
  rate (a tariff's by tension state), the thread's leads on every channel of its type's set (the final notice capped
  at the proposal's lead, as for real threads), and a duration and severity from the type's laws (never acted on, so
  every sh_* row passes the one event writer and ``marks.read_events(prefix='sh')``; design §12 "Stream 7, the
  shadows"). Offspring counts by inversion are non-decreasing in Gamma, so the shadows nest across gamma rungs like
  the real events (V2), and across phi (CRN of (49)); a larger k* redraws them (design §12 rows "Shadows and V2's
  subset rule", "A larger phi-bar redraws Y").
- **Blackout spells** (stream 11, key (spell kind)): one spell per kind of ``codes.BLACKOUT_SPELLS``, stored in kind
  order in ``blk_start`` and ``blk_end`` (weeks blk_start <= t < blk_end; to_end ends at T + 1), the onset uniform over
  the admissible weeks by one uniform (``InformationParams.blackout_onset``, SYNTHETIC(placeholder); design §12
  "Stream 11, the blackout spells").

The event-free twin (57) keeps the shadow group, X, W and z_c_own and empties ``ev_lead`` and ``V_lead``
(``omega.injected.event_free``); an injected omega stays as today (every lead NaN, no shadow, no spell, no M3 array).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from shockbench_flow.disruption import events as _events
from shockbench_flow.disruption.events import MarkedEvent
from shockbench_flow.disruption.hawkes import (
    RawEvent,
    branching_matrix,
    immigrant_rates,
    onset_in_week,
    spectral_radius,
)
from shockbench_flow.disruption.laws import Law, categorical_ppf, poisson_ppf
from shockbench_flow.disruption.params import GeneratorParams, InformationParams, generator_id
from shockbench_flow.disruption.regime import RegimePaths
from shockbench_flow.disruption.targets import Target, TypeRules
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import MILITARISED_CLOSURE, SANCTION, TARIFF
from shockbench_flow.omega import codes
from shockbench_flow.omega.assembly import InformationMeta
from shockbench_flow.omega.seeds import KeyedStream, check_key


if TYPE_CHECKING:
    from shockbench_flow.disruption.sampler import EventSample


FORMAL, INFORMAL, FINAL, LEGAL, TIES, MID = (
    codes.CHANNELS.index(n)
    for n in ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
)
DECOY_CODES = tuple(codes.CHANNELS.index(n) for n in codes.DECOY_CHANNELS)  # (0, 1, 4, 5): the shadow processes
CHANNEL_TYPE = {FORMAL: TARIFF, INFORMAL: TARIFF, TIES: SANCTION, MID: MILITARISED_CLOSURE}  # a process's event type
SECOND_NOTICE = {FORMAL: FINAL, INFORMAL: FINAL, TIES: LEGAL}  # the thread's second channel (Q97 sets); MID has none
ANNOUNCED_TYPES = frozenset(CHANNEL_TYPE.values())  # the types of codes.TYPE_CHANNELS
NOOP = -1  # the type code of a no-op event (``targets.Part``)
# stream 7's key kinds (design §12 "Stream 7, the shadows"): stream 1's cluster keys (``hawkes.sample_cluster``), then
# a shadow's marks vector and its cluster's uniform U_c
SH_COUNT, SH_ONSET, SH_OFFSPRING, SH_DELAY, SH_MARKS, SH_CLUSTER = 0, 1, 2, 3, 4, 5
# the marks vector of key (SH_MARKS, *shadow key), by position
U_TYPE, U_TARGET, U_INFORMAL, U_RATE, U_LEAD, U_SECOND, U_DURATION, U_SEVERITY = 0, 1, 2, 3, 4, 5, 6, 8
SHADOW_UNIFORMS = 9  # the duration takes two uniforms (6, 7) whatever its law (a mixture needs both)
_NAN = float("nan")


def _information(params: GeneratorParams) -> InformationParams:
    """``params.information``, or ValueError: the announcement stage needs it (None is the M2 generator)."""
    if params.information is None:
        raise ValueError("the announcement stage needs GeneratorParams.information (None: the M2 generator draws none)")
    return params.information


def _lead_laws(info: InformationParams) -> dict[int, Law]:
    """The lead law of every channel code, or ValueError naming a channel without one (Q97)."""
    laws = {codes.CHANNELS.index(ch): law for ch, law in info.lead}
    missing = [codes.CHANNELS[c] for c, law in laws.items() if law is None]
    if missing:
        raise ValueError(f"no lead law for the channels {missing}: the laws are built from the committed records (Q97)")
    return laws


def thread_leads(channel: int, u_first: float, u_second: float, laws: Mapping[int, Law]) -> np.ndarray:
    """(6,) the leads of one thread in weeks, NaN off its channels: the decoy-bearing channel's and its second notice.

    ``channel`` is the thread's decoy-bearing channel (a proposal channel, TIES or MID), ``u_first`` its lead uniform
    (V_q), ``u_second`` the second notice's (unused for MID): a tariff's final notice has lead
    min(F^{-1}_final(u_second), the proposal's lead), never before the proposal (Q12.3); a sanction's legal
    publication F^{-1}_legal(u_second), independent (design §12 "Stream 4, the leads"). Real and shadow threads alike.
    """
    out = np.full(len(codes.CHANNELS), np.nan)
    out[channel] = laws[channel].ppf(u_first)
    second = SECOND_NOTICE.get(channel)
    if second == FINAL:
        out[FINAL] = min(laws[FINAL].ppf(u_second), float(out[channel]))
    elif second == LEGAL:
        out[LEGAL] = laws[LEGAL].ppf(u_second)
    return out


@dataclass(frozen=True)
class ShadowEvent:
    """One stored shadow event j of Y (26), the would-be event a decoy thread announces (49)."""

    event: MarkedEvent  # its marks as a stored event would carry them; ``key`` its genealogical shadow key
    channel: int  # its thread's decoy-bearing channel (codes.CHANNELS), omega's sh_channel
    lead: tuple[float, ...]  # (6,) lead per channel of codes.CHANNELS in weeks, NaN off the thread's channels
    u_decoy: float  # U_j, the thinning uniform of (49): U_c k*/k_y < 1 (``type_odds``), omega's U_decoy


def shadow_odds(info: InformationParams) -> dict[int, float]:
    """phi-bar_ch/(1 - phi-bar_ch) of every decoy-bearing channel code (49): its shadow threads per real thread.

    The odds are those of ``information.decoys.show_ratio``; ``type_odds`` combines them per announced type.

    Raises:
        ValueError: if a decoy-bearing channel has no phi-bar or one outside (0, 1).

    """
    bars = dict(info.phi_bar)
    out = {}
    for c in DECOY_CODES:
        phi_bar = bars.get(codes.CHANNELS[c])
        if phi_bar is None or not 0.0 < phi_bar < 1.0:
            raise ValueError(f"phi-bar of {codes.CHANNELS[c]} must lie in (0, 1) (49), got {phi_bar!r}")
        out[c] = phi_bar / (1.0 - phi_bar)
    return out


def type_odds(info: InformationParams) -> dict[int, float]:
    """The shadow threads per real event of each announced type (49): k_y, whose largest is k* (``shadow_cascade``).

    A tariff's is its proposal channels' odds weighted by the informal share p, k_T = (1 - p) k_formal + p k_informal,
    so both proposal channels of a tariff shadow share one retention threshold (``sample_shadows``); a sanction's is
    the TIES threat's, a militarised closure's the MID threat's (``shadow_odds``).
    """
    odds, p = shadow_odds(info), info.informal_share
    return {
        TARIFF: (1.0 - p) * odds[FORMAL] + p * odds[INFORMAL],
        SANCTION: odds[TIES],
        MILITARISED_CLOSURE: odds[MID],
    }


@dataclass(frozen=True)
class ShadowCascade:
    """The shadow cluster process of one episode (Q101): (30) at k* times the real immigrant rates, on stream 7."""

    raw: tuple[RawEvent, ...]  # every shadow, burn-in and no-op included, by (onset, key); ``parent`` indexes raw
    root: tuple[int, ...]  # per shadow, the index in raw of its cluster's immigrant (its own for an immigrant)
    scale: float  # k* = the largest shadow threads per real event of a type (``type_odds``)


def shadow_cascade(
    inst: Instance,
    params: GeneratorParams,
    regimes: RegimePaths,
    entropy: int,
    episode: int,
    *,
    rules: TypeRules | None = None,
) -> ShadowCascade:
    """The shadow events of (49) as Q101 amends it: the cluster representation of (30), keyed on stream 7.

    ``hawkes.sample_cluster``'s algorithm and key layout on stream 7 with every immigrant rate of
    ``hawkes.immigrant_rates`` times k*: immigrants of (block B, region m, week t), t = -B_burn + 1 .. T, number
    Poisson(k* rate) by inversion of the uniform of key (SH_COUNT, B, m, t + B_burn) (a week of rate 0 draws no key);
    the rank-r immigrant has key (B, m, t + B_burn, r) and onset t - 1 + u, u of key (SH_ONSET, B, m, t + B_burn, r); a
    shadow of key k and onset s has, per child block B' and region m', Poisson(Gamma^{B'B}_{m'm}) children by inversion
    of uniform B' x R + m' of the 2R uniforms of key (SH_OFFSPRING, *k); child r has key (*k, B', m', r) and onset
    s + d, d = -ln(1 - u) / beta, u of key (SH_DELAY, *k, B', m', r); children with onset >= T are not kept. Only
    shadows excite shadows (Q101): no real event enters, and none of these enters the real genealogy. Raising gamma
    keeps every shadow with its key and onset (V2), as for the real events.

    Raises:
        ValueError: if ``params.information`` is None, a phi-bar is not in (0, 1), spr(Gamma) >= 1 (32), or the
            regime paths do not match the params and instance; an immigrant rate beyond the exact range of the Poisson
            inversion (``laws.poisson_ppf``).

    """
    scale = max(type_odds(_information(params)).values())
    G = branching_matrix(inst, params)
    rho = spectral_radius(G)
    if not rho < 1.0:
        raise ValueError(f"spr(Gamma) = {rho:.6f} >= 1: the process must be subcritical (32)")
    beta_inv = float(params.hawkes.beta_inv)
    rates = immigrant_rates(inst, params, regimes, rules=rules) * scale
    R, B, T = len(inst.regions), params.burn_in, inst.T
    draw = KeyedStream(entropy, episode, codes.STREAM_SHADOW)
    onsets: list[float] = []
    keys: list[tuple[int, ...]] = []
    streams: list[int] = []
    parents: list[int] = []  # positions in generation order
    roots: list[int] = []  # positions in generation order

    for s in np.flatnonzero(rates.any(axis=1)):  # immigrants: keys (SH_COUNT, ...) and (SH_ONSET, ...)
        b, m = divmod(int(s), R)
        for wk in np.flatnonzero(rates[s] > 0.0):
            wk = int(wk)
            for r in range(poisson_ppf(draw.uniform((SH_COUNT, b, m, wk)), float(rates[s, wk]))):
                roots.append(len(keys))
                onsets.append(onset_in_week(wk - B, draw.uniform((SH_ONSET, b, m, wk, r))))
                keys.append((b, m, wk, r))
                streams.append(int(s))
                parents.append(-1)

    i = 0
    while i < len(keys):  # offspring, breadth first: keys (SH_OFFSPRING, ...) and (SH_DELAY, ...)
        column = G[:, streams[i]]
        kids = np.flatnonzero(column)
        if kids.size:
            key, onset = keys[i], onsets[i]
            u = draw.uniforms((SH_OFFSPRING, *key), 2 * R)
            for s2 in kids:
                s2 = int(s2)
                b2, m2 = divmod(s2, R)
                for r in range(poisson_ppf(float(u[s2]), float(column[s2]))):
                    child = onset + (-math.log1p(-draw.uniform((SH_DELAY, *key, b2, m2, r)))) * beta_inv
                    if child < T:
                        roots.append(roots[i])
                        onsets.append(child)
                        keys.append((*key, b2, m2, r))
                        streams.append(s2)
                        parents.append(i)
        i += 1

    order = sorted(range(len(keys)), key=lambda j: (onsets[j], keys[j]))
    pos = {j: k for k, j in enumerate(order)}
    raw = tuple(
        RawEvent(keys[j], streams[j] // R, streams[j] % R, onsets[j], pos[parents[j]] if parents[j] >= 0 else -1)
        for j in order
    )
    return ShadowCascade(raw=raw, root=tuple(pos[roots[j]] for j in order), scale=scale)


def sample_leads(
    inst: Instance, params: GeneratorParams, stored: Sequence[MarkedEvent], entropy: int, episode: int
) -> tuple[np.ndarray, np.ndarray]:
    """(ev_lead (n, 6), V_lead (n,)) of the stored events, stream 4 (module docstring; (49)).

    Raises:
        ValueError: if ``params.information`` is None, or a channel of an announced type has no lead law yet (the
            laws are built from the committed records, Q97).

    """
    info = _information(params)
    laws = _lead_laws(info)
    n = len(stored)
    ev_lead, V_lead = np.full((n, len(codes.CHANNELS)), np.nan), np.full(n, np.nan)
    draw = None
    for i, ev in enumerate(stored):
        if ev.type not in ANNOUNCED_TYPES:
            continue
        if draw is None:  # E_split and n are checked at the first keyed draw (27)
            draw = KeyedStream(entropy, episode, codes.STREAM_LEAD)
        key = check_key(ev.key)
        if ev.type == TARIFF:
            u = draw.uniforms((*key, FORMAL), 2)
            channel = INFORMAL if float(u[0]) < info.informal_share else FORMAL
            v, second = float(u[1]), draw.uniform((*key, FINAL))
        elif ev.type == SANCTION:
            channel, v, second = TIES, draw.uniform((*key, TIES)), draw.uniform((*key, LEGAL))
        else:
            channel, v, second = MID, draw.uniform((*key, MID)), _NAN
        ev_lead[i] = thread_leads(channel, v, second, laws)
        V_lead[i] = v
    return ev_lead, V_lead


def _shadow(
    params: GeneratorParams,
    regimes: RegimePaths,
    rules: TypeRules,
    laws: Mapping[int, Law],
    ev: RawEvent,
    col: int,
    ty: int,
    target: Target | None,
    channel: int,
    u: np.ndarray,
    u_decoy: float,
) -> ShadowEvent:
    """The decoy thread of one typed shadow ``ev`` from its marks vector ``u`` (design §12 "Stream 7, the shadows").

    The target by a real event's rule (a closure immigrant's part fixes its chokepoint; else ``TypeRules.candidates``
    of its region, §2.4 "Targets"), a tariff's rate by its region's tension state in the onset column ``col`` (Q32),
    the leads of ``thread_leads``, and a duration and severity from the type's laws (stored, never acted on).
    """
    name = codes.EVENT_TYPES[ty]
    if target is None:
        cands = rules.candidates(ty, ev.region)
        target = cands[categorical_ppf(float(u[U_TARGET]), [w for _, w in cands])][0]
    rate = _NAN
    if ty == TARIFF:
        z_p = int(regimes.z_p[ev.region, col])
        rate = Law("constant", (params.laws.tariff_rate_by_tension[z_p],)).ppf(float(u[U_RATE]))
    dur_law = dict(params.laws.duration)[name]
    u_dur = u[U_DURATION : U_DURATION + dur_law.uniforms]
    lead = thread_leads(channel, float(u[U_LEAD]), float(u[U_SECOND]), laws)
    event = MarkedEvent(
        key=ev.key,
        type=ty,
        block=ev.block,
        region=ev.region,
        counterpart=target.counterpart if ty == TARIFF else -1,
        target_kind=int(target.kind),
        target=int(target.index),
        commodity=int(target.commodity),
        onset=float(ev.onset),
        duration=float(dur_law.ppf(u_dur)),
        severity=float(dict(params.laws.severity)[name].ppf(float(u[U_SEVERITY]))),
        rate=float(rate),
        persistent=bool(dur_law.is_persistent(u_dur)),
    )
    return ShadowEvent(event=event, channel=channel, lead=tuple(float(x) for x in lead), u_decoy=float(u_decoy))


def sample_shadows(
    inst: Instance,
    params: GeneratorParams,
    regimes: RegimePaths,
    entropy: int,
    episode: int,
    rules: TypeRules | None = None,
) -> tuple[ShadowEvent, ...]:
    """The decoy threads Y of (26) by (49) as Q101 amends it, on stream 7 (module docstring), ordered by (effect, key).

    Of ``shadow_cascade``'s shadows with effect instant in [0, T) (a thread whose effect is before 0 is not shown, §11
    row 47), each is typed by ``events.type_parts`` at the type uniform of its marks key (SH_MARKS, *key). One of an
    announced type y has U_decoy = U_c k* / k_y (``type_odds``), U_c the uniform of key (SH_CLUSTER, *its cluster's
    immigrant key), and is stored when U_decoy < 1 (a whole cluster's members of one type together); its channel is a
    sanction's TIES threat, a militarised closure's MID threat, and a tariff's proposal informal iff its U_INFORMAL
    uniform < p k_informal / k_T (p the informal share), so each channel ch keeps phi-bar_ch/(1 - phi-bar_ch) shadow
    threads per real one in expectation, and ``decoys.shown_mask`` shows it iff U_decoy < ``show_ratio``. ``rules``
    is the episode's ``targets.TypeRules`` (``EventSample.rules``, built once per episode), else built here.

    Raises:
        ValueError: if ``params.information`` is None, a phi-bar is not in (0, 1), or ``shadow_cascade`` raises.

    """
    info = _information(params)
    laws = _lead_laws(info)
    k_type = type_odds(info)
    informal = info.informal_share * shadow_odds(info)[INFORMAL] / k_type[TARIFF]  # P(informal | tariff decoy)
    rules = TypeRules(inst, params, regimes) if rules is None else rules.check(inst, params, regimes)
    cascade = shadow_cascade(inst, params, regimes, entropy, episode, rules=rules)
    draw = KeyedStream(entropy, episode, codes.STREAM_SHADOW)
    channel_of = {SANCTION: TIES, MILITARISED_CLOSURE: MID}
    cluster: dict[int, float] = {}  # U_c by the cluster's immigrant (its index in cascade.raw)
    out: list[ShadowEvent] = []
    for i, ev in enumerate(cascade.raw):
        if not 0.0 <= ev.onset < inst.T:
            continue
        u = draw.uniforms((SH_MARKS, *ev.key), SHADOW_UNIFORMS)
        col, parts = _events.type_parts(rules, regimes, ev, params.burn_in)
        ty, target, _ = parts[categorical_ppf(float(u[U_TYPE]), [p[2] for p in parts])]
        if ty not in ANNOUNCED_TYPES:  # latent: a no-op or a type with no announcement channel
            continue
        root = cascade.root[i]
        if root not in cluster:
            cluster[root] = draw.uniform((SH_CLUSTER, *cascade.raw[root].key))
        u_decoy = cluster[root] * (cascade.scale / k_type[ty])
        if not u_decoy < 1.0:  # shown at no phi <= phi-bar: latent
            continue
        if ty == TARIFF:
            channel = INFORMAL if float(u[U_INFORMAL]) < informal else FORMAL
        else:
            channel = channel_of[ty]
        out.append(_shadow(params, regimes, rules, laws, ev, col, ty, target, channel, u, u_decoy))
    out.sort(key=lambda s: (s.event.onset, s.event.key))
    return tuple(out)


def sample_blackouts(
    inst: Instance, params: GeneratorParams, entropy: int, episode: int
) -> tuple[np.ndarray, np.ndarray]:
    """(blk_start, blk_end) int32, one spell per kind in ``codes.BLACKOUT_SPELLS`` order, stream 11.

    The uniform of key (spell kind code) picks the onset among the admissible weeks, 1..T - length + 1 (1..T to the
    end), by categorical inversion with equal weights; ``blk_end`` = onset + length, T + 1 to the end (design §12
    "Stream 11, the blackout spells").

    Raises:
        ValueError: if ``params.information`` is None or a spell does not fit the episode.

    """
    info = _information(params)
    lengths = dict(info.blackout_lengths)
    if tuple(lengths) != codes.BLACKOUT_SPELLS or info.blackout_onset != "uniform":
        raise ValueError(
            f"blackout spells: one length per kind of {codes.BLACKOUT_SPELLS} and the 'uniform' onset, got"
            f" {info.blackout_lengths!r}, {info.blackout_onset!r}"
        )
    T = inst.T
    draw = KeyedStream(entropy, episode, codes.STREAM_BLACKOUT)
    starts, ends = [], []
    for k, kind in enumerate(codes.BLACKOUT_SPELLS):
        length = lengths[kind]
        n = T if length is None else T - int(length) + 1
        if length is not None and (int(length) < 1 or n < 1):
            raise ValueError(f"blackout spell {kind!r} of {length} weeks does not fit an episode of {T} weeks")
        onset = 1 + categorical_ppf(draw.uniform((k,)), np.ones(n))
        starts.append(onset)
        ends.append(T + 1 if length is None else onset + int(length))
    return np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32)


@dataclass(frozen=True)
class AnnouncementSample:
    """Everything the announcement stage drew for one episode, before assembly."""

    ev_lead: np.ndarray  # (n, 6) float64, aligned with EventSample.stored
    V_lead: np.ndarray  # (n,) float64
    shadows: tuple[ShadowEvent, ...]
    blk_start: np.ndarray  # (spells,) int32
    blk_end: np.ndarray  # (spells,) int32


def sample_announcements(
    inst: Instance, params: GeneratorParams, sample: "EventSample", entropy: int, episode: int
) -> AnnouncementSample:
    """Leads, shadows and blackout spells of one episode (streams 4, 7, 11).

    The leads, shadows and blackout spells of one episode's ``EventSample`` (streams 4, 7, 11), the one entry
    ``sampler.sample_omega`` calls when ``params.information`` is set.

    Raises:
        ValueError: if the sample's provenance is not (``entropy``, ``episode``, ``generator_id(params, inst)``).

    """
    drawn, asked = (sample.entropy, sample.episode, sample.generator_id), (entropy, episode, generator_id(params, inst))
    if drawn != asked:
        raise ValueError(f"the sample was drawn for (E_split, episode, generator) {drawn}, not {asked} (§4.1)")
    ev_lead, V_lead = sample_leads(inst, params, sample.stored, entropy, episode)
    rules = sample.rules  # the episode's own, unless the sample came from another process (then built again)
    rules = rules if rules is not None and rules.fits(inst, params, sample.regimes) else None
    shadows = sample_shadows(inst, params, sample.regimes, entropy, episode, rules=rules)  # no real event enters
    blk_start, blk_end = sample_blackouts(inst, params, entropy, episode)
    return AnnouncementSample(ev_lead, V_lead, shadows, blk_start, blk_end)


def announcement_arrays(inst: Instance, ann: AnnouncementSample, burn_in: int) -> dict[str, np.ndarray]:
    """The announcement arrays ``assemble_omega`` writes for a generated omega.

    ``assemble_omega``'s ``announcements``: ev_lead, V_lead, the sh_* group through the one event writer
    (``events.event_arrays(..., prefix='sh')``, SIMP-M2R2-01) with sh_lead, sh_channel, U_decoy, blk_start, blk_end.
    """
    shadows = ann.shadows
    sh = _events.event_arrays(inst, [s.event for s in shadows], burn_in, prefix="sh")
    sh["sh_lead"] = np.array([s.lead for s in shadows], dtype=np.float64).reshape(len(shadows), len(codes.CHANNELS))
    return {
        "ev_lead": np.asarray(ann.ev_lead, dtype=np.float64),
        "V_lead": np.asarray(ann.V_lead, dtype=np.float64),
        **sh,
        "sh_channel": np.array([s.channel for s in shadows], dtype=np.int8),
        "U_decoy": np.array([s.u_decoy for s in shadows], dtype=np.float64),
        "blk_start": np.asarray(ann.blk_start, dtype=np.int32),
        "blk_end": np.asarray(ann.blk_end, dtype=np.int32),
    }


def information_meta(params: GeneratorParams, xi: bool = True) -> InformationMeta:
    """The content of omega's ``meta_information_params``: P^yr, phi-bar by channel name, and ``xi``.

    Raises:
        ValueError: if ``params.information`` is None.

    """
    info = _information(params)
    return InformationMeta(P_yr=tuple(tuple(r) for r in params.regime.P_yr), phi_bar=tuple(info.phi_bar), xi=xi)

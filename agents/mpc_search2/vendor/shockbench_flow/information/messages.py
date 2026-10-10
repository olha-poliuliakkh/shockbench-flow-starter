"""Announcements, their threads and the §9.2 messages and pending_prohibitions fields (design §5.1, §5.3; (1), (49)).

Trusted side: every real announcement comes from omega's ``ev_lead`` (the lead per channel of each stored event, NaN
where the event is not announced there) and every shadow one from the ``sh_*`` group, which the decoy stream fills
(``disruption.announce``); ``decoys.shown_mask`` thins the shadows by (49). A *thread* is one event's announcements:
a tariff's proposal (formal or informal) and final notice, a sanction's TIES threat and legal publication, a
militarised closure's MID threat (symmetric channel sets, Q97, frozen in ``omega.codes.TYPE_CHANNELS``), plus a shown
shadow's withdrawal at its effect time. Readings of design §12 (M3 rows):

- The instant of an announcement is a = t_on - l (weeks); it is shown from week w(a) = ceil(a) + 1, that is at every
  instant t - 1 >= a (1). A shadow's withdrawal has the instant t_j, its shadow effect time. A real event is never
  withdrawn, and a shadow announcement with lead <= 0 is never shown (49).
- ``msg_id`` is the thread's rank by its first shown announcement instant, over the real and the shown shadow threads
  alike, ties broken by public fields only (channel, region, target kind, target, k): never by row, source or key,
  and never over unshown shadows (gaps would reveal the thinning). Every message of a thread, its withdrawal included,
  carries its msg_id; ids shown at week t never depend on later messages.
- ``messages`` is cumulative: every shown message with instant <= t - 1, from the episode start; an announcement made
  in a blackout spell is shown after the spell. Threads whose effect instant is < 0 are not shown (no pre-episode
  history, §11 row 47). ``stated_effective_week`` is w(t_on) for proposals, final notices and publications, null for
  TIES and MID threats, which state no date. No rate field (the signed-off schema has none).
- A tariff's final notice never comes before its proposal (Q12.3's order, a proposal then a final notice): its lead
  is min(its own draw, the proposal's lead), real and shadow alike (``disruption.announce``); a sanction thread's two
  leads are independent draws.
- ``pending_prohibitions`` lists the (edge, k) pairs of every shown ``sanction_legal`` publication (the sanction's
  final notice) announced by t - 1 whose effective week w(t_on) > t, from the thread's own event row by the marks rule
  for that event alone (``marks.graph_marks`` of one event, V3) less the instance's ``prohibitions_at_reset`` (Z_0,
  which ``graph_marks`` includes); ``pending_at`` takes every shown message and keeps the legal publications itself,
  nothing from a TIES threat or a tariff, and nothing read from ``ev_*`` directly, so a pending entry never tells a
  real thread from a decoy. Clairvoyant's exact list is ``view.clairvoyant_pending``, not this.

``announced_week`` of an announcement made before the instant 0 (a < 0, effect in [0, T)) is w(a) unclipped, real and
shadow alike (<= 0 for a <= -1).
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from shockbench_flow.information.decoys import shown_mask
from shockbench_flow.information.theta import Theta, phi_by_channel
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import SANCTION, Event, MarkParams, graph_marks, read_events
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega


PROPOSAL, FINAL_NOTICE, THREAT, PUBLICATION, WITHDRAWAL = (
    codes.MESSAGE_KINDS.index(n) for n in ("proposal", "final_notice", "threat", "publication", "withdrawal")
)
FORMAL, INFORMAL, FINAL, LEGAL, TIES, MID = (
    codes.CHANNELS.index(n)
    for n in ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
)
KIND_OF_CHANNEL = {
    FORMAL: PROPOSAL,
    INFORMAL: PROPOSAL,
    FINAL: FINAL_NOTICE,
    LEGAL: PUBLICATION,
    TIES: THREAT,
    MID: THREAT,
}
DATED_KINDS = frozenset({PROPOSAL, FINAL_NOTICE, PUBLICATION})  # the kinds that state an effective week
DECOY_CODES = frozenset(codes.CHANNELS.index(n) for n in codes.DECOY_CHANNELS)
# the channel codes each event type may carry a lead on (codes.TYPE_CHANNELS, 'proposal' formal or informal)
_TYPE_CODES = {
    codes.EVENT_TYPES.index(ty): frozenset(
        c for ch in chans for c in ((FORMAL, INFORMAL) if ch == "proposal" else (codes.CHANNELS.index(ch),))
    )
    for ty, chans in codes.TYPE_CHANNELS
}


def week_of(instant: float) -> int:
    """w(a) = ceil(a) + 1: the first week whose observation, at the instant t - 1, is at or after ``a`` (1)."""
    return math.ceil(instant) + 1


def _thread(a: "Announcement") -> tuple[bool, int]:
    """The thread of an announcement: its group (real or shadow) and its event's row there (trusted)."""
    return a.decoy, a.event.row


@dataclass(frozen=True)
class Announcement:
    """One §9.2 message on the trusted side (the fields the wire never carries are marked trusted).

    ``kind`` and ``channel`` are codes of ``codes.MESSAGE_KINDS`` and ``codes.CHANNELS``; a withdrawal carries its
    thread's decoy-bearing channel (tariff_formal or tariff_informal, ties_threat, mid_threat).
    """

    thread: int  # msg_id: the thread's rank by its first shown announcement instant (-1 before ``assign_threads``)
    channel: int  # codes.CHANNELS index
    kind: int  # codes.MESSAGE_KINDS index
    region: int  # m_q of the thread's event, by the marks.read_events rule
    target_kind: int  # codes.TARGET_KINDS index
    target: int  # node, edge or region index (a chokepoint by its node index)
    k: int | None  # the event commodity, None for every commodity
    instant: float  # a = t_on - l in weeks (a withdrawal: t_j); shown from w(a) = ceil(a) + 1 (1)
    effect: float  # t_on of the thread's event (a shadow's effect time t_j)
    stated_effective_week: int | None  # w(t_on) for proposals, final notices and publications; None for threats
    decoy: bool  # trusted: a shadow thread's message; never serialised
    event: Event  # trusted: the thread's event row as marks reads it (ev_* or sh_*), for pending_at's pairs


def announcements(inst: Instance, omega: Omega, events: tuple[Event, ...] | None = None) -> tuple[Announcement, ...]:
    """Every announcement omega holds, real and shadow, before thinning (49).

    Every announcement omega holds, before thinning: the real ones from ``ev_lead`` and the shadow ones from the
    ``sh_*`` group ((49); only events and shadows with effect in [0, T)), each with ``thread`` -1.

    Shadow rows are decoded by ``marks.read_events(inst, omega, prefix='sh')`` (the one event decoder,
    SIMP-M2R2-01), so a shadow and a real event with the same marks give the same fields. An injected omega (every lead
    NaN, no shadows) gives (). ``events`` are omega's ``ev_*`` events when the caller has decoded them already
    (``read_events(inst, omega)``, as ``view.build_view`` does); None decodes them here.

    Raises:
        ValueError: on a lead on a channel the event's type does not announce (``codes.TYPE_CHANNELS``, which
            ``InformationParams.channels`` equals), a shadow channel that bears no decoys, or ``sh_*`` rows
            ``read_events`` refuses.

    """
    out: list[Announcement] = []
    real = read_events(inst, omega) if events is None else events
    for prefix, decoy in (("ev", False), ("sh", True)):
        group = read_events(inst, omega, prefix) if decoy else real
        if not group:
            continue
        leads = np.asarray(omega[f"{prefix}_lead"], dtype=np.float64)
        channels = np.asarray(omega["sh_channel"]) if decoy else None
        for ev in group:
            row = leads[ev.row]
            with_lead = [c for c in range(len(codes.CHANNELS)) if not math.isnan(row[c])]
            allowed = _TYPE_CODES.get(ev.type, frozenset())
            if not set(with_lead) <= allowed or {FORMAL, INFORMAL} <= set(with_lead):
                raise ValueError(
                    f"{prefix} {ev.name}: leads on {[codes.CHANNELS[c] for c in with_lead]}, which its type does not"
                    " announce (codes.TYPE_CHANNELS; one proposal channel per tariff)"
                )
            if decoy:
                ch = int(channels[ev.row])
                if ch not in DECOY_CODES or ch not in with_lead:
                    raise ValueError(f"shadow {ev.row}: channel {ch} bears no decoys here (codes.DECOY_CHANNELS)")
            if not with_lead or not 0.0 <= ev.onset < inst.T:  # unannounced, or its effect outside [0, T) (§11 row 47)
                continue
            base = dict(
                thread=-1,
                region=ev.region,
                target_kind=ev.target_kind,
                target=ev.target,
                k=None if ev.commodity < 0 else ev.commodity,
                effect=ev.onset,
                decoy=decoy,
                event=ev,
            )
            for c in with_lead:
                kind = KIND_OF_CHANNEL[c]
                stated = week_of(ev.onset) if kind in DATED_KINDS else None
                out.append(
                    Announcement(
                        channel=c, kind=kind, instant=ev.onset - float(row[c]), stated_effective_week=stated, **base
                    )
                )
            if decoy:  # the decoy is withdrawn at its shadow effect time (Q97; §11 row 71)
                out.append(
                    Announcement(channel=ch, kind=WITHDRAWAL, instant=ev.onset, stated_effective_week=None, **base)
                )
    return tuple(out)


def shown(
    anns: Sequence[Announcement], theta: Theta, omega: Omega, phi_bar: dict[str, float]
) -> tuple[Announcement, ...]:
    """The announcements shown under ``theta`` (49).

    The announcements shown under ``theta``: every real one; a shadow thread's pre-effect ones (lead > 0) iff its
    thread is shown by (49) (``decoys.shown_mask`` on omega's ``U_decoy`` and ``sh_channel``), with its withdrawal;
    none when theta has no messages feed. Threads are then numbered by ``assign_threads``.

    A shadow thread's withdrawal is shown only with at least one of its announcements (a thread never announced is
    never withdrawn); a real thread is never withdrawn (design §12 "Message threads as built").

    Raises:
        ValueError: if a phi of ``theta`` exceeds ``phi_bar`` of its channel (``decoys.show_ratio``).

    """
    if not phi_by_channel(theta):  # no messages feed
        return ()
    mask = shown_mask(omega, theta, phi_bar)
    threads: dict[tuple[bool, int], list[Announcement]] = {}
    for a in anns:
        threads.setdefault(_thread(a), []).append(a)
    keep: list[Announcement] = []
    for (decoy, row), msgs in threads.items():
        if not decoy:
            keep += msgs
        elif mask[row]:
            pre = [a for a in msgs if a.kind != WITHDRAWAL and a.instant < a.effect]  # lead > 0 only (49)
            if pre:
                keep += pre + [a for a in msgs if a.kind == WITHDRAWAL]
    return assign_threads(keep)


def assign_threads(anns: Sequence[Announcement]) -> tuple[Announcement, ...]:
    """``anns`` with ``thread`` set to the msg_id rule of the module docstring, sorted by (instant, msg_id, kind).

    Threads rank by their first instant, ties broken by the first message's public fields (channel, kind, region,
    target kind, target, k, stated effective week), then by the thread's whole public message list, then by the row
    within its group: a tie that far needs equal effect instants (a null event of the continuous laws), and a real
    thread (never withdrawn) and a shadow thread (withdrawn) never tie on the whole list (design §12).
    """

    def public(a: Announcement) -> tuple:
        return (
            a.channel,
            a.kind,
            a.region,
            a.target_kind,
            a.target,
            -1 if a.k is None else a.k,
            -1 if a.stated_effective_week is None else a.stated_effective_week,
        )

    threads: dict[tuple[bool, int], list[Announcement]] = {}
    for a in anns:
        threads.setdefault(_thread(a), []).append(a)
    for msgs in threads.values():
        msgs.sort(key=lambda a: (a.instant, a.kind))
    order = sorted(
        threads,
        key=lambda th: (
            threads[th][0].instant,
            public(threads[th][0]),
            tuple((a.instant, *public(a)) for a in threads[th]),
            th[1],
        ),
    )
    rank = {th: i for i, th in enumerate(order)}
    out = [dataclasses.replace(a, thread=rank[_thread(a)]) for a in anns]
    out.sort(key=lambda a: (a.instant, a.thread, a.kind))
    return tuple(out)


def messages_at(anns: Sequence[Announcement], t: int) -> dict | None:
    """The §9.2 ``messages`` value of week t (1).

    Every message of ``anns`` (shown, threads assigned) with instant <= t - 1, as columns {msg_id, channel (name),
    kind (name), region, target_kind (name), target, k, announced_week = w(a), stated_effective_week}; empty columns
    when none is shown yet. ``wrap`` nulls it outside the feed. The instant a <= t - 1 exactly when w(a) <= t.
    """
    now = [a for a in anns if week_of(a.instant) <= t]
    return {
        "msg_id": [int(a.thread) for a in now],
        "channel": [codes.CHANNELS[a.channel] for a in now],
        "kind": [codes.MESSAGE_KINDS[a.kind] for a in now],
        "region": [int(a.region) for a in now],
        "target_kind": [codes.TARGET_KINDS[a.target_kind] for a in now],
        "target": [int(a.target) for a in now],
        "k": [None if a.k is None else int(a.k) for a in now],
        "announced_week": [week_of(a.instant) for a in now],
        "stated_effective_week": [
            None if a.stated_effective_week is None else int(a.stated_effective_week) for a in now
        ],
    }


def pending_at(shown: Sequence[Announcement], t: int, inst: Instance, mark_params: MarkParams) -> dict:
    """The §9.2 ``pending_prohibitions`` value of week t in 1..T (``wrap`` nulls it outside the feed).

    ``shown`` is every shown message (``shown`` then ``assign_threads``); this keeps the ``sanction_legal``
    publications itself. Columns {edge, k, effective_week} of every (edge, k) pair such a publication with instant
    <= t - 1 adds to Z_t at its effective week w(t_on) > t: the pairs of the thread's own event by
    ``marks.graph_marks`` of that event alone, less ``inst.prohibitions_at_reset`` (Z_0), in (effective_week, edge, k)
    order (§5.1; Q12.4; design §12 M3 rows). It binds M4's ``mpc_det``, which reads it as prohibitions taking effect
    then.

    Raises:
        ValueError: if a ``sanction_legal`` publication's event is not a sanction or export control (a message on
            any other channel is skipped, never refused).

    """
    return pending_columns(_publications(shown, inst, mark_params), t)


def _publications(
    shown: Sequence[Announcement], inst: Instance, mark_params: MarkParams
) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    """(announced week, effective week, pairs) of every shown legal publication, each pair set computed once.

    The pairs are those ``marks.graph_marks`` of the publication's event alone marks prohibited in its effective week
    w(t_on), less Z_0; an effective week beyond T lists none (the prohibition never enters force in the episode;
    design §12 "Pending pairs as built").

    Raises:
        ValueError: as ``pending_at``.

    """
    z0 = {(int(e), int(k)) for e, k in inst.prohibitions_at_reset}
    out = []
    for a in shown:
        if a.channel != LEGAL:  # a tariff, TIES or MID message: skipped, never refused
            continue
        if a.event.type != SANCTION:
            raise ValueError(f"a sanction_legal publication of {a.event.name}, which is not a sanction (Q97 sets)")
        eff = week_of(a.effect)
        pairs: tuple[tuple[int, int], ...] = ()
        if eff <= inst.T:
            prohibited = graph_marks(inst, (a.event,), mark_params)["prohibited"][eff - 1]
            pairs = tuple((int(e), int(k)) for e, k in np.argwhere(prohibited) if (int(e), int(k)) not in z0)
        out.append((week_of(a.instant), eff, pairs))
    return tuple(out)


def pending_columns(pubs: Sequence[tuple[int, int, tuple[tuple[int, int], ...]]], t: int) -> dict:
    """The pending_prohibitions columns of week t from (shown-from week, effective week, pairs) rows.

    The rows of ``_publications`` (or of ``view.clairvoyant_pending``, each shown from week 0): every pair of a row
    announced by t - 1 (shown-from week <= t) and effective after t, in (effective_week, edge, k) order.
    """
    rows = sorted((eff, e, k) for shown_from, eff, pairs in pubs if shown_from <= t < eff for e, k in pairs)
    return {"edge": [e for _, e, _ in rows], "k": [k for _, _, k in rows], "effective_week": [w for w, _, _ in rows]}


def pending_table(shown: Sequence[Announcement], inst: Instance, mark_params: MarkParams) -> tuple[dict, ...]:
    """(T,) ``pending_at(shown, t, inst, mark_params)`` for t = 1..T, row t - 1 = week t, built once at reset.

    Each publication's pairs are computed once (one ``graph_marks`` per thread, not per week); ``view.build_view``
    stores the table.

    Raises:
        ValueError: as ``pending_at``.

    """
    pubs = _publications(shown, inst, mark_params)
    return tuple(pending_columns(pubs, t) for t in range(1, inst.T + 1))

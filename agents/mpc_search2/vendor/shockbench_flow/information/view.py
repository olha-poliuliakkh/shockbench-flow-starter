"""The reset-time information view of one episode: every feed of theta, built once (design §5.1, §5.4; (1)).

``build_view(inst, omega, theta, *, marks)`` is the one builder: pure in (instance, omega, theta), it never reads the
State and holds no simulator (§5.4 "Rules"), so the wrapper outside the environment core (``observe.wrap``) degrades
the core observation from these tables alone, at reset, every step, the D9 fallback and restore alike. One omega serves
every theta: nothing is drawn or re-sampled per regime (R1 A4; pairing R5.11). The view is immutable and shared, like
the instance and the marks, by ``Env``'s episode copies and snapshots, so ``restore`` rebuilds nothing from omega
(``Static`` included: ``dyads`` and ``theta`` travel here). The week tables of ``pending_prohibitions`` and
``closure_end`` are built here once, for every week 1..T (never per step).

Readings of design §12 (M3 rows):

- Resets without what a regime needs: a marks-only reset (no omega, the M1 fixture path) accepts prediction_free only,
  with ``demand_forecast`` null, since (48) needs omega's ``eps_parts``. A generated omega without
  ``meta_information_params`` (every M2 omega: ``sample_omega`` with ``GeneratorParams.information`` None) keeps the
  M2 observation under prediction_free and the coverage rungs, ``demand_forecast`` null, so no M2 pin moves; any
  regime with a warning or messages raises on it. An injected omega needs no meta (no conflict layer, so
  (48) has no Xi term, and no shadows): prediction_free shows (48) on it, and a regime with a warning or messages
  raises on it (no latent paths). A blackout rung raises on an omega without a spell of its kind.
- Blackout: the spell of theta's kind is omega's ``blk_start[k] <= t < blk_end[k]``, k its ``codes.BLACKOUT_SPELLS``
  code (a to_end spell has ``blk_end`` = T + 1); in those weeks the own state and ``last_week`` are kept and every
  external feed is null (``graph_now``, its ``fab`` and ``osat`` included, ``slot_mask`` and the five feeds).
- closure_end (theta.chi): one entry per closure event (militarised or weather) acting on a chokepoint at the instant
  t - 1 (t_on <= t - 1 < t_on + T_q), ``end_week`` = w(t_on + T_q), which may exceed T; never null in a named regime.
  Several closures may act on one chokepoint at once, so the list is per event, never per chokepoint.
- pending_prohibitions: clairvoyant's is exact (``clairvoyant_pending``: every future prohibition of omega's ``ev_*``
  events, announced or not); every other regime with the feed reads the shown publications (``messages.pending_at``).
- Clairvoyant: ``omega_public`` is ``wire.public_omega(omega)``, omega without ``meta_episode`` and ``meta_split``
  (trusted side only), an immutable ``Omega`` of its read-only arrays; ``Env`` hands out ``wire.omega_payload`` of it,
  built afresh at each reset and restore, as ``info['omega']`` and ``Reset.omega`` in local and dev runs only, to a
  policy that declares ``wants_omega`` (``policies.base``).
- messages: the cumulative feed of week t is a prefix of the shown messages (sorted by instant, ``assign_threads``, and
  w(a) is monotone in the instant a), so the view keeps their §9.2 columns (``messages.messages_at`` at T) and the
  length of each week's prefix; ``observe.wrap`` slices them.
"""

import bisect
import dataclasses
from dataclasses import dataclass

import numpy as np

from shockbench_flow.information import messages, wire
from shockbench_flow.information.forecast import exact_forecast, mmfe_forecast
from shockbench_flow.information.messages import Announcement, week_of
from shockbench_flow.information.theta import Theta, check_theta, phi_by_channel
from shockbench_flow.information.warning import Unit, unit_separation, unit_table, warning_scores
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import (
    MILITARISED_CLOSURE,
    SANCTION,
    WEATHER_CLOSURE,
    Event,
    MarkParams,
    WeeklyMarks,
    graph_marks,
    mark_params_from_json,
    osat_restoration_now,
    read_events,
)
from shockbench_flow.omega import codes
from shockbench_flow.omega.assembly import information_meta_from_json
from shockbench_flow.omega.container import Omega, immutable_copy


@dataclass(frozen=True, eq=False)
class InformationView:
    """Everything ``observe.wrap`` reads, built once at reset by ``build_view``; arrays read-only, never the State."""

    theta: Theta
    T: int
    instance_digest: str  # content digest of the instance the view was built for (V3)
    omega_hash: str | None  # None on a marks-only reset
    dyads: tuple[tuple[int, int], ...]  # Static.dyads rows (a, b), omega's dyad_regions; () without them
    units: tuple[Unit, ...]  # the warning's signal units in X row order; () without a warning feed
    scores: np.ndarray | None  # (T, U) S of (45), row t - 1 = week t; None without a warning feed
    forecast: np.ndarray | None  # (T, D, 8) (48), or the exact d under clairvoyant; None where the forecast is null
    messages: tuple[Announcement, ...]  # shown under theta, threads assigned (``messages.assign_threads``); ()
    pending: tuple[dict, ...] | None  # (T,) the pending_prohibitions value of week t at row t - 1; None: no feed
    closure_end: tuple[dict, ...] | None  # (T,) the closure_end value of week t at row t - 1; None: chi False
    blackout_weeks: frozenset[int]  # the weeks of theta's spell; empty outside the blackout rung
    omega_public: Omega | None  # clairvoyant only: wire.public_omega(omega), handed out as wire.omega_payload of it
    # (T, O) R^osat_i at the instant t - 1 (``marks.osat_restoration_now`` of omega's events), from which ``wrap``
    # fills ``graph_now.osat``'s R and thr_eff = thr_i R (Q97; ``marks.osat_throughput``); None on a marks-only reset
    osat_now: np.ndarray | None = None
    # the §9.2 ``messages`` columns of every message shown by week T (``messages.messages_at(messages, T)``), as
    # (field, column) pairs, and (T,) the number of them shown by week t at row t - 1: week t's feed is that prefix
    message_columns: tuple[tuple[str, tuple], ...] = ()
    message_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the arrays (private read-only copies over immutable buffers), as the marks do (V3 by content)."""
        for name in ("scores", "forecast", "osat_now"):
            a = getattr(self, name)
            if a is not None:
                object.__setattr__(self, name, immutable_copy(np.asarray(a)))

    def __reduce__(self):
        """Pickle as the field values; the rebuild goes through the constructor, so restored arrays stay read-only."""
        return (type(self), tuple(getattr(self, f.name) for f in dataclasses.fields(self)))

    @property
    def has_messages(self) -> bool:
        """Whether theta shows the ``messages`` feed (its phi are set, (49))."""
        return bool(phi_by_channel(self.theta))


def _needs_omega(theta: Theta) -> list[str]:
    """What of omega ``theta`` reads beyond the marks (a marks-only reset serves none of it)."""
    return [
        what
        for what, on in (
            ("a warning (45)", theta.L is not None),
            ("messages (49)", bool(phi_by_channel(theta))),
            ("pending_prohibitions", theta.pending),
            ("closure_end (chi)", theta.chi),
            ("the exact forecast", theta.forecast == "exact"),
            ("omega at reset", theta.omega_at_reset),
            ("a blackout spell", theta.blackout is not None),
        )
        if on
    ]


def _closures(events: tuple[Event, ...]) -> tuple[tuple[int, float, float], ...]:
    """(chokepoint node, t_on, t_on + T_q) of every closure event (militarised or weather), in event row order."""
    return tuple((q.target, q.onset, q.end) for q in events if q.type in (MILITARISED_CLOSURE, WEATHER_CLOSURE))


def build_view(inst: Instance, omega: Omega | None, theta: Theta, *, marks: WeeklyMarks) -> InformationView:
    """Build the information view of ``theta`` for one episode, once, at reset.

    The view of ``theta`` on (``inst``, ``omega``): scores (45), forecast (48) (its Xi term on when
    ``theta.forecast_shock`` and the meta's ``xi`` are both true), shown messages (49), the week tables of
    ``pending_prohibitions`` (``messages.pending_table``, or ``clairvoyant_pending``) and ``closure_end``
    (``closure_end_table``), blackout weeks, dyads and the clairvoyant payload, from omega's arrays and its
    ``meta_information_params`` (``omega.assembly.information_meta_from_json``); the rules for an omega without what a
    regime needs are the module docstring's.

    ``marks`` are the episode's (``compute_marks(inst, omega)``, or the marks of a marks-only reset); their digest,
    instance and omega hashes must agree with ``inst`` and ``omega`` (V3); ``mark_params`` for the pending rule are
    omega's ``meta_mark_params``. The exact pending list is the one of a theta that sees omega whole
    (``theta.omega_at_reset``, clairvoyant); the forecast of a theta with the 'exact' forecast is omega's realised d.

    Raises:
        ValueError: when theta needs what omega lacks (latent paths, shadows, ``meta_information_params``, a spell of
            the rung's kind) or there is no omega (a marks-only reset under any regime but prediction_free); when
            ``marks`` belong to another instance or omega; or a phi of theta above omega's phi-bar.

    """
    check_theta(theta)
    inst = inst.at_digest(marks.instance_digest)  # the marks' rung: its warm start block (§2.3; M5-O37 (b))
    if marks.instance_hash != inst.hash or marks.T != inst.T or marks.instance_digest != inst.content_digest:
        raise ValueError("the marks belong to another instance than the view's (V3)")
    T = inst.T
    common = dict(theta=theta, T=T, instance_digest=inst.content_digest)
    if omega is None:
        need = _needs_omega(theta)
        if need:
            raise ValueError(f"a reset from marks alone serves no feed that reads omega: {theta.name!r} needs {need}")
        return InformationView(
            **common,
            omega_hash=None,
            dyads=(),
            units=(),
            scores=None,
            forecast=None,  # (48) needs omega's eps_parts (design §12 "Resets without what a regime needs")
            messages=(),
            pending=None,
            closure_end=None,
            blackout_weeks=frozenset(),
            omega_public=None,
        )
    if marks.omega_hash != omega.hash:
        raise ValueError("the marks are not omega's (V3)")
    meta = None
    if "meta_information_params" in omega:
        meta = information_meta_from_json(str(omega["meta_information_params"]))
    phis = phi_by_channel(theta)
    if (theta.L is not None or phis) and meta is None:
        raise ValueError(
            f"{theta.name!r} shows a warning or messages, which need a generated omega with meta_information_params "
            "(an injected omega, and every M2 omega, has none; design §12 'Resets without what a regime needs')"
        )
    dyads = ()
    if "dyad_regions" in omega:
        dyads = tuple((int(a), int(b)) for a, b in np.asarray(omega["dyad_regions"]).tolist())
    units, scores = (), None
    if theta.L is not None:  # (45)
        units = unit_table(inst, omega)
        a = unit_separation(theta, units)
        scores = warning_scores(np.asarray(omega["X"]), np.asarray(omega["W"]), a, theta.L, omega.burn_in, T)
    if theta.forecast == "exact":  # clairvoyant: the realised series
        fc = exact_forecast(inst, np.asarray(omega["d"]))
    elif omega.generated and meta is None:  # an M2 omega: no Pi^c for the Xi term, so no forecast (today's view)
        fc = None
    else:  # (48)
        xi = bool(theta.forecast_shock and meta is not None and meta.xi)
        z_c = np.asarray(omega["z_c"]) if omega.generated else None
        P_yr = meta.P_yr if meta is not None else None
        fc = mmfe_forecast(inst, np.asarray(omega["eps_parts"]), z_c, omega.burn_in, P_yr, xi)
    shown: tuple[Announcement, ...] = ()
    if phis:  # (49)
        phi_bar = dict(meta.phi_bar)
        for ch, phi in phis.items():
            if not phi <= phi_bar.get(ch, -1.0):
                raise ValueError(
                    f"phi of {ch} ({phi}) exceeds omega's phi-bar {phi_bar.get(ch)}: a larger phi-bar "
                    "redraws the shadows (design §12 M3 rows)"
                )
    events = read_events(inst, omega)  # omega's ev_* events, decoded once for the messages and the tables below
    if phis:
        anns = messages.announcements(inst, omega, events)
        shown = messages.shown(anns, theta, omega, phi_bar)  # threads assigned (``messages.assign_threads``)
    mark_params = mark_params_from_json(str(omega["meta_mark_params"]))
    osat_now = osat_restoration_now(inst, events, mark_params)  # graph_now.osat at the instant t - 1 (Q97)
    pending = None
    if theta.pending:
        if theta.omega_at_reset:  # clairvoyant: exact, announced or not
            pending = clairvoyant_pending(inst, events, mark_params)
        else:  # the shown legal publications only
            pending = tuple(messages.pending_table(shown, inst, mark_params))
    return InformationView(
        **common,
        omega_hash=omega.hash,
        dyads=dyads,
        units=units,
        scores=scores,
        forecast=fc,
        messages=tuple(shown),
        pending=pending,
        closure_end=closure_end_table(_closures(events), T) if theta.chi else None,
        blackout_weeks=blackout_weeks(omega, theta.blackout, T) if theta.blackout is not None else frozenset(),
        omega_public=wire.public_omega(omega) if theta.omega_at_reset else None,
        osat_now=osat_now,
        **_message_feed(shown, T),
    )


def _message_feed(shown: tuple[Announcement, ...], T: int) -> dict:
    """The view's ``message_columns`` and ``message_counts`` of the shown messages (sorted by instant).

    Week t's ``messages.messages_at(shown, t)`` is the first ``message_counts[t - 1]`` entries of every column:
    ``assign_threads`` sorts the messages by instant and w(a) = ceil(a) + 1 is monotone in it, so the messages with
    w(a) <= t are a prefix.
    """
    weeks = [week_of(a.instant) for a in shown]
    cols = messages.messages_at(shown, T)
    return {
        "message_columns": tuple((name, tuple(column)) for name, column in cols.items()),
        "message_counts": tuple(bisect.bisect_right(weeks, t) for t in range(1, T + 1)),
    }


def static_dyads(dyads: tuple[tuple[int, int], ...]) -> dict:
    """The §9.2 ``Static.dyads`` table {"a": [int], "b": [int]}: region indices of each dyad, in z_dyad row order."""
    rows = [(int(a), int(b)) for a, b in dyads]
    return {"a": [a for a, _ in rows], "b": [b for _, b in rows]}


def blackout_weeks(omega: Omega, spell: str, T: int) -> frozenset[int]:
    """The weeks t in 1..T of omega's spell of kind ``spell`` (``codes.BLACKOUT_SPELLS``): blk_start <= t < blk_end.

    Raises:
        ValueError: if omega stores no spell of that kind (every injected omega), or its spell arrays are not one
            spell per kind in kind order.

    """
    if spell not in codes.BLACKOUT_SPELLS:
        raise ValueError(f"unknown blackout spell kind {spell!r}: one of {codes.BLACKOUT_SPELLS}")
    start = np.asarray(omega["blk_start"]) if "blk_start" in omega else np.zeros(0)
    end = np.asarray(omega["blk_end"]) if "blk_end" in omega else np.zeros(0)
    if start.size == 0 and end.size == 0:
        raise ValueError(f"omega holds no blackout spell, so no spell of kind {spell!r} (the blackout rung needs one)")
    n = len(codes.BLACKOUT_SPELLS)
    if start.shape != (n,) or end.shape != (n,):
        raise ValueError(f"omega's blk_start and blk_end must hold one spell per kind, {n}, got {start.shape}")
    k = codes.BLACKOUT_SPELLS.index(spell)
    return frozenset(t for t in range(1, T + 1) if int(start[k]) <= t < int(end[k]))


def closure_end_at(closures: tuple[tuple[int, float, float], ...], t: int) -> dict:
    """The §9.2 ``closure_end`` value of week t in 1..T (chi); ``wrap`` nulls it in the final observation.

    ``closures`` are (chokepoint node, t_on, t_on + T_q) of every closure event (militarised or weather) of omega's
    ``ev_*``, in event row order. Columns {chokepoint: [node], end_week: [w(t_on + T_q)]} of every closure acting at
    the instant t - 1 (t_on <= t - 1 < t_on + T_q), one entry per event in that order (end_week never null).
    """
    now = t - 1.0
    rows = [(int(c), week_of(end)) for c, on, end in closures if on <= now < end]
    return {"chokepoint": [c for c, _ in rows], "end_week": [w for _, w in rows]}


def closure_end_table(closures: tuple[tuple[int, float, float], ...], T: int) -> tuple[dict, ...]:
    """(T,) ``closure_end_at(closures, t)`` for t = 1..T, row t - 1 = week t: the view's table, built once."""
    return tuple(closure_end_at(closures, t) for t in range(1, T + 1))


def clairvoyant_pending(inst: Instance, events: tuple[Event, ...], mark_params: MarkParams) -> tuple[dict, ...]:
    """(T,) clairvoyant's exact ``pending_prohibitions`` of every week t = 1..T, row t - 1 = week t (§5.1; §12).

    Columns {edge, k, effective_week} of every (edge, k) pair that an event of ``events`` (omega's ``ev_*``,
    ``marks.read_events``), announced or not, adds to Z_t with its first week in force (its effective week) > t: each
    event's pairs are those ``marks.graph_marks`` of that event alone marks prohibited, less the instance's
    ``prohibitions_at_reset`` (Z_0), each event's graph marks computed once; in (effective_week, edge, k) order,
    one entry per (event, edge, k) (``messages.pending_columns``, every event shown from week 0). The effective week
    is w(t_on) = ceil(t_on) + 1, the week a binary mark of the event enters force by (1), as ``messages.pending_at``
    reads a publication's. Only a sanction marks a pair prohibited beyond Z_0 (``marks``), so only sanctions are
    marked; the type table has already accepted every event (``compute_marks`` of the same omega, V3).
    """
    Z0 = set(inst.prohibitions_at_reset)
    pubs = []
    for q in events:
        if q.type != SANCTION:  # its prohibited mark is Z_0 alone, which adds no pending pair
            continue
        Z = graph_marks(inst, (q,), mark_params)["prohibited"]
        e_idx, k_idx = np.nonzero(Z.any(axis=0))
        pairs = tuple((int(e), int(k)) for e, k in zip(e_idx.tolist(), k_idx.tolist()) if (e, k) not in Z0)
        pubs.append((0, week_of(q.onset), pairs))
    return tuple(messages.pending_columns(pubs, t) for t in range(1, inst.T + 1))

"""The generator: one omega of the public generator G_pub for (instance, params, E_split, episode) (design §4).

``sample_events`` runs the subsystems in order (regime layers and latent risk, the cluster sampler over burn-in and
episode, event typing and marks, the unexcited Poisson components, then the dyad strait closures that marked regional
conflicts derive on `small` and `full`, ``events.strait_closures``, M5); ``sample_omega`` assembles the schema-1
container (§4.1 table) with demand (21) on the regime path, the stored marks, and empty shadow, opponent and blackout
groups (filled by milestone M3 and phase 4). Every draw is keyed (27), so the same (E_split, n, params) gives the same
omega in every process, and a lower rung's events are a subset of a higher rung's with identical marks (V2).
``sample_closures`` draws only what naive's F_Q (66) reads: the militarised and weather closures ``sample_events``
would store, less the strait closures, which naive's transient F_Q leaves out (M5-O13 default).
An ``EventSample`` records the draw it comes from (E_split, n and the generator id of (params, instance)), and
``sample_omega`` refuses to assemble a reused sample under any other (DET-M2-5): omega is a function of (E_split, n,
params) and its meta fields name them (§4.1). The three entry points run under the fixed floating-point error state
``marks.FP_ERRORS`` (``marks.fixed_fp_errors``, §12 'Floating-point error state'), whatever the caller's NumPy error
state, so every subsystem they call sees one state (SIMP-M2R2-11).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from shockbench_flow.disruption import targets
from shockbench_flow.disruption.events import (
    MarkedEvent,
    event_arrays,
    in_episode,
    mark_closures,
    mark_events,
    poisson_events,
    strait_closures,
    weather_events,
)
from shockbench_flow.disruption.hawkes import RawEvent, check_regimes, sample_cluster
from shockbench_flow.disruption.params import GeneratorParams, generator_id
from shockbench_flow.disruption.regime import RegimePaths, _dyad_pairs, sample_regimes
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import fixed_fp_errors


@dataclass(frozen=True)
class EventSample:
    """Everything the generator drew for one episode, before assembly (tests read the full genealogy).

    The last three fields are the draw's provenance, set by ``sample_events``; a sample built by hand without them
    (None) serves every reader but ``sample_omega``, which refuses it.
    """

    regimes: RegimePaths
    raw: tuple[RawEvent, ...]  # the whole cluster process, burn-in included
    marked: tuple[MarkedEvent | None, ...]  # aligned with ``raw``; None for a no-op event
    poisson: tuple[MarkedEvent, ...]
    stored: tuple[MarkedEvent, ...]  # the events omega stores: graph operation, window reaching the episode
    entropy: int | None = None  # E_split of (27) the draw used
    episode: int | None = None  # n of (27)
    generator_id: str | None = None  # ``params.generator_id`` of the draw's (params, instance), omega's meta field
    # the one composition of (30) the draw typed its events by (``targets.TypeRules``, built once per episode and
    # shared with the shadow sampler, SIMP-M2R2-05); a cache of (inst, params, regimes), so never compared
    rules: targets.TypeRules | None = field(default=None, compare=False, repr=False)
    # events derived from marked ones outside the cluster process, burn-in included: the dyad strait closures of
    # ``events.strait_closures`` (M5; none on `tiny`)
    derived: tuple[MarkedEvent, ...] = ()


def _stored(inst: Instance, params: GeneratorParams, *groups: Iterable[MarkedEvent | None]) -> tuple[MarkedEvent, ...]:
    """The events omega stores, in (onset, key) order: the non-None events of ``groups`` in the episode (§4.3).

    ``events.in_episode`` at the params' war-profile window keeps the events whose storage window reaches past the
    instant 0 and whose onset is before T; the one storage rule of ``sample_events`` and ``sample_closures``
    (SIMP-M2R2-12). Keys are unique, so the order restricts to any subset.
    """
    window = params.marks.war_profile_window
    stored = [e for g in groups for e in g if e is not None and in_episode(e, inst.T, window)]
    stored.sort(key=lambda e: (e.onset, e.key))
    return tuple(stored)


@fixed_fp_errors
def sample_events(
    inst: Instance, params: GeneratorParams, entropy: int, episode: int, noise: bool = True
) -> EventSample:
    """Run the disruption subsystems of §4.2-4.4 for one episode, under ``marks.FP_ERRORS``.

    ``noise`` False skips the score noise W of (45) (``regime.sample_regimes``; W is then None): for callers that
    discard it; every event is the same either way. ``sample_omega`` stores W and needs it. The sample records its
    provenance (``entropy``, ``episode``, ``generator_id(params, inst)``).
    """
    regimes = sample_regimes(inst, params, entropy, episode, noise)
    rules = targets.TypeRules(inst, params, regimes)  # the one composition of (30), shared (SIMP-M2R2-05)
    raw = sample_cluster(inst, params, regimes, entropy, episode, rules=rules)
    marked = mark_events(inst, params, regimes, raw, entropy, episode, rules=rules)
    pois = poisson_events(inst, params, entropy, episode)
    derived = strait_closures(inst, params, marked)  # M5-O13: after the cluster process, so they excite nothing
    stored = _stored(inst, params, marked, pois, derived)
    gid = generator_id(params, inst)
    return EventSample(
        regimes, tuple(raw), tuple(marked), tuple(pois), stored, entropy, episode, gid, rules, tuple(derived)
    )


@dataclass(frozen=True)
class ClosureSample:
    """The chokepoint closures of one episode's ``EventSample.stored`` (naive's F_Q, (66)), in the same order."""

    stored: tuple[MarkedEvent, ...]  # militarised and weather closures omega stores, by (onset, key)


@fixed_fp_errors
def sample_closures(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> ClosureSample:
    """The militarised and weather closures of ``sample_events(inst, params, entropy, episode).stored``, alone.

    Naive's F_Q (``policies.naive_fq.fq_laws``) reads only the transient closures of a chokepoint, so this draws the
    regime paths without the noise W, the whole cluster process (every event excites (30)), every event's type and a
    militarised closure's target, duration (39) and severity (``events.mark_closures``, the code of ``mark_events``),
    and the weather closures (``events.weather_events``); strikes and every other event's marks are not drawn. The
    events kept, their fields and their (onset, key) order are those of ``sample_events`` (keys are unique, so the
    order restricts), equality tested over many replications at every rung, less the dyad strait closures of `small`
    and `full` (``events.strait_closures``), which naive's transient F_Q leaves out (M5-O13 default; design §12 M5
    generator rows) and so are not drawn here.
    """
    regimes = sample_regimes(inst, params, entropy, episode, noise=False)
    rules = targets.TypeRules(inst, params, regimes)
    raw = sample_cluster(inst, params, regimes, entropy, episode, rules=rules)
    closures = mark_closures(inst, params, regimes, raw, entropy, episode, rules=rules)
    return ClosureSample(_stored(inst, params, closures, weather_events(inst, params, entropy, episode)))


@fixed_fp_errors
def sample_omega(inst: Instance, params: GeneratorParams, entropy: int, episode: int, split: str = "dev", sample=None):
    """One omega (26) as a schema-1 container; ``sample`` reuses an ``EventSample`` already drawn (with its W).

    A reused sample must be the draw of these arguments: its provenance (E_split, n, generator id) equal to
    (``entropy``, ``episode``, ``generator_id(params, inst)``) and its regime paths of this burn-in, T and shape
    (``hawkes.check_regimes``), since omega is a function of (E_split, n, params) and its meta fields say so (§4.1).
    The container is ``omega.assembly.assemble_omega`` of the stored events, the regime paths (W included), the
    profile's mark parameters and demand (21) with its noise on the conflict layer, the one assembly shared with the
    injected builder. omega is the episode's at the generator's rung: it records the content digest of
    ``inst.at_rung(params.hawkes.gamma)``, the instance whose warm start is naive's steady state with that rung's F_Q
    (§2.3; owner queue M5-O37 (b)), so the reset, the marks and (51) start from that rung's block whichever rung's
    instance of the file the caller holds (``Instance.at_digest``); on `tiny`, and at a gamma the instance keeps no
    block for, it is ``inst``'s own. Runs under the fixed floating-point error state ``marks.FP_ERRORS``, the draw of
    the sample included, whatever the caller's NumPy error state.

    Raises:
        ValueError: if ``sample`` was drawn for another E_split, episode or generator (or carries no provenance), has
            regime paths of another burn-in, T or shape, or was drawn without the score noise W (``noise=False``),
            which omega stores (45).

    """
    from shockbench_flow.omega import assembly

    if sample is None:
        s = sample_events(inst, params, entropy, episode)
        gid = s.generator_id
    else:
        s, gid = sample, generator_id(params, inst)
        drawn, asked = (s.entropy, s.episode, s.generator_id), (entropy, episode, gid)
        if drawn != asked:
            raise ValueError(
                f"the sample was drawn for (E_split, episode, generator) {drawn}, not {asked}: omega is a function of"
                " (E_split, n, params) (§4.1)"
            )
        check_regimes(inst, params, s.regimes)
    r = s.regimes
    if r.W is None:
        raise ValueError("the sample was drawn without the score noise W (noise=False); omega stores W (45)")
    regimes = {"z_c": r.z_c, "z_p": r.z_p, "z_dyad": r.z_dyad, "X": r.X, "W": r.W}
    announcements = information = None
    if params.information is not None:  # M3: the announcement stage (streams 4, 7, 11) and the own chains
        from shockbench_flow.disruption import announce

        ann = announce.sample_announcements(inst, params, s, entropy, episode)
        announcements = announce.announcement_arrays(inst, ann, params.burn_in)
        information = announce.information_meta(params)
        regimes.update(z_c_own=r.z_c_own, dyad_regions=np.array(dyad_rows(inst, params), dtype=np.int16).reshape(-1, 2))
    gamma = params.hawkes.gamma
    return assembly.assemble_omega(
        inst.at_rung(gamma) if gamma in inst.rung_stock else inst,  # the rung's warm start (§2.3; M5-O37 (b))
        event_arrays(inst, list(s.stored), params.burn_in),
        generator_id=gid,
        split=split,
        episode=episode,
        burn_in=params.burn_in,
        params=params.marks,
        entropy=entropy,
        demand_noise=True,
        regimes=regimes,
        announcements=announcements,
        information=information,
    )


def dyad_rows(inst: Instance, params: GeneratorParams) -> tuple[tuple[int, int], ...]:
    """The region indices (a, b) of each dyad of ``params.regime.dyads``, in z_dyad row order: omega's dyad_regions.

    SIMP-M2R3-05: omega names its dyad rows, published as ``Static.dyads``; the same rule as the regime sampler's.
    """
    return tuple(_dyad_pairs(inst, params.regime))

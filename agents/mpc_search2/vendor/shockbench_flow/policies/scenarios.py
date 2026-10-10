"""Scenarios of ``mpc_scen`` and ``hindsight_consensus``: G_pub(. | observations to t) as built (M4, "lp-baselines").

Design (70), §8.2; Q17, Q100 (1).

The owner's rule "residual durations" (Q100 (1), 2026-09-27), as the design §12 M4 row "Scenario draws" records it and
the stream's rows "Scenario windows as built", "Inferred types and residual keys as built", "Residual life as built"
and "Scenario library as built" refine it:

- **A scenario** is the observed present plus future onsets. Each impairment observed at the instant t - 1 (a
  chokepoint's open fraction below 1, an edge's capacity below u^0, a prohibition, a tariff, a war-risk class, a fab's
  or OSAT's R below 1, a grid's G-bar below nominal, a supply cap below nominal, a freight rate above c^0) ends after a
  residual duration drawn each week from the published duration law of its inferred event type, conditioned on its
  observed age a (the owner's "Condition on age", Q100 (1)): S(a + r) / S(a), S the type's survival function, where the
  start was seen (the element shown nominal the week before it was first shown impaired); the equilibrium
  (length-biased) residual life (density S(r) / E[T]), conditioned the same way on the weeks observed since, where it
  was not (carried in at week 1, or first seen after a blackout or a masked week). ``ImpairmentAges`` books the ages;
  the future onsets are the events of one public-generator draw with onset > t - 1. Pending prohibitions
  (``obs["pending_prohibitions"]``) switch on from their effective week in every scenario, as in
  ``lp_common.persistence_arrays``, so the pair mpc_scen > mpc_det differs by the scenarios only. Demand is the point
  forecast of ``lp_common.persistence_arrays`` in every scenario. §8.2's "regime-conditioned" is read as conditioning on
  the observed present only: warnings and messages are not used, and the future onsets come from the public generator's
  own regime path (G_pub's draw of episode 0), not conditioned on the observed regime.
- **The window's marks** (design §12 "Scenario windows as built"): no event type can carry most present impairments
  alone without touching elements observed nominal (no event cuts one edge or restores one fab), so a window composes
  two layers as ``marks.py`` composes events (Q50): the *future layer* is ``marks.compute_marks`` of the draw's events
  with onset > t - 1 on a synthetic omega, never re-coded; the *present layer* holds each observed element at its
  observed severity over [t - 1, t - 1 + r) by (37)'s overlap rule (``marks.open_fraction``) for week averages and by
  (1)'s in-force rule for binary marks. Capacities multiply, tariffs add, prohibitions unite, classes take the larger;
  sigma^scr and the fab hits come from the future layer only (observed WIP is net of the present's scrap).
- **The inferred type** of an observed impairment (design §12 "Inferred types and residual keys as built"): a
  chokepoint closure is a militarised closure when its war-risk class is not none, else a weather closure; a war-risk
  class is the militarised closure of c when c is observed impaired or the class is red_sea (class 1, which only a
  closure sets), else (class 2 with c open) a regional conflict in a region adjacent to c; a prohibition is a
  sanction; a tariff a tariff; an edge capacity loss a port strike, under the generator's strike duration law as drawn
  (stoppage with probability ``stoppage_share``, else slowdown: the observed severity is pro-rated within a week and
  does not tell them apart); a fab or OSAT restoration a regional conflict; a G-bar loss an energy shock; a supply cap
  below nominal (varsigma-bar 0) a material outage; a freight surcharge (c above c^0) piracy.
- **The public generator** is the split's public profile and rung named in the policy config (``GeneratorRef``;
  the anchor rung of the instance's profile on an injected list); nothing comes from omega or Static.
- **Entropy**: public keyed streams, like naive's stream 22, never E_split and never the policy seed. Stream 23's keys
  lead with a form code, as stream 1's do (design §4.1 table): scenario i's generator run takes the 128-bit entropy of
  key (``FORM_ENTROPY``, instance kind, rung index, baseline tag, i) on the public root 0, episode component 0, and
  samples episode 0; the residual uniforms come from key (``FORM_RESIDUAL``, instance kind, rung index, baseline tag,
  i, week, impairment code, element), the week coded as calendar week + B_burn (§4.1, Q87). So the scenario library
  is a function of (instance, generator, baseline, count): the same draws in every episode (common random numbers
  across episodes), drawn once per process and cached like naive's F_Q (0.4-1.3 s per draw on `tiny`), and an
  omega-hat is never omega (whose generator run takes E_split itself).
- **Stream 20** (the (70) tie-break weights): one (T, nc) array of uniforms on [0, 1) per instance, on the public
  root 0, episode 0, key (instance kind,); a rolled window takes the calendar weeks t..t + H - 1, so a column keeps its
  weight across re-solves. The key lies in any dev split's keyspace at E_split = 0 (n = 0, stream 20), harmless since
  omega never draws stream 20; a V2 draw audit covers omega sampling only.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtr

from shockbench_flow import marks
from shockbench_flow.disruption.events import event_arrays
from shockbench_flow.disruption.laws import Law
from shockbench_flow.disruption.params import GeneratorParams, generator_id
from shockbench_flow.disruption.sampler import sample_events
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes, seeds
from shockbench_flow.omega.assembly import meta
from shockbench_flow.omega.container import ARRAY_DTYPES, EVENT_FIELDS, Omega, immutable_copy
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.naive_fq import RUNGS, rung_index


PUBLIC_ROOT = 0  # the scenario library is public and reproducible by anyone (like naive's stream 22, §8.1)
SCENARIO_EPISODE = 0  # the episode component of the stream-23 key, and the episode each draw samples
TAG_MPC_SCEN = 0  # baseline tags of the stream-23 key: mpc_scen and hindsight_consensus never share a draw
TAG_HINDSIGHT = 1
FORM_ENTROPY = 0  # stream-23 key forms (the leading component, design §4.1 row 23): a generator run's entropy
FORM_RESIDUAL = 1  # a residual-duration uniform
# the impairment codes of the residual-uniform key (instance ordinals follow as the element)
IMPAIRMENT_KINDS = (
    "closure",
    "capacity",
    "prohibition",
    "tariff",
    "war_risk",
    "fab",
    "osat",
    "grid",
    "supply",
    "freight",
)
_TYPE = {name: codes.EVENT_TYPES.index(name) for name in codes.EVENT_TYPES}
_CONTINUOUS = ("closure", "capacity", "fab", "osat", "grid", "supply")  # week averages: (37)'s overlap rule
_BISECTIONS = 2000  # the residual inversion stops earlier, when the midpoint no longer moves (about one ulp)


@dataclass(frozen=True)
class ScenarioDraw:
    """One public-generator draw: the stored events (``ev_*`` arrays, read-only) of a G_pub run, and its provenance."""

    index: int
    entropy_sha256: str  # SHA-256 of the 128-bit entropy (the entropy itself is public, but never an E_split)
    generator_id: str
    events: Mapping[str, np.ndarray]  # the ``ev_*`` arrays of the draw (``omega.container`` names)


def _check_tag(tag: int) -> int:
    if isinstance(tag, bool) or tag not in (TAG_MPC_SCEN, TAG_HINDSIGHT):
        raise ValueError(f"the baseline tag must be TAG_MPC_SCEN or TAG_HINDSIGHT, got {tag!r}")
    return int(tag)


def scenario_key(inst: Instance, params: GeneratorParams, tag: int, i: int) -> tuple[int, ...]:
    """The stream-23 entropy key (``FORM_ENTROPY``, instance kind, rung index, baseline tag, scenario index).

    Raises:
        ValueError: on a tag outside (TAG_MPC_SCEN, TAG_HINDSIGHT), i < 0, or a gamma that is no rung.

    """
    tag = _check_tag(tag)
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0:
        raise ValueError(f"the scenario index must be an integer >= 0, got {i!r}")
    return (FORM_ENTROPY, codes.INSTANCE_KINDS.index(inst.kind), rung_index(params), tag, int(i))


def scenario_entropy(inst: Instance, params: GeneratorParams, tag: int, i: int) -> int:
    """The 128-bit generator entropy of scenario i, from stream 23 on ``PUBLIC_ROOT`` (``seeds.seed_sequence``)."""
    ss = seeds.seed_sequence(PUBLIC_ROOT, SCENARIO_EPISODE, codes.STREAM_SCENARIOS, scenario_key(inst, params, tag, i))
    words = ss.generate_state(2, np.uint64)
    return int(words[0]) | (int(words[1]) << 64)


# (content digest, generator_id, tag, i) -> the draw (read-only arrays): drawn once per process (docstring)
_DRAWS: dict[tuple[str, str, int, int], ScenarioDraw] = {}


def _draw(inst: Instance, params: GeneratorParams, tag: int, i: int) -> ScenarioDraw:
    gid = generator_id(params, inst)
    key = (inst.content_digest, gid, tag, i)
    if key not in _DRAWS:
        entropy = scenario_entropy(inst, params, tag, i)
        sample = sample_events(inst, params, entropy, SCENARIO_EPISODE, noise=False)
        ev = event_arrays(inst, list(sample.stored), params.burn_in)
        _DRAWS[key] = ScenarioDraw(
            index=i,
            entropy_sha256=hashlib.sha256(entropy.to_bytes(16, "little")).hexdigest(),
            generator_id=gid,
            events={name: immutable_copy(np.asarray(a)) for name, a in ev.items()},
        )
    return _DRAWS[key]


def scenario_library(inst: Instance, params: GeneratorParams, tag: int, n: int) -> tuple[ScenarioDraw, ...]:
    """The first ``n`` draws of the library, cached per process by (content digest, ``generator_id``, tag, n).

    Each draw is the stored events of ``sampler.sample_events(inst, params, scenario_entropy(...),
    SCENARIO_EPISODE, noise=False)`` written by ``disruption.events.event_arrays``: the ``ev_*`` rows
    ``sampler.sample_omega`` stores for that entropy (design §12 "Scenario library as built"; neither ``omega.seeds``
    nor ``disruption.sampler`` is edited). Draws are cached one by one, so libraries of different sizes nest. The cache
    holds immutable copies.

    Raises:
        ValueError: if n < 1, or as ``scenario_key``.

    """
    tag = _check_tag(tag)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"a scenario library holds n >= 1 draws, got {n!r}")
    return tuple(_draw(inst, params, tag, i) for i in range(int(n)))


# the public generator of a policy: (content digest, profile, gamma) -> GeneratorParams, built once per process
_PARAMS: dict[tuple[str, str, float], GeneratorParams] = {}


def generator_params(inst: Instance, generator) -> GeneratorParams:
    """The public generator of a scenario baseline, built once per (content digest, profile, gamma).

    ``generator.params(inst)``, or on an injected list (``generator`` None) the anchor rung of the profile named by
    the instance kind (design §12 M4 row "Scenario draws").

    Raises:
        ValueError: if ``generator`` is None and no profile is named after the instance kind.

    """
    from shockbench_flow.policies.registry import PROFILES, GeneratorRef

    if generator is None:
        if inst.kind not in PROFILES:
            raise ValueError(f"no public generator profile is named after the instance kind {inst.kind!r}")
        generator = GeneratorRef(inst.kind, RUNGS[0])
    key = (inst.content_digest, generator.profile, float(generator.gamma))
    if key not in _PARAMS:
        _PARAMS[key] = generator.params(inst)
    return _PARAMS[key]


# ----- residual life -------------------------------------------------------------------------------------------------
def duration_components(params: GeneratorParams, type_code: int) -> tuple[tuple[float, Law], ...]:
    """The duration law of event type ``type_code`` as (weight, one-uniform law) components (``Law.components``).

    A port strike is the stoppage/slowdown mixture of ``PoissonParams`` at ``stoppage_share``; every other type its
    ``MarkLaws.duration`` law.

    Raises:
        ValueError: if the type has no duration law.

    """
    if isinstance(type_code, bool) or not isinstance(type_code, (int, np.integer)):
        raise ValueError(f"an event type code is an integer, got {type_code!r}")
    if not 0 <= type_code < len(codes.EVENT_TYPES):
        raise ValueError(f"unknown event type code {type_code}")
    name = codes.EVENT_TYPES[type_code]
    if name == "port_strike":
        p = params.poisson.stoppage_share
        return tuple((p * w, law) for w, law in params.poisson.stoppage_duration.components()) + tuple(
            ((1.0 - p) * w, law) for w, law in params.poisson.slowdown_duration.components()
        )
    laws = dict(params.laws.duration)
    if name not in laws:
        raise ValueError(f"event type {name!r} has no duration law in the generator")
    return laws[name].components()


def _log_weeks(law: Law) -> tuple[float, float]:
    mu, s = law.params
    return (mu - math.log(7.0), s) if law.kind == "lognormal_days" else (mu, s)


def _component_moments(law: Law) -> float:
    """E[T] of one component, in weeks."""
    if law.kind == "constant":
        return float(law.params[0])
    if law.kind in ("lognormal_days", "lognormal_weeks"):
        mu, s = _log_weeks(law)
        return math.exp(mu + 0.5 * s * s)
    if law.kind == "weibull_days":  # Q106: lambda Gamma(1 + 1/k), lambda in weeks (``Law.mean``)
        return law.mean()
    raise ValueError(f"residual life has no closed form for a {law.kind!r} component")


def _integrated_survival(law: Law, r: float) -> float:
    """G(r) = int_0^r S(x) dx of one component (design §12 "Residual life as built")."""
    if r <= 0.0:
        return 0.0
    if law.kind == "constant":
        return min(r, float(law.params[0]))
    if law.kind == "weibull_days":  # E[T] - int_r^inf S (Q106; ``Law.tail_integral``)
        return law.mean() if math.isinf(r) else max(law.mean() - law.tail_integral(r), 0.0)
    mu, s = _log_weeks(law)
    if math.isinf(r):
        return math.exp(mu + 0.5 * s * s)
    z = (math.log(r) - mu) / s
    return r * float(ndtr(-z)) + math.exp(mu + 0.5 * s * s) * float(ndtr(z - s))


def residual_cdf(params: GeneratorParams, type_code: int, r: float) -> float:
    """F_R(r) = int_0^r S(x) dx / E[T] of the type's duration law, r in weeks."""
    parts = duration_components(params, type_code)
    mean = math.fsum(w * _component_moments(law) for w, law in parts if w > 0)
    return math.fsum(w * _integrated_survival(law, r) for w, law in parts if w > 0) / mean


def _invert(F, u: float, hi: float) -> float:
    """The smallest r >= 0 with F(r) >= u for a non-decreasing F: bisection on [0, h], h doubled from ``hi``.

    h doubles until F(h) >= u; the bisection stops when the midpoint no longer moves (about one ulp).
    """
    while F(hi) < u:
        hi *= 2.0
        if hi > 1e300:
            return hi
    lo = 0.0
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        if F(mid) >= u:
            hi = mid
        else:
            lo = mid
    return hi


def _log_survival(law: Law, x: float) -> float:
    """The log survival log P(T > x) of one component, x >= 0 in weeks (so a long observed age never underflows)."""
    if law.kind == "constant":
        return 0.0 if x < float(law.params[0]) else -math.inf
    if law.kind == "weibull_days":  # log S(x) = -(7 x / lambda)^k (Q106), exact in log space
        return 0.0 if x <= 0.0 else -((7.0 * x / law.params[1]) ** law.params[0])
    if law.kind not in ("lognormal_days", "lognormal_weeks"):
        raise ValueError(f"residual life has no closed form for a {law.kind!r} component")
    if x <= 0.0:
        return 0.0
    mu, s = _log_weeks(law)
    return float(log_ndtr(-(math.log(x) - mu) / s))


def _tail_integral(law: Law, x: float) -> float:
    """int_x^inf S(y) dy = E[T] - G(x) of one component, x >= 0 in weeks (clamped at 0 against cancellation)."""
    if law.kind == "constant":
        return max(float(law.params[0]) - x, 0.0)
    if law.kind == "weibull_days":  # lambda Gamma(1 + 1/k) Q(1/k, (x / lambda)^k) (Q106; ``Law.tail_integral``)
        return law.tail_integral(max(x, 0.0))
    mu, s = _log_weeks(law)
    m = math.exp(mu + 0.5 * s * s)
    if x <= 0.0:
        return m
    z = (math.log(x) - mu) / s
    return max(m * float(ndtr(s - z)) - x * float(ndtr(-z)), 0.0)


def _mixture_log_survival(parts, x: float) -> float:
    """The log of sum_i w_i S_i(x) over the components of positive weight."""
    terms = [math.log(w) + _log_survival(law, x) for w, law in parts if w > 0]
    top = max(terms)
    if top == -math.inf:
        return top
    return top + math.log(math.fsum(math.exp(v - top) for v in terms))


def residual_duration(
    inst: Instance, params: GeneratorParams, type_code: int, u: float, *, age: float = 0.0, start_seen: bool = False
) -> float:
    """The residual life of event type ``type_code`` at the uniform ``u`` in (0, 1), given its observed age, in weeks.

    Q100 (1) ("Condition on age"; design §12 "Residual life as built"): the survival function S of the type's duration
    law of the generator (``disruption.laws``; the persistent/transient mixture of (39) for militarised closures; for
    port strikes the stoppage/slowdown mixture of ``PoissonParams`` at ``stoppage_share``) conditioned on the observed
    age a = ``age`` (weeks since the week whose observation first showed the element impaired, ``ImpairmentAges``):

    - ``start_seen`` (the element was shown nominal the week before): P(R > r) = S(a + r) / S(a), in log space; at
      a = 0 the type's own duration law;
    - the start not seen (carried in at week 1, or first seen after a blackout or a masked week): the equilibrium
      (length-biased) residual, F_R(r) = int_0^r S(x) dx / E[T] at a = 0, and past a weeks of observed survival
      P(R > r) = (1 - F_R(a + r)) / (1 - F_R(a)); where 1 - F_R(a) underflows to 0 (an age far beyond the law's
      support) the seen form S(a + r) / S(a) is taken.

    The smallest r with F(r) >= u, by bisection. Every type of the inferred-type table (module docstring) has a law.

    Raises:
        ValueError: if ``u`` is not in (0, 1), ``age`` is not finite and >= 0, or the type has no duration law.

    """
    if isinstance(u, bool) or not 0.0 < u < 1.0:
        raise ValueError(f"a residual uniform lies in (0, 1), got {u!r}")
    if isinstance(age, bool) or not (math.isfinite(age) and age >= 0.0):
        raise ValueError(f"an observed age is a finite number of weeks >= 0, got {age!r}")
    parts = duration_components(params, type_code)
    mean = math.fsum(w * _component_moments(law) for w, law in parts if w > 0)
    if not start_seen and age == 0.0:  # the equilibrium residual F_R^{-1}(u)

        def F_eq(r: float) -> float:
            return math.fsum(w * _integrated_survival(law, r) for w, law in parts if w > 0) / mean

        return _invert(F_eq, u, mean)
    if not start_seen:
        base_tail = math.fsum(w * _tail_integral(law, age) for w, law in parts if w > 0)
        if base_tail > 0.0:

            def F_eq_aged(r: float) -> float:
                return 1.0 - math.fsum(w * _tail_integral(law, age + r) for w, law in parts if w > 0) / base_tail

            return _invert(F_eq_aged, u, max(mean, 1.0))
    base = _mixture_log_survival(parts, age)
    if base == -math.inf:  # every component's support ends by a: the impairment ends at once
        return 0.0

    def F_aged(r: float) -> float:
        return -math.expm1(_mixture_log_survival(parts, age + r) - base)

    return _invert(F_aged, u, max(mean, 1.0))


class ImpairmentAges:
    """The observed age of every present impairment element, per episode (mutable; created at reset, never pickled).

    Q100 (1) (design §12 "Residual life as built"). ``update(inst, obs, memory)`` runs each week after
    ``memory.update``: an element of ``present_elements(inst, memory)`` not yet in the book opens a run at week t, its
    start seen when week t - 1's observation showed the element nominal (a non-null entry, ``shown_elements``) and not
    seen otherwise (week 1, a blackout week, a coverage-masked entry, or no update in week t - 1); an element that left
    ``present_elements`` (shown nominal) leaves the book, so it opens a new run when shown impaired again. A masked
    element keeps its last observed value in ``memory`` and so its run. ``age(t, el)`` is (t - first week, start seen);
    an element outside the book reads (0, not seen).
    """

    def __init__(self) -> None:
        self.book: dict[tuple[str, int], tuple[int, bool]] = {}  # (kind, element) -> (first week shown, start seen)
        self.week: int | None = None  # the week of the last update
        self.shown: dict[str, object] | None = None  # that week's ``shown_elements``, as plain lists and bools

    def update(self, inst: Instance, obs: dict, memory: L.ObservedGraph) -> None:
        """Book week t's present elements (class docstring); ``memory`` already holds week t's observation."""
        t = int(obs["week"])
        previous = self.shown if self.week == t - 1 else None
        book: dict[tuple[str, int], tuple[int, bool]] = {}
        for el in present_elements(inst, memory):
            key = (el.kind, el.element)
            if key in self.book:
                book[key] = self.book[key]
            else:
                book[key] = (t, previous is not None and _is_shown(previous, el.kind, el.element))
        self.book, self.week = book, t
        self.shown = {k: (v if isinstance(v, bool) else v.tolist()) for k, v in shown_elements(inst, obs).items()}

    def age(self, t: int, el: "Element") -> tuple[float, bool]:
        """(observed age in weeks, start seen) of ``el`` in week t."""
        first, seen = self.book.get((el.kind, el.element), (int(t), False))
        return float(int(t) - first), seen

    def state(self) -> dict:
        """Plain data for an in-process resume."""
        return {"book": [[k, e, f, s] for (k, e), (f, s) in self.book.items()], "week": self.week, "shown": self.shown}

    @classmethod
    def from_state(cls, state: dict) -> "ImpairmentAges":
        """The book ``state()`` wrote."""
        out = cls()
        out.book = {(str(k), int(e)): (int(f), bool(s)) for k, e, f, s in state["book"]}
        out.week = None if state["week"] is None else int(state["week"])
        out.shown = state["shown"]
        return out


_SHOWN_KEYS = {
    "closure": "open",
    "capacity": "u",
    "war_risk": "war_risk",
    "fab": "fab_R",
    "osat": "osat_R",
    "grid": "grid_G",
    "supply": "supply",
    "freight": "c",
}


def shown_elements(inst: Instance, obs: dict) -> dict[str, np.ndarray | bool]:
    """Which elements week t's ``graph_now`` shows (a non-null entry), per impairment kind of ``IMPAIRMENT_KINDS``.

    A boolean array over the element ordinals, read by ``ObservedGraph.update`` itself on a probe memory of null
    sentinels; one bool for "prohibition" and "tariff", which are shown whole (Z_t, never masked by a coverage rung,
    and the tariff table). A blackout week (null ``graph_now``) shows nothing.
    """
    probe = L.ObservedGraph.nominal(inst)
    for key in _SHOWN_KEYS.values():
        a = probe.values[key]
        probe.values[key] = np.full(a.shape, -1, dtype=np.int8) if key == "war_risk" else np.full(a.shape, np.nan)
    probe.update(inst, obs)
    out: dict[str, np.ndarray | bool] = {}
    for kind, key in _SHOWN_KEYS.items():
        v = probe.values[key]
        out[kind] = v >= 0 if key == "war_risk" else ~np.isnan(v)
    g = obs.get("graph_now")
    out["prohibition"] = g is not None and g.get("prohibited") is not None
    out["tariff"] = g is not None and g.get("tariff") is not None
    return out


def _is_shown(shown: Mapping[str, object], kind: str, element: int) -> bool:
    v = shown[kind]
    return bool(v) if isinstance(v, bool) else bool(v[element])


# ----- the window's two layers ---------------------------------------------------------------------------------------
# (content digest, generator_id, entropy digest, t) -> the future layer's marks over weeks 1..T (read-only)
_FUTURE: dict[tuple[str, str, str, int], marks.WeeklyMarks] = {}


def future_marks(inst: Instance, params: GeneratorParams, draw: ScenarioDraw, t: int) -> marks.WeeklyMarks:
    """``marks.compute_marks`` of the draw's events with onset > t - 1, weeks 1..T, cached per (draw, t).

    The synthetic omega holds those ``ev_*`` rows (keys rebuilt in CSR form), the instance's hash and content digest,
    the generator's ``meta_mark_params`` and d-bar as ``d`` (the window's demand is the point forecast), and no stored
    marks, so the recomputation is the realisation (design §12 "Scenario windows as built").
    """
    key = (inst.content_digest, draw.generator_id, draw.entropy_sha256, int(t))
    if key in _FUTURE:
        return _FUTURE[key]
    ev = draw.events
    keep = np.flatnonzero(np.asarray(ev["ev_onset"]) > t - 1)
    arrays: dict[str, np.ndarray] = {}
    for name, _dtype in EVENT_FIELDS:
        arrays[f"ev_{name}"] = np.asarray(ev[f"ev_{name}"])[keep]
    ptr, words = np.asarray(ev["ev_key_ptr"]), np.asarray(ev["ev_key"])
    rows = [words[ptr[i] : ptr[i + 1]] for i in keep]
    arrays["ev_key_ptr"] = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    arrays["ev_key"] = (np.concatenate(rows) if rows else np.zeros(0)).astype(np.int64)
    arrays["meta_instance_hash"] = meta(inst.hash, "meta_instance_hash")
    arrays["meta_instance_digest"] = meta(inst.content_digest, "meta_instance_digest")
    arrays["meta_mark_params"] = np.array(marks.mark_params_json(params.marks), dtype=ARRAY_DTYPES["meta_mark_params"])
    arrays["d"] = np.array([[d.dbar] * inst.T for d in inst.demands], dtype=np.float64).reshape(
        len(inst.demands), inst.T
    )
    wm = marks.compute_marks(inst, Omega(arrays))
    _FUTURE[key] = wm
    return wm


@dataclass(frozen=True)
class Element:
    """One observed impairment element of the present layer (design §12 "Inferred types and residual keys as built")."""

    kind: str  # one of IMPAIRMENT_KINDS
    element: int  # the ordinal of the residual key
    type_code: int  # the inferred event type (codes.EVENT_TYPES), whose duration law gives the residual
    value: float  # the observed value (a class code for "war_risk", a rate for "tariff", c_obs / c^0 for "freight")
    nominal: float  # the nominal value (1 for binary marks)


def present_elements(inst: Instance, memory: L.ObservedGraph) -> tuple[Element, ...]:
    """The observed impairments of the remembered instant t - 1, in ``IMPAIRMENT_KINDS`` order, then ordinal order."""
    v = memory.values
    K = len(inst.commodities)
    out: list[Element] = []
    closed = v["open"] < 1.0
    for ci in np.flatnonzero(closed):
        ty = _TYPE["militarised_closure"] if v["war_risk"][ci] > 0 else _TYPE["weather_closure"]
        out.append(Element("closure", int(ci), ty, float(v["open"][ci]), 1.0))
    for e, edge in enumerate(inst.edges):
        if edge.u0 is not None and v["u"][e] < edge.u0:
            out.append(Element("capacity", e, _TYPE["port_strike"], float(v["u"][e]), float(edge.u0)))
    Z0 = set(inst.prohibitions_at_reset)
    for e, k in zip(*np.nonzero(v["prohibited"]), strict=True):
        if (int(e), int(k)) not in Z0:
            out.append(Element("prohibition", int(e) * K + int(k), _TYPE["sanction"], 1.0, 1.0))
    for e, k in zip(*np.nonzero(v["tariff"] > 0.0), strict=True):
        out.append(Element("tariff", int(e) * K + int(k), _TYPE["tariff"], float(v["tariff"][e, k]), 0.0))
    for ci in np.flatnonzero(v["war_risk"] > 0):
        militarised = closed[ci] or v["war_risk"][ci] == 1
        ty = _TYPE["militarised_closure"] if militarised else _TYPE["regional_conflict"]
        out.append(Element("war_risk", int(ci), ty, float(v["war_risk"][ci]), 0.0))
    for fi in np.flatnonzero(v["fab_R"] < 1.0):
        out.append(Element("fab", int(fi), _TYPE["regional_conflict"], float(v["fab_R"][fi]), 1.0))
    for oi in np.flatnonzero(v["osat_R"] < 1.0):
        out.append(Element("osat", int(oi), _TYPE["regional_conflict"], float(v["osat_R"][oi]), 1.0))
    for gi, g in enumerate(inst.grids):
        nom = inst.nodes[g].grid.deliverable
        if v["grid_G"][gi] < nom:
            out.append(Element("grid", gi, _TYPE["energy_shock"], float(v["grid_G"][gi]), float(nom)))
    supply = set(inst.supply_nodes)
    for s, slot in enumerate(inst.stock_slots):
        if slot.node in supply and slot.supply > 0.0 and v["supply"][s] < slot.supply:
            out.append(Element("supply", s, _TYPE["material_outage"], float(v["supply"][s]), float(slot.supply)))
    for e, edge in enumerate(inst.edges):
        if edge.c0 > 0.0 and v["c"][e] > edge.c0:
            out.append(Element("freight", e, _TYPE["piracy"], float(v["c"][e] / edge.c0), 1.0))
    order = {k: i for i, k in enumerate(IMPAIRMENT_KINDS)}
    return tuple(sorted(out, key=lambda el: (order[el.kind], el.element)))


def residuals(
    inst: Instance,
    params: GeneratorParams,
    tag: int,
    i: int,
    t: int,
    elements: Sequence[Element],
    ages: ImpairmentAges,
) -> np.ndarray:
    """One residual per element of scenario i in week t (keys of the module docstring), in weeks.

    Each is ``residual_duration`` at the element's keyed uniform, conditioned on its observed age and whether its
    start was seen, as ``ages`` books them (Q100 (1)). The key's week is the frozen code t + B_burn (§4.1, Q87), B_burn
    the public generator's ``params.burn_in`` (one per size and generator family, so a week keeps its code at every
    rung).
    """
    stream = seeds.KeyedStream(PUBLIC_ROOT, SCENARIO_EPISODE, codes.STREAM_SCENARIOS)
    week = int(t) + int(params.burn_in)  # §4.1: week = calendar week + B_burn (Q87)
    head = (FORM_RESIDUAL, codes.INSTANCE_KINDS.index(inst.kind), rung_index(params), _check_tag(tag), int(i), week)
    out = np.empty(len(elements))
    for n, el in enumerate(elements):
        u = stream.uniform((*head, IMPAIRMENT_KINDS.index(el.kind), el.element))
        age, seen = ages.age(t, el)
        out[n] = residual_duration(inst, params, el.type_code, u, age=age, start_seen=seen)
    return out


def scenario_windows(
    inst: Instance,
    params: GeneratorParams,
    obs: dict,
    memory,
    library: Sequence[ScenarioDraw],
    tag: int,
    H: int,
    *,
    ages: ImpairmentAges,
) -> list[tuple[dict[str, np.ndarray], tuple]]:
    """One (window arrays, fab hits) pair per draw for weeks t..t + H - 1 of week t.

    The arrays hold every field of ``lp_common.WINDOW_FIELDS``, (H, ...); the fab hits are the draw's
    ``WeeklyMarks.fab_hits`` with onset in the window, weeks renumbered, for ``lp_common.rolled_lp(..., fab_hits=)``
    (``build_lp`` scraps the observed initial WIP by them). ``memory`` is the policy's ``lp_common.ObservedGraph``
    (last observed values under blackout and coverage, and the pending prohibitions; it already holds week t's
    observation), ``ages`` the policy's ``ImpairmentAges`` (updated with the same observation). The present's
    impairments end after their residual durations (uniforms keyed by week and impairment, conditioned on the observed
    age, Q100 (1)), pending prohibitions switch on from their effective week, the draw's events with onset > t - 1 are
    added, the marks formed by ``marks.py`` on those events; demand is the point forecast (design §12 "Scenario
    windows as built").

    Raises:
        ValueError: if the observation's week is not an integer in 1..T, H < 1 or t + H - 1 > T.

    """
    T = inst.T
    t = L.observed_week(obs, T)
    L.check_window(inst, t, H)
    tag = _check_tag(tag)
    v = memory.values
    sl = slice(t - 1, t - 1 + H)
    weeks = np.arange(t, t + H)
    elements = present_elements(inst, memory)
    demand = L.point_demand(inst, obs, H)
    pending = L.pending_mask(memory, t, H, v["prohibited"].shape)
    k_mu = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=np.float64).reshape(-1, 2)
    K = len(inst.commodities)
    out = []
    for draw in library:
        fut = future_marks(inst, params, draw, t)
        u, o, supply = fut.u[sl].copy(), fut.o[sl].copy(), fut.supply[sl].copy()
        G, R, R_osat = fut.G_bar[sl].copy(), fut.R[sl].copy(), fut.R_osat[sl].copy()
        c, tariff = fut.c[sl].copy(), fut.tariff[sl].copy()
        prohibited = fut.prohibited[sl] | pending
        wr = fut.wr_class[sl].copy()
        target = {"closure": o, "capacity": u, "fab": R, "osat": R_osat, "grid": G, "supply": supply}
        for el, r in zip(elements, residuals(inst, params, tag, draw.index, t, elements, ages), strict=True):
            if el.kind in _CONTINUOUS:
                sev = 1.0 - el.value / el.nominal  # the observed severity at the instant t - 1
                target[el.kind][:, el.element] *= marks.open_fraction(T, [(t - 1.0, t - 1.0 + r, sev)])[sl]
                continue
            on = weeks < math.ceil(t - 1.0 + r) + 1  # (1): in force from w(t - 1) = t to w(t - 1 + r) - 1
            if el.kind == "prohibition":
                prohibited[on, el.element // K, el.element % K] = True
            elif el.kind == "tariff":
                tariff[on, el.element // K, el.element % K] += el.value
            elif el.kind == "war_risk":
                wr[on, el.element] = np.maximum(wr[on, el.element], np.int8(el.value))
            else:  # freight: c^0 x (c_obs / c^0) while in force, surcharges multiplying (Q50)
                c[on, el.element] *= el.value
        hq, cwr = L.window_queue_and_transit(inst, wr)
        arrays = {
            "u": u,
            "c": c,
            "o": o,
            "kappa": k_mu[None, :, :] * o[:, :, None],
            "supply": supply,
            "G_bar": G,
            "y_bar": np.broadcast_to(v["grid_y"], (H, len(inst.grids))).copy(),
            "R": R,
            "alpha_bar": np.maximum(fut.alpha_bar[sl], v["fab_alpha"][None, :]),
            "sigma_scr": fut.sigma_scr[sl].copy(),
            "R_osat": R_osat,
            "demand": demand.copy(),
            "prohibited": prohibited,
            "tariff": tariff,
            "wr_class": wr,
            "h_queue": hq,
            "c_wr": cwr,
        }
        L.with_now(arrays)
        hits = tuple(
            marks.FabHit(fab=h.fab, onset=h.onset - (t - 1), severity=h.severity)
            for h in fut.fab_hits
            if h.onset > t - 1 and t <= h.onset_week <= t + H - 1
        )
        out.append((L.read_only(arrays), hits))
    return out


def stream20_weights(inst: Instance, nc: int) -> np.ndarray:
    """(T, nc) read-only uniforms on [0, 1): the (70) tie-break weights w of one instance (module docstring).

    Raises:
        ValueError: if nc < 1.

    """
    if isinstance(nc, bool) or not isinstance(nc, (int, np.integer)) or nc < 1:
        raise ValueError(f"stream 20 draws nc >= 1 weights per week, got {nc!r}")
    g = seeds.generator(PUBLIC_ROOT, 0, codes.STREAM_HC_WEIGHTS, (codes.INSTANCE_KINDS.index(inst.kind),))
    w = g.random((inst.T, int(nc)))
    w.flags.writeable = False
    return w

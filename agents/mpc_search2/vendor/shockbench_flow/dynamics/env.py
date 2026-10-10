"""Gymnasium-shaped environment without a gymnasium dependency (design §9.1, §9.3; Q68, Q86).

``Env.reset(instance, regime, omega, policy_seed) -> (obs, info)``; ``Env.step(action) -> (obs, reward, terminated,
truncated, info)`` with ``reward`` = r_t / 100 USD by (25) (integer cents in ``info["reward_cents"]``), ``terminated``
at T, ``truncated`` never set. ``info`` (local runs only) carries ``cost_components``, ``clipped``, ``invalid``,
``clamps``, ``edge_clamps`` (action slots) and ``override_clamps`` (override slots) (Q95). Actions are dicts of the
wire schema (§9.2): ``{"week", "flows": {"slot", "qty"}, "overrides": {"slot", "qty"} | None, "hold": {"chokepoint",
"k"} | None}``.

Readings chosen where §9.3 leaves room (conservative; design §12 'Phase-3 implementation notes'):

- Whole-week failures (``ValueError``, D9 fallback; design §12 "Whole-week failures"): the action is not a dict;
  ``week`` is missing, not an integer or not the week asked for; ``flows``, ``overrides`` or ``hold`` is present and
  not a dict of equal-length lists under the schema's keys (the reply cannot be paired into entries), a field holding
  a number without a JSON value (NaN, an infinity, an int or ``Fraction`` beyond the float range) included. A missing
  or null ``flows`` is an empty request (a missing slot is 0); a null ``overrides`` or ``hold`` is none.
- Entry rules, checked in this order, each entry dropped and logged on the first failure: the slot is not an integer
  index (bools are not), out of range, a duplicate (an earlier entry with the same slot passed these two checks,
  whatever its fate), the quantity is not a finite real number (bools are not), is negative, or the slot is masked
  this week (an override slot by its own (out-edge, k)). Hold entries
  must name a chokepoint node and a tanker commodity (K^ov, Q34); containerised cargo always follows the default.
- Log strings carry no value that JSON cannot hold, so replaying the stored (JSON-safe) actions reproduces them.
- The trajectory stores each action as received, made JSON-safe: non-finite floats, numbers that ``float()`` cannot
  convert (an int from about 1.8e308, a huge ``Fraction``) and containers nested deeper than 16 levels become ``None``,
  except that a ``flows``, ``overrides`` or ``hold`` value becomes the string ``"<TypeName>"`` of its type instead, so
  the field stays present and malformed on replay (M1 pre-gate round 2, INT-3); numpy scalars and arrays
  Python numbers, strs and lists (a str subclass, NumPy's included, its plain text); tuples lists; mapping keys strings
  (a str key its text, an int or bool key with a float value its decimal integer, any other key the string
  ``"<TypeName>"``); any other object the string ``"<TypeName>"`` of its type, never its ``repr``. An in-process
  reply that cannot be copied because a conversion raises (a ``numbers.Real`` whose ``float()`` raises ValueError, a
  Mapping whose iteration raises, anywhere in the reply) is stored whole as the string ``"<TypeName>"`` of the reply:
  not a dict, a whole-week failure, so ``step`` plays the fallback instead of raising and the stored copy fails the
  same way on replay (M1 pre-gate 3, INT-3). A whole-week failure is also logged as the first entry of
  ``StepRecord.invalid``.
- On a whole-week failure the environment plays ``fallback(obs)`` (the naive action from the policy's own observation
  of that week, D9, rebuilt from the unchanged state) when a fallback was given, else the empty action; a fallback
  that raises or itself fails validation is replaced by the empty action, and that failure is the second entry of
  ``StepRecord.invalid`` (only the exception's type when the fallback raised, so the record holds no arbitrary text).
- A snapshot carries the episode's fallback spec (``"naive"``, ``"naive_plain"``, a ``NaiveFallback``, None or the
  callable) and ``restore`` re-installs it for the restored episode, so a snapshot restored into an ``Env`` built with
  another fallback continues bit-identically (B2); a later ``reset`` installs the Env's own fallback again, so the same
  reset and policy give the same trajectory whether or not the Env restored a snapshot before (M1 pre-gate 3,
  DET-P3-3).
- The naive fallback is the end-aware naive (``policies.naive``, Q98; §9.3), the anchor's rule: ``"naive"`` and a
  ``NaiveFallback`` (``end_aware`` True by default). ``"naive_plain"`` and a ``NaiveFallback`` with
  ``end_aware=False`` play plain naive, the horizon-blind rule without the cut (tests and evidence only). The cut reads
  the observation's week alone, so a masked or blanked observation (M3) changes it no more than it changes naive.
- Naive's F_Q in the fallback (§8.1, §9.3): naive is a function of the instance and the generator only, so the
  fallback of a generated omega (``Omega.generated``: it stores regime paths) is naive with that generator's
  quantiles, received as the spec ``NaiveFallback`` (``policies.naive_fq.naive_fallback(inst, params)``, which carries
  the ``generator_id`` they belong to and the content digest of the instance they were computed for). ``reset`` refuses
  a ``NaiveFallback`` computed for other instance content (its ``instance_digest`` is not the instance's
  ``family_digest``, its content at its file's own rung, which F_Q reads, since it reads no stock (§12 'Warm start per
  rung'): a variant of other content keeps the hash label that ``generator_id`` reads, REG-M2R2-01; a spec
  built by hand without a digest binds no content and is checked by its generator only), one whose ``generator_id`` is
  not omega's (or that of omega's event-free twin (57)), a ``NaiveFallback`` on marks that record no generator
  (hand-built marks), and a string spec (``"naive"``, ``"naive_plain"``) on a generated omega, whose point mass at 0
  is the rule of injected lists only; a string spec on an injected list (or on marks without a generator) keeps the
  point mass at 0, so the frozen cents do not move. The guard reads the generator the marks record
  (``WeeklyMarks.generator_id`` and ``generated``, set by ``compute_marks`` from omega), so a reset from
  ``compute_marks(instance, omega)`` alone is checked exactly as one from omega (INT-M2-R1-1).
- ``reset`` and ``restore`` are atomic: the initial state, the observation, the Static tables and the fallback (naive's
  plan included) are built before anything is installed, so one that raises leaves the environment as it was
  (M1 pre-gate round 2, DET-P2-3).
- Observations are built fresh by ``observe`` and never aliased to the state, so the environment hands them out
  without copying and keeps none.
- One observation path (M3; design §5.4 "Rules"): ``reset`` resolves ``regime`` (a registry name or a theta of (47),
  ``information.theta.resolve_regime``), builds the episode's information view once (``information.view.build_view``,
  from omega and its marks, never the state), and every observation handed out, at reset, each step, to the D9
  fallback and after restore, is ``Env._observe(ep) = wrap_owned(observe(ep.inst, ep.marks, ep.state), ep.view,
  ep.inst)`` (``information.observe.wrap`` filling the fresh core observation in place), so the fallback sees exactly
  the policy's observation (under blackout the blanked one; Q71 X9). The view is shared
  by the episode's copies and snapshots like the instance and the marks, so ``restore`` rebuilds nothing from omega.
  ``Trajectory.regime`` records ``theta.regime_label`` (the registry name, or a custom theta's name and digest), and
  ``info["omega"]`` carries the omega payload under a theta that sees omega at reset (clairvoyant) only, built afresh
  (``wire.omega_payload``) from the view's public omega at each reset and restore.
- A step whose simulation, charge or observation raises marks the episode failed and re-raises: a valid action never
  makes the simulator raise (§9.3), so the exception is a bug, and the simulator may already have updated part of the
  week in place (shipments, lots, stocks). Later ``step`` and ``snapshot`` calls raise ``RuntimeError`` until ``reset``
  or ``restore`` (a snapshot taken before the failure continues bit-identically). Wherever the step failed (simulation,
  charge, terminal credit or observation), ``trajectory`` stores the failed week's action (as received, JSON-safe) as
  its last action and marks the week in ``Trajectory.failed``, while its records and terminal credit hold only the
  weeks completed before it; so replaying the stored actions in a fresh ``Env`` reproduces the failure, in the same
  week (M1 pre-gate 3, DET-P3-5). Validation and the fallback change nothing, so an exception there leaves the episode
  as it was and stores nothing.
"""

import copy
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shockbench_flow.dynamics.observe import observe, static_view
from shockbench_flow.dynamics.sim import initial_state, step, terminal_salvage
from shockbench_flow.dynamics.state import State, Trajectory, cents
from shockbench_flow.information.observe import wrap_owned
from shockbench_flow.information.theta import Theta, regime_label, resolve_regime
from shockbench_flow.information.view import InformationView, build_view
from shockbench_flow.information.wire import WIRE_FAILURES, omega_payload
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import WeeklyMarks, compute_marks
from shockbench_flow.omega.container import Omega, load_omega
from shockbench_flow.omega.injected import event_free_generator_id
from shockbench_flow.policies.base import empty_action, reset_policy
from shockbench_flow.policies.naive import NaiveFallback, NaivePolicy


def _is_int(x) -> bool:
    return type(x) is int or (isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_)))


def _is_real(x) -> bool:
    t = type(x)
    return t is float or t is int or (isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_)))


def _finite_real(x) -> bool:
    """A real number (not bool) that converts to a finite float; an int beyond the float range is not (§9.3)."""
    if not _is_real(x):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _entries(action: Mapping, name: str, keys: tuple[str, str]) -> list[tuple]:
    """The paired entries of one action field; a missing or null field has none (§9.3).

    Raises:
        ValueError: the field is present but not a dict of equal-length lists under ``keys``, so the reply cannot be
            paired into entries: a whole-week failure (design §12 "Whole-week failures"; D9).

    """
    field = action.get(name)
    if field is None:
        return []
    if not isinstance(field, Mapping) or any(not isinstance(field.get(k), (list, tuple)) for k in keys):
        raise ValueError(f"{name}: not {{{keys[0]!r}: [...], {keys[1]!r}: [...]}}")
    a, b = field[keys[0]], field[keys[1]]
    if len(a) != len(b):
        raise ValueError(f"{name}: {keys[0]} and {keys[1]} differ in length")
    return list(zip(a, b))


def validate_action(inst: Instance, marks: WeeklyMarks, week: int, action: dict):
    """Apply the validity rules of §9.3 (Q68, Q86): the whole-week checks first, then the entry checks.

    Non-finite or negative qty, out-of-range slot, duplicate slot, a slot whose own (edge, k) is in Z_t (an override
    slot's (out-edge, k)): the entry is dropped and logged ("masked this week"); the rest stands. ``slot_mask`` is
    stricter since Q111 (any edge of a lane slot's lane), and a lane slot masked only for a later edge is not dropped.
    A dropped override or hold entry is ignored (the default release stays on unless another valid override or hold for
    (c, k) remains).

    Returns:
        (flows, overrides, holds, invalid): slot -> qty dicts, a frozenset of (chokepoint node, k), and log strings.

    Raises:
        ValueError: a whole-week failure, which takes the D9 fallback (design §12 "Whole-week failures"): the action
            is not a dict; ``week`` is missing, not an integer (a bool is not) or not ``week``; ``flows`` or
            ``overrides`` is present (not None) but not a dict whose ``slot`` and ``qty`` are lists of equal length;
            ``hold`` is present but not a dict whose ``chokepoint`` and ``k`` are lists of equal length. A missing or
            null ``flows`` is an empty request; a missing or null ``overrides`` or ``hold`` is none.

    """
    if not isinstance(action, Mapping):
        raise ValueError("the action is not a dict")
    w = action.get("week")
    if not _is_int(w) or int(w) != week:
        raise ValueError(f"the action's week must be {week}")
    Z = marks.prohibited[week - 1]
    flow_entries = _entries(action, "flows", ("slot", "qty"))
    override_entries = _entries(action, "overrides", ("slot", "qty"))
    hold_entries = _entries(action, "hold", ("chokepoint", "k"))
    invalid: list[str] = []

    def slot_entries(name: str, entries: list[tuple], table: list, edge_k) -> dict[int, float]:
        out: dict[int, float] = {}
        seen: set[int] = set()
        for i, (slot, qty) in enumerate(entries):
            where = f"{name}[{i}]"
            if not _is_int(slot):
                invalid.append(f"{where}: slot is not an integer index")
                continue
            if not 0 <= slot < len(table):
                invalid.append(f"{where}: slot out of range")  # the value is not echoed: it may be a huge int
                continue
            s = int(slot)
            if s in seen:
                invalid.append(f"{where}: duplicate slot {s}")
                continue
            seen.add(s)
            if not _finite_real(qty):
                invalid.append(f"{where}: slot {s} qty is not a finite number")
                continue
            if float(qty) < 0:
                invalid.append(f"{where}: slot {s} qty is negative")
                continue
            e, k = edge_k(table[s])
            if Z[e, k]:
                invalid.append(f"{where}: slot {s} masked this week")
                continue
            out[s] = float(qty)
        return out

    flows = slot_entries("flows", flow_entries, inst.action_slots, lambda a: (a[0], a[1]))
    overrides = slot_entries("overrides", override_entries, inst.override_slots, lambda o: (o[2], o[1]))
    chk = set(inst.chokepoints)
    holds: set[tuple[int, int]] = set()
    for i, (c, k) in enumerate(hold_entries):
        where = f"hold[{i}]"
        if not _is_int(c) or int(c) not in chk:
            invalid.append(f"{where}: not a chokepoint node")
        elif not _is_int(k) or not 0 <= int(k) < len(inst.commodities) or not inst.commodities[int(k)].override:
            invalid.append(f"{where}: not tanker cargo (K^ov)")
        elif (int(c), int(k)) in holds:
            invalid.append(f"{where}: duplicate hold")
        else:
            holds.add((int(c), int(k)))
    return flows, overrides, frozenset(holds), tuple(invalid)


_MAX_DEPTH = 16  # a well-formed action nests 3 levels; anything deeper is not a valid entry (§9.3)


def _has_float(i: int) -> bool:
    """Whether an int converts to a finite float; ``float()`` overflows from 2**1024 - 2**970 (about 1.8e308) up."""
    try:
        float(i)
    except OverflowError:
        return False
    return True


def _json_key(k) -> str:
    """A mapping key as a JSON object key that never carries an object's ``str`` (which may hold a memory address).

    A str key keeps its text (a str subclass, such as a ``StrEnum`` member or a NumPy str, its plain text); an int or
    bool key with a finite float value becomes its decimal integer (``True`` -> ``"1"``); any other key becomes
    ``"<TypeName>"``.
    """
    if isinstance(k, str):
        return str.__str__(k)
    if isinstance(k, (bool, np.bool_, numbers.Integral)):
        i = int(k)
        if _has_float(i):
            return str(i)
    return f"<{type(k).__name__}>"


def _json_safe(x, depth: int = 0):
    """A JSON-safe copy of a received action; validation runs on this copy, so stored-action replay is exact (B2).

    Non-finite floats and integers without a finite float value (``float()`` overflows) become None; numpy values
    become Python values (a NumPy str its plain text, as any str subclass); containers nested deeper than
    ``_MAX_DEPTH`` become None; mapping keys become strings by ``_json_key``; any other object becomes the string
    ``"<TypeName>"`` of its type.
    """
    t = type(x)
    if t is float:
        return x if math.isfinite(x) else None
    if t is int:
        return x if _has_float(x) else None
    if t is str or t is bool or x is None:
        return x
    if isinstance(x, str):
        return str.__str__(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (Mapping, list, tuple, np.ndarray)) and depth >= _MAX_DEPTH:
        return None
    if isinstance(x, Mapping):
        return {_json_key(k): _json_safe(v, depth + 1) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist(), depth)
    if isinstance(x, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in x]
    if isinstance(x, numbers.Integral):
        return _json_safe(int(x), depth)
    if isinstance(x, numbers.Real):
        try:
            f = float(x)
        except OverflowError:  # no finite float value (a huge Fraction), like an int from 2**1024 - 2**970
            return None
        return _json_safe(f, depth)
    return f"<{type(x).__name__}>"


_FIELDS = ("flows", "overrides", "hold")  # the action fields that hold entries (§9.2)


def _json_safe_action(action):
    """The JSON-safe copy of a received action (``_json_safe``), with its entry fields kept present when malformed.

    A ``flows``, ``overrides`` or ``hold`` value that is not None but whose copy would be None (a non-finite float, a
    number without a finite float value) becomes the string ``"<TypeName>"`` of that value instead: the field is
    present and cannot be paired into entries, a whole-week failure (design §12 "Whole-week failures"), and the stored
    copy stays one on replay. Nulling it would turn a malformed field into a missing one, neither substituted nor logged
    (M1 pre-gate round 2, INT-3).
    """
    if not isinstance(action, Mapping):
        return _json_safe(action)
    out = {}
    for k, v in action.items():
        key, safe = _json_key(k), _json_safe(v, 1)
        if safe is None and v is not None and key in _FIELDS:
            safe = f"<{type(v).__name__}>"
        out[key] = safe
    return out


def received_copy(action: object) -> object:
    """The JSON-safe copy of ``action`` that ``Env.step`` stores and validates (B2): the one rule for it.

    ``_json_safe_action`` of the action, or the string ``"<TypeName>"`` of its type when the conversion raises (the
    reply cannot be read: a whole-week failure, INT-3): non-finite floats and ints without a finite float value null,
    NumPy values as Python values, mapping keys by ``_json_key`` (a bool key ``"1"``, a float key ``"<float>"``),
    containers deeper than 16 levels null, any other object ``"<TypeName>"``. A tree of exactly the JSON types, str keys
    and finite floats, at most 16 levels deep. The wire's policy side sends this copy (``information.runner``), so the
    Env's copy of the decoded line is the same copy again and the wire moves nothing (W2).
    """
    try:
        return _json_safe_action(action)
    except Exception:  # noqa: BLE001 - a conversion raised: the reply cannot be read, a whole-week failure (INT-3)
        return f"<{type(action).__name__}>"


@dataclass
class _Episode:
    inst: Instance
    marks: WeeklyMarks
    view: InformationView  # the regime's reset-time feeds (information.view), immutable and shared like the marks
    state: State
    traj: Trajectory
    done: bool = False
    failed: str | None = None  # "week t (ExceptionType)" once a step raised: the state may be half-updated

    def copy(self) -> "_Episode":
        """A deep copy that shares the (immutable) instance, marks and information view."""
        return copy.deepcopy(
            self, memo={id(self.inst): self.inst, id(self.marks): self.marks, id(self.view): self.view}
        )


def _check_not_failed(ep: _Episode) -> None:
    """Refuse to go on from a failed step, whose state may be half-updated (M1 pre-gate DET-1)."""
    if ep.failed is not None:
        raise RuntimeError(f"the episode failed in {ep.failed}: reset, or restore a snapshot taken before the failure")


FallbackSpec = Callable[[dict], dict] | NaiveFallback | str | None
WHOLE_WEEK_FAILURE = "whole-week failure"  # the first ``StepRecord.invalid`` line of a week the D9 fallback replaced


def took_fallback(record) -> bool:
    """Whether the week of ``record`` took the D9 fallback (§9.3): its invalid log starts with the whole-week failure.

    ``Env.step`` writes that line first in every such week (the fallback's own lines follow it).
    """
    return bool(record.invalid) and record.invalid[0].startswith(WHOLE_WEEK_FAILURE)


NAIVE_SPECS = ("naive", "naive_plain")  # the end-aware naive (Q98) and plain naive, F_Q the point mass at 0


def _is_naive(spec: FallbackSpec) -> bool:
    """Naive, built from the instance at reset: a spec of ``NAIVE_SPECS`` (the point mass at 0) or a NaiveFallback."""
    return isinstance(spec, NaiveFallback) or (isinstance(spec, str) and spec in NAIVE_SPECS)


def _check_fallback_spec(spec: FallbackSpec) -> None:
    """Raise ValueError unless ``spec`` is a spec of ``NAIVE_SPECS``, a ``NaiveFallback``, a callable or None."""
    if not (spec is None or _is_naive(spec) or callable(spec)):
        raise ValueError(f"fallback must be one of {NAIVE_SPECS}, a NaiveFallback, a callable or None, got {spec!r}")


def _check_fallback(spec: FallbackSpec, instance: Instance, marks: WeeklyMarks) -> None:
    """The fallback's F_Q belongs to the episode's instance and generator (§8.1): the checks of the module docstring.

    The generator is the one ``marks`` record (``compute_marks`` copies omega's ``meta_generator_id`` and
    ``Omega.generated``), so both reset paths, from omega and from its marks alone, run this one guard. The instance is
    matched by content (``Instance.family_digest``, its content at its file's own rung: F_Q reads no stock, so a
    fallback computed on the file serves the episode at every rung, §12 'Warm start per rung'), since ``generator_id``
    reads only the hash label, which a ``dataclasses.replace`` variant of other content carries over (REG-M2R2-01).

    Raises:
        ValueError: a ``NaiveFallback`` whose ``instance_digest`` is not the instance's family digest, on marks that
            record no generator or for another generator, or a string spec on the marks of a generated omega.

    """
    if isinstance(spec, NaiveFallback):
        if spec.instance_digest is not None and spec.instance_digest != instance.family_digest:
            raise ValueError(
                f"the fallback's F_Q was computed for instance content {spec.instance_digest[:12]}..., this instance "
                f"holds {instance.family_digest[:12]}... (naive is a function of the instance's content and the "
                "generator, §8.1; the hash label alone does not identify it)"
            )
        if not marks.generator_id:
            raise ValueError(
                "a NaiveFallback needs the episode's generator, to check whose F_Q it carries (§8.1): these marks "
                "record no generator (pass omega, or its compute_marks)"
            )
        if marks.generator_id not in (spec.generator_id, event_free_generator_id(spec.generator_id)):
            raise ValueError(
                f"the fallback's F_Q belongs to generator {spec.generator_id[:12]}..., omega was drawn by "
                f"{marks.generator_id[:12]}... (naive is a function of the instance and the generator, §8.1)"
            )
    elif _is_naive(spec) and marks.generated:  # "naive" or "naive_plain": a NaiveFallback took the branch above
        raise ValueError(
            "a generated omega needs naive with its generator's F_Q (§8.1, §9.3): pass fallback="
            f"policies.naive_fq.naive_fallback(instance, params); {spec!r} is the point mass at 0 of injected lists"
        )


def _build_fallback(spec: FallbackSpec, static: dict, obs: dict) -> Callable[[dict], dict] | None:
    """The episode's D9 fallback: naive is stateless, so one policy built from the instance serves every week.

    ``"naive"`` is the end-aware naive (Q98) with F_Q the point mass at 0 (injected lists); a ``NaiveFallback`` is naive
    with its generator's quantiles (§8.1), end-aware unless its ``end_aware`` is False; ``"naive_plain"`` is plain naive
    with F_Q the point mass at 0 (tests and evidence only).

    Raises:
        Exception: whatever naive's reset raises (an instance whose plan (22) cannot be solved); the caller installs
            nothing then (reset and restore are atomic).

    """
    if _is_naive(spec):
        generated = isinstance(spec, NaiveFallback)
        anchor = NaivePolicy(
            fq_quantile=spec.fq_quantile if generated else None,
            end_aware=spec.end_aware if generated else spec == "naive",
        )
        anchor.reset(static, obs, 0)
        return anchor.act
    return spec


@dataclass(frozen=True)
class _Snapshot:
    episode: _Episode  # a private deep copy, copied again on every restore
    fallback: FallbackSpec = "naive"  # the Env's fallback spec, re-installed on restore (B2)


class Env:
    """One episode at a time on the trusted side; never exposes omega or the simulator to the policy."""

    def __init__(self, fallback: FallbackSpec = "naive") -> None:
        """Create an environment.

        Args:
            fallback: the D9 fallback, called with the policy's own observation of the week on a whole-week failure.
                ``"naive"`` (default) is §9.3's rule, the end-aware naive's action (Q98) computed from that
                observation, with F_Q the point mass at 0 (injected lists; a generated omega, or its marks, then raises
                at reset). A ``NaiveFallback`` (``policies.naive_fq.naive_fallback(instance, params)``) is the same
                rule with the generator's F_Q, for omegas of that generator and their marks (§8.1). ``"naive_plain"``
                (and a ``NaiveFallback`` with ``end_aware=False``) is plain naive, without the end-aware cut (tests and
                evidence only). A callable replaces it (tests); None plays the empty action.

        """
        _check_fallback_spec(fallback)
        self._ep: _Episode | None = None
        # the Env's own rule, which every reset installs and nothing changes; and the running episode's rule, the Env's
        # own after reset and the snapshot's after restore, which a snapshot carries (DET-P3-3)
        self._fallback_spec = self._episode_spec = fallback
        # naive (a NaiveFallback included) is built from the instance at reset
        self.fallback: Callable[[dict], dict] | None = None if _is_naive(fallback) else fallback

    def reset(
        self,
        instance: Instance,
        regime: str | Theta = "prediction_free",
        omega: Omega | str | Path | None = None,
        policy_seed: int = 0,
        *,
        marks: WeeklyMarks | None = None,
        policy_name: str = "",
    ) -> tuple[dict, dict]:
        """Start an episode and return (obs, info).

        ``omega`` is an omega container or its `.npz` path; ``marks`` may be passed instead (fixture tests) or beside
        it (then its hashes must match). ``obs`` is the Reset payload's observation and ``info["static"]`` the Static
        tables. The episode starts from the block of the warm start per rung that omega was drawn on (§2.3; M5-O37
        (b)): ``instance`` may be the file's at any rung, and the episode plays ``instance.at_digest`` of the marks'
        instance digest, which is ``instance`` itself on `tiny` and whenever the digests already agree.

        ``regime`` is a registry name or a theta of (47) (``information.theta.resolve_regime``); the view is built
        once here by ``information.view.build_view`` and every observation goes through ``information.observe.wrap``
        (``_observe``), with ``info["omega"]`` (``wire.omega_payload``) under a theta that sees omega at reset
        (clairvoyant) only; ``Trajectory.regime`` is ``theta.regime_label(theta)``, the registry name or a custom
        theta's name and digest.

        Raises:
            ValueError: on an unknown regime, marks that are not omega's own or belong to another instance, neither
                omega nor marks, a fallback whose F_Q does not belong to the instance's content or to the generator
                of omega or of the marks passed alone (§8.1; module docstring), or a regime whose feeds need what
                omega (or a marks-only reset) lacks (``build_view``).
            TypeError: if ``regime`` is neither a registry name nor a theta.

        """
        theta = resolve_regime(regime)
        om = None
        if omega is not None:
            if isinstance(omega, Omega):
                om = omega
            elif isinstance(omega, Mapping):
                om = Omega(dict(omega))
            else:
                om = load_omega(omega)
            own = compute_marks(instance, om)
            if marks is not None and marks.digest != own.digest:  # V3: the marks must be omega's own, not a lookalike
                raise ValueError("the marks passed are not compute_marks(instance, omega) (V3)")
            marks = own
        if marks is None:
            raise ValueError("reset needs omega or marks")
        instance = instance.at_digest(marks.instance_digest)  # omega's rung: its warm start block (§2.3; M5-O37 (b))
        _check_fallback(self._fallback_spec, instance, marks)  # omega's own marks, or the marks passed alone
        if (
            marks.instance_hash != instance.hash
            or marks.T != instance.T
            or marks.instance_digest != instance.content_digest
        ):
            raise ValueError("the marks belong to another instance (V3)")
        view = build_view(instance, om, theta, marks=marks)  # once per episode (information.view)
        state = initial_state(instance)
        traj = Trajectory(
            instance_hash=instance.hash,
            omega_hash=marks.omega_hash,
            policy=policy_name,
            regime=regime_label(theta),
            instance_digest=instance.content_digest,
            marks_digest=marks.digest,
        )
        ep = _Episode(instance, marks, view, state, traj)
        obs = self._observe(ep)
        static = static_view(instance, theta, view.dyads)
        fallback = _build_fallback(self._fallback_spec, static, obs)
        info = {"static": static, "policy_seed": policy_seed}
        if view.omega_public is not None:  # full omega at reset (§5.1), local and dev runs only; a fresh payload
            info["omega"] = omega_payload(view.omega_public)
        # atomic: everything that can raise is built above, so a failed reset leaves the environment as it was
        self._ep, self.fallback = ep, fallback
        self._episode_spec = self._fallback_spec  # a new episode runs under the Env's own rule, never a snapshot's
        return obs, info

    def step(self, action: dict, *, wire_failure: str | None = None) -> tuple[dict, float, bool, bool, dict]:
        """Validate, simulate one week, charge, and observe the next week (or the final state at T).

        ``wire_failure`` (M3; the runner's only, ``information.runner``) is a code of
        ``information.wire.WIRE_FAILURES``: the week is a whole-week failure whatever ``action`` holds (the D9
        fallback, §9.3 row 2; ``action`` is never validated), its invalid log starts with 'whole-week failure: ' and
        the code's reason, the stored action is ``action`` made JSON-safe, and ``Trajectory.wire_failures`` records
        (week, code) beside it (on a failed step too), so replay reproduces it and no reply value can produce it.
        None (the default) is today's step.

        Raises:
            RuntimeError: no episode was started, it is finished, or an earlier step failed.
            ValueError: ``wire_failure`` is not a code of ``WIRE_FAILURES``.
            Exception: whatever the simulator, the charge or the observation raised (a bug: a valid action never makes
                the simulator raise, §9.3); the episode is then marked failed, since the simulator may have updated
                part of the week in place.

        """
        if wire_failure is not None and not (isinstance(wire_failure, str) and wire_failure in WIRE_FAILURES):
            raise ValueError(f"wire_failure must be None or a code of {tuple(WIRE_FAILURES)}, got {wire_failure!r}")
        ep = self._ep
        if ep is None:
            raise RuntimeError("call reset before step")
        _check_not_failed(ep)
        if ep.done:
            raise RuntimeError("the episode is finished")
        inst, marks = ep.inst, ep.marks
        week = ep.state.week + 1
        received, flows, overrides, holds, invalid, info = self._decide(ep, week, action, wire_failure)  # unchanged
        terminated = week == inst.T
        try:
            rec = step(inst, marks, ep.state, flows, overrides, holds, invalid)
            reward_cents = -rec.cost_cents
            if terminated:
                S = terminal_salvage(inst, ep.state)
                S_cents = cents(S)
                reward_cents += S_cents
            obs = self._observe(ep)
        except BaseException as err:  # one rule wherever the step failed: the action stored, no record (DET-P3-5)
            ep.failed = ep.traj.failed = f"week {week} ({type(err).__name__})"
            self._store(ep, week, received, wire_failure)
            raise
        self._store(ep, week, received, wire_failure)
        ep.traj.records.append(rec)
        if terminated:
            ep.traj.salvage, ep.traj.salvage_cents = S, S_cents
            ep.done = True
            info["salvage"], info["salvage_cents"] = S, S_cents
        clipped = [s for s in sorted(rec.requested) if rec.executed.get(s, 0.0) < rec.requested[s]]
        info.update(
            cost_components=rec.costs.as_dict(),
            clipped={
                "slot": clipped,
                "requested": [rec.requested[s] for s in clipped],
                "executed": [rec.executed.get(s, 0.0) for s in clipped],
            },
            invalid=list(rec.invalid),
            clamps=[list(c) for c in rec.clamps],
            edge_clamps=[list(c) for c in rec.edge_clamps],
            override_clamps=[list(c) for c in rec.override_clamps],
            reward_cents=reward_cents,
        )
        return obs, reward_cents / 100, terminated, False, info

    @staticmethod
    def _observe(ep: _Episode) -> dict:
        """The one observation path: the core observation of the episode's state, wrapped by its regime's view.

        ``observe`` builds a fresh dict that aliases nothing, so ``wrap_owned`` fills it in place (``wrap`` less its
        copy).
        """
        return wrap_owned(observe(ep.inst, ep.marks, ep.state), ep.view, ep.inst)

    @staticmethod
    def _store(ep: _Episode, week: int, received, wire_failure: str | None) -> None:
        """Append the week's stored action, and its wire failure when the runner gave one (replay reads both)."""
        ep.traj.actions.append(received)
        if wire_failure is not None:
            ep.traj.wire_failures.append((week, wire_failure))

    def _decide(self, ep: _Episode, week: int, action: dict, wire_failure: str | None = None) -> tuple:
        """The week's JSON-safe action and what is simulated: the validated action, else the D9 fallback (§9.3).

        A ``wire_failure`` code skips the validation: the week is a whole-week failure with the code's reason.

        Returns:
            (received, flows, overrides, holds, invalid, info); nothing of the episode is changed.

        """
        inst, marks = ep.inst, ep.marks
        info: dict = {}
        received = received_copy(action)  # stored as received, and validated in this form so replay is exact (B2)
        try:
            if wire_failure is not None:  # the runner's cause, never a reply value (design §12 "Wire exchange")
                raise ValueError(WIRE_FAILURES[wire_failure])
            flows, overrides, holds, invalid = validate_action(inst, marks, week, received)
        except ValueError as err:
            used = "fallback" if self.fallback is not None else "empty action"
            info["fallback"] = f"week {week}: whole-week failure ({err}); {used} played (D9)"
            failed = None  # the record's line for a failing fallback: validate_action's message, else the type only
            try:
                if self.fallback is not None:  # the week's observation, rebuilt from the state not yet stepped
                    fb = self.fallback(self._observe(ep))
                else:
                    fb = empty_action(week)
            except Exception as err2:  # noqa: BLE001 - a failing fallback degrades to the empty action, logged (D9)
                failed, detail = f"the fallback raised {type(err2).__name__}", f"{type(err2).__name__}: {err2}"
            else:
                try:
                    flows, overrides, holds, fb_invalid = validate_action(inst, marks, week, fb)
                except Exception as err2:  # noqa: BLE001 - as above
                    own = isinstance(err2, ValueError)  # a whole-week failure: a fixed message with no reply value
                    failed = f"the fallback failed validation ({err2 if own else type(err2).__name__})"
                    detail = f"{type(err2).__name__}: {err2}"
            if failed is not None:  # logged in info and in the record, after the whole-week failure (D9)
                info["fallback"] += f"; the fallback failed too ({detail}), empty action played"
                flows, overrides, holds, fb_invalid = {}, {}, frozenset(), (f"{failed}, empty action played",)
            invalid = (f"{WHOLE_WEEK_FAILURE}: {err}",) + tuple(fb_invalid)
        return received, flows, overrides, holds, invalid, info

    @property
    def trajectory(self) -> Trajectory:
        """The running episode's trajectory (actions as received, records, terminal credit at T)."""
        if self._ep is None:
            raise RuntimeError("call reset first")
        return self._ep.traj

    def snapshot(self) -> object:
        """An opaque deep copy of the episode state with the episode's fallback spec, for mid-episode resume (B2).

        The spec is the running episode's (the one a restore installed, else the Env's own), kept by reference; a
        snapshot whose fallback is a callable pickles only if the callable does.
        """
        ep = self._ep
        if ep is None:
            raise RuntimeError("call reset first")
        _check_not_failed(ep)
        return _Snapshot(ep.copy(), self._episode_spec)

    def restore(self, snapshot: object) -> tuple[dict, dict]:
        """Resume from a snapshot under its fallback spec; the continuation is bit-identical (B2).

        The spec serves the restored episode only: a later ``reset`` installs the Env's own fallback again (DET-P3-3).
        Returns (obs, info) as ``reset`` does: the restored week's observation and ``info`` with ``static`` and, under
        a theta that sees omega at reset, ``omega``; the policy seed is the caller's (a snapshot holds none). The pair
        is what ``information.runner.play_wire_episode(resume=...)`` takes.
        """
        if not isinstance(snapshot, _Snapshot):
            raise TypeError("not a snapshot of this environment")
        spec = snapshot.fallback
        _check_fallback_spec(spec)
        ep = snapshot.episode.copy()
        obs = self._observe(ep)
        static = static_view(ep.inst, ep.view.theta, ep.view.dyads)
        fallback = _build_fallback(spec, static, obs)
        info = {"static": static}
        if ep.view.omega_public is not None:
            info["omega"] = omega_payload(ep.view.omega_public)
        self._episode_spec, self._ep, self.fallback = spec, ep, fallback  # atomic, as in reset
        return obs, info


def rollout(
    instance: Instance,
    policy,
    omega: Omega | str | Path | None = None,
    regime: str | Theta = "prediction_free",
    policy_seed: int = 0,
    *,
    marks: WeeklyMarks | None = None,
    fallback: FallbackSpec = "naive",
) -> Trajectory:
    """Run one episode of ``policy`` (``policies.base.Policy``) and return its finished trajectory.

    ``regime`` is a registry name or a theta of (47) (``Env.reset``). The policy's reset gets the keyword ``omega``
    exactly when it declares ``wants_omega`` (``policies.base``), the one rule ``information.runner.serve`` applies
    too: the payload under clairvoyant, None in every other regime.
    """
    env = Env(fallback=fallback)
    obs, info = env.reset(instance, regime, omega, policy_seed, marks=marks, policy_name=getattr(policy, "name", ""))
    reset_policy(policy, info["static"], obs, policy_seed, info.get("omega"))  # omega= iff wants_omega (policies.base)
    done = False
    while not done:
        obs, _reward, done, _truncated, _info = env.step(policy.act(obs))
    return env.trajectory

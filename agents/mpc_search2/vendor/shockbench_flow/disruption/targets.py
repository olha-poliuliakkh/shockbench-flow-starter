"""Event types and targets per region: the type law and target rules of design §4.4 and the §2.4 `tiny` profile.

One module decides, for an event of block B in region m at a week with regime (z^c, z^p), which types are feasible,
with which weights, and which targets each type can hit. The cluster sampler (``hawkes``) reads it for the immigrant
intensity of the militarised block, whose closure part is modulated by the chokepoints' latent risk (Q51); the event
stage (``events``) reads it to type every event. Both therefore agree by construction.

Rules (design §2.4 profile, "Type laws per block" and "Targets"):
- a type with no target in the region is infeasible; a regional conflict is feasible exactly in the war state
  (z^c = 2), where it always has a candidate (below); feasible weights are renormalised; a block with no feasible type
  leaves its event without a graph operation (a no-op event: it keeps its place in the genealogy of (30), but omega
  does not store it);
- targets: closure and piracy an adjacent chokepoint (uniform); tariff (importer m, exporter uniform among regions
  with a non-coupling edge into m, then commodity uniform among the commodities of that exporter's edges into m);
  sanction an edge from m into another region (uniform, all of K_e); material outage a material node in m; energy
  shock a grid in m; regional conflict its own region with a counterpart: a partner of a dyad at war with m in the
  week (uniform among several, Q59), which takes precedence, else a region drawn with weight A_{m m'}, else none
  (counterpart -1, Q94 (f)). Tariff and sanction targets count an edge's commodities less its pairs prohibited from
  week 0 (Z_0 of (3), in force all episode; `tiny`'s E7), since an event on such a pair has no effect (reported
  reading).
- Q114, the sanction target rules of `small` and `full` (design §2.4 "Targets" and "Type laws per block", §12 row
  "Sanction target rules V3 (Q114)"), read from ``params.targeting`` (``params.TargetingRules``; None, `tiny`'s, keeps
  the rules above): V1 drops every sanction candidate whose tail is a chokepoint (a transit leg is not an export of the
  chokepoint's coastal region); V2 keeps the policy block's (block P) infeasible share as one no-op part (-1) after the
  feasible types, each at its raw share over the block's full share total, instead of renormalising (a block with no
  feasible type stays one no-op event). The militarised block and the tariff rule are unchanged.
- A regional conflict's candidates are its counterparts: the partners of the dyads at war with m in the week, else
  the regions with A_{m m'} > 0, else (a war-state region with neither, Q94 (f)) the single target (m, counterpart
  -1), which acts through its restoration (13) and war-risk class only, with no (40) edge operation (``marks``). A
  dyad war thus gives a region without trade partners its conflict against the dyad partner (CMP-1), and a war of its
  own chain a partnerless one. The regime state of a (region, column) is ``TypeRules.state``: the effective z^c, which
  sets the feasible types, and those dyad partners, which name the counterpart.

**One immigrant-parts rule** (30), Q51, composed once, in ``TypeRules``: ``TypeRules.week_parts`` gives the typed
parts of a (block, region, week), read at the week's column t + B_burn of the regime paths: the effective conflict
layer z^c (dyad wars applied) sets the feasible types (``TypeRules.parts``, the memoised ``immigrant_parts``; since a
regional conflict at war always has a candidate, Q94 (f), neither the week's dyad partners nor A change them), and
every militarised-closure part aimed at chokepoint c is multiplied by g(X^t_c)/c_X of the week
(``closure_modulation``, applied by ``modulate``). ``TypeRules.week_totals`` is the ``math.fsum`` of those shares per
week, and the cluster sampler's immigrant intensity is lambda^0 of the week's regime times it
(``hawkes.immigrant_rates``); the event stage types an immigrant by the same parts at the week of its key (block,
region, t + B_burn, rank), so the intensity and the type law agree by construction. The module's ``week_parts`` is
``TypeRules.week_parts`` for one call. The adjacency A has one rule too (``adjacency_dict``): the params' A when set,
else the instance's. A child event belongs to its onset week t = floor(s) + 1.
"""

import math
from dataclasses import dataclass

import numpy as np

from shockbench_flow.disruption.params import GeneratorParams, TargetingRules
from shockbench_flow.disruption.regime import WAR, tilt
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import K_CHOKEPOINT, K_EDGE, K_NODE, K_REGION, MILITARISED_CLOSURE
from shockbench_flow.omega import codes


CHOKEPOINT, EDGE, NODE, REGION = K_CHOKEPOINT, K_EDGE, K_NODE, K_REGION  # codes.TARGET_KINDS, defined in ``marks``
POLICY_BLOCK = 0  # block P (``hawkes.P_BLOCK``, which imports this module), the block V2's no-op rule reads
NOOP = -1  # the type code of the no-op part (``Part``)


def targeting_rules(params: GeneratorParams) -> TargetingRules | None:
    """The params' target rules of Q114 (``params.targeting``; None: §2.4's rules as written, `tiny`'s)."""
    return getattr(params, "targeting", None)


@dataclass(frozen=True)
class Target:
    """One candidate target of an event (codes of design §4.1)."""

    kind: int  # 0 chokepoint (node index), 1 edge, 2 node, 3 region
    index: int
    commodity: int = -1  # -1 = every commodity
    counterpart: int = -1  # region index, -1 if none


def region_of_node(inst: Instance, node: int) -> int:
    return inst.nodes[node].region


def strait_chokepoints(inst: Instance, params: GeneratorParams) -> dict[frozenset[int], int]:
    """{frozenset(region a, region b): chokepoint node} of ``params.laws.strait_closures`` (M5-O13; {} when none).

    The targets of the dyad strait closures (``events.strait_closures``; §4.4 "a CN-TW event also closes the Taiwan
    Strait"): chokepoints named by node id, regions by id, as every params reference (``params``); a tree without
    the field (`tiny`'s ``MarkLaws``) has none.

    Raises:
        ValueError: if a named region is not the instance's, the named node is not a chokepoint of it, or a pair is
            listed twice.

    """
    out: dict[frozenset[int], int] = {}
    node_index = {n.id: i for i, n in enumerate(inst.nodes)}
    for a, b, cid in getattr(params.laws, "strait_closures", ()):
        if a not in inst.region_index or b not in inst.region_index or a == b:
            raise ValueError(f"strait closure ({a}, {b}, {cid}): two distinct regions of the instance")
        c = node_index.get(cid)
        if c is None or c not in inst.chokepoints:
            raise ValueError(f"strait closure ({a}, {b}, {cid}): {cid} is not a chokepoint of {inst.instance_id}")
        key = frozenset((inst.region_index[a], inst.region_index[b]))
        if key in out:
            raise ValueError(f"strait closure ({a}, {b}): the pair is listed twice")
        out[key] = c
    return out


def candidate_targets(
    inst: Instance,
    type_code: int,
    region: int,
    adjacency: dict[int, dict[int, float]],
    partners: tuple[int, ...] = (),
    *,
    targeting: TargetingRules | None = None,
) -> list:
    """The candidate targets of type ``type_code`` in ``region`` with their (unnormalised) weights, [(Target, w)].

    ``adjacency`` is the trade-and-tie adjacency A as {m: {m': A_mm'}} for regional-conflict counterparts. Tariffs and
    sanctions read an edge's open commodities K_e less its pairs of Z_0 (``prohibitions_at_reset``, in force the
    whole episode): a pair closed from week 0 is traded never, so an event on it would have no effect.
    ``partners`` are the other regions of the dyads at war with ``region`` in the week, in dyad order
    (``TypeRules.state``): a regional conflict's candidates are those partners, weight 1 each, when there are any (a
    war from a named dyad takes precedence, Q59), else the regions m' with weight A_{m m'} > 0 (§2.4 "Targets"), else
    the single target (``region``, counterpart -1) of weight 1 (Q94 (f)), so its list is never empty. Other types
    ignore them. ``targeting`` with ``sanction_no_transit`` (V1, Q114) drops the sanction edges whose tail is a
    chokepoint; None keeps them (module docstring).
    """
    name = codes.EVENT_TYPES[type_code]
    out: list[tuple[Target, float]] = []
    z0 = set(inst.prohibitions_at_reset)
    no_transit = targeting is not None and targeting.sanction_no_transit

    def open_k(j: int, e) -> list[int]:
        """Edge j's commodities K_e less its pairs of Z_0."""
        return [k for k in e.K if (j, k) not in z0]

    if name in ("militarised_closure", "piracy"):
        for c in inst.chokepoints:
            if region in inst.chokepoint_adjacency.get(c, ()):
                out.append((Target(CHOKEPOINT, c), 1.0))
    elif name == "tariff":  # exporter uniform, then commodity uniform among that exporter's edges into m (§2.4)
        by_exporter: dict[int, set[int]] = {}
        for j, e in enumerate(inst.edges):
            m2 = region_of_node(inst, e.tail)
            if not e.coupling and region_of_node(inst, e.head) == region and m2 != region and (ks := open_k(j, e)):
                by_exporter.setdefault(m2, set()).update(ks)
        for m2 in sorted(by_exporter):
            ks = sorted(by_exporter[m2])
            w = 1.0 / (len(by_exporter) * len(ks))
            out += [(Target(REGION, region, commodity=k, counterpart=m2), w) for k in ks]
    elif name == "sanction":
        out = [
            (Target(EDGE, j), 1.0)
            for j, e in enumerate(inst.edges)
            if not e.coupling
            and region_of_node(inst, e.tail) == region
            and region_of_node(inst, e.head) != region
            and open_k(j, e)
            and not (no_transit and e.tail in inst.chokepoint_ordinal)  # V1: a transit leg is not an export
        ]
    elif name == "material_outage":
        out = [(Target(NODE, n), 1.0) for n in inst.supply_nodes if inst.nodes[n].type == "material"]
        out = [(t, w) for t, w in out if region_of_node(inst, t.index) == region]
    elif name == "energy_shock":
        out = [(Target(NODE, g), 1.0) for g in inst.grids if region_of_node(inst, g) == region]
    elif name == "regional_conflict" and partners:  # the war comes from a named dyad (Q59): its partners, uniform
        out = [(Target(REGION, region, counterpart=m2), 1.0) for m2 in partners]
    elif name == "regional_conflict":
        out = [
            (Target(REGION, region, counterpart=m2), w) for m2, w in sorted(adjacency.get(region, {}).items()) if w > 0
        ]
        if not out:  # no dyad partner at war and no A partner (Q94 (f)): its restoration (13) and war-risk class only
            out = [(Target(REGION, region, counterpart=-1), 1.0)]
    return out


def type_weights(
    inst: Instance,
    shares: tuple[tuple[str, float], ...],
    region: int,
    z_c: int,
    adjacency: dict[int, dict[int, float]],
    partners: tuple[int, ...] = (),
    *,
    targeting: TargetingRules | None = None,
    block: int | None = None,
) -> list[tuple[int, float]]:
    """Feasible types of one block in ``region`` at conflict state ``z_c``, with renormalised weights (sum 1).

    A type is feasible when it has a candidate target (``candidate_targets``, under ``targeting``); a regional conflict
    needs the war state too, and there it always has one (the dyad partners at war, else A, else counterpart -1, Q94
    (f)), so the weights depend on neither the dyad partners nor A. ``partners`` is accepted for its old callers and
    ignored (SIMP-M2R2-04: the partners name a conflict's counterpart, never its feasibility). Empty when no type of
    the block is feasible there (a no-op event). With ``targeting.policy_no_op`` (V2, Q114) and ``block`` the policy
    block, the feasible types keep their shares over the block's full share total and the rest is one no-op part
    (``NOOP``, last), which the event stage types as a no-op event; the weights still sum to 1.
    """
    del partners  # the type law does not depend on them (Q94 (f))
    feas = []
    for name, w in shares:
        code = codes.EVENT_TYPES.index(name)
        if w <= 0 or (name == "regional_conflict" and z_c != WAR):
            continue
        if candidate_targets(inst, code, region, adjacency, targeting=targeting):
            feas.append((code, w))
    total = sum(w for _, w in feas)
    if targeting is not None and targeting.policy_no_op and block == POLICY_BLOCK and total > 0:
        full = sum(w for _, w in shares if w > 0)
        out = [(c, w / full) for c, w in feas]
        if total < full:  # the infeasible types' share: a no-op part, not renormalised (V2)
            out.append((NOOP, (full - total) / full))
        return out
    return [(c, w / total) for c, w in feas] if total > 0 else []


Part = tuple[int, Target | None, float]  # (type code or -1 for the no-op part, fixed target or None, share of lambda^0)


def immigrant_parts(
    inst: Instance,
    shares: tuple[tuple[str, float], ...],
    region: int,
    z_c: int,
    adjacency: dict[int, dict[int, float]],
    partners: tuple[int, ...] = (),
    *,
    targeting: TargetingRules | None = None,
    block: int | None = None,
) -> list[Part]:
    """The immigrant intensity of one block in ``region`` split into (type, target, share of lambda^0) parts.

    The feasible types of ``type_weights`` at ``z_c`` (``partners`` accepted and ignored, as there; ``targeting`` and
    ``block`` passed on). Militarised closures are split over the region's adjacent chokepoints, each part fixing its
    target; ``modulate`` multiplies each by g(X^t_c)/c_X of a week (Q51; ``TypeRules.week_parts``). Other types keep
    ``target`` None (drawn later, uniform or by A). The shares sum to 1; a block with no feasible type gives one no-op
    part (-1, None, 1.0), which still counts toward the rate (§2.4 profile), as does V2's no-op part (-1, None, share).
    """
    del partners  # the type law does not depend on them (Q94 (f)), so neither do its parts
    parts: list[Part] = []
    tw = type_weights(inst, shares, region, z_c, adjacency, targeting=targeting, block=block)
    if not tw:
        return [(NOOP, None, 1.0)]
    for code, w in tw:
        if code == MILITARISED_CLOSURE:
            cands = candidate_targets(inst, code, region, adjacency, targeting=targeting)
            parts += [(code, tgt, w / len(cands)) for tgt, _ in cands]
        else:
            parts.append((code, None, w))
    return parts


def modulate(parts: list[Part], closure_modulation: dict[int, float] | None) -> list[Part]:
    """The parts with every militarised-closure share aimed at chokepoint c multiplied by g(X^t_c)/c_X (Q51).

    The one place the closure factor enters: ``TypeRules.week_parts``, whose shares the cluster sampler's rate
    (``hawkes.immigrant_rates``, through ``TypeRules.week_totals``) and the event stage's type draw both read. ``None``
    (or a chokepoint without a factor) leaves the shares as they are; other types are never modulated (offspring
    neither, §4.2), so on parts without a militarised closure ``modulate`` is the identity.
    """
    if closure_modulation is None:
        return list(parts)
    return [
        (code, tgt, w * closure_modulation.get(tgt.index, 1.0)) if code == MILITARISED_CLOSURE else (code, tgt, w)
        for code, tgt, w in parts
    ]


def block_shares(params: GeneratorParams, block: int) -> tuple[tuple[str, float], ...]:
    """The type shares of block P (0) or M (1) (§2.4 "Type laws per block")."""
    if block not in (0, 1):
        raise ValueError(f"block must be 0 (P) or 1 (M), got {block}")
    return params.types.policy if block == 0 else params.types.militarised


def adjacency_dict(inst: Instance, params: GeneratorParams) -> dict[int, dict[int, float]]:
    """The trade-and-tie adjacency A as {m: {m': A_mm'}}, m != m', positive entries only (R3 §3; §2.4).

    The one adjacency rule of the generator: the params' ``trade_adjacency`` when set, else the instance's
    ``trade_adjacency``; a later entry for the same ordered pair replaces an earlier one, and a later 0 removes it;
    diagonal entries are dropped (tests/test_disruption_adjacency.py). The branching matrix (31)
    (``hawkes.adjacency_matrix``), the counterpart weights of a regional conflict and the profile's closure
    calibration all read it.

    Raises:
        ValueError: on a weight that is negative or not finite (a counterpart is drawn with weight A, and (31) scales
            A / spr(A); the instance loader refuses them too).

    """
    entries = params.hawkes.trade_adjacency or tuple(
        (inst.regions[a], inst.regions[b], w) for a, b, w in inst.trade_adjacency
    )
    out: dict[int, dict[int, float]] = {}
    for a, b, w in entries:
        if not (math.isfinite(w) and w >= 0.0):
            raise ValueError(f"trade adjacency {a}-{b}: the weight must be finite and >= 0, got {w!r} (§4.2)")
        i, j = inst.region_index[a], inst.region_index[b]
        if i == j:
            continue
        if w > 0:
            out.setdefault(i, {})[j] = float(w)
        elif j in out.get(i, {}):
            del out[i][j]
    return {i: row for i, row in out.items() if row}


def closure_modulation(inst: Instance, params: GeneratorParams, regimes, week: int) -> dict[int, float]:
    """g(X^t_c)/c_X of (29) per chokepoint node for week t (Q51), from the chokepoint units of ``regimes.X``.

    Signal units are ordered regions, dyads, chokepoints (§4.1); week t's value is column ``regimes.col(t)``, the
    state in force during [t - 1, t). Weeks run -B_burn + 1 .. T, the weeks that draw immigrants.

    Raises:
        ValueError: on a week outside -B_burn + 1 .. T.

    """
    if not -regimes.burn_in + 1 <= week <= regimes.T:
        raise ValueError(f"week {week} outside the sampled weeks {-regimes.burn_in + 1} .. {regimes.T}")
    base = len(inst.regions) + len(params.regime.dyads)
    col = regimes.col(week)
    return {c: float(tilt(regimes.X[base + i, col], params.latent_chokepoint)) for i, c in enumerate(inst.chokepoints)}


def week_parts(
    inst: Instance,
    params: GeneratorParams,
    regimes,
    block: int,
    region: int,
    week: int,
    adjacency: dict[int, dict[int, float]] | None = None,
) -> list[Part]:
    """The immigrant parts of (block, region, week), (30) with Q51: ``TypeRules.week_parts`` for one call.

    ``immigrant_parts`` at the region's effective z^c (dyad wars applied) in column ``regimes.col(week)``, with the
    militarised closures modulated by ``closure_modulation`` of the same week (a type rule, so whichever block's
    shares hold closures). The cluster sampler's immigrant intensity of the week is lambda^0 of its regime times the
    ``math.fsum`` of these shares (``hawkes.immigrant_rates``, through ``TypeRules.week_totals``), and the event stage
    types an immigrant of key (block, region, week + B_burn, rank) by these parts. ``adjacency`` defaults to
    ``adjacency_dict(inst, params)``.
    """
    return TypeRules(inst, params, regimes, adjacency).week_parts(block, region, week)


State = tuple[int, tuple[int, ...]]  # (effective z^c, partners of the dyads at war) of a region in a column


class TypeRules:
    """The type laws and target rules of one (instance, params, regime paths), memoised: the one composition of (30).

    ``candidates``, ``type_weights`` and ``parts`` give what this module's ``candidate_targets``, ``type_weights`` and
    ``immigrant_parts`` give for the same arguments with the params' ``targeting`` (and the block), bit for bit, each
    computed once per key and kept; ``state`` reads
    the regime state of a (region, column); ``week_parts`` modulates the parts of a week and ``week_totals`` sums them
    per week (the cluster sampler's rate per unit lambda^0). The kept lists are shared between calls and never to be
    modified (``week_parts`` returns a fresh list, ``modulate``'s).

    Raises:
        ValueError: (at construction) a dyad of ``params.regime.dyads`` naming a region the instance does not have.

    """

    def __init__(
        self, inst: Instance, params: GeneratorParams, regimes, adjacency: dict[int, dict[int, float]] | None = None
    ) -> None:
        self.inst, self.params, self.regimes = inst, params, regimes
        self.adjacency = adjacency_dict(inst, params) if adjacency is None else adjacency
        self.targeting = targeting_rules(params)  # Q114's rules (None: §2.4's as written)
        try:  # region indices of params.regime.dyads, in dyad order (the rows of z_dyad)
            self.dyads = tuple((inst.region_index[a], inst.region_index[b]) for a, b in params.regime.dyads)
        except KeyError as err:
            raise ValueError(f"(Q59): dyad region {err.args[0]!r} is not an instance region") from None
        self._candidates: dict[tuple[int, int, tuple[int, ...]], list[tuple[Target, float]]] = {}
        self._weights: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
        self._parts: dict[tuple[int, int, int], list[Part]] = {}
        self._modulation: dict[int, dict[int, float]] = {}
        self._states: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def check(self, inst: Instance, params: GeneratorParams, regimes) -> "TypeRules":
        """``self`` if it was built for (``inst``, ``params``, ``regimes``), the one composition a caller may share.

        The cluster sampler, the event stage and the shadow sampler take the episode's rules as ``rules=`` (built once
        per episode by ``sampler.sample_events``, SIMP-M2R2-05); rules of another episode would type its events by
        another regime path.

        Raises:
            ValueError: if the rules were built for another instance, params tree or regime-path object.

        """
        if not self.fits(inst, params, regimes):
            raise ValueError("the type rules were built for another (instance, params, regime paths) (SIMP-M2R2-05)")
        return self

    def fits(self, inst: Instance, params: GeneratorParams, regimes) -> bool:
        """Whether ``self`` was built for these very (``inst``, ``regimes``) objects and an equal ``params``.

        Rules that crossed a process boundary (an ``EventSample`` returned by a joblib worker) hold a copy of the
        instance, not the caller's object, so they do not fit; the caller builds the rules again (the same values).
        """
        return self.inst is inst and self.regimes is regimes and self.params == params

    def state(self, region: int, col: int) -> State:
        """(effective z^c, dyad partners) of ``region`` in column ``col``: what its type law and counterpart read.

        The partners are the other regions of the dyads that contain ``region`` and are at war in the column
        (``z_dyad`` = 2), in dyad order (§4.4, Q59).
        """
        partners = tuple(
            b if a == region else a
            for d, (a, b) in enumerate(self.dyads)
            if region in (a, b) and int(self.regimes.z_dyad[d, col]) == WAR
        )
        return int(self.regimes.z_c[region, col]), partners

    def candidates(self, type_code: int, region: int, partners: tuple[int, ...] = ()) -> list[tuple[Target, float]]:
        """``candidate_targets(inst, type_code, region, adjacency, partners)`` under the params' targeting, kept."""
        key = (type_code, region, partners)
        if key not in self._candidates:
            self._candidates[key] = candidate_targets(
                self.inst, type_code, region, self.adjacency, partners, targeting=self.targeting
            )
        return self._candidates[key]

    def type_weights(
        self, block: int, region: int, z_c: int, partners: tuple[int, ...] = ()
    ) -> list[tuple[int, float]]:
        """``type_weights`` of block P (0) or M (1) in ``region`` at the effective conflict state ``z_c``, kept.

        The dyad partners of ``state`` name a regional conflict's counterpart only; the type law does not read them
        (Q94 (f); SIMP-M2R2-04), so neither the key nor the list does (``partners`` is accepted for callers that pass
        a whole ``state``, and ignored).
        """
        del partners
        key = (block, region, z_c)
        if key not in self._weights:
            shares = block_shares(self.params, block)
            self._weights[key] = type_weights(
                self.inst, shares, region, z_c, self.adjacency, targeting=self.targeting, block=block
            )
        return self._weights[key]

    def parts(self, block: int, region: int, z_c: int, partners: tuple[int, ...] = ()) -> list[Part]:
        """The unmodulated ``immigrant_parts`` of block P (0) or M (1) in ``region`` at ``z_c``, kept.

        ``partners`` is accepted and ignored, as in ``type_weights``.
        """
        del partners
        key = (block, region, z_c)
        if key not in self._parts:
            shares = block_shares(self.params, block)
            self._parts[key] = immigrant_parts(
                self.inst, shares, region, z_c, self.adjacency, targeting=self.targeting, block=block
            )
        return self._parts[key]

    def modulation(self, week: int) -> dict[int, float]:
        """``closure_modulation`` of the week, kept."""
        if week not in self._modulation:
            self._modulation[week] = closure_modulation(self.inst, self.params, self.regimes, week)
        return self._modulation[week]

    def week_parts(self, block: int, region: int, week: int) -> list[Part]:
        """The immigrant parts of (block, region, week), (30) with Q51 (the module's ``week_parts``).

        ``parts`` at the region's effective z^c in column ``regimes.col(week)``, the militarised closures multiplied by
        ``modulation`` of the week through ``modulate``.
        """
        z_c = int(self.regimes.z_c[region, self.regimes.col(week)])
        return modulate(self.parts(block, region, z_c), self.modulation(week))

    def week_totals(self, block: int, region: int) -> np.ndarray:
        """(B_burn + T,) the ``math.fsum`` of the shares of ``week_parts(block, region, t)``, t = -B_burn + 1 .. T.

        The immigrant intensity of those weeks per unit lambda^0 (30): lambda^0 of the week's regime times this entry
        is the cluster sampler's rate (``hawkes.immigrant_rates``). The region's effective conflict states are read
        once over the weeks; when no state's parts hold a militarised closure, ``modulate`` is the identity and each
        state's sum serves all its weeks; otherwise every week sums its ``week_parts``. Either way each entry is the
        fsum of that week's ``week_parts`` shares, bit for bit.
        """
        states, inverse = self._region_states(region)
        parts = [self.parts(block, region, int(z_c)) for z_c in states]
        if any(code == MILITARISED_CLOSURE for p in parts for code, _, _ in p):  # closure parts move with X^t_c
            weeks = range(-self.regimes.burn_in + 1, self.regimes.T + 1)
            return np.array([math.fsum(w for _, _, w in self.week_parts(block, region, t)) for t in weeks])
        return np.array([math.fsum(w for _, _, w in p) for p in parts])[inverse[1:]]

    def _region_states(self, region: int) -> tuple[np.ndarray, np.ndarray]:
        """The distinct effective z^c values of ``region`` over every column, and each column's index into them, kept.

        The type law reads z^c alone (``type_weights``), so the dyad partners at war do not split the states
        (SIMP-M2R2-04).
        """
        if region not in self._states:
            unique, inverse = np.unique(np.asarray(self.regimes.z_c[region], dtype=np.int64), return_inverse=True)
            self._states[region] = (unique, np.asarray(inverse).reshape(-1))
        return self._states[region]

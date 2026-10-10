"""The one assembly of a schema-1 omega container (design §4.1 table), shared by both omega builders.

``omega.injected.build_omega`` (an explicit event list, M1) and ``disruption.sampler.sample_omega`` (the generator, M2)
differ only in their event group, their ``generator_id``, their burn-in and their regime paths; everything else of the
container (the meta arrays, demand (21) with its parts, the stored marks of (13), (14) and §3.4, and the groups that
milestone M3 and phase 4 fill: latent decoys, leads, the adversarial member, blackouts, shadow and opponent events) is
written here once, under the fixed floating-point error state ``marks.FP_ERRORS`` that the §12 row 'Floating-point error
state' puts every omega builder under (M2R1-SIMP-01: the generator's copy of this code had dropped it).
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shockbench_flow import marks
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import ARRAY_DTYPES, Omega, empty_events
from shockbench_flow.omega.demand import demand_path
from shockbench_flow.omega.seeds import check_entropy, check_episode


META_LEN = 64  # <U64 of the meta_* string arrays; a longer string would be truncated silently
INFORMATION_META_LEN = 4096  # <U4096 of meta_information_params (container.ARRAY_DTYPES)
REGIME_ARRAYS = ("z_c", "z_p", "z_dyad", "X", "W")  # the regime paths a generated omega stores (§4.2), R rows by weeks
# M3: the own chains and the dyad rows, written for the generated kind only (container.GENERATED_ONLY; §12 M3 rows)
OWN_CHAIN_ARRAYS = ("z_c_own", "dyad_regions")
# M3: the arrays of the announcement stage (``disruption.announce``), the generated kind only: leads and their
# uniforms (stream 4), the shadow group Y with its channels and thinning uniforms (stream 7), blackout spells
# (stream 11)
ANNOUNCEMENT_ARRAYS = ("ev_lead", "V_lead", "U_decoy", "sh_channel", "blk_start", "blk_end")  # plus every sh_* column


@dataclass(frozen=True)
class InformationMeta:
    """What the observation wrapper needs of the generator that omega's arrays do not give back (M3, option B).

    Stored as ``meta_information_params`` (0-d ``<U4096``) for generated omegas, like ``meta_mark_params``, so
    ``Env.reset(instance, regime, omega, policy_seed)`` keeps the §9.1 signature (no ``params=``): ``information.view``
    reads it; an injected omega needs none (no conflict layer, so no Xi term in (48), and no shadows).
    """

    P_yr: tuple[tuple[float, float, float], ...]  # the yearly conflict chain of (28): Pi^c of the forecast (48)
    phi_bar: tuple[tuple[str, float], ...]  # phi-bar_ch of (49) per decoy-bearing channel name (codes.CHANNELS)
    xi: bool  # whether omega's d carries the shock factor Xi: True from the generator, False on omega^0 (57)


_INFO_KEYS = ("P_yr", "phi_bar", "xi")


def _real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _no_constant(name: str):
    """``json.loads``' hook for NaN and Infinity tokens: canonical JSON holds none (``allow_nan=False``)."""
    raise ValueError(f"{name} is not a finite JSON number")


def _check_information_meta(P_yr, phi_bar, xi) -> InformationMeta:
    """The ``InformationMeta`` of the parts, with Python floats and tuples, or ValueError (the codec's one check)."""
    if not (isinstance(P_yr, (list, tuple)) and len(P_yr) == 3):
        raise ValueError(f"meta_information_params: P_yr must be a 3 x 3 matrix, got {P_yr!r}")
    rows = []
    for row in P_yr:
        if not (isinstance(row, (list, tuple)) and len(row) == 3 and all(_real(x) for x in row)):
            raise ValueError(f"meta_information_params: P_yr must be a 3 x 3 matrix of real numbers, got {P_yr!r}")
        vals = tuple(float(x) for x in row)
        if not all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in vals):  # rows are the published rounded values
            raise ValueError(f"meta_information_params: P_yr entries must lie in [0, 1] (28), got the row {vals!r}")
        rows.append(vals)
    if not isinstance(phi_bar, (list, tuple)):
        raise ValueError(f"meta_information_params: phi_bar must be (channel, value) pairs, got {phi_bar!r}")
    pairs = []
    for pair in phi_bar:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[0], str) and _real(pair[1])):
            raise ValueError(f"meta_information_params: phi_bar must be (channel, value) pairs, got {pair!r}")
        name, value = str(pair[0]), float(pair[1])
        if name not in codes.DECOY_CHANNELS or name in (n for n, _ in pairs):
            raise ValueError(f"meta_information_params: phi_bar channel {name!r} is not a new decoy-bearing channel")
        if not 0.0 < value < 1.0:  # NaN fails too: (49) needs phi-bar < 1, and 0 has no shadow process
            raise ValueError(f"meta_information_params: phi_bar of {name} must lie in (0, 1) (49), got {value!r}")
        pairs.append((name, value))
    if not isinstance(xi, bool):
        raise ValueError(f"meta_information_params: xi must be a bool, got {xi!r}")
    return InformationMeta(P_yr=tuple(rows), phi_bar=tuple(pairs), xi=xi)


def information_meta_json(info: InformationMeta) -> str:
    """The canonical JSON of ``info`` (sorted keys, floats by repr, no NaN), as ``marks.mark_params_json`` writes.

    {"P_yr": [[3 floats] x 3], "phi_bar": [[channel, float], ...], "xi": bool}, the pairs in ``info``'s order.

    Raises:
        ValueError: if the text would not fit ``<U4096``, holds a non-finite number, or ``info`` holds a value
            ``information_meta_from_json`` would refuse.

    """
    ok = _check_information_meta(info.P_yr, info.phi_bar, info.xi)
    doc = {"P_yr": [list(r) for r in ok.P_yr], "phi_bar": [list(p) for p in ok.phi_bar], "xi": ok.xi}
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text) > INFORMATION_META_LEN:
        raise ValueError(f"meta_information_params: {len(text)} characters do not fit <U{INFORMATION_META_LEN}")
    return text


def information_meta_from_json(text: str) -> InformationMeta:
    """Parse ``meta_information_params`` back; the inverse of ``information_meta_json``, bit for bit.

    Raises:
        ValueError: on a malformed text, an unknown or missing key, a phi-bar outside (0, 1) or on a channel that
            bears no decoys, or a P^yr that is not a 3 x 3 matrix with entries in [0, 1] (its rows are the published
            rounded values of ``RegimeParams.P_yr``, one summing to 0.9999, so the sums are not re-checked here).

    """
    if not isinstance(text, str) or len(text) > INFORMATION_META_LEN:
        raise ValueError(f"meta_information_params must be a str of at most {INFORMATION_META_LEN} characters")
    try:
        raw = json.loads(text, parse_constant=_no_constant)
    except (TypeError, ValueError) as err:
        raise ValueError(f"meta_information_params is not canonical JSON: {err}") from err
    if not isinstance(raw, dict) or set(raw) != set(_INFO_KEYS):
        raise ValueError(f"meta_information_params must be a JSON object with exactly the keys {_INFO_KEYS}")
    return _check_information_meta(raw["P_yr"], raw["phi_bar"], raw["xi"])


def meta(value: str, name: str) -> np.ndarray:
    """The 0-d ``<U64`` meta array ``name`` holding ``value``.

    Raises:
        ValueError: if ``value`` is longer than 64 characters, which the dtype would truncate silently.

    """
    if len(value) > META_LEN:
        raise ValueError(f"{name} longer than {META_LEN} characters would be truncated: {value!r}")
    return np.array(value, dtype=ARRAY_DTYPES[name])


@marks.fixed_fp_errors
def assemble_omega(
    inst: Instance,
    ev: Mapping[str, np.ndarray],
    *,
    generator_id: str,
    split: str,
    episode: int,
    burn_in: int,
    params: marks.MarkParams,
    entropy: int,
    demand_noise: bool,
    regimes: Mapping[str, np.ndarray] | None = None,
    announcements: Mapping[str, np.ndarray] | None = None,
    information: InformationMeta | None = None,
) -> Omega:
    """A complete schema-1 omega from its event group ``ev`` (the ``ev_*`` arrays and keys), in one place.

    Writes the meta arrays (schema version, instance hash and content digest, ``generator_id``, split, the decimal of
    the episode index n that ``seeds.check_episode`` returns, not the caller's ``str(episode)``, B_burn, the canonical
    JSON of ``params``), the regime paths (``regimes``: ``z_c``, ``z_p``, ``z_dyad``, ``X``, ``W`` cast to their schema
    dtypes; None, an injected list, stores ``z_c`` and ``z_p`` with R rows and no weeks and the others empty), the
    groups later milestones fill (no latent decoys, a NaN lead per event, adversarial member -1, no blackouts, empty
    shadow and opponent groups), demand ``d`` and ``eps_parts`` by (21) from stream 5 (``demand_path`` on the
    conflict layer ``regimes["z_c"]``, Xi = 1 without one) and the stored marks of the events under ``params``
    (``marks.stored_mark_arrays``). Runs under the fixed floating-point error state ``marks.FP_ERRORS``, whatever the
    caller's NumPy error state.

    Milestone M3 (the generated kind only; None, the default, writes exactly the arrays of today, so an injected
    omega and its hash do not move): ``regimes`` may also carry ``OWN_CHAIN_ARRAYS`` (``z_c_own``, checked by
    ``regime.check_own_chain``, and ``dyad_regions``); ``announcements`` carries ``ANNOUNCEMENT_ARRAYS`` and the
    ``sh_*`` columns of ``disruption.announce.announcement_arrays``, which replace the placeholders (``ev_lead`` the
    NaN column of ``ev``); ``information`` is written as ``meta_information_params`` (``information_meta_json``).

    Raises:
        ValueError: on an identifier longer than 64 characters, a bad event (``marks.stored_mark_arrays``), an
            entropy outside [0, 2**128) or an episode outside [0, 2**32) (27); or, M3, an own chain that is not the
            effective layer's (``regime.check_own_chain``), an M3 array without regime paths (the generated kind
            only), or announcement arrays of the wrong names, dtypes or lengths.
        TypeError: if the entropy or the episode is not an integer (a bool included; ``seeds.check_episode``).

    """
    check_entropy(entropy)  # (27), in demand_path's order: the entropy's error before the episode's
    episode = check_episode(episode)  # a Python int: np.int64(2), an int subclass and 2 give one meta_episode (§12)
    R, n = len(inst.regions), int(ev["ev_type"].shape[0])
    if regimes is None:
        layers = {
            "z_c": np.zeros((R, 0), dtype=np.int8),
            "z_p": np.zeros((R, 0), dtype=np.int8),
            "z_dyad": np.zeros((0, 0), dtype=np.int8),
            "X": np.zeros((0, 0), dtype=np.float64),
            "W": np.zeros((0, 0), dtype=np.float64),
        }
        z_c = None
    else:
        layers = {name: np.asarray(regimes[name]).astype(ARRAY_DTYPES[name]) for name in REGIME_ARRAYS}
        z_c = regimes["z_c"]
    arrays: dict[str, np.ndarray] = {
        "meta_schema_version": meta(codes.SCHEMA_VERSION, "meta_schema_version"),
        "meta_instance_hash": meta(inst.hash, "meta_instance_hash"),
        "meta_instance_digest": meta(inst.content_digest, "meta_instance_digest"),
        "meta_generator_id": meta(generator_id, "meta_generator_id"),
        "meta_split": meta(split, "meta_split"),
        "meta_episode": meta(str(episode), "meta_episode"),
        "meta_burn_in": np.array(burn_in, dtype=np.int64),
        "meta_mark_params": np.array(marks.mark_params_json(params), dtype=ARRAY_DTYPES["meta_mark_params"]),
        **layers,
        "U_decoy": np.zeros(0, dtype=np.float64),
        "V_lead": np.full(n, np.nan, dtype=np.float64),
        "adv_family": np.array(-1, dtype=np.int16),
        "adv_member": np.array(-1, dtype=np.int16),
        "blk_start": np.zeros(0, dtype=np.int32),
        "blk_end": np.zeros(0, dtype=np.int32),
        "sh_channel": np.zeros(0, dtype=np.int8),
        **empty_events("sh"),
        **empty_events("op"),
        **ev,
    }
    arrays.update(_m3_arrays(ev, regimes, announcements, information))  # the generated kind only (M3); {} otherwise
    arrays["d"], arrays["eps_parts"] = demand_path(inst, entropy, episode, burn_in=burn_in, z_c=z_c, noise=demand_noise)
    arrays.update(marks.stored_mark_arrays(inst, ev, params))  # from the ev_* arrays alone (13), (14), §3.4
    return Omega(arrays)


def _m3_arrays(
    ev: Mapping[str, np.ndarray],
    regimes: Mapping[str, np.ndarray] | None,
    announcements: Mapping[str, np.ndarray] | None,
    information: InformationMeta | None,
) -> dict[str, np.ndarray]:
    """The M3 arrays of a generated omega: the own chains, the announcement stage and the information meta.

    ``z_c_own`` and ``dyad_regions`` come with the regime paths and are checked against the effective layer
    (``regime.check_own_chain``); the announcement arrays replace the placeholders (``ev_lead`` of ``ev``, the empty
    shadow group, ``V_lead`` NaN, no spell) and must match the event and shadow counts; ``information`` is written as
    ``meta_information_params``. Nothing is written without them, so an injected omega and every M2 omega keep their
    arrays and hash (design §12 M3 rows).

    Raises:
        ValueError: on an M3 array without regime paths, only one of the own-chain arrays, a bad own chain, or
            announcement arrays of other names, dtypes or lengths.

    """
    own = [n for n in OWN_CHAIN_ARRAYS if n in (regimes or {})]
    if regimes is None and (announcements is not None or information is not None):
        raise ValueError("the announcement arrays and meta_information_params belong to a generated omega (regimes)")
    out: dict[str, np.ndarray] = {}
    if own:
        if len(own) != len(OWN_CHAIN_ARRAYS):
            raise ValueError(f"z_c_own and dyad_regions come together, got {own}")
        from shockbench_flow.disruption.regime import check_own_chain  # the generator's rule (omega stays below it)

        z_c_own = np.asarray(regimes["z_c_own"]).astype(ARRAY_DTYPES["z_c_own"])
        dyad_regions = np.asarray(regimes["dyad_regions"]).astype(ARRAY_DTYPES["dyad_regions"]).reshape(-1, 2)
        check_own_chain(np.asarray(regimes["z_c"]), z_c_own, np.asarray(regimes["z_dyad"]), dyad_regions)
        out["z_c_own"], out["dyad_regions"] = z_c_own, dyad_regions
    if announcements is not None:
        names = {*ANNOUNCEMENT_ARRAYS, *empty_events("sh")}
        if set(announcements) != names:
            raise ValueError(f"announcement arrays: expected {sorted(names)}, got {sorted(announcements)}")
        for name, a in announcements.items():
            if np.asarray(a).dtype != np.dtype(ARRAY_DTYPES[name]):
                raise ValueError(f"announcement array {name} has dtype {np.asarray(a).dtype}, not {ARRAY_DTYPES[name]}")
        n, n_sh = int(ev["ev_type"].shape[0]), int(announcements["sh_type"].shape[0])
        shapes = {
            "ev_lead": (n, len(codes.CHANNELS)),
            "V_lead": (n,),
            "sh_lead": (n_sh, len(codes.CHANNELS)),
            "sh_channel": (n_sh,),
            "U_decoy": (n_sh,),
            "blk_start": (len(codes.BLACKOUT_SPELLS),),
            "blk_end": (len(codes.BLACKOUT_SPELLS),),
        }
        for name, shape in shapes.items():
            if np.shape(announcements[name]) != shape:
                raise ValueError(f"announcement array {name} has shape {np.shape(announcements[name])}, not {shape}")
        out.update({name: np.asarray(a) for name, a in announcements.items()})
    if information is not None:
        text = information_meta_json(information)
        out["meta_information_params"] = np.array(text, dtype=ARRAY_DTYPES["meta_information_params"])
    return out

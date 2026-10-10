"""Injected event lists: omega for milestone M1 without the generator (design §2.4 "Events"; Q85).

``build_omega`` gives a complete schema-1 omega for an explicit event list: its ``ev_*`` arrays and ``generator_id``
here, and the rest of the container (demand (21) from stream 5, the stored marks ``R_f``, ``R_osat``, ``alpha_bar``,
``sigma_scr``, ``wr_class`` of ``marks.stored_mark_arrays``, empty regime, latent, shadow, opponent and blackout arrays)
from ``omega.assembly.assemble_omega``, the one assembly the generator's ``sample_omega`` shares. ``event_free`` gives
omega^0 of (57).
"""

import dataclasses
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from shockbench_flow import marks
from shockbench_flow.instance.io import sha256_hex
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import MarkParams
from shockbench_flow.omega import codes
from shockbench_flow.omega.assembly import assemble_omega, information_meta_from_json, information_meta_json, meta
from shockbench_flow.omega.container import (
    ARRAY_DTYPES,
    EVENT_FLOATS,
    Omega,
    chokepoint_region,
    empty_events,
    event_float,
    event_group,
)
from shockbench_flow.omega.demand import demand_from_parts
from shockbench_flow.omega.seeds import check_entropy, check_episode


@dataclass(frozen=True)
class InjectedEvent:
    """One event of an injected list; ids are instance ids, names are the codes of ``omega.codes``."""

    type: str  # one of codes.EVENT_TYPES
    target_kind: str  # one of codes.TARGET_KINDS
    target: str  # node id (chokepoint or node kind), edge id, or region name
    onset: float  # t^on_q, continuous weeks (negative when carried in)
    duration: float  # T_q in weeks
    severity: float  # sigma_q
    region: str | None = None  # m_q; default: the target's region (a chokepoint's first adjacent region)
    counterpart: str | None = None  # region, or None
    commodity: str | None = None  # None = all commodities
    rate: float = float("nan")  # tariff rate or piracy surcharge
    restoration: int = -1  # regime A-D = 0-3 for fab events
    T0: float = float("nan")  # dead time (weeks) for fab events
    tau_rho: float = float("nan")  # restoration time scale (weeks) for fab events


# the fixed event list of the hand-solved checks (§2.4; Q85): one militarised closure of `chk`, onset 5.5, 1 week,
# severity 0.5 (Red Sea class in week 7)
TINY_FIXED_EVENTS: tuple[InjectedEvent, ...] = (
    InjectedEvent(
        type="militarised_closure", target_kind="chokepoint", target="chk", onset=5.5, duration=1.0, severity=0.5
    ),
)


def _code(value: str, table: tuple[str, ...], what: str) -> int:
    if value not in table:
        raise ValueError(f"unknown {what} {value!r}; expected one of {table}")
    return table.index(value)


def _index(value: str, table: Mapping[str, int], what: str) -> int:
    if value not in table:
        raise ValueError(f"unknown {what} {value!r} in this instance")
    return table[value]


def _restoration(i: int, value: object) -> int:
    """The restoration code -1..3 (regime A-D, -1 if not a fab event) as an int; ``-1.0`` states ``-1``.

    Raises:
        TypeError: if the value is not a real number (a bool included).
        ValueError: if it is not one of -1..3.

    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"event {i}: restoration must be an integer code -1..3, got {value!r}")
    if value not in (-1, *range(len(codes.RESTORATION_REGIMES))):
        raise ValueError(f"event {i}: restoration code {value} outside -1..3")
    return int(value)


def _target(inst: Instance, ev: InjectedEvent) -> tuple[int, int, int]:
    """(target_kind code, target index, default region index) of one event."""
    kind = _code(ev.target_kind, codes.TARGET_KINDS, "target kind")
    if ev.target_kind == "chokepoint":
        node = _index(ev.target, inst.node_index, "node")
        if inst.nodes[node].type != "chokepoint":
            raise ValueError(f"target {ev.target!r} of kind 'chokepoint' is a {inst.nodes[node].type} node")
        return kind, node, chokepoint_region(inst, node)
    if ev.target_kind == "edge":
        edge = _index(ev.target, inst.edge_index, "edge")
        return kind, edge, inst.nodes[inst.edges[edge].tail].region
    if ev.target_kind == "node":
        node = _index(ev.target, inst.node_index, "node")
        return kind, node, inst.nodes[node].region
    region = _index(ev.target, inst.region_index, "region")
    return kind, region, region


def event_arrays(inst: Instance, events: Sequence[InjectedEvent], burn_in: int = 0) -> dict[str, np.ndarray]:
    """The ``ev_*`` arrays (and CSR keys) of an injected list, in list order, in the dtypes of ``EVENT_FIELDS``.

    Region m_q: the event's own ``region`` if given, else the target's: a chokepoint's first adjacent region in
    ``inst.chokepoint_adjacency`` (the node's region when it has none), an edge's tail-node region, a node's region, a
    region itself. Leads are NaN on every channel (injected events are unannounced).

    Injected events get immigrant-form keys (type code, region, week + burn_in, rank), with week = floor(onset) + 1 the
    onset week t_q and rank counting earlier events of the list with the same (type, region, week). The type code stands
    where a generated immigrant has its block (§4.2), since the unexcited types have block -1 and keys must be >= 0.

    Numeric fields are stored in canonical form (``container.event_float``, as generated events are): every NaN as
    the one quiet NaN, -0.0 as 0.0.

    Raises:
        ValueError: on an unknown type, target kind, id or name, a NaN onset, an infinite numeric field, a restoration
            code outside -1..3, or a negative week key (onset < -burn_in - 1).
        TypeError: if a numeric field or the restoration code is not a real number (a bool included).

    """
    rows: list[dict[str, object]] = []
    rank: dict[tuple[int, int, int], int] = {}
    for i, ev in enumerate(events):
        typ = _code(ev.type, codes.EVENT_TYPES, "event type")
        kind, target, region = _target(inst, ev)
        if ev.region is not None:
            region = _index(ev.region, inst.region_index, "region")
        fl = {name: event_float(i, name, getattr(ev, name)) for name in EVENT_FLOATS}
        restoration = _restoration(i, ev.restoration)
        if math.isnan(fl["onset"]):
            raise ValueError(f"event {i}: onset must be finite, got {ev.onset}")
        week = math.floor(fl["onset"]) + 1 + burn_in  # onset week t_q = floor(t^on_q) + 1, plus B_burn (§4.1)
        if week < 0:
            raise ValueError(f"event {i}: week key {week} < 0 (onset {ev.onset} before week -{burn_in})")
        r = rank.get((typ, region, week), 0)
        rank[(typ, region, week)] = r + 1
        rows.append(
            {
                "key": (typ, region, week, r),
                "type": typ,
                "block": codes.EVENT_BLOCK[typ],
                "region": region,
                "counterpart": -1 if ev.counterpart is None else _index(ev.counterpart, inst.region_index, "region"),
                "target_kind": kind,
                "target": target,
                "commodity": -1 if ev.commodity is None else _index(ev.commodity, inst.commodity_index, "commodity"),
                "restoration": restoration,
                **fl,
            }
        )
    return event_group(rows, "ev")  # the one event-group writer (SIMP-M2R2-01)


def _event_config(i: int, ev: InjectedEvent, region: str) -> dict:
    """Event ``i`` of the generator config in canonical types and resolved, as the ``ev_*`` arrays store it.

    Numeric fields become canonical floats (``container.event_float``; NaN: None), ``restoration`` an int, and
    ``region`` the region m_q that ``event_arrays`` resolved (``region``, a name), so onset 5 and 5.0, a NumPy scalar,
    -0.0 and 0.0, any NaN, or a default region left implicit and spelled out give the same ``generator_id`` exactly when
    they give the same arrays (§12 'Mark parameters in omega': the canonical *resolved* config).
    """
    out = asdict(ev)
    for name in EVENT_FLOATS:
        v = event_float(i, name, out[name])
        out[name] = None if math.isnan(v) else v
    out["restoration"] = _restoration(i, out["restoration"])
    out["region"] = region
    return out


@marks.fixed_fp_errors
def build_omega(
    inst: Instance,
    events: Sequence[InjectedEvent] = (),
    *,
    entropy: int,
    episode: int = 0,
    split: str = "dev",
    demand_noise: bool = True,
    params: MarkParams = MarkParams(),
) -> Omega:
    """A complete schema-1 omega for an injected event list (burn-in 0).

    ``meta_generator_id`` is the SHA-256 of the canonical resolved generator config (schema version, instance hash, the
    event list in canonical types with each event's resolved region (``_event_config``), the demand-noise flag, the
    MarkParams; §4.1), and ``meta_mark_params`` stores the MarkParams, so the mark parameters are bound to omega's hash
    and ``marks.compute_marks`` reads them from omega (V3). ``meta_episode`` is the decimal episode index.

    The events come from ``event_arrays``, and the container from ``omega.assembly.assemble_omega``: demand and its
    parts from ``demand_path`` (stream 5, Xi = 1 since there is no conflict layer), the stored marks from
    ``marks.stored_mark_arrays``, regime layers with R rows and no weeks, empty latent, shadow, opponent and blackout
    groups, ``V_lead`` NaN per event, the adversarial member -1. Everything runs under the fixed floating-point error
    state ``marks.FP_ERRORS``, not NumPy's global one.

    Raises:
        ValueError: on an entropy outside [0, 2**128) or an episode outside [0, 2**32), checked even when
            ``demand_noise`` is False and no key is drawn (27), a bad event (``event_arrays``, or a target kind its type
            does not accept, ``marks.stored_mark_arrays``) or an identifier longer than 64 characters.
        TypeError: if the entropy or the episode is not an integer (a bool included), ``demand_noise`` is not a bool,
            or an event field has the wrong type (``event_arrays``).

    """
    check_entropy(entropy)  # (27), §12 'omega uniforms and demand': even when the noise-free path draws nothing
    episode = check_episode(episode)  # a Python int: np.int64(1) and 1 state one episode, True is refused
    if isinstance(demand_noise, np.bool_):
        demand_noise = bool(demand_noise)
    if not isinstance(demand_noise, bool):  # 1 and True would give one omega but two generator configs
        raise TypeError(f"demand_noise must be a bool, got {demand_noise!r}")
    ev_arrays = event_arrays(inst, events, burn_in=0)  # validates every event before it enters the config
    regions = [inst.regions[int(r)] for r in ev_arrays["ev_region"]]  # m_q as event_arrays resolved it
    config = {
        "schema_version": codes.SCHEMA_VERSION,
        "instance_hash": inst.hash,
        "generator": "injected",
        "events": [_event_config(i, e, regions[i]) for i, e in enumerate(events)],
        "demand_noise": demand_noise,
        "burn_in": 0,
        "mark_params": asdict(params),
    }
    return assemble_omega(
        inst,
        ev_arrays,
        generator_id=sha256_hex(config),
        split=split,
        episode=episode,
        burn_in=0,
        params=params,
        entropy=entropy,
        demand_noise=demand_noise,
    )


def event_free_generator_id(generator_id: str) -> str:
    """``meta_generator_id`` of the event-free twin (57) of an omega drawn with ``generator_id``."""
    return sha256_hex({"event_free_of": generator_id})


@marks.fixed_fp_errors
def event_free(omega: Omega, inst: Instance) -> Omega:
    """omega^0 of (57), the event-free twin.

    Every event is removed (carried-in ones included), and so are the demand shock Xi and the tariff and sanction
    paths; AR(1) demand noise and seasonality stay (proposal F4, §11 row 76).

    Concretely: the ``ev_*`` and ``op_*`` groups become empty and ``V_lead`` (one uniform per event) has length 0;
    ``d`` is rebuilt from the stored ``eps_parts`` by the same recursion (21) with Xi = 1, so a noise-free omega gives
    d-bar exactly; the stored marks are recomputed by ``marks.stored_mark_arrays`` for the empty list, without the
    conflict layer, so no war-state war-risk class survives. Everything else is kept, ``z_c`` and the shadow group
    included (M3: X, W, ``z_c_own``, ``dyad_regions``, ``sh_*``, ``sh_channel``, ``U_decoy`` and the blackout spells,
    so informed and standard on omega^0 see decoys only), and a generated omega's ``meta_information_params`` is
    rewritten with ``xi`` False (its d carries no Xi; design §12 "Event-free twin and decoys"). Runs under the fixed
    floating-point error state ``marks.FP_ERRORS``.

    Raises:
        ValueError: if ``omega`` was not built on ``inst`` (instance hash, V3) or states no ``meta_mark_params`` (the
            stored marks of omega^0 are built under omega's own parameters, never defaults).

    """
    if omega.instance_hash != inst.hash:
        raise ValueError(f"omega is for instance {omega.instance_hash[:12]}..., not {inst.hash[:12]}... (V3)")
    if "meta_mark_params" not in omega:
        raise ValueError("omega has no 'meta_mark_params'; every omega builder sets it (V3)")
    params = marks.mark_params_from_json(str(omega["meta_mark_params"]))
    ev = empty_events("ev")
    new: dict[str, np.ndarray] = {
        **ev,
        **empty_events("op"),
        "V_lead": np.zeros(0, dtype=np.float64),
        "d": demand_from_parts(inst, omega["eps_parts"]),  # (21) without Xi (57)
        "meta_generator_id": meta(event_free_generator_id(str(omega["meta_generator_id"])), "meta_generator_id"),
        **marks.stored_mark_arrays(inst, ev, params),  # of the empty list, without z_c (57)
    }
    if "meta_information_params" in omega:  # M3: d carries no Xi on omega^0 (57), so the forecast's term is off
        info = information_meta_from_json(str(omega["meta_information_params"]))
        text = information_meta_json(dataclasses.replace(info, xi=False))
        new["meta_information_params"] = np.array(text, dtype=ARRAY_DTYPES["meta_information_params"])
    return omega.replace(**new)

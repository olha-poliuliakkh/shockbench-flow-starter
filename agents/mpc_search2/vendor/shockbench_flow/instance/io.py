"""Instance files: canonical JSON, hash, loader and validation (design §2.3; Q70, V20).

The instance hash is the SHA-256 of ``canonical_json(obj)``: ``json.dumps(obj, sort_keys=True, separators=(",", ":"),
allow_nan=False)`` with floats by ``repr`` (Python's default). It is computed over the *parsed* object, so the file's
whitespace does not matter, but a float must stay a float (``1.0``, never ``1``).

Cross-references inside the file are ids (strings); the loader turns them into the integer codes of design §4.1.

The file format is published as a JSON Schema (draft-07), ``data/instance.schema.json`` (Q95): closed key sets, types,
required keys, vocabularies and the single-value ranges the checks below enforce. The loader validates it right after
the ``schema_version`` check and before its own checks, which stay and keep the rules the schema does not state: across
fields, ids, uniqueness, sums, and finite numbers (JSON Schema admits NaN, infinities and integers beyond a float).
"""

import copy
import functools
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

fastjsonschema = None  # vendored copy: the scoring container has no fastjsonschema; schema validation skipped

from shockbench_flow.instance.schema import (
    FAB_CLASSES,
    FLAGSHIP_KINDS,
    INSTANCE_KINDS,
    MODES,
    NODE_TYPES,
    POOLS,
    PRIORITIES,
    SCHEMA_VERSION,
    SUPPLY_TYPES,
    UNMODELLED,
    WEEKS_PER_YEAR,
    AltRef,
    ChokepointAttrs,
    Commodity,
    Demand,
    Edge,
    FabAttrs,
    GridAttrs,
    InitialLot,
    InitialShipment,
    InitialState,
    InitialWip,
    Instance,
    Lane,
    Node,
    OsatAttrs,
    Params,
    Provenance,
    StockSlot,
    TerminalAttrs,
    frozen_map,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
SCHEMA_FILE = "instance.schema.json"  # the published file format of schema_version 1, package data in data/ (Q95)


class InstanceError(ValueError):
    """An instance file that breaks the contract of design §2.3."""


@functools.cache
def _schema_text() -> str:
    return resources.files("shockbench_flow.instance").joinpath("data", SCHEMA_FILE).read_text(encoding="utf-8")


def instance_schema() -> dict:
    """The published JSON Schema of the instance file (§2.3; Q95), as a fresh dict the caller may change."""
    return json.loads(_schema_text())


@functools.cache
def _schema_validator() -> Callable[[Any], Any]:
    """The schema compiled once per process, with ``use_default=False``.

    A schema default would be written into the parsed dict before the hash is taken, moving the instance hash,
    ``Instance.raw`` and ``Static.instance`` (docs/004 §3.5); the schema declares none, and this keeps it so.
    """
    return fastjsonschema.compile(json.loads(_schema_text()), use_default=False, use_formats=False)


def _schema_path(raw: Any, err) -> str:
    """Where a schema error is, as ``/``-joined keys: list elements by their string id, else their position (V20).

    Falls back to fastjsonschema's dotted name when a key holds one of its separators (``.``, ``[``, ``]``).
    """
    parts, x = [], raw
    for token in err.path[1:]:  # path[0] is fastjsonschema's root name "data"
        if isinstance(x, (list, tuple)) and token.isdigit() and int(token) < len(x):
            x = x[int(token)]
            parts.append(x["id"] if isinstance(x, dict) and isinstance(x.get("id"), str) else token)
        elif isinstance(x, dict) and token in x:
            x = x[token]
            parts.append(token)
        else:
            return err.name.removeprefix("data").lstrip(".")
    return "/".join(parts)


def _validate_schema(raw: Any) -> None:
    return None  # vendored: skipped


def _validate_schema_original(raw: Any) -> None:
    """Validate the parsed instance against the published schema (§2.3; Q95).

    Raises:
        InstanceError: with the schema's first error, where it is (``_schema_path``) and what it says; unknown keys
            are named in sorted order, the offending value is shown unless it is an object or an array.

    """
    try:
        _schema_validator()(raw)
    except fastjsonschema.JsonSchemaValueException as err:
        what = err.message.removeprefix(err.name).strip()
        if err.rule == "additionalProperties" and isinstance(err.value, dict) and isinstance(err.definition, dict):
            known = err.definition.get("properties", {})
            what = f"must not contain {sorted((k for k in err.value if k not in known), key=str)!r} properties"
        got = "" if isinstance(err.value, (dict, list, tuple)) else f", got {err.value!r}"
        where = _schema_path(raw, err) or "the instance"
        raise InstanceError(f"instance schema (§2.3): {where} {what}{got}") from err


def canonical_json(obj: Any) -> str:
    """Canonical form of design §2.3 (also used for ``generator_id``, §4.1)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(obj: Any) -> str:
    """SHA-256 hex digest of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def write_instance(raw: dict, path: str | Path) -> str:
    """Write an instance dict as sorted, indented JSON and return its hash (the canonical form is what is hashed)."""
    text = json.dumps(raw, sort_keys=True, indent=1, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return sha256_hex(raw)


def leaf_paths(raw: dict) -> list[str]:
    """Every numeric leaf parameter of an instance dict, as a ``/``-joined path (V20).

    List elements that are objects with an ``id`` are addressed by that id, other list elements by position. Booleans,
    strings and the ``provenance`` block are not leaves. Structural integers (``T``, lead times, weeks) are leaves.
    """
    out: list[str] = []

    def walk(x: Any, path: tuple[str, ...]) -> None:
        if isinstance(x, bool) or x is None or isinstance(x, str):
            return
        if isinstance(x, (int, float)):
            out.append("/".join(path))
        elif isinstance(x, dict):
            for key in sorted(x):
                walk(x[key], path + (str(key),))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                name = v["id"] if isinstance(v, dict) and "id" in v else str(i)
                walk(v, path + (name,))

    for key in sorted(raw):
        if key in ("provenance", "schema_version"):
            continue
        walk(raw[key], (key,))
    return out


def missing_provenance(raw: dict) -> list[str]:
    """Leaf paths without a provenance entry (V20); empty for a valid instance."""
    prov = raw.get("provenance", {})
    return [p for p in leaf_paths(raw) if p not in prov]


def placeholder_leaves(raw: dict) -> list[str]:
    """Leaf paths tagged SYNTHETIC(placeholder); a split with any cannot be sealed (V23)."""
    prov = raw.get("provenance", {})
    return sorted(p for p, v in prov.items() if str(v.get("tag", "")).startswith("SYNTHETIC(placeholder"))


def _alt(ref: dict | None, lane_index: dict, edge_index: dict) -> AltRef | None:
    if ref is None:
        return None
    if set(ref) == {"lane"}:
        return AltRef("lane", lane_index[ref["lane"]])
    if set(ref) == {"edge"}:
        return AltRef("edge", edge_index[ref["edge"]])
    raise InstanceError(f"alt_of must be {{'lane': id}} or {{'edge': id}}, got {ref!r}")


def _as_float(value: int | float) -> float:
    """``float(value)``, or an infinity of its sign for an integer beyond the float range (``float`` would overflow)."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _weeks(value: Any, label: str, minimum: int | None = None, ref: str = "(§2.3)") -> int:
    """A week, lead time or window in whole weeks, at least ``minimum`` if given; a whole float ``2.0`` reads as 2.

    Raises:
        InstanceError: for a boolean, a non-number, a fraction, a non-finite value (an integer beyond the float range
            included) or a value below ``minimum``.

    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(_as_float(value))  # NaN, infinity, and an integer no float can hold
        or (isinstance(value, float) and not value.is_integer())
        or (minimum is not None and value < minimum)
    ):
        bound = "" if minimum is None else f" >= {minimum}"
        raise InstanceError(f"{label} must be a whole number of weeks{bound} {ref}, got {value!r}")
    return int(value)


def _number(value: Any, label: str) -> float:
    """A finite real number, an int or a float (never a boolean, a string or null), as a float.

    Raises:
        InstanceError: for anything else, an integer beyond the float range or one no float equals included (the
            instance would simulate a value other than the one its hash covers).

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InstanceError(f"{label} must be a finite number, got {value!r}")
    x = _as_float(value)
    if not math.isfinite(x) or x != value:
        raise InstanceError(f"{label} must be a finite number a float represents exactly, got {value!r}")
    return x


def _at_least(value: Any, label: str, lo: float, ref: str) -> float:
    """A finite number >= ``lo`` (``lo`` a bound the design states, cited by ``ref``)."""
    x = _number(value, label)
    if x < lo:
        raise InstanceError(f"{label} must be >= {lo:g} {ref}, got {value!r}")
    return x


def _flag(value: Any, label: str) -> bool:
    """A JSON boolean; ``bool()`` of a string or a number would read ``"false"`` or 2 as a flag."""
    if not isinstance(value, bool):
        raise InstanceError(f"{label} must be a boolean, got {value!r}")
    return value


def _unique(ids: list, what: str) -> None:
    """Ids of one list are distinct: an index keeps the last duplicate, so a reference could not reach the first."""
    seen = set()
    for x in ids:
        if x in seen:
            raise InstanceError(f"{what} id {x!r} is not unique (§2.3)")
        seen.add(x)


_REQUIRED = object()


def _field(block: Mapping, key: str, label: str, default: Any = _REQUIRED) -> Any:
    """``block[key]``, or ``default`` when the design states one for an absent key; no default means required.

    Raises:
        InstanceError: if the key is absent and required.

    """
    if key in block:
        return block[key]
    if default is _REQUIRED:
        raise InstanceError(f"{label}: {key} is required, since the design gives it no default (§2.1, §2.3)")
    return default


def _capacity(e: dict) -> float | None:
    """The capacity u^0_e of an edge: null exactly on coupling edges, else a finite number >= 0 (§2.1, (4)).

    A coupling edge keeps u = inf in the marks and shows null in the observation; any other edge with a null or infinite
    u^0 would reach the naive flow LP and the observation as inf (INT-3). u^0 is read as every other number
    (``_number``): an integer beyond the float range, or one no float equals, is refused (DOC-4).
    """
    u0 = e.get("u0")
    if e["mode"] == "grid":
        if u0 is not None:
            raise InstanceError(f"coupling edge {e['id']}: u0 is null, since a coupling edge has no capacity (Q58 M5)")
        return None
    try:
        return _at_least(u0, f"edge {e['id']}: u0", 0.0, "(§2.1, (4))")
    except InstanceError as err:
        raise InstanceError(
            f"edge {e['id']}: u0 must be a finite number >= 0 off coupling edges (§2.1, (4)), one a float represents "
            f"exactly, got {u0!r}"
        ) from err


def _triple(x: Any, label: str, ref: str) -> tuple[float, float, float]:
    """Three finite numbers >= 0, one per war-risk class (h^Q, c-bar^wr) or conflict layer (Xi)."""
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise InstanceError(f"{label}: expected one value per war-risk class or conflict layer, got {x!r}")
    a, b, c = (_at_least(v, label, 0.0, ref) for v in x)
    return (a, b, c)


def _sorted_map(pairs) -> Mapping:
    """An immutable map in key order (integer codes or strings ascending, None last), whatever the JSON key order.

    The hash is over sorted keys (§2.3), so two files with equal hashes must give maps that iterate alike.
    """
    return frozen_map(dict(sorted(pairs, key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0))))


def _source_path(source: str | Path) -> Path:
    """The file of a packaged instance for a bare name (no suffix, no directory) when one exists, else ``source``.

    A bare name such as ``"tiny"`` resolves to ``data/tiny.json`` before any file of that name in the current
    directory; ``"./tiny"`` or any path with a directory or suffix is read as given.
    """
    name = str(source)
    packaged = DATA_DIR / f"{name}.json"
    if Path(name).name == name and not Path(name).suffix and packaged.is_file():
        return packaged
    return Path(source)


def _hash(raw: dict) -> str:
    """The instance hash (§2.3); a NaN or infinity anywhere, which canonical JSON cannot carry, is a breach.

    Raises:
        InstanceError: if ``canonical_json`` refuses a non-finite float.

    """
    try:
        return sha256_hex(raw)
    except (ValueError, TypeError) as err:
        raise InstanceError(
            f"the instance holds a non-finite number or a non-JSON value, which canonical JSON refuses (§2.3): {err}"
        ) from err


def _divergence(edges, lanes, li: int) -> int:
    """Where duplicate lane ``li`` diverges: its first edge off the lane it replaces, or out of the replaced tail.

    Raises:
        InstanceError: if the lane never leaves that route.

    """
    ln = lanes[li]
    ref = ln.alt_of
    if ref.kind == "lane":
        route = set(lanes[ref.index].edges)
        off = [x for x in ln.edges if x not in route]
    else:
        off = [x for x in ln.edges if edges[x].tail == edges[ref.index].tail and x != ref.index]
    if not off:
        raise InstanceError(f"duplicate lane {ln.id} never leaves the route it replaces (7)")
    return off[0]


def _lane_item(edges, lanes, li: int) -> tuple[int, int | None, int] | None:
    """The fleet-slack item of lane ``li`` if it is a sea duplicate, else None (see ``_dup_items``).

    Both transits are counted from where the lane leaves the route it replaces (§12 'Fleet slack order (7)'): for
    ``{"lane": id}`` the two lanes' summed lead times, whose shared prefix cancels; for ``{"edge": id}`` the lane's lead
    times from its divergence edge on, less that edge's, so a shared entry edge before the divergence is not extra.

    Raises:
        InstanceError: if the duplicate never leaves its route or is faster than the route it replaces.

    """
    ln = lanes[li]
    if ln.alt_of is None or any(edges[x].mode != "sea" for x in ln.edges):
        return None
    ref = ln.alt_of
    e = _divergence(edges, lanes, li)
    if ref.kind == "lane":
        d = sum(edges[x].tau for x in ln.edges) - sum(edges[x].tau for x in lanes[ref.index].edges)
    else:
        d = sum(edges[x].tau for x in ln.edges[ln.edges.index(e) :]) - edges[ref.index].tau
    if d < 0:
        raise InstanceError(f"duplicate lane {ln.id}: its transit is shorter than the route it replaces (7)")
    return (e, li, d) if e == ln.edges[0] else (e, None, d)


def _dup_items(edges, lanes) -> tuple[tuple[int, int | None, int], ...]:
    """Fleet-slack terms of (7): edge-level sea duplicates, then sea duplicate lanes where they diverge (§3.3).

    An edge-level item ``(e, None, Delta tau_e)`` of a sea edge marked ``alt_of`` counts all the flow of e, with
    Delta tau_e its lead time less the replaced lane's summed lead times or the replaced edge's. A sea duplicate lane
    l costs its transit less that of the route it replaces, both counted from where it leaves that route (see
    ``_lane_item``), charged on the edge where it leaves that route: on its entry edge the item is
    ``(e_1(l), l, Delta tau_l)`` and counts the flow dispatched on lane l; on an edge out of a chokepoint, a turn-back
    into the duplicate (design §3.3: "turn-back edges into them"), the item is edge-level, ``(e, None, Delta tau_l)``,
    since the queue there pools tanker cargo over lanes (Q34), so every lane on that edge must be a duplicate turning
    back there with the same Delta tau (one item for all).

    Raises:
        InstanceError: if a duplicate is faster than the route it replaces; a sea duplicate lane has an edge that is
            itself a sea duplicate, which (7) would count twice; or a turn-back item would charge a lane that is not a
            duplicate turning back there with the same Delta tau.

    """
    dup = [j for j, e in enumerate(edges) if e.mode == "sea" and e.alt_of is not None]  # E^dup (§3.3)
    items: list[tuple[int, int | None, int]] = []
    for j in dup:
        ref = edges[j].alt_of
        replaced = sum(edges[x].tau for x in lanes[ref.index].edges) if ref.kind == "lane" else edges[ref.index].tau
        if edges[j].tau < replaced:
            raise InstanceError(f"duplicate {edges[j].id}: its transit is shorter than the route it replaces (7)")
        items.append((j, None, edges[j].tau - replaced))
    marked = set(dup)
    per_lane = {li: it for li in range(len(lanes)) if (it := _lane_item(edges, lanes, li)) is not None}
    for li, it in per_lane.items():
        for x in lanes[li].edges:
            if x in marked:
                raise InstanceError(
                    f"edge {edges[x].id} is marked alt_of and lies on duplicate lane {lanes[li].id}: mark the lane or "
                    "the edge, since (7) would count its flow twice"
                )
        if it[1] is None:
            for lj, other in enumerate(lanes):
                if it[0] in other.edges and per_lane.get(lj) != it:
                    raise InstanceError(
                        f"lane {other.id} runs on {edges[it[0]].id}, where duplicate lane {lanes[li].id} turns back "
                        f"with extra transit {it[2]}: (7) charges every flow on that edge once (§3.3)"
                    )
        if it not in items:
            items.append(it)
    return tuple(items)


def load_instance(source: str | Path | dict, *, strict: bool = True) -> Instance:
    """Load an instance file (or its parsed dict) into frozen dataclasses (§2.3).

    Every map and every list the loader derives from a JSON object follows key order (commodity index, node index or
    the key itself), never the object's insertion order, so two files with equal hashes load to equal instances.

    Args:
        source: path to the JSON file, the name of a packaged instance (``"tiny"``: a bare name resolves to the
            packaged file first), or the parsed dict.
        strict: also require provenance for every leaf (V20).

    Raises:
        InstanceError: on any breach of the instance contract: of the published schema (``instance_schema``: an
            unknown key, a missing key or a value of the wrong JSON type), then of the loader's own checks (an unknown
            id included).

    """
    if isinstance(source, dict):
        raw = copy.deepcopy(source)  # the instance owns its parsed JSON: the caller's dict cannot move its hash
    else:
        try:
            raw = json.loads(_source_path(source).read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise InstanceError(f"{source}: not valid JSON (§2.3): {err}") from err
    try:
        return _build(raw, strict)
    except InstanceError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError) as err:  # a missing key, an unknown id, a wrong JSON type
        raise InstanceError(f"malformed instance (§2.3): {type(err).__name__}: {err}") from err


def _stock_slot(n: dict, i: int, k: int, kid: str, s: dict) -> StockSlot:
    """One stock block of node row ``n`` (§2.1; the file layout of §12), with the zeros the design states and no other.

    ``storage`` is I^max off chokepoints and null (uncapped) at them (Q58 M4). ``supply_rate`` is the availability of a
    supply node, required there, and absent or 0 elsewhere (Q79). ``holding_cost`` and ``salvage`` are 0 at supply
    nodes (Q79), and ``holding_cost`` is 0 at chokepoints (queued cargo pays h^Q, (23)), checked if stated; elsewhere
    they are required, since their rules (h_k = v_k r / 52 and nu = c^min x a share, §11 rows 1 and 3) carry per-split
    values the file declares; so is a chokepoint's salvage, valued by the c^min rule like every other node's (Q93).
    """
    label = f"stock {n['id']}/{kid}"
    supply_node, chokepoint = n["type"] in SUPPLY_TYPES, n["type"] == "chokepoint"
    storage = s.get("storage")
    if (storage is None) != chokepoint:
        raise InstanceError(f"node {n['id']}: storage is uncapped exactly at chokepoints (Q58 M4)")
    supply = _at_least(
        _field(s, "supply_rate", label, 0.0 if not supply_node else _REQUIRED), f"{label}: supply_rate", 0.0, "(Q79)"
    )
    no_holding = 0.0 if supply_node or chokepoint else _REQUIRED
    holding = _at_least(_field(s, "holding_cost", label, no_holding), f"{label}: holding_cost", 0.0, "(23)")
    no_salvage = 0.0 if supply_node else _REQUIRED
    salvage = _at_least(_field(s, "salvage", label, no_salvage), f"{label}: salvage", 0.0, "(23)")
    if supply and not supply_node:
        raise InstanceError(f"node {n['id']}: supply_rate only at source and material nodes (Q79)")
    if supply_node and (holding or salvage):
        raise InstanceError(f"node {n['id']}: supply-node stock has no holding cost and no salvage (Q79)")
    if chokepoint and holding:
        raise InstanceError(f"{label}: queued cargo pays h^Q, not a holding cost ((23), §2.1)")
    return StockSlot(
        node=i,
        k=k,
        storage=None if storage is None else _at_least(storage, f"{label}: storage", 0.0, "(8)"),
        holding=holding,
        salvage=salvage,
        supply=supply,
    )


def _grid(nid: str, g: dict, kx: dict) -> GridAttrs:
    """A grid block (§2.1, §3.5): shares >= 0 summing to 1, one buffer per fuel, the rationed fuel named or null."""
    label = f"grid {nid}"
    if g["priority"] not in PRIORITIES:
        raise InstanceError(f"grid {nid}: unknown priority {g['priority']!r}")
    shares = _sorted_map(
        ((None if k == UNMODELLED else kx[k]), _at_least(v, f"{label}: shares/{k}", 0.0, "(15)"))
        for k, v in g["shares"].items()
    )
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise InstanceError(f"grid {nid}: shares sum to {sum(shares.values())}, not 1")
    fuels = sorted(k for k in shares if k is not None)
    days_cover = _sorted_map(
        (kx[k], _at_least(v, f"{label}: days_cover/{k}", 0.0, "(16)")) for k, v in g["days_cover"].items()
    )
    ibar = _sorted_map((kx[k], _at_least(v, f"{label}: ibar/{k}", 0.0, "(16)")) for k, v in g["ibar"].items())
    names = {v: k for k, v in kx.items()}
    for key, table in (("days_cover", days_cover), ("ibar", ibar)):
        if extra := [names[k] for k in table if k not in fuels]:
            raise InstanceError(f"{label}: {key} lists {extra}, which are not among its fuels (§2.1)")
    rationed = _field(g, "rationed", label)  # the fuel whose segment (15) rations, or an explicit null (Q27)
    if rationed is not None:
        if kx[rationed] not in fuels:
            raise InstanceError(f"{label}: rationed fuel {rationed} is not one of its fuels ((15), Q27)")
        rationed = kx[rationed]
        for key, table in (("days_cover", days_cover), ("ibar", ibar)):  # (15) reads I-bar_g, which (16) takes from D_g
            if rationed not in table:
                raise InstanceError(f"{label}: {key} has no entry for its rationed fuel {names[rationed]} ((15), (16))")
    return GridAttrs(
        base_load=_at_least(g["base_load"], f"{label}: base_load", 0.0, "(18)"),
        deliverable=_at_least(g["deliverable"], f"{label}: deliverable", 0.0, "(15)"),
        shares=shares,
        days_cover=days_cover,
        ibar=ibar,
        rationed=rationed,
        voll=_at_least(g["voll"], f"{label}: voll", 0.0, "(23)"),
        priority=g["priority"],
    )


def _demand(nid: str, i: int, kid: str, k: int, d: dict) -> Demand:
    """One demand block of a sink (§2.1, (20)-(22)); (21) starts from the stationary law sigma / sqrt(1 - phi^2)."""
    label = f"sink {nid}: demand of {kid}"
    seasonal = d.get("seasonal")
    if seasonal is not None and len(seasonal) != WEEKS_PER_YEAR:
        raise InstanceError(f"sink {nid}: a seasonal profile has {WEEKS_PER_YEAR} values (§2.1)")
    phi = _number(d["phi"], f"{label}: phi")
    if not -1.0 < phi < 1.0:
        raise InstanceError(f"{label}: phi must lie in (-1, 1), for the stationary start of (21), got {d['phi']!r}")
    return Demand(
        node=i,
        k=k,
        dbar=_at_least(d["dbar"], f"{label}: dbar", 0.0, "(22)"),
        pi=_at_least(d["pi"], f"{label}: pi", 0.0, "(23)"),
        backlog=_flag(d["backlog"], f"{label}: backlog"),
        phi=phi,
        sigma=_at_least(d["sigma"], f"{label}: sigma", 0.0, "(21)"),
        seasonal=None if seasonal is None else tuple(_at_least(x, f"{label}: seasonal", 0.0, "(21)") for x in seasonal),
        shock=_triple(d["shock"], f"{label}: shock", "(21)"),
    )


def _params(p: dict) -> Params:
    """Instance-wide parameters (§2.4, §3.3, §3.5) within the ranges their equations give."""
    psi = _number(p["psi"], "params/psi")
    if not 0.0 < psi <= 1.0:
        raise InstanceError(f"params/psi must lie in (0, 1], the rationing threshold of (15) (Q36), got {p['psi']!r}")
    tau_alpha = _number(p["tau_alpha"], "params/tau_alpha")
    if not tau_alpha > 0.0:
        raise InstanceError(
            f"params/tau_alpha must be > 0, since (13) decays by exp(-1 / tau_alpha), got {tau_alpha!r}"
        )
    share = []
    for b in POOLS:
        s = _number(p["fleet_share"][b], f"params/fleet_share/{b}")
        if not 0.0 <= s <= 1.0:
            raise InstanceError(f"params/fleet_share/{b} must lie in [0, 1], a share of the fleet (7), got {s!r}")
        share.append(s)
    params = Params(
        psi=psi,
        alpha_max=_at_least(p["alpha_max"], "params/alpha_max", 1.0, "(13): an overproduction ceiling"),
        tau_alpha=tau_alpha,
        upsilon=_at_least(p["upsilon"], "params/upsilon", 0.0, "(22)"),
        fleet_share=(share[0], share[1]),
        fleet_measure=tuple(_at_least(p["fleet_measure"][b], f"params/fleet_measure/{b}", 0.0, "(7)") for b in POOLS),
        top_tariff=_at_least(p["top_tariff"], "params/top_tariff", 0.0, "(Q85)"),
        forecast_shares=tuple(_at_least(x, "params/forecast_shares", 0.0, "(48)") for x in p["forecast_shares"]),
    )
    if len(params.forecast_shares) != 9 or abs(sum(params.forecast_shares) - 1.0) > 1e-12:
        raise InstanceError("forecast_shares must be 9 shares summing to 1 (48)")
    return params


def _build(raw: dict, strict: bool) -> Instance:
    """The body of ``load_instance`` on the parsed dict (the caller maps a malformed dict to ``InstanceError``)."""
    if raw.get("schema_version") != SCHEMA_VERSION:  # the version picks the schema: another one is not read against it
        raise InstanceError(f"schema_version {raw.get('schema_version')!r}: this package reads {SCHEMA_VERSION!r}")
    _validate_schema(raw)  # the published file format first (Q95); the checks below keep the rules it cannot state
    if strict and (miss := missing_provenance(raw)):
        raise InstanceError(f"{len(miss)} leaves without provenance, e.g. {miss[:5]}")
    T = _weeks(raw["T"], "T", 1)
    _check_kind(raw["kind"])

    _unique(raw["regions"], "region")
    for what in ("commodities", "nodes", "edges", "lanes"):
        _unique([x["id"] for x in raw[what]], what[:-1] if what != "commodities" else "commodity")
    regions = tuple(raw["regions"])
    region_index = {r: i for i, r in enumerate(regions)}
    commodities = []
    for c in raw["commodities"]:
        label = f"commodity {c['id']}"
        if c["pool"] not in POOLS:
            raise InstanceError(f"{label}: pool must be one of {POOLS} (§2.1)")
        commodities.append(
            Commodity(
                c["id"],
                c["unit"],
                c["pool"],
                _at_least(c["v"], f"{label}: v", 0.0, "(23)"),
                _flag(c["override"], f"{label}: override"),
                _at_least(c["disposal_cost"], f"{label}: disposal_cost", 0.0, "(23)"),
            )
        )
    commodities = tuple(commodities)
    kx = {c.id: i for i, c in enumerate(commodities)}
    node_index = {n["id"]: i for i, n in enumerate(raw["nodes"])}
    edge_index = {e["id"]: i for i, e in enumerate(raw["edges"])}
    lane_index = {ln["id"]: i for i, ln in enumerate(raw["lanes"])}

    nodes, slots, demands = [], [], []
    for i, n in enumerate(raw["nodes"]):
        if n["type"] not in NODE_TYPES:
            raise InstanceError(f"node {n['id']}: unknown type {n['type']!r}")
        cp = gr = fb = os_ = tm = None
        if n["type"] == "chokepoint":
            c, label = n["chokepoint"], f"chokepoint {n['id']}"
            if not isinstance(c.get("class"), str):
                raise InstanceError(f"{label}: class must be a string (§2.1), got {c.get('class')!r}")
            cp = ChokepointAttrs(
                mu=tuple(_at_least(c["mu"][b], f"{label}: mu/{b}", 0.0, "(9)") for b in POOLS),
                k_c=_at_least(c["k_c"], f"{label}: k_c", 0.0, "(9)"),
                cls=c["class"],
                queue_holding=_sorted_map(
                    (kx[k], _triple(v, f"{label}: queue_holding/{k}", "(23)")) for k, v in c["queue_holding"].items()
                ),
                war_risk_cost=_sorted_map(
                    (kx[k], _triple(v, f"{label}: war_risk_cost/{k}", "(11)")) for k, v in c["war_risk_cost"].items()
                ),
            )
        elif n["type"] == "grid":
            gr = _grid(n["id"], n["grid"], kx)
        elif n["type"] == "fab":
            f, label = n["fab"], f"fab {n['id']}"
            fb = FabAttrs(
                cap0=_at_least(f["cap0"], f"{label}: cap0", 0.0, "(12)"),
                e=_at_least(f["e"], f"{label}: e", 0.0, "(12)"),
                grid=None if f.get("grid") is None else node_index[f["grid"]],
                cls=f["class"],
                input=kx[f["input"]],
                product=kx[f["product"]],
                tau=_weeks(f["tau"], f"{label}: tau", 1, "(12)"),
                w_scr=_weeks(f["w_scr"], f"{label}: w_scr", 1, "(14)"),
            )
            if not 1 <= fb.w_scr <= fb.tau:
                raise InstanceError(f"fab {n['id']}: w_scr must lie in [1, tau] (14)")
            if fb.cls not in FAB_CLASSES:
                raise InstanceError(f"fab {n['id']}: class {fb.cls!r} is not one of {FAB_CLASSES} (§2.1)")
            if fb.grid is None and fb.e != 0.0:
                raise InstanceError(
                    f"fab {n['id']}: e_f = 0 without a grid, since its energy is outside the model (§2.1)"
                )
        elif n["type"] == "osat":
            o, label = n["osat"], f"osat {n['id']}"
            packages = _sorted_map((kx[a], kx[b]) for a, b in o["packages"].items())
            made: dict[int, int] = {}
            for a, b in packages.items():  # (19): k^raw is *the* raw form of packaged chip k
                if b in made:
                    raise InstanceError(
                        f"{label}: packages {commodities[b].id} from two raw chips, {commodities[made[b]].id} and "
                        f"{commodities[a].id}, but (19) gives each packaged chip one raw form"
                    )
                made[b] = a
            for a in packages:
                if a in made:
                    raise InstanceError(f"{label}: {commodities[a].id} is both a raw and a packaged chip (19)")
            os_ = OsatAttrs(
                thr=_at_least(o["thr"], f"{label}: thr", 0.0, "(19)"),
                tau=_weeks(o["tau"], f"{label}: tau", 1, "(19)"),  # tau 0 would book xi^t on a popped week
                packages=packages,
            )
        elif n["type"] == "terminal":
            tp = _at_least(n["terminal"]["throughput"], f"terminal {n['id']}: throughput", 0.0, "(4)")
            tm = TerminalAttrs(throughput=tp)
        elif n["type"] == "sink":
            demand = n["sink"]["demand"]
            for k in sorted(demand, key=lambda k: kx[k]):  # demand ordinals: node order, then commodity index
                demands.append(_demand(n["id"], i, k, kx[k], demand[k]))
        for k in sorted(n.get("stock", {}), key=lambda k: kx[k]):
            slots.append(_stock_slot(n, i, kx[k], k, n["stock"][k]))
        nodes.append(Node(n["id"], n["type"], region_index[n["region"]], cp, gr, fb, os_, tm))
    slot_keys = {(sl.node, sl.k) for sl in slots}
    for d in demands:  # (20) serves from the sink's stock of k
        if (d.node, d.k) not in slot_keys:
            raise InstanceError(f"sink {nodes[d.node].id}: demand of {commodities[d.k].id} needs a stock slot (20)")

    lanes_raw = raw["lanes"]
    edges = []
    for e in raw["edges"]:
        if e["mode"] not in MODES:
            raise InstanceError(f"edge {e['id']}: unknown mode {e['mode']!r}")
        K = tuple(sorted(kx[k] for k in e["K"]))
        if len(set(K)) != len(K):  # two action slots for one (e, k) would clip, charge and observe it twice
            raise InstanceError(f"edge {e['id']}: K_e lists a commodity twice (§2.1)")
        pool = e.get("pool")
        if e["mode"] == "grid":
            if K or pool is not None:
                raise InstanceError(f"coupling edge {e['id']} carries no commodity (Q58 M5)")
        else:
            if pool not in POOLS or any(commodities[k].pool != pool for k in K):
                raise InstanceError(f"edge {e['id']}: every edge carries commodities of one pool only (§2.1)")
        edges.append(
            Edge(
                id=e["id"],
                tail=node_index[e["tail"]],
                head=node_index[e["head"]],
                mode=e["mode"],
                K=K,
                tau=_weeks(e["tau"], f"edge {e['id']}: tau", 0, "(2)"),  # arrival t' + tau_e >= t'
                c0=_at_least(e["c0"], f"edge {e['id']}: c0", 0.0, "(23)"),
                u0=_capacity(e),
                alt_of=_alt(e.get("alt_of"), lane_index, edge_index),
                pool=pool,
            )
        )
    for e in edges:
        if nodes[e.head].type == "chokepoint" and e.tau < 1:
            raise InstanceError(f"edge {e.id}: lead time into a chokepoint must be >= 1 week (§2.2, Q25)")
    supply_set = {i for i, n in enumerate(nodes) if n.type in SUPPLY_TYPES}
    for e in edges:
        if e.head in supply_set:  # (8): supply nodes refill from availability only, so nothing can push them over I^max
            raise InstanceError(f"edge {e.id}: supply nodes (source, material) have no in-edges (Q79)")
    lanes = []
    for ln in lanes_raw:
        es = tuple(edge_index[x] for x in ln["edges"])
        if len(es) < 2:
            raise InstanceError(f"lane {ln['id']}: a lane passes at least one chokepoint (Q4)")
        for a, b in zip(es, es[1:]):
            if edges[a].head != edges[b].tail or nodes[edges[a].head].type != "chokepoint":
                raise InstanceError(f"lane {ln['id']}: interior nodes must be chokepoints on a path (Q4)")
        interior = tuple(edges[a].head for a in es[:-1])
        for j, c in enumerate(interior):  # (52): e_l(c) is one edge per (lane, chokepoint), so a lane visits c once
            if c in interior[:j]:
                raise InstanceError(
                    f"lane {ln['id']} visits chokepoint {nodes[c].id} twice, so e_l(c) of (52) is not one edge"
                )
        if tuple(node_index[c] for c in ln.get("chokepoints", [])) != interior:
            raise InstanceError(f"lane {ln['id']}: declared chokepoints do not match its edges")
        if nodes[edges[es[0]].tail].type == "chokepoint" or nodes[edges[es[-1]].head].type == "chokepoint":
            # l = (i, c_1, ..., c_r, j): no action slot leaves a chokepoint, and a lot needs e_l(c) (52) to leave one
            raise InstanceError(f"lane {ln['id']}: a lane starts and ends off chokepoints (Q4)")
        lanes.append(Lane(ln["id"], es, interior, _alt(ln.get("alt_of"), lane_index, edge_index)))

    ist = raw["initial_state"]
    stock = _stock_block(ist["stock"], node_index, kx, "initial stock")
    start_rung, rung_stock = _rung_blocks(ist, stock, node_index, kx)
    pipeline = []
    for s in ist["pipeline"]:
        label = f"initial shipment on {s['edge']}"
        pipeline.append(
            InitialShipment(
                edge=edge_index[s["edge"]],
                k=kx[s["k"]],
                lane=None if s.get("lane") is None else lane_index[s["lane"]],
                qty=_number(s["qty"], f"{label}: qty"),
                dispatch_week=_weeks(s["dispatch_week"], f"{label}: dispatch_week"),
                arrival_week=_weeks(s["arrival_week"], f"{label}: arrival_week"),
            )
        )
    wips = {}
    for part, kind in (("fab_wip", "fab WIP"), ("osat_wip", "OSAT WIP")):
        wips[part] = tuple(
            InitialWip(
                node_index[w["node"]],
                kx[w["k"]],
                _number(w["qty"], f"initial {kind} at {w['node']}: qty"),
                _weeks(w["out_week"], f"initial {kind} at {w['node']}: out_week"),
            )
            for w in ist[part]
        )
    queue_lots = []
    for q in ist["queue_lots"]:
        label = f"initial queue lot at {q['chokepoint']}"
        queue_lots.append(
            InitialLot(
                chokepoint=node_index[q["chokepoint"]],
                k=kx[q["k"]],
                qty=_number(q["qty"], f"{label}: qty"),
                lane=lane_index[q["lane"]],
                next_edge=edge_index[q["next_edge"]],
                dispatch_week=_weeks(q["dispatch_week"], f"{label}: dispatch_week"),
                entry_edge=edge_index[q["entry_edge"]],
                arrival_week=_weeks(q["arrival_week"], f"{label}: arrival_week"),
            )
        )
    initial = InitialState(
        stock=tuple(stock),
        pipeline=tuple(pipeline),
        fab_wip=wips["fab_wip"],
        osat_wip=wips["osat_wip"],
        queue_lots=tuple(queue_lots),
    )
    params = _params(raw["params"])

    n_nodes = len(nodes)
    out_edges = tuple(tuple(j for j, e in enumerate(edges) if e.tail == i) for i in range(n_nodes))
    in_edges = tuple(tuple(j for j, e in enumerate(edges) if e.head == i) for i in range(n_nodes))
    by_type = {t: tuple(i for i, n in enumerate(nodes) if n.type == t) for t in NODE_TYPES}
    # lanes: commodities permitted on every edge, and the edges into and out of each interior chokepoint (52)
    lane_K = tuple(tuple(k for k in edges[ln.edges[0]].K if all(k in edges[x].K for x in ln.edges)) for ln in lanes)
    lane_through: dict[tuple[int, int], tuple[int, int]] = {}
    lanes_continuing: dict[tuple[int, int], list[int]] = {}
    for li, ln in enumerate(lanes):
        for a, b in zip(ln.edges, ln.edges[1:]):
            c = edges[a].head
            lane_through[(li, c)] = (a, b)
            cont = lanes_continuing.setdefault((c, b), [])
            if li not in cont:
                cont.append(li)
    chk = set(by_type["chokepoint"])
    action_slots = []
    for j, e in enumerate(edges):
        if e.tail in chk or e.coupling:
            continue
        for k in e.K:
            if e.head in chk:
                action_slots.extend((j, k, li) for li, ln in enumerate(lanes) if ln.edges[0] == j and k in lane_K[li])
            else:
                action_slots.append((j, k, None))
    override_slots = []
    for c in by_type["chokepoint"]:
        for k, com in enumerate(commodities):
            if not com.override:
                continue
            for j in out_edges[c]:
                if k not in edges[j].K:
                    continue
                # one slot per continuation lane that permits k on every edge; an out-edge into a non-chokepoint on
                # no such lane (a turn-back) gets one slot with lane None; cargo never enters a chokepoint laneless
                conts = [li for li in lanes_continuing.get((c, j), ()) if k in lane_K[li]]
                if conts:
                    override_slots.extend((c, k, j, li) for li in conts)
                elif edges[j].head not in chk:
                    override_slots.append((c, k, j, None))
    ordinal = {t: {n: i for i, n in enumerate(by_type[t])} for t in ("chokepoint", "grid", "fab", "osat")}

    inst = Instance(
        schema_version=raw["schema_version"],
        instance_id=raw["instance_id"],
        kind=raw["kind"],
        T=T,
        units=_sorted_map(raw["units"].items()),
        regions=regions,
        region_class=tuple(raw["region_class"][r] for r in regions),
        commodities=commodities,
        nodes=tuple(nodes),
        edges=tuple(edges),
        lanes=tuple(lanes),
        stock_slots=tuple(slots),
        demands=tuple(demands),
        routing_table=tuple(
            (region_index[r["origin"]], region_index[r["dest"]], tuple(lane_index[x] for x in r["lanes"]))
            for r in raw["routing_table"]
        ),
        compatibility=tuple((node_index[r["source"]], node_index[r["terminal"]]) for r in raw["compatibility"]),
        use=tuple((node_index[r["material"]], node_index[r["fab"]]) for r in raw["use"]),
        chokepoint_adjacency=_sorted_map(
            (node_index[c], tuple(region_index[r] for r in rs)) for c, rs in raw["chokepoint_adjacency"].items()
        ),
        trade_adjacency=tuple(  # weights of A >= 0: a counterpart is drawn from A, and Gamma scales A / spr(A) (§4.2)
            (
                region_index[a["a"]],
                region_index[a["b"]],
                _at_least(a["w"], f"trade_adjacency {a['a']}-{a['b']}: w", 0.0, "(§4.2)"),
            )
            for a in raw["trade_adjacency"]
        ),
        initial_state=initial,
        prohibitions_at_reset=tuple((edge_index[z["edge"]], kx[z["k"]]) for z in raw["prohibitions_at_reset"]),
        params=params,
        provenance=_sorted_map(
            (path, Provenance(v["tag"], v["source"], v.get("lo"), v.get("hi"))) for path, v in raw["provenance"].items()
        ),
        hash=_hash(raw),
        raw=raw,
        node_index=frozen_map(node_index),
        edge_index=frozen_map(edge_index),
        lane_index=frozen_map(lane_index),
        commodity_index=frozen_map(kx),
        region_index=frozen_map(region_index),
        out_edges=out_edges,
        in_edges=in_edges,
        chokepoints=by_type["chokepoint"],
        grids=by_type["grid"],
        fabs=by_type["fab"],
        osats=by_type["osat"],
        sinks=by_type["sink"],
        supply_nodes=tuple(i for i, n in enumerate(nodes) if n.type in SUPPLY_TYPES),
        slot_index=frozen_map({(s.node, s.k): j for j, s in enumerate(slots)}),
        dup_items=_dup_items(edges, lanes),
        action_slots=tuple(action_slots),
        override_slots=tuple(override_slots),
        action_slot_index=frozen_map({a: i for i, a in enumerate(action_slots)}),
        override_slot_index=frozen_map({o: i for i, o in enumerate(override_slots)}),
        chokepoint_ordinal=frozen_map(ordinal["chokepoint"]),
        grid_ordinal=frozen_map(ordinal["grid"]),
        fab_ordinal=frozen_map(ordinal["fab"]),
        osat_ordinal=frozen_map(ordinal["osat"]),
        commodity_pool=tuple(c.pool_index for c in commodities),
        grid_fabs=tuple(
            tuple(fi for fi, f in enumerate(by_type["fab"]) if nodes[f].fab.grid == g) for g in by_type["grid"]
        ),
        lane_K=lane_K,
        lane_through=frozen_map(lane_through),
        lanes_continuing=frozen_map({key: tuple(v) for key, v in lanes_continuing.items()}),
        continuing_edges=frozenset(b for _c, b in lanes_continuing),
        start_rung=start_rung,
        rung_stock=rung_stock,
    )
    _check_couplings(inst)
    _check_chokepoint_entries(inst)
    _check_stock_slots(inst)
    _check_salvage_invariant(inst)
    _check_initial_state(inst)
    _check_tables(inst)
    if inst.kind in FLAGSHIP_KINDS:
        _check_flagship_structure(inst)
    return inst


def _stock_block(entries: list, node_index: Mapping, kx: Mapping, label: str) -> tuple[tuple[int, int, float], ...]:
    """A stock block of the file as (node, k, qty) entries in file order (§2.3); ids must exist."""
    return tuple(
        (node_index[s["node"]], kx[s["k"]], _number(s["qty"], f"{label} at {s['node']}/{s['k']}: qty")) for s in entries
    )


def _rung_blocks(ist: Mapping, stock: tuple, node_index: Mapping, kx: Mapping) -> tuple[float | None, Mapping]:
    """(the rung of ``stock``, every rung's block) of the warm start per rung (§2.3; M5-O37 (b)); (None, {}) without.

    The schema makes ``rung`` and ``stock_by_rung`` come together; here ``rung`` is a number in (0, 1) and each key of
    ``stock_by_rung`` the canonical decimal repr of another number in (0, 1), so one rung has one key.

    Raises:
        InstanceError: on a rung outside (0, 1), a key that is not a rung's canonical repr, or the file's own rung.

    """
    if "rung" not in ist:
        return None, frozen_map({})
    own = _number(ist["rung"], "initial_state rung")
    if not 0.0 < own < 1.0:
        raise InstanceError(f"initial_state rung must lie in (0, 1), got {own!r} (§2.3)")
    blocks = {own: stock}
    for key, entries in ist["stock_by_rung"].items():
        try:
            g = float(key)
        except ValueError:
            g = math.nan
        if not 0.0 < g < 1.0 or repr(g) != key:
            raise InstanceError(
                f"initial_state stock_by_rung: key {key!r} is not the canonical decimal repr of a rung in (0, 1) (§2.3)"
            )
        if g == own:
            raise InstanceError(f"initial_state stock_by_rung: {key!r} is the file's own rung, whose block is stock")
        blocks[g] = _stock_block(entries, node_index, kx, f"rung {key} initial stock")
    return own, frozen_map({g: blocks[g] for g in sorted(blocks)})


def _check_kind(kind: Any) -> None:
    """The instance kind is one of ``INSTANCE_KINDS``, whose index is the kind code of the seed keys (§4.1; Q87).

    The schema types ``kind`` only as a string (§2.3); an unknown kind would reach the stream keys of naive's F_Q and
    the generator profiles (``codes.INSTANCE_KINDS.index``) and fail there, far from the file.
    """
    if kind not in INSTANCE_KINDS:
        raise InstanceError(f"kind must be one of {INSTANCE_KINDS} (§2.3, §4.1), got {kind!r}")


def _check_flagship_structure(inst: Instance) -> None:
    """The structure §2.2 requires of `small` and `full` (design §12 'Instance kind and structure checks').

    Every fab has a grid (v0.2 §16), the graph is weakly connected over its non-coupling edges, and every sink demand
    is reachable from a supply node: through non-coupling edges between non-chokepoint nodes, whole lanes (cargo
    crosses a chokepoint only on a lane, Q4), a fab's conversion of its input into its product (12) and an OSAT's
    packaging (19). Z_0 is ignored here: it is a state of the episode, not of the graph (a demand it closes, such as
    `chip_le` into CN, still has a route). `tiny` is not checked: its `fab_cn` has no grid by design (§2.4).

    Raises:
        InstanceError: naming the first fab without a grid, the component sizes of a disconnected graph, or the first
            unreachable sink demand.

    """
    nodes, K = inst.nodes, inst.commodities
    for f in inst.fabs:
        if nodes[f].fab.grid is None:
            raise InstanceError(f"fab {nodes[f].id}: every fab of a {inst.kind} instance has a grid (§2.2)")
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in inst.edges:
        if not e.coupling:
            parent[find(e.tail)] = find(e.head)
    sizes: dict[int, int] = {}
    for i in range(len(nodes)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    if len(sizes) > 1:
        raise InstanceError(
            f"the graph over non-coupling edges is not connected (§2.2): components of {sorted(sizes.values())} nodes"
        )
    chk = set(inst.chokepoints)
    moves: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for e in inst.edges:
        if not e.coupling and e.tail not in chk and e.head not in chk:
            for k in e.K:
                moves.setdefault((e.tail, k), set()).add((e.head, k))
    for li in range(len(inst.lanes)):
        origin = inst.edges[inst.lanes[li].edges[0]].tail
        for k in inst.lane_K[li]:
            moves.setdefault((origin, k), set()).add((inst.lane_destination(li), k))
    for f in inst.fabs:
        fab = nodes[f].fab
        moves.setdefault((f, fab.input), set()).add((f, fab.product))
    for o in inst.osats:
        for raw_k, packaged in nodes[o].osat.packages.items():
            moves.setdefault((o, raw_k), set()).add((o, packaged))
    reached = {(s.node, s.k) for s in inst.stock_slots if s.node in set(inst.supply_nodes)}
    frontier = list(reached)
    while frontier:
        for nxt in moves.get(frontier.pop(), ()):
            if nxt not in reached:
                reached.add(nxt)
                frontier.append(nxt)
    for d in inst.demands:
        if (d.node, d.k) not in reached:
            raise InstanceError(
                f"sink {nodes[d.node].id}: its demand of {K[d.k].id} is reachable from no supply node (§2.2)"
            )


def _check_couplings(inst: Instance) -> None:
    """Coupling edges go from a grid to a fab, one per fab with a grid (§2.1, §2.2; Q58 M5).

    A fab's ``grid`` is the tail of its one coupling edge, and a fab without a grid has none; a terminal's throughput
    is the capacity of each of its edges to a grid (§2.2).
    """
    nodes = inst.nodes
    coupled: dict[int, list[int]] = {f: [] for f in inst.fabs}
    for e in inst.edges:
        if e.coupling:
            if nodes[e.tail].type != "grid" or nodes[e.head].type != "fab":
                raise InstanceError(f"coupling edge {e.id}: coupling edges go from a grid to a fab (§2.1, Q58 M5)")
            coupled[e.head].append(e.tail)
    for f in inst.fabs:
        if coupled[f] != ([] if nodes[f].fab.grid is None else [nodes[f].fab.grid]):
            raise InstanceError(
                f"fab {nodes[f].id}: its grid must be the tail of its one coupling edge, and a fab without a grid has "
                "no coupling edge (§2.1)"
            )
    for i, n in enumerate(nodes):
        if n.type != "terminal":
            continue
        for j in inst.out_edges[i]:
            e = inst.edges[j]
            if nodes[e.head].type == "grid" and e.u0 != n.terminal.throughput:
                raise InstanceError(
                    f"terminal {n.id}: throughput {n.terminal.throughput!r} must equal the capacity of each edge to a "
                    f"grid, but {e.id} has u0 {e.u0!r} (§2.2)"
                )


def _check_chokepoint_entries(inst: Instance) -> None:
    """Every edge into a chokepoint lies on a lane that carries each of its commodities on every edge (Q4).

    Cargo enters a chokepoint only on a lane: an action slot exists for a lane's first edge (52) and an override slot
    or default release for its continuation, so an edge into a chokepoint on no such lane could never be used.
    """
    for j, e in enumerate(inst.edges):
        if e.head not in inst.chokepoint_ordinal:
            continue
        on = [li for li, ln in enumerate(inst.lanes) if j in ln.edges]
        if not on:
            raise InstanceError(f"edge {e.id} enters chokepoint {inst.nodes[e.head].id} on no lane (Q4)")
        for k in e.K:
            if not any(k in inst.lane_K[li] for li in on):
                raise InstanceError(
                    f"edge {e.id}: no lane through {e.id} carries {inst.commodities[k].id} on every edge, so no action "
                    "or override slot can use it (Q4)"
                )


def _check_stock_slots(inst: Instance) -> None:
    """Every stock the simulator and the LP index exists as a stock slot I_ik of (8), checked once here.

    Each non-coupling edge's tail and head off chokepoints hold a slot of every k in K_e (dispatch (5) draws from the
    tail, arrival (2) adds to the head); each chokepoint on a lane holds a slot of every commodity the lane permits
    (its queue I_ck, charged h^Q in (23)); a fab holds its input and its product (12), a grid its fuels (15), an OSAT
    the raw and packaged chips it packages (19). A chokepoint lists h^Q and the war-risk cost for exactly the
    commodities of its slots (§2.1: per commodity it can hold), so none is missing and none is left unread.
    """
    nodes, K, slot = inst.nodes, inst.commodities, inst.slot_index
    for e in inst.edges:
        if e.coupling:
            continue
        for k in e.K:
            for end, i in (("tail", e.tail), ("head", e.head)):
                if i not in inst.chokepoint_ordinal and (i, k) not in slot:
                    raise InstanceError(
                        f"edge {e.id}: {end} {nodes[i].id} has no stock slot of {K[k].id}, which (2), (5) and (8) "
                        "move its flow through"
                    )
    for li, ln in enumerate(inst.lanes):
        for c in ln.chokepoints:
            for k in inst.lane_K[li]:
                if (c, k) not in slot:
                    raise InstanceError(
                        f"lane {ln.id} carries {K[k].id} through {nodes[c].id}, which has no stock slot of it (its "
                        "queue I_ck of (8), charged h^Q in (23))"
                    )
    for f in inst.fabs:
        fab = nodes[f].fab
        for role, k in (("input", fab.input), ("product", fab.product)):
            if (f, k) not in slot:
                raise InstanceError(f"fab {nodes[f].id}: no stock slot of its {role} {K[k].id} (12)")
    for g in inst.grids:
        for k in nodes[g].grid.fuels:
            if (g, k) not in slot:
                raise InstanceError(f"grid {nodes[g].id}: no stock slot of its fuel {K[k].id} (15)")
    for o in inst.osats:
        for pair in nodes[o].osat.packages.items():
            for k in pair:
                if (o, k) not in slot:
                    raise InstanceError(f"osat {nodes[o].id}: no stock slot of {K[k].id}, which it packages (19)")
    for c in inst.chokepoints:
        held = [sl.k for sl in inst.stock_slots if sl.node == c]
        attrs = nodes[c].chokepoint
        for key, table in (("queue_holding", attrs.queue_holding), ("war_risk_cost", attrs.war_risk_cost)):
            if sorted(table) != sorted(held):
                raise InstanceError(
                    f"chokepoint {nodes[c].id}: {key} lists {[K[k].id for k in table]} but its stock slots hold "
                    f"{[K[k].id for k in sorted(held)]} (§2.1: one entry per commodity the chokepoint can hold)"
                )


def _check_salvage_invariant(inst: Instance) -> None:
    """The salvage invariant nu_head - nu_tail - c0 <= 0 (§11 row 1; Q78, Q93), checked rather than assumed.

    Under Q93 a unit still in transit at T is credited at the nu of its edge's head, so an edge whose head is worth
    more than its tail plus the freight would pay out at T, bounded only by its capacity: ship into the head in week T.
    Checked on every non-coupling edge and commodity open at reset (not in Z_0, which stays in force all episode),
    chokepoint ends included; nu is the slot's salvage, 0 at supply nodes (Q79), and an end without a slot of k holds
    none to credit. The check is the relaxation nu_head <= nu_tail + c0, which a c^min formed by a shortest-path
    relaxation meets exactly. A lane taken whole then holds as well: its edges' inequalities chain through its
    chokepoints, and rounded addition is monotone, so nu_dest <= (nu_origin + c0_1) + ... + c0_n in floats too.
    """
    nodes, K, slot, supply = inst.nodes, inst.commodities, inst.slot_index, set(inst.supply_nodes)
    z0 = set(inst.prohibitions_at_reset)

    def nu(node: int, k: int) -> float | None:
        if (node, k) not in slot:
            return None
        return 0.0 if node in supply else inst.stock_slots[slot[(node, k)]].salvage

    for j, e in enumerate(inst.edges):
        if e.coupling:
            continue
        for k in e.K:
            tail, head = nu(e.tail, k), nu(e.head, k)
            if (j, k) in z0 or tail is None or head is None:
                continue
            if head > tail + e.c0:
                raise InstanceError(
                    f"edge {e.id}: salvage of {K[k].id} at {nodes[e.head].id} ({head!r}) exceeds that at "
                    f"{nodes[e.tail].id} ({tail!r}) plus the freight c0 ({e.c0!r}), so a unit in transit at T would "
                    "earn more than its freight (salvage invariant, §11 row 1; Q93)"
                )


def _check_tables(inst: Instance) -> None:
    """The compatibility, use, routing and chokepoint-adjacency tables agree with the nodes, edges and lanes (§2.3).

    A compatibility pair joins a source to a terminal and a use pair a material to a fab, each listed once; each pair
    is joined by an edge or a lane, and each edge or lane from a source to a terminal, or from a material to a fab, is
    a listed pair (§2.2: each source ships to terminals from the compatibility table, each material to every fab using
    it), so the edges and lanes the simulator reads say what the tables say. A routing-table key (origin region,
    destination region) appears once, and each of its lanes runs from a node of the origin region to one of the
    destination region (§2.1). The chokepoint adjacency maps chokepoints only.
    """
    nodes, regions = inst.nodes, inst.regions
    joined: dict[tuple[int, int], str] = {}
    for e in inst.edges:
        if not e.coupling:
            joined.setdefault((e.tail, e.head), f"edge {e.id}")
    for li, ln in enumerate(inst.lanes):
        joined.setdefault((inst.edges[ln.edges[0]].tail, inst.lane_destination(li)), f"lane {ln.id}")
    for name, table, (ta, tb) in (
        ("compatibility", inst.compatibility, ("source", "terminal")),
        ("use", inst.use, ("material", "fab")),
    ):
        listed: set[tuple[int, int]] = set()
        for a, b in table:
            pair = f"{name} pair {nodes[a].id} -> {nodes[b].id}"
            if nodes[a].type != ta or nodes[b].type != tb:
                raise InstanceError(f"{pair}: a pair joins a {ta} to a {tb} (§2.3)")
            if (a, b) in listed:
                raise InstanceError(f"{pair} is listed twice (§2.3)")
            listed.add((a, b))
            if (a, b) not in joined:
                raise InstanceError(f"{pair}: no edge or lane joins them (§2.2)")
        for (a, b), what in joined.items():
            if nodes[a].type == ta and nodes[b].type == tb and (a, b) not in listed:
                raise InstanceError(
                    f"{what} joins {ta} {nodes[a].id} to {tb} {nodes[b].id}, a pair the {name} table does not list "
                    "(§2.2)"
                )
    keys: set[tuple[int, int]] = set()
    for o, d, lanes in inst.routing_table:
        entry = f"routing_table {regions[o]} -> {regions[d]}"
        if (o, d) in keys:
            raise InstanceError(f"{entry}: listed twice, but the table maps each region pair to its lanes (§2.1)")
        keys.add((o, d))
        for li in lanes:
            a, b = nodes[inst.edges[inst.lanes[li].edges[0]].tail], nodes[inst.lane_destination(li)]
            if (a.region, b.region) != (o, d):
                raise InstanceError(
                    f"{entry}: lane {inst.lanes[li].id} runs from {regions[a.region]} to {regions[b.region]} (§2.1)"
                )
    for c in inst.chokepoint_adjacency:
        if nodes[c].type != "chokepoint":
            raise InstanceError(f"chokepoint_adjacency: {nodes[c].id} is not a chokepoint (§2.1)")


def _check_initial_state(inst: Instance) -> None:
    """The declared initial state against the graph (§2.3, §2.4; Q58 M13), entry by entry as the simulator places it.

    Stock, and every other rung's block of the warm start per rung (§2.3; M5-O37 (b)), sits in a stock slot within
    [0, I^max] (8), repeated entries of a slot adding up, and a declared chokepoint
    stock (the sum of its entries) equals the sum of that chokepoint's queue lots of the commodity within
    1e-9 max(1, declared), the lots summed as the simulator and the LP take I^0 from them (``initial_stock``: the fsum
    over lanes of each lane's lots in book order); fab WIP is the fab's product k^raw(f) maturing in weeks
    1..tau^fab (12); OSAT WIP is a packaged commodity of the OSAT maturing in weeks 1..tau^osat (19); a pipeline
    shipment carries a permitted commodity, dispatched at t' <= 0 and arriving at t' + tau >= 1, and a lane when it
    enters a chokepoint (Q4); a queue lot waits at a chokepoint of its lane, entered on the lane's edge into it, bound
    for e_l(c) (52), arrived by week 0. Quantities are non-negative.
    """
    nodes, edges, lanes, K = inst.nodes, inst.edges, inst.lanes, inst.commodities
    ist = inst.initial_state
    slot = inst.slot_index

    def lane_carries(label: str, lane: int, k: int) -> None:
        if k not in inst.lane_K[lane]:
            raise InstanceError(f"{label}: lane {lanes[lane].id} does not carry {K[k].id} on every edge")

    def has_slot(label: str, i: int, k: int) -> None:
        if (i, k) not in slot:
            raise InstanceError(f"{label}: no stock slot {nodes[i].id}/{K[k].id}")

    def non_negative(label: str, q: float) -> None:
        if not q >= 0.0:
            raise InstanceError(f"{label}: negative quantity {q!r}")

    def declared_stock(entries, label: str) -> dict[tuple[int, int], float]:
        declared: dict[tuple[int, int], float] = {}
        for i, k, q in entries:
            if (i, k) not in slot:
                raise InstanceError(f"{label} at {nodes[i].id}/{K[k].id}: no such stock slot")
            cap = inst.stock_slots[slot[(i, k)]].storage
            declared[(i, k)] = declared.get((i, k), 0.0) + q  # entries add up, as in dynamics.sim.initial_stock
            if not q >= 0.0 or (cap is not None and declared[(i, k)] > cap):  # (8): 0 <= I^0 <= I^max off chokepoints
                raise InstanceError(f"{label} at {nodes[i].id}/{K[k].id} is outside [0, I^max]")
        return declared

    blocks = [("initial stock", declared_stock(ist.stock, "initial stock"))]
    for g, entries in inst.rung_stock.items():  # the warm start per rung (§2.3): each block obeys the rules of stock
        if g != inst.start_rung:
            label = f"rung {g!r} initial stock"
            blocks.append((label, declared_stock(entries, label)))
    for w in ist.fab_wip:
        label, fab = f"initial fab WIP at {nodes[w.node].id}", nodes[w.node].fab
        if fab is None:
            raise InstanceError(f"{label}: not a fab")
        if w.k != fab.product:
            raise InstanceError(f"{label}: a fab's WIP is its product {K[fab.product].id}, not {K[w.k].id} (12)")
        if not 1 <= w.out_week <= fab.tau:
            raise InstanceError(f"{label}: out_week must lie in [1, tau^fab] = [1, {fab.tau}] (§2.3)")
        non_negative(label, w.qty)
    for w in ist.osat_wip:
        label, osat = f"initial OSAT WIP at {nodes[w.node].id}", nodes[w.node].osat
        if osat is None:
            raise InstanceError(f"{label}: not an OSAT")
        if w.k not in osat.packages.values():
            raise InstanceError(f"{label}: {K[w.k].id} is not a packaged commodity it makes (19)")
        if not 1 <= w.out_week <= osat.tau:
            raise InstanceError(f"{label}: out_week must lie in [1, tau^osat] = [1, {osat.tau}] (§2.3)")
        non_negative(label, w.qty)
    for s in ist.pipeline:
        e = edges[s.edge]
        label = f"initial shipment on {e.id}"
        if s.arrival_week != s.dispatch_week + e.tau or s.dispatch_week > 0 or s.arrival_week < 1:
            raise InstanceError(f"{label}: needs t' <= 0 and arrival t' + tau >= 1")
        if s.k not in e.K:
            raise InstanceError(f"{label}: {K[s.k].id} is not permitted on the edge (K_e)")
        non_negative(label, s.qty)
        if s.lane is None:
            if e.head in inst.chokepoint_ordinal:
                raise InstanceError(f"{label}: cargo enters a chokepoint only on a lane (Q4)")
        else:
            if s.edge not in lanes[s.lane].edges:
                raise InstanceError(f"{label}: lane {lanes[s.lane].id} does not contain the edge")
            lane_carries(label, s.lane, s.k)
        has_slot(label, e.head, s.k)
    for q in ist.queue_lots:
        c = q.chokepoint
        label = f"initial queue lot at {nodes[c].id}"
        if nodes[c].type != "chokepoint":
            raise InstanceError(f"{label}: not a chokepoint")
        if c not in lanes[q.lane].chokepoints:
            raise InstanceError(f"{label}: lane {lanes[q.lane].id} does not pass {nodes[c].id}")
        if q.entry_edge != inst.lane_through[(q.lane, c)][0]:
            raise InstanceError(f"{label}: entry_edge must be the lane's edge into the chokepoint")
        if q.next_edge != inst.lane_next_edge(q.lane, q.entry_edge):
            raise InstanceError(f"{label}: next_edge must be e_l(c), the lane's edge out of the chokepoint (52)")
        lane_carries(label, q.lane, q.k)
        if q.arrival_week > 0:
            raise InstanceError(f"{label}: needs arrival_week <= 0")
        if q.arrival_week != q.dispatch_week + edges[q.entry_edge].tau:
            raise InstanceError(f"{label}: needs arrival_week = dispatch_week + tau of the entry edge")
        non_negative(label, q.qty)
        has_slot(label, c, q.k)
    # I^0 at a chokepoint as dynamics.sim.initial_stock forms it (§12 'Loader checks', 'Cost terms (23)'): each lane's
    # lots added in book order (sim.lane_queues), then the fsum over lanes (cost.queue_totals). Imported here, since the
    # dynamics package imports this module.
    from shockbench_flow.dynamics.cost import queue_totals
    from shockbench_flow.dynamics.sim import lane_queues

    lot_total = queue_totals(lane_queues(ist.queue_lots))
    for label, declared in blocks:  # a chokepoint's declared stock is the sum of its entries, as at every slot
        for (i, k), q in declared.items():
            if i in inst.chokepoint_ordinal:
                total = lot_total.get((i, k), 0.0)
                if abs(total - q) > 1e-9 * max(1.0, q):  # the tolerance of dynamics.sim.initial_stock
                    raise InstanceError(
                        f"{label} at {nodes[i].id}/{K[k].id} ({q!r}) differs from the sum of its queue lots "
                        f"({total!r}) (§2.3)"
                    )

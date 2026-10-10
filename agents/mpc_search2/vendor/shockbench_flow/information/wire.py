"""The §9.2 line-JSON wire schema and the line-level rules of §9.3 (design §9.2-9.4; Q68, Q71, Q86, Q95).

Stdlib only: the shapes are ``typing.TypedDict``s, which the tests validate through a strict pydantic subclass
(``tests/_wire_models.py``; ``ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)``) and whose JSON schema is
published as ``information/data/wire.schema.json`` (Q95); nothing here imports pydantic. Vocabulary fields are
``Literal`` types built from the frozen tuples (``instance.schema``, ``omega.codes``). ``schema_version`` stays '1'
until M3 freezes the wire (design §12 M3 rows).

Framing (§9.2): one JSON object per line, UTF-8, ``json.dumps(msg, allow_nan=False, separators=(',', ':'))`` plus
a newline, floats by repr (an exact round trip); every array a JSON list, a missing value null (never NaN), integers
indexing the Static tables, no tuple keys.

Exchange (design §12 M3 rows): the Reset gets no reply; one Request and one Reply per week t = 1..T; the week-1
Request repeats the Reset's obs; after week T's reply nothing is sent and stdin is closed (the final observation exists
in process only). ``episode`` and ``nonce`` are one fresh 128-bit hex token each per (submission, episode) (Q86; §9.2,
§9.4 "Tags"), drawn outside omega and never entering the Trajectory, its hash or the action log (FR2 M6); every
Request of the episode carries the same two. A reply whose action names an earlier week (``action.week`` an int < t)
is stale: discarded, and reading goes on until the deadline (Q71 X5); a wrong or missing episode or nonce is a
whole-week failure, and a missing, wrong or later week is the environment's whole-week failure (``validate_action``).
NaN and Infinity tokens are accepted (the entry drop of §9.3 row 1, as in process); a duplicate object key, non-UTF-8
bytes, a nesting deeper than ``MAX_REPLY_DEPTH``, an integer of more than ``MAX_INT_DIGITS`` digits or a JSON value
that is not an object are unparsable: both limits are fixed constants checked before or while parsing, so a line's
verdict never depends on the interpreter (its C recursion budget at the call site, ``sys.set_int_max_str_digits``;
M3 attempt 1's gate, DET-M3-1). The maximum reply length and the bank are
SYNTHETIC(placeholder) config values until Q13, read by ``scripts/python/run_wire_episode.py`` alone; the local runner
sends ``remaining_bank`` but does not meter it (M6).

A whole-week wire failure reaches the environment as ``Env.step(None, wire_failure=<code>)`` with a code of
``WIRE_FAILURES``: the runner alone produces it, the trajectory stores it in its own field
(``Trajectory.wire_failures``), and no reply value can forge it (a reply whose action is the string of a code is 'the
action is not a dict').
"""

import array
import itertools
import json
import math
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np

from shockbench_flow.instance.schema import MODES, NODE_TYPES, POOLS, WAR_RISK_CLASSES
from shockbench_flow.instance.schema import SCHEMA_VERSION as SCHEMA_VERSION  # the one schema_version, re-exported
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega


# ----- vocabularies (§9.2; Q86), frozen with schema_version ----------------------------------------------------------
NodeType = Literal[NODE_TYPES]
Mode = Literal[MODES]
Pool = Literal[POOLS]
WarRisk = Literal[WAR_RISK_CLASSES]
UnitKind = Literal[codes.UNIT_KINDS]
Channel = Literal[codes.CHANNELS]
MessageKind = Literal[codes.MESSAGE_KINDS]
TargetKind = Literal[codes.TARGET_KINDS]
BlackoutSpell = Literal[codes.BLACKOUT_SPELLS]

# whole-week wire failures (§9.3 row 2; D9): code -> the reason logged as StepRecord.invalid[0] after
# 'whole-week failure: '; the reason never echoes a reply value
WIRE_FAILURES: dict[str, str] = {
    "unparsable": (
        "the reply line is not a JSON object within the wire's limits (not UTF-8, not JSON, a duplicate key, nested"
        " too deep, an integer too long, or not an object)"
    ),
    "too_long": "the reply line exceeds the maximum length",
    "tags": "the reply's episode or nonce is missing or wrong",
    "timeout": "no reply by the deadline",
    "killed": "the policy process is gone (it exited, crashed or was killed; under M6, the overage bank is empty)",
    # the hosted runner's meter, never a reply line (``runner.WeekMeter``; Q113): the week's CPU, read from the policy
    # container's cgroup outside it, went over the week's budget
    "cpu": "the week used more CPU than its budget (metered from the policy container's cgroup)",
}


# ----- Static (§9.2), with the M3 additions: dyads, regime.skill, regime.blackout, a nullable regime -----------------
class StaticRegime(TypedDict):
    name: str
    L: int | None
    a: dict[UnitKind, float]
    phi: dict[Channel, float]
    chi: bool
    h_cov: int | None
    skill: dict[UnitKind, float | None]  # declared AUROC_13 (46); null until V27 measures a chokepoint's (Q80)
    blackout: BlackoutSpell | None


class AltLane(TypedDict):
    lane: int


class AltEdge(TypedDict):
    edge: int


class StaticNodes(TypedDict):
    id: list[str]
    type: list[NodeType]
    region: list[int]


class StaticCommodities(TypedDict):
    id: list[str]
    v: list[float]
    pool: list[Pool]
    override: list[bool]


class StaticEdges(TypedDict):
    id: list[str]
    tail: list[int]
    head: list[int]
    mode: list[Mode]
    tau0: list[int]
    c0: list[float]
    u0: list[float | None]  # null on coupling edges (§12 M1 row)
    K: list[list[int]]
    alt_of: list[AltLane | AltEdge | None]
    pool: list[Pool | None]


class StaticLanes(TypedDict):
    id: list[str]
    edges: list[list[int]]
    chokepoints: list[list[int]]
    alt_of: list[AltLane | AltEdge | None]


class ActionSlots(TypedDict):
    edge: list[int]
    k: list[int]
    lane: list[int | None]


class OverrideSlots(TypedDict):
    chokepoint: list[int]
    k: list[int]
    out_edge: list[int]
    lane: list[int | None]


class Sinks(TypedDict):
    node: list[int]
    k: list[int]
    backlog: list[bool]
    pi: list[float]


class StaticDyads(TypedDict):
    a: list[int]  # region index of each dyad's first region, in z_dyad row order (the warning's dyad unit id)
    b: list[int]


class Static(TypedDict):
    instance: dict[str, Any]  # the canonical instance JSON of §2.3 (public, Q70)
    instance_id: str
    instance_hash: str
    T: int
    units: dict[str, str]
    regions: list[str]
    nodes: StaticNodes
    commodities: StaticCommodities
    edges: StaticEdges
    lanes: StaticLanes
    action_slots: ActionSlots
    override_slots: OverrideSlots
    sinks: Sinks
    dyads: StaticDyads
    regime: StaticRegime | None  # null when a runner scores several regimes in one run (hide_regime)


# ----- Obs (§9.2), with graph_now.osat (Q97) and the nullability of the §12 M1 rows ----------------------------------
class NodeKQty(TypedDict):
    node: list[int]
    k: list[int]
    qty: list[float]


class Pipeline(TypedDict):
    edge: list[int]
    k: list[int]
    lane: list[int | None]
    qty: list[float]
    arrival_week: list[int]


class QueueLots(TypedDict):
    lot_id: list[int]
    chokepoint: list[int]
    k: list[int]
    qty: list[float]
    lane: list[int]
    next_edge: list[int]
    arrival_week: list[int]
    dispatch_week: list[int]
    entry_edge: list[int]


class Wip(TypedDict):
    node: list[int]
    k: list[int]
    qty: list[float]
    out_week: list[int]


class EdgeK(TypedDict):
    edge: list[int]
    k: list[int]


class Tariff(TypedDict):
    edge: list[int]
    k: list[int]
    rate: list[float]


class Kappa(TypedDict):
    tb: list[float | None]
    ct: list[float | None]


class Supply(TypedDict):
    node: list[int]
    k: list[int]
    avail: list[float | None]


class FabNow(TypedDict):
    node: list[int]
    R: list[float | None]
    alpha_bar: list[float | None]
    cap_eff: list[float | None]


class GridNow(TypedDict):
    node: list[int]
    G_bar: list[float | None]
    y_bar: list[float | None]


class OsatNow(TypedDict):
    node: list[int]
    R: list[float | None]  # R^osat_i at the instant t - 1 (13), like the fab's R_f (Q97)
    thr_eff: list[float | None]  # thr_i R^osat_i


class GraphNow(TypedDict):
    u: list[float | None]
    c: list[float | None]
    tau: list[int | None]
    prohibited: EdgeK
    tariff: Tariff
    open: list[float | None]
    kappa: Kappa
    war_risk: list[WarRisk | None]
    supply: Supply
    fab: FabNow
    grid: GridNow
    osat: OsatNow


class Clip(TypedDict):
    slot: list[int]
    requested: list[float]
    executed: list[float]


class CostComponents(TypedDict):
    freight: float
    war_risk: float
    tariff: float
    holding: float
    queue_holding: float
    shortage: float
    disposal: float
    shed: float


class SinksWeek(TypedDict):
    node: list[int]
    k: list[int]
    demand: list[float]
    served: list[float]
    lost: list[float]


class Shed(TypedDict):
    node: list[int]
    qty: list[float]


class LastWeek(TypedDict):
    clip: Clip
    cost_components: CostComponents
    sinks: SinksWeek
    shed: Shed


class DemandForecast(TypedDict):
    node: list[int]
    k: list[int]
    h: list[int]
    qty: list[float]


class WarningScores(TypedDict):
    unit_kind: list[UnitKind]
    unit: list[int]
    score: list[float]


class Messages(TypedDict):
    msg_id: list[int]
    channel: list[Channel]
    kind: list[MessageKind]
    region: list[int]
    target_kind: list[TargetKind]
    target: list[int]
    k: list[int | None]
    announced_week: list[int]
    stated_effective_week: list[int | None]


class PendingProhibitions(TypedDict):
    edge: list[int]
    k: list[int]
    effective_week: list[int]


class ClosureEnd(TypedDict):
    chokepoint: list[int]
    end_week: list[int | None]


class Obs(TypedDict):
    week: int
    stock: NodeKQty
    backlog: NodeKQty
    pipeline: Pipeline
    queue_lots: QueueLots
    wip: Wip
    graph_now: GraphNow | None  # null in the final observation and in a blackout week
    slot_mask: list[bool] | None  # a route edge prohibited (Q111); null in the final observation and a blackout week
    last_week: LastWeek | None  # null in the week-1 observation
    demand_forecast: DemandForecast | None
    warning: WarningScores | None
    messages: Messages | None
    pending_prohibitions: PendingProhibitions | None
    closure_end: ClosureEnd | None


# ----- Action and the messages ---------------------------------------------------------------------------------------
class FlowsField(TypedDict):
    slot: list[int]
    # a null qty is a missing value (§9.2; the kit's non-finite quantity), dropped as one invalid entry (§9.3 row 1)
    qty: list[float | None]


class HoldField(TypedDict):
    chokepoint: list[int]
    k: list[int]


class Action(TypedDict):
    week: int
    # absent as in process (§12 "Whole-week failures"): a missing flows is an empty request, a missing overrides or
    # hold none; ``reply_message`` sends the action as the policy returned it (W2)
    flows: NotRequired[FlowsField | None]
    overrides: NotRequired[FlowsField | None]
    hold: NotRequired[HoldField | None]


class OmegaArray(TypedDict):
    dtype: str
    shape: list[int]
    data: Any  # nested lists in C order, NaN as null


class Reset(TypedDict):
    type: Literal["reset"]
    episode: str
    nonce: str
    schema_version: str
    policy_seed: int
    static: Static
    obs: Obs
    remaining_bank: float  # CPU-seconds
    omega: NotRequired[dict[str, OmegaArray]]  # clairvoyant only, local and dev runs (design §12 M3 rows)


class Request(TypedDict):
    type: Literal["step"]
    episode: str
    week: int
    nonce: str
    obs: Obs
    remaining_bank: float


class Reply(TypedDict):
    episode: str
    nonce: str
    # null: the policy gives the week up, a whole-week failure the D9 fallback plays ('action'; §9.3, design §12
    # "Reply line rules (M3 wire)"), as the kit's shim sends when act raises or its action is malformed
    action: Action | None


# ----- line-level functions ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WireLimits:
    """The runner's line limits, SYNTHETIC(placeholder) until Q13.

    The runner's line limits (§9.2 'the maximum reply line length waits on Q13'; §9.6): both
    SYNTHETIC(placeholder) config values until Q13, set at the Hydra boundary, never defaulted here.
    """

    max_reply_bytes: int  # a longer reply line is a whole-week failure ('too_long'); its remainder is discarded
    bank_seconds: float  # the per-episode overage bank sent as remaining_bank (not metered locally, M6)


@dataclass(frozen=True)
class ReplyOutcome:
    """What ``decode_reply`` made of one reply line.

    What ``decode_reply`` made of one line: a reply (its ``action`` whatever it is; the Env validates it), a
    stale line (its action names an earlier week: discarded), or a whole-week failure (``failure`` a code of
    ``WIRE_FAILURES``).
    """

    kind: Literal["reply", "stale", "failure"]
    action: object = None
    failure: str | None = None


_PLAIN = (str, int, float, bool, type(None))  # the JSON scalars, by exact type (design §12 "Wire line encoding")


def _plain(x) -> bool:
    """Whether ``x`` is a tree of exactly dict (str keys), list, str, int, float, bool and None (``_check_plain``).

    The same exact-type tests in the same depth-first order as ``_check_plain``, without building the path of every
    value: ``encode`` runs it on every line and calls ``_check_plain`` only to name a refused value.
    """
    t = type(x)
    if t is dict:
        for k, v in x.items():
            if type(k) is not str or not _plain(v):
                return False
        return True
    if t is list:
        for v in x:
            if not _plain(v):
                return False
        return True
    return t in _PLAIN


def _check_plain(x, path: str = "") -> None:
    """Raise TypeError unless ``x`` is a tree of exactly dict (str keys), list, str, int, float, bool and None.

    Exact types: a NumPy scalar (``np.float64`` is a float subclass), a tuple or a str subclass is refused, so no
    runner line carries a value its decoder would not give back as it was sent. It refuses exactly the trees
    ``_plain`` refuses, naming the first offending value by its path.
    """
    t = type(x)
    if t is dict:
        for k, v in x.items():
            if type(k) is not str:
                raise TypeError(f"{path}: key {k!r} of type {type(k).__name__} is not a str (§9.2: no tuple keys)")
            _check_plain(v, f"{path}.{k}")
    elif t is list:
        for i, v in enumerate(x):
            _check_plain(v, f"{path}[{i}]")
    elif t not in _PLAIN:
        raise TypeError(f"{path}: a {t.__name__} is not a JSON value of the wire (§9.2)")


def encode(msg: Mapping) -> bytes:
    """One ASCII wire line: ``json.dumps(msg, allow_nan=False, separators=(',', ':'), ensure_ascii=True)``, newline.

    Every value is checked to be exactly a JSON type first (``_plain``, then ``_check_plain`` to name a refused one);
    floats go by repr, so each round-trips exactly, ``-0.0`` included (design §12 "Wire line encoding (M3 wire)").

    Raises:
        ValueError: on a NaN or an infinity (a bug: a missing value is null).
        TypeError: on a NumPy scalar or any non-JSON object (a bug upstream).

    """
    if not _plain(msg):
        _check_plain(msg)  # raises the TypeError that names the first offending value
    text = json.dumps(msg, allow_nan=False, separators=(",", ":"), ensure_ascii=True)
    return text.encode("ascii") + b"\n"


def new_token() -> str:
    """A fresh 128-bit token as 32 lowercase hex characters (``secrets.token_hex(16)``; Q86)."""
    return secrets.token_hex(16)


def reset_message(
    *, episode: str, nonce: str, policy_seed: int, static: Static, obs: Obs, remaining_bank: float, omega=None
) -> Reset:
    """The Reset message of §9.2; ``omega`` (clairvoyant only) becomes ``Reset.omega``, absent when None."""
    msg: Reset = {
        "type": "reset",
        "episode": episode,
        "nonce": nonce,
        "schema_version": SCHEMA_VERSION,
        "policy_seed": policy_seed,
        "static": static,
        "obs": obs,
        "remaining_bank": remaining_bank,
    }
    if omega is not None:
        msg["omega"] = omega
    return msg


def request_message(*, episode: str, week: int, nonce: str, obs: Obs, remaining_bank: float) -> Request:
    """The Request message of §9.2 for week ``week``."""
    return {
        "type": "step",
        "episode": episode,
        "week": week,
        "nonce": nonce,
        "obs": obs,
        "remaining_bank": remaining_bank,
    }


def reply_message(*, episode: str, nonce: str, action: Action | None) -> Reply:
    """The Reply message of §9.2 (the policy side, ``runner.serve``): ``action`` passed through unchanged.

    No field is added or filled (a missing ``overrides`` stays missing), so the stored action equals the in-process
    one and W2's hash holds.
    """
    return {"episode": episode, "nonce": nonce, "action": action}


# The reply line's own limits (design §12 "Reply line rules (M3 wire)"), fixed so that a line's verdict is the line's:
# the parser's recursion limit falls as the caller's C-call depth grows, and ``int_max_str_digits`` is whatever the
# process set (M3 attempt 1's gate, DET-M3-1). Not model values.
MAX_REPLY_DEPTH = 32  # arrays and objects nested, the reply object depth 1; the policy side writes at most 17 levels
MAX_INT_DIGITS = 640  # digits of one JSON integer, sign aside: CPython's smallest int_max_str_digits, so int() takes it
_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"?', re.DOTALL)  # a string literal; an unterminated one to the end
_NOT_BRACKET = re.compile(r"[^\[\]{}]+")
_BRACKET_STEP = bytes.maketrans(b"[{]}", b"\x01\x01\xff\xff")  # +1 and -1 as signed bytes


def _depth(text: str) -> int:
    """The deepest nesting of arrays and objects in ``text``, strings aside (``{}`` is 1); linear, never recursive.

    Exact on valid JSON, whose strings the pattern cuts out as its parser reads them (a backslash escapes the next
    character), so only structural brackets count; on other text it is some number and the parser refuses the text.
    """
    brackets = _NOT_BRACKET.sub("", _STRING.sub("", text)).encode("ascii").translate(_BRACKET_STEP)
    return max(itertools.accumulate(array.array("b", brackets)), default=0)


def _bounded_int(digits: str) -> int:
    """A JSON integer of at most ``MAX_INT_DIGITS`` digits (the parser's ``parse_int``), else ValueError."""
    if len(digits) - digits.startswith("-") > MAX_INT_DIGITS:
        raise ValueError(f"an integer of more than {MAX_INT_DIGITS} digits")
    return int(digits)


def decode_reply(line: bytes, *, episode: str, nonce: str, week: int, limits: WireLimits) -> ReplyOutcome:
    """Read one reply line of week ``week`` by the §9.3 line rules; never raises on a hostile line.

    Read one reply line, in this order (§9.3 row 2; Q71 X5): longer than ``limits.max_reply_bytes`` -> failure
    'too_long'; not UTF-8, nested deeper than ``MAX_REPLY_DEPTH`` (counted before parsing), not JSON (NaN and Infinity
    tokens accepted; a duplicate key refused), an integer of more than ``MAX_INT_DIGITS`` digits, or not an object ->
    'unparsable'; ``episode`` or ``nonce`` missing or not the episode's -> 'tags'; an action that is a dict whose
    ``week`` is an int (not a bool) below ``week`` -> stale; else a reply carrying ``reply.get('action')`` (a missing,
    wrong or later week is the Env's whole-week failure). The length is counted without the line's final newline;
    keys beside ``episode``, ``nonce`` and ``action`` are ignored (design §12 "Reply line rules (M3 wire)").

    Raises:
        RecursionError: only when a line within the limits meets the caller's exhausted stack (fewer than
            ``MAX_REPLY_DEPTH`` frames left): the caller's state, never a verdict on the line.

    """
    body = line[:-1] if line.endswith(b"\n") else line
    if len(body) > limits.max_reply_bytes:
        return ReplyOutcome("failure", failure="too_long")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return ReplyOutcome("failure", failure="unparsable")
    if _depth(text) > MAX_REPLY_DEPTH:
        return ReplyOutcome("failure", failure="unparsable")
    try:
        reply = _DECODER.decode(text)
    except RecursionError:
        raise
    except Exception:  # noqa: BLE001 - not JSON, a duplicate key, an integer beyond MAX_INT_DIGITS
        return ReplyOutcome("failure", failure="unparsable")
    if type(reply) is not dict:
        return ReplyOutcome("failure", failure="unparsable")
    if reply.get("episode") != episode or reply.get("nonce") != nonce:
        return ReplyOutcome("failure", failure="tags")
    action = reply.get("action")
    if type(action) is dict and type(action.get("week")) is int and action["week"] < week:
        return ReplyOutcome("stale", action=action)
    return ReplyOutcome("reply", action=action)


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """The JSON object of ``pairs``, refusing a repeated key (a duplicate key is unparsable, design §12)."""
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


_DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicates, parse_int=_bounded_int)


OMEGA_WITHHELD = ("meta_episode", "meta_split")  # trusted side only (§4.1): never in the clairvoyant payload


def _nan_as_null(x):
    """``tolist`` output with every float NaN as None, recursively (the wire has no NaN, §9.2)."""
    if isinstance(x, list):
        return [_nan_as_null(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def public_omega(omega: Omega) -> Omega:
    """Omega without the arrays the clairvoyant payload withholds (``OMEGA_WITHHELD``), as an immutable ``Omega``.

    What ``omega_payload`` reads, so ``omega_payload(public_omega(omega)) == omega_payload(omega)``. The clairvoyant
    view keeps it in place of the payload (``view.InformationView.omega_public``): each hand-out builds a fresh payload
    from read-only arrays, and a snapshot pickles arrays, never nested lists.
    """
    return Omega({name: a for name, a in omega.arrays.items() if name not in OMEGA_WITHHELD})


def omega_payload(omega: Omega) -> dict[str, OmegaArray]:
    """The clairvoyant 'full omega at reset' as JSON-able arrays (§5.1).

    The clairvoyant 'full omega at reset' (§5.1): every omega array but ``meta_episode`` and ``meta_split``
    (trusted side only, §4.1) as {dtype, shape, data} with NaN as null, in sorted name order; it passes ``encode``.
    ``dtype`` is the name ``container.ARRAY_DTYPES`` uses (``float64``, ``int8``, ``<U64``), ``shape`` a list of ints
    and ``data`` the nested lists of ``ndarray.tolist`` in C order (a 0-d array its scalar), Python ints, floats and
    strs only.
    """
    out: dict[str, OmegaArray] = {}
    for name in sorted(omega.arrays):
        if name in OMEGA_WITHHELD:
            continue
        a = omega.arrays[name]
        dtype = a.dtype.str if a.dtype.kind == "U" else a.dtype.name
        data = a.tolist()
        if a.dtype.kind == "f" and np.isnan(a).any():  # only a float array holds a NaN to write as null
            data = _nan_as_null(data)
        out[name] = {"dtype": dtype, "shape": [int(n) for n in a.shape], "data": data}
    return out

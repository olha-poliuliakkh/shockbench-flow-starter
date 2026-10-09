"""Simulator state, the per-week record, the trajectory and integer cents (design §3.2-3.6, §9.1; Q68, V1, B2).

The ``StepRecord`` holds every realised quantity of one week, so the replay test (53) can feed a trajectory into the
rows of the LP (51) and the cost can be recomputed. Integer codes: stock slots S, action slots, override slots, and the
ordinals of chokepoints, grids G, fabs F, OSATs and demands D of the instance.
"""

import math
from dataclasses import dataclass, field
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

import numpy as np

from sbfplan.instance.io import sha256_hex


# (24) runs under this private context, never the ambient one, so no decimal setting of the caller (precision, rounding,
# traps, or decimal.DefaultContext, from which Context() copies any field left out) changes a cent (DET-1). A float
# repr has at most 17 significant digits and the largest finite float 309 integer digits (311 after the * 100), so 400
# digits hold every product and every quantized result exactly.
_CENTS_CONTEXT = Context(
    prec=400,
    rounding=ROUND_HALF_EVEN,
    Emin=MIN_EMIN,
    Emax=MAX_EMAX,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_ONE = Decimal(1)


def cents(x: float) -> int:
    """(24): round_half_even(100 * Decimal(repr(x))) — the one conversion from USD floats to integer cents.

    Every step uses the methods of the module's own context, so the result is independent of
    ``decimal.getcontext()``. A non-finite ``x`` has no cents: ``decimal.InvalidOperation`` (inf) or ``ValueError``
    (nan).
    """
    ctx = _CENTS_CONTEXT
    return int(ctx.quantize(ctx.multiply(ctx.create_decimal(repr(float(x))), 100), _ONE))


COST_COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


@dataclass(frozen=True)
class CostComponents:
    """The terms of C_t in (23), USD, each a ``math.fsum`` over explicit terms (no ``@`` or ``np.dot``; §3.6)."""

    freight: float = 0.0
    war_risk: float = 0.0
    tariff: float = 0.0
    holding: float = 0.0
    queue_holding: float = 0.0
    shortage: float = 0.0
    disposal: float = 0.0
    shed: float = 0.0

    def total(self) -> float:
        """C_t = fsum of the components."""
        return math.fsum(getattr(self, k) for k in COST_COMPONENTS)

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in COST_COMPONENTS}


@dataclass
class Shipment:
    """A shipment in the pipeline: dispatched on ``edge`` in ``dispatch_week``, arriving in ``arrival_week`` (2)."""

    edge: int
    k: int
    lane: int | None  # the lane declared at dispatch (kept after a chokepoint release); None off lanes
    qty: float
    dispatch_week: int
    arrival_week: int
    lot_id: int | None = None  # set at dispatch for shipments into a chokepoint (§3.4 step 1), in action-slot order


@dataclass
class Lot:
    """A queue lot at a chokepoint (§3.4 pseudocode, step 1)."""

    lot_id: int
    chokepoint: int  # node index
    k: int
    qty: float
    lane: int
    next_edge: int
    dispatch_week: int
    entry_edge: int
    arrival_week: int


@dataclass
class State:
    """Mutable simulator state at the end of week ``week`` (0 at reset); the next step simulates week ``week + 1``."""

    week: int
    stock: np.ndarray  # (S,) I^t per stock slot; at a chokepoint slot the queue total, the sum of its lots
    pipeline: list[Shipment]
    lots: list[Lot]
    fab_wip: dict[int, dict[int, float]]  # fab ordinal -> start week -> gross remaining lots (mature at start + tau)
    osat_wip: dict[int, dict[int, dict[int, float]]]  # osat ordinal -> out week -> packaged k -> qty
    backlog: np.ndarray  # (D,) B^t per demand
    next_lot_id: int = 0
    last: "StepRecord | None" = None  # the record of week ``week`` (feeds ``last_week`` of the observation)


_WHEN_LOGGED = ("edge_clamps", "override_clamps")  # StepRecord fields in as_json only when not empty (Q95)


@dataclass(frozen=True)
class StepRecord:
    """Everything realised in one week (§3.2 steps 3-9), in the instance's integer codes."""

    week: int
    requested: dict[int, float]  # action slot -> requested qty after the validity rules (§9.3)
    executed: dict[int, float]  # action slot -> executed dispatch after the clip (3)-(7)
    override_requested: dict[int, float]  # override slot -> requested
    override_executed: dict[int, float]  # override slot -> released
    x: dict[tuple[int, int, int | None], float]  # (edge, k, lane) -> x^t_ekl: dispatches and chokepoint releases
    stock: np.ndarray  # (S,) end-of-week I^t (chokepoint slots: queue totals)
    queue: dict[tuple[int, int, int], float]  # (chokepoint node, k, lane) -> end-of-week queue content, (52)
    disposal: np.ndarray  # (S,) O^t
    lift: np.ndarray  # (S,) varsigma^t, 0 at non-supply slots (Q79)
    lots_started: np.ndarray  # (F,) p^t_f
    scrapped: np.ndarray  # (F,) gross WIP scrapped in this onset week (14), booked by the simulator (Q44, Q69)
    packaged: dict[tuple[int, int], float]  # (osat ordinal, packaged k) -> xi^t_ik (19)
    segment: dict[tuple[int, int | None], float]  # (grid ordinal, fuel k or None for ∅) -> G^t_gk (15), (18)
    energy: np.ndarray  # (F,) E^t_f; 0 at fabs without a grid
    served_load: np.ndarray  # (G,) y^t_g
    shed: np.ndarray  # (G,) y^sh,t_g
    demand: np.ndarray  # (D,) d^t
    served: np.ndarray  # (D,) D^t
    lost: np.ndarray  # (D,) U^t (0 at backlog sinks)
    backlog: np.ndarray  # (D,) B^t (0 at lost-sales sinks)
    costs: CostComponents
    cost_cents: int  # C^¢_t (24)
    # (stock slot, value), in step order: a stock that a dispatch, a grid burn or packaging left in [-b, 0) and that was
    # set to 0, with b = 1e-12 max(A-bar, |q|, draw, 2^-1022) (Q58 M11; design §12 "Stock clamps and dust lots"), value
    # < 0; and a queue lot with 0 < qty <= 1e-12 that left the book, at its (chokepoint, k) slot, value its qty > 0
    clamps: tuple[tuple[int, float], ...] = ()
    invalid: tuple[str, ...] = ()  # entries dropped by the validity rules, logged (§9.3)
    # (action slot, value), in slot order: an executed request on an edge with a subnormal factor (4) that the edge
    # clamp lowered to the edge's residual u'_e, value < 0 the residual it would have left, within the band of
    # ``clamps`` with q the request (Q95; ``clip.edge_clamp``)
    edge_clamps: tuple[tuple[int, float], ...] = ()
    # (override slot, value), per chokepoint in slot order: an override on an out-edge with a subnormal factor (4) that
    # the edge clamp lowered to the out-edge's residual u'_e before (6), value < 0 the residual it would have left,
    # within the band of ``edge_clamps`` with q the override request (Q95; ``chokepoint._overrides``)
    override_clamps: tuple[tuple[int, float], ...] = ()

    def as_json(self) -> dict:
        """JSON-able form with sorted keys and floats by repr (for the trajectory hash).

        ``edge_clamps`` and ``override_clamps`` enter only when a clamp was logged in them, so a record without one
        hashes as before the fields existed (Q95).
        """

        def enc(v):
            if isinstance(v, np.ndarray):
                return v.tolist()  # float64 arrays: Python floats, by repr in the JSON
            if isinstance(v, dict):
                return [[list(k) if isinstance(k, tuple) else k, float(val)] for k, val in sorted(v.items(), key=_key)]
            if isinstance(v, CostComponents):
                return v.as_dict()
            if isinstance(v, tuple):
                return [list(x) if isinstance(x, tuple) else x for x in v]
            return v

        return {
            name: enc(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in _WHEN_LOGGED or getattr(self, name)
        }


def _key(item):
    k = item[0]
    return tuple(-1 if x is None else x for x in k) if isinstance(k, tuple) else (k,)


@dataclass
class Trajectory:
    """One episode: actions as received, the weekly records and the terminal credit (23)-(25)."""

    instance_hash: str
    omega_hash: str
    policy: str
    regime: str
    actions: list[dict] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    salvage: float | None = None  # S_T in USD at T (23)
    salvage_cents: int | None = None  # S^¢_T (24)
    instance_digest: str = ""  # content digest of the instance simulated (V3)
    marks_digest: str = ""  # content digest of the marks simulated (V3)
    # "week t (ExceptionType)" once a step raised: ``actions`` then ends with week t's action, and ``records`` (and the
    # terminal credit) hold the completed weeks only, wherever in the step it failed (§12 'Failed steps'; DET-P3-5)
    failed: str | None = None
    # (week, code of information.wire.WIRE_FAILURES) of every week the runner stepped with a whole-week wire failure
    # (``Env.step(..., wire_failure=code)``, M3): the runner's cause, which no reply value can forge; hashed only when
    # set, so an in-process trajectory hashes as before the field existed, and replay passes it back (design §12 M3)
    wire_failures: list[tuple[int, str]] = field(default_factory=list)

    @property
    def J_cents(self) -> int:
        """J^¢ = sum_t C^¢_t - S^¢_T (24); requires a finished episode."""
        if self.salvage_cents is None:
            raise ValueError("episode not finished")
        return sum(r.cost_cents for r in self.records) - self.salvage_cents

    def rewards_cents(self) -> list[int]:
        """r_t of (25): -C^¢_t, and -C^¢_T + S^¢_T at T; they sum to -J^¢ exactly."""
        r = [-rec.cost_cents for rec in self.records]
        if self.salvage_cents is not None and r:
            r[-1] += self.salvage_cents
        return r

    def sha256(self) -> str:
        """SHA-256 of the canonical JSON (§2.3) of the actions, records and terminal credit (V1, B2).

        A failed trajectory adds its ``failed`` mark, and one with wire failures its ``wire_failures``; a trajectory
        without them hashes as before they existed.
        """
        doc = {
            "instance_hash": self.instance_hash,
            "omega_hash": self.omega_hash,
            "policy": self.policy,
            "regime": self.regime,
            "actions": self.actions,
            "records": [r.as_json() for r in self.records],
            "salvage": self.salvage,
            "salvage_cents": self.salvage_cents,
            "instance_digest": self.instance_digest,
            "marks_digest": self.marks_digest,
        }
        if self.failed is not None:
            doc["failed"] = self.failed
        if self.wire_failures:
            doc["wire_failures"] = [[int(w), str(code)] for w, code in self.wire_failures]
        return sha256_hex(doc)

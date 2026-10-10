"""The warning scores (45) and their wire field (design §5.2, §9.2; Q46, Q51, Q80).

S^t_theta,x = a_theta,kind(x) X^{t-L}_x + sqrt(1 - a^2) W^{t-L}_x (45), for every signal unit x in omega's X row order
(all R regions, the inactive ones too, then the D dyads, then the chokepoints by ordinal; ``regime.signal_units``) and
t = 1..T. Column t + B_burn of X and W is the state in force during week t, so the week-t score reads column
t - L + B_burn: at L = 0 it is X^t, week t's hazard (29), which counts as timestamped at the instant t - 1 (a hazard,
not a realisation, §5.2; design §3.1 reading). The float order is (a X) + (s W) with s = sqrt(1 - a^2) formed once per
kind, so a = 1 gives S = X bit for bit. The same X and W serve every regime (paired at the path level, R5.11); no
random draw happens here.

Wire ids (§9.2 ``warning.unit``): a region's index, a dyad's row of ``Static.dyads`` (omega's ``dyad_regions``), a
chokepoint's node index (as ``queue_lots.chokepoint`` and ``hold``).
"""

from dataclasses import dataclass

import numpy as np

from shockbench_flow.information.theta import Theta
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega, immutable_copy


@dataclass(frozen=True)
class Unit:
    """One signal unit: its kind (``codes.UNIT_KINDS``), its wire id, and its row in omega's X and W."""

    kind: str
    unit: int  # region index, dyad row, or chokepoint node index
    row: int  # row of X and W


def unit_table(inst: Instance, omega: Omega) -> tuple[Unit, ...]:
    """The signal units in X row order: regions, then the dyads of omega's ``dyad_regions``, then the chokepoints.

    One row order, ``disruption.regime.signal_units``'s (the one rule, which the generator writes X and W by): all R
    regions, the D dyads in ``dyad_regions`` row order, the chokepoints in ``inst.chokepoints`` order by node index;
    the table asserts that X and W have exactly R + D + C rows.

    Raises:
        ValueError: if omega has no latent paths (an injected omega), no ``dyad_regions``, or its X or W rows are not
            R + D + C.

    """
    if not omega.generated:
        raise ValueError("the warning (45) needs omega's latent paths X and W: an injected omega has none")
    if "dyad_regions" not in omega:
        raise ValueError("the warning (45) needs omega's dyad_regions to name its dyad units (Static.dyads)")
    R, D, C = len(inst.regions), int(np.asarray(omega["dyad_regions"]).shape[0]), len(inst.chokepoints)
    X, W, z_dyad = np.asarray(omega["X"]), np.asarray(omega["W"]), np.asarray(omega["z_dyad"])
    if X.shape[0] != R + D + C or W.shape != X.shape or z_dyad.shape[0] != D:
        raise ValueError(
            f"omega's X {X.shape}, W {W.shape} and z_dyad {z_dyad.shape} must hold the R + D + C = {R + D + C} signal"
            f" units and the {D} dyads of dyad_regions (regime.signal_units)"
        )
    region, dyad, chokepoint = codes.UNIT_KINDS
    units = [Unit(region, m, m) for m in range(R)]
    units += [Unit(dyad, d, R + d) for d in range(D)]
    units += [Unit(chokepoint, int(c), R + D + ci) for ci, c in enumerate(inst.chokepoints)]
    return tuple(units)


def warning_scores(X: np.ndarray, W: np.ndarray, a: np.ndarray, L: int, burn_in: int, T: int) -> np.ndarray:
    """(T, U) read-only S of (45): row t - 1 is a * X[:, t - L + B] + sqrt(1 - a * a) * W[:, t - L + B], t = 1..T.

    ``a`` is (U,), per unit from theta's a_<kind>; the row order matches the forecast's (T, D, 8) and WeeklyMarks'
    (T, ...) conventions.

    Raises:
        ValueError: on L < 0, a column t - L + B < 0, |a| > 1, or X and W of other shapes.

    """
    X, W, a = np.asarray(X, dtype=np.float64), np.asarray(W, dtype=np.float64), np.asarray(a, dtype=np.float64)
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)) or L < 0:
        raise ValueError(f"(45) needs a lag L >= 0, got {L!r}")
    if X.ndim != 2 or W.shape != X.shape or a.shape != (X.shape[0],):
        raise ValueError(f"(45) needs X, W of one (U, weeks) shape and a of (U,): {X.shape}, {W.shape}, {a.shape}")
    if not bool(np.all(np.abs(a) <= 1.0)):  # NaN fails too
        raise ValueError("(45) needs |a| <= 1 for every unit")
    cols = np.arange(1, T + 1) - int(L) + int(burn_in)  # week t reads column t - L + B_burn
    if T and (cols[0] < 0 or cols[-1] >= X.shape[1]):
        raise ValueError(f"(45): the columns t - L + B of weeks 1..{T} lie outside X's {X.shape[1]} columns")
    s = np.sqrt(1.0 - a * a)  # once per unit (its kind's a), so a = 1 gives S = X bit for bit
    S = a[:, None] * X[:, cols] + s[:, None] * W[:, cols]  # (a X) + (s W), the fixed float order
    return immutable_copy(np.ascontiguousarray(S.T))


def unit_separation(theta: Theta, units: tuple[Unit, ...]) -> np.ndarray:
    """(U,) the a of each unit by its kind (theta's ``a_region``, ``a_dyad``, ``a_chokepoint``).

    Raises:
        ValueError: if theta has no warning feed (L None) or no a for a kind the units hold.

    """
    if theta.L is None:
        raise ValueError(f"theta {theta.name!r} has no warning feed (L None)")
    a = [getattr(theta, f"a_{u.kind}", None) for u in units]
    if any(x is None for x in a):
        raise ValueError(f"theta {theta.name!r} gives no a for a unit kind of the signal units")
    return np.array(a, dtype=np.float64)


def warning_field(units: tuple[Unit, ...], scores: np.ndarray, t: int) -> dict | None:
    """The §9.2 ``warning`` value of week t.

    Columns {unit_kind: [...], unit: [...], score: [...]} in unit order, the scores as Python floats; None after T
    (the final observation) (1).
    """
    if not 1 <= t <= scores.shape[0]:
        return None
    return {"unit_kind": [u.kind for u in units], "unit": [u.unit for u in units], "score": scores[t - 1].tolist()}

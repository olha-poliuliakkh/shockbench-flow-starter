"""Onset labels of the warning signal's units, for V27 and the per-episode eta_n (design §5.2 "Labels"; Q80).

Readings of design §12 (M3 rows), in the alignment of ``scripts/python/evidence/f3_mc.py``:

- Region and dyad units (``chain_labels``): a chain's onset week is s with z^{s-1} = none and z^s != none (the step
  from column s - 1 to s reads the hazard of week s - 1, (29)). Score week t (column t + B) is a none-week iff
  z[t + B] = none, and label_h(t) = 1 iff an onset week lies in (t, t + h]. A region unit reads its own chain
  (omega's ``z_c_own``), never the effective layer: an onset a dyad war forces belongs to the dyad unit. A dyad unit
  reads its ``z_dyad`` row (0 = at peace).
- Chokepoint units (``chokepoint_labels``): week t is a none-week iff no militarised closure (type 3) on c acts in
  [t - 1, t) (f^t_q = 0 of (37)); weather closures (type 7) exclude no week; label_h(t) = 1 iff a militarised closure
  on c has its onset week floor(t_on) + 1 in (t, t + h]. It takes an event list and a week range, since V27's
  chokepoint worlds are long-horizon, not omega's T weeks.
- A week whose horizon t + h runs past the span is invalid (censored). h = 13 is declared (46); h in {1, 4, 13, 52}
  is reported (V27 (iii)).

eta_n (the per-episode Brier score of §5.4) is logged by the evaluation runner (M4/M6), from these labels and
``z_c_own``; M3 supplies both (design §12 M3 row).
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata

from shockbench_flow.marks import K_CHOKEPOINT, MILITARISED_CLOSURE, Event
from shockbench_flow.omega import codes


DECLARED_HORIZON = 13  # AUROC_13 of (46)
REPORTED_HORIZONS = (1, 4, 13, 52)  # V27 (iii)


def _check_h(h: int) -> int:
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError(f"a label horizon h must be an integer >= 1 week, got {h!r}")
    return int(h)


def _in_window(onset_at: np.ndarray, h: int) -> np.ndarray:
    """label[..., c] = an onset at a position in (c, c + h] along the last axis (positions past the end count none)."""
    n = onset_at.shape[-1]
    cs = np.cumsum(onset_at, axis=-1, dtype=np.int64)  # onsets at positions <= c
    ahead = np.minimum(np.arange(n) + h, n - 1)
    return (cs[..., ahead] - cs) > 0


def chain_labels(z: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(label, none, valid), each (N, ncol) bool, for chains ``z`` (N, ncol) (z_c_own rows or z_dyad rows).

    Column c is a none-week iff z[c] = none; an onset lies at column s iff z[s - 1] = none and z[s] != none;
    label_h[c] iff an onset lies in (c, c + h]; valid[c] iff c + h <= ncol - 1 (the horizon inside the span). A score
    week t reads column t + B_burn (module docstring); AUROC_h of (46) is over ``none & valid``.

    Raises:
        ValueError: if h < 1 or ``z`` is not 2-d with codes of ``codes.CONFLICT_STATES``.

    """
    h = _check_h(h)
    z = np.asarray(z)
    if z.ndim != 2 or z.dtype.kind not in "iu":
        raise ValueError(f"chains must be a 2-d integer array (units, columns), got {z.dtype} {z.shape}")
    if z.size and (z.min() < 0 or z.max() >= len(codes.CONFLICT_STATES)):
        raise ValueError(f"chains hold a code outside the conflict states {codes.CONFLICT_STATES}")
    none = z == 0
    onset = np.zeros(z.shape, dtype=bool)
    onset[:, 1:] = none[:, :-1] & ~none[:, 1:]
    valid = np.broadcast_to(np.arange(z.shape[1]) + h <= z.shape[1] - 1, z.shape).copy()
    return _in_window(onset, h), none, valid


def chokepoint_labels(
    events: Sequence[Event], c: int, weeks: np.ndarray, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(label, none, valid), each (len(weeks),) bool, for the chokepoint node ``c`` over the score weeks ``weeks``.

    Only militarised closures on c (type ``militarised_closure``, target kind chokepoint, target c) count: week t is a
    none-week iff none of them acts in [t - 1, t) (f^t_q > 0 of (37)); label_h(t) iff one has its onset week
    floor(t_on) + 1 in (t, t + h]; valid(t) iff t + h <= the last week of ``weeks`` (the span's end). Weather closures
    and every other event are ignored (module docstring).

    Raises:
        ValueError: if h < 1 or ``weeks`` is not strictly increasing.

    """
    h = _check_h(h)
    w = np.asarray(weeks)
    if w.ndim != 1 or w.dtype.kind not in "iu" or (w.size > 1 and not np.all(np.diff(w) > 0)):
        raise ValueError("chokepoint label weeks must be a strictly increasing 1-d integer array")
    t = w.astype(np.float64)
    none = np.ones(w.shape, dtype=bool)
    label = np.zeros(w.shape, dtype=bool)
    for q in events:
        if q.type != MILITARISED_CLOSURE or q.target_kind != K_CHOKEPOINT or q.target != c:
            continue
        acts = np.maximum(0.0, np.minimum(t, q.onset + q.duration) - np.maximum(t - 1.0, q.onset)) > 0.0  # (37)
        none &= ~acts
        s = math.floor(q.onset) + 1  # the onset week w of (1): t_on in [s - 1, s)
        label |= (w < s) & (s <= w + h)
    valid = w + h <= (w[-1] if w.size else 0)
    return label, none, valid


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """The rank AUROC of a score against binary labels.

    The rank AUROC of ``score`` against the labels ``y`` (ties at half weight; ``scipy.stats.rankdata``, as
    ``f3_mc.py`` computes it), for (46) and V27.

    Raises:
        ValueError: if ``y`` holds no positive or no negative.

    """
    s = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"score {s.shape} and labels {y.shape} differ in length")
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        raise ValueError(f"AUROC needs positives and negatives, got {n1} and {n0}")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

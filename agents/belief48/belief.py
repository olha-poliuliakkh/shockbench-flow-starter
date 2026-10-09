"""The strait belief: the expected open fraction E[o_{c,t+h}] of each strait for the window's weeks h = 1..H-1.

Calibrated by scripts/research/belief_calibrate.py on a private training root (never the dev split); the constants are
belief_params.json beside this file. Three parts, each a switch of ``expected_open``:

- recovery: a strait closed for ``a`` weeks is still closed h weeks later with probability S(a + h) / S(a), S the
  closure-duration mixture P(L >= n) = pi0 (1 - q)^(n - 1) + (1 - pi0) (1 - q2)^(n - 1); while closed it keeps
  today's open fraction, and reopens fully;
- threats: an open strait targeted by a live mid_threat thread of age a closes by week t + j with the fitted
  probability onset[bucket(a)][j - 1], then stays closed as S says, at the mean depth while closed;
- base: the same for an open strait with no such thread, at the base rate onset_none.

Row 0 of the window (this week) is never touched: the week's action plans on the strait as observed.
Imports: the standard library and numpy.
"""

import json
from pathlib import Path

import numpy as np


P = json.loads((Path(__file__).resolve().parent / "belief_params.json").read_text())
PI0, Q, Q2, DEPTH = float(P["pi0"]), float(P["q"]), float(P["q2"]), float(P["depth"])
AGES = tuple(int(a) for a in P["age_buckets"])
ONSET = np.asarray(P["onset"], dtype=float)  # (age bucket, j - 1): P(onset by week t + j | live thread of that age)
ONSET_NONE = np.asarray(P["onset_none"], dtype=float)


def survival(n) -> np.ndarray:
    """P(L >= n), L the length in weeks of a closure."""
    n = np.asarray(n, dtype=float)
    return PI0 * (1 - Q) ** (n - 1) + (1 - PI0) * (1 - Q2) ** (n - 1)


def _cdf(curve: np.ndarray, H: int) -> np.ndarray:
    """An onset CDF over j = 1..H, its last value carried past the fitted weeks."""
    return np.concatenate([curve, np.full(max(0, H - len(curve)), curve[-1])])[:H]


def threat_ages(s) -> dict[int, int]:
    """{strait position: age in weeks of the youngest live mid_threat thread that targets the strait}."""
    out = {}
    for m in s.threads:
        if m["channel"] != "mid_threat" or m["target_kind"] != "chokepoint" or m["target"] is None:
            continue
        pos = s.strait_pos(int(m["target"]))
        if pos is None or m["announced_week"] is None:
            continue
        out[pos] = min(out.get(pos, 10**9), max(0, s.week - int(m["announced_week"])))
    return out


def expected_open(o_now: float, closed_for: int, age: int | None, H: int, recovery=True, threats=True, base=False):
    """E[o] of one strait for the window's rows 1..H-1 (weeks t + 1 ..), or None where the persistence row stands."""
    h = np.arange(1, H)
    if o_now < 1.0:
        if not recovery:
            return None
        a = max(1, int(closed_for))
        still = survival(a + h) / survival(a)
        return still * o_now + (1.0 - still)
    if age is not None and threats:
        cdf = _cdf(ONSET[max(i for i, lo in enumerate(AGES) if age >= lo)], H - 1)
    elif base:
        cdf = _cdf(ONSET_NONE, H - 1)
    else:
        return None
    p_on = np.diff(np.concatenate([[0.0], cdf]))  # P(onset at week t + i), i = 1..H-1
    closed = np.array([float(np.dot(p_on[:k], survival(k - np.arange(k)))) for k in h])
    return 1.0 - closed * (1.0 - DEPTH)


def apply(w, s, recovery=True, threats=True, base=False) -> int:
    """Write each strait's E[o] into rows 1..H-1 of the window ``w``; the number of straits changed."""
    ages = threat_ages(s)
    changed = 0
    for pos in range(len(s.open_now)):
        e = expected_open(float(s.open_now[pos]), int(s.closed_for[pos]), ages.get(pos), w.H, recovery, threats, base)
        if e is None:
            continue
        for h, v in enumerate(e, start=1):
            w.set_open(pos, float(v), start=h, end=h + 1)
        changed += 1
    return changed

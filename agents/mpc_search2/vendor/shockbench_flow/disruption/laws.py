"""Distribution laws drawn by inversion of keyed uniforms (design §4.1).

"Every parameter that varies across rungs or regimes is drawn by inversion of a keyed uniform", so the same uniform
moves a draw monotonically.

A ``Law`` is a frozen (kind, params) pair; ``ppf(u)`` maps uniforms in (0, 1) to values. Durations are returned in
weeks (days divided by 7, design §4.4: "Durations sampled in days convert as T_q/7 weeks"), and so are the
announcement leads of the kind ``atom_or_knots_days`` (M3, (49)), which may be negative.

``mean()`` and ``tail_integral(x)`` = int_x^inf S(y) dy give a duration law's stationary active stock and the part of
it a burn-in of x weeks misses (Q97's 1 % rule, design §12 M5 row "Burn-in of `small` and `full`"), in closed form per
component kind.
"""

import bisect
import functools
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammaincc, ndtr, ndtri


LAW_KINDS = (
    "constant",
    "uniform",
    "lognormal_days",
    "lognormal_weeks",
    "one_day_or_lognormal_days",
    "persistent_mix",
    "atom_or_knots_days",
    "weibull_days",
)
_KNOTS = "atom_or_knots_days"
_WEIBULL = "weibull_days"


Part = tuple[tuple[float, ...], tuple[float, ...]]  # one part of an atom_or_knots_days law: (knot p's, knot days)


@functools.lru_cache(maxsize=256)
def _knot_parts(params: tuple[float, ...]) -> tuple[float, Part, Part]:
    """(p_le0, the <= 0 part, the > 0 part) of an ``atom_or_knots_days`` law, checked; parsed once per params.

    ``params`` = (p_le0, m, p_1, d_1, ..., p_m, d_m, q_1, e_1, ..., q_k, e_k): the m knots (p, days) of the <= 0 part,
    then the k knots of the > 0 part. A part of positive mass has at least two knots, its p increasing strictly from
    0 to 1 and its days non-decreasing, <= 0 in the first part and > 0 in the second; a part of mass 0 has none (an
    empty part is ((), ())).

    Raises:
        ValueError: naming the kind, on any other layout.

    """
    if len(params) < 2 or not all(math.isfinite(x) for x in params):
        raise ValueError(f"{_KNOTS} needs (p_le0, m, knots...) of finite numbers, got {params!r}")
    p_le0, m = float(params[0]), params[1]
    if not 0.0 <= p_le0 <= 1.0 or m != int(m) or m < 0 or 2 + 2 * int(m) > len(params) or len(params) % 2:
        raise ValueError(f"{_KNOTS}: p_le0 in [0, 1] and a knot count m that fits the parameters, got {params[:2]!r}")
    m = int(m)
    flat = [float(x) for x in params[2:]]
    parts = []
    for name, lo, hi, mass, ok in (
        ("<= 0", 0, 2 * m, p_le0, lambda d: d <= 0.0),
        ("> 0", 2 * m, len(flat), 1.0 - p_le0, lambda d: d > 0.0),
    ):
        ps, ds = tuple(flat[lo:hi:2]), tuple(flat[lo + 1 : hi : 2])
        if mass == 0.0 and not ps:
            parts.append(((), ()))
            continue
        if (
            mass == 0.0
            or len(ps) < 2
            or ps[0] != 0.0
            or ps[-1] != 1.0
            or any(b <= a for a, b in zip(ps, ps[1:]))
            or any(b < a for a, b in zip(ds, ds[1:]))
            or not all(ok(d) for d in ds)
        ):
            raise ValueError(
                f"{_KNOTS}: the {name} part of mass {mass} needs knots (p, days) with p rising from 0 to 1 and days"
                f" non-decreasing and {name}, got {tuple(zip(ps, ds))!r}"
            )
        parts.append((ps, ds))
    return p_le0, parts[0], parts[1]


def _interp(v: float, part: Part) -> float:
    """The linear interpolant of a part's knots at v in [0, 1] (np.interp's formula), clamped to its segment's values.

    The clamp makes the draw monotone in v to the last bit: within a segment the rounded value is monotone in v, and
    the segments' value ranges follow each other.
    """
    ps, xs = part
    j = min(max(bisect.bisect_right(ps, v) - 1, 0), len(ps) - 2)
    if v >= ps[-1]:
        return xs[-1]
    x0, x1 = xs[j], xs[j + 1]
    return min(max((x1 - x0) / (ps[j + 1] - ps[j]) * (v - ps[j]) + x0, x0), x1)


def _inverse(d: float, part: Part) -> float:
    """sup{v in [0, 1] : interpolant(v) <= d} of a part: its CDF at d days (0 below the first knot, 1 from the last).

    A d within 8 ulps of a knot's days is read as that knot: ``ppf`` returns days / 7, and 7 (x / 7) need not be x
    again, which at a run of equal knots (an atom at x days) would drop the whole run.
    """
    ps, xs = part
    k = bisect.bisect_left(xs, d)
    for near in (k - 1, k):
        if 0 <= near < len(xs) and abs(d - xs[near]) <= 8.0 * math.ulp(xs[near]):
            d = xs[near]
            break
    if d < xs[0]:
        return 0.0
    j = bisect.bisect_right(xs, d) - 1
    if j >= len(xs) - 1:
        return 1.0
    x0, x1 = xs[j], xs[j + 1]  # x0 <= d < x1, so x1 > x0
    return ps[j] + (ps[j + 1] - ps[j]) * (d - x0) / (x1 - x0)


@dataclass(frozen=True)
class Law:
    """A law by kind, with ``params`` per kind.

    - ``constant``: (value,)
    - ``uniform``: (lo, hi)
    - ``lognormal_days``: (mu, sigma) of ln(days); returns weeks
    - ``lognormal_weeks``: (mu, sigma) of ln(weeks)
    - ``one_day_or_lognormal_days``: (p_one_day, mu, sigma): one day with probability p, else LN(mu, sigma) days (the
      MID5 use-of-force shape of §4.4, row "Material outage")
    - ``persistent_mix``: (pi_long, mu_t, sigma_t, mu_p, sigma_p): transient LN(mu_t, sigma_t) days with probability
      1 - pi_long, persistent LN(mu_p, sigma_p) weeks with probability pi_long (39); needs two uniforms
    - ``atom_or_knots_days``: (p_le0, m, the m knots (p, days) of the <= 0 part, the knots of the > 0 part): an
      announcement lead in days (49), returned in weeks. One uniform u reads the <= 0 part's linear interpolant at
      u / p_le0 when u < p_le0, else the > 0 part's at (u - p_le0) / (1 - p_le0); each part is the type-7 quantile
      function of its sample (``lead_law``; ``scripts/python/evidence/lead_laws.py``), so the law puts no mass below
      the sample minimum or above its maximum (Q97; design §12 "Lead laws")
    - ``weibull_days``: (shape k, scale lambda in days): x = lambda (-ln(1 - u))^(1/k) / 7 weeks by inversion
      (-ln(1 - u) as -log1p(-u)), survival S(x) = exp(-(7 x / lambda)^k); the TIES sanction law of `small` and `full`
      (Q106; design §12 M5 row "Weibull law kind"); k and lambda finite and > 0
    """

    kind: str
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in LAW_KINDS:
            raise ValueError(f"unknown law kind {self.kind!r}")
        if self.kind == _KNOTS:
            _knot_parts(self.params)
            return
        need = {
            "constant": 1,
            "uniform": 2,
            "lognormal_days": 2,
            "lognormal_weeks": 2,
            "one_day_or_lognormal_days": 3,
            "persistent_mix": 5,
            _WEIBULL: 2,
        }[self.kind]
        if len(self.params) != need:
            raise ValueError(f"{self.kind} needs {need} parameters, got {self.params!r}")
        if self.kind == _WEIBULL and not all(
            not isinstance(v, bool) and math.isfinite(v) and v > 0.0 for v in self.params
        ):
            raise ValueError(f"{_WEIBULL} needs a finite shape and scale (days) > 0, got {self.params!r}")

    @property
    def uniforms(self) -> int:
        """How many keyed uniforms one draw consumes."""
        return 2 if self.kind in ("persistent_mix", "one_day_or_lognormal_days") else 1

    def ppf(self, u) -> float:
        """The draw for uniforms ``u`` (a float, or a sequence of ``self.uniforms`` floats), by inversion."""
        us = [float(u)] if np.ndim(u) == 0 else [float(x) for x in u]
        p = self.params
        if self.kind == "constant":
            return float(p[0])
        if self.kind == "uniform":
            return p[0] + (p[1] - p[0]) * us[0]
        if self.kind == "lognormal_days":
            return math.exp(p[0] + p[1] * float(ndtri(us[0]))) / 7.0
        if self.kind == "lognormal_weeks":
            return math.exp(p[0] + p[1] * float(ndtri(us[0])))
        if self.kind == "one_day_or_lognormal_days":  # first uniform picks the branch, second the duration
            return 1.0 / 7.0 if us[0] < p[0] else math.exp(p[1] + p[2] * float(ndtri(us[1]))) / 7.0
        if self.kind == "persistent_mix":  # (39): first uniform picks transient/persistent, second the duration
            if us[0] < p[0]:
                return math.exp(p[3] + p[4] * float(ndtri(us[1])))
            return math.exp(p[1] + p[2] * float(ndtri(us[1]))) / 7.0
        if self.kind == _KNOTS:  # (49): the atom's part below p_le0, the > 0 part above; days / 7
            p_le0, le0, pos = _knot_parts(p)
            u = us[0]
            if u < p_le0:
                return _interp(u / p_le0, le0) / 7.0
            return _interp((u - p_le0) / (1.0 - p_le0), pos) / 7.0
        if self.kind == _WEIBULL:  # Q106: lambda (-ln(1 - u))^(1/k) days, / 7 weeks; monotone in u (§4.1)
            return p[1] * (-math.log1p(-us[0])) ** (1.0 / p[0]) / 7.0
        raise AssertionError(self.kind)

    def is_persistent(self, u) -> bool:
        """For ``persistent_mix``: whether the draw falls in the persistent component (39); False otherwise."""
        return self.kind == "persistent_mix" and float(np.atleast_1d(u)[0]) < self.params[0]

    def components(self) -> tuple[tuple[float, "Law"], ...]:
        """The law as a mixture of one-uniform laws: (weight, law) pairs, weights summing to 1.

        ``one_day_or_lognormal_days``: one day (1/7 week) with weight p, LN(mu, sigma) days with 1 - p;
        ``persistent_mix``: transient LN(mu_t, sigma_t) days with 1 - pi_long, persistent LN(mu_p, sigma_p) weeks with
        pi_long (39); every other kind is itself with weight 1.
        """
        p = self.params
        if self.kind == "one_day_or_lognormal_days":
            return ((p[0], Law("constant", (1.0 / 7.0,))), (1.0 - p[0], Law("lognormal_days", (p[1], p[2]))))
        if self.kind == "persistent_mix":
            return ((1.0 - p[0], Law("lognormal_days", (p[1], p[2]))), (p[0], Law("lognormal_weeks", (p[3], p[4]))))
        return ((1.0, self),)

    def cdf(self, x: float) -> float:
        """P(draw <= x), x in the law's unit (weeks for the duration kinds)."""
        if len(parts := self.components()) > 1:
            return math.fsum(w * law.cdf(x) for w, law in parts)
        p = self.params
        if self.kind == _KNOTS:  # the generalised inverse of each part's interpolant, at x weeks = 7 x days
            p_le0, le0, pos = _knot_parts(p)
            d = 7.0 * float(x)
            below = p_le0 * _inverse(d, le0) if le0[0] else 0.0
            return below + ((1.0 - p_le0) * _inverse(d, pos) if pos[0] else 0.0)
        if self.kind == "constant":
            return 1.0 if x >= p[0] else 0.0
        if self.kind == "uniform":
            return 1.0 if x >= p[1] else 0.0 if x < p[0] else (x - p[0]) / (p[1] - p[0])
        if x <= 0.0:
            return 0.0
        if self.kind == _WEIBULL:  # 1 - exp(-(7 x / lambda)^k)
            return -math.expm1(-((7.0 * x / p[1]) ** p[0]))
        days = 7.0 if self.kind == "lognormal_days" else 1.0
        return float(ndtr((math.log(days * x) - p[0]) / p[1]))

    def quantile(self, q: float) -> float:
        """F^{-1}(q) = the smallest x with ``cdf(x) >= q``, for q in (0, 1) (``mixture_quantile`` of the components)."""
        return mixture_quantile(q, self.components())

    def mean(self) -> float:
        """E[T] in the law's unit (weeks for the duration kinds), the sum over ``components`` in closed form.

        Raises:
            ValueError: for ``atom_or_knots_days``, a lead law, which has no duration moments here.

        """
        return math.fsum(w * _component_tail(law, -math.inf) for w, law in self.components() if w > 0.0)

    def tail_integral(self, x: float) -> float:
        """int_x^inf S(y) dy = E[(T - x)^+] for x >= 0, in the law's unit.

        Per unit onset rate, the stationary active stock of events that started more than x weeks ago, i.e. the part a
        burn-in of x weeks misses (Q97). Closed form per component: constant max(c - x, 0); uniform
        (hi - x)^2 / (2 (hi - lo)) inside [lo, hi]; the lognormals' partial expectation e^{mu + s^2/2} Phi(s - z)
        - x Phi(-z), z = (ln x - mu) / s in weeks; the Weibull's lambda Gamma(1 + 1/k) Q(1/k, (x / lambda)^k), Q the
        regularised upper incomplete gamma, lambda in weeks. A mixture sums its weighted components (``components``).

        Raises:
            ValueError: for ``atom_or_knots_days``, or x negative or NaN.

        """
        if not x >= 0.0:
            raise ValueError(f"a tail integral needs x >= 0, got {x!r}")
        return math.fsum(w * _component_tail(law, x) for w, law in self.components() if w > 0.0)


def _component_tail(law: "Law", x: float) -> float:
    """int_x^inf S(y) dy of one one-uniform component; x = -inf gives the mean (the integral from 0 of S, T >= 0)."""
    p = law.params
    if law.kind == "constant":
        return float(p[0]) if x == -math.inf else max(float(p[0]) - x, 0.0)
    if law.kind == "uniform":
        lo, hi = float(p[0]), float(p[1])
        if x == -math.inf:
            return 0.5 * (lo + hi)
        if x <= lo:
            return 0.5 * (lo + hi) - x
        return 0.0 if x >= hi else (hi - x) ** 2 / (2.0 * (hi - lo))
    if law.kind in ("lognormal_days", "lognormal_weeks"):
        mu, s = (p[0] - math.log(7.0), p[1]) if law.kind == "lognormal_days" else (p[0], p[1])
        m = math.exp(mu + 0.5 * s * s)
        if x == -math.inf or x <= 0.0:
            return m
        z = (math.log(x) - mu) / s
        return max(m * float(ndtr(s - z)) - x * float(ndtr(-z)), 0.0)
    if law.kind == _WEIBULL:
        k, lam = p[0], p[1] / 7.0
        m = lam * float(gamma_fn(1.0 + 1.0 / k))
        if x == -math.inf or x <= 0.0:
            return m
        return m * float(gammaincc(1.0 / k, (x / lam) ** k))
    raise ValueError(f"a {law.kind!r} law has no duration moments (mean, tail integral)")


def mixture_quantile(q: float, parts) -> float:
    """F^{-1}(q) of the mixture sum_i w_i F_i of one-uniform laws, ``parts`` = ((w_i, law_i), ...), q in (0, 1).

    A single law inverts exactly (``ppf``). Otherwise the quantile lies between the smallest and the largest component
    quantile at q (at the largest every F_i >= q, below the smallest every F_i < q, up to the rounding of ``ppf``), and
    bisection on the mixture CDF there converges to the smallest x with F(x) >= q, to about one ulp.

    Raises:
        ValueError: if q is not in (0, 1).

    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"a quantile needs q in (0, 1), got {q}")
    parts = [(float(w), law) for w, law in parts if w > 0.0]
    if len(parts) == 1:
        return parts[0][1].ppf(q)
    qs = [law.ppf(q) for _, law in parts]
    lo, hi = min(qs), max(qs)
    for _ in range(2000):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        if math.fsum(w * law.cdf(mid) for w, law in parts) >= q:
            hi = mid
        else:
            lo = mid
    return hi


def lead_law(n_le0: int, n_pos: int, le0: Sequence[tuple[int, float]], pos: Sequence[tuple[int, float]]) -> Law:
    """The ``atom_or_knots_days`` law of a lead sample with ``n_le0`` leads <= 0 and ``n_pos`` leads > 0 (Q97).

    ``le0`` and ``pos`` are the kept order statistics (i, days) of each part's type-7 quantile function, i the 0-based
    rank in the sorted part of m values, at p = i / (m - 1), as ``scripts/python/evidence/lead_laws.py`` prints them
    (``type7_knots``: every run of equal values kept as its first and last knot); a part of one value has the knots
    (0, x) and (1, x). p_le0 = n_le0 / (n_le0 + n_pos), the sample's share of leads <= 0.

    Raises:
        ValueError: if a part's ranks are not 0 .. m - 1 in increasing order, or ``Law``'s checks refuse the knots.

    """

    def knots(pairs: Sequence[tuple[int, float]], m: int) -> list[float]:
        if m == 0:
            if pairs:
                raise ValueError(f"lead_law: a part of no leads has no knots, got {pairs!r}")
            return []
        ranks = [int(i) for i, _ in pairs]
        if m == 1:
            if ranks != [0]:
                raise ValueError(f"lead_law: a part of one lead has the rank 0 alone, got {ranks!r}")
            return [0.0, float(pairs[0][1]), 1.0, float(pairs[0][1])]
        if not ranks or ranks[0] != 0 or ranks[-1] != m - 1 or any(b <= a for a, b in zip(ranks, ranks[1:])):
            raise ValueError(f"lead_law: ranks must rise from 0 to {m - 1}, got {ranks!r}")
        return [v for i, d in pairs for v in (int(i) / (m - 1), float(d))]

    k_le0, k_pos = knots(le0, n_le0), knots(pos, n_pos)
    return Law(_KNOTS, (n_le0 / (n_le0 + n_pos), float(len(k_le0) // 2), *k_le0, *k_pos))


POISSON_COUNT_CAP = 10_000  # a guard: no accepted rate comes near it (u = 1 - 2^-53 at rate 708 gives 935)
POISSON_MAX_RATE = -math.log(sys.float_info.min)  # 708.3964...: exp(-lam) is a normal float up to this rate


def poisson_ppf(u: float, lam: float) -> int:
    """Poisson(lam) by inversion of one uniform: the smallest n with P(N <= n) >= u.

    The CDF is summed from P(N = 0) = exp(-lam) up, term by term, which is exact while exp(-lam) is a normal float:
    lam <= ``POISSON_MAX_RATE`` (about 708.40), where the counts equal SciPy's Poisson quantile (tested over rates
    0-700 at 999 uniforms). A larger rate raises: a subnormal start shifts the counts and a zero one leaves the CDF at
    0 (SPEC-M2R2-01). A uniform above the float CDF's limit (the float sum stalls a few ulps below 1, which a uniform
    of 1 - 2^-53 exceeds at many rates) gets the smallest count at which the float CDF reaches its limit: past the
    mode (n > lam) each term is smaller than the last, so once a term leaves the float sum unchanged every later one
    does too. The count cap ``POISSON_COUNT_CAP`` is a guard that no accepted rate reaches; reaching it raises.
    Monotone in u; in lam up to the float CDF's rounding, which matters only there: at u = 1 - 2^-53, 12 of 20,000
    rates in [1e-6, 500] give one count fewer at 0.79/0.62 times the rate (the old cap gave 1,721 such pairs).

    Raises:
        ValueError: if u is not in (0, 1), if exp(-lam) is not a normal float (lam above about 708.40, infinite or
            NaN), or if the count reaches ``POISSON_COUNT_CAP``; each message names the rate.

    """
    if not 0.0 < u < 1.0:
        raise ValueError(f"poisson_ppf needs a uniform in (0, 1), got {u!r} (rate {lam!r})")
    if lam <= 0.0:
        return 0
    n, term = 0, math.exp(-lam)
    if not term >= sys.float_info.min:
        raise ValueError(
            f"Poisson rate {lam!r}: exp(-rate) = {term!r} is not a normal float, so the inversion by summed terms"
            f" would be wrong; it is exact for rates up to -ln of the smallest normal float, {POISSON_MAX_RATE:.4f}"
        )
    cdf = term
    while cdf < u:
        if n >= POISSON_COUNT_CAP:
            raise ValueError(f"Poisson rate {lam!r}: the inversion of u = {u!r} reached the count cap {n}")
        n += 1
        term *= lam / n
        if n > lam and cdf + term == cdf:  # the float CDF has reached its limit below u: count n - 1 attains it
            return n - 1
        cdf += term
    return n


def categorical_ppf(u: float, weights) -> int:
    """Index i with cumsum(w)[i-1] <= u * sum(w) < cumsum(w)[i], by inversion (weights >= 0, sum > 0)."""
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("categorical_ppf needs a positive total weight")
    c = np.cumsum(w) / total
    return int(min(np.searchsorted(c, u, side="right"), len(w) - 1))

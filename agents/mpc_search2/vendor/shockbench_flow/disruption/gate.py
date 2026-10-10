"""The V12 generator gate, per rung (design §10 V12 and its targets table; Q60, Q88; §11 row 81).

The statistics, each a package function so that the release gate can call them:

(a) **Time rescaling (36).** Every (block, region) stream's onsets, rescaled by the compensator of (30) given the
    episode's regime paths and X (the arrays omega stores), are i.i.d. Exp(1) under the model (Brown et al. 2002;
    Ogata 1988). The streams of all episodes are laid end to end with each stream's censored tail carried into the
    next gap (``hawkes.pooled_residuals``; dropping the tails biases short streams low) and tested by KS against
    Exp(1): pooled, per block, and per stream (Holm within the stream family). The compensator reads the whole cluster
    draw on [-B_burn, T): omega stores neither the burn-in events whose window ended before the instant 0 nor the
    no-op events (no feasible type), yet both excite (30), so the gate takes the generator's full draw of (E_split, n)
    (deterministic by (27)) and, given the omega, checks that the draw's regime paths and X are omega's, bit for bit.
(b) **Marks.** Durations, severities, restoration and rates, per event type, against their laws (``MarkLaws``,
    ``PoissonParams``). A law is split into its atoms and its continuous part by its own components and CDF
    (``Law.components``, ``Law.cdf``; ``split_law``): each atom's share by an exact binomial test (a law that is one
    atom, a constant, exactly), the draws off the atoms by KS against the renormalised continuous CDF (the persistent
    mixture (39), the MID5 one-day atom, the stoppage/slowdown mixture of strikes).
    Tariff rates are the imposing region's tension-state rate at onset exactly (Q32), piracy's rate its severity
    exactly, and a conflict's restoration code the profile's regime exactly. Every draw is tested, burn-in included:
    omega keeps only the events whose window reaches the episode, a length-biased subset (Q40: every draw is kept).
(c) **Fano with regimes frozen.** On constant regime paths the baseline is constant and (35) gives the exact
    covariance of the 104-week window counts; the counts of consecutive windows ending at T, after a warm-up of
    3 t_95 (33), give the Fano factor of every active region and of the total, each tested by TOST +-10 % (alpha 0.05
    per side) with SEs across episodes, Holm across the targets (§10 preamble). The Fano is a variance estimate of
    heavy-tailed counts, so the one-sided p-values are the studentised bootstrap's over episodes (``fano_tost``).
(d) **Conflict layer.** On the generator's regime draws, the weekly onset rate among none-weeks against m_cls
    p-bar_on (TOST +-10 %) and the none / minor / war shares against 0.82 / 0.13 / 0.05 (TOST +-0.01 absolute),
    Holm (``conflict_layer_gate``).
(e) **Tension layer.** On the generator's regime draws, the declared uptime u_p and mean spell T-bar_p of (28), read
    from the chain's own steps: the mean spell as tension weeks per exit (TOST +-10 %), the uptime as the normal share
    h / (h + p_0) of the two-state chain with the estimated exit and none-state entry probabilities (TOST +-10 % of the
    tension share 1 - u_p), Holm (``tension_layer_gate``).
(f) **Militarised closures per year at the anchor rung.** V22's record estimator (``realism``: every militarised
    closure of the whole generated process after its Hawkes transient, per year, SE across episodes) against the
    calibration target of §2.4, TOST +-10 % (``closure_rate_gate``). The calibration itself is analytic, by the
    product of means of (32) under pi^c; only generated episodes, regimes on, test the emergent rate.

Reported, not gated: the unconditional 104-week Fano with regimes on against a high-replication run of the cluster
sampler (``unconditional_fano``).

Pass rules (V12): every KS and binomial test p > 0.001 at a fixed seed, the stream and mark families after Holm (the
pooled and per-block tests unadjusted); every TOST target rejects non-equivalence after Holm.

Power (§11 row 81): ``rate_matched`` builds the hardest gamma alternative, a generator whose rung is off by a given
amount but whose stationary event rate (32) is the declared one (the alternative of ``f5_fano.py`` part D), so the KS
must see the clustering, not the rate. The sizing study itself (episodes for a target power, detections on disjoint
groups of episodes) is the evidence script ``scripts/python/evidence/v12_power.py``, which calls this module.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg, stats

from shockbench_flow.disruption.events import MarkedEvent, regime_week
from shockbench_flow.disruption.hawkes import (
    M_BLOCK,
    P_BLOCK,
    Rescaled,
    baseline_vector,
    branching_matrix,
    immigrant_rates,
    pooled_residuals,
    relaxation_time,
    rescaled_residuals,
    sample_cluster,
    stationary_rates,
)
from shockbench_flow.disruption.laws import Law
from shockbench_flow.disruption.params import GeneratorParams, generator_id
from shockbench_flow.disruption.regime import (
    RegimePaths,
    _class_multipliers,
    normaliser,
    onset_split,
    read_only_paths,
    sample_regimes,
    signal_units,
    weekly_conflict_matrix,
)
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import PIRACY, PORT_STRIKE, REGIONAL_CONFLICT, TARIFF, restoration_regions
from shockbench_flow.omega import codes
from shockbench_flow.omega.seeds import check_entropy
from shockbench_flow.parallel import ordered_map


P_MIN = 1e-3  # V12 pass rule: KS (and binomial) p > 0.001 at a fixed seed; the V22 band test's level too
TOST_ALPHA = 0.05  # §10: TOST alpha 0.05 per side
TOST_MARGIN = 0.10  # §10: +-10 % relative tolerance
FANO_WINDOW = 104  # V12: 104-week Fano with regimes frozen (weeks)
WARMUP_T95 = 3.0  # warm-up before the first Fano window, in t_95 of (33): the mean transient is 20**-3 of its size
FANO_RESAMPLES = 9_999  # bootstrap-t resamples of the Fano TOST: p-values in steps of 1e-4 (Holm's least level 0.05/9)
FANO_SEED = 0x0B12_F4E0_2026_0926  # the resampling entropy, a fixed test entropy outside omega (§10; R5 §4.3)
_RESAMPLE_CHUNK = 500  # resamples per weight matrix
CONFLICT_SHARES = (0.82, 0.13, 0.05)  # V12 targets table: stationary none / minor / war shares (R3 §3; Q32)
SHARE_MARGIN = 0.01  # +-0.01 absolute on each share: twice the rounding of the stated values (``conflict_layer_gate``)
TENSION_COLUMNS = 4  # ``tension_layer_stats``: tension weeks, exits, normal none-weeks, entries


# ----- results ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Test:
    """One test of the gate: ``pvalue`` raw, ``adjusted`` after Holm within its family (equal when alone)."""

    name: str
    kind: str  # "ks", "binomial" or "exact"
    n: int
    statistic: float
    pvalue: float
    adjusted: float

    def passed(self, alpha: float = P_MIN) -> bool:
        return self.adjusted > alpha


@dataclass(frozen=True)
class Report:
    """A family of KS / binomial / exact tests with the V12 pass rule p > ``alpha`` (after Holm where stated)."""

    tests: tuple[Test, ...]
    alpha: float = P_MIN

    @property
    def passed(self) -> bool:
        return all(t.passed(self.alpha) for t in self.tests)

    @property
    def failures(self) -> tuple[Test, ...]:
        return tuple(t for t in self.tests if not t.passed(self.alpha))

    def get(self, name: str) -> Test:
        """The test called ``name``.

        Raises:
            KeyError: if there is none.

        """
        for t in self.tests:
            if t.name == name:
                return t
        raise KeyError(name)

    def lines(self) -> list[str]:
        return [
            f"{t.name}: {t.kind} n={t.n} stat={t.statistic:.4g} p={t.pvalue:.3g} adj={t.adjusted:.3g}"
            for t in self.tests
        ]


class _Named:
    """``get(name)`` over a report's ``targets``."""

    targets: tuple

    def get(self, name: str):
        """The target called ``name``.

        Raises:
            KeyError: if there is none.

        """
        for t in self.targets:
            if t.name == name:
                return t
        raise KeyError(name)


@dataclass(frozen=True)
class FanoTarget:
    """One TOST target: the estimated 104-week Fano factor against (35)."""

    name: str
    estimate: float
    target: float
    ratio: float  # estimate / target
    se: float  # delta-method SE of the ratio across episodes (asymptotically the jackknife's); studentises t*
    p_lower: float  # H0: ratio <= 1 - margin, bootstrap-t
    p_upper: float  # H0: ratio >= 1 + margin, bootstrap-t
    pvalue: float  # max(p_lower, p_upper), the TOST p-value
    adjusted: float  # Holm across the targets


@dataclass(frozen=True)
class FanoReport(_Named):
    """TOST +-``margin`` of every target; passes when every Holm-adjusted TOST p-value is below ``alpha``."""

    targets: tuple[FanoTarget, ...]
    episodes: int
    windows: int  # windows per episode
    margin: float = TOST_MARGIN
    alpha: float = TOST_ALPHA
    resamples: int = 0  # bootstrap-t resamples over episodes

    @property
    def passed(self) -> bool:
        return all(t.adjusted < self.alpha for t in self.targets)

    def lines(self) -> list[str]:
        return [
            f"{t.name}: Fano {t.estimate:.4f} vs (35) {t.target:.4f} ratio {t.ratio:.4f} (SE {t.se:.4f}) "
            f"TOST p={t.pvalue:.3g} adj={t.adjusted:.3g}"
            for t in self.targets
        ]


@dataclass(frozen=True)
class LayerTarget:
    """One TOST target of the conflict layer: H0 ``estimate`` <= ``lower`` or >= ``upper`` (t on episodes - 1 df)."""

    name: str
    estimate: float  # the onset rate as a ratio to m_cls p-bar_on, or a state share
    target: float  # 1 for the ratio, the stated share
    se: float  # SE across episodes
    lower: float
    upper: float
    p_lower: float
    p_upper: float
    pvalue: float  # max(p_lower, p_upper)
    adjusted: float  # Holm across the targets


@dataclass(frozen=True)
class LayerReport(_Named):
    """The conflict-layer TOST family; passes when every Holm-adjusted TOST p-value is below ``alpha``."""

    targets: tuple[LayerTarget, ...]
    episodes: int
    alpha: float = TOST_ALPHA

    @property
    def passed(self) -> bool:
        return all(t.adjusted < self.alpha for t in self.targets)

    def lines(self) -> list[str]:
        return [
            f"{t.name}: {t.estimate:.5g} (SE {t.se:.3g}) vs {t.target:.5g} in [{t.lower:.5g}, {t.upper:.5g}] "
            f"TOST p={t.pvalue:.3g} adj={t.adjusted:.3g}"
            for t in self.targets
        ]


@dataclass(frozen=True)
class UnconditionalFano:
    """The 104-week Fano with regimes on of one target, against the high-replication run (reported, not gated)."""

    name: str
    fano: float  # the gate's episodes
    se: float  # delta-method SE across episodes
    reference: float  # the high-replication run of the cluster sampler
    reference_se: float
    ratio: float  # fano / reference
    pvalue: float  # two-sided z test of fano = reference on the two SEs


@dataclass(frozen=True)
class UnconditionalFanoReport(_Named):
    """Reported per active region and in total (V12 targets table: not gated)."""

    targets: tuple[UnconditionalFano, ...]
    episodes: int
    reference_episodes: int

    def lines(self) -> list[str]:
        return [
            f"{t.name}: Fano {t.fano:.4f} (SE {t.se:.4f}) vs reference {t.reference:.4f} (SE {t.reference_se:.4f}) "
            f"ratio {t.ratio:.4f} p={t.pvalue:.3g}"
            for t in self.targets
        ]


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, in the input order: max over j <= i of min(1, (k - j + 1) p_(j))."""
    p = np.asarray(pvalues, dtype=float)
    k = len(p)
    if k == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((k - np.arange(k)) * p[order]))
    out = np.empty(k)
    out[order] = adj
    return out


def _family(raw: list[tuple[str, str, int, float, float]], adjust: bool) -> list[Test]:
    adj = holm([r[4] for r in raw]) if adjust else [r[4] for r in raw]
    return [Test(name, kind, n, float(stat), float(p), float(a)) for (name, kind, n, stat, p), a in zip(raw, adj)]


# ----- (a) time rescaling (36) ------------------------------------------------------------------------------------


def check_stored_regimes(inst: Instance, params: GeneratorParams, regimes: RegimePaths, omega) -> None:
    """Omega was drawn by ``params`` on ``inst`` and stores exactly ``regimes`` (z_c, z_p, z_dyad, X), bit for bit.

    The residuals (36) are conditional on the regime paths and X stored in omega; this ties a draw to its omega.

    Raises:
        ValueError: on another generator id, instance, burn-in or regime array.

    """
    if omega.generator_id != generator_id(params, inst) or omega.instance_hash != inst.hash:
        raise ValueError("omega was not drawn by these generator params on this instance")
    if omega.burn_in != params.burn_in:
        raise ValueError(f"omega burn-in {omega.burn_in} differs from the params' {params.burn_in}")
    for name, arr in (("z_c", regimes.z_c), ("z_p", regimes.z_p), ("z_dyad", regimes.z_dyad), ("X", regimes.X)):
        stored = omega[name]
        if stored.shape != arr.shape or not np.array_equal(stored.astype(arr.dtype), arr):
            raise ValueError(f"omega's stored {name} is not the draw's regime path")


def episode_residuals(inst: Instance, params: GeneratorParams, draw, omega=None) -> Rescaled:
    """The time-rescaled streams (36) of one episode's draw under ``params`` (the declared model).

    Args:
        inst: the instance.
        params: the declared generator (the model whose compensator rescales the onsets).
        draw: the episode's full draw with ``regimes`` and ``raw`` (``sampler.EventSample``: the whole cluster
            process on [-B_burn, T), burn-in and no-op events included).
        omega: the episode's omega; when given, its stored regime paths and X must be the draw's
            (``check_stored_regimes``).

    """
    if omega is not None:
        check_stored_regimes(inst, params, draw.regimes, omega)
    return rescaled_residuals(inst, params, draw.regimes, list(draw.raw))


def residual_gate(rescaled: Sequence[Rescaled], alpha: float = P_MIN) -> Report:
    """KS of the pooled residuals (36) against Exp(1): all streams, per block, and per (block, region) stream.

    The pooled and the two per-block tests stand alone (Q60: per block and rung); the per-stream tests (the V12 target
    "per block and region") are one family, Holm-adjusted. Streams are those whose compensator reaches T positive in
    some episode.
    """
    streams = sorted({s for res in rescaled for s in res.tails})
    blocks = sorted({b for b, _ in streams})

    def ks(name: str, subset) -> tuple[str, str, int, float, float] | None:
        tau = pooled_residuals(rescaled, streams=subset)
        if len(tau) == 0:
            return None
        k = stats.kstest(tau, "expon")
        return (name, "ks", len(tau), k.statistic, k.pvalue)

    head = [ks("pooled", None)]
    head += [ks(f"block {codes.BLOCKS[b]}", {s for s in streams if s[0] == b}) for b in blocks]
    per = [ks(f"stream {codes.BLOCKS[b]}/{m}", {(b, m)}) for b, m in streams]
    tests = _family([r for r in head if r], adjust=False) + _family([r for r in per if r], adjust=True)
    return Report(tuple(tests), alpha)


# ----- (b) marks --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Split:
    """A law as atoms {value: probability} and continuous parts [(weight, cdf)] (weights summing to 1 overall)."""

    atoms: tuple[tuple[float, float], ...]
    parts: tuple[tuple[float, Callable[[np.ndarray], np.ndarray]], ...]


def split_law(law: Law, weight: float = 1.0) -> _Split:
    """The atoms and continuous parts of a ``laws.Law`` (§4.4 laws, drawn by ``Law.ppf``), scaled by ``weight``.

    The decomposition is the law's own: ``Law.components`` gives its one-uniform components and their weights, and
    ``Law.cdf`` each part's CDF (element-wise). A component's ``ppf`` increases, strictly unless it draws one value (a
    constant, a uniform on a point, a lognormal with sigma 0), so a component whose quartiles coincide is an atom, at
    the float ``Law.ppf`` returns there (``1.0 / 7.0`` for the one-day branch): a draw on an atom equals it exactly.
    Every other component is a continuous part.
    """
    atoms, parts = [], []
    for w, component in law.components():
        if component.ppf(0.25) == component.ppf(0.75):
            atoms.append((component.ppf(0.5), weight * w))
        else:
            parts.append((weight * w, np.vectorize(component.cdf, otypes=[float])))
    return _Split(tuple(atoms), tuple(parts))


def mixture(*weighted: tuple[float, Law]) -> _Split:
    """The split of a mixture sum_i w_i law_i (atoms with equal values merged)."""
    atoms: dict[float, float] = {}
    parts: list = []
    for w, law in weighted:
        s = split_law(law, w)
        for v, a in s.atoms:
            atoms[v] = atoms.get(v, 0.0) + a
        parts += list(s.parts)
    return _Split(tuple(sorted(atoms.items())), tuple(parts))


def law_tests(name: str, values, law: _Split) -> list[tuple[str, str, int, float, float]]:
    """Tests of a sample against a split law: a binomial test per atom (exact for a one-atom law), KS off the atoms.

    A value off every atom when the law has no continuous part fails exactly (p = 0).
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return []
    out = []
    on_atom = np.zeros(n, dtype=bool)
    for v, a in law.atoms:
        hit = x == v
        on_atom |= hit
        k = int(hit.sum())
        if a >= 1.0:
            out.append((f"{name} = {v:g}", "exact", n, float(n - k), 1.0 if k == n else 0.0))
        else:
            out.append((f"{name} atom {v:g}", "binomial", n, k / n, stats.binomtest(k, n, a).pvalue))
    rest = x[~on_atom]
    total = math.fsum(w for w, _ in law.parts)
    if law.parts and len(rest):
        cdf = lambda y: sum(w * f(y) for w, f in law.parts) / total  # noqa: E731
        k = stats.kstest(rest, cdf)
        out.append((f"{name} continuous", "ks", len(rest), k.statistic, k.pvalue))
    elif len(rest) and not any(a >= 1.0 for _, a in law.atoms):
        out.append((f"{name} off the atoms", "exact", n, float(len(rest)), 0.0))
    return out


def mark_gate(inst: Instance, params: GeneratorParams, draws: Sequence, alpha: float = P_MIN) -> Report:
    """KS, binomial and exact tests of every mark by type against the declared laws, one Holm family.

    Args:
        inst: the instance (a regional conflict draws its restoration exactly when its region hosts a fab or an OSAT,
            the nodes that get (13) under §4.4).
        params: the declared generator (its ``laws`` and ``poisson`` are the laws under test).
        draws: the episodes' full draws with ``regimes``, ``marked`` (None for no-op events) and ``poisson``
            (``sampler.EventSample``).
        alpha: the pass rule's level.

    """
    fab_regions = restoration_regions(inst)
    by_type: dict[int, list[MarkedEvent]] = {}
    tariffs: list[tuple[MarkedEvent, RegimePaths]] = []
    for d in draws:
        for ev in list(d.marked) + list(d.poisson):
            if ev is None:
                continue
            by_type.setdefault(ev.type, []).append(ev)
            if ev.type == TARIFF:
                tariffs.append((ev, d.regimes))
    laws = params.laws
    duration, severity = dict(laws.duration), dict(laws.severity)
    raw: list[tuple[str, str, int, float, float]] = []
    for ty in sorted(by_type):
        name = codes.EVENT_TYPES[ty]
        evs = by_type[ty]
        dur = [e.duration for e in evs]
        sev = [e.severity for e in evs]
        if ty == PORT_STRIKE:
            pp = params.poisson
            s = pp.stoppage_share
            raw += law_tests(f"{name} duration", dur, mixture((s, pp.stoppage_duration), (1 - s, pp.slowdown_duration)))
            raw += law_tests(f"{name} severity", sev, mixture((s, pp.stoppage_severity), (1 - s, pp.slowdown_severity)))
            continue
        raw += law_tests(f"{name} duration", dur, split_law(duration[name]))
        raw += law_tests(f"{name} severity", sev, split_law(severity[name]))
        if ty == PIRACY:  # the surcharge c (1 + rate) of §4.4 is the drawn severity
            bad = sum(e.rate != e.severity for e in evs)
            raw.append((f"{name} rate = severity", "exact", len(evs), float(bad), 1.0 if bad == 0 else 0.0))
        if ty == REGIONAL_CONFLICT:
            bad = sum((e.restoration >= 0) != (e.region in fab_regions) for e in evs)
            raw.append((f"{name} restoration where fabs or OSATs", "exact", len(evs), float(bad), float(bad == 0)))
            hit = [e for e in evs if e.restoration >= 0]
            if hit:
                bad = sum(e.restoration != laws.conflict_restoration_regime for e in hit)
                raw.append((f"{name} restoration regime", "exact", len(hit), float(bad), 1.0 if bad == 0 else 0.0))
                raw += law_tests(f"{name} dead time T0", [e.T0 for e in hit], split_law(laws.conflict_dead_time))
                raw += law_tests(f"{name} tau_rho", [e.tau_rho for e in hit], split_law(laws.conflict_tau_rho))
    if tariffs:  # the imposing region's tension state at onset sets the rate (Q32; §2.4 profile)
        bad = 0
        for ev, reg in tariffs:
            z_p = int(reg.z_p[ev.region, reg.col(regime_week(ev, params.burn_in))])
            bad += ev.rate != laws.tariff_rate_by_tension[z_p]
        raw.append(("tariff rate by tension state", "exact", len(tariffs), float(bad), 1.0 if bad == 0 else 0.0))
    return Report(tuple(_family(raw, adjust=True)), alpha)


# ----- (c) Fano with regimes frozen (35) --------------------------------------------------------------------------


def frozen_regimes(inst: Instance, params: GeneratorParams, z_c: int = 0, z_p: int = 0, x=None) -> RegimePaths:
    """Constant regime paths over weeks -B_burn .. T: every region in (z_c, z_p), no dyad at war, X frozen.

    ``x`` is every latent value; by default 0 for regions and dyads and, for chokepoints, the value where
    g(x)/c_X = 1 (29), so the closure modulation is neutral and the baseline is lambda^0 at the frozen regime.

    Raises:
        ValueError: by default, if g(x)/c_X = 1 has no solution below x_cap.

    """
    B, T, R = params.burn_in, inst.T, len(inst.regions)
    D = len(params.regime.dyads)
    units = signal_units(inst, params)
    cols = B + T + 1
    X = np.zeros((len(units), cols))
    if x is None:
        lat = params.latent_chokepoint
        neutral = math.log(normaliser(lat)) / lat.k
        if not neutral < lat.x_cap:
            raise ValueError("g(x)/c_X = 1 has no solution below x_cap (29); pass x explicitly")
        X[R + D :, :] = neutral
    else:
        X[:, :] = float(x)
    zc = np.full((R, cols), z_c, dtype=np.int8)
    z_dyad = np.zeros((D, cols), dtype=np.int8)
    return read_only_paths(B, T, zc, zc.copy(), np.full((R, cols), z_p, dtype=np.int8), z_dyad, X, np.zeros_like(X))


def window_covariance(G: np.ndarray, lam0: np.ndarray, beta_inv: float, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Cov N(T) of the stationary window counts and Lambda, exactly, for a constant baseline (35).

    L = beta (Gamma - Id); L Psi + Psi L' + beta^2 Gamma diag(Lambda) Gamma' = 0; Cov N(T) = diag(Lambda) T + K + K',
    K = [-T L^{-1} + L^{-2} (e^{L T} - Id)] (Psi + beta Gamma diag(Lambda)) (Bacry-Mastromatteo-Muzy 2015 Eq. 29).
    Univariately it is (34).

    Returns:
        (Cov N(window), Lambda) with Lambda = (Id - Gamma)^{-1} lambda^0 of (32).

    """
    n = len(lam0)
    beta = 1.0 / float(beta_inv)
    Lam = stationary_rates(G, lam0)
    L = beta * (G - np.eye(n))
    DL = np.diag(Lam)
    Psi = linalg.solve_continuous_lyapunov(L, -(beta**2) * G @ DL @ G.T)
    Li = np.linalg.inv(L)
    K = (-window * Li + Li @ Li @ (linalg.expm(L * window) - np.eye(n))) @ (Psi + beta * G @ DL)
    return DL * window + K + K.T, Lam


def constant_baseline(inst: Instance, params: GeneratorParams, regimes: RegimePaths) -> np.ndarray:
    """The immigrant rate lambda^0 (2R,) of frozen regime paths (``hawkes.immigrant_rates``, constant over the weeks).

    Raises:
        ValueError: if the rates move from week to week (the regimes are not frozen).

    """
    rates = immigrant_rates(inst, params, regimes)[:, 1:]
    if not np.all(rates == rates[:, :1]):
        raise ValueError("the immigrant rates are not constant: (35) needs frozen regime paths")
    return rates[:, 0].copy()


def _selectors(inst: Instance) -> tuple[list[str], np.ndarray]:
    """Target names and 0/1 rows over the 2R streams: every region (both blocks), then the total."""
    R = len(inst.regions)
    names, rows = [], []
    for m in range(R):
        row = np.zeros(2 * R)
        row[[P_BLOCK * R + m, M_BLOCK * R + m]] = 1.0
        names.append(inst.regions[m])
        rows.append(row)
    names.append("total")
    rows.append(np.ones(2 * R))
    return names, np.array(rows)


def fano_targets(
    inst: Instance, params: GeneratorParams, regimes: RegimePaths, window: float = FANO_WINDOW
) -> dict[str, float]:
    """The window Fano factors Var N / E N of (35) per region with a stationary rate > 0 and in total."""
    lam0 = constant_baseline(inst, params, regimes)
    C, Lam = window_covariance(branching_matrix(inst, params), lam0, params.hawkes.beta_inv, window)
    names, M = _selectors(inst)
    mean = M @ Lam * window
    var = np.einsum("ij,jk,ik->i", M, C, M)
    return {nm: float(v / m) for nm, v, m in zip(names, var, mean) if m > 0}


def window_edges(inst: Instance, params: GeneratorParams, window: float = FANO_WINDOW, warmup: float | None = None):
    """Edges of the consecutive windows ending at T after the warm-up (default ``WARMUP_T95`` t_95 of (33)).

    Raises:
        ValueError: if not one window fits in [-B_burn + warm-up, T).

    """
    if warmup is None:
        warmup = WARMUP_T95 * relaxation_time(branching_matrix(inst, params), params.hawkes.beta_inv)
    span = inst.T + params.burn_in - warmup
    K = int(span // window)
    if K < 1:
        raise ValueError(f"no {window}-week window fits after a warm-up of {warmup:.1f} weeks")
    return inst.T - window * np.arange(K, -1, -1, dtype=float)


def window_counts(inst: Instance, raw, edges: np.ndarray) -> np.ndarray:
    """Counts (K, 2R) of the cluster events per window [edges[k], edges[k + 1]) and (block, region) stream."""
    R = len(inst.regions)
    K = len(edges) - 1
    out = np.zeros((K, 2 * R), dtype=np.int64)
    if not raw:
        return out
    on = np.array([e.onset for e in raw], dtype=float)
    s = np.array([e.block * R + e.region for e in raw], dtype=np.int64)
    w = np.searchsorted(edges, on, side="right") - 1
    ok = (w >= 0) & (w < K)
    np.add.at(out, (w[ok], s[ok]), 1)
    return out


def _fano_moments(a: np.ndarray, b: np.ndarray, windows: int, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pooled Fano factors and their delta-method SEs for each row of episode weights.

    Args:
        a: (N, J) per-episode sums of the window counts of each target.
        b: (N, J) per-episode sums of their squares.
        windows: windows per episode K.
        weights: (rows, N) episode weights, each row summing to N (ones: the sample; multinomial: a resample).

    Returns:
        (rows, J) Fano factors F = s^2 / m over the N K windows (sample variance with ddof 1) and (rows, J) SEs
        sqrt(sum_n w_n IF_n^2 / (N (N - 1))), IF_n = (b_n / K - mu2) / m - (mu2 / m^2 + 1)(a_n / K - m) the
        influence of episode n on F = mu2 / m - m (the delete-one-episode jackknife's SE, asymptotically).

    """
    N = a.shape[0]
    n = N * windows
    A, S = weights @ a, weights @ b
    m, mu2 = A / n, S / n
    fano = (S - A * A / n) / (n - 1) / m
    c1 = 1.0 / (m * windows)  # IF_n = c1 b_n + c2 a_n + m
    c2 = -(mu2 / (m * m) + 1.0) / windows
    q = (
        c1 * c1 * (weights @ (b * b))
        + c2 * c2 * (weights @ (a * a))
        + N * m * m
        + 2.0 * c1 * c2 * (weights @ (a * b))
        + 2.0 * c1 * m * S
        + 2.0 * c2 * m * A
    )
    return fano, np.sqrt(np.maximum(q, 0.0) / (N * (N - 1)))


def evaluation_generator(seed: int) -> np.random.Generator:
    """``Generator(PCG64DXSM(SeedSequence(seed)))``: the resampling stream of a gate test, outside omega (§4.1).

    The §4.1 rule table constructs PCG64DXSM explicitly and keeps bootstrap, sign-flip and TOST entropy outside omega:
    ``numpy.random.default_rng`` would pick PCG64, whose choice NumPy may change between releases. ``seed`` is a fixed
    test entropy (R5 §4.3), never a split's E_split, and is checked as one (an integer in [0, 2**128)).

    Raises:
        ValueError: if the seed is negative or not below 2**128.
        TypeError: if it is not an integer (a bool included).

    """
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(check_entropy(seed))))


def fano_tost(
    x: np.ndarray,
    targets: Sequence[float],
    names: Sequence[str],
    margin: float = TOST_MARGIN,
    alpha: float = TOST_ALPHA,
    resamples: int = FANO_RESAMPLES,
    seed: int = FANO_SEED,
) -> FanoReport:
    """TOST +-``margin`` of pooled window Fano factors against their targets, bootstrap-t over episodes, Holm.

    Each target's Fano is the sample variance (ddof 1) over the mean of its counts pooled over every window of every
    episode, as a ratio to its target, with the delta-method SE across episodes (``_fano_moments``). The Fano is a
    variance estimate of heavy-tailed counts: its sampling law is skewed and its SE moves with it, so the two
    one-sided p-values are the studentised bootstrap's (Hall 1992), not a t law's: episodes (the independent units;
    windows within an episode are dependent) are resampled with replacement ``resamples`` times, t* = (F* - F) / SE*
    each time, and p_lower = (1 + #{t* >= (r - (1 - margin)) / SE}) / (resamples + 1), p_upper = (1 + #{t* <= (r -
    (1 + margin)) / SE}) / (resamples + 1), r the ratio. Resampling draws from ``evaluation_generator(seed)``, an
    explicitly constructed PCG64DXSM on a fixed test entropy outside omega (§4.1 rule table; §10, R5 §4.3), so a report
    is reproducible.

    Args:
        x: (episodes, windows, targets) counts per target.
        targets: the target Fano factors (> 0), one per target.
        names: one name per target.
        margin: the relative tolerance (0.10 by §10).
        alpha: the TOST level per side (0.05 by §10).
        resamples: bootstrap resamples.
        seed: the resampling entropy.

    Raises:
        ValueError: on fewer than two episodes, mismatched targets or names, a target that is not positive and finite,
            a target without counts, no spread across episodes, or a seed outside [0, 2**128).
        TypeError: on a seed that is not an integer.

    """
    x = np.asarray(x, dtype=float)
    tgt = np.asarray(targets, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != len(tgt) or len(names) != len(tgt):
        raise ValueError(f"x must be (episodes >= 2, windows, {len(tgt)}) with a name per target, got {x.shape}")
    if not np.all(np.isfinite(tgt) & (tgt > 0.0)):
        raise ValueError("every target Fano factor must be positive and finite")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    N, K, J = x.shape
    a, b = x.sum(axis=1), (x * x).sum(axis=1)
    if np.any(a.sum(axis=0) == 0.0):
        raise ValueError("a target has no count in any window")
    fano, se = (v[0] for v in _fano_moments(a, b, K, np.ones((1, N))))
    if np.any(se == 0.0):
        raise ValueError("a target's Fano does not vary across episodes: no SE")
    ratio, se_r = fano / tgt, se / tgt
    t_lo, t_hi = (ratio - (1.0 - margin)) / se_r, (ratio - (1.0 + margin)) / se_r
    rng = evaluation_generator(seed)
    above, below = np.zeros(J), np.zeros(J)
    for start in range(0, resamples, _RESAMPLE_CHUNK):
        w = rng.multinomial(N, np.full(N, 1.0 / N), size=min(_RESAMPLE_CHUNK, resamples - start)).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):  # a resample without counts or spread gives no t* (nan)
            fb, sb = _fano_moments(a, b, K, w)
            tb = (fb - fano) / sb  # counted on neither side when nan
        above += np.count_nonzero(tb >= t_lo, axis=0)
        below += np.count_nonzero(tb <= t_hi, axis=0)
    p_lo, p_hi = (1.0 + above) / (resamples + 1), (1.0 + below) / (resamples + 1)
    p = np.maximum(p_lo, p_hi)
    adj = holm(p)
    out = tuple(
        FanoTarget(str(nm), float(f), float(t), float(r), float(s), float(lo), float(hi), float(pp), float(q))
        for nm, f, t, r, s, lo, hi, pp, q in zip(names, fano, tgt, ratio, se_r, p_lo, p_hi, p, adj)
    )
    return FanoReport(out, N, K, margin, alpha, resamples)


def fano_gate(
    inst: Instance,
    params: GeneratorParams,
    regimes: RegimePaths,
    counts: np.ndarray,
    window: float = FANO_WINDOW,
    margin: float = TOST_MARGIN,
    alpha: float = TOST_ALPHA,
    resamples: int = FANO_RESAMPLES,
    seed: int = FANO_SEED,
) -> FanoReport:
    """V12 (c): TOST +-``margin`` of every active region's and the total window Fano against (35), Holm (``fano_tost``).

    Args:
        inst: the instance.
        params: the declared generator (the (35) targets).
        regimes: the frozen regime paths the counts were drawn on (``frozen_regimes``).
        counts: (episodes, windows, 2R) window counts per (block, region) stream (``window_counts``).
        window: the window length in weeks (104 by V12).
        margin: the relative tolerance (0.10 by §10).
        alpha: the TOST level per side (0.05 by §10).
        resamples: bootstrap-t resamples over episodes.
        seed: the resampling entropy.

    Raises:
        ValueError: on fewer than two episodes or counts of another stream count (and as ``fano_tost``).

    """
    counts = np.asarray(counts, dtype=float)
    R = len(inst.regions)
    if counts.ndim != 3 or counts.shape[0] < 2 or counts.shape[2] != 2 * R:
        raise ValueError(f"counts must be (episodes >= 2, windows, {2 * R}), got {counts.shape}")
    want = fano_targets(inst, params, regimes, window)
    names, M = _selectors(inst)
    keep = [i for i, nm in enumerate(names) if nm in want]
    names, M = [names[i] for i in keep], M[keep]
    return fano_tost(counts @ M.T, [want[nm] for nm in names], names, margin, alpha, resamples, seed)


# ----- reported: the unconditional Fano with regimes on ------------------------------------------------------------


def unconditional_fano(inst: Instance, counts: np.ndarray, reference: np.ndarray) -> UnconditionalFanoReport:
    """The 104-week Fano with regimes on, per active region and in total, against a high-replication run (reported).

    The V12 targets table: "unconditional Fano with regimes on | reported against a high-replication run of the
    cluster sampler, not gated". With the regime layers moving, the baseline is not constant and (35) does not apply,
    so the reference is the cluster sampler's own Fano on many more episodes, each on its own regime draw
    (``regime_counts``). Both Fano factors pool every window of every episode (``_fano_moments``, delta-method SE);
    the p-value is the two-sided z test of equality on the two SEs. Regions without a count in the reference are left
    out.

    Args:
        inst: the instance.
        counts: (episodes, windows, 2R) window counts of the episodes under report (the gate's full draws).
        reference: (reference episodes, windows, 2R) window counts of the high-replication run, on the same windows.

    Raises:
        ValueError: on fewer than two episodes in either, counts of another stream or window count, or a target the
            reference counts that the episodes under report never count.

    """
    counts, reference = np.asarray(counts, dtype=float), np.asarray(reference, dtype=float)
    R = len(inst.regions)
    for arr in (counts, reference):
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[2] != 2 * R:
            raise ValueError(f"counts must be (episodes >= 2, windows, {2 * R}), got {arr.shape}")
    if counts.shape[1] != reference.shape[1]:
        raise ValueError("the counts and the reference must share their windows")
    names, M = _selectors(inst)
    keep = np.flatnonzero((reference @ M.T).sum(axis=(0, 1)) > 0.0)  # regions the reference ever counts, and the total
    if np.any((counts @ M[keep].T).sum(axis=(0, 1)) == 0.0):
        raise ValueError("a target the reference counts has no count in the episodes under report")
    names, M = [names[j] for j in keep], M[keep]
    moments = []
    for arr in (counts, reference):
        x = arr @ M.T
        a, b = x.sum(axis=1), (x * x).sum(axis=1)
        moments.append([v[0] for v in _fano_moments(a, b, x.shape[1], np.ones((1, len(x))))])
    (fano, se), (ref, ref_se) = moments
    z = (fano - ref) / np.hypot(se, ref_se)
    out = tuple(
        UnconditionalFano(
            nm, float(f), float(s), float(r), float(rs), float(f / r), float(2.0 * stats.norm.sf(abs(zz)))
        )
        for nm, f, s, r, rs, zz in zip(names, fano, se, ref, ref_se, z)
    )
    return UnconditionalFanoReport(out, len(counts), len(reference))


# ----- the conflict-layer targets -----------------------------------------------------------------------------------


def onset_targets(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """(R,) m_cls p-bar_on of each region, the mean weekly onset probability among none-weeks under the tilt (29).

    c_X is fixed so that the tilted onset rate over none-weeks stays m_cls p-bar_on (§4.3 "Latent risk", Q51).
    """
    return _class_multipliers(inst, params.regime) * onset_split(weekly_conflict_matrix(params.regime.P_yr))[0]


def conflict_layer_stats(inst: Instance, params: GeneratorParams, regimes: RegimePaths) -> np.ndarray:
    """(5,) of one episode's regime paths: onsets, their expected count, and the region-weeks in none, minor and war.

    Read on each region's own conflict chain (``z_c_own``, the chain (29) acts on) over its whole path, weeks
    -B_burn .. T: an onset is a week-to-week step from none to minor or war; the expected count is the sum over
    regions of m_cls p-bar_on (``onset_targets``) times the region's none-weeks that have a next week.
    """
    z = np.asarray(regimes.z_c_own)
    none = z[:, :-1] == 0
    onsets = np.count_nonzero(none & (z[:, 1:] > 0))
    expected = math.fsum((onset_targets(inst, params) * np.count_nonzero(none, axis=1)).tolist())
    return np.array([onsets, expected, *np.bincount(z.ravel(), minlength=3)[:3]], dtype=float)


def _conflict_stats(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> np.ndarray:
    return conflict_layer_stats(inst, params, sample_regimes(inst, params, entropy, episode, noise=False))


def conflict_layer_draws(
    inst: Instance, params: GeneratorParams, entropy: int, episodes: Sequence[int], n_jobs: int = 1
) -> np.ndarray:
    """(episodes, 5) ``conflict_layer_stats`` of the generator's regime draws (``regime.sample_regimes``, no W)."""
    return np.array(ordered_map(functools.partial(_conflict_stats, inst, params, entropy), episodes, n_jobs))


def conflict_layer_gate(
    per_episode: np.ndarray,
    shares: Sequence[float] = CONFLICT_SHARES,
    rate_margin: float = TOST_MARGIN,
    share_margin: float = SHARE_MARGIN,
    alpha: float = TOST_ALPHA,
) -> LayerReport:
    """The V12 conflict-layer target: the onset rate among none-weeks and the stationary shares, TOST, Holm.

    The onset rate is the ratio of the episodes' onsets to their expected count (``conflict_layer_stats``), a ratio
    estimator with its SE across episodes, against 1 +- ``rate_margin`` (§10's relative tolerance). Each share is
    the mean over episodes of the episode's share of region-weeks in the state, against the stated share +-
    ``share_margin`` absolute: the V12 row states 0.82 / 0.13 / 0.05 to two decimals, the untilted chain's law is
    0.8197 / 0.1323 / 0.0481, and the tilt of (29) moves the war share to about 0.046 (8 % below 0.05; the design's
    Monte Carlo 0.047), so a relative margin on the war share would not be testable at feasible episode counts. One-
    sided t tests on episodes - 1 degrees of freedom; Holm over the four.

    Args:
        per_episode: (episodes, 5) of ``conflict_layer_stats``.
        shares: the stated none / minor / war shares.
        rate_margin: the onset ratio's relative tolerance.
        share_margin: the shares' absolute tolerance.
        alpha: the TOST level per side.

    Raises:
        ValueError: on fewer than two episodes, another column count, or no expected onset.

    """
    s = np.asarray(per_episode, dtype=float)
    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != 5:
        raise ValueError(f"per_episode must be (episodes >= 2, 5), got {s.shape}")
    N = len(s)
    onsets, expected = s[:, 0], s[:, 1]
    if expected.sum() <= 0.0:
        raise ValueError("no expected onset: the episodes have no none-week")
    r = float(onsets.sum() / expected.sum())
    se_rate = math.sqrt(np.sum((onsets - r * expected) ** 2) / (N * (N - 1))) / float(expected.mean())
    rows = [("onset rate", r, 1.0, se_rate, 1.0 - rate_margin, 1.0 + rate_margin)]
    frac = s[:, 2:] / s[:, 2:].sum(axis=1, keepdims=True)
    for k, name in enumerate(("none", "minor", "war")):
        m, sd = float(frac[:, k].mean()), float(frac[:, k].std(ddof=1)) / math.sqrt(N)
        rows.append((f"share {name}", m, float(shares[k]), sd, shares[k] - share_margin, shares[k] + share_margin))
    return _layer_report(rows, N, alpha)


def _layer_report(rows: list[tuple], N: int, alpha: float) -> LayerReport:
    """TOST of each row (name, estimate, target, SE, lower, upper): one-sided t tests on N - 1 df, Holm."""
    p_lo = [float(stats.t.sf((est - lo) / se, N - 1)) if se > 0 else float(est <= lo) for _, est, _, se, lo, _ in rows]
    p_hi = [float(stats.t.cdf((est - hi) / se, N - 1)) if se > 0 else float(est >= hi) for _, est, _, se, _, hi in rows]
    p = np.maximum(p_lo, p_hi)
    adj = holm(p)
    out = tuple(
        LayerTarget(nm, float(est), float(tg), float(se), float(lo), float(hi), a, b, float(pp), float(q))
        for (nm, est, tg, se, lo, hi), a, b, pp, q in zip(rows, p_lo, p_hi, p, adj)
    )
    return LayerReport(out, N, alpha)


# ----- the tension-layer targets ------------------------------------------------------------------------------------


def tension_layer_stats(regimes: RegimePaths) -> np.ndarray:
    """(4,) of one episode's regime paths: the steps of the tension layer (28) on every region's whole path.

    Weeks -B_burn .. T; a step goes from week t to t + 1 and reads week t's effective conflict state z^c (the state
    ``regime.tension_step`` reads). Columns: the tension weeks with a next week and the exits among them (tension ->
    normal, probability 1/T-bar_p whatever z^c), then the normal weeks at effective z^c = none with a next week and the
    entries among them (normal -> tension, probability p_0 = (1 - u_p)/(u_p T-bar_p) at none).
    """
    zp, zc = np.asarray(regimes.z_p), np.asarray(regimes.z_c)
    now, nxt = zp[:, :-1], zp[:, 1:]
    tension = now == 1
    normal_none = (now == 0) & (zc[:, :-1] == 0)
    return np.array(
        [
            np.count_nonzero(tension),
            np.count_nonzero(tension & (nxt == 0)),
            np.count_nonzero(normal_none),
            np.count_nonzero(normal_none & (nxt == 1)),
        ],
        dtype=float,
    )


def _tension_stats(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> np.ndarray:
    return tension_layer_stats(sample_regimes(inst, params, entropy, episode, noise=False))


def tension_layer_draws(
    inst: Instance, params: GeneratorParams, entropy: int, episodes: Sequence[int], n_jobs: int = 1
) -> np.ndarray:
    """(episodes, 4) ``tension_layer_stats`` of the generator's regime draws (``regime.sample_regimes``, no W)."""
    return np.array(ordered_map(functools.partial(_tension_stats, inst, params, entropy), episodes, n_jobs))


def tension_layer_gate(
    per_episode: np.ndarray,
    uptime: float,
    spell: float,
    margin: float = TOST_MARGIN,
    alpha: float = TOST_ALPHA,
) -> LayerReport:
    """The V12 tension-layer target: the declared uptime u_p and mean spell T-bar_p of (28), TOST, Holm.

    Read from the chain's own steps (``tension_layer_stats``), which (28) fixes whatever the conflict layer does: the
    exit probability h = exits / tension weeks is 1/T-bar_p, and the entry probability at none p_0 = entries / normal
    none-weeks is (1 - u_p)/(u_p T-bar_p). So the mean spell 1/h is T-bar_p and the normal share of the two-state chain
    at none, h / (h + p_0), is u_p, exactly. The direct readings are biased: completed spells of a finite path are
    length-biased short, and tension begun in minor or war weeks carries into none-weeks. Each estimate is a ratio
    estimator pooled over the episodes, with its delta-method SE across episodes. The mean spell is tested against
    T-bar_p +- ``margin`` relative (§10). The uptime is tested against u_p +- ``margin`` (1 - u_p), the relative margin
    read on the tension share 1 - u_p, the smaller share: on u_p = 0.8 a relative margin on u_p itself would let the
    tension share be 40 % off, and would pass the p_0 of (28) with u_p dropped from its denominator (uptime 5/6).
    One-sided t tests on episodes - 1 degrees of freedom; Holm over the two.

    Args:
        per_episode: (episodes, 4) of ``tension_layer_stats``.
        uptime: the declared u_p, in (0, 1).
        spell: the declared T-bar_p in weeks, >= 1.
        margin: the relative tolerance of the spell and of the tension share.
        alpha: the TOST level per side.

    Raises:
        ValueError: on fewer than two episodes, another column count, no exit or no entry step at none, or a declared
            u_p outside (0, 1) or T-bar_p below 1.

    """
    s = np.asarray(per_episode, dtype=float)
    if s.ndim != 2 or s.shape[0] < 2 or s.shape[1] != TENSION_COLUMNS:
        raise ValueError(f"per_episode must be (episodes >= 2, {TENSION_COLUMNS}), got {s.shape}")
    if not (0.0 < uptime < 1.0 and spell >= 1.0 and math.isfinite(spell)):
        raise ValueError(
            f"(28): the declared u_p must lie in (0, 1) and T-bar_p be finite and >= 1, got {uptime}, {spell}"
        )
    N = len(s)
    weeks, exits, none_weeks, entries = s.T
    if exits.sum() <= 0.0 or none_weeks.sum() <= 0.0:
        raise ValueError("no exit from tension or no normal none-week: the tension targets are not estimable")
    h = float(exits.sum() / weeks.sum())
    p0 = float(entries.sum() / none_weeks.sum())
    mean_spell = 1.0 / h
    se_spell = math.sqrt(np.sum((weeks - mean_spell * exits) ** 2) / (N * (N - 1))) / float(exits.mean())
    if_h = (exits - h * weeks) / float(weeks.mean())  # influence of each episode on h and p_0 (ratio estimators)
    if_p = (entries - p0 * none_weeks) / float(none_weeks.mean())
    u = h / (h + p0)
    if_u = (p0 * if_h - h * if_p) / (h + p0) ** 2  # the delta method on u = h / (h + p_0)
    se_u = math.sqrt(float(np.sum(if_u**2)) / (N * (N - 1)))
    band = margin * (1.0 - uptime)
    rows = [
        ("mean tension spell", mean_spell, spell, se_spell, (1.0 - margin) * spell, (1.0 + margin) * spell),
        ("uptime at none", u, uptime, se_u, uptime - band, uptime + band),
    ]
    return _layer_report(rows, N, alpha)


# ----- the militarised closure rate at the anchor rung --------------------------------------------------------------


def closure_rate_gate(
    mean: float, se: float, episodes: int, target: float, margin: float = TOST_MARGIN, alpha: float = TOST_ALPHA
) -> LayerReport:
    """The V12 target "militarised closures per year at the anchor rung": TOST +-``margin`` against ``target``.

    ``mean`` and ``se`` are closures per year over independent episodes with the SE across them, the record estimator
    of V22 (``realism.summarize``, ``record_closures_per_year``: the militarised closures of each episode's whole
    generated process with onset in [``realism.record_start``, T)); ``target`` is the calibration target of §2.4 at the
    anchor, 0.43 per year on the seven flagship chokepoints, 0.43 x 4/7 on `tiny`'s `chk`. One-sided t tests on
    episodes - 1 degrees of freedom against (1 -+ margin) target (§10); one target, so Holm leaves its p-value as is.

    Raises:
        ValueError: on fewer than two episodes, a non-finite mean, a negative or non-finite SE, or a target <= 0.

    """
    if episodes < 2 or not math.isfinite(mean) or not (se >= 0.0 and math.isfinite(se)):
        raise ValueError(
            f"the closure-rate TOST needs >= 2 episodes, a finite mean and a finite SE >= 0, got {(mean, se, episodes)}"
        )
    if not (target > 0.0 and math.isfinite(target)):
        raise ValueError(f"the closure-rate target must be finite and > 0, got {target!r}")
    row = ("militarised closures per year", float(mean), float(target), float(se))
    return _layer_report([(*row, (1.0 - margin) * target, (1.0 + margin) * target)], int(episodes), alpha)


# ----- drivers ----------------------------------------------------------------------------------------------------


def _draw(inst: Instance, params: GeneratorParams, entropy: int, episode: int):
    from shockbench_flow.disruption.sampler import sample_events

    return sample_events(inst, params, entropy, episode)


def draw_episodes(inst: Instance, params: GeneratorParams, entropy: int, episodes: Sequence[int], n_jobs: int = 1):
    """The generator's full draws (``sampler.sample_events``) of the given episodes, in order."""
    return ordered_map(functools.partial(_draw, inst, params, entropy), episodes, n_jobs)


def _residuals(
    inst: Instance,
    model: GeneratorParams,
    generator: GeneratorParams,
    entropy: int,
    regimes: RegimePaths | None,
    episode: int,
) -> Rescaled:
    reg = regimes if regimes is not None else sample_regimes(inst, generator, entropy, episode, noise=False)
    raw = sample_cluster(inst, generator, reg, entropy, episode)
    return rescaled_residuals(inst, model, reg, raw)


def residual_draws(
    inst: Instance,
    model: GeneratorParams,
    generator: GeneratorParams,
    entropy: int,
    episodes: Sequence[int],
    regimes: RegimePaths | None = None,
    n_jobs: int = 1,
) -> list[Rescaled]:
    """Residuals (36) under ``model`` of the cluster draws of ``generator`` (the same params for the size of the test).

    Each episode's regime paths are the generator's (``regime.sample_regimes``), or the given paths for every episode
    (the residuals are conditional on the paths, so any path serves; one shared path saves the regime draws).
    """
    return ordered_map(functools.partial(_residuals, inst, model, generator, entropy, regimes), episodes, n_jobs)


def _counts(
    inst: Instance, params: GeneratorParams, regimes: RegimePaths, entropy: int, edges: np.ndarray, episode: int
) -> np.ndarray:
    return window_counts(inst, sample_cluster(inst, params, regimes, entropy, episode), edges)


def fano_counts(
    inst: Instance,
    params: GeneratorParams,
    regimes: RegimePaths,
    entropy: int,
    episodes: Sequence[int],
    edges: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Window counts (episodes, windows, 2R) of the cluster draws of ``params`` on (frozen) ``regimes``."""
    edges = np.asarray(edges, dtype=float)
    return np.array(ordered_map(functools.partial(_counts, inst, params, regimes, entropy, edges), episodes, n_jobs))


def _regime_counts(inst: Instance, params: GeneratorParams, entropy: int, edges: np.ndarray, episode: int):
    reg = sample_regimes(inst, params, entropy, episode, noise=False)
    return window_counts(inst, sample_cluster(inst, params, reg, entropy, episode), edges)


def regime_counts(
    inst: Instance, params: GeneratorParams, entropy: int, episodes: Sequence[int], edges: np.ndarray, n_jobs: int = 1
) -> np.ndarray:
    """Window counts (episodes, windows, 2R) of the cluster draws of ``params``, each on its own regime draw.

    The high-replication reference of ``unconditional_fano``: the regime layers and X of each episode are the
    generator's (``regime.sample_regimes``, without the score noise W, which the cluster process does not read).
    """
    edges = np.asarray(edges, dtype=float)
    return np.array(ordered_map(functools.partial(_regime_counts, inst, params, entropy, edges), episodes, n_jobs))


# ----- power (§11 row 81) -----------------------------------------------------------------------------------------


def stationary_total_rate(inst: Instance, params: GeneratorParams) -> float:
    """1' Lambda of (32), events per week, with the regime multipliers at their stationary means (as the profile).

    The militarised multiplier E[m_M(z^c)] and the policy multiplier over the stationary conflict law
    (``profiles.stationary_baseline``, the baseline of ``profiles.closure_rate``); the closure modulation has mean 1 by
    c_X (29). The baselines are checked first (``hawkes.baseline_vector``).
    """
    from shockbench_flow.disruption.profiles import stationary_baseline

    baseline_vector(inst, params)  # refuses a baseline listed twice, negative, not finite or outside the active regions
    return float(stationary_rates(branching_matrix(inst, params), stationary_baseline(inst, params)).sum())


def rate_matched(inst: Instance, params: GeneratorParams, gamma: float) -> GeneratorParams:
    """A biased generator at rung ``gamma`` with every baseline scaled so its stationary rate (32) is ``params``'.

    The alternative of ``f5_fano.py`` part D (the mis-specified model keeps the true stationary rate): the gate must
    detect the clustering, not the event rate. Only the baselines and the rung change.
    """
    q = params.with_rung(gamma)
    s = stationary_total_rate(inst, params) / stationary_total_rate(inst, q)
    base = tuple((r, s * a, s * b) for r, a, b in q.hawkes.baselines)
    return dataclasses.replace(q, hawkes=dataclasses.replace(q.hawkes, baselines=base))

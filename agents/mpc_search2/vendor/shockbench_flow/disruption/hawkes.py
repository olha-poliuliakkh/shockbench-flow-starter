"""The two-block regime-modulated Hawkes process, sampled exactly by its cluster representation (design §4.2).

Blocks P (policy, 0) and M (militarised, 1); regions in instance order. The branching matrix (31) is block-structured,
Gamma^{BB'}_{mm'} = the mean number of block-B children in region m of a block-B' event in region m', with
Gamma^{PP} = gamma A-bar / spr(A-bar), Gamma^{MM} = min(gamma, gamma^max_M) A-bar / spr(A-bar), Gamma^{PM} = 0.10 (...),
Gamma^{MP} = 0.02 (...), A-bar_mm = 1, A-bar_mm' = r_x A_mm' (Q52). The kernel is beta exp(-beta s), integrating to 1.

Sampling (Q58 E23, §4.1 keys): immigrants per (block, region, week) from the piecewise-constant baseline of the week
(regime multipliers; militarised closures aimed at chokepoint c further multiplied by g(X^t_c)/c_X, Q51) by Poisson
inversion of a keyed uniform, stream 1 key (0, block, region, week + B_burn); onset offsets inside the week, key
(1, block, region, week + B_burn, rank); offspring counts per (child block, child region) by Poisson inversion of the
uniforms of key (2, *parent key); exponential delays, key (3, *parent key, child block, child region, rank). Event keys
are genealogies (Hawkes-Oakes 1974): an immigrant is (block, region, week + B_burn, rank), a child (*parent key, child
block, child region, rank). Offspring counts by inversion are non-decreasing in Gamma, so a lower rung's events are a
subset of a higher rung's with identical marks (§2.6, V2). The burn-in covers weeks -B_burn+1 .. 0 (Q87).

Week t covers [t - 1, t) (1); its baseline and closure factor read column t + B_burn of the regime paths (the state in
force during week t), which is also the week key. ``immigrant_rates`` gives every (block, region, week) intensity:
lambda^0 of the week's regime times the ``math.fsum`` of the shares of ``targets.week_parts``, the typed parts that
``events`` types an immigrant by (one rule, composed once in ``targets.TypeRules``, bit for bit).
``rescaled_residuals`` and ``pooled_residuals`` give the time-rescaling statistic (36) of V9 and V12.
"""

import math
from dataclasses import dataclass

import numpy as np

from shockbench_flow.disruption import targets
from shockbench_flow.disruption.laws import poisson_ppf
from shockbench_flow.disruption.params import GeneratorParams
from shockbench_flow.disruption.regime import signal_units
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes, seeds


P_BLOCK, M_BLOCK = 0, 1


def adjacency_matrix(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """A (R x R) of ``targets.adjacency_dict``, the one adjacency rule (the params' A when set, else the instance's)."""
    R = len(inst.regions)
    A = np.zeros((R, R))
    for i, row in targets.adjacency_dict(inst, params).items():
        for j, w in row.items():
            A[i, j] = w
    return A


def active_mask(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """(R,) bool: regions with a baseline (the profile's active regions)."""
    m = np.zeros(len(inst.regions), dtype=bool)
    for r in params.active_regions:
        m[inst.region_index[r]] = True
    return m


def branching_matrix(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """Gamma of (31) as a (2R x 2R) array indexed [block * R + region]; rows = child, columns = parent.

    A-bar is restricted to the active regions (inactive regions neither receive nor send children).
    """
    R = len(inst.regions)
    act = active_mask(inst, params)
    Abar = params.hawkes.r_cross * adjacency_matrix(inst, params)
    np.fill_diagonal(Abar, 1.0)
    Abar = Abar * act[:, None] * act[None, :]
    spr = float(max(abs(np.linalg.eigvals(Abar)))) if act.any() else 1.0
    base = Abar / spr
    h = params.hawkes
    G = np.zeros((2 * R, 2 * R))
    G[:R, :R] = h.gamma * base  # PP
    G[R:, R:] = min(h.gamma, h.gamma_M_max) * base  # MM
    G[:R, R:] = h.cross_PM * base  # P children of M parents
    G[R:, :R] = h.cross_MP * base  # M children of P parents
    return G


def spectral_radius(G: np.ndarray) -> float:
    """spr(Gamma) (32); must be below 1 (unit test, §4.2)."""
    return float(max(abs(np.linalg.eigvals(G))))


def stationary_rates(G: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Lambda = (Id - Gamma)^{-1} lambda-bar^0 (32), events per week per (block, region)."""
    return np.linalg.solve(np.eye(G.shape[0]) - G, baseline)


def relaxation_time(G: np.ndarray, beta_inv: float) -> float:
    """t_95 = ln 20 / (beta (1 - spr(Gamma))) of (33), in weeks."""
    return math.log(20.0) * beta_inv / (1.0 - spectral_radius(G))


@dataclass(frozen=True)
class RawEvent:
    """One event of the cluster representation, before typing (``events`` types it)."""

    key: tuple[int, ...]  # genealogical key (§4.1)
    block: int  # 0 P, 1 M
    region: int
    onset: float  # weeks, continuous; negative in the burn-in
    parent: int  # index of the parent in the event list, -1 for an immigrant


def baseline_vector(inst: Instance, params: GeneratorParams) -> np.ndarray:
    """(2R,) lambda^0 per [block * R + region] at the neutral regime (multiplier 1), from ``params.hawkes.baselines``.

    Raises:
        ValueError: a region listed twice, a negative or non-finite baseline, or a positive baseline outside the
            active regions (only active regions have a baseline, §2.4 profile).

    """
    R = len(inst.regions)
    active = set(params.active_regions)
    lam = np.zeros(2 * R)
    seen: set[str] = set()
    for r, bp, bm in params.hawkes.baselines:
        if r in seen:
            raise ValueError(f"baseline of region {r!r} listed twice")
        seen.add(r)
        if not (math.isfinite(bp) and math.isfinite(bm) and bp >= 0 and bm >= 0):
            raise ValueError(f"baseline of region {r!r} must be finite and >= 0, got ({bp!r}, {bm!r})")
        if r not in active and (bp > 0 or bm > 0):
            raise ValueError(f"positive baseline for region {r!r} outside the active regions (§2.4 profile)")
        i = inst.region_index[r]
        lam[P_BLOCK * R + i], lam[M_BLOCK * R + i] = bp, bm
    return lam


def check_regimes(inst: Instance, params: GeneratorParams, regimes) -> None:
    """The regime paths span weeks -B_burn .. T of this instance and burn-in (``regime.RegimePaths`` layout).

    The sampler and the event stage read week t at column t + B_burn with B_burn = ``params.burn_in`` (an immigrant's
    week is its key's third component less ``params.burn_in``), so the paths must have exactly that burn-in; z_c and
    z_p have one row per region, z_dyad one per dyad of ``params.regime.dyads`` (a regional conflict's counterpart reads
    every dyad's row) and X one per signal unit.

    Raises:
        ValueError: on another burn-in or T, a burn-in outside the week keys of (27), or arrays of another shape.

    """
    B, T, R = params.burn_in, inst.T, len(inst.regions)
    if regimes.burn_in != B or regimes.T != T:
        raise ValueError(
            f"regime paths cover burn-in {regimes.burn_in} and T {regimes.T}; the params and instance need burn-in {B}"
            f" and T {T}"
        )
    if B < 0 or B + T >= 1 << 32:
        raise ValueError(f"burn-in {B} out of range for the week keys of (27)")
    cols = B + T + 1
    U, D = len(signal_units(inst, params)), len(params.regime.dyads)
    for name, arr, rows in (
        ("z_c", regimes.z_c, R),
        ("z_p", regimes.z_p, R),
        ("z_dyad", regimes.z_dyad, D),
        ("X", regimes.X, U),
    ):
        if np.shape(arr) != (rows, cols):
            raise ValueError(f"regime array {name} has shape {np.shape(arr)}, expected {(rows, cols)}")


def immigrant_rates(
    inst: Instance, params: GeneratorParams, regimes, *, rules: targets.TypeRules | None = None
) -> np.ndarray:
    """Immigrant intensities (events per week), (2R, B_burn + T + 1): row block * R + region, column t + B_burn.

    Week t (column t + B_burn, the week key of §4.1) covers [t - 1, t); column 0 (week -B_burn, before the sampled
    window [-B_burn, T)) is zero. The rate is lambda^0_{m,B}(z) = baseline x multiplier (policy by z^p, militarised by
    z^c, of the week; §2.4 profile) times the ``math.fsum`` of the shares of the week's parts, the parts ``events``
    types an immigrant by: 1 for parts without a militarised closure, each closure part aimed at chokepoint c carrying
    g(X^t_c)/c_X (Q51; offspring stay unmodulated, §4.2). The sums come from the one composition of the rule,
    ``targets.TypeRules.week_totals`` (the fsum of ``TypeRules.week_parts`` per week), so each entry equals lambda^0 x
    fsum(shares of ``week_parts``) bit for bit. ``rules`` is the episode's ``targets.TypeRules`` when the caller has
    built it (``sampler.sample_events`` builds it once per episode, SIMP-M2R2-05), else it is built here.
    """
    check_regimes(inst, params, regimes)
    R, B, T = len(inst.regions), params.burn_in, inst.T
    cols = B + T + 1
    base = baseline_vector(inst, params)
    rules = targets.TypeRules(inst, params, regimes) if rules is None else rules.check(inst, params, regimes)
    h = params.hawkes
    weeks = np.arange(1, cols)  # columns of weeks -B_burn + 1 .. T
    rates = np.zeros((2 * R, cols))
    for s in np.flatnonzero(base):
        b, m = divmod(int(s), R)
        mult = np.asarray(h.policy_by_tension if b == P_BLOCK else h.militarised_by_conflict, dtype=float)
        z_mult = (regimes.z_p if b == P_BLOCK else regimes.z_c)[m, weeks].astype(np.int64)
        rates[s, 1:] = base[s] * mult[z_mult] * rules.week_totals(b, m)  # lambda^0 x fsum of the week's parts (30)
    return rates


def onset_in_week(week: int, u: float) -> float:
    """The onset (week - 1) + u inside week t's interval [t - 1, t) (1); a sum that rounds up to t is moved below it.

    u < 1 always, but (t - 1) + u can round to t when |t| is large (at t = 26 for u within 2e-15 of 1), which would
    put the event in the next week and, at t = T, outside [-B_burn, T). The cluster sampler's immigrants and the
    unexcited Poisson components place their onsets by it.
    """
    s = (week - 1) + u
    return s if s < week else math.nextafter(float(week), -math.inf)


def sample_cluster(
    inst: Instance,
    params: GeneratorParams,
    regimes,
    entropy: int,
    episode: int,
    *,
    rules: targets.TypeRules | None = None,
) -> list[RawEvent]:
    """All events of the two-block process with onset in [-B_burn, T) (burn-in and episode), parents before children.

    The exact cluster representation of (30) (Hawkes-Oakes 1974; §4.2 "Sampling"), keyed by (27) on stream 1.
    Immigrants of (block B, region m, week t), t = -B_burn + 1 .. T, number Poisson(``immigrant_rates``) by inversion
    of the uniform of key (0, B, m, t + B_burn); the rank-r immigrant has key (B, m, t + B_burn, r) and onset t - 1 + u,
    u the uniform of key (1, B, m, t + B_burn, r). An event of key k and onset s has, per child block B' and region m',
    Poisson(Gamma^{B'B}_{m'm}) children by inversion of uniform B' x R + m' of the 2R uniforms of key (2, *k); child r
    has key (*k, B', m', r) and onset s + d, d = -ln(1 - u) / beta (the kernel beta e^{-beta s}), u the uniform of key
    (3, *k, B', m', r). Children with onset >= T are not kept. Each count is non-decreasing in its rate at a fixed
    uniform and no draw moves when another count grows, so raising gamma or a baseline keeps every event with its key
    and onset (V2).

    Args:
        inst: the instance.
        params: the generator parameters (rung in ``params.hawkes.gamma``).
        regimes: ``regime.RegimePaths`` of the episode (baselines and the closure modulation read them).
        entropy: E_split.
        episode: n of (27).
        rules: the episode's ``targets.TypeRules`` (built once per episode by ``sampler.sample_events``,
            SIMP-M2R2-05), or None to build them here.

    Returns:
        The events sorted by (onset, key); ``parent`` indexes this list (-1 for an immigrant). The immigrant
        intensity of a (block, region, week) is lambda^0 of the week's regime times the ``math.fsum`` of the shares
        of ``targets.week_parts`` (closures modulated by g(X^t_c)/c_X); ``events`` types an immigrant by the same
        parts at its key's week, so the modulation and the type draw agree.

    Raises:
        ValueError: spr(Gamma) >= 1 (32), a kernel time that is not positive and finite, regime paths that do not
            match the params and instance, or a bad baseline (``baseline_vector``); an immigrant rate beyond the
            exact range of the Poisson inversion (``laws.poisson_ppf``: above about 708 a week); the entropy and
            episode checks of ``omega.seeds``.

    """
    draw = seeds.KeyedStream(entropy, episode, codes.STREAM_HAWKES)  # keys built here, in [0, 2**32)
    G = branching_matrix(inst, params)
    rho = spectral_radius(G)
    if not rho < 1.0:
        raise ValueError(f"spr(Gamma) = {rho:.6f} >= 1: the process must be subcritical (32)")
    beta_inv = float(params.hawkes.beta_inv)
    if not (beta_inv > 0 and math.isfinite(beta_inv)):
        raise ValueError(f"kernel time 1/beta must be positive and finite, got {beta_inv!r}")
    rates = immigrant_rates(inst, params, regimes, rules=rules)
    R, B, T = len(inst.regions), params.burn_in, inst.T
    onsets: list[float] = []
    keys: list[tuple[int, ...]] = []
    streams: list[int] = []
    parents: list[int] = []  # positions in generation order

    for s in np.flatnonzero(rates.any(axis=1)):  # immigrants: stream 1 keys (0, ...) and (1, ...)
        b, m = divmod(int(s), R)
        for wk in np.flatnonzero(rates[s] > 0.0):
            wk = int(wk)
            count = poisson_ppf(draw.uniform((0, b, m, wk)), float(rates[s, wk]))
            for r in range(count):  # week t = wk - B covers [t - 1, t)
                onsets.append(onset_in_week(wk - B, draw.uniform((1, b, m, wk, r))))
                keys.append((b, m, wk, r))
                streams.append(int(s))
                parents.append(-1)

    i = 0
    while i < len(keys):  # offspring, breadth first: stream 1 keys (2, ...) and (3, ...)
        column = G[:, streams[i]]
        kids = np.flatnonzero(column)
        if kids.size:
            key, onset = keys[i], onsets[i]
            u = draw.uniforms((2, *key), 2 * R)
            for s2 in kids:
                s2 = int(s2)
                b2, m2 = divmod(s2, R)
                for r in range(poisson_ppf(float(u[s2]), float(column[s2]))):
                    child = onset + (-math.log1p(-draw.uniform((3, *key, b2, m2, r)))) * beta_inv
                    if child < T:
                        onsets.append(child)
                        keys.append((*key, b2, m2, r))
                        streams.append(s2)
                        parents.append(i)
        i += 1

    order = sorted(range(len(keys)), key=lambda j: (onsets[j], keys[j]))
    pos = {j: k for k, j in enumerate(order)}
    return [
        RawEvent(keys[j], streams[j] // R, streams[j] % R, onsets[j], pos[parents[j]] if parents[j] >= 0 else -1)
        for j in order
    ]


@dataclass(frozen=True)
class Rescaled:
    """The time-rescaled process of one episode (36), per (block, region) stream.

    ``gaps[s]`` are the compensator increments between the stream's consecutive onsets, the first from -B_burn;
    ``tails[s]`` is the censored increment from its last onset (or -B_burn) to T, for every stream whose compensator
    reaches T positive. Each stream maps to a unit-rate Poisson process on [0, sum(gaps) + tail].
    """

    gaps: dict[tuple[int, int], np.ndarray]
    tails: dict[tuple[int, int], float]


def rescaled_residuals(
    inst: Instance, params: GeneratorParams, regimes, events: list[RawEvent], *, rules: targets.TypeRules | None = None
) -> Rescaled:
    """Time-rescaling of every (block, region) stream by its compensator (36), given the regime paths and X.

    The compensator of a stream is the integral from -B_burn (the process starts empty there) of its conditional
    intensity (30): the piecewise-constant immigrant rate of ``immigrant_rates`` plus sum_q Gamma (1 - e^{-beta (t -
    t_q)}) over earlier events of every stream. Under the model the rescaled streams are independent unit-rate Poisson
    processes (Brown et al. 2002; Ogata 1988; the multivariate time change). ``events`` is the whole cluster process
    (burn-in included), as ``sample_cluster`` returns it. ``pooled_residuals`` turns the result into an i.i.d. Exp(1)
    sample for V9 and V12. ``rules`` as ``immigrant_rates``.
    """
    G = branching_matrix(inst, params)
    rates = immigrant_rates(inst, params, regimes, rules=rules)
    R, B, T = len(inst.regions), params.burn_in, inst.T
    beta = 1.0 / float(params.hawkes.beta_inv)
    csum = np.cumsum(rates, axis=1)  # csum[:, col - 1]: integral of the immigrant rate from -B to the week's start
    S1 = np.zeros(2 * R)  # sum over earlier events of Gamma[:, source]
    S2 = np.zeros(2 * R)  # sum over earlier events of Gamma[:, source] e^{-beta (t - t_q)}
    last = np.zeros(2 * R)
    out: dict[int, list[float]] = {}
    t_prev = -float(B)
    for e in sorted(events, key=lambda ev: (ev.onset, ev.key)):
        x = float(e.onset)
        S2 *= math.exp(-beta * (x - t_prev))
        t_prev = x
        s = e.block * R + e.region
        col = math.floor(x) + 1 + B
        comp = float(csum[s, col - 1] + rates[s, col] * (x - (col - 1 - B)) + (S1[s] - S2[s]))
        out.setdefault(s, []).append(comp - last[s])
        last[s] = comp
        S1 += G[:, s]
        S2 += G[:, s]
    S2 *= math.exp(-beta * (T - t_prev))
    at_T = csum[:, B + T] + (S1 - S2)  # the compensator at T
    gaps = {(s // R, s % R): np.asarray(v) for s, v in sorted(out.items())}
    tails = {(s // R, s % R): float(at_T[s] - last[s]) for s in range(2 * R) if at_T[s] > 0.0}
    return Rescaled(gaps, tails)


def pooled_residuals(results, streams=None) -> np.ndarray:
    """An i.i.d. Exp(1) sample from rescaled episodes (36): their unit-rate processes laid end to end.

    The streams of every episode (in ``results`` order, streams sorted, restricted to ``streams`` if given) are
    concatenated into one unit-rate Poisson process; its gaps are the within-stream gaps, with each stream's censored
    tail added to the next stream's first gap (or carried past a stream without events); only the last, censored gap
    is dropped. Dropping every stream's tail instead would bias short streams' gaps low (mean near 1 - 1/L for a
    stream of compensator L), which a pooled KS test detects.
    """
    out: list[float] = []
    carry = 0.0
    for res in results:
        for s in sorted(res.tails):
            if streams is not None and s not in streams:
                continue
            g = res.gaps.get(s)
            if g is not None and len(g):
                out.append(carry + float(g[0]))
                out.extend(float(x) for x in g[1:])
                carry = 0.0
            carry += res.tails[s]
    return np.asarray(out)

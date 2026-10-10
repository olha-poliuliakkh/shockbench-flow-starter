"""Regime layers (28), dyad chains (Q59) and the latent risks X and noises W of every signal unit (29), (45).

Signal units are ordered regions, then dyads, then chokepoints (codes of §4.1). Arrays span the burn-in and the
episode: column j is week j - B_burn, weeks -B_burn .. T (B_burn + T + 1 columns); week t's value is column t + B_burn.

- Conflict layer z^c (none 0, minor 1, war 2): the weekly embedding Pi^c = expm(logm(P^yr) / 52) of (28), formed
  once per P^yr by SciPy's logm with exact 1-norms, which draws nothing from the global np.random (B2), except that
  the onset probability none -> {minor, war} of region m in week t is p = min(1, m_cls p-bar_on g(X^t_m)/c_X) (29),
  split minor/war in the proportions a : 1 - a of Pi^c's none row, a = Pi^c[0,1] / (Pi^c[0,1] + Pi^c[0,2]);
  p-bar_on = 1 - Pi^c[0, 0]. The none row is [1 - p, p a, p (1 - a)]; the minor and war rows are Pi^c's. The hazard of
  week t reads X at week t and decides the state of week t + 1. A region class missing from ``class_multiplier``
  raises (no default).
- Dyad chains: the same law with m_cls = 1 and the dyad's own latent unit; while a dyad chain is at war, both its
  regions' effective z^c are 2 (``z_c``); a dyad at minor changes nothing; each region's own chain is kept in
  ``z_c_own`` (the region unit's labels, §5.2).
- Tension layer z^p (normal 0, tension 1): normal -> tension with probability q_z = 1 - (1 - p_0)^{c_z},
  p_0 = (1 - u_p)/(u_p T-bar_p), c = (1, c_minor, c_war) by the EFFECTIVE z^c of week t; tension -> normal with
  probability 1/T-bar_p. u_p is the share of time in the normal state: P(tension | z^c) = q_z / (q_z + 1/T-bar_p),
  which is 1 - u_p at z^c = none. u_p T-bar_p >= 1 - u_p and T-bar_p >= 1 are required (§4.2).
- Latent risk and noise: X^t = rho X^{t-1} + sqrt(1 - rho^2) eps^t (29) and W likewise with the unit's own rho (45),
  per signal-unit kind (``latent_region``, ``latent_dyad``, ``latent_chokepoint``).
- Start (§4.3, quasi-static): X and W ~ N(0, 1) at week -B_burn (their column-0 draws); z^c from the stationary law of
  the tilted chain at X^{-B_burn} (closed form by the Markov chain tree theorem); z^p from P(tension | effective z^c)
  at column 0.
- Draws (27): stream 0, key (layer, region or dyad) with layer codes conflict 0, tension 1, dyad 2: one Generator per
  chain drawing B_burn + T + 1 open-interval uniforms (``to_open_unit`` of its raw 64-bit words); u[0] draws the start
  state, u[j] the step from column j - 1 to j. Categorical inversion: the category is the number of cumulative row
  thresholds <= u, categories in code order (the minor/war split inverted from the same uniform as the onset). Stream
  13, key (unit, j, 0) for X and (unit, j, 1) for W: one ``standard_normal()`` of the key's Generator each (the draw of
  ``omega.demand``); column 0's normal is the start value, column j's the innovation eps^j.
"""

import functools
import math
import threading
from dataclasses import dataclass

import numpy as np
from scipy.linalg import _matfuncs_inv_ssq as _inv_ssq  # logm's inverse scaling and squaring (SciPy 1.18.1)
from scipy.linalg import expm, logm
from scipy.sparse.linalg import onenormest
from scipy.stats import norm

from shockbench_flow.disruption.params import WEEKS_PER_YEAR, GeneratorParams, LatentParams, RegimeParams
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes, seeds
from shockbench_flow.omega.seeds import KeyedStream, check_entropy, check_episode, generator, to_open_unit


LAYER_CONFLICT, LAYER_TENSION, LAYER_DYAD = (codes.LAYERS.index(n) for n in ("conflict", "tension", "dyad"))
WAR = codes.CONFLICT_STATES.index("war")


@dataclass(frozen=True)
class RegimePaths:
    """Regime layers and latent risks of one episode (the omega arrays z_c, z_p, z_dyad, X, W); arrays read-only."""

    burn_in: int
    T: int
    z_c: np.ndarray  # (R, B+T+1) int8: effective conflict layer (dyad wars applied)
    z_c_own: np.ndarray  # (R, B+T+1) int8: each region's own chain
    z_p: np.ndarray  # (R, B+T+1) int8
    z_dyad: np.ndarray  # (D, B+T+1) int8
    X: np.ndarray  # (U, B+T+1) float64, units = regions, dyads, chokepoints
    W: np.ndarray | None  # (U, B+T+1) float64; None when drawn without the noise (``sample_regimes(noise=False)``)

    def col(self, week: int) -> int:
        """Column of week ``week`` (weeks -B_burn .. T)."""
        return week + self.burn_in


_LOGM_LOCK = threading.Lock()


def _exact_onenormest(A, t=2, itmax=5, compute_v=False, compute_w=False):
    """SciPy's ``onenormest`` on its exact branch (t = n: the largest column sum of |A|), which draws nothing."""
    return onenormest(A, t=A.shape[1], itmax=itmax, compute_v=compute_v, compute_w=compute_w)


def deterministic_logm(A: np.ndarray) -> np.ndarray:
    """SciPy's ``logm`` with its 1-norm estimates computed exactly: no draw from NumPy's global RandomState (B2).

    ``logm`` picks its number of square roots and its Pade degree from ``onenormest`` estimates, whose random start
    columns come from the legacy global ``np.random``, so a plain call advances the caller's stream. SciPy's
    ``onenormest`` is exact, and draws nothing, when its column count t reaches the order n; this runs ``logm`` with
    the estimator of ``scipy.linalg._matfuncs_inv_ssq`` swapped for that exact branch, under a lock, and restores it.
    On P^yr of (28) every estimate equals the exact norm, so the bits are those of the plain call (the B2 test checks
    them under five global seeds, and on 200 other stochastic matrices). A call of SciPy's logm from another thread
    during the swap gets exact norms too.

    Raises:
        RuntimeError: if SciPy no longer estimates logm's norms through ``_matfuncs_inv_ssq.onenormest`` (the swap
            would not take effect).

    """
    with _LOGM_LOCK:
        if getattr(_inv_ssq, "onenormest", None) is not onenormest:
            raise RuntimeError("scipy.linalg._matfuncs_inv_ssq.onenormest is not SciPy's onenormest: cannot run logm")
        _inv_ssq.onenormest = _exact_onenormest
        try:
            return logm(A)
        finally:
            _inv_ssq.onenormest = onenormest


@functools.cache
def _conflict_matrix(P_yr: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """Pi^c of (28) for one P^yr (a tuple of float rows), once per process, read-only; no global draw (B2, §10)."""
    Pi = np.real(expm(deterministic_logm(np.array(P_yr, dtype=float)) / WEEKS_PER_YEAR))
    Pi = np.clip(Pi, 0.0, None)
    Pi = Pi / Pi.sum(axis=1, keepdims=True)
    Pi.flags.writeable = False
    return Pi


def weekly_conflict_matrix(P_yr) -> np.ndarray:
    """Pi^c = exp(log(P^yr) / 52) of (28), real part, rows renormalised to sum 1 (a valid generator per R3.4).

    Computed once per P^yr (``_conflict_matrix``, through ``deterministic_logm``, which leaves NumPy's global random
    state untouched, B2); each call returns its own writable copy.
    """
    key = tuple(tuple(float(x) for x in row) for row in np.asarray(P_yr, dtype=float))
    return _conflict_matrix(key).copy()


@functools.cache
def normaliser(lat: LatentParams) -> float:
    """c_X: the given constant, or E[exp(k min(X, x_cap))] under X ~ N(0, 1) in closed form (cached per params)."""
    if lat.c is not None:
        return lat.c
    k, a = lat.k, lat.x_cap
    return float(np.exp(k * k / 2.0) * norm.cdf(a - k) + np.exp(k * a) * norm.sf(a))


def tilt(x, lat: LatentParams):
    """g(x)/c_X with g(x) = exp(k min(x, x_cap)) (29)."""
    return np.exp(lat.k * np.minimum(x, lat.x_cap)) / normaliser(lat)


def signal_units(inst: Instance, params: GeneratorParams) -> list[tuple[str, int]]:
    """(kind, index) of every signal unit in order: regions, dyads (index into params.regime.dyads), chokepoints.

    The kinds are ``codes.UNIT_KINDS``, the §9.2 vocabulary; a chokepoint's index is its node index, as the wire's
    ``warning.unit`` carries it (``information.warning.unit_table``).
    """
    region, dyad, chokepoint = codes.UNIT_KINDS
    units = [(region, m) for m in range(len(inst.regions))]
    units += [(dyad, d) for d in range(len(params.regime.dyads))]
    units += [(chokepoint, c) for c in inst.chokepoints]
    return units


# ----- conflict layer (28)-(29) -----------------------------------------------------------------------------------


def onset_split(Pi: np.ndarray) -> tuple[float, float]:
    """(p-bar_on, a): the untilted weekly onset 1 - Pi[0,0] and the minor share a = Pi[0,1] / (Pi[0,1] + Pi[0,2])."""
    return float(1.0 - Pi[0, 0]), float(Pi[0, 1] / (Pi[0, 1] + Pi[0, 2]))


def onset_probability(x, m_cls, Pi: np.ndarray, lat: LatentParams):
    """Tilted onset probability of a none week (29): min(1, m_cls p-bar_on g(x)/c_X), vectorised over x and m_cls."""
    pbar, _ = onset_split(Pi)
    return np.minimum(1.0, np.asarray(m_cls, dtype=float) * pbar * tilt(np.asarray(x, dtype=float), lat))


def _check_codes(z: np.ndarray, n_states: int, what: str) -> np.ndarray:
    z = np.asarray(z)
    if z.size and (z.min() < 0 or z.max() >= n_states):
        raise ValueError(f"{what} holds a code outside 0..{n_states - 1}")
    return z


def conflict_rows(z_prev, x_prev, m_cls, Pi: np.ndarray, lat: LatentParams) -> np.ndarray:
    """Transition rows (..., 3) of the conflict step from week t, state ``z_prev``, latent risk ``x_prev`` (28)-(29).

    None: [1 - p, p a, p (1 - a)] with p of ``onset_probability``; minor and war: the rows of Pi^c.

    Raises:
        ValueError: if a state code is outside 0..2.

    """
    z = _check_codes(z_prev, len(codes.CONFLICT_STATES), "z_c")
    p = onset_probability(x_prev, m_cls, Pi, lat)
    _, a = onset_split(Pi)
    none = z == 0
    rows = np.take(Pi, z, axis=0)  # always a copy: Pi^c's rows, replaced by the tilted none row where z is none
    rows[..., 0] = np.where(none, 1.0 - p, rows[..., 0])
    rows[..., 1] = np.where(none, p * a, rows[..., 1])
    rows[..., 2] = np.where(none, p * (1.0 - a), rows[..., 2])
    return rows


def conflict_stationary(x, m_cls, Pi: np.ndarray, lat: LatentParams) -> np.ndarray:
    """Stationary law (..., 3) of the tilted conflict chain with X frozen at ``x`` (the quasi-static start, §4.3).

    Markov chain tree theorem for three states: pi_i is proportional to the sum, over the spanning trees directed into
    i, of the product of their transition probabilities (off-diagonal entries only, so no cancellation).
    """
    p = onset_probability(x, m_cls, Pi, lat)
    _, a = onset_split(Pi)
    p01, p02 = p * a, p * (1.0 - a)
    p10, p12, p20, p21 = Pi[1, 0], Pi[1, 2], Pi[2, 0], Pi[2, 1]
    w0 = np.broadcast_to(p10 * p20 + p10 * p21 + p12 * p20, np.shape(p))
    w1 = p01 * p21 + p01 * p20 + p02 * p21
    w2 = p02 * p12 + p02 * p10 + p01 * p12
    w = np.stack([w0, w1, w2], axis=-1)
    return w / w.sum(axis=-1, keepdims=True)


def invert(u, rows) -> np.ndarray:
    """Categorical inversion (27): the category is the number of cumulative row thresholds <= u (int8).

    The thresholds are the cumulative sums of each row without its last entry, in category order, so a threshold equal
    to u counts (ties go to the higher category) and u in (0, 1) never exceeds the implicit last threshold 1.
    """
    rows, u = np.asarray(rows, dtype=float), np.asarray(u, dtype=float)
    cum = rows[..., 0]
    out = np.asarray(cum <= u, dtype=np.int8)
    for k in range(1, rows.shape[-1] - 1):
        cum = cum + rows[..., k]  # left-to-right partial sums, as np.cumsum
        out = out + (cum <= u)
    return np.asarray(out, dtype=np.int8)


def conflict_step(z_prev, x_prev, u, m_cls, Pi: np.ndarray, lat: LatentParams) -> np.ndarray:
    """State of week t + 1 from the state and latent risk of week t and the step's uniform (28)-(29)."""
    return invert(u, conflict_rows(z_prev, x_prev, m_cls, Pi, lat))


def conflict_paths(u, X, m_cls, Pi: np.ndarray, lat: LatentParams) -> np.ndarray:
    """Conflict chains (N, ncol) int8 from their uniforms and latent paths (N, ncol); ``m_cls`` scalar or (N,).

    Column 0 is drawn from ``conflict_stationary`` at X[:, 0]; column j from the step of column j - 1.
    """
    u, X = np.asarray(u, dtype=float), np.asarray(X, dtype=float)
    if u.shape != X.shape or u.ndim != 2:
        raise ValueError(f"uniforms {u.shape} and latent paths {X.shape} must be the same (N, ncol)")
    z = np.empty(u.shape, dtype=np.int8)
    if u.shape[1] == 0:
        return z
    z[:, 0] = invert(u[:, 0], conflict_stationary(X[:, 0], m_cls, Pi, lat))  # §4.3 quasi-static start
    for j in range(1, u.shape[1]):
        z[:, j] = conflict_step(z[:, j - 1], X[:, j - 1], u[:, j], m_cls, Pi, lat)  # (28)-(29)
    return z


def effective_conflict(z_own, z_dyad, pairs) -> np.ndarray:
    """Effective conflict layer (Q59): 2 wherever a dyad of the region is at war, else the region's own chain."""
    z = np.array(z_own, dtype=np.int8, copy=True)
    zd = np.asarray(z_dyad)
    for d, (a, b) in enumerate(pairs):
        war = zd[d] == WAR
        z[a, war] = WAR
        z[b, war] = WAR
    return z


def check_own_chain(z_c: np.ndarray, z_c_own: np.ndarray, z_dyad: np.ndarray, dyad_regions: np.ndarray) -> None:
    """Refuse an own-chain array that is not the one the effective layer was built from (M3; §12 'Conflict layer').

    omega stores ``z_c_own`` (R, B+T+1) for the region-unit labels of §5.2 (V27, eta_n) beside the effective ``z_c``
    that drives the dynamics; this checks ``z_c == effective_conflict(z_c_own, z_dyad, dyad_regions)`` bit for bit,
    ``dyad_regions`` the (D, 2) region rows of omega (``omega.container.ARRAY_DTYPES['dyad_regions']``).

    Raises:
        ValueError: on mismatched shapes, a region index outside the layer, or any column where the stored effective
            layer differs.

    """
    z_c, own, zd, pairs = (np.asarray(x) for x in (z_c, z_c_own, z_dyad, dyad_regions))
    if own.shape != z_c.shape or zd.ndim != 2 or pairs.shape != (zd.shape[0], 2) or zd.shape[1] != z_c.shape[1]:
        raise ValueError(
            f"own chain {own.shape}, effective layer {z_c.shape}, dyad chains {zd.shape} and dyad rows {pairs.shape}"
            " do not match (R, weeks), (D, weeks), (D, 2)"
        )
    if pairs.size and not (pairs.min() >= 0 and pairs.max() < z_c.shape[0]):
        raise ValueError(f"dyad rows name regions outside 0..{z_c.shape[0] - 1}: {pairs.tolist()}")
    effective = effective_conflict(own, zd, [(int(a), int(b)) for a, b in pairs])
    bad = np.flatnonzero((effective != z_c).any(axis=0))
    if bad.size:
        raise ValueError(
            f"z_c is not the effective layer of z_c_own and z_dyad (Q59) in {bad.size} columns, the first {int(bad[0])}"
        )


# ----- tension layer (28) -----------------------------------------------------------------------------------------


def tension_entry(params: RegimeParams) -> np.ndarray:
    """P(normal -> tension | z^c) for z^c = none, minor, war (28): 1 - (1 - p_0)^{c_z}, c = (1, c_minor, c_war).

    Raises:
        ValueError: unless 0 < u_p <= 1, T-bar_p >= 1, u_p T-bar_p >= 1 - u_p (so p_0 <= 1) and c_minor, c_war >= 0, all
            finite (§4.2).

    """
    u_p, T_p = float(params.tension_uptime), float(params.tension_spell)
    c = np.array([1.0, params.tension_entry_minor, params.tension_entry_war], dtype=float)
    if not (math.isfinite(u_p) and math.isfinite(T_p) and 0.0 < u_p <= 1.0):
        raise ValueError(f"(28): the tension uptime u_p must be a share in (0, 1] and T-bar_p finite, got {u_p}, {T_p}")
    if not T_p >= 1.0:
        raise ValueError(f"(28): the mean tension spell T-bar_p must be >= 1 week, got {T_p}")
    if not u_p * T_p >= 1.0 - u_p:
        raise ValueError(f"(28): u_p T-bar_p >= 1 - u_p is required (p_0 <= 1), got u_p {u_p}, T-bar_p {T_p}")
    if not (np.all(np.isfinite(c)) and np.all(c >= 0.0)):
        raise ValueError(f"(28): the tension entry multipliers c_minor, c_war must be finite and >= 0, got {c[1:]}")
    p0 = (1.0 - u_p) / (u_p * T_p)
    return 1.0 - (1.0 - p0) ** c


def tension_stationary(z_c, params: RegimeParams):
    """P(tension | z^c) = q_z / (q_z + 1/T-bar_p), q_z = ``tension_entry``; 1 - u_p at z^c = none (28).

    Raises:
        ValueError: on invalid tension parameters or a conflict code outside 0..2.

    """
    q = tension_entry(params)[_check_codes(z_c, len(codes.CONFLICT_STATES), "z_c")]
    return q / (q + 1.0 / float(params.tension_spell))


def tension_rows(zp_prev, zc_prev, params: RegimeParams) -> np.ndarray:
    """Transition rows (..., 2) of the tension step from week t (28): normal [1 - q_z, q_z], tension [1/T, 1 - 1/T].

    ``zc_prev`` is the effective conflict layer of week t.
    """
    zp = _check_codes(zp_prev, len(codes.TENSION_STATES), "z_p")
    q = tension_entry(params)[_check_codes(zc_prev, len(codes.CONFLICT_STATES), "z_c")]
    out = 1.0 / float(params.tension_spell)
    tension = zp == 1
    rows = np.empty(np.broadcast_shapes(zp.shape, q.shape) + (2,), dtype=float)
    rows[..., 0] = np.where(tension, out, 1.0 - q)
    rows[..., 1] = np.where(tension, 1.0 - out, q)
    return rows


def tension_step(zp_prev, zc_prev, u, params: RegimeParams) -> np.ndarray:
    """Tension state of week t + 1 from week t's tension state, effective conflict state and the step's uniform."""
    return invert(u, tension_rows(zp_prev, zc_prev, params))


def tension_paths(u, z_c, params: RegimeParams) -> np.ndarray:
    """Tension chains (R, ncol) int8 from their uniforms and the effective conflict layer (R, ncol).

    Column 0 from P(tension | z^c at column 0); column j from the step of column j - 1.
    """
    u, z_c = np.asarray(u, dtype=float), np.asarray(z_c)
    if u.shape != z_c.shape or u.ndim != 2:
        raise ValueError(f"uniforms {u.shape} and conflict layer {z_c.shape} must be the same (R, ncol)")
    z = np.empty(u.shape, dtype=np.int8)
    if u.shape[1] == 0:
        return z
    p = tension_stationary(z_c[:, 0], params)
    z[:, 0] = invert(u[:, 0], np.stack([1.0 - p, p], axis=-1))  # §4.3
    for j in range(1, u.shape[1]):
        z[:, j] = tension_step(z[:, j - 1], z_c[:, j - 1], u[:, j], params)  # (28)
    return z


# ----- latent risk and noise (29), (45) ---------------------------------------------------------------------------


def ar1_path(eps, rho) -> np.ndarray:
    """AR(1) paths (N, ncol): column 0 = eps[:, 0] (the N(0, 1) start); X^j = rho X^{j-1} + sqrt(1 - rho^2) eps^j.

    Looped over columns, vectorised over units, in the fixed float order (rho X) + (s eps); ``rho`` scalar or (N,).
    """
    eps = np.asarray(eps, dtype=float)
    rho = np.asarray(rho, dtype=float)
    s = np.sqrt(1.0 - rho * rho)
    X = np.empty_like(eps)
    if eps.shape[1] == 0:
        return X
    X[:, 0] = eps[:, 0]
    for j in range(1, eps.shape[1]):
        X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
    return X


def _check_latent(lat: LatentParams, kind: str) -> None:
    if not (math.isfinite(lat.rho) and abs(lat.rho) <= 1.0):
        raise ValueError(f"(29): latent rho of the {kind} units must be finite with |rho| <= 1, got {lat.rho}")
    if not (math.isfinite(lat.k) and math.isfinite(lat.x_cap)):
        raise ValueError(f"(29): latent k and x_cap of the {kind} units must be finite, got {lat.k}, {lat.x_cap}")
    if lat.c is not None and not (math.isfinite(lat.c) and lat.c > 0.0):
        raise ValueError(f"(29): the normaliser c_X of the {kind} units must be finite and > 0, got {lat.c}")


# ----- the sampler ------------------------------------------------------------------------------------------------


def _class_multipliers(inst: Instance, reg: RegimeParams) -> np.ndarray:
    """m_cls of every region by its class (Q59); a class without a multiplier raises (no default)."""
    table: dict[str, float] = {}
    for name, m in reg.class_multiplier:
        if name in table:
            raise ValueError(f"(29): duplicate region class {name!r} in class_multiplier")
        if not (math.isfinite(m) and m >= 0.0):
            raise ValueError(f"(29): the class multiplier of {name!r} must be finite and >= 0, got {m}")
        table[name] = float(m)
    missing = sorted({c for c in inst.region_class if c not in table})
    if missing:
        raise ValueError(f"(29): region classes {missing} have no class multiplier m_cls")
    return np.array([table[c] for c in inst.region_class], dtype=float)


def _dyad_pairs(inst: Instance, reg: RegimeParams) -> list[tuple[int, int]]:
    """Region indices of every named dyad (Q59); both regions must exist and be distinct."""
    pairs = []
    for a, b in reg.dyads:
        for r in (a, b):
            if r not in inst.region_index:
                raise ValueError(f"(Q59): dyad region {r!r} is not an instance region")
        if a == b:
            raise ValueError(f"(Q59): a dyad needs two distinct regions, got ({a!r}, {b!r})")
        pairs.append((inst.region_index[a], inst.region_index[b]))
    return pairs


def _chain_uniforms(entropy: int, episode: int, keys: list[tuple[int, int]], ncol: int) -> np.ndarray:
    """(len(keys), ncol) open uniforms: one stream-0 Generator per chain key, its raw words mapped by to_open_unit."""
    out = np.empty((len(keys), ncol), dtype=float)
    for i, key in enumerate(keys):
        out[i] = to_open_unit(generator(entropy, episode, codes.STREAM_REGIME, key).bit_generator.random_raw(ncol))
    return out


def _latent_normals(stream: KeyedStream, n_units: int, ncol: int, which: int) -> np.ndarray:
    """(U, ncol) standard normals of stream 13, key (unit, j, which): one standard_normal() per key (27).

    The keys are built here (unit < U, j < ncol, which in {0, 1}), so they are not re-validated per draw. With
    ``seeds.KERNELS`` on, the block comes from the numba kernel ``omega._kernels.latent_block`` (imported here, on the
    first routed draw), bit for bit the loop below (Q95), which stays the reference path.
    """
    if seeds.KERNELS:
        from shockbench_flow.omega import _kernels

        block = _kernels.latent_block(stream.head, stream.stream, n_units, ncol, which)
        if block is not None:
            return block
    out = np.empty((n_units, ncol), dtype=float)
    for x in range(n_units):
        for j in range(ncol):
            out[x, j] = stream.generator((x, j, which)).standard_normal()
    return out


def read_only_paths(
    burn_in: int,
    T: int,
    z_c: np.ndarray,
    z_c_own: np.ndarray,
    z_p: np.ndarray,
    z_dyad: np.ndarray,
    X: np.ndarray,
    W: np.ndarray | None,
) -> RegimePaths:
    """``RegimePaths`` over the given arrays, each made read-only in place (``W`` may be None): the one freezing rule.

    The caller hands over its arrays: they are not copied, and none can be written afterwards.
    """
    arrays = (z_c, z_c_own, z_p, z_dyad, X) + (() if W is None else (W,))
    for a in arrays:
        a.flags.writeable = False
    return RegimePaths(burn_in=burn_in, T=T, z_c=z_c, z_c_own=z_c_own, z_p=z_p, z_dyad=z_dyad, X=X, W=W)


def sample_regimes(
    inst: Instance, params: GeneratorParams, entropy: int, episode: int, noise: bool = True
) -> RegimePaths:
    """The regime layers, dyad chains, latent risks and noises of one episode (streams 0 and 13).

    Args:
        inst: the instance (regions, their classes, chokepoints).
        params: the generator params (``regime``, ``latent_region``, ``latent_dyad``, ``latent_chokepoint``,
            ``burn_in``); nothing else is read, so the paths are the same at every rung (V2).
        entropy: E_split of (27).
        episode: the episode index n of (27).
        noise: draw the score noise W of (45) (stream 13, keys (unit, j, 1)); a caller that discards it (naive's F_Q
            replications, the V12 residual draws, the strata harms) passes False, which draws no W key and leaves W
            None. Every other path is the same either way (each key is its own stream, (27)).

    Returns:
        ``RegimePaths`` with B_burn + T + 1 columns, column j = week j - B_burn, arrays read-only.

    Raises:
        ValueError: on invalid tension parameters (28), a region class without a multiplier, an invalid dyad, invalid
            latent parameters, a negative burn-in, or an entropy or episode outside the seed rule's bounds (27).
        TypeError: if the entropy or the episode is not an integer.

    """
    check_entropy(entropy)
    check_episode(episode)
    if isinstance(params.burn_in, bool) or not isinstance(params.burn_in, (int, np.integer)):
        raise TypeError(f"(38): the burn-in must be an integer number of weeks, got {params.burn_in!r}")
    B, T = int(params.burn_in), int(inst.T)
    if B < 0:
        raise ValueError(f"(38): the burn-in must be >= 0 weeks, got {B}")
    ncol = B + T + 1
    reg = params.regime
    tension_entry(reg)  # validate before drawing
    lats = {"region": params.latent_region, "dyad": params.latent_dyad, "chokepoint": params.latent_chokepoint}
    for kind, lat in lats.items():
        _check_latent(lat, kind)
    m_cls = _class_multipliers(inst, reg)
    pairs = _dyad_pairs(inst, reg)
    Pi = weekly_conflict_matrix(reg.P_yr)
    units = signal_units(inst, params)
    R, D, U = len(inst.regions), len(pairs), len(units)

    rho = np.array([lats[kind].rho for kind, _ in units], dtype=float)
    latent = KeyedStream(entropy, episode, codes.STREAM_LATENT)
    X = ar1_path(_latent_normals(latent, U, ncol, 0), rho)  # (29)
    W = ar1_path(_latent_normals(latent, U, ncol, 1), rho) if noise else None  # (45)

    u_conflict = _chain_uniforms(entropy, episode, [(LAYER_CONFLICT, m) for m in range(R)], ncol)
    u_dyad = _chain_uniforms(entropy, episode, [(LAYER_DYAD, d) for d in range(D)], ncol)
    u_tension = _chain_uniforms(entropy, episode, [(LAYER_TENSION, m) for m in range(R)], ncol)
    z_own = conflict_paths(u_conflict, X[:R], m_cls, Pi, params.latent_region)
    z_dyad = conflict_paths(u_dyad, X[R : R + D], 1.0, Pi, params.latent_dyad)
    z_c = effective_conflict(z_own, z_dyad, pairs)
    z_p = tension_paths(u_tension, z_c, reg)
    return read_only_paths(B, T, z_c, z_own, z_p, z_dyad, X, W)

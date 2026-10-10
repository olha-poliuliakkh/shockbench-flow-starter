"""Demand path (21) from stream 5, with the stationary start and the forecast parts of (48) (design §3.6, §5.3; Q85).

ln d^t = ln d-bar + ln m^sea(t) + eps^t + ln Xi(z^c,t of the sink's region); eps^t = phi eps^{t-1} + sigma eps^{d,t};
eps^0 = sigma eps^{d,0} / sqrt(1 - phi^2) (stationary law, §2.4); eps^{d,s} = sum_j sqrt(w^f_j) eps^{(j),s} with the
standard-normal parts eps^{(j),s} drawn from stream 5 at key (sink node, commodity, week + B_burn, part j), j = 0..8.
Computed as d = d-bar * m^sea(t) * exp(eps^t) * Xi so that zero noise gives d = d-bar exactly.
"""

import math
from typing import Sequence

import numpy as np

from shockbench_flow.instance.schema import Demand, Instance
from shockbench_flow.omega import codes
from shockbench_flow.omega.seeds import check_entropy, check_episode, generator


N_PARTS = 9  # forecast parts j = 0..8 of (48)


def eps_path(parts: np.ndarray, shares: Sequence[float], phi: float, sigma: float) -> np.ndarray:
    """The AR(1) noise eps^s, s = 0..T, of (21) from one demand's parts (T + 1, 9), stationary start included.

    eps^{d,s} = sum_j sqrt(w^f_j) eps^{(j),s} (48), summed with ``math.fsum`` so it is correctly rounded and
    independent of the order of the parts; eps^0 = sigma eps^{d,0} / sqrt(1 - phi^2) (§2.4); eps^s = phi eps^{s-1} +
    sigma eps^{d,s} (21).

    Raises:
        ValueError: if ``|phi| >= 1`` (no stationary law) or the parts do not have 9 columns.

    """
    p = np.asarray(parts, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != N_PARTS or len(shares) != N_PARTS:
        raise ValueError(f"(48) needs {N_PARTS} forecast parts per week, got parts {p.shape}, {len(shares)} shares")
    if not abs(phi) < 1.0:
        raise ValueError(f"(21) stationary start needs |phi| < 1, got {phi}")
    roots = [math.sqrt(w) for w in shares]
    eps = np.empty(p.shape[0], dtype=np.float64)
    prev = 0.0
    for s, row in enumerate(p.tolist()):
        eps_d = math.fsum(r * x for r, x in zip(roots, row))  # (48)
        prev = sigma * eps_d / math.sqrt(1.0 - phi * phi) if s == 0 else phi * prev + sigma * eps_d  # (21), §2.4
        eps[s] = prev
    return eps


def _seasonal(dem: Demand, T: int) -> np.ndarray:
    """m^sea(t) for t = 1..T (``Demand.m_sea``; the loader checks that a profile has 52 values)."""
    return np.array([dem.m_sea(t) for t in range(1, T + 1)], dtype=np.float64)


def _shock(inst: Instance, dem: Demand, burn_in: int, z_c: np.ndarray | None) -> np.ndarray:
    """Xi(z^{c,t}_{m(i)}) for t = 1..T from the conflict layer (index 0 = week -burn_in), or ones without a layer."""
    T = inst.T
    if z_c is None:
        return np.ones(T)
    z = np.asarray(z_c)
    if z.ndim != 2 or z.shape[0] != len(inst.regions) or z.shape[1] < burn_in + T + 1:
        raise ValueError(f"z_c must be (R={len(inst.regions)}, >= {burn_in + T + 1}) from week -burn_in, got {z.shape}")
    row = z[inst.nodes[dem.node].region, burn_in + 1 : burn_in + T + 1].astype(np.int64)
    if row.min(initial=0) < 0 or row.max(initial=0) >= len(codes.CONFLICT_STATES):
        raise ValueError("z_c holds a code outside the conflict states (0 none, 1 minor, 2 war)")
    return np.array([dem.shock[c] for c in row.tolist()], dtype=np.float64)


def demand_from_parts(
    inst: Instance, eps_parts: np.ndarray, burn_in: int = 0, z_c: np.ndarray | None = None
) -> np.ndarray:
    """Demand ``d`` (D, T) of (21) from stored parts (D, T + 1, 9): d = d-bar * m^sea(t) * exp(eps^t) * Xi.

    eps^t comes from ``eps_path``; a demand whose parts are all zero gets exp(eps) = 1 exactly, so the noise-free path
    is d-bar bit for bit. ``event_free`` calls this with ``z_c=None`` to drop the shock Xi (57).

    Raises:
        ValueError: on parts of the wrong shape, a malformed ``z_c``, or ``|phi| >= 1`` with noise.

    """
    T, D = inst.T, len(inst.demands)
    parts = np.asarray(eps_parts, dtype=np.float64)
    if parts.shape != (D, T + 1, N_PARTS):
        raise ValueError(f"eps_parts must be (D={D}, T+1={T + 1}, {N_PARTS}), got {parts.shape}")
    d = np.empty((D, T), dtype=np.float64)
    for di, dem in enumerate(inst.demands):
        if parts[di].any():
            growth = np.exp(eps_path(parts[di], inst.params.forecast_shares, dem.phi, dem.sigma)[1:])
        else:
            growth = np.ones(T)  # exp(0): d = d-bar exactly
        d[di] = dem.dbar * _seasonal(dem, T) * growth * _shock(inst, dem, burn_in, z_c)  # (21)
    return d


def demand_path(
    inst: Instance,
    entropy: int,
    episode: int,
    burn_in: int = 0,
    z_c: np.ndarray | None = None,
    noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Demand ``d`` (D, T) and the parts ``eps_parts`` (D, T + 1, 9), week 0 included, of every instance demand.

    Part j of week s (s = 0..T) is ``generator(entropy, episode, 5, (sink node, commodity, s + burn_in, j))
    .standard_normal()``, one key per part (§4.1 stream 5). Column t-1 of ``d`` is week t = 1..T.

    Args:
        inst: the instance (its ``demands`` in order).
        entropy: E_split of (27).
        episode: the episode index n of (27).
        burn_in: B_burn, the week-key offset.
        z_c: conflict layer (R, weeks) from week -burn_in, or None for Xi = 1 (no regime path yet).
        noise: False gives the noise-free path of the hand-solved fixture (sigma^d treated as 0, parts all zero).

    Raises:
        ValueError: on an entropy outside [0, 2**128) or an episode outside [0, 2**32), with or without noise (27), a
            malformed ``z_c``, or ``|phi| >= 1`` with noise.
        TypeError: if the entropy or the episode is not an integer (a bool included).

    """
    check_entropy(entropy)  # (27): checked even when the noise-free path draws no key (DC-6)
    check_episode(episode)
    T, D = inst.T, len(inst.demands)
    parts = np.zeros((D, T + 1, N_PARTS), dtype=np.float64)
    if noise:
        for di, dem in enumerate(inst.demands):
            for s in range(T + 1):
                for j in range(N_PARTS):
                    key = (dem.node, dem.k, s + burn_in, j)
                    parts[di, s, j] = generator(entropy, episode, codes.STREAM_DEMAND, key).standard_normal()
    return demand_from_parts(inst, parts, burn_in, z_c), parts

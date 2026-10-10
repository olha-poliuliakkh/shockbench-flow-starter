"""The 8-week demand forecast (48) and its wire field (design §5.3, §9.2; Q51 X6, Q55).

ln d-hat^{t+h}_ik = ln d-bar_ik + ln m^sea_ik(t+h) + phi^{h+1} eps^{t-1}_ik
    + sigma sum_{s=t}^{t+h} phi^{t+h-s} sum_{j >= max(1, s-t)} sqrt(w^f_j) eps^{(j),s}_ik
    + ln sum_z [(Pi^c)^{h+1}]_{z^{c,t-1}_{m(i)}, z} Xi_ik(z),        h = 0..7 (48),
shown as its exponential, from the demand noise only: eps^{t-1} by ``omega.demand.eps_path`` from omega's
``eps_parts`` and ``inst.params.forecast_shares``; part j of week s is observable from the instant s - 1 - j <= t - 1.
Readings of design §12 (M3 rows): the Xi term reads the *effective* z^{c,t-1} (column t - 1 + B_burn) of the sink's
region under the untilted Pi^c of P^yr (omega's ``meta_information_params``), the literal (48): the exact conditional
expectation under dyads would publish the hidden own chain. The Xi term is on when theta's ``forecast_shock`` (§11 row
70: whether last week's conflict state enters the forecast's shock factor; yes by default, SYNTHETIC(placeholder)) and
the meta's ``xi`` (whether d carries Xi) are both true: it is 0 for an injected omega (no layer, no meta) and for the
event-free twin (57) (``InformationMeta.xi`` False), matching their d; entries run only to t + h <= T (NaN beyond).
No future layer value, event, X or own chain enters it (X6; threats row 14). Clairvoyant sees the realised d instead.
A generated omega without the meta shows no forecast (``view`` module docstring).
"""

import math

import numpy as np

from shockbench_flow.disruption.regime import weekly_conflict_matrix  # the one Pi^c of (28)
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import immutable_copy
from shockbench_flow.omega.demand import N_PARTS, eps_path


FORECAST_WEEKS = 8  # h = 0..7 (48)


def _shock_expectation(dem, P_yr, h_max: int) -> np.ndarray:
    """(3, h_max) E[Xi(z^{c,t+h}) | z^{c,t-1} = z] = sum_z' [(Pi^c)^{h+1}]_{z, z'} Xi(z') for h = 0..h_max - 1 (48)."""
    Pi = weekly_conflict_matrix(P_yr)
    xi = np.array(dem.shock, dtype=np.float64)
    out = np.empty((len(codes.CONFLICT_STATES), h_max))
    step = np.eye(len(codes.CONFLICT_STATES))
    for h in range(h_max):
        step = step @ Pi  # (Pi^c)^{h+1}
        out[:, h] = step @ xi
    return out


def mmfe_forecast(
    inst: Instance, eps_parts: np.ndarray, z_c: np.ndarray | None, burn_in: int, P_yr, xi: bool
) -> np.ndarray:
    """(T, D, 8) read-only forecast (48): row t - 1, demand i, column h is d-hat^{t+h}_i, NaN where t + h > T.

    ``z_c`` is omega's effective layer (None on an injected omega); ``xi`` is whether the Xi term is on, the view's
    ``theta.forecast_shock and meta.xi`` (False drops it: omega^0, or the §11 row 70 switch off). Shown as
    d-bar m^sea(t + h) exp(eps-hat^{t+h}) E[Xi | z^{c,t-1}], the product form of (21), so a noise-free demand without
    the Xi term forecasts d-bar m^sea exactly; eps-hat^{t+h} = phi^{h+1} eps^{t-1} + sigma sum_s phi^{t+h-s}
    sum_{j >= max(1, s - t)} sqrt(w^f_j) eps^{(j),s}, eps^{t-1} by ``omega.demand.eps_path`` (the realised path).

    Raises:
        ValueError: on ``eps_parts`` not (D, T + 1, 9), a malformed ``z_c``, or ``xi`` True without a layer.

    """
    T, D, H = inst.T, len(inst.demands), FORECAST_WEEKS
    parts = np.asarray(eps_parts, dtype=np.float64)
    if parts.shape != (D, T + 1, N_PARTS):
        raise ValueError(f"(48) needs eps_parts of shape (D={D}, T+1={T + 1}, {N_PARTS}), got {parts.shape}")
    if xi:
        if z_c is None:
            raise ValueError("(48): the Xi term needs the conflict layer z_c (an injected omega has none)")
        z = np.asarray(z_c)
        if z.ndim != 2 or z.shape[0] != len(inst.regions) or z.shape[1] < burn_in + T + 1:
            raise ValueError(f"(48): z_c must be (R={len(inst.regions)}, >= {burn_in + T + 1}), got {z.shape}")
        if z.size and (z.min() < 0 or z.max() >= len(codes.CONFLICT_STATES)):
            raise ValueError("(48): z_c holds a code outside the conflict states")
    roots = [math.sqrt(w) for w in inst.params.forecast_shares]
    out = np.full((T, D, H), np.nan, dtype=np.float64)
    for di, dem in enumerate(inst.demands):
        p = parts[di]
        eps = eps_path(p, inst.params.forecast_shares, dem.phi, dem.sigma) if p.any() else np.zeros(T + 1)
        # tail[s][c] = sum_{j >= c} sqrt(w^f_j) eps^{(j),s}: the parts of week s observable c weeks ahead or more
        tail = [[math.fsum(r * x for r, x in zip(roots[c:], row[c:])) for c in range(N_PARTS + 1)] for row in p]
        m_sea = [dem.m_sea(w) for w in range(1, T + 1)]
        shock = _shock_expectation(dem, P_yr, H) if xi else None
        region = inst.nodes[dem.node].region
        phi, sig = dem.phi, dem.sigma
        for t in range(1, T + 1):
            z_prev = int(z_c[region, burn_in + t - 1]) if xi else 0  # z^{c,t-1}: the effective layer of week t - 1
            for h in range(min(H, T - t + 1)):
                known = math.fsum(phi ** (t + h - s) * tail[s][max(1, s - t)] for s in range(t, t + h + 1))
                eps_hat = phi ** (h + 1) * eps[t - 1] + sig * known  # (48)
                d_hat = dem.dbar * m_sea[t + h - 1] * math.exp(eps_hat)
                out[t - 1, di, h] = d_hat * shock[z_prev, h] if xi else d_hat
    return immutable_copy(out)


def exact_forecast(inst: Instance, d: np.ndarray) -> np.ndarray:
    """(T, D, 8) read-only: the realised d^{t+h} of omega's ``d`` (clairvoyant, §5.1), NaN where t + h > T."""
    T, D, H = inst.T, len(inst.demands), FORECAST_WEEKS
    d = np.asarray(d, dtype=np.float64)
    if d.shape != (D, T):
        raise ValueError(f"the exact forecast needs omega's d of shape (D={D}, T={T}), got {d.shape}")
    out = np.full((T, D, H), np.nan, dtype=np.float64)
    for t in range(1, T + 1):
        for h in range(min(H, T - t + 1)):
            out[t - 1, :, h] = d[:, t + h - 1]
    return immutable_copy(out)


def forecast_field(inst: Instance, table: np.ndarray, t: int) -> dict | None:
    """The §9.2 ``demand_forecast`` value of week t.

    Columns {node, k, h, qty} per (demand, h) with t + h <= T, demands in instance order, h ascending, qty Python
    floats; None after T (1).
    """
    T = table.shape[0]
    if not 1 <= t <= T:
        return None
    hs = range(min(FORECAST_WEEKS, T - t + 1))
    rows = [(d.node, d.k, h, float(table[t - 1, di, h])) for di, d in enumerate(inst.demands) for h in hs]
    return {name: [r[i] for r in rows] for i, name in enumerate(("node", "k", "h", "qty"))}

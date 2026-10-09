"""Step 7 of design §3.2: energy allocation (15)-(18) and OSAT packaging (19) as pure functions (Q27, Q57, Q58 M6).

The simulator (``sim.py``) applies them in the §3.2 order: supply lift, grids (segments, rationing on last week's gas
stock, fuel on hand after steps 5-6, allocation by ``pri_g``, pro-rata segment loading), fab lots with gross WIP, scrap
booked in onset weeks, fab output, OSAT output, then packaging. Divisions by zero follow 0/0 := 0 (§3.5).
"""

from collections.abc import Sequence


def allocate_energy(priority: str, g_av: float, y_bar: float, e_hat: Sequence[float]) -> tuple[float, list[float]]:
    """(18): served base load y and energy E_f per fab of one grid under ``pri_g``.

    Args:
        priority: ``base_first``, ``proportional`` or ``industrial_first`` (Q57).
        g_av: available output G^av_g, the sum over segments after rationing and fuel on hand.
        y_bar: base load y-bar^t_g.
        e_hat: requested draw E-hat_f = e_f p-hat_f / R_f per fab of the grid, in fab order.

    Returns:
        (y, [E_f]); the unserved base load y-bar - y is shed at VOLL (17).

    """
    tot = sum(e_hat)
    if priority == "base_first":
        y = min(y_bar, g_av)
        E = [(eh * min(1.0, (g_av - y) / tot) if tot > 0 else 0.0) for eh in e_hat]
    elif priority == "proportional":
        den = y_bar + tot
        eta = min(1.0, g_av / den) if den > 0 else 0.0
        y = eta * y_bar
        E = [eta * eh for eh in e_hat]
    elif priority == "industrial_first":
        E = [(eh * min(1.0, g_av / tot) if tot > 0 else 0.0) for eh in e_hat]
        y = min(y_bar, max(0.0, g_av - sum(E)))
    else:
        raise ValueError(f"unknown energy priority {priority!r}")
    return y, E


def package(raw: Sequence[float], thr: float) -> list[float]:
    """(19): OSAT starts xi_ik from raw-chip stock, up to the throughput thr_i, pro rata across packaged commodities.

    With one packaged commodity this is ``min(thr, raw)`` exactly, the reference's rule.
    """
    tot = sum(raw)
    if tot <= thr:
        return [float(r) for r in raw]
    return [thr * (r / tot) for r in raw]

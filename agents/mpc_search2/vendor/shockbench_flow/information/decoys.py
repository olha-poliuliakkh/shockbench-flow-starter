"""Thinning of the shadow process into decoys, per regime (design §5.3; (49); Q51 X8, Q89, Q97).

omega holds the decoy threads of one shadow cluster process Y, whatever the regime (``disruption.announce``, stream
7; (49) as Q101 amends it): shadows excite shadow children as real events excite real ones, and each decoy-bearing
channel (tariff_formal, tariff_informal, ties_threat, mid_threat) carries phi-bar/(1 - phi-bar) decoy threads per real
thread in expectation. A regime shows shadow j iff U_j < r_theta,ch = [phi_ch(theta)/(1 - phi_ch(theta))] /
[phi-bar_ch/(1 - phi-bar_ch)] (49), ch = ``sh_channel[j]``, so the shown sets are nested in phi (common random numbers
across the dial, R5.11) and the shown decoy share of a channel is phi_ch(theta) in expectation (V12). Clairvoyant
(phi = 0) shows no decoy; at phi = phi-bar the ratio is exactly 1.0. One U_j thins the whole thread (Q97: a later
message on a thread tells nothing about whether it is real), and the U_j of one cluster's threads of a type are one
uniform, so a regime shows or hides them together.
"""

from collections.abc import Mapping

import numpy as np

from shockbench_flow.information.theta import Theta, phi_by_channel
from shockbench_flow.omega import codes
from shockbench_flow.omega.container import Omega


def show_ratio(phi: float, phi_bar: float) -> float:
    """The threshold r of (49) for one channel: (phi/(1 - phi)) / (phi_bar/(1 - phi_bar)).

    Returns 0.0 exactly at phi == 0 and 1.0 exactly at phi == phi_bar.

    Raises:
        ValueError: unless 0 <= phi <= phi_bar < 1 and phi_bar > 0 (a phase-4 flood at phi > phi-bar needs a larger
            release-level phi-bar, which redraws Y).

    """
    phi, phi_bar = float(phi), float(phi_bar)
    if not (0.0 < phi_bar < 1.0 and 0.0 <= phi <= phi_bar):  # NaN fails too
        raise ValueError(f"(49) needs 0 <= phi <= phi-bar < 1 and phi-bar > 0, got phi {phi!r}, phi-bar {phi_bar!r}")
    if phi == 0.0:
        return 0.0
    if phi == phi_bar:
        return 1.0
    return (phi / (1.0 - phi)) / (phi_bar / (1.0 - phi_bar))


def shown_mask(omega: Omega, theta: Theta, phi_bar: Mapping[str, float]) -> np.ndarray:
    """(n_sh,) bool: shadow j is shown under ``theta`` iff ``U_decoy[j] < show_ratio(phi_ch(theta), phi_bar[ch])``.

    ``phi_bar`` is omega's (``omega.assembly.InformationMeta.phi_bar`` by channel name); all False when theta has no
    messages feed or phi = 0. Whether each announcement of a shown thread appears (lead > 0) is ``messages.shown``'s.

    Raises:
        ValueError: on a shadow channel outside the decoy-bearing ones, or ``show_ratio``'s refusals.

    """
    channels = np.asarray(omega["sh_channel"], dtype=np.int64)
    u = np.asarray(omega["U_decoy"], dtype=np.float64)
    if channels.shape != u.shape:
        raise ValueError(f"sh_channel {channels.shape} and U_decoy {u.shape} must hold one entry per shadow")
    decoy_codes = [codes.CHANNELS.index(ch) for ch in codes.DECOY_CHANNELS]
    if channels.size and not np.isin(channels, decoy_codes).all():
        raise ValueError(
            f"a shadow channel outside the decoy-bearing ones {codes.DECOY_CHANNELS}: {sorted(set(channels))}"
        )
    phi = phi_by_channel(theta)
    out = np.zeros(u.shape, dtype=bool)
    if not phi:  # no messages feed: nothing is shown
        return out
    for ch, value in phi.items():
        if ch not in phi_bar:
            raise ValueError(f"omega states no phi-bar for {ch}, which theta's phi needs (49)")
        ratio = show_ratio(value, phi_bar[ch])
        mine = channels == codes.CHANNELS.index(ch)
        out[mine] = u[mine] < ratio  # (49): the same U_j at every phi, so the shown sets nest (R5.11)
    return out

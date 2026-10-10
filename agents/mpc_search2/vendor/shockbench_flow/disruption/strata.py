"""Harm-quantile strata (44) (design §4.5; Q61, Q88).

s(omega) = 1 + #{j in (0.50, 0.80, 0.95): H(omega) > h_j}; the cut points h_j are the j-quantiles of H under the
public generator per (size, rung), from M event-list draws (design M = 1e5; no LP, no demand needed); weights p_s =
(0.50, 0.30, 0.15, 0.05). Cut-point draws use their own entropy root and episode indices 0..M-1, independent of any
split.

Readings where the design leaves a choice (fixed here, reported for design §12):

- **Quantile rule**: h_j is the empirical inverse CDF of the M draws, the smallest draw x with F_M(x) >= j, i.e. the
  ceil(j M)-th smallest, with j M evaluated exactly on the decimal ``repr`` of j; naive's F^{-1} of (67) uses the same
  rule (``policies.naive_fq.quantile``). Without ties at a cut exactly ceil(j M) draws lie at or below h_j, so the
  realised shares on the cut draws are p_s exactly when j M is an integer; a tie at a cut (an atom of H, e.g. episodes
  with no active event) puts the whole atom in the lower stratum, since s counts H > h_j strictly.
- **Draw n**: H of ``sampler.sample_omega(inst, params, entropy, n)`` by ``harm.episode_harm`` under the params'
  ``MarkParams``. Harm reads omega's ``ev_*`` arrays and its instance hash and ``meta_mark_params`` only, so the draw
  builds just those (``events.event_arrays`` of ``sampler.sample_events(..., noise=False).stored``): no demand, no W,
  no stored marks, the same H.
- **Domain**: H is a sum of non-negative USD terms (41), so a draw or a harm that is NaN, infinite or negative raises.
- **Whose cut points**: h_j is per (size, rung), so ``stratum`` and ``realised_shares`` take the generator id of the
  omega whose harm they stratify (omega's ``meta_generator_id``, ``params.generator_id(params, inst)`` of the
  generator that drew it) and refuse cut points of another generator (``CutPoints.generator_id``): read against the
  0.62 cut points, six 0.97 omegas of `tiny` land in strata 3-4 instead of 1-2 (cut points from 40 draws, the red
  team's probe; INT-M2-R2-2).
"""

import functools
import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from shockbench_flow.disruption.params import GeneratorParams, generator_id
from shockbench_flow.instance.schema import Instance
from shockbench_flow.omega.seeds import check_entropy
from shockbench_flow.parallel import ordered_map
from shockbench_flow.scoring.rss import STRATUM_WEIGHTS


QUANTILES = (0.50, 0.80, 0.95)  # SYNTHETIC cut points (Q61, §11 row 28)
WEIGHTS = STRATUM_WEIGHTS  # p_s of (44), fixed by construction at every size and rung (Q61); one definition in rss


@dataclass(frozen=True)
class CutPoints:
    quantiles: tuple[float, ...]
    values: tuple[float, ...]  # h_j, USD
    draws: int  # M
    generator_id: str  # ``params.generator_id(params, inst)`` of the draws' (size, rung); ``stratum`` checks it


def inverse_cdf(sorted_values: np.ndarray, p: float) -> float:
    """The smallest sample x with F(x) >= p: the ceil(p n)-th smallest of n sorted values (p n exact on repr(p)).

    Raises:
        ValueError: on an empty sample.

    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("the quantile of an empty sample is undefined")
    i = math.ceil(Fraction(repr(float(p))) * n) - 1
    return float(sorted_values[min(max(i, 0), n - 1)])


def _check_harm(value: float) -> float:
    h = float(value)
    if not (math.isfinite(h) and h >= 0.0):
        raise ValueError(f"a harm (41)-(42) is a finite USD amount >= 0, got {value!r}")
    return h


def _generated_harm(inst: Instance, params: GeneratorParams, entropy: int, episode: int) -> float:
    """H(omega) of (42) for draw ``episode`` of the public generator on ``entropy``.

    From the arrays of omega that harm reads (module docstring): the stored events as ``sample_omega`` writes them
    (``events.event_arrays``), omega's ``meta_instance_hash`` and ``meta_mark_params``.
    """
    from shockbench_flow.disruption.events import event_arrays
    from shockbench_flow.disruption.harm import episode_harm
    from shockbench_flow.disruption.sampler import sample_events
    from shockbench_flow.marks import mark_params_json
    from shockbench_flow.omega.container import ARRAY_DTYPES

    s = sample_events(inst, params, entropy, episode, noise=False)
    arrays = event_arrays(inst, list(s.stored), params.burn_in)
    arrays["meta_instance_hash"] = np.array(inst.hash, dtype=ARRAY_DTYPES["meta_instance_hash"])
    arrays["meta_mark_params"] = np.array(mark_params_json(params.marks), dtype=ARRAY_DTYPES["meta_mark_params"])
    return episode_harm(inst, arrays, params.marks)


def _harm(inst: Instance, params: GeneratorParams, entropy: int, harm_of: Callable, episode: int) -> float:
    return _check_harm(harm_of(inst, params, entropy, episode))


def episode_harms(
    inst: Instance,
    params: GeneratorParams,
    draws: int,
    entropy: int,
    n_jobs: int = 1,
    harm_of: Callable | None = None,
) -> np.ndarray:
    """H(omega_n) of (42) for the episodes n = 0..draws-1 on ``entropy``, in episode order.

    Args:
        inst: the instance.
        params: the public generator at the (size, rung).
        draws: M.
        entropy: the cut points' own root (27), never a split's.
        n_jobs: joblib workers over chunks of episodes (imported only when not 1); the result does not depend on it.
        harm_of: ``(inst, params, entropy, episode) -> H``; default the generator's omega and ``harm.episode_harm``.

    Raises:
        TypeError: if ``draws`` is not an integer (a bool included) or the entropy not an integer.
        ValueError: if ``draws`` < 1, the entropy is outside [0, 2**128) or a harm is not a finite number >= 0.

    """
    if isinstance(draws, bool) or not isinstance(draws, numbers.Integral):
        raise TypeError(f"draws must be an integer, got {draws!r}")
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    entropy = check_entropy(entropy)
    harm_of = _generated_harm if harm_of is None else harm_of
    values = ordered_map(functools.partial(_harm, inst, params, entropy, harm_of), range(draws), n_jobs)
    return np.array(values, dtype=np.float64)


def cut_points(
    inst: Instance,
    params: GeneratorParams,
    draws: int,
    entropy: int,
    n_jobs: int = 1,
    harm_of: Callable | None = None,
) -> CutPoints:
    """The cut points of (44) from ``draws`` event-list draws (episodes 0..draws-1 on ``entropy``).

    ``harm_of`` replaces the generator's harm (``episode_harms``); the cut points are ``inverse_cdf`` of the sorted
    harms at ``QUANTILES``.
    """
    harms = np.sort(episode_harms(inst, params, draws, entropy, n_jobs, harm_of), kind="stable")
    values = tuple(inverse_cdf(harms, j) for j in QUANTILES)
    return CutPoints(QUANTILES, values, int(draws), generator_id(params, inst))


def _check_generator(cuts: CutPoints, generator_id: str) -> None:
    """Refuse cut points drawn under another generator than the harm's (h_j is per (size, rung), (44))."""
    if not isinstance(generator_id, str):
        raise TypeError(f"the harm's generator id must be a string (omega's meta_generator_id), got {generator_id!r}")
    if generator_id != cuts.generator_id:
        raise ValueError(
            f"(44) cuts per (size, rung): cut points of generator {cuts.generator_id!r} cannot stratify a harm of"
            f" generator {generator_id!r}"
        )


def stratum(harm: float, cuts: CutPoints, *, generator_id: str) -> int:
    """s(omega) of (44), 1..4, for a harm of the generator ``generator_id`` (omega's ``meta_generator_id``).

    Raises:
        ValueError: if the harm is not a finite number >= 0, or ``generator_id`` is not ``cuts.generator_id``.
        TypeError: if ``generator_id`` is not a string.

    """
    _check_generator(cuts, generator_id)
    h = _check_harm(harm)
    return 1 + sum(h > c for c in cuts.values)


def realised_shares(harms, cuts: CutPoints, *, generator_id: str) -> tuple[float, ...]:
    """The share of ``harms`` in each stratum 1..len(cuts.values) + 1 (V29 compares them with ``WEIGHTS``).

    ``harms`` are harms of the generator ``generator_id``; errors as ``stratum``, and a ValueError on no harms.
    """
    _check_generator(cuts, generator_id)
    s = [stratum(h, cuts, generator_id=generator_id) for h in harms]
    if not s:
        raise ValueError("no harms to share out")
    counts = np.bincount(s, minlength=len(cuts.values) + 2)[1:]
    return tuple(float(c) / len(s) for c in counts)

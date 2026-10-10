"""The package's one parallel pattern: an ordered map over joblib workers (design §4.1: every draw is keyed).

Every parallel loop of the package (naive's F_Q replications, the strata harms, the realism measures, the V12 gate's
draws, the evaluation runner's episodes) maps a function of one episode or replication index over a range; the draws
are keyed by (27), so the results do not depend on the worker count or the batching, and ``ordered_map`` returns them
in input order. joblib is imported only when ``n_jobs`` is not 1, so a serial run never loads it (Q94 (c) admits it in
the package). The batching is joblib's own (``Parallel(batch_size=BATCH_SIZE)``, Q95; design §12 rows "joblib
``batch_size``" and "``batch_size`` measured"), chosen by measurement (``scripts/python/evidence/m4_conf_timings.py``,
part B).
"""

from collections.abc import Callable, Iterable


# joblib's batch size (Q95): "auto" lets joblib grow a batch while one takes less than about 0.2 s of worker time.
# Measured on 8 workers (docs/evidence/m4_conf_timings_8_B.txt; design §12 row "batch_size measured"): the package's
# maps, whose items take 0.17-0.63 s, run within 1.6 % of each other under "auto", 1, 4, 16 and the chunks of
# ceil(len / (4 x workers)) used before M4, and 20,000 microsecond items take 0.13 s under "auto", 5.7 s at size 1.
BATCH_SIZE: int | str = "auto"


def ordered_map(fn: Callable, items: Iterable, n_jobs: int = 1) -> list:
    """``[fn(x) for x in items]``, over joblib workers when ``n_jobs`` != 1 (imported only then), in input order.

    Serial when ``n_jobs`` is 1 or there is at most one item. Otherwise one ``delayed(fn)(x)`` per item goes to
    ``joblib.Parallel(n_jobs=n_jobs, batch_size=BATCH_SIZE)``, which returns the results in input order whatever the
    order in which the workers finish. ``fn`` must pickle (a module-level function, or a ``functools.partial`` of one).
    """
    xs = list(items)
    if n_jobs == 1 or len(xs) <= 1:
        return [fn(x) for x in xs]
    from joblib import Parallel, delayed

    return list(Parallel(n_jobs=n_jobs, batch_size=BATCH_SIZE)(delayed(fn)(x) for x in xs))

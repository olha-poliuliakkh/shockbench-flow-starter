"""Week templates for the time-expanded LP (51): one week's columns and rows, replicated over t = 1..T.

A column template is a key without its week; the column of week t sits at ``(t - 1) * nc + j`` (week-major layout).
A row entry ``(r, j, lag, coef)`` puts ``coef`` (a scalar, or an array over the row's week t) on the column of week
``t - lag`` in the row of week t; an entry whose column week would be < 1 is dropped, since the initial state it stands
for enters that row's right-hand side instead. Row names get their week inserted after the tag, like column keys
(``with_week``).
"""

from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from sbfplan.dynamics.state import COST_COMPONENTS


def with_week(templates: Sequence[tuple], i: int) -> tuple:
    """The key (or row name) at week-major index ``i``: template ``i % n``, week ``i // n + 1`` after the tag."""
    n = len(templates)
    key = templates[i % n]
    return (key[0], i // n + 1, *key[1:])


def all_with_week(templates: Sequence[tuple], T: int) -> tuple[tuple, ...]:
    """Every key (or row name) of weeks 1..T, week-major."""
    return tuple((key[0], t, *key[1:]) for t in range(1, T + 1) for key in templates)


class Columns:
    """Column templates of one week: bounds, cost terms by component of (23), and salvage coefficients."""

    def __init__(self, T: int) -> None:
        self.T = T
        self.keys: list[tuple] = []
        self.ub: list = []
        self.salvage: list = []
        self.parts: dict[str, list] = {p: [] for p in COST_COMPONENTS}

    def add(self, key: tuple, ub=np.inf, salvage=0.0, **costs) -> int:
        """Add one template; ``costs`` maps components of ``COST_COMPONENTS`` to USD per unit (scalar or (T,))."""
        if unknown := set(costs) - set(COST_COMPONENTS):
            raise KeyError(f"unknown cost components {unknown}")
        self.keys.append(key)
        self.ub.append(ub)
        self.salvage.append(salvage)
        for p in COST_COMPONENTS:
            self.parts[p].append(costs.get(p, 0.0))
        return len(self.keys) - 1

    def stack(self, values: list) -> np.ndarray:
        """Flatten per-template scalars or (T,) arrays into the week-major column vector."""
        out = np.empty((self.T, len(values)))
        for j, v in enumerate(values):
            out[:, j] = v
        return out.ravel()


class Rows:
    """Row templates of one week (all equalities, or all <= rows), with week-indexed right-hand sides."""

    def __init__(self, T: int) -> None:
        self.T = T
        self.names: list[tuple] = []
        self.rhs: list[np.ndarray] = []
        self._r: list[int] = []
        self._j: list[int] = []
        self._lag: list[int] = []
        self._coef: list = []

    def row(self, name: tuple, rhs=0.0) -> int:
        """Add a row template with right-hand side ``rhs`` (scalar or (T,)) and return its index."""
        self.names.append(name)
        self.rhs.append(np.broadcast_to(np.asarray(rhs, dtype=float), (self.T,)).copy())
        return len(self.names) - 1

    def add(self, r: int, j: int, coef, lag: int = 0) -> None:
        """Put ``coef`` on column template ``j`` of week ``t - lag`` in row template ``r`` of week t."""
        self._r.append(r)
        self._j.append(j)
        self._lag.append(lag)
        self._coef.append(coef)

    def add_rhs(self, r: int, t: int, value: float) -> None:
        """Add a constant (initial state) to the right-hand side of row ``r`` in week ``t`` (ignored beyond T)."""
        if 1 <= t <= self.T:
            self.rhs[r][t - 1] += value

    def build(self, nc: int) -> tuple[csr_matrix, np.ndarray]:
        """Replicate over the weeks: the sparse matrix and the right-hand side, week-major."""
        T, nr = self.T, len(self.names)
        b = np.array(self.rhs).T.ravel() if nr else np.zeros(0)
        if not self._r:
            return csr_matrix((T * nr, T * nc)), b
        r = np.array(self._r)[:, None]
        j = np.array(self._j)[:, None]
        lag = np.array(self._lag)[:, None]
        coef = np.empty((len(self._coef), T))
        for i, c in enumerate(self._coef):
            coef[i] = c
        t = np.arange(1, T + 1)[None, :]
        keep = t > lag
        rows = ((t - 1) * nr + r)[keep]
        cols = ((t - lag - 1) * nc + j)[keep]
        A = coo_matrix((coef[keep], (rows, cols)), shape=(T * nr, T * nc)).tocsr()
        A.eliminate_zeros()
        return A, b

"""CMA-ES over the constants a candidate declared (EVOLVE_DESIGN.md 4.5).

Each declared param has a range; CMA-ES searches the unit cube mapped onto them. All members of one generation play the
same stratified subset of the training pool (paired), and the subset rotates between generations. The best point is
kept only if it beats the untuned values on the whole training pool (paired gap > 0).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from sbf_starter.evolve.evaluate import paired_gap


def _write(folder: Path, values: dict) -> None:
    (folder / "params.json").write_text(json.dumps(values, sort_keys=True) + "\n")


def tune(
    evaluator,
    folder: Path,
    spec: list[dict],
    values: dict,
    subsets: list[str],
    train: str,
    work: Path,
    generations: int = 4,
    popsize: int = 8,
    seed: int = 0,
    log=print,
) -> tuple[dict, dict]:
    """Tune ``spec``'s params of the candidate in ``folder``; returns (values to keep, a summary)."""
    import cma

    if not spec:
        return values, {"tuned": False, "why": "no declared params"}
    lo = np.array([s["low"] for s in spec], dtype=float)
    hi = np.array([s["high"] for s in spec], dtype=float)
    x0 = np.clip(
        (np.array([values.get(s["name"], s["default"]) for s in spec]) - lo) / np.maximum(hi - lo, 1e-12), 0, 1
    )
    es = cma.CMAEvolutionStrategy(
        x0.tolist(), 0.3, {"popsize": popsize, "bounds": [0, 1], "seed": seed + 1, "verbose": -9}
    )
    work.mkdir(parents=True, exist_ok=True)
    best = (-np.inf, dict(values))
    for g in range(generations):
        subset = subsets[g % len(subsets)]
        xs = es.ask()
        scores = []
        for i, x in enumerate(xs):
            trial = dict(values, **{s["name"]: float(lo[j] + x[j] * (hi[j] - lo[j])) for j, s in enumerate(spec)})
            f = work / f"g{g}_{i}"
            if f.exists():
                shutil.rmtree(f)
            shutil.copytree(folder, f, ignore=shutil.ignore_patterns("__pycache__"))
            _write(f, trial)
            res = evaluator.play(f, subset)
            score = res.rss if res.rss is not None and res.fallback_weeks == 0 else -np.inf
            scores.append(score)
            if score > best[0]:
                best = (score, trial)
            shutil.rmtree(f)
        es.tell(xs, [-s if np.isfinite(s) else 1e9 for s in scores])
        log(f"  tune generation {g + 1}/{generations} on {subset}: best {max(scores):.4f}")
    # the winner against the untuned values on the whole training pool
    cand = work / "best"
    if cand.exists():
        shutil.rmtree(cand)
    shutil.copytree(folder, cand, ignore=shutil.ignore_patterns("__pycache__"))
    _write(cand, best[1])
    base, tuned = evaluator.play(folder, train), evaluator.play(cand, train)
    gap = paired_gap(evaluator.pools[train], tuned.J, base.J)
    keep = gap["diff"] > 0 and tuned.fallback_weeks == 0
    shutil.rmtree(cand)
    return (best[1] if keep else values), {"tuned": keep, "gap": gap, "values": best[1]}

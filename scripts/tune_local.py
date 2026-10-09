"""Tune the numeric params of an agent with Optuna, locally (no API calls), then write the winner to its params.json.

    uv sync --extra evolve
    uv run python scripts/tune_local.py                                  # 100 trials on agents/mine
    uv run python scripts/tune_local.py --n_trials=20 --per_level=4      # a quick run
    uv run python scripts/tune_local.py --resume=outputs/tune_local/<date_time>

The search space is the agent's ``params_spec.json`` ([{name, default, low, high}, ...]); integer bounds give an
integer param. Each trial plays a copy of the agent with its candidate params on a private, stratified training pool
(``--root``, ``--per_level`` episodes per harm level) through shockbench-flow's own scoring API, in-process and in
parallel: nothing is parsed from printed output, and the public dev split is never used, so the search cannot fit it.
The current params.json is queued as the first trial.

At the end the best trial is compared with the current params on a second private pool (``--holdout_root``), paired
on the same episodes. params.json is overwritten only if the best trial wins there (``--always_write`` overwrites
regardless). Nothing else in the agent folder changes; trial copies live under outputs/tune_local/<date_time>/.
"""

import json
import shutil
import time
import traceback
from pathlib import Path

import fire
import optuna

from sbf_starter.evolve import pools
from sbf_starter.evolve.evaluate import paired_gap


FAILED = -1.0  # a trial that crashed or handed weeks to naive: below any score worth keeping


def _is_int(spec: dict) -> bool:
    return all(isinstance(spec[k], int) and not isinstance(spec[k], bool) for k in ("default", "low", "high"))


def _play(es, agent: Path, values: dict, work: Path, workers: int) -> tuple[float, list[int], int]:
    """(RSS, per-episode costs, fallback weeks) of the agent with ``values`` as its params.json, on ``es``."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(agent, work, ignore=shutil.ignore_patterns("__pycache__"))
    (work / "params.json").write_text(json.dumps(values, indent=1, sort_keys=True) + "\n")
    try:
        rows = es.play(str(work), cpu_budget=True, n_jobs=workers)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    J = [int(r["J_policy_cents"]) for r in rows]
    return es.rss(J)["rss"], J, sum(r["fallback_weeks"] for r in rows)


def main(
    agent: str = "agents/mine",
    task: str = "small",
    n_trials: int = 100,
    root: int = 20261004,
    per_level: int = 8,
    holdout_root: int = 20261005,
    holdout_per_level: int = 16,
    workers: int = 8,
    seed: int = 0,
    always_write: bool = False,
    resume: str | None = None,
) -> None:
    """Search the params of ``params_spec.json``, confirm on held-out episodes, write params.json.

    Args:
        agent: the agent folder (params_spec.json and params.json beside agent.py).
        task: small (the public board's network) or full.
        n_trials: Optuna trials in this call (a resumed study adds to its earlier ones).
        root: the private root of the training pool (never 0, the public dev root).
        per_level: training episodes per harm level (4 levels; each trial plays all of them).
        holdout_root: a second private root for the final confirmation.
        holdout_per_level: held-out episodes per harm level.
        workers: processes that play episodes.
        seed: the TPE sampler's seed.
        always_write: write the best trial's params even if it does not win on the held-out pool.
        resume: a run folder whose study (study.db) to continue.

    """
    agent_dir = Path(agent)
    spec = json.loads((agent_dir / "params_spec.json").read_text())
    current = json.loads((agent_dir / "params.json").read_text()) if (agent_dir / "params.json").is_file() else {}
    current = {s["name"]: current.get(s["name"], s["default"]) for s in spec}
    run = Path(resume or f"outputs/tune_local/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    run.mkdir(parents=True, exist_ok=True)
    print(f"run folder {run}; {len(spec)} params; loading the training pool (built once, then cached)")
    train = pools.episode_set(task, root, pools.take(pools.stratified(task, root, [per_level] * 4)))

    def objective(trial: optuna.Trial) -> float:
        values = {
            s["name"]: (trial.suggest_int if _is_int(s) else trial.suggest_float)(s["name"], s["low"], s["high"])
            for s in spec
        }
        start = time.perf_counter()
        try:
            score, _J, fallback = _play(train, agent_dir, values, run / "trials" / f"t{trial.number}", workers)
        except Exception as err:  # an agent or evaluator fault fails this trial only
            (run / "errors").mkdir(exist_ok=True)
            (run / "errors" / f"t{trial.number}.txt").write_text(traceback.format_exc())
            print(f"trial {trial.number}: failed ({type(err).__name__}: {err})")
            return FAILED
        if score is None or fallback:
            print(f"trial {trial.number}: rejected (score {score}, fallback weeks {fallback})")
            return FAILED
        print(f"trial {trial.number}: RSS {score:.4f} ({time.perf_counter() - start:.0f} s)")
        return float(score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="tune_local",
        storage=f"sqlite:///{run / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    if not study.trials:
        study.enqueue_trial(current)  # the current params are trial 0: the bar to beat
    study.optimize(objective, n_trials=n_trials)

    best = {s["name"]: study.best_params[s["name"]] for s in spec}
    base = next((t.value for t in study.trials if t.number == 0), None)
    print(
        f"\nbest training RSS {study.best_value:.4f} (trial {study.best_trial.number}); current params "
        f"{base if base is None else round(base, 4)}"
    )
    print("confirming on the held-out pool (paired against the current params)")
    hold = pools.episode_set(
        task, holdout_root, pools.take(pools.stratified(task, holdout_root, [holdout_per_level] * 4))
    )
    s_best, J_best, f_best = _play(hold, agent_dir, best, run / "trials" / "holdout_best", workers)
    s_cur, J_cur, _ = _play(hold, agent_dir, current, run / "trials" / "holdout_current", workers)
    gap = paired_gap(hold, J_best, J_cur)
    print(
        f"held out: best {s_best:.4f}, current {s_cur:.4f}; gap {gap['diff']:+.4f} "
        f"(90 % paired interval {gap['lo']:+.4f} to {gap['hi']:+.4f})"
    )
    (run / "best_params.json").write_text(json.dumps(best, indent=1, sort_keys=True) + "\n")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "best_training_rss": study.best_value,
                "best_params": best,
                "holdout": {"best": s_best, "current": s_cur, **gap},
            },
            indent=1,
        )
    )
    if (gap["diff"] > 0 and not f_best) or always_write:
        (agent_dir / "params.json").write_text(json.dumps(best, indent=1, sort_keys=True) + "\n")
        print(f"written {agent_dir / 'params.json'}")
    else:
        print(
            f"params.json kept: the best trial does not beat the current params held out "
            f"(saved as {run / 'best_params.json'}; --always_write to write it anyway)"
        )
    print("best params:\n" + "\n".join(f"  {k} = {v}" for k, v in best.items()))
    print(f"next: uv run sbf check {agent} --task=small && uv run sbf check {agent} --task=full")


if __name__ == "__main__":
    fire.Fire(main)

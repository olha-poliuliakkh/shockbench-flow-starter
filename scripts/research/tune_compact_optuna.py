"""Optuna search of ``agents/compact_hierarchical``'s knobs on Small: 64 episodes of root 202610, in this process.

    uv sync --extra evolve                                                     # optuna
    uv run python scripts/research/tune_compact_optuna.py --trials=80 \
        --seed_params=outputs/research/tune_compact_optuna/compact_small/best_small_params.json   # pass 2
    uv run python scripts/research/tune_compact_optuna.py --space=pass1 --trials=50   # continue the first sweep
    uv run python scripts/research/tune_compact_optuna.py --smoke              # plumbing check: 2 episodes, no files

Each trial plays the agent with the trial's knobs (every other knob from ``config.DEFAULTS`` and ``params.json``,
never from ``SBF_PARAM_*`` variables) on the same Small episodes 0..63 of root 202610, serially, in this process:
``EpisodeSet.play`` with a factory building ``Agent(config, params=...)``, so no zip, subprocess or file per trial.
The objective, maximized, is the episodes' RSS.

**Spaces** (``SPACES``). ``pass1`` is the first sweep's. ``pass2`` (the default) widens the bounds pass 1's best sat
on: ``short_price`` from 0.01 (was 1), ``safety_price`` to 5 (was 0.5), ``safety_frac`` to 0.85, ``reroute_max_wait``
to 10. It also draws ``crisis_gap`` in [0, 1] in place of ``crisis_frac``, with
crisis_frac = safety_frac + crisis_gap x (0.95 - safety_frac), so a crisis floor is never below the static one.
A study keeps the space it started with: a pass-2 run uses its own study (``compact_small_pass2`` by default).

**Rejection.** A trial is rejected at the first episode chunk where the agent
- ran over 1.5 s of CPU in some week (the harness's meter, ``cpu_budget=1.5``: the naive rule plays that week), or
- fell back in some week: its LP had no optimum before ``deadline_small`` or raised (the agent's own log), or the
  harness substituted the naive rule for another cause.

``--reject=penalize`` (default) records it with the RSS so far minus 1, so the sampler learns where the budget breaks;
``--reject=prune`` raises ``optuna.TrialPruned``.

**Early stop.** Episodes are played in 4 chunks of 16. After each, the trial reports the RSS of the episodes played so
far (the same prefix for every trial), and the median pruner may stop it (``--noprune`` turns it off).

**Anchors.** Queued first: ``params.json`` as it stands ("defaults"), the file given by ``--seed_params`` ("seed"),
and in pass 1 also the configuration measured at 0.7848 (24-week window, terminal credit 0.7 on Small). The best
trial's paired gap against the defaults is computed on the same episodes; it is in-sample, so confirm on another root
and on dev (the commands are printed at the end, with any parameter of the best trial within 3 % of a bound).

**Outputs** (``outputs/research/tune_compact_optuna/<study>/``, rewritten after every trial):
- ``study.db``: the Optuna storage (SQLite); running the script again with the same ``--study`` continues it. Several
  copies with the same ``--study`` can share it, one per idle core.
- ``best_small_params.json``: the best accepted trial's complete knob set, usable as a preset
  (``SBF_PARAMS_FILE=/abs/path/best_small_params.json``).
- ``best_small_trial.json``: its number, RSS by level, paired gap against the defaults, and the search's counts.
- ``trials.csv``: one row per finished trial.

A trial costs one rollout per episode: about 4 to 30 minutes for 64 episodes, depending on the window length.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agents" / "compact_hierarchical"
OUT_DIR = ROOT / "outputs" / "research" / "tune_compact_optuna"

TASK = "small"
EPISODES = 64
ENTROPY = 202610
CPU_LIMIT_S = 1.5  # per week, the harness's meter (a week over it is played by the naive rule)
CHUNKS = 4

# the queued anchors of pass 1: the defaults (params.json), and the configuration measured at 0.7848 on 64 episodes
ANCHORS = {
    "defaults": {},
    "h24_credit07": {"horizon_small": 24, "terminal_frac_small": 0.7},
}

CRISIS_MAX = 0.95  # pass 2: crisis_frac = safety_frac + crisis_gap x (CRISIS_MAX - safety_frac), so crisis >= safety
# (kind, low, high, step): "int" steps, "float" uniform, "log" log-uniform
SPACES = {
    "pass1": {
        "horizon_small": ("int", 16, 52, 4),
        "terminal_frac_small": ("float", 0.0, 1.0, None),
        "floor_taper_weeks": ("int", 0, 16, 1),
        "safety_frac": ("float", 0.0, 0.7, None),
        "safety_price": ("log", 0.005, 0.5, None),
        "short_price": ("log", 1.0, 30.0, None),
        "crisis_frac": ("float", 0.3, 0.95, None),
        "crisis_price_mult": ("log", 1.0, 10.0, None),
        "reroute_max_wait": ("int", 0, 6, 1),
    },
    # pass 1's best sat on short_price's lower bound (1.018 of 1) and near safety_price's upper one (0.48 of 0.5),
    # with crisis_frac (0.32) below safety_frac (0.59), where the crisis floor raise does nothing
    "pass2": {
        "horizon_small": ("int", 16, 52, 4),
        "terminal_frac_small": ("float", 0.0, 1.0, None),
        "floor_taper_weeks": ("int", 0, 16, 1),
        "safety_frac": ("float", 0.0, 0.85, None),
        "safety_price": ("log", 0.005, 5.0, None),
        "short_price": ("log", 0.01, 30.0, None),
        "crisis_gap": ("float", 0.0, 1.0, None),
        "crisis_price_mult": ("log", 1.0, 10.0, None),
        "reroute_max_wait": ("int", 0, 10, 1),
    },
}

KNOBS = (
    "horizon_small",
    "terminal_frac_small",
    "floor_taper_weeks",
    "safety_frac",
    "safety_price",
    "short_price",
    "crisis_frac",
    "crisis_price_mult",
    "reroute_max_wait",
)


def suggest(trial, space: str) -> dict:
    """The trial's agent knobs (KNOBS) drawn from ``SPACES[space]``; every other knob keeps params.json's value."""
    drawn = {}
    for name, (kind, lo, hi, step) in SPACES[space].items():
        if kind == "int":
            drawn[name] = trial.suggest_int(name, lo, hi, step=step)
        else:
            drawn[name] = trial.suggest_float(name, lo, hi, log=kind == "log")
    return knobs_from(drawn)


def knobs_from(drawn: dict) -> dict:
    """The agent's knobs from a trial's parameters (pass 2's crisis_gap becomes crisis_frac)."""
    knobs = {k: v for k, v in drawn.items() if k in KNOBS}
    if "crisis_gap" in drawn:
        knobs["crisis_frac"] = knobs["safety_frac"] + drawn["crisis_gap"] * (CRISIS_MAX - knobs["safety_frac"])
    return knobs


def trial_params(knobs: dict, space: str) -> dict:
    """The inverse of ``suggest`` for a queued configuration, clipped into the space (ints onto their grid)."""
    out = {}
    for name, (kind, lo, hi, step) in SPACES[space].items():
        if name == "crisis_gap":
            room = CRISIS_MAX - knobs["safety_frac"]
            value = (knobs["crisis_frac"] - knobs["safety_frac"]) / room if room > 0 else 0.0
        else:
            value = knobs[name]
        value = min(max(value, lo), hi)
        out[name] = int(lo + round((value - lo) / step) * step) if kind == "int" else float(value)
    return out


def near_bounds(params: dict, space: str, margin: float = 0.03) -> dict:
    """The trial's parameters within ``margin`` of a bound (in log terms for log-uniform ones)."""
    out = {}
    for name, (kind, lo, hi, _step) in SPACES[space].items():
        if name not in params:
            continue
        v = params[name]
        x = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if kind == "log" else (v - lo) / (hi - lo)
        if x <= margin or x >= 1 - margin:
            out[name] = {"value": v, "low": lo, "high": hi}
    return out


def knobs_of(trial) -> dict:
    """A finished trial's agent knobs (pass-1 trials without the attribute: its parameters are the knobs)."""
    return trial.user_attrs.get("knobs") or {k: v for k, v in trial.params.items() if k in KNOBS}


def load_agent():
    """The agent's class (loaded as the scorer loads it) and its config module."""
    from shockbench_flow_agent.shim import load_agent_class

    cls = load_agent_class(AGENT_DIR, "compact_hierarchical_tune")
    return cls, sys.modules["compact_hier.config"]


class Evaluator:
    """Plays one knob set on the fixed episodes, chunk by chunk, and tells whether the agent stayed in budget."""

    def __init__(self, agent_cls, ns: list[int], quick: bool, ref_jobs: int):
        from shockbench_flow_agent import EpisodeSet

        kw = {"entropy": ENTROPY, "n_jobs": ref_jobs, "quick": quick}
        self.agent_cls = agent_cls
        self.ns = ns
        self.full = EpisodeSet.build(TASK, ns, **kw)  # computes and caches the references on the first run
        bounds = np.cumsum([len(c) for c in np.array_split(np.array(ns), min(CHUNKS, len(ns)))])
        self.chunks = [EpisodeSet.build(TASK, ns[a:b], **kw) for a, b in zip([0, *bounds[:-1]], bounds)]
        self.prefixes = [EpisodeSet.build(TASK, ns[:b], **kw) for b in bounds]

    def run(self, params: dict, report=None) -> dict:
        """Play ``params`` on the chunks; ``report(step, rss_so_far)`` may raise to stop. The result's ``rejected`` is
        None or the reason."""
        made = []

        def factory(config):
            agent = self.agent_cls(config, params=params)
            made.append(agent)
            return agent

        J, levels, start = [], {}, time.perf_counter()
        for step, (chunk, prefix) in enumerate(zip(self.chunks, self.prefixes)):
            rows = chunk.play(factory, cpu_budget=CPU_LIMIT_S, n_jobs=1)
            J += [int(r["J_policy_cents"]) for r in rows]
            over = sum(int(r["cpu_weeks"]) for r in rows)
            naive = sum(int(r["fallback_weeks"]) for r in rows)
            own = sum(1 for a in made for w in a.log if not w["plan"])
            errors = [w["error"] for a in made for w in a.log if w["error"]]
            made.clear()
            score = prefix.rss(J)
            levels = score["rss_by_stratum"]
            rejected = None
            if over:
                rejected = f"{over} week(s) over {CPU_LIMIT_S} s"
            elif naive:
                rejected = f"{naive} week(s) played by the naive rule"
            elif own:
                rejected = f"{own} week(s) of the agent's own fallback" + (f" ({errors[0]})" if errors else "")
            result = {
                "rss": float(score["rss"]),
                "levels": {int(k): float(v) for k, v in levels.items()},
                "J": J,
                "episodes": len(J),
                "rejected": rejected,
                "seconds": round(time.perf_counter() - start, 1),
            }
            if rejected:
                return result
            if report is not None:
                report(step, result["rss"])
        return result


def complete_params(base: dict, values: dict) -> dict:
    params = dict(base)
    params.update(values)
    return params


def write_outputs(study, base: dict, full_set, out: Path, space: str) -> None:
    """best_small_params.json, best_small_trial.json and trials.csv from the study's finished trials."""
    import optuna

    from sbf_starter.evolve.evaluate import paired_gap

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = []
    for t in done:
        rows.append(
            {
                "number": t.number,
                "value": t.value,
                "rss": t.user_attrs.get("rss"),
                "rejected": t.user_attrs.get("rejected") or "",
                "anchor": t.user_attrs.get("anchor", ""),
                "seconds": t.user_attrs.get("seconds"),
                **{k: knobs_of(t).get(k) for k in KNOBS},
            }
        )
    with open(out / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["number"])
        writer.writeheader()
        writer.writerows(rows)
    accepted = [t for t in done if not t.user_attrs.get("rejected") and t.user_attrs.get("episodes") == EPISODES]
    if not accepted:
        return
    best = max(accepted, key=lambda t: t.user_attrs["rss"])
    params = complete_params(base, knobs_of(best))
    (out / "best_small_params.json").write_text(json.dumps(params, indent=2) + "\n")
    info = {
        "trial": best.number,
        "rss": best.user_attrs["rss"],
        "levels": best.user_attrs.get("levels"),
        "knobs": knobs_of(best),
        "space": space,
        "near_bounds": near_bounds(best.params, space),
        "task": TASK,
        "episodes": EPISODES,
        "entropy": ENTROPY,
        "trials_finished": len(done),
        "trials_accepted": len(accepted),
        "trials_rejected": sum(1 for t in done if t.user_attrs.get("rejected")),
        "trials_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
    }
    anchor = next((t for t in accepted if t.user_attrs.get("anchor") == "defaults"), None)
    if anchor is not None and anchor.number != best.number:
        try:  # needs harm levels on the episodes (absent with quick references)
            g = paired_gap(full_set, best.user_attrs["J"], anchor.user_attrs["J"])
            gap = {"diff": g["diff"], "lo": g["lo"], "hi": g["hi"]}
        except (ZeroDivisionError, ValueError, KeyError) as exc:
            gap = {"error": f"{type(exc).__name__}: {exc}"}
        info["vs_defaults_in_sample"] = gap | {"defaults_rss": anchor.user_attrs.get("rss")}
    (out / "best_small_trial.json").write_text(json.dumps(info, indent=2) + "\n")


def main(
    trials: int = 100,
    space: str = "pass2",
    study: str = "",
    seed_params: str = "",
    reject: str = "penalize",
    prune: bool = True,
    seed: int = 0,
    ref_jobs: int = -1,
    smoke: bool = False,
) -> None:
    """Run (or continue) the study.

    Args:
        trials: trials to run in this call (a continued study adds them).
        space: "pass2" (default: widened bounds, crisis_frac >= safety_frac) or "pass1" (the first sweep's space).
        study: the study's name (default compact_small for pass1, compact_small_pass2 for pass2); its folder and
            SQLite storage are outputs/research/tune_compact_optuna/<study>/.
        seed_params: a complete knob file queued as the first trial (pass 1's best_small_params.json, say).
        reject: "penalize" (the RSS so far minus 1) or "prune" (optuna.TrialPruned) for a trial over budget.
        prune: the median pruner on the RSS after each chunk (--noprune: every trial plays all 64 episodes).
        seed: the TPE sampler's seed.
        ref_jobs: workers for the episodes' references on the first run (-1: all cores); trials are serial.
        smoke: 2 trials on 2 episodes with quick references, in memory (no files): checks the plumbing only.

    """
    try:
        import optuna
    except ImportError:
        sys.exit("optuna is not installed: uv sync --extra evolve")
    if reject not in ("penalize", "prune"):
        raise ValueError(f"reject must be 'penalize' or 'prune', got {reject!r}")
    if space not in SPACES:
        raise ValueError(f"space must be one of {sorted(SPACES)}, got {space!r}")
    study = study or ("compact_small" if space == "pass1" else f"compact_small_{space}")
    stray = sorted(k for k in os.environ if k.startswith("SBF_PARAM"))
    if stray:
        print(f"note: {stray} are ignored here: trials use params.json and the trial's knobs only", file=sys.stderr)

    agent_cls, config = load_agent()
    base = config.load(AGENT_DIR, environ={})
    ns = [0, 1] if smoke else list(range(EPISODES))
    print(f"{TASK}: {len(ns)} episodes of root {ENTROPY}; building or reading their references ...", flush=True)
    evaluator = Evaluator(agent_cls, ns, quick=smoke, ref_jobs=ref_jobs)

    out = OUT_DIR / study
    storage = None
    if not smoke:
        out.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{out / 'study.db'}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0) if prune else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=12)
    st = optuna.create_study(
        study_name=study, storage=storage, direction="maximize", sampler=sampler, pruner=pruner, load_if_exists=True
    )
    foreign = sorted({k for t in st.trials for k in t.params} - set(SPACES[space]))
    if foreign:
        sys.exit(f"study {study!r} was run with another space (parameters {foreign}): pass another --study")
    anchors = {"defaults": {}} if space != "pass1" else dict(ANCHORS)
    if seed_params:
        anchors["seed"] = config.load(AGENT_DIR, environ={"SBF_PARAMS_FILE": str(Path(seed_params).resolve())})
    for name, values in anchors.items():
        knobs = {k: complete_params(base, values)[k] for k in KNOBS}
        st.enqueue_trial(trial_params(knobs, space), user_attrs={"anchor": name}, skip_if_exists=True)

    def objective(trial):
        knobs = suggest(trial, space)
        trial.set_user_attr("knobs", knobs)
        params = complete_params(base, knobs)

        def report(step, value):
            trial.report(value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = evaluator.run(params, report=report)
        for key in ("rss", "levels", "J", "episodes", "rejected", "seconds"):
            trial.set_user_attr(key, result[key])
        levels = " ".join(f"L{k} {v:.3f}" for k, v in sorted(result["levels"].items()))
        tag = f" REJECTED: {result['rejected']}" if result["rejected"] else ""
        print(
            f"trial {trial.number}: RSS {result['rss']:.4f} on {result['episodes']} episodes ({levels}), "
            f"{result['seconds']} s{tag}",
            flush=True,
        )
        if result["rejected"]:
            if reject == "prune":
                raise optuna.TrialPruned()
            return result["rss"] - 1.0
        return result["rss"]

    def save(study_, _trial):
        if smoke:
            return
        try:  # a file that fails to write must not stop a sweep of hours
            write_outputs(study_, base, evaluator.full, out, space)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: outputs not written after this trial: {type(exc).__name__}: {exc}", file=sys.stderr)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, n_trials=2 if smoke else trials, callbacks=[save], gc_after_trial=True)

    if smoke:
        print("smoke: the plumbing works (2 trials on 2 episodes; nothing written)")
        return
    best_file = out / "best_small_params.json"
    if best_file.is_file():
        info = json.loads((out / "best_small_trial.json").read_text())
        print(f"best accepted trial {info['trial']}: RSS {info['rss']:.4f}; written {best_file}")
        if info.get("near_bounds"):
            print(f"  near a bound of the search (widen before trusting): {info['near_bounds']}")
        if "vs_defaults_in_sample" in info:
            g = info["vs_defaults_in_sample"]
            gap = f"{g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]" if "diff" in g else g.get("error")
            print(f"  vs the defaults on these episodes (in-sample): {gap}")
        print("confirm out of sample (another root, then dev):")
        command = f"SBF_PARAMS_FILE={best_file} uv run sbf evaluate compact_hierarchical --task=small"
        print(f"  {command} --entropy=12345 --episodes=64")
        print(f"  {command}")
    else:
        print("no accepted trial with all 64 episodes yet")


if __name__ == "__main__":
    fire.Fire(main)

"""Record the signals an agent sees, week by week, for offline analysis (research doc 4.1).

    uv run python scripts/research/signals.py      # 52 Small episodes: 32 of private root 20261004 and the 20 dev

Per episode and week: the open fraction of each strait, the warning score of each strait unit, the live message
threads (id, channel, kind, target kind, target), the pending prohibitions (edge, commodity, effect week) and the
number of prohibited (edge, commodity) pairs. A do-nothing policy plays (the disruptions do not depend on actions).
Output: outputs/research/signals/<date_time>/signals.pkl, a tuple (jobs, levels, records).

The statistics of the research document (warning AUROC, closure durations, threat leads) were computed from this
pickle interactively and are not scripted yet; ``auroc`` is the estimator used.
"""

import pickle

import fire
import numpy as np
from common import episodes, run_dir
from joblib import Parallel, delayed


def record(root: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401  (registers the environments)
    from shockbench_flow_agent import agent_config

    env = gym.make("ShockBench/Small-v0", entropy=root)
    obs, info = env.reset(options={"episode": n})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    units = cfg["layout"]["warning_units"]
    strait_units = [i for i, u in enumerate(units) if u[0] == "chokepoint"]
    nothing = {"flows": np.zeros(env.unwrapped.layout.n_slots)}
    rec = {"open": [], "warn": [], "threads": [], "pending": [], "prohib": []}
    done = False
    while not done:
        rec["open"].append(obs["graph_now.open"].tolist())
        rec["warn"].append(obs["warning.score"][strait_units].tolist())
        live = np.nonzero(obs["messages.msg_id.observed"].astype(bool))[0]
        rec["threads"].append(
            [
                (
                    int(obs["messages.msg_id"][i]),
                    int(obs["messages.channel"][i]),
                    int(obs["messages.kind"][i]),
                    int(obs["messages.target_kind"][i]),
                    int(obs["messages.target"][i]) if obs["messages.target.observed"][i] else -1,
                )
                for i in live
            ]
        )
        pend = np.nonzero(obs["pending_prohibitions.edge.observed"].astype(bool))[0]
        rec["pending"].append(
            [
                (
                    int(obs["pending_prohibitions.edge"][i]),
                    int(obs["pending_prohibitions.k"][i]),
                    int(obs["pending_prohibitions.effective_week"][i]),
                )
                for i in pend
            ]
        )
        rec["prohib"].append(int(obs["graph_now.prohibited"].sum()))
        obs, _r, term, trunc, _i = env.step(nothing)
        done = term or trunc
    rec["chk_nodes"] = list(cfg["layout"]["chokepoints"])
    return rec


def auroc(score, label) -> float:
    """The Mann-Whitney AUROC of ``score`` for the binary ``label`` (nan without both classes)."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(root: int = 20261004, per_level: int = 8, workers: int = 3) -> None:
    """Record the private pool of ``root`` (``per_level`` per harm level) and the 20 Small dev episodes."""
    jobs, levels = [], []
    for es, r in ((episodes("small", root, per_level, workers), root), (episodes("small", 0, 0, workers), 0)):
        for ref in es.references:
            jobs.append((r, ref["episode"]))
            levels.append(ref["stratum"])
    recs = Parallel(n_jobs=workers)(delayed(record)(*j) for j in jobs)
    out = run_dir("signals") / "signals.pkl"
    out.write_bytes(pickle.dumps((jobs, levels, recs)))
    print(f"saved {len(recs)} episodes to {out}")


if __name__ == "__main__":
    fire.Fire(main)

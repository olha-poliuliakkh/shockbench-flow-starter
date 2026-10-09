"""Routing churn and detour costs of two agents on the same episodes (did a forecast change cut false-alarm detours?).

    uv run python scripts/research/route_churn.py --a=belief48 --b=twopass48            # Small dev
    uv run python scripts/research/route_churn.py --a=belief48 --b=twopass48 --task=full --root=20261006 --per_level=2

Replays each agent (agents/<name>, or an agent_sweep.py variant ``name:NAME=value,params.NAME=value``) episode by
episode as the scorer does (metered shim, its policy seed) and reports the RSS of each and their paired gap, the mean
of every cost component, and, by harm level:
- routing cost: freight + tariff + war risk + queue holding (USD per episode; detours and waiting at straits show here);
- churn: the mean over weeks of sum |flows_t - flows_{t-1}| / sum flows_{t-1}, the share of the routing plan that
  changes week to week;
- shed + shortage (USD per episode), which a cautious forecast trades against routing cost.
Results: outputs/research/route_churn/<date_time>/results.json.
"""

import json

import fire
import numpy as np
from agent_sweep import materialize
from common import components, episodes, run_dir, world
from joblib import Parallel, delayed


ROUTING = ("freight", "tariff", "war_risk", "queue_holding")


def replay(task: str, root: int, n: int, folder: str) -> dict:
    import io
    from contextlib import redirect_stderr

    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback, _seed = world(task, root, n)
    shim = _metered_shim(load_agent_class(folder, "churn_agent"), None)
    with redirect_stderr(io.StringIO()):
        traj = rollout(
            inst, shim, omega, "standard", _policy_seed(root, n, NO_ZIP_SHA256), marks=marks, fallback=fallback
        )
    unload_agent()
    S = len(inst.action_slots)
    flows = np.zeros((len(traj.actions), S))
    for t, a in enumerate(traj.actions):
        f = (a or {}).get("flows") or {}
        for s, q in zip(f.get("slot", []), f.get("qty", [])):
            if 0 <= int(s) < S and np.isfinite(float(q)):
                flows[t, int(s)] = max(0.0, float(q))
    prev, diff = flows[:-1].sum(axis=1), np.abs(np.diff(flows, axis=0)).sum(axis=1)
    churn = float(np.mean(diff[prev > 0] / prev[prev > 0])) if np.any(prev > 0) else 0.0
    comps = components(traj.records)
    return {
        "episode": n,
        "J": int(traj.J_cents),
        "routing": float(sum(comps.get(k, 0.0) for k in ROUTING)),
        "shed_shortage": float(comps.get("shed", 0.0) + comps.get("shortage", 0.0)),
        "churn": churn,
        "comps": comps,
    }


def main(a: str, b: str, task: str = "small", root: int = 0, per_level: int = 0, workers: int = 3) -> None:
    """Replay agents ``a`` and ``b`` and print the comparison by harm level (module docstring)."""
    es = episodes(task, root, per_level, workers)
    refs = es.references
    run = run_dir("route_churn")
    from sbf_starter.evolve.evaluate import paired_gap

    rows = {}
    for name in (a, b):
        _label, folder = materialize(name, run)
        rows[name] = {
            r["episode"]: r
            for r in Parallel(n_jobs=workers)(delayed(replay)(task, root, ref["episode"], folder) for ref in refs)
        }
    print(f"{len(refs)} {task} episodes (root {root}): {a} vs {b}; run folder {run}")
    ns = [r["episode"] for r in refs]
    Ja, Jb = [rows[a][n]["J"] for n in ns], [rows[b][n]["J"] for n in ns]
    g = paired_gap(es, Ja, Jb)
    print(f"RSS {es.rss(Ja)['rss']:.4f} / {es.rss(Jb)['rss']:.4f}", end="; ")
    print(f"paired gap {g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    print("| component, $B per episode | a | b | a - b |")
    for k in rows[a][ns[0]]["comps"]:
        ca, cb = (float(np.mean([rows[x][n]["comps"][k] for n in ns])) / 1e9 for x in (a, b))
        print(f"| {k} | {ca:,.3f} | {cb:,.3f} | {ca - cb:+,.3f} |")
    print("| level | routing $B/ep (a / b) | churn (a / b) | shed + shortage $B/ep (a / b) | episodes |")
    for lv in sorted({r["stratum"] for r in refs}) + ["all"]:
        ns = [r["episode"] for r in refs if lv == "all" or r["stratum"] == lv]

        def mean(name, key):
            return float(np.mean([rows[name][n][key] for n in ns]))

        print(
            f"| {lv} | {mean(a, 'routing') / 1e9:,.2f} / {mean(b, 'routing') / 1e9:,.2f} | "
            f"{mean(a, 'churn'):.3f} / {mean(b, 'churn'):.3f} | "
            f"{mean(a, 'shed_shortage') / 1e9:,.1f} / {mean(b, 'shed_shortage') / 1e9:,.1f} | {len(ns)} |"
        )
    (run / "results.json").write_text(json.dumps({k: list(v.values()) for k, v in rows.items()}, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

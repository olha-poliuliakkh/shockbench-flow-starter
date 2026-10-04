"""Diagnostics and the evaluation report the mutator reads (docs/EVOLVE_DESIGN.md, section 4.3).

``replay`` re-plays one episode behind the agent shim, the path ``EpisodeSet.play`` takes, and keeps what the scored
cost hides: cost by component, power shed by grid, unmet demand by market, and the agent's own stderr lines (a block
that raised, a failed solve). It uses the package's private ``_world``/``_policy_seed``/``_metered_shim``, pinned by
``uv.lock``; ``tests/test_evolve.py`` checks that a replay costs what the cached score says, to the cent.

Not built yet from the design: the forecast audit and the signal audit (report parts 4 and 5).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from sbf_starter.evolve.evaluate import WEIGHTS, dollars_by_level


COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
LOGISTICS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "disposal")


def _world(task: str, root: int, episode: int, fq_replications: int | None = None):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world as world

    return world(task, root, episode, fq_replications or REPLICATIONS, str(default_cache_dir()))


def _summarize(traj, inst) -> dict:
    comps = {c: 0.0 for c in COMPONENTS}
    shed = np.zeros(len(inst.grids))
    lost = np.zeros(len(inst.demands))
    weekly = []
    for r in traj.records:
        for k, v in r.costs.as_dict().items():
            comps[k] += v
        shed += np.asarray(r.shed, dtype=float)
        lost += np.asarray(r.lost, dtype=float)
        weekly.append(r.cost_cents / 100)
    comps["salvage"] = -(traj.salvage or 0.0)
    names = [n.id for n in inst.nodes]
    backlog_end = np.asarray(traj.records[-1].backlog, dtype=float) if traj.records else np.zeros(len(inst.demands))
    return {
        "J_cents": int(traj.J_cents),
        "components": comps,
        "shed_by_grid": {names[g]: float(q) for g, q in zip(inst.grids, shed)},
        "unserved_by_market": {
            f"{names[d.node]}/{inst.commodities[d.k].id}": float(q + b)
            for d, q, b in zip(inst.demands, lost, backlog_end)
        },
        "weekly_cost": weekly,
    }


def replay(task: str, root: int, episode: int, folder: str, fq_replications: int | None = None) -> dict:
    """One episode of the agent folder, with per-component costs and its stderr (seeded as a class is)."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = _world(task, root, episode, fq_replications)
    shim = _metered_shim(load_agent_class(folder, "trace_agent"), None)
    err = io.StringIO()
    with redirect_stderr(err):
        traj = rollout(
            inst, shim, omega, "standard", _policy_seed(root, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback
        )
    unload_agent()
    out = _summarize(traj, inst)
    lines = [ln for ln in err.getvalue().splitlines() if "week" in ln]
    out["agent_messages"] = lines[:5]
    out["agent_message_count"] = len(lines)
    return out


def replay_naive(task: str, root: int, episode: int) -> dict:
    """The naive anchor's components on the episode (agent-independent)."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS, anchor_policy
    from shockbench_flow_agent.local_eval import ANCHOR_REGIME, NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fallback = _world(task, root, episode)
    _i, params = task_generator(task)
    traj = rollout(
        inst,
        anchor_policy(inst, params, REPLICATIONS),
        omega,
        ANCHOR_REGIME,
        _policy_seed(root, episode, NO_ZIP_SHA256),
        marks=marks,
        fallback=fallback,
    )
    return _summarize(traj, inst)


def events(task: str, root: int, episode: int, limit: int = 8) -> list[str]:
    """A few lines of what an agent saw happen in the episode (straits, prohibitions, threads)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_gym.timeline import episode_events

    from sbf_starter import env_id

    ev = episode_events(gym.make(env_id(task), entropy=root), options={"episode": episode})
    keep = [
        e
        for e in ev
        if e["signal"]
        in ("chokepoint", "prohibited", "pending", "message", "message_ended", "capacity", "fab", "grid", "supply")
    ]
    lines = [
        f"week {e['week']}: {e['signal']} {e['subject']} {e['before']} -> {e['after']} {e.get('note') or ''}"
        for e in keep
    ]
    if len(lines) > limit:
        lines = lines[: limit - 1] + [f"... and {len(lines) - limit + 1} more events"]
    return lines


class Tracer:
    """Replays cached on disk by (file hash, task, root, episode); naive's by (task, root, episode)."""

    def __init__(self, cache_dir: Path, workers: int = 4):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def _cached(self, name: str, fn, *args) -> dict:
        path = self.dir / f"{name}.json"
        if path.is_file():
            return json.loads(path.read_text())
        doc = fn(*args)
        path.write_text(json.dumps(doc) + "\n")
        return doc

    def many(self, sha: str, folder: str, task: str, root: int, episodes: list[int]) -> dict[int, dict]:
        from joblib import Parallel, delayed

        todo = [n for n in episodes if not (self.dir / f"{sha[:24]}-{task}-{root}-{n}.json").is_file()]
        if todo:
            docs = Parallel(n_jobs=min(self.workers, len(todo)))(delayed(replay)(task, root, n, folder) for n in todo)
            for n, d in zip(todo, docs):
                (self.dir / f"{sha[:24]}-{task}-{root}-{n}.json").write_text(json.dumps(d) + "\n")
        return {n: self._cached(f"{sha[:24]}-{task}-{root}-{n}", replay, task, root, n, folder) for n in episodes}

    def naive(self, task: str, root: int, episode: int) -> dict:
        return self._cached(f"naive-{task}-{root}-{episode}", replay_naive, task, root, episode)

    def events(self, task: str, root: int, episode: int) -> list[str]:
        return self._cached(f"events-{task}-{root}-{episode}", lambda *a: {"lines": events(*a)}, task, root, episode)[
            "lines"
        ]


def pick_episodes(es, J, J_ref, k: int = 2) -> list[int]:
    """k episodes where the candidate lost most against the reference, then k with the largest gap to clairvoyant."""
    refs = es.references
    loss = sorted(range(len(J)), key=lambda i: -(J[i] - J_ref[i]))[:k]
    gap = [
        i for i in sorted(range(len(J)), key=lambda i: -(J[i] - (refs[i]["J_oracle_cents"] or J[i]))) if i not in loss
    ][:k]
    return [refs[i]["episode"] for i in loss + gap]


def _b(usd: float) -> str:
    return f"{usd / 1e9:+,.1f}B" if abs(usd) < 1e13 else f"{usd / 1e12:+,.2f}T"


def report(
    es,
    task: str,
    root: int,
    J,
    J_ref,
    ref_name: str,
    traces: dict,
    ref_traces: dict,
    tracer: Tracer,
    health: dict,
    gap: dict | None = None,
    predicted: dict | None = None,
) -> str:
    """The Markdown report of a candidate against its reference (its parent, or naive for the seed)."""
    lines = []
    rss = es.rss([int(j) for j in J])["rss"]
    head = f"Training RSS {rss:.4f} on {len(J)} episodes"
    if gap:
        head += (
            f"; against {ref_name}: {gap['diff']:+.4f} (90 % paired interval {gap['lo']:+.4f} to "
            f"{gap['hi']:+.4f}; better on {gap['p_better']:.0%} of resampled sets)"
        )
    lines += ["### Headline", head, ""]
    mine, ref = dollars_by_level(es, J), dollars_by_level(es, J_ref)
    lines += [
        "### Dollars by harm level (USD per episode; RSS sums p * g)",
        "| level | p | n | saving g vs naive | attainable D | p*g | p*g change vs " + ref_name + " | ratio g/D |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s, d in mine.items():
        delta = d["p_g"] - ref[s]["p_g"] if s in ref else 0.0
        lines.append(
            f"| {s} | {WEIGHTS[s - 1]} | {d['n']} | {_b(d['g'])} | {_b(d['D'])} | {_b(d['p_g'])} | "
            f"{_b(delta)} | {d['ratio']:.3f} |"
        )
    lines.append("")
    if traces:
        lines += [
            f"### Cost components on {len(traces)} replayed episodes, candidate minus {ref_name} (USD per episode)"
        ]
        comp = {
            c: np.mean([traces[n]["components"][c] - ref_traces[n]["components"][c] for n in traces])
            for c in (*COMPONENTS, "salvage")
        }
        lines.append(", ".join(f"{c} {_b(v)}" for c, v in comp.items()))
        shed = {
            g: np.mean([traces[n]["shed_by_grid"][g] - ref_traces[n]["shed_by_grid"][g] for n in traces])
            for g in next(iter(traces.values()))["shed_by_grid"]
        }
        lost = {
            m: np.mean([traces[n]["unserved_by_market"][m] - ref_traces[n]["unserved_by_market"][m] for n in traces])
            for m in next(iter(traces.values()))["unserved_by_market"]
        }
        lines.append(
            "power shed by grid (units, candidate minus "
            + ref_name
            + "): "
            + ", ".join(f"{g} {v:+,.0f}" for g, v in shed.items())
        )
        lines.append("unserved demand by market (units): " + ", ".join(f"{m} {v:+,.0f}" for m, v in lost.items()))
        lines += ["", "### Replayed episodes"]
        level = {r["episode"]: r["stratum"] for r in es.references}
        for n, tr in traces.items():
            d = tr["J_cents"] / 100 - ref_traces[n]["J_cents"] / 100
            big = sorted(
                ((c, tr["components"][c] - ref_traces[n]["components"][c]) for c in COMPONENTS),
                key=lambda x: -abs(x[1]),
            )[:3]
            lines.append(
                f"- episode {n} (level {level.get(n)}): cost {_b(d)} against {ref_name}; largest changes "
                + ", ".join(f"{c} {_b(v)}" for c, v in big)
            )
            for e in tracer.events(task, root, n):
                lines.append(f"  - {e}")
            for m in tr.get("agent_messages", [])[:2]:
                lines.append(f"  - agent stderr: {m[:200]}")
        lines.append("")
    lines += ["### Health", ", ".join(f"{k} {v}" for k, v in health.items()), ""]
    if predicted:
        meas = {s: mine[s]["g"] - ref[s]["g"] for s in mine if s in ref}
        lines += [
            "### Calibration (USD per episode, change against " + ref_name + ")",
            ", ".join(
                f"level {s}: predicted {_b(float(predicted.get(f'level_{s}', 0)))}, measured {_b(meas[s])}"
                for s in meas
            ),
            "",
        ]
    return "\n".join(lines)

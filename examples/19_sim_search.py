"""Improve the MPC's weekly action by a search evaluated in the real simulator (offline; rebuilt 2026-10-10).

Every week ``agents/<agent>`` proposes its action. For each decision group (the fuel bound for a grid, a strait's
tanker releases of one commodity, the wafers for a fab, the raw chips for an OSAT) and each multiplier in ``LEVELS``
the group's flows are scaled (capped at capacity) and the candidate is played to the episode's end on a copy of the
environment: the first ``head_weeks`` weeks by the same agent (deterministic through a node limit), the rest by the
``--tail`` policy (another agent folder, or ``<agent>:k=v,...`` the same agent with PARAMS overrides; none: the agent
throughout). A candidate replaces the action when it saves more than ``min_gain`` of the base branch's cost. On Small
dev with 6 candidates a week and the MPC playing the whole tail this gained +0.026 RSS (2026-10-09).

    uv run python examples/19_sim_search.py --episodes=4 --evals=6 --head_weeks=8 --tail=nn_student
    uv run python examples/19_sim_search.py --episodes=4 --evals=6 --head_weeks=8 \\
        --tail="mpc_tuned:fab_threshold=0,horizon=8,milp_time=0.1"
"""

import copy
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
LEVELS = (1.5, 2.0, 3.0)  # up only: the MPC under-ships more often than it over-ships


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def agent_module(spec_str: str, name: str, nodes: int):
    """``folder`` or ``folder:k=v,...``: the agent's module with its params.json and the overrides applied."""
    folder, _, over = spec_str.partition(":")
    mod = load_module(ROOT / "agents" / folder / "agent.py", name)
    pj = ROOT / "agents" / folder / "params.json"
    if hasattr(mod, "PARAMS") and pj.is_file():
        mod.PARAMS.update(json.loads(pj.read_text()))
    if over and hasattr(mod, "PARAMS"):
        mod.PARAMS.update({k: float(v) for k, v in (kv.split("=") for kv in over.split(","))})
    if nodes > 0 and hasattr(mod, "PARAMS") and "milp_nodes" in mod.PARAMS:
        mod.PARAMS.update({"milp_nodes": nodes, "milp_time": 5.0})
    return mod


def decision_groups(static, layout):
    """(kind, index, slot indices): fuel->grid, tanker (strait, k) releases, wafer->fab, raw->OSAT."""
    slots, lanes, edges, nodes = static["action_slots"], static["lanes"], static["edges"], static["nodes"]
    com = static["commodities"]["id"]
    dest = []
    for e, lane in zip(slots["edge"], slots["lane"]):
        last = lanes["edges"][lane][-1] if lane is not None else e
        dest.append(edges["head"][last])
    feeds = {}
    for t_, h in zip(edges["tail"], edges["head"]):
        if nodes["type"][t_] == "terminal" and nodes["type"][h] == "grid":
            feeds.setdefault(t_, set()).add(h)
    kind = [com[k] for k in slots["k"]]
    groups = []
    for i, g in enumerate(layout["grids"]):
        s = [
            j
            for j, (d, k) in enumerate(zip(dest, kind))
            if k in ("lng", "crude", "nucfuel") and (d == g or g in feeds.get(d, ()))
        ]
        groups.append(("fuel", i, np.array(s, dtype=int)))
    ov = static["override_slots"]
    for i, (c, k) in enumerate(tuple(x) for x in layout.get("release_pairs", [])):
        s = [j for j, (cc, kk) in enumerate(zip(ov["chokepoint"], ov["k"])) if (cc, kk) == (c, k)]
        groups.append(("tanker", i, np.array(s, dtype=int)))
    for i, f in enumerate(layout["fabs"]):
        groups.append(
            ("wafer", i, np.array([j for j, (d, k) in enumerate(zip(dest, kind)) if d == f and k == "wafer"], int))
        )
    for i, oo in enumerate(layout["osats"]):
        s = [j for j, (d, k) in enumerate(zip(dest, kind)) if d == oo and k.endswith("_raw")]
        groups.append(("raw", i, np.array(s, dtype=int)))
    return [g for g in groups if len(g[2])]


def scaled(action, group, a, u0, ov_u0, mask):
    act = {k: np.array(v, copy=True) for k, v in action.items()}
    kind, _i, idx = group
    if kind == "tanker":
        if "override_qty" in act:
            q = act["override_qty"]
            q[idx] = np.minimum(q[idx] * a, ov_u0[idx])
    else:
        f = act["flows"]
        f[idx] = np.minimum(f[idx] * a, u0[idx] * mask[idx])
    return act


def play_from(env, shim, obs, wire, tail, head_weeks: int) -> float:
    """USD from this week to the end: ``wire`` now, the shim's agent for head_weeks, ``tail`` (or the shim) after."""
    obs, reward, done, _tr, _info = env.step(wire)
    total, n = -reward, 1
    while not done:
        player = shim if tail is None or n <= head_weeks else tail
        obs, reward, done, _tr, _info = env.step(player.act(obs))
        total -= reward
        n += 1
    return total


def episode(task, ep, spec, agent, tail, evals, head_weeks, min_gain, nodes) -> dict:
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow_agent import agent_config
    from shockbench_flow_agent.convert import action_to_wire, observation_dict
    from shockbench_flow_agent.scoring import _world
    from shockbench_flow_agent.shim import AgentShim

    mod = agent_module(agent, f"{agent.split(':')[0]}_search", nodes)
    _task, entropy, regime, fq, cache = spec
    inst, omega, marks, fallback = _world(task, entropy, ep, fq, cache)
    env = Env(fallback=fallback)
    obs, info = env.reset(inst, regime, omega, 0, marks=marks)
    shim = AgentShim(mod.Agent)
    shim.reset(info["static"], obs, 0)
    tail_shim = None
    if tail:
        tmod = agent_module(tail, f"{tail.split(':')[0]}_tail", nodes)
        tail_shim = AgentShim(tmod.Agent)
        tail_shim.reset(info["static"], obs, 0)
    layout = shim.layout
    o0 = observation_dict(layout, obs)
    config = agent_config(copy.deepcopy(info["static"]), 0, layout, o0)
    groups = decision_groups(info["static"], config["layout"])
    edges, slots = info["static"]["edges"], info["static"]["action_slots"]
    u0 = np.array([edges["u0"][e] or np.inf for e in slots["edge"]], dtype=float)
    ov_u0 = np.array([edges["u0"][e] or np.inf for e in info["static"]["override_slots"]["out_edge"]], dtype=float)
    rng = np.random.default_rng(ep)
    improved, weeks, applied = 0, 0, []
    done = False
    while not done:
        o = observation_dict(layout, obs)
        week = int(o["week"][0])
        base_action = {k: np.array(v, copy=True) for k, v in shim.agent.act(o).items()}
        mask = o["action_mask"].astype(float)

        def cost_of(action):
            ec, sc = copy.deepcopy((env, shim))
            tc = None if tail_shim is None else copy.deepcopy(tail_shim)
            return play_from(ec, sc, obs, action_to_wire(layout, week, action), tc, head_weeks)

        best, best_cost = base_action, cost_of(base_action)
        base_cost, n_eval = best_cost, 1
        for g in rng.permutation(len(groups)):
            if n_eval >= evals:
                break
            for a in LEVELS:
                if n_eval >= evals:
                    break
                cand = scaled(best, groups[g], a, u0, ov_u0, mask)
                c = cost_of(cand)
                n_eval += 1
                if c < best_cost - 1e-6 - min_gain * abs(best_cost):
                    applied.append((week, groups[g][0], int(groups[g][1]), float(a), float(best_cost - c)))
                    best, best_cost = cand, c
        weeks += 1
        if best_cost < base_cost - 1e-6:
            improved += 1
        obs, _r, done, _tr, _info = env.step(action_to_wire(layout, week, best))
    return {
        "episode": ep,
        "J_cents": int(env.trajectory.J_cents),
        "weeks_improved": improved,
        "weeks": weeks,
        "applied": applied,
    }


def main(
    episodes: str = "4",
    agent: str = "mpc_tuned",
    tail: str = "",
    evals: int = 6,
    head_weeks: int = 8,
    min_gain: float = 0.0,
    nodes: int = 100,
    jobs: int = 4,
    task: str = "small",
    base_rows: str = "",
):
    """Play the search on Small dev episodes; RSS (and paired vs ``base_rows`` J per episode JSON if given)."""
    from shockbench_flow_agent import EpisodeSet

    dev = EpisodeSet.build(task, "dev")
    eps = list(dev.episodes) if episodes == "dev" else list(dev.episodes)[: int(episodes)]
    es = EpisodeSet.build(task, eps)
    out_dir = ROOT / "outputs" / "19_sim_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = Parallel(n_jobs=jobs)(
        delayed(episode)(task, ep, es._spec, agent, tail, evals, head_weeks, min_gain, nodes) for ep in eps
    )
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1))
    rss = es.rss([r["J_cents"] for r in rows])["rss"]
    print(
        f"agent={agent} tail={tail or 'self'} head={head_weeks} evals={evals}: "
        f"{len(rows)} eps in {time.time() - t0:.0f} s -> {out_dir}"
    )
    print(
        f"RSS {rss:.4f}; action changed in {sum(r['weeks_improved'] for r in rows)}"
        f"/{sum(r['weeks'] for r in rows)} weeks"
    )
    if base_rows:
        base = {int(k): v for k, v in json.loads(Path(base_rows).read_text()).items()}
        print(f"base RSS on these episodes {es.rss([base[e] for e in eps])['rss']:.4f}")


if __name__ == "__main__":
    fire.Fire(main)

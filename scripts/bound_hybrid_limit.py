"""Bound the simulator-in-the-loop gain: a cheating rollout lookahead on top of the perfect-information LP planner.

    uv run python scripts/bound_hybrid_limit.py                          # the 20 Small dev episodes, 3 workers
    uv run python scripts/bound_hybrid_limit.py --limit=3 --workers=3    # a sanity check on 3 episodes
    uv run python scripts/bound_hybrid_limit.py --depth=2                # 2 real weeks before the LP's cost-to-go
    uv run python scripts/bound_hybrid_limit.py --H="max(26,2L)"         # the H48 oracle window as the backbone
    uv run python scripts/bound_hybrid_limit.py --fab_factors=0.5 --osat_factors=0.5   # add the chip-side families

An offline upper-bound test, not an agent: it reads the episode's true future and steps the real simulator, so it
ignores the CPU budget and could never be submitted.

Each week t of each episode:

1. A_0 is the week-1 action of ``mpc_det`` solved on the TRUE disruption marks of its window (the oracle-window
   planner of outputs/dev/oracle_window.py; demand stays the forecast unless ``--true_demand``).
2. Candidates A_1..A_k rescale A_0's fuel flows. Fuel reaches a grid from its terminal (term_X -> grid_X) or straight
   from a source, every grid powers fabs, and a grid serves its base load first, so a fab's lot starts are cut by the
   energy its grid has left. The families: every fuel flow scaled by each of ``fuel_factors``, and, with
   ``--per_grid``, the fuel flows feeding one grid (into the grid or its terminal) scaled by each of
   ``grid_factors``. A request above capacity is clipped by the simulator, not refused. ``fab_factors`` and
   ``osat_factors`` add the earlier chip-side families (wafers into fabs, raw chips into OSATs; off by default).
3. Each candidate is played from an ``Env.snapshot()`` of the real episode, restored afterwards (``Env.restore`` is
   bit-identical, so this is an exact rollback): the candidate in week t, then the backbone planner for the next
   ``depth`` - 1 weeks, all in the real simulator. Its score is the real cost of those ``depth`` weeks plus the LP's
   cost-to-go from the observation that follows (the same oracle window, J with the terminal-credit offset). If the
   episode ends inside the rollout, the cost-to-go is 0 and the real cost includes the salvage credit. The rollout's
   planner works on copies of its memory with a cold LP session, so the real planner's state is untouched.
4. The argmin is played for real.

The same driver also plays A_0 alone (the backbone), so the two RSS are paired on the same episodes. Results go to
outputs/bound_hybrid_limit/<date_time>/results.json.
"""

import copy
import json
import os
import time
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy: one thread per worker process

import fire
import numpy as np
from joblib import Parallel, delayed


TASK, ROOT, REGIME = "small", 0, "standard"


def _world(n: int):
    """The episode's instance, omega, true marks, fallback spec and policy seed, as the scorer builds them."""
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed
    from shockbench_flow_agent.scoring import _world as world

    inst, omega, marks, fallback = world(TASK, ROOT, n, REPLICATIONS, str(default_cache_dir()))
    return inst, omega, marks, fallback, _policy_seed(ROOT, n, NO_ZIP_SHA256)


def _oracle_planner(inst, marks, H: str, true_demand: bool):
    """``mpc_det`` with the true disruption marks in its window, plus a side-effect-free ``plan(obs, memory)``."""
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
    from shockbench_flow.policies.naive_fq import generator_quantiles
    from shockbench_flow.policies.registry import PolicyContext

    class OracleWindow(MpcDet):
        def _horizon(self, inst_, plan):
            return inst_.T if H == "full" else super()._horizon(inst_, plan)

        def _window_arrays(self, inst_, obs, H_t, memory=None):
            arr = dict(L.persistence_arrays(inst_, obs, self._memory if memory is None else memory, H_t))
            t = int(obs["week"])
            sl = slice(t - 1, t - 1 + H_t)
            for k in L.WINDOW_FIELDS:
                if k == "demand" and not true_demand:
                    continue
                arr[k] = np.array(getattr(marks, k)[sl]).copy()
            return L.read_only(arr)

        def memory_copy(self):
            return copy.deepcopy(self._memory)

        def plan(self, obs: dict, memory) -> tuple[dict, float]:
            """(week-1 action, window J in USD with the offset) from ``obs``; updates ``memory``, a copy, only."""
            memory.update(self._inst, obs)
            t = int(obs["week"])
            H_t = L.window_length(self._H, t, self._inst.T)
            model = L.rolled_lp(
                self._inst,
                obs,
                self._window_arrays(self._inst, obs, H_t, memory),
                H_t,
                planning_rules=self.params.planning_rules,
            )
            res = L.LPSession().solve(L.to_highs_lp(model), L.WindowShape.of(model))
            if not res.ok:
                raise RuntimeError(f"week {t}: rollout LP failed ({res.status})")
            action = L.week1_action(self._inst, model, res.x, obs, L.prohibited_now(memory, t))
            return action, float(res.objective)

    _i, gen = task_generator(TASK)
    params = MpcDetParams(H=H) if H not in ("L", "full") else MpcDetParams()
    return OracleWindow(params, PolicyContext(fq_quantile=generator_quantiles(inst, gen)))


def _families(inst, fuel_factors, per_grid: bool, grid_factors, fab_factors, osat_factors) -> list[tuple[str, dict]]:
    """Candidate generators: (label, {slot: factor}). Slots are the action slots; a slot not named keeps 1."""
    N = inst.nodes
    grids, fabs, osats = set(inst.grids), set(inst.fabs), set(inst.osats)
    feeds = {g: g for g in grids}  # a node whose deliveries feed a grid: the grid, or a terminal with an edge to it
    for e in inst.edges:
        if e.head in grids and e.tail not in feeds and N[e.tail].id.startswith("term_"):
            feeds[e.tail] = e.head

    def dest(e, lane):
        return inst.lane_destination(lane) if lane is not None else inst.edges[e].head

    into = [dest(e, lane) for e, _k, lane in inst.action_slots]
    fuel = [s for s, d in enumerate(into) if d in feeds]
    out = [(f"fuel*{f:g}", dict.fromkeys(fuel, f)) for f in fuel_factors]
    if per_grid:
        for g in inst.grids:
            slots = [s for s in fuel if feeds[into[s]] == g]
            out += [(f"{N[g].id}*{f:g}", dict.fromkeys(slots, f)) for f in grid_factors if slots]
    out += [(f"fab*{f:g}", dict.fromkeys([s for s, d in enumerate(into) if d in fabs], f)) for f in fab_factors]
    out += [(f"osat*{f:g}", dict.fromkeys([s for s, d in enumerate(into) if d in osats], f)) for f in osat_factors]
    return out


def _scaled(action: dict, factors: dict) -> dict:
    """A copy of the wire action with each listed slot's flow multiplied by its factor (overrides and holds kept)."""
    new = copy.deepcopy(action)
    fl = new["flows"]
    fl["qty"] = [q * factors.get(s, 1.0) for s, q in zip(fl["slot"], fl["qty"], strict=True)]
    return new


def _rollout(env, pol, snap, cand: dict, depth: int) -> float | None:
    """USD: ``cand`` then the backbone for ``depth`` - 1 weeks in the real simulator, plus the LP's cost-to-go.

    ``depth`` 0 follows the backbone to the episode's end: the real cost-to-go, no LP estimate (a full rollout).
    None if the candidate is refused as a whole week (never expected for a rescaled valid action).
    """
    env.restore(snap)
    obs, _r, done, _tr, info = env.step(cand)
    if info.get("fallback"):
        return None
    cost = -int(info["reward_cents"]) / 100
    memory = pol.memory_copy()
    for _ in range(depth - 1) if depth > 0 else iter(int, 1):  # iter(int, 1): until the episode ends
        if done:
            return cost
        action, _J = pol.plan(obs, memory)
        obs, _r, done, _tr, info = env.step(action)
        cost += -int(info["reward_cents"]) / 100
    return cost if done else cost + pol.plan(obs, memory)[1]


def play(n: int, H: str, true_demand: bool, lookahead: bool, depth: int, families_kw: dict) -> dict:
    """One episode with the backbone alone (``lookahead`` False) or with the rollout lookahead."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies.base import reset_policy

    start = time.perf_counter()
    inst, omega, marks, fallback, seed = _world(n)
    pol = _oracle_planner(inst, marks, H, true_demand)
    families = _families(inst, **families_kw)
    env = Env(fallback=fallback)
    obs, info = env.reset(inst, REGIME, omega, seed, marks=marks, policy_name="bound_hybrid_limit")
    reset_policy(pol, info["static"], obs, seed, info.get("omega"))
    chosen: dict[str, int] = {}
    gains, J_real = [], 0
    done = False
    while not done:
        base = pol.act(obs)
        label, action = "base", base
        if lookahead:
            snap = env.snapshot()
            scores = {}
            for name, factors in [("base", {})] + families:
                cand = _scaled(base, factors)
                score = _rollout(env, pol, snap, cand, depth)
                if score is not None:
                    scores[name] = (score, cand)
            label = min(scores, key=lambda k: scores[k][0])
            action = scores[label][1]
            gains.append(scores["base"][0] - scores[label][0])
            env.restore(snap)
        chosen[label] = chosen.get(label, 0) + 1
        obs, _r, done, _tr, inf = env.step(action)
        J_real -= int(inf["reward_cents"])
    records = env.trajectory.records
    return {
        "episode": n,
        "J": int(env.trajectory.J_cents),
        "J_from_rewards": J_real,
        "chosen": chosen,
        "predicted_gain_usd": float(np.sum(gains)) if gains else 0.0,
        "lots_started": float(sum(np.sum(r.lots_started) for r in records)),
        "shed_usd": float(sum(r.costs.as_dict()["shed"] for r in records)),
        "shortage_usd": float(sum(r.costs.as_dict()["shortage"] for r in records)),
        "fallback_weeks": sum(1 for s in pol.telemetry if s.fallback),
        "seconds": time.perf_counter() - start,
    }


def main(
    H: str = "L",
    true_demand: bool = False,
    depth: int = 3,
    fuel_factors: str = "0.75,1.25,1.5",
    per_grid: bool = True,
    grid_factors: str = "0.75,1.5",
    fab_factors: str = "",
    osat_factors: str = "",
    limit: int = 0,
    workers: int = 3,
) -> None:
    """Play the backbone and the lookahead on the Small dev episodes and print both RSS, paired.

    Args:
        H: the oracle window's horizon label: L (the planner's own, 24 weeks on Small), "max(26,2L)" (48), "L+8", full.
        true_demand: put the true demand in the window too (default: the forecast, as in the 0.786 baseline).
        depth: real weeks simulated per candidate (the candidate, then the backbone) before the LP's cost-to-go;
            0 plays to the end of the episode (no LP estimate).
        fuel_factors: comma-separated factors for every fuel flow at once.
        per_grid: also rescale the fuel feeding one grid at a time.
        grid_factors: comma-separated factors of the per-grid candidates.
        fab_factors: comma-separated factors for every wafer-to-fab flow at once (empty: none).
        osat_factors: comma-separated factors for every raw-chip-to-OSAT flow at once (empty: none).
        limit: play only the first ``limit`` dev episodes (0: all 20).
        workers: processes (each holds one episode; keep at 3-4 on 16 GB).

    """
    from shockbench_flow_agent import EpisodeSet

    from sbf_starter.evolve.evaluate import paired_gap

    def floats(v):
        return [float(x) for x in (v if isinstance(v, (list, tuple)) else str(v).split(",")) if str(x).strip()]

    kw = {
        "fuel_factors": floats(fuel_factors),
        "per_grid": per_grid,
        "grid_factors": floats(grid_factors),
        "fab_factors": floats(fab_factors),
        "osat_factors": floats(osat_factors),
    }
    run = Path(f"outputs/bound_hybrid_limit/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    run.mkdir(parents=True, exist_ok=True)
    es = EpisodeSet.build(TASK, "dev", n_jobs=workers)
    if limit:
        es = EpisodeSet.build(TASK, [r["episode"] for r in es.references][:limit], n_jobs=workers)
    ns = [r["episode"] for r in es.references]
    print(f"{len(ns)} {TASK} dev episodes; H={H}, true_demand={true_demand}, depth={depth}; candidates {kw}")
    print(f"run folder {run}", flush=True)

    out = {"settings": {"H": H, "true_demand": true_demand, "depth": depth, **kw}}
    for lookahead in (False, True):
        tag = "lookahead" if lookahead else "backbone"
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=workers)(delayed(play)(n, H, true_demand, lookahead, depth, kw) for n in ns)
        J = [r["J"] for r in rows]
        bad = [r["episode"] for r in rows if r["J"] != r["J_from_rewards"]]
        if bad:
            print(f"  warning: the rewards do not sum to J on episodes {bad}")
        score = es.rss(J)
        chosen: dict[str, int] = {}
        for r in rows:
            for k, v in r["chosen"].items():
                chosen[k] = chosen.get(k, 0) + v
        out[tag] = {"rss": score["rss"], "score": score, "rows": rows, "chosen": chosen}
        print(f"{tag}: RSS {score['rss']:.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if lookahead:
            ranked = sorted(chosen.items(), key=lambda x: -x[1])
            print("  actions chosen (weeks): " + ", ".join(f"{k} {v}" for k, v in ranked))
    la, bb = out["lookahead"]["rows"], out["backbone"]["rows"]
    gap = paired_gap(es, [r["J"] for r in la], [r["J"] for r in bb])
    out["gap"] = gap

    def ratio(key):
        return sum(r[key] for r in la) / max(sum(r[key] for r in bb), 1e-9)

    realised = sum(b["J"] - a["J"] for a, b in zip(la, bb, strict=True)) / 100
    predicted = sum(r["predicted_gain_usd"] for r in la)
    print(
        f"\nFINAL RSS: lookahead {out['lookahead']['rss']:.4f} vs backbone {out['backbone']['rss']:.4f}; "
        f"paired gap {gap['diff']:+.4f} (90 % interval {gap['lo']:+.4f} to {gap['hi']:+.4f})\n"
        f"lookahead / backbone: lots started {ratio('lots_started'):.3f}x, shed {ratio('shed_usd'):.3f}x, "
        f"shortage {ratio('shortage_usd'):.3f}x; gain realised ${realised / 1e9:,.2f}B "
        f"(predicted per week, summed: ${predicted / 1e9:,.2f}B)"
    )
    (run / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

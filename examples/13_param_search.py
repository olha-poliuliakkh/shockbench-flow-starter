"""Coordinate search over the MPC's PARAMS on stratified episode sets of our own roots, with paired comparisons.

The agent is ``agents/mpc_safe2/agent.py`` (the board agent) started from its ``params.json``; a candidate is that plus
a few PARAMS overrides. Stages (``--stage``):

    sets     build the stratified sets: from a pool of episodes of a root, only the harm level is computed (no
             oracle), ``per`` episodes of each of the four levels are kept, and their references (naive, oracle) are
             computed once by ``EpisodeSet.build`` and cached. S1 (``search_root``) is searched on, S2
             (``confirm_root``) confirms.
    search   calibrate the run-to-run noise (the base ``repeats`` times on S1: sigma of RSS), then ``rounds`` rounds of
             coordinate descent: for every parameter of ``SPACE`` its two neighbouring values are played on S1 and the
             best is taken if its paired gain over the current point is >= delta = max(min_gain, 2 sigma), its 90 %
             interval lies above 0, and a second run of it keeps the gain. Candidates whose agent used more than
             ``cpu_limit`` CPU seconds in any week are rejected.
    confirm  the best point against the base on S2 and on the dev episodes (paired), optionally on ``full_episodes``
             dev episodes of Full.
    export   write ``agents/<export_name>/`` (the agent's code and the best PARAMS); the base agent is never changed.

Every played episode set is a line of ``outputs/13_param_search/<name>/trials.jsonl`` (params, J per episode, RSS, CPU
per week); a rerun with the same ``--name`` reuses them, so a stopped search continues where it was.

    uv run python examples/13_param_search.py --stage=sets
    uv run python examples/13_param_search.py --stage=search --name=night5        # ~5 min per candidate on 8 cores
    uv run python examples/13_param_search.py --stage=confirm --name=night5 --full_episodes=6
    uv run python examples/13_param_search.py --stage=export --name=night5 --export_name=mpc_tuned
"""

import importlib.util
import json
import shutil
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "agents" / "mpc_safe2" / "agent.py"
BASE = json.loads((ROOT / "agents" / "mpc_safe2" / "params.json").read_text())
OUT = ROOT / "outputs" / "13_param_search"
WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's harm-level weights
SPACE = {  # the values tried per parameter; the search moves one step at a time from the base's value
    "horizon": [12, 16, 20, 24],
    "terminal_frac": [0.25, 0.5, 0.8],
    "shortage_weight": [0.7, 1.0, 1.5],
    "shed_weight": [0.7, 1.0, 1.5],
    "safety_frac": [1.0, 1.5, 2.0],
    "safety_price": [0.01, 0.02, 0.05],
    "short_price": [3.0, 10.0, 30.0],
    "tau_grid": [0.0, 6.0, 12.0],
    "tau_fab": [0.0, 12.0, 24.0],
    "reopen_trust": [0.5, 0.8, 1.0],
    "holding_weight": [0.5, 1.0, 2.0],
    "osat_raw_price": [0.0, 2.0],
    "demand_scale": [1.0, 1.1],
    "milp_weeks": [3, 4, 5],
}


# ----------------------------------------------------------------------------------------------- stratified sets


def harm_row(entropy: int, n: int) -> tuple[int, float, str]:
    """Episode n's harm (USD) and generator id, without naive or the oracle."""
    from shockbench_flow.disruption.harm import episode_harm
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow_agent.scoring import _label

    inst, params = task_generator("small")
    omega = sample_omega(inst, params, entropy, n, _label(entropy))
    return n, float(episode_harm(inst, omega)), omega.generator_id


def stratified(entropy: int, pool: int, per: int, jobs: int) -> list[int]:
    """The first ``per`` episodes of each harm level among episodes 0..pool-1 of ``entropy`` (cached as JSON)."""
    path = OUT / "sets" / f"small_{entropy}_{pool}_{per}.json"
    if path.is_file():
        return json.loads(path.read_text())
    from shockbench_flow.disruption.strata import stratum
    from shockbench_flow.evaluation.cache import cut_points_cached, default_cache_dir
    from shockbench_flow.hosting.split import CUT_ENTROPY
    from shockbench_flow.hosting.tasks import CUT_DRAWS, task_generator

    inst, params = task_generator("small")
    cuts = cut_points_cached(inst, params, CUT_DRAWS, CUT_ENTROPY, jobs, default_cache_dir())
    rows = Parallel(n_jobs=jobs)(delayed(harm_row)(entropy, n) for n in range(pool))
    picked = {s: [] for s in WEIGHTS}
    for n, harm, gid in rows:
        s = stratum(harm, cuts, generator_id=gid)
        if len(picked[s]) < per:
            picked[s].append(n)
    short = {s: len(v) for s, v in picked.items() if len(v) < per}
    if short:
        raise ValueError(f"pool {pool} of root {entropy} has too few episodes of levels {short}: raise --pool")
    eps = sorted(n for v in picked.values() for n in v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eps))
    return eps


def episode_set(task: str, entropy: int, eps):
    """``EpisodeSet`` of the episodes (references computed once and cached by the package)."""
    from shockbench_flow_agent import EpisodeSet

    return EpisodeSet.build(task, eps, entropy=entropy) if entropy else EpisodeSet.build(task, eps)


# ----------------------------------------------------------------------------------------------- playing


def load_agent(over: dict):
    """A fresh module of the agent's code with the base PARAMS and ``over`` applied."""
    spec = importlib.util.spec_from_file_location(f"cand_{abs(hash(json.dumps(over, sort_keys=True)))}", CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PARAMS.update(BASE)
    mod.PARAMS.update(over)
    return mod


def play(task: str, ep: int, spec: tuple, over: dict) -> tuple[int, int, float, float]:
    """(episode, J in cents, mean and largest CPU seconds of the agent per week)."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow_agent.scoring import _world
    from shockbench_flow_agent.shim import AgentShim

    _task, entropy, regime, fq_replications, cache = spec
    inst, omega, marks, fallback = _world(task, entropy, ep, fq_replications, cache)
    cpu = []
    shim = AgentShim(load_agent(over).Agent, on_week=lambda _w, s: cpu.append(s))
    traj = rollout(inst, shim, omega, regime, 0, marks=marks, fallback=fallback)
    return ep, int(traj.J_cents), float(np.mean(cpu)), float(np.max(cpu))


class Trials:
    """The played sets of one search: a JSON line each, read back on a rerun."""

    def __init__(self, name: str):
        self.path = OUT / name / "trials.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.is_file() else []

    @staticmethod
    def key(set_name: str, over: dict, repeat: int) -> str:
        return json.dumps([set_name, dict(sorted(over.items())), repeat])

    def get(self, set_name: str, over: dict, repeat: int):
        k = self.key(set_name, over, repeat)
        return next((r for r in self.rows if r["key"] == k), None)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def run_set(trials: Trials, set_name: str, es, task: str, over: dict, repeat: int, jobs: int) -> dict:
    """Play ``over`` on the set ``es`` once (or read the earlier run): J per episode, RSS, CPU."""
    row = trials.get(set_name, over, repeat)
    if row is not None:
        return row
    t0 = time.time()
    eps = list(es.episodes)
    out = Parallel(n_jobs=jobs)(delayed(play)(task, ep, es._spec, over) for ep in eps)
    J = {str(ep): j for ep, j, _m, _x in out}
    row = {
        "key": Trials.key(set_name, over, repeat),
        "set": set_name,
        "over": over,
        "repeat": repeat,
        "J": J,
        "rss": es.rss([J[str(ep)] for ep in eps])["rss"],
        "cpu_mean": float(np.mean([m for _e, _j, m, _x in out])),
        "cpu_max": float(np.max([x for _e, _j, _m, x in out])),
        "seconds": round(time.time() - t0),
    }
    trials.add(row)
    print(
        f"  {set_name} {over or 'base'} r{repeat}: RSS {row['rss']:.4f}, CPU max {row['cpu_max']:.2f} s/week",
        flush=True,
    )
    return row


# ----------------------------------------------------------------------------------------------- paired RSS


def rss_of(refs: dict, J: dict, sel: list) -> float:
    """The board's RSS of the episodes ``sel`` (repeats allowed): per level mean saving over mean attainable saving."""
    num = den = 0.0
    for s, w in WEIGHTS.items():
        es_ = [e for e in sel if refs[e]["stratum"] == s]
        if not es_:
            continue
        num += w * np.mean([refs[e]["J_naive_cents"] - J[str(e)] for e in es_])
        den += w * np.mean([refs[e]["J_naive_cents"] - refs[e]["J_oracle_cents"] for e in es_])
    return num / den


def paired(es, Ja: dict, Jb: dict, draws: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """RSS(a) - RSS(b) on the same episodes and its 90 % bootstrap interval (resampled within harm levels)."""
    refs = {r["episode"]: r for r in es.references}
    eps = [e for e in es.episodes if refs[e]["J_oracle_cents"] is not None]
    by = {s: [e for e in eps if refs[e]["stratum"] == s] for s in WEIGHTS}
    rng = np.random.default_rng(seed)
    d = rss_of(refs, Ja, eps) - rss_of(refs, Jb, eps)
    bs = []
    for _ in range(draws):
        sel = [int(e) for s in WEIGHTS if by[s] for e in rng.choice(by[s], len(by[s]))]
        bs.append(rss_of(refs, Ja, sel) - rss_of(refs, Jb, sel))
    return d, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def mean_J(rows: list[dict]) -> dict:
    """Per-episode mean cost of several runs of the same point."""
    return {k: float(np.mean([r["J"][k] for r in rows])) for k in rows[0]["J"]}


# ----------------------------------------------------------------------------------------------- stages


def neighbours(name: str, value) -> list:
    vals = SPACE[name]
    if value not in vals:
        return [v for v in vals if v != value][:2]
    i = vals.index(value)
    return [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]


def search(trials: Trials, s1, jobs: int, rounds: int, repeats: int, min_gain: float, cpu_limit: float) -> dict:
    base_runs = [run_set(trials, "S1", s1, "small", {}, r, jobs) for r in range(repeats)]
    sigma = float(np.std([r["rss"] for r in base_runs], ddof=1)) if repeats > 1 else 0.0
    delta = max(min_gain, 2 * sigma)
    print(
        f"noise: base RSS {[round(r['rss'], 4) for r in base_runs]}, sigma {sigma:.4f} -> delta {delta:.4f}", flush=True
    )
    point, point_J = {}, mean_J(base_runs)
    start = load_agent({}).PARAMS  # the code's defaults with the base applied
    log = []
    for rnd in range(rounds):
        moved = False
        for name in SPACE:
            current = {**start, **point}[name]
            best = None
            for value in neighbours(name, current):
                over = {**point, name: value}
                row = run_set(trials, "S1", s1, "small", over, 0, jobs)
                if row["cpu_max"] > cpu_limit:
                    print(f"    {name}={value}: rejected, CPU {row['cpu_max']:.2f} s/week > {cpu_limit}", flush=True)
                    continue
                d, lo, hi = paired(s1, row["J"], point_J)
                print(f"    {name}={value}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
                if d >= delta and lo > 0 and (best is None or d > best[1]):
                    best = (value, d, row)
            if best is None:
                continue
            value, d, row = best
            over = {**point, name: value}
            again = run_set(trials, "S1", s1, "small", over, 1, jobs)  # a second run against the winner's curse
            J2 = mean_J([row, again])
            d2, lo2, hi2 = paired(s1, J2, point_J)
            if d2 >= delta and lo2 > 0:
                point, point_J, moved = over, J2, True
                log.append({"round": rnd, "param": name, "value": value, "gain": d2, "lo": lo2, "hi": hi2})
                print(f"  accepted {name}={value}: {d2:+.4f} [{lo2:+.4f}, {hi2:+.4f}] over 2 runs", flush=True)
            else:
                print(f"  {name}={value} did not hold on a second run: {d2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]", flush=True)
        if not moved:
            print(f"round {rnd}: no move, stop", flush=True)
            break
    return {"point": point, "params": {**BASE, **point}, "delta": delta, "sigma": sigma, "accepted": log}


def main(
    stage: str = "search",
    name: str = "night5",
    search_root: int = 20261008,
    confirm_root: int = 20261009,
    pool: int = 400,
    per: int = 12,
    rounds: int = 3,
    repeats: int = 3,
    min_gain: float = 0.004,
    cpu_limit: float = 1.5,
    full_episodes: int = 0,
    export_name: str = "mpc_tuned",
    jobs: int = 8,
):
    """Run one stage (sets, search, confirm, export) of the parameter search; see the module docstring."""
    best_path = OUT / name / "best.json"
    if stage == "sets":
        for root in (search_root, confirm_root):
            eps = stratified(root, pool, per, jobs)
            es = episode_set("small", root, eps)
            print(f"root {root}: {len(eps)} episodes, references ready for {len(es.references)}")
        return
    if stage == "search":
        s1 = episode_set("small", search_root, stratified(search_root, pool, per, jobs))
        result = search(Trials(name), s1, jobs, rounds, repeats, min_gain, cpu_limit)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(json.dumps(result, indent=1))
        print(f"best point {result['point'] or 'base'} -> {best_path}")
        return
    if stage == "confirm":
        point = json.loads(best_path.read_text())["point"]
        trials = Trials(name)
        sets = [("S2", "small", confirm_root, stratified(confirm_root, pool, per, jobs)), ("dev", "small", 0, "dev")]
        if full_episodes:
            from shockbench_flow_agent import EpisodeSet

            dev_full = list(EpisodeSet.build("full", "dev").episodes)[:full_episodes]
            sets.append((f"full{full_episodes}", "full", 0, dev_full))
        for set_name, task, root, eps in sets:
            es = episode_set(task, root, eps)
            a = run_set(trials, set_name, es, task, point, 0, jobs)
            b = run_set(trials, set_name, es, task, {}, 0, jobs)
            d, lo, hi = paired(es, a["J"], b["J"])
            print(
                f"{set_name}: best {a['rss']:.4f} vs base {b['rss']:.4f}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], "
                f"CPU max {a['cpu_max']:.2f} s/week"
            )
        return
    if stage == "export":
        params = json.loads(best_path.read_text())["params"]
        folder = ROOT / "agents" / export_name
        if folder.exists():
            raise ValueError(f"{folder} exists: pick another --export_name")
        folder.mkdir(parents=True)
        shutil.copy(CODE, folder / "agent.py")
        (folder / "params.json").write_text(json.dumps(params))
        print(f"written {folder}; check it with: uv run sbf check {export_name} --task=small (and --task=full)")
        return
    raise ValueError(f"unknown stage {stage!r}: sets, search, confirm or export")


if __name__ == "__main__":
    fire.Fire(main)

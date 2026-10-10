"""Gate 0, D1: where the tuned agent's cost goes, against the naive rule and the clairvoyant plan.

    uv run python scripts/research/gate0_cost_breakdown.py                        # pass 1's best, levels 1 and 2
    uv run python scripts/research/gate0_cost_breakdown.py --levels=all --workers=4
    uv run python scripts/research/gate0_cost_breakdown.py --params=path/to/params.json
    uv run python scripts/research/gate0_cost_breakdown.py --smoke                # 2 episodes: plumbing only

Episodes: Small, root 202610, episodes 0..63 (the tuning set), those of the chosen harm levels; excluded ones are
skipped. Per episode, three breakdowns over the simulator's eight cost components:
- **agent**: ``agents/compact_hierarchical`` with ``--params`` (default: the Optuna study's
  ``best_small_params.json``), played as ``EpisodeSet.play`` plays a factory: the same policy seed, and the task's
  CPU meter (2 s; a week over it is played by the naive rule, as on the server);
- **naive**: the reference's anchor policy (``prediction_free`` regime) played again; its cost is checked against the
  cached reference;
- **clairvoyant**: the reference LP (``build_lp`` on the true marks) solved again, its plan split by ``lp_costs``;
  its cost is checked against the cached reference.

The agents and the naive rule play in one worker pool and the clairvoyant LPs solve in another, so no LP solver
thread runs in a process whose agent is being metered.

**Groups.** Loss = shortage + shed. Friction = freight + war_risk + tariff + holding + queue_holding + disposal. End
stock = J - (the sum of the components): the salvage credit and the initial-state constants. In a level,
1 - RSS = sum(agent - clairvoyant) / sum(naive - clairvoyant), and it splits exactly over the components and groups:
the "RSS points" columns.

**Week classes**, from the true marks. An onset is a week where a strait's open fraction, an edge's capacity or a
supply falls by half or more from the week before, a grid's deliverable or a fab or OSAT factor by more than 5 %, or a
prohibition starts; consecutive onset weeks count as one.
- shock: an onset week and the ``--shock_weeks`` - 1 weeks after it (default 4 weeks in all);
- pre: the ``--pre_weeks`` weeks before an onset (default 4; where pre-positioning would act);
- calm: the rest.

A week's gap is the (agent - clairvoyant) cost booked in that week. A cost can be booked after its cause (a shortage
follows a missed shipment), so read the split as indicative.

Outputs, in ``outputs/research/gate0_cost_breakdown/<date_time>/``:
- ``summary.md``: the tables (also printed); this is the file to paste back;
- ``episodes.csv``: one row per (episode, policy) with every component;
- ``results.json``: everything, with the weekly series.
"""

import csv
import json
import sys
import time
from pathlib import Path

import fire
import numpy as np
from common import run_dir, world
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agents" / "compact_hierarchical"
DEFAULT_PARAMS = ROOT / "outputs" / "research" / "tune_compact_optuna" / "compact_small" / "best_small_params.json"

TASK = "small"
EPISODES = 64
ENTROPY = 202610
COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
LOSS = ("shortage", "shed")
FRICTION = ("freight", "war_risk", "tariff", "holding", "queue_holding", "disposal")
POLICIES = ("agent", "naive", "clairvoyant")
# a disruption's onset: these marks falling by more than the share given from one week to the next
TRIGGERS = {"o": 0.5, "u": 0.5, "supply": 0.5, "G_bar": 0.05, "R": 0.05, "R_osat": 0.05, "alpha_bar": 0.05}
SEARCH_KNOBS = (
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

_AGENT_CLASS = None


def _agent_class():
    global _AGENT_CLASS
    if _AGENT_CLASS is None:
        from shockbench_flow_agent.shim import load_agent_class

        _AGENT_CLASS = load_agent_class(AGENT_DIR, "gate0_agent")
    return _AGENT_CLASS


def weekly(records) -> np.ndarray:
    """(T, 8) USD per week and component, from a trajectory's records or a plan's ``lp_costs`` weeks."""
    rows = [r.costs.as_dict() if hasattr(r, "costs") else r.as_dict() for r in records]
    return np.array([[row[k] for k in COMPONENTS] for row in rows], dtype=float)


def onsets(marks) -> dict[str, list[int]]:
    """0-based week indices where a disruption starts, by kind (module docstring); consecutive weeks are one onset."""
    T = int(marks.T)
    out = {}
    for name, drop in TRIGGERS.items():
        a = np.nan_to_num(np.asarray(getattr(marks, name), dtype=float), posinf=1e30).reshape(T, -1)
        hit = np.zeros(T, dtype=bool)
        hit[1:] = (a[1:] < a[:-1] * (1.0 - drop) - 1e-9).any(axis=1)
        out[name] = hit
    p = np.asarray(marks.prohibited, dtype=bool).reshape(T, -1)
    out["prohibited"] = np.zeros(T, dtype=bool)
    out["prohibited"][1:] = (p[1:] & ~p[:-1]).any(axis=1)
    starts = {}
    for name, hit in out.items():
        first = hit & ~np.concatenate([[False], hit[:-1]])
        starts[name] = np.flatnonzero(first).tolist()
    return starts


def merged(starts: dict[str, list[int]]) -> list[int]:
    """All kinds' onsets, consecutive weeks merged into one."""
    weeks = sorted({w for ws in starts.values() for w in ws})
    return [w for i, w in enumerate(weeks) if i == 0 or w > weeks[i - 1] + 1]


def week_classes(T: int, starts: list[int], shock: int, pre: int) -> np.ndarray:
    """Per week: 'shock' (an onset and the shock - 1 weeks after), 'pre' (the pre weeks before), else 'calm'."""
    cls = np.array(["calm"] * T, dtype=object)
    for o in starts:
        cls[max(0, o - pre) : o] = "pre"
    for o in starts:  # after the pre weeks, so a shock week stays a shock week
        cls[o : min(T, o + shock)] = "shock"
    return cls


def play_agent_and_naive(n: int, params: dict) -> dict:
    """The agent's and the naive rule's weekly components on episode n, with their J and the agent's fallbacks."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS, anchor_policy
    from shockbench_flow_agent.local_eval import ANCHOR_REGIME
    from shockbench_flow_agent.scoring import CPU_BUDGET_S, _metered_shim

    inst, omega, marks, fallback, seed = world(TASK, ENTROPY, n)
    cls, made = _agent_class(), []

    def factory(config):
        agent = cls(config, params=params)
        made.append(agent)
        return agent

    shim = _metered_shim(factory, CPU_BUDGET_S[TASK])
    with without_secret_like():
        traj = rollout(inst, shim, omega, "standard", seed, marks=marks, fallback=fallback)
    _inst, gen = task_generator(TASK)
    naive = rollout(
        inst, anchor_policy(inst, gen, REPLICATIONS), omega, ANCHOR_REGIME, seed, marks=marks, fallback=fallback
    )
    log = made[0].log if made else []
    return {
        "episode": n,
        "agent": {"J_cents": int(traj.J_cents), "weeks": weekly(traj.records).tolist()},
        "naive": {"J_cents": int(naive.J_cents), "weeks": weekly(naive.records).tolist()},
        "onsets": onsets(marks),
        "agent_own_fallback_weeks": sum(1 for w in log if not w["plan"]),
        "agent_over_budget_weeks": len(shim.cpu_weeks),
    }


def solve_clairvoyant(n: int) -> dict:
    """The reference LP's plan on episode n, split into weekly components."""
    from shockbench_flow.evaluation.results import oracle_optimal
    from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, lp_costs, solve_oracle

    inst, _omega, marks, _fb, _seed = world(TASK, ENTROPY, n)
    model = build_lp(inst, marks)
    res = solve_oracle(model, method=ORACLE_METHOD)
    if not oracle_optimal(res):
        return {"episode": n, "clairvoyant": None}
    weeks, _salvage = lp_costs(model, res.x)
    return {"episode": n, "clairvoyant": {"J_cents": int(res.J_cents), "weeks": weekly(weeks).tolist()}}


def _usd_b(x: float) -> str:
    return f"{x / 1e9:,.2f}"


def summarize(rows: list[dict], levels: list[int], header: list[str], shock: int, pre: int) -> str:
    """The markdown tables of summary.md."""
    out = list(header)
    for level in levels:
        group = [r for r in rows if r["level"] == level]
        if not group:
            continue
        n = len(group)
        tot = {p: np.array([np.array(r[p]["weeks"]).sum(axis=0) for r in group]) for p in POLICIES}  # (n, 8)
        J = {p: np.array([r[p]["J_cents"] / 100.0 for r in group]) for p in POLICIES}
        end = {p: J[p] - tot[p].sum(axis=1) for p in POLICIES}  # salvage credit and constants
        attainable = float((J["naive"] - J["clairvoyant"]).sum())
        gap_total = float((J["agent"] - J["clairvoyant"]).sum())
        rss = 1.0 - gap_total / attainable if attainable else float("nan")
        mean = {p: tot[p].mean(axis=0) for p in POLICIES}
        label = "all episodes" if level == 0 else f"Level {level}"
        out += [
            "",
            f"## {label}: {n} episodes, RSS {rss:.4f} (1 - RSS = {1 - rss:.4f})",
            "",
            "Mean USD billions per episode. RSS points: the component's share of 1 - RSS (they sum to 1 - RSS).",
            "",
            "| Component | Agent | Naive | Clairvoyant | Gap (agent - clairvoyant) | RSS points |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for i, k in enumerate(COMPONENTS):
            gap = mean["agent"][i] - mean["clairvoyant"][i]
            pts = (tot["agent"][:, i] - tot["clairvoyant"][:, i]).sum() / attainable
            out.append(
                f"| {k} | {_usd_b(mean['agent'][i])} | {_usd_b(mean['naive'][i])} | {_usd_b(mean['clairvoyant'][i])} "
                f"| {_usd_b(gap)} | {pts:+.4f} |"
            )
        gap_end = float((end["agent"] - end["clairvoyant"]).mean())
        out.append(
            f"| end stock (J - components) | {_usd_b(end['agent'].mean())} | {_usd_b(end['naive'].mean())} "
            f"| {_usd_b(end['clairvoyant'].mean())} | {_usd_b(gap_end)} "
            f"| {(end['agent'] - end['clairvoyant']).sum() / attainable:+.4f} |"
        )
        out.append(
            f"| **J** | {_usd_b(J['agent'].mean())} | {_usd_b(J['naive'].mean())} | {_usd_b(J['clairvoyant'].mean())} "
            f"| {_usd_b(gap_total / n)} | {gap_total / attainable:+.4f} |"
        )
        idx = {k: COMPONENTS.index(k) for k in COMPONENTS}
        out += ["", "| Group | Gap (agent - clairvoyant) | RSS points |", "| --- | ---: | ---: |"]
        for name, keys in (
            ("loss: shortage + shed", LOSS),
            ("friction: freight, war risk, tariff, holding, queue, disposal", FRICTION),
        ):
            g = sum((tot["agent"][:, idx[k]] - tot["clairvoyant"][:, idx[k]]).sum() for k in keys)
            out.append(f"| {name} | {_usd_b(g / n)} | {g / attainable:+.4f} |")
        out.append(f"| end stock | {_usd_b(gap_end)} | {(end['agent'] - end['clairvoyant']).sum() / attainable:+.4f} |")
        # the gap by week class
        by = {c: {"weeks": 0, "loss": 0.0, "friction": 0.0} for c in ("calm", "pre", "shock")}
        for r in group:
            a, c = np.array(r["agent"]["weeks"]), np.array(r["clairvoyant"]["weeks"])
            classes = week_classes(len(a), merged(r["onsets"]), shock, pre)
            for w, cl in enumerate(classes):
                by[cl]["weeks"] += 1
                by[cl]["loss"] += sum(a[w, idx[k]] - c[w, idx[k]] for k in LOSS)
                by[cl]["friction"] += sum(a[w, idx[k]] - c[w, idx[k]] for k in FRICTION)
        out += [
            "",
            f"Onsets per episode: {np.mean([len(merged(r['onsets'])) for r in group]):.2f} merged ("
            + ", ".join(f"{k} {np.mean([len(r['onsets'][k]) for r in group]):.2f}" for k in group[0]["onsets"])
            + f"). Gap booked by week class (shock: onset + {shock - 1} weeks; pre: {pre} weeks before an onset):",
            "",
            "| Week class | Weeks | Share of weeks | Loss gap | Friction gap | RSS points (loss + friction) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        weeks_all = sum(v["weeks"] for v in by.values())
        for cl, v in by.items():
            out.append(
                f"| {cl} | {v['weeks']} | {v['weeks'] / weeks_all:.0%} | {_usd_b(v['loss'] / n)} "
                f"| {_usd_b(v['friction'] / n)} | {(v['loss'] + v['friction']) / attainable:+.4f} |"
            )
    return "\n".join(out) + "\n"


def main(
    params: str = "", levels="1,2", workers: int = 4, shock_weeks: int = 4, pre_weeks: int = 4, smoke: bool = False
) -> None:
    """Break the tuned agent's, naive's and the clairvoyant plan's cost down by component, level and week class.

    Args:
        params: a complete knob file for the agent (default: the Optuna study's best_small_params.json).
        levels: harm levels to break down, "1,2" (default) or "all".
        workers: processes per phase (the agents and naive, then the clairvoyant LPs).
        shock_weeks: weeks from an onset counted as its shock (the onset week included).
        pre_weeks: weeks before an onset counted as its run-up.
        smoke: episodes 0 and 1 with quick references (no harm levels): checks the plumbing only.

    """
    from shockbench_flow_agent import EpisodeSet

    sys.path.insert(0, str(AGENT_DIR))
    from compact_hier import config as agent_config

    path = Path(params) if params else DEFAULT_PARAMS
    if not path.is_file():
        sys.exit(f"no params file at {path}: pass --params=... (the Optuna run writes best_small_params.json)")
    knobs = agent_config.load(AGENT_DIR, environ={"SBF_PARAMS_FILE": str(path.resolve())})
    if smoke:
        es = EpisodeSet.build(TASK, [0, 1], entropy=ENTROPY, quick=True, n_jobs=workers)
        wanted = [0]
    else:
        es = EpisodeSet.build(TASK, EPISODES, entropy=ENTROPY, n_jobs=workers)
        wanted = [1, 2, 3, 4] if str(levels) == "all" else [int(x) for x in str(levels).strip("()[] ").split(",") if x]
    refs = {
        int(r["episode"]): r
        for r in es.references
        if r.get("excluded") is None and r.get("J_oracle_cents") is not None and (smoke or r["stratum"] in wanted)
    }
    ns = sorted(refs)
    run = run_dir("gate0_cost_breakdown")
    print(f"{len(ns)} episodes of {TASK} root {ENTROPY} (levels {wanted if not smoke else 'smoke'}); run folder {run}")
    start = time.perf_counter()
    played = Parallel(n_jobs=workers)(delayed(play_agent_and_naive)(n, knobs) for n in ns)
    print(f"agents and naive played ({time.perf_counter() - start:.0f} s); solving the clairvoyant LPs ...", flush=True)
    solved = Parallel(n_jobs=workers, backend="multiprocessing")(delayed(solve_clairvoyant)(n) for n in ns)
    by_n = {s["episode"]: s for s in solved}

    rows, naive_ok, oracle_ok, skipped = [], 0, 0, []
    for p in played:
        n = p["episode"]
        if by_n[n]["clairvoyant"] is None:
            skipped.append(n)
            continue
        ref = refs[n]
        row = {**p, "clairvoyant": by_n[n]["clairvoyant"], "level": 0 if smoke else int(ref["stratum"])}
        naive_ok += int(smoke or row["naive"]["J_cents"] == int(ref["J_naive_cents"]))
        oracle_ok += int(smoke or row["clairvoyant"]["J_cents"] == int(ref["J_oracle_cents"]))
        rows.append(row)

    header = [
        "# Gate 0, D1: cost breakdown of the tuned agent",
        "",
        f"- Params: `{path}`",
        "- Search knobs: " + ", ".join(f"{k} {knobs[k]:.4g}" for k in SEARCH_KNOBS),
        f"- Episodes: {len(rows)} of Small root {ENTROPY} (0..{EPISODES - 1}), levels {wanted}"
        + (f"; skipped (clairvoyant not optimal): {skipped}" if skipped else ""),
        f"- Reference checks: naive J equal to the cached reference on {naive_ok}/{len(rows)} episodes, "
        f"clairvoyant J on {oracle_ok}/{len(rows)}",
        f"- Agent weeks over the 2 s budget: {sum(r['agent_over_budget_weeks'] for r in rows)}; "
        f"weeks of the agent's own fallback: {sum(r['agent_own_fallback_weeks'] for r in rows)}",
    ]
    text = summarize(rows, [0] if smoke else wanted, header, shock_weeks, pre_weeks)
    (run / "summary.md").write_text(text)
    with open(run / "episodes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "level", "policy", "J_usd", *COMPONENTS, "onsets"])
        for r in rows:
            for pol in POLICIES:
                tot = np.array(r[pol]["weeks"]).sum(axis=0)
                writer.writerow(
                    [r["episode"], r["level"], pol, r[pol]["J_cents"] / 100.0, *tot.round(2), len(merged(r["onsets"]))]
                )
    (run / "results.json").write_text(json.dumps({"params": knobs, "rows": rows}, default=float))
    print(text)
    print(f"written {run / 'summary.md'} (paste this file back), episodes.csv, results.json")


if __name__ == "__main__":
    fire.Fire(main)

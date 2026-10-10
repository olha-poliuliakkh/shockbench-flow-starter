"""Step 1 of the chip-chain roadmap: where the agent's chip supply falls short of the clairvoyant plan, link by link.

    uv run python scripts/research/chip_chain_diagnostic.py                         # levels 1 and 2, params.json
    uv run python scripts/research/chip_chain_diagnostic.py --levels=all --workers=4
    uv run python scripts/research/chip_chain_diagnostic.py --set=rule_chips=1      # the same with knobs changed
    uv run python scripts/research/chip_chain_diagnostic.py --smoke                 # 2 episodes: plumbing only

Episodes: Small, root 202610, episodes 0..63 (the tuning set, Gate 0's), those of the chosen harm levels. The agent is
``agents/compact_hierarchical`` with ``--params`` (default: its ``params.json``) and ``--set`` overrides, played as
``EpisodeSet.play`` plays a factory (policy seed, the 2 s CPU meter). The naive rule is the reference's anchor policy;
the clairvoyant plan is the reference LP solved again (Gate 0 checked both against the cached references, 64/64).

The simulator's rules (``sim.py`` step 7, ``production.py``), which the attribution follows exactly:
- a fab starts p-hat = min(alpha-bar R cap0, W) lots, W its wafers on hand, then p = p-hat x its energy share:
  a grid that sheds any base load gives its fabs no energy (base load first); otherwise the fabs share the headroom
  G_av - y-bar in proportion to their requests e p-hat / R;
- an OSAT packages min(raw, thr x R_osat), pro rata over its raw chips;
- a sink's shortage is lost + backlog, charged pi per unit and week.

**Tables** (means per episode, by level):
1. Chip shortage by product: cost for the agent, naive and the clairvoyant plan, the gap, and its RSS points (its
   share of 1 - RSS, as Gate 0).
2. Mass balance: wafers lifted, lots started, chips packaged and delivered, wafers and chips disposed above
   storage, demand, shortage unit-weeks.
3. Timing: how far the agent's cumulative lot starts, packaging and deliveries run behind the clairvoyant plan's, in
   unit-weeks (a unit one week late counts 1). Equal totals with a large lag mean the same chips, later.
4. Fabs: capacity-weeks used and unused for lack of wafers, and wafer-weeks idle on hand because the fab got no
   energy: its grid shed base load (base load first), or the headroom was short. Idle wafers can start a later week,
   so these are delays, not lost lots. The weeks the fab's grid shed, for the agent and the clairvoyant plan.
5. The agent against the clairvoyant plan, fab-week by fab-week: lots it started fewer (by the cause above) and more.
   Lots are also valued at their chip's shortage penalty pi, an upper bound of what a lot is worth at a sink.
6. The agent's plan against its execution: the LP's first-week lot starts and packaging against what the simulator
   did. More than planned is the simulator starting wafers (or packaging raw chips) the plan meant to hold.
7. OSATs: packaging against throughput, and against the clairvoyant plan (short of raw chips, or at throughput).
8. Sinks: the shortage cost split by whether that chip sat in stock elsewhere (OSATs, hubs, strait queues for the
   agent; non-sink stock for the plan) at the end of the week, or was not on hand anywhere (in transit or not made).

**Reading.** The summary ends with what each outcome points to.

Outputs, in ``outputs/research/chip_chain_diagnostic/<date_time>/``:
- ``summary.md``: the tables (also printed); this is the file to paste back;
- ``fabs.csv``: one row per (episode, fab) with the agent's and the clairvoyant plan's ledgers;
- ``results.json``: the per-episode totals.
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

TASK = "small"
EPISODES = 64
ENTROPY = 202610
TOL = 1e-6

_CACHE = {}

READING = [
    "- **Table 1**: which chip carries the gap. **Table 2**: whether the agent makes fewer chips (lots, packaging) or",
    "  makes them and fails to deliver them (shortage with packaging near the plan's). **Table 3**: whether it is late",
    "  (equal totals with a large lag: the same chips, later).",
    "- **Tables 4 and 5, no wafers** dominant: wafer supply and positioning (wafer floors at fabs, the base-stock",
    "  rule, lift timing). **Grid shedding** dominant, with the agent's grids shedding in weeks the plan's do not:",
    "  base load first in execution (a no-shed reserve at fab grids, the two-pass fix). **Headroom**: fab-grid",
    "  generation.",
    "- **Table 6, beyond the plan** large: the LP plans to hold wafers or raw chips the simulator starts anyway",
    "  (planning-rule prices on held wafers and raw chips). **Below the plan** large: the LP's production model is",
    "  optimistic.",
    "- **Table 7, no raw chips** dominant: fab output or raw-chip routing to OSATs. **Raw chips on hand**: OSAT",
    "  throughput or the pro-rata split.",
    "- **Table 8, in stock upstream** larger for the agent than for the plan: distribution (chip routing, lead times,",
    "  sink stocks). **Not on hand**: production, or chips too long in transit.",
]


# ----- the chain's static structure ---------------------------------------------------------------------------------
class Chain:
    """Fabs, OSAT (raw -> packaged) pairs, demands and the stock slots each link reads, from the instance."""

    def __init__(self, inst):
        ids = [c.id for c in inst.commodities]
        self.wafer = ids.index("wafer")
        self.F, self.G, self.O = len(inst.fabs), len(inst.grids), len(inst.osats)
        fabs = [inst.nodes[f].fab for f in inst.fabs]
        self.cap0 = np.array([f.cap0 for f in fabs], dtype=float)
        self.e = np.array([f.e for f in fabs], dtype=float)
        self.wafer_slot = [inst.slot_index[(f, fab.input)] for f, fab in zip(inst.fabs, fabs)]
        self.fab_grid = np.full(self.F, -1)
        for gi, members in enumerate(inst.grid_fabs):
            for fi in members:
                self.fab_grid[fi] = gi
        self.pairs = []  # (osat ordinal, raw k, packaged k, raw slot)
        for oo, o in enumerate(inst.osats):
            for raw, pk in inst.nodes[o].osat.packages.items():
                self.pairs.append((oo, int(raw), int(pk), inst.slot_index[(o, int(raw))]))
        self.thr = np.array([inst.nodes[o].osat.thr for o in inst.osats], dtype=float)
        self.demands = [(int(d.k), float(d.pi)) for d in inst.demands]
        self.pi = np.array([pi for _k, pi in self.demands])
        self.packaged = sorted({pk for _o, _r, pk, _s in self.pairs} | {k for k, _pi in self.demands})
        self.names = {k: ids[k] for k in range(len(ids))}
        sinks = set(inst.sinks)
        self.upstream_slots = {
            k: [s for (node, kk), s in inst.slot_index.items() if kk == k and node not in sinks] for k in self.packaged
        }
        raw_ks = sorted({raw for _o, raw, _pk, _s in self.pairs})
        self.disposal_groups = {  # stock slots by stage of the chain, for the disposal rows of the mass balance
            "wafers": [s for (_n, k), s in inst.slot_index.items() if k == self.wafer],
            "raw chips": [s for (_n, k), s in inst.slot_index.items() if k in raw_ks],
            "packaged chips": [s for (_n, k), s in inst.slot_index.items() if k in self.packaged],
        }
        self.packaged_slots = self.disposal_groups["packaged chips"]
        self.slot_node = {s: inst.nodes[node].id for (node, _k), s in inst.slot_index.items()}
        supply = set(inst.supply_nodes)
        self.wafer_supply = [s for (node, k), s in inst.slot_index.items() if k == self.wafer and node in supply]
        # a lot's and a package's worth at a sink: the largest pi of the chip it becomes
        pi_k = {}
        for k, pi in self.demands:
            pi_k[k] = max(pi_k.get(k, 0.0), pi)
        raw_to_pk = {raw: pk for _o, raw, pk, _s in self.pairs}
        self.lot_value = np.array([pi_k.get(raw_to_pk.get(f.product, -1), 0.0) for f in fabs])
        self.pair_value = np.array([pi_k.get(pk, 0.0) for _o, _r, pk, _s in self.pairs])


def chain(inst) -> Chain:
    if "chain" not in _CACHE:
        _CACHE["chain"] = Chain(inst)
    return _CACHE["chain"]


def capacities(marks, ch: Chain, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, F) fab capacity alpha-bar R cap0 and (T, O) OSAT throughput thr R_osat, from the true marks."""
    cap = np.asarray(marks.alpha_bar[:T], dtype=float) * np.asarray(marks.R[:T], dtype=float) * ch.cap0
    thr = np.asarray(marks.R_osat[:T], dtype=float) * ch.thr
    return cap, thr


def executed(records, marks, ch: Chain) -> dict:
    """A played trajectory's chain, week by week (arrays, lists for JSON)."""
    T = len(records)
    lots = np.array([r.lots_started for r in records], dtype=float)
    stock = np.array([r.stock for r in records], dtype=float)
    disp = np.array([r.disposal for r in records], dtype=float)
    shed = np.array([r.shed for r in records], dtype=float)
    W = stock[:, ch.wafer_slot] + lots + disp[:, ch.wafer_slot]  # wafers on hand when the fab started
    fab_shed = np.where(ch.fab_grid >= 0, shed[:, np.maximum(ch.fab_grid, 0)], 0.0)
    pkg = np.array([[r.packaged.get((oo, pk), 0.0) for oo, _r, pk, _s in ch.pairs] for r in records], dtype=float)
    raw_slots = [s for _o, _r, _pk, s in ch.pairs]
    raw = stock[:, raw_slots] + pkg + disp[:, raw_slots]  # raw chips on hand when the OSAT packaged
    cap, thr = capacities(marks, ch, T)
    return {
        "lots": lots,
        "W": W,
        "fab_shed": fab_shed,
        "cap": cap,
        "pkg": pkg,
        "raw": raw,
        "thr": thr,
        "short": np.array([r.lost + r.backlog for r in records], dtype=float),
        "demand": np.array([r.demand for r in records], dtype=float),
        "served": np.array([r.served for r in records], dtype=float),
        "disposed": np.array([disp[:, slots].sum(axis=1) for slots in ch.disposal_groups.values()]).T,
        "disposed_packaged": disp[:, ch.packaged_slots].sum(axis=0),  # per packaged-chip slot, over the episode
        "upstream": np.array([stock[:, ch.upstream_slots[k]].sum(axis=1) for k in ch.packaged]).T,
        "wafer_lift": np.array([r.lift for r in records], dtype=float)[:, ch.wafer_supply].sum(axis=1),
    }


def _arrays(d: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ----- phase 1: the agent and the naive rule ------------------------------------------------------------------------
def _agent_class():
    if "agent" not in _CACHE:
        from shockbench_flow_agent.shim import load_agent_class

        _CACHE["agent"] = load_agent_class(AGENT_DIR, "chip_chain_agent")
    return _CACHE["agent"]


def play(n: int, params: dict) -> dict:
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS, anchor_policy
    from shockbench_flow_agent.local_eval import ANCHOR_REGIME
    from shockbench_flow_agent.scoring import CPU_BUDGET_S, _metered_shim

    inst, omega, marks, fallback, seed = world(TASK, ENTROPY, n)
    ch = chain(inst)
    cls, made, plan_p, plan_q = _agent_class(), [], [], []

    class Recording(cls):  # the LP's first-week lot starts and packaging, as planned before the week runs
        def act(self, observation):
            action = super().act(observation)
            plan = self.lp.last_plan if self.log and self.log[-1]["plan"] else None
            if plan is None:
                plan_p.append([np.nan] * ch.F)
                plan_q.append([np.nan] * len(ch.pairs))
                return action
            x, off, net = plan["x"], plan["off"], self.net
            plan_p.append([float(v) for v in x[off["p"] : off["p"] + len(net.fabs)]])
            q = {}
            for m, (_j_raw, j_pk, _tau, oo) in enumerate(net.osats):
                if j_pk >= 0:
                    q[(oo, int(net.stock[j_pk][1]))] = float(x[off["q"] + m])
            rule = net.rule_on  # with the rule on the LP plans no packaging (the rule ships the chips)
            plan_q.append([np.nan if rule else q.get((oo, pk), 0.0) for oo, _r, pk, _s in ch.pairs])
            return action

    def factory(config):
        agent = Recording(config, params=params)
        made.append(agent)
        return agent

    shim = _metered_shim(factory, CPU_BUDGET_S[TASK])
    with without_secret_like():
        traj = rollout(inst, shim, omega, "standard", seed, marks=marks, fallback=fallback)
    _inst, gen = task_generator(TASK)
    anchor = anchor_policy(inst, gen, REPLICATIONS)
    naive = rollout(inst, anchor, omega, ANCHOR_REGIME, seed, marks=marks, fallback=fallback)
    log = made[0].log if made else []
    agent = executed(traj.records, marks, ch) | {"plan_p": np.array(plan_p), "plan_q": np.array(plan_q)}
    return {
        "episode": n,
        "agent": _arrays(agent) | {"J_cents": int(traj.J_cents)},
        "naive": _arrays(executed(naive.records, marks, ch)) | {"J_cents": int(naive.J_cents)},
        "agent_fallback_weeks": sum(1 for w in log if not w["plan"]),
        "agent_over_budget_weeks": len(shim.cpu_weeks),
    }


# ----- phase 2: the clairvoyant plan --------------------------------------------------------------------------------
def solve_clairvoyant(n: int) -> dict:
    from shockbench_flow.evaluation.results import oracle_optimal
    from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, solve_oracle

    inst, _omega, marks, _fb, _seed = world(TASK, ENTROPY, n)
    ch = chain(inst)
    model = build_lp(inst, marks)
    res = solve_oracle(model, method=ORACLE_METHOD)
    if not oracle_optimal(res):
        return {"episode": n, "clairvoyant": None}
    x, idx, T = np.asarray(res.x), model.index, int(inst.T)

    def get(key):
        j = idx.get(key)
        return float(x[j]) if j is not None else 0.0

    weeks = range(1, T + 1)
    cap, thr = capacities(marks, ch, T)
    plan = {
        "lots": np.array([[get(("p", t, f)) for f in range(ch.F)] for t in weeks]),
        "cap": cap,
        "pkg": np.array([[get(("xi", t, oo, pk)) for oo, _r, pk, _s in ch.pairs] for t in weeks]),
        "thr": thr,
        "short": np.array([[get(("U", t, d)) + get(("B", t, d)) for d in range(len(ch.demands))] for t in weeks]),
        "demand": np.array([[float(marks.demand[t - 1][d]) for d in range(len(ch.demands))] for t in weeks]),
        "wafer_lift": np.array([sum(get(("lift", t, s)) for s in ch.wafer_supply) for t in weeks]),
        "served": np.array([[get(("D", t, d)) for d in range(len(ch.demands))] for t in weeks]),
        "disposed": np.array(
            [[sum(get(("O", t, s)) for s in slots) for slots in ch.disposal_groups.values()] for t in weeks]
        ),
        "disposed_packaged": np.array([sum(get(("O", t, s)) for t in weeks) for s in ch.packaged_slots]),
        "upstream": np.array(
            [[sum(get(("I", t, s)) for s in ch.upstream_slots[k]) for k in ch.packaged] for t in weeks]
        ),
        "fab_shed": np.array([[get(("ysh", t, int(g))) if g >= 0 else 0.0 for g in ch.fab_grid] for t in weeks]),
    }
    return {"episode": n, "clairvoyant": _arrays(plan) | {"J_cents": int(res.J_cents)}}


# ----- the ledgers --------------------------------------------------------------------------------------------------
def fab_losses(a: dict) -> dict:
    """The agent's (or naive's) lots lost against capacity: no wafers, energy cut by shedding, energy headroom."""
    cap, W, lots, shed = a["cap"], a["W"], a["lots"], a["fab_shed"] > TOL
    phat = np.minimum(cap, W)
    energy = np.maximum(phat - lots, 0.0)
    return {
        "cap": cap.sum(axis=0),
        "lots": lots.sum(axis=0),
        "no_wafers": np.maximum(cap - W, 0.0).sum(axis=0),
        "energy_shed": np.where(shed, energy, 0.0).sum(axis=0),
        "energy_headroom": np.where(shed, 0.0, energy).sum(axis=0),
        "weeks_no_wafers": ((cap - W) > TOL * np.maximum(cap, 1.0)).sum(axis=0),
        "weeks_shed_cut": (shed & (energy > TOL)).sum(axis=0),
    }


def lag(a: np.ndarray, c: np.ndarray) -> float:
    """Unit-weeks the cumulative series ``a`` runs behind ``c`` (both (T, n)): the sum over weeks of the gap between
    their running totals, positive when ``a`` is behind."""
    return float((np.cumsum(c.sum(axis=1)) - np.cumsum(a.sum(axis=1))).sum())


def lot_gap(a: dict, c: dict) -> dict:
    """Per fab: lots the agent started fewer than the clairvoyant plan, by cause, and more."""
    g = c["lots"] - a["lots"]
    short = np.maximum(g, 0.0)
    wafers = np.minimum(short, np.maximum(c["lots"] - a["W"], 0.0))
    energy = short - wafers  # with W >= the plan's lots, the shortfall is the energy share
    shed = a["fab_shed"] > TOL
    return {
        "fewer": short.sum(axis=0),
        "fewer_no_wafers": wafers.sum(axis=0),
        "fewer_energy_shed": np.where(shed, energy, 0.0).sum(axis=0),
        "fewer_energy_headroom": np.where(shed, 0.0, energy).sum(axis=0),
        "more": np.maximum(-g, 0.0).sum(axis=0),
    }


def plan_fidelity(a: dict) -> dict:
    """The agent's first-week plan against the simulator: lots and packaging above and below the plan."""
    p, lots, W = a["plan_p"], a["lots"], a["W"]
    ok = ~np.isnan(p).any(axis=1)
    p, lots, W, shed = p[ok], lots[ok], W[ok], (a["fab_shed"] > TOL)[ok]
    under = np.maximum(p - lots, 0.0)
    wafers = np.minimum(under, np.maximum(p - W, 0.0))
    out = {
        "weeks": int(ok.sum()),
        "planned_lots": float(p.sum()),
        "started_lots": float(lots.sum()),
        "more_than_planned": float(np.maximum(lots - p, 0.0).sum()),
        "less_no_wafers": float(wafers.sum()),
        "less_energy_shed": float(np.where(shed, under - wafers, 0.0).sum()),
        "less_energy_headroom": float(np.where(shed, 0.0, under - wafers).sum()),
    }
    q = a["plan_q"]
    okq = ~np.isnan(q).any(axis=1) if q.size else np.zeros(0, dtype=bool)
    if okq.any():
        q, pkg = q[okq], a["pkg"][okq]
        out |= {
            "planned_packaging": float(q.sum()),
            "packaged": float(pkg.sum()),
            "packaged_more": float(np.maximum(pkg - q, 0.0).sum()),
            "packaged_less": float(np.maximum(q - pkg, 0.0).sum()),
        }
    return out


def osat_ledger(a: dict, c: dict, ch: Chain) -> dict:
    """OSAT packaging: against throughput (weeks short of raw chips) and against the clairvoyant plan."""
    oo = np.array([o for o, _r, _pk, _s in ch.pairs])
    raw_tot = np.zeros((len(a["pkg"]), ch.O))
    for i, o in enumerate(oo):
        raw_tot[:, o] += a["raw"][:, i]
    starved = raw_tot < a["thr"] - TOL * np.maximum(a["thr"], 1.0)
    g = c["pkg"] - a["pkg"]
    short = np.maximum(g, 0.0)
    no_raw = np.minimum(short, np.maximum(c["pkg"] - a["raw"], 0.0))
    return {
        "thr": a["thr"].sum(),
        "packaged": a["pkg"].sum(),
        "packaged_clairvoyant": c["pkg"].sum(),
        "osat_weeks_short_of_raw": int(starved.sum()),
        "osat_weeks": int(starved.size),
        "fewer": short.sum(),
        "fewer_no_raw": no_raw.sum(),
        "fewer_at_throughput": (short - no_raw).sum(),
        "more": np.maximum(-g, 0.0).sum(),
        "fewer_value": (short * ch.pair_value).sum(),
        "fewer_no_raw_value": (no_raw * ch.pair_value).sum(),
    }


def sink_ledger(a: dict, ch: Chain) -> dict:
    """The agent's shortage cost per chip, split by whether the chip sat in stock elsewhere that week."""
    out = {}
    for i, k in enumerate(ch.packaged):
        ds = [d for d, (kk, _pi) in enumerate(ch.demands) if kk == k]
        units = a["short"][:, ds]  # (T, n_d)
        tot = units.sum(axis=1)
        held = np.minimum(tot, a["upstream"][:, i])
        share = np.divide(held, tot, out=np.zeros_like(tot), where=tot > 0)
        cost = units @ ch.pi[ds]
        out[k] = {"cost": float(cost.sum()), "cost_held_upstream": float((cost * share).sum())}
    return out


# ----- the summary --------------------------------------------------------------------------------------------------
def _b(x: float) -> str:
    return f"{x / 1e9:,.2f}"


def _m(x: float) -> str:
    return f"{x / 1e6:,.2f}"


def summarize(rows: list[dict], ch: Chain, levels: list[int], header: list[str]) -> str:
    out = list(header)
    for level in levels:
        group = [r for r in rows if r["level"] == level]
        if not group:
            continue
        n = len(group)
        J = {p: np.array([r[p]["J_cents"] / 100.0 for r in group]) for p in ("agent", "naive", "clairvoyant")}
        attainable = float((J["naive"] - J["clairvoyant"]).sum())
        rss = 1.0 - float((J["agent"] - J["clairvoyant"]).sum()) / attainable
        label = "All episodes" if level == 0 else f"Level {level}"
        out += ["", f"## {label}: {n} episodes, RSS {rss:.4f}", ""]

        # 1. shortage by product
        out += [
            "### 1. Chip shortage by product (USD billions per episode; RSS points: share of 1 - RSS)",
            "",
            "| Chip | Agent | Naive | Clairvoyant | Gap (agent - clairvoyant) | RSS points | Agent's share of the "
            "attainable cut |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for k in ch.packaged:
            ds = [d for d, (kk, _pi) in enumerate(ch.demands) if kk == k]
            cost = {
                p: np.array([(np.array(r[p]["short"])[:, ds] @ ch.pi[ds]).sum() for r in group])
                for p in ("agent", "naive", "clairvoyant")
            }
            gap = float((cost["agent"] - cost["clairvoyant"]).sum())
            cut = float((cost["naive"] - cost["clairvoyant"]).sum())
            captured = float((cost["naive"] - cost["agent"]).sum()) / cut if cut else float("nan")
            out.append(
                f"| {ch.names[k]} | {_b(cost['agent'].mean())} | {_b(cost['naive'].mean())} "
                f"| {_b(cost['clairvoyant'].mean())} | {_b(gap / n)} | {gap / attainable:+.4f} | {captured:.0%} |"
            )

        # 2. mass balance
        def tot(p, key, cols=None):
            vals = [np.array(r[p][key]) for r in group]
            return float(np.mean([v[:, cols].sum() if cols is not None else v.sum() for v in vals]))

        out += [
            "",
            "### 2. Mass balance (millions of units per episode)",
            "",
            "| Quantity | Agent | Naive | Clairvoyant |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, key in (("wafers lifted", "wafer_lift"), ("lots started", "lots"), ("chips packaged", "pkg")):
            out.append(
                f"| {name} | {_m(tot('agent', key))} | {_m(tot('naive', key))} | {_m(tot('clairvoyant', key))} |"
            )
        for i, name in enumerate(ch.disposal_groups):
            cells = " | ".join(_m(tot(p, "disposed", [i])) for p in ("agent", "naive", "clairvoyant"))
            out.append(f"| {name} disposed (above storage) | {cells} |")
        for k in ch.packaged:
            ds = [d for d, (kk, _pi) in enumerate(ch.demands) if kk == k]
            pairs = [i for i, (_o, _r, pk, _s) in enumerate(ch.pairs) if pk == k]
            out.append(
                f"| {ch.names[k]}: packaged | {_m(tot('agent', 'pkg', pairs))} | {_m(tot('naive', 'pkg', pairs))} "
                f"| {_m(tot('clairvoyant', 'pkg', pairs))} |"
            )
            out.append(
                f"| {ch.names[k]}: demand / delivered / shortage unit-weeks (lost + backlog, pi's base) "
                f"| {_m(tot('agent', 'demand', ds))} / {_m(tot('agent', 'served', ds))} / "
                f"{_m(tot('agent', 'short', ds))} | {_m(tot('naive', 'demand', ds))} / "
                f"{_m(tot('naive', 'served', ds))} / {_m(tot('naive', 'short', ds))} "
                f"| {_m(tot('clairvoyant', 'demand', ds))} / "
                f"{_m(tot('clairvoyant', 'served', ds))} / {_m(tot('clairvoyant', 'short', ds))} |"
            )

        by_node = {}
        for p in ("agent", "naive", "clairvoyant"):
            per_slot = np.mean([np.array(r[p]["disposed_packaged"]) for r in group], axis=0)
            for s_, v in zip(ch.packaged_slots, per_slot):
                key = ch.slot_node[s_]
                by_node.setdefault(key, {"agent": 0.0, "naive": 0.0, "clairvoyant": 0.0})[p] += float(v)
        top = [kv for kv in sorted(by_node.items(), key=lambda kv: -kv[1]["agent"]) if max(kv[1].values()) > TOL][:8]
        if top and top[0][1]["agent"] > 0:
            out += [
                "",
                "Where packaged chips were disposed (millions per episode, the agent's eight largest nodes):",
                "",
                "| Node | Agent | Naive | Clairvoyant |",
                "| --- | ---: | ---: | ---: |",
            ]
            for node, v in top:
                out.append(f"| {node} | {_m(v['agent'])} | {_m(v['naive'])} | {_m(v['clairvoyant'])} |")

        # 3. timing
        def lags(key, cols=None):
            out_ = {}
            for p in ("agent", "naive"):
                vals = []
                for r in group:
                    a, c = np.array(r[p][key]), np.array(r["clairvoyant"][key])
                    if cols is not None:
                        a, c = a[:, cols], c[:, cols]
                    vals.append(lag(a, c))
                out_[p] = float(np.mean(vals))
            return out_

        out += [
            "",
            "### 3. Timing: how far behind the clairvoyant plan's running totals (millions of unit-weeks per episode)",
            "",
            "| Series | Agent | Naive |",
            "| --- | ---: | ---: |",
        ]
        for name, key, cols in [("lots started", "lots", None), ("chips packaged", "pkg", None)] + [
            (f"{ch.names[k]} delivered to sinks", "served", [d for d, (kk, _pi) in enumerate(ch.demands) if kk == k])
            for k in ch.packaged
        ]:
            lg = lags(key, cols)
            out.append(f"| {name} | {_m(lg['agent'])} | {_m(lg['naive'])} |")

        # 4. fabs
        led = {
            p: [fab_losses({k: np.array(r[p][k]) for k in ("cap", "W", "lots", "fab_shed")}) for r in group]
            for p in ("agent", "naive")
        }
        cl_lots = np.mean([np.array(r["clairvoyant"]["lots"]).sum(axis=0) for r in group], axis=0)
        shed_weeks = {
            p: np.mean([(np.array(r[p]["fab_shed"]) > TOL).sum(axis=0) for r in group], axis=0)
            for p in ("agent", "clairvoyant")
        }
        out += [
            "",
            "### 4. Fabs (millions per episode; idle wafers can start a later week: wafer-weeks are delays)",
            "",
            "| Fab (pi of its chip) | Capacity (lot-weeks) | Started: agent / naive / clairvoyant "
            "| Agent capacity unused for lack of wafers | Agent wafer-weeks idle: grid shedding "
            "| Agent wafer-weeks idle: headroom | Weeks its grid shed: agent / clairvoyant |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for f in range(ch.F):

            def mean(p, key, f=f):
                return float(np.mean([x[key][f] for x in led[p]]))

            out.append(
                f"| {f} ({ch.lot_value[f]:,.0f}) | {_m(mean('agent', 'cap'))} | {_m(mean('agent', 'lots'))} / "
                f"{_m(mean('naive', 'lots'))} / {_m(cl_lots[f])} | {_m(mean('agent', 'no_wafers'))} "
                f"| {_m(mean('agent', 'energy_shed'))} | {_m(mean('agent', 'energy_headroom'))} "
                f"| {shed_weeks['agent'][f]:.1f} / {shed_weeks['clairvoyant'][f]:.1f} |"
            )
        gaps = [
            lot_gap(
                {k: np.array(r["agent"][k]) for k in ("lots", "W", "fab_shed")},
                {"lots": np.array(r["clairvoyant"]["lots"])},
            )
            for r in group
        ]
        out += [
            "",
            "### 5. Lots against the clairvoyant plan, fab-week by fab-week (millions per episode; USD billions at the "
            "chip's penalty pi, an upper bound of a lot's worth)",
            "",
            "| Cause | Lots | Value |",
            "| --- | ---: | ---: |",
        ]
        for key, name in (
            ("fewer", "agent started fewer: all"),
            ("fewer_no_wafers", "  no wafers on hand"),
            ("fewer_energy_shed", "  energy: its grid shed base load (base load first)"),
            ("fewer_energy_headroom", "  energy: headroom short, no shedding"),
            ("more", "agent started more"),
        ):
            lots = np.mean([gp[key] for gp in gaps], axis=0)
            out.append(f"| {name} | {_m(lots.sum())} | {_b(float(lots @ ch.lot_value))} |")

        # 5. the agent's plan against its execution
        fid = [
            plan_fidelity({k: np.array(r["agent"][k]) for k in ("plan_p", "plan_q", "lots", "W", "fab_shed", "pkg")})
            for r in group
        ]
        out += [
            "",
            "### 6. The agent's first-week plan against the simulator (millions per episode)",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
        ]
        for key, name in (
            ("planned_lots", "lots planned"),
            ("started_lots", "lots started"),
            ("more_than_planned", "started beyond the plan (the simulator starts every wafer it can)"),
            ("less_no_wafers", "started below the plan: no wafers"),
            ("less_energy_shed", "started below the plan: energy, grid shedding"),
            ("less_energy_headroom", "started below the plan: energy, headroom"),
            ("planned_packaging", "packaging planned"),
            ("packaged", "packaged"),
            ("packaged_more", "packaged beyond the plan"),
            ("packaged_less", "packaged below the plan"),
        ):
            vals = [x[key] for x in fid if key in x]
            out.append(f"| {name} | {_m(float(np.mean(vals))) if vals else 'n/a (rule_chips 1: no packaging plan)'} |")

        # 6. OSATs
        osat = [
            osat_ledger(
                {k: np.array(r["agent"][k]) for k in ("pkg", "raw", "thr")},
                {"pkg": np.array(r["clairvoyant"]["pkg"])},
                ch,
            )
            for r in group
        ]

        def om(key):
            return float(np.mean([x[key] for x in osat]))

        out += [
            "",
            "### 7. OSATs (millions per episode)",
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| throughput available | {_m(om('thr'))} |",
            f"| packaged: agent / clairvoyant | {_m(om('packaged'))} / {_m(om('packaged_clairvoyant'))} |",
            f"| OSAT-weeks short of raw chips (raw < throughput) | {om('osat_weeks_short_of_raw'):.1f} of "
            f"{om('osat_weeks'):.0f} |",
            f"| packaged fewer than the plan: all (USD billions at pi) | {_m(om('fewer'))} ({_b(om('fewer_value'))}) |",
            f"|   no raw chips on hand | {_m(om('fewer_no_raw'))} ({_b(om('fewer_no_raw_value'))}) |",
            f"|   raw chips on hand (throughput or the pro-rata split) | {_m(om('fewer_at_throughput'))} |",
            f"| packaged more than the plan | {_m(om('more'))} |",
        ]

        # 8. sinks
        sinks = {
            p: [sink_ledger({k: np.array(r[p][k]) for k in ("short", "upstream")}, ch) for r in group]
            for p in ("agent", "clairvoyant")
        }
        out += [
            "",
            "### 8. Shortage while the chip was in stock elsewhere that week (USD billions per episode)",
            "",
            "| Chip | Agent: shortage cost | Agent: with the chip in stock upstream | Clairvoyant: shortage cost "
            "| Clairvoyant: with the chip in stock upstream |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for k in ch.packaged:
            cells = []
            for p in ("agent", "clairvoyant"):
                cells += [
                    _b(float(np.mean([x[k]["cost"] for x in sinks[p]]))),
                    _b(float(np.mean([x[k]["cost_held_upstream"] for x in sinks[p]]))),
                ]
            out.append(f"| {ch.names[k]} | " + " | ".join(cells) + " |")

    out += ["", "## Reading", ""] + READING
    return "\n".join(out) + "\n"


def parse_overrides(text, parse_value) -> dict:
    """'--set=rule_chips=1,lot_value=0.5' -> {'rule_chips': 1, 'lot_value': 0.5}."""
    out = {}
    for item in [x for x in str(text or "").split(",") if x.strip()]:
        key, _, value = item.partition("=")
        out[key.strip()] = parse_value(key.strip(), value)
    return out


def main(params: str = "", set: str = "", levels="1,2", workers: int = 4, smoke: bool = False) -> None:
    """Trace the chip chain of the agent, naive and the clairvoyant plan.

    Args:
        params: a complete knob file for the agent (default: agents/compact_hierarchical/params.json).
        set: knob overrides on top, "key=value,key=value" (for example rule_chips=1).
        levels: harm levels, "1,2" (default) or "all".
        workers: processes per phase (the agents and naive, then the clairvoyant LPs).
        smoke: episodes 0 and 1 with quick references (no harm levels): checks the plumbing only.

    """
    from shockbench_flow_agent import EpisodeSet

    sys.path.insert(0, str(AGENT_DIR))
    from compact_hier import config as agent_config

    path = Path(params) if params else AGENT_DIR / "params.json"
    if not path.is_file():
        sys.exit(f"no params file at {path}")
    knobs = agent_config.load(AGENT_DIR, environ={"SBF_PARAMS_FILE": str(path.resolve())})
    overrides = parse_overrides(set, agent_config.parse_value)
    unknown = sorted(k for k in overrides if k not in agent_config.DEFAULTS)
    if unknown:
        sys.exit(f"--set: unknown knobs {unknown}")
    knobs |= overrides
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
    run = run_dir("chip_chain_diagnostic")
    print(f"{len(ns)} episodes of {TASK} root {ENTROPY} (levels {'smoke' if smoke else wanted}); run folder {run}")
    start = time.perf_counter()
    played = Parallel(n_jobs=workers)(delayed(play)(n, knobs) for n in ns)
    print(f"agents and naive played ({time.perf_counter() - start:.0f} s); solving the clairvoyant LPs ...", flush=True)
    solved = {
        s["episode"]: s
        for s in Parallel(n_jobs=workers, backend="multiprocessing")(delayed(solve_clairvoyant)(n) for n in ns)
    }
    rows, skipped = [], []
    for p in played:
        n = p["episode"]
        if solved[n]["clairvoyant"] is None:
            skipped.append(n)
            continue
        rows.append({**p, "clairvoyant": solved[n]["clairvoyant"], "level": 0 if smoke else int(refs[n]["stratum"])})

    inst = world(TASK, ENTROPY, ns[0])[0]
    ch = chain(inst)
    header = [
        "# Step 1: chip-chain diagnostic",
        "",
        f"- Params: `{path}`" + (f", overrides {overrides}" if overrides else ""),
        f"- Episodes: {len(rows)} of Small root {ENTROPY} (0..{EPISODES - 1}), levels {wanted}"
        + (f"; skipped (clairvoyant not optimal): {skipped}" if skipped else ""),
        f"- Agent weeks over the 2 s budget: {sum(r['agent_over_budget_weeks'] for r in rows)}; weeks of the "
        f"agent's own fallback: {sum(r['agent_fallback_weeks'] for r in rows)}",
        "- Fabs: "
        + ", ".join(f"{f}: {ch.names.get(int(inst.nodes[inst.fabs[f]].fab.product), '?')}" for f in range(ch.F))
        + "; table 3 gives each fab's lot worth at a sink (pi, USD per lot).",
    ]
    text = summarize(rows, ch, [0] if smoke else wanted, header)
    (run / "summary.md").write_text(text)
    with open(run / "fabs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "level",
                "fab",
                "capacity",
                "agent_lots",
                "agent_no_wafers",
                "agent_energy_shed",
                "agent_energy_headroom",
                "clairvoyant_lots",
            ]
        )
        for r in rows:
            led = fab_losses({k: np.array(r["agent"][k]) for k in ("cap", "W", "lots", "fab_shed")})
            cl = np.array(r["clairvoyant"]["lots"]).sum(axis=0)
            for fi in range(ch.F):
                writer.writerow(
                    [
                        r["episode"],
                        r["level"],
                        fi,
                        led["cap"][fi],
                        led["lots"][fi],
                        led["no_wafers"][fi],
                        led["energy_shed"][fi],
                        led["energy_headroom"][fi],
                        cl[fi],
                    ]
                )
    totals = [
        {
            "episode": r["episode"],
            "level": r["level"],
            **{p: r[p]["J_cents"] for p in ("agent", "naive", "clairvoyant")},
        }
        for r in rows
    ]
    (run / "results.json").write_text(json.dumps({"params": knobs, "overrides": overrides, "J_cents": totals}))
    print(text)
    print(f"written {run / 'summary.md'} (paste this file back), fabs.csv, results.json")


if __name__ == "__main__":
    fire.Fire(main)

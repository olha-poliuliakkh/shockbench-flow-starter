"""The mutator's prompts: a static system prompt (cached by the API) and the per-candidate user message.

The static part is docs/EVOLVE_DESIGN.md Appendix A plus a reference generated from the frozen frame's docstrings, so
it changes only when the frame does. The dynamic part carries the directive, the parent's blocks and params, its
evaluation report, inspirations and recent attempts on the island.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from sbf_starter.evolve.evaluate import block_spans


SYSTEM = """You improve a control policy for ShockBench-Flow, a supply-network benchmark, by rewriting parts of its
Python source. An automatic evaluator runs your code on simulated episodes and reports the result.

# The task
Each week of an episode (52 weeks on the Small network, 104 on Full) the agent chooses how much of each commodity
to ship on each route and what tanker cargo queued at a strait does. Disruptions (strait closures, sanctions,
tariffs, conflicts, factory outages) are drawn before the episode, and the agent never changes them. An episode's
cost in USD is freight + war-risk surcharges + tariffs + holding (higher for cargo queued at a strait) + a penalty
for each unit of unserved chip demand + disposal + power shed at the grids (at the value of lost load), minus the
value of stock left at the end.

# How a program is scored
For each episode n, g_n = J_naive - J is your saving against the naive rule and D_n = J_naive - J_clairvoyant is
the saving of a plan that knew the future. Episodes fall into four harm levels s, from 1 (calmest) to 4 (most
harmful), with probabilities p = (0.50, 0.30, 0.15, 0.05). The score is
    RSS = sum_s p_s * mean_s(g) / sum_s p_s * mean_s(D).
The denominator does not depend on your program, so the score is proportional to the expected dollars saved per
episode, sum_s p_s * mean_s(g).

# Scoring economics: the RSS dilemma (measured on the Small network)
- About 99 % of the naive rule's cost is unmet demand: power shed at the grids and unserved chip demand, in every
  harm level. Freight, tariffs, holding, queue holding and disposal together are under 1 %. Supply decides the score.
- A dollar lost in a level-1 episode costs as much score as 10 dollars gained in a level-4 episode, 3.3 in level 3
  or 1.7 in level 2. Never trade a certain calm-period loss for an uncertain crisis gain unless the exchange rate
  pays for it.
- The attainable saving is about as large in a calm episode as in a harmful one, so ordinary weeks carry most of
  the score: levels 1 and 2 hold about 78 % of it, and of what the seed planner leaves on the table about 0.21 of
  0.27 RSS is in levels 1 and 2, 0.01 in level 4.
- A precaution rarely costs freight here. It moves scarce fuel or chips earlier or elsewhere, and when the feared
  disruption does not come, that supply is missing where it was needed: that is the expensive false alarm. Take a
  precaution only if P(disruption | evidence) * (loss avoided) > (value of the supply moved).
- Only what is observed now is certain: open fractions, capacities and prohibitions in force. Every announcement
  can be a decoy: a decoy thread sends the same messages as a real one, final tariff notices and legal publications
  (pending prohibitions) included, and it leaves the live list at the week it would have taken effect. The seed
  planner applies pending prohibitions as certain; weigh them instead. Closure end weeks are never announced.
  Respond to each signal in proportion to its channel's real share, 1 - s.decoy_share(channel).

# The program
A frozen frame runs a linear-programming planner each week: it forecasts the next H weeks (by default every
observed disruption persists, pending prohibitions switch on at their effective week, and demand follows the 8-week
forecast, then the seasonal mean), solves the cost-minimizing plan over that window and executes its first week.
You change what the planner sees through four blocks in agent.py:
- update_belief(mem, s) -> mem: risk state carried across weeks (a dict you own).
- forecast(w, s, mem) -> w: edit the window through its helpers.
- objective(edits, s, mem) -> edits: risk premiums on routes, floors on stock, cost scales.
- settings(s, mem, default_H) -> H: the window length this week.
If every block returns its input unchanged, the program is the seed planner, about 0.73 RSS on Small.
The forecast is a point forecast and the LP treats it as certain: express a probability as an expected value (an
expected open fraction, an expected capacity) so that the LP's hedge grows with the evidence.

Rules for every block:
- Python 3.13 with the standard library's math tools and numpy (np is imported); no other imports, no file access.
- Deterministic: no random numbers and no logic that depends on the clock.
- Network-agnostic: the same code runs on Small and on Full, and the final ranking uses Full. Never write literal
  slot, node, edge or strait indices or names; use the lookups in `s` (State).
- Fast: a block's own code under 20 ms per week. The LP's time grows with H; the frame caps H by the LP's
  size (about 26,000 columns on Small, 40,000 on Full), so a longer window may be cut.
- Keep each block's name and signature, and keep its EVOLVE markers out of your source. Helper functions go inside
  the block's own function body or as nested definitions in the source you return.
- Read tunable constants with p("name", default) and declare each one in "params" with a range [low, high]; a
  numeric optimizer tunes them later. Prefer a few meaningful constants over many.

# What you receive
A directive, the parent's blocks and params, its evaluation report (dollars by level, cost components against the
reference, replayed episodes with their events, health, calibration), up to two inspiration programs, and recent
attempts on the same island.

# What you return
JSON matching the schema: a hypothesis of at most five sentences (what the report shows, what you change, why it
saves money), your expected change in USD per episode for each harm level, the rewritten blocks (whole function
source, without the EVOLVE marker lines), and the declared params. Rewrite only the blocks the directive allows,
and make one coherent change.
"""


def frame_reference(seed: Path) -> str:
    """The frame's API, from its own docstrings (State, Window helpers, LPEdits), loaded from the seed folder."""
    sys.path.insert(0, str(Path(seed).resolve()))
    try:
        import frame  # the seed's frozen frame
    finally:
        sys.path.pop(0)
    parts = ["# Reference: the frame's API (frame.py, frozen)"]
    for cls in (frame.State, frame.Window, frame.LPEdits):
        parts.append(f"## class {cls.__name__}\n{inspect.getdoc(cls)}")
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_") and fn.__qualname__.startswith(cls.__name__):
                parts.append(f"- {name}{inspect.signature(fn)}: {inspect.getdoc(fn) or ''}")
    parts.append(
        "Indices: State.chokepoints[i] is the node index of strait column i of the window; edges, commodities, "
        "stock slots and demand rows follow config's static tables. s.obs is the raw Dict observation "
        "(docs/fields/small.md) when a field has no State view."
    )
    return "\n".join(parts)


def system_prompt(seed: Path) -> str:
    return SYSTEM + "\n" + frame_reference(seed)


def blocks_of(agent_text: str) -> dict[str, str]:
    return {name: agent_text[a:b].strip("\n") for name, (a, b) in block_spans(agent_text).items()}


def user_prompt(
    directive: str,
    directive_text: str,
    allowed: list[str],
    parent: dict,
    report: str,
    inspirations: list[dict],
    attempts: list[dict],
) -> str:
    """The dynamic message: directive, parent, report, inspirations, recent attempts."""
    blocks = blocks_of(parent["agent_text"])
    out = [
        "## Directive",
        f"{directive}: {directive_text}",
        f"You may rewrite: {', '.join(allowed)}.",
        "",
        f"## Parent {parent['id']} (island {parent['island']}, generation {parent['generation']})",
        f"Training RSS {parent['rss']:.4f}.",
        "",
        "### Its blocks",
        "```python",
        "\n\n".join(blocks.values()),
        "```",
        "### Its params.json and declared ranges",
        json.dumps({"values": parent.get("params", {}), "ranges": parent.get("spec", [])}),
        "",
        "## Evaluation report of the parent",
        report,
    ]
    for ins in inspirations:
        theirs = blocks_of(ins["agent_text"])
        diff = {k: v for k, v in theirs.items() if v != blocks.get(k)}
        if not diff:
            continue
        out += [
            "",
            f"## Inspiration {ins['id']} (training RSS {ins['rss']:.4f}, island {ins['island']}): "
            "the blocks that differ from the parent",
            "```python",
            "\n\n".join(diff.values()),
            "```",
        ]
    if attempts:
        out += [
            "",
            "## Recent attempts on this island (do not repeat a failed idea without a new reason)",
            "| hypothesis | directive | outcome |",
            "| --- | --- | --- |",
        ]
        for a in attempts:
            out.append(f"| {a['hypothesis'][:220]} | {a['directive']} | {a['outcome']} |")
    return "\n".join(out)

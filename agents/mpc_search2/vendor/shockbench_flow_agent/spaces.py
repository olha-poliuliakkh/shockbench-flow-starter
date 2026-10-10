"""The Dict observation and action documented field by field, generated from the flat layout.

``markdown(config, observation)`` renders the tables of one episode's ``Agent`` config and Dict observation, so the
shapes, dtypes and index sets printed are the ones an agent receives: one block per network (`tiny`, and the grouped
layout of `small` and `full`). ``DESCRIPTIONS`` holds one line per observation field (the path of the flat key); a key
without one raises, so a new field cannot reach the tables undocumented; ``GROUPED_DESCRIPTIONS`` replaces the pipeline
and queue_lots lines on a grouped layout (`small` and `full`: the pipeline grouped, the queued lots a dense block).
``STATIC`` does the same for the public tables of ``config['static']`` (a table's columns, or a scalar key), with each
column's length on the network. A chokepoint is a sea passage ships queue at: a strait, a canal or the Cape route.
"""

from collections.abc import Mapping

import numpy as np

from shockbench_flow.instance.schema import WAR_RISK_CLASSES
from shockbench_flow.omega import codes


OBSERVED = ".observed"


def _codes(vocabulary: tuple[str, ...]) -> str:
    return ", ".join(f"{i} {v}" for i, v in enumerate(vocabulary))


# the flat key (its path in the observation) -> (what its positions index, meaning)
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "week": ("-", "t, the week to decide (1-based); every other field is the state at instant t - 1"),
    "stock.qty": ("layout.stock_slots", "on-hand stock I^{t-1} per (node, k), chokepoints excluded (see queue_lots)"),
    "backlog.qty": ("layout.demands", "unserved demand carried at each backlog sink (0 at lost-sales sinks)"),
    "pipeline.edge": ("padded list", "shipments in transit: the edge they travel"),
    "pipeline.k": ("padded list", "commodity index"),
    "pipeline.lane": ("padded list", "lane index; unobserved for a shipment off any lane"),
    "pipeline.qty": ("padded list", "quantity; its observed-mask marks the live entries of the list"),
    "pipeline.arrival_week": ("padded list", "week the shipment reaches the edge's head"),
    "queue_lots.lot_id": ("padded list", "lots waiting at a chokepoint (FIFO book): lot id"),
    "queue_lots.chokepoint": ("padded list", "chokepoint node"),
    "queue_lots.k": ("padded list", "commodity index"),
    "queue_lots.qty": ("padded list", "quantity"),
    "queue_lots.lane": ("padded list", "lane the lot follows"),
    "queue_lots.next_edge": ("padded list", "edge the lot leaves the chokepoint by"),
    "queue_lots.arrival_week": ("padded list", "week the lot reached the chokepoint"),
    "queue_lots.dispatch_week": ("padded list", "week the lot was dispatched"),
    "queue_lots.entry_edge": ("padded list", "edge the lot entered the chokepoint by"),
    "wip.node": ("padded list", "work in process at fabs and OSATs (gross): node"),
    "wip.k": ("padded list", "output commodity"),
    "wip.qty": ("padded list", "quantity"),
    "wip.out_week": ("padded list", "week it becomes stock"),
    "graph_now.u": ("edges", "capacity u per edge this week; unobserved on grid couplings"),
    "graph_now.c": ("edges", "freight cost per unit per edge"),
    "graph_now.tau": ("edges", "lead time in weeks per edge"),
    "graph_now.prohibited": ("edges x commodities", "1 where (edge, k) is prohibited (sanctions, export controls)"),
    "graph_now.tariff": ("edges x commodities", "tariff rate on (edge, k)"),
    "graph_now.open": (
        "layout.chokepoints",
        "open fraction o_c of each chokepoint (a strait, canal or the Cape route; 1 open, 0 closed)",
    ),
    "graph_now.kappa.tb": ("layout.chokepoints", "throughput kappa_cb of the tanker/bulk pool"),
    "graph_now.kappa.ct": ("layout.chokepoints", "throughput kappa_cb of the container pool"),
    "graph_now.war_risk": ("layout.chokepoints", "war-risk class code: " + _codes(WAR_RISK_CLASSES)),
    "graph_now.supply.avail": ("layout.supply_slots", "supply available at each source and material slot"),
    "graph_now.fab.R": ("layout.fabs", "restoration factor R_f of each fab"),
    "graph_now.fab.alpha_bar": ("layout.fabs", "power multiplier alpha-bar_f of each fab"),
    "graph_now.fab.cap_eff": ("layout.fabs", "effective wafer capacity of each fab"),
    "graph_now.grid.G_bar": ("layout.grids", "deliverable generation G-bar_g of each grid"),
    "graph_now.grid.y_bar": ("layout.grids", "base load y-bar_g of each grid"),
    "graph_now.osat.R": ("layout.osats", "restoration factor R^osat of each OSAT"),
    "graph_now.osat.thr_eff": ("layout.osats", "effective throughput thr R^osat of each OSAT"),
    "slot_mask": (
        "action slots",
        "1 where an edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity "
        "this week (the wire's convention; see action_mask)",
    ),
    "last_week.clip.requested": ("action slots", "flow you requested last week"),
    "last_week.clip.executed": ("action slots", "flow executed after the capacity clip"),
    "last_week.cost_components": ("layout.cost_components", "last week's cost by component, USD"),
    "last_week.sinks.demand": ("layout.demands", "last week's demand"),
    "last_week.sinks.served": ("layout.demands", "last week's demand served"),
    "last_week.sinks.lost": ("layout.demands", "last week's demand lost"),
    "last_week.shed.qty": ("layout.grids", "power shed at each grid last week"),
    "demand_forecast.qty": ("layout.demands x h", "demand forecast for weeks t + h, h = 0..7"),
    "warning.score": ("layout.warning_units", "early-warning score S^t per region, dyad and chokepoint"),
    "messages.msg_id": (
        "padded list",
        "live announcement threads (announced, not effective, not withdrawn): thread id",
    ),
    "messages.channel": ("padded list", "channel code: " + _codes(codes.CHANNELS)),
    "messages.kind": ("padded list", "message kind code: " + _codes(codes.MESSAGE_KINDS)),
    "messages.region": ("padded list", "region index"),
    "messages.target_kind": ("padded list", "target kind code: " + _codes(codes.TARGET_KINDS)),
    "messages.target": ("padded list", "target index"),
    "messages.k": ("padded list", "commodity index; unobserved when the message names none"),
    "messages.announced_week": ("padded list", "week announced"),
    "messages.stated_effective_week": ("padded list", "stated effective week; unobserved when none is stated"),
    "pending_prohibitions.edge": ("padded list", "announced prohibitions not yet in force: edge"),
    "pending_prohibitions.k": ("padded list", "commodity index"),
    "pending_prohibitions.effective_week": ("padded list", "week it takes effect"),
    "closure_end.chokepoint": ("padded list", "closures acting now: chokepoint"),
    "closure_end.end_week": ("padded list", "announced end week; unobserved when unknown"),
    "action_mask": (
        "action slots",
        "1 where no edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity "
        "this week (the inverse of slot_mask); capacities and closures are not checked: read graph_now.u and "
        "graph_now.open",
    ),
    "action_mask.observed": ("-", "1 when this week's mask was observed (0 in a blackout week: all slots allowed)"),
    "override_mask": (
        "override slots",
        "1 where the slot's own out edge is not prohibited for its commodity this week; release_mode 1 sends every "
        "override slot of its pair, and a slot at 0 is dropped whatever its override_qty (one invalid entry, no "
        "cost): the pair's other slots stand, and with none valid the default release stays on",
    ),
    "override_mask.observed": ("-", "1 when this week's override mask was observed (0 in a blackout week)"),
}
# the masks' observed flags: one entry each, not an observed-mask of their mask's shape (``primary_keys``)
MASK_FLAGS = ("action_mask.observed", "override_mask.observed")
# the grouped layout of `small` and `full` (``FlatLayout.grouped``): the pipeline lists groups and the queue lots are
# one dense block, whose keys replace DESCRIPTIONS' entries of those blocks
GROUPED_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "pipeline.edge": ("grouped list", "shipments in transit, one entry per (edge, k, lane, arrival week): the edge"),
    "pipeline.k": ("grouped list", "commodity index"),
    "pipeline.lane": ("grouped list", "lane index; unobserved for shipments off any lane"),
    "pipeline.qty": ("grouped list", "the group's total quantity; its observed-mask marks the live entries"),
    "pipeline.arrival_week": ("grouped list", "week the group reaches the edge's head"),
    "queue_lots.qty": (
        "layout.lot_keys x week",
        "quantity waiting at a chokepoint, one row per (chokepoint, k, lane, next edge) of layout.lot_keys, one column "
        "per week the lots reached it (column w - 1 is week w; their FIFO cohort): the total of those lots, observed "
        "where lots wait",
    ),
}
ACTION: dict[str, tuple[str, str]] = {
    "flows": ("action slots", "quantity to dispatch on each (edge, commodity, lane) slot; 0 sends nothing"),
    "override_qty": ("override slots", "tanker cargo to release on each override slot, read where release_mode is 1"),
    "release_mode": (
        "layout.release_pairs",
        "per (chokepoint, tanker commodity): 0 default release, 1 override, 2 hold",
    ),
}


# config["static"]: the key, or "table.column" of a table of equal-length columns -> meaning
STATIC: dict[str, str] = {
    "instance": "the full public instance JSON; `initial_state.pipeline` holds the week-0 shipments of "
    "the nominal plan (the nominal flows the heuristic sample reads); edges there call the lead time `tau`",
    "instance_id": "the instance's name",
    "instance_hash": "SHA-256 of the instance",
    "T": "the horizon, in weeks",
    "units": "the unit of each commodity's quantities, and of costs",
    "regions": "region names: `nodes.region`, `messages.region`, `dyads.*` and the warning's region units index them",
    "nodes.id": "node name",
    "nodes.type": "source, terminal, grid, chokepoint (a strait, canal or the Cape route), material, fab, osat or sink",
    "nodes.region": "region index",
    "commodities.id": "commodity name",
    "commodities.v": "customs value v_k, USD per unit",
    "commodities.pool": "chokepoint throughput pool: tb (tanker/bulk, `graph_now.kappa.tb`) or ct (container)",
    "commodities.override": "true for a tanker commodity, whose cargo queued at a chokepoint `release_mode` and "
    "`override_qty` steer",
    "edges.id": "edge name",
    "edges.tail": "node index the edge leaves",
    "edges.head": "node index the edge reaches",
    "edges.mode": "sea, air, pipeline or grid",
    "edges.tau0": "nominal lead time in weeks (`graph_now.tau` is this week's)",
    "edges.c0": "nominal freight, USD per unit (`graph_now.c` is this week's)",
    "edges.u0": "nominal capacity per week, null on a grid coupling (`graph_now.u` is this week's)",
    "edges.K": "commodity indices the edge may carry (empty on a grid coupling)",
    "edges.alt_of": "the route the edge duplicates, {'lane': i} or {'edge': i}, or null",
    "edges.pool": "chokepoint pool of its traffic, null on a grid coupling",
    "lanes.id": "lane name (a route through one or more chokepoints)",
    "lanes.edges": "edge indices along the lane, in order",
    "lanes.chokepoints": "chokepoint node indices it passes, in order",
    "lanes.alt_of": "the route the lane duplicates, {'lane': i} or {'edge': i}, or null",
    "action_slots.edge": "each slot's edge (on a lane, the lane's first edge)",
    "action_slots.k": "each slot's commodity index",
    "action_slots.lane": "each slot's lane index, null off any lane",
    "override_slots.chokepoint": "chokepoint node index",
    "override_slots.k": "tanker commodity index",
    "override_slots.out_edge": "edge the released cargo leaves the chokepoint by",
    "override_slots.lane": "lane index, null off any lane",
    "sinks.node": "demand node index (the rows of `layout.demands`)",
    "sinks.k": "commodity demanded",
    "sinks.backlog": "true: unserved demand is carried to later weeks; false: it is lost",
    "sinks.pi": "shortage penalty pi, USD per unit of demand not served",
    "dyads.a": "first region index of each dyad (the warning's dyad units)",
    "dyads.b": "second region index of each dyad",
    "regime": "the information regime's published parameters (`name`, `L`, `a`, `phi`, `chi`, `h_cov`, `skill`, "
    "`blackout`); null when the runner hides it",
}


def static_fields(static: Mapping) -> list[tuple[str, str]]:
    """(``STATIC`` key, what it holds on this instance) for every key of ``static``, in its order.

    A key whose value is a dict of equal-length lists is a table: one item per column, "N entries"; any other key is
    one entry (``instance``: its top-level keys; ``units``: unit by commodity; ``regime``: its name).
    """

    def entries(n: int) -> str:
        return f"{n} entry" if n == 1 else f"{n} entries"

    out = []
    for key, value in static.items():
        cols = value if isinstance(value, Mapping) else None
        if key not in ("instance", "units", "regime") and cols and all(isinstance(v, list) for v in cols.values()):
            out += [(f"{key}.{c}", entries(len(v))) for c, v in cols.items()]
        elif key == "instance":
            out.append((key, "keys " + ", ".join(f"`{k}`" for k in value)))
        elif key == "units":
            out.append((key, ", ".join(f"{k}: {v}" for k, v in value.items())))
        elif key == "regime":
            out.append((key, f"`{value.get('name')}`" if isinstance(value, Mapping) else "null"))
        elif isinstance(value, list):
            out.append((key, entries(len(value))))
        else:
            out.append((key, f"`{value}`" if key != "instance_hash" else f"`{str(value)[:12]}...`"))
    return out


def static_table(config: Mapping) -> list[str]:
    """One row per Static key or table column: field, what it holds here, meaning.

    Raises:
        KeyError: on a field ``STATIC`` does not document.

    """
    rows = [_row("field", "here", "meaning"), _row("---", "---", "---")]
    for key, here in static_fields(config["static"]):
        rows.append(_row(f"`{key}`", here, STATIC[key]))
    return rows


def primary_keys(observation: Mapping[str, np.ndarray]) -> list[str]:
    """The observation's keys without their observed-masks, in order (the masks' flags ``MASK_FLAGS`` are fields)."""
    return [
        k for k in observation if k in MASK_FLAGS or not (k.endswith(OBSERVED) and k[: -len(OBSERVED)] in observation)
    ]


def _shape(a) -> str:
    return "(" + ", ".join(str(n) for n in a) + (",)" if len(a) == 1 else ")")


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def descriptions(grouped: bool = False) -> dict[str, tuple[str, str]]:
    """The field descriptions of a layout: ``DESCRIPTIONS``, its grouped blocks' from ``GROUPED_DESCRIPTIONS``.

    A grouped layout (``FlatLayout.grouped``) documents exactly its keys: the list fields its dense queue_lots drops
    are left out.
    """
    if not grouped:
        return DESCRIPTIONS
    blocks = {key.split(".")[0] for key in GROUPED_DESCRIPTIONS}
    return {
        k: GROUPED_DESCRIPTIONS.get(k, v)
        for k, v in DESCRIPTIONS.items()
        if k.split(".")[0] not in blocks or k in GROUPED_DESCRIPTIONS
    }


def observation_table(observation: Mapping[str, np.ndarray], grouped: bool = False) -> list[str]:
    """One row per observation field: key, shape, dtype, index set, meaning, and whether it has an observed-mask.

    ``grouped``: the layout's pipeline and queue_lots are grouped (``descriptions``).

    Raises:
        KeyError: on a field the layout's descriptions do not document.

    """
    documented = descriptions(grouped)
    rows = [_row("key", "shape", "dtype", "indexed by", "meaning"), _row("---", "---", "---", "---", "---")]
    for key in primary_keys(observation):
        index, meaning = documented[key]
        a = observation[key]
        rows.append(_row(f"`{key}`", _shape(a.shape), a.dtype.name, index, meaning))
    return rows


def action_table(config: Mapping) -> list[str]:
    rows = [_row("key", "shape", "dtype", "indexed by", "meaning"), _row("---", "---", "---", "---", "---")]
    for key, spec in config["spaces"]["action"].items():
        index, meaning = ACTION[key]
        rows.append(_row(f"`{key}`", _shape(spec["shape"]), spec["dtype"], index, meaning))
    return rows


def _names(static: Mapping, table: str) -> list[str]:
    return static[table]["id"]


def index_tables(config: Mapping) -> list[str]:
    """The index sets of the instance: action slots, override slots, release pairs and the ``layout`` tables."""
    st, lay = config["static"], config["layout"]
    node, edge, k, lane = (_names(st, t) for t in ("nodes", "edges", "commodities", "lanes"))
    out = ["**Action slots** (`flows`, `action_mask`, `slot_mask`, `last_week.clip.*`):", ""]
    out += [_row("slot", "edge", "from", "to", "commodity", "lane"), _row("---", "---", "---", "---", "---", "---")]
    a = st["action_slots"]
    for s, (e, kk, l) in enumerate(zip(a["edge"], a["k"], a["lane"])):
        tail, head = node[st["edges"]["tail"][e]], node[st["edges"]["head"][e]]
        out.append(_row(str(s), edge[e], tail, head, k[kk], "-" if l is None else lane[l]))
    out += ["", "**Override slots** (`override_qty`, `override_mask`):", ""]
    out += [_row("slot", "chokepoint", "commodity", "out edge", "lane"), _row("---", "---", "---", "---", "---")]
    o = st["override_slots"]
    for s, (c, kk, e, l) in enumerate(zip(o["chokepoint"], o["k"], o["out_edge"], o["lane"])):
        out.append(_row(str(s), node[c], k[kk], edge[e], "-" if l is None else lane[l]))
    out += ["", "**Layout tables** (`config['layout']`, the positions of the densified blocks):", ""]
    out += [_row("table", "entries"), _row("---", "---")]
    named = {
        "stock_slots": lambda x: f"{node[x[0]]}/{k[x[1]]}",
        "supply_slots": lambda x: f"{node[x[0]]}/{k[x[1]]}",
        "demands": lambda x: f"{node[x[0]]}/{k[x[1]]}",
        "chokepoints": lambda x: node[x],
        "fabs": lambda x: node[x],
        "grids": lambda x: node[x],
        "osats": lambda x: node[x],
        "warning_units": lambda x: (
            f"{x[0]} {st['regions'][x[1]] if x[0] == 'region' else x[1] if x[0] == 'dyad' else node[x[1]]}"
        ),
        "cost_components": lambda x: x,
        "release_pairs": lambda x: f"{node[x[0]]}/{k[x[1]]}",
    }
    if "lot_keys" in lay:  # a grouped layout's dense queue lots: chokepoint/commodity/lane/next edge
        named["lot_keys"] = lambda x: f"{node[x[0]]}/{k[x[1]]}/{lane[x[2]]}/{edge[x[3]]}"
    for table, fmt in named.items():
        out.append(_row(f"`{table}`", ", ".join(f"{i}: {fmt(x)}" for i, x in enumerate(lay[table]))))
    return out


def markdown(
    config: Mapping, observation: Mapping[str, np.ndarray], grouped: bool = False, generated_by: str | None = None
) -> str:
    """The field tables of one network in Markdown: observation, action, index and public tables.

    ``grouped``: the layout's ``FlatLayout.grouped`` (`small` and `full`: the grouped pipeline and the dense queue
    lots; `tiny` is not grouped). ``generated_by``: the command that generated the text, named in its first line
    (None names none).
    """
    st = config["static"]
    by = "" if generated_by is None else f" by `{generated_by}`"
    lines = [
        f"Generated for instance `{st['instance_id']}` (T = {config['T']}, regime "
        f"`{(config['regime'] or {}).get('name')}`){by}.",
        "",
        "**Observation** (`act(observation)`): a dict of numpy arrays. Every key below except `action_mask.observed`"
        " and `override_mask.observed` is followed by `<key>.observed`, an int8 array of the same shape, 1 where the"
        " value is present and 0 where it is unobserved or padding (the value is then 0).",
        "",
        *observation_table(observation, grouped),
        "",
        "**Action** (the return value of `act`): a dict of numpy arrays (`override_qty` and `release_mode` may be left"
        " out: zeros, the default release).",
        "",
        *action_table(config),
        "",
        *index_tables(config),
        "",
        "**Static tables** (`config['static']`, the episode's public tables): a table is a dict of equal-length "
        "lists, one entry per "
        "node, edge, lane, commodity, slot or sink, and the indices above point into them.",
        "",
        *static_table(config),
    ]
    return "\n".join(lines) + "\n"

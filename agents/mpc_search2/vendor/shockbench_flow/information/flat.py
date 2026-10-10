"""The flat view of §9.1: fixed-shape arrays for an observation and an action (design §9.1, §9.2; Q68, Q86, Q95).

No gymnasium here: the adapter ``shockbench_flow_gym`` builds its spaces from ``FlatLayout``. Readings of design §12
(M3 rows; the layout itself is the row "Flat view layout (M3 wire)"):

- Action: the flows vector over ``Static.action_slots`` in their fixed order (an entry of qty exactly 0 is not sent,
  since a missing slot is 0, so a masked slot at 0 logs nothing; any other value is sent and checked by §9.3), the
  override qty vector over ``override_slots``, and one release mode per (chokepoint, tanker commodity) pair:
  ``RELEASE_DEFAULT`` sends no override or hold entry for the pair, ``RELEASE_OVERRIDE`` sends every override slot of
  the pair (qty 0 included, which turns the default release off and releases nothing there), ``RELEASE_HOLD`` sends a
  hold entry. A fixed-length qty vector alone could not say 'no override'.
- Observation: blocks whose index set Static fixes are densified (stock by plain stock slot, backlog by demand,
  supply, prohibited and tariff as E x K, clip by action slot, forecast D x 8, warning over the R + D + C signal units
  of Static, whatever the regime, so a hidden ``Static.regime`` changes no shape: its observed-mask is 0 where
  ``warning`` is null);
  variable-length blocks (pipeline, queue_lots, wip, messages, pending_prohibitions, and closure_end, one entry per
  closure event since several closures can act on one chokepoint at once) are padded to a cap per instance kind
  (``maxima_for``; M5 stream "wire") that raises on overflow and never truncates, a caller's ``maxima`` overriding it.
  Each null becomes 0 with a 0 in its field's observed-mask. On `small` and `full` (``GROUPED_KINDS``; M5-O35 (c),
  design §12 "Flat view grouped lists (M5 wire3)") the pipeline is grouped first (``group_rows``): one entry per
  (edge, k, lane, arrival_week), qty the exact sum of the group's. There the queue lots are a dense block (M5-O38 (b),
  design §12 "Flat view dense queue lots (M5 wire4)"; ``dense_lots``): ``queue_lots.qty`` of shape
  (len(``FlatLayout.lot_keys``), T), one row per (chokepoint, k, lane, next_edge) a lot can hold, one column per
  arrival week 1..T, each cell the exact sum of its lots, observed where lots wait; the lots' index fields go. The
  §9.2 lists stay in the observation.
- The action mask is 1 = valid (the inverse of the wire's ``slot_mask``, True where an edge of the slot's route is
  prohibited for its commodity, Q111); in a blackout week (``slot_mask`` null) it is all-valid with its observed-mask 0.
- The override mask is 1 = valid per override slot, 0 where the slot's own (out-edge, k) is prohibited this week (the
  entry §9.3 drops), from ``graph_now.prohibited``; all-valid with its observed flag 0 when ``graph_now`` is null (M5
  re-gate INT-M5R-03, design §9.1). Neither mask is a wire field.

Keys are the §9.2 paths of the leaf fields in §9.2 order (``graph_now.kappa.tb``), each followed by
``<path>.observed`` (int8, 1 where the value is present); float64 quantities, int64 indices, weeks and vocabulary codes
(the index in the frozen tuple), int8 for the booleans ``prohibited`` and ``slot_mask``.

Caps (design §12 "Flat view layout (M3 wire)", "Flat caps per instance kind (M5 wire)" and "Flat view grouped lists
(M5 wire3)"). Three structural bounds are proved. wip holds one entry per fab and start week in the last tau^fab
weeks, booked at every start even when nothing starts (12), and one per OSAT, out week in the next tau^osat weeks and
package, booked for every package pair (19), so its length is ``wip_bound`` in every week once the initial state fits
it. Grouped, pipeline and queue_lots have bounds too, since lead times are fixed (Q87): a dispatch or release of week
d >= 1 on edge e arrives in d + tau_e (2), so the observation of week t holds at most tau_e arrival weeks per (edge, k,
lane) (``pipeline_bound``), and a lot at chokepoint c on lane l arrived in a week of [1 + tau of l's edge into c, T]
(``queue_lots_bound``, the cells a lot can occupy), besides the initial state's. `small` and `full` pad wip and the
pipeline to their bounds, and ``FlatLayout.from_static`` refuses a cap below one; their dense queue lots have a cell
for every lot, so no cap. The feed blocks have none to prove: ``messages`` holds the live
threads among omega's real and shadow ones, ``pending_prohibitions`` holds one entry per (event, edge, k) and
``closure_end`` one per closure event acting at t - 1, and the generator's event and shadow counts are Poisson and
Hawkes draws with unbounded support. Nor have the lists of pipeline and queue_lots a useful one: a chokepoint's lots
grow by one per arriving shipment for as long as it stays closed or held, and each lot released in pieces sends one
shipment per piece (§3.4). So those caps are measured maxima with the margin 2: `tiny`'s feed caps over its generated
episodes (M3 attempt 1's gate, INT-M3-01, where the placeholders 256 / 128 / 16 raised from rung 0.79 up), `tiny`'s
own-state caps SYNTHETIC(placeholder), and the feed caps of `small` and `full` over their generated episodes at every
rung (``scripts/python/evidence/flat_caps.py generated``). The messages block carries only the live threads
(``live_messages``, Q105), while the §9.2 observation keeps the cumulative list, whose longest measured list (`tiny`
6,364, `full` 11,856) would need a cap of 12,736 (23,744) by the same rule.
"""

import math
from dataclasses import dataclass

import numpy as np

from shockbench_flow.dynamics.state import COST_COMPONENTS  # the order of last_week.cost_components (re-exported)
from shockbench_flow.instance import load_instance
from shockbench_flow.instance.schema import WAR_RISK_CLASSES, Instance
from shockbench_flow.omega import codes


RELEASE_DEFAULT, RELEASE_OVERRIDE, RELEASE_HOLD = 0, 1, 2  # the release mode per (chokepoint, tanker commodity) pair
RELEASE_MODES = (RELEASE_DEFAULT, RELEASE_OVERRIDE, RELEASE_HOLD)
HORIZONS = 8  # h = 0..7 of the demand forecast (48), fixed by §9.2
# padded lengths of the variable-length blocks (no structural bound is proved for any). The own-state caps are
# SYNTHETIC(placeholder), with no source. The feed caps are the smallest multiple of 64 at least twice the longest list
# over 1,000 generated `tiny` episodes per rung (E_split 0, dev) and every regime showing the feed: the live messages
# (``live_messages``, Q105) 1,130, pending_prohibitions 1,131, closure_end 78 (scripts/python/evidence/flat_caps.py 1000
# on the worker VM, Linux x86_64, docs/evidence/flat_caps_1000.txt; the command is in the script's docstring and design
# §12 "Flat view layout (M3 wire)"). The margin 2 is SYNTHETIC(placeholder). These are `tiny`'s caps, which do not
# move (the hackathon's shapes); `small` and `full` have their own (``MAXIMA_BY_KIND``).
DEFAULT_MAXIMA: dict[str, int] = {
    "pipeline": 256,
    "queue_lots": 128,
    "wip": 128,
    "messages": 2304,
    "pending_prohibitions": 2304,
    "closure_end": 192,
}
# The padded lengths per instance kind (M5 stream "wire"; design §12 "Flat caps per instance kind (M5 wire)", "Flat
# caps on generated small and full (M5 wire)", "Flat view grouped lists (M5 wire3)" and "Flat view dense queue lots
# (M5 wire4)"), the one place a cap is set. `tiny` keeps DEFAULT_MAXIMA, as does `abstract` (no instance of it is
# measured). On `small` and `full`, whose pipeline is grouped (GROUPED_KINDS, M5-O35 (c)) and whose queue lots are a
# dense block with no cap (M5-O38 (b)): wip and pipeline are the structural bounds ``wip_bound`` and
# ``pipeline_bound`` of the committed instance (``FlatLayout.from_static`` refuses a smaller cap, so a change of the
# lead times, lanes, slots, T, the initial state, tau^fab, tau^osat or the package tables that raises one fails loudly
# until this table follows). The feed blocks are
# the smallest multiple of 64 at least twice the longest list measured, the margin 2 SYNTHETIC(placeholder) as
# `tiny`'s, over 1,000 generated episodes per rung of §2.6 (E_split 0, split dev) and every regime that shows them (`uv
# run --no-project --with numpy==2.4.5 --with scipy==1.18.1 --with fastjsonschema~=2.22.2 --with joblib --with
# numba==0.67.0 python scripts/python/evidence/flat_caps.py generated 1000` on the worker VM, Linux x86_64, 8 loky
# workers, docs/evidence/flat_caps_generated_1000.txt; re-measured on the round-2 merged tree, M5-O21, O24 and O25 in);
# each maximum comes from rung 0.97, and play beyond the measured feeds overflows loudly. The maxima below are that
# tree's (the round-2 integration, before Q114's V3 sanction rules and round 4's warm start per rung), not re-measured
# since (the M5 gate's INT-M5-04); the held-out episodes of tests/test_m5_wire.py::test_feed_caps_on_small_and_full
# (E_split 1, 800 per size in the release tier) check the caps on the current generator. Before the grouping, pipeline
# and queue_lots took measured caps on these kinds too (`small` 2,368 and 3,520, `full` 6,464 and 20,928; M5-O29, O35);
# grouped, queue_lots was padded to ``queue_lots_bound`` (3,938 and 21,214) until its dense block (M5-O38 (b)).
MAXIMA_BY_KIND: dict[str, dict[str, int]] = {
    "tiny": DEFAULT_MAXIMA,
    "abstract": DEFAULT_MAXIMA,
    # longest feed lists at the round-2 integration (above): messages 1,201, pending 2,368, closure_end 44
    "small": {
        "pipeline": 269,
        "wip": 54,
        "messages": 2432,
        "pending_prohibitions": 4736,
        "closure_end": 128,
    },
    # longest feed lists at the round-2 integration (above): messages 1,066, pending 2,481, closure_end 40
    "full": {
        "pipeline": 883,
        "wip": 136,
        "messages": 2176,
        "pending_prohibitions": 4992,
        "closure_end": 128,
    },
}
GROUPED_KINDS = frozenset({"small", "full"})  # kinds whose pipeline and queue_lots the flat view groups (M5-O35 (c))
GROUPED_BLOCKS = ("pipeline", "queue_lots")  # the §9.2 lists a grouped layout groups
DENSE_BLOCK = "queue_lots"  # on a grouped layout a dense (lot key, arrival week) block, no padded list (M5-O38 (b))
F8, I8, I64 = np.float64, np.int8, np.int64
# variable-length blocks: (field, dtype, vocabulary or None), in §9.2 order
_VARIABLE: dict[str, tuple[tuple[str, type, tuple[str, ...] | None], ...]] = {
    "pipeline": (
        ("edge", I64, None),
        ("k", I64, None),
        ("lane", I64, None),
        ("qty", F8, None),
        ("arrival_week", I64, None),
    ),
    "queue_lots": tuple(
        (f, F8 if f == "qty" else I64, None)
        for f in (
            "lot_id",
            "chokepoint",
            "k",
            "qty",
            "lane",
            "next_edge",
            "arrival_week",
            "dispatch_week",
            "entry_edge",
        )
    ),
    "wip": (("node", I64, None), ("k", I64, None), ("qty", F8, None), ("out_week", I64, None)),
    "messages": (
        ("msg_id", I64, None),
        ("channel", I64, codes.CHANNELS),
        ("kind", I64, codes.MESSAGE_KINDS),
        ("region", I64, None),
        ("target_kind", I64, codes.TARGET_KINDS),
        ("target", I64, None),
        ("k", I64, None),
        ("announced_week", I64, None),
        ("stated_effective_week", I64, None),
    ),
    "pending_prohibitions": (("edge", I64, None), ("k", I64, None), ("effective_week", I64, None)),
    "closure_end": (("chokepoint", I64, None), ("end_week", I64, None)),
}
# the grouped blocks: the §9.2 fields that key a group (qty is summed), and the fields a grouped layout keeps, in §9.2
# order: the grouped pipeline its five, the dense queue lots qty alone, the row and column carrying the key and the
# week (design §12 "Flat view grouped lists (M5 wire3)", "Flat view dense queue lots (M5 wire4)")
_GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "pipeline": ("edge", "k", "lane", "arrival_week"),
    "queue_lots": ("chokepoint", "k", "lane", "next_edge", "arrival_week"),
}
_GROUPED: dict[str, tuple[tuple[str, type, tuple[str, ...] | None], ...]] = {
    "pipeline": tuple(f for f in _VARIABLE["pipeline"] if f[0] in (*_GROUP_KEYS["pipeline"], "qty")),
    DENSE_BLOCK: (("qty", F8, None),),
}


def maxima_for(kind: str) -> dict[str, int]:
    """The padded length of each variable-length block on an instance of ``kind`` (``MAXIMA_BY_KIND``), a fresh dict.

    Raises:
        ValueError: on a kind outside the table (``INSTANCE_KINDS``).

    """
    if kind not in MAXIMA_BY_KIND:
        raise ValueError(f"flat caps: no instance kind {kind!r} ({', '.join(MAXIMA_BY_KIND)})")
    return dict(MAXIMA_BY_KIND[kind])


def wip_bound(inst: Instance) -> int:
    """The structural bound of the wip block: sum over fabs of tau^fab plus sum over OSATs of tau^osat x |packages|.

    Every week a fab books its start (12), zero included, and matures the start of tau^fab weeks ago, so it lists
    tau^fab start weeks; an OSAT books every package pair's output for its out week (19), zero included, and delivers
    this week's, so it lists tau^osat out weeks per package (the observation lists every booked entry, zeros too). The
    initial state's WIP lies in the same windows (``tests/test_m5_wire.py`` checks the length every week).
    """
    fabs = sum(inst.nodes[f].fab.tau for f in inst.fabs)
    return fabs + sum(inst.nodes[o].osat.tau * len(inst.nodes[o].osat.packages) for o in inst.osats)


def _lane_commodities(inst: Instance) -> list[set[int]]:
    """K*_l per lane: the commodities every edge of the lane carries (``lane_K``) and any the initial state adds."""
    ks = [set(K) for K in inst.lane_K]
    for s in inst.initial_state.pipeline:
        if s.lane is not None:
            ks[s.lane].add(s.k)
    for q in inst.initial_state.queue_lots:
        ks[q.lane].add(q.k)
    return ks


def pipeline_keys(inst: Instance) -> dict[tuple[int, int, int | None], frozenset[int]]:
    """Every (edge, k, lane) a shipment can carry, mapped to its initial shipments' arrival weeks outside [1, tau_e].

    The set P of design §12 "Flat view grouped lists (M5 wire3)": the action slots (a dispatch, (2)), every edge of
    every lane with every k of K*_l (a default release keeps the lot's lane and leaves by e_l(c), (10)), the override
    slots' (out-edge, k, lane) (an override ships with the slot's lane, §3.4 step 2) and the initial pipeline's.
    """
    ks = _lane_commodities(inst)
    keys = set(inst.action_slots)
    keys |= {(e, k, li) for li, lane in enumerate(inst.lanes) for e in lane.edges for k in ks[li]}
    keys |= {(e, k, li) for _c, k, e, li in inst.override_slots}
    keys |= {(s.edge, s.k, s.lane) for s in inst.initial_state.pipeline}
    extra: dict[tuple[int, int, int | None], set[int]] = {key: set() for key in keys}
    for s in inst.initial_state.pipeline:
        if not 1 <= s.arrival_week <= inst.edges[s.edge].tau:
            extra[(s.edge, s.k, s.lane)].add(s.arrival_week)
    return {key: frozenset(weeks) for key, weeks in extra.items()}


def lot_weeks(inst: Instance) -> dict[tuple[int, int, int, int], frozenset[int]]:
    """Every (chokepoint, k, lane, next edge) a lot can hold, mapped to the weeks it can have arrived in.

    The set L of design §12 "Flat view grouped lists (M5 wire3)": a lot forms when a shipment on lane l reaches
    chokepoint c by the lane's edge e' into it (``lane_through``) and takes e_l(c); it arrived in d + tau_e' for a
    dispatch or release of week d >= 1 (2), so in [1 + tau_e', T], or in an initial shipment's arrival week on
    (e', k, l) up to T, or it is an initial lot, at its own week.
    """
    ks, T = _lane_commodities(inst), inst.T
    init0 = inst.initial_state
    out: dict[tuple[int, int, int, int], set[int]] = {}
    for li, lane in enumerate(inst.lanes):
        for c in lane.chokepoints:
            e_in, e_out = inst.lane_through[(li, c)]
            for k in ks[li]:
                weeks = set(range(1 + inst.edges[e_in].tau, T + 1))
                weeks |= {s.arrival_week for s in init0.pipeline if (s.edge, s.k, s.lane) == (e_in, k, li)}
                out[(c, k, li, e_out)] = {w for w in weeks if w <= T}
    for q in init0.queue_lots:
        out.setdefault((q.chokepoint, q.k, q.lane, q.next_edge), set()).add(q.arrival_week)
    return {key: frozenset(weeks) for key, weeks in out.items()}


def _grouped_bounds(inst: Instance) -> dict[str, int]:
    return {
        "pipeline": sum(inst.edges[e].tau + len(x) for (e, _k, _l), x in pipeline_keys(inst).items()),
        "queue_lots": sum(len(weeks) for weeks in lot_weeks(inst).values()),
    }


def pipeline_bound(inst: Instance) -> int:
    """The structural bound of the grouped pipeline block: over P, tau_e plus the initial weeks outside [1, tau_e].

    Lead times are fixed (Q87), so a dispatch or release of week d >= 1 on e arrives in d + tau_e (2), and the
    observation of week t (the state at t - 1) lists the shipments arriving in week t or later: at most tau_e arrival
    weeks per (edge, k, lane) (``pipeline_keys``). ``tests/test_m5_wire.py`` checks every group of drawn episodes.
    """
    return inst.memo("information.flat.grouped_bounds", _grouped_bounds)["pipeline"]


def queue_lots_bound(inst: Instance) -> int:
    """The cells a lot can occupy in the dense queue_lots block: the arrival weeks ``lot_weeks`` allows, summed.

    The observation of week t <= T + 1 lists the lots arrived by t - 1 <= T; a lot can wait out the episode behind a
    closure or a queue longer than the throughput (10), so nothing tighter holds in general. It bounds the groups of
    any observation's lots; the dense block has len(lot_weeks) x T cells (M5-O38 (b)).
    """
    return inst.memo("information.flat.grouped_bounds", _grouped_bounds)["queue_lots"]


def group_rows(block: str, cols: dict | None) -> dict | None:
    """The grouped columns of a §9.2 pipeline or queue_lots list; None stays None.

    One row per distinct group key (``_GROUP_KEYS``: (edge, k, lane, arrival_week) or (chokepoint, k, lane, next_edge,
    arrival_week)), in the order of the group's first entry, with ``qty`` the ``math.fsum`` of its entries: the
    correctly rounded total, whatever their order (design §12 "Flat view grouped lists (M5 wire3)").
    """
    if cols is None:
        return None
    keys = _GROUP_KEYS[block]
    parts: dict[tuple, list[float]] = {}
    for key, q in zip(zip(*(cols[f] for f in keys)), cols["qty"]):
        parts.setdefault(key, []).append(q)
    out = {f: [key[j] for key in parts] for j, f in enumerate(keys)}
    out["qty"] = [math.fsum(qs) for qs in parts.values()]
    return out


def dense_lots(lot_keys: tuple[tuple[int, int, int, int], ...], T: int, cols: dict | None) -> tuple[np.ndarray, ...]:
    """The dense queue_lots block of a §9.2 lots list: (qty, observed), each of shape (len(lot_keys), T).

    Row i is the (chokepoint, k, lane, next_edge) ``lot_keys[i]``, column w - 1 the arrival week w in 1..T; a cell's
    qty is the ``math.fsum`` of the lots of its row that arrived in its week (``group_rows``), and its observed flag 1
    where a lot is listed, else 0 with qty 0 (design §12 "Flat view dense queue lots (M5 wire4)", M5-O38 (b)). None is
    all 0, unobserved.

    Raises:
        ValueError: on a lot whose key is not in ``lot_keys`` or whose arrival week is outside 1..T (never dropped).

    """
    qty, seen = np.zeros((len(lot_keys), T), dtype=F8), np.zeros((len(lot_keys), T), dtype=I8)
    if cols is None:
        return qty, seen
    rows = {key: i for i, key in enumerate(lot_keys)}
    g = group_rows(DENSE_BLOCK, cols)
    for c, k, lane, nxt, w, q in zip(*(g[f] for f in _GROUP_KEYS[DENSE_BLOCK]), g["qty"]):
        i = rows.get((c, k, lane, nxt))
        if i is None or not 1 <= w <= T:
            raise ValueError(
                f"queue_lots: a lot at (chokepoint {c}, k {k}, lane {lane}, next edge {nxt}) arrived in week {w}, "
                f"outside the dense block (layout.lot_keys x weeks 1..{T}; flat.lot_weeks; never dropped)"
            )
        qty[i, w - 1], seen[i, w - 1] = q, 1
    return qty, seen


@dataclass(frozen=True)
class FlatLayout:
    """The fixed shapes of one instance's flat view, from its Static tables (and the caps of ``maxima``)."""

    n_slots: int  # len(Static.action_slots)
    n_override_slots: int  # len(Static.override_slots)
    pairs: tuple[tuple[int, int], ...]  # (chokepoint node, tanker commodity k) in node, then commodity order
    override_pair: tuple[int, ...]  # the pair index of each override slot
    n_units: int  # warning signal units: always R + D + C of Static (regions, dyads, chokepoints), any regime
    maxima: tuple[tuple[str, int], ...]  # (block, padded length) of each variable-length block
    # index sets of the densified blocks, from Static (and the stock slots of the instance it carries)
    n_edges: int = 0
    n_commodities: int = 0
    n_regions: int = 0
    n_dyads: int = 0
    stock_slots: tuple[tuple[int, int], ...] = ()  # (node, k) of the plain (non-chokepoint) stock slots, slot order
    supply_slots: tuple[tuple[int, int], ...] = ()  # (node, k) of the supply nodes' stock slots, slot order
    demands: tuple[tuple[int, int], ...] = ()  # (sink node, k) of Static.sinks
    chokepoints: tuple[int, ...] = ()  # chokepoint nodes by node index (their ordinal order)
    fabs: tuple[int, ...] = ()
    grids: tuple[int, ...] = ()
    osats: tuple[int, ...] = ()
    kind: str = "tiny"  # the instance kind whose caps (``maxima_for``) the layout took, named in overflow errors
    # pipeline grouped (``group_rows``) and queue_lots dense (``dense_lots``): GROUPED_KINDS by default (M5-O35 (c),
    # M5-O38 (b))
    grouped: bool = False
    # the dense queue_lots block's rows on a grouped layout (``lot_weeks``' keys, sorted), else (); its columns the
    # arrival weeks 1..T
    lot_keys: tuple[tuple[int, int, int, int], ...] = ()
    T: int = 0  # the horizon (Static.T)
    # (out-edge, k) of each override slot, slot order: the pair whose prohibition drops its entry (§9.3,
    # ``override_mask``; M5 re-gate INT-M5R-03)
    override_edge_k: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_static(
        cls,
        static: dict,
        maxima: dict[str, int] | None = None,
        *,
        inst: Instance | None = None,
        grouped: bool | None = None,
    ) -> "FlatLayout":
        """The layout of ``static``, padded to its instance kind's caps (``maxima_for``), which ``maxima`` overrides.

        ``inst`` is the instance ``static`` was built from, when the caller holds it (the gymnasium adapter); else it
        is loaded from ``static['instance']``, the canonical instance JSON Static carries. ``grouped`` None builds the
        grouped layout (the pipeline grouped, the queue lots dense) on the kinds of ``GROUPED_KINDS``; True or False
        forces it (the evidence and the tests compare both layouts; the list layout of `small` or `full` needs its
        pipeline and queue_lots caps in ``maxima``). A grouped layout pads no queue_lots list, so it has no cap there:
        its ``maxima`` leave the block out.

        Raises:
            ValueError: on a cap in ``maxima`` that is not an integer >= 0 or names a block that is not
                variable-length (on a grouped layout, queue_lots), a ``grouped`` that is not None or a bool, a
                padded block without a cap, a cap below a structural bound of the instance (``wip_bound``; on a
                grouped layout also ``pipeline_bound``), or a grouped layout of an instance whose lots can have
                arrived outside the weeks 1..T (an initial lot, arrival week <= 0), which the dense block has no
                column for.

        """
        if inst is None:  # the stock slot table is the loader's (§2.3), not a Static table
            inst = load_instance(static["instance"])
        if grouped is not None and not isinstance(grouped, bool):
            raise ValueError(f"grouped must be None, True or False, got {grouped!r}")
        grouped = inst.kind in GROUPED_KINDS if grouped is None else grouped
        padded = tuple(b for b in _VARIABLE if not (grouped and b == DENSE_BLOCK))
        caps = {b: n for b, n in maxima_for(inst.kind).items() if b in padded}
        for block, n in (maxima or {}).items():
            if grouped and block == DENSE_BLOCK:
                raise ValueError(
                    f"maxima: {block!r} is the dense block of a grouped layout (kind {inst.kind}), which has a cell "
                    "for every lot and no cap"
                )
            if block not in _VARIABLE:
                raise ValueError(f"maxima: {block!r} is not a variable-length block ({', '.join(_VARIABLE)})")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError(f"maxima[{block!r}] must be an integer >= 0, got {n!r}")
            caps[block] = n
        missing = [b for b in padded if b not in caps]
        if missing:
            raise ValueError(
                f"flat caps: the list layout of kind {inst.kind} has no cap for {', '.join(missing)} "
                "(flat.MAXIMA_BY_KIND): pass one in maxima"
            )
        bounds = {"wip": (wip_bound(inst), "wip_bound")}
        lot_keys: tuple[tuple[int, int, int, int], ...] = ()
        if grouped:
            bounds |= {"pipeline": (pipeline_bound(inst), "pipeline_bound")}
            weeks = lot_weeks(inst)
            outside = sorted({w for ws in weeks.values() for w in ws if not 1 <= w <= inst.T})
            if outside:
                raise ValueError(
                    f"flat layout: lots of {inst.instance_id} can have arrived in weeks {outside}, outside the dense "
                    f"queue_lots block's columns 1..{inst.T} (initial lots; flat.lot_weeks): no grouped layout"
                )
            lot_keys = tuple(sorted(weeks))
        for block, (bound, name) in bounds.items():
            if caps[block] < bound:
                raise ValueError(
                    f"flat caps: {block}'s cap {caps[block]} is below the structural bound {bound} of "
                    f"{inst.instance_id} (kind {inst.kind}; flat.{name}, flat.MAXIMA_BY_KIND)"
                )
        types = static["nodes"]["type"]
        chk = tuple(i for i, t in enumerate(types) if t == "chokepoint")
        tanker = [k for k, ov in enumerate(static["commodities"]["override"]) if ov]
        pairs = tuple((c, k) for c in chk for k in tanker)
        index = {p: i for i, p in enumerate(pairs)}
        ov = static["override_slots"]
        supply = set(inst.supply_nodes)
        return cls(
            n_slots=len(static["action_slots"]["edge"]),
            n_override_slots=len(ov["chokepoint"]),
            pairs=pairs,
            override_pair=tuple(index[(c, k)] for c, k in zip(ov["chokepoint"], ov["k"])),
            n_units=len(static["regions"]) + len(static["dyads"]["a"]) + len(chk),
            maxima=tuple((b, caps[b]) for b in padded),
            n_edges=len(static["edges"]["id"]),
            n_commodities=len(static["commodities"]["id"]),
            n_regions=len(static["regions"]),
            n_dyads=len(static["dyads"]["a"]),
            stock_slots=tuple((s.node, s.k) for s in inst.stock_slots if s.node not in chk),
            supply_slots=tuple((s.node, s.k) for s in inst.stock_slots if s.node in supply),
            demands=tuple(zip(static["sinks"]["node"], static["sinks"]["k"])),
            chokepoints=chk,
            fabs=tuple(i for i, t in enumerate(types) if t == "fab"),
            grids=tuple(i for i, t in enumerate(types) if t == "grid"),
            osats=tuple(i for i, t in enumerate(types) if t == "osat"),
            kind=inst.kind,
            grouped=grouped,
            lot_keys=lot_keys,
            T=int(static["T"]),
            override_edge_k=tuple(zip(ov["out_edge"], ov["k"])),
        )

    def cap(self, block: str) -> int:
        """The padded length of a variable-length block.

        Raises:
            ValueError: on the dense queue_lots block of a grouped layout, which has none (M5-O38 (b)).

        """
        caps = dict(self.maxima)
        if block not in caps and self.grouped and block == DENSE_BLOCK:
            raise ValueError(f"{block} is the dense block of a grouped layout (kind {self.kind}): no padded length")
        return caps[block]

    def fields(self, block: str) -> tuple[tuple[str, type, tuple[str, ...] | None], ...]:
        """(field, dtype, vocabulary or None) of a variable-length block on this layout, in §9.2 order.

        On a grouped layout, the grouped pipeline's five fields and the dense queue lots' one, ``qty``.
        """
        return (_GROUPED if self.grouped and block in _GROUPED else _VARIABLE)[block]

    def unit_position(self, kind: str, unit: int) -> int:
        """The warning block's position of a signal unit: regions, then dyads, then chokepoints by ordinal.

        Raises:
            ValueError: on a unit outside Static's tables.

        """
        if kind == "region" and 0 <= unit < self.n_regions:
            return unit
        if kind == "dyad" and 0 <= unit < self.n_dyads:
            return self.n_regions + unit
        if kind == "chokepoint" and unit in self.chokepoints:
            return self.n_regions + self.n_dyads + self.chokepoints.index(unit)
        raise ValueError(f"warning unit ({kind!r}, {unit!r}) is not a signal unit of Static")


# ----- the observation -----------------------------------------------------------------------------------------------
class _Out:
    """The arrays of one flat observation, in insertion (§9.2) order, each followed by its observed-mask."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}

    def put(self, key: str, values, observed, dtype) -> None:
        self.arrays[key] = np.asarray(values, dtype=dtype)
        self.arrays[f"{key}.observed"] = np.asarray(observed, dtype=I8)


def _code(value, vocabulary: tuple[str, ...] | None):
    return value if vocabulary is None else vocabulary.index(value)


def _dense(out: _Out, key: str, n: int, index: dict, cols: dict | None, keys: tuple[str, ...], value: str, dtype):
    """A column table scattered onto the fixed index set ``index`` ({key tuple: position}); unlisted entries 0."""
    vals, seen = np.zeros(n, dtype=dtype), np.zeros(n, dtype=I8)
    if cols is not None:
        seen[:] = 1  # an entry the table does not list is 0 and observed (stock, backlog, tariff are sparse)
        for *k, v in zip(*(cols[c] for c in keys), cols[value]):
            i = index[tuple(k)]
            vals[i], seen[i] = (0 if v is None else v), (v is not None)
    out.put(key, vals, seen, dtype)


def _by_node(out: _Out, prefix: str, nodes: tuple[int, ...], table: dict | None, fields: tuple[str, ...]) -> None:
    index = {n: i for i, n in enumerate(nodes)}
    for f in fields:
        vals, seen = np.zeros(len(nodes), dtype=F8), np.zeros(len(nodes), dtype=I8)
        if table is not None:
            for n, v in zip(table["node"], table[f]):
                vals[index[n]], seen[index[n]] = (0.0 if v is None else v), (v is not None)
        out.put(f"{prefix}.{f}", vals, seen, F8)


def _column(out: _Out, key: str, values: list | None, n: int, dtype, vocabulary=None) -> None:
    """A list of n values (null entries 0, unobserved); a null list all 0, unobserved."""
    vals, seen = np.zeros(n, dtype=dtype), np.zeros(n, dtype=I8)
    if values is not None:
        for i, v in enumerate(values):
            if v is not None:
                vals[i], seen[i] = _code(v, vocabulary), 1
    out.put(key, vals, seen, dtype)


def _variable(out: _Out, layout: FlatLayout, block: str, cols: dict | None) -> None:
    if layout.grouped and block == DENSE_BLOCK:  # a (lot key, arrival week) block, M5-O38 (b)
        out.put(f"{block}.qty", *dense_lots(layout.lot_keys, layout.T, cols), F8)
        return
    if layout.grouped and block in _GROUPED:
        cols = group_rows(block, cols)
    cap, fields = layout.cap(block), layout.fields(block)
    n = 0 if cols is None else len(cols[fields[0][0]])
    if n > cap:
        raise ValueError(
            f"{block}: {n} entries exceed its padded length {cap} on instance kind {layout.kind} "
            "(flat.maxima_for; never truncated)"
        )
    for f, dtype, vocabulary in fields:
        _column(out, f"{block}.{f}", None if cols is None else cols[f], cap, dtype, vocabulary)


def live_messages(messages: dict | None, week: int) -> dict | None:
    """The messages of the live threads in week ``week``'s cumulative §9.2 ``messages`` columns (Q105), in list order.

    A thread (one ``msg_id``) is live in week t when it has been announced (every message listed has an instant at or
    before t - 1), is not yet effective and is not withdrawn, as the observation shows them: no message of it is a
    withdrawal, and none states an effective week at or before t (``stated_effective_week`` is w(t_on), so a stated
    week <= t is an event that took effect by the instant t - 1, (1)). A thread whose messages state no week (a TIES or
    MID threat without a dated message) is live until it is withdrawn: the observation never shows when a real one
    takes effect, and the flat view shows only what the observation shows (design §12 "Flat view messages (Q105)").
    Every message of a live thread is kept. None stays None (the feed not shown, or a blackout week).
    """
    if messages is None:
        return None
    ids = messages["msg_id"]
    over = {
        m
        for m, kind, stated in zip(ids, messages["kind"], messages["stated_effective_week"])
        if kind == "withdrawal" or (stated is not None and stated <= week)
    }
    keep = [i for i, m in enumerate(ids) if m not in over]
    return {field: [column[i] for i in keep] for field, column in messages.items()}


def flatten_obs(layout: FlatLayout, obs: dict) -> dict[str, np.ndarray]:
    """The observation as fixed-shape arrays in §9.2 field order, each with its observed-mask.

    The ``messages`` block carries only the live threads (``live_messages``, Q105); the cumulative list stays in the
    §9.2 observation. On a grouped layout the pipeline carries its groups (``group_rows``, M5-O35 (c)) and the queue
    lots are the dense block of ``dense_lots`` (M5-O38 (b)), and the observation keeps its lists.

    Raises:
        ValueError: if a variable-length block exceeds its padded length (never truncated), a lot has no cell in the
            dense block, or an entry names an index outside Static's tables.

    """
    out = _Out()
    E, K, S = layout.n_edges, layout.n_commodities, layout.n_slots
    demand_index = {d: i for i, d in enumerate(layout.demands)}
    out.put("week", [obs["week"]], [1], I64)
    stock_index = {s: i for i, s in enumerate(layout.stock_slots)}
    _dense(out, "stock.qty", len(layout.stock_slots), stock_index, obs["stock"], ("node", "k"), "qty", F8)
    _dense(out, "backlog.qty", len(layout.demands), demand_index, obs["backlog"], ("node", "k"), "qty", F8)
    for block in ("pipeline", "queue_lots", "wip"):
        _variable(out, layout, block, obs[block])
    g = obs["graph_now"]
    C = len(layout.chokepoints)
    _column(out, "graph_now.u", None if g is None else g["u"], E, F8)
    _column(out, "graph_now.c", None if g is None else g["c"], E, F8)
    _column(out, "graph_now.tau", None if g is None else g["tau"], E, I64)
    ek = {(e, k): e * K + k for e in range(E) for k in range(K)}
    pro = None if g is None else {**g["prohibited"], "value": [1] * len(g["prohibited"]["edge"])}
    _dense(out, "graph_now.prohibited", E * K, ek, pro, ("edge", "k"), "value", I8)
    _dense(out, "graph_now.tariff", E * K, ek, None if g is None else g["tariff"], ("edge", "k"), "rate", F8)
    for key in ("graph_now.prohibited", "graph_now.tariff"):
        for name in (key, f"{key}.observed"):
            out.arrays[name] = out.arrays[name].reshape(E, K)
    _column(out, "graph_now.open", None if g is None else g["open"], C, F8)
    for pool in ("tb", "ct"):
        _column(out, f"graph_now.kappa.{pool}", None if g is None else g["kappa"][pool], C, F8)
    _column(out, "graph_now.war_risk", None if g is None else g["war_risk"], C, I64, WAR_RISK_CLASSES)
    supply_index = {s: i for i, s in enumerate(layout.supply_slots)}
    sup = None if g is None else g["supply"]
    _dense(out, "graph_now.supply.avail", len(layout.supply_slots), supply_index, sup, ("node", "k"), "avail", F8)
    if sup is not None:  # a supply slot is always listed: one unlisted is unobserved, not 0
        listed = {(n, k) for n, k in zip(sup["node"], sup["k"])}
        seen = out.arrays["graph_now.supply.avail.observed"]
        seen[[i for s, i in supply_index.items() if s not in listed]] = 0
    _by_node(out, "graph_now.fab", layout.fabs, None if g is None else g["fab"], ("R", "alpha_bar", "cap_eff"))
    _by_node(out, "graph_now.grid", layout.grids, None if g is None else g["grid"], ("G_bar", "y_bar"))
    _by_node(out, "graph_now.osat", layout.osats, None if g is None else g["osat"], ("R", "thr_eff"))
    mask = obs["slot_mask"]
    _column(out, "slot_mask", None if mask is None else [int(m) for m in mask], S, I8)
    lw = obs["last_week"]
    for f in ("requested", "executed"):  # by action slot, observed where the slot was requested last week
        vals, seen = np.zeros(S, dtype=F8), np.zeros(S, dtype=I8)
        if lw is not None:
            for s, v in zip(lw["clip"]["slot"], lw["clip"][f]):
                vals[s], seen[s] = v, 1
        out.put(f"last_week.clip.{f}", vals, seen, F8)
    cc = None if lw is None else lw["cost_components"]
    costs = None if cc is None else [cc[c] for c in COST_COMPONENTS]
    _column(out, "last_week.cost_components", costs, len(COST_COMPONENTS), F8)
    for f in ("demand", "served", "lost"):
        sinks = None if lw is None else lw["sinks"]
        _dense(out, f"last_week.sinks.{f}", len(layout.demands), demand_index, sinks, ("node", "k"), f, F8)
    _by_node(out, "last_week.shed", layout.grids, None if lw is None else lw["shed"], ("qty",))
    fc = obs["demand_forecast"]
    vals, seen = np.zeros((len(layout.demands), HORIZONS), dtype=F8), np.zeros((len(layout.demands), HORIZONS), I8)
    if fc is not None:
        for n, k, h, q in zip(fc["node"], fc["k"], fc["h"], fc["qty"]):
            if not 0 <= h < HORIZONS:
                raise ValueError(f"demand_forecast: horizon {h} outside 0..{HORIZONS - 1} (§9.2)")
            vals[demand_index[(n, k)], h], seen[demand_index[(n, k)], h] = q, 1
    out.put("demand_forecast.qty", vals, seen, F8)
    wn = obs["warning"]
    vals, seen = np.zeros(layout.n_units, dtype=F8), np.zeros(layout.n_units, dtype=I8)
    if wn is not None:
        for kind, unit, score in zip(wn["unit_kind"], wn["unit"], wn["score"]):
            i = layout.unit_position(kind, unit)
            vals[i], seen[i] = score, 1
    out.put("warning.score", vals, seen, F8)
    _variable(out, layout, "messages", live_messages(obs["messages"], obs["week"]))
    for block in ("pending_prohibitions", "closure_end"):
        _variable(out, layout, block, obs[block])
    return out.arrays


def action_mask(layout: FlatLayout, obs: dict) -> tuple[np.ndarray, int]:
    """The action mask of one observation, 1 = valid, and whether it was observed.

    (mask, observed): the (n_slots,) action mask, 1 = valid, and 1 if ``slot_mask`` was observed, else 0 (an
    all-valid mask in a blackout week or the final observation).
    """
    mask = obs["slot_mask"]
    if mask is None:
        return np.ones(layout.n_slots, dtype=I8), 0
    return np.array([0 if m else 1 for m in mask], dtype=I8), 1


def override_mask(layout: FlatLayout, obs: dict) -> tuple[np.ndarray, int]:
    """The override mask of one observation, 1 = valid, and whether it was observed (design §9.1; INT-M5R-03).

    (mask, observed): the (n_override_slots,) mask, 0 where the override slot's own (out-edge, k) is prohibited this
    week (``graph_now.prohibited``, Z_t: the entry §9.3 drops, "masked this week"), else 1; and 1 if ``graph_now`` was
    observed, else 0 with an all-valid mask (a blackout week, the final observation), as ``action_mask``.
    """
    g = obs["graph_now"]
    if g is None:
        return np.ones(layout.n_override_slots, dtype=I8), 0
    z = set(zip(g["prohibited"]["edge"], g["prohibited"]["k"]))
    return np.array([0 if ek in z else 1 for ek in layout.override_edge_k], dtype=I8), 1


def observation_arrays(layout: FlatLayout, obs: dict) -> dict[str, np.ndarray]:
    """The Dict observation of the gymnasium adapter and the agent kit: ``flatten_obs`` plus the two masks.

    ``action_mask`` is the (n_slots,) mask of ``action_mask`` (1 = valid) and ``action_mask.observed`` a (1,) int8
    array, 1 when ``slot_mask`` was observed; then ``override_mask``, the (n_override_slots,) mask of
    ``override_mask``, and ``override_mask.observed``, 1 when ``graph_now`` was observed (M5 re-gate INT-M5R-03; the
    arrays before them are unchanged).

    Raises:
        ValueError: as ``flatten_obs``.

    """
    arrays = flatten_obs(layout, obs)
    mask, observed = action_mask(layout, obs)
    arrays["action_mask"] = mask
    arrays["action_mask.observed"] = np.array([observed], dtype=I8)
    mask, observed = override_mask(layout, obs)
    arrays["override_mask"] = mask
    arrays["override_mask.observed"] = np.array([observed], dtype=I8)
    return arrays


# ----- the action ----------------------------------------------------------------------------------------------------
def _vector(x, n: int, what: str) -> np.ndarray:
    a = np.asarray(x)
    if a.shape != (n,):
        raise ValueError(f"{what}: shape {a.shape}, expected ({n},)")
    return a


def action_from_flat(
    layout: FlatLayout, week: int, flows: np.ndarray, override_qty: np.ndarray, release_mode: np.ndarray
) -> dict:
    """The §9.2 action of week ``week`` from the flat vectors (module docstring); Python ints and floats only.

    ``overrides`` is present when a mode-1 pair has override slots, ``hold`` when a pair is in mode 2; a pair in mode 1
    without override slots sends nothing (its default release stays on).

    Raises:
        ValueError: on vectors of the wrong length or a release mode outside the three.

    """
    flows = _vector(flows, layout.n_slots, "flows").astype(F8)
    override_qty = _vector(override_qty, layout.n_override_slots, "override_qty").astype(F8)
    modes = _vector(release_mode, len(layout.pairs), "release_mode")
    if not all(float(m) in RELEASE_MODES for m in modes.tolist()):
        raise ValueError(f"release_mode: every mode must be one of {RELEASE_MODES}, got {modes.tolist()}")
    modes = [int(m) for m in modes.tolist()]
    slots = [s for s, q in enumerate(flows.tolist()) if q != 0]
    action: dict = {"week": int(week), "flows": {"slot": slots, "qty": [float(flows[s]) for s in slots]}}
    ov = [o for o, p in enumerate(layout.override_pair) if modes[p] == RELEASE_OVERRIDE]
    if ov:
        action["overrides"] = {"slot": ov, "qty": [float(override_qty[o]) for o in ov]}
    held = [layout.pairs[p] for p, m in enumerate(modes) if m == RELEASE_HOLD]
    if held:
        action["hold"] = {"chokepoint": [int(c) for c, _k in held], "k": [int(k) for _c, k in held]}
    return action


def flat_from_action(layout: FlatLayout, action: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(flows, override_qty, release_mode) of a well-formed action, the inverse of ``action_from_flat`` (tests).

    A pair named in ``hold`` is mode 2 (a hold releases nothing, whatever overrides say), else mode 1 when one of its
    override slots is listed; the override qty of a pair not in mode 1 is 0.
    """
    flows = np.zeros(layout.n_slots, dtype=F8)
    for s, q in zip((action.get("flows") or {}).get("slot", []), (action.get("flows") or {}).get("qty", [])):
        flows[s] = q
    modes = np.zeros(len(layout.pairs), dtype=I64)
    override_qty = np.zeros(layout.n_override_slots, dtype=F8)
    ov = action.get("overrides") or {}
    for o, q in zip(ov.get("slot", []), ov.get("qty", [])):
        modes[layout.override_pair[o]] = RELEASE_OVERRIDE
        override_qty[o] = q
    index = {p: i for i, p in enumerate(layout.pairs)}
    hold = action.get("hold") or {}
    for c, k in zip(hold.get("chokepoint", []), hold.get("k", [])):
        modes[index[(c, k)]] = RELEASE_HOLD
    for o, p in enumerate(layout.override_pair):
        if modes[p] != RELEASE_OVERRIDE:
            override_qty[o] = 0.0
    return flows, override_qty, modes

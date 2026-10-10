"""The reset-time min-cost flow of (22): nominal delivered flows and the routes of naive's plan (design §3.6, §8.1).

At nominal capacities (u^0, kappa = k_c mu, supply availabilities, fab cap^0, OSAT thr), the steady weekly flow that
delivers the most packaged chips to the sinks and, among those, costs least in freight; grids receive their fuel burn
zeta_gk G-bar^0_g. F-bar_ik (22) is the packaged-chip flow into sink i; the routes (lanes through chokepoints, or single
internal edges) and their flows are what naive's step 1 plans on. Solved lexicographically with ``linprog``
(``highs-ds``): first the delivery, then the cost at that delivery (design §12, "Reset-time flow (22)"; Q84).

The static weekly LP (one variable per flow, no time index):

- ``x_ek`` on every non-coupling edge whose tail and head are not chokepoints, per permitted k not prohibited at reset;
- ``y_lk`` per lane and commodity permitted on every edge of the lane, no (edge, k) of the lane prohibited at reset.
  Cargo crosses chokepoints only on lanes (the queue keeps each lot's lane, §3.4), so an edge into or out of a
  chokepoint carries only lane flow, and a turn-back edge on no lane carries nothing (it needs an override, which the
  nominal flow and naive never use);
- supply lifts ``s_ik`` <= the availability varsigma-bar^0_ik at supply nodes (unlifted availability is lost, Q79);
- fab starts ``p_f`` <= cap^0_f: a wafer in is a raw chip out (no scrap at nominal);
- OSAT packaging ``xi_ik`` per raw chip k, sum over k <= thr_i: a raw chip in is a packaged chip out;
- sink deliveries ``D_ik`` per sink demand.

Rows: flow balance (in - out + lift + production - consumption - delivery = requirement) at every non-chokepoint node
and commodity, the requirement being zeta_gk G-bar^0_g for grid g's fuel k and 0 elsewhere (steady state: nothing
accumulates, nothing is disposed of); the joint capacity u^0_e of each edge over commodities and lanes (4); the
chokepoint throughput k_c mu_cb per pool b over the lanes through c (9) at open fraction 1. Stage 1 maximises
sum D; stage 2 minimises freight sum c^0_e (flow on e) with the row sum D >= D* of stage 1 added.

Readings (design §12 "Reset-time flow (22), formulation"): freight is c^0 on the flow on each edge, a lane paying for
every edge it uses (war-risk transit costs and tariffs are 0 at reset with no events). The §2.3 tables
``compatibility`` (source -> terminal) and ``use`` (material -> fab) are not read here, since the loader checks that the
edges and lanes realise exactly those pairs (``instance.io._check_tables``, §2.2): an edge or lane from a source to a
terminal, or from a material to a fab, whose pair its table does not list, or a listed pair that no edge or lane joins,
never loads, so the routes the edges, their permitted commodities and the lanes allow join exactly the tables' pairs.
Terminals conserve flow and their throughput is the capacity of their grid edge (§2.1). Routes are read from the
positive flows: a lane route per ``y_lk`` and an edge route per ``x_ek``, both ending at their destination node. Flows
at or below ``FLOW_TOL`` (a numerical tolerance, not a model parameter) are treated as zero.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from shockbench_flow.instance.schema import SUPPLY_TYPES, Instance


FLOW_TOL = 1e-9  # numerical: flows at or below this are zero when reading routes


@dataclass(frozen=True)
class NominalRoute:
    """One route of the reset-time flow: a lane through chokepoints or a single edge between non-chokepoint nodes."""

    dest: int  # node receiving k
    k: int
    first_edge: int
    lane: int | None  # lane when the route passes chokepoints
    edges: tuple[int, ...]
    flow: float  # weekly flow at nominal capacity


@dataclass(frozen=True)
class NominalFlow:
    """The reset-time flow of (22): its routes, edge flows and packaged-chip flows into sinks."""

    routes: tuple[NominalRoute, ...]
    edge_flow: dict[tuple[int, int], float]  # (edge, k) -> weekly flow
    sink_flow: dict[tuple[int, int], float]  # (sink node, packaged k) -> F-bar_ik of (22)


def nominal_flow(inst: Instance) -> NominalFlow:
    """Solve the reset-time flow of (22) and decompose it into routes.

    Raises:
        ValueError: if either stage of the LP has no optimal solution (for example a grid whose fuel burn cannot be
            met at nominal capacity).

    """
    chk = set(inst.chokepoints)
    prohibited = set(inst.prohibitions_at_reset)
    cols: list[tuple] = []
    ub: list[float | None] = []

    def var(key: tuple, upper: float | None = None) -> None:
        cols.append(key)
        ub.append(upper)

    for e, edge in enumerate(inst.edges):
        if edge.coupling or edge.tail in chk or edge.head in chk:
            continue
        for k in edge.K:
            if (e, k) not in prohibited:
                var(("x", e, k))
    for li, lane in enumerate(inst.lanes):
        for k in inst.lane_K[li]:
            if not any((e, k) in prohibited for e in lane.edges):
                var(("y", li, k))
    for s in inst.stock_slots:
        if s.supply > 0 and inst.nodes[s.node].type in SUPPLY_TYPES:
            var(("lift", s.node, s.k), s.supply)
    for f in inst.fabs:
        var(("p", f), inst.nodes[f].fab.cap0)
    for o in inst.osats:
        for raw in sorted(inst.nodes[o].osat.packages):
            var(("xi", o, raw))
    for dem in inst.demands:
        var(("D", dem.node, dem.k))
    col = {key: j for j, key in enumerate(cols)}

    # flow balance (equality) at every non-chokepoint node and commodity
    bal: dict[tuple[int, int], dict[int, float]] = defaultdict(lambda: defaultdict(float))
    rhs: dict[tuple[int, int], float] = defaultdict(float)
    for j, key in enumerate(cols):
        kind = key[0]
        if kind == "x":
            _, e, k = key
            bal[(inst.edges[e].tail, k)][j] -= 1.0
            bal[(inst.edges[e].head, k)][j] += 1.0
        elif kind == "y":
            _, li, k = key
            es = inst.lanes[li].edges
            bal[(inst.edges[es[0]].tail, k)][j] -= 1.0
            bal[(inst.edges[es[-1]].head, k)][j] += 1.0
        elif kind == "lift":
            bal[(key[1], key[2])][j] += 1.0
        elif kind == "p":
            fab = inst.nodes[key[1]].fab
            bal[(key[1], fab.input)][j] -= 1.0
            bal[(key[1], fab.product)][j] += 1.0
        elif kind == "xi":
            _, o, raw = key
            bal[(o, raw)][j] -= 1.0
            bal[(o, inst.nodes[o].osat.packages[raw])][j] += 1.0
        elif kind == "D":
            bal[(key[1], key[2])][j] -= 1.0
    for g in inst.grids:
        grid = inst.nodes[g].grid
        for k in grid.fuels:
            rhs[(g, k)] = grid.shares[k] * grid.deliverable  # the fuel burn zeta_gk G-bar^0_g, met exactly
            bal[(g, k)]  # a requirement with no inflow variable makes the LP infeasible, as it should
    eq_rows = [(dict(bal[key]), rhs[key]) for key in sorted(bal)]

    # joint edge capacity (4), chokepoint throughput per pool (9), OSAT throughput (19)
    ub_rows: list[tuple[dict[int, float], float]] = []
    on_edge: dict[int, dict[int, float]] = defaultdict(dict)
    through: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for j, key in enumerate(cols):
        if key[0] == "x":
            on_edge[key[1]][j] = 1.0
        elif key[0] == "y":
            _, li, k = key
            for e in inst.lanes[li].edges:
                on_edge[e][j] = 1.0
            for c in inst.lanes[li].chokepoints:
                through[(c, inst.commodity_pool[k])][j] = 1.0
    for e in sorted(on_edge):
        ub_rows.append((on_edge[e], inst.edges[e].u0))
    for c, b in sorted(through):
        ub_rows.append((through[(c, b)], inst.nodes[c].chokepoint.kappa0[b]))
    for o in inst.osats:
        row = {col[("xi", o, raw)]: 1.0 for raw in inst.nodes[o].osat.packages}
        ub_rows.append((row, inst.nodes[o].osat.thr))

    n = len(cols)
    d_cols = [j for j, key in enumerate(cols) if key[0] == "D"]
    c1 = np.zeros(n)
    c1[d_cols] = -1.0
    x1 = _solve(c1, eq_rows, ub_rows, ub, n, "stage 1 (largest delivery)")
    d_star = math.fsum(x1[j] for j in d_cols)

    freight = np.zeros(n)
    for j, key in enumerate(cols):
        if key[0] == "x":
            freight[j] = inst.edges[key[1]].c0
        elif key[0] == "y":
            freight[j] = math.fsum(inst.edges[e].c0 for e in inst.lanes[key[1]].edges)
    delivery = {j: -1.0 for j in d_cols}
    try:
        x2 = _solve(freight, eq_rows, ub_rows + [(delivery, -d_star)], ub, n, "stage 2 (least freight)")
    except ValueError:  # D* recomputed from a stage-1 vertex can exceed what stage 2 reaches by a few ulps
        slack = [(delivery, -d_star * (1.0 - FLOW_TOL))]
        x2 = _solve(freight, eq_rows, ub_rows + slack, ub, n, "stage 2 (least freight, delivery within FLOW_TOL)")

    routes: list[NominalRoute] = []
    edge_flow: dict[tuple[int, int], float] = defaultdict(float)
    for j, key in enumerate(cols):
        q = float(x2[j])
        if key[0] not in ("x", "y") or q <= FLOW_TOL:
            continue
        if key[0] == "x":
            _, e, k = key
            routes.append(NominalRoute(inst.edges[e].head, k, e, None, (e,), q))
            edge_flow[(e, k)] += q
        else:
            _, li, k = key
            es = inst.lanes[li].edges
            routes.append(NominalRoute(inst.lane_destination(li), k, es[0], li, es, q))
            for e in es:
                edge_flow[(e, k)] += q
    routes.sort(key=lambda r: (r.dest, r.k, r.first_edge, -1 if r.lane is None else r.lane))
    sink_flow = {(dem.node, dem.k): max(0.0, float(x2[col[("D", dem.node, dem.k)]])) for dem in inst.demands}
    return NominalFlow(tuple(routes), dict(edge_flow), sink_flow)


def _solve(c: np.ndarray, eq_rows: list, ub_rows: list, ub: list, n: int, stage: str) -> np.ndarray:
    """One stage of the lexicographic LP with HiGHS dual simplex; raises ValueError unless optimal."""
    a_eq, b_eq = _matrix(eq_rows, n) if eq_rows else (None, None)
    a_ub, b_ub = _matrix(ub_rows, n) if ub_rows else (None, None)
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, u) for u in ub],
        method="highs-ds",
    )
    if res.status != 0:
        raise ValueError(f"reset-time flow (22) {stage}: {'infeasible' if res.status == 2 else res.message}")
    return res.x


def _matrix(rows: list[tuple[dict[int, float], float]], n: int):
    r, cc, vv = [], [], []
    for i, (row, _) in enumerate(rows):
        for j, a in row.items():
            r.append(i)
            cc.append(j)
            vv.append(a)
    b = np.array([rhs for _, rhs in rows], dtype=float)
    return coo_matrix((vv, (r, cc)), shape=(len(rows), n)).tocsr(), b

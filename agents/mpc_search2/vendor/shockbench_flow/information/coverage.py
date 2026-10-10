"""The spatial-coverage rung (design §5.3 table; Q10, Q38 C10; W2-information §2.3).

Prediction-free, with the graph state shown only within ``h_cov`` hops of the policy's goods (Canadian traveller
regime at 0, stochastic CTP at 1). Readings of design §12 (M3 rows), the metric R9 of the observation map:

- Anchors at the instant t - 1, read from the core observation: the nodes holding the policy's stock or WIP (a
  nonzero quantity: the observation lists every WIP ledger row, and the simulator books a fab's row every week, even
  one that starts no lot, so a zero row holds none of the policy's goods; SPEC-M3-01), the edges carrying its
  pipeline shipments and the chokepoints holding its queue lots.
- Distance on the undirected supply graph, chokepoints counting as edges (a chokepoint and its incident edges are one
  element): an element carrying goods, or an edge whose tail node is an anchor, is at distance 0; any other element at
  1 + the least node distance of its endpoints to an anchor. So at h_cov = 0 an edge is seen from its tail node.
- Shown iff d <= h_cov. Masked (null) outside, exactly: ``graph_now.u``, ``c`` and ``tau`` per edge and the chokepoint
  state ``open``, ``kappa`` and ``war_risk``; every other field (prohibited, tariff, supply, fab, grid, osat,
  slot_mask, the own state) is kept. Plant damage is public (Q97), so neither ``graph_now.fab`` nor ``graph_now.osat``
  is masked; the blackout rung blanks both with the rest of ``graph_now``.

On ``tiny`` nearly every node holds goods and src_gulf -> chk shows chk at h = 0, so the rung is near-degenerate there;
its tests use constructed states (risk of the observation map).
"""

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np

from shockbench_flow.instance.schema import POOLS, Instance


UNREACHED = 10**9  # the distance of an element no anchor reaches (it is shown at no h_cov of the rung)
OWN_FIELDS = ("stock", "pipeline", "queue_lots", "wip")  # the own-state fields the anchors are read from


@dataclass(frozen=True)
class _Elements:
    """The elements of the metric on one instance (kept in ``Instance.memo``): plain edges, then chokepoints.

    Element i < n_plain is the plain edge ``plain[i]`` (no chokepoint end); element n_plain + ci is chokepoint ci with
    its incident edges. ``edge_elements[e]`` lists the elements edge e belongs to (two for an edge between two
    chokepoints), ``ends[x]`` the non-chokepoint nodes element x touches (its endpoints), ``edges[x]`` its edges.
    """

    n_plain: int
    plain: tuple[int, ...]
    edge_elements: tuple[tuple[int, ...], ...]
    ends: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]
    neighbours: tuple[tuple[int, ...], ...]  # per node: the nodes one element away (undirected)

    @staticmethod
    def build(inst: Instance) -> "_Elements":
        chk = set(inst.chokepoints)
        plain = tuple(e for e, ed in enumerate(inst.edges) if ed.tail not in chk and ed.head not in chk)
        n_plain = len(plain)
        edge_elements: list[list[int]] = [[] for _ in inst.edges]
        ends: list[tuple[int, ...]] = []
        edges: list[tuple[int, ...]] = []
        for i, e in enumerate(plain):
            edge_elements[e].append(i)
            ends.append(tuple(sorted({inst.edges[e].tail, inst.edges[e].head})))
            edges.append((e,))
        for ci, c in enumerate(inst.chokepoints):
            inc = tuple(sorted(set(inst.in_edges[c]) | set(inst.out_edges[c])))
            for e in inc:
                edge_elements[e].append(n_plain + ci)
            nodes = {inst.edges[e].tail for e in inc} | {inst.edges[e].head for e in inc}
            ends.append(tuple(sorted(n for n in nodes if n not in chk)))
            edges.append(inc)
        nbr: list[set[int]] = [set() for _ in inst.nodes]
        for x_ends in ends:
            for a in x_ends:
                nbr[a].update(b for b in x_ends if b != a)
        return _Elements(
            n_plain=n_plain,
            plain=plain,
            edge_elements=tuple(tuple(x) for x in edge_elements),
            ends=tuple(ends),
            edges=tuple(edges),
            neighbours=tuple(tuple(sorted(s)) for s in nbr),
        )


def _anchors(inst: Instance, core_obs: dict) -> tuple[set[int], set[int], set[int]]:
    """(anchor nodes, goods-carrying edges, chokepoint nodes holding lots) at the instant t - 1."""
    for name in OWN_FIELDS:
        if not isinstance(core_obs.get(name), dict):
            raise ValueError(f"the coverage rung reads the own-state field {name!r}, which the observation lacks")
    nodes = {int(n) for n, q in zip(core_obs["stock"]["node"], core_obs["stock"]["qty"]) if q != 0.0}
    nodes |= {int(n) for n, q in zip(core_obs["wip"]["node"], core_obs["wip"]["qty"]) if q != 0.0}
    edges = {int(e) for e in core_obs["pipeline"]["edge"]}
    lots = {int(c) for c in core_obs["queue_lots"]["chokepoint"]}
    return nodes, edges, lots


def coverage_distance(inst: Instance, core_obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hop distances of every edge and chokepoint from the policy's goods.

    Hop distances (int) from the anchors of ``core_obs``: per edge (E,) and per chokepoint ordinal (C,), by the metric
    of the module docstring; a large sentinel where no anchor is reachable. An element (a plain edge, or a chokepoint
    with its incident edges) is at 0 when it carries goods (a pipeline shipment on one of its edges, lots at its
    chokepoint) or one of its edges leaves an anchor node; else at 1 + the least node distance of its endpoints, the
    node distance being the hop count, one per element, from the nearest node of an anchor (an anchor node, or an
    endpoint of a goods-carrying element). An edge between two chokepoints takes the nearer of its two elements.

    Raises:
        ValueError: if ``core_obs`` lacks an own-state field.

    """
    el: _Elements = inst.memo("information.coverage", _Elements.build)
    nodes, goods_edges, lots = _anchors(inst, core_obs)
    n_el = len(el.ends)
    goods = [False] * n_el
    for e in goods_edges:
        for x in el.edge_elements[e]:
            goods[x] = True
    for c in lots:
        goods[el.n_plain + inst.chokepoint_ordinal[c]] = True
    zero = [goods[x] or any(inst.edges[e].tail in nodes for e in el.edges[x]) for x in range(n_el)]
    # node distances: multi-source BFS, one hop per element, from the anchor nodes and the goods' endpoints
    dist = [UNREACHED] * len(inst.nodes)
    queue: deque[int] = deque()
    seeds = set(nodes) | {n for x in range(n_el) if goods[x] for n in el.ends[x]}
    for n in sorted(seeds):
        dist[n] = 0
        queue.append(n)
    while queue:
        a = queue.popleft()
        for b in el.neighbours[a]:
            if dist[b] > dist[a] + 1:
                dist[b] = dist[a] + 1
                queue.append(b)
    d_el = [
        0 if zero[x] else min((1 + dist[n] for n in el.ends[x] if dist[n] < UNREACHED), default=UNREACHED)
        for x in range(n_el)
    ]
    d_edge = np.array([min(d_el[x] for x in el.edge_elements[e]) for e in range(len(inst.edges))], dtype=np.int64)
    d_chk = np.array(d_el[el.n_plain :], dtype=np.int64)
    return d_edge, d_chk


def apply_coverage(obs: dict, inst: Instance, h: int) -> dict:
    """Null the coverage rung's masked ``graph_now`` entries beyond ``h`` hops.

    A fresh observation equal to ``obs`` with the masked ``graph_now`` entries null where the distance exceeds
    ``h``; ``obs`` is not modified. The final observation (``graph_now`` None) is returned as a copy.

    Raises:
        ValueError: if ``h`` < 0.

    """
    out = copy.deepcopy(obs)
    mask_coverage(out, inst, h)
    return out


def mask_coverage(obs: dict, inst: Instance, h: int) -> None:
    """``apply_coverage`` in place: null ``obs``'s masked ``graph_now`` entries beyond ``h`` hops.

    For a caller that already holds a private copy (``information.observe.wrap``), so each week's observation is
    copied once (SIMP-M3-11). The own-state fields the distances read are not masked, so they are read from ``obs``
    itself; a final observation (``graph_now`` None) is left as it is.

    Raises:
        ValueError: if ``h`` < 0.

    """
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or h < 0:
        raise ValueError(f"the coverage radius h_cov must be an integer >= 0, got {h!r}")
    g = obs.get("graph_now")
    if g is None:
        return
    d_edge, d_chk = coverage_distance(inst, obs)
    for e in np.flatnonzero(d_edge > h).tolist():
        g["u"][e] = g["c"][e] = g["tau"][e] = None
    for ci in np.flatnonzero(d_chk > h).tolist():
        g["open"][ci] = g["war_risk"][ci] = None
        for b in POOLS:
            g["kappa"][b][ci] = None

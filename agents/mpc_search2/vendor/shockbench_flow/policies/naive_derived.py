"""Shared parts of the M4 naive-derived baselines that are not naive's own code (design §8.2; Q100 (3)).

``naive_parts`` names naive's private helpers and holds no new code; what ``nd``, ``sz_state_base_stock``,
``human_ref`` and ``greedy_lp`` share beyond them lives here: their threshold parameters and naive's step-4 books
rebuilt from an observation. ``naive.naive_action`` keeps its own copy of step 4, so naive's arithmetic and every
frozen cent stay in ``policies.naive``.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple

from shockbench_flow.instance.schema import Instance
from shockbench_flow.policies.naive import CLOSED_THRESHOLD, GIVE_UP_RATIO, check_thresholds
from shockbench_flow.policies.naive_parts import destination


@dataclass(frozen=True)
class ThresholdParams:
    """Naive's step-6 thresholds (the 0.6 cut is SYNTHETIC, §11 row 83), the base of each derived baseline's params.

    Each baseline declares its own frozen subclass, so ``registry.make_policy`` and the Hydra schema tell them apart.

    Raises:
        ValueError: as ``naive.check_thresholds``.

    """

    closed_threshold: float = CLOSED_THRESHOLD
    give_up_ratio: float = GIVE_UP_RATIO

    def __post_init__(self) -> None:
        check_thresholds(self.closed_threshold, self.give_up_ratio)


class Books(NamedTuple):
    """Naive's step-4 books of one observation, per (j, k): built as ``naive.naive_action`` builds them."""

    on_hand: dict[tuple[int, int], float]  # stock entries summed in observation order
    backlog: dict[tuple[int, int], float]
    in_transit: dict[tuple[int, int], list[float]]  # pipeline quantities bound for j, in observation order
    queued: dict[tuple[int, int], list[float]]  # queue lots whose lane ends at j, in observation order


def step4_books(inst: Instance, obs: dict, *, strict: bool = False) -> Books:
    """The step-4 books of ``obs`` in ``naive_action``'s dicts and order (``strict`` zips the columns strictly).

    Raises:
        ValueError: with ``strict``, if an observation table's columns differ in length.

    """
    on_hand: dict[tuple[int, int], float] = defaultdict(float)
    st = obs["stock"]
    for i, k, q in zip(st["node"], st["k"], st["qty"], strict=strict):
        on_hand[(i, k)] += q
    backlog: dict[tuple[int, int], float] = defaultdict(float)
    bl = obs.get("backlog") or {"node": [], "k": [], "qty": []}
    for i, k, q in zip(bl["node"], bl["k"], bl["qty"], strict=strict):
        backlog[(i, k)] += q
    in_transit: dict[tuple[int, int], list[float]] = defaultdict(list)
    pp = obs["pipeline"]
    for e, k, lane, q in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], strict=strict):
        in_transit[(destination(inst, e, lane), k)].append(q)
    queued: dict[tuple[int, int], list[float]] = defaultdict(list)
    ql = obs["queue_lots"]
    for lane, k, q in zip(ql["lane"], ql["k"], ql["qty"], strict=strict):
        queued[(inst.lane_destination(lane), k)].append(q)
    return Books(on_hand, backlog, in_transit, queued)


def step4_ip(books: Books, jk: tuple[int, int]) -> float:
    """IP_jk of step 4 in naive's order: on hand, ``+=`` the pipeline sum, ``+=`` the queued sum, ``-=`` the backlog."""
    ip = books.on_hand.get(jk, 0.0)
    ip += sum(books.in_transit.get(jk, []))
    ip += sum(books.queued.get(jk, []))
    ip -= books.backlog.get(jk, 0.0)
    return ip

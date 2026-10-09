"""agents/belief48's strait belief: recovery of closed straits, onsets after threats, and this week left as observed."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import ROOT


@pytest.fixture
def belief():
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    load_agent_class(ROOT / "agents" / "belief48", "belief_agent")
    try:
        yield sys.modules["belief"]
    finally:
        unload_agent()


def test_survival_is_a_survival_function(belief):
    S = belief.survival(np.arange(1, 60))
    assert S[0] == pytest.approx(1.0) and np.all(np.diff(S) <= 0) and S[-1] > 0


def test_a_closed_strait_reopens_in_expectation(belief):
    e = belief.expected_open(0.1, 1, None, 25)
    assert e.shape == (24,) and np.all((e >= 0.1) & (e <= 1.0)) and np.all(np.diff(e) >= 0)
    old = belief.expected_open(0.1, 30, None, 25)  # a long closure is more likely a persistent one
    assert np.all(old <= e + 1e-12)
    assert belief.expected_open(0.1, 1, None, 25, recovery=False) is None


def test_threats_lower_an_open_strait_and_fade_with_age(belief):
    young, stale = belief.expected_open(1.0, 0, 0, 13), belief.expected_open(1.0, 0, 20, 13)
    assert np.all(young <= 1.0) and young[-1] < stale[-1] <= 1.0
    assert belief.expected_open(1.0, 0, None, 13) is None  # no threat, base rate off: persistence stands
    assert np.all(belief.expected_open(1.0, 0, None, 13, base=True) <= 1.0)


def test_apply_leaves_this_week_as_observed(belief):
    class W:
        def __init__(self, H, C):
            self.H, self.o = H, np.ones((H, C))

        def set_open(self, pos, v, start, end):
            self.o[start:end, pos] = v

    w = W(10, 3)
    w.o[:, 1] = 0.0  # strait 1 closed in the persistence window
    s = SimpleNamespace(
        open_now=np.array([1.0, 0.0, 1.0]),
        closed_for=np.array([0, 2, 0]),
        week=5,
        threads=[{"channel": "mid_threat", "target_kind": "chokepoint", "target": 72, "announced_week": 5}],
        strait_pos=lambda node: {72: 2}.get(node),
    )
    assert belief.apply(w, s) == 2  # the closed strait and the threatened one
    assert w.o[0, 1] == 0.0 and w.o[0, 2] == 1.0  # row 0: as observed
    assert np.all(w.o[1:, 1] > 0.0) and np.all(w.o[1:, 2] < 1.0) and np.all(w.o[:, 0] == 1.0)

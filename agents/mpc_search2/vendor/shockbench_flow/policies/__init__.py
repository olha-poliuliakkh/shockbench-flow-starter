"""Baselines (design §8): `zero` and the naive anchor in milestone M1; the suite in M4."""

from shockbench_flow.policies.base import Policy, ZeroPolicy, empty_action
from shockbench_flow.policies.naive import NaiveFallback, NaivePolicy


__all__ = ["NaiveFallback", "NaivePolicy", "Policy", "ZeroPolicy", "empty_action"]

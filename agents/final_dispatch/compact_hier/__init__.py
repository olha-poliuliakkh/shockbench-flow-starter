"""The compact hierarchical planner: a pre-solve maritime controller and dynamic safety floors over a compact LP.

Modules, in the order a week runs them:

- ``config``: the knobs, defaults overridden by ``params.json`` beside ``agent.py``.
- ``network``: the static network read once from the agent's config (routes, stock slots, grids, tanker queues).
- ``forecast``: what the window expects each week (persistence, announced sanctions and reopenings, recoveries).
- ``maritime``: which routes may not be dispatched into a closed strait, and the fallback's rerouting.
- ``safety``: fuel safety floors, raised where a store's inbound routes are interrupted.
- ``compact_lp``: the window LP (``mpc_nobuf``'s formulation, base-load-first relaxed to continuous z) and its solve.
"""

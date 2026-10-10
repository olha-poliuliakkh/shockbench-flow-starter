"""The shim's conversions between the scorer's JSON messages and the Dict observation and action.

The Dict observation is exactly the gymnasium adapter's (``shockbench_flow_gym.ShockBenchFlowEnv``): the flat view's
arrays (``information.flat.flatten_obs``; the messages block holds the live announcement threads only; on `small` and
`full` the pipeline is grouped and the queued lots are a dense (lot key, arrival week) block) plus ``action_mask`` (1 =
valid) and ``action_mask.observed``, then ``override_mask`` and ``override_mask.observed``. One function builds both,
``flat.observation_arrays``, and tests hold the two equal week by week on every network. The Dict action is the
adapter's too: ``flows`` over the action slots, ``override_qty`` over the override slots and one ``release_mode`` per
(chokepoint, tanker commodity) pair (``flat.RELEASE_MODES``), turned into the protocol's action by
``flat.action_from_flat``. A chokepoint is a sea passage ships queue at: a strait, a canal or the Cape route.

Readings of a Dict action:

- ``override_qty`` and ``release_mode`` may be left out (zeros: the default release everywhere); ``flows`` may not.
  What cannot be paired into entries voids the week: not a mapping, a missing ``flows``, a vector of the wrong shape,
  or one that is not an array of numbers (text, objects, booleans) raises ``ValueError`` or ``TypeError`` here, and
  the shim replies with a null action: the naive rule plays that week.
- A release mode outside {0, 1, 2} (3, -1, 1.5, NaN; a whole float is its integer) is one pair's invalid entry,
  voided alone: the pair is sent as its override slots with a null quantity each, which the environment drops and
  counts as invalid entries, so the pair keeps its default release and the rest of the week stands. A pair without an
  override slot (none on ``tiny``) keeps its default release with nothing sent.
- A non-finite flow or override quantity is sent as null (a missing value), which the environment drops as one
  invalid entry, as it drops the same value in process; a negative one is sent as is and dropped the same way. So the
  line the shim writes decodes to the action the environment would have stored in process: the protocol changes
  nothing.
- ``agent_config`` is the ``config`` of ``Agent(config)``: the public tables of the episode, the information regime as
  published (null when the scorer hides it), T and the policy seed, plus the layout tables (what each position of a
  densified block stands for; on `small` and `full` also ``lot_keys``, the dense queue lots' rows) and the spaces'
  shapes, all derived from the public tables. Nothing hidden: under the clairvoyant regime the scenario is never
  passed on.

Imports: numpy and the flat view only (its instance loader and vocabularies come with it); never the simulator.
"""

import math
from collections.abc import Mapping

import numpy as np

from shockbench_flow.information.flat import (
    COST_COMPONENTS,
    RELEASE_MODES,
    FlatLayout,
    action_from_flat,
    observation_arrays,
)


ACTION_KEYS = ("flows", "override_qty", "release_mode")  # the Dict action of the gymnasium adapter
MASK_KEY, MASK_OBSERVED_KEY = "action_mask", "action_mask.observed"
OVERRIDE_MASK_KEY, OVERRIDE_MASK_OBSERVED_KEY = "override_mask", "override_mask.observed"


def observation_dict(layout: FlatLayout, obs: dict) -> dict[str, np.ndarray]:
    """The Dict observation of one protocol observation: the gymnasium adapter's, array for array.

    ``flat.observation_arrays``: the flat view's arrays, then ``action_mask``, ``action_mask.observed``,
    ``override_mask`` and ``override_mask.observed``.

    Raises:
        ValueError: as ``flat.flatten_obs`` (a variable-length block over its padded length, an index outside Static).

    """
    return observation_arrays(layout, obs)


def spaces(layout: FlatLayout, observation: Mapping[str, np.ndarray]) -> dict:
    """The shapes and dtypes of the Dict observation and the Dict action, as plain data.

    ``{"observation": {key: {"shape": [..], "dtype": str}}, "action": {...}}``; the observation's from a sample (every
    observation of an instance has the same shapes, the flat view's point).
    """
    obs = {k: {"shape": list(v.shape), "dtype": v.dtype.name} for k, v in observation.items()}
    action = {
        "flows": {"shape": [layout.n_slots], "dtype": "float64"},
        "override_qty": {"shape": [layout.n_override_slots], "dtype": "float64"},
        "release_mode": {"shape": [len(layout.pairs)], "dtype": "int64"},
    }
    return {"observation": obs, "action": action}


def layout_tables(layout: FlatLayout) -> dict:
    """What each position of the densified Dict blocks stands for, as plain lists (Static indices).

    ``stock_slots`` index ``stock.qty``; ``demands`` (sink node, k) index ``backlog.qty``, the rows of
    ``demand_forecast.qty`` and ``last_week.sinks.*``; ``supply_slots`` index ``graph_now.supply.avail``;
    ``chokepoints`` (the sea passages: straits, canals and the Cape route; the kit's docs call them straits) index
    ``graph_now.open``, ``graph_now.kappa.*`` and ``graph_now.war_risk``; ``fabs``, ``grids`` and ``osats`` index
    ``graph_now.fab.*``, ``graph_now.grid.*`` (and ``last_week.shed.qty``) and ``graph_now.osat.*``;
    ``warning_units`` ([kind, unit]) index ``warning.score``; ``release_pairs`` ([chokepoint, k]) index
    ``release_mode``. On a grouped layout (`small` and `full`) ``lot_keys`` ([chokepoint, k, lane, next_edge]) index
    the rows of the dense ``queue_lots.qty``, whose column w - 1 is the arrival week w; a list layout (`tiny`) has no
    such table. Action slots, override slots, edges and commodities keep the public tables' own order.
    """
    units = [["region", r] for r in range(layout.n_regions)] + [["dyad", d] for d in range(layout.n_dyads)]
    tables = {
        "stock_slots": [[int(n), int(k)] for n, k in layout.stock_slots],
        "supply_slots": [[int(n), int(k)] for n, k in layout.supply_slots],
        "demands": [[int(n), int(k)] for n, k in layout.demands],
        "chokepoints": [int(c) for c in layout.chokepoints],
        "fabs": [int(n) for n in layout.fabs],
        "grids": [int(n) for n in layout.grids],
        "osats": [int(n) for n in layout.osats],
        "warning_units": units + [["chokepoint", int(c)] for c in layout.chokepoints],
        "cost_components": list(COST_COMPONENTS),
        "release_pairs": [[int(c), int(k)] for c, k in layout.pairs],
    }
    if layout.grouped:
        tables["lot_keys"] = [[int(x) for x in key] for key in layout.lot_keys]
    return tables


def agent_config(static: dict, policy_seed: int, layout: FlatLayout, observation: Mapping[str, np.ndarray]) -> dict:
    """The ``config`` of ``Agent(config)`` for one episode (module docstring), as the scorer builds it.

    Keys: ``static`` (the episode's public tables, the public instance JSON under ``static['instance']``), ``regime``
    (the information regime's published parameters, or None), ``T``, ``policy_seed``, ``layout`` (``layout_tables``:
    ``layout['chokepoints']`` lists the sea passages, straits and canals), ``release_modes`` and ``spaces``.

    ``static`` is a Reset's (or ``reset``'s ``info['static']``), ``policy_seed`` the episode's seed, ``layout`` the
    episode's ``FlatLayout`` (``env.unwrapped.layout``) and ``observation`` its Dict observation (the first one). Under
    gymnasium, ``shockbench_flow_gym.agent_config_from_reset(env, obs, info)`` makes this call.
    """
    return {
        "static": static,
        "regime": static.get("regime"),
        "T": static["T"],
        "policy_seed": policy_seed,
        "layout": layout_tables(layout),
        "release_modes": {"default": RELEASE_MODES[0], "override": RELEASE_MODES[1], "hold": RELEASE_MODES[2]},
        "spaces": spaces(layout, observation),
    }


def _numbers(x, n: int, what: str) -> np.ndarray:
    """``x`` as a float64 vector of length ``n``; a bool or non-numeric array is refused."""
    a = np.asarray(x)
    if a.dtype.kind not in "iuf":  # b (bool), O (objects), U/S (text), c (complex) are not quantities
        raise TypeError(f"{what}: an array of numbers is required, got dtype {a.dtype}")
    if a.shape != (n,):
        raise ValueError(f"{what}: shape {a.shape}, expected ({n},)")
    return a.astype(np.float64)


def _json_safe(values: list) -> list:
    """Non-finite floats as None (the environment's JSON-safe copy; dropped as invalid entries)."""
    return [v if math.isfinite(v) else None for v in values]


def action_to_wire(layout: FlatLayout, week: int, action: object) -> dict:
    """The protocol's action of week ``week`` from a Dict action (module docstring); Python values only.

    Raises:
        TypeError: if ``action`` is not a mapping, or a vector is not numeric.
        ValueError: if ``flows`` is missing or a vector has the wrong length.

    """
    if not isinstance(action, Mapping):
        raise TypeError(f"the action must be a dict with {ACTION_KEYS}, got {type(action).__name__}")
    if "flows" not in action:
        raise ValueError("the action has no 'flows'")
    flows = _numbers(action["flows"], layout.n_slots, "flows")
    ov = action.get("override_qty")
    override_qty = (
        np.zeros(layout.n_override_slots) if ov is None else _numbers(ov, layout.n_override_slots, "override_qty")
    )
    modes = action.get("release_mode")
    modes = np.zeros(len(layout.pairs), dtype=np.int64) if modes is None else np.asarray(modes)
    if modes.dtype.kind not in "iuf":
        raise TypeError(f"release_mode: an array of integers is required, got dtype {modes.dtype}")
    if modes.shape != (len(layout.pairs),):
        raise ValueError(f"release_mode: shape {modes.shape}, expected ({len(layout.pairs)},)")
    void = [p for p, m in enumerate(modes.tolist()) if m not in RELEASE_MODES]  # NaN, 3, -1, 1.5 (1.0 == 1)
    clean = np.where(np.isin(np.arange(len(layout.pairs)), void), RELEASE_MODES[0], modes)
    wire = action_from_flat(layout, week, flows, override_qty, clean)
    wire["flows"]["qty"] = _json_safe(wire["flows"]["qty"])
    if "overrides" in wire:
        wire["overrides"]["qty"] = _json_safe(wire["overrides"]["qty"])
    voided = [o for o, p in enumerate(layout.override_pair) if p in void]  # null qty: dropped, counted as invalid
    if voided:
        ov = wire.setdefault("overrides", {"slot": [], "qty": []})
        entries = sorted([*zip(ov["slot"], ov["qty"]), *((o, None) for o in voided)])
        ov["slot"], ov["qty"] = [o for o, _q in entries], [q for _o, q in entries]
    return wire

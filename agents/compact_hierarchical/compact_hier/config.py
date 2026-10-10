"""The agent's knobs. ``load`` returns the defaults below, updated in this order:

1. ``params.json`` beside ``agent.py``;
2. the preset file named by the environment variable ``SBF_PARAMS_FILE`` (a file name in the agent's folder, or a
   path), e.g. ``params_baseline.json``;
3. environment variables ``SBF_PARAM_<KEY>`` (the key in upper case): ``SBF_PARAM_REROUTE=0``,
   ``SBF_PARAM_CRISIS=false``. Values are read as JSON (``0``, ``1.5``, ``true``, ``[1, 2]``); ``true`` and
   ``false`` in any case are booleans.

The LP's knobs keep ``mpc_nobuf``'s names and meanings (branch ``mpc-evolve``); the maritime controller's, the crisis
floors' and the CPU deadlines' are new. An unknown key anywhere (a file or a variable) is an error, so a typo cannot
pass silently, and every override in force is printed once to stderr when the agent loads.

The variables are read where the agent is loaded: ``sbf evaluate`` (this process or its workers) sees them; ``sbf
check`` and ``--docker`` run the agent in a child with a cleaned environment and always use ``params.json``, as the
server does.
"""

import json
import math
import os
import sys
from pathlib import Path


DEFAULTS = {
    # ---- the window and its objective (mpc_nobuf)
    "horizon_small": 16,  # weeks of the window on networks with T <= 52 (Small, Tiny); 24 measured +0.010 with binaries
    "horizon_full": 24,  # weeks of the window on longer networks (Full)
    "terminal_frac_small": 0.0,  # Small (T <= 52): end-of-window value of stock and goods in transit, x v_k, only
    #   while the window ends before T (0: salvage only; the scorer pays 1.5-3.6 % of fuel value for leftovers)
    "terminal_frac_full": 0.7,  # Full: the same (0.7 measured +0.25 on 8 Full episodes on twopass48, doc 5.15)
    "shortage_weight": 1.0,  # multiplies the shortage penalty pi
    "shed_weight": 1.0,  # multiplies the grids' value of lost load
    "holding_weight": 1.0,  # multiplies holding costs
    "closed_below": 0.05,  # a strait this open or less carries nothing in the forecast
    "reopen_trust": 0.8,  # share of a strait's nominal throughput expected from its announced end-of-closure week
    "queue_scale": 0.0,  # queue delay and holding expected at a partly open strait
    "recover_weeks": 0.0,  # a cut edge capacity expected back at nominal with this time constant (0: persists)
    "buffer_weight": 0.0,  # weight of the rationing buffer: a grid's rationed fuel kept at psi x ibar (0: ignore)
    "pipeline_closure": 1.0,  # 1: cargo bound for a closed strait arrives after its announced reopening, else never
    "tanker_control": 1.0,  # 1: the LP releases tanker cargo queued at straits itself (release_mode 1, override_qty)
    "base_first": 1.0,  # 1: base load first as continuous z in [0, 1] (shed <= y_bar (1 - z), fab energy <= E_max z)
    "short_price": 10.0,  # x v_k: a fuel segment burning below the grid's load factor (fuel kept back)
    "rule_chips": 0.0,  # 1: a base-stock rule ships wafers and chips, the LP plans fuel; 0: the LP ships all
    "rule_cover": 3.0,  # weeks of use a destination holds beyond lead + 1 (capped by its storage)
    "lot_value": 0.8,  # a lot started is worth this share of its chip's penalty pi (rule_chips 1)
    "tau_grid": 6.0,  # weeks: a cut G_bar recovers toward the grid's deliverable (0: persists)
    "tau_fab": 12.0,  # weeks: a fab's cut capacity recovers toward cap0 (0: persists)
    # ---- production aligned with the simulator (compact_lp.py; 0 builds the LP exactly as before)
    "align_chip_production": 1.0,  # 1: the parts below in force; 0: none of them
    "align_osat_week1": 1.0,  # 1: week-1 packaging fixed at the simulator's rule (raw chips on hand, throughput)
    "align_fab_week1": 1.0,  # 1: week-1 lot starts split as the simulator does (every fab of a grid the same share)
    "align_fab_second_solve": 1.0,  # 1: base load first in week 1 by a second solve when the plan breaks it
    "osat_hold_price": 1.0,  # x v_k per unit-week of raw chips held (or disposed) at OSATs: package, then ship
    "wafer_hold_price": 1.0,  # x v_wafer per unit-week of wafers held (or disposed) at fabs in weeks with headroom
    "two_pass_baseload": 0.0,  # 1: base load first in every window week by a second solve (off: week 1 at most)
    "two_pass_rounds": 1,  # rounds of bounds and solves with two_pass_baseload (1: the two-pass solve)
    # ---- the maritime controller (maritime.py)
    "reroute": 1.0,  # 1: bound dispatches into closed straits before the solve; 0: off (mpc_nobuf's LP alone)
    "reroute_open_below": 0.05,  # a strait this open or less counts as closed for routing
    "reroute_max_wait": 2,  # weeks cargo may wait at a closed strait for its announced reopening before it is blocked
    "strand_credit": 0.0,  # share of the terminal credit kept by tanker cargo queued at a strait closed past the window
    "reroute_fallback": 1.0,  # 1: when no plan comes back, move the fallback's flows off blocked routes
    # ---- fuel safety floors (safety.py)
    "safety_frac": 0.5,  # share of each fuel store's storage kept as a safety stock (0: off)
    "safety_price": 0.05,  # USD per unit-week below the floor, x v_k
    "floor_taper_weeks": 0,  # the floors fall linearly to 0 over the episode's last N weeks (0: constant to T)
    "crisis": 1.0,  # 1: raise the floors of stores whose inbound fuel routes are interrupted; 0: static floors
    "crisis_frac": 0.7,  # a store in crisis keeps this share of its storage
    "crisis_price_mult": 3.0,  # a store in crisis pays this multiple of safety_price per unit-week short
    "crisis_share": 0.0,  # a store is in crisis when more than this share of its inbound route capacity is interrupted
    "crisis_cut_below": 0.5,  # a route with an edge below this share of its largest capacity seen counts as interrupted
    "crisis_hops": 1,  # stores this many shipments downstream of a store in crisis are in crisis too
    "crisis_voll_cap": 0.5,  # a grid store's crisis price over the whole window stays below this share of its VOLL
    # ---- CPU (seconds of the agent's own CPU per week; the budgets are 2 s Small and 4 s Full)
    "deadline_small": 1.5,  # the solve stops here on T <= 52 and the week falls back
    "deadline_full": 3.2,  # the same on Full
}


FILE_VAR = "SBF_PARAMS_FILE"
PARAM_PREFIX = "SBF_PARAM_"


def parse_value(key: str, text: str):
    """An environment variable's value: ``true``/``false`` (any case) as bools, else JSON (a finite number, a list)."""
    word = text.strip()
    if word.lower() in ("true", "false"):
        return word.lower() == "true"
    try:
        value = json.loads(word)
    except json.JSONDecodeError:
        raise ValueError(f"{PARAM_PREFIX}{key.upper()}={text!r}: not a number, a boolean or JSON") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{PARAM_PREFIX}{key.upper()}={text!r}: not finite")
    if value is None or isinstance(value, (str, dict)):
        raise ValueError(f"{PARAM_PREFIX}{key.upper()}={text!r}: expected a number, a boolean or a list")
    return value


def _read(path: Path) -> dict:
    extra = json.loads(path.read_text())
    unknown = sorted(set(extra) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"{path.name}: unknown keys {unknown}")
    return extra


def load(folder: Path, environ=None) -> dict:
    """The defaults, updated by ``params.json``, the ``SBF_PARAMS_FILE`` preset and the ``SBF_PARAM_*`` variables."""
    env = os.environ if environ is None else environ
    folder = Path(folder)
    params = dict(DEFAULTS)
    if (folder / "params.json").is_file():
        params.update(_read(folder / "params.json"))
    applied = {}
    preset = env.get(FILE_VAR, "").strip()
    if preset:
        path = Path(preset) if Path(preset).is_absolute() else folder / preset
        if not path.is_file():
            raise FileNotFoundError(f"{FILE_VAR}={preset!r}: no such file ({path})")
        applied[FILE_VAR] = path.name
        params.update(_read(path))
    by_upper = {key.upper(): key for key in DEFAULTS}
    for name, text in env.items():
        if not name.startswith(PARAM_PREFIX):
            continue
        key = by_upper.get(name[len(PARAM_PREFIX) :])
        if key is None:
            raise KeyError(f"{name}: unknown parameter (the keys are {sorted(DEFAULTS)})")
        params[key] = parse_value(key, text)
        applied[name] = params[key]
    if applied:
        print(f"compact_hierarchical: overrides in force {applied}", file=sys.stderr)
    return params

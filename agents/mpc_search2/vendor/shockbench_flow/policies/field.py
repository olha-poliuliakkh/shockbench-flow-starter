"""The leaderboard field: participant-like entries between naive (RSS 0) and the best baseline (branch lb-field).

How many episodes the hackathon's boards need depends on the gaps between entries like a participant's, near-ties above
all, which the §2.6 baseline pairs and the kit's two samples (both below naive on `small` and `full`) do not show. The
field adds entries of two kinds, each built from a baseline's or the kit's real parameters:

- **same-family variants**, close to a baseline by construction: ``mpc_det`` with a fixed window of ``H_weeks`` weeks
  (the Q67 sweep only lengthens H from L) or with a safety factor on its point forecast (``WindowSpec``: the demand
  it plans for times ``demand_scale``, the chokepoint throughput times ``throughput_scale``); the existing parameters
  (``mpc_det``'s H and ``planning_rules``, ``greedy_lp``'s thresholds, ``human_ref``'s Sterman parameters) need no
  code here and are played as ``policy/`` configs;
- **different-mechanism entries**, for near-ties and a continuum of scores: ``BlendSpec``, which each week plays
  ``mpc_det``'s action with probability p and the kit heuristic's otherwise; and ``HeuristicSpec``, the kit heuristic
  (``docs/hackathon/samples/heuristic``) with its two SYNTHETIC constants changed, as a participant tuning the starting
  kit would. The tuning root showed those constants move the kit heuristic by at most 0.3 RSS, all of it below -1.7
  (docs/evidence/lb_field/explore_*_r1.txt), so no ``HeuristicSpec`` is in ``FIELD``: the field's tuned heuristic is
  ``human_ref`` with a tuned desired stock (``s_prime``), an existing parameter.

``FIELD`` names the entries the pilots play that need code here (``registry.FIELD`` lists the same names, so the
registry builds them by name without importing this module until then); ``build`` makes a policy of any spec (the
evidence scripts' candidates).

**Seeds** (as the kit agents'): each entry declares ``seed_id``, so the runner salts its policy seed with it
(``evaluation.runner.episode_policy_seed``, the trusted runner's rule (27) for a submission): a tuned heuristic's is the
SHA-256 of its submission zip (``kit_agents.build_zip`` of its ``agent.py``), any other entry's the SHA-256 of its
descriptor text (``descriptor``: this module, the name and the spec's repr), so a changed parameter is a changed seed.
A blend's weekly choices are drawn once at reset from that seed alone (``blend_coins``: numpy's ``SeedSequence`` of
(seed, ``BLEND_STREAM``)), so an episode's trajectory is the same in any process, worker or order (V1). ``provenance``
records the descriptor (and a tuned heuristic's files), which the runner writes into ``setup["policy_provenance"]``.

A blend runs both parts every week, so ``mpc_det``'s memory of the observed graph and its basis chain follow the
episode whichever action is played; its per-week CPU is ``mpc_det``'s plus the heuristic's. A heuristic week that
raises or returns a malformed action is null (the shim's rule), and a blend playing it passes the null on: the
environment's D9 fallback, as for a kit agent (§9.3).

Like the kit agents, these are the organisers' entries, played in the trusted process; the heuristic parts are read
from the checkout (``kit_agents.checkout_root``), so the entry points and the tests run from it.
"""

import dataclasses
import functools
import hashlib
import math
import re
import sys
import tempfile
import types
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shockbench_flow.instance.schema import FrozenDict
from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
from shockbench_flow.policies.registry import PolicyContext


SOURCE = "shockbench_flow.policies.field"  # the descriptor's first word and the provenance's source
BLEND_STREAM = 0x626C656E64  # the blend coins' stream word ("blend"); a constant, not a model value
HEURISTIC = "sample_heuristic"  # the kit agent a blend mixes in and a tuned heuristic is made from
KIT_COVER_WEEKS = 1.0  # the kit heuristic's own constants (docs/hackathon/samples/heuristic/agent.py), SYNTHETIC there
KIT_CLOSED = 0.5


@dataclass(frozen=True)
class WindowSpec:
    """An ``mpc_det`` variant: a fixed window and a safety factor on the point forecast it plans on.

    Attributes:
        H_weeks: the window length in weeks, H_t = min(H_weeks, T - t + 1); None keeps ``mpc_det``'s H = L.
        demand_scale: the forecast demand of every window week times this (> 1 plans for more than forecast).
        throughput_scale: every chokepoint's forecast throughput kappa times this (< 1 plans for less than observed).

    Raises:
        ValueError: on H_weeks not None and not an integer >= 1, or a scale that is not a positive finite number.

    """

    H_weeks: int | None = None
    demand_scale: float = 1.0
    throughput_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.H_weeks is not None and (
            isinstance(self.H_weeks, bool) or not isinstance(self.H_weeks, int) or self.H_weeks < 1
        ):
            raise ValueError(f"H_weeks must be None or an integer >= 1, got {self.H_weeks!r}")
        for name in ("demand_scale", "throughput_scale"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < math.inf:
                raise ValueError(f"{name} must be a positive finite number, got {v!r}")


@dataclass(frozen=True)
class BlendSpec:
    """Each week ``mpc_det``'s action with probability ``p``, else the kit heuristic's (``blend_coins``).

    Raises:
        ValueError: if ``p`` is not a number in [0, 1].

    """

    p: float

    def __post_init__(self) -> None:
        if isinstance(self.p, bool) or not isinstance(self.p, (int, float)) or not 0 <= self.p <= 1:
            raise ValueError(f"p must be a number in [0, 1], got {self.p!r}")


@dataclass(frozen=True)
class HeuristicSpec:
    """The kit heuristic with its two constants set: ``COVER_WEEKS`` and ``CLOSED`` of its ``agent.py``.

    Raises:
        ValueError: if ``cover_weeks`` is not a finite number or ``closed`` not a number in [0, 1].

    """

    cover_weeks: float = KIT_COVER_WEEKS
    closed: float = KIT_CLOSED

    def __post_init__(self) -> None:
        if isinstance(self.cover_weeks, bool) or not isinstance(self.cover_weeks, (int, float)):
            raise ValueError(f"cover_weeks must be a finite number, got {self.cover_weeks!r}")
        if not math.isfinite(self.cover_weeks):
            raise ValueError(f"cover_weeks must be a finite number, got {self.cover_weeks!r}")
        if isinstance(self.closed, bool) or not isinstance(self.closed, (int, float)) or not 0 <= self.closed <= 1:
            raise ValueError(f"closed must be a number in [0, 1], got {self.closed!r}")


Spec = WindowSpec | BlendSpec | HeuristicSpec

# The pilots' field entries that need code here, read off the tuning root's candidates (docs/evidence/lb_field/README,
# scripts/python/evidence/lb_field_explore.py), each value SYNTHETIC(placeholder): a participant's choice, no model
# value. The field's other entries are existing parameters, played as policy/ configs (configs/experiment/lb_field_*).
# No HeuristicSpec: the kit heuristic's two constants moved it by at most 0.3 RSS, all of it below -1.7 on both sizes,
# so a tuned heuristic of the field is human_ref's s_prime instead. registry.FIELD lists the same names.
FIELD: Mapping[str, Spec] = FrozenDict(
    {
        "mpc_det_safety": WindowSpec(demand_scale=1.1),  # plans for 10 % more demand than forecast
        "mpc_det_h20": WindowSpec(H_weeks=20),  # a window 4 weeks shorter than L = 24 (`small`, `full`)
        "mpc_det_h12": WindowSpec(H_weeks=12),  # half of L
        "blend_p95": BlendSpec(0.95),
        "blend_p90": BlendSpec(0.9),  # `full`'s field: 0.70 and 0.60 fell below naive on `full`'s pilot
        "blend_p85": BlendSpec(0.85),
        "blend_p80": BlendSpec(0.8),
        "blend_p70": BlendSpec(0.7),  # `small`'s field
        "blend_p60": BlendSpec(0.6),  # 0.5 scored -0.30 on `full` and 0.00 on `small`: below or at naive
    }
)


def descriptor(name: str, spec: Spec) -> str:
    """The text an entry's seed and provenance name: this module, the entry's name and its spec's repr."""
    return f"{SOURCE}:{name}:{spec!r}"


def descriptor_seed_id(name: str, spec: Spec) -> str:
    """The SHA-256 of ``descriptor``: the seed salt of an entry that is not a submission zip (module docstring)."""
    return hashlib.sha256(descriptor(name, spec).encode()).hexdigest()


def blend_coins(policy_seed: int, T: int, p: float) -> np.ndarray:
    """(T,) bool, week t's entry True when the blend plays ``mpc_det`` in week t (index t - 1): U_t < p.

    U_1..U_T are the uniforms of numpy's ``PCG64`` on ``SeedSequence([policy_seed, BLEND_STREAM])``, drawn at once, so
    week t's choice depends on the seed and t alone; p 1 is every week, p 0 none.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(policy_seed), BLEND_STREAM])))
    return rng.random(int(T)) < p


class MpcDetVariant(MpcDet):
    """``mpc_det`` with a ``WindowSpec``: its horizon and window arrays through ``MpcDet``'s seams, nothing else."""

    def __init__(self, name: str, spec: WindowSpec, context: PolicyContext | None = None) -> None:
        super().__init__(MpcDetParams(), context)
        self.name, self.spec = name, spec
        self.seed_id = descriptor_seed_id(name, spec)
        self.provenance = {"source": SOURCE, "descriptor": descriptor(name, spec), "seed_id": self.seed_id}

    def __reduce__(self) -> tuple:
        return (MpcDetVariant, (self.name, self.spec, self.context))

    def _horizon(self, inst, plan) -> int:
        return super()._horizon(inst, plan) if self.spec.H_weeks is None else self.spec.H_weeks

    def _window_arrays(self, inst, obs: dict, H_t: int) -> dict:
        arrays = super()._window_arrays(inst, obs, H_t)
        scaled = {"demand": self.spec.demand_scale, "kappa": self.spec.throughput_scale}
        scaled |= {"kappa_now": self.spec.throughput_scale}
        if all(v == 1 for v in scaled.values()):
            return arrays
        out = dict(arrays)
        for key, factor in scaled.items():
            if factor != 1:
                out[key] = arrays[key] * factor
                out[key].flags.writeable = False
        return out


class Blend:
    """``BlendSpec``: ``mpc_det`` and the kit heuristic side by side, one's action per week by ``blend_coins``."""

    def __init__(self, name: str, spec: BlendSpec, context: PolicyContext | None = None) -> None:
        from shockbench_flow.policies.kit_agents import KitAgentPolicy

        self.name, self.spec = name, spec
        self.context = PolicyContext() if context is None else context
        self.seed_id = descriptor_seed_id(name, spec)
        self.provenance = {"source": SOURCE, "descriptor": descriptor(name, spec), "seed_id": self.seed_id}
        self._mpc = MpcDet(MpcDetParams(), self.context)
        self._heuristic = KitAgentPolicy(HEURISTIC)
        self._coins = np.zeros(0, dtype=bool)

    def __reduce__(self) -> tuple:
        return (Blend, (self.name, self.spec, self.context))

    @property
    def telemetry(self) -> list:
        """``mpc_det``'s solver telemetry, every week (its solves run whichever action is played)."""
        return self._mpc.telemetry

    @property
    def errors(self) -> list[tuple[int, str]]:
        """The heuristic's null weeks (``KitAgentPolicy.errors``), played or not."""
        return self._heuristic.errors

    @property
    def coins(self) -> np.ndarray:
        """The current episode's weekly choices (``blend_coins``; empty before a reset)."""
        return self._coins

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self._mpc.reset(static, obs, policy_seed)
        self._heuristic.reset(static, obs, policy_seed)
        self._coins = blend_coins(policy_seed, self._mpc._inst.T, self.spec.p)

    def act(self, obs: dict) -> dict | None:
        mpc, heuristic = self._mpc.act(obs), self._heuristic.act(obs)
        return mpc if self._coins[int(obs["week"]) - 1] else heuristic


def heuristic_source(spec: HeuristicSpec, root: str | Path | None = None) -> str:
    """The kit heuristic's ``agent.py`` with ``COVER_WEEKS`` and ``CLOSED`` set to the spec's values.

    Each constant's line is replaced whole (its value and its comment), so the text differs from the kit's in those two
    lines only.

    Raises:
        ValueError: if either constant's line is not in the kit's file exactly once.

    """
    from shockbench_flow.policies.kit_agents import agent_source

    text = (agent_source(HEURISTIC, root) / "agent.py").read_text()
    for constant, value in (("COVER_WEEKS", spec.cover_weeks), ("CLOSED", spec.closed)):
        line = f"{constant} = {float(value)!r}  # set by {SOURCE} (HeuristicSpec)"
        text, n = re.subn(rf"^{constant} = .*$", lambda _m, line=line: line, text, flags=re.M)
        if n != 1:
            raise ValueError(f"the kit heuristic's agent.py sets {constant} {n} times, not once")
    return text


@functools.cache
def _heuristic_build(name: str, spec: HeuristicSpec, root: str) -> tuple[type, dict]:
    """The tuned agent's ``Agent`` class (module ``sbf_field_<name>``, once per process) and its provenance.

    The provenance is the kit agents' (``kit_agents.provenance``): the zip ``kit_agents.build_zip`` makes of the tuned
    ``agent.py``, its SHA-256 the seed salt, and each zipped file's SHA-256; beside them this module's descriptor.
    """
    from shockbench_flow.policies.kit_agents import build_zip

    text = heuristic_source(spec, root)
    with tempfile.TemporaryDirectory(prefix="sbf-field-") as tmp:
        folder = Path(tmp) / name
        folder.mkdir()
        (folder / "agent.py").write_text(text)
        path = build_zip(folder, Path(tmp) / f"{name}.zip")
        with zipfile.ZipFile(path) as zf:
            files = {m: hashlib.sha256(zf.read(m)).hexdigest() for m in sorted(zf.namelist())}
        seed_id = hashlib.sha256(path.read_bytes()).hexdigest()
    module = types.ModuleType(f"sbf_field_{name}")
    module.__file__ = f"<{SOURCE}:{name}>"
    sys.modules[module.__name__] = module
    exec(compile(text, module.__file__, "exec"), module.__dict__)  # the kit's own code, two constants changed
    provenance = {"source": SOURCE, "descriptor": descriptor(name, spec), "seed_id": seed_id, "files": files}
    return module.Agent, provenance


class TunedHeuristic:
    """``HeuristicSpec``: the tuned kit heuristic behind the kit's shim, in process, as ``KitAgentPolicy`` plays one."""

    def __init__(self, name: str, spec: HeuristicSpec, root: str | Path | None = None) -> None:
        from shockbench_flow.policies.kit_agents import checkout_root
        from shockbench_flow_agent.shim import AgentShim

        self.name, self.spec = name, spec
        self.root = checkout_root(root)
        agent, provenance = _heuristic_build(name, spec, str(self.root))
        self.provenance = dict(provenance)
        self.seed_id = provenance["seed_id"]
        self._shim = AgentShim(agent)

    def __reduce__(self) -> tuple:
        return (TunedHeuristic, (self.name, self.spec, str(self.root)))

    @property
    def errors(self) -> list[tuple[int, str]]:
        return self._shim.errors

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self._shim.reset(static, obs, policy_seed)

    def act(self, obs: dict) -> dict | None:
        return self._shim.act(obs)


def build(name: str, spec: Spec, context: PolicyContext | None = None):
    """A fresh policy named ``name`` playing ``spec`` (a new object per episode, as every registry policy).

    Raises:
        TypeError: if ``spec`` is not a ``WindowSpec``, ``BlendSpec`` or ``HeuristicSpec``.

    """
    if isinstance(spec, WindowSpec):
        return MpcDetVariant(name, spec, context)
    if isinstance(spec, BlendSpec):
        return Blend(name, spec, context)
    if isinstance(spec, HeuristicSpec):
        return TunedHeuristic(name, spec)
    raise TypeError(f"a field spec is a WindowSpec, BlendSpec or HeuristicSpec, got {type(spec).__name__}")


def make_entry(name: str, context: PolicyContext | None = None):
    """The ``FIELD`` entry ``name`` as a fresh policy (``registry.make_policy``'s path for the field's names).

    Raises:
        ValueError: on a name outside ``FIELD``.

    """
    if name not in FIELD:
        raise ValueError(f"unknown field entry {name!r}: one of {tuple(FIELD)}")
    return build(name, FIELD[name], context)


def spec_record(spec: Spec) -> dict:
    """A spec as plain data for a summary: its class name and fields."""
    return {"kind": type(spec).__name__, **dataclasses.asdict(spec)}

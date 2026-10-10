"""The one name-to-policy map of the baseline suite (design §8.2; M4), shared by the runner, scripts and tests.

A baseline is built fresh for every episode (``make_policy``), never reused across episodes: the LP baselines keep a
warm-start basis inside an episode (Q95), and one object carried into the next episode would make episode n depend on
episode n - 1 and on the worker schedule (V1). Every M4 baseline takes ``(params, context)``: its own frozen parameter
dataclass (Q18: the Hydra boundary maps ``configs/policy/<name>.yaml`` onto it; the yaml carries no literal parameter
default, the schema takes the defaults from ``params_class(name)``, so a calibrated default such as ``mpc_scen``'s S
never leaves a stale literal that renames the canonical run) and a ``PolicyContext`` of small public inputs. Naive's
parameters (``naive.NaiveParams``, its step-6 cut) are built here too, so no entry point special-cases a baseline and
one config gives one policy on every path (SIM-M4-01); only ``zero`` has none. A policy
object holds no instance: ``reset`` loads it from ``static["instance"]`` under strict provenance
(``naive_parts.cached_plan``), so it pickles small for the wire runner (``information.runner.child_argv``, which
hands the pickle over a file; design §12 "Subprocess transport": an ``Instance`` pickles to about 83 kB). Solver
objects (highspy's ``Highs``, which does not pickle) are created in ``reset``.

The baseline modules are imported lazily, only by ``make_policy`` and ``params_class``: ``dynamics.env``,
``information.runner`` and ``disruption.realism`` import ``shockbench_flow.policies``, and an eager import here would
load highspy and the scenario code on every ``Env`` import (and risk a policies <-> disruption import cycle).

Besides the baselines, the registry builds the hackathon kit's own agents by name (``KIT_AGENTS``,
``policies.kit_agents``: the M5 pilot's participant-like entries, played behind the kit's shim in process), so the
runner plays them on the same episodes as the baselines; they are not baselines of §8.2 and stay out of ``BASELINES``.
The leaderboard field's entries (``FIELD``, ``policies.field``: participant-like entries between naive and the best
baseline, for the boards' split sizes) are built by name the same way, without parameters, and stay out of it too.
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass

from shockbench_flow.instance.schema import Instance, frozen_map
from shockbench_flow.policies.base import Policy, ZeroPolicy
from shockbench_flow.policies.naive import NaiveParams, NaivePolicy, canonical_quantiles
from shockbench_flow.policies.naive_fq import REPLICATIONS, check_replications


# every baseline the M4 suite builds (design §8.2 table; plan M4 scope), in the order reports list them
BASELINES = (
    "zero",
    "naive",
    "nd",
    "sz_state_base_stock",
    "human_ref",
    "greedy_lp",
    "mpc_det",
    "mpc_scen",
    "hindsight_consensus",
)
# the hackathon kit's own agents as policies (``policies.kit_agents``): the M5 pilot's participant-like entries
# (docs/099-split-variance.md §8), never baselines of §8.2; no parameters, no ``PolicyContext``. The two samples: the
# re-gate follow-up dropped ``send_max``, the Codabench bundle's baseline solution, from the pilot (owner, 2026-09-29)
KIT_AGENTS = ("sample_random", "sample_heuristic")
# the leaderboard field's entries (``policies.field.FIELD``, the same names in its order): participant-like entries
# between naive and the best baseline, each a fixed spec, so no parameters; never baselines of §8.2
FIELD = (
    "mpc_det_safety",
    "mpc_det_h20",
    "mpc_det_h12",
    "blend_p95",
    "blend_p90",
    "blend_p85",
    "blend_p80",
    "blend_p70",
    "blend_p60",
)
NAMES = BASELINES + KIT_AGENTS + FIELD  # every name ``make_policy`` builds
PARAMETERLESS = ("zero", *KIT_AGENTS, *FIELD)  # the names without a parameter class (``params_class`` refuses them)
NAIVE_DERIVED = ("nd", "sz_state_base_stock", "human_ref", "greedy_lp")  # carry the end-aware cut (71)-(72) (Q100 (3))
LP_BASELINES = ("greedy_lp", "mpc_det", "mpc_scen", "hindsight_consensus")  # solved with highspy (Q95)
# labelled so in every report (§8.2 table); M5-O6's default (a): mpc_scen plays at the S its size's step budget allows
# (mpc_scen.S_BUDGET), so only hindsight_consensus is listed. On `full` S 2 is within the 4 s budget of Q115 at the
# scoring worker's load (design §12 M5 row "`mpc_scen`'s S by the step budget"), so M5-O6 (b)'s label is not needed
NOT_BUDGET_COMPLIANT = ("hindsight_consensus",)
# name -> (module, policy class, parameter class) of the ``cls(params, context)`` baselines; zero (no parameters) and
# naive (``NaiveParams``, the anchor's constructor) are built in ``make_policy`` itself
_ENTRIES: Mapping[str, tuple[str, str, str]] = {
    "nd": ("shockbench_flow.policies.nd", "NdPolicy", "NdParams"),
    "sz_state_base_stock": ("shockbench_flow.policies.sz_state_base_stock", "SzPolicy", "SzParams"),
    "human_ref": ("shockbench_flow.policies.human_ref", "HumanRefPolicy", "HumanRefParams"),
    "greedy_lp": ("shockbench_flow.policies.greedy_lp", "GreedyLP", "GreedyLPParams"),
    "mpc_det": ("shockbench_flow.policies.mpc_det", "MpcDet", "MpcDetParams"),
    "mpc_scen": ("shockbench_flow.policies.mpc_scen", "MpcScen", "MpcScenParams"),
    "hindsight_consensus": ("shockbench_flow.policies.hindsight_consensus", "HindsightConsensus", "HindsightParams"),
}
# public generator profiles by name: name -> function in disruption.profiles (design §2.4 "`tiny` generator profile";
# §12 M5 "`small` and `full` generator profile", a stub that refuses until the M5 stream "generator" fills it). The one
# map of profile names: the Hydra boundary (``sbf_boundary.PROFILES``), ``GeneratorRef`` and the scenario baselines'
# profile named after the instance kind (``scenarios.generator_params``) all read it (docs/099-m5-streams.md)
PROFILES: Mapping[str, str] = {"tiny": "tiny_profile", "small": "flagship_profile", "full": "flagship_profile"}


def profile_function(name: str):
    """The profile function ``(inst, gamma) -> GeneratorParams`` named ``name`` in ``PROFILES`` (imported lazily).

    Raises:
        ValueError: if ``name`` is not a key of ``PROFILES``.

    """
    if name not in PROFILES:
        raise ValueError(f"generator profile must be one of {tuple(PROFILES)}, got {name!r}")
    return getattr(importlib.import_module("shockbench_flow.disruption.profiles"), PROFILES[name])


@dataclass(frozen=True)
class GeneratorRef:
    """The public generator a baseline may use: the split's profile and rung, named in the policy config (Q100 (1)).

    Never read from omega or Static (Static carries theta, not the generator profile or rung); the ranked split's rung
    is public knowledge (§9.5). ``params(inst)`` builds the ``GeneratorParams`` (``disruption.profiles``).

    Raises:
        ValueError: if ``profile`` is not a key of ``PROFILES`` or ``gamma`` not a rung of ``naive_fq.RUNGS``.

    """

    profile: str
    gamma: float

    def __post_init__(self) -> None:
        from shockbench_flow.policies.naive_fq import RUNGS

        if self.profile not in PROFILES:
            raise ValueError(f"generator profile must be one of {tuple(PROFILES)}, got {self.profile!r}")
        if isinstance(self.gamma, bool) or self.gamma not in RUNGS:
            raise ValueError(f"generator gamma must be a rung of {RUNGS} (§2.6), got {self.gamma!r}")

    def params(self, inst: Instance):
        """The ``GeneratorParams`` of this profile at this rung for ``inst`` (a profile refuses another kind)."""
        return profile_function(self.profile)(inst, self.gamma)


_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class PathsFile:
    """``sz_state_base_stock``'s transient paths as a public data file its parent wrote (INT-M4-02).

    ``path`` names the ``.npz`` ``sz_state_base_stock.save_paths`` wrote (no pickles, its key inside) and ``sha256`` the
    SHA-256 hex digest of its bytes. A policy process that holds no paths for its key (the wire child, whose parent
    cannot seed its memory) reads them at reset instead of drawing the 1,000 stream-22 replications per pair behind
    F_Q, minutes of reset work that would count against week 1 (design §12 rows "`sz_state_base_stock` (M4 reading)"
    and "Subprocess transport (M3 wire)"). Two short strings: the context still pickles small.

    Raises:
        ValueError: if ``path`` is not a non-empty string or ``sha256`` not a SHA-256 hex digest.

    """

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"the paths file is named by a non-empty path string, got {self.path!r}")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64 and set(self.sha256) <= _HEX):
            raise ValueError(f"the paths file's digest must be a SHA-256 hex digest, got {self.sha256!r}")


@dataclass(frozen=True)
class PolicyContext:
    """Small public inputs every M4 baseline may take, computed once in the parent process (never per worker).

    Attributes:
        fq_quantile: naive's F_Q quantiles of (67) for the episode's generator (``naive_fq.generator_quantiles``),
            keyed (c, pool, j, k); None on an injected list (the point mass at 0). The LP baselines' internal naive
            fallback (``NaivePolicy(fq_quantile, end_aware=True)``) and ``sz_state_base_stock``'s unconditional
            fallback read them; a worker never recomputes them (1,000 stream-22 replications per pair).
        generator: the public generator (``GeneratorRef``) of ``mpc_scen``, ``hindsight_consensus`` and
            ``sz_state_base_stock``; None on an injected list, where the scenario baselines use the anchor rung of the
            instance's profile (design §12 M4 row "Scenario draws") and ``sz_state_base_stock`` equals naive.
        fq_replications: the replication count behind ``fq_quantile`` and ``sz_state_base_stock``'s transient paths
            (their cache key); the design's 1,000 unless a smoke run says otherwise.
        paths_file: where a process that holds no ``sz_state_base_stock`` paths reads the parent's (``PathsFile``);
            None: such a process draws them (``run_episode``, V24 replays, a generated V5 test's joblib workers). The
            wire entry point sets it for the child (INT-M4-02); every other baseline ignores it.

    Raises:
        ValueError: as ``naive.canonical_quantiles`` (TypeError for a non-real quantile) and
            ``naive_fq.check_replications``, or if ``paths_file`` is neither None nor a ``PathsFile``.

    """

    fq_quantile: Mapping[tuple, float] | None = None
    generator: GeneratorRef | None = None
    fq_replications: int = REPLICATIONS
    paths_file: PathsFile | None = None

    def __post_init__(self) -> None:
        if self.fq_quantile is not None:
            object.__setattr__(self, "fq_quantile", frozen_map(canonical_quantiles(self.fq_quantile)))
        check_replications(self.fq_replications)
        if self.paths_file is not None and not isinstance(self.paths_file, PathsFile):
            raise ValueError(f"paths_file must be a PathsFile or None, got {type(self.paths_file).__name__}")


def params_class(name: str) -> type:
    """The frozen parameter dataclass of baseline ``name`` (imports its module; the Hydra boundary's schema source).

    Naive's is ``naive.NaiveParams`` (its step-6 cut); zero and the kit agents have none (``PARAMETERLESS``).

    Raises:
        ValueError: on a name outside ``BASELINES`` or one without parameters (``PARAMETERLESS``).

    """
    if name == "naive":
        return NaiveParams
    if name not in _ENTRIES:
        raise ValueError(f"no parameter class for {name!r}: one of {('naive', *_ENTRIES)}")
    module, _cls, params = _ENTRIES[name]
    return getattr(importlib.import_module(module), params)


def make_policy(name: str, context: PolicyContext | None = None, params: object | None = None) -> Policy:
    """A fresh baseline object named ``name`` (never one reused across episodes; module docstring).

    The one place a baseline's parameters become a policy (SIM-M4-01): ``zero`` takes none; ``naive`` is the end-aware
    anchor (Q98) with ``context.fq_quantile`` and the cut of its ``NaiveParams``; every other baseline is
    ``cls(params, context)``. ``params`` None means the parameter class's defaults (for naive, the anchor's cut 0.6).
    A kit agent of ``KIT_AGENTS`` takes no parameters and ignores ``context`` (``kit_agents.KitAgentPolicy``); a field
    entry of ``FIELD`` takes no parameters and reads ``context`` as ``mpc_det`` does (``field.make_entry``).

    Raises:
        ValueError: on a name outside ``NAMES``, or parameters given to a name of ``PARAMETERLESS``.
        TypeError: if ``params`` is not an instance of the baseline's parameter class.

    """
    ctx = PolicyContext() if context is None else context
    if name not in NAMES:
        raise ValueError(f"unknown baseline {name!r}: one of {NAMES}")
    if name in PARAMETERLESS and params is not None:
        raise ValueError(f"{name} takes no parameters, got {params!r}")
    if name == "zero":
        return ZeroPolicy()
    if name in KIT_AGENTS:
        return importlib.import_module("shockbench_flow.policies.kit_agents").KitAgentPolicy(name)
    if name in FIELD:  # the field's own spec (``policies.field.FIELD``); the context as a baseline's
        return importlib.import_module("shockbench_flow.policies.field").make_entry(name, ctx)
    p_cls = params_class(name)  # imports the baseline's module (lazily, module docstring)
    if params is None:
        params = p_cls()
    elif not isinstance(params, p_cls):
        raise TypeError(f"{name} takes {p_cls.__name__}, got {type(params).__name__}")
    if name == "naive":
        return NaivePolicy(closed_threshold=params.closed_threshold, fq_quantile=ctx.fq_quantile)
    module, cls_name, _params = _ENTRIES[name]
    return getattr(importlib.import_module(module), cls_name)(params, ctx)

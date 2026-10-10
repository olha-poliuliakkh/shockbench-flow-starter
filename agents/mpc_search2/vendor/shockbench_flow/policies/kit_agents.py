"""The hackathon kit's own agents as registry policies: the M5 pilot's participant-like entries.

The leaderboard's split sizes come from the variance of entries like a participant's, which the §2.6 pairs understate
by about 20× on `tiny` (docs/099-split-variance.md, "What the M5 pilot must record", item 4), so the pilot plays the
kit's two samples on the same episodes as the baselines (owner, 2026-09-29):

- ``sample_random``: ``docs/hackathon/samples/random`` (uniform random flows, override quantities and release modes,
  from ``config["policy_seed"]``);
- ``sample_heuristic``: ``docs/hackathon/samples/heuristic`` (the cover heuristic, the Codabench starting kit's agent).

The M5 gate's pilots also played ``send_max`` (flows = capacity x ``action_mask``; from ``examples/send_max_agent.py``,
then, from the M5 re-gate, from the Codabench bundle's baseline solution under ``deploy/``). The re-gate follow-up
dropped it from the pilot (the owner, 2026-09-29, "Drop it from the pilot"): it stays solely the bundle's baseline
submission, so the package reads nothing under ``deploy/``, and the committed pilot summaries keep its runs.

Every kit agent passes the kit's own container check (``shockbench_flow_agent.submission.agent_warnings``, what
``sbf-validate`` prints) and imports with only the policy image's packages (tests/test_kit_agents.py).

``KitAgentPolicy`` plays the agent in process behind the policy container's shim (``shockbench_flow_agent.AgentShim``:
the flat layout, ``Agent(config)``, the Dict observation, the Dict action to the wire action, a failed week's null
action), so an episode is the one ``shockbench_flow_agent.evaluate`` plays in process, week for week. Its policy seed is
the trusted runner's for that agent's submission (``hosting.trusted``: (27) salted by the submission's SHA-256): the
policy declares ``seed_id``, the SHA-256 of the zip the starter kit builds from its source
(``hosting.submission.build_submission``, the sample's ``README.md`` left out as ``scripts/python/build_starter_kit.py``
leaves it; a single file is zipped as ``agent.py``), and the runner salts the seed with it
(``evaluation.runner.episode_policy_seed``). So on the same episode, root and split, its J equals ``evaluate``'s on
that zip in process, to the cent (tests/test_kit_agents.py). Since the seed and every trajectory of an agent that
draws follow from the files' bytes, the policy also declares ``provenance`` (its source, ``seed_id`` and the SHA-256
of each file of the zip), which the runner records in ``EvalResult.setup["policy_provenance"]``, so a summary names
the bytes its kit rows were played with (M5 re-gate DET-M5R-2).

A ``KitAgentPolicy`` pickles by name and checkout (``__reduce__``): the ``Agent`` class lives under a module name
only the importing process has (``sbf_kit_<name>``), so a subprocess child (``information.runner.child_argv``, the
M3 wire's subprocess transport) rebuilds the policy from the checkout's files instead of failing to unpickle it
(M5 re-gate INT-M5R-01). A policy is pickled between episodes: an unpickled one is the fresh policy, whose ``reset``
builds its agent, as every ``reset`` does.

The agents are the organisers' own code, played in the trusted process like a baseline; a participant's submission is
never played this way (the hidden split scores it through the policy container only, §9.4). They take no parameters and
read no ``PolicyContext`` (a participant sees Static and the observations alone). Their sources are repository files,
never package data: they resolve against the checkout that holds the working directory (``checkout_root``: the first
directory from the working directory up holding ``.project-root``, the anchor the scripts find with ``rootutils``), so
the entry points and the tests run from the checkout, and the joblib workers inherit it. ``shockbench_flow_agent`` is
imported inside the functions only, so the package's module-level imports stay the simulator's.
"""

import functools
import hashlib
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path


# registry name -> the agent's source relative to the checkout (a submission directory holding agent.py, or one file)
KIT_AGENTS: Mapping[str, str] = {
    "sample_random": "docs/hackathon/samples/random",
    "sample_heuristic": "docs/hackathon/samples/heuristic",
}
ROOT_MARKER = ".project-root"  # the repository's anchor file (AGENTS.md; the scripts' rootutils indicator)
README_NAME = "README.md"  # a sample's explanation beside its agent.py, left out of its zip (build_starter_kit.py)


def checkout_root(start: str | Path | None = None) -> Path:
    """The repository checkout: the first directory from ``start`` (the working directory) up holding ``.project-root``.

    Raises:
        FileNotFoundError: when no directory on the way holds it (the kit agents are repository files, not package
            data: run from the checkout).

    """
    here = Path.cwd() if start is None else Path(start)
    for d in (here.resolve(), *here.resolve().parents):
        if (d / ROOT_MARKER).is_file():
            return d
    raise FileNotFoundError(
        f"no {ROOT_MARKER} above {here}: the kit agents ({', '.join(KIT_AGENTS)}) are repository files, so run from "
        "the ShockBench-Flow checkout"
    )


def agent_source(name: str, root: str | Path | None = None) -> Path:
    """The agent's source in the checkout: a directory holding ``agent.py``, or the agent's one file.

    Raises:
        ValueError: on a name outside ``KIT_AGENTS``.
        FileNotFoundError: when the checkout (``checkout_root``) or the source is missing.

    """
    if name not in KIT_AGENTS:
        raise ValueError(f"unknown kit agent {name!r}: one of {tuple(KIT_AGENTS)}")
    path = checkout_root(root) / KIT_AGENTS[name]
    if not (path / "agent.py" if path.is_dir() else path).is_file():
        raise FileNotFoundError(f"kit agent {name!r}: no agent at {path}")
    return path


def build_zip(source: Path, zip_path: Path) -> Path:
    """The submission zip of ``source`` as the starter kit builds it (``hosting.submission.build_submission``).

    A directory is zipped without its ``README.md``; a single file is zipped alone as ``agent.py``, the zip a
    participant makes of it (``examples/make_submission.py``). Byte for byte repeatable: the bytes, and so the SHA-256
    that salts the policy seed (27), follow from the files alone.
    """
    from shockbench_flow.hosting.submission import AGENT_FILE, build_submission

    if source.is_dir():
        return build_submission(source, zip_path, exclude=(README_NAME,))
    folder = zip_path.parent / f"{zip_path.stem}_src"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, folder / AGENT_FILE)
    return build_submission(folder, zip_path)


@functools.cache
def provenance(name: str, root: str | Path | None = None) -> dict:
    """What the agent plays from: its source (relative to the checkout), ``seed_id`` and each zipped file's SHA-256.

    ``{"source": ..., "seed_id": <the zip's SHA-256>, "files": {<path in the zip>: <SHA-256 of its bytes>}}``, from the
    zip ``build_zip`` makes (a single file is the zip's ``agent.py``); one byte changed in a file changes both hashes.
    """
    with tempfile.TemporaryDirectory(prefix="sbf-kit-") as tmp:
        path = build_zip(agent_source(name, root), Path(tmp) / f"{name}.zip")
        with zipfile.ZipFile(path) as zf:
            files = {m: hashlib.sha256(zf.read(m)).hexdigest() for m in sorted(zf.namelist())}
        return {"source": KIT_AGENTS[name], "seed_id": hashlib.sha256(path.read_bytes()).hexdigest(), "files": files}


def submission_sha256(name: str, root: str | Path | None = None) -> str:
    """The SHA-256 of the agent's submission zip (``build_zip``): its id, the salt of its policy seed (27)."""
    return provenance(name, root)["seed_id"]


@functools.cache
def agent_class(name: str, root: str | Path | None = None) -> type:
    """The agent's ``Agent`` class, imported once per process and checkout under the module name ``sbf_kit_<name>``."""
    from shockbench_flow_agent.shim import load_agent_class

    return load_agent_class(agent_source(name, root), f"sbf_kit_{name}")


class KitAgentPolicy:
    """One kit agent as a ``Policy`` (``policies.base``): ``AgentShim`` over its ``Agent``, in process (module doc).

    ``name`` is the registry name, which names the run in every report; ``root`` the checkout its files are read from
    (``checkout_root`` of ``root``, the working directory's by default); ``seed_id`` is the submission SHA-256 the
    runner salts the policy seed with (``evaluation.runner.episode_policy_seed``) and ``provenance`` the files it came
    from (``provenance``). A week whose ``act`` raises or whose action is malformed returns None, the whole-week failure
    the environment answers with the D9 fallback (§9.3), as in the container; ``errors`` lists them. It pickles by name
    and checkout (module docstring).

    Raises:
        ValueError: on a name outside ``KIT_AGENTS``.
        FileNotFoundError: when the checkout or the agent's source is missing (``agent_source``).

    """

    def __init__(self, name: str, root: str | Path | None = None) -> None:
        from shockbench_flow_agent.shim import AgentShim

        self.name = name
        self.root = checkout_root(root)
        self.provenance = provenance(name, self.root)
        self.seed_id = self.provenance["seed_id"]
        self._shim = AgentShim(agent_class(name, self.root))

    def __reduce__(self) -> tuple:
        """Pickled as its name and checkout, rebuilt from the files where it is unpickled (INT-M5R-01)."""
        return (KitAgentPolicy, (self.name, str(self.root)))

    @property
    def errors(self) -> list[tuple[int, str]]:
        """(week, 'ExceptionType: message') of every week whose action became null (week 0 for ``Agent(config)``)."""
        return self._shim.errors

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self._shim.reset(static, obs, policy_seed)

    def act(self, obs: dict) -> dict | None:
        return self._shim.act(obs)

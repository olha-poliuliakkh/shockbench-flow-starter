"""The scoring container's shim: a participant's ``Agent`` behind the scorer's line protocol.

``serve_submission(agent_path)`` is the container's entry point (``python -m shockbench_flow_agent <dir>``): it
imports ``agent.py`` from the submission directory, does the kit's first-use work (``warm_up``), tells the container's
launcher it is ready when one runs it (so the import and the warm-up count toward the start-up, never toward a week),
then reads one JSON message per line from stdin, and for each

- Reset (the start of an episode): builds the flat layout from the episode's public tables (``FlatLayout.from_static``),
  the ``config`` of ``Agent(config)`` (``convert.agent_config``: the public tables, the information regime as
  published, T and the policy seed) and a fresh ``Agent(config)``, one per episode; no reply;
- Request (one week): converts the observation to the Dict observation (``convert.observation_dict``), calls
  ``agent.act``, converts the Dict action to the protocol's action (``convert.action_to_wire``) and writes the Reply.

Failure rules:

- An exception in ``act`` (``SystemExit`` included), a malformed Dict action, or a later week's observation the flat
  view refuses (a list over its cap) gives a Reply whose action is null for that week: the naive rule plays that week
  and it is counted. The loop never stops on it. A release mode outside the three voids its pair only (``convert``).
- An exception in ``Agent(config)`` leaves the episode without an agent: the naive rule plays every week of it.
- A Reset the flat view cannot carry is the kit's error, not the participant's, and never a silent naive episode:
  ``reset`` raises ``LayoutError`` naming the block, its cap and the instance kind, so an in-process rollout stops on
  it, and ``serve_submission`` writes the reason to stderr and ends with status ``LAYOUT_FAILED`` (4).
- ``agent.py`` that fails to import, or defines no ``Agent``, ends the process with status ``IMPORT_FAILED`` (2)
  before any line is read: the naive rule plays the episode.
- Tracebacks go to stderr (the first ``TRACEBACKS`` in full, then one line each); stdout carries the protocol alone:
  ``serve_stdio`` moves it to a private copy of fd 1 and points fd 1 and ``sys.stdout`` at stderr, and gives the agent
  an empty stdin, so a ``print`` or a read in participant code can neither corrupt nor steal a line.

``AgentShim`` has the policy shape (``name``, ``reset``, ``act``), so an in-process rollout drives the same
conversions; its trajectory equals the container's bit for bit. ``AgentShim(..., on_week=f)`` calls ``f(week,
seconds)`` with each week's CPU time in this process (week 1 includes ``Agent(config)`` and the reading of the
episode's tables, as the server's meter charges them); ``serve_submission`` writes those as JSON lines ``[week,
seconds]`` to the file ``cpu_log`` names, or ``CPU_LOG_VAR`` in the environment (``play_isolated`` sets it; the
scoring container never does).

Imports: the standard library, numpy, the protocol's codec and the flat view only; never the simulator, never the
scenario. The container also holds SciPy and CPU torch for participant code; the kit imports neither, and
``serve_submission`` pins torch's thread pools to one thread when ``agent.py`` imported it (``cap_torch_threads``).
"""

import copy
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from collections.abc import Callable
from pathlib import Path

import numpy as np

from shockbench_flow.information.flat import FlatLayout
from shockbench_flow.information.wire import encode, reply_message
from shockbench_flow.instance import load_instance
from shockbench_flow.instance.io import DATA_DIR, SCHEMA_FILE
from shockbench_flow.instance.schema import Instance
from shockbench_flow_agent.convert import action_to_wire, agent_config, observation_dict


AGENT_FILE = "agent.py"  # the submission's entry module
AGENT_CLASS = "Agent"
IMPORT_FAILED = 2  # exit status of a submission whose agent.py cannot be imported or defines no Agent
# exit status of a Reset the flat view cannot carry (LayoutError); 3 is the launcher's own (hosting/launch.py)
LAYOUT_FAILED = 4
CPU_LOG_VAR = "SBF_CPU_LOG"  # a file that receives each week's CPU seconds (module docstring); unset when scored
TRACEBACKS = 3  # full tracebacks written to stderr per process; later failures get one line each (not a model value)
PREFIX = "shockbench-agent"
READY_HOOK = "sbf_ready"  # the launcher's ready-signal module (hosting/launch.py READY_HOOK), when a launcher runs us
PRELOADED: dict[str, Instance] = {}  # Instance.hash -> the packaged public instances ``warm_up`` loaded


class LayoutError(RuntimeError):
    """The flat view cannot carry an episode's Reset: the kit's failure, never the participant's (module docstring)."""


class AgentShim:
    """One submission's ``Agent`` class behind the conversions of ``convert`` (module docstring).

    ``errors`` lists (week, 'ExceptionType: message') of every week whose action became null (week 0 for the reset),
    for local reports; stderr gets the tracebacks. ``on_week``, when given, is called after each week's ``act`` with
    (week, CPU seconds of this process): the reset's own CPU time is added to the first week that follows it.
    """

    name = "submission"

    def __init__(
        self,
        agent_class: Callable[[dict], object],
        *,
        maxima: dict[str, int] | None = None,
        on_week: Callable[[int, float], None] | None = None,
    ) -> None:
        self.agent_class, self.maxima, self.on_week = agent_class, maxima, on_week
        self.layout: FlatLayout | None = None
        self.agent: object | None = None
        self.episode: str | None = None
        self.nonce: str | None = None
        self.errors: list[tuple[int, str]] = []
        self._logged = 0
        self._reset_cpu = 0.0

    def _fail(self, week: int, what: str) -> None:
        exc = sys.exc_info()[1]
        text = f"{type(exc).__name__}: {exc}"[:300]
        self.errors.append((week, text))
        if self._logged < TRACEBACKS:
            print(f"{PREFIX}: week {week}: {what} failed; the naive rule plays this week:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"{PREFIX}: week {week}: {what} failed; the naive rule plays this week: {text}", file=sys.stderr)
        self._logged += 1

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        """Build the layout, the config and a fresh ``Agent(config)``; an ``Agent(config)`` failure leaves no agent.

        Raises:
            LayoutError: when the flat layout cannot be built from Static or the reset observation exceeds a cap (the
                kit's failure, module docstring).

        """
        if self.on_week is None:
            self._reset(static, obs, policy_seed)
            return
        start = time.process_time()
        try:
            self._reset(static, obs, policy_seed)
        finally:
            self._reset_cpu = time.process_time() - start

    def _reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        self.layout = self.agent = None
        try:
            inst = PRELOADED.get(static.get("instance_hash"))  # the same instance as Static.instance, by its hash
            layout = FlatLayout.from_static(static, self.maxima, inst=inst)
            observation = observation_dict(layout, obs)
        except Exception as exc:  # the kit's own failure: raised, never played as a silent naive episode
            raise LayoutError(f"the flat view cannot carry this episode's Reset: {type(exc).__name__}: {exc}") from exc
        self.layout = layout
        config = agent_config(copy.deepcopy(static), policy_seed, layout, observation)
        try:
            self.agent = self.agent_class(config)
        except (Exception, SystemExit):  # noqa: BLE001 - participant code
            self._fail(0, "Agent(config)")

    def act(self, obs: dict) -> dict | None:
        """The protocol's action of one observation, or None (the naive rule plays the week) when anything fails."""
        if self.on_week is None:
            return self._act(obs)
        start = time.process_time()
        try:
            return self._act(obs)
        finally:
            used, self._reset_cpu = time.process_time() - start + self._reset_cpu, 0.0
            self.on_week(obs["week"], used)

    def _act(self, obs: dict) -> dict | None:
        week = obs["week"]
        if self.agent is None:
            return None
        try:
            observation = observation_dict(self.layout, obs)
        except Exception:  # noqa: BLE001 - a list over its cap in a later week: that week's naive, logged as the kit's
            self._fail(week, "the flat view")
            return None
        try:
            action = self.agent.act(observation)
            return action_to_wire(self.layout, week, action)
        except (Exception, SystemExit):  # noqa: BLE001 - participant code and its action
            self._fail(week, "act")
            return None

    def on_line(self, line: bytes) -> bytes | None:
        """Answer one wire line: a Reset resets (no reply), a Request returns the Reply line; others are skipped."""
        try:
            msg = json.loads(line)
            kind = msg.get("type") if isinstance(msg, dict) else None
        except ValueError:
            kind = None
        if kind == "reset":
            self.episode, self.nonce = msg.get("episode"), msg.get("nonce")
            self.reset(msg["static"], msg["obs"], msg["policy_seed"])
            return None
        if kind == "step":
            action = self.act(msg["obs"])
            return encode(reply_message(episode=msg["episode"], nonce=msg["nonce"], action=action))
        print(f"{PREFIX}: a line that is neither a Reset nor a Request was skipped", file=sys.stderr)
        return None


def serve(shim: AgentShim, read_line: Callable[[], bytes | None], write_line: Callable[[bytes], None]) -> None:
    """``shim.on_line`` on every line read until EOF (``read_line`` None); each reply written with ``write_line``."""
    while (line := read_line()) is not None:
        reply = shim.on_line(line)
        if reply is not None:
            write_line(reply)


def _claim_stdio():
    """Private copies of fds 0 and 1 for the wire; fd 1 and ``sys.stdout`` then point at stderr, stdin at devnull."""
    wire_in = os.fdopen(os.dup(0), "rb", buffering=0)
    wire_out = os.fdopen(os.dup(1), "wb", buffering=0)
    sys.stdout.flush()
    os.dup2(2, 1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    sys.stdout = sys.stderr
    sys.stdin = open(os.devnull)  # noqa: SIM115 - the agent's stdin for the life of the process
    return wire_in, wire_out


def serve_stdio(shim: AgentShim, wire_in, wire_out) -> None:
    """``serve`` over the wire's private stdin and stdout copies (``_claim_stdio``)."""
    reader = _LineReader(wire_in)

    def write(line: bytes) -> None:
        view = memoryview(line)
        while view:
            view = view[wire_out.write(view) :]

    serve(shim, reader.readline, write)


class _LineReader:
    """``readline`` over an unbuffered binary file: one line with its newline, or None at EOF (a partial line too)."""

    def __init__(self, f) -> None:
        self.f, self.buf, self.scanned = f, bytearray(), 0  # bytes of buf already known to hold no newline

    def readline(self) -> bytes | None:
        while True:
            i = self.buf.find(b"\n", self.scanned)
            if i >= 0:
                line = bytes(self.buf[: i + 1])
                del self.buf[: i + 1]
                self.scanned = 0
                return line
            self.scanned = len(self.buf)
            chunk = self.f.read(1 << 16)
            if not chunk:
                return None
            self.buf += chunk


class _Loaded:
    """The submission ``load_agent_class`` loaded last: its directory and the host modules its names displaced."""

    root: Path | None = None
    displaced: dict[str, types.ModuleType] = {}


def _under(module: object, root: Path) -> bool:
    """Whether a module's file, or any directory of its package path, lies under ``root``."""
    attrs = getattr(module, "__dict__", None)
    if not isinstance(attrs, dict):  # read the dict, never getattr: a lazy module's __getattr__ would import
        return False
    places = [attrs.get("__file__")]
    try:
        places += list(attrs.get("__path__") or ())
    except TypeError:  # a package path that is not a list of directories (a lazy or synthetic module)
        pass
    for place in places:
        if not isinstance(place, str):
            continue
        try:
            if Path(place).resolve().is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


def _top_names(root: Path) -> set[str]:
    """The top-level module names a submission directory provides: its ``*.py`` files and its directories."""
    names = set()
    for child in root.iterdir():
        if child.suffix == ".py" and child.stem.isidentifier():
            names.add(child.stem)
        elif child.is_dir() and child.name.isidentifier():
            names.add(child.name)
    return names


def unload_agent() -> None:
    """Undo the last ``load_agent_class``: its modules leave ``sys.modules``, its directory ``sys.path``.

    Every module whose file lies under the submission's directory goes (those imported later, inside ``act``, too),
    and the host's modules its names displaced come back. ``load_agent_class`` calls it first, so one process loads
    one submission at a time; an agent built from the previous one keeps its objects, not its lazy imports.
    """
    root = _Loaded.root
    if root is None:
        return
    for name, module in list(sys.modules.items()):
        if module is not None and _under(module, root):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not _same_dir(p, root)]
    sys.modules.update(_Loaded.displaced)
    _Loaded.root, _Loaded.displaced = None, {}
    importlib.invalidate_caches()


def _same_dir(entry: str, root: Path) -> bool:
    try:
        return Path(entry or ".").resolve() == root
    except (OSError, ValueError):
        return False


def load_agent_class(path: str | Path, module_name: str = "agent") -> type:
    """Import ``agent.py`` (``path`` the file or its directory) and return its ``Agent``.

    The directory goes first on ``sys.path``, so the submission's own modules import, and each load stands alone
    (``unload_agent`` of the previous one first): two folders with their own ``utils.py`` get their own, and a module
    of the submission named like one the process has already imported (``helper``, ``tests``, ``configs``) is the
    submission's file; the host's comes back at ``unload_agent`` (or the next load). Runs participant code: the
    container does this, and local runs of one's own agent; the trusted runner never.

    Raises:
        Exception: whatever the import raises; AttributeError if the module defines no ``Agent``.

    """
    unload_agent()
    file = Path(path)
    file = file / AGENT_FILE if file.is_dir() else file
    root = file.resolve().parent
    displaced = {}
    names = _top_names(root) | {module_name}
    for name, module in list(sys.modules.items()):
        top = name.partition(".")[0]
        if top in names and module is not None and not _under(module, root):
            displaced[name] = sys.modules.pop(name)
    sys.path[:] = [p for p in sys.path if not _same_dir(p, root)]
    sys.path.insert(0, str(root))
    _Loaded.root, _Loaded.displaced = root, displaced
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"{file}: not an importable Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, AGENT_CLASS, None)
    if agent is None:
        raise AttributeError(f"{file} defines no {AGENT_CLASS}")
    return agent


def serve_submission(
    agent_path: str | Path, *, maxima: dict[str, int] | None = None, cpu_log: str | Path | None = None
) -> int:
    """The container's entry point: serve the submission in ``agent_path`` over stdin and stdout until EOF.

    ``agent_path`` is the submission directory (holding ``agent.py``) or ``agent.py`` itself; the process works in
    that directory, so relative paths in participant code find the weights. ``cpu_log`` names a file that receives
    each week's CPU seconds as a JSON line ``[week, seconds]`` (``AgentShim``'s ``on_week``); None writes none.
    Returns the exit status: 0 at EOF, ``IMPORT_FAILED`` when ``agent.py`` cannot be imported or defines no ``Agent``
    (nothing is read then), ``LAYOUT_FAILED`` when a Reset is one the flat view cannot carry (``LayoutError``, its
    reason on stderr; no line is read after it).
    """
    log = None if cpu_log is None else open(cpu_log, "a", buffering=1)  # noqa: SIM115 - open for the process's life
    try:
        return _serve_submission(agent_path, maxima, log)
    finally:
        if log is not None:
            log.close()


def _serve_submission(agent_path: str | Path, maxima: dict[str, int] | None, log) -> int:
    path = Path(agent_path).resolve()
    root = path if path.is_dir() else path.parent
    wire_in, wire_out = _claim_stdio()
    os.chdir(root)
    try:
        agent_class = load_agent_class(root / AGENT_FILE)
    except (Exception, SystemExit):  # noqa: BLE001 - participant code
        print(f"{PREFIX}: importing {AGENT_FILE} failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return IMPORT_FAILED
    cap_torch_threads()
    warm_up()
    signal_ready()
    on_week = None if log is None else (lambda week, seconds: log.write(json.dumps([int(week), seconds]) + "\n"))
    try:
        serve_stdio(AgentShim(agent_class, maxima=maxima, on_week=on_week), wire_in, wire_out)
    except LayoutError:
        print(
            f"{PREFIX}: the agent kit's flat view cannot carry this episode (a kit error, not your agent):",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return LAYOUT_FAILED
    return 0


def warm_up() -> None:
    """The kit's once-per-process work, done before the ready line so that no week is charged for it.

    Without it the first Reset compiles the instance schema's validator (fastjsonschema) and imports the modules of
    the instance loader's initial-state check, once per process, and the server's CPU meter, which starts once the
    Reset is sent, would charge that to week 1 (about 0.3 s). Loading every packaged public instance (``tiny``,
    ``small`` and ``full``) into ``PRELOADED``, drawing from a seeded numpy generator (an ``Agent(config)`` often seeds
    one) and encoding one Reply does all of it, and spares each Reset the build of its instance: the Reset names it by
    ``instance_hash``, so the layout is the one the Reset's instance gives. Nothing here reads the submission, a Reset
    or a scenario, and nothing imports SciPy or torch. Never raises: a failure writes one line to stderr and the first
    Reset pays that work as before.
    """
    try:
        np.random.default_rng(0).random(1)
        for path in sorted(DATA_DIR.glob("*.json")):
            if path.name != SCHEMA_FILE:
                inst = load_instance(path)
                PRELOADED[inst.hash] = inst
        json.loads(encode(reply_message(episode="warm-up", nonce="warm-up", action=None)))
    except Exception as err:  # noqa: BLE001 - a warm-up, never a failure of the container
        print(f"{PREFIX}: warm-up failed, the first Reset does its work: {type(err).__name__}: {err}", file=sys.stderr)


TORCH_THREADS = 1  # torch's intra- and inter-op pools in the agent's process: the container has one CPU


def cap_torch_threads() -> None:
    """Pin torch's thread pools to ``TORCH_THREADS`` when ``agent.py`` imported torch; never imports torch itself.

    The image's ``OMP_NUM_THREADS=1`` caps the intra-op pool; the inter-op pool reads no variable, so the container's
    entry sets both (a courtesy, as the variables: ``--cpus`` enforces). An agent that imports torch later keeps the
    variables' cap only. A pool already started keeps its size (torch refuses the change).
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    for get, put in (
        (torch.get_num_threads, torch.set_num_threads),
        (torch.get_num_interop_threads, torch.set_num_interop_threads),
    ):
        try:
            if get() != TORCH_THREADS:
                put(TORCH_THREADS)
        except RuntimeError:  # the inter-op pool has started (parallel work in the agent's import)
            pass


def signal_ready() -> None:
    """Call the launcher's ``ready()`` (``READY_HOOK``) when a launcher runs this process; else nothing."""
    ready = getattr(sys.modules.get(READY_HOOK), "ready", None)
    if callable(ready):
        ready()

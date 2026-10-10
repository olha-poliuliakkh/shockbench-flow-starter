"""Timed runs of a submission as the server runs it: in an isolated child process, or in a local scoring container.

- ``play_isolated(submission_dir, task, episode)``: the folder, checked and copied as the server checks and copies
  it, plays one episode over the scorer's protocol in a child ``python -I -S -B`` whose import path holds only the
  scoring image's packages as installed here (numpy, SciPy, torch, fastjsonschema and what torch needs:
  ``image_lib``) and the kit, under the server's per-week wall clock, start-up and reply limits (``LIMITS``). The
  child starts through the container's own entry point, so the import of ``agent.py`` counts toward the start-up as
  on the server, and the shim reports each week's CPU seconds (week 1 includes ``Agent(config)``, as the server's
  meter charges it). A package installed here but not in the image cannot hide a missing import: an ``agent.py`` that
  imports one fails to start as on the server (``imported`` False, the reason in ``stderr``). The CPU seconds are
  this machine's: a guide to the server's meter, not its count.
- ``play_container(submission_dir, image, task, episode)``: the same episode in the scoring image (``build_image``),
  one container with the server's flags (no network, a read-only file system, one CPU, 4 GB, 128 processes, an
  unprivileged user: ``LIMITS.container``), each week's CPU read by the server's meter from Docker's statistics, the
  server's week rule (over the task's CPU budget, the week is the naive rule's and ends at once) and the episode's
  kill timer (``LIMITS.episode_timeout_s``). On an ARM machine the image runs under x86_64 emulation, which is slower:
  a week near its budget here may be within it on the server; a week far over it will not be.

Neither reports a cost: the naive rule does not stand behind the agent here (a failed week plays an empty action), so
these runs time an agent and show its failures; ``EpisodeSet`` scores it. Both refuse to run while the scorer's
secrets are set, as the local evaluation does, and play public roots only.
"""

import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from shockbench_flow.hosting.limits import LIMITS, Limits
from shockbench_flow.hosting.split import DEV_ENTROPY
from shockbench_flow_agent.image import KIT_PACKAGES, LAUNCHER, SHIM_MODULE, image_distributions
from shockbench_flow_agent.shim import CPU_LOG_VAR


STDERR_KEEP = 1 << 20  # bytes of the agent's stderr returned, its end kept (not a model value)
DEFAULT_PYTHON = (sys.executable, "-I", "-S", "-B")  # isolated: no user site, no site-packages, no bytecode written


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def image_lib(dest: str | Path, *, packages: bool = True) -> tuple[Path, list[str]]:
    """``dest`` filled with symbolic links to the kit's packages and, with ``packages``, the image's distributions.

    The image's distributions (``image_distributions``) are linked as installed in this environment, each top-level
    module of theirs (a Linux wheel's ``<name>.libs`` beside it too). Returns (``dest``, the image's distributions not
    installed here): an ``agent.py`` that imports one of those fails in ``play_isolated`` where it would work on the
    server (torch without the starter's ``rl`` extra).
    """
    from importlib import metadata, util

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {_norm(d) for d in image_distributions()} if packages else set()
    installed = metadata.packages_distributions()  # top-level module -> distributions
    tops = {t for t, dists in installed.items() if t.isidentifier() and any(_norm(d) in wanted for d in dists)}
    found = set()
    for top in sorted(tops | set(KIT_PACKAGES)):
        try:
            spec = util.find_spec(top)
        except (ImportError, ValueError):
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            src = Path(next(iter(spec.submodule_search_locations)))
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            src = Path(spec.origin)
        else:
            continue
        if not (dest / src.name).exists():
            (dest / src.name).symlink_to(src)
        libs = src.parent / f"{top}.libs"
        if libs.is_dir() and not (dest / libs.name).exists():
            (dest / libs.name).symlink_to(libs)
        found |= {_norm(d) for d in installed.get(top, [])}
    return dest, sorted(wanted - found)


def _prepare(submission_dir: str | Path, work: Path, limits: Limits):
    """The server's check and clean copy of the folder (``hosting.codabench.prepare_submission``)."""
    from shockbench_flow.hosting.codabench import prepare_submission

    return prepare_submission(submission_dir, work / "submission", limits.submission)


def _episode(task: str, episode: int, entropy: int, sha256: str, policy_seed: int | None) -> tuple:
    """(instance, omega, marks, the policy seed) of dev (or training) episode ``episode`` of ``task``."""
    from shockbench_flow.hosting.tasks import scenario, split_label, task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.omega.seeds import policy_seed as seed_rule

    inst, _params = task_generator(task)
    omega = scenario(task, episode, entropy=entropy)
    seed = seed_rule(entropy, split_label(entropy), episode, sha256) if policy_seed is None else policy_seed
    return inst, omega, compute_marks(inst, omega), seed


def _tail(path: Path, keep: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - keep))
        return f.read().decode(errors="replace")


def play_isolated(
    submission_dir: str | Path,
    task: str = "tiny",
    episode: int = 0,
    *,
    python_argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    regime: str = "standard",
    entropy: int = DEV_ENTROPY,
    policy_seed: int | None = None,
    limits: Limits = LIMITS,
) -> dict:
    """Play one episode of the folder's agent in an isolated child process and time each week (module docstring).

    Args:
        submission_dir: the folder holding ``agent.py`` (checked by the server's rules first; its canonical zip's
            SHA-256 salts the policy seed, as on the server).
        task: a public task (``tiny``, ``small``, ``full``).
        episode: the episode index of ``entropy``'s scenarios (the dev episodes by default).
        python_argv: the child's interpreter and its options (default: this interpreter, ``-I -S -B``, with links to
            the image's packages as installed here). Another interpreter (``uv run --isolated --with ... python``)
            gets the kit's packages only on its import path and brings its own numpy, SciPy and torch.
        env: variables added to the child's environment (it gets ``PATH``, a private ``HOME``, one BLAS thread and
            the launcher's variables, nothing else of this process's).
        regime: the information regime (``standard``, the scored one).
        entropy: the scenarios' root: the public dev root by default, or a training root of your own.
        policy_seed: the agent's policy seed (default: the server's rule for this folder and episode).
        limits: the per-week, start-up and reply limits (the server's by default).

    Returns:
        ``{task, episode, weeks, cpu_s, cpu_budget_s, over_budget, substitutions, imported, ready_s, missing,
        stderr, submission_sha256, policy_seed}``: ``cpu_s`` is each week's CPU seconds in the child (index w - 1 is
        week w, None for a week the agent did not reach), ``over_budget`` the weeks over the task's budget (on the
        server the naive rule plays them), ``substitutions`` the ``[week, cause]`` the server would give to the naive
        rule for other causes ('action', 'timeout', 'killed', ...), ``imported`` whether ``agent.py`` imported and the
        child came up, ``ready_s`` its start-up in seconds, ``missing`` the image's distributions not installed here
        and ``stderr`` the end of the agent's standard error (its tracebacks and prints).

    Raises:
        SubmissionError: when the folder breaks the server's rules.
        ValueError: while a scorer secret is set, on an unknown task, or a root of the hidden split's size.

    """
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting.docker import THREAD_ENV, ReadyTransport
    from shockbench_flow.information.runner import play_wire_episode
    from shockbench_flow.information.wire import WireLimits
    from shockbench_flow_agent.local_eval import SPLIT, check_local_run

    check_local_run(SPLIT, entropy)
    budget = limits.cpu_budget_s[task]
    with tempfile.TemporaryDirectory(prefix="sbf-isolated-") as tmp:
        work = Path(tmp)
        checked = _prepare(submission_dir, work, limits)
        inst, omega, marks, seed = _episode(task, episode, entropy, checked.sha256, policy_seed)
        lib, missing = image_lib(work / "lib", packages=python_argv is None)
        home, timing, errlog = work / "home", work / "cpu.jsonl", work / "stderr.txt"
        home.mkdir()
        child_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "SBF_SHIM_MODULE": SHIM_MODULE,
            "SBF_KIT_PATH": str(lib),
            CPU_LOG_VAR: str(timing),
            **THREAD_ENV,
            **dict(env or {}),
        }
        argv = [*(python_argv or DEFAULT_PYTHON), "-u", str(LAUNCHER), str(checked.clean_dir)]
        with errlog.open("wb") as err:
            transport = ReadyTransport(
                argv,
                deadline_s=limits.deadline_s,
                startup_s=limits.startup_s,
                max_reply_bytes=limits.max_reply_bytes,
                env=child_env,
                stderr=err.fileno(),
            )
            wired = play_wire_episode(
                Env(fallback=None),
                inst,
                transport,
                limits=WireLimits(max_reply_bytes=limits.max_reply_bytes, bank_seconds=limits.bank_seconds),
                regime=regime,
                omega=omega,
                policy_seed=seed,
                marks=marks,
                policy_name="submission",
            )
        cpu: list[float | None] = [None] * inst.T
        for line in timing.read_text().splitlines() if timing.exists() else []:
            week, seconds = json.loads(line)
            if 1 <= week <= inst.T:
                cpu[week - 1] = float(seconds)
        return {
            "task": task,
            "episode": episode,
            "weeks": inst.T,
            "cpu_s": cpu,
            "cpu_budget_s": budget,
            "over_budget": [w for w, c in enumerate(cpu, start=1) if c is not None and c > budget],
            "substitutions": [[int(w), str(code)] for w, code in wired.substitutions],
            "imported": transport.ready_s is not None,
            "ready_s": transport.ready_s,
            "missing": missing,
            "stderr": _tail(errlog, STDERR_KEEP),
            "submission_sha256": checked.sha256,
            "policy_seed": seed,
        }


def docker_socket() -> str:
    """The Docker daemon's unix socket: ``DOCKER_HOST``'s when it is ``unix://...``, else /var/run/docker.sock."""
    from shockbench_flow.hosting.metering import DOCKER_SOCKET

    host = os.environ.get("DOCKER_HOST", "")
    return host[len("unix://") :] if host.startswith("unix://") else DOCKER_SOCKET


def play_container(
    submission_dir: str | Path,
    image: str,
    task: str = "small",
    episode: int = 0,
    *,
    regime: str = "standard",
    entropy: int = DEV_ENTROPY,
    policy_seed: int | None = None,
    limits: Limits = LIMITS,
    episode_timeout_s: float | None = None,
    socket_path: str | None = None,
    docker: Sequence[str] = ("docker",),
) -> dict:
    """Play one episode of the folder's agent in a local scoring container, metered as the server meters it.

    Args:
        submission_dir: the folder holding ``agent.py`` (checked and copied by the server's rules first).
        image: the policy image, a tag or an ID (``build_image`` builds it from this wheel).
        task: a public task; `small` and `full` carry the server's kill timer, `tiny` (not hosted) the runner's
            bound of 2 start-ups plus 15 s a week.
        episode: the episode index of ``entropy``'s scenarios (the dev episodes by default).
        regime: the information regime (``standard``, the scored one).
        entropy: the scenarios' root: the public dev root by default, or a training root of your own.
        policy_seed: the agent's policy seed (default: the server's rule for this folder and episode).
        limits: the server's limits (the week's wall clock, start-up, reply size, CPU budget, container caps).
        episode_timeout_s: the kill timer (default: the task's, ``limits.episode_timeout_s``).
        socket_path: the Docker daemon's socket the meter reads (default: ``docker_socket()``).
        docker: the ``docker`` command.

    Returns:
        ``{task, episode, image, weeks, cpu_s, cpu_budget_s, over_budget, substitutions, ready_s, wall_s,
        episode_timeout_s, watchdog_fired, container_gone, meter_errors, stderr, submission_sha256, policy_seed}``:
        ``cpu_s`` each week's CPU seconds as the meter read them (None where a read failed), ``over_budget`` the
        weeks over the task's budget (the naive rule's on the server; ``substitutions`` holds them as 'cpu' beside
        the other causes).

    Raises:
        SubmissionError: when the folder breaks the server's rules.
        ValueError: while a scorer secret is set, on an unknown task, or a root of the hidden split's size.

    """
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting import metering
    from shockbench_flow.hosting.codabench import Wire
    from shockbench_flow.hosting.docker import DockerTransport
    from shockbench_flow.information.runner import play_wire_episode
    from shockbench_flow.information.wire import WireLimits
    from shockbench_flow_agent.local_eval import SPLIT, check_local_run

    check_local_run(SPLIT, entropy)
    budget = limits.cpu_budget_s[task]
    socket = socket_path or docker_socket()
    with tempfile.TemporaryDirectory(prefix="sbf-container-") as tmp:
        work = Path(tmp)
        checked = _prepare(submission_dir, work, limits)
        inst, omega, marks, seed = _episode(task, episode, entropy, checked.sha256, policy_seed)
        timer = episode_timeout_s if episode_timeout_s is not None else limits.episode_timeout_s.get(task)
        if timer is None:
            timer = Wire(limits.deadline_s, limits.startup_s, limits.max_reply_bytes, budget).watchdog_s(inst.T)
        cid, errlog = work / "cid", work / "stderr.txt"

        def resolve():
            return metering.cpu_reader("docker_stats", metering.read_cidfile(cid), socket_path=socket)

        meter = metering.WeekCpu(resolve, budget_s=budget, poll_s=metering.POLL_S["docker_stats"])
        start = time.perf_counter()
        with errlog.open("wb") as err:
            transport = DockerTransport(
                image,
                checked.clean_dir,
                deadline_s=limits.deadline_s,
                startup_s=limits.startup_s,
                max_reply_bytes=limits.max_reply_bytes,
                limits=limits.container,
                docker=tuple(docker),
                stderr=err.fileno(),
                episode_timeout_s=float(timer),
                cidfile=cid,
            )
            transport.interrupt, transport.interrupt_poll_s = meter.interrupt, meter.poll_s
            wired = play_wire_episode(
                Env(fallback=None),
                inst,
                transport,
                limits=WireLimits(max_reply_bytes=limits.max_reply_bytes, bank_seconds=limits.bank_seconds),
                regime=regime,
                omega=omega,
                policy_seed=seed,
                marks=marks,
                policy_name="submission",
                meter=meter,
            )
        return {
            "task": task,
            "episode": episode,
            "image": image,
            "weeks": inst.T,
            "cpu_s": list(meter.weeks),
            "cpu_budget_s": budget,
            "over_budget": list(meter.over),
            "substitutions": [[int(w), str(code)] for w, code in wired.substitutions],
            "ready_s": transport.ready_s,
            "wall_s": time.perf_counter() - start,
            "episode_timeout_s": float(timer),
            "watchdog_fired": transport.watchdog_fired,
            "container_gone": transport.container_gone,
            "meter_errors": list(meter.errors),
            "stderr": _tail(errlog, STDERR_KEEP),
            "submission_sha256": checked.sha256,
            "policy_seed": seed,
        }


__all__ = ["DEFAULT_PYTHON", "docker_socket", "image_lib", "play_container", "play_isolated"]

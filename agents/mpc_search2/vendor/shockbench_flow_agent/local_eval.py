"""Local evaluation of an agent: its cost J, its score against the naive rule and the clairvoyant plan, fallbacks.

``evaluate`` is the participant's entry: an ``Agent`` class, a factory of agents, a submission folder, a zip or an
``agent.py`` file, on a public task's dev episodes; a folder or a zip is checked and extracted by the server's own
validator first (``hosting.submission``), so a zip the server would refuse is refused here. ``report`` prints its
summary in plain words. ``score_agent`` is the scoring core both use (the organisers' evaluation script calls it too,
so the two give the same summary to the cent).

The public dev split only: no mode isolates the agent (a child runs with this process's user and can read its
environment and files; in process it shares the process), so the scorer's secrets never meet it. Every entry plays
``SPLIT`` on a public root, below the hidden split's size (a hidden root would play the hidden scenarios here), and
refuses to start while ``SBF_ENTROPY`` or ``SBF_SCORES_KEY`` is set in the environment or in the ``.env`` of the
working directory or a parent, which the agent could read (``check_local_run``); the hidden split is scored only in
the scoring container.

How an episode n is scored (``score_agent``): its scenario is episode n of the public generator on the dev root; the
policy seed is salted by the submission's SHA-256; the agent plays either through the container's entry point over a
real subprocess (``mode='subprocess'``: ``python -m shockbench_flow_agent <dir>``, with the server's per-week wall
clock, start-up and reply limits, ``LIMITS``) or in this process (``mode='in_process'``: ``AgentShim`` under a
rollout); both give the same trajectory bit for bit when the agent meets the deadline (``in_process`` enforces none).
The subprocess child gets a scrubbed environment (``PATH``, a private ``HOME``, the kit on ``PYTHONPATH`` and one BLAS
thread, as the container), never the caller's variables; in process the agent is imported and played with the
secret-like variables (API keys, tokens) out of ``os.environ``. Then J^naive (the naive rule, which plays without
predictions whatever the agent plays, with a demand model of ``fq_replications`` replications cached on disk), the
clairvoyant plan's cost on the scenario and on its event-free twin, the harm level on cut points from ``cut_draws``
draws (disk-cached; 0 draws: no harm levels), the fallbacks (whole weeks the naive rule played, by cause: 'action' for
an exception or a malformed action, or the protocol's 'timeout', 'unparsable', 'tags', 'too_long', 'killed') and the
invalid entries dropped. Scores (``scores``): RSS over all episodes, per harm level and pooled with the leaderboard's
weights, by the scorer's own rule: an episode whose clairvoyant plan, on the scenario or its event-free twin, is not
solved to optimality is excluded and counted by cause, and every other one counts whatever the sign of its headroom
J^naive - J^oracle. The per-episode RSS is a display only, null where the episode is excluded or its headroom is not
positive. The summary names the root by its commitment, as the scorer's documents do.
"""

import contextlib
import functools
import hashlib
import math
import os
import platform
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import shockbench_flow_agent
from shockbench_flow.disruption.harm import episode_harm
from shockbench_flow.disruption.params import GeneratorParams, generator_id
from shockbench_flow.disruption.sampler import sample_omega
from shockbench_flow.disruption.strata import CutPoints, stratum
from shockbench_flow.dynamics.env import Env, rollout, took_fallback
from shockbench_flow.evaluation.cache import cut_points_cached, default_cache_dir, fq_quantiles
from shockbench_flow.evaluation.results import EXCLUSION_CAUSES, episode_rss, exclusion_cause, oracle_optimal
from shockbench_flow.hosting import docker as hdocker
from shockbench_flow.hosting.docker import THREAD_ENV, check_unisolated, without_secret_like
from shockbench_flow.hosting.limits import LIMITS
from shockbench_flow.hosting.split import CUT_ENTROPY, DEV_ENTROPY, check_public_root, entropy_commitment, fill_strata
from shockbench_flow.hosting.submission import (
    AGENT_FILE,
    Submission,
    SubmissionError,
    agent_warnings,
    build_submission,
    extract_submission,
)
from shockbench_flow.hosting.tasks import (
    CUT_DRAWS,
    DEV_PER_STRATUM,
    DEV_SPLIT,
    MAX_CANDIDATES,
    get_task,
    task_generator,
)
from shockbench_flow.information.runner import SubprocessTransport, play_wire_episode
from shockbench_flow.information.theta import REGIME_NAMES
from shockbench_flow.information.wire import WireLimits
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import compute_marks
from shockbench_flow.omega.injected import event_free
from shockbench_flow.omega.seeds import policy_seed
from shockbench_flow.oracle.lp import ORACLE_METHOD, build_lp, solve_oracle
from shockbench_flow.policies.naive_fq import REPLICATIONS, anchor_policy, fallback_spec
from shockbench_flow.scoring.rss import STRATUM_WEIGHTS
from shockbench_flow_agent.shim import AgentShim, load_agent_class, unload_agent


MODES = ("subprocess", "in_process")
ANCHOR_REGIME = "prediction_free"  # the naive anchor's regime whatever the agent plays
LIMITS_TAG = "the server's limits: shockbench_flow_agent.LIMITS"  # what the summary's wire field follows
WIRE_KEYS = ("max_reply_bytes", "bank_seconds", "deadline_s", "startup_s")
SPLIT = DEV_SPLIT  # the one split played here (module docstring): a public root, never E_split
CHILD_PASS = ("SYSTEMROOT", "TMPDIR", "LANG")  # variables the interpreter itself may need; nothing secret
DEV_EPISODES = "dev"  # ``evaluate``'s default: the dev split, N_s = ``tasks.DEV_PER_STRATUM`` per harm stratum
# the SHA-256 that salts the policy seed of an agent given as a class or factory: it has no zip (the empty file's)
NO_ZIP_SHA256 = hashlib.sha256(b"").hexdigest()
# what each fallback cause means (the protocol's codes, and 'action' from the shim or in process), for the organisers
CAUSES = {
    "action": "act (or Agent(config)) raised, or returned a malformed action; the full traceback is in the agent's "
    "stderr above (lines starting 'shockbench-agent:')",
    "timeout": "no reply within submission.deadline_s seconds",
    "killed": "the agent's process is gone: agent.py failed to import (its stderr says why), or the process exited",
    "unparsable": "the reply line was not JSON (something wrote to the wire)",
    "too_long": "the reply line was over submission.max_reply_bytes",
    "tags": "the reply named another episode or nonce",
    "cpu": "the week used more CPU than its budget (the server's meter; local runs do not meter)",
}

AgentFactory = Callable[[dict], object]  # ``Agent(config)``: a class, or any callable of the config returning an agent


@dataclass(frozen=True)
class SubmissionRunConfig:
    """How a submission runs: its zip, the mode, how many episodes, the public roots and the protocol's limits.

    ``evaluate`` builds one, its ``path`` the zip (or, for an agent given as a class or factory, the agent's name); the
    limits default to the server's (``LIMITS``). ``bank_seconds`` is sent to the agent's process and never metered.
    """

    path: str | None = None
    mode: str = "subprocess"
    episodes: int = 8
    episode_list: list[int] | None = None  # exact episode indices n to play instead of eval.episode + range(episodes)
    dev_entropy: int = DEV_ENTROPY
    cut_draws: int = CUT_DRAWS
    cut_entropy: int = CUT_ENTROPY
    max_reply_bytes: int = LIMITS.max_reply_bytes
    bank_seconds: float = 60.0
    deadline_s: float = LIMITS.deadline_s
    startup_s: float = LIMITS.startup_s

    def __post_init__(self) -> None:
        """Refuse a missing path, an unknown mode and counts or budgets outside their domains.

        Raises:
            ValueError: on any of them.

        """
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("submission.path: the submission zip is required (submission.path=<file>.zip)")
        if self.mode not in MODES:
            raise ValueError(f"submission.mode must be one of {MODES}, got {self.mode!r}")
        if self.episode_list is not None:
            ns = list(self.episode_list)
            if not ns or len(set(ns)) != len(ns):
                raise ValueError(f"submission.episode_list must be distinct episode indices, got {ns!r}")
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in ns):
                raise ValueError(f"submission.episode_list must hold integers >= 0, got {ns!r}")
        for name in ("episodes", "max_reply_bytes"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"submission.{name} must be an integer >= 1, got {v!r}")
        for name in ("dev_entropy", "cut_draws", "cut_entropy"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"submission.{name} must be an integer >= 0, got {v!r}")
        for name in ("bank_seconds", "deadline_s", "startup_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (math.isfinite(v) and v > 0):
                raise ValueError(f"submission.{name} must be a positive number of seconds ({LIMITS_TAG}), got {v!r}")


def check_local_run(split: str = SPLIT, entropy: int = DEV_ENTROPY) -> None:
    """Refuse what would bring the scorer's secrets near participant code run here (module docstring).

    Raises:
        ValueError: on a split other than dev, a root of the hidden split's size (``hosting.split.check_public_root``),
            or while ``SBF_ENTROPY`` or ``SBF_SCORES_KEY`` is set in the environment
            (``hosting.docker.check_unisolated``) or in the ``.env`` beside the run (``hosting.docker.check_dotenv``).

    """
    if split != SPLIT:
        raise ValueError(
            f"split {split!r}: the local evaluation plays the public dev split only, since the agent runs here "
            "without isolation; the hidden split is scored only in the scoring container"
        )
    check_public_root(entropy)
    check_unisolated()
    hdocker.check_dotenv()


def _substitutions(traj, wire_subs: tuple | None) -> list[list]:
    """[week, cause] of every week the naive fallback played: the protocol's own log, or the trajectory's in process."""
    if wire_subs is not None:
        return [list(s) for s in wire_subs]
    return [[w, "action"] for w, r in enumerate(traj.records, start=1) if took_fallback(r)]


STDERR_KEEP = 1 << 16  # bytes of each episode's agent stderr kept to find its first error (not a model value)
_HEADER = re.compile(
    r"^shockbench-agent: (?:week (\d+): (.+?) failed; the naive rule plays this week:|importing (\S+) failed:)(.*)$"
)


def first_error(text: str) -> str | None:
    """The first failure the shim wrote to stderr ('week 1: KeyError: 2', 'importing agent.py: ...'), or None.

    The shim writes a header line (``shim._fail``, ``serve_submission``) followed by the traceback, whose last
    unindented line is the exception, or, after its first tracebacks, the exception on the header line itself.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if not m:
            continue
        where = f"week {m[1]}" if m[1] is not None else f"importing {m[3]}"
        if m[4].strip():
            return f"{where}: {m[4].strip()}"
        block = []
        for nxt in lines[i + 1 :]:
            if nxt.startswith("shockbench-agent:"):
                break
            block.append(nxt)
        last = [b for b in block if b.strip() and not b[0].isspace() and not b.startswith("Traceback")]
        return f"{where}: {last[-1].strip()}" if last else where
    return None


class StderrTee:
    """A write end for the agent's stderr: passed through to ours, its first ``STDERR_KEEP`` bytes kept."""

    def __init__(self) -> None:
        read, self.fd = os.pipe()
        self.kept = bytearray()
        self._thread = threading.Thread(target=self._pump, args=(read,), daemon=True)
        self._thread.start()

    def _pump(self, read: int) -> None:
        while data := os.read(read, 1 << 16):
            os.write(2, data)
            self.kept += data[: max(0, STDERR_KEEP - len(self.kept))]
        os.close(read)

    def close(self) -> str | None:
        """Close our copy of the write end, wait for the pump, and return the first error it saw."""
        os.close(self.fd)
        self._thread.join(timeout=30)
        return first_error(self.kept.decode(errors="replace"))


def kit_path() -> Path:
    """The directory that holds the kit's packages (``src/`` of a checkout, ``site-packages`` of an install)."""
    return Path(shockbench_flow_agent.__file__).resolve().parent.parent


def child_env(home: Path) -> dict[str, str]:
    """The agent process's environment (module docstring): nothing of the caller's but ``PATH`` and ``CHILD_PASS``."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "PYTHONPATH": str(kit_path())}
    env |= {k: os.environ[k] for k in CHILD_PASS if k in os.environ}
    return env | THREAD_ENV


def play(
    agent: Submission | AgentFactory,
    sub: SubmissionRunConfig,
    inst,
    omega,
    marks,
    regime: str,
    fallback,
    pseed: int,
    *,
    home: Path,
):
    """(trajectory, substitutions, seconds, first error or None) of the agent on one episode, by ``sub.mode``.

    ``agent`` is an extracted submission (its ``root`` holds ``agent.py``) or, in process only, an ``Agent`` class or
    factory. The first error is 'week N: Type: message' (week 0 for ``Agent(config)``): in process from the shim's own
    log, over the subprocess from the agent's stderr (``StderrTee``, which still passes it through to the terminal).
    """
    start = time.perf_counter()
    if sub.mode == "in_process":
        with without_secret_like():  # the agent's import and play see no API key or token (module docstring)
            if isinstance(agent, Submission):
                agent = load_agent_class(agent.root, f"submission_{agent.root.name}")
            shim = AgentShim(agent)
            traj = rollout(inst, shim, omega, regime, pseed, marks=marks, fallback=fallback)
        first = f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None
        return traj, _substitutions(traj, None), time.perf_counter() - start, first
    home.mkdir(parents=True, exist_ok=True)
    tee = StderrTee()
    try:
        transport = SubprocessTransport(
            [sys.executable, "-m", "shockbench_flow_agent", str(agent.root)],
            deadline_s=float(sub.deadline_s),
            startup_s=float(sub.startup_s),
            max_reply_bytes=sub.max_reply_bytes,
            env=child_env(home),
            stderr=tee.fd,
        )
        wired = play_wire_episode(
            Env(fallback=fallback),
            inst,
            transport,
            limits=WireLimits(max_reply_bytes=sub.max_reply_bytes, bank_seconds=float(sub.bank_seconds)),
            regime=regime,
            omega=omega,
            policy_seed=pseed,
            marks=marks,
            policy_name=AgentShim.name,
        )
    finally:
        first = tee.close()
    subs = _substitutions(wired.trajectory, wired.substitutions)
    return wired.trajectory, subs, time.perf_counter() - start, first


NO_STRATA = "no strata (submission.cut_draws=0)"


def scores(rows: list[dict], *, stratified: bool) -> dict:
    """The summary's RSS fields from the episode rows, by the scorer's one rule (``results.episode_rss``).

    Each row carries ``stratum``, ``excluded`` and the three integer-cent costs, as a scores.json episode does. RSS
    over all episodes is one harm level of weight 1; without harm levels (``stratified`` False) no pooled score is
    claimed.
    """
    one = episode_rss([{**r, "stratum": 1} for r in rows], (1.0,))
    out = {
        "excluded_from_rss": [r["episode"] for r in rows if r["excluded"] is not None],
        "exclusions": {cause: sum(r["excluded"] == cause for r in rows) for cause in EXCLUSION_CAUSES},
        "rss_all": one["strata"]["1"]["rss"],
        "rss_by_stratum": {str(s): None for s in range(1, len(STRATUM_WEIGHTS) + 1)},
        "rss_pooled": None,
        "rss_pooled_null_reason": NO_STRATA,
    }
    if stratified:
        table = episode_rss(rows)
        out["rss_by_stratum"] = {s: v["rss"] for s, v in table["strata"].items()}
        out["rss_pooled"], out["rss_pooled_null_reason"] = table["pooled"], table["pooled_reason"]
    return out


def check_submission(path: str | Path, workdir: Path) -> Submission:
    """Check and extract a submission zip under ``workdir/sub`` as the server does; print ``agent_warnings``' notes.

    Raises:
        SubmissionError: when the validator refuses the zip.

    """
    checked = extract_submission(path, workdir / "sub")
    for warning in agent_warnings((checked.root / AGENT_FILE).read_bytes(), [f for f, _n in checked.files]):
        print(f"WARNING: {warning}", file=sys.stderr, flush=True)
    return checked


def score_agent(
    agent: Submission | AgentFactory,
    inst: Instance,
    params: GeneratorParams,
    sub: SubmissionRunConfig,
    *,
    episodes: Sequence[int],
    regime: str,
    entropy: int,
    entropy_source: str,
    gamma: float,
    fq_replications: int,
    oracle_method: str = ORACLE_METHOD,
    n_jobs: int = 1,
    cache_dir: str | Path | None = None,
    command: str,
    workdir: Path,
    cuts: CutPoints | None = None,
    verbose: bool = True,
) -> dict:
    """The JSON summary of the agent on dev ``episodes`` of (``inst``, ``params``, ``entropy``); see the module.

    ``agent`` is a submission checked and extracted by ``check_submission`` (subprocess or in process) or an ``Agent``
    class or factory (in process only; its policy seed is salted by ``NO_ZIP_SHA256``). ``entropy`` is a public dev
    root (``submission.dev_entropy``). ``sub`` gives the mode, the cut points' draws and root, and the protocol's
    limits (its path and episode fields are the caller's); ``cache_dir`` None computes the naive rule's demand model
    and the cut points in this process; ``cuts`` passes cut points already computed for (``sub.cut_draws``,
    ``sub.cut_entropy``); ``verbose`` prints a progress line per episode.

    Raises:
        ValueError: on a root of the hidden split's size, or while ``SBF_ENTROPY`` or ``SBF_SCORES_KEY`` is set
            (``check_local_run``), before anything runs, or on an agent given as a class or factory in subprocess mode.

    """
    check_local_run(SPLIT, entropy)
    is_zip = isinstance(agent, Submission)
    if sub.mode == "subprocess" and not is_zip:
        raise ValueError("an Agent class or factory runs in process only (mode='in_process'); pass a folder or a zip")
    sha = agent.sha256 if is_zip else NO_ZIP_SHA256
    say = functools.partial(print, flush=True) if verbose else (lambda *args, **kwargs: None)
    say(
        f"the naive rule's demand model ({fq_replications} replications) and the harm cut points ({sub.cut_draws} "
        f"draws): read from {cache_dir or 'no cache'}, or computed on a first run (a few minutes)"
    )
    fq = fq_quantiles(inst, params, fq_replications, n_jobs=n_jobs, cache_dir=cache_dir)
    anchor = anchor_policy(inst, params, fq_replications, n_jobs)
    fallback = fallback_spec(inst, params, fq_replications, n_jobs)
    if not sub.cut_draws:
        cuts = None
    elif cuts is None:
        cuts = cut_points_cached(inst, params, sub.cut_draws, sub.cut_entropy, n_jobs, cache_dir)
    rows = []
    for n in episodes:
        omega = sample_omega(inst, params, entropy, n, SPLIT)
        marks = compute_marks(inst, omega)
        pseed = policy_seed(entropy, SPLIT, n, sha)
        traj, subs, seconds, first_err = play(
            agent, sub, inst, omega, marks, regime, fallback, pseed, home=workdir / "home"
        )
        naive = rollout(inst, anchor, omega, ANCHOR_REGIME, pseed, marks=marks, fallback=fallback)
        oracle = solve_oracle(build_lp(inst, marks), method=oracle_method)
        oracle0 = solve_oracle(build_lp(inst, compute_marks(inst, event_free(omega, inst))), method=oracle_method)
        excluded = exclusion_cause(oracle, oracle0)  # the scorer's rule
        headroom = naive.J_cents - oracle.J_cents if oracle_optimal(oracle) else None
        harm = episode_harm(inst, omega)
        rows.append(
            {
                "episode": n,
                "stratum": None if cuts is None else stratum(harm, cuts, generator_id=omega.generator_id),
                "harm_usd": harm,
                "J_policy_cents": traj.J_cents,
                "J_naive_cents": naive.J_cents,
                "J_oracle_cents": oracle.J_cents if oracle_optimal(oracle) else None,
                "oracle_status": oracle.status,
                "oracle_solver": oracle.solver,
                "J_oracle0_cents": oracle0.J_cents if oracle_optimal(oracle0) else None,
                "oracle0_status": oracle0.status,
                "oracle0_solver": oracle0.solver,
                "excluded": excluded,
                "rss": (naive.J_cents - traj.J_cents) / headroom if excluded is None and headroom > 0 else None,
                "d9_substitutions": subs,
                "first_error": first_err,
                "invalid_entries": sum(len(r.invalid) for r in traj.records if not took_fallback(r)),
                "policy_seconds": seconds,
                "trajectory_sha256": traj.sha256(),
                "omega_hash": omega.hash,
                "fallback_weeks": sum(took_fallback(r) for r in traj.records),  # every week naive played
            }
        )
        say(
            f"episode {n}: cost {traj.J_cents} cents, naive rule {naive.J_cents}, clairvoyant plan {oracle.J_cents}, "
            f"score {rows[-1]['rss']}, weeks the naive rule played {len(subs)}"
        )
    return {
        "command": command,
        "submission_sha256": agent.sha256 if is_zip else None,
        "submission_files": [list(f) for f in agent.files] if is_zip else [],
        "mode": sub.mode,
        "instance": inst.instance_id,
        "generator_id": generator_id(params, inst),
        "gamma": gamma,
        "fq_replications": fq_replications,
        "fq_cache": fq.cache,
        "regime": regime,
        "split": SPLIT,
        "entropy_source": entropy_source,
        "entropy_commitment": entropy_commitment(SPLIT, entropy),  # as scores.json names its root
        "cut_points": None if cuts is None else {"draws": cuts.draws, "entropy": sub.cut_entropy, "h": cuts.values},
        "wire": {k: v for k, v in asdict(sub).items() if k in WIRE_KEYS},
        "wire_tag": LIMITS_TAG,
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "episodes": rows,
        **scores(rows, stratified=cuts is not None),
        "J_policy_cents_total": sum(r["J_policy_cents"] for r in rows),
        "d9_weeks_total": sum(len(r["d9_substitutions"]) for r in rows),
        "weeks_total": sum(inst.T for _ in rows),
    }


# ----- the terminal report ------------------------------------------------------------------------------------------
def _runs(weeks: list[int]) -> str:
    """'weeks 1-3, 7' of a list of week numbers ('week 5' for one)."""
    ws, parts, i = sorted(weeks), [], 0
    while i < len(ws):
        j = i
        while j + 1 < len(ws) and ws[j + 1] == ws[j] + 1:
            j += 1
        parts.append(str(ws[i]) if i == j else f"{ws[i]}-{ws[j]}")
        i = j + 1
    return ("week " if len(ws) == 1 else "weeks ") + ", ".join(parts)


def warning_lines(out: dict) -> list[str]:
    """The report's warnings: weeks naive played for the agent (by cause, per episode) and entries dropped."""
    lines = []
    d9, weeks = out["d9_weeks_total"], out["weeks_total"]
    if d9:
        causes: dict[str, int] = {}
        for r in out["episodes"]:
            for _, cause in r["d9_substitutions"]:
                causes[cause] = causes.get(cause, 0) + 1
        counts = ", ".join(f"{c} {k}" for c, k in causes.items())
        lines.append(f"WARNING: naive played {d9} of {weeks} weeks for your agent (the fallback), by cause: {counts}")
        lines += [f"  {c}: {CAUSES.get(c, 'a cause of the protocol')}" for c in causes]
        for r in out["episodes"]:
            by: dict[str, list[int]] = {}
            for w, cause in r["d9_substitutions"]:
                by.setdefault(cause, []).append(w)
            if by:
                what = "; ".join(f"{c} {_runs(ws)}" for c, ws in by.items())
                first = f" (first error {r['first_error']})" if r.get("first_error") else ""
                lines.append(f"  episode {r['episode']}: {what}{first}")
        if d9 == weeks:
            lines.append("  Every week was naive's, so the score is exactly 0 whatever the agent does: fix the error.")
    dropped = sum(r["invalid_entries"] for r in out["episodes"])
    if dropped:
        lines.append(
            f"NOTE: {dropped} action entries were dropped as invalid (a negative or non-finite quantity, a flow or "
            "override on an edge prohibited this week: multiply flows by observation['action_mask'], or a "
            "release_mode outside 0, 1, 2, which voids its pair's overrides and keeps the default release); the rest "
            "of each action stood"
        )
    return lines


def diagnostic_report(out: dict) -> str:
    """The diagnostic table of a summary (costs in integer cents), for the organisers."""
    head = (
        f"{'n':>4} {'s':>2} {'J_policy_cents':>16} {'J_naive_cents':>16} {'J_oracle_cents':>16} {'RSS':>9} {'fb':>4} "
        f"{'inv':>5}"
    )
    lines = [head]
    for r in out["episodes"]:
        rss = "-" if r["rss"] is None else f"{r['rss']:.4f}"
        lines.append(
            f"{r['episode']:>4} {r['stratum'] or '-':>2} {r['J_policy_cents']:>16,} {r['J_naive_cents']:>16,} "
            f"{r['J_oracle_cents'] or 0:>16,} {rss:>9} {len(r['d9_substitutions']):>4} {r['invalid_entries']:>5}"
        )

    def fmt(x):
        return "null" if x is None else f"{x:.4f}"

    lines.append(
        "n episode; s harm stratum (1 calmest .. 4 most harmful); J costs in integer cents, lower is better; RSS 0 is "
        "naive, 1 the oracle; fb weeks naive played for your agent (fallbacks); inv action entries dropped as invalid"
    )
    strata = ", ".join(f"{s}: {fmt(v)}" for s, v in out["rss_by_stratum"].items())
    lines.append(f"RSS over all episodes: {fmt(out['rss_all'])}; by stratum: {strata}")
    why = "" if out["rss_pooled"] is not None else f" ({out['rss_pooled_null_reason']})"
    lines.append(f"RSS pooled over the strata: {fmt(out['rss_pooled'])}{why}")
    if out["excluded_from_rss"]:
        causes = ", ".join(f"{k} {v}" for k, v in out["exclusions"].items() if v)
        lines.append(f"excluded from RSS (an oracle not optimal): episodes {out['excluded_from_rss']} ({causes})")
    lines.append(f"fallbacks: {out['d9_weeks_total']} of {out['weeks_total']} weeks (naive played them)")
    return "\n".join(lines + warning_lines(out))


# what each fallback cause means, in the participant's words (``CAUSES`` for the organisers)
PLAIN_CAUSES = {
    "action": "your code raised an error or returned a malformed action (the traceback is above, in the lines "
    "starting 'shockbench-agent:')",
    "timeout": "no reply within the per-week time limit of {deadline:g} s",
    "killed": "your agent's process was gone: agent.py failed to import (the lines above say why) or the process "
    "exited",
    "unparsable": "the reply was not valid JSON: something in your code wrote to standard output's channel",
    "too_long": "the reply was longer than the size limit",
    "tags": "the reply named another episode",
    "cpu": "the week used more CPU time than its budget",
}
SCALE = "0 = the naive rule, 1 = the clairvoyant plan; higher is better"


def _usd(cents: int | float | None) -> str:
    return "-" if cents is None else f"${cents / 100:,.0f}"


def _plain_warnings(out: dict) -> list[str]:
    lines = []
    fell, weeks = out["d9_weeks_total"], out["weeks_total"]
    if fell:
        causes: dict[str, int] = {}
        for r in out["episodes"]:
            for _, cause in r["d9_substitutions"]:
                causes[cause] = causes.get(cause, 0) + 1
        deadline = float(out.get("wire", {}).get("deadline_s", 10.0))
        lines.append(f"WARNING: the naive rule played {fell} of {weeks} weeks for your agent:")
        for cause, k in causes.items():
            why = PLAIN_CAUSES.get(cause, "the agent's reply was not usable").format(deadline=deadline)
            lines.append(f"  {k} week{'s' if k != 1 else ''}: {why}")
        for r in out["episodes"]:
            ws = [w for w, _ in r["d9_substitutions"]]
            if ws:
                first = f" (first error in {r['first_error']})" if r.get("first_error") else ""
                lines.append(f"  episode {r['episode']}: {_runs(ws)}{first}")
        if fell == weeks:
            lines.append("  Every week was the naive rule's, so the score is exactly 0 whatever your agent does.")
    dropped = sum(r["invalid_entries"] for r in out["episodes"])
    if dropped:
        lines.append(
            f"NOTE: {dropped} action entries were ignored as invalid (a negative or non-finite quantity, a flow on a "
            "route closed this week: multiply flows by observation['action_mask'], or a release_mode outside 0, 1, "
            "2); the rest of each action was played"
        )
    return lines


def plain_report(out: dict) -> str:
    """The participant's report of a summary: the score on its 0-1 scale, its interval, costs in dollars, warnings."""
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS
    from shockbench_flow_agent.scoring import _boot_stats, _interval

    rows = out["episodes"]
    pooled = out["rss_pooled"] is not None
    score = out["rss_pooled"] if pooled else out["rss_all"]
    lines = [
        "Per episode (costs in dollars, lower is better; harm level 1 calmest .. 4 most harmful):",
        f"{'episode':>8} {'harm':>5} {'your agent':>18} {'naive rule':>18} {'clairvoyant plan':>18} {'score':>8} "
        f"{'naive weeks':>12}",
    ]
    for r in rows:
        rss = "-" if r["rss"] is None else f"{r['rss']:.4f}"
        lines.append(
            f"{r['episode']:>8} {r['stratum'] or '-':>5} {_usd(r['J_policy_cents']):>18} "
            f"{_usd(r['J_naive_cents']):>18} {_usd(r['J_oracle_cents']):>18} {rss:>8} {len(r['d9_substitutions']):>12}"
        )
    lines.append("")
    lines.append(f"Score: {'undefined' if score is None else f'{score:.4f}'}  ({SCALE})")
    kept = [r for r in rows if r["excluded"] is None and r["J_oracle_cents"] is not None]
    if score is not None and len(kept) >= 2:
        (boot,) = _boot_stats(rows, [[r["J_policy_cents"] for r in rows]], pooled, 2000, 0)
        ci = _interval(boot, 0.9)
        if ci:
            lines.append(f"  90% interval over the choice of episodes: {ci[0]:.4f} to {ci[1]:.4f}")
    weights = ", ".join(f"{w:.0%}" for w in STRATUM_WEIGHTS)
    if pooled:
        lines.append(f"  each harm level weighted as on the leaderboard ({weights})")
    else:
        lines.append(
            "  all episodes counted alike (the leaderboard weights the harm levels, which need one episode each)"
        )
    if any(v is not None for v in out["rss_by_stratum"].values()):
        by = ", ".join(f"{s}: {'-' if v is None else f'{v:.4f}'}" for s, v in out["rss_by_stratum"].items())
        lines.append(f"  by harm level: {by}")
    lines.append(f"Episodes: {len(rows)} ({out['instance']}, information regime {out['regime']})")
    if rows:
        mean = [sum(r[k] for r in rows) / len(rows) for k in ("J_policy_cents", "J_naive_cents")]
        oracle = None if len(kept) != len(rows) else sum(r["J_oracle_cents"] for r in rows) / len(rows)
        lines.append(
            f"Mean cost per episode: your agent {_usd(mean[0])}, naive rule {_usd(mean[1])}, clairvoyant plan "
            f"{_usd(oracle)}"
        )
    lines.append(f"Weeks the naive rule played for your agent: {out['d9_weeks_total']} of {out['weeks_total']}")
    if out["excluded_from_rss"]:
        lines.append(
            f"Left out of the score (the clairvoyant plan was not solved exactly): episodes {out['excluded_from_rss']}"
        )
    return "\n".join(lines + _plain_warnings(out))


def report(out: dict, verbose: bool = False) -> str:
    """The terminal report of a summary: plain for participants, with ``verbose`` the organisers' diagnostic table."""
    return diagnostic_report(out) if verbose else plain_report(out)


# ----- the participant's entry --------------------------------------------------------------------------------------
def dev_episodes(task: str, cuts: CutPoints, n_jobs: int = 1) -> list[int]:
    """The dev split's episode indices of a task: the first ``DEV_PER_STRATUM`` of each harm stratum.

    The trusted runner's rule and sizes (``hosting.split.fill_strata`` on the public dev root, the board's dev split).

    Raises:
        ValueError: if a stratum stays unfilled within ``tasks.MAX_CANDIDATES`` candidates.

    """
    inst, params = task_generator(task)
    scenarios, _drawn, unfilled = fill_strata(
        inst, params, DEV_ENTROPY, DEV_SPLIT, cuts, DEV_PER_STRATUM, max_candidates=MAX_CANDIDATES, n_jobs=n_jobs
    )
    if unfilled:
        raise ValueError(f"the dev split of {task!r} leaves strata {list(unfilled)} unfilled")
    return sorted(s.episode for s in scenarios)


def episode_list(episodes: str | int | Sequence[int]) -> list[int] | None:
    """The indices ``evaluate``'s ``episodes`` names: None for ``'dev'``, 0..k-1 for an int k, else the list.

    Raises:
        ValueError: on another string, an int < 1, or a list that is empty, repeats an index or holds a negative one.

    """
    if isinstance(episodes, str):
        if episodes != DEV_EPISODES:
            raise ValueError(f"episodes must be 'dev', a count or a list of indices, got {episodes!r}")
        return None
    if isinstance(episodes, bool):
        raise ValueError(f"episodes must be 'dev', a count or a list of indices, got {episodes!r}")
    if isinstance(episodes, int):
        if episodes < 1:
            raise ValueError(f"episodes: a count >= 1, got {episodes}")
        return list(range(episodes))
    ns = list(episodes)
    if not ns or len(set(ns)) != len(ns) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in ns):
        raise ValueError(f"episodes: distinct episode indices >= 0, got {ns!r}")
    return ns


def _agent_name(agent: object) -> str:
    return getattr(agent, "__qualname__", None) or type(agent).__qualname__


def agent_file(agent: object) -> Path | None:
    """The path of ``agent`` when it names a Python file (an ``agent.py``, as a ``str`` or a ``Path``), else None."""
    if isinstance(agent, (str, os.PathLike)):
        path = Path(agent)
        if path.suffix == ".py" and path.is_file():
            return path
    return None


def resolve_agent(agent: object, workdir: Path) -> tuple[Submission | AgentFactory, str]:
    """(the agent as ``score_agent`` takes it, its label): a zip or folder checked and extracted, a callable as is.

    A folder is zipped first as ``hosting.submission.build_submission`` zips it (every file but the ``SKIPPED``
    names), so its SHA-256, which salts the policy seed, is that zip's. A path to a ``.py`` file (``agent_file``) is
    that file's ``Agent`` class, loaded into this process (``shim.load_agent_class``; ``shim.unload_agent`` undoes it)
    and seeded as a class is.

    Raises:
        SubmissionError: when the validator refuses the zip (or the zipped folder).
        TypeError: when ``agent`` is none of a path, a class or a callable.
        FileNotFoundError: when a path names nothing.

    """
    file = agent_file(agent)
    if file is not None:
        return load_agent_class(file, f"submission_{file.stem}"), str(agent)
    if isinstance(agent, (str, os.PathLike)):
        path = Path(agent)
        if path.is_dir():
            path = build_submission(path, workdir / f"{path.resolve().name or 'submission'}.zip")
        elif not path.is_file():
            raise FileNotFoundError(f"{agent}: no such zip, folder or agent.py")
        return check_submission(path, workdir), str(agent)
    if callable(agent):
        return agent, _agent_name(agent)
    raise TypeError(
        f"agent must be an Agent class, a factory, a folder, a zip or an agent.py, got {type(agent).__name__}"
    )


@contextlib.contextmanager
def _unloaded(loaded: bool):
    """``shim.unload_agent`` at the end of the block when it loads an agent file into this process."""
    try:
        yield
    finally:
        if loaded:
            unload_agent()


def _command(agent_label: str, episodes, regime: str, mode: str, **options) -> str:
    """This call spelled out: the agent's label, episodes, regime and mode, and each option not at its default."""
    defaults = {"task": "tiny", "fq_replications": REPLICATIONS, "cut_draws": CUT_DRAWS}
    args = [repr(agent_label), f"episodes={episodes!r}", f"regime={regime!r}", f"mode={mode!r}"]
    args += [f"{k}={v!r}" for k, v in options.items() if v != defaults[k]]
    return f"shockbench_flow_agent.evaluate({', '.join(args)})"


def evaluate(
    agent: object,
    episodes: str | int | Sequence[int] = DEV_EPISODES,
    regime: str = "standard",
    n_jobs: int = -1,
    cache_dir: str | Path | bool | None = None,
    *,
    task: str = "tiny",
    mode: str | None = None,
    fq_replications: int = REPLICATIONS,
    cut_draws: int = CUT_DRAWS,
    command: str | None = None,
    verbose: bool = False,
) -> dict:
    """Score an agent on a public task's dev episodes: one summary dict, to the cent the organisers' evaluation's.

    Args:
        agent: an ``Agent`` class or any factory ``config -> agent`` (in process), a submission folder holding
            ``agent.py``, a submission zip (checked by the server's validator first), or a path to an ``agent.py``
            file (its ``Agent`` class, in process).
        episodes: ``'dev'``, the dev split the board scores locally (5 per harm level: 20 on every task); a count k,
            dev episodes 0..k-1; or a list of dev episode indices.
        regime: the information regime the agent plays (``standard``, the scored one; the naive rule always plays
            without predictions). ``clairvoyant`` is allowed: the dev scenarios are public.
        n_jobs: joblib workers of a first run's naive demand model and cut points (-1 all cores); no result depends
            on it.
        cache_dir: where the naive rule's demand model and the cut points are kept; None the user cache
            (``SBF_CACHE_DIR``, else ``~/.cache/shockbench-flow``); False computes them without a cache.
        task: a public task (``tiny``, ``small``, ``full``).
        mode: ``'subprocess'`` (the container's entry point: the default for a folder or a zip) or ``'in_process'``
            (the default, and the only mode, for a class, a factory or an agent file; no deadline is enforced).
        fq_replications: the naive rule's demand-model replications (keep the default for the board's numbers; 2
            for a smoke run).
        cut_draws: the harm cut points' draws (keep the default for the board's harm levels; 0 for none, which
            ``episodes='dev'`` needs).
        command: the summary's ``command`` field (default: this call, spelled out).
        verbose: print a progress line per episode; off by default.

    Returns:
        The summary dict (``score_agent``); ``report(summary)`` is its plain report, ``report(summary, verbose=True)``
        the organisers' diagnostic table.

    Raises:
        SubmissionError: when the validator refuses the zip or the zipped folder.
        ValueError: while ``SBF_ENTROPY`` or ``SBF_SCORES_KEY`` is set (``check_local_run``, before anything runs), on
            an unknown task, mode or regime, ``episodes`` outside its forms, a class in subprocess mode, or
            ``episodes='dev'`` with ``cut_draws=0``.

    """
    check_local_run(SPLIT)
    if regime not in REGIME_NAMES:
        raise ValueError(f"regime must be one of {REGIME_NAMES}, got {regime!r}")
    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    tsk = get_task(task)
    ns = episode_list(episodes)
    if ns is None and not cut_draws:
        raise ValueError("episodes='dev' fills the harm strata: it needs cut_draws >= 1")
    cache = None if cache_dir is False else (default_cache_dir() if cache_dir is None else Path(cache_dir))
    inst, params = task_generator(task)
    with tempfile.TemporaryDirectory(prefix="sbf-eval-") as work, _unloaded(agent_file(agent) is not None):
        workdir = Path(work)
        resolved, label = resolve_agent(agent, workdir)
        is_zip = isinstance(resolved, Submission)
        run_mode = mode or ("subprocess" if is_zip else "in_process")
        if ns is None:
            cuts = cut_points_cached(inst, params, cut_draws, CUT_ENTROPY, n_jobs, cache)
            ns = dev_episodes(task, cuts, n_jobs)
        else:
            cuts = None  # score_agent computes them (or none, cut_draws=0)
        sub = SubmissionRunConfig(path=label, mode=run_mode, episodes=len(ns), episode_list=ns, cut_draws=cut_draws)
        options = {"task": task, "fq_replications": fq_replications, "cut_draws": cut_draws}
        return score_agent(
            resolved,
            inst,
            params,
            sub,
            episodes=ns,
            regime=regime,
            entropy=DEV_ENTROPY,
            entropy_source="submission.dev_entropy (public)",
            gamma=tsk.gamma,
            fq_replications=fq_replications,
            n_jobs=n_jobs,
            cache_dir=cache,
            command=command or _command(label, episodes, regime, run_mode, **options),
            cuts=cuts,
            workdir=workdir,
            verbose=verbose,
        )


__all__ = [
    "CAUSES",
    "LIMITS_TAG",
    "agent_file",
    "SubmissionError",
    "SubmissionRunConfig",
    "check_local_run",
    "check_submission",
    "child_env",
    "dev_episodes",
    "diagnostic_report",
    "plain_report",
    "evaluate",
    "first_error",
    "report",
    "score_agent",
    "scores",
    "warning_lines",
]

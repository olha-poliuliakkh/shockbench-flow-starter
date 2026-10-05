"""The evaluator: fitness on cached episodes, paired statistics, static and smoke checks (EVOLVE_DESIGN.md 4.2).

Fitness is the board's own formula on integer cents (``EpisodeSet.rss``), computed in-process: nothing parses printed
output. A candidate is a submission folder played as the server plays it (validated as a zip, seeded by its SHA-256,
weeks over the CPU budget handed to naive), and its per-episode rows are cached on disk by (file hash, pool), so two
evaluated programs compare without replaying.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


WEIGHTS = (0.50, 0.30, 0.15, 0.05)
FRAME_FILES = ("frame.py", "sbfplan")  # frozen: identical to the seed's in every candidate
ALLOWED_IMPORTS = {
    "json",
    "sys",
    "math",
    "pathlib",
    "numpy",
    "frame",
    "collections",
    "itertools",
    "functools",
    "heapq",
    "bisect",
    "statistics",
    "dataclasses",
    "typing",
    "__future__",
}
BANNED_CALLS = {"open", "eval", "exec", "compile", "__import__", "globals", "input", "breakpoint", "setattr", "delattr"}
SMOKE_CPU_S = {"small": 0.5, "tiny": 0.5, "full": 1.5}  # p99 CPU per week allowed at S1 / S5


def files_sha256(folder: Path, only: tuple[str, ...] | None = None) -> str:
    """SHA-256 of the folder's files (relative path and bytes), caches excluded; ``only`` limits it to those entries."""
    h = hashlib.sha256()
    folder = Path(folder)
    for f in sorted(folder.rglob("*")):
        rel = f.relative_to(folder)
        if not f.is_file() or "__pycache__" in rel.parts:
            continue
        if only is not None and rel.parts[0] not in only:
            continue
        h.update(rel.as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


@dataclass
class Result:
    """A candidate's play of one pool: per-episode costs in cents, the score and the health counts."""

    pool: str
    J: list[int]
    rss: float | None
    fallback_weeks: int
    cpu_weeks: int
    invalid_entries: int
    first_error: str | None
    seconds: float
    rows: list[dict] = field(repr=False, default_factory=list)


def rss_of(es, J) -> float | None:
    return es.rss([int(j) for j in J])["rss"]


def dollars_by_level(es, J) -> dict[int, dict]:
    """Per level: n, mean saving g, mean attainable D, p_s * g (what RSS sums) and the level's ratio, in USD."""
    out = {}
    for s, p in enumerate(WEIGHTS, start=1):
        kept = [(r, j) for r, j in zip(es.references, J) if r["stratum"] == s and r["excluded"] is None]
        if not kept:
            continue
        g = sum(r["J_naive_cents"] - j for r, j in kept) / len(kept) / 100
        D = sum(r["J_naive_cents"] - r["J_oracle_cents"] for r, _ in kept) / len(kept) / 100
        out[s] = {"n": len(kept), "g": g, "D": D, "p_g": p * g, "ratio": g / D if D else None}
    return out


def descriptors(es, J) -> tuple[float, float]:
    """(calm RSS over levels 1-2, crisis RSS over levels 3-4), each pooled with the board's weights."""
    d = dollars_by_level(es, J)

    def pooled(levels):
        num = sum(WEIGHTS[s - 1] * d[s]["g"] for s in levels if s in d)
        den = sum(WEIGHTS[s - 1] * d[s]["D"] for s in levels if s in d)
        return num / den if den else 0.0

    return pooled((1, 2)), pooled((3, 4))


def paired_gap(es, J_a, J_b, n_boot: int = 2000, seed: int = 0, level: float = 0.90) -> dict:
    """RSS(a) - RSS(b) on the same episodes, with a stratified paired bootstrap (EpisodeSet.compare's method).

    Episodes are resampled within each harm level, the same draws for both programs, so the noise they share cancels.
    The score is pooled over the levels present with the board's weights: with all four levels it is the board's RSS;
    with some (a crises-only pool) it is that pool's own weighted score. Returns {diff, lo, hi, p_better}.
    """
    refs = es.references
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    oracle = np.array([r["J_oracle_cents"] if r["J_oracle_cents"] is not None else np.nan for r in refs], dtype=float)
    Ja, Jb = np.asarray(J_a, dtype=float), np.asarray(J_b, dtype=float)
    rng = np.random.default_rng(seed)
    groups = []
    for s, w in enumerate(WEIGHTS, start=1):
        idx = np.array([i for i, r in enumerate(refs) if r["stratum"] == s and r["excluded"] is None], dtype=int)
        if idx.size:
            groups.append((w, idx))

    def gap(picks) -> np.ndarray:
        num_a = sum(w * (naive[p] - Ja[p]).mean(axis=-1) for w, p in picks)
        num_b = sum(w * (naive[p] - Jb[p]).mean(axis=-1) for w, p in picks)
        den = sum(w * (naive[p] - oracle[p]).mean(axis=-1) for w, p in picks)
        return (num_a - num_b) / den

    diff = float(gap(groups))
    d = gap([(w, idx[rng.integers(0, idx.size, size=(n_boot, idx.size))]) for w, idx in groups])
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [(1 - level) / 2, (1 + level) / 2])
    return {"diff": diff, "lo": float(lo), "hi": float(hi), "p_better": float((d > 0).mean())}


def pool_key(es) -> str:
    """A short fingerprint of an EpisodeSet's content: task, regime, root, episodes and reference settings.

    Cached plays are keyed by it, so two pools that share a name (``screen`` at 8 or at 32 episodes per level, or the
    same name on another root) never read each other's results.
    """
    doc = [es.task, es.regime, int(es.entropy), [int(n) for n in es.episodes], es.fq_replications, es.cut_draws]
    return hashlib.sha256(json.dumps(doc).encode()).hexdigest()[:12]


class Evaluator:
    """Plays candidate folders on named EpisodeSets, caching per-episode rows on disk by (file hash, pool)."""

    def __init__(self, pools: dict, cache_dir: Path, workers: int = 8, composites: dict | None = None):
        """``pools``: name -> EpisodeSet. ``composites``: name -> part names, whose EpisodeSet (also in ``pools``)
        lists the parts' episodes in order: its rows are the parts' rows, so no episode is played twice."""
        self.pools, self.workers = pools, workers
        self.composites = composites or {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def play(self, folder: Path, pool: str) -> Result:
        import time

        es = self.pools[pool]
        if pool in self.composites:
            parts = [self.play(folder, part) for part in self.composites[pool]]
            J = [j for r in parts for j in r.J]
            if len(J) != len(es.episodes):
                raise ValueError(
                    f"composite pool {pool}: its parts hold {len(J)} episodes, the pool {len(es.episodes)}"
                )
            return Result(
                pool=pool,
                J=J,
                rss=rss_of(es, J),
                fallback_weeks=sum(r.fallback_weeks for r in parts),
                cpu_weeks=sum(r.cpu_weeks for r in parts),
                invalid_entries=sum(r.invalid_entries for r in parts),
                first_error=next((r.first_error for r in parts if r.first_error), None),
                seconds=sum(r.seconds for r in parts),
                rows=[row for r in parts for row in r.rows],
            )
        sha = files_sha256(folder)
        path = self.cache_dir / f"{sha[:24]}-{pool}-{pool_key(es)}.json"
        if path.is_file():
            doc = json.loads(path.read_text())
            if [r.get("episode") for r in doc.get("rows", [])] == list(es.episodes):
                return Result(**doc)  # else a stale or foreign entry: play again
        t = time.perf_counter()
        rows = es.play(str(folder), cpu_budget=True, n_jobs=self.workers)
        J = [int(r["J_policy_cents"]) for r in rows]
        keep = ("episode", "J_policy_cents", "fallback_weeks", "cpu_weeks", "invalid_entries", "first_error")
        res = Result(
            pool=pool,
            J=J,
            rss=rss_of(es, J),
            fallback_weeks=sum(r["fallback_weeks"] for r in rows),
            cpu_weeks=sum(r["cpu_weeks"] for r in rows),
            invalid_entries=sum(r["invalid_entries"] for r in rows),
            first_error=next((r["first_error"] for r in rows if r["first_error"]), None),
            seconds=round(time.perf_counter() - t, 1),
            rows=[{k: r.get(k) for k in keep} for r in rows],
        )
        path.write_text(json.dumps(res.__dict__) + "\n")
        return res


# ----- S0 and S1 ---------------------------------------------------------------------------------------------------
def block_spans(text: str) -> dict[str, tuple[int, int]]:
    """{block name: (start, end)} character spans of the text between each EVOLVE-BLOCK-START/END marker pair."""
    spans = {}
    for line_start in _find_all(text, "# EVOLVE-BLOCK-START "):
        name_end = text.index("\n", line_start)
        name = text[line_start + len("# EVOLVE-BLOCK-START ") : name_end].strip()
        end = text.index(f"# EVOLVE-BLOCK-END {name}", name_end)
        spans[name] = (name_end + 1, end)
    return spans


def _find_all(text: str, needle: str):
    i = text.find(needle)
    while i >= 0:
        yield i
        i = text.find(needle, i + 1)


def outside_blocks(text: str) -> str:
    """The text of agent.py with every block's body removed: the part that must equal the seed's."""
    out, last = [], 0
    for a, b in sorted(block_spans(text).values()):
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return "".join(out)


def static_check(folder: Path, seed: Path) -> list[str]:
    """S0: what the scorer and this loop refuse, read without importing anything. Empty when the candidate passes."""
    from shockbench_flow_agent.submission import agent_warnings, missing_imports

    folder, seed = Path(folder), Path(seed)
    problems = []
    text = (folder / "agent.py").read_text()
    try:
        ast.parse(text)
    except SyntaxError as err:
        return [f"agent.py does not parse: {err}"]
    if files_sha256(folder, FRAME_FILES) != files_sha256(seed, FRAME_FILES):
        problems.append("frame.py or sbfplan/ differs from the seed's (the frame is frozen)")
    if outside_blocks(text) != outside_blocks((seed / "agent.py").read_text()):
        problems.append("agent.py differs from the seed's outside the EVOLVE blocks")
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    for name in missing_imports(text.encode(), files):
        problems.append(f"agent.py imports {name}, which the scoring image lacks")
    for w in agent_warnings(text.encode(), files):
        problems.append(f"the scorer's kit warns: {w}")
    spans = block_spans(text)
    for name, (a, b) in spans.items():
        body = ast.parse(text[a:b])
        for node in ast.walk(body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS]
                if bad:
                    problems.append(f"block {name} imports {bad}: only the standard library's math tools and numpy")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                problems.append(f"block {name} calls {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "remove", "rmtree"}:
                problems.append(f"block {name} uses .{node.attr}")
    try:
        params = json.loads((folder / "params.json").read_text()) if (folder / "params.json").is_file() else {}
        if not all(isinstance(v, (int, float)) for v in params.values()):
            problems.append("params.json must map names to numbers")
    except ValueError as err:
        problems.append(f"params.json is not JSON: {err}")
    return problems


def smoke(folder: Path, task: str = "small", episode: int = 80, determinism_es=None, workers: int = 2):
    """S1: one dev episode in a process holding only the scoring image's packages (imports, CPU, substitutions), and
    a determinism check: the same episodes from a byte-different copy must cost the same.

    Returns (problems, stats): problems is empty when it passes; stats holds the CPU p99 and max per week.
    """
    from shockbench_flow_agent import play_isolated

    folder = Path(folder)
    problems = []
    row = play_isolated(folder.resolve(), task, episode)
    if not row["imported"]:
        return [f"agent.py does not import with the scoring image's packages: {row['stderr'][-800:]}"], {}
    if row["substitutions"]:
        problems.append(f"naive would play weeks {row['substitutions'][:10]} (exception, malformed action or timeout)")
    cpu = [c for c in row["cpu_s"] if c is not None]
    if cpu and float(np.quantile(cpu, 0.99)) > SMOKE_CPU_S[task]:
        problems.append(f"CPU p99 {np.quantile(cpu, 0.99):.3f} s per week over the {SMOKE_CPU_S[task]} s allowance")
    errors = [ln for ln in row["stderr"].splitlines() if "seed inputs this week" in ln or "sending the maximum" in ln]
    if errors:
        problems.append(f"{len(errors)} week(s) fell back inside the agent, first: {errors[0][:300]}")
    if determinism_es is not None and not problems:
        with tempfile.TemporaryDirectory(prefix="sbf-det-") as tmp:
            twin = Path(tmp) / "twin"
            shutil.copytree(folder, twin, ignore=shutil.ignore_patterns("__pycache__"))
            (twin / "agent.py").write_text((twin / "agent.py").read_text() + "\n# determinism probe\n")
            a = [r["J_policy_cents"] for r in determinism_es.play(str(folder), n_jobs=workers)]
            b = [r["J_policy_cents"] for r in determinism_es.play(str(twin), n_jobs=workers)]
        if a != b:
            problems.append("costs depend on the policy seed: the agent is not deterministic")
    return problems, {"cpu_p99": float(np.quantile(cpu, 0.99)) if cpu else None, "cpu_max": max(cpu, default=None)}

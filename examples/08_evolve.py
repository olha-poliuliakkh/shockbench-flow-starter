"""An AlphaEvolve-style search: Claude rewrites agent.py, the kit's score selects (06_policy_search with a model).

    uv run python examples/08_evolve.py                                    # Tiny, numeric mutation only
    uv run python examples/08_evolve.py --task=small --generations=8 --n_llm=4 --n_mutate=8 --train_episodes=16

A candidate is a submission folder. Each generation, Claude reads a parent's agent.py and its diagnostics (where it
loses against the clairvoyant plan, what disrupted those episodes) and writes a new agent.py (``--n_llm``; through
the Claude Code CLI ``claude -p`` on your subscription with ``--llm=cli``, or the API with ``--llm=api`` and
ANTHROPIC_API_KEY); the mutation of 06 perturbs the numbers of ``params.json`` or the ``PARAMS`` dict (``--n_mutate``).
Every candidate must pass the server's import rule, then plays the same training episodes of your own root under the
CPU budget. At the end the best is compared, paired, with the champion on the held-out dev episodes and replaces it
in ``agents/`` only when it wins there. Nothing here talks to Codabench.
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from shockbench_flow_agent.submission import agent_warnings, missing_imports
from shockbench_flow_gym import agent_config_from_reset
from shockbench_flow_gym.timeline import episode_events

from sbf_starter import ROOT, cpu_budget_s, env_id, scoring
from sbf_starter.agents import resolve


FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
EDIT = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", re.DOTALL)
WORKERS = 1  # joblib workers that play one candidate's episodes (set from --workers)
SYSTEM = """You take part in an evolutionary search over control programs for ShockBench-Flow, a Gymnasium
supply-network task (network `{task}`, T = {T} weeks). Each round you get one parent program (a complete agent.py),
its measured results and diagnostics, and you write ONE improved agent.py. A scorer plays it on many episodes; the
best programs become the next parents.

THE TASK. Each week the agent chooses how much of each commodity to send along each action slot (an edge, or the
first edge of a sea lane through one or more straits). The environment clips orders to stock and capacity, moves
goods with lead times, runs fabs and power grids, serves demand and charges the week's cost: freight + war-risk +
tariffs + holding (higher for cargo queued at a strait) + a penalty per unit of unmet demand + disposal + power shed,
minus the value of what is left at the end. Lower is better. Disruptions (strait closures, sanctions, tariffs,
conflicts, factory outages) are drawn before the episode and nothing the agent does changes them. Score: 0 = the
naive rule (ships the nominal plan, sees no disruption), 1 = the clairvoyant plan (a linear program that knew every
disruption in advance). "Send the maximum on every allowed route" scores about {send_max} on this network.
The naive rule does NOT see `warning.score` (early-warning scores per region, rival pair and strait, one-week lag),
`messages.*` (tariff proposals and notices, sanction and military threats; some are false alarms) and
`pending_prohibitions.*` (announced sanctions with the week they take effect): using them well is where to gain.
Also: `graph_now.open` (how open each strait is, 1 to 0; cargo sent into a closed strait waits and pays queue
holding), `graph_now.u` (this week's capacities), `demand_forecast.qty` (8 weeks ahead), `stock.qty`, `backlog.qty`,
`last_week.cost_components`.

THE CONTRACT (a violation hands weeks to the naive rule):
  class Agent:
      def __init__(self, config=None): ...        # once per episode; counts toward week 1's CPU budget
      def act(self, observation): return {{"flows": flows}}   # optionally "override_qty" and "release_mode" too
- config: `static` (tables nodes, edges, lanes, commodities, action_slots, override_slots, sinks; the public instance
  JSON under static["instance"]), `T`, `policy_seed` (seed all randomness from it), `layout` (what each row of a dense
  block means: stock_slots, demands, chokepoints, fabs, grids, osats, warning_units, cost_components, release_pairs,
  lot_keys when present), `spaces` (every array's shape and dtype). READ EVERY SHAPE FROM config["spaces"].
- observation: a dict of numpy arrays keyed by strings; every field x has a mask x.observed. Padded lists.
- flows: one float >= 0 per action slot; entries on a slot with action_mask == 0, negative or non-finite are
  dropped: multiply by action_mask. action_mask ignores closures and capacity: read graph_now.open and graph_now.u.
- CPU: {budget:g} s per week for the whole process. Vectorise with numpy.
- Imports: ONLY the standard library, numpy, scipy and torch. No file writing, no network, no subprocesses or threads.
  Files beside agent.py are read at module level with Path(__file__).resolve().parent / "name".
- The nominal plan (what naive ships): entries of static["instance"]["initial_state"]["pipeline"] with
  dispatch_week == 0, each with edge (id string), k (commodity id), lane (lane id or null), qty. Slot s matches
  (static["edges"]["id"][slots["edge"][s]], static["commodities"]["id"][slots["k"][s]], lane id or None).
- Nominal capacities: static["edges"]["u0"]; a lane's straits: static["lanes"]["chokepoints"] (node indices), and
  config["layout"]["chokepoints"] maps a strait node to its row of graph_now.open.

FIELDS OF THIS NETWORK:
{fields}

{answer}"""
FOCUS = (  # one direction per parallel call of a generation, so the calls do not all propose the same idea
    "the signals the naive rule does not see: warning.score, messages.* (tariff and sanction notices with their stated "
    "effective weeks, threats, some false alarms) and pending_prohibitions",
    "strait closures and tanker cargo: when to send into, hold at or release from a strait, and rerouting around it",
    "power grids and fuel: rationing, shed, the fuel the grids and the fabs burn, and stock built before a cut",
    "the chip chain and unmet demand: materials, fabs, OSATs, lead times, and which demand to serve first",
    "the plan itself: the window, the terminal value of stock and cargo, the demand forecast beyond 8 weeks, and "
    "planning against uncertain capacities instead of the forecast as if it were certain",
)
ANSWER = {
    "full": """
HOW TO ANSWER. Use the diagnostics: which episodes lose, what disrupted them, which cost dominates. Change one or
two clear things per round and keep what works. Put tunable numbers in a module-level PARAMS dict, overridden by an
optional params.json beside the file (a numeric mutation perturbs them). Reply with a few sentences on what you
changed and why, then the COMPLETE agent.py in one fenced block tagged python; optionally a fenced json block with
params.json. The program must also run on larger networks (read all sizes from config).""",
    "diff": """
HOW TO ANSWER. Use the diagnostics (which episodes lose, which cost dominates, in which weeks) and the log of
ideas already tried: do not repeat an idea that failed, build on ideas that helped. Make ONE clear change per
reply. Start with a few sentences: the idea and why it should help on these diagnostics. Then the change as one or
more edit blocks on agent.py, each exactly:
<<<<<<< SEARCH
lines copied character for character from the parent's agent.py, enough of them to occur exactly once
=======
the lines that replace them
>>>>>>> REPLACE
Blocks apply in order; a SEARCH that is missing or occurs twice rejects the whole reply. Put tunable numbers in the
module-level PARAMS dict (with "_bounds"; a numeric mutation perturbs them). A params.json beside the file overrides
PARAMS: to change a number it sets, add a fenced json block with the COMPLETE new params.json. The program must also
run on larger networks (read all sizes from config).""",
}


def apply_edits(code: str, text: str) -> tuple[str | None, str | None]:
    """(new code, None), or (None, why not): the reply's SEARCH/REPLACE blocks applied in order, else its complete
    agent.py in a python block."""
    blocks = EDIT.findall(text)
    if not blocks:
        for lang, body in FENCE.findall(text):
            if lang.lower() in ("python", "py") and "class Agent" in body:
                return body, None
        return None, "the reply holds no edit block and no agent.py"
    for i, (search, replace) in enumerate(blocks, 1):
        n = code.count(search)
        if n != 1:
            first = search.strip().splitlines()[0][:80] if search.strip() else ""
            return None, f"edit block {i}: its SEARCH occurs {n} times ({first!r})"
        code = code.replace(search, replace, 1)
    return code, None


def short(note: str | None, n: int = 300) -> str:
    return " ".join((note or "").split())[:n]


def check(folder: Path) -> list[str]:
    """What the server would not run: an import it lacks in any .py, or a broken contract in agent.py."""
    if not (folder / "agent.py").is_file():
        return ["no agent.py at the folder's root"]
    files = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
    problems = []
    for rel in files:
        if rel.endswith(".py"):
            problems += [
                f"{rel} imports {m}, which the server lacks"
                for m in missing_imports((folder / rel).read_bytes(), files)
            ]
    problems += [w for w in agent_warnings((folder / "agent.py").read_bytes(), files) if "imports" not in w]
    return problems


def params_of(folder: Path) -> dict | None:
    """params.json beside agent.py, else the module-level ``PARAMS = {...}`` literal of agent.py."""
    if (folder / "params.json").is_file():
        return json.loads((folder / "params.json").read_text())
    for node in ast.parse((folder / "agent.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PARAMS" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


def mutate(params, rng: np.random.Generator, sigma: float, bounds: dict | None = None, key=None):
    """A Gaussian step on every number (nested lists and dicts too), relative to its size, kept at 0 or above.

    ``params["_bounds"] = {"key": [lo, hi]}`` bounds the top-level keys it names; ``params["_fixed"]`` lists the
    top-level keys left as they are (switches, features turned off).
    """
    top = bounds is None and isinstance(params, dict)
    fixed = set(params.get("_fixed", ())) if top else set()
    bounds = params.get("_bounds", {}) if top else (bounds or {})
    if isinstance(params, dict):
        return {
            k: v if k in ("_bounds", "_fixed") or k in fixed else mutate(v, rng, sigma, bounds, key or k)
            for k, v in params.items()
        }
    if isinstance(params, list):
        return [mutate(v, rng, sigma, bounds, key) for v in params]
    if isinstance(params, bool) or not isinstance(params, (int, float)):
        return params
    lo, hi = bounds.get(key, (0.0, np.inf))
    new = float(np.clip(params + rng.normal(0.0, sigma) * max(abs(params), 0.1), lo, hi))
    return int(round(new)) if isinstance(params, int) else new


def fitness(episodes, folder: Path) -> dict:
    """The kit's score on the training episodes under the CPU budget, with one compact row per episode."""
    s = episodes.score(str(folder), name=folder.name, cpu_budget=True, n_jobs=WORKERS)
    return {
        "rss": s.rss,
        "by_harm": dict(s.rss_by_stratum or {}),
        "fallback_weeks": s.fallback_weeks,
        "cpu_weeks": s.cpu_weeks,
        "rows": rows_of(s),
    }


def rows_of(s) -> list[dict]:
    """One compact row per episode of a kit Score."""
    return [
        {
            "episode": r["episode"],
            "harm": r.get("stratum"),
            "cost": r["J_policy_cents"] / 100,
            "naive": r["J_naive_cents"] / 100,
            "clairvoyant": (r.get("J_clairvoyant_cents") or 0) / 100,
            "fallback_weeks": r.get("fallback_weeks", 0),
            "first_error": r.get("first_error"),
        }
        for r in s.rows
    ]


def loss(row: dict) -> float:
    """The share of the attainable saving NOT kept on an episode: 0 is the clairvoyant plan, 1 is naive."""
    attainable = row["naive"] - row["clairvoyant"]
    return (row["cost"] - row["clairvoyant"]) / attainable if attainable > 0 else 0.0


def narrative(env, episode: int, limit: int = 25) -> str:
    """The disruptions of an episode after week 1 (week 1 is the state in force at the start), one line each."""
    events = [
        e
        for e in episode_events(env, options={"episode": episode})
        if e["week"] > 1 and e["signal"] != "message_update"
    ]
    lines = [
        f"  week {e['week']:>3}: {e['signal']} {e['subject']} {e.get('before')} -> {e.get('after')} "
        f"{e.get('note') or ''}".rstrip()
        for e in events[:limit]
    ]
    if len(events) > limit:
        lines.append(f"  ... {len(events) - limit} more")
    return "\n".join(lines) or "  (no disruption after week 1)"


def breakdown(env, folder: Path, episode: int, top: int = 5) -> str:
    """The candidate replayed on one episode in this process: its cost by component and its costliest weeks."""
    try:
        spec = importlib.util.spec_from_file_location(f"candidate_{folder.name}_{episode}", folder / "agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        obs, info = env.reset(options={"episode": episode})
        config = agent_config_from_reset(env, obs, info)
        names = list(config["layout"]["cost_components"])
        agent, weeks, done = module.Agent(config), [], False
        while not done:
            obs, _reward, terminated, truncated, _info = env.step(agent.act(obs))
            weeks.append(np.asarray(obs["last_week.cost_components"], dtype=float))
            done = terminated or truncated
    except Exception as err:  # noqa: BLE001 - a diagnostic must not stop the search
        return f"  (replay failed: {type(err).__name__}: {err})"
    W = np.array(weeks)
    total = W.sum(0)
    lines = ["  cost by component: " + ", ".join(f"{n} ${c:,.0f}" for n, c in zip(names, total) if abs(c) > 0)]
    for t in np.argsort(-W.sum(1))[:top]:
        parts = ", ".join(f"{names[i]} ${W[t, i]:,.0f}" for i in np.argsort(-W[t])[:3] if W[t, i] > 0)
        lines.append(f"  week {t + 1:>3}: ${W[t].sum():,.0f} ({parts})")
    return "\n".join(lines)


def feedback(result: dict, env, budget: float, folder: Path | None = None, worst: int = 2) -> str:
    rows = sorted(result["rows"], key=loss, reverse=True)
    out = [
        f"training score {result['rss']:.4f} on {len(rows)} episodes; by harm level {result['by_harm']}; "
        f"weeks played by the naive rule (crash or malformed action) {result['fallback_weeks']}, "
        f"weeks over the {budget:g} s CPU budget {result['cpu_weeks']}",
        "per episode (loss = share of the attainable saving not kept; 0 clairvoyant, 1 naive, above 1 worse than "
        "naive):",
    ]
    for r in rows:
        err = f"  FIRST ERROR {r['first_error']}" if r.get("first_error") else ""
        out.append(
            f"  episode {r['episode']:>3} harm {r['harm']}: loss {loss(r):.2f}  cost ${r['cost']:,.0f}  "
            f"naive ${r['naive']:,.0f}  clairvoyant ${r['clairvoyant']:,.0f}{err}"
        )
    for r in rows[:worst]:
        out += [f"what happened in episode {r['episode']} (loss {loss(r):.2f}):", narrative(env, r["episode"])]
        if folder is not None:
            out += [f"where this program paid in episode {r['episode']}:", breakdown(env, folder, r["episode"])]
    return "\n".join(out)


def ask_claude(system: str, user: str, model: str, effort: str) -> tuple[str, str]:
    """(reply text, stop reason); the system prompt is cached between calls."""
    import anthropic

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=32_000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    ) as stream:
        message = stream.get_final_message()
    return "".join(b.text for b in message.content if b.type == "text"), message.stop_reason


def ask_claude_cli(system: str, user: str, model: str, effort: str) -> tuple[str, str]:
    """The same question through the Claude Code CLI (``claude -p``, your subscription): a text-only call, no tools.

    Runs outside the repository so the CLI reads no project instructions; one call may take several minutes.
    """
    cmd = ["claude", "-p", "--output-format", "text", "--tools", "", "--no-session-persistence", "--strict-mcp-config"]
    cmd += ["--system-prompt", system, "--model", model, "--effort", effort]
    proc = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=1800, cwd=tempfile.gettempdir())
    if proc.returncode != 0:
        return proc.stdout, f"exit {proc.returncode}: {proc.stderr.strip()[-500:]}"
    return proc.stdout, "end_turn"


ASK = {"api": ask_claude, "cli": ask_claude_cli}
DEFAULT_MODEL = {"api": "claude-opus-5-5", "cli": "opus"}
DEFAULT_FAST = {"api": "claude-sonnet-5-5", "cli": "sonnet"}


def llm_available(llm: str) -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) if llm == "api" else shutil.which("claude") is not None


def propose_llm(
    parent: dict,
    folder: Path,
    system: str,
    model: str,
    effort: str,
    llm: str = "cli",
    edit: str = "diff",
    context: str = "",
) -> tuple[str | None, str | None]:
    """Ask for a child of ``parent`` and write it to ``folder``: (the model's explanation, None), or (explanation or
    None, why there is no child)."""
    code = Path(parent["folder"], "agent.py").read_text()
    pjson = Path(parent["folder"], "params.json")
    user = (
        f"# Parent program {parent['id']} (training score {parent['rss']:.4f})\n```python\n{code}\n```\n"
        + (f"params.json:\n```json\n{pjson.read_text()}\n```\n" if pjson.is_file() else "")
        + (f"\n{context}\n" if context else "")
        + f"\n# Diagnostics of the parent\n{parent['feedback']}\n"
        + (
            f"On the larger Full network (104 weeks, the private board) this program scores {parent['full_rss']:.4f}; "
            "a child that loses there is not kept.\n"
            if parent.get("full_rss") is not None
            else ""
        )
        + ("\nWrite your one change now as edit blocks." if edit == "diff" else "\nWrite the improved agent.py now.")
    )
    text, stop = ASK[llm](system, user, model, effort)
    folder.mkdir(parents=True)
    (folder / "reply.md").write_text(f"# parent {parent['id']}, model {model} (stop {stop})\n\n{text}")
    note = EDIT.sub("", FENCE.sub("", text)).strip()[:1000] or None
    params = next((body for lang, body in FENCE.findall(text) if lang.lower() == "json"), None)
    if edit == "diff":
        new, error = apply_edits(code, text)
    else:
        new, error = next(
            ((b, None) for lang, b in FENCE.findall(text) if lang.lower() in ("python", "py") and "class Agent" in b),
            (None, "the reply holds no agent.py"),
        )
    if new is None:
        return note, error
    if edit == "diff":  # the parent's whole program travels: other modules, data files, its params.json
        shutil.copytree(
            parent["folder"],
            folder,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "reply.md", "feedback.md", "agent.py"),
        )
    else:
        for name in Path(parent["folder"]).iterdir():  # the parent's data files (weights, tables) travel with the code
            if name.is_file() and name.suffix not in (".py", ".json", ".md"):
                shutil.copy(name, folder / name.name)
    (folder / "agent.py").write_text(new.rstrip() + "\n")
    if params is not None:
        (folder / "params.json").write_text(params.rstrip() + "\n")
    return note, None


def main(
    task: str = "tiny",
    entropy: int = 20261003,
    train_episodes: int = 8,
    holdout: str | int | list[int] = "dev",
    generations: int = 2,
    rotate: bool = False,
    full_episodes: int = 0,
    full_entropy: int = 777002,
    full_tol: float = 0.01,
    n_llm: int = 0,
    llm_every: int = 1,
    edit: str = "diff",
    model_fast: str | None = None,
    n_fast: int = 0,
    cascade_episodes: int = 0,
    cascade_tol: float = 0.02,
    n_mutate: int = 4,
    elite: int = 3,
    sigma: float = 0.15,
    quick: bool = False,
    n_jobs: int = -1,
    workers: int = 4,
    llm: str = "cli",
    model: str | None = None,
    effort: str = "high",
    seeds: tuple[str, ...] = ("mine", "heuristic"),
    champion: str = "mine",
    promote: bool = True,
    promote_floor: float = -0.05,
    seed: int = 0,
    out: str | None = None,
) -> None:
    """Search, then confirm the best against the champion held out; promote it into agents/<champion> if it wins.

    Args:
        task: tiny, small (the public board's) or full.
        entropy: your training root: any integer but 0 (the dev episodes).
        train_episodes: episodes of that root every candidate plays (the same ones: paired differences).
        holdout: held-out dev episodes for the final check: dev, a count or a list.
        generations: rounds of the search.
        rotate: each generation plays new training episodes (root entropy + generation); the parents play them too,
            so a generation's ranking stays paired and no fixed set is fitted.
        full_episodes: the Full gate: a candidate becomes a parent only if, on this many Full episodes of root
            full_entropy, it scores at least the seeds' best minus full_tol with no fallback or over-budget week
            (0: no gate). The Full references take minutes per episode the first time.
        full_entropy: the Full gate's root.
        full_tol: how much below the seeds a candidate may score on Full.
        n_llm: children per generation written by Claude.
        llm_every: Claude writes children only every this many generations (saves the subscription's limits).
        edit: diff (Claude answers with SEARCH/REPLACE blocks on the parent's agent.py: one idea, few tokens) or full
            (the complete agent.py). The generation's calls run in parallel; each sees the log of ideas tried.
        model_fast: the model of the first n_fast calls of a generation (breadth; default sonnet for cli,
            claude-sonnet-5-5 for api); the others use model (depth).
        n_fast: how many of the n_llm calls use model_fast.
        cascade_episodes: a child first plays this many of the training episodes; it goes on to all of them only if
            its mean loss there is at most cascade_tol above its parent's on the same episodes (0: no cascade).
        cascade_tol: see cascade_episodes.
        n_mutate: children per generation from the numeric mutation.
        elite: candidates kept as parents.
        sigma: the mutation's scale.
        quick: seconds, not the leaderboard's numbers (smoke tests only).
        n_jobs: workers of the references' first computation (-1: all cores).
        workers: joblib workers that play a candidate's episodes in parallel.
        llm: how Claude is called: cli (``claude -p``, your Claude Code subscription) or api (ANTHROPIC_API_KEY).
        model: the Claude model of the proposer (default: opus for cli, claude-opus-5-5 for api).
        effort: its effort level (low, medium, high, xhigh, max).
        seeds: agents that start the archive (names of agents/ or folders).
        champion: the agent of agents/ the best must beat held out; replaced by it on a win (a copy is kept).
        promote: False never touches agents/.
        promote_floor: the lower end of the paired 90 % interval must be above this, and the difference above 0.
        seed: the search's generator.
        out: the run folder (default: outputs/08_evolve/<date_time>).

    """
    if entropy == 0:
        raise ValueError("train on a root of your own (--entropy=...): root 0 holds the dev episodes of the check")
    out = Path(out or f"outputs/08_evolve/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    budget = cpu_budget_s(task)
    env = gym.make(env_id(task), entropy=entropy)
    T = int(env.unwrapped.instance.T)
    global WORKERS
    WORKERS = max(1, min(workers, train_episodes))
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)
    fields = (ROOT / "docs" / "fields" / f"{task}.md").read_text().partition("**Action slots**")[0]
    if edit not in ANSWER:
        raise ValueError(f"edit must be one of {list(ANSWER)}, got {edit!r}")
    system = SYSTEM.format(
        task=task,
        T=T,
        budget=budget,
        fields=fields.strip(),
        send_max={"tiny": 0.68, "small": 0.41}.get(task, "?"),
        answer=ANSWER[edit],
    )
    if llm not in ASK:
        raise ValueError(f"llm must be one of {list(ASK)}, got {llm!r}")
    model = model or DEFAULT_MODEL[llm]
    model_fast = model_fast or DEFAULT_FAST[llm]
    if n_llm and not llm_available(llm):
        need = "ANTHROPIC_API_KEY" if llm == "api" else "the claude command"
        print(f"{need} is not available: n_llm = 0 (numeric mutation only)")
        n_llm = 0
    archive: list[dict] = []
    log = out / "archive.jsonl"

    def cascade_set(root: int):
        if not 0 < cascade_episodes < train_episodes:
            return None
        return scoring.episode_set(task, cascade_episodes, quick=quick, entropy=root, n_jobs=n_jobs)

    # the training episodes the current generation plays, and their first cascade_episodes
    cur = {"train": train, "env": env, "entropy": entropy, "cascade": cascade_set(entropy)}
    ideas: list[dict] = []  # Claude's children: the idea, its model, what became of it
    ideas_log = out / "ideas.jsonl"
    full_set = (
        scoring.episode_set("full", full_episodes, quick=quick, entropy=full_entropy, n_jobs=n_jobs)
        if full_episodes
        else None
    )
    full_ref = {"rss": -np.inf}  # the seeds' best Full score

    def write(rec: dict) -> None:
        with log.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def write_idea(idea: dict) -> None:
        with ideas_log.open("a") as f:
            f.write(json.dumps(idea, default=str) + "\n")

    def cascade(folder: Path, parent: dict) -> str | None:
        """Why ``folder`` stops after the cascade's episodes, or None when it goes on (or there is no cascade)."""
        if cur["cascade"] is None or parent.get("entropy") != cur["entropy"] or not parent.get("rows"):
            return None
        s = cur["cascade"].score(str(folder), name=folder.name, cpu_budget=True, n_jobs=WORKERS)
        if s.rss is None or s.fallback_weeks or s.cpu_weeks:
            return f"on {cascade_episodes} episodes: {s.fallback_weeks} fallback weeks, {s.cpu_weeks} over budget"
        mine = {r["episode"]: loss(r) for r in rows_of(s)}
        theirs = [loss(r) for r in parent["rows"] if r["episode"] in mine]
        if len(theirs) != len(mine):
            return None
        a, b = float(np.mean(list(mine.values()))), float(np.mean(theirs))
        if a - b > cascade_tol:
            return f"mean loss {a:.3f} against the parent's {b:.3f} on the first {cascade_episodes} episodes"
        return None

    def evaluate(
        folder: Path,
        gen: int,
        parents: list[str],
        note: str,
        rec: dict | None = None,
        parent: dict | None = None,
        model: str | None = None,
    ) -> dict:
        """Score a new candidate (through the cascade when it has a ``parent``), or ``rec`` again."""
        rec = rec or {"id": folder.name, "folder": str(folder), "gen": gen, "parents": parents, "note": note}
        rec |= {"rss": None, "scored_gen": gen, "entropy": cur["entropy"]}
        if model:
            rec["model"] = model
        problems = check(folder)
        dropped = None if problems or parent is None else cascade(folder, parent)
        if problems:
            rec["feedback"] = "REJECTED before scoring: " + "; ".join(problems)
        elif dropped:
            rec["cascade"] = dropped
            rec["feedback"] = f"dropped at the cascade: {dropped}"
        else:
            rec |= fitness(cur["train"], folder)
            rec["feedback"] = (
                feedback(rec, cur["env"], budget, folder)
                if rec["rss"] is not None
                else "the kit could not compute a score"
            )
        (folder / "feedback.md").write_text(rec["feedback"] + "\n")
        if not any(r is rec for r in archive):
            archive.append(rec)
        write(rec)
        shown = (
            f"score {rec['rss']:.4f}, fallback weeks {rec['fallback_weeks']}, over budget {rec['cpu_weeks']}"
            if rec["rss"] is not None
            else rec["feedback"][:200]
        )
        print(f"  {folder.name}: {shown}")
        return rec

    def full_ok(rec: dict) -> bool:
        """The Full gate, scored once per candidate."""
        if full_set is None:
            return True
        if "full_rss" not in rec:
            s = full_set.score(rec["folder"], name=rec["id"], cpu_budget=True, n_jobs=WORKERS)
            rec |= {"full_rss": s.rss, "full_fallback_weeks": s.fallback_weeks, "full_cpu_weeks": s.cpu_weeks}
            rec["full_ok"] = (
                s.rss is not None and s.fallback_weeks == 0 and s.cpu_weeks == 0 and s.rss >= full_ref["rss"] - full_tol
            )
            write({k: rec[k] for k in ("id", "folder", "gen", "full_rss", "full_fallback_weeks", "full_cpu_weeks")})
            if "idea" in rec:
                rec["idea"] |= {"full": s.rss, "full_ok": rec["full_ok"]}
                write_idea(rec["idea"])
            verdict = "passes" if rec["full_ok"] else f"held back (seeds {full_ref['rss']:.4f}, tolerance {full_tol})"
            print(
                f"  {rec['id']}: Full {s.rss if s.rss is None else round(s.rss, 4)}, fallback weeks "
                f"{s.fallback_weeks}, over budget {s.cpu_weeks}: {verdict}"
            )
        return rec["full_ok"]

    def context(others: list[dict], last: int = 20) -> str:
        """The other parents' ideas and the log of Claude's ideas so far, for a prompt."""
        out_ = []
        if others:
            out_.append("# Other programs of the elite (their ideas, scored on the same episodes)")
            out_ += [f"- {o['id']} (gen {o['gen']}, score {o['rss']:.4f}): {short(o['note'])}" for o in others]
        if ideas:
            out_.append(
                "# Ideas already tried, oldest first (delta: the child's score minus its parent's on the same "
                "episodes; above 0 helped)"
            )
            for d in ideas[-last:]:
                if "delta" in d:
                    what = f"delta {d['delta']:+.4f}"
                    if "full" in d:
                        what += f", Full {d['full']:.4f}" + ("" if d.get("full_ok") else " (held back: lost on Full)")
                else:
                    what = d["status"]
                out_.append(f"- gen {d['gen']} ({d['model']}, from {d['parent']}): {d['idea']} => {what}")
        return "\n".join(out_)

    def ranking(pool: list[dict]) -> list[dict]:
        """The ``elite`` best of ``pool`` by training score that pass the Full gate."""
        chosen = []
        for r in sorted((r for r in pool if r["rss"] is not None), key=lambda r: -r["rss"]):
            if len(chosen) == elite:
                break
            if full_ok(r):
                chosen.append(r)
        return chosen

    print(f"{task}: training on {train_episodes} episodes of root {entropy}; CPU budget {budget:g} s/week")
    if rotate:
        print(f"each generation g plays the {train_episodes} episodes of root {entropy} + g")
    if full_set is not None:
        print(f"Full gate: {full_episodes} episodes of root {full_entropy}, tolerance {full_tol}")
    print(f"run folder: {out}")
    print("generation 0: seeds")
    for name in seeds:
        folder = out / "gen0" / f"seed_{Path(name).name}"
        shutil.copytree(resolve(name), folder, ignore=shutil.ignore_patterns("__pycache__"))
        evaluate(folder, 0, [], f"seed {name}")
    for r in archive:
        if r["rss"] is not None and full_ok(r) and r.get("full_rss") is not None:
            full_ref["rss"] = max(full_ref["rss"], r["full_rss"])
    current = list(archive)  # the candidates scored on the current training episodes
    for g in range(1, generations + 1):
        parents = ranking(current)
        if not parents:
            raise SystemExit("no candidate is scored and passes the Full gate: see the archive")
        if rotate:
            cur["entropy"] = entropy + g
            cur["train"] = scoring.episode_set(task, train_episodes, quick=quick, entropy=cur["entropy"], n_jobs=n_jobs)
            cur["env"] = gym.make(env_id(task), entropy=cur["entropy"])
            cur["cascade"] = cascade_set(cur["entropy"])
            print(f"generation {g}: root {cur['entropy']}; the parents play it again")
            current = [evaluate(Path(p["folder"]), g, p["parents"], p["note"], rec=p) for p in parents]
            parents = ranking(current) or parents
        print(f"generation {g}: parents {[(p['id'], round(p['rss'], 4)) for p in parents]}")
        jobs = []
        for i in range(n_llm if g % max(1, llm_every) == 0 else 0):
            parent = parents[i % len(parents)]
            m = model_fast if i < n_fast else model
            others = [p for p in parents if p is not parent]
            focus = FOCUS[(g * max(1, n_llm) + i) % len(FOCUS)]
            ctx = context(others) + f"\n# Direction for this reply\nLook for your one change in {focus}, unless the "
            ctx += "diagnostics point clearly elsewhere."
            jobs.append((i, parent, m, out / f"gen{g}" / f"llm_{i:02d}", ctx))
        replies = []
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures = [pool.submit(propose_llm, p, f, system, m, effort, llm, edit, c) for _, p, m, f, c in jobs]
                for fut in futures:
                    try:
                        replies.append(fut.result())
                    except Exception as err:  # noqa: BLE001 - an API error must not stop the search
                        replies.append((None, f"{type(err).__name__}: {err}"))
        for (i, parent, m, folder, _), (note, error) in zip(jobs, replies, strict=True):
            idea = {"gen": g, "id": f"gen{g}/llm_{i:02d}", "parent": parent["id"], "model": m, "idea": short(note)}
            idea["focus"] = FOCUS[(g * max(1, n_llm) + i) % len(FOCUS)][:40]
            if error:
                idea["status"] = error
                print(f"  llm_{i:02d} ({m}): {error}")
            else:
                rec = evaluate(folder, g, [parent["id"]], note, parent=parent, model=m)
                if rec["rss"] is not None:
                    idea |= {"status": "scored", "delta": rec["rss"] - parent["rss"]}
                else:
                    idea["status"] = rec["feedback"][:200]
                rec["idea"] = idea
                current.append(rec)
            ideas.append(idea)
            write_idea(idea)
        with_params = [p for p in parents if params_of(Path(p["folder"]))]
        for i in range(n_mutate if with_params else 0):
            parent = with_params[rng.integers(len(with_params))]
            folder = out / f"gen{g}" / f"mutate_{i:02d}"
            shutil.copytree(
                parent["folder"], folder, ignore=shutil.ignore_patterns("__pycache__", "reply.md", "feedback.md")
            )
            (folder / "params.json").write_text(
                json.dumps(mutate(params_of(Path(parent["folder"])), rng, sigma)) + "\n"
            )
            current.append(evaluate(folder, g, [parent["id"]], f"mutation of {parent['id']}", parent=parent))
        top = ranking(current)
        if top:
            print(f"generation {g}: best {top[0]['id']} (gen {top[0]['gen']}) {top[0]['rss']:.4f}")

    top = ranking(current)
    if not top:
        raise SystemExit("no candidate passes the Full gate: see the archive")
    best = top[0]
    full = f", Full {best['full_rss']:.4f}" if best.get("full_rss") is not None else ""
    print(f"\nthe best candidate: {best['id']} training score {best['rss']:.4f}{full} at {best['folder']}")
    (out / "best.txt").write_text(best["folder"] + "\n")
    champ = resolve(champion)
    if best["note"] == f"seed {champion}":
        print("it is the champion itself: nothing to promote")
        return
    cmp = held_out.compare(best["folder"], str(champ), names=(best["id"], champion), cpu_budget=True, n_jobs=WORKERS)
    print(f"held out, on {len(held_out.episodes)} dev episodes:\n{cmp}")
    wins = cmp.diff is not None and cmp.diff > 0 and cmp.interval is not None and cmp.interval[0] > promote_floor
    if wins and promote:
        backup = out / f"{champion}_before_promotion"
        shutil.copytree(champ, backup, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.rmtree(champ)
        shutil.copytree(best["folder"], champ, ignore=shutil.ignore_patterns("__pycache__", "reply.md", "feedback.md"))
        print(f"promoted {best['id']} into {champ} (the previous {champion} is in {backup})")
        print(f"next: uv run sbf check {champion} --task={task}")
    elif wins:
        print(f"it beats {champion} held out; promotion is off (--promote): copy {best['folder']} yourself")
    else:
        print(f"not promoted: the held-out episodes do not confirm a gain over {champion}")


if __name__ == "__main__":
    fire.Fire(main)

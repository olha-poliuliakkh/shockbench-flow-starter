"""Every example runs on Tiny with small settings, in a subprocess, as `uv run python examples/<name>.py` runs it."""

import importlib.util
import json
import subprocess
import sys

import pytest

from tests.conftest import ROOT


EXAMPLES = ROOT / "examples"


def run(name: str, *args: str, env: dict, cwd, timeout: float = 300) -> str:
    """Run an example in ``cwd``; its output (stdout and stderr) when it exits 0."""
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, f"{name} {args} exited {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout + proc.stderr


def test_quickstart(env_with_cache, tmp_path):
    out = run("01_quickstart.py", env=env_with_cache, cwd=tmp_path)
    assert "26 weeks" in out and "options={'episode': 0}" in out and "USD with random actions" in out


def test_play_agents(env_with_cache, tmp_path):
    out = run("02_play_agents.py", "--episodes=1", env=env_with_cache, cwd=tmp_path)
    assert "random" in out and "template" in out and "USD" in out


def test_heuristic_agent(env_with_cache, tmp_path):
    out = run("03_heuristic_agent.py", "--episodes=1", env=env_with_cache, cwd=tmp_path)
    assert "a strait falls below 50% open: [29]" in out
    assert "in total the rule saves " in out and "in total the rule saves -" not in out  # it acts where it closes


def test_evaluate_and_compare(env_with_cache, tmp_path):
    out = run("04_evaluate.py", "--quick", "--n_jobs=1", env=env_with_cache, cwd=tmp_path)
    assert "Score:" in out and "interval" in out and "quick:" in out
    agent_file = ROOT / "agents" / "heuristic" / "agent.py"
    args = [f"--agent={agent_file}", "--against=template", "--quick", "--n_jobs=1"]
    out = run("04_evaluate.py", *args, env=env_with_cache, cwd=tmp_path)
    assert "A - B:" in out and "paired interval" in out


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="the rl extra is not installed")
def test_train_ppo(env_with_cache, tmp_path):
    args = ["--total_timesteps=416", "--n_envs=1", "--n_scenarios=2", "--check_episodes=1", "--activation=relu"]
    args += ["--net_arch=[16]", "--out=run"]
    out = run("05_train_ppo.py", *args, env=env_with_cache, cwd=tmp_path)
    folder = tmp_path / "run"
    assert (folder / "submission.zip").is_file() and (folder / "submission" / "policy.pt").is_file()
    assert "POLICY = torch.jit.load" in (folder / "submission" / "agent.py").read_text()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["settings"]["net_arch"] == [16] and "max_flow_difference" in out


def test_policy_search(env_with_cache, tmp_path):
    args = ["--generations=1", "--population=2", "--train_episodes=2", "--quick", "--n_jobs=1", "--out=run"]
    out = run("06_policy_search.py", *args, env=env_with_cache, cwd=tmp_path)
    assert "held out, on 4 dev episodes" in out and "A - B:" in out
    best = tmp_path / "run" / "best"
    assert best.is_dir() == ("not written" not in out)  # written only when it beats send-the-maximum held out
    if best.is_dir():
        params = json.loads((best / "params.json").read_text())
        assert len(params["fraction"]) == 20 and (best / "agent.py").is_file()


def test_evolve(env_with_cache, tmp_path):
    args = ["--generations=1", "--n_mutate=2", "--train_episodes=2", "--quick", "--n_jobs=1", "--workers=1"]
    args += ["--seeds=[heuristic]", "--champion=heuristic", "--nopromote", "--out=run"]
    out = run("08_evolve.py", *args, env=env_with_cache, cwd=tmp_path)
    assert "it is the champion itself" in out or "held out" in out
    archive = [json.loads(ln) for ln in (tmp_path / "run" / "archive.jsonl").read_text().splitlines()]
    assert [r["gen"] for r in archive] == [0, 1, 1] and all(r["rss"] is not None for r in archive)
    assert (tmp_path / "run" / "gen1" / "mutate_01" / "params.json").is_file()


def test_evolve_rotate(env_with_cache, tmp_path):
    args = ["--generations=2", "--n_mutate=1", "--train_episodes=2", "--quick", "--n_jobs=1", "--workers=1"]
    args += ["--rotate", "--elite=1", "--seeds=[heuristic]", "--champion=heuristic", "--nopromote", "--out=run"]
    run("08_evolve.py", *args, env=env_with_cache, cwd=tmp_path)
    archive = [json.loads(ln) for ln in (tmp_path / "run" / "archive.jsonl").read_text().splitlines()]
    assert [r["scored_gen"] for r in archive] == [0, 1, 1, 2, 2]  # each generation's parent plays its root again
    assert [r["entropy"] for r in archive] == [20261003, 20261004, 20261004, 20261005, 20261005]
    assert "where this program paid" in archive[0]["feedback"]


def test_dashboard(env_with_cache, tmp_path):
    run("07_dashboard.py", "--episode=29", "--quick", "--n_jobs=1", "--out=run", env=env_with_cache, cwd=tmp_path)
    for name in ("network.png", "dashboard_max.png", "episode_max.gif", "record_random.npz"):
        assert (tmp_path / "run" / name).is_file(), name


def test_apply_edits():
    spec = importlib.util.spec_from_file_location("evolve", EXAMPLES / "08_evolve.py")
    evolve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evolve)
    code = "a = 1\nb = 2\nc = 2\n"
    edit = "idea\n<<<<<<< SEARCH\na = 1\n=======\na = 10\n>>>>>>> REPLACE\n"
    assert evolve.apply_edits(code, edit) == ("a = 10\nb = 2\nc = 2\n", None)
    removed = "<<<<<<< SEARCH\nb = 2\nc = 2\n=======\n>>>>>>> REPLACE"
    assert evolve.apply_edits(code, removed) == ("a = 1\n\n", None)
    missing = evolve.apply_edits(code, "<<<<<<< SEARCH\nd = 4\n=======\nd = 5\n>>>>>>> REPLACE")
    assert missing[0] is None and "occurs 0 times" in missing[1]
    twice = evolve.apply_edits(code, "<<<<<<< SEARCH\n= 2\n=======\n= 3\n>>>>>>> REPLACE")
    assert twice[0] is None and "occurs 2 times" in twice[1]
    full = "here\n```python\nclass Agent:\n    pass\n```\n"
    assert evolve.apply_edits(code, full) == ("class Agent:\n    pass\n", None)
    assert evolve.apply_edits(code, "no code")[0] is None

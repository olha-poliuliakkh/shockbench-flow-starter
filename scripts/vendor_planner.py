"""Vendor the installed shockbench_flow into an agent folder as ``sbfplan``, for the LP planner (mpc_det) to run there.

    uv run python scripts/vendor_planner.py agents/mine            # copy, then prune to the modules the agent loads
    uv run python scripts/vendor_planner.py agents/mine --noprune  # copy every module (development)

The scoring server has numpy, SciPy and torch only, so the copy (MIT, its LICENSE kept) is renamed ``sbfplan`` (it
never shadows the host's package during local runs) and its one module-level third-party import, fastjsonschema in
``instance/io.py``, is replaced by a no-op stub: the instance JSON comes from the server's own Static and needs no
validation. Pruning plays the agent on whole dev episodes in a fresh process and keeps the modules it imported. Rerun
this after ``uv sync --upgrade-package shockbench-flow``, then ``uv run pytest tests/test_planner_port.py``.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import fire


NAME = "sbfplan"
WORD = re.compile(r"\bshockbench_flow\b(?!_)")
STUB = '''"""A stand-in for fastjsonschema, which the scoring image lacks.

Validation is skipped: the instance JSON is the server's own Static.
"""


class JsonSchemaValueException(ValueError):
    message = name = rule = path = value = definition = ""


def compile(schema, **kwargs):  # noqa: A001 - the replaced library's name
    return lambda data: data
'''
PROBE = """
import sys
import gymnasium as gym
import shockbench_flow_gym
from shockbench_flow_agent import load_agent_class
cls = load_agent_class({root!r})
for task, n in {episodes!r}:
    shockbench_flow_gym.play_episode(gym.make("ShockBench/" + task.capitalize() + "-v0"), cls, n)
print("\\n".join(sorted(m for m in sys.modules if m == {name!r} or m.startswith({name!r} + "."))))
"""


def copy_package(dest: Path) -> Path:
    import shockbench_flow

    src = Path(shockbench_flow.__file__).resolve().parent
    out = dest / NAME
    if out.exists():
        shutil.rmtree(out)
    for f in sorted(src.rglob("*.py")):
        rel = f.relative_to(src)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = WORD.sub(NAME, f.read_text())
        if rel.as_posix() == "instance/io.py":
            text = text.replace("import fastjsonschema\n", f"from {NAME} import _fastjsonschema as fastjsonschema\n")
        # highspy is imported only inside the warm-start solver, which the frame never calls (it solves with SciPy)
        text = re.sub(r"^(\s*)import highspy$", r"\1highspy = None  # not on the scoring server", text, flags=re.M)
        target.write_text(text)
    (out / "_fastjsonschema.py").write_text(STUB)
    schema = src / "instance" / "data" / "instance.schema.json"
    (out / "instance" / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(schema, out / "instance" / "data" / schema.name)
    dist = next(src.parent.glob("shockbench_flow-*.dist-info"))
    for lic in (dist / "licenses").glob("*") if (dist / "licenses").is_dir() else []:
        shutil.copy(lic, out / lic.name)
    return out


def prune(dest: Path, episodes) -> list[str]:
    """Keep the modules a fresh process imports while the agent plays ``episodes`` ((task, dev episode) pairs)."""
    code = PROBE.format(root=str(dest.resolve()), episodes=[tuple(e) for e in episodes], name=NAME)
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=dest)
    if run.returncode:
        raise SystemExit(f"the probe failed:\n{run.stderr[-3000:]}")
    keep = {m for m in run.stdout.split() if m.startswith(NAME)}
    out = dest / NAME
    for f in sorted(out.rglob("*.py")):
        rel = f.relative_to(dest).with_suffix("")
        mod = ".".join(rel.parts[:-1]) if rel.name == "__init__" else ".".join(rel.parts)
        if mod not in keep and f.name != "_fastjsonschema.py":
            f.unlink()
    for d in sorted(out.rglob("__pycache__"), reverse=True):
        shutil.rmtree(d)
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return sorted(keep)


def main(dest: str = "agents/mine", noprune: bool = False, probe=(("small", 80), ("tiny", 29))) -> None:
    """Copy the package into ``dest``/sbfplan and, unless ``--noprune``, keep the modules the agent imports.

    Args:
        dest: the agent folder (its agent.py must import the planner through ``sbfplan``).
        noprune: keep every module.
        probe: (task, dev episode) pairs the agent plays to find its modules (Small 80 is a level-4 crisis).

    """
    d = Path(dest)
    out = copy_package(d)
    print(f"copied into {out}")
    if not noprune:
        keep = prune(d, probe)
        n = len(list(out.rglob("*.py")))
        print(f"kept {len(keep)} modules ({n} files) the agent imports on {list(probe)}")


if __name__ == "__main__":
    fire.Fire(main)

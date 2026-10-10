"""Build one submission that plays a different agent on each network: Small (T <= 52) and Full.

    uv run python scripts/build_dispatch_agent.py              # Small: compact_hierarchical, Full: twopass48_credit
    uv run python scripts/build_dispatch_agent.py --small=compact_hierarchical --full=compact_hierarchical --out=...

The submission (``agents/final_dispatch`` by default) holds:
- ``agent.py``: an ``Agent`` that builds the Small agent when ``config["T"] <= 52``, else the Full one, and hands it
  every ``act``;
- ``small/`` and ``full/``: each source agent's ``agent.py`` and ``params*.json`` (each reads its ``params.json``
  beside its own file, so the two keep their own knobs);
- every other top-level module, package and data file of the two sources (``compact_hier/``, ``frame.py``,
  ``sbfplan/``) at the root, where the scorer's loader and ``sbf check`` look for a submission's own imports.
  Two sources with a same-named module must hold the same file; otherwise the build stops.
- ``SOURCES.json``: the source folders and the SHA-256 of every copied file.

Edit the source agents and rebuild; never edit the copies.
"""

import hashlib
import json
import shutil
from pathlib import Path

import fire


ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__"}

DISPATCH = '''"""ShockBench-Flow agent: a different planner per network, each the best measured on it.

- Small (T <= 52): ``small/agent.py``, a copy of ``agents/{small}``.
- Full: ``full/agent.py``, a copy of ``agents/{full}``.

Built by ``scripts/build_dispatch_agent.py`` (``SOURCES.json`` lists the sources); rebuild instead of editing the
copies. Both planners are imported here, at module level, so their imports never count toward a week's CPU.
"""

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SMALL_MAX_T = 52


def _load(sub: str, name: str):
    if str(HERE) not in sys.path:  # the shared modules sit beside this file
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / sub / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Agent


SmallAgent = _load("small", "final_dispatch_small")
FullAgent = _load("full", "final_dispatch_full")


class Agent:
    def __init__(self, config):
        impl = SmallAgent if int(config["T"]) <= SMALL_MAX_T else FullAgent
        self.impl = impl(config)

    def act(self, observation):
        return self.impl.act(observation)
'''


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(folder: Path) -> dict[str, Path]:
    """Relative path -> file, every file under ``folder`` but caches."""
    return {
        str(p.relative_to(folder)): p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not SKIP & set(p.relative_to(folder).parts)
    }


def build(small: str = "compact_hierarchical", full: str = "twopass48_credit", out: str = "") -> Path:
    """Write the submission folder and return it.

    Args:
        small: the agent (a folder of agents/, or a path) played on Small.
        full: the agent played on Full.
        out: the submission folder (default agents/final_dispatch); replaced if it exists.

    """
    sources = {}
    for role, name in (("small", small), ("full", full)):
        folder = Path(name) if Path(name).is_dir() else ROOT / "agents" / name
        if not (folder / "agent.py").is_file():
            raise FileNotFoundError(f"{role}: no agent.py in {folder}")
        sources[role] = folder.resolve()
    target = Path(out) if out else ROOT / "agents" / "final_dispatch"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    copied, shared = {}, {}
    for role, folder in sources.items():
        for rel, path in _files(folder).items():
            top = rel.split("/")[0]
            if top == "agent.py" or (top.startswith("params") and top.endswith(".json")):
                dest = target / role / rel  # each agent's own file and knobs
            else:
                if rel in shared and shared[rel] != _sha(path):
                    raise ValueError(f"{rel}: the two sources hold different files of that name")
                shared[rel] = _sha(path)
                dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            copied[str(dest.relative_to(target))] = _sha(path)
    (target / "agent.py").write_text(DISPATCH.format(small=sources["small"].name, full=sources["full"].name))
    sources_out = {
        role: str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder)
        for role, folder in sources.items()
    }
    (target / "SOURCES.json").write_text(json.dumps({"sources": sources_out, "files": copied}, indent=1) + "\n")
    print(f"built {target}: Small -> {sources_out['small']}, Full -> {sources_out['full']}, {len(copied)} files")
    return target


if __name__ == "__main__":
    fire.Fire(build)

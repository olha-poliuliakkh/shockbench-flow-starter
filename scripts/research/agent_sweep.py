"""Play agent variants on the same episodes, under the CPU budget, and print RSS by level with paired gaps.

    uv run python scripts/research/agent_sweep.py --variants="twopass;twopass48"
    uv run python scripts/research/agent_sweep.py --variants="twopass;twopass48;twopass48:LONG_HORIZON=False" \\
        --task=full --root=20261006 --per_level=2

A variant is an agent folder name (agents/<name>), optionally followed by ``:NAME=value,NAME=value`` overrides of
module-level constants of its agent.py (the line ``NAME = ...`` is replaced), as ``frame.NAME=value`` of its frame.py,
or, as ``params.NAME=value``, of entries of its params.json (the copy lives in the run folder). The
first variant is the reference of the paired gaps. Plays go through shockbench-flow's own ``EpisodeSet.play`` with
``cpu_budget=True``: a week over the budget is played by the naive rule, as on the server (CPU is metered on this
machine; ``sbf check --docker`` meters the scoring container's). Results: outputs/research/agent_sweep/<date_time>/.
"""

import json
import re
import shutil
from pathlib import Path

import fire
from common import ROOT_DIR, episodes, fmt_row, run_dir


def materialize(variant: str, run: Path) -> tuple[str, str]:
    """(label, folder) of a variant: the agent folder itself, or a copy with its constants overridden."""
    name, _, sets = variant.partition(":")
    src = ROOT_DIR / "agents" / name
    if not sets:
        return variant, str(src)
    dest = run / "agents" / re.sub(r"[^A-Za-z0-9_.-]+", "_", variant)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    text = (dest / "agent.py").read_text()
    frame = (dest / "frame.py").read_text() if (dest / "frame.py").is_file() else ""
    params = json.loads((dest / "params.json").read_text()) if (dest / "params.json").is_file() else {}
    for item in sets.split(","):
        key, value = (x.strip() for x in item.split("=", 1))
        if key.startswith("params."):
            params[key.removeprefix("params.")] = json.loads(value)
            continue
        if key.startswith("frame."):
            name_ = key.removeprefix("frame.")
            frame, n = re.subn(rf"^{name_} = .*$", f"{name_} = {value}", frame, count=1, flags=re.M)
            if n != 1:
                raise ValueError(f"{name}/frame.py has no module-level line '{name_} = ...'")
            continue
        text, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"{name}/agent.py has no module-level line '{key} = ...'")
    (dest / "agent.py").write_text(text)
    if frame:
        (dest / "frame.py").write_text(frame)
    (dest / "params.json").write_text(json.dumps(params, indent=1, sort_keys=True) + "\n")
    return variant, str(dest)


def main(variants: str, task: str = "small", root: int = 0, per_level: int = 0, workers: int = 3) -> None:
    """Play every variant and print the table (module docstring).

    Args:
        variants: ';'-separated variants, the first the reference.
        task: small or full.
        root: 0 with ``per_level`` 0 is the Small dev split; otherwise a private root.
        per_level: episodes per harm level of a stratified private pool (0: the dev split).
        workers: processes.

    """
    from sbf_starter.evolve.evaluate import paired_gap

    es = episodes(task, root, per_level, workers)
    run = run_dir("agent_sweep")
    print(f"{len(es.references)} {task} episodes (root {root}); run folder {run}")
    print("| variant | RSS | L1 | L2 | L3 | L4 | paired gap vs the first [90 %] | weeks over budget | naive weeks |")
    out, ref = {}, None
    for v in [x for x in variants.split(";") if x]:
        label, folder = materialize(v, run)
        rows = es.play(folder, cpu_budget=True, n_jobs=workers)
        J = [int(r["J_policy_cents"]) for r in rows]
        if ref is None:
            ref, gap = J, "-"
        else:
            g = paired_gap(es, J, ref)
            gap = f"{g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]"
        cpu = sum(int(r.get("cpu_weeks", 0)) for r in rows)
        naive = sum(int(r.get("fallback_weeks", 0)) for r in rows)
        print(f"{fmt_row(label, es.rss(J))} {gap} | {cpu} | {naive} |", flush=True)
        out[label] = {"rss": es.rss(J)["rss"], "rows": rows}
    (run / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

"""Play one agent folder on episodes, serially, and pair it against stored per-episode costs (no reruns).

    uv run python scripts/research/play_folder.py --folder=<path> --stored="scripts/research/results/x.json:key;..."
    uv run python scripts/research/play_folder.py --folder=agents/milp48_eco --task=full --root=20261006 --per_level=2

A stored reference is ``file.json:key``: an agent_sweep results.json (rows with J_policy_cents) or a research
results.json (rows with J), the variant ``key`` inside it; its episodes must be the same. Plays through
shockbench-flow's ``EpisodeSet.play`` (``cpu_budget`` as on the server, metered on this machine). Results:
outputs/research/play_folder/<date_time>/results.json.
"""

import json

import fire
from common import episodes, fmt_row, run_dir


def stored_costs(spec: str, ns: list[int]) -> tuple[str, list[int]]:
    path, _, key = spec.partition(":")
    d = json.load(open(path))
    rows = d[key]["rows"] if isinstance(d, dict) else d
    by = {int(r["episode"]): int(r.get("J_policy_cents", r.get("J"))) for r in rows}
    return key, [by[n] for n in ns]


def main(
    folder: str, stored: str = "", task: str = "small", root: int = 0, per_level: int = 0, cpu_budget: bool = True
):
    """Play ``folder`` serially and print RSS by level, the paired gaps against ``stored`` and the naive weeks."""
    from sbf_starter.evolve.evaluate import paired_gap

    es = episodes(task, root, per_level, 1)
    ns = [r["episode"] for r in es.references]
    rows = es.play(folder, cpu_budget=cpu_budget, n_jobs=1)
    J = [int(r["J_policy_cents"]) for r in rows]
    naive = sum(int(r.get("fallback_weeks", 0)) for r in rows)
    over = sum(int(r.get("cpu_weeks", 0)) for r in rows)
    print(f"{len(ns)} {task} episodes (root {root}); weeks over budget {over}, weeks played by naive {naive}")
    print("| agent | RSS | L1 | L2 | L3 | L4 |")
    print(fmt_row(folder, es.rss(J)))
    for spec in [s for s in stored.split(";") if s]:
        key, ref = stored_costs(spec, ns)
        g = paired_gap(es, J, ref)
        print(f"vs {key} ({es.rss(ref)['rss']:.4f}): {g['diff']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]")
    run = run_dir("play_folder")
    (run / "results.json").write_text(json.dumps({folder: {"rss": es.rss(J)["rss"], "rows": rows}}, default=str))
    print(f"written {run / 'results.json'}")


if __name__ == "__main__":
    fire.Fire(main)

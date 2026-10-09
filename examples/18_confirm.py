"""Confirm a candidate against the base agent, paired, on the stratified sets of our own roots and on dev.

The candidate is ``agents/<agent>`` with PARAMS overrides; the base is the same code with none. Both play the same
episodes; the difference of the board's RSS gets a 90 % bootstrap interval resampled within harm levels
(``examples/13_param_search.py``'s helpers). The stratified sets come from ``13_param_search.py --stage=sets``.

    uv run python examples/18_confirm.py --agent=mpc_safe2 --over='{"safety_frac": 2.0}'
    uv run python examples/18_confirm.py --agent=mpc_safe2 --over=... --sets=dev --task=full
"""

import importlib.util
import json
from pathlib import Path

import fire
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]


def main(
    agent: str = "mpc_safe2", over: str = "{}", sets: str = "20261008,20261009", jobs: int = 8, task: str = "small"
):
    """``sets``: comma-separated own roots (their stratified sets must exist) and/or ``dev``."""
    from shockbench_flow_agent import EpisodeSet

    sp = importlib.util.spec_from_file_location("ps", ROOT / "examples" / "13_param_search.py")
    ps = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(ps)
    ps.CODE = ROOT / "agents" / agent / "agent.py"
    ps.BASE = json.loads((ROOT / "agents" / agent / "params.json").read_text())
    over = json.loads(over) if isinstance(over, str) else dict(over)
    for name in str(sets).split(","):
        if name == "dev":
            es = EpisodeSet.build(task, "dev")
        else:
            root = int(name)
            eps = json.loads((ps.OUT / "sets" / f"{task}_{root}_400_12.json").read_text())
            es = EpisodeSet.build(task, eps, entropy=root)
        eps = list(es.episodes)
        J = {}
        for label, o in (("cand", over), ("base", {})):
            out = Parallel(n_jobs=jobs)(delayed(ps.play)(task, ep, es._spec, o) for ep in eps)
            J[label] = {str(ep): j for ep, j, _m, _x in out}
            rss = es.rss([J[label][str(ep)] for ep in eps])["rss"]
            print(f"  {name} {label}: RSS {rss:.4f}, CPU max {max(x for *_a, x in out):.2f} s/week", flush=True)
        d, lo, hi = ps.paired(es, J["cand"], J["base"])
        print(f"{name}: cand - base {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    fire.Fire(main)

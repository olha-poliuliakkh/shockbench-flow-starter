"""The kit's former console scripts, each a Fire command (no longer installed since 0.1.2).

The wheel installed ``sbf-validate`` and ``sbf-eval`` up to 0.1.1; the starter repository's ``sbf`` command replaces
them. The functions stay: ``validate(zip)`` and ``evaluate(agent, ...)`` below, and ``validate_main()`` and
``eval_main()``, which need the ``fire`` package (no longer a dependency of the wheel).

- ``validate(zip_path)``: the server's check of a submission zip, never importing it (``submission.main``): exit
  status 0 with the zip's SHA-256, or 2 with the refusal code and reason.
- ``evaluate(agent, episodes='dev', regime='standard', out=None, verbose=False, ...)``: the local evaluation
  (``local_eval.evaluate``) on a public task's dev episodes, printed in plain words (``verbose``: the organisers'
  diagnostic table); exit status 2 when the validator refuses the zip. ``episodes`` is ``dev`` (the board's dev split),
  a count k (episodes 0..k-1) or a list (``'[0,3,5]'``).
"""

import json
import shlex
import sys
from pathlib import Path


REFUSED = 2  # exit status of a zip the validator refuses (the runner's own)


def validate(zip_path: str) -> None:
    """Check a submission zip as the trusted runner does; exit 0 when it passes, 2 when it is refused."""
    from shockbench_flow_agent.submission import main

    sys.exit(main([str(zip_path)]))


def evaluate(
    agent: str,
    episodes: str | int | list[int] = "dev",
    regime: str = "standard",
    n_jobs: int = -1,
    cache_dir: str | None = None,
    mode: str | None = None,
    task: str = "tiny",
    fq_replications: int | None = None,
    cut_draws: int | None = None,
    out: str | None = None,
    verbose: bool = False,
) -> None:
    """Score a submission folder or zip on dev episodes and print the report (``shockbench_flow_agent.evaluate``).

    Args:
        agent: a submission zip, or a folder holding ``agent.py``.
        episodes: ``dev`` (the board's dev split), a count k (dev episodes 0..k-1) or a list of indices.
        regime: the information regime the agent plays (``standard``, the scored one).
        n_jobs: workers of a first run's naive demand model and cut points (-1 all cores).
        cache_dir: the cache of the naive rule's demand model and the cut points (default ``SBF_CACHE_DIR``, else
            ``~/.cache/shockbench-flow``).
        mode: ``subprocess`` (default, the container's entry point) or ``in_process``.
        task: a public task (``tiny``).
        fq_replications: the naive rule's demand-model replications (default 1,000, the board's; 2 for a smoke run).
        cut_draws: the harm cut points' draws (default 2,000, the board's; 0 for no strata).
        out: a path to write the JSON summary to.
        verbose: the organisers' diagnostic table and a progress line per episode instead of the plain report.

    """
    from shockbench_flow_agent.local_eval import SubmissionError, evaluate, report

    kwargs = {"task": task, "mode": mode, "command": shlex.join(["sbf-eval", *sys.argv[1:]]), "verbose": verbose}
    kwargs |= {k: v for k, v in (("fq_replications", fq_replications), ("cut_draws", cut_draws)) if v is not None}
    try:
        summary = evaluate(str(agent), episodes, regime, n_jobs, cache_dir, **kwargs)
    except SubmissionError as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        sys.exit(REFUSED)
    print(report(summary, verbose=verbose))
    if summary["submission_sha256"]:
        print(f"submission id (the zip's SHA-256, which seeds your agent): {summary['submission_sha256']}")
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"written {path}")


def _fire():
    try:
        import fire
    except ImportError:
        raise ImportError("the kit's former console scripts need fire: pip install fire") from None
    return fire


def validate_main() -> None:
    """``sbf-validate`` (needs ``fire``)."""
    _fire().Fire(validate, name="sbf-validate")


def eval_main() -> None:
    """``sbf-eval`` (needs ``fire``)."""
    _fire().Fire(evaluate, name="sbf-eval")

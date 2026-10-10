"""The submission validator and zip builder, exactly as the scoring server runs them.

A re-export: the one implementation lives on the scorer's side (``shockbench_flow.hosting.submission``, standard
library only), so a local ``check_zip`` refuses exactly what the server refuses, with the same codes and limits
(``SubmissionLimits()``: 500 MiB unpacked, 1,000 files; ``shockbench_flow_agent.LIMITS.submission``).

``python -m shockbench_flow_agent.submission <zip>`` runs that check and says the verdict in words: exit status 0 and
the zip's SHA-256 (the submission id that salts the policy seed) when it passes, with ``agent_warnings``' notes on
mistakes the server accepts but that hand weeks to the naive rule; exit status 2, the refusal code and reason, and
what to fix (``REFUSAL_FIXES``) when it is refused; ``help(shockbench_flow.hosting.submission)`` lists every refusal
code and its rule. ``build_submission(folder, zip)`` zips a folder the way the server
rebuilds it (sorted members, fixed times and modes, so the same files give the same SHA-256).
"""

import sys
import zipfile

from shockbench_flow.hosting.submission import (
    AGENT_CLASS,
    AGENT_FILE,
    IMAGE_PACKAGES,
    REFUSAL_FIXES,
    Submission,
    SubmissionError,
    SubmissionLimits,
    agent_warnings,
    build_submission,
    check_zip,
    extract_submission,
    missing_imports,
)


__all__ = [
    "AGENT_CLASS",
    "AGENT_FILE",
    "IMAGE_PACKAGES",
    "REFUSAL_FIXES",
    "Submission",
    "SubmissionError",
    "SubmissionLimits",
    "agent_warnings",
    "build_submission",
    "check_zip",
    "extract_submission",
    "main",
    "missing_imports",
]

REFUSED, USAGE = 2, 64  # exit statuses: a refused zip (the runner's own), a usage error (EX_USAGE)


def main(argv: list[str]) -> int:
    """The check of the module docstring on ``argv[0]``; prints the verdict and returns the exit status."""
    if len(argv) != 1:
        print("usage: python -m shockbench_flow_agent.submission <submission.zip>", file=sys.stderr)
        return USAGE
    try:
        sub = check_zip(argv[0])
    except SubmissionError as err:
        print(f"REFUSED: {err}")
        if err.code in REFUSAL_FIXES:
            print(f"what to fix: {REFUSAL_FIXES[err.code]}")
        print("every code and its rule: python -m pydoc shockbench_flow.hosting.submission")
        return REFUSED
    print(
        f"OK: {argv[0]} passes the runner's checks: {len(sub.files)} file(s), {sub.zip_bytes:,} bytes zipped, "
        f"{sub.total_bytes:,} unpacked"
    )
    print(f"sha256 {sub.sha256}: the submission id, which salts your policy seed (any byte change changes it)")
    with zipfile.ZipFile(argv[0]) as zf:
        for warning in agent_warnings(zf.read(AGENT_FILE), [f for f, _n in sub.files]):
            print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

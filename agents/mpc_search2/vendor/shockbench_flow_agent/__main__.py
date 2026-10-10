"""``python -m shockbench_flow_agent <submission dir>``: the scoring container's entry point (``serve_submission``).

Exit status 0 at the end of the input, ``shim.IMPORT_FAILED`` (2) when ``agent.py`` cannot be imported, 64 on a usage
error. With ``SBF_CPU_LOG`` (``shim.CPU_LOG_VAR``) set to a file, each week's CPU seconds are appended to it as JSON
lines ``[week, seconds]`` (``shockbench_flow_agent.play_isolated`` sets it; the scoring container does not).
"""

import os
import sys

from shockbench_flow_agent.shim import CPU_LOG_VAR, serve_submission


USAGE = 64  # EX_USAGE of sysexits.h


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m shockbench_flow_agent <submission directory or agent.py>", file=sys.stderr)
        return USAGE
    return serve_submission(argv[0], cpu_log=os.environ.get(CPU_LOG_VAR) or None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""ShockBench-Flow: the portable benchmark package (design `docs/design.md`; phase 3).

It depends on numpy, scipy (HiGHS), the standard library and the libraries that suit it (Q94 (c): joblib for the
parallel loops, fastjsonschema for the instance file's closed schema, Q95), and never imports hydra (CLAUDE.md).

Its loguru lines (a cache miss of naive's demand model, a kernel fallback) are off unless the application turns them on
with ``loguru.logger.enable("shockbench_flow")``, loguru's convention for libraries (since 0.1.2): the repository's
scripts do, in their Hydra boundary module. The scoring container holds no loguru, so the import is optional.
"""

__version__ = "0.1.2"

try:
    from loguru import logger as _logger
except ImportError:  # the policy container's image has no loguru (the package logs nothing there)
    pass
else:
    _logger.disable(__name__)
    del _logger

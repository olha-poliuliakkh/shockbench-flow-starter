"""scripts/research/tune_compact_optuna.py: the rejection of a trial that falls back, and the files it writes."""

import importlib.util
import json
import sys

import pytest

from tests.conftest import ROOT


optuna = pytest.importorskip("optuna")


@pytest.fixture
def tune():
    from shockbench_flow_agent.shim import unload_agent

    spec = importlib.util.spec_from_file_location(
        "tune_compact_optuna", ROOT / "scripts/research/tune_compact_optuna.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        unload_agent()
        for name in [m for m in sys.modules if m == "compact_hier" or m.startswith("compact_hier.")]:
            del sys.modules[name]


def test_a_trial_whose_agent_falls_back_is_rejected(tune):
    cls, config = tune.load_agent()
    base = config.load(tune.AGENT_DIR, environ={})
    evaluator = tune.Evaluator(cls, [0], quick=True, ref_jobs=1)
    result = evaluator.run(base | {"deadline_small": 0.0})  # no time for the LP: every week falls back
    assert result["rejected"] and "fallback" in result["rejected"]
    ok = evaluator.run(base)
    assert ok["rejected"] is None and len(ok["J"]) == 1


def test_outputs_keep_the_best_accepted_trial(tune, tmp_path):
    cls, config = tune.load_agent()
    base = config.load(tune.AGENT_DIR, environ={})
    full = tune.Evaluator(cls, [0, 1], quick=True, ref_jobs=1).full
    study = optuna.create_study(direction="maximize")
    space = {
        "horizon_small": optuna.distributions.IntDistribution(16, 52, step=4),
        "safety_frac": optuna.distributions.FloatDistribution(0.0, 0.7),
    }

    def add(value, rss, rejected, horizon, anchor=""):
        attrs = {"rss": rss, "rejected": rejected, "episodes": tune.EPISODES, "J": [100, 200], "levels": {}}
        attrs |= {"seconds": 1.0, "anchor": anchor}
        study.add_trial(
            optuna.trial.create_trial(
                params={"horizon_small": horizon, "safety_frac": 0.5},
                distributions=space,
                value=value,
                user_attrs=attrs,
            )
        )

    add(0.70, 0.70, None, 16, anchor="defaults")
    add(0.75, 0.75, None, 24)
    add(0.90 - 1.0, 0.90, "3 week(s) over 1.5 s", 52)  # penalized: never the best, whatever its RSS
    tune.write_outputs(study, base, full, tmp_path)
    best = json.loads((tmp_path / "best_small_params.json").read_text())
    info = json.loads((tmp_path / "best_small_trial.json").read_text())
    assert best["horizon_small"] == 24 and info["trial"] == 1 and info["trials_rejected"] == 1
    assert set(best) == set(config.DEFAULTS)  # a complete preset
    params = config.load(tune.AGENT_DIR, environ={"SBF_PARAMS_FILE": str(tmp_path / "best_small_params.json")})
    assert params["horizon_small"] == 24
    assert (tmp_path / "trials.csv").read_text().count("\n") == 4
    assert "vs_defaults_in_sample" in info  # quick references have no harm levels: the gap records why it is absent

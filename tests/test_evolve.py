"""The evolution loop's pieces: block patching, static checks, paired statistics, the archive, replays."""

import json
import shutil
from types import SimpleNamespace

import pytest

from tests.conftest import ROOT


SEED = ROOT / "agents" / "mine"


def _copy_seed(tmp_path):
    dest = tmp_path / "agent"
    shutil.copytree(SEED, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def test_every_scripted_edit_applies_and_keeps_the_frame():
    from sbf_starter.evolve.evaluate import block_spans, outside_blocks
    from sbf_starter.evolve.mutate import BLOCKS, FAKE_EDITS, apply_proposal

    text = (SEED / "agent.py").read_text()
    assert set(block_spans(text)) == set(BLOCKS)
    for block, source, _params in FAKE_EDITS:
        new = apply_proposal(text, {"blocks": [{"name": block, "source": source}]}, list(BLOCKS))
        assert outside_blocks(new) == outside_blocks(text)
        assert source.strip() in new


def test_a_wrong_block_is_refused():
    from sbf_starter.evolve.mutate import ProposalError, apply_proposal

    text = (SEED / "agent.py").read_text()
    bad = [
        {"name": "settings", "source": "def settings(s, H):\n    return H\n"},  # wrong signature
        {"name": "settings", "source": "def settings(s, mem, default_H):\n    return (\n"},  # does not parse
        {"name": "settings", "source": "# EVOLVE-BLOCK-END settings\ndef settings(s, mem, default_H): return 1\n"},
    ]
    for block in bad:
        with pytest.raises(ProposalError):
            apply_proposal(text, {"blocks": [block]}, ["settings"])
    with pytest.raises(ProposalError):  # a block the directive does not allow
        apply_proposal(
            text,
            {"blocks": [{"name": "objective", "source": "def objective(edits, s, mem):\n    return edits\n"}]},
            ["settings"],
        )


def test_static_check(tmp_path):
    from sbf_starter.evolve.evaluate import static_check
    from sbf_starter.evolve.mutate import apply_proposal

    child = _copy_seed(tmp_path)
    assert static_check(child, SEED) == []
    text = (child / "agent.py").read_text()
    evil = "def settings(s, mem, default_H):\n    import os\n    open('/tmp/x', 'w')\n    return default_H\n"
    (child / "agent.py").write_text(
        apply_proposal(text, {"blocks": [{"name": "settings", "source": evil}]}, ["settings"])
    )
    problems = " ".join(static_check(child, SEED))
    assert "imports ['os']" in problems and "calls open()" in problems
    (child / "agent.py").write_text(text)
    (child / "frame.py").write_text((child / "frame.py").read_text() + "\n# edited\n")
    assert any("frozen" in p for p in static_check(child, SEED))


def test_paired_gap_is_zero_on_identical_costs_and_signed():
    from sbf_starter.evolve.evaluate import paired_gap

    refs = [
        {"episode": i, "stratum": 1 + i % 4, "excluded": None, "J_naive_cents": 1000, "J_oracle_cents": 500}
        for i in range(16)
    ]
    es = SimpleNamespace(references=refs)
    J = [800 - 10 * (i % 3) for i in range(16)]
    g = paired_gap(es, J, J)
    assert g["diff"] == 0 and g["lo"] == 0 and g["hi"] == 0
    better = paired_gap(es, [j - 50 for j in J], J)
    assert better["diff"] == pytest.approx(0.1) and better["lo"] > 0 and better["p_better"] == 1.0


def test_merged_params_clamps_defaults():
    from sbf_starter.evolve.mutate import merged_params

    values, spec = merged_params(
        {"a": 1.0},
        [{"name": "a", "default": 1.0, "low": 0, "high": 2}],
        {"params": [{"name": "b", "default": 9.0, "low": 3.0, "high": 1.0}]},
    )
    assert values == {"a": 1.0, "b": 3.0} and {s["name"] for s in spec} == {"a", "b"}


def test_archive_samples_and_records(tmp_path):
    from sbf_starter.evolve.archive import Archive

    arch = Archive(tmp_path / "a.sqlite", seed=1)
    arch.add(
        id="seed",
        island="all",
        generation=0,
        directive="seed",
        folder=str(SEED),
        stage="accepted",
        rss=0.7,
        calm=0.7,
        crisis=0.7,
        cell="2-2",
        params={},
        spec=[],
    )
    arch.crown("seed", None, None)
    assert arch.directive("forecast") == "announcements"
    parent, insp = arch.sample("forecast")
    assert parent["id"] == "seed" and insp == []
    arch.add(
        id="c0001",
        parent="seed",
        island="forecast",
        generation=1,
        directive="announcements",
        folder=str(SEED),
        stage="accepted",
        rss=0.75,
        calm=0.76,
        crisis=0.7,
        cell=arch.cell(0.76, 0.7),
        params={"x": 1.0},
        spec=[{"name": "x", "default": 1.0, "low": 0.0, "high": 2.0}],
        hypothesis="h",
    )
    assert arch.get("c0001")["params"] == {"x": 1.0} and arch.get("c0001")["cell"] == "4-2"
    assert {c["id"] for c in arch.accepted("forecast")} == {"seed", "c0001"}
    assert arch.attempts("forecast")[0]["outcome"].startswith("accepted")
    assert arch.champion() == "seed"


def test_replay_costs_what_the_score_says():
    from shockbench_flow_agent import EpisodeSet

    from sbf_starter.evolve.trace import replay

    template = ROOT / "agents" / "template"
    es = EpisodeSet.build("tiny", [29], quick=True, n_jobs=1)
    rows = es.play(str(template))
    doc = replay("tiny", 0, 29, str(template), fq_replications=es.fq_replications)
    assert doc["J_cents"] == rows[0]["J_policy_cents"]
    assert set(doc["components"]) >= {"shed", "shortage", "freight"}


def test_prompts_carry_the_economics_and_the_frame_api():
    from sbf_starter.evolve import prompts

    system = prompts.system_prompt(SEED)
    assert "99 %" in system and "10 dollars gained in a level-4" in system
    assert "class Window" in system and "scale_open" in system and "class LPEdits" in system
    parent = {
        "id": "seed",
        "island": "forecast",
        "generation": 0,
        "rss": 0.73,
        "agent_text": (SEED / "agent.py").read_text(),
        "params": {},
        "spec": [],
    }
    user = prompts.user_prompt("announcements", "text", ["belief", "forecast"], parent, "report", [], [])
    assert "def forecast(w, s, mem)" in user and "You may rewrite: belief, forecast." in user
    json.dumps(user)

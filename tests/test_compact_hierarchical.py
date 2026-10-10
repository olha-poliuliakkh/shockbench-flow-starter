"""agents/compact_hierarchical: the maritime controller's bounds, the crisis floors, and a few weeks without a crash.

No score is computed: the weeks played only check that every module runs on Small and Full and returns a valid action.
"""

import copy
import sys

import numpy as np
import pytest

from tests.conftest import ROOT


AGENT = ROOT / "agents" / "compact_hierarchical"


@pytest.fixture
def agent_class():
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cls = load_agent_class(AGENT, "compact_hierarchical_agent")
    try:
        yield cls
    finally:
        unload_agent()
        for name in [m for m in sys.modules if m == "compact_hier" or m.startswith("compact_hier.")]:
            del sys.modules[name]


def _reset(task: str, episode: int = 0):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow_gym import agent_config_from_reset

    from sbf_starter import env_id

    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": episode})
    return env, obs, agent_config_from_reset(env, obs, info)


def _busiest_strait(net):
    """The strait on the most routes."""
    counts = {}
    for pos in net.route_chk_pos:
        for c, _i in pos:
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get)


def _close(obs, net, c, end_week=None):
    o = copy.deepcopy(obs)
    o["graph_now.open"][net.chk_row[c]] = 0.0
    if end_week is not None:
        o["closure_end.chokepoint"][0] = c
        o["closure_end.end_week"][0] = end_week
        o["closure_end.end_week.observed"][0] = 1
    return o


@pytest.mark.parametrize("task, weeks", [("small", 3), ("full", 2)])
def test_weeks_run_and_return_valid_actions(agent_class, task, weeks):
    env, obs, config = _reset(task)
    agent = agent_class(config)
    S = config["spaces"]["action"]["flows"]["shape"][0]
    for _ in range(weeks):
        action = agent.act(obs)
        flows = action["flows"]
        assert flows.shape == (S,) and np.isfinite(flows).all() and (flows >= 0).all()
        obs, _reward, terminated, truncated, _info = env.step(action)
        assert not (terminated or truncated)
    assert all(row["error"] is None for row in agent.log), agent.log
    assert any(row["plan"] for row in agent.log), agent.log


def test_closed_strait_bars_its_routes(agent_class):
    _env, obs, config = _reset("small")
    agent = agent_class(config)
    net, t, H = agent.net, int(obs["week"][0]), agent.net.H
    c = _busiest_strait(net)
    through = [s for s, pos in enumerate(net.route_chk_pos) if any(cc == c for cc, _i in pos)]

    # closed with no announced end: every route through it is barred for the whole window
    bounds = agent.maritime.decide(_close(obs, net, c), t, H)
    assert c in bounds.blocked and all(bounds.x_until[s] == H for s in through)
    assert not bounds.x_until[[s for s in range(net.S) if s not in through]].any()

    # an announced end: barred until the cargo would wait at most reroute_max_wait weeks
    end, wait = t + 10, float(agent.params["reroute_max_wait"])
    bounds = agent.maritime.decide(_close(obs, net, c, end_week=end), t, H)
    tau = np.asarray(obs["graph_now.tau"], dtype=int)
    for s in through:
        lead = min(int(tau[net.route_edges[s][: i + 1]].sum()) for cc, i in net.route_chk_pos[s] if cc == c)
        assert bounds.x_until[s] == min(H, max(0, int(np.ceil(end - t - lead - wait))))

    # the plan sends nothing on a barred route this week
    o = _close(obs, net, c)
    flows = agent.act(o)["flows"]
    assert agent.log[-1]["plan"], agent.log[-1]
    assert np.allclose(flows[through], 0.0)


def test_fallback_moves_barred_flows_to_open_alternatives(agent_class):
    _env, obs, config = _reset("small")
    agent = agent_class(config)
    net, t, H = agent.net, int(obs["week"][0]), agent.net.H
    c = _busiest_strait(net)
    o = _close(obs, net, c)
    bounds = agent.maritime.decide(o, t, H)
    movable = [s for s in np.flatnonzero(bounds.barred_now) if any(bounds.x_until[a] == 0 for a in net.alternatives[s])]
    if not movable:
        pytest.skip("no barred route with an open alternative on this network")
    s = movable[0]
    flows = np.zeros(net.S)
    flows[s] = 1.0
    out = agent.maritime.reroute(flows, bounds, np.ones(net.S), o)
    assert out[s] == 0.0 and out.sum() == pytest.approx(1.0)
    assert all(bounds.x_until[a] == 0 for a in np.flatnonzero(out))


def test_crisis_raises_the_floors_of_cut_off_stores(agent_class):
    _env, obs, config = _reset("small")
    agent = agent_class(config)
    net, t, H = agent.net, int(obs["week"][0]), agent.net.H
    calm = {j: (floor, price) for j, floor, price in agent.safety.floors(obs, t, H, agent.maritime.decide(obs, t, H))}
    c = _busiest_strait(net)
    o = _close(obs, net, c)
    floors = agent.safety.floors(o, t, H, agent.maritime.decide(o, t, H))
    crisis = set(agent.safety.last["crisis_stores"])
    fed = {net.dest_stock[s] for s, pos in enumerate(net.route_chk_pos) if any(cc == c for cc, _i in pos)}
    assert crisis and fed & set(calm) <= crisis
    for j, floor, price in floors:
        if j in crisis:
            assert floor >= calm[j][0] and price >= calm[j][1]
        else:
            assert (floor, price) == calm[j]


def _config(agent_class):
    return sys.modules["compact_hier.config"]


def test_env_overrides_parse_and_take_precedence(agent_class, capsys):
    config = _config(agent_class)
    env = {"SBF_PARAM_REROUTE": "0", "SBF_PARAM_CRISIS": "False", "SBF_PARAM_HORIZON_SMALL": "20"}
    env |= {"SBF_PARAM_TERMINAL_FRAC_SMALL": "0.5", "SBF_PARAM_SAFETY_FRAC": "1e-1", "UNRELATED": "x"}
    params = config.load(AGENT, environ=env)
    assert params["reroute"] == 0 and params["crisis"] is False and params["horizon_small"] == 20
    assert params["terminal_frac_small"] == 0.5 and params["safety_frac"] == pytest.approx(0.1)
    assert params["short_price"] == config.load(AGENT, environ={})["short_price"]  # the rest falls back to params.json
    assert "overrides in force" in capsys.readouterr().err


def test_env_overrides_reject_typos_and_bad_values(agent_class):
    config = _config(agent_class)
    with pytest.raises(KeyError, match="SBF_PARAM_REROUT"):
        config.load(AGENT, environ={"SBF_PARAM_REROUT": "0"})
    for bad in ("abc", "nan", "null", '"x"'):
        with pytest.raises(ValueError):
            config.load(AGENT, environ={"SBF_PARAM_REROUTE": bad})
    with pytest.raises(FileNotFoundError):
        config.load(AGENT, environ={"SBF_PARAMS_FILE": "no_such_preset.json"})


def test_baseline_preset_is_mpc_nobuf_without_the_new_logic(agent_class):
    """params_baseline.json: the rerouting and crisis logic off, mpc_nobuf's safety stocks and short price (whatever
    params.json is tuned to)."""
    config = _config(agent_class)
    base = config.load(AGENT, environ={"SBF_PARAMS_FILE": "params_baseline.json"})
    assert base["reroute"] == 0 and base["crisis"] == 0 and base["safety_frac"] == 0.5 and base["short_price"] == 10
    # a variable beats the preset
    both = config.load(AGENT, environ={"SBF_PARAMS_FILE": "params_baseline.json", "SBF_PARAM_REROUTE": "1"})
    assert both["reroute"] == 1 and both["crisis"] == 0


def test_params_json_lists_every_default(agent_class):
    import json

    config = _config(agent_class)
    for name in ("params.json", "params_baseline.json"):
        assert set(json.loads((AGENT / name).read_text())) == set(config.DEFAULTS), name


def test_baseline_agent_does_not_reroute_or_scale_floors(agent_class, monkeypatch):
    """The agent built under the baseline variables bars nothing, whatever is closed (checked at construction)."""
    monkeypatch.setenv("SBF_PARAM_REROUTE", "0")
    monkeypatch.setenv("SBF_PARAM_CRISIS", "0")
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    unload_agent()
    cls = load_agent_class(AGENT, "compact_hierarchical_agent")
    _env, obs, config = _reset("small")
    agent = cls(config)
    net = agent.net
    c = _busiest_strait(net)
    bounds = agent.maritime.decide(_close(obs, net, c), int(obs["week"][0]), net.H)
    assert not bounds.x_until.any() and not bounds.blocked
    assert not agent.safety.crisis_on
    unload_agent()


def test_each_network_reads_its_own_terminal_fraction(agent_class):
    for task, key in (("small", "terminal_frac_small"), ("full", "terminal_frac_full")):
        _env, _obs, config = _reset(task)
        params = dict(agent_class(config).params)
        params |= {"terminal_frac_small": 0.25, "terminal_frac_full": 0.6}
        agent = agent_class(config, params=params)
        assert agent.net.terminal_frac == params[key]


def test_floor_taper_reaches_zero_in_the_last_week(agent_class):
    taper = sys.modules["compact_hier.compact_lp"].floor_taper
    assert (taper(1, 16, 52, 0) == 1.0).all()  # off
    t, H, T, N = 41, 12, 52, 8  # window weeks 41..52
    got = taper(t, H, T, N)
    weeks = t + np.arange(H)
    assert np.allclose(got, np.clip((T - weeks) / N, 0, 1)) and got[-1] == 0.0 and got[0] == 1.0
    assert (taper(1, 24, 52, 8) == 1.0).all()  # a window far from T keeps the full floor


def test_floor_taper_lets_the_plan_drain_the_stores(agent_class):
    """In the episode's last weeks the plan pays no floor penalty once the taper reaches the week."""
    _env, obs, config = _reset("small")
    base = dict(agent_class(config).params)
    agent = agent_class(config, params=base | {"floor_taper_weeks": 4, "crisis": 0.0})
    o = copy.deepcopy(obs)
    o["week"][0] = 52  # the last week: the floor is 0
    agent.act(o)
    plan = agent.lp.last_plan
    assert plan is not None and plan["sizes"]["safe"] > 0
    a = plan["off"]["safe"]
    assert np.allclose(plan["x"][a : a + plan["sizes"]["safe"]], 0.0)


# ----- production aligned with the simulator (align_chip_production) -------------------------------------------------
def _plan(agent, obs):
    agent.act(copy.deepcopy(obs))
    assert agent.log[-1]["plan"], agent.log[-1]
    return agent.lp.last_plan


def _block(plan, name, h, n):
    a = plan["off"][name] + h * plan["sizes"][name]
    return plan["x"][a : a + n]


def test_package_is_the_simulators_rule(agent_class):
    from shockbench_flow.dynamics.production import package as simulator_package

    package = sys.modules["compact_hier.compact_lp"].package
    rng = np.random.default_rng(0)
    for _ in range(50):
        raw, thr = list(rng.uniform(0, 10, size=rng.integers(1, 4))), float(rng.uniform(0, 20))
        assert np.allclose(package(raw, thr), simulator_package(raw, thr))


def test_alignment_off_builds_the_lp_as_before(agent_class):
    _env, obs, config = _reset("small")
    base = dict(agent_class(config).params)
    off = _plan(agent_class(config, params=base | {"align_chip_production": 0.0}), obs)
    inert = base | {"align_chip_production": 1.0, "align_osat_week1": 0.0, "align_fab_week1": 0.0}
    inert |= {"osat_hold_price": 0.0, "wafer_hold_price": 0.0, "align_fab_second_solve": 0.0}
    same = _plan(agent_class(config, params=inert), obs)
    assert off["sizes"]["rho"] == 0 and off["align"] == {"aligned": False}
    assert off["objective"] == pytest.approx(same["objective"], rel=1e-12) and np.allclose(off["x"], same["x"])


def test_week1_packaging_and_lot_starts_follow_the_simulator(agent_class):
    _env, obs, config = _reset("small")
    agent = agent_class(config, params=dict(agent_class(config).params) | {"align_chip_production": 1.0})
    plan = _plan(agent, obs)
    net, info = agent.net, plan["align"]
    # OSATs: week-1 packaging fixed at the rule; no raw chip left on hand unless the throughput binds
    q0 = _block(plan, "q", 0, len(net.osats))
    I0 = _block(plan, "I", 0, net.J)
    assert info["osat_week1"] and set(info["osat_week1"]) == {m for m, e in enumerate(net.osats) if e[0] >= 0}
    for m, q in info["osat_week1"].items():
        assert q0[m] == pytest.approx(q)
    thr = obs["graph_now.osat.thr_eff"]
    for osat, ms in enumerate(net.osat_ms):
        if sum(q0[m] for m in ms) < thr[osat] * (1 - 1e-9):
            assert all(I0[net.osats[m][0]] <= 1e-6 * max(1.0, q0[m]) for m in ms)
    # fabs: every fab of a grid starts the same share of p-hat; a grid that sheds powers no fab
    p0, rho = _block(plan, "p", 0, len(net.fabs)), _block(plan, "rho", 0, net.NG)
    for f, phat in info["fab_week1_phat"].items():
        g = net.fab_grid[f]
        expected = rho[g] * phat if g >= 0 and net.fab_e[f] > 0 else phat
        assert p0[f] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    sh0 = _block(plan, "sh", 0, net.NG)
    for g, fabs in enumerate(net.grid_fabs):
        energy = sum(e * p0[f] for f, e in fabs)
        tol = 1e-6 * max(1.0, float(obs["graph_now.grid.y_bar"][g]))
        assert not (sh0[g] > tol and energy > tol) or not info.get("week1_second_solve", True)


def test_the_raw_chip_price_packages_instead_of_holding(agent_class):
    _env, obs, config = _reset("small")
    base = dict(agent_class(config).params) | {"align_chip_production": 1.0}
    net = agent_class(config).net
    raw = [j for j, *_ in net.osats if j >= 0]

    def held(params):
        plan = _plan(agent_class(config, params=params), obs)
        return sum(_block(plan, "I", h, net.J)[raw].sum() for h in range(1, plan["H"]))

    assert held(base) <= held(base | {"align_chip_production": 0.0}) + 1e-6


def test_a_failed_aligned_solve_is_repeated_without_the_alignment(agent_class):
    _env, obs, config = _reset("small")
    agent = agent_class(config, params=dict(agent_class(config).params) | {"align_chip_production": 1.0})
    real = agent.lp._solve

    def failing(*args):
        return None if args[-1] is not None else real(*args)

    agent.lp._solve = failing
    agent.act(copy.deepcopy(obs))
    assert agent.log[-1]["plan"] and agent.log[-1]["align"]["retried_without_alignment"]


def test_the_second_solve_can_be_switched_off(agent_class):
    _env, obs, config = _reset("small")
    base = dict(agent_class(config).params) | {"align_chip_production": 1.0}
    plan = _plan(agent_class(config, params=base | {"align_fab_second_solve": 0.0}), obs)
    assert plan["align"]["aligned"] and "week1_base_first_bounds" not in plan["align"]
    assert plan["sizes"]["rho"] > 0  # the week-1 split stays


# ----- base load first in every window week (two_pass_baseload) ----------------------------------------------------
def _violations(plan, net, obs_y_bar):
    """(week, grid) pairs of the plan that both shed base load and power fabs."""
    out = []
    for h in range(plan["H"]):
        sh, p = _block(plan, "sh", h, net.NG), _block(plan, "p", h, len(net.fabs))
        for g, fabs in enumerate(net.grid_fabs):
            tol = 1e-6 * max(1.0, float(obs_y_bar[g]))
            if sh[g] > tol and sum(e * p[f] for f, e in fabs) > tol:
                out.append((h, g))
    return out


def _violation_energy(plan, net, y_bar) -> float:
    total = 0.0
    for h, g in _violations(plan, net, y_bar):
        p = _block(plan, "p", h, len(net.fabs))
        total += min(float(_block(plan, "sh", h, net.NG)[g]), sum(e * p[f] for f, e in net.grid_fabs[g]))
    return total


@pytest.mark.parametrize("align", [1.0, 0.0])
def test_two_pass_serves_base_load_first_where_it_bounds(agent_class, align):
    _env, obs, config = _reset("small")
    base = dict(agent_class(config).params) | {"align_chip_production": align, "align_fab_second_solve": 0.0}
    net, y_bar = agent_class(config).net, obs["graph_now.grid.y_bar"]
    first = _plan(agent_class(config, params=base), obs)
    plan = _plan(agent_class(config, params=base | {"two_pass_baseload": 1.0}), obs)
    info = plan["align"]
    bounded = {tuple(x) for x in info["two_pass_bounded"]}
    assert bounded == set(_violations(first, net, y_bar)) and info["second_solve"]
    assert not bounded & set(_violations(plan, net, y_bar))  # every bounded grid-week now serves base load first
    assert _violation_energy(plan, net, y_bar) < _violation_energy(first, net, y_bar)
    more = _plan(agent_class(config, params=base | {"two_pass_baseload": 1.0, "two_pass_rounds": 3}), obs)
    assert _violation_energy(more, net, y_bar) <= _violation_energy(plan, net, y_bar) + 1e-6
    assert (plan["sizes"]["rho"] > 0) == (align == 1.0)


def test_fixes_a_and_b_are_off_by_default(agent_class):
    """Both measured below the tuned configuration on Small root 202610 (config.py's comments)."""
    import json

    config = _config(agent_class)
    shipped = json.loads((AGENT / "params.json").read_text())
    for key in ("align_chip_production", "two_pass_baseload"):
        assert config.DEFAULTS[key] == 0.0 and shipped[key] == 0.0, key

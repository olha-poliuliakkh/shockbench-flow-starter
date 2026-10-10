"""The observation wrapper outside the environment core (design §5.1, §5.4 "Rules"; Q10, Q68 E13, Q71 X9).

``wrap(core_obs, view, inst)`` turns the core observation of ``dynamics.observe.observe`` (own state, ``graph_now``,
``slot_mask``, ``last_week``; the five feeds None) into theta's observation: it fills ``demand_forecast``
(``forecast.forecast_field``, which needs the instance's demands), ``warning``, ``messages``, ``pending_prohibitions``
and ``closure_end`` per theta from the view's reset-time tables (the last two read row t - 1 of the week tables
``build_view`` made once), nulls the coverage rung's masked ``graph_now`` entries (``coverage.mask_coverage``, on the
instance's graph, in place on its one private copy of the core observation), and in a blackout week nulls every
external feed (``EXTERNAL``) while keeping the own state and ``last_week``. It is pure: a fresh dict that aliases
neither the core observation nor the view. ``wrap_owned`` fills a core observation its caller owns in place instead
(the pair ``coverage.apply_coverage`` and ``coverage.mask_coverage`` make): ``Env._observe(ep) =
wrap_owned(dynamics.observe.observe(ep.inst, ep.marks, ep.state), ep.view, ep.inst)``, on the fresh dict ``observe``
builds, is the only observation path (reset, step, the D9 fallback, restore), so the fallback sees exactly what the
policy sees, the blanked observation under blackout (Q71 X9; threats row 5). The final observation (week T + 1) keeps
the own state and ``last_week`` with every external feed null (§12 "Observation contents").
"""

import copy

from shockbench_flow.information.coverage import mask_coverage
from shockbench_flow.information.forecast import forecast_field
from shockbench_flow.information.view import InformationView
from shockbench_flow.information.warning import warning_field
from shockbench_flow.instance.schema import Instance
from shockbench_flow.marks import osat_throughput


FEEDS = ("demand_forecast", "warning", "messages", "pending_prohibitions", "closure_end")  # filled per theta (§5.1)
EXTERNAL = ("graph_now", "slot_mask", *FEEDS)  # blanked in a blackout week (§5.1; Q68 E13)
OWN_STATE = ("week", "stock", "backlog", "pipeline", "queue_lots", "wip", "last_week")  # kept in every regime


def wrap(core_obs: dict, view: InformationView, inst: Instance) -> dict:
    """Theta's observation of week ``core_obs['week']`` (1): the core observation with the feeds of ``view``.

    ``inst`` is the episode's instance, the one the view was built for (``view.instance_digest``; the file's instance
    at another rung reads the view's, ``Instance.at_digest``, §2.3). In the weeks
    1..T: ``graph_now.osat``'s R and thr_eff from the view's R^osat at the instant t - 1 (Q97), the forecast (48),
    the warning (45), the messages (49), row t - 1 of the pending and closure_end tables, each where theta shows it
    (None elsewhere); then the coverage rung's mask (``coverage.mask_coverage`` at ``theta.h_cov``, on the core's own
    state, in place on the one copy of ``core_obs``), then, in a week of theta's blackout spell, every ``EXTERNAL``
    field None. The final observation (week T + 1) is the core's (its feeds None). Nothing of the result aliases
    ``core_obs`` or the view.

    Raises:
        ValueError: if ``core_obs`` is not a core observation (a feed already set, or a week outside 1..T + 1), or
            ``inst`` is not the view's instance.

    """
    inst = inst.at_digest(view.instance_digest)  # the view's rung: its warm start block (§2.3; M5-O37 (b))
    _check(core_obs, view, inst)
    return _fill(copy.deepcopy(core_obs), view, inst)


def wrap_owned(core_obs: dict, view: InformationView, inst: Instance) -> dict:
    """``wrap`` on a core observation the caller owns: the same observation, filled in place and returned.

    For a fresh dict of ``dynamics.observe.observe``, whose containers alias nothing (``Env._observe``): the fill sets
    top-level fields and ``graph_now``'s lists, so ``core_obs`` becomes theta's observation, never copied.

    Raises:
        ValueError: as ``wrap``, before anything is changed.

    """
    inst = inst.at_digest(view.instance_digest)  # the view's rung: its warm start block (§2.3; M5-O37 (b))
    _check(core_obs, view, inst)
    return _fill(core_obs, view, inst)


def _check(core_obs: dict, view: InformationView, inst: Instance) -> None:
    """Refuse what ``wrap`` refuses: another instance than the view's, or a dict that is not a core observation."""
    if inst.content_digest != view.instance_digest:
        raise ValueError("the instance is not the one the information view was built for (V3)")
    t = core_obs.get("week")
    if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= view.T + 1:
        raise ValueError(f"not a core observation of weeks 1..{view.T + 1}: week {t!r}")
    if any(core_obs.get(name) is not None for name in FEEDS) or set(OWN_STATE + EXTERNAL) - set(core_obs):
        raise ValueError("not a core observation: every field of §9.2 present and its five feeds None")


def _fill(obs: dict, view: InformationView, inst: Instance) -> dict:
    """Fill a checked core observation in place with theta's feeds, mask and blackout (``wrap``); return it."""
    t = obs["week"]
    if t > view.T:  # the final observation: own state and last_week, every external field None (§12)
        return obs
    theta = view.theta
    g = obs["graph_now"]
    if view.osat_now is not None and g is not None:
        R = view.osat_now[t - 1]
        g["osat"]["R"], g["osat"]["thr_eff"] = R.tolist(), osat_throughput(inst, R).tolist()
    if view.forecast is not None:
        obs["demand_forecast"] = forecast_field(inst, view.forecast, t)
    if view.scores is not None:
        obs["warning"] = warning_field(view.units, view.scores, t)
    if view.has_messages:  # ``messages.messages_at(view.messages, t)``: the prefix the view keeps for week t
        n = view.message_counts[t - 1]
        obs["messages"] = {name: list(column[:n]) for name, column in view.message_columns}
    if view.pending is not None:  # fresh lists of ints: the view's week tables are shared
        obs["pending_prohibitions"] = {k: list(v) for k, v in view.pending[t - 1].items()}
    if view.closure_end is not None:
        obs["closure_end"] = {k: list(v) for k, v in view.closure_end[t - 1].items()}
    if theta.h_cov is not None:
        mask_coverage(obs, inst, theta.h_cov)
    if t in view.blackout_weeks:
        for name in EXTERNAL:
            obs[name] = None
    return obs

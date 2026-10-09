"""Calibrate the strait belief of agents/belief48 on a private training root (never the dev split).

    uv run python scripts/research/belief_calibrate.py                    # 128 episodes of root 20261004
    uv run python scripts/research/belief_calibrate.py --episodes=256 --root=20261007

Records, per episode and week, each strait's open fraction and the live message threads (a do-nothing policy plays:
disruptions do not depend on actions), on episodes 0..N-1 of ``root`` (the generator's own mix of harm levels). Fits
on the even episodes and scores on the odd ones:

- closure durations: a run is a maximal stretch of weeks with open < 1; its length L >= 1 (censored at the episode's
  end) is fitted by maximum likelihood to a mixture, P(L >= n) = pi0 (1 - q)^(n-1) + (1 - pi0) (1 - q2)^(n-1);
- closure depth: the mean open fraction of a strait while closed;
- onsets after a live mid_threat thread targeting an open strait: P(onset within j weeks | thread age a), by age
  bucket, and the base rate of onsets with no such thread;
- the expected open fraction E[o_{c,t+h}] these imply, against the persistence forecast (o_{c,t+h} = o_{c,t}):
  mean squared error per horizon h on the odd episodes.

Prints the constants for agents/belief48/belief.py and writes them to outputs/research/belief_calibrate/<date_time>/.
"""

import json

import fire
import numpy as np
from common import run_dir
from joblib import Parallel, delayed


MID = 5  # messages.channel code of mid_threat
CHK = {}  # chokepoint node index -> strait position (messages target nodes; graph_now.open is by position)
AGES = (0, 2, 4, 8)  # age buckets (weeks since announced): [0, 2), [2, 4), [4, 8), [8, inf)
J_MAX = 12  # onset curves up to 12 weeks ahead


def record(root: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401  (registers the environments)

    env = gym.make("ShockBench/Small-v0", entropy=root)
    obs, _info = env.reset(options={"episode": n})
    nothing = {"flows": np.zeros(env.unwrapped.layout.n_slots)}
    opens, threads = [], []
    done = False
    while not done:
        opens.append(np.asarray(obs["graph_now.open"], dtype=float).tolist())
        live = np.nonzero(obs["messages.msg_id.observed"].astype(bool))[0]
        threads.append(
            [
                (
                    int(obs["messages.msg_id"][i]),
                    int(obs["messages.channel"][i]),
                    int(obs["messages.target_kind"][i]),
                    int(obs["messages.target"][i]) if obs["messages.target.observed"][i] else -1,
                    int(obs["messages.announced_week"][i]),
                )
                for i in live
            ]
        )
        obs, _r, term, trunc, _i = env.step(nothing)
        done = term or trunc
    return {"episode": n, "open": opens, "threads": threads}


def runs(o: np.ndarray):
    """Closure runs of one strait's weekly series: (start index, length, censored, mean open while closed)."""
    out, t, T = [], 0, len(o)
    while t < T:
        if o[t] < 1.0:
            s = t
            while t < T and o[t] < 1.0:
                t += 1
            out.append((s, t - s, t == T, float(np.mean(o[s:t]))))
        else:
            t += 1
    return out


def survival(n, pi0, q, q2):
    """P(L >= n) of the duration mixture."""
    n = np.asarray(n, dtype=float)
    return pi0 * (1 - q) ** (n - 1) + (1 - pi0) * (1 - q2) ** (n - 1)


def fit_durations(lengths, censored):
    """Grid maximum likelihood of (pi0, q, q2)."""
    L, C = np.asarray(lengths), np.asarray(censored, dtype=bool)
    best = None
    for pi0 in np.linspace(0.05, 0.95, 91):
        for q in np.linspace(0.05, 0.95, 91):
            for q2 in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
                s_n = survival(L, pi0, q, q2)
                p = np.where(C, s_n, s_n - survival(L + 1, pi0, q, q2))
                ll = float(np.sum(np.log(np.maximum(p, 1e-300))))
                if best is None or ll > best[0]:
                    best = (ll, float(pi0), float(q), float(q2))
    return best


def bucket(age: int) -> int:
    return max(i for i, a in enumerate(AGES) if age >= a)


def onset_table(recs, n_chk: int):
    """P(onset within j weeks | open now, live mid_threat on the strait in age bucket b) and with no such thread."""
    hit = np.zeros((len(AGES) + 1, J_MAX))  # rows: age buckets, then "no thread"
    count = np.zeros(len(AGES) + 1)
    for rec in recs:
        o = np.asarray(rec["open"])
        T = len(o)
        onset = np.zeros_like(o, dtype=bool)
        onset[1:] = (o[1:] < 1.0) & (o[:-1] >= 1.0)
        for t in range(T - 1):
            ages = {}
            for _id, ch, tk, tgt, ann in rec["threads"][t]:
                if ch == MID and tk == 0 and tgt in CHK:
                    ages[CHK[tgt]] = min(ages.get(CHK[tgt], 10**9), t + 1 - ann)
            for c in range(n_chk):
                if o[t, c] < 1.0 or t + J_MAX >= T:
                    continue
                row = bucket(ages[c]) if c in ages else len(AGES)
                count[row] += 1
                first = next((j for j in range(1, J_MAX + 1) if onset[t + j, c]), None)
                if first is not None:
                    hit[row, first - 1 :] += 1
    return hit / np.maximum(count[:, None], 1), count


def expected_open(o_now, closed_for, threat_age, params, h_max):
    """E[o_{c,t+h}] for h = 1..h_max of one strait (the belief agents/belief48 uses), from the fitted constants."""
    pi0, q, q2, depth, F, F0 = (params[k] for k in ("pi0", "q", "q2", "depth", "onset", "onset_none"))
    h = np.arange(1, h_max + 1)
    if o_now < 1.0:  # closed: the probability it is still closed h weeks on, at today's open fraction
        a = max(1, closed_for)
        still = survival(a + h, pi0, q, q2) / survival(a, pi0, q, q2)
        return still * o_now + (1 - still) * 1.0
    cdf = np.asarray(F0 if threat_age is None else F[bucket(threat_age)])
    cdf = np.concatenate([cdf, np.full(max(0, h_max - len(cdf)), cdf[-1])])[:h_max]
    p_on = np.diff(np.concatenate([[0.0], cdf]))  # P(onset at t + i)
    closed = np.array([sum(p_on[i] * survival(hh - i, pi0, q, q2) for i in range(hh)) for hh in h])
    return 1.0 - closed * (1.0 - depth)


def main(root: int = 20261004, episodes: int = 128, h_max: int = 12, workers: int = 3) -> None:
    """Record, fit on the even episodes, score on the odd ones, print and write the constants."""
    from common import world

    CHK.update({node: pos for pos, node in enumerate(world("small", root, 0)[0].chokepoints)})
    recs = Parallel(n_jobs=workers)(delayed(record)(root, n) for n in range(episodes))
    fit, test = recs[0::2], recs[1::2]
    n_chk = len(recs[0]["open"][0])
    allruns = [r for rec in fit for c in range(n_chk) for r in runs(np.asarray(rec["open"])[:, c])]
    ll, pi0, q, q2 = fit_durations([r[1] for r in allruns], [r[2] for r in allruns])
    depth = float(np.mean([r[3] for r in allruns])) if allruns else 0.0
    F, count = onset_table(fit, n_chk)
    params = {
        "pi0": pi0,
        "q": q,
        "q2": q2,
        "depth": depth,
        "onset": F[: len(AGES)].round(4).tolist(),
        "onset_none": F[len(AGES)].round(4).tolist(),
        "age_buckets": list(AGES),
        "fit": {"root": root, "episodes": len(fit), "runs": len(allruns), "censored": int(sum(r[2] for r in allruns))},
        "onset_counts": count.astype(int).tolist(),
    }
    print(f"{len(allruns)} closure runs on {len(fit)} fit episodes ({params['fit']['censored']} censored)")
    print(f"duration mixture: pi0 {pi0:.2f}, q {q:.2f}, q2 {q2:.3f} (log-likelihood {ll:.1f}); mean depth {depth:.2f}")
    for b, (lo, row, cnt) in enumerate(zip(list(AGES) + ["none"], F, count)):
        label = f"mid_threat age >= {lo}" if b < len(AGES) else "no mid_threat"
        print(
            f"  P(onset within 1/2/4/8/12 weeks | {label}, {int(cnt)} strait-weeks): "
            + " / ".join(f"{row[j - 1]:.3f}" for j in (1, 2, 4, 8, 12))
        )
    se_b, se_p, n = np.zeros(h_max), np.zeros(h_max), np.zeros(h_max)
    for rec in test:
        o = np.asarray(rec["open"])
        T = len(o)
        closed_for = np.zeros(n_chk, dtype=int)
        for t in range(T):
            closed_for = np.where(o[t] < 1.0, closed_for + 1, 0)
            ages = {}
            for _id, ch, tk, tgt, ann in rec["threads"][t]:
                if ch == MID and tk == 0 and tgt in CHK:
                    ages[CHK[tgt]] = min(ages.get(CHK[tgt], 10**9), t + 1 - ann)
            for c in range(n_chk):
                e = expected_open(o[t, c], closed_for[c], ages.get(c), params, h_max)
                for h in range(1, h_max + 1):
                    if t + h < T:
                        se_b[h - 1] += (e[h - 1] - o[t + h, c]) ** 2
                        se_p[h - 1] += (o[t, c] - o[t + h, c]) ** 2
                        n[h - 1] += 1
    print("held-out mean squared error of E[o_{t+h}], belief / persistence:")
    print(
        "  " + "  ".join(f"h={h}: {se_b[h - 1] / n[h - 1]:.4f}/{se_p[h - 1] / n[h - 1]:.4f}" for h in (1, 2, 4, 8, 12))
    )
    params["heldout_mse"] = {"belief": (se_b / n).tolist(), "persistence": (se_p / n).tolist()}
    run = run_dir("belief_calibrate")
    (run / "belief_params.json").write_text(json.dumps(params, indent=1))
    print(f"written {run / 'belief_params.json'}")


if __name__ == "__main__":
    fire.Fire(main)

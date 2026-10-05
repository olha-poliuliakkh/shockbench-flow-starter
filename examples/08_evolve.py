"""The AlphaEvolve loop of docs/EVOLVE_DESIGN.md: an LLM rewrites the EVOLVE blocks of the LP agent (agents/mine),
CMA-ES tunes the constants they declare, and paired, stratified, held-out evaluation decides what survives.

    uv sync --extra evolve
    uv run python examples/08_evolve.py --dry_run --max_candidates=4              # offline: scripted edits, no API
    uv run python examples/08_evolve.py --max_candidates=40 --budget_usd=25       # Claude: ANTHROPIC_API_KEY in .env
    uv run python examples/08_evolve.py --resume=outputs/08_evolve/<date_time>    # continue a run

Every candidate passes, in order: S0 static checks (frozen frame, imports, banned calls), S1 a smoke run in a process
holding only the server's packages plus a determinism check, S2 a paired screen against its parent on 8 training
episodes per level, S3 the whole training pool (a private root, stratified like the board). Promising ones get S4,
CMA-ES on their declared constants. A program that beats the champion on training then faces S5: a positive lower
90 % bound of its paired gap on a second private root, and no regression, fallback or CPU excess on Full episodes.
Winners are copied to agents/evo_champion. Nothing uploads. The first run builds the pools (references of every
episode): minutes on Small, about an hour on Full; later runs read them from the cache.
"""

import json
import shutil
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fire

from sbf_starter import ROOT
from sbf_starter.evolve import pools, prompts, trace
from sbf_starter.evolve.archive import DIRECTIVES, FIX, ISLANDS, Archive
from sbf_starter.evolve.evaluate import (
    SMOKE_CPU_S,
    Evaluator,
    descriptors,
    files_sha256,
    paired_gap,
    smoke,
    static_check,
)
from sbf_starter.evolve.mutate import Budget, FakeMutator, LLMMutator, ProposalError, apply_proposal, merged_params


CACHE = ROOT / "outputs" / "evolve_cache"
IGNORE = shutil.ignore_patterns("__pycache__")


def main(
    task: str = "small",
    seed_agent: str = "agents/mine",
    train_root: int = 20261004,
    valid_root: int = 20261005,
    full_root: int = 20261006,
    per_level: int = 32,
    valid_per_level: int = 32,
    full_per_level: int = 8,
    full_gate: bool = True,
    workers: int = 8,
    llm_concurrency: int = 3,
    max_candidates: int = 40,
    budget_usd: float = 25.0,
    effort: str = "high",
    model: str = "claude-opus-5-5",
    dry_run: bool = False,
    tune: bool = True,
    tune_generations: int = 4,
    tune_popsize: int = 8,
    challenge_margin: float = 0.005,
    trace_episodes: int = 2,
    champion_dir: str = "agents/evo_champion",
    resume: str | None = None,
    out: str | None = None,
    seed: int = 0,
) -> None:
    """Run the loop; see the module docstring.

    Args:
        task: the network evolved on (small: the public board's).
        seed_agent: the frozen-frame agent whose blocks evolve.
        train_root, valid_root, full_root: private roots of the training, validation and Full pools (never 0).
        per_level: training episodes per harm level (a multiple of 4: four disjoint tuning subsets).
        valid_per_level, full_per_level: validation and Full episodes per harm level.
        full_gate: require the Full check before a promotion (``--nofull_gate`` skips it and its pool).
        workers: processes that play episodes.
        llm_concurrency: mutator calls in flight.
        max_candidates: children to generate in this call.
        budget_usd: stop once the API spend reaches this.
        effort: the mutator's effort (``xhigh`` is used for ``explore``).
        model: the mutator's model.
        dry_run: scripted mutations instead of Claude (no API key, no spend).
        tune: run CMA-ES on promising children's declared params.
        tune_generations, tune_popsize: the CMA-ES budget per tuned child.
        challenge_margin: the training RSS a child must add over the champion's to be challenged.
        trace_episodes: episodes replayed per report, twice over (worst against the reference, largest gap left).
        champion_dir: where the current champion is copied.
        resume: a run folder to continue.
        out: the run folder (default outputs/08_evolve/<date_time>).
        seed: the loop's random generator (sampling, scripted edits, CMA-ES).

    """
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    if per_level % 4:
        raise ValueError("per_level must be a multiple of 4 (four disjoint tuning subsets)")
    run = Path(resume or out or f"outputs/08_evolve/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    (run / "candidates").mkdir(parents=True, exist_ok=True)
    log_file = (run / "log.txt").open("a")

    def say(text: str) -> None:
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    def event(**row) -> None:
        with (run / "events.jsonl").open("a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), **row}) + "\n")

    seed_dir = Path(seed_agent)
    say(f"run folder {run}; building or loading pools (cached after the first run)")

    # ----- pools: train = screen + rest_a + rest_b + rest_c (disjoint, 1/4 of per_level each), valid, full_valid ----
    q = per_level // 4
    tr = pools.stratified(task, train_root, [per_level] * 4)
    parts = ["screen", "rest_a", "rest_b", "rest_c"]
    lists = {name: pools.take(tr, i * q, q) for i, name in enumerate(parts)}
    sets = {name: pools.episode_set(task, train_root, eps) for name, eps in lists.items()}
    sets["train"] = pools.episode_set(task, train_root, [n for name in parts for n in lists[name]], verbose=False)
    sets["rest"] = pools.episode_set(task, train_root, [n for name in parts[1:] for n in lists[name]], verbose=False)
    sets["det"] = pools.episode_set(task, train_root, [tr[1][0], tr[4][0]], verbose=False)
    va = pools.stratified(task, valid_root, [valid_per_level] * 4)
    sets["valid"] = pools.episode_set(task, valid_root, pools.take(va))
    if full_gate:
        fu = pools.stratified("full", full_root, [full_per_level] * 4)
        sets["full_valid"] = pools.episode_set("full", full_root, pools.take(fu))
    ev = Evaluator(sets, CACHE / "plays", workers, composites={"train": parts, "rest": parts[1:]})
    tracer = trace.Tracer(CACHE / "traces", workers=min(4, workers))
    arch = Archive(run / "archive.sqlite", seed)
    smoke_episode = 80 if task == "small" else 0

    def train_report(cid, folder, res, ref_J, ref_name, ref_folder=None, gap=None, predicted=None) -> str:
        es = sets["train"]
        eps = trace.pick_episodes(es, res.J, ref_J, k=trace_episodes)
        mine = tracer.many(files_sha256(folder), str(folder), task, train_root, eps)
        if ref_folder is None:
            ref = {n: tracer.naive(task, train_root, n) for n in eps}
        else:
            ref = tracer.many(files_sha256(ref_folder), str(ref_folder), task, train_root, eps)
        health = {
            "fallback weeks": res.fallback_weeks,
            "weeks over the CPU budget": res.cpu_weeks,
            "invalid entries": res.invalid_entries,
            "agent stderr lines in replays": sum(m.get("agent_message_count", 0) for m in mine.values()),
        }
        return trace.report(es, task, train_root, res.J, ref_J, ref_name, mine, ref, tracer, health, gap, predicted)

    # ----- bootstrap: the seed --------------------------------------------------------------------------------------
    if arch.get("seed") is None:
        sdir = run / "candidates" / "seed" / "agent"
        if sdir.exists():
            shutil.rmtree(sdir)
        shutil.copytree(seed_dir, sdir, ignore=IGNORE)
        problems = static_check(sdir, seed_dir)
        if problems:
            raise SystemExit(f"the seed fails its own static checks: {problems}")
        say("evaluating the seed on the training pool")
        res = ev.play(sdir, "train")
        naive_J = [r["J_naive_cents"] for r in sets["train"].references]
        calm, crisis = descriptors(sets["train"], res.J)
        arch.add(
            id="seed",
            parent=None,
            island="all",
            generation=0,
            directive="seed",
            folder=str(sdir),
            sha=files_sha256(sdir),
            stage="accepted",
            rss=res.rss,
            calm=calm,
            crisis=crisis,
            cell="2-2",
            params=json.loads((sdir / "params.json").read_text()) if (sdir / "params.json").exists() else {},
            spec=json.loads((sdir / "params_spec.json").read_text()) if (sdir / "params_spec.json").exists() else [],
            usd=0.0,
        )
        arch.update("seed", report=train_report("seed", sdir, res, naive_J, "naive"))
        arch.crown("seed", None, None)
        arch.meta("seed_invalid", res.invalid_entries)
        _publish(sdir, Path(champion_dir))
        say(
            f"seed: training RSS {res.rss:.4f} (calm {calm:.4f}, crisis {crisis:.4f}), fallback weeks "
            f"{res.fallback_weeks}, invalid entries {res.invalid_entries}"
        )
        event(id="seed", stage="accepted", rss=res.rss)

    budget = Budget(budget_usd)
    mutator = FakeMutator(seed) if dry_run else LLMMutator(budget, model=model)
    system = prompts.system_prompt(seed_dir)
    (run / "system_prompt.md").write_text(system)
    islands = list(ISLANDS)
    fixes: deque = deque()
    made = 0

    def request(k: int, pool: ThreadPoolExecutor) -> dict:
        island = islands[k % len(islands)]
        if fixes:
            failed, reason = fixes.popleft()
            parent = arch.get(failed)
            island = parent["island"]
            directive, text, allowed = "fix", f"{FIX[1]}\nError: {reason}", ISLANDS[island]
            insp, ref = [], arch.get(parent["parent"])  # judged against the broken child's own parent
        else:
            parent, insp = arch.sample(island)
            directive = arch.directive(island)
            text, blocks = DIRECTIVES[directive]
            allowed = [b for b in (blocks or ISLANDS[island]) if b in ISLANDS[island]] or ISLANDS[island]
            ref = parent

        def info(c: dict) -> dict:
            text = (Path(c["folder"]) / "agent.py").read_text()
            return {**c, "agent_text": text, "rss": c["rss"] if c["rss"] is not None else float("nan")}

        user = prompts.user_prompt(
            directive,
            text,
            allowed,
            info(parent),
            parent.get("report") or "",
            [info(i) for i in insp],
            arch.attempts(island),
        )
        eff = "xhigh" if directive == "explore" else effort
        if dry_run:
            fut = pool.submit(mutator.propose, system, user, eff, allowed)
        else:
            fut = pool.submit(mutator.propose, system, user, eff)
        return {
            "island": island,
            "parent": parent,
            "ref": ref,
            "directive": directive,
            "allowed": allowed,
            "user": user,
            "future": fut,
        }

    with ThreadPoolExecutor(max_workers=max(1, llm_concurrency)) as pool:
        inflight: deque = deque()
        k = arch.count()
        while made < max_candidates and not budget.exhausted:
            while len(inflight) < max(1, llm_concurrency) and made + len(inflight) < max_candidates:
                inflight.append(request(k, pool))
                k += 1
            job = inflight.popleft()
            try:
                proposal, meta = job["future"].result()
            except Exception as err:  # an API error: logged, the slot is skipped
                proposal, meta = None, {"usd": 0.0, "error": f"{type(err).__name__}: {err}"}
            made += 1
            cid = f"c{arch.count() + 1:04d}"
            try:
                outcome = _evaluate(
                    cid,
                    job,
                    proposal,
                    meta,
                    run,
                    arch,
                    ev,
                    sets,
                    seed_dir,
                    task,
                    smoke_episode,
                    tune,
                    tune_generations,
                    tune_popsize,
                    seed,
                    challenge_margin,
                    full_gate,
                    champion_dir,
                    train_report,
                    fixes,
                    say,
                )
            except Exception as err:  # an evaluator fault on one candidate never stops the run
                import traceback

                (run / "candidates" / cid).mkdir(parents=True, exist_ok=True)
                (run / "candidates" / cid / "error.txt").write_text(traceback.format_exc())
                if arch.get(cid) is None:
                    arch.add(
                        id=cid,
                        parent=job["parent"]["id"],
                        island=job["island"],
                        directive=job["directive"],
                        stage="error",
                        reason=f"{type(err).__name__}: {err}"[:2000],
                    )
                say(f"{cid}: evaluator error {type(err).__name__}: {err} (traceback in candidates/{cid}/error.txt)")
                outcome = {"stage": "error", "reason": f"{type(err).__name__}: {err}"[:300]}
            event(
                id=cid,
                island=job["island"],
                directive=job["directive"],
                parent=job["parent"]["id"],
                usd=round(meta.get("usd", 0.0), 4),
                **outcome,
            )
            if made % 10 == 0 or made == max_candidates:
                _summary(arch, budget, say)
    champ = arch.champion()
    say(f"done: {made} candidates, ${budget.spent:.2f} spent; champion {champ} (copied to {champion_dir})")
    say(f"next: uv run sbf check {champion_dir} --task=small && uv run sbf check {champion_dir} --task=full")


def _publish(folder: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(folder, dest, ignore=IGNORE)


def _evaluate(
    cid,
    job,
    proposal,
    meta,
    run,
    arch,
    ev,
    sets,
    seed_dir,
    task,
    smoke_episode,
    tune,
    tune_generations,
    tune_popsize,
    seed,
    challenge_margin,
    full_gate,
    champion_dir,
    train_report,
    fixes,
    say,
) -> dict:
    """One child through S0..S5; returns the outcome for the event log."""
    from sbf_starter.evolve.tune import tune as cma_tune

    parent = job["parent"]
    cdir = run / "candidates" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "prompt.md").write_text(job["user"])
    (cdir / "response.json").write_text(json.dumps({"proposal": proposal, "meta": meta}, indent=1, default=str))
    row = {
        "id": cid,
        "parent": parent["id"],
        "island": job["island"],
        "generation": (parent["generation"] or 0) + 1,
        "directive": job["directive"],
        "usd": meta.get("usd", 0.0),
        "hypothesis": (proposal or {}).get("hypothesis", ""),
        "predicted": (proposal or {}).get("expected_usd_per_episode", {}),
    }

    def reject(stage: str, reason: str, fixable: bool = False) -> dict:
        arch.add(**row, stage=stage, reason=reason[:2000])
        say(f"{cid} [{job['island']}/{job['directive']}] rejected at {stage}: {reason[:200]}")
        if fixable and job["directive"] != "fix" and "folder" in row:
            fixes.append((cid, reason[:1500]))
        return {"stage": stage, "reason": reason[:300]}

    if proposal is None:
        return reject("no_proposal", meta.get("error") or f"stop reason {meta.get('stop_reason')}")
    pfolder = Path(parent["folder"])  # the code the child starts from
    ref = job["ref"]  # the program it is compared with (the parent, or a broken parent's own parent)
    rfolder = Path(ref["folder"])
    try:
        text = apply_proposal((pfolder / "agent.py").read_text(), proposal, job["allowed"])
    except (ProposalError, SyntaxError) as err:
        return reject("S0", f"the proposal cannot be applied: {err}")
    folder = cdir / "agent"
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(pfolder, folder, ignore=IGNORE)
    (folder / "agent.py").write_text(text)
    values, spec = merged_params(parent["params"] or {}, parent["spec"] or [], proposal)
    (folder / "params.json").write_text(json.dumps(values, sort_keys=True) + "\n")
    row.update(folder=str(folder), params=values, spec=spec)

    problems = static_check(folder, seed_dir)  # S0
    if problems:
        return reject("S0", "; ".join(problems), fixable=True)
    problems, stats = smoke(folder, task, smoke_episode, determinism_es=sets["det"])  # S1
    if problems:
        return reject("S1", "; ".join(problems), fixable=True)
    child_s, parent_s = ev.play(folder, "screen"), ev.play(rfolder, "screen")  # S2
    g2 = paired_gap(sets["screen"], child_s.J, parent_s.J)
    if child_s.fallback_weeks or g2["hi"] < 0:
        return reject(
            "S2",
            f"screen gap to the parent {g2['diff']:+.4f} (90 % {g2['lo']:+.4f} to {g2['hi']:+.4f}), "
            f"fallback weeks {child_s.fallback_weeks}",
        )
    res, pres = ev.play(folder, "train"), ev.play(rfolder, "train")  # S3
    seed_invalid = arch.meta("seed_invalid") or 0
    if res.fallback_weeks or res.invalid_entries > seed_invalid:
        return reject("S3", f"fallback weeks {res.fallback_weeks}, invalid entries {res.invalid_entries}")
    island_best = max((c["rss"] for c in arch.accepted(job["island"])), default=res.rss)
    if tune and spec and res.rss >= island_best - 0.002:  # S4
        say(f"{cid}: tuning {len(spec)} params with CMA-ES")
        values, summary = cma_tune(
            ev,
            folder,
            spec,
            values,
            ["screen", "rest_a", "rest_b", "rest_c"],
            "train",
            run / "tuning" / cid,
            tune_generations,
            tune_popsize,
            seed,
            log=say,
        )
        if summary["tuned"]:
            (folder / "params.json").write_text(json.dumps(values, sort_keys=True) + "\n")
            res = ev.play(folder, "train")
            row.update(params=values)
        kept = "kept" if summary["tuned"] else "dropped"
        say(f"{cid}: tuning {kept} (gap {summary.get('gap', {}).get('diff', 0):+.4f})")
    gap = paired_gap(sets["train"], res.J, pres.J)
    calm, crisis = descriptors(sets["train"], res.J)
    arch.add(
        **row,
        sha=files_sha256(folder),
        stage="accepted",
        rss=res.rss,
        calm=calm,
        crisis=crisis,
        cell=arch.cell(calm, crisis),
    )
    arch.update(
        cid, report=train_report(cid, folder, res, pres.J, f"its parent {ref['id']}", rfolder, gap, row["predicted"])
    )
    say(
        f"{cid} [{job['island']}/{job['directive']}] accepted: training RSS {res.rss:.4f}, gap to parent "
        f"{gap['diff']:+.4f} ({gap['lo']:+.4f} to {gap['hi']:+.4f})"
    )
    out = {"stage": "accepted", "rss": res.rss, "gap": gap["diff"]}

    champ = arch.get(arch.champion())  # S5
    cfolder = Path(champ["folder"])
    if res.rss >= ev.play(cfolder, "train").rss + challenge_margin:
        v, cv = ev.play(folder, "valid"), ev.play(cfolder, "valid")
        gv = paired_gap(sets["valid"], v.J, cv.J)
        verdict = f"valid gap {gv['diff']:+.4f} (lower bound {gv['lo']:+.4f})"
        ok = gv["lo"] > 0 and v.fallback_weeks == 0
        gf = None
        if ok and full_gate:
            f, cf = ev.play(folder, "full_valid"), ev.play(cfolder, "full_valid")
            gf = paired_gap(sets["full_valid"], f.J, cf.J)
            problems, stats = smoke(folder, "full", 0)
            allowance = SMOKE_CPU_S["full"] / (2 if job["island"] == "lean" else 1)
            cpu_ok = stats.get("cpu_p99") is not None and stats["cpu_p99"] <= allowance and not problems
            ok = gf["diff"] >= -0.02 and f.fallback_weeks == 0 and cpu_ok
            verdict += f"; Full gap {gf['diff']:+.4f}, Full CPU p99 {stats.get('cpu_p99')}"
        if ok:
            arch.crown(cid, gv, gf)
            _publish(folder, Path(champion_dir))
            say(f"{cid}: NEW CHAMPION ({verdict}); copied to {champion_dir}")
        else:
            say(f"{cid}: challenge lost ({verdict})")
        out["challenge"] = {"won": ok, "verdict": verdict}
    return out


def _summary(arch, budget, say) -> None:
    lines = []
    for island in ISLANDS:
        acc = arch.accepted(island)
        best = max(acc, key=lambda c: c["rss"])
        lines.append(f"{island} best {best['id']} {best['rss']:.4f} ({len(acc) - 1} accepted)")
    rows = arch.db.execute("SELECT directive, stage FROM candidates WHERE id != 'seed'").fetchall()
    rate = {}
    for r in rows:
        n, a = rate.get(r["directive"], (0, 0))
        rate[r["directive"]] = (n + 1, a + (r["stage"] == "accepted"))
    say(
        "summary: " + "; ".join(lines) + f"; champion {arch.champion()}; ${budget.spent:.2f} over {budget.calls} "
        "calls; acceptance " + ", ".join(f"{d} {a}/{n}" for d, (n, a) in rate.items())
    )


if __name__ == "__main__":
    fire.Fire(main)

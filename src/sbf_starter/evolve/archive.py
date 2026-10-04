"""The program database: candidates in SQLite, islands, a behaviour grid, sampling, directives, the champion.

Each island keeps a MAP-Elites grid over (calm RSS, crisis RSS) measured on the training pool, binned by their gap to
the seed's, so a program that is better in crises keeps its own cell even while a calm-optimized one leads. The
champion changes only through the challenge stage (examples/08_evolve.py).
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from pathlib import Path


ISLANDS = {  # name -> blocks its children may rewrite
    "forecast": ["belief", "forecast", "settings"],
    "objective": ["belief", "objective"],
    "lean": ["belief", "forecast", "objective", "settings"],  # half the CPU allowance on Full (insurance)
    "open": ["belief", "forecast", "objective", "settings"],
}
DIRECTIVES = {  # name -> (text, blocks it may touch, None = the island's)
    "announcements": (
        "The seed planner applies every pending prohibition as certain and ignores announcement "
        "threads. Weigh pending prohibitions and tariff notices by their channel's real share "
        "(1 - decoy share) instead, in the forecast.",
        ["belief", "forecast"],
    ),
    "risk_forecast": (
        "Turn warning scores and threat threads into expected open fractions or capacities for the "
        "straits and edges they name, in proportion to the evidence, decaying as it ages.",
        ["belief", "forecast"],
    ),
    "false_alarms": (
        "Find the precautions that cost money in calm episodes (levels 1 and 2 in the report) and make "
        "them proportional to the evidence or remove them. Keep crisis gains only where the exchange "
        "rate pays for them.",
        None,
    ),
    "ordinary_weeks": (
        "Most of the remaining score is in levels 1 and 2, where few disruptions happen. Improve what "
        "the planner assumes in ordinary weeks: demand beyond the 8-week forecast, the window length, "
        "how long an observed closure lasts, the last weeks of the episode.",
        ["forecast", "settings", "belief"],
    ),
    "objective_shaping": (
        "Where the report shows a grid or market starved, shape the LP's objective: premiums on "
        "exposed routes, stock floors behind a threatened strait, penalty scales.",
        ["belief", "objective"],
    ),
    "crossover": ("Combine the parent with what works in the inspirations into one coherent program.", None),
    "simplify": ("Make the blocks shorter and clearer without changing what they do.", None),
    "explore": ("Try a mechanism none of the island's programs uses yet.", None),
    "params_only": (
        "Expose the parent's hard-coded numbers as declared params with sensible ranges, changing no "
        "logic, so the tuner can search them.",
        None,
    ),
}
FIX = ("fix", "The child below failed its checks with this error. Fix it with the smallest change.")
GAP_EDGES = (-0.02, -0.005, 0.005, 0.02)  # descriptor bins: the gap to the seed's calm / crisis RSS


def _bin(x: float) -> int:
    return sum(x > e for e in GAP_EDGES)


class Archive:
    def __init__(self, path: Path, seed: int = 0):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.rng = random.Random(seed)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, parent TEXT, island TEXT, generation INTEGER, directive TEXT,
                folder TEXT, sha TEXT, stage TEXT, reason TEXT, rss REAL, calm REAL, crisis REAL, cell TEXT,
                hypothesis TEXT, predicted TEXT, params TEXT, spec TEXT, report TEXT, usd REAL, created REAL);
            CREATE TABLE IF NOT EXISTS champions (id TEXT, at REAL, valid_gap TEXT, full_gap TEXT);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self.db.commit()

    # ----- records --------------------------------------------------------------------------------------------------
    def add(self, **row) -> None:
        row.setdefault("created", time.time())
        for k in ("predicted", "params", "spec"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        cols = ", ".join(row)
        self.db.execute(
            f"INSERT OR REPLACE INTO candidates ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values())
        )
        self.db.commit()

    def update(self, cid: str, **row) -> None:
        for k in ("predicted", "params", "spec"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        sets = ", ".join(f"{k} = ?" for k in row)
        self.db.execute(f"UPDATE candidates SET {sets} WHERE id = ?", [*row.values(), cid])
        self.db.commit()

    def get(self, cid: str) -> dict:
        r = self.db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
        return self._row(r) if r else None

    def _row(self, r) -> dict:
        d = dict(r)
        for k in ("predicted", "params", "spec"):
            d[k] = json.loads(d[k]) if d.get(k) else ({} if k == "params" else [])
        return d

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM candidates WHERE id != 'seed'").fetchone()[0]

    def meta(self, key: str, value=None):
        if value is None:
            r = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return json.loads(r[0]) if r else None
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    # ----- the grid -------------------------------------------------------------------------------------------------
    def cell(self, calm: float, crisis: float) -> str:
        seed = self.get("seed")
        return f"{_bin(calm - seed['calm'])}-{_bin(crisis - seed['crisis'])}"

    def accepted(self, island: str | None = None) -> list[dict]:
        q = "SELECT * FROM candidates WHERE stage = 'accepted'" + (" AND (island = ? OR id = 'seed')" if island else "")
        return [self._row(r) for r in self.db.execute(q, (island,) if island else ())]

    def elites(self, island: str) -> dict[str, dict]:
        best: dict[str, dict] = {}
        for c in self.accepted(island):
            if c["cell"] not in best or c["rss"] > best[c["cell"]]["rss"]:
                best[c["cell"]] = c
        return best

    def sample(self, island: str) -> tuple[dict, list[dict]]:
        """(parent, inspirations): the parent from the island's top 5 by RSS (p 0.7) or a random elite cell."""
        pool = sorted(self.accepted(island), key=lambda c: -c["rss"])
        elites = list(self.elites(island).values())
        if self.rng.random() < 0.7 or len(elites) < 2:
            top = pool[:5]
            w = [math.exp((c["rss"] - top[0]["rss"]) / 0.01) for c in top]
            parent = self.rng.choices(top, weights=w)[0]
        else:
            parent = self.rng.choice(elites)
        insp = []
        if pool and pool[0]["id"] != parent["id"]:
            insp.append(pool[0])
        others = [
            c
            for c in self.accepted()
            if c["cell"] != parent["cell"] and c["id"] not in {parent["id"], *(i["id"] for i in insp)}
        ]
        if others:
            insp.append(max(others, key=lambda c: c["rss"]))
        return parent, insp[:2]

    # ----- directives -----------------------------------------------------------------------------------------------
    def directive(self, island: str) -> str:
        """A directive drawn by its acceptance rate on the island (Laplace-smoothed), announcements first."""
        rows = self.db.execute("SELECT directive, stage FROM candidates WHERE island = ?", (island,)).fetchall()
        tried = {d: [0, 0] for d in DIRECTIVES}
        for r in rows:
            if r["directive"] in tried:
                tried[r["directive"]][0] += 1
                tried[r["directive"]][1] += r["stage"] == "accepted"
        if tried["announcements"][0] == 0:
            return "announcements"
        weights = [(a + 1) / (n + 2) for n, a in tried.values()]
        return self.rng.choices(list(tried), weights=weights)[0]

    def attempts(self, island: str, limit: int = 8) -> list[dict]:
        rows = self.db.execute(
            "SELECT hypothesis, directive, stage, reason, rss FROM candidates WHERE island = ? AND id != 'seed' "
            "ORDER BY created DESC LIMIT ?",
            (island, limit),
        ).fetchall()
        return [
            {
                "hypothesis": r["hypothesis"] or "",
                "directive": r["directive"],
                "outcome": f"{r['stage']}"
                + (f" (RSS {r['rss']:.4f})" if r["rss"] is not None else "")
                + (f": {r['reason'][:120]}" if r["reason"] else ""),
            }
            for r in rows
        ]

    # ----- champion -------------------------------------------------------------------------------------------------
    def champion(self) -> str:
        r = self.db.execute("SELECT id FROM champions ORDER BY at DESC LIMIT 1").fetchone()
        return r[0] if r else "seed"

    def crown(self, cid: str, valid_gap: dict | None, full_gap: dict | None) -> None:
        self.db.execute(
            "INSERT INTO champions VALUES (?, ?, ?, ?)", (cid, time.time(), json.dumps(valid_gap), json.dumps(full_gap))
        )
        self.db.commit()

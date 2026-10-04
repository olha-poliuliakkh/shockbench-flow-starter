"""The mutator: Claude through the Messages API, or a scripted stand-in for dry runs (EVOLVE_DESIGN.md 4.4).

``LLMMutator`` sends the cached static system prompt and the candidate's user message, asks for JSON matching
``SCHEMA`` (structured output), streams the answer, prices it from ``usage``, and returns the parsed proposal.
``apply_proposal`` writes it into a copy of the parent's agent.py, block by block, after checking each source
defines its block's function with the block's signature. ``FakeMutator`` proposes small scripted edits with
declared params, so the whole loop can run without an API key or spend.
"""

from __future__ import annotations

import ast
import json
import random
import threading
from pathlib import Path

from sbf_starter.evolve.evaluate import block_spans


BLOCKS = {  # block name -> (function name, argument names)
    "belief": ("update_belief", ["mem", "s"]),
    "forecast": ("forecast", ["w", "s", "mem"]),
    "objective": ("objective", ["edits", "s", "mem"]),
    "settings": ("settings", ["s", "mem", "default_H"]),
}
LEVEL_KEYS = ["level_1", "level_2", "level_3", "level_4"]
SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "expected_usd_per_episode": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in LEVEL_KEYS},
            "required": LEVEL_KEYS,
            "additionalProperties": False,
        },
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(BLOCKS)}, "source": {"type": "string"}},
                "required": ["name", "source"],
                "additionalProperties": False,
            },
        },
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "default": {"type": "number"},
                    "low": {"type": "number"},
                    "high": {"type": "number"},
                },
                "required": ["name", "default", "low", "high"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "expected_usd_per_episode", "blocks", "params"],
    "additionalProperties": False,
}
# Claude Opus 5.5 prices in USD per million tokens; cache writes at the 5-minute TTL cost 1.25x input
PRICES = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}


class ProposalError(ValueError):
    pass


def check_block(name: str, source: str) -> None:
    """The source parses and defines the block's function with the block's arguments."""
    func, args = BLOCKS[name]
    if "EVOLVE-BLOCK" in source:
        raise ProposalError(f"block {name}: the source must not contain EVOLVE markers")
    try:
        tree = ast.parse(source)
    except SyntaxError as err:
        raise ProposalError(f"block {name} does not parse: {err}") from None
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func]
    if not defs:
        raise ProposalError(f"block {name} must define {func}({', '.join(args)})")
    got = [a.arg for a in defs[0].args.args]
    if got != args:
        raise ProposalError(f"block {name}: {func} takes {got}, expected {args}")


def apply_proposal(agent_text: str, proposal: dict, allowed: list[str]) -> str:
    """agent.py with the proposal's blocks in place; ProposalError on a block that is not allowed or not valid."""
    new = agent_text
    for b in proposal.get("blocks", []):
        name, source = b["name"], b["source"].strip("\n")
        if name not in allowed:
            raise ProposalError(f"the directive does not allow rewriting block {name}")
        check_block(name, source)
        a, z = block_spans(new)[name]
        new = new[:a] + source + "\n" + new[z:]
    ast.parse(new)
    return new


def merged_params(values: dict, spec: list[dict], proposal: dict) -> tuple[dict, list[dict]]:
    """The child's params.json values and declared ranges: the parent's, updated by the proposal's declarations."""
    values, by_name = dict(values), {s["name"]: dict(s) for s in spec}
    for p in proposal.get("params", []):
        lo, hi = sorted((float(p["low"]), float(p["high"])))
        default = min(max(float(p["default"]), lo), hi)
        by_name[p["name"]] = {"name": p["name"], "default": default, "low": lo, "high": hi}
        values[p["name"]] = default
    return values, list(by_name.values())


class Budget:
    """Dollars spent on the API, from each response's usage; thread-safe."""

    def __init__(self, limit_usd: float):
        self.limit, self.spent, self.calls = float(limit_usd), 0.0, 0
        self._lock = threading.Lock()

    def add(self, usage) -> float:
        cost = (
            (getattr(usage, "input_tokens", 0) or 0) * PRICES["input"]
            + (getattr(usage, "output_tokens", 0) or 0) * PRICES["output"]
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * PRICES["cache_read"]
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * PRICES["cache_write"]
        ) / 1e6
        with self._lock:
            self.spent += cost
            self.calls += 1
        return cost

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit


class LLMMutator:
    """Claude Opus 5.5 with adaptive thinking (always on), structured output and server-side refusal fallbacks."""

    def __init__(self, budget: Budget, model: str = "claude-opus-5-5", max_tokens: int = 64000):
        import anthropic

        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment or .env
        self.model, self.max_tokens, self.budget = model, max_tokens, budget

    def propose(self, system: str, user: str, effort: str = "high") -> tuple[dict | None, dict]:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = stream.get_final_message()
        u = message.usage
        meta = {
            "usd": self.budget.add(u),
            "stop_reason": message.stop_reason,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0),
            "cache_write": getattr(u, "cache_creation_input_tokens", 0),
            "model": message.model,
        }
        if message.stop_reason in ("refusal", "max_tokens"):
            return None, meta
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            return None, meta
        try:
            return json.loads(text), meta
        except ValueError:
            meta["error"] = "the response is not JSON"
            return None, meta


FAKE_EDITS = [  # (block, source, params): small, valid, plausible edits for dry runs
    (
        "settings",
        'def settings(s, mem, default_H):\n    return default_H + int(round(p("h_offset", 0.0)))\n',
        [("h_offset", 0.0, -6.0, 8.0)],
    ),
    (
        "forecast",
        'def forecast(w, s, mem):\n    w.scale_demand(p("demand_scale", 1.0))\n    return w\n',
        [("demand_scale", 1.0, 0.9, 1.15)],
    ),
    (
        "forecast",
        "def forecast(w, s, mem):\n"
        "    for pos, weeks in enumerate(s.closed_for):\n"
        '        if weeks >= int(p("reopen_after", 99)):\n'
        '            w.set_open(pos, 1.0, start=int(p("reopen_in", 4)))\n'
        "    return w\n",
        [("reopen_after", 6.0, 2.0, 20.0), ("reopen_in", 4.0, 1.0, 12.0)],
    ),
    (
        "objective",
        "def objective(edits, s, mem):\n"
        '    k = p("shed_weight", 1.0)\n'
        "    if k != 1.0:\n"
        '        edits.scale_cost("ysh", k)\n'
        "    return edits\n",
        [("shed_weight", 1.0, 0.8, 1.5)],
    ),
]


class FakeMutator:
    """A scripted stand-in: picks an allowed edit, perturbs its defaults. No API call, no spend."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, system: str, user: str, effort: str = "high", allowed: list[str] | None = None):
        choices = [e for e in FAKE_EDITS if allowed is None or e[0] in allowed] or FAKE_EDITS[:1]
        block, source, params = self.rng.choice(choices)
        declared = []
        for name, default, lo, hi in params:
            d = min(max(default + self.rng.uniform(-0.25, 0.25) * (hi - lo), lo), hi)
            declared.append({"name": name, "default": round(d, 4), "low": lo, "high": hi})
        prop = {
            "hypothesis": f"dry run: scripted edit of {block}",
            "expected_usd_per_episode": {k: 0.0 for k in LEVEL_KEYS},
            "blocks": [{"name": block, "source": source}],
            "params": declared,
        }
        return prop, {"usd": 0.0, "stop_reason": "end_turn", "model": "fake"}


def read_text(path: Path) -> str:
    return Path(path).read_text()
